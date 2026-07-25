"""Default chat model served by the Amazon Bedrock Mantle endpoint.

Requests are passed through unchanged when the model natively supports the
inbound API, and converted otherwise. Which Mantle API a model supports is
seeded per family (``NATIVE_APIS``) and refined at runtime: on a Mantle
"model does not support this API" error the binding is demoted along the
``responses -> chat_completions -> messages`` chain and the result is cached
in memory, so only the first request pays the extra round trip. The OpenAI
routing surface (``/openai/v1`` vs ``/v1``) is learned the same way.
"""

from json import JSONDecodeError, dumps, loads
from typing import TYPE_CHECKING, Any, ClassVar

from sse_starlette import EventSourceResponse, ServerSentEvent

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock_mantle import (
    API_PATHS,
    MantleApiUnsupportedError,
    MantleSurfaceUnsupportedError,
    cache_response_surface,
    decode_mantle_response_id,
    encode_mantle_response_id,
    invoke,
    invoke_stream,
    mantle_request_headers,
    usage_from_chat_completion,
    usage_from_message,
    usage_from_response,
    validate_pruning_extras,
)
from stdapi.config import SETTINGS
from stdapi.models import MANTLE_MODELS, route_and_execute, set_effective_region
from stdapi.models.chat import ChatModelBase
from stdapi.models.chat._mantle import _convert as convert
from stdapi.monitoring import (
    log_error_details,
    log_request_sse_stream_event,
    log_response_params,
)
from stdapi.pricing import Service
from stdapi.region_routing import REGION_ROUTER
from stdapi.types.anthropic_messages import Message
from stdapi.types.openai_chat_completions import ChatCompletion
from stdapi.types.openai_responses import Response
from stdapi.usage import record_bedrock_usage
from stdapi.utils import hide_security_details

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Mapping

    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.aws_bedrock_mantle import MantleApi, SseEvent, Surface
    from stdapi.types.anthropic_messages import MessageCreateParams
    from stdapi.types.openai import ResponseModeration
    from stdapi.types.openai_chat_completions import (
        CompletionCreateParams as ChatCompletionCreateParams,
    )
    from stdapi.types.openai_completions import Completion, CompletionCreateParams
    from stdapi.types.openai_responses import ResponseCreateParams

#: API fallback order when the inbound API is not supported by the model.
_API_FALLBACK_ORDER: tuple[MantleApi, ...] = (
    "responses",
    "chat_completions",
    "messages",
)

#: Learned per-model supported APIs (refined from Mantle 400 errors).
_LEARNED_APIS: dict[str, frozenset[MantleApi]] = {}

#: Learned per-model OpenAI routing surface.
_LEARNED_SURFACE: dict[str, Surface] = {}


class ChatModel(ChatModelBase[Any, Any]):
    """Default Mantle chat model (unknown models assume the Responses API)."""

    #: Mantle transport marker (service-aware dispatch and capabilities).
    IS_MANTLE: ClassVar[bool] = True

    #: Seeded Mantle APIs natively supported by this family.
    NATIVE_APIS: ClassVar[frozenset[MantleApi]] = frozenset({"responses"})

    #: Seeded OpenAI routing surface; ``None`` probes /openai/v1 then /v1.
    SURFACE: ClassVar[Surface | None] = None

    #: Input modalities advertised for this family.
    INPUT_MODALITIES: ClassVar[tuple[str, ...]] = ("TEXT",)

    def _supported_apis(self) -> frozenset[MantleApi]:
        """Return the model's currently-known supported Mantle APIs."""
        return _LEARNED_APIS.get(self._model_id, self.NATIVE_APIS)

    def native_store_supported(self) -> bool:
        """Whether this model stores responses natively on Mantle.

        Returns:
            True when the model serves the Responses API upstream.
        """
        return "responses" in self._supported_apis()

    def _select_api(self, inbound: MantleApi, tried: set[MantleApi]) -> MantleApi:
        """Select the upstream Mantle API to serve an inbound request.

        Args:
            inbound: API matching the inbound route.
            tried: APIs already attempted for this request.

        Returns:
            The upstream API to use next.

        Raises:
            ApiError: When every known candidate API was already tried.
        """
        supported = self._supported_apis()
        if inbound in supported and inbound not in tried:
            return inbound
        for candidate in _API_FALLBACK_ORDER:
            if candidate in supported and candidate not in tried:
                return candidate
        if untried := [api for api in _API_FALLBACK_ORDER if api not in tried]:
            # Optimistic probe: the learned set may be stale or empty.
            return untried[0]
        msg = f"Model '{self._model_id}' cannot serve this request on Bedrock Mantle."
        raise ApiError(msg, status=400)

    def _mantle_regions(self, region: RegionName | None) -> list[RegionName]:
        """Return candidate regions for this model, honoring a region pin.

        Args:
            region: Pinned region (stored-conversation locality), if any.

        Returns:
            Ordered non-empty candidate region list.
        """
        if region:
            return [region]
        if model := MANTLE_MODELS.get(self._model_id):
            return model.regions
        return SETTINGS.aws_bedrock_mantle_regions

    def _api_paths(self, api: MantleApi) -> list[str]:
        """Return the candidate request paths for *api*, surface-aware.

        Args:
            api: Target Mantle API.

        Returns:
            Paths to try in order (known or default surface first; the
            alternate stays as fallback so a stale learned or seeded surface
            self-heals instead of failing until restart).
        """
        if api == "messages":
            return [API_PATHS["messages"]]
        surfaces: tuple[Surface, ...] = ("/openai/v1", "/v1")
        if known := _LEARNED_SURFACE.get(self._model_id) or self.SURFACE:
            surfaces = (known, "/v1" if known == "/openai/v1" else "/openai/v1")
        return [f"{surface}{API_PATHS[api]}" for surface in surfaces]

    async def _invoke_api(
        self,
        api: MantleApi,
        payload: dict[str, Any],
        *,
        stream: bool,
        region: RegionName | None = None,
    ) -> tuple[RegionName, Any]:
        """Invoke *api* with surface probing and region failover.

        Args:
            api: Target Mantle API.
            payload: JSON request body.
            stream: Whether to open a streaming invocation.
            region: Pinned region, if any.

        Returns:
            Tuple of (serving region, parsed JSON response or SSE generator).

        Raises:
            MantleApiUnsupportedError: When the model rejects *api* (caller
                demotes the binding and retries with another API).
            MantleError: On other upstream errors.
        """
        headers = mantle_request_headers(api)
        regions = self._mantle_regions(region)
        # route_and_execute only retries across regions when the region
        # router is enabled and there is more than one candidate; otherwise
        # it calls the first candidate exactly once, so the in-region retry
        # below must cover it instead.
        single_region = len(regions) == 1 or REGION_ROUTER is None
        paths = self._api_paths(api)
        last_surface_error: MantleSurfaceUnsupportedError | None = None
        for path in paths:

            async def call(r: RegionName, path: str = path) -> tuple[RegionName, Any]:
                """Invoke the request on one region, tagging it for usage."""
                set_effective_region(self._model_id, r)
                if stream:
                    return r, await invoke_stream(
                        r, path, payload, single_region=single_region, headers=headers
                    )
                return r, await invoke(
                    r, path, payload, single_region=single_region, headers=headers
                )

            try:
                result = await route_and_execute(self._model_id, regions, call)
            except MantleSurfaceUnsupportedError as error:
                last_surface_error = error
                continue
            if api != "messages":
                surface: Surface = "/openai/v1" if path.startswith("/openai") else "/v1"
                _LEARNED_SURFACE[self._model_id] = surface
            return result
        raise last_surface_error  # type: ignore[misc]

    async def _serve(
        self,
        inbound: MantleApi,
        payload: dict[str, Any],
        *,
        stream: bool,
        region: RegionName | None = None,
    ) -> tuple[MantleApi, RegionName, Any]:
        """Serve an inbound request, converting and learning bindings as needed.

        Args:
            inbound: API matching the inbound route.
            payload: Inbound request body, normalized to *inbound* wire format.
            stream: Whether to open a streaming invocation.
            region: Pinned region, if any.

        Returns:
            Tuple of (upstream API used, serving region, raw result).

        Raises:
            ApiError: When no Mantle API can serve the request.
        """
        tried: set[MantleApi] = set()
        while True:
            api = self._select_api(inbound, tried)
            if (
                inbound == "responses"
                and api != "responses"
                and payload.get("previous_response_id")
            ):
                # Converting away from Responses would silently drop the
                # stored conversation history behind previous_response_id.
                msg = (
                    "previous_response_id cannot be honored: the model does "
                    "not serve the Responses API upstream."
                )
                raise ApiError(msg, status=400)
            if (
                api != "responses"
                and "responses" in tried
                and payload.pop("store", None)
            ):
                # Native storage was lost with the API fallback; agent
                # harnesses set store unconditionally, so it is dropped with
                # a warning instead of failing the request.
                log_error_details(
                    "'store' cannot be honored: the model does not serve the "
                    "Responses API upstream; ignored.",
                    level="warning",
                )
            tried.add(api)
            upstream_payload = convert.convert_payload(inbound, api, payload)
            if stream:
                upstream_payload = convert.enable_stream_usage(api, upstream_payload)
            try:
                serving_region, result = await self._invoke_api(
                    api, upstream_payload, stream=stream, region=region
                )
            except MantleApiUnsupportedError:
                _LEARNED_APIS[self._model_id] = self._supported_apis() - {api}
                continue
            if api not in self._supported_apis():
                _LEARNED_APIS[self._model_id] = self._supported_apis() | {api}
            return api, serving_region, result

    async def _serve_validated(
        self,
        inbound: MantleApi,
        payload: dict[str, Any],
        *,
        region: RegionName | None = None,
    ) -> tuple[MantleApi, RegionName, dict[str, Any]]:
        """Serve a non-streaming request, billing it and converting the result.

        Usage is recorded from the raw upstream-shaped response before any
        wire conversion, so the api-keyed extractor reads the original keys.

        Args:
            inbound: API matching the inbound route.
            payload: Inbound request body, normalized to *inbound* wire format.
            region: Pinned region, if any.

        Returns:
            Tuple of (upstream API used, serving region, response converted
            to the *inbound* shape).
        """
        api, serving_region, raw = await self._serve(
            inbound, payload, stream=False, region=region
        )
        self._record_usage(
            api, raw.get("usage") or {}, serving_region, raw.get("service_tier")
        )
        if api != inbound:
            raw = convert.convert_response(api, inbound, raw)
        return api, serving_region, raw

    async def _stream_serve(
        self,
        inbound: MantleApi,
        payload: dict[str, Any],
        *,
        strip_usage_chunk: bool,
        region: RegionName | None = None,
        id_rewrites: dict[str, str] | None = None,
        wrap: Callable[
            [AsyncGenerator[ServerSentEvent]], AsyncGenerator[ServerSentEvent]
        ]
        | None = None,
    ) -> EventSourceResponse:
        """Serve a streaming request as a logged ``EventSourceResponse``.

        Args:
            inbound: API matching the inbound route.
            payload: Inbound request body, normalized to *inbound* wire format.
            strip_usage_chunk: Drop the final usage chunk that was forced
                upstream when the client did not request it (chat completions).
            region: Pinned region, if any.
            id_rewrites: Mutable native ID -> public ID substitutions applied
                to the relayed events (stored-response region tagging).
            wrap: Optional converter applied to the relayed event stream.

        Returns:
            Streaming response relaying the upstream events.
        """
        api, serving_region, events = await self._serve(
            inbound, payload, stream=True, region=region
        )
        relayed = self._relay_stream(
            api,
            inbound,
            events,
            serving_region,
            strip_usage_chunk=strip_usage_chunk,
            id_rewrites=id_rewrites,
        )
        if wrap is not None:
            relayed = wrap(relayed)
        return EventSourceResponse(log_request_sse_stream_event(relayed))

    def _record_usage(
        self,
        api: MantleApi,
        usage: Mapping[str, Any],
        region: RegionName,
        tier: str | None,
    ) -> None:
        """Record billed usage from an upstream *api*-shaped usage block.

        Args:
            api: Upstream Mantle API that produced the usage.
            usage: Raw usage object from the response.
            region: Region that served the call.
            tier: Service tier reported by the response, if any.
        """
        extract = {
            "chat_completions": usage_from_chat_completion,
            "responses": usage_from_response,
            "messages": usage_from_message,
        }[api]
        record_bedrock_usage(
            self._model_id,
            service=Service.BEDROCK_MANTLE,
            region=region,
            tier=tier,
            routing="",
            **extract(usage),
        )

    async def _relay_stream(
        self,
        api: MantleApi,
        inbound: MantleApi,
        events: AsyncGenerator[SseEvent],
        region: RegionName,
        *,
        strip_usage_chunk: bool,
        id_rewrites: dict[str, str] | None = None,
    ) -> AsyncGenerator[ServerSentEvent]:
        """Relay upstream SSE events to the client, recording billed usage.

        When *api* differs from *inbound* the events are converted to the
        inbound wire format first. Public/native ID rewrites are applied to
        the raw event payloads (stored-response region tagging).

        Args:
            api: Upstream Mantle API serving the stream.
            inbound: API matching the inbound route.
            events: Upstream SSE event generator.
            region: Region serving the stream.
            strip_usage_chunk: Drop the final usage chunk that was forced
                upstream when the client did not request it (chat completions).
            id_rewrites: Mutable native ID -> public ID substitutions, may be
                pre-seeded (chained ``previous_response_id``) and is extended
                in-place as stored-response IDs appear in the stream.

        Yields:
            Server-sent events in the inbound wire format.
        """
        rewrites = id_rewrites if id_rewrites is not None else {}
        observed = self._observe_stream(api, events, region, rewrites)
        if api != inbound:
            observed = convert.convert_stream(api, inbound, observed)
        async for event, data in observed:
            if (
                strip_usage_chunk
                and inbound == "chat_completions"
                and _is_usage_chunk(data)
            ):
                continue
            if event in ("error", "response.failed") or (
                event is None and '"error"' in data
            ):
                data = _scrub_error_event(data)
            if rewrites:
                for native, public in rewrites.items():
                    data = data.replace(native, public)
            yield ServerSentEvent(data=data, event=event)
        if inbound == "chat_completions":
            yield ServerSentEvent(data="[DONE]", event=None)

    async def _observe_stream(
        self,
        api: MantleApi,
        events: AsyncGenerator[SseEvent],
        region: RegionName,
        rewrites: dict[str, str],
    ) -> AsyncGenerator[SseEvent]:
        """Observe the raw upstream stream: tap stored IDs and record usage.

        Runs before any wire conversion so IDs, usage and tier are read in
        the upstream *api* shape; each event is parsed at most once. Malformed
        frames are relayed unmodified, without observation.

        Args:
            api: Upstream Mantle API serving the stream.
            events: Upstream SSE event generator.
            region: Region serving the stream.
            rewrites: Mutable native ID -> public ID mapping, updated in-place.

        Yields:
            Unmodified upstream events.
        """
        seen_id = False
        input_usage: dict[str, Any] = {}
        last_usage: tuple[Mapping[str, Any], str | None] | None = None
        try:
            async for event, data in events:
                parsed = (
                    _try_loads(data)
                    if _needs_parse(api, data, seen_id=seen_id)
                    else None
                )
                if parsed is not None:
                    if api == "responses" and not seen_id:
                        seen_id = self._tap_response_id(parsed, region, rewrites)
                    if _may_carry_usage(data):
                        last_usage = (
                            self._tap_usage(api, event, parsed, input_usage, region)
                            or last_usage
                        )
                yield event, data
        finally:
            if last_usage is not None:
                usage, tier = last_usage
                self._record_usage("chat_completions", usage, region, tier)

    def _tap_response_id(
        self, parsed: dict[str, Any], region: RegionName, rewrites: dict[str, str]
    ) -> bool:
        """Record a native Responses id and cache its routing surface, if present.

        Args:
            parsed: Parsed SSE data payload.
            region: Region serving the stream.
            rewrites: Mutable native ID -> public ID mapping, updated in-place.

        Returns:
            True when a native response id was found.
        """
        if not (native := (parsed.get("response") or parsed).get("id")):
            return False
        rewrites.setdefault(native, encode_mantle_response_id(region, native))
        if surface := _LEARNED_SURFACE.get(self._model_id):
            cache_response_surface(native, surface)
        return True

    def _tap_usage(
        self,
        api: MantleApi,
        event: str | None,
        parsed: dict[str, Any],
        input_usage: dict[str, Any],
        region: RegionName,
    ) -> tuple[Mapping[str, Any], str | None] | None:
        """Record usage from a parsed stream event, deferring chat_completions billing.

        Args:
            api: Upstream Mantle API shape of the stream.
            event: SSE event name, if any.
            parsed: Parsed SSE data payload.
            input_usage: Scratch storage carrying Anthropic input usage.
            region: Region serving the stream.

        Returns:
            The ``(usage, tier)`` to bill once a chat_completions stream ends,
            or ``None`` when usage was recorded immediately or is absent.
        """
        if (usage := _event_usage(api, event, parsed, input_usage)) is None:
            return None
        if api != "chat_completions":
            self._record_usage(api, usage, region, _event_tier(parsed))
            return None
        # Chunks may carry cumulative usage: bill only the last value, once
        # the stream ends.
        return usage, _event_tier(parsed)

    async def create_completion(
        self,
        request: ChatCompletionCreateParams,
        completion_id: str,  # noqa: ARG002 (passthrough keeps upstream IDs)
        created: int,  # noqa: ARG002
    ) -> ChatCompletion | EventSourceResponse:
        """Handle a chat completion request via the OpenAI route.

        Args:
            request: OpenAI-format completion request.
            completion_id: Unused; passthrough keeps the upstream identifier.
            created: Unused; passthrough keeps the upstream timestamp.

        Returns:
            Completed response or streaming ``EventSourceResponse``.
        """
        payload = await convert.chat_completions_payload(request, self._model_id)
        if request.stream:
            return await self._stream_serve(
                "chat_completions",
                payload,
                strip_usage_chunk=not _include_usage(request),
            )
        _, _, raw = await self._serve_validated("chat_completions", payload)
        return log_response_params(validate_pruning_extras(ChatCompletion, raw))

    async def create_text_completion(
        self,
        request: CompletionCreateParams,
        completion_id: str,
        created: int,  # noqa: ARG002 (passthrough keeps upstream timestamps)
    ) -> Completion | EventSourceResponse:
        """Handle a legacy completion request (OpenAI ``POST /v1/completions``).

        Mantle has no legacy completions endpoint: the request is always
        converted (prompt to chat messages) before serving.

        Args:
            request: Completion creation request following the OpenAI spec.
            completion_id: Stable identifier for the completion.
            created: Unix timestamp (seconds) of the request.

        Returns:
            ``Completion`` or streaming ``EventSourceResponse``.
        """
        payload = await convert.text_completion_as_chat_payload(request, self._model_id)
        if request.stream:
            return await self._stream_serve(
                "chat_completions",
                payload,
                strip_usage_chunk=not _include_usage(request),
                wrap=lambda events: convert.chat_stream_as_text_completion(
                    events, completion_id
                ),
            )
        _, _, raw = await self._serve_validated("chat_completions", payload)
        return log_response_params(
            convert.chat_response_as_text_completion(raw, completion_id)
        )

    async def create_message(
        self,
        request: MessageCreateParams,
        message_id: str,  # noqa: ARG002 (passthrough keeps upstream IDs)
    ) -> Message | EventSourceResponse:
        """Handle a message request via the Anthropic Messages route.

        Args:
            request: Anthropic-format message creation request.
            message_id: Stable identifier for the message.

        Returns:
            ``Message`` or streaming ``EventSourceResponse``.
        """
        payload = await convert.messages_payload(request, self._model_id)
        if request.stream:
            return await self._stream_serve(
                "messages", payload, strip_usage_chunk=False
            )
        _, _, raw = await self._serve_validated("messages", payload)
        return log_response_params(validate_pruning_extras(Message, raw))

    async def create_response(
        self,
        request: ResponseCreateParams,
        response_id: str,
        created_at: float,  # noqa: ARG002 (passthrough keeps upstream timestamps)
        moderation_builder: Callable[[], ResponseModeration | None] | None = None,
    ) -> Response | EventSourceResponse:
        """Handle a response request via the OpenAI Responses route.

        Responses served natively use Mantle's storage: their IDs are tagged
        with the serving region so chained requests stay region-local, and
        are therefore not reusable against the Mantle API directly. Converted
        (non-native) results carry *response_id* instead, so the route's
        local response store works for models without native storage.

        Args:
            request: Responses API creation request.
            response_id: Unique identifier for the response.
            created_at: Unix timestamp of request creation.
            moderation_builder: Optional callable building the response
                ``moderation`` field.

        Returns:
            Completed response or streaming ``EventSourceResponse``.
        """
        if (previous_id := request.previous_response_id) and (
            decode_mantle_response_id(previous_id) is None
        ):
            # A local-store ID: the route already merged its conversation
            # inline and restored the ID only for response echoing; only
            # Mantle-tagged IDs can chain upstream.
            request = request.model_copy(update={"previous_response_id": None})
        payload, pinned_region = await convert.responses_payload(
            request, self._model_id
        )
        rewrites: dict[str, str] = {}
        if (native_previous := payload.get("previous_response_id")) and (
            public_previous := request.previous_response_id
        ):
            # Chained requests echo the previous ID: map it back to public.
            rewrites[native_previous] = public_previous
        if request.stream:
            return await self._stream_serve(
                "responses",
                payload,
                strip_usage_chunk=False,
                region=pinned_region,
                id_rewrites=rewrites,
            )
        api, region, raw = await self._serve_validated(
            "responses", payload, region=pinned_region
        )
        if not request.background and raw.get("status") == "failed":
            # A synchronous request must not swallow an upstream failure into a
            # 200 with empty output; background requests keep the failed
            # terminal state so the client can poll it.
            raise _failed_response_error(raw)
        if request.previous_response_id and raw.get("previous_response_id"):
            raw["previous_response_id"] = request.previous_response_id
        if api == "responses" and (native_id := raw.get("id")):
            # Responses-served results are stored upstream: tag the region.
            raw["id"] = encode_mantle_response_id(region, native_id)
            if surface := _LEARNED_SURFACE.get(self._model_id):
                cache_response_surface(native_id, surface)
        else:
            # Converted results carry the server-assigned ID so the local
            # response store (used when native storage is unavailable) can
            # serve GET/DELETE/previous_response_id on the returned ID.
            raw["id"] = response_id
        response = validate_pruning_extras(Response, raw)
        if moderation_builder is not None:
            response.moderation = moderation_builder()
        return log_response_params(response)


def _failed_response_error(raw: dict[str, Any]) -> ApiError:
    """Build the error raised for a synchronous terminal ``failed`` Response.

    A ``status="failed"`` Response carries the upstream failure in its ``error``
    field and no usable output. The upstream message is scrubbed of security
    details and surfaced as a 502 so a synchronous request reports the failure
    instead of returning an empty 200 body.

    Args:
        raw: Upstream Responses-shaped result with ``status == "failed"``.

    Returns:
        The ``ApiError`` to raise for the failed response.
    """
    error = raw.get("error")
    message = error.get("message") if isinstance(error, dict) else error
    if not isinstance(message, str) or not message:
        message = "The upstream model response failed."
    return ApiError(hide_security_details(502, message), status=502)


def _scrub_error_event(data: str) -> str:
    """Scrub security details from a relayed upstream error payload.

    Passthrough error events reach the client verbatim: the upstream message
    is rewritten through :func:`hide_security_details` while the payload
    shape is preserved.

    Args:
        data: Raw SSE error event data payload.

    Returns:
        The payload with its error message scrubbed, or unchanged when it
        carries no top-level ``error``.
    """
    try:
        payload = loads(data)
    except JSONDecodeError:
        return data
    if not isinstance(payload, dict):
        return data
    error = payload.get("error")
    # response.failed events nest the error under "response"; Responses-shaped
    # error events carry the message at the payload's top level.
    if error is None and isinstance(response := payload.get("response"), dict):
        error = response.get("error")
    if error is None and isinstance(payload.get("message"), str):
        error = payload
    if isinstance(error, str):
        payload["error"] = hide_security_details(502, error)
    elif isinstance(error, dict) and error.get("message") is not None:
        message = error["message"]
        # Structured message content is serialized before scrubbing.
        error["message"] = hide_security_details(
            502, message if isinstance(message, str) else dumps(message)
        )
    else:
        return data
    return dumps(payload)


def _include_usage(
    request: ChatCompletionCreateParams | CompletionCreateParams,
) -> bool:
    """Whether the streaming request opted into the final usage chunk.

    Args:
        request: OpenAI-format (chat) completion request.

    Returns:
        True when ``stream_options.include_usage`` is set.
    """
    return (
        request.stream_options is not None
        and request.stream_options.include_usage is True
    )


def _may_carry_usage(data: str) -> bool:
    """Whether a raw SSE data payload may carry a usage block.

    Fast pre-filter avoiding JSON parsing: OpenAI delta chunks carrying a
    literal ``"usage":null`` field are skipped without parsing.

    Args:
        data: Raw SSE data payload.

    Returns:
        True when the payload may hold a non-null ``usage`` object.
    """
    return (
        '"usage"' in data and '"usage":null' not in data and '"usage": null' not in data
    )


def _needs_parse(api: MantleApi, data: str, *, seen_id: bool) -> bool:
    """Whether an observed stream event's data payload needs parsing.

    Args:
        api: Upstream Mantle API shape of the stream.
        data: Raw SSE data payload.
        seen_id: Whether the native response id was already captured.

    Returns:
        True when the event may carry an unseen Responses id or a usage block.
    """
    return (api == "responses" and not seen_id and '"id"' in data) or _may_carry_usage(
        data
    )


def _try_loads(data: str) -> dict[str, Any] | None:
    """Parse a raw SSE data payload, tolerating malformed relayed frames.

    Args:
        data: Raw SSE data payload.

    Returns:
        The parsed JSON object, or ``None`` when parsing fails or the payload
        is not a JSON object.
    """
    try:
        parsed = loads(data)
    except JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_usage_chunk(data: str) -> bool:
    """Whether an inbound-shaped chunk is the final Chat Completions usage chunk.

    A chunk that carries usage alongside content choices is not strippable:
    dropping it would lose the content and finish reason it also delivers.

    Args:
        data: Raw Chat Completions chunk SSE data payload.

    Returns:
        True when the parsed chunk carries a truthy ``usage`` block and no
        choices; ``False`` for a malformed payload.
    """
    if not _may_carry_usage(data):
        return False
    chunk = _try_loads(data)
    return chunk is not None and bool(chunk.get("usage")) and not chunk.get("choices")


def _event_usage(
    api: MantleApi,
    event: str | None,
    parsed: dict[str, Any],
    input_usage: dict[str, Any],
) -> Mapping[str, Any] | None:
    """Extract a complete usage block from a stream event, if it carries one.

    Args:
        api: Upstream Mantle API shape of the stream.
        event: SSE event name, if any.
        parsed: Parsed SSE data payload.
        input_usage: Scratch storage carrying Anthropic ``message_start``
            input usage until the final ``message_delta`` event.

    Returns:
        A usage mapping in the *api* wire shape, or ``None``.
    """
    match api:
        case "chat_completions":
            return parsed.get("usage") or None
        case "responses" if event in (
            "response.completed",
            "response.incomplete",
            "response.failed",
        ):
            return (parsed.get("response") or {}).get("usage") or None
        case "messages" if event == "message_start":
            input_usage.update((parsed.get("message") or {}).get("usage") or {})
        case "messages" if event == "message_delta":
            return {**input_usage, **(parsed.get("usage") or {})}
        case _:
            pass
    return None


def _event_tier(parsed: dict[str, Any]) -> str | None:
    """Extract the service tier from a parsed usage-bearing event payload.

    Args:
        parsed: Parsed SSE data payload.

    Returns:
        Service tier string, or ``None``.
    """
    tier = parsed.get("service_tier") or (parsed.get("response") or {}).get(
        "service_tier"
    )
    return tier or None
