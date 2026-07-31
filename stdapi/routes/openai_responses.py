"""OpenAI Responses API endpoint implementation.

This module implements the OpenAI-compatible /v1/responses endpoints, providing
AWS Bedrock Converse integration while maintaining full API compatibility.

The module provides:
    - POST /v1/responses — create a model response
    - POST /v1/responses/input_tokens — count input tokens without generating a response
    - POST /v1/responses/compact — compact a conversation into a reusable summary item
    - GET /v1/responses/{response_id} — retrieve a stored model response
    - POST /v1/responses/{response_id}/cancel — cancel a background model response
    - DELETE /v1/responses/{response_id} — delete a stored model response
    - GET /v1/responses/{response_id}/input_items — list a stored model response's input items
"""

from functools import partial
from re import fullmatch
from typing import TYPE_CHECKING, Annotated, Any, Literal, Never, cast, get_args

from fastapi import APIRouter, Depends, Path, Query
from pydantic import TypeAdapter, ValidationError

from stdapi.api_errors import ApiError
from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.aws_bedrock import BEDROCK_PROMPT_VAR
from stdapi.aws_bedrock_mantle import (
    MantleError,
    cache_response_surface,
    cached_response_surface,
    decode_mantle_response_id,
    encode_mantle_response_id,
    mantle_request_headers,
    request_json,
    validate_pruning_extras,
)
from stdapi.config import SETTINGS
from stdapi.models import resolve_bedrock_prompt, validate_model
from stdapi.models.capabilities import Capability, register_route_capability
from stdapi.models.chat import get_chat_model, serves_via_mantle
from stdapi.models.chat._adapters._openai_responses import (
    count_input_tokens_via_bedrock,
    encode_compaction_content,
)
from stdapi.monitoring import (
    REQUEST_ID,
    REQUEST_TIME,
    log_error_details,
    log_request_params,
    log_response_params,
)
from stdapi.responses_store import (
    RESPONSE_ID_PATTERN,
    delete_stored_response,
    discard_stored_response_session,
    load_stored_response,
    save_stored_response,
    try_create_stored_response_session,
)
from stdapi.routes._moderation import (
    apply_request_moderation,
    build_response_moderation,
)
from stdapi.types.openai_responses import (
    CompactedResponse,
    CompactionUserMessage,
    CompactParams,
    EasyInputMessage,
    InputTokenCountParams,
    InputTokenCountResponse,
    InputTokensDetails,
    OutputTokensDetails,
    Response,
    ResponseCompactionItem,
    ResponseCreateParams,
    ResponseDeleted,
    ResponseIncludable,
    ResponseInputItem,
    ResponseItem,
    ResponseItemList,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponsePrompt,
    ResponseUsage,
)
from stdapi.utils import hide_security_details, validation_error_handler

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sse_starlette import EventSourceResponse
    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.aws_bedrock_mantle import Surface
    from stdapi.models.chat import ChatModelBase
    from stdapi.models.chat._default import ChatModel

register_route_capability(
    "openai_response", f"{SETTINGS.openai_routes_prefix}/v1/responses", "TEXT", "TEXT"
)

register_route_capability(
    "openai_response_input_tokens",
    f"{SETTINGS.openai_routes_prefix}/v1/responses/input_tokens",
    "TEXT",
    "TEXT",
    required_capability=Capability.COUNT_TOKENS,
)

register_route_capability(
    "openai_response_compact",
    f"{SETTINGS.openai_routes_prefix}/v1/responses/compact",
    "TEXT",
    "TEXT",
)

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1/responses", tags=["Chat", TAG_OPENAI]
)

#: Directive appended to the conversation to produce the compaction summary.
_COMPACTION_PROMPT = (
    "Summarize the conversation above in detail, preserving every fact, "
    "decision, constraint, open task, and tool result needed to continue it. "
    "Reply with the summary only."
)

#: Accepts native stored IDs (``resp-``) and region-tagged Mantle IDs (``resp_``).
_RESPONSE_ID_PATTERN = r"^resp[-_][A-Za-z0-9-]+$"

#: Reusable path annotation for the ``response_id`` path parameter.
_ResponseId = Annotated[
    str,
    Path(description="The ID of the stored response.", pattern=_RESPONSE_ID_PATTERN),
]


def _decode_mantle_id(response_id: str) -> tuple[RegionName, str] | None:
    """Decode a region-tagged Mantle response ID, gated on Mantle support.

    Args:
        response_id: Public response identifier.

    Returns:
        Tuple of (region, native Mantle response ID), or ``None`` when Mantle
        support is disabled or the ID is not a region-tagged Mantle ID.
    """
    if not SETTINGS.aws_bedrock_mantle_enabled:
        return None
    return decode_mantle_response_id(response_id)


def _require_local_response_id(response_id: str) -> None:
    """Reject undecodable Mantle-form (``resp_``) IDs before the local store.

    Local store IDs use ``resp-``; a ``resp_`` ID that failed Mantle decoding
    can never exist locally and would be mangled into an invalid Bedrock
    session identifier by the local-store lookup.

    Args:
        response_id: Public response identifier that failed Mantle decoding.

    Raises:
        ApiError: 404 when the ID has the Mantle ``resp_`` form.
    """
    if response_id.startswith("resp_"):
        msg = f"Response '{response_id}' not found."
        raise ApiError(msg, status=404)


def _previous_response_not_found(previous_response_id: str) -> Never:
    """Raise the upstream-worded 404 for a missing or invalid previous_response_id.

    Args:
        previous_response_id: The ``previous_response_id`` that is invalid or
            does not resolve to a stored response.

    Raises:
        ApiError: Always, with status 404, matching OpenAI's wording and the
            ``previous_response_id`` param.
    """
    msg = f"Previous response with id '{previous_response_id}' not found."
    error = ApiError(msg, status=404)
    error.param = "previous_response_id"
    raise error


def _failed_response_error(response: Response) -> Never:
    """Raise the 502 for a synchronous terminal ``failed`` Response.

    A ``status="failed"`` Response carries the upstream failure in its ``error``
    field and no usable output; a synchronous request must report the failure
    instead of returning an empty 200 body, matching the Mantle-served path.

    Args:
        response: The Response object with ``status == "failed"``.

    Raises:
        ApiError: Always, with status 502.
    """
    message = response.error.message if response.error else None
    if not message:
        message = "The model failed to generate a valid response."
    raise ApiError(hide_security_details(502, message), status=502)


async def _apply_previous_response(
    request: ResponseCreateParams, *, native_supported: bool
) -> ResponseCreateParams:
    """Resolve ``previous_response_id`` against the target model's storage.

    Local store IDs get their stored conversation merged into the request;
    Mantle-tagged IDs are kept for native upstream chaining, which requires a
    model with native store support.

    Args:
        request: The incoming request.
        native_supported: Whether the target model chains conversations
            natively (Bedrock Mantle Responses API).

    Returns:
        The request, rebuilt with merged history for local store IDs.

    Raises:
        ApiError: 404 when the stored response does not exist, is not a
            well-formed local ID, or a Mantle stored conversation is
            continued with a non-Mantle-native model.
    """
    previous_response_id = request.previous_response_id
    if not previous_response_id:
        return request
    if _decode_mantle_id(previous_response_id) is None:
        _require_local_response_id(previous_response_id)
        if not fullmatch(RESPONSE_ID_PATTERN, previous_response_id):
            # Rejects e.g. a session ARN smuggled past the `resp_` prefix
            # check, which the store would otherwise pass on to AWS verbatim.
            _previous_response_not_found(previous_response_id)
        merged = await _merge_previous_response(request, previous_response_id)
        if native_supported:
            # Falling back to native (Mantle) generation: the merged input
            # already carries the stored conversation inline, and the Mantle
            # payload builder rejects a previous_response_id that is not a
            # Mantle-tagged ID.
            return merged
        # Restored so downstream consumers (e.g. streaming SSE events built
        # from this request) echo it, as the local-store backend does.
        return merged.model_copy(update={"previous_response_id": previous_response_id})
    if not native_supported:
        msg = (
            "previous_response_id cannot be continued with this model. "
            "Retry with the model that created it."
        )
        raise ApiError(msg, status=404)
    return request


async def _merge_previous_response(
    request: ResponseCreateParams, previous_response_id: str
) -> ResponseCreateParams:
    """Prepend a stored response's conversation to the request input.

    Rebuilds the request with the stored input items, the stored output
    items, and the new input, in that order. Instructions are not carried
    over, per the OpenAI API.

    Args:
        request: The incoming request.
        previous_response_id: ID of the stored response to continue.

    Returns:
        The rebuilt request without ``previous_response_id``.

    Raises:
        ApiError: 404 when the stored response does not exist.
    """
    try:
        stored = await load_stored_response(previous_response_id, "response")
    except ApiError as error:
        if error.status != 404:
            raise
        _previous_response_not_found(previous_response_id)
    data = request.model_dump(mode="json", exclude_unset=True, by_alias=True)
    data.pop("previous_response_id", None)
    new_input = data.get("input") or []
    if isinstance(new_input, str):
        new_input = [{"role": "user", "content": new_input}]
    stored_input = stored.get("input") or []
    if isinstance(stored_input, str):
        stored_input = [{"role": "user", "content": stored_input}]
    data["input"] = [
        *stored_input,
        *stored.get("response", {}).get("output", []),
        *new_input,
    ]
    with validation_error_handler():
        return ResponseCreateParams.model_validate(data)


def _malformed_stored_document(response_id: str, detail: str) -> Never:
    """Log a malformed stored response document and raise the standard 404.

    Guards against a foreign or corrupt document (schema drift, a document
    written by an incompatible version) crashing route handling instead of
    surfacing as a normal not-found error.

    Args:
        response_id: Stored response identifier.
        detail: Diagnostic detail recorded in the warning log.

    Raises:
        ApiError: Always, with status 404.
    """
    log_error_details(
        f"Discarding malformed stored response document for '{response_id}': {detail}",
        level="warning",
    )
    msg = f"Response with id '{response_id}' not found."
    raise ApiError(msg, status=404)


def _normalized_input_items(stored_input: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """Normalize a stored request input into listable input items.

    Plain strings and string-content messages become ``message`` items with
    content parts; every item gets an ID for cursor pagination.

    Args:
        stored_input: The ``input`` value of a stored response document.

    Returns:
        Input items as JSON objects, in conversation order.
    """
    raw = stored_input or []
    if isinstance(raw, str):
        raw = [{"role": "user", "content": raw}]
    items: list[dict[str, Any]] = []
    for index, entry in enumerate(raw):
        # Drop null fields from already-stored documents (e.g. a null `phase`
        # or `type` dumped before storage started excluding them).
        item = {key: value for key, value in dict(entry).items() if value is not None}
        if isinstance(content := item.get("content"), str):
            if item.get("role") == "assistant":
                item["content"] = [
                    {"type": "output_text", "text": content, "annotations": []}
                ]
            else:
                item["content"] = [{"type": "input_text", "text": content}]
            item.setdefault("status", "completed")
        item.setdefault("type", "message")
        item.setdefault("id", f"msg-{index}")
        items.append(item)
    return items


#: Adapter validating a normalized input item against the listable ResponseItem union.
_RESPONSE_ITEM_ADAPTER: TypeAdapter[ResponseItem] = TypeAdapter[ResponseItem](
    ResponseItem
)

#: Safe default backfilled onto a stored item missing this field, keyed by field name.
_ITEM_FIELD_DEFAULTS: dict[str, Any] = {"status": "completed", "summary": []}


def _coercible_field_defaults() -> dict[str, dict[str, Any]]:
    """Map each ResponseItem type literal to its coercible required-field defaults.

    Derived from the ResponseItem union members: a field is coercible for a
    given item type when it is required (no default) on the matching member
    and has a known safe default in ``_ITEM_FIELD_DEFAULTS``. This lets
    canonical shapes clients legitimately store (e.g. a ``function_call``
    without ``status``) survive strict validation instead of being dropped.

    Returns:
        Item type literal to the ``{field: default}`` pairs safe to backfill.
    """
    defaults_by_type: dict[str, dict[str, Any]] = {}
    for member in get_args(ResponseItem):
        type_field = member.model_fields.get("type")
        if type_field is None:
            continue
        type_args = get_args(type_field.annotation)
        if len(type_args) != 1:
            continue
        defaults = {
            name: default
            for name, default in _ITEM_FIELD_DEFAULTS.items()
            if (field := member.model_fields.get(name)) is not None
            and field.is_required()
        }
        if defaults:
            defaults_by_type.setdefault(type_args[0], {}).update(defaults)
    return defaults_by_type


#: Item type literal to required-field defaults, derived from the ResponseItem union.
_COERCIBLE_FIELD_DEFAULTS: dict[str, dict[str, Any]] = _coercible_field_defaults()


def _listable_input_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop stored input items whose type is not part of the ResponseItem union.

    Mirrors the input's accepted-and-dropped semantics: an item type that
    request creation accepts but never turns into conversation history
    (e.g. ``item_reference``) is silently absent from the listing too.
    Items missing a required field with a known safe default (e.g. a
    ``function_call_output`` without ``status``) are backfilled before
    validation so canonical stored shapes are not dropped.

    Args:
        items: Normalized input items, in listing order.

    Returns:
        The items (backfilled where applicable) that validate against the
        ResponseItem union.
    """
    listable = []
    for item in items:
        candidate = item
        item_type = item.get("type")
        defaults = (
            _COERCIBLE_FIELD_DEFAULTS.get(item_type)
            if isinstance(item_type, str)
            else None
        )
        if defaults and (
            missing := {k: v for k, v in defaults.items() if k not in item}
        ):
            candidate = {**item, **missing}
        try:
            _RESPONSE_ITEM_ADAPTER.validate_python(candidate)
        except ValidationError:
            continue
        listable.append(candidate)
    return listable


def _with_public_mantle_ids(
    region: RegionName, payload: dict[str, Any]
) -> dict[str, Any]:
    """Rewrite native Mantle response IDs to their public region-tagged form.

    Args:
        region: Region storing the Mantle response.
        payload: Raw Mantle response payload.

    Returns:
        The payload with public ``id`` and ``previous_response_id`` values.
    """
    if native_id := payload.get("id"):
        payload["id"] = encode_mantle_response_id(region, native_id)
    if previous_id := payload.get("previous_response_id"):
        payload["previous_response_id"] = encode_mantle_response_id(region, previous_id)
    return payload


def _compaction_user_messages(items: Sequence[Any]) -> list[CompactionUserMessage]:
    """Echo the conversation's user messages for the compacted output.

    Args:
        items: Validated input items of the compaction generation request.

    Returns:
        The user messages as output items, in conversation order.
    """
    messages: list[CompactionUserMessage] = []
    for item in items:
        if getattr(item, "role", None) != "user":
            continue
        content = item.content
        parts: list[Any] = (
            [{"type": "input_text", "text": content}]
            if isinstance(content, str)
            else [
                part
                if isinstance(part, dict)
                else part.model_dump(mode="json", by_alias=True, exclude_none=True)
                for part in content or ()
            ]
        )
        messages.append(
            CompactionUserMessage(
                id=f"msg-{REQUEST_ID.get()}-{len(messages)}", content=parts
            )
        )
    return messages


async def _apply_prompt_template(
    prompt: ResponsePrompt | None, chat_model: ChatModelBase[Any, Any], model_id: str
) -> None:
    """Resolve a ``prompt`` reference and pin the request to the Bedrock prompt resource.

    Args:
        prompt: The request's ``prompt`` template reference, if any.
        chat_model: Chat model selected for the request.
        model_id: Model ID resolved from the request's ``model`` field.

    Raises:
        ApiError: If the model cannot serve managed prompts, or if the request's
            ``model`` is not the model configured on the prompt.
    """
    if prompt is None:
        return
    if chat_model.IS_MANTLE:
        msg = f"Model '{model_id}' does not support the 'prompt' parameter."
        raise ApiError(msg)
    resolved = await resolve_bedrock_prompt(prompt.id, prompt.version)
    if resolved.model_id != model_id:
        msg = (
            f"Prompt '{prompt.id}' runs on model '{resolved.model_id}', which does "
            f"not match the requested model '{model_id}'."
        )
        raise ApiError(msg)
    BEDROCK_PROMPT_VAR.set(resolved)


@router.post(
    "",
    summary="Generate a model response using the Responses API (OpenAI format)",
    operation_id="openai_response",
    description=(
        "Creates a model response (OpenAI Responses API).\n\n"
        "Supports streaming, tool calling, and structured outputs. "
        "Returns a `Response` object, or a stream of `ResponseStreamEvent` objects when `stream=true`.\n\n"
        "**Supported input modalities:**\n"
        "- **Text:** Plain strings or `input_text` content blocks.\n"
        "- **Images:** `input_image` content blocks with a URL, data URI, base64 image, "
        "or Files API `file_id` obtained from `openai_file`.\n"
        "- **Files:** `input_file` content blocks with a URL, base64 data, "
        "or Files API `file_id` obtained from `openai_file`.\n"
        "- **Audio input** is not supported — use `openai_chat_completion` for audio input.\n\n"
        "**When to use:** This is the newer OpenAI API style. For the classic `messages`-array format, "
        "use `openai_chat_completion` instead. For Anthropic SDK compatibility, use `anthropic_message`.\n\n"
        "**Find compatible models:** Call `search_models` with `mcp_tool=openai_response` "
        "to discover model IDs that support this endpoint. "
        "For image inputs, also add `input_modalities=IMAGE` to the filter."
    ),
    response_description="A model response.",
    status_code=200,
    response_model=Response,
    responses={
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
    },
    response_model_exclude_none=True,
)
async def create_response(
    request: ResponseCreateParams, _: Annotated[None, Depends(authenticate)] = None
) -> Response | EventSourceResponse:
    """Create a model response using AWS Bedrock Converse APIs.

    Compatible with the OpenAI Responses API. Maps input items and parameters
    to the Bedrock Converse/ConverseStream APIs and returns an OpenAI-compatible
    response.

    Args:
        request: Responses API creation request.

    Returns:
        - Response when stream is False.
        - EventSourceResponse streaming ResponseStreamEvent events when stream is True.

    Note:
        Mantle models without native Responses storage use the local response
        store like classic models. On the Mantle native-store path, ``store``
        is dropped with a logged warning when the model falls back
        mid-request to an upstream API without storage (agent harnesses
        request it unconditionally): the response is served but its ID is
        not retrievable later.

    Raises:
        ApiError: If the model is invalid, or ``previous_response_id`` does
            not exist or cannot be continued with the target model (404).
    """
    log_request_params(request, user_id=request.safety_identifier or request.user)
    store = bool(request.store)
    apply_request_moderation(request.moderation)
    model_id = (
        await validate_model(
            request.model, input_modality="TEXT", output_modality="TEXT"
        )
    ).id
    chat_model = get_chat_model(model_id)
    await _apply_prompt_template(request.prompt, chat_model, model_id)
    previous_response_id = request.previous_response_id
    native_supported = chat_model.native_store_supported()
    request = await _apply_previous_response(request, native_supported=native_supported)
    if native_supported:
        # Mantle native storage handles store/previous_response_id upstream.
        store = False
    elif store and request.stream:
        log_error_details(
            "'store' is not supported with streaming on this backend: ignored.",
            level="warning",
        )
        store = False
    session_id = await try_create_stored_response_session("response") if store else None
    store = session_id is not None
    response_id = f"resp-{session_id}" if store else f"resp-{REQUEST_ID.get()}"
    created_at = REQUEST_TIME.get().timestamp()
    try:
        result = await chat_model.create_response(
            request,
            response_id,
            created_at,
            moderation_builder=partial(build_response_moderation, request.moderation),
        )
        if (
            isinstance(result, Response)
            and result.status == "failed"
            and not request.background
        ):
            # A synchronous request must not swallow an upstream failure into
            # a 200 with empty output (Mantle models already enforce this
            # upstream); background requests keep the failed terminal state.
            _failed_response_error(result)
    except BaseException:
        if store:
            await discard_stored_response_session(response_id, "response")
        raise
    if isinstance(result, Response):
        if previous_response_id:
            result.previous_response_id = previous_response_id
        if store:
            try:
                await save_stored_response(
                    response_id,
                    {
                        # Absent for a `prompt` request: Bedrock renders the input.
                        "input": request.model_dump(
                            mode="json",
                            by_alias=True,
                            include={"input"},
                            exclude_none=True,
                        ).get("input", []),
                        "response": result.model_dump(
                            mode="json", by_alias=True, exclude_none=True
                        ),
                    },
                )
            except BaseException:
                await discard_stored_response_session(response_id, "response")
                raise
    return result


@router.post(
    "/input_tokens",
    summary="Count input tokens for a Responses request without generating a response (OpenAI format)",
    operation_id="openai_response_input_tokens",
    description=(
        "Counts the number of tokens a given request would consume, "
        "without creating a response.\n\n"
        "Accepts the same input as `openai_response` (messages, instructions, "
        "tools, images, files) and returns only the token count. Useful for "
        "estimating costs or checking context-window fit before making a full "
        "`openai_response` call.\n\n"
        "**Find compatible models:** Call `search_models` with "
        "`mcp_tool=openai_response_input_tokens` to discover model IDs that "
        "support this endpoint."
    ),
    response_description="Token count for the provided input.",
    status_code=200,
    response_model=InputTokenCountResponse,
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {"object": "response.input_tokens", "input_tokens": 142}
                }
            },
        },
        400: {"description": "Invalid request or unsupported parameters."},
    },
    response_model_exclude_none=True,
)
async def count_input_tokens(
    request: InputTokenCountParams, _: Annotated[None, Depends(authenticate)] = None
) -> InputTokenCountResponse:
    """Count the number of input tokens for a Responses request.

    Uses the AWS Bedrock CountTokens API to return an accurate,
    model-specific token count without generating a response.

    Args:
        request: Input-token count request following the OpenAI Responses spec.

    Returns:
        ResponseInputTokensCount with the input token count.

    Raises:
        ApiError: If the model is invalid, the request is unsupported, or the
            model is served by Bedrock Mantle (400).
    """
    log_request_params(request)
    model = await validate_model(
        request.model, input_modality="TEXT", output_modality="TEXT", error_status=400
    )
    model_id = model.get_id()
    if serves_via_mantle(model_id):
        msg = "Token counting is not supported for this model on this endpoint."
        raise ApiError(msg, status=400)
    # Mantle is excluded above, so the model is always a Converse ChatModel.
    chat_model = cast("ChatModel", get_chat_model(model_id))
    return log_response_params(
        InputTokenCountResponse(
            input_tokens=await count_input_tokens_via_bedrock(
                request,
                model_id,
                model.regions[0],
                reasoning_signature_required=chat_model.REASONING_SIGNATURE_REQUIRED,
            )
        )
    )


@router.post(
    "/compact",
    summary="Compact a conversation into a reusable summary item (OpenAI format)",
    operation_id="openai_response_compact",
    description=(
        "Compacts a conversation into a single `compaction` output item "
        "(OpenAI Responses API).\n\n"
        "The model summarises the provided `input`; the summary is returned as "
        "an opaque `compaction` item. Include that item in the `input` of later "
        "`openai_response` calls to continue the conversation with a reduced "
        "context window.\n\n"
        "The compaction content is self-contained, so no conversation state "
        "is needed on the server; `previous_response_id` may reference a "
        "stored response to compact its conversation too.\n\n"
        "**Find compatible models:** Call `search_models` with "
        "`mcp_tool=openai_response_compact` to discover model IDs that "
        "support this endpoint."
    ),
    response_description="The compacted response.",
    responses={
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
    },
    response_model_exclude_none=True,
)
async def compact_response(
    request: CompactParams, _: Annotated[None, Depends(authenticate)] = None
) -> CompactedResponse:
    """Compact a conversation into a single compaction item.

    Runs a summarisation pass on AWS Bedrock and wraps the resulting summary
    in an opaque ``compaction`` item that later requests can send back as
    input.

    Args:
        request: Compaction request following the OpenAI Responses spec.

    Returns:
        CompactedResponse holding the compaction item and token usage.

    Raises:
        ApiError: If the model is invalid or there is no conversation to
            compact (400), or ``previous_response_id`` does not exist or
            references a Bedrock Mantle stored conversation (404).
    """
    log_request_params(request)
    model_id = (
        await validate_model(
            request.model, input_modality="TEXT", output_modality="TEXT"
        )
    ).id
    response_id = f"resp-{REQUEST_ID.get()}"
    created_at = REQUEST_TIME.get().timestamp()
    items: list[ResponseInputItem] = (
        [EasyInputMessage(role="user", content=request.input)]
        if isinstance(request.input, str)
        else list(request.input or ())
    )
    generation = ResponseCreateParams(
        model=request.model,
        input=items,
        instructions=request.instructions,
        prompt_cache_key=request.prompt_cache_key,
        prompt_cache_options=request.prompt_cache_options,
        prompt_cache_retention=request.prompt_cache_retention,
        service_tier=request.service_tier,
        previous_response_id=request.previous_response_id,
    )
    # Compaction never chains natively, so a Mantle-stored ID is a 404.
    generation = await _apply_previous_response(generation, native_supported=False)
    items = (
        [EasyInputMessage(role="user", content=generation.input)]
        if isinstance(generation.input, str)
        else list(generation.input or ())
    )
    if not items:
        msg = "There is no conversation to compact."
        raise ApiError(msg)
    user_messages = _compaction_user_messages(items)
    generation = generation.model_copy(
        update={
            "input": [*items, EasyInputMessage(role="user", content=_COMPACTION_PROMPT)]
        }
    )
    response = await get_chat_model(model_id).create_response(
        generation, response_id, created_at
    )
    if not isinstance(response, Response):  # pragma: no cover - stream is never set
        msg = "Unexpected streaming response."
        raise TypeError(msg)
    summary = "".join(
        part.text
        for item in response.output
        if isinstance(item, ResponseOutputMessage)
        for part in item.content
        if isinstance(part, ResponseOutputText)
    )
    return log_response_params(
        CompactedResponse(
            id=response_id,
            created_at=int(created_at),
            output=[
                *user_messages,
                ResponseCompactionItem(
                    id=f"ci-{REQUEST_ID.get()}",
                    encrypted_content=encode_compaction_content(summary),
                    type="compaction",
                ),
            ],
            usage=response.usage
            or ResponseUsage(
                input_tokens=0,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=0,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=0,
            ),
        )
    )


async def _mantle_stored_response(
    region: RegionName,
    method: str,
    native_id: str,
    suffix: str = "",
    missing_msg: str | None = None,
) -> dict[str, Any]:
    """Proxy a stored-response operation to Mantle, probing both routing surfaces.

    Stored responses live on the surface that served the model (``/v1`` or
    ``/openai/v1``); a 404 on the first surface falls through to the second.
    The working surface per response ID is seeded at creation and cached, so
    the 404 probe only runs on unknown IDs (e.g. after a restart). Requests
    carry the caller's ``OpenAI-Project`` scoping so the response is
    addressed in the same project it was created under.

    Args:
        region: Region storing the response.
        method: HTTP method (GET, POST or DELETE).
        native_id: Native Mantle response identifier.
        suffix: Optional sub-resource path (e.g. ``/input_items``).
        missing_msg: Optional 404 message override.

    Returns:
        Parsed JSON response body.

    Raises:
        ApiError: 404 (with the public ID) when the response is missing on
            both surfaces.
        MantleError: On other upstream errors.
    """
    prefix: Surface = cached_response_surface(native_id) or "/openai/v1"
    headers = mantle_request_headers("responses")
    try:
        payload = await request_json(
            region, method, f"{prefix}/responses/{native_id}{suffix}", headers=headers
        )
    except MantleError as error:
        if error.status != 404:
            raise
        prefix = "/openai/v1" if prefix == "/v1" else "/v1"
        try:
            payload = await request_json(
                region,
                method,
                f"{prefix}/responses/{native_id}{suffix}",
                headers=headers,
            )
        except MantleError as retry_error:
            if retry_error.status != 404:
                raise
            # Normalize the upstream 404 to the public identifier.
            public_id = encode_mantle_response_id(region, native_id)
            msg = missing_msg or f"Response '{public_id}' not found."
            raise ApiError(msg, status=404) from retry_error
    cache_response_surface(native_id, prefix)
    return payload


@router.get(
    "/{response_id}",
    summary="Retrieve a stored model response (OpenAI format)",
    operation_id="openai_response_get",
    description=(
        "Returns a model response previously persisted with `store=true` "
        "(OpenAI Responses API).\n\n"
        "Pass the response ID as `previous_response_id` in `openai_response` "
        "to continue the conversation."
    ),
    response_description="The stored response.",
    responses={
        200: {"description": "The stored response."},
        404: {"description": "Response not found."},
    },
    response_model_exclude_none=True,
)
async def retrieve_response(
    response_id: _ResponseId,
    stream: Annotated[
        bool | None,
        Query(
            description="Retrieving a response as a stream is not supported "
            "on this implementation."
        ),
    ] = None,
    include: Annotated[
        list[ResponseIncludable] | None,
        Query(description="Accepted for compatibility and ignored."),
    ] = None,
    starting_after: Annotated[
        int | None, Query(description="Accepted for compatibility and ignored.")
    ] = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> Response:
    """Retrieve a stored model response.

    Region-tagged Mantle IDs are fetched from the Mantle native store.

    Args:
        response_id: Stored response identifier.
        stream: Whether to retrieve the response as an SSE stream (unsupported).
        include: Additional output data to include. Accepted and ignored.
        starting_after: Streaming event cursor. Accepted and ignored.

    Returns:
        The stored response.

    Raises:
        ApiError: With 400 if ``stream`` is set, or 404 if the stored
            response does not exist.
    """
    log_request_params(
        {
            "response_id": response_id,
            "stream": stream,
            "include": include,
            "starting_after": starting_after,
        }
    )
    if stream:
        msg = (
            "Retrieving a response as a stream is not supported on this implementation."
        )
        raise ApiError(msg, status=400)
    if mantle := _decode_mantle_id(response_id):
        region, native_id = mantle
        payload = await _mantle_stored_response(region, "GET", native_id)
        return log_response_params(
            validate_pruning_extras(Response, _with_public_mantle_ids(region, payload))
        )
    _require_local_response_id(response_id)
    stored = await load_stored_response(response_id, "response")
    try:
        result = Response.model_validate(stored["response"])
    except (KeyError, TypeError, ValidationError) as error:
        _malformed_stored_document(response_id, str(error))
    return log_response_params(result)


@router.post(
    "/{response_id}/cancel",
    summary="Cancel a background model response (OpenAI format)",
    operation_id="openai_response_cancel",
    description=(
        "Cancels a model response (OpenAI Responses API). Only responses "
        "created with `background=true` can be cancelled; cancelling any "
        "other response fails with the OpenAI error for synchronous "
        "responses."
    ),
    response_description="The cancelled response.",
    responses={
        200: {"description": "The cancelled response."},
        400: {"description": "The response is synchronous and cannot be cancelled."},
        404: {"description": "Response not found."},
    },
    response_model_exclude_none=True,
)
async def cancel_response(
    response_id: _ResponseId, _: Annotated[None, Depends(authenticate)] = None
) -> Response:
    """Cancel a background model response.

    Region-tagged Mantle IDs are proxied to the Mantle native store; local
    responses are always synchronous and cannot be cancelled.

    Args:
        response_id: Stored response identifier.

    Returns:
        The cancelled response (Mantle-stored responses only).

    Raises:
        ApiError: With 404 if the stored response does not exist, else with
            400 when the response is synchronous and cannot be cancelled.
    """
    log_request_params({"response_id": response_id})
    if mantle := _decode_mantle_id(response_id):
        region, native_id = mantle
        payload = await _mantle_stored_response(region, "POST", native_id, "/cancel")
        return log_response_params(
            validate_pruning_extras(Response, _with_public_mantle_ids(region, payload))
        )
    _require_local_response_id(response_id)
    await load_stored_response(response_id, "response")
    msg = "Cannot cancel a synchronous response."
    raise ApiError(msg)


@router.delete(
    "/{response_id}",
    summary="Delete a stored model response (OpenAI format)",
    operation_id="openai_response_delete",
    description=(
        "Deletes a model response previously persisted with `store=true`, "
        "along with its stored conversation state (OpenAI Responses API)."
    ),
    response_description="Deletion confirmation.",
    responses={
        200: {"description": "Response deleted."},
        404: {"description": "Response not found."},
    },
    response_model_exclude_none=True,
)
async def delete_response(
    response_id: _ResponseId, _: Annotated[None, Depends(authenticate)] = None
) -> ResponseDeleted:
    """Delete a stored model response.

    Region-tagged Mantle IDs are deleted from the Mantle native store.

    Args:
        response_id: Stored response identifier.

    Returns:
        Deletion confirmation.

    Raises:
        ApiError: With 404 if the stored response does not exist.
    """
    log_request_params({"response_id": response_id})
    if mantle := _decode_mantle_id(response_id):
        await _mantle_stored_response(mantle[0], "DELETE", mantle[1])
    else:
        _require_local_response_id(response_id)
        await delete_stored_response(response_id, "response")
    return log_response_params(ResponseDeleted(id=response_id))


@router.get(
    "/{response_id}/input_items",
    summary="List the input items of a stored model response (OpenAI format)",
    operation_id="openai_response_input_items",
    description=(
        "Returns the input items that produced a stored model response "
        "(OpenAI Responses API)."
    ),
    response_description="A paginated list of input items.",
    responses={
        200: {"description": "The input items."},
        404: {"description": "Response not found."},
    },
    response_model_exclude_none=True,
)
async def list_response_input_items(
    response_id: _ResponseId,
    after: Annotated[
        str | None,
        Query(
            description=(
                "Cursor for pagination: the item ID to start after "
                "(the last ID from a previous page)."
            ),
            max_length=255,
        ),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=100, description="A limit on the number of objects returned."),
    ] = 20,
    order: Annotated[
        Literal["asc", "desc"],
        Query(description="Sort order: `asc` is conversation order."),
    ] = "desc",
    include: Annotated[
        list[ResponseIncludable] | None,
        Query(description="Accepted for compatibility and ignored."),
    ] = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> ResponseItemList:
    """List the input items of a stored model response.

    Region-tagged Mantle IDs are proxied to the Mantle native store; the
    upstream list is returned as-is (pagination parameters are not forwarded).

    Args:
        response_id: Stored response identifier.
        after: Item ID cursor; only items strictly after it are returned.
        limit: Maximum number of items to return.
        order: Sort order relative to the conversation order.
        include: Additional output data to include. Accepted and ignored.

    Returns:
        Paginated list of input items.

    Raises:
        ApiError: With 404 if the stored response does not exist, or if
            ``after`` does not match any input item.
    """
    log_request_params(
        {
            "response_id": response_id,
            "after": after,
            "limit": limit,
            "order": order,
            "include": include,
        }
    )
    if mantle := _decode_mantle_id(response_id):
        region, native_id = mantle
        # Bedrock Mantle does not currently serve input item listings.
        missing = (
            f"Input items for response '{response_id}' are not available: "
            "Bedrock Mantle stored responses do not serve input item listings."
        )
        payload = await _mantle_stored_response(
            region, "GET", native_id, "/input_items", missing_msg=missing
        )
        for entry in payload.get("data") or ():
            # Only response IDs are region-tagged; message item IDs pass through.
            if isinstance(entry, dict) and str(entry.get("id") or "").startswith(
                "resp"
            ):
                _with_public_mantle_ids(region, entry)
        return log_response_params(validate_pruning_extras(ResponseItemList, payload))
    _require_local_response_id(response_id)
    stored = await load_stored_response(response_id, "response")
    if not isinstance(stored.get("response"), dict):
        _malformed_stored_document(response_id, "'response' is not a JSON object")
    try:
        items = _listable_input_items(_normalized_input_items(stored.get("input")))
    except (KeyError, TypeError, ValueError) as error:
        _malformed_stored_document(response_id, str(error))
    if order == "desc":
        items.reverse()
    if after is not None:
        index = next(
            (i for i, item in enumerate(items) if item.get("id") == after), None
        )
        if index is None:
            msg = f"No input item with id '{after}'."
            raise ApiError(msg, status=404)
        items = items[index + 1 :]
    page, has_more = items[:limit], len(items) > limit
    return log_response_params(
        ResponseItemList.model_validate(
            {
                "object": "list",
                "data": page,
                "first_id": page[0]["id"] if page else None,
                "last_id": page[-1]["id"] if page else None,
                "has_more": has_more,
            }
        )
    )
