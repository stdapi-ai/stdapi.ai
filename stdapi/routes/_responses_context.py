"""Context-window handling of POST /v1/responses: truncation and compaction.

A model served natively on Bedrock Mantle's Responses API honours
``truncation``, ``context_management`` and ``compaction_trigger`` itself, so
its requests pass through untouched. Every other model gets them from here.
"""

from dataclasses import dataclass
from functools import partial
from sys import maxsize
from typing import TYPE_CHECKING, Any, Final, Never

from botocore.exceptions import BotoCoreError, ClientError
from sse_starlette import EventSourceResponse

from stdapi.api_errors import ApiError
from stdapi.models import reject_unsupported_token_counting
from stdapi.models.chat import serves_via_mantle
from stdapi.models.chat._adapters._openai_responses import (
    close_stream,
    compaction_response,
    count_input_tokens_via_bedrock,
    encode_compaction_state,
    expand_compaction_items,
    format_compaction_stream,
    format_failed_stream,
    join_summaries,
    merge_usage,
    replay_stream,
    with_user_text,
)
from stdapi.models.chat._adapters._responses_context import (
    CompactionSplit,
    ContextLengthExceededError,
    collect_stream_open_errors,
    context_overflow,
    estimate_tokens,
    split_everything,
    split_for_compaction,
    truncate_input,
)
from stdapi.monitoring import REQUEST_ID, log_request_sse_stream_event
from stdapi.types.openai_responses import (
    CompactionItemParam,
    CompactionTrigger,
    EasyInputMessage,
    InputTokenCountParams,
    Response,
    ResponseCompactionItem,
    ResponseCreateParams,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from stdapi.utils import hide_security_details

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence

    from sse_starlette import ServerSentEvent

    from stdapi.models import ModelDetails
    from stdapi.models.chat import ChatModelBase
    from stdapi.models.chat._adapters._responses_context import ContextOverflow
    from stdapi.types.openai import ResponseModeration
    from stdapi.types.openai_responses import ResponseInputItem, ResponseInputParam

#: Directive appended to the conversation to produce the compaction summary.
COMPACTION_PROMPT: Final = (
    "Summarize the conversation above in detail, preserving every fact, "
    "decision, constraint, open task, and tool result needed to continue it. "
    "Reply with the summary only."
)

#: Retries a ``truncation: "auto"`` request gets after its input overflows the context window.
_MAX_TRUNCATION_RETRIES: Final = 3

#: Refusal of the token counter for a model it cannot count.
_UNCOUNTABLE_MESSAGE: Final = "support counting tokens"

#: Models the token counter refused, whose compaction threshold is estimated instead.
_UNCOUNTABLE_MODELS: set[str] = set()


@dataclass(frozen=True, slots=True)
class Generated:
    """A generated response, and what produced it."""

    #: The request the response was generated from, after truncation or compaction.
    request: ResponseCreateParams
    result: Response | EventSourceResponse
    #: The compaction item the (non-streamed) response starts with, if any.
    compaction: ResponseCompactionItem | None = None
    #: Token usage of that compaction.
    compaction_usage: ResponseUsage | None = None
    #: Whether the model answered, so the response's usage measures the conversation.
    answered: bool = True


@dataclass(frozen=True, slots=True)
class _Call:
    """What the model calls producing one response share."""

    chat_model: ChatModelBase[Any, Any]
    model_id: str
    response_id: str
    created_at: float
    #: Builds the response's ``moderation`` field.
    moderation_builder: Callable[[], ResponseModeration | None] | None
    #: Whether the backend refuses local compaction items and consecutive user messages.
    expands_items: bool = False

    async def compact(
        self, request: ResponseCreateParams, split: CompactionSplit
    ) -> tuple[str, ResponseUsage | None]:
        """Summarize a split input into the content of a compaction item.

        Args:
            request: The request whose settings the summarisation uses.
            split: What to summarize and what to keep verbatim.

        Returns:
            The compaction item content, and the usage the summarisation cost.
        """
        summary, usage = await summarize(
            self.chat_model,
            request,
            split.summarized,
            self.response_id,
            self.created_at,
            joins_user_messages=self.expands_items,
        )
        return await encode_compaction_state(summary, split.before, split.after), usage

    async def open(
        self, request: ResponseCreateParams
    ) -> tuple[ResponseCreateParams, Response | EventSourceResponse]:
        """Generate the answer, under the request's ``truncation``.

        Args:
            request: The request.

        Returns:
            The request that was answered, and its response or stream.
        """
        if self.expands_items and isinstance(request.input, list):
            request = request.model_copy(
                update={"input": join_summaries(await _expanded(request.input))}
            )
        return await open_response(
            self.chat_model,
            request,
            response_id=self.response_id,
            created_at=self.created_at,
            moderation_builder=self.moderation_builder,
        )

    async def open_stream(self, request: ResponseCreateParams) -> EventSourceResponse:
        """Open the answer's stream, under the request's ``truncation``.

        Args:
            request: The streamed request.

        Returns:
            The stream.

        Raises:
            TypeError: If the model answered without streaming.
        """
        _, result = await self.open(request)
        if isinstance(result, Response):  # pragma: no cover - stream is set
            msg = "Unexpected non-streaming response."
            raise TypeError(msg)
        return result

    @staticmethod
    def stream(events: AsyncGenerator[ServerSentEvent]) -> EventSourceResponse:
        """Serve stream events, logged and error-guarded.

        Args:
            events: The events.

        Returns:
            The streaming response.
        """
        return EventSourceResponse(log_request_sse_stream_event(events))


def failed_response_error(response: Response) -> Never:
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
        # Same fallback wording as the Mantle passthrough failed-response guard.
        message = "The upstream model response failed."
    raise ApiError(hide_security_details(502, message), status=502)


def compaction_item_id() -> str:
    """Return the ID of the compaction item this request produces.

    Returns:
        The ID, with upstream's ``cmp_`` prefix.
    """
    return f"cmp_{REQUEST_ID.get()}"


async def summarize(
    chat_model: ChatModelBase[Any, Any],
    settings: ResponseCreateParams,
    items: Sequence[ResponseInputItem],
    response_id: str,
    created_at: float,
    *,
    joins_user_messages: bool = False,
) -> tuple[str, ResponseUsage | None]:
    """Summarize a conversation with the model that continues it.

    Args:
        chat_model: The model.
        settings: Request whose instructions, caching and service tier apply.
        items: The conversation to summarize.
        response_id: Identifier of the response being produced.
        created_at: Unix timestamp of the request.
        joins_user_messages: Whether the backend refuses consecutive user
            messages, so summaries and the directive join their neighbours.

    Returns:
        The summary, and the usage the summarisation cost.

    Raises:
        TypeError: If the model streamed its answer.
        ContextLengthExceededError: When the conversation does not fit the
            context window, and ``truncation`` cannot make it fit.
    """
    native = chat_model.native_store_supported()
    generation = ResponseCreateParams(
        model=settings.model,
        # A native model reads its own compaction items; others get ours expanded.
        input=list(items) if native else await _expanded(items),
        instructions=settings.instructions,
        prompt_cache_key=settings.prompt_cache_key,
        prompt_cache_options=settings.prompt_cache_options,
        prompt_cache_retention=settings.prompt_cache_retention,
        service_tier=settings.service_tier,
        truncation=settings.truncation,
    )

    def _sent(conversation: ResponseCreateParams) -> ResponseCreateParams:
        # The directive joins each attempt, so truncation protects the
        # conversation's latest turn rather than the directive.
        items = conversation.input if isinstance(conversation.input, list) else []
        if joins_user_messages and not native:
            items = with_user_text(join_summaries(items), COMPACTION_PROMPT)
        else:
            items = [*items, EasyInputMessage(role="user", content=COMPACTION_PROMPT)]
        return conversation.model_copy(update={"input": items})

    _, response = await truncating(
        generation,
        lambda conversation, _retryable: chat_model.create_response(
            _sent(conversation), response_id, created_at
        ),
    )
    if not isinstance(response, Response):  # pragma: no cover - stream is never set
        msg = "Unexpected streaming response."
        raise TypeError(msg)
    if response.status == "failed":
        # A failed summarisation run must not become an empty summary.
        failed_response_error(response)
    summary = "".join(
        part.text
        for item in response.output
        if isinstance(item, ResponseOutputMessage)
        for part in item.content
        if isinstance(part, ResponseOutputText)
    )
    return summary, response.usage


async def _expanded(
    value: ResponseInputParam | Sequence[ResponseInputItem] | None,
) -> list[ResponseInputItem]:
    """Return an input as items, locally-produced compaction items expanded.

    Args:
        value: The request's ``input``.

    Returns:
        The items.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [EasyInputMessage(role="user", content=value)]
    if any(isinstance(item, CompactionItemParam) for item in value):
        return await expand_compaction_items(value)
    return list(value)


def _answered_from(
    request: ResponseCreateParams, item: ResponseCompactionItem
) -> ResponseCreateParams:
    """Return the request answered from a compaction item instead of its input.

    Args:
        request: The request.
        item: The compaction item standing for its input.

    Returns:
        The request, rebuilt.
    """
    return request.model_copy(
        update={
            "input": [
                CompactionItemParam(
                    id=item.id,
                    encrypted_content=item.encrypted_content,
                    type="compaction",
                )
            ]
        }
    )


async def generate(
    chat_model: ChatModelBase[Any, Any],
    model: ModelDetails,
    request: ResponseCreateParams,
    response_id: str,
    created_at: float,
    moderation_builder: Callable[[], ResponseModeration | None],
    *,
    history_tokens: int | None,
    own_tokens: int,
) -> Generated:
    """Generate a response, applying the request's context-window handling.

    Args:
        chat_model: The model.
        model: The model's details.
        request: The request, its history already merged.
        response_id: Identifier of the response.
        created_at: Unix timestamp of the request.
        moderation_builder: Builds the response's ``moderation`` field.
        history_tokens: Tokens the merged stored history took, when known.
        own_tokens: Low-biased token estimate of the request's own input.

    Returns:
        The generated response.

    Raises:
        ContextLengthExceededError: When a non-streamed input does not fit the
            context window and ``truncation`` cannot make it fit.
    """
    if chat_model.native_store_supported():
        return Generated(
            request,
            await chat_model.create_response(
                request, response_id, created_at, moderation_builder=moderation_builder
            ),
        )
    call = _Call(
        chat_model,
        model.id,
        response_id,
        created_at,
        moderation_builder,
        # Bedrock Mantle refuses local compaction items; some served models, consecutive user messages.
        expands_items=serves_via_mantle(model.id),
    )
    if isinstance(request.input, list) and any(
        isinstance(item, CompactionTrigger) for item in request.input[-1:]
    ):
        return await _compact_on_demand(call, request)
    split = await plan_compaction(
        chat_model, model, request, history_tokens=history_tokens, own_tokens=own_tokens
    )
    if split is None:
        try:
            return Generated(*await call.open(request))
        except ContextLengthExceededError as exc:
            if not request.stream:
                raise
            return Generated(
                request,
                call.stream(
                    format_failed_stream(
                        response_id, created_at, model.id, request, exc
                    )
                ),
            )
    if request.stream:

        async def answer(item: CompactionItemParam) -> EventSourceResponse:
            return await call.open_stream(request.model_copy(update={"input": [item]}))

        return Generated(
            request,
            call.stream(
                format_compaction_stream(
                    response_id,
                    created_at,
                    model.id,
                    request,
                    compaction_item_id(),
                    partial(call.compact, request, split),
                    answer,
                )
            ),
        )
    content, usage = await call.compact(request, split)
    item = ResponseCompactionItem(
        id=compaction_item_id(), encrypted_content=content, type="compaction"
    )
    answered, result = await call.open(_answered_from(request, item))
    return Generated(answered, result, item, usage)


async def _compact_on_demand(call: _Call, request: ResponseCreateParams) -> Generated:
    """Answer a request ending with a ``compaction_trigger`` with its compaction.

    Everything before the trigger is summarized; system and developer
    messages are kept verbatim.

    Args:
        call: The model calls' shared settings.
        request: The request.

    Returns:
        The response holding the compaction item alone.

    Raises:
        ApiError: 400 when nothing precedes the trigger.
    """
    items = request.input if isinstance(request.input, list) else []
    split = split_everything(await _expanded(items[:-1]))
    if split is None:
        msg = "There is no conversation to compact."
        raise ApiError(msg)
    if request.stream:
        return Generated(
            request,
            call.stream(
                format_compaction_stream(
                    call.response_id,
                    call.created_at,
                    call.model_id,
                    request,
                    compaction_item_id(),
                    partial(call.compact, request, split),
                    None,
                )
            ),
        )
    content, usage = await call.compact(request, split)
    item = ResponseCompactionItem(
        id=compaction_item_id(), encrypted_content=content, type="compaction"
    )
    return Generated(
        request,
        compaction_response(
            call.response_id, call.created_at, call.model_id, request, item, usage
        ),
        answered=False,
    )


def with_compaction(
    response: Response, item: ResponseCompactionItem, usage: ResponseUsage | None
) -> Response:
    """Put a compaction item ahead of a response's output, and add its usage.

    Args:
        response: The response generated from the compacted context.
        item: The compaction item.
        usage: Token usage of the compaction.

    Returns:
        The response, updated.
    """
    response.output = [item, *response.output]
    response.usage = merge_usage(usage, response.usage)
    return response


async def open_response(
    chat_model: ChatModelBase[Any, Any],
    request: ResponseCreateParams,
    *,
    response_id: str,
    created_at: float,
    moderation_builder: Callable[[], ResponseModeration | None] | None,
) -> tuple[ResponseCreateParams, Response | EventSourceResponse]:
    """Generate a response, dropping the oldest input when it overflows.

    Under ``truncation: "auto"``, an input refused as larger than the context
    window is trimmed and sent again, at most three times.

    Args:
        chat_model: The model.
        request: The request.
        response_id: Identifier of the response.
        created_at: Unix timestamp of the request.
        moderation_builder: Builds the response's ``moderation`` field.

    Returns:
        The request that was answered, and its response or stream.

    Raises:
        ContextLengthExceededError: When the input does not fit and cannot
            be trimmed to fit.
    """
    return await truncating(
        request,
        partial(
            _open_once,
            chat_model,
            response_id=response_id,
            created_at=created_at,
            moderation_builder=moderation_builder,
        ),
    )


async def truncating[RequestT: (ResponseCreateParams, InputTokenCountParams), T](
    request: RequestT, call: Callable[[RequestT, bool], Awaitable[T]]
) -> tuple[RequestT, T]:
    """Run a model call, dropping the oldest input when it overflows the window.

    Under ``truncation: "auto"``, an input refused as larger than the context
    window is trimmed and sent again, at most three times.

    Args:
        request: The request.
        call: Runs the call on a request; its second argument says whether an
            overflow will be retried, so a stream need not report it.

    Returns:
        The request the call succeeded with, and its result.

    Raises:
        ContextLengthExceededError: When the input does not fit and cannot
            be trimmed to fit.
    """
    attempt = 0
    while True:
        retryable = request.truncation == "auto" and attempt < _MAX_TRUNCATION_RETRIES
        try:
            return request, await call(request, retryable)
        except Exception as exc:
            if (overflow := context_overflow(exc)) is None:
                raise
            retry = (
                await _truncated(request.input, overflow, attempt)
                if retryable
                else None
            )
            if retry is None:
                raise ContextLengthExceededError from exc
        request = request.model_copy(update={"input": retry})
        attempt += 1


async def _truncated(
    value: ResponseInputParam | None, overflow: ContextOverflow, attempt: int
) -> list[ResponseInputItem] | None:
    """Plan the input of a truncation retry.

    Args:
        value: The refused input.
        overflow: The refusal.
        attempt: How many retries already failed.

    Returns:
        The input to retry with, or None when nothing is left to remove.
    """
    return truncate_input(await _expanded(value), overflow, attempt)


async def _open_once(
    chat_model: ChatModelBase[Any, Any],
    request: ResponseCreateParams,
    retryable: bool,  # noqa: FBT001 - the positional contract of `truncating`
    *,
    response_id: str,
    created_at: float,
    moderation_builder: Callable[[], ResponseModeration | None] | None,
) -> Response | EventSourceResponse:
    """Generate a response once, reading a stream's first event before returning.

    Args:
        chat_model: The model.
        request: The request.
        retryable: Whether an overflow refused on the stream's first event is
            raised for a retry, instead of being streamed to the client.
        response_id: Identifier of the response.
        created_at: Unix timestamp of the request.
        moderation_builder: Builds the response's ``moderation`` field.

    Returns:
        The response, or its stream.

    Raises:
        Exception: Whatever the model raised, including an overflow refused
            on the stream's first event when ``retryable``.
    """
    with collect_stream_open_errors() as errors:
        result = await chat_model.create_response(
            request, response_id, created_at, moderation_builder=moderation_builder
        )
        if isinstance(result, Response):
            return result
        body = aiter(result.body_iterator)
        first = await anext(body, None)
    if retryable and (
        overflow := next((error for error in errors if context_overflow(error)), None)
    ):
        await close_stream(body)
        raise overflow
    result.body_iterator = replay_stream(() if first is None else (first,), body)
    return result


async def plan_compaction(
    chat_model: ChatModelBase[Any, Any],
    model: ModelDetails,
    request: ResponseCreateParams,
    *,
    history_tokens: int | None,
    own_tokens: int,
) -> CompactionSplit | None:
    """Decide whether the request's input crosses its compaction threshold.

    The input is counted where the model's token counter serves it, and
    estimated otherwise, biased low: the stored history's recorded usage plus
    the new input, or the input's text alone.

    Args:
        chat_model: The model.
        model: The model's details.
        request: The request, its history already merged.
        history_tokens: Tokens the merged stored history took, when known.
        own_tokens: Low-biased token estimate of the request's own input.

    Returns:
        How to compact the input, or None when it stays below the threshold
        or there is nothing to compact.
    """
    threshold = min(
        (
            entry.compact_threshold
            for entry in request.context_management or ()
            if entry.compact_threshold is not None
        ),
        default=None,
    )
    if threshold is None:
        return None
    expanded = await _expanded(request.input)
    split = split_for_compaction(expanded)
    if split is None:
        return None
    tokens = await _counted_tokens(chat_model, model, request)
    if tokens is None:
        tokens = (
            history_tokens + own_tokens
            if history_tokens is not None
            else estimate_tokens(expanded, request.instructions, request.tools)
        )
    return split if tokens >= threshold else None


async def _counted_tokens(
    chat_model: ChatModelBase[Any, Any],
    model: ModelDetails,
    request: ResponseCreateParams,
) -> int | None:
    """Count the request's input tokens where the model's token counter serves it.

    Args:
        chat_model: The model.
        model: The model's details.
        request: The request.

    Returns:
        The count, the largest integer for an input over the context window,
        or None when the model cannot be counted.
    """
    if serves_via_mantle(model.id) or model.id in _UNCOUNTABLE_MODELS:
        return None
    try:
        reject_unsupported_token_counting(model)
        return await count_input_tokens_via_bedrock(
            InputTokenCountParams(
                model=request.model,
                input=request.input,
                instructions=request.instructions,
                tools=request.tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
                reasoning=request.reasoning,
            ),
            model.id,
            model.regions[0],
            # Not Mantle-served, so this is always a Converse chat model.
            chat_model,  # type: ignore[arg-type]
        )
    except (ApiError, ClientError, BotoCoreError) as exc:
        if context_overflow(exc) is not None:
            return maxsize
        if isinstance(exc, ClientError) and _UNCOUNTABLE_MESSAGE in str(
            exc.response.get("Error", {}).get("Message")
        ):
            _UNCOUNTABLE_MODELS.add(model.id)
        return None
