"""OpenAI-compatible completions adapter for Bedrock Converse.

Translates the ``POST /v1/completions`` request/response shape into Bedrock
Converse primitives.  Intentionally narrower than the chat-completions adapter:
no tools, reasoning, or structured output.  Multimodal content is supported
only through the *text + files collapse* path (see :func:`build_user_messages`).
"""

from asyncio import CancelledError, Queue, create_task, gather
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from sse_starlette import JSONServerSentEvent, ServerSentEvent

from stdapi.aws import raise_first_exception
from stdapi.aws_bedrock import set_inference_configuration
from stdapi.input_file import InputFileUrl
from stdapi.models.chat._adapters import _openai_common
from stdapi.types.openai_chat_completions import (
    CompletionUsage,
    PromptTokensDetails,
    ServiceTiers,
)
from stdapi.types.openai_completions import (
    Completion,
    CompletionChoice,
    CompletionCreateParams,
    CompletionFinishReasonLiteral,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from types_aiobotocore_bedrock_runtime.literals import (
        ServiceTierTypeType,
        StopReasonType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockTypeDef,
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
        InferenceConfigurationTypeDef,
        MessageTypeDef,
    )

#: Bedrock stop-reason → OpenAI completion finish-reason.
_FINISH_REASONS: dict[str, CompletionFinishReasonLiteral] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "incomplete": "length",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
    "malformed_model_output": "content_filter",
}


def _map_finish_reason(
    stop_reason: StopReasonType | str | None,
) -> CompletionFinishReasonLiteral:
    """Map a Bedrock stop reason to an OpenAI completion ``finish_reason``.

    Falls back to ``"stop"`` for unknown or missing reasons.

    Args:
        stop_reason: Bedrock ``stopReason`` string, or ``None``.

    Returns:
        ``"stop"``, ``"length"``, or ``"content_filter"``.
    """
    return _FINISH_REASONS.get(stop_reason or "", "stop")


async def build_user_messages(
    prompt: InputFileUrl | str | list[InputFileUrl | str],
) -> list[MessageTypeDef]:
    """Build Bedrock user messages from a completion prompt.

    Each ``str`` prompt becomes a ``text`` block and each ``InputFileUrl``
    becomes an ``image`` / ``video`` / ``audio`` / ``document`` block — the
    block type is auto-detected from the file's content type and the target
    model decides what it supports.  Mapping from request shape to output:

    * A bare ``str`` or ``InputFileUrl`` → one message, one block.
    * **Text + files collapse**: a list with exactly one ``str`` and one or
      more ``InputFileUrl`` → one multimodal message combining them in input
      order, returning a single choice.  The natural "ask once using these
      files as context" pattern.
    * Any other list → one message per element, i.e. one ``Completion``
      choice per prompt.

    The model layer fills the file source (``bytes`` or ``s3Location``) for
    the target region via :func:`resolve_all_bedrock_content_blocks`, so this
    function only builds the partial content blocks.

    Args:
        prompt: Raw prompt from the request body.

    Returns:
        Ordered list of Bedrock user ``MessageTypeDef`` dicts.
    """

    async def _block(item: InputFileUrl | str) -> ContentBlockTypeDef:
        if isinstance(item, str):
            return {"text": item}
        return await item.to_bedrock_content_block()

    if isinstance(prompt, (str, InputFileUrl)):
        return [{"role": "user", "content": [await _block(prompt)]}]

    blocks = list(await gather(*(_block(item) for item in prompt)))
    if len(blocks) >= 2 and sum(1 for item in prompt if isinstance(item, str)) == 1:
        return [{"role": "user", "content": blocks}]
    return [{"role": "user", "content": [block]} for block in blocks]


def translate_request(
    request: CompletionCreateParams, model_id: str
) -> tuple[
    InferenceConfigurationTypeDef,
    dict[str, Any],
    ServiceTierTypeType | None,
    ServiceTiers | None,
    int,
    dict[str, str] | None,
]:
    """Translate a completion request into Bedrock-Converse primitives.

    Args:
        request: Validated completion create parameters.
        model_id: Bedrock model identifier (used to clamp inference fields).

    Returns:
        ``(inference_cfg, additional_request_fields, bedrock_service_tier,
        openai_service_tier, n, request_metadata)``.
    """
    additional_request_fields: dict[str, Any] = {}
    bedrock_service_tier, openai_service_tier = _openai_common.map_service_tier(
        request.service_tier
    )
    return (
        set_inference_configuration(
            model_id,
            additional_request_fields,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop_sequences=request.stop,
            # frequency_penalty, presence_penalty, logit_bias and seed are
            # accepted and dropped rather than forwarded like the Chat
            # Completions twin does: the text-completion models reject them
            # outright ("extraneous key [frequency_penalty] is not permitted",
            # measured live 2026-07-31), so forwarding them turns a working
            # request into a 400.
            **(request.model_extra or {}),
        ),
        additional_request_fields,
        bedrock_service_tier,
        openai_service_tier,
        request.n or 1,
        (
            {"user": identifier}
            if (identifier := request.safety_identifier or request.user)
            else None
        ),
    )


def format_response(
    completion_id: str,
    created: int,
    model_id: str,
    responses: list[ConverseResponseTypeDef],
    openai_service_tier: ServiceTiers | None,
) -> Completion:
    """Aggregate parallel Bedrock Converse responses into a ``Completion``.

    ``responses`` is a flat list ordered by ``(prompt_i, choice_j)``: the caller
    fans out ``len(prompts) * n`` converses preserving order, so ``enumerate``
    yields the OpenAI-spec global index ``prompt_i * n + choice_j``.

    Args:
        completion_id: Stable identifier for the completion.
        created: Unix timestamp (seconds).
        model_id: Model identifier echoed back to the client.
        responses: Ordered Bedrock responses (``prompt_count * n`` entries).
        openai_service_tier: Echoed service tier.

    Returns:
        Populated ``Completion`` with summed usage and ordered choices.
    """
    choices: list[CompletionChoice] = []
    usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    cached_tokens = 0
    cache_write_tokens = 0
    for index, response in enumerate(responses):
        response_usage = response["usage"]
        # OpenAI semantics: prompt_tokens covers the full prompt, cache buckets included.
        cache_read = response_usage.get("cacheReadInputTokens", 0)
        cache_write = response_usage.get("cacheWriteInputTokens", 0)
        usage.prompt_tokens += response_usage["inputTokens"] + cache_read + cache_write
        usage.completion_tokens += response_usage["outputTokens"]
        cached_tokens += cache_read
        cache_write_tokens += cache_write
        choices.append(
            CompletionChoice(
                text="".join(
                    block["text"]
                    for block in response["output"]["message"]["content"]
                    if "text" in block
                ),
                index=index,
                finish_reason=_map_finish_reason(response.get("stopReason")),
            )
        )
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
    if cached_tokens or cache_write_tokens:
        usage.prompt_tokens_details = PromptTokensDetails(
            cached_tokens=cached_tokens, cache_write_tokens=cache_write_tokens or None
        )
    return Completion(
        id=completion_id,
        created=created,
        model=model_id,
        choices=choices,
        usage=usage,
        service_tier=openai_service_tier,
    )


def _chunk(
    completion_id: str,
    created: int,
    model_id: str,
    text: str,
    index: int,
    finish_reason: CompletionFinishReasonLiteral | None,
    openai_service_tier: ServiceTiers | None,
    usage: CompletionUsage | None,
) -> JSONServerSentEvent:
    """Serialize a single completion streaming chunk as an SSE event.

    Args:
        completion_id: Stable identifier for the completion.
        created: Unix timestamp (seconds).
        model_id: Model identifier echoed back to the client.
        text: Delta text for this chunk (empty for a terminal chunk).
        index: Choice index — the position of the target prompt in the batch.
        finish_reason: Terminal finish reason, or ``None`` for intermediate chunks.
        openai_service_tier: Echoed service tier.
        usage: Aggregated usage totals, populated only on the last terminal chunk
            when ``include_usage`` is ``True``.

    Returns:
        ``JSONServerSentEvent`` wrapping the serialized ``Completion``.
    """
    return JSONServerSentEvent(
        data=Completion(
            id=completion_id,
            created=created,
            model=model_id,
            choices=[
                CompletionChoice(text=text, index=index, finish_reason=finish_reason)
            ],
            usage=usage,
            service_tier=openai_service_tier,
        ).model_dump(mode="json", exclude_none=True)
    )


async def _drain(
    stream: AsyncIterator[ConverseStreamOutputTypeDef],
    index: int,
    queue: Queue[tuple[int, ConverseStreamOutputTypeDef | None]],
) -> None:
    """Forward events from a single Bedrock stream to a shared queue.

    Emits a sentinel ``(index, None)`` on termination (normal or error) so the
    consumer can count completed streams.

    Args:
        stream: Bedrock ``converseStream`` event iterator for a single prompt.
        index: Position of the target prompt in the batch.
        queue: Shared queue consumed by :func:`format_stream`.
    """
    try:
        async for event in stream:
            await queue.put((index, event))
    finally:
        await queue.put((index, None))


async def format_stream(
    completion_id: str,
    created: int,
    model_id: str,
    streams: list[AsyncIterator[ConverseStreamOutputTypeDef]],
    openai_service_tier: ServiceTiers | None,
    *,
    include_usage: bool,
) -> AsyncGenerator[ServerSentEvent]:
    """Stream one or more Bedrock Converse iterators as Completion SSE chunks.

    Each ``contentBlockDelta`` text fragment is yielded immediately as a chunk
    whose ``choices[0].text`` is the DELTA — per OpenAI spec, clients
    concatenate.  When multiple prompts are streamed in parallel, their deltas
    interleave and ``choices[0].index`` identifies the originating prompt.
    Per-prompt ``messageStop`` and ``metadata`` events are buffered; a terminal
    chunk is emitted for every prompt once all streams drain, with the
    aggregated ``usage`` attached to the last one when ``include_usage`` is
    ``True``.  The stream ends with a ``[DONE]`` sentinel.

    If any underlying stream fails (e.g. a mid-stream Bedrock error), the
    first such exception is re-raised once all streams have drained, so the
    caller's monitoring wrapper can log it and emit an error SSE event.

    Args:
        completion_id: Stable identifier for the completion.
        created: Unix timestamp (seconds).
        model_id: Model identifier echoed back to the client.
        streams: Ordered Bedrock stream iterators, one per prompt in the batch.
        openai_service_tier: Echoed service tier.
        include_usage: Populate ``usage`` on the final chunk when ``True``.

    Yields:
        ``JSONServerSentEvent`` chunks, terminated by the ``[DONE]`` sentinel.

    Raises:
        BaseException: The first exception raised by any of the underlying
            streams, if any.
    """
    prompt_count = len(streams)
    finish_reasons: list[CompletionFinishReasonLiteral | None] = [None] * prompt_count
    usage_total = (
        CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        if include_usage
        else None
    )
    queue: Queue[tuple[int, ConverseStreamOutputTypeDef | None]] = Queue()
    tasks = [create_task(_drain(s, i, queue)) for i, s in enumerate(streams)]
    try:
        remaining = prompt_count
        while remaining:
            index, event = await queue.get()
            if event is None:
                remaining -= 1
                continue
            match event:
                case {"contentBlockDelta": {"delta": {"text": text}}} if text:
                    yield _chunk(
                        completion_id,
                        created,
                        model_id,
                        text,
                        index,
                        None,
                        openai_service_tier,
                        None,
                    )
                case {"messageStop": {"stopReason": stop_reason}}:
                    finish_reasons[index] = _map_finish_reason(stop_reason)
                case {"metadata": _} if usage_total is not None:
                    if partial := _openai_common.extract_stream_usage(event):
                        usage_total.prompt_tokens += partial.prompt_tokens
                        usage_total.completion_tokens += partial.completion_tokens
                        usage_total.total_tokens += partial.total_tokens
    finally:
        for task in tasks:
            task.cancel()
        with suppress(CancelledError):
            results = await gather(*tasks, return_exceptions=True)
            raise_first_exception(
                [exc for exc in results if not isinstance(exc, CancelledError)]
            )
    for index in range(prompt_count):
        yield _chunk(
            completion_id,
            created,
            model_id,
            "",
            index,
            finish_reasons[index] or "stop",
            openai_service_tier,
            usage_total if index == prompt_count - 1 else None,
        )
    yield ServerSentEvent(data="[DONE]", event=None)
