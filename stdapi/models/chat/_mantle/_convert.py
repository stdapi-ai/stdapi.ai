"""Wire-format converters for the Bedrock Mantle chat backend.

Builds passthrough payloads for the three Mantle APIs (OpenAI Chat
Completions, OpenAI Responses, Anthropic Messages) and converts requests,
complete responses and SSE streams between those wire shapes when a model
does not natively support the inbound API.

Cross-format conversion always goes through the Chat Completions shape:
``messages <-> responses`` is composed from the two precise ``<-> chat``
converters. Fields without an equivalent in the intermediate shape are
silently dropped (Anthropic thinking blocks and budgets, Responses reasoning
items, Chat Completions audio parts, penalties, logit biases, seeds).

A reasoning model's thinking text is the exception: the Chat Completions
``reasoning_content`` field carries it, and the Responses conversion re-emits
it as a ``reasoning`` output item. The Anthropic Messages shape cannot carry
it, since a ``thinking`` block requires a signature that cannot be produced.
"""

from asyncio import gather
from dataclasses import dataclass, field
from hashlib import sha256
from json import JSONDecodeError, dumps, loads
from time import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sse_starlette import ServerSentEvent

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock_mantle import MantleError, decode_mantle_response_id
from stdapi.input_file import FileIdInputFile, InputFile, prefetch_all_content_types
from stdapi.models.chat._adapters._openai_responses import COMPACTION_CONTENT_PREFIX
from stdapi.types.anthropic_messages import (
    Base64ImageSource,
    Base64PDFSource,
    ContentBlockSourceParam,
    FileSource,
    URLImageSource,
    URLPDFSource,
)
from stdapi.types.openai_chat_completions import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    CompletionUsage,
    File,
)
from stdapi.types.openai_completions import Completion

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Coroutine, Sequence

    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.aws_bedrock_mantle import MantleApi, SseEvent
    from stdapi.types.anthropic_messages import MessageCreateParams
    from stdapi.types.openai_chat_completions import (
        CompletionCreateParams as ChatCompletionCreateParams,
    )
    from stdapi.types.openai_completions import CompletionCreateParams
    from stdapi.types.openai_responses import ResponseCreateParams

#: Default Anthropic ``max_tokens`` injected when a request does not set one.
_DEFAULT_MAX_TOKENS = 4096

#: Assistant field carrying a reasoning model's thinking text upstream.
_REASONING_KEY = "reasoning"

#: Assistant field exposing the thinking text on the Chat Completions surface.
_REASONING_CONTENT_KEY = "reasoning_content"

#: Fields stripped from passthrough Chat Completions payloads (not forwarded upstream).
_CHAT_EXTENSION_FIELDS = (
    "moderation",
    "amazon_bedrock_guardrail_config",
    "amazon-bedrock-guardrailConfig",
    "store",  # Persisted locally; forwarding would create unreachable objects.
)

#: stdapi extension fields stripped from passthrough Responses payloads.
_RESPONSES_EXTENSION_FIELDS = ("moderation",)

#: Anthropic stop reason keyed by Chat Completions finish reason.
_FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}

#: Chat Completions finish reason keyed by Anthropic stop reason.
_STOP_TO_FINISH = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
    "refusal": "content_filter",
}

#: Responses ``incomplete_details.reason`` keyed by Chat Completions finish reason.
_FINISH_TO_INCOMPLETE = {
    "length": "max_output_tokens",
    "content_filter": "content_filter",
}

#: Chat Completions finish reason keyed by Responses ``incomplete_details.reason``.
_INCOMPLETE_TO_FINISH = {
    "max_output_tokens": "length",
    "content_filter": "content_filter",
}

#: Finish reasons natively supported by the legacy text-completion shape.
_TEXT_FINISH_REASONS = frozenset({"stop", "length", "content_filter"})

#: OpenAI request fields copied verbatim between Chat Completions and Responses.
_OPENAI_COMMON_FIELDS = (
    "temperature",
    "top_p",
    "metadata",
    "user",
    "stream",
    # "store" is intentionally excluded: storage is only meaningful on the
    # API that owns the stored object, so conversions must not forward it.
    "service_tier",
    "safety_identifier",
    "prompt_cache_key",
    "prompt_cache_retention",
    "parallel_tool_calls",
)

#: Anthropic ``output_config.effort`` keyed by OpenAI reasoning effort.
_EFFORT_TO_ANTHROPIC = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}

#: Anthropic server tool type prefixes (no Chat Completions equivalent).
_ANTHROPIC_SERVER_TOOL_PREFIXES = (
    "web_search",
    "code_execution",
    "bash_",
    "text_editor_",
    "computer_",
    "memory_",
    "web_fetch_",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _id_token(raw_id: str) -> str:
    """Return the bare token of an upstream identifier, without its prefix.

    Args:
        raw_id: Upstream identifier (``chatcmpl-``, ``resp_`` or ``msg_``).

    Returns:
        The identifier without any known wire-shape prefix.
    """
    return raw_id.removeprefix("chatcmpl-").removeprefix("resp_").removeprefix("msg_")


def _optional_fields(source: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    """Return the non-``None`` *fields* of *source* as a dict.

    Args:
        source: Source payload.
        fields: Field names to extract.

    Returns:
        Mapping of present field names to their values.
    """
    return {name: source[name] for name in fields if source.get(name) is not None}


def _map_service_tier(tier: object) -> str | None:
    """Map ``service_tier`` between the OpenAI and Anthropic shapes.

    Only ``auto`` exists on both sides; every other value is dropped
    (e.g. Anthropic ``standard_only``, OpenAI ``flex``).

    Args:
        tier: Source service tier value.

    Returns:
        ``"auto"`` when mappable, else ``None``.
    """
    return "auto" if tier == "auto" else None


def _json_object(arguments: str | None) -> dict[str, Any]:
    """Parse a tool-arguments JSON string into an object, defaulting to empty.

    Args:
        arguments: JSON string, possibly empty or invalid.

    Returns:
        The parsed JSON object, or ``{}`` when not a valid JSON object.
    """
    if not arguments:
        return {}
    try:
        parsed = loads(arguments)
    except JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _split_data_uri(value: str) -> tuple[str, str] | None:
    """Split a base64 ``data:`` URI into media type and payload.

    Media type parameters (e.g. ``;charset=utf-8``) are stripped.

    Args:
        value: Candidate data URI string.

    Returns:
        Tuple of (bare media type, base64 data), or ``None`` when not a data URI.
    """
    if not value.startswith("data:"):
        return None
    head, separator, data = value.partition(";base64,")
    if not separator:
        return None
    return head[5:].partition(";")[0] or "application/octet-stream", data


def _chat_text(content: object) -> str:
    """Extract plain text from Chat Completions message content.

    Args:
        content: Message content (string or list of content parts).

    Returns:
        Concatenated text (including refusal parts), empty when none.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text") or part.get("refusal") or "" for part in content
        )
    return ""


def _anthropic_text(content: object) -> str:
    """Extract plain text from Anthropic content (string or block list).

    Args:
        content: Anthropic content value.

    Returns:
        Concatenated text of ``text`` blocks, empty when none.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text") or "" for block in content if block.get("type") == "text"
        )
    return ""


def _responses_text(content: object) -> str:
    """Extract plain text from Responses input or output content.

    Args:
        content: Responses content value (string or list of parts).

    Returns:
        Concatenated text of textual parts, empty when none.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text") or ""
            for part in content
            if part.get("type") in ("input_text", "output_text", "text")
        )
    return ""


def _ensure_single_choice(payload: dict[str, Any]) -> None:
    """Reject multi-choice requests for APIs without ``n`` support.

    Args:
        payload: Chat Completions request payload.

    Raises:
        ApiError: When the payload requests more than one choice.
    """
    if (n := payload.get("n")) is not None and n != 1:
        msg = "Multiple choices (n>1) are not supported by this model."
        raise ApiError(msg, status=400)


def rename_reasoning_field(payload: dict[str, Any]) -> dict[str, Any]:
    """Rename ``reasoning`` to ``reasoning_content`` on a Chat Completions payload.

    Covers the non-streaming ``choices[].message`` and the streaming
    ``choices[].delta``. A payload already carrying ``reasoning_content`` is
    left untouched.

    Only a string value is renamed. Anything else stays under its original name
    and is pruned as an unknown field, as it was before this rename existed:
    ``reasoning_content`` is declared as text, so promoting an unexpected shape
    into it would turn a harmless extra field into a validation failure — and,
    mid-stream, into a broken response.

    Args:
        payload: Chat Completions response or chunk dict, modified in place.

    Returns:
        The same payload.
    """
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        for key in ("message", "delta"):
            target = choice.get(key)
            if (
                isinstance(target, dict)
                and isinstance(target.get(_REASONING_KEY), str)
                and _REASONING_CONTENT_KEY not in target
            ):
                target[_REASONING_CONTENT_KEY] = target.pop(_REASONING_KEY)
    return payload


def _finish_from_response(response: dict[str, Any], *, has_tool_calls: bool) -> str:
    """Derive a Chat Completions finish reason from a Responses response.

    Args:
        response: Responses API response object.
        has_tool_calls: Whether function calls were already observed.

    Returns:
        Chat Completions finish reason.
    """
    if has_tool_calls or any(
        item.get("type") == "function_call" for item in response.get("output") or []
    ):
        return "tool_calls"
    if response.get("status") == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        return _INCOMPLETE_TO_FINISH.get(reason or "", "stop")
    return "stop"


# ---------------------------------------------------------------------------
# Usage converters
# ---------------------------------------------------------------------------


def _chat_usage_from_responses(usage: dict[str, Any]) -> dict[str, Any]:
    """Convert a Responses usage block to the Chat Completions shape.

    Args:
        usage: Responses API usage object.

    Returns:
        Chat Completions usage object.
    """
    input_tokens = usage.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens") or input_tokens + output_tokens,
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    }


def _chat_usage_from_messages(usage: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic usage block to the Chat Completions shape.

    Cache write tokens are folded into ``prompt_tokens`` (the Chat
    Completions shape has no separate cache-write bucket).

    Args:
        usage: Anthropic Messages usage object.

    Returns:
        Chat Completions usage object.
    """
    cached = usage.get("cache_read_input_tokens") or 0
    cache_write = usage.get("cache_creation_input_tokens") or 0
    prompt = (usage.get("input_tokens") or 0) + cached + cache_write
    completion = usage.get("output_tokens") or 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }


def _responses_usage_from_chat(usage: dict[str, Any]) -> dict[str, Any]:
    """Convert a Chat Completions usage block to the Responses shape.

    Args:
        usage: Chat Completions usage object.

    Returns:
        Responses API usage object.
    """
    prompt = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    reasoning = (usage.get("completion_tokens_details") or {}).get(
        "reasoning_tokens"
    ) or 0
    return {
        "input_tokens": prompt,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": completion,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": usage.get("total_tokens") or prompt + completion,
    }


def _messages_usage_from_chat(usage: dict[str, Any]) -> dict[str, Any]:
    """Convert a Chat Completions usage block to the Anthropic shape.

    Args:
        usage: Chat Completions usage object.

    Returns:
        Anthropic Messages usage object.
    """
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    prompt = usage.get("prompt_tokens") or 0
    return {
        "input_tokens": max(prompt - cached, 0),
        "output_tokens": usage.get("completion_tokens") or 0,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,
    }


# ---------------------------------------------------------------------------
# Passthrough payload builders
# ---------------------------------------------------------------------------


async def chat_completions_payload(
    request: ChatCompletionCreateParams, model_id: str
) -> dict[str, Any]:
    """Build the passthrough Chat Completions payload for Mantle.

    Non-inline file inputs (URLs, S3 URIs, Files API references) are
    resolved to data URIs or base64, and stdapi extension fields are removed.

    Args:
        request: OpenAI-format completion request.
        model_id: Mantle model identifier to set on the payload.

    Returns:
        JSON-ready request payload.

    Raises:
        ApiError: When the request sets the ``moderation`` parameter.
    """
    _reject_moderation_param(request.moderation)
    await prefetch_all_content_types()
    payload = request.model_dump(mode="json", by_alias=True, exclude_unset=True)
    payload["model"] = model_id
    for tool in payload.get("tools") or ():
        if parameters := (tool.get("function") or {}).get("parameters"):
            sanitize_tool_schema(parameters)
    for name in _CHAT_EXTENSION_FIELDS:
        payload.pop(name, None)
    _restore_reasoning_field(payload)
    await gather(
        *(
            _resolve_chat_message_files(message, dumped)
            for message, dumped in zip(
                request.messages, payload["messages"], strict=True
            )
        )
    )
    return payload


def _restore_reasoning_field(payload: dict[str, Any]) -> None:
    """Send thinking text back under the name the upstream gave it.

    Mirror of :func:`rename_reasoning_field`. A client replaying an assistant
    turn sends back the message it was handed -- the OpenAI SDK idiom is to
    append the whole message object -- so the field has to travel under the
    upstream's own name, not the one this surface exposes.

    Args:
        payload: Outgoing Chat Completions payload, modified in place.
    """
    for message in payload.get("messages") or []:
        if not isinstance(message, dict) or _REASONING_CONTENT_KEY not in message:
            continue
        value = message.pop(_REASONING_CONTENT_KEY)
        if isinstance(value, list):
            # This surface also accepts the text split into parts; upstream takes
            # one string.
            value = "".join(
                part.get("text", "")
                for part in value
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        if isinstance(value, str) and value:
            message.setdefault(_REASONING_KEY, value)


async def _resolve_chat_message_files(message: object, dumped: dict[str, Any]) -> None:
    """Resolve file-backed content parts of one Chat Completions message.

    Args:
        message: Validated message model (source of ``InputFile`` objects).
        dumped: Matching dumped message dict, updated in place.
    """
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return
    await gather(
        *(
            _resolve_chat_part(part, dumped_part)
            for part, dumped_part in zip(
                content, dumped.get("content") or [], strict=True
            )
        )
    )


async def _resolve_chat_part(part: object, dumped: dict[str, Any]) -> None:
    """Resolve one Chat Completions content part into inline content.

    Args:
        part: Validated content part model.
        dumped: Matching dumped content part dict, updated in place.
    """
    match part:
        case ChatCompletionContentPartImageParam():
            dumped["image_url"]["url"] = await part.image_url.url.to_data_uri()
        case ChatCompletionContentPartInputAudioParam():
            dumped["input_audio"]["data"] = await part.input_audio.data.to_base64()
        case File() if source := part.file.file_id or part.file.file_data:
            file_content: dict[str, Any] = {"file_data": await source.to_data_uri()}
            if filename := part.file.filename or await source.get_filename():
                file_content["filename"] = filename
            dumped["file"] = file_content
        case _:
            pass


async def messages_payload(
    request: MessageCreateParams,
    model_id: str,
    *,
    system_message_as_messages: bool = False,
) -> dict[str, Any]:
    """Build the passthrough Anthropic Messages payload for Mantle.

    File-backed image and document sources (URLs, S3 URIs, Files API
    references) are inlined as base64 sources, inline ``system``-role
    messages are folded into the ``system`` field, and the
    ``anthropic_version`` body field is dropped (Mantle takes the version as
    an HTTP header).

    Args:
        request: Anthropic-format message creation request.
        model_id: Mantle model identifier to set on the payload.
        system_message_as_messages: When True, mid-conversation ``system``-role
            messages the model handles natively are forwarded as-is; every
            other one is folded into the ``system`` field, which is also the
            only behavior when False.

    Returns:
        JSON-ready request payload.
    """
    await prefetch_all_content_types()
    payload = request.model_dump(mode="json", by_alias=True, exclude_unset=True)
    payload["model"] = model_id
    for tool in payload.get("tools") or ():
        if schema := tool.get("input_schema"):
            sanitize_tool_schema(schema)
    payload.pop("anthropic_version", None)
    if payload.get("max_tokens") is None:
        payload["max_tokens"] = _DEFAULT_MAX_TOKENS
    await gather(
        *(
            _resolve_anthropic_blocks(message.content, dumped["content"])
            for message, dumped in zip(
                request.messages, payload["messages"], strict=True
            )
            if isinstance(message.content, list)
            and isinstance(dumped.get("content"), list)
        )
    )
    _fold_inline_system(payload, forward_native=system_message_as_messages)
    return payload


async def _resolve_anthropic_blocks(
    blocks: Sequence[object], dumped_blocks: list[dict[str, Any]]
) -> None:
    """Resolve file-backed sources in a list of Anthropic content blocks.

    Recurses into nested block lists (``tool_result`` and content-source
    documents).

    Args:
        blocks: Validated content block models.
        dumped_blocks: Matching dumped block dicts, updated in place.
    """
    coroutines: list[Coroutine[Any, Any, None]] = []
    for block, dumped in zip(blocks, dumped_blocks, strict=True):
        source = getattr(block, "source", None)
        if source is not None and isinstance(dumped.get("source"), dict):
            coroutines.append(_resolve_anthropic_source(source, dumped["source"]))
        content = getattr(block, "content", None)
        if isinstance(content, list) and isinstance(dumped.get("content"), list):
            coroutines.append(_resolve_anthropic_blocks(content, dumped["content"]))
    await gather(*coroutines)


async def _resolve_anthropic_source(source: object, dumped: dict[str, Any]) -> None:
    """Resolve one Anthropic image or document source into a base64 source.

    Args:
        source: Validated source model.
        dumped: Matching dumped source dict, updated in place.
    """
    match source:
        case Base64ImageSource() | Base64PDFSource():
            dumped["data"] = await source.data.to_base64()
        case URLImageSource() | URLPDFSource():
            await _inline_source(source.url, dumped)
        case FileSource():
            await _inline_source(source.file_id, dumped)
        case ContentBlockSourceParam() if isinstance(
            source.content, list
        ) and isinstance(dumped.get("content"), list):
            await _resolve_anthropic_blocks(source.content, dumped["content"])
        case _:
            pass


async def _inline_source(file: InputFile, dumped: dict[str, Any]) -> None:
    """Replace a dumped Anthropic source with an inline base64 source.

    Args:
        file: File to inline.
        dumped: Dumped source dict, replaced in place.
    """
    media_type = await file.get_content_type()
    data = await file.to_base64()
    dumped.clear()
    dumped.update({"type": "base64", "media_type": media_type, "data": data})


def _is_native_system_placement(messages: list[dict[str, Any]], index: int) -> bool:
    """Whether the ``system``-role message at *index* is placed where models accept it.

    Models supporting mid-conversation system messages require them to follow a
    user turn and to either precede an assistant turn or end the message list;
    consecutive system messages are accepted as a single section.

    Args:
        messages: Full message list of the payload.
        index: Index of the system-role message in *messages*.

    Returns:
        True when the message sits in an accepted placement.
    """
    before = index - 1
    while before >= 0 and messages[before].get("role") == "system":
        before -= 1
    after = index + 1
    while after < len(messages) and messages[after].get("role") == "system":
        after += 1
    return (
        before >= 0
        and messages[before].get("role") == "user"
        and (after == len(messages) or messages[after].get("role") == "assistant")
    )


def _fold_inline_system(
    payload: dict[str, Any], *, forward_native: bool = False
) -> None:
    """Fold inline ``system``-role messages into the ``system`` field.

    Args:
        payload: Anthropic Messages request payload, updated in place.
        forward_native: When True, system-role messages in a placement the model
            accepts natively are left in the message list.
    """
    messages = payload.get("messages") or []
    kept: list[dict[str, Any]] = []
    inline: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "system" or (
            forward_native and _is_native_system_placement(messages, index)
        ):
            kept.append(message)
        else:
            inline.append(message)
    if not inline:
        return
    payload["messages"] = kept
    texts = [_anthropic_text(payload.get("system"))]
    texts += [_anthropic_text(message.get("content")) for message in inline]
    payload["system"] = "\n\n".join(text for text in texts if text)


async def responses_payload(
    request: ResponseCreateParams, model_id: str
) -> tuple[dict[str, Any], RegionName | None]:
    """Build the passthrough Responses payload for Mantle.

    File inputs (``input_image`` URLs or file IDs, ``input_file`` URLs or
    file IDs) are inlined as data URIs, stdapi extension fields are removed,
    and a region-tagged ``previous_response_id`` is decoded back to its
    native Mantle identifier.

    Args:
        request: Responses API creation request.
        model_id: Mantle model identifier to set on the payload.

    Returns:
        Tuple of (JSON-ready request payload, pinned region or ``None``).

    Raises:
        ApiError: When the request sets the ``moderation`` parameter, when
            ``previous_response_id`` is not a Mantle-tagged ID, or when the
            input contains a locally-produced ``compaction`` item.
    """
    _reject_moderation_param(request.moderation)
    payload = request.model_dump(mode="json", by_alias=True, exclude_unset=True)
    payload["model"] = model_id
    for name in _RESPONSES_EXTENSION_FIELDS:
        payload.pop(name, None)
    _reject_local_compaction_items(payload.get("input"))
    region = _pin_previous_response(payload)
    for tool in payload.get("tools") or ():
        # Mantle's server-side web_search only runs in cache-only mode; live
        # web access always fails upstream, so it is forced off.
        if str(tool.get("type", "")).startswith("web_search"):
            tool["external_web_access"] = False
        if parameters := tool.get("parameters"):
            sanitize_tool_schema(parameters)
    if pairs := _collect_responses_files(payload.get("input")):
        await prefetch_all_content_types()
        await gather(*(_apply_responses_file(part, file) for part, file in pairs))
    return payload, region


def _reject_moderation_param(moderation: object) -> None:
    """Reject the stdapi ``moderation`` extension on Mantle passthrough requests.

    Mantle passthrough requests bypass the Bedrock Converse API, so
    moderation guardrails cannot be applied or reported.

    Args:
        moderation: The request ``moderation`` field value, if any.

    Raises:
        ApiError: When the field is set.
    """
    if moderation is not None:
        msg = (
            "The 'moderation' parameter is not available with this model. "
            "Remove the parameter, or use a model that supports moderation."
        )
        raise ApiError(msg, status=400)


def _reject_local_compaction_items(input_value: object) -> None:
    """Reject ``compaction`` items carrying this server's local marker.

    Locally-produced compaction content is Bedrock-specific encoding that the
    Mantle upstream cannot decrypt; unmarked items (produced by the upstream
    itself) pass through verbatim.

    Args:
        input_value: Dumped ``input`` payload value.

    Raises:
        ApiError: When a locally-produced compaction item is present.
    """
    if not isinstance(input_value, list):
        return
    for item in input_value:
        if (
            isinstance(item, dict)
            and item.get("type") == "compaction"
            and str(item.get("encrypted_content") or "").startswith(
                COMPACTION_CONTENT_PREFIX
            )
        ):
            msg = (
                "This compaction item is not compatible with the selected "
                "model. Compact the conversation again, or select a "
                "different model."
            )
            raise ApiError(msg, status=400)


def _pin_previous_response(payload: dict[str, Any]) -> RegionName | None:
    """Decode a region-tagged ``previous_response_id`` in place.

    A falsy value (absent, ``None``, empty) is removed from the payload so
    it is never forwarded upstream as an explicit ``null``.

    Args:
        payload: Responses request payload, updated in place.

    Returns:
        The region pinned by the previous response, or ``None``.

    Raises:
        ApiError: When the ID is not a region-tagged Mantle response ID.
    """
    if not (previous_id := payload.pop("previous_response_id", None)):
        return None
    if (decoded := decode_mantle_response_id(previous_id)) is None:
        msg = (
            "Unknown previous_response_id: only responses created by this "
            "server can be chained."
        )
        raise ApiError(msg, status=400)
    region, native_id = decoded
    payload["previous_response_id"] = native_id
    return region


def _collect_responses_files(
    input_value: object,
) -> list[tuple[dict[str, Any], InputFile]]:
    """Collect file-backed Responses input parts and their file sources.

    Args:
        input_value: Dumped ``input`` payload value.

    Returns:
        List of (dumped content part, file to inline) pairs.
    """
    pairs: list[tuple[dict[str, Any], InputFile]] = []
    if not isinstance(input_value, list):
        return pairs
    for item in input_value:
        if not isinstance(item, dict) or not isinstance(
            content := item.get("content"), list
        ):
            continue
        pairs += [
            (part, file)
            for part in content
            if isinstance(part, dict) and (file := _responses_part_file(part))
        ]
    return pairs


def _responses_part_file(part: dict[str, Any]) -> InputFile | None:
    """Build the file source referenced by a Responses input part, if any.

    File ID and URL reference keys are removed from the part; the file
    content replaces them once resolved.

    Args:
        part: Dumped input content part, updated in place.

    Returns:
        The referenced file, or ``None`` for inline or non-file parts.
    """
    match part.get("type"):
        case "input_image":
            if file_id := part.pop("file_id", None):
                return FileIdInputFile(file_id)
            if (url := part.get("image_url")) and not url.startswith("data:"):
                return InputFile(url)
        case "input_file":
            if file_id := part.pop("file_id", None):
                return FileIdInputFile(file_id)
            if url := part.pop("file_url", None):
                return InputFile(url)
        case _:
            pass
    return None


async def _apply_responses_file(part: dict[str, Any], file: InputFile) -> None:
    """Inline a resolved file into a Responses input part.

    Args:
        part: Dumped input content part, updated in place.
        file: File to inline.
    """
    if part.get("type") == "input_image":
        part["image_url"] = await file.to_data_uri()
        return
    part["file_data"] = await file.to_data_uri()
    if not part.get("filename"):
        part["filename"] = await file.get_filename() or "file"


def enable_stream_usage(api: MantleApi, payload: dict[str, Any]) -> dict[str, Any]:
    """Force streaming with usage reporting on an upstream payload.

    Args:
        api: Upstream Mantle API the payload targets.
        payload: Upstream request payload, updated in place.

    Returns:
        The updated payload.
    """
    payload["stream"] = True
    if api == "chat_completions":
        payload["stream_options"] = {
            **(payload.get("stream_options") or {}),
            "include_usage": True,
        }
    return payload


# ---------------------------------------------------------------------------
# Request payload conversion
# ---------------------------------------------------------------------------


def _chat_to_responses_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a Chat Completions request payload to the Responses shape.

    Args:
        payload: Chat Completions request payload.

    Returns:
        Responses API request payload.

    Raises:
        ApiError: When the payload requests more than one choice.
    """
    _ensure_single_choice(payload)
    instructions, items = _chat_messages_to_input(payload.get("messages") or [])
    out: dict[str, Any] = {"model": payload.get("model"), "input": items}
    if instructions:
        out["instructions"] = instructions
    out.update(_optional_fields(payload, _OPENAI_COMMON_FIELDS))
    if tokens := payload.get("max_completion_tokens") or payload.get("max_tokens"):
        out["max_output_tokens"] = tokens
    if effort := payload.get("reasoning_effort"):
        out["reasoning"] = {"effort": effort}
    if text_format := _text_format_from_response_format(payload.get("response_format")):
        out["text"] = {"format": text_format}
    if tools := _responses_tools_from_chat(payload.get("tools") or []):
        out["tools"] = tools
    if (
        choice := _responses_tool_choice_from_chat(payload.get("tool_choice"))
    ) is not None:
        out["tool_choice"] = choice
    return out


def _chat_messages_to_input(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert Chat Completions messages to Responses instructions and input.

    Args:
        messages: Chat Completions message dicts.

    Returns:
        Tuple of (instructions text, Responses input items).
    """
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for message in messages:
        match message.get("role"):
            case "system" | "developer":
                instructions.append(_chat_text(message.get("content")))
            case "user":
                content = message.get("content")
                items.append(
                    {
                        "role": "user",
                        "content": content
                        if isinstance(content, str)
                        else _input_parts_from_chat(content or []),
                    }
                )
            case "assistant":
                items += _input_items_from_chat_assistant(message)
            case "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.get("tool_call_id") or "",
                        "output": _chat_text(message.get("content")),
                    }
                )
            case _:
                pass
    return "\n\n".join(text for text in instructions if text), items


def _input_parts_from_chat(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Chat Completions content parts to Responses input parts.

    Audio parts have no Responses equivalent and are dropped.

    Args:
        parts: Chat Completions content part dicts.

    Returns:
        Responses input content parts.
    """
    converted: list[dict[str, Any]] = []
    for part in parts:
        match part.get("type"):
            case "text":
                converted.append({"type": "input_text", "text": part.get("text") or ""})
            case "image_url":
                image = part.get("image_url") or {}
                image_part: dict[str, Any] = {
                    "type": "input_image",
                    "image_url": image.get("url") or "",
                }
                if detail := image.get("detail"):
                    image_part["detail"] = detail
                converted.append(image_part)
            case "file":
                file = part.get("file") or {}
                if file.get("file_data"):
                    converted.append(
                        {
                            "type": "input_file",
                            **_optional_fields(file, ("file_data", "filename")),
                        }
                    )
            case _:
                pass
    return converted


def _input_items_from_chat_assistant(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one Chat Completions assistant message to Responses input items.

    Args:
        message: Assistant message dict.

    Returns:
        Responses input items (assistant message and/or function calls).
    """
    items: list[dict[str, Any]] = []
    if text := _chat_text(message.get("content")):
        items.append({"role": "assistant", "content": text})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        items.append(
            {
                "type": "function_call",
                "call_id": call.get("id") or "",
                "name": function.get("name") or "",
                "arguments": function.get("arguments") or "{}",
            }
        )
    return items


def _responses_tools_from_chat(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Chat Completions tool definitions to the Responses shape.

    Args:
        tools: Chat Completions tool definition dicts.

    Returns:
        Responses tool definitions (non-function tools are dropped).
    """
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function = tool.get("function") or {}
        entry: dict[str, Any] = {"type": "function", "name": function.get("name") or ""}
        entry.update(
            _optional_fields(function, ("description", "parameters", "strict"))
        )
        converted.append(entry)
    return converted


def _responses_tool_choice_from_chat(
    tool_choice: object,
) -> str | dict[str, Any] | None:
    """Convert a Chat Completions tool choice to the Responses shape.

    Args:
        tool_choice: Chat Completions tool choice value.

    Returns:
        Responses tool choice value, or ``None`` when unmappable.
    """
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = (tool_choice.get("function") or {}).get("name") or ""
        return {"type": "function", "name": name}
    return None


def _text_format_from_response_format(
    response_format: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Convert a Chat Completions ``response_format`` to a Responses text format.

    Args:
        response_format: Chat Completions response format dict.

    Returns:
        Responses ``text.format`` value, or ``None`` when unmappable.
    """
    match (response_format or {}).get("type"):
        case "json_object":
            return {"type": "json_object"}
        case "json_schema":
            schema = (response_format or {}).get("json_schema") or {}
            text_format: dict[str, Any] = {
                "type": "json_schema",
                "name": schema.get("name") or "response",
                "schema": schema.get("schema") or {},
            }
            text_format.update(_optional_fields(schema, ("strict", "description")))
            return text_format
        case _:
            return None


def _responses_to_chat_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a Responses request payload to the Chat Completions shape.

    ``previous_response_id`` and Responses-only options are dropped.

    Args:
        payload: Responses API request payload.

    Returns:
        Chat Completions request payload.
    """
    messages: list[dict[str, Any]] = []
    if isinstance(instructions := payload.get("instructions"), str) and instructions:
        messages.append({"role": "system", "content": instructions})
    input_value = payload.get("input")
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
    else:
        for item in input_value or []:
            for message in _chat_messages_from_input_item(item):
                _append_chat_message(messages, message)
    out: dict[str, Any] = {"model": payload.get("model"), "messages": messages}
    out.update(_optional_fields(payload, _OPENAI_COMMON_FIELDS))
    if tokens := payload.get("max_output_tokens"):
        out["max_completion_tokens"] = tokens
    if effort := (payload.get("reasoning") or {}).get("effort"):
        out["reasoning_effort"] = effort
    if response_format := _response_format_from_text(payload.get("text")):
        out["response_format"] = response_format
    if tools := _chat_tools_from_responses(payload.get("tools") or []):
        out["tools"] = tools
    if (
        choice := _chat_tool_choice_from_responses(payload.get("tool_choice"))
    ) is not None:
        out["tool_choice"] = choice
    return out


def _append_chat_message(
    messages: list[dict[str, Any]], message: dict[str, Any]
) -> None:
    """Append a Chat Completions message, merging parallel tool calls.

    Chat Completions requires an assistant ``tool_calls`` message to be
    followed by one tool message per call, so the ``function_call`` items of
    a same turn must share a single assistant message.

    Args:
        messages: Chat Completions messages, updated in place.
        message: Message to append.
    """
    previous = messages[-1] if messages else {}
    if (
        message.get("tool_calls")
        and previous.get("role") == "assistant"
        and previous.get("tool_calls")
    ):
        previous["tool_calls"] += message["tool_calls"]
    else:
        messages.append(message)


def _chat_messages_from_input_item(item: object) -> list[dict[str, Any]]:
    """Convert one Responses input item to Chat Completions messages.

    Reasoning items and unsupported tool item types are dropped.

    Args:
        item: Responses input item dict.

    Returns:
        Chat Completions message dicts.
    """
    if not isinstance(item, dict):
        return []
    match item.get("type") or ("message" if "role" in item else ""):
        case "message":
            return _chat_messages_from_input_message(item)
        case "function_call":
            return [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or "",
                            "type": "function",
                            "function": {
                                "name": item.get("name") or "",
                                "arguments": item.get("arguments") or "{}",
                            },
                        }
                    ],
                }
            ]
        case "function_call_output":
            return [
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or "",
                    "content": _responses_text(item.get("output")),
                }
            ]
        case _:
            return []


def _chat_messages_from_input_message(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one Responses message input item to Chat Completions messages.

    Args:
        item: Responses message input item dict.

    Returns:
        Chat Completions message dicts.
    """
    role = item.get("role") or "user"
    content = item.get("content")
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if role in ("system", "developer", "assistant"):
        text = _responses_text(content)
        return [{"role": role, "content": text}] if text else []
    parts = [part for raw in content or [] if (part := _chat_part_from_input(raw))]
    return [{"role": "user", "content": parts}] if parts else []


def _chat_part_from_input(part: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one Responses input content part to a Chat Completions part.

    Args:
        part: Responses input content part dict.

    Returns:
        Chat Completions content part, or ``None`` when unmappable.
    """
    match part.get("type"):
        case "input_text" | "output_text":
            return {"type": "text", "text": part.get("text") or ""}
        case "input_image":
            image: dict[str, Any] = {"url": part.get("image_url") or ""}
            if (detail := part.get("detail")) in ("low", "high", "auto"):
                image["detail"] = detail
            return {"type": "image_url", "image_url": image}
        case "input_file" if part.get("file_data"):
            return {
                "type": "file",
                "file": _optional_fields(part, ("file_data", "filename")),
            }
        case _:
            return None


def _chat_tools_from_responses(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Responses tool definitions to the Chat Completions shape.

    Args:
        tools: Responses tool definition dicts.

    Returns:
        Chat Completions tool definitions (non-function tools are dropped).
    """
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function: dict[str, Any] = {"name": tool.get("name") or ""}
        function.update(_optional_fields(tool, ("description", "parameters", "strict")))
        converted.append({"type": "function", "function": function})
    return converted


def _chat_tool_choice_from_responses(
    tool_choice: object,
) -> str | dict[str, Any] | None:
    """Convert a Responses tool choice to the Chat Completions shape.

    Args:
        tool_choice: Responses tool choice value.

    Returns:
        Chat Completions tool choice value, or ``None`` when unmappable.
    """
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {"type": "function", "function": {"name": tool_choice.get("name") or ""}}
    return None


def _response_format_from_text(
    text_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Convert a Responses ``text.format`` to a Chat Completions format.

    Args:
        text_config: Responses ``text`` configuration dict.

    Returns:
        Chat Completions ``response_format`` value, or ``None``.
    """
    text_format = (text_config or {}).get("format") or {}
    match text_format.get("type"):
        case "json_object":
            return {"type": "json_object"}
        case "json_schema":
            schema: dict[str, Any] = {
                "name": text_format.get("name") or "response",
                "schema": text_format.get("schema") or {},
            }
            schema.update(_optional_fields(text_format, ("strict", "description")))
            return {"type": "json_schema", "json_schema": schema}
        case _:
            return None


def _chat_to_messages_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a Chat Completions request payload to the Anthropic shape.

    Penalties, ``response_format`` (rejected by Mantle Messages as
    ``output_config.format``) and other unmappable options are dropped;
    the temperature is clamped to the Anthropic 0-1 range.

    Args:
        payload: Chat Completions request payload.

    Returns:
        Anthropic Messages request payload.

    Raises:
        ApiError: When the payload requests more than one choice.
    """
    _ensure_single_choice(payload)
    system, turns = _anthropic_messages_from_chat(payload.get("messages") or [])
    out: dict[str, Any] = {
        "model": payload.get("model"),
        "max_tokens": payload.get("max_completion_tokens")
        or payload.get("max_tokens")
        or _DEFAULT_MAX_TOKENS,
        "messages": turns,
    }
    if system:
        out["system"] = system
    if (temperature := payload.get("temperature")) is not None:
        out["temperature"] = min(float(temperature), 1.0)
    out.update(_optional_fields(payload, ("top_p", "stream")))
    if tier := _map_service_tier(payload.get("service_tier")):
        out["service_tier"] = tier
    if effort := _EFFORT_TO_ANTHROPIC.get(str(payload.get("reasoning_effort"))):
        out["output_config"] = {"effort": effort}
    if stop := payload.get("stop"):
        out["stop_sequences"] = [stop] if isinstance(stop, str) else stop
    if user := payload.get("user"):
        out["metadata"] = {"user_id": user}
    if tools := _anthropic_tools_from_chat(payload.get("tools") or []):
        out["tools"] = tools
    parallel = payload.get("parallel_tool_calls") if tools else None
    if (
        choice := _anthropic_tool_choice_from_chat(payload.get("tool_choice"), parallel)
    ) is not None:
        out["tool_choice"] = choice
    return out


def _anthropic_messages_from_chat(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert Chat Completions messages to Anthropic system and turns.

    Args:
        messages: Chat Completions message dicts.

    Returns:
        Tuple of (system prompt text, Anthropic message turns).
    """
    system_parts: list[str] = []
    turns: list[dict[str, Any]] = []
    for message in messages:
        match message.get("role"):
            case "system" | "developer":
                system_parts.append(_chat_text(message.get("content")))
            case "user":
                _append_anthropic_turn(
                    turns,
                    "user",
                    _anthropic_blocks_from_chat_content(message.get("content")),
                )
            case "assistant":
                _append_anthropic_turn(
                    turns, "assistant", _anthropic_blocks_from_chat_assistant(message)
                )
            case "tool":
                _append_anthropic_turn(
                    turns,
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id") or "",
                            "content": _chat_text(message.get("content")),
                        }
                    ],
                )
            case _:
                pass
    return "\n\n".join(text for text in system_parts if text), turns


def _append_anthropic_turn(
    turns: list[dict[str, Any]], role: str, blocks: list[dict[str, Any]]
) -> None:
    """Append content blocks to the turn list, merging same-role turns.

    Args:
        turns: Anthropic message turns, updated in place.
        role: Turn role.
        blocks: Content blocks to append.
    """
    if not blocks:
        return
    if turns and turns[-1]["role"] == role:
        turns[-1]["content"] += blocks
    else:
        turns.append({"role": role, "content": blocks})


def _anthropic_blocks_from_chat_content(content: object) -> list[dict[str, Any]]:
    """Convert Chat Completions message content to Anthropic content blocks.

    Audio parts have no Anthropic equivalent and are dropped.

    Args:
        content: Chat Completions message content.

    Returns:
        Anthropic content blocks.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        return []
    return [
        block for part in content if (block := _anthropic_block_from_chat_part(part))
    ]


def _anthropic_block_from_chat_part(part: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one Chat Completions content part to an Anthropic block.

    Args:
        part: Chat Completions content part dict.

    Returns:
        Anthropic content block, or ``None`` when unmappable.
    """
    match part.get("type"):
        case "text" | "refusal":
            return {
                "type": "text",
                "text": part.get("text") or part.get("refusal") or "",
            }
        case "image_url":
            return _anthropic_image_block(
                (part.get("image_url") or {}).get("url") or ""
            )
        case "file":
            return _anthropic_document_block(
                (part.get("file") or {}).get("file_data") or ""
            )
        case _:
            return None


def _anthropic_image_block(url: str) -> dict[str, Any] | None:
    """Build an Anthropic image block from an image URL or data URI.

    Args:
        url: Image URL or data URI.

    Returns:
        Anthropic image block, or ``None`` when *url* is empty.
    """
    if parsed := _split_data_uri(url):
        media_type, data = parsed
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    if url:
        return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def _anthropic_document_block(file_data: str) -> dict[str, Any] | None:
    """Build an Anthropic document block from an inline file data URI.

    Args:
        file_data: File content as a data URI.

    Returns:
        Anthropic document block, or ``None`` when not a data URI.
    """
    if parsed := _split_data_uri(file_data):
        media_type, data = parsed
        return {
            "type": "document",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return None


def _anthropic_blocks_from_chat_assistant(
    message: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert one Chat Completions assistant message to Anthropic blocks.

    Args:
        message: Assistant message dict.

    Returns:
        Anthropic content blocks (text followed by ``tool_use`` blocks).
    """
    blocks = _anthropic_blocks_from_chat_content(message.get("content"))
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or "",
                "name": function.get("name") or "",
                "input": _json_object(function.get("arguments")),
            }
        )
    return blocks


def _anthropic_tools_from_chat(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Chat Completions tool definitions to the Anthropic shape.

    Args:
        tools: Chat Completions tool definition dicts.

    Returns:
        Anthropic tool definitions (non-function tools are dropped).
    """
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function = tool.get("function") or {}
        entry: dict[str, Any] = {
            "name": function.get("name") or "",
            "input_schema": function.get("parameters") or {"type": "object"},
        }
        if description := function.get("description"):
            entry["description"] = description
        converted.append(entry)
    return converted


def _anthropic_tool_choice_from_chat(
    tool_choice: str | dict[str, Any] | None, parallel_tool_calls: object
) -> dict[str, Any] | None:
    """Convert a Chat Completions tool choice to the Anthropic shape.

    Anthropic carries the parallel-call switch on the tool choice itself, so
    ``parallel_tool_calls: false`` adds ``disable_parallel_tool_use`` (and
    synthesises the default ``auto`` choice when none was requested).

    Args:
        tool_choice: Chat Completions tool choice value.
        parallel_tool_calls: Chat Completions ``parallel_tool_calls`` value.

    Returns:
        Anthropic tool choice value, or ``None`` when unmappable.
    """
    disabled = parallel_tool_calls is False
    match tool_choice:
        case "auto":
            choice: dict[str, Any] = {"type": "auto"}
        case "required":
            choice = {"type": "any"}
        case "none":
            return {"type": "none"}
        case {"type": "function", "function": dict() as function_choice}:
            choice = {"type": "tool", "name": function_choice.get("name") or ""}
        case {"type": "function"}:
            choice = {"type": "tool", "name": ""}
        case None if disabled:
            choice = {"type": "auto"}
        case _:
            return None
    if disabled:
        choice["disable_parallel_tool_use"] = True
    return choice


def _messages_to_chat_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic request payload to the Chat Completions shape.

    Thinking configuration, ``top_k`` and cache controls are dropped.

    Args:
        payload: Anthropic Messages request payload.

    Returns:
        Chat Completions request payload.
    """
    messages: list[dict[str, Any]] = []
    if system_text := _anthropic_text(payload.get("system")):
        messages.append({"role": "system", "content": system_text})
    for turn in payload.get("messages") or []:
        messages += _chat_messages_from_anthropic_turn(turn)
    out: dict[str, Any] = {"model": payload.get("model"), "messages": messages}
    if tokens := payload.get("max_tokens"):
        out["max_completion_tokens"] = tokens
    out.update(_optional_fields(payload, ("temperature", "top_p", "stream")))
    if tier := _map_service_tier(payload.get("service_tier")):
        out["service_tier"] = tier
    out.update(_chat_fields_from_output_config(payload.get("output_config") or {}))
    if stop := payload.get("stop_sequences"):
        out["stop"] = stop
    if user := (payload.get("metadata") or {}).get("user_id"):
        out["user"] = _openai_user(user)
    if tools := _chat_tools_from_anthropic(payload.get("tools") or []):
        out["tools"] = tools
    tool_choice = payload.get("tool_choice")
    if (choice := _chat_tool_choice_from_anthropic(tool_choice)) is not None:
        out["tool_choice"] = choice
    if isinstance(tool_choice, dict) and tool_choice.get("disable_parallel_tool_use"):
        out["parallel_tool_calls"] = False
    return out


def _chat_fields_from_output_config(output_config: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic ``output_config`` to Chat Completions request fields.

    Args:
        output_config: Anthropic ``output_config`` value.

    Returns:
        ``reasoning_effort`` and/or ``response_format`` fields, when mappable.
    """
    fields: dict[str, Any] = {}
    if effort := output_config.get("effort"):
        fields["reasoning_effort"] = effort
    output_format = output_config.get("format") or {}
    if output_format.get("type") == "json_schema":
        fields["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": output_format.get("schema") or {},
            },
        }
    return fields


def _chat_messages_from_anthropic_turn(turn: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one Anthropic message turn to Chat Completions messages.

    Thinking blocks are dropped; ``tool_result`` blocks become ``tool``-role
    messages emitted before the remaining turn content.

    Args:
        turn: Anthropic message turn dict.

    Returns:
        Chat Completions message dicts.
    """
    role = turn.get("role") or "user"
    content = turn.get("content")
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    tool_messages: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content or []:
        match block.get("type"):
            case "text":
                parts.append({"type": "text", "text": block.get("text") or ""})
            case "image" if part := _chat_part_from_anthropic_image(
                block.get("source") or {}
            ):
                parts.append(part)
            case "document" if part := _chat_part_from_anthropic_document(
                block.get("source") or {}
            ):
                parts.append(part)
            case "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or "",
                        "type": "function",
                        "function": {
                            "name": block.get("name") or "",
                            "arguments": dumps(block.get("input") or {}),
                        },
                    }
                )
            case "tool_result":
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id") or "",
                        "content": _anthropic_text(block.get("content")),
                    }
                )
            case _:
                pass
    return tool_messages + _assemble_chat_message(role, parts, tool_calls)


def _assemble_chat_message(
    role: str, parts: list[dict[str, Any]], tool_calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Assemble a Chat Completions message from converted Anthropic blocks.

    Args:
        role: Message role.
        parts: Converted content parts.
        tool_calls: Converted tool calls.

    Returns:
        A single-message list, or an empty list when there is no content.
    """
    if not parts and not tool_calls:
        return []
    if role != "assistant":
        return [{"role": role, "content": parts}]
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(part["text"] for part in parts if part.get("type") == "text")
        or None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return [message]


def _chat_part_from_anthropic_image(source: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an Anthropic image source to a Chat Completions image part.

    Args:
        source: Anthropic image source dict.

    Returns:
        Chat Completions ``image_url`` part, or ``None`` when unmappable.
    """
    match source.get("type"):
        case "base64":
            media_type = source.get("media_type") or "image/png"
            url = f"data:{media_type};base64,{source.get('data') or ''}"
            return {"type": "image_url", "image_url": {"url": url}}
        case "url":
            return {"type": "image_url", "image_url": {"url": source.get("url") or ""}}
        case _:
            return None


def _chat_part_from_anthropic_document(source: dict[str, Any]) -> dict[str, Any] | None:
    """Convert an Anthropic document source to a Chat Completions part.

    URL document sources are dropped (payload builders inline them first).

    Args:
        source: Anthropic document source dict.

    Returns:
        Chat Completions ``file`` or ``text`` part, or ``None``.
    """
    match source.get("type"):
        case "base64":
            media_type = source.get("media_type") or "application/pdf"
            file_data = f"data:{media_type};base64,{source.get('data') or ''}"
            return {"type": "file", "file": {"file_data": file_data}}
        case "text":
            return {"type": "text", "text": source.get("data") or ""}
        case _:
            return None


def _openai_user(user_id: str) -> str:
    """Adapt an Anthropic ``metadata.user_id`` to the OpenAI ``user`` field.

    OpenAI caps ``user`` at 64 characters while Anthropic allows longer IDs;
    over-long values are replaced by their SHA-256 hex digest (64 characters,
    deterministic per original ID).

    Args:
        user_id: Anthropic user identifier.

    Returns:
        A value acceptable for the OpenAI ``user`` field.
    """
    if len(user_id) <= 64:
        return user_id
    return sha256(user_id.encode()).hexdigest()


#: JSON Schema keywords that silently break Mantle upstream tool templating.
_UNSUPPORTED_SCHEMA_KEYWORDS = ("propertyNames",)


def sanitize_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip JSON Schema keywords that Mantle upstream models cannot handle.

    Some open-weight tool templates silently emit an empty generation when a
    tool schema contains rarely-supported keywords (observed live with
    ``propertyNames`` on Gemma). The keywords are removed recursively.

    Args:
        schema: Tool input JSON schema (mutated in place).

    Returns:
        The sanitized schema.
    """
    if isinstance(schema, dict):
        for keyword in _UNSUPPORTED_SCHEMA_KEYWORDS:
            schema.pop(keyword, None)
        for value in schema.values():
            if isinstance(value, dict):
                sanitize_tool_schema(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        sanitize_tool_schema(item)
    return schema


def _chat_tools_from_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions to the Chat Completions shape.

    Args:
        tools: Anthropic tool definition dicts.

    Returns:
        Chat Completions tool definitions.

    Raises:
        ApiError: When the request includes an Anthropic server tool.
    """
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if str(tool.get("type") or "").startswith(_ANTHROPIC_SERVER_TOOL_PREFIXES):
            msg = "Anthropic server tools are not supported for this model."
            raise ApiError(msg, status=400)
        if not tool.get("name") or "input_schema" not in tool:
            continue
        function: dict[str, Any] = {
            "name": tool["name"],
            "parameters": sanitize_tool_schema(
                tool.get("input_schema") or {"type": "object"}
            ),
        }
        if description := tool.get("description"):
            function["description"] = description
        converted.append({"type": "function", "function": function})
    return converted


def _chat_tool_choice_from_anthropic(
    tool_choice: object,
) -> str | dict[str, Any] | None:
    """Convert an Anthropic tool choice to the Chat Completions shape.

    Args:
        tool_choice: Anthropic tool choice value.

    Returns:
        Chat Completions tool choice value, or ``None`` when unmappable.
    """
    if not isinstance(tool_choice, dict):
        return None
    match tool_choice.get("type"):
        case "auto":
            return "auto"
        case "any":
            return "required"
        case "none":
            return "none"
        case "tool":
            return {
                "type": "function",
                "function": {"name": tool_choice.get("name") or ""},
            }
        case _:
            return None


#: Request converters into the Chat Completions shape, keyed by source API.
_TO_CHAT_REQUEST: dict[MantleApi, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "responses": _responses_to_chat_request,
    "messages": _messages_to_chat_request,
}

#: Request converters out of the Chat Completions shape, keyed by target API.
_FROM_CHAT_REQUEST: dict[MantleApi, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "responses": _chat_to_responses_request,
    "messages": _chat_to_messages_request,
}


def convert_payload(
    inbound: MantleApi, upstream: MantleApi, payload: dict[str, Any]
) -> dict[str, Any]:
    """Convert a request payload between Mantle wire formats.

    Conversion composes through the Chat Completions shape; fields without
    an equivalent there (e.g. Anthropic thinking budgets) are dropped.

    Args:
        inbound: Wire format of *payload*.
        upstream: Target wire format.
        payload: Request payload in the *inbound* shape.

    Returns:
        Request payload in the *upstream* shape (unchanged when identical).
    """
    if inbound == upstream:
        return payload
    if inbound != "chat_completions":
        payload = _TO_CHAT_REQUEST[inbound](payload)
    if upstream != "chat_completions":
        payload = _FROM_CHAT_REQUEST[upstream](payload)
    return payload


# ---------------------------------------------------------------------------
# Complete response conversion
# ---------------------------------------------------------------------------


def _responses_to_chat_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a Responses API response to the Chat Completions shape.

    Reasoning output items are dropped.

    Args:
        raw: Responses API response dict.

    Returns:
        Chat Completions response dict.
    """
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in raw.get("output") or []:
        match item.get("type"):
            case "message":
                texts += [
                    part.get("text") or ""
                    for part in item.get("content") or []
                    if part.get("type") == "output_text"
                ]
            case "function_call":
                tool_calls.append(
                    {
                        "id": item.get("call_id") or item.get("id") or "",
                        "type": "function",
                        "function": {
                            "name": item.get("name") or "",
                            "arguments": item.get("arguments") or "{}",
                        },
                    }
                )
            case _:
                pass
    message: dict[str, Any] = {"role": "assistant", "content": "".join(texts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{_id_token(raw.get('id') or uuid4().hex)}",
        "object": "chat.completion",
        "created": int(raw.get("created_at") or time()),
        "model": raw.get("model") or "",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _finish_from_response(
                    raw, has_tool_calls=bool(tool_calls)
                ),
                "logprobs": None,
            }
        ],
        "usage": _chat_usage_from_responses(raw.get("usage") or {}),
        **_optional_fields(raw, ("service_tier",)),
    }


def _messages_to_chat_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic Messages response to the Chat Completions shape.

    Thinking blocks are dropped.

    Args:
        raw: Anthropic Messages response dict.

    Returns:
        Chat Completions response dict.
    """
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in raw.get("content") or []:
        match block.get("type"):
            case "text":
                texts.append(block.get("text") or "")
            case "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or "",
                        "type": "function",
                        "function": {
                            "name": block.get("name") or "",
                            "arguments": dumps(block.get("input") or {}),
                        },
                    }
                )
            case _:
                pass
    message: dict[str, Any] = {"role": "assistant", "content": "".join(texts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    finish = (
        "tool_calls"
        if tool_calls
        else _STOP_TO_FINISH.get(raw.get("stop_reason") or "", "stop")
    )
    return {
        "id": f"chatcmpl-{_id_token(raw.get('id') or uuid4().hex)}",
        "object": "chat.completion",
        "created": int(time()),
        "model": raw.get("model") or "",
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish, "logprobs": None}
        ],
        "usage": _chat_usage_from_messages(raw.get("usage") or {}),
    }


def _chat_to_responses_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a Chat Completions response to the Responses API shape.

    Reasoning text becomes a ``reasoning`` output item preceding the message,
    as on the Converse path.

    Args:
        raw: Chat Completions response dict.

    Returns:
        Responses API response dict.
    """
    choice = (raw.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    response_id = f"resp_{_id_token(raw.get('id') or uuid4().hex)}"
    output: list[dict[str, Any]] = []
    if reasoning := message.get(_REASONING_CONTENT_KEY):
        # Same shape the Converse-served path emits, so a client reads reasoning
        # from one field whichever backend answered.
        output.append(
            {
                "type": "reasoning",
                "id": f"{response_id}-rs-0",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": reasoning}],
                "status": "completed",
            }
        )
    if content := message.get("content"):
        output.append(
            {
                "type": "message",
                "id": f"{response_id}-msg-0",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": content, "annotations": []}
                ],
            }
        )
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        call_id = call.get("id") or f"call_{index}"
        output.append(
            {
                "type": "function_call",
                "id": f"{response_id}-fc-{call_id}",
                "call_id": call_id,
                "name": function.get("name") or "",
                "arguments": function.get("arguments") or "{}",
                "status": "completed",
            }
        )
    finish = choice.get("finish_reason") or "stop"
    response = {
        "id": response_id,
        "object": "response",
        "created_at": int(raw.get("created") or time()),
        "model": raw.get("model") or "",
        "status": "incomplete" if finish in _FINISH_TO_INCOMPLETE else "completed",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": _responses_usage_from_chat(raw.get("usage") or {}),
        **_optional_fields(raw, ("service_tier",)),
    }
    if reason := _FINISH_TO_INCOMPLETE.get(finish):
        response["incomplete_details"] = {"reason": reason}
    return response


def _chat_to_messages_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a Chat Completions response to the Anthropic Messages shape.

    Args:
        raw: Chat Completions response dict.

    Returns:
        Anthropic Messages response dict.
    """
    choice = (raw.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    if text := message.get("content"):
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") or "",
                "name": function.get("name") or "",
                "input": _json_object(function.get("arguments")),
            }
        )
    finish = choice.get("finish_reason") or "stop"
    return {
        "id": f"msg_{_id_token(raw.get('id') or uuid4().hex)}",
        "type": "message",
        "role": "assistant",
        "model": raw.get("model") or "",
        "content": content,
        "stop_reason": _FINISH_TO_STOP.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": _messages_usage_from_chat(raw.get("usage") or {}),
    }


#: Response converters into the Chat Completions shape, keyed by source API.
_TO_CHAT_RESPONSE: dict[MantleApi, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "responses": _responses_to_chat_response,
    "messages": _messages_to_chat_response,
}

#: Response converters out of the Chat Completions shape, keyed by target API.
_FROM_CHAT_RESPONSE: dict[MantleApi, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "responses": _chat_to_responses_response,
    "messages": _chat_to_messages_response,
}


def convert_response(
    upstream: MantleApi, inbound: MantleApi, raw: dict[str, Any]
) -> dict[str, Any]:
    """Convert a complete upstream response between Mantle wire formats.

    Conversion composes through the Chat Completions shape. Chat Completions
    reasoning text becomes a Responses reasoning item; Anthropic thinking
    blocks and Responses reasoning items are dropped in the other direction.

    Args:
        upstream: Wire format of *raw*.
        inbound: Target wire format.
        raw: Complete upstream response dict.

    Returns:
        Response dict in the *inbound* shape (unchanged when identical).
    """
    if upstream == inbound:
        return raw
    if upstream != "chat_completions":
        raw = _TO_CHAT_RESPONSE[upstream](raw)
    if inbound != "chat_completions":
        raw = _FROM_CHAT_RESPONSE[inbound](raw)
    return raw


# ---------------------------------------------------------------------------
# Stream conversion
# ---------------------------------------------------------------------------

#: Default message when an upstream stream error event carries no message.
_STREAM_ERROR_FALLBACK = "The upstream model stream reported an error."


def _stream_error_message(data: str) -> str | None:
    """Extract the error message carried by an in-band stream error payload.

    Handles Anthropic ``error`` events, Responses ``error`` and
    ``response.failed`` events, and chat-shaped ``{"error": ...}`` payloads.

    Args:
        data: Raw SSE data payload.

    Returns:
        The upstream error message, or ``None`` when no error is carried.
    """
    try:
        payload = loads(data)
    except JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error") or (payload.get("response") or {}).get("error")
    if error is None and payload.get("type") != "error":
        return None
    if isinstance(error, dict):
        error = error.get("message")
    message = error if isinstance(error, str) else str(payload.get("message") or "")
    return message or _STREAM_ERROR_FALLBACK


def _parsed_chunk(data: str) -> dict[str, Any] | None:
    """Parse one SSE data payload, tolerating malformed frames.

    Args:
        data: Raw SSE data payload.

    Returns:
        The parsed JSON object, or None for a malformed or non-object frame
        (skipped rather than aborting the stream, mirroring the passthrough
        path's tolerance).
    """
    try:
        payload = loads(data)
    except JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _chunk_template(
    raw_id: str | None, created: float | None, model: str | None
) -> dict[str, Any]:
    """Build the invariant fields shared by all Chat Completions chunks.

    Args:
        raw_id: Upstream identifier, or ``None`` to generate one.
        created: Upstream creation timestamp, or ``None`` for now.
        model: Upstream model identifier.

    Returns:
        Template with ``id``, ``created`` and ``model`` fields.
    """
    token = _id_token(raw_id) if raw_id else uuid4().hex
    return {
        "id": f"chatcmpl-{token}",
        "created": int(created or time()),
        "model": model or "",
    }


def _chat_chunk(
    template: dict[str, Any],
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> SseEvent:
    """Build one Chat Completions streaming chunk event.

    Args:
        template: Shared ``id``/``created``/``model`` fields.
        delta: Choice delta payload, if any.
        finish_reason: Finish reason closing the choice, if any.
        usage: Usage block for the final usage chunk, if any.

    Returns:
        Unnamed SSE event carrying the chunk JSON.
    """
    chunk: dict[str, Any] = {
        **template,
        "object": "chat.completion.chunk",
        "choices": [],
    }
    if delta is not None or finish_reason is not None:
        chunk["choices"] = [
            {"index": 0, "delta": delta or {}, "finish_reason": finish_reason}
        ]
    if usage is not None:
        chunk["usage"] = usage
    return None, dumps(chunk)


async def _responses_stream_to_chat(
    events: AsyncGenerator[SseEvent],
) -> AsyncGenerator[SseEvent]:
    """Convert a Responses SSE stream to Chat Completions chunks.

    Reasoning events are dropped. A final usage chunk is always emitted from
    ``response.completed``/``response.incomplete``. Event payloads are only
    parsed once per event, and only for the event names that contribute
    chunks; a malformed data payload is skipped rather than aborting the
    stream (mirroring the passthrough path's tolerance).

    Args:
        events: Upstream Responses SSE events.

    Yields:
        Chat Completions chunk events.

    Raises:
        MantleError: When the upstream stream reports an in-band error.
    """
    template = _chunk_template(None, None, None)
    tool_index = -1
    async for event, data in events:
        parsed = _parsed_chunk(data)
        match event or (parsed or {}).get("type"):
            case "response.created":
                response = (parsed or {}).get("response") or {}
                template = _chunk_template(
                    response.get("id"),
                    response.get("created_at"),
                    response.get("model"),
                )
                yield _chat_chunk(template, delta={"role": "assistant", "content": ""})
            case "response.output_text.delta":
                yield _chat_chunk(
                    template, delta={"content": (parsed or {}).get("delta") or ""}
                )
            case "response.output_item.added" if (
                item := (parsed or {}).get("item") or {}
            ).get("type") == "function_call":
                tool_index += 1
                yield _chat_chunk(
                    template,
                    delta={
                        "tool_calls": [
                            {
                                "index": tool_index,
                                "id": item.get("call_id") or item.get("id") or "",
                                "type": "function",
                                "function": {
                                    "name": item.get("name") or "",
                                    "arguments": item.get("arguments") or "",
                                },
                            }
                        ]
                    },
                )
            case "response.function_call_arguments.delta" if tool_index >= 0:
                yield _chat_chunk(
                    template,
                    delta={
                        "tool_calls": [
                            {
                                "index": tool_index,
                                "function": {
                                    "arguments": (parsed or {}).get("delta") or ""
                                },
                            }
                        ]
                    },
                )
            case "response.completed" | "response.incomplete":
                response = (parsed or {}).get("response") or {}
                yield _chat_chunk(
                    template,
                    finish_reason=_finish_from_response(
                        response, has_tool_calls=tool_index >= 0
                    ),
                )
                yield _chat_chunk(
                    template,
                    usage=_chat_usage_from_responses(response.get("usage") or {}),
                )
            case "error" | "response.failed":
                raise MantleError(
                    _stream_error_message(data) or _STREAM_ERROR_FALLBACK, status=502
                )
            case None if '"error"' in data and (message := _stream_error_message(data)):
                raise MantleError(message, status=502)
            case _:
                pass


async def _messages_stream_to_chat(
    events: AsyncGenerator[SseEvent],
) -> AsyncGenerator[SseEvent]:
    """Convert an Anthropic SSE stream to Chat Completions chunks.

    Thinking deltas are dropped. A final usage chunk is always emitted,
    combining ``message_start`` input usage with ``message_delta`` usage.
    Event payloads are only parsed once per event, and only for the event
    names that contribute chunks; a malformed data payload is skipped rather
    than aborting the stream (mirroring the passthrough path's tolerance).

    Args:
        events: Upstream Anthropic SSE events.

    Yields:
        Chat Completions chunk events.

    Raises:
        MantleError: When the upstream stream reports an in-band error.
    """
    template = _chunk_template(None, None, None)
    input_usage: dict[str, Any] = {}
    tool_index = -1
    tool_block = False
    async for event, data in events:
        parsed = _parsed_chunk(data)
        match event or (parsed or {}).get("type"):
            case "message_start":
                message = (parsed or {}).get("message") or {}
                template = _chunk_template(
                    message.get("id"), None, message.get("model")
                )
                input_usage = message.get("usage") or {}
                yield _chat_chunk(template, delta={"role": "assistant", "content": ""})
            case "content_block_start":
                block = (parsed or {}).get("content_block") or {}
                tool_block = block.get("type") == "tool_use"
                if tool_block:
                    tool_index += 1
                    yield _chat_chunk(
                        template,
                        delta={
                            "tool_calls": [
                                {
                                    "index": tool_index,
                                    "id": block.get("id") or "",
                                    "type": "function",
                                    "function": {
                                        "name": block.get("name") or "",
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                    )
            case "content_block_delta":
                delta = _chat_delta_from_messages(
                    (parsed or {}).get("delta") or {},
                    tool_block=tool_block,
                    tool_index=tool_index,
                )
                if delta is not None:
                    yield _chat_chunk(template, delta=delta)
            case "message_delta":
                stop = ((parsed or {}).get("delta") or {}).get("stop_reason")
                yield _chat_chunk(
                    template, finish_reason=_STOP_TO_FINISH.get(stop or "", "stop")
                )
                yield _chat_chunk(
                    template,
                    usage=_chat_usage_from_messages(
                        {**input_usage, **((parsed or {}).get("usage") or {})}
                    ),
                )
            case "error":
                raise MantleError(
                    _stream_error_message(data) or _STREAM_ERROR_FALLBACK, status=502
                )
            case None if '"error"' in data and (message := _stream_error_message(data)):
                raise MantleError(message, status=502)
            case _:
                pass


def _chat_delta_from_messages(
    delta: dict[str, Any], *, tool_block: bool, tool_index: int
) -> dict[str, Any] | None:
    """Convert one Anthropic content block delta to a chunk delta.

    Args:
        delta: Anthropic ``content_block_delta`` delta payload.
        tool_block: Whether the open block is a ``tool_use`` block.
        tool_index: Chat Completions index of the open tool call.

    Returns:
        Chunk delta payload, or ``None`` for unmappable deltas.
    """
    if (text := delta.get("text")) is not None:
        return {"content": text}
    if tool_block and (fragment := delta.get("partial_json")) is not None:
        return {
            "tool_calls": [{"index": tool_index, "function": {"arguments": fragment}}]
        }
    return None


@dataclass(slots=True)
class _ResponsesStreamState:
    """Accumulated state while emitting Responses events from chunks."""

    response: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    output_index: int = -1
    kind: str = ""
    item_id: str = ""
    call_id: str = ""
    tool_name: str = ""
    text_parts: list[str] = field(default_factory=list)
    args_parts: list[str] = field(default_factory=list)
    output: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    completed: bool = False

    def next_seq(self) -> int:
        """Return the current sequence number and advance the counter.

        Returns:
            The sequence number assigned to the caller.
        """
        self.seq += 1
        return self.seq - 1


def _responses_event(
    state: _ResponsesStreamState, name: str, payload: dict[str, Any]
) -> SseEvent:
    """Build one named Responses SSE event with its sequence number.

    Args:
        state: Stream state providing the sequence counter.
        name: Event name (also set as the payload ``type``).
        payload: Event payload fields.

    Returns:
        Named SSE event.
    """
    return name, dumps({"type": name, **payload, "sequence_number": state.next_seq()})


async def _chat_stream_to_responses(
    events: AsyncGenerator[SseEvent], response_id: str | None = None
) -> AsyncGenerator[SseEvent]:
    """Convert Chat Completions chunks to a Responses SSE stream.

    The Converse adapter emits the same wire grammar from Bedrock events
    (``_adapters/_openai_responses.py``): event shapes must stay in sync.

    Args:
        events: Chat Completions chunk events.
        response_id: Route-assigned response ID carried by every emitted
            event, so the streamed ID is the one the route can retrieve.

    Yields:
        Named Responses SSE events, ending with ``response.completed``.

    Raises:
        MantleError: When the upstream stream reports an in-band error.
    """
    state = _ResponsesStreamState()
    async for _, data in events:
        if '"error"' in data and (message := _stream_error_message(data)):
            raise MantleError(message, status=502)
        if (chunk := _parsed_chunk(data)) is None:
            continue
        for event in _responses_chunk_events(state, chunk, response_id):
            yield event
    for event in _responses_stream_tail(state):
        yield event


def _responses_chunk_events(
    state: _ResponsesStreamState, chunk: dict[str, Any], response_id: str | None
) -> list[SseEvent]:
    """Emit the Responses events produced by one Chat Completions chunk.

    Args:
        state: Mutable stream state.
        chunk: Parsed Chat Completions chunk.
        response_id: Route-assigned response ID to use instead of a minted
            one; item IDs derive from it, as on the Converse path.

    Returns:
        Responses SSE events.
    """
    events: list[SseEvent] = []
    if not state.response:
        state.response = {
            "id": response_id or f"resp_{_id_token(chunk.get('id') or uuid4().hex)}",
            "object": "response",
            "created_at": int(chunk.get("created") or time()),
            "model": chunk.get("model") or "",
            "status": "in_progress",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        events += [
            _responses_event(state, "response.created", {"response": state.response}),
            _responses_event(
                state, "response.in_progress", {"response": state.response}
            ),
        ]
    if usage := chunk.get("usage"):
        state.usage = usage
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if reasoning := delta.get(_REASONING_CONTENT_KEY):
            events += _responses_reasoning_delta(state, reasoning)
        if content := delta.get("content"):
            events += _responses_text_delta(state, content)
        for tool_delta in delta.get("tool_calls") or []:
            events += _responses_tool_delta(state, tool_delta)
        if finish := choice.get("finish_reason"):
            state.finish_reason = finish
            events += _close_responses_item(state)
    if (
        state.usage is not None
        and state.finish_reason is not None
        and not state.completed
    ):
        events.append(_responses_completed(state))
    return events


def _responses_reasoning_delta(
    state: _ResponsesStreamState, content: str
) -> list[SseEvent]:
    """Emit the events for one reasoning delta, opening a reasoning item if needed.

    Args:
        state: Mutable stream state.
        content: Reasoning text delta.

    Returns:
        Responses SSE events.
    """
    events: list[SseEvent] = []
    if state.kind != "reasoning":
        events += _close_responses_item(state)
        state.output_index += 1
        state.kind = "reasoning"
        state.item_id = f"{state.response['id']}-rs-{state.output_index}"
        state.text_parts = []
        item = {
            "type": "reasoning",
            "id": state.item_id,
            "summary": [],
            "status": "in_progress",
        }
        events.append(
            _responses_event(
                state,
                "response.output_item.added",
                {"item": item, "output_index": state.output_index},
            )
        )
        events.append(
            _responses_event(
                state,
                "response.content_part.added",
                {
                    "item_id": state.item_id,
                    "output_index": state.output_index,
                    "content_index": 0,
                    "part": {"type": "reasoning_text", "text": ""},
                },
            )
        )
    state.text_parts.append(content)
    events.append(
        _responses_event(
            state,
            "response.reasoning_text.delta",
            {
                "item_id": state.item_id,
                "output_index": state.output_index,
                "content_index": 0,
                "delta": content,
            },
        )
    )
    return events


def _responses_text_delta(state: _ResponsesStreamState, content: str) -> list[SseEvent]:
    """Emit the events for one text delta, opening a message item if needed.

    Args:
        state: Mutable stream state.
        content: Text delta.

    Returns:
        Responses SSE events.
    """
    events: list[SseEvent] = []
    if state.kind != "text":
        events += _close_responses_item(state)
        state.output_index += 1
        state.kind = "text"
        state.item_id = f"{state.response['id']}-msg-{state.output_index}"
        state.text_parts = []
        item = {
            "type": "message",
            "id": state.item_id,
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        }
        events.append(
            _responses_event(
                state,
                "response.output_item.added",
                {"item": item, "output_index": state.output_index},
            )
        )
        events.append(
            _responses_event(
                state,
                "response.content_part.added",
                {
                    "item_id": state.item_id,
                    "output_index": state.output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            )
        )
    state.text_parts.append(content)
    events.append(
        _responses_event(
            state,
            "response.output_text.delta",
            {
                "item_id": state.item_id,
                "output_index": state.output_index,
                "content_index": 0,
                "delta": content,
                "logprobs": [],
            },
        )
    )
    return events


def _responses_tool_delta(
    state: _ResponsesStreamState, tool_delta: dict[str, Any]
) -> list[SseEvent]:
    """Emit the events for one tool-call delta entry.

    Args:
        state: Mutable stream state.
        tool_delta: Chat Completions ``delta.tool_calls`` entry.

    Returns:
        Responses SSE events.
    """
    events: list[SseEvent] = []
    function = tool_delta.get("function") or {}
    if tool_delta.get("id") or function.get("name"):
        events += _close_responses_item(state)
        state.output_index += 1
        state.kind = "tool"
        state.call_id = tool_delta.get("id") or f"call_{uuid4().hex}"
        state.tool_name = function.get("name") or ""
        state.item_id = f"{state.response['id']}-fc-{state.call_id}"
        state.args_parts = []
        item = {
            "type": "function_call",
            "id": state.item_id,
            "call_id": state.call_id,
            "name": state.tool_name,
            "arguments": "",
            "status": "in_progress",
        }
        events.append(
            _responses_event(
                state,
                "response.output_item.added",
                {"item": item, "output_index": state.output_index},
            )
        )
    if (fragment := function.get("arguments")) and state.kind == "tool":
        state.args_parts.append(fragment)
        events.append(
            _responses_event(
                state,
                "response.function_call_arguments.delta",
                {
                    "item_id": state.item_id,
                    "output_index": state.output_index,
                    "delta": fragment,
                },
            )
        )
    return events


def _close_responses_item(state: _ResponsesStreamState) -> list[SseEvent]:
    """Close the open output item, recording it in the final output list.

    Args:
        state: Mutable stream state.

    Returns:
        Responses SSE events closing the item (empty when none is open).
    """
    if state.kind == "reasoning":
        events = _close_responses_reasoning(state)
    elif state.kind == "text":
        events = _close_responses_text(state)
    elif state.kind == "tool":
        events = _close_responses_tool(state)
    else:
        return []
    state.kind = ""
    return events


def _close_responses_reasoning(state: _ResponsesStreamState) -> list[SseEvent]:
    """Close the open reasoning item.

    Args:
        state: Mutable stream state.

    Returns:
        Responses SSE events closing the reasoning item.
    """
    text = "".join(state.text_parts)
    part = {"type": "reasoning_text", "text": text}
    item = {
        "type": "reasoning",
        "id": state.item_id,
        "summary": [],
        "content": [part],
        "status": "completed",
    }
    state.output.append(item)
    common = {
        "item_id": state.item_id,
        "output_index": state.output_index,
        "content_index": 0,
    }
    return [
        _responses_event(
            state, "response.reasoning_text.done", {**common, "text": text}
        ),
        _responses_event(state, "response.content_part.done", {**common, "part": part}),
        _responses_event(
            state,
            "response.output_item.done",
            {"item": item, "output_index": state.output_index},
        ),
    ]


def _close_responses_text(state: _ResponsesStreamState) -> list[SseEvent]:
    """Close the open text message item.

    Args:
        state: Mutable stream state.

    Returns:
        Responses SSE events closing the message item.
    """
    text = "".join(state.text_parts)
    part = {"type": "output_text", "text": text, "annotations": []}
    item = {
        "type": "message",
        "id": state.item_id,
        "role": "assistant",
        "status": "completed",
        "content": [part],
    }
    state.output.append(item)
    common = {
        "item_id": state.item_id,
        "output_index": state.output_index,
        "content_index": 0,
    }
    return [
        _responses_event(
            state, "response.output_text.done", {**common, "text": text, "logprobs": []}
        ),
        _responses_event(state, "response.content_part.done", {**common, "part": part}),
        _responses_event(
            state,
            "response.output_item.done",
            {"item": item, "output_index": state.output_index},
        ),
    ]


def _close_responses_tool(state: _ResponsesStreamState) -> list[SseEvent]:
    """Close the open function-call item.

    Args:
        state: Mutable stream state.

    Returns:
        Responses SSE events closing the function-call item.
    """
    arguments = "".join(state.args_parts) or "{}"
    item = {
        "type": "function_call",
        "id": state.item_id,
        "call_id": state.call_id,
        "name": state.tool_name,
        "arguments": arguments,
        "status": "completed",
    }
    state.output.append(item)
    return [
        _responses_event(
            state,
            "response.function_call_arguments.done",
            {
                "item_id": state.item_id,
                "output_index": state.output_index,
                "arguments": arguments,
            },
        ),
        _responses_event(
            state,
            "response.output_item.done",
            {"item": item, "output_index": state.output_index},
        ),
    ]


def _responses_completed(state: _ResponsesStreamState) -> SseEvent:
    """Build the terminal ``response.completed``/``response.incomplete`` event.

    Matches the sibling Converse adapter's wire grammar: a truncated
    response is announced as ``response.incomplete``, not as a
    ``response.completed`` event carrying ``status: "incomplete"``.

    Args:
        state: Mutable stream state.

    Returns:
        The terminal SSE event, carrying the final usage.
    """
    state.completed = True
    finish = state.finish_reason or "stop"
    incomplete = finish in _FINISH_TO_INCOMPLETE
    response = {
        **state.response,
        "status": "incomplete" if incomplete else "completed",
        "output": state.output,
        "usage": _responses_usage_from_chat(state.usage or {}),
    }
    if reason := _FINISH_TO_INCOMPLETE.get(finish):
        response["incomplete_details"] = {"reason": reason}
    event_name = "response.incomplete" if incomplete else "response.completed"
    return _responses_event(state, event_name, {"response": response})


def _responses_stream_tail(state: _ResponsesStreamState) -> list[SseEvent]:
    """Emit the closing events when the upstream stream ends early.

    Args:
        state: Mutable stream state.

    Returns:
        Responses SSE events (empty when already completed or never started).
    """
    if not state.response or state.completed:
        return []
    events = _close_responses_item(state)
    events.append(_responses_completed(state))
    return events


@dataclass(slots=True)
class _MessagesStreamState:
    """Accumulated state while emitting Anthropic events from chunks.

    The Converse adapter emits the same wire grammar from Bedrock events
    (``_adapters/_anthropic_message.py``): event shapes must stay in sync.
    """

    started: bool = False
    block_index: int = -1
    kind: str = ""
    stop_reason: str | None = None
    usage: dict[str, Any] | None = None


def _messages_event(name: str, payload: dict[str, Any]) -> SseEvent:
    """Build one named Anthropic SSE event.

    Args:
        name: Event name (also set as the payload ``type``).
        payload: Event payload fields.

    Returns:
        Named SSE event.
    """
    return name, dumps({"type": name, **payload})


async def _chat_stream_to_messages(
    events: AsyncGenerator[SseEvent],
    response_id: str | None = None,  # noqa: ARG001 (uniform converter signature)
) -> AsyncGenerator[SseEvent]:
    """Convert Chat Completions chunks to an Anthropic SSE stream.

    Args:
        events: Chat Completions chunk events.
        response_id: Unused; Anthropic message IDs are not retrievable.

    Yields:
        Named Anthropic SSE events, ending with ``message_delta`` (carrying
        the stop reason and full usage) and ``message_stop``.

    Raises:
        MantleError: When the upstream stream reports an in-band error.
    """
    state = _MessagesStreamState()
    async for _, data in events:
        if '"error"' in data and (message := _stream_error_message(data)):
            raise MantleError(message, status=502)
        if (chunk := _parsed_chunk(data)) is None:
            continue
        for event in _messages_chunk_events(state, chunk):
            yield event
    for event in _messages_stream_tail(state):
        yield event


def _messages_chunk_events(
    state: _MessagesStreamState, chunk: dict[str, Any]
) -> list[SseEvent]:
    """Emit the Anthropic events produced by one Chat Completions chunk.

    Args:
        state: Mutable stream state.
        chunk: Parsed Chat Completions chunk.

    Returns:
        Anthropic SSE events.
    """
    events: list[SseEvent] = []
    if not state.started:
        state.started = True
        message: dict[str, Any] = {
            "id": f"msg_{_id_token(chunk.get('id') or uuid4().hex)}",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": chunk.get("model") or "",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        events.append(_messages_event("message_start", {"message": message}))
    if usage := chunk.get("usage"):
        state.usage = usage
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if content := delta.get("content"):
            events += _messages_text_delta(state, content)
        for tool_delta in delta.get("tool_calls") or []:
            events += _messages_tool_delta(state, tool_delta)
        if finish := choice.get("finish_reason"):
            events += _close_messages_block(state)
            state.stop_reason = _FINISH_TO_STOP.get(finish, "end_turn")
    return events


def _messages_text_delta(state: _MessagesStreamState, content: str) -> list[SseEvent]:
    """Emit the events for one text delta, opening a text block if needed.

    Args:
        state: Mutable stream state.
        content: Text delta.

    Returns:
        Anthropic SSE events.
    """
    events: list[SseEvent] = []
    if state.kind != "text":
        events += _close_messages_block(state)
        state.block_index += 1
        state.kind = "text"
        events.append(
            _messages_event(
                "content_block_start",
                {
                    "index": state.block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
    events.append(
        _messages_event(
            "content_block_delta",
            {
                "index": state.block_index,
                "delta": {"type": "text_delta", "text": content},
            },
        )
    )
    return events


def _messages_tool_delta(
    state: _MessagesStreamState, tool_delta: dict[str, Any]
) -> list[SseEvent]:
    """Emit the events for one tool-call delta entry.

    Args:
        state: Mutable stream state.
        tool_delta: Chat Completions ``delta.tool_calls`` entry.

    Returns:
        Anthropic SSE events.
    """
    events: list[SseEvent] = []
    function = tool_delta.get("function") or {}
    if tool_delta.get("id") or function.get("name"):
        events += _close_messages_block(state)
        state.block_index += 1
        state.kind = "tool"
        events.append(
            _messages_event(
                "content_block_start",
                {
                    "index": state.block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_delta.get("id") or f"toolu_{uuid4().hex}",
                        "name": function.get("name") or "",
                        "input": {},
                    },
                },
            )
        )
    if (fragment := function.get("arguments")) and state.kind == "tool":
        events.append(
            _messages_event(
                "content_block_delta",
                {
                    "index": state.block_index,
                    "delta": {"type": "input_json_delta", "partial_json": fragment},
                },
            )
        )
    return events


def _close_messages_block(state: _MessagesStreamState) -> list[SseEvent]:
    """Close the open content block, if any.

    Args:
        state: Mutable stream state.

    Returns:
        Anthropic SSE events (empty when no block is open).
    """
    if not state.kind:
        return []
    state.kind = ""
    return [_messages_event("content_block_stop", {"index": state.block_index})]


def _messages_stream_tail(state: _MessagesStreamState) -> list[SseEvent]:
    """Emit the terminal ``message_delta`` and ``message_stop`` events.

    Args:
        state: Mutable stream state.

    Returns:
        Anthropic SSE events (empty when the stream never started).
    """
    if not state.started:
        return []
    events = _close_messages_block(state)
    events.append(
        _messages_event(
            "message_delta",
            {
                "delta": {
                    "stop_reason": state.stop_reason or "end_turn",
                    "stop_sequence": None,
                },
                "usage": _messages_usage_from_chat(state.usage or {}),
            },
        )
    )
    events.append(_messages_event("message_stop", {}))
    return events


#: Stream converters into the Chat Completions shape, keyed by source API.
_TO_CHAT_STREAM: dict[
    MantleApi, Callable[[AsyncGenerator[SseEvent]], AsyncGenerator[SseEvent]]
] = {"responses": _responses_stream_to_chat, "messages": _messages_stream_to_chat}

#: Stream converters out of the Chat Completions shape, keyed by target API.
_FROM_CHAT_STREAM: dict[
    MantleApi,
    Callable[[AsyncGenerator[SseEvent], str | None], AsyncGenerator[SseEvent]],
] = {"responses": _chat_stream_to_responses, "messages": _chat_stream_to_messages}


def convert_stream(
    upstream: MantleApi,
    inbound: MantleApi,
    events: AsyncGenerator[SseEvent],
    response_id: str | None = None,
) -> AsyncGenerator[SseEvent]:
    """Convert an upstream SSE stream between Mantle wire formats.

    Conversion composes through the Chat Completions chunk shape. Chat
    Completions reasoning deltas become Responses reasoning summary events;
    Anthropic thinking deltas and Responses reasoning events are dropped in
    the other direction. Chat Completions output always ends with a usage
    chunk (the caller strips it when the client did not opt in) and never
    includes a ``[DONE]`` sentinel.

    Args:
        upstream: Wire format of *events*.
        inbound: Target wire format.
        events: Upstream SSE event generator.
        response_id: Route-assigned response ID stamped on converted
            Responses events, so the streamed ID stays retrievable.

    Returns:
        SSE event generator in the *inbound* shape (unchanged when identical).
    """
    if upstream == inbound:
        return events
    if upstream != "chat_completions":
        events = _TO_CHAT_STREAM[upstream](events)
    if inbound != "chat_completions":
        events = _FROM_CHAT_STREAM[inbound](events, response_id)
    return events


# ---------------------------------------------------------------------------
# Legacy text completions
# ---------------------------------------------------------------------------


async def text_completion_as_chat_payload(
    request: CompletionCreateParams, model_id: str
) -> dict[str, Any]:
    """Build a Chat Completions payload from a legacy completion request.

    Args:
        request: Completion creation request following the OpenAI spec.
        model_id: Mantle model identifier to set on the payload.

    Returns:
        JSON-ready Chat Completions request payload.

    Raises:
        ApiError: When the request uses unsupported options (``echo``,
            ``suffix``, ``logprobs``), multiple prompts, or file prompts.
    """
    for name in ("echo", "suffix", "logprobs"):
        if getattr(request, name):
            msg = f"`{name}` is not supported by this model."
            raise ApiError(msg, status=400)
    prompt = request.prompt
    if isinstance(prompt, list):
        if len(prompt) != 1:
            msg = "Multiple prompts are not supported by this model."
            raise ApiError(msg, status=400)
        prompt = prompt[0]
    if not isinstance(prompt, str):
        msg = "File prompts are not supported by this model."
        raise ApiError(msg, status=400)
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
    }
    for name in ("max_tokens", "temperature", "top_p", "n", "stop", "stream", "user"):
        if (value := getattr(request, name)) is not None:
            payload[name] = value
    return payload


def _text_finish(finish_reason: str | None) -> str | None:
    """Map a Chat Completions finish reason to the text-completion shape.

    Args:
        finish_reason: Chat Completions finish reason.

    Returns:
        Text-completion finish reason (tool finishes map to ``stop``).
    """
    if finish_reason is None:
        return None
    return finish_reason if finish_reason in _TEXT_FINISH_REASONS else "stop"


def chat_response_as_text_completion(
    raw: dict[str, Any], completion_id: str
) -> Completion:
    """Convert a Chat Completions response to a legacy ``Completion``.

    Args:
        raw: Chat Completions response dict.
        completion_id: Identifier for the completion.

    Returns:
        Validated ``Completion`` response model.
    """
    choices = [
        {
            "text": (choice.get("message") or {}).get("content") or "",
            "index": choice.get("index", index),
            "finish_reason": _text_finish(choice.get("finish_reason")),
            "logprobs": None,
        }
        for index, choice in enumerate(raw.get("choices") or [])
    ]
    usage = {
        key: value
        for key, value in (raw.get("usage") or {}).items()
        if key in CompletionUsage.model_fields
    }
    return Completion.model_validate(
        {
            "id": completion_id,
            "object": "text_completion",
            "created": int(raw.get("created") or time()),
            "model": raw.get("model") or "",
            "choices": choices,
            "usage": usage or None,
            **_optional_fields(raw, ("service_tier", "system_fingerprint")),
        }
    )


async def chat_stream_as_text_completion(
    events: AsyncGenerator[ServerSentEvent], completion_id: str
) -> AsyncGenerator[ServerSentEvent]:
    """Wrap a Chat Completions SSE stream as text-completion chunks.

    The ``[DONE]`` sentinel and named events carrying neither content nor
    usage (e.g. relayed errors) are passed through unchanged.

    Args:
        events: Inbound-shaped Chat Completions server-sent events.
        completion_id: Identifier set on the emitted chunks.

    Yields:
        Text-completion chunk server-sent events.

    Raises:
        MantleError: When an unnamed upstream chunk reports an in-band error.
    """
    async for event in events:
        data = event.data
        if not isinstance(data, str) or data == "[DONE]":
            yield event
            continue
        if (
            event.event is None
            and '"error"' in data
            and (message := _stream_error_message(data))
        ):
            raise MantleError(message, status=502)
        if (chunk := _parsed_chunk(data)) is None:
            if event.event is not None:
                yield event
            continue
        choices = _text_completion_choices(chunk)
        usage = chunk.get("usage")
        if not choices and not usage:
            if event.event is not None:
                yield event
            continue
        converted: dict[str, Any] = {
            "id": completion_id,
            "object": "text_completion",
            "created": chunk.get("created") or int(time()),
            "model": chunk.get("model") or "",
            "choices": choices,
        }
        if usage:
            converted["usage"] = {
                key: value
                for key, value in usage.items()
                if key in CompletionUsage.model_fields
            }
        yield ServerSentEvent(data=dumps(converted), event=event.event)


def _text_completion_choices(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Chat Completions chunk choices to text-completion choices.

    Args:
        chunk: Parsed Chat Completions chunk.

    Returns:
        Text-completion choices (empty for role-only chunks).
    """
    choices: list[dict[str, Any]] = []
    for index, choice in enumerate(chunk.get("choices") or []):
        delta = choice.get("delta") or {}
        text = delta.get("content")
        finish = _text_finish(choice.get("finish_reason"))
        if text is None and finish is None:
            continue
        choices.append(
            {
                "text": text or "",
                "index": choice.get("index", index),
                "finish_reason": finish,
                "logprobs": None,
            }
        )
    return choices
