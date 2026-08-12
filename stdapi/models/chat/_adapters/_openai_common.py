"""Shared helpers for OpenAI-shaped adapters (Chat Completions and Responses)."""

from typing import TYPE_CHECKING, Any

from stdapi.aws_bedrock import PROMPT_CACHING, PROMPT_CACHING_DEFAULT
from stdapi.types.openai_chat_completions import CompletionUsage, PromptTokensDetails
from stdapi.utils import try_parse_json

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.literals import (
        CacheTTLType,
        ServiceTierTypeType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockTypeDef,
        ConverseStreamOutputTypeDef,
        MessageTypeDef,
        SystemContentBlockTypeDef,
        ToolConfigurationTypeDef,
        ToolResultContentBlockUnionTypeDef,
    )

    from stdapi.aws_bedrock import PromptCaching
    from stdapi.types.openai_chat_completions import (
        PromptCacheOptions,
        PromptCacheRetention,
        ServiceTiers,
    )
    from stdapi.types.openai_responses import (
        PromptCacheOptions as ResponsesPromptCacheOptions,
    )


#: OpenAI service tiers to Bedrock mapping
_SERVICES_TIERS: dict[ServiceTiers, ServiceTierTypeType] = {
    "priority": "priority",
    "flex": "flex",
    # Extra bedrock specific values
    "reserved": "reserved",
}

#: `prompt_cache_options.mode` value disabling the `prompt_cache_key` heuristic
EXPLICIT_CACHE_MODE = "explicit"

#: OpenAI to Bedrock prompt cache retention mapping
CACHE_TTL: dict[PromptCacheRetention | None, CacheTTLType | None] = {
    "in_memory": None,
    "24h": "1h",  # max current value
    "1h": "1h",
    "5m": "5m",
}

#: OpenAI `prompt_cache_options.ttl` to Bedrock cache TTL mapping
_OPTIONS_CACHE_TTL: dict[str, CacheTTLType] = {
    "30m": "1h"  # closest AWS Bedrock TTL covering 30 minutes
}


def parse_tool_content(text: str) -> ToolResultContentBlockUnionTypeDef:
    """Return ``{"json": obj}`` for JSON objects, ``{"text": text}`` otherwise.

    Bedrock's ``toolResult.content[].json`` only accepts JSON objects, so
    arrays, scalars and non-JSON payloads fall through to ``text``.

    Args:
        text: Raw tool output string.

    Returns:
        A Bedrock ``toolResult`` content block.
    """
    if isinstance(parsed := try_parse_json(text), dict):
        return {"json": parsed}
    return {"text": text}


def map_service_tier(
    value: ServiceTiers | None,
) -> tuple[ServiceTierTypeType | None, ServiceTiers | None]:
    """Map OpenAI service tier to Bedrock service tier.

    Only the request's own value is translated here: the alias and
    server-configured tiers resolve where the Bedrock request is built
    (:func:`stdapi.aws_bedrock.resolve_service_tier`), alongside the tier
    header, and the response echoes the requested value.

    Args:
        value: OpenAI service tier.

    Returns:
        Bedrock service tier, Echoed OpenAI service tier.
    """
    if value is None:
        return None, None
    if value in _SERVICES_TIERS:
        return _SERVICES_TIERS[value], value
    return None, "default"


def resolve_cache_ttl(
    prompt_cache_retention: PromptCacheRetention | None,
    prompt_cache_options: PromptCacheOptions
    | ResponsesPromptCacheOptions
    | None = None,
) -> CacheTTLType | None:
    """Resolve the Bedrock cache TTL requested by a request.

    Args:
        prompt_cache_retention: Requested cache retention, which takes precedence.
        prompt_cache_options: Request prompt cache options, whose ``ttl`` applies
            when no retention is set.

    Returns:
        The Bedrock cache TTL, or ``None`` for the Bedrock default.
    """
    if prompt_cache_retention:
        return CACHE_TTL.get(prompt_cache_retention)
    if prompt_cache_options is not None and prompt_cache_options.ttl:
        return _OPTIONS_CACHE_TTL.get(prompt_cache_options.ttl)
    return None


def build_cache_point(ttl: CacheTTLType | None = None) -> ContentBlockTypeDef:
    """Build a Bedrock cache point block.

    Args:
        ttl: Cache TTL, or ``None`` for the Bedrock default.

    Returns:
        Bedrock cache point block.
    """
    if ttl:
        return {"cachePoint": {"type": "default", "ttl": ttl}}
    return PROMPT_CACHING_DEFAULT


def cap_cache_points(
    system_blocks: list[SystemContentBlockTypeDef] | None,
    tool_config: ToolConfigurationTypeDef | None,
    bedrock_messages: list[MessageTypeDef],
    max_cache_points: int,
) -> None:
    """Drop the oldest cache points exceeding the Bedrock per-request limit.

    Blocks are scanned in the order Bedrock reads them (system, tools, then
    messages), so the surviving cache points are always the latest ones, which
    cover the longest prompt prefixes.

    Args:
        system_blocks: System content blocks list.
        tool_config: Bedrock tool configuration.
        bedrock_messages: Bedrock message list.
        max_cache_points: Maximum number of cache points allowed per request.
    """
    block_lists: list[list[Any]] = []
    if system_blocks:
        block_lists.append(system_blocks)
    if tool_config and (tools := tool_config.get("tools")):
        block_lists.append(tools)  # type: ignore[arg-type]
    block_lists += [message["content"] for message in bedrock_messages]  # type: ignore[misc]

    excess = (
        sum(sum("cachePoint" in block for block in blocks) for blocks in block_lists)
        - max_cache_points
    )
    for blocks in block_lists:
        if excess <= 0:
            return
        kept = []
        for block in blocks:
            if excess > 0 and "cachePoint" in block:
                excess -= 1
            else:
                kept.append(block)
        blocks[:] = kept


#: Appended to the system prompt when Bedrock's ``outputConfig`` applies no JSON schema.
JSON_OBJECT_SYSTEM_INSTRUCTION: SystemContentBlockTypeDef = {
    "text": "Respond with a single syntactically valid JSON object and no other text."
}


def enforce_json_object(
    system_blocks: list[SystemContentBlockTypeDef], *, requested: bool
) -> None:
    """Append a JSON-only instruction to *system_blocks* when requested.

    Bedrock applies no decoding constraint for ``json_object`` output (unlike
    ``json_schema``), so this system-prompt nudge is the only enforcement
    available.  It is appended as its own block, leaving any explicit
    user/system prompt untouched.

    Args:
        system_blocks: Mutable system content blocks list to append to.
        requested: Whether the request asked for unconstrained JSON-object
            output (``response_format``/``text.format`` of type ``json_object``).
    """
    if requested:
        system_blocks.append(JSON_OBJECT_SYSTEM_INSTRUCTION)


def drop_tool_turn_cache_points(bedrock_messages: list[MessageTypeDef]) -> None:
    """Remove cache points from messages carrying tool use or tool result blocks.

    Models without tool caching reject a ``cachePoint`` in such turns.

    Args:
        bedrock_messages: Bedrock message list.
    """
    for message in bedrock_messages:
        content: list[Any] = message["content"]  # type: ignore[assignment]
        if any("toolUse" in block or "toolResult" in block for block in content):
            content[:] = [block for block in content if "cachePoint" not in block]


def parse_prompt_cache_key(
    prompt_cache_key: str | None,
    prompt_cache_options: PromptCacheOptions
    | ResponsesPromptCacheOptions
    | None = None,
) -> set[PromptCaching]:
    """Map a ``prompt_cache_key`` value to the set of cache components to enable.

    A dot-separated string like ``"system.tools"`` enables specific components;
    any unrecognised values are ignored.  A non-empty string with no recognisable
    tokens enables all components (``PROMPT_CACHING``).

    Args:
        prompt_cache_key: Dot-separated cache component selector, or ``None``.
        prompt_cache_options: Request prompt cache options.  The ``explicit`` mode
            disables this key-driven placement: only content parts marked with
            ``prompt_cache_breakpoint`` are then cached.

    Returns:
        Set of ``PromptCaching`` values to enable, or an empty set when caching is disabled.
    """
    if (
        prompt_cache_options is not None
        and prompt_cache_options.mode == EXPLICIT_CACHE_MODE
    ):
        return set()
    if prompt_cache_key:
        return (
            set(prompt_cache_key.split(".")) & PROMPT_CACHING  # type: ignore[return-value]
        ) or PROMPT_CACHING
    return set()


def extract_stream_usage(
    stream_event: ConverseStreamOutputTypeDef,
) -> CompletionUsage | None:
    """Extract ``CompletionUsage`` from a Bedrock ``metadata`` stream event.

    Args:
        stream_event: A single event from ``ConverseStream``.

    Returns:
        Populated ``CompletionUsage`` when the event carries ``metadata.usage``,
        else ``None``.
    """
    if "metadata" not in stream_event:
        return None
    usage = stream_event["metadata"]["usage"]
    # OpenAI semantics: prompt_tokens covers the full prompt, cache buckets included.
    cache_read = usage.get("cacheReadInputTokens", 0)
    cache_write = usage.get("cacheWriteInputTokens", 0)
    prompt_tokens = usage["inputTokens"] + cache_read + cache_write
    completion_usage = CompletionUsage(
        completion_tokens=usage["outputTokens"],
        prompt_tokens=prompt_tokens,
        total_tokens=prompt_tokens + usage["outputTokens"],
    )
    if cache_read or cache_write:
        completion_usage.prompt_tokens_details = PromptTokensDetails(
            cached_tokens=cache_read, cache_write_tokens=cache_write or None
        )
    return completion_usage
