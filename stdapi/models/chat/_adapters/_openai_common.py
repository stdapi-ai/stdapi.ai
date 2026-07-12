"""Shared helpers for OpenAI-shaped adapters (Chat Completions and Responses)."""

from typing import TYPE_CHECKING

from stdapi.aws_bedrock import PROMPT_CACHING
from stdapi.types.openai_chat_completions import CompletionUsage, PromptTokensDetails
from stdapi.utils import try_parse_json

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.literals import (
        CacheTTLType,
        ServiceTierTypeType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ConverseStreamOutputTypeDef,
        ToolResultContentBlockUnionTypeDef,
    )

    from stdapi.aws_bedrock import PromptCaching
    from stdapi.types.openai_chat_completions import PromptCacheRetention, ServiceTiers


#: OpenAI service tiers to Bedrock mapping
_SERVICES_TIERS: dict[ServiceTiers, ServiceTierTypeType] = {
    "priority": "priority",
    "flex": "flex",
    # Extra bedrock specific values
    "reserved": "reserved",
}

#: OpenAI to Bedrock prompt cache retention mapping
CACHE_TTL: dict[PromptCacheRetention | None, CacheTTLType | None] = {
    "in_memory": None,
    "24h": "1h",  # max current value
    "1h": "1h",
    "5m": "5m",
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

    Args:
        value: OpenAI service tier.

    Returns:
        Bedrock service tier, Effective OpenAI service tier.
    """
    if value is None:
        return None, None
    if value in _SERVICES_TIERS:
        return _SERVICES_TIERS[value], value
    return None, "default"


def parse_prompt_cache_key(prompt_cache_key: str | None) -> set[PromptCaching]:
    """Map a ``prompt_cache_key`` value to the set of cache components to enable.

    A dot-separated string like ``"system.tools"`` enables specific components;
    any unrecognised values are ignored.  A non-empty string with no recognisable
    tokens enables all components (``PROMPT_CACHING``).

    Args:
        prompt_cache_key: Dot-separated cache component selector, or ``None``.

    Returns:
        Set of ``PromptCaching`` values to enable, or an empty set when caching is disabled.
    """
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
    cache_read = usage.get("cacheReadInputTokens")
    prompt_tokens = (
        usage["inputTokens"] + (cache_read or 0) + usage.get("cacheWriteInputTokens", 0)
    )
    completion_usage = CompletionUsage(
        completion_tokens=usage["outputTokens"],
        prompt_tokens=prompt_tokens,
        total_tokens=prompt_tokens + usage["outputTokens"],
    )
    if cache_read:
        completion_usage.prompt_tokens_details = PromptTokensDetails(
            cached_tokens=cache_read
        )
    return completion_usage
