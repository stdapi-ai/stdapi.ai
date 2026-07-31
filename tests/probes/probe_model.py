"""Probe one model for the parameters and features it actually accepts.

Service models and model cards declare shapes, not runtime acceptance, and the
divergence is where the gateway's bugs live: a knob a model silently ignores, a
value only one generation rejects, a toggle that needs a second field to do
anything. This script settles those questions once per model, empirically, and
records the answers next to the tests so a later change can be checked against
them instead of against memory.

Each probe sends one request that differs from a known-good baseline in exactly
one way, then classifies the outcome:

``supported``
    The request succeeded and the feature's own observable effect appeared —
    a reasoning block, a tool call, a cache write.
``accepted``
    The request succeeded but nothing observable changed. The parameter is
    tolerated and ignored, which is the gateway's default posture but also the
    signature of a knob that silently does nothing.
``rejected``
    The backend refused it. The recorded message is what a caller would see.
``error``
    Anything else — recorded verbatim rather than guessed at.

Usage::

    uv run python -m tests.probes.probe_model amazon.nova-2-lite-v1:0
    uv run python -m tests.probes.probe_model --all-new
    uv run python -m tests.probes.probe_model deepseek.v3.2 --region us-west-2

Results land in ``tests/probes/results/<model-id>.json``. Commit them: they are
the evidence for every model-specific branch in ``stdapi/models/chat/``.

Ref: AGENTS.md "Probing a new model"
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    import re
    from collections.abc import Callable, Sequence

    from aiobotocore.config import AioConfig
    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.monitoring import EventLog

#: Where a probe run writes its record, one file per model.
RESULTS_DIR = Path(__file__).parent / "results"

#: Bumped when the probe set changes in a way that invalidates older records.
SCHEMA_VERSION = 4

#: Outcome of a single probe.
Outcome = Literal["supported", "accepted", "rejected", "error", "skipped"]

#: Prompt that needs several arithmetic steps, so a reasoning model has a reason to reason.
_REASONING_PROMPT = (
    "Three friends split a bill. Ana paid 3/5 of the total, Ben paid 12 euros, "
    "and Cleo paid the rest, which was half of what Ben paid. What was the "
    "total? Answer with the number only."
)

#: Trivial prompt for probes that only need the call to go through.
_BASELINE_PROMPT = "Reply with the single word OK."

#: Filler long enough to clear the smallest documented prompt-cache minimum.
_CACHE_FILLER = ("The gateway translates requests between API dialects. " * 220).strip()


def _probe_png() -> bytes:
    """Build a small opaque PNG for the image-input probe.

    Deliberately not a 1x1 pixel: several models answer "Could not process
    image" for one, which reads as a refusal of image input when it is really a
    refusal of that image.

    Returns:
        PNG bytes, 64x64, single colour.
    """
    from io import BytesIO  # noqa: PLC0415 - only needed to build the fixture

    from PIL import Image  # noqa: PLC0415 - test-only dependency

    with BytesIO() as buffer:
        Image.new("RGB", (64, 64), (32, 96, 160)).save(buffer, format="PNG")
        return buffer.getvalue()


#: Image handed to the image-input probe.
_TINY_PNG = _probe_png()

#: The same PNG as a data URI, for the Chat Completions image part.
_TINY_PNG_DATA_URI = f"data:image/png;base64,{b64encode(_TINY_PNG).decode()}"

#: The weather tool in the OpenAI function shape, for Mantle probes.
_OPENAI_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

#: Structured-output schema for the outputConfig probe, serialized as Bedrock wants it.
_PLACE_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"],
    }
)

#: Tool the model is offered when a probe needs a tool call to happen.
_WEATHER_TOOL = {
    "toolSpec": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }
        },
    }
}


@dataclass
class ProbeResult:
    """One probe's outcome, as recorded in the model's result file.

    Attributes:
        name: Probe identifier, stable across runs.
        feature: Human-readable capability being probed.
        outcome: Classification of what the backend did.
        detail: Error message, or what was observed for a supported probe.
        request: The field(s) the probe added to the baseline request.
    """

    name: str
    feature: str
    outcome: Outcome
    detail: str = ""
    request: dict[str, Any] = field(default_factory=dict)


@dataclass
class Probe:
    """A single request variation and the effect that proves it took.

    Attributes:
        name: Stable probe identifier.
        feature: Human-readable capability.
        overrides: Extra/replacement keys merged into the baseline Converse request.
        observe: Predicate on the response returning the observed effect, or an
            empty string when the parameter was accepted without visible effect.
        applies_to: Optional filter on the model id.
    """

    name: str
    feature: str
    overrides: dict[str, Any]
    observe: Callable[[dict[str, Any]], str] = lambda _response: ""
    applies_to: re.Pattern[str] | None = None


def _blocks(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the assistant content blocks of a Converse response.

    Args:
        response: Parsed Converse response.

    Returns:
        The content blocks, or an empty list when the shape is unexpected.
    """
    message = (response.get("output") or {}).get("message") or {}
    blocks = message.get("content")
    return blocks if isinstance(blocks, list) else []


def _has_reasoning(response: dict[str, Any]) -> str:
    """Report the reasoning block a response carries, if any.

    Args:
        response: Parsed Converse response.

    Returns:
        A description of the reasoning found, or an empty string.
    """
    for block in _blocks(response):
        if "reasoningContent" in block:
            text = (block["reasoningContent"].get("reasoningText") or {}).get(
                "text"
            ) or ""
            return f"reasoningContent block, {len(text)} chars"
    return ""


def _has_tool_use(response: dict[str, Any]) -> str:
    """Report the tool call a response carries, if any.

    Args:
        response: Parsed Converse response.

    Returns:
        A description of the tool call found, or an empty string.
    """
    for block in _blocks(response):
        if "toolUse" in block:
            return f"toolUse: {block['toolUse'].get('name')}"
    return ""


def _wrote_cache(response: dict[str, Any]) -> str:
    """Report prompt-cache write/read counters, if the model billed any.

    Args:
        response: Parsed Converse response.

    Returns:
        A description of the cache accounting, or an empty string.
    """
    usage = response.get("usage") or {}
    written = usage.get("cacheWriteInputTokens") or 0
    read = usage.get("cacheReadInputTokens") or 0
    return f"cacheWrite={written} cacheRead={read}" if written or read else ""


def _stopped_on_sequence(response: dict[str, Any]) -> str:
    """Report whether generation halted at the requested stop sequence.

    Bedrock keeps the sequence in the returned text and several models report
    ``end_turn`` rather than ``stop_sequence``, so neither the stop reason nor
    the sequence's absence proves anything: what proves it is that nothing
    follows the sequence.

    Args:
        response: Parsed Converse response.

    Returns:
        A description of where generation stopped, or an empty string.
    """
    text = "".join(block.get("text", "") for block in _blocks(response))
    _, sep, tail = text.partition("STOPHERE")
    if sep and not tail.strip():
        return f"halted at the sequence, stopReason={response.get('stopReason')}"
    return ""


def _is_json_object(response: dict[str, Any]) -> str:
    """Report whether the answer parses as a JSON object.

    Args:
        response: Parsed Converse response.

    Returns:
        A description of the parsed object, or an empty string.
    """
    text = "".join(block.get("text", "") for block in _blocks(response)).strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        return ""
    return f"JSON object with keys {sorted(parsed)}" if isinstance(parsed, dict) else ""


#: Every probe run against a Converse-served chat model, in reporting order.
PROBES: tuple[Probe, ...] = (
    Probe(
        name="system_prompt",
        feature="System prompt block",
        overrides={"system": [{"text": "You always answer in exactly one word."}]},
    ),
    Probe(
        name="temperature",
        feature="inferenceConfig.temperature",
        overrides={"inferenceConfig": {"maxTokens": 64, "temperature": 0.2}},
    ),
    Probe(
        name="top_p",
        feature="inferenceConfig.topP",
        overrides={"inferenceConfig": {"maxTokens": 64, "topP": 0.5}},
    ),
    Probe(
        name="temperature_and_top_p",
        feature="temperature and topP together",
        overrides={
            "inferenceConfig": {"maxTokens": 64, "temperature": 0.2, "topP": 0.5}
        },
    ),
    Probe(
        name="stop_sequences",
        feature="inferenceConfig.stopSequences",
        overrides={
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": "Write exactly: BEGIN STOPHERE END"}],
                }
            ],
            "inferenceConfig": {"maxTokens": 64, "stopSequences": ["STOPHERE"]},
        },
        observe=_stopped_on_sequence,
    ),
    Probe(
        name="top_k",
        feature="additionalModelRequestFields.top_k",
        overrides={"additionalModelRequestFields": {"top_k": 10}},
    ),
    Probe(
        name="tool_use",
        feature="toolConfig with an auto tool choice",
        overrides={
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": "What is the weather in Lisbon?"}],
                }
            ],
            "toolConfig": {"tools": [_WEATHER_TOOL]},
        },
        observe=_has_tool_use,
    ),
    Probe(
        name="tool_choice_any",
        feature="toolChoice.any",
        overrides={
            "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
            "toolConfig": {"tools": [_WEATHER_TOOL], "toolChoice": {"any": {}}},
        },
        observe=_has_tool_use,
    ),
    Probe(
        name="tool_choice_tool",
        feature="toolChoice.tool (forced function)",
        overrides={
            "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
            "toolConfig": {
                "tools": [_WEATHER_TOOL],
                "toolChoice": {"tool": {"name": "get_weather"}},
            },
        },
        observe=_has_tool_use,
    ),
    Probe(
        name="thinking_enabled",
        feature='additionalModelRequestFields.thinking = {"type": "enabled"}',
        overrides={
            "messages": [{"role": "user", "content": [{"text": _REASONING_PROMPT}]}],
            "inferenceConfig": {"maxTokens": 4096},
            "additionalModelRequestFields": {"thinking": {"type": "enabled"}},
        },
        observe=_has_reasoning,
    ),
    Probe(
        name="reasoning_config_enabled",
        feature='additionalModelRequestFields.reasoning_config = {"type": "enabled"}',
        overrides={
            "messages": [{"role": "user", "content": [{"text": _REASONING_PROMPT}]}],
            "inferenceConfig": {"maxTokens": 4096},
            "additionalModelRequestFields": {
                "reasoning_config": {"type": "enabled", "budget_tokens": 1024}
            },
        },
        observe=_has_reasoning,
    ),
    Probe(
        name="reasoning_effort_high",
        feature="additionalModelRequestFields.reasoning_effort = high",
        overrides={
            "messages": [{"role": "user", "content": [{"text": _REASONING_PROMPT}]}],
            "inferenceConfig": {"maxTokens": 4096},
            "additionalModelRequestFields": {"reasoning_effort": "high"},
        },
        observe=_has_reasoning,
    ),
    Probe(
        name="reasoning_effort_low",
        feature="additionalModelRequestFields.reasoning_effort = low",
        overrides={
            "messages": [{"role": "user", "content": [{"text": _REASONING_PROMPT}]}],
            "inferenceConfig": {"maxTokens": 4096},
            "additionalModelRequestFields": {"reasoning_effort": "low"},
        },
        observe=_has_reasoning,
    ),
    Probe(
        name="reasoning_effort_minimal",
        feature="additionalModelRequestFields.reasoning_effort = minimal",
        overrides={
            "messages": [{"role": "user", "content": [{"text": _REASONING_PROMPT}]}],
            "inferenceConfig": {"maxTokens": 4096},
            "additionalModelRequestFields": {"reasoning_effort": "minimal"},
        },
        observe=_has_reasoning,
    ),
    Probe(
        name="prompt_cache",
        feature="cachePoint in the message content",
        overrides={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": _CACHE_FILLER},
                        {"cachePoint": {"type": "default"}},
                        {"text": _BASELINE_PROMPT},
                    ],
                }
            ]
        },
        observe=_wrote_cache,
    ),
    Probe(
        name="json_mode",
        feature="A prompt demanding a JSON object",
        overrides={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": "Return a JSON object with the keys city and "
                            "country for Lisbon. Output JSON only."
                        }
                    ],
                }
            ]
        },
        observe=_is_json_object,
    ),
    Probe(
        name="service_tier_flex",
        feature="performanceConfig latency=optimized",
        overrides={"performanceConfig": {"latency": "optimized"}},
    ),
    Probe(
        name="service_tier_priority",
        feature="serviceTier priority",
        overrides={"serviceTier": {"type": "priority"}},
    ),
    Probe(
        name="service_tier_flex_structured",
        feature="serviceTier flex",
        overrides={"serviceTier": {"type": "flex"}},
    ),
    Probe(
        name="whitespace_stop_sequence",
        feature="A whitespace-only stop sequence",
        overrides={"inferenceConfig": {"maxTokens": 64, "stopSequences": [" "]}},
    ),
    Probe(
        name="system_only",
        feature="A system block with no user text beyond it",
        overrides={
            "system": [{"text": "Answer every question with the word OK."}],
            "messages": [{"role": "user", "content": [{"text": "Anything."}]}],
        },
    ),
    Probe(
        name="assistant_prefill",
        feature="A trailing assistant turn the model must continue",
        overrides={
            "messages": [
                {"role": "user", "content": [{"text": "Name a colour."}]},
                {"role": "assistant", "content": [{"text": "The colour is"}]},
            ]
        },
    ),
    Probe(
        name="tool_result_round_trip",
        feature="A toolResult block replayed on the next turn",
        overrides={
            "messages": [
                {"role": "user", "content": [{"text": "Weather in Lisbon?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "t1",
                                "name": "get_weather",
                                "input": {"city": "Lisbon"},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": "t1",
                                "content": [{"text": "18C and sunny"}],
                            }
                        }
                    ],
                },
            ],
            "toolConfig": {"tools": [_WEATHER_TOOL]},
        },
    ),
    Probe(
        name="output_config_json_schema",
        feature="outputConfig with a JSON schema",
        overrides={
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": "Describe Lisbon as city and country."}],
                }
            ],
            "outputConfig": {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        # The schema travels as a JSON string, not as a nested
                        # document: botocore types this member as `str`.
                        "jsonSchema": {"schema": _PLACE_SCHEMA}
                    },
                }
            },
        },
        observe=_is_json_object,
    ),
    Probe(
        name="image_input",
        feature="An image content block",
        overrides={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "Reply with the single word OK."},
                        {"image": {"format": "png", "source": {"bytes": _TINY_PNG}}},
                    ],
                }
            ]
        },
    ),
    Probe(
        name="document_input",
        feature="A document content block",
        overrides={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": "Reply with the single word OK."},
                        {
                            "document": {
                                "format": "txt",
                                "name": "note",
                                "source": {"bytes": b"a short note"},
                            }
                        },
                    ],
                }
            ]
        },
    ),
    Probe(
        name="request_metadata",
        feature="requestMetadata attribution",
        overrides={"requestMetadata": {"probe": "stdapi"}},
    ),
    Probe(
        name="unknown_additional_field",
        feature="An unknown key in additionalModelRequestFields",
        overrides={"additionalModelRequestFields": {"stdapi_probe_unknown": 1}},
    ),
)


def _chat_message(response: dict[str, Any]) -> dict[str, Any]:
    """Return the assistant message of a Chat Completions response.

    Args:
        response: Parsed Chat Completions response.

    Returns:
        The message dict, or an empty dict when the shape is unexpected.
    """
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message")
    return message if isinstance(message, dict) else {}


def _chat_reasoning(response: dict[str, Any]) -> str:
    """Report the reasoning text a Chat Completions response carries.

    Both spellings are checked: upstream vendors are split between ``reasoning``
    and DeepSeek's ``reasoning_content``, and which one arrives is itself the
    finding.

    Args:
        response: Parsed Chat Completions response.

    Returns:
        A description naming the field found, or an empty string.
    """
    message = _chat_message(response)
    for key in ("reasoning", "reasoning_content"):
        if isinstance(text := message.get(key), str) and text:
            return f"{key}: {len(text)} chars"
    return ""


def _chat_tool_call(response: dict[str, Any]) -> str:
    """Report the tool call a Chat Completions response carries.

    Args:
        response: Parsed Chat Completions response.

    Returns:
        A description of the first tool call, or an empty string.
    """
    for call in _chat_message(response).get("tool_calls") or []:
        if isinstance(call, dict):
            return f"tool_call: {(call.get('function') or {}).get('name')}"
    return ""


def _chat_is_json_object(response: dict[str, Any]) -> str:
    """Report whether a Chat Completions answer parses as a JSON object.

    Args:
        response: Parsed Chat Completions response.

    Returns:
        A description of the parsed object, or an empty string.
    """
    try:
        parsed = json.loads(str(_chat_message(response).get("content") or "").strip())
    except ValueError:
        return ""
    return f"JSON object with keys {sorted(parsed)}" if isinstance(parsed, dict) else ""


def _chat_cached_tokens(response: dict[str, Any]) -> str:
    """Report prompt-cache accounting on a Chat Completions response.

    Args:
        response: Parsed Chat Completions response.

    Returns:
        A description of the cached-token counters, or an empty string.
    """
    details = (response.get("usage") or {}).get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    return f"cached_tokens={cached}" if cached else ""


#: The Chat Completions equivalents of the Converse probes, for Mantle-served models.
MANTLE_PROBES: tuple[Probe, ...] = (
    Probe(
        name="system_prompt",
        feature="A system-role message",
        overrides={
            "messages": [
                {"role": "system", "content": "You always answer in one word."},
                {"role": "user", "content": _BASELINE_PROMPT},
            ]
        },
    ),
    Probe(name="temperature", feature="temperature", overrides={"temperature": 0.2}),
    Probe(name="top_p", feature="top_p", overrides={"top_p": 0.5}),
    Probe(
        name="temperature_and_top_p",
        feature="temperature and top_p together",
        overrides={"temperature": 0.2, "top_p": 0.5},
    ),
    Probe(
        name="stop_sequences",
        feature="stop",
        overrides={
            "messages": [
                {"role": "user", "content": "Write exactly: BEGIN STOPHERE END"}
            ],
            "stop": ["STOPHERE"],
        },
        observe=lambda response: (
            "halted at the sequence"
            if "STOPHERE" not in str(_chat_message(response).get("content") or "")
            else ""
        ),
    ),
    Probe(
        name="frequency_penalty",
        feature="frequency_penalty",
        overrides={"frequency_penalty": 0.5},
    ),
    Probe(name="seed", feature="seed", overrides={"seed": 42}),
    Probe(name="logprobs", feature="logprobs", overrides={"logprobs": True}),
    Probe(name="n_choices", feature="n = 2", overrides={"n": 2}),
    Probe(
        name="tool_use",
        feature="tools with an auto tool choice",
        overrides={
            "messages": [{"role": "user", "content": "What is the weather in Lisbon?"}],
            "tools": [_OPENAI_WEATHER_TOOL],
        },
        observe=_chat_tool_call,
    ),
    Probe(
        name="tool_choice_any",
        feature='tool_choice = "required"',
        overrides={
            "messages": [{"role": "user", "content": "Say hello."}],
            "tools": [_OPENAI_WEATHER_TOOL],
            "tool_choice": "required",
        },
        observe=_chat_tool_call,
    ),
    Probe(
        name="tool_choice_tool",
        feature="tool_choice naming one function",
        overrides={
            "messages": [{"role": "user", "content": "Say hello."}],
            "tools": [_OPENAI_WEATHER_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        },
        observe=_chat_tool_call,
    ),
    Probe(
        name="parallel_tool_calls_false",
        feature="parallel_tool_calls = false",
        overrides={
            "messages": [{"role": "user", "content": "Weather in Lisbon and Porto?"}],
            "tools": [_OPENAI_WEATHER_TOOL],
            "parallel_tool_calls": False,
        },
        observe=_chat_tool_call,
    ),
    Probe(
        name="reasoning_effort_high",
        feature="reasoning_effort = high",
        overrides={
            "messages": [{"role": "user", "content": _REASONING_PROMPT}],
            "max_tokens": 4096,
            "reasoning_effort": "high",
        },
        observe=_chat_reasoning,
    ),
    Probe(
        name="reasoning_effort_low",
        feature="reasoning_effort = low",
        overrides={
            "messages": [{"role": "user", "content": _REASONING_PROMPT}],
            "max_tokens": 4096,
            "reasoning_effort": "low",
        },
        observe=_chat_reasoning,
    ),
    Probe(
        name="reasoning_effort_minimal",
        feature="reasoning_effort = minimal",
        overrides={
            "messages": [{"role": "user", "content": _REASONING_PROMPT}],
            "max_tokens": 4096,
            "reasoning_effort": "minimal",
        },
        observe=_chat_reasoning,
    ),
    Probe(
        name="reasoning_request_object",
        feature="OpenRouter-style reasoning object",
        overrides={
            "messages": [{"role": "user", "content": _REASONING_PROMPT}],
            "max_tokens": 4096,
            "reasoning": {"effort": "high"},
        },
        observe=_chat_reasoning,
    ),
    Probe(
        name="prompt_cache",
        feature="A prompt long enough to be cached",
        overrides={
            "messages": [
                {"role": "system", "content": _CACHE_FILLER},
                {"role": "user", "content": _BASELINE_PROMPT},
            ]
        },
        observe=_chat_cached_tokens,
    ),
    Probe(
        name="json_mode",
        feature='response_format = {"type": "json_object"}',
        overrides={
            "messages": [
                {
                    "role": "user",
                    "content": "Return a JSON object with the keys city and "
                    "country for Lisbon.",
                }
            ],
            "response_format": {"type": "json_object"},
        },
        observe=_chat_is_json_object,
    ),
    Probe(
        name="json_schema",
        feature='response_format = {"type": "json_schema"}',
        overrides={
            "messages": [
                {"role": "user", "content": "Describe Lisbon as city and country."}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "place",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "country": {"type": "string"},
                        },
                        "required": ["city", "country"],
                        "additionalProperties": False,
                    },
                },
            },
        },
        observe=_chat_is_json_object,
    ),
    Probe(
        name="image_input",
        feature="An image_url content part",
        overrides={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _BASELINE_PROMPT},
                        {"type": "image_url", "image_url": {"url": _TINY_PNG_DATA_URI}},
                    ],
                }
            ]
        },
    ),
    Probe(
        name="unknown_field",
        feature="An unknown top-level request field",
        overrides={"stdapi_probe_unknown": 1},
    ),
)

#: Probes run over the streaming API rather than the unary one.
STREAM_PROBES: tuple[Probe, ...] = (
    Probe(
        name="streaming",
        feature="ConverseStream",
        overrides={},
        observe=lambda response: (
            f"{response['_events']} events" if response.get("_events") else ""
        ),
    ),
)


#: Backend refusals that arrive as something other than a ValidationException.
_REFUSAL_MARKERS = ("unsupported model", "does not support", "isn't supported")

#: Seconds one probe may take. Botocore retries a slow model for minutes on its
#: own, which stalls a whole sweep; a probe that needs longer than this is not
#: measuring a feature any more.
_PROBE_TIMEOUT = 90


def _classify(exc: Exception) -> Outcome:
    """Tell a backend refusal apart from a fault in the probe itself.

    A model declining a capability is the answer being looked for, whatever
    exception class carries it -- prompt caching, for one, is refused with an
    ``AccessDeniedException``. A botocore ``ParamValidationError`` is the
    opposite: the request never left the process, so it says nothing about the
    model and everything about the probe.

    Args:
        exc: The exception the call raised.

    Returns:
        The outcome to record.
    """
    name, message = type(exc).__name__, str(exc)
    if "ValidationException" in name:
        return "rejected"
    if any(marker in message for marker in _REFUSAL_MARKERS):
        return "rejected"
    return "error"


def _client_config() -> AioConfig:
    """Build the bedrock-runtime config the probes call through.

    Botocore defaults to a 60s read timeout and five attempts, so one unhealthy
    model can hold a probe for minutes.  A probe only needs to learn whether the
    API accepts a shape, and a refusal comes back immediately.

    Returns:
        The client configuration.
    """
    from aiobotocore.config import AioConfig  # noqa: PLC0415 - optional at import

    return AioConfig(
        connect_timeout=10,
        read_timeout=60,
        retries={"max_attempts": 2, "mode": "standard"},
    )


async def _run_probe(
    client: Any,  # noqa: ANN401 - the aiobotocore client type is not exported
    model_id: str,
    probe: Probe,
    baseline: dict[str, Any],
) -> ProbeResult:
    """Send one probe and classify the outcome.

    Args:
        client: Open bedrock-runtime client.
        model_id: Model under probe.
        probe: The variation to send.
        baseline: Known-good request the probe is layered onto.

    Returns:
        The classified result.
    """
    if probe.applies_to is not None and not probe.applies_to.search(model_id):
        return ProbeResult(probe.name, probe.feature, "skipped", "not applicable")
    request = {**baseline, **probe.overrides, "modelId": model_id}
    try:
        async with asyncio.timeout(_PROBE_TIMEOUT):
            response = await client.converse(**request)
    except TimeoutError:
        return ProbeResult(
            probe.name,
            probe.feature,
            "error",
            f"no answer within {_PROBE_TIMEOUT}s",
            probe.overrides,
        )
    except Exception as exc:  # noqa: BLE001 - every failure mode is a result
        return ProbeResult(
            probe.name, probe.feature, _classify(exc), str(exc), probe.overrides
        )
    observed = probe.observe(response)
    return ProbeResult(
        probe.name,
        probe.feature,
        "supported" if observed else "accepted",
        observed or "no observable effect",
        probe.overrides,
    )


async def _run_stream_probe(
    client: Any,  # noqa: ANN401 - the aiobotocore client type is not exported
    model_id: str,
    probe: Probe,
    baseline: dict[str, Any],
) -> ProbeResult:
    """Send one probe over ConverseStream and classify the outcome.

    Args:
        client: Open bedrock-runtime client.
        model_id: Model under probe.
        probe: The variation to send.
        baseline: Known-good request the probe is layered onto.

    Returns:
        The classified result, with the event count as the observed effect.
    """
    request = {**baseline, **probe.overrides, "modelId": model_id}
    try:
        async with asyncio.timeout(_PROBE_TIMEOUT):
            response = await client.converse_stream(**request)
            events = 0
            async for _ in response["stream"]:
                events += 1
    except TimeoutError:
        return ProbeResult(
            probe.name,
            probe.feature,
            "error",
            f"no answer within {_PROBE_TIMEOUT}s",
            probe.overrides,
        )
    except Exception as exc:  # noqa: BLE001 - every failure mode is a result
        return ProbeResult(
            probe.name, probe.feature, _classify(exc), str(exc), probe.overrides
        )
    observed = probe.observe({"_events": events})
    return ProbeResult(
        probe.name,
        probe.feature,
        "supported" if observed else "accepted",
        observed or "no observable effect",
        probe.overrides,
    )


async def probe_model(model_id: str, region: str) -> dict[str, Any]:
    """Run every probe against one Converse-served model.

    Args:
        model_id: Bedrock model id or inference profile.
        region: AWS region to call.

    Returns:
        The record to write to the model's result file.
    """
    from aiobotocore.session import get_session  # noqa: PLC0415 - optional at import

    baseline: dict[str, Any] = {
        "messages": [{"role": "user", "content": [{"text": _BASELINE_PROMPT}]}],
        "inferenceConfig": {"maxTokens": 64},
    }
    results: list[ProbeResult] = []
    session = get_session()
    invoked = model_id
    baseline_probe = Probe(name="baseline", feature="Plain text request", overrides={})
    async with session.create_client(
        "bedrock-runtime", region_name=region, config=_client_config()
    ) as client:
        baseline_result = await _run_probe(client, invoked, baseline_probe, baseline)
        if _needs_inference_profile(baseline_result):
            # Several families are cross-region only; the catalog id alone is
            # refused, so the probes run against the profile the gateway uses.
            invoked = f"{region.split('-', maxsplit=1)[0]}.{model_id}"
            print(f"  retrying via inference profile {invoked}", file=sys.stderr)  # noqa: T201
            baseline_result = await _run_probe(
                client, invoked, baseline_probe, baseline
            )
        results.append(baseline_result)
        if baseline_result.outcome not in {"supported", "accepted"}:
            return _record(model_id, invoked, region, results)
        for probe in PROBES:
            results.append(await _run_probe(client, invoked, probe, baseline))
            print(f"  {results[-1].outcome:<10} {probe.name}", file=sys.stderr)  # noqa: T201
        for probe in STREAM_PROBES:
            results.append(await _run_stream_probe(client, invoked, probe, baseline))
            print(f"  {results[-1].outcome:<10} {probe.name}", file=sys.stderr)  # noqa: T201
    return _record(model_id, invoked, region, results)


async def probe_mantle_model(model_id: str, region: str) -> dict[str, Any]:
    """Run the Chat Completions equivalent of the probe set against Mantle.

    Mantle-served models never answer Converse, so the same questions are asked
    in the OpenAI wire format the endpoint speaks. Probe names match the Converse
    set wherever the capability is the same, so records stay comparable.

    Args:
        model_id: Mantle model identifier.
        region: AWS region to call.

    Returns:
        The record to write to the model's result file.
    """
    from stdapi import aws_bedrock_mantle as mantle  # noqa: PLC0415 - heavy import
    from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415 - heavy import

    # The Mantle client logs into the request context, which no CLI ever binds.
    REQUEST_LOG.set(cast("EventLog", {"level": "info"}))
    baseline: dict[str, Any] = {
        "messages": [{"role": "user", "content": _BASELINE_PROMPT}],
        "max_tokens": 64,
    }
    results: list[ProbeResult] = []
    async with mantle.mantle_http_session():
        for probe in (
            Probe(name="baseline", feature="Plain text request", overrides={}),
            *MANTLE_PROBES,
        ):
            payload = {**baseline, **probe.overrides, "model": model_id}
            try:
                async with asyncio.timeout(_PROBE_TIMEOUT):
                    response = await mantle.invoke(
                        cast("RegionName", region),
                        "/v1/chat/completions",
                        payload,
                        single_region=True,
                    )
            except TimeoutError:
                results.append(
                    ProbeResult(
                        probe.name,
                        probe.feature,
                        "error",
                        f"no answer within {_PROBE_TIMEOUT}s",
                        probe.overrides,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - every failure mode is a result
                results.append(
                    ProbeResult(
                        probe.name, probe.feature, "rejected", str(exc), probe.overrides
                    )
                )
            else:
                observed = probe.observe(response)
                results.append(
                    ProbeResult(
                        probe.name,
                        probe.feature,
                        "supported" if observed else "accepted",
                        observed or "no observable effect",
                        probe.overrides,
                    )
                )
            print(f"  {results[-1].outcome:<10} {probe.name}", file=sys.stderr)  # noqa: T201
            if probe.name == "baseline" and results[-1].outcome == "rejected":
                break
    return _record(model_id, model_id, region, results, transport="mantle")


def _needs_inference_profile(result: ProbeResult) -> bool:
    """Report whether a baseline failed only for want of an inference profile.

    Args:
        result: The baseline probe's outcome.

    Returns:
        True when the model exists but is cross-region only.
    """
    return result.outcome == "rejected" and "inference profile" in result.detail


def _record(
    model_id: str,
    invoked_id: str,
    region: str,
    results: Sequence[ProbeResult],
    transport: str = "converse",
) -> dict[str, Any]:
    """Build the on-disk record for a probe run.

    Args:
        model_id: Model that was probed, as the catalog names it.
        invoked_id: Id the probes were actually sent to, which is an inference
            profile for cross-region-only models.
        region: Region the probes ran in.
        results: Probe outcomes in run order.
        transport: Wire protocol the probes used.

    Returns:
        A JSON-serialisable record.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "invoked_id": invoked_id,
        "transport": transport,
        "region": region,
        "probed_at": datetime.now(UTC).date().isoformat(),
        "probes": [
            {
                "name": r.name,
                "feature": r.feature,
                "outcome": r.outcome,
                "detail": r.detail,
                "request": r.request,
            }
            for r in results
        ],
    }


def write_record(record: dict[str, Any]) -> Path:
    """Write a probe record next to the tests.

    Args:
        record: The record returned by :func:`probe_model`.

    Returns:
        The path written.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = record["model_id"].replace("/", "_").replace(":", "_")
    path = RESULTS_DIR / f"{slug}.json"
    path.write_text(
        json.dumps(record, indent=1, sort_keys=False, default=_jsonable) + "\n"
    )
    return path


def _jsonable(value: object) -> str:
    """Render a value JSON cannot hold, for the recorded request.

    Args:
        value: The unserialisable value, in practice the raw bytes of a probe's
            image or document block.

    Returns:
        A short placeholder naming the type and size.
    """
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return repr(value)


#: Output modalities whose probes would cost far more than the answer is worth.
_EXCLUDED_OUTPUT_MODALITIES = frozenset({"VIDEO", "IMAGE", "EMBEDDING", "SPEECH"})


def discover_chat_models() -> list[tuple[str, str, str]]:
    """List every text-generating model the gateway serves, with how to reach it.

    The catalog is read through the application itself, so the list is exactly
    what this deployment offers -- including Mantle-served models, which no
    Bedrock catalog call returns.

    Returns:
        Tuples of (model id, transport, region), skipping only the output
        modalities too costly to probe.
    """
    from starlette.testclient import TestClient  # noqa: PLC0415 - heavy import

    import tests.conftest  # noqa: F401, PLC0415 - applies the suite's own settings
    from stdapi.main import app  # noqa: PLC0415 - heavy import
    from stdapi.models import get_all_models_details, is_mantle_served  # noqa: PLC0415

    with TestClient(app):
        details = asyncio.run(get_all_models_details())
    models: list[tuple[str, str, str]] = []
    for model_id, detail in sorted(details.items()):
        # Legacy models are probed too: they are still served, and a retired
        # generation is exactly where an unnoticed capability gap sits.
        if "TEXT" not in detail.output_modalities:
            continue
        if _EXCLUDED_OUTPUT_MODALITIES & set(detail.output_modalities):
            continue
        models.append(
            (
                model_id,
                "mantle" if is_mantle_served(model_id) else "converse",
                _preferred_region(detail.regions or ()),
            )
        )
    # US first: those deployments carry capabilities the other partitions lag on,
    # so their answers are the ones a feature decision is usually made against.
    models.sort(key=lambda entry: (not entry[2].startswith("us-"), entry[0]))
    return models


def _preferred_region(regions: Sequence[str]) -> str:
    """Pick the region a model should be probed in.

    Args:
        regions: Every region the catalog reports the model in.

    Returns:
        A US region when the model has one, otherwise the first region listed,
        falling back to ``us-east-1`` for a model with no region at all.
    """
    ordered = sorted(regions)
    return next(
        (r for r in ordered if r.startswith("us-")),
        ordered[0] if ordered else "us-east-1",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Probe the requested models and record the results.

    Args:
        argv: Command-line arguments, defaulting to ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_id", nargs="*", help="Bedrock model id(s) to probe.")
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region to probe in (default us-east-1).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Probe every text-generating model the gateway serves.",
    )
    parser.add_argument(
        "--skip-recorded",
        action="store_true",
        help="Leave models that already have a result file alone.",
    )
    args = parser.parse_args(argv)

    if args.all:
        targets = discover_chat_models()
        print(f"discovered {len(targets)} model(s)", file=sys.stderr)  # noqa: T201
    elif args.model_id:
        targets = [(model_id, "auto", args.region) for model_id in args.model_id]
    else:
        parser.error("name at least one model id, or pass --all")

    for model_id, transport, region in targets:
        slug = model_id.replace("/", "_").replace(":", "_")
        if args.skip_recorded and (RESULTS_DIR / f"{slug}.json").exists():
            continue
        if transport == "auto":
            from stdapi.models import is_mantle_served  # noqa: PLC0415 - heavy import

            transport = "mantle" if is_mantle_served(model_id) else "converse"
        print(f"probing {model_id} ({transport}) in {region}", file=sys.stderr)  # noqa: T201
        probe = probe_mantle_model if transport == "mantle" else probe_model
        try:
            record = asyncio.run(probe(model_id, region))
        except Exception as exc:  # noqa: BLE001 - one model must not stop the sweep
            print(f"  FAILED {type(exc).__name__}: {exc}", file=sys.stderr)  # noqa: T201
            continue
        print(f"wrote {write_record(record)}", file=sys.stderr)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
