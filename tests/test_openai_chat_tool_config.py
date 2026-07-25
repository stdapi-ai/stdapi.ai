"""Unit tests for Converse tool-config handling (no AWS calls).

Covers ``tool_choice='none'`` semantics and agent round-trip history where the
final turn omits ``tools`` but still contains ``toolUse``/``toolResult`` blocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from stdapi.models.chat._adapters._openai_chat_completion import build_tool_config
from stdapi.models.chat._default import ChatModel
from stdapi.types.openai_chat_completions import CompletionCreateParams

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.type_defs import (
        MessageTypeDef,
        ToolConfigurationTypeDef,
    )

pytestmark = pytest.mark.local

#: Model instance under test; construction is side-effect free.
_MODEL = ChatModel("amazon.nova-2-lite-v1:0")

#: A minimal OpenAI function-tool definition.
_WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _request(**kwargs: object) -> CompletionCreateParams:
    """Build a validated chat completion request with the given overrides."""
    base: dict[str, Any] = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(kwargs)
    return CompletionCreateParams.model_validate(base)


def _tool_call_history() -> list[MessageTypeDef]:
    """Return a Bedrock message history containing toolUse and toolResult blocks."""
    return [
        {"role": "user", "content": [{"text": "weather?"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "call_1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "call_1", "content": [{"text": "sunny"}]}}
            ],
        },
    ]


async def _prepare(
    messages: list[MessageTypeDef], tool_config: ToolConfigurationTypeDef | None
) -> dict[str, Any]:
    """Assemble a Converse request payload from messages and a tool config."""
    return dict(
        await _MODEL._prepare_converse_request(  # noqa: SLF001
            bedrock_messages=messages,
            inference_cfg={},
            system_blocks=None,
            tool_config=tool_config,
            additional_request_fields={},
            service_tier=None,
        )
    )


def test_tool_choice_none_with_tools_omits_tool_config() -> None:
    """``tool_choice='none'`` validates and yields no tool config (no forced tools)."""
    request = _request(tools=[_WEATHER_TOOL], tool_choice="none")
    assert build_tool_config(request) is None


def test_tool_choice_none_legacy_function_call_omits_tool_config() -> None:
    """Legacy ``function_call='none'`` also yields no tool config."""
    request = _request(functions=[_WEATHER_TOOL["function"]], function_call="none")
    assert build_tool_config(request) is None


async def test_tool_choice_none_with_tool_history_still_succeeds() -> None:
    """``tool_choice='none'`` plus tool history keeps a synthesized config so Converse accepts it."""
    request = _request(tools=[_WEATHER_TOOL], tool_choice="none")
    payload = await _prepare(_tool_call_history(), build_tool_config(request))
    tool_config = payload["toolConfig"]
    names = {tool["toolSpec"]["name"] for tool in tool_config["tools"]}
    assert names == {"get_weather"}
    # 'none' must never force a tool call: no explicit toolChoice is emitted.
    assert "toolChoice" not in tool_config


async def test_round_trip_history_without_tools_synthesizes_tool_config() -> None:
    """History with toolUse/toolResult but no request tools gets a synthesized config."""
    payload = await _prepare(_tool_call_history(), None)
    tool_config = payload["toolConfig"]
    names = {tool["toolSpec"]["name"] for tool in tool_config["tools"]}
    assert names == {"get_weather"}
    assert "toolChoice" not in tool_config


async def test_round_trip_with_tools_left_unchanged() -> None:
    """A provided tool config passes through untouched even with tool history present."""
    provided: ToolConfigurationTypeDef = {
        "tools": [
            {
                "toolSpec": {
                    "name": "get_weather",
                    "description": "Look up weather.",
                    "inputSchema": {"json": {"type": "object"}},
                }
            }
        ],
        "toolChoice": {"auto": {}},
    }
    payload = await _prepare(_tool_call_history(), provided)
    assert payload["toolConfig"] == provided


async def test_no_tools_no_history_has_no_tool_config() -> None:
    """Plain history without tool blocks stays free of any tool config."""
    payload = await _prepare([{"role": "user", "content": [{"text": "hi"}]}], None)
    assert "toolConfig" not in payload
