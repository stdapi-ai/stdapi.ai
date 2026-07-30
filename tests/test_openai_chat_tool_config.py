"""Unit tests for Converse tool-config handling (no AWS calls).

Covers ``tool_choice='none'`` semantics and agent round-trip history where the
final turn omits ``tools`` but still contains ``toolUse``/``toolResult`` blocks.

Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     stdapi/models/chat/_adapters/_openai_chat_completion.py:build_tool_config
     stdapi/models/chat/_default.py:_synthesize_tool_config_from_history
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from stdapi.models.chat._adapters._anthropic_message import _build_tool_config
from stdapi.models.chat._adapters._openai_chat_completion import build_tool_config
from stdapi.models.chat._default import ChatModel
from stdapi.types.anthropic_messages import (
    ToolChoiceAutoParam,
    ToolChoiceNoneParam,
    ToolParam,
)
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

#: The same tool declared in Anthropic's Messages API shape.
_ANTHROPIC_WEATHER_TOOL: dict[str, Any] = {
    "name": "get_weather",
    "input_schema": {"type": "object", "properties": {}},
}

#: The permissive ``toolSpec`` the model layer synthesizes from tool history.
_SYNTHESIZED_WEATHER_SPEC: dict[str, Any] = {
    "toolSpec": {"name": "get_weather", "inputSchema": {"json": {"type": "object"}}}
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
    """``tool_choice='none'`` validates and yields no tool config (no forced tools).

    Bedrock Converse has no ``none`` ``toolChoice``, so upstream's "the model
    must not call a tool" is expressed by dropping ``toolConfig`` entirely.
    The control request pins that the omission comes from ``none`` and not from
    the tool list being ignored.
    """
    assert (
        build_tool_config(_request(tools=[_WEATHER_TOOL], tool_choice="none")) is None
    )
    control = build_tool_config(_request(tools=[_WEATHER_TOOL]))
    assert control is not None
    assert [tool["toolSpec"]["name"] for tool in control["tools"]] == ["get_weather"]


def test_tool_choice_none_legacy_function_call_omits_tool_config() -> None:
    """Legacy ``function_call='none'`` also yields no tool config.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
    """
    assert (
        build_tool_config(
            _request(functions=[_WEATHER_TOOL["function"]], function_call="none")
        )
        is None
    )
    control = build_tool_config(_request(functions=[_WEATHER_TOOL["function"]]))
    assert control is not None
    assert [tool["toolSpec"]["name"] for tool in control["tools"]] == ["get_weather"]


async def test_tool_choice_none_with_tool_history_still_succeeds() -> None:
    """``tool_choice='none'`` plus tool history keeps a synthesized config so Converse accepts it.

    Converse rejects ``toolUse``/``toolResult`` blocks without a ``toolConfig``,
    so the dropped config is replaced by one permissive ``toolSpec`` per tool
    name found in history — never a ``toolChoice``, which would force a call.
    """
    request = _request(tools=[_WEATHER_TOOL], tool_choice="none")
    payload = await _prepare(_tool_call_history(), build_tool_config(request))
    tool_config = payload["toolConfig"]
    assert tool_config["tools"] == [_SYNTHESIZED_WEATHER_SPEC]
    # 'none' must never force a tool call: no explicit toolChoice is emitted.
    assert "toolChoice" not in tool_config


async def test_round_trip_history_without_tools_synthesizes_tool_config() -> None:
    """History with toolUse/toolResult but no request tools gets a synthesized config.

    The synthesized spec carries the permissive ``{"type": "object"}`` input
    schema, since the original tool declaration is no longer in the request.
    """
    payload = await _prepare(_tool_call_history(), None)
    tool_config = payload["toolConfig"]
    assert tool_config["tools"] == [_SYNTHESIZED_WEATHER_SPEC]
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
    messages: list[MessageTypeDef] = [{"role": "user", "content": [{"text": "hi"}]}]
    payload = await _prepare(messages, None)
    assert "toolConfig" not in payload
    assert payload["messages"] == messages


class TestAnthropicToolChoiceNone:
    """The Anthropic route disables tool calling the same way as Chat Completions.

    Both surfaces reach the same ``_prepare_converse_request``, so ``none`` can
    drop the tool config on both: when history still carries ``toolUse``/
    ``toolResult`` blocks the model layer synthesizes a permissive config, which
    is what makes dropping it safe.

    Ref: https://docs.claude.com/en/api/messages
         stdapi/models/chat/_adapters/_anthropic_message.py:_build_tool_config
    """

    def test_tool_choice_none_omits_tool_config(self) -> None:
        """``tool_choice: {"type": "none"}`` yields no tool config at all."""
        assert (
            _build_tool_config(
                [ToolParam.model_validate(_ANTHROPIC_WEATHER_TOOL)],
                ToolChoiceNoneParam(type="none"),
            )
            is None
        )

    def test_other_choices_still_build_a_config(self) -> None:
        """``auto`` still produces a config, so ``none`` is the only disabling value."""
        tool_config = _build_tool_config(
            [ToolParam.model_validate(_ANTHROPIC_WEATHER_TOOL)],
            ToolChoiceAutoParam(type="auto"),
        )
        assert tool_config is not None
        assert tool_config["toolChoice"] == {"auto": {}}

    async def test_tool_choice_none_with_tool_history_still_succeeds(self) -> None:
        """Dropping the config leaves Converse a synthesized one for the history.

        Converse rejects ``toolUse``/``toolResult`` blocks without a
        ``toolConfig``. This is the case the audit judged unfixable; it is safe
        because the synthesis is shared by every route.
        """
        tool_config = _build_tool_config(
            [ToolParam.model_validate(_ANTHROPIC_WEATHER_TOOL)],
            ToolChoiceNoneParam(type="none"),
        )
        payload = await _prepare(_tool_call_history(), tool_config)
        assert payload["toolConfig"]["tools"] == [_SYNTHESIZED_WEATHER_SPEC]
        # 'none' must never force a tool call: no explicit toolChoice is emitted.
        assert "toolChoice" not in payload["toolConfig"]
