"""A conversation whose tool set changes between turns, on every chat dialect.

A client may add, remove or replace tools from one turn to the next while
replaying a history that still names the tools of the earlier turns.  The
contract every dialect shares is that the tools declared on *this* turn are the
only ones the model may call, whatever the history contains.

The one case that cannot be served that way is the empty tool set: it is
documented as a limitation on each dialect's page and pinned by
:class:`TestToolCallingTurnedOffMidConversation` below.

Ref: https://developers.openai.com/api/docs/guides/function-calling
     https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/overview
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     stdapi/models/chat/_default.py:ChatModel._prepare_converse_request
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from stdapi.models.chat._default import ChatModel
from stdapi.types.anthropic_messages import MessageCreateParams
from stdapi.types.openai_chat_completions import CompletionCreateParams
from stdapi.types.openai_responses import ResponseCreateParams

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from anthropic import Anthropic
    from openai import OpenAI

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef

#: Cheap Converse model used to build request payloads; construction is side-effect free.
_MODEL_ID = "amazon.nova-micro-v1:0"

#: Identifier shared by the replayed tool call and its result.
_CALL_ID = "call_paris_weather"

#: Tool the first turn declared and the later turns drop.
_WEATHER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}

#: Tool the later turns declare in its place.
_TIME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}


def _capturing_converse(
    captured: dict[str, Any],
) -> Callable[[ChatModel, ConverseRequestBaseTypeDef], Awaitable[dict[str, Any]]]:
    """Build a ``ChatModel.converse`` replacement recording the request body.

    Args:
        captured: Mapping updated with the Converse request body.

    Returns:
        A coroutine function returning a canned one-token Converse response.
    """

    async def fake_converse(
        _self: ChatModel, request: ConverseRequestBaseTypeDef
    ) -> dict[str, Any]:
        captured.update(request)
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }

    return fake_converse


def _declared_tool_names(captured: dict[str, Any]) -> list[str]:
    """Return the tool names the captured Converse request declares."""
    return [
        entry["toolSpec"]["name"]
        for entry in captured.get("toolConfig", {}).get("tools", ())
        if "toolSpec" in entry
    ]


def _openai_tool(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Return a Chat Completions function tool declaration."""
    return {"type": "function", "function": {"name": name, "parameters": schema}}


def _openai_history() -> list[dict[str, Any]]:
    """Return a Chat Completions history whose only tool call is ``get_weather``."""
    return [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": _CALL_ID,
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "Paris"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": _CALL_ID, "content": "18C, sunny"},
        {"role": "user", "content": "Now what time is it in Paris?"},
    ]


def _anthropic_history() -> list[dict[str, Any]]:
    """Return a Messages history whose only tool use is ``get_weather``."""
    return [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": _CALL_ID,
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": _CALL_ID,
                    "content": "18C, sunny",
                },
                {"type": "text", "text": "Now what time is it in Paris?"},
            ],
        },
    ]


def _responses_history() -> list[dict[str, Any]]:
    """Return a Responses input whose only function call is ``get_weather``."""
    return [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "type": "function_call",
            "call_id": _CALL_ID,
            "name": "get_weather",
            "arguments": json.dumps({"city": "Paris"}),
        },
        {"type": "function_call_output", "call_id": _CALL_ID, "output": "18C, sunny"},
        {"role": "user", "content": "Now what time is it in Paris?"},
    ]


@pytest.mark.local
class TestToolSetChangedBetweenTurns:
    """The tools declared on a turn are the only ones the model is offered.

    The history replayed on each request still names the tool of the first
    turn, so the request the backend receives is the only place the change is
    observable.

    Ref: https://developers.openai.com/api/docs/guides/function-calling
         stdapi/models/chat/_default.py:ChatModel._prepare_converse_request
    """

    async def test_chat_completions_drops_the_tool_the_turn_no_longer_declares(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """Chat Completions: a replaced tool set replaces the declared tools.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             stdapi/models/chat/_adapters/_openai_chat_completion.py:build_tool_config
        """
        del request_log
        captured: dict[str, Any] = {}
        monkeypatch.setattr(ChatModel, "converse", _capturing_converse(captured))
        request = CompletionCreateParams.model_validate(
            {
                "model": _MODEL_ID,
                "messages": _openai_history(),
                "tools": [_openai_tool("get_time", _TIME_SCHEMA)],
            }
        )
        await ChatModel(_MODEL_ID).create_completion(request, "cmpl-1", 0)
        assert _declared_tool_names(captured) == ["get_time"]

    async def test_chat_completions_keeps_a_tool_added_mid_conversation(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """Chat Completions: a grown tool set reaches the model whole.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             stdapi/models/chat/_adapters/_openai_chat_completion.py:build_tool_config
        """
        del request_log
        captured: dict[str, Any] = {}
        monkeypatch.setattr(ChatModel, "converse", _capturing_converse(captured))
        request = CompletionCreateParams.model_validate(
            {
                "model": _MODEL_ID,
                "messages": _openai_history(),
                "tools": [
                    _openai_tool("get_weather", _WEATHER_SCHEMA),
                    _openai_tool("get_time", _TIME_SCHEMA),
                ],
            }
        )
        await ChatModel(_MODEL_ID).create_completion(request, "cmpl-1", 0)
        assert _declared_tool_names(captured) == ["get_weather", "get_time"]

    async def test_messages_drops_the_tool_the_turn_no_longer_declares(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """Messages: a replaced tool set replaces the declared tools.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_adapters/_anthropic_message.py:_build_tool_config
        """
        del request_log
        captured: dict[str, Any] = {}
        monkeypatch.setattr(ChatModel, "converse", _capturing_converse(captured))
        request = MessageCreateParams.model_validate(
            {
                "model": _MODEL_ID,
                "max_tokens": 16,
                "messages": _anthropic_history(),
                "tools": [{"name": "get_time", "input_schema": _TIME_SCHEMA}],
            }
        )
        await ChatModel(_MODEL_ID).create_message(request, "msg-1")
        assert _declared_tool_names(captured) == ["get_time"]

    async def test_responses_drops_the_tool_the_turn_no_longer_declares(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """Responses: a replaced tool set replaces the declared tools.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/models/chat/_adapters/_openai_responses.py:build_tool_config
        """
        del request_log
        captured: dict[str, Any] = {}
        monkeypatch.setattr(ChatModel, "converse", _capturing_converse(captured))
        request = ResponseCreateParams.model_validate(
            {
                "model": _MODEL_ID,
                "input": _responses_history(),
                "tools": [
                    {
                        "type": "function",
                        "name": "get_time",
                        "parameters": _TIME_SCHEMA,
                        "strict": False,
                    }
                ],
            }
        )
        await ChatModel(_MODEL_ID).create_response(request, "resp-1", 0.0)
        assert _declared_tool_names(captured) == ["get_time"]


@pytest.mark.local
class TestToolCallingTurnedOffMidConversation:
    """Turning tool calling off after a tool call is a documented limitation.

    A request whose history carries a tool call is refused outright unless it
    declares at least one tool, and the tool-selection values available on this
    backend cannot express "declared but not callable".  A turn that declares
    no tool therefore falls back to the tools its history names, and the model
    may call one of them again — pinned here so the documented behaviour cannot
    change unnoticed.

    Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
         stdapi/models/chat/_adapters/_anthropic_message.py:_synthesize_tool_config_from_history
    """

    async def test_tool_choice_none_falls_back_to_the_tools_in_history(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """``tool_choice='none'`` after a tool call still declares that tool.

        Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py:build_tool_config
        """
        del request_log
        captured: dict[str, Any] = {}
        monkeypatch.setattr(ChatModel, "converse", _capturing_converse(captured))
        request = CompletionCreateParams.model_validate(
            {
                "model": _MODEL_ID,
                "messages": _openai_history(),
                "tools": [_openai_tool("get_time", _TIME_SCHEMA)],
                "tool_choice": "none",
            }
        )
        await ChatModel(_MODEL_ID).create_completion(request, "cmpl-1", 0)
        assert _declared_tool_names(captured) == ["get_weather"], (
            "the declared tools are dropped, and the history supplies the "
            "configuration the backend requires"
        )
        assert "toolChoice" not in captured["toolConfig"], (
            "no tool may ever be forced on a turn that asked for none"
        )

    async def test_no_tools_and_no_tool_history_declares_nothing(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """Without a tool call in history, dropping the tools really drops them.

        This is the control for the limitation above: the fallback is caused by
        the history, not by the request.

        Ref: stdapi/models/chat/_default.py:ChatModel._prepare_converse_request
        """
        del request_log
        captured: dict[str, Any] = {}
        monkeypatch.setattr(ChatModel, "converse", _capturing_converse(captured))
        request = CompletionCreateParams.model_validate(
            {
                "model": _MODEL_ID,
                "messages": [{"role": "user", "content": "What time is it in Paris?"}],
                "tools": [_openai_tool("get_time", _TIME_SCHEMA)],
                "tool_choice": "none",
            }
        )
        await ChatModel(_MODEL_ID).create_completion(request, "cmpl-1", 0)
        assert "toolConfig" not in captured


class TestToolSetChangesReachTheModel:
    """A real conversation whose tool set changes is answered from the new set.

    The turn forces a tool call while declaring only the replacement tool, so
    the answer names the replacement or the request has not honoured the
    change.

    Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
         https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/overview
    """

    def test_chat_completions_replaced_tool_set(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """The replayed tool is gone and the forced call names the new tool.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        """
        response = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=chat_model,
            messages=_openai_history(),
            tools=[_openai_tool("get_time", _TIME_SCHEMA)],
            tool_choice="required",
        )
        (choice,) = response.choices
        assert choice.message.tool_calls, "tool_choice='required' must call a tool"
        assert {call.function.name for call in choice.message.tool_calls} == {
            "get_time"
        }, "only the tools declared on this turn may be called"

    def test_messages_replaced_tool_set(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """The Messages dialect honours the same change.

        Ref: https://platform.claude.com/docs/en/api/messages
        """
        message = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_basic_model,
            max_tokens=64,
            messages=_anthropic_history(),
            tools=[{"name": "get_time", "input_schema": _TIME_SCHEMA}],
            tool_choice={"type": "any"},
        )
        called = {block.name for block in message.content if block.type == "tool_use"}
        assert called == {"get_time"}, (
            "only the tools declared on this turn may be called"
        )
