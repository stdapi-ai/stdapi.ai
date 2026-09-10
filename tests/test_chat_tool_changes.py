"""A conversation whose tool set changes between turns, on every chat dialect.

A client may add, remove or replace tools from one turn to the next while
replaying a history that still names the tools of the earlier turns.  The
contract every dialect shares is that the tools declared on *this* turn are the
only ones the model may call, whatever the history contains.

The one case that cannot be served that way is the empty tool set: it is
documented as a limitation on each dialect's page and pinned by
:class:`TestToolCallingTurnedOffMidConversation` below.

The backend can also be *told* about the change, through ``toolAddition`` and
``toolRemoval`` content blocks in a ``system``-role message.  That emission is
gated off on every model (:class:`TestToolSetChangeBlocksWhenEnabled`), so the
request a tool-set change produces is the one pinned here.

Ref: https://developers.openai.com/api/docs/guides/function-calling
     https://platform.claude.com/docs/en/docs/agents-and-tools/tool-use/overview
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     stdapi/models/chat/_default.py:ChatModel._prepare_converse_request
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

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


#: Failure message for a request carrying an announcement no model accepts.
_NO_BLOCKS_BY_DEFAULT = "the tool-set-change blocks stay off until a model takes them"


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


def _tool_set_change_blocks(captured: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the tool-set-change blocks the captured Converse request carries."""
    return [
        block
        for message in captured.get("messages", ())
        for block in message["content"]
        if "toolAddition" in block or "toolRemoval" in block
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
        assert not _tool_set_change_blocks(captured), _NO_BLOCKS_BY_DEFAULT

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
        assert not _tool_set_change_blocks(captured), _NO_BLOCKS_BY_DEFAULT

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
        assert not _tool_set_change_blocks(captured), _NO_BLOCKS_BY_DEFAULT

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
        assert not _tool_set_change_blocks(captured), _NO_BLOCKS_BY_DEFAULT

    async def test_the_whole_request_a_changed_tool_set_produces_is_pinned(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """Nothing is added to the request the tool-set change already produces.

        ``TOOL_SET_CHANGE_BLOCKS_SUPPORTED`` is off on every model, and the
        payload below is the one this conversation produced before that flag
        existed: the whole request is compared, so an announcement leaking into
        the default path fails here whatever shape it takes.

        Ref: stdapi/models/chat/_default.py:ChatModel._prepare_converse_request
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
        assert captured == {
            # placeholder: the region-specific model ID is injected on the call
            "modelId": "",
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": "What is the weather in Paris?"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": _CALL_ID,
                                "name": "get_weather",
                                "input": {"city": "Paris"},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": _CALL_ID,
                                "content": [{"text": "18C, sunny"}],
                            }
                        },
                        {"text": "Now what time is it in Paris?"},
                    ],
                },
            ],
            "inferenceConfig": {},
            "toolConfig": {
                "tools": [
                    {
                        "toolSpec": {
                            "name": "get_time",
                            "description": "function",
                            "inputSchema": {"json": _TIME_SCHEMA},
                        }
                    }
                ]
            },
        }


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


class _AnnouncingChatModel(ChatModel):
    """A chat model whose family has switched the tool-set-change blocks on."""

    __slots__ = ()

    TOOL_SET_CHANGE_BLOCKS_SUPPORTED: ClassVar[bool] = True


@pytest.mark.local
class TestToolSetChangeBlocksWhenEnabled:
    """A model that takes the blocks is told what changed, where that is legal.

    These are request-shape tests and can only ever be: no model accepts
    ``toolAddition`` or ``toolRemoval`` today.  Every Claude model answers
    *"This model doesn't support the toolRemoval field for system messages"*
    and every other family *"This model doesn't support system messages"*,
    measured in three regions on 2026-09-10, so the emission stays behind
    ``TOOL_SET_CHANGE_BLOCKS_SUPPORTED`` and no live assertion exists to
    write.  The absence of a live test here is that measurement, not an
    oversight.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         https://github.com/stdapi-ai/stdapi.ai/issues/104
         stdapi/models/chat/_default.py:ChatModel._req_announce_tool_set_change
    """

    async def test_a_replaced_tool_set_is_announced_before_the_last_assistant_turn(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """Both halves of the change are announced, and the turn order stays legal.

        A ``system``-role message is accepted only where it precedes an
        ``assistant`` message, and the conversation must still end on a
        ``user`` message.

        Ref: stdapi/models/chat/_default.py:ChatModel._req_announce_tool_set_change
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
        await _AnnouncingChatModel(_MODEL_ID).create_completion(request, "cmpl-1", 0)

        assert _tool_set_change_blocks(captured) == [
            {"toolAddition": {"tool": {"name": "get_time"}}},
            {"toolRemoval": {"tool": {"name": "get_weather"}}},
        ]
        assert _declared_tool_names(captured) == ["get_time"], (
            "announcing a removal must not re-declare the tool that was removed"
        )
        messages = captured["messages"]
        announced = [i for i, m in enumerate(messages) if m["role"] == "system"]
        assert len(announced) == 1, "one message carries the whole change"
        assert messages[announced[0] + 1]["role"] == "assistant"
        assert messages[-1]["role"] == "user"

    async def test_a_grown_tool_set_announces_only_the_addition(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A tool kept from the earlier turns is not announced as removed.

        Ref: stdapi/models/chat/_default.py:ChatModel._req_announce_tool_set_change
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
        await _AnnouncingChatModel(_MODEL_ID).create_completion(request, "cmpl-1", 0)
        assert _tool_set_change_blocks(captured) == [
            {"toolAddition": {"tool": {"name": "get_time"}}}
        ]

    async def test_messages_announces_the_change_the_same_way(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """The announcement is built from the Converse request, not the dialect.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_default.py:ChatModel._prepare_converse_request
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
        await _AnnouncingChatModel(_MODEL_ID).create_message(request, "msg-1")
        assert _tool_set_change_blocks(captured) == [
            {"toolAddition": {"tool": {"name": "get_time"}}},
            {"toolRemoval": {"tool": {"name": "get_weather"}}},
        ]

    async def test_a_conversation_that_called_no_tool_announces_nothing(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A history with no tool call is no evidence of an earlier tool set.

        The tools a history names are the ones that were *called*, so a
        conversation that called none says nothing about what was offered and
        gets no announcement — the tools declared here may have been declared
        all along.

        Ref: stdapi/models/chat/_default.py:ChatModel._req_announce_tool_set_change
        """
        del request_log
        captured: dict[str, Any] = {}
        monkeypatch.setattr(ChatModel, "converse", _capturing_converse(captured))
        request = CompletionCreateParams.model_validate(
            {
                "model": _MODEL_ID,
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                    {"role": "user", "content": "What time is it in Paris?"},
                ],
                "tools": [_openai_tool("get_time", _TIME_SCHEMA)],
            }
        )
        await _AnnouncingChatModel(_MODEL_ID).create_completion(request, "cmpl-1", 0)
        assert not _tool_set_change_blocks(captured)
        assert not [m for m in captured["messages"] if m["role"] == "system"]


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
