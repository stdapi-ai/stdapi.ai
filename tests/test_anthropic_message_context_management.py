"""Context editing (``context_management``) on the Anthropic Messages route.

Upstream contract first: every test outside the ``gateway`` class runs unchanged
against the vendor API (``--use-official-api``), then against the gateway.
The history carries three completed tool round trips with filler results, so a
``clear_tool_uses_20250919`` edit triggered at 100 input tokens and keeping one
tool use clears exactly two of them on every target.

Ref: https://platform.claude.com/docs/en/build-with-claude/context-editing
     https://platform.claude.com/docs/en/api/beta/messages/create
     https://platform.claude.com/docs/en/api/beta/messages/count_tokens
     stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_context_management
     stdapi/models/chat/_adapters/_anthropic_message.py:format_response
"""

from __future__ import annotations

from contextlib import contextmanager
from json import loads
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin, get_type_hints

import pytest
from anthropic import BadRequestError
from anthropic.types.beta.beta_clear_thinking_20251015_edit_param import (
    BetaClearThinking20251015EditParam as SdkClearThinking,
)
from anthropic.types.beta.beta_clear_tool_uses_20250919_edit_param import (
    BetaClearToolUses20250919EditParam as SdkClearToolUses,
)
from anthropic.types.beta.beta_context_management_config_param import Edit as SdkEdit
from anthropic.types.beta.beta_context_management_response import (
    AppliedEdit as SdkAppliedEdit,
)
from botocore.exceptions import ClientError

import stdapi.models.chat._adapters._anthropic_message as anthropic_message_adapter
from stdapi.aws_bedrock_mantle import mantle_request_headers
from stdapi.models.chat import get_chat_model
from stdapi.models.chat._adapters._anthropic_message import (
    count_tokens_via_bedrock,
    format_response,
    format_stream,
)
from stdapi.models.chat._default import ChatModel
from stdapi.models.chat._mantle import get_mantle_chat_model
from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel
from stdapi.models.chat._mantle._default import messages_request_headers
from stdapi.monitoring import REQUEST
from stdapi.routes import anthropic_messages
from stdapi.types.anthropic_messages import (
    APPLIED_EDIT_TYPES,
    ClearThinking20251015EditParam,
    ClearToolUses20250919EditParam,
    ContextManagementConfigParam,
    ContextManagementEditParam,
    ContextManagementResponse,
    Message,
    MessageCountTokensParams,
    MessageCreateParams,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from anthropic import Anthropic

#: Beta flag the vendor requires for ``context_management``.
_BETA = "context-management-2025-06-27"

#: Client tool the recorded history calls.
_WEATHER_TOOL: Any = {
    "name": "get_weather",
    "description": "Get the weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def _tool_history(round_trips: int = 3) -> list[dict[str, Any]]:
    """Build a conversation with *round_trips* completed tool calls and a final ask.

    Args:
        round_trips: Number of ``tool_use``/``tool_result`` pairs.

    Returns:
        Messages ending on a user turn.
    """
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What is the weather in several cities?"}
    ]
    for index in range(round_trips):
        tool_id = f"toolu_0{index}"
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": "get_weather",
                        "input": {"city": f"City{index}"},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": f"City{index}: sunny, 2{index} degrees. "
                        + "filler text " * 40,
                    }
                ],
            }
        )
    messages[-1]["content"].append({"type": "text", "text": "Summarize in five words."})
    return messages


#: Conversation with three tool round trips (~1k input tokens).
_HISTORY: Any = _tool_history()

#: Tool-result clearing triggered well below the history size, keeping one tool use.
_CLEAR_TOOL_USES: Any = {
    "edits": [
        {
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": 100},
            "keep": {"type": "tool_uses", "value": 1},
        }
    ]
}

#: Tool-result clearing at the default trigger (100,000 tokens), never reached here.
_CLEAR_TOOL_USES_DEFAULT: Any = {"edits": [{"type": "clear_tool_uses_20250919"}]}

#: Thinking budget for the thinking-clearing tests (the minimum accepted).
_THINKING_BUDGET = 1024


class TestContextManagement:
    """Context editing on a Claude model, against the upstream contract."""

    def test_clear_tool_uses_reports_applied_edit(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A triggered ``clear_tool_uses_20250919`` edit is reported in ``applied_edits``.

        Three tool uses with one kept leaves two cleared on every target; the
        cleared token count depends on the tokenizer, so only its sign is asserted.

        Ref: https://platform.claude.com/docs/en/build-with-claude/context-editing
             stdapi/models/chat/_adapters/_anthropic_message.py:format_response
        """
        message = anthropic_client.beta.messages.create(
            model=anthropic_chat_model,
            max_tokens=20,
            messages=_HISTORY,
            tools=[_WEATHER_TOOL],
            context_management=_CLEAR_TOOL_USES,
            betas=[_BETA],
        )

        assert message.context_management is not None
        (edit,) = message.context_management.applied_edits
        assert edit.type == "clear_tool_uses_20250919"
        assert edit.cleared_tool_uses == 2
        assert edit.cleared_input_tokens > 0

    def test_clear_tool_uses_streams_applied_edit_on_message_delta(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Streaming carries the applied edits on the final ``message_delta`` event only.

        Upstream puts ``context_management`` top-level on ``message_delta``, next
        to ``delta`` and ``usage``; ``message_start`` carries no applied edit.

        Ref: https://platform.claude.com/docs/en/build-with-claude/context-editing
             stdapi/models/chat/_adapters/_anthropic_message.py:_make_message_delta_event
        """
        stream = anthropic_client.beta.messages.create(
            model=anthropic_chat_model,
            max_tokens=20,
            messages=_HISTORY,
            tools=[_WEATHER_TOOL],
            context_management=_CLEAR_TOOL_USES,
            betas=[_BETA],
            stream=True,
        )
        events = list(stream)

        (start,) = [event for event in events if event.type == "message_start"]
        assert not (
            start.message.context_management
            and start.message.context_management.applied_edits
        )
        (delta,) = [event for event in events if event.type == "message_delta"]
        assert delta.context_management is not None
        (edit,) = delta.context_management.applied_edits
        assert edit.type == "clear_tool_uses_20250919"
        assert edit.cleared_tool_uses == 2
        assert edit.cleared_input_tokens > 0

    def test_untriggered_edit_reports_no_applied_edit(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """An edit whose trigger is not reached answers an empty ``applied_edits`` list.

        The default trigger is 100,000 input tokens, far above this history.

        Ref: https://platform.claude.com/docs/en/build-with-claude/context-editing
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._prepare_converse_request
        """
        message = anthropic_client.beta.messages.create(
            model=anthropic_chat_model,
            max_tokens=20,
            messages=_HISTORY,
            tools=[_WEATHER_TOOL],
            context_management=_CLEAR_TOOL_USES_DEFAULT,
            betas=[_BETA],
        )

        assert message.context_management is not None
        assert message.context_management.applied_edits == []

    def test_count_tokens_counts_the_edited_prompt(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """``count_tokens`` counts after the edits and reports the unedited count.

        ``original_input_tokens`` equals the count of the same request without
        ``context_management``; absolute numbers differ between backends, so only
        the relations are asserted.

        Ref: https://platform.claude.com/docs/en/api/beta/messages/count_tokens
             stdapi/models/chat/_adapters/_anthropic_message.py:count_tokens_via_bedrock
        """
        edited = anthropic_client.beta.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=_HISTORY,
            tools=[_WEATHER_TOOL],
            context_management=_CLEAR_TOOL_USES,
            betas=[_BETA],
        )
        unedited = anthropic_client.beta.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=_HISTORY,
            tools=[_WEATHER_TOOL],
            betas=[_BETA],
        )

        assert edited.context_management is not None
        assert edited.context_management.original_input_tokens == unedited.input_tokens
        assert 0 < edited.input_tokens < unedited.input_tokens

    def test_count_tokens_rejects_an_invalid_edit(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """``count_tokens`` refuses an edit the model would refuse, with the same 400.

        Ref: https://platform.claude.com/docs/en/api/beta/messages/count_tokens
             stdapi/models/chat/_adapters/_anthropic_message.py:count_tokens_via_bedrock
        """
        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.beta.messages.count_tokens(
                model=anthropic_count_tokens_model,
                messages=[{"role": "user", "content": "Say OK."}],
                context_management={"edits": [{"type": "clear_thinking_20251015"}]},
                betas=[_BETA],
            )

        body = excinfo.value.body
        assert excinfo.value.status_code == 400
        assert isinstance(body, dict)
        assert body["error"]["type"] == "invalid_request_error"
        assert "requires `thinking`" in body["error"]["message"]

    def test_clear_thinking_reports_cleared_turns(
        self, anthropic_client: Anthropic, anthropic_chat_reasoning_model: str
    ) -> None:
        """``clear_thinking_20251015`` keeping one turn clears the older thinking turns.

        The first turn sends the shape Claude Code sends whenever thinking is on
        (``keep: "all"``), which must be served and report no cleared turn. The
        third turn keeps one thinking turn out of two, so at least one is cleared.

        Ref: https://platform.claude.com/docs/en/build-with-claude/context-editing
             stdapi/types/anthropic_messages.py:ClearThinking20251015EditParam
        """
        messages: list[Any] = []
        prompts = (
            "What is 17*23? Answer with the number only.",
            "Now add 49. Number only.",
            "Now subtract 100. Number only.",
        )
        keep_all: Any = {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}
        keep_one: Any = {
            "edits": [
                {
                    "type": "clear_thinking_20251015",
                    "keep": {"type": "thinking_turns", "value": 1},
                }
            ]
        }
        last = None
        for index, prompt in enumerate(prompts):
            messages.append({"role": "user", "content": prompt})
            last = anthropic_client.beta.messages.create(
                model=anthropic_chat_reasoning_model,
                max_tokens=_THINKING_BUDGET + 100,
                thinking={"type": "enabled", "budget_tokens": _THINKING_BUDGET},
                messages=messages,
                context_management=keep_all if index < 2 else keep_one,
                betas=[_BETA],
            )
            if index == 0:
                assert last.context_management is not None
                assert not any(
                    edit.type == "clear_thinking_20251015"
                    and edit.cleared_thinking_turns
                    for edit in last.context_management.applied_edits
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": [block.model_dump() for block in last.content],
                }
            )

        assert last is not None
        assert last.context_management is not None
        cleared = [
            edit
            for edit in last.context_management.applied_edits
            if edit.type == "clear_thinking_20251015"
        ]
        assert cleared
        assert cleared[0].cleared_thinking_turns >= 1

    def test_clear_thinking_without_thinking_is_rejected(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """``clear_thinking_20251015`` without thinking enabled is a 400.

        Ref: https://platform.claude.com/docs/en/build-with-claude/context-editing
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_context_management
        """
        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.beta.messages.create(
                model=anthropic_chat_model,
                max_tokens=20,
                messages=[{"role": "user", "content": "Say OK."}],
                context_management={"edits": [{"type": "clear_thinking_20251015"}]},
                betas=[_BETA],
            )

        body = excinfo.value.body
        assert excinfo.value.status_code == 400
        assert isinstance(body, dict)
        assert body["error"]["type"] == "invalid_request_error"
        assert "requires `thinking`" in body["error"]["message"]

    def test_unknown_edit_type_is_rejected(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """An edit type the API does not define is a 400 naming the offending tag.

        Ref: https://platform.claude.com/docs/en/api/beta/messages/create
             stdapi/types/anthropic_messages.py:ContextManagementConfigParam
        """
        unknown_edit: Any = {"edits": [{"type": "clear_everything_20990101"}]}
        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.beta.messages.create(
                model=anthropic_chat_model,
                max_tokens=20,
                messages=[{"role": "user", "content": "Say OK."}],
                context_management=unknown_edit,
                betas=[_BETA],
            )

        body = excinfo.value.body
        assert excinfo.value.status_code == 400
        assert isinstance(body, dict)
        assert body["error"]["type"] == "invalid_request_error"
        assert "clear_everything_20990101" in body["error"]["message"]

    @pytest.mark.parametrize(
        ("edits", "message"),
        [
            (
                [
                    {
                        "type": "clear_tool_uses_20250919",
                        "trigger": {"type": "input_tokens", "value": 0},
                    }
                ],
                "greater than or equal to 1",
            ),
            (
                [
                    {"type": "clear_tool_uses_20250919"},
                    {"type": "clear_tool_uses_20250919"},
                ],
                "duplicate context management strategy",
            ),
            (
                [
                    {"type": "clear_tool_uses_20250919"},
                    {"type": "clear_thinking_20251015"},
                ],
                "must be the first strategy",
            ),
        ],
        ids=["trigger-below-one", "duplicate-strategy", "thinking-edit-not-first"],
    )
    def test_invalid_edits_are_rejected(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_reasoning_model: str,
        edits: list[dict[str, object]],
        message: str,
    ) -> None:
        """An edit list the API cannot apply is a 400 ``invalid_request_error``.

        Thinking is enabled so that only the rule under test is broken. Each
        request is refused before inference, so nothing is billed.

        Ref: https://platform.claude.com/docs/en/build-with-claude/context-editing
             stdapi/types/anthropic_messages.py:ContextManagementConfigParam
        """
        context_management: Any = {"edits": edits}
        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.beta.messages.create(
                model=anthropic_chat_reasoning_model,
                max_tokens=_THINKING_BUDGET + 100,
                thinking={"type": "enabled", "budget_tokens": _THINKING_BUDGET},
                messages=[{"role": "user", "content": "Say OK."}],
                context_management=context_management,
                betas=[_BETA],
            )

        body = excinfo.value.body
        assert excinfo.value.status_code == 400
        assert isinstance(body, dict)
        assert body["error"]["type"] == "invalid_request_error"
        assert message in body["error"]["message"]

    def test_beta_flag_alone_reports_no_applied_edit(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """The beta flag without ``context_management`` answers an empty ``applied_edits``.

        Ref: https://platform.claude.com/docs/en/build-with-claude/context-editing
             stdapi/models/chat/_adapters/_anthropic_message.py:format_response
        """
        message = anthropic_client.beta.messages.create(
            model=anthropic_chat_model,
            max_tokens=5,
            messages=[{"role": "user", "content": "Say OK."}],
            betas=[_BETA],
        )

        assert message.context_management is not None
        assert message.context_management.applied_edits == []

    def test_untriggered_edit_streams_no_applied_edit(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A streamed untriggered edit reports an empty list on ``message_delta``.

        Ref: https://platform.claude.com/docs/en/build-with-claude/context-editing
             stdapi/models/chat/_adapters/_anthropic_message.py:_make_message_delta_event
        """
        stream = anthropic_client.beta.messages.create(
            model=anthropic_chat_model,
            max_tokens=5,
            messages=[{"role": "user", "content": "Say OK."}],
            context_management=_CLEAR_TOOL_USES_DEFAULT,
            betas=[_BETA],
            stream=True,
        )

        (delta,) = [event for event in stream if event.type == "message_delta"]
        assert delta.context_management is not None
        assert delta.context_management.applied_edits == []

    def test_context_management_without_beta_header(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_official_api: bool,
    ) -> None:
        """``context_management`` sent without the beta header.

        The vendor rejects the field unless ``context-management-2025-06-27`` is
        listed in ``anthropic-beta`` (400 ``Extra inputs are not permitted``). The
        gateway adds the flag itself whenever the field is set, like it does for
        the memory tool, so header-less clients are served and the edit applies.

        Ref: https://platform.claude.com/docs/en/api/beta-headers
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_context_management
        """
        kwargs: dict[str, Any] = {
            "model": anthropic_chat_model,
            "max_tokens": 20,
            "messages": _HISTORY,
            "tools": [_WEATHER_TOOL],
            "extra_body": {"context_management": _CLEAR_TOOL_USES},
        }
        if use_official_api:
            with pytest.raises(BadRequestError) as excinfo:
                anthropic_client.messages.create(**kwargs)
            body = excinfo.value.body
            assert isinstance(body, dict)
            assert body["error"]["type"] == "invalid_request_error"
            assert "Extra inputs are not permitted" in body["error"]["message"]
            return

        raw: Any = anthropic_client.messages.with_raw_response.create(**kwargs).json()
        (edit,) = raw["context_management"]["applied_edits"]
        assert edit["type"] == "clear_tool_uses_20250919"
        assert edit["cleared_tool_uses"] == 2


#: Mantle-only undated Claude ID (native Messages passthrough).
_CLAUDE_MANTLE = "anthropic.claude-haiku-4-5"

#: Non-Claude model on the Anthropic route.
_NON_CLAUDE_MODEL = "amazon.nova-micro-v1:0"


@pytest.mark.gateway("Bedrock Mantle and non-Claude models do not exist upstream")
class TestContextManagementGateway:
    """Context editing on the gateway-only backends."""

    def test_mantle_claude_reports_applied_edit(
        self, anthropic_client: Anthropic
    ) -> None:
        """A Mantle-served Claude model applies the edit and reports it, like the vendor.

        Ref: https://platform.claude.com/docs/en/build-with-claude/context-editing
             stdapi/models/chat/_mantle/_default.py:ChatModel.create_message
        """
        message = anthropic_client.beta.messages.create(
            model=_CLAUDE_MANTLE,
            max_tokens=20,
            messages=_HISTORY,
            tools=[_WEATHER_TOOL],
            context_management=_CLEAR_TOOL_USES,
        )

        assert message.context_management is not None
        (edit,) = message.context_management.applied_edits
        assert edit.type == "clear_tool_uses_20250919"
        assert edit.cleared_tool_uses == 2

    def test_mantle_claude_streams_applied_edit(
        self, anthropic_client: Anthropic
    ) -> None:
        """A Mantle-served Claude stream carries the edits on ``message_delta``.

        Ref: https://platform.claude.com/docs/en/build-with-claude/context-editing
             stdapi/models/chat/_mantle/_default.py:ChatModel.create_message
        """
        events = list(
            anthropic_client.beta.messages.create(
                model=_CLAUDE_MANTLE,
                max_tokens=20,
                messages=_HISTORY,
                tools=[_WEATHER_TOOL],
                context_management=_CLEAR_TOOL_USES,
                stream=True,
            )
        )

        (delta,) = [event for event in events if event.type == "message_delta"]
        assert delta.context_management is not None
        (edit,) = delta.context_management.applied_edits
        assert edit.type == "clear_tool_uses_20250919"
        assert edit.cleared_tool_uses == 2

    def test_mantle_claude_count_tokens(self, anthropic_client: Anthropic) -> None:
        """Mantle ``count_tokens`` counts the edited prompt and reports the original.

        Ref: https://platform.claude.com/docs/en/api/beta/messages/count_tokens
             stdapi/routes/anthropic_messages.py:_count_tokens_via_mantle
        """
        count = anthropic_client.beta.messages.count_tokens(
            model=_CLAUDE_MANTLE,
            messages=_HISTORY,
            tools=[_WEATHER_TOOL],
            context_management=_CLEAR_TOOL_USES,
        )

        assert count.context_management is not None
        assert 0 < count.input_tokens < count.context_management.original_input_tokens

    def test_non_claude_model_ignores_context_management(
        self, anthropic_client: Anthropic
    ) -> None:
        """A non-Claude model serves the request unedited and reports no edits.

        Claude Code sends ``context_management`` whenever thinking is on, so the
        field is accepted and ignored rather than failing the turn.

        Ref: stdapi/models/chat/_default.py:ChatModel._req_configure_context_management
        """
        message = anthropic_client.beta.messages.create(
            model=_NON_CLAUDE_MODEL,
            max_tokens=20,
            messages=_HISTORY,
            tools=[_WEATHER_TOOL],
            context_management=_CLEAR_TOOL_USES,
            betas=[_BETA],
        )

        assert message.content
        assert message.context_management is None


#: Claude model on the Converse path, for the offline request-building tests.
_CLAUDE_CONVERSE = "anthropic.claude-haiku-4-5-20251001-v1:0"

#: Non-Claude Mantle model, served by converting Messages to Chat Completions.
_NON_CLAUDE_MANTLE = "google.gemma-3-4b-it"

#: A reported applied edit, as the Converse and Messages APIs return it.
_APPLIED: dict[str, Any] = {
    "applied_edits": [
        {
            "type": "clear_tool_uses_20250919",
            "cleared_input_tokens": 252,
            "cleared_tool_uses": 2,
        }
    ]
}


def _literals(hint: Any) -> set[str]:  # noqa: ANN401
    """Return the string literals a type hint allows, unwrapping ``Required``/``Annotated``.

    Args:
        hint: A type hint read with ``include_extras=True``.

    Returns:
        The literal values it names.
    """
    if get_origin(hint) is Literal:
        return set(get_args(hint))
    return {value for arg in get_args(hint) for value in _literals(arg)}


def _converse_model(model_id: str) -> ChatModel:
    """Return the Converse chat model implementation for *model_id*."""
    model = get_chat_model(model_id)
    assert isinstance(model, ChatModel)
    return model


@contextmanager
def _request_headers(headers: dict[str, str]) -> Iterator[None]:
    """Bind the ``REQUEST`` contextvar to a stub request carrying *headers*."""
    token = REQUEST.set(SimpleNamespace(headers=headers))  # type: ignore[arg-type]
    try:
        yield
    finally:
        REQUEST.reset(token)


def _message_request(model: str, **fields: Any) -> MessageCreateParams:  # noqa: ANN401
    """Build a one-turn message request for *model* with extra *fields*."""
    return MessageCreateParams.model_validate(
        {
            "model": model,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "hi"}],
            **fields,
        }
    )


@pytest.mark.local
@pytest.mark.usefixtures("request_log")
class TestContextManagementOffline:
    """Request building, response mapping and parity, without any AWS call."""

    def test_edit_types_match_the_sdk_without_compaction(self) -> None:
        """The request edits and applied edits mirror the SDK's, compaction excluded.

        ``compact_20260112`` is a separate feature with its own beta, so it is
        rejected as an unknown edit type like the vendor does without that beta.

        Ref: https://platform.claude.com/docs/en/api/beta/messages/create
             stdapi/types/anthropic_messages.py:ContextManagementEditParam
        """
        sdk_edits = {
            value
            for member in get_args(SdkEdit)
            for value in _literals(get_type_hints(member, include_extras=True)["type"])
        }
        ours = {
            value
            for member in get_args(get_args(ContextManagementEditParam)[0])
            for value in _literals(member.model_fields["type"].annotation)
        }
        sdk_applied = {
            value
            for member in get_args(get_args(SdkAppliedEdit)[0])
            for value in _literals(member.model_fields["type"].annotation)
        }

        assert ours
        assert ours == sdk_edits - {"compact_20260112"}
        assert sdk_applied == APPLIED_EDIT_TYPES

    @pytest.mark.parametrize(
        ("ours", "sdk"),
        [
            (ClearToolUses20250919EditParam, SdkClearToolUses),
            (ClearThinking20251015EditParam, SdkClearThinking),
        ],
    )
    def test_edit_fields_match_the_sdk(self, ours: type[Any], sdk: type[Any]) -> None:
        """Every edit option the SDK sends is declared, so none is silently dropped.

        Ref: https://platform.claude.com/docs/en/build-with-claude/context-editing
             stdapi/types/anthropic_messages.py:ClearToolUses20250919EditParam
        """
        assert set(ours.model_fields) == set(get_type_hints(sdk))

    @pytest.mark.parametrize(
        "keep", ["all", {"type": "all"}, {"type": "thinking_turns", "value": 2}]
    )
    def test_clear_thinking_keep_forms(self, keep: Any) -> None:  # noqa: ANN401
        """The three documented ``keep`` forms of ``clear_thinking_20251015`` validate.

        Ref: https://platform.claude.com/docs/en/build-with-claude/context-editing
             stdapi/types/anthropic_messages.py:ClearThinking20251015EditParam
        """
        config = ContextManagementConfigParam.model_validate(
            {"edits": [{"type": "clear_thinking_20251015", "keep": keep}]}
        )
        assert config.model_dump(mode="json", exclude_none=True) == {
            "edits": [{"type": "clear_thinking_20251015", "keep": keep}]
        }

    async def test_claude_forwards_edits_with_the_beta_flag(self) -> None:
        """A Claude request carries the edits, the beta flag and the response path.

        Ref: stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_context_management
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._prepare_converse_request
        """
        request = _message_request(
            _CLAUDE_CONVERSE, context_management=_CLEAR_TOOL_USES
        )
        with _request_headers({}):
            payload, _ = await _converse_model(_CLAUDE_CONVERSE).build_message_request(
                request
            )

        fields = payload["additionalModelRequestFields"]
        assert fields["context_management"] == _CLEAR_TOOL_USES
        assert fields["anthropic_beta"] == [_BETA]
        assert list(payload["additionalModelResponseFieldPaths"]) == [
            "/context_management"
        ]

    async def test_beta_header_alone_requests_the_applied_edits(self) -> None:
        """The beta flag sent as a header also makes the applied edits come back.

        Upstream reports ``context_management`` whenever the beta is on, which
        is also how its default thinking clearing is reported.

        Ref: https://platform.claude.com/docs/en/api/beta-headers
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._prepare_converse_request
        """
        with _request_headers(
            {"anthropic-beta": f"interleaved-thinking-2025-05-14, {_BETA}"}
        ):
            payload, _ = await _converse_model(_CLAUDE_CONVERSE).build_message_request(
                _message_request(_CLAUDE_CONVERSE)
            )

        assert "context_management" not in payload["additionalModelRequestFields"]
        assert list(payload["additionalModelResponseFieldPaths"]) == [
            "/context_management"
        ]

    async def test_no_beta_requests_no_response_path(self) -> None:
        """Without the beta flag no response path is requested.

        Ref: stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._prepare_converse_request
        """
        with _request_headers({}):
            payload, _ = await _converse_model(_CLAUDE_CONVERSE).build_message_request(
                _message_request(_CLAUDE_CONVERSE)
            )

        assert "additionalModelResponseFieldPaths" not in payload

    async def test_non_claude_model_ignores_edits_with_one_warning(
        self, request_log: dict[str, Any]
    ) -> None:
        """A non-Claude model receives no edit and the operator is warned once.

        Ref: stdapi/models/chat/_default.py:ChatModel._req_configure_context_management
        """
        model = _converse_model(_NON_CLAUDE_MODEL)
        request = _message_request(
            _NON_CLAUDE_MODEL, context_management=_CLEAR_TOOL_USES
        )
        with _request_headers({}):
            payload, _ = await model.build_message_request(request)
            await model.build_message_request(request)

        fields = payload.get("additionalModelRequestFields") or {}
        assert "context_management" not in fields
        assert "anthropic_beta" not in fields
        assert "additionalModelResponseFieldPaths" not in payload
        warnings = [
            detail
            for detail in request_log.get("error_detail") or ()
            if "context_management" in str(detail)
        ]
        assert len(warnings) == 1

    async def test_format_response_maps_applied_edits(self) -> None:
        """The applied edits reported by the model become ``Message.context_management``.

        Ref: stdapi/models/chat/_adapters/_anthropic_message.py:format_response
        """
        args: tuple[Any, ...] = (
            [{"text": "ok"}],
            "end_turn",
            {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "msg_1",
            "model-x",
            None,
            lambda *_: None,
        )
        message = await format_response(*args, context_management=_APPLIED)
        without = await format_response(*args)

        assert message.model_dump(exclude_none=True)["context_management"] == _APPLIED
        assert "context_management" not in without.model_dump(exclude_none=True)

    @pytest.mark.parametrize("reported", [True, False])
    async def test_stream_puts_applied_edits_on_message_delta(
        self, *, reported: bool
    ) -> None:
        """``messageStop`` edits land top-level on ``message_delta``, and only there.

        Ref: https://platform.claude.com/docs/en/build-with-claude/context-editing
             stdapi/models/chat/_adapters/_anthropic_message.py:_process_stream_events
        """
        stop: dict[str, Any] = {"stopReason": "end_turn"}
        if reported:
            stop["additionalModelResponseFields"] = {"context_management": _APPLIED}
        events = [
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "ok"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": stop},
            {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
        ]

        async def _stream() -> AsyncIterator[Any]:
            for event in events:
                yield event

        sent = [
            (sse.event, loads(str(sse.data)))
            async for sse in format_stream("msg_1", "model-x", _stream(), None)
        ]

        (delta,) = [data for name, data in sent if name == "message_delta"]
        assert delta.get("context_management") == (_APPLIED if reported else None)
        (start,) = [data for name, data in sent if name == "message_start"]
        assert "context_management" not in start["message"]

    async def test_unexpected_report_shapes_never_fail_the_response(
        self, request_log: dict[str, Any]
    ) -> None:
        """Extra upstream fields are ignored and a malformed report is dropped, not a 500.

        Ref: stdapi/models/chat/_adapters/_anthropic_message.py:_map_context_management
        """
        edit = _APPLIED["applied_edits"][0]
        extended = {"applied_edits": [edit | {"cleared_tool_inputs": 3}], "extra": 1}
        malformed = {"applied_edits": [{"type": edit["type"]}]}
        args: tuple[Any, ...] = (
            [{"text": "ok"}],
            "end_turn",
            {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "msg_1",
            "model-x",
            None,
            lambda *_: None,
        )

        kept = await format_response(*args, context_management=extended)
        dropped = await format_response(*args, context_management=malformed)

        assert kept.model_dump(exclude_none=True)["context_management"] == _APPLIED
        assert dropped.context_management is None
        assert any(
            "unexpected shape" in str(detail)
            for detail in request_log.get("error_detail") or ()
        )

    def test_unknown_applied_edit_type_is_dropped(
        self, request_log: dict[str, Any]
    ) -> None:
        """An applied edit of an undescribed type is dropped with a warning, never a 500.

        Ref: stdapi/types/anthropic_messages.py:ContextManagementResponse
        """
        future = {"type": "clear_everything_20990101", "cleared_input_tokens": 1}
        result = ContextManagementResponse.model_validate(
            {"applied_edits": [*_APPLIED["applied_edits"], future]}
        )

        assert result.model_dump() == _APPLIED
        assert any(
            "clear_everything_20990101" in str(detail)
            for detail in request_log.get("error_detail") or ()
        )

    async def test_count_tokens_reports_the_unedited_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Counting with edits makes two calls: with the edits, and without them.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/count-tokens.html
             stdapi/models/chat/_adapters/_anthropic_message.py:count_tokens_via_bedrock
        """
        calls: list[dict[str, Any]] = []

        class _Client:
            async def count_tokens(self, **kwargs: Any) -> dict[str, int]:  # noqa: ANN401
                calls.append(kwargs)
                fields = kwargs["input"]["converse"]["additionalModelRequestFields"]
                return {"inputTokens": 900 if "context_management" in fields else 1100}

        monkeypatch.setattr(
            anthropic_message_adapter, "get_client", lambda *_a, **_k: _Client()
        )
        request = MessageCountTokensParams.model_validate(
            {
                "model": _CLAUDE_CONVERSE,
                "messages": [{"role": "user", "content": "hi"}],
                "context_management": _CLEAR_TOOL_USES,
            }
        )

        count = await count_tokens_via_bedrock(
            request, _CLAUDE_CONVERSE, "us-east-1", _converse_model(_CLAUDE_CONVERSE)
        )

        assert count.input_tokens == 900
        assert count.context_management is not None
        assert count.context_management.original_input_tokens == 1100
        assert len(calls) == 2
        for call in calls:
            fields = call["input"]["converse"]["additionalModelRequestFields"]
            assert _BETA in fields["anthropic_beta"]

    async def test_count_tokens_fails_when_the_edited_count_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refused edit fails the count, even though the unedited count succeeds.

        Ref: stdapi/models/chat/_adapters/_anthropic_message.py:count_tokens_via_bedrock
        """
        refusal = ClientError(
            {
                "Error": {
                    "Code": "ValidationException",
                    "Message": "The model returned the following errors: invalid edit",
                }
            },
            "CountTokens",
        )

        class _Client:
            async def count_tokens(self, **kwargs: Any) -> dict[str, int]:  # noqa: ANN401
                fields = kwargs["input"]["converse"]["additionalModelRequestFields"]
                if "context_management" in fields:
                    raise refusal
                return {"inputTokens": 1100}

        monkeypatch.setattr(
            anthropic_message_adapter, "get_client", lambda *_a, **_k: _Client()
        )
        request = MessageCountTokensParams.model_validate(
            {
                "model": _CLAUDE_CONVERSE,
                "messages": [{"role": "user", "content": "hi"}],
                "context_management": _CLEAR_TOOL_USES,
            }
        )

        with pytest.raises(ClientError) as excinfo:
            await count_tokens_via_bedrock(
                request,
                _CLAUDE_CONVERSE,
                "us-east-1",
                _converse_model(_CLAUDE_CONVERSE),
            )

        assert excinfo.value is refusal

    @pytest.mark.parametrize(
        ("model_id", "forwarded"), [(_CLAUDE_MANTLE, True), (_NON_CLAUDE_MANTLE, False)]
    )
    async def test_mantle_count_tokens_forwards_edits_to_claude_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
        request_log: dict[str, Any],
        model_id: str,
        *,
        forwarded: bool,
    ) -> None:
        """Mantle ``count_tokens`` sends the edits and their beta flag to Claude only.

        Ref: stdapi/routes/anthropic_messages.py:_count_tokens_via_mantle
        """
        sent: dict[str, Any] = {}

        async def _invoke(
            _region: object,
            _path: object,
            payload: dict[str, Any],
            *,
            headers: dict[str, str] | None = None,
            **_kw: object,
        ) -> dict[str, Any]:
            sent.update(payload=payload, headers=headers or {})
            return {"input_tokens": 3}

        async def _route_and_execute(
            _model_id: object,
            regions: list[Any],
            fn: Any,  # noqa: ANN401
        ) -> Any:  # noqa: ANN401
            return await fn(regions[0])

        monkeypatch.setattr(anthropic_messages, "invoke", _invoke)
        monkeypatch.setattr(anthropic_messages, "route_and_execute", _route_and_execute)
        monkeypatch.setattr(
            anthropic_messages, "set_effective_region", lambda *_a, **_k: None
        )
        request = MessageCountTokensParams.model_validate(
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
                "context_management": _CLEAR_TOOL_USES,
            }
        )

        count = await anthropic_messages._count_tokens_via_mantle(  # noqa: SLF001
            request, model_id
        )

        assert count.input_tokens == 3
        assert ("context_management" in sent["payload"]) is forwarded
        assert (sent["headers"].get("anthropic-beta") == _BETA) is forwarded
        warned = any(
            "context_management" in str(detail)
            for detail in request_log.get("error_detail") or ()
        )
        assert warned is not forwarded

    def test_mantle_headers_carry_the_beta_flag_only_with_edits(self) -> None:
        """Only the context-management beta is sent to Mantle, and only when needed.

        Ref: stdapi/models/chat/_mantle/_default.py:messages_request_headers
        """
        base = mantle_request_headers("messages") or {}

        assert messages_request_headers({"model": "m"}) == (base or None)
        assert messages_request_headers({"context_management": {}}) == base | {
            "anthropic-beta": _BETA
        }

    @pytest.mark.parametrize(
        ("model_id", "forwarded"), [(_CLAUDE_MANTLE, True), (_NON_CLAUDE_MANTLE, False)]
    )
    async def test_mantle_forwards_edits_to_claude_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
        request_log: dict[str, Any],
        model_id: str,
        *,
        forwarded: bool,
    ) -> None:
        """Mantle Claude keeps the edits and returns them; other models drop them with a warning.

        Ref: stdapi/models/chat/_mantle/_default.py:ChatModel.create_message
        """
        sent: dict[str, Any] = {}

        async def _serve_validated(
            _self: object, inbound: str, payload: dict[str, Any], **_kw: object
        ) -> tuple[str, str, dict[str, Any]]:
            sent.update(payload)
            raw: dict[str, Any] = {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": model_id,
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
            # Upstream reports applied edits only when the edits reached it.
            if "context_management" in payload:
                raw["context_management"] = _APPLIED
            return inbound, "us-east-1", raw

        monkeypatch.setattr(MantleChatModel, "_serve_validated", _serve_validated)
        message = await get_mantle_chat_model(model_id).create_message(
            _message_request(model_id, context_management=_CLEAR_TOOL_USES), "msg_1"
        )

        assert isinstance(message, Message)
        assert ("context_management" in sent) is forwarded
        assert (message.context_management is not None) is forwarded
        warned = any(
            "context_management" in str(detail)
            for detail in request_log.get("error_detail") or ()
        )
        assert warned is not forwarded
