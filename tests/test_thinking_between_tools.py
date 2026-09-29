"""Claude ``between_tools`` thinking: the lowest setting of a model that cannot disable it.

Claude Sonnet 5.5 rejects ``thinking: {"type": "disabled"}`` and offers
``between_tools`` instead: no thinking before the answer, only short progress
updates between tool calls. It takes no other field and at most ``high`` effort.
Every other model rejects it. The Anthropic routes accept it as reasoning turned
off; every route's own way of turning reasoning off reaches Sonnet 5.5 and later
as ``between_tools``, a model accepting ``disabled`` as that, and a model that
always reasons as nothing, with a warning. Other models see no difference.

Ref: https://platform.claude.com/docs/en/build-with-claude/thinking
     https://platform.claude.com/docs/en/models/sonnet-5-5/whats-new-sonnet-5-5
     tests/probes/results/anthropic.claude-sonnet-5-5.json
     stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_reasoning
     stdapi/models/chat/_mantle/_convert.py:_thinking_off
"""

from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import ValidationError

from stdapi.models.chat import get_chat_model
from stdapi.models.chat._adapters import _anthropic_message as anthropic_adapter
from stdapi.models.chat._adapters import _ollama as ollama_adapter
from stdapi.models.chat._adapters import _openai_chat_completion as chat_adapter
from stdapi.models.chat._adapters import _openai_responses as responses_adapter
from stdapi.models.chat._mantle._convert import convert_payload, messages_payload
from stdapi.types.anthropic_messages import (
    MessageCountTokensParams,
    MessageCreateParams,
)
from stdapi.types.ollama import ChatRequest
from stdapi.types.openai_chat_completions import (
    CompletionCreateParams as ChatCompletionCreateParams,
)
from stdapi.types.openai_responses import ResponseCreateParams

if TYPE_CHECKING:
    from collections.abc import Callable

    from stdapi.models.chat import ReasoningParams
    from stdapi.models.chat._anthropic_claude import AnthropicClaudeChatModel
    from stdapi.models.chat._default import ChatModel
    from stdapi.types import JsonMapping

pytestmark = pytest.mark.local

#: The model offering ``between_tools``.
_SONNET_5_5 = "anthropic.claude-sonnet-5-5"

#: A model accepting ``disabled`` and rejecting ``between_tools``.
_SONNET_5 = "anthropic.claude-sonnet-5"

#: A model that always reasons and rejects both.
_OPUS_5_5 = "anthropic.claude-opus-5-5"

#: The Messages request fields every test here shares.
_MESSAGE: dict[str, Any] = {
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}],
}

#: The Chat Completions request fields every test here shares.
_CHAT: dict[str, Any] = {
    "model": _SONNET_5_5,
    "messages": [{"role": "user", "content": "hi"}],
}


def _claude(model_id: str) -> AnthropicClaudeChatModel:
    """Return the runtime Claude implementation selected for *model_id*."""
    return cast(
        "AnthropicClaudeChatModel", get_chat_model(model_id, allow_mantle=False)
    )


def _messages(thinking: dict[str, str], **fields: object) -> ReasoningParams | None:
    """Return the reasoning a Messages request with *thinking* resolves to."""
    request = MessageCreateParams.model_validate(
        {"model": _SONNET_5_5, **_MESSAGE, "thinking": thinking, **fields}
    )
    return anthropic_adapter.extract_reasoning(request)


def _chat(**fields: object) -> ReasoningParams | None:
    """Return the reasoning a Chat Completions request resolves to."""
    request = ChatCompletionCreateParams.model_validate({**_CHAT, **fields})
    return chat_adapter.extract_reasoning(request)


def _responses(**fields: object) -> ReasoningParams | None:
    """Return the reasoning a Responses request resolves to."""
    request = ResponseCreateParams.model_validate(
        {"model": _SONNET_5_5, "input": "hi", **fields}
    )
    return responses_adapter.extract_reasoning(request)


def _ollama(**fields: object) -> ReasoningParams | None:
    """Return the reasoning an Ollama chat request resolves to."""
    request = ChatRequest.model_validate(
        {
            "model": _SONNET_5_5,
            "messages": [{"role": "user", "content": "hi"}],
            **fields,
        }
    )
    return chat_adapter.extract_reasoning(
        ollama_adapter.to_chat_completion_params(request, _SONNET_5_5)
    )


class TestRequestTypes:
    """``between_tools`` is an Anthropic request shape, accepted bare only.

    Ref: https://platform.claude.com/docs/en/api/messages/create
         stdapi/types/anthropic_messages.py:ThinkingConfigBetweenToolsParam
    """

    def test_messages_accepts_it(self) -> None:
        """The Messages and count-tokens requests parse it."""
        for params in (MessageCreateParams, MessageCountTokensParams):
            request = params.model_validate(
                {
                    "model": _SONNET_5_5,
                    **_MESSAGE,
                    "thinking": {"type": "between_tools"},
                }
            )
            assert request.thinking is not None
            assert request.thinking.type == "between_tools"

    @pytest.mark.parametrize(
        "extra", [{"display": "summarized"}, {"budget_tokens": 1024}]
    )
    def test_another_field_is_rejected(self, extra: dict[str, object]) -> None:
        """``between_tools`` takes no other field, as upstream answers with a 400."""
        with pytest.raises(ValidationError):
            _messages({"type": "between_tools", **extra})  # type: ignore[dict-item]

    def test_the_moonshot_thinking_field_does_not_take_it(self) -> None:
        """No OpenAI-dialect client sends it: those turn reasoning off their own way."""
        with pytest.raises(ValidationError):
            _chat(thinking={"type": "between_tools"})


class TestEveryRouteTurnsSonnet55Off:
    """Each route's own switch reaches Sonnet 5.5 as ``between_tools``.

    Ref: stdapi/models/chat/_adapters/_anthropic_message.py:extract_reasoning
         stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_reasoning
         stdapi/models/chat/_adapters/_openai_responses.py:extract_reasoning
         stdapi/models/chat/_adapters/_ollama.py:to_chat_completion_params
    """

    @pytest.mark.parametrize(
        "resolve",
        [
            lambda: _messages({"type": "disabled"}),
            lambda: _messages({"type": "between_tools"}),
            lambda: _chat(reasoning_effort="none"),
            lambda: _chat(enable_thinking=False),
            lambda: _chat(thinking={"type": "disabled"}),
            lambda: _responses(reasoning={"effort": "none"}),
            lambda: _responses(reasoning={"effort": "none", "summary": "auto"}),
            lambda: _ollama(think=False),
        ],
        ids=[
            "messages-disabled",
            "messages-between-tools",
            "chat-effort-none",
            "chat-enable-thinking-false",
            "chat-moonshot-thinking-disabled",
            "responses-effort-none",
            "responses-effort-none-with-summary",
            "ollama-think-false",
        ],
    )
    def test_reasoning_off(
        self, resolve: Callable[[], ReasoningParams | None], request_log: dict[str, Any]
    ) -> None:
        """Honoured with the lowest setting, so nothing is warned and no display is sent."""
        reasoning = resolve()
        assert reasoning is not None
        assert reasoning["enabled"] is False
        fields: JsonMapping = {}

        _claude(_SONNET_5_5)._req_configure_reasoning(fields, **reasoning)  # noqa: SLF001

        assert fields == {"reasoning_config": {"type": "between_tools"}}
        assert request_log["level"] == "info"

    @pytest.mark.parametrize("thinking", ["between_tools", "disabled"])
    def test_messages_keeps_an_effort_with_thinking_off(self, thinking: str) -> None:
        """An effort alone turns reasoning on; with thinking off it rides along."""
        reasoning = _messages({"type": thinking}, output_config={"effort": "low"})
        assert reasoning is not None
        assert reasoning["enabled"] is False
        fields: JsonMapping = {}

        _claude(_SONNET_5_5)._req_configure_reasoning(fields, **reasoning)  # noqa: SLF001

        assert fields == {
            "reasoning_config": {"type": "between_tools"},
            "output_config": {"effort": "low"},
        }


class TestRuntimeMapping:
    """What bedrock-runtime is sent when reasoning is off.

    Ref: stdapi/models/chat/anthropic_claude_5.py:ChatModel.BETWEEN_TOOLS_SUPPORTED
    """

    @pytest.mark.parametrize(
        "model_id",
        ["anthropic.claude-sonnet-5-5-20270101-v1:0", "anthropic.claude-sonnet-6"],
    )
    def test_later_versions_get_between_tools(self, model_id: str) -> None:
        """Dated variants and later versions are assumed to keep the setting."""
        fields: JsonMapping = {}

        _claude(model_id)._req_configure_reasoning(fields, enabled=False)  # noqa: SLF001

        assert fields == {"reasoning_config": {"type": "between_tools"}}

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            (_SONNET_5, {"reasoning_config": {"type": "disabled"}}),
            ("anthropic.claude-opus-5", {"reasoning_config": {"type": "disabled"}}),
            (_OPUS_5_5, {}),
            ("anthropic.claude-fable-5", {}),
        ],
    )
    def test_other_claude_models_keep_their_mapping(
        self, model_id: str, expected: dict[str, object]
    ) -> None:
        """Sonnet 5 and Opus 5 are disabled; the always-reasoning models keep their default."""
        fields: JsonMapping = {}

        _claude(model_id)._req_configure_reasoning(fields, enabled=False)  # noqa: SLF001

        assert fields == expected
        assert not _claude(model_id).BETWEEN_TOOLS_SUPPORTED

    def test_disabled_wins_over_an_effort(self) -> None:
        """``disabled`` with an effort turns thinking off and keeps the effort, as upstream."""
        reasoning = _messages({"type": "disabled"}, output_config={"effort": "high"})
        assert reasoning is not None
        fields: JsonMapping = {}

        _claude(_SONNET_5)._req_configure_reasoning(  # noqa: SLF001
            fields, **reasoning
        )

        assert fields == {
            "reasoning_config": {"type": "disabled"},
            "output_config": {"effort": "high"},
        }

    def test_an_always_reasoning_model_keeps_the_effort(
        self, request_log: dict[str, Any]
    ) -> None:
        """Opus 5.5 cannot turn thinking off, but the effort still reaches it.

        Ref: stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_reasoning
        """
        reasoning = _messages({"type": "disabled"}, output_config={"effort": "low"})
        assert reasoning is not None
        fields: JsonMapping = {}

        _claude(_OPUS_5_5)._req_configure_reasoning(fields, **reasoning)  # noqa: SLF001

        assert fields == {"output_config": {"effort": "low"}}
        assert request_log["level"] == "warning"


class TestNoSideEffectOnOtherModels:
    """``between_tools`` on the Messages route is exactly ``disabled`` elsewhere.

    Ref: stdapi/models/chat/_adapters/_anthropic_message.py:extract_reasoning
    """

    @pytest.mark.parametrize(
        "model_id",
        [
            "amazon.nova-2-lite-v1:0",
            "deepseek.v3-v1:0",
            "deepseek.v3.2",
            "moonshotai.kimi-k2.5",
            "moonshotai.kimi-k2-thinking",
            "openai.gpt-oss-120b-1:0",
            "openai.gpt-5.6",
            "qwen.qwen3-32b-v1:0",
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
            _SONNET_5,
        ],
    )
    def test_between_tools_sends_what_disabled_sends(
        self, model_id: str, request_log: dict[str, Any]
    ) -> None:
        """Even with an effort, as ``disabled`` with an effort does."""
        between = _messages({"type": "between_tools"}, output_config={"effort": "low"})
        disabled = _messages({"type": "disabled"}, output_config={"effort": "low"})
        assert between is not None
        assert disabled is not None
        model = cast("ChatModel", get_chat_model(model_id, allow_mantle=False))
        sent: JsonMapping = {}
        expected: JsonMapping = {}

        model._req_configure_reasoning(sent, **between)  # noqa: SLF001
        level = request_log["level"]
        model._req_configure_reasoning(expected, **disabled)  # noqa: SLF001

        assert sent == expected
        assert level == request_log["level"]


class TestMantleMapping:
    """The same mapping on the Bedrock Mantle paths.

    Ref: stdapi/models/chat/_mantle/_convert.py:messages_payload
         stdapi/models/chat/_mantle/_convert.py:_anthropic_reasoning_fields
         stdapi/models/chat/_mantle/_convert.py:_request_thinking_summary
    """

    @pytest.mark.parametrize(
        ("model", "sent", "expected"),
        [
            (_SONNET_5_5, "disabled", {"type": "between_tools"}),
            (_SONNET_5_5, "between_tools", {"type": "between_tools"}),
            (_SONNET_5, "between_tools", {"type": "disabled"}),
            (_OPUS_5_5, "between_tools", None),
            ("openai.gpt-oss-120b", "disabled", {"type": "disabled"}),
            ("openai.gpt-oss-120b", "between_tools", {"type": "disabled"}),
        ],
    )
    async def test_messages_passthrough(
        self,
        model: str,
        sent: str,
        expected: dict[str, str] | None,
        request_log: dict[str, Any],
    ) -> None:
        """Reasoning off reaches each model as the setting it accepts."""
        request = MessageCreateParams.model_validate(
            {"model": model, **_MESSAGE, "thinking": {"type": sent}}
        )

        payload = await messages_payload(request, model)

        assert payload.get("thinking") == expected
        assert request_log["level"] == ("warning" if expected is None else "info")

    @pytest.mark.parametrize(
        ("model", "fields", "expected"),
        [
            (_SONNET_5_5, {"reasoning_effort": "none"}, {"type": "between_tools"}),
            (_SONNET_5_5, {"enable_thinking": False}, {"type": "between_tools"}),
            (_SONNET_5, {"reasoning_effort": "none"}, {"type": "disabled"}),
            (_OPUS_5_5, {"reasoning_effort": "none"}, None),
        ],
        ids=["effort-none", "enable-thinking-false", "sonnet-5", "opus-5-5"],
    )
    def test_chat_completions_conversion(
        self,
        model: str,
        fields: dict[str, object],
        expected: dict[str, str] | None,
        request_log: dict[str, Any],
    ) -> None:
        """Reasoning turned off on Chat Completions reaches Claude on Mantle alike."""
        out = convert_payload(
            "chat_completions",
            "messages",
            {"model": model, "messages": [{"role": "user", "content": "hi"}], **fields},
        )

        assert out.get("thinking") == expected
        assert "output_config" not in out

    def test_a_responses_summary_adds_no_display(self) -> None:
        """``between_tools`` takes no ``display``; its updates come back summarized anyway."""
        out = convert_payload(
            "responses",
            "messages",
            {
                "model": _SONNET_5_5,
                "input": "hi",
                "reasoning": {"effort": "none", "summary": "auto"},
                "max_output_tokens": 4096,
            },
        )

        assert out.get("thinking") == {"type": "between_tools"}
