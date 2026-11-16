"""Reasoning control of the models taking ``additionalModelRequestFields.reasoning.effort``.

Probed on Converse: these models reason by default, honour
``reasoning: {"effort": ...}`` with ``none``, ``low``, ``medium``, ``high``,
``xhigh`` and ``max``, and reject ``minimal``. ``none`` is their only off switch.

Ref: https://developers.openai.com/api/docs/guides/reasoning
     stdapi/models/chat/_reasoning_effort.py:ReasoningEffortChatModel
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from stdapi.models.chat._adapters import _openai_responses as responses_adapter
from stdapi.models.chat._reasoning_effort import ReasoningEffortChatModel
from stdapi.models.chat.kimi_k3 import ChatModel as KimiK3ChatModel
from stdapi.models.chat.openai_gpt import ChatModel as OpenAIGPTChatModel
from stdapi.types.anthropic_messages import MessageCreateParams
from stdapi.types.openai_chat_completions import CompletionCreateParams
from stdapi.types.openai_responses import ResponseCreateParams

if TYPE_CHECKING:
    from stdapi.models.chat import Effort
    from stdapi.types import JsonMapping

pytestmark = pytest.mark.local

#: A model id the effort object is sent to; the class under test does not match on it.
_MODEL_ID = "openai.gpt-6-luna"

#: Every requested level with the value the model receives.
_EFFORTS = [
    ("none", "none"),
    ("minimal", "low"),
    ("low", "low"),
    ("medium", "medium"),
    ("high", "high"),
    ("xhigh", "xhigh"),
    ("max", "max"),
]


def _model() -> ReasoningEffortChatModel:
    """Return the model class under test."""
    return ReasoningEffortChatModel(_MODEL_ID)


def _messages() -> list[dict[str, Any]]:
    """Return a one-turn conversation."""
    return [{"role": "user", "content": "Reply with OK."}]


class TestConfigureReasoning:
    """The requested level becomes ``reasoning.effort``, and only that.

    Ref: stdapi/models/chat/_reasoning_effort.py:ReasoningEffortChatModel._req_configure_reasoning
    """

    @pytest.mark.parametrize(("requested", "expected"), _EFFORTS)
    def test_effort_maps_onto_the_accepted_values(
        self, requested: str, expected: str
    ) -> None:
        """Each level is sent as the effort object; ``minimal`` is lifted to ``low``."""
        fields: JsonMapping = {}

        _model()._req_configure_reasoning(  # noqa: SLF001
            fields,
            enabled=requested != "none",
            reasoning_effort=cast("Effort", requested),
        )

        assert fields == {"reasoning": {"effort": expected}}

    def test_disabled_sends_none(self) -> None:
        """Turning reasoning off names ``none``, the only value that stops it."""
        fields: JsonMapping = {}

        _model()._req_configure_reasoning(fields, enabled=False)  # noqa: SLF001

        assert fields == {"reasoning": {"effort": "none"}}

    @pytest.mark.parametrize(
        ("enabled", "requested"), [(False, None), (False, "none"), (True, "none")]
    )
    def test_a_model_that_always_reasons_is_never_sent_none(
        self, enabled: bool, requested: str | None, request_log: dict[str, Any]
    ) -> None:
        """Disabling reasoning where ``none`` is rejected keeps the default and warns.

        Ref: stdapi/models/chat/_reasoning_effort.py:ReasoningEffortChatModel.REASONING_DISABLE_SUPPORTED
        """

        class _AlwaysReasons(ReasoningEffortChatModel):
            __slots__ = ()
            REASONING_DISABLE_SUPPORTED = False

        fields: JsonMapping = {}

        _AlwaysReasons(_MODEL_ID)._req_configure_reasoning(  # noqa: SLF001
            fields, enabled=enabled, reasoning_effort=cast("Effort | None", requested)
        )

        assert fields == {}
        assert request_log["level"] == "warning"
        assert "cannot be disabled" in str(request_log["error_detail"])

    def test_a_model_that_always_reasons_still_takes_a_level(self) -> None:
        """Only the off switch is withheld: every other level is still sent."""

        class _AlwaysReasons(ReasoningEffortChatModel):
            __slots__ = ()
            REASONING_DISABLE_SUPPORTED = False

        fields: JsonMapping = {}

        _AlwaysReasons(_MODEL_ID)._req_configure_reasoning(  # noqa: SLF001
            fields, enabled=True, reasoning_effort="minimal"
        )

        assert fields == {"reasoning": {"effort": "low"}}

    def test_enabled_without_a_level_leaves_the_model_default(self) -> None:
        """Asking for reasoning without a level sends nothing: the model reasons by default."""
        fields: JsonMapping = {}

        _model()._req_configure_reasoning(  # noqa: SLF001
            fields, enabled=True, budget_tokens=2048, max_tokens=4096
        )

        assert fields == {}

    def test_no_other_reasoning_field_is_sent(self) -> None:
        """Neither ``thinking`` nor a flat ``reasoning_effort`` is sent.

        Both are rejected by the OpenAI GPT models and silently ignored by Kimi K3.
        """
        fields: JsonMapping = {}

        _model()._req_configure_reasoning(  # noqa: SLF001
            fields, enabled=True, reasoning_effort="high"
        )

        assert set(fields) == {"reasoning"}


class TestEveryDialectReachesTheEffort:
    """Each client dialect's reasoning field reaches the outgoing request.

    Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_reasoning
         stdapi/models/chat/_adapters/_anthropic_message.py:extract_reasoning
         stdapi/models/chat/_adapters/_openai_responses.py:extract_reasoning
    """

    @pytest.mark.parametrize(("requested", "expected"), _EFFORTS)
    async def test_chat_completions_reasoning_effort(
        self, requested: str, expected: str
    ) -> None:
        """``reasoning_effort`` is sent as the effort object.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        """
        request = CompletionCreateParams.model_validate(
            {"model": _MODEL_ID, "messages": _messages(), "reasoning_effort": requested}
        )

        payload, _, _ = await _model().build_completion_request(request)

        assert payload["additionalModelRequestFields"] == {
            "reasoning": {"effort": expected}
        }

    async def test_chat_completions_without_reasoning_sends_nothing(self) -> None:
        """A request naming no reasoning field leaves the model default alone."""
        request = CompletionCreateParams.model_validate(
            {"model": _MODEL_ID, "messages": _messages()}
        )

        payload, _, _ = await _model().build_completion_request(request)

        assert "reasoning" not in payload.get("additionalModelRequestFields", {})

    async def test_chat_completions_thinking_disabled(self) -> None:
        """The Moonshot-style ``thinking`` toggle turns reasoning off with ``none``.

        Ref: https://platform.kimi.ai/docs/api/chat
        """
        request = CompletionCreateParams.model_validate(
            {
                "model": _MODEL_ID,
                "messages": _messages(),
                "thinking": {"type": "disabled"},
            }
        )

        payload, _, _ = await _model().build_completion_request(request)

        assert payload["additionalModelRequestFields"] == {
            "reasoning": {"effort": "none"}
        }

    async def test_anthropic_output_config_effort(self) -> None:
        """``output_config.effort`` is sent as the effort object.

        Ref: https://docs.claude.com/en/api/messages
        """
        request = MessageCreateParams.model_validate(
            {
                "model": _MODEL_ID,
                "max_tokens": 1024,
                "messages": _messages(),
                "output_config": {"effort": "xhigh"},
            }
        )

        payload, _ = await _model().build_message_request(request)

        assert payload["additionalModelRequestFields"] == {
            "reasoning": {"effort": "xhigh"}
        }

    async def test_anthropic_thinking_disabled(self) -> None:
        """``thinking: {"type": "disabled"}`` turns reasoning off with ``none``.

        Ref: https://docs.claude.com/en/api/messages
        """
        request = MessageCreateParams.model_validate(
            {
                "model": _MODEL_ID,
                "max_tokens": 1024,
                "messages": _messages(),
                "thinking": {"type": "disabled"},
            }
        )

        payload, _ = await _model().build_message_request(request)

        assert payload["additionalModelRequestFields"] == {
            "reasoning": {"effort": "none"}
        }

    @pytest.mark.parametrize(
        ("reasoning", "expected"),
        [({"effort": "none"}, "none"), ({"effort": "max"}, "max"), ({}, "medium")],
    )
    def test_responses_reasoning_effort(
        self, reasoning: dict[str, str], expected: str
    ) -> None:
        """``reasoning.effort`` is sent as is; an object without one asks for ``medium``.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        request = ResponseCreateParams.model_validate(
            {"model": _MODEL_ID, "input": "Reply with OK.", "reasoning": reasoning}
        )
        params = responses_adapter.extract_reasoning(request)
        assert params is not None
        fields: JsonMapping = {}

        _model()._req_configure_reasoning(fields, **params)  # noqa: SLF001

        assert fields == {"reasoning": {"effort": expected}}


class TestFixedSampling:
    """``temperature`` and ``top_p`` are dropped where the model refuses them.

    Kimi K3 and GPT-6 refuse both with ``400`` (``tests/probes/results/``,
    ``temperature`` and ``top_p`` probes); the parameters only tune sampling, so
    they are accepted and ignored, while ``stop`` still reaches the model.

    Ref: stdapi/models/chat/_reasoning_effort.py:ReasoningEffortChatModel._prepare_converse_request
    """

    @pytest.mark.parametrize(
        ("model_class", "model_id"),
        [
            (KimiK3ChatModel, "moonshotai.kimi-k3"),
            (KimiK3ChatModel, "moonshotai.kimi-k3.1"),
            (OpenAIGPTChatModel, "openai.gpt-6-luna"),
            (OpenAIGPTChatModel, "openai.gpt-6-astra"),
            (OpenAIGPTChatModel, "openai.gpt-6.1-sol"),
            (OpenAIGPTChatModel, "openai.gpt-7"),
        ],
    )
    async def test_sampling_is_dropped_and_stop_is_kept(
        self,
        model_class: type[ReasoningEffortChatModel],
        model_id: str,
        request_log: dict[str, Any],
    ) -> None:
        """The request goes out without both, and one warning names them."""
        request = CompletionCreateParams.model_validate(
            {
                "model": model_id,
                "messages": _messages(),
                "temperature": 0.2,
                "top_p": 0.9,
                "stop": ["END"],
            }
        )

        payload, _, _ = await model_class(model_id).build_completion_request(request)

        assert payload["inferenceConfig"].get("stopSequences") == ["END"]
        assert "temperature" not in payload["inferenceConfig"]
        assert "topP" not in payload["inferenceConfig"]
        warnings = [
            detail
            for detail in request_log.get("error_detail") or ()
            if "temperature" in str(detail)
        ]
        assert len(warnings) == 1
        assert "top_p" in str(warnings[0])
        assert request_log["level"] == "warning"

    async def test_anthropic_sampling_is_dropped(
        self, request_log: dict[str, Any]
    ) -> None:
        """The Anthropic dialect's ``temperature`` and ``top_p`` are dropped too.

        Ref: https://docs.claude.com/en/api/messages
        """
        request = MessageCreateParams.model_validate(
            {
                "model": "moonshotai.kimi-k3.1",
                "max_tokens": 1024,
                "messages": _messages(),
                "temperature": 0.2,
                "top_p": 0.9,
                "stop_sequences": ["END"],
            }
        )

        payload, _ = await KimiK3ChatModel(
            "moonshotai.kimi-k3.1"
        ).build_message_request(request)

        assert payload["inferenceConfig"] == {
            "maxTokens": 1024,
            "stopSequences": ["END"],
        }
        assert request_log["level"] == "warning"

    async def test_only_the_parameter_sent_is_named(
        self, request_log: dict[str, Any]
    ) -> None:
        """A request with ``top_p`` alone is warned about ``top_p`` alone."""
        request = CompletionCreateParams.model_validate(
            {"model": "moonshotai.kimi-k3", "messages": _messages(), "top_p": 0.9}
        )

        await KimiK3ChatModel("moonshotai.kimi-k3").build_completion_request(request)

        (warning,) = request_log["error_detail"]
        assert "top_p" in str(warning)
        assert "temperature" not in str(warning)

    @pytest.mark.parametrize(
        ("model_class", "model_id"),
        [
            (ReasoningEffortChatModel, "openai.gpt-6-luna"),
            (OpenAIGPTChatModel, "openai.gpt-5.6-sol"),
        ],
    )
    async def test_a_model_without_the_flag_keeps_sampling(
        self,
        model_class: type[ReasoningEffortChatModel],
        model_id: str,
        request_log: dict[str, Any],
    ) -> None:
        """Sampling is forwarded where no refusal was recorded, with no warning.

        GPT-5.6 is served by both endpoints and reaches this class wherever it
        is not routed to Bedrock Mantle.
        """
        request = CompletionCreateParams.model_validate(
            {
                "model": model_id,
                "messages": _messages(),
                "temperature": 0.2,
                "top_p": 0.9,
            }
        )

        payload, _, _ = await model_class(model_id).build_completion_request(request)

        assert payload["inferenceConfig"]["temperature"] == pytest.approx(0.2)
        assert payload["inferenceConfig"]["topP"] == pytest.approx(0.9)
        assert not request_log.get("error_detail")
