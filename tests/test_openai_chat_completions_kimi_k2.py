"""Chat Completions thinking control on the Moonshot Kimi K2 family.

Kimi takes reasoning as ``additionalModelRequestFields.thinking = {"type": …}``
together with a ``reasoning_effort`` level, and no token budget.

Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-moonshot-ai.html
     stdapi/models/chat/kimi_k25.py:ChatModel._req_configure_reasoning
"""

from typing import TYPE_CHECKING, cast

import pytest

from stdapi.models import _find_model_class
from stdapi.models.chat.kimi_k25 import ChatModel as KimiChatModel
from tests.conftest import FINISH_REASONS

if TYPE_CHECKING:
    from openai import OpenAI

    from stdapi.models.chat import Effort
    from stdapi.types import JsonMapping

#: Kimi K2.5, whose thinking is off by default.
KIMI_K2_5 = "moonshotai.kimi-k2.5"

#: Kimi K2 Thinking, which always reasons.
KIMI_K2_THINKING = "moonshot.kimi-k2-thinking"

#: Both Kimi K2 revisions, swept by the live tests.
KIMI_ALL = (KIMI_K2_5, KIMI_K2_THINKING)


@pytest.mark.local
class TestKimiK2Matcher:
    """``ChatModel.MATCHER`` dispatches both Bedrock provider prefixes to the Kimi class.

    Bedrock exposes Kimi K2 models under two different provider prefixes
    (``moonshotai.kimi-k2.5`` and ``moonshot.kimi-k2-thinking``); both must
    resolve to the same Kimi-specific reasoning implementation (issue #98).

    Ref: stdapi/models/chat/kimi_k25.py:ChatModel.MATCHER
         stdapi/models/__init__.py:_find_model_class
    """

    @pytest.mark.parametrize("model_id", KIMI_ALL)
    def test_dispatches_to_kimi_chat_model(self, model_id: str) -> None:
        """Both Kimi K2 provider prefixes resolve to the Kimi ``ChatModel`` class."""
        assert _find_model_class(model_id) is KimiChatModel


@pytest.mark.local
class TestKimiReasoningFields:
    """Enabling thinking also names an effort level, because the level decides.

    Measured on Bedrock: ``thinking={"type": "enabled"}`` alone leaves Kimi K2.5
    answering without any ``reasoningContent`` block, and only
    ``reasoning_effort="high"`` produces one. ``minimal`` is rejected outright, so
    it maps down to ``low``.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-moonshot-ai.html
         stdapi/models/chat/kimi_k25.py:ChatModel._req_configure_reasoning
    """

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            (None, "high"),
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "high"),
            ("max", "high"),
        ],
    )
    def test_enabled_thinking_carries_an_effort_level(
        self, requested: str | None, expected: str
    ) -> None:
        """Every accepted effort maps onto Kimi's own scale, defaulting to the one that reasons."""
        fields: JsonMapping = {}

        KimiChatModel(KIMI_K2_5)._req_configure_reasoning(  # noqa: SLF001
            fields, enabled=True, reasoning_effort=cast("Effort | None", requested)
        )

        assert fields == {"thinking": {"type": "enabled"}, "reasoning_effort": expected}

    def test_disabled_thinking_names_no_effort(self) -> None:
        """Turning thinking off sends the toggle alone: an effort would re-enable it."""
        fields: JsonMapping = {}

        KimiChatModel(KIMI_K2_5)._req_configure_reasoning(fields, enabled=False)  # noqa: SLF001

        assert fields == {"thinking": {"type": "disabled"}}


@pytest.mark.gateway("Kimi is not supported on the official API")
class TestKimiK25ChatCompletions:
    """Moonshot Kimi K2.5 chat completions tests.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-moonshot-ai.html
         stdapi/models/chat/kimi_k25.py:ChatModel
    """

    @pytest.mark.parametrize("model", KIMI_ALL)
    def test_thinking_not_set(self, openai_client: OpenAI, model: str) -> None:
        """Omitting every reasoning field leaves the model's own thinking default alone.

        ``extract_reasoning`` returns ``None`` when ``reasoning_effort``,
        ``enable_thinking`` and ``thinking`` are all unset, so no ``thinking`` entry is
        added to ``additionalModelRequestFields`` and Bedrock applies the model default
        (on for Kimi K2 Thinking, off for Kimi K2.5).

        Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_reasoning
             https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-moonshot-ai-kimi-k2-thinking.html
        """
        resp = openai_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Reply with OK."}]
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) >= 1
        choice = resp.choices[0]
        assert choice.finish_reason in FINISH_REASONS
        msg = choice.message
        assert msg.role == "assistant"
        assert msg.content
        assert resp.usage is not None
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0

    @pytest.mark.parametrize("model", KIMI_ALL)
    def test_thinking_disabled(self, openai_client: OpenAI, model: str) -> None:
        """``enable_thinking=False`` is accepted, and only Kimi K2.5 answers without reasoning.

        Kimi K2.5 returns no ``reasoningContent`` block, so ``reasoning_content`` is
        absent from the message.  Kimi K2 Thinking is thinking-only: Bedrock accepts
        ``thinking={"type": "disabled"}`` for it and still returns ``reasoningContent``,
        which the gateway surfaces as ``reasoning_content``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
             stdapi/models/chat/kimi_k25.py:ChatModel._req_configure_reasoning
             stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_output_text
        """
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            extra_body={"enable_thinking": False},
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) >= 1
        choice = resp.choices[0]
        assert choice.finish_reason in FINISH_REASONS
        msg = choice.message
        assert msg.role == "assistant"
        assert msg.content
        # ``reasoning_content`` is an extra field: absent, not null, when unused.
        reasoning = getattr(msg, "reasoning_content", None)
        if model == KIMI_K2_THINKING:
            assert reasoning, (
                "Kimi K2 Thinking always reasons, even with thinking disabled"
            )
        else:
            assert reasoning is None, (
                "thinking disabled must not return reasoning content"
            )
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model", KIMI_ALL)
    def test_thinking_enabled(self, openai_client: OpenAI, model: str) -> None:
        """``enable_thinking=True`` turns thinking on and surfaces ``reasoning_content``.

        The gateway sends ``thinking={"type": "enabled"}``; the Bedrock
        ``reasoningContent`` blocks it produces are split out of the assistant text into
        the ``reasoning_content`` field.  Reasoning tokens count against
        ``max_completion_tokens``, so a short answer can still finish on ``length``.

        The question needs several arithmetic steps: K2.5 skips reasoning entirely on a
        question it can answer from memory, which would prove nothing about the split.

        Ref: https://developers.openai.com/api/docs/guides/reasoning
             stdapi/models/chat/kimi_k25.py:ChatModel._req_configure_reasoning
        """
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "A train leaves at 09:40 and the trip takes 2 hours and "
                        "50 minutes, but it is delayed by 35 minutes. At what time "
                        "does it arrive? Reply with the time only."
                    ),
                }
            ],
            extra_body={"enable_thinking": True},
            max_completion_tokens=4096,
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) >= 1
        choice = resp.choices[0]
        assert choice.finish_reason in FINISH_REASONS
        msg = choice.message
        assert msg.role == "assistant"
        assert getattr(msg, "reasoning_content", None), (
            f"thinking enabled must return reasoning content for {model!r}"
        )
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0
