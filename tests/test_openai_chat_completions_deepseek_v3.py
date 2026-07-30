"""Chat Completions ``reasoning_effort`` handling on the DeepSeek V3 family.

DeepSeek takes reasoning as a bare string in
``additionalModelRequestFields.reasoning_config``, so the OpenAI effort scale is
collapsed to ``low``/``medium``/``high`` by the model class.

Ref: https://developers.openai.com/api/docs/guides/reasoning
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html
     stdapi/models/chat/deepseek_v3.py:ChatModel._req_configure_reasoning
"""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

DEEPSEEK_V3 = "deepseek.v3-v1:0"
DEEPSEEK_V3_2 = "deepseek.v3.2"

DEEPSEEK_ALL = (DEEPSEEK_V3, DEEPSEEK_V3_2)
DEEPSEEK_SAMPLE = (DEEPSEEK_V3_2,)

#: finish_reason values the OpenAI Chat Completions reference defines.
_FINISH_REASONS = frozenset({"stop", "length", "content_filter", "tool_calls"})


class TestDeepseekChatCompletions:
    """Deepseek chat completions tests.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html
         stdapi/models/chat/deepseek_v3.py:ChatModel
    """

    @pytest.mark.parametrize("model", DEEPSEEK_SAMPLE)
    def test_reasoning_effort_parameter(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """``reasoning_effort="minimal"`` is accepted and answers normally.

        ``extract_reasoning`` reports ``enabled=True`` for any effort other than
        ``"none"``, and ``_REASONING_OVERRIDE`` folds ``minimal`` onto DeepSeek's
        lowest documented level, ``low`` — DeepSeek has no per-token reasoning budget,
        so ``thinking_budget`` would be rejected instead.

        Ref: https://developers.openai.com/api/docs/guides/reasoning
             stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_reasoning
        """
        if use_official_api:
            pytest.skip("Deepseek is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="minimal",
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) >= 1
        choice = resp.choices[0]
        assert choice.finish_reason in _FINISH_REASONS
        msg = choice.message
        assert msg.role == "assistant"
        assert msg.content
        assert resp.usage is not None
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0

    @pytest.mark.parametrize("model", DEEPSEEK_SAMPLE)
    def test_reasoning_effort_none_explicit_disable(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """``reasoning_effort="none"`` is accepted and produces a plain answer.

        ``extract_reasoning`` maps ``"none"`` to ``enabled=False``; unlike Nova or
        Claude, ``deepseek_v3.ChatModel._req_configure_reasoning`` then writes no
        ``reasoning_config`` at all rather than an explicit ``disabled`` entry, so the
        model's own default applies and the response carries no reasoning text of the
        gateway's making.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html
             stdapi/models/chat/deepseek_v3.py:ChatModel._req_configure_reasoning
        """
        if use_official_api:
            pytest.skip("Deepseek is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="none",
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) >= 1
        choice = resp.choices[0]
        assert choice.finish_reason in _FINISH_REASONS
        msg = choice.message
        assert msg.role == "assistant"
        assert msg.content
        assert "ok" in msg.content.lower(), (
            f"Expected the pinned reply for {model!r}, got: {msg.content!r}"
        )
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0
