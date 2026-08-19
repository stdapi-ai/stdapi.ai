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

from tests.conftest import FINISH_REASONS

if TYPE_CHECKING:
    from openai import OpenAI

#: Newest DeepSeek V3 revision; the older ``deepseek.v3-v1:0`` shares its model class
#: and is swept by the multi-model module, so the reasoning tests run on this one only.
DEEPSEEK_SAMPLE = ("deepseek.v3.2",)


@pytest.mark.gateway("Deepseek is not supported on the official API")
class TestDeepseekChatCompletions:
    """Deepseek chat completions tests.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html
         stdapi/models/chat/deepseek_v3.py:ChatModel
    """

    @pytest.mark.parametrize("model", DEEPSEEK_SAMPLE)
    def test_reasoning_effort_parameter(
        self, openai_client: OpenAI, model: str
    ) -> None:
        """``reasoning_effort="minimal"`` is accepted and answers normally.

        ``extract_reasoning`` reports ``enabled=True`` for any effort other than
        ``"none"``, and ``_REASONING_OVERRIDE`` folds ``minimal`` onto DeepSeek's
        lowest documented level, ``low`` — DeepSeek has no per-token reasoning budget,
        so a ``thinking_budget`` is accepted and the effort scale decides the depth.

        Ref: https://developers.openai.com/api/docs/guides/reasoning
             stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_reasoning
        """
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="minimal",
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

    @pytest.mark.parametrize("model", DEEPSEEK_SAMPLE)
    def test_reasoning_effort_none_explicit_disable(
        self, openai_client: OpenAI, model: str
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
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="none",
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) >= 1
        choice = resp.choices[0]
        assert choice.finish_reason in FINISH_REASONS
        msg = choice.message
        assert msg.role == "assistant"
        assert msg.content
        assert "ok" in msg.content.lower(), (
            f"Expected the pinned reply for {model!r}, got: {msg.content!r}"
        )
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0
