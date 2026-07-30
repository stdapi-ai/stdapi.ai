"""Chat Completions on the Mistral 7B family, which has no system-prompt slot.

Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-text-completion.html
     stdapi/models/chat/mistral_7b.py:ChatModel
"""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

#: Mistral models without system prompt support
MISTRAL_7B_MODELS = (
    "mistral.mistral-7b-instruct-v0:2",
    "mistral.mixtral-8x7b-instruct-v0:1",
)

#: finish_reason values the OpenAI Chat Completions reference defines.
_FINISH_REASONS = frozenset({"stop", "length", "content_filter", "tool_calls"})


class TestMistral7bChatCompletions:
    """Mistral 7b chat completions tests.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html
         stdapi/models/chat/mistral_7b.py:ChatModel
    """

    @pytest.mark.parametrize("model", MISTRAL_7B_MODELS)
    def test_system_prompt_silently_dropped_when_enabled(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """A ``system`` message is dropped instead of failing on Mistral 7B / Mixtral 8x7B.

        These models encode roles inside the ``[INST] … [/INST]`` prompt template and
        have no Converse ``system`` field, so Bedrock rejects a request carrying one.
        ``ChatModel.SYSTEM_PROMPT_SUPPORTED = False`` plus the default
        ``drop_unsupported_system_prompt=True`` makes the gateway omit the block, so the
        request succeeds; with the drop removed this call would surface a 400 instead.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-text-completion.html
             stdapi/models/chat/_default.py:ChatModel._prepare_converse_request
             stdapi/config.py:Settings.drop_unsupported_system_prompt
        """
        if use_official_api:
            pytest.skip(
                "Mistral 7b models are not supported on the official OpenAI API"
            )
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with OK."},
            ],
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) >= 1
        choice = resp.choices[0]
        assert choice.index == 0
        assert choice.finish_reason in _FINISH_REASONS
        msg = choice.message
        assert msg.role == "assistant"
        assert msg.content
        assert "ok" in msg.content.lower(), (
            f"Expected the pinned reply for {model!r}, got: {msg.content!r}"
        )
        assert resp.usage is not None
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0
        assert (
            resp.usage.total_tokens
            == resp.usage.prompt_tokens + resp.usage.completion_tokens
        )
