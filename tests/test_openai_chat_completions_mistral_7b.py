"""Tests specific to Mistral 7b chat completions.

Tests system prompt handling for models that don't support system prompts.
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


class TestMistral7bChatCompletions:
    """Mistral 7b chat completions tests."""

    @pytest.mark.expensive
    @pytest.mark.parametrize("model", MISTRAL_7B_MODELS)
    def test_system_prompt_silently_dropped_when_enabled(
        self, openai_client: OpenAI, use_openai_api: bool, model: str
    ) -> None:
        """System prompt is silently dropped when DROP_UNSUPPORTED_SYSTEM_PROMPT=true."""
        if use_openai_api:
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
        assert hasattr(resp, "choices")
        assert len(resp.choices) >= 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"
        assert msg.content is not None
        assert len(msg.content) > 0
