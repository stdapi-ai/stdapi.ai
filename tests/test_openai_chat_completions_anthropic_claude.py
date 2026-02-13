"""Tests specific to Anthropic Claude chat completions."""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

#: Anthropic models supporting reasoning
CLAUDE_ALL = (
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-opus-4-1-20250805-v1:0",
    "anthropic.claude-opus-4-20250514-v1:0",
    "anthropic.claude-opus-4-5-20251101-v1:0",
    "anthropic.claude-opus-4-6-v1",
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
)


class TestAnthropicClaudeChatCompletions:
    """Anthropic Claude chat completions tests."""

    @pytest.mark.expensive
    @pytest.mark.parametrize("model", CLAUDE_ALL)
    def test_reasoning_effort_parameter(
        self, openai_client: OpenAI, use_openai_api: bool, model: str
    ) -> None:
        """reasoning_effort parameter: accepted and yields valid response on this backend."""
        if use_openai_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="minimal",
            max_completion_tokens=4096,  # Required for Opus 4.1
        )
        assert hasattr(resp, "choices")
        assert len(resp.choices) >= 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"
