"""Tests specific to Anthropic Claude Responses API behavior."""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

#: Anthropic models supporting all Claude features.
CLAUDE_ALL = (
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-fable-5",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-opus-4-7",
    "anthropic.claude-opus-4-8",
    "anthropic.claude-opus-5",
    "anthropic.claude-sonnet-5",
)


class TestClaudeReasoning:
    """Tests for Claude extended thinking via the OpenAI Responses route."""

    @pytest.mark.expensive
    @pytest.mark.parametrize("model", CLAUDE_ALL)
    def test_reasoning_effort_accepted(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """reasoning.effort is accepted and produces a response with text output.

        Validates:
            - ``reasoning.effort`` does not raise an error
            - Response contains a message output item with non-empty text
            - Response status is ``"completed"``
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        resp = openai_client.responses.create(
            model=model,
            input="Reply with OK.",
            reasoning={"effort": "low"},
            max_output_tokens=4096,
        )
        assert resp.output_text
        assert resp.status == "completed"
