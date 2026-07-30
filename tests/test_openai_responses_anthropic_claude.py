"""Anthropic Claude-specific behavior on the OpenAI /v1/responses route.

Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
     https://platform.claude.com/docs/en/build-with-claude/extended-thinking
     stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel
"""

from typing import TYPE_CHECKING

import pytest

from tests.test_openai_chat_completions_anthropic_claude import CLAUDE_ALL

if TYPE_CHECKING:
    from openai import OpenAI


class TestClaudeReasoning:
    """Claude extended thinking driven by the Responses ``reasoning`` object.

    Ref: https://developers.openai.com/api/docs/guides/reasoning
         stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_reasoning
    """

    @pytest.mark.expensive
    @pytest.mark.parametrize("model", CLAUDE_ALL)
    def test_reasoning_effort_accepted(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """``reasoning.effort`` is accepted by every Claude generation in the catalog.

        Without an explicit ``budget_tokens`` the gateway configures adaptive
        thinking (``reasoning_config`` plus ``output_config.effort``), which every
        Claude generation from 4.5 through 5 accepts — including Fable, which
        cannot have reasoning disabled at all.  ``max_output_tokens`` is 4096 so
        thinking tokens cannot starve the visible answer and truncate the response.

        The presence of a ``reasoning`` output item is deliberately not asserted:
        Claude models served through Bedrock Mantle answer on the Messages API and
        their thinking blocks are dropped by the Responses conversion.

        Ref: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
             stdapi/models/chat/_mantle/_convert.py:_chat_to_responses_response
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        resp = openai_client.responses.create(
            model=model,
            input="Reply with OK.",
            reasoning={"effort": "low"},
            max_output_tokens=4096,
        )
        assert resp.status == "completed"
        assert resp.error is None
        assert resp.incomplete_details is None

        messages = [item for item in resp.output if item.type == "message"]
        assert messages, (
            f"No message item for {model!r}: {[i.type for i in resp.output]}"
        )
        assert messages[0].role == "assistant"
        assert resp.output_text

        assert resp.usage is not None
        assert resp.usage.input_tokens > 0
        assert resp.usage.output_tokens > 0
        assert resp.usage.total_tokens == (
            resp.usage.input_tokens + resp.usage.output_tokens
        )
