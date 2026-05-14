"""Tests specific to Deepseek chat completions."""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

DEEPSEEK_V3 = "deepseek.v3-v1:0"
DEEPSEEK_V3_2 = "deepseek.v3.2"

DEEPSEEK_ALL = (DEEPSEEK_V3, DEEPSEEK_V3_2)
DEEPSEEK_SAMPLE = (DEEPSEEK_V3_2,)


class TestDeepseekChatCompletions:
    """Deepseek chat completions tests."""

    @pytest.mark.parametrize("model", DEEPSEEK_SAMPLE)
    def test_reasoning_effort_parameter(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """reasoning_effort parameter: accepted and yields valid response on this backend."""
        if use_official_api:
            pytest.skip("Deepseek is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="minimal",
        )
        assert hasattr(resp, "choices")
        assert len(resp.choices) >= 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"

    @pytest.mark.expensive
    @pytest.mark.parametrize("model", DEEPSEEK_SAMPLE)
    def test_reasoning_effort_none_explicit_disable(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """reasoning_effort='none' explicitly disables reasoning."""
        if use_official_api:
            pytest.skip("Deepseek is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="none",
        )
        assert hasattr(resp, "choices")
        assert len(resp.choices) >= 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"
