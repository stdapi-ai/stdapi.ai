"""Tests specific to Moonshot Kimi K2.5 chat completions."""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

KIMI_K2_5 = "moonshotai.kimi-k2.5"
KIMI_K2_THINKING = "moonshot.kimi-k2-thinking"

KIMI_ALL = (KIMI_K2_5, KIMI_K2_THINKING)


class TestKimiK25ChatCompletions:
    """Moonshot Kimi K2.5 chat completions tests."""

    @pytest.mark.parametrize("model", KIMI_ALL)
    def test_thinking_not_set(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """Without enable_thinking: defaults to no thinking (or model default)."""
        if use_official_api:
            pytest.skip("Kimi is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Reply with OK."}]
        )
        assert hasattr(resp, "choices")
        assert len(resp.choices) >= 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"

    @pytest.mark.parametrize("model", KIMI_ALL)
    def test_thinking_disabled(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """enable_thinking=False explicitly disables thinking."""
        if use_official_api:
            pytest.skip("Kimi is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            extra_body={"enable_thinking": False},
        )
        assert hasattr(resp, "choices")
        assert len(resp.choices) >= 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"

    @pytest.mark.expensive
    @pytest.mark.parametrize("model", KIMI_ALL)
    def test_thinking_enabled(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """enable_thinking=True enables thinking."""
        if use_official_api:
            pytest.skip("Kimi is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What is 1+1? Reply with the answer only."}
            ],
            extra_body={"enable_thinking": True},
            max_completion_tokens=512,
        )
        assert hasattr(resp, "choices")
        assert len(resp.choices) >= 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"
