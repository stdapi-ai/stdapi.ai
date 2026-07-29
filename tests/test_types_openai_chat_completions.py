"""Offline unit tests for the OpenAI Chat Completions request types (no AWS calls)."""

from __future__ import annotations

from typing import Any

import pytest

from stdapi.types.openai_chat_completions import CompletionCreateParams

pytestmark = pytest.mark.local

#: Minimal valid request payload used as a base by the tests.
_BASE_REQUEST: dict[str, Any] = {
    "model": "model",
    "messages": [{"role": "user", "content": "hi"}],
}


class TestReasoningEffort:
    """``reasoning_effort`` accepts every upstream OpenAI literal."""

    @pytest.mark.parametrize(
        "effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    )
    def test_reasoning_effort_literal_is_accepted(self, effort: str) -> None:
        """Each upstream effort level validates and is preserved."""
        request = CompletionCreateParams.model_validate(
            _BASE_REQUEST | {"reasoning_effort": effort}
        )
        assert request.reasoning_effort == effort
