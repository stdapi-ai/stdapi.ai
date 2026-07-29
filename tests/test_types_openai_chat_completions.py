"""Offline unit tests for the OpenAI Chat Completions request types (no AWS calls)."""

from __future__ import annotations

from typing import Any

import pytest

from stdapi.api_errors import UnsupportedParameterError
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


class TestUnsupportedParameters:
    """Unsupported parameters are rejected only when actually requested."""

    @pytest.mark.parametrize("value", [False, None])
    def test_disabled_logprobs_is_accepted(self, value: bool | None) -> None:
        """`logprobs` set to its default (`false`/`null`) behaves like omission."""
        request = CompletionCreateParams.model_validate(
            _BASE_REQUEST | {"logprobs": value}
        )
        assert request.logprobs == value

    @pytest.mark.parametrize(
        "key", ["prediction", "verbosity", "web_search_options", "translation_options"]
    )
    def test_explicit_null_unsupported_parameter_is_accepted(self, key: str) -> None:
        """An unsupported parameter explicitly set to `null` is accepted."""
        assert (
            CompletionCreateParams.model_validate(_BASE_REQUEST | {key: None})
            is not None
        )

    def test_enabled_logprobs_is_rejected(self) -> None:
        """`logprobs: true` still raises UnsupportedParameterError."""
        with pytest.raises(UnsupportedParameterError):
            CompletionCreateParams.model_validate(_BASE_REQUEST | {"logprobs": True})

    def test_verbosity_value_is_rejected(self) -> None:
        """A non-null unsupported parameter still raises UnsupportedParameterError."""
        with pytest.raises(UnsupportedParameterError):
            CompletionCreateParams.model_validate(_BASE_REQUEST | {"verbosity": "low"})
