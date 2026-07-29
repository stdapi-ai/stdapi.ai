"""Offline unit tests for the OpenAI Chat Completions request types (no AWS calls)."""

from __future__ import annotations

from typing import Any

import pytest

from stdapi.api_errors import UnsupportedParameterError
from stdapi.models.chat._adapters._openai_common import resolve_cache_ttl
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


class TestPromptCacheOptions:
    """``prompt_cache_options`` is modeled, not forwarded to the model as an extra."""

    def test_ttl_does_not_mutate_the_request(self) -> None:
        """`ttl` resolves at consumption: the parsed request stays what the client sent.

        Injecting the Bedrock-mapped retention here would leak it into the
        Mantle passthrough payload, where it is not a valid upstream value.
        """
        request = CompletionCreateParams.model_validate(
            _BASE_REQUEST | {"prompt_cache_options": {"mode": "explicit", "ttl": "30m"}}
        )
        assert request.prompt_cache_retention is None
        assert "prompt_cache_retention" not in request.model_fields_set
        assert request.model_extra == {}
        assert resolve_cache_ttl(None, request.prompt_cache_options) is not None

    def test_explicit_prompt_cache_retention_wins(self) -> None:
        """An explicit `prompt_cache_retention` takes precedence over `ttl`."""
        request = CompletionCreateParams.model_validate(
            _BASE_REQUEST
            | {"prompt_cache_options": {"ttl": "30m"}, "prompt_cache_retention": "5m"}
        )
        assert request.prompt_cache_retention == "5m"
        assert resolve_cache_ttl(
            request.prompt_cache_retention, request.prompt_cache_options
        ) == resolve_cache_ttl("5m", None)


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
