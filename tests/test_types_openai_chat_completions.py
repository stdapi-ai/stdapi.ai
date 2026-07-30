"""Offline unit tests for the OpenAI Chat Completions request model (no AWS calls).

Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
     stdapi/types/openai_chat_completions.py:CompletionCreateParams
"""

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
    """``reasoning_effort`` accepts every upstream OpenAI literal.

    ``CompletionCreateParams`` allows extra fields, so an unmodelled parameter
    would still be readable as an attribute: acceptance is only meaningful when
    the value is also shown to land on the declared field instead of
    ``model_extra``.

    Ref: https://developers.openai.com/api/docs/guides/reasoning
    """

    @pytest.mark.parametrize(
        "effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    )
    def test_reasoning_effort_literal_is_accepted(self, effort: str) -> None:
        """Each upstream effort level validates onto the declared field.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        """
        request = CompletionCreateParams.model_validate(
            _BASE_REQUEST | {"reasoning_effort": effort}
        )
        assert request.reasoning_effort == effort
        assert request.model_extra == {}, (
            "reasoning_effort must be consumed by the declared Literal, "
            "not stored as an extra field"
        )


class TestPromptCacheOptions:
    """``prompt_cache_options`` is modeled, not forwarded to the model as an extra.

    Bedrock has no 30-minute cache TTL, so the gateway resolves the upstream
    ``30m`` minimum lifetime to the closest covering Bedrock TTL (``1h``), and
    resolves it at consumption time rather than by rewriting the request.

    Ref: https://developers.openai.com/api/docs/guides/prompt-caching
         stdapi/models/chat/_adapters/_openai_common.py:resolve_cache_ttl
    """

    def test_ttl_does_not_mutate_the_request(self) -> None:
        """`ttl` resolves at consumption: the parsed request stays what the client sent.

        Injecting the Bedrock-mapped retention here would leak it into the
        Mantle passthrough payload, where it is not a valid upstream value.
        """
        request = CompletionCreateParams.model_validate(
            _BASE_REQUEST | {"prompt_cache_options": {"mode": "explicit", "ttl": "30m"}}
        )
        assert request.prompt_cache_options is not None
        assert request.prompt_cache_options.mode == "explicit"
        assert request.prompt_cache_options.ttl == "30m"
        assert request.prompt_cache_retention is None
        assert "prompt_cache_retention" not in request.model_fields_set
        assert request.model_extra == {}
        assert resolve_cache_ttl(None, request.prompt_cache_options) == "1h"

    def test_explicit_prompt_cache_retention_wins(self) -> None:
        """An explicit `prompt_cache_retention` takes precedence over `ttl`."""
        request = CompletionCreateParams.model_validate(
            _BASE_REQUEST
            | {"prompt_cache_options": {"ttl": "30m"}, "prompt_cache_retention": "5m"}
        )
        assert request.prompt_cache_retention == "5m"
        resolved = resolve_cache_ttl(
            request.prompt_cache_retention, request.prompt_cache_options
        )
        assert resolved == "5m"
        assert resolved != resolve_cache_ttl(None, request.prompt_cache_options), (
            "retention must win over the `30m` ttl, which maps to Bedrock `1h`"
        )


class TestUnsupportedParameters:
    """Unsupported parameters are rejected only when actually requested.

    ``_UNSUPPORTED`` names upstream-documented parameters Bedrock cannot honor.
    A ``null``/``false`` value asks for the behavior the gateway already
    provides, so it is treated as omission instead of a 400.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         stdapi/types/openai_chat_completions.py:CompletionCreateParams._unsupported
    """

    @pytest.mark.parametrize("value", [False, None])
    def test_disabled_logprobs_is_accepted(self, value: bool | None) -> None:
        """`logprobs` set to its default (`false`/`null`) behaves like omission."""
        request = CompletionCreateParams.model_validate(
            _BASE_REQUEST | {"logprobs": value}
        )
        assert "logprobs" in request.model_fields_set, (
            "the value must be seen as explicitly sent, not dropped by validation"
        )
        assert request.logprobs is value

    @pytest.mark.parametrize(
        "key", ["prediction", "verbosity", "web_search_options", "translation_options"]
    )
    def test_explicit_null_unsupported_parameter_is_accepted(self, key: str) -> None:
        """An unsupported parameter explicitly set to `null` is accepted."""
        request = CompletionCreateParams.model_validate(_BASE_REQUEST | {key: None})
        assert key in request.model_fields_set
        assert getattr(request, key) is None

    def test_enabled_logprobs_is_rejected(self) -> None:
        """`logprobs: true` still raises UnsupportedParameterError.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """
        with pytest.raises(UnsupportedParameterError) as excinfo:
            CompletionCreateParams.model_validate(_BASE_REQUEST | {"logprobs": True})
        assert excinfo.value.status == 400
        assert excinfo.value.code == "unsupported_parameter"
        assert excinfo.value.param == "logprobs"
        assert "'logprobs' is not supported" in str(excinfo.value)

    def test_verbosity_value_is_rejected(self) -> None:
        """A non-null unsupported parameter still raises UnsupportedParameterError.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """
        with pytest.raises(UnsupportedParameterError) as excinfo:
            CompletionCreateParams.model_validate(_BASE_REQUEST | {"verbosity": "low"})
        assert excinfo.value.status == 400
        assert excinfo.value.code == "unsupported_parameter"
        assert excinfo.value.param == "verbosity"
        assert "'verbosity' is not supported" in str(excinfo.value)
