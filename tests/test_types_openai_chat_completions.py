"""Offline unit tests for the OpenAI Chat Completions request model (no AWS calls).

Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
     stdapi/types/openai_chat_completions.py:CompletionCreateParams
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from stdapi.api_errors import UnsupportedParameterError
from stdapi.config import SETTINGS
from stdapi.models.chat._adapters._openai_common import resolve_cache_ttl
from stdapi.types.openai_chat_completions import (
    ChatCompletionMessage,
    ChoiceDelta,
    CompletionCreateParams,
    ReasoningOptions,
)

pytestmark = pytest.mark.local

#: Minimal valid request payload used as a base by the tests.
_BASE_REQUEST: dict[str, Any] = {
    "model": "model",
    "messages": [{"role": "user", "content": "hi"}],
}

#: Tool result carrying the extra ``name`` field clients inherited from the
#: deprecated function-calling API.
_NAMED_TOOL_RESULT: dict[str, Any] = {
    "role": "tool",
    "tool_call_id": "call_1",
    "content": "42",
    "name": "compute",
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


class TestReasoningObjectNormalizationInputs:
    """``reasoning`` is folded onto the flat fields whatever shape it arrives in.

    The normalizer runs in ``before`` mode, so it sees whatever the caller
    passed: a JSON object from a client, an already-built ``ReasoningOptions``
    from in-process construction, or a whole request object being re-validated.
    Only the mapping shapes carry values to fold; the rest must pass through
    untouched instead of raising.

    Ref: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
         stdapi/types/openai_chat_completions.py:CompletionCreateParams._normalize_reasoning
    """

    def test_a_reasoning_options_instance_is_folded(self) -> None:
        """An in-process ``ReasoningOptions`` configures the flat fields too."""
        request = CompletionCreateParams.model_validate(
            _BASE_REQUEST | {"reasoning": ReasoningOptions(max_tokens=2048)}
        )
        assert request.thinking_budget == 2048
        assert request.enable_thinking is True, (
            "an explicit budget implies reasoning is on, as thinking_budget requires"
        )

    def test_an_instance_still_enforces_the_conflict_rules(self) -> None:
        """The mutual exclusions apply to the object form as well as the JSON one."""
        with pytest.raises(ValidationError, match="Only one of"):
            CompletionCreateParams.model_validate(
                _BASE_REQUEST
                | {"reasoning": ReasoningOptions(effort="high", max_tokens=2048)}
            )

    def test_a_non_mapping_body_is_a_validation_error(self) -> None:
        """A JSON array body fails validation instead of crashing the normalizer.

        The normalizer runs before pydantic has checked the payload shape, so it
        must hand a non-mapping straight back; touching it would turn a client's
        malformed body into a 500.
        """
        with pytest.raises(ValidationError) as excinfo:
            CompletionCreateParams.model_validate([_BASE_REQUEST])
        assert excinfo.value.errors()[0]["type"] == "model_type"

    def test_a_non_mapping_reasoning_value_is_left_to_the_field(self) -> None:
        """A ``reasoning`` that is not an object is rejected by the field, not folded."""
        with pytest.raises(ValidationError, match="reasoning"):
            CompletionCreateParams.model_validate(_BASE_REQUEST | {"reasoning": "high"})


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


class TestStrictValidationOfMessageFields:
    """``strict_input_validation`` decides the fate of an undeclared message field.

    The setting is what makes the difference between a deployment that tolerates
    a client's extra field and one that refuses the request, and it is read when
    the models are built, so only a whole session can hold one value. This pins
    the strict half; the permissive half is the shipped default and what the
    agentic lane's clients are run against.

    ``name`` on a ``tool`` message is the real case: it belongs to the legacy
    ``function`` role, is absent from the tool message the OpenAI SDK defines,
    and clients still send it -- Hermes does on every tool result.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         stdapi/types/__init__.py:BaseModelRequest
         stdapi/config.py:Settings.strict_input_validation
         tests/agentic/_server.py:_OVERRIDDEN_SETTINGS
    """

    @pytest.mark.skipif(
        not SETTINGS.strict_input_validation,
        reason="the models were built permissive; extra fields are ignored",
    )
    def test_an_undeclared_tool_message_field_is_refused(self) -> None:
        """Strict mode rejects ``name`` on a tool message, naming the field."""
        with pytest.raises(ValidationError) as excinfo:
            CompletionCreateParams.model_validate(
                _BASE_REQUEST | {"messages": [_NAMED_TOOL_RESULT]}
            )
        assert "name" in str(excinfo.value)

    def test_the_declared_fields_still_validate(self) -> None:
        """The same message without the extra field validates in either mode.

        The control: it is the undeclared field that is refused above, not the
        tool result itself.
        """
        message = {k: v for k, v in _NAMED_TOOL_RESULT.items() if k != "name"}
        request = CompletionCreateParams.model_validate(
            _BASE_REQUEST | {"messages": [message]}
        )
        assert request.messages[0].tool_call_id == "call_1"  # type: ignore[union-attr]


@pytest.mark.local
class TestReasoningFieldSetting:
    """The emitted reasoning field name is an operator setting, with an off switch.

    OpenAI's own Chat Completions returns no thinking text, so the vendors that
    do have split on the name: DeepSeek and the clients that followed it read
    ``reasoning_content``, while OpenRouter and vLLM emit ``reasoning``. A
    per-request selector would be a gateway-specific API field, which the design
    rules forbid, so the choice is made once by whoever runs the gateway --
    including ``none``, which keeps responses strictly OpenAI-shaped.

    Ref: https://api-docs.deepseek.com/guides/reasoning_model
         https://openrouter.ai/docs/use-cases/reasoning-tokens
         stdapi/types/openai_chat_completions.py:_rename_emitted_reasoning
    """

    @pytest.mark.parametrize(
        ("setting", "expected"),
        [
            ("reasoning_content", "reasoning_content"),
            ("reasoning", "reasoning"),
            ("none", None),
        ],
    )
    def test_the_message_carries_the_configured_field(
        self, monkeypatch: pytest.MonkeyPatch, setting: str, expected: str | None
    ) -> None:
        """A completed message emits the thinking text under the chosen name."""
        monkeypatch.setattr(SETTINGS, "chat_completions_reasoning_field", setting)

        dumped = ChatCompletionMessage(
            role="assistant", content="45", reasoning_content="Let total be T."
        ).model_dump(exclude_none=True)

        assert dumped["content"] == "45"
        if expected is None:
            assert "reasoning" not in dumped
            assert "reasoning_content" not in dumped
        else:
            assert dumped[expected] == "Let total be T."
            assert len({"reasoning", "reasoning_content"} & dumped.keys()) == 1

    @pytest.mark.parametrize(
        ("setting", "expected"),
        [
            ("reasoning_content", "reasoning_content"),
            ("reasoning", "reasoning"),
            ("none", None),
        ],
    )
    def test_the_streamed_delta_follows_the_same_setting(
        self, monkeypatch: pytest.MonkeyPatch, setting: str, expected: str | None
    ) -> None:
        """A stream must not name the field differently from the final message."""
        monkeypatch.setattr(SETTINGS, "chat_completions_reasoning_field", setting)

        dumped = ChoiceDelta(
            content="45", reasoning_content="Let total be T."
        ).model_dump(exclude_none=True)

        if expected is None:
            assert "reasoning" not in dumped
            assert "reasoning_content" not in dumped
        else:
            assert dumped[expected] == "Let total be T."

    @pytest.mark.parametrize("setting", ["reasoning_content", "reasoning", "none"])
    def test_a_message_without_reasoning_is_untouched(
        self, monkeypatch: pytest.MonkeyPatch, setting: str
    ) -> None:
        """No setting adds a reasoning key to a message that carries none."""
        monkeypatch.setattr(SETTINGS, "chat_completions_reasoning_field", setting)

        dumped = ChatCompletionMessage(role="assistant", content="45").model_dump(
            exclude_none=True
        )

        assert dumped == {"role": "assistant", "content": "45"}
