"""Extra model parameters, inference config and guardrail helpers in stdapi.aws_bedrock.

Covers gateway-internal helpers with no upstream analogue: the merge of
``default_model_params`` with per-request extras, the denylist filtering out
leaked LiteLLM client-control parameters, the ``inferenceConfig`` builder, the
``X-Amzn-Bedrock-Guardrail*`` header override, guardrail Region resolution,
and the guardrail-assessment to OpenAI-moderation mapping.

Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html
     stdapi/aws_bedrock.py
"""

from typing import TYPE_CHECKING, Any

import pytest
from starlette.datastructures import Headers

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import (
    GUARDRAIL_CONFIG_VAR,
    filter_extra_model_parameters,
    get_extra_model_parameters,
    guardrail_region,
    map_guardrail_filters,
    resolve_guardrail_model,
    set_guardrail_configuration,
    set_inference_configuration,
)
from stdapi.config import SETTINGS
from stdapi.types import BaseModelRequestWithExtra

if TYPE_CHECKING:
    from collections.abc import Iterator

    from types_aiobotocore_bedrock_runtime.type_defs import (
        GuardrailStreamConfigurationTypeDef,
    )

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


@pytest.fixture
def configured_guardrail() -> Iterator[None]:
    """Configure a guardrail (id "gr123", version "1") for the test's context."""
    config: GuardrailStreamConfigurationTypeDef = {
        "guardrailIdentifier": "gr123",
        "guardrailVersion": "1",
    }
    token = GUARDRAIL_CONFIG_VAR.set(config)
    yield
    GUARDRAIL_CONFIG_VAR.reset(token)


class _Request(BaseModelRequestWithExtra):
    """Minimal request carrying arbitrary extra fields."""

    model: str


class TestGetExtraModelParameters:
    """get_extra_model_parameters: default/extra merge and settings isolation.

    The merged mapping is what the gateway forwards as Converse's
    ``additionalModelRequestFields``.

    Ref: stdapi/aws_bedrock.py:get_extra_model_parameters
    """

    def test_merges_defaults_with_request_extras(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Request extras are merged over the configured defaults, taking precedence."""
        monkeypatch.setattr(
            SETTINGS,
            "default_model_params",
            {"model-a": {"temperature": 0.5, "top_k": 10}},
        )
        request = _Request.model_validate(
            {"model": "model-a", "top_k": 99, "max_tokens": 42}
        )

        params = get_extra_model_parameters("model-a", request)

        assert params == {"temperature": 0.5, "top_k": 99, "max_tokens": 42}

    def test_unknown_model_returns_extras_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model with no configured defaults returns only the request extras."""
        monkeypatch.setattr(SETTINGS, "default_model_params", {})
        request = _Request.model_validate({"model": "unknown", "foo": "bar"})

        params = get_extra_model_parameters("unknown", request)

        assert params == {"foo": "bar"}

    def test_extras_do_not_leak_between_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: one request's extras must not persist in SETTINGS or a later call."""
        monkeypatch.setattr(
            SETTINGS, "default_model_params", {"model-a": {"temperature": 0.5}}
        )
        first_request = _Request.model_validate({"model": "model-a", "top_k": 1})
        second_request = _Request.model_validate({"model": "model-a", "max_tokens": 2})

        first_params = get_extra_model_parameters("model-a", first_request)
        second_params = get_extra_model_parameters("model-a", second_request)

        assert first_params == {"temperature": 0.5, "top_k": 1}
        assert second_params == {"temperature": 0.5, "max_tokens": 2}
        assert SETTINGS.default_model_params["model-a"] == {"temperature": 0.5}

    def test_default_dropped_param_does_not_reach_the_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A LiteLLM control key leaked via extra_body never reaches the model body.

        Regression for RAGFlow-style clients hardcoding ``drop_params`` into
        ``extra_body``: the merged parameters sent to Bedrock as
        ``additionalModelRequestFields`` must not include it.
        """
        monkeypatch.setattr(SETTINGS, "default_model_params", {})
        request = _Request.model_validate(
            {"model": "model-a", "drop_params": True, "temperature": 0.5}
        )

        params = get_extra_model_parameters("model-a", request)

        assert params == {"temperature": 0.5}

    def test_non_listed_extra_is_still_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An extra that is not on any denylist keeps reaching the model body.

        Regression guard for the passthrough feature itself: filtering must not
        turn into an accidental allowlist.
        """
        monkeypatch.setattr(SETTINGS, "default_model_params", {})
        request = _Request.model_validate({"model": "model-a", "custom_field": "x"})

        params = get_extra_model_parameters("model-a", request)

        assert params == {"custom_field": "x"}


class TestFilterExtraModelParameters:
    """filter_extra_model_parameters: the denylist shared by every extra-params funnel.

    Ref: stdapi/aws_bedrock.py:filter_extra_model_parameters
    """

    def test_default_dropped_key_is_removed(self) -> None:
        """A key from the built-in denylist is stripped."""
        assert filter_extra_model_parameters({"drop_params": True, "top_k": 1}) == {
            "top_k": 1
        }

    def test_unlisted_key_passes_through(self) -> None:
        """A key absent from every denylist is forwarded unchanged."""
        assert filter_extra_model_parameters({"reasoning_effort": "high"}) == {
            "reasoning_effort": "high"
        }

    def test_denylist_setting_drops_additional_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """extra_model_params_denylist adds to, rather than replaces, the built-in set."""
        monkeypatch.setattr(
            SETTINGS,
            "extra_model_params_denylist",
            SETTINGS.extra_model_params_denylist | {"x_custom_flag"},
        )

        result = filter_extra_model_parameters(
            {"x_custom_flag": True, "drop_params": True, "top_k": 1}
        )

        assert result == {"top_k": 1}

    def test_drop_all_setting_removes_every_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """extra_model_params_drop_all disables the passthrough entirely."""
        monkeypatch.setattr(SETTINGS, "extra_model_params_drop_all", True)

        assert (
            filter_extra_model_parameters({"top_k": 1, "reasoning_effort": "x"}) == {}
        )

    def test_built_in_denylist_excludes_real_request_fields(self) -> None:
        """Regression guard: real request field names must never be denylisted.

        A name that is a legitimate field on some route must never be in the
        built-in denylist, or a real user parameter would be silently swallowed.
        """
        assert SETTINGS.extra_model_params_denylist.isdisjoint(
            {
                "metadata",
                "user",
                "stream",
                "tags",
                "ttl",
                "order",
                "id",
                "client",
                "weight",
                "headers",
                "roles",
                "self",
                "region_name",
            }
        )


@pytest.mark.usefixtures("configured_guardrail")
class TestResolveGuardrailModel:
    """resolve_guardrail_model: version handling against the configured guardrail.

    A moderation ``model`` value is parsed as ``<identifier>[:<version>]``, where
    the version is a numeric version or the literal ``DRAFT``, matching
    ApplyGuardrail's ``guardrailVersion`` pattern.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
         stdapi/aws_bedrock.py:resolve_guardrail_model
    """

    def test_no_version_uses_configured_version(self) -> None:
        """Naming the configured guardrail with no version returns its configured version."""
        assert resolve_guardrail_model("gr123") == ("gr123", "1")

    def test_matching_version_uses_configured_version(self) -> None:
        """Naming the configured guardrail with its own version returns that version."""
        assert resolve_guardrail_model("gr123:1") == ("gr123", "1")

    def test_different_version_is_honored_when_override_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: a different explicit version must not be silently replaced."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", True)
        assert resolve_guardrail_model("gr123:2") == ("gr123", "2")

    def test_different_version_is_rejected_when_override_disallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A different explicit version falls through to the override-permission check.

        The identifier matches the configured guardrail, so only the version
        differs — that alone must be treated as an override attempt and refused
        as a 400 rather than being silently coerced to the configured version.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", False)
        with pytest.raises(ApiError) as not_allowed:
            resolve_guardrail_model("gr123:2")
        assert not_allowed.value.status == 400
        assert "Selecting a guardrail via 'model' is not allowed" in str(
            not_allowed.value
        )


def _content_filter_assessment(
    filter_type: str, confidence: str, action: str = "BLOCKED"
) -> dict[str, Any]:
    """Build a guardrail assessment with one content policy filter entry."""
    return {
        "contentPolicy": {
            "filters": [
                {"type": filter_type, "confidence": confidence, "action": action}
            ]
        }
    }


class TestSetInferenceConfiguration:
    """set_inference_configuration: explicit falsy values must not be dropped.

    The result is Converse's ``inferenceConfig``, whose members are exactly
    ``maxTokens``, ``stopSequences``, ``temperature`` and ``topP`` — hence the
    camelCase renaming of the OpenAI-style arguments.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         stdapi/aws_bedrock.py:set_inference_configuration
    """

    def test_zero_temperature_and_top_p_are_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: temperature=0 and top_p=0 are valid values, not "unset"."""
        monkeypatch.setattr(SETTINGS, "default_model_params", {})

        config = set_inference_configuration(
            "model-a", {}, temperature=0.0, top_p=0.0, max_tokens=100
        )

        assert config == {"temperature": 0.0, "topP": 0.0, "maxTokens": 100}

    def test_zero_temperature_overrides_configured_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit request temperature of 0 must take precedence over the default."""
        monkeypatch.setattr(
            SETTINGS, "default_model_params", {"model-a": {"temperature": 0.7}}
        )

        config = set_inference_configuration("model-a", {}, temperature=0.0)

        assert config["temperature"] == 0.0

    def test_unset_temperature_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omitting temperature still falls back to the configured default."""
        monkeypatch.setattr(
            SETTINGS, "default_model_params", {"model-a": {"temperature": 0.7}}
        )

        config = set_inference_configuration("model-a", {})

        assert config["temperature"] == 0.7


@pytest.fixture
def _guardrail_config_var_isolated() -> Iterator[None]:
    """Restore GUARDRAIL_CONFIG_VAR to its pre-test state, whatever the test sets it to."""
    token = GUARDRAIL_CONFIG_VAR.set(None)  # type: ignore[arg-type]
    try:
        yield
    finally:
        GUARDRAIL_CONFIG_VAR.reset(token)


@pytest.mark.usefixtures("_guardrail_config_var_isolated")
class TestSetGuardrailConfiguration:
    """set_guardrail_configuration: header-based override, incl. stream processing mode.

    The inbound header names mirror InvokeModel's ``X-Amzn-Bedrock-Guardrail*``
    headers, and ``streamProcessingMode`` is allow-listed against Bedrock's
    ``sync``/``async`` enum because it exists only on ConverseStream's
    ``GuardrailStreamConfiguration``.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html
         https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html
         stdapi/aws_bedrock.py:set_guardrail_configuration
    """

    def test_stream_processing_mode_header_is_applied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A valid streamProcessingMode header is forwarded into the guardrail config."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", True)
        headers = Headers(
            {
                "X-Amzn-Bedrock-GuardrailIdentifier": "gr123",
                "X-Amzn-Bedrock-GuardrailVersion": "1",
                "X-Amzn-Bedrock-GuardrailStreamProcessingMode": "async",
            }
        )

        set_guardrail_configuration(headers)

        assert GUARDRAIL_CONFIG_VAR.get() == {
            "guardrailIdentifier": "gr123",
            "guardrailVersion": "1",
            "streamProcessingMode": "async",
        }

    def test_invalid_stream_processing_mode_header_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unrecognised streamProcessingMode value is dropped, not forwarded verbatim.

        The rest of the header-supplied guardrail still applies: only the
        off-enum value is discarded, so Bedrock never sees it.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", True)
        headers = Headers(
            {
                "X-Amzn-Bedrock-GuardrailIdentifier": "gr123",
                "X-Amzn-Bedrock-GuardrailVersion": "1",
                "X-Amzn-Bedrock-GuardrailStreamProcessingMode": "bogus",
            }
        )

        set_guardrail_configuration(headers)

        assert GUARDRAIL_CONFIG_VAR.get() == {
            "guardrailIdentifier": "gr123",
            "guardrailVersion": "1",
        }

    def test_stream_processing_mode_header_ignored_without_override_permission(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The header is ignored (falls back to settings) when override is disallowed.

        Both the identifier/version pair and the stream processing mode come
        from the server configuration, so a client cannot redirect its request
        to another guardrail.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", False)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", "gr-default")
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", "2")
        headers = Headers(
            {
                "X-Amzn-Bedrock-GuardrailIdentifier": "gr123",
                "X-Amzn-Bedrock-GuardrailVersion": "1",
                "X-Amzn-Bedrock-GuardrailStreamProcessingMode": "async",
            }
        )

        set_guardrail_configuration(headers)

        config = GUARDRAIL_CONFIG_VAR.get()
        assert config["guardrailIdentifier"] == "gr-default"
        assert config["guardrailVersion"] == "2"
        assert "streamProcessingMode" not in config


class TestGuardrailRegion:
    """guardrail_region: region validation against configured Bedrock regions.

    ApplyGuardrail accepts either a bare identifier or a full guardrail ARN; the
    gateway reads the Region out of ARN field 3 and refuses to call a Region the
    server was not configured for.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
         stdapi/aws_bedrock.py:guardrail_region
    """

    def test_arn_region_is_returned_when_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ARN naming a configured region resolves to that region."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1", "eu-west-1"])
        arn = "arn:aws:bedrock:eu-west-1:123456789012:guardrail/gr123"
        assert guardrail_region(arn) == "eu-west-1"

    def test_arn_region_not_configured_raises_api_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: an ARN naming an unconfigured region must not crash with KeyError."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1"])
        arn = "arn:aws:bedrock:ap-south-1:123456789012:guardrail/gr123"
        with pytest.raises(ApiError) as unconfigured:
            guardrail_region(arn)
        assert unconfigured.value.status == 400
        assert "ap-south-1" in str(unconfigured.value), (
            "the rejected region must be named in the error"
        )
        assert "not a configured Bedrock region" in str(unconfigured.value)

    def test_bare_identifier_uses_primary_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare (non-ARN) identifier resolves to the first configured region."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1", "eu-west-1"])
        assert guardrail_region("gr123") == "us-east-1"


class TestMapGuardrailFilters:
    """map_guardrail_filters: pinned filter/category and confidence/score tables.

    Bedrock Guardrails report a confidence level per content filter, not a
    score, so both the five-filter to OpenAI-category mapping and the
    level-to-float table are gateway inventions with no upstream equivalent.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
         stdapi/aws_bedrock.py:map_guardrail_filters
    """

    @pytest.mark.parametrize(
        ("filter_type", "category"),
        [
            ("HATE", "hate"),
            ("INSULTS", "harassment"),
            ("SEXUAL", "sexual"),
            ("VIOLENCE", "violence"),
            ("MISCONDUCT", "illicit"),
        ],
    )
    def test_full_filter_to_category_mapping(
        self, filter_type: str, category: str
    ) -> None:
        """Every guardrail content filter maps to its OpenAI category."""
        categories, scores, intervened = map_guardrail_filters(
            [_content_filter_assessment(filter_type, "HIGH")]
        )
        assert categories == {category: True}
        assert scores == {category: 0.75}
        assert intervened is True

    @pytest.mark.parametrize(
        ("confidence", "score"),
        [("NONE", 0.0), ("LOW", 0.25), ("MEDIUM", 0.5), ("HIGH", 0.75)],
    )
    def test_full_confidence_to_score_mapping(
        self, confidence: str, score: float
    ) -> None:
        """Every guardrail confidence level maps to its OpenAI-style score.

        ``action: NONE`` means the guardrail did not act, so the score is
        reported while the category stays false and no intervention is flagged.
        """
        categories, scores, intervened = map_guardrail_filters(
            [_content_filter_assessment("HATE", confidence, action="NONE")]
        )
        assert scores == {"hate": score}
        assert categories == {"hate": False}
        assert intervened is False
