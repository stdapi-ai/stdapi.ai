"""Tests for stdapi.aws_bedrock.get_extra_model_parameters and resolve_guardrail_model."""

from typing import TYPE_CHECKING

import pytest

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import (
    GUARDTRAIL_CONFIG_VAR,
    get_extra_model_parameters,
    resolve_guardrail_model,
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
    token = GUARDTRAIL_CONFIG_VAR.set(config)
    yield
    GUARDTRAIL_CONFIG_VAR.reset(token)


class _Request(BaseModelRequestWithExtra):
    """Minimal request carrying arbitrary extra fields."""

    model: str


class TestGetExtraModelParameters:
    """get_extra_model_parameters: default/extra merge and settings isolation."""

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


@pytest.mark.usefixtures("configured_guardrail")
class TestResolveGuardrailModel:
    """resolve_guardrail_model: version handling against the configured guardrail."""

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
        """A different explicit version falls through to the override-permission check."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", False)
        with pytest.raises(ApiError, match="not allowed"):
            resolve_guardrail_model("gr123:2")
