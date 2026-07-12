"""Tests for stdapi.aws_bedrock.resolve_guardrail_model."""

from typing import TYPE_CHECKING

import pytest

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import GUARDTRAIL_CONFIG_VAR, resolve_guardrail_model
from stdapi.config import SETTINGS

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
