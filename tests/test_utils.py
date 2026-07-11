"""Unit tests for utility helpers (:mod:`stdapi.utils`)."""

from __future__ import annotations

import pytest

from stdapi.utils import hide_security_details


def test_hide_security_details_masks_auth_statuses() -> None:
    """401 and 403 responses return fixed, detail-free messages."""
    assert hide_security_details(401, "invalid key sk-secret") == "Unauthorized"
    assert hide_security_details(403, "forbidden host 10.0.0.1") == "Forbidden"


def test_hide_security_details_redacts_arn() -> None:
    """ARNs are stripped from client-facing error messages on any status."""
    message = (
        "The provided model identifier arn:aws:bedrock:us-east-1:123456789012:"
        "application-inference-profile/abc is invalid"
    )
    redacted = hide_security_details(400, message)
    assert "arn:aws" not in redacted
    assert "123456789012" not in redacted
    assert "<arn>" in redacted


def test_hide_security_details_redacts_bare_account_id() -> None:
    """A bare 12-digit account ID is redacted even without a surrounding ARN."""
    redacted = hide_security_details(502, "Access denied for account 123456789012")
    assert "123456789012" not in redacted
    assert "<account-id>" in redacted


@pytest.mark.parametrize("status", [400, 404, 429, 500, 502, 503])
def test_hide_security_details_preserves_safe_message(status: int) -> None:
    """A message without sensitive identifiers is returned unchanged."""
    message = "Validation error at body.model: field required"
    assert hide_security_details(status, message) == message
