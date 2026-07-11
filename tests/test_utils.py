"""Unit tests for utility helpers (:mod:`stdapi.utils`)."""

from __future__ import annotations

import pytest

from stdapi.utils import hide_security_details, strip_url_query


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


def test_strip_url_query_removes_presigned_signature() -> None:
    """A presigned URL query is replaced by an explicit redaction marker."""
    url = (
        "https://bucket.s3.amazonaws.com/key.png"
        "?X-Amz-Credential=AKIAEXAMPLE&X-Amz-Signature=deadbeef"
    )
    result = strip_url_query(url)
    assert result == "https://bucket.s3.amazonaws.com/key.png?<redacted>"
    assert "X-Amz-Signature" not in result
    assert "AKIAEXAMPLE" not in result


def test_strip_url_query_leaves_plain_url_untouched() -> None:
    """A URL without a query string is returned unchanged."""
    assert strip_url_query("https://example.com/a/b") == "https://example.com/a/b"


def test_http_input_file_repr_hides_presigned_signature() -> None:
    """An HTTP InputFile never exposes its presigned signature via repr."""
    from stdapi.input_file import InputFile  # noqa: PLC0415

    url = (
        "https://bucket.s3.amazonaws.com/key.png"
        "?X-Amz-Credential=AKIAEXAMPLE&X-Amz-Signature=deadbeef"
    )
    text = repr(InputFile(url))
    assert "X-Amz-Signature" not in text
    assert "AKIAEXAMPLE" not in text
    assert text == "https://bucket.s3.amazonaws.com/key.png?<redacted>"
