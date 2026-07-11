"""Unit tests for utility helpers (:mod:`stdapi.utils`)."""

from __future__ import annotations

import warnings
from io import BytesIO

import pytest
from PIL import Image
from pybase64 import b64encode

from stdapi import utils
from stdapi.utils import (
    get_base64_image_size,
    hide_security_details,
    match_bedrock_app_profile_arn,
    match_bedrock_prompt_router_arn,
    strip_url_query,
)


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


def test_arn_matcher_accepts_valid_arn() -> None:
    """A well-formed inference-profile ARN matches."""
    arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"
    assert match_bedrock_app_profile_arn(arn) is not None


def test_arn_matcher_rejects_trailing_data() -> None:
    """An ARN followed by extra data is rejected (end-anchored)."""
    arn = (
        "arn:aws:bedrock:us-east-1:123456789012:"
        "application-inference-profile/abc123 and more"
    )
    assert match_bedrock_app_profile_arn(arn) is None


def test_prompt_router_arn_matcher_rejects_trailing_data() -> None:
    """A prompt-router ARN followed by extra data is rejected (end-anchored)."""
    arn = (
        "arn:aws:bedrock:us-east-1:123456789012:"
        "default-prompt-router/my-router injected"
    )
    assert match_bedrock_prompt_router_arn(arn) is None


def test_arn_matcher_rejects_trailing_newline() -> None:
    r"""A trailing newline is rejected (``\Z`` anchor, unlike ``$``)."""
    arn = (
        "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123\n"
    )
    assert match_bedrock_app_profile_arn(arn) is None


async def test_get_base64_image_size_rejects_decompression_bomb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An image exceeding the pixel limit raises a ValueError rather than allocating."""
    buffer = BytesIO()
    Image.new("RGB", (100, 100)).save(buffer, format="PNG")
    encoded = b64encode(buffer.getvalue()).decode()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="pixel size"):
        await get_base64_image_size(encoded)


def test_pillow_threshold_is_half_the_documented_pixel_cap() -> None:
    """Pillow's own threshold must be set to half the documented cap.

    Pillow only raises DecompressionBombError above 2x its threshold, so
    halving it makes the documented `_MAX_IMAGE_PIXELS` the real hard limit.
    """
    assert Image.MAX_IMAGE_PIXELS == utils._MAX_IMAGE_PIXELS // 2  # noqa: SLF001


async def test_get_base64_image_size_rejects_image_just_above_the_documented_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An image just above the documented cap must be rejected (the real hard limit)."""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 50)  # cap/2, cap == 100
    buffer = BytesIO()
    Image.new("RGB", (11, 10)).save(buffer, format="PNG")  # 110 px > 100 cap
    encoded = b64encode(buffer.getvalue()).decode()
    with pytest.raises(ValueError, match="pixel size"):
        await get_base64_image_size(encoded)


async def test_get_base64_image_size_accepts_image_just_below_cap_without_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An image between cap/2 and cap decodes cleanly, with no DecompressionBombWarning."""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 50)  # cap/2, cap == 100
    buffer = BytesIO()
    Image.new("RGB", (11, 9)).save(buffer, format="PNG")  # 99 px: cap/2 < px < cap
    encoded = b64encode(buffer.getvalue()).decode()

    # pytest resets warning filters per-test; reinstall the module-level ignore.
    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
        width, height = await get_base64_image_size(encoded)

    assert (width, height) == (11, 9)
    assert not any(
        issubclass(warning.category, Image.DecompressionBombWarning)
        for warning in caught
    )


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
