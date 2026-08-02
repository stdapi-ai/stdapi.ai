"""Utility helpers in :mod:`stdapi.utils`.

Three groups: the client-facing message redaction applied to every error body,
the end-anchored Bedrock ARN matchers used to validate configured model ARNs, and
the image helpers backing the Images routes (decompression-bomb cap and
OpenAI-alpha-mask to Bedrock-black/white conversion).

Ref: stdapi/utils.py
"""

from __future__ import annotations

import warnings
from io import BytesIO
from json import loads

import pytest
from PIL import Image
from pybase64 import b64decode as pybase64_b64decode
from pybase64 import b64encode

from stdapi import utils
from stdapi.utils import (
    alpha_mask_to_bw,
    get_base64_image_size,
    hide_security_details,
    match_bedrock_app_profile_arn,
    match_bedrock_prompt_router_arn,
    strip_url_query,
)

pytestmark = pytest.mark.local


def test_hide_security_details_masks_auth_statuses() -> None:
    """401 and 403 responses return fixed, detail-free messages.

    The original text is discarded entirely, so a rejected credential or a
    blocked SSRF target cannot be inferred from the response body.

    Ref: stdapi/utils.py:hide_security_details
    """
    assert hide_security_details(401, "invalid key sk-secret") == "Unauthorized"
    assert hide_security_details(403, "forbidden host 10.0.0.1") == "Forbidden"


def test_hide_security_details_redacts_arn() -> None:
    """ARNs are stripped from client-facing error messages on any status.

    Bedrock echoes the full inference-profile ARN in its ``ValidationException``
    text, which would disclose the gateway's own account ID to callers.
    """
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
    """A presigned URL query is replaced by an explicit redaction marker.

    Presigned S3 URLs appear in logs and ``InputFile`` reprs; the query carries
    the signature and access key ID, so it is replaced wholesale rather than
    filtered parameter by parameter.

    Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html
    """
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
    """A well-formed inference-profile ARN matches and exposes its region.

    The ``region`` group is what routes the request, so the match must span the
    whole ARN and capture the region rather than merely finding a prefix.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-use.html
         stdapi/utils.py:match_bedrock_app_profile_arn
    """
    arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123"
    match = match_bedrock_app_profile_arn(arn)
    assert match is not None
    assert match.group(0) == arn
    assert match.group("region") == "us-east-1"


def test_arn_matcher_rejects_trailing_data() -> None:
    """An ARN followed by extra data is rejected (end-anchored).

    A non-anchored matcher would accept an attacker-supplied suffix appended to a
    legitimate ARN and forward the whole string to Bedrock as a model ID.
    """
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
    """An image exceeding the pixel limit raises a ValueError rather than allocating.

    Pillow's ``DecompressionBombError`` is translated into a ``ValueError`` with a
    fixed message so the route layer reports a 400 instead of a 500; the chained
    cause proves the rejection came from the pixel guard and not from a decode
    failure.

    Ref: stdapi/utils.py:get_base64_image_size
    """
    buffer = BytesIO()
    Image.new("RGB", (100, 100)).save(buffer, format="PNG")
    encoded = b64encode(buffer.getvalue()).decode()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="pixel size") as excinfo:
        await get_base64_image_size(encoded)
    assert str(excinfo.value) == "Image exceeds the maximum allowed pixel size."
    assert isinstance(excinfo.value.__cause__, Image.DecompressionBombError)


def test_pillow_threshold_is_half_the_documented_pixel_cap() -> None:
    """Pillow's own threshold is set to half the documented cap.

    Pillow only raises DecompressionBombError above 2x its threshold, so
    halving it makes the documented `_MAX_IMAGE_PIXELS` the real hard limit.

    Ref: stdapi/utils.py:_MAX_IMAGE_PIXELS
    """
    assert Image.MAX_IMAGE_PIXELS == utils._MAX_IMAGE_PIXELS // 2  # noqa: SLF001


async def test_get_base64_image_size_rejects_image_just_above_the_documented_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An image just above the documented cap is rejected (the real hard limit).

    Pillow raises only above ``2 * MAX_IMAGE_PIXELS``, so with the threshold set
    to cap/2 an image of 110 px must fail against a cap of 100 -- this is the
    boundary that makes the documented cap enforceable.

    Ref: stdapi/utils.py:_MAX_IMAGE_PIXELS
    """
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 50)  # cap/2, cap == 100
    buffer = BytesIO()
    Image.new("RGB", (11, 10)).save(buffer, format="PNG")  # 110 px > 100 cap
    encoded = b64encode(buffer.getvalue()).decode()
    with pytest.raises(ValueError, match="pixel size") as excinfo:
        await get_base64_image_size(encoded)
    assert str(excinfo.value) == "Image exceeds the maximum allowed pixel size."
    assert isinstance(excinfo.value.__cause__, Image.DecompressionBombError)


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
    """An HTTP InputFile never exposes its presigned signature via repr.

    ``InputFile`` instances end up in log payloads and exception messages, so the
    repr goes through ``strip_url_query`` rather than printing the source URL.

    Ref: stdapi/input_file.py:InputFile.__repr__
    """
    from stdapi.input_file import InputFile  # noqa: PLC0415

    url = (
        "https://bucket.s3.amazonaws.com/key.png"
        "?X-Amz-Credential=AKIAEXAMPLE&X-Amz-Signature=deadbeef"
    )
    text = repr(InputFile(url))
    assert "X-Amz-Signature" not in text
    assert "AKIAEXAMPLE" not in text
    assert text == "https://bucket.s3.amazonaws.com/key.png?<redacted>"


def _b64_png(image: Image.Image) -> str:
    """Encode a Pillow image as a base64 PNG string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return b64encode(buffer.getvalue()).decode()


async def test_alpha_mask_to_bw_passes_through_masks_without_alpha() -> None:
    """A mask with no alpha channel (already black/white RGB) is returned unchanged.

    Callers may pass a Bedrock-style mask directly; re-encoding it would be
    lossy work for no gain.

    Ref: https://stdapi.ai/api_openai_images_edits/
         stdapi/utils.py:alpha_mask_to_bw
    """
    source = _b64_png(Image.new("RGB", (4, 4), (0, 0, 0)))
    assert await alpha_mask_to_bw(source) == source


async def test_alpha_mask_to_bw_passes_through_undecodable_content() -> None:
    """Non-image content is returned unchanged so downstream validation rejects it.

    The converter must not turn a malformed upload into its own opaque failure:
    the request-level image validation produces the client-facing error.
    """
    source = b64encode(b"not a valid image").decode()
    assert await alpha_mask_to_bw(source) == source


async def test_alpha_mask_to_bw_converts_palette_masks_with_trns() -> None:
    """A palette ("P") mask with a ``tRNS`` transparency chunk is still converted.

    PNG optimizers (pngquant) and "PNG-8" exports routinely turn an RGBA mask into
    a palette image carrying transparency via ``tRNS`` instead of an alpha channel
    proper; that shape must still be recognised, not passed through untouched.

    Ref: https://www.w3.org/TR/png/#11tRNS
         stdapi/utils.py:_alpha_mask_to_bw
    """
    rgba = Image.new("RGBA", (2, 2), (0, 0, 0, 255))
    rgba.putpixel((0, 0), (0, 0, 0, 0))  # fully transparent -> edit region
    palette = rgba.convert("P")  # Pillow derives a tRNS chunk from the alpha channel
    buffer = BytesIO()
    palette.save(buffer, format="PNG")
    source = b64encode(buffer.getvalue()).decode()

    result = await alpha_mask_to_bw(source)

    with BytesIO(pybase64_b64decode(result)) as out, Image.open(out) as converted:
        assert converted.mode == "RGB"
        assert converted.getpixel((0, 0)) == (0, 0, 0)
        assert converted.getpixel((1, 1)) == (255, 255, 255)


async def _mask_pixel(
    alpha: int, *, invert: bool = False, threshold: int | None = None
) -> tuple[int, ...]:
    """Convert a one-pixel RGBA mask and return the resulting RGB pixel.

    The source colour under the alpha is deliberately non-grey so a pass-through
    bug cannot be mistaken for a correct conversion.
    """
    image = Image.new("RGBA", (1, 1), (10, 20, 30, alpha))
    kwargs = {} if threshold is None else {"threshold": threshold}
    result = await alpha_mask_to_bw(_b64_png(image), invert=invert, **kwargs)

    with BytesIO(pybase64_b64decode(result)) as buffer, Image.open(buffer) as converted:
        assert converted.mode == "RGB", "the alpha channel must be dropped"
        pixel = converted.getpixel((0, 0))
    assert isinstance(pixel, tuple)
    return pixel


@pytest.mark.parametrize(
    ("alpha", "invert", "threshold", "expected"),
    [
        pytest.param(0, False, None, (0, 0, 0), id="transparent-inpaint-black"),
        pytest.param(255, False, None, (255, 255, 255), id="opaque-inpaint-white"),
        pytest.param(127, False, 127, (255, 255, 255), id="alpha-at-threshold-white"),
        pytest.param(0, True, None, (255, 255, 255), id="transparent-outpaint-white"),
        pytest.param(255, True, None, (0, 0, 0), id="opaque-outpaint-black"),
    ],
)
async def test_alpha_mask_to_bw_maps_alpha_to_black_and_white(
    alpha: int, invert: bool, threshold: int | None, expected: tuple[int, ...]
) -> None:
    """Alpha decides the output colour, and ``invert`` flips the mask polarity.

    OpenAI marks the region to regenerate with transparency; Titan/Nova inpainting
    expects that region black, while Nova Canvas outpainting uses the opposite
    polarity. The edit region is ``alpha < threshold``, so a pixel sitting exactly at
    the threshold falls on the preserve (white) side.

    Ref: https://stdapi.ai/api_openai_images_edits/
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
         https://docs.aws.amazon.com/nova/latest/userguide/image-gen-access.html
    """
    assert await _mask_pixel(alpha, invert=invert, threshold=threshold) == expected


class TestJsonResponseRendering:
    """The gateway renders every response body with pydantic_core, not the stdlib.

    ``JSONResponse`` is installed as the app's ``default_response_class``, so
    its output is the wire format of every route: compact separators and raw
    UTF-8, byte-identical to what the FastAPI default would have produced.

    Ref: stdapi/utils.py:JSONResponse.render
         stdapi/main.py:app
    """

    def test_content_renders_compactly_without_ascii_escaping(self) -> None:
        """A payload is rendered with no spaces between items and non-ASCII left raw."""
        body = utils.JSONResponse(content={"model": "nova", "text": "héllo"}).body
        assert body == '{"model":"nova","text":"héllo"}'.encode()
        assert loads(bytes(body).decode()) == {"model": "nova", "text": "héllo"}

    def test_a_lone_surrogate_is_escaped_instead_of_failing_the_response(self) -> None:
        """Text pydantic_core refuses still renders, through the escaping fallback.

        A lone surrogate has no UTF-8 encoding, so both encoders raise unless the
        fallback escapes it. Without the escape the render itself fails, turning
        a served answer into an unhandled error after the status was decided.

        Ref: https://docs.python.org/3/library/json.html#json.dumps
        """
        body = utils.JSONResponse(content={"text": "a\ud800b"}).body

        assert body == b'{"text":"a\\ud800b"}'
        assert loads(bytes(body).decode())["text"] == "a\ud800b"
