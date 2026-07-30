"""Unit tests for input file handling (:mod:`stdapi.input_file`).

``InputFile`` is the single ingestion point for every file-shaped request field:
it detects the origin (raw base64, data URI, HTTPS URL, ``s3://`` URI,
``file-id:`` reference or an uploaded multipart part) and enforces the size and
allowlist limits before any AWS call.

Ref: https://stdapi.ai/api_openai_files/
     stdapi/input_file.py:InputFile
"""

from __future__ import annotations

import pytest
from pybase64 import b64encode

from stdapi.api_errors import ApiError
from stdapi.aws_s3 import BUCKET_TO_REGION
from stdapi.config import SETTINGS
from stdapi.files._multipart import create_multipart_session
from stdapi.input_file import InputFile


async def test_to_bytes_rejects_oversized_inline_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inline base64 input larger than the limit is rejected with HTTP 413.

    The size is checked against the decoded payload before it is read, so the
    request is refused without buffering the whole body.

    Ref: stdapi/input_file.py:InputFile.to_bytes
         stdapi/config.py:_Settings.max_input_file_size
    """
    monkeypatch.setattr(SETTINGS, "max_input_file_size", 8)
    payload = b64encode(b"x" * 64).decode()
    with pytest.raises(ApiError) as exc:
        await InputFile(payload).to_bytes()
    assert exc.value.status == 413
    assert "8 bytes" in str(exc.value), exc.value.args


async def test_to_bytes_allows_input_within_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inline input within the configured limit is returned decoded and unchanged.

    Ref: stdapi/input_file.py:InputFile.to_bytes
    """
    monkeypatch.setattr(SETTINGS, "max_input_file_size", 1024)
    data = b"x" * 64
    assert await InputFile(b64encode(data).decode()).to_bytes() == data


async def test_to_bytes_unlimited_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the limit disabled (0), inputs of any size are accepted.

    ``0`` is falsy on purpose: it short-circuits the size resolution entirely
    rather than comparing against a zero maximum.

    Ref: stdapi/input_file.py:InputFile.to_bytes
         stdapi/config.py:_Settings.max_input_file_size
    """
    monkeypatch.setattr(SETTINGS, "max_input_file_size", 0)
    data = b"y" * 4096
    assert await InputFile(b64encode(data).decode()).to_bytes() == data


def test_max_concurrent_input_downloads_default() -> None:
    """The per-request input-download concurrency limit defaults to 8.

    Ref: stdapi/config.py:_Settings.max_concurrent_input_downloads
    """
    assert SETTINGS.max_concurrent_input_downloads == 8


def test_s3_uri_rejects_unlisted_bucket() -> None:
    """An ``s3://`` URI pointing outside the bucket allowlist is rejected.

    The allowlist is the SSRF guard for S3 inputs: without it any caller could
    make the gateway read an arbitrary bucket with the task role's credentials.

    Ref: stdapi/input_file.py:InputFile._normalize_and_detect_origin
    """
    with pytest.raises(ValueError, match="not allowed") as exc:
        InputFile("s3://an-unconfigured-external-bucket-xyz/key.png")
    assert "an-unconfigured-external-bucket-xyz" in str(exc.value), (
        "the rejection must name the refused bucket"
    )


def test_s3_uri_accepts_configured_bucket() -> None:
    """An ``s3://`` URI for a configured bucket resolves to that bucket's region.

    The region is bound at construction time from the bucket→region map, which is
    what later S3 calls are routed with.

    Ref: stdapi/input_file.py:InputFile._normalize_and_detect_origin
         stdapi/aws_s3.py:BUCKET_TO_REGION
    """
    bucket = next(iter(BUCKET_TO_REGION), None)
    if bucket is None:
        pytest.skip("No S3 bucket configured in this environment")
    input_file = InputFile(f"s3://{bucket}/key.png")
    assert input_file.is_s3 is True
    assert input_file.region == BUCKET_TO_REGION[bucket]


async def test_create_multipart_session_rejects_unsafe_filename() -> None:
    """A filename with header-injection characters is rejected before any S3 call.

    The filename ends up in the object's ``Content-Disposition`` header, so the
    forbidden-character and length checks run ahead of the bucket lookup — which
    is why this test needs no AWS access.

    Ref: stdapi/files/_core.py:_validate_filename
         stdapi/files/_multipart.py:create_multipart_session
    """
    with pytest.raises(ApiError) as exc:
        await create_multipart_session('bad"name.txt', "text/plain", "", 10)
    assert exc.value.status == 400
    assert "forbidden characters" in str(exc.value), exc.value.args


async def test_to_base64_rejects_oversized_inline_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversized inline base64 input is rejected via the to_base64 path (413).

    Ref: stdapi/input_file.py:InputFile.to_base64
    """
    monkeypatch.setattr(SETTINGS, "max_input_file_size", 8)
    payload = b64encode(b"x" * 64).decode()
    with pytest.raises(ApiError) as exc:
        await InputFile(payload).to_base64()
    assert exc.value.status == 413
    assert "8 bytes" in str(exc.value), exc.value.args


async def test_to_data_uri_rejects_oversized_inline_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversized inline data URI is rejected via the to_data_uri path (413).

    Ref: stdapi/input_file.py:InputFile.to_data_uri
    """
    monkeypatch.setattr(SETTINGS, "max_input_file_size", 8)
    data_uri = f"data:image/png;base64,{b64encode(b'x' * 64).decode()}"
    with pytest.raises(ApiError) as exc:
        await InputFile(data_uri).to_data_uri()
    assert exc.value.status == 413
    assert "8 bytes" in str(exc.value), exc.value.args
