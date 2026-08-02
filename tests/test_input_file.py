"""Unit tests for input file handling (:mod:`stdapi.input_file`).

``InputFile`` is the single ingestion point for every file-shaped request field:
it detects the origin (raw base64, data URI, HTTPS URL, ``s3://`` URI,
``file-id:`` reference or an uploaded multipart part) and enforces the size and
allowlist limits before any AWS call.

Ref: https://stdapi.ai/api_openai_files/
     stdapi/input_file.py:InputFile
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import pytest
from pybase64 import b64encode

from stdapi import input_file
from stdapi.api_errors import ApiError
from stdapi.aws_s3 import BUCKET_TO_REGION
from stdapi.config import SETTINGS
from stdapi.files._multipart import create_multipart_session
from stdapi.input_file import InputFile

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.local


@pytest.mark.parametrize(
    ("accessor", "payload"),
    [
        pytest.param("to_bytes", b64encode(b"x" * 64).decode(), id="to_bytes"),
        pytest.param("to_base64", b64encode(b"x" * 64).decode(), id="to_base64"),
        pytest.param(
            "to_data_uri",
            f"data:image/png;base64,{b64encode(b'x' * 64).decode()}",
            id="to_data_uri",
        ),
    ],
)
async def test_accessors_reject_oversized_inline_input(
    monkeypatch: pytest.MonkeyPatch, accessor: str, payload: str
) -> None:
    """Every inline accessor rejects an over-limit payload with the same HTTP 413.

    The size is checked against the decoded payload before it is read, so the request
    is refused without buffering the whole body — and that check lives below the three
    accessors rather than in any one of them.

    Ref: stdapi/input_file.py:InputFile.to_bytes
         stdapi/config.py:_Settings.max_input_file_size
    """
    monkeypatch.setattr(SETTINGS, "max_input_file_size", 8)
    with pytest.raises(ApiError, match="8 bytes") as exc:
        await getattr(InputFile(payload), accessor)()
    assert exc.value.status == 413


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


async def test_invalid_base64_input_returns_a_fixed_message() -> None:
    """Malformed base64 input is rejected with a fixed message, not the raw decoder error.

    The ``binascii``/``pybase64`` error text names internal decoding details; per
    AGENTS.md ("Never leak internals") the caller gets a fixed message instead.

    Ref: stdapi/input_file.py:_Base64Source._read
    """
    with pytest.raises(ApiError, match=r"^Invalid base64 data\.$") as exc:
        await InputFile("not-valid-base64!!!").to_bytes()
    assert exc.value.status == 400


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


def test_s3_uri_accepts_bucket_declared_only_as_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An accepted external bucket resolves to the region the operator declared for it.

    ``AWS_S3_ACCEPTED_BUCKETS`` maps bucket names the gateway may read but does not
    own to their region.  Those buckets are absent from the app-owned bucket→region
    map, so the source must fall back to the declared region instead of failing the
    lookup.

    Ref: stdapi/config.py:_Settings.aws_s3_accepted_buckets
         stdapi/input_file.py:_S3Source.__init__
         stdapi/aws_s3.py:BUCKET_TO_REGION
    """
    bucket = "an-accepted-external-bucket-xyz"
    assert bucket not in BUCKET_TO_REGION, "the bucket must not be app-owned"
    monkeypatch.setattr(
        input_file, "_ACCEPTED_BUCKETS", frozenset({bucket}), raising=True
    )
    monkeypatch.setattr(
        input_file, "_ACCEPTED_BUCKET_REGIONS", {bucket: "eu-west-3"}, raising=True
    )

    file = InputFile(f"s3://{bucket}/key.png")

    assert file.is_s3 is True
    assert file.region == "eu-west-3", (
        "the declared region must route the S3 calls, not the default region"
    )


async def test_unsupported_document_type_is_a_caller_error() -> None:
    """A file Bedrock has no document format for is refused with a 400, naming the type.

    Every non-image/video/audio input falls through to the document branch, so an
    unhandled type there would reach Converse and come back as a 500 instead of
    telling the caller which attachment it has to drop.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
         stdapi/input_file.py:InputFile.to_bedrock_content_block
    """
    payload = b64encode(b"x" * 32).decode()
    file = InputFile(f"data:application/x-tar;base64,{payload}")

    with pytest.raises(ApiError, match="Unsupported document type") as exc:
        await file.to_bedrock_content_block()

    assert exc.value.status == 400
    assert "application/x-tar" in str(exc.value), (
        "the caller needs to know which attachment was refused"
    )


class _StubHttpResponse:
    """Minimal aiohttp response stand-in serving a fixed body and headers."""

    def __init__(self, body: bytes, content_length: int | None = None) -> None:
        self.body = body
        self.headers = {
            "Content-Type": "application/pdf",
            "Content-Length": str(
                len(body) if content_length is None else content_length
            ),
        }
        self.content = self

    async def __aenter__(self) -> Self:
        """Enter the response context."""
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Leave the response context."""

    def raise_for_status(self) -> None:
        """Accept the stubbed 200 response."""

    async def read(self) -> bytes:
        """Return the whole stubbed body."""
        return self.body

    async def iter_chunked(self, size: int) -> AsyncIterator[bytes]:
        """Yield the stubbed body in *size*-byte chunks."""
        for start in range(0, len(self.body), size):
            yield self.body[start : start + size]


class _StubHttpSession:
    """Minimal aiohttp session stand-in recording the requests it served."""

    def __init__(self, response: _StubHttpResponse) -> None:
        self.response = response
        self.requests: list[str] = []

    async def __aenter__(self) -> Self:
        """Enter the session context."""
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Leave the session context."""

    def head(self, url: str) -> _StubHttpResponse:
        """Serve the stubbed response to a ``HEAD``."""
        self.requests.append(f"HEAD {url}")
        return self.response

    def get(self, url: str) -> _StubHttpResponse:
        """Serve the stubbed response to a ``GET``."""
        self.requests.append(f"GET {url}")
        return self.response


def _patch_http(
    monkeypatch: pytest.MonkeyPatch, response: _StubHttpResponse
) -> _StubHttpSession:
    """Serve *response* to every request the HTTPS input source makes.

    Returns:
        The stub session, for asserting on the requests it served.
    """
    session = _StubHttpSession(response)
    monkeypatch.setattr(
        input_file._HttpSource,  # noqa: SLF001
        "_client_session",
        lambda _self, _extra_headers=None: session,
    )
    return session


class TestHttpsSourceDownload:
    """An ``https://`` input is fetched server-side, under the configured size cap.

    ``max_input_file_size`` is the memory-exhaustion guard for remote inputs. It
    cannot be enforced from the ``Content-Length`` alone: that header is supplied
    by the origin the *caller* chose, so the body itself has to be metered while it
    streams in.

    Ref: stdapi/input_file.py:_HttpSource._read
         stdapi/input_file.py:_HttpSource._read_capped
         stdapi/config.py:_Settings.max_input_file_size
    """

    async def test_body_is_downloaded_and_returned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no size cap the body is read in one shot and returned verbatim."""
        monkeypatch.setattr(SETTINGS, "max_input_file_size", 0)
        session = _patch_http(monkeypatch, _StubHttpResponse(b"%PDF-1.7 body"))

        assert await InputFile("https://example.com/doc.pdf").to_bytes() == (
            b"%PDF-1.7 body"
        )

        assert session.requests == ["GET https://example.com/doc.pdf"], (
            "an unlimited read must not pay for a metadata HEAD"
        )

    async def test_oversized_body_is_rejected_despite_an_honest_looking_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A body larger than the cap is refused even when Content-Length understates it.

        The origin is attacker-controlled, so the declared length only decides
        whether the download starts; the streamed body is what the cap is applied
        to.
        """
        monkeypatch.setattr(SETTINGS, "max_input_file_size", 1024)
        session = _patch_http(
            monkeypatch, _StubHttpResponse(b"x" * 4096, content_length=1)
        )

        with pytest.raises(ApiError, match="1024 bytes") as exc:
            await InputFile("https://example.com/big.pdf").to_bytes()

        assert exc.value.status == 413
        assert session.requests[0].startswith("HEAD "), (
            "the declared size is probed before the body is pulled"
        )

    async def test_empty_body_is_a_download_error_not_an_empty_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 200 with no body is reported as a download failure, with the query redacted.

        An empty input would otherwise travel on and fail deep inside a model call,
        and the URL may carry a pre-signed token that must not reach the message.
        """
        monkeypatch.setattr(SETTINGS, "max_input_file_size", 0)
        _patch_http(monkeypatch, _StubHttpResponse(b""))

        with pytest.raises(ApiError, match="Empty body") as exc:
            await InputFile("https://example.com/none.pdf?sig=secret").to_bytes()

        assert "secret" not in str(exc.value)


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


async def test_create_multipart_session_rejects_overlong_filename() -> None:
    """A filename longer than 500 characters is rejected before any S3 call.

    The filename is interpolated into the object's ``Content-Disposition`` header, and
    500 is the Anthropic Files API cap the gateway mirrors.

    Ref: stdapi/files/_core.py:_validate_filename
         stdapi/files/_multipart.py:create_multipart_session
    """
    with pytest.raises(ApiError, match="maximum length of 500") as exc:
        await create_multipart_session("a" * 501, "text/plain", "", 10)
    assert exc.value.status == 400


async def test_filename_length_check_runs_before_the_character_check() -> None:
    """At exactly 500 characters the length branch passes and the character check runs.

    The length branch is evaluated first, so it would mask the character rejection for
    any name at or above the cap; a 500-character name carrying a forbidden character
    must still report the character failure.

    Ref: stdapi/files/_core.py:_validate_filename
    """
    filename = 'a"' + "a" * 498
    assert len(filename) == 500
    with pytest.raises(ApiError, match="forbidden characters") as exc:
        await create_multipart_session(filename, "text/plain", "", 10)
    assert exc.value.status == 400
