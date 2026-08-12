"""Unit tests for input file handling (:mod:`stdapi.input_file`).

``InputFile`` is the single ingestion point for every file-shaped request field:
it detects the origin (raw base64, data URI, HTTPS URL, ``s3://`` URI,
``file-id:`` reference or an uploaded multipart part) and enforces the size and
allowlist limits before any AWS call.

Ref: https://stdapi.ai/api_openai_files/
     stdapi/input_file.py:InputFile
"""

from __future__ import annotations

import re
from asyncio import gather, sleep
from typing import TYPE_CHECKING, Self

import pytest
from pybase64 import b64encode

from stdapi import input_file
from stdapi.api_errors import ApiError
from stdapi.aws_s3 import BUCKET_TO_REGION, UPLOAD_CHUNK_SIZE, S3Object
from stdapi.config import SETTINGS
from stdapi.files._multipart import create_multipart_session
from stdapi.input_file import (
    InlineMediaLimits,
    InputFile,
    inline_media_storage_error,
    pin_bedrock_upload_region,
    plan_bedrock_media_transport,
    resolve_all_bedrock_content_blocks,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

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
        self._offset = 0
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

    async def read(self, size: int = -1) -> bytes:
        """Return up to *size* bytes, advancing the cursor as a stream reader does.

        ``read_chunks`` pulls a fixed size at a time and stops on an empty
        result, so the stub has to consume rather than replay — otherwise a
        streamed read never terminates.
        """
        if size < 0:
            chunk, self._offset = self.body[self._offset :], len(self.body)
            return chunk
        chunk = self.body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

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

    async def test_oversized_body_is_rejected_while_being_staged_to_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Staging to S3 meters the streamed body, not the declared length.

        The cap has to hold on both routes an attachment can take. Staging pulls
        the body a second time, into the operator's bucket rather than into
        memory, so trusting ``Content-Length`` there would let a lying origin
        write past the limit at the operator's expense.
        """
        monkeypatch.setattr(SETTINGS, "max_input_file_size", 1024)
        _patch_http(monkeypatch, _StubHttpResponse(b"x" * 4096, content_length=1))
        staged = bytearray()

        async def _consume(
            body: AsyncIterator[bytes], *_args: object, **_kw: object
        ) -> None:
            async for chunk in body:
                staged.extend(chunk)

        monkeypatch.setattr(input_file, "put_s3_object", _consume)

        with pytest.raises(ApiError, match="1024 bytes") as exc:
            await InputFile("https://example.com/big.pdf").to_s3("us-east-1")

        assert exc.value.status == 413
        assert len(staged) <= 1024 + UPLOAD_CHUNK_SIZE, (
            "the upload must abort at the cap, not after the whole body is sent"
        )


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


@pytest.fixture
def input_files() -> Iterator[None]:
    """Bind a fresh per-request input-file registry for the duration of the test."""
    token = input_file._CURRENT_INPUT_FILES.set([])  # noqa: SLF001
    try:
        yield
    finally:
        input_file._CURRENT_INPUT_FILES.reset(token)  # noqa: SLF001


def _data_uri(base64_length: int, media_type: str = "image/png") -> str:
    """Return a data URI whose base64 payload is exactly *base64_length* long.

    Returns:
        A ``data:`` URI string.
    """
    assert base64_length % 4 == 0, "a base64 payload is a whole number of quads"
    return f"data:{media_type};base64,{'A' * base64_length}"


async def _fake_upload(*_args: object, **_kwargs: object) -> S3Object:
    """Stand in for an S3 upload without touching AWS.

    Returns:
        A fixed object reference.
    """
    return S3Object(bucket="a-bucket", key="uploaded")


async def _stub_unknown_size(source: input_file._FileSource) -> None:
    """Resolve a remote source whose origin declared no content length."""
    source._content_type = "application/pdf"  # noqa: SLF001
    source._filename = None  # noqa: SLF001
    source._size = 0  # noqa: SLF001


def _remote_file(declared_size: int, content_type: str = "image/png") -> InputFile:
    """Return a remote attachment whose origin declares *declared_size* bytes.

    The size a remote origin declares is what the transport decision reads, so a
    stub of it is enough to exercise any size band without moving the bytes.

    Returns:
        An ``InputFile`` over an HTTPS URL.
    """
    file = InputFile("https://example.com/attachment", content_type=content_type)
    source = file._source  # noqa: SLF001
    source._content_type = content_type  # noqa: SLF001
    source._filename = None  # noqa: SLF001
    source._size = declared_size  # noqa: SLF001
    return file


#: Stands in for a model that reads an image from storage rather than from the request.
_STORED_IMAGES: frozenset[input_file.BedrockMediaType] = frozenset({"image"})


def _allowed_bucket(monkeypatch: pytest.MonkeyPatch) -> str:
    """Allow an ``s3://`` source without depending on a configured deployment.

    Returns:
        The name of the bucket the gateway now accepts as an input source.
    """
    bucket = "a-stored-attachment-bucket"
    monkeypatch.setattr(
        input_file, "_ACCEPTED_BUCKETS", frozenset({bucket}), raising=True
    )
    monkeypatch.setattr(
        input_file, "_ACCEPTED_BUCKET_REGIONS", {bucket: "us-east-1"}, raising=True
    )
    return bucket


#: Backend names a caller-facing message must never contain.
_INTERNAL_WORDS: frozenset[str] = frozenset(
    {"s3", "bucket", "bedrock", "aws", "amazon", "setting"}
)

#: Shapes an internal name takes, none of which reads as a prose word.
_INTERNAL_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("a setting name", re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")),
    ("a module path", re.compile(r"[\w.-]*(?:/[\w.-]+|\.py\b)")),
    ("an internal identifier", re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")),
)


def _assert_names_no_internals(message: str) -> None:
    """Assert a caller-facing refusal describes the API rather than the machinery.

    What leaks is never a lowercase prose word: a setting name is uppercase with
    underscores, a module path carries a separator or a ``.py`` suffix, and an
    internal attribute is snake_case — so each shape is matched as itself, and
    the backend names are read from a tokenization that also splits them out of
    a compound identifier.
    """
    for shape, pattern in _INTERNAL_SHAPES:
        leaked = pattern.search(message)
        assert leaked is None, (
            f"the message names {shape} ({leaked.group()!r}): {message}"
        )
    named = _INTERNAL_WORDS & set(re.findall(r"[a-z0-9]+", message.lower()))
    assert not named, (
        f"the message names the machinery ({sorted(named)}), not the API: {message}"
    )


@pytest.mark.parametrize(
    "leak",
    [
        "Attachments of this size need AWS_S3_BUCKET to be set.",
        "Staging failed in stdapi/input_file.py.",
        "The attachment exceeds max_input_file_size.",
        "The upload to the Amazon S3 bucket failed.",
    ],
    ids=["setting", "module-path", "identifier", "backend-name"],
)
def test_the_internals_guard_catches_a_leaked_name(leak: str) -> None:
    """The guard behind the refusal tests fails on the leaks it exists to catch.

    It is the only check that the 413 messages name no setting and no internal,
    so a guard matching lowercase prose words alone would pass every shape a
    leak actually takes and assert nothing.

    Ref: stdapi/input_file.py:_too_large_error
         stdapi/input_file.py:inline_media_storage_error
    """
    with pytest.raises(AssertionError):
        _assert_names_no_internals(leak)


async def _stub_s3_read(_source: input_file._FileSource) -> bytes:
    """Serve a stored object's content without an S3 call.

    Returns:
        Fixed object content.
    """
    return b"PNGDATA"


@pytest.mark.usefixtures("input_files")
class TestInlineMediaTransport:
    """Attachments too large to travel inline are handed to the model by reference.

    The gateway accepts an attachment far larger than a model reads inline, so the
    transport is chosen per request from the size the caller sent: small payloads
    stay embedded in the request, oversized ones are staged and referenced. The
    size that decides is the base64 one, because that is the form the payload
    travels in and the form the model's limit is expressed against.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageSource.html
         stdapi/input_file.py:plan_bedrock_media_transport
    """

    async def test_payload_at_the_limit_stays_inline(self) -> None:
        """A payload exactly at the per-file limit is still sent inline."""
        file = InputFile(_data_uri(40))
        block = await file.to_bedrock_content_block()

        assert not await plan_bedrock_media_transport(
            InlineMediaLimits(max_file_base64_size=40, max_total_base64_size=40)
        )

        await file.resolve_bedrock_content_block("us-east-1")
        assert "bytes" in block["image"]["source"]

    async def test_a_stored_object_is_referenced_whichever_way_it_is_named(
        self,
    ) -> None:
        """Both spellings of an already-stored object skip measurement alike.

        A Files API object is the same object whether the caller names it in a
        typed ``file_id`` field or through the ``file-id:`` URI, so measuring one
        and referencing the other would download the gateway's own object only to
        embed it again.

        Ref: stdapi/input_file.py:plan_bedrock_media_transport
             stdapi/input_file.py:_ALREADY_STORED_ORIGINS
        """
        bucket = next(iter(BUCKET_TO_REGION), None)
        if bucket is None:
            pytest.skip("No S3 bucket configured in this environment")

        origins = {
            value: InputFile(value)._origin  # noqa: SLF001
            for value in (
                f"s3://{bucket}/key.png",
                "file-id:file_0123456789abcdef0123456789abcdef",
            )
        }

        assert set(origins.values()) <= input_file._ALREADY_STORED_ORIGINS, (  # noqa: SLF001
            f"both spellings name an object already in S3, but got {origins}"
        )

    async def test_payload_over_the_limit_is_sent_by_reference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A payload past the per-file limit is staged and referenced instead."""
        monkeypatch.setattr(InputFile, "to_s3", _fake_upload)
        file = InputFile(_data_uri(44))
        block = await file.to_bedrock_content_block()

        assert await plan_bedrock_media_transport(
            InlineMediaLimits(max_file_base64_size=40, max_total_base64_size=40),
            s3_location_media_types=_STORED_IMAGES,
        )

        await file.resolve_bedrock_content_block(
            "us-east-1", s3_location_media_types=_STORED_IMAGES
        )
        assert block["image"]["source"] == {
            "s3Location": {"uri": "s3://a-bucket/uploaded"}
        }

    async def test_the_limit_is_measured_on_the_base64_length(self) -> None:
        """The decision reads the encoded length, not the decoded one.

        A 40-character base64 payload decodes to 30 bytes: a limit of 32 accepts it
        when the wrong quantity is measured and refuses it when the right one is.
        """
        file = InputFile(_data_uri(40))
        assert await file.get_base64_size() == 40
        await file.to_bedrock_content_block()

        assert await plan_bedrock_media_transport(
            InlineMediaLimits(max_file_base64_size=32, max_total_base64_size=32),
            s3_location_media_types=_STORED_IMAGES,
        )

    async def test_the_decision_never_decodes_the_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Choosing a transport must not cost a decode of the attachment.

        The base64 length is read from the string the caller sent, so a request
        carrying tens of megabytes of attachments pays no decode to be routed.
        """

        async def _forbidden(*_args: object, **_kwargs: object) -> bytes:
            """Fail the test if the payload is decoded."""
            pytest.fail("the transport decision decoded the payload")

        # A bare base64 payload is the one whose size the source can only
        # otherwise learn by decoding a prefix of it for content sniffing.
        bare = InputFile("A" * 40, content_type="image/png")
        data_uri = InputFile(_data_uri(40))
        await bare.to_bedrock_content_block(content_type="image/png")
        await data_uri.to_bedrock_content_block()
        monkeypatch.setattr(input_file, "b64decode", _forbidden)

        assert await bare.get_base64_size() == 40
        assert not await plan_bedrock_media_transport(
            InlineMediaLimits(max_file_base64_size=400, max_total_base64_size=400)
        )

    async def test_an_unmeasurable_payload_stays_inline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A remote input whose size the origin does not declare is left inline.

        A ``Content-Length``-less origin reports a size of zero. Routing it by
        reference on a size nobody knows would download and re-upload every such
        input; inline is what the request does today, and the model reports the
        real limit if it is genuinely too large.
        """
        monkeypatch.setattr(
            input_file._HttpSource,  # noqa: SLF001
            "_resolve_metadata",
            _stub_unknown_size,
        )
        file = InputFile("https://example.com/doc.pdf")
        await file.to_bedrock_content_block(content_type="application/pdf")

        assert not await plan_bedrock_media_transport(
            InlineMediaLimits(max_file_base64_size=4, max_total_base64_size=4)
        )

    async def test_the_largest_attachment_moves_first_when_the_request_is_too_big(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every file fits but the request does not, the biggest one moves.

        Each attachment is small enough to travel inline, so only their total is
        over the limit: moving the largest one is what brings the request back
        under it while touching the fewest payloads.
        """
        monkeypatch.setattr(InputFile, "to_s3", _fake_upload)
        small = InputFile(_data_uri(40))
        large = InputFile(_data_uri(80))
        small_block = await small.to_bedrock_content_block()
        large_block = await large.to_bedrock_content_block()

        assert await plan_bedrock_media_transport(
            InlineMediaLimits(max_file_base64_size=100, max_total_base64_size=100),
            s3_location_media_types=_STORED_IMAGES,
        )

        await small.resolve_bedrock_content_block(
            "us-east-1", s3_location_media_types=_STORED_IMAGES
        )
        await large.resolve_bedrock_content_block(
            "us-east-1", s3_location_media_types=_STORED_IMAGES
        )
        assert "bytes" in small_block["image"]["source"]
        assert "s3Location" in large_block["image"]["source"]

    async def test_an_explicit_transport_overrides_the_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller that asked for inline content gets it, whatever the policy decided."""
        monkeypatch.setattr(InputFile, "to_s3", _fake_upload)
        file = InputFile(_data_uri(44))
        block = await file.to_bedrock_content_block()
        await plan_bedrock_media_transport(
            InlineMediaLimits(max_file_base64_size=40, max_total_base64_size=40),
            s3_location_media_types=_STORED_IMAGES,
        )

        await file.resolve_bedrock_content_block("us-east-1", to_s3=False)
        assert "bytes" in block["image"]["source"]

    async def test_an_oversized_attachment_is_refused_for_a_model_that_reads_no_reference(
        self,
    ) -> None:
        """A model that only reads inline attachments refuses the oversized one itself.

        Leaving it inline would fail at the model with a message the caller cannot
        act on, so the request is refused here, naming the size that is accepted.
        """
        file = InputFile(_data_uri(44))
        await file.to_bedrock_content_block()

        with pytest.raises(ApiError) as exc:
            await plan_bedrock_media_transport(
                InlineMediaLimits(max_file_base64_size=40, max_total_base64_size=40),
                s3_location_media_types=frozenset(),
            )

        assert exc.value.status == 413
        message = str(exc.value)
        assert "30 bytes" in message, "the caller needs the size that is accepted"
        _assert_names_no_internals(message)

    async def test_a_request_too_large_in_total_is_refused_when_nothing_can_move(
        self,
    ) -> None:
        """Attachments that each fit but together do not are refused as a whole."""
        for _ in range(2):
            file = InputFile(_data_uri(40))
            await file.to_bedrock_content_block()

        with pytest.raises(ApiError) as exc:
            await plan_bedrock_media_transport(
                InlineMediaLimits(max_file_base64_size=40, max_total_base64_size=60),
                s3_location_media_types=frozenset(),
            )

        assert exc.value.status == 413
        assert "45 bytes" in str(exc.value)

    async def test_a_stored_attachment_is_inlined_for_a_model_that_reads_no_reference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ``s3://`` image is inlined for a model that cannot read a stored reference.

        The origin is a preference, not an instruction: a model that only accepts
        inline attachments must receive the bytes rather than a reference it
        rejects.

        Ref: stdapi/input_file.py:InputFile.resolve_bedrock_content_block
        """
        monkeypatch.setattr(SETTINGS, "max_input_file_size", 0)
        monkeypatch.setattr(input_file._S3Source, "_read", _stub_s3_read)  # noqa: SLF001
        bucket = _allowed_bucket(monkeypatch)
        file = InputFile(f"s3://{bucket}/key.png", content_type="image/png")
        block = await file.to_bedrock_content_block()

        await file.resolve_bedrock_content_block(
            "us-east-1", s3_location_media_types=frozenset()
        )

        assert block["image"]["source"] == {"bytes": b"PNGDATA"}

    async def test_a_realistic_attachment_is_left_inline_by_the_default_limits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Payloads of the size clients actually send keep travelling inline.

        The defaults exist to catch what the model would refuse, not to move
        ordinary traffic through storage: a megabyte-sized attachment must not
        newly acquire an upload.
        """

        def _fail(*_args: object, **_kwargs: object) -> S3Object:
            """Fail the test if an upload is attempted."""
            pytest.fail("an ordinary attachment was routed through storage")

        monkeypatch.setattr(InputFile, "to_s3", _fail)
        blocks = [
            await InputFile(_data_uri(length)).to_bedrock_content_block()
            for length in (4, 4_000, 4_000_000, 20_000_000)
        ]

        assert not await plan_bedrock_media_transport(InlineMediaLimits())

        await resolve_all_bedrock_content_blocks("us-east-1")
        assert all("bytes" in block["image"]["source"] for block in blocks)

    async def test_a_model_that_declares_no_stored_kind_gets_no_reference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omitting the model's policy denies storage rather than assuming it.

        A caller that forgets the argument must get the transport every model
        accepts — inline bytes — not a reference most of them refuse.

        Ref: stdapi/models/__init__.py:ModelBase.S3_LOCATION_MEDIA_TYPES
        """
        monkeypatch.setattr(SETTINGS, "max_input_file_size", 0)
        monkeypatch.setattr(input_file._S3Source, "_read", _stub_s3_read)  # noqa: SLF001
        bucket = _allowed_bucket(monkeypatch)
        file = InputFile(f"s3://{bucket}/key.png", content_type="image/png")
        block = await file.to_bedrock_content_block()

        await file.resolve_bedrock_content_block("us-east-1")

        assert block["image"]["source"] == {"bytes": b"PNGDATA"}

    async def test_the_transport_survives_a_second_planning_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Planning again after the blocks are resolved still reports the upload.

        One request plans once per requested choice, and resolution consumes the
        pending blocks: a later call sees nothing left to weigh and must still
        report that the request is pinned to a region with storage.
        """
        monkeypatch.setattr(InputFile, "to_s3", _fake_upload)
        file = InputFile(_data_uri(44))
        await file.to_bedrock_content_block()
        limits = InlineMediaLimits(max_file_base64_size=40, max_total_base64_size=40)

        assert await plan_bedrock_media_transport(
            limits, s3_location_media_types=_STORED_IMAGES
        )
        await resolve_all_bedrock_content_blocks(
            "us-east-1", s3_location_media_types=_STORED_IMAGES
        )

        assert await plan_bedrock_media_transport(
            limits, s3_location_media_types=_STORED_IMAGES
        ), "the request stays pinned once its media has been staged"

    async def test_concurrent_calls_stage_the_attachment_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The calls of one request share the staged attachment instead of racing.

        ``n=2`` runs one Converse call per requested choice over the same content
        blocks: uploading twice would bill twice and leave the loser resolving a
        block whose source was already consumed.
        """
        uploads = 0

        async def _counted_upload(*_args: object, **_kwargs: object) -> S3Object:
            """Record an upload and yield to the event loop, as a real one does.

            Returns:
                A fixed object reference.
            """
            nonlocal uploads
            uploads += 1
            await sleep(0)
            return S3Object(bucket="a-bucket", key="uploaded")

        monkeypatch.setattr(InputFile, "to_s3", _counted_upload)
        file = InputFile(_data_uri(44))
        block = await file.to_bedrock_content_block()
        await plan_bedrock_media_transport(
            InlineMediaLimits(max_file_base64_size=40, max_total_base64_size=40),
            s3_location_media_types=_STORED_IMAGES,
        )

        await gather(
            *(
                resolve_all_bedrock_content_blocks(
                    "us-east-1", s3_location_media_types=_STORED_IMAGES
                )
                for _ in range(2)
            )
        )

        assert uploads == 1, "one attachment is staged once, whatever the choice count"
        assert block["image"]["source"] == {
            "s3Location": {"uri": "s3://a-bucket/uploaded"}
        }

    async def test_a_failed_staging_fails_every_call_of_the_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When staging fails, the calls sharing the block fail with the same error.

        Only one call stages the attachment; releasing the others with a content
        block whose source was never filled would send the model a request no
        client asked for and no error the caller can read.

        Ref: stdapi/input_file.py:InputFile.resolve_bedrock_content_block
        """

        async def _refused_upload(*_args: object, **_kwargs: object) -> S3Object:
            """Fail the way a denied upload does, after yielding to the loop.

            Raises:
                ApiError: Always.
            """
            await sleep(0)
            msg = "Storage is unavailable."
            raise ApiError(msg, status=503)

        monkeypatch.setattr(InputFile, "to_s3", _refused_upload)
        file = InputFile(_data_uri(44))
        block = await file.to_bedrock_content_block()
        await plan_bedrock_media_transport(
            InlineMediaLimits(max_file_base64_size=40, max_total_base64_size=40),
            s3_location_media_types=_STORED_IMAGES,
        )

        results = await gather(
            *(
                file.resolve_bedrock_content_block(
                    "us-east-1", s3_location_media_types=_STORED_IMAGES
                )
                for _ in range(2)
            ),
            return_exceptions=True,
        )

        assert all(isinstance(result, ApiError) for result in results), (
            f"every call must fail, got {results}"
        )
        assert block["image"]["source"] == {}, "no call may send an empty source"

    async def test_an_attachment_over_the_server_limit_is_refused_before_storing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``MAX_INPUT_FILE_SIZE`` still bounds an attachment that would be stored.

        Staging streams the payload straight to storage instead of reading it, so
        an attachment past the operator's limit has to be refused while planning
        — otherwise a size the deployment refuses to read for a model becomes a
        size anyone can write into its bucket.

        Ref: stdapi/input_file.py:plan_bedrock_media_transport
        """

        def _fail(*_args: object, **_kwargs: object) -> S3Object:
            """Fail the test if the oversized attachment reaches storage."""
            pytest.fail("an attachment over the server limit was stored")

        monkeypatch.setattr(SETTINGS, "max_input_file_size", 10_485_760)
        monkeypatch.setattr(InputFile, "to_s3", _fail)
        file = _remote_file(50_000_000)
        await file.to_bedrock_content_block()

        with pytest.raises(ApiError) as exc:
            await plan_bedrock_media_transport(
                InlineMediaLimits(
                    max_file_base64_size=25_000_000, max_total_base64_size=25_000_000
                ),
                s3_location_media_types=_STORED_IMAGES,
            )

        assert exc.value.status == 413
        assert "10485760" in str(exc.value)

    async def test_an_attachment_within_the_server_limit_is_still_stored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The server limit refuses only what exceeds it, not every stored attachment."""
        monkeypatch.setattr(SETTINGS, "max_input_file_size", 50_000_000)
        monkeypatch.setattr(InputFile, "to_s3", _fake_upload)
        file = _remote_file(30_000_000)
        block = await file.to_bedrock_content_block()

        assert await plan_bedrock_media_transport(
            InlineMediaLimits(
                max_file_base64_size=25_000_000, max_total_base64_size=25_000_000
            ),
            s3_location_media_types=_STORED_IMAGES,
        )

        await file.resolve_bedrock_content_block(
            "us-east-1", s3_location_media_types=_STORED_IMAGES
        )
        assert "s3Location" in block["image"]["source"]

    async def test_the_default_limits_accept_the_largest_measured_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default boundary is the one measured against the backend.

        Every family that declares no limit of its own inherits this one, so a
        default lowered by accident would newly route working requests through
        storage — or refuse them outright on a model that reads no reference.
        The sizes are the ones the backend was measured against — 31,998,668
        base64 bytes accepted, 32,000,000 refused — written as literals so the
        probe survives an edit of the constant they were derived into.

        Ref: stdapi/input_file.py:CONVERSE_INLINE_BASE64_LIMIT
        """
        monkeypatch.setattr(InputFile, "to_s3", _fake_upload)
        at_limit = _remote_file(23_999_001)
        await at_limit.to_bedrock_content_block()

        assert await at_limit.get_base64_size() == 31_998_668
        assert not await plan_bedrock_media_transport(
            InlineMediaLimits(), s3_location_media_types=_STORED_IMAGES
        ), "the largest payload measured as accepted must still travel inline"

        past_limit = _remote_file(24_000_000)
        await past_limit.to_bedrock_content_block()

        assert await past_limit.get_base64_size() == 32_000_000
        assert await plan_bedrock_media_transport(
            InlineMediaLimits(), s3_location_media_types=_STORED_IMAGES
        ), "the smallest payload measured as refused must be read from storage"

    async def test_the_region_of_a_staged_attachment_is_pinned_for_the_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second call of a request follows the region the first one pinned.

        Each call picks a region on its own — and region routing hands successive
        calls different ones — while the attachment is staged in exactly one of
        them; only that region can read it back.

        Ref: stdapi/input_file.py:pin_bedrock_upload_region
        """
        monkeypatch.setattr(InputFile, "to_s3", _fake_upload)
        file = InputFile(_data_uri(44))
        await file.to_bedrock_content_block()
        await plan_bedrock_media_transport(
            InlineMediaLimits(max_file_base64_size=40, max_total_base64_size=40),
            s3_location_media_types=_STORED_IMAGES,
        )

        assert pin_bedrock_upload_region("us-east-1") == "us-east-1"
        assert pin_bedrock_upload_region("us-west-2") == "us-east-1"

    async def test_the_refusal_states_the_per_request_total_as_well(self) -> None:
        """Refusing a request over the total quotes the total, not only the per-file size.

        The staging can be forced by either bound, and a caller told only the
        per-file size cannot see that no attachment of theirs exceeds it.  What
        the deployment is missing to serve it is written to the server log, and
        must stay out of the message the caller reads.

        Ref: stdapi/input_file.py:inline_media_storage_error
        """
        error = inline_media_storage_error(
            InlineMediaLimits(max_file_base64_size=40, max_total_base64_size=80)
        )

        assert error.status == 413
        message = str(error)
        assert "30 bytes" in message, "the per-attachment size the model accepts"
        assert "60 bytes" in message, "the per-request size the model accepts"
        _assert_names_no_internals(message)
