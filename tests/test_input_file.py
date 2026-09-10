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
from datetime import UTC, datetime
from inspect import CORO_CLOSED, getcoroutinestate
from typing import TYPE_CHECKING, Any, NoReturn, Self, cast

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from botocore.exceptions import ClientError
from pybase64 import b64encode

from stdapi import aws_s3, input_file, security
from stdapi.api_errors import ApiError, denied_feature_unavailable
from stdapi.aws_s3 import BUCKET_TO_REGION, UPLOAD_CHUNK_SIZE, S3Object
from stdapi.cleanup import CLEANUPS
from stdapi.config import SETTINGS
from stdapi.files import encode_id_payload
from stdapi.files._core import _sanitize_filename
from stdapi.files._multipart import create_multipart_session
from stdapi.input_file import (
    FileIdInputFile,
    InlineMediaLimits,
    InputFile,
    inline_media_storage_error,
    pin_bedrock_upload_region,
    plan_bedrock_media_transport,
    resolve_all_bedrock_content_blocks,
)
from stdapi.utils import now_utc_timestamp
from tests._helpers import make_client_error

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator

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


@pytest.mark.parametrize(
    "accessor", ["get_content_type", "get_size", "get_filename", "to_data_uri"]
)
async def test_invalid_base64_is_a_caller_error_on_every_accessor(
    accessor: str,
) -> None:
    """Malformed base64 is a 400 whichever accessor decodes it first.

    Metadata resolution decodes a prefix of the payload to detect its type, and
    every ingest route reaches it before reading the content: ``POST /v1/files``
    asks for the filename to build the object's ``Content-Disposition``, and
    image moderation asks for the content type. Letting the decoder's
    ``ValueError`` escape from there answers a malformed request with a 500.

    Ref: stdapi/input_file.py:_Base64Source._resolve_metadata
         stdapi/files/_core.py:upload_file
    """
    with pytest.raises(ApiError, match=r"^Invalid base64 data\.$") as exc:
        await getattr(InputFile("not-valid-base64!!!"), accessor)()
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


#: An S3 denial message, as IAM writes one: principal, action, resource.
_S3_DENIAL_MESSAGE = (
    "User: arn:aws:sts::123456789012:assumed-role/stdapi-ai/task is not "
    "authorized to perform: s3:GetObject on resource: "
    "arn:aws:s3:::a-caller-owned-bucket/private.png"
)


def _s3_denial(operation: str) -> ClientError:
    """Build the denial S3 answers an ungranted read of *operation* with.

    Args:
        operation: The S3 API operation that was refused.

    Returns:
        The corresponding botocore error.
    """
    return make_client_error(
        "AccessDenied", operation, message=_S3_DENIAL_MESSAGE, status=403
    )


class _DeniedS3Client:
    """Stub S3 client refusing a metadata read the way S3 refuses one."""

    async def head_object(self, **_kwargs: object) -> NoReturn:
        """Refuse the read.

        Raises:
            ClientError: Always, with the bare ``AccessDenied`` code S3 uses.
        """
        denial = _s3_denial("HeadObject")
        raise denial


@pytest.fixture
def denied_input_buckets(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Accept two input buckets whose reads are denied, one external, one the server's own.

    Both are declared in ``aws_s3_accepted_buckets``, so only ownership tells
    them apart -- which is the split the guard is built on, rebuilt here from
    the settings rather than restated.

    Returns:
        The caller-owned bucket and the deployment-owned bucket, in that order.
    """
    caller_bucket = "a-caller-owned-bucket"
    own_bucket = "a-deployment-owned-bucket"
    monkeypatch.setitem(aws_s3.BUCKET_TO_REGION, own_bucket, "us-east-1")
    monkeypatch.setattr(
        SETTINGS,
        "aws_s3_accepted_buckets",
        {caller_bucket: "us-east-1", own_bucket: "us-east-1"},
    )
    monkeypatch.setattr(
        aws_s3,
        "CALLER_INPUT_BUCKETS",
        aws_s3._caller_input_buckets(),  # noqa: SLF001
    )
    monkeypatch.setattr(
        input_file, "_ACCEPTED_BUCKETS", frozenset({caller_bucket, own_bucket})
    )
    monkeypatch.setattr(input_file, "get_client", lambda *_a, **_k: _DeniedS3Client())
    return caller_bucket, own_bucket


class TestDeniedS3InputNamesWhoCanFixIt:
    """A denied read of an S3 input answers the audience that can act on it.

    A denial is normally the operator's misconfiguration, so it reads as a
    feature this deployment cannot run.  An object the *request* named is the
    exception: it sits in storage the deployment does not own, so the refusal
    is attributable to that request -- the caller's own bucket policy, key or
    typo -- exactly as a denial evaluated against an end user's own session is.

    Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/troubleshoot-403-errors.html
         stdapi/aws_s3.py:caller_input_denial_guard
         stdapi/api_errors.py:denied_feature_unavailable
    """

    async def test_an_object_the_request_named_is_refused_as_the_callers_own(
        self, denied_input_buckets: tuple[str, str]
    ) -> None:
        """A denied caller-supplied object answers 4xx, naming the input it could not read.

        Answering "not available on this deployment, contact your administrator"
        would send the caller to someone who cannot fix their bucket policy, and
        hide the one input they had to correct.
        """
        caller_bucket, _own_bucket = denied_input_buckets
        uri = f"s3://{caller_bucket}/private.png"

        with pytest.raises(ApiError) as exc_info:
            await InputFile(uri).get_size()

        error = exc_info.value
        message = error.args[0]
        assert error.status == 400, "the caller's own input is a request error"
        assert error.code == "input_access_denied"
        assert uri in message, "the refusal must name the input that could not be read"
        assert "administrator" not in message, (
            "the caller can fix this one themselves; the operator cannot"
        )
        assert "arn:aws" not in message, "AWS's own denial names a role, and is not it"
        assert "123456789012" not in message, "nor is the account it names"

    async def test_the_deployments_own_storage_stays_a_feature_it_cannot_run(
        self, denied_input_buckets: tuple[str, str]
    ) -> None:
        """A denied read of a deployment-owned bucket keeps the generic 503.

        The very same code, operation and settings entry: only the bucket's
        ownership differs, and it alone decides.  Nothing of that bucket reaches
        the caller either -- naming it would map the deployment's storage.
        """
        _caller_bucket, own_bucket = denied_input_buckets

        with pytest.raises(ClientError) as exc_info:
            await InputFile(f"s3://{own_bucket}/private.png").get_size()

        denied = denied_feature_unavailable(exc_info.value)
        assert denied is not None, "an unattributable denial is the deployment's"
        assert denied.status == 503
        assert denied.code == "feature_unavailable"
        assert own_bucket not in denied.args[0]

    async def test_a_download_of_a_caller_named_object_is_refused_the_same_way(
        self, denied_input_buckets: tuple[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The body read answers as the metadata read does, not only the ``HeadObject``.

        Metadata and content are resolved by separate calls, so a guard on one
        of them alone still surfaces the other denial as an outage.
        """
        caller_bucket, _own_bucket = denied_input_buckets
        uri = f"s3://{caller_bucket}/private.png"

        async def _denied_get(*_args: object, **_kwargs: object) -> NoReturn:
            """Refuse the download.

            Raises:
                ClientError: Always, as S3 refuses an ungranted ``GetObject``.
            """
            denial = _s3_denial("GetObject")
            raise denial

        monkeypatch.setattr(input_file, "get_bytes_from_s3", _denied_get)

        with pytest.raises(ApiError) as exc_info:
            await InputFile(uri).to_bytes()

        assert exc_info.value.code == "input_access_denied"
        assert uri in exc_info.value.args[0]

    async def test_a_copy_denial_is_not_attributed_to_the_caller(
        self, denied_input_buckets: tuple[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A denied ``CopyObject`` keeps the 503: its resource may be either side.

        A cross-region copy needs read on the caller's object *and* write on the
        deployment's bucket, and one ``AccessDenied`` does not say which was
        refused -- so it stays the answer that blames nobody.
        """
        caller_bucket, _own_bucket = denied_input_buckets

        async def _denied_copy(*_args: object, **_kwargs: object) -> NoReturn:
            """Refuse the copy.

            Raises:
                ClientError: Always, as S3 refuses an ungranted ``CopyObject``.
            """
            denial = _s3_denial("CopyObject")
            raise denial

        monkeypatch.setattr(input_file, "copy_s3_object", _denied_copy)

        with pytest.raises(ClientError) as exc_info:
            await InputFile(f"s3://{caller_bucket}/private.png").to_s3("eu-west-3")

        denied = denied_feature_unavailable(exc_info.value)
        assert denied is not None, "an ambiguous denial is still the deployment's"
        assert denied.status == 503


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

    def head(self, url: str, *, allow_redirects: bool = True) -> _StubHttpResponse:
        """Serve the stubbed response to a ``HEAD``.

        The stub answers for the resource itself, which the probe only ever
        reaches by following redirects, so an unfollowed probe is refused here
        rather than silently served the wrong body.
        """
        assert allow_redirects, "the metadata probe must land on the final resource"
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


#: Body a local origin serves; a real signature so a sniffed type is right too.
_ORIGIN_BODY: bytes = b"%PDF-1.7\n" + b"remote object\n" * 1024

#: Loopback address of the local origin, the only one the SSRF policy is relaxed for.
_LOOPBACK: str = "127.0.0.1"


@pytest.fixture
async def serve_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Callable[[web.Application], Awaitable[str]]]:
    """Start local origins the input downloader is allowed to reach.

    Every connect target is validated before the connection is made, and
    loopback is refused by that policy, so the test origin's own address is
    allowed for the duration of the test and nothing else is.

    Yields:
        A coroutine returning the base URL of a started application.
    """
    monkeypatch.setattr(security, "_is_unsafe_ip", lambda ip: str(ip) != _LOOPBACK)
    servers: list[TestServer] = []

    async def _serve(app: web.Application) -> str:
        server = TestServer(app, host=_LOOPBACK)
        await server.start_server()
        servers.append(server)
        return str(server.make_url("")).rstrip("/")

    yield _serve
    for server in servers:
        await server.close()


def _serve_object(
    requests: list[str],
    *,
    head_status: int = 200,
    get_status: int = 200,
    declares_total: bool = True,
    declares_type: bool = True,
    content_type: str = "application/pdf",
) -> web.Application:
    """Return an application serving ``/object.pdf`` and recording every request.

    ``HEAD`` and ``GET`` are answered separately so a probe an origin refuses —
    a URL signed for ``GET``, or a server that does not implement the method —
    is reproduced exactly. The ``GET`` honours a single byte range, as an origin
    a ranged probe is worth issuing against does.

    Args:
        requests: Collects ``"<method> <path>"`` for every request served.
        head_status: Status answered to ``HEAD``.
        get_status: Status answered to ``GET``.
        declares_total: When ``False`` a ranged answer declares an unknown
            total, as an origin streaming a resource of unknown length does.
        declares_type: When ``False`` the origin names no content type.
        content_type: Content type the origin declares the object with.

    Returns:
        The application, ready to be started.
    """

    def _headers() -> dict[str, str]:
        """Return the headers the origin describes the object with.

        Returns:
            A ``Content-Disposition``, and a ``Content-Type`` unless the origin
            declares none.
        """
        return {
            "Content-Disposition": 'attachment; filename="object.pdf"',
            "Content-Type": content_type if declares_type else "",
        }

    async def _head(request: web.Request) -> web.Response:
        requests.append(f"{request.method} {request.rel_url.path}")
        if head_status != 200:
            return web.Response(status=head_status)
        return web.Response(body=_ORIGIN_BODY, headers=_headers())

    async def _get(request: web.Request) -> web.Response:
        requests.append(f"{request.method} {request.rel_url.path}")
        if get_status != 200:
            return web.Response(status=get_status)
        headers = _headers()
        if range_header := request.headers.get("Range"):
            first, _, last = range_header.removeprefix("bytes=").partition("-")
            start, stop = int(first), min(int(last) + 1, len(_ORIGIN_BODY))
            total = str(len(_ORIGIN_BODY)) if declares_total else "*"
            headers["Content-Range"] = f"bytes {start}-{stop - 1}/{total}"
            return web.Response(
                status=206, body=_ORIGIN_BODY[start:stop], headers=headers
            )
        return web.Response(body=_ORIGIN_BODY, headers=headers)

    app = web.Application()
    app.router.add_route("HEAD", "/object.pdf", _head)
    app.router.add_get("/object.pdf", _get, allow_head=False)
    return app


@pytest.mark.usefixtures("input_files")
class TestRemoteInputMetadata:
    """A remote input is measured from the resource its download will fetch.

    Both properties resolved here decide the request: the content type decides
    how the attachment is described to the model, and the size decides whether
    it travels inline. A probe that describes a redirect page instead of the
    object, or that fails where the download would have succeeded, therefore
    answers for something the caller never named — while upstream simply
    accepts "a fully qualified URL to an image file" and fetches it.

    Ref: https://developers.openai.com/api/docs/guides/images-vision
         https://www.rfc-editor.org/rfc/rfc9110.html#name-content-range
         stdapi/input_file.py:_HttpSource._resolve_metadata
    """

    async def test_the_probe_describes_the_target_of_a_redirect(
        self, serve_origin: Callable[[web.Application], Awaitable[str]]
    ) -> None:
        """A redirecting URL is typed and sized from the object it redirects to.

        The download follows the redirect, so a probe that stops at the 3xx
        reports the redirect page's own type and length — an image arrives
        described as a few hundred bytes of markup.
        """
        requests: list[str] = []
        app = _serve_object(requests)

        async def _redirect(request: web.Request) -> web.Response:
            requests.append(f"{request.method} {request.rel_url.path}")
            return web.Response(
                status=302,
                text="",
                content_type="text/html",
                headers={"Location": "/object.pdf"},
            )

        app.router.add_route("*", "/redirected.pdf", _redirect)
        base = await serve_origin(app)

        file = InputFile(f"{base}/redirected.pdf")

        assert await file.get_content_type() == "application/pdf"
        assert await file.get_size() == len(_ORIGIN_BODY)
        assert requests == ["HEAD /redirected.pdf", "HEAD /object.pdf"], (
            "the redirect is followed by the probe itself, not by a second download"
        )

    @pytest.mark.parametrize("head_status", [403, 405])
    async def test_an_origin_refusing_the_probe_is_measured_by_a_ranged_read(
        self,
        serve_origin: Callable[[web.Application], Awaitable[str]],
        head_status: int,
    ) -> None:
        """A URL whose origin refuses ``HEAD`` is still accepted as an input.

        A pre-signed link signs the method it was issued for, so a ``HEAD``
        against one signed for ``GET`` is refused with 403; other origins answer
        405. Neither says anything about the object, which the ranged read then
        reports in full.
        """
        requests: list[str] = []
        base = await serve_origin(_serve_object(requests, head_status=head_status))

        file = InputFile(f"{base}/object.pdf")

        assert await file.get_content_type() == "application/pdf"
        assert await file.get_size() == len(_ORIGIN_BODY), (
            "the total size is read from the range, not the length of the part served"
        )
        assert await file.get_filename() == "object.pdf"
        assert requests == ["HEAD /object.pdf", "GET /object.pdf"], (
            "one ranged read answers for all three properties"
        )

    async def test_a_ranged_read_declaring_no_total_leaves_the_size_unknown(
        self, serve_origin: Callable[[web.Application], Awaitable[str]]
    ) -> None:
        """An origin that declares no total size is read as declaring none at all.

        A resource streamed without a known length is answered with an unknown
        total. Taking the served part's own length for it would report a large
        attachment as a few kilobytes, which is how it ends up sent by a route
        that cannot carry it; an undeclared size is reported as undeclared.
        """
        requests: list[str] = []
        base = await serve_origin(
            _serve_object(requests, head_status=405, declares_total=False)
        )

        assert await InputFile(f"{base}/object.pdf").get_size() == 0

    async def test_an_origin_naming_no_type_has_the_content_identified_for_it(
        self, serve_origin: Callable[[web.Application], Awaitable[str]]
    ) -> None:
        """An input an origin describes with no content type is identified from its bytes.

        The type decides how the attachment is sent on, so an origin that names
        none cannot be left to make the input unusable: the same read that
        measures it also identifies it.
        """
        requests: list[str] = []
        base = await serve_origin(
            _serve_object(requests, head_status=405, declares_type=False)
        )

        file = InputFile(f"{base}/object.pdf")

        assert await file.get_content_type() == "application/pdf"
        assert await file.get_size() == len(_ORIGIN_BODY)

    @pytest.mark.parametrize(
        "declared", ["application/octet-stream", "binary/octet-stream"]
    )
    async def test_an_origin_naming_a_generic_type_has_the_content_identified_for_it(
        self, serve_origin: Callable[[web.Application], Awaitable[str]], declared: str
    ) -> None:
        """An ``octet-stream`` header is read as naming no type at all.

        It is what an origin answers about a resource it cannot describe -- and
        what a signed link to any object commonly carries. The format is what
        selects the block an attachment travels in, so trusting that header
        would make every such input unusable; the bytes are read instead.
        """
        requests: list[str] = []
        base = await serve_origin(_serve_object(requests, content_type=declared))

        file = InputFile(f"{base}/object.pdf")

        assert await file.get_content_type() == "application/pdf"
        assert await file.get_filename() == "object.pdf", (
            "the header still answers for everything it does describe"
        )
        assert requests == ["HEAD /object.pdf", "GET /object.pdf"], (
            "the type the origin declined to name costs one ranged read"
        )

    async def test_a_ranged_read_that_is_refused_too_is_reported(
        self, serve_origin: Callable[[web.Application], Awaitable[str]]
    ) -> None:
        """A URL refusing both requests is refused, with what the origin answered.

        A signed link that has expired refuses every method, so the fallback is
        no more entitled to an answer than the probe was — and the caller is
        told the URL is unusable rather than left with an empty description.
        """
        requests: list[str] = []
        base = await serve_origin(
            _serve_object(requests, head_status=405, get_status=403)
        )

        with pytest.raises(ApiError) as exc:
            await InputFile(f"{base}/object.pdf").get_size()

        assert exc.value.status == 400
        assert "403" in str(exc.value)
        assert requests == ["HEAD /object.pdf", "GET /object.pdf"]

    async def test_an_unreadable_url_is_refused_without_a_second_request(
        self, serve_origin: Callable[[web.Application], Awaitable[str]]
    ) -> None:
        """A probe that fails because the object is missing is reported as such.

        Only a refusal of the method is worth retrying with a ranged read; a
        404 is the answer about the object itself, and asking twice would double
        the cost of every unusable URL.
        """
        requests: list[str] = []
        base = await serve_origin(_serve_object(requests, head_status=404))

        with pytest.raises(ApiError) as exc:
            await InputFile(f"{base}/object.pdf").get_size()

        assert exc.value.status == 400
        assert requests == ["HEAD /object.pdf"]

    async def test_a_probe_blocked_by_the_egress_policy_is_not_retried(
        self, serve_origin: Callable[[web.Application], Awaitable[str]]
    ) -> None:
        """A URL redirecting into a refused address is reported as forbidden, once.

        The refusal comes from the deployment's own egress policy rather than
        from the origin, so it is not the kind an origin can be asked again
        about: reporting it as a method refusal would send a second request to
        the same blocked target and answer 400 for what is a 403.
        """
        requests: list[str] = []
        app = _serve_object(requests)

        async def _redirect(request: web.Request) -> web.Response:
            requests.append(f"{request.method} {request.rel_url.path}")
            return web.Response(
                status=302, headers={"Location": "http://169.254.169.254/"}
            )

        app.router.add_route("*", "/blocked.pdf", _redirect)
        base = await serve_origin(app)

        with pytest.raises(ApiError) as exc:
            await InputFile(f"{base}/blocked.pdf").get_size()

        assert exc.value.status == 403
        assert requests == ["HEAD /blocked.pdf"]

    async def test_a_refused_input_never_repeats_the_url_query(
        self, serve_origin: Callable[[web.Application], Awaitable[str]]
    ) -> None:
        """The refusal redacts the query, which carries the signature of a signed link.

        A signed URL's query is a credential for the object: quoting it back in
        an error hands it to whoever reads the response or the log.
        """
        requests: list[str] = []
        base = await serve_origin(_serve_object(requests, head_status=404))

        with pytest.raises(ApiError) as exc:
            await InputFile(f"{base}/object.pdf?X-Amz-Signature=00secret00").get_size()

        assert "00secret00" not in str(exc.value)
        assert "<redacted>" in str(exc.value)

    async def test_an_unusable_url_is_refused_without_repeating_its_query(self) -> None:
        """A URL refused before any request is reported with its query redacted too.

        A malformed URL never reaches an origin, so the refusal comes from the
        client library and quotes what it was given — which is the whole URL,
        credential included.
        """
        with pytest.raises(ApiError) as exc:
            await InputFile("https:///object.pdf?X-Amz-Signature=00secret00").get_size()

        assert "00secret00" not in str(exc.value)
        assert "<redacted>" in str(exc.value)
        assert exc.value.status == 400


@pytest.mark.usefixtures("input_files")
class TestRemoteInputFetchFailures:
    """A remote input that cannot be fetched is answered as a caller error.

    Whichever way the attachment travels — read into the request, or staged
    first because it is too large for that — it is fetched from an origin the
    caller chose: a link that has expired, a host that has gone away, a redirect
    into an address the deployment refuses to connect to. None of them is a
    fault of this deployment, and none may surface as one.

    Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html
         stdapi/input_file.py:_HttpSource._read
         stdapi/input_file.py:_HttpSource.to_s3
    """

    async def test_a_url_that_fails_on_download_is_a_client_error(
        self,
        serve_origin: Callable[[web.Application], Awaitable[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An origin refusing the download is reported as a bad input, with its status."""
        monkeypatch.setattr(SETTINGS, "max_input_file_size", 0)
        requests: list[str] = []
        base = await serve_origin(_serve_object(requests, get_status=403))

        with pytest.raises(ApiError) as exc:
            await InputFile(f"{base}/object.pdf").to_bytes()

        assert exc.value.status == 400
        assert "403" in str(exc.value)

    async def test_a_url_that_fails_at_staging_is_a_client_error(
        self,
        serve_origin: Callable[[web.Application], Awaitable[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An origin refusing the staged download is reported as a bad input, not a fault."""
        monkeypatch.setattr(input_file, "put_s3_object", _fake_upload)
        requests: list[str] = []
        base = await serve_origin(_serve_object(requests, get_status=404))

        with pytest.raises(ApiError) as exc:
            await InputFile(f"{base}/object.pdf").to_s3("us-east-1")

        assert exc.value.status == 400
        assert "404" in str(exc.value), (
            "the caller needs to know what the origin answered for their URL"
        )
        assert requests == ["GET /object.pdf"]

    async def test_a_staged_download_redirected_into_a_refused_target_is_forbidden(
        self,
        serve_origin: Callable[[web.Application], Awaitable[str]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A redirect into an address the egress policy blocks is refused as forbidden.

        The block is a policy decision about where the deployment may connect,
        which is reported as 403 on every other path an input takes; staging
        must not be the one that reports it as a server fault.
        """
        monkeypatch.setattr(input_file, "put_s3_object", _fake_upload)

        async def _redirect(_request: web.Request) -> web.Response:
            return web.Response(
                status=302, headers={"Location": "http://169.254.169.254/"}
            )

        app = web.Application()
        app.router.add_route("*", "/object.pdf", _redirect)
        base = await serve_origin(app)

        with pytest.raises(ApiError) as exc:
            await InputFile(f"{base}/object.pdf").to_s3("us-east-1")

        assert exc.value.status == 403


class _RecordingS3Client:
    """S3 stand-in recording the key every ``HeadObject`` addressed."""

    def __init__(self) -> None:
        self.keys: list[str] = []

    async def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        """Record the addressed key and answer with a minimal object description.

        Returns:
            The head of a small object.
        """
        self.keys.append(Key)
        return {"ContentLength": len(_ORIGIN_BODY), "ContentType": "application/pdf"}


@pytest.mark.usefixtures("input_files")
class TestStoredInputAddressing:
    """An S3 HTTP URL addresses the object the caller copied it from.

    A URL carries its key percent-encoded — that is what the console and every
    signed link produce — while the object itself is named by the decoded key.
    Reading the URL form literally turns every key containing a space, an
    accent or a reserved character into a file that does not exist.

    Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html
         stdapi/input_file.py:InputFile._normalize_and_detect_origin
    """

    @staticmethod
    def _record_s3(monkeypatch: pytest.MonkeyPatch) -> _RecordingS3Client:
        """Answer the metadata lookup locally, recording what it addressed.

        Returns:
            The recording client.
        """
        client = _RecordingS3Client()
        monkeypatch.setattr(input_file, "get_client", lambda *_a, **_kw: client)
        return client

    @pytest.mark.parametrize(
        "url_form",
        [
            pytest.param(
                "https://{bucket}.s3.us-east-1.amazonaws.com/{key}", id="host"
            ),
            pytest.param(
                "https://s3.us-east-1.amazonaws.com/{bucket}/{key}", id="path"
            ),
        ],
    )
    async def test_an_encoded_url_key_is_decoded_before_the_object_is_read(
        self, monkeypatch: pytest.MonkeyPatch, url_form: str
    ) -> None:
        """The key a URL encodes is decoded, in both URL forms."""
        bucket = _allowed_bucket(monkeypatch)
        client = self._record_s3(monkeypatch)
        url = url_form.format(
            bucket=bucket, key="reports/q1%202026%20%C3%A9t%C3%A9.pdf"
        )

        file = InputFile(url)
        await file.get_size()

        assert client.keys == ["reports/q1 2026 été.pdf"]
        assert repr(file) == f"s3://{bucket}/reports/q1 2026 été.pdf"

    async def test_a_stored_uri_key_is_taken_as_written(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key given as an ``s3://`` URI is used verbatim.

        Only the URL form is encoded. A key written directly is already the
        object's own name, and decoding it a second time would address a
        different object — one whose name happens to contain a percent sign.
        """
        bucket = _allowed_bucket(monkeypatch)
        client = self._record_s3(monkeypatch)

        await InputFile(f"s3://{bucket}/reports/q1%202026.pdf").get_size()

        assert client.keys == ["reports/q1%202026.pdf"]


@pytest.mark.parametrize(
    ("sent", "stored"),
    [
        pytest.param("reports/q3.pdf", "q3.pdf", id="path"),
        pytest.param("/var/../reports/q3.pdf", "q3.pdf", id="relative-path"),
        pytest.param(r"C:\reports\q3.pdf", "q3.pdf", id="windows-path"),
        pytest.param("Q3: results.pdf", "Q3: results.pdf", id="colon"),
        pytest.param("what?.pdf", "what?.pdf", id="question-mark"),
        pytest.param("a*b<c>d|e.pdf", "a*b<c>d|e.pdf", id="shell-glob-and-redirection"),
        pytest.param("2026-09-09T10:00:00.log", "2026-09-09T10:00:00.log", id="stamp"),
        pytest.param("a" * 500, "a" * 500, id="at-the-length-cap"),
    ],
)
def test_only_the_final_path_component_of_a_filename_is_kept(
    sent: str, stored: str
) -> None:
    """A filename is stored as sent, minus any path leading up to it.

    The uploading client's name is kept whole wherever a
    ``Content-Disposition`` header can hold it: the characters a filesystem
    dislikes are irrelevant here, because the name is never a path.  A leading
    path is dropped rather than refused, since a legal upload must not become a
    400.

    Ref: https://platform.claude.com/docs/en/api/files/upload
         stdapi/files/_core.py:_sanitize_filename
    """
    assert _sanitize_filename(sent, "application/pdf") == stored


@pytest.mark.parametrize(
    ("mime_type", "stored"),
    [
        pytest.param("application/pdf", "unnamed.pdf", id="pdf"),
        pytest.param("text/plain", "unnamed.txt", id="text"),
        pytest.param("application/x-nonesuch", "unnamed", id="unknown-type"),
        pytest.param("", "unnamed", id="no-type"),
    ],
)
def test_a_filename_that_is_absent_or_only_a_path_falls_back_to_unnamed(
    mime_type: str, stored: str
) -> None:
    """A name that survives no path component becomes ``unnamed`` plus the type's extension.

    The extension is dropped rather than guessed when the media type names
    none, so the fallback never invents a type the bytes do not have.

    Ref: https://platform.claude.com/docs/en/api/files/upload
         stdapi/files/_core.py:_sanitize_filename
    """
    assert _sanitize_filename("", mime_type) == stored
    assert _sanitize_filename("reports/", mime_type) == stored


async def test_create_multipart_session_rejects_unsafe_filename() -> None:
    """A filename carrying a quote is rejected before any S3 call.

    The filename ends up in the object's ``Content-Disposition`` header, whose
    quoted form a quote would close early, so it is one of the few names that
    cannot be kept.  The check runs ahead of the bucket lookup — which is why
    this test needs no AWS access.

    Ref: stdapi/files/_core.py:_sanitize_filename
         stdapi/files/_multipart.py:create_multipart_session
    """
    with pytest.raises(ApiError) as exc:
        await create_multipart_session('bad"name.txt', "text/plain", "", 10)
    assert exc.value.status == 400
    assert "cannot be stored" in str(exc.value), exc.value.args


@pytest.mark.parametrize(
    "filename",
    [
        pytest.param('bad"name.txt', id="quote"),
        pytest.param("bad\nname.txt", id="newline"),
        pytest.param("bad\x7fname.txt", id="delete"),
    ],
)
def test_a_filename_a_stored_header_cannot_hold_is_refused(filename: str) -> None:
    """Only what breaks the header the name is written to is refused.

    A quote closes the quoted value early, and a control character cannot
    appear in a header value at all; both would corrupt the stored metadata
    rather than merely look odd.

    Ref: https://www.rfc-editor.org/rfc/rfc9110.html#name-field-values
         stdapi/files/_core.py:_sanitize_filename
    """
    with pytest.raises(ApiError, match="cannot be stored") as exc:
        _sanitize_filename(filename, "text/plain")
    assert exc.value.status == 400


async def test_create_multipart_session_rejects_overlong_filename() -> None:
    """A filename longer than 500 characters is rejected before any S3 call.

    The filename is interpolated into the object's ``Content-Disposition`` header, and
    500 is the Files API cap the gateway mirrors.

    Ref: https://platform.claude.com/docs/en/api/files/upload
         stdapi/files/_core.py:_sanitize_filename
         stdapi/files/_multipart.py:create_multipart_session
    """
    with pytest.raises(ApiError, match="maximum length of 500") as exc:
        await create_multipart_session("a" * 501, "text/plain", "", 10)
    assert exc.value.status == 400


def test_the_length_cap_applies_to_the_kept_component_only() -> None:
    """An over-long path whose final component fits is accepted, not refused.

    The cap describes the name the API reports, and that is what is left once
    the path is dropped; measuring the value as sent would refuse a filename
    the response would have shown as well within the limit.

    Ref: https://platform.claude.com/docs/en/api/files/upload
         stdapi/files/_core.py:_sanitize_filename
    """
    assert _sanitize_filename("a" * 600 + "/q3.pdf", "application/pdf") == "q3.pdf"


async def test_filename_length_check_runs_before_the_character_check() -> None:
    """At exactly 500 characters the length branch passes and the character check runs.

    The length branch is evaluated first, so it would mask the character rejection for
    any name at or above the cap; a 500-character name carrying a quote must still
    report the character failure.

    Ref: stdapi/files/_core.py:_sanitize_filename
    """
    filename = 'a"' + "a" * 498
    assert len(filename) == 500
    with pytest.raises(ApiError, match="cannot be stored") as exc:
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

        # A bare base64 payload's size is otherwise learned by decoding a prefix.
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


#: Content the stored-object stub serves, and the length its metadata reports.
_STORED_FILE_CONTENT = b"stored file content"


class _StubFilesS3Client:
    """Stub S3 holding one uploaded file, answering as S3 answers for a real one.

    ``HeadObject`` carries the ``expires-at`` user metadata the Files API writes
    at upload, alongside the fields every response carries.
    """

    def __init__(self, expires_at: int | None) -> None:
        self.expires_at = expires_at
        self.head_calls = 0
        self.reads: list[str] = []
        self.deleted: list[str] = []

    async def head_object(self, **_kwargs: object) -> dict[str, Any]:
        """Describe the stored object.

        Returns:
            The ``HeadObject`` response for the file.
        """
        self.head_calls += 1
        return {
            "ContentDisposition": 'attachment; filename="note.txt"',
            "ContentType": "text/plain",
            "ContentLength": len(_STORED_FILE_CONTENT),
            "LastModified": datetime(2026, 1, 1, tzinfo=UTC),
            "Metadata": {
                "purpose": "user_data",
                "expires-at": "" if self.expires_at is None else str(self.expires_at),
            },
        }

    async def delete_object(self, **kwargs: object) -> None:
        """Record the deletion an expired object is queued for."""
        self.deleted.append(str(kwargs["Key"]))


@pytest.fixture
def stored_file(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[int | None], tuple[str, _StubFilesS3Client]]:
    """Serve one uploaded file from a stub, whatever storage the deployment has.

    Returns:
        A factory taking the expiry stored on the object — Unix seconds, or
        ``None`` for a file that never expires — and returning the file
        identifier a client references it by, with the stub answering for it.
    """
    bucket = "a-files-bucket"
    monkeypatch.setattr(SETTINGS, "aws_s3_bucket", bucket)

    def _serve(expires_at: int | None) -> tuple[str, _StubFilesS3Client]:
        """Install the stub and mint an identifier resolving to it.

        Returns:
            The file identifier and the stub answering for it.
        """
        stub = _StubFilesS3Client(expires_at)

        async def _read(_bucket: str, key: str) -> bytes:
            """Serve the stored bytes, recording that they were read.

            Returns:
                The stored content.
            """
            stub.reads.append(key)
            return _STORED_FILE_CONTENT

        monkeypatch.setattr(input_file, "get_client", lambda *_a, **_k: stub)
        monkeypatch.setattr(aws_s3, "get_client", lambda *_a, **_k: stub)
        monkeypatch.setattr(input_file, "get_bytes_from_s3", _read)
        return f"file-{encode_id_payload(bucket)}", stub

    return _serve


@pytest.fixture
def scheduled_cleanups() -> Iterator[list[Awaitable[None]]]:
    """Bind the cleanup context a request schedules its background work in.

    Whatever the test leaves pending is dropped the way a request cancelled
    before its cleanups run drops them.

    Yields:
        The pending cleanups, for the test to await when it wants them run.
    """
    token = CLEANUPS.set([])
    pending = CLEANUPS.get()
    try:
        yield pending
    finally:
        CLEANUPS.reset(token)
        for cleanup in pending:
            cast("Coroutine[Any, Any, None]", cleanup).close()


def _assert_expired(error: ApiError, file_id: str) -> None:
    """Assert *error* is the refusal a client gets for a file that no longer exists."""
    assert error.status == 404
    assert error.code == "not_found"
    message = str(error)
    assert file_id in message, "the caller needs to know which file is gone"
    assert "expired" in message.lower(), "and why it is gone"
    _assert_names_no_internals(message)


@pytest.mark.usefixtures("scheduled_cleanups")
class TestUploadedFileExpiry:
    """A file past its expiry is gone for inference too, not only for the Files API.

    Expiry is what a client uses to bound how long content it uploaded stays
    readable, so it has to hold on every path that reads the file — a request
    attaching it to a model included. Storage deletes expired objects on its own
    schedule, which is why the check is made against the clock at every read
    rather than trusted to happen on time.

    Ref: https://stdapi.ai/api_openai_files/
         stdapi/input_file.py:_S3Source
    """

    async def test_the_metadata_of_an_expired_file_is_refused(
        self, stored_file: Callable[[int | None], tuple[str, _StubFilesS3Client]]
    ) -> None:
        """Reading an expired file's metadata answers 404, as retrieving it does.

        Ref: stdapi/input_file.py:_S3Source._resolve_metadata
        """
        file_id, _stub = stored_file(now_utc_timestamp() - 60)

        with pytest.raises(ApiError) as exc:
            await InputFile(f"file-id:{file_id}").get_size()

        _assert_expired(exc.value, file_id)

    async def test_the_content_of_an_expired_file_is_never_read(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stored_file: Callable[[int | None], tuple[str, _StubFilesS3Client]],
    ) -> None:
        """An expired file's bytes are refused without being downloaded.

        With ``max_input_file_size`` disabled nothing measures the file before
        reading it, so a check made only while resolving its size would let the
        content through on exactly the deployments that set no limit.

        Ref: stdapi/input_file.py:_S3Source._read
        """
        monkeypatch.setattr(SETTINGS, "max_input_file_size", 0)
        file_id, stub = stored_file(now_utc_timestamp() - 60)

        with pytest.raises(ApiError) as exc:
            await InputFile(f"file-id:{file_id}").to_bytes()

        _assert_expired(exc.value, file_id)
        assert stub.reads == [], "the content of an expired file must not be read"

    async def test_an_expired_file_is_not_handed_to_the_model_by_reference(
        self, stored_file: Callable[[int | None], tuple[str, _StubFilesS3Client]]
    ) -> None:
        """An expired file is refused when it would travel as a stored reference.

        Large attachments are handed to the model as a reference instead of
        inline bytes, and that path reads no metadata of its own: the file would
        otherwise expire for small requests and stay readable for big ones.

        Ref: stdapi/input_file.py:_S3Source.to_s3
        """
        file_id, _stub = stored_file(now_utc_timestamp() - 60)
        file = InputFile(f"file-id:{file_id}")

        with pytest.raises(ApiError) as exc:
            await file.to_s3(SETTINGS.aws_bedrock_regions[0])

        _assert_expired(exc.value, file_id)

    async def test_a_typed_file_id_is_checked_like_the_uri(
        self, stored_file: Callable[[int | None], tuple[str, _StubFilesS3Client]]
    ) -> None:
        """Both spellings of a file reference expire together.

        A chat content part names the file in a typed ``file_id`` field while a
        string-overloaded field takes the ``file-id:`` URI: the same file, so
        the same answer once it has expired.

        Ref: stdapi/input_file.py:FileIdInputFile
        """
        file_id, _stub = stored_file(now_utc_timestamp() - 60)

        with pytest.raises(ApiError) as exc:
            await FileIdInputFile(file_id).get_size()

        _assert_expired(exc.value, file_id)

    async def test_an_expired_file_is_queued_for_deletion(
        self,
        stored_file: Callable[[int | None], tuple[str, _StubFilesS3Client]],
        scheduled_cleanups: list[Awaitable[None]],
    ) -> None:
        """The refused object is deleted after the response, not left to storage alone.

        A file whose expiry a request notices is removed there and then, so
        content a client asked to expire does not sit in storage until the
        storage-side sweep gets to it.

        Ref: stdapi/files/_core.py:_get_file_impl
        """
        file_id, stub = stored_file(now_utc_timestamp() - 60)

        with pytest.raises(ApiError):
            await InputFile(f"file-id:{file_id}").get_size()

        assert len(scheduled_cleanups) == 1, "the expired object is deleted once"
        await scheduled_cleanups.pop()
        stored_key = f"{SETTINGS.aws_s3_files_prefix}{file_id.removeprefix('file-')}"
        assert stub.deleted == [stored_key]

    async def test_a_file_within_its_expiry_costs_one_metadata_read(
        self, stored_file: Callable[[int | None], tuple[str, _StubFilesS3Client]]
    ) -> None:
        """A live file is served, and the expiry check adds no second lookup.

        Ref: stdapi/input_file.py:_S3Source._enforce_expiry
        """
        file_id, stub = stored_file(now_utc_timestamp() + 3600)
        file = InputFile(f"file-id:{file_id}")

        assert await file.get_size() == len(_STORED_FILE_CONTENT)
        assert await file.to_bytes() == _STORED_FILE_CONTENT
        assert stub.head_calls == 1, "the file is described once, then read"

    async def test_a_file_with_no_expiry_is_served(
        self, stored_file: Callable[[int | None], tuple[str, _StubFilesS3Client]]
    ) -> None:
        """A file uploaded without ``expires_after`` never expires.

        Ref: stdapi/files/_core.py:_record_from_head
        """
        file_id, _stub = stored_file(None)

        assert await InputFile(f"file-id:{file_id}").to_bytes() == _STORED_FILE_CONTENT

    async def test_an_object_named_by_uri_keeps_its_own_lifetime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ``s3://`` input is not subject to the Files API expiry.

        Expiry belongs to the file identifier a client uploaded and can delete;
        an object the request names by URI is the caller's own reference to
        storage, and whatever metadata it carries is theirs, not a lifetime this
        server enforces on them.

        Ref: stdapi/input_file.py:_S3Source._enforce_expiry
        """
        bucket = _allowed_bucket(monkeypatch)
        stub = _StubFilesS3Client(now_utc_timestamp() - 60)
        monkeypatch.setattr(input_file, "get_client", lambda *_a, **_k: stub)

        assert await InputFile(f"s3://{bucket}/note.txt").get_size() == len(
            _STORED_FILE_CONTENT
        )


@pytest.mark.usefixtures("input_files")
class TestBedrockDocumentName:
    """A document block is named with something Bedrock accepts.

    Bedrock validates the name of a document block and refuses the whole
    request over it: the name must not be empty, must not exceed 200
    characters, and must repeat no whitespace character. The name comes from
    what the caller called the file, so none of those three is the caller's to
    guarantee -- an attachment named in a non-Latin script leaves nothing
    behind once the characters Bedrock refuses are dropped, and punctuation
    between two words leaves the spaces that surrounded it side by side.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
         stdapi/input_file.py:_bedrock_document_name
    """

    @staticmethod
    async def _name(filename: str) -> str:
        """Return the name the document block carries for *filename*.

        Args:
            filename: File name the request attached the document under.

        Returns:
            The block's ``name`` value.
        """
        file = InputFile(
            f"data:application/pdf;base64,{b64encode(b'%PDF-1.7').decode()}"
        )
        block = await file.to_bedrock_content_block(filename=filename)
        return block["document"]["name"]

    async def test_a_name_of_refused_characters_is_replaced(self) -> None:
        """A name holding nothing Bedrock allows is named for the caller.

        Sending what is left of it is sending an empty name, which Bedrock
        refuses -- so a document attached under a name written in another
        script would fail the request rather than the name.
        """
        assert await self._name("報告書.文書") == "file"

    async def test_a_name_never_repeats_a_whitespace_character(self) -> None:
        """Characters dropped from between two spaces do not leave a run of them."""
        name = await self._name("Q1 . report.pdf")

        assert "  " not in name
        assert name == "Q1 reportpdf"

    async def test_a_long_name_is_cut_to_what_bedrock_accepts(self) -> None:
        """A 200-character ceiling is applied after the name is sanitized."""
        name = await self._name(f"{'a' * 199} bcd.pdf")

        assert len(name) <= 200
        assert name == "a" * 199, "the cut leaves no trailing whitespace behind"


class TestBoundedInputConcurrency:
    """A fan-out of input reads leaves nothing running, and nothing unopened.

    One request can name many remote inputs, so they are read under a
    concurrency bound: only some run at a time and the rest wait their turn.
    The first failure answers the request, which cancels the ones still
    waiting -- and one cancelled before its turn came was never awaited at all,
    so unless it is closed it is left to the garbage collector to complain
    about, holding whatever it captured until then.

    Ref: stdapi/input_file.py:_gather_bounded
    """

    async def test_a_read_cancelled_before_its_turn_is_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every coroutine handed to the fan-out is closed once one of them fails."""
        monkeypatch.setattr(SETTINGS, "max_concurrent_input_downloads", 1)
        started: list[int] = []

        async def _read(index: int) -> int:
            """Fail the first read, once the others are queued behind it.

            Returns:
                The index of the read, for the ones that get to run.
            """
            started.append(index)
            await sleep(0)
            if index == 0:
                msg = "the first input could not be read"
                raise ApiError(msg)
            return index

        reads = [_read(index) for index in range(8)]

        with pytest.raises(BaseExceptionGroup):
            await input_file._gather_bounded(reads)  # noqa: SLF001

        assert started[0] == 0
        assert len(started) < len(reads), (
            "the reads held behind the bound never got their turn"
        )
        assert [getcoroutinestate(read) for read in reads] == [CORO_CLOSED] * 8, (
            "a coroutine cancelled before it ran is closed, not left never-awaited"
        )
