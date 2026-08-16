"""Tests for the OpenAI-compatible ``/v1/uploads`` multipart-session route.

Broader ``/v1/uploads`` state-machine coverage (create -> add part ->
complete/cancel against S3) lives in ``tests/test_openai_files.py::TestOpenAIUploads``,
which shares the ``openai_files``-namespace fixtures with the ``/v1/files``
tests. This module covers the ``purpose=batch`` default-expiry resolution, the
bounded per-process session cache, the completion checksum, and the JSON-body
part route's remote-source handling. Everything is offline (no AWS credentials,
no S3 calls, no network) except ``TestCompleteUploadChecksumOnS3``, which runs
the checksum against a real two-part upload in the sandbox bucket.

Ref: stdapi/routes/openai_uploads.py:create_upload_endpoint
     stdapi/routes/openai_uploads.py:add_upload_part
     stdapi/routes/openai_files.py:_resolve_expires_after_seconds
"""

import asyncio
import io
from contextlib import suppress
from datetime import UTC, datetime
from hashlib import md5
from typing import TYPE_CHECKING, Any, Self

import pytest
from aiobotocore.session import get_session
from botocore.exceptions import ClientError
from openai import APIStatusError, BadRequestError
from openai import NotFoundError as OpenAINotFoundError

from stdapi import input_file
from stdapi.aws_s3 import BUCKET_TO_REGION
from stdapi.config import SETTINGS
from stdapi.files import MultipartSession, _multipart
from stdapi.files._core import file_id_s3_key, resolve_file_bucket
from stdapi.routes import openai_files as openai_files_routes
from stdapi.routes import openai_uploads as openai_uploads_routes

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    # Starlette types TestClient against httpx2; these helpers only ever carry its
    # responses, so they are typed from the same module it returns them from.
    from httpx2 import Response
    from openai import OpenAI
    from starlette.testclient import TestClient
    from types_aiobotocore_s3.type_defs import HeadObjectOutputTypeDef

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Payload the stubbed HTTPS origin serves for the remote-source part tests.
_REMOTE_PART_BYTES = b"remote part payload"

#: Part ID the stubbed ``add_part`` returns; must match ``PART_ID_PATTERN``.
_STUB_PART_ID = f"part_{'a' * 32}"

#: Upload ID the part route is called with; must match ``UPLOAD_ID_PATTERN``.
_STUB_UPLOAD_ID = f"upload_{'b' * 32}"


class TestCreateUploadBatchDefaultExpiry:
    """POST /v1/uploads applies the documented 30-day default expiry for purpose=batch.

    ``create_multipart_session`` is stubbed, so these tests pin the TTL the
    route computes rather than S3 behavior.

    Ref: stdapi/routes/openai_uploads.py:create_upload_endpoint
         stdapi/routes/openai_files.py:_resolve_expires_after_seconds
    """

    @staticmethod
    def _fake_create_multipart_session(captured: dict[str, Any]) -> Any:  # noqa: ANN401
        """Build a fake ``create_multipart_session`` that records the ``expires_after`` it receives."""

        async def fake_create_multipart_session(
            filename: str,
            mime_type: str,
            purpose: str,
            total_bytes: int,
            expires_after: int | None = None,
        ) -> MultipartSession:
            captured["expires_after"] = expires_after
            return MultipartSession(
                upload_id=f"upload_{'a' * 32}",
                file_id="a" * 32,
                s3_bucket="bucket",
                s3_key=f"uploads/{'a' * 32}",
                filename=filename,
                mime_type=mime_type,
                purpose=purpose,
                total_bytes=total_bytes,
                expires_at=0,
                created_at=0,
            )

        return fake_create_multipart_session

    def test_batch_purpose_gets_default_expiry(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """purpose=batch with no expires_after gets the 30-day (2 592 000 s) default TTL.

        Ref: stdapi/routes/openai_uploads.py:create_upload_endpoint
        """
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            openai_uploads_routes,
            "create_multipart_session",
            self._fake_create_multipart_session(captured),
        )
        response = app_client.post(
            "/v1/uploads",
            json={
                "filename": "batch.jsonl",
                "mime_type": "text/plain",
                "purpose": "batch",
                "bytes": 10,
            },
        )
        assert response.status_code == 200, response.text
        assert (
            captured["expires_after"] == openai_files_routes._EXPIRES_AFTER_SECONDS_MAX  # noqa: SLF001
        )
        assert captured["expires_after"] == 2_592_000

    def test_explicit_expiry_is_respected(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit ``expires_after`` for purpose=batch is preserved, not replaced.

        Ref: stdapi/routes/openai_uploads.py:create_upload_endpoint
        """
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            openai_uploads_routes,
            "create_multipart_session",
            self._fake_create_multipart_session(captured),
        )
        response = app_client.post(
            "/v1/uploads",
            json={
                "filename": "batch.jsonl",
                "mime_type": "text/plain",
                "purpose": "batch",
                "bytes": 10,
                "expires_after": {"anchor": "created_at", "seconds": 3600},
            },
        )
        assert response.status_code == 200, response.text
        assert captured["expires_after"] == 3600

    def test_non_batch_purpose_has_no_default_expiry(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Purposes other than batch never expire unless a TTL is explicitly given.

        Ref: stdapi/routes/openai_uploads.py:create_upload_endpoint
        """
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            openai_uploads_routes,
            "create_multipart_session",
            self._fake_create_multipart_session(captured),
        )
        response = app_client.post(
            "/v1/uploads",
            json={
                "filename": "t.txt",
                "mime_type": "text/plain",
                "purpose": "assistants",
                "bytes": 10,
            },
        )
        assert response.status_code == 200, response.text
        assert captured["expires_after"] is None


class _StubCreateMultipartS3Client:
    """Minimal S3 client stand-in for ``create_multipart_session``."""

    def __init__(self) -> None:
        self.created = 0

    async def create_multipart_upload(self, **_kwargs: object) -> dict[str, Any]:
        self.created += 1
        return {"UploadId": f"s3-upload-id-{self.created}"}

    async def put_object(self, **_kwargs: object) -> dict[str, Any]:
        return {}


class TestMultipartSessionCacheBound:
    """The per-process multipart session cache stays bounded (unit, stubbed S3).

    ``POST /v1/uploads`` costs the caller a JSON body and no bytes, and a session
    abandoned without a completion or cancellation is by definition never looked
    up again — so an unbounded cache would retain its entry, S3 upload ID
    included, for the whole life of the process.

    Ref: https://developers.openai.com/api/reference/resources/uploads
         stdapi/files/_multipart.py:_cache_set
    """

    @pytest.fixture
    def cache(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Isolate the session cache, shrink its bound, and stub S3 away.

        Returns:
            The cache dict the multipart module reads and writes.
        """
        cache: dict[str, Any] = {}
        stub = _StubCreateMultipartS3Client()
        monkeypatch.setattr(_multipart, "_cache", cache)
        monkeypatch.setattr(_multipart, "_CACHE_MAX", 4)
        monkeypatch.setattr(_multipart, "get_client", lambda *_: stub)
        monkeypatch.setattr(_multipart, "_require_bucket", lambda: "bucket")
        return cache

    @staticmethod
    async def _create() -> str:
        """Create a session and return its upload ID.

        Returns:
            The new session's upload ID.
        """
        session = await _multipart.create_multipart_session(
            "f.bin", "text/plain", "assistants", 1
        )
        return session.upload_id

    async def test_abandoned_sessions_never_grow_the_cache(
        self, cache: dict[str, Any]
    ) -> None:
        """Sessions created and left pending stop accumulating at the bound.

        Ref: stdapi/files/_multipart.py:_cache_set
        """
        upload_ids = [await self._create() for _ in range(12)]

        assert len(cache) == 4
        assert len(set(upload_ids)) == 12, "each session must get its own upload ID"

    async def test_a_reused_session_outlives_the_idle_ones(
        self, cache: dict[str, Any]
    ) -> None:
        """Eviction drops the least recently used entry, not the oldest still in use.

        A session whose parts are still arriving would otherwise be evicted by
        newer idle sessions, costing every remaining part a ``ListMultipartUploads``
        call to rebuild what the cache already held.

        Ref: stdapi/files/_multipart.py:_cache_get
        """
        upload_ids = [await self._create() for _ in range(4)]
        assert _multipart._cache_get(upload_ids[0]) is not None  # noqa: SLF001

        for _ in range(3):
            await self._create()

        assert _multipart._cache_get(upload_ids[0]) is not None, (  # noqa: SLF001
            "the session used most recently must survive the eviction pressure"
        )
        assert _multipart._cache_get(upload_ids[1]) is None  # noqa: SLF001
        assert len(cache) == 4


class _StubHttpResponse:
    """Minimal aiohttp response stand-in serving a fixed body."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(body)),
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


class TestAddUploadPartJsonBodyRemoteSources:
    """POST /v1/uploads/{id}/parts resolves a remote ``data`` reference before storing.

    The JSON body accepts the same reference forms as ``/v1/files`` (base64, data
    URI, HTTPS URL, ``s3://`` URI), but unlike ``/v1/files`` the part route reads
    the reference itself: a URL that stayed an opaque string would be stored as
    the literal ASCII of the URL instead of the file it names, and an unlisted
    bucket would be read with the gateway's own credentials.

    Ref: https://stdapi.ai/api_openai_files/
         stdapi/routes/openai_uploads.py:add_upload_part
         stdapi/types/openai_uploads.py:AddUploadPartJsonBody
    """

    @staticmethod
    def _record_add_part(chunks: list[bytes]) -> Any:  # noqa: ANN401
        """Build a fake ``add_part`` recording the chunk instead of reaching S3.

        Args:
            chunks: List each call appends its chunk to.

        Returns:
            A replacement for ``stdapi.routes.openai_uploads.add_part``.
        """

        async def fake_add_part(_upload_id: str, chunk: bytes) -> tuple[str, int]:
            chunks.append(chunk)
            return _STUB_PART_ID, 0

        return fake_add_part

    def test_https_url_body_is_downloaded_into_the_part(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An HTTPS ``data`` value is fetched and its body becomes the part content.

        The recorded chunk must be the origin's bytes, not the URL text, and the
        gateway must actually issue the ``GET`` rather than treat the string as
        base64. The size cap is disabled so the fetch stays a single ``GET``: with
        a cap set the source probes the metadata with a ``HEAD`` first.

        Ref: stdapi/input_file.py:_HttpSource._read
             stdapi/config.py:_Settings.max_input_file_size
        """
        monkeypatch.setattr(SETTINGS, "max_input_file_size", 0)
        session = _StubHttpSession(_StubHttpResponse(_REMOTE_PART_BYTES))
        monkeypatch.setattr(
            input_file._HttpSource,  # noqa: SLF001
            "_client_session",
            lambda _self, _extra_headers=None: session,
        )
        chunks: list[bytes] = []
        monkeypatch.setattr(
            openai_uploads_routes, "add_part", self._record_add_part(chunks)
        )

        response = app_client.post(
            f"/v1/uploads/{_STUB_UPLOAD_ID}/parts",
            json={"data": "https://example.com/chunk.bin?signature=secret"},
        )

        assert response.status_code == 200, response.text
        assert response.json()["id"] == _STUB_PART_ID
        assert chunks == [_REMOTE_PART_BYTES]
        assert session.requests == [
            "GET https://example.com/chunk.bin?signature=secret"
        ]

    def test_s3_uri_outside_the_allowlist_is_rejected(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ``s3://`` ``data`` value naming an unlisted bucket is refused with a 400.

        This is the SSRF guard for S3 references on the part route: without it any
        caller could have the gateway read an arbitrary bucket with the server's
        own credentials and store the result as an upload part.

        Ref: stdapi/input_file.py:InputFile._normalize_and_detect_origin
        """
        chunks: list[bytes] = []
        monkeypatch.setattr(
            openai_uploads_routes, "add_part", self._record_add_part(chunks)
        )

        response = app_client.post(
            f"/v1/uploads/{_STUB_UPLOAD_ID}/parts",
            json={"data": "s3://an-unconfigured-external-bucket-xyz/chunk.bin"},
        )

        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "an-unconfigured-external-bucket-xyz" in error["message"], error
        assert chunks == []


class _StubStreamingBody:
    """Minimal S3 ``StreamingBody`` stand-in yielding a fixed payload in chunks."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    async def iter_chunks(self, chunk_size: int) -> AsyncIterator[bytes]:
        """Yield the payload in *chunk_size*-byte chunks."""
        for start in range(0, len(self.payload), chunk_size):
            yield self.payload[start : start + chunk_size]


class _StubMultipartS3Client:
    """In-memory S3 multipart stand-in that really assembles the parts it is given.

    Faithful on the two points the checksum tests depend on: parts are stored
    and concatenated in the order ``CompleteMultipartUpload`` lists them, and
    each part's ``ETag`` is that part's own MD5 (never the whole file's).
    """

    def __init__(self) -> None:
        self.parts: dict[int, bytes] = {}
        self.assembled: bytes | None = None
        self.deleted: list[str] = []
        self.reads = 0
        self.total_bytes = 0
        self.filename = "f.bin"

    async def create_multipart_upload(self, **_kwargs: object) -> dict[str, Any]:
        """Open the session."""
        return {"UploadId": "s3-upload-id"}

    async def put_object(self, **kwargs: object) -> dict[str, Any]:
        """Record the declared total size and filename from the session marker."""
        metadata: dict[str, str] = kwargs["Metadata"]  # type: ignore[assignment]
        self.total_bytes = int(metadata["total-bytes"])
        self.filename = metadata["filename"]
        return {}

    async def list_multipart_uploads(self, **kwargs: object) -> dict[str, Any]:
        """Report the single in-progress upload."""
        return {"Uploads": [{"Key": kwargs["Prefix"], "UploadId": "s3-upload-id"}]}

    @staticmethod
    def _etag(data: bytes) -> str:
        """Return the entity tag S3 gives a single uploaded part: its own MD5.

        Returns:
            The quoted hex digest.
        """
        return f'"{md5(data, usedforsecurity=False).hexdigest()}"'

    async def list_parts(self, **_kwargs: object) -> dict[str, Any]:
        """Report every stored part with its entity tag and size."""
        return {
            "Parts": [
                {"PartNumber": number, "ETag": self._etag(data), "Size": len(data)}
                for number, data in sorted(self.parts.items())
            ]
        }

    async def upload_part(self, **kwargs: object) -> dict[str, Any]:
        """Store one part and report its entity tag."""
        number: int = kwargs["PartNumber"]  # type: ignore[assignment]
        body: bytes = kwargs["Body"]  # type: ignore[assignment]
        self.parts[number] = data = bytes(body)
        return {"ETag": self._etag(data)}

    async def complete_multipart_upload(self, **kwargs: object) -> dict[str, Any]:
        """Concatenate the listed parts in the order given."""
        listed: dict[str, list[dict[str, Any]]] = kwargs["MultipartUpload"]  # type: ignore[assignment]
        self.assembled = b"".join(
            self.parts[part["PartNumber"]] for part in listed["Parts"]
        )
        return {}

    async def head_object(self, **kwargs: object) -> dict[str, Any]:
        """Answer for the session marker or for the assembled object."""
        key: str = kwargs["Key"]  # type: ignore[assignment]
        if key.startswith(SETTINGS.aws_s3_tmp_prefix):
            return {
                "Metadata": {
                    "filename": self.filename,
                    "mime-type": "text/plain",
                    "purpose": "assistants",
                    "total-bytes": str(self.total_bytes),
                }
            }
        return {
            "ContentLength": len(self.assembled or b""),
            "ContentType": "text/plain",
            "LastModified": datetime(2026, 1, 1, tzinfo=UTC),
            "Metadata": {"purpose": "assistants"},
        }

    async def get_object(self, **_kwargs: object) -> dict[str, Any]:
        """Stream the assembled object back."""
        self.reads += 1
        return {"Body": _StubStreamingBody(self.assembled or b"")}

    async def delete_object(self, **kwargs: object) -> dict[str, Any]:
        """Record a deletion."""
        key: str = kwargs["Key"]  # type: ignore[assignment]
        self.deleted.append(key)
        return {}


class TestCompleteUploadChecksum:
    """POST /v1/uploads/{id}/complete verifies the ``md5`` the client declares.

    ``md5`` is the digest of the **file contents** — the parts concatenated in
    ``part_ids`` order — so the multi-part case is what pins the combination
    rule: neither the per-part digests nor S3's own multipart ``ETag`` (an MD5
    of the concatenated part MD5s) is the value a client sends.

    Ref: https://github.com/openai/openai-openapi/blob/master/openapi.yaml
         (``CompleteUploadRequest.md5``: "The optional md5 checksum for the file
         contents to verify if the bytes uploaded matches what you expect.")
         stdapi/files/_multipart.py:complete_multipart_session
    """

    #: Parts of a two-part upload; the first is short because the 5 MiB floor is relaxed.
    _PARTS = (b"the first part of the file, ", b"and the second part of it.")

    @pytest.fixture
    def s3(self, monkeypatch: pytest.MonkeyPatch) -> _StubMultipartS3Client:
        """Point the multipart module at an in-memory S3 and empty its caches.

        Returns:
            The stub S3 client the routes drive.
        """
        stub = _StubMultipartS3Client()
        monkeypatch.setattr(SETTINGS, "aws_s3_bucket", "bucket")
        monkeypatch.setattr(_multipart, "get_client", lambda *_a, **_k: stub)
        monkeypatch.setattr(_multipart, "track_temporary_s3_objects", lambda *_a: None)
        monkeypatch.setattr(_multipart, "_cache", {})
        monkeypatch.setattr(_multipart, "_MIN_PART_SIZE", 1)
        return stub

    @staticmethod
    def _upload_parts(
        client: TestClient, parts: tuple[bytes, ...]
    ) -> tuple[str, list[str]]:
        """Create a session and upload *parts* through the public routes.

        Returns:
            ``(upload_id, part_ids)``.
        """
        created = client.post(
            "/v1/uploads",
            json={
                "filename": "f.bin",
                "mime_type": "text/plain",
                "purpose": "assistants",
                "bytes": sum(len(part) for part in parts),
            },
        )
        assert created.status_code == 200, created.text
        upload_id = created.json()["id"]
        part_ids = []
        for part in parts:
            added = client.post(
                f"/v1/uploads/{upload_id}/parts", files={"data": ("chunk", part)}
            )
            assert added.status_code == 200, added.text
            part_ids.append(added.json()["id"])
        return upload_id, part_ids

    @classmethod
    def _complete(
        cls, client: TestClient, parts: tuple[bytes, ...], checksum: str | None
    ) -> Response:
        """Run a whole upload and return the raw completion response.

        Returns:
            The ``POST /v1/uploads/{id}/complete`` response.
        """
        upload_id, part_ids = cls._upload_parts(client, parts)
        body: dict[str, Any] = {"part_ids": part_ids}
        if checksum is not None:
            body["md5"] = checksum
        return client.post(f"/v1/uploads/{upload_id}/complete", json=body)

    @staticmethod
    def _assert_refused(response: Response) -> None:
        """Assert the completion was refused as a checksum mismatch."""
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "md5" in error["message"], error
        assert "checksum" in error["message"], error

    def test_matching_checksum_completes_the_upload(
        self, app_client: TestClient, s3: _StubMultipartS3Client
    ) -> None:
        """A single-part upload whose declared md5 matches its bytes is accepted.

        The assembled object is never read back: the digest was accumulated as
        the part was proxied, so verifying it costs no extra storage traffic.

        Ref: stdapi/files/_multipart.py:complete_multipart_session
        """
        payload = b"the whole file, in one part"
        response = self._complete(
            app_client, (payload,), md5(payload, usedforsecurity=False).hexdigest()
        )

        assert response.status_code == 200, response.text
        assert response.json()["status"] == "completed"
        assert s3.assembled == payload
        assert s3.reads == 0, "a proxied upload must not be re-read to be verified"

    def test_uppercase_hex_digest_is_accepted(
        self, app_client: TestClient, s3: _StubMultipartS3Client
    ) -> None:
        """The digest comparison is case-insensitive, as hex encoding is.

        Ref: stdapi/types/openai_uploads.py:CompleteUploadBody
        """
        payload = b"case does not change a digest"
        response = self._complete(
            app_client,
            (payload,),
            md5(payload, usedforsecurity=False).hexdigest().upper(),
        )

        assert response.status_code == 200, response.text
        assert s3.assembled == payload

    def test_mismatched_checksum_is_refused_before_the_file_exists(
        self, app_client: TestClient, s3: _StubMultipartS3Client
    ) -> None:
        """A wrong md5 is a 400, and no file is produced from the mismatching bytes.

        Ref: stdapi/files/_multipart.py:complete_multipart_session
        """
        response = self._complete(
            app_client,
            (b"the bytes that were actually uploaded",),
            md5(
                b"the bytes the client thought it sent", usedforsecurity=False
            ).hexdigest(),
        )

        self._assert_refused(response)
        assert s3.assembled is None, "a refused upload must not produce a file"

    def test_multi_part_checksum_covers_the_concatenation(
        self, app_client: TestClient, s3: _StubMultipartS3Client
    ) -> None:
        """Across several parts the md5 is taken over the assembled file, not per part.

        Ref: stdapi/files/_multipart.py:complete_multipart_session
        """
        response = self._complete(
            app_client,
            self._PARTS,
            md5(b"".join(self._PARTS), usedforsecurity=False).hexdigest(),
        )

        assert response.status_code == 200, response.text
        assert s3.assembled == b"".join(self._PARTS)
        assert s3.reads == 0

    def test_the_s3_multipart_etag_is_not_the_declared_checksum(
        self, app_client: TestClient, s3: _StubMultipartS3Client
    ) -> None:
        """An MD5 of the concatenated part digests is refused, not mistaken for a match.

        This is the dimension a storage-layer answer would have supplied: S3's
        multipart ``ETag`` is the MD5 of the part MD5s, which no client computes.

        Ref: stdapi/files/_multipart.py:complete_multipart_session
        """
        etag_dimension = md5(
            b"".join(md5(part, usedforsecurity=False).digest() for part in self._PARTS),
            usedforsecurity=False,
        ).hexdigest()

        self._assert_refused(self._complete(app_client, self._PARTS, etag_dimension))
        assert s3.assembled is None

    def test_no_checksum_completes_without_reading_anything_back(
        self, app_client: TestClient, s3: _StubMultipartS3Client
    ) -> None:
        """``md5`` stays optional: omitting it verifies nothing and costs nothing.

        Ref: https://github.com/openai/openai-openapi/blob/master/openapi.yaml
             (``CompleteUploadRequest`` requires only ``part_ids``)
        """
        response = self._complete(app_client, self._PARTS, None)

        assert response.status_code == 200, response.text
        assert s3.assembled == b"".join(self._PARTS)
        assert s3.reads == 0

    def test_a_checksum_that_is_not_a_digest_is_rejected(
        self, app_client: TestClient, s3: _StubMultipartS3Client
    ) -> None:
        """A value that cannot be an MD5 digest fails validation rather than the comparison.

        Ref: stdapi/types/openai_uploads.py:CompleteUploadBody
        """
        response = self._complete(app_client, (b"a short file",), "not-a-digest")

        assert response.status_code == 400, response.text
        assert response.json()["error"]["type"] == "invalid_request_error"
        assert s3.assembled is None

    def test_parts_proxied_elsewhere_are_verified_against_stored_bytes(
        self,
        app_client: TestClient,
        s3: _StubMultipartS3Client,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With no running digest for the session, the stored object answers instead.

        Parts of one upload can be served by different instances, and dropping
        the whole check in that case would report an unverified upload as
        verified.

        Ref: stdapi/files/_multipart.py:_object_md5
        """
        upload_id, part_ids = self._upload_parts(app_client, self._PARTS)
        monkeypatch.setattr(_multipart, "_cache", {})

        response = app_client.post(
            f"/v1/uploads/{upload_id}/complete",
            json={
                "part_ids": part_ids,
                "md5": md5(b"".join(self._PARTS), usedforsecurity=False).hexdigest(),
            },
        )

        assert response.status_code == 200, response.text
        assert s3.reads == 1, "the stored object is the only remaining source of truth"
        assert s3.deleted == []

    def test_a_mismatch_found_after_assembly_removes_the_file(
        self,
        app_client: TestClient,
        s3: _StubMultipartS3Client,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A mismatch detected from stored bytes deletes the object it just assembled.

        Otherwise the refused upload would leave a file behind at an identifier
        the caller can already derive, and read back as if it had succeeded.

        Ref: stdapi/files/_multipart.py:complete_multipart_session
        """
        upload_id, part_ids = self._upload_parts(app_client, self._PARTS)
        monkeypatch.setattr(_multipart, "_cache", {})

        response = app_client.post(
            f"/v1/uploads/{upload_id}/complete",
            json={
                "part_ids": part_ids,
                "md5": md5(
                    b"other contents entirely", usedforsecurity=False
                ).hexdigest(),
            },
        )

        self._assert_refused(response)
        assert s3.reads == 1
        assert s3.deleted == [file_id_s3_key(upload_id[7:])]

    def test_a_part_replaced_after_proxying_does_not_pass_on_the_running_digest(
        self, app_client: TestClient, s3: _StubMultipartS3Client
    ) -> None:
        """Bytes stored under a part number this process already hashed are re-verified.

        Two instances can pick the same part number for one session, so the
        running digest is trusted only while every part still carries the entity
        tag it had when it was proxied. Here the stored bytes are swapped for
        others of the same length, which the size check cannot see.

        Ref: stdapi/files/_multipart.py:_fold_part
        """
        upload_id, part_ids = self._upload_parts(app_client, self._PARTS)
        s3.parts[2] = b"?" * len(self._PARTS[1])

        response = app_client.post(
            f"/v1/uploads/{upload_id}/complete",
            json={
                "part_ids": part_ids,
                "md5": md5(b"".join(self._PARTS), usedforsecurity=False).hexdigest(),
            },
        )

        self._assert_refused(response)
        assert s3.reads == 1, "the stored bytes must be the ones verified"


#: Minimum size S3 enforces on every part of a multipart upload except the last.
_S3_MIN_PART_SIZE: int = 5 * 1024 * 1024

#: A genuine two-part upload: only the last part may sit under the S3 floor.
_LIVE_PARTS: tuple[bytes, ...] = (b"A" * _S3_MIN_PART_SIZE, b"and the trailing part.")


def _multipart_etag_digest(parts: tuple[bytes, ...]) -> str:
    """Return the digest S3 itself puts in a multipart object's entity tag.

    Taken over the concatenated *binary* part digests rather than over the file,
    so it answers a different question from the checksum a client declares.

    Args:
        parts: The parts, in the order S3 assembles them.

    Returns:
        Lowercase hex digest, without the ``-<part count>`` suffix S3 appends.
    """
    return md5(
        b"".join(md5(part, usedforsecurity=False).digest() for part in parts),
        usedforsecurity=False,
    ).hexdigest()


def _stored_object(payload: str) -> HeadObjectOutputTypeDef | None:
    """Return what S3 holds for a file payload, asking S3 rather than the gateway.

    Uses its own client on its own loop: the pooled clients belong to the loop
    the app lifespan runs in.

    Args:
        payload: Bare 32-char file payload.

    Returns:
        The ``head_object`` response, or ``None`` when no such object exists.
    """
    bucket = resolve_file_bucket(payload)
    key = file_id_s3_key(payload)

    async def head() -> HeadObjectOutputTypeDef | None:
        """Head the object, mapping an absent key to ``None``.

        Returns:
            The ``head_object`` response, or ``None``.

        Raises:
            ClientError: Any failure other than the object not existing.
        """
        async with get_session().create_client(
            "s3", region_name=BUCKET_TO_REGION[bucket]
        ) as s3:
            try:
                return await s3.head_object(Bucket=bucket, Key=key)
            except ClientError as error:
                if error.response["Error"]["Code"] in {"404", "NoSuchKey"}:
                    return None
                raise

    return asyncio.run(head())


@pytest.mark.slow
class TestCompleteUploadChecksumOnS3:
    """The declared ``md5`` checked against a real two-part upload on S3.

    A stub can only confirm the gateway hashes what it hashed; what it cannot
    produce is the value the storage layer computes for the same object. S3
    gives a multipart object an entity tag in another dimension entirely -- an
    MD5 over the concatenated part digests, suffixed with the part count -- and
    sending that instead of the file's own digest is the mistake this check
    exists to catch. Two parts are the fewest that tell the two apart, so the
    first sits at the 5 MiB floor S3 enforces on every part but the last.

    Slow rather than expensive: the bytes cost storage for the seconds they
    exist, and nothing here calls a model.

    Ref: https://developers.openai.com/api/reference/resources/uploads
         https://docs.aws.amazon.com/AmazonS3/latest/userguide/checking-object-integrity.html
         stdapi/files/_multipart.py:complete_multipart_session
    """

    @staticmethod
    def _upload(client: OpenAI, parts: tuple[bytes, ...]) -> tuple[str, list[str]]:
        """Create a session and send *parts* through the public routes.

        Args:
            client: Client bound to the gateway.
            parts: Part payloads, in order.

        Returns:
            ``(upload_id, part_ids)``.
        """
        created = client.uploads.create(
            bytes=sum(len(part) for part in parts),
            filename="checksum_multipart.bin",
            mime_type="application/octet-stream",
            purpose="assistants",
        )
        return created.id, [
            client.uploads.parts.create(upload_id=created.id, data=io.BytesIO(part)).id
            for part in parts
        ]

    @staticmethod
    def _assert_refused(raised: pytest.ExceptionInfo[BadRequestError]) -> None:
        """Assert the completion was refused as a checksum mismatch, not for another reason."""
        error = raised.value.body
        assert isinstance(error, dict), error
        assert raised.value.status_code == 400, error
        assert error["type"] == "invalid_request_error", error
        assert "md5" in str(error["message"]), error
        assert "checksum" in str(error["message"]), error

    @staticmethod
    def _assert_nothing_left_behind(client: OpenAI, payload: str) -> None:
        """Assert the refused upload produced no file, in the API and in the bucket.

        Args:
            client: Client bound to the gateway.
            payload: Bare file payload the refused upload would have produced.
        """
        assert _stored_object(payload) is None, "the refused bytes are still stored"
        with pytest.raises(OpenAINotFoundError):
            client.files.retrieve(f"file-{payload}")

    def test_a_two_part_upload_with_a_matching_checksum_completes(
        self, openai_client: OpenAI
    ) -> None:
        """A real multipart upload whose md5 covers the assembled file is accepted.

        The entity tag S3 gives that very object is read back alongside: it
        carries the part count and a digest of the part digests, so it is not
        the value the client declared and never could be.

        Ref: stdapi/files/_multipart.py:complete_multipart_session
        """
        upload_id, part_ids = self._upload(openai_client, _LIVE_PARTS)
        contents = b"".join(_LIVE_PARTS)
        checksum = md5(contents, usedforsecurity=False).hexdigest()

        completed = openai_client.uploads.complete(
            upload_id=upload_id, part_ids=part_ids, md5=checksum
        )

        assert completed.file is not None, completed
        try:
            assert completed.status == "completed", completed
            assert completed.file.bytes == len(contents), completed.file

            stored = _stored_object(upload_id.removeprefix("upload_"))
            assert stored is not None, "the completed upload stored no object"
            etag = str(stored["ETag"]).strip('"')
            assert (
                etag == f"{_multipart_etag_digest(_LIVE_PARTS)}-{len(_LIVE_PARTS)}"
            ), f"S3 reported an unexpected multipart entity tag: {etag}"
            assert etag != checksum, "the two digests must not be interchangeable"
        finally:
            openai_client.files.delete(completed.file.id)

    def test_the_entity_tag_s3_computes_is_refused_as_the_checksum(
        self, openai_client: OpenAI
    ) -> None:
        """S3's own digest for the object is not the file's, and is refused as one.

        The dimension a storage-layer answer would have supplied: a client that
        reads the entity tag back and declares it has verified nothing.

        Ref: stdapi/files/_multipart.py:complete_multipart_session
        """
        upload_id, part_ids = self._upload(openai_client, _LIVE_PARTS)

        try:
            with pytest.raises(BadRequestError) as raised:
                openai_client.uploads.complete(
                    upload_id=upload_id,
                    part_ids=part_ids,
                    md5=_multipart_etag_digest(_LIVE_PARTS),
                )

            self._assert_refused(raised)
            self._assert_nothing_left_behind(
                openai_client, upload_id.removeprefix("upload_")
            )
        finally:
            # The session is still pending: a refusal before assembly leaves the
            # parts S3 holds to be aborted rather than to expire.
            with suppress(APIStatusError):
                openai_client.uploads.cancel(upload_id)

    def test_parts_this_process_never_hashed_are_verified_from_the_stored_bytes(
        self, openai_client: OpenAI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no running digest the assembled object is read back, and matches.

        Emptying the session cache is what another instance having served the
        parts leaves behind: no digest for them here. The check then has to hash
        what S3 assembled, over a file larger than one read.

        Ref: stdapi/files/_multipart.py:_object_md5
        """
        upload_id, part_ids = self._upload(openai_client, _LIVE_PARTS)
        monkeypatch.setattr(_multipart, "_cache", {})

        completed = openai_client.uploads.complete(
            upload_id=upload_id,
            part_ids=part_ids,
            md5=md5(b"".join(_LIVE_PARTS), usedforsecurity=False).hexdigest(),
        )

        assert completed.file is not None, completed
        try:
            assert completed.status == "completed", completed
            assert completed.file.bytes == len(b"".join(_LIVE_PARTS)), completed.file
        finally:
            openai_client.files.delete(completed.file.id)

    def test_a_mismatch_found_after_assembly_leaves_nothing_behind(
        self, openai_client: OpenAI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wrong md5 caught from the stored bytes deletes the object it assembled.

        This is the only path where the file briefly exists, so it is the only
        one where a refusal can leave one behind -- at an identifier the caller
        already knows, and can read back as though the upload had succeeded.

        Ref: stdapi/files/_multipart.py:complete_multipart_session
        """
        upload_id, part_ids = self._upload(openai_client, _LIVE_PARTS)
        monkeypatch.setattr(_multipart, "_cache", {})

        with pytest.raises(BadRequestError) as raised:
            openai_client.uploads.complete(
                upload_id=upload_id,
                part_ids=part_ids,
                md5=md5(
                    b"contents that were never uploaded", usedforsecurity=False
                ).hexdigest(),
            )

        self._assert_refused(raised)
        self._assert_nothing_left_behind(
            openai_client, upload_id.removeprefix("upload_")
        )
