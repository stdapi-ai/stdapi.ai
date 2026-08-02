"""Tests for the OpenAI-compatible ``/v1/uploads`` multipart-session route.

Broader ``/v1/uploads`` state-machine coverage (create -> add part ->
complete/cancel against S3) lives in ``tests/test_openai_files.py::TestOpenAIUploads``,
which shares the ``openai_files``-namespace fixtures with the ``/v1/files``
tests. This module covers the ``purpose=batch`` default-expiry resolution and
the JSON-body part route's remote-source handling, offline (no AWS
credentials, no S3 calls, no network).

Ref: stdapi/routes/openai_uploads.py:create_upload_endpoint
     stdapi/routes/openai_uploads.py:add_upload_part
     stdapi/routes/openai_files.py:_resolve_expires_after_seconds
"""

from typing import TYPE_CHECKING, Any, Self

import pytest

from stdapi import input_file
from stdapi.config import SETTINGS
from stdapi.files import MultipartSession
from stdapi.routes import openai_files as openai_files_routes
from stdapi.routes import openai_uploads as openai_uploads_routes

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.testclient import TestClient

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
