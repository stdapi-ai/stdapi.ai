"""Tests for the OpenAI-compatible ``/v1/uploads`` multipart-session route.

Broader ``/v1/uploads`` state-machine coverage (create -> add part ->
complete/cancel against S3) lives in ``tests/test_openai_files.py::TestOpenAIUploads``,
which shares the ``openai_files``-namespace fixtures with the ``/v1/files``
tests. This module covers only the ``purpose=batch`` default-expiry
resolution, offline (no AWS credentials, no S3 calls).

Ref: stdapi/routes/openai_uploads.py:create_upload_endpoint
     stdapi/routes/openai_files.py:_resolve_expires_after_seconds
"""

from typing import TYPE_CHECKING, Any

import pytest

from stdapi.files import MultipartSession
from stdapi.routes import openai_files as openai_files_routes
from stdapi.routes import openai_uploads as openai_uploads_routes

if TYPE_CHECKING:
    from starlette.testclient import TestClient

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


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
