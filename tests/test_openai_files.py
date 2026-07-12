"""Tests for the OpenAI-compatible Files and Uploads API endpoints.

Tests cover upload, list, get, delete, content streaming, pagination,
expiry enforcement, chat integration, and the full multipart upload
lifecycle (create, add parts, complete, cancel, error cases).
"""

import base64
import io
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from openai import BadRequestError, OpenAI
from openai import NotFoundError as OpenAINotFoundError
from openai.types import FileObject

from stdapi.files import _multipart

if TYPE_CHECKING:
    from starlette.testclient import TestClient

#: Minimal valid PDF bytes for testing document endpoints.
_MINIMAL_PDF: bytes = (
    b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
    b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
    b"0000000058 00000 n \n0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
)

#: Simple plain-text file bytes for chat integration tests.
_TEXT_FILE: bytes = b"The capital of France is Paris."

#: Minimum part size enforced by S3 for all parts except the last (5 MiB).
_MIN_PART_SIZE: int = 5 * 1024 * 1024

#: First part — must be at or above the S3 minimum non-last part size.
_PART_A: bytes = b"A" * _MIN_PART_SIZE

#: Second (last) part — may be any size.
_PART_B: bytes = b"B" * 1024


class TestOpenAIFiles:
    """Test suite for the OpenAI-compatible /v1/files endpoints."""

    # --- Upload ---

    def test_upload_returns_file_object(self, openai_client: OpenAI) -> None:
        """Upload a small file and assert all required fields are present with correct types.

        Validates:
            - Response is a FileObject with all mandatory fields
            - id starts with 'file-'
            - object is 'file'
            - bytes equals the uploaded file size
            - created_at is a positive integer (Unix timestamp)
            - status is 'processed'
        """
        content = _MINIMAL_PDF
        result = openai_client.files.create(
            file=("test.pdf", io.BytesIO(content), "application/pdf"),
            purpose="assistants",
        )
        try:
            assert isinstance(result, FileObject)
            assert result.id.startswith("file-")
            assert result.object == "file"
            assert result.bytes == len(content)
            assert isinstance(result.created_at, int)
            assert result.created_at > 0
            assert result.filename == "test.pdf"
            assert result.purpose == "assistants"
            assert result.status == "processed"
        finally:
            openai_client.files.delete(result.id)

    def test_upload_purpose_echoed(self, openai_client: OpenAI) -> None:
        """Upload with purpose=user_data and assert the returned purpose matches.

        Validates:
            - purpose field in response matches the uploaded purpose
        """
        result = openai_client.files.create(
            file=("data.txt", io.BytesIO(_TEXT_FILE), "text/plain"), purpose="user_data"
        )
        try:
            assert result.purpose == "user_data"
        finally:
            openai_client.files.delete(result.id)

    def test_upload_with_expires_after(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Upload with expires_after and assert expires_at is set in the future.

        Validates:
            - expires_at is set
            - expires_at is greater than the current time
        """
        if use_official_api:
            pytest.skip("expires_after behavior may differ on official API")
        result = openai_client.files.create(
            file=("expire.txt", io.BytesIO(_TEXT_FILE), "text/plain"),
            purpose="assistants",
            extra_body={"expires_after": {"anchor": "created_at", "seconds": 3600}},
        )
        try:
            assert result.expires_at is not None
            assert result.expires_at > int(time.time())
        finally:
            openai_client.files.delete(result.id)

    # --- Get metadata ---

    def test_get_metadata(self, openai_client: OpenAI) -> None:
        """Upload a file then retrieve it by ID and assert metadata matches.

        Validates:
            - Retrieved file id matches
            - bytes, filename, purpose are consistent with upload
        """
        uploaded = openai_client.files.create(
            file=("meta.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf"),
            purpose="assistants",
        )
        try:
            retrieved = openai_client.files.retrieve(uploaded.id)
            assert retrieved.id == uploaded.id
            assert retrieved.bytes == uploaded.bytes
            assert retrieved.filename == uploaded.filename
            assert retrieved.purpose == uploaded.purpose
        finally:
            openai_client.files.delete(uploaded.id)

    # --- List ---

    def test_list_files(self, openai_client: OpenAI) -> None:
        """Upload two files and assert both appear in GET /files.

        Validates:
            - Both uploaded file IDs appear in the list response
            - has_more is a bool
        """
        f1 = openai_client.files.create(
            file=("list1.txt", io.BytesIO(_TEXT_FILE), "text/plain"),
            purpose="assistants",
        )
        f2 = openai_client.files.create(
            file=("list2.txt", io.BytesIO(_TEXT_FILE), "text/plain"),
            purpose="assistants",
        )
        try:
            files = list(openai_client.files.list(limit=100))
            ids = {f.id for f in files}
            assert f1.id in ids
            assert f2.id in ids
        finally:
            openai_client.files.delete(f1.id)
            openai_client.files.delete(f2.id)

    def test_list_order_desc(self, openai_client: OpenAI) -> None:
        """Assert that the default list order is descending (newest first).

        Validates:
            - First file in list has created_at >= last file's created_at
        """
        f1 = openai_client.files.create(
            file=("ord1.txt", io.BytesIO(_TEXT_FILE), "text/plain"),
            purpose="assistants",
        )
        f2 = openai_client.files.create(
            file=("ord2.txt", io.BytesIO(_TEXT_FILE), "text/plain"),
            purpose="assistants",
        )
        try:
            files = list(openai_client.files.list(limit=10))
            assert len(files) >= 2
            # Descending: newer files appear first
            assert files[0].created_at >= files[-1].created_at
        finally:
            openai_client.files.delete(f1.id)
            openai_client.files.delete(f2.id)

    def test_list_order_asc(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Assert that order=asc returns oldest first.

        Validates:
            - First file in list has created_at <= last file's created_at
        """
        if use_official_api:
            pytest.skip(
                "files uploaded within the same second share created_at; order not guaranteed on official API"
            )
        f1 = openai_client.files.create(
            file=("asc1.txt", io.BytesIO(_TEXT_FILE), "text/plain"),
            purpose="assistants",
        )
        f2 = openai_client.files.create(
            file=("asc2.txt", io.BytesIO(_TEXT_FILE), "text/plain"),
            purpose="assistants",
        )
        try:
            files = list(openai_client.files.list(order="asc", limit=10))
            assert len(files) >= 2
            assert files[0].created_at <= files[-1].created_at
        finally:
            openai_client.files.delete(f1.id)
            openai_client.files.delete(f2.id)

    def test_list_cursor_pagination(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Upload several files and paginate with 'after' cursor; assert no overlap.

        Validates:
            - Cursor excludes the anchor file
            - Files created after the cursor appear on the next page
        """
        if use_official_api:
            pytest.skip(
                "official API cursor uses creation-time order; files uploaded in the same second have indeterminate relative position"
            )
        uploaded = [
            openai_client.files.create(
                file=(f"page{i}.txt", io.BytesIO(_TEXT_FILE), "text/plain"),
                purpose="assistants",
            )
            for i in range(3)
        ]
        try:
            # Sort by ID (UUIDv7 — lexicographic order == creation order)
            uploaded.sort(key=lambda f: f.id)
            # Use the first uploaded file as cursor; the other two must appear after it
            after_page = openai_client.files.list(
                order="asc", limit=100, after=uploaded[0].id
            ).data
            ids_after = {f.id for f in after_page}
            assert uploaded[0].id not in ids_after
            assert uploaded[1].id in ids_after
            assert uploaded[2].id in ids_after
        finally:
            for f in uploaded:
                openai_client.files.delete(f.id)

    def test_list_purpose_filter(self, openai_client: OpenAI) -> None:
        """Upload files with different purposes and filter by purpose.

        Validates:
            - Only files matching the purpose are returned
        """
        fa = openai_client.files.create(
            file=("pf_a.txt", io.BytesIO(_TEXT_FILE), "text/plain"),
            purpose="assistants",
        )
        fb = openai_client.files.create(
            file=("pf_b.txt", io.BytesIO(_TEXT_FILE), "text/plain"), purpose="user_data"
        )
        try:
            files = list(openai_client.files.list(purpose="user_data", limit=100))
            ids = {f.id for f in files}
            assert fb.id in ids
            assert fa.id not in ids
        finally:
            openai_client.files.delete(fa.id)
            openai_client.files.delete(fb.id)

    # --- Delete ---

    def test_delete(self, openai_client: OpenAI) -> None:
        """Upload then delete a file; assert 404 on subsequent get.

        Validates:
            - Delete response has deleted=True
            - Follow-up retrieve raises 404
        """
        f = openai_client.files.create(
            file=("del.txt", io.BytesIO(_TEXT_FILE), "text/plain"), purpose="assistants"
        )
        result = openai_client.files.delete(f.id)
        assert result.deleted is True
        with pytest.raises(OpenAINotFoundError):
            openai_client.files.retrieve(f.id)

    # --- Content ---

    def test_download_content(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Upload bytes and download via /content; assert byte equality.

        Validates:
            - Downloaded content matches uploaded content exactly
        """
        if use_official_api:
            pytest.skip("Official OpenAI API restricts file downloads by purpose")
        content = b"Hello, Files API content download test!"
        f = openai_client.files.create(
            file=("content.txt", io.BytesIO(content), "text/plain"),
            purpose="assistants",
        )
        try:
            downloaded = openai_client.files.content(f.id).content
            assert downloaded == content
        finally:
            openai_client.files.delete(f.id)

    # --- Error cases ---

    def test_not_found(self, openai_client: OpenAI) -> None:
        """Assert 404 error format for get/delete of non-existent file.

        Validates:
            - NotFoundError is raised for both retrieve and delete
        """
        fake_id = "file-" + "a" * 32
        with pytest.raises(OpenAINotFoundError):
            openai_client.files.retrieve(fake_id)
        with pytest.raises(OpenAINotFoundError):
            openai_client.files.delete(fake_id)

    def test_expired_file_returns_404(
        self, openai_client: OpenAI, test_client: TestClient | None
    ) -> None:
        """Upload with short expires_after; mock time past expiry; assert 404.

        Validates:
            - Expired files return 404 as if they don't exist
        """
        if test_client is None:
            pytest.skip("requires local time control")
        f = openai_client.files.create(
            file=("exp.txt", io.BytesIO(_TEXT_FILE), "text/plain"),
            purpose="assistants",
            extra_body={"expires_after": {"anchor": "created_at", "seconds": 3600}},
        )
        try:
            future_ts = int(time.time()) + 7200  # 2 hours in the future
            with patch("stdapi.files._core.now_utc_timestamp") as mock_now:
                mock_now.return_value = future_ts
                with pytest.raises(OpenAINotFoundError):
                    openai_client.files.retrieve(f.id)
        finally:
            with suppress(OpenAINotFoundError):
                # File may already be gone via background deletion triggered by the expired retrieve
                openai_client.files.delete(f.id)

    # --- Chat integration ---

    def test_file_in_chat_completion(
        self, openai_client: OpenAI, chat_vision_model: str, use_official_api: bool
    ) -> None:
        """Upload a PDF file and reference it in a chat completion message.

        Validates:
            - The API accepts a file reference in a chat completion message
            - Response contains assistant content (local server only; official API may return empty)
        """
        f = openai_client.files.create(
            file=("doc.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf"),
            purpose="assistants",
        )
        try:
            response = openai_client.chat.completions.create(
                model=chat_vision_model,
                max_completion_tokens=50,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this document in one sentence.",
                            },
                            {"type": "file", "file": {"file_id": f.id}},
                        ],
                    }
                ],
            )
            assert len(response.choices) > 0
            if not use_official_api:
                content = response.choices[0].message.content
                assert content is not None
                assert len(content) > 0
        finally:
            openai_client.files.delete(f.id)

    def test_file_id_uri_scheme_in_chat_image_url(
        self,
        openai_client: OpenAI,
        chat_vision_model: str,
        sample_image_file: bytes,
        use_official_api: bool,
    ) -> None:
        """Reference an uploaded file via the ``file-id:`` URI scheme.

        Uploads a small PNG via the Files API, then passes
        ``file-id:<file-id>`` as the ``image_url.url`` (string-overloaded
        field) of a chat completion content part.  This exercises the
        project-local ``file-id:`` resolver end-to-end without any
        monkey-patching.

        Validates:
            - The string-overloaded ``image_url.url`` accepts ``file-id:``.
            - The chat completion returns a non-empty assistant message
              when run against the local server.
        """
        if use_official_api:
            pytest.skip("`file-id:` is a project-local URI scheme")
        uploaded = openai_client.files.create(
            file=("image.png", io.BytesIO(sample_image_file), "image/png"),
            purpose="vision",
        )
        try:
            response = openai_client.chat.completions.create(
                model=chat_vision_model,
                max_completion_tokens=50,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image briefly."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"file-id:{uploaded.id}"},
                            },
                        ],
                    }
                ],
            )
            assert len(response.choices) > 0
            content = response.choices[0].message.content
            assert content is not None
            assert len(content) > 0
        finally:
            openai_client.files.delete(uploaded.id)


class _StubMultipartS3Client:
    """Stub S3 client capturing create_multipart_upload/put_object kwargs."""

    def __init__(self) -> None:
        self.create_kwargs: dict[str, Any] = {}

    async def create_multipart_upload(self, **kwargs: object) -> dict[str, Any]:
        self.create_kwargs = kwargs
        return {"UploadId": "s3-upload-id"}

    async def put_object(self, **kwargs: object) -> dict[str, Any]:
        return {}


@pytest.mark.local
class TestCreateMultipartSessionUnit:
    """Unit tests for expiry stamping in create_multipart_session (stubbed S3)."""

    @pytest.fixture
    def stub_s3(self, monkeypatch: pytest.MonkeyPatch) -> _StubMultipartS3Client:
        """Patch the S3 client and bucket resolution with stubs."""
        stub = _StubMultipartS3Client()
        monkeypatch.setattr(_multipart, "get_client", lambda *_: stub)
        monkeypatch.setattr(_multipart, "_require_bucket", lambda: "bucket")
        return stub

    async def test_expires_after_stamps_metadata_and_tag(
        self, stub_s3: _StubMultipartS3Client
    ) -> None:
        """expires_after sets the expires-at metadata and the Lifecycle expiry tag."""
        session = await _multipart.create_multipart_session(
            "f.bin", "a/b", "assistants", 1, 3600
        )
        metadata = stub_s3.create_kwargs["Metadata"]
        assert metadata["expires-at"] == str(session.created_at + 3600)
        assert "stdapi-ai.expires=true" in stub_s3.create_kwargs["Tagging"]

    async def test_no_expiry_leaves_metadata_empty(
        self, stub_s3: _StubMultipartS3Client
    ) -> None:
        """Without expires_after the metadata stays empty and no expiry tag is set."""
        await _multipart.create_multipart_session("f.bin", "a/b", "assistants", 1)
        assert stub_s3.create_kwargs["Metadata"]["expires-at"] == ""
        assert "stdapi-ai.expires" not in stub_s3.create_kwargs["Tagging"]


class TestOpenAIUploads:
    """Test suite for the OpenAI-compatible /v1/uploads endpoints."""

    # --- Create ---

    def test_create_returns_upload_object(self, openai_client: OpenAI) -> None:
        """Creating an upload returns a pending Upload object with correct fields."""
        total = len(_PART_A) + len(_PART_B)
        upload = openai_client.uploads.create(
            bytes=total,
            filename="test.txt",
            mime_type="text/plain",
            purpose="assistants",
        )
        try:
            assert upload.object == "upload"
            assert upload.status == "pending"
            assert upload.bytes == total
            assert upload.filename == "test.txt"
            assert upload.purpose == "assistants"
            assert upload.id.startswith("upload_")
            assert upload.expires_at > upload.created_at
        finally:
            openai_client.uploads.cancel(upload.id)

    def test_create_with_expires_after_file_expires(
        self, openai_client: OpenAI
    ) -> None:
        """The file assembled from an upload with expires_after carries expires_at."""
        upload = openai_client.uploads.create(
            bytes=len(_PART_A),
            filename="expire_upload.bin",
            mime_type="application/octet-stream",
            purpose="assistants",
            expires_after={"anchor": "created_at", "seconds": 3600},
        )
        part = openai_client.uploads.parts.create(
            upload_id=upload.id, data=io.BytesIO(_PART_A)
        )
        completed = openai_client.uploads.complete(
            upload_id=upload.id, part_ids=[part.id]
        )
        try:
            assert completed.file is not None
            assert completed.file.expires_at is not None
            assert completed.file.expires_at > int(time.time())
        finally:
            assert completed.file is not None
            openai_client.files.delete(completed.file.id)

    def test_create_expires_after_out_of_range_rejected(
        self, openai_client: OpenAI
    ) -> None:
        """expires_after.seconds below 1 hour is rejected with a validation error."""
        with pytest.raises(BadRequestError, match="seconds"):
            openai_client.uploads.create(
                bytes=1024,
                filename="bad_expiry.bin",
                mime_type="application/octet-stream",
                purpose="assistants",
                expires_after={"anchor": "created_at", "seconds": 60},
            )

    def test_create_expires_after_above_maximum_rejected(
        self, openai_client: OpenAI
    ) -> None:
        """expires_after.seconds above 30 days is rejected with a validation error."""
        with pytest.raises(BadRequestError, match="seconds"):
            openai_client.uploads.create(
                bytes=1024,
                filename="bad_expiry_max.bin",
                mime_type="application/octet-stream",
                purpose="assistants",
                expires_after={"anchor": "created_at", "seconds": 2_592_001},
            )

    def test_create_expires_after_unsupported_anchor_rejected(
        self, openai_client: OpenAI
    ) -> None:
        """expires_after.anchor other than 'created_at' is rejected with a validation error."""
        with pytest.raises(BadRequestError, match="anchor"):
            openai_client.uploads.create(
                bytes=1024,
                filename="bad_anchor.bin",
                mime_type="application/octet-stream",
                purpose="assistants",
                expires_after={"anchor": "updated_at", "seconds": 3600},  # type: ignore[arg-type]
            )

    # --- Add parts ---

    def test_add_part_returns_upload_part(self, openai_client: OpenAI) -> None:
        """Adding a part returns an UploadPart with the correct upload_id."""
        upload = openai_client.uploads.create(
            bytes=len(_PART_A),
            filename="part_test.bin",
            mime_type="application/octet-stream",
            purpose="assistants",
        )
        try:
            part = openai_client.uploads.parts.create(
                upload_id=upload.id, data=io.BytesIO(_PART_A)
            )
            assert part.object == "upload.part"
            assert part.upload_id == upload.id
            assert part.id.startswith("part_")
        finally:
            openai_client.uploads.cancel(upload.id)

    # --- Complete ---

    def test_complete_produces_file(self, openai_client: OpenAI) -> None:
        """Completing an upload returns a completed Upload with an embedded FileObject."""
        content = _PART_A + _PART_B
        upload = openai_client.uploads.create(
            bytes=len(content),
            filename="complete_test.txt",
            mime_type="text/plain",
            purpose="assistants",
        )
        part_a = openai_client.uploads.parts.create(
            upload_id=upload.id, data=io.BytesIO(_PART_A)
        )
        part_b = openai_client.uploads.parts.create(
            upload_id=upload.id, data=io.BytesIO(_PART_B)
        )
        completed = openai_client.uploads.complete(
            upload_id=upload.id, part_ids=[part_a.id, part_b.id]
        )
        try:
            assert completed.status == "completed"
            assert completed.file is not None
            assert completed.file.bytes == len(content)
            assert completed.file.filename == "complete_test.txt"
            assert completed.file.purpose == "assistants"
        finally:
            assert completed.file is not None
            openai_client.files.delete(completed.file.id)

    def test_complete_file_downloadable(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """The file produced by a completed upload can be downloaded."""
        if use_official_api:
            pytest.skip("Official OpenAI API restricts file downloads by purpose")
        payload = b"Multipart content for download test."
        upload = openai_client.uploads.create(
            bytes=len(payload),
            filename="download_test.txt",
            mime_type="text/plain",
            purpose="assistants",
        )
        part = openai_client.uploads.parts.create(
            upload_id=upload.id, data=io.BytesIO(payload)
        )
        completed = openai_client.uploads.complete(
            upload_id=upload.id, part_ids=[part.id]
        )
        try:
            assert completed.file is not None
            downloaded = openai_client.files.content(completed.file.id).content
            assert downloaded == payload
        finally:
            assert completed.file is not None
            openai_client.files.delete(completed.file.id)

    def test_complete_wrong_size_rejected(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Completing with parts whose total size mismatches the declared bytes is rejected."""
        if use_official_api:
            pytest.skip(
                "official API rejects at add_part time (not complete time) when part exceeds declared bytes"
            )
        upload = openai_client.uploads.create(
            bytes=999999,  # wrong: doesn't match part sizes
            filename="size_mismatch.bin",
            mime_type="application/octet-stream",
            purpose="assistants",
        )
        part = openai_client.uploads.parts.create(
            upload_id=upload.id, data=io.BytesIO(_PART_A)
        )
        with pytest.raises(BadRequestError):
            openai_client.uploads.complete(upload_id=upload.id, part_ids=[part.id])
        openai_client.uploads.cancel(upload.id)

    def test_complete_unknown_part_id_rejected(self, openai_client: OpenAI) -> None:
        """Completing with a part ID that was never added is rejected."""
        upload = openai_client.uploads.create(
            bytes=len(_PART_A),
            filename="bad_part.bin",
            mime_type="application/octet-stream",
            purpose="assistants",
        )
        try:
            with pytest.raises(BadRequestError):
                openai_client.uploads.complete(
                    upload_id=upload.id, part_ids=["part_" + "a" * 32]
                )
        finally:
            openai_client.uploads.cancel(upload.id)

    # --- Cancel ---

    def test_cancel_sets_status_cancelled(self, openai_client: OpenAI) -> None:
        """Cancelling an upload returns an Upload with status 'cancelled'."""
        upload = openai_client.uploads.create(
            bytes=len(_PART_A),
            filename="cancel_test.bin",
            mime_type="application/octet-stream",
            purpose="assistants",
        )
        cancelled = openai_client.uploads.cancel(upload.id)
        assert cancelled.status == "cancelled"

    def test_cancel_prevents_further_parts(self, openai_client: OpenAI) -> None:
        """Adding parts to a cancelled upload is rejected (400 or 404 depending on marker cleanup timing)."""
        upload = openai_client.uploads.create(
            bytes=len(_PART_A),
            filename="cancel_parts.bin",
            mime_type="application/octet-stream",
            purpose="assistants",
        )
        openai_client.uploads.cancel(upload.id)
        with pytest.raises((BadRequestError, OpenAINotFoundError)):
            openai_client.uploads.parts.create(
                upload_id=upload.id, data=io.BytesIO(_PART_A)
            )

    # --- Error cases ---

    def test_not_found_upload(self, openai_client: OpenAI) -> None:
        """Referencing a non-existent upload ID returns 404."""
        fake_id = "upload_" + "a" * 32
        with pytest.raises(OpenAINotFoundError):
            openai_client.uploads.cancel(fake_id)

    def test_not_found_add_part(self, openai_client: OpenAI) -> None:
        """Adding a part to a non-existent upload ID returns 404."""
        fake_id = "upload_" + "a" * 32
        with pytest.raises(OpenAINotFoundError):
            openai_client.uploads.parts.create(
                upload_id=fake_id, data=io.BytesIO(_PART_A)
            )


class TestOpenAIUploadsJsonBody:
    """Tests for POST /v1/uploads/{upload_id}/parts using an application/json body.

    Verifies that the JSON body path (base64, data URI) is accepted and allows
    MCP agents to perform multipart uploads end-to-end without multipart/form-data.
    """

    @pytest.fixture(autouse=True)
    def _skip_on_official_api(self, use_official_api: bool) -> None:
        """JSON body input is an extension not supported by the official API."""
        if use_official_api:
            pytest.skip("JSON body input not supported by the official OpenAI API")

    def test_json_body_missing_data_returns_400(self, openai_client: OpenAI) -> None:
        """JSON body without the required data field returns 400."""
        http_client = openai_client._client  # noqa: SLF001
        upload = openai_client.uploads.create(
            bytes=len(_PART_A),
            filename="json_missing.bin",
            mime_type="application/octet-stream",
            purpose="assistants",
        )
        try:
            response = http_client.post(
                f"{openai_client.base_url}uploads/{upload.id}/parts",
                json={},
                headers={"Authorization": f"Bearer {openai_client.api_key}"},
            )
            assert response.status_code == 400
        finally:
            openai_client.uploads.cancel(upload.id)

    def test_json_body_part_upload_with_data_uri(self, openai_client: OpenAI) -> None:
        """Full multipart upload workflow using JSON body for the parts endpoint."""
        http_client = openai_client._client  # noqa: SLF001
        data_uri = (
            f"data:application/octet-stream;base64,{base64.b64encode(_PART_A).decode()}"
        )
        upload = openai_client.uploads.create(
            bytes=len(_PART_A),
            filename="json_part.bin",
            mime_type="application/octet-stream",
            purpose="assistants",
        )
        try:
            response = http_client.post(
                f"{openai_client.base_url}uploads/{upload.id}/parts",
                json={"data": data_uri},
                headers={"Authorization": f"Bearer {openai_client.api_key}"},
            )
            assert response.status_code == 200
            part = response.json()
            assert part["object"] == "upload.part"
            assert part["upload_id"] == upload.id
            assert part["id"].startswith("part_")

            completed = openai_client.uploads.complete(
                upload_id=upload.id, part_ids=[part["id"]]
            )
            assert completed.status == "completed"
            assert completed.file is not None
        finally:
            with suppress(OpenAINotFoundError, BadRequestError):
                openai_client.uploads.cancel(upload.id)


class TestOpenAIFilesJsonBody:
    """Tests for POST /v1/files using an application/json body.

    Verifies that the JSON body path (base64, data URI, URL) is accepted and
    behaves identically to the multipart form upload.
    """

    @pytest.fixture(autouse=True)
    def _skip_on_official_api(self, use_official_api: bool) -> None:
        """JSON body input is an extension not supported by the official API."""
        if use_official_api:
            pytest.skip("JSON body input not supported by the official OpenAI API")

    def test_json_body_missing_file_returns_400(self, openai_client: OpenAI) -> None:
        """JSON body without the required file field returns 400."""
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}files",
            json={"purpose": "user_data"},
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 400

    def test_json_body_upload_with_data_uri(self, openai_client: OpenAI) -> None:
        """Upload via JSON body with a data URI creates a FileObject."""
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}files",
            json={
                "file": "data:text/plain;base64,SGVsbG8gV29ybGQ=",
                "purpose": "user_data",
            },
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["id"].startswith("file-")
        assert body["object"] == "file"
        assert body["purpose"] == "user_data"
        openai_client.files.delete(body["id"])

    def test_json_body_upload_with_raw_base64(self, openai_client: OpenAI) -> None:
        """Upload via JSON body with a raw base64 string creates a FileObject."""
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}files",
            json={"file": "SGVsbG8gV29ybGQ=", "purpose": "user_data"},
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["id"].startswith("file-")
        openai_client.files.delete(body["id"])
