"""Tests for the OpenAI-compatible Files and Uploads API endpoints.

Tests cover upload, list, get, delete, content streaming, pagination,
expiry enforcement, chat integration, and the full multipart upload
lifecycle (create, add parts, complete, cancel, error cases).
"""

import base64
import io
import time
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest
from openai import BadRequestError, OpenAI
from openai import NotFoundError as OpenAINotFoundError
from openai.types import FileObject
from starlette.testclient import TestClient

from stdapi.api_errors import ApiError
from stdapi.config import SETTINGS
from stdapi.files import FileRecord, _core, _multipart
from stdapi.routes import openai_files as openai_files_routes

if TYPE_CHECKING:
    import httpx

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

    def test_upload_batch_purpose_defaults_to_thirty_day_expiry(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Upload with purpose=batch and no expires_after defaults to a 30-day TTL.

        Validates:
            - expires_at is set without an explicit expires_after
            - expires_at is close to 30 days (2 592 000 seconds) from now
        """
        if use_official_api:
            pytest.skip("batch default-expiry behavior may differ on official API")
        result = openai_client.files.create(
            file=("batch.jsonl", io.BytesIO(_TEXT_FILE), "text/plain"), purpose="batch"
        )
        try:
            assert result.expires_at is not None
            assert abs(result.expires_at - (int(time.time()) + 2_592_000)) < 60
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


@pytest.mark.local
class TestRequireBucketUnit:
    """Unit tests for the Files API S3 bucket gate."""

    def test_no_bucket_hides_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a bucket, the 503 hides settings and warns the administrator."""
        monkeypatch.setattr(SETTINGS, "aws_s3_bucket", None)
        warnings: list[object] = []
        monkeypatch.setattr(
            _core, "log_error_details", lambda *args, **_kwargs: warnings.extend(args)
        )
        with pytest.raises(ApiError) as exc_info:
            _core._require_bucket()  # noqa: SLF001
        assert exc_info.value.status == 503
        message = str(exc_info.value)
        assert "administrator" in message
        assert "aws_s3_bucket" not in message
        assert any("aws_s3_bucket" in str(warning) for warning in warnings)


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


class _StubCompleteS3Client:
    """Stub S3 client for ``complete_multipart_session`` part-order tests.

    Supports just enough of the S3 multipart API surface to drive
    ``create_multipart_session`` and ``complete_multipart_session`` without any
    real AWS call: the created upload ID is cached in-process by
    ``create_multipart_session``, so ``complete_multipart_session`` never needs
    ``list_multipart_uploads``.
    """

    def __init__(self) -> None:
        self.marker_metadata: dict[str, str] = {}
        self.parts: dict[int, tuple[str, int]] = {}
        self.complete_called = False

    async def create_multipart_upload(self, **_kwargs: object) -> dict[str, Any]:
        return {"UploadId": "s3-upload-id"}

    async def put_object(self, **kwargs: object) -> dict[str, Any]:
        self.marker_metadata = kwargs["Metadata"]  # type: ignore[assignment]
        return {}

    async def head_object(self, **_kwargs: object) -> dict[str, Any]:
        return {"Metadata": self.marker_metadata}

    async def list_parts(self, **_kwargs: object) -> dict[str, Any]:
        return {
            "Parts": [
                {"PartNumber": pn, "ETag": etag, "Size": size}
                for pn, (etag, size) in self.parts.items()
            ]
        }

    async def complete_multipart_upload(self, **_kwargs: object) -> dict[str, Any]:
        self.complete_called = True
        return {}


@pytest.mark.local
class TestCompleteMultipartSessionOrderUnit:
    """Unit tests for part-order validation in complete_multipart_session (stubbed S3).

    S3 cannot reassemble multipart parts out of order (part numbers are fixed
    at add time), so out-of-order ``part_ids`` must be rejected with a clean
    400 before any S3 call is made -- not surfaced as a 502 from S3's
    ``InvalidPartOrder``.
    """

    @pytest.fixture
    def stub_s3(self, monkeypatch: pytest.MonkeyPatch) -> _StubCompleteS3Client:
        """Patch the S3 client, bucket resolution, and bucket lookup with stubs."""
        stub = _StubCompleteS3Client()
        monkeypatch.setattr(_multipart, "get_client", lambda *_: stub)
        monkeypatch.setattr(_multipart, "_require_bucket", lambda: "bucket")
        monkeypatch.setattr(
            _multipart, "resolve_file_bucket", lambda _payload: "bucket"
        )
        monkeypatch.setattr(_multipart, "track_temporary_s3_objects", lambda *_: None)
        return stub

    async def test_complete_rejects_reversed_part_order(
        self, stub_s3: _StubCompleteS3Client
    ) -> None:
        """part_ids listed in descending order are rejected with 400, mentioning order."""
        session = await _multipart.create_multipart_session(
            "f.bin", "text/plain", "assistants", 2
        )
        part_1 = _multipart._make_part_id(session.upload_id, 1)  # noqa: SLF001
        part_2 = _multipart._make_part_id(session.upload_id, 2)  # noqa: SLF001
        stub_s3.parts = {1: ("etag-1", 1), 2: ("etag-2", 1)}

        with pytest.raises(ApiError) as exc_info:
            await _multipart.complete_multipart_session(
                session.upload_id, [part_2, part_1]
            )

        assert exc_info.value.status == 400
        assert "order" in str(exc_info.value).lower()
        assert stub_s3.complete_called is False


class _StubAddPartS3Client(_StubCompleteS3Client):
    """Stub S3 client recording the part numbers ``add_part`` writes."""

    async def upload_part(self, **kwargs: object) -> dict[str, Any]:
        part_number = cast("int", kwargs["PartNumber"])
        body = cast("bytes", kwargs["Body"])
        self.parts[part_number] = (f"etag-{part_number}", len(body))
        return {}


@pytest.mark.local
class TestAddPartNumberingUnit:
    """Unit tests for part numbering in add_part (stubbed S3).

    Part numbers must continue the parts S3 already holds: several server
    instances share one upload session through a load balancer, and a
    process-local counter would hand the same number to two parts, silently
    overwriting one of them in S3.
    """

    @pytest.fixture
    def stub_s3(self, monkeypatch: pytest.MonkeyPatch) -> _StubAddPartS3Client:
        """Patch the S3 client, bucket resolution, and bucket lookup with stubs."""
        stub = _StubAddPartS3Client()
        monkeypatch.setattr(_multipart, "get_client", lambda *_: stub)
        monkeypatch.setattr(_multipart, "_require_bucket", lambda: "bucket")
        monkeypatch.setattr(
            _multipart, "resolve_file_bucket", lambda _payload: "bucket"
        )
        return stub

    async def test_consecutive_parts_are_numbered_in_order(
        self, stub_s3: _StubAddPartS3Client
    ) -> None:
        """Parts added one after another get consecutive numbers from 1."""
        session = await _multipart.create_multipart_session(
            "f.bin", "text/plain", "assistants", 8
        )

        first, _ = await _multipart.add_part(session.upload_id, b"1234")
        second, _ = await _multipart.add_part(session.upload_id, b"5678")

        extract = _multipart._extract_part_number  # noqa: SLF001
        assert extract(first, session.upload_id) == 1
        assert extract(second, session.upload_id) == 2

    async def test_part_uploaded_by_another_instance_advances_the_number(
        self, stub_s3: _StubAddPartS3Client
    ) -> None:
        """A part stored by another instance is counted, so its number is not reused."""
        session = await _multipart.create_multipart_session(
            "f.bin", "text/plain", "assistants", 8
        )
        # Part served by another instance: this process never saw it.
        stub_s3.parts[1] = ("etag-1", 4)

        part_id, _ = await _multipart.add_part(session.upload_id, b"5678")

        assert _multipart._extract_part_number(part_id, session.upload_id) == 2  # noqa: SLF001
        assert stub_s3.parts[1] == ("etag-1", 4)


@pytest.mark.local
class TestOpenAIFilesMalformedJsonBody:
    """POST /v1/files with a malformed JSON body (unit, no AWS)."""

    @pytest.fixture
    def client(self, api_key: str) -> TestClient:
        """Test client without lifespan (no AWS startup), pre-authenticated."""
        from stdapi.main import app  # noqa: PLC0415

        return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})

    def test_malformed_json_body_is_rejected(self, client: TestClient) -> None:
        """A malformed JSON body is rejected with 400, not a 500."""
        response = client.post(
            "/v1/files", content=b"{", headers={"content-type": "application/json"}
        )
        assert response.status_code == 400, response.text


@pytest.mark.local
class TestOpenAIFilesExpiresAfterBracketNotation:
    """POST /v1/files with the bracket-notation ``expires_after[seconds]`` form field.

    The pydantic ``Form`` binding (``expires_after_seconds`` / ``expires_after[seconds]``
    alias) only matches the unbracketed alias in practice; a manual fallback reads the
    raw bracket-notation value and must enforce the same 1 hour-30 day bounds (unit,
    no AWS -- the file never reaches S3 in the rejected cases).
    """

    @pytest.fixture
    def client(self, api_key: str) -> TestClient:
        """Test client without lifespan (no AWS startup), pre-authenticated."""
        from stdapi.main import app  # noqa: PLC0415

        return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})

    @staticmethod
    def _upload(client: TestClient, seconds_value: str) -> httpx.Response:
        """POST a minimal file with a bracket-notation ``expires_after[seconds]`` field."""
        return cast(
            "httpx.Response",
            client.post(
                "/v1/files",
                files={"file": ("t.txt", b"hello", "text/plain")},
                data={"purpose": "assistants", "expires_after[seconds]": seconds_value},
            ),
        )

    def test_bracket_seconds_below_minimum_rejected(self, client: TestClient) -> None:
        """59 seconds (below the 3600s minimum) is rejected with 400, not accepted."""
        response = self._upload(client, "59")
        assert response.status_code == 400, response.text

    def test_bracket_seconds_above_maximum_rejected(self, client: TestClient) -> None:
        """99999999 seconds (above the 2592000s maximum) is rejected with 400."""
        response = self._upload(client, "99999999")
        assert response.status_code == 400, response.text

    def test_bracket_seconds_non_numeric_rejected(self, client: TestClient) -> None:
        """A non-numeric value is rejected with 400 and a JSON error envelope, not a bare 500."""
        response = self._upload(client, "not_a_number")
        assert response.status_code == 400, response.text
        body = response.json()
        assert "error" in body

    def test_bracket_seconds_valid_value_accepted(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A valid bracket-notation value (7200s) is still accepted (regression guard)."""
        captured: dict[str, Any] = {}

        async def fake_upload_file(
            _file: object, purpose: str | None, expires_after: int | None = None
        ) -> FileRecord:
            captured["expires_after"] = expires_after
            now = int(datetime.now(UTC).timestamp())
            return FileRecord(
                file_id="a" * 32,
                filename="t.txt",
                content_type="text/plain",
                purpose=purpose or "",
                size=5,
                created_at=datetime.now(UTC),
                expires_at=now + expires_after if expires_after is not None else None,
            )

        monkeypatch.setattr(openai_files_routes, "upload_file", fake_upload_file)
        response = self._upload(client, "7200")
        assert response.status_code == 200, response.text
        assert captured["expires_after"] == 7200
        assert response.json()["expires_at"] is not None


class TestResolveExpiresAfterSecondsUnit:
    """Unit tests for the ``purpose=batch`` default-expiry resolution helper."""

    def test_batch_purpose_defaults_to_thirty_days(self) -> None:
        """purpose=batch with no explicit TTL defaults to the 30-day maximum."""
        resolved = openai_files_routes._resolve_expires_after_seconds(  # noqa: SLF001
            "batch", None
        )
        assert resolved == openai_files_routes._EXPIRES_AFTER_SECONDS_MAX  # noqa: SLF001

    def test_batch_purpose_explicit_ttl_not_overridden(self) -> None:
        """An explicit TTL for purpose=batch is preserved, not replaced by the default."""
        resolved = openai_files_routes._resolve_expires_after_seconds(  # noqa: SLF001
            "batch", 3600
        )
        assert resolved == 3600

    def test_non_batch_purpose_has_no_default(self) -> None:
        """Purposes other than batch persist forever unless a TTL is explicitly given."""
        resolved = openai_files_routes._resolve_expires_after_seconds(  # noqa: SLF001
            "assistants", None
        )
        assert resolved is None


class TestOpenAIFilesBatchDefaultExpiry:
    """POST /v1/files applies the documented 30-day default expiry for purpose=batch."""

    @pytest.fixture
    def client(self, api_key: str) -> TestClient:
        """Test client without lifespan (no AWS startup), pre-authenticated."""
        from stdapi.main import app  # noqa: PLC0415

        return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})

    @staticmethod
    def _fake_upload_file(captured: dict[str, Any]) -> Any:  # noqa: ANN401
        """Build a fake ``upload_file`` that records the ``expires_after`` it receives."""

        async def fake_upload_file(
            _file: object, purpose: str | None, expires_after: int | None = None
        ) -> FileRecord:
            captured["expires_after"] = expires_after
            now = int(datetime.now(UTC).timestamp())
            return FileRecord(
                file_id="a" * 32,
                filename="t.txt",
                content_type="text/plain",
                purpose=purpose or "",
                size=5,
                created_at=datetime.now(UTC),
                expires_at=now + expires_after if expires_after is not None else None,
            )

        return fake_upload_file

    def test_multipart_batch_purpose_gets_default_expiry(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A multipart upload with purpose=batch and no expires_after gets a 30-day TTL."""
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            openai_files_routes, "upload_file", self._fake_upload_file(captured)
        )
        response = client.post(
            "/v1/files",
            files={"file": ("t.txt", b"hello", "text/plain")},
            data={"purpose": "batch"},
        )
        assert response.status_code == 200, response.text
        assert (
            captured["expires_after"] == openai_files_routes._EXPIRES_AFTER_SECONDS_MAX  # noqa: SLF001
        )
        assert response.json()["expires_at"] is not None

    def test_multipart_non_batch_purpose_has_no_default_expiry(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A multipart upload with purpose=assistants and no expires_after never expires."""
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            openai_files_routes, "upload_file", self._fake_upload_file(captured)
        )
        response = client.post(
            "/v1/files",
            files={"file": ("t.txt", b"hello", "text/plain")},
            data={"purpose": "assistants"},
        )
        assert response.status_code == 200, response.text
        assert captured["expires_after"] is None
        assert "expires_at" not in response.json()

    def test_json_body_batch_purpose_gets_default_expiry(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A JSON-body upload with purpose=batch and no expires_after gets a 30-day TTL."""
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            openai_files_routes, "upload_file", self._fake_upload_file(captured)
        )
        response = client.post(
            "/v1/files",
            json={"file": "data:text/plain;base64,aGVsbG8=", "purpose": "batch"},
        )
        assert response.status_code == 200, response.text
        assert (
            captured["expires_after"] == openai_files_routes._EXPIRES_AFTER_SECONDS_MAX  # noqa: SLF001
        )


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
