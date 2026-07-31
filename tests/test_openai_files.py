"""Tests for the OpenAI-compatible ``/v1/files`` and ``/v1/uploads`` routes backed by S3.

Files are plain S3 objects with no external database: the 32-char ID payload is
``base32hex(uuid7_bytes + crc32(bucket))``, so IDs sort by creation time (which
the listing order and the ``after`` cursor rely on) and any instance resolves
any ID without shared state.

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://stdapi.ai/api_openai_files/
     stdapi/routes/openai_files.py
     stdapi/routes/openai_uploads.py
     stdapi/files/_core.py
"""

import base64
import io
import time
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest
from openai import APIStatusError, BadRequestError, OpenAI
from openai import NotFoundError as OpenAINotFoundError
from openai.types import FileObject

from stdapi.api_errors import ApiError
from stdapi.aws_s3 import S3Object
from stdapi.config import SETTINGS
from stdapi.files import FileRecord, _core, _multipart
from stdapi.routes import openai_files as openai_files_routes

if TYPE_CHECKING:
    import httpx
    from anthropic import Anthropic
    from starlette.testclient import TestClient

    from stdapi.input_file import InputFile

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
#: The Files API is one namespace shared by the whole account, and these tests
#: create, list and delete in it. Without a group, ``--dist=loadgroup`` spreads even
#: a single module's tests across workers, so they would race against each other:
#: a listing assertion can see files another test is midway through deleting.
pytestmark = pytest.mark.xdist_group("openai_files")

_TEXT_FILE: bytes = b"The capital of France is Paris."

#: Minimum part size enforced by S3 for all parts except the last (5 MiB).
_MIN_PART_SIZE: int = 5 * 1024 * 1024

#: First part — must be at or above the S3 minimum non-last part size.
_PART_A: bytes = b"A" * _MIN_PART_SIZE

#: Second (last) part — may be any size.
_PART_B: bytes = b"B" * 1024


def _error_envelope(error: APIStatusError, status: int) -> dict[str, Any]:
    """Return the inner ``error`` object of *error* after checking status and type.

    On the OpenAI client ``APIStatusError.body`` is already the inner envelope
    (the client stores ``body.get("error", body)``), and both OpenAI and the
    gateway report every 400/404 as ``invalid_request_error``.

    Ref: https://developers.openai.com/api/docs/guides/error-codes
         stdapi/api_providers/openai.py:_format_error
    """
    assert error.status_code == status
    body = error.body
    assert isinstance(body, dict), f"expected a JSON error envelope, got {body!r}"
    assert body["type"] == "invalid_request_error", body
    return body


class TestOpenAIFiles:
    """OpenAI ``/v1/files`` upload, retrieve, list, delete and content contract.

    Ref: https://developers.openai.com/api/reference/resources/files
         stdapi/routes/openai_files.py:upload
         stdapi/files/_core.py:upload_file
    """

    # --- Upload ---

    def test_upload_returns_file_object(self, openai_client: OpenAI) -> None:
        """A small upload returns a File object carrying every required field.

        ``OpenAIFile`` requires id/object/bytes/created_at/filename/purpose/status.
        ``status`` is deprecated upstream but still mandatory, and S3 has no
        equivalent notion, so the gateway always reports ``processed``.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_files.py:_to_file_object
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
        """``purpose=user_data`` is stored with the object, not just echoed back.

        The response projection falls back to ``user_data`` when the stored
        purpose is missing, so the echoed field alone cannot distinguish "stored"
        from "defaulted"; the purpose-filtered listing reads the raw record and
        does.

        Ref: https://developers.openai.com/api/reference/resources/files
             stdapi/files/_core.py:upload_file
             stdapi/routes/openai_files.py:_to_file_object
        """
        result = openai_client.files.create(
            file=("data.txt", io.BytesIO(_TEXT_FILE), "text/plain"), purpose="user_data"
        )
        try:
            assert result.purpose == "user_data"
            listed = openai_client.files.list(purpose="user_data", limit=100)
            assert result.id in {f.id for f in listed.data}, (
                "purpose must be persisted with the stored object, not defaulted"
            )
        finally:
            openai_client.files.delete(result.id)

    @pytest.mark.gateway(
        "cross-API file sharing (Anthropic upload, OpenAI listing) is a gateway feature"
    )
    def test_purposeless_upload_lists_as_user_data(
        self, openai_client: OpenAI, anthropic_client: Anthropic
    ) -> None:
        """A file uploaded with no purpose concept lists consistently as ``user_data``.

        The Anthropic Files API has no ``purpose`` field, and its files share
        storage with the OpenAI Files API (docs/api_anthropic_files.md). The
        gateway stores the ``user_data`` default at write time rather than only
        at response time, so such a file both displays as ``user_data`` *and*
        matches the ``purpose=user_data`` list filter -- it does not silently
        drop out of the filtered listing the way a response-only default would.

        Ref: docs/api_anthropic_files.md#configuration
             stdapi/files/_core.py:upload_file
        """
        created = anthropic_client.beta.files.upload(
            file=("data.txt", io.BytesIO(_TEXT_FILE), "text/plain")
        )
        try:
            retrieved = openai_client.files.retrieve(created.id)
            assert retrieved.purpose == "user_data"
            listed = openai_client.files.list(purpose="user_data", limit=100)
            # Each surface prefixes the same 32-char payload its own way --
            # "file_" for Anthropic, "file-" for OpenAI -- so compare payloads.
            payload = created.id.split("_", 1)[1]
            assert payload in {f.id.split("-", 1)[1] for f in listed.data}, (
                "a purposeless file must match the purpose=user_data list filter "
                "the same way its displayed purpose does"
            )
        finally:
            openai_client.files.delete(created.id)

    @pytest.mark.gateway("expires_after behavior may differ on official API")
    def test_upload_with_expires_after(self, openai_client: OpenAI) -> None:
        """``expires_after`` anchors ``expires_at`` at ``created_at`` plus the requested TTL.

        The gateway stores the absolute expiry in S3 user metadata and tags the
        object for Lifecycle cleanup; expiry itself is enforced in code on read.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/files/_core.py:upload_file
        """
        result = openai_client.files.create(
            file=("expire.txt", io.BytesIO(_TEXT_FILE), "text/plain"),
            purpose="assistants",
            extra_body={"expires_after": {"anchor": "created_at", "seconds": 3600}},
        )
        try:
            assert result.expires_at is not None
            assert result.expires_at > int(time.time())
            assert abs(result.expires_at - (result.created_at + 3600)) <= 60, (
                f"expires_at {result.expires_at} is not created_at "
                f"{result.created_at} plus the requested 3600s TTL"
            )
        finally:
            openai_client.files.delete(result.id)

    @pytest.mark.gateway("batch default-expiry behavior may differ on official API")
    def test_upload_batch_purpose_defaults_to_thirty_day_expiry(
        self, openai_client: OpenAI
    ) -> None:
        """``purpose=batch`` with no ``expires_after`` defaults to a 30-day TTL.

        30 days (2 592 000 s) is also the maximum accepted ``expires_after``
        value, so the default is the upper bound rather than an arbitrary window.

        Ref: https://stdapi.ai/api_openai_files/
             stdapi/routes/openai_files.py:_resolve_expires_after_seconds
        """
        result = openai_client.files.create(
            file=("batch.jsonl", io.BytesIO(_TEXT_FILE), "text/plain"), purpose="batch"
        )
        try:
            assert result.purpose == "batch"
            assert result.expires_at is not None
            assert abs(result.expires_at - (int(time.time()) + 2_592_000)) < 60
        finally:
            openai_client.files.delete(result.id)

    # --- Get metadata ---

    def test_get_metadata(self, openai_client: OpenAI) -> None:
        """Retrieving a file by ID returns the same metadata the upload reported.

        The gateway rebuilds the record from S3 ``HeadObject`` on every call, so
        the filename (read back from ``Content-Disposition``), size, purpose and
        creation time must survive the round trip unchanged.

        Ref: https://developers.openai.com/api/reference/resources/files
             stdapi/files/_core.py:_record_from_head
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
            assert retrieved.created_at == uploaded.created_at
        finally:
            openai_client.files.delete(uploaded.id)

    # --- List ---

    def test_list_files(self, openai_client: OpenAI) -> None:
        """``GET /files`` returns a paginated envelope containing both uploaded files.

        The listing is assembled from ``ListObjectsV2`` pages across every
        configured bucket, so ``has_more`` is always reported alongside the data.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_files.py:list_files_endpoint
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
            page = openai_client.files.list(limit=100)
            assert page.has_more is not None, "the page must report has_more"
            ids = {f.id for f in page.data}
            assert f1.id in ids
            assert f2.id in ids
        finally:
            openai_client.files.delete(f1.id)
            openai_client.files.delete(f2.id)

    def test_list_order_desc(self, openai_client: OpenAI) -> None:
        """The default list order is newest first.

        ``order`` defaults to ``desc`` and is defined on ``created_at``. Sorting by
        file ID would be a gateway-only shortcut: its IDs carry a UUIDv7 prefix, so
        lexicographic order happens to match creation order, but OpenAI's file IDs
        are random and never sort. The gateway's ``created_at`` is the S3
        ``LastModified`` and only second-granular, so two files created in the same
        second can tie or invert; the assertion is therefore that the page is
        non-increasing to the second, plus the relative position of the two files
        this test created, which is exact on both targets.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/files/_core.py:list_files
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
            # .data is the first page. Iterating the page object instead would
            # auto-paginate the whole account, and ordering only has to hold within
            # one server-side snapshot, not across pages fetched seconds apart.
            files = openai_client.files.list(limit=10).data
            assert len(files) >= 2
            created = [f.created_at for f in files]
            assert created == sorted(created, reverse=True), (
                f"the default page must be newest-first by created_at, got {created}"
            )
            # Exact even when the two share a created_at second.
            ids = [f.id for f in files]
            assert ids.index(f2.id) < ids.index(f1.id)
        finally:
            openai_client.files.delete(f1.id)
            openai_client.files.delete(f2.id)

    @pytest.mark.gateway(
        "files uploaded within the same second share created_at; order not guaranteed on official API"
    )
    def test_list_order_asc(self, openai_client: OpenAI) -> None:
        """``order=asc`` returns the oldest files first.

        Ascending order without a purpose filter is the fast path: it pages S3
        keys directly instead of scanning every key, so it is worth pinning
        separately from the descending default.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/files/_core.py:list_files
        """
        f1 = openai_client.files.create(
            file=("asc1.txt", io.BytesIO(_TEXT_FILE), "text/plain"),
            purpose="assistants",
        )
        f2 = openai_client.files.create(
            file=("asc2.txt", io.BytesIO(_TEXT_FILE), "text/plain"),
            purpose="assistants",
        )
        try:
            # First page only — see test_list_order_desc on why not to auto-paginate.
            files = openai_client.files.list(order="asc", limit=10).data
            assert len(files) >= 2
            timestamps = [f.created_at for f in files]
            assert timestamps == sorted(timestamps), (
                f"order=asc must return oldest-first, got {timestamps}"
            )
        finally:
            openai_client.files.delete(f1.id)
            openai_client.files.delete(f2.id)

    @pytest.mark.gateway(
        "official API cursor uses creation-time order; files uploaded in the same second have indeterminate relative position"
    )
    def test_list_cursor_pagination(self, openai_client: OpenAI) -> None:
        """The ``after`` cursor excludes the anchor and returns only later IDs.

        The cursor is passed to S3 as ``StartAfter`` on the object key, so
        "after" is exact and exclusive rather than timestamp-based — files
        created within the same second still page deterministically.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/files/_core.py:list_files
        """
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
            assert all(f.id > uploaded[0].id for f in after_page), (
                "every file on the page must sort strictly after the cursor"
            )
        finally:
            for f in uploaded:
                openai_client.files.delete(f.id)

    def test_list_purpose_filter(self, openai_client: OpenAI) -> None:
        """``purpose`` filters the listing to exactly the matching files.

        S3 keys carry no purpose, so filtering forces a ``HeadObject`` fan-out
        over every key and an exact match on the stored metadata.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/files/_core.py:list_files
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
            assert all(f.purpose == "user_data" for f in files), (
                "the filter must apply to every returned file"
            )
        finally:
            openai_client.files.delete(fa.id)
            openai_client.files.delete(fb.id)

    # --- Delete ---

    def test_delete(self, openai_client: OpenAI) -> None:
        """Deleting a file confirms the deletion and leaves the ID unresolvable.

        Ref: https://developers.openai.com/api/reference/resources/files
             stdapi/routes/openai_files.py:delete_file_endpoint
        """
        f = openai_client.files.create(
            file=("del.txt", io.BytesIO(_TEXT_FILE), "text/plain"), purpose="assistants"
        )
        result = openai_client.files.delete(f.id)
        assert result.deleted is True
        assert result.id == f.id
        assert result.object == "file"
        with pytest.raises(OpenAINotFoundError) as exc_info:
            openai_client.files.retrieve(f.id)
        body = _error_envelope(exc_info.value, 404)
        message = str(body["message"]).lower()
        assert "not found" in message or "no such" in message, body

    # --- Content ---

    @pytest.mark.gateway("Official OpenAI API restricts file downloads by purpose")
    def test_download_content(self, openai_client: OpenAI) -> None:
        """``GET /files/{id}/content`` streams back the exact uploaded bytes and MIME type.

        The content type is not stored separately: it is the S3 object's own
        ``ContentType``, replayed as the streaming response media type.

        Ref: https://developers.openai.com/api/reference/resources/files
             stdapi/routes/openai_files.py:get_content
        """
        content = b"Hello, Files API content download test!"
        f = openai_client.files.create(
            file=("content.txt", io.BytesIO(content), "text/plain"),
            purpose="assistants",
        )
        try:
            downloaded = openai_client.files.content(f.id)
            assert downloaded.content == content
            assert downloaded.response.headers["content-type"].startswith("text/plain")
        finally:
            openai_client.files.delete(f.id)

    # --- Error cases ---

    def test_not_found(self, openai_client: OpenAI) -> None:
        """Retrieving or deleting an unknown but well-formed file ID returns 404.

        The ID matches the route's ``file-<32 chars>`` pattern, so the request
        reaches the store and fails on the missing S3 object — a 404, not the 400
        an ill-formed ID would produce.

        Ref: https://developers.openai.com/api/reference/resources/files
             stdapi/files/_core.py:_get_file_impl
        """
        fake_id = "file-" + "a" * 32
        with pytest.raises(OpenAINotFoundError) as retrieve_exc:
            openai_client.files.retrieve(fake_id)
        retrieve_message = str(
            _error_envelope(retrieve_exc.value, 404)["message"]
        ).lower()
        assert "not found" in retrieve_message or "no such" in retrieve_message

        with pytest.raises(OpenAINotFoundError) as delete_exc:
            openai_client.files.delete(fake_id)
        delete_message = str(_error_envelope(delete_exc.value, 404)["message"]).lower()
        assert "not found" in delete_message or "no such" in delete_message

    def test_expired_file_returns_404(
        self, openai_client: OpenAI, test_client: TestClient | None
    ) -> None:
        """A file past its ``expires_at`` reads as ``not_found`` even though S3 still holds it.

        S3 Lifecycle deletion is asynchronous, so expiry is enforced in code on
        every read: the record is compared against the clock and the object is
        queued for background deletion.  Advancing that clock is why this test is
        local-only.

        Ref: https://stdapi.ai/api_openai_files/
             stdapi/files/_core.py:_get_file_impl
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
                with pytest.raises(OpenAINotFoundError) as exc_info:
                    openai_client.files.retrieve(f.id)
            body = _error_envelope(exc_info.value, 404)
            assert body["code"] == "not_found"
            assert "expired" in str(body["message"]).lower(), body
        finally:
            with suppress(OpenAINotFoundError):
                # File may already be gone via background deletion triggered by the expired retrieve
                openai_client.files.delete(f.id)

    # --- Chat integration ---

    def test_file_in_chat_completion(
        self, openai_client: OpenAI, chat_vision_model: str, use_official_api: bool
    ) -> None:
        """An uploaded PDF is usable in a chat message through a ``file.file_id`` part.

        The gateway resolves the file ID back to its S3 object and forwards the
        bytes to Bedrock as a document content block, so a completion that
        charges prompt tokens is the observable proof the reference resolved.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             stdapi/models/chat/_adapters/_openai_chat_completion.py:_convert_content_part
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
            choice = response.choices[0]
            assert choice.message.role == "assistant"
            assert choice.finish_reason in {"stop", "length"}
            assert response.usage is not None
            assert response.usage.prompt_tokens > 0, (
                "the referenced document must be billed as prompt tokens"
            )
            if not use_official_api:
                content = choice.message.content
                assert content is not None
                assert len(content) > 0
        finally:
            openai_client.files.delete(f.id)

    @pytest.mark.gateway("`file-id:` is a project-local URI scheme")
    def test_file_id_uri_scheme_in_chat_image_url(
        self, openai_client: OpenAI, chat_vision_model: str, sample_image_file: bytes
    ) -> None:
        """``image_url.url`` accepts the project-local ``file-id:`` URI scheme.

        ``file-id:<id>`` is resolved by the shared input-file layer into the S3
        object behind the Files API entry, so the scheme works anywhere the
        gateway accepts a file input — including a field OpenAI defines as a
        plain URL.

        Ref: https://stdapi.ai/api_openai_files/
             stdapi/input_file.py:InputFile._normalize_and_detect_origin
        """
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
            choice = response.choices[0]
            assert choice.message.role == "assistant"
            assert choice.finish_reason in {"stop", "length"}
            assert response.usage is not None
            assert response.usage.prompt_tokens > 0, (
                "the resolved image must be billed as prompt tokens"
            )
            content = choice.message.content
            assert content is not None
            assert len(content) > 0
        finally:
            openai_client.files.delete(uploaded.id)


@pytest.mark.local
class TestRequireBucketUnit:
    """The Files API S3 bucket gate (unit, no AWS).

    Ref: stdapi/files/_core.py:_require_bucket
    """

    def test_no_bucket_hides_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a bucket, the 503 hides settings and warns the administrator.

        The client-facing message must not leak internal setting names; the
        operator gets them through the error log instead.

        Ref: stdapi/files/_core.py:_require_bucket
        """
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
    """Expiry stamping in ``create_multipart_session`` (unit, stubbed S3).

    The expiry is stamped on the S3 multipart upload itself, so the assembled
    object inherits both the ``expires-at`` metadata and the Lifecycle tag with
    no extra call at completion time.

    Ref: https://developers.openai.com/api/reference/resources/uploads
         stdapi/files/_multipart.py:create_multipart_session
    """

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
        """expires_after sets the expires-at metadata and the Lifecycle expiry tag.

        Ref: stdapi/files/_multipart.py:create_multipart_session
        """
        session = await _multipart.create_multipart_session(
            "f.bin", "a/b", "assistants", 1, 3600
        )
        metadata = stub_s3.create_kwargs["Metadata"]
        assert metadata["expires-at"] == str(session.created_at + 3600)
        assert "stdapi-ai.expires=true" in stub_s3.create_kwargs["Tagging"]

    async def test_no_expiry_leaves_metadata_empty(
        self, stub_s3: _StubMultipartS3Client
    ) -> None:
        """Without expires_after the metadata stays empty and no expiry tag is set.

        Ref: stdapi/files/_multipart.py:create_multipart_session
        """
        await _multipart.create_multipart_session("f.bin", "a/b", "assistants", 1)
        assert stub_s3.create_kwargs["Metadata"]["expires-at"] == ""
        assert "stdapi-ai.expires" not in stub_s3.create_kwargs["Tagging"]


class _FakeS3SourceInputFile:
    """Fake ``InputFile`` mimicking ``_S3Source``, whose ``to_s3`` ignores the requested metadata.

    Exactly like a real S3-to-S3 server-side copy.

    Ref: stdapi/input_file.py:_S3Source.to_s3
    """

    def __init__(self, filename: str) -> None:
        self._filename = filename

    async def get_filename(self) -> str | None:
        return self._filename

    async def to_s3(
        self,
        _region: object,
        *,
        bucket: str | None = None,
        key: str | None = None,
        temporary: bool = False,
        content_disposition: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> S3Object:
        return S3Object(bucket=bucket or "", key=key or "")


class _StubS3SourceCorrectionClient:
    """Stub S3 client modelling an object already copied with the *source's* own metadata.

    ``upload_file`` must detect the mismatch against what it requested and issue
    a corrective ``copy_object`` with ``MetadataDirective=REPLACE``.
    """

    def __init__(self) -> None:
        self.content_disposition = 'attachment; filename="source-object-name"'
        self.metadata: dict[str, str] = {"purpose": "fine-tune", "expires-at": ""}
        self.copy_object_kwargs: dict[str, Any] | None = None
        self.head_object_calls = 0

    async def head_object(self, **_kwargs: object) -> dict[str, Any]:
        self.head_object_calls += 1
        return {
            "ContentDisposition": self.content_disposition,
            "ContentType": "application/octet-stream",
            "Metadata": self.metadata,
            "ContentLength": 42,
            "LastModified": datetime.now(UTC),
        }

    async def copy_object(self, **kwargs: object) -> dict[str, Any]:
        self.copy_object_kwargs = kwargs
        self.content_disposition = cast("str", kwargs["ContentDisposition"])
        self.metadata = cast("dict[str, str]", kwargs["Metadata"])
        return {}


@pytest.mark.local
class TestUploadFileS3SourceMetadataUnit:
    """``upload_file`` forces purpose/filename onto S3-to-S3 copy sources (unit, stubbed S3).

    Issue #99(a): a server-side copy (used for ``s3://``/``file-id:`` upload
    sources) keeps the *source* object's own metadata and content-disposition,
    silently dropping the requested ``purpose``/filename. Without a fix, the
    resulting file both displays and is filtered as ``user_data`` regardless of
    what was requested, and lists under the source's raw key as its filename.

    Ref: https://platform.openai.com/docs/api-reference/files/create
         stdapi/input_file.py:_S3Source.to_s3
         stdapi/files/_core.py:upload_file
    """

    @pytest.fixture
    def stub_s3(self, monkeypatch: pytest.MonkeyPatch) -> _StubS3SourceCorrectionClient:
        """Patch the S3 client, bucket resolution, and region map with stubs."""
        stub = _StubS3SourceCorrectionClient()
        monkeypatch.setattr(_core, "get_client", lambda *_: stub)
        monkeypatch.setattr(_core, "_require_bucket", lambda: "bucket")
        monkeypatch.setattr(_core, "BUCKET_TO_REGION", {"bucket": "us-east-1"})
        return stub

    async def test_purpose_and_filename_are_forced_onto_s3_copy_source(
        self, stub_s3: _StubS3SourceCorrectionClient
    ) -> None:
        """A requested purpose/filename reach the record despite the copy dropping them.

        Ref: stdapi/files/_core.py:upload_file
        """
        fake_file = cast("InputFile", _FakeS3SourceInputFile("wanted.jsonl"))

        record = await _core.upload_file(fake_file, purpose="batch")

        assert record.purpose == "batch"
        assert record.filename == "wanted.jsonl"
        assert stub_s3.copy_object_kwargs is not None
        assert stub_s3.copy_object_kwargs["MetadataDirective"] == "REPLACE"

    async def test_matching_metadata_skips_the_corrective_copy(
        self, stub_s3: _StubS3SourceCorrectionClient
    ) -> None:
        """When the copy already carries the requested metadata, no extra copy is issued.

        A single ``HeadObject`` call is a discriminating check: a broken guard
        that unconditionally treats the copy as mismatched would trigger
        ``_force_s3_metadata`` (and hence its own extra ``HeadObject`` re-fetch)
        even though nothing here actually needs correcting.

        Ref: stdapi/files/_core.py:upload_file
        """
        stub_s3.content_disposition = 'attachment; filename="wanted.jsonl"'
        stub_s3.metadata = {"purpose": "batch", "expires-at": ""}
        fake_file = cast("InputFile", _FakeS3SourceInputFile("wanted.jsonl"))

        record = await _core.upload_file(fake_file, purpose="batch")

        assert record.purpose == "batch"
        assert stub_s3.copy_object_kwargs is None
        assert stub_s3.head_object_calls == 1


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
        # ContentLength/LastModified are only read by the final HeadObject that
        # builds the FileRecord after a successful completion.
        return {
            "Metadata": self.marker_metadata,
            "ContentLength": sum(size for _etag, size in self.parts.values()),
            "LastModified": datetime.now(UTC),
        }

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
    """Part-order validation in ``complete_multipart_session`` (unit, stubbed S3).

    S3 cannot reassemble multipart parts out of order (part numbers are fixed
    at add time), so out-of-order ``part_ids`` must be rejected with a clean
    400 before any S3 call is made -- not surfaced as a 502 from S3's
    ``InvalidPartOrder``.

    Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html
         stdapi/files/_multipart.py:complete_multipart_session
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
        """part_ids listed in descending order are rejected with 400, mentioning order.

        Ref: stdapi/files/_multipart.py:complete_multipart_session
        """
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
        message = str(exc_info.value)
        assert "order" in message.lower()
        assert part_1 in message, (
            "the rejection must name the part that broke the ascending order"
        )
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
    """Part numbering in ``add_part`` (unit, stubbed S3).

    Part numbers must continue the parts S3 already holds: several server
    instances share one upload session through a load balancer, and a
    process-local counter would hand the same number to two parts, silently
    overwriting one of them in S3.

    Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html
         stdapi/files/_multipart.py:add_part
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
        """Parts added one after another get consecutive numbers from 1.

        Ref: stdapi/files/_multipart.py:_make_part_id
        """
        session = await _multipart.create_multipart_session(
            "f.bin", "text/plain", "assistants", 8
        )

        first, _ = await _multipart.add_part(session.upload_id, b"1234")
        second, _ = await _multipart.add_part(session.upload_id, b"5678")

        extract = _multipart._extract_part_number  # noqa: SLF001
        assert extract(first, session.upload_id) == 1
        assert extract(second, session.upload_id) == 2
        assert stub_s3.parts == {1: ("etag-1", 4), 2: ("etag-2", 4)}, (
            "both parts must reach S3 under distinct part numbers"
        )

    async def test_part_uploaded_by_another_instance_advances_the_number(
        self, stub_s3: _StubAddPartS3Client
    ) -> None:
        """A part stored by another instance is counted, so its number is not reused.

        Ref: stdapi/files/_multipart.py:add_part
        """
        session = await _multipart.create_multipart_session(
            "f.bin", "text/plain", "assistants", 8
        )
        # Part served by another instance: this process never saw it.
        stub_s3.parts[1] = ("etag-1", 4)

        part_id, _ = await _multipart.add_part(session.upload_id, b"5678")

        assert _multipart._extract_part_number(part_id, session.upload_id) == 2  # noqa: SLF001
        assert stub_s3.parts[1] == ("etag-1", 4)

    async def test_part_number_ceiling_rejected(
        self, stub_s3: _StubAddPartS3Client
    ) -> None:
        """A session already holding S3's 10 000-part maximum rejects one more, with 400.

        The rejection is raised before any S3 call, so S3's own ``InvalidArgument``
        for an out-of-range ``PartNumber`` never has to be mapped.

        Ref: https://docs.aws.amazon.com/AmazonS3/latest/API/API_UploadPart.html
             stdapi/files/_multipart.py:add_part
        """
        session = await _multipart.create_multipart_session(
            "f.bin", "text/plain", "assistants", 1
        )
        max_part_number = _multipart._MAX_PART_NUMBER  # noqa: SLF001
        stub_s3.parts[max_part_number] = ("etag-max", 1)

        with pytest.raises(ApiError) as exc_info:
            await _multipart.add_part(session.upload_id, b"1234")

        assert exc_info.value.status == 400
        assert str(max_part_number) in str(exc_info.value)
        assert len(stub_s3.parts) == 1, (
            "no new part must reach S3 once the session is already at the ceiling"
        )


@pytest.mark.local
class TestCompleteMultipartSessionMinPartSizeUnit:
    """Non-last part minimum size validation in ``complete_multipart_session`` (unit, stubbed S3).

    S3 enforces its 5 MiB minimum part size (every part except the last) at
    ``CompleteMultipartUpload`` time, surfacing as ``EntityTooSmall``; the
    gateway checks it first so an undersized part is rejected with a clean,
    OpenAI-shaped 400 instead of a raw S3 error.

    Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html
         stdapi/files/_multipart.py:complete_multipart_session
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

    async def test_undersized_non_last_part_rejected(
        self, stub_s3: _StubCompleteS3Client
    ) -> None:
        """A non-last part below 5 MiB is rejected with 400, naming the part and no backend.

        Ref: stdapi/files/_multipart.py:complete_multipart_session
        """
        min_size = _multipart._MIN_PART_SIZE  # noqa: SLF001
        session = await _multipart.create_multipart_session(
            "f.bin", "text/plain", "assistants", min_size + 9
        )
        part_1 = _multipart._make_part_id(session.upload_id, 1)  # noqa: SLF001
        part_2 = _multipart._make_part_id(session.upload_id, 2)  # noqa: SLF001
        stub_s3.parts = {1: ("etag-1", min_size - 1), 2: ("etag-2", 10)}

        with pytest.raises(ApiError) as exc_info:
            await _multipart.complete_multipart_session(
                session.upload_id, [part_1, part_2]
            )

        assert exc_info.value.status == 400
        message = str(exc_info.value)
        assert str(min_size) in message
        assert part_1 in message, "the rejection must name the undersized part"
        assert "s3" not in message.lower(), "must not leak the backing storage service"
        assert "bucket" not in message.lower(), "must not leak the S3 bucket concept"
        assert stub_s3.complete_called is False

    async def test_undersized_last_part_is_accepted(
        self, stub_s3: _StubCompleteS3Client
    ) -> None:
        """The last part may be smaller than 5 MiB; only non-last parts are checked.

        The same undersized size is exercised in both positions on the same
        session: rejected as part 1 (non-last), accepted as part 2 (last). A
        check that covered every part, or no size check at all, would fail
        one of the two assertions, so this discriminates the "last part is
        exempt" behavior rather than just showing completion can succeed.

        Ref: stdapi/files/_multipart.py:complete_multipart_session
        """
        min_size = _multipart._MIN_PART_SIZE  # noqa: SLF001
        small = min_size - 1
        session = await _multipart.create_multipart_session(
            "f.bin", "text/plain", "assistants", min_size + small
        )
        part_1 = _multipart._make_part_id(session.upload_id, 1)  # noqa: SLF001
        part_2 = _multipart._make_part_id(session.upload_id, 2)  # noqa: SLF001

        stub_s3.parts = {1: ("etag-1", small), 2: ("etag-2", min_size)}
        with pytest.raises(ApiError):
            await _multipart.complete_multipart_session(
                session.upload_id, [part_1, part_2]
            )
        assert stub_s3.complete_called is False

        stub_s3.parts = {1: ("etag-1", min_size), 2: ("etag-2", small)}
        await _multipart.complete_multipart_session(session.upload_id, [part_1, part_2])

        assert stub_s3.complete_called is True


@pytest.mark.local
class TestOpenAIFilesMalformedJsonBody:
    """POST /v1/files with a malformed JSON body (unit, no AWS).

    Ref: stdapi/routes/openai_files.py:upload
         stdapi/utils.py:validation_error_handler
    """

    def test_malformed_json_body_is_rejected(self, app_client: TestClient) -> None:
        """A malformed JSON body is rejected with 400 and a JSON decode error, not a 500.

        The body is read with ``Request.json()``, so the ``JSONDecodeError`` has
        to be converted into a request-validation error to keep the OpenAI
        envelope instead of bubbling up as a 500.

        Ref: stdapi/utils.py:validation_error_handler
        """
        response = app_client.post(
            "/v1/files", content=b"{", headers={"content-type": "application/json"}
        )
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "json" in error["message"].lower(), error


@pytest.mark.local
class TestOpenAIFilesExpiresAfterBracketNotation:
    """POST /v1/files with the bracket-notation ``expires_after[seconds]`` form field.

    The pydantic ``Form`` binding (``expires_after_seconds`` / ``expires_after[seconds]``
    alias) only matches the unbracketed alias in practice; a manual fallback reads the
    raw bracket-notation value and must enforce the same 1 hour-30 day bounds (unit,
    no AWS -- the file never reaches S3 in the rejected cases).

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/routes/openai_files.py:upload
    """

    @staticmethod
    def _upload(app_client: TestClient, seconds_value: str) -> httpx.Response:
        """POST a minimal file with a bracket-notation ``expires_after[seconds]`` field."""
        return cast(
            "httpx.Response",
            app_client.post(
                "/v1/files",
                files={"file": ("t.txt", b"hello", "text/plain")},
                data={"purpose": "assistants", "expires_after[seconds]": seconds_value},
            ),
        )

    def test_bracket_seconds_below_minimum_rejected(
        self, app_client: TestClient
    ) -> None:
        """59 seconds (below the 3600s minimum) is rejected with 400, not accepted.

        Ref: stdapi/routes/openai_files.py:upload
        """
        response = self._upload(app_client, "59")
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "3600" in error["message"], error

    def test_bracket_seconds_above_maximum_rejected(
        self, app_client: TestClient
    ) -> None:
        """99999999 seconds (above the 2592000s maximum) is rejected with 400.

        Ref: stdapi/routes/openai_files.py:upload
        """
        response = self._upload(app_client, "99999999")
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "2592000" in error["message"], error

    def test_bracket_seconds_non_numeric_rejected(self, app_client: TestClient) -> None:
        """A non-numeric value is rejected with 400 and a JSON error envelope, not a bare 500.

        Ref: stdapi/routes/openai_files.py:upload
        """
        response = self._upload(app_client, "not_a_number")
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "integer" in error["message"].lower(), error

    def test_bracket_seconds_valid_value_accepted(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A valid bracket-notation value (7200s) is still accepted (regression guard).

        Ref: stdapi/routes/openai_files.py:upload
        """
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
        response = self._upload(app_client, "7200")
        assert response.status_code == 200, response.text
        assert captured["expires_after"] == 7200
        assert response.json()["expires_at"] is not None


class TestResolveExpiresAfterSecondsUnit:
    """The ``purpose=batch`` default-expiry resolution helper (unit, no AWS).

    Ref: https://stdapi.ai/api_openai_files/
         stdapi/routes/openai_files.py:_resolve_expires_after_seconds
    """

    def test_batch_purpose_defaults_to_thirty_days(self) -> None:
        """purpose=batch with no explicit TTL defaults to the 30-day maximum.

        Ref: stdapi/routes/openai_files.py:_resolve_expires_after_seconds
        """
        resolved = openai_files_routes._resolve_expires_after_seconds(  # noqa: SLF001
            "batch", None
        )
        assert resolved == openai_files_routes._EXPIRES_AFTER_SECONDS_MAX  # noqa: SLF001
        assert resolved == 2_592_000

    def test_batch_purpose_explicit_ttl_not_overridden(self) -> None:
        """An explicit TTL for purpose=batch is preserved, not replaced by the default.

        Ref: stdapi/routes/openai_files.py:_resolve_expires_after_seconds
        """
        resolved = openai_files_routes._resolve_expires_after_seconds(  # noqa: SLF001
            "batch", 3600
        )
        assert resolved == 3600

    def test_non_batch_purpose_has_no_default(self) -> None:
        """Purposes other than batch persist forever unless a TTL is explicitly given.

        Ref: stdapi/routes/openai_files.py:_resolve_expires_after_seconds
        """
        resolved = openai_files_routes._resolve_expires_after_seconds(  # noqa: SLF001
            "assistants", None
        )
        assert resolved is None


class TestOpenAIFilesBatchDefaultExpiry:
    """POST /v1/files applies the documented 30-day default expiry for purpose=batch.

    ``upload_file`` is stubbed, so these tests pin the TTL the route computes for
    each body form rather than S3 behavior.

    Ref: https://stdapi.ai/api_openai_files/
         stdapi/routes/openai_files.py:upload
    """

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
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A multipart upload with purpose=batch and no expires_after gets a 30-day TTL.

        Ref: stdapi/routes/openai_files.py:upload
        """
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            openai_files_routes, "upload_file", self._fake_upload_file(captured)
        )
        response = app_client.post(
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
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A multipart upload with purpose=assistants and no expires_after never expires.

        ``expires_at`` is omitted from the response entirely (the route is
        declared ``response_model_exclude_none``) rather than sent as null.

        Ref: stdapi/routes/openai_files.py:upload
        """
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            openai_files_routes, "upload_file", self._fake_upload_file(captured)
        )
        response = app_client.post(
            "/v1/files",
            files={"file": ("t.txt", b"hello", "text/plain")},
            data={"purpose": "assistants"},
        )
        assert response.status_code == 200, response.text
        assert captured["expires_after"] is None
        assert "expires_at" not in response.json()

    def test_json_body_batch_purpose_gets_default_expiry(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A JSON-body upload with purpose=batch and no expires_after gets a 30-day TTL.

        Ref: stdapi/routes/openai_files.py:upload
        """
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            openai_files_routes, "upload_file", self._fake_upload_file(captured)
        )
        response = app_client.post(
            "/v1/files",
            json={"file": "data:text/plain;base64,aGVsbG8=", "purpose": "batch"},
        )
        assert response.status_code == 200, response.text
        assert (
            captured["expires_after"] == openai_files_routes._EXPIRES_AFTER_SECONDS_MAX  # noqa: SLF001
        )


class TestOpenAIUploads:
    """OpenAI ``/v1/uploads`` create → add part → complete/cancel state machine on S3.

    A session is an S3 native multipart upload plus a zero-byte marker object
    holding the fields S3 does not expose for an in-progress upload; the upload ID
    and the file it produces share one payload (``upload_X`` → ``file-X``).

    Ref: https://developers.openai.com/api/reference/resources/uploads
         https://stdapi.ai/api_openai_files/
         stdapi/files/_multipart.py:MultipartSession
    """

    # --- Create ---

    def test_create_returns_upload_object(self, openai_client: OpenAI) -> None:
        """Creating an upload returns a pending Upload echoing the declared metadata.

        ``file`` stays null until the upload completes, and ``expires_at`` is the
        session's own cleanup window, not the final file's TTL.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_uploads.py:_to_upload
        """
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
            assert upload.created_at > 0
            assert upload.expires_at > upload.created_at
            assert upload.file is None, "a pending upload has produced no file yet"
        finally:
            openai_client.uploads.cancel(upload.id)

    def test_create_with_expires_after_file_expires(
        self, openai_client: OpenAI
    ) -> None:
        """``expires_after`` on the upload lands on the assembled file's ``expires_at``.

        The TTL is stamped on the S3 multipart upload at creation time, so the
        assembled object inherits it; the anchor is the session's ``created_at``,
        which is decoded from the UUIDv7 inside the upload ID.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/files/_multipart.py:create_multipart_session
        """
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
            assert completed.status == "completed"
            assert completed.file is not None
            assert completed.file.expires_at is not None
            assert completed.file.expires_at > int(time.time())
            assert abs(completed.file.expires_at - (upload.created_at + 3600)) <= 60, (
                "the file TTL must be anchored to the upload's creation time"
            )
        finally:
            assert completed.file is not None
            openai_client.files.delete(completed.file.id)

    def test_create_expires_after_out_of_range_rejected(
        self, openai_client: OpenAI
    ) -> None:
        """``expires_after.seconds`` below the 1 hour minimum is rejected with 400.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_uploads.py:UploadExpiresAfter
        """
        with pytest.raises(BadRequestError, match="seconds") as exc_info:
            openai_client.uploads.create(
                bytes=1024,
                filename="bad_expiry.bin",
                mime_type="application/octet-stream",
                purpose="assistants",
                expires_after={"anchor": "created_at", "seconds": 60},
            )
        message = str(_error_envelope(exc_info.value, 400)["message"])
        assert "3600" in message or "hour" in message.lower(), message

    def test_create_expires_after_above_maximum_rejected(
        self, openai_client: OpenAI
    ) -> None:
        """``expires_after.seconds`` above the 30-day maximum is rejected with 400.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_uploads.py:UploadExpiresAfter
        """
        with pytest.raises(BadRequestError, match="seconds") as exc_info:
            openai_client.uploads.create(
                bytes=1024,
                filename="bad_expiry_max.bin",
                mime_type="application/octet-stream",
                purpose="assistants",
                expires_after={"anchor": "created_at", "seconds": 2_592_001},
            )
        message = str(_error_envelope(exc_info.value, 400)["message"])
        assert "2592000" in message or "day" in message.lower(), message

    def test_create_expires_after_unsupported_anchor_rejected(
        self, openai_client: OpenAI
    ) -> None:
        """``created_at`` is the only accepted ``expires_after.anchor``.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_uploads.py:UploadExpiresAfter
        """
        with pytest.raises(BadRequestError, match="anchor") as exc_info:
            openai_client.uploads.create(
                bytes=1024,
                filename="bad_anchor.bin",
                mime_type="application/octet-stream",
                purpose="assistants",
                expires_after={"anchor": "updated_at", "seconds": 3600},  # type: ignore[arg-type]
            )
        message = str(_error_envelope(exc_info.value, 400)["message"])
        assert "created_at" in message or "updated_at" in message, message

    # --- Add parts ---

    def test_add_part_returns_upload_part(self, openai_client: OpenAI) -> None:
        """Adding a part returns an ``upload.part`` object bound to its session.

        The part ID encodes the session fingerprint plus the 1-based S3 part
        number, so completion can validate ownership and ordering without any
        server-side state.

        Ref: https://developers.openai.com/api/reference/resources/uploads
             stdapi/files/_multipart.py:_make_part_id
        """
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
            assert part.created_at >= upload.created_at
        finally:
            openai_client.uploads.cancel(upload.id)

    # --- Complete ---

    def test_complete_produces_file(self, openai_client: OpenAI) -> None:
        """Completing an upload returns a ``completed`` Upload wrapping the assembled File.

        S3 reassembles the parts in part-number order, and the resulting File
        carries the size, filename and purpose declared when the session was
        created -- none of which S3 exposes for an in-progress multipart upload.

        Ref: https://developers.openai.com/api/reference/resources/uploads
             stdapi/files/_multipart.py:complete_multipart_session
        """
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
            assert completed.id == upload.id
            assert completed.file is not None
            assert completed.file.object == "file"
            assert completed.file.bytes == len(content)
            assert completed.file.filename == "complete_test.txt"
            assert completed.file.purpose == "assistants"
        finally:
            assert completed.file is not None
            openai_client.files.delete(completed.file.id)

    @pytest.mark.gateway("Official OpenAI API restricts file downloads by purpose")
    def test_complete_file_downloadable(self, openai_client: OpenAI) -> None:
        """The file produced by a completed upload is downloadable byte for byte.

        A single part below the S3 5 MiB minimum is legal because the last part
        is exempt from that limit.

        Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html
             stdapi/files/_multipart.py:complete_multipart_session
        """
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
            assert completed.file.bytes == len(payload)
            downloaded = openai_client.files.content(completed.file.id).content
            assert downloaded == payload
        finally:
            assert completed.file is not None
            openai_client.files.delete(completed.file.id)

    @pytest.mark.gateway(
        "official API rejects at add_part time (not complete time) when part exceeds declared bytes"
    )
    def test_complete_wrong_size_rejected(self, openai_client: OpenAI) -> None:
        """Completing with parts that do not add up to the declared ``bytes`` is rejected.

        The declared total is recorded on the session marker at creation and
        compared against the summed S3 part sizes before
        ``CompleteMultipartUpload`` is issued.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/files/_multipart.py:complete_multipart_session
        """
        upload = openai_client.uploads.create(
            bytes=999999,  # wrong: doesn't match part sizes
            filename="size_mismatch.bin",
            mime_type="application/octet-stream",
            purpose="assistants",
        )
        try:
            part = openai_client.uploads.parts.create(
                upload_id=upload.id, data=io.BytesIO(_PART_A)
            )
            with pytest.raises(BadRequestError) as exc_info:
                openai_client.uploads.complete(upload_id=upload.id, part_ids=[part.id])
            message = str(_error_envelope(exc_info.value, 400)["message"])
            assert "999999" in message, message
            assert str(len(_PART_A)) in message, message
        finally:
            # AWS bills the uploaded parts until the multipart upload is aborted.
            with suppress(OpenAINotFoundError, BadRequestError):
                openai_client.uploads.cancel(upload.id)

    def test_complete_unknown_part_id_rejected(self, openai_client: OpenAI) -> None:
        """Completing with a part ID that was never added is rejected with 400.

        Part IDs embed the session fingerprint, so an ID from outside this upload
        is refused without consulting S3 at all.

        Ref: https://developers.openai.com/api/reference/resources/uploads
             stdapi/files/_multipart.py:_extract_part_number
        """
        upload = openai_client.uploads.create(
            bytes=len(_PART_A),
            filename="bad_part.bin",
            mime_type="application/octet-stream",
            purpose="assistants",
        )
        try:
            with pytest.raises(BadRequestError) as exc_info:
                openai_client.uploads.complete(
                    upload_id=upload.id, part_ids=["part_" + "a" * 32]
                )
            message = str(_error_envelope(exc_info.value, 400)["message"]).lower()
            assert "part" in message, message
        finally:
            openai_client.uploads.cancel(upload.id)

    # --- Cancel ---

    def test_cancel_sets_status_cancelled(self, openai_client: OpenAI) -> None:
        """Cancelling a pending upload returns the same Upload with status ``cancelled``.

        Ref: https://developers.openai.com/api/reference/resources/uploads
             stdapi/files/_multipart.py:cancel_multipart_session
        """
        upload = openai_client.uploads.create(
            bytes=len(_PART_A),
            filename="cancel_test.bin",
            mime_type="application/octet-stream",
            purpose="assistants",
        )
        cancelled = openai_client.uploads.cancel(upload.id)
        assert cancelled.status == "cancelled"
        assert cancelled.id == upload.id
        assert cancelled.object == "upload"
        assert cancelled.bytes == upload.bytes
        assert cancelled.filename == upload.filename

    def test_cancel_prevents_further_parts(self, openai_client: OpenAI) -> None:
        """A cancelled upload accepts no further parts.

        Cancelling aborts the S3 multipart upload and queues the session marker
        for background deletion, so the follow-up part fails either as "not
        pending" (400) or, once the marker is gone, as "not found" (404).

        Ref: https://developers.openai.com/api/reference/resources/uploads
             stdapi/files/_multipart.py:_check_not_pending
        """
        upload = openai_client.uploads.create(
            bytes=len(_PART_A),
            filename="cancel_parts.bin",
            mime_type="application/octet-stream",
            purpose="assistants",
        )
        openai_client.uploads.cancel(upload.id)
        with pytest.raises((BadRequestError, OpenAINotFoundError)) as exc_info:
            openai_client.uploads.parts.create(
                upload_id=upload.id, data=io.BytesIO(_PART_A)
            )
        assert exc_info.value.status_code in {400, 404}
        body = exc_info.value.body
        assert isinstance(body, dict), f"expected a JSON error envelope, got {body!r}"
        assert body["type"] == "invalid_request_error", body
        message = str(body["message"]).lower()
        assert any(text in message for text in ("pending", "not found", "no such")), (
            body
        )

    # --- Error cases ---

    def test_not_found_upload(self, openai_client: OpenAI) -> None:
        """Cancelling an unknown but well-formed upload ID returns 404.

        The session marker is what proves existence, so a missing marker is a
        404 rather than the 400 used for a session that exists but is no longer
        pending.

        Ref: https://developers.openai.com/api/reference/resources/uploads
             stdapi/files/_multipart.py:_load_multipart_session
        """
        fake_id = "upload_" + "a" * 32
        with pytest.raises(OpenAINotFoundError) as exc_info:
            openai_client.uploads.cancel(fake_id)
        message = str(_error_envelope(exc_info.value, 404)["message"]).lower()
        assert "not found" in message or "no such" in message, message

    def test_not_found_add_part(self, openai_client: OpenAI) -> None:
        """Adding a part to an unknown upload ID returns 404.

        Ref: https://developers.openai.com/api/reference/resources/uploads
             stdapi/files/_multipart.py:_check_not_pending
        """
        fake_id = "upload_" + "a" * 32
        with pytest.raises(OpenAINotFoundError) as exc_info:
            openai_client.uploads.parts.create(
                upload_id=fake_id, data=io.BytesIO(_PART_A)
            )
        message = str(_error_envelope(exc_info.value, 404)["message"]).lower()
        assert "not found" in message or "no such" in message, message


@pytest.mark.gateway("JSON body input not supported by the official OpenAI API")
class TestOpenAIUploadsJsonBody:
    """POST /v1/uploads/{upload_id}/parts with an ``application/json`` body.

    The JSON form (base64, data URI, HTTPS URL, S3 URI) is a gateway extension so
    that MCP agents unable to build multipart requests can still run the upload
    flow end to end.

    Ref: https://stdapi.ai/api_openai_files/
         stdapi/routes/openai_uploads.py:add_upload_part
    """

    def test_json_body_missing_data_returns_400(self, openai_client: OpenAI) -> None:
        """A JSON body without the required ``data`` field returns 400 naming the field.

        Ref: stdapi/types/openai_uploads.py:AddUploadPartJsonBody
        """
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
            assert response.status_code == 400, response.text
            error = response.json()["error"]
            assert error["type"] == "invalid_request_error"
            assert "data" in error["message"], error
        finally:
            openai_client.uploads.cancel(upload.id)

    def test_json_body_part_upload_with_data_uri(self, openai_client: OpenAI) -> None:
        """A part sent as a data URI completes the upload exactly like a binary part.

        Ref: https://stdapi.ai/api_openai_files/
             stdapi/routes/openai_uploads.py:add_upload_part
        """
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
        file_id: str | None = None
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
            assert completed.file.bytes == len(_PART_A), (
                "the base64 payload must be decoded before it reaches S3"
            )
            file_id = completed.file.id
        finally:
            with suppress(OpenAINotFoundError, BadRequestError):
                openai_client.uploads.cancel(upload.id)
            # A completed upload leaves a 5 MiB S3 object behind; delete it too.
            if file_id is not None:
                with suppress(OpenAINotFoundError):
                    openai_client.files.delete(file_id)


@pytest.mark.gateway("JSON body input not supported by the official OpenAI API")
class TestOpenAIFilesJsonBody:
    """POST /v1/files with an ``application/json`` body.

    The ``file`` field accepts a raw base64 string, a data URI, an HTTPS URL or an
    S3 URI; the gateway detects the encoding and MIME type itself, so the result
    is a File object identical to the multipart form upload.

    Ref: https://stdapi.ai/api_openai_files/
         stdapi/types/openai_files.py:FileUploadJsonBody
    """

    def test_json_body_missing_file_returns_400(self, openai_client: OpenAI) -> None:
        """A JSON body without the required ``file`` field returns 400 naming the field.

        Ref: stdapi/types/openai_files.py:FileUploadJsonBody
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}files",
            json={"purpose": "user_data"},
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "file" in error["message"], error

    def test_json_body_upload_with_data_uri(self, openai_client: OpenAI) -> None:
        """A data URI in the JSON ``file`` field is decoded and stored as a File object.

        The stored size proves the base64 payload was decoded rather than saved
        verbatim: ``SGVsbG8gV29ybGQ=`` is 16 characters but 11 bytes.

        Ref: https://stdapi.ai/api_openai_files/
             stdapi/routes/openai_files.py:upload
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}files",
            json={
                "file": "data:text/plain;base64,SGVsbG8gV29ybGQ=",
                "purpose": "user_data",
            },
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"].startswith("file-")
        assert body["object"] == "file"
        assert body["purpose"] == "user_data"
        assert body["bytes"] == 11
        openai_client.files.delete(body["id"])

    def test_json_body_upload_with_raw_base64(self, openai_client: OpenAI) -> None:
        """A bare base64 string in the JSON ``file`` field is accepted like a data URI.

        Ref: https://stdapi.ai/api_openai_files/
             stdapi/input_file.py:InputFile._normalize_and_detect_origin
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}files",
            json={"file": "SGVsbG8gV29ybGQ=", "purpose": "user_data"},
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"].startswith("file-")
        assert body["object"] == "file"
        assert body["bytes"] == 11
        openai_client.files.delete(body["id"])
