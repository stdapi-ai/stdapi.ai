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
from asyncio import Event, wait_for
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
from tests._helpers import make_client_error

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

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

    @pytest.mark.usefixtures("local_test_client")
    def test_expired_file_returns_404(self, openai_client: OpenAI) -> None:
        """A file past its ``expires_at`` reads as ``not_found`` even though S3 still holds it.

        S3 Lifecycle deletion is asynchronous, so expiry is enforced in code on
        every read: the record is compared against the clock and the object is
        queued for background deletion.  Advancing that clock is why this test is
        local-only.

        Ref: https://stdapi.ai/api_openai_files/
             stdapi/files/_core.py:_get_file_impl
        """
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


class TestForceS3MetadataMultipart:
    """The metadata-forcing self-copy above 5 GiB fans its parts out concurrently.

    Ref: https://docs.aws.amazon.com/AmazonS3/latest/API/API_UploadPartCopy.html
         stdapi/files/_core.py:_force_s3_metadata
         stdapi/aws_s3.py:multipart_copy_parts
    """

    async def test_multipart_metadata_fix_copies_parts_concurrently(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ranged self-copies overlap and complete in part-number order.

        Each stubbed part copy blocks until all three are in flight, so this
        test fails (times out) if the metadata fix regresses to sequential
        copies. The requested metadata must still reach the multipart create.
        """
        monkeypatch.setattr(_core, "_COPY_OBJECT_MAX_BYTES", 20)
        monkeypatch.setattr(_core, "_METADATA_FIX_PART_SIZE", 10)
        all_started = Event()
        in_flight = 0
        create_kwargs: dict[str, Any] = {}
        copy_ranges: dict[int, str] = {}
        completed: list[dict[str, Any]] = []

        class _StubS3Client:
            async def create_multipart_upload(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
                create_kwargs.update(kwargs)
                return {"UploadId": "mpu-1"}

            async def upload_part_copy(
                self,
                *,
                PartNumber: int,  # noqa: N803
                CopySourceRange: str,  # noqa: N803
                **_kwargs: object,
            ) -> dict[str, Any]:
                nonlocal in_flight
                in_flight += 1
                if in_flight >= 3:
                    all_started.set()
                # Times out (instead of hanging) if copies are sequential.
                await wait_for(all_started.wait(), timeout=5)
                copy_ranges[PartNumber] = CopySourceRange
                return {"CopyPartResult": {"ETag": f'"etag-{PartNumber}"'}}

            async def complete_multipart_upload(
                self,
                *,
                MultipartUpload: dict[str, Any],  # noqa: N803
                **_kwargs: object,
            ) -> dict[str, Any]:
                completed.extend(MultipartUpload["Parts"])
                return {}

        await wait_for(
            _core._force_s3_metadata(  # noqa: SLF001
                cast("Any", _StubS3Client()),
                "bucket",
                "key",
                25,
                "text/plain",
                'attachment; filename="wanted.jsonl"',
                {"purpose": "batch", "expires-at": ""},
            ),
            timeout=5,
        )

        assert create_kwargs["ContentType"] == "text/plain"
        assert create_kwargs["Metadata"] == {"purpose": "batch", "expires-at": ""}
        assert [part["PartNumber"] for part in completed] == [1, 2, 3]
        assert copy_ranges == {1: "bytes=0-9", 2: "bytes=10-19", 3: "bytes=20-24"}


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

    async def test_a_part_never_crosses_into_a_sibling_session(
        self, stub_s3: _StubAddPartS3Client
    ) -> None:
        """Two sessions created back to back do not accept each other's parts.

        The part id carries a fingerprint of its session, and that is the only
        thing stopping a part from being completed against the wrong upload. The
        session id opens with a millisecond timestamp, so a fingerprint sliced
        off its front is identical for any two sessions created inside the same
        millisecond -- which is what creating them back to back does.

        Ref: stdapi/files/_multipart.py:_upload_fingerprint
        """
        first = await _multipart.create_multipart_session(
            "a.bin", "text/plain", "assistants", 8
        )
        second = await _multipart.create_multipart_session(
            "b.bin", "text/plain", "assistants", 8
        )

        part_of_second = _multipart._make_part_id(second.upload_id, 3)  # noqa: SLF001

        with pytest.raises(ApiError, match="does not belong to upload"):
            _multipart._extract_part_number(part_of_second, first.upload_id)  # noqa: SLF001
        assert _multipart._extract_part_number(part_of_second, second.upload_id) == 3  # noqa: SLF001

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

        The body is parsed with ``model_validate_json``, so the pydantic
        ``json_invalid`` error has to be converted into a request-validation
        error to keep the OpenAI envelope instead of bubbling up as a 500.

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
        """A valid bracket-notation value reaches the route as a 7200 s TTL.

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


def _recording_upload_file(calls: list[dict[str, Any]]) -> Any:  # noqa: ANN401
    """Build a fake ``upload_file`` that records every call instead of reaching S3.

    Args:
        calls: List each call appends its ``file``/``purpose``/``expires_after`` to.

    Returns:
        A replacement for ``stdapi.routes.openai_files.upload_file``.
    """

    async def fake_upload_file(
        file: object, purpose: str | None = None, expires_after: int | None = None
    ) -> FileRecord:
        calls.append({"file": file, "purpose": purpose, "expires_after": expires_after})
        return FileRecord(
            file_id="a" * 32,
            filename="t.txt",
            content_type="text/plain",
            purpose=purpose or "",
            size=5,
            created_at=datetime.now(UTC),
            expires_at=None,
        )

    return fake_upload_file


class _StubListS3Client:
    """Stub S3 client serving a fixed key set to the listing scan and its HeadObject fan-out.

    ``list_objects_v2`` honours ``StartAfter`` (the ascending fast path) and always
    reports a single non-truncated page, which is enough for the listing paths that
    scan every key before slicing in Python.
    """

    def __init__(self, keys: list[str], purposes: dict[str, str] | None = None) -> None:
        self.keys = sorted(keys)
        self.purposes = purposes or {}

    async def list_objects_v2(self, **kwargs: object) -> dict[str, Any]:
        start_after = cast("str | None", kwargs.get("StartAfter"))
        return {
            "Contents": [
                {"Key": key}
                for key in self.keys
                if start_after is None or key > start_after
            ],
            "IsTruncated": False,
        }

    async def head_object(self, **kwargs: object) -> dict[str, Any]:
        key = cast("str", kwargs["Key"])
        return {
            "ContentLength": 3,
            "LastModified": datetime.now(UTC),
            "Metadata": {
                "purpose": self.purposes.get(key, "user_data"),
                "expires-at": "",
            },
            "ContentDisposition": 'attachment; filename="f.txt"',
            "ContentType": "text/plain",
        }


@pytest.mark.local
class TestListFilesDescendingCursorUnit:
    """The ``after`` cursor under the default ``order=desc`` (unit, stubbed S3).

    ``desc`` is the default order and the one the OpenAI SDK auto-paginates with,
    yet it takes a different branch from ``asc``: instead of handing the cursor to
    the storage scan, it scans everything and keeps only the keys sorting strictly
    below the cursor. A wrong comparison there re-returns the cursor file or drops
    a page silently rather than failing.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/files/_core.py:list_files
    """

    @pytest.fixture
    def stored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[_StubListS3Client, list[str]]:
        """Store three files, oldest first, behind a stubbed S3 client.

        Returns:
            The stub client and the three bare payloads in creation order.
        """
        payloads = sorted(_core.encode_id_payload("bucket") for _ in range(3))
        stub = _StubListS3Client([_core.file_id_s3_key(p) for p in payloads])
        monkeypatch.setattr(_core, "get_client", lambda *_: stub)
        monkeypatch.setattr(_core, "_require_bucket", lambda: "bucket")
        monkeypatch.setattr(_core, "BUCKET_TO_REGION", {"bucket": "us-east-1"})
        return stub, payloads

    async def test_desc_after_returns_the_older_files_newest_first(
        self, stored: tuple[_StubListS3Client, list[str]]
    ) -> None:
        """``after`` excludes the cursor itself and yields only files created before it.

        The cursor is the newest of the three, so a correct descending page is the
        two older ones, newest first; returning the cursor or reversing the pair
        would both fail here.

        Ref: stdapi/files/_core.py:list_files
        """
        _stub, payloads = stored

        records, has_more = await _core.list_files(payloads[2], None, 100, "desc", None)

        assert [r.file_id for r in records] == [payloads[1], payloads[0]]
        assert has_more is False

    async def test_desc_after_reports_has_more_on_a_truncated_page(
        self, stored: tuple[_StubListS3Client, list[str]]
    ) -> None:
        """``has_more`` counts the files left below the cursor, not the whole bucket.

        Two files sit below the cursor and only one fits the page, so ``has_more``
        must be true; a count taken before the cursor filter would be true here too,
        which is why the complementary full-page case above asserts it is false.

        Ref: stdapi/files/_core.py:list_files
        """
        _stub, payloads = stored

        records, has_more = await _core.list_files(payloads[2], None, 1, "desc", None)

        assert [r.file_id for r in records] == [payloads[1]]
        assert has_more is True

    async def test_desc_after_combined_with_the_purpose_filter(
        self, stored: tuple[_StubListS3Client, list[str]]
    ) -> None:
        """The purpose filter and the ``after`` cursor both apply, still newest first.

        A purpose filter routes the page through a separate slice, so the cursor
        exclusion has to survive it: the newest ``batch`` file is the cursor and
        must not come back.

        Ref: stdapi/files/_core.py:list_files
        """
        stub, payloads = stored
        stub.purposes = {
            _core.file_id_s3_key(payloads[0]): "batch",
            _core.file_id_s3_key(payloads[1]): "user_data",
            _core.file_id_s3_key(payloads[2]): "batch",
        }

        records, _has_more = await _core.list_files(
            payloads[2], None, 100, "desc", "batch"
        )

        assert [r.file_id for r in records] == [payloads[0]]


@pytest.mark.local
class TestListFilesEnvelopeCursorsUnit:
    """GET /v1/files reports the page's edge IDs (unit, stubbed storage).

    ``first_id``/``last_id`` are what a client feeds back as the next ``after``
    cursor, so they must be the prefixed IDs of the page's own first and last
    entries, and a defined empty-string sentinel when the page is empty.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/routes/openai_files.py:list_files_endpoint
    """

    @staticmethod
    def _stub_list_files(
        monkeypatch: pytest.MonkeyPatch, payloads: list[str], *, has_more: bool = False
    ) -> dict[str, Any]:
        """Return the records for *payloads* from the route's storage call.

        Returns:
            The dict the stub records the received query arguments into.
        """
        captured: dict[str, Any] = {}

        async def fake_list_files(
            after: str | None,
            before: str | None,
            limit: int,
            order: str,
            purpose: str | None,
        ) -> tuple[list[FileRecord], bool]:
            captured.update(
                after=after, before=before, limit=limit, order=order, purpose=purpose
            )
            return [
                FileRecord(
                    file_id=payload,
                    filename="f.txt",
                    content_type="text/plain",
                    purpose="user_data",
                    size=3,
                    created_at=datetime.now(UTC),
                    expires_at=None,
                )
                for payload in payloads
            ], has_more

        monkeypatch.setattr(openai_files_routes, "list_files", fake_list_files)
        return captured

    def test_cursors_are_the_first_and_last_ids_of_the_page(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A populated page advertises its own edge IDs and echoes ``has_more``.

        Ref: stdapi/routes/openai_files.py:list_files_endpoint
        """
        payloads = ["a" * 32, "b" * 32, "c" * 32]
        self._stub_list_files(monkeypatch, payloads, has_more=True)

        response = app_client.get("/v1/files")

        assert response.status_code == 200, response.text
        body = response.json()
        assert [f["id"] for f in body["data"]] == [f"file-{p}" for p in payloads]
        assert body["first_id"] == body["data"][0]["id"]
        assert body["last_id"] == body["data"][-1]["id"]
        assert body["has_more"] is True

    def test_empty_page_reports_empty_string_cursors(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty page still carries both cursor fields, as empty strings.

        The fields are declared non-nullable, so omitting them or sending null
        would break a client that reads them unconditionally.

        Ref: stdapi/types/openai_files.py:ListFilesResponse
        """
        self._stub_list_files(monkeypatch, [])

        response = app_client.get("/v1/files")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["data"] == []
        assert body["first_id"] == ""
        assert body["last_id"] == ""


@pytest.mark.local
class TestListFilesQueryValidationUnit:
    """GET /v1/files query bounds and defaults (unit, storage never reached).

    ``limit`` is bounded at both ends and the cursor is pattern-checked, so a bad
    page request is refused with an OpenAI-shaped 400 before any listing work.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/routes/openai_files.py:list_files_endpoint
    """

    @staticmethod
    def _reject(app_client: TestClient, params: dict[str, Any]) -> str:
        """Return the error message of a rejected listing request.

        Returns:
            The ``error.message`` string of the 400 response.
        """
        response = app_client.get("/v1/files", params=params)
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        return str(error["message"])

    def test_limit_below_one_is_rejected(self, app_client: TestClient) -> None:
        """``limit=0`` is refused: a page has to hold at least one object.

        Ref: stdapi/routes/openai_files.py:list_files_endpoint
        """
        assert "greater than or equal to 1" in self._reject(app_client, {"limit": 0})

    def test_limit_above_the_maximum_is_rejected(self, app_client: TestClient) -> None:
        """``limit=10001`` is refused rather than silently clamped.

        Ref: stdapi/routes/openai_files.py:list_files_endpoint
        """
        assert "less than or equal to 10000" in self._reject(
            app_client, {"limit": 10001}
        )

    def test_malformed_cursor_is_rejected(self, app_client: TestClient) -> None:
        """An ``after`` value that is not a file ID is refused before any listing.

        Ref: stdapi/types/__init__.py:FILE_ID_PATTERN
        """
        assert "pattern" in self._reject(app_client, {"after": "not-a-file-id"})

    @pytest.mark.parametrize("limit", [1, 10000])
    def test_limit_bounds_themselves_are_accepted(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch, limit: int
    ) -> None:
        """Both ends of the accepted range are inclusive and reach the listing.

        Ref: stdapi/routes/openai_files.py:list_files_endpoint
        """
        captured = TestListFilesEnvelopeCursorsUnit._stub_list_files(monkeypatch, [])  # noqa: SLF001

        response = app_client.get("/v1/files", params={"limit": limit})

        assert response.status_code == 200, response.text
        assert captured["limit"] == limit

    def test_defaults_are_the_maximum_page_and_descending_order(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omitting both knobs lists the newest files first, up to the full 10 000.

        Ref: stdapi/routes/openai_files.py:list_files_endpoint
        """
        captured = TestListFilesEnvelopeCursorsUnit._stub_list_files(monkeypatch, [])  # noqa: SLF001

        response = app_client.get("/v1/files")

        assert response.status_code == 200, response.text
        assert captured["limit"] == 10000
        assert captured["order"] == "desc"
        assert captured["after"] is None
        assert captured["purpose"] is None


@pytest.mark.local
class TestOpenAIFilesPurposeValidationUnit:
    """POST /v1/files validates ``purpose`` against the OpenAI enum (unit, no AWS).

    The stored purpose is what the listing filter matches on, so an unrecognised
    value would produce a file no ``purpose`` filter can ever return.

    Ref: https://developers.openai.com/api/reference/resources/files
         stdapi/types/openai_files.py:FilePurpose
    """

    @pytest.mark.parametrize(
        "purpose", ["assistants", "batch", "fine-tune", "vision", "user_data", "evals"]
    )
    def test_every_documented_purpose_is_accepted(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch, purpose: str
    ) -> None:
        """All six upstream purposes reach storage and are echoed back unchanged.

        Ref: stdapi/types/openai_files.py:FilePurpose
        """
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            openai_files_routes, "upload_file", _recording_upload_file(calls)
        )

        response = app_client.post(
            "/v1/files",
            files={"file": ("t.txt", b"hello", "text/plain")},
            data={"purpose": purpose},
        )

        assert response.status_code == 200, response.text
        assert response.json()["purpose"] == purpose
        assert [call["purpose"] for call in calls] == [purpose]

    def test_unknown_purpose_is_rejected_on_the_multipart_form(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unrecognised form ``purpose`` is refused, and nothing is stored.

        The message lists the accepted values so the caller can correct the
        request without consulting the schema.

        Ref: stdapi/routes/openai_files.py:upload
        """
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            openai_files_routes, "upload_file", _recording_upload_file(calls)
        )

        response = app_client.post(
            "/v1/files",
            files={"file": ("t.txt", b"hello", "text/plain")},
            data={"purpose": "not_a_purpose"},
        )

        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "purpose" in error["message"]
        assert "'user_data'" in error["message"], error
        assert calls == [], "a rejected purpose must not reach storage"

    def test_unknown_purpose_is_rejected_on_the_json_body(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The JSON body enforces the same enum as the multipart form.

        Ref: stdapi/types/openai_files.py:FileUploadJsonBody
        """
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            openai_files_routes, "upload_file", _recording_upload_file(calls)
        )

        response = app_client.post(
            "/v1/files",
            json={
                "file": "data:text/plain;base64,aGVsbG8=",
                "purpose": "not_a_purpose",
            },
        )

        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "purpose" in error["message"]
        assert calls == [], "a rejected purpose must not reach storage"

    def test_omitted_purpose_defaults_to_assistants(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A body with no ``purpose`` stores ``assistants``, matching upstream's default.

        Ref: stdapi/routes/openai_files.py:upload
        """
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            openai_files_routes, "upload_file", _recording_upload_file(calls)
        )

        response = app_client.post(
            "/v1/files", files={"file": ("t.txt", b"hello", "text/plain")}
        )

        assert response.status_code == 200, response.text
        assert [call["purpose"] for call in calls] == ["assistants"]


@pytest.mark.local
class TestOpenAIFilesExpiresAfterAnchorUnit:
    """POST /v1/files accepts ``created_at`` as the only expiry anchor (unit, no AWS).

    The anchor names what the TTL is counted from; the gateway only ever counts
    from creation time, so any other anchor would silently change the meaning of
    ``expires_after[seconds]``. Both spellings are checked: the OpenAI SDK sends
    the bracketed ``expires_after[anchor]``, which FastAPI's Form aliasing does
    not bind, so the route reads it back off the parsed form itself.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/routes/openai_files.py:upload
    """

    def test_unsupported_anchor_is_rejected_on_the_multipart_form(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``updated_at`` is refused, naming the anchor the API does support.

        Ref: stdapi/routes/openai_files.py:upload
        """
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            openai_files_routes, "upload_file", _recording_upload_file(calls)
        )

        response = app_client.post(
            "/v1/files",
            files={"file": ("t.txt", b"hello", "text/plain")},
            data={
                "purpose": "assistants",
                "expires_after_anchor": "updated_at",
                "expires_after_seconds": "3600",
            },
        )

        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "expires_after_anchor" in error["message"]
        assert "'created_at'" in error["message"], error
        assert calls == [], "a rejected anchor must not reach storage"

    def test_unsupported_anchor_is_rejected_in_the_sdk_bracketed_form(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``expires_after[anchor]`` — the only spelling a real client sends — is validated.

        FastAPI's ``Form(validation_alias=...)`` does not bind the bracketed name,
        so without the route's own fallback this request is accepted with any
        anchor at all and the file quietly expires from the wrong instant.

        Ref: stdapi/routes/openai_files.py:upload
        """
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            openai_files_routes, "upload_file", _recording_upload_file(calls)
        )

        response = app_client.post(
            "/v1/files",
            files={"file": ("t.txt", b"hello", "text/plain")},
            data={
                "purpose": "assistants",
                "expires_after[anchor]": "updated_at",
                "expires_after[seconds]": "3600",
            },
        )

        assert response.status_code == 400, response.text
        assert "'created_at'" in response.json()["error"]["message"]
        assert calls == [], "a rejected anchor must not reach storage"

    def test_supported_anchor_is_accepted_in_the_sdk_bracketed_form(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bracketed anchor the API does support still goes through with its TTL.

        Ref: stdapi/routes/openai_files.py:upload
        """
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            openai_files_routes, "upload_file", _recording_upload_file(calls)
        )

        response = app_client.post(
            "/v1/files",
            files={"file": ("t.txt", b"hello", "text/plain")},
            data={
                "purpose": "assistants",
                "expires_after[anchor]": "created_at",
                "expires_after[seconds]": "3600",
            },
        )

        assert response.status_code == 200, response.text
        assert calls
        assert calls[0]["expires_after"] == 3600

    def test_unsupported_anchor_is_rejected_on_the_json_body(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The JSON body enforces the same single anchor as the multipart form.

        Ref: stdapi/types/openai_files.py:FileUploadJsonBody
        """
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            openai_files_routes, "upload_file", _recording_upload_file(calls)
        )

        response = app_client.post(
            "/v1/files",
            json={
                "file": "data:text/plain;base64,aGVsbG8=",
                "expires_after_anchor": "updated_at",
                "expires_after_seconds": 3600,
            },
        )

        assert response.status_code == 400, response.text
        assert "expires_after_anchor" in response.json()["error"]["message"]
        assert calls == []

    def test_created_at_anchor_is_accepted_with_its_ttl(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The supported anchor is not rejected, and the TTL beside it is honoured.

        Ref: stdapi/routes/openai_files.py:upload
        """
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            openai_files_routes, "upload_file", _recording_upload_file(calls)
        )

        response = app_client.post(
            "/v1/files",
            json={
                "file": "data:text/plain;base64,aGVsbG8=",
                "expires_after_anchor": "created_at",
                "expires_after_seconds": 3600,
            },
        )

        assert response.status_code == 200, response.text
        assert [call["expires_after"] for call in calls] == [3600]


@pytest.mark.local
class TestOpenAIFilesJsonBodyReferenceSourcesUnit:
    """POST /v1/files with a URL-shaped ``file`` value (unit, nothing is fetched).

    A JSON ``file`` may reference remote content instead of carrying it inline;
    the reference has to be recognised as such (a URL fetched server-side, an S3
    URI checked against the allowlist) rather than mistaken for inline base64.

    Ref: https://stdapi.ai/api_openai_files/
         stdapi/types/openai_files.py:FileUploadJsonBody
    """

    def test_https_url_reaches_storage_as_a_url_reference(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An HTTPS URL is kept as a remote reference, not decoded as base64.

        The upload receives an input whose representation is the URL itself, with
        the query string redacted so a signed URL never reaches a log.

        Ref: stdapi/input_file.py:InputFile._normalize_and_detect_origin
        """
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            openai_files_routes, "upload_file", _recording_upload_file(calls)
        )

        response = app_client.post(
            "/v1/files",
            json={
                "file": "https://example.com/document.pdf?signature=secret",
                "purpose": "assistants",
            },
        )

        assert response.status_code == 200, response.text
        (call,) = calls
        assert repr(call["file"]).startswith("https://example.com/document.pdf")
        assert "secret" not in repr(call["file"])

    def test_s3_uri_outside_the_allowlist_is_rejected(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ``s3://`` URI naming an unlisted bucket is refused with a 400.

        This is the SSRF guard for S3 references: without it any caller could have
        the gateway read an arbitrary bucket with the server's own credentials.

        Ref: stdapi/input_file.py:InputFile._normalize_and_detect_origin
        """
        calls: list[dict[str, Any]] = []
        monkeypatch.setattr(
            openai_files_routes, "upload_file", _recording_upload_file(calls)
        )

        response = app_client.post(
            "/v1/files",
            json={"file": "s3://an-unconfigured-external-bucket-xyz/key.pdf"},
        )

        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "an-unconfigured-external-bucket-xyz" in error["message"], error
        assert calls == []


@pytest.mark.local
class TestMultipartIdDerivationUnit:
    """An upload session and the file it produces share one payload (unit, stubbed S3).

    ``upload_<payload>`` becomes ``file-<payload>``, which is what lets any
    instance resolve the finished file's storage location from the upload ID alone
    with no shared state; a session whose file ID were minted independently would
    break that.

    Ref: https://developers.openai.com/api/reference/resources/uploads
         stdapi/files/_multipart.py:_multipart_ids_from_bucket
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

    def test_the_pair_is_one_payload_under_two_prefixes(self) -> None:
        """The generated upload ID is the file payload with an ``upload_`` prefix.

        Ref: stdapi/files/_multipart.py:_multipart_ids_from_bucket
        """
        upload_id, payload = _multipart._multipart_ids_from_bucket("bucket")  # noqa: SLF001

        assert upload_id == f"upload_{payload}"
        assert len(payload) == 32
        assert _multipart._file_id_from_upload_id(upload_id) == payload  # noqa: SLF001

    async def test_a_session_exposes_the_derived_file_id_and_key(
        self, stub_s3: _StubCompleteS3Client
    ) -> None:
        """A created session already knows the ID and storage key of its future file.

        Ref: stdapi/files/_multipart.py:create_multipart_session
        """
        session = await _multipart.create_multipart_session(
            "f.bin", "text/plain", "assistants", 1
        )

        assert session.upload_id.startswith("upload_")
        assert session.file_id == session.upload_id.removeprefix("upload_")
        assert session.s3_key == _core.file_id_s3_key(session.file_id)

    async def test_completion_returns_the_file_derived_from_the_upload_id(
        self, stub_s3: _StubCompleteS3Client
    ) -> None:
        """The assembled file carries the session's payload, so ``upload_X`` gives ``file-X``.

        Ref: stdapi/files/_multipart.py:complete_multipart_session
        """
        session = await _multipart.create_multipart_session(
            "f.bin", "text/plain", "assistants", 7
        )
        part_1 = _multipart._make_part_id(session.upload_id, 1)  # noqa: SLF001
        stub_s3.parts = {1: ("etag-1", 7)}

        completed_session, record = await _multipart.complete_multipart_session(
            session.upload_id, [part_1]
        )

        assert completed_session.upload_id == session.upload_id
        assert record.file_id == _multipart._file_id_from_upload_id(session.upload_id)  # noqa: SLF001

    async def test_a_part_id_from_another_fingerprint_is_refused(
        self, stub_s3: _StubCompleteS3Client
    ) -> None:
        """A part ID whose embedded fingerprint is not this session's is refused.

        The rejection names both the part and the upload it was offered to, and
        happens before the assembly call, so a mis-addressed part can never join
        another session's file.

        Ref: stdapi/files/_multipart.py:_extract_part_number
        """
        session = await _multipart.create_multipart_session(
            "f.bin", "text/plain", "assistants", 7
        )
        foreign_part = "part_" + "a" * 32
        stub_s3.parts = {1: ("etag-1", 7)}

        with pytest.raises(ApiError) as exc_info:
            await _multipart.complete_multipart_session(
                session.upload_id, [foreign_part]
            )

        assert exc_info.value.status == 400
        message = str(exc_info.value)
        assert foreign_part in message
        assert session.upload_id in message
        assert stub_s3.complete_called is False


@pytest.mark.local
class TestCompleteMultipartSessionMissingPartUnit:
    """Completing with a part number that was never uploaded (unit, stubbed S3).

    A part ID can carry this session's own fingerprint and still name a part the
    session does not hold — a caller that dropped an ``add part`` response, or
    retried one that never landed. It has to be a 400 naming the part rather than
    an assembly attempt that fails deeper down.

    Ref: https://developers.openai.com/api/reference/resources/uploads
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

    async def test_never_uploaded_part_number_is_rejected(
        self, stub_s3: _StubCompleteS3Client
    ) -> None:
        """A valid part ID for a number the session never received is refused.

        The fingerprint check passes, so this exercises the missing-part branch
        rather than the "does not belong to this upload" one: the message says the
        part was not uploaded and names its number.

        Ref: stdapi/files/_multipart.py:complete_multipart_session
        """
        min_size = _multipart._MIN_PART_SIZE  # noqa: SLF001
        session = await _multipart.create_multipart_session(
            "f.bin", "text/plain", "assistants", min_size + 10
        )
        part_1 = _multipart._make_part_id(session.upload_id, 1)  # noqa: SLF001
        part_2 = _multipart._make_part_id(session.upload_id, 2)  # noqa: SLF001
        stub_s3.parts = {1: ("etag-1", min_size)}

        with pytest.raises(ApiError) as exc_info:
            await _multipart.complete_multipart_session(
                session.upload_id, [part_1, part_2]
            )

        assert exc_info.value.status == 400
        message = str(exc_info.value)
        assert "not uploaded" in message
        assert part_2 in message, "the rejection must name the missing part"
        assert "does not belong" not in message
        assert stub_s3.complete_called is False

    async def test_a_held_part_still_completes(
        self, stub_s3: _StubCompleteS3Client
    ) -> None:
        """The same session completes once the part it names is actually held.

        Without this counterpart the rejection above could come from any unrelated
        failure in the completion path.

        Ref: stdapi/files/_multipart.py:complete_multipart_session
        """
        session = await _multipart.create_multipart_session(
            "f.bin", "text/plain", "assistants", 10
        )
        part_1 = _multipart._make_part_id(session.upload_id, 1)  # noqa: SLF001
        stub_s3.parts = {1: ("etag-1", 10)}

        await _multipart.complete_multipart_session(session.upload_id, [part_1])

        assert stub_s3.complete_called is True


class _StubMissingS3Client:
    """Stub S3 client whose ``HeadObject`` always reports the key as absent."""

    def __init__(self, code: str) -> None:
        self.code = code

    async def head_object(self, **_kwargs: object) -> dict[str, Any]:
        """Raise the configured not-found ``ClientError``."""
        raise make_client_error(self.code, "HeadObject")


@pytest.mark.local
@pytest.mark.parametrize("code", ["404", "NoSuchKey"])
class TestMissingFileIsNotFoundUnit:
    """A well-formed ID whose S3 object is gone answers 404, not 500 (unit, stubbed S3).

    The ID carries its own bucket fingerprint, so an ID for a deleted — or never
    created — file passes every format check and only ``HeadObject`` can tell it
    is missing. S3 reports that with two different codes depending on the caller's
    permissions (``404`` and ``NoSuchKey``); a code that fell through would surface
    as a 500, which the OpenAI SDK retries instead of reporting to the user.

    Ref: https://docs.aws.amazon.com/AmazonS3/latest/API/API_HeadObject.html
         stdapi/files/_core.py:_get_file_impl
         stdapi/api_errors.py:FileNotExistError
    """

    @pytest.fixture
    def file_id(self, monkeypatch: pytest.MonkeyPatch, code: str) -> str:
        """Point the Files API at a stubbed bucket holding nothing.

        Returns:
            A well-formed file ID for that bucket.
        """
        monkeypatch.setattr(_core, "get_client", lambda *_: _StubMissingS3Client(code))
        monkeypatch.setattr(_core, "_require_bucket", lambda: "bucket")
        monkeypatch.setattr(_core, "BUCKET_TO_REGION", {"bucket": "us-east-1"})
        return f"file-{_core.encode_id_payload('bucket')}"

    def test_metadata_lookup_reports_not_found(
        self, app_client: TestClient, file_id: str
    ) -> None:
        """GET /v1/files/{id} answers 404 and names the file it could not find.

        The message carries the ID payload without its ``file-`` prefix, which is
        what identifies the object to the caller.
        """
        response = app_client.get(f"/v1/files/{file_id}")

        assert response.status_code == 404, response.text
        error = response.json()["error"]
        assert error["code"] == "not_found"
        assert error["type"] == "invalid_request_error"
        assert file_id.removeprefix("file-") in error["message"], error

    def test_content_download_reports_not_found(
        self, app_client: TestClient, file_id: str
    ) -> None:
        """GET /v1/files/{id}/content answers 404 rather than streaming an empty body.

        The download route resolves the record before opening the stream, so it
        must fail the same way the metadata route does.
        """
        response = app_client.get(f"/v1/files/{file_id}/content")

        assert response.status_code == 404, response.text
        assert response.json()["error"]["code"] == "not_found"


@pytest.mark.local
class TestFileContentDownloadHardening:
    """Browser-safety headers on ``GET /v1/files/{id}/content`` (unit, stubbed storage).

    The stored content type comes from the uploading client (``mime_type`` on an
    upload, the multipart part otherwise) and is echoed back verbatim, so the
    gateway would otherwise serve attacker-supplied bytes as active content on its
    own origin. The declared type is kept for API clients while
    ``Content-Disposition`` and ``X-Content-Type-Options`` deny both inline
    rendering and MIME sniffing.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/X-Content-Type-Options
         stdapi/routes/openai_files.py:get_content
    """

    def test_html_content_is_served_as_a_non_sniffable_attachment(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``text/html`` file streams back verbatim but cannot be rendered by a browser.

        The body and the declared type must be untouched; only the two hardening
        headers are added, which is what turns a stored payload into a download.
        """
        payload = b"<script>alert(document.domain)</script>"

        async def _fake_get_file_content(_: str) -> tuple[AsyncIterator[bytes], str]:
            async def _stream() -> AsyncIterator[bytes]:
                yield payload

            return _stream(), "text/html"

        monkeypatch.setattr(
            openai_files_routes, "get_file_content", _fake_get_file_content
        )
        response = app_client.get(f"/v1/files/file-{'b' * 32}/content")

        assert response.status_code == 200, response.text
        assert response.content == payload
        assert response.headers["content-type"].startswith("text/html")
        assert response.headers["content-disposition"] == "attachment"
        assert response.headers["x-content-type-options"] == "nosniff"
