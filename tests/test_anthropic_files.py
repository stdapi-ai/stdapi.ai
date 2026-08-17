"""Tests for the Anthropic-compatible Files API endpoints.

Anthropic's Files API is explicitly unavailable on Amazon Bedrock, so the gateway
implements ``/v1/files`` itself on S3. Only the payload field names
(``id`` / ``type`` / ``filename`` / ``mime_type`` / ``size_bytes`` / ``created_at`` /
``downloadable``) and the cursor envelope follow upstream; the storage semantics —
including files being downloadable, which upstream refuses for uploaded files — are
gateway behavior.

Anthropic's upload accepts no ``expires_after``, so expiry is covered here only
for the listing, which shares its namespace — and its storage layer — with the
OpenAI route that does set expiries; expired retrieval is exercised through
``TestOpenAIFiles.test_expired_file_returns_404`` instead.

Ref: https://platform.claude.com/docs/en/build-with-claude/files
     stdapi/routes/anthropic_files.py:upload
     stdapi/routes/anthropic_files.py:_to_file_metadata
"""

import io
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from anthropic import Anthropic
from anthropic import NotFoundError as AnthropicNotFoundError

from stdapi import input_file as input_file_mod
from stdapi.aws_s3 import BUCKET_TO_REGION
from stdapi.config import SETTINGS
from stdapi.files import FileRecord, _core
from stdapi.routes import anthropic_files

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

    from anthropic.types.beta import FileMetadata
    from openai import OpenAI
    from starlette.testclient import TestClient

    from stdapi.input_file import InputFile

#: The Files API is one namespace shared by the whole account, and the cursor
#: pagination tests here read it across two requests. Without a group,
#: ``--dist=loadgroup`` spreads even a single module's tests across workers, so an
#: upload from a sibling test can land between those two requests and shift the page.
pytestmark = pytest.mark.xdist_group("anthropic_files")

#: Simple plain-text file bytes for general tests.
_TEXT_FILE: bytes = b"The capital of France is Paris."

#: Bucket added to the input allowlist so an ``s3://`` body reaches the resolver.
_ALLOWED_BUCKET: str = "stdapi-test-accepted-bucket"


@pytest.fixture
def upload_file(
    anthropic_client: Anthropic,
) -> Iterator[Callable[[str, bytes, str], FileMetadata]]:
    """Upload files through the Anthropic Files API, deleting them at teardown.

    Every upload creates a real S3 object, so the teardown runs even when the test
    body fails; a delete of an already-deleted file is ignored.
    """
    uploaded: list[FileMetadata] = []

    def _upload(filename: str, content: bytes, mime_type: str) -> FileMetadata:
        metadata = anthropic_client.beta.files.upload(
            file=(filename, io.BytesIO(content), mime_type)
        )
        uploaded.append(metadata)
        return metadata

    yield _upload

    for metadata in uploaded:
        # A test may have deleted the file itself.
        with suppress(AnthropicNotFoundError):
            anthropic_client.beta.files.delete(metadata.id)


class TestAnthropicFiles:
    """The gateway's S3-backed Anthropic ``/v1/files`` upload, list, retrieve and delete routes.

    Ref: https://platform.claude.com/docs/en/build-with-claude/files
         stdapi/routes/anthropic_files.py:upload
    """

    @pytest.fixture(autouse=True)
    def _skip_on_bedrock(self, is_bedrock_direct: bool) -> None:
        """Skip all tests in this class when running against AWS Bedrock directly.

        The Files API is not available via the AnthropicBedrock client.
        """
        if is_bedrock_direct:
            pytest.skip("Files API not available on Bedrock")

    # --- Upload ---

    def test_upload_returns_file_object(
        self,
        upload_file: Callable[[str, bytes, str], FileMetadata],
        sample_pdf_file: bytes,
    ) -> None:
        """A multipart upload returns ``FileMetadata`` echoing the filename, MIME type and byte size.

        ``size_bytes`` equal to the posted byte count is what proves the body was stored
        verbatim rather than re-encoded, and the ``file_`` prefix is the ID form the
        Messages route accepts in ``{"type": "file", "file_id": ...}`` sources.

        Ref: stdapi/routes/anthropic_files.py:_to_file_metadata
        """
        result = upload_file("test.pdf", sample_pdf_file, "application/pdf")
        assert result.id.startswith("file_")
        assert result.id != "file_"
        assert result.type == "file"
        assert result.size_bytes == len(sample_pdf_file)
        assert result.created_at.tzinfo is not None, (
            f"created_at is not an RFC 3339 instant: {result.created_at!r}"
        )
        assert result.mime_type == "application/pdf"
        assert result.filename == "test.pdf"

    def test_get_metadata(
        self,
        anthropic_client: Anthropic,
        upload_file: Callable[[str, bytes, str], FileMetadata],
        sample_pdf_file: bytes,
    ) -> None:
        """Retrieving an uploaded file by ID returns the same metadata the upload reported.

        Ref: stdapi/routes/anthropic_files.py:retrieve_file
        """
        uploaded = upload_file("meta.pdf", sample_pdf_file, "application/pdf")
        retrieved = anthropic_client.beta.files.retrieve_metadata(uploaded.id)
        assert retrieved.id == uploaded.id
        assert retrieved.type == "file"
        assert retrieved.size_bytes == uploaded.size_bytes
        assert retrieved.size_bytes == len(sample_pdf_file)
        assert retrieved.filename == uploaded.filename
        assert retrieved.mime_type == uploaded.mime_type
        assert retrieved.created_at == uploaded.created_at

    # --- List ---

    def test_list_files(
        self,
        anthropic_client: Anthropic,
        upload_file: Callable[[str, bytes, str], FileMetadata],
    ) -> None:
        """Listed entries carry each file's own metadata and the page's edge IDs as cursors.

        The listed metadata is rebuilt from the S3 objects rather than replayed from the
        upload response, so filename and size must survive the round trip. The default
        page is the most recently created files, so two fresh uploads are on it whatever
        else the workspace already holds — an oldest-first listing would push them off
        it entirely once there are more than ``limit`` files.

        Ref: https://platform.claude.com/docs/en/api/beta/files/list
             stdapi/routes/anthropic_files.py:list_files_endpoint
        """
        f1 = upload_file("lst1.txt", _TEXT_FILE, "text/plain")
        f2 = upload_file("lst2.txt", _TEXT_FILE, "text/plain")
        page = anthropic_client.beta.files.list()
        by_id = {f.id: f for f in page.data}
        assert f1.id in by_id
        assert f2.id in by_id
        for uploaded in (f1, f2):
            listed = by_id[uploaded.id]
            assert listed.filename == uploaded.filename
            assert listed.size_bytes == len(_TEXT_FILE)
            assert listed.type == "file"
        ids = [f.id for f in page.data]
        assert ids.index(f2.id) < ids.index(f1.id), (
            f"the later upload must be listed first, got {ids}"
        )
        assert page.first_id == page.data[0].id
        assert page.last_id == page.data[-1].id

    def test_anthropic_list_after_id(
        self,
        anthropic_client: Anthropic,
        upload_file: Callable[[str, bytes, str], FileMetadata],
    ) -> None:
        """``after_id`` excludes the cursor file and keeps the files that follow it in list order.

        Both targets list newest first, so "after" the newest of three uploads is the
        two older ones — the same expectation on either lane.

        Ref: https://platform.claude.com/docs/en/api/beta/files/list
             stdapi/files/_core.py:list_files
        """
        files = [upload_file(f"aft{i}.txt", _TEXT_FILE, "text/plain") for i in range(3)]
        # The first file of the page is a cursor that must exclude itself.
        page1 = anthropic_client.beta.files.list(limit=1)
        assert len(page1.data) == 1
        assert page1.has_more is True, "the three uploads should not fit in one page"
        cursor_id = page1.data[0].id
        page2 = anthropic_client.beta.files.list(after_id=cursor_id)
        ids2 = {f.id for f in page2.data}
        assert cursor_id not in ids2

        # A cursor on the earliest of our uploads in list order must retain the two others.
        after_own = anthropic_client.beta.files.list(after_id=files[2].id, limit=1000)
        ids_after_own = {f.id for f in after_own.data}
        assert files[2].id not in ids_after_own
        assert {f.id for f in files[:2]} <= ids_after_own

    def test_anthropic_list_before_id(
        self,
        anthropic_client: Anthropic,
        upload_file: Callable[[str, bytes, str], FileMetadata],
    ) -> None:
        """``before_id`` excludes the cursor file and keeps the files that precede it in list order.

        Reverse of the ``after_id`` case: both targets list newest first, so "before"
        the oldest of three uploads is the two newer ones.

        Ref: https://platform.claude.com/docs/en/api/beta/files/list
             stdapi/files/_core.py:list_files
        """
        files = [upload_file(f"bef{i}.txt", _TEXT_FILE, "text/plain") for i in range(3)]
        all_files = anthropic_client.beta.files.list(limit=100)
        assert all_files.data, "the three uploads must be listed"
        cursor_id = all_files.data[-1].id
        before_page = anthropic_client.beta.files.list(before_id=cursor_id)
        ids_before = {f.id for f in before_page.data}
        assert cursor_id not in ids_before

        # A cursor on the latest of our uploads in list order must retain the two others.
        before_own = anthropic_client.beta.files.list(before_id=files[0].id, limit=1000)
        ids_before_own = {f.id for f in before_own.data}
        assert files[0].id not in ids_before_own
        assert {f.id for f in files[1:]} <= ids_before_own

    # --- Delete ---

    def test_delete(
        self,
        anthropic_client: Anthropic,
        upload_file: Callable[[str, bytes, str], FileMetadata],
    ) -> None:
        """Deleting a file confirms the ID and makes later retrievals 404.

        Ref: https://platform.claude.com/docs/en/api/errors
             stdapi/routes/anthropic_files.py:delete_file_endpoint
        """
        f = upload_file("delme.txt", _TEXT_FILE, "text/plain")
        result = anthropic_client.beta.files.delete(f.id)
        assert result.type == "file_deleted"
        assert result.id == f.id
        with pytest.raises(AnthropicNotFoundError) as excinfo:
            anthropic_client.beta.files.retrieve_metadata(f.id)
        assert excinfo.value.status_code == 404
        assert excinfo.value.type == "not_found_error"

    # --- Content ---

    @pytest.mark.gateway("the official API only allows downloading API-created files")
    def test_download_content(
        self,
        anthropic_client: Anthropic,
        upload_file: Callable[[str, bytes, str], FileMetadata],
    ) -> None:
        """Uploaded bytes are served back byte-for-byte by ``/v1/files/{id}/content``.

        Upstream marks uploaded files ``downloadable: false`` and rejects downloading them
        with a 400; the gateway stores them on S3 and always reports them downloadable,
        which is why this test can only run against the gateway.

        Ref: stdapi/routes/anthropic_files.py:get_content
             stdapi/routes/anthropic_files.py:_to_file_metadata
        """
        content = b"Anthropic Files API content test!"
        f = upload_file("dl.txt", content, "text/plain")
        assert f.downloadable is True
        assert f.size_bytes == len(content)
        downloaded = anthropic_client.beta.files.download(f.id).read()
        assert downloaded == content

    # --- Error cases ---

    def test_not_found(self, anthropic_client: Anthropic) -> None:
        """Retrieving or deleting an unknown file ID yields a 404 ``not_found_error``.

        The ID is well-formed (``file_`` plus 32 characters) so it passes the route's path
        pattern and reaches the storage lookup instead of failing validation with a 400.

        Ref: https://platform.claude.com/docs/en/api/errors
             stdapi/api_providers/anthropic.py:_format_error
        """
        fake_id = "file_" + "a" * 32
        with pytest.raises(AnthropicNotFoundError) as retrieve_exc:
            anthropic_client.beta.files.retrieve_metadata(fake_id)
        assert retrieve_exc.value.status_code == 404
        assert retrieve_exc.value.type == "not_found_error"

        with pytest.raises(AnthropicNotFoundError) as delete_exc:
            anthropic_client.beta.files.delete(fake_id)
        assert delete_exc.value.status_code == 404
        assert delete_exc.value.type == "not_found_error"

    # --- Chat integration ---

    @pytest.mark.gateway("this endpoint does not accept `file` document sources")
    def test_file_in_anthropic_message(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        upload_file: Callable[[str, bytes, str], FileMetadata],
        sample_pdf_file: bytes,
    ) -> None:
        """A ``document`` block sourced from ``{"type": "file", "file_id": ...}`` is resolved and answered.

        The gateway fetches the stored object and inlines it as a Bedrock Converse document
        block, so a non-empty text answer plus non-zero input tokens is what shows the PDF
        actually reached the model rather than being dropped.

        Ref: https://platform.claude.com/docs/en/build-with-claude/files
             stdapi/models/chat/_adapters/_anthropic_message.py:translate_request
        """
        f = upload_file("doc.pdf", sample_pdf_file, "application/pdf")
        response = anthropic_client.beta.messages.create(
            model=anthropic_chat_model,
            max_tokens=50,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "file", "file_id": f.id},
                        },
                        {
                            "type": "text",
                            "text": "Describe this document in one sentence.",
                        },
                    ],
                }
            ],
        )
        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) > 0
        assert response.content[0].type == "text"
        assert len(response.content[0].text) > 0
        assert response.stop_reason in {"end_turn", "max_tokens"}
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    @pytest.mark.gateway("this endpoint does not accept `file` image sources")
    def test_image_file_id_source_reaches_model(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_vision_model: str,
        upload_file: Callable[[str, bytes, str], FileMetadata],
        sample_image_file: bytes,
    ) -> None:
        """A ``file`` source inside an ``image`` block is resolved into a Bedrock image block.

        The image branch of the source union is a different code path from the
        document one: it infers the Bedrock image format from the stored MIME type
        instead of building a document block.  A 512x512 PNG costs far more input
        tokens than the one-sentence prompt, so the token count is what shows the
        image reached the model rather than being silently dropped.

        Ref: https://platform.claude.com/docs/en/build-with-claude/files
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_image_to_bedrock
        """
        f = upload_file("pic.png", sample_image_file, "image/png")
        response = anthropic_client.beta.messages.create(
            model=anthropic_chat_vision_model,
            max_tokens=50,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "file", "file_id": f.id}},
                        {
                            "type": "text",
                            "text": "Describe this image in one sentence.",
                        },
                    ],
                }
            ],
        )
        assert response.type == "message"
        assert response.content
        assert response.content[0].type == "text"
        assert response.content[0].text
        assert response.usage.input_tokens > 100, (
            f"a 512x512 image must dominate the prompt cost, got "
            f"{response.usage.input_tokens} input tokens"
        )


@pytest.mark.gateway("JSON body input not supported by the official Anthropic API")
class TestAnthropicFilesJsonBody:
    """POST /anthropic/v1/files with an ``application/json`` body instead of multipart.

    Accepting a JSON body (base64, data URI, HTTPS URL or S3 URI in ``file``) is a
    gateway extension for MCP tools and agents that cannot build multipart requests;
    upstream only accepts ``multipart/form-data``.

    Ref: stdapi/routes/anthropic_files.py:upload
         stdapi/types/anthropic_files.py:AnthropicFileUploadJsonBody
    """

    def test_json_body_missing_file_returns_400(
        self, openai_client: OpenAI, anthropic_client: Anthropic
    ) -> None:
        """An empty JSON body is rejected as a 400 ``invalid_request_error`` naming ``file``.

        The route validates the body with Pydantic inside ``validation_error_handler``, so
        the resulting ``RequestValidationError`` is rendered through Anthropic's envelope
        rather than FastAPI's default ``detail`` payload.

        Ref: https://platform.claude.com/docs/en/api/errors
             stdapi/api_providers/anthropic.py:_format_error
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{anthropic_client.base_url}v1/files",
            json={},
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 400
        body = response.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert "file" in body["error"]["message"].lower()

    def test_json_body_upload_with_data_uri(
        self, openai_client: OpenAI, anthropic_client: Anthropic
    ) -> None:
        """A data URI body is base64-decoded and stored with its declared MIME type.

        ``size_bytes == 11`` (the decoded ``Hello World``) is what distinguishes decoding
        the payload from storing the 16-character base64 text verbatim.

        Ref: stdapi/routes/anthropic_files.py:upload
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{anthropic_client.base_url}v1/files",
            json={"file": "data:text/plain;base64,SGVsbG8gV29ybGQ="},
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["id"].startswith("file_")
        assert body["type"] == "file"
        assert body["size_bytes"] == len(b"Hello World")
        assert body["mime_type"].startswith("text/")
        # Clean up: use the Anthropic client to delete
        delete_response = http_client.delete(
            f"{anthropic_client.base_url}v1/files/{body['id']}",
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert delete_response.status_code == 200


class TestAnthropicFilesJsonBodySources:
    """Which ``file`` string forms the ``application/json`` upload body accepts.

    The remote forms are the ones highlighted for MCP tools and agents, and they
    are the only ones that never carry their bytes in the request: the body string
    is parsed into an ``IngestInputFile`` whose content is fetched server-side. The
    storage call is stubbed here so the accepted-source contract is pinned without
    reading S3 or the network.

    Ref: stdapi/routes/anthropic_files.py:upload
         stdapi/types/anthropic_files.py:AnthropicFileUploadJsonBody
         stdapi/input_file.py:IngestInputFile
    """

    pytestmark = pytest.mark.local

    @staticmethod
    @pytest.fixture
    def uploaded_source(monkeypatch: pytest.MonkeyPatch) -> list[InputFile]:
        """Record the ``InputFile`` the route hands to the storage layer."""
        # A ``file-id:`` body resolves its bucket while the source is parsed, so
        # the Files API has to be configured for the parser to reject it itself.
        monkeypatch.setattr(SETTINGS, "aws_s3_bucket", "test-bucket")
        recorded: list[InputFile] = []

        async def _fake_upload_file(file: InputFile, *_args: object) -> FileRecord:
            recorded.append(file)
            return FileRecord(
                file_id="a" * 32,
                filename="remote.png",
                content_type="image/png",
                purpose="",
                size=11,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                expires_at=None,
            )

        monkeypatch.setattr(anthropic_files, "upload_file", _fake_upload_file)
        return recorded

    def test_https_url_body_is_handed_to_the_resolver(
        self, anthropic_app_client: TestClient, uploaded_source: list[InputFile]
    ) -> None:
        """An HTTPS URL in ``file`` is stored as a remote reference, not as its own text.

        Detecting it as a URL rather than as raw base64 is the whole difference
        between ingesting the remote document and storing the URL string itself.
        """
        response = anthropic_app_client.post(
            "/anthropic/v1/files", json={"file": "https://example.com/document.pdf"}
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"] == f"file_{'a' * 32}"
        (source,) = uploaded_source
        assert source.is_s3 is False
        assert str(source) == "https://example.com/document.pdf"

    def test_s3_uri_body_is_handed_to_the_resolver(
        self,
        anthropic_app_client: TestClient,
        uploaded_source: list[InputFile],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An ``s3://`` URI for an allowed bucket resolves to that bucket's region.

        The bucket allowlist is the SSRF guard for this input: a URI is only routed
        to S3 once its bucket is configured, and the region bound at that point is
        what the later object read is issued against.
        """
        monkeypatch.setattr(
            input_file_mod, "_ACCEPTED_BUCKETS", frozenset({_ALLOWED_BUCKET})
        )
        monkeypatch.setitem(BUCKET_TO_REGION, _ALLOWED_BUCKET, "us-east-1")

        response = anthropic_app_client.post(
            "/anthropic/v1/files",
            json={"file": f"s3://{_ALLOWED_BUCKET}/inbox/document.pdf"},
        )
        assert response.status_code == 200, response.text
        (source,) = uploaded_source
        assert source.is_s3 is True
        assert source.region == "us-east-1"

    def test_s3_uri_outside_the_allow_list_is_rejected(
        self, anthropic_app_client: TestClient, uploaded_source: list[InputFile]
    ) -> None:
        """An ``s3://`` URI for an unconfigured bucket is refused before any S3 call.

        Ref: https://platform.claude.com/docs/en/api/errors
             stdapi/input_file.py:InputFile._normalize_and_detect_origin
        """
        response = anthropic_app_client.post(
            "/anthropic/v1/files",
            json={"file": "s3://an-unconfigured-external-bucket-xyz/document.pdf"},
        )
        assert response.status_code == 400, response.text
        body = response.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert not uploaded_source, "the rejected bucket must never be read"

    def test_file_id_reference_is_rejected(
        self, anthropic_app_client: TestClient, uploaded_source: list[InputFile]
    ) -> None:
        """A ``file-id:`` reference is refused: an ingest endpoint takes content only.

        Resolving it here would clone an existing stored object into a second one,
        billing the same bytes twice, so this source is excluded from the ingest
        variant of the input parser even though the Messages route accepts it.

        Ref: https://platform.claude.com/docs/en/api/errors
             stdapi/input_file.py:IngestInputFile
        """
        response = anthropic_app_client.post(
            "/anthropic/v1/files", json={"file": f"file-id:file_{'0' * 32}"}
        )
        assert response.status_code == 400, response.text
        body = response.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert not uploaded_source, "no object may be created for a rejected source"


class TestAnthropicFileContentDownloadHardening:
    """Browser-safety headers on the Anthropic ``/v1/files/{id}/content`` download.

    ``mime_type`` is chosen by the uploading client and echoed back as the
    response ``Content-Type``, so the gateway would otherwise serve attacker-
    supplied bytes as active content on its own origin. The declared type is kept
    for API clients while ``Content-Disposition`` and ``X-Content-Type-Options``
    deny both inline rendering and MIME sniffing.

    Ref: https://platform.claude.com/docs/en/build-with-claude/files
         https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/X-Content-Type-Options
         stdapi/routes/anthropic_files.py:get_content
    """

    pytestmark = pytest.mark.local

    def test_html_content_is_served_as_a_non_sniffable_attachment(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
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

        monkeypatch.setattr(anthropic_files, "get_file_content", _fake_get_file_content)
        response = anthropic_app_client.get(
            f"/anthropic/v1/files/file_{'b' * 32}/content"
        )
        assert response.status_code == 200, response.text
        assert response.content == payload
        assert response.headers["content-type"].startswith("text/html")
        assert response.headers["content-disposition"] == "attachment"
        assert response.headers["x-content-type-options"] == "nosniff"


class _StubListS3Client:
    """Stub S3 client serving one live and one expired object to the listing scan."""

    def __init__(self, keys: list[str], expired_key: str) -> None:
        self.keys = sorted(keys)
        self.expired_key = expired_key

    async def list_objects_v2(self, **_kwargs: object) -> dict[str, Any]:
        return {"Contents": [{"Key": key} for key in self.keys], "IsTruncated": False}

    async def head_object(self, **kwargs: object) -> dict[str, Any]:
        expired = str(int(datetime.now(UTC).timestamp()) - 10)
        return {
            "ContentLength": 3,
            "LastModified": datetime.now(UTC),
            "Metadata": {
                "purpose": "user_data",
                "expires-at": expired if kwargs["Key"] == self.expired_key else "",
            },
            "ContentDisposition": 'attachment; filename="f.txt"',
            "ContentType": "text/plain",
        }


class TestAnthropicListExpiredFilesUnit:
    """Expired files stay out of the Anthropic listing (unit, stubbed S3).

    The Anthropic and OpenAI listings share one storage layer and one namespace,
    so a file given an expiry through the OpenAI upload must disappear from both
    once it passes — otherwise this route advertises an object its own retrieve
    route reports as gone.

    Ref: https://platform.claude.com/docs/en/build-with-claude/files
         stdapi/routes/anthropic_files.py:list_files_endpoint
         stdapi/files/_core.py:_head_record
    """

    pytestmark = pytest.mark.local

    def test_an_expired_file_is_absent_from_the_listing(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the live file is listed.

        Ref: stdapi/files/_core.py:list_files
        """
        payloads = sorted(_core.encode_id_payload("bucket") for _ in range(2))
        expired_key = _core.file_id_s3_key(payloads[0])
        stub = _StubListS3Client(
            [_core.file_id_s3_key(p) for p in payloads], expired_key
        )
        scheduled: list[tuple[str, str]] = []
        monkeypatch.setattr(_core, "get_client", lambda *_: stub)
        monkeypatch.setattr(_core, "_require_bucket", lambda: "bucket")
        monkeypatch.setattr(_core, "BUCKET_TO_REGION", {"bucket": "us-east-1"})
        monkeypatch.setattr(
            _core,
            "track_temporary_s3_objects",
            lambda bucket, key: scheduled.append((bucket, key)),
        )

        response = anthropic_app_client.get("/anthropic/v1/files")

        assert response.status_code == 200, response.text
        assert [f["id"] for f in response.json()["data"]] == [f"file_{payloads[1]}"]
        assert scheduled == [], "the listing leaves deletion to the retrieve path"
