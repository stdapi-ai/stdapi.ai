"""Tests for the Anthropic-compatible Files API endpoints.

Anthropic's Files API is explicitly unavailable on Amazon Bedrock, so the gateway
implements ``/v1/files`` itself on S3. Only the payload field names
(``id`` / ``type`` / ``filename`` / ``mime_type`` / ``size_bytes`` / ``created_at`` /
``downloadable``) and the cursor envelope follow upstream; the storage semantics —
including files being downloadable, which upstream refuses for uploaded files — are
gateway behavior.

File expiry has no coverage here: Anthropic's upload accepts no ``expires_after``,
so the shared enforcement is exercised through
``TestOpenAIFiles.test_expired_file_returns_404`` instead.

Ref: https://platform.claude.com/docs/en/build-with-claude/files
     stdapi/routes/anthropic_files.py:upload
     stdapi/routes/anthropic_files.py:_to_file_metadata
"""

import io
from contextlib import suppress
from typing import TYPE_CHECKING

import pytest
from anthropic import Anthropic
from anthropic import NotFoundError as AnthropicNotFoundError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from anthropic.types.beta import FileMetadata
    from openai import OpenAI

#: Simple plain-text file bytes for general tests.
_TEXT_FILE: bytes = b"The capital of France is Paris."


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
        upload response, so filename and size must survive the round trip.

        Ref: stdapi/routes/anthropic_files.py:list_files_endpoint
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
        assert page.first_id == page.data[0].id
        assert page.last_id == page.data[-1].id

    def test_anthropic_list_after_id(
        self,
        anthropic_client: Anthropic,
        upload_file: Callable[[str, bytes, str], FileMetadata],
        use_official_api: bool,
    ) -> None:
        """``after_id`` excludes the cursor file and keeps the files that follow it in list order.

        The two targets sort the listing in opposite directions: upstream returns files
        newest first, while the gateway lists them in ascending creation order (S3 keys are
        UUIDv7, so lexicographic key order equals creation order). The cursor is therefore
        set on the first of the three uploads in each target's own order, and the other two
        must follow it.

        Ref: https://platform.claude.com/docs/en/build-with-claude/files
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
        cursor, expected = (
            (files[2], files[:2]) if use_official_api else (files[0], files[1:])
        )
        after_own = anthropic_client.beta.files.list(after_id=cursor.id, limit=1000)
        ids_after_own = {f.id for f in after_own.data}
        assert cursor.id not in ids_after_own
        assert {f.id for f in expected} <= ids_after_own

    def test_anthropic_list_before_id(
        self,
        anthropic_client: Anthropic,
        upload_file: Callable[[str, bytes, str], FileMetadata],
        use_official_api: bool,
    ) -> None:
        """``before_id`` excludes the cursor file and keeps the files that precede it in list order.

        Reverse of the ``after_id`` case, and lane-conditional for the same reason: the
        cursor is the last of the three uploads in each target's own order — the newest one
        on the gateway (ascending), the oldest one upstream (newest first) — and the two
        others must precede it.

        Ref: https://platform.claude.com/docs/en/build-with-claude/files
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
        cursor, expected = (
            (files[0], files[1:]) if use_official_api else (files[2], files[:2])
        )
        before_own = anthropic_client.beta.files.list(before_id=cursor.id, limit=1000)
        ids_before_own = {f.id for f in before_own.data}
        assert cursor.id not in ids_before_own
        assert {f.id for f in expected} <= ids_before_own

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

    def test_download_content(
        self,
        anthropic_client: Anthropic,
        upload_file: Callable[[str, bytes, str], FileMetadata],
        use_official_api: bool,
    ) -> None:
        """Uploaded bytes are served back byte-for-byte by ``/v1/files/{id}/content``.

        Upstream marks uploaded files ``downloadable: false`` and rejects downloading them
        with a 400; the gateway stores them on S3 and always reports them downloadable,
        which is why this test can only run against the gateway.

        Ref: stdapi/routes/anthropic_files.py:get_content
             stdapi/routes/anthropic_files.py:_to_file_metadata
        """
        if use_official_api:
            pytest.skip("the official API only allows downloading API-created files")
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

    def test_file_in_anthropic_message(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        upload_file: Callable[[str, bytes, str], FileMetadata],
        sample_pdf_file: bytes,
        use_official_api: bool,
    ) -> None:
        """A ``document`` block sourced from ``{"type": "file", "file_id": ...}`` is resolved and answered.

        The gateway fetches the stored object and inlines it as a Bedrock Converse document
        block, so a non-empty text answer plus non-zero input tokens is what shows the PDF
        actually reached the model rather than being dropped.

        Ref: https://platform.claude.com/docs/en/build-with-claude/files
             stdapi/models/chat/_adapters/_anthropic_message.py:translate_request
        """
        if use_official_api:
            pytest.skip("this endpoint does not accept `file` document sources")
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

    def test_image_file_id_source_reaches_model(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_vision_model: str,
        upload_file: Callable[[str, bytes, str], FileMetadata],
        sample_image_file: bytes,
        use_official_api: bool,
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
        if use_official_api:
            pytest.skip("this endpoint does not accept `file` image sources")
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


class TestAnthropicFilesJsonBody:
    """POST /anthropic/v1/files with an ``application/json`` body instead of multipart.

    Accepting a JSON body (base64, data URI, HTTPS URL or S3 URI in ``file``) is a
    gateway extension for MCP tools and agents that cannot build multipart requests;
    upstream only accepts ``multipart/form-data``.

    Ref: stdapi/routes/anthropic_files.py:upload
         stdapi/types/anthropic_files.py:AnthropicFileUploadJsonBody
    """

    @pytest.fixture(autouse=True)
    def _skip_on_official_api(self, use_official_api: bool) -> None:
        """JSON body input is an extension not supported by the official API."""
        if use_official_api:
            pytest.skip("JSON body input not supported by the official Anthropic API")

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
