"""Tests for the Anthropic-compatible Files API endpoints.

Anthropic's Files API is explicitly unavailable on Amazon Bedrock, so the gateway
implements ``/v1/files`` itself on S3. Only the payload field names
(``id`` / ``type`` / ``filename`` / ``mime_type`` / ``size_bytes`` / ``created_at`` /
``downloadable``) and the cursor envelope follow upstream; the storage semantics —
including files being downloadable, which upstream refuses for uploaded files — are
gateway behavior.

Ref: https://platform.claude.com/docs/en/build-with-claude/files
     stdapi/routes/anthropic_files.py:upload
     stdapi/routes/anthropic_files.py:_to_file_metadata
"""

import io
from typing import TYPE_CHECKING

import pytest
from anthropic import Anthropic
from anthropic import NotFoundError as AnthropicNotFoundError

if TYPE_CHECKING:
    from openai import OpenAI

#: Minimal valid PDF bytes for testing document endpoints.
_MINIMAL_PDF: bytes = (
    b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
    b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n"
    b"0000000058 00000 n \n0000000115 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
)

#: Simple plain-text file bytes for general tests.
_TEXT_FILE: bytes = b"The capital of France is Paris."


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

    def test_upload_returns_file_object(self, anthropic_client: Anthropic) -> None:
        """A multipart upload returns ``FileMetadata`` echoing the filename, MIME type and byte size.

        ``size_bytes`` equal to the posted byte count is what proves the body was stored
        verbatim rather than re-encoded, and the ``file_`` prefix is the ID form the
        Messages route accepts in ``{"type": "file", "file_id": ...}`` sources.

        Ref: stdapi/routes/anthropic_files.py:_to_file_metadata
        """
        content = _MINIMAL_PDF
        result = anthropic_client.beta.files.upload(
            file=("test.pdf", io.BytesIO(content), "application/pdf")
        )
        try:
            assert result.id.startswith("file_")
            assert result.id != "file_"
            assert result.type == "file"
            assert result.size_bytes == len(content)
            assert result.created_at.tzinfo is not None, (
                f"created_at is not an RFC 3339 instant: {result.created_at!r}"
            )
            assert result.mime_type == "application/pdf"
            assert result.filename == "test.pdf"
        finally:
            anthropic_client.beta.files.delete(result.id)

    def test_get_metadata(self, anthropic_client: Anthropic) -> None:
        """Retrieving an uploaded file by ID returns the same metadata the upload reported.

        Ref: stdapi/routes/anthropic_files.py:retrieve_file
        """
        uploaded = anthropic_client.beta.files.upload(
            file=("meta.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")
        )
        try:
            retrieved = anthropic_client.beta.files.retrieve_metadata(uploaded.id)
            assert retrieved.id == uploaded.id
            assert retrieved.type == "file"
            assert retrieved.size_bytes == uploaded.size_bytes
            assert retrieved.size_bytes == len(_MINIMAL_PDF)
            assert retrieved.filename == uploaded.filename
            assert retrieved.mime_type == uploaded.mime_type
            assert retrieved.created_at == uploaded.created_at
        finally:
            anthropic_client.beta.files.delete(uploaded.id)

    # --- List ---

    def test_list_files(self, anthropic_client: Anthropic) -> None:
        """Listed entries carry each file's own metadata and the page's edge IDs as cursors.

        The listed metadata is rebuilt from the S3 objects rather than replayed from the
        upload response, so filename and size must survive the round trip.

        Ref: stdapi/routes/anthropic_files.py:list_files_endpoint
        """
        f1 = anthropic_client.beta.files.upload(
            file=("lst1.txt", io.BytesIO(_TEXT_FILE), "text/plain")
        )
        f2 = anthropic_client.beta.files.upload(
            file=("lst2.txt", io.BytesIO(_TEXT_FILE), "text/plain")
        )
        try:
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
        finally:
            anthropic_client.beta.files.delete(f1.id)
            anthropic_client.beta.files.delete(f2.id)

    def test_anthropic_list_after_id(self, anthropic_client: Anthropic) -> None:
        """``after_id`` excludes the cursor file and keeps the files created after it.

        The gateway lists files in ascending creation order (S3 keys are UUIDv7, so
        lexicographic key order equals creation order), so the second and third upload of
        this test must both appear after a cursor set on the first one.

        Ref: stdapi/files/_core.py:list_files
        """
        files = [
            anthropic_client.beta.files.upload(
                file=(f"aft{i}.txt", io.BytesIO(_TEXT_FILE), "text/plain")
            )
            for i in range(3)
        ]
        try:
            # The oldest file overall is a cursor that must exclude itself.
            page1 = anthropic_client.beta.files.list(limit=1)
            assert len(page1.data) == 1
            assert page1.has_more is True, (
                "the three uploads should not fit in one page"
            )
            cursor_id = page1.data[0].id
            page2 = anthropic_client.beta.files.list(after_id=cursor_id)
            ids2 = {f.id for f in page2.data}
            assert cursor_id not in ids2

            # A cursor on our own first upload must retain the two later ones.
            after_own = anthropic_client.beta.files.list(
                after_id=files[0].id, limit=1000
            )
            ids_after_own = {f.id for f in after_own.data}
            assert files[0].id not in ids_after_own
            assert files[1].id in ids_after_own
            assert files[2].id in ids_after_own
        finally:
            for f in files:
                anthropic_client.beta.files.delete(f.id)

    def test_anthropic_list_before_id(self, anthropic_client: Anthropic) -> None:
        """``before_id`` excludes the cursor file and keeps the files created before it.

        Reverse of the ``after_id`` case: a cursor set on the newest of three uploads must
        return the two earlier ones and drop the cursor itself.

        Ref: stdapi/files/_core.py:list_files
        """
        files = [
            anthropic_client.beta.files.upload(
                file=(f"bef{i}.txt", io.BytesIO(_TEXT_FILE), "text/plain")
            )
            for i in range(3)
        ]
        try:
            all_files = anthropic_client.beta.files.list(limit=100)
            assert all_files.data, "the three uploads must be listed"
            cursor_id = all_files.data[-1].id
            before_page = anthropic_client.beta.files.list(before_id=cursor_id)
            ids_before = {f.id for f in before_page.data}
            assert cursor_id not in ids_before

            # A cursor on our own newest upload must retain the two earlier ones.
            before_own = anthropic_client.beta.files.list(
                before_id=files[2].id, limit=1000
            )
            ids_before_own = {f.id for f in before_own.data}
            assert files[2].id not in ids_before_own
            assert files[0].id in ids_before_own
            assert files[1].id in ids_before_own
        finally:
            for f in files:
                anthropic_client.beta.files.delete(f.id)

    # --- Delete ---

    def test_delete(self, anthropic_client: Anthropic) -> None:
        """Deleting a file confirms the ID and makes later retrievals 404.

        Ref: https://platform.claude.com/docs/en/api/errors
             stdapi/routes/anthropic_files.py:delete_file_endpoint
        """
        f = anthropic_client.beta.files.upload(
            file=("delme.txt", io.BytesIO(_TEXT_FILE), "text/plain")
        )
        result = anthropic_client.beta.files.delete(f.id)
        assert result.type == "file_deleted"
        assert result.id == f.id
        with pytest.raises(AnthropicNotFoundError) as excinfo:
            anthropic_client.beta.files.retrieve_metadata(f.id)
        assert excinfo.value.status_code == 404
        assert excinfo.value.type == "not_found_error"

    # --- Content ---

    def test_download_content(
        self, anthropic_client: Anthropic, use_official_api: bool
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
        f = anthropic_client.beta.files.upload(
            file=("dl.txt", io.BytesIO(content), "text/plain")
        )
        try:
            assert f.downloadable is True
            assert f.size_bytes == len(content)
            downloaded = anthropic_client.beta.files.download(f.id).read()
            assert downloaded == content
        finally:
            anthropic_client.beta.files.delete(f.id)

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

    def test_expired_file_returns_404(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """File expiry is not reachable through the Anthropic client, so this test only skips.

        Anthropic's upload accepts no ``expires_after``, leaving no way to age a file out
        from this surface. The gateway's shared expiry enforcement is covered by
        ``TestOpenAIFiles.test_expired_file_returns_404``, which advances the clock with
        ``patch("stdapi.files._core.now_utc_timestamp")``.

        Ref: https://platform.claude.com/docs/en/build-with-claude/files
             stdapi/files/_core.py:get_file
        """
        pytest.skip(
            "Anthropic upload has no expires_after; expiry tested via OpenAI client"
        )

    # --- Chat integration ---

    def test_file_in_anthropic_message(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
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
        f = anthropic_client.beta.files.upload(
            file=("doc.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")
        )
        try:
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
        finally:
            anthropic_client.beta.files.delete(f.id)


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
