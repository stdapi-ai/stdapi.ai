"""Tests for the Anthropic-compatible Files API endpoints.

Tests cover upload, list, get, delete, content streaming, pagination,
and integration with Anthropic messages.
"""

import io
from datetime import datetime

import pytest
from anthropic import Anthropic
from anthropic import NotFoundError as AnthropicNotFoundError

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
    """Test suite for the Anthropic-compatible /v1/files endpoints."""

    @pytest.fixture(autouse=True)
    def _skip_on_bedrock(self, is_bedrock_direct: bool) -> None:
        """Skip all tests in this class when running against AWS Bedrock directly.

        The Files API is not available via the AnthropicBedrock client.
        """
        if is_bedrock_direct:
            pytest.skip("Files API not available on Bedrock")

    # --- Upload ---

    def test_upload_returns_file_object(self, anthropic_client: Anthropic) -> None:
        """Upload a file and assert all required fields are present.

        Validates:
            - id starts with 'file-'
            - type is 'file'
            - size_bytes equals uploaded file size
            - created_at is an ISO 8601 / RFC 3339 string
            - mime_type is set
        """
        content = _MINIMAL_PDF
        result = anthropic_client.beta.files.upload(
            file=("test.pdf", io.BytesIO(content), "application/pdf")
        )
        try:
            assert result.id.startswith("file_")
            assert result.type == "file"
            assert result.size_bytes == len(content)
            assert isinstance(result.created_at, (str, datetime))
            assert result.mime_type == "application/pdf"
            assert result.filename == "test.pdf"
        finally:
            anthropic_client.beta.files.delete(result.id)

    def test_get_metadata(self, anthropic_client: Anthropic) -> None:
        """Upload a file then retrieve it by ID and assert metadata matches.

        Validates:
            - Retrieved id matches uploaded id
            - size_bytes and filename are consistent
        """
        uploaded = anthropic_client.beta.files.upload(
            file=("meta.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")
        )
        try:
            retrieved = anthropic_client.beta.files.retrieve_metadata(uploaded.id)
            assert retrieved.id == uploaded.id
            assert retrieved.size_bytes == uploaded.size_bytes
            assert retrieved.filename == uploaded.filename
        finally:
            anthropic_client.beta.files.delete(uploaded.id)

    # --- List ---

    def test_list_files(self, anthropic_client: Anthropic) -> None:
        """Upload two files and assert both appear in the list response.

        Validates:
            - Both file IDs appear in the list
        """
        f1 = anthropic_client.beta.files.upload(
            file=("lst1.txt", io.BytesIO(_TEXT_FILE), "text/plain")
        )
        f2 = anthropic_client.beta.files.upload(
            file=("lst2.txt", io.BytesIO(_TEXT_FILE), "text/plain")
        )
        try:
            page = anthropic_client.beta.files.list()
            ids = {f.id for f in page.data}
            assert f1.id in ids
            assert f2.id in ids
        finally:
            anthropic_client.beta.files.delete(f1.id)
            anthropic_client.beta.files.delete(f2.id)

    def test_anthropic_list_after_id(self, anthropic_client: Anthropic) -> None:
        """Test Anthropic after_id cursor pagination.

        Validates:
            - Files returned after the cursor do not include the cursor file
        """
        files = [
            anthropic_client.beta.files.upload(
                file=(f"aft{i}.txt", io.BytesIO(_TEXT_FILE), "text/plain")
            )
            for i in range(3)
        ]
        try:
            # Get ascending list to find the oldest file
            page1 = anthropic_client.beta.files.list(limit=1)
            assert len(page1.data) == 1
            cursor_id = page1.data[0].id
            page2 = anthropic_client.beta.files.list(after_id=cursor_id)
            ids2 = {f.id for f in page2.data}
            assert cursor_id not in ids2
        finally:
            for f in files:
                anthropic_client.beta.files.delete(f.id)

    def test_anthropic_list_before_id(self, anthropic_client: Anthropic) -> None:
        """Test Anthropic before_id cursor (reverse pagination).

        Validates:
            - Files returned before the cursor do not include the cursor file
        """
        files = [
            anthropic_client.beta.files.upload(
                file=(f"bef{i}.txt", io.BytesIO(_TEXT_FILE), "text/plain")
            )
            for i in range(3)
        ]
        try:
            # Get the newest file ID as cursor — ascending order means data[-1] is newest
            all_files = anthropic_client.beta.files.list(limit=100)
            if all_files.data:
                newest_id = all_files.data[-1].id
                before_page = anthropic_client.beta.files.list(before_id=newest_id)
                ids_before = {f.id for f in before_page.data}
                assert newest_id not in ids_before
        finally:
            for f in files:
                anthropic_client.beta.files.delete(f.id)

    # --- Delete ---

    def test_delete(self, anthropic_client: Anthropic) -> None:
        """Upload then delete a file; assert 404 on subsequent metadata retrieval.

        Validates:
            - Delete response type is 'file_deleted'
            - Subsequent metadata retrieval raises NotFoundError
        """
        f = anthropic_client.beta.files.upload(
            file=("delme.txt", io.BytesIO(_TEXT_FILE), "text/plain")
        )
        result = anthropic_client.beta.files.delete(f.id)
        assert result.type == "file_deleted"
        with pytest.raises(AnthropicNotFoundError):
            anthropic_client.beta.files.retrieve_metadata(f.id)

    # --- Content ---

    def test_download_content(self, anthropic_client: Anthropic) -> None:
        """Upload bytes and download via /content; assert byte equality.

        Validates:
            - Downloaded content matches uploaded content exactly
        """
        content = b"Anthropic Files API content test!"
        f = anthropic_client.beta.files.upload(
            file=("dl.txt", io.BytesIO(content), "text/plain")
        )
        try:
            downloaded = anthropic_client.beta.files.download(f.id).read()
            assert downloaded == content
        finally:
            anthropic_client.beta.files.delete(f.id)

    # --- Error cases ---

    def test_not_found(self, anthropic_client: Anthropic) -> None:
        """Assert 404 error for get/delete of non-existent file.

        Validates:
            - NotFoundError is raised for both retrieve_metadata and delete
        """
        fake_id = "file_" + "a" * 32
        with pytest.raises(AnthropicNotFoundError):
            anthropic_client.beta.files.retrieve_metadata(fake_id)
        with pytest.raises(AnthropicNotFoundError):
            anthropic_client.beta.files.delete(fake_id)

    def test_expired_file_returns_404(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Skip: Anthropic upload has no expires_after, so expiry cannot be triggered via this client.

        Expiry enforcement is already covered by ``TestOpenAIFiles.test_expired_file_returns_404``
        which controls the clock via ``patch("stdapi.files._core.now_utc_timestamp")``.
        """
        pytest.skip(
            "Anthropic upload has no expires_after; expiry tested via OpenAI client"
        )

    # --- Chat integration ---

    def test_file_in_anthropic_message(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Upload a PDF file and reference it in an Anthropic message document block.

        Validates:
            - Message creation succeeds with a file reference
            - Response contains assistant content
        """
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
            assert len(response.content) > 0
            assert response.content[0].type == "text"
            assert len(response.content[0].text) > 0
        finally:
            anthropic_client.beta.files.delete(f.id)
