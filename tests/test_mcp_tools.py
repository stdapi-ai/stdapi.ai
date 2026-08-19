"""Every exposed MCP tool called end to end, via the official MCP client.

Unlike the JSON-RPC plumbing tests in ``test_mcp.py``, these travel the exact
path a real MCP client uses: the ``mcp`` SDK's streamable-HTTP client performs
the ``initialize`` handshake and ``tools/call`` requests against ``/mcp``, and
``fastapi_mcp`` re-enters the gateway over its in-process ASGI transport. All
53 exposed tools get at least one call (stored/derived resources are created
through MCP too), so a schema or transport regression on any tool is caught by
the suite. The costly lanes keep their usual gates: image tools require
``--expensive`` and the video lifecycle requires ``--video``.

Every MCP interaction runs inside the TestClient's portal event loop: the MCP
session manager's task group binds to the loop of the first ``/mcp`` request
(the portal loop, where the app lifespan runs), and touching its anyio streams
from per-test pytest-asyncio loops raises ``RuntimeError``.

Multipart routes (audio, images, files) are called through their JSON bodies
(base64 data URIs), which is the documented MCP/agent request format.

Ref: stdapi/mcp.py:mount_mcp
     https://modelcontextprotocol.io/docs/concepts/transports#streamable-http
"""

from __future__ import annotations

from base64 import b64decode
from json import loads
from typing import TYPE_CHECKING, Any

import pytest
from httpx import ASGITransport, AsyncClient

from stdapi.config import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Callable

    from httpx import Auth, Timeout
    from mcp.types import CallToolResult
    from starlette.testclient import TestClient

    #: Callable running one tool call over a fresh MCP session.
    type McpCall = Callable[[str, dict[str, Any]], CallToolResult]

#: A 1x1 PNG, the smallest binary file the storage tools can be exercised with.
_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGA"
    "hKmMIQAAAABJRU5ErkJggg=="
)

#: These tests speak the streamable-HTTP transport, the only one the SDK mounts here.
pytestmark = pytest.mark.skipif(
    not SETTINGS.enable_mcp_streamable_http, reason="MCP streamable HTTP is not enabled"
)


async def _call_in_loop(
    client: TestClient, api_key: str, name: str, arguments: dict[str, Any]
) -> CallToolResult:
    """Open an MCP session against the app and perform one tool call.

    The SDK client's HTTP layer is bound to the in-process app with an ASGI
    transport, so the whole MCP stack (handshake, session management, tool
    dispatch, internal API re-entry) runs without a network socket.
    """
    from mcp import ClientSession  # noqa: PLC0415
    from mcp.client.streamable_http import streamablehttp_client  # noqa: PLC0415

    def _asgi_client(
        headers: dict[str, str] | None = None,
        timeout: Timeout | None = None,
        auth: Auth | None = None,
    ) -> AsyncClient:
        """Build the httpx client the MCP SDK uses, bound to the ASGI app."""
        return AsyncClient(
            transport=ASGITransport(app=client.app),
            base_url="http://mcp-test",
            headers=headers,
            timeout=timeout if timeout is not None else 300,
            auth=auth,
        )

    async with (
        streamablehttp_client(
            "http://mcp-test/mcp",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=300,
            httpx_client_factory=_asgi_client,
        ) as (read_stream, write_stream, _),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        return await session.call_tool(name, arguments)


@pytest.fixture
def mcp_call(local_test_client: TestClient, api_key: str) -> McpCall:
    """Run one MCP tool call inside the app's portal event loop."""
    portal = local_test_client.portal
    assert portal is not None, "TestClient context is not entered"

    def call(name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Perform the tool call and return its result."""
        result: CallToolResult = portal.call(
            _call_in_loop, local_test_client, api_key, name, arguments
        )
        return result

    return call


def _json_result(result: CallToolResult) -> Any:  # noqa: ANN401
    """Assert a tool call succeeded and parse its JSON text content."""
    assert not result.isError, result.content
    assert result.content, "Tool returned no content"
    return loads(result.content[0].text)  # type: ignore[union-attr]


class TestModelDiscoveryTools:
    """Catalog tools: the first calls the MCP workflow documents for agents."""

    def test_search_models(self, mcp_call: McpCall) -> None:
        """search_models filtered by MCP tool name returns extended model details.

        Ref: stdapi/routes/core_models.py:search_models
        """
        payload = _json_result(
            mcp_call("search_models", {"route": "openai_chat_completion"})
        )
        assert payload
        assert all(
            "openai_chat_completion" in model["supported_mcp_tools"]
            for model in payload
        )

    def test_openai_model_list(self, mcp_call: McpCall) -> None:
        """The OpenAI model listing is served as a tool.

        Ref: stdapi/routes/openai_models.py:openai_model_list
        """
        payload = _json_result(mcp_call("openai_model_list", {}))
        assert payload["object"] == "list"
        assert payload["data"]

    def test_anthropic_model_list(self, mcp_call: McpCall) -> None:
        """The Anthropic model listing is served as a tool.

        Ref: stdapi/routes/anthropic_models.py:anthropic_model_list
        """
        payload = _json_result(mcp_call("anthropic_model_list", {}))
        assert payload["data"]

    def test_openai_model_get(self, mcp_call: McpCall, chat_model: str) -> None:
        """A single OpenAI model is retrievable by ID through the tool.

        Ref: stdapi/routes/openai_models.py:openai_model_get
        """
        payload = _json_result(mcp_call("openai_model_get", {"model": chat_model}))
        assert payload["id"] == chat_model

    def test_anthropic_model_get(
        self, mcp_call: McpCall, anthropic_chat_basic_model: str
    ) -> None:
        """A single Anthropic model is retrievable by ID through the tool.

        Ref: stdapi/routes/anthropic_models.py:anthropic_model_get
        """
        payload = _json_result(
            mcp_call("anthropic_model_get", {"model_id": anthropic_chat_basic_model})
        )
        assert payload["id"] == anthropic_chat_basic_model

    def test_model_pricing(self, mcp_call: McpCall, chat_model: str) -> None:
        """Model pricing answers through the tool, per the cost-tracking setting.

        With cost tracking disabled the route's documented clean error must
        surface in-band; with it enabled, pricing rows come back.

        Ref: stdapi/routes/core_models.py:model_pricing
        """
        result = mcp_call("model_pricing", {"model": [chat_model]})
        if SETTINGS.cost_tracking:
            assert _json_result(result)
        else:
            assert result.isError
            assert "model pricing is not available" in result.content[0].text.lower()  # type: ignore[union-attr]


class TestGenerationTools:
    """Text-generation route families called through their MCP tools."""

    def test_openai_chat_completion(self, mcp_call: McpCall, chat_model: str) -> None:
        """A chat completion round-trips through the tool with usable content.

        Ref: stdapi/routes/openai_chat_completions.py:openai_chat_completion
        """
        payload = _json_result(
            mcp_call(
                "openai_chat_completion",
                {
                    "model": chat_model,
                    "messages": [{"role": "user", "content": "Reply with: pong"}],
                    "max_completion_tokens": 64,
                },
            )
        )
        assert payload["choices"][0]["message"]["content"]

    def test_openai_completion(self, mcp_call: McpCall, completion_model: str) -> None:
        """A legacy text completion round-trips through the tool.

        Ref: stdapi/routes/openai_completions.py:openai_completion
        """
        payload = _json_result(
            mcp_call(
                "openai_completion",
                {"model": completion_model, "prompt": "Say pong", "max_tokens": 16},
            )
        )
        assert payload["choices"][0]["text"] is not None

    def test_openai_response(self, mcp_call: McpCall, responses_model: str) -> None:
        """A Responses API generation round-trips through the tool.

        Ref: stdapi/routes/openai_responses.py:openai_response
        """
        payload = _json_result(
            mcp_call(
                "openai_response",
                {
                    "model": responses_model,
                    "input": "Reply with: pong",
                    "max_output_tokens": 64,
                },
            )
        )
        assert payload["status"] == "completed"
        assert payload["output"]

    def test_anthropic_message(
        self, mcp_call: McpCall, anthropic_chat_basic_model: str
    ) -> None:
        """An Anthropic message round-trips through the tool.

        Ref: stdapi/routes/anthropic_messages.py:anthropic_message
        """
        payload = _json_result(
            mcp_call(
                "anthropic_message",
                {
                    "model": anthropic_chat_basic_model,
                    "messages": [{"role": "user", "content": "Reply with: pong"}],
                    "max_tokens": 64,
                },
            )
        )
        assert payload["content"]

    def test_anthropic_message_count_tokens(
        self, mcp_call: McpCall, anthropic_count_tokens_model: str
    ) -> None:
        """Token counting answers through the tool without a generation.

        Ref: stdapi/routes/anthropic_messages.py:anthropic_message_count_tokens
        """
        payload = _json_result(
            mcp_call(
                "anthropic_message_count_tokens",
                {
                    "model": anthropic_count_tokens_model,
                    "messages": [{"role": "user", "content": "Count these tokens."}],
                },
            )
        )
        assert payload["input_tokens"] > 0

    def test_openai_response_input_tokens(
        self, mcp_call: McpCall, models: dict[str, str]
    ) -> None:
        """Responses input-token counting answers through the tool.

        Ref: stdapi/routes/openai_responses.py:openai_response_input_tokens
        """
        payload = _json_result(
            mcp_call(
                "openai_response_input_tokens",
                {"model": models["input_tokens"], "input": "Count these tokens."},
            )
        )
        assert payload["input_tokens"] > 0

    def test_openai_response_compact(
        self, mcp_call: McpCall, responses_model: str
    ) -> None:
        """A conversation compacts into a compaction item through the tool.

        Ref: stdapi/routes/openai_responses.py:openai_response_compact
        """
        payload = _json_result(
            mcp_call(
                "openai_response_compact",
                {
                    "model": responses_model,
                    "input": [
                        {"role": "user", "content": "My favourite colour is teal."},
                        {"role": "assistant", "content": "Noted: teal."},
                    ],
                },
            )
        )
        assert payload["output"]

    def test_openai_response_stored_lifecycle(
        self, mcp_call: McpCall, responses_model: str
    ) -> None:
        """A stored response is created, fetched, cancelled, and deleted via tools.

        Cancelling a locally stored (synchronous) response must surface the
        documented OpenAI error in-band rather than crash the tool.

        Ref: stdapi/routes/openai_responses.py:openai_response_get
             stdapi/routes/openai_responses.py:openai_response_input_items
             stdapi/routes/openai_responses.py:openai_response_cancel
             stdapi/routes/openai_responses.py:openai_response_delete
        """
        created = _json_result(
            mcp_call(
                "openai_response",
                {
                    "model": responses_model,
                    "input": "Reply with: pong",
                    "max_output_tokens": 64,
                    "store": True,
                },
            )
        )
        response_id = created["id"]
        fetched = _json_result(
            mcp_call("openai_response_get", {"response_id": response_id})
        )
        assert fetched["id"] == response_id
        items = _json_result(
            mcp_call("openai_response_input_items", {"response_id": response_id})
        )
        assert items["data"]
        cancel = mcp_call("openai_response_cancel", {"response_id": response_id})
        assert cancel.isError
        assert "cancel" in cancel.content[0].text.lower()  # type: ignore[union-attr]
        deleted = _json_result(
            mcp_call("openai_response_delete", {"response_id": response_id})
        )
        assert deleted["deleted"] is True

    def test_openai_chat_completion_stored_lifecycle(
        self, mcp_call: McpCall, chat_model: str
    ) -> None:
        """A stored chat completion supports list/get/update/messages/delete via tools.

        Ref: stdapi/routes/openai_chat_completions.py:openai_chat_completion_list
             stdapi/routes/openai_chat_completions.py:openai_chat_completion_get
             stdapi/routes/openai_chat_completions.py:openai_chat_completion_update
             stdapi/routes/openai_chat_completions.py:openai_chat_completion_messages
             stdapi/routes/openai_chat_completions.py:openai_chat_completion_delete
        """
        created = _json_result(
            mcp_call(
                "openai_chat_completion",
                {
                    "model": chat_model,
                    "messages": [{"role": "user", "content": "Reply with: pong"}],
                    "max_completion_tokens": 64,
                    "store": True,
                },
            )
        )
        completion_id = created["id"]
        listed = _json_result(mcp_call("openai_chat_completion_list", {}))
        assert any(item["id"] == completion_id for item in listed["data"])
        fetched = _json_result(
            mcp_call("openai_chat_completion_get", {"completion_id": completion_id})
        )
        assert fetched["id"] == completion_id
        updated = _json_result(
            mcp_call(
                "openai_chat_completion_update",
                {"completion_id": completion_id, "metadata": {"channel": "mcp"}},
            )
        )
        assert updated["metadata"] == {"channel": "mcp"}
        messages = _json_result(
            mcp_call(
                "openai_chat_completion_messages", {"completion_id": completion_id}
            )
        )
        assert messages["data"]
        deleted = _json_result(
            mcp_call("openai_chat_completion_delete", {"completion_id": completion_id})
        )
        assert deleted["deleted"] is True


class TestEmbeddingAndRerankTools:
    """Embedding and rerank route families called through their MCP tools."""

    def test_openai_embedding(self, mcp_call: McpCall, embedding_model: str) -> None:
        """An embedding request returns a vector through the tool.

        Ref: stdapi/routes/openai_embeddings.py:openai_embedding
        """
        payload = _json_result(
            mcp_call("openai_embedding", {"model": embedding_model, "input": "hello"})
        )
        assert payload["data"][0]["embedding"]

    def test_cohere_embed(
        self, mcp_call: McpCall, cohere_embed_multilingual_model: str
    ) -> None:
        """A Cohere v2 embed request returns vectors through the tool.

        Ref: stdapi/routes/cohere_embed.py:cohere_embed
        """
        payload = _json_result(
            mcp_call(
                "cohere_embed",
                {
                    "model": cohere_embed_multilingual_model,
                    "texts": ["hello world"],
                    "input_type": "search_query",
                    "embedding_types": ["float"],
                },
            )
        )
        assert payload["embeddings"]["float"]

    def test_cohere_embed_v1(
        self, mcp_call: McpCall, cohere_embed_multilingual_model: str
    ) -> None:
        """A Cohere v1 embed request returns vectors through the tool.

        Ref: stdapi/routes/cohere_embed_v1.py:cohere_embed_v1
        """
        payload = _json_result(
            mcp_call(
                "cohere_embed_v1",
                {"model": cohere_embed_multilingual_model, "texts": ["hello world"]},
            )
        )
        assert payload["embeddings"]

    def test_cohere_rerank(self, mcp_call: McpCall, cohere_rerank_model: str) -> None:
        """A Cohere v2 rerank request scores documents through the tool.

        Ref: stdapi/routes/cohere_rerank.py:cohere_rerank
        """
        payload = _json_result(
            mcp_call(
                "cohere_rerank",
                {
                    "model": cohere_rerank_model,
                    "query": "a fruit",
                    "documents": ["an apple", "a brick"],
                },
            )
        )
        assert payload["results"]

    def test_cohere_rerank_v1(
        self, mcp_call: McpCall, cohere_rerank_model: str
    ) -> None:
        """A Cohere v1 rerank request scores documents through the tool.

        Ref: stdapi/routes/cohere_rerank_v1.py:cohere_rerank_v1
        """
        payload = _json_result(
            mcp_call(
                "cohere_rerank_v1",
                {
                    "model": cohere_rerank_model,
                    "query": "a fruit",
                    "documents": ["an apple", "a brick"],
                },
            )
        )
        assert payload["results"]


class TestModerationTool:
    """Moderation route family called through its MCP tool."""

    def test_openai_moderation(self, mcp_call: McpCall) -> None:
        """A moderation request classifies text through the tool (default model).

        Ref: stdapi/routes/openai_moderations.py:openai_moderation
        """
        payload = _json_result(mcp_call("openai_moderation", {"input": "hello world"}))
        assert payload["results"][0]["categories"] is not None


class TestAudioTools:
    """Audio route families called through their MCP tools (JSON request format)."""

    def test_openai_audio_speech(
        self, mcp_call: McpCall, speech_standard_model: str
    ) -> None:
        """Speech synthesis returns audio bytes rendered as tool content.

        The route answers binary audio, which ``fastapi_mcp`` surfaces as the
        response text; the call succeeding without an in-band error is the
        contract under test.

        Ref: stdapi/routes/openai_audio_speech.py:openai_audio_speech
        """
        result = mcp_call(
            "openai_audio_speech",
            {"model": speech_standard_model, "voice": "alloy", "input": "Test."},
        )
        assert not result.isError, result.content
        assert result.content

    def test_openai_audio_transcription(
        self,
        mcp_call: McpCall,
        transcription_model: str,
        sample_audio_mp3_file_base64: str,
    ) -> None:
        """Audio passed as a base64 data URI is transcribed through the tool.

        Ref: stdapi/routes/openai_audio_transcriptions.py:openai_audio_transcription
        """
        payload = _json_result(
            mcp_call(
                "openai_audio_transcription",
                {
                    "file": sample_audio_mp3_file_base64,
                    "model": transcription_model,
                    "response_format": "json",
                },
            )
        )
        assert payload["text"]

    def test_openai_audio_translation(
        self,
        mcp_call: McpCall,
        transcription_model: str,
        sample_audio_mp3_file_base64: str,
    ) -> None:
        """Audio passed as a base64 data URI is translated through the tool.

        Ref: stdapi/routes/openai_audio_translations.py:openai_audio_translation
        """
        payload = _json_result(
            mcp_call(
                "openai_audio_translation",
                {
                    "file": sample_audio_mp3_file_base64,
                    "model": transcription_model,
                    "response_format": "json",
                },
            )
        )
        assert payload["text"]


class TestImageTools:
    """Image route families called through their MCP tools (JSON request format)."""

    @pytest.mark.expensive
    def test_openai_image_generation(
        self, mcp_call: McpCall, image_generation_model: str, image_generation_size: str
    ) -> None:
        """An image is generated and returned as base64 through the tool.

        Ref: stdapi/routes/openai_images_generations.py:openai_image_generation
        """
        payload = _json_result(
            mcp_call(
                "openai_image_generation",
                {
                    "model": image_generation_model,
                    "prompt": "A plain red circle on white background",
                    "size": image_generation_size,
                    "response_format": "b64_json",
                },
            )
        )
        assert payload["data"][0]["b64_json"]

    @pytest.mark.expensive
    def test_openai_image_edit(
        self, mcp_call: McpCall, sample_image_file_base64: str
    ) -> None:
        """An image passed as a base64 data URI is edited through the tool.

        Ref: stdapi/routes/openai_images_edits.py:openai_image_edit
        """
        payload = _json_result(
            mcp_call(
                "openai_image_edit",
                {
                    "image": [sample_image_file_base64],
                    "prompt": "Make the background blue",
                    "model": "stability.stable-image-inpaint-v1:0",
                    "response_format": "b64_json",
                },
            )
        )
        assert payload["data"][0]["b64_json"]

    @pytest.mark.expensive
    def test_openai_image_variation(
        self, mcp_call: McpCall, sample_image_file_base64: str
    ) -> None:
        """An image passed as a base64 data URI is varied through the tool.

        Ref: stdapi/routes/openai_images_variations.py:openai_image_variation
        """
        payload = _json_result(
            mcp_call(
                "openai_image_variation",
                {
                    "image": sample_image_file_base64,
                    "model": "stability.sd3-5-large-v1:0",
                    "response_format": "b64_json",
                },
            )
        )
        assert payload["data"][0]["b64_json"]


class TestStorageTools:
    """File, upload and video-listing route families through their MCP tools."""

    def test_openai_file_lifecycle(self, mcp_call: McpCall) -> None:
        """A file uploaded as a data URI is listed, fetched, read, and deleted via tools.

        Ref: stdapi/routes/openai_files.py:openai_file
             stdapi/routes/openai_files.py:openai_file_list
             stdapi/routes/openai_files.py:openai_files_get
             stdapi/routes/openai_files.py:openai_file_content
             stdapi/routes/openai_files.py:openai_files_delete
        """
        created = _json_result(
            mcp_call(
                "openai_file",
                {
                    "file": "data:text/plain;base64,aGVsbG8gbWNwCg==",
                    "purpose": "assistants",
                },
            )
        )
        file_id = created["id"]
        listed = _json_result(mcp_call("openai_file_list", {}))
        assert any(item["id"] == file_id for item in listed["data"])
        fetched = _json_result(mcp_call("openai_files_get", {"file_id": file_id}))
        assert fetched["id"] == file_id
        content = mcp_call("openai_file_content", {"file_id": file_id})
        assert not content.isError, content.content
        assert "hello mcp" in content.content[0].text  # type: ignore[union-attr]
        deleted = _json_result(mcp_call("openai_files_delete", {"file_id": file_id}))
        assert deleted["deleted"] is True

    def test_openai_file_content_of_a_non_text_file(self, mcp_call: McpCall) -> None:
        """A binary file read through the tool comes back usable, not as mojibake.

        ``fastapi_mcp`` decodes every response body as text, which no binary
        file survives, so the whole file-content tool was text-only. The stored
        content type decides the result: a PNG the agent can look at comes back
        as image content, byte for byte.

        Ref: stdapi/mcp.py:_bind_media_results
             stdapi/routes/openai_files.py:openai_file_content
        """
        created = _json_result(
            mcp_call(
                "openai_file",
                {"file": f"data:image/png;base64,{_PNG_BASE64}", "purpose": "vision"},
            )
        )
        file_id = created["id"]
        try:
            content = mcp_call("openai_file_content", {"file_id": file_id})
            assert not content.isError, content.content
            block = content.content[0]
            assert block.type == "image"
            assert block.mimeType == "image/png"
            assert b64decode(block.data) == b64decode(_PNG_BASE64)
        finally:
            mcp_call("openai_files_delete", {"file_id": file_id})

    def test_anthropic_file_list(self, mcp_call: McpCall) -> None:
        """The Anthropic file listing is served as a tool.

        Ref: stdapi/routes/anthropic_files.py:anthropic_file_list
        """
        payload = _json_result(mcp_call("anthropic_file_list", {}))
        assert "data" in payload

    def test_anthropic_file_lifecycle(self, mcp_call: McpCall) -> None:
        """An Anthropic file uploads, fetches, reads, and deletes through tools.

        Ref: stdapi/routes/anthropic_files.py:anthropic_file
             stdapi/routes/anthropic_files.py:anthropic_files_get
             stdapi/routes/anthropic_files.py:anthropic_file_content
             stdapi/routes/anthropic_files.py:anthropic_files_delete
        """
        created = _json_result(
            mcp_call(
                "anthropic_file", {"file": "data:text/plain;base64,aGVsbG8gbWNwCg=="}
            )
        )
        file_id = created["id"]
        fetched = _json_result(mcp_call("anthropic_files_get", {"file_id": file_id}))
        assert fetched["id"] == file_id
        content = mcp_call("anthropic_file_content", {"file_id": file_id})
        assert not content.isError, content.content
        assert "hello mcp" in content.content[0].text  # type: ignore[union-attr]
        deleted = _json_result(mcp_call("anthropic_files_delete", {"file_id": file_id}))
        assert deleted["id"] == file_id

    def test_openai_upload_part_and_complete(self, mcp_call: McpCall) -> None:
        """A multipart upload accepts a base64 part and completes into a file via tools.

        Ref: stdapi/routes/openai_uploads.py:openai_upload_part
             stdapi/routes/openai_uploads.py:openai_upload_complete
        """
        upload = _json_result(
            mcp_call(
                "openai_upload",
                {
                    "filename": "mcp-upload.txt",
                    "purpose": "assistants",
                    "bytes": 10,
                    "mime_type": "text/plain",
                },
            )
        )
        part = _json_result(
            mcp_call(
                "openai_upload_part",
                {
                    "upload_id": upload["id"],
                    "data": "data:text/plain;base64,aGVsbG8gbWNwCg==",
                },
            )
        )
        assert part["upload_id"] == upload["id"]
        completed = _json_result(
            mcp_call(
                "openai_upload_complete",
                {"upload_id": upload["id"], "part_ids": [part["id"]]},
            )
        )
        assert completed["status"] == "completed"
        _json_result(
            mcp_call("openai_files_delete", {"file_id": completed["file"]["id"]})
        )

    def test_openai_upload_create_and_cancel(self, mcp_call: McpCall) -> None:
        """A multipart upload is created, then cancelled, through tools.

        Ref: stdapi/routes/openai_uploads.py:openai_upload
             stdapi/routes/openai_uploads.py:openai_upload_cancel
        """
        upload = _json_result(
            mcp_call(
                "openai_upload",
                {
                    "filename": "mcp-upload.txt",
                    "purpose": "assistants",
                    "bytes": 9,
                    "mime_type": "text/plain",
                },
            )
        )
        assert upload["status"] == "pending"
        cancelled = _json_result(
            mcp_call("openai_upload_cancel", {"upload_id": upload["id"]})
        )
        assert cancelled["status"] == "cancelled"

    def test_openai_video_list(self, mcp_call: McpCall) -> None:
        """The video job listing is served as a tool.

        Ref: stdapi/routes/openai_videos.py:openai_video_list
        """
        payload = _json_result(mcp_call("openai_video_list", {}))
        assert "data" in payload

    @pytest.mark.video
    @pytest.mark.slow
    def test_openai_video_lifecycle(
        self, mcp_call: McpCall, video_generation_model: str
    ) -> None:
        """A video generates, polls to completion, downloads, and deletes via tools.

        No MCP content type carries video, so the content tool answers with the
        URL the finished clip downloads from rather than with its bytes.

        Ref: stdapi/routes/openai_videos.py:openai_video_generation
             stdapi/routes/openai_videos.py:openai_video_get
             stdapi/routes/openai_videos.py:openai_video_content
             stdapi/routes/openai_videos.py:openai_video_delete
             stdapi/mcp.py:_bind_media_results
        """
        from time import monotonic, sleep  # noqa: PLC0415

        created = _json_result(
            mcp_call(
                "openai_video_generation",
                {
                    "model": video_generation_model,
                    "prompt": "A calico cat playing a piano",
                    # Luma bills 540p at half the 720p rate.
                    "size": "960x540",
                },
            )
        )
        video_id = created["id"]
        try:
            assert created["status"] in ("queued", "in_progress")
            deadline = monotonic() + 600
            video = created
            while video["status"] not in ("completed", "failed"):
                assert monotonic() < deadline, "video generation timed out"
                sleep(10)
                video = _json_result(
                    mcp_call("openai_video_get", {"video_id": video_id})
                )
            assert video["status"] == "completed", video
            content = mcp_call("openai_video_content", {"video_id": video_id})
            assert not content.isError, content.content
            reference = loads(content.content[0].text)  # type: ignore[union-attr]
            assert reference["content_type"] == "video/mp4"
            assert reference["url"].endswith(f"/videos/{video_id}/content")
        finally:
            deleted = _json_result(
                mcp_call("openai_video_delete", {"video_id": video_id})
            )
            assert deleted["deleted"] is True
