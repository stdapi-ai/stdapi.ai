"""Tests for the OpenAI-compatible POST /v1/responses/compact route (unit)."""

from base64 import urlsafe_b64decode
from typing import TYPE_CHECKING, Any

import pytest
from starlette.testclient import TestClient

from stdapi.api_errors import ApiError

if TYPE_CHECKING:
    from openai import OpenAI
from stdapi.models import ModelDetails
from stdapi.models.chat._adapters._openai_responses import (
    encode_compaction_content,
    map_input,
)
from stdapi.routes import openai_responses
from stdapi.types.openai_responses import (
    CompactionItemParam,
    EasyInputMessage,
    InputMessage,
    InputTokensDetails,
    OutputTokensDetails,
    Response,
    ResponseCreateParams,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)


def _usage() -> ResponseUsage:
    return ResponseUsage(
        input_tokens=11,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens=7,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=18,
    )


class _StubChatModel:
    """Stub chat backend recording the generation request."""

    def __init__(self) -> None:
        self.requests: list[ResponseCreateParams] = []

    async def create_response(
        self, request: ResponseCreateParams, response_id: str, created_at: float
    ) -> Response:
        """Record the request and return a canned summary response."""
        self.requests.append(request)
        return Response(
            id=response_id,
            created_at=created_at,
            model=request.model,
            object="response",
            output=[
                ResponseOutputMessage(
                    id="msg-1",
                    content=[
                        ResponseOutputText(
                            annotations=[], text="THE SUMMARY", type="output_text"
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
            usage=_usage(),
        )


@pytest.fixture
def client(api_key: str) -> TestClient:
    """Test client without lifespan (no AWS startup), pre-authenticated."""
    from stdapi.main import app  # noqa: PLC0415

    return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})


@pytest.fixture
def chat_backend(monkeypatch: pytest.MonkeyPatch) -> _StubChatModel:
    """Stub model validation and the chat generation backend."""

    async def _validate_model(
        model_id: str, *_args: object, **_kwargs: object
    ) -> ModelDetails:
        return ModelDetails(
            id=model_id,
            name=model_id,
            provider="Vendor",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            regions=["us-east-1"],
        )

    stub = _StubChatModel()
    monkeypatch.setattr(openai_responses, "validate_model", _validate_model)
    monkeypatch.setattr(openai_responses, "get_chat_model", lambda _model_id: stub)
    return stub


@pytest.mark.local
class TestResponsesCompactRoute:
    """POST /v1/responses/compact: response shape and generation request."""

    def test_compact_returns_compaction_item(
        self, client: TestClient, chat_backend: _StubChatModel
    ) -> None:
        """String input yields a user message echo followed by the compaction item."""
        response = client.post(
            "/v1/responses/compact",
            json={"model": "amazon.nova-pro-v1:0", "input": "a long conversation"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "response.compaction"
        echo, item = body["output"]
        assert echo["type"] == "message"
        assert echo["status"] == "completed"
        assert echo["role"] == "user"
        assert echo["content"] == [
            {"type": "input_text", "text": "a long conversation"}
        ]
        assert item["type"] == "compaction"
        assert urlsafe_b64decode(item["encrypted_content"]) == b"THE SUMMARY"
        assert body["usage"]["total_tokens"] == 18

    def test_compact_echoes_only_user_messages_in_order(
        self, client: TestClient, chat_backend: _StubChatModel
    ) -> None:
        """Assistant messages are dropped; user echoes stay ordered before the compaction item."""
        response = client.post(
            "/v1/responses/compact",
            json={
                "model": "amazon.nova-pro-v1:0",
                "input": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "reply"},
                    {"role": "user", "content": "second"},
                ],
            },
        )
        assert response.status_code == 200, response.text
        *echoes, item = response.json()["output"]
        assert [echo["content"][0]["text"] for echo in echoes] == ["first", "second"]
        assert all(echo["role"] == "user" for echo in echoes)
        assert item["type"] == "compaction"

    def test_compact_echoes_part_list_content_as_dicts(
        self, client: TestClient, chat_backend: _StubChatModel
    ) -> None:
        """List-based user content parts are echoed back as dicts."""
        parts = [
            {"type": "input_text", "text": "look at this"},
            {"type": "input_image", "image_url": "https://example.com/img.png"},
        ]
        response = client.post(
            "/v1/responses/compact",
            json={
                "model": "amazon.nova-pro-v1:0",
                "input": [{"role": "user", "content": parts}],
            },
        )
        assert response.status_code == 200, response.text
        echo, item = response.json()["output"]
        assert echo["content"] == parts
        assert item["type"] == "compaction"

    def test_compact_appends_summarization_directive(
        self, client: TestClient, chat_backend: _StubChatModel
    ) -> None:
        """The generation request keeps the input and appends the directive."""
        client.post(
            "/v1/responses/compact",
            json={
                "model": "amazon.nova-pro-v1:0",
                "input": [{"role": "user", "content": "hello"}],
                "instructions": "be nice",
            },
        )
        (request,) = chat_backend.requests
        assert request.instructions == "be nice"
        assert isinstance(request.input, list)
        first = request.input[0]
        assert isinstance(first, EasyInputMessage | InputMessage)
        assert first.content == "hello"
        last = request.input[-1]
        assert isinstance(last, EasyInputMessage | InputMessage)
        assert "Summarize the conversation" in str(last.content)

    def test_previous_response_id_compacts_stored_conversation(
        self,
        client: TestClient,
        chat_backend: _StubChatModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored conversation is prepended before the compaction directive."""

        async def _load(response_id: str, kind: str) -> dict[str, Any]:
            assert response_id == "resp-1"
            assert kind == "response"
            return {"input": [{"role": "user", "content": "first"}], "response": {}}

        monkeypatch.setattr(openai_responses, "load_stored_response", _load)
        response = client.post(
            "/v1/responses/compact",
            json={"model": "m", "input": "second", "previous_response_id": "resp-1"},
        )
        assert response.status_code == 200, response.text
        (request,) = chat_backend.requests
        assert request.previous_response_id is None
        assert isinstance(request.input, list)
        first = request.input[0]
        assert isinstance(first, EasyInputMessage | InputMessage)
        assert first.content == "first"
        second = request.input[1]
        assert isinstance(second, EasyInputMessage | InputMessage)
        assert second.content == "second"
        last = request.input[-1]
        assert isinstance(last, EasyInputMessage | InputMessage)
        assert "Summarize the conversation" in str(last.content)
        *echoes, item = response.json()["output"]
        assert [echo["content"][0]["text"] for echo in echoes] == ["first", "second"]
        assert item["type"] == "compaction"

    def test_unknown_previous_response_id_is_not_found(
        self,
        client: TestClient,
        chat_backend: _StubChatModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unknown previous_response_id surfaces as 404."""

        async def _load(response_id: str, kind: str) -> dict[str, Any]:  # noqa: ARG001
            from stdapi.api_errors import ApiError  # noqa: PLC0415

            msg = f"Response with id '{response_id}' not found."
            raise ApiError(msg, status=404)

        monkeypatch.setattr(openai_responses, "load_stored_response", _load)
        response = client.post(
            "/v1/responses/compact",
            json={"model": "m", "input": "x", "previous_response_id": "resp-zzz"},
        )
        assert response.status_code == 404
        assert not chat_backend.requests


@pytest.mark.local
class TestCompactionItemRoundTrip:
    """Compaction items round-trip through the Responses input mapping."""

    async def test_compaction_item_maps_to_user_summary_message(self) -> None:
        """The decoded summary is injected as a user message.

        The summary's UTF-8 bytes base64-encode to characters that differ
        between the standard and urlsafe alphabets ('-'/'_' vs '+'/'/'),
        catching an encode/decode alphabet mismatch.
        """
        summary = "Summary: \xff\xff\xff details preserved."
        encrypted_content = encode_compaction_content(summary)
        assert "+" not in encrypted_content
        assert "/" not in encrypted_content
        assert "-" in encrypted_content or "_" in encrypted_content
        item = CompactionItemParam(
            encrypted_content=encrypted_content, type="compaction"
        )
        messages, system = await map_input([item], None)
        assert system == []
        (message,) = messages
        assert message["role"] == "user"
        assert summary in message["content"][0]["text"]

    async def test_invalid_compaction_content_is_rejected(self) -> None:
        """Undecodable compaction content raises a 400 error."""
        item = CompactionItemParam(encrypted_content="!!!", type="compaction")
        with pytest.raises(ApiError, match="compaction"):
            await map_input([item], None)


class TestResponsesCompactLive:
    """Live conversation compaction and round-trip continuation."""

    def test_compact_and_continue(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A compacted conversation carries its facts into the next turn."""
        compacted = openai_client.responses.compact(
            model=responses_model,
            input=[
                {"role": "user", "content": "My favorite color is teal."},
                {
                    "role": "assistant",
                    "content": "Understood, your favorite color is teal.",
                },
            ],
        )
        assert compacted.object == "response.compaction"
        assert compacted.usage.total_tokens > 0
        # The official API may add a message item next to the compaction item.
        item = next(part for part in compacted.output if part.type == "compaction")
        assert item.encrypted_content

        follow = openai_client.responses.create(
            model=responses_model,
            input=[
                {
                    "type": "compaction",
                    "id": item.id,
                    "encrypted_content": item.encrypted_content,
                },
                {
                    "role": "user",
                    "content": "What is my favorite color? Reply with one word.",
                },
            ],
            store=False,
        )
        assert "teal" in follow.output_text.lower()
