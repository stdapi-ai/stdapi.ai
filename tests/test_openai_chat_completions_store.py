"""Tests for stored chat completions routes (unit)."""

from typing import TYPE_CHECKING, Any

import pytest
from starlette.testclient import TestClient

from stdapi.api_errors import ApiError

if TYPE_CHECKING:
    from openai import OpenAI
from stdapi.models import ModelDetails
from stdapi.routes import openai_chat_completions
from stdapi.types.openai_chat_completions import ChatCompletion, CompletionCreateParams


def _canned_completion(completion_id: str, model: str) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": completion_id,
            "created": 1752000000,
            "model": model,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                }
            ],
        }
    )


class _StubChatBackend:
    """Stub chat backend recording generation requests."""

    def __init__(self) -> None:
        self.requests: list[tuple[CompletionCreateParams, str]] = []

    async def create_completion(
        self, request: CompletionCreateParams, completion_id: str, created: int
    ) -> ChatCompletion:
        """Record the request and return a canned completion."""
        self.requests.append((request, completion_id))
        return _canned_completion(completion_id, request.model)


class _StubStore:
    """Stub persistence layer recording store operations."""

    def __init__(self) -> None:
        self.saved: list[tuple[str, dict[str, Any]]] = []
        self.deleted: list[str] = []
        self.discarded: list[str] = []
        self.documents: dict[str, dict[str, Any]] = {}

    async def create_session(self) -> str:
        return "sess-1"

    async def save(self, completion_id: str, document: dict[str, Any]) -> None:
        self.saved.append((completion_id, document))

    async def load(self, completion_id: str) -> dict[str, Any]:
        if completion_id not in self.documents:
            msg = f"Chat completion with id '{completion_id}' not found."
            raise ApiError(msg, status=404)
        return self.documents[completion_id]

    async def delete(self, completion_id: str) -> None:
        if completion_id not in self.documents:
            msg = f"Chat completion with id '{completion_id}' not found."
            raise ApiError(msg, status=404)
        self.deleted.append(completion_id)

    async def discard(self, completion_id: str) -> None:
        self.discarded.append(completion_id)


@pytest.fixture
def client(api_key: str) -> TestClient:
    """Test client without lifespan (no AWS startup), pre-authenticated."""
    from stdapi.main import app  # noqa: PLC0415

    return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> _StubChatBackend:
    """Stub model validation and the generation backend."""

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

    stub = _StubChatBackend()
    monkeypatch.setattr(openai_chat_completions, "validate_model", _validate_model)
    monkeypatch.setattr(
        openai_chat_completions, "get_chat_model", lambda _model_id: stub
    )
    return stub


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _StubStore:
    """Stub the stored-completion persistence functions."""
    stub = _StubStore()
    monkeypatch.setattr(
        openai_chat_completions,
        "try_create_stored_response_session",
        stub.create_session,
    )
    monkeypatch.setattr(openai_chat_completions, "save_stored_response", stub.save)
    monkeypatch.setattr(openai_chat_completions, "load_stored_response", stub.load)
    monkeypatch.setattr(openai_chat_completions, "delete_stored_response", stub.delete)
    monkeypatch.setattr(
        openai_chat_completions, "discard_stored_response_session", stub.discard
    )
    return stub


@pytest.mark.local
class TestStoreOnChatCreate:
    """store=true on POST /v1/chat/completions."""

    def test_store_persists_completion(
        self, client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """The completion is generated under the session ID and persisted."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "store": True,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"] == "chatcmpl-sess-1"
        ((completion_id, document),) = store.saved
        assert completion_id == "chatcmpl-sess-1"
        assert document["messages"][0]["content"] == "hello"
        assert document["response"]["id"] == "chatcmpl-sess-1"

    def test_store_with_stream_is_ignored(
        self, client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """store=true with streaming is ignored: no session is involved."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "store": True,
                "stream": True,
            },
        )
        assert response.status_code == 200, response.text
        assert not store.created_kinds
        assert not store.saved

    def test_store_without_session_storage_is_ignored(
        self,
        client: TestClient,
        backend: _StubChatBackend,
        store: _StubStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """store=true is ignored when session storage is unavailable."""

        async def _unavailable(_kind: str) -> None:
            return None

        monkeypatch.setattr(
            openai_chat_completions, "try_create_stored_response_session", _unavailable
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "store": True,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"].startswith("chatcmpl-")
        assert "None" not in response.json()["id"]
        assert not store.saved
        assert not store.discarded

    def test_without_store_nothing_is_persisted(
        self, client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """Without store=true no session is involved."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"] != "chatcmpl-sess-1"
        assert not store.saved

    def test_generation_failure_discards_pending_session(
        self, client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """A failed generation discards the pending stored session."""

        async def _raise(
            _request: CompletionCreateParams, _completion_id: str, _created: int
        ) -> ChatCompletion:
            msg = "backend failure"
            raise ApiError(msg, status=502)

        backend.create_completion = _raise  # type: ignore[method-assign]
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "store": True,
            },
        )
        assert response.status_code == 502
        assert store.discarded == ["chatcmpl-sess-1"]


@pytest.mark.local
class TestStoredChatCompletionRoutes:
    """GET/DELETE /v1/chat/completions/{id} and messages listing."""

    def test_retrieve_stored_completion(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """A stored chat completion is returned as-is."""
        store.documents["chatcmpl-sess-1"] = {
            "messages": [{"role": "user", "content": "hello"}],
            "response": _canned_completion("chatcmpl-sess-1", "m").model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
        }
        response = client.get("/v1/chat/completions/chatcmpl-sess-1")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"] == "chatcmpl-sess-1"
        assert body["choices"][0]["message"]["content"] == "answer"

    def test_retrieve_unknown_is_not_found(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """An unknown stored chat completion surfaces as 404."""
        response = client.get("/v1/chat/completions/chatcmpl-zzz")
        assert response.status_code == 404
        assert "chatcmpl-zzz" in response.json()["error"]["message"]

    def test_delete_stored_completion(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """Deletion returns a confirmation object."""
        store.documents["chatcmpl-sess-1"] = {"messages": [], "response": {}}
        response = client.delete("/v1/chat/completions/chatcmpl-sess-1")
        assert response.status_code == 200, response.text
        assert response.json() == {
            "id": "chatcmpl-sess-1",
            "object": "chat.completion.deleted",
            "deleted": True,
        }
        assert store.deleted == ["chatcmpl-sess-1"]

    def test_messages_listing_with_cursor(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """Messages get IDs, keep conversation order, and page with after."""
        store.documents["chatcmpl-sess-1"] = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
            "response": {},
        }
        response = client.get("/v1/chat/completions/chatcmpl-sess-1/messages")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "list"
        assert [message["id"] for message in body["data"]] == ["msg-0", "msg-1"]
        assert body["data"][1]["content"] == "hello"
        assert body["has_more"] is False

        response = client.get(
            "/v1/chat/completions/chatcmpl-sess-1/messages?after=msg-0&limit=1"
        )
        body = response.json()
        assert [message["id"] for message in body["data"]] == ["msg-1"]

    def test_messages_listing_order_desc(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """order=desc reverses the conversation order."""
        store.documents["chatcmpl-sess-1"] = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
            "response": {},
        }
        response = client.get(
            "/v1/chat/completions/chatcmpl-sess-1/messages?order=desc"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert [message["id"] for message in body["data"]] == ["msg-1", "msg-0"]


class TestStoredChatCompletionsLive:
    """Live stored chat completions lifecycle against AWS Bedrock sessions."""

    def test_store_lifecycle(
        self, openai_client: OpenAI, chat_model: str, use_official_api: bool
    ) -> None:
        """store=true persists; retrieve, messages, and delete work."""
        if use_official_api:
            pytest.skip("official API stores completions asynchronously (delayed)")
        from openai import NotFoundError  # noqa: PLC0415

        created = openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Reply with the word: banana"}],
            store=True,
        )
        try:
            assert created.id.startswith("chatcmpl-")
            retrieved = openai_client.chat.completions.retrieve(created.id)
            assert retrieved.id == created.id
            assert (
                retrieved.choices[0].message.content
                == created.choices[0].message.content
            )
            messages = list(openai_client.chat.completions.messages.list(created.id))
            assert messages
            assert "banana" in messages[0].content
        finally:
            deleted = openai_client.chat.completions.delete(created.id)
            assert deleted.deleted is True
        with pytest.raises(NotFoundError):
            openai_client.chat.completions.retrieve(created.id)
