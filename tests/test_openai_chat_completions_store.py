"""Tests for stored chat completions routes (unit)."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

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


#: Stored object kind declared by each document ``response.object`` field.
_KIND_BY_OBJECT = {"response": "response", "chat.completion": "chat_completion"}


class _StubStore:
    """Stub persistence layer recording store operations."""

    def __init__(self) -> None:
        self.saved: list[tuple[str, dict[str, Any]]] = []
        self.deleted: list[str] = []
        self.discarded: list[str] = []
        self.documents: dict[str, dict[str, Any]] = {}
        self.created_kinds: list[str] = []
        #: Canned ``(session_id, created_at)`` pairs returned by ``list_sessions``.
        self.sessions: list[tuple[str, datetime]] = []
        self.list_sessions_kinds: list[str] = []
        #: Errors to raise from ``load`` for specific completion IDs.
        self.load_errors: dict[str, Exception] = {}

    async def create_session(self, kind: str) -> str:
        self.created_kinds.append(kind)
        return "sess-1"

    async def save(self, completion_id: str, document: dict[str, Any]) -> None:
        self.saved.append((completion_id, document))

    def _document_or_not_found(self, completion_id: str, kind: str) -> dict[str, Any]:
        """Return the stored document, 404ing when absent or of a different kind."""
        if completion_id in self.load_errors:
            raise self.load_errors[completion_id]
        document = self.documents.get(completion_id)
        declared = document and _KIND_BY_OBJECT.get(
            document.get("response", {}).get("object")
        )
        if document is None or (declared is not None and declared != kind):
            msg = f"Chat completion with id '{completion_id}' not found."
            raise ApiError(msg, status=404)
        return document

    async def load(self, completion_id: str, kind: str) -> dict[str, Any]:
        return self._document_or_not_found(completion_id, kind)

    async def delete(self, completion_id: str, kind: str) -> None:
        self._document_or_not_found(completion_id, kind)
        self.deleted.append(completion_id)

    async def discard(self, completion_id: str, kind: str) -> None:
        del kind
        self.discarded.append(completion_id)

    async def list_sessions(self, kind: str) -> list[tuple[str, datetime]]:
        self.list_sessions_kinds.append(kind)
        return self.sessions


def _store_completion(
    store: _StubStore,
    session_id: str,
    *,
    model: str = "m",
    metadata: dict[str, str] | None = None,
) -> None:
    """Register a stored chat completion document for *session_id*."""
    completion = _canned_completion(f"chatcmpl-{session_id}", model)
    completion.metadata = metadata
    store.documents[f"chatcmpl-{session_id}"] = {
        "messages": [],
        "response": completion.model_dump(
            mode="json", by_alias=True, exclude_none=True
        ),
    }


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
    monkeypatch.setattr(
        openai_chat_completions, "list_stored_sessions", stub.list_sessions
    )
    return stub


@pytest.mark.local
class TestStoreOnChatCreate:
    """store=true on POST /v1/chat/completions."""

    def test_store_persists_completion(
        self, client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """The completion is generated under a chat_completion session and persisted."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "store": True,
                "metadata": {"team": "x"},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"] == "chatcmpl-sess-1"
        assert response.json()["metadata"] == {"team": "x"}
        assert store.created_kinds == ["chat_completion"]
        ((completion_id, document),) = store.saved
        assert completion_id == "chatcmpl-sess-1"
        assert document["messages"][0]["content"] == "hello"
        assert document["response"]["id"] == "chatcmpl-sess-1"
        assert document["response"]["metadata"] == {"team": "x"}

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

        backend.create_completion = _raise  # type: ignore[method-assign, assignment]
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

    def test_save_failure_discards_pending_session(
        self,
        client: TestClient,
        backend: _StubChatBackend,
        store: _StubStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed save after a successful generation discards the pending session."""

        async def _raise(_completion_id: str, _document: dict[str, Any]) -> None:
            msg = "save failure"
            raise ApiError(msg, status=502)

        monkeypatch.setattr(openai_chat_completions, "save_stored_response", _raise)
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

    def test_retrieve_response_kind_session_is_not_found(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """Retrieving a response-kind session as a chat completion 404s."""
        store.documents["chatcmpl-sess-1"] = {
            "input": [],
            "response": {"object": "response"},
        }
        response = client.get("/v1/chat/completions/chatcmpl-sess-1")
        assert response.status_code == 404

    def test_delete_response_kind_session_is_not_found_and_not_deleted(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """Deleting a response-kind session as a chat completion 404s without deleting."""
        store.documents["chatcmpl-sess-1"] = {
            "input": [],
            "response": {"object": "response"},
        }
        response = client.delete("/v1/chat/completions/chatcmpl-sess-1")
        assert response.status_code == 404
        assert not store.deleted

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

    def test_messages_listing_splits_array_content(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """Array content is split into concatenated text `content` and `content_parts`."""
        store.documents["chatcmpl-sess-1"] = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look at "},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x/img.png"},
                        },
                        {"type": "text", "text": "this"},
                    ],
                },
                {"role": "system", "content": "sys"},
            ],
            "response": {},
        }
        response = client.get("/v1/chat/completions/chatcmpl-sess-1/messages")
        assert response.status_code == 200, response.text
        body = response.json()
        array_message, string_message = body["data"]
        assert array_message["content"] == "look at this"
        assert array_message["content_parts"] == [
            {"type": "text", "text": "look at "},
            {"type": "image_url", "image_url": {"url": "https://x/img.png"}},
            {"type": "text", "text": "this"},
        ]
        assert string_message["content"] == "sys"
        assert "content_parts" not in string_message

    def test_messages_listing_unknown_after_cursor_is_not_found(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """An `after` cursor matching no message 404s instead of an empty page."""
        store.documents["chatcmpl-sess-1"] = {
            "messages": [{"role": "user", "content": "hello"}],
            "response": {},
        }
        response = client.get(
            "/v1/chat/completions/chatcmpl-sess-1/messages?after=msg-99"
        )
        assert response.status_code == 404
        assert "msg-99" in response.json()["error"]["message"]

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


@pytest.mark.local
class TestListChatCompletions:
    """GET /v1/chat/completions."""

    def test_default_order_is_ascending_by_created_at(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """Without an explicit order, the oldest session comes first."""
        store.sessions = [
            ("s2", datetime(2024, 1, 2, tzinfo=UTC)),
            ("s1", datetime(2024, 1, 1, tzinfo=UTC)),
        ]
        _store_completion(store, "s1")
        _store_completion(store, "s2")
        response = client.get("/v1/chat/completions")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["chatcmpl-s1", "chatcmpl-s2"]

    def test_order_desc_returns_newest_first(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """order=desc reverses the creation-time order."""
        store.sessions = [
            ("s1", datetime(2024, 1, 1, tzinfo=UTC)),
            ("s2", datetime(2024, 1, 2, tzinfo=UTC)),
        ]
        _store_completion(store, "s1")
        _store_completion(store, "s2")
        response = client.get("/v1/chat/completions?order=desc")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["chatcmpl-s2", "chatcmpl-s1"]

    def test_after_cursor_returns_later_items(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """A known after cursor skips itself and every earlier item."""
        store.sessions = [
            ("s1", datetime(2024, 1, 1, tzinfo=UTC)),
            ("s2", datetime(2024, 1, 2, tzinfo=UTC)),
            ("s3", datetime(2024, 1, 3, tzinfo=UTC)),
        ]
        for session_id in ("s1", "s2", "s3"):
            _store_completion(store, session_id)
        response = client.get("/v1/chat/completions?after=chatcmpl-s1")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["chatcmpl-s2", "chatcmpl-s3"]

    def test_after_unknown_cursor_returns_empty_page(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """An unknown after cursor yields an empty page."""
        store.sessions = [("s1", datetime(2024, 1, 1, tzinfo=UTC))]
        _store_completion(store, "s1")
        response = client.get("/v1/chat/completions?after=chatcmpl-zzz")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["data"] == []
        # response_model_exclude_none drops null first_id/last_id from the wire payload.
        assert body.get("first_id") is None
        assert body.get("last_id") is None

    def test_limit_reports_has_more(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """A limit lower than the total sets has_more and truncates the page."""
        store.sessions = [
            (f"s{i}", datetime(2024, 1, i + 1, tzinfo=UTC)) for i in range(3)
        ]
        for i in range(3):
            _store_completion(store, f"s{i}")
        response = client.get("/v1/chat/completions?limit=2")
        assert response.status_code == 200, response.text
        body = response.json()
        assert len(body["data"]) == 2
        assert body["has_more"] is True

    def test_model_filter(self, client: TestClient, store: _StubStore) -> None:
        """Only completions generated by the given model are returned."""
        store.sessions = [
            ("s1", datetime(2024, 1, 1, tzinfo=UTC)),
            ("s2", datetime(2024, 1, 2, tzinfo=UTC)),
        ]
        _store_completion(store, "s1", model="model-a")
        _store_completion(store, "s2", model="model-b")
        response = client.get("/v1/chat/completions?model=model-b")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["chatcmpl-s2"]

    def test_metadata_filter(self, client: TestClient, store: _StubStore) -> None:
        """metadata[key]=value filters on the stored completion's metadata."""
        store.sessions = [
            ("s1", datetime(2024, 1, 1, tzinfo=UTC)),
            ("s2", datetime(2024, 1, 2, tzinfo=UTC)),
        ]
        _store_completion(store, "s1", metadata={"team": "a"})
        _store_completion(store, "s2", metadata={"team": "b"})
        response = client.get("/v1/chat/completions?metadata[team]=a")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["chatcmpl-s1"]

    def test_response_envelope(self, client: TestClient, store: _StubStore) -> None:
        """The list response carries the list envelope fields."""
        store.sessions = [
            ("s1", datetime(2024, 1, 1, tzinfo=UTC)),
            ("s2", datetime(2024, 1, 2, tzinfo=UTC)),
        ]
        _store_completion(store, "s1")
        _store_completion(store, "s2")
        response = client.get("/v1/chat/completions")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "list"
        assert body["first_id"] == "chatcmpl-s1"
        assert body["last_id"] == "chatcmpl-s2"

    def test_lists_chat_completion_kind_only(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """The route only scans sessions tagged as chat_completion."""
        response = client.get("/v1/chat/completions")
        assert response.status_code == 200, response.text
        assert store.list_sessions_kinds == ["chat_completion"]

    def test_non_404_load_error_propagates(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """A non-404 ApiError while loading a candidate session is not swallowed."""
        store.sessions = [("s1", datetime(2024, 1, 1, tzinfo=UTC))]
        store.load_errors["chatcmpl-s1"] = ApiError("throttled", status=429)
        response = client.get("/v1/chat/completions")
        assert response.status_code == 429

    def test_corrupt_stored_completion_is_skipped(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """A corrupt stored session is skipped instead of failing the whole list."""
        store.sessions = [
            ("s1", datetime(2024, 1, 1, tzinfo=UTC)),
            ("s2", datetime(2024, 1, 2, tzinfo=UTC)),
        ]
        store.load_errors["chatcmpl-s1"] = ValueError("invalid JSON")
        _store_completion(store, "s2")
        response = client.get("/v1/chat/completions")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["chatcmpl-s2"]


@pytest.mark.local
class TestUpdateChatCompletion:
    """POST /v1/chat/completions/{id}."""

    def test_replaces_metadata(self, client: TestClient, store: _StubStore) -> None:
        """A metadata body replaces the stored metadata."""
        _store_completion(store, "s1", metadata={"team": "a"})
        response = client.post(
            "/v1/chat/completions/chatcmpl-s1", json={"metadata": {"team": "b"}}
        )
        assert response.status_code == 200, response.text
        assert response.json()["metadata"] == {"team": "b"}

    def test_null_metadata_clears_it(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """A null metadata body clears the stored metadata."""
        _store_completion(store, "s1", metadata={"team": "a"})
        response = client.post(
            "/v1/chat/completions/chatcmpl-s1", json={"metadata": None}
        )
        assert response.status_code == 200, response.text
        assert "metadata" not in response.json()

    def test_persists_updated_document(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """The updated document is saved back to the store."""
        _store_completion(store, "s1")
        response = client.post(
            "/v1/chat/completions/chatcmpl-s1", json={"metadata": {"k": "v"}}
        )
        assert response.status_code == 200, response.text
        ((completion_id, document),) = store.saved
        assert completion_id == "chatcmpl-s1"
        assert document["response"]["metadata"] == {"k": "v"}

    def test_unknown_id_is_not_found(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """Updating an unknown stored chat completion surfaces as 404."""
        response = client.post(
            "/v1/chat/completions/chatcmpl-zzz", json={"metadata": {"k": "v"}}
        )
        assert response.status_code == 404


class TestStoredChatCompletionsLive:
    """Live stored chat completions lifecycle against AWS Bedrock sessions."""

    def test_store_lifecycle(
        self, openai_client: OpenAI, chat_model: str, use_official_api: bool
    ) -> None:
        """store=true persists; retrieve, messages, list, update, and delete work."""
        if use_official_api:
            pytest.skip("official API stores completions asynchronously (delayed)")
        from openai import NotFoundError  # noqa: PLC0415

        marker = uuid4().hex
        created = openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Reply with the word: banana"}],
            store=True,
            metadata={"test-marker": marker},
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
            assert messages[0].content is not None
            assert "banana" in messages[0].content

            page = openai_client.chat.completions.list(
                order="desc", metadata={"test-marker": marker}
            )
            assert created.id in [item.id for item in page.data]

            updated = openai_client.chat.completions.update(
                created.id, metadata={"test-marker": marker, "stage": "updated"}
            )
            assert cast("Any", updated).metadata == {
                "test-marker": marker,
                "stage": "updated",
            }
            reretrieved = openai_client.chat.completions.retrieve(created.id)
            assert cast("Any", reretrieved).metadata == {
                "test-marker": marker,
                "stage": "updated",
            }
        finally:
            deleted = openai_client.chat.completions.delete(created.id)
            assert deleted.deleted is True
        with pytest.raises(NotFoundError):
            openai_client.chat.completions.retrieve(created.id)
