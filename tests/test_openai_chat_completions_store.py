"""Tests for the stored chat completions surface: ``store=true`` and its CRUD routes.

The route layer is exercised in-process against stubbed persistence, so the
gateway-only behaviors (session-derived IDs, the streaming downgrade, the
foreign/corrupt-document guard) are asserted without touching Bedrock session
storage. The final class runs the same lifecycle live.

Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/retrieve
     https://docs.aws.amazon.com/bedrock/latest/userguide/sessions.html
     stdapi/routes/openai_chat_completions.py
"""

from datetime import UTC, datetime
from re import fullmatch
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import pytest
from sse_starlette import EventSourceResponse

from stdapi.api_errors import ApiError
from stdapi.responses_store import COMPLETION_ID_PATTERN

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from openai import OpenAI
    from starlette.testclient import TestClient

    from stdapi.models import ModelDetails
from stdapi.routes import openai_chat_completions
from stdapi.types.openai_chat_completions import ChatCompletion, CompletionCreateParams
from tests._helpers import make_model_details


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
        #: Overrides the returned completion ID, mimicking Mantle passthrough.
        self.upstream_id: str | None = None

    async def create_completion(
        self, request: CompletionCreateParams, completion_id: str, created: int
    ) -> ChatCompletion | EventSourceResponse:
        """Record the request and return a canned completion, or a stream when requested."""
        self.requests.append((request, completion_id))
        completion = _canned_completion(
            self.upstream_id or completion_id, request.model
        )
        if not request.stream:
            return completion

        async def _events() -> AsyncIterator[dict[str, str]]:
            yield {
                "event": "chat.completion.chunk",
                "data": completion.model_dump_json(by_alias=True, exclude_none=True),
            }

        return EventSourceResponse(_events())


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
        #: Completion IDs passed to ``load``, in call order.
        self.load_calls: list[str] = []

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
        declared = None
        if document is not None and isinstance(document.get("response"), dict):
            declared = _KIND_BY_OBJECT.get(document["response"].get("object"))
        if document is None or (declared is not None and declared != kind):
            msg = f"Chat completion with id '{completion_id}' not found."
            raise ApiError(msg, status=404)
        return document

    async def load(self, completion_id: str, kind: str) -> dict[str, Any]:
        self.load_calls.append(completion_id)
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
def backend(monkeypatch: pytest.MonkeyPatch) -> _StubChatBackend:
    """Stub model validation and the generation backend."""

    async def _validate_model(
        model_id: str, *_args: object, **_kwargs: object
    ) -> ModelDetails:
        return make_model_details(model_id)

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
    """store=true on POST /v1/chat/completions.

    Unlike upstream, a stored completion's ID is derived from the Bedrock
    session that backs it (``chatcmpl-{session_id}``), and ``store`` is
    silently downgraded when it cannot be honored.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         stdapi/routes/openai_chat_completions.py:create_chat_completion
    """

    def test_store_persists_completion(
        self, app_client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """The completion is generated under a chat_completion session and persisted.

        The session kind tag is what keeps Responses documents from being read
        back through the Chat Completions routes.
        """
        response = app_client.post(
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
        assert response.json()["object"] == "chat.completion"
        assert response.json()["metadata"] == {"team": "x"}
        assert store.created_kinds == ["chat_completion"]
        assert not store.discarded
        ((completion_id, document),) = store.saved
        assert completion_id == "chatcmpl-sess-1"
        assert document["messages"][0]["content"] == "hello"
        assert document["response"]["id"] == "chatcmpl-sess-1"
        assert document["response"]["metadata"] == {"team": "x"}

    def test_store_rewrites_upstream_backend_id(
        self, app_client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """A backend that ignores completion_id (e.g. Mantle passthrough) is rewritten when stored.

        The stored surface addresses documents by the server-assigned ID, so a
        backend-chosen ID must not survive into the response or the document.
        """
        backend.upstream_id = "chatcmpl-upstream-xyz"
        response = app_client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "store": True,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"] == "chatcmpl-sess-1"
        ((_, requested_id),) = backend.requests
        assert requested_id == "chatcmpl-sess-1", (
            "the backend is handed the session-derived ID even when it ignores it"
        )
        ((completion_id, document),) = store.saved
        assert completion_id == "chatcmpl-sess-1"
        assert document["response"]["id"] == "chatcmpl-sess-1"

    def test_without_store_upstream_backend_id_passes_through(
        self, app_client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """Without store=true, the backend's own upstream ID is returned untouched."""
        backend.upstream_id = "chatcmpl-upstream-xyz"
        response = app_client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"] == "chatcmpl-upstream-xyz"
        assert not store.created_kinds
        assert not store.saved

    def test_store_with_stream_is_ignored(
        self, app_client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """store=true with streaming is ignored: no session is involved.

        Upstream has no such restriction; here the request is still served as a
        stream, only unstored, with the request-scoped ID instead of a
        session-derived one.
        """
        response = app_client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "store": True,
                "stream": True,
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert not store.created_kinds
        assert not store.saved
        assert "chatcmpl-sess-1" not in response.text

    def test_store_without_session_storage_is_ignored(
        self,
        app_client: TestClient,
        backend: _StubChatBackend,
        store: _StubStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """store=true is ignored when session storage is unavailable.

        The completion is still served, with a request-scoped ID that must
        remain a syntactically valid completion ID (never ``chatcmpl-None``).
        """

        async def _unavailable(_kind: str) -> None:
            return None

        monkeypatch.setattr(
            openai_chat_completions, "try_create_stored_response_session", _unavailable
        )
        response = app_client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "store": True,
            },
        )
        assert response.status_code == 200, response.text
        assert fullmatch(COMPLETION_ID_PATTERN, response.json()["id"]), (
            "the fallback ID must still match the stored-completion path pattern"
        )
        assert "None" not in response.json()["id"]
        assert not store.saved
        assert not store.discarded

    def test_without_store_nothing_is_persisted(
        self, app_client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """Without store=true no session is involved."""
        response = app_client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"] != "chatcmpl-sess-1"
        assert not store.created_kinds
        assert not store.saved

    def test_generation_failure_discards_pending_session(
        self, app_client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """A failed generation discards the pending stored session.

        The session is created before generation, so an aborted generation must
        not leave an empty session behind.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """

        async def _raise(
            _request: CompletionCreateParams, _completion_id: str, _created: int
        ) -> ChatCompletion:
            msg = "backend failure"
            raise ApiError(msg, status=502)

        backend.create_completion = _raise  # type: ignore[method-assign, assignment]
        response = app_client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "store": True,
            },
        )
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "server_error"
        assert response.json()["error"]["message"] == "backend failure"
        assert store.discarded == ["chatcmpl-sess-1"]
        assert not store.saved

    def test_save_failure_discards_pending_session(
        self,
        app_client: TestClient,
        backend: _StubChatBackend,
        store: _StubStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed save after a successful generation discards the pending session.

        The generated completion is not returned either: a stored completion the
        client could not retrieve afterwards would be worse than an error.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """

        async def _raise(_completion_id: str, _document: dict[str, Any]) -> None:
            msg = "save failure"
            raise ApiError(msg, status=502)

        monkeypatch.setattr(openai_chat_completions, "save_stored_response", _raise)
        response = app_client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": [{"role": "user", "content": "hello"}],
                "store": True,
            },
        )
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "server_error"
        assert response.json()["error"]["message"] == "save failure"
        assert store.discarded == ["chatcmpl-sess-1"]


@pytest.mark.local
class TestStoredChatCompletionRoutes:
    """GET/DELETE /v1/chat/completions/{id} and messages listing.

    A document that is missing, corrupt, or tagged as another stored kind is
    reported as a plain 404 rather than a 500, so a foreign session can never
    be read through these routes.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/retrieve
         stdapi/routes/openai_chat_completions.py:_malformed_stored_document
    """

    def test_retrieve_stored_completion(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """A stored chat completion is returned as-is."""
        store.documents["chatcmpl-sess-1"] = {
            "messages": [{"role": "user", "content": "hello"}],
            "response": _canned_completion("chatcmpl-sess-1", "m").model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
        }
        response = app_client.get("/v1/chat/completions/chatcmpl-sess-1")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"] == "chatcmpl-sess-1"
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "answer"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert store.load_calls == ["chatcmpl-sess-1"]

    def test_retrieve_unknown_is_not_found(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """An unknown stored chat completion surfaces as 404."""
        response = app_client.get("/v1/chat/completions/chatcmpl-zzz")
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "chatcmpl-zzz" in error["message"]

    def test_delete_stored_completion(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Deletion returns a confirmation object.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/delete
        """
        store.documents["chatcmpl-sess-1"] = {"messages": [], "response": {}}
        response = app_client.delete("/v1/chat/completions/chatcmpl-sess-1")
        assert response.status_code == 200, response.text
        assert response.json() == {
            "id": "chatcmpl-sess-1",
            "object": "chat.completion.deleted",
            "deleted": True,
        }
        assert store.deleted == ["chatcmpl-sess-1"]

    def test_retrieve_response_kind_session_is_not_found(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Retrieving a response-kind session as a chat completion 404s.

        Both surfaces store documents in Bedrock sessions, so the ``kind`` tag
        is the only thing keeping a Responses document out of this route.
        """
        store.documents["chatcmpl-sess-1"] = {
            "input": [],
            "response": {"object": "response"},
        }
        response = app_client.get("/v1/chat/completions/chatcmpl-sess-1")
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "chatcmpl-sess-1" in error["message"]

    def test_delete_response_kind_session_is_not_found_and_not_deleted(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Deleting a response-kind session as a chat completion 404s without deleting."""
        store.documents["chatcmpl-sess-1"] = {
            "input": [],
            "response": {"object": "response"},
        }
        response = app_client.delete("/v1/chat/completions/chatcmpl-sess-1")
        assert response.status_code == 404
        assert "chatcmpl-sess-1" in response.json()["error"]["message"]
        assert not store.deleted

    @pytest.mark.parametrize(
        "document", [{"response": None}, {}], ids=["null-response", "no-response-key"]
    )
    def test_retrieve_malformed_document_is_not_found(
        self, app_client: TestClient, store: _StubStore, document: dict[str, Any]
    ) -> None:
        """A foreign or corrupt stored document 404s instead of 500ing."""
        store.documents["chatcmpl-sess-1"] = document
        response = app_client.get("/v1/chat/completions/chatcmpl-sess-1")
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "chatcmpl-sess-1" in error["message"]

    def test_messages_listing_with_cursor(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Messages get IDs, keep conversation order, and page with after.

        Message IDs are positional (``msg-{index}``), which is what makes them
        usable as an opaque cursor into the stored request messages.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/subresources/messages/methods/list
        """
        store.documents["chatcmpl-sess-1"] = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
            "response": {},
        }
        response = app_client.get("/v1/chat/completions/chatcmpl-sess-1/messages")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "list"
        assert [message["id"] for message in body["data"]] == ["msg-0", "msg-1"]
        assert [message["role"] for message in body["data"]] == ["system", "user"]
        assert body["data"][1]["content"] == "hello"
        assert body["has_more"] is False
        assert (body["first_id"], body["last_id"]) == ("msg-0", "msg-1")

        response = app_client.get(
            "/v1/chat/completions/chatcmpl-sess-1/messages?after=msg-0&limit=1"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert [message["id"] for message in body["data"]] == ["msg-1"]
        assert body["has_more"] is False, "msg-1 is the last message"

    def test_messages_listing_splits_array_content(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Array content is split into concatenated text `content` and `content_parts`.

        A string-content message keeps a bare `content` with no `content_parts`,
        which is how a client tells the two request shapes apart on read-back.
        """
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
        response = app_client.get("/v1/chat/completions/chatcmpl-sess-1/messages")
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
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """An `after` cursor matching no message 404s instead of an empty page."""
        store.documents["chatcmpl-sess-1"] = {
            "messages": [{"role": "user", "content": "hello"}],
            "response": {},
        }
        response = app_client.get(
            "/v1/chat/completions/chatcmpl-sess-1/messages?after=msg-99"
        )
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "msg-99" in error["message"]

    def test_messages_listing_order_desc(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """order=desc reverses the conversation order."""
        store.documents["chatcmpl-sess-1"] = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
            "response": {},
        }
        response = app_client.get(
            "/v1/chat/completions/chatcmpl-sess-1/messages?order=desc"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert [message["id"] for message in body["data"]] == ["msg-1", "msg-0"]
        assert [message["content"] for message in body["data"]] == ["hello", "sys"]

    def test_messages_listing_after_combined_with_desc_order(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """After applies to the already-reversed sequence when order=desc."""
        store.documents["chatcmpl-sess-1"] = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
            ],
            "response": {},
        }
        response = app_client.get(
            "/v1/chat/completions/chatcmpl-sess-1/messages?order=desc&after=msg-2"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        # desc reverses to [msg-2, msg-1, msg-0]; after msg-2 leaves [msg-1, msg-0].
        assert [message["id"] for message in body["data"]] == ["msg-1", "msg-0"]

    def test_messages_listing_truncates_and_reports_has_more(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """A limit smaller than the remaining messages truncates the page and sets has_more."""
        store.documents["chatcmpl-sess-1"] = {
            "messages": [{"role": "user", "content": f"m{i}"} for i in range(5)],
            "response": {},
        }
        response = app_client.get(
            "/v1/chat/completions/chatcmpl-sess-1/messages?limit=2"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert [message["id"] for message in body["data"]] == ["msg-0", "msg-1"]
        assert body["has_more"] is True
        assert body["last_id"] == "msg-1", "last_id is the cursor for the next page"

    def test_messages_listing_cursor_at_last_message_returns_empty_page(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """An after cursor matching the last message returns an empty page, not a 404.

        Exhausting a cursor is normal pagination, unlike a cursor that never
        matched any message, which is a client error.
        """
        store.documents["chatcmpl-sess-1"] = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
            "response": {},
        }
        response = app_client.get(
            "/v1/chat/completions/chatcmpl-sess-1/messages?after=msg-1"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["data"] == []
        assert body["has_more"] is False
        assert "first_id" not in body, "an empty page carries no cursor bounds"
        assert "last_id" not in body

    def test_messages_listing_non_dict_message_entry_is_not_found(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """A non-dict entry in the stored messages 404s instead of 500ing."""
        store.documents["chatcmpl-sess-1"] = {
            "messages": ["not-a-dict"],
            "response": {},
        }
        response = app_client.get("/v1/chat/completions/chatcmpl-sess-1/messages")
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "chatcmpl-sess-1" in error["message"]

    def test_store_round_trip_preserves_multipart_and_tool_messages(
        self, app_client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """A store request with multi-part and tool messages round-trips through /messages.

        Only text and image parts are re-exposed as ``content_parts``, and
        assistant/tool messages keep their `tool_calls`/`tool_call_id` linkage so
        a stored agent turn can be replayed.

        Ref: https://developers.openai.com/api/docs/guides/function-calling
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://x/img.png"}},
                    {"type": "text", "text": "describe this"},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result data"},
        ]
        response = app_client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "messages": messages,
                "store": True,
            },
        )
        assert response.status_code == 200, response.text
        completion_id = response.json()["id"]
        ((_, document),) = store.saved
        assert document["messages"][0]["content"][0]["type"] == "image_url"
        assert document["messages"][1]["tool_calls"][0]["id"] == "call_1"
        assert document["messages"][2]["tool_call_id"] == "call_1"
        # The stub's save() doesn't feed load(): register the saved document.
        store.documents[completion_id] = document

        messages_response = app_client.get(
            f"/v1/chat/completions/{completion_id}/messages"
        )
        assert messages_response.status_code == 200, messages_response.text
        body = messages_response.json()
        array_message, tool_call_message, tool_message = body["data"]
        assert array_message["content"] == "describe this"
        assert [part["type"] for part in array_message["content_parts"]] == [
            "image_url",
            "text",
        ]
        assert "content_parts" not in tool_call_message
        assert tool_call_message["tool_calls"][0]["id"] == "call_1"
        assert "content_parts" not in tool_message
        assert tool_message["content"] == "result data"


@pytest.mark.local
class TestListChatCompletions:
    """GET /v1/chat/completions.

    Bedrock session storage has no server-side filtering or ordering, so the
    route lists sessions, sorts them by creation time and loads candidate
    documents in batches of ``_LIST_LOAD_BATCH_SIZE`` to apply the filters.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/list
         stdapi/routes/openai_chat_completions.py:list_chat_completions
    """

    def test_default_order_is_ascending_by_created_at(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Without an explicit order, the oldest session comes first.

        The stub returns the sessions newest-first so the ascending order can
        only come from the route's own sort, not from the store's iteration order.
        """
        store.sessions = [
            ("s2", datetime(2024, 1, 2, tzinfo=UTC)),
            ("s1", datetime(2024, 1, 1, tzinfo=UTC)),
        ]
        _store_completion(store, "s1")
        _store_completion(store, "s2")
        response = app_client.get("/v1/chat/completions")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["chatcmpl-s1", "chatcmpl-s2"]

    def test_order_desc_returns_newest_first(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """order=desc reverses the creation-time order."""
        store.sessions = [
            ("s1", datetime(2024, 1, 1, tzinfo=UTC)),
            ("s2", datetime(2024, 1, 2, tzinfo=UTC)),
        ]
        _store_completion(store, "s1")
        _store_completion(store, "s2")
        response = app_client.get("/v1/chat/completions?order=desc")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["chatcmpl-s2", "chatcmpl-s1"]

    def test_after_cursor_returns_later_items(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """A known after cursor skips itself and every earlier item."""
        store.sessions = [
            ("s1", datetime(2024, 1, 1, tzinfo=UTC)),
            ("s2", datetime(2024, 1, 2, tzinfo=UTC)),
            ("s3", datetime(2024, 1, 3, tzinfo=UTC)),
        ]
        for session_id in ("s1", "s2", "s3"):
            _store_completion(store, session_id)
        response = app_client.get("/v1/chat/completions?after=chatcmpl-s1")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["chatcmpl-s2", "chatcmpl-s3"]

    def test_after_unknown_cursor_is_not_found(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """An unknown after cursor 404s instead of stranding pagination on an empty page."""
        store.sessions = [("s1", datetime(2024, 1, 1, tzinfo=UTC))]
        _store_completion(store, "s1")
        response = app_client.get("/v1/chat/completions?after=chatcmpl-zzz")
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "chatcmpl-zzz" in error["message"]

    def test_limit_reports_has_more(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """A limit lower than the total sets has_more and truncates the page."""
        store.sessions = [
            (f"s{i}", datetime(2024, 1, i + 1, tzinfo=UTC)) for i in range(3)
        ]
        for i in range(3):
            _store_completion(store, f"s{i}")
        response = app_client.get("/v1/chat/completions?limit=2")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["chatcmpl-s0", "chatcmpl-s1"]
        assert body["has_more"] is True
        assert body["last_id"] == "chatcmpl-s1"

    def test_unfiltered_has_more_skips_probe_load(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Without model/metadata filters, has_more is answered without loading the limit+1'th document.

        Every remaining session ID is a match by definition, so the extra
        Bedrock read a probe load would cost is avoidable.
        """
        store.sessions = [
            (f"s{i}", datetime(2024, 1, i + 1, tzinfo=UTC)) for i in range(3)
        ]
        for i in range(3):
            _store_completion(store, f"s{i}")
        response = app_client.get("/v1/chat/completions?limit=2")
        assert response.status_code == 200, response.text
        assert response.json()["has_more"] is True
        assert store.load_calls == ["chatcmpl-s0", "chatcmpl-s1"]

    def test_model_filter(self, app_client: TestClient, store: _StubStore) -> None:
        """Only completions generated by the given model are returned."""
        store.sessions = [
            ("s1", datetime(2024, 1, 1, tzinfo=UTC)),
            ("s2", datetime(2024, 1, 2, tzinfo=UTC)),
        ]
        _store_completion(store, "s1", model="model-a")
        _store_completion(store, "s2", model="model-b")
        response = app_client.get("/v1/chat/completions?model=model-b")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["chatcmpl-s2"]
        assert [item["model"] for item in body["data"]] == ["model-b"]
        assert body["has_more"] is False

    def test_metadata_filter(self, app_client: TestClient, store: _StubStore) -> None:
        """metadata[key]=value filters on the stored completion's metadata.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/list
        """
        store.sessions = [
            ("s1", datetime(2024, 1, 1, tzinfo=UTC)),
            ("s2", datetime(2024, 1, 2, tzinfo=UTC)),
        ]
        _store_completion(store, "s1", metadata={"team": "a"})
        _store_completion(store, "s2", metadata={"team": "b"})
        response = app_client.get("/v1/chat/completions?metadata[team]=a")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["chatcmpl-s1"]
        assert [item["metadata"] for item in body["data"]] == [{"team": "a"}]
        assert body["has_more"] is False

    def test_response_envelope(self, app_client: TestClient, store: _StubStore) -> None:
        """The list response carries the list envelope fields."""
        store.sessions = [
            ("s1", datetime(2024, 1, 1, tzinfo=UTC)),
            ("s2", datetime(2024, 1, 2, tzinfo=UTC)),
        ]
        _store_completion(store, "s1")
        _store_completion(store, "s2")
        response = app_client.get("/v1/chat/completions")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "list"
        assert body["first_id"] == "chatcmpl-s1"
        assert body["last_id"] == "chatcmpl-s2"
        assert body["has_more"] is False
        assert [item["object"] for item in body["data"]] == [
            "chat.completion",
            "chat.completion",
        ]

    def test_lists_chat_completion_kind_only(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """The route only scans sessions tagged as chat_completion.

        Responses and Chat Completions share the same session store, so the kind
        tag is what keeps the two listings disjoint.
        """
        response = app_client.get("/v1/chat/completions")
        assert response.status_code == 200, response.text
        assert store.list_sessions_kinds == ["chat_completion"]
        assert response.json()["data"] == []

    def test_non_404_load_error_propagates(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """A non-404 ApiError while loading a candidate session is not swallowed.

        Only a 404 means "deleted between the scan and the read"; a throttle
        must surface so the client retries instead of seeing a short list.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """
        store.sessions = [("s1", datetime(2024, 1, 1, tzinfo=UTC))]
        store.load_errors["chatcmpl-s1"] = ApiError("throttled", status=429)
        response = app_client.get("/v1/chat/completions")
        assert response.status_code == 429
        error = response.json()["error"]
        assert error["type"] == "rate_limit_error"
        assert error["message"] == "throttled"

    def test_list_scans_multiple_batches_and_stops_early(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """More than one load batch is scanned, stopping once the limit is reached.

        A limit of 12 spans two ``_LIST_LOAD_BATCH_SIZE`` (10) batches, and no
        document past the limit is read.
        """
        store.sessions = [
            (f"s{i}", datetime(2024, 1, i + 1, tzinfo=UTC)) for i in range(15)
        ]
        for i in range(15):
            _store_completion(store, f"s{i}")
        response = app_client.get("/v1/chat/completions?limit=12")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == [
            f"chatcmpl-s{i}" for i in range(12)
        ]
        assert body["has_more"] is True
        assert store.load_calls == [f"chatcmpl-s{i}" for i in range(12)], (
            "only the first 12 documents may be loaded, across two batches"
        )

    def test_corrupt_stored_completion_is_skipped(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """A corrupt stored session is skipped instead of failing the whole list.

        A document that no longer validates would otherwise make the whole
        listing unusable.
        """
        store.sessions = [
            ("s1", datetime(2024, 1, 1, tzinfo=UTC)),
            ("s2", datetime(2024, 1, 2, tzinfo=UTC)),
        ]
        store.load_errors["chatcmpl-s1"] = ValueError("invalid JSON")
        _store_completion(store, "s2")
        response = app_client.get("/v1/chat/completions")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["chatcmpl-s2"]
        assert store.load_calls == ["chatcmpl-s1", "chatcmpl-s2"], (
            "the corrupt candidate must be attempted, then dropped from the page"
        )


@pytest.mark.local
class TestUpdateChatCompletion:
    """POST /v1/chat/completions/{id}.

    ``metadata`` is the only updatable field, and the update is written back to
    the store so a later retrieve sees it.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/update
         stdapi/routes/openai_chat_completions.py:update_chat_completion
    """

    def test_replaces_metadata(self, app_client: TestClient, store: _StubStore) -> None:
        """A metadata body replaces the stored metadata."""
        _store_completion(store, "s1", metadata={"team": "a"})
        response = app_client.post(
            "/v1/chat/completions/chatcmpl-s1", json={"metadata": {"team": "b"}}
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"] == "chatcmpl-s1"
        assert response.json()["metadata"] == {"team": "b"}

    def test_null_metadata_clears_it(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """A null metadata body clears the stored metadata."""
        _store_completion(store, "s1", metadata={"team": "a"})
        response = app_client.post(
            "/v1/chat/completions/chatcmpl-s1", json={"metadata": None}
        )
        assert response.status_code == 200, response.text
        assert "metadata" not in response.json()
        ((_, document),) = store.saved
        assert "metadata" not in document["response"]

    def test_persists_updated_document(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """The updated document is saved back to the store."""
        _store_completion(store, "s1")
        response = app_client.post(
            "/v1/chat/completions/chatcmpl-s1", json={"metadata": {"k": "v"}}
        )
        assert response.status_code == 200, response.text
        ((completion_id, document),) = store.saved
        assert completion_id == "chatcmpl-s1"
        assert document["response"]["metadata"] == {"k": "v"}

    def test_unknown_id_is_not_found(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Updating an unknown stored chat completion surfaces as 404."""
        response = app_client.post(
            "/v1/chat/completions/chatcmpl-zzz", json={"metadata": {"k": "v"}}
        )
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "chatcmpl-zzz" in error["message"]
        assert not store.saved

    @pytest.mark.parametrize(
        "document", [{"response": None}, {}], ids=["null-response", "no-response-key"]
    )
    def test_update_malformed_document_is_not_found(
        self, app_client: TestClient, store: _StubStore, document: dict[str, Any]
    ) -> None:
        """A foreign or corrupt stored document 404s instead of 500ing."""
        store.documents["chatcmpl-sess-1"] = document
        response = app_client.post(
            "/v1/chat/completions/chatcmpl-sess-1", json={"metadata": {"k": "v"}}
        )
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "chatcmpl-sess-1" in error["message"]
        assert not store.saved, "a document that failed to update must not be written"


@pytest.mark.local
class TestStoredChatCompletionRoutesAuthRejection:
    """A missing bearer token is rejected with a 401 OpenAI envelope, no store access.

    Uses the session-wide ``test_client`` (lifespan-started, unlike the
    lifespan-free ``app_client`` fixture) so the auth handler is actually
    initialized and able to reject a missing token.

    Ref: https://developers.openai.com/api/docs/guides/error-codes
         https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/api_providers/openai.py:_format_error
    """

    @pytest.fixture(autouse=True)
    def _skip_non_local(self, test_client: TestClient | None) -> None:
        """Skip when running against a remote server instead of the in-process app."""
        if not test_client:
            pytest.skip("Unit test only for local, in-process runs.")

    @pytest.mark.parametrize(
        ("method", "path", "json_body"),
        [
            ("get", "/v1/chat/completions/chatcmpl-sess-1", None),
            ("delete", "/v1/chat/completions/chatcmpl-sess-1", None),
            ("get", "/v1/chat/completions/chatcmpl-sess-1/messages", None),
            ("get", "/v1/chat/completions", None),
            ("post", "/v1/chat/completions/chatcmpl-sess-1", {"metadata": {"k": "v"}}),
        ],
        ids=["retrieve", "delete", "messages", "list", "update"],
    )
    def test_missing_bearer_token_is_rejected(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None,
        test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No Authorization header yields 401 without reaching the store layer.

        The message is the fixed ``Unauthorized`` string: authentication
        failures must not leak why the credential was refused.
        """
        calls: list[str] = []

        async def _counting_load(completion_id: str, _kind: str) -> dict[str, Any]:
            calls.append(completion_id)
            return {"messages": [], "response": {}}

        async def _counting_list_sessions(_kind: str) -> list[tuple[str, Any]]:
            calls.append("list_sessions")
            return []

        monkeypatch.setattr(
            openai_chat_completions, "load_stored_response", _counting_load
        )
        monkeypatch.setattr(
            openai_chat_completions, "list_stored_sessions", _counting_list_sessions
        )
        response = test_client.request(method.upper(), path, json=json_body)

        assert response.status_code == 401
        body = response.json()
        assert set(body.keys()) == {"error"}
        err = body["error"]
        assert set(err.keys()) == {"message", "type", "param", "code"}
        assert err["type"] == "authentication_error"
        assert err["message"] == "Unauthorized"
        assert not calls


@pytest.mark.local
class TestInvalidCompletionIdPattern:
    """A completion ID that fails the path pattern is rejected before the store layer.

    Pins the current OpenAI-shaped 400 ``invalid_request_error`` envelope
    produced by FastAPI/Pydantic path validation (see ``main.py``'s
    ``RequestValidationError`` handler). The ``resp_`` cases matter because the
    Responses surface uses that prefix over the same session store.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/routes/openai_chat_completions.py:_CompletionId
         stdapi/main.py:handle_validation_exception
    """

    @pytest.mark.parametrize(
        ("method", "path", "json_body"),
        [
            ("get", "/v1/chat/completions/resp_x", None),
            ("delete", "/v1/chat/completions/resp_x", None),
            ("get", "/v1/chat/completions/resp_x/messages", None),
            ("post", "/v1/chat/completions/resp_x", {"metadata": {"k": "v"}}),
            ("get", "/v1/chat/completions/chatcmpl_x!", None),
            ("get", "/v1/chat/completions/x", None),
        ],
        ids=[
            "retrieve-resp-prefix",
            "delete-resp-prefix",
            "messages-resp-prefix",
            "update-resp-prefix",
            "retrieve-bad-char",
            "retrieve-too-short",
        ],
    )
    def test_invalid_id_pattern_is_rejected(
        self,
        app_client: TestClient,
        store: _StubStore,
        method: str,
        path: str,
        json_body: dict[str, Any] | None,
    ) -> None:
        """A path-pattern-invalid completion ID 400s with an OpenAI invalid_request_error envelope."""
        response = app_client.request(method.upper(), path, json=json_body)
        assert response.status_code == 400, response.text
        body = response.json()
        assert set(body.keys()) == {"error"}
        err = body["error"]
        assert set(err.keys()) == {"message", "type", "param", "code"}
        assert err["type"] == "invalid_request_error"
        assert not store.load_calls, "rejection must happen before any store read"
        assert not store.deleted


class TestStoredChatCompletionsLive:
    """Live stored chat completions lifecycle against AWS Bedrock sessions.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/retrieve
         https://docs.aws.amazon.com/bedrock/latest/userguide/sessions.html
    """

    @pytest.mark.gateway("official API stores completions asynchronously (delayed)")
    def test_store_lifecycle(self, openai_client: OpenAI, chat_model: str) -> None:
        """store=true persists; retrieve, messages, list, update, and delete work.

        ``/messages`` returns the *request* messages, so the prompt text is the
        deterministic thing to assert on rather than the generated answer.
        """
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
            assert messages[0].id == "msg-0"
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
            assert deleted.id == created.id
            assert deleted.object == "chat.completion.deleted"
        with pytest.raises(NotFoundError) as excinfo:
            openai_client.chat.completions.retrieve(created.id)
        assert excinfo.value.status_code == 404
        assert created.id in excinfo.value.message
