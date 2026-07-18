"""Tests for stored responses routes and previous_response_id (unit)."""

import contextlib
from typing import TYPE_CHECKING, Any

import pytest
from openai import NotFoundError
from openai.types.responses import ResponseInputMessageItem, ResponseInputText
from starlette.testclient import TestClient

from stdapi.api_errors import ApiError

if TYPE_CHECKING:
    from openai import OpenAI
from stdapi.models import ModelDetails
from stdapi.routes import openai_responses
from stdapi.types.openai_responses import (
    EasyInputMessage,
    InputTokensDetails,
    OutputTokensDetails,
    Response,
    ResponseCreateParams,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)


def _canned_response(response_id: str, model: str) -> Response:
    return Response(
        id=response_id,
        created_at=1752000000.0,
        model=model,
        object="response",
        output=[
            ResponseOutputMessage(
                id="msg-out",
                content=[
                    ResponseOutputText(
                        annotations=[], text="answer", type="output_text"
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
        usage=ResponseUsage(
            input_tokens=1,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=1,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=2,
        ),
    )


class _StubChatBackend:
    """Stub chat backend recording generation requests."""

    def __init__(self) -> None:
        self.requests: list[tuple[ResponseCreateParams, str]] = []

    def native_store_supported(self) -> bool:
        """Local-store stub: no Mantle native storage."""
        return False

    async def create_response(
        self,
        request: ResponseCreateParams,
        response_id: str,
        created_at: float,
        moderation_builder: Any = None,  # noqa: ANN401
    ) -> Response:
        """Record the request and return a canned response."""
        self.requests.append((request, response_id))
        return _canned_response(response_id, request.model)


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

    async def create_session(self, kind: str) -> str:
        self.created_kinds.append(kind)
        return "sess-1"

    async def save(self, response_id: str, document: dict[str, Any]) -> None:
        self.saved.append((response_id, document))

    def _document_or_not_found(self, response_id: str, kind: str) -> dict[str, Any]:
        """Return the stored document, 404ing when absent or of a different kind."""
        document = self.documents.get(response_id)
        declared = document and _KIND_BY_OBJECT.get(
            document.get("response", {}).get("object")
        )
        if document is None or (declared is not None and declared != kind):
            msg = f"Response with id '{response_id}' not found."
            raise ApiError(msg, status=404)
        return document

    async def load(self, response_id: str, kind: str) -> dict[str, Any]:
        return self._document_or_not_found(response_id, kind)

    async def delete(self, response_id: str, kind: str) -> None:
        self._document_or_not_found(response_id, kind)
        self.deleted.append(response_id)

    async def discard(self, response_id: str, kind: str) -> None:
        del kind
        self.discarded.append(response_id)


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
    monkeypatch.setattr(openai_responses, "validate_model", _validate_model)
    monkeypatch.setattr(openai_responses, "get_chat_model", lambda _model_id: stub)
    return stub


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _StubStore:
    """Stub the stored-response persistence functions."""
    stub = _StubStore()
    monkeypatch.setattr(
        openai_responses, "try_create_stored_response_session", stub.create_session
    )
    monkeypatch.setattr(openai_responses, "save_stored_response", stub.save)
    monkeypatch.setattr(openai_responses, "load_stored_response", stub.load)
    monkeypatch.setattr(openai_responses, "delete_stored_response", stub.delete)
    monkeypatch.setattr(
        openai_responses, "discard_stored_response_session", stub.discard
    )
    return stub


@pytest.mark.local
class TestStoreOnCreate:
    """store=true on POST /v1/responses."""

    def test_store_persists_response(
        self, client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """The response is generated under the session ID and persisted."""
        response = client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "hello",
                "instructions": "sys",
                "store": True,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"] == "resp-sess-1"
        assert store.created_kinds == ["response"]
        ((response_id, document),) = store.saved
        assert response_id == "resp-sess-1"
        assert document["input"] == "hello"
        assert "instructions" not in document
        assert document["response"]["id"] == "resp-sess-1"
        (_, generated_id) = backend.requests[0]
        assert generated_id == "resp-sess-1"

    def test_store_with_stream_is_ignored(
        self, client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """store=true with streaming is ignored: no session is involved."""
        response = client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "hello",
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
            openai_responses, "try_create_stored_response_session", _unavailable
        )
        response = client.post(
            "/v1/responses",
            json={"model": "amazon.nova-micro-v1:0", "input": "hello", "store": True},
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"].startswith("resp-")
        assert "None" not in response.json()["id"]
        assert not store.saved
        assert not store.discarded

    def test_without_store_nothing_is_persisted(
        self, client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """Without store=true no session is involved."""
        response = client.post(
            "/v1/responses", json={"model": "amazon.nova-micro-v1:0", "input": "hello"}
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"].startswith("resp-")
        assert response.json()["id"] != "resp-sess-1"
        assert not store.saved

    def test_store_and_list_input_items_with_message_object_input(
        self, client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """A message-object input round-trips through store and input_items listing."""
        response = client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": [{"role": "user", "content": "question"}],
                "store": True,
            },
        )
        assert response.status_code == 200, response.text
        response_id = response.json()["id"]
        ((_, document),) = store.saved
        store.documents[response_id] = document

        items_response = client.get(f"/v1/responses/{response_id}/input_items")
        assert items_response.status_code == 200, items_response.text
        assert items_response.json()["data"][0]["content"][0] == {
            "type": "input_text",
            "text": "question",
        }

    def test_generation_failure_discards_pending_session(
        self, client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """A failed generation discards the pending stored session."""

        async def _raise(
            _request: ResponseCreateParams,
            _response_id: str,
            _created_at: float,
            moderation_builder: Any = None,  # noqa: ANN401, ARG001
        ) -> Response:
            msg = "backend failure"
            raise ApiError(msg, status=502)

        backend.create_response = _raise  # type: ignore[method-assign, assignment]
        response = client.post(
            "/v1/responses",
            json={"model": "amazon.nova-micro-v1:0", "input": "hello", "store": True},
        )
        assert response.status_code == 502
        assert store.discarded == ["resp-sess-1"]

    def test_save_failure_discards_pending_session(
        self,
        client: TestClient,
        backend: _StubChatBackend,
        store: _StubStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed save after a successful generation discards the pending session."""

        async def _raise(_response_id: str, _document: dict[str, Any]) -> None:
            msg = "save failure"
            raise ApiError(msg, status=502)

        monkeypatch.setattr(openai_responses, "save_stored_response", _raise)
        response = client.post(
            "/v1/responses",
            json={"model": "amazon.nova-micro-v1:0", "input": "hello", "store": True},
        )
        assert response.status_code == 502
        assert store.discarded == ["resp-sess-1"]


@pytest.mark.local
class TestPreviousResponseId:
    """previous_response_id continuation on POST /v1/responses."""

    def test_previous_conversation_is_prepended(
        self, client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """The stored input and output precede the new input."""
        store.documents["resp-sess-1"] = {
            "input": [{"role": "user", "content": "first"}],
            "instructions": "old sys",
            "response": _canned_response("resp-sess-1", "m").model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
        }
        response = client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "second",
                "previous_response_id": "resp-sess-1",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["previous_response_id"] == "resp-sess-1"
        ((request, _),) = backend.requests
        assert request.previous_response_id is None
        assert request.instructions is None
        assert isinstance(request.input, list)
        assert isinstance(request.input[0], EasyInputMessage)
        assert request.input[0].content == "first"
        assert request.input[1].content[0].text == "answer"  # type: ignore[union-attr]
        assert isinstance(request.input[2], EasyInputMessage)
        assert request.input[2].content == "second"

    def test_unknown_previous_response_is_not_found(
        self, client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """An unknown previous_response_id surfaces as 404."""
        response = client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "x",
                "previous_response_id": "resp-zzz",
            },
        )
        assert response.status_code == 404
        assert not backend.requests


@pytest.mark.local
class TestStoredResponseRoutes:
    """GET/DELETE /v1/responses/{id} and input items listing."""

    def test_retrieve_stored_response(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """A stored response is returned as-is."""
        store.documents["resp-sess-1"] = {
            "input": "hello",
            "response": _canned_response("resp-sess-1", "m").model_dump(
                mode="json", by_alias=True, exclude_none=True
            ),
        }
        response = client.get("/v1/responses/resp-sess-1")
        assert response.status_code == 200, response.text
        assert response.json()["id"] == "resp-sess-1"
        assert response.json()["output"][0]["content"][0]["text"] == "answer"

    def test_retrieve_unknown_is_not_found(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """An unknown stored response surfaces as 404."""
        response = client.get("/v1/responses/resp-zzz")
        assert response.status_code == 404
        assert "resp-zzz" in response.json()["error"]["message"]

    def test_delete_stored_response(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """Deletion returns a confirmation object."""
        store.documents["resp-sess-1"] = {"input": "x", "response": {}}
        response = client.delete("/v1/responses/resp-sess-1")
        assert response.status_code == 200, response.text
        assert response.json() == {
            "id": "resp-sess-1",
            "object": "response.deleted",
            "deleted": True,
        }
        assert store.deleted == ["resp-sess-1"]

    def test_retrieve_chat_completion_kind_session_is_not_found(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """Retrieving a chat-completion-kind session as a response 404s."""
        store.documents["resp-sess-1"] = {
            "messages": [],
            "response": {"object": "chat.completion"},
        }
        response = client.get("/v1/responses/resp-sess-1")
        assert response.status_code == 404

    def test_delete_chat_completion_kind_session_is_not_found_and_not_deleted(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """Deleting a chat-completion-kind session as a response 404s without deleting."""
        store.documents["resp-sess-1"] = {
            "messages": [],
            "response": {"object": "chat.completion"},
        }
        response = client.delete("/v1/responses/resp-sess-1")
        assert response.status_code == 404
        assert not store.deleted

    def test_input_items_listing(self, client: TestClient, store: _StubStore) -> None:
        """Input items are normalized, ordered, and paginated."""
        store.documents["resp-sess-1"] = {
            "input": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "prior answer"},
            ],
            "response": {},
        }
        response = client.get("/v1/responses/resp-sess-1/input_items?order=asc")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "list"
        assert [item["id"] for item in body["data"]] == ["msg-0", "msg-1"]
        assert body["data"][0]["content"][0] == {
            "type": "input_text",
            "text": "question",
        }
        assert body["data"][1]["content"][0]["type"] == "output_text"
        assert body["first_id"] == "msg-0"
        assert body["last_id"] == "msg-1"
        assert body["has_more"] is False

    def test_input_items_cursor_and_default_order(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """Default order is descending, order=asc restores it, and after pages it."""
        store.documents["resp-sess-1"] = {
            "input": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "prior answer"},
            ],
            "response": {},
        }
        response = client.get("/v1/responses/resp-sess-1/input_items")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["msg-1", "msg-0"]

        response = client.get("/v1/responses/resp-sess-1/input_items?order=asc")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["msg-0", "msg-1"]

        response = client.get(
            "/v1/responses/resp-sess-1/input_items?order=asc&after=msg-0"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["msg-1"]
        assert body["has_more"] is False

    def test_input_items_after_accepts_non_msg_prefixed_ids(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """The `after` cursor accepts non-`msg-N` IDs the listing itself emits."""
        store.documents["resp-sess-1"] = {
            "input": [
                {"role": "user", "content": "question", "id": "resp-sess-1-msg-0"},
                {"role": "assistant", "content": "prior answer"},
            ],
            "response": {},
        }
        response = client.get(
            "/v1/responses/resp-sess-1/input_items?order=asc&after=resp-sess-1-msg-0"
        )
        assert response.status_code == 200, response.text
        assert [item["id"] for item in response.json()["data"]] == ["msg-1"]

    def test_input_items_unknown_after_cursor_is_not_found(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """An `after` cursor matching no item 404s instead of returning an empty page."""
        store.documents["resp-sess-1"] = {
            "input": [{"role": "user", "content": "question"}],
            "response": {},
        }
        response = client.get(
            "/v1/responses/resp-sess-1/input_items?after=msg-does-not-exist"
        )
        assert response.status_code == 404
        assert "msg-does-not-exist" in response.json()["error"]["message"]

    def test_input_items_listing_coerces_missing_required_fields(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """Canonical stored items missing an optional-in-practice field still list."""
        store.documents["resp-sess-1"] = {
            "input": [
                {
                    "type": "function_call_output",
                    "id": "fc-out-1",
                    "call_id": "call-1",
                    "output": "42",
                },
                {"type": "reasoning", "id": "rs-1"},
            ],
            "response": {},
        }
        response = client.get("/v1/responses/resp-sess-1/input_items?order=asc")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["type"] for item in body["data"]] == [
            "function_call_output",
            "reasoning",
        ]
        assert body["data"][0]["status"] == "completed"
        assert body["data"][1]["summary"] == []

    def test_input_items_listing_drops_unlistable_item_types(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """A stored `item_reference` entry is dropped instead of 500ing the listing."""
        store.documents["resp-sess-1"] = {
            "input": [
                {"role": "user", "content": "question"},
                {"type": "item_reference", "id": "ref_1"},
            ],
            "response": {},
        }
        response = client.get("/v1/responses/resp-sess-1/input_items?order=asc")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["type"] for item in body["data"]] == ["message"]
        assert body["has_more"] is False

    def test_input_items_listing_tolerates_legacy_none_fields(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """Stored items with leftover null `type`/`phase` fields still list correctly."""
        store.documents["resp-sess-1"] = {
            "input": [
                {"role": "user", "content": "question", "type": None, "phase": None}
            ],
            "response": {},
        }
        response = client.get("/v1/responses/resp-sess-1/input_items")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["data"][0]["type"] == "message"
        assert "phase" not in body["data"][0]


@pytest.mark.local
class TestCancelResponse:
    """POST /v1/responses/{id}/cancel."""

    def test_stored_response_cannot_be_cancelled(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """Cancelling any stored (synchronous) response fails with 400."""
        store.documents["resp-sess-1"] = {"input": "x", "response": {}}
        response = client.post("/v1/responses/resp-sess-1/cancel")
        assert response.status_code == 400
        assert (
            response.json()["error"]["message"]
            == "Cannot cancel a synchronous response."
        )

    def test_unknown_id_is_not_found(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """Cancelling an unknown stored response surfaces as 404."""
        response = client.post("/v1/responses/resp-zzz/cancel")
        assert response.status_code == 404

    def test_invalid_id_pattern_is_rejected(
        self, client: TestClient, store: _StubStore
    ) -> None:
        """An ID not matching the stored response pattern is rejected."""
        response = client.post("/v1/responses/not-a-response-id/cancel")
        assert response.status_code == 400


@pytest.mark.local
class TestUndecodableMantleResponseId:
    """An undecodable Mantle-form ID (``resp_...``) 404s before touching the local store."""

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/v1/responses/resp_notdecodable0"),
            ("delete", "/v1/responses/resp_notdecodable0"),
            ("post", "/v1/responses/resp_notdecodable0/cancel"),
            ("get", "/v1/responses/resp_notdecodable0/input_items"),
        ],
    )
    def test_returns_404_without_touching_store(
        self,
        method: str,
        path: str,
        client: TestClient,
        store: _StubStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A `resp_` ID that fails Mantle decoding 404s before any store lookup."""
        calls = 0

        async def _counting_load(_response_id: str, _kind: str) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return store.documents[_response_id]

        async def _counting_delete(_response_id: str, _kind: str) -> None:
            nonlocal calls
            calls += 1

        monkeypatch.setattr(openai_responses, "load_stored_response", _counting_load)
        monkeypatch.setattr(
            openai_responses, "delete_stored_response", _counting_delete
        )

        response = getattr(client, method)(path)
        assert response.status_code == 404
        assert calls == 0

    def test_create_with_undecodable_previous_response_id_returns_404(
        self, client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """A `resp_` `previous_response_id` that fails Mantle decoding 404s before generation."""
        response = client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "hello",
                "previous_response_id": "resp_notdecodable0",
            },
        )
        assert response.status_code == 404
        assert not backend.requests
        assert not store.saved


class TestStoredResponsesLive:
    """Live stored-responses lifecycle (AWS Bedrock sessions or official API)."""

    def test_store_lifecycle_and_continuation(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """store=true persists; retrieve, input_items, cancel, continuation, delete work."""
        from openai import BadRequestError  # noqa: PLC0415

        created = openai_client.responses.create(
            model=responses_model,
            input="The secret word is 'xylophone'. Acknowledge briefly.",
            store=True,
        )
        follow = None
        try:
            assert created.id.startswith("resp")
            retrieved = openai_client.responses.retrieve(created.id)
            assert retrieved.id == created.id
            assert retrieved.output_text == created.output_text

            with pytest.raises(BadRequestError, match="synchronous response"):
                openai_client.responses.cancel(created.id)

            items = list(
                openai_client.responses.input_items.list(created.id, order="asc")
            )
            assert items
            assert isinstance(items[0], ResponseInputMessageItem)
            assert isinstance(items[0].content[0], ResponseInputText)
            assert "xylophone" in items[0].content[0].text

            follow = openai_client.responses.create(
                model=responses_model,
                input="What is the secret word? Reply with that word only.",
                previous_response_id=created.id,
                store=True,
            )
            assert "xylophone" in follow.output_text.lower()
            assert follow.previous_response_id == created.id
        finally:
            with contextlib.suppress(Exception):
                openai_client.responses.delete(created.id)
            if follow is not None:
                with contextlib.suppress(Exception):
                    openai_client.responses.delete(follow.id)
        with pytest.raises(NotFoundError):
            openai_client.responses.retrieve(created.id)
