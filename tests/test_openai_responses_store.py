"""Stored-response routes and ``previous_response_id`` chaining on /v1/responses.

Ref: https://developers.openai.com/api/reference/resources/responses
     https://developers.openai.com/api/docs/guides/conversation-state#passing-context-from-the-previous-response
     stdapi/routes/openai_responses.py
"""

import contextlib
from json import loads
from typing import TYPE_CHECKING, Any

import pytest
from openai import NotFoundError
from openai.types.responses import ResponseInputMessageItem, ResponseInputText
from sse_starlette import EventSourceResponse

from stdapi.api_errors import ApiError
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
from tests._helpers import make_model_details

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from openai import OpenAI
    from starlette.testclient import TestClient

    from stdapi.models import ModelDetails


def _sse_events(text: str) -> list[dict[str, Any]]:
    """Parse a raw SSE response body into its decoded ``data`` payloads.

    Args:
        text: Raw ``text/event-stream`` response body.

    Returns:
        The JSON-decoded payload of each event, in stream order.
    """
    events = []
    for block in text.strip().split("\n\n"):
        data = "".join(
            line.removeprefix("data:").strip()
            for line in block.splitlines()
            if line.startswith("data:")
        )
        if data:
            events.append(loads(data))
    return events


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


def _stored_document(
    response_id: str, input_: object, **extra: object
) -> dict[str, Any]:
    """Build the stored document shape persisted by a ``store=true`` create.

    Args:
        response_id: Public response ID the canned response is built around.
        input_: Value of the document's ``input`` key.
        **extra: Additional top-level document keys (e.g. ``instructions``).

    Returns:
        The document as ``save_stored_response`` writes it.
    """
    return {
        "input": input_,
        "response": _canned_response(response_id, "m").model_dump(
            mode="json", by_alias=True, exclude_none=True
        ),
        **extra,
    }


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
    ) -> Response | EventSourceResponse:
        """Record the request and return a canned response.

        When ``request.stream`` is set, mirrors the real backend's behavior
        of echoing ``request.previous_response_id`` on the SSE response
        object instead of returning it directly.
        """
        self.requests.append((request, response_id))
        response = _canned_response(response_id, request.model)
        response.previous_response_id = request.previous_response_id
        if not request.stream:
            return response

        async def _events() -> AsyncIterator[dict[str, str]]:
            yield {
                "event": "response.created",
                "data": response.model_dump_json(by_alias=True, exclude_none=True),
            }

        return EventSourceResponse(_events())


class _StubNativeChatBackend(_StubChatBackend):
    """Stub chat backend advertising Bedrock Mantle native response storage."""

    def native_store_supported(self) -> bool:
        """Native-store stub: storage and chaining are handled upstream."""
        return True


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
        declared = None
        if document is not None and isinstance(document.get("response"), dict):
            declared = _KIND_BY_OBJECT.get(document["response"].get("object"))
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
def backend(monkeypatch: pytest.MonkeyPatch) -> _StubChatBackend:
    """Stub model validation and the generation backend."""

    async def _validate_model(
        model_id: str, *_args: object, **_kwargs: object
    ) -> ModelDetails:
        return make_model_details(model_id)

    stub = _StubChatBackend()
    monkeypatch.setattr(openai_responses, "validate_model", _validate_model)
    monkeypatch.setattr(openai_responses, "get_chat_model", lambda _model_id: stub)
    return stub


@pytest.fixture
def native_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend: _StubChatBackend,  # noqa: ARG001
) -> _StubChatBackend:
    """Replace the stub backend with one reporting native (Mantle) storage."""
    stub = _StubNativeChatBackend()
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
    """``store`` handling on POST /v1/responses.

    Unlike upstream OpenAI and Bedrock Mantle, which default ``store`` to
    true, this implementation defaults it to false and silently ignores it
    when streaming or when Bedrock session storage is unavailable.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         https://docs.aws.amazon.com/bedrock/latest/userguide/sessions.html
         stdapi/routes/openai_responses.py:create_response
    """

    def test_store_persists_response(
        self, app_client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """A stored response is generated under its session ID and persisted with its input.

        The public ID is ``resp-<session ID>``, so the backend must generate
        under that ID for the stored document and the returned object to agree.
        ``instructions`` is deliberately not persisted: it is not carried over
        to a ``previous_response_id`` continuation.

        Ref: https://developers.openai.com/api/docs/guides/migrate-to-responses
             stdapi/responses_store.py:save_stored_response
        """
        response = app_client.post(
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

    @pytest.mark.usefixtures("backend")
    def test_store_with_stream_is_ignored(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """``store`` is downgraded on a streaming request: the stream is served, unstored.

        Storage is not supported alongside streaming on this backend, so the
        request succeeds as a normal SSE stream with a request-scoped ID and no
        session is created — the response is simply not retrievable later.

        Ref: stdapi/routes/openai_responses.py:create_response
        """
        response = app_client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "hello",
                "store": True,
                "stream": True,
            },
        )
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert not store.created_kinds
        assert not store.saved
        (created_event,) = _sse_events(response.text)
        assert created_event["id"].startswith("resp-")
        assert created_event["id"] != "resp-sess-1", (
            "streamed response used a session ID"
        )

    @pytest.mark.usefixtures("backend")
    def test_store_without_session_storage_is_ignored(
        self, app_client: TestClient, store: _StubStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``store`` is ignored when session storage is unavailable on the server.

        ``try_create_stored_response_session`` returns None when the server
        lacks the Bedrock session permissions; the response must then fall back
        to a request-scoped ID instead of the literal ``resp-None``.

        Ref: stdapi/responses_store.py:try_create_stored_response_session
        """

        async def _unavailable(_kind: str) -> None:
            return None

        monkeypatch.setattr(
            openai_responses, "try_create_stored_response_session", _unavailable
        )
        response = app_client.post(
            "/v1/responses",
            json={"model": "amazon.nova-micro-v1:0", "input": "hello", "store": True},
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"].startswith("resp-")
        assert "None" not in response.json()["id"]
        assert not store.saved
        assert not store.discarded

    @pytest.mark.usefixtures("backend")
    def test_without_store_nothing_is_persisted(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Omitting ``store`` persists nothing: ``store`` defaults to false here.

        Upstream OpenAI defaults ``store`` to true; this implementation
        defaults it to false, so a plain create is not retrievable afterwards.

        Ref: https://developers.openai.com/api/docs/guides/conversation-state#passing-context-from-the-previous-response
             stdapi/types/openai_responses.py:ResponseCreateParams
        """
        response = app_client.post(
            "/v1/responses", json={"model": "amazon.nova-micro-v1:0", "input": "hello"}
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"].startswith("resp-")
        assert response.json()["id"] != "resp-sess-1"
        assert not store.created_kinds
        assert not store.saved

    @pytest.mark.usefixtures("backend")
    def test_store_and_list_input_items_with_message_object_input(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """A message-object input round-trips through store and input_items listing.

        A string message ``content`` is stored verbatim and normalized into an
        ``input_text`` content part only when listed.

        Ref: https://developers.openai.com/api/reference/resources/responses/subresources/input_items/methods/list
             stdapi/routes/openai_responses.py:_normalized_input_items
        """
        response = app_client.post(
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

        assert document["input"] == [{"role": "user", "content": "question"}]

        items_response = app_client.get(f"/v1/responses/{response_id}/input_items")
        assert items_response.status_code == 200, items_response.text
        items = items_response.json()["data"]
        assert len(items) == 1
        assert items[0]["role"] == "user"
        assert items[0]["content"][0] == {"type": "input_text", "text": "question"}

    def test_generation_failure_discards_pending_session(
        self, app_client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """A failed generation discards the pending session and stores nothing.

        The backing session is created before generation, so a generation
        failure must not leave an empty session behind.

        Ref: stdapi/responses_store.py:discard_stored_response_session
        """

        async def _raise(
            _request: ResponseCreateParams,
            _response_id: str,
            _created_at: float,
            moderation_builder: Any = None,  # noqa: ANN401, ARG001
        ) -> Response:
            msg = "backend failure"
            raise ApiError(msg, status=502)

        backend.create_response = _raise  # type: ignore[method-assign, assignment]
        response = app_client.post(
            "/v1/responses",
            json={"model": "amazon.nova-micro-v1:0", "input": "hello", "store": True},
        )
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "server_error"
        assert store.created_kinds == ["response"]
        assert store.discarded == ["resp-sess-1"]
        assert not store.saved

    @pytest.mark.usefixtures("backend")
    def test_save_failure_discards_pending_session(
        self, app_client: TestClient, store: _StubStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A save failure after a successful generation discards the pending session.

        Storage failures are not hidden behind a 200: the client must not
        receive an ID it can never retrieve.

        Ref: stdapi/routes/openai_responses.py:create_response
        """

        async def _raise(_response_id: str, _document: dict[str, Any]) -> None:
            msg = "save failure"
            raise ApiError(msg, status=502)

        monkeypatch.setattr(openai_responses, "save_stored_response", _raise)
        response = app_client.post(
            "/v1/responses",
            json={"model": "amazon.nova-micro-v1:0", "input": "hello", "store": True},
        )
        assert response.status_code == 502
        assert store.discarded == ["resp-sess-1"]


@pytest.mark.local
class TestPreviousResponseId:
    """``previous_response_id`` chaining on POST /v1/responses.

    Local-store chaining is done gateway-side: the stored conversation is
    merged into the new request's input, since Bedrock Converse is stateless.

    Ref: https://developers.openai.com/api/docs/guides/conversation-state#passing-context-from-the-previous-response
         stdapi/routes/openai_responses.py:_apply_previous_response
    """

    def test_previous_conversation_is_prepended(
        self, app_client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """The stored input and output precede the new input, and instructions are dropped.

        Upstream specifies that instructions from a previous response are not
        carried over, so the stored ``instructions`` must not reappear on the
        continuation; ``previous_response_id`` is restored on the merged
        request so the response object still echoes it.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/routes/openai_responses.py:_merge_previous_response
        """
        store.documents["resp-sess-1"] = _stored_document(
            "resp-sess-1",
            [{"role": "user", "content": "first"}],
            instructions="old sys",
        )
        response = app_client.post(
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
        # Restored after the merge so streaming SSE events echo it too.
        assert request.previous_response_id == "resp-sess-1"
        assert request.instructions is None
        assert isinstance(request.input, list)
        assert isinstance(request.input[0], EasyInputMessage)
        assert request.input[0].content == "first"
        assert request.input[1].content[0].text == "answer"  # type: ignore[union-attr]
        assert isinstance(request.input[2], EasyInputMessage)
        assert request.input[2].content == "second"

    def test_unknown_previous_response_is_not_found(
        self, app_client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """An unknown ``previous_response_id`` 404s with the upstream wording, before generation.

        Ref: stdapi/routes/openai_responses.py:_previous_response_not_found
        """
        response = app_client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "x",
                "previous_response_id": "resp-zzz",
            },
        )
        assert response.status_code == 404
        assert not backend.requests
        err = response.json()["error"]
        assert err["message"] == "Previous response with id 'resp-zzz' not found."
        assert err["param"] == "previous_response_id"
        assert err["type"] == "invalid_request_error"

    def test_arn_injection_previous_response_id_is_not_found(
        self,
        app_client: TestClient,
        backend: _StubChatBackend,
        store: _StubStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A session ARN smuggled as ``previous_response_id`` never reaches the store.

        The route's ``resp[-_]`` path pattern does not constrain a body field,
        so the ID is re-matched against the stored-response pattern; otherwise
        the ARN would be passed to AWS as a session identifier.

        Ref: stdapi/responses_store.py:RESPONSE_ID_PATTERN
             stdapi/routes/openai_responses.py:_apply_previous_response
        """
        calls = 0

        async def _counting_load(_response_id: str, _kind: str) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return {}

        monkeypatch.setattr(openai_responses, "load_stored_response", _counting_load)
        arn_id = "resp-arn:aws:bedrock:eu-west-1:123456789012:session/x"
        response = app_client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "x",
                "previous_response_id": arn_id,
            },
        )
        assert response.status_code == 404
        assert calls == 0
        assert not backend.requests
        err = response.json()["error"]
        assert err["message"].startswith("Previous response with id")
        # The ARN is redacted from the client-facing message (defense in depth);
        # what matters is the store was never asked to resolve it (calls == 0).
        assert "123456789012" not in err["message"]
        assert err["param"] == "previous_response_id"

    def test_store_with_previous_response_id_saves_merged_history(
        self, app_client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """A stored continuation persists the merged history, in conversation order.

        The stored document of the new response holds the previous input, the
        previous output and the new input, so the next continuation only needs
        the newest response ID.

        Ref: stdapi/routes/openai_responses.py:_merge_previous_response
        """
        store.documents["resp-old"] = _stored_document(
            "resp-old", [{"role": "user", "content": "first"}]
        )
        response = app_client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "second",
                "previous_response_id": "resp-old",
                "store": True,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["previous_response_id"] == "resp-old"
        ((_, document),) = store.saved
        contents = [item.get("content") for item in document["input"]]
        assert contents[0] == "first"
        assert contents[-1] == "second"
        # The previous response's assistant output sits between the two turns.
        assert contents[1][0]["text"] == "answer"

    def test_streaming_continuation_echoes_previous_response_id(
        self, app_client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """A streamed continuation echoes ``previous_response_id`` on its first event.

        Chaining strips ``previous_response_id`` while merging the stored
        conversation into the input; it is restored on the request afterwards
        so the SSE events built from it still report the chained ID.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/routes/openai_responses.py:_apply_previous_response
        """
        store.documents["resp-old"] = _stored_document(
            "resp-old", [{"role": "user", "content": "first"}]
        )
        response = app_client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "second",
                "previous_response_id": "resp-old",
                "stream": True,
            },
        )
        assert response.status_code == 200, response.text
        (created_event,) = _sse_events(response.text)
        assert created_event["previous_response_id"] == "resp-old"
        ((request, _),) = backend.requests
        assert isinstance(request.input, list)
        assert isinstance(request.input[0], EasyInputMessage)
        assert request.input[0].content == "first"


@pytest.mark.local
class TestStoredResponseRoutes:
    """GET/DELETE /v1/responses/{id} and GET /v1/responses/{id}/input_items.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/retrieve
         https://developers.openai.com/api/reference/resources/responses/methods/delete
         https://developers.openai.com/api/reference/resources/responses/subresources/input_items/methods/list
         stdapi/routes/openai_responses.py
    """

    def test_retrieve_stored_response(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Retrieval replays the stored response document unchanged.

        The stored document holds the whole serialized Response, so output
        items and usage counters come back exactly as persisted rather than
        being rebuilt.

        Ref: stdapi/routes/openai_responses.py:retrieve_response
        """
        store.documents["resp-sess-1"] = _stored_document("resp-sess-1", "hello")
        response = app_client.get("/v1/responses/resp-sess-1")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"] == "resp-sess-1"
        assert body["object"] == "response"
        assert body["output"][0]["content"][0]["text"] == "answer"
        assert body["usage"]["total_tokens"] == 2

    def test_retrieve_unknown_is_not_found(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """An unknown stored response surfaces as a 404 naming the requested ID.

        Ref: stdapi/responses_store.py:load_stored_response
        """
        response = app_client.get("/v1/responses/resp-zzz")
        assert response.status_code == 404
        err = response.json()["error"]
        assert err["message"] == "Response with id 'resp-zzz' not found."
        assert err["type"] == "invalid_request_error"

    def test_delete_stored_response(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Deletion returns a confirmation object and discards the backing session.

        The envelope's ``object`` is ``response.deleted`` on this
        implementation, where the upstream reference documents ``response``.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/delete
             stdapi/types/openai_responses.py:ResponseDeleted
        """
        store.documents["resp-sess-1"] = {"input": "x", "response": {}}
        response = app_client.delete("/v1/responses/resp-sess-1")
        assert response.status_code == 200, response.text
        assert response.json() == {
            "id": "resp-sess-1",
            "object": "response.deleted",
            "deleted": True,
        }
        assert store.deleted == ["resp-sess-1"]

    def test_retrieve_chat_completion_kind_session_is_not_found(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Retrieving a chat-completion document through the Responses route 404s.

        Both routes share the same Bedrock session store, so the stored kind is
        what keeps a chat completion from being read as a response.

        Ref: stdapi/responses_store.py:_kind_mismatches
        """
        store.documents["resp-sess-1"] = {
            "messages": [],
            "response": {"object": "chat.completion"},
        }
        response = app_client.get("/v1/responses/resp-sess-1")
        assert response.status_code == 404
        assert "resp-sess-1" in response.json()["error"]["message"]

    def test_delete_chat_completion_kind_session_is_not_found_and_not_deleted(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Deleting a chat-completion document through the Responses route 404s, undeleted.

        Ref: stdapi/responses_store.py:delete_stored_response
        """
        store.documents["resp-sess-1"] = {
            "messages": [],
            "response": {"object": "chat.completion"},
        }
        response = app_client.delete("/v1/responses/resp-sess-1")
        assert response.status_code == 404
        assert "resp-sess-1" in response.json()["error"]["message"]
        assert not store.deleted

    def test_input_items_listing(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Listed items carry generated IDs and role-appropriate content parts.

        A stored string ``content`` becomes an ``input_text`` part for user
        messages and an ``output_text`` part for assistant ones; every item gets
        a positional ``msg-N`` ID so it can be used as an ``after`` cursor.

        Ref: stdapi/routes/openai_responses.py:_normalized_input_items
        """
        store.documents["resp-sess-1"] = {
            "input": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "prior answer"},
            ],
            "response": {},
        }
        response = app_client.get("/v1/responses/resp-sess-1/input_items?order=asc")
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
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """``order`` defaults to desc, ``asc`` restores conversation order, ``after`` pages.

        The default order matches the upstream list contract (``desc``), and
        ``after`` returns only the items strictly following the cursor.

        Ref: https://developers.openai.com/api/reference/resources/responses/subresources/input_items/methods/list
             stdapi/routes/openai_responses.py:list_response_input_items
        """
        store.documents["resp-sess-1"] = {
            "input": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "prior answer"},
            ],
            "response": {},
        }
        response = app_client.get("/v1/responses/resp-sess-1/input_items")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["msg-1", "msg-0"]
        assert body["first_id"] == "msg-1"
        assert body["last_id"] == "msg-0"

        response = app_client.get("/v1/responses/resp-sess-1/input_items?order=asc")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["msg-0", "msg-1"]

        response = app_client.get(
            "/v1/responses/resp-sess-1/input_items?order=asc&after=msg-0"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["msg-1"]
        assert body["has_more"] is False

    def test_input_items_after_accepts_non_msg_prefixed_ids(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """The ``after`` cursor matches a stored item's own ID, not just generated ``msg-N``.

        A positional ID is only generated for items that lack one, so the
        cursor must be compared against the emitted IDs rather than parsed.

        Ref: stdapi/routes/openai_responses.py:list_response_input_items
        """
        store.documents["resp-sess-1"] = {
            "input": [
                {"role": "user", "content": "question", "id": "resp-sess-1-msg-0"},
                {"role": "assistant", "content": "prior answer"},
            ],
            "response": {},
        }
        response = app_client.get(
            "/v1/responses/resp-sess-1/input_items?order=asc&after=resp-sess-1-msg-0"
        )
        assert response.status_code == 200, response.text
        assert [item["id"] for item in response.json()["data"]] == ["msg-1"]

    def test_input_items_unknown_after_cursor_is_not_found(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """An ``after`` cursor matching no item 404s instead of returning an empty page.

        Ref: stdapi/routes/openai_responses.py:list_response_input_items
        """
        store.documents["resp-sess-1"] = {
            "input": [{"role": "user", "content": "question"}],
            "response": {},
        }
        response = app_client.get(
            "/v1/responses/resp-sess-1/input_items?after=msg-does-not-exist"
        )
        assert response.status_code == 404
        err = response.json()["error"]
        assert err["message"] == "No input item with id 'msg-does-not-exist'."
        assert err["type"] == "invalid_request_error"

    def test_input_items_listing_coerces_missing_required_fields(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """A stored item missing a required field is backfilled instead of dropped.

        Clients legitimately store canonical shapes such as a
        ``function_call_output`` without ``status`` or a ``reasoning`` item
        without ``summary``; those fields are required by the listable item
        union, so a known safe default is backfilled before validation.

        Ref: stdapi/routes/openai_responses.py:_listable_input_items
        """
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
        response = app_client.get("/v1/responses/resp-sess-1/input_items?order=asc")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["type"] for item in body["data"]] == [
            "function_call_output",
            "reasoning",
        ]
        assert body["data"][0]["status"] == "completed"
        assert body["data"][1]["summary"] == []

    def test_input_items_listing_drops_unlistable_item_types(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """A stored ``item_reference`` entry is dropped instead of failing the listing.

        ``item_reference`` is accepted on create but never becomes conversation
        history, so it is absent from the listing exactly as upstream, and the
        remaining items still list.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/routes/openai_responses.py:_listable_input_items
        """
        store.documents["resp-sess-1"] = {
            "input": [
                {"role": "user", "content": "question"},
                {"type": "item_reference", "id": "ref_1"},
            ],
            "response": {},
        }
        response = app_client.get("/v1/responses/resp-sess-1/input_items?order=asc")
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["type"] for item in body["data"]] == ["message"]
        assert body["has_more"] is False

    def test_input_items_listing_tolerates_legacy_none_fields(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Null ``type`` / ``phase`` fields from older documents are dropped, not replayed.

        Documents written before storage excluded null fields still hold them;
        a null ``type`` must not defeat the ``message`` default, and a null
        ``phase`` must not reappear in the listed item.

        Ref: stdapi/routes/openai_responses.py:_normalized_input_items
        """
        store.documents["resp-sess-1"] = {
            "input": [
                {"role": "user", "content": "question", "type": None, "phase": None}
            ],
            "response": {},
        }
        response = app_client.get("/v1/responses/resp-sess-1/input_items")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["data"][0]["type"] == "message"
        assert body["data"][0]["content"][0] == {
            "type": "input_text",
            "text": "question",
        }
        assert "phase" not in body["data"][0]

    @pytest.mark.parametrize(
        "document", [{"response": None}, {}], ids=["null-response", "no-response-key"]
    )
    def test_retrieve_malformed_document_is_not_found(
        self, app_client: TestClient, store: _StubStore, document: dict[str, Any]
    ) -> None:
        """A foreign or corrupt stored document 404s instead of failing with a 500.

        A session holding a document without a usable ``response`` object (a
        schema drift or a session written by another tool) is reported as not
        found rather than crashing route handling.

        Ref: stdapi/routes/openai_responses.py:_malformed_stored_document
        """
        store.documents["resp-sess-1"] = document
        response = app_client.get("/v1/responses/resp-sess-1")
        assert response.status_code == 404
        assert "resp-sess-1" in response.json()["error"]["message"]

    @pytest.mark.parametrize(
        "document", [{"response": None}, {}], ids=["null-response", "no-response-key"]
    )
    def test_input_items_malformed_document_is_not_found(
        self, app_client: TestClient, store: _StubStore, document: dict[str, Any]
    ) -> None:
        """A foreign or corrupt stored document 404s the input-items listing too.

        Ref: stdapi/routes/openai_responses.py:_malformed_stored_document
        """
        store.documents["resp-sess-1"] = document
        response = app_client.get("/v1/responses/resp-sess-1/input_items")
        assert response.status_code == 404
        assert "resp-sess-1" in response.json()["error"]["message"]

    def test_input_items_non_mapping_entry_is_not_found(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """A stored input list with a non-mapping entry 404s instead of failing with a 500.

        Ref: stdapi/routes/openai_responses.py:_normalized_input_items
        """
        store.documents["resp-sess-1"] = {"input": ["not-a-dict"], "response": {}}
        response = app_client.get("/v1/responses/resp-sess-1/input_items")
        assert response.status_code == 404
        assert "resp-sess-1" in response.json()["error"]["message"]

    def test_retrieve_rejects_stream_query_param(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """``stream=true`` on retrieval is rejected with a 400, not silently ignored.

        Upstream documents streaming a stored response back (with
        ``starting_after`` resumption); this implementation does not support it
        and says so instead of returning a non-streamed body.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/retrieve
             stdapi/routes/openai_responses.py:retrieve_response
        """
        store.documents["resp-sess-1"] = _stored_document("resp-sess-1", "hello")
        response = app_client.get("/v1/responses/resp-sess-1?stream=true")
        assert response.status_code == 400
        err = response.json()["error"]
        assert "stream" in err["message"]
        assert "not supported" in err["message"]
        assert err["type"] == "invalid_request_error"

    def test_retrieve_accepts_and_ignores_include_param(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """``include`` is accepted for compatibility and changes nothing in the body.

        The stored document is replayed as-is, so no ``include`` value can add
        data to it; the parameter is accepted only so SDK calls do not fail.

        Ref: stdapi/routes/openai_responses.py:retrieve_response
        """
        store.documents["resp-sess-1"] = _stored_document("resp-sess-1", "hello")
        response = app_client.get(
            "/v1/responses/resp-sess-1?include=reasoning.encrypted_content"
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"] == "resp-sess-1"
        plain = app_client.get("/v1/responses/resp-sess-1")
        assert plain.status_code == 200, plain.text
        assert response.json() == plain.json()

    def test_retrieve_accepts_and_ignores_starting_after(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """``starting_after`` is accepted for compatibility and changes nothing in the body.

        Upstream uses it to resume a streamed replay from an event sequence
        number; replay is unsupported here, so the whole stored document comes
        back regardless of the cursor.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/retrieve
             stdapi/routes/openai_responses.py:retrieve_response
        """
        store.documents["resp-sess-1"] = _stored_document("resp-sess-1", "hello")
        response = app_client.get("/v1/responses/resp-sess-1?starting_after=5")
        assert response.status_code == 200, response.text
        plain = app_client.get("/v1/responses/resp-sess-1")
        assert plain.status_code == 200, plain.text
        assert response.json() == plain.json()

    def test_input_items_accepts_and_ignores_include(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """``include`` on the input-items listing is accepted and changes nothing.

        The listing replays stored items, so no ``include`` value can add data;
        it is accepted only so the SDK's ``input_items.list(include=[...])``
        call does not fail validation.

        Ref: https://developers.openai.com/api/reference/resources/responses/subresources/input_items/methods/list
             stdapi/routes/openai_responses.py:list_response_input_items
        """
        store.documents["resp-sess-1"] = {
            "input": [{"role": "user", "content": "question"}],
            "response": {},
        }
        response = app_client.get(
            "/v1/responses/resp-sess-1/input_items"
            "?order=asc&include=reasoning.encrypted_content"
        )
        assert response.status_code == 200, response.text
        plain = app_client.get("/v1/responses/resp-sess-1/input_items?order=asc")
        assert plain.status_code == 200, plain.text
        assert response.json() == plain.json()

    def test_input_items_limit_truncates_and_reports_has_more(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """``limit`` truncates the page and sets ``has_more`` when items remain.

        ``has_more`` compares the item count against the limit *before* the
        page slice, so the last page of an exactly-consumed list reports false;
        the SDK's auto-pagination relies on that to stop.

        Ref: https://developers.openai.com/api/reference/resources/responses/subresources/input_items/methods/list
             stdapi/routes/openai_responses.py:list_response_input_items
        """
        store.documents["resp-sess-1"] = {
            "input": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "second"},
            ],
            "response": {},
        }
        response = app_client.get(
            "/v1/responses/resp-sess-1/input_items?order=asc&limit=2"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["msg-0", "msg-1"]
        assert body["first_id"] == "msg-0"
        assert body["last_id"] == "msg-1"
        assert body["has_more"] is True

        response = app_client.get(
            "/v1/responses/resp-sess-1/input_items?order=asc&limit=2&after=msg-1"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert [item["id"] for item in body["data"]] == ["msg-2"]
        assert body["has_more"] is False

    def test_input_items_overlong_after_cursor_is_rejected(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """An ``after`` cursor longer than 255 characters is rejected before any lookup.

        The cursor is matched linearly against every stored item, so its length
        is bounded at the route; the request fails validation (400) rather than
        the 404 an unknown-but-short cursor would produce.

        Ref: https://developers.openai.com/api/reference/resources/responses/subresources/input_items/methods/list
             stdapi/routes/openai_responses.py:list_response_input_items
        """
        response = app_client.get(
            f"/v1/responses/resp-sess-1/input_items?after={'m' * 256}"
        )
        assert response.status_code == 400, response.text
        err = response.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert "after" in err["message"]
        assert not store.documents, "the store was reached despite an invalid cursor"


@pytest.mark.local
class TestNativeStoreModel:
    """POST /v1/responses against a model with Bedrock Mantle native storage.

    Mantle keeps the response upstream, so the gateway must neither create a
    Bedrock session nor persist a document, and a local ``previous_response_id``
    is resolved by inlining the stored conversation instead of forwarding an ID
    the Mantle payload builder would reject.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
         stdapi/routes/openai_responses.py:create_response
    """

    def test_store_is_forced_off_and_no_local_session_is_created(
        self,
        app_client: TestClient,
        native_backend: _StubChatBackend,
        store: _StubStore,
    ) -> None:
        """``store=true`` creates no local session and persists no document.

        Storing gateway-side as well would pay for an orphan Bedrock session on
        every request and hand back a ``resp-<session>`` ID the Mantle store
        never saw.

        Ref: stdapi/routes/openai_responses.py:create_response
        """
        response = app_client.post(
            "/v1/responses",
            json={"model": "amazon.nova-micro-v1:0", "input": "hello", "store": True},
        )
        assert response.status_code == 200, response.text
        assert not store.created_kinds
        assert not store.saved
        assert not store.discarded
        assert response.json()["id"].startswith("resp-")
        assert response.json()["id"] != "resp-sess-1"

    def test_local_previous_response_is_merged_without_the_id(
        self,
        app_client: TestClient,
        native_backend: _StubChatBackend,
        store: _StubStore,
    ) -> None:
        """A local ``resp-`` chain is inlined and ``previous_response_id`` is dropped.

        The merged input already carries the stored conversation; forwarding a
        non-Mantle-tagged ID as well would be rejected by the Mantle payload
        builder. The non-native path restores the ID instead.

        Ref: https://developers.openai.com/api/docs/guides/conversation-state#passing-context-from-the-previous-response
             stdapi/routes/openai_responses.py:_apply_previous_response
        """
        store.documents["resp-sess-1"] = _stored_document(
            "resp-sess-1", [{"role": "user", "content": "first"}]
        )
        response = app_client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "second",
                "previous_response_id": "resp-sess-1",
            },
        )
        assert response.status_code == 200, response.text
        ((request, _),) = native_backend.requests
        assert request.previous_response_id is None
        assert isinstance(request.input, list)
        assert isinstance(request.input[0], EasyInputMessage)
        assert request.input[0].content == "first"
        assert request.input[1].content[0].text == "answer"  # type: ignore[union-attr]
        assert isinstance(request.input[2], EasyInputMessage)
        assert request.input[2].content == "second"


@pytest.mark.local
class TestInvalidResponseIdPattern:
    """Malformed response IDs are rejected on every ``{response_id}`` route.

    The path parameter is constrained to ``resp-`` (local store) and ``resp_``
    (region-tagged Mantle) IDs, so anything else never reaches the store.

    Ref: stdapi/routes/openai_responses.py:_RESPONSE_ID_PATTERN
    """

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/v1/responses/not-a-response-id"),
            ("delete", "/v1/responses/not-a-response-id"),
            ("get", "/v1/responses/not-a-response-id/input_items"),
        ],
        ids=["get", "delete", "input_items"],
    )
    def test_invalid_id_pattern_is_rejected(
        self, method: str, path: str, app_client: TestClient, store: _StubStore
    ) -> None:
        """An ID not matching the response ID pattern is a 400 validation error, not a 404.

        Ref: stdapi/main.py:handle_validation_exception
        """
        response = getattr(app_client, method)(path)
        assert response.status_code == 400
        err = response.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert "response_id" in err["message"]
        assert not store.deleted


@pytest.mark.local
class TestCancelResponse:
    """POST /v1/responses/{id}/cancel on locally stored responses.

    Upstream only allows cancelling ``background=true`` responses; locally
    stored responses are always generated synchronously, so cancel exists only
    to return the upstream synchronous-response error.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/cancel
         https://developers.openai.com/api/docs/guides/background
         stdapi/routes/openai_responses.py:cancel_response
    """

    def test_stored_response_cannot_be_cancelled(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Cancelling an existing stored response fails with the synchronous-response 400.

        The response must be resolved first: an existing ID gives 400, an
        unknown one still gives 404.

        Ref: stdapi/routes/openai_responses.py:cancel_response
        """
        store.documents["resp-sess-1"] = {"input": "x", "response": {}}
        response = app_client.post("/v1/responses/resp-sess-1/cancel")
        assert response.status_code == 400
        err = response.json()["error"]
        assert err["message"] == "Cannot cancel a synchronous response."
        assert err["type"] == "invalid_request_error"

    def test_unknown_id_is_not_found(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """Cancelling an unknown stored response is a 404, not the synchronous-response 400.

        Ref: stdapi/routes/openai_responses.py:cancel_response
        """
        response = app_client.post("/v1/responses/resp-zzz/cancel")
        assert response.status_code == 404
        assert response.json()["error"]["message"] == (
            "Response with id 'resp-zzz' not found."
        )

    def test_invalid_id_pattern_is_rejected(
        self, app_client: TestClient, store: _StubStore
    ) -> None:
        """An ID not matching the response ID pattern is rejected before any lookup.

        Ref: stdapi/routes/openai_responses.py:_RESPONSE_ID_PATTERN
        """
        response = app_client.post("/v1/responses/not-a-response-id/cancel")
        assert response.status_code == 400
        err = response.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert "response_id" in err["message"]
        assert "synchronous" not in err["message"]


@pytest.mark.local
class TestUndecodableMantleResponseId:
    """An undecodable Mantle-form ID (``resp_...``) 404s before touching the local store.

    ID prefixes are load-bearing: ``resp-`` is a local store ID, ``resp_`` is a
    region-tagged Mantle ID. A ``resp_`` ID that fails region decoding cannot
    exist locally and would be mangled into an invalid Bedrock session ID.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
         stdapi/routes/openai_responses.py:_require_local_response_id
    """

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
        app_client: TestClient,
        store: _StubStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A ``resp_`` ID that fails Mantle decoding 404s before any store lookup.

        Ref: stdapi/aws_bedrock_mantle.py:decode_mantle_response_id
        """
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

        response = getattr(app_client, method)(path)
        assert response.status_code == 404
        assert calls == 0, "the local store was queried with a Mantle-form ID"
        assert "resp_notdecodable0" in response.json()["error"]["message"]

    def test_create_with_undecodable_previous_response_id_returns_404(
        self, app_client: TestClient, backend: _StubChatBackend, store: _StubStore
    ) -> None:
        """A ``resp_`` ``previous_response_id`` failing Mantle decoding 404s before generation.

        Ref: stdapi/routes/openai_responses.py:_apply_previous_response
        """
        response = app_client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": "hello",
                "previous_response_id": "resp_notdecodable0",
            },
        )
        assert response.status_code == 404
        assert "resp_notdecodable0" in response.json()["error"]["message"]
        assert not backend.requests, "generation ran despite an unresolvable chain"
        assert not store.saved


@pytest.mark.local
class TestStoredResponseRoutesAuthRejection:
    """A missing bearer token is rejected with a 401 OpenAI envelope, no store access.

    Uses the session-wide ``test_client`` (lifespan-started, unlike the
    lifespan-free ``app_client`` fixture) so the auth handler is actually
    initialized and able to reject a missing token.

    Ref: https://developers.openai.com/api/docs/guides/error-codes
         stdapi/auth.py:authenticate
    """

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/v1/responses/resp-sess-1"),
            ("delete", "/v1/responses/resp-sess-1"),
            ("get", "/v1/responses/resp-sess-1/input_items"),
            ("post", "/v1/responses/resp-sess-1/cancel"),
        ],
        ids=["retrieve", "delete", "input_items", "cancel"],
    )
    def test_missing_bearer_token_is_rejected(
        self,
        method: str,
        path: str,
        test_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No Authorization header yields the 401 ``authentication_error`` envelope.

        Authentication is a route dependency, so it must reject before the
        handler resolves the ID: an unauthenticated caller learns nothing about
        which response IDs exist.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/auth.py:authenticate
        """
        calls: list[str] = []

        async def _counting_load(response_id: str, _kind: str) -> dict[str, Any]:
            calls.append(response_id)
            return {"input": "x", "response": {}}

        monkeypatch.setattr(openai_responses, "load_stored_response", _counting_load)
        response = getattr(test_client, method)(path)

        assert response.status_code == 401
        body = response.json()
        assert set(body.keys()) == {"error"}
        err = body["error"]
        assert set(err.keys()) == {"message", "type", "param", "code"}
        assert err["type"] == "authentication_error"
        assert err["message"] == "Unauthorized", "the 401 message leaks internal detail"
        assert not calls, "the store was queried by an unauthenticated caller"


class TestStoredResponsesLive:
    """Live stored-response lifecycle against AWS Bedrock sessions or the official API.

    Ref: https://developers.openai.com/api/reference/resources/responses
         https://docs.aws.amazon.com/bedrock/latest/userguide/sessions.html
         stdapi/routes/openai_responses.py
    """

    def test_store_lifecycle_and_continuation(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A stored response is retrievable, listable, chainable and finally deleted.

        Walks the whole documented lifecycle in one billed conversation:
        ``store=true`` create, retrieve, the synchronous-response 400 from
        cancel, input-items listing, a ``previous_response_id`` continuation
        that must recall the secret word, then delete and the resulting 404.

        Ref: https://developers.openai.com/api/docs/guides/conversation-state#passing-context-from-the-previous-response
             https://developers.openai.com/api/reference/resources/responses/methods/delete
        """
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
            assert retrieved.object == "response"
            assert retrieved.output_text == created.output_text
            assert retrieved.usage is not None
            assert (
                retrieved.usage.total_tokens
                == retrieved.usage.input_tokens + retrieved.usage.output_tokens
            )

            with pytest.raises(BadRequestError, match="synchronous response") as cancel:
                openai_client.responses.cancel(created.id)
            assert cancel.value.status_code == 400

            items = list(
                openai_client.responses.input_items.list(created.id, order="asc")
            )
            assert items
            assert isinstance(items[0], ResponseInputMessageItem)
            assert items[0].role == "user"
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
        with pytest.raises(NotFoundError) as deleted:
            openai_client.responses.retrieve(created.id)
        assert deleted.value.status_code == 404
