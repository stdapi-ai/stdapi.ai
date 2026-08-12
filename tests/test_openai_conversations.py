"""OpenAI-compatible Conversations API routes and their Responses integration.

The eight conversation routes are exercised end to end against an in-memory
stand-in for the conversation store, so the whole surface — object shapes,
metadata semantics, identifier validation, pagination and item deletion — runs
without credentials. The Responses integration covers the two behaviours the
route owns: the conversation's items become the input prefix, and the turn is
appended unless ``store`` is false.

Those tests prove the route bodies, not the storage behind them, so
``TestConversationsLive`` walks the same surface against whichever target is
selected — real session storage, a deployed gateway, or the official API this
mirrors — with no stub anywhere in the path.

Ref: https://developers.openai.com/api/reference/resources/conversations.md
     https://developers.openai.com/api/docs/guides/conversation-state.md
     stdapi/routes/openai_conversations.py
     stdapi/routes/openai_responses.py:_apply_conversation
     stdapi/conversations.py
"""

import contextlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
from botocore.exceptions import ClientError
from openai import NotFoundError
from sse_starlette import EventSourceResponse

from stdapi import conversations
from stdapi.responses_store import KIND_TAG
from stdapi.routes import openai_responses
from stdapi.types.openai_responses import (
    InputTokensDetails,
    OutputTokensDetails,
    Response,
    ResponseCompletedEvent,
    ResponseCreateParams,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from stdapi.utils import json_sse
from tests._helpers import make_model_details

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from openai import OpenAI
    from openai.types.conversations import ConversationItem
    from starlette.testclient import TestClient

    from stdapi.models import ModelDetails


class _FakeSessionClient:
    """In-memory stand-in for the Bedrock session API the store calls."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.steps: dict[str, list[dict[str, Any]]] = {}
        self.calls: list[str] = []
        self._counter = 0

    def _session(self, identifier: str) -> dict[str, Any]:
        """Return a session, raising the AWS not-found error when absent."""
        session = self.sessions.get(identifier)
        if session is None:
            raise ClientError(
                {"Error": {"Code": "ResourceNotFoundException", "Message": "missing"}},
                "GetSession",
            )
        return session

    async def create_session(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.calls.append("create_session")
        self._counter += 1
        session_id = f"sess{self._counter}"
        session = {
            "sessionId": session_id,
            "sessionArn": f"arn:aws:bedrock:::session/{session_id}",
            "createdAt": datetime.now(tz=UTC),
            "sessionMetadata": dict(params.get("sessionMetadata") or {}),
            "tags": dict(params.get("tags") or {}),
        }
        self.sessions[session_id] = session
        self.steps[session_id] = []
        return session

    async def get_session(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.calls.append("get_session")
        return self._session(params["sessionIdentifier"])

    async def list_tags_for_resource(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.calls.append("list_tags_for_resource")
        for session in self.sessions.values():
            if session["sessionArn"] == params["resourceArn"]:
                return {"tags": session["tags"]}
        return {"tags": {}}

    async def update_session(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.calls.append("update_session")
        session = self._session(params["sessionIdentifier"])
        session["sessionMetadata"] = dict(params["sessionMetadata"])
        return session

    async def create_invocation(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.calls.append("create_invocation")
        self._session(params["sessionIdentifier"])
        return {"invocationId": str(uuid4())}

    async def put_invocation_step(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.calls.append("put_invocation_step")
        session_id = params["sessionIdentifier"]
        self._session(session_id)
        step = {
            "invocationId": params["invocationIdentifier"],
            "invocationStepId": str(uuid4()),
            "invocationStepTime": params["invocationStepTime"],
            "text": params["payload"]["contentBlocks"][0]["text"],
        }
        self.steps[session_id].append(step)
        return {"invocationStepId": step["invocationStepId"]}

    async def list_invocation_steps(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.calls.append("list_invocation_steps")
        session_id = params["sessionIdentifier"]
        self._session(session_id)
        return {
            "invocationStepSummaries": [
                {key: value for key, value in step.items() if key != "text"}
                for step in self.steps[session_id]
            ]
        }

    async def get_invocation_step(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.calls.append("get_invocation_step")
        session_id = params["sessionIdentifier"]
        for step in self.steps[session_id]:
            if step["invocationStepId"] == params["invocationStepId"]:
                return {
                    "invocationStep": {
                        "payload": {"contentBlocks": [{"text": step["text"]}]}
                    }
                }
        raise AssertionError(params["invocationStepId"])

    async def end_session(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.calls.append("end_session")
        self._session(params["sessionIdentifier"])
        return {}

    async def delete_session(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.calls.append("delete_session")
        session_id = params["sessionIdentifier"]
        self._session(session_id)
        del self.sessions[session_id]
        del self.steps[session_id]
        return {}


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _FakeSessionClient:
    """Serve the conversation store from an in-memory session backend."""
    client = _FakeSessionClient()
    monkeypatch.setattr(conversations, "_client", lambda: client)
    return client


def _usage() -> ResponseUsage:
    """Token usage for the stub backend's canned response."""
    return ResponseUsage(
        input_tokens=5,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens=3,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=8,
    )


class _StubChatModel:
    """Stub chat backend recording the generation request it was given."""

    def __init__(self) -> None:
        self.requests: list[ResponseCreateParams] = []

    def native_store_supported(self) -> bool:
        """Whether the backend chains conversations itself (it does not)."""
        return False

    async def create_response(
        self,
        request: ResponseCreateParams,
        response_id: str,
        created_at: float,
        **_kwargs: object,
    ) -> Response | EventSourceResponse:
        """Record the request and answer with one canned assistant message."""
        self.requests.append(request)
        response = Response(
            id=response_id,
            created_at=int(created_at),
            model=request.model,
            object="response",
            output=[
                ResponseOutputMessage(
                    id=f"{response_id}-msg-0",
                    content=[
                        ResponseOutputText(
                            annotations=[], text="canned answer", type="output_text"
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
            status="completed",
            usage=_usage(),
        )
        if not request.stream:
            return response

        async def _events() -> AsyncGenerator[Any]:
            """Emit the single terminal event of a completed stream."""
            yield json_sse(
                "response.completed",
                ResponseCompletedEvent(
                    response=response, sequence_number=1, type="response.completed"
                ),
            )

        return EventSourceResponse(_events())


@pytest.fixture
def chat_backend(monkeypatch: pytest.MonkeyPatch) -> _StubChatModel:
    """Stub model validation and the chat generation backend."""

    async def _validate_model(
        model_id: str, *_args: object, **_kwargs: object
    ) -> ModelDetails:
        return make_model_details(model_id)

    stub = _StubChatModel()
    monkeypatch.setattr(openai_responses, "validate_model", _validate_model)
    monkeypatch.setattr(openai_responses, "get_chat_model", lambda _model_id: stub)
    return stub


def _create(client: TestClient, **body: Any) -> dict[str, Any]:  # noqa: ANN401
    """Create a conversation and return its object."""
    response = client.post("/v1/conversations", json=body or None)
    assert response.status_code == 200, response.text
    payload: dict[str, Any] = response.json()
    return payload


def _error(response: Any) -> dict[str, Any]:  # noqa: ANN401
    """Return the error envelope of a failed response."""
    payload: dict[str, Any] = response.json()["error"]
    return payload


@pytest.mark.local
@pytest.mark.usefixtures("store")
class TestConversationLifecycle:
    """Create, retrieve, update and delete a conversation.

    Ref: https://developers.openai.com/api/reference/resources/conversations.md
         stdapi/routes/openai_conversations.py:create
    """

    def test_create_without_body(self, app_client: TestClient) -> None:
        """A conversation is created with no request body at all.

        Ref: stdapi/routes/openai_conversations.py:create
        """
        response = app_client.post("/v1/conversations")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["object"] == "conversation"
        assert payload["id"].startswith("conv-")
        assert isinstance(payload["created_at"], int)
        assert payload["metadata"] == {}

    def test_create_with_metadata_and_items(self, app_client: TestClient) -> None:
        """Metadata is echoed and initial items are stored.

        Ref: stdapi/conversations.py:create_conversation
        """
        conversation = _create(
            app_client,
            metadata={"topic": "weather"},
            items=[{"role": "user", "content": "hello"}],
        )
        assert conversation["metadata"] == {"topic": "weather"}
        items = app_client.get(f"/v1/conversations/{conversation['id']}/items").json()
        assert [item["role"] for item in items["data"]] == ["user"]

    def test_retrieve(self, app_client: TestClient) -> None:
        """Retrieving a conversation returns the object it was created as.

        Ref: stdapi/routes/openai_conversations.py:retrieve
        """
        conversation = _create(app_client, metadata={"a": "b"})
        response = app_client.get(f"/v1/conversations/{conversation['id']}")
        assert response.status_code == 200, response.text
        assert response.json() == conversation

    def test_update_merges_and_null_deletes(self, app_client: TestClient) -> None:
        """An update merges keys, and a null value removes one.

        Ref: stdapi/conversations.py:update_conversation
        """
        conversation = _create(app_client, metadata={"a": "1", "b": "2"})
        url = f"/v1/conversations/{conversation['id']}"
        merged = app_client.post(url, json={"metadata": {"c": "3"}})
        assert merged.status_code == 200, merged.text
        assert merged.json()["metadata"] == {"a": "1", "b": "2", "c": "3"}
        removed = app_client.post(url, json={"metadata": {"b": None}})
        assert removed.json()["metadata"] == {"a": "1", "c": "3"}
        unchanged = app_client.post(url, json={"metadata": {}})
        assert unchanged.json()["metadata"] == {"a": "1", "c": "3"}

    def test_update_requires_metadata(self, app_client: TestClient) -> None:
        """An update without ``metadata`` and one with null are distinct errors.

        Ref: stdapi/types/openai_conversations.py:ConversationUpdateParams
        """
        conversation = _create(app_client)
        url = f"/v1/conversations/{conversation['id']}"
        missing = app_client.post(url, json={})
        assert missing.status_code == 400, missing.text
        assert _error(missing)["code"] == "missing_required_parameter"
        null = app_client.post(url, json={"metadata": None})
        assert null.status_code == 400, null.text
        assert _error(null)["code"] == "invalid_type"

    def test_delete(self, app_client: TestClient) -> None:
        """Deleting a conversation makes every route on it answer 404.

        Ref: stdapi/conversations.py:delete_conversation
        """
        conversation = _create(app_client)
        url = f"/v1/conversations/{conversation['id']}"
        deleted = app_client.delete(url)
        assert deleted.status_code == 200, deleted.text
        assert deleted.json() == {
            "id": conversation["id"],
            "object": "conversation.deleted",
            "deleted": True,
        }
        assert app_client.get(url).status_code == 404
        assert app_client.delete(url).status_code == 404
        assert app_client.get(f"{url}/items").status_code == 404


@pytest.mark.local
@pytest.mark.usefixtures("store")
class TestConversationMetadataLimits:
    """Metadata limits, each reported with its own error code.

    Ref: stdapi/types/openai_conversations.py:validate_metadata
    """

    @pytest.mark.parametrize(
        ("metadata", "code"),
        [
            ({f"k{i}": "v" for i in range(17)}, "object_above_max_properties"),
            ({"k" * 65: "v"}, "property_name_above_max_length"),
            ({"k": "v" * 513}, "string_above_max_length"),
            ({"k": 1}, "invalid_type"),
        ],
    )
    def test_rejected(
        self, app_client: TestClient, metadata: dict[str, Any], code: str
    ) -> None:
        """Each metadata limit is rejected with its own error code.

        Ref: stdapi/types/openai_conversations.py:validate_metadata
        """
        response = app_client.post("/v1/conversations", json={"metadata": metadata})
        assert response.status_code == 400, response.text
        error = _error(response)
        assert error["code"] == code
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "metadata"

    def test_boundaries_accepted(self, app_client: TestClient) -> None:
        """The largest metadata the limits allow round-trips unchanged.

        Ref: stdapi/types/openai_conversations.py:METADATA_MAX_KEYS
        """
        metadata = {f"k{i}": "v" * 512 for i in range(16)}
        metadata["j" * 64] = ""
        del metadata["k0"]
        conversation = _create(app_client, metadata=metadata)
        assert conversation["metadata"] == metadata


@pytest.mark.local
@pytest.mark.usefixtures("store")
class TestConversationIdentifiers:
    """The boundary between a malformed identifier and an unknown one.

    Ref: stdapi/conversations.py:validate_conversation_id
    """

    @pytest.mark.parametrize(
        "conversation_id", ["notaconvid", "resp_abc", "CONV_abc", "conv", "conv_"]
    )
    def test_malformed_conversation_id(
        self, app_client: TestClient, conversation_id: str
    ) -> None:
        """A malformed conversation ID is a 400 naming the parameter.

        Ref: stdapi/conversations.py:validate_conversation_id
        """
        response = app_client.get(f"/v1/conversations/{conversation_id}")
        assert response.status_code == 400, response.text
        error = _error(response)
        assert error["code"] == "invalid_value"
        assert error["param"] == "conversation_id"

    def test_well_formed_unknown_conversation_id(self, app_client: TestClient) -> None:
        """A well-formed ID minted elsewhere is a 404, not a validation error.

        Ref: stdapi/conversations.py:validate_conversation_id
        """
        response = app_client.get("/v1/conversations/conv_ABC-123")
        assert response.status_code == 404, response.text
        error = _error(response)
        assert error["param"] is None
        assert error["code"] is None

    def test_conversation_id_too_long(self, app_client: TestClient) -> None:
        """An identifier above the length limit is its own error code.

        Ref: stdapi/conversations.py:_too_long
        """
        response = app_client.get(f"/v1/conversations/conv_{'a' * 64}")
        assert response.status_code == 400, response.text
        assert _error(response)["code"] == "string_above_max_length"

    def test_item_id_validation_is_charset_based(self, app_client: TestClient) -> None:
        """An item ID is checked for shape, not for a known prefix.

        Ref: stdapi/conversations.py:validate_item_id
        """
        conversation = _create(app_client)
        url = f"/v1/conversations/{conversation['id']}/items"
        assert app_client.get(f"{url}/abc_def").status_code == 404
        malformed = app_client.get(f"{url}/msg-abc")
        assert malformed.status_code == 400, malformed.text
        assert _error(malformed)["param"] == "item_id"


@pytest.mark.local
@pytest.mark.usefixtures("store")
class TestConversationItems:
    """Adding, listing, retrieving and deleting conversation items.

    Ref: https://developers.openai.com/api/reference/resources/conversations.md
         stdapi/routes/openai_conversations.py:add_items
    """

    def test_add_returns_only_the_new_items(self, app_client: TestClient) -> None:
        """Adding items returns the batch that was added, not the conversation.

        Ref: stdapi/routes/openai_conversations.py:add_items
        """
        conversation = _create(app_client, items=[{"role": "user", "content": "first"}])
        url = f"/v1/conversations/{conversation['id']}/items"
        response = app_client.post(
            url, json={"items": [{"role": "user", "content": "second"}]}
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["object"] == "list"
        assert len(payload["data"]) == 1
        assert payload["first_id"] == payload["data"][0]["id"]
        assert payload["last_id"] == payload["data"][-1]["id"]
        assert payload["has_more"] is False
        assert len(app_client.get(url).json()["data"]) == 2

    def test_server_assigns_item_ids(self, app_client: TestClient) -> None:
        """A client-supplied item ID is replaced by the server's own.

        Ref: stdapi/conversations.py:stored_item
        """
        conversation = _create(app_client)
        response = app_client.post(
            f"/v1/conversations/{conversation['id']}/items",
            json={"items": [{"id": "msg_client", "role": "user", "content": "hello"}]},
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"][0]["id"] != "msg_client"

    @pytest.mark.parametrize(
        ("body", "code"),
        [
            ({}, "missing_required_parameter"),
            ({"items": []}, "empty_array"),
            (
                {"items": [{"role": "user", "content": "x"}] * 21},
                "array_above_max_length",
            ),
        ],
    )
    def test_items_limits(
        self, app_client: TestClient, body: dict[str, Any], code: str
    ) -> None:
        """``items`` is required, non-empty and capped, each with its own code.

        Ref: stdapi/types/openai_conversations.py:_validate_items
        """
        conversation = _create(app_client)
        response = app_client.post(
            f"/v1/conversations/{conversation['id']}/items", json=body
        )
        assert response.status_code == 400, response.text
        assert _error(response)["code"] == code

    def test_unknown_item_reference(self, app_client: TestClient) -> None:
        """An ``item_reference`` to an absent item is a 404.

        Ref: stdapi/routes/openai_conversations.py:_new_items
        """
        conversation = _create(app_client)
        response = app_client.post(
            f"/v1/conversations/{conversation['id']}/items",
            json={"items": [{"type": "item_reference", "id": "msg_missing"}]},
        )
        assert response.status_code == 404, response.text

    def test_pagination_across_a_page_boundary(self, app_client: TestClient) -> None:
        """``order=asc`` with ``after`` walks the conversation in order.

        Ref: stdapi/routes/openai_conversations.py:_item_list
        """
        conversation = _create(
            app_client, items=[{"role": "user", "content": str(i)} for i in range(5)]
        )
        url = f"/v1/conversations/{conversation['id']}/items"
        first = app_client.get(url, params={"order": "asc", "limit": 2}).json()
        assert [item["content"][0]["text"] for item in first["data"]] == ["0", "1"]
        assert first["has_more"] is True
        second = app_client.get(
            url, params={"order": "asc", "limit": 2, "after": first["last_id"]}
        ).json()
        assert [item["content"][0]["text"] for item in second["data"]] == ["2", "3"]

    def test_listing_defaults_to_newest_first(self, app_client: TestClient) -> None:
        """The listing defaults to reverse conversation order.

        Ref: stdapi/routes/openai_conversations.py:list_items
        """
        conversation = _create(
            app_client, items=[{"role": "user", "content": str(i)} for i in range(3)]
        )
        data = app_client.get(f"/v1/conversations/{conversation['id']}/items").json()[
            "data"
        ]
        assert [item["content"][0]["text"] for item in data] == ["2", "1", "0"]

    def test_empty_listing_has_no_cursors(self, app_client: TestClient) -> None:
        """An empty conversation lists no items and no cursors.

        Ref: stdapi/routes/openai_conversations.py:_item_list
        """
        conversation = _create(app_client)
        payload = app_client.get(f"/v1/conversations/{conversation['id']}/items").json()
        assert payload["data"] == []
        assert payload.get("first_id") is None
        assert payload.get("last_id") is None
        assert payload["has_more"] is False

    def test_limit_above_maximum_is_rejected(self, app_client: TestClient) -> None:
        """A limit above the documented maximum is a 400.

        Ref: stdapi/routes/openai_conversations.py:_MAX_LIMIT
        """
        conversation = _create(app_client)
        response = app_client.get(
            f"/v1/conversations/{conversation['id']}/items", params={"limit": 101}
        )
        assert response.status_code == 400, response.text

    def test_unknown_after_cursor(self, app_client: TestClient) -> None:
        """An ``after`` cursor naming no item is a 404 on that parameter.

        Ref: stdapi/routes/openai_conversations.py:_item_list
        """
        conversation = _create(app_client, items=[{"role": "user", "content": "hello"}])
        response = app_client.get(
            f"/v1/conversations/{conversation['id']}/items",
            params={"after": "msg_absent"},
        )
        assert response.status_code == 404, response.text
        assert _error(response)["param"] == "after"

    def test_retrieve_and_delete_item(self, app_client: TestClient) -> None:
        """Deleting an item returns the conversation and removes the item.

        Ref: stdapi/routes/openai_conversations.py:delete_conversation_item
        """
        conversation = _create(app_client, items=[{"role": "user", "content": "hello"}])
        url = f"/v1/conversations/{conversation['id']}/items"
        item_id = app_client.get(url).json()["data"][0]["id"]
        retrieved = app_client.get(f"{url}/{item_id}")
        assert retrieved.status_code == 200, retrieved.text
        assert retrieved.json()["id"] == item_id
        deleted = app_client.delete(f"{url}/{item_id}")
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["object"] == "conversation"
        assert deleted.json()["id"] == conversation["id"]
        assert app_client.get(f"{url}/{item_id}").status_code == 404
        assert app_client.delete(f"{url}/{item_id}").status_code == 404
        assert app_client.get(url).json()["data"] == []

    def test_include_controls_reasoning_encrypted_content(
        self, app_client: TestClient
    ) -> None:
        """Reasoning encrypted content is returned only when asked for.

        Ref: stdapi/routes/openai_conversations.py:_visible_items
        """
        conversation = _create(
            app_client,
            items=[{"type": "reasoning", "summary": [], "encrypted_content": "opaque"}],
        )
        url = f"/v1/conversations/{conversation['id']}/items"
        assert "encrypted_content" not in app_client.get(url).json()["data"][0]
        included = app_client.get(
            url, params={"include": ["reasoning.encrypted_content"]}
        ).json()
        assert included["data"][0]["encrypted_content"] == "opaque"

    def test_unsupported_include_value(self, app_client: TestClient) -> None:
        """An unsupported ``include`` value is rejected.

        Ref: stdapi/routes/openai_conversations.py:_Include
        """
        conversation = _create(app_client)
        response = app_client.get(
            f"/v1/conversations/{conversation['id']}/items",
            params={"include": ["bogus"]},
        )
        assert response.status_code == 400, response.text


@pytest.mark.local
@pytest.mark.usefixtures("store", "chat_backend")
class TestResponsesConversationParameter:
    """``conversation`` on POST /v1/responses.

    Ref: https://developers.openai.com/api/docs/guides/conversation-state.md
         stdapi/routes/openai_responses.py:_apply_conversation
    """

    def test_mutually_exclusive_with_previous_response_id(
        self, app_client: TestClient
    ) -> None:
        """A request cannot chain on a response and a conversation at once.

        Ref: stdapi/types/openai_responses.py:reject_conversation_with_previous_response
        """
        response = app_client.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "hi",
                "conversation": "conv-1",
                "previous_response_id": "resp-1",
            },
        )
        assert response.status_code == 400, response.text
        assert _error(response)["code"] == "mutually_exclusive_parameters"

    def test_malformed_conversation_is_rejected(self, app_client: TestClient) -> None:
        """A malformed conversation ID names the ``conversation`` parameter.

        Ref: stdapi/routes/openai_responses.py:_resolve_conversation
        """
        response = app_client.post(
            "/v1/responses",
            json={"model": "m", "input": "hi", "conversation": "notaconvid"},
        )
        assert response.status_code == 400, response.text
        error = _error(response)
        assert error["code"] == "invalid_value"
        assert error["param"] == "conversation"

    def test_unknown_conversation_is_not_found(self, app_client: TestClient) -> None:
        """An unknown conversation is a 404.

        Ref: stdapi/conversations.py:conversation_not_found
        """
        response = app_client.post(
            "/v1/responses",
            json={"model": "m", "input": "hi", "conversation": "conv-unknown"},
        )
        assert response.status_code == 404, response.text

    @pytest.mark.parametrize("as_object", [False, True])
    def test_turn_is_appended_and_echoed(
        self, app_client: TestClient, as_object: bool
    ) -> None:
        """Both parameter forms are accepted, echoed, and append the turn.

        Ref: stdapi/routes/openai_responses.py:_append_turn
        """
        conversation = _create(app_client)
        reference: Any = {"id": conversation["id"]} if as_object else conversation["id"]
        response = app_client.post(
            "/v1/responses",
            json={"model": "m", "input": "hello", "conversation": reference},
        )
        assert response.status_code == 200, response.text
        assert response.json()["conversation"] == {"id": conversation["id"]}
        items = app_client.get(
            f"/v1/conversations/{conversation['id']}/items", params={"order": "asc"}
        ).json()["data"]
        assert [item["role"] for item in items] == ["user", "assistant"]
        assert items[0]["content"][0]["text"] == "hello"

    def test_stored_items_are_the_next_turn_prefix(
        self, app_client: TestClient, chat_backend: _StubChatModel
    ) -> None:
        """The second turn reaches the model with the first turn ahead of it.

        Ref: stdapi/routes/openai_responses.py:_with_conversation_prefix
        """
        conversation = _create(app_client)
        body = {"model": "m", "conversation": conversation["id"]}
        app_client.post("/v1/responses", json={**body, "input": "first"})
        app_client.post("/v1/responses", json={**body, "input": "second"})
        second_turn = chat_backend.requests[-1].input
        assert isinstance(second_turn, list)
        assert len(second_turn) == 3

    def test_store_false_appends_nothing(self, app_client: TestClient) -> None:
        """``store=false`` uses the conversation without adding to it.

        Ref: stdapi/routes/openai_responses.py:_apply_conversation
        """
        conversation = _create(app_client)
        response = app_client.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "hello",
                "conversation": conversation["id"],
                "store": False,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["conversation"] == {"id": conversation["id"]}
        items = app_client.get(f"/v1/conversations/{conversation['id']}/items").json()
        assert items["data"] == []

    def test_streamed_turn_is_appended_once_the_stream_ends(
        self, app_client: TestClient
    ) -> None:
        """A streamed response appends its turn and echoes the conversation.

        Ref: stdapi/routes/openai_responses.py:_append_streamed_turn
        """
        conversation = _create(app_client)
        streamed = app_client.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "hello",
                "conversation": conversation["id"],
                "stream": True,
            },
        )
        assert streamed.status_code == 200, streamed.text
        assert f'"conversation":{{"id":"{conversation["id"]}"}}' in streamed.text
        items = app_client.get(
            f"/v1/conversations/{conversation['id']}/items", params={"order": "asc"}
        ).json()["data"]
        assert [item["role"] for item in items] == ["user", "assistant"]

    def test_streamed_store_false_appends_nothing(self, app_client: TestClient) -> None:
        """A streamed response with ``store=false`` adds nothing either.

        Ref: stdapi/routes/openai_responses.py:_streamed_result
        """
        conversation = _create(app_client)
        streamed = app_client.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "hello",
                "conversation": conversation["id"],
                "store": False,
                "stream": True,
            },
        )
        assert streamed.status_code == 200, streamed.text
        items = app_client.get(f"/v1/conversations/{conversation['id']}/items").json()
        assert items["data"] == []

    def test_input_tokens_rejects_both_chaining_parameters(
        self, app_client: TestClient
    ) -> None:
        """Token counting enforces the same mutual exclusion.

        Ref: stdapi/types/openai_responses.py:InputTokenCountParams
        """
        response = app_client.post(
            "/v1/responses/input_tokens",
            json={
                "model": "m",
                "input": "hi",
                "conversation": "conv-1",
                "previous_response_id": "resp-1",
            },
        )
        assert response.status_code == 400, response.text
        assert _error(response)["code"] == "mutually_exclusive_parameters"


@pytest.mark.local
def test_stored_response_is_not_a_conversation(
    app_client: TestClient, store: _FakeSessionClient
) -> None:
    """A session holding another object kind is not reachable as a conversation.

    Ref: stdapi/conversations.py:_session_metadata
    """
    store.sessions["other"] = {
        "sessionId": "other",
        "sessionArn": "arn:aws:bedrock:::session/other",
        "createdAt": datetime.now(tz=UTC),
        "sessionMetadata": {},
        "tags": {KIND_TAG: "response"},
    }
    store.steps["other"] = []
    assert app_client.get("/v1/conversations/conv-other").status_code == 404


@pytest.mark.local
def test_minted_item_ids_pass_the_item_id_validator() -> None:
    """Minted item IDs match the shape the item routes accept.

    Ref: stdapi/conversations.py:new_item_id
    """
    for item_type in ("message", "reasoning", "unknown_type"):
        item_id = conversations.new_item_id(item_type)
        conversations.validate_item_id(item_id)


def _item_text(item: ConversationItem) -> str:
    """Return the text a message item carries, whichever content part holds it.

    Args:
        item: A listed conversation item.

    Returns:
        The text of its first content part.
    """
    content = getattr(item, "content", None)
    assert content, f"item {item.id} carries no content"
    return getattr(content[0], "text", "")


def _item_id(item: ConversationItem) -> str:
    """Return a listed item's identifier, which every stored item carries.

    Args:
        item: A listed conversation item.

    Returns:
        Its identifier.
    """
    assert item.id is not None, "a stored item must be addressable by ID"
    return item.id


#: These tests write into one account-wide conversation store, and a cursor read spans
#: two requests; without a group ``--dist=loadgroup`` would spread them across workers.
@pytest.mark.xdist_group("openai_conversations")
class TestConversationsLive:
    """The Conversations API against real storage, on whichever target is selected.

    The stubbed suite above never reaches a store, so nothing there can catch a
    conversation that writes but does not read back, a cursor the backend
    orders differently, or an item the store drops. Every assertion here is
    upstream behaviour, which is what lets the same bodies run against the
    official API.

    Ref: https://developers.openai.com/api/reference/resources/conversations.md
         stdapi/conversations.py
    """

    def test_lifecycle_and_item_pagination(self, openai_client: OpenAI) -> None:
        """A conversation round-trips its metadata, added items, cursors and deletion.

        One conversation walks the whole surface — create with initial items,
        retrieve, update metadata, add more items, page through them with
        ``after``, retrieve and delete one, then delete the conversation — so a
        single stored object proves the read-back of every write.

        Ref: https://developers.openai.com/api/reference/resources/conversations.md
             stdapi/conversations.py:create_conversation
        """
        conversation = openai_client.conversations.create(
            metadata={"topic": "storage"},
            items=[{"role": "user", "content": f"line {index}"} for index in range(2)],
        )
        try:
            assert conversation.object == "conversation"
            assert conversation.metadata == {"topic": "storage"}
            assert openai_client.conversations.retrieve(conversation.id).id == (
                conversation.id
            )

            # Sent whole rather than as a patch: the update merges here and the
            # same call must leave the same metadata on a target that replaces.
            updated = openai_client.conversations.update(
                conversation.id, metadata={"topic": "storage", "stage": "live"}
            )
            assert updated.metadata == {"topic": "storage", "stage": "live"}

            added = openai_client.conversations.items.create(
                conversation.id,
                items=[
                    {"role": "user", "content": f"line {index}"}
                    for index in range(2, 5)
                ],
            )
            assert [_item_text(item) for item in added.data] == [
                "line 2",
                "line 3",
                "line 4",
            ]

            first = openai_client.conversations.items.list(
                conversation.id, order="asc", limit=2
            )
            assert [_item_text(item) for item in first.data] == ["line 0", "line 1"]
            assert first.has_more is True
            second = openai_client.conversations.items.list(
                conversation.id, order="asc", limit=2, after=_item_id(first.data[-1])
            )
            assert [_item_text(item) for item in second.data] == ["line 2", "line 3"]

            item_id = _item_id(first.data[0])
            retrieved = openai_client.conversations.items.retrieve(
                item_id, conversation_id=conversation.id
            )
            assert retrieved.id == item_id
            assert _item_text(retrieved) == "line 0"

            openai_client.conversations.items.delete(
                item_id, conversation_id=conversation.id
            )
            with pytest.raises(NotFoundError):
                openai_client.conversations.items.retrieve(
                    item_id, conversation_id=conversation.id
                )

            deleted = openai_client.conversations.delete(conversation.id)
            assert deleted.deleted is True
            assert deleted.id == conversation.id
            with pytest.raises(NotFoundError) as gone:
                openai_client.conversations.retrieve(conversation.id)
            assert gone.value.status_code == 404
        finally:
            with contextlib.suppress(Exception):
                openai_client.conversations.delete(conversation.id)

    def test_unknown_conversation_is_not_found(self, openai_client: OpenAI) -> None:
        """A well-formed identifier naming no conversation answers 404.

        An ID minted elsewhere is "not found" rather than a validation error, so
        a client migrating from another provider reads the same answer here.

        Ref: stdapi/conversations.py:validate_conversation_id
        """
        with pytest.raises(NotFoundError) as unknown:
            openai_client.conversations.retrieve(f"conv_{uuid4().hex}")
        assert unknown.value.status_code == 404

    def test_response_reads_and_appends_to_the_conversation(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A second response is prefixed with the conversation and appends its turn.

        The stored prefix is measured rather than recited: the second turn sends
        a shorter input than the first yet bills far more input tokens, which
        only the conversation ahead of it can account for. Whether the model
        then repeats what it was told is the model's decision, not the
        gateway's, so nothing here asserts on its answer.

        Ref: https://developers.openai.com/api/docs/guides/conversation-state.md
             stdapi/routes/openai_responses.py:_apply_conversation
        """
        conversation = openai_client.conversations.create()
        try:
            first = openai_client.responses.create(
                model=responses_model,
                input="The code word for today is xylophone. Acknowledge it.",
                conversation=conversation.id,
            )
            assert first.conversation is not None
            assert first.conversation.id == conversation.id
            assert first.usage is not None

            second = openai_client.responses.create(
                model=responses_model,
                input="Repeat the code word.",
                conversation=conversation.id,
            )
            assert second.usage is not None
            assert second.usage.input_tokens > first.usage.input_tokens

            items = openai_client.conversations.items.list(
                conversation.id, order="asc", limit=100
            )
            messages = [item for item in items.data if item.type == "message"]
            assert [item.role for item in messages] == [
                "user",
                "assistant",
                "user",
                "assistant",
            ]
            assert _item_text(messages[0]) == (
                "The code word for today is xylophone. Acknowledge it."
            )
            assert _item_text(messages[2]) == "Repeat the code word."
        finally:
            with contextlib.suppress(Exception):
                openai_client.conversations.delete(conversation.id)
