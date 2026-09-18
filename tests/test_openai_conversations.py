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
from json import loads
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from openai import BadRequestError, NotFoundError
from sse_starlette import EventSourceResponse

from stdapi import conversations
from stdapi.api_errors import ApiError
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
    from collections.abc import AsyncGenerator, Iterator

    from openai import OpenAI
    from openai.types.conversations import Conversation, ConversationItem
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


#: An ``additional_tools`` item, the shape a client replays from a prior transcript.
_ADDITIONAL_TOOLS_ITEM: dict[str, Any] = {
    "type": "additional_tools",
    # Upstream's stored item type allows eight roles, but the live API accepts only
    # this one on a write: `role: "assistant"` answers 400 `invalid_value`.
    "role": "developer",
    "tools": [
        {
            "type": "function",
            "name": "get_weather",
            "parameters": {"type": "object"},
            "strict": True,
        }
    ],
}


def _stored_item_ids(store: _FakeSessionClient, conversation_id: str) -> list[str]:
    """Return the ID of every item written to a conversation, listable or not.

    Args:
        store: The in-memory session backend the conversation was written to.
        conversation_id: Public conversation identifier.

    Returns:
        The identifiers in write order.
    """
    return [
        item["id"]
        for step in store.steps[conversation_id.split("-", 1)[-1]]
        for item in loads(step["text"])["conversation"].get("items") or ()
    ]


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
class TestConversationsUnavailable:
    """A deployment whose session storage is denied or absent answers one 503.

    Conversations are the only API served entirely by Bedrock session storage,
    which is optional: an account without the permissions, or a region that does
    not offer the endpoint at all, must read as "not available here" rather than
    as a permission error the caller could act on.

    Ref: stdapi/conversations.py:_session_calls
         stdapi/api_errors.py:feature_unavailable_guard
    """

    @pytest.mark.parametrize(
        ("call", "permission"),
        [
            ("create_session", "bedrock:CreateSession"),
            ("get_session", "bedrock:GetSession"),
            ("update_session", "bedrock:UpdateSession"),
        ],
    )
    def test_a_denied_session_call_names_its_permission(
        self,
        app_client: TestClient,
        store: _FakeSessionClient,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
        call: str,
        permission: str,
    ) -> None:
        """Each session call answers 503, with the permission in the server log.

        ``bedrock:UpdateSession`` is the one a deployment predating conversations
        lacks, and it serves the metadata update alone.

        Ref: stdapi/conversations.py:update_conversation
        """
        conversation = _create(app_client, metadata={"topic": "weather"})

        async def _denied(**_params: object) -> dict[str, Any]:
            raise ClientError(
                {"Error": {"Code": "AccessDeniedException", "Message": "denied"}}, call
            )

        monkeypatch.setattr(store, call, _denied)
        capfd.readouterr()
        response = (
            app_client.post("/v1/conversations")
            if call == "create_session"
            else app_client.post(
                f"/v1/conversations/{conversation['id']}",
                json={"metadata": {"topic": "rain"}},
            )
        )

        assert response.status_code == 503, response.text
        error = _error(response)
        assert "not available on the current server" in error["message"]
        assert "bedrock" not in error["message"]
        assert permission in capfd.readouterr().out

    def test_a_region_without_the_endpoint_is_not_a_server_error(
        self,
        app_client: TestClient,
        store: _FakeSessionClient,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """An unreachable session endpoint answers 503 naming the region setting.

        Session storage covers fewer Regions than model inference, so a
        deployment can sit where the endpoint does not resolve at all.

        Ref: stdapi/conversations.py:_UNREACHABLE_DETAIL
        """

        async def _unreachable(**_params: object) -> dict[str, Any]:
            raise EndpointConnectionError(endpoint_url="https://x.invalid")

        monkeypatch.setattr(store, "create_session", _unreachable)
        capfd.readouterr()
        response = app_client.post("/v1/conversations")

        assert response.status_code == 503, response.text
        assert "not available on the current server" in _error(response)["message"]
        assert "aws_bedrock_regions" in capfd.readouterr().out


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

    def test_reference_to_an_item_already_held_is_refused(
        self, app_client: TestClient
    ) -> None:
        """Referencing an item the conversation holds is a 400, as upstream.

        An ``item_reference`` asks for an item that exists elsewhere to be
        brought in, so naming one already here is refused rather than answered
        with a page -- the code is asserted because a client tells this refusal
        apart from the other 400s by it.

        Ref: https://developers.openai.com/api/reference/resources/conversations.md
             Live: POST /v1/conversations/{id}/items with a reference to an item
             of that conversation answers 400 {'message': 'Item already in
             conversation', 'type': 'invalid_request_error', 'param': 'items',
             'code': 'item_already_in_conversation'}
        """
        conversation = _create(app_client, items=[{"role": "user", "content": "first"}])
        url = f"/v1/conversations/{conversation['id']}/items"
        item_id = app_client.get(url).json()["data"][0]["id"]

        response = app_client.post(
            url, json={"items": [{"type": "item_reference", "id": item_id}]}
        )

        assert response.status_code == 400, response.text
        assert _error(response)["code"] == "item_already_in_conversation"
        assert _error(response)["param"] == "items"
        assert len(app_client.get(url).json()["data"]) == 1

    def test_a_refused_reference_stores_none_of_the_batch(
        self, app_client: TestClient
    ) -> None:
        """One bad reference refuses the whole add, storing nothing.

        Ref: Live: the same request against the official API answers 400
             `item_already_in_conversation` and the listing afterwards holds
             only the item created with the conversation.
        """
        conversation = _create(app_client, items=[{"role": "user", "content": "first"}])
        url = f"/v1/conversations/{conversation['id']}/items"
        item_id = app_client.get(url).json()["data"][0]["id"]

        response = app_client.post(
            url,
            json={
                "items": [
                    {"type": "item_reference", "id": item_id},
                    {"role": "user", "content": "second"},
                ]
            },
        )

        assert response.status_code == 400, response.text
        listed = app_client.get(url, params={"order": "asc"}).json()["data"]
        assert [item["content"][0]["text"] for item in listed] == ["first"]

    def test_an_unknown_reference_wins_over_a_known_one(
        self, app_client: TestClient
    ) -> None:
        """An unresolvable reference is the 404, wherever it sits in the batch.

        Ref: Live: a batch holding a known and an unknown reference answers 404
             for the unknown one in either order, never the 400 the known one
             alone would earn.
        """
        conversation = _create(app_client, items=[{"role": "user", "content": "first"}])
        url = f"/v1/conversations/{conversation['id']}/items"
        item_id = app_client.get(url).json()["data"][0]["id"]

        response = app_client.post(
            url,
            json={
                "items": [
                    {"type": "item_reference", "id": item_id},
                    {"type": "item_reference", "id": "msg_missing"},
                ]
            },
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

    def test_an_additional_tools_item_round_trips(self, app_client: TestClient) -> None:
        """An ``additional_tools`` item is stored, listed and retrieved.

        Upstream carries this item in the conversation item union as well as in
        the input one, so a client replaying a prior transcript reads it back
        rather than losing the conversation.

        Ref: https://developers.openai.com/api/reference/resources/conversations.md
             openai.types.conversations.conversation_item.ConversationItem
             openai.types.responses.response_item.ResponseItem
        """
        conversation = _create(app_client, items=[_ADDITIONAL_TOOLS_ITEM])
        url = f"/v1/conversations/{conversation['id']}/items"

        listed = app_client.get(url)

        assert listed.status_code == 200, listed.text
        (stored,) = listed.json()["data"]
        assert stored["type"] == "additional_tools"
        assert stored["role"] == "developer"
        assert stored["tools"][0]["name"] == "get_weather"
        retrieved = app_client.get(f"{url}/{stored['id']}")
        assert retrieved.status_code == 200, retrieved.text
        assert retrieved.json()["id"] == stored["id"]

    def test_an_assistant_turn_replayed_as_input_content_round_trips(
        self, app_client: TestClient
    ) -> None:
        """An assistant message whose parts were relabelled reads back.

        A client replaying a prior assistant turn may send its text as
        ``input_text`` rather than ``output_text`` -- Codex does. The input union
        accepts that, the store keeps it and the model is replayed with it, so a
        listing that cannot express it would hand back a conversation missing a
        turn the caller is still being billed for.

        Ref: https://developers.openai.com/api/reference/resources/conversations.md
             stdapi/types/openai_responses.py:ResponseOutputMessageInput
        """
        item = {"role": "assistant", "content": [{"type": "input_text", "text": "hi"}]}
        conversation = _create(app_client, items=[item])
        url = f"/v1/conversations/{conversation['id']}/items"

        listed = app_client.get(url)

        assert listed.status_code == 200, listed.text
        (stored,) = listed.json()["data"]
        assert stored["role"] == "assistant"
        assert stored["content"][0]["text"] == "hi"
        retrieved = app_client.get(f"{url}/{stored['id']}")
        assert retrieved.status_code == 200, retrieved.text

    def test_a_function_call_without_a_status_round_trips(
        self, app_client: TestClient
    ) -> None:
        """A tool call sent without ``status`` reads back as completed.

        ``status`` is optional on a sent item and always present on a listed
        one, so the canonical shape a client sends has to survive both reads.

        Ref: https://developers.openai.com/api/reference/resources/conversations.md
             openai.types.responses.response_item.ResponseFunctionToolCallItem
        """
        conversation = _create(
            app_client,
            items=[
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{}",
                }
            ],
        )
        url = f"/v1/conversations/{conversation['id']}/items"

        listed = app_client.get(url)

        assert listed.status_code == 200, listed.text
        (stored,) = listed.json()["data"]
        assert stored["type"] == "function_call"
        assert stored["call_id"] == "call_1"
        assert stored["status"] == "completed"
        retrieved = app_client.get(f"{url}/{stored['id']}")
        assert retrieved.status_code == 200, retrieved.text
        assert retrieved.json()["status"] == "completed"

    def test_an_input_only_item_is_accepted_and_left_out_of_the_conversation(
        self, app_client: TestClient
    ) -> None:
        """A ``compaction_trigger`` is accepted, and never becomes an item.

        It instructs the request it travels with; no conversation item type can
        express it, so it is dropped from the answer the way an item the
        conversation cannot hold has always been, while the rest of the batch
        is stored.

        Ref: https://developers.openai.com/api/reference/resources/conversations.md
             openai.types.conversations.conversation_item.ConversationItem
        """
        conversation = _create(app_client)
        url = f"/v1/conversations/{conversation['id']}/items"

        added = app_client.post(
            url,
            json={
                "items": [
                    {"role": "user", "content": "summarise this"},
                    {"type": "compaction_trigger"},
                ]
            },
        )

        assert added.status_code == 200, added.text
        assert [item["type"] for item in added.json()["data"]] == ["message"]
        listed = app_client.get(url)
        assert listed.status_code == 200, listed.text
        assert [item["type"] for item in listed.json()["data"]] == ["message"]

    def test_a_dropped_item_is_absent_from_the_cursors_and_from_a_retrieval(
        self, app_client: TestClient, store: _FakeSessionClient
    ) -> None:
        """An item that cannot be listed counts for no page and no retrieval.

        The store is append-only, so a page that refused itself over one item
        it cannot express would lock the client out of the conversation for
        good: the listing skips it, the page bounds ignore it, and its
        identifier -- which the client was never given -- answers 404.

        Ref: stdapi/conversations.py:listable_items
             stdapi/routes/openai_conversations.py:_item_list
        """
        conversation = _create(app_client)
        url = f"/v1/conversations/{conversation['id']}/items"
        app_client.post(
            url,
            json={
                "items": [
                    {"type": "compaction_trigger"},
                    *({"role": "user", "content": str(index)} for index in range(2)),
                ]
            },
        )

        page = app_client.get(url, params={"order": "asc", "limit": 2})

        assert page.status_code == 200, page.text
        payload = page.json()
        assert [item["content"][0]["text"] for item in payload["data"]] == ["0", "1"]
        assert payload["has_more"] is False
        dropped = [
            item_id
            for item_id in _stored_item_ids(store, conversation["id"])
            if item_id not in {item["id"] for item in payload["data"]}
        ]
        assert dropped, "the trigger must still be in the store to be filtered out"
        assert app_client.get(f"{url}/{dropped[0]}").status_code == 404


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

    @pytest.mark.parametrize("stream", [False, True])
    def test_a_failed_append_never_takes_the_answer_with_it(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch, stream: bool
    ) -> None:
        """A conversation deleted mid-request costs the turn, not the response.

        The answer is generated, billed and -- when asked for -- stored before
        the turn is appended, so failing the request there returns an error for
        work the caller has already paid for and cannot retrieve. Both the
        streamed and the non-streamed path log it and answer normally.

        Ref: stdapi/routes/openai_responses.py:_append_turn
        """
        conversation = _create(app_client)

        message = "The conversation was deleted while this request ran."

        async def _deleted(*_args: object, **_kwargs: object) -> None:
            raise ApiError(message, status=404)

        monkeypatch.setattr(openai_responses, "append_items", _deleted)

        response = app_client.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": "hello",
                "conversation": conversation["id"],
                "stream": stream,
            },
        )

        assert response.status_code == 200, response.text
        assert "canned answer" in response.text
        items = app_client.get(f"/v1/conversations/{conversation['id']}/items").json()
        assert items["data"] == []

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

    def test_a_turn_carrying_an_input_only_item_stays_readable(
        self, app_client: TestClient
    ) -> None:
        """An input-only item replayed in ``input`` never costs the conversation.

        Replaying a prior transcript into ``input`` is ordinary client
        behaviour, and every item of that turn is appended to the conversation,
        so an item no conversation item type can express must leave the rest of
        the turn readable instead of taking the conversation with it.

        Ref: stdapi/routes/openai_responses.py:_turn_input_items
             stdapi/conversations.py:listable_items
        """
        conversation = _create(app_client)

        response = app_client.post(
            "/v1/responses",
            json={
                "model": "m",
                "input": [
                    {"role": "user", "content": "hello"},
                    _ADDITIONAL_TOOLS_ITEM,
                    {"type": "compaction_trigger"},
                ],
                "conversation": conversation["id"],
            },
        )

        assert response.status_code == 200, response.text
        listed = app_client.get(
            f"/v1/conversations/{conversation['id']}/items", params={"order": "asc"}
        )
        assert listed.status_code == 200, listed.text
        assert [item["type"] for item in listed.json()["data"]] == [
            "message",
            "additional_tools",
            "message",
        ]

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
def test_stored_response_cannot_be_written_to_as_a_conversation(
    app_client: TestClient, store: _FakeSessionClient
) -> None:
    """Appending items to a session of another kind is a 404 that writes nothing.

    The read routes check the session tag, so the write route must check it too:
    an append that lands in a stored response's session both corrupts that
    object and turns it into a readable conversation, since a session holding a
    conversation document is one by :func:`stdapi.conversations.load_items`.

    Ref: stdapi/conversations.py:append_items
         stdapi/routes/openai_conversations.py:add_items
    """
    store.sessions["other"] = {
        "sessionId": "other",
        "sessionArn": "arn:aws:bedrock:::session/other",
        "createdAt": datetime.now(tz=UTC),
        "sessionMetadata": {},
        "tags": {KIND_TAG: "response"},
    }
    store.steps["other"] = []

    response = app_client.post(
        "/v1/conversations/conv-other/items",
        json={"items": [{"role": "user", "content": "injected"}]},
    )

    assert response.status_code == 404, response.text
    assert store.steps["other"] == [], "a foreign session must not be written to"
    assert app_client.get("/v1/conversations/conv-other/items").status_code == 404


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


@contextlib.contextmanager
def _live_conversation(client: OpenAI) -> Iterator[Conversation]:
    """Yield a freshly created conversation and delete it however the test ends.

    Args:
        client: The OpenAI SDK client bound to the selected target.

    Yields:
        The created conversation.
    """
    conversation = client.conversations.create()
    try:
        yield conversation
    finally:
        with contextlib.suppress(Exception):
            client.conversations.delete(conversation.id)


def _field(item: object, path: str) -> Any:  # noqa: ANN401
    """Read a dotted path out of a listed item.

    A segment is a list index when it is a digit, a mapping key when the value
    reached is a ``dict`` -- which is what a field the SDK model does not declare
    comes back as, kept only by its ``extra="allow"`` -- and an attribute
    otherwise.

    Args:
        item: A listed or retrieved conversation item.
        path: Segments separated by dots.

    Returns:
        The value at that path.
    """
    value: Any = item
    for name in path.split("."):
        if name.isdigit():
            value = value[int(name)]
        elif isinstance(value, dict):
            value = value[name]
        else:
            value = getattr(value, name)
    return value


#: The replayed detail each round-trip item carries, distinctive enough to trace.
_REPLAYED_DETAIL = "conversation item round trip"

#: Replayed tool-call items a write accepts, with the path proving they read back.
_REPLAYED_TOOL_CALL_ITEMS = [
    pytest.param(
        {
            "type": "computer_call",
            "id": "cu_replayed",
            "call_id": "call_replayed_computer",
            "action": {"type": "screenshot"},
            "pending_safety_checks": [],
            "status": "completed",
        },
        "call_id",
        "call_replayed_computer",
        id="computer_call",
    ),
    pytest.param(
        {
            "type": "web_search_call",
            "id": "ws_replayed",
            "action": {"type": "search", "query": _REPLAYED_DETAIL},
            "status": "completed",
        },
        "action.query",
        _REPLAYED_DETAIL,
        id="web_search_call",
    ),
    pytest.param(
        {
            # Upstream mints the ID but still validates the one sent: a
            # `tool_search_call` whose ID does not start with `tsc` answers 400.
            "type": "tool_search_call",
            "id": "tsc_replayed",
            "call_id": "call_replayed_tool_search",
            "arguments": {"query": _REPLAYED_DETAIL},
            "status": "completed",
        },
        "arguments",
        {"query": _REPLAYED_DETAIL},
        id="tool_search_call",
    ),
    pytest.param(
        {
            "type": "tool_search_output",
            "id": "tso_replayed",
            "call_id": "call_replayed_tool_search",
            "status": "completed",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ],
        },
        "tools.0.name",
        "get_weather",
        id="tool_search_output",
    ),
]

#: Replayed tool-call items whose detail the official API stores as null.
_DISCARDED_TOOL_CALL_DETAILS = [
    pytest.param(
        {
            "type": "file_search_call",
            "id": "fs_replayed",
            "queries": [_REPLAYED_DETAIL],
            "status": "completed",
            "results": [
                {
                    "file_id": "file-replayed",
                    "filename": "replayed.txt",
                    "score": 0.5,
                    "text": _REPLAYED_DETAIL,
                    "attributes": {"origin": "replay"},
                }
            ],
        },
        "results.0.text",
        _REPLAYED_DETAIL,
        id="file_search_call_results",
    ),
    pytest.param(
        {
            "type": "code_interpreter_call",
            "id": "ci_replayed",
            "code": "print('replayed')",
            "container_id": "cntr_replayed",
            "status": "completed",
            "outputs": [{"type": "logs", "logs": _REPLAYED_DETAIL}],
        },
        "outputs.0.logs",
        _REPLAYED_DETAIL,
        id="code_interpreter_call_outputs",
    ),
    pytest.param(
        {
            "type": "shell_call",
            "id": "sh_replayed",
            "call_id": "call_replayed_shell",
            "action": {"commands": ["ls"]},
            "status": "completed",
            "environment": {
                "type": "local",
                "skills": [
                    {
                        "name": "round_trip",
                        "description": _REPLAYED_DETAIL,
                        "path": "/tmp/round_trip",  # noqa: S108
                    }
                ],
            },
        },
        "environment.skills.0.description",
        _REPLAYED_DETAIL,
        id="shell_call_environment",
    ),
]


#: One account-wide store and cursor reads spanning requests: pin to one worker.
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

            # Sent whole: a merging and a replacing target then agree.
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

    def test_adding_a_reference_to_an_item_already_held_is_refused(
        self, openai_client: OpenAI
    ) -> None:
        """An ``item_reference`` to an item already here answers a typed 400.

        An ``item_reference`` brings an item that exists elsewhere into the
        conversation, so one naming an item it already holds is refused. The
        error code carries the whole meaning for a client, which is why it is
        asserted rather than the message.

        Ref: https://developers.openai.com/api/reference/resources/conversations.md
             stdapi/routes/openai_conversations.py:_new_items
        """
        conversation = openai_client.conversations.create(
            items=[{"role": "user", "content": "line 0"}]
        )
        try:
            listed = openai_client.conversations.items.list(conversation.id, limit=1)
            item_id = _item_id(listed.data[0])

            with pytest.raises(BadRequestError) as refused:
                openai_client.conversations.items.create(
                    conversation.id, items=[{"type": "item_reference", "id": item_id}]
                )

            assert refused.value.status_code == 400
            assert refused.value.code == "item_already_in_conversation"
            assert refused.value.param == "items"
        finally:
            with contextlib.suppress(Exception):
                openai_client.conversations.delete(conversation.id)

    def test_an_additional_tools_item_round_trips(self, openai_client: OpenAI) -> None:
        """An ``additional_tools`` item survives a write and reads back whole.

        The item union a listing validates against is wider than the one a
        stub exercises, and a written item cannot be taken back, so the only
        proof that a stored item is readable is storing one and reading it.

        Ref: https://developers.openai.com/api/reference/resources/conversations.md
             openai.types.conversations.conversation_item.ConversationItem
        """
        conversation = openai_client.conversations.create()
        try:
            added = openai_client.conversations.items.create(
                conversation.id,
                items=[_ADDITIONAL_TOOLS_ITEM],  # type: ignore[list-item]
            )
            assert [item.type for item in added.data] == ["additional_tools"]

            listed = openai_client.conversations.items.list(conversation.id, limit=10)

            (stored,) = [
                item for item in listed.data if item.type == "additional_tools"
            ]
            assert stored.role == "developer"
            assert [tool.name for tool in stored.tools] == ["get_weather"]  # type: ignore[union-attr]
            retrieved = openai_client.conversations.items.retrieve(
                _item_id(stored), conversation_id=conversation.id
            )
            assert retrieved.type == "additional_tools"
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

    @staticmethod
    def _write_and_read_back(
        client: OpenAI, conversation_id: str, item: dict[str, Any], item_type: str
    ) -> tuple[ConversationItem, ConversationItem]:
        """Write one item to a conversation and read it back both ways.

        Args:
            client: The OpenAI SDK client bound to the selected target.
            conversation_id: The conversation to write into.
            item: The item to write, in its request shape.
            item_type: The ``type`` the stored item is expected to carry.

        Returns:
            The item as listed and as retrieved, in that order.
        """
        added = client.conversations.items.create(
            conversation_id,
            items=[item],  # type: ignore[list-item]
        )
        assert [stored.type for stored in added.data] == [item_type]

        listed = client.conversations.items.list(conversation_id, order="asc", limit=10)
        (stored,) = [entry for entry in listed.data if entry.type == item_type]
        retrieved = client.conversations.items.retrieve(
            _item_id(stored), conversation_id=conversation_id
        )
        assert retrieved.id == stored.id
        assert retrieved.type == item_type
        return stored, retrieved

    @pytest.mark.parametrize(("item", "path", "expected"), _REPLAYED_TOOL_CALL_ITEMS)
    def test_a_replayed_tool_call_item_round_trips(
        self,
        openai_client: OpenAI,
        item: dict[str, Any],
        path: str,
        expected: Any,  # noqa: ANN401
    ) -> None:
        """A tool-call item a client replays reads back with its detail intact.

        A client rebuilding a transcript posts the tool calls of the previous
        turn, so a listing union narrower than the write side would hand the
        conversation back missing them -- which no stub can catch, because the
        item has to reach real storage to be dropped by it.

        These four are the tool-call shapes the official API genuinely accepts
        on a write. It refuses the rest of the family outright: a ``reasoning``
        item answers 400 ``invalid_item`` in every form (``content`` is capped at
        length zero), and so does any item carrying an action, an output, an
        argument or a result the strict shapes cannot express -- an unknown
        ``computer_call`` or ``web_search_call`` action, a ``code_interpreter_call``
        output type outside ``logs``/``image``, ``tool_search_call`` arguments that
        are not an object, and a file-search result with an undeclared key.

        Ref: https://developers.openai.com/api/reference/resources/conversations.md
             openai.types.conversations.conversation_item.ConversationItem
             stdapi/types/openai_responses.py:ResponseItem
        """
        with _live_conversation(openai_client) as conversation:
            stored, retrieved = self._write_and_read_back(
                openai_client, conversation.id, item, str(item["type"])
            )

            assert _field(stored, path) == expected
            assert _field(retrieved, path) == expected

    @pytest.mark.parametrize(("item", "path", "expected"), _DISCARDED_TOOL_CALL_DETAILS)
    def test_a_replayed_tool_call_detail_reads_back_per_target(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        item: dict[str, Any],
        path: str,
        expected: Any,  # noqa: ANN401
    ) -> None:
        """A replayed file-search, code-interpreter or shell detail is target-specific.

        The three items are accepted on a write by both targets, and the two
        disagree on the detail: the official API stores ``results``, ``outputs``
        and ``environment`` as ``null`` and never hands them back, while this
        gateway keeps what was written and lists it. Keeping is the friendlier
        answer and the one the read union was widened for, but it is a
        divergence from the API being mirrored, so both halves are asserted
        rather than either being accepted quietly.

        The shell call goes one step further: upstream's own stored-item model
        does not declare ``skills`` on a local environment, so what the gateway
        returns survives the SDK only as an undeclared extra.

        Ref: https://developers.openai.com/api/reference/resources/conversations.md
             stdapi/types/openai_responses.py:ResponseItem
        """
        container = path.split(".", 1)[0]
        with _live_conversation(openai_client) as conversation:
            stored, retrieved = self._write_and_read_back(
                openai_client, conversation.id, item, str(item["type"])
            )

            if use_official_api:
                assert _field(stored, container) is None
                assert _field(retrieved, container) is None
            else:
                assert _field(stored, path) == expected
                assert _field(retrieved, path) == expected

    def test_an_assistant_message_phase_is_accepted_and_dropped(
        self, openai_client: OpenAI
    ) -> None:
        """An assistant message's ``phase`` is written, stored without it, and read back.

        ``EasyInputMessageParam`` types ``phase`` in the official client and the
        reference tells a client to "preserve and resend phase on all assistant
        messages", so a client that replays a transcript sends it. The API takes
        the write and keeps the message without it, which is what is mirrored
        here: the write succeeds, the message survives in full, and ``phase``
        comes back absent on both the listing and the retrieval.

        Ref: https://developers.openai.com/api/reference/resources/conversations.md
             openai.types.conversations.message.Message
             stdapi/types/openai_responses.py:EasyInputMessage
        """
        item = {
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": _REPLAYED_DETAIL,
        }
        with _live_conversation(openai_client) as conversation:
            stored, retrieved = self._write_and_read_back(
                openai_client, conversation.id, item, "message"
            )

            assert _item_text(stored) == _REPLAYED_DETAIL
            assert _field(stored, "phase") is None
            assert _field(retrieved, "phase") is None

    @pytest.mark.parametrize("role", ["user", "system", "developer"])
    def test_a_non_assistant_message_phase_is_refused(
        self, openai_client: OpenAI, role: str
    ) -> None:
        """A ``phase`` under any role but ``assistant`` is refused as an unknown parameter.

        ``phase`` is typed on the official client's input message independently
        of ``role``, and the API accepts it under one role only: every other one
        answers 400 ``unknown_parameter`` naming ``items[<n>].phase``, whatever
        ``type`` the item declares. The client type being wider than the server
        is the trap here -- the server is what this mirrors, so the same refusal
        has to come back from both targets, down to the code and the parameter.

        Ref: https://developers.openai.com/api/reference/resources/conversations.md
             stdapi/types/openai_responses.py:reject_input_message_phase
        """
        item = {
            "type": "message",
            "role": role,
            "phase": "commentary",
            "content": _REPLAYED_DETAIL,
        }
        with _live_conversation(openai_client) as conversation:
            with pytest.raises(BadRequestError) as refused:
                openai_client.conversations.items.create(
                    conversation.id,
                    items=[item],  # type: ignore[list-item]
                )

            assert refused.value.status_code == 400
            assert refused.value.type == "invalid_request_error"
            assert refused.value.code == "unknown_parameter"
            assert refused.value.param == "items[0].phase"
            assert openai_client.conversations.items.list(conversation.id).data == []
