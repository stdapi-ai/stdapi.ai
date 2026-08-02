"""Tests for the OpenAI-compatible POST /v1/responses/compact route.

The gateway rejects ``context_management``, so the standalone compact endpoint is
the only compaction path available.  It runs a summarisation turn on Bedrock and
returns the conversation's user messages followed by one opaque ``compaction``
item; the item content is self-contained (marker-prefixed base64url, not
encrypted), so replaying it needs no server-side state.

Ref: https://developers.openai.com/api/reference/resources/responses/methods/compact
     https://developers.openai.com/api/docs/guides/compaction
     stdapi/routes/openai_responses.py:compact_response
     stdapi/types/openai_responses.py:CompactedResponse
"""

from base64 import urlsafe_b64decode, urlsafe_b64encode
from typing import TYPE_CHECKING, Any, Literal

import pytest
from openai._models import construct_type
from openai.types.responses import CompactedResponse as SdkCompactedResponse

from stdapi.api_errors import ApiError
from stdapi.models.chat._adapters._openai_responses import (
    COMPACTION_CONTENT_PREFIX,
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
    ResponseError,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from tests._helpers import make_model_details

if TYPE_CHECKING:
    from openai import OpenAI
    from starlette.testclient import TestClient

    from stdapi.models import ModelDetails


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
        self.usage: ResponseUsage | None = _usage()
        #: Terminal state overrides applied to the canned response, when set.
        self.status: Literal["failed"] | None = None
        self.error: ResponseError | None = None

    async def create_response(
        self, request: ResponseCreateParams, response_id: str, created_at: float
    ) -> Response:
        """Record the request and return a canned summary response."""
        self.requests.append(request)
        response = self._response(request, response_id, created_at)
        if self.status is not None:
            response.status = self.status
            response.error = self.error
        return response

    def _response(
        self, request: ResponseCreateParams, response_id: str, created_at: float
    ) -> Response:
        """Build the canned summary response."""
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
            usage=self.usage,
        )


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


@pytest.mark.local
class TestResponsesCompactRoute:
    """POST /v1/responses/compact: response shape and generation request.

    Ref: stdapi/routes/openai_responses.py:_compaction_user_messages
    """

    @pytest.mark.usefixtures("chat_backend")
    def test_compact_returns_compaction_item(self, app_client: TestClient) -> None:
        """String input yields a user message echo followed by the compaction item.

        A bare string input is the shorthand for a single ``user`` message, so
        it is echoed as an ``input_text`` part, and the summary the backend
        produced is what the compaction item carries.
        """
        response = app_client.post(
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
        assert item["encrypted_content"].startswith(COMPACTION_CONTENT_PREFIX)
        encoded = item["encrypted_content"].removeprefix(COMPACTION_CONTENT_PREFIX)
        assert urlsafe_b64decode(encoded) == b"THE SUMMARY"
        assert body["id"] == item["id"].replace("ci-", "resp-", 1)
        assert body["usage"]["total_tokens"] == 18, (
            "the summarisation turn's usage is billed to the caller"
        )

    def test_failed_generation_is_a_502_not_an_empty_compaction(
        self, app_client: TestClient, chat_backend: _StubChatModel
    ) -> None:
        """A failed summarisation run surfaces its error instead of a 200.

        Wrapping a failed run in a 200 would hand the client an empty summary
        that silently replaces its conversation; the guard mirrors the one on
        POST /v1/responses.

        Ref: stdapi/routes/openai_responses.py:_failed_response_error
        """
        chat_backend.status = "failed"
        chat_backend.error = ResponseError(
            code="server_error",
            message="The model failed to generate a valid response.",
        )
        response = app_client.post(
            "/v1/responses/compact",
            json={"model": "amazon.nova-pro-v1:0", "input": "a long conversation"},
        )
        assert response.status_code == 502, response.text
        assert (
            response.json()["error"]["message"]
            == "The model failed to generate a valid response."
        )

    @pytest.mark.usefixtures("chat_backend")
    def test_compact_echoes_only_user_messages_in_order(
        self, app_client: TestClient
    ) -> None:
        """Assistant messages are dropped; user echoes stay ordered before the compaction item.

        The echoed window is what callers replay, and the assistant turns are
        already folded into the summary, so re-sending them would duplicate
        context.
        """
        response = app_client.post(
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

    @pytest.mark.usefixtures("chat_backend")
    def test_compact_echoes_part_list_content_as_dicts(
        self, app_client: TestClient
    ) -> None:
        """List-based user content parts are echoed back verbatim as dicts.

        The echo is a replayable window, so image and text parts must survive
        the round trip unchanged instead of being flattened to text.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        parts = [
            {"type": "input_text", "text": "look at this"},
            {"type": "input_image", "image_url": "https://example.com/img.png"},
        ]
        response = app_client.post(
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
        self, app_client: TestClient, chat_backend: _StubChatModel
    ) -> None:
        """The generation request keeps the input and appends the directive.

        Compaction is a normal generation turn, so the conversation is sent
        unchanged with the summarisation instruction as a trailing user
        message, and the caller's own ``instructions`` are preserved.
        """
        app_client.post(
            "/v1/responses/compact",
            json={
                "model": "amazon.nova-pro-v1:0",
                "input": [{"role": "user", "content": "hello"}],
                "instructions": "be nice",
            },
        )
        (request,) = chat_backend.requests
        assert request.model == "amazon.nova-pro-v1:0"
        assert request.instructions == "be nice"
        assert isinstance(request.input, list)
        assert len(request.input) == 2, "the input plus the appended directive"
        first = request.input[0]
        assert isinstance(first, EasyInputMessage | InputMessage)
        assert first.content == "hello"
        last = request.input[-1]
        assert isinstance(last, EasyInputMessage | InputMessage)
        assert last.role == "user"
        assert "Summarize the conversation" in str(last.content)

    def test_previous_response_id_compacts_stored_conversation(
        self,
        app_client: TestClient,
        chat_backend: _StubChatModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stored conversation is prepended before the compaction directive.

        ``previous_response_id`` resolves against the local Bedrock-session
        store, whose items are merged ahead of the new input so the summary
        covers the whole thread; the echoed window covers it too.

        Ref: stdapi/routes/openai_responses.py:_apply_previous_response
             stdapi/responses_store.py:load_stored_response
        """

        async def _load(response_id: str, kind: str) -> dict[str, Any]:
            assert response_id == "resp-1"
            assert kind == "response"
            return {"input": [{"role": "user", "content": "first"}], "response": {}}

        monkeypatch.setattr(openai_responses, "load_stored_response", _load)
        response = app_client.post(
            "/v1/responses/compact",
            json={"model": "m", "input": "second", "previous_response_id": "resp-1"},
        )
        assert response.status_code == 200, response.text
        (request,) = chat_backend.requests
        # Restored after the merge, though CompactedResponse has no field to echo it.
        assert request.previous_response_id == "resp-1"
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
        stored_echo = echoes[0]
        assert stored_echo["type"] == "message"
        assert stored_echo["role"] == "user"
        assert item["type"] == "compaction"

    @pytest.mark.parametrize("body_input", [None, []], ids=["omitted", "empty-list"])
    def test_empty_input_is_rejected(
        self,
        app_client: TestClient,
        chat_backend: _StubChatModel,
        body_input: list[Any] | None,
    ) -> None:
        """An empty conversation is rejected with 400 before any generation.

        ``input`` is optional on the compact route (a stored conversation may
        supply it), so an omitted and an empty list both have to be caught
        explicitly rather than by schema validation.
        """
        body: dict[str, Any] = {"model": "amazon.nova-pro-v1:0"}
        if body_input is not None:
            body["input"] = body_input
        response = app_client.post("/v1/responses/compact", json=body)
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert "no conversation to compact" in error["message"]
        assert error["type"] == "invalid_request_error"
        assert not chat_backend.requests, "no model is billed for an empty request"

    def test_generation_parameters_are_forwarded(
        self, app_client: TestClient, chat_backend: _StubChatModel
    ) -> None:
        """Caching and service-tier parameters reach the generation request.

        These are the compact-route parameters with no equivalent on
        ``CompactedResponse``, so the only observable effect is that they are
        forwarded to the summarisation turn.

        Ref: stdapi/types/openai_responses.py:CompactParams
        """
        response = app_client.post(
            "/v1/responses/compact",
            json={
                "model": "amazon.nova-pro-v1:0",
                "input": "x",
                "service_tier": "flex",
                "prompt_cache_key": "cache-key",
                "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
                "prompt_cache_retention": "24h",
            },
        )
        assert response.status_code == 200, response.text
        (request,) = chat_backend.requests
        assert request.service_tier == "flex"
        assert request.prompt_cache_key == "cache-key"
        assert request.prompt_cache_retention == "24h"
        assert request.prompt_cache_options is not None
        assert request.prompt_cache_options.mode == "explicit"
        assert request.prompt_cache_options.ttl == "30m"

    def test_missing_usage_falls_back_to_zeros(
        self, app_client: TestClient, chat_backend: _StubChatModel
    ) -> None:
        """A backend response without usage yields a zeroed usage envelope.

        ``usage`` is required on ``CompactedResponse``, so a backend that
        reports none must still serialise a complete envelope.
        """
        chat_backend.usage = None
        response = app_client.post(
            "/v1/responses/compact",
            json={"model": "amazon.nova-pro-v1:0", "input": "x"},
        )
        assert response.status_code == 200, response.text
        usage = response.json()["usage"]
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0
        assert usage["total_tokens"] == 0
        assert response.json()["output"][-1]["type"] == "compaction"

    def test_mantle_previous_response_id_is_not_found(
        self,
        app_client: TestClient,
        chat_backend: _StubChatModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A Mantle-stored previous_response_id is rejected with 404.

        Compaction always runs locally, and a Mantle conversation is only
        readable through Mantle's own native chaining, so the region-tagged
        ``resp_`` ID cannot be continued here.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
             stdapi/routes/openai_responses.py:_decode_mantle_id
        """
        monkeypatch.setattr(
            openai_responses,
            "_decode_mantle_id",
            lambda _response_id: ("us-east-1", "resp_native123"),
        )
        response = app_client.post(
            "/v1/responses/compact",
            json={
                "model": "amazon.nova-pro-v1:0",
                "input": "x",
                "previous_response_id": "resp_native123",
            },
        )
        assert response.status_code == 404, response.text
        error = response.json()["error"]
        assert "cannot be continued with this model" in error["message"]
        assert error["type"] == "invalid_request_error"
        assert not chat_backend.requests

    @pytest.mark.usefixtures("chat_backend")
    def test_sdk_parses_compaction_envelope(self, app_client: TestClient) -> None:
        """The response JSON validates against the SDK's ``CompactedResponse`` model.

        Clients call this route through ``openai.responses.compact()``, so the
        echoed user messages must be parseable alongside the compaction item.

        Ref: https://github.com/openai/openai-python/tree/main/src/openai/types/responses
        """
        response = app_client.post(
            "/v1/responses/compact",
            json={
                "model": "amazon.nova-pro-v1:0",
                "input": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 200, response.text
        parsed = construct_type(type_=SdkCompactedResponse, value=response.json())
        assert isinstance(parsed, SdkCompactedResponse)
        assert [part.type for part in parsed.output] == ["message", "compaction"]
        item = next(part for part in parsed.output if part.type == "compaction")
        assert item.id
        assert item.encrypted_content.startswith(COMPACTION_CONTENT_PREFIX)
        assert parsed.object == "response.compaction"
        assert parsed.usage.total_tokens == 18

    def test_unknown_previous_response_id_is_not_found(
        self,
        app_client: TestClient,
        chat_backend: _StubChatModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unknown previous_response_id surfaces as 404 with the requested ID.

        Ref: stdapi/routes/openai_responses.py:_previous_response_not_found
        """

        async def _load(response_id: str, kind: str) -> dict[str, Any]:  # noqa: ARG001
            from stdapi.api_errors import ApiError  # noqa: PLC0415

            msg = f"Response with id '{response_id}' not found."
            raise ApiError(msg, status=404)

        monkeypatch.setattr(openai_responses, "load_stored_response", _load)
        response = app_client.post(
            "/v1/responses/compact",
            json={"model": "m", "input": "x", "previous_response_id": "resp-zzz"},
        )
        assert response.status_code == 404, response.text
        error = response.json()["error"]
        assert "resp-zzz" in error["message"]
        assert "not found" in error["message"]
        assert not chat_backend.requests


@pytest.mark.local
class TestCompactionItemRoundTrip:
    """Compaction items round-trip through the Responses input mapping.

    Ref: https://developers.openai.com/api/docs/guides/compaction
         stdapi/models/chat/_adapters/_openai_responses.py:encode_compaction_content
         stdapi/models/chat/_adapters/_openai_responses.py:_map_compaction_item
    """

    async def test_compaction_item_maps_to_user_summary_message(self) -> None:
        """The decoded summary is injected as a labelled user message.

        The summary's UTF-8 bytes base64-encode to characters that differ
        between the standard and urlsafe alphabets ('-'/'_' vs '+'/'/'),
        catching an encode/decode alphabet mismatch.
        """
        summary = "Summary: \xff\xff\xff details preserved."
        encrypted_content = encode_compaction_content(summary)
        assert encrypted_content.startswith(COMPACTION_CONTENT_PREFIX)
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
        assert message["content"] == [
            {"text": f"Summary of the earlier conversation:\n{summary}"}
        ]

    async def test_invalid_compaction_content_is_rejected(self) -> None:
        """Undecodable compaction content raises a 400 error.

        ``!!!`` carries the local marker but is not valid base64url, so the
        decode failure must surface as a client error rather than a 500.
        """
        item = CompactionItemParam(
            encrypted_content=f"{COMPACTION_CONTENT_PREFIX}!!!", type="compaction"
        )
        with pytest.raises(ApiError, match="produced by this server") as excinfo:
            await map_input([item], None)
        assert excinfo.value.status == 400
        assert "Invalid compaction item content" in str(excinfo.value)

    async def test_unmarked_content_is_rejected(self) -> None:
        """Content without the local marker is rejected even when decodable.

        Upstream ciphertext that happens to be valid base64url of UTF-8 text
        must not be silently injected as a summary.
        """
        item = CompactionItemParam(
            encrypted_content=urlsafe_b64encode(b"upstream ciphertext").decode(),
            type="compaction",
        )
        with pytest.raises(ApiError, match="produced by this server") as excinfo:
            await map_input([item], None)
        assert excinfo.value.status == 400
        assert "Invalid compaction item content" in str(excinfo.value)


class TestResponsesCompactLive:
    """Live conversation compaction and round-trip continuation.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/compact
         https://developers.openai.com/api/docs/guides/compaction
    """

    @pytest.mark.retry(
        "What survives compaction is the model's own summarization choice: it "
        "may drop the fact the continuation is asked to recall"
    )
    def test_compact_and_continue(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A compacted conversation carries its facts into the next turn.

        The compaction item is self-contained, so replaying it with ``store``
        disabled is enough for the model to recover the earlier fact.
        """
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
        assert follow.status == "completed", follow.model_dump_json()
        assert "teal" in follow.output_text.lower()
