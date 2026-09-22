"""POST /v1/responses context-window handling, against a scripted Converse backend.

The real Converse model class and Responses adapter run; only the Bedrock calls
are replaced, by a backend refusing every request above a size, so retries,
compaction passes and billed calls can be counted exactly.

Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
     https://developers.openai.com/api/docs/guides/compaction
     stdapi/routes/_responses_context.py
"""

import asyncio
from base64 import urlsafe_b64decode
from itertools import pairwise
from json import dumps, loads
from typing import TYPE_CHECKING, Any

import pytest
from sse_starlette import EventSourceResponse, ServerSentEvent

from stdapi.api_errors import ApiError
from stdapi.models.chat._adapters._openai_responses import (
    encode_compaction_content,
    encode_compaction_state,
    merge_usage,
)
from stdapi.models.chat._default import ChatModel
from stdapi.models.chat._mantle._convert import _reject_local_compaction_items
from stdapi.routes import _responses_context, openai_responses
from stdapi.types.openai_responses import (
    CompactionItemParam,
    EasyInputMessage,
    Response,
    ResponseCreateParams,
    ResponseOutputMessage,
    ResponseOutputText,
)
from tests._helpers import make_client_error, make_model_details

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from starlette.testclient import TestClient

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef
    from stdapi.models import ModelDetails

pytestmark = pytest.mark.local

#: Nova Micro's refusal of an input over its window, as recorded.
_NOVA_OVERFLOW = (
    "The model returned the following errors: Input Tokens Exceeded: Number of "
    "input tokens exceeds maximum length. Please update the input to try again."
)

#: The token counter's refusal of a model it cannot count, as recorded.
_UNCOUNTABLE = "The provided model doesn't support counting tokens."

#: Directive text the compaction pass appends.
_DIRECTIVE = "Summarize the conversation above"


def _text_of(request: ConverseRequestBaseTypeDef) -> str:
    """Return every text a Converse request carries, joined.

    Args:
        request: The request.

    Returns:
        The text.
    """
    return " ".join(
        block["text"]
        for message in request["messages"]
        for block in message["content"]
        if "text" in block
    )


def _overflow(code: str) -> Exception:
    """Return Nova Micro's refusal of an input over its window.

    Args:
        code: The error code, capitalized off-stream and not in-stream.

    Returns:
        The error.
    """
    return make_client_error(code, "Converse", message=_NOVA_OVERFLOW)


class _Backend:
    """Scripted Converse backend refusing any request over ``limit`` characters."""

    def __init__(self) -> None:
        #: Requests received, in order.
        self.requests: list[ConverseRequestBaseTypeDef] = []
        #: Largest request text accepted.
        self.limit = 1_000_000
        #: Whether a streamed refusal comes on the first event instead of at open.
        self.refuse_in_stream = False
        #: Token counts the counter returns, or the error it raises.
        self.count: int | Exception = make_client_error(
            "ValidationException", "CountTokens", message=_UNCOUNTABLE
        )
        #: Calls the token counter received.
        self.counted = 0

    def _check(self, request: ConverseRequestBaseTypeDef) -> None:
        self.requests.append(request)
        if len(_text_of(request)) > self.limit:
            code = "ValidationException"
            raise _overflow(code)

    @staticmethod
    def _answer(request: ConverseRequestBaseTypeDef) -> str:
        return "SUMMARY" if _DIRECTIVE in _text_of(request) else "ANSWER"

    async def converse(self, request: ConverseRequestBaseTypeDef) -> dict[str, Any]:
        self._check(request)
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": self._answer(request)}],
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": len(self.requests) * 100,
                "outputTokens": 10,
                "totalTokens": len(self.requests) * 100 + 10,
            },
        }

    async def converse_stream(
        self, request: ConverseRequestBaseTypeDef
    ) -> dict[str, Any]:
        in_stream = self.refuse_in_stream
        if not in_stream:
            self._check(request)
        else:
            self.requests.append(request)
        refused = in_stream and len(_text_of(request)) > self.limit
        answer = self._answer(request)

        async def _events() -> AsyncGenerator[dict[str, Any]]:
            if refused:
                code = "validationException"
                raise _overflow(code)
            for event in (
                {"messageStart": {"role": "assistant"}},
                {"contentBlockStart": {"start": {}, "contentBlockIndex": 0}},
                {
                    "contentBlockDelta": {
                        "delta": {"text": answer},
                        "contentBlockIndex": 0,
                    }
                },
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "end_turn"}},
                {"metadata": {"usage": {"inputTokens": 7, "outputTokens": 3}}},
            ):
                yield event

        return {"stream": _events()}

    async def count_tokens(self, *_args: object, **_kwargs: object) -> int:
        self.counted += 1
        if isinstance(self.count, Exception):
            raise self.count
        return self.count


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> _Backend:
    """Serve the Responses route from the real Converse model over a scripted backend."""

    async def _validate_model(
        model_id: str, *_args: object, **_kwargs: object
    ) -> ModelDetails:
        return make_model_details(model_id)

    stub = _Backend()
    monkeypatch.setattr(openai_responses, "validate_model", _validate_model)
    monkeypatch.setattr(openai_responses, "get_chat_model", ChatModel)
    monkeypatch.setattr(ChatModel, "converse", stub.converse)
    monkeypatch.setattr(ChatModel, "converse_stream", stub.converse_stream)
    monkeypatch.setattr(
        _responses_context, "count_input_tokens_via_bedrock", stub.count_tokens
    )
    monkeypatch.setattr(_responses_context, "_UNCOUNTABLE_MODELS", set())
    return stub


def _turns(count: int, size: int = 200) -> list[dict[str, Any]]:
    """Build a conversation of *count* sized turns, then a question.

    Args:
        count: Number of turns.
        size: Characters of each turn's user message.

    Returns:
        The input items.
    """
    items: list[dict[str, Any]] = []
    for index in range(count):
        items.append({"role": "user", "content": f"u{index} " + "x" * size})
        items.append({"role": "assistant", "content": f"a{index}"})
    items.append({"role": "user", "content": "question"})
    return items


def _post(client: TestClient, **body: Any) -> Any:  # noqa: ANN401
    """Create a response.

    Args:
        client: The app client.
        **body: Request body.

    Returns:
        The HTTP response.
    """
    return client.post(
        "/v1/responses", json={"model": "amazon.nova-micro-v1:0", **body}
    )


def _events(response: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """Decode a streamed response's event payloads.

    Args:
        response: The HTTP response.

    Returns:
        The payloads, in order.
    """
    return [
        loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


class TestTruncationRetries:
    """``truncation: "auto"`` retries a refused input, at most three times.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/routes/_responses_context.py:truncating
    """

    def test_disabled_answers_an_overflow_with_the_upstream_error(
        self, app_client: TestClient, backend: _Backend
    ) -> None:
        """Without truncation, the refusal is upstream's 400, the backend text hidden.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """
        backend.limit = 0
        response = _post(app_client, input=_turns(4))
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["code"] == "context_length_exceeded"
        assert error["param"] == "input"
        assert "Input Tokens Exceeded" not in error["message"]
        assert len(backend.requests) == 1

    def test_retries_are_bounded(
        self, app_client: TestClient, backend: _Backend
    ) -> None:
        """An input that never fits costs four calls, then the upstream error.

        Ref: stdapi/routes/_responses_context.py:truncating
        """
        backend.limit = 0
        response = _post(app_client, input=_turns(40), truncation="auto")
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "context_length_exceeded"
        assert len(backend.requests) == 4

    def test_the_oldest_turns_are_dropped_until_it_fits(
        self, app_client: TestClient, backend: _Backend
    ) -> None:
        """The retry sends the latest turns only, and is answered.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:truncate_input
        """
        backend.limit = 1000
        response = _post(app_client, input=_turns(8), truncation="auto")
        assert response.status_code == 200, response.text
        assert response.json()["truncation"] == "auto"
        assert len(backend.requests) >= 2
        kept = _text_of(backend.requests[-1])
        assert "u0 " not in kept
        assert "u7 " in kept
        assert "question" in kept

    def test_a_stream_refused_on_its_first_event_is_retried(
        self, app_client: TestClient, backend: _Backend
    ) -> None:
        """A refusal arriving as the stream's first event is retried before it starts.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/routes/_responses_context.py:_open_once
        """
        backend.limit = 1000
        backend.refuse_in_stream = True
        response = _post(app_client, input=_turns(8), truncation="auto", stream=True)
        assert response.status_code == 200
        events = _events(response)
        assert [event["type"] for event in events[:2]] == [
            "response.created",
            "response.in_progress",
        ]
        assert events[-1]["type"] == "response.completed"
        assert "error" not in [event["type"] for event in events]
        assert len(backend.requests) >= 2

    def test_a_stream_refused_at_open_streams_the_refusal(
        self, app_client: TestClient, backend: _Backend
    ) -> None:
        """A refusal before the stream opens still reaches the client as events.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
        """
        backend.limit = 0
        response = _post(app_client, input=_turns(2), stream=True)
        assert response.status_code == 200
        events = _events(response)
        assert [event["type"] for event in events] == [
            "response.created",
            "response.in_progress",
            "error",
            "response.failed",
        ]
        assert events[2]["code"] == "context_length_exceeded"

    def test_input_token_count_retries_the_same_way(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Counting under ``truncation: "auto"`` counts the input a response keeps.

        Ref: https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens/methods/count
        """
        counted: list[int] = []
        too_long = make_client_error(
            "ValidationException",
            "CountTokens",
            message="prompt is too long: 250000 tokens > 200000 maximum",
        )

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> ModelDetails:
            return make_model_details(model_id)

        async def _count(request: Any, *_args: object, **_kwargs: object) -> int:  # noqa: ANN401
            counted.append(len(request.input))
            if len(request.input) > 5:
                raise too_long
            return 42

        monkeypatch.setattr(openai_responses, "validate_model", _validate_model)
        monkeypatch.setattr(openai_responses, "count_input_tokens_via_bedrock", _count)
        monkeypatch.setattr(openai_responses, "serves_via_mantle", lambda _id: False)
        body = {"model": "anthropic.claude-haiku-4-5", "input": _turns(8)}
        response = app_client.post(
            "/v1/responses/input_tokens", json={**body, "truncation": "auto"}
        )
        assert response.status_code == 200, response.text
        assert response.json()["input_tokens"] == 42
        assert counted[0] == 17
        assert counted[-1] <= 5
        refused = app_client.post("/v1/responses/input_tokens", json=body)
        assert refused.status_code == 400
        assert refused.json()["error"]["code"] == "context_length_exceeded"


class TestCompactionPass:
    """``context_management`` compacts before generating, billing both calls.

    Ref: https://developers.openai.com/api/docs/guides/compaction
         stdapi/routes/_responses_context.py:generate
    """

    def test_above_the_threshold_the_answer_follows_a_compaction(
        self, app_client: TestClient, backend: _Backend
    ) -> None:
        """The summary call runs first, and the answer sees only its result.

        The response's usage adds both calls up.

        Ref: https://developers.openai.com/api/docs/guides/compaction
        """
        backend.count = 5000
        response = _post(
            app_client,
            input=_turns(3),
            context_management=[{"type": "compaction", "compact_threshold": 3000}],
        )
        assert response.status_code == 200, response.text
        summary_call, answer_call = backend.requests
        assert _DIRECTIVE in _text_of(summary_call)
        assert "u0 " in _text_of(summary_call)
        answered = _text_of(answer_call)
        assert "u0 " not in answered
        assert "SUMMARY" in answered
        assert "question" in answered
        body = response.json()
        compaction = body["output"][0]
        assert compaction["type"] == "compaction"
        assert compaction["id"].startswith("cmp_")
        assert body["output"][1]["type"] == "message"
        assert body["usage"]["input_tokens"] == 100 + 200
        assert body["usage"]["total_tokens"] == 110 + 210

    def test_below_the_counted_threshold_nothing_is_compacted(
        self, app_client: TestClient, backend: _Backend
    ) -> None:
        """A counted input under the threshold is answered in one call.

        Ref: stdapi/routes/_responses_context.py:plan_compaction
        """
        backend.count = 100
        response = _post(
            app_client,
            input=_turns(3, size=50_000),
            context_management=[{"type": "compaction", "compact_threshold": 3000}],
        )
        assert response.status_code == 200, response.text
        assert len(backend.requests) == 1
        assert backend.counted == 1

    def test_an_uncountable_model_is_estimated_and_remembered(
        self, app_client: TestClient, backend: _Backend
    ) -> None:
        """A model the counter refuses is estimated, and never counted again.

        Ref: stdapi/routes/_responses_context.py:_counted_tokens
        """
        body = {
            "input": _turns(3, size=30_000),
            "context_management": [{"type": "compaction", "compact_threshold": 3000}],
        }
        assert _post(app_client, **body).status_code == 200
        assert _post(app_client, **body).status_code == 200
        assert backend.counted == 1
        assert len(backend.requests) == 4, "each request compacts, then answers"

    def test_the_estimate_counts_text_only(
        self, app_client: TestClient, backend: _Backend
    ) -> None:
        """A short conversation stays under the threshold, whatever its images weigh.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:estimate_tokens
        """
        image = "data:image/png;base64," + "A" * 40_000
        response = _post(
            app_client,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "look"},
                        {"type": "input_image", "image_url": image},
                    ],
                },
                {"role": "assistant", "content": "seen"},
                {"role": "user", "content": "question"},
            ],
            context_management=[{"type": "compaction", "compact_threshold": 3000}],
        )
        assert response.status_code == 200, response.text
        assert len(backend.requests) == 1

    def test_a_stored_conversation_is_measured_by_its_recorded_usage(
        self, app_client: TestClient, backend: _Backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A continuation's history weighs what its stored response recorded.

        The stored text is short, but the conversation it stands for took
        50,000 tokens, so the continuation compacts.

        Ref: stdapi/routes/openai_responses.py:_merge_previous_response
        """
        documents: dict[str, dict[str, Any]] = {
            "resp-small": {
                "input": [{"role": "user", "content": "hi"}],
                "response": {"output": []},
            },
            "resp-large": {
                "input": [{"role": "user", "content": "hi"}],
                "response": {"output": []},
                "context_tokens": 50_000,
            },
        }

        async def _load(response_id: str, _kind: str) -> dict[str, Any]:
            return documents[response_id]

        monkeypatch.setattr(openai_responses, "load_stored_response", _load)
        cm = [{"type": "compaction", "compact_threshold": 3000}]
        for previous, calls in (("resp-small", 1), ("resp-large", 2)):
            backend.requests.clear()
            response = _post(
                app_client,
                input="question",
                previous_response_id=previous,
                context_management=cm,
            )
            assert response.status_code == 200, response.text
            assert len(backend.requests) == calls, previous

    def test_a_stored_response_records_the_answers_usage_and_the_compaction(
        self, app_client: TestClient, backend: _Backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The stored input is the compaction item, and its size the answer's usage.

        Ref: stdapi/routes/openai_responses.py:_save_response
        """
        saved: dict[str, Any] = {}

        async def _session(_kind: str) -> str:
            return "sess"

        async def _save(response_id: str, document: dict[str, Any]) -> None:
            saved[response_id] = document

        monkeypatch.setattr(
            openai_responses, "try_create_stored_response_session", _session
        )
        monkeypatch.setattr(openai_responses, "save_stored_response", _save)
        backend.count = 5000
        response = _post(
            app_client,
            input=_turns(3),
            context_management=[{"type": "compaction", "compact_threshold": 3000}],
            store=True,
        )
        assert response.status_code == 200, response.text
        document = saved["resp-sess"]
        assert [item["type"] for item in document["input"]] == ["compaction"]
        assert document["context_tokens"] == 210, "the answer's call alone"

    def test_a_tool_loop_too_large_to_summarize_keeps_its_task(
        self, app_client: TestClient, backend: _Backend
    ) -> None:
        """Trimming a single task's tool loop for its summary keeps the task.

        The conversation has one user turn, so nothing can be dropped whole: its
        largest text is cut instead, and the task stays in what is summarized.

        Ref: stdapi/routes/_responses_context.py:summarize
        """
        backend.limit = 15_000
        backend.count = 5000

        def _step(call_id: str) -> list[dict[str, Any]]:
            return [
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "lookup",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": call_id, "output": "ok"},
            ]

        response = _post(
            app_client,
            instructions="Use the tools.",
            tools=[{"type": "function", "name": "lookup", "parameters": {}}],
            input=[
                {"role": "user", "content": "fix the bug"},
                *_step("c1"),
                {"role": "assistant", "content": "notes " + "x" * 20_000},
                *_step("c2"),
                *_step("c3"),
            ],
            truncation="auto",
            context_management=[{"type": "compaction", "compact_threshold": 3000}],
        )
        assert response.status_code == 200, response.text
        summary_calls = [r for r in backend.requests if _DIRECTIVE in _text_of(r)]
        assert len(summary_calls) == 2, "refused, then cut and summarized"
        summarized = _text_of(summary_calls[-1])
        assert "fix the bug" in summarized
        assert "notes x" in summarized

    def test_the_estimate_reads_the_expanded_input(
        self, app_client: TestClient, backend: _Backend
    ) -> None:
        """Items a compaction item supersedes do not count; what it keeps does.

        Ref: stdapi/routes/_responses_context.py:plan_compaction
        """
        cm = [{"type": "compaction", "compact_threshold": 10_000}]
        light = asyncio.run(encode_compaction_state("S", [], []))
        heavy = asyncio.run(
            encode_compaction_state(
                "S", [], [EasyInputMessage(role="assistant", content="k " * 30_000)]
            )
        )
        question = {"role": "user", "content": "question"}
        superseded = _post(
            app_client,
            input=[
                *_turns(3, size=30_000),
                {"type": "compaction", "encrypted_content": light},
                question,
            ],
            context_management=cm,
        )
        assert superseded.status_code == 200, superseded.text
        assert len(backend.requests) == 1, "the superseded history is not counted"
        backend.requests.clear()
        kept = _post(
            app_client,
            input=[{"type": "compaction", "encrypted_content": heavy}, question],
            context_management=cm,
        )
        assert kept.status_code == 200, kept.text
        assert len(backend.requests) == 2, "the kept items are counted"

    @pytest.mark.parametrize("truncation", ["auto", "disabled"])
    def test_a_conversation_too_large_to_summarize(
        self, app_client: TestClient, backend: _Backend, truncation: str
    ) -> None:
        """An input over the window is trimmed before its summary, or refused.

        The counter refuses the input as too long, so it crosses any threshold,
        and the summary call is refused too until its oldest turns are dropped.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/routes/_responses_context.py:summarize
        """
        backend.limit = 1000
        backend.count = make_client_error(
            "ValidationException",
            "CountTokens",
            message="prompt is too long: 250000 tokens > 200000 maximum",
        )
        response = _post(
            app_client,
            input=_turns(8),
            truncation=truncation,
            context_management=[{"type": "compaction", "compact_threshold": 3000}],
        )
        if truncation == "disabled":
            assert response.status_code == 400, response.text
            error = response.json()["error"]
            assert (error["code"], error["param"]) == (
                "context_length_exceeded",
                "input",
            )
            assert len(backend.requests) == 1
            return
        assert response.status_code == 200, response.text
        assert response.json()["output"][0]["type"] == "compaction"
        summary_calls = [r for r in backend.requests if _DIRECTIVE in _text_of(r)]
        assert len(summary_calls) >= 2, "refused, then trimmed and summarized"
        assert "u0 " not in _text_of(summary_calls[-1])

    def test_a_stored_file_search_turn_records_one_calls_usage(
        self, app_client: TestClient, backend: _Backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """File search rounds resend the conversation; their sum is not its size.

        Ref: stdapi/routes/openai_responses.py:_completed_response
        """
        saved: dict[str, Any] = {}

        async def _session(_kind: str) -> str:
            return "sess"

        async def _save(response_id: str, document: dict[str, Any]) -> None:
            saved[response_id] = document

        async def _two_rounds(response: Response, *_args: object) -> Response:
            assert response.usage is not None
            response.usage = merge_usage(response.usage, response.usage)
            return response

        monkeypatch.setattr(
            openai_responses, "try_create_stored_response_session", _session
        )
        monkeypatch.setattr(openai_responses, "save_stored_response", _save)
        monkeypatch.setattr(openai_responses, "execute_file_search_calls", _two_rounds)
        response = _post(app_client, input="question", store=True)
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["total_tokens"] == 220
        assert saved["resp-sess"]["context_tokens"] == 110

    def test_a_streamed_compaction_leads_the_stream(
        self, app_client: TestClient, backend: _Backend
    ) -> None:
        """Streamed, the compaction item is announced and finished before the answer.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
        """
        backend.count = 5000
        response = _post(
            app_client,
            input=_turns(3),
            context_management=[{"type": "compaction", "compact_threshold": 3000}],
            stream=True,
        )
        events = _events(response)
        types = [event["type"] for event in events]
        assert types[:4] == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.output_item.done",
        ]
        assert types.count("response.created") == 1
        assert events[2]["item"]["type"] == "compaction"
        final = events[-1]["response"]
        assert events[-1]["type"] == "response.completed"
        assert [item["type"] for item in final["output"]] == ["compaction", "message"]
        assert final["usage"]["input_tokens"] == 100 + 7
        assert [event["sequence_number"] for event in events] == list(
            range(len(events))
        )

    def test_a_trigger_compacts_without_answering(
        self, app_client: TestClient, backend: _Backend
    ) -> None:
        """A ``compaction_trigger`` request makes the summary call alone.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        response = _post(
            app_client, input=[*_turns(2)[:-1], {"type": "compaction_trigger"}]
        )
        assert response.status_code == 200, response.text
        (summary_call,) = backend.requests
        assert _DIRECTIVE in _text_of(summary_call)
        body = response.json()
        assert [item["type"] for item in body["output"]] == ["compaction"]
        assert body["usage"]["total_tokens"] == 110

    def test_a_trigger_with_nothing_to_compact_is_refused(
        self, app_client: TestClient, backend: _Backend
    ) -> None:
        """A trigger alone is a 400 costing no call.

        Ref: stdapi/routes/_responses_context.py:_compact_on_demand
        """
        response = _post(app_client, input=[{"type": "compaction_trigger"}])
        assert response.status_code == 400, response.text
        assert "no conversation to compact" in response.json()["error"]["message"]
        assert not backend.requests


def _has_consecutive_user_messages(request: ResponseCreateParams) -> bool:
    """Whether an input holds two user messages in a row.

    Args:
        request: The request.

    Returns:
        True when a user message directly follows another.
    """
    items = request.input if isinstance(request.input, list) else []
    roles = [getattr(item, "role", None) for item in items]
    return any(a == b == "user" for a, b in pairwise(roles))


def _state(content: str) -> dict[str, Any]:
    """Decode a response-produced compaction item's content.

    Args:
        content: The item's ``encrypted_content``.

    Returns:
        The summary and the items kept around it.
    """
    state: dict[str, Any] = loads(urlsafe_b64decode(content.removeprefix("v2:")))
    return state


class _NativeModel:
    """Stand-in for a Bedrock Mantle model, refusing what Bedrock Mantle refuses.

    Local compaction items are refused, and, for a converted model, two user
    messages in a row, as some served models' chat templates do.
    """

    def __init__(self, *, native: bool) -> None:
        self.native = native
        self.requests: list[ResponseCreateParams] = []

    def native_store_supported(self) -> bool:
        return self.native

    async def create_response(
        self,
        request: ResponseCreateParams,
        response_id: str,
        created_at: float,
        moderation_builder: object = None,
    ) -> Response | EventSourceResponse:
        self.requests.append(request)
        _reject_local_compaction_items(
            request.model_dump(mode="json", exclude_none=True).get("input")
        )
        if not self.native and _has_consecutive_user_messages(request):
            msg = "Conversation roles must alternate user/assistant/user/assistant."
            raise ApiError(msg, status=400)
        text = "SUMMARY" if _DIRECTIVE in str(request.input) else "ANSWER"
        response = Response(
            id=response_id,
            created_at=int(created_at),
            model=request.model,
            object="response",
            output=[
                ResponseOutputMessage(
                    id="msg-1",
                    content=[
                        ResponseOutputText(
                            annotations=[], text=text, type="output_text"
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
        )
        if not request.stream:
            return response
        snapshot = response.model_dump(mode="json", exclude_none=True)

        async def _events() -> AsyncGenerator[ServerSentEvent]:
            for index, kind in enumerate(
                ("response.created", "response.in_progress", "response.completed")
            ):
                payload = {"type": kind, "sequence_number": index, "response": snapshot}
                yield ServerSentEvent(data=dumps(payload), event=kind)

        return EventSourceResponse(_events())


class TestMantleServedModels:
    """A native Responses model honours the parameters itself; others get them here.

    Ref: stdapi/routes/_responses_context.py:generate
    """

    @pytest.fixture
    def model(self, monkeypatch: pytest.MonkeyPatch) -> _NativeModel:
        """Serve the route from a stand-in model."""

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> ModelDetails:
            return make_model_details(model_id)

        stub = _NativeModel(native=True)
        monkeypatch.setattr(openai_responses, "validate_model", _validate_model)
        monkeypatch.setattr(openai_responses, "get_chat_model", lambda _id: stub)
        monkeypatch.setattr(openai_responses, "serves_via_mantle", lambda _id: True)
        monkeypatch.setattr(_responses_context, "serves_via_mantle", lambda _id: True)
        return stub

    def test_a_native_model_receives_the_parameters_untouched(
        self, app_client: TestClient, model: _NativeModel
    ) -> None:
        """Truncation, compaction and a trigger all reach the native API as sent.

        Ref: stdapi/routes/_responses_context.py:generate
        """
        response = _post(
            app_client,
            input=[{"role": "user", "content": "x"}, {"type": "compaction_trigger"}],
            truncation="auto",
            context_management=[{"type": "compaction", "compact_threshold": 1000}],
        )
        assert response.status_code == 200, response.text
        (request,) = model.requests
        assert request.truncation == "auto"
        assert request.context_management is not None
        assert isinstance(request.input, list)
        assert request.input[-1].model_dump()["type"] == "compaction_trigger"

    @pytest.mark.parametrize("stream", [False, True])
    def test_a_converted_model_compacts_and_answers(
        self, app_client: TestClient, model: _NativeModel, stream: bool
    ) -> None:
        """A converted Mantle model is answered from the compaction, expanded.

        The model refuses a local compaction item, so the item the compaction
        pass produced must reach it expanded, streamed or not.

        Ref: stdapi/routes/_responses_context.py:_Call.open
        """
        model.native = False
        response = _post(
            app_client,
            input=_turns(3, size=30_000),
            context_management=[{"type": "compaction", "compact_threshold": 3000}],
            stream=stream,
        )
        assert response.status_code == 200, response.text
        final = _events(response)[-1]["response"] if stream else response.json()
        if stream:
            assert _events(response)[-1]["type"] == "response.completed"
        assert final["output"][0]["type"] == "compaction"
        _summary_call, answer_call = model.requests
        assert isinstance(answer_call.input, list)
        assert not any(
            isinstance(item, CompactionItemParam) for item in answer_call.input
        )
        assert "SUMMARY" in str(answer_call.input)

    def test_a_converted_model_receives_expanded_compaction_items(
        self, app_client: TestClient, model: _NativeModel
    ) -> None:
        """Bedrock Mantle refuses local compaction items, so they arrive expanded.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:expand_compaction_items
        """
        model.native = False
        item = CompactionItemParam(
            type="compaction", encrypted_content=encode_compaction_content("S")
        )
        response = _post(
            app_client,
            input=[
                item.model_dump(exclude_none=True),
                {"role": "user", "content": "q"},
            ],
        )
        assert response.status_code == 200, response.text
        (request,) = model.requests
        assert isinstance(request.input, list)
        assert [getattr(entry, "type", None) for entry in request.input] != [
            "compaction",
            None,
        ]
        assert not any(
            isinstance(entry, CompactionItemParam) for entry in request.input
        )

    def test_consecutive_compactions_of_a_tool_loop_keep_only_the_task(
        self, app_client: TestClient, model: _NativeModel
    ) -> None:
        """Each compaction of one task's tool loop keeps the task alone verbatim.

        The previous summary is summarized again rather than kept inside the
        task, so kept items never accumulate summaries; and no call, the
        summaries included, sends two user messages in a row.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:split_for_compaction
             stdapi/models/chat/_adapters/_openai_responses.py:join_summaries
        """
        model.native = False
        cm = [{"type": "compaction", "compact_threshold": 3000}]
        task = {"role": "user", "content": "fix the bug"}

        def _step(call_id: str) -> list[dict[str, Any]]:
            return [
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "lookup",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": "ok " + "x" * 20_000,
                },
            ]

        items: list[dict[str, Any]] = [task, *_step("c1"), *_step("c2")]
        for round_ in range(2):
            response = _post(app_client, input=items, context_management=cm)
            assert response.status_code == 200, response.text
            compaction = response.json()["output"][0]
            assert compaction["type"] == "compaction"
            state = _state(compaction["encrypted_content"])
            assert state["before"] == [{"role": "user", "content": "fix the bug"}]
            assert state["summary"] == "SUMMARY"
            items = [compaction, *_step(f"c{round_ + 3}"), *_step(f"c{round_ + 5}")]
        second_summary = model.requests[-2]
        assert _DIRECTIVE in str(second_summary.input)
        assert "Summary of the earlier conversation" in str(second_summary.input)

    def test_the_compact_endpoint_joins_the_directive_for_a_converted_model(
        self, app_client: TestClient, model: _NativeModel
    ) -> None:
        """A summary as the last item never meets the directive as a second user message.

        Ref: stdapi/routes/_responses_context.py:summarize
        """
        model.native = False
        item = CompactionItemParam(
            type="compaction", encrypted_content=encode_compaction_content("S")
        )
        response = app_client.post(
            "/v1/responses/compact",
            json={
                "model": "amazon.nova-micro-v1:0",
                "input": [
                    {"role": "assistant", "content": "a"},
                    item.model_dump(exclude_none=True),
                ],
            },
        )
        assert response.status_code == 200, response.text
        (request,) = model.requests
        assert isinstance(request.input, list)
        assert _DIRECTIVE in str(request.input[-1])
        assert "Summary of the earlier conversation" in str(request.input[-1])
