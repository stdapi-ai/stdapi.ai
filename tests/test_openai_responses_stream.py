"""The Responses adapter maps Bedrock Converse output onto the OpenAI SSE taxonomy.

In-process tests: fabricated Bedrock Converse / ConverseStream payloads are fed
straight to the adapter, so every assertion here is deterministic and no AWS
call is made.

Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
     https://developers.openai.com/api/docs/guides/streaming-responses
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
     stdapi/models/chat/_adapters/_openai_responses.py:format_stream
     stdapi/models/chat/_adapters/_openai_responses.py:format_response
"""

import json
from asyncio import Event, wait_for
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, cast, get_args

import pytest
from openai.types.responses.response_completed_event import (
    ResponseCompletedEvent as SDKResponseCompletedEvent,
)
from openai.types.responses.response_error_event import (
    ResponseErrorEvent as SDKResponseErrorEvent,
)
from openai.types.responses.response_failed_event import (
    ResponseFailedEvent as SDKResponseFailedEvent,
)
from openai.types.responses.response_incomplete_event import (
    ResponseIncompleteEvent as SDKResponseIncompleteEvent,
)
from sse_starlette import JSONServerSentEvent, ServerSentEvent

from stdapi import monitoring
from stdapi.aws_bedrock import GUARDRAIL_TRACE_VAR
from stdapi.cleanup import CLEANUPS
from stdapi.models import ModelBase
from stdapi.models import chat as models_chat
from stdapi.models.chat._adapters import _openai_chat_completion as chat_adapter
from stdapi.models.chat._adapters import _openai_responses as responses_adapter
from stdapi.models.chat._adapters._openai_common import extract_stream_usage
from stdapi.models.chat._adapters._openai_responses import (
    execute_image_generation_calls,
    format_response,
    format_stream,
)
from stdapi.monitoring import REQUEST_ID, SseHandledStreamError
from stdapi.routes._moderation import build_response_moderation
from stdapi.types.openai import ModerationResult, RequestModeration, ResponseModeration
from stdapi.types.openai_responses import (
    AnnotationURLCitation,
    ImageGeneration,
    ImageGenerationCall,
    Response,
    ResponseCreateParams,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseIncludable,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    WebSearchActionSearch,
)
from stdapi.vector_stores import SearchResult, StoreRecord

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Generator, Sequence

    from types_aiobotocore_bedrock_runtime.type_defs import (
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
    )

    from stdapi.types.openai_chat_completions import ChatCompletion

#: Local, in-process tests with the request-log context response logging requires.
pytestmark = [pytest.mark.local, pytest.mark.usefixtures("request_log")]


def _request(**kwargs: object) -> ResponseCreateParams:
    """Build a Responses creation request with optional extra fields."""
    return ResponseCreateParams.model_validate(
        {"model": "anthropic.claude-sonnet-5", "input": "hi", **kwargs}
    )


def _bedrock_response(
    contents: list[dict[str, object]],
    stop_reason: str = "end_turn",
    usage: dict[str, object] | None = None,
) -> ConverseResponseTypeDef:
    """Build a minimal Bedrock Converse response around content blocks."""
    return cast(
        "ConverseResponseTypeDef",
        {
            "output": {"message": {"role": "assistant", "content": contents}},
            "usage": usage or {"inputTokens": 3, "outputTokens": 5},
            "stopReason": stop_reason,
        },
    )


async def _stream(
    events: list[dict[str, object]],
) -> AsyncGenerator[ConverseStreamOutputTypeDef]:
    """Yield fabricated Bedrock ConverseStream events."""
    for event in events:
        yield cast("ConverseStreamOutputTypeDef", event)


def _payload(sse: JSONServerSentEvent) -> dict[str, Any]:
    """Return the decoded data payload of an SSE event."""
    if isinstance(sse.data, dict):
        return sse.data
    assert isinstance(sse.data, str | bytes | bytearray)
    return cast("dict[str, Any]", json.loads(sse.data))


def _text_stream_events(
    stop_reason: str = "end_turn", usage: dict[str, object] | None = None
) -> list[dict[str, object]]:
    """Build a minimal Bedrock stream producing one text block."""
    return [
        {"contentBlockStart": {"start": {}, "contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "Hello"}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": stop_reason}},
        {"metadata": {"usage": usage or {"inputTokens": 3, "outputTokens": 5}}},
    ]


async def _collect(
    stream: AsyncGenerator[JSONServerSentEvent],
) -> list[JSONServerSentEvent]:
    """Drain a format_stream generator into a list of SSE events."""
    return [sse async for sse in stream]


def _event_names(events: Sequence[ServerSentEvent]) -> list[str | None]:
    """Return the SSE ``event:`` names in emission order."""
    return [sse.event for sse in events]


def _sequence_numbers(events: list[JSONServerSentEvent]) -> list[int]:
    """Return the ``sequence_number`` of every event payload, in order."""
    return [_payload(sse)["sequence_number"] for sse in events]


#: Events a single-text-block stream emits before its terminal event.
_TEXT_STREAM_EVENT_NAMES = [
    "response.created",
    "response.in_progress",
    "response.output_item.added",
    "response.content_part.added",
    "response.output_text.delta",
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
]


class TestTerminalEvents:
    """The terminal stream event matches the Bedrock Converse stop reason.

    ``messageStop.stopReason`` has no OpenAI counterpart, so the adapter maps it
    onto the three terminal events plus the ``incomplete_details.reason`` enum
    (``max_output_tokens`` / ``content_filter``); reasons Bedrock may add later
    fall back to ``max_output_tokens`` with a logged warning.

    Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
         https://developers.openai.com/api/reference/resources/responses/methods/retrieve
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
         stdapi/models/chat/_adapters/_openai_responses.py:_map_stop_reason
         stdapi/models/chat/_adapters/_openai_responses.py:_terminal_event
    """

    async def test_completed_on_end_turn(self) -> None:
        """``end_turn`` closes the documented lifecycle with response.completed.

        Also pins the whole event sequence and the contiguous ``sequence_number``
        counter, which upstream requires to be monotonically increasing.

        Ref: https://developers.openai.com/api/docs/guides/streaming-responses
             https://github.com/openai/openai-python/tree/main/src/openai/types/responses
        """
        events = await _collect(
            format_stream(
                "resp-1", 1.0, "model", _stream(_text_stream_events()), _request()
            )
        )
        assert _event_names(events) == [*_TEXT_STREAM_EVENT_NAMES, "response.completed"]
        assert _sequence_numbers(events) == list(range(len(events)))
        payload = _payload(events[-1])
        assert payload["type"] == "response.completed"
        assert payload["response"]["status"] == "completed"
        assert "incomplete_details" not in payload["response"]
        assert "error" not in payload["response"]
        message = payload["response"]["output"][-1]
        assert message["content"][0]["text"] == "Hello", (
            "the terminal snapshot must carry the text streamed by the deltas"
        )
        sdk_event = SDKResponseCompletedEvent.model_validate(payload)
        assert sdk_event.sequence_number == payload["sequence_number"]

    async def test_incomplete_on_max_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hitting max tokens ends with response.incomplete, no response.completed, and no warning.

        ``max_tokens`` is a known Bedrock stop reason, so it maps to
        ``incomplete_details.reason = max_output_tokens`` without the
        unknown-reason warning.
        """
        logged: list[object] = []
        monkeypatch.setattr(
            responses_adapter, "log_error_details", lambda *a, **_kw: logged.append(a)
        )
        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(_text_stream_events(stop_reason="max_tokens")),
                _request(),
            )
        )
        assert _event_names(events) == [
            *_TEXT_STREAM_EVENT_NAMES,
            "response.incomplete",
        ]
        payload = _payload(events[-1])
        assert payload["type"] == "response.incomplete"
        assert payload["response"]["status"] == "incomplete"
        assert payload["response"]["incomplete_details"] == {
            "reason": "max_output_tokens"
        }
        assert "error" not in payload["response"]
        assert logged == [], "a documented stop reason must not log a warning"
        sdk_event = SDKResponseIncompleteEvent.model_validate(payload)
        assert sdk_event.sequence_number == payload["sequence_number"]

    @pytest.mark.parametrize(
        "stop_reason", ["guardrail_intervened", "content_filtered"]
    )
    async def test_incomplete_on_filtered_content(self, stop_reason: str) -> None:
        """Both filtering stop reasons map to incomplete_details.reason content_filter.

        Bedrock reports guardrail interventions and model-side content filtering
        as two distinct stop reasons; OpenAI has a single ``content_filter``
        reason for both.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html
        """
        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(_text_stream_events(stop_reason=stop_reason)),
                _request(),
            )
        )
        assert _event_names(events) == [
            *_TEXT_STREAM_EVENT_NAMES,
            "response.incomplete",
        ]
        payload = _payload(events[-1])
        assert payload["response"]["status"] == "incomplete"
        assert payload["response"]["incomplete_details"] == {"reason": "content_filter"}

    async def test_incomplete_on_unknown_reason_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A genuinely unknown stop reason maps to max_output_tokens and logs a warning.

        The reason enum is open-ended on the Bedrock side, so an unrecognised
        value degrades to the closest OpenAI reason rather than failing the
        stream, and is reported once at ``warning`` level.
        """
        logged: list[tuple[object, ...]] = []
        logged_kwargs: list[dict[str, object]] = []

        def _capture(*args: object, **kwargs: object) -> None:
            logged.append(args)
            logged_kwargs.append(kwargs)

        monkeypatch.setattr(responses_adapter, "log_error_details", _capture)
        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(_text_stream_events(stop_reason="something_new")),
                _request(),
            )
        )
        assert _event_names(events) == [
            *_TEXT_STREAM_EVENT_NAMES,
            "response.incomplete",
        ]
        payload = _payload(events[-1])
        assert payload["response"]["status"] == "incomplete"
        assert payload["response"]["incomplete_details"] == {
            "reason": "max_output_tokens"
        }
        assert len(logged) == 1
        message = logged[0][0]
        assert isinstance(message, str)
        assert "something_new" in message
        assert logged_kwargs == [{"level": "warning"}]

    async def test_failed_on_malformed_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed model output ends with response.failed carrying an error.

        ``malformed_model_output`` is not a truncation, so it becomes
        ``status=failed`` with a ``server_error`` ResponseError.  The Bedrock
        stop reason is a backend detail: it goes to the server log only, never
        into the client-visible error message.
        """
        logged: list[tuple[object, ...]] = []

        def _capture(*args: object, **_kwargs: object) -> None:
            logged.append(args)

        monkeypatch.setattr(responses_adapter, "log_error_details", _capture)
        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(_text_stream_events(stop_reason="malformed_model_output")),
                _request(),
            )
        )
        assert _event_names(events) == [*_TEXT_STREAM_EVENT_NAMES, "response.failed"]
        payload = _payload(events[-1])
        assert payload["type"] == "response.failed"
        assert payload["response"]["status"] == "failed"
        assert "incomplete_details" not in payload["response"]
        error = payload["response"]["error"]
        assert error["code"] == "server_error"
        assert error["message"] == "The model failed to generate a valid response."
        assert any(
            isinstance(args[0], str) and "malformed_model_output" in args[0]
            for args in logged
        )
        sdk_event = SDKResponseFailedEvent.model_validate(payload)
        assert sdk_event.sequence_number == payload["sequence_number"]


class TestFailedResponseError:
    """Failed non-streaming responses carry a populated error object.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/retrieve
         stdapi/models/chat/_adapters/_openai_responses.py:_map_stop_reason
         stdapi/models/chat/_adapters/_openai_responses.py:format_response
    """

    async def test_error_populated_on_failed_status(self) -> None:
        """A malformed_tool_use stop reason yields status failed with error.

        The stop reason itself stays out of the client-visible message.
        ``completed_at`` stays ``None`` because the gateway only stamps it for a
        ``completed`` status.
        """
        response = await format_response(
            "resp-1",
            1.0,
            "model",
            _bedrock_response([{"text": "oops"}], stop_reason="malformed_tool_use"),
            _request(),
        )
        assert response.status == "failed"
        assert response.error is not None
        assert response.error.code == "server_error"
        assert (
            response.error.message == "The model failed to generate a valid response."
        )
        assert response.incomplete_details is None
        assert response.completed_at is None

    async def test_no_error_on_completed(self) -> None:
        """A completed response has no error object."""
        response = await format_response(
            "resp-1", 1.0, "model", _bedrock_response([{"text": "hi"}]), _request()
        )
        assert response.status == "completed"
        assert response.error is None
        assert response.incomplete_details is None
        assert isinstance(response.completed_at, int)


class TestTextBlockJoinMatchesStreaming:
    """Contiguous Bedrock text blocks concatenate identically on both paths.

    Streaming emits each block's deltas joined with an empty separator, so the
    non-streaming path must not inject newlines between blocks: the same
    Converse output must produce the same ``output_text`` either way.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/models/chat/_adapters/_openai_responses.py:_flush_message_item
    """

    async def test_text_blocks_concatenate_without_separator(self) -> None:
        """Two text blocks yield one message whose text is their concatenation."""
        response = await format_response(
            "resp-1",
            1.0,
            "model",
            _bedrock_response([{"text": "Hello"}, {"text": " world"}]),
            _request(),
        )
        message = response.output[0]
        assert isinstance(message, ResponseOutputMessage)
        part = message.content[0]
        assert isinstance(part, ResponseOutputText)
        assert part.text == "Hello world"


class TestOpenBlockFlushedAtStreamEnd:
    """A stream truncated before ``contentBlockStop`` still delivers its block.

    Reasoning blocks are flushed defensively at stream end; an open text block
    gets the same treatment instead of being silently dropped from the final
    response.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
         stdapi/models/chat/_adapters/_openai_responses.py:format_stream
    """

    async def test_text_block_without_stop_is_flushed(self) -> None:
        """The message closes and lands in the terminal response snapshot."""
        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(
                    [
                        {"contentBlockStart": {"start": {}, "contentBlockIndex": 0}},
                        {
                            "contentBlockDelta": {
                                "delta": {"text": "Hello"},
                                "contentBlockIndex": 0,
                            }
                        },
                        {"messageStop": {"stopReason": "end_turn"}},
                        {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 5}}},
                    ]
                ),
                _request(),
            )
        )
        assert _event_names(events) == [*_TEXT_STREAM_EVENT_NAMES, "response.completed"]
        final = _payload(events[-1])["response"]
        message = final["output"][0]
        assert message["status"] == "completed"
        assert message["content"][0]["text"] == "Hello"


class TestAcceptedButUnsupportedRequestFields:
    """Fields Converse cannot honor are echoed on the response and change nothing else.

    ``top_logprobs``, ``text.verbosity`` and every ``include`` value other than
    ``reasoning.encrypted_content`` are accepted for client compatibility —
    notably Codex, which always sends ``text.verbosity``. Bedrock Converse
    exposes no equivalent, so they must round-trip on the response object
    without fabricating data the backend never produced.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/models/chat/_adapters/_openai_responses.py:_build_response_object
    """

    async def test_top_logprobs_is_echoed_and_no_logprobs_are_fabricated(self) -> None:
        """``top_logprobs`` is echoed while output parts stay free of ``logprobs``.

        Bedrock Converse returns no token log probabilities, so the only correct
        behavior is echoing the request value and leaving ``logprobs`` unset
        rather than inventing entries.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        response = await format_response(
            "resp-1",
            1.0,
            "model",
            _bedrock_response([{"text": "hi"}]),
            _request(top_logprobs=3),
        )
        assert response.top_logprobs == 3
        message = response.output[0]
        assert isinstance(message, ResponseOutputMessage)
        part = message.content[0]
        assert isinstance(part, ResponseOutputText)
        assert getattr(part, "logprobs", None) is None

    async def test_text_verbosity_is_echoed_and_ignored(self) -> None:
        """``text.verbosity`` round-trips on the response and reaches no Bedrock field.

        Both surfaces accept the hint and drop it; here it also round-trips on
        the response, and only ``text.format`` is translated into the Converse
        output configuration.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:_build_output_config
        """
        request = _request(text={"verbosity": "low", "format": {"type": "text"}})
        response = await format_response(
            "resp-1", 1.0, "model", _bedrock_response([{"text": "hi"}]), request
        )
        assert response.text is not None
        assert response.text.verbosity == "low"
        assert responses_adapter.translate_request(request, "model")[3] is None, (
            "verbosity must not produce a Converse output configuration"
        )

    async def test_every_include_value_is_accepted_and_ignored(self) -> None:
        """The whole ``include`` enum is accepted and changes nothing on a text answer.

        Clients routinely ask for values such as
        ``web_search_call.action.sources``; only
        ``reasoning.encrypted_content`` has an effect here, and only on a
        response that carries a reasoning item.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:_includes_encrypted_reasoning
             stdapi/types/openai_responses.py:ResponseIncludable
        """
        includes = list(get_args(ResponseIncludable))
        assert len(includes) == 8, "the upstream include enum has eight values"
        baseline = await format_response(
            "resp-1", 1.0, "model", _bedrock_response([{"text": "hi"}]), _request()
        )
        response = await format_response(
            "resp-1",
            1.0,
            "model",
            _bedrock_response([{"text": "hi"}]),
            _request(include=includes),
        )
        assert response.model_dump(exclude={"completed_at"}) == baseline.model_dump(
            exclude={"completed_at"}
        )


class TestMidStreamErrors:
    """Mid-stream exceptions emit spec events and a single log record.

    Upstream defines a bare ``error`` event (``ResponseErrorEvent`` with code /
    message / param) for mid-stream failures; the gateway emits it and then a
    ``response.failed`` snapshot before re-raising ``SseHandledStreamError`` so
    ``log_request_sse_stream_event`` records the failure once and suppresses the
    legacy REST-envelope error event.

    Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
         https://developers.openai.com/api/docs/guides/streaming-responses
         stdapi/models/chat/_adapters/_openai_responses.py:_classify_stream_error
         stdapi/monitoring.py:log_request_sse_stream_event
    """

    @staticmethod
    def _failing_stream(exc: Exception) -> AsyncGenerator[ConverseStreamOutputTypeDef]:
        """Build a Bedrock stream raising *exc* after the first text delta."""

        async def generate() -> AsyncGenerator[ConverseStreamOutputTypeDef]:
            yield cast(
                "ConverseStreamOutputTypeDef",
                {"contentBlockDelta": {"delta": {"text": "par"}}},
            )
            raise exc

        return generate()

    @staticmethod
    async def _drain_until_error(
        stream: AsyncGenerator[JSONServerSentEvent],
    ) -> tuple[list[JSONServerSentEvent], SseHandledStreamError]:
        """Collect stream events until SseHandledStreamError is raised."""
        events: list[JSONServerSentEvent] = []
        try:
            # Not an async comprehension: events emitted before the raise matter.
            async for sse in stream:
                events.append(sse)  # noqa: PERF401
        except SseHandledStreamError as exc:
            return events, exc
        pytest.fail("Expected SseHandledStreamError")

    async def test_spec_error_event_then_failed(self) -> None:
        """An unexpected exception emits error + response.failed, then re-raises.

        A non-API exception is sanitised to ``Internal Server Error`` /
        ``server_error`` so no internal detail reaches the client, while the raw
        traceback is carried on the re-raised ``SseHandledStreamError`` for the
        request log.
        """
        events, error = await self._drain_until_error(
            format_stream(
                "resp-1",
                1.0,
                "model",
                self._failing_stream(RuntimeError("boom")),
                _request(),
            )
        )
        assert error.status == 500
        assert error.level == "critical"
        assert isinstance(error.__cause__, RuntimeError)
        assert "RuntimeError: boom" in error.args[0], (
            "the log detail must keep the original exception"
        )

        assert _event_names(events) == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "error",
            "response.failed",
        ]
        assert _sequence_numbers(events) == list(range(len(events))), (
            "the error and failed events must continue the same sequence counter"
        )
        assert _payload(events[4])["delta"] == "par", (
            "deltas emitted before the failure must still reach the client"
        )

        error_payload = _payload(events[-2])
        assert error_payload["type"] == "error"
        assert error_payload["code"] == "server_error"
        assert error_payload["message"] == "Internal Server Error"
        assert "boom" not in json.dumps(error_payload)
        SDKResponseErrorEvent.model_validate(error_payload)

        failed_payload = _payload(events[-1])
        assert failed_payload["response"]["status"] == "failed"
        assert failed_payload["response"]["error"]["code"] == "server_error"
        assert failed_payload["response"]["error"]["message"] == "Internal Server Error"
        SDKResponseFailedEvent.model_validate(failed_payload)

    async def test_api_error_carries_code_and_param(self) -> None:
        """An ApiError surfaces its message, code, and param in the error event.

        A 4xx ``ApiError`` is client-caused, so its message is forwarded verbatim
        (no sanitising) and the request log records it at ``warning`` rather than
        ``critical``.

        Ref: stdapi/api_errors.py:ApiError
        """
        from stdapi.api_errors import ApiError  # noqa: PLC0415

        exc = ApiError("bad tool", status=400)
        exc.code = "invalid_value"
        exc.param = "tools"
        events, error = await self._drain_until_error(
            format_stream("resp-1", 1.0, "model", self._failing_stream(exc), _request())
        )
        assert error.status == 400
        assert error.level == "warning"
        assert _event_names(events)[-2:] == ["error", "response.failed"]
        error_payload = _payload(events[-2])
        assert error_payload["type"] == "error"
        assert error_payload["message"] == "bad tool"
        assert error_payload["code"] == "invalid_value"
        assert error_payload["param"] == "tools"
        SDKResponseErrorEvent.model_validate(error_payload)
        failed_error = _payload(events[-1])["response"]["error"]
        assert failed_error["code"] == "server_error", (
            "ResponseError.code only carries the upstream enum, not the ApiError code"
        )
        assert failed_error["message"] == "bad tool"

    async def test_wrapper_logs_once_without_legacy_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """log_request_sse_stream_event records once and emits no legacy error.

        The monitoring wrapper normally appends a REST-envelope ``error`` event of
        its own; recognising ``SseHandledStreamError`` is what keeps the Responses
        stream to the two spec events and the request log to one record.

        Ref: stdapi/monitoring.py:SseHandledStreamError
        """
        written: list[dict[str, object]] = []
        logged: list[tuple[tuple[object, ...], dict[str, object]]] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)
        monkeypatch.setattr(
            monitoring, "log_error_details", lambda *a, **kw: logged.append((a, kw))
        )
        id_token = REQUEST_ID.set("test-request-id")
        try:
            stream = monitoring.log_request_sse_stream_event(
                format_stream(
                    "resp-1",
                    1.0,
                    "model",
                    self._failing_stream(RuntimeError("boom")),
                    _request(),
                )
            )
            events = [sse async for sse in stream]
        finally:
            REQUEST_ID.reset(id_token)

        # Spec events only: the legacy REST-envelope error event is not appended.
        error_sse = events[-2]
        assert isinstance(error_sse, JSONServerSentEvent)
        assert _event_names(events)[-2:] == ["error", "response.failed"]
        assert _event_names(events).count("error") == 1, (
            "the wrapper must not append a second, REST-envelope error event"
        )
        assert _payload(error_sse)["type"] == "error"
        # The error is recorded exactly once in the request log.
        assert len(logged) == 1
        logged_message = logged[0][0][0]
        assert isinstance(logged_message, str)
        assert "RuntimeError: boom" in logged_message
        assert logged[0][1] == {"status": 500, "level": "critical"}
        # The request_stream log entry is still written, without duplication.
        assert len([w for w in written if w["type"] == "request_stream"]) == 1


class TestUsageAccounting:
    """Cache buckets are folded into input/prompt tokens (OpenAI semantics).

    Bedrock's ``TokenUsage.inputTokens`` EXCLUDES ``cacheReadInputTokens`` and
    ``cacheWriteInputTokens``, whereas OpenAI's ``input_tokens`` /
    ``prompt_tokens`` include the cached prefix, so every adapter adds the two
    buckets back (10 + 20 + 10 = 40).  The fixture's ``totalTokens`` is
    deliberately inconsistent with that sum: the gateway recomputes the total
    from its own input + output and never forwards Bedrock's value.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
         https://developers.openai.com/api/reference/resources/responses/methods/retrieve
         https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
    """

    #: Bedrock usage payload with both cache buckets populated and a bogus total.
    _CACHED_USAGE: ClassVar[dict[str, object]] = {
        "inputTokens": 10,
        "outputTokens": 5,
        "totalTokens": 999,
        "cacheReadInputTokens": 20,
        "cacheWriteInputTokens": 10,
    }

    async def test_responses_batch_usage(self) -> None:
        """Non-streaming usage: input includes cache buckets, cached is read bucket.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:format_response
        """
        response = await format_response(
            "resp-1",
            1.0,
            "model",
            _bedrock_response([{"text": "hi"}], usage=self._CACHED_USAGE),
            _request(),
        )
        assert response.usage is not None
        assert response.usage.input_tokens == 40
        assert response.usage.input_tokens_details.cached_tokens == 20
        assert response.usage.input_tokens_details.cache_write_tokens == 10
        assert response.usage.output_tokens == 5
        assert response.usage.total_tokens == 45, (
            "total must be recomputed, not copied from Bedrock's totalTokens"
        )
        assert response.usage.output_tokens_details is not None
        assert response.usage.output_tokens_details.reasoning_tokens == 0

    async def test_responses_stream_usage(self) -> None:
        """Streamed usage: the terminal event applies the same cache math.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStreamMetadataEvent.html
             stdapi/models/chat/_adapters/_openai_responses.py:format_stream
        """
        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(_text_stream_events(usage=self._CACHED_USAGE)),
                _request(),
            )
        )
        usage = _payload(events[-1])["response"]["usage"]
        assert usage["input_tokens"] == 40
        assert usage["input_tokens_details"]["cached_tokens"] == 20
        assert usage["input_tokens_details"]["cache_write_tokens"] == 10
        assert usage["output_tokens"] == 5
        assert usage["total_tokens"] == 45, (
            "total must be recomputed, not copied from Bedrock's totalTokens"
        )
        assert usage["output_tokens_details"] == {"reasoning_tokens": 0}

    async def test_chat_batch_usage(self) -> None:
        """Chat completions prompt_tokens include both cache buckets.

        Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py:format_response
        """
        legacy_token = chat_adapter._LEGACY_FUNCTION.set(False)  # noqa: SLF001
        try:
            completion = await self._format_chat_response()
        finally:
            chat_adapter._LEGACY_FUNCTION.reset(legacy_token)  # noqa: SLF001
        assert completion.usage is not None
        assert completion.usage.prompt_tokens == 40
        assert completion.usage.completion_tokens == 5
        assert completion.usage.total_tokens == 45, (
            "total must be recomputed, not copied from Bedrock's totalTokens"
        )
        assert completion.usage.prompt_tokens_details is not None
        assert completion.usage.prompt_tokens_details.cached_tokens == 20
        assert completion.usage.prompt_tokens_details.cache_write_tokens == 10

    async def _format_chat_response(self) -> ChatCompletion:
        """Format a canned Bedrock response through the chat adapter."""
        return await chat_adapter.format_response(
            "cmpl-1",
            1,
            "model",
            cast(
                "list[ConverseResponseTypeDef]",
                [
                    {
                        "output": {"message": {"content": [{"text": "hi"}]}},
                        "usage": self._CACHED_USAGE,
                        "stopReason": "end_turn",
                    }
                ],
            ),
            None,
            None,
            ["text"],
        )

    def test_chat_stream_usage(self) -> None:
        """Chat streaming usage extraction applies the same cache math.

        Ref: stdapi/models/chat/_adapters/_openai_common.py:extract_stream_usage
        """
        usage = extract_stream_usage(
            cast(
                "ConverseStreamOutputTypeDef",
                {"metadata": {"usage": self._CACHED_USAGE}},
            )
        )
        assert usage is not None
        assert usage.prompt_tokens == 40
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 45, (
            "total must be recomputed, not copied from Bedrock's totalTokens"
        )
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 20
        assert usage.prompt_tokens_details.cache_write_tokens == 10


class TestAnnotations:
    """Citations surface as url_citation annotations.

    Upstream attaches ``url_citation`` annotations to the ``output_text`` part
    with character offsets.  Bedrock's ``citationsContent`` blocks carry no
    character indices, so the adapter anchors each citation at the length of the
    text accumulated so far, making ``start_index == end_index``.

    Ref: https://developers.openai.com/api/docs/guides/tools-web-search
         https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
         stdapi/models/chat/_adapters/_openai_responses.py:_citation_annotations
         stdapi/models/chat/_adapters/_openai_responses.py:_record_citation
    """

    async def test_non_streaming_annotations(self) -> None:
        """CitationsContent blocks yield annotations on the output message.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:_record_block_citations
        """
        contents: list[dict[str, object]] = [
            {"text": "Paris is the capital."},
            {
                "citationsContent": {
                    "citations": [
                        {
                            "title": "Paris",
                            "location": {"web": {"url": "https://example.com/paris"}},
                        }
                    ]
                }
            },
        ]
        response = await format_response(
            "resp-1", 1.0, "model", _bedrock_response(contents), _request()
        )
        message = response.output[-1]
        assert isinstance(message, ResponseOutputMessage)
        (annotation,) = message.content[0].annotations  # type: ignore[union-attr]
        assert isinstance(annotation, AnnotationURLCitation)
        assert annotation.type == "url_citation"
        assert annotation.url == "https://example.com/paris"
        assert annotation.title == "Paris"
        assert annotation.start_index == len("Paris is the capital.")
        assert annotation.end_index == annotation.start_index

    async def test_streaming_annotation_added_event(self) -> None:
        """A citation delta during a text block emits annotation.added.

        The annotation must also be present on the ``content_part.done`` part and
        on the message in the terminal snapshot, so a client that only reads the
        final response sees the same citations as a delta consumer.
        """
        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(
                    [
                        {"contentBlockStart": {"start": {}}},
                        {"contentBlockDelta": {"delta": {"text": "Hello"}}},
                        {
                            "contentBlockDelta": {
                                "delta": {
                                    "citation": {
                                        "title": "Src",
                                        "location": {
                                            "web": {"url": "https://example.com"}
                                        },
                                    }
                                }
                            }
                        },
                        {"contentBlockStop": {}},
                        {"messageStop": {"stopReason": "end_turn"}},
                        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
                    ]
                ),
                _request(),
            )
        )
        (added,) = [
            sse
            for sse in events
            if sse.event == "response.output_text.annotation.added"
        ]
        added_payload = _payload(added)
        assert added_payload["annotation_index"] == 0
        assert added_payload["annotation"]["type"] == "url_citation"
        assert added_payload["annotation"]["url"] == "https://example.com"
        assert added_payload["annotation"]["title"] == "Src"
        assert added_payload["annotation"]["start_index"] == len("Hello")
        assert added_payload["annotation"]["end_index"] == len("Hello")

        by_event = {sse.event: _payload(sse) for sse in events}
        assert (
            added_payload["item_id"]
            == by_event["response.output_item.added"]["item"]["id"]
        )
        text_done = by_event["response.output_text.done"]
        assert text_done["text"] == "Hello"
        part_done = by_event["response.content_part.done"]
        assert part_done["part"]["annotations"][0]["url"] == "https://example.com"
        message = by_event["response.completed"]["response"]["output"][-1]
        assert message["content"][0]["annotations"][0]["url"] == "https://example.com"

    async def test_streaming_two_citations_in_one_part(self) -> None:
        """Two citations in one text part emit annotation_index 0 then 1.

        ``annotation_index`` is per content part, so it restarts at 0 for the
        part and counts up as citations arrive interleaved with text deltas.
        """
        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(
                    [
                        {"contentBlockStart": {"start": {}}},
                        {"contentBlockDelta": {"delta": {"text": "Hello "}}},
                        {
                            "contentBlockDelta": {
                                "delta": {
                                    "citation": {
                                        "title": "First",
                                        "location": {
                                            "web": {"url": "https://example.com/1"}
                                        },
                                    }
                                }
                            }
                        },
                        {"contentBlockDelta": {"delta": {"text": "world"}}},
                        {
                            "contentBlockDelta": {
                                "delta": {
                                    "citation": {
                                        "title": "Second",
                                        "location": {
                                            "web": {"url": "https://example.com/2"}
                                        },
                                    }
                                }
                            }
                        },
                        {"contentBlockStop": {}},
                        {"messageStop": {"stopReason": "end_turn"}},
                        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
                    ]
                ),
                _request(),
            )
        )
        added_payloads = [
            _payload(sse)
            for sse in events
            if sse.event == "response.output_text.annotation.added"
        ]
        assert len(added_payloads) == 2
        assert [payload["annotation_index"] for payload in added_payloads] == [0, 1]
        assert added_payloads[0]["annotation"]["url"] == "https://example.com/1"
        assert added_payloads[1]["annotation"]["url"] == "https://example.com/2"
        assert [payload["annotation"]["start_index"] for payload in added_payloads] == [
            len("Hello "),
            len("Hello world"),
        ], "each citation anchors at the text length streamed so far"

        message = _payload(events[-1])["response"]["output"][-1]
        assert message["content"][0]["text"] == "Hello world"
        annotations = message["content"][0]["annotations"]
        assert [a["url"] for a in annotations] == [
            "https://example.com/1",
            "https://example.com/2",
        ]

    async def test_streaming_citation_after_text_block(self) -> None:
        """A citation arriving after the text block is patched into the final message.

        Bedrock emits ``citationsContent`` after the text block has stopped, so no
        ``annotation.added`` event can be emitted; the annotation is held pending
        and folded into the last message before the terminal event.  A citation
        without a title falls back to its URL.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:_finalize_output_items
        """
        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(
                    [
                        {"contentBlockStart": {"start": {}}},
                        {"contentBlockDelta": {"delta": {"text": "Hello"}}},
                        {"contentBlockStop": {}},
                        {
                            "contentBlockDelta": {
                                "delta": {
                                    "citation": {
                                        "location": {
                                            "web": {"url": "https://late.example"}
                                        }
                                    }
                                }
                            }
                        },
                        {"messageStop": {"stopReason": "end_turn"}},
                        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
                    ]
                ),
                _request(),
            )
        )
        assert "response.output_text.annotation.added" not in _event_names(events)
        by_event = {sse.event: _payload(sse) for sse in events}
        assert by_event["response.content_part.done"]["part"]["annotations"] == [], (
            "the part was already closed when the citation arrived"
        )
        message = _payload(events[-1])["response"]["output"][-1]
        (annotation,) = message["content"][0]["annotations"]
        assert annotation["url"] == "https://late.example"
        assert annotation["title"] == "https://late.example"
        assert annotation["start_index"] == 0


class TestNonStreamOutputShape:
    """Non-streaming output items keep their Bedrock block positions.

    Responses returns a typed ``output`` array rather than ``choices``, and item
    order is part of the contract because callers replay it verbatim on the next
    turn.  Item ids are derived from the block's output index so the
    non-streaming and streaming paths agree.

    Ref: https://developers.openai.com/api/docs/guides/migrate-to-responses
         https://developers.openai.com/api/docs/guides/function-calling#tool-choice
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
         stdapi/models/chat/_adapters/_openai_responses.py:_extract_output_items
    """

    async def test_text_then_tool_use_orders_message_first(self) -> None:
        """A [text, toolUse] response yields [message, function_call].

        Bedrock's ``toolUse.input`` is a JSON object while OpenAI's
        ``function_call.arguments`` is a JSON string, so the adapter serialises it.
        """
        response = await format_response(
            "resp-1",
            1.0,
            "model",
            _bedrock_response(
                [
                    {"text": "Let me check."},
                    {"toolUse": {"toolUseId": "t1", "name": "fn", "input": {"a": 1}}},
                ],
                stop_reason="tool_use",
            ),
            _request(),
        )
        message, call = response.output
        assert isinstance(message, ResponseOutputMessage)
        assert message.id == "resp-1-msg-0"
        assert message.role == "assistant"
        assert message.status == "completed"
        part = message.content[0]
        assert isinstance(part, ResponseOutputText)
        assert part.text == "Let me check."
        assert isinstance(call, ResponseFunctionToolCall)
        assert call.call_id == "t1"
        assert call.id == "resp-1-fc-t1"
        assert call.name == "fn"
        assert json.loads(call.arguments) == {"a": 1}
        assert response.status == "completed", (
            "Bedrock's tool_use stop reason is a normal completion upstream"
        )

    async def test_tool_use_between_text_runs_splits_messages(self) -> None:
        """Each contiguous text run becomes its own message at its position.

        The second message id is ``-msg-2`` rather than ``-msg-1`` because the id
        encodes the item's index in the output array, which the interleaved
        ``function_call`` occupies.
        """
        response = await format_response(
            "resp-1",
            1.0,
            "model",
            _bedrock_response(
                [
                    {"text": "first"},
                    {"toolUse": {"toolUseId": "t1", "name": "fn", "input": {}}},
                    {"text": "second"},
                ],
                stop_reason="tool_use",
            ),
            _request(),
        )
        first, call, second = response.output
        assert isinstance(call, ResponseFunctionToolCall)
        assert call.id == "resp-1-fc-t1"
        assert call.arguments == "{}"
        for message, item_id, text in (
            (first, "resp-1-msg-0", "first"),
            (second, "resp-1-msg-2", "second"),
        ):
            assert isinstance(message, ResponseOutputMessage)
            assert message.id == item_id
            part = message.content[0]
            assert isinstance(part, ResponseOutputText)
            assert part.text == text


class TestWebSearchSources:
    """Web-search sources attach to the nearest preceding web_search_call.

    Bedrock's ``nova_grounding`` system tool reports its results as separate
    ``citationsContent`` blocks with no back-reference to the ``toolUse`` that
    produced them, so the adapter attributes each block to the nearest preceding
    ``web_search_call`` item.

    Ref: https://developers.openai.com/api/docs/guides/tools-web-search
         https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_SystemTool.html
         stdapi/models/chat/_adapters/_openai_responses.py:_attach_web_search_sources
    """

    @staticmethod
    def _citations_block(url: str) -> dict[str, object]:
        """Build a citationsContent block with a single web citation."""
        return {
            "citationsContent": {"citations": [{"location": {"web": {"url": url}}}]}
        }

    @staticmethod
    def _sources(item: object) -> list[str]:
        """Return the source URLs of a web_search_call output item."""
        assert isinstance(item, ResponseFunctionWebSearch)
        assert isinstance(item.action, WebSearchActionSearch)
        assert item.action.sources is not None
        return [source.url for source in item.action.sources]

    async def test_sources_attach_to_nearest_preceding_call(self) -> None:
        """Each citation block feeds the closest preceding web_search_call.

        The tool is passed via ``web_search_tool_names``, which is what turns a
        Bedrock ``toolUse`` into a ``web_search_call`` item instead of a
        ``function_call``.
        """
        response = await format_response(
            "resp-1",
            1.0,
            "model",
            _bedrock_response(
                [
                    {
                        "toolUse": {
                            "toolUseId": "w1",
                            "name": "nova_grounding",
                            "input": {"query": "q1"},
                        }
                    },
                    self._citations_block("https://a.example"),
                    {
                        "toolUse": {
                            "toolUseId": "w2",
                            "name": "nova_grounding",
                            "input": {"query": "q2"},
                        }
                    },
                    self._citations_block("https://b.example"),
                    {"text": "answer"},
                ]
            ),
            _request(),
            web_search_tool_names=frozenset({"nova_grounding"}),
        )
        ws1, ws2, message = response.output
        assert self._sources(ws1) == ["https://a.example"]
        assert self._sources(ws2) == ["https://b.example"]
        assert isinstance(ws1, ResponseFunctionWebSearch)
        assert isinstance(ws2, ResponseFunctionWebSearch)
        assert ws1.id == "resp-1-ws-w1"
        assert ws1.status == "completed"
        assert [
            cast("WebSearchActionSearch", ws.action).query for ws in (ws1, ws2)
        ] == ["q1", "q2"], (
            "the Bedrock toolUse input query is echoed as the search action query"
        )
        assert isinstance(message, ResponseOutputMessage)

    async def test_sources_before_any_call_go_to_first_call(self) -> None:
        """Citations with no preceding call attach to the first call.

        A ``citationsContent`` block can precede every ``toolUse``; rather than
        dropping those sources they are attributed to the first
        ``web_search_call`` of the response.
        """
        response = await format_response(
            "resp-1",
            1.0,
            "model",
            _bedrock_response(
                [
                    self._citations_block("https://early.example"),
                    {
                        "toolUse": {
                            "toolUseId": "w1",
                            "name": "nova_grounding",
                            "input": {"query": "q1"},
                        }
                    },
                    {"text": "answer"},
                ]
            ),
            _request(),
            web_search_tool_names=frozenset({"nova_grounding"}),
        )
        assert self._sources(response.output[0]) == ["https://early.example"]
        assert len(response.output) == 2, (
            "the early citation block must not create an extra output item"
        )


class TestWebSearchLifecycleEvents:
    """The streamed web_search_call lifecycle matches the upstream event sequence.

    Ref: https://developers.openai.com/api/docs/guides/tools-web-search
         https://developers.openai.com/api/reference/resources/responses/streaming-events
         stdapi/models/chat/_adapters/_openai_responses.py:_handle_block_start
         stdapi/models/chat/_adapters/_openai_responses.py:_handle_block_stop
    """

    async def test_in_progress_then_searching_then_completed(self) -> None:
        """in_progress and searching precede completed, in that order.

        A web-search ``toolUse`` block is reported as a ``web_search_call`` item
        rather than a ``function_call``, and emits no
        ``function_call_arguments.*`` events.
        """
        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(
                    [
                        {
                            "contentBlockStart": {
                                "start": {
                                    "toolUse": {
                                        "toolUseId": "w1",
                                        "name": "nova_grounding",
                                    }
                                },
                                "contentBlockIndex": 0,
                            }
                        },
                        {"contentBlockStop": {"contentBlockIndex": 0}},
                        {"messageStop": {"stopReason": "end_turn"}},
                        {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 5}}},
                    ]
                ),
                _request(),
                web_search_tool_names=frozenset({"nova_grounding"}),
            )
        )
        assert _event_names(events) == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
            "response.output_item.done",
            "response.completed",
        ]
        by_event = {sse.event: _payload(sse) for sse in events}
        assert by_event["response.output_item.added"]["item"] == {
            "id": "resp-1-ws-w1",
            "type": "web_search_call",
            "status": "in_progress",
            "action": {"type": "search", "query": ""},
        }
        assert {
            by_event[name]["item_id"]
            for name in (
                "response.web_search_call.in_progress",
                "response.web_search_call.searching",
                "response.web_search_call.completed",
            )
        } == {"resp-1-ws-w1"}
        done_item = by_event["response.output_item.done"]["item"]
        assert done_item["type"] == "web_search_call"
        assert done_item["status"] == "completed"
        assert _payload(events[-1])["response"]["output"] == [done_item]


class TestEmptyToolArguments:
    """Streamed tool calls without input deltas emit `{}` like non-streaming.

    ``function_call.arguments`` is documented as a JSON string, so a parameterless
    tool call must still parse as JSON; Bedrock sends no ``toolUse`` delta at all
    for such a call, which would otherwise leave ``arguments`` empty.

    Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
         stdapi/models/chat/_adapters/_openai_responses.py:_emit_tool_done
    """

    async def test_empty_arguments_default_to_json_object(self) -> None:
        """No argument deltas yields "{}" in done event, item, and response."""
        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(
                    [
                        {
                            "contentBlockStart": {
                                "start": {"toolUse": {"toolUseId": "t1", "name": "fn"}}
                            }
                        },
                        {"contentBlockStop": {}},
                        {"messageStop": {"stopReason": "tool_use"}},
                        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1}}},
                    ]
                ),
                _request(),
            )
        )
        assert "response.function_call_arguments.delta" not in _event_names(events)
        by_event = {sse.event: _payload(sse) for sse in events}
        args_done = by_event["response.function_call_arguments.done"]
        assert args_done["arguments"] == "{}"
        assert args_done["name"] == "fn"
        assert args_done["item_id"] == "resp-1-fc-t1"
        assert by_event["response.output_item.done"]["item"]["arguments"] == "{}"
        assert by_event["response.output_item.done"]["item"]["status"] == "completed"
        final_call = _payload(events[-1])["response"]["output"][-1]
        assert final_call["arguments"] == "{}"
        assert json.loads(final_call["arguments"]) == {}


class TestEchoFields:
    """Request parameters are echoed on the response object.

    ``created_at`` and ``completed_at`` are Unix second timestamps upstream, so the
    float the gateway carries internally must be truncated to ``int`` in both the
    non-streaming object and every streamed snapshot.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         https://developers.openai.com/api/reference/resources/responses/methods/retrieve
         stdapi/models/chat/_adapters/_openai_responses.py:_build_response_object
    """

    async def test_echoed_fields_and_timestamps(self) -> None:
        """instructions, service_tier, cache fields, and timestamps are set.

        ``flex`` is one of the tiers that maps onto a real Bedrock service tier, so
        it is echoed back unchanged rather than collapsing to ``default``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
             https://developers.openai.com/api/docs/guides/prompt-caching
        """
        response = await format_response(
            "resp-1",
            1234.56,
            "model",
            _bedrock_response([{"text": "hi"}]),
            _request(
                instructions="be brief",
                service_tier="flex",
                prompt_cache_key="cache-key",
                prompt_cache_retention="24h",
            ),
        )
        assert response.instructions == "be brief"
        assert response.service_tier == "flex"
        assert response.prompt_cache_key == "cache-key"
        assert response.prompt_cache_retention == "24h"
        assert response.created_at == 1234, (
            "the float created_at is truncated, not rounded"
        )
        assert isinstance(response.created_at, int)
        assert isinstance(response.completed_at, int)
        assert response.object == "response"
        assert response.model == "model"
        assert response.id == "resp-1"
        assert response.tool_choice == "auto", (
            "tool_choice defaults to auto on the response even when unset"
        )
        assert response.parallel_tool_calls is True

    async def test_streamed_created_at_is_int(self) -> None:
        """Streaming lifecycle and terminal events carry an int created_at.

        Every streamed snapshot rebuilds the Response object, so the truncation
        must hold for the lifecycle events as well as the terminal one.
        """
        events = await _collect(
            format_stream(
                "resp-1", 1234.56, "model", _stream(_text_stream_events()), _request()
            )
        )
        snapshots = [
            payload["response"]
            for payload in (_payload(sse) for sse in events)
            if "response" in payload
        ]
        assert len(snapshots) == 3, (
            "response.created, response.in_progress and the terminal event carry snapshots"
        )
        assert [snapshot["created_at"] for snapshot in snapshots] == [1234, 1234, 1234]
        assert all(isinstance(snapshot["created_at"], int) for snapshot in snapshots)
        assert "completed_at" not in snapshots[0], (
            "an in_progress snapshot has no completion timestamp"
        )
        assert snapshots[-1]["completed_at"] >= 1234


class TestStreamedModeration:
    """The streamed terminal event carries the moderation field.

    ``moderation`` is a gateway extension with no upstream analogue: the Bedrock
    guardrail trace only arrives on the trailing ``metadata`` event, so it is
    captured into ``GUARDRAIL_TRACE_VAR`` and mapped at stream end.  Note the
    trace's two directions have different shapes — ``inputAssessment`` maps a
    guardrail id to one assessment object while ``outputAssessments`` maps it to a
    list.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_GuardrailTraceAssessment.html
         https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-content-filters.html
         stdapi/routes/_moderation.py:build_response_moderation
         stdapi/models/chat/_adapters/_openai_responses.py:format_stream
    """

    #: Guardrail trace carried by the trailing metadata event.
    _TRACE: ClassVar[dict[str, Any]] = {
        "inputAssessment": {
            "gr123": {
                "contentPolicy": {
                    "filters": [
                        {"type": "HATE", "confidence": "HIGH", "action": "BLOCKED"}
                    ]
                }
            }
        },
        "outputAssessments": {
            "gr123": [
                {
                    "contentPolicy": {
                        "filters": [
                            {"type": "VIOLENCE", "confidence": "LOW", "action": "NONE"}
                        ]
                    }
                }
            ]
        },
    }

    async def test_moderation_in_completed_event(self) -> None:
        """The moderation builder result lands on the response.completed payload.

        The builder is invoked once at stream end (after the trailing metadata
        event), not per streamed event.
        """
        result = ModerationResult(
            flagged=False,
            categories={},
            category_scores={},
            category_applied_input_types={},
            model="gr123",
        )
        moderation = ResponseModeration(input=result, output=result)
        calls = 0

        def _builder() -> ResponseModeration:
            nonlocal calls
            calls += 1
            return moderation

        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(_text_stream_events()),
                _request(),
                moderation_builder=_builder,
            )
        )
        assert calls == 1, "the builder runs once, at stream end"
        assert events[-1].event == "response.completed"
        payload = _payload(events[-1])
        assert payload["response"]["moderation"]["input"]["model"] == "gr123"
        assert payload["response"]["moderation"]["output"]["flagged"] is False
        assert all(
            "moderation" not in _payload(sse).get("response", {}) for sse in events[:-1]
        ), "only the terminal snapshot carries moderation"

    async def test_moderation_built_from_captured_stream_trace(self) -> None:
        """The real builder maps the trace captured from the metadata event.

        Bedrock reports confidence levels rather than scores, so the gateway
        derives ``category_scores`` from a fixed HIGH→0.75 / LOW→0.25 table, and a
        ``BLOCKED`` action is what sets ``flagged``.

        Ref: https://stdapi.ai/api_openai_moderations/
             stdapi/models/__init__.py:ModelBase._capture_stream_usage
        """
        stream_events = _text_stream_events()
        stream_events[-1]["metadata"]["trace"] = {"guardrail": self._TRACE}  # type: ignore[index]
        holder: dict[str, Any] = {}
        token = GUARDRAIL_TRACE_VAR.set(holder)
        try:
            events = await _collect(
                format_stream(
                    "resp-1",
                    1.0,
                    "model",
                    ModelBase("model")._capture_stream_usage(  # noqa: SLF001
                        _stream(stream_events)
                    ),
                    _request(),
                    moderation_builder=partial(
                        build_response_moderation, RequestModeration(model="gr123")
                    ),
                )
            )
        finally:
            GUARDRAIL_TRACE_VAR.reset(token)
        assert holder == self._TRACE
        moderation = _payload(events[-1])["response"]["moderation"]
        assert moderation["input"]["model"] == "gr123"
        assert moderation["input"]["flagged"] is True
        assert moderation["input"]["categories"] == {"hate": True}
        assert moderation["input"]["category_scores"] == {"hate": 0.75}
        assert moderation["output"]["flagged"] is False
        assert moderation["output"]["categories"] == {"violence": False}
        assert moderation["output"]["category_scores"] == {"violence": 0.25}


class TestPolicySwitches:
    """safety_identifier and stream_options are accepted and ignored.

    The gateway rejects ``context_management``, ``conversation``,
    ``max_tool_calls`` and ``truncation`` with a 400, but deliberately keeps
    ``background``, ``safety_identifier`` and ``stream_options`` out of that set so
    OpenAI clients that always send them keep working; they have no Bedrock
    equivalent and are dropped.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/types/openai_responses.py:ResponseCreateParams
    """

    def test_safety_identifier_accepted(self) -> None:
        """safety_identifier binds to its declared field and is not unsupported."""
        request = _request(safety_identifier="user-1")
        assert request.safety_identifier == "user-1"
        assert "safety_identifier" in request.model_fields_set, (
            "the value must bind to the declared field, not be swallowed as an extra"
        )
        assert "safety_identifier" not in ResponseCreateParams._UNSUPPORTED  # noqa: SLF001

    def test_stream_options_accepted(self) -> None:
        """stream_options binds to its declared field and is not unsupported."""
        request = _request(stream=True, stream_options={"include_obfuscation": False})
        assert request.stream_options is not None
        assert request.stream_options.include_obfuscation is False
        assert "stream_options" in request.model_fields_set
        assert "stream_options" not in ResponseCreateParams._UNSUPPORTED  # noqa: SLF001


class _StubImageResult:
    """Fake ``generate_images()`` result item carrying a base64 payload."""

    def __init__(self, image: str) -> None:
        self.image = image


class _StubImageJob:
    """Fake image generation job returning or raising a canned outcome."""

    def __init__(self, image: str | None, raises: Exception | None) -> None:
        self._image = image
        self._raises = raises

    async def generate_images(self) -> list[_StubImageResult]:
        """Return the canned image, or raise the canned exception."""
        if self._raises:
            raise self._raises
        return [_StubImageResult(self._image)] if self._image else []


class _StubImageModel:
    """Fake image model recording ``get_image_generation_job`` call kwargs."""

    def __init__(
        self, image: str | None = "ZmFrZQ==", raises: Exception | None = None
    ) -> None:
        self.image = image
        self.raises = raises
        self.calls: list[dict[str, object]] = []

    def get_image_generation_job(self, **kwargs: object) -> _StubImageJob:
        """Record kwargs and return a job yielding the canned outcome."""
        self.calls.append(kwargs)
        return _StubImageJob(self.image, self.raises)


class _StubValidatedModel:
    """Fake ``ModelDetails`` exposing only the ``.id`` attribute used downstream."""

    id = "stub-model"


async def _stub_validate_model(
    *_args: object, **_kwargs: object
) -> _StubValidatedModel:
    """Stand in for ``validate_model`` without any AWS lookup."""
    return _StubValidatedModel()


def _image_tool_call(arguments: dict[str, object]) -> ResponseFunctionToolCall:
    """Build an ``image_generation`` function-call output item."""
    return ResponseFunctionToolCall(
        type="function_call",
        call_id="call_1",
        name="image_generation",
        arguments=json.dumps(arguments),
    )


class TestImageGenerationExecution:
    """execute_image_generation_calls tolerates malformed input and errors.

    ``image_generation`` is not a hosted tool here: the gateway presents it to the
    model as a synthetic function tool and runs the generation itself, so the
    arguments are model-authored JSON that may be malformed, and a failure must
    degrade to a ``failed`` item rather than failing the whole response.

    Ref: https://developers.openai.com/api/docs/guides/tools-image-generation
         stdapi/models/chat/_adapters/_openai_responses.py:execute_image_generation_calls
         stdapi/models/chat/_adapters/_openai_responses.py:get_image_generation_tool
    """

    async def test_malformed_size_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed size argument falls back to the default 1024x1024 size."""
        stub_model = _StubImageModel()
        monkeypatch.setattr(responses_adapter, "validate_model", _stub_validate_model)
        monkeypatch.setattr(responses_adapter, "get_image_model", lambda _: stub_model)
        item = _image_tool_call({"prompt": "a cat", "size": "not-a-size"})
        result = await execute_image_generation_calls(
            [item], ImageGeneration(type="image_generation"), "resp-1", "fallback-model"
        )
        assert len(stub_model.calls) == 1
        assert stub_model.calls[0]["width"] == 1024
        assert stub_model.calls[0]["height"] == 1024
        assert stub_model.calls[0]["prompt"] == "a cat", (
            "the model-authored prompt must survive the size fallback"
        )
        assert stub_model.calls[0]["count"] == 1
        assert stub_model.calls[0]["output_format"] == "png"
        assert len(result) == 1
        assert isinstance(result[0], ImageGenerationCall)
        assert result[0].id == "resp-1-img-1"
        assert result[0].type == "image_generation_call"
        assert result[0].status == "completed"
        assert result[0].result == "ZmFrZQ==", (
            "the generated base64 payload is returned on the item"
        )

    async def test_model_guessed_quality_is_not_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Quality comes from the tool definition, never from the model's arguments.

        Most image models have no quality control, so a value the model
        invented would be a request the caller never made; the caller's own
        tool-level quality is an explicit ask and still reaches the job.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._validate_no_quality
        """
        stub_model = _StubImageModel()
        monkeypatch.setattr(responses_adapter, "validate_model", _stub_validate_model)
        monkeypatch.setattr(responses_adapter, "get_image_model", lambda _: stub_model)
        advertised = responses_adapter._IMAGE_GENERATION_SCHEMA["properties"]  # noqa: SLF001
        assert isinstance(advertised, dict)
        assert "quality" not in advertised, (
            "the model must not be offered a knob most image models reject"
        )

        item = _image_tool_call({"prompt": "a cat", "quality": "high"})
        await execute_image_generation_calls(
            [item], ImageGeneration(type="image_generation"), "resp-1", "fallback-model"
        )
        assert stub_model.calls[0]["quality"] is None

        item = _image_tool_call({"prompt": "a cat", "quality": "high"})
        await execute_image_generation_calls(
            [item],
            ImageGeneration(type="image_generation", quality="low"),
            "resp-1",
            "fallback-model",
        )
        assert stub_model.calls[1]["quality"] == "low"

    async def test_generation_error_produces_failed_item(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A generation failure is caught and yields a status="failed" item.

        The failure is reported at ``warning`` level only: the surrounding response
        still completes.
        """
        stub_model = _StubImageModel(image=None, raises=RuntimeError("boom"))
        logged: list[tuple[tuple[object, ...], dict[str, object]]] = []
        monkeypatch.setattr(responses_adapter, "validate_model", _stub_validate_model)
        monkeypatch.setattr(responses_adapter, "get_image_model", lambda _: stub_model)
        monkeypatch.setattr(
            responses_adapter,
            "log_error_details",
            lambda *args, **kwargs: logged.append((args, kwargs)),
        )
        item = _image_tool_call({"prompt": "a cat"})
        result = await execute_image_generation_calls(
            [item], ImageGeneration(type="image_generation"), "resp-1", "fallback-model"
        )
        assert isinstance(result[0], ImageGenerationCall)
        assert result[0].id == "resp-1-img-1"
        assert result[0].status == "failed"
        assert result[0].result is None
        assert len(logged) == 1
        message = logged[0][0][0]
        assert isinstance(message, str)
        assert "image_generation tool call failed" in message
        assert "boom" in message, "the underlying error must reach the request log"
        assert logged[0][1] == {"level": "warning"}


class TestImageGenerationStreamEvents:
    """The streaming image post-handler emits OpenAI lifecycle events.

    Generation runs gateway-side after the Bedrock stream has finished, so the
    ``image_generation_call`` events are replayed from the suppressed tool call
    before the terminal event, keeping the upstream
    in_progress → generating → completed order.

    Ref: https://developers.openai.com/api/docs/guides/tools-image-generation
         https://developers.openai.com/api/reference/resources/responses/streaming-events
         stdapi/models/chat/_adapters/_openai_responses.py:image_generation_stream_handler
    """

    async def _run_handler(
        self, stub_model: _StubImageModel, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[Any, list[JSONServerSentEvent]]:
        """Run the post-stream handler over one suppressed image call."""
        monkeypatch.setattr(responses_adapter, "validate_model", _stub_validate_model)
        monkeypatch.setattr(responses_adapter, "get_image_model", lambda _: stub_model)
        state = responses_adapter._StreamState("resp-1")  # noqa: SLF001
        state.suppressed_tool_calls.append(
            ("t1", "image_generation", '{"prompt": "a cat"}')
        )
        events = [
            sse
            async for sse in responses_adapter.image_generation_stream_handler(
                state,
                ImageGeneration(type="image_generation"),
                "resp-1",
                "fallback-model",
            )
        ]
        return state, events

    async def test_completed_call_emits_lifecycle_events(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful generation emits in_progress, generating, completed.

        The materialised item is also appended to ``state.output_items`` so it
        reaches the terminal ``response.completed`` snapshot, and
        ``state.output_index`` advances for any item that follows.
        """
        state, events = await self._run_handler(_StubImageModel(), monkeypatch)
        assert _event_names(events) == [
            "response.output_item.added",
            "response.image_generation_call.in_progress",
            "response.image_generation_call.generating",
            "response.image_generation_call.completed",
            "response.output_item.done",
        ]
        payloads = [_payload(sse) for sse in events]
        assert payloads[0]["item"]["status"] == "in_progress"
        assert payloads[0]["item"]["id"] == "resp-1-img-1"
        assert "result" not in payloads[0]["item"]
        for payload in payloads[1:4]:
            assert payload["item_id"] == "resp-1-img-1"
        assert payloads[4]["item"]["status"] == "completed"
        assert payloads[4]["item"]["result"] == "ZmFrZQ=="
        assert [payload["sequence_number"] for payload in payloads] == list(range(5))
        assert {payload["output_index"] for payload in payloads} == {0}
        (item,) = state.output_items
        assert isinstance(item, ImageGenerationCall)
        assert (item.id, item.status, item.result) == (
            "resp-1-img-1",
            "completed",
            "ZmFrZQ==",
        )
        assert state.output_index == 1

    async def test_failed_call_skips_completed_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed generation emits no image_generation_call.completed event.

        The item is still added and closed, so a client sees a terminated
        ``image_generation_call`` with ``status="failed"`` and no result.
        """
        state, events = await self._run_handler(
            _StubImageModel(image=None, raises=RuntimeError("boom")), monkeypatch
        )
        assert _event_names(events) == [
            "response.output_item.added",
            "response.image_generation_call.in_progress",
            "response.image_generation_call.generating",
            "response.output_item.done",
        ]
        done_item = _payload(events[-1])["item"]
        assert done_item["status"] == "failed"
        assert done_item["id"] == "resp-1-img-1"
        assert "result" not in done_item
        (item,) = state.output_items
        assert isinstance(item, ImageGenerationCall)
        assert item.status == "failed"
        assert item.result is None


class _BarrierImageModel:
    """Fake image model whose jobs all block until every expected job started."""

    def __init__(
        self, expected: int, fail_prompts: frozenset[str] = frozenset()
    ) -> None:
        self.expected = expected
        self.fail_prompts = fail_prompts
        self.started: list[str] = []
        self.all_started = Event()

    def get_image_generation_job(self, **kwargs: object) -> object:
        """Return a job that only finishes once all expected jobs started."""
        prompt = str(kwargs["prompt"])
        model = self

        class _Job:
            async def generate_images(self) -> list[_StubImageResult]:
                model.started.append(prompt)
                if len(model.started) >= model.expected:
                    model.all_started.set()
                # Times out (instead of hanging) if generations are sequential.
                await wait_for(model.all_started.wait(), timeout=5)
                if prompt in model.fail_prompts:
                    msg = f"boom-{prompt}"
                    raise RuntimeError(msg)
                return [_StubImageResult(f"b64-{prompt}")]

        return _Job()


class TestImageGenerationConcurrency:
    """Multiple image_generation calls in one response run concurrently.

    Each stubbed generation blocks until every expected generation has
    started, so these tests fail (time out) if the adapter regresses to
    one-at-a-time awaits. The ordering contracts must hold unchanged: result
    items and stream events keep the original call order, and a single failed
    generation degrades to its own ``failed`` item without touching siblings.

    Ref: https://developers.openai.com/api/docs/guides/tools-image-generation
         stdapi/models/chat/_adapters/_openai_responses.py:execute_image_generation_calls
         stdapi/models/chat/_adapters/_openai_responses.py:image_generation_stream_handler
    """

    async def test_non_stream_generations_overlap_and_keep_item_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concurrent generations land on their own items in item order.

        A failing middle generation must not disturb its siblings' results or
        the stable per-item IDs, and non-image items pass through in place.
        """
        stub_model = _BarrierImageModel(3, fail_prompts=frozenset({"two"}))
        logged: list[str] = []
        monkeypatch.setattr(responses_adapter, "validate_model", _stub_validate_model)
        monkeypatch.setattr(responses_adapter, "get_image_model", lambda _: stub_model)
        monkeypatch.setattr(
            responses_adapter,
            "log_error_details",
            lambda message, **_: logged.append(message),
        )
        passthrough = ResponseFunctionToolCall(
            type="function_call", call_id="call_x", name="other_tool", arguments="{}"
        )
        items: list[ResponseOutputItem] = [
            _image_tool_call({"prompt": "one"}),
            passthrough,
            _image_tool_call({"prompt": "two"}),
            _image_tool_call({"prompt": "three"}),
        ]

        result = await wait_for(
            execute_image_generation_calls(
                items,
                ImageGeneration(type="image_generation"),
                "resp-1",
                "fallback-model",
            ),
            timeout=5,
        )

        assert sorted(stub_model.started) == ["one", "three", "two"]
        assert result[1] is passthrough, "non-image items must pass through in place"
        first, _, second, third = result
        assert isinstance(first, ImageGenerationCall)
        assert isinstance(second, ImageGenerationCall)
        assert isinstance(third, ImageGenerationCall)
        assert [first.id, second.id, third.id] == [
            "resp-1-img-1",
            "resp-1-img-2",
            "resp-1-img-3",
        ]
        assert (first.status, first.result) == ("completed", "b64-one")
        assert (second.status, second.result) == ("failed", None)
        assert (third.status, third.result) == ("completed", "b64-three")
        assert [message for message in logged if "boom-two" in message], (
            "the failed sibling must still be logged"
        )

    async def test_stream_generations_overlap_and_keep_event_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Eagerly started generations emit their events strictly in call order.

        Concurrency must not show: the per-item event sequence, output indices
        and sequence numbers are those of a strictly sequential emission.
        """
        stub_model = _BarrierImageModel(2)
        monkeypatch.setattr(responses_adapter, "validate_model", _stub_validate_model)
        monkeypatch.setattr(responses_adapter, "get_image_model", lambda _: stub_model)
        state = responses_adapter._StreamState("resp-1")  # noqa: SLF001
        state.suppressed_tool_calls.extend(
            [
                ("t1", "image_generation", '{"prompt": "one"}'),
                ("t2", "other_tool", "{}"),
                ("t3", "image_generation", '{"prompt": "two"}'),
            ]
        )

        async def _collect() -> list[JSONServerSentEvent]:
            return [
                sse
                async for sse in responses_adapter.image_generation_stream_handler(
                    state,
                    ImageGeneration(type="image_generation"),
                    "resp-1",
                    "fallback-model",
                )
            ]

        events = await wait_for(_collect(), timeout=5)

        item_lifecycle = [
            "response.output_item.added",
            "response.image_generation_call.in_progress",
            "response.image_generation_call.generating",
            "response.image_generation_call.completed",
            "response.output_item.done",
        ]
        assert _event_names(events) == item_lifecycle * 2
        payloads = [_payload(sse) for sse in events]
        assert [payload["sequence_number"] for payload in payloads] == list(range(10))
        assert [payload["output_index"] for payload in payloads] == [0] * 5 + [1] * 5
        assert payloads[4]["item"]["id"] == "resp-1-img-1"
        assert payloads[4]["item"]["result"] == "b64-one"
        assert payloads[9]["item"]["id"] == "resp-1-img-2"
        assert payloads[9]["item"]["result"] == "b64-two"
        assert [item.id for item in state.output_items] == [
            "resp-1-img-1",
            "resp-1-img-2",
        ], "materialised items must keep the call order"
        assert state.output_index == 2


#: A syntactically valid vector store identifier, so `parse_store_id` accepts it.
_STORE_ID = "vs_" + "0" * 26


def _stream_search_result(text: str, score: float = 0.9) -> SearchResult:
    """Build one vector store hit the stubbed search answers with."""
    return SearchResult(
        file_id="file-1", filename="file-1.md", score=score, text=text, attributes={}
    )


@pytest.fixture
def cleanups() -> Generator[list[Awaitable[None]]]:
    """Bind the request cleanup list the retrieval loop defers store writes to."""
    pending: list[Awaitable[None]] = []
    token = CLEANUPS.set(pending)
    yield pending
    for task in pending:
        task.close()  # type: ignore[attr-defined]
    CLEANUPS.reset(token)


def _file_search_stream_events() -> list[dict[str, object]]:
    """Build a Bedrock stream whose only block is a ``file_search`` tool call."""
    return [
        {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": "tu1", "name": "file_search"}},
                "contentBlockIndex": 0,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": '{"query": "vacation days"}'}},
                "contentBlockIndex": 0,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 4}}},
    ]


class _StubGroundedModel:
    """Chat model answering the continued turn from a fabricated Converse reply."""

    def __init__(self, answer: str = "Twenty-five days.") -> None:
        self.answer = answer
        self.inputs: list[Any] = []

    async def create_response(
        self,
        request: ResponseCreateParams,
        response_id: str,
        created_at: float,
        **_kwargs: object,
    ) -> Response:
        """Record the continued input and answer it with a text message."""
        self.inputs.append(request.input)
        return await format_response(
            response_id,
            created_at,
            "amazon.nova-2-lite-v1:0",
            _bedrock_response(
                [{"text": self.answer}], usage={"inputTokens": 7, "outputTokens": 3}
            ),
            request,
        )


class _StubSearchingModel:
    """Chat model that answers every continued turn with one more search."""

    def __init__(self) -> None:
        self.turns = 0

    async def create_response(
        self,
        request: ResponseCreateParams,
        response_id: str,
        created_at: float,
        **_kwargs: object,
    ) -> Response:
        """Answer with a ``file_search`` tool call instead of a message."""
        self.turns += 1
        return await format_response(
            response_id,
            created_at,
            "amazon.nova-2-lite-v1:0",
            _bedrock_response(
                [
                    {
                        "toolUse": {
                            "toolUseId": f"tu-{self.turns}",
                            "name": "file_search",
                            "input": {"query": "again"},
                        }
                    }
                ],
                stop_reason="tool_use",
                usage={"inputTokens": 1, "outputTokens": 1},
            ),
            request,
        )


def _stub_streamed_file_search(
    monkeypatch: pytest.MonkeyPatch, results: list[SearchResult]
) -> _StubGroundedModel:
    """Stub the vector store and the model that answers the retrieved passages."""
    grounded = _StubGroundedModel()

    async def _read_store(store_id: str) -> StoreRecord:
        return StoreRecord(
            id=store_id,
            created_at=0,
            last_active_at=0,
            embedding_model="amazon.titan-embed-text-v2:0",
            dimensions=8,
        )

    async def _search(
        _store: StoreRecord, _queries: Sequence[str], **_kwargs: object
    ) -> list[SearchResult]:
        return results

    async def _touch_store(_store: StoreRecord) -> None:
        return None

    monkeypatch.setattr(responses_adapter, "read_store", _read_store)
    monkeypatch.setattr(responses_adapter, "search", _search)
    monkeypatch.setattr(responses_adapter, "touch_store", _touch_store)
    monkeypatch.setattr(models_chat, "get_chat_model", lambda _model_id: grounded)
    return grounded


def _file_search_request(**fields: object) -> ResponseCreateParams:
    """Build a streamed request carrying a ``file_search`` tool."""
    return _request(
        stream=True,
        tools=[{"type": "file_search", "vector_store_ids": [_STORE_ID]}],
        **fields,
    )


@pytest.mark.usefixtures("cleanups")
class TestFileSearchStreamEvents:
    """A streamed retrieval reports the search, then streams the grounded answer.

    The model's ``file_search`` call never reaches the client as a function
    call: it is suppressed, answered from the vector stores as a
    ``file_search_call`` item with its three lifecycle events, and the turn
    continues inside the same stream.

    Ref: https://developers.openai.com/api/docs/guides/tools-file-search
         https://developers.openai.com/api/reference/resources/responses/streaming-events
         openai.types.responses.response_file_search_call_searching_event.ResponseFileSearchCallSearchingEvent
         stdapi/models/chat/_adapters/_openai_responses.py:file_search_stream_handler
    """

    async def test_event_sequence_of_a_streamed_retrieval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The three file search events frame the item, and the answer follows.

        Sequence numbers stay contiguous across the retrieval and the continued
        turn, and the terminal snapshot carries both output items.
        """
        _stub_streamed_file_search(
            monkeypatch, [_stream_search_result("Employees receive 25 days.")]
        )

        events = await _collect(
            format_stream(
                "resp-1",
                0.0,
                "amazon.nova-2-lite-v1:0",
                _stream(_file_search_stream_events()),
                _file_search_request(),
            )
        )

        assert _event_names(events) == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.file_search_call.in_progress",
            "response.file_search_call.searching",
            "response.file_search_call.completed",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
        assert _sequence_numbers(events) == list(range(len(events)))
        payloads = [_payload(sse) for sse in events]
        assert payloads[2]["item"]["status"] == "in_progress"
        assert payloads[2]["item"]["queries"] == ["vacation days"]
        assert payloads[6]["item"]["status"] == "completed"
        assert [payload["output_index"] for payload in payloads[2:7]] == [0] * 5
        assert [payload["output_index"] for payload in payloads[7:13]] == [1] * 6
        snapshot = payloads[-1]["response"]
        assert [item["type"] for item in snapshot["output"]] == [
            "file_search_call",
            "message",
        ]
        assert snapshot["output"][1]["content"][0]["text"] == "Twenty-five days."
        assert snapshot["status"] == "completed"

    async def test_passages_reach_the_continued_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The continued invocation reads the retrieved passages as a tool result.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:_map_file_search_call
        """
        grounded = _stub_streamed_file_search(
            monkeypatch, [_stream_search_result("Employees receive 25 days.")]
        )

        await _collect(
            format_stream(
                "resp-1",
                0.0,
                "amazon.nova-2-lite-v1:0",
                _stream(_file_search_stream_events()),
                _file_search_request(),
            )
        )

        (continued,) = grounded.inputs
        assert isinstance(continued, list)
        answered = continued[-1]
        assert answered.type == "file_search_call"
        assert answered.queries == ["vacation days"]
        assert "Employees receive 25 days." in json.dumps(answered.results)

    async def test_results_are_reported_only_when_included(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``include=["file_search_call.results"]`` is what attaches the passages.

        Ref: https://developers.openai.com/api/docs/guides/tools-file-search
        """
        _stub_streamed_file_search(
            monkeypatch, [_stream_search_result("Employees receive 25 days.")]
        )

        silent = await _collect(
            format_stream(
                "resp-1",
                0.0,
                "amazon.nova-2-lite-v1:0",
                _stream(_file_search_stream_events()),
                _file_search_request(),
            )
        )
        included = await _collect(
            format_stream(
                "resp-1",
                0.0,
                "amazon.nova-2-lite-v1:0",
                _stream(_file_search_stream_events()),
                _file_search_request(include=["file_search_call.results"]),
            )
        )

        assert "results" not in _payload(silent[6])["item"]
        (result,) = _payload(included[6])["item"]["results"]
        assert result["text"] == "Employees receive 25 days."
        assert result["file_id"] == "file-1"
        assert result["score"] == 0.9

    async def test_usage_covers_both_invocations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The terminal event bills the retrieval turn and the grounded answer.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:_accumulate_stream_usage
        """
        _stub_streamed_file_search(monkeypatch, [])

        events = await _collect(
            format_stream(
                "resp-1",
                0.0,
                "amazon.nova-2-lite-v1:0",
                _stream(_file_search_stream_events()),
                _file_search_request(),
            )
        )

        usage = _payload(events[-1])["response"]["usage"]
        assert usage["input_tokens"] == 17, "10 for the search turn, 7 for the answer"
        assert usage["output_tokens"] == 7
        assert usage["total_tokens"] == 24

    async def test_a_model_that_keeps_searching_is_stopped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The retrieval loop is bounded, and the last call is reported incomplete.

        A model asking for one more search on every turn would otherwise loop
        without end; the last request is answered with an ``incomplete`` item
        and no further invocation.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:_stream_unanswered_file_search
        """
        _stub_streamed_file_search(monkeypatch, [])
        searching = _StubSearchingModel()
        monkeypatch.setattr(models_chat, "get_chat_model", lambda _id: searching)

        events = await _collect(
            format_stream(
                "resp-1",
                0.0,
                "amazon.nova-2-lite-v1:0",
                _stream(_file_search_stream_events()),
                _file_search_request(),
            )
        )

        assert _event_names(events).count("response.file_search_call.searching") == 3
        assert _event_names(events).count("response.file_search_call.completed") == 2, (
            "only the searches that ran report a completion"
        )
        assert searching.turns == 2, "the loop stops at the round limit"
        snapshot = _payload(events[-1])["response"]
        assert [item["status"] for item in snapshot["output"]] == [
            "completed",
            "completed",
            "incomplete",
        ]
