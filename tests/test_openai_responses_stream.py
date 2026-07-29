"""Unit tests for OpenAI Responses API output, streaming, and usage semantics."""

import json
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, cast

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
from sse_starlette import JSONServerSentEvent

from stdapi import monitoring
from stdapi.aws_bedrock import GUARDRAIL_TRACE_VAR
from stdapi.models import ModelBase
from stdapi.models.chat._adapters import _openai_chat_completion as chat_adapter
from stdapi.models.chat._adapters import _openai_responses as responses_adapter
from stdapi.models.chat._adapters._openai_common import extract_stream_usage
from stdapi.models.chat._adapters._openai_responses import (
    execute_image_generation_calls,
    format_response,
    format_stream,
)
from stdapi.monitoring import REQUEST_ID, REQUEST_LOG, SseHandledStreamError
from stdapi.routes._moderation import build_response_moderation
from stdapi.types.openai import ModerationResult, RequestModeration, ResponseModeration
from stdapi.types.openai_responses import (
    AnnotationURLCitation,
    ImageGeneration,
    ImageGenerationCall,
    ResponseCreateParams,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseOutputMessage,
    ResponseOutputText,
    WebSearchActionSearch,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from types_aiobotocore_bedrock_runtime.type_defs import (
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
    )

    from stdapi.types.openai_chat_completions import ChatCompletion

#: Mark the whole module as local (in-process, no AWS calls).
pytestmark = pytest.mark.local


@pytest.fixture(autouse=True)
def _request_log() -> Generator[None]:
    """Provide the request log context required by response logging."""
    token = REQUEST_LOG.set({"level": "info"})  # type: ignore[typeddict-item]
    yield
    REQUEST_LOG.reset(token)


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


class TestTerminalEvents:
    """The terminal stream event matches the Bedrock stop reason."""

    async def test_completed_on_end_turn(self) -> None:
        """A normal stop emits response.completed as the final event."""
        events = await _collect(
            format_stream(
                "resp-1", 1.0, "model", _stream(_text_stream_events()), _request()
            )
        )
        assert events[-1].event == "response.completed"
        payload = _payload(events[-1])
        assert payload["response"]["status"] == "completed"
        assert "error" not in payload["response"]
        sdk_event = SDKResponseCompletedEvent.model_validate(payload)
        assert sdk_event.sequence_number == payload["sequence_number"]

    async def test_incomplete_on_max_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hitting max tokens ends with response.incomplete, no response.completed, and no warning."""
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
        assert events[-1].event == "response.incomplete"
        assert all(sse.event != "response.completed" for sse in events)
        payload = _payload(events[-1])
        assert payload["type"] == "response.incomplete"
        assert payload["response"]["status"] == "incomplete"
        assert payload["response"]["incomplete_details"] == {
            "reason": "max_output_tokens"
        }
        assert logged == []
        sdk_event = SDKResponseIncompleteEvent.model_validate(payload)
        assert sdk_event.sequence_number == payload["sequence_number"]

    async def test_incomplete_on_guardrail(self) -> None:
        """A guardrail stop maps to incomplete_details.reason content_filter."""
        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(_text_stream_events(stop_reason="guardrail_intervened")),
                _request(),
            )
        )
        assert events[-1].event == "response.incomplete"
        payload = _payload(events[-1])
        assert payload["response"]["incomplete_details"] == {"reason": "content_filter"}

    async def test_incomplete_on_content_filtered(self) -> None:
        """A content_filtered stop maps to incomplete_details.reason content_filter."""
        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(_text_stream_events(stop_reason="content_filtered")),
                _request(),
            )
        )
        assert events[-1].event == "response.incomplete"
        payload = _payload(events[-1])
        assert payload["response"]["incomplete_details"] == {"reason": "content_filter"}

    async def test_incomplete_on_unknown_reason_logs_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A genuinely unknown stop reason maps to max_output_tokens and logs a warning."""
        logged: list[tuple[object, ...]] = []
        monkeypatch.setattr(
            responses_adapter, "log_error_details", lambda *a, **_kw: logged.append(a)
        )
        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(_text_stream_events(stop_reason="something_new")),
                _request(),
            )
        )
        assert events[-1].event == "response.incomplete"
        payload = _payload(events[-1])
        assert payload["response"]["incomplete_details"] == {
            "reason": "max_output_tokens"
        }
        assert len(logged) == 1
        message = logged[0][0]
        assert isinstance(message, str)
        assert "something_new" in message

    async def test_failed_on_malformed_output(self) -> None:
        """A malformed model output ends with response.failed carrying an error."""
        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(_text_stream_events(stop_reason="malformed_model_output")),
                _request(),
            )
        )
        assert events[-1].event == "response.failed"
        assert all(sse.event != "response.completed" for sse in events)
        payload = _payload(events[-1])
        assert payload["response"]["status"] == "failed"
        error = payload["response"]["error"]
        assert error["code"] == "server_error"
        assert "malformed_model_output" in error["message"]
        sdk_event = SDKResponseFailedEvent.model_validate(payload)
        assert sdk_event.sequence_number == payload["sequence_number"]


class TestFailedResponseError:
    """Failed non-streaming responses carry a populated error object."""

    async def test_error_populated_on_failed_status(self) -> None:
        """A malformed_tool_use stop reason yields status failed with error."""
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
        assert "malformed_tool_use" in response.error.message
        assert response.completed_at is None

    async def test_no_error_on_completed(self) -> None:
        """A completed response has no error object."""
        response = await format_response(
            "resp-1", 1.0, "model", _bedrock_response([{"text": "hi"}]), _request()
        )
        assert response.status == "completed"
        assert response.error is None


class TestMidStreamErrors:
    """Mid-stream exceptions emit spec events and a single log record."""

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
        """An unexpected exception emits error + response.failed, then re-raises."""
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

        error_sse = events[-2]
        assert error_sse.event == "error"
        error_payload = _payload(error_sse)
        assert error_payload["type"] == "error"
        assert error_payload["code"] == "server_error"
        assert error_payload["message"] == "Internal Server Error"
        assert "sequence_number" in error_payload
        SDKResponseErrorEvent.model_validate(error_payload)

        failed_payload = _payload(events[-1])
        assert events[-1].event == "response.failed"
        assert failed_payload["response"]["status"] == "failed"
        assert failed_payload["response"]["error"]["code"] == "server_error"
        SDKResponseFailedEvent.model_validate(failed_payload)

    async def test_api_error_carries_code_and_param(self) -> None:
        """An ApiError surfaces its message, code, and param in the error event."""
        from stdapi.api_errors import ApiError  # noqa: PLC0415

        exc = ApiError("bad tool", status=400)
        exc.code = "invalid_value"
        exc.param = "tools"
        events, error = await self._drain_until_error(
            format_stream("resp-1", 1.0, "model", self._failing_stream(exc), _request())
        )
        assert error.status == 400
        assert error.level == "warning"
        error_payload = _payload(events[-2])
        assert error_payload["message"] == "bad tool"
        assert error_payload["code"] == "invalid_value"
        assert error_payload["param"] == "tools"

    async def test_wrapper_logs_once_without_legacy_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """log_request_sse_stream_event records once and emits no legacy error."""
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
        assert error_sse.event == "error"
        assert events[-1].event == "response.failed"
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
    """Cache buckets are folded into input/prompt tokens (OpenAI semantics)."""

    #: Bedrock usage payload with both cache buckets populated.
    _CACHED_USAGE: ClassVar[dict[str, object]] = {
        "inputTokens": 10,
        "outputTokens": 5,
        "totalTokens": 45,
        "cacheReadInputTokens": 20,
        "cacheWriteInputTokens": 10,
    }

    async def test_responses_batch_usage(self) -> None:
        """Non-streaming usage: input includes cache buckets, cached is read bucket."""
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
        assert response.usage.output_tokens == 5
        assert response.usage.total_tokens == 45
        assert response.usage.output_tokens_details is not None
        assert response.usage.output_tokens_details.reasoning_tokens == 0

    async def test_responses_stream_usage(self) -> None:
        """Streamed usage: the terminal event applies the same cache math."""
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
        assert usage["output_tokens"] == 5
        assert usage["total_tokens"] == 45

    async def test_chat_batch_usage(self) -> None:
        """Chat completions prompt_tokens include both cache buckets."""
        legacy_token = chat_adapter._LEGACY_FUNCTION.set(False)  # noqa: SLF001
        try:
            completion = await self._format_chat_response()
        finally:
            chat_adapter._LEGACY_FUNCTION.reset(legacy_token)  # noqa: SLF001
        assert completion.usage is not None
        assert completion.usage.prompt_tokens == 40
        assert completion.usage.completion_tokens == 5
        assert completion.usage.total_tokens == 45
        assert completion.usage.prompt_tokens_details is not None
        assert completion.usage.prompt_tokens_details.cached_tokens == 20

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
        """Chat streaming usage extraction applies the same cache math."""
        usage = extract_stream_usage(
            cast(
                "ConverseStreamOutputTypeDef",
                {"metadata": {"usage": self._CACHED_USAGE}},
            )
        )
        assert usage is not None
        assert usage.prompt_tokens == 40
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 45
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 20


class TestAnnotations:
    """Citations surface as url_citation annotations."""

    async def test_non_streaming_annotations(self) -> None:
        """CitationsContent blocks yield annotations on the output message."""
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
        """A citation delta during a text block emits annotation.added."""
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
        assert added_payload["annotation"]["url"] == "https://example.com"
        assert added_payload["annotation"]["start_index"] == len("Hello")

        by_event = {sse.event: _payload(sse) for sse in events}
        text_done = by_event["response.output_text.done"]
        assert text_done["text"] == "Hello"
        part_done = by_event["response.content_part.done"]
        assert part_done["part"]["annotations"][0]["url"] == "https://example.com"
        message = by_event["response.completed"]["response"]["output"][-1]
        assert message["content"][0]["annotations"][0]["url"] == "https://example.com"

    async def test_streaming_two_citations_in_one_part(self) -> None:
        """Two citations in one text part emit annotation_index 0 then 1."""
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

        message = _payload(events[-1])["response"]["output"][-1]
        annotations = message["content"][0]["annotations"]
        assert [a["url"] for a in annotations] == [
            "https://example.com/1",
            "https://example.com/2",
        ]

    async def test_streaming_citation_after_text_block(self) -> None:
        """A citation arriving after the text block is patched into the final message."""
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
        assert all(
            sse.event != "response.output_text.annotation.added" for sse in events
        )
        message = _payload(events[-1])["response"]["output"][-1]
        (annotation,) = message["content"][0]["annotations"]
        assert annotation["url"] == "https://late.example"
        assert annotation["title"] == "https://late.example"


class TestNonStreamOutputShape:
    """Non-streaming output items keep their Bedrock block positions."""

    async def test_text_then_tool_use_orders_message_first(self) -> None:
        """A [text, toolUse] response yields [message, function_call]."""
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
        part = message.content[0]
        assert isinstance(part, ResponseOutputText)
        assert part.text == "Let me check."
        assert isinstance(call, ResponseFunctionToolCall)
        assert call.call_id == "t1"

    async def test_tool_use_between_text_runs_splits_messages(self) -> None:
        """Each contiguous text run becomes its own message at its position."""
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
    """Web-search sources attach to the nearest preceding web_search_call."""

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
        """Each citation block feeds the closest preceding web_search_call."""
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
        assert isinstance(message, ResponseOutputMessage)

    async def test_sources_before_any_call_go_to_first_call(self) -> None:
        """Citations with no preceding call attach to the first call."""
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


class TestWebSearchLifecycleEvents:
    """The streamed web_search_call lifecycle matches the upstream event sequence."""

    async def test_in_progress_then_searching_then_completed(self) -> None:
        """in_progress and searching precede completed, in that order."""
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
        event_types = [sse.event for sse in events]
        assert (
            event_types.index("response.web_search_call.in_progress")
            < event_types.index("response.web_search_call.searching")
            < event_types.index("response.web_search_call.completed")
        )


class TestEmptyToolArguments:
    """Streamed tool calls without input deltas emit `{}` like non-streaming."""

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
        by_event = {sse.event: _payload(sse) for sse in events}
        assert by_event["response.function_call_arguments.done"]["arguments"] == "{}"
        assert by_event["response.output_item.done"]["item"]["arguments"] == "{}"
        final_call = _payload(events[-1])["response"]["output"][-1]
        assert final_call["arguments"] == "{}"


class TestEchoFields:
    """Request parameters are echoed on the response object."""

    async def test_echoed_fields_and_timestamps(self) -> None:
        """instructions, service_tier, cache fields, and timestamps are set."""
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
        assert response.created_at == 1234
        assert isinstance(response.created_at, int)
        assert isinstance(response.completed_at, int)

    async def test_streamed_created_at_is_int(self) -> None:
        """Streaming lifecycle and terminal events carry an int created_at."""
        events = await _collect(
            format_stream(
                "resp-1", 1234.56, "model", _stream(_text_stream_events()), _request()
            )
        )
        assert _payload(events[0])["response"]["created_at"] == 1234
        assert _payload(events[-1])["response"]["created_at"] == 1234
        assert _payload(events[-1])["response"]["completed_at"] >= 1234


class TestStreamedModeration:
    """The streamed terminal event carries the moderation field."""

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
        """The moderation builder result lands on the response.completed payload."""
        result = ModerationResult(
            flagged=False,
            categories={},
            category_scores={},
            category_applied_input_types={},
            model="gr123",
        )
        moderation = ResponseModeration(input=result, output=result)
        events = await _collect(
            format_stream(
                "resp-1",
                1.0,
                "model",
                _stream(_text_stream_events()),
                _request(),
                moderation_builder=lambda: moderation,
            )
        )
        payload = _payload(events[-1])
        assert payload["response"]["moderation"]["input"]["model"] == "gr123"
        assert payload["response"]["moderation"]["output"]["flagged"] is False

    async def test_moderation_built_from_captured_stream_trace(self) -> None:
        """The real builder maps the trace captured from the metadata event."""
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
    """safety_identifier and stream_options are accepted and ignored."""

    def test_safety_identifier_accepted(self) -> None:
        """safety_identifier no longer raises an unsupported-parameter error."""
        request = _request(safety_identifier="user-1")
        assert request.safety_identifier == "user-1"

    def test_stream_options_accepted(self) -> None:
        """stream_options no longer raises an unsupported-parameter error."""
        request = _request(stream=True, stream_options={"include_obfuscation": False})
        assert request.stream_options is not None
        assert request.stream_options.include_obfuscation is False


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
    """execute_image_generation_calls tolerates malformed input and errors."""

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
        assert stub_model.calls[0]["width"] == 1024
        assert stub_model.calls[0]["height"] == 1024
        assert isinstance(result[0], ImageGenerationCall)
        assert result[0].id == "resp-1-img-1"
        assert result[0].status == "completed"

    async def test_generation_error_produces_failed_item(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A generation failure is caught and yields a status="failed" item."""
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
        assert logged[0][1] == {"level": "warning"}


class TestImageGenerationStreamEvents:
    """The streaming image post-handler emits OpenAI lifecycle events."""

    async def _run_handler(
        self, stub_model: _StubImageModel, monkeypatch: pytest.MonkeyPatch
    ) -> list[JSONServerSentEvent]:
        """Run the post-stream handler over one suppressed image call."""
        monkeypatch.setattr(responses_adapter, "validate_model", _stub_validate_model)
        monkeypatch.setattr(responses_adapter, "get_image_model", lambda _: stub_model)
        state = responses_adapter._StreamState("resp-1")  # noqa: SLF001
        state.suppressed_tool_calls.append(
            ("t1", "image_generation", '{"prompt": "a cat"}')
        )
        return [
            sse
            async for sse in responses_adapter.image_generation_stream_handler(
                state,
                ImageGeneration(type="image_generation"),
                "resp-1",
                "fallback-model",
            )
        ]

    async def test_completed_call_emits_lifecycle_events(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful generation emits in_progress, generating, completed."""
        events = await self._run_handler(_StubImageModel(), monkeypatch)
        assert [sse.event for sse in events] == [
            "response.output_item.added",
            "response.image_generation_call.in_progress",
            "response.image_generation_call.generating",
            "response.image_generation_call.completed",
            "response.output_item.done",
        ]
        payloads = [_payload(sse) for sse in events]
        assert payloads[0]["item"]["status"] == "in_progress"
        for payload in payloads[1:4]:
            assert payload["item_id"] == "resp-1-img-1"
        assert payloads[4]["item"]["status"] == "completed"
        assert payloads[4]["item"]["result"] == "ZmFrZQ=="
        assert [payload["sequence_number"] for payload in payloads] == list(range(5))

    async def test_failed_call_skips_completed_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed generation emits no image_generation_call.completed event."""
        events = await self._run_handler(
            _StubImageModel(image=None, raises=RuntimeError("boom")), monkeypatch
        )
        assert [sse.event for sse in events] == [
            "response.output_item.added",
            "response.image_generation_call.in_progress",
            "response.image_generation_call.generating",
            "response.output_item.done",
        ]
        assert _payload(events[-1])["item"]["status"] == "failed"
