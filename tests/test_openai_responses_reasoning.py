"""Tests for reasoning content in the OpenAI Responses API (unit and live)."""

import json
from base64 import urlsafe_b64encode
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest

from stdapi.models.chat._adapters._openai_responses import (
    decode_reasoning_content,
    encode_reasoning_content,
    format_response,
    format_stream,
    map_input,
)
from stdapi.monitoring import REQUEST_LOG
from stdapi.types.openai_responses import (
    Reasoning,
    ReasoningItemContentInput,
    ReasoningItemSummary,
    Response,
    ResponseCreateParams,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningItemInput,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from openai import OpenAI
    from sse_starlette import JSONServerSentEvent
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
    )

    from stdapi.types.openai_responses import ResponseInputItem


@pytest.fixture(autouse=True)
def _request_log() -> Generator[None]:
    """Provide the request log context required by response logging."""
    token = REQUEST_LOG.set({"level": "info"})  # type: ignore[typeddict-item]
    yield
    REQUEST_LOG.reset(token)


#: Bedrock usage payload shared by fabricated Converse responses.
_USAGE = {"inputTokens": 3, "outputTokens": 5}
#: Bedrock reasoningText block with a signature.
_SIGNED_REASONING_BLOCK = {
    "reasoningContent": {"reasoningText": {"text": "think", "signature": "sig-1"}}
}


def _bedrock_response(contents: list[dict[str, Any]]) -> ConverseResponseTypeDef:
    """Build a minimal Bedrock Converse response around content blocks."""
    return cast(
        "ConverseResponseTypeDef",
        {
            "output": {"message": {"role": "assistant", "content": contents}},
            "usage": _USAGE,
            "stopReason": "end_turn",
        },
    )


def _request(**kwargs: Any) -> ResponseCreateParams:  # noqa: ANN401
    """Build a Responses creation request with optional extra fields."""
    return ResponseCreateParams(model="anthropic.claude-sonnet-5", input="hi", **kwargs)


def _payload(sse: JSONServerSentEvent) -> dict[str, Any]:
    """Return the decoded data payload of an SSE event."""
    if isinstance(sse.data, dict):
        return sse.data
    assert isinstance(sse.data, str)
    return json.loads(sse.data)  # type: ignore[no-any-return]


async def _stream(
    events: list[dict[str, Any]],
) -> AsyncGenerator[ConverseStreamOutputTypeDef]:
    """Yield fabricated Bedrock ConverseStream events."""
    for event in events:
        yield cast("ConverseStreamOutputTypeDef", event)


@pytest.mark.local
class TestReasoningContentCodec:
    """encode/decode of the local encrypted_content envelope."""

    def test_round_trip(self) -> None:
        """Signatures and redacted payloads survive the envelope round trip."""
        encoded = encode_reasoning_content(["sig-1", "sig-2"], [b"\x00\xff"])
        assert decode_reasoning_content(encoded) == (["sig-1", "sig-2"], [b"\x00\xff"])

    @pytest.mark.parametrize(
        "content",
        [
            "",
            "!!!",
            "gAAAAABfoo",  # OpenAI-encrypted style, not base64-decodable
            encode_reasoning_content([], []).replace("e", "a"),  # tampered
            "bm90IGpzb24=",  # valid base64, not JSON
            "WzEsIDJd",  # JSON list, not the envelope mapping
            "bnVsbA==",  # JSON null
        ],
    )
    def test_foreign_content_decodes_to_none(self, content: str) -> None:
        """Foreign or tampered content is rejected gracefully."""
        assert decode_reasoning_content(content) is None

    def test_invalid_payload_shapes_decode_to_none(self) -> None:
        """Envelopes with wrong field types are rejected."""
        payloads: tuple[dict[str, object], ...] = (
            {"signatures": "sig-1", "redacted": []},
            {"signatures": [1], "redacted": []},
            {"signatures": []},
            {"signatures": [], "redacted": ["%%%"]},
        )
        for payload in payloads:
            encoded = urlsafe_b64encode(json.dumps(payload).encode()).decode()
            assert decode_reasoning_content(encoded) is None


@pytest.mark.local
class TestReasoningOutput:
    """Non-streaming reasoning output items."""

    async def test_reasoning_then_message(self) -> None:
        """A reasoning block yields a reasoning item before the message item."""
        response = await format_response(
            "resp-1",
            0.0,
            "model",
            _bedrock_response([_SIGNED_REASONING_BLOCK, {"text": "Hello"}]),
            _request(),
        )
        reasoning, message = response.output
        assert isinstance(reasoning, ResponseReasoningItem)
        assert reasoning.id == "resp-1-rs-0"
        assert reasoning.type == "reasoning"
        assert reasoning.status == "completed"
        assert reasoning.summary == []
        assert reasoning.content is not None
        assert [(part.type, part.text) for part in reasoning.content] == [
            ("reasoning_text", "think")
        ]
        assert reasoning.encrypted_content is None
        assert isinstance(message, ResponseOutputMessage)
        text_part = message.content[0]
        assert isinstance(text_part, ResponseOutputText)
        assert text_part.text == "Hello"

    async def test_encrypted_content_requires_include(self) -> None:
        """encrypted_content is set only when include requests it."""
        response = await format_response(
            "resp-1",
            0.0,
            "model",
            _bedrock_response([_SIGNED_REASONING_BLOCK, {"text": "Hello"}]),
            _request(include=["reasoning.encrypted_content"]),
        )
        reasoning = response.output[0]
        assert isinstance(reasoning, ResponseReasoningItem)
        assert reasoning.encrypted_content
        assert decode_reasoning_content(reasoning.encrypted_content) == (["sig-1"], [])

    async def test_redacted_only_block(self) -> None:
        """RedactedContent yields a reasoning item with no text content."""
        response = await format_response(
            "resp-1",
            0.0,
            "model",
            _bedrock_response(
                [{"reasoningContent": {"redactedContent": b"\x01\x02"}}, {"text": "Hi"}]
            ),
            _request(include=["reasoning.encrypted_content"]),
        )
        reasoning = response.output[0]
        assert isinstance(reasoning, ResponseReasoningItem)
        assert reasoning.content == []
        assert reasoning.encrypted_content is not None
        assert decode_reasoning_content(reasoning.encrypted_content) == (
            [],
            [b"\x01\x02"],
        )

    async def test_contiguous_blocks_yield_one_item_per_block(self) -> None:
        """Each Bedrock reasoning block becomes its own reasoning item."""
        response = await format_response(
            "resp-1",
            0.0,
            "model",
            _bedrock_response(
                [
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": "a", "signature": "s1"}
                        }
                    },
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": "b", "signature": "s2"}
                        }
                    },
                    {"text": "Hello"},
                ]
            ),
            _request(include=["reasoning.encrypted_content"]),
        )
        first, second, message = response.output
        assert isinstance(first, ResponseReasoningItem)
        assert isinstance(second, ResponseReasoningItem)
        assert first.id == "resp-1-rs-0"
        assert second.id == "resp-1-rs-1"
        for item, text, signature in ((first, "a", "s1"), (second, "b", "s2")):
            assert item.content is not None
            assert item.content[0].text == text
            assert item.encrypted_content is not None
            assert decode_reasoning_content(item.encrypted_content) == ([signature], [])
        assert message.type == "message"

    async def test_reasoning_config_is_echoed(self) -> None:
        """response.reasoning echoes the request's reasoning parameter."""
        response = await format_response(
            "resp-1",
            0.0,
            "model",
            _bedrock_response([{"text": "Hello"}]),
            _request(reasoning=Reasoning(effort="low", summary="auto")),
        )
        assert response.reasoning == Reasoning(effort="low", summary="auto")

    async def test_reasoning_config_not_fabricated(self) -> None:
        """response.reasoning stays None when the request has none."""
        response = await format_response(
            "resp-1", 0.0, "model", _bedrock_response([{"text": "Hello"}]), _request()
        )
        assert response.reasoning is None


@pytest.mark.local
class TestReasoningStreaming:
    """Streaming reasoning events and final response parity."""

    #: Bedrock stream: one reasoning block (text + signature) then a text block.
    _EVENTS: ClassVar[list[dict[str, Any]]] = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "thi"}}}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "nk"}}}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"signature": "sig-1"}}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockDelta": {"delta": {"text": "Hello"}}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": _USAGE}},
    ]

    async def _collect(
        self, events: list[dict[str, Any]], request: ResponseCreateParams
    ) -> list[JSONServerSentEvent]:
        return [
            sse
            async for sse in format_stream(
                "resp-1", 0.0, "model", _stream(events), request
            )
        ]

    async def test_event_sequence(self) -> None:
        """Reasoning events precede the message events in the exact order."""
        sses = await self._collect(
            self._EVENTS,
            _request(
                include=["reasoning.encrypted_content"],
                reasoning=Reasoning(effort="low"),
            ),
        )
        assert [sse.event for sse in sses] == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
        payloads = [_payload(sse) for sse in sses]
        added, delta_1, delta_2, done_text, done_item = payloads[2:7]
        assert added["item"] == {
            "id": "resp-1-rs-0",
            "summary": [],
            "type": "reasoning",
            "content": [],
            "status": "in_progress",
        }
        assert added["output_index"] == 0
        for delta, text in ((delta_1, "thi"), (delta_2, "nk")):
            assert delta["item_id"] == "resp-1-rs-0"
            assert delta["output_index"] == 0
            assert delta["content_index"] == 0
            assert delta["delta"] == text
        assert done_text["text"] == "think"
        assert done_item["item"]["status"] == "completed"
        assert done_item["item"]["content"] == [
            {"text": "think", "type": "reasoning_text"}
        ]
        assert decode_reasoning_content(done_item["item"]["encrypted_content"]) == (
            ["sig-1"],
            [],
        )
        # The message item takes the next output_index.
        assert payloads[7]["output_index"] == 1
        assert payloads[7]["item"]["id"] == "resp-1-msg-1"
        # Sequence numbers are strictly increasing from zero.
        sequence_numbers = [payload["sequence_number"] for payload in payloads]
        assert sequence_numbers == list(range(len(payloads)))
        # The final response echoes reasoning and contains the reasoning item.
        completed = Response(**payloads[-1]["response"])
        assert completed.reasoning == Reasoning(effort="low")
        assert completed.output[0].model_dump(exclude_none=True) == done_item["item"]

    async def test_final_response_matches_non_streaming(self) -> None:
        """response.completed carries the same reasoning item as non-streaming."""
        request = _request(include=["reasoning.encrypted_content"])
        sses = await self._collect(self._EVENTS, request)
        completed = Response(**_payload(sses[-1])["response"])
        non_streaming = await format_response(
            "resp-1",
            0.0,
            "model",
            _bedrock_response([_SIGNED_REASONING_BLOCK, {"text": "Hello"}]),
            request,
        )
        assert completed.output[0] == non_streaming.output[0]
        completed_message = completed.output[1]
        non_streaming_message = non_streaming.output[1]
        assert isinstance(completed_message, ResponseOutputMessage)
        assert isinstance(non_streaming_message, ResponseOutputMessage)
        assert completed_message.content == non_streaming_message.content

    async def test_redacted_only_stream(self) -> None:
        """Redacted deltas open and close the item without text events."""
        events: list[dict[str, Any]] = [
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"redactedContent": b"\x01"}}
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": "Hi"}}},
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": _USAGE}},
        ]
        sses = await self._collect(
            events, _request(include=["reasoning.encrypted_content"])
        )
        added, done = _payload(sses[2]), _payload(sses[3])
        assert sses[2].event == "response.output_item.added"
        assert sses[3].event == "response.output_item.done"
        assert added["item"]["type"] == "reasoning"
        assert done["item"]["content"] == []
        assert decode_reasoning_content(done["item"]["encrypted_content"]) == (
            [],
            [b"\x01"],
        )
        assert not any(
            sse.event is not None and sse.event.startswith("response.reasoning_text")
            for sse in sses
        )

    async def test_no_encrypted_content_without_include(self) -> None:
        """Streaming omits encrypted_content when include does not ask for it."""
        sses = await self._collect(self._EVENTS, _request())
        done_item = _payload(sses[6])
        assert sses[6].event == "response.output_item.done"
        assert "encrypted_content" not in done_item["item"]


@pytest.mark.local
class TestReasoningInputRoundTrip:
    """Echoed reasoning items map back to Bedrock reasoningContent blocks."""

    async def test_emitted_item_maps_back_with_signature_and_redacted(self) -> None:
        """Our own encrypted_content re-attaches signatures and redacted bytes."""
        item = ResponseReasoningItemInput(
            id="rs-1",
            summary=[],
            type="reasoning",
            content=[ReasoningItemContentInput(text="think", type="reasoning_text")],
            encrypted_content=encode_reasoning_content(["sig-1"], [b"\x00\x01"]),
            status="completed",
        )
        messages, system = await map_input(
            cast("list[ResponseInputItem]", [item]), None
        )
        assert system == []
        (message,) = messages
        assert message["role"] == "assistant"
        assert message["content"] == [
            {
                "reasoningContent": {
                    "reasoningText": {"text": "think", "signature": "sig-1"}
                }
            },
            {"reasoningContent": {"redactedContent": b"\x00\x01"}},
        ]

    async def test_multi_block_run_reattaches_each_signature(self) -> None:
        """Multi-block reasoning items echo back with their own signatures."""
        response = await format_response(
            "resp-1",
            0.0,
            "model",
            _bedrock_response(
                [
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": "a", "signature": "s1"}
                        }
                    },
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": "b", "signature": "s2"}
                        }
                    },
                    {"text": "Hello"},
                ]
            ),
            _request(include=["reasoning.encrypted_content"]),
        )
        reasoning_items = [
            item for item in response.output if isinstance(item, ResponseReasoningItem)
        ]
        assert len(reasoning_items) == 2
        messages, _ = await map_input(
            cast("list[ResponseInputItem]", reasoning_items), None
        )
        (message,) = messages
        assert message["content"] == [
            {"reasoningContent": {"reasoningText": {"text": "a", "signature": "s1"}}},
            {"reasoningContent": {"reasoningText": {"text": "b", "signature": "s2"}}},
        ]

    async def test_summary_only_item_maps_to_text(self) -> None:
        """OpenAI-style summary-only items are injected as reasoning text."""
        item = ResponseReasoningItemInput(
            id="rs-1",
            summary=[ReasoningItemSummary(text="the summary", type="summary_text")],
            type="reasoning",
        )
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        (message,) = messages
        assert message["content"] == [
            {"reasoningContent": {"reasoningText": {"text": "the summary"}}}
        ]

    async def test_content_preferred_over_summary(self) -> None:
        """Content parts win over summary parts when both are present."""
        item = ResponseReasoningItemInput(
            id="rs-1",
            summary=[ReasoningItemSummary(text="summary", type="summary_text")],
            type="reasoning",
            content=[ReasoningItemContentInput(text="raw", type="reasoning_text")],
        )
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages[0]["content"] == [
            {"reasoningContent": {"reasoningText": {"text": "raw"}}}
        ]

    async def test_foreign_encrypted_content_is_ignored(self) -> None:
        """OpenAI-encrypted content falls back to text-only mapping."""
        item = ResponseReasoningItemInput(
            id="rs-1",
            summary=[],
            type="reasoning",
            content=[ReasoningItemContentInput(text="think", type="reasoning_text")],
            encrypted_content="gAAAAABforeign-openai-content",
        )
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages[0]["content"] == [
            {"reasoningContent": {"reasoningText": {"text": "think"}}}
        ]

    async def test_empty_item_is_dropped(self) -> None:
        """Items without any text or envelope produce no Bedrock message."""
        item = ResponseReasoningItemInput(id="rs-1", summary=[], type="reasoning")
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages == []


#: Bedrock reasoning model used for live tests (Claude extended thinking).
_LIVE_BEDROCK_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"

#: Official API reasoning model used for live tests.
_LIVE_OFFICIAL_MODEL = "gpt-5-nano"


class TestReasoningLive:
    """Reasoning content end-to-end against a real backend."""

    @pytest.mark.expensive
    def test_reasoning_item_and_round_trip(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """A reasoning request returns a reasoning item that round-trips.

        Validates:
            - The output contains a ``reasoning`` item before the message
            - ``include=["reasoning.encrypted_content"]`` attaches the envelope
            - Echoing the output as input continues the conversation
        """
        model = _LIVE_OFFICIAL_MODEL if use_official_api else _LIVE_BEDROCK_MODEL
        response = openai_client.responses.create(
            model=model,
            input="What is 2+2? Think briefly, then answer with the number only.",
            reasoning={"effort": "low"},
            include=["reasoning.encrypted_content"],
            store=False,
            max_output_tokens=4096,
        )
        assert response.status == "completed"
        assert response.reasoning is not None
        reasoning_items = [item for item in response.output if item.type == "reasoning"]
        assert reasoning_items, "Expected a reasoning output item"
        item = reasoning_items[0]
        assert response.output.index(item) < len(response.output) - 1
        if not use_official_api:
            # Bedrock chain of thought maps to reasoning_text content parts.
            assert item.content
            assert item.content[0].text
            # Claude extended thinking signatures ride the round-trip envelope.
            assert item.encrypted_content

        follow_up = openai_client.responses.create(  # type: ignore[call-overload]
            model=model,
            input=[
                *[
                    output.model_dump(mode="json", exclude_none=True)
                    for output in response.output
                ],
                {"role": "user", "content": "Now add 3 to your previous answer."},
            ],
            reasoning={"effort": "low"},
            store=False,
            max_output_tokens=4096,
        )
        assert follow_up.status == "completed"
        assert follow_up.output_text

    @pytest.mark.expensive
    def test_reasoning_streaming_events(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Streaming delivers the reasoning item through output_item events.

        Validates:
            - ``response.output_item.added`` announces a ``reasoning`` item
            - The final response contains the completed reasoning item
            - Locally, ``response.reasoning_text.delta``/``.done`` are emitted
        """
        model = _LIVE_OFFICIAL_MODEL if use_official_api else _LIVE_BEDROCK_MODEL
        event_types: list[str] = []
        added_item_types: list[str] = []
        with openai_client.responses.stream(
            model=model,
            input="What is 3+3? Think briefly, then answer with the number only.",
            reasoning={"effort": "low"},
            store=False,
            max_output_tokens=4096,
        ) as stream:
            for event in stream:
                event_types.append(event.type)
                if event.type == "response.output_item.added":
                    added_item_types.append(event.item.type)
        final = stream.get_final_response()
        assert final.status == "completed"
        assert any(item.type == "reasoning" for item in final.output)
        assert "reasoning" in added_item_types
        if not use_official_api:
            assert "response.reasoning_text.delta" in event_types
            assert "response.reasoning_text.done" in event_types
            reasoning_item = next(
                item for item in final.output if item.type == "reasoning"
            )
            assert reasoning_item.content
            assert reasoning_item.content[0].text
