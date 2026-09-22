"""Reasoning items on the OpenAI Responses API, from Bedrock reasoningContent blocks.

Ref: https://developers.openai.com/api/docs/guides/reasoning#preserve-reasoning-without-stored-responses
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
     stdapi/models/chat/_adapters/_openai_responses.py:_build_reasoning_item
"""

import json
from base64 import urlsafe_b64encode
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest
from openai.types.responses.response_content_part_added_event import (
    ResponseContentPartAddedEvent as SDKResponseContentPartAddedEvent,
)
from openai.types.responses.response_content_part_done_event import (
    ResponseContentPartDoneEvent as SDKResponseContentPartDoneEvent,
)
from openai.types.responses.response_reasoning_summary_part_added_event import (
    ResponseReasoningSummaryPartAddedEvent as SDKReasoningSummaryPartAddedEvent,
)
from openai.types.responses.response_reasoning_summary_part_done_event import (
    ResponseReasoningSummaryPartDoneEvent as SDKReasoningSummaryPartDoneEvent,
)
from openai.types.responses.response_reasoning_summary_text_delta_event import (
    ResponseReasoningSummaryTextDeltaEvent as SDKReasoningSummaryTextDeltaEvent,
)
from openai.types.responses.response_reasoning_summary_text_done_event import (
    ResponseReasoningSummaryTextDoneEvent as SDKReasoningSummaryTextDoneEvent,
)

import stdapi.models.chat._adapters._openai_responses as responses_adapter
from stdapi.models.chat._adapters._openai_responses import (
    count_input_tokens_via_bedrock,
    decode_reasoning_content,
    encode_reasoning_content,
    format_response,
    format_stream,
    map_input,
)
from stdapi.models.chat._default import ChatModel
from stdapi.types.openai_responses import (
    EasyInputMessage,
    InputTokenCountParams,
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
    from collections.abc import AsyncGenerator

    from openai import OpenAI
    from sse_starlette import JSONServerSentEvent
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
    )

    from stdapi.monitoring import EventLog
    from stdapi.types.openai_responses import ResponseInputItem


#: Bind the request-log context that response logging requires.
pytestmark = pytest.mark.usefixtures("request_log")


#: Bedrock usage payload shared by fabricated Converse responses.
_USAGE = {"inputTokens": 3, "outputTokens": 5}
#: Bedrock reasoningText block with a signature.
_SIGNED_REASONING_BLOCK = {
    "reasoningContent": {"reasoningText": {"text": "think", "signature": "sig-1"}}
}

#: Bedrock stream: one reasoning block (text + signature) then a text block.
_REASONING_STREAM: list[dict[str, Any]] = [
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
    """The ``encrypted_content`` envelope carries Bedrock signatures losslessly.

    Upstream treats ``encrypted_content`` as opaque ciphertext.  This gateway is
    stateless, so it encodes (does not encrypt) the Bedrock ``reasoningText``
    signatures and ``redactedContent`` payloads into that field; anything it did
    not produce — including real OpenAI ciphertext — must decode to ``None``
    rather than raise, so the item can still be replayed as plain text.

    Ref: https://developers.openai.com/api/docs/guides/reasoning#preserve-reasoning-without-stored-responses
         stdapi/models/chat/_adapters/_openai_responses.py:encode_reasoning_content
         stdapi/models/chat/_adapters/_openai_responses.py:decode_reasoning_content
    """

    def test_round_trip(self) -> None:
        """Signatures and redacted payloads survive the envelope round trip.

        Signature order is load-bearing: Bedrock binds each signature to its own
        reasoning block.
        """
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
        """Foreign or tampered content decodes to ``None`` instead of raising."""
        assert decode_reasoning_content(content) is None

    def test_invalid_payload_shapes_decode_to_none(self) -> None:
        """A well-formed envelope with wrong field types decodes to ``None``."""
        payloads: tuple[dict[str, object], ...] = (
            {"signatures": "sig-1", "redacted": []},
            {"signatures": [1], "redacted": []},
            {"signatures": []},
            {"signatures": [], "redacted": ["%%%"]},
        )
        for payload in payloads:
            encoded = urlsafe_b64encode(json.dumps(payload).encode()).decode()
            assert decode_reasoning_content(encoded) is None

    def test_envelope_without_a_summary_mark_stays_content_bound(self) -> None:
        """An envelope issued before the ``summary`` mark existed still decodes.

        Its signatures bind to ``content``, so a replayed conversation keeps its
        signed reasoning; replayed as a summary, they are dropped.
        """
        issued = "eyJzaWduYXR1cmVzIjpbInNpZy0xIl0sInJlZGFjdGVkIjpbXX0="  # no mark

        assert decode_reasoning_content(issued) == (["sig-1"], [])
        assert decode_reasoning_content(issued, summary=True) == ([], [])

    def test_json_true_marks_the_signatures_summary_bound(self) -> None:
        """A JSON ``true`` mark binds the signatures to ``summary``."""
        encoded = urlsafe_b64encode(
            json.dumps(
                {"signatures": ["sig-1"], "redacted": [], "summary": True}
            ).encode()
        ).decode()

        assert decode_reasoning_content(encoded, summary=True) == (["sig-1"], [])
        assert decode_reasoning_content(encoded) == ([], [])

    @pytest.mark.parametrize("mark", ["true", 1, "false", False, None])
    def test_only_json_true_is_the_summary_mark(self, mark: object) -> None:
        """Any other ``summary`` value leaves the signatures bound to ``content``."""
        encoded = urlsafe_b64encode(
            json.dumps(
                {"signatures": ["sig-1"], "redacted": [], "summary": mark}
            ).encode()
        ).decode()

        assert decode_reasoning_content(encoded) == (["sig-1"], [])
        assert decode_reasoning_content(encoded, summary=True) == ([], [])


@pytest.mark.local
class TestReasoningOutput:
    """Bedrock reasoningContent blocks become ``reasoning`` items in ``output``.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
         stdapi/models/chat/_adapters/_openai_responses.py:_extract_output_items
    """

    async def test_reasoning_then_message(self) -> None:
        """A reasoning block yields a reasoning item ahead of the message item.

        Raw Bedrock reasoning text is exposed as ``content`` parts of type
        ``reasoning_text`` with an empty ``summary``, and without ``include`` the
        round-trip envelope is withheld.
        """
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
        """``include=["reasoning.encrypted_content"]`` attaches the Bedrock signature.

        This gateway gates the envelope on the ``include`` value; upstream now
        populates ``encrypted_content`` by default on stateless responses and
        only accepts the include for compatibility, so the assertion targets
        ``_includes_encrypted_reasoning`` rather than the upstream guide.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:_includes_encrypted_reasoning
        """
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
        """A ``redactedContent`` block yields a reasoning item with empty content.

        Bedrock redacts reasoning it will not disclose, so no text can be
        exposed, but the opaque payload still has to survive in the envelope for
        the next turn to be accepted.
        """
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
        """Each Bedrock reasoning block becomes its own item with its own signature.

        Merging two blocks into one item would pair the second signature with the
        wrong text and invalidate it on replay; item ids are suffixed by block
        index so the pairing stays explicit.
        """
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
        """``response.reasoning`` echoes the request's ``reasoning`` object."""
        response = await format_response(
            "resp-1",
            0.0,
            "model",
            _bedrock_response([{"text": "Hello"}]),
            _request(reasoning=Reasoning(effort="low", summary="auto")),
        )
        assert response.reasoning == Reasoning(effort="low", summary="auto")

    async def test_reasoning_config_not_fabricated(self) -> None:
        """``response.reasoning`` stays ``None`` when the request sent none.

        The field is an echo, not a report of what the model did, so it must not
        be invented from the effort the backend happened to use.
        """
        response = await format_response(
            "resp-1", 0.0, "model", _bedrock_response([{"text": "Hello"}]), _request()
        )
        assert response.reasoning is None


@pytest.mark.local
class TestReasoningStreaming:
    """Reasoning is streamed as ``response.reasoning_text.*`` inside item events.

    Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
         stdapi/models/chat/_adapters/_openai_responses.py:format_stream
    """

    #: Bedrock stream: one reasoning block (text + signature) then a text block.
    _EVENTS: ClassVar[list[dict[str, Any]]] = _REASONING_STREAM

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
        """The reasoning item streams as a complete, correctly indexed event run.

        Every event of the run carries the reasoning item's id, ``output_index``
        0 and ``content_index`` 0, ``sequence_number`` starts at zero and
        increments by one, and the terminal ``response.completed`` payload
        repeats the finished item verbatim.

        Ref: https://github.com/openai/openai-python/tree/main/src/openai/types/responses
        """
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
            "response.content_part.added",
            "response.reasoning_text.delta",
            "response.reasoning_text.delta",
            "response.reasoning_text.done",
            "response.content_part.done",
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
        added, part_added, delta_1, delta_2, done_text, part_done, done_item = payloads[
            2:9
        ]
        assert added["item"] == {
            "id": "resp-1-rs-0",
            "summary": [],
            "type": "reasoning",
            "content": [],
            "status": "in_progress",
        }
        assert added["output_index"] == 0
        for part_event, text in ((part_added, ""), (part_done, "think")):
            assert part_event["item_id"] == "resp-1-rs-0"
            assert part_event["output_index"] == 0
            assert part_event["content_index"] == 0
            assert part_event["part"] == {"type": "reasoning_text", "text": text}
        # The raw payloads validate against the openai SDK's own event models,
        # confirming they match the upstream PartReasoningText content-part shape.
        SDKResponseContentPartAddedEvent.model_validate(part_added)
        SDKResponseContentPartDoneEvent.model_validate(part_done)
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
        assert payloads[9]["output_index"] == 1
        assert payloads[9]["item"]["id"] == "resp-1-msg-1"
        # Sequence numbers are strictly increasing from zero.
        sequence_numbers = [payload["sequence_number"] for payload in payloads]
        assert sequence_numbers == list(range(len(payloads)))
        # The final response echoes reasoning and contains the reasoning item.
        completed = Response(**payloads[-1]["response"])
        assert completed.reasoning == Reasoning(effort="low")
        assert completed.output[0].model_dump(exclude_none=True) == done_item["item"]

    async def test_final_response_matches_non_streaming(self) -> None:
        """The streamed and non-streamed responses agree on the reasoning item.

        The two paths build output items from different Bedrock shapes (delta
        accumulation versus a whole message), so parity is what guarantees a
        client can switch ``stream`` without changing its replay logic.
        """
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
        """A redacted-only block opens and closes its item without text events.

        No ``response.reasoning_text.*`` event may be emitted for content the
        model refused to disclose, yet the item still has to close with the
        envelope needed to replay it.
        """
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
        added, done = _payload(sses[2]), _payload(sses[5])
        assert sses[2].event == "response.output_item.added"
        assert sses[3].event == "response.content_part.added"
        assert sses[4].event == "response.content_part.done"
        assert sses[5].event == "response.output_item.done"
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
        """Streaming withholds only the envelope when ``include`` omits it.

        The reasoning text is still streamed and still closes the item; a
        response without the include is simply not replayable.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:_includes_encrypted_reasoning
        """
        sses = await self._collect(self._EVENTS, _request())
        done_item = _payload(sses[8])
        assert sses[8].event == "response.output_item.done"
        assert "encrypted_content" not in done_item["item"]
        assert done_item["item"]["content"] == [
            {"text": "think", "type": "reasoning_text"}
        ], "only the envelope is withheld, not the reasoning text"


@pytest.mark.local
class TestReasoningSummaryShape:
    """A requested ``reasoning.summary`` puts the reasoning in ``summary``, as upstream does.

    Upstream (probed on gpt-5-nano, 2026-09-23) returns ``summary_text`` parts
    with ``content: []``, streams them through the ``reasoning_summary_part`` /
    ``reasoning_summary_text`` events and sends no reasoning content part.
    Without a summary request the item keeps its ``reasoning_text`` content.

    Ref: https://developers.openai.com/api/docs/guides/reasoning#reasoning-summaries
         https://developers.openai.com/api/reference/resources/responses/streaming-events
         stdapi/models/chat/_adapters/_openai_responses.py:_build_reasoning_item
         stdapi/models/chat/_adapters/_openai_responses.py:_handle_reasoning_delta
    """

    #: Request asking for a reasoning summary and the round-trip envelope.
    _SUMMARY_REQUEST: ClassVar[ResponseCreateParams] = _request(
        reasoning=Reasoning(effort="low", summary="auto"),
        include=["reasoning.encrypted_content"],
    )

    async def _collect(
        self, events: list[dict[str, Any]], request: ResponseCreateParams
    ) -> list[JSONServerSentEvent]:
        return [
            sse
            async for sse in format_stream(
                "resp-1", 0.0, "model", _stream(events), request
            )
        ]

    @pytest.mark.parametrize(
        "reasoning",
        [
            Reasoning(summary="auto"),
            Reasoning(summary="concise"),
            Reasoning(summary="detailed"),
            Reasoning(generate_summary="auto"),
        ],
    )
    async def test_summary_request_fills_summary(self, reasoning: Reasoning) -> None:
        """Every summary request returns the text as ``summary_text`` with empty content."""
        response = await format_response(
            "resp-1",
            0.0,
            "model",
            _bedrock_response([_SIGNED_REASONING_BLOCK, {"text": "Hello"}]),
            _request(reasoning=reasoning),
        )
        item = response.output[0]
        assert isinstance(item, ResponseReasoningItem)
        assert item.summary == [ReasoningItemSummary(text="think", type="summary_text")]
        assert item.content == []

    async def test_no_summary_request_keeps_content(self) -> None:
        """Without a summary request the text stays ``reasoning_text`` content."""
        response = await format_response(
            "resp-1",
            0.0,
            "model",
            _bedrock_response([_SIGNED_REASONING_BLOCK, {"text": "Hello"}]),
            _request(reasoning=Reasoning(effort="low")),
        )
        item = response.output[0]
        assert isinstance(item, ResponseReasoningItem)
        assert item.summary == []
        assert item.content is not None
        assert [(part.type, part.text) for part in item.content] == [
            ("reasoning_text", "think")
        ]

    async def test_stream_event_sequence(self) -> None:
        """The summary streams through upstream's summary events, never content parts.

        Every payload validates against the openai SDK's own event model, which
        pins the field names (``summary_index``, no ``content_index``).
        """
        sses = await self._collect(_REASONING_STREAM, self._SUMMARY_REQUEST)
        assert [sse.event for sse in sses[:9]] == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
            "response.output_item.done",
        ]
        added, part_added, delta_1, delta_2, text_done, part_done, done = [
            _payload(sse) for sse in sses[2:9]
        ]
        assert added["item"]["summary"] == []
        assert added["item"]["content"] == []
        for payload, model in (
            (part_added, SDKReasoningSummaryPartAddedEvent),
            (delta_1, SDKReasoningSummaryTextDeltaEvent),
            (delta_2, SDKReasoningSummaryTextDeltaEvent),
            (text_done, SDKReasoningSummaryTextDoneEvent),
            (part_done, SDKReasoningSummaryPartDoneEvent),
        ):
            model.model_validate(payload)
            assert payload["item_id"] == "resp-1-rs-0"
            assert payload["output_index"] == 0
            assert payload["summary_index"] == 0
            assert "content_index" not in payload
        assert part_added["part"] == {"type": "summary_text", "text": ""}
        assert [delta_1["delta"], delta_2["delta"]] == ["thi", "nk"]
        assert text_done["text"] == "think"
        assert part_done["part"] == {"type": "summary_text", "text": "think"}
        assert done["item"]["summary"] == [{"type": "summary_text", "text": "think"}]
        assert done["item"]["content"] == []
        assert not any(
            sse.event is not None
            and (
                sse.event.startswith("response.reasoning_text")
                or (
                    sse.event.startswith("response.content_part")
                    and _payload(sse)["item_id"] == "resp-1-rs-0"
                )
            )
            for sse in sses
        )
        sequence_numbers = [_payload(sse)["sequence_number"] for sse in sses]
        assert sequence_numbers == list(range(len(sses)))

    async def test_stream_matches_non_streaming(self) -> None:
        """The streamed and non-streamed summary items are identical, envelope included."""
        sses = await self._collect(_REASONING_STREAM, self._SUMMARY_REQUEST)
        completed = Response(**_payload(sses[-1])["response"])
        non_streaming = await format_response(
            "resp-1",
            0.0,
            "model",
            _bedrock_response([_SIGNED_REASONING_BLOCK, {"text": "Hello"}]),
            self._SUMMARY_REQUEST,
        )
        assert completed.output[0] == non_streaming.output[0]

    async def test_redacted_only_stream_opens_no_summary_part(self) -> None:
        """A redacted-only block streams no summary part and closes with ``summary: []``."""
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
        sses = await self._collect(events, self._SUMMARY_REQUEST)
        assert [sse.event for sse in sses[2:4]] == [
            "response.output_item.added",
            "response.output_item.done",
        ]
        done = _payload(sses[3])
        assert done["item"]["summary"] == []
        assert decode_reasoning_content(
            done["item"]["encrypted_content"], summary=True
        ) == ([], [b"\x01"])

    @pytest.mark.parametrize("stream", [False, True])
    async def test_summary_item_replays_with_its_signature(self, stream: bool) -> None:
        """A summary item echoed back as input replays signed, for a model requiring it.

        The client serializes the item and sends it back as the next turn's
        input: the summary text must return to Bedrock with its signature, or a
        signature-requiring model drops it.

        Ref: https://developers.openai.com/api/docs/guides/reasoning#preserve-reasoning-without-stored-responses
             stdapi/models/chat/_adapters/_openai_responses.py:_map_reasoning_item
        """
        if stream:
            sses = await self._collect(_REASONING_STREAM, self._SUMMARY_REQUEST)
            output = _payload(sses[-1])["response"]["output"]
        else:
            response = await format_response(
                "resp-1",
                0.0,
                "model",
                _bedrock_response([_SIGNED_REASONING_BLOCK, {"text": "Hello"}]),
                self._SUMMARY_REQUEST,
            )
            output = [item.model_dump(exclude_none=True) for item in response.output]
        echoed = ResponseReasoningItemInput.model_validate(
            json.loads(json.dumps(output[0]))
        )
        messages, _ = await map_input(
            cast(
                "list[ResponseInputItem]",
                [echoed, EasyInputMessage(role="user", content="next")],
            ),
            None,
            reasoning_signature_required=True,
        )
        assert messages[0]["content"] == [
            {
                "reasoningContent": {
                    "reasoningText": {"text": "think", "signature": "sig-1"}
                }
            }
        ]


@pytest.mark.local
class TestReasoningInputRoundTrip:
    """Echoed reasoning items map back to Bedrock reasoningContent blocks.

    Upstream tells callers managing context themselves to include reasoning items
    in the next request's ``input``.  Bedrock requires each replayed
    ``reasoningText`` to carry back its original signature byte-identically, so
    the envelope is decoded and re-attached block by block.

    Ref: https://developers.openai.com/api/docs/guides/reasoning#preserve-reasoning-without-stored-responses
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
         stdapi/models/chat/_adapters/_openai_responses.py:_map_reasoning_item
    """

    async def test_emitted_item_maps_back_with_signature_and_redacted(self) -> None:
        """A gateway-issued envelope re-attaches its signature and redacted bytes.

        Redacted payloads are appended after the text blocks, matching the order
        Bedrock produced them in.
        """
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
        """Replaying two emitted items restores the original two signed blocks.

        The items are taken straight from ``format_response`` output, so this
        closes the emit/replay loop rather than a hand-written envelope, and both
        blocks merge back into a single assistant message.
        """
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
        """A summary-only item is replayed as unsigned reasoning text.

        Upstream never exposes raw reasoning, so an item that travelled through
        the official API carries only ``summary`` parts; they are the best
        available reconstruction of the turn.
        """
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
        """``content`` parts win over ``summary`` parts when both are present.

        Only ``content`` is signature-bearing, so preferring the summary would
        discard the signature and the model's actual chain of thought.
        """
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
        """Real OpenAI ciphertext is dropped and the item replays as plain text.

        A client migrating from the official API sends envelopes this gateway
        cannot read; rejecting them would break the conversation, so they are
        ignored.
        """
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
        """An item with neither text nor envelope produces no Bedrock message.

        Emitting an empty assistant message would be rejected by Converse.
        """
        item = ResponseReasoningItemInput(id="rs-1", summary=[], type="reasoning")
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages == []

    async def test_unsigned_replay_without_encrypted_content(self) -> None:
        """An echoed item with no ``encrypted_content`` maps to an unsigned block.

        A client replaying a reasoning item that was never requested with
        ``include=["reasoning.encrypted_content"]`` gets a ``reasoningText``
        block with no ``signature`` key, which is what models accepting an
        unsigned replay receive.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
             stdapi/models/chat/_adapters/_openai_responses.py:_map_reasoning_item
        """
        item = ResponseReasoningItemInput(
            id="rs-1",
            summary=[],
            type="reasoning",
            content=[ReasoningItemContentInput(text="think", type="reasoning_text")],
        )
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages[0]["content"] == [
            {"reasoningContent": {"reasoningText": {"text": "think"}}}
        ]
        content_block = messages[0]["content"][0]
        assert "signature" not in content_block["reasoningContent"]["reasoningText"]


class TestReasoningInputSignatureRequired:
    """Models rejecting an unsigned replay receive only the blocks they signed.

    A reasoning item echoed without its ``encrypted_content`` envelope has lost
    its signatures, and Claude answers such a replay with
    ``messages.1.content.0.thinking.signature: Field required``.  Those texts are
    dropped so the turn still gets an answer, at the cost of thinking continuity.

    Ref: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
         stdapi/models/chat/_adapters/_openai_responses.py:_map_reasoning_item
    """

    async def test_unsigned_item_is_dropped_and_warns(
        self, request_log: EventLog
    ) -> None:
        """An envelope-less item yields no message, and the drop is logged."""
        item = ResponseReasoningItemInput(
            id="rs-1",
            summary=[],
            type="reasoning",
            content=[ReasoningItemContentInput(text="think", type="reasoning_text")],
        )

        messages, _ = await map_input(
            cast("list[ResponseInputItem]", [item]),
            None,
            reasoning_signature_required=True,
        )

        assert messages == []
        assert request_log["level"] == "warning"
        assert any(
            "reasoning" in str(detail) for detail in request_log["error_detail"]
        ), "the dropped reasoning content must be reported in the request log"

    async def test_signed_item_is_replayed_untouched(
        self, request_log: EventLog
    ) -> None:
        """A gateway-issued envelope still replays its signed and redacted blocks."""
        item = ResponseReasoningItemInput(
            id="rs-1",
            summary=[],
            type="reasoning",
            content=[ReasoningItemContentInput(text="think", type="reasoning_text")],
            encrypted_content=encode_reasoning_content(["sig-1"], [b"\x00\x01"]),
            status="completed",
        )

        messages, _ = await map_input(
            cast("list[ResponseInputItem]", [item]),
            None,
            reasoning_signature_required=True,
        )

        assert messages[0]["content"] == [
            {
                "reasoningContent": {
                    "reasoningText": {"text": "think", "signature": "sig-1"}
                }
            },
            {"reasoningContent": {"redactedContent": b"\x00\x01"}},
        ]
        assert request_log["level"] == "info", "a signed replay warns about nothing"

    async def test_summary_fallback_is_dropped(self) -> None:
        """A summary text is never signature-bearing, so it cannot be replayed."""
        item = ResponseReasoningItemInput(
            id="rs-1",
            summary=[ReasoningItemSummary(text="the summary", type="summary_text")],
            type="reasoning",
        )

        messages, _ = await map_input(
            cast("list[ResponseInputItem]", [item]),
            None,
            reasoning_signature_required=True,
        )

        assert messages == []

    async def test_surrounding_items_are_untouched(self) -> None:
        """Dropping the reasoning leaves the messages around it in place and in order."""
        item = ResponseReasoningItemInput(
            id="rs-1",
            summary=[],
            type="reasoning",
            content=[ReasoningItemContentInput(text="think", type="reasoning_text")],
        )

        messages, _ = await map_input(
            cast(
                "list[ResponseInputItem]",
                [
                    EasyInputMessage(role="user", content="q", type="message"),
                    item,
                    EasyInputMessage(role="assistant", content="a", type="message"),
                ],
            ),
            None,
            reasoning_signature_required=True,
        )

        assert messages == [
            {"role": "user", "content": [{"text": "q"}]},
            {"role": "assistant", "content": [{"text": "a"}]},
        ]


class _SignatureRequiredChatModel(ChatModel):
    """Chat model whose family rejects unsigned replayed reasoning blocks."""

    #: Mirrors the model families that require signed reasoning replays.
    REASONING_SIGNATURE_REQUIRED: ClassVar[bool] = True


class TestInputTokenCountSignatureRequired:
    """The token count reflects the input the generation path would actually send.

    ``POST /v1/responses/input_tokens`` answers "how many tokens will this cost",
    so it must apply the same unsigned-reasoning drop as ``create_response``;
    counting text the model never receives over-reports the prompt.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/input-tokens
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html
         stdapi/models/chat/_adapters/_openai_responses.py:count_input_tokens_via_bedrock
    """

    @staticmethod
    def _counted_messages(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        """Patch the Bedrock client and return the list the counted messages land in."""
        counted: list[dict[str, Any]] = []

        class _FakeClient:
            async def count_tokens(
                self,
                *,
                modelId: str,  # noqa: N803 (mirrors the boto3 client's camelCase kwarg)
                input: dict[str, Any],  # noqa: A002
            ) -> dict[str, int]:
                counted.extend(input["converse"]["messages"])
                return {"inputTokens": 7}

        monkeypatch.setattr(
            responses_adapter,
            "get_client",
            lambda _service, _region=None: _FakeClient(),
        )
        return counted

    @staticmethod
    def _request(encrypted_content: str | None) -> InputTokenCountParams:
        """Build a count request replaying one reasoning item after a question."""
        return InputTokenCountParams.model_validate(
            {
                "model": "m",
                "input": [
                    EasyInputMessage(role="user", content="q", type="message"),
                    ResponseReasoningItemInput(
                        id="rs-1",
                        summary=[],
                        type="reasoning",
                        content=[
                            ReasoningItemContentInput(
                                text="think", type="reasoning_text"
                            )
                        ],
                        encrypted_content=encrypted_content,
                    ),
                ],
            }
        )

    async def test_unsigned_reasoning_is_not_counted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model requiring a signature is not billed for reasoning it will not see."""
        counted = self._counted_messages(monkeypatch)

        tokens = await count_input_tokens_via_bedrock(
            self._request(None), "m", "us-east-1", _SignatureRequiredChatModel("m")
        )

        assert tokens == 7, "the Bedrock inputTokens value is returned unchanged"
        assert counted == [{"role": "user", "content": [{"text": "q"}]}]

    async def test_signed_reasoning_is_counted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reasoning that survives the drop is part of the prompt, so it is counted."""
        counted = self._counted_messages(monkeypatch)

        await count_input_tokens_via_bedrock(
            self._request(encode_reasoning_content(["sig-1"], [])),
            "m",
            "us-east-1",
            _SignatureRequiredChatModel("m"),
        )

        assert counted[1]["content"] == [
            {
                "reasoningContent": {
                    "reasoningText": {"text": "think", "signature": "sig-1"}
                }
            }
        ]

    async def test_other_models_still_count_unsigned_reasoning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Model families accepting an unsigned replay receive, and pay for, it."""
        counted = self._counted_messages(monkeypatch)

        await count_input_tokens_via_bedrock(
            self._request(None), "m", "us-east-1", ChatModel("m")
        )

        assert counted[1]["content"] == [
            {"reasoningContent": {"reasoningText": {"text": "think"}}}
        ]


class TestReasoningLive:
    """Reasoning content end-to-end against a real backend.

    Ref: https://developers.openai.com/api/docs/guides/reasoning#preserve-reasoning-without-stored-responses
         https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/routes/openai_responses.py:create_response
    """

    @pytest.mark.expensive
    def test_reasoning_item_and_round_trip(
        self, openai_client: OpenAI, use_official_api: bool, chat_reasoning_model: str
    ) -> None:
        """A reasoning item is returned before the answer and replays as context.

        ``store=False`` makes the turn stateless, so the second call has to carry
        the whole previous ``output`` — reasoning item included — in its
        ``input``.  The follow-up question is only answerable from that replayed
        context, which is what makes the round trip observable.
        """
        response = openai_client.responses.create(
            model=chat_reasoning_model,
            input="What is 2+2? Think briefly, then answer with the number only.",
            reasoning={"effort": "low"},
            include=["reasoning.encrypted_content"],
            store=False,
            max_output_tokens=4096,
        )
        assert response.status == "completed"
        assert response.reasoning is not None
        assert response.reasoning.effort == "low", "reasoning.effort is echoed back"
        assert response.usage is not None
        assert (
            response.usage.total_tokens
            == response.usage.input_tokens + response.usage.output_tokens
        )
        reasoning_items = [item for item in response.output if item.type == "reasoning"]
        assert reasoning_items, "Expected a reasoning output item"
        item = reasoning_items[0]
        message_indexes = [
            index
            for index, output in enumerate(response.output)
            if output.type == "message"
        ]
        assert message_indexes, "Expected an assistant message output item"
        assert response.output.index(item) < message_indexes[0], (
            "the reasoning item must precede the message it produced"
        )
        if not use_official_api:
            # Bedrock chain of thought maps to reasoning_text content parts.
            assert item.content
            assert item.content[0].type == "reasoning_text"
            assert item.content[0].text
            # Claude extended thinking signatures ride the round-trip envelope.
            assert item.encrypted_content

        follow_up = openai_client.responses.create(  # type: ignore[call-overload]
            model=chat_reasoning_model,
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
        answer = follow_up.output_text.lower()
        assert "7" in answer or "seven" in answer, (
            f"the replayed items must reach the model: {follow_up.output_text!r}"
        )

    @pytest.mark.expensive
    def test_reasoning_streaming_events(
        self, openai_client: OpenAI, use_official_api: bool, chat_reasoning_model: str
    ) -> None:
        """A streamed reasoning item is announced, deltaed and repeated on completion.

        The stream must open with ``response.created`` and reach
        ``response.completed`` with strictly increasing ``sequence_number``s.
        Raw reasoning deltas (``response.reasoning_text.*``) only exist on the
        Bedrock path — upstream streams summaries instead.

        Ref: https://developers.openai.com/api/docs/guides/streaming-responses
             https://developers.openai.com/api/reference/resources/responses/streaming-events
        """
        event_types: list[str] = []
        added_item_types: list[str] = []
        sequence_numbers: list[int] = []
        with openai_client.responses.stream(
            model=chat_reasoning_model,
            input="What is 3+3? Think briefly, then answer with the number only.",
            reasoning={"effort": "low"},
            store=False,
            max_output_tokens=4096,
        ) as stream:
            for event in stream:
                event_types.append(event.type)
                if (sequence := getattr(event, "sequence_number", None)) is not None:
                    sequence_numbers.append(sequence)
                if event.type == "response.output_item.added":
                    added_item_types.append(event.item.type)
        final = stream.get_final_response()
        assert final.status == "completed"
        assert event_types[0] == "response.created"
        assert "response.completed" in event_types
        assert sequence_numbers == sorted(set(sequence_numbers)), (
            "sequence_number must increase strictly across the stream"
        )
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

    @pytest.mark.expensive
    def test_reasoning_summary_request(
        self, openai_client: OpenAI, chat_reasoning_model: str
    ) -> None:
        """``reasoning.summary`` returns the reasoning text, streamed as it is produced.

        The summary fills the item's ``summary`` with ``summary_text`` parts and
        leaves ``content`` empty, streamed through
        ``response.reasoning_summary_part.added``,
        ``response.reasoning_summary_text.delta`` / ``.done`` and
        ``response.reasoning_summary_part.done``, so a client reading only
        ``summary`` gets the reasoning.

        Ref: https://developers.openai.com/api/docs/guides/reasoning#reasoning-summaries
             https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/models/chat/_adapters/_openai_responses.py:format_stream
        """
        event_types: list[str] = []
        with openai_client.responses.stream(
            model=chat_reasoning_model,
            input=(
                "How many positive integers below 60 are divisible by 3 or 5? "
                "Answer with the number only."
            ),
            reasoning={"effort": "low", "summary": "detailed"},
            store=False,
            max_output_tokens=4096,
        ) as stream:
            event_types.extend(event.type for event in stream)
        final = stream.get_final_response()

        assert final.status == "completed"
        assert final.reasoning is not None
        assert final.reasoning.summary == "detailed", "reasoning.summary is echoed"
        item = next(item for item in final.output if item.type == "reasoning")
        assert item.summary
        assert all(part.type == "summary_text" for part in item.summary)
        assert all(part.text for part in item.summary)
        assert not item.content, "the reasoning is in summary, not content"
        assert not any(
            event_type.startswith("response.reasoning_text.")
            for event_type in event_types
        ), event_types
        expected = (
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
        )
        firsts = [event_types.index(event_type) for event_type in expected]
        assert firsts == sorted(firsts), event_types
        assert firsts[-1] < event_types.index("response.output_text.delta"), (
            "the reasoning streams before the answer"
        )
