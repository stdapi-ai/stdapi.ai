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

        Pins current behavior when a client replays a reasoning item that was
        never requested with ``include=["reasoning.encrypted_content"]``: the
        resulting ``reasoningText`` block carries no ``signature`` key.
        Anthropic models may reject unsigned thinking blocks when they are
        replayed as part of a tool-use continuation; clients should request
        ``include=["reasoning.encrypted_content"]`` to avoid this.

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


#: Bedrock reasoning model used for live tests (Claude extended thinking).
_LIVE_BEDROCK_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"

#: Official API reasoning model used for live tests.
_LIVE_OFFICIAL_MODEL = "gpt-5-nano"


class TestReasoningLive:
    """Reasoning content end-to-end against a real backend.

    Ref: https://developers.openai.com/api/docs/guides/reasoning#preserve-reasoning-without-stored-responses
         https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/routes/openai_responses.py:create_response
    """

    @pytest.mark.expensive
    def test_reasoning_item_and_round_trip(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """A reasoning item is returned before the answer and replays as context.

        ``store=False`` makes the turn stateless, so the second call has to carry
        the whole previous ``output`` — reasoning item included — in its
        ``input``.  The follow-up question is only answerable from that replayed
        context, which is what makes the round trip observable.
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
        answer = follow_up.output_text.lower()
        assert "7" in answer or "seven" in answer, (
            f"the replayed items must reach the model: {follow_up.output_text!r}"
        )

    @pytest.mark.expensive
    def test_reasoning_streaming_events(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """A streamed reasoning item is announced, deltaed and repeated on completion.

        The stream must open with ``response.created`` and reach
        ``response.completed`` with strictly increasing ``sequence_number``s.
        Raw reasoning deltas (``response.reasoning_text.*``) only exist on the
        Bedrock path — upstream streams summaries instead.

        Ref: https://developers.openai.com/api/docs/guides/streaming-responses
             https://developers.openai.com/api/reference/resources/responses/streaming-events
        """
        model = _LIVE_OFFICIAL_MODEL if use_official_api else _LIVE_BEDROCK_MODEL
        event_types: list[str] = []
        added_item_types: list[str] = []
        sequence_numbers: list[int] = []
        with openai_client.responses.stream(
            model=model,
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
