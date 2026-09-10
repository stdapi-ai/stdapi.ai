"""Anthropic SSE event synthesis from a Bedrock Converse stream (no AWS calls).

Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
     stdapi/models/chat/_adapters/_anthropic_message.py:format_stream
"""

from __future__ import annotations

from json import loads
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest
from anthropic.types import CitationCharLocation as SdkCitationCharLocation
from anthropic.types import CitationsDelta as SdkCitationsDelta
from anthropic.types import RawContentBlockDeltaEvent as SdkRawContentBlockDeltaEvent

from stdapi.models.chat._adapters._anthropic_message import (
    format_response,
    format_stream,
)
from stdapi.types.anthropic_messages import TextBlock

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockOutputTypeDef,
        ConverseStreamOutputTypeDef,
    )

# The streaming adapter writes into the request log, which only exists inside a
# request, so every test needs the shared context fixture.
pytestmark = [pytest.mark.local, pytest.mark.usefixtures("request_log")]


async def _collect(
    events: list[dict[str, Any]], forced_tool: str | None = None
) -> list[tuple[str, dict[str, Any]]]:
    """Run *events* through ``format_stream`` and return ``(event, data)`` pairs."""

    async def _stream() -> AsyncIterator[dict[str, Any]]:
        for event in events:
            yield event

    stream = cast("AsyncIterator[ConverseStreamOutputTypeDef]", _stream())
    return [
        (sse.event or "", loads(cast("str", sse.data)))
        async for sse in format_stream("msg_1", "model-x", stream, forced_tool)
    ]


def _text_stream_events(text: str = "hi") -> list[dict[str, Any]]:
    """Return the Converse events of a stream emitting *text* as one text block."""
    return [
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]


def _tool_input_json(pairs: list[tuple[str, dict[str, Any]]], index: int) -> str:
    """Concatenate the ``input_json_delta`` fragments emitted for block *index*."""
    return "".join(
        data["delta"]["partial_json"]
        for event, data in pairs
        if event == "content_block_delta"
        and data["index"] == index
        and data["delta"]["type"] == "input_json_delta"
    )


async def test_tool_use_without_input_delta_emits_empty_object() -> None:
    """A tool_use block with no input delta yields an ``{}`` input_json_delta.

    The Anthropic SDK accumulates tool input as partial JSON and calls
    ``from_json(buffer)`` at ``content_block_stop``, which raises on an empty
    buffer, so the synthetic delta must be emitted before that stop frame.

    Ref: stdapi/models/chat/_adapters/_anthropic_message.py:_process_content_block_stop
    """
    pairs = await _collect(
        [
            {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "t1", "name": "get_time"}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
    )
    json_buf = _tool_input_json(pairs, 0)
    assert json_buf == "{}"
    assert loads(json_buf) == {}
    assert [event for event, _data in pairs] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ], "the backfilled input delta must precede content_block_stop"


async def test_tool_use_with_input_delta_is_unchanged() -> None:
    """A tool_use block with real input deltas keeps them and gets no extra ``{}``.

    Bedrock streams the tool input as partial JSON fragments, which map 1:1 onto
    Anthropic ``input_json_delta`` frames; the concatenation is the final input.

    Ref: stdapi/models/chat/_adapters/_anthropic_message.py:_map_delta
    """
    pairs = await _collect(
        [
            {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "t1", "name": "get_time"}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": '{"tz":'}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": '"utc"}'}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "tool_use"}},
        ]
    )
    json_buf = _tool_input_json(pairs, 0)
    assert json_buf == '{"tz":"utc"}'
    assert loads(json_buf) == {"tz": "utc"}
    input_deltas = [
        data
        for event, data in pairs
        if event == "content_block_delta"
        and data["delta"]["type"] == "input_json_delta"
    ]
    assert len(input_deltas) == 2, "no empty-object delta may be appended"


async def test_text_block_is_not_given_a_tool_input_delta() -> None:
    """A text delta with no preceding start frame yields text_delta only.

    The gateway synthesizes the missing ``content_block_start``; the empty-object
    backfill is reserved for tool-use blocks, whose input the SDK parses as JSON.

    Ref: stdapi/models/chat/_adapters/_anthropic_message.py:_emit_synthesized_block
    """
    pairs = await _collect(_text_stream_events("hello"))
    (start_data,) = [data for event, data in pairs if event == "content_block_start"]
    assert start_data["content_block"] == {"type": "text", "text": ""}
    (delta_data,) = [data for event, data in pairs if event == "content_block_delta"]
    assert delta_data["delta"] == {"type": "text_delta", "text": "hello"}
    assert not any(
        data.get("delta", {}).get("type") == "input_json_delta"
        for _event, data in pairs
    )


async def test_message_delta_always_carries_stop_sequence_key() -> None:
    """``message_delta.delta`` always includes ``stop_sequence``, null when unused.

    Converse-served (non-Claude) models never report a matched stop sequence, but
    Anthropic's wire format always includes the key, which ``exclude_none`` drops.

    Ref: stdapi/models/chat/_adapters/_anthropic_message.py:_make_message_delta_event
    """
    pairs = await _collect(_text_stream_events())
    (delta_data,) = [data for event, data in pairs if event == "message_delta"]
    assert "stop_sequence" in delta_data["delta"]
    assert delta_data["delta"]["stop_sequence"] is None
    assert delta_data["delta"]["stop_reason"] == "end_turn"


async def test_a_stream_closing_without_a_message_stop_still_ends_the_turn() -> None:
    """A backend that sends no ``messageStop`` still gets a final ``stop_reason``.

    A deployed Amazon Bedrock Marketplace model endpoint emits
    ``contentBlockStop`` and then simply closes the stream, so nothing carries a
    stop reason; ``exclude_none`` would then drop the key from ``message_delta``
    altogether and an SDK client branching on ``stop_reason`` -- ``tool_use``
    against ``end_turn`` -- would read the turn as unterminated. The default is
    the one the non-streamed path answers for the same backend, so the two agree.

    Ref: stdapi/models/chat/_adapters/_anthropic_message.py:_process_stream_events
         stdapi/models/marketplace_endpoints.py
         https://platform.claude.com/docs/en/api/messages-streaming
    """
    pairs = await _collect(
        [
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
        ]
    )
    (delta_data,) = [data for event, data in pairs if event == "message_delta"]

    assert delta_data["delta"]["stop_reason"] == "end_turn"
    assert pairs[-1][0] == "message_stop"


async def test_message_start_always_carries_stop_reason_and_sequence_keys() -> None:
    """``message_start.message`` always includes ``stop_reason``/``stop_sequence``.

    Anthropic's wire format serializes both as explicit ``null`` in
    ``message_start``, whose ``Message`` also carries an empty ``content`` list
    and echoes the model, but ``exclude_none`` drops null fields by default.

    Ref: stdapi/models/chat/_adapters/_anthropic_message.py:_make_message_start_event
    """
    pairs = await _collect(_text_stream_events())
    (start_data,) = [data for event, data in pairs if event == "message_start"]
    message = start_data["message"]
    assert "stop_reason" in message
    assert message["stop_reason"] is None
    assert "stop_sequence" in message
    assert message["stop_sequence"] is None
    assert message["type"] == "message"
    assert message["role"] == "assistant"
    assert message["content"] == []
    assert message["model"] == "model-x"


async def test_redacted_thinking_delta_is_not_dropped() -> None:
    """A ``reasoningContent.redactedContent`` delta yields a ``redacted_thinking`` block.

    Bedrock delivers redacted reasoning as raw bytes with no textual delta, so the
    payload must be base64-encoded into the emitted start block rather than
    surfacing as an empty ``thinking`` block.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_process_content_block_stop
    """
    pairs = await _collect(
        [
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"redactedContent": b"secret"}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )
    (start_data,) = [data for event, data in pairs if event == "content_block_start"]
    block = start_data["content_block"]
    assert block["type"] == "redacted_thinking"
    assert block["data"] == "c2VjcmV0", "the redacted bytes must be base64-encoded"
    assert not any(event == "content_block_delta" for event, _data in pairs), (
        "redacted reasoning carries no delta; the payload lives in the start block"
    )


async def test_redacted_thinking_spanning_several_deltas_keeps_every_chunk() -> None:
    """``redactedContent`` split across deltas is emitted as one complete block.

    Anthropic streaming has no ``redacted_thinking`` delta type: the whole
    payload lives in a single ``content_block_start``.  Bedrock may chunk the
    bytes over several deltas of the same block, so every chunk must be
    buffered until ``contentBlockStop`` — emitting only the first one would
    corrupt the payload Anthropic expects replayed verbatim on the next turn.

    Ref: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlockDelta.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_process_content_block_stop
    """
    pairs = await _collect(
        [
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"redactedContent": b"sec"}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"redactedContent": b"ret"}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "hi"}}},
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )
    starts = [data for event, data in pairs if event == "content_block_start"]
    assert [start["content_block"]["type"] for start in starts] == [
        "redacted_thinking",
        "text",
    ]
    assert starts[0]["content_block"]["data"] == "c2VjcmV0", (
        "both chunks must be concatenated before base64 encoding"
    )
    assert starts[0]["index"] == 0
    assert starts[1]["index"] == 1, "the following text block must get the next index"


class TestForcedToolSuppression:
    """``tool_choice`` naming one tool drops the streamed calls to any other tool.

    Converse's ``toolChoice.tool`` mandates a tool but does not forbid the others,
    so the model may still stream a call to a tool the caller excluded; the
    non-streaming path filters those blocks out and the stream must match, keeping
    the Anthropic block indices contiguous from zero.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
         https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools
         stdapi/models/chat/_adapters/_anthropic_message.py:_process_content_block_start
    """

    @staticmethod
    def _tool_start(index: int, name: str) -> dict[str, Any]:
        """Return a ``contentBlockStart`` event opening a tool_use block for *name*."""
        return {
            "contentBlockStart": {
                "contentBlockIndex": index,
                "start": {"toolUse": {"toolUseId": f"t{index}", "name": name}},
            }
        }

    async def test_unforced_tool_block_emits_no_events(self) -> None:
        """A tool_use block for another tool produces no SSE event at all.

        Neither ``content_block_start`` nor the synthetic empty-input delta nor
        ``content_block_stop`` may be emitted: a half-open block would leave the
        Anthropic SDK accumulating a block it never closes.
        """
        pairs = await _collect(
            [
                self._tool_start(0, "other_tool"),
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "tool_use"}},
            ],
            forced_tool="get_time",
        )
        assert [event for event, _data in pairs] == [
            "message_start",
            "message_delta",
            "message_stop",
        ]

    async def test_forced_tool_keeps_index_zero_after_a_dropped_block(self) -> None:
        """The forced tool's block is indexed from zero despite the dropped one.

        Anthropic indices count emitted blocks, not Bedrock ones, so a suppressed
        block must not consume an index — a gap would break clients that address
        content by index.
        """
        pairs = await _collect(
            [
                self._tool_start(0, "other_tool"),
                {"contentBlockStop": {"contentBlockIndex": 0}},
                self._tool_start(1, "get_time"),
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 1,
                        "delta": {"toolUse": {"input": '{"tz":"utc"}'}},
                    }
                },
                {"contentBlockStop": {"contentBlockIndex": 1}},
                {"messageStop": {"stopReason": "tool_use"}},
            ],
            forced_tool="get_time",
        )
        starts = [data for event, data in pairs if event == "content_block_start"]
        assert [start["content_block"]["name"] for start in starts] == ["get_time"]
        assert starts[0]["index"] == 0
        assert _tool_input_json(pairs, 0) == '{"tz":"utc"}'

    async def test_dropped_block_keeps_its_input_deltas_off_the_wire(self) -> None:
        """The excluded tool's arguments never reach the client.

        A suppressed block still receives its ``input_json_delta`` events from
        Bedrock. Those deltas carry the arguments of the tool the caller ruled
        out, and they arrive with no open Anthropic block, so the delta handler
        would otherwise synthesize one for them -- publishing the excluded call
        under a nameless ``tool_use`` block the non-streaming path drops.

        Ref: stdapi/models/chat/_adapters/_anthropic_message.py:_process_content_block_delta
        """
        pairs = await _collect(
            [
                self._tool_start(0, "other_tool"),
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"toolUse": {"input": '{"secret":"leak"}'}},
                    }
                },
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "tool_use"}},
            ],
            forced_tool="get_time",
        )
        assert [event for event, _data in pairs] == [
            "message_start",
            "message_delta",
            "message_stop",
        ]
        assert "leak" not in str(pairs)

    async def test_matching_tool_block_is_untouched(self) -> None:
        """The forced tool's own block streams normally when it is the only one.

        This is the control case: the filter must not fire when every streamed
        tool_use names the forced tool.
        """
        pairs = await _collect(
            [
                self._tool_start(0, "get_time"),
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "tool_use"}},
            ],
            forced_tool="get_time",
        )
        (start_data,) = [
            data for event, data in pairs if event == "content_block_start"
        ]
        assert start_data["content_block"]["name"] == "get_time"
        assert any(event == "content_block_stop" for event, _data in pairs)


async def test_message_delta_usage_reads_bedrock_cache_token_keys() -> None:
    """Streaming usage maps Bedrock's ``cacheRead/WriteInputTokens`` counters.

    Bedrock's ``TokenUsage`` has no ``cacheCreationInputTokens`` key: cache
    writes arrive as ``cacheWriteInputTokens`` in the ``metadata`` event and
    must surface as ``cache_creation_input_tokens`` in the final
    ``message_delta`` usage instead of being omitted.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
         https://platform.claude.com/docs/en/build-with-claude/streaming
         stdapi/models/chat/_adapters/_anthropic_message.py:_make_message_delta_event
    """
    pairs = await _collect(
        [
            *_text_stream_events(),
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 10,
                        "outputTokens": 5,
                        "cacheReadInputTokens": 3,
                        "cacheWriteInputTokens": 7,
                    }
                }
            },
        ]
    )
    (delta_data,) = [data for event, data in pairs if event == "message_delta"]
    usage = delta_data["usage"]
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 5
    assert usage["cache_read_input_tokens"] == 3
    assert usage["cache_creation_input_tokens"] == 7


class TestStreamedCitations:
    """A cited answer carries the same citations streamed as it does whole.

    Anthropic streams each citation as a ``citations_delta`` inside a
    ``content_block_delta`` event, which adds it to the ``citations`` list of the
    current ``text`` block; Bedrock carries the same metadata in a ``citation``
    content block delta.

    Ref: https://platform.claude.com/docs/en/build-with-claude/citations
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlockDelta.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_delta
    """

    #: Bedrock citation of a document passage, as carried by a ``citation`` delta.
    CITATION: ClassVar[dict[str, Any]] = {
        "title": "Guide",
        "source": "https://example.com/guide",
        "sourceContent": [{"text": "42"}],
        "location": {"documentChar": {"documentIndex": 0, "start": 0, "end": 2}},
    }
    #: Answer text the cited passage supports.
    TEXT: ClassVar[str] = "The answer is 42."

    @staticmethod
    def _citation_delta(
        citation: dict[str, Any], index: int = 0
    ) -> dict[str, dict[str, Any]]:
        """Return a Converse ``contentBlockDelta`` event carrying *citation*."""
        return {
            "contentBlockDelta": {
                "contentBlockIndex": index,
                "delta": {"citation": citation},
            }
        }

    @staticmethod
    def _citations(pairs: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        """Return the ``citations_delta`` payloads among *pairs*."""
        return [
            data
            for event, data in pairs
            if event == "content_block_delta"
            and data["delta"]["type"] == "citations_delta"
        ]

    async def test_citation_delta_reaches_the_open_text_block(self) -> None:
        """A ``citation`` delta becomes a ``citations_delta`` on the streamed text block.

        Anthropic attaches the citation to the block being written, so the delta
        must carry the index of the open text block rather than opening one of
        its own, and the cited passage comes from the Bedrock ``sourceContent``.
        """
        pairs = await _collect(
            [
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": self.TEXT},
                    }
                },
                self._citation_delta(self.CITATION),
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
        assert [event for event, _data in pairs] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        (citation_data,) = self._citations(pairs)
        (start_data,) = [
            data for event, data in pairs if event == "content_block_start"
        ]
        assert citation_data["index"] == start_data["index"]
        assert start_data["content_block"]["type"] == "text"
        citation = citation_data["delta"]["citation"]
        assert citation["type"] == "char_location"
        assert citation["cited_text"] == "42"
        assert citation["document_index"] == 0
        assert citation["start_char_index"] == 0
        assert citation["end_char_index"] == 2
        assert citation["document_title"] == "Guide"

    async def test_streamed_citation_frame_is_the_anthropic_one(self) -> None:
        """The emitted frame validates as the SDK's ``citations_delta`` event.

        The Anthropic SDK discriminates both the delta and the citation on their
        ``type``, so a frame it cannot resolve to ``CitationsDelta`` and
        ``CitationCharLocation`` is a frame its citation accumulator drops.
        """
        pairs = await _collect(
            [
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": self.TEXT},
                    }
                },
                self._citation_delta(self.CITATION),
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
        (citation_data,) = self._citations(pairs)
        event = SdkRawContentBlockDeltaEvent.model_validate(citation_data)
        assert isinstance(event.delta, SdkCitationsDelta)
        assert isinstance(event.delta.citation, SdkCitationCharLocation)
        assert event.delta.citation.cited_text == "42"

    async def test_streamed_citation_matches_the_non_streamed_one(self) -> None:
        """The same answer carries the same citation streamed or whole.

        Bedrock reports a cited answer as one ``citationsContent`` block when the
        response is whole and as ``text`` plus ``citation`` deltas when it
        streams, so a client that turns streaming on must not lose the citations
        the non-streamed call returns.
        """
        pairs = await _collect(
            [
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": self.TEXT},
                    }
                },
                self._citation_delta(self.CITATION),
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
        message = await format_response(
            contents=cast(
                "list[ContentBlockOutputTypeDef]",
                [
                    {
                        "citationsContent": {
                            "content": [{"text": self.TEXT}],
                            "citations": [self.CITATION],
                        }
                    }
                ],
            ),
            stop_reason="end_turn",
            usage={"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            message_id="msg_1",
            model_id="model-x",
            forced_tool=None,
            resp_map_tool_result=lambda *_args: None,
        )
        (block,) = message.content
        assert isinstance(block, TextBlock)
        assert block.citations is not None
        streamed_text = "".join(
            data["delta"]["text"]
            for event, data in pairs
            if event == "content_block_delta" and data["delta"]["type"] == "text_delta"
        )
        assert streamed_text == block.text
        (citation_data,) = self._citations(pairs)
        assert citation_data["delta"]["citation"] == block.citations[0].model_dump(
            mode="json", exclude_none=True
        )

    async def test_citation_after_the_text_block_still_reaches_the_client(self) -> None:
        """A citation arriving once the text block closed opens a text block of its own.

        A backend that reports its citations after the answer would otherwise
        have them dropped, since a ``citations_delta`` needs an open text block;
        the synthesized block mirrors the non-streamed answer, where a citation
        with no generated text of its own becomes an empty text block carrying it.
        """
        pairs = await _collect(
            [
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": self.TEXT},
                    }
                },
                {"contentBlockStop": {"contentBlockIndex": 0}},
                self._citation_delta(self.CITATION, index=1),
                {"contentBlockStop": {"contentBlockIndex": 1}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
        starts = [data for event, data in pairs if event == "content_block_start"]
        assert [start["content_block"]["type"] for start in starts] == ["text", "text"]
        assert [start["index"] for start in starts] == [0, 1]
        (citation_data,) = self._citations(pairs)
        assert citation_data["index"] == 1
        assert citation_data["delta"]["citation"]["cited_text"] == "42"

    async def test_a_citation_sent_before_its_text_shares_the_same_block(self) -> None:
        """A citation leading its answer opens the block the answer then fills.

        Nothing guarantees the cited passage is reported after the text it
        supports, and a citation that opened a block of its own would leave the
        client with an empty text block beside the answer.
        """
        pairs = await _collect(
            [
                self._citation_delta(self.CITATION),
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": self.TEXT},
                    }
                },
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
        (start_data,) = [
            data for event, data in pairs if event == "content_block_start"
        ]
        assert start_data["content_block"]["type"] == "text"
        (citation_data,) = self._citations(pairs)
        (text_data,) = [
            data
            for event, data in pairs
            if event == "content_block_delta" and data["delta"]["type"] == "text_delta"
        ]
        assert citation_data["index"] == start_data["index"]
        assert text_data["index"] == start_data["index"]

    @pytest.mark.parametrize(
        ("location", "citation_type"),
        [
            ({"documentChar": {"start": 0, "end": 2}}, "char_location"),
            ({"documentPage": {"start": 1, "end": 2}}, "page_location"),
            ({"documentChunk": {"start": 0, "end": 1}}, "content_block_location"),
            ({"web": {"url": "https://example.com"}}, "web_search_result_location"),
            (
                {
                    "searchResultLocation": {
                        "searchResultIndex": 0,
                        "start": 0,
                        "end": 1,
                    }
                },
                "search_result_location",
            ),
        ],
    )
    async def test_every_citation_location_streams_its_own_type(
        self, location: dict[str, Any], citation_type: str
    ) -> None:
        """Each Bedrock citation location streams as its Anthropic citation type.

        Bedrock reports where a citation points with a different member per
        source kind -- a document offset, a page, a chunk, a web result, a search
        result -- and Anthropic gives each one its own citation type, which the
        SDK selects on.
        """
        pairs = await _collect(
            [
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": self.TEXT},
                    }
                },
                self._citation_delta({**self.CITATION, "location": location}),
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
        (citation_data,) = self._citations(pairs)
        assert citation_data["delta"]["citation"]["type"] == citation_type

    async def test_a_citation_with_no_anthropic_equivalent_is_not_streamed(
        self,
    ) -> None:
        """A citation location the API cannot express streams no delta and no error.

        Bedrock may report a location kind Anthropic has no citation type for.
        The answer itself must still stream to the end, exactly as the
        non-streamed path answers with the text and without the citation.
        """
        pairs = await _collect(
            [
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": self.TEXT},
                    }
                },
                self._citation_delta(
                    {**self.CITATION, "location": {"somethingNewLocation": {"id": "1"}}}
                ),
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
        assert not self._citations(pairs)
        assert [event for event, _data in pairs] == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
