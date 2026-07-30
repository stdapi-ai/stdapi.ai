"""Anthropic SSE event synthesis from a Bedrock Converse stream (no AWS calls).

Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
     stdapi/models/chat/_adapters/_anthropic_message.py:format_stream
"""

from __future__ import annotations

from json import loads
from typing import TYPE_CHECKING, Any, cast

import pytest

from stdapi.models.chat._adapters._anthropic_message import format_stream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from types_aiobotocore_bedrock_runtime.type_defs import ConverseStreamOutputTypeDef

# The streaming adapter writes into the request log, which only exists inside a
# request, so every test needs the shared context fixture.
pytestmark = [pytest.mark.local, pytest.mark.usefixtures("request_log")]


async def _collect(events: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Run *events* through ``format_stream`` and return ``(event, data)`` pairs."""

    async def _stream() -> AsyncIterator[dict[str, Any]]:
        for event in events:
            yield event

    stream = cast("AsyncIterator[ConverseStreamOutputTypeDef]", _stream())
    return [
        (sse.event or "", loads(cast("str", sse.data)))
        async for sse in format_stream("msg_1", "model-x", stream, None)
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
    payload must be base64-encoded into the synthesized start block rather than
    surfacing as an empty ``thinking`` block.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_synthesize_block_from_delta
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
