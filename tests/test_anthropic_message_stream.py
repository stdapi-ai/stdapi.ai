"""Unit tests for the Anthropic messages streaming adapter.

A ``tool_use`` block that carries no input delta must still yield a valid
``input_json_delta``: the Anthropic SDK accumulates tool input as partial JSON
and calls ``from_json(buffer)`` at ``content_block_stop``, which raises on an
empty buffer.  Some models (e.g. weaker open-weight models) emit a tool call
with no arguments and no input delta, so the adapter backfills an ``{}`` delta.
"""

from __future__ import annotations

from json import loads
from typing import TYPE_CHECKING, Any, cast

import pytest

from stdapi.models.chat._adapters._anthropic_message import format_stream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from types_aiobotocore_bedrock_runtime.type_defs import ConverseStreamOutputTypeDef

pytestmark = pytest.mark.local


@pytest.fixture(autouse=True)
def _request_log_context() -> Iterator[None]:
    """Provide the request-log context the streaming adapter logs into."""
    from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

    token = REQUEST_LOG.set({"level": "info"})  # type: ignore[typeddict-item]
    yield
    REQUEST_LOG.reset(token)


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
    """A tool_use block with no input delta yields an ``{}`` input_json_delta."""
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


async def test_tool_use_with_input_delta_is_unchanged() -> None:
    """A tool_use block with a real input delta keeps it and gets no extra ``{}``."""
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


async def test_text_block_is_not_given_a_tool_input_delta() -> None:
    """A plain text block never receives a synthetic input_json_delta."""
    pairs = await _collect(
        [
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hello"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )
    assert not any(
        data.get("delta", {}).get("type") == "input_json_delta"
        for _event, data in pairs
    )


async def test_message_delta_always_carries_stop_sequence_key() -> None:
    """``message_delta.delta`` always includes ``stop_sequence``, null when unused.

    Converse-served (non-Claude) models never report a matched stop sequence,
    but Anthropic's wire format always includes the key.
    """
    pairs = await _collect(
        [
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )
    (delta_data,) = [data for event, data in pairs if event == "message_delta"]
    assert "stop_sequence" in delta_data["delta"]
    assert delta_data["delta"]["stop_sequence"] is None


async def test_message_start_always_carries_stop_reason_and_sequence_keys() -> None:
    """``message_start.message`` always includes ``stop_reason``/``stop_sequence``.

    Anthropic's wire format serializes both as explicit ``null`` in
    ``message_start``, but exclude_none drops them by default.
    """
    pairs = await _collect(
        [
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
    )
    (start_data,) = [data for event, data in pairs if event == "message_start"]
    message = start_data["message"]
    assert "stop_reason" in message
    assert message["stop_reason"] is None
    assert "stop_sequence" in message
    assert message["stop_sequence"] is None


async def test_redacted_thinking_delta_is_not_dropped() -> None:
    """A ``reasoningContent.redactedContent`` delta yields a real block.

    It must produce a ``redacted_thinking`` block instead of an empty
    ``thinking`` block.
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
    assert block["data"]
