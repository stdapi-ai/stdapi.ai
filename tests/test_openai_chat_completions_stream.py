"""Offline unit tests for the OpenAI Chat Completions streamed chunk shape.

``format_stream`` is driven over canned Bedrock Converse stream events, so the
emitted SSE frames can be compared to the upstream ones without any AWS call.

Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
     stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
"""

from __future__ import annotations

import json as _json
from typing import TYPE_CHECKING, Any

import pytest

from stdapi.config import SETTINGS
from stdapi.models.chat._adapters._openai_chat_completion import (
    _LEGACY_FUNCTION,
    format_stream,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

pytestmark = pytest.mark.local

#: Canned Bedrock Converse stream for one tool call: the opening block, an
#: argument fragment, the stop, then usage metadata.
_TOOL_STREAM_EVENTS: list[dict[str, Any]] = [
    {
        "contentBlockStart": {
            "contentBlockIndex": 0,
            "start": {"toolUse": {"toolUseId": "tooluse_1", "name": "get_weather"}},
        }
    },
    {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {"toolUse": {"input": '{"city": "Paris"}'}},
        }
    },
    {"messageStop": {"stopReason": "tool_use"}},
    {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
]

#: Canned Bedrock Converse stream for one text turn.
_TEXT_STREAM_EVENTS: list[dict[str, Any]] = [
    {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hi"}}},
    {"messageStop": {"stopReason": "end_turn"}},
    {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
]


async def _stub_converse_stream(
    events: list[dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:
    """Yield the given Bedrock Converse stream event dicts one by one.

    Args:
        events: Converse stream event dicts to replay.

    Yields:
        Each event dict, in order.
    """
    for event in events:
        yield event


@pytest.fixture
def _adapter_call_context(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Bind the per-request state ``format_stream`` reads outside a request.

    Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py:_LEGACY_FUNCTION
    """
    monkeypatch.setattr(SETTINGS, "log_request_params", False)
    token = _LEGACY_FUNCTION.set(False)
    try:
        yield
    finally:
        _LEGACY_FUNCTION.reset(token)


async def _chunks(
    events: list[dict[str, Any]], *, include_usage: bool = True
) -> list[dict[str, Any]]:
    """Drive ``format_stream`` over canned events and decode every chunk.

    Args:
        events: Converse stream event dicts to replay.
        include_usage: Value forwarded to ``format_stream``.

    Returns:
        The decoded JSON chunks, without the ``[DONE]`` sentinel.
    """
    return [
        _json.loads(data)
        async for event in format_stream(
            completion_id="chatcmpl-1",
            created=0,
            model_id="model",
            stream=_stub_converse_stream(events),  # type: ignore[arg-type]
            service_tier=None,
            include_usage=include_usage,
        )
        if isinstance(data := event.data, str) and data != "[DONE]"
    ]


@pytest.mark.usefixtures("_adapter_call_context")
class TestOpeningToolCallArguments:
    """The opening tool-call delta carries ``arguments: ""`` beside the name.

    Upstream opens a streamed tool call with ``{"name": ..., "arguments": ""}``,
    and every client accumulates the call with ``buffer += delta.function
    .arguments``. Omitting the key yields ``None`` in the OpenAI Python SDK --
    a ``TypeError`` on that line -- and ``undefined`` in a raw-HTTP JavaScript
    client, which corrupts the accumulated JSON.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
         stdapi/models/chat/_adapters/_openai_chat_completion.py:_stream_delta_chunk
    """

    async def test_tool_call_delta_opens_with_empty_arguments(self) -> None:
        """The first ``tool_calls`` delta names the tool and opens its arguments.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
        """
        chunks = await _chunks(_TOOL_STREAM_EVENTS)

        opening = next(
            chunk
            for chunk in chunks
            if chunk["choices"] and "tool_calls" in chunk["choices"][0]["delta"]
        )
        tool_call = opening["choices"][0]["delta"]["tool_calls"][0]
        assert tool_call["function"] == {"name": "get_weather", "arguments": ""}
        assert tool_call["id"] == "tooluse_1"

    async def test_accumulating_the_arguments_never_sees_a_null(self) -> None:
        """Concatenating every ``function.arguments`` fragment rebuilds the call.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
        """
        chunks = await _chunks(_TOOL_STREAM_EVENTS)

        buffer = ""
        for chunk in chunks:
            for choice in chunk["choices"]:
                for tool_call in choice["delta"].get("tool_calls", ()):
                    buffer += tool_call["function"]["arguments"]
        assert _json.loads(buffer) == {"city": "Paris"}

    async def test_legacy_function_call_opens_with_empty_arguments(self) -> None:
        """The deprecated ``function_call`` delta opens the same way upstream does.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
        """
        token = _LEGACY_FUNCTION.set(True)
        try:
            chunks = await _chunks(_TOOL_STREAM_EVENTS)
        finally:
            _LEGACY_FUNCTION.reset(token)

        opening = next(
            chunk
            for chunk in chunks
            if chunk["choices"] and "function_call" in chunk["choices"][0]["delta"]
        )
        assert opening["choices"][0]["delta"]["function_call"] == {
            "name": "get_weather",
            "arguments": "",
        }


@pytest.mark.usefixtures("_adapter_call_context")
class TestChunkChoiceAlwaysPresentKeys:
    """Every streamed choice carries ``finish_reason`` and ``logprobs``.

    Upstream sends both keys in every chunk choice, ``null`` until the terminal
    chunk. A statically-typed client decoding ``finish_reason`` into a plain
    string reads an absent key as ``""``, indistinguishable from a real value,
    and middleware detecting the end of a turn with ``"finish_reason" in choice``
    never fires.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
         stdapi/models/chat/_adapters/_openai_chat_completion.py:_dump_chunk
    """

    @pytest.mark.parametrize("events", [_TEXT_STREAM_EVENTS, _TOOL_STREAM_EVENTS])
    async def test_both_keys_are_present_in_every_choice(
        self, events: list[dict[str, Any]]
    ) -> None:
        """Both keys are present from the role chunk to the finish chunk.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
        """
        chunks = await _chunks(events)

        assert chunks, "the canned stream must produce chunks"
        for chunk in chunks:
            for choice in chunk["choices"]:
                assert "finish_reason" in choice
                assert "logprobs" in choice
                assert choice["logprobs"] is None
        assert chunks[0]["choices"][0]["finish_reason"] is None
        assert chunks[-2]["choices"][0]["finish_reason"] in {"stop", "tool_calls"}

    async def test_the_usage_chunk_keeps_its_empty_choices(self) -> None:
        """The trailing usage-only chunk carries no choice to annotate.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        """
        chunks = await _chunks(_TEXT_STREAM_EVENTS)

        assert chunks[-1]["choices"] == []
        assert chunks[-1]["usage"]["total_tokens"] == 15

    async def test_the_synthesized_finish_chunk_carries_both_keys(self) -> None:
        """A stream closing without ``messageStop`` still gets a complete choice.

        Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
        """
        chunks = await _chunks(
            [{"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hi"}}}]
        )

        final_choice = chunks[-1]["choices"][0]
        assert final_choice["finish_reason"] == "stop"
        assert final_choice["logprobs"] is None
