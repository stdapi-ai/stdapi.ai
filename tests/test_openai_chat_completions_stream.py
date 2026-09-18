"""Offline unit tests for the OpenAI Chat Completions streamed chunk shape.

``format_stream`` is driven over canned Bedrock Converse stream events, so the
emitted SSE frames can be compared to the upstream ones without any AWS call.

Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
     stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
"""

from __future__ import annotations

import json as _json
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from stdapi.config import SETTINGS
from stdapi.models.chat._adapters._openai_chat_completion import (
    _LEGACY_FUNCTION,
    format_stream,
)
from stdapi.types.openai import ChatModeration, ChatModerationResults, ModerationResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator

    from stdapi.types.openai_chat_completions import ServiceTiers

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
    events: list[dict[str, Any]],
    *,
    include_usage: bool = True,
    moderation_builder: Callable[[], ChatModeration | None] | None = None,
    service_tier: ServiceTiers | None = None,
) -> list[dict[str, Any]]:
    """Drive ``format_stream`` over canned events and decode every chunk.

    Args:
        events: Converse stream event dicts to replay.
        include_usage: Value forwarded to ``format_stream``.
        moderation_builder: Value forwarded to ``format_stream``.
        service_tier: Value forwarded to ``format_stream``.

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
            service_tier=service_tier,
            include_usage=include_usage,
            moderation_builder=moderation_builder,
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

    ``finish_reason`` holds across the fleet; ``logprobs`` does not. Measured on
    the vendor lane, ``gpt-4o-mini`` sends it on every choice and ``gpt-5-nano``
    never sends it at all, so emitting it always is a superset of the newer
    generation rather than a mirror of it. The upstream half of this is proved
    against ``gpt-4o-mini``, the one mapping that can still show it.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
         tests/test_openai_chat_completions.py:test_streaming_raw_chunk_keys_and_tool_call_opening_delta
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


def _moderation(*, flagged: bool) -> ChatModeration:
    """Build a canned ``moderation`` payload for a streamed completion.

    Args:
        flagged: Value carried by both directions of the result.

    Returns:
        A ``ChatModeration`` a stub builder can return.
    """
    result = ModerationResult(
        flagged=flagged,
        categories={"hate": flagged},
        category_scores={"hate": 0.75 if flagged else 0.0},
        category_applied_input_types={"hate": ["text"]},
        model="gr123",
    )
    return ChatModeration(
        input=ChatModerationResults(model="gr123", results=[result]),
        output=ChatModerationResults(model="gr123", results=[result]),
    )


@pytest.mark.usefixtures("_adapter_call_context")
class TestStreamedServiceTier:
    """The trailing chunks report the tier AWS says served the stream.

    Upstream sets the streamed ``service_tier`` from the processing mode that
    actually served the request. Amazon Bedrock only names it in the trailing
    ``ConverseStreamMetadataEvent``, by which time the content chunks are gone:
    they carry the tier the call was sent on -- already more accurate than the
    requested one, since an alias, a configured default or the tier header can
    set it in the request's place -- and the chunks after that event carry the
    served one.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStreamMetadataEvent.html
         stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
    """

    #: Canned text stream whose metadata event names the tier that served it.
    _SERVED_EVENTS: ClassVar[list[dict[str, Any]]] = [
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hi"}}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {"inputTokens": 10, "outputTokens": 5},
                "serviceTier": {"type": "flex"},
            }
        },
    ]

    async def test_the_usage_chunk_reports_the_served_tier(self) -> None:
        """A call sent on ``priority`` but served on ``flex`` ends on ``flex``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStreamMetadataEvent.html
        """
        chunks = await _chunks(self._SERVED_EVENTS, service_tier="priority")

        assert chunks[-1]["usage"]["total_tokens"] == 15
        assert chunks[-1]["service_tier"] == "flex"
        # The content chunks were already sent when the metadata event arrived.
        assert {chunk["service_tier"] for chunk in chunks[:-1]} == {"priority"}

    async def test_a_stream_naming_no_tier_keeps_the_one_it_was_sent_on(self) -> None:
        """Without a served tier every chunk reports the tier the call carried."""
        chunks = await _chunks(_TEXT_STREAM_EVENTS, service_tier="flex")

        assert {chunk["service_tier"] for chunk in chunks} == {"flex"}

    async def test_the_moderation_chunk_reports_the_served_tier(self) -> None:
        """The chunk sent after the usage one agrees with it on the tier."""
        chunks = await _chunks(
            self._SERVED_EVENTS,
            service_tier="priority",
            moderation_builder=partial(_moderation, flagged=False),
        )

        assert "moderation" in chunks[-1]
        assert chunks[-1]["service_tier"] == "flex"


@pytest.mark.usefixtures("_adapter_call_context")
class TestStreamedModeration:
    """A streamed completion reports its guardrail verdict in its own chunk.

    Upstream carries ``moderation`` on ``ChatCompletionChunk`` and delivers it
    on a dedicated moderation chunk rather than on a content one. The guardrail
    runs on a streamed request as it does on a buffered one, so the verdict is
    reported instead of dropped. Two divergences: Bedrock only sends the
    guardrail trace with the final metadata event, so the chunk lands after the
    usage one, and both directions arrive together rather than the input
    verdict arriving early.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
         openai.types.chat.chat_completion_chunk.ChatCompletionChunk.moderation
         stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
    """

    async def test_results_land_on_a_dedicated_trailing_chunk(self) -> None:
        """The builder result is sent alone, on the last chunk before ``[DONE]``.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
        """
        chunks = await _chunks(
            _TEXT_STREAM_EVENTS, moderation_builder=partial(_moderation, flagged=True)
        )

        moderation_chunk = chunks[-1]
        assert moderation_chunk["choices"] == []
        assert moderation_chunk["object"] == "chat.completion.chunk"
        assert moderation_chunk["id"] == "chatcmpl-1"
        assert moderation_chunk["moderation"]["input"]["results"][0]["flagged"] is True
        assert moderation_chunk["moderation"]["output"]["model"] == "gr123"
        # The verdict rides its own chunk: no content or usage chunk carries it.
        assert not any("moderation" in chunk for chunk in chunks[:-1])
        assert chunks[-2]["usage"]["total_tokens"] == 15

    async def test_no_extra_chunk_when_the_builder_returns_nothing(self) -> None:
        """A request without ``moderation`` keeps the upstream chunk sequence.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
        """
        chunks = await _chunks(_TEXT_STREAM_EVENTS, moderation_builder=lambda: None)

        assert chunks[-1]["choices"] == []
        assert chunks[-1]["usage"]["total_tokens"] == 15
        assert not any("moderation" in chunk for chunk in chunks)

    async def test_no_extra_chunk_without_a_builder(self) -> None:
        """No builder at all leaves the stream exactly as it was.

        Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
        """
        chunks = await _chunks(_TEXT_STREAM_EVENTS)

        assert not any("moderation" in chunk for chunk in chunks)
