"""Stop sequences cut the returned content right after the first sequence its text holds.

Amazon Nova Micro checks stop sequences once per output chunk of about six
tokens and, when a chunk holds several matches, cuts it after the last one:
``stop=["green", "yellow"]`` on ``"red, blue, green, yellow"`` came back whole.
The gateway documents that generation halts at the first sequence matched and
nothing after it is returned, so ``converse``/``converse_stream`` cut the
content there themselves, keeping the matched sequence as the backend does.

All AWS calls are replaced by fakes: no Bedrock client is contacted.

Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html
     https://developers.openai.com/api/reference/resources/chat.md
     stdapi/models/_stop_sequences.py
     stdapi/models/__init__.py:ModelBase.converse
     stdapi/models/__init__.py:ModelBase.converse_stream
"""

from typing import TYPE_CHECKING, Any, cast

import pytest

import stdapi.models
from stdapi import usage
from stdapi.models import _DISCONNECT_DRAIN_MAX_EVENTS, ModelBase
from stdapi.models._stop_sequences import StreamCut, truncate_response_text
from stdapi.pricing import Dimension

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator

    from types_aiobotocore_bedrock_runtime.type_defs import (
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
    )

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef

#: Local, in-process tests metering into the per-request usage scopes.
pytestmark = [pytest.mark.local, pytest.mark.usefixtures("usage_scope")]

#: Usage block of the fake responses.
_USAGE = {"inputTokens": 13, "outputTokens": 7, "totalTokens": 20}

#: A tool call block of a Converse response.
_TOOL_USE = {"toolUse": {"toolUseId": "1", "name": "lookup", "input": {}}}


def _response(
    *content: dict[str, Any], stop_reason: str = "end_turn"
) -> ConverseResponseTypeDef:
    """Build a Converse response holding *content* blocks.

    Args:
        *content: Output content blocks.
        stop_reason: Reported ``stopReason``.

    Returns:
        A minimal Converse response.
    """
    return cast(
        "ConverseResponseTypeDef",
        {
            "output": {"message": {"role": "assistant", "content": list(content)}},
            "stopReason": stop_reason,
            "usage": dict(_USAGE),
        },
    )


def _text_delta(text: str, index: int = 0) -> dict[str, Any]:
    """Build a ConverseStream text delta event.

    Args:
        text: Delta text.
        index: Content block index.

    Returns:
        A ``contentBlockDelta`` event.
    """
    return {"contentBlockDelta": {"delta": {"text": text}, "contentBlockIndex": index}}


def _block_stop(index: int) -> dict[str, Any]:
    """Build a ConverseStream ``contentBlockStop`` event.

    Args:
        index: Content block index.

    Returns:
        The event.
    """
    return {"contentBlockStop": {"contentBlockIndex": index}}


def _tool_use_start(index: int) -> dict[str, Any]:
    """Build the ``contentBlockStart`` event opening a tool call block.

    Args:
        index: Content block index.

    Returns:
        The event.
    """
    return {
        "contentBlockStart": {
            "start": {"toolUse": {"toolUseId": "1", "name": "lookup"}},
            "contentBlockIndex": index,
        }
    }


def _message_stop(stop_reason: str) -> dict[str, Any]:
    """Build a ConverseStream ``messageStop`` event.

    Args:
        stop_reason: Reported ``stopReason``.

    Returns:
        The event.
    """
    return {"messageStop": {"stopReason": stop_reason}}


#: The trailing usage event of the fake streams.
_METADATA = {"metadata": {"usage": _USAGE}}


def _cut(stop_sequences: list[str], *events: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the *events* a :class:`StreamCut` lets through.

    Args:
        stop_sequences: The request's stop sequences.
        *events: Stream events, in order.

    Returns:
        The events kept, as edited by the cut.
    """
    cut = StreamCut(stop_sequences)
    return [
        event
        for event in events
        if cut.keep(cast("ConverseStreamOutputTypeDef", event))
    ]


def _streamed_text(events: list[Any]) -> str:
    """Concatenate the text deltas of *events*.

    Args:
        events: Stream events.

    Returns:
        The streamed text.
    """
    return "".join(
        event["contentBlockDelta"]["delta"].get("text", "")
        for event in events
        if "contentBlockDelta" in event
    )


async def _stream(
    *events: dict[str, Any],
) -> AsyncIterator[ConverseStreamOutputTypeDef]:
    """Yield *events* as a ConverseStream event stream.

    Args:
        *events: Stream events, in order.

    Yields:
        Each event.
    """
    for event in events:
        yield cast("ConverseStreamOutputTypeDef", event)


async def _collect(
    stream: AsyncIterable[ConverseStreamOutputTypeDef],
) -> list[ConverseStreamOutputTypeDef]:
    """Return every event of *stream*.

    Args:
        stream: Event stream.

    Returns:
        The events, in order.
    """
    return [event async for event in stream]


class TestTruncateResponseText:
    """truncate_response_text: a Converse response ends at the first stop sequence.

    Ref: stdapi/models/_stop_sequences.py:truncate_response_text
    """

    def test_cuts_after_the_first_of_several_matched_sequences(self) -> None:
        """The sequence matched first wins, whatever its place in the list."""
        response = _response({"text": "red, blue, green, yellow"})

        truncate_response_text(response, ["yellow", "green"])

        assert response["output"]["message"]["content"] == [
            {"text": "red, blue, green"}
        ]

    def test_cuts_after_the_first_occurrence_of_a_repeated_sequence(self) -> None:
        """A sequence occurring twice halts at its first occurrence."""
        response = _response({"text": "red, green, blue, green"})

        truncate_response_text(response, ["green"])

        assert response["output"]["message"]["content"] == [{"text": "red, green"}]

    def test_overlapping_sequences_halt_where_the_first_one_completes(self) -> None:
        """Generation halts on the sequence completed first, not the one started first."""
        response = _response({"text": "xabcdy"})

        truncate_response_text(response, ["abcd", "bc"])

        assert response["output"]["message"]["content"] == [{"text": "xabc"}]

    def test_drops_every_later_block(self) -> None:
        """Nothing after the match is returned, text or tool call alike."""
        response = _response(
            {"text": "one STOP two"}, {"text": "three"}, _TOOL_USE, {"text": "four"}
        )

        truncate_response_text(response, ["STOP"])

        assert response["output"]["message"]["content"] == [{"text": "one STOP"}]

    def test_a_tool_call_cut_off_reports_end_turn(self) -> None:
        """A ``tool_use`` stop with its only tool call cut becomes ``end_turn``."""
        response = _response(
            {"text": "one STOP two"}, _TOOL_USE, stop_reason="tool_use"
        )

        truncate_response_text(response, ["STOP"])

        assert response["output"]["message"]["content"] == [{"text": "one STOP"}]
        assert response["stopReason"] == "end_turn"

    def test_a_tool_call_before_the_match_keeps_tool_use(self) -> None:
        """A tool call generated before the match stays, with its stop reason."""
        response = _response(
            _TOOL_USE, {"text": "one STOP two"}, stop_reason="tool_use"
        )

        truncate_response_text(response, ["STOP"])

        assert response["output"]["message"]["content"] == [
            _TOOL_USE,
            {"text": "one STOP"},
        ]
        assert response["stopReason"] == "tool_use"

    def test_content_without_a_match_is_unchanged(self) -> None:
        """A response holding no stop sequence is returned as it came."""
        response = _response({"text": "red, blue"}, _TOOL_USE, stop_reason="tool_use")

        truncate_response_text(response, ["yellow"])

        assert response == _response(
            {"text": "red, blue"}, _TOOL_USE, stop_reason="tool_use"
        )


class TestStreamCut:
    """StreamCut: a ConverseStream ends at the first stop sequence.

    Ref: stdapi/models/_stop_sequences.py:StreamCut
    """

    def test_cuts_the_delta_and_drops_later_text(self) -> None:
        """Text after the match is dropped; the closing and usage events still pass."""
        events = _cut(
            ["green", "yellow"],
            {"messageStart": {"role": "assistant"}},
            _text_delta("red"),
            _text_delta(", blue, green, yellow"),
            _text_delta(", purple"),
            _block_stop(0),
            _message_stop("end_turn"),
            _METADATA,
        )

        assert events == [
            {"messageStart": {"role": "assistant"}},
            _text_delta("red"),
            _text_delta(", blue, green"),
            _block_stop(0),
            _message_stop("end_turn"),
            _METADATA,
        ]

    def test_a_sequence_spread_over_many_deltas_still_halts(self) -> None:
        """A match built one character at a time is caught where it completes."""
        events = _cut(
            ["green"],
            _text_delta("red, "),
            _text_delta("g"),
            _text_delta("r"),
            _text_delta("e"),
            _text_delta("e"),
            _text_delta("n, blue"),
            _text_delta(", green"),
        )

        assert _streamed_text(events) == "red, green"
        assert events[-1] == _text_delta("n")

    def test_no_match_across_two_content_blocks(self) -> None:
        """The end of one block and the start of the next never form a match."""
        events = _cut(["green"], _text_delta("red, gr", 0), _text_delta("een", 1))

        assert _streamed_text(events) == "red, green"

    def test_drops_every_later_block_and_reports_end_turn(self) -> None:
        """Reasoning, text and tool calls after the match never reach the client."""
        events = _cut(
            ["STOP"],
            _text_delta("one STOP two", 0),
            _block_stop(0),
            {
                "contentBlockDelta": {
                    "delta": {"reasoningContent": {"text": "think"}},
                    "contentBlockIndex": 1,
                }
            },
            _block_stop(1),
            _text_delta("three", 2),
            _block_stop(2),
            _tool_use_start(3),
            {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": "{}"}},
                    "contentBlockIndex": 3,
                }
            },
            _block_stop(3),
            _message_stop("tool_use"),
            _METADATA,
        )

        assert events == [
            _text_delta("one STOP", 0),
            _block_stop(0),
            _message_stop("end_turn"),
            _METADATA,
        ]

    def test_a_tool_call_before_the_match_keeps_tool_use(self) -> None:
        """A tool call streamed before the match stays, with its stop reason."""
        events = _cut(
            ["STOP"],
            _tool_use_start(0),
            _block_stop(0),
            _text_delta("one STOP two", 1),
            _block_stop(1),
            _message_stop("tool_use"),
        )

        assert events == [
            _tool_use_start(0),
            _block_stop(0),
            _text_delta("one STOP", 1),
            _block_stop(1),
            _message_stop("tool_use"),
        ]

    def test_content_without_a_match_is_unchanged(self) -> None:
        """A stream holding no stop sequence passes every event as it came."""
        source = [
            _text_delta("red"),
            _block_stop(0),
            _tool_use_start(1),
            _block_stop(1),
            _message_stop("tool_use"),
        ]

        assert _cut(["yellow"], *source) == source


class TestStreamCutUsage:
    """_capture_stream_usage with a StreamCut: billing reads every event, kept or not.

    Ref: stdapi/models/__init__.py:ModelBase._capture_stream_usage
         stdapi/models/__init__.py:_DISCONNECT_DRAIN_MAX_EVENTS
    """

    async def test_a_disconnect_after_the_cut_still_records_usage(self) -> None:
        """The drain reads through the dropped deltas to the trailing usage event."""
        model: ModelBase[Any, Any] = ModelBase("stopmodel")
        wrapped = model._capture_stream_usage(  # noqa: SLF001 -- the stream wrapper under test
            _stream(
                _text_delta("red, green"),
                *(_text_delta(", blue") for _ in range(3)),
                _METADATA,
            ),
            keep=StreamCut(["green"]).keep,
        )

        assert await anext(wrapped) == _text_delta("red, green")
        await wrapped.aclose()

        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.INPUT_TOKENS] == 13
        assert record.quantities[Dimension.OUTPUT_TOKENS] == 7

    async def test_dropped_events_count_toward_the_drain_bound(self) -> None:
        """A disconnect reads no more upstream events than the drain allows."""
        pulled = 0

        async def counted() -> AsyncIterator[ConverseStreamOutputTypeDef]:
            nonlocal pulled
            for event in (
                _text_delta("red, green"),
                *(
                    _text_delta(", blue")
                    for _ in range(2 * _DISCONNECT_DRAIN_MAX_EVENTS)
                ),
                _METADATA,
            ):
                pulled += 1
                yield cast("ConverseStreamOutputTypeDef", event)

        model: ModelBase[Any, Any] = ModelBase("stopmodel")
        wrapped = model._capture_stream_usage(  # noqa: SLF001 -- the stream wrapper under test
            counted(), keep=StreamCut(["green"]).keep
        )

        await anext(wrapped)
        await wrapped.aclose()

        assert pulled == 1 + _DISCONNECT_DRAIN_MAX_EVENTS


async def _noop_prepare(_self: object, _request: object, _region: str) -> None:
    """No-op stand-in for ModelBase._prepare_converse_request_for_region."""


class _FakeClient:
    """Fake Bedrock runtime client answering as Nova Micro does on ``stop``."""

    async def converse(self, **_request: object) -> dict[str, Any]:
        """Return the whole text, cut after the last of two matches.

        Returns:
            A Converse response.
        """
        return dict(_response({"text": "red, blue, green, yellow"}))

    async def converse_stream(self, **_request: object) -> dict[str, Any]:
        """Return the same answer as a ConverseStream event stream.

        Returns:
            A ConverseStream response.
        """
        return {
            "stream": _stream(
                _text_delta("red"),
                _text_delta(", blue, green, yellow"),
                _block_stop(0),
                _message_stop("end_turn"),
                {"metadata": {"usage": dict(_USAGE)}},
            )
        }


@pytest.mark.usefixtures("request_log")
class TestConverseHonorsStopSequences:
    """converse/converse_stream: the request's stop sequences cut the returned content.

    Ref: stdapi/models/__init__.py:ModelBase._converse
         stdapi/models/__init__.py:ModelBase._converse_stream
    """

    @pytest.fixture(autouse=True)
    def _fake_bedrock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Route every Converse call to :class:`_FakeClient` in one region."""

        async def _one_region(_model_id: str, **_kwargs: object) -> list[str]:
            return ["us-east-1"]

        monkeypatch.setattr(stdapi.models, "compute_candidate_regions", _one_region)
        monkeypatch.setattr(
            ModelBase, "_prepare_converse_request_for_region", _noop_prepare
        )
        monkeypatch.setattr(
            stdapi.models, "bedrock_client", lambda _region, **_kwargs: _FakeClient()
        )

    @staticmethod
    def _request(*stops: str) -> ConverseRequestBaseTypeDef:
        """Build a Converse request carrying *stops*.

        Args:
            *stops: Stop sequences; none leaves ``inferenceConfig`` empty.

        Returns:
            A Converse request.
        """
        config = {"stopSequences": list(stops)} if stops else {}
        return cast(
            "ConverseRequestBaseTypeDef",
            {"modelId": "stopmodel", "inferenceConfig": config},
        )

    async def test_converse_halts_at_the_first_matched_sequence(self) -> None:
        """Text the backend returned past the first match is not returned."""
        response = await ModelBase("stopmodel").converse(
            self._request("green", "yellow")
        )

        assert response["output"]["message"]["content"] == [
            {"text": "red, blue, green"}
        ]

    async def test_converse_bills_grounding_calls_the_cut_drops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A grounding call after the match is dropped from the reply, not the bill."""
        grounded = _response(
            {"text": "a STOP b"},
            {"toolUse": {"toolUseId": "t1", "name": "nova_grounding", "input": {}}},
        )

        async def _converse(_self: object, **_request: object) -> dict[str, Any]:
            return dict(grounded)

        billed: list[int] = []

        def _record(_self: object, _response: object, **kwargs: Any) -> None:  # noqa: ANN401 -- mirrors the recorder's keyword arguments
            billed.append(kwargs["grounding_requests"])

        monkeypatch.setattr(_FakeClient, "converse", _converse)
        monkeypatch.setattr(ModelBase, "_record_converse_usage", _record)

        response = await ModelBase("stopmodel").converse(self._request("STOP"))

        assert response["output"]["message"]["content"] == [{"text": "a STOP"}]
        assert billed == [1]

    async def test_converse_stream_halts_and_still_records_usage(self) -> None:
        """Streamed text past the first match is not returned; its usage is billed."""
        response = await ModelBase("stopmodel").converse_stream(
            self._request("green", "yellow")
        )

        assert _streamed_text(await _collect(response["stream"])) == (
            "red, blue, green"
        )
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.OUTPUT_TOKENS] == 7

    async def test_converse_without_stop_sequences_returns_the_text_whole(self) -> None:
        """Without ``stopSequences`` the text is returned as the backend sent it."""
        response = await ModelBase("stopmodel").converse(self._request())

        assert response["output"]["message"]["content"] == [
            {"text": "red, blue, green, yellow"}
        ]
