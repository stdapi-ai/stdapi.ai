"""Amazon Nova Sonic as the Realtime API's backend, offline.

The live conversation protocol is the part of this model that a live test can
only sample: the handshake order, the one-content-block-per-turn framing, the
trailing silence that ends an utterance, the keepalive that keeps the service
from dropping a thinking pause, and the stage rule that decides whether a text
event is the answer or a restatement of it. All of that is driven here against
C-0's fake duplex stream, so a wrong move in it fails in CI rather than in a
customer's session.

Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-input-events.html
     https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-output-events.html
     stdapi/models/realtime/amazon_nova_sonic.py:RealtimeModel
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from json import dumps, loads
from math import ceil
from typing import TYPE_CHECKING, Any

import pytest
from aws_sdk_bedrock_runtime.models import (
    BidirectionalOutputPayloadPart,
    InvokeModelWithBidirectionalStreamOutputChunk,
)

import stdapi.aws_bidi
from stdapi.models.realtime import (
    InputTranscript,
    OutputAudio,
    OutputTranscript,
    ResponseFinished,
    ResponseStarted,
    SpeechStarted,
    SpeechStopped,
    UsageReport,
)
from stdapi.models.realtime import amazon_nova_sonic as sonic
from tests.test_aws_bidi import FakeDuplexStream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from stdapi.models.realtime import BackendEvent, RealtimeBackendSession

pytestmark = pytest.mark.local

#: The shipping Nova Sonic identifier this model class serves.
_MODEL = "amazon.nova-2-sonic-v1:0"

#: Region the offline tests pretend served the session.
_REGION = "us-east-1"

#: Sample rate every test opens its session at, in hertz.
_RATE = 24000

#: Seconds a test waits for the keepalive to send its silent frame.
_KEEPALIVE_TIMEOUT = 5.0

#: Bytes of 16-bit audio one input event carries at :data:`_RATE`.
_FRAME_BYTES = _RATE * 2 * 32 // 1000


def _output_chunk(event: dict[str, Any]) -> Any:  # noqa: ANN401
    """Wrap one output event in the SDK envelope the stream yields.

    Args:
        event: The event body, without its ``event`` wrapper.

    Returns:
        The SDK chunk carrying it as JSON bytes.
    """
    return InvokeModelWithBidirectionalStreamOutputChunk(
        value=BidirectionalOutputPayloadPart(bytes_=dumps({"event": event}).encode())
    )


def _sent(stream: FakeDuplexStream) -> list[dict[str, Any]]:
    """Decode every input event a fake stream received.

    Args:
        stream: The fake stream to read.

    Returns:
        The decoded event bodies, in send order, without the ``event`` wrapper.
    """
    return [loads(event.value.bytes_)["event"] for event in stream.input_stream.sent]


def _names(sent: list[dict[str, Any]]) -> list[str]:
    """Return the name of each sent event, for readable assertion messages."""
    return [next(iter(event)) for event in sent]


class _StreamPool:
    """Stands in for the generated client, handing out scripted streams."""

    def __init__(self) -> None:
        """Start with nothing scripted and nothing opened."""
        self.scripted: list[FakeDuplexStream] = []
        self.opened: list[FakeDuplexStream] = []

    def script(self, stream: FakeDuplexStream) -> None:
        """Queue *stream* as the next one opened.

        Args:
            stream: The scripted stream.
        """
        self.scripted.append(stream)

    async def invoke_model_with_bidirectional_stream(
        self, _input: object
    ) -> FakeDuplexStream:
        """Return the next scripted stream, or an empty one.

        Args:
            _input: The operation input (unused).

        Returns:
            The scripted duplex stream.
        """
        stream = self.scripted.pop(0) if self.scripted else FakeDuplexStream()
        self.opened.append(stream)
        return stream


@pytest.fixture
def streams(
    monkeypatch: pytest.MonkeyPatch,
    request_log: dict[str, Any],  # noqa: ARG001  (binds the log the region setter writes)
) -> _StreamPool:
    """Serve every bidirectional stream from a scripted fake in one region."""
    pool = _StreamPool()
    monkeypatch.setattr(
        stdapi.aws_bidi, "get_bidi_client", lambda _service, _region=None: pool
    )

    async def _regions(*_args: object, **_kwargs: object) -> list[str]:
        return [_REGION]

    monkeypatch.setattr(sonic, "compute_candidate_regions", _regions)
    return pool


@asynccontextmanager
async def _session(**kwargs: Any) -> AsyncIterator[RealtimeBackendSession]:  # noqa: ANN401
    """Open one conversation with the defaults every test shares.

    Args:
        kwargs: Overrides for the ``open_session`` arguments.

    Yields:
        The open backend session.
    """
    arguments: dict[str, Any] = {
        "instructions": "Be brief.",
        "input_sample_rate": _RATE,
        "output_sample_rate": _RATE,
    }
    arguments.update(kwargs)
    async with sonic.RealtimeModel(_MODEL).open_session(**arguments) as session:
        yield session


async def _translated() -> list[BackendEvent]:
    """Read the scripted stream through a session and return what it reported.

    Returns:
        Every neutral event the session yielded.
    """
    async with _session() as session:
        return [event async for event in session.events()]


class TestHandshake:
    """The service answers nothing at all until the priming events arrive.

    Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-input-events.html
    """

    async def test_the_session_primes_in_the_only_accepted_order(
        self, streams: _StreamPool
    ) -> None:
        """Session, prompt, then the system content block, which must come first."""
        async with _session(voice="sage", max_output_tokens=256, temperature=0.4):
            pass

        sent = _sent(streams.opened[0])
        assert _names(sent)[:5] == [
            "sessionStart",
            "promptStart",
            "contentStart",
            "textInput",
            "contentEnd",
        ], _names(sent)
        assert sent[0]["sessionStart"]["inferenceConfiguration"] == {
            "temperature": 0.4,
            "maxTokens": 256,
        }
        assert sent[2]["contentStart"]["role"] == "SYSTEM"
        assert sent[3]["textInput"]["content"] == "Be brief."
        audio = sent[1]["promptStart"]["audioOutputConfiguration"]
        assert audio["sampleRateHertz"] == _RATE
        assert audio["voiceId"] == "ruth"

    @pytest.mark.parametrize(
        ("requested", "served"),
        [
            ("alloy", "tiffany"),
            ("verse", "stephen"),
            ("joanna", "joanna"),
            (None, "matthew"),
        ],
    )
    async def test_the_voice_is_mapped_or_passed_through(
        self, streams: _StreamPool, requested: str | None, served: str
    ) -> None:
        """An OpenAI voice name is served by its match; anything else is the model's."""
        async with _session(voice=requested):
            pass

        sent = _sent(streams.opened[0])
        assert sent[1]["promptStart"]["audioOutputConfiguration"]["voiceId"] == served

    async def test_the_session_reports_the_region_that_served_it(
        self, streams: _StreamPool
    ) -> None:
        """Usage and the request log are attributed to the serving region."""
        assert streams is not None
        async with _session() as session:
            assert session.region == _REGION

    async def test_closing_ends_the_prompt_and_the_session(
        self, streams: _StreamPool
    ) -> None:
        """The goodbye is what releases the service's own session."""
        async with _session():
            pass

        assert _names(_sent(streams.opened[0]))[-2:] == ["promptEnd", "sessionEnd"]


class TestTurnFraming:
    """One turn is one audio content block, and ending it is what starts the answer.

    Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-input-events.html
    """

    async def test_audio_opens_one_block_and_is_framed(
        self, streams: _StreamPool
    ) -> None:
        """The block is opened once, and the audio is split into fixed frames."""
        async with _session() as session:
            await session.send_audio(bytes(_FRAME_BYTES * 2))
            await session.send_audio(bytes(_FRAME_BYTES))
            sent = _sent(streams.opened[0])[5:]
        assert _names(sent) == [
            "contentStart",
            "audioInput",
            "audioInput",
            "audioInput",
        ], _names(sent)
        opened = sent[0]["contentStart"]
        assert opened["type"] == "AUDIO"
        assert opened["role"] == "USER"
        assert opened["audioInputConfiguration"]["sampleRateHertz"] == _RATE
        assert sent[1]["audioInput"]["contentName"] == opened["contentName"]

    async def test_ending_a_turn_appends_silence_and_closes_the_block(
        self, streams: _StreamPool
    ) -> None:
        """A block ending on the last spoken sample is never answered at all."""
        async with _session() as session:
            await session.send_audio(bytes(_FRAME_BYTES))
            await session.end_turn()
            sent = _sent(streams.opened[0])[5:]

        # One frame of speech, then a whole second of trailing silence.
        assert _names(sent) == [
            "contentStart",
            *["audioInput"] * (1 + ceil(_RATE * 2 / _FRAME_BYTES)),
            "contentEnd",
        ], _names(sent)
        assert (
            sent[-1]["contentEnd"]["contentName"]
            == sent[0]["contentStart"]["contentName"]
        )

    async def test_an_ended_turn_is_not_ended_twice(self, streams: _StreamPool) -> None:
        """``end_turn`` with no open block sends nothing, rather than a stray end."""
        async with _session() as session:
            await session.end_turn()
            sent = _sent(streams.opened[0])[5:]

        assert sent == []

    async def test_the_next_turn_gets_its_own_content_block(
        self, streams: _StreamPool
    ) -> None:
        """Reusing the closed block's name attaches audio to a block already ended."""
        async with _session() as session:
            await session.send_audio(bytes(_FRAME_BYTES))
            await session.end_turn()
            await session.send_audio(bytes(_FRAME_BYTES))
            sent = _sent(streams.opened[0])[5:]

        starts = [event["contentStart"] for event in sent if "contentStart" in event]
        assert len(starts) == 2
        assert starts[0]["contentName"] != starts[1]["contentName"]

    async def test_a_written_message_is_its_own_text_block(
        self, streams: _StreamPool
    ) -> None:
        """Text is sent as a complete USER block, not into the audio one."""
        async with _session() as session:
            await session.send_text("Hello there.")
            sent = _sent(streams.opened[0])[5:]

        assert _names(sent) == ["contentStart", "textInput", "contentEnd"]
        assert sent[0]["contentStart"]["role"] == "USER"
        assert sent[0]["contentStart"]["type"] == "TEXT"
        assert sent[1]["textInput"]["content"] == "Hello there."

    async def test_a_written_turn_is_ended_through_a_silent_audio_block(
        self, streams: _StreamPool
    ) -> None:
        """A turn made only of text is still ended the way the model answers one.

        Verified against the model: a written block on its own is read and
        billed, and no answer ever comes -- the session then dies on the
        service's own idle timeout. The same text followed by an audio block
        carrying only silence is answered.

        Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-input-events.html
             stdapi/models/realtime/amazon_nova_sonic.py:_NovaSonicSession.end_turn
        """
        async with _session() as session:
            await session.send_text("Hello there.")
            await session.end_turn()
            sent = _sent(streams.opened[0])[5:]

        assert _names(sent) == [
            "contentStart",
            "textInput",
            "contentEnd",
            "contentStart",
            *["audioInput"] * ceil(_RATE * 2 / _FRAME_BYTES),
            "contentEnd",
        ], _names(sent)
        assert sent[3]["contentStart"]["type"] == "AUDIO"

    async def test_closing_a_session_does_not_start_one_last_answer(
        self, streams: _StreamPool
    ) -> None:
        """A session on its way out is not billed for an answer nobody reads.

        Ref: stdapi/models/realtime/amazon_nova_sonic.py:_NovaSonicSession.close
        """
        async with _session() as session:
            await session.send_text("Hello there.")
        sent = _sent(streams.opened[0])[5:]

        assert _names(sent) == [
            "contentStart",
            "textInput",
            "contentEnd",
            "promptEnd",
            "sessionEnd",
        ], _names(sent)

    async def test_the_keepalive_sends_silence_after_a_quiet_gap(
        self, streams: _StreamPool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller pausing between two questions must not lose the session."""
        import asyncio  # noqa: PLC0415

        monkeypatch.setattr(sonic, "_KEEPALIVE_SECONDS", 0.01)

        async with _session(), asyncio.timeout(_KEEPALIVE_TIMEOUT):
            # Polled: the frame is sent by the session's own task, which nothing awaits.
            while "audioInput" not in _names(_sent(streams.opened[0])):  # noqa: ASYNC110
                await asyncio.sleep(0.01)
            sent = _sent(streams.opened[0])[5:]

        assert _names(sent)[:2] == ["contentStart", "audioInput"], _names(sent)


class TestEventTranslation:
    """Every backend event that carries meaning, and every one that does not.

    Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-output-events.html
    """

    async def test_speech_boundaries_carry_their_offsets(
        self, streams: _StreamPool
    ) -> None:
        """Voice activity detection is reported with the offsets the client shows."""
        streams.script(
            FakeDuplexStream(
                events=[
                    _output_chunk({"userSpeechStart": {"inputAudioOffsetMs": 120}}),
                    _output_chunk({"userSpeechEnd": {"inputAudioOffsetMs": 2480}}),
                ]
            )
        )

        reported = await _translated()

        assert reported == [SpeechStarted(120), SpeechStopped(2480)]

    async def test_a_caller_block_is_reported_as_the_input_transcript(
        self, streams: _StreamPool
    ) -> None:
        """A text block the caller spoke is their transcript, not an answer."""
        streams.script(
            FakeDuplexStream(
                events=[
                    _output_chunk(
                        {"contentStart": {"contentId": "c1", "role": "USER"}}
                    ),
                    _output_chunk(
                        {
                            "textOutput": {
                                "contentId": "c1",
                                "content": "what time is it",
                            }
                        }
                    ),
                ]
            )
        )

        reported = await _translated()

        assert reported == [InputTranscript("what time is it")]

    async def test_a_final_block_does_not_repeat_the_answer(
        self, streams: _StreamPool
    ) -> None:
        """The FINAL stage restates the speculative answer; reporting both doubles it."""
        streams.script(
            FakeDuplexStream(
                events=[
                    _output_chunk({"completionStart": {}}),
                    _output_chunk(
                        {
                            "contentStart": {
                                "contentId": "spec",
                                "role": "ASSISTANT",
                                "additionalModelFields": dumps(
                                    {"generationStage": "SPECULATIVE"}
                                ),
                            }
                        }
                    ),
                    _output_chunk(
                        {"textOutput": {"contentId": "spec", "content": "Half past."}}
                    ),
                    _output_chunk(
                        {
                            "contentStart": {
                                "contentId": "final",
                                "role": "ASSISTANT",
                                "additionalModelFields": dumps(
                                    {"generationStage": "FINAL"}
                                ),
                            }
                        }
                    ),
                    _output_chunk(
                        {"textOutput": {"contentId": "final", "content": "Half past."}}
                    ),
                ]
            )
        )

        reported = await _translated()

        assert reported == [ResponseStarted(), OutputTranscript("Half past.")]

    async def test_empty_text_and_unknown_events_report_nothing(
        self, streams: _StreamPool
    ) -> None:
        """Whitespace, an unknown speaker and an unknown event name all pass."""
        streams.script(
            FakeDuplexStream(
                events=[
                    _output_chunk({"textOutput": {"contentId": "c1", "content": "  "}}),
                    _output_chunk(
                        {
                            "textOutput": {
                                "contentId": "c2",
                                "role": "TOOL",
                                "content": "x",
                            }
                        }
                    ),
                    _output_chunk({"somethingNew": {"whatever": 1}}),
                ]
            )
        )

        assert await _translated() == []

    async def test_spoken_audio_is_decoded(self, streams: _StreamPool) -> None:
        """The model's speech reaches the session as raw samples."""
        from base64 import b64encode  # noqa: PLC0415

        streams.script(
            FakeDuplexStream(
                events=[
                    _output_chunk(
                        {"audioOutput": {"content": b64encode(b"\x01\x00").decode()}}
                    )
                ]
            )
        )

        assert await _translated() == [OutputAudio(b"\x01\x00")]

    async def test_a_text_only_session_reports_no_audio(
        self, streams: _StreamPool
    ) -> None:
        """``speech_output=False`` drops the speech instead of sending it on."""
        from base64 import b64encode  # noqa: PLC0415

        streams.script(
            FakeDuplexStream(
                events=[
                    _output_chunk(
                        {"audioOutput": {"content": b64encode(b"\x01\x00").decode()}}
                    )
                ]
            )
        )

        async with _session(speech_output=False) as session:
            reported = [event async for event in session.events()]

        assert reported == []

    @pytest.mark.parametrize(
        ("stop_reason", "expected"),
        [
            ("INTERRUPTED", [ResponseFinished(interrupted=True)]),
            ("END_TURN", [ResponseFinished()]),
            ("PARTIAL_TURN", []),
        ],
    )
    async def test_the_stop_reason_decides_how_an_answer_ended(
        self, streams: _StreamPool, stop_reason: str, expected: list[BackendEvent]
    ) -> None:
        """A caller who spoke over the answer must not see it reported as complete."""
        streams.script(
            FakeDuplexStream(
                events=[_output_chunk({"contentEnd": {"stopReason": stop_reason}})]
            )
        )

        assert await _translated() == expected

    async def test_metering_is_read_as_running_totals(
        self, streams: _StreamPool
    ) -> None:
        """Totals are restated by every event, so a lost one costs nothing."""
        streams.script(
            FakeDuplexStream(
                events=[
                    _output_chunk(
                        {
                            "usageEvent": {
                                "totalTokens": 623,
                                "details": {
                                    "total": {
                                        "input": {
                                            "speechTokens": 197,
                                            "textTokens": 348,
                                        },
                                        "output": {
                                            "speechTokens": 52,
                                            "textTokens": 26,
                                        },
                                    }
                                },
                            }
                        }
                    )
                ]
            )
        )

        assert await _translated() == [
            UsageReport(
                input_speech_tokens=197,
                input_text_tokens=348,
                output_speech_tokens=52,
                output_text_tokens=26,
                total_tokens=623,
            )
        ]
