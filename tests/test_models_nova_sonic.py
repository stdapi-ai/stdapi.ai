"""Amazon Nova Sonic as a transcription and translation backend.

Nova Sonic answers on a *bidirectional* model stream rather than a
request/response API, and every wrong move on it fails silently: a mis-declared
audio format, a missing handshake block or a session torn down too early produce
no error and no transcript, only a connection the service closes about a minute
later. The offline tests here drive the session against C-0's fake duplex stream
and pin the bounds and the accounting; the live ones pin what the model actually
returns, tolerantly.

Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-input-events.html
     https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-output-events.html
     https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
     stdapi/models/audio/amazon_nova_sonic.py:AudioModel
"""

from __future__ import annotations

import io
import wave
from asyncio import Future
from json import dumps
from math import ceil
from typing import TYPE_CHECKING, Any

import pytest
from aws_sdk_bedrock_runtime.models import (
    BidirectionalOutputPayloadPart,
    InvokeModelWithBidirectionalStreamOutputChunk,
)
from openai import BadRequestError

import stdapi.aws_bidi
import stdapi.routes.openai_audio_transcriptions
import stdapi.routes.openai_audio_translations
from stdapi.api_errors import ApiError
from stdapi.aws_bidi import BidiSession
from stdapi.models import ModelDetails, _compute_model_capabilities
from stdapi.models.audio import amazon_nova_sonic as sonic
from stdapi.models.audio import get_audio_model
from stdapi.pricing import Dimension, Service, resolve_price
from stdapi.types.openai_audio import TranscriptionTextDoneEvent
from stdapi.usage import USAGE, init_usage, usage_log_entries
from tests.conftest import set_test_price
from tests.test_aws_bidi import FakeDuplexStream

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from openai import OpenAI

#: The shipping Nova Sonic identifier; the v1 generation is legacy on AWS.
_MODEL = "amazon.nova-2-sonic-v1:0"

#: The legacy identifier, reachable only when legacy models are enabled.
_LEGACY_MODEL = "amazon.nova-sonic-v1:0"

#: Region the offline tests pretend served the session.
_REGION = "us-east-1"

#: Sentence the pause fixture speaks before its silent gap.
_PAUSE_FIRST = "The harbour was quiet."

#: Sentence the pause fixture speaks after its silent gap.
_PAUSE_SECOND = "Seventeen violet umbrellas."

#: Seconds of silence separating the pause fixture's two sentences.
_PAUSE_SECONDS = 3

#: French sentence the translation fixtures speak.
_FRENCH_SENTENCE = "Je travaille à Paris depuis trois ans."

#: Words that must survive transcription of the French fixture.
_FRENCH_WORDS = ("paris", "travaille")

#: Words an English translation of :data:`_FRENCH_SENTENCE` must contain.
_ENGLISH_WORDS = ("work", "paris", "years")

#: What the encoding pipeline says when the upload itself could not be decoded.
_ENCODE_FAILED = "Failed to encode the audio to 'pcm'."

#: What it says when the encode produced nothing in time.
_ENCODE_STALLED = "Timed out encoding the audio to 'pcm'."


def _model_details(model_id: str) -> ModelDetails:
    """Build catalog details for a speech-in, text-out model.

    Args:
        model_id: The Bedrock model identifier.

    Returns:
        Details shaped as the catalog holds them.
    """
    return ModelDetails(
        id=model_id,
        name=model_id,
        provider="Amazon",
        input_modalities=["SPEECH", "TEXT"],
        output_modalities=["TEXT"],
        regions=[_REGION],  # type: ignore[list-item]
    )


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


def _transcript_events(*sentences: str) -> list[Any]:
    """Script the frames a transcription session receives.

    One ``contentStart``/``textOutput``/``contentEnd`` triple per utterance, as
    the model emits per voice-activity segment, then the assistant's turn, which
    is the signal that the transcript is complete.

    Args:
        sentences: Utterances the model transcribed.

    Returns:
        Output chunks in arrival order.
    """
    events: list[dict[str, Any]] = [{"completionStart": {"completionId": "c1"}}]
    for index, sentence in enumerate(sentences):
        content_id = f"u{index}"
        events += [
            {
                "contentStart": {
                    "contentId": content_id,
                    "type": "TEXT",
                    "role": "USER",
                    "additionalModelFields": '{"generationStage":"FINAL"}',
                }
            },
            {"textOutput": {"contentId": content_id, "content": sentence}},
            {"contentEnd": {"contentId": content_id, "stopReason": "PARTIAL_TURN"}},
        ]
    events.append(
        {
            "contentStart": {
                "contentId": "a0",
                "type": "TEXT",
                "role": "ASSISTANT",
                "additionalModelFields": '{"generationStage":"SPECULATIVE"}',
            }
        }
    )
    return [_output_chunk(event) for event in events]


def _translation_events(*, speculative: str, final: str) -> list[Any]:
    """Script the frames a translation session receives.

    The assistant previews its reply as one ``SPECULATIVE`` text block, speaks
    it, then restates what was spoken as a ``FINAL`` block. Only the preview is
    the translation; the turn ends with the audio block's ``END_TURN``.

    Args:
        speculative: The previewed reply.
        final: The restatement of what was spoken.

    Returns:
        Output chunks in arrival order.
    """
    events: list[dict[str, Any]] = [
        {
            "contentStart": {
                "contentId": "u0",
                "type": "TEXT",
                "role": "USER",
                "additionalModelFields": '{"generationStage":"FINAL"}',
            }
        },
        {"textOutput": {"contentId": "u0", "content": "source language"}},
        {"contentEnd": {"contentId": "u0", "stopReason": "PARTIAL_TURN"}},
        {
            "contentStart": {
                "contentId": "a0",
                "type": "TEXT",
                "role": "ASSISTANT",
                "additionalModelFields": '{"generationStage":"SPECULATIVE"}',
            }
        },
        {"textOutput": {"contentId": "a0", "content": speculative}},
        {"contentEnd": {"contentId": "a0", "stopReason": "PARTIAL_TURN"}},
        {"contentStart": {"contentId": "a1", "type": "AUDIO", "role": "ASSISTANT"}},
        {"audioOutput": {"contentId": "a1", "content": "ZmFrZQ=="}},
        {
            "contentStart": {
                "contentId": "a2",
                "type": "TEXT",
                "role": "ASSISTANT",
                "additionalModelFields": '{"generationStage":"FINAL"}',
            }
        },
        {"textOutput": {"contentId": "a2", "content": final}},
        {"contentEnd": {"contentId": "a2", "stopReason": "PARTIAL_TURN"}},
        {"contentEnd": {"contentId": "a1", "stopReason": "END_TURN", "type": "AUDIO"}},
    ]
    return [_output_chunk(event) for event in events]


def _usage_event(
    input_speech: int, input_text: int, output_speech: int, output_text: int
) -> Any:  # noqa: ANN401
    """Script one cumulative metering frame.

    Args:
        input_speech: Cumulative input speech tokens.
        input_text: Cumulative input text tokens.
        output_speech: Cumulative output speech tokens.
        output_text: Cumulative output text tokens.

    Returns:
        The output chunk carrying it.
    """
    return _output_chunk(
        {
            "usageEvent": {
                "details": {
                    "total": {
                        "input": {
                            "speechTokens": input_speech,
                            "textTokens": input_text,
                        },
                        "output": {
                            "speechTokens": output_speech,
                            "textTokens": output_text,
                        },
                    }
                },
                "totalTokens": input_speech + input_text + output_speech + output_text,
            }
        }
    )


class _BlockedReceiver:
    """Output half that opens and then never yields an event."""

    def __init__(self) -> None:
        """Start with nothing closed."""
        self.closed = 0

    def __aiter__(self) -> _BlockedReceiver:
        """Return self, as the SDK's receiver protocol does."""
        return self

    async def __anext__(self) -> Any:  # noqa: ANN401
        """Never answer, exactly as a session given bad audio does not."""
        await Future()
        raise StopAsyncIteration

    async def close(self) -> None:
        """Count one close."""
        self.closed += 1


class SilentStream(FakeDuplexStream):
    """A stream that opens and then sends nothing at all."""

    async def await_output(self) -> tuple[object, Any]:
        """Resolve the response with a receiver that never yields.

        Returns:
            The initial response and the blocked receiver.
        """
        self.output_stream = _BlockedReceiver()  # type: ignore[assignment]
        return object(), self.output_stream


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


class _FakeAudio:
    """Minimal ``InputFile`` stand-in for the offline session tests."""

    def __init__(self, media_type: str = "audio", file_format: str = "wav") -> None:
        """Report *media_type*/*file_format* and a fixed payload.

        Args:
            media_type: Media type the content sniffer reports.
            file_format: Format subtype the content sniffer reports.
        """
        self._content_type = (media_type, file_format)

    async def get_content_type_tuple(self) -> tuple[str, str]:
        """Return the reported (media type, format) pair."""
        return self._content_type

    async def to_bytes(self) -> bytes:
        """Return the upload payload."""
        return b"fake-audio"


class _Conditioning:
    """Stands in for the ffmpeg conditioning, recording how it was called."""

    def __init__(self) -> None:
        """Produce one second of silence until a test asks for more."""
        self.size = sonic.BYTES_PER_SECOND
        self.calls: list[dict[str, object]] = []

    async def encode(
        self, stream: AsyncGenerator[bytes], *_args: object, **kwargs: object
    ) -> AsyncGenerator[bytes]:
        """Consume the source and yield ``size`` bytes of silence.

        Args:
            stream: The upload the real encoder would read.
            _args: Positional encoder arguments (unused).
            kwargs: Keyword encoder arguments, recorded for assertions.

        Yields:
            Blocks of silence totalling ``size`` bytes.
        """
        await stream.aclose()
        self.calls.append(kwargs)
        remaining = self.size
        while remaining > 0:
            block = min(remaining, 1 << 16)
            remaining -= block
            yield bytes(block)


@pytest.fixture
def fake_pcm(monkeypatch: pytest.MonkeyPatch) -> _Conditioning:
    """Replace the ffmpeg conditioning with a fixed one second of silence."""
    conditioning = _Conditioning()
    monkeypatch.setattr(sonic, "encode_audio_stream", conditioning.encode)
    return conditioning


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


async def _transcribe(**kwargs: Any) -> Any:  # noqa: ANN401
    """Transcribe the fake upload with the Nova Sonic backend.

    Args:
        kwargs: Overrides for the ``stt`` arguments.

    Returns:
        Whatever ``stt`` returned.
    """
    arguments: dict[str, Any] = {
        "audio_content": _FakeAudio(),
        "response_format": "json",
        "logprobs": False,
    }
    arguments.update(kwargs)
    return await get_audio_model(_MODEL).stt(**arguments)


class TestModelResolution:
    """The model reaches this backend, and the catalog advertises it there.

    Ref: stdapi/models/__init__.py:_compute_model_capabilities
         stdapi/models/audio/__init__.py:get_audio_model
    """

    @pytest.mark.local
    def test_the_identifier_resolves_to_the_nova_sonic_backend(self) -> None:
        """Naming the model reaches this class rather than the Converse default.

        Ref: stdapi/models/audio/amazon_nova_sonic.py:AudioModel
        """
        assert isinstance(get_audio_model(_MODEL), sonic.AudioModel)

    @pytest.mark.local
    def test_catalog_advertises_transcription_and_translation(self) -> None:
        """The model advertises both audio routes from its own class capabilities.

        This runs against the *whole* model registry, where the chat families
        also match this identifier's prefix: a matcher that loses there leaves
        the model listed as a chat model and absent from both audio routes.

        Ref: stdapi/models/__init__.py:_compute_model_capabilities
        """
        routes, tools = _compute_model_capabilities(_MODEL, _model_details(_MODEL))

        assert "/v1/audio/transcriptions" in routes
        assert "/v1/audio/translations" in routes
        assert "openai_audio_transcription" in tools
        assert "openai_audio_translation" in tools

    @pytest.mark.local
    def test_the_legacy_generation_is_not_advertised_for_audio(self) -> None:
        """The superseded identifier claims no transcription capability.

        It speaks the same protocol but reaches end of life within weeks of this
        release, so it is not served here. What it must not do is keep being
        advertised as transcribing and then fail at request time.

        Ref: stdapi/models/__init__.py:NON_CONVERSE_SPEECH_MODEL_PREFIXES
        """
        routes, _tools = _compute_model_capabilities(
            _LEGACY_MODEL, _model_details(_LEGACY_MODEL)
        )

        assert "/v1/audio/transcriptions" not in routes
        assert "/v1/audio/translations" not in routes

    @pytest.mark.local
    async def test_the_legacy_generation_is_refused_cleanly(self) -> None:
        """Asking the superseded identifier to transcribe returns a clear error."""
        with pytest.raises(ApiError, match="not supported"):
            await get_audio_model(_LEGACY_MODEL).stt(
                _FakeAudio(),  # type: ignore[arg-type]
                "json",
                logprobs=False,
            )


class TestUnsupportedOutputs:
    """Formats this backend cannot produce are refused, not silently downgraded.

    The protocol carries no timestamp of any kind and never reports a detected
    language, so subtitles and verbose JSON would have to be fabricated.

    Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-output-events.html
         stdapi/models/audio/amazon_nova_sonic.py:AudioModel
    """

    @pytest.mark.local
    @pytest.mark.parametrize(
        "response_format", ["srt", "vtt", "verbose_json", "diarized_json"]
    )
    async def test_timestamped_formats_are_refused(self, response_format: str) -> None:
        """Each unsupported format names itself, the condition and the way forward."""
        with pytest.raises(ApiError) as failure:
            await _transcribe(response_format=response_format)

        message = str(failure.value)
        assert response_format in message
        assert "timestamp" in message.lower()
        assert "amazon.transcribe" in message

    @pytest.mark.local
    async def test_timestamp_granularities_are_refused(self) -> None:
        """Asking for word or segment timestamps is refused rather than ignored."""
        with pytest.raises(ApiError, match="timestamp"):
            await _transcribe(timestamp_granularities=["word"])

    @pytest.mark.local
    async def test_non_audio_upload_is_refused(self) -> None:
        """A file carrying no audio track is refused with the accepted formats."""
        with pytest.raises(ApiError, match="Unsupported audio format"):
            await _transcribe(audio_content=_FakeAudio("application", "pdf"))

    @pytest.mark.local
    async def test_undecodable_audio_is_refused_as_a_client_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_pcm: _Conditioning,
        streams: _StreamPool,
    ) -> None:
        """A file that decodes to nothing is a 400, never the encoder's own 500."""

        async def _fail(
            stream: AsyncGenerator[bytes], *_args: object, **_kwargs: object
        ) -> AsyncGenerator[bytes]:
            await stream.aclose()
            raise ApiError(_ENCODE_FAILED, status=500)
            yield b""

        monkeypatch.setattr(sonic, "encode_audio_stream", _fail)

        with pytest.raises(ApiError) as failure:
            await _transcribe()

        assert failure.value.status == 400
        assert "decodable audio track" in str(failure.value)
        assert not streams.opened

    @pytest.mark.local
    async def test_encoder_failures_that_are_not_the_upload_keep_their_status(
        self, monkeypatch: pytest.MonkeyPatch, fake_pcm: _Conditioning
    ) -> None:
        """A stalled or unavailable encoder is not reported as a bad upload."""

        async def _stall(
            stream: AsyncGenerator[bytes], *_args: object, **_kwargs: object
        ) -> AsyncGenerator[bytes]:
            await stream.aclose()
            raise ApiError(_ENCODE_STALLED, status=504)
            yield b""

        monkeypatch.setattr(sonic, "encode_audio_stream", _stall)

        with pytest.raises(ApiError) as failure:
            await _transcribe()

        assert failure.value.status == 504

    @pytest.mark.local
    @pytest.mark.parametrize("route", ["stt", "stt_translate"])
    async def test_text_format_returns_a_plain_body(
        self, fake_pcm: _Conditioning, streams: _StreamPool, route: str
    ) -> None:
        """``response_format=text`` returns the transcript as a plain-text body."""
        events = (
            _transcript_events("hello there")
            if route == "stt"
            else _translation_events(speculative="hello there", final="hello there")
        )
        streams.script(FakeDuplexStream(events=events))

        if route == "stt":
            response = await _transcribe(response_format="text")
        else:
            response = await get_audio_model(_MODEL).stt_translate(
                _FakeAudio(),  # type: ignore[arg-type]
                "text",
                None,
            )

        assert response.media_type == "text/plain; charset=utf-8"
        assert response.body == b"hello there"

    @pytest.mark.local
    async def test_translation_refuses_the_same_formats(self) -> None:
        """The translation route applies the same format policy."""
        with pytest.raises(ApiError, match=r"amazon\.transcribe"):
            await get_audio_model(_MODEL).stt_translate(
                _FakeAudio(),  # type: ignore[arg-type]
                "srt",
                None,
            )


class TestAudioCeiling:
    """Audio longer than one session can carry is refused before the session opens.

    Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-conversational-speech.html
         stdapi/models/audio/amazon_nova_sonic.py:AudioModel
    """

    @pytest.mark.local
    async def test_audio_above_the_ceiling_is_refused(
        self, fake_pcm: _Conditioning, streams: _StreamPool
    ) -> None:
        """The ceiling is enforced from the conditioned byte count, opening no session."""
        fake_pcm.size = (sonic.MAX_AUDIO_SECONDS + 1) * sonic.BYTES_PER_SECOND

        with pytest.raises(ApiError) as failure:
            await _transcribe()

        message = str(failure.value)
        assert "too long" in message
        assert "amazon.transcribe" in message
        assert not streams.opened, "No session may open for audio it cannot carry"

    @pytest.mark.local
    async def test_audio_at_the_ceiling_is_accepted(
        self, fake_pcm: _Conditioning, streams: _StreamPool
    ) -> None:
        """Audio exactly at the ceiling still runs, so the bound is inclusive."""
        fake_pcm.size = sonic.MAX_AUDIO_SECONDS * sonic.BYTES_PER_SECOND
        streams.script(FakeDuplexStream(events=_transcript_events("hello")))

        response = await _transcribe()

        assert response.text == "hello"


class TestSessionBounds:
    """A session that answers nothing is abandoned, and its stream is always closed.

    Every audio-configuration mistake on this protocol is silent: the service
    simply stops answering. Without a bound, one bad upload holds a request open
    until the service closes the connection about a minute later.

    Ref: stdapi/aws_bidi.py:open_bidi_stream
         stdapi/models/audio/amazon_nova_sonic.py:AudioModel
    """

    @pytest.mark.local
    async def test_a_silent_session_is_abandoned(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_pcm: _Conditioning,
        streams: _StreamPool,
    ) -> None:
        """No event within the receive bound ends the request instead of hanging."""
        monkeypatch.setattr(sonic, "_EVENT_TIMEOUT", 0.1)
        streams.script(SilentStream())

        with pytest.raises(ApiError) as failure:
            await _transcribe()

        assert failure.value.status == 504
        assert streams.opened[0].input_stream.closed == 1

    @pytest.mark.local
    async def test_the_stream_is_closed_exactly_once(
        self, fake_pcm: _Conditioning, streams: _StreamPool
    ) -> None:
        """Both halves of the stream are closed once when the transcript completes."""
        streams.script(FakeDuplexStream(events=_transcript_events("hello")))

        await _transcribe()

        stream = streams.opened[0]
        assert stream.input_stream.closed == 1
        assert stream.output_stream is not None
        assert stream.output_stream.closed == 1

    @pytest.mark.local
    async def test_utterances_are_joined_with_a_space(
        self, fake_pcm: _Conditioning, streams: _StreamPool
    ) -> None:
        """Per-utterance frames become one transcript, including across a pause.

        The model emits one frame per voice-activity segment, so a file with a
        silent gap yields several; the transcript is their concatenation.
        """
        streams.script(
            FakeDuplexStream(events=_transcript_events("first part.", "second part."))
        )

        response = await _transcribe()

        assert response.text == "first part. second part."

    @pytest.mark.local
    async def test_the_handshake_opens_with_the_system_block(
        self, fake_pcm: _Conditioning, streams: _StreamPool
    ) -> None:
        """The session is primed in the only order the service accepts.

        A session whose first content block is not the system one, or whose
        prompt declares no audio output, produces nothing at all -- silently.
        """
        streams.script(FakeDuplexStream(events=_transcript_events("hello")))

        await _transcribe()

        sent = _sent_events(streams.opened[0])
        names = [next(iter(payload["event"])) for payload in sent]
        assert names[:6] == [
            "sessionStart",
            "promptStart",
            "contentStart",
            "textInput",
            "contentEnd",
            "contentStart",
        ]
        assert names[-2:] == ["promptEnd", "sessionEnd"]
        prompt_start = sent[1]["event"]["promptStart"]
        assert "audioOutputConfiguration" in prompt_start
        assert sent[2]["event"]["contentStart"]["role"] == "SYSTEM"
        audio_start = sent[5]["event"]["contentStart"]
        assert audio_start["role"] == "USER"
        assert audio_start["audioInputConfiguration"] == {
            "mediaType": "audio/lpcm",
            "sampleRateHertz": sonic.SAMPLE_RATE,
            "sampleSizeBits": 16,
            "channelCount": 1,
            "audioType": "SPEECH",
        }

    @pytest.mark.local
    async def test_the_upload_is_conditioned_to_mono_16_khz(
        self, fake_pcm: _Conditioning, streams: _StreamPool
    ) -> None:
        """The upload is decoded to exactly the layout the session declares.

        The declared sample rate is accepted on trust: a session told 16 kHz and
        fed anything else still transcribes, but segments words wrongly and bills
        for a duration that never existed. A stereo upload is worse -- the
        session simply goes silent.
        """
        streams.script(FakeDuplexStream(events=_transcript_events("hello")))

        await _transcribe()

        assert fake_pcm.calls[0]["output_sample_rate"] == sonic.SAMPLE_RATE
        assert fake_pcm.calls[0]["output_channels"] == 1

    @pytest.mark.local
    async def test_silence_is_appended_to_the_upload(self) -> None:
        """The upload is followed by silence, then by the block's own end.

        Without the silence the model never treats the closing utterance as
        finished, and never transcribes it.

        Ref: stdapi/models/audio/amazon_nova_sonic.py:_send_audio
        """
        stream = FakeDuplexStream()
        session: BidiSession[Any, Any] = BidiSession(stream, _REGION)  # type: ignore[arg-type]
        names = sonic._SessionNames()  # noqa: SLF001
        pcm = bytes(sonic.FRAME_BYTES * 3)

        await sonic._send_audio(session, names, pcm)  # noqa: SLF001

        sent = [next(iter(payload["event"])) for payload in _sent_events(stream)]
        silence_frames = ceil(len(sonic.TRAILING_SILENCE) / sonic.FRAME_BYTES)
        assert sent == ["audioInput"] * (3 + silence_frames) + ["contentEnd"]


def _sent_events(stream: FakeDuplexStream) -> list[dict[str, Any]]:
    """Decode every input event a fake stream received.

    Args:
        stream: The fake stream to read.

    Returns:
        The decoded event payloads, in send order.
    """
    from json import loads  # noqa: PLC0415

    return [loads(event.value.bytes_) for event in stream.input_stream.sent]


class TestTranslation:
    """Translation reads the assistant's reply, not the source-language transcript.

    Ref: stdapi/models/audio/amazon_nova_sonic.py:AudioModel
    """

    @pytest.mark.local
    async def test_the_reply_is_returned_once(
        self, fake_pcm: _Conditioning, streams: _StreamPool
    ) -> None:
        """The previewed reply is returned, and its spoken restatement is not appended."""
        streams.script(
            FakeDuplexStream(
                events=_translation_events(
                    speculative="Translated text.", final="Translated text."
                )
            )
        )

        response = await get_audio_model(_MODEL).stt_translate(
            _FakeAudio(),  # type: ignore[arg-type]
            "json",
            None,
        )

        assert response.text == "Translated text."  # type: ignore[union-attr]


class TestUsageAccounting:
    """Metering comes from the last cumulative frame, split by token modality.

    Speech and text tokens are priced an order of magnitude apart, so recording
    them under one dimension without the breakdown under-reports the bill.

    Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-output-events.html
         stdapi/usage.py:record_bedrock_usage
    """

    @pytest.mark.local
    async def test_speech_and_text_tokens_are_recorded_apart(
        self, fake_pcm: _Conditioning, streams: _StreamPool
    ) -> None:
        """Speech tokens land under the speech spec, text tokens under the default one."""
        events = _transcript_events("hello")
        # Metering runs throughout, so the last frame before the model's turn
        # carries the session's totals; the earlier one must not be summed in.
        events.insert(0, _usage_event(10, 20, 0, 0))
        events.insert(-1, _usage_event(100, 200, 3, 7))
        streams.script(FakeDuplexStream(events=events))

        token = init_usage()
        try:
            await _transcribe()
            entries = usage_log_entries()
        finally:
            USAGE.reset(token)

        entry = next(item for item in entries if item["model"] == _MODEL)
        assert entry["input_tokens"] == 300
        assert entry["output_tokens"] == 10
        assert entry["input_tokens_by_spec"] == {"speech": 100}
        assert entry["output_tokens_by_spec"] == {"speech": 3}

    @pytest.mark.local
    async def test_usage_survives_a_session_that_never_completes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_pcm: _Conditioning,
        streams: _StreamPool,
    ) -> None:
        """Tokens already metered are recorded even when the session is abandoned."""
        monkeypatch.setattr(sonic, "_EVENT_TIMEOUT", 0.1)
        streams.script(
            FakeDuplexStream(
                events=[_usage_event(40, 60, 0, 0)], receive_error=_never_ends()
            )
        )

        token = init_usage()
        try:
            with pytest.raises(ApiError):
                await _transcribe()
            entries = usage_log_entries()
        finally:
            USAGE.reset(token)

        entry = next(item for item in entries if item["model"] == _MODEL)
        assert entry["input_tokens"] == 100
        assert entry["input_tokens_by_spec"] == {"speech": 40}

    @pytest.mark.local
    def test_speech_tokens_price_apart_from_text_tokens(self) -> None:
        """The price index answers separately for the speech and default buckets.

        Without the per-spec breakdown the whole input would resolve at the text
        rate, which is an order of magnitude cheaper than the speech one.

        Ref: stdapi/pricing.py:resolve_price
        """
        model = "novasonicspectest"
        set_test_price(model, _REGION, Dimension.INPUT_TOKENS, "0.00000033", "USD")
        set_test_price(
            model, _REGION, Dimension.INPUT_TOKENS, "0.000003", "USD", spec="speech"
        )

        text = resolve_price(Service.BEDROCK, model, _REGION, Dimension.INPUT_TOKENS)
        speech = resolve_price(
            Service.BEDROCK, model, _REGION, Dimension.INPUT_TOKENS, spec="speech"
        )

        assert text is not None
        assert speech is not None
        assert speech.amount > text.amount


def _never_ends() -> BaseException:
    """Return an error a receiver raises only after it has run dry.

    Returns:
        A ``TimeoutError``, which the bounded receive turns into a gateway
        timeout rather than letting the iteration end quietly.
    """
    return TimeoutError("the service stopped answering")


class TestStreamingContract:
    """The streamed shape rebuilds the non-streamed transcript, offline.

    Ref: stdapi/models/audio/amazon_nova_sonic.py:AudioModel
    """

    @pytest.mark.local
    async def test_deltas_concatenate_to_the_done_text(
        self, fake_pcm: _Conditioning, streams: _StreamPool
    ) -> None:
        """Deltas carry the separator, so joining them rebuilds the transcript."""
        streams.script(
            FakeDuplexStream(events=_transcript_events("first part.", "second part."))
        )

        events = [
            event
            async for event in get_audio_model(_MODEL).stt_stream(
                _FakeAudio(),  # type: ignore[arg-type]
                "json",
                logprobs=False,
            )
        ]

        done = events[-1]
        assert isinstance(done, TranscriptionTextDoneEvent)
        deltas = "".join(
            event.delta
            for event in events
            if not isinstance(event, TranscriptionTextDoneEvent)
        )
        assert deltas == done.text == "first part. second part."
        assert done.usage is not None


def _wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap mono 16-bit PCM in a WAV container.

    Args:
        pcm: Raw little-endian 16-bit mono samples.
        sample_rate: Sample rate of *pcm*, in hertz.

    Returns:
        The WAV file bytes.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm)
    return buffer.getvalue()


def _speak(client: OpenAI, model: str, text: str, voice: str) -> bytes:
    """Synthesize *text* through the gateway's own speech route.

    Args:
        client: Client pointed at the gateway.
        model: Text-to-speech model.
        text: What to say.
        voice: Voice identifier.

    Returns:
        WAV bytes.
    """
    return client.audio.speech.create(
        model=model, voice=voice, input=text, response_format="wav"
    ).read()


def _pcm_of(wav_bytes: bytes) -> tuple[bytes, int]:
    """Read a WAV file's samples and rate.

    Args:
        wav_bytes: The WAV file.

    Returns:
        Its frames and sample rate.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as source:
        return source.readframes(source.getnframes()), source.getframerate()


@pytest.fixture(scope="module")
def paused_audio(openai_client: OpenAI, speech_standard_model: str) -> bytes:
    """Two spoken sentences separated by three seconds of silence.

    A silent gap splits the model's own voice-activity detection into two
    segments, which is the case a naive end-of-speech terminator truncates.
    """
    first, rate = _pcm_of(
        _speak(openai_client, speech_standard_model, _PAUSE_FIRST, "joanna")
    )
    second, _ = _pcm_of(
        _speak(openai_client, speech_standard_model, _PAUSE_SECOND, "joanna")
    )
    return _wav(first + bytes(rate * 2 * _PAUSE_SECONDS) + second, rate)


@pytest.fixture(scope="module")
def french_audio(openai_client: OpenAI, speech_standard_model: str) -> bytes:
    """A French sentence, for the transcription-versus-translation pair."""
    return _speak(openai_client, speech_standard_model, _FRENCH_SENTENCE, "lea")


@pytest.mark.gateway(
    "no vendor equivalent: an alternative transcription backend selected by model id"
)
class TestLiveTranscription:
    """What the model returns for real audio, asserted tolerantly.

    Transcripts differ between the two model generations in punctuation and
    capitalisation, so only distinctive words are asserted -- never the text.

    Ref: https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
         stdapi/models/audio/amazon_nova_sonic.py:AudioModel
    """

    def test_transcription_returns_the_spoken_words(
        self, openai_client: OpenAI, sample_audio_file: bytes
    ) -> None:
        """A short upload transcribes, including its final word.

        The final word is the regression: without the silence this backend
        appends after the upload, the closing utterance is never finalised and
        never transcribed at all.
        """
        response = openai_client.audio.transcriptions.create(
            file=("audio.wav", io.BytesIO(sample_audio_file)), model=_MODEL
        )

        text = response.text.lower()
        assert "test" in text, f"Transcript is missing the final word: {text!r}"

    def test_a_silent_gap_does_not_truncate_the_transcript(
        self, openai_client: OpenAI, paused_audio: bytes
    ) -> None:
        """Speech on both sides of a three-second pause reaches the transcript."""
        response = openai_client.audio.transcriptions.create(
            file=("audio.wav", io.BytesIO(paused_audio)), model=_MODEL
        )

        text = response.text.lower()
        assert "harbour" in text or "harbor" in text, text
        assert "umbrella" in text, text

    def test_streamed_deltas_rebuild_the_transcript(
        self, openai_client: OpenAI, sample_audio_file: bytes
    ) -> None:
        """Concatenated deltas equal the terminal event's text, which ends the stream."""
        events = list(
            openai_client.audio.transcriptions.create(
                file=("audio.wav", io.BytesIO(sample_audio_file)),
                model=_MODEL,
                stream=True,
            )
        )

        assert events, "The stream produced no event"
        done = events[-1]
        assert done.type == "transcript.text.done"
        deltas = "".join(
            event.delta for event in events if event.type == "transcript.text.delta"
        )
        assert deltas == done.text
        assert "test" in done.text.lower()

    def test_the_transcript_keeps_the_spoken_language(
        self, openai_client: OpenAI, french_audio: bytes
    ) -> None:
        """Transcription returns the source language, unlike translation."""
        response = openai_client.audio.transcriptions.create(
            file=("audio.wav", io.BytesIO(french_audio)), model=_MODEL
        )

        text = response.text.lower()
        assert any(word in text for word in _FRENCH_WORDS), text

    @pytest.mark.parametrize("response_format", ["srt", "vtt", "verbose_json"])
    def test_timestamped_formats_are_refused_over_the_api(
        self, openai_client: OpenAI, sample_audio_file: bytes, response_format: str
    ) -> None:
        """The refusal reaches the client as a 400 naming the way forward."""
        with pytest.raises(BadRequestError) as failure:
            openai_client.audio.transcriptions.create(  # type: ignore[call-overload]
                file=("audio.wav", io.BytesIO(sample_audio_file)),
                model=_MODEL,
                response_format=response_format,
            )

        assert failure.value.status_code == 400
        assert "amazon.transcribe" in str(failure.value)


@pytest.mark.gateway(
    "no vendor equivalent: an alternative translation backend selected by model id"
)
class TestLiveTranslation:
    """The translation route reads the model's own English reply.

    Ref: https://developers.openai.com/api/reference/resources/audio/subresources/translations/methods/create
         stdapi/models/audio/amazon_nova_sonic.py:AudioModel
    """

    def test_french_audio_translates_to_english(
        self, openai_client: OpenAI, french_audio: bytes
    ) -> None:
        """French speech comes back as English text, not as its transcript."""
        response = openai_client.audio.translations.create(
            file=("audio.wav", io.BytesIO(french_audio)), model=_MODEL
        )

        text = response.text.lower()
        assert any(word in text for word in _ENGLISH_WORDS), text
