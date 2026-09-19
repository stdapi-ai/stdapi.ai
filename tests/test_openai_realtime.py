"""OpenAI Realtime API: ephemeral client secrets and the WebSocket session.

Driven by the official client's realtime interface, so the same bodies run
against the gateway and against OpenAI itself. The WebSocket target comes from
``async_openai_client``: the in-process ASGI transport cannot upgrade a
connection, so that fixture dials a real socket on every lane.

Ref: https://developers.openai.com/api/reference/resources/realtime
     stdapi/routes/openai_realtime.py:router
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import wave
from typing import TYPE_CHECKING, Any

import pytest
from openai import AsyncOpenAI
from websockets.exceptions import ConnectionClosed

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.resources.realtime.realtime import AsyncRealtimeConnection
    from openai.types.realtime.realtime_response import RealtimeResponse

#: Close code an ended-in-error session carries, measured against OpenAI.
_ERROR_CLOSE_CODE = 3000

#: Seconds one spoken turn may take end to end, model latency included.
_TURN_TIMEOUT = 120.0

#: Audio sent per ``input_audio_buffer.append`` (~100 ms at 24 kHz, 16-bit mono).
_APPEND_BYTES = 4800

#: Seconds a caller stays silent in the idle test, over any backend idle limit.
_IDLE_SECONDS = 65.0

#: Bytes of one millisecond of the session's speech (24 kHz, 16-bit, mono).
_PCM24_BYTES_PER_MS = 48

#: Sample rate of the session's audio, in hertz.
_PCM24_RATE = 24000

#: Sample rate both G.711 formats are defined at, in hertz.
_G711_RATE = 8000

#: ffmpeg codec name of each companded format, keyed by its media type.
_G711_CODECS = {"audio/pcmu": "mulaw", "audio/pcma": "alaw"}

#: Sentence a session is told to speak, so its answer has a known length.
_PINNED_SENTENCE = "The quick brown fox jumps over the lazy dog and then runs away."

#: Instructions pinning the answer, so both its words and its duration are known.
_PINNED_ANSWER_INSTRUCTIONS = (
    "Whatever the caller says, reply out loud with exactly this sentence, and "
    f"nothing else: {_PINNED_SENTENCE}"
)

#: Word the caller's recorded sample says, which its transcript must carry.
_SPOKEN_WORD = "test"

#: Share of a claimed transcript's words a real transcription must also carry.
_TRANSCRIPT_OVERLAP = 0.6

#: Words a transcript needs before its speaking rate means anything.
_MIN_TRANSCRIPT_WORDS = 5

#: Slowest and fastest credible speaking rate of an answer, in words per second.
_SPEECH_RATE_BAND = (1.5, 6.0)

#: Peak sample an answer must reach, against the 32767 of full scale.
_MIN_PEAK_AMPLITUDE = 3000

#: Seconds of silence appended to speech, for turn detection to end the turn on.
_TRAILING_SILENCE_SECONDS = 2.0

#: Seconds a stopped answer is listened to, to prove nothing more is spoken.
_SILENCE_AFTER_STOP = 1.0

#: Question the caller speaks, which cannot be answered without the tool.
_SPOKEN_QUESTION = "What is the weather in Paris?"

#: City the spoken question names, as the tool call must carry it.
_SPOKEN_LOCATION = "paris"

#: Temperature the tool answers with: no weather answer for Paris reaches it by chance.
_TOOL_TEMPERATURE_C = 47

#: Renderings of `_TOOL_TEMPERATURE_C` a transcript may use, once normalised.
_TEMPERATURE_SPELLINGS = ("47", "forty seven")

#: Runs of anything but a letter or a digit, collapsed before matching a transcript.
_NOT_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")

#: Milliseconds of the answer a caller hears before speaking over it.
_BARGE_IN_MS = 200

#: Session configuration whose turns the caller ends itself.
_MANUAL_TURN_SESSION: Any = {
    "type": "realtime",
    "instructions": "Reply with one short spoken sentence.",
    "audio": {
        "input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "turn_detection": None,
        },
        "output": {"format": {"type": "audio/pcm", "rate": 24000}},
    },
}

#: Session declaring one tool the model must call, with turns the caller ends.
_TOOL_SESSION: Any = {
    **_MANUAL_TURN_SESSION,
    "instructions": "Use the tools you are given, then answer in one sentence.",
    "tools": [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ],
    "tool_choice": "required",
}

#: Tool session whose spoken answer must state what the tool returned.
#:
#: ``tool_choice`` stays ``auto`` for the whole session because the answering
#: turn is part of it: upstream honours ``required`` per response, so a session
#: left on it calls the tool again instead of ever speaking. Changing it between
#: the two turns is not an option either -- that reopens the conversation, which
#: is where the tool's answer lives.
_SPOKEN_TOOL_SESSION: Any = {
    **_TOOL_SESSION,
    "instructions": (
        "You do not know the weather yourself. Call the tool you are given, then "
        "answer in one short sentence which states the temperature it returned, "
        "in degrees Celsius."
    ),
    "tool_choice": "auto",
}

#: Manual-turn session answering a known sentence, so its length can be measured.
_PINNED_ANSWER_SESSION: Any = {
    **_MANUAL_TURN_SESSION,
    "instructions": _PINNED_ANSWER_INSTRUCTIONS,
}

#: Session whose turns the backend's own voice activity detection ends.
_SERVER_VAD_SESSION: Any = {
    **_MANUAL_TURN_SESSION,
    "audio": {
        "input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "turn_detection": {"type": "server_vad"},
        },
        "output": {"format": {"type": "audio/pcm", "rate": 24000}},
    },
}

#: Transcription-only session, whose type is fixed by the secret it is opened with.
_TRANSCRIPTION_SESSION: Any = {
    "type": "transcription",
    "audio": {
        "input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": {},
            "turn_detection": None,
        }
    },
}

#: Fields a response object always carries, measured against upstream 2026-08-16.
_RESPONSE_FIELDS = (
    "status_details",
    "conversation_id",
    "output_modalities",
    "max_output_tokens",
    "audio",
    "metadata",
)

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def _drain_until(connection: AsyncRealtimeConnection, terminal: str) -> list[Any]:
    """Collect the connection's events up to and including *terminal*.

    Args:
        connection: The open realtime connection.
        terminal: ``type`` of the event that ends the collection.

    Returns:
        Every event received, in order.
    """
    events: list[Any] = []
    async for event in connection:
        events.append(event)
        if event.type in {terminal, "error"}:
            break
    return events


def _g711_session(media_type: str) -> Any:  # noqa: ANN401
    """Return a manual-turn session both listening and speaking in *media_type*.

    Args:
        media_type: ``audio/pcmu`` or ``audio/pcma``.

    Returns:
        The session configuration, reporting the caller's transcript too.
    """
    return {
        "type": "realtime",
        "instructions": _PINNED_ANSWER_INSTRUCTIONS,
        "audio": {
            "input": {
                "format": {"type": media_type},
                "transcription": {},
                "turn_detection": None,
            },
            "output": {"format": {"type": media_type}},
        },
    }


def _types(events: list[Any]) -> list[str]:
    """Return the ``type`` of each event, for readable assertion messages."""
    return [event.type for event in events]


def _assert_response_is_whole(response: RealtimeResponse) -> None:
    """Fail unless the response object carries every field upstream sends.

    An application validates the frame it was given, and a field its model
    declares without a default is required: Pipecat declares ``status_details``
    that way, so an omitted key kills its reader task on the first
    ``response.created`` and the session never speaks. The official models
    parse a partial object, so only the fields that were actually sent tell
    these two apart.

    Args:
        response: The ``response`` of a ``response.created`` or ``response.done``.
    """
    missing = [
        field for field in _RESPONSE_FIELDS if field not in response.model_fields_set
    ]
    assert not missing, f"the response object omitted {missing}: {response}"


async def _send_audio(connection: AsyncRealtimeConnection, pcm: bytes) -> None:
    """Append *pcm* to the input buffer in the chunks a caller would send.

    Args:
        connection: The open realtime connection.
        pcm: 24 kHz mono 16-bit samples of the caller's speech.
    """
    view = memoryview(pcm)
    for start in range(0, len(view), _APPEND_BYTES):
        await connection.input_audio_buffer.append(
            audio=base64.b64encode(view[start : start + _APPEND_BYTES]).decode()
        )


def _spoken_audio(events: list[Any]) -> bytes:
    """Return the speech an answer produced, from its audio deltas.

    Args:
        events: Events of one response, in order.

    Returns:
        24 kHz mono 16-bit samples, empty when the answer spoke nothing.
    """
    return b"".join(
        base64.b64decode(event.delta)
        for event in events
        if event.type == "response.output_audio.delta"
    )


def _claimed_transcript(events: list[Any]) -> str:
    """Return what the model says it said, from its transcript deltas.

    Args:
        events: Events of one response, in order.

    Returns:
        The concatenated transcript, empty when none was sent.
    """
    return "".join(
        event.delta
        for event in events
        if event.type == "response.output_audio_transcript.delta"
    )


def _as_wav(pcm: bytes, rate: int = _PCM24_RATE) -> bytes:
    """Wrap the session's raw samples in a WAV container, for an upload.

    Args:
        pcm: Mono 16-bit samples.
        rate: Sample rate they are defined at, in hertz.

    Returns:
        The same samples as a WAV file.
    """
    container = io.BytesIO()
    with wave.open(container, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return container.getvalue()


def _ffmpeg(arguments: list[str], data: bytes) -> bytes:
    """Convert *data* with one bounded ffmpeg run, as the audio fixtures do.

    Args:
        arguments: Everything between the program name and the output pipe.
        data: What to feed its standard input.

    Returns:
        What it wrote to its standard output.
    """
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, "ffmpeg is required to compand the Realtime audio"
    converted = subprocess.run(  # noqa: S603
        [ffmpeg, "-v", "quiet", *arguments, "pipe:1"],
        input=data,
        capture_output=True,
        check=False,
    )
    assert converted.returncode == 0, converted.stderr.decode(errors="replace")
    assert converted.stdout, "ffmpeg produced no output"
    return converted.stdout


def _companded(wav: bytes, media_type: str) -> bytes:
    """Compand a WAV recording into the G.711 codec *media_type* names.

    ffmpeg owns both G.711 tables here, so nothing of the gateway's own is
    involved in producing what the session is asked to understand.

    Args:
        wav: The recording, in any container ffmpeg reads.
        media_type: ``audio/pcmu`` or ``audio/pcma``.

    Returns:
        One companded byte per 8 kHz mono sample.
    """
    return _ffmpeg(
        [
            *("-i", "pipe:0"),
            *("-f", _G711_CODECS[media_type]),
            *("-ar", str(_G711_RATE)),
            *("-ac", "1"),
        ],
        wav,
    )


def _expanded(companded: bytes, media_type: str) -> bytes:
    """Expand companded bytes back to samples, again with ffmpeg's own tables.

    Args:
        companded: One companded byte per sample.
        media_type: ``audio/pcmu`` or ``audio/pcma``.

    Returns:
        8 kHz mono 16-bit little-endian samples.
    """
    return _ffmpeg(
        [
            *("-f", _G711_CODECS[media_type]),
            *("-ar", str(_G711_RATE)),
            *("-ac", "1"),
            *("-i", "pipe:0"),
            *("-f", "s16le"),
        ],
        companded,
    )


def _words(transcript: str) -> list[str]:
    """Return the lowercase words of a transcript, punctuation dropped.

    Args:
        transcript: Text to split.

    Returns:
        The words, in order.
    """
    return _NOT_ALPHANUMERIC.sub(" ", transcript.lower()).split()


def _peak_amplitude(pcm: bytes) -> int:
    """Return the loudest sample of 16-bit little-endian *pcm*.

    Args:
        pcm: Mono 16-bit little-endian samples.

    Returns:
        The largest absolute sample value, zero when there are none.
    """
    import struct  # noqa: PLC0415

    usable = len(pcm) // 2
    samples: tuple[int, ...] = struct.unpack(f"<{usable}h", pcm[: usable * 2])
    return max((abs(sample) for sample in samples), default=0)


def _shared_word_share(claimed: str, heard: str) -> float:
    """Return the share of *claimed*'s words that *heard* also carries.

    Args:
        claimed: What the speaker says it said.
        heard: What a recognizer made of the same audio.

    Returns:
        A share between 0 and 1, and 0 when nothing was claimed.
    """
    said = set(_words(claimed))
    if not said:
        return 0.0
    return len(said & set(_words(heard))) / len(said)


def _assert_duration_matches_transcript(
    transcript: str, seconds: float, what: str
) -> None:
    """Fail unless *seconds* of audio plausibly carries the words of *transcript*.

    This is what catches a mislabelled sample rate, which transcription alone
    survives: the samples are intact, so a recognizer reads them correctly, but
    a client plays them at the wrong speed. The two rates a session can confuse
    differ by a factor of three, so the band only has to be narrower than that.
    It is deliberately wide -- any speaking rate, a leading breath, a trailing
    pause and a recognizer's own word splitting all move the ratio, and none of
    them moves it anywhere near threefold.

    Args:
        transcript: What the audio was recognized as saying.
        seconds: How long the audio lasts at its declared rate.
        what: Name of the audio, for the failure message.

    Raises:
        AssertionError: Too few words to judge, or an implausible speaking rate.
    """
    words = len(_words(transcript))
    assert words >= _MIN_TRANSCRIPT_WORDS, (
        f"{what} says too little to time: {transcript!r}"
    )
    rate = words / seconds
    slowest, fastest = _SPEECH_RATE_BAND
    assert slowest <= rate <= fastest, (
        f"{what} carries {words} words in {seconds:.2f}s, which is {rate:.2f} "
        f"words per second and outside {slowest}-{fastest}: its declared sample "
        f"rate does not match the speech it holds. Transcript: {transcript!r}"
    )


def _states_the_temperature(transcript: str) -> bool:
    """Whether *transcript* states the temperature the tool returned.

    A recognizer renders a spoken number either way -- "47" or "forty-seven" --
    so every plausible spelling is accepted, on a transcript reduced to
    lowercase words separated by single spaces.

    Args:
        transcript: Text to search.

    Returns:
        True when one of the spellings appears as a whole word.
    """
    normalized = f" {_NOT_ALPHANUMERIC.sub(' ', transcript.lower()).strip()} "
    return any(f" {spelling} " in normalized for spelling in _TEMPERATURE_SPELLINGS)


async def _play_answer_until(
    connection: AsyncRealtimeConnection, milliseconds: int
) -> tuple[str, int]:
    """Read the answer's speech until *milliseconds* of it have arrived.

    What the caller has heard is what it has received, so the cut-off point is
    measured from the audio deltas rather than assumed: a session refuses a
    truncation past the audio it has actually produced.

    Args:
        connection: The open realtime connection, with an answer in flight.
        milliseconds: How much of the answer to let play.

    Returns:
        ``(item_id, played_ms)`` -- the answering item and the speech received
        from it, rounded down to the millisecond.

    Raises:
        AssertionError: The answer errored, or ended before playing that much.
    """
    played = 0
    async for event in connection:
        if event.type == "error":
            msg = f"the answer failed before it could be interrupted: {event.error}"
            raise AssertionError(msg)
        if event.type != "response.output_audio.delta":
            continue
        played += len(base64.b64decode(event.delta))
        if (played_ms := played // _PCM24_BYTES_PER_MS) >= milliseconds:
            return event.item_id, played_ms
    msg = f"the answer spoke less than {milliseconds} ms in total"
    raise AssertionError(msg)


async def _collect_until_closed(
    client: AsyncOpenAI, model: str, received: list[Any]
) -> None:
    """Read events into *received* until the server closes the connection.

    Args:
        client: The client to connect with.
        model: Model to open the session for.
        received: Accumulator the events are appended to.
    """
    async with client.realtime.connect(model=model) as connection:
        while True:
            received.append(await connection.recv())


class TestClientSecrets:
    """Ephemeral client secrets carry a session configuration and an expiry.

    Ref: https://developers.openai.com/api/reference/resources/realtime/subresources/client_secrets/methods/create
         stdapi/routes/openai_realtime.py:create_realtime_client_secret
    """

    def test_a_client_secret_carries_the_session_it_was_minted_for(
        self, openai_client: OpenAI, realtime_model: str
    ) -> None:
        """The response holds the secret, its expiry, and the effective session."""
        created = openai_client.realtime.client_secrets.create(
            session={
                "type": "realtime",
                "model": realtime_model,
                "instructions": "Answer in one short sentence.",
            }
        )

        assert created.value, "no client secret value was returned"
        assert created.expires_at > 0, "the client secret carries no expiry"
        assert created.session.type == "realtime"

    def test_a_session_takes_its_voice_as_a_custom_voice_object(
        self, openai_client: OpenAI, realtime_model: str, use_official_api: bool
    ) -> None:
        """``audio.output.voice`` accepts the ``{"id": ...}`` custom voice object.

        The OpenAI specification types the voice as a name or a custom voice
        object; the session the secret carries is echoed back with the name the
        object holds, which is what every session opened with that secret then
        asks the model for.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_realtime.py:AudioOutputConfig
        """
        if use_official_api:
            pytest.skip(
                "A custom voice object names a voice the account itself owns, "
                "and the account under test has none."
            )

        created = openai_client.realtime.client_secrets.create(
            session={
                "type": "realtime",
                "model": realtime_model,
                "audio": {"output": {"voice": {"id": "alloy"}}},
            }
        )

        assert created.session.type == "realtime"
        # The SDK types the audio configuration as optional on both session kinds.
        assert created.session.audio.output.voice == "alloy"  # type: ignore[union-attr]

    def test_a_transcription_session_secret_is_minted(
        self, openai_client: OpenAI
    ) -> None:
        """A transcription session configuration is accepted and echoed back."""
        created = openai_client.realtime.client_secrets.create(
            session={"type": "transcription"}
        )

        assert created.value, "no client secret value was returned"
        assert created.session.type == "transcription"


class TestRealtimeSession:
    """The WebSocket session: handshake, configuration, and one spoken turn.

    Ref: https://developers.openai.com/api/docs/guides/realtime
         stdapi/routes/openai_realtime.py:realtime_websocket
    """

    @pytest.mark.image
    async def test_the_connection_opens_with_a_session_created_event(
        self, async_openai_client: AsyncOpenAI, realtime_model: str
    ) -> None:
        """``session.created`` is the first event, and names the session.

        Carries the ``image`` marker because the upgrade is what a served image
        can break on its own: a server started with WebSockets disabled answers
        the handshake 404, with nothing in its log, and nothing run outside a
        container reproduces it.
        """
        async with async_openai_client.realtime.connect(
            model=realtime_model
        ) as connection:
            event = await connection.recv()

        assert event.type == "session.created", event
        # The SDK types the created session from the request types, which have no id.
        assert event.session.id, "the created session carries no id"  # type: ignore[union-attr]

    async def test_session_update_is_acknowledged_with_the_effective_session(
        self, async_openai_client: AsyncOpenAI, realtime_model: str
    ) -> None:
        """``session.update`` is answered by ``session.updated``, not by silence."""
        async with async_openai_client.realtime.connect(
            model=realtime_model
        ) as connection:
            await connection.recv()
            await connection.session.update(
                session={
                    "type": "realtime",
                    "instructions": "Answer in one short sentence.",
                }
            )
            events = await _drain_until(connection, "session.updated")

        assert events[-1].type == "session.updated", _types(events)
        assert events[-1].session.instructions == "Answer in one short sentence."

    async def test_session_update_takes_the_voice_as_a_custom_voice_object(
        self,
        async_openai_client: AsyncOpenAI,
        realtime_model: str,
        use_official_api: bool,
    ) -> None:
        """``session.update`` accepts the ``{"id": ...}`` custom voice object.

        The only path where the client's raw JSON is merged into the session in
        force before being validated, so the object has to survive a merge onto
        a voice the session already carries as a plain name or as null.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/realtime.py:_update_session
        """
        if use_official_api:
            pytest.skip(
                "A custom voice object names a voice the account itself owns, "
                "and the account under test has none."
            )

        async with async_openai_client.realtime.connect(
            model=realtime_model
        ) as connection:
            await connection.recv()
            await connection.session.update(
                session={
                    "type": "realtime",
                    "audio": {"output": {"voice": {"id": "alloy"}}},
                }
            )
            events = await _drain_until(connection, "session.updated")

        assert events[-1].type == "session.updated", _types(events)
        assert events[-1].session.audio.output.voice == "alloy"

    async def test_clearing_the_input_buffer_is_acknowledged(
        self, async_openai_client: AsyncOpenAI, realtime_model: str
    ) -> None:
        """``input_audio_buffer.clear`` answers ``input_audio_buffer.cleared``."""
        async with async_openai_client.realtime.connect(
            model=realtime_model
        ) as connection:
            await connection.recv()
            await connection.input_audio_buffer.clear()
            events = await _drain_until(connection, "input_audio_buffer.cleared")

        assert events[-1].type == "input_audio_buffer.cleared", _types(events)

    async def test_a_written_item_is_added_and_then_done(
        self, async_openai_client: AsyncOpenAI, realtime_model: str
    ) -> None:
        """A written item settles through the GA ``added``/``done`` pair.

        A voice framework advances its turn on ``conversation.item.done``; a
        session that only sends the superseded ``conversation.item.created``
        leaves it waiting forever.
        """
        async with async_openai_client.realtime.connect(
            model=realtime_model
        ) as connection:
            await connection.recv()
            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello there."}],
                }
            )
            events = await _drain_until(connection, "conversation.item.done")

        kinds = _types(events)
        assert "conversation.item.added" in kinds, kinds
        assert events[-1].type == "conversation.item.done", kinds
        assert events[-1].item.id, "the settled item carries no id"

    @pytest.mark.slow
    async def test_a_written_turn_is_answered_like_a_spoken_one(
        self, async_openai_client: AsyncOpenAI, realtime_model: str
    ) -> None:
        """A written item plus ``response.create`` answers, with speech.

        Sending text into a voice session is how an application nudges the model
        without a microphone, and it is the SDK's own written-turn API: a
        session that only answers audio leaves it waiting until the turn times
        out.

        Ref: https://developers.openai.com/api/reference/resources/realtime/client-events
             stdapi/models/realtime/amazon_nova_sonic.py:_NovaSonicSession.end_turn
        """
        async with async_openai_client.realtime.connect(
            model=realtime_model
        ) as connection:
            await connection.recv()
            await connection.session.update(session=_MANUAL_TURN_SESSION)
            await _drain_until(connection, "session.updated")
            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Say hello, briefly."}],
                }
            )
            await connection.response.create()

            async with asyncio.timeout(_TURN_TIMEOUT):
                events = await _drain_until(connection, "response.done")

        kinds = _types(events)
        assert "error" not in kinds, [
            event for event in events if event.type == "error"
        ]
        assert events[-1].type == "response.done", kinds
        assert events[-1].response.output, f"the written turn answered nothing: {kinds}"
        assert "response.output_audio.delta" in kinds, kinds
        _assert_response_is_whole(events[-1].response)

    @pytest.mark.slow
    async def test_a_session_survives_a_caller_who_says_nothing_for_a_while(
        self,
        async_openai_client: AsyncOpenAI,
        realtime_model: str,
        sample_audio_pcm24: bytes,
    ) -> None:
        """A quiet minute does not end the session: the next turn still answers.

        A caller who pauses between two questions sends nothing at all, and the
        session has to stay open across that pause.

        Ref: https://developers.openai.com/api/reference/resources/realtime/server-events
             stdapi/realtime.py:RealtimeSession._serve
        """
        async with async_openai_client.realtime.connect(
            model=realtime_model
        ) as connection:
            await connection.recv()
            await connection.session.update(session=_MANUAL_TURN_SESSION)
            await _drain_until(connection, "session.updated")

            await asyncio.sleep(_IDLE_SECONDS)

            await _send_audio(connection, sample_audio_pcm24)
            await connection.input_audio_buffer.commit()
            await connection.response.create()
            async with asyncio.timeout(_TURN_TIMEOUT):
                events = await _drain_until(connection, "response.done")

        kinds = _types(events)
        assert "error" not in kinds, [
            event for event in events if event.type == "error"
        ]
        assert events[-1].type == "response.done", kinds
        _assert_response_is_whole(events[-1].response)

    @pytest.mark.slow
    async def test_a_committed_turn_answers_with_audio_and_its_transcript(
        self,
        async_openai_client: AsyncOpenAI,
        realtime_model: str,
        sample_audio_pcm24: bytes,
        transcription_model: str,
    ) -> None:
        """One spoken turn produces audio deltas, transcript deltas and usage.

        Turn detection is off, so the turn boundary is the client's own
        ``commit`` -- the sequence a caller controls, rather than one that
        depends on the backend's voice activity detection firing.

        Both response events describe the same answer whole: a client that
        validates the frame refuses one missing a field its model requires.

        The audio itself is then inspected rather than counted: it has to be
        loud enough to be speech at all, to last as long as the words it is
        recognized as saying, and to say what the session claimed it said.
        Counting bytes passes on silence, on white noise and on the wrong
        sample rate; each of those fails one of the three. The answer is pinned
        to a known sentence so that timing it means something: a model free to
        reply "sure" produces too few words to time at all.

        Ref: https://developers.openai.com/api/reference/resources/realtime/server-events
             openai.types.realtime.realtime_response.RealtimeResponse
        """
        async with async_openai_client.realtime.connect(
            model=realtime_model
        ) as connection:
            await connection.recv()
            await connection.session.update(session=_PINNED_ANSWER_SESSION)
            await _drain_until(connection, "session.updated")

            await _send_audio(connection, sample_audio_pcm24)
            await connection.input_audio_buffer.commit()
            await connection.response.create()

            async with asyncio.timeout(_TURN_TIMEOUT):
                events = await _drain_until(connection, "response.done")

        kinds = _types(events)
        assert "error" not in kinds, [
            event for event in events if event.type == "error"
        ]
        assert "input_audio_buffer.committed" in kinds, kinds
        assert "response.output_audio.delta" in kinds, kinds
        assert "response.output_audio_transcript.delta" in kinds, kinds

        audio = _spoken_audio(events)
        assert audio, "the answer carried no audio"
        assert _peak_amplitude(audio) >= _MIN_PEAK_AMPLITUDE, (
            "the answer's audio never rises above a whisper: its peak sample is "
            f"{_peak_amplitude(audio)}"
        )

        heard = (
            await async_openai_client.audio.transcriptions.create(
                file=("answer.wav", io.BytesIO(_as_wav(audio))),
                model=transcription_model,
            )
        ).text
        claimed = _claimed_transcript(events)
        assert _shared_word_share(claimed, heard) >= _TRANSCRIPT_OVERLAP, (
            f"the answer's audio does not say what it claimed: said {claimed!r}, "
            f"heard {heard!r}"
        )
        _assert_duration_matches_transcript(
            heard, len(audio) / (_PCM24_BYTES_PER_MS * 1000), "the answer's audio"
        )

        done = events[-1]
        assert done.type == "response.done", kinds
        assert done.response.usage is not None, "response.done reported no usage"
        assert done.response.usage.output_tokens > 0, done.response.usage

        created = next(event for event in events if event.type == "response.created")
        _assert_response_is_whole(created.response)
        _assert_response_is_whole(done.response)
        assert created.response.status_details is None, created.response
        assert done.response.status_details is None, done.response
        assert done.response.conversation_id, done.response
        assert done.response.output_modalities == ["audio"], done.response
        assert done.response.max_output_tokens == "inf", done.response
        assert done.response.audio is not None, done.response
        assert done.response.audio.output is not None, done.response.audio
        assert done.response.audio.output.format is not None, done.response.audio
        assert done.response.audio.output.format.type == "audio/pcm", (
            done.response.audio
        )

    @pytest.mark.slow
    async def test_a_silent_pause_ends_the_turn_and_the_answer_starts_itself(
        self,
        async_openai_client: AsyncOpenAI,
        realtime_model: str,
        sample_audio_pcm24: bytes,
    ) -> None:
        """Default turn detection commits the turn itself and answers it.

        ``server_vad`` is the default of the API and what every voice framework
        leaves in place, yet every other spoken test here turns it off and ends
        its turns by hand. The caller commits nothing and asks for nothing: the
        backend notices the silence, and reports the turn it took with the same
        events a hand-ended one gets -- ``input_audio_buffer.committed``, then
        the item added and done. A client waiting on any of them against a
        detected turn would otherwise wait forever.

        Ref: https://developers.openai.com/api/reference/resources/realtime/server-events
             stdapi/realtime.py:RealtimeSession._report_speech_stopped
        """
        silence = bytes(int(_TRAILING_SILENCE_SECONDS * _PCM24_RATE) * 2)

        async with async_openai_client.realtime.connect(
            model=realtime_model
        ) as connection:
            await connection.recv()
            await connection.session.update(session=_SERVER_VAD_SESSION)
            configured = await _drain_until(connection, "session.updated")

            await _send_audio(connection, sample_audio_pcm24 + silence)

            async with asyncio.timeout(_TURN_TIMEOUT):
                events = await _drain_until(connection, "response.done")

        assert configured[-1].type == "session.updated", _types(configured)
        assert configured[-1].session.audio.input.turn_detection is not None, (
            configured[-1].session
        )
        assert configured[-1].session.audio.input.turn_detection.type == "server_vad", (
            configured[-1].session.audio.input
        )

        kinds = _types(events)
        assert "error" not in kinds, [
            event for event in events if event.type == "error"
        ]
        assert "input_audio_buffer.speech_started" in kinds, kinds
        assert "input_audio_buffer.speech_stopped" in kinds, kinds
        assert "input_audio_buffer.committed" in kinds, (
            f"the detected turn was never committed: {kinds}"
        )
        assert "conversation.item.added" in kinds, kinds
        assert kinds.index("input_audio_buffer.committed") < kinds.index(
            "conversation.item.added"
        ), f"the item was announced before the commit that takes the turn: {kinds}"
        stopped = events[kinds.index("input_audio_buffer.speech_stopped")]
        committed = events[kinds.index("input_audio_buffer.committed")]
        assert committed.item_id == stopped.item_id, (
            f"the commit names another turn than the one detected: "
            f"{committed.item_id} != {stopped.item_id}"
        )
        assert events[-1].type == "response.done", kinds
        assert events[-1].response.output, (
            f"the detected turn answered nothing: {kinds}"
        )
        assert _spoken_audio(events), f"the detected turn spoke nothing: {kinds}"
        _assert_response_is_whole(events[-1].response)

    @pytest.mark.slow
    async def test_a_caller_speaking_over_the_answer_truncates_it(
        self,
        async_openai_client: AsyncOpenAI,
        realtime_model: str,
        sample_audio_pcm24: bytes,
    ) -> None:
        """Truncating mid-answer is acknowledged, and the session keeps serving.

        Barge-in is the one path every voice framework sends: the caller speaks,
        the client stops playback and tells the session how much of the answer
        was actually heard, so that what the model is later told it said matches
        what the caller got. It arrives while the answer is still being spoken,
        against an item that has not settled -- and a session that answers it
        with an error, or stops responding afterwards, drops the call.

        Truncating synchronizes the record with what was played; ``response.cancel``
        is what stops the speech, and a barge-in client sends both. The answer
        must then really go quiet: the model keeps producing past the stop, and
        a session still forwarding it would speak over the caller who interrupted
        it. The stopped answer is therefore listened to for a further second.

        Ref: https://developers.openai.com/api/reference/resources/realtime/client-events
             stdapi/realtime.py:RealtimeSession._cancel_response
        """
        async with async_openai_client.realtime.connect(
            model=realtime_model
        ) as connection:
            await connection.recv()
            await connection.session.update(session=_PINNED_ANSWER_SESSION)
            await _drain_until(connection, "session.updated")

            await _send_audio(connection, sample_audio_pcm24)
            await connection.input_audio_buffer.commit()
            await connection.response.create()

            async with asyncio.timeout(_TURN_TIMEOUT):
                item_id, heard_ms = await _play_answer_until(connection, _BARGE_IN_MS)
                await connection.conversation.item.truncate(
                    item_id=item_id, content_index=0, audio_end_ms=heard_ms
                )
                interrupted = await _drain_until(
                    connection, "conversation.item.truncated"
                )
                # Nothing is left to stop once the answer has ended by itself,
                # and waiting for a second "response.done" would then hang.
                assert "response.done" not in _types(interrupted), (
                    "the answer ended before it could be interrupted: "
                    f"{_types(interrupted)}"
                )
                await connection.response.cancel()
                stopped = await _drain_until(connection, "response.done")
                await asyncio.sleep(_SILENCE_AFTER_STOP)
                # Answered out of band after a silent second, so the
                # acknowledgement also shows the session is still reading.
                await connection.input_audio_buffer.clear()
                resumed = await _drain_until(connection, "input_audio_buffer.cleared")

        kinds = _types(interrupted)
        assert interrupted[-1].type == "conversation.item.truncated", kinds
        assert interrupted[-1].item_id == item_id, interrupted[-1]
        assert interrupted[-1].content_index == 0, interrupted[-1]
        assert interrupted[-1].audio_end_ms == heard_ms, interrupted[-1]
        assert stopped[-1].type == "response.done", _types(stopped)
        assert stopped[-1].response.status == "cancelled", stopped[-1].response
        assert resumed[-1].type == "input_audio_buffer.cleared", _types(resumed)
        spoken_on = [
            event
            for event in resumed
            if event.type == "response.output_audio.delta" and event.item_id == item_id
        ]
        assert not spoken_on, (
            f"the interrupted answer kept speaking: {len(spoken_on)} further "
            f"audio deltas for {item_id}"
        )

    async def test_an_unauthenticated_connection_is_refused_before_any_audio(
        self, live_server: str | None, realtime_model: str
    ) -> None:
        """A bad credential ends the session on an ``error`` event, before audio.

        Measured against OpenAI: the upgrade itself succeeds, the first and only
        event is ``error`` with ``code='invalid_api_key'``, and the connection is
        then closed with close code 3000 carrying ``<type>.<code>`` as its
        reason. Nothing is ever sent on the session.
        """
        client = (
            AsyncOpenAI(api_key="sk-not-a-valid-key", max_retries=0)
            if live_server is None
            else AsyncOpenAI(
                base_url=f"{live_server}/v1", api_key="not-a-valid-key", max_retries=0
            )
        )

        received: list[Any] = []
        with pytest.raises(ConnectionClosed) as raised:
            await _collect_until_closed(client, realtime_model, received)

        assert [event.type for event in received] == ["error"], received
        error = received[0].error
        assert error.code == "invalid_api_key", error
        assert error.type == "invalid_request_error", error
        closed = raised.value.rcvd
        assert closed is not None, raised.value
        assert closed.code == _ERROR_CLOSE_CODE, closed
        assert closed.reason == "invalid_request_error.invalid_api_key", closed

    async def test_an_error_correlates_to_the_client_event_that_caused_it(
        self, async_openai_client: AsyncOpenAI, realtime_model: str
    ) -> None:
        """``error.event_id`` echoes the client's id, and never a stale one.

        ``error.event_id`` is the id of the *client* event that caused the
        error -- distinct from the envelope's own ``event_id``, which the
        server assigns to the error event itself. A client multiplexing
        several requests over one socket has no other way to match an error
        back to the request that caused it. Both bad events are sent over the
        same connection so the second one also proves the field is not left
        over from the first: an id belongs to the event that carried it, or
        to none.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             (RealtimeServerEventError)
             stdapi/realtime.py:RealtimeSession._error
        """
        async with async_openai_client.realtime.connect(
            model=realtime_model
        ) as connection:
            await connection.recv()
            await connection.send_raw(
                json.dumps({"type": "wat", "event_id": "evt-from-the-client"})
            )
            named = (await _drain_until(connection, "error"))[-1]
            await connection.send_raw(json.dumps({"type": "wat"}))
            anonymous = (await _drain_until(connection, "error"))[-1]

        assert named.type == "error", named
        assert named.error.event_id == "evt-from-the-client", named
        # The envelope id is the server's own, never borrowed from the client.
        assert named.event_id != "evt-from-the-client", named
        assert anonymous.error.event_id != "evt-from-the-client", anonymous
        assert anonymous.error.event_id is None, anonymous


class TestG711Turn:
    """A telephony session speaks and listens in G.711, at 8 kHz.

    The conversion tables themselves are proved numerically elsewhere; what is
    proved here is everything around them -- that the media type the client
    asked for selects the right pair, that 8 kHz is what the backend is told the
    audio is, and that what comes back is companded at that same rate.

    Ref: https://www.itu.int/rec/T-REC-G.711
         stdapi/realtime.py:_COMPANDED
    """

    @pytest.mark.slow
    @pytest.mark.gateway(
        "the official API answers this G.711 session update with an error, "
        "though its own RealtimeAudioFormats schema names both media types"
    )
    @pytest.mark.parametrize("media_type", ["audio/pcmu", "audio/pcma"])
    async def test_a_companded_turn_is_heard_and_answered_in_its_own_codec(
        self,
        async_openai_client: AsyncOpenAI,
        realtime_model: str,
        sample_audio_file: bytes,
        transcription_model: str,
        media_type: str,
    ) -> None:
        """A turn companded by ffmpeg is understood, and answered in the same codec.

        ffmpeg owns both ends of the companding: it encodes what the caller
        sends and expands what the session answers with. Using the gateway's own
        encoder for either would let a symmetrically wrong pair of tables cancel
        out and pass -- which is the exact shape of the A-law sign defect that
        shipped in this release.

        Each direction is then proved on its own. The caller's transcript has to
        carry the word the recording says, which only happens if what the
        session decoded is the speech ffmpeg companded. The answer is expanded,
        transcribed, and has to say what the session was told to say -- and to
        last as long as saying it takes, because a mislabelled rate is audible
        as a third of the speed and nothing else here would notice it.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             (RealtimeAudioFormats)
             stdapi/types/openai_realtime.py:FORMAT_SAMPLE_RATES
        """
        spoken = _companded(sample_audio_file, media_type)

        async with async_openai_client.realtime.connect(
            model=realtime_model
        ) as connection:
            await connection.recv()
            await connection.session.update(session=_g711_session(media_type))
            configured = await _drain_until(connection, "session.updated")

            await _send_audio(connection, spoken)
            await connection.input_audio_buffer.commit()
            await connection.response.create()

            async with asyncio.timeout(_TURN_TIMEOUT):
                events = await _drain_until(connection, "response.done")

        assert configured[-1].type == "session.updated", _types(configured)
        session = configured[-1].session
        assert session.audio.input.format.type == media_type, session.audio.input
        assert session.audio.output.format.type == media_type, session.audio.output

        kinds = _types(events)
        assert "error" not in kinds, [
            event for event in events if event.type == "error"
        ]
        assert events[-1].type == "response.done", kinds

        transcribed = next(
            (
                event
                for event in events
                if event.type == "conversation.item.input_audio_transcription.completed"
            ),
            None,
        )
        assert transcribed is not None, (
            f"the companded turn was never transcribed: {kinds}"
        )
        assert _SPOKEN_WORD in _words(transcribed.transcript), (
            "the session did not understand the companded speech it was sent: "
            f"{transcribed.transcript!r}"
        )

        answer = _spoken_audio(events)
        assert answer, f"the companded turn was answered without audio: {kinds}"
        expanded = _expanded(answer, media_type)
        assert len(expanded) == 2 * len(answer), (
            "the answer is not one companded byte per sample: "
            f"{len(answer)} bytes expanded to {len(expanded)}"
        )
        assert _peak_amplitude(expanded) >= _MIN_PEAK_AMPLITUDE, (
            "the expanded answer never rises above a whisper: its peak sample is "
            f"{_peak_amplitude(expanded)}"
        )

        heard = (
            await async_openai_client.audio.transcriptions.create(
                file=("answer.wav", io.BytesIO(_as_wav(expanded, _G711_RATE))),
                model=transcription_model,
            )
        ).text
        assert _shared_word_share(_PINNED_SENTENCE, heard) >= _TRANSCRIPT_OVERLAP, (
            f"the expanded answer does not say what it was told to: heard {heard!r}, "
            f"claimed {_claimed_transcript(events)!r}"
        )
        _assert_duration_matches_transcript(
            heard, len(answer) / _G711_RATE, f"the {media_type} answer"
        )

        done = events[-1]
        assert done.response.audio is not None, done.response
        assert done.response.audio.output is not None, done.response.audio
        assert done.response.audio.output.format is not None, done.response.audio
        assert done.response.audio.output.format.type == media_type, done.response.audio


class TestTranscriptionSession:
    """A transcription session reports the caller, and answers nothing.

    Ref: https://developers.openai.com/api/reference/resources/realtime/subresources/client_secrets/methods/create
         stdapi/types/openai_realtime.py:TranscriptionSessionConfig
    """

    @pytest.mark.slow
    @pytest.mark.gateway(
        "Upstream selects a transcription session with an 'intent' query "
        "parameter; this gateway takes the session kind from the secret instead."
    )
    async def test_a_transcription_session_reports_what_was_spoken_and_nothing_else(
        self,
        async_openai_client: AsyncOpenAI,
        realtime_model: str,
        sample_audio_pcm24: bytes,
    ) -> None:
        """The caller's words come back, and the session never speaks.

        The transcription session kind is fixed when the session is opened and
        cannot be updated into afterwards, so the secret the connection is
        authenticated with is what selects it. Everything downstream branches on
        that kind: the instructions the backend is opened with, the tools it is
        given, the content part an answer is written into, and whether any
        speech is generated at all. No test had ever opened one.

        Ref: https://developers.openai.com/api/reference/resources/realtime
             stdapi/realtime.py:RealtimeSession._speech_output
        """
        created = await async_openai_client.realtime.client_secrets.create(
            session=_TRANSCRIPTION_SESSION
        )
        assert created.session.type == "transcription", created.session

        holder = AsyncOpenAI(
            base_url=str(async_openai_client.base_url),
            api_key=created.value,
            max_retries=0,
        )
        async with holder.realtime.connect(model=realtime_model) as connection:
            opened = await connection.recv()
            await _send_audio(connection, sample_audio_pcm24)
            await connection.input_audio_buffer.commit()

            async with asyncio.timeout(_TURN_TIMEOUT):
                events = await _drain_until(
                    connection, "conversation.item.input_audio_transcription.completed"
                )

        assert opened.type == "session.created", opened
        assert opened.session.type == "transcription", opened.session

        kinds = _types(events)
        assert "error" not in kinds, [
            event for event in events if event.type == "error"
        ]
        assert events[-1].type == (
            "conversation.item.input_audio_transcription.completed"
        ), kinds
        assert _SPOKEN_WORD in _words(events[-1].transcript), (
            f"the transcription session misheard the caller: {events[-1].transcript!r}"
        )
        assert "response.output_audio.delta" not in kinds, (
            f"a transcription session spoke back: {kinds}"
        )


class TestFunctionTools:
    """The session calls a tool the client declared, and speaks its answer.

    Ref: https://developers.openai.com/api/docs/guides/realtime-function-calling
         stdapi/realtime.py:RealtimeSession._report_tool_call
    """

    @pytest.mark.slow
    async def test_a_required_tool_is_called_and_its_answer_is_used(
        self, async_openai_client: AsyncOpenAI, realtime_model: str
    ) -> None:
        """A written turn produces a function call, and its output is answered.

        ``tool_choice='required'`` is what makes this a test of the session
        rather than of the model's judgement: the call is asked for, so the
        assertion is on the gateway carrying it, not on the model choosing it.
        """
        async with async_openai_client.realtime.connect(
            model=realtime_model
        ) as connection:
            await connection.recv()
            await connection.session.update(session=_TOOL_SESSION)
            await _drain_until(connection, "session.updated")
            await connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is the weather in Paris?"}
                    ],
                }
            )
            await connection.response.create()

            async with asyncio.timeout(_TURN_TIMEOUT):
                called = await _drain_until(connection, "response.done")
                call = next(
                    event
                    for event in called
                    if event.type == "response.function_call_arguments.done"
                )
                await connection.conversation.item.create(
                    item={
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": '{"temperature_c": 14, "condition": "rain"}',
                    }
                )
                await connection.response.create()
                answered = await _drain_until(connection, "response.done")

        kinds = _types(called)
        assert "error" not in kinds, [
            event for event in called if event.type == "error"
        ]
        assert call.name == "get_weather", call
        assert "location" in call.arguments, call
        called_item = next(
            item for item in called[-1].response.output if item.type == "function_call"
        )
        assert called_item.call_id == call.call_id, called_item
        assert "error" not in _types(answered), [
            event for event in answered if event.type == "error"
        ]
        assert answered[-1].response.output, f"the tool answered nothing: {kinds}"
        _assert_response_is_whole(answered[-1].response)

    @pytest.mark.slow
    async def test_a_spoken_question_is_answered_with_what_the_tool_returned(
        self,
        async_openai_client: AsyncOpenAI,
        realtime_model: str,
        speech_standard_model: str,
        transcription_model: str,
    ) -> None:
        """The whole voice-agent loop, proved on the audio the caller receives.

        The caller speaks a question it cannot answer alone, the model calls the
        declared tool, the client answers it with a temperature no weather answer
        for Paris reaches by chance, and the answer's speech is then transcribed
        through the Transcriptions endpoint. What that transcript says is the
        only evidence that the tool's output travelled the whole way into the
        audio: ``response.output_audio_transcript`` is the model's own claim
        about what it said, and a session that spoke something else would carry
        it unchanged.

        The question is synthesized rather than canned, and the tool call is
        checked against the city it names, so the input path is load-bearing
        too -- audio the model could make nothing of would still reach the tool,
        but not with Paris in its arguments. Speech and transcription go through
        the same client as the session, since ``openai_client`` drives the app on
        another event loop than the one serving these routes.

        One tool is declared and the instructions leave the model nothing else to
        answer with, so what is asserted is the loop carrying a call, not a
        judgement between tools.

        Ref: https://developers.openai.com/api/docs/guides/realtime-function-calling
             stdapi/realtime.py:RealtimeSession._answer_tool_call
        """
        # The Speech endpoint's `pcm` is 24 kHz mono 16-bit little-endian, which
        # is exactly what `audio/pcm` means to a session: fed in unconverted.
        question = (
            await async_openai_client.audio.speech.create(
                model=speech_standard_model,
                voice="alloy",
                input=_SPOKEN_QUESTION,
                response_format="pcm",
            )
        ).content
        assert question, "the speech endpoint returned no samples for the question"

        async with async_openai_client.realtime.connect(
            model=realtime_model
        ) as connection:
            await connection.recv()
            await connection.session.update(session=_SPOKEN_TOOL_SESSION)
            await _drain_until(connection, "session.updated")

            await _send_audio(connection, question)
            await connection.input_audio_buffer.commit()
            await connection.response.create()
            async with asyncio.timeout(_TURN_TIMEOUT):
                called = await _drain_until(connection, "response.done")

            assert "error" not in _types(called), [
                event for event in called if event.type == "error"
            ]
            call = next(
                (
                    event
                    for event in called
                    if event.type == "response.function_call_arguments.done"
                ),
                None,
            )
            assert call is not None, (
                f"the spoken question called no tool: {_types(called)}"
            )
            assert call.name == "get_weather", call
            assert _SPOKEN_LOCATION in call.arguments.lower(), (
                f"the tool call does not name the city that was spoken: {call}"
            )

            await connection.conversation.item.create(
                item={
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": (
                        f'{{"temperature_c": {_TOOL_TEMPERATURE_C},'
                        ' "condition": "rain"}'
                    ),
                }
            )
            await connection.response.create()
            async with asyncio.timeout(_TURN_TIMEOUT):
                answered = await _drain_until(connection, "response.done")

        kinds = _types(answered)
        assert "error" not in kinds, [
            event for event in answered if event.type == "error"
        ]
        speech = _spoken_audio(answered)
        assert speech, f"the answer to the tool spoke nothing: {kinds}"

        transcribed = (
            await async_openai_client.audio.transcriptions.create(
                file=("answer.wav", io.BytesIO(_as_wav(speech))),
                model=transcription_model,
            )
        ).text
        assert _states_the_temperature(transcribed), (
            f"the spoken answer does not state what the tool returned: "
            f"{transcribed!r}, claimed {_claimed_transcript(answered)!r}"
        )
