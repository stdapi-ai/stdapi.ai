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


def _as_wav(pcm: bytes) -> bytes:
    """Wrap the session's raw samples in a WAV container, for an upload.

    Args:
        pcm: 24 kHz mono 16-bit samples.

    Returns:
        The same samples as a WAV file.
    """
    container = io.BytesIO()
    with wave.open(container, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(_PCM24_RATE)
        wav.writeframes(pcm)
    return container.getvalue()


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
    ) -> None:
        """One spoken turn produces audio deltas, transcript deltas and usage.

        Turn detection is off, so the turn boundary is the client's own
        ``commit`` -- the sequence a caller controls, rather than one that
        depends on the backend's voice activity detection firing.

        Both response events describe the same answer whole: a client that
        validates the frame refuses one missing a field its model requires.

        Ref: https://developers.openai.com/api/reference/resources/realtime/server-events
             openai.types.realtime.realtime_response.RealtimeResponse
        """
        async with async_openai_client.realtime.connect(
            model=realtime_model
        ) as connection:
            await connection.recv()
            await connection.session.update(session=_MANUAL_TURN_SESSION)
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

        audio = b"".join(
            base64.b64decode(event.delta)
            for event in events
            if event.type == "response.output_audio.delta"
        )
        assert audio, "the answer carried no audio"

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

        Ref: https://developers.openai.com/api/reference/resources/realtime/client-events
             stdapi/realtime.py:RealtimeSession._truncate_item
        """
        async with async_openai_client.realtime.connect(
            model=realtime_model
        ) as connection:
            await connection.recv()
            await connection.session.update(session=_MANUAL_TURN_SESSION)
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
                # Answered out of band while the answer is still streaming, so
                # the acknowledgement also shows the session is still reading.
                await connection.input_audio_buffer.clear()
                resumed = await _drain_until(connection, "input_audio_buffer.cleared")

        kinds = _types(interrupted)
        assert interrupted[-1].type == "conversation.item.truncated", kinds
        assert interrupted[-1].item_id == item_id, interrupted[-1]
        assert interrupted[-1].content_index == 0, interrupted[-1]
        assert interrupted[-1].audio_end_ms == heard_ms, interrupted[-1]
        assert resumed[-1].type == "input_audio_buffer.cleared", _types(resumed)

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
