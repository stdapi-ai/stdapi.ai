"""wyoming-openai bridging Home Assistant's voice protocol onto the audio routes.

The lane's second service-shaped client, and the only one that consumes
``/v1/audio/speech`` the way a real-time voice assistant does:

- it asks for ``response_format="wav"`` and reads the body **as it arrives**,
  buffering only until the RIFF header parses, then forwarding raw PCM frames --
  so the sample rate, sample width and channel count it announces to its own
  client are read out of the gateway's stream rather than assumed;
- it **pipelines** synthesis. A streaming request is split into sentences and up
  to three ``/v1/audio/speech`` calls are in flight at once, while the audio is
  replayed in the original order.

Every other client here reads that route's body in one piece, one call at a time,
so a gateway that only ever produced a complete, correct WAV *after* the last
Polly frame -- or that serialised concurrent synthesis behind a shared resource --
would pass everywhere else and fail here.

The container is a proxy, not an agent: it speaks the Wyoming protocol on a TCP
port and OpenAI HTTP to the gateway. The test side is therefore a plain asyncio
Wyoming client rather than a registered CLI, which is also why the lane's autouse
model-identity check has no tool to attribute requests to and every test pins its
models on the server log itself.

Requires ``--agentic``, podman, and Bedrock credentials.

Ref: https://github.com/roryeckel/wyoming_openai
     https://github.com/OHF-Voice/wyoming#event-types
     stdapi/routes/openai_audio_speech.py:create_speech
     stdapi/routes/openai_audio_transcriptions.py:create_transcription
     tests/agentic/_podman.py:start_service_container
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import monotonic
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeVoice,
)

from ._podman import start_service_container, stop_service_container
from ._server import find_free_port

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from wyoming.event import Event

    from ._podman import ServiceContainer
    from ._server import AgenticServer

pytestmark = [pytest.mark.agentic, pytest.mark.asyncio(loop_scope="module")]

#: Image driven here. ``:latest`` moves with upstream, as the CLIs' ``@latest`` does.
_IMAGE = "ghcr.io/roryeckel/wyoming_openai:latest"

#: Seconds allowed for the container to publish its Wyoming port.
_STARTUP_TIMEOUT = 180

#: Seconds allowed for the proxy to answer a Wyoming handshake after launch.
_READY_TIMEOUT = 180.0

#: Seconds between two handshake attempts while the proxy is still starting.
_READY_POLL_INTERVAL = 1.0

#: Seconds spent waiting for one Wyoming event; a Bedrock call runs behind each.
_EVENT_TIMEOUT = 300.0

#: Speech-to-text model, resolved by the gateway's Transcribe integration.
_STT_MODEL = "amazon.transcribe"

#: Text-to-speech model, resolved by the gateway's Polly integration.
_TTS_MODEL = "amazon.polly-neural"

#: Voice requested for every synthesis, in OpenAI's own vocabulary; the gateway
#: maps it onto a Polly voice of the matching gender and language.
_TTS_VOICE = "alloy"

#: Backend the proxy is pinned to.
#:
#: Left unset it probes ``/readyz``, ``/health`` and ``/test`` to recognise
#: Speaches, LocalAI or Kokoro before falling back to OpenAI. The gateway answers
#: none of them the way those backends do, so the fallback is already correct --
#: pinning it just keeps the boot free of three requests no test asserts on.
_BACKEND = "OPENAI"

#: Sample rate the gateway's WAV bodies carry, in Hz.
#:
#: Polly is asked for 16 kHz PCM and ffmpeg wraps it without resampling
#: (``stdapi/models/audio/amazon_polly.py:_POLLY_DEFAULT_PCM_SAMPLE_RATE``).
_GATEWAY_SAMPLE_RATE = 16000

#: Rate wyoming-openai announces when it cannot parse a WAV header
#: (``handler.py:TTS_AUDIO_RATE``). Reading it back proves the header was *not*
#: parsed, so no assertion here may accept it.
_UNPARSED_SAMPLE_RATE = 24000

#: Bytes per sample of the PCM the gateway produces.
_SAMPLE_WIDTH = 2

#: Channels the gateway produces.
_CHANNELS = 1

#: First bytes of a RIFF/WAVE container.
_RIFF_MAGIC = b"RIFF"

#: Sentence synthesised then transcribed back, chosen for phonetic breadth.
_SPOKEN_SENTENCE = "The quick brown fox jumps over the lazy dog."

#: Words the transcript of that sentence must carry.
_SPOKEN_WORDS = ("quick", "brown", "fox", "lazy", "dog")

#: Sentences sent as one streaming synthesis request.
#:
#: The proxy hands every complete sentence to a synthesis task at once and only
#: the trailing, possibly unfinished one waits for the stop event, so four
#: sentences produce three concurrent calls plus a final one.
_STREAMED_SENTENCES = (
    "The harbour beacon flashes twice every minute.",
    "A grey heron waits beside the lower sluice gate.",
    "Two boats returned before the evening tide.",
    "The keeper logs the weather at eight o'clock.",
)

#: Synthesis calls the proxy runs at once, from ``handler.py:TTS_CONCURRENT_REQUESTS``.
_CONCURRENT_REQUESTS = 3

#: Bytes of PCM per audio chunk sent back for transcription, a fifth of a second
#: at the gateway's rate, so the upload is streamed rather than handed over whole.
_AUDIO_CHUNK_BYTES = 6400

#: Shortest plausible duration, in seconds, for either utterance under test.
_MINIMUM_SECONDS = 1.0


@dataclass(frozen=True)
class _Speech:
    """One synthesised utterance, with the audio format the proxy announced.

    Attributes:
        rate: Sample rate in hertz, from the ``audio-start`` event.
        width: Bytes per sample, from the same event.
        channels: Channel count, from the same event.
        chunks: Raw PCM payloads, in the order they arrived.
        stop_timestamp: Milliseconds the proxy reported for the whole utterance.
        log_start: Index into the gateway's log taken before the synthesis.
    """

    rate: int
    width: int
    channels: int
    chunks: tuple[bytes, ...]
    stop_timestamp: int | None
    log_start: int

    @property
    def audio(self) -> bytes:
        """The utterance's PCM frames, reassembled."""
        return b"".join(self.chunks)

    @property
    def seconds(self) -> float:
        """Duration the reassembled PCM represents, from its own frame count."""
        frame_size = self.width * self.channels
        if frame_size <= 0 or self.rate <= 0:
            return 0.0
        return len(self.audio) / frame_size / self.rate


def _gateway_url(server: AgenticServer) -> str:
    """Return the gateway's base URL as seen from inside the container.

    Args:
        server: Gateway under test.

    Returns:
        The loopback URL pasta forwards, or the external deployment's own URL
        when ``--server-url`` selected one.
    """
    if server.forward_port is None:
        return server.base_url
    return f"http://127.0.0.1:{server.forward_port}"


def _environment(server: AgenticServer, port: int) -> Mapping[str, str]:
    """Return the container environment configuring both halves of the proxy.

    The one model named in ``TTS_MODELS`` is also named in
    ``TTS_STREAMING_MODELS``, which puts its voice in the proxy's streaming TTS
    program: a plain ``synthesize`` event then still takes the incremental path,
    and a ``synthesize-start``/``-chunk``/``-stop`` exchange additionally takes
    the concurrent one, so both are reachable with a single Polly model.

    Args:
        server: Gateway the proxy is pointed at.
        port: Port the Wyoming server listens on, published on the same host port.

    Returns:
        The environment to start the container with.
    """
    base_url = f"{_gateway_url(server)}/v1"
    api_key = server.api_key
    return {
        # pasta forwards an inbound connection to the container's own address, so
        # the default bind on all interfaces is required rather than incidental.
        "WYOMING_URI": f"tcp://0.0.0.0:{port}",
        "WYOMING_LOG_LEVEL": "DEBUG",
        "HOME": "/work/home",
        "STT_OPENAI_URL": base_url,
        "STT_OPENAI_KEY": api_key,
        "STT_MODELS": _STT_MODEL,
        "STT_BACKEND": _BACKEND,
        "TTS_OPENAI_URL": base_url,
        "TTS_OPENAI_KEY": api_key,
        "TTS_MODELS": _TTS_MODEL,
        "TTS_STREAMING_MODELS": _TTS_MODEL,
        "TTS_VOICES": _TTS_VOICE,
        "TTS_BACKEND": _BACKEND,
    }


async def _await_ready(service: ServiceContainer) -> None:
    """Wait until the proxy answers a Wyoming handshake.

    The service container's own health poll cannot tell: Wyoming speaks
    length-prefixed JSON over a socket and answers no HTTP path, and the bare TCP
    probe left for that case is satisfied by pasta itself -- the published port
    accepts a connection from the host the moment the container is created, then
    resets it until the process inside has bound the port. Only an answered
    ``describe`` proves the proxy is serving.

    Args:
        service: Container started for this module.

    Raises:
        AssertionError: If the proxy never answers within the deadline.
    """
    deadline = monotonic() + _READY_TIMEOUT
    while monotonic() < deadline:
        with suppress(OSError, TimeoutError):
            async with AsyncTcpClient("127.0.0.1", service.port) as client:
                await client.write_event(Describe().event())
                if await asyncio.wait_for(client.read_event(), _READY_TIMEOUT):
                    return
        await asyncio.sleep(_READY_POLL_INTERVAL)
    pytest.fail(
        f"wyoming-openai did not answer on port {service.port} within "
        f"{_READY_TIMEOUT}s.\nLast output:\n{service.logs()[-3000:]}"
    )


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def wyoming_service(
    request: pytest.FixtureRequest,
    agentic_server: AgenticServer,
    tmp_path_factory: pytest.TempPathFactory,
) -> AsyncIterator[ServiceContainer]:
    """One wyoming-openai proxy, wired to the gateway for the whole module.

    Module-scoped because the proxy holds no per-test state: everything a test
    changes lives on its own Wyoming connection, and the container's own
    configuration is fixed at launch.

    The container is run as the owner of the working directory. Its image
    declares no ``USER``, and container root under ``--userns=keep-id`` is a
    subordinate host UID that cannot write into that directory.

    Yields:
        The running service, already answering the protocol.
    """
    workdir = tmp_path_factory.mktemp("wyoming-openai")
    port = find_free_port()
    container = start_service_container(
        image=_IMAGE,
        port=port,
        workdir=workdir,
        env=_environment(agentic_server, port),
        forward_port=agentic_server.forward_port,
        data_dirs=("home",),
        health_path=None,
        startup_timeout=_STARTUP_TIMEOUT,
        user=f"{workdir.stat().st_uid}:{workdir.stat().st_gid}",
        refresh=request.config.getoption("--agentic-rebuild"),
    )
    try:
        await _await_ready(container)
        yield container
    finally:
        stop_service_container(container)


@pytest_asyncio.fixture(loop_scope="module")
async def wyoming_client(
    wyoming_service: ServiceContainer,
) -> AsyncIterator[AsyncTcpClient]:
    """A Wyoming connection to the proxy, one per test.

    Per-test because the proxy keeps its recording buffer, synthesis buffer and
    sentence segmenters on the connection, so a shared one would let a test read
    a neighbour's leftovers.

    Yields:
        The connected client.
    """
    async with AsyncTcpClient("127.0.0.1", wyoming_service.port) as client:
        yield client


async def _next_event(client: AsyncTcpClient, service: ServiceContainer) -> Event:
    """Return the next event from the proxy, failing on a closed connection.

    Args:
        client: Connected Wyoming client.
        service: Service the client is talking to, for its log on failure.

    Returns:
        The event read.

    Raises:
        AssertionError: If the proxy closed the connection instead of answering.
    """
    event = await asyncio.wait_for(client.read_event(), timeout=_EVENT_TIMEOUT)
    if event is None:
        pytest.fail(
            "wyoming-openai closed the connection without answering.\n"
            f"Last output:\n{service.logs()[-3000:]}"
        )
    return event


async def _collect_audio(
    client: AsyncTcpClient,
    service: ServiceContainer,
    *,
    log_start: int,
    until_stopped: bool,
) -> _Speech:
    """Read one synthesised utterance off the connection.

    Args:
        client: Connected Wyoming client.
        service: Service the client is talking to, for its log on failure.
        log_start: Gateway log index taken before the request, carried through.
        until_stopped: Read on past ``audio-stop`` until ``synthesize-stopped``,
            which only a streaming synthesis sends.

    Returns:
        The utterance, with the format the proxy announced for it.
    """
    rate = width = channels = 0
    chunks: list[bytes] = []
    stop_timestamp: int | None = None
    while True:
        event = await _next_event(client, service)
        if AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            rate, width, channels = start.rate, start.width, start.channels
        elif AudioChunk.is_type(event.type):
            chunks.append(AudioChunk.from_event(event).audio)
        elif AudioStop.is_type(event.type):
            stop_timestamp = AudioStop.from_event(event).timestamp
            if not until_stopped:
                break
        elif SynthesizeStopped.is_type(event.type):
            break
    return _Speech(
        rate=rate,
        width=width,
        channels=channels,
        chunks=tuple(chunks),
        stop_timestamp=stop_timestamp,
        log_start=log_start,
    )


def _speech_requests(
    server: AgenticServer, log_start: int, path: str
) -> list[dict[str, object]]:
    """Return the gateway's request log entries for one route.

    Args:
        server: Gateway the proxy was pointed at.
        log_start: Log index captured before the exchange.
        path: Route path to select.

    Returns:
        The matching entries, oldest first; empty for an external server, whose
        log is not observable here.

    Ref: stdapi/monitoring.py:EventLog
    """
    return [
        entry
        for entry in server.log_entries(log_start)
        if entry.get("type") == "request" and entry.get("path") == path
    ]


def _assert_route(
    server: AgenticServer, log_start: int, path: str, model: str, count: int
) -> list[dict[str, object]]:
    """Assert a route was called *count* times, each on the model it was given.

    Nothing else pins the model here: wyoming-openai is not a registered CLI, so
    the lane's autouse identity check has no tool to attribute requests to, and a
    proxy quietly falling back to its own default would otherwise still pass.

    Args:
        server: Gateway the proxy was pointed at.
        log_start: Log index captured before the exchange.
        path: Route path that must have been called.
        model: Bedrock model every call to it must have resolved.
        count: Number of calls expected.

    Returns:
        The matching entries, so a caller can assert further on their timings.
    """
    if server.process is None:
        return []  # External server: its log is not observable here.
    entries = _speech_requests(server, log_start, path)
    resolved = {str(entry.get("model_id") or "") for entry in entries}
    assert len(entries) == count, f"{path} was called {len(entries)} times, not {count}"
    assert resolved == {model}, (
        f"{path} resolved {sorted(resolved)}, expected [{model!r}]"
    )
    return entries


def _overlapping(entries: Sequence[Mapping[str, object]]) -> int:
    """Return the largest number of logged requests that were in flight together.

    Each entry carries the instant the gateway started serving it and how long it
    took, which is the only window the server log exposes; counting how many of
    those windows contain a common instant is therefore a one-way proof -- an
    overlap cannot happen unless the client really did issue the calls in
    parallel, while a client that serialises them can never produce one.

    Args:
        entries: Request log entries for a single route.

    Returns:
        The peak number of simultaneously open requests.
    """
    spans = [
        (
            datetime.fromisoformat(str(entry["date"])),
            datetime.fromisoformat(str(entry["date"]))
            + timedelta(milliseconds=int(str(entry.get("execution_time_ms") or 0))),
        )
        for entry in entries
    ]
    return max(
        (sum(start <= instant < end for start, end in spans) for instant, _ in spans),
        default=0,
    )


class TestWyomingInfo:
    """The service description the proxy derives from its configuration.

    Ref: https://github.com/OHF-Voice/wyoming#event-types
         https://github.com/roryeckel/wyoming_openai
    """

    async def test_describe_advertises_the_configured_models(
        self, wyoming_client: AsyncTcpClient, wyoming_service: ServiceContainer
    ) -> None:
        """The proxy offers the gateway's STT model and a streaming TTS voice.

        This is the configuration the rest of the module depends on and the only
        exchange that costs nothing, so a misspelled model name or a voice that
        never reached the streaming program fails here rather than as an
        unexplained empty transcript three tests later.
        """
        await wyoming_client.write_event(Describe().event())
        event = await _next_event(wyoming_client, wyoming_service)
        assert Info.is_type(event.type), event.type
        info = Info.from_event(event)

        asr_models = [model.name for program in info.asr for model in program.models]
        assert asr_models == [_STT_MODEL], asr_models

        streaming_voices = [
            voice.name
            for program in info.tts
            if program.supports_synthesize_streaming
            for voice in program.voices
        ]
        assert streaming_voices == [_TTS_VOICE], streaming_voices


class TestWyomingSynthesis:
    """A single synthesis, read off ``/v1/audio/speech`` as it is produced.

    Ref: https://github.com/roryeckel/wyoming_openai
         stdapi/routes/openai_audio_speech.py:create_speech
         stdapi/models/audio/amazon_polly.py:AmazonPollyModel
    """

    @pytest_asyncio.fixture(loop_scope="module", scope="module")
    async def synthesized_speech(
        self, wyoming_service: ServiceContainer, agentic_server: AgenticServer
    ) -> _Speech:
        """Synthesise one sentence and keep it for the tests that read it.

        Module-scoped so the sentence is spoken once and both the format
        assertions and the transcription round trip read that one utterance,
        which is also what makes the round trip a round trip.

        Returns:
            The utterance, with the format the proxy announced.
        """
        log_start = len(agentic_server.logs)
        async with AsyncTcpClient("127.0.0.1", wyoming_service.port) as client:
            await client.write_event(
                Synthesize(
                    text=_SPOKEN_SENTENCE, voice=SynthesizeVoice(name=_TTS_VOICE)
                ).event()
            )
            return await _collect_audio(
                client, wyoming_service, log_start=log_start, until_stopped=False
            )

    def test_announced_format_comes_from_the_streamed_wav_header(
        self, synthesized_speech: _Speech, agentic_server: AgenticServer
    ) -> None:
        """The proxy reports the gateway's own rate, width and channel count.

        wyoming-openai buffers the response body only until ``wave`` can parse a
        RIFF header out of it, then strips that header and forwards the frames;
        when no header ever parses it falls back to a fixed 24 kHz. Reading back
        the gateway's 16 kHz therefore proves the streamed body carried a valid
        header early enough to be parsed mid-stream -- a body whose header only
        became correct once the last Polly frame had been encoded would arrive
        here as the fallback rate instead.

        Ref: stdapi/media.py:encode_audio_stream
             stdapi/types/openai_audio.py:SpeechCreateParams
        """
        assert synthesized_speech.rate != _UNPARSED_SAMPLE_RATE, (
            "the proxy fell back to its default rate, so no WAV header parsed "
            "out of the streamed body"
        )
        assert synthesized_speech.rate == _GATEWAY_SAMPLE_RATE
        assert synthesized_speech.width == _SAMPLE_WIDTH
        assert synthesized_speech.channels == _CHANNELS
        assert not synthesized_speech.chunks[0].startswith(_RIFF_MAGIC), (
            "the container header reached the audio payload, so it was never "
            "recognised and stripped"
        )
        assert synthesized_speech.seconds > _MINIMUM_SECONDS, (
            f"only {synthesized_speech.seconds:.2f}s of audio arrived"
        )
        _assert_route(
            agentic_server,
            synthesized_speech.log_start,
            "/v1/audio/speech",
            _TTS_MODEL,
            count=1,
        )

    async def test_synthesised_audio_transcribes_back_to_its_own_words(
        self,
        synthesized_speech: _Speech,
        wyoming_client: AsyncTcpClient,
        wyoming_service: ServiceContainer,
        agentic_server: AgenticServer,
    ) -> None:
        """The PCM the proxy emitted, sent back as chunks, says what it said.

        The frames are handed over exactly as they arrived -- the proxy rebuilds
        a WAV around them from the rate, width and channel count it announced --
        so a body that was truncated, mis-declared or forwarded from the wrong
        offset transcribes to nothing recognisable even though every event
        looked well formed. Neither half needs a sample file.

        Ref: stdapi/models/audio/amazon_transcribe.py:AmazonTranscribeModel
        """
        log_start = len(agentic_server.logs)
        audio = synthesized_speech.audio
        await wyoming_client.write_event(Transcribe(name=_STT_MODEL).event())
        await wyoming_client.write_event(
            AudioStart(
                rate=synthesized_speech.rate,
                width=synthesized_speech.width,
                channels=synthesized_speech.channels,
            ).event()
        )
        for offset in range(0, len(audio), _AUDIO_CHUNK_BYTES):
            await wyoming_client.write_event(
                AudioChunk(
                    audio=audio[offset : offset + _AUDIO_CHUNK_BYTES],
                    rate=synthesized_speech.rate,
                    width=synthesized_speech.width,
                    channels=synthesized_speech.channels,
                ).event()
            )
        await wyoming_client.write_event(AudioStop().event())

        while True:
            event = await _next_event(wyoming_client, wyoming_service)
            if Transcript.is_type(event.type):
                break
        text = Transcript.from_event(event).text.lower()
        missing = [word for word in _SPOKEN_WORDS if word not in text]
        assert not missing, f"transcript lost {missing}: {text!r}"
        _assert_route(
            agentic_server, log_start, "/v1/audio/transcriptions", _STT_MODEL, count=1
        )


class TestWyomingStreamingSynthesis:
    """A streaming synthesis, which pipelines several requests at once.

    Ref: https://github.com/OHF-Voice/wyoming#event-types
         stdapi/routes/openai_audio_speech.py:create_speech
    """

    async def test_sentences_are_synthesised_concurrently_and_replayed_in_order(
        self,
        wyoming_client: AsyncTcpClient,
        wyoming_service: ServiceContainer,
        agentic_server: AgenticServer,
    ) -> None:
        """Four sentences become four calls, three of them in flight together.

        The proxy hands every sentence it has already seen to its own synthesis
        task immediately and replays the results in order, so this is the only
        traffic in the lane where ``/v1/audio/speech`` is served several times
        over at once. The count proves the text was split rather than sent
        whole, and the overlap proves the calls were not merely queued -- a
        gateway serialising them behind one shared resource keeps every
        assertion above green and fails this one.

        Ref: stdapi/models/audio/amazon_polly.py:AmazonPollyModel
        """
        log_start = len(agentic_server.logs)
        await wyoming_client.write_event(
            SynthesizeStart(voice=SynthesizeVoice(name=_TTS_VOICE)).event()
        )
        await wyoming_client.write_event(
            SynthesizeChunk(text=" ".join(_STREAMED_SENTENCES)).event()
        )
        await wyoming_client.write_event(SynthesizeStop().event())
        speech = await _collect_audio(
            wyoming_client, wyoming_service, log_start=log_start, until_stopped=True
        )

        assert speech.rate == _GATEWAY_SAMPLE_RATE
        assert speech.seconds > _MINIMUM_SECONDS, (
            f"only {speech.seconds:.2f}s of audio arrived"
        )
        entries = _assert_route(
            agentic_server,
            log_start,
            "/v1/audio/speech",
            _TTS_MODEL,
            count=len(_STREAMED_SENTENCES),
        )
        if entries:
            assert _overlapping(entries) >= _CONCURRENT_REQUESTS, (
                "the gateway never served two synthesis requests at the same "
                "time, so the pipeline was serialised"
            )
