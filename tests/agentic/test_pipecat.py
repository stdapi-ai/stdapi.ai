"""Pipecat driven against this gateway's Realtime WebSocket.

``docs/api_openai_realtime.md`` offers Pipecat as the second way to put WebRTC
or a phone line in front of this deployment, and this module is that recipe
executed: the service is constructed exactly as the page constructs it, wired
into a real ``Pipeline``, and asked to hold one spoken turn.

Pipecat terminates WebRTC on its own side, so the half this API serves is the
WebSocket between ``OpenAIRealtimeLLMService`` and the model. Unlike LiveKit's
plugin, this one takes the **WebSocket** URL whole -- ``/v1/realtime``
included -- and appends only the model query parameter, which is the difference
the documentation warns a reader about.

What a Pipecat application actually consumes is frames, not events: the answer
reaches a bot as ``TTSAudioRawFrame`` and ``LLMTextFrame``, produced by the
service's own parsing of the gateway's events. Asserting on those frames is
therefore the only way to prove the gateway's events were *usable*, rather than
merely sent -- the service parses every frame with its own pydantic models and
its reader task stops on the first one it refuses, which turns a single missing
field into a session that connects, accepts audio and answers nothing.

That failure mode is silent by design: the refusal is logged, not raised, and
no frame ever reaches the pipeline. The turn test therefore captures Pipecat's
own error log for the duration of the turn and asserts it stayed empty, so the
event the service could not read is named in the failure instead of a timeout.

Requires --agentic, podman, and Bedrock credentials.

Ref: https://docs.pipecat.ai/api-reference/server/services/s2s/openai
     https://developers.openai.com/api/docs/guides/realtime
     docs/api_openai_realtime.md
     stdapi/routes/openai_realtime.py:router
"""

from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest
from loguru import logger
from openai import OpenAI
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.workers.runner import WorkerRunner

from ._runner import ModelConfig
from ._tools import AgenticTool

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ._server import AgenticServer
    from ._tools import AgenticResult, Command, Invocation

pytestmark = pytest.mark.agentic


def _unused_build(invocation: Invocation) -> Command:
    """Never called: this module drives no CLI, only the shared identity check needs a tool."""
    raise NotImplementedError


def _unused_parse(stdout: str) -> AgenticResult:
    """Never called: see :func:`_unused_build`."""
    raise NotImplementedError


def _unused_prepare_workdir(invocation: Invocation) -> None:
    """Never called: see :func:`_unused_build`."""
    raise NotImplementedError


#: Registered so the autouse model-identity check has a tool to attribute
#: requests to; pipecat-ai is a Python library, never run in a container.
TOOL = AgenticTool(
    id="pipecat",
    npm_package=None,
    binary="python",
    route="/v1",
    metrics_prefix="PC-METRICS",
    build=_unused_build,
    parse=_unused_parse,
    prepare_workdir=_unused_prepare_workdir,
    # The service sends no per-run identifier the gateway logs, so its requests
    # can only be attributed positionally.
    attributes_sessions=False,
)

#: Speech-to-speech model every session in this module is opened for.
_REALTIME_MODEL_CONFIG = pytest.param(
    ModelConfig(model="amazon.nova-2-sonic-v1:0"), id="nova-2-sonic"
)

#: Model synthesizing the caller's speech; the cheapest voice on the gateway.
_SPEECH_MODEL = "amazon.polly-standard"

#: Instructions the voice bot answers under.
_INSTRUCTIONS = "Answer with one short spoken sentence."

#: Sentence the caller speaks, synthesized once per module.
_SPOKEN_QUESTION = "Hello there. Please say something back to me."

#: Audio carried per input frame, ~100 ms at 24 kHz 16-bit mono.
_FRAME_BYTES = 4800

#: Sample rate a realtime session takes its input audio at.
_SAMPLE_RATE = 24000

#: Seconds the service may take to open the session and have its update acked.
_READY_TIMEOUT = 60.0

#: Seconds one spoken turn may take end to end, model latency included.
_TURN_TIMEOUT = 120.0

#: Seconds the pipeline is given to shut down once the turn is over.
_SHUTDOWN_TIMEOUT = 30.0

#: Seconds a closed realtime session is given to finish tearing its backend down.
_TEARDOWN_SETTLE = 1.5

#: Deployment the documentation's snippet is written against.
_DOCUMENTED_BASE_URL = "wss://your-deployment.example.com/v1/realtime"


@pytest.fixture(scope="module")
def gateway_client(agentic_server: AgenticServer) -> OpenAI:
    """Synchronous OpenAI SDK client bound to the gateway under test."""
    return OpenAI(
        base_url=agentic_server.url("/v1"),
        api_key=agentic_server.api_key,
        max_retries=0,
    )


@pytest.fixture(scope="module")
def spoken_pcm(gateway_client: OpenAI) -> bytes:
    """The caller's question as 24 kHz mono 16-bit PCM, the Realtime input format.

    Synthesized through the gateway's own speech route so the lane needs no
    audio fixture on disk and no transcoder; ``pcm`` is defined at exactly the
    rate a realtime session accepts.
    """
    audio = gateway_client.audio.speech.create(
        model=_SPEECH_MODEL,
        voice="alloy",
        input=_SPOKEN_QUESTION,
        response_format="pcm",
    ).content
    assert audio, "the gateway synthesized no speech for the realtime caller"
    return audio


@pytest.fixture
def client_errors() -> Iterator[list[str]]:
    """Every error Pipecat logs while the test runs.

    The service reports an event it cannot parse by logging it and stopping its
    reader task, so without this the only symptom is a pipeline that produced no
    frames.

    Yields:
        The messages, appended as they are logged.
    """
    captured: list[str] = []
    sink = logger.add(lambda message: captured.append(str(message)), level="ERROR")
    try:
        yield captured
    finally:
        logger.remove(sink)


def _realtime_service(server: AgenticServer, model: str) -> OpenAIRealtimeLLMService:
    """Build the service exactly as ``docs/api_openai_realtime.md`` builds it.

    Server-side turn detection is switched off so the caller's own stop-speaking
    frame ends the turn: the alternative depends on the backend's voice activity
    detector firing, which is not the gateway behaviour under test.

    Args:
        server: Gateway serving the session.
        model: Realtime model the session is opened for.

    Returns:
        The service, not yet connected.
    """
    return OpenAIRealtimeLLMService(
        base_url=server.url("/v1/realtime").replace("http://", "ws://", 1),
        api_key=server.api_key,
        settings=OpenAIRealtimeLLMService.Settings(
            model=model,
            system_instruction=_INSTRUCTIONS,
            session_properties=SessionProperties(
                audio=AudioConfiguration(input=AudioInput(turn_detection=False))
            ),
        ),
    )


@dataclass(slots=True)
class _Turn:
    """Everything one spoken exchange delivered to the bot behind the service.

    Attributes:
        audio: Bytes of the answer's speech, as the service decoded them.
        text: The answer's transcript, as the service assembled it.
        kinds: Every frame type the bot saw, in arrival order.
    """

    audio: bytearray = field(default_factory=bytearray)
    text: str = ""
    kinds: list[str] = field(default_factory=list)


class _Bot(FrameProcessor):
    """The application end of the pipeline, recording what reached it.

    A real bot plays the audio out and shows the text; this one keeps both, so
    the assertions are made on what a Pipecat application would actually have.
    """

    def __init__(self, turn: _Turn) -> None:
        super().__init__()
        self._turn = turn
        self.answered = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Record *frame* and pass it on.

        Args:
            frame: The frame reaching the bot.
            direction: Direction the frame travels in.
        """
        await super().process_frame(frame, direction)
        self._turn.kinds.append(type(frame).__name__)
        if isinstance(frame, TTSAudioRawFrame):
            self._turn.audio.extend(frame.audio)
        elif isinstance(frame, LLMTextFrame):
            self._turn.text += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            self.answered.set()
        await self.push_frame(frame, direction)


async def _await_session(service: OpenAIRealtimeLLMService) -> None:
    """Wait until the service has opened the session and had its update acked.

    Audio pushed before that would be appended to a session still being
    configured, so the wait is what makes the caller's first words part of the
    turn rather than a race. The service exposes no event for it, hence the
    poll.

    Giving up is not raised: a session that never opened leaves the assertions
    with an empty turn and whatever the client logged, which says more than a
    timeout raised from here.

    Args:
        service: The service under test.
    """
    with suppress(TimeoutError):
        async with asyncio.timeout(_READY_TIMEOUT):
            # ASYNC110: there is no event to await -- the service sets this flag
            # when its session.update is acknowledged, with no handler for it.
            while not service._api_session_ready:  # noqa: SLF001, ASYNC110
                await asyncio.sleep(0.1)


async def _hold_a_turn(service: OpenAIRealtimeLLMService, pcm: bytes) -> _Turn:
    """Run one spoken turn through a real Pipecat pipeline.

    Args:
        service: The service under test.
        pcm: 24 kHz mono 16-bit samples of the caller's speech.

    Returns:
        What the bot behind the service received.
    """
    turn = _Turn()
    bot = _Bot(turn)
    worker = PipelineWorker(Pipeline([service, bot]), idle_timeout_secs=None)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    running = asyncio.ensure_future(runner.run())
    try:
        await _await_session(service)
        view = memoryview(pcm)
        await worker.queue_frames(
            [
                UserStartedSpeakingFrame(),
                *(
                    InputAudioRawFrame(
                        audio=bytes(view[start : start + _FRAME_BYTES]),
                        sample_rate=_SAMPLE_RATE,
                        num_channels=1,
                    )
                    for start in range(0, len(view), _FRAME_BYTES)
                ),
                UserStoppedSpeakingFrame(),
            ]
        )
        # A turn that never completes is reported by the assertions on what the
        # bot received -- and, before those, by whatever the client logged. The
        # timeout itself explains nothing, so it never leaves this function.
        with suppress(TimeoutError):
            async with asyncio.timeout(_TURN_TIMEOUT):
                await bot.answered.wait()
    finally:
        await worker.stop_when_done()  # type: ignore[no-untyped-call]
        try:
            async with asyncio.timeout(_SHUTDOWN_TIMEOUT):
                await running
        except TimeoutError:
            running.cancel()
        await asyncio.sleep(_TEARDOWN_SETTLE)
    return turn


@pytest.mark.parametrize("model_config", [_REALTIME_MODEL_CONFIG])
class TestPipecatRealtimeSession:
    """A Pipecat voice bot holding a spoken turn with this gateway.

    Ref: https://docs.pipecat.ai/api-reference/server/services/s2s/openai
         docs/api_openai_realtime.md
         stdapi/routes/openai_realtime.py:openai_realtime
    """

    async def test_a_spoken_turn_answers_with_audio_and_a_transcript(
        self,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        spoken_pcm: bytes,
        client_errors: list[str],
    ) -> None:
        """The caller's speech reaches the bot as playable audio and readable text.

        Both are asserted because the service builds them from different events:
        the audio from ``response.output_audio.delta`` and the text from
        ``response.output_audio_transcript.delta``, so a gateway that stopped
        sending either would still look like it answered.

        Ref: https://developers.openai.com/api/reference/resources/realtime
             stdapi/realtime.py:RealtimeSession
        """
        service = _realtime_service(agentic_server, model_config.model)

        turn = await _hold_a_turn(service, spoken_pcm)

        assert not client_errors, (
            "Pipecat could not read the session; it logged:\n"
            + "\n".join(client_errors)
        )
        assert turn.audio, f"the spoken turn returned no audio: {Counter(turn.kinds)}"
        assert turn.text.strip(), (
            f"the spoken turn returned no transcript: {Counter(turn.kinds)}"
        )

    async def test_the_documented_snippet_dials_this_deployment_websocket(
        self, model_config: ModelConfig, agentic_server: AgenticServer
    ) -> None:
        """The documented ``base_url`` is dialled whole, with the model appended.

        The recipe hands the service the full WebSocket path rather than an HTTP
        base URL, which is the one thing that differs from the LiveKit recipe on
        the same page. Pinning it here means a release that started deriving the
        path -- as the other plugin does -- is reported as a wrong URL rather
        than as a connection failure in the turn test above.

        Ref: https://docs.pipecat.ai/api-reference/server/services/s2s/openai
             docs/api_openai_realtime.md
        """
        service = OpenAIRealtimeLLMService(
            base_url=_DOCUMENTED_BASE_URL,
            api_key=agentic_server.api_key,
            settings=OpenAIRealtimeLLMService.Settings(model=model_config.model),
        )

        assert service.base_url == f"{_DOCUMENTED_BASE_URL}?model={model_config.model}"
