"""LiveKit Agents driven against this gateway's Realtime WebSocket.

``docs/api_openai_realtime.md`` tells a customer who needs WebRTC or a phone
line to put LiveKit Agents in front of this deployment and change one thing --
the base URL. This module is that recipe, executed: the configuration here is
the documented snippet, and the turn it holds is the one a voice application
holds.

LiveKit terminates WebRTC on its own side, so the half this API serves is the
WebSocket between ``RealtimeModel`` and the model. ``AgentSession`` is the
wrapper a deployed agent runs inside -- it needs a LiveKit room, which no test
here provides -- while ``RealtimeModel.session()`` is that same WebSocket
without the room, which is why the turn is driven through it.

Two behaviours of the plugin shape what is asserted:

- ``base_url`` is an **HTTP** base URL. The plugin turns it into the WebSocket
  URL itself, appending ``/realtime`` and adding the model as a query
  parameter, so a deployment served under the default prefix names ``/v1`` and
  nothing else. That derivation is the sentence the documentation asks a reader
  to rely on, so it is pinned here as well as exercised.
- the plugin's own defaults are sent verbatim -- its voice (``marin``) and its
  modalities -- so this module is also the proof that a session configured for
  OpenAI is accepted without the caller adapting anything.

Requires --agentic, podman, and Bedrock credentials.

Ref: https://docs.livekit.io/agents/models/realtime/plugins/openai/
     https://developers.openai.com/api/docs/guides/realtime
     docs/api_openai_realtime.md
     stdapi/routes/openai_realtime.py:router
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import pytest
from livekit import rtc
from livekit.agents import AgentSession
from livekit.plugins import openai as livekit_openai
from livekit.plugins.openai.realtime.realtime_model import process_base_url
from openai import OpenAI

from ._runner import ModelConfig
from ._tools import AgenticTool

if TYPE_CHECKING:
    from collections.abc import AsyncIterable

    from livekit.agents.llm.realtime import RealtimeSession

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
#: requests to; livekit-agents is a Python library, never run in a container.
TOOL = AgenticTool(
    id="livekit-agents",
    npm_package=None,
    binary="python",
    route="/v1",
    metrics_prefix="LK-METRICS",
    build=_unused_build,
    parse=_unused_parse,
    prepare_workdir=_unused_prepare_workdir,
    # The plugin sends no per-run identifier the gateway logs, so its requests
    # can only be attributed positionally.
    attributes_sessions=False,
)

#: Speech-to-speech model every session in this module is opened for.
_REALTIME_MODEL_CONFIG = pytest.param(
    ModelConfig(model="amazon.nova-2-sonic-v1:0"), id="nova-2-sonic"
)

#: Model synthesizing the caller's speech; the cheapest voice on the gateway.
_SPEECH_MODEL = "amazon.polly-standard"

#: Instructions the voice agent answers under.
_INSTRUCTIONS = "Answer with one short spoken sentence."

#: Sentence the caller speaks, synthesized once per module.
_SPOKEN_QUESTION = "Hello there. Please say something back to me."

#: Audio pushed per frame, ~100 ms at 24 kHz 16-bit mono.
_FRAME_BYTES = 4800

#: Sample rate a realtime session takes its input audio at.
_SAMPLE_RATE = 24000

#: Bytes per 16-bit mono sample.
_SAMPLE_WIDTH = 2

#: Seconds one spoken turn may take end to end, model latency included.
_TURN_TIMEOUT = 120.0

#: Seconds a closed realtime session is given to finish tearing its backend down.
_TEARDOWN_SETTLE = 1.5

#: Deployment the documentation's snippet is written against.
_DOCUMENTED_BASE_URL = "https://your-deployment.example.com/v1"


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


def _realtime_model(
    server: AgenticServer, model: str
) -> livekit_openai.realtime.RealtimeModel:
    """Build the plugin exactly as ``docs/api_openai_realtime.md`` builds it.

    Args:
        server: Gateway serving the session.
        model: Realtime model the session is opened for.

    Returns:
        The model object, not yet connected.
    """
    return livekit_openai.realtime.RealtimeModel(
        model=model, base_url=server.url("/v1"), api_key=server.api_key
    )


@dataclass(slots=True)
class _Turn:
    """Everything one spoken exchange produced, as the plugin reported it.

    Attributes:
        text: The answer's transcript, as the plugin streamed it.
        audio: Bytes of the answer's speech, as the plugin decoded them.
        heard: Final transcripts of what the caller said.
        errors: Gateway errors the plugin surfaced.
    """

    text: str = ""
    audio: bytearray = field(default_factory=bytearray)
    heard: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


async def _speak(session: RealtimeSession[Any], pcm: bytes) -> None:
    """Push *pcm* the way a microphone would, committing once it is all sent.

    Args:
        session: The open realtime session.
        pcm: 24 kHz mono 16-bit samples of the caller's speech.
    """
    view = memoryview(pcm)
    for start in range(0, len(view), _FRAME_BYTES):
        chunk = bytes(view[start : start + _FRAME_BYTES])
        session.push_audio(
            rtc.AudioFrame(chunk, _SAMPLE_RATE, 1, len(chunk) // _SAMPLE_WIDTH)
        )
    session.commit_audio()


async def _hold_a_turn(session: RealtimeSession[Any], pcm: bytes) -> _Turn:
    """Speak into *session* and read the answer back through the plugin.

    The answer is read from the ``generation_created`` event rather than from
    the wire: those streams are what an ``AgentSession`` plays out and puts in
    its history, so an answer the plugin cannot assemble is empty here even
    though the gateway sent it.

    Args:
        session: The open realtime session.
        pcm: The caller's speech.

    Returns:
        The exchange.
    """
    turn = _Turn()
    generations: asyncio.Queue[Any] = asyncio.Queue()
    session.on("generation_created", generations.put_nowait)
    session.on("error", lambda event: turn.errors.append(str(event)))
    session.on(
        "input_audio_transcription_completed",
        lambda event: turn.heard.append(event.transcript) if event.is_final else None,
    )

    await session.update_instructions(_INSTRUCTIONS)
    await _speak(session, pcm)

    async with asyncio.timeout(_TURN_TIMEOUT):
        generation = await generations.get()
        async for message in generation.message_stream:
            # Both streams have to be drained together: the plugin fills them
            # from one WebSocket reader, which stalls as soon as either backs up.
            await asyncio.gather(
                _drain_text(message.text_stream, turn),
                _drain_audio(message.audio_stream, turn),
            )
    return turn


async def _drain_text(stream: AsyncIterable[str], turn: _Turn) -> None:
    """Append every text part of one answer to *turn*.

    Args:
        stream: The answer's text stream.
        turn: Exchange to record into.
    """
    async for part in stream:
        turn.text += part


async def _drain_audio(stream: AsyncIterable[rtc.AudioFrame], turn: _Turn) -> None:
    """Append every audio frame of one answer to *turn*.

    Args:
        stream: The answer's audio stream.
        turn: Exchange to record into.
    """
    async for frame in stream:
        turn.audio.extend(frame.data)


@pytest.mark.parametrize("model_config", [_REALTIME_MODEL_CONFIG])
class TestLiveKitRealtimeSession:
    """A LiveKit voice agent holding a spoken turn with this gateway.

    Ref: https://docs.livekit.io/agents/models/realtime/plugins/openai/
         docs/api_openai_realtime.md
         stdapi/routes/openai_realtime.py:openai_realtime
    """

    async def test_a_spoken_turn_answers_with_audio_and_a_transcript(
        self,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        spoken_pcm: bytes,
    ) -> None:
        """The caller's speech comes back as audio and text through the plugin.

        Turn detection is switched off so the caller's own commit ends the turn:
        the alternative depends on the backend's voice activity detector firing,
        which is not the gateway behaviour under test.

        The caller's transcript is asserted alongside the answer because the
        plugin only reports it when the gateway sends the input-transcription
        events, which is what an agent shows in its own conversation view.

        Ref: https://developers.openai.com/api/reference/resources/realtime
             stdapi/realtime.py:RealtimeSession
        """
        model = _realtime_model(agentic_server, model_config.model)
        session = model.session(turn_detection_disabled=True)
        try:
            turn = await _hold_a_turn(session, spoken_pcm)
        finally:
            await session.aclose()
            await model.aclose()
            await asyncio.sleep(_TEARDOWN_SETTLE)

        assert not turn.errors, f"the session reported an error: {turn.errors}"
        assert turn.audio, "the spoken turn returned no audio"
        assert turn.text.strip(), "the spoken turn returned no transcript"
        assert any(heard.strip() for heard in turn.heard), (
            f"the plugin never received a transcript of the caller: {turn.heard}"
        )

    async def test_the_documented_snippet_derives_this_deployment_websocket(
        self, model_config: ModelConfig, agentic_server: AgenticServer
    ) -> None:
        """The documented ``base_url`` becomes ``/v1/realtime`` on this deployment.

        The recipe hands the plugin an **HTTP** base URL and promises that
        ``/realtime`` is appended for the caller. That promise is the plugin's
        to keep, so it is pinned here: a release that stopped appending the path
        would leave every reader of that page dialling the wrong URL, and the
        spoken-turn test above would only report it as a connection failure.

        Ref: https://docs.livekit.io/agents/models/realtime/plugins/openai/
             docs/api_openai_realtime.md
        """
        session: AgentSession[Any] = AgentSession(
            llm=livekit_openai.realtime.RealtimeModel(
                model=model_config.model,
                base_url=_DOCUMENTED_BASE_URL,
                api_key=agentic_server.api_key,
            )
        )

        assert isinstance(session.llm, livekit_openai.realtime.RealtimeModel)
        dialled = process_base_url(_DOCUMENTED_BASE_URL, model_config.model)
        assert dialled.startswith("wss://your-deployment.example.com/v1/realtime?")
        assert f"model={quote(model_config.model)}" in dialled, dialled
