"""The OpenAI Agents SDK driven against the surfaces this gateway added last.

``openai-agents`` is the only client in this lane that speaks the Realtime
WebSocket the way a voice application does: ``RealtimeRunner`` opens the session,
sends the caller's microphone frames, and turns the server events back into its
own conversation history -- so a gateway event the SDK cannot parse fails here
rather than in a hand-written protocol test. The same package reaches three more
surfaces over ``/v1/responses``: the hosted web-search tool, ``/v1/conversations``
through ``OpenAIConversationsSession``, and ``/v1/vector_stores`` through a
retrieval tool.

Two client behaviours shape the configuration below:

- the SDK's default session settings name OpenAI's own voice (``ash``), its own
  transcription model and semantic turn detection. All three are sent verbatim,
  so this module is also the proof that the gateway accepts a session configured
  for OpenAI without the caller adapting anything.
- ``RealtimeSession.send_message`` -- a written turn -- is not exercised here:
  this gateway's realtime models answer a *spoken* turn, and a text-only turn
  leaves the session waiting. The voice path is the one a ``RealtimeRunner``
  application uses, and the one covered.

Tracing is switched off at import: the SDK otherwise exports every run to
OpenAI's own backend, which would send this deployment's traffic to a third
party.

Requires --agentic, podman, and Bedrock credentials.

Ref: https://openai.github.io/openai-agents-python/realtime/guide/
     https://openai.github.io/openai-agents-python/sessions/
     https://openai.github.io/openai-agents-python/tools/
     https://developers.openai.com/api/docs/guides/realtime
     stdapi/routes/openai_realtime.py:router
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from agents import (
    Agent,
    FileSearchTool,
    ModelSettings,
    Runner,
    WebSearchTool,
    function_tool,
    set_tracing_disabled,
)
from agents.memory import OpenAIConversationsSession
from agents.models.openai_responses import OpenAIResponsesModel
from agents.realtime import RealtimeAgent, RealtimeRunner
from openai import AsyncOpenAI, BadRequestError, NotFoundError, OpenAI

from ._runner import ModelConfig, grounding_requests
from ._tools import AgenticTool

if TYPE_CHECKING:
    from collections.abc import Iterator

    from agents.realtime import RealtimeSession

    from ._server import AgenticServer
    from ._tools import AgenticResult, Command, Invocation

pytestmark = pytest.mark.agentic

# Runs are never exported to OpenAI's tracing backend.
set_tracing_disabled(True)


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
#: requests to; openai-agents is a Python library, never run in a container.
TOOL = AgenticTool(
    id="openai-agents",
    npm_package=None,
    binary="python",
    route="/v1",
    metrics_prefix="OA-METRICS",
    build=_unused_build,
    parse=_unused_parse,
    prepare_workdir=_unused_prepare_workdir,
    # The SDK sends no per-run identifier the gateway logs, so its requests can
    # only be attributed positionally.
    attributes_sessions=False,
)

#: Speech-to-speech model the realtime session is opened for.
_REALTIME_MODEL_CONFIG = pytest.param(
    ModelConfig(model="amazon.nova-2-sonic-v1:0"), id="nova-2-sonic"
)

#: Cheap chat model behind every ``/v1/responses`` agent in this module.
_CHAT_MODEL_CONFIG = pytest.param(
    ModelConfig(model="amazon.nova-2-lite-v1:0"), id="nova-2-lite"
)

#: Model synthesizing the caller's speech; the cheapest voice on the gateway.
_SPEECH_MODEL = "amazon.polly-standard"

#: Instructions carried by both the realtime agent and the minted client secret.
#:
#: They have to match: a session opened with a client secret refuses an update
#: that changes the instructions it was minted for.
_REALTIME_INSTRUCTIONS = "Answer with one short spoken sentence."

#: Sentence the caller speaks, synthesized once per module.
_SPOKEN_QUESTION = "Hello there. Please say something back to me."

#: Audio sent per ``input_audio_buffer.append``, ~100 ms at 24 kHz 16-bit mono.
_APPEND_BYTES = 4800

#: Seconds one spoken turn may take end to end, model latency included.
_TURN_TIMEOUT = 120.0

#: Seconds the handshake of a session that never speaks may take.
_HANDSHAKE_TIMEOUT = 60.0

#: Seconds a vector store may spend indexing its single small file.
_INDEX_TIMEOUT = 240.0

#: Seconds a closed realtime session is given to finish tearing its backend down.
_TEARDOWN_SETTLE = 1.5


@pytest.fixture(scope="module")
def gateway_client(agentic_server: AgenticServer) -> OpenAI:
    """Synchronous OpenAI SDK client bound to the gateway under test."""
    return OpenAI(
        base_url=agentic_server.url("/v1"),
        api_key=agentic_server.api_key,
        max_retries=0,
    )


def _async_client(server: AgenticServer) -> AsyncOpenAI:
    """Return the asynchronous client the SDK's agents talk to the gateway with.

    Args:
        server: Gateway the client is bound to.

    Returns:
        A client with retries off, so a gateway failure surfaces as itself.
    """
    return AsyncOpenAI(
        base_url=server.url("/v1"), api_key=server.api_key, max_retries=0
    )


def _agent_model(server: AgenticServer, model: str) -> OpenAIResponsesModel:
    """Return the SDK model object routing an agent to the gateway.

    Args:
        server: Gateway serving the agent.
        model: Model the gateway must route to.

    Returns:
        A Responses-API model bound to the gateway.
    """
    return OpenAIResponsesModel(model=model, openai_client=_async_client(server))


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


async def _realtime_session(
    server: AgenticServer, model: str, *, credential: str
) -> RealtimeSession:
    """Build the realtime session an agent application would open.

    Turn detection is switched off so the caller's ``commit`` is what ends the
    turn: the alternative depends on the backend's voice activity detector
    firing, which is not the gateway behaviour under test.

    Args:
        server: Gateway serving the session.
        model: Realtime model the session is opened for.
        credential: API key or minted client secret to authenticate with.

    Returns:
        The session, not yet connected: entering it opens the WebSocket.
    """
    agent = RealtimeAgent(name="voice-caller", instructions=_REALTIME_INSTRUCTIONS)
    url = server.base_url.replace("http://", "ws://", 1)
    return await RealtimeRunner(agent).run(
        model_config={
            "api_key": credential,
            "url": f"{url}/v1/realtime?model={model}",
            "initial_model_settings": {
                "model_name": model,
                "instructions": _REALTIME_INSTRUCTIONS,
                "turn_detection": None,
            },
        }
    )


async def _speak(session: RealtimeSession, pcm: bytes) -> None:
    """Send *pcm* the way a microphone would, committing on the last chunk.

    Args:
        session: The open realtime session.
        pcm: 24 kHz mono 16-bit samples of the caller's speech.
    """
    view = memoryview(pcm)
    for start in range(0, len(view), _APPEND_BYTES):
        end = start + _APPEND_BYTES
        await session.send_audio(bytes(view[start:end]), commit=end >= len(view))


def _raw_server_event(event: Any) -> dict[str, Any] | None:  # noqa: ANN401
    """Return the gateway event *event* wraps, or None for an SDK-level event.

    Args:
        event: One event yielded by a realtime session.

    Returns:
        The raw server event payload, or None.
    """
    if event.type != "raw_model_event" or event.data.type != "raw_server_event":
        return None
    payload: dict[str, Any] = event.data.data
    return payload


#: The rejected event's type and the first offending field, in a validation report.
#:
#: The SDK reports one error per member of a 45-way tagged union, so the raw text
#: of a single rejection runs to a hundred lines of branches that never applied.
_REJECTION = re.compile(
    r"^`(?P<event>[^`]+)`(?P<path>\S*)\n\s+(?P<detail>.+)$", re.MULTILINE
)


def _rejection(error: str) -> str:
    """Summarize one SDK validation report down to what the gateway got wrong.

    Args:
        error: The report the SDK attached to its ``error`` event.

    Returns:
        The event type, the field, and why it was refused.
    """
    if (found := _REJECTION.search(error)) is None:
        return error.splitlines()[0]
    return f"{found['event']}{found['path']}: {found['detail']}"


@dataclass(slots=True)
class _Turn:
    """Everything one realtime exchange produced, from both sides of the SDK.

    Attributes:
        audio: Bytes of the answer's speech, as the SDK decoded them.
        received: Every raw gateway event, in order.
        history: The conversation the SDK built out of them.
        rejected: Gateway events the SDK's own types refused to parse.
    """

    audio: bytearray = field(default_factory=bytearray)
    received: list[dict[str, Any]] = field(default_factory=list)
    history: list[Any] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


async def _collect_until(
    session: RealtimeSession, terminal: str, seconds: float
) -> _Turn:
    """Read a session until the gateway sends *terminal*.

    Args:
        session: The open realtime session.
        terminal: ``type`` of the gateway event that ends the collection.
        seconds: Time the whole collection may take.

    Returns:
        The exchange, up to and including the terminal event.
    """
    turn = _Turn()
    try:
        async with asyncio.timeout(seconds):
            async for event in session:
                if event.type == "audio":
                    turn.audio.extend(event.audio.data)
                elif event.type in {"history_updated", "history_added"}:
                    turn.history = list(getattr(event, "history", turn.history))
                elif event.type == "error":
                    turn.rejected.append(_rejection(str(event.error)))
                if (payload := _raw_server_event(event)) is None:
                    continue
                turn.received.append(payload)
                if payload.get("type") in {terminal, "error"}:
                    return turn
    except TimeoutError:
        pytest.fail(
            f"no {terminal!r} in {seconds}s; the session sent "
            f"{[payload.get('type') for payload in turn.received]}"
        )
    pytest.fail(
        f"the session ended before {terminal!r}: "
        f"{[payload.get('type') for payload in turn.received]}"
    )


async def _settle(seconds: float = _TEARDOWN_SETTLE) -> None:
    """Wait for the gateway to finish tearing the closed session down.

    The backend of a realtime session is stopped after the WebSocket closes, so
    a fault raised there reaches the server's stderr a moment after the test
    body ends. Without this wait the lane's traceback watchdog reads stderr too
    early and blames whichever test runs next.

    Args:
        seconds: Time to wait.
    """
    await asyncio.sleep(seconds)


def _assert_no_error(turn: _Turn) -> None:
    """Fail when the gateway reported an error event.

    Args:
        turn: The exchange to inspect.
    """
    errors = [payload for payload in turn.received if payload.get("type") == "error"]
    assert not errors, f"the session reported an error: {errors}"


@pytest.mark.parametrize("model_config", [_REALTIME_MODEL_CONFIG])
class TestRealtimeVoiceSession:
    """A ``RealtimeRunner`` application holding a spoken turn with the gateway.

    Ref: https://openai.github.io/openai-agents-python/realtime/guide/
         https://developers.openai.com/api/docs/guides/realtime
         stdapi/routes/openai_realtime.py:openai_realtime
    """

    async def test_a_spoken_turn_answers_with_audio_a_transcript_and_usage(
        self,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        spoken_pcm: bytes,
    ) -> None:
        """The caller's speech comes back as audio, a transcript and billed usage.

        The SDK's own history is asserted alongside the wire events: it is built
        by parsing ``response.output_item`` and ``conversation.item`` events, so
        an answer the gateway reports in a shape the SDK cannot read produces
        audio and an empty conversation.

        Ref: https://developers.openai.com/api/reference/resources/realtime
             stdapi/realtime.py:RealtimeSession
        """
        async with await _realtime_session(
            agentic_server, model_config.model, credential=agentic_server.api_key
        ) as session:
            await _speak(session, spoken_pcm)
            turn = await _collect_until(session, "response.done", _TURN_TIMEOUT)
        await _settle()

        _assert_no_error(turn)
        kinds = [payload.get("type") for payload in turn.received]
        assert "input_audio_buffer.committed" in kinds, kinds
        assert turn.audio, "the spoken turn returned no audio"

        answer = turn.received[-1]["response"]
        transcripts = [
            part.get("transcript")
            for item in answer.get("output", [])
            for part in item.get("content", [])
        ]
        assert any(transcripts), f"the answer carried no transcript: {answer}"
        assert answer["usage"]["output_tokens"] > 0, answer["usage"]
        assert [item for item in turn.history if item.role == "assistant"], (
            f"the SDK built no assistant turn from the session: {turn.history}"
        )

    async def test_a_minted_client_secret_opens_the_session_it_carries(
        self,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        gateway_client: OpenAI,
    ) -> None:
        """A browser-style client secret authenticates the SDK's own handshake.

        This is the flow the client-secret route exists for: the deployment's key
        never leaves the server, and the untrusted client connects with a secret
        that pins the session. The SDK sends its own ``session.update`` on every
        connection, so the pinned instructions have to survive it -- an update
        that contradicted them would be refused instead of acknowledged.

        Ref: https://developers.openai.com/api/reference/resources/realtime/subresources/client_secrets/methods/create
             stdapi/realtime.py:read_client_secret
        """
        minted = gateway_client.realtime.client_secrets.create(
            session={
                "type": "realtime",
                "model": model_config.model,
                "instructions": _REALTIME_INSTRUCTIONS,
            }
        )

        async with await _realtime_session(
            agentic_server, model_config.model, credential=minted.value
        ) as session:
            turn = await _collect_until(session, "session.updated", _HANDSHAKE_TIMEOUT)
        await _settle()

        _assert_no_error(turn)
        opened = turn.received[0]
        assert opened["type"] == "session.created", opened
        assert opened["session"]["id"], "the created session carries no id"
        assert turn.received[-1]["session"]["instructions"] == _REALTIME_INSTRUCTIONS

    async def test_every_event_of_a_spoken_turn_parses_with_the_official_types(
        self,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        spoken_pcm: bytes,
    ) -> None:
        """No event of a spoken turn is refused by the SDK's own event models.

        The SDK validates every frame against the ``openai`` package's Realtime
        types and surfaces a rejected one as an ``error`` event, so this is the
        one assertion that measures the gateway against the published event
        shapes rather than against what a reader of the events can cope with.

        Ref: https://developers.openai.com/api/reference/resources/realtime
             stdapi/realtime.py:_item_body
        """
        async with await _realtime_session(
            agentic_server, model_config.model, credential=agentic_server.api_key
        ) as session:
            await _speak(session, spoken_pcm)
            turn = await _collect_until(session, "response.done", _TURN_TIMEOUT)
        await _settle()

        assert not turn.rejected, (
            "the SDK could not parse "
            f"{len(turn.rejected)} of the session's events:\n"
            + "\n".join(turn.rejected)
        )


@pytest.mark.expensive
@pytest.mark.parametrize("model_config", [_CHAT_MODEL_CONFIG])
class TestHostedWebSearch:
    """The SDK's ``WebSearchTool`` reaches the gateway's billed web search.

    Ref: https://openai.github.io/openai-agents-python/tools/
         https://developers.openai.com/api/docs/guides/tools-web-search
    """

    async def test_the_web_search_tool_runs_and_is_billed(
        self, model_config: ModelConfig, agentic_server: AgenticServer
    ) -> None:
        """A searched answer carries completed ``web_search_call`` items and usage.

        ``tool_choice="required"`` removes the model's own decision from the
        assertion. The billed grounding count is read from the gateway's usage
        log because the search happens inside the model invocation: a response
        shaped like a search but never billed would otherwise look identical.

        Ref: https://developers.openai.com/api/docs/guides/tools-web-search
             tests/agentic/_runner.py:grounding_requests
        """
        log_start = len(agentic_server.logs)
        agent = Agent(
            name="searcher",
            instructions="Search the web, then answer in one short sentence.",
            model=_agent_model(agentic_server, model_config.model),
            tools=[WebSearchTool()],
            model_settings=ModelSettings(tool_choice="required"),
        )

        result = await Runner.run(agent, "What is the current version of Python?")

        items = [item for raw in result.raw_responses for item in raw.output]
        searches = [item for item in items if item.type == "web_search_call"]
        assert searches, f"no web_search_call item was produced: {items}"
        assert all(item.status == "completed" for item in searches), searches
        assert not [item for item in items if item.type == "function_call"], (
            f"the hosted search leaked as a client-side function call: {items}"
        )
        assert result.final_output
        assert grounding_requests(agentic_server, log_start) >= 1, (
            "the gateway recorded no billed web-grounding call"
        )


#: Fact planted in the first turn, whose recall proves the stored turns were read
#: back and replayed to the model.
_PLANTED_NUMBER = "4817"


@pytest.mark.parametrize("model_config", [_CHAT_MODEL_CONFIG])
class TestConversationsSession:
    """``OpenAIConversationsSession`` stores an agent's turns on the gateway.

    Ref: https://openai.github.io/openai-agents-python/sessions/
         https://developers.openai.com/api/reference/resources/conversations
    """

    async def test_a_two_turn_run_is_stored_replayed_and_deletable(
        self, model_config: ModelConfig, agentic_server: AgenticServer
    ) -> None:
        """Both turns are stored in order, reach the second run, and delete cleanly.

        The SDK keeps no history of its own: the second run's input is exactly
        what it read back from the conversation, which is why that input is
        asserted rather than the follow-up answer. A store that lost a turn,
        reordered two or dropped the assistant's reply is visible there, while
        the answer itself is the model's to choose.

        Ref: https://developers.openai.com/api/reference/resources/conversations
             stdapi/routes/openai_conversations.py:add_items
        """
        client = _async_client(agentic_server)
        session = OpenAIConversationsSession(openai_client=client)
        agent = Agent(
            name="rememberer",
            instructions="Answer in one short sentence.",
            model=_agent_model(agentic_server, model_config.model),
        )

        await Runner.run(
            agent,
            f"The mirror was recoated under reference {_PLANTED_NUMBER}. Reply OK.",
            session=session,
        )
        follow_up = await Runner.run(
            agent, "Which reference number did I mention?", session=session
        )
        conversation_id = session.session_id
        items = await session.get_items()

        assert [item.get("role") for item in items] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ], items
        assert _PLANTED_NUMBER in str(items[0]), (
            f"the stored first turn lost the text it was created with: {items[0]}"
        )
        replayed = follow_up.input
        assert isinstance(replayed, list)
        assert [item.get("role") for item in replayed] == [
            "user",
            "assistant",
            "user",
        ], replayed
        assert _PLANTED_NUMBER in str(replayed[0]), (
            f"the stored turns never reached the second run: {replayed}"
        )

        await session.clear_session()
        with pytest.raises(NotFoundError):
            await client.conversations.retrieve(conversation_id)


#: Note indexed in the vector store, with a fact no model can already know.
_NOTE = (
    "Kestrel Observatory maintenance log.\n\n"
    "The primary mirror was recoated on 14 March 2031 by the night crew.\n"
    f"They logged the job under reference number {_PLANTED_NUMBER}.\n"
).encode()


@pytest.fixture(scope="module")
def indexed_store(gateway_client: OpenAI) -> Iterator[str]:
    """A vector store holding one indexed note, deleted with its file at the end.

    Module-scoped: indexing costs a real embedding call per chunk, and every test
    here reads the same note.

    Yields:
        The vector store ID.
    """
    uploaded = gateway_client.files.create(
        file=("kestrel.txt", _NOTE, "text/plain"), purpose="assistants"
    )
    store = gateway_client.vector_stores.create(
        name="stdapi-agentic-openai-agents", file_ids=[uploaded.id]
    )
    try:
        deadline = time.monotonic() + _INDEX_TIMEOUT
        while True:
            counts = gateway_client.vector_stores.retrieve(store.id).file_counts
            if counts.in_progress == 0 and counts.total >= 1:
                break
            assert time.monotonic() < deadline, (
                f"vector store {store.id} still indexing after "
                f"{_INDEX_TIMEOUT}s: {counts}"
            )
            time.sleep(2.0)
        assert counts.completed == 1, counts
        yield store.id
    finally:
        gateway_client.vector_stores.delete(store.id)
        gateway_client.files.delete(uploaded.id)


@pytest.mark.parametrize("model_config", [_CHAT_MODEL_CONFIG])
class TestVectorStoreRetrieval:
    """An agent retrieving from a gateway vector store, hosted and client-side.

    Ref: https://openai.github.io/openai-agents-python/tools/
         https://developers.openai.com/api/reference/resources/vector_stores
    """

    async def test_a_retrieval_tool_answers_from_the_indexed_note(
        self,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        indexed_store: str,
    ) -> None:
        """Searching the store inside a tool call puts the note in the answer.

        The reference number exists only in the indexed file, so an answer
        carrying it proves the search returned the chunk and the agent loop
        carried it back through the gateway.

        Ref: https://developers.openai.com/api/reference/resources/vector_stores
             stdapi/routes/openai_vector_stores.py:search_vector_store
        """
        client = _async_client(agentic_server)

        @function_tool
        async def search_notes(query: str) -> str:
            """Search the observatory maintenance notes.

            Args:
                query: What to look for.

            Returns:
                The text of every matching passage.
            """
            found = await client.vector_stores.search(
                vector_store_id=indexed_store, query=query
            )
            return "\n".join(
                part.text
                for hit in found.data
                for part in hit.content
                if part.type == "text"
            )

        agent = Agent(
            name="librarian",
            instructions=(
                "Answer only from the maintenance notes returned by search_notes."
            ),
            model=_agent_model(agentic_server, model_config.model),
            tools=[search_notes],
        )

        result = await Runner.run(
            agent, "Which reference number did the crew log for the mirror job?"
        )

        calls = [
            item
            for raw in result.raw_responses
            for item in raw.output
            if item.type == "function_call"
        ]
        assert calls, "the agent never searched the vector store"
        assert _PLANTED_NUMBER in result.final_output, result.final_output

    async def test_a_hosted_file_search_tool_is_refused_with_the_way_forward(
        self,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        indexed_store: str,
    ) -> None:
        """``FileSearchTool`` is refused as a request, not silently dropped.

        Hosted retrieval *is* the request: an answer produced without it would
        read as grounded in the attached store while coming from the model's own
        knowledge. The refusal has to reach the SDK as a plain 400 naming the
        route that does serve the search -- which is what the test above then
        does -- rather than as a 500 or an ungrounded answer.

        Ref: https://developers.openai.com/api/docs/guides/tools
             stdapi/models/chat/_adapters/_openai_responses.py:_build_tool_config
        """
        agent = Agent(
            name="reader",
            instructions="Answer in one short sentence.",
            model=_agent_model(agentic_server, model_config.model),
            tools=[FileSearchTool(vector_store_ids=[indexed_store])],
        )

        with pytest.raises(BadRequestError) as raised:
            await Runner.run(agent, "Reply with the word OK.")

        error = raised.value.body
        assert isinstance(error, dict)
        assert error["type"] == "invalid_request_error", error
        assert "file_search" in str(error["message"]), error
        assert "vector store" in str(error["message"]).lower(), error
