"""pydantic-ai driven against ``/v1/chat/completions`` and ``/v1/responses``.

Its ``OpenAIChatModel`` carries the module's central finding, on reasoning replay.
``OpenAIResponsesModel`` carries two more: it is the only client in this lane that
reaches ``/v1/conversations`` by naming a conversation on the turn itself rather
than through a session object, and the only one whose ``file_search`` tool is
declared under a *renamed* field -- ``FileSearchTool.file_store_ids``, which has to
arrive as ``vector_store_ids`` on our wire.

pydantic-ai's ``OpenAIChatModel`` reads a model's thinking text back from whichever
of the ``reasoning``/``reasoning_content`` fields the response actually carries --
falling back through both when no custom field is configured on the model profile,
verified empirically in the installed package's
``pydantic_ai.models.openai.OpenAIChatModel._process_thinking`` -- and replays it on
the next turn under that same field name, with no signature: the OpenAI Chat
Completions wire format has nowhere to carry one
(``pydantic_ai.messages.ThinkingPart.signature`` stays ``None`` for a field-sourced
part). Claude models on this gateway reject exactly that kind of unsigned replay, so
the gateway drops the block instead of the whole request. This module is the
empirical proof that a real pydantic-ai multi-turn tool loop survives that drop.

Requires --agentic, podman, and Bedrock credentials.

Ref: https://pydantic.dev/docs/ai/models/openai/
     https://api-docs.deepseek.com/api/create-chat-completion
     docs/api_openai_chat_completions.md#replaying-reasoning-in-a-multi-turn-conversation
     stdapi/models/chat/__init__.py:ChatModelBase.REASONING_SIGNATURE_REQUIRED
     stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel
     stdapi/models/chat/_adapters/_openai_chat_completion.py:_map_assistant_reasoning_content
     stdapi/config.py:_Settings.chat_completions_reasoning_field
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from openai import OpenAI
from pydantic_ai import Agent, FileSearchTool
from pydantic_ai.capabilities import AbstractCapability, NativeTool
from pydantic_ai.messages import (
    ModelResponse,
    NativeToolCallPart,
    ThinkingPart,
    ToolCallPart,
)
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.providers.openai import OpenAIProvider

from ._runner import ModelConfig
from ._tools import AgenticTool
from ._vector_store import PLANTED_NUMBER, indexed_store

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from pydantic_ai.messages import ModelMessage

    from ._server import AgenticServer
    from ._tools import AgenticResult, Command, Invocation

pytestmark = pytest.mark.agentic

#: Seconds allowed for a two-turn Bedrock round trip through pydantic-ai.
_TIMEOUT = 120


def _unused_build(invocation: Invocation) -> Command:
    """Never called: this module drives no CLI, only the shared identity check needs a tool."""
    raise NotImplementedError


def _unused_parse(stdout: str) -> AgenticResult:
    """Never called: see :func:`_unused_build`."""
    raise NotImplementedError


def _unused_prepare_workdir(invocation: Invocation) -> None:
    """Never called: see :func:`_unused_build`."""
    raise NotImplementedError


#: Registered purely so the autouse model-identity check has a tool to attribute
#: requests to; pydantic-ai is a plain HTTP client library, never run in a container.
TOOL = AgenticTool(
    id="pydantic-ai",
    npm_package=None,
    binary="python",
    route="/v1",
    metrics_prefix="PA-METRICS",
    build=_unused_build,
    parse=_unused_parse,
    prepare_workdir=_unused_prepare_workdir,
    # pydantic-ai sends no per-run identifier the gateway records, so its requests
    # can only be attributed positionally.
    attributes_sessions=False,
)

#: Models exercised by the general tool-call round trip.
_MODEL_CONFIGS = [
    pytest.param(
        ModelConfig(model="anthropic.claude-haiku-4-5-20251001-v1:0", timeout=_TIMEOUT),
        id="claude-haiku-4-5",
    ),
    pytest.param(
        ModelConfig(model="amazon.nova-2-lite-v1:0", timeout=_TIMEOUT), id="nova-2-lite"
    ),
]

#: Cheapest model of the one family this gateway forces to drop a replayed
#: reasoning block; the flag is set once for every Claude model.
_REASONING_MODEL_CONFIG = pytest.param(
    ModelConfig(model="anthropic.claude-haiku-4-5-20251001-v1:0", timeout=_TIMEOUT),
    id="claude-haiku-4-5",
)


def _response_parts[ResponsePart: (ThinkingPart, ToolCallPart, NativeToolCallPart)](
    messages: Sequence[ModelMessage], part_type: type[ResponsePart]
) -> list[ResponsePart]:
    """Return every part of *part_type* across every assistant turn in *messages*.

    Args:
        messages: Full run history, as returned by ``AgentRunResult.all_messages()``.
        part_type: Concrete ``ModelResponse`` part type to collect.

    Returns:
        Every matching part, in turn order.
    """
    return [
        part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, part_type)
    ]


def _agent_for(server: AgenticServer, config: ModelConfig) -> Agent[None, str]:
    """Build a pydantic-ai agent whose model points at the gateway under test.

    Args:
        server: Gateway the agent talks to.
        config: Model under test.

    Returns:
        An agent with no tools registered yet.
    """
    model = OpenAIChatModel(
        config.model,
        provider=OpenAIProvider(base_url=server.url("/v1"), api_key=server.api_key),
    )
    return Agent(
        model, system_prompt="Call the registered tool to answer the question asked."
    )


#: Values only the tool call can reveal, so the model cannot answer without calling it.
_MAGIC_NUMBERS = {"zephyr": 4817, "quoll": 2603}


def _register_lookup_tool(agent: Agent[None, str]) -> None:
    """Register the one tool whose result the tests look for in the final answer.

    Unlike arithmetic, a model cannot guess the registered value, so a correct
    answer is proof the tool was actually called and its result read back.

    Args:
        agent: Agent to register the tool on.
    """

    @agent.tool_plain
    def magic_number(key: str) -> int:
        """Look up the registered magic number for *key*."""
        return _MAGIC_NUMBERS[key]


@pytest.mark.parametrize("model_config", [_REASONING_MODEL_CONFIG])
class TestReasoningReplaySurvivesToolLoop:
    """The central finding: an unsigned reasoning replay does not break Claude's loop.

    ``reasoning_effort="low"`` turns on Claude Haiku 4.5's extended thinking, which
    forces a ``reasoning_content`` block on the first turn. pydantic-ai then replays
    that block, unsigned, when it sends the tool result back -- exactly the case
    ``_map_assistant_reasoning_content`` drops with a warning instead of rejecting.

    Ref: stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel
         stdapi/models/chat/_adapters/_openai_chat_completion.py:_map_assistant_reasoning_content
    """

    def test_multiturn_tool_call_completes_with_reasoning_enabled(
        self, model_config: ModelConfig, agentic_server: AgenticServer
    ) -> None:
        """The run completes and answers correctly despite the dropped reasoning block.

        The model must both reason (proving the block existed to drop) and call the
        tool (proving the loop reached a second turn that replayed it), or the test
        proves nothing about the code path it targets.
        """
        agent = _agent_for(agentic_server, model_config)
        _register_lookup_tool(agent)
        result = agent.run_sync(
            "Call the magic_number tool with key='zephyr' to find the magic "
            "number, then state it in your answer.",
            model_settings=OpenAIChatModelSettings(openai_reasoning_effort="low"),
        )
        messages = result.all_messages()
        thinking_parts = _response_parts(messages, ThinkingPart)
        tool_calls = _response_parts(messages, ToolCallPart)
        print(  # noqa: T201
            f"\n{TOOL.metrics_prefix} | {model_config.model:<30} | "
            "test_multiturn_tool_call_completes_with_reasoning_enabled "
            f"| reasoning_parts={len(thinking_parts):>2} tool_calls={len(tool_calls):>2}"
        )
        assert thinking_parts, (
            "Claude Haiku 4.5 never reasoned with reasoning_effort='low'"
        )
        assert tool_calls, "the agent never called the magic_number tool"
        assert all(part.signature is None for part in thinking_parts), (
            "a signed reasoning part would not exercise the unsigned-replay path"
        )
        assert "4817" in result.output


#: Bedrock Mantle model verified (tests/probes/results/) to emit reasoning text
#: under `reasoning` at high effort, plus an observed tool call, on
#: /v1/chat/completions.
_MANTLE_REASONING_MODEL_CONFIG = pytest.param(
    ModelConfig(model="qwen.qwen3-32b", timeout=_TIMEOUT), id="qwen3-32b"
)


@pytest.mark.parametrize("model_config", [_MANTLE_REASONING_MODEL_CONFIG])
class TestMantleReasoningReplaySurvivesToolLoop:
    """pydantic-ai's `reasoning`/`reasoning_content` fallback, over a Mantle model.

    Bedrock Mantle emits a model's reasoning text under `reasoning`, where
    Converse emits it under `reasoning_content`; ``OpenAIChatModel`` falls back
    through both fields with no provider profile configured for this model, so
    this is the client-level guard for that gateway-specific field rename.

    Ref: stdapi/types/openai_chat_completions.py:_rename_emitted_reasoning
    """

    def test_multiturn_tool_call_completes_with_reasoning_enabled(
        self, model_config: ModelConfig, agentic_server: AgenticServer
    ) -> None:
        """The run completes and answers correctly with reasoning read from `reasoning`.

        qwen.qwen3-32b only emits reasoning text at ``reasoning_effort="high"``
        (verified in the probe corpus; ``"low"`` has no observable effect), unlike
        Claude Haiku 4.5 in :class:`TestReasoningReplaySurvivesToolLoop` above.
        """
        agent = _agent_for(agentic_server, model_config)
        _register_lookup_tool(agent)
        result = agent.run_sync(
            "Call the magic_number tool with key='zephyr' to find the magic "
            "number, then state it in your answer.",
            model_settings=OpenAIChatModelSettings(openai_reasoning_effort="high"),
        )
        messages = result.all_messages()
        thinking_parts = _response_parts(messages, ThinkingPart)
        tool_calls = _response_parts(messages, ToolCallPart)
        print(  # noqa: T201
            f"\n{TOOL.metrics_prefix} | {model_config.model:<30} | "
            "test_multiturn_tool_call_completes_with_reasoning_enabled "
            f"| reasoning_parts={len(thinking_parts):>2} tool_calls={len(tool_calls):>2}"
        )
        assert thinking_parts, "qwen3-32b never reasoned with reasoning_effort='high'"
        assert tool_calls, "the agent never called the magic_number tool"
        assert "4817" in result.output


@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
class TestToolRoundTripAcrossModels:
    """The same tool-call round trip generalizes beyond Claude's reasoning path.

    No reasoning is requested here: this is the plain multi-turn tool loop every
    model in the lane must complete, Claude's reasoning-drop notwithstanding.
    """

    def test_tool_call_round_trip(
        self, model_config: ModelConfig, agentic_server: AgenticServer
    ) -> None:
        """The agent calls the tool and reports its result, on every model."""
        agent = _agent_for(agentic_server, model_config)
        _register_lookup_tool(agent)
        result = agent.run_sync(
            "Call the magic_number tool with key='quoll' to find the magic "
            "number, then state it in your answer."
        )
        tool_calls = _response_parts(result.all_messages(), ToolCallPart)
        print(  # noqa: T201
            f"\n{TOOL.metrics_prefix} | {model_config.model:<30} | "
            f"test_tool_call_round_trip | tool_calls={len(tool_calls):>2}"
        )
        assert tool_calls, "the agent never called the magic_number tool"
        assert "2603" in result.output


#: Cheap chat model behind the two ``/v1/responses`` surfaces below.
_RESPONSES_MODEL_CONFIG = pytest.param(
    ModelConfig(model="amazon.nova-2-lite-v1:0", timeout=_TIMEOUT), id="nova-2-lite"
)


@pytest.fixture(scope="module")
def gateway_client(agentic_server: AgenticServer) -> OpenAI:
    """Synchronous OpenAI SDK client bound to the gateway under test.

    pydantic-ai creates neither a conversation nor a vector store: both are
    resources an application provisions with the OpenAI SDK and then names on the
    turn, so the SDK is what stands in for that application here.
    """
    return OpenAI(
        base_url=agentic_server.url("/v1"),
        api_key=agentic_server.api_key,
        max_retries=0,
    )


@pytest.fixture(scope="module")
def note_store(gateway_client: OpenAI) -> Iterator[str]:
    """A vector store holding one indexed note, deleted with its file at the end.

    Module-scoped: indexing costs a real embedding call per chunk, and the note is
    the same for every reader of it.

    Yields:
        The vector store ID.
    """
    with indexed_store(gateway_client, "stdapi-agentic-pydantic-ai") as store_id:
        yield store_id


def _responses_agent(
    server: AgenticServer,
    config: ModelConfig,
    capabilities: Sequence[AbstractCapability[None]] = (),
) -> Agent[None, str]:
    """Build an agent whose Responses model points at the gateway under test.

    Args:
        server: Gateway the agent talks to.
        config: Model under test.
        capabilities: Native tools the model serves itself, if any.

    Returns:
        The agent, answering in one sentence.
    """
    model = OpenAIResponsesModel(
        config.model,
        provider=OpenAIProvider(base_url=server.url("/v1"), api_key=server.api_key),
    )
    return Agent(
        model, instructions="Answer in one short sentence.", capabilities=capabilities
    )


def _logged_requests(
    server: AgenticServer, log_start: int, method: str, path: str
) -> list[Mapping[str, object]]:
    """Return the requests the gateway logged for *method* and *path*.

    Args:
        server: Gateway the client was pointed at.
        log_start: Log index captured before the test ran.
        method: HTTP method to match.
        path: Exact request path to match.

    Returns:
        One log entry per matching request, in order.
    """
    return [
        entry
        for entry in server.log_entries(log_start)
        if entry.get("type") == "request"
        and entry.get("method") == method
        and str(entry.get("path") or "") == path
    ]


def _conversation_ids(messages: Sequence[ModelMessage]) -> list[str]:
    """Return the conversation ID every assistant turn in *messages* reports.

    Args:
        messages: Run history, as returned by ``AgentRunResult.all_messages()``.

    Returns:
        One ID per response the gateway echoed a conversation on, in turn order.
    """
    return [
        found
        for message in messages
        if isinstance(message, ModelResponse)
        and isinstance(
            found := (message.provider_details or {}).get("conversation_id"), str
        )
    ]


#: Reference number stored in the conversation, never sent by the client again.
_STORED_NUMBER = "3517"

#: The item planted server-side, in the shape the conversation items route takes.
_STORED_ITEM = {
    "type": "message",
    "role": "user",
    "content": [
        {
            "type": "input_text",
            "text": (
                "Remember this for later: the mirror was recoated under "
                f"reference number {_STORED_NUMBER}."
            ),
        }
    ],
}


@pytest.mark.parametrize("model_config", [_RESPONSES_MODEL_CONFIG])
class TestServerSideConversationState:
    """``openai_conversation_id`` names a gateway conversation on the turn itself.

    pydantic-ai holds no session object and creates no conversation: it sends
    ``conversation`` on the request and reads the ID back off the response into
    ``provider_details``, which is what ``'auto'`` then chains on. So the whole
    round trip depends on the gateway both *serving* the stored items and
    *echoing* the conversation it served them from -- a response that dropped the
    echo silently downgrades the next turn to a full-history resend.

    Ref: https://pydantic.dev/docs/ai/models/openai/#using-durable-conversations
         https://developers.openai.com/api/reference/resources/conversations
         docs/api_openai_conversations.md
         stdapi/routes/openai_responses.py:_echo_chaining
    """

    def test_a_stored_conversation_answers_a_turn_that_never_carried_the_fact(
        self,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        gateway_client: OpenAI,
    ) -> None:
        """The answer comes from items the gateway stored, not from the input sent.

        The reference number is planted straight into the conversation and never
        appears in either run's own input, so recalling it can only come from the
        gateway prepending the stored items. The second run is handed the first
        run's history with ``'auto'``: it reuses the conversation the response
        reported, which is what distinguishes real server-side state from
        ``previous_response_id`` chaining -- and the logged request bodies are
        asserted for it, because a client resending the whole history would
        answer just as well.

        Ref: https://developers.openai.com/api/reference/resources/conversations
             stdapi/routes/openai_conversations.py:add_items
             stdapi/routes/openai_responses.py:_apply_conversation
        """
        log_start = len(agentic_server.logs)
        conversation = gateway_client.conversations.create()
        try:
            gateway_client.conversations.items.create(
                conversation.id,
                items=[_STORED_ITEM],  # type: ignore[list-item]
            )
            agent = _responses_agent(agentic_server, model_config)
            first = agent.run_sync(
                "Which reference number should you remember?",
                model_settings=OpenAIResponsesModelSettings(
                    openai_conversation_id=conversation.id
                ),
            )
            second = agent.run_sync(
                "State that same reference number again.",
                message_history=first.all_messages(),
                model_settings=OpenAIResponsesModelSettings(
                    openai_conversation_id="auto"
                ),
            )
        finally:
            gateway_client.conversations.delete(conversation.id)

        print(  # noqa: T201
            f"\n{TOOL.metrics_prefix} | {model_config.model:<30} | "
            f"test_a_stored_conversation_answers_a_turn_that_never_carried_the_fact "
            f"| conversation={conversation.id}"
        )
        assert _logged_requests(agentic_server, log_start, "POST", "/v1/conversations")
        assert _logged_requests(
            agentic_server,
            log_start,
            "POST",
            f"/v1/conversations/{conversation.id}/items",
        )
        assert _conversation_ids(first.all_messages()) == [conversation.id], (
            "the gateway did not report the conversation the first turn named"
        )
        assert _conversation_ids(second.all_messages())[-1:] == [conversation.id], (
            "openai_conversation_id='auto' did not chain on the stored conversation"
        )
        turns = _logged_requests(agentic_server, log_start, "POST", "/v1/responses")
        assert len(turns) == 2, turns
        for turn in turns:
            params = turn.get("request_params")
            assert isinstance(params, dict)
            assert params.get("conversation") == conversation.id, params
            assert _STORED_NUMBER not in str(params.get("input")), (
                "the client sent the stored fact itself, so the answer proves "
                f"nothing about the conversation: {params}"
            )
        assert _STORED_NUMBER in second.output, second.output


@pytest.mark.parametrize("model_config", [_RESPONSES_MODEL_CONFIG])
class TestNativeFileSearchTool:
    """``FileSearchTool.file_store_ids`` has to reach our wire as ``vector_store_ids``.

    The tool is pydantic-ai's own abstraction over four providers, so its field
    carries a provider-neutral name and the OpenAI model class renames it on the
    way out. A rename landing on the wrong key would leave the tool attached to no
    store at all, which the model answers anyway -- from its own knowledge -- so
    the logged request body is asserted alongside the answer.

    Ref: https://pydantic.dev/docs/ai/native-tools/#file-search-tool
         docs/api_openai_responses.md#file-search
         stdapi/models/chat/_adapters/_openai_responses.py:get_file_search_tool
    """

    def test_the_agent_answers_from_the_store_it_named(
        self, model_config: ModelConfig, agentic_server: AgenticServer, note_store: str
    ) -> None:
        """The retrieved reference number reaches the answer, through our own search.

        The gateway runs the retrieval itself and reports it as a
        ``file_search_call`` item; pydantic-ai parses that item with the
        ``openai`` package's own types into a native tool call carrying the
        queries, so an item the gateway shapes wrongly is dropped there rather
        than surfacing as an error.

        Ref: https://developers.openai.com/api/reference/resources/vector_stores
             stdapi/models/chat/_adapters/_openai_responses.py:execute_file_search_calls
        """
        log_start = len(agentic_server.logs)
        agent = _responses_agent(
            agentic_server,
            model_config,
            [NativeTool(FileSearchTool(file_store_ids=[note_store]))],
        )

        result = agent.run_sync(
            "Which reference number did the crew log for the mirror job?"
        )

        searches = _response_parts(result.all_messages(), NativeToolCallPart)
        print(  # noqa: T201
            f"\n{TOOL.metrics_prefix} | {model_config.model:<30} | "
            f"test_the_agent_answers_from_the_store_it_named "
            f"| file_search_calls={len(searches):>2}"
        )
        turns = _logged_requests(agentic_server, log_start, "POST", "/v1/responses")
        assert turns, "no request reached the Responses route"
        params = turns[0].get("request_params")
        assert isinstance(params, dict)
        assert [
            tool for tool in params.get("tools") or () if isinstance(tool, dict)
        ] == [{"type": "file_search", "vector_store_ids": [note_store]}], (
            f"file_store_ids did not arrive as vector_store_ids: {params.get('tools')}"
        )
        assert searches, (
            f"the gateway reported no file_search_call the client could read: "
            f"{result.all_messages()}"
        )
        assert all(part.tool_name == "file_search" for part in searches), searches
        assert PLANTED_NUMBER in result.output, result.output
