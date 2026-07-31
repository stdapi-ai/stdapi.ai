"""pydantic-ai driven against ``/v1/chat/completions``, focused on reasoning replay.

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
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ThinkingPart, ToolCallPart
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from ._runner import ModelConfig
from ._tools import AgenticTool

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def _response_parts[ResponsePart: (ThinkingPart, ToolCallPart)](
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
