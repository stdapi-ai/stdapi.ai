"""pi driven end-to-end against all three of the gateway's chat routes.

Claude Code and Codex each pin one wire format, so a failure they report is
ambiguous: adapter, model, or prompt. pi reaches Chat Completions, Responses and
Anthropic Messages through its own provider abstraction, so the same binary, the
same prompt and the same assertions run over all three. A failure on exactly one
route therefore isolates the fault to that adapter.

What this exercises that the other tools do not:

- ``/v1/chat/completions`` in a real agent loop. Claude Code drives Anthropic
  Messages and Codex drives Responses, so before this module the gateway's oldest
  and most-used route had no agentic coverage at all.
- DeepSeek's reasoning plumbing. ``reasoning_effort`` reaches Bedrock as a
  string-valued ``reasoning_config`` for DeepSeek alone, and the resulting
  reasoning text comes back in the non-standard ``reasoning_content`` field that
  ``stdapi/types/openai_chat_completions.py`` declares on three separate models.
- The ``CHAT_COMPLETIONS_REASONING_FIELD`` operator setting. It is read once
  into the settings singleton at process start, so proving it actually relocates
  (or drops) a live response's reasoning text needs its own gateway per value; a
  plain ``openai`` client reads back each of the three.

Requires ``--agentic``, podman, and Bedrock credentials.

Ref: https://pi.dev/docs/latest
     https://api-docs.deepseek.com/guides/reasoning_model
     stdapi/models/chat/deepseek_v3.py:ChatModel
     stdapi/models/chat/_adapters/_openai_chat_completion.py:translate_request
     stdapi/config.py:_Settings.chat_completions_reasoning_field
     stdapi/types/openai_chat_completions.py:_rename_emitted_reasoning
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pytest
from openai import OpenAI

from ._runner import ModelConfig, assert_result, log_metrics, run_agent
from ._server import start_server, stop_server
from ._tools import (
    PI_CHAT_COMPLETIONS,
    PI_MESSAGES,
    PI_RESPONSES,
    SRC_MOUNT,
    AgenticTool,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ._server import AgenticServer

pytestmark = pytest.mark.agentic

#: pi plans more tool calls than Claude Code for the same task.
_TIMEOUT = 1800

#: The three gateway routes pi speaks, one entry per wire format.
_TOOLS = [
    pytest.param(PI_CHAT_COMPLETIONS, id="chat-completions"),
    pytest.param(PI_RESPONSES, id="responses"),
    pytest.param(PI_MESSAGES, id="anthropic-messages"),
]

#: Models exercised on every route.
#:
#: Deliberately small: this module multiplies by three routes, so each entry
#: costs three runs per test. One Anthropic model and one Amazon model cover the
#: two native Converse dialects; DeepSeek covers the open-weight path and is the
#: only model whose reasoning knob is string-valued. Bedrock Mantle is reached
#: separately, by :class:`TestPiMantleCoverage`, which pairs each model with the
#: one route that serves it rather than crossing all three.
_MODEL_CONFIGS = [
    pytest.param(
        ModelConfig(model="anthropic.claude-haiku-4-5-20251001-v1:0", timeout=_TIMEOUT),
        id="claude-haiku-4-5",
    ),
    pytest.param(
        ModelConfig(model="amazon.nova-2-lite-v1:0", timeout=_TIMEOUT), id="nova-2-lite"
    ),
    pytest.param(
        ModelConfig(model="deepseek.v3.2", timeout=_TIMEOUT, supports_effort=True),
        id="deepseek-v3.2",
    ),
]

#: Reasoning-capable models, exercised additionally with an effort level.
_REASONING_MODEL_CONFIGS = [
    pytest.param(
        ModelConfig(model="deepseek.v3.2", timeout=_TIMEOUT, supports_effort=True),
        id="deepseek-v3.2",
    )
]


@pytest.fixture(params=_TOOLS)
def agentic_tool(request: pytest.FixtureRequest) -> AgenticTool:
    """The pi entry under test; read by the autouse model-identity fixture."""
    tool: AgenticTool = request.param
    return tool


# ---------------------------------------------------------------------------
# Prompts — each forces the agent to open files it cannot answer from memory.
# ---------------------------------------------------------------------------

_PROMPT_ADAPTER_LAYOUT = f"""\
You are working in the stdapi.ai source tree at {SRC_MOUNT}.

Identify every chat API adapter the gateway implements and what each one
translates between.

Use shell commands to read the actual source. Do not guess:
  1. List every file in {SRC_MOUNT}/stdapi/models/chat/_adapters/
  2. Read the module docstring of each adapter and quote it
  3. For each adapter, name the client API it accepts and quote the exact
     signature of the function that converts a request into Bedrock's shape

Report one section per adapter, with real code quotes.
"""

_PROMPT_TOOL_ROUND_TRIP = f"""\
You are working in the stdapi.ai source tree at {SRC_MOUNT}.

Explain how a tool call made by the model travels back to the gateway and is
replayed to Bedrock on the following turn.

Use shell commands to read source files. Quote actual code, never guess:
  1. Find where a tool result from the client is converted into a Bedrock
     toolResult block — quote the function and its signature
  2. Find where an assistant tool call is converted into a Bedrock toolUse
     block — quote it
  3. Explain what happens when two consecutive messages map to the same
     Bedrock role, and quote the code that handles it

Read at least three distinct files and quote code from each.
"""

#: Vocabulary that appears only in the adapter files the prompt forces open.
_ADAPTER_KEYWORDS = (
    "_openai_chat_completion",
    "_openai_responses",
    "_anthropic_message",
    "translate_request",
    "adapter",
)

#: Vocabulary any correct account of the tool round trip uses.
_TOOL_ROUND_TRIP_KEYWORDS = (
    "tooluse",
    "toolresult",
    "append_or_merge",
    "tool_call",
    "bedrock",
)


@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
class TestPiAcrossRoutes:
    """The same agent task over Chat Completions, Responses and Anthropic Messages.

    Every assertion here is route-agnostic on purpose: the three parametrizations
    differ only in which adapter the gateway runs, so a single-route failure is a
    single-adapter failure.

    Ref: https://pi.dev/docs/latest
         stdapi/models/chat/_adapters/
    """

    def test_enumerate_adapters(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """Pi enumerates the adapters after reading them, on every route.

        The answer names files the prompt does not, so it can only come from tool
        output carried back through the gateway's own streaming translation.

        Ref: stdapi/models/chat/_adapters/__init__.py
        """
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_PROMPT_ADAPTER_LAYOUT,
            workdir=agentic_workdir,
            test_name=f"{request.node.originalname}[{agentic_tool.id}]",
        )
        log_metrics(agentic_tool, result, model_config, "test_enumerate_adapters")
        assert_result(
            result,
            config=model_config,
            contains="adapter",
            any_of=_ADAPTER_KEYWORDS,
            min_steps=2,
        )

    def test_trace_tool_round_trip(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """Pi explains the tool round trip, having performed one to find it.

        The task is self-referential: describing how tool results are replayed
        requires several replayed tool results, so a broken multi-turn tool
        mapping on any route fails the run rather than degrading the answer.

        Ref: stdapi/models/chat/_adapters/_common.py:append_or_merge
        """
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_PROMPT_TOOL_ROUND_TRIP,
            workdir=agentic_workdir,
            test_name=f"{request.node.originalname}[{agentic_tool.id}]",
        )
        log_metrics(agentic_tool, result, model_config, "test_trace_tool_round_trip")
        assert_result(
            result,
            config=model_config,
            contains="bedrock",
            any_of=_TOOL_ROUND_TRIP_KEYWORDS,
            min_steps=2,
        )


#: One Bedrock Mantle model per route rather than a cross product.
#:
#: The probe corpus (tests/probes/results/) covers Chat Completions only, and on
#: that route qwen.qwen3-next-80b-a3b-instruct answers while openai.gpt-5.6-luna
#: and google.gemma-4-31b refuse outright -- so each of the latter two is paired
#: with the route Mantle does serve it on. Qwen3-Next rather than Qwen3-32B
#: because an agent loop does not fit the latter's 32K context.
_MANTLE_ROUTE_MODELS = [
    pytest.param(
        PI_CHAT_COMPLETIONS,
        ModelConfig(model="qwen.qwen3-next-80b-a3b-instruct", timeout=_TIMEOUT),
        id="qwen3-next-80b-chat-completions",
    ),
    pytest.param(
        PI_RESPONSES,
        ModelConfig(model="openai.gpt-5.6-luna", timeout=_TIMEOUT),
        id="gpt-5.6-luna-responses",
    ),
    pytest.param(
        PI_MESSAGES,
        # Measured at ~45 min for two runs, and Mantle answers it with an
        # upstream 500 often enough to matter -- the same reason
        # `test_claude_code.py` already carries this model as flaky. Both are
        # upstream conditions: the conversion itself completes its tool loop.
        ModelConfig(model="google.gemma-4-31b", timeout=3600, flaky=True),
        id="gemma-4-31b-messages",
    ),
]


@pytest.mark.parametrize(("agentic_tool", "model_config"), _MANTLE_ROUTE_MODELS)
class TestPiMantleCoverage:
    """The same pi task, run once per route against the Mantle model that serves it.

    Bedrock Mantle rejects each of these models on the other two routes, so unlike
    :class:`TestPiAcrossRoutes` -- which crosses every model with every route --
    this class pairs exactly one model with the single route it accepts.

    Ref: stdapi/models/chat/_adapters/
    """

    def test_enumerate_adapters(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """Pi enumerates the adapters after reading them, over Mantle.

        Ref: stdapi/models/chat/_adapters/__init__.py
        """
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_PROMPT_ADAPTER_LAYOUT,
            workdir=agentic_workdir,
            test_name=f"{request.node.originalname}[{agentic_tool.id}]",
        )
        log_metrics(agentic_tool, result, model_config, "test_enumerate_adapters")
        assert_result(
            result,
            config=model_config,
            contains="adapter",
            any_of=_ADAPTER_KEYWORDS,
            min_steps=2,
        )

    def test_trace_tool_round_trip(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """Pi explains the tool round trip, having performed one to find it, over Mantle.

        Ref: stdapi/models/chat/_adapters/_common.py:append_or_merge
        """
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_PROMPT_TOOL_ROUND_TRIP,
            workdir=agentic_workdir,
            test_name=f"{request.node.originalname}[{agentic_tool.id}]",
        )
        log_metrics(agentic_tool, result, model_config, "test_trace_tool_round_trip")
        assert_result(
            result,
            config=model_config,
            contains="bedrock",
            any_of=_TOOL_ROUND_TRIP_KEYWORDS,
            min_steps=2,
        )


@pytest.mark.parametrize("model_config", _REASONING_MODEL_CONFIGS)
@pytest.mark.parametrize("effort", ["low", "high"])
class TestPiReasoning:
    """DeepSeek's reasoning knob driven through a real agent loop.

    DeepSeek is the only family whose reasoning budget is expressed as a string
    ``reasoning_config`` rather than a token count, and its reasoning text returns
    in ``reasoning_content`` rather than in the content blocks. Both are
    gateway-specific translations that unit tests can only assert in isolation.

    Ref: https://api-docs.deepseek.com/guides/reasoning_model
         stdapi/models/chat/deepseek_v3.py:_req_configure_reasoning
    """

    def test_effort_level_completes(
        self,
        request: pytest.FixtureRequest,
        effort: str,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """Each effort level still yields a complete, file-derived answer.

        A reasoning payload the gateway mistranslates surfaces here as a failed
        run rather than as a silently ignored parameter: the model either never
        starts, or returns reasoning text the client cannot separate from the
        answer.

        Ref: stdapi/models/chat/deepseek_v3.py:_REASONING_OVERRIDE
        """
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_PROMPT_ADAPTER_LAYOUT,
            workdir=agentic_workdir,
            test_name=f"{request.node.originalname}[{agentic_tool.id}-{effort}]",
            effort=effort,
        )
        log_metrics(
            agentic_tool,
            result,
            model_config,
            f"test_effort_level_completes[{agentic_tool.id}-{effort}]",
        )
        assert_result(
            result,
            config=model_config,
            contains="adapter",
            any_of=_ADAPTER_KEYWORDS,
            min_steps=2,
        )


#: Model whose reasoning text the field setting relocates or drops; the only
#: model in this module whose reasoning comes back as ``reasoning_content``.
_REASONING_FIELD_MODEL = "deepseek.v3.2"

#: Effort level this model actually reasons at.
#:
#: It is a hybrid: ``tests/probes/results/deepseek.v3.2.json`` records
#: ``reasoning_effort_low`` as accepted with "no observable effect" and only
#: ``reasoning_effort_high`` as producing a reasoningContent block. Asking for
#: less leaves nothing for the setting under test to relocate, so every value
#: would pass without proving anything.
_REASONING_FIELD_EFFORT: Literal["high"] = "high"

#: Values the operator setting accepts, each proven against a live response.
_REASONING_FIELD_VALUES = ["reasoning_content", "reasoning", "none"]

#: Environment variable the gateway reads the reasoning-field setting from.
_REASONING_FIELD_VAR = "CHAT_COMPLETIONS_REASONING_FIELD"


@dataclass(frozen=True)
class _ReasoningFieldServer:
    """A gateway started with one value of ``CHAT_COMPLETIONS_REASONING_FIELD``.

    Attributes:
        server: The running gateway.
        field: The value its environment was started with.
    """

    server: AgenticServer
    field: str


@pytest.fixture(
    params=[pytest.param(value, id=value) for value in _REASONING_FIELD_VALUES],
    scope="module",
)
def reasoning_field_server(
    request: pytest.FixtureRequest,
) -> Iterator[_ReasoningFieldServer]:
    """A dedicated gateway, restarted once per reasoning-field setting.

    The setting is read once into the settings singleton when the process
    starts, so it cannot be varied per request on the shared, session-scoped
    ``agentic_server`` -- each value needs its own gateway.

    Yields:
        The server, paired with the value its environment was started with.
    """
    field = request.param
    # The child inherits os.environ, so the setting is put there just long
    # enough to launch, then whatever the operator had is put back.
    previous = os.environ.get(_REASONING_FIELD_VAR)
    os.environ[_REASONING_FIELD_VAR] = field
    try:
        server = start_server()
    finally:
        if previous is None:
            del os.environ[_REASONING_FIELD_VAR]
        else:
            os.environ[_REASONING_FIELD_VAR] = previous
    try:
        yield _ReasoningFieldServer(server=server, field=field)
    finally:
        stop_server(server)


class TestChatCompletionsReasoningField:
    """The ``CHAT_COMPLETIONS_REASONING_FIELD`` operator setting, live end to end.

    The Qwen Code suite was meant to carry this coverage but was never written.
    pi's ``/v1/chat/completions`` route is the closest existing driver of that
    route, and DeepSeek is the model whose reasoning text this setting relocates.
    A plain ``openai`` client reads the response back directly -- pi itself
    declares this model non-reasoning and never looks at the field.

    Ref: stdapi/config.py:_Settings.chat_completions_reasoning_field
         stdapi/types/openai_chat_completions.py:_rename_emitted_reasoning
    """

    def test_reasoning_text_lands_in_the_configured_field(
        self, reasoning_field_server: _ReasoningFieldServer
    ) -> None:
        """A real client reads DeepSeek's reasoning text from the configured field.

        ``reasoning_content`` and ``reasoning`` are mutually exclusive: the
        response must carry the text under the field this gateway was started
        with, under no other field, and under neither once the setting is
        ``"none"``. The effort asked for is the one this model reasons at, so
        the ``"none"`` case proves suppression rather than absence.

        Ref: https://developers.openai.com/api/docs/guides/reasoning
             tests/probes/results/deepseek.v3.2.json
             stdapi/models/chat/deepseek_v3.py:ChatModel._req_configure_reasoning
        """
        server, field = reasoning_field_server.server, reasoning_field_server.field
        client = OpenAI(
            base_url=server.url("/v1"), api_key=server.api_key, max_retries=0
        )
        response = client.chat.completions.create(
            model=_REASONING_FIELD_MODEL,
            messages=[
                {"role": "user", "content": "What is 6 times 7? Reply with the number."}
            ],
            reasoning_effort=_REASONING_FIELD_EFFORT,
        )
        message = response.choices[0].message
        assert message.content, f"no answer content came back: {response!r}"
        extra = message.model_extra or {}
        reasoning_fields = {"reasoning_content", "reasoning"} & extra.keys()
        if field == "none":
            assert not reasoning_fields, (
                f"reasoning text leaked under {reasoning_fields}: {extra!r}"
            )
        else:
            assert reasoning_fields == {field}, (
                f"reasoning text under {reasoning_fields}, expected only "
                f"{field!r}: {extra!r}"
            )
            assert extra[field], f"empty reasoning text under {field!r}"
