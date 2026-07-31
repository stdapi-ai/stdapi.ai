"""Claude Code driven end-to-end against the gateway's Anthropic Messages route.

Each test runs the real ``claude`` CLI in ``--print`` mode inside the agentic
container, pointed at a session-scoped stdapi.ai server, with one Bedrock model bound
to the ``sonnet`` slot. What this covers that no unit test can: a full agentic loop of
tool-use round trips -- the CLI's own system prompt, tool definitions, ``tool_use``
blocks and ``tool_result`` replays -- all translated by the gateway for a model that
is usually not a Claude model at all.

The prompts are written to need roughly ten tool calls across several files. The
asserted step floor is far lower on purpose: it only rejects answers produced without
opening the source, because the weaker models legitimately differ in how many turns
they take.

Requires ``--agentic``, podman, and Bedrock credentials.

Ref: https://platform.claude.com/docs/en/api/messages
     https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
     stdapi/routes/anthropic_messages.py:create_message
     stdapi/models/chat/_adapters/_anthropic_message.py:translate_request
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ._runner import ModelConfig, assert_result, log_metrics, run_agent
from ._tools import CLAUDE_CODE, SRC_MOUNT

if TYPE_CHECKING:
    from pathlib import Path

    from ._server import AgenticServer

#: Tool this module drives; read by the autouse model-identity fixture.
TOOL = CLAUDE_CODE

pytestmark = pytest.mark.agentic

#: Env disabling the CLI's own thinking budget for models that reason internally.
_NO_THINKING = {"DISABLE_PROMPT_CACHING": "1", "MAX_THINKING_TOKENS": "0"}

_MODEL_CONFIGS = [
    # Reference baseline: the only model here that is natively Anthropic.
    pytest.param(
        ModelConfig(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            extra_env={
                "ANTHROPIC_DEFAULT_SONNET_MODEL_SUPPORTED_CAPABILITIES": (
                    "effort,thinking,adaptive_thinking,interleaved_thinking"
                )
            },
            supports_effort=True,
        ),
        id="claude-haiku-4-5",
    ),
    pytest.param(
        ModelConfig(
            model="amazon.nova-2-lite-v1:0",
            extra_env={
                "ANTHROPIC_DEFAULT_SONNET_MODEL_SUPPORTED_CAPABILITIES": "effort",
                "MAX_THINKING_TOKENS": "0",
            },
            supports_effort=True,
            flaky=True,
        ),
        id="nova-2-lite",
    ),
    pytest.param(
        ModelConfig(model="moonshotai.kimi-k2.5", extra_env=_NO_THINKING, flaky=True),
        id="kimi-k2.5",
    ),
    pytest.param(
        ModelConfig(
            model="qwen.qwen3-coder-30b-a3b-v1:0", extra_env=_NO_THINKING, flaky=True
        ),
        id="qwen3-coder-30b",
    ),
    pytest.param(
        # Notably slower than its siblings; give each run extra headroom.
        ModelConfig(
            model="qwen.qwen3-coder-next",
            extra_env=_NO_THINKING,
            timeout=2400,
            flaky=True,
        ),
        id="qwen3-coder-next",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        # M2.5 reasons internally, so the CLI's own thinking budget is suppressed.
        # That internal reasoning also makes it the slowest model here, by enough
        # of a margin to need a ceiling of its own under a parallel run.
        ModelConfig(model="minimax.minimax-m2.5", extra_env=_NO_THINKING, timeout=2400),
        id="minimax-m2.5",
    ),
    pytest.param(
        ModelConfig(
            model="mistral.devstral-2-123b", extra_env=_NO_THINKING, flaky=True
        ),
        id="devstral-2",
    ),
    pytest.param(
        ModelConfig(model="zai.glm-5", extra_env=_NO_THINKING, flaky=True), id="glm-5"
    ),
    pytest.param(
        # Mantle-served: exercises the Anthropic-messages to OpenAI conversion path.
        ModelConfig(model="google.gemma-4-31b", extra_env=_NO_THINKING, flaky=True),
        id="gemma-4-31b",
    ),
    pytest.param(
        ModelConfig(model="xai.grok-4.3", extra_env=_NO_THINKING, flaky=True),
        id="grok-4.3",
    ),
]

# ---------------------------------------------------------------------------
# Prompts — each is written to force multi-file exploration through the gateway.
# ---------------------------------------------------------------------------

_PROMPT_REQUEST_PIPELINE = f"""\
You are an AI coding assistant with access to the stdapi.ai source at {SRC_MOUNT}.

Your task: Trace the COMPLETE execution path for an incoming POST /v1/chat/completions
HTTP request from the route handler through to the actual AWS Bedrock Converse API call.

You MUST read the actual source files and quote real code — do not rely on prior
knowledge.  For each step in the call chain you MUST:
  - Open and read the file containing that step
  - Copy the EXACT function signature (name + all parameters) as it appears in the code
  - State the file path and describe what the function does in one sentence

Explore and read at minimum these layers:
  1. The file that registers the /v1/chat/completions route
  2. The request handler it calls
  3. The translate_request adapter (read the file, quote its return type annotation)
  4. The _prepare_converse_request function (read its full signature + all parameters)
  5. The line where converse() or converse_stream() is finally called

List the steps in execution order with real code quotes.  Show at least 6 steps.
"""

_PROMPT_STREAMING_PATH = f"""\
You are an AI coding assistant with access to the stdapi.ai source at {SRC_MOUNT}.

Your task: Trace the streaming code path for POST /v1/chat/completions with stream=True.

You MUST open and read the relevant source files.  Do not guess — quote actual code.
For each stage of the pipeline, copy the relevant code snippet (a few lines) and
explain its role.

Investigate and document all of the following:
  1. The exact condition in _default.py that branches on stream=True — quote it
  2. The call to converse_stream (quote the call site and its arguments)
  3. The SSE event adapter/generator class(es) — find and read those files too
  4. How each raw Bedrock event type (contentBlockDelta, messageStop, etc.) maps to
     an OpenAI SSE chunk type — quote the mapping code
  5. The final streaming response construction

Read at least 3 distinct source files and quote code from each one.
"""

_PROMPT_PARAMETER_MAPPING = f"""\
You are an AI coding assistant with access to the stdapi.ai source at {SRC_MOUNT}.

Your task: Produce a precise, code-backed mapping of OpenAI chat completions API
parameters to AWS Bedrock Converse API fields as implemented in stdapi.ai.

You MUST open and read each of these files before answering:
  1. The OpenAI types file — find CompletionCreateParams and read ALL its fields
  2. The translate_request function in the adapter — read its full body
  3. {SRC_MOUNT}/stdapi/models/chat/_default.py — read _prepare_converse_request in full

For each parameter you find, quote the EXACT line(s) of code that handle it and state:
  • OpenAI parameter name  →  Bedrock Converse field name  (quote the assignment)

Document at least 10 parameter mappings with real code quotes.  Do not skip any.
Include: temperature, max_tokens, top_p, stop_sequences, stream, system messages,
tools/toolConfig, metadata/requestMetadata, and any others you find.
"""

_PROMPT_MODEL_OVERRIDES = f"""\
You are an AI coding assistant with access to the stdapi.ai source at {SRC_MOUNT}.

Your task: Find ALL model-specific behavior override files in the chat completions
module and document the custom logic each one implements.

You MUST:
  1. List ALL Python files inside {SRC_MOUNT}/stdapi/models/chat/ (use Glob or Bash)
  2. Read _default.py briefly to understand what the base class provides
  3. For each model-specific file (not _default.py, __init__.py, or _adapters/):
     - Read the file
     - State which Bedrock model family it targets
     - Quote the method signature(s) it overrides
     - Explain in 1-2 sentences what custom behavior it adds vs the default

Document at least 5 model-specific files with real code quotes.
"""

#: Function names that only appear in the files the pipeline prompts force open.
_PIPELINE_KEYWORDS = ("translate_request", "_prepare_converse_request", "_default")
#: Vocabulary any correct summary of the streaming path uses.
_STREAMING_KEYWORDS = ("stream", "sse", "generator", "event", "converse_stream")
#: Parameter names that only appear in the mapped source files.
_PARAMETER_KEYWORDS = ("temperature", "max_tokens", "inferenceconfig", "messages")
#: Model families named only inside the per-model override files.
_MODEL_FAMILY_KEYWORDS = ("nova", "claude", "deepseek", "mistral", "llama", "qwen")


@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
class TestClaudeCodePipeline:
    """Claude Code drives multi-file exploration turns through the Messages route.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/routes/anthropic_messages.py:create_message
    """

    def test_trace_request_pipeline(
        self,
        request: pytest.FixtureRequest,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """A multi-file source trace completes over at least two turns and names ``converse``.

        Naming ``converse`` plus one of the pipeline's real function names is the
        evidence that the tool-use round trips actually carried file contents back
        through the gateway; neither string appears in the prompt.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             stdapi/models/chat/_adapters/_anthropic_message.py:format_response
        """
        result = run_agent(
            tool=TOOL,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_PROMPT_REQUEST_PIPELINE,
            workdir=agentic_workdir,
            test_name=request.node.originalname,
        )
        log_metrics(TOOL, result, model_config, "test_trace_request_pipeline")
        assert_result(
            result,
            config=model_config,
            contains="converse",
            any_of=_PIPELINE_KEYWORDS,
            min_steps=2,
        )

    def test_trace_streaming_path(
        self,
        request: pytest.FixtureRequest,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """A streaming-path trace completes over at least two turns and reports SSE vocabulary.

        The keyword set is deliberately broad because the wording of the summary is
        model-dependent; the load-bearing assertions are the step floor and the
        autouse model-identity check.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:format_stream
        """
        result = run_agent(
            tool=TOOL,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_PROMPT_STREAMING_PATH,
            workdir=agentic_workdir,
            test_name=request.node.originalname,
        )
        log_metrics(TOOL, result, model_config, "test_trace_streaming_path")
        assert_result(
            result, config=model_config, any_of=_STREAMING_KEYWORDS, min_steps=2
        )


@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
class TestClaudeCodeAnalysis:
    """Claude Code sustains longer multi-file analysis sessions through the gateway.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/routes/anthropic_messages.py:create_message
    """

    def test_audit_parameter_mapping(
        self,
        request: pytest.FixtureRequest,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """A three-file parameter audit completes over at least two turns and names real parameters.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/models/chat/_adapters/_anthropic_message.py:translate_request
        """
        result = run_agent(
            tool=TOOL,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_PROMPT_PARAMETER_MAPPING,
            workdir=agentic_workdir,
            test_name=request.node.originalname,
        )
        log_metrics(TOOL, result, model_config, "test_audit_parameter_mapping")
        assert_result(
            result, config=model_config, any_of=_PARAMETER_KEYWORDS, min_steps=2
        )

    def test_enumerate_model_overrides(
        self,
        request: pytest.FixtureRequest,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """A Glob-then-read enumeration of the model override files completes over four turns.

        The longest task in the module -- a directory listing followed by five or
        more reads -- hence the higher step floor.

        Ref: stdapi/models/chat/__init__.py:get_chat_model
        """
        result = run_agent(
            tool=TOOL,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_PROMPT_MODEL_OVERRIDES,
            workdir=agentic_workdir,
            test_name=request.node.originalname,
        )
        log_metrics(TOOL, result, model_config, "test_enumerate_model_overrides")
        assert_result(
            result, config=model_config, any_of=_MODEL_FAMILY_KEYWORDS, min_steps=4
        )


@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
class TestClaudeCodeEffortLevels:
    """Claude Code's ``--effort`` levels are accepted end-to-end by the gateway.

    Only models advertising the ``effort`` capability through
    ``ANTHROPIC_DEFAULT_SONNET_MODEL_SUPPORTED_CAPABILITIES`` are exercised.

    Ref: stdapi/types/anthropic_messages.py:ThinkingEffort
         stdapi/models/chat/_adapters/_anthropic_message.py:translate_request
    """

    @pytest.mark.parametrize("effort", ["low", "high"])
    def test_effort_parameter_mapping(
        self,
        request: pytest.FixtureRequest,
        model_config: ModelConfig,
        effort: str,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """The parameter audit completes at both low and high effort without changing model.

        Reuses the analysis task so the metric lines stay comparable across effort
        levels. The effort value is asserted to be accepted end-to-end, not to be
        forwarded to Bedrock as any particular field.

        Ref: stdapi/types/anthropic_messages.py:ThinkingEffort
        """
        if not model_config.supports_effort:
            pytest.skip(f"{model_config.model} does not support effort levels")
        result = run_agent(
            tool=TOOL,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_PROMPT_PARAMETER_MAPPING,
            workdir=agentic_workdir,
            test_name=request.node.originalname,
            effort=effort,
        )
        log_metrics(
            TOOL, result, model_config, f"test_effort_parameter_mapping[{effort}]"
        )
        assert_result(
            result, config=model_config, any_of=_PARAMETER_KEYWORDS, min_steps=2
        )
