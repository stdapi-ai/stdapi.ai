"""Claude Code driven end-to-end against the gateway's Anthropic Messages route.

Each test runs the real ``claude`` CLI in ``--print`` mode inside the agentic
container, pointed at a session-scoped stdapi.ai server, with one Bedrock model bound
to the ``sonnet`` slot. What this covers that no unit test can: a full agentic loop of
tool-use round trips -- the CLI's own system prompt, tool definitions, ``tool_use``
blocks and ``tool_result`` replays -- all translated by the gateway for a model that
is usually not a Claude model at all.

Each prompt asks for the smallest exploration that still produces what the test
asserts -- a bounded number of named files, and an answer whose vocabulary the
prompt itself withholds. Every extra file demanded is a full model round trip, on
ten models, that nothing checks; the wire-format translation this module exists
to prove is exercised by the first tool-use round trip and by every one after it
equally. The asserted step floor sits below even that, because the weaker models
legitimately differ in how many turns they take.

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

#: Ceiling for the ``slow``-marked models, which sit outside the ``--agentic`` budget.
#: Every other model here runs under the shared default, itself measured.
_SLOW_TIMEOUT = 1200

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
        # Declared flaky for this client only: the same model is not flaky in
        # test_codex.py or test_qwen_code.py, which drive it on its own
        # OpenAI-shaped routes. Here it has to speak the Anthropic tool protocol
        # through the gateway's translation with the client's thinking disabled,
        # which is where the small open-weight models in this list answer
        # inconsistently -- eight of the ten entries need the downgrade.
        ModelConfig(
            model="qwen.qwen3-coder-30b-a3b-v1:0", extra_env=_NO_THINKING, flaky=True
        ),
        id="qwen3-coder-30b",
    ),
    pytest.param(
        # Notably slower than its siblings, and outside the --agentic budget
        # anyway; give each run extra headroom.
        ModelConfig(
            model="qwen.qwen3-coder-next",
            extra_env=_NO_THINKING,
            timeout=_SLOW_TIMEOUT,
            flaky=True,
        ),
        id="qwen3-coder-next",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        # M2.5 reasons internally, so the CLI's own thinking budget is suppressed.
        ModelConfig(model="minimax.minimax-m2.5", extra_env=_NO_THINKING),
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
        # Mantle serves this one over Responses only, so reaching it from an
        # Anthropic Messages client is the widest conversion the gateway performs.
        ModelConfig(model="openai.gpt-5.6-luna", extra_env=_NO_THINKING, flaky=True),
        id="gpt-5.6-luna",
    ),
]

# ---------------------------------------------------------------------------
# Prompts — each is written to force multi-file exploration through the gateway.
# ---------------------------------------------------------------------------

_PROMPT_REQUEST_PIPELINE = f"""\
You are an AI coding assistant with access to the stdapi.ai source at {SRC_MOUNT}.

Your task: trace how an incoming POST /v1/chat/completions request reaches AWS
Bedrock. Read the source; do not answer from prior knowledge.

Read these files and no others:
  1. {SRC_MOUNT}/stdapi/routes/openai_chat_completions.py
  2. {SRC_MOUNT}/stdapi/models/chat/_adapters/_openai_chat_completion.py
  3. {SRC_MOUNT}/stdapi/models/chat/_default.py — grep it for the function that
     builds the Bedrock request payload, and read that function only

Report exactly three steps, in execution order, each with the real code quote:
  1. The handler registered for /v1/chat/completions
  2. The function that converts the request body into Bedrock's shape
  3. The function that assembles the Bedrock payload, and the name of the AWS
     Bedrock API operation the payload is finally sent to
"""

_PROMPT_STREAMING_PATH = f"""\
You are an AI coding assistant with access to the stdapi.ai source at {SRC_MOUNT}.

Your task: trace the streaming path of POST /v1/chat/completions with stream=True.
Read the source; do not guess, and quote the code you read.

Read these files and no others:
  1. {SRC_MOUNT}/stdapi/models/chat/_default.py — grep it for the branch taken
     when the request is streamed, and read around it only
  2. {SRC_MOUNT}/stdapi/models/chat/_adapters/_openai_chat_completion.py — find
     the function that turns the Bedrock stream into the client's own
     server-sent events, and read it

Report exactly three things, with the real code quote for each:
  1. The streaming branch, and the AWS Bedrock streaming call it makes
  2. The signature of the function that formats those server-sent events
  3. One raw Bedrock stream event name and the client chunk it becomes
"""

_PROMPT_PARAMETER_MAPPING = f"""\
You are an AI coding assistant with access to the stdapi.ai source at {SRC_MOUNT}.

Your task: map OpenAI chat completion parameters onto the AWS Bedrock Converse
request fields, as this gateway actually implements it.

Read one function: _prepare_converse_request in
{SRC_MOUNT}/stdapi/models/chat/_default.py. Read that function only, not the rest
of the file, and read no other file.

Report four of the mappings it performs, each on one line:
  • OpenAI parameter  →  Bedrock request field, with the exact line that assigns it

Do not report a mapping you did not see in that function.
"""

_PROMPT_MODEL_OVERRIDES = f"""\
You are an AI coding assistant with access to the stdapi.ai source at {SRC_MOUNT}.

Your task: report the model-specific behavior overrides in the chat module.

  1. List the Python files directly inside {SRC_MOUNT}/stdapi/models/chat/
     (use Glob or Bash)
  2. Pick three of the model-specific ones (not _default.py, __init__.py or
     _adapters/) and read those three. Read no others.

For each of the three, give its file name, the method signature it overrides, and
one sentence on what it changes. You are not finished until you have quoted real
code from three separate files — a list of file names is not an answer.
"""

#: Function names that only appear in the files the pipeline prompts force open.
#: None of them appears in the prompt, so naming one is evidence of a real read.
_PIPELINE_KEYWORDS = ("translate_request", "_prepare_converse_request")
#: Streaming vocabulary the prompt withholds, so only the source can supply it.
_STREAMING_KEYWORDS = (
    "converse_stream",
    "contentblockdelta",
    "messagestop",
    "format_stream",
)
#: Bedrock request fields ``_prepare_converse_request`` builds. Bedrock-side names
#: on purpose: the OpenAI-side ones are recitable without opening anything.
_PARAMETER_KEYWORDS = (
    "inferenceconfig",
    "toolconfig",
    "additionalmodelrequestfields",
    "requestmetadata",
)
#: Module names that only a listing of the chat package reveals.
_MODEL_FAMILY_KEYWORDS = (
    "amazon_nova",
    "anthropic_claude",
    "deepseek_v3",
    "kimi_k25",
    "mistral_7b",
    "openai_gpt",
    "twelvelabs_pegasus",
)

#: One-file task for the tests whose subject is a request parameter, not a workload.
#:
#: The functions listed below are the assertion: they exist only in that file, so
#: the answer cannot come from the model's own knowledge, and one read is all the
#: agent loop needs to prove the parameter was accepted end to end.
_PROMPT_READ_ONE_FILE = f"""\
Read {SRC_MOUNT}/stdapi/models/chat/_adapters/_common.py and reply with the name
of every function defined at module level in it, one per line.

Read nothing else, and answer nothing else.
"""

#: The first function of ``_common.py``, which every correct listing names.
_COMMON_FIRST_FUNCTION = "append_or_merge"

#: The rest of that file's module-level functions.
_COMMON_OTHER_FUNCTIONS = (
    "reject_unsupported_web_search_fields",
    "resolve_external_web_access",
    "inference_extras",
)


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

        The accepted keywords are the streaming identifiers the prompt withholds,
        so any one of them is evidence the source was read; the summary's wording
        is model-dependent, which is why any one of them suffices.

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
        """A parameter audit completes over at least two turns and names real Bedrock fields.

        The asserted vocabulary is the Bedrock side of the mapping, which appears
        only inside the function the prompt points at -- the OpenAI side would be
        recited correctly without opening anything.

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

        The longest task in the module -- a directory listing followed by three
        reads -- hence the higher step floor. The answer must name a module of
        that package, which only the listing reveals.

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
        """A one-file read completes at both low and high effort without changing model.

        What is under test is the flag, not the workload: a run that reaches the
        model at all has proven the effort value survived the gateway's
        translation, and the autouse identity check proves it was still this
        model. The task is therefore the smallest one that still needs a tool-use
        round trip -- and its answer names functions that exist in one file, so it
        cannot be produced without that round trip.

        Ref: stdapi/types/anthropic_messages.py:ThinkingEffort
        """
        if not model_config.supports_effort:
            pytest.skip(f"{model_config.model} does not support effort levels")
        result = run_agent(
            tool=TOOL,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_PROMPT_READ_ONE_FILE,
            workdir=agentic_workdir,
            test_name=request.node.originalname,
            effort=effort,
        )
        log_metrics(
            TOOL, result, model_config, f"test_effort_parameter_mapping[{effort}]"
        )
        assert_result(
            result,
            config=model_config,
            contains=_COMMON_FIRST_FUNCTION,
            any_of=_COMMON_OTHER_FUNCTIONS,
            min_steps=1,
        )
