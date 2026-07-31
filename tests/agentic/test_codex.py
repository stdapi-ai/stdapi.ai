"""Codex driven end-to-end against the gateway's OpenAI Responses route.

Each test runs the real ``codex exec --json`` CLI inside the agentic container against
a session-scoped stdapi.ai server, with one Bedrock model per family. What this
exercises that no unit test does: a roughly 7600-token ``instructions`` field,
``developer``-role messages in ``input``, multi-turn ``function_call`` items replayed
as input alongside their ``function_call_output`` results, and real SSE streaming --
the gateway's own stream must stay parseable by a third-party client for a turn to
complete at all.

Requires ``--agentic``, podman, and Bedrock credentials.

Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
     https://developers.openai.com/api/docs/guides/function-calling
     stdapi/routes/openai_responses.py:create_response
     stdapi/models/chat/_adapters/_openai_responses.py:_map_function_call
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ._runner import (
    ModelConfig,
    assert_result,
    grounding_requests,
    log_metrics,
    run_agent,
)
from ._tools import CODEX, SRC_MOUNT

if TYPE_CHECKING:
    from pathlib import Path

    from ._server import AgenticServer

#: Tool this module drives; read by the autouse model-identity fixture.
TOOL = CODEX

pytestmark = pytest.mark.agentic

#: Codex runs longer than Claude Code on the same task; it plans more shell calls.
#: Deliberately generous: a ceiling costs nothing when it is not reached, whereas a
#: run killed part-way is indistinguishable from a gateway fault.
_TIMEOUT = 1800

_MODEL_CONFIGS = [
    pytest.param(
        ModelConfig(model="anthropic.claude-haiku-4-5-20251001-v1:0", timeout=_TIMEOUT),
        id="claude-haiku-4-5",
    ),
    pytest.param(
        # Reaches the answer through many more shell round trips than the larger
        # models: the parameter-mapping prompt points it at the largest file in
        # the tree, and it reads it in small chunks. That takes about seven
        # minutes alone and over twenty when it shares the machine, so it gets
        # its own ceiling rather than a downgraded timeout signature.
        ModelConfig(model="amazon.nova-2-lite-v1:0", timeout=3600),
        id="nova-2-lite",
    ),
    pytest.param(
        ModelConfig(model="moonshotai.kimi-k2.5", timeout=_TIMEOUT, flaky=True),
        id="kimi-k2.5",
    ),
    pytest.param(
        ModelConfig(model="qwen.qwen3-coder-30b-a3b-v1:0", timeout=_TIMEOUT),
        id="qwen3-coder-30b",
    ),
    pytest.param(
        # Notably slower than its siblings; give each run extra headroom.
        ModelConfig(model="qwen.qwen3-coder-next", timeout=3600, flaky=True),
        id="qwen3-coder-next",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        ModelConfig(model="minimax.minimax-m2.5", timeout=_TIMEOUT), id="minimax-m2.5"
    ),
    pytest.param(
        ModelConfig(model="mistral.devstral-2-123b", timeout=_TIMEOUT, flaky=True),
        id="devstral-2",
    ),
    pytest.param(
        ModelConfig(model="zai.glm-5", timeout=_TIMEOUT, flaky=True), id="glm-5"
    ),
    pytest.param(
        # Responses-only reasoning model served natively by Bedrock Mantle.
        ModelConfig(model="openai.gpt-5.6-luna", timeout=_TIMEOUT),
        id="gpt-5.6-luna",
    ),
    pytest.param(
        ModelConfig(model="xai.grok-4.3", timeout=_TIMEOUT, flaky=True), id="grok-4.3"
    ),
]

# ---------------------------------------------------------------------------
# Prompts — each forces shell exploration, driving the function_call cycle.
# ---------------------------------------------------------------------------

_PROMPT_REQUEST_PIPELINE = f"""\
You are working in the stdapi.ai source tree at {SRC_MOUNT}.

Trace the COMPLETE execution path for an incoming POST /v1/responses HTTP request
from the route handler to the AWS Bedrock Converse API call.

Use shell commands to read the actual source files. For each step in the call chain:
  - Read the file and quote the EXACT function signature
  - State the file path and describe what the function does in one sentence

Explore at minimum these layers:
  1. The file that registers the /v1/responses route
  2. The create_response handler it calls
  3. The translate_request adapter function
  4. The map_input function
  5. The final Bedrock converse or converse_stream call

List the steps in execution order. Show at least 5 steps with real code quotes.
"""

_PROMPT_STREAMING_PATH = f"""\
You are working in the stdapi.ai source tree at {SRC_MOUNT}.

Trace the streaming code path for POST /v1/responses with stream=True.

Use shell commands to read source files. Do not guess — quote actual code.

Investigate and document:
  1. Where create_response branches on stream=True — quote the condition
  2. The SSE event adapter/generator that formats streaming output
  3. How Bedrock stream events (contentBlockDelta, messageStop) map to
     OpenAI SSE event types — quote the mapping code
  4. The format_stream function — read it and explain its structure

Read at least 3 distinct source files and quote code from each.
"""

_PROMPT_PARAMETER_MAPPING = f"""\
You are working in the stdapi.ai source tree at {SRC_MOUNT}.

Produce a precise, code-backed mapping of Responses API parameters to AWS Bedrock
Converse API fields as implemented in stdapi.ai.

Use shell commands to read:
  1. {SRC_MOUNT}/stdapi/types/openai_responses.py — find ResponseCreateParams fields
  2. The translate_request function in the adapter — read its full body
  3. The map_input function — read how input items map to Bedrock messages

For each parameter you find, quote the EXACT line(s) of code that handle it and state:
  OpenAI param name  →  Bedrock field name  (with code quote)

Document at least 6 parameter mappings. Include: model, instructions, input,
temperature/inferenceConfig, tools/toolConfig, and stream handling.
"""

_PROMPT_MODEL_OVERRIDES = f"""\
You are working in the stdapi.ai source tree at {SRC_MOUNT}.

Find ALL model-specific behavior override files in the chat module and document
the custom logic each one implements.

You MUST:
  1. List ALL Python files inside {SRC_MOUNT}/stdapi/models/chat/ using shell commands
  2. Read _default.py briefly to understand what the base class provides
  3. For each model-specific file:
     - Read the file
     - State which model family it targets
     - Quote at least one overridden method signature
     - Explain in 1 sentence what custom behavior it adds

Document at least 4 model-specific files with real code quotes.
"""

#: Gateway function names that appear only in the files the prompt forces open.
_PIPELINE_KEYWORDS = ("create_response", "translate_request", "map_input", "_default")
#: Vocabulary any correct summary of the streaming path uses.
_STREAMING_KEYWORDS = ("stream", "sse", "event", "format_stream", "converse_stream")
#: Parameter names that appear only in the mapped source files.
_PARAMETER_KEYWORDS = (
    "instructions",
    "inferenceconfig",
    "temperature",
    "toolconfig",
    "messages",
)
#: Model families and override vocabulary found only in the per-model files.
_MODEL_FAMILY_KEYWORDS = (
    "nova",
    "claude",
    "deepseek",
    "mistral",
    "llama",
    "qwen",
    "amazon",
    "moonshot",
    "writer",
    "_default",
    "override",
    "model-specific",
)


@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
class TestCodexPipeline:
    """Codex traces multi-file execution paths via /v1/responses with shell tools.

    Ref: https://developers.openai.com/api/docs/guides/streaming-responses
         stdapi/models/chat/_adapters/_openai_responses.py:format_stream
    """

    def test_trace_request_pipeline(
        self,
        request: pytest.FixtureRequest,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """Codex completes a route-to-Bedrock trace over at least two shell calls.

        The task cannot be answered from the prompt alone, so the run only succeeds
        if the ``function_call`` / ``function_call_output`` round trip survives
        across turns.

        Ref: https://developers.openai.com/api/docs/guides/function-calling
             stdapi/models/chat/_adapters/_openai_responses.py:_map_function_call_output
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
        """Codex reports on the SSE code path after at least two shell calls.

        The run itself is streamed, so the gateway's own SSE output has to stay
        parseable by the Codex client for the turn to complete.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/models/chat/_adapters/_openai_responses.py:format_stream
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
class TestCodexAnalysis:
    """Codex performs multi-file analysis tasks via /v1/responses with shell tools.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/types/openai_responses.py:ResponseCreateParams
    """

    def test_audit_parameter_mapping(
        self,
        request: pytest.FixtureRequest,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """Codex audits the Responses-to-Converse parameter mapping over two shell calls.

        Ref: https://developers.openai.com/api/docs/guides/migrate-to-responses
             stdapi/models/chat/_adapters/_openai_responses.py:translate_request
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
        """Codex enumerates the model-specific chat overrides over at least three shell calls.

        The highest step floor in this module: listing the directory then reading
        several files keeps the ``function_call`` cycle running long enough to
        exercise a steadily growing input array.

        Ref: https://developers.openai.com/api/docs/guides/function-calling
             stdapi/models/chat/__init__.py:get_chat_model
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
            result, config=model_config, any_of=_MODEL_FAMILY_KEYWORDS, min_steps=3
        )


#: Nova 2 Lite with Codex's built-in web search turned back on for one test.
_WEB_SEARCH_CONFIG = ModelConfig(
    model="amazon.nova-2-lite-v1:0",
    timeout=_TIMEOUT,
    extra_args=("-c", 'web_search="live"'),
)

_PROMPT_WEB_SEARCH = """\
Search the web for the current stable version number of the Linux kernel.

Answer with the version number and the URL you took it from, nothing else.
Do not guess from memory and do not run any shell command.
"""


class TestCodexWebSearch:
    """Codex's built-in web search, served by Amazon Nova's grounding tool.

    Codex declares its own ``web_search`` tool rather than a function the harness
    supplies, so this is the only test covering the gateway's translation of a
    third-party client's hosted-tool declaration. The rest of the module turns the
    tool off: it is answered with a ``400`` on any model that has no grounding
    behind it, which every other entry in ``_MODEL_CONFIGS`` is.

    Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
         https://developers.openai.com/codex/reference/settings
         stdapi/models/chat/_default.py:ChatModel._canonical_name_for
    """

    @pytest.mark.expensive
    def test_web_search_reaches_the_grounding_tool(
        self,
        request: pytest.FixtureRequest,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """A Codex run asking for the web bills a grounding call and answers from it.

        Bedrock runs the search inside the invocation, so the grounding call is
        visible only in the gateway's usage log -- which is also where AWS bills
        it. Nova 2 Lite is pinned to ``us-east-1`` by the suite's configuration
        because no EU inference profile serves the grounding tool.

        ``expensive``: AWS charges per grounding request on top of the tokens.

        Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
             stdapi/usage.py:record_bedrock_usage
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=TOOL,
            server=agentic_server,
            image=agentic_image,
            config=_WEB_SEARCH_CONFIG,
            prompt=_PROMPT_WEB_SEARCH,
            workdir=agentic_workdir,
            test_name=request.node.originalname,
        )
        log_metrics(TOOL, result, _WEB_SEARCH_CONFIG, "test_web_search")
        assert_result(result, config=_WEB_SEARCH_CONFIG, any_of=("http", "linux"))
        assert grounding_requests(agentic_server, log_start) > 0, (
            "Codex asked for the web but the gateway billed no grounding request, "
            "so its web_search declaration never reached the model's search tool"
        )
