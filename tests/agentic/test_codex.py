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
#: Still bounded well under the lane's 15-minute budget: the slowest measured run
#: here is ~280 s, so this is a wide margin rather than an expectation.
_TIMEOUT = 900

#: Ceiling for the ``slow``-marked models, which sit outside the ``--agentic`` budget.
_SLOW_TIMEOUT = 1200

_MODEL_CONFIGS = [
    pytest.param(
        ModelConfig(model="anthropic.claude-haiku-4-5-20251001-v1:0", timeout=_TIMEOUT),
        id="claude-haiku-4-5",
    ),
    pytest.param(
        # Reaches the answer through many more shell round trips than the larger
        # models, so it is the one this module's prompts are sized for.
        ModelConfig(model="amazon.nova-2-lite-v1:0", timeout=_TIMEOUT),
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
        ModelConfig(model="qwen.qwen3-coder-next", timeout=_SLOW_TIMEOUT, flaky=True),
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

Trace how an incoming POST /v1/responses request reaches AWS Bedrock. Use shell
commands to read the source; do not answer from prior knowledge.

Read these files and no others, with `grep -n` and a `sed -n` line range rather
than paging through them:
  1. {SRC_MOUNT}/stdapi/routes/openai_responses.py
  2. {SRC_MOUNT}/stdapi/models/chat/_adapters/_openai_responses.py
  3. {SRC_MOUNT}/stdapi/models/chat/_default.py — the function that builds the
     Bedrock request payload, and that function only

Report exactly three steps, in execution order, each with the real code quote:
  1. The handler registered for /v1/responses
  2. The function that converts the request body into Bedrock's shape
  3. The function that assembles the Bedrock payload, and the name of the AWS
     Bedrock API operation the payload is finally sent to
"""

_PROMPT_STREAMING_PATH = f"""\
You are working in the stdapi.ai source tree at {SRC_MOUNT}.

Trace the streaming path of POST /v1/responses with stream=True. Use shell
commands to read source files; do not guess, and quote the code you read.

Read these files and no others, with `grep -n` and a `sed -n` line range rather
than paging through them:
  1. {SRC_MOUNT}/stdapi/models/chat/_default.py — the branch taken when the
     request is streamed, and the lines around it only
  2. {SRC_MOUNT}/stdapi/models/chat/_adapters/_openai_responses.py — the function
     that turns the Bedrock stream into the client's own server-sent events

Report exactly three things, with the real code quote for each:
  1. The streaming branch, and the AWS Bedrock streaming call it makes
  2. The signature of the function that formats those server-sent events
  3. One raw Bedrock stream event name and the client event it becomes
"""

_PROMPT_PARAMETER_MAPPING = f"""\
You are working in the stdapi.ai source tree at {SRC_MOUNT}.

Map Responses API parameters onto the AWS Bedrock Converse request fields, as this
gateway actually implements it.

Read one function: _prepare_converse_request in
{SRC_MOUNT}/stdapi/models/chat/_default.py. Read that function only, not the rest
of the file, and read no other file — `sed -n` a line range rather than paging
through it.

Report four of the mappings it performs, each on one line:
  OpenAI parameter  →  Bedrock request field, with the exact line that assigns it

Do not report a mapping you did not see in that function.
"""

_PROMPT_MODEL_OVERRIDES = f"""\
You are working in the stdapi.ai source tree at {SRC_MOUNT}.

Report what three of the gateway's model-specific chat modules override.

Read these three files, one shell command each, and read nothing else:
  1. {SRC_MOUNT}/stdapi/models/chat/amazon_nova_2.py
  2. {SRC_MOUNT}/stdapi/models/chat/deepseek_v3.py
  3. {SRC_MOUNT}/stdapi/models/chat/twelvelabs_pegasus.py

Your answer is three sections, one per file, each giving the exact `def` line of
one method that file overrides, copied from the file, and one sentence on what
that override changes. A file name without a quoted `def` line is not an answer.
"""

#: Gateway function names that appear only in the files the prompt forces open.
#: None of them appears in the prompt, so naming one is evidence of a real read.
_PIPELINE_KEYWORDS = ("create_response", "translate_request", "map_input")
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
#: Methods the three named modules override; only reading one of them supplies a name.
_MODEL_OVERRIDE_KEYWORDS = (
    "_req_configure_reasoning",
    "_prepare_converse_request",
    "_resp_map_tool_result",
    "_build_code_execution_result",
    "_build_pegasus_body",
    "_format_converse_stream",
)

#: Fragment of any quoted signature, which a bare directory listing cannot contain.
_SIGNATURE_MARKER = "def "


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
        """Codex audits the Responses-to-Converse parameter mapping from the source.

        The asserted vocabulary is the Bedrock side of the mapping, which appears
        only inside the function the prompt points at -- the OpenAI side would be
        recited correctly without opening anything. That is what carries the test:
        the step floor is one because one ``sed`` range is all the task needs, and
        a floor above what the task requires fails a correct answer.

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
            result, config=model_config, any_of=_PARAMETER_KEYWORDS, min_steps=1
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

        The highest step floor in this module: three separate reads keep the
        ``function_call`` cycle running long enough to exercise a steadily
        growing input array. The files are named rather than discovered, so the
        floor measures the reads rather than a listing the model may treat as the
        deliverable, and the answer has to carry a quoted signature -- a name a
        directory listing cannot supply.

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
            result,
            config=model_config,
            contains=_SIGNATURE_MARKER,
            any_of=_MODEL_OVERRIDE_KEYWORDS,
            min_steps=3,
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
