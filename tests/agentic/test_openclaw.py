"""OpenClaw driven end-to-end over all three of the gateway's chat wire formats.

OpenClaw is the only client in the lane that exposes the wire format as a
first-class switch on one flag: ``--custom-compatibility openai |
openai-responses | anthropic`` picks Chat Completions, Responses or Anthropic
Messages for the same endpoint, model and prompt. pi reaches the same three
routes, but through three separate provider declarations; here the declaration
is identical and only the switch moves, so a failure on exactly one value
isolates the fault to that adapter rather than to the model, the prompt or the
provider configuration.

The run is two commands, both inside the container: ``openclaw onboard
--non-interactive`` writes the custom provider (that is where the switch lives),
then ``openclaw agent --local`` executes one embedded agent turn against it. No
Gateway daemon is started, and no OpenClaw account or vendor token is involved --
the gateway's own API key is the only credential that crosses the boundary.

Assertions are on the gateway's own request log rather than on the CLI's exit
code: the route each compatibility value actually reached, and the model every
request targeted.

Requires ``--agentic``, podman, and Bedrock credentials.

Ref: https://docs.openclaw.ai/cli/onboard
     https://docs.openclaw.ai/cli/agent
     https://docs.openclaw.ai/gateway/config-tools
     tests/agentic/_tools.py:_openclaw_build
     stdapi/models/chat/_adapters/
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ._runner import ModelConfig, assert_result, log_metrics, run_agent
from ._tools import (
    OPENCLAW_ANTHROPIC,
    OPENCLAW_OPENAI,
    OPENCLAW_RESPONSES,
    SRC_MOUNT,
    AgenticTool,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ._server import AgenticServer

pytestmark = pytest.mark.agentic

#: OpenClaw onboards, then runs an embedded turn; both share one container run.
_TIMEOUT = 1800

#: Gateway route each compatibility value must be observed reaching.
#:
#: This is the assertion the module exists for: the switch is a client-side flag,
#: and only the server log proves it selected the wire format it names.
_EXPECTED_PATHS = {
    OPENCLAW_OPENAI.id: "/v1/chat/completions",
    OPENCLAW_RESPONSES.id: "/v1/responses",
    OPENCLAW_ANTHROPIC.id: "/anthropic/v1/messages",
}

#: Short label of each compatibility value, for readable test ids.
_COMPATIBILITY_LABELS = {
    OPENCLAW_OPENAI.id: "openai",
    OPENCLAW_RESPONSES.id: "openai-responses",
    OPENCLAW_ANTHROPIC.id: "anthropic",
}

#: Converse-backed models every compatibility value is exercised with.
#:
#: Deliberately small: this module multiplies by three wire formats, so each entry
#: costs three runs per test. Claude covers the Anthropic-native dialect this
#: branch's reasoning-signature replay fixes, Nova the Amazon one.
_CONVERSE_MODELS = (
    ("claude-haiku-4-5", "anthropic.claude-haiku-4-5-20251001-v1:0"),
    ("nova-2-lite", "amazon.nova-2-lite-v1:0"),
)

#: Bedrock Mantle model paired with each compatibility value.
#:
#: One per wire format rather than one across all three, because Mantle serves a
#: given model on some routes only: ``openai.gpt-5.6-luna`` and
#: ``google.gemma-4-31b`` answer a Chat Completions request with "does not
#: support the '/v1/chat/completions' API" / "isn't supported on this route"
#: (``tests/probes/results/``), while Qwen 3 32B is the family with observed tool
#: calls on that route. Mantle is also where Chat Completions reasoning text comes
#: back under ``reasoning`` rather than ``reasoning_content``, which no other
#: client test covers.
_MANTLE_MODELS = {
    OPENCLAW_OPENAI.id: ("qwen3-next-80b", "qwen.qwen3-next-80b-a3b-instruct"),
    OPENCLAW_RESPONSES.id: ("gpt-5-6-luna", "openai.gpt-5.6-luna"),
    OPENCLAW_ANTHROPIC.id: ("gemma-4-31b", "google.gemma-4-31b"),
}

#: Models that sometimes answer this prompt without calling a tool at all.
#:
#: Observed on qwen3-next-80b over Chat Completions: a full turn, 188K input
#: tokens, and no tool call. The gateway forwards the tools either way -- Claude
#: and Nova drive the loop over the same wire format -- so this is the model
#: declining to explore, not a routing fault.
_MAY_ANSWER_WITHOUT_TOOLS = {"qwen.qwen3-next-80b-a3b-instruct"}

#: One (wire format, model) pair per run, since the two are not independent.
_CASES = [
    pytest.param(
        tool,
        ModelConfig(
            model=model, timeout=_TIMEOUT, flaky=model in _MAY_ANSWER_WITHOUT_TOOLS
        ),
        id=f"{_COMPATIBILITY_LABELS[tool.id]}-{label}",
    )
    for tool in (OPENCLAW_OPENAI, OPENCLAW_RESPONSES, OPENCLAW_ANTHROPIC)
    for label, model in (*_CONVERSE_MODELS, _MANTLE_MODELS[tool.id])
]


@pytest.fixture
def agentic_tool(request: pytest.FixtureRequest) -> AgenticTool:
    """The OpenClaw entry under test; read by the autouse model-identity fixture."""
    tool: AgenticTool = request.param
    return tool


_PROMPT_ADAPTER_LAYOUT = f"""\
You are working in the stdapi.ai source tree at {SRC_MOUNT}.

Identify every chat API adapter the gateway implements and what each one
translates between.

Use your tools to read the actual source. Do not guess:
  1. List every file in {SRC_MOUNT}/stdapi/models/chat/_adapters/
  2. Read the module docstring of each adapter and quote it
  3. For each adapter, name the client API it accepts and quote the exact
     signature of the function that converts a request into Bedrock's shape

Report one section per adapter, with real code quotes.
"""

#: Vocabulary that appears only in the adapter files the prompt forces open.
_ADAPTER_KEYWORDS = (
    "_openai_chat_completion",
    "_openai_responses",
    "_anthropic_message",
    "translate_request",
    "adapter",
)


def chat_request_paths(server: AgenticServer, log_start: int) -> list[str]:
    """Return the chat routes the gateway logged since *log_start*.

    Args:
        server: Gateway the CLI was pointed at.
        log_start: Log index captured before the run.

    Returns:
        One path per logged chat request, in order. Empty when the gateway's log
        is not observable, which is the case for the external deployment
        ``--server-url`` selects, so a caller may treat an empty list as "skip
        the route assertion".
    """
    if server.process is None:
        return []
    return [
        path
        for entry in server.log_entries(log_start)
        if entry.get("type") == "request"
        and (path := str(entry.get("path") or ""))
        and path in set(_EXPECTED_PATHS.values())
    ]


@pytest.mark.parametrize(
    ("agentic_tool", "model_config"), _CASES, indirect=["agentic_tool"]
)
class TestOpenClawAcrossCompatibilityModes:
    """The same agent task over each value of ``--custom-compatibility``.

    Every assertion is wire-format-agnostic except the route one: the
    parametrizations differ only in which adapter the gateway runs and which
    model it routes to, so a failure on one value is a failure of that adapter or
    that model's own path rather than of the prompt.

    Ref: https://docs.openclaw.ai/start/wizard-cli-automation
         stdapi/models/chat/_adapters/
    """

    def test_enumerate_adapters_on_the_selected_wire_format(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """OpenClaw answers from the tree, over the route its switch names.

        Two statements in one run, because one run is one billed agent loop: the
        answer names files the prompt does not, so it can only come from tool
        output carried back through the gateway's own streaming translation; and
        every chat request the gateway logged for this test landed on the route
        this compatibility value selects, so a client that silently fell back to
        its default wire format fails here rather than passing quietly.

        Ref: https://docs.openclaw.ai/cli/onboard
             stdapi/models/chat/_adapters/__init__.py
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_PROMPT_ADAPTER_LAYOUT,
            workdir=agentic_workdir,
            test_name=f"{request.node.originalname}[{agentic_tool.id}]",
        )
        log_metrics(
            agentic_tool,
            result,
            model_config,
            "test_enumerate_adapters_on_the_selected_wire_format",
        )
        assert_result(
            result,
            config=model_config,
            contains="adapter",
            any_of=_ADAPTER_KEYWORDS,
            min_steps=2,
        )
        expected = _EXPECTED_PATHS[agentic_tool.id]
        paths = chat_request_paths(agentic_server, log_start)
        if paths:
            assert set(paths) == {expected}, (
                f"{agentic_tool.id} was configured for {expected!r} but the "
                f"gateway served {sorted(set(paths))}"
            )
