"""Hermes driven end-to-end over its three transports, plus Anthropic cache TTLs.

Hermes is the lane's second three-wire-format client and its only Python agent
binary. Its ``providers.<name>.transport`` key takes ``chat_completions``,
``anthropic_messages`` or ``codex_responses``, so one config file, one prompt and
one model cover all three of the gateway's chat routes; a failure on exactly one
value isolates the fault to that adapter.

What this exercises that nothing else in the lane does:

- **Anthropic prompt-caching breakpoints, chosen by the client.** Hermes turns
  ``prompt_caching.cache_ttl`` into ``cache_control`` markers on the system
  prompt and the last three messages of every Anthropic-wire request to a
  Claude-named model, and emits ``{"type": "ephemeral", "ttl": "1h"}`` for the
  ``1h`` tier against a bare ``{"type": "ephemeral"}`` for ``5m``. Every other
  client here either sends no ``cache_control`` at all or sends the gateway's
  automatic form; this is the only one that puts explicit, TTL-tiered
  breakpoints on the wire, which is the branch
  ``_anthropic_message.py:_build_cache_point`` exists for.

The whole run happens inside the container, with Hermes's own state
(``HERMES_HOME``), config and usage report under the writable ``/work`` mount and
its tool set restricted to file reads and searches -- no terminal, no web, no
browser, and no external execution backend.

Requires ``--agentic``, podman, and Bedrock credentials.

Ref: https://hermes-agent.nousresearch.com/docs/reference/cli-commands
     https://hermes-agent.nousresearch.com/docs/integrations/providers
     https://hermes-agent.nousresearch.com/docs/user-guide/configuration
     https://platform.claude.com/docs/en/build-with-claude/prompt-caching
     tests/agentic/_tools.py:_hermes_prepare
     stdapi/models/chat/_adapters/_anthropic_message.py:_build_cache_point
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ._runner import ModelConfig, assert_result, log_metrics, run_agent
from ._tools import (
    HERMES_ANTHROPIC,
    HERMES_CACHE_TTL_VAR,
    HERMES_CHAT_COMPLETIONS,
    HERMES_RESPONSES,
    SRC_MOUNT,
    AgenticTool,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ._server import AgenticServer

pytestmark = pytest.mark.agentic

#: Hermes plans several file reads per turn; the same ceiling pi uses.
_TIMEOUT = 1800

#: Gateway route each transport must be observed reaching.
_EXPECTED_PATHS = {
    HERMES_CHAT_COMPLETIONS.id: "/v1/chat/completions",
    HERMES_RESPONSES.id: "/v1/responses",
    HERMES_ANTHROPIC.id: "/anthropic/v1/messages",
}

#: Short label of each transport, for readable test ids.
_TRANSPORT_LABELS = {
    HERMES_CHAT_COMPLETIONS.id: "chat-completions",
    HERMES_RESPONSES.id: "codex-responses",
    HERMES_ANTHROPIC.id: "anthropic-messages",
}

#: Claude model backing both suites.
#:
#: Mandatory for the caching suite -- Hermes gates its Anthropic prompt caching on
#: a Claude-named model -- and the family whose reasoning-signature replay this
#: branch fixes, so it is the one model both suites share.
_CLAUDE_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"

#: Converse-backed models every transport is exercised with.
_CONVERSE_MODELS = (
    ("claude-haiku-4-5", _CLAUDE_MODEL),
    ("nova-2-lite", "amazon.nova-2-lite-v1:0"),
)

#: Bedrock Mantle model paired with each transport.
#:
#: One per transport rather than one across all three, because Mantle serves a
#: given model on some routes only: ``openai.gpt-5.6-luna`` and
#: ``google.gemma-4-31b`` answer a Chat Completions request with "does not
#: support the '/v1/chat/completions' API" / "isn't supported on this route"
#: (``tests/probes/results/``), while Qwen 3 32B is the family with observed tool
#: calls on that route. Mantle is also where Chat Completions reasoning text comes
#: back under ``reasoning`` rather than ``reasoning_content``, which no other
#: client test covers.
_MANTLE_MODELS = {
    HERMES_CHAT_COMPLETIONS.id: ("qwen3-next-80b", "qwen.qwen3-next-80b-a3b-instruct"),
    HERMES_RESPONSES.id: ("gpt-5-6-luna", "openai.gpt-5.6-luna"),
    HERMES_ANTHROPIC.id: ("gemma-4-31b", "google.gemma-4-31b"),
}

#: One (transport, model) pair per run, since the two are not independent.
_CASES = [
    pytest.param(
        tool,
        ModelConfig(model=model, timeout=_TIMEOUT),
        id=f"{_TRANSPORT_LABELS[tool.id]}-{label}",
    )
    for tool in (HERMES_CHAT_COMPLETIONS, HERMES_RESPONSES, HERMES_ANTHROPIC)
    for label, model in (*_CONVERSE_MODELS, _MANTLE_MODELS[tool.id])
]

#: The two TTL tiers Anthropic supports, each on its own run.
#:
#: Carried through ``extra_env`` because that is the one per-test channel a tool's
#: ``prepare_workdir`` can read, and the tier is a config-file value rather than a
#: command-line flag.
_CACHE_TTL_MODEL_CONFIGS = [
    pytest.param(
        ModelConfig(
            model=_CLAUDE_MODEL,
            timeout=_TIMEOUT,
            extra_env={HERMES_CACHE_TTL_VAR: "5m"},
        ),
        id="5m",
    ),
    pytest.param(
        ModelConfig(
            model=_CLAUDE_MODEL,
            timeout=_TIMEOUT,
            extra_env={HERMES_CACHE_TTL_VAR: "1h"},
        ),
        id="1h",
    ),
]


@pytest.fixture
def agentic_tool(request: pytest.FixtureRequest) -> AgenticTool:
    """The Hermes entry under test; read by the autouse model-identity fixture."""
    tool: AgenticTool = request.param
    return tool


_PROMPT_ADAPTER_LAYOUT = f"""\
You are working in the stdapi.ai source tree at {SRC_MOUNT}.

Identify every chat API adapter the gateway implements and what each one
translates between.

Use your file tools to read the actual source. Do not guess:
  1. List every file in {SRC_MOUNT}/stdapi/models/chat/_adapters/
  2. Read the module docstring of each adapter and quote it
  3. For each adapter, name the client API it accepts and quote the exact
     signature of the function that converts a request into Bedrock's shape

Report one section per adapter, with real code quotes.
"""

_PROMPT_ADAPTER_COMMON = f"""\
You are working in the stdapi.ai source tree at {SRC_MOUNT}.

Read {SRC_MOUNT}/stdapi/models/chat/_adapters/_common.py and report what the
module is for, naming at least two functions it defines and quoting the exact
signature of each. Read the file before answering; do not guess.
"""

#: Vocabulary that appears only in the adapter files the prompt forces open.
_ADAPTER_KEYWORDS = (
    "_openai_chat_completion",
    "_openai_responses",
    "_anthropic_message",
    "translate_request",
    "adapter",
)

#: Vocabulary any correct account of the shared adapter helpers uses.
_COMMON_KEYWORDS = ("append_or_merge", "def ", "bedrock", "message")


def chat_request_entries(
    server: AgenticServer, log_start: int, path: str
) -> list[dict[str, object]]:
    """Return the requests the gateway logged for *path* since *log_start*.

    Args:
        server: Gateway the CLI was pointed at.
        log_start: Log index captured before the run.
        path: Route to collect, e.g. ``/anthropic/v1/messages``.

    Returns:
        One log entry per matching request, in order. Empty when the gateway's
        log is not observable, which is the case for the external deployment
        ``--server-url`` selects, so a caller may treat an empty list as "skip
        the log assertions".
    """
    if server.process is None:
        return []
    return [
        entry
        for entry in server.log_entries(log_start)
        if entry.get("type") == "request" and str(entry.get("path") or "") == path
    ]


def cache_control_markers(value: object) -> list[dict[str, object]]:
    """Return every ``cache_control`` marker nested anywhere inside *value*.

    Anthropic accepts the marker on system blocks, message content blocks and
    tool definitions alike, and Hermes places one on each of four breakpoints, so
    the whole logged body is walked rather than any single field.

    Args:
        value: Decoded JSON fragment of a logged request body.

    Returns:
        One dict per marker found.
    """
    markers: list[dict[str, object]] = []
    if isinstance(value, dict):
        marker = value.get("cache_control")
        if isinstance(marker, dict):
            markers.append(marker)
        markers += [
            found
            for key, nested in value.items()
            if key != "cache_control"
            for found in cache_control_markers(nested)
        ]
    elif isinstance(value, list):
        markers += [found for item in value for found in cache_control_markers(item)]
    return markers


@pytest.mark.parametrize(
    ("agentic_tool", "model_config"), _CASES, indirect=["agentic_tool"]
)
class TestHermesAcrossTransports:
    """The same agent task over Chat Completions, Responses and Anthropic Messages.

    The parametrizations differ only in the ``transport`` value written to
    Hermes's provider entry and the model it routes to, so a failure on one is a
    failure of that adapter or that model's own path rather than of the prompt.

    Ref: https://hermes-agent.nousresearch.com/docs/integrations/providers
         stdapi/models/chat/_adapters/
    """

    def test_enumerate_adapters_on_the_selected_transport(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """Hermes answers from the tree, over the route its transport names.

        The answer names files the prompt does not, so it can only come from tool
        output carried back through the gateway's own streaming translation, and
        every request the gateway logged for this test landed on the route this
        transport selects -- a client that silently fell back to Chat Completions
        would otherwise answer correctly and prove nothing.

        Ref: https://hermes-agent.nousresearch.com/docs/reference/cli-commands
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
            "test_enumerate_adapters_on_the_selected_transport",
        )
        assert_result(
            result,
            config=model_config,
            contains="adapter",
            any_of=_ADAPTER_KEYWORDS,
            min_steps=2,
        )
        expected = _EXPECTED_PATHS[agentic_tool.id]
        if agentic_server.process is not None:
            served = {
                str(entry.get("path") or "")
                for entry in agentic_server.log_entries(log_start)
                if entry.get("type") == "request"
                and str(entry.get("path") or "") in set(_EXPECTED_PATHS.values())
            }
            assert served == {expected}, (
                f"{agentic_tool.id} was configured for {expected!r} but the "
                f"gateway served {sorted(served)}"
            )


@pytest.mark.parametrize("model_config", _CACHE_TTL_MODEL_CONFIGS)
class TestHermesAnthropicPromptCaching:
    """Hermes's TTL-tiered ``cache_control`` breakpoints, as the gateway sees them.

    Hermes auto-enables Anthropic prompt caching for a Claude-named model on the
    ``anthropic_messages`` transport and marks four breakpoints -- the system
    prompt plus the last three non-system messages -- at the tier named by
    ``prompt_caching.cache_ttl``. The gateway turns each marker into a Bedrock
    ``cachePoint``, carrying the TTL only when the client sent one.

    The assertion is the marker on the wire, not the cache hit: a breakpoint
    below the model's minimum cacheable prompt length caches nothing and returns
    no error, so a token-count assertion would pass on a gateway that dropped the
    TTL entirely.

    Ref: https://hermes-agent.nousresearch.com/docs/user-guide/configuration
         https://platform.claude.com/docs/en/build-with-claude/prompt-caching
         stdapi/models/chat/_adapters/_anthropic_message.py:_build_cache_point
    """

    @pytest.fixture
    def agentic_tool(self) -> AgenticTool:
        """Only the Anthropic transport carries ``cache_control`` breakpoints."""
        return HERMES_ANTHROPIC

    def test_cache_ttl_reaches_the_gateway_on_every_breakpoint(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """Each TTL tier arrives in the shape Hermes documents for it.

        ``1h`` must reach the gateway as ``{"type": "ephemeral", "ttl": "1h"}``
        on every breakpoint, and ``5m`` as a bare ``{"type": "ephemeral"}`` --
        Hermes omits the field for the default tier rather than spelling it out,
        so asserting ``ttl == "5m"`` would fail against a correct client. A run
        that produced no marker at all fails too: that is what a gateway
        rejecting the Anthropic caching contract, or a transport quietly
        downgraded to Chat Completions, looks like.

        Ref: https://hermes-agent.nousresearch.com/docs/user-guide/configuration
             stdapi/types/anthropic_messages.py:CacheControlEphemeralParam
        """
        ttl = model_config.extra_env[HERMES_CACHE_TTL_VAR]
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_PROMPT_ADAPTER_COMMON,
            workdir=agentic_workdir,
            test_name=f"{request.node.originalname}[{ttl}]",
        )
        log_metrics(
            agentic_tool,
            result,
            model_config,
            f"test_cache_ttl_reaches_the_gateway[{ttl}]",
        )
        assert_result(result, config=model_config, any_of=_COMMON_KEYWORDS, min_steps=1)

        entries = chat_request_entries(
            agentic_server, log_start, _EXPECTED_PATHS[agentic_tool.id]
        )
        if not entries:
            return  # External server: no log to inspect.
        markers = [
            marker
            for entry in entries
            for marker in cache_control_markers(entry.get("request_params"))
        ]
        assert markers, (
            "Hermes sent no cache_control breakpoint on the Anthropic route; "
            f"logged {len(entries)} request(s) for {model_config.model}"
        )
        assert all(marker.get("type") == "ephemeral" for marker in markers), (
            f"unexpected cache_control marker types: {markers}"
        )
        ttls = {marker.get("ttl") for marker in markers}
        expected_ttl = "1h" if ttl == "1h" else None
        assert ttls == {expected_ttl}, (
            f"cache_ttl={ttl!r} should reach the gateway with ttl={expected_ttl!r} "
            f"on every breakpoint, got {sorted(str(value) for value in ttls)}"
        )
