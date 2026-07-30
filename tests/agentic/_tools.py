"""Registry of agentic CLIs exercised against the gateway, and their wire formats.

Each :class:`AgenticTool` describes everything that differs between one CLI and
another: the npm package baked into the image, the gateway route it speaks, how to
turn an :class:`Invocation` into a command line, and how to normalise the CLI's own
output into an :class:`AgenticResult`. Everything else -- container sandboxing,
the server under test, assertions, metrics and model-identity checking -- is shared.

Adding a tool means appending one entry to :data:`AGENTIC_TOOLS` and writing a test
module; the image picks the new package up automatically because
:func:`npm_packages` feeds the Containerfile's ``PACKAGES`` build argument.

Ref: https://platform.claude.com/docs/en/agent-sdk/headless
     https://developers.openai.com/codex/local-config
     tests/agentic/Containerfile
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

#: Read-only mount point of the gateway source inside the container.
SRC_MOUNT = "/src"

#: Writable per-test working directory inside the container.
WORK_MOUNT = "/work"

#: Spend ceiling handed to any CLI that supports one, as a runaway-loop backstop.
_MAX_BUDGET_USD = "10"


@dataclass(frozen=True)
class Invocation:
    """One agentic CLI run requested by a test."""

    workdir: Path
    port: int
    api_key: str
    model: str
    prompt: str
    session_id: str
    effort: str | None = None
    extra_env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Command:
    """The container command line, environment and stdin for one invocation."""

    argv: tuple[str, ...]
    env: dict[str, str]
    stdin: str | None = None


@dataclass(frozen=True)
class AgenticResult:
    """A CLI run's outcome, normalised across tools.

    ``steps`` is the tool-specific evidence that the model actually worked the
    codebase rather than answering from memory: Claude Code reports conversation
    turns, Codex reports completed shell executions. Both count up as the agent
    does more work, so a single floor expresses the same intent for either.
    """

    text: str
    steps: int
    input_tokens: int
    output_tokens: int
    cache_read: int = 0
    cache_created: int = 0
    is_error: bool = False
    error_detail: str = ""


@dataclass(frozen=True)
class AgenticTool:
    """A CLI driven end-to-end against the gateway.

    Attributes:
        id: Short identifier, used in metric lines and skip messages.
        npm_package: npm specifier installed into the container image.
        binary: Executable name inside the image.
        route: Gateway path prefix the CLI talks to (``/anthropic`` or ``/v1``).
        metrics_prefix: Tag opening this tool's ``-s`` benchmark lines.
        build: Turns an invocation into the container command.
        parse: Turns the CLI's stdout into a normalised result.
        prepare_workdir: Seeds the per-test working directory (config, policy).
        attributes_sessions: True when the CLI propagates an identifier the
            gateway logs, letting requests be attributed to one test exactly.
    """

    id: str
    npm_package: str
    binary: str
    route: str
    metrics_prefix: str
    build: Callable[[Invocation], Command]
    parse: Callable[[str], AgenticResult]
    prepare_workdir: Callable[[Invocation], None]
    attributes_sessions: bool


# ---------------------------------------------------------------------------
# Claude Code — Anthropic Messages route
# ---------------------------------------------------------------------------

#: Permission policy written to each run's ``.claude/settings.json``.
#:
#: Belt-and-braces with ``--allowedTools``/``--disallowedTools``: the container is
#: the real boundary, but a CLI bug that ignored one layer still hits the other.
_CLAUDE_SETTINGS: dict[str, object] = {
    "defaultMode": "dontAsk",
    "permissions": {
        "allow": [
            "Read(*)",
            "Glob(**)",
            "Bash(find *)",
            "Bash(ls *)",
            "Bash(cat *)",
            "Bash(grep *)",
            "Bash(rg *)",
            "Bash(head *)",
            "Bash(echo *)",
        ],
        "deny": [
            "Write(*)",
            "Edit(*)",
            "Bash(rm *)",
            "Bash(mv *)",
            "Bash(cp *)",
            "Bash(git commit*)",
            "Bash(git push*)",
            "Bash(pip *)",
            "Bash(curl *)",
            "Bash(wget *)",
        ],
    },
}


def _claude_prepare(invocation: Invocation) -> None:
    """Write Claude Code's permission policy and a pre-trusted sandbox config."""
    workdir = invocation.workdir
    (workdir / "home").mkdir(exist_ok=True)
    claude_dir = workdir / ".claude"
    claude_dir.mkdir(exist_ok=True)
    (claude_dir / "settings.json").write_text(json.dumps(_CLAUDE_SETTINGS, indent=2))
    config_dir = workdir / "claude-config"
    config_dir.mkdir(exist_ok=True)
    # Pre-accepting onboarding and directory trust keeps --print non-interactive;
    # the paths are the container's, not the host's.
    (config_dir / ".claude.json").write_text(
        json.dumps(
            {
                "hasCompletedOnboarding": True,
                "bypassPermissionsModeAccepted": True,
                "projects": {
                    WORK_MOUNT: {"hasTrustDialogAccepted": True},
                    SRC_MOUNT: {"hasTrustDialogAccepted": True},
                },
            }
        )
    )


def _claude_build(invocation: Invocation) -> Command:
    """Build the ``claude --print`` command for one invocation.

    The model under test is bound to the ``sonnet`` slot rather than passed to
    ``--model`` directly, because Claude Code only routes aliases through the
    configured provider.
    """
    env = {
        "HOME": f"{WORK_MOUNT}/home",
        "CLAUDE_CONFIG_DIR": f"{WORK_MOUNT}/claude-config",
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{invocation.port}/anthropic",
        "ANTHROPIC_AUTH_TOKEN": invocation.api_key,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": invocation.model,
        **invocation.extra_env,
    }
    argv = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--model",
        "sonnet",
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
        "--session-id",
        invocation.session_id,
        "--max-budget-usd",
        _MAX_BUDGET_USD,
        "--add-dir",
        SRC_MOUNT,
        "--allowedTools",
        "Read,Glob,Bash",
        "--disallowedTools",
        "Write,Edit,WebFetch,WebSearch",
    ]
    if invocation.effort:
        argv += ["--effort", invocation.effort]
    # --add-dir is variadic, so a positional prompt would be swallowed as another
    # directory; stdin is the reliable channel.
    return Command(tuple(argv), env, stdin=invocation.prompt)


def _claude_parse(stdout: str) -> AgenticResult:
    """Normalise ``claude --output-format json`` output.

    Raises:
        ValueError: If the output is not the documented JSON result object.
    """
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        msg = f"claude output is not valid JSON: {exc}\nOutput: {stdout[:500]}"
        raise ValueError(msg) from exc
    usage = data.get("usage", {})
    return AgenticResult(
        text=data.get("result", ""),
        steps=data.get("num_turns", 0),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_read=usage.get("cache_read_input_tokens", 0),
        cache_created=usage.get("cache_creation_input_tokens", 0),
        is_error=bool(data.get("is_error", False)),
        error_detail=json.dumps(data)[:500] if data.get("is_error") else "",
    )


# ---------------------------------------------------------------------------
# Codex — OpenAI Responses route
# ---------------------------------------------------------------------------


def _codex_prepare(invocation: Invocation) -> None:
    """Create the writable HOME and CODEX_HOME the CLI expects."""
    (invocation.workdir / "home").mkdir(exist_ok=True)
    (invocation.workdir / "codex-home").mkdir(exist_ok=True)


def _codex_build(invocation: Invocation) -> Command:
    """Build the ``codex exec --json`` command for one invocation.

    A dedicated ``stdapi`` provider is declared because current Codex releases
    refuse overrides of the reserved built-in ``openai`` provider. ``env_key``
    makes Codex send the gateway's API key as a bearer token.

    The model-generated shell commands run with Codex's own ``read-only``
    sandbox, nested inside the container that is the actual security boundary.
    """
    base_url = f"http://127.0.0.1:{invocation.port}/v1"
    env = {
        "HOME": f"{WORK_MOUNT}/home",
        "CODEX_HOME": f"{WORK_MOUNT}/codex-home",
        "OPENAI_API_KEY": invocation.api_key,
        "OPENAI_BASE_URL": base_url,
        # The container cannot resolve the vendor's telemetry host, and each run
        # would otherwise stall several seconds on that lookup before starting.
        "OTEL_SDK_DISABLED": "true",
        **invocation.extra_env,
    }
    argv = [
        "codex",
        "exec",
        "-c",
        'model_provider="stdapi"',
        "-c",
        'model_providers.stdapi.name="stdapi.ai"',
        "-c",
        f'model_providers.stdapi.base_url="{base_url}"',
        "-c",
        'model_providers.stdapi.env_key="OPENAI_API_KEY"',
        "-c",
        'model_providers.stdapi.wire_api="responses"',
        "-m",
        invocation.model,
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "-s",
        "read-only",
        "-C",
        SRC_MOUNT,
        invocation.prompt,
    ]
    return Command(tuple(argv), env)


def _codex_parse(stdout: str) -> AgenticResult:
    """Normalise ``codex exec --json`` JSONL output.

    Raises:
        ValueError: If no JSONL events were emitted.
    """
    events: list[dict[str, object]] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not events:
        msg = f"codex produced no JSONL events\nOutput: {stdout[:500]}"
        raise ValueError(msg)

    def _items(item_type: str) -> list[dict[str, object]]:
        return [
            item
            for event in events
            if event.get("type") == "item.completed"
            and isinstance(item := event.get("item"), dict)
            and item.get("type") == item_type
        ]

    failed = [event for event in events if event.get("type") == "turn.failed"]
    messages = [str(item.get("text", "")) for item in _items("agent_message")]
    completed = next(
        (event for event in events if event.get("type") == "turn.completed"), {}
    )
    usage = completed.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    return AgenticResult(
        text=messages[-1] if messages else "",
        steps=len(_items("command_execution")),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_read=usage.get("cached_input_tokens", 0),
        is_error=bool(failed),
        error_detail=json.dumps(failed[0])[:500] if failed else "",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CLAUDE_CODE = AgenticTool(
    id="claude-code",
    npm_package="@anthropic-ai/claude-code@latest",
    binary="claude",
    route="/anthropic",
    metrics_prefix="CC-METRICS",
    build=_claude_build,
    parse=_claude_parse,
    prepare_workdir=_claude_prepare,
    # Claude Code puts --session-id in the request metadata, which the gateway
    # logs as request_user_id.
    attributes_sessions=True,
)

CODEX = AgenticTool(
    id="codex",
    npm_package="@openai/codex@latest",
    binary="codex",
    route="/v1",
    metrics_prefix="CO-METRICS",
    build=_codex_build,
    parse=_codex_parse,
    prepare_workdir=_codex_prepare,
    # Codex sends no per-run identifier the gateway records, so its requests can
    # only be attributed positionally.
    attributes_sessions=False,
)

#: Every agentic CLI the suite knows how to drive.
AGENTIC_TOOLS: tuple[AgenticTool, ...] = (CLAUDE_CODE, CODEX)


def npm_packages(tools: Sequence[AgenticTool] = AGENTIC_TOOLS) -> tuple[str, ...]:
    """Return the npm specifiers to install into the container image."""
    return tuple(sorted({tool.npm_package for tool in tools}))
