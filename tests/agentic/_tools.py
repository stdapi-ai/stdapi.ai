"""Registry of agentic CLIs exercised against the gateway, and their wire formats.

Each :class:`AgenticTool` describes everything that differs between one CLI and
another: the npm package baked into the image, the gateway route it speaks, how to
turn an :class:`Invocation` into a command line, and how to normalise the CLI's own
output into an :class:`AgenticResult`. Everything else -- container sandboxing,
the server under test, assertions, metrics and model-identity checking -- is shared.

Adding a tool means appending one entry to :data:`AGENTIC_TOOLS` and writing a test
module; the image picks the new package up automatically because
:func:`npm_packages` feeds the Containerfile's ``PACKAGES`` build argument.

Tools that do not fit the shared Node.js image name another :class:`ImageGroup`
instead: one image per group, built from that group's own Containerfile and its own
tools' npm packages, so a single heavyweight install never lands in the image every
other tool waits for.

Ref: https://platform.claude.com/docs/en/agent-sdk/headless
     https://developers.openai.com/codex/local-config
     https://docs.n8n.io/deploy/host-n8n/configure-n8n/use-the-command-line
     https://qwenlm.github.io/qwen-code-docs/en/users/features/headless/
     https://docs.openclaw.ai/cli/onboard
     https://hermes-agent.nousresearch.com/docs/reference/cli-commands
     https://inspect.aisi.org.uk/models.html
     tests/agentic/Containerfile
     tests/agentic/workflows/
     tests/agentic/rag_pipeline.py
     tests/agentic/inspect_eval.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

#: Read-only mount point of the gateway source inside the container.
SRC_MOUNT = "/src"

#: Writable per-test working directory inside the container.
WORK_MOUNT = "/work"

#: Spend ceiling handed to any CLI that supports one, as a runaway-loop backstop.
_MAX_BUDGET_USD = "10"


@dataclass(frozen=True)
class ImageGroup:
    """One container image the lane builds, shared by the tools naming it.

    Attributes:
        name: Identifier a tool selects through :attr:`AgenticTool.image_group`.
        containerfile: Build file, relative to this directory. A group whose
            tools are not npm packages -- or whose install tree is too large to
            impose on every other tool -- names its own instead of the shared
            Node.js one.
    """

    name: str
    containerfile: str = "Containerfile"


#: Group of the shared Node.js image, holding every CLI installable with npm.
DEFAULT_IMAGE_GROUP = "default"

#: Group of the n8n image; its ~2 GB tree stays out of every other tool's image.
N8N_IMAGE_GROUP = "n8n"

#: Group of the Haystack image, whose client is a Python library, not an npm package.
RAG_IMAGE_GROUP = "rag"

#: Group of the Hermes image, whose client is a PyPI package, not an npm one.
HERMES_IMAGE_GROUP = "hermes"

#: Group of the inspect-ai image, whose client cannot share the lane's overlay.
INSPECT_IMAGE_GROUP = "inspect"

#: Every container image the lane can build, keyed by group name.
#:
#: A tool naming a group absent here fails to resolve an image, which is how a
#: typo surfaces instead of silently landing the tool in the shared image.
IMAGE_GROUPS: Mapping[str, ImageGroup] = {
    DEFAULT_IMAGE_GROUP: ImageGroup(name=DEFAULT_IMAGE_GROUP),
    N8N_IMAGE_GROUP: ImageGroup(name=N8N_IMAGE_GROUP),
    RAG_IMAGE_GROUP: ImageGroup(
        name=RAG_IMAGE_GROUP, containerfile="Containerfile.rag"
    ),
    HERMES_IMAGE_GROUP: ImageGroup(
        name=HERMES_IMAGE_GROUP, containerfile="Containerfile.hermes"
    ),
    INSPECT_IMAGE_GROUP: ImageGroup(
        name=INSPECT_IMAGE_GROUP, containerfile="Containerfile.inspect"
    ),
}


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
    extra_args: tuple[str, ...] = ()


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
        npm_package: npm specifier installed into the container image, or None
            for a tool its image already ships (a Python package, a base image's
            own interpreter).
        binary: Executable name inside the image.
        route: Gateway path prefix the CLI talks to (``/anthropic`` or ``/v1``).
        metrics_prefix: Tag opening this tool's ``-s`` benchmark lines.
        build: Turns an invocation into the container command.
        parse: Turns the CLI's stdout into a normalised result.
        prepare_workdir: Seeds the per-test working directory (config, policy).
        attributes_sessions: True when the CLI propagates an identifier the
            gateway logs, letting requests be attributed to one test exactly.
        image_group: Name of the :class:`ImageGroup` whose image runs this tool.
    """

    id: str
    npm_package: str | None
    binary: str
    route: str
    metrics_prefix: str
    build: Callable[[Invocation], Command]
    parse: Callable[[str], AgenticResult]
    prepare_workdir: Callable[[Invocation], None]
    attributes_sessions: bool
    image_group: str = DEFAULT_IMAGE_GROUP


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

    Codex's own Linux sandbox is turned off: it is built on Landlock, which the
    container runtime's seccomp profile denies, and Codex then refuses to run any
    shell command at all. The container is the real boundary anyway -- no
    capabilities, no new privileges, a read-only root and read-only source mounts,
    with ``/work`` and ``/tmp`` the only writable paths.
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
        # Codex declares a web_search tool on every request unless told not to,
        # and the answer depends on the model and the region rather than on
        # anything these tests exercise: Nova maps it to nova_grounding, which
        # no EU inference profile serves, and a Mantle-native model answers 400
        # for a hosted tool it does not implement. Both are correct 400s, and
        # both would fail every run here for a reason unrelated to the shell
        # round trip under test.
        "-c",
        'web_search="disabled"',
        "-m",
        invocation.model,
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "-s",
        "danger-full-access",
        "-C",
        SRC_MOUNT,
        *invocation.extra_args,
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
# pi — one CLI over all three chat routes
# ---------------------------------------------------------------------------
#
# pi reaches each route through its own provider abstraction, so the same binary,
# prompts and assertions cover Chat Completions, Responses and Anthropic Messages.
# That makes it the only tool in the lane whose results are comparable *across*
# routes: a failure on exactly one of the three isolates the fault to that
# adapter rather than to the model or the prompt.

#: pi's built-in read-only tools, mirroring the other CLIs' read-only posture.
_PI_TOOLS = "read,bash,grep,find,ls"

#: Context window pi is told the model under test has.
#:
#: Deliberately permissive: the gateway, not the client, enforces the real
#: per-model limit, and a client-side cap would silently truncate a prompt
#: instead of surfacing the gateway's own error.
_PI_CONTEXT_WINDOW = 200_000

#: Output token ceiling pi is told the model under test has.
#:
#: Unlike the context window this cannot be permissive: pi sends it as
#: ``max_tokens`` on every request, and a value above the model's own output
#: limit (64000 on Claude Haiku 4.5, 65535 on Nova 2 Lite) is rejected before the
#: run starts. Well below every limit in the lane, and far above what these
#: prompts generate.
_PI_MAX_TOKENS = 32_768

#: Environment variable carrying the key into the container for pi.
#:
#: The released pi sends ``apiKey`` verbatim rather than resolving a variable it
#: names, so the value itself goes into ``models.json``. That file lives in the
#: run's own throwaway workdir, which keeps the key out of the command line and
#: out of any committed file; the variable is kept only so the environment the
#: container gets still carries it, as every other tool here expects.
_PI_API_KEY_VAR = "STDAPI_API_KEY"


def _pi_prepare(api: str, route: str) -> Callable[[Invocation], None]:
    """Return a ``prepare_workdir`` writing pi's provider declaration for one route.

    pi declares custom providers in ``~/.pi/agent/models.json``: ``baseUrl``,
    ``api`` and one ``id`` per model are the whole contract, and the model is then
    selected as ``<provider>/<model id>``. That is the mechanism the gateway's own
    documentation promises, so it is the one the lane exercises.

    Args:
        api: pi provider API kind (``openai-completions``, ``openai-responses``
            or ``anthropic-messages``).
        route: Gateway path prefix serving that API.

    Returns:
        A ``prepare_workdir`` callable for an :class:`AgenticTool`.
    """

    def prepare(invocation: Invocation) -> None:
        config_dir = invocation.workdir / "home" / ".pi" / "agent"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "models.json").write_text(
            json.dumps(
                {
                    "providers": {
                        "stdapi": {
                            "name": "stdapi.ai",
                            "baseUrl": f"http://127.0.0.1:{invocation.port}{route}",
                            "api": api,
                            "apiKey": invocation.api_key,
                            "authHeader": True,
                            "models": [
                                {
                                    "id": invocation.model,
                                    "name": invocation.model,
                                    "reasoning": False,
                                    "input": ["text"],
                                    "cost": {
                                        "input": 0,
                                        "output": 0,
                                        "cacheRead": 0,
                                        "cacheWrite": 0,
                                    },
                                    "contextWindow": _PI_CONTEXT_WINDOW,
                                    "maxTokens": _PI_MAX_TOKENS,
                                }
                            ],
                        }
                    }
                },
                indent=2,
            )
        )

    return prepare


def _pi_build(invocation: Invocation) -> Command:
    """Build the ``pi --print --mode json`` command for one invocation.

    The route and wire format live in ``models.json`` (see :func:`_pi_prepare`),
    so the command line only has to select the provider and model that file
    declares.
    """
    env = {
        "HOME": f"{WORK_MOUNT}/home",
        _PI_API_KEY_VAR: invocation.api_key,
        # The container cannot resolve pi's update-check host, and startup
        # would otherwise stall on that lookup before the first request.
        "PI_OFFLINE": "1",
        **invocation.extra_env,
    }
    argv = [
        "pi",
        "--print",
        "--mode",
        "json",
        "--offline",
        "--no-session",
        # Discovery would pick up the gateway's own AGENTS.md from the source
        # mount and change the prompt under test between runs.
        "--no-extensions",
        "--no-context-files",
        "--no-skills",
        "--provider",
        "stdapi",
        "--model",
        f"stdapi/{invocation.model}",
        "--tools",
        _PI_TOOLS,
    ]
    if invocation.effort:
        argv += ["--thinking", invocation.effort]
    argv.append(invocation.prompt)
    return Command(tuple(argv), env)


def _pi_parse(stdout: str) -> AgenticResult:
    """Normalise ``pi --mode json`` JSONL output.

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
        msg = f"pi produced no JSONL events\nOutput: {stdout[:500]}"
        raise ValueError(msg)

    texts: list[str] = []
    usage: dict[str, object] = {}
    error_detail = ""
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        # Usage is cumulative per assistant message; the last one totals the run.
        if isinstance(message_usage := message.get("usage"), dict):
            usage = message_usage
        if detail := message.get("errorMessage"):
            error_detail = str(detail)
        texts += [
            str(part.get("text", ""))
            for part in message.get("content", [])
            if isinstance(part, dict) and part.get("type") == "text"
        ]

    def _tokens(key: str) -> int:
        value = usage.get(key, 0)
        return value if isinstance(value, int) else 0

    return AgenticResult(
        text=texts[-1] if texts else "",
        # Every completed tool call counts, matching Codex's shell-call floor:
        # both count up as the agent does more work in the tree.
        steps=sum(event.get("type") == "tool_execution_end" for event in events),
        input_tokens=_tokens("input"),
        output_tokens=_tokens("output"),
        cache_read=_tokens("cacheRead"),
        cache_created=_tokens("cacheWrite"),
        is_error=bool(error_detail),
        error_detail=error_detail[:500],
    )


# ---------------------------------------------------------------------------
# Qwen Code — the only agent that replays reasoning back to the gateway
# ---------------------------------------------------------------------------
#
# Qwen Code reads an assistant message's ``reasoning_content`` into a thought part
# and re-emits it on the next request's assistant message, so a multi-turn tool
# loop puts the gateway's own reasoning text back on the wire. No other CLI in the
# lane does that: Claude Code, Codex and pi read reasoning and drop it.

#: Tools denied to Qwen Code, mirroring the other CLIs' read-only posture.
#:
#: ``--exclude-tools`` rather than ``--core-tools``: the latter is ignored under
#: ``--safe-mode``, which the CLI says on stderr and then registers every tool.
_QWEN_EXCLUDED_TOOLS = (
    "write_file",
    "edit",
    "notebook_edit",
    "web_fetch",
    "web_search",
)

#: File the message array is written to, then echoed to stdout.
#:
#: The CLI prints its whole transcript in one write and exits, and a large write to
#: a pipe is asynchronous in Node: the tail is lost, which cut a run's JSON
#: mid-string. A write to a file is synchronous, so redirecting and echoing back
#: keeps the whole array -- the same fix as the n8n runs below.
_QWEN_RUN_OUTPUT = "run.json"

#: Shell running the CLI through that file. ``"$@"`` avoids re-quoting the prompt.
_QWEN_SCRIPT = f"""\
"$@" > {WORK_MOUNT}/{_QWEN_RUN_OUTPUT}
status=$?
cat {WORK_MOUNT}/{_QWEN_RUN_OUTPUT}
exit $status
"""

#: Settings pinning the provider, and silencing everything the container cannot reach.
#:
#: ``security.auth.selectedType`` is what makes the run non-interactive: without a
#: selected auth type the CLI refuses to start rather than inferring one.
_QWEN_SETTINGS: dict[str, object] = {
    "security": {"auth": {"selectedType": "openai"}},
    "general": {"enableAutoUpdate": False},
    "privacy": {"usageStatisticsEnabled": False},
    "telemetry": {"enabled": False},
}


def _qwen_prepare(invocation: Invocation) -> None:
    """Write Qwen Code's writable HOME and the settings for this run.

    ``model.reasoningEffort`` is only written when the test asked for an effort
    level: the CLI turns it into a ``reasoning`` object on every request, which a
    model without a reasoning knob would reject.
    """
    home = invocation.workdir / "home"
    config_dir = home / ".qwen"
    config_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(_QWEN_SETTINGS)
    if invocation.effort:
        settings["model"] = {"reasoningEffort": invocation.effort}
    (config_dir / "settings.json").write_text(json.dumps(settings, indent=2))


def _qwen_build(invocation: Invocation) -> Command:
    """Build the ``qwen -p --output-format json`` command for one invocation.

    ``--safe-mode`` is the equivalent of pi's ``--no-extensions``/
    ``--no-context-files``: it drops context files, hooks, extensions, skills and
    MCP servers, so the prompt under test is the prompt the test wrote rather than
    that plus whatever the source mount happens to carry.

    The CLI runs under :data:`_QWEN_SCRIPT` rather than directly, so its whole
    transcript reaches the test instead of as much of it as the pipe accepted
    before the process exited.
    """
    env = {
        "HOME": f"{WORK_MOUNT}/home",
        "OPENAI_API_KEY": invocation.api_key,
        "OPENAI_BASE_URL": f"http://127.0.0.1:{invocation.port}/v1",
        "OPENAI_MODEL": invocation.model,
        # The container cannot resolve the npm registry, and the CLI would
        # otherwise stall on its update check -- the same class of fix as Codex's
        # OTEL_SDK_DISABLED.
        "QWEN_CODE_SKIP_UPDATE_CHECK_ONCE": "true",
        **invocation.extra_env,
    }
    argv = [
        "qwen",
        "--output-format",
        "json",
        "--yolo",
        "--safe-mode",
        "--auth-type",
        "openai",
        "--model",
        invocation.model,
        "--session-id",
        invocation.session_id,
        "--exclude-tools",
        ",".join(_QWEN_EXCLUDED_TOOLS),
        "--include-directories",
        SRC_MOUNT,
        *invocation.extra_args,
        # -p keeps the prompt off the positional slot, which --include-directories
        # would otherwise swallow as another directory.
        "-p",
        invocation.prompt,
    ]
    return Command(("sh", "-c", _QWEN_SCRIPT, "sh", *argv), env)


def _qwen_parse(stdout: str) -> AgenticResult:
    """Normalise ``qwen --output-format json`` output.

    The CLI prints one JSON array holding every message of the run, terminated by
    a ``result`` envelope. ``steps`` counts the tool-use blocks the assistant
    messages carry, matching Codex's shell-call floor: both count up as the agent
    does more work in the tree.

    Raises:
        ValueError: If the output holds no message array.
    """
    messages = _qwen_messages(stdout)
    result = next(
        (message for message in reversed(messages) if message.get("type") == "result"),
        {},
    )
    usage = result.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    error = result.get("error")
    return AgenticResult(
        text=str(result.get("result", "")),
        steps=sum(
            block.get("type") == "tool_use"
            for message in messages
            for block in _qwen_assistant_blocks(message)
        ),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_read=int(usage.get("cache_read_input_tokens") or 0),
        is_error=bool(result.get("is_error")),
        error_detail=json.dumps(error)[:500] if error else "",
    )


def _qwen_messages(stdout: str) -> list[dict[str, object]]:
    """Return the message array the CLI printed.

    The array is written as a single line, but the CLI's own diagnostics land on
    stdout too, so it is located as the last line that decodes to a list.

    Args:
        stdout: Everything the container wrote to stdout.

    Returns:
        One dict per emitted message.

    Raises:
        ValueError: If no message array is present.
    """
    for line in reversed(stdout.splitlines()):
        if not line.startswith("["):
            continue
        try:
            messages = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(messages, list):
            return [message for message in messages if isinstance(message, dict)]
    msg = f"qwen printed no message array\nOutput: {stdout[-500:]}"
    raise ValueError(msg)


def _qwen_assistant_blocks(message: Mapping[str, object]) -> list[dict[str, object]]:
    """Return the content blocks of *message* when it is an assistant message.

    Args:
        message: One entry of the CLI's message array.

    Returns:
        The assistant message's content blocks; empty for any other message.
    """
    if message.get("type") != "assistant":
        return []
    inner = message.get("message")
    if not isinstance(inner, dict) or not isinstance(
        blocks := inner.get("content"), list
    ):
        return []
    return [block for block in blocks if isinstance(block, dict)]


# ---------------------------------------------------------------------------
# n8n — one workflow per gateway surface the coding agents never touch
# ---------------------------------------------------------------------------
#
# n8n is not an agent: it is a workflow runner whose OpenAI node speaks the whole
# OpenAI REST surface, and whose credential carries a base URL that redirects every
# one of those calls at the gateway. That makes it the lane's only vehicle for the
# non-chat routes -- embeddings, audio, images, files, moderations, videos and
# legacy completions -- none of which a coding agent ever calls.
#
# Each surface is one committed workflow template plus one registry entry; the
# command line, the environment and the output contract are identical across them.

#: npm specifier shared by every n8n entry.
_N8N_PACKAGE = "n8n@latest"

#: Directory holding the committed workflow templates, one per surface.
_N8N_WORKFLOW_DIR = Path(__file__).parent / "workflows"

#: Workflow id every template pins, so ``n8n execute --id`` needs no lookup.
_N8N_WORKFLOW_ID = "stdapi-test"

#: Credential id every template's OpenAI nodes reference.
_N8N_CREDENTIAL_ID = "stdapi-openai"

#: Credential id the Anthropic Messages template's node references.
_N8N_ANTHROPIC_CREDENTIAL_ID = "stdapi-anthropic"

#: Only directory the workflows' file nodes may read or write.
#:
#: n8n restricts file access to ``~/.n8n-files`` by default and answers "Access to
#: the file is not allowed" for anything else, so the allowed root has to be named
#: explicitly. Naming one directory rather than ``/work`` also keeps the run's own
#: database and credential file out of reach of the workflow under test.
N8N_FILES_DIR = f"{WORK_MOUNT}/files"

#: File the execution record is written to, then echoed to stdout.
#:
#: ``n8n execute`` exits as soon as it has printed, and a large write to a pipe is
#: asynchronous in Node: the tail is lost, which truncated a 400 kB embeddings run
#: mid-number. A write to a file is synchronous, so redirecting and echoing back
#: keeps the whole record.
N8N_RUN_OUTPUT = "run.json"

#: Directories seeded in the working directory before the run.
#:
#: Created on the host so they belong to the test runner rather than to the
#: container, matching what service containers do with their ``data_dirs``.
_N8N_WORK_DIRS = ("home", "n8n", "files")

#: Shell driving one workflow: import the credential, import the workflow, run it.
#:
#: ``--rawOutput`` prints the execution record and nothing else, so the parse
#: contract is the record itself. The imports' own output is kept out of the way
#: because it carries a full migration log on the first (and only) run.
_N8N_SCRIPT = f"""\
set -e
n8n import:credentials --input={WORK_MOUNT}/credentials.json > {WORK_MOUNT}/import.log
n8n import:workflow --input={WORK_MOUNT}/workflow.json >> {WORK_MOUNT}/import.log
n8n execute --id={_N8N_WORKFLOW_ID} --rawOutput > {WORK_MOUNT}/{N8N_RUN_OUTPUT} \
    || {{ cat {WORK_MOUNT}/{N8N_RUN_OUTPUT}; exit 1; }}
cat {WORK_MOUNT}/{N8N_RUN_OUTPUT}
"""


def _n8n_substitutions(invocation: Invocation) -> dict[str, str]:
    """Return the template placeholders and their JSON-escaped values.

    n8n takes no arguments from a test, so :attr:`Invocation.extra_env` doubles as
    the template's substitution table: every entry is exported to the container
    *and* is available to the template as ``__KEY__``.

    Args:
        invocation: Run being prepared.

    Returns:
        Mapping of ``__PLACEHOLDER__`` to a value safe to paste inside a JSON
        string literal.
    """
    values = {
        "PORT": str(invocation.port),
        "MODEL": invocation.model,
        "PROMPT": invocation.prompt,
        "FILES": N8N_FILES_DIR,
        **invocation.extra_env,
    }
    # [1:-1] strips the quotes json.dumps adds: the template already has them.
    return {f"__{key}__": json.dumps(value)[1:-1] for key, value in values.items()}


def _n8n_prepare(surface: str) -> Callable[[Invocation], None]:
    """Return a ``prepare_workdir`` bound to one surface's workflow template.

    Args:
        surface: Template stem under ``tests/agentic/workflows/``.

    Returns:
        A ``prepare_workdir`` callable for an :class:`AgenticTool`.
    """

    def prepare(invocation: Invocation) -> None:
        workdir = invocation.workdir
        for name in _N8N_WORK_DIRS:
            (workdir / name).mkdir(exist_ok=True)
        (workdir / "credentials.json").write_text(
            json.dumps(
                [
                    {
                        "id": _N8N_CREDENTIAL_ID,
                        "name": "stdapi.ai",
                        "type": "openAiApi",
                        # The node rewrites https://api.openai.com/v1 to this URL
                        # for every resource it implements, which is what points
                        # the whole OpenAI surface at the gateway under test.
                        "data": {
                            "apiKey": invocation.api_key,
                            "url": f"http://127.0.0.1:{invocation.port}/v1",
                        },
                    },
                    {
                        "id": _N8N_ANTHROPIC_CREDENTIAL_ID,
                        "name": "stdapi.ai anthropic",
                        "type": "anthropicApi",
                        # The Anthropic node appends /v1/messages to this URL, so
                        # it has to carry the gateway's Anthropic routes prefix
                        # rather than the bare host used by the OpenAI credential.
                        "data": {
                            "apiKey": invocation.api_key,
                            "url": f"http://127.0.0.1:{invocation.port}/anthropic",
                        },
                    },
                ]
            )
        )
        template = (_N8N_WORKFLOW_DIR / f"{surface}.json").read_text()
        for placeholder, value in _n8n_substitutions(invocation).items():
            template = template.replace(placeholder, value)
        (workdir / "workflow.json").write_text(template)

    return prepare


def _n8n_build(invocation: Invocation) -> Command:
    """Build the ``n8n execute`` command for one invocation.

    The encryption key is derived from the run's session id rather than fixed in
    the repository: it protects one throwaway credential in one throwaway SQLite
    database, and deriving it keeps a key-shaped literal out of the source.

    The remaining variables are the offline switches. n8n otherwise polls its
    version, template and telemetry hosts, none of which the container can
    resolve -- the same class of fix as Codex's ``OTEL_SDK_DISABLED``.
    """
    env = {
        "HOME": f"{WORK_MOUNT}/home",
        "N8N_USER_FOLDER": f"{WORK_MOUNT}/n8n",
        "N8N_ENCRYPTION_KEY": sha256(invocation.session_id.encode()).hexdigest()[:32],
        "N8N_RESTRICT_FILE_ACCESS_TO": N8N_FILES_DIR,
        "N8N_DIAGNOSTICS_ENABLED": "false",
        "N8N_VERSION_NOTIFICATIONS_ENABLED": "false",
        "N8N_TEMPLATES_ENABLED": "false",
        "N8N_PERSONALIZATION_ENABLED": "false",
        "N8N_ONBOARDING_FLOW_DISABLED": "true",
        **invocation.extra_env,
    }
    return Command(("sh", "-c", _N8N_SCRIPT), env)


def n8n_files_dir(workdir: Path) -> Path:
    """Return the host view of the directory the workflow's file nodes may use.

    Args:
        workdir: Per-test directory bind-mounted at :data:`WORK_MOUNT`.

    Returns:
        The same directory :data:`N8N_FILES_DIR` names inside the container.
    """
    return workdir / N8N_FILES_DIR[len(WORK_MOUNT) + 1 :]


def n8n_execution_record(stdout: str) -> dict[str, object]:
    """Return the execution record ``n8n execute --rawOutput`` printed.

    The record is preceded by the run's log lines and, when a node fails, followed
    by the CLI's own error report, so it is located as the last pretty-printed
    object starting at column zero rather than by parsing the whole stream.

    Args:
        stdout: Everything the container wrote to stdout.

    Returns:
        The decoded record.

    Raises:
        ValueError: If no execution record is present.
    """
    decoder = json.JSONDecoder()
    offset = 0
    starts: list[int] = []
    for line in stdout.splitlines(keepends=True):
        if line.rstrip("\r\n") == "{":
            starts.append(offset)
        offset += len(line)
    for start in reversed(starts):
        try:
            record, _ = decoder.raw_decode(stdout, start)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            return record
    msg = f"n8n printed no execution record\nOutput: {stdout[-500:]}"
    raise ValueError(msg)


def n8n_node_items(record: Mapping[str, object], node: str) -> list[dict[str, object]]:
    """Return the ``json`` payloads *node* emitted on its first output.

    Args:
        record: Execution record from :func:`n8n_execution_record`.
        node: Workflow node name.

    Returns:
        One dict per item the node produced; empty when it produced none or did
        not run at all.
    """
    runs = _n8n_result_data(record).get("runData", {})
    if not isinstance(runs, dict) or not isinstance(node_runs := runs.get(node), list):
        return []
    items: list[dict[str, object]] = []
    for run in node_runs:
        outputs = (run.get("data") or {}).get("main") or []
        items.extend(
            payload
            for item in (outputs[0] if outputs else [])
            if isinstance(payload := item.get("json"), dict)
        )
    return items


def _n8n_result_data(record: Mapping[str, object]) -> dict[str, object]:
    """Return the record's ``resultData`` block, or an empty one."""
    data = record.get("data")
    if isinstance(data, dict) and isinstance(result := data.get("resultData"), dict):
        return result
    return {}


def _n8n_parse(stdout: str) -> AgenticResult:
    """Normalise one ``n8n execute --rawOutput`` run.

    ``steps`` counts the nodes that actually executed, which is the same
    "counts up with work" contract as Codex's shell calls: a workflow whose
    gateway call failed stops early and reports fewer nodes than it declares.

    Token counts stay zero: the surfaces these workflows drive are embeddings,
    audio, images and files, which the gateway does not bill in tokens.

    Raises:
        ValueError: If no execution record is present.
    """
    record = n8n_execution_record(stdout)
    result_data = _n8n_result_data(record)
    runs = result_data.get("runData")
    terminal = result_data.get("lastNodeExecuted")
    items = n8n_node_items(record, terminal) if isinstance(terminal, str) else []
    error = result_data.get("error")
    return AgenticResult(
        text=json.dumps(items),
        steps=len(runs) if isinstance(runs, dict) else 0,
        input_tokens=0,
        output_tokens=0,
        is_error=bool(error),
        error_detail=json.dumps(error)[:500] if error else "",
    )


# ---------------------------------------------------------------------------
# Haystack — the retrieval stack that reaches the Cohere Rerank route
# ---------------------------------------------------------------------------
#
# Haystack is a library rather than a CLI, so the "client" here is one committed
# script run to completion by the shared runner. It is the lane's only vehicle for
# ``/cohere/v2/rerank``: the reranker n8n ships takes no base URL, and no coding
# agent reranks anything.

#: Committed pipeline the container runs, copied into each run's working directory.
_HAYSTACK_PIPELINE = Path(__file__).parent / "rag_pipeline.py"

#: Name of that script inside the container's working directory.
_HAYSTACK_SCRIPT = "rag_pipeline.py"

#: File the pipeline record is written to, beside the script.
#:
#: The same object is printed, which is what :func:`_haystack_parse` normalises;
#: the assertions need every ranking in the record rather than the normalised
#: result, and read this file instead.
HAYSTACK_RUN_OUTPUT = "rag_run.json"

#: Name planted in the corpus and in no other document.
#:
#: Owned here rather than by the script so a test can assert on it without
#: restating the corpus: the script interpolates it into the planted document.
HAYSTACK_PLANTED_NAME = "Brindlewick relay module"


def _haystack_prepare(invocation: Invocation) -> None:
    """Copy the pipeline script into the run's working directory."""
    (invocation.workdir / _HAYSTACK_SCRIPT).write_text(
        _HAYSTACK_PIPELINE.read_text(), encoding="utf-8"
    )


def _haystack_build(invocation: Invocation) -> Command:
    """Build the ``python /work/rag_pipeline.py`` command for one invocation.

    The script takes no arguments: the gateway, the models and the question all
    arrive through the environment, which is what keeps the committed file
    identical for every run. ``EMBED_MODEL`` and ``RERANK_MODEL`` come from the
    test through :attr:`Invocation.extra_env`; the model under test is the chat
    model, because it is the one the shared identity check can attribute.
    """
    base_url = f"http://127.0.0.1:{invocation.port}"
    env = {
        "HOME": WORK_MOUNT,
        # Both routes authenticate with the same gateway key; the Cohere client
        # reads its own variable and the OpenAI client its own.
        "OPENAI_API_KEY": invocation.api_key,
        "COHERE_API_KEY": invocation.api_key,
        "STDAPI_BASE_URL": base_url,
        "CHAT_MODEL": invocation.model,
        "QUERY": invocation.prompt,
        "PLANTED_NAME": HAYSTACK_PLANTED_NAME,
        # Haystack posts an anonymous usage event per pipeline run, and the
        # container cannot resolve its host -- the same class of fix as Codex's
        # OTEL_SDK_DISABLED.
        "HAYSTACK_TELEMETRY_ENABLED": "False",
        # The script is copied into a directory the test runner owns; a __pycache__
        # written next to it would belong to the container's view of the user.
        "PYTHONDONTWRITEBYTECODE": "1",
        **invocation.extra_env,
    }
    return Command(("python", f"{WORK_MOUNT}/{_HAYSTACK_SCRIPT}"), env)


def haystack_record(workdir: Path) -> dict[str, object]:
    """Return the record the run left in the working directory.

    Args:
        workdir: Per-test directory bind-mounted at :data:`WORK_MOUNT`.

    Returns:
        The decoded record.

    Raises:
        ValueError: If the run wrote no record, which is what a component
            failing before the end of the pipeline looks like.
    """
    return _haystack_decode((workdir / HAYSTACK_RUN_OUTPUT).read_text())


def _haystack_decode(stdout: str) -> dict[str, object]:
    """Return the JSON record in *stdout*, taking the last object it holds.

    Args:
        stdout: Text holding the record on a line of its own.

    Returns:
        The decoded record.

    Raises:
        ValueError: If no record is present.
    """
    for line in reversed(stdout.splitlines()):
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            return record
    msg = f"the pipeline printed no JSON record\nOutput: {stdout[-500:]}"
    raise ValueError(msg)


def _haystack_parse(stdout: str) -> AgenticResult:
    """Normalise one pipeline run.

    ``steps`` counts the components that produced output, which is the same
    "counts up with work" contract as Codex's shell calls: a run whose gateway
    call failed stops at that component and reports fewer than the pipeline
    declares.

    Raises:
        ValueError: If the script printed no record.
    """
    record = _haystack_decode(stdout)
    components = record.get("components")
    usage = record.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return AgenticResult(
        text=str(record.get("answer", "")),
        steps=len(components) if isinstance(components, list) else 0,
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
    )


# ---------------------------------------------------------------------------
# OpenClaw — the wire format as a single flag
# ---------------------------------------------------------------------------
#
# OpenClaw is the only client in the lane that picks its wire format from one
# switch: ``--custom-compatibility openai | openai-responses | anthropic`` on the
# onboarding command decides which of the gateway's three chat routes the very
# same provider declaration talks to. pi and Hermes reach the same routes, but
# each through its own provider entry; here the entry is identical and only the
# switch moves.
#
# The run is two commands: onboarding writes the provider (that is where the
# switch lives), then one embedded agent turn runs against it. ``--local`` keeps
# that turn in-process, so no Gateway daemon is ever started.

#: File the agent's prompt is written to, beside the run's config.
#:
#: ``--message-file`` rather than ``--message``: the prompt is multi-line and the
#: run is driven through ``sh -c``, where a positional argument would have to be
#: re-quoted.
_OPENCLAW_PROMPT_FILE = "prompt.md"

#: Config patch narrowing the agent's tools, applied after onboarding.
#:
#: ``config patch`` merges recursively, so this only has to name what differs from
#: the config onboarding just wrote.
_OPENCLAW_POLICY_FILE = "openclaw-policy.json"

#: File the run's JSON envelope is written to, then echoed to stdout.
#:
#: The CLI prints its whole envelope in one write and exits, and a large write to
#: a pipe is asynchronous in Node: the tail is lost. A write to a file is
#: synchronous -- the same fix as the Qwen Code and n8n runs above.
OPENCLAW_RUN_OUTPUT = "run.json"

#: Provider id the onboarding command registers the gateway under.
#:
#: Pinned rather than left to be derived from the base URL, so ``--model`` can
#: name ``<provider>/<model id>`` without guessing what the derivation produced.
_OPENCLAW_PROVIDER_ID = "stdapi"

#: Seconds one embedded agent turn may run before the CLI ends it itself.
#:
#: Below every model timeout in this lane, so a stuck run still exits with a
#: parsable envelope instead of being killed mid-write by the container timeout.
_OPENCLAW_TIMEOUT = "1200"

#: Directories seeded in the working directory before the run.
#:
#: Created on the host so they belong to the test runner rather than to the
#: container, matching what the n8n runs and the service containers do.
_OPENCLAW_WORK_DIRS = ("home", "openclaw", "workspace")

#: Tool policy for the agent turn: read files, run commands, nothing else.
#:
#: ``profile: "minimal"`` is the narrowest base OpenClaw ships (``session_status``
#: alone); ``alsoAllow`` adds back exactly the tools the task needs, and ``deny``
#: restates the mutating and outbound ones because deny wins over everything.
#: ``alsoAllow`` rather than ``allow``: a configured tool section no longer widens
#: a profile implicitly, so ``allow`` alone leaves the agent with no tools at all.
#: ``process`` rides along because OpenClaw refuses ``exec`` without it.
#:
#: ``exec.mode: "full"`` runs host commands without an approval prompt. That is
#: required, not incidental: a headless run has no UI to answer a prompt with, so
#: any other mode either blocks every command or hangs. It does NOT mean the run
#: is unsandboxed -- the podman container is the boundary (no capabilities, no new
#: privileges, read-only root, source mounts read-only, ``/work`` the only
#: writable path, own network namespace with a single forwarded port). Never run
#: this configuration outside that container.
#:
#: ``sandbox.mode: "off"`` keeps OpenClaw's own sandbox backends out of the
#: picture: its alternatives route tool execution into a Docker container, an SSH
#: target or a remote shell, all of which would need a socket or a credential this
#: run must never be handed, and all of which would execute code outside the
#: boundary above.
_OPENCLAW_POLICY: dict[str, object] = {
    "tools": {
        "profile": "minimal",
        "alsoAllow": ["read", "exec", "process"],
        "deny": [
            "write",
            "edit",
            "apply_patch",
            "code_execution",
            "group:web",
            "group:ui",
            "group:messaging",
            "group:media",
            "group:nodes",
            "group:automation",
            "group:sessions",
        ],
        "exec": {"mode": "full"},
        "elevated": {"enabled": False},
    },
    "agents": {"defaults": {"sandbox": {"mode": "off"}}},
}

#: Shell driving one OpenClaw run: onboard, narrow the tools, take one turn.
#:
#: Onboarding's own output is kept in a file because it is long and, on failure,
#: is the only explanation of what went wrong; it is echoed only then.
_OPENCLAW_SCRIPT = f"""\
set -e
openclaw onboard --non-interactive --accept-risk \
    --auth-choice custom-api-key \
    --secret-input-mode plaintext \
    --custom-provider-id {_OPENCLAW_PROVIDER_ID} \
    --custom-base-url "$OPENCLAW_BASE_URL" \
    --custom-model-id "$OPENCLAW_MODEL" \
    --custom-compatibility "$OPENCLAW_COMPATIBILITY" \
    --skip-health --no-install-daemon --skip-bootstrap --skip-skills \
    --skip-ui --skip-hooks --skip-channels --skip-search \
    --suppress-gateway-token-output \
    > {WORK_MOUNT}/onboard.log 2>&1 || {{ cat {WORK_MOUNT}/onboard.log; exit 1; }}
openclaw config patch --file {WORK_MOUNT}/{_OPENCLAW_POLICY_FILE} \
    >> {WORK_MOUNT}/onboard.log 2>&1 || {{ cat {WORK_MOUNT}/onboard.log; exit 1; }}
openclaw agent --local --agent main \
    --model "{_OPENCLAW_PROVIDER_ID}/$OPENCLAW_MODEL" \
    --message-file {WORK_MOUNT}/{_OPENCLAW_PROMPT_FILE} \
    --timeout {_OPENCLAW_TIMEOUT} --json \
    > {WORK_MOUNT}/{OPENCLAW_RUN_OUTPUT} \
    || {{ cat {WORK_MOUNT}/{OPENCLAW_RUN_OUTPUT}; exit 1; }}
cat {WORK_MOUNT}/{OPENCLAW_RUN_OUTPUT}
"""


def _openclaw_prepare(invocation: Invocation) -> None:
    """Seed OpenClaw's state directories, its prompt file and its tool policy."""
    workdir = invocation.workdir
    for name in _OPENCLAW_WORK_DIRS:
        (workdir / name).mkdir(exist_ok=True)
    (workdir / _OPENCLAW_PROMPT_FILE).write_text(invocation.prompt, encoding="utf-8")
    (workdir / _OPENCLAW_POLICY_FILE).write_text(json.dumps(_OPENCLAW_POLICY, indent=2))


def _openclaw_build(compatibility: str, route: str) -> Callable[[Invocation], Command]:
    """Return a ``build`` bound to one compatibility value and its gateway route.

    The gateway's API key reaches the CLI as ``CUSTOM_API_KEY``, the environment
    variable onboarding falls back to when no key flag is passed, and
    ``--secret-input-mode plaintext`` writes it into the config OpenClaw builds.
    ``ref`` would store a reference instead, but resolving one goes through
    OpenClaw's own Gateway daemon over a WebSocket, which a one-shot container
    run has no reason to start. The key still never reaches a command line, and
    the config holding it is inside the run's throwaway workdir.

    Args:
        compatibility: ``openai``, ``openai-responses`` or ``anthropic``.
        route: Gateway path prefix serving that wire format.

    Returns:
        A ``build`` callable for an :class:`AgenticTool`.
    """

    def build(invocation: Invocation) -> Command:
        env = {
            "HOME": f"{WORK_MOUNT}/home",
            # Every path OpenClaw persists to is redirected under the writable
            # per-test mount: nothing is written into the image, a shared cache or
            # the host's real home.
            "OPENCLAW_STATE_DIR": f"{WORK_MOUNT}/openclaw",
            "OPENCLAW_CONFIG_PATH": f"{WORK_MOUNT}/openclaw/openclaw.json",
            "OPENCLAW_WORKSPACE_DIR": f"{WORK_MOUNT}/workspace",
            "CUSTOM_API_KEY": invocation.api_key,
            "OPENCLAW_BASE_URL": f"http://127.0.0.1:{invocation.port}{route}",
            "OPENCLAW_MODEL": invocation.model,
            "OPENCLAW_COMPATIBILITY": compatibility,
            **invocation.extra_env,
        }
        return Command(("sh", "-c", _OPENCLAW_SCRIPT), env)

    return build


def _openclaw_parse(stdout: str) -> AgenticResult:
    """Normalise one ``openclaw agent --json`` envelope.

    ``steps`` counts the tool calls the turn made, which is the same "counts up
    with work" contract as Codex's shell calls: an agent that answered from its
    own knowledge never opened a file and reports none.

    Raises:
        ValueError: If the output holds no JSON envelope.
    """
    data = _openclaw_envelope(stdout)
    meta = data.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    agent_meta = meta.get("agentMeta")
    agent_meta = agent_meta if isinstance(agent_meta, dict) else {}
    usage = agent_meta.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    tool_summary = meta.get("toolSummary")
    tool_summary = tool_summary if isinstance(tool_summary, dict) else {}

    raw_payloads = data.get("payloads")
    payloads = [
        item
        for item in (raw_payloads if isinstance(raw_payloads, list) else [])
        if isinstance(item, dict)
    ]
    texts = [
        text
        for item in payloads
        if not item.get("isError") and (text := str(item.get("text") or "").strip())
    ]
    error = meta.get("error")
    failed = [item for item in payloads if item.get("isError")]
    return AgenticResult(
        text="\n\n".join(texts),
        steps=int(tool_summary.get("calls") or 0),
        input_tokens=int(usage.get("input") or 0),
        output_tokens=int(usage.get("output") or 0),
        cache_read=int(usage.get("cacheRead") or 0),
        cache_created=int(usage.get("cacheWrite") or 0),
        is_error=bool(error or failed),
        error_detail=json.dumps(error or failed)[:500] if (error or failed) else "",
    )


def _openclaw_envelope(stdout: str) -> dict[str, object]:
    """Return the JSON envelope ``openclaw agent --json`` printed.

    ``--json`` reserves stdout for the envelope and sends diagnostics to stderr,
    but the failure path echoes the onboarding log first, so the object is located
    rather than assumed to be the whole stream.

    Args:
        stdout: Everything the container wrote to stdout.

    Returns:
        The decoded envelope.

    Raises:
        ValueError: If no envelope is present.
    """
    decoder = json.JSONDecoder()
    for index, char in enumerate(stdout):
        if char != "{":
            continue
        try:
            envelope, _ = decoder.raw_decode(stdout, index)
        except json.JSONDecodeError:
            continue
        if isinstance(envelope, dict) and "meta" in envelope:
            return envelope
    msg = f"openclaw printed no JSON envelope\nOutput: {stdout[-500:]}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Hermes — three transports, and the lane's only client-chosen cache TTL
# ---------------------------------------------------------------------------
#
# Hermes selects its wire format with ``transport`` on the provider entry of its
# ``config.yaml``, so the same one-shot run covers all three chat routes. It is
# also the only client here that puts explicit Anthropic prompt-caching
# breakpoints on the wire: for a Claude-named model on the Anthropic transport it
# marks the system prompt and the last three messages with ``cache_control`` at
# the tier named by ``prompt_caching.cache_ttl``, spelling out ``ttl: "1h"`` for
# the long tier and leaving it implicit for the 5-minute default.
#
# Being a PyPI package, it gets its own image group rather than the shared
# Node.js one.

#: Environment variable Hermes' provider entry names as its API key source.
_HERMES_API_KEY_VAR = "STDAPI_API_KEY"

#: Environment variable a test sets to pick the Anthropic cache TTL tier.
#:
#: Read out of :attr:`Invocation.extra_env` by :func:`_hermes_prepare`, because the
#: tier is a ``config.yaml`` value rather than a command-line flag. Anthropic
#: supports ``5m`` and ``1h``; Hermes ignores anything else.
HERMES_CACHE_TTL_VAR = "HERMES_PROMPT_CACHE_TTL"

#: Toolset the agent is given: file reads and searches, nothing else.
#:
#: The narrowest set that still drives a tool round trip through the gateway. It
#: excludes ``terminal`` (shell and process control), ``browser``, ``web`` and
#: ``delegation``, so the run has no way to execute a command or reach the network
#: beyond the one forwarded gateway port.
_HERMES_TOOLSETS = "file"

#: Tool-calling iterations one turn may take, as a runaway-loop backstop.
_HERMES_MAX_TURNS = 30

#: Output token ceiling requested per call, as for pi (:data:`_PI_MAX_TOKENS`).
#:
#: Left to itself Hermes asks for more than the model allows and the request is
#: refused before it starts: "the maximum tokens you requested exceeds the model
#: limit of 64000" on Claude Haiku 4.5, 65535 on Nova 2 Lite. Well under every
#: limit in the lane, and far above what these prompts generate.
_HERMES_MAX_TOKENS = 32_768

#: File the final answer is written to, then echoed to stdout.
#:
#: ``-z`` prints the final response text and nothing else, so the answer and the
#: usage report have to be recombined for the shared parse contract.
_HERMES_ANSWER_FILE = "hermes_answer.txt"

#: File ``--usage-file`` writes its JSON token report to.
_HERMES_USAGE_FILE = "hermes_usage.json"

#: Separator between the answer text and the usage report on stdout.
_HERMES_USAGE_MARKER = "----- hermes usage -----"

#: Shell running the CLI, then appending its usage report. ``"$@"`` avoids
#: re-quoting the prompt, as in the Qwen Code run above.
_HERMES_SCRIPT = (
    f'"$@" > {WORK_MOUNT}/{_HERMES_ANSWER_FILE}\n'
    "status=$?\n"
    f"cat {WORK_MOUNT}/{_HERMES_ANSWER_FILE}\n"
    f"printf '\\n%s\\n' '{_HERMES_USAGE_MARKER}'\n"
    f"cat {WORK_MOUNT}/{_HERMES_USAGE_FILE} 2>/dev/null || printf '{{}}\\n'\n"
    "exit $status\n"
)


def _hermes_prepare(transport: str, route: str) -> Callable[[Invocation], None]:
    """Return a ``prepare_workdir`` writing Hermes' config for one transport.

    The file is emitted as JSON, which is a subset of YAML: it needs no YAML
    dependency in the test process and cannot trip the quoting traps that make
    hand-written Hermes configs fragile (``off`` and ``5m`` are a boolean and a
    string to a YAML parser respectively).

    Approvals are left alone deliberately: ``-z`` already bypasses them for its
    own run, and writing ``approvals.mode: off`` would only widen that beyond the
    one-shot path.

    Args:
        transport: ``chat_completions``, ``anthropic_messages`` or
            ``codex_responses``.
        route: Gateway path prefix serving that transport.

    Returns:
        A ``prepare_workdir`` callable for an :class:`AgenticTool`.
    """

    def prepare(invocation: Invocation) -> None:
        home = invocation.workdir / "home"
        home.mkdir(exist_ok=True)
        hermes_home = invocation.workdir / "hermes"
        hermes_home.mkdir(exist_ok=True)
        config: dict[str, object] = {
            "providers": {
                "stdapi": {
                    "name": "stdapi.ai",
                    "api": f"http://127.0.0.1:{invocation.port}{route}",
                    "key_env": _HERMES_API_KEY_VAR,
                    "transport": transport,
                    "default_model": invocation.model,
                    "max_tokens": _HERMES_MAX_TOKENS,
                }
            },
            # The ceiling is set on both blocks: which one a given transport
            # reads is not documented, and setting the other is inert.
            "model": {
                "provider": "stdapi",
                "model": invocation.model,
                "max_tokens": _HERMES_MAX_TOKENS,
            },
            "agent": {"max_turns": _HERMES_MAX_TURNS},
            # Several optional code paths pip-install on demand; the container
            # cannot reach an index, so a lazy install would fail slowly instead
            # of the feature degrading immediately.
            "security": {"allow_lazy_installs": False},
            "prompt_caching": {
                "cache_ttl": invocation.extra_env.get(HERMES_CACHE_TTL_VAR, "5m")
            },
        }
        (hermes_home / "config.yaml").write_text(json.dumps(config, indent=2))

    return prepare


def _hermes_build(invocation: Invocation) -> Command:
    """Build the ``hermes -z`` one-shot command for one invocation.

    ``HERMES_HOME`` moves the whole state directory -- config, sessions, memory
    and their SQLite files -- under the writable per-test mount, so nothing the
    agent persists lands in the image or in the host's home.

    The CLI runs under :data:`_HERMES_SCRIPT` so the usage report reaches the test
    alongside the answer: ``-z`` prints the final response text alone.
    """
    env = {
        "HOME": f"{WORK_MOUNT}/home",
        "HERMES_HOME": f"{WORK_MOUNT}/hermes",
        _HERMES_API_KEY_VAR: invocation.api_key,
        # The agent's own working directory belongs to the test runner; a
        # __pycache__ written next to it would belong to the container's user.
        "PYTHONDONTWRITEBYTECODE": "1",
        **invocation.extra_env,
    }
    argv = [
        "hermes",
        "-z",
        invocation.prompt,
        "--model",
        invocation.model,
        "--provider",
        "stdapi",
        "--toolsets",
        _HERMES_TOOLSETS,
        "--usage-file",
        f"{WORK_MOUNT}/{_HERMES_USAGE_FILE}",
        *invocation.extra_args,
    ]
    return Command(("sh", "-c", _HERMES_SCRIPT, "sh", *argv), env)


def _hermes_parse(stdout: str) -> AgenticResult:
    """Normalise one ``hermes -z`` run and the usage report beside it.

    ``steps`` counts the model API calls the run made, which is the same "counts
    up with work" contract as Codex's shell calls: a turn that read a file needs a
    second call to say what it found, so a run that answered from memory reports
    one and fails the floor.

    Raises:
        ValueError: If the run emitted no usage report, which is what the CLI
            failing before it produced an answer looks like.
    """
    answer, marker, trailer = stdout.partition(_HERMES_USAGE_MARKER)
    if not marker:
        msg = f"hermes printed no usage report\nOutput: {stdout[-500:]}"
        raise ValueError(msg)
    try:
        decoded = json.loads(trailer.strip() or "{}")
    except json.JSONDecodeError as exc:
        msg = f"hermes usage report is not valid JSON: {exc}\nOutput: {trailer[:500]}"
        raise ValueError(msg) from exc
    if not isinstance(decoded, dict):
        # ValueError, not TypeError: this is malformed CLI output, not a bad
        # argument, and ValueError is the parse contract run_agent handles.
        msg = f"hermes usage report is not a JSON object\nOutput: {trailer[:500]}"
        raise ValueError(msg)  # noqa: TRY004
    report = decoded
    failure = report.get("failure")
    return AgenticResult(
        text=answer.strip(),
        steps=int(report.get("api_calls") or 0),
        input_tokens=int(report.get("input_tokens") or 0),
        output_tokens=int(report.get("output_tokens") or 0),
        cache_read=int(report.get("cache_read_tokens") or 0),
        cache_created=int(report.get("cache_write_tokens") or 0),
        is_error=bool(report.get("failed")),
        error_detail=str(failure)[:500] if failure else "",
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

#: npm specifier shared by every pi route entry.
_PI_PACKAGE = "@earendil-works/pi-coding-agent@latest"


def _pi_tool(suffix: str, api: str, route: str, metrics_prefix: str) -> AgenticTool:
    """Build one pi entry bound to a single gateway route.

    Args:
        suffix: Route discriminator appended to the tool id.
        api: pi provider API kind.
        route: Gateway path prefix serving it.
        metrics_prefix: Tag opening this entry's ``-s`` benchmark lines.

    Returns:
        The configured tool.
    """
    return AgenticTool(
        id=f"pi-{suffix}",
        npm_package=_PI_PACKAGE,
        binary="pi",
        route=route,
        metrics_prefix=metrics_prefix,
        build=_pi_build,
        parse=_pi_parse,
        prepare_workdir=_pi_prepare(api, route),
        # pi sends no per-run identifier the gateway records, so its requests can
        # only be attributed positionally.
        attributes_sessions=False,
    )


PI_CHAT_COMPLETIONS = _pi_tool(
    "chat", "openai-completions", "/v1", metrics_prefix="PI-CC-METRICS"
)

PI_RESPONSES = _pi_tool(
    "responses", "openai-responses", "/v1", metrics_prefix="PI-RS-METRICS"
)

PI_MESSAGES = _pi_tool(
    "messages", "anthropic-messages", "/anthropic", metrics_prefix="PI-MG-METRICS"
)

#: npm specifier shared by every OpenClaw entry.
_OPENCLAW_PACKAGE = "openclaw@latest"


def _openclaw_tool(
    suffix: str, compatibility: str, route: str, metrics_prefix: str
) -> AgenticTool:
    """Build one OpenClaw entry bound to a single compatibility value.

    Args:
        suffix: Wire-format discriminator appended to the tool id.
        compatibility: ``--custom-compatibility`` value.
        route: Gateway path prefix serving that wire format.
        metrics_prefix: Tag opening this entry's ``-s`` benchmark lines.

    Returns:
        The configured tool.
    """
    return AgenticTool(
        id=f"openclaw-{suffix}",
        npm_package=_OPENCLAW_PACKAGE,
        binary="openclaw",
        route=route,
        metrics_prefix=metrics_prefix,
        build=_openclaw_build(compatibility, route),
        parse=_openclaw_parse,
        prepare_workdir=_openclaw_prepare,
        # OpenClaw sends no per-run identifier the gateway records, so its
        # requests can only be attributed positionally.
        attributes_sessions=False,
    )


OPENCLAW_OPENAI = _openclaw_tool(
    "openai", "openai", "/v1", metrics_prefix="OC-CC-METRICS"
)

OPENCLAW_RESPONSES = _openclaw_tool(
    "responses", "openai-responses", "/v1", metrics_prefix="OC-RS-METRICS"
)

OPENCLAW_ANTHROPIC = _openclaw_tool(
    "anthropic", "anthropic", "/anthropic", metrics_prefix="OC-MG-METRICS"
)


def _hermes_tool(
    suffix: str, transport: str, route: str, metrics_prefix: str
) -> AgenticTool:
    """Build one Hermes entry bound to a single transport.

    Args:
        suffix: Wire-format discriminator appended to the tool id.
        transport: ``providers.stdapi.transport`` value.
        route: Gateway path prefix serving that transport.
        metrics_prefix: Tag opening this entry's ``-s`` benchmark lines.

    Returns:
        The configured tool.
    """
    return AgenticTool(
        id=f"hermes-{suffix}",
        # Its image ships hermes-agent itself, installed by Containerfile.hermes,
        # so there is no npm package to add to any image.
        npm_package=None,
        binary="hermes",
        route=route,
        metrics_prefix=metrics_prefix,
        build=_hermes_build,
        parse=_hermes_parse,
        prepare_workdir=_hermes_prepare(transport, route),
        # Hermes sends no per-run identifier the gateway records, so its requests
        # can only be attributed positionally.
        attributes_sessions=False,
        image_group=HERMES_IMAGE_GROUP,
    )


HERMES_CHAT_COMPLETIONS = _hermes_tool(
    "chat", "chat_completions", "/v1", metrics_prefix="HM-CC-METRICS"
)

HERMES_RESPONSES = _hermes_tool(
    "responses", "codex_responses", "/v1", metrics_prefix="HM-RS-METRICS"
)

HERMES_ANTHROPIC = _hermes_tool(
    "messages", "anthropic_messages", "/anthropic", metrics_prefix="HM-MG-METRICS"
)


def _n8n_tool(surface: str, route: str = "/v1") -> AgenticTool:
    """Build one n8n entry bound to a single workflow template.

    Args:
        surface: Template stem under ``tests/agentic/workflows/``, also the tool
            id's suffix.
        route: Gateway path prefix the template's node talks to.

    Returns:
        The configured tool.
    """
    return AgenticTool(
        id=f"n8n-{surface}",
        npm_package=_N8N_PACKAGE,
        binary="n8n",
        route=route,
        metrics_prefix="N8N-METRICS",
        build=_n8n_build,
        parse=_n8n_parse,
        prepare_workdir=_n8n_prepare(surface),
        # n8n sends no per-run identifier the gateway records, so its requests
        # can only be attributed positionally.
        attributes_sessions=False,
        image_group=N8N_IMAGE_GROUP,
    )


N8N_MODERATIONS = _n8n_tool("moderations")

N8N_EMBEDDINGS = _n8n_tool("embeddings")

N8N_COMPLETIONS = _n8n_tool("completions")

N8N_CHAT_COMPLETIONS = _n8n_tool("chat_completions")

N8N_RESPONSES = _n8n_tool("responses")

N8N_MESSAGES = _n8n_tool("messages", route="/anthropic")

N8N_SPEECH = _n8n_tool("speech")

N8N_TRANSCRIPTIONS = _n8n_tool("transcriptions")

N8N_TRANSLATIONS = _n8n_tool("translations")

N8N_FILES = _n8n_tool("files")

N8N_IMAGES_GENERATIONS = _n8n_tool("images_generations")

N8N_IMAGES_EDITS = _n8n_tool("images_edits")

N8N_VIDEOS = _n8n_tool("videos")

QWEN_CODE = AgenticTool(
    id="qwen-code",
    npm_package="@qwen-code/qwen-code@latest",
    binary="qwen",
    route="/v1",
    metrics_prefix="QW-METRICS",
    build=_qwen_build,
    parse=_qwen_parse,
    prepare_workdir=_qwen_prepare,
    # Qwen Code sends no per-run identifier the gateway records, so its requests
    # can only be attributed positionally.
    attributes_sessions=False,
)

# inspect-ai, the lane's only client for either Batch API. Its evaluation is a
# committed script rather than a CLI invocation, for the same reason Haystack's
# pipeline is: the client is a library, and rendering it per run would make the
# file unreadable to ruff and mypy.

#: Committed evaluation the container runs, copied into each run's working directory.
_INSPECT_EVAL = Path(__file__).parent / "inspect_eval.py"

#: Name of that script inside the container's working directory.
_INSPECT_SCRIPT = "inspect_eval.py"

#: File the evaluation record is written to, beside the script.
#:
#: The same object is printed, which is what :func:`_inspect_parse` normalises;
#: the assertions need every sample in the record rather than the normalised
#: result, and read this file instead.
INSPECT_RUN_OUTPUT = "inspect_run.json"

#: Tokens one batched answer may take; every sample asks for a single short line.
_INSPECT_MAX_TOKENS = 32

#: File the run's stdout is written to before it is printed back.
#:
#: The evaluation waits tens of minutes for a batch and prints its progress and
#: every retry as it goes, but ``subprocess.run(capture_output=True)`` discards
#: what it buffered when the deadline kills the run -- so a stalled run reported
#: nothing at all. Writing into the mounted directory first leaves that output on
#: the host either way.
_INSPECT_STDOUT = "inspect_stdout.log"

#: File the run's stderr is written to, for the same reason.
_INSPECT_STDERR = "inspect_stderr.log"


def _inspect_prepare(invocation: Invocation) -> None:
    """Copy the evaluation script into the run's working directory."""
    (invocation.workdir / _INSPECT_SCRIPT).write_text(
        _INSPECT_EVAL.read_text(), encoding="utf-8"
    )


def _inspect_build(invocation: Invocation) -> Command:
    """Build the ``python /work/inspect_eval.py`` command for one invocation.

    The script takes no arguments: the gateway, the provider, the model and the
    batch's shape all arrive through the environment, which is what keeps the
    committed file identical for every run. ``INSPECT_PROVIDER``,
    ``INSPECT_ROUTE``, ``BATCH_SIZE`` and any ``MODEL_ARGS`` come from the test
    through :attr:`Invocation.extra_env`: the first two are what distinguish the
    two batch surfaces, and the batch size is the gateway's own minimum, which
    this module deliberately does not import.

    Both streams are written into the working directory unbuffered and printed
    back afterwards, so the client's own log survives a run the caller's deadline
    kills -- the one failure mode a batch that waits an hour actually has. The
    exit status is carried across the redirection, since it is what the runner
    reports a failed client by.
    """
    return Command(
        (
            "sh",
            "-c",
            (
                'python -u "$0" > "$1" 2> "$2"; status=$?; '
                'cat "$1"; cat "$2" >&2; exit $status'
            ),
            f"{WORK_MOUNT}/{_INSPECT_SCRIPT}",
            f"{WORK_MOUNT}/{_INSPECT_STDOUT}",
            f"{WORK_MOUNT}/{_INSPECT_STDERR}",
        ),
        {
            "HOME": WORK_MOUNT,
            "STDAPI_BASE_URL": f"http://127.0.0.1:{invocation.port}",
            "STDAPI_API_KEY": invocation.api_key,
            "CHAT_MODEL": invocation.model,
            # The prompt is the reference every sample asks to have repeated;
            # the test owns it, because the test is what asserts on it.
            "MARKER": invocation.prompt,
            "MAX_TOKENS": str(_INSPECT_MAX_TOKENS),
            # The framework's live display redraws a terminal it does not have,
            # which fills the captured output with control sequences.
            "INSPECT_DISPLAY": "plain",
            # The script is copied into a directory the test runner owns; a
            # __pycache__ written next to it would belong to the container's view
            # of the user.
            "PYTHONDONTWRITEBYTECODE": "1",
            **invocation.extra_env,
        },
    )


def inspect_record(workdir: Path) -> dict[str, object]:
    """Return the record the evaluation left in the working directory.

    Args:
        workdir: Per-test directory bind-mounted at :data:`WORK_MOUNT`.

    Returns:
        The decoded record.

    Raises:
        ValueError: If the run wrote no record, which is what an evaluation
            failing before it finished looks like.
    """
    # Shared with the Haystack pipeline: both scripts print one JSON object on a
    # line of its own, which is the whole contract the decoder depends on.
    return _haystack_decode((workdir / INSPECT_RUN_OUTPUT).read_text())


def _inspect_parse(stdout: str) -> AgenticResult:
    """Normalise one evaluation run.

    ``steps`` counts the samples that came back with an answer, which is the same
    "counts up with work" contract as Codex's shell calls: a batch that failed
    part way answers fewer samples than the dataset declares.

    Raises:
        ValueError: If the script printed no record.
    """
    record = _haystack_decode(stdout)
    samples = record.get("samples")
    samples = samples if isinstance(samples, list) else []
    usage = record.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    answered = [
        str(sample.get("completion") or "")
        for sample in samples
        if isinstance(sample, dict) and str(sample.get("completion") or "").strip()
    ]
    return AgenticResult(
        # Every answer, not just the first: the shared assertion looks for the
        # reference the requests asked to have repeated, and each sample carries
        # its own.
        text=" ".join(answered),
        steps=len(answered),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
    )


HAYSTACK = AgenticTool(
    id="haystack",
    # Its image ships Haystack itself; the Cohere integration is installed by
    # Containerfile.rag, so there is no npm package to add to any image.
    npm_package=None,
    binary="python",
    route="/v1",
    metrics_prefix="HS-METRICS",
    build=_haystack_build,
    parse=_haystack_parse,
    prepare_workdir=_haystack_prepare,
    # Neither client sends a per-run identifier the gateway records, so the
    # pipeline's requests can only be attributed positionally.
    attributes_sessions=False,
    image_group=RAG_IMAGE_GROUP,
)

INSPECT_AI = AgenticTool(
    id="inspect-ai",
    # A PyPI package, installed by Containerfile.inspect; nothing for npm to add
    # to any image.
    npm_package=None,
    binary="python",
    route="/v1",
    metrics_prefix="IA-METRICS",
    build=_inspect_build,
    parse=_inspect_parse,
    prepare_workdir=_inspect_prepare,
    # The framework sends no per-run identifier the gateway records, so its
    # requests can only be attributed positionally.
    attributes_sessions=False,
    image_group=INSPECT_IMAGE_GROUP,
)

#: Every agentic CLI the suite knows how to drive.
AGENTIC_TOOLS: tuple[AgenticTool, ...] = (
    CLAUDE_CODE,
    CODEX,
    PI_CHAT_COMPLETIONS,
    PI_RESPONSES,
    PI_MESSAGES,
    OPENCLAW_OPENAI,
    OPENCLAW_RESPONSES,
    OPENCLAW_ANTHROPIC,
    HERMES_CHAT_COMPLETIONS,
    HERMES_RESPONSES,
    HERMES_ANTHROPIC,
    N8N_MODERATIONS,
    N8N_EMBEDDINGS,
    N8N_COMPLETIONS,
    N8N_CHAT_COMPLETIONS,
    N8N_RESPONSES,
    N8N_MESSAGES,
    N8N_SPEECH,
    N8N_TRANSCRIPTIONS,
    N8N_TRANSLATIONS,
    N8N_FILES,
    N8N_IMAGES_GENERATIONS,
    N8N_IMAGES_EDITS,
    N8N_VIDEOS,
    QWEN_CODE,
    HAYSTACK,
    INSPECT_AI,
)


def npm_packages(
    group: str = DEFAULT_IMAGE_GROUP, tools: Sequence[AgenticTool] = AGENTIC_TOOLS
) -> tuple[str, ...]:
    """Return the npm specifiers to install into *group*'s container image.

    Args:
        group: Image group to collect packages for.
        tools: Registry to collect them from.

    Returns:
        The group's npm specifiers, deduplicated and sorted; empty for a group
        whose tools ship with their image.
    """
    return tuple(
        sorted(
            {
                tool.npm_package
                for tool in tools
                if tool.image_group == group and tool.npm_package
            }
        )
    )
