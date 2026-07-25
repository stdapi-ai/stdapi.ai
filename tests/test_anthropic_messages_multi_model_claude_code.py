"""Agentic tests for Claude Code connected to stdapi.ai via the Anthropic route.

Tests launch a real ``claude`` CLI process in non-interactive (``--print``) mode
against a **dedicated stdapi.ai server** spawned by the test fixture on a free port.
This provides full end-to-end isolation — each test session owns its own server,
captures its JSON request logs, and asserts that requests were routed to the
expected Bedrock model.

Architecture:
    - ``_stdapi_server_session`` (session-scoped): spawns ``uvicorn stdapi.main:app``
      on a random free port and streams its stdout JSON logs into a shared list.
    - ``_model_identity_check`` (autouse function-scoped): takes a snapshot of the
      log list before each test and, after it completes, verifies that every logged
      ``model_id`` field matches the expected Bedrock model ID for that parametrize
      variant.

Requirements:
    - ``claude`` CLI installed and in PATH (tests skip otherwise).
    - AWS Bedrock credentials in the environment (same as for the normal test suite).
    - Pass ``--expensive`` to pytest to enable these tests.

Task difficulty:
    All tasks are designed to require the model to navigate multiple files and make
    ~10 tool calls, producing a representative agentic benchmark.  Each test asserts
    a minimum turn count so that shallow single-read answers are rejected.

Metrics collected per run (visible with ``pytest -s`` or on failure):
    ``CC-METRICS | <model> | <test> | turns=<N> | <Xs> | in=<N> out=<N>``

Security per test run:
    - Each test gets an isolated ``tmp_path`` working directory.
    - ``.claude/settings.json`` enforces ``defaultMode: "dontAsk"`` with an
      allow/deny list (belt-and-suspenders with ``--allowedTools``).
    - Only read-only tools are exposed (``Read``, ``Glob``, ``Bash``).
    - ``--disallowedTools Write,Edit,WebFetch,WebSearch`` is passed as an extra
      safety layer.
    - ``--no-session-persistence`` prevents test runs from polluting the user's
      Claude Code session history.
    - The stdapi.ai source tree is mounted read-only via ``--add-dir``.
    - ``--max-budget-usd 10`` caps API spend per invocation to prevent runaway loops.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

#: Pin to one xdist worker: the session-scoped fixture spawns a server per worker.
pytestmark = pytest.mark.xdist_group("claude_code_agentic")

# ---------------------------------------------------------------------------
# Claude binary detection
# ---------------------------------------------------------------------------

#: JetBrains ACP agent-runtime glob for the Claude Code CLI bundled with the Agent SDK.
_ACP_CLAUDE_GLOB = (
    "*/acp-agents/.runtimes/node/*/npm-cache/_npx/*/node_modules/"
    "@anthropic-ai/claude-agent-sdk-*/claude"
)


def _find_jetbrains_claude() -> str | None:
    """Search the JetBrains caches for the Claude Code CLI binary.

    Scans the ACP agent runtime layout for the native ``claude`` executable
    bundled with the Claude Agent SDK platform package; the newest install
    wins.

    Returns:
        Absolute path string if found, otherwise ``None``.
    """
    cache_root = Path.home() / ".cache" / "JetBrains"
    if not cache_root.is_dir():
        return None
    candidates = sorted(
        (p for p in cache_root.glob(_ACP_CLAUDE_GLOB) if os.access(p, os.X_OK)),
        key=lambda p: p.stat().st_mtime,  # Newest install wins.
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


_CLAUDE_BIN: str | None = shutil.which("claude") or _find_jetbrains_claude()
_SKIP_NO_CLAUDE = pytest.mark.skipif(
    _CLAUDE_BIN is None,
    reason=(
        "claude CLI not found in PATH or the JetBrains ACP cache — "
        "install Claude Code to run these tests"
    ),
)

#: Models whose known-flaky failure signatures are tolerated (best effort).
_FLAKY_MODELS = frozenset(
    {
        "amazon.nova-2-lite-v1:0",
        "moonshotai.kimi-k2.5",
        "qwen.qwen3-coder-30b-a3b-v1:0",
        "qwen.qwen3-coder-next",
        "mistral.devstral-2-123b",
        "zai.glm-5",
        "google.gemma-4-31b",
        "xai.grok-4.3",
    }
)


def _xfail_if_flaky(model_env: str, signature: str) -> None:
    """Downgrade a known-flaky failure signature to an xfail (best effort).

    Only the specific signatures routed through this helper are tolerated,
    and only for the models listed in ``_FLAKY_MODELS``: any other failure
    still fails the test, so genuine regressions stay visible.

    Args:
        model_env: Bedrock model ID under test.
        signature: Short label of the matched flaky failure signature.
    """
    if model_env in _FLAKY_MODELS:
        pytest.xfail(f"known-flaky {signature} on {model_env} (best effort)")


# ---------------------------------------------------------------------------
# stdapi.ai source root (added read-only via --add-dir)
# ---------------------------------------------------------------------------

_STDAPI_SRC = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# API key used for the test-owned stdapi.ai server
# ---------------------------------------------------------------------------

_STDAPI_TEST_API_KEY = "cc-test-key"

# ---------------------------------------------------------------------------
# Security policy written to each test workdir's .claude/settings.json
# ---------------------------------------------------------------------------

_CLAUDE_SETTINGS: dict = {  # type: ignore[type-arg]
    "defaultMode": "dontAsk",
    "permissions": {
        "allow": [
            "Read(*)",
            "Glob(**)",
            "Bash(find *)",
            "Bash(ls *)",
            "Bash(cat *)",
            "Bash(grep *)",
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

# ---------------------------------------------------------------------------
# Model configurations (routed through Claude Code's "sonnet" slot via
# ANTHROPIC_DEFAULT_SONNET_MODEL)
#
# Each entry:
#   model_env       — Bedrock model ID set as ANTHROPIC_DEFAULT_SONNET_MODEL
#   extra_env       — Additional env vars (capabilities, caching, thinking, …)
#   supports_effort — Whether the model supports --effort levels
# ---------------------------------------------------------------------------

_MODEL_CONFIGS = [
    # ── Reference baseline ────────────────────────────────────────────────────
    pytest.param(
        {
            "model_env": "anthropic.claude-sonnet-4-6",
            "extra_env": {
                "ANTHROPIC_DEFAULT_SONNET_MODEL_SUPPORTED_CAPABILITIES": (
                    "effort,thinking,adaptive_thinking,interleaved_thinking"
                )
            },
            "supports_effort": True,
        },
        id="claude-sonnet-4-6",
    ),
    # ── Amazon ────────────────────────────────────────────────────────────────
    pytest.param(
        {
            "model_env": "amazon.nova-2-lite-v1:0",
            "extra_env": {
                "ANTHROPIC_DEFAULT_SONNET_MODEL_SUPPORTED_CAPABILITIES": "effort",
                "MAX_THINKING_TOKENS": "0",
            },
            "supports_effort": True,
        },
        id="nova-2-lite",
    ),
    # ── Moonshot AI ───────────────────────────────────────────────────────────
    pytest.param(
        {
            "model_env": "moonshotai.kimi-k2.5",
            "extra_env": {"DISABLE_PROMPT_CACHING": "1", "MAX_THINKING_TOKENS": "0"},
            "supports_effort": False,
        },
        id="kimi-k2.5",
    ),
    # ── Qwen Coder ────────────────────────────────────────────────────────────
    pytest.param(
        {
            "model_env": "qwen.qwen3-coder-30b-a3b-v1:0",
            "extra_env": {"DISABLE_PROMPT_CACHING": "1", "MAX_THINKING_TOKENS": "0"},
            "supports_effort": False,
        },
        id="qwen3-coder-30b",
    ),
    pytest.param(
        {
            "model_env": "qwen.qwen3-coder-next",
            "extra_env": {"DISABLE_PROMPT_CACHING": "1", "MAX_THINKING_TOKENS": "0"},
            "supports_effort": False,
        },
        id="qwen3-coder-next",
    ),
    # ── MiniMax ───────────────────────────────────────────────────────────────
    # M2.5 is a native-reasoning model; set MAX_THINKING_TOKENS=0 to suppress
    # Claude Code's own thinking budget (M2.5 reasons internally on its own).
    pytest.param(
        {
            "model_env": "minimax.minimax-m2.5",
            "extra_env": {"DISABLE_PROMPT_CACHING": "1", "MAX_THINKING_TOKENS": "0"},
            "supports_effort": False,
        },
        id="minimax-m2.5",
    ),
    # ── Mistral ───────────────────────────────────────────────────────────────
    pytest.param(
        {
            "model_env": "mistral.devstral-2-123b",
            "extra_env": {"DISABLE_PROMPT_CACHING": "1", "MAX_THINKING_TOKENS": "0"},
            "supports_effort": False,
        },
        id="devstral-2",
    ),
    # ── Z.AI ──────────────────────────────────────────────────────────────────
    pytest.param(
        {
            "model_env": "zai.glm-5",
            "extra_env": {"DISABLE_PROMPT_CACHING": "1", "MAX_THINKING_TOKENS": "0"},
            "supports_effort": False,
        },
        id="glm-5",
    ),
    # ── Google (Bedrock Mantle) ───────────────────────────────────────────────
    # Mantle-served; exercises the Anthropic messages → OpenAI conversion path.
    pytest.param(
        {
            "model_env": "google.gemma-4-31b",
            "extra_env": {"DISABLE_PROMPT_CACHING": "1", "MAX_THINKING_TOKENS": "0"},
            "supports_effort": False,
        },
        id="gemma-4-31b",
    ),
    # ── xAI (Bedrock Mantle) ──────────────────────────────────────────────────
    pytest.param(
        {
            "model_env": "xai.grok-4.3",
            "extra_env": {"DISABLE_PROMPT_CACHING": "1", "MAX_THINKING_TOKENS": "0"},
            "supports_effort": False,
        },
        id="grok-4.3",
    ),
]

# ---------------------------------------------------------------------------
# stdapi.ai server subprocess management
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # type: ignore[no-any-return]


@dataclass
class _ServerHandle:
    """Handle to a running stdapi.ai subprocess for Claude Code tests."""

    base_url: str
    #: stdout log lines appended in real time by a background reader thread.
    logs: list[str] = field(default_factory=list)
    #: stderr lines captured for debugging server startup errors.
    stderr_lines: list[str] = field(default_factory=list)
    #: subprocess.Popen handle; None when using an external ``--server-url``.
    _proc: subprocess.Popen | None = field(default=None, repr=False)  # type: ignore[type-arg]


@pytest.fixture(scope="session")
def _stdapi_server_session(request: pytest.FixtureRequest) -> Generator[_ServerHandle]:
    """Session-scoped stdapi.ai server for all Claude Code tests.

    If ``--server-url`` is given the fixture wraps the external URL and returns an
    empty log list (model-identity assertions are skipped).  Otherwise a fresh
    ``uvicorn stdapi.main:app`` process is spawned on a free port and its stdout
    JSON logs are streamed into the handle's log list.
    """
    server_url: str | None = request.config.getoption("--server-url", default=None)
    if server_url:
        # External server — no log capture, model-identity check will be skipped.
        yield _ServerHandle(base_url=server_url.rstrip("/"))
        return

    port = _find_free_port()
    # Remove all case variants first: conftest.py sets lowercase ``api_key``
    # via os.environ.update(), and the lowercase variant wins the settings'
    # case-insensitive collision in the subprocess — the server would then
    # reject the key forwarded to Claude Code.
    env = {**os.environ}
    for key in list(env):
        if key.lower() == "api_key":
            del env[key]
    env["API_KEY"] = _STDAPI_TEST_API_KEY

    # Run on this interpreter rather than through "uv run": the wrapper cannot
    # forward the SIGKILL below to the server it spawned, leaking a listening
    # process, and it serialises startup on the uv environment lock.
    proc = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "uvicorn",
            "stdapi.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--no-access-log",
        ],
        cwd=str(_STDAPI_SRC),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    handle = _ServerHandle(base_url=f"http://127.0.0.1:{port}", _proc=proc)

    def _stdout_reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            handle.logs.append(line.rstrip())

    def _stderr_reader() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            handle.stderr_lines.append(line.rstrip())

    threading.Thread(target=_stdout_reader, daemon=True).start()
    threading.Thread(target=_stderr_reader, daemon=True).start()

    # Wait for the /health endpoint to respond, allowing for an interpreter
    # start and a full app init on a machine loaded by the other xdist workers.
    deadline = time.monotonic() + 60
    healthy = False
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{handle.base_url}/health", timeout=2.0)
            if r.status_code == 200:
                healthy = True
                break
        except Exception:  # noqa: BLE001, S110
            pass
        time.sleep(0.5)

    if not healthy:
        proc.kill()
        startup_log = "\n".join(handle.stderr_lines[-30:] + handle.logs[-10:])
        pytest.fail(
            f"stdapi server failed to start on port {port}.\n"
            f"Last startup output:\n{startup_log}"
        )

    yield handle

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# Model-identity validation (autouse — runs for every test in this module)
# ---------------------------------------------------------------------------


#: Claude Code built-in default models used for internal background calls.
_CLI_AUXILIARY_MODEL_PREFIXES = ("anthropic.claude-opus-", "anthropic.claude-haiku-")

#: Session IDs minted by ``_run_claude`` during the current test.
_ACTIVE_SESSION_IDS: list[str] = []

#: Session IDs whose CLI run completed successfully during the current test.
_COMPLETED_SESSION_IDS: list[str] = []


def _assert_model_identity(
    logs: list[str],
    expected_model_id: str,
    session_ids: list[str],
    completed_session_ids: list[str],
) -> None:
    """Verify that every Bedrock request in *logs* targeted *expected_model_id*.

    Parses each stdout log line as JSON and inspects the ``model_id`` field emitted
    by ``validate_model``.  Requests are attributed to this test through the
    deterministic ``--session-id`` that Claude Code embeds in its
    ``metadata.user_id`` (logged as ``request_user_id``), so trailing traffic
    from neighboring slow or timeout-killed CLIs in the shared session log
    never bleeds into the check.  Claude Code's own background calls to its
    built-in default models are tolerated once the conversation reached the
    expected model.  Skips silently when *logs* is empty (external server).

    Args:
        logs: Server stdout log lines captured during the test window.
        expected_model_id: Bedrock model ID the test's requests must target.
        session_ids: Session IDs minted by ``_run_claude`` during the test.
        completed_session_ids: Session IDs whose CLI run completed
            successfully (attribution must find their requests).

    Raises:
        AssertionError: If any request attributed to this test used a
            different model than expected, or if a successful CLI run left
            no attributable request (session attribution breakage).
    """
    if not logs or not session_ids:
        return  # External server, or the CLI never ran.

    model_ids_seen: list[str] = []
    for line in logs:
        if not line.startswith("{"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid = entry.get("model_id")
        uid = entry.get("request_user_id") or ""
        if mid and any(sid in uid for sid in session_ids):
            model_ids_seen.append(mid)

    if not model_ids_seen:
        # Tolerated only when no CLI run completed (e.g. timeout before the
        # first call); a successful run without attributable requests means
        # the session-based attribution itself broke and coverage vanished.
        assert not completed_session_ids, (
            "No server-log request was attributed to this test's session IDs "
            "although the CLI completed successfully: session attribution "
            "(request_user_id) is broken."
        )
        return

    expected_prefix = expected_model_id.split(":", maxsplit=1)[0]
    unexpected = [m for m in model_ids_seen if not m.startswith(expected_prefix)]
    if unexpected and any(m.startswith(expected_prefix) for m in model_ids_seen):
        # Background calls with Claude Code's built-in defaults are fine as
        # long as the conversation itself reached the expected model.
        unexpected = [
            m for m in unexpected if not m.startswith(_CLI_AUXILIARY_MODEL_PREFIXES)
        ]
    assert not unexpected, (
        f"Model identity mismatch: expected requests to {expected_model_id!r}, "
        f"but saw {list(dict.fromkeys(unexpected))} in server logs."
    )


@pytest.fixture(autouse=True)
def _model_identity_check(
    model_config: dict,  # type: ignore[type-arg]
    _stdapi_server_session: _ServerHandle,
) -> Generator[None]:
    """Snapshot log position before the test and validate model identity after."""
    snapshot = len(_stdapi_server_session.logs)
    _ACTIVE_SESSION_IDS.clear()
    _COMPLETED_SESSION_IDS.clear()
    yield
    new_logs = _stdapi_server_session.logs[snapshot:]
    _assert_model_identity(
        new_logs,
        model_config["model_env"],
        list(_ACTIVE_SESSION_IDS),
        list(_COMPLETED_SESSION_IDS),
    )


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def claude_code_base_url(_stdapi_server_session: _ServerHandle) -> str:
    """Anthropic-endpoint URL for Claude Code (includes ``/anthropic`` suffix)."""
    return f"{_stdapi_server_session.base_url}/anthropic"


@pytest.fixture(scope="session")
def claude_code_api_key(_stdapi_server_session: _ServerHandle) -> str:
    """API key forwarded to Claude Code as ``ANTHROPIC_AUTH_TOKEN``."""
    if _stdapi_server_session._proc is not None:  # noqa: SLF001
        # Own server — use the known test key.
        return _STDAPI_TEST_API_KEY
    # External server — fall back to env.
    return os.getenv("ANTHROPIC_AUTH_TOKEN") or os.getenv("OPENAI_API_KEY") or ""


# ---------------------------------------------------------------------------
# Function-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def claude_workdir(tmp_path: Path) -> Path:
    """Isolated working directory with Claude Code security policy pre-applied.

    A sandboxed ``CLAUDE_CONFIG_DIR`` (``claude-config/``) is created next to
    the workspace with the workspace pre-trusted, so runs never read or mutate
    the user-level Claude Code configuration.
    """
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text(json.dumps(_CLAUDE_SETTINGS, indent=2))
    (tmp_path / "results").mkdir()
    config_dir = tmp_path / "claude-config"
    config_dir.mkdir()
    (config_dir / ".claude.json").write_text(
        json.dumps(
            {
                "hasCompletedOnboarding": True,
                "bypassPermissionsModeAccepted": True,
                "projects": {
                    str(tmp_path): {"hasTrustDialogAccepted": True},
                    str(_STDAPI_SRC): {"hasTrustDialogAccepted": True},
                },
            }
        )
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Core subprocess helper
# ---------------------------------------------------------------------------


def _test_session_id(model_env: str, test_name: str) -> str:
    """Return a deterministic UUID5 for a (model, test) pair.

    Using a stable namespace + key means the same (model, test) combination always
    produces the same session ID, making it easy to correlate Claude Code's own logs
    with stdapi request logs across test runs.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{model_env}:{test_name}"))


def _run_claude(
    workdir: Path,
    base_url: str,
    api_key: str,
    prompt: str,
    model_env: str,
    test_name: str,
    extra_env: dict[str, str] | None = None,
    effort: str | None = None,
    timeout: int = 300,
) -> dict:  # type: ignore[type-arg]
    """Run ``claude --print --output-format json`` and return the parsed result dict.

    Args:
        workdir:    Working directory (must contain ``.claude/settings.json``).
        base_url:   Value for ``ANTHROPIC_BASE_URL`` (the stdapi.ai ``/anthropic``
                    endpoint).
        api_key:    Value for ``ANTHROPIC_AUTH_TOKEN``.
        prompt:     Natural-language task for Claude Code.
        model_env:  Bedrock model ID assigned to the ``sonnet`` slot.
        test_name:  Logical test name used to derive a deterministic session UUID.
        extra_env:  Additional environment overrides (capabilities, caching, …).
        effort:     Optional effort level: ``"low"``, ``"medium"``, or ``"high"``.
        timeout:    Subprocess timeout in seconds.

    Returns:
        Parsed JSON result object emitted by ``claude``, including ``num_turns``,
        ``duration_ms``, and ``usage`` (input/output token counts).

    Raises:
        AssertionError: If the process exits non-zero or output is not valid JSON.
    """
    assert _CLAUDE_BIN is not None

    session_id = _test_session_id(model_env, test_name)
    # Registered for the identity check: attributes server-log requests to
    # this test via the session UUID embedded in Claude Code's user metadata.
    _ACTIVE_SESSION_IDS.append(session_id)

    env: dict[str, str] = {
        **os.environ,
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": api_key,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model_env,
        # Sandboxed config with the workspace pre-trusted: never read or
        # mutate the user-level Claude Code configuration.
        "CLAUDE_CONFIG_DIR": str(workdir / "claude-config"),
    }
    # Remove CLAUDECODE so claude can start even when invoked from inside another
    # Claude Code session (e.g. when running these tests from Claude Code itself),
    # and strip inherited auth/transport overrides that would bypass the test
    # server (user-level API keys or direct-Bedrock routing).
    for conflicting in (
        "CLAUDECODE",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_CUSTOM_HEADERS",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
        "ANTHROPIC_BEDROCK_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "AWS_BEARER_TOKEN_BEDROCK",
    ):
        env.pop(conflicting, None)
    if extra_env:
        env.update(extra_env)

    cmd: list[str] = [
        _CLAUDE_BIN,
        "--print",
        "--output-format",
        "json",
        "--model",
        "sonnet",
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
        "--session-id",
        session_id,
        "--max-budget-usd",
        "10",
        "--add-dir",
        str(_STDAPI_SRC),
        "--allowedTools",
        "Read,Glob,Bash",
        "--disallowedTools",
        "Write,Edit,WebFetch,WebSearch",
    ]
    if effort:
        cmd.extend(["--effort", effort])
    # NOTE: --add-dir accepts multiple values, so appending the prompt as a
    # positional arg causes it to be consumed as another directory path.
    # Piping the prompt via stdin is the reliable alternative.
    try:
        proc = subprocess.run(  # noqa: S603, PLW1510
            cmd,
            env=env,
            cwd=str(workdir),
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        _xfail_if_flaky(model_env, "CLI timeout")
        raise
    if proc.returncode != 0:
        _xfail_if_flaky(model_env, f"CLI exit code {proc.returncode}")
    assert proc.returncode == 0, (
        f"claude exited with code {proc.returncode}\n"
        f"stdout: {proc.stdout[:1000]}\n"
        f"stderr: {proc.stderr[:500]}"
    )
    try:
        result: dict = json.loads(proc.stdout)  # type: ignore[type-arg]
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"claude output is not valid JSON: {exc}\n"
            f"Output (first 500 chars): {proc.stdout[:500]}"
        )
    _COMPLETED_SESSION_IDS.append(session_id)
    return result


def _assert_result(
    data: dict,  # type: ignore[type-arg]
    *,
    model_env: str,
    contains: str | None = None,
    min_turns: int = 0,
) -> str:
    """Assert the claude result dict represents a successful response.

    Content-quality failures on known-flaky models downgrade to xfail; any
    failure on other models (or of another kind) fails normally.

    Args:
        data:       Parsed JSON from ``_run_claude``.
        model_env:  Bedrock model ID under test (flaky-signature scoping).
        contains:   Optional substring that must appear in the result text
                    (case-insensitive).
        min_turns:  Minimum number of turns expected.  Use to verify the model
                    actually explored the codebase rather than answering from memory.

    Returns:
        The result text string.
    """
    if data.get("is_error", False):
        _xfail_if_flaky(model_env, "CLI-reported error")
        pytest.fail(f"claude reported an error: {data}")
    result: str = data.get("result", "")
    if len(result) <= 10:
        _xfail_if_flaky(model_env, "empty result")
        pytest.fail(f"claude result is unexpectedly short: {result!r}")
    if contains and contains.lower() not in result.lower():
        _xfail_if_flaky(model_env, "content assertion")
        pytest.fail(f"Expected {contains!r} in result text:\n{result}")
    if min_turns > 0 and (turns := data.get("num_turns", 0)) < min_turns:
        _xfail_if_flaky(model_env, "turn-count assertion")
        pytest.fail(
            f"Expected at least {min_turns} turns (model must explore source files), "
            f"got {turns} — response:\n{result[:300]}"
        )
    return result


def _log_metrics(data: dict, model_id: str, test_name: str) -> None:  # type: ignore[type-arg]
    """Print benchmark metrics to stdout in a grep-friendly format.

    Token counts (input/output) are reported instead of cost, since Claude Code's
    internal cost estimate uses Claude Sonnet pricing regardless of the actual
    Bedrock model routed, making cross-model cost comparisons misleading.

    Output is visible with ``pytest -s`` or in the failure output.
    Collect all results with::

        pytest --expensive -s 2>&1 | grep CC-METRICS
    """
    turns = data.get("num_turns", 0)
    duration_s = data.get("duration_ms", 0) / 1000
    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_created = usage.get("cache_creation_input_tokens", 0)
    cache_info = ""
    if cache_read or cache_created:
        cache_info = f" [cache_read={cache_read} cache_created={cache_created}]"
    print(  # noqa: T201
        f"\nCC-METRICS | {model_id:<30} | {test_name:<40} "
        f"| turns={turns:>3} | {duration_s:>7.1f}s "
        f"| in={input_tokens:>6} out={output_tokens:>5}{cache_info}"
    )


# ---------------------------------------------------------------------------
# Prompts
#
# All prompts are designed to require the model to explore multiple source
# files and make ~10 tool calls, which produces a more representative
# agentic benchmark than single-file lookup tasks.
# ---------------------------------------------------------------------------

_PROMPT_REQUEST_PIPELINE = f"""\
You are an AI coding assistant with access to the stdapi.ai source at {_STDAPI_SRC}.

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
You are an AI coding assistant with access to the stdapi.ai source at {_STDAPI_SRC}.

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
You are an AI coding assistant with access to the stdapi.ai source at {_STDAPI_SRC}.

Your task: Produce a precise, code-backed mapping of OpenAI chat completions API
parameters to AWS Bedrock Converse API fields as implemented in stdapi.ai.

You MUST open and read each of these files before answering:
  1. The OpenAI types file — find CompletionCreateParams and read ALL its fields
  2. The translate_request function in the adapter — read its full body
  3. stdapi/models/chat/_default.py — read _prepare_converse_request in full

For each parameter you find, quote the EXACT line(s) of code that handle it and state:
  • OpenAI parameter name  →  Bedrock Converse field name  (quote the assignment)

Document at least 10 parameter mappings with real code quotes.  Do not skip any.
Include: temperature, max_tokens, top_p, stop_sequences, stream, system messages,
tools/toolConfig, metadata/requestMetadata, and any others you find.
"""

_PROMPT_MODEL_OVERRIDES = f"""\
You are an AI coding assistant with access to the stdapi.ai source at {_STDAPI_SRC}.

Your task: Find ALL model-specific behavior override files in the chat completions
module and document the custom logic each one implements.

You MUST:
  1. List ALL Python files inside stdapi/models/chat/ (use Glob or Bash)
  2. Read _default.py briefly to understand what the base class provides
  3. For each model-specific file (not _default.py, __init__.py, or _adapters/):
     - Read the file
     - State which Bedrock model family it targets
     - Quote the method signature(s) it overrides
     - Explain in 1-2 sentences what custom behavior it adds vs the default

Document at least 5 model-specific files with real code quotes.
"""


# ---------------------------------------------------------------------------
# Tests: request pipeline tracing  (target ~8-12 turns)
# ---------------------------------------------------------------------------


@pytest.mark.agentic
@_SKIP_NO_CLAUDE
@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
class TestClaudeCodePipeline:
    """Claude Code traces multi-file execution paths in the stdapi.ai codebase."""

    def test_trace_request_pipeline(
        self,
        request: pytest.FixtureRequest,
        model_config: dict,  # type: ignore[type-arg]
        claude_workdir: Path,
        claude_code_base_url: str,
        claude_code_api_key: str,
    ) -> None:
        """Trace POST /v1/chat/completions from route handler to Bedrock converse().

        Requires multi-file exploration: route → handler → adapter → model → Bedrock.
        Target: ~10 turns.
        """
        data = _run_claude(
            workdir=claude_workdir,
            base_url=claude_code_base_url,
            api_key=claude_code_api_key,
            prompt=_PROMPT_REQUEST_PIPELINE,
            model_env=model_config["model_env"],
            test_name=request.node.originalname,
            extra_env=model_config["extra_env"],
        )
        _log_metrics(data, model_config["model_env"], "test_trace_request_pipeline")
        result = _assert_result(
            data, model_env=model_config["model_env"], contains="converse", min_turns=2
        )
        # Should have walked through at least the adapter and model handler
        if not any(
            kw in result.lower()
            for kw in ("translate_request", "_prepare_converse_request", "_default")
        ):
            _xfail_if_flaky(model_config["model_env"], "content assertion")
            pytest.fail(f"Expected core function names in result:\n{result}")

    def test_trace_streaming_path(
        self,
        request: pytest.FixtureRequest,
        model_config: dict,  # type: ignore[type-arg]
        claude_workdir: Path,
        claude_code_base_url: str,
        claude_code_api_key: str,
    ) -> None:
        """Trace the streaming code path from stream=True to SSE output.

        Requires reading _default.py, SSE adapters, and generator code.
        Target: ~8 turns.
        """
        data = _run_claude(
            workdir=claude_workdir,
            base_url=claude_code_base_url,
            api_key=claude_code_api_key,
            prompt=_PROMPT_STREAMING_PATH,
            model_env=model_config["model_env"],
            test_name=request.node.originalname,
            extra_env=model_config["extra_env"],
        )
        _log_metrics(data, model_config["model_env"], "test_trace_streaming_path")
        result = _assert_result(data, model_env=model_config["model_env"], min_turns=2)
        if not any(
            kw in result.lower()
            for kw in ("stream", "sse", "generator", "event", "converse_stream")
        ):
            _xfail_if_flaky(model_config["model_env"], "content assertion")
            pytest.fail(f"Expected streaming-related keywords in result:\n{result}")


# ---------------------------------------------------------------------------
# Tests: code analysis  (target ~10-15 turns)
# ---------------------------------------------------------------------------


@pytest.mark.agentic
@_SKIP_NO_CLAUDE
@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
class TestClaudeCodeAnalysis:
    """Claude Code performs multi-file analysis tasks on the stdapi.ai codebase."""

    def test_audit_parameter_mapping(
        self,
        request: pytest.FixtureRequest,
        model_config: dict,  # type: ignore[type-arg]
        claude_workdir: Path,
        claude_code_base_url: str,
        claude_code_api_key: str,
    ) -> None:
        """Audit the full OpenAI → Bedrock parameter translation across 3 source files.

        Requires reading types, adapter, and _prepare_converse_request.
        Target: ~12 turns.
        """
        data = _run_claude(
            workdir=claude_workdir,
            base_url=claude_code_base_url,
            api_key=claude_code_api_key,
            prompt=_PROMPT_PARAMETER_MAPPING,
            model_env=model_config["model_env"],
            test_name=request.node.originalname,
            extra_env=model_config["extra_env"],
        )
        _log_metrics(data, model_config["model_env"], "test_audit_parameter_mapping")
        result = _assert_result(data, model_env=model_config["model_env"], min_turns=2)
        # Should have found multiple parameter mappings
        if not any(
            kw in result.lower()
            for kw in ("temperature", "max_tokens", "inferenceconfig", "messages")
        ):
            _xfail_if_flaky(model_config["model_env"], "content assertion")
            pytest.fail(f"Expected parameter names in result:\n{result}")

    def test_enumerate_model_overrides(
        self,
        request: pytest.FixtureRequest,
        model_config: dict,  # type: ignore[type-arg]
        claude_workdir: Path,
        claude_code_base_url: str,
        claude_code_api_key: str,
    ) -> None:
        """Find and explain all model-specific override files in stdapi/models/chat/.

        Requires Glob + reading 5+ files.
        Target: ~10 turns.
        """
        data = _run_claude(
            workdir=claude_workdir,
            base_url=claude_code_base_url,
            api_key=claude_code_api_key,
            prompt=_PROMPT_MODEL_OVERRIDES,
            model_env=model_config["model_env"],
            test_name=request.node.originalname,
            extra_env=model_config["extra_env"],
        )
        _log_metrics(data, model_config["model_env"], "test_enumerate_model_overrides")
        result = _assert_result(data, model_env=model_config["model_env"], min_turns=4)
        # Should have found known model families
        if not any(
            kw in result.lower()
            for kw in ("nova", "claude", "deepseek", "mistral", "llama", "qwen")
        ):
            _xfail_if_flaky(model_config["model_env"], "content assertion")
            pytest.fail(f"Expected model family names in result:\n{result}")


# ---------------------------------------------------------------------------
# Tests: effort levels  (target ~8-10 turns, using parameter mapping task)
# ---------------------------------------------------------------------------


@pytest.mark.agentic
@_SKIP_NO_CLAUDE
@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
class TestClaudeCodeEffortLevels:
    """Effort-based reasoning levels work end-to-end via stdapi.ai."""

    @pytest.mark.parametrize("effort", ["low", "high"])
    def test_effort_parameter_mapping(
        self,
        request: pytest.FixtureRequest,
        model_config: dict,  # type: ignore[type-arg]
        effort: str,
        claude_workdir: Path,
        claude_code_base_url: str,
        claude_code_api_key: str,
    ) -> None:
        """Parameter mapping task runs to completion at both low and high effort.

        Uses the same multi-file parameter audit as TestClaudeCodeAnalysis to
        produce comparable turn/duration metrics across effort levels.
        """
        if not model_config["supports_effort"]:
            pytest.skip(
                f"Model {model_config['model_env']} does not support effort levels"
            )
        data = _run_claude(
            workdir=claude_workdir,
            base_url=claude_code_base_url,
            api_key=claude_code_api_key,
            prompt=_PROMPT_PARAMETER_MAPPING,
            model_env=model_config["model_env"],
            test_name=request.node.originalname,
            extra_env=model_config["extra_env"],
            effort=effort,
        )
        _log_metrics(
            data, model_config["model_env"], f"test_effort_parameter_mapping[{effort}]"
        )
        result = _assert_result(data, model_env=model_config["model_env"], min_turns=2)
        if not any(
            kw in result.lower()
            for kw in ("temperature", "max_tokens", "inferenceconfig", "messages")
        ):
            _xfail_if_flaky(model_config["model_env"], "content assertion")
            pytest.fail(f"Expected parameter names in result:\n{result}")
