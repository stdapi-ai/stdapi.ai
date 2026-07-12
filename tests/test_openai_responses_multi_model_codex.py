"""Agentic tests for Codex CLI connected to stdapi.ai via the Responses API route.

Tests launch a real ``codex exec`` process in non-interactive (``--json``) mode
against a **dedicated stdapi.ai server** spawned by the test fixture on a free port.
This provides full end-to-end isolation — each test session owns its own server,
captures its JSON request logs, and asserts that requests were routed to the
expected Bedrock model.

Architecture:
    - ``_stdapi_server_session`` (session-scoped): spawns ``uvicorn stdapi.main:app``
      on a random free port **without an API key** (no-auth mode), and streams its
      stdout JSON logs into a shared list.
    - ``_model_identity_check`` (autouse function-scoped): takes a snapshot of the
      log list before each test and, after it completes, verifies that every logged
      ``model_id`` field matches the expected Bedrock model ID for that parametrize
      variant.

Requirements:
    - ``codex`` CLI binary at the JetBrains IDE cache location (tests skip otherwise).
    - AWS Bedrock credentials in the environment (same as for the normal test suite).
    - Pass ``--agentic`` to pytest to enable these tests.

What these tests uniquely exercise on /v1/responses:
    - Codex sends a large ``instructions`` field (~7600 tokens of system prompt).
    - Multi-turn tool use: ``function_call`` items echoed back as input followed by
      ``function_call_output`` results — the pattern that required the bug fix for
      ``FunctionCallInput`` in ``ResponseInputItem``.
    - ``developer`` role messages in the ``input`` array.
    - Real SSE streaming of multi-turn agentic responses.

Metrics collected per run (visible with ``pytest -s`` or on failure):
    ``CO-METRICS | <model> | <test> | turns=<N> | <Xs> | in=<N> out=<N>``
"""

import contextlib
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# ---------------------------------------------------------------------------
# Codex binary detection
# ---------------------------------------------------------------------------

#: Platform-specific binary name fragment for the Codex CLI under the JetBrains AIA cache.
#: Only the ``codex-*-linux-musl`` variant is the full CLI; the ``codex-acp-*`` sibling
#: is a different tool and should not be used here.
_CODEX_BIN_NAME = "codex-x86_64-unknown-linux-musl"

#: Codex CLI vendored by the JetBrains ACP agent runtime (newer IDE layout).
_ACP_CODEX_GLOB = (
    "*/acp-agents/.runtimes/node/*/npm-cache/_npx/*/node_modules/@openai/"
    "codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex"
)


def _find_jetbrains_codex() -> str | None:
    """Search the JetBrains caches for the Codex CLI binary.

    Scans the ACP agent runtime layout
    (``~/.cache/JetBrains/*/acp-agents/.runtimes/node/...``) first, then the
    legacy AIA layout (``~/.cache/JetBrains/*/aia/codex/bin/<name>``); the
    newest match wins within each layout.

    Returns:
        Absolute path string if found, otherwise ``None``.
    """
    cache_root = Path.home() / ".cache" / "JetBrains"
    if not cache_root.is_dir():
        return None
    acp_candidates = sorted(
        cache_root.glob(_ACP_CODEX_GLOB),
        key=lambda p: p.stat().st_mtime,  # Newest install wins.
        reverse=True,
    )
    if acp_candidates:
        return str(acp_candidates[0])
    # Sort descending so the newest IDE version wins.
    candidates = sorted(
        cache_root.glob(f"*/aia/codex/bin/{_CODEX_BIN_NAME}"),
        key=lambda p: p.parts[-5],  # IDE version dir, e.g. "PyCharm2026.1"
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def _resolve_codex_bin() -> str | None:
    """Resolve the Codex CLI binary path using ``CODEX_BIN`` env var or autodetection.

    Resolution order:
    1. ``$CODEX_BIN`` environment variable (explicit path).
    2. JetBrains AIA cache autodetection (newest IDE version first).
    3. ``codex`` on ``$PATH`` (system install).

    Returns:
        Absolute path string if a usable binary is found, otherwise ``None``.
    """
    if env_bin := os.environ.get("CODEX_BIN", ""):
        p = Path(env_bin)
        return str(p) if p.exists() else None
    if jetbrains := _find_jetbrains_codex():
        return jetbrains
    return shutil.which("codex")


_CODEX_BIN: str | None = _resolve_codex_bin()

_SKIP_NO_CODEX = pytest.mark.skipif(
    _CODEX_BIN is None,
    reason=(
        "codex CLI not found — install Codex via a JetBrains IDE, set CODEX_BIN to the "
        f"binary path, or place {_CODEX_BIN_NAME!r} on $PATH"
    ),
)

# ---------------------------------------------------------------------------
# stdapi.ai source root
# ---------------------------------------------------------------------------

_STDAPI_SRC = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Model configurations
#
# Each entry:
#   model_env — Bedrock model ID passed to ``codex exec -m``
#   extra_env — Additional env vars (disable caching, etc.)
# ---------------------------------------------------------------------------

_MODEL_CONFIGS = [
    # ── Reference baseline ────────────────────────────────────────────────────
    pytest.param(
        {"model_env": "anthropic.claude-sonnet-4-6", "extra_env": {}},
        id="claude-sonnet-4-6",
    ),
    # ── Amazon ────────────────────────────────────────────────────────────────
    pytest.param(
        {"model_env": "amazon.nova-2-lite-v1:0", "extra_env": {}}, id="nova-2-lite"
    ),
    # ── Moonshot AI ───────────────────────────────────────────────────────────
    pytest.param(
        {"model_env": "moonshotai.kimi-k2.5", "extra_env": {}}, id="kimi-k2.5"
    ),
    # ── Qwen Coder ────────────────────────────────────────────────────────────
    pytest.param(
        {"model_env": "qwen.qwen3-coder-30b-a3b-v1:0", "extra_env": {}},
        id="qwen3-coder-30b",
    ),
    pytest.param(
        # Notably slower than its siblings; give each agent run extra headroom.
        {"model_env": "qwen.qwen3-coder-next", "extra_env": {}, "timeout": 1200},
        id="qwen3-coder-next",
    ),
    # ── MiniMax ───────────────────────────────────────────────────────────────
    pytest.param(
        {"model_env": "minimax.minimax-m2.5", "extra_env": {}}, id="minimax-m2.5"
    ),
    # ── Mistral ───────────────────────────────────────────────────────────────
    pytest.param(
        {"model_env": "mistral.devstral-2-123b", "extra_env": {}}, id="devstral-2"
    ),
    # ── Z.AI ──────────────────────────────────────────────────────────────────
    pytest.param({"model_env": "zai.glm-5", "extra_env": {}}, id="glm-5"),
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
    """Handle to a running stdapi.ai subprocess for Codex tests."""

    base_url: str
    #: stdout log lines appended in real time by a background reader thread.
    logs: list[str] = field(default_factory=list)
    #: stderr lines captured for debugging server startup errors.
    stderr_lines: list[str] = field(default_factory=list)
    #: subprocess.Popen handle; None when using an external ``--server-url``.
    _proc: subprocess.Popen | None = field(default=None, repr=False)  # type: ignore[type-arg]


@pytest.fixture(scope="session")
def _stdapi_server_session(request: pytest.FixtureRequest) -> Generator[_ServerHandle]:
    """Session-scoped stdapi.ai server for all Codex tests.

    Starts a fresh ``uvicorn stdapi.main:app`` process **without an API key**
    (no-auth mode) so that the Codex CLI — which sends no ``Authorization`` header
    when using a custom model provider — can communicate freely.

    If ``--server-url`` is given the fixture wraps the external URL and returns an
    empty log list (model-identity assertions are skipped).  Otherwise a fresh
    process is spawned on a free port and its stdout JSON logs are streamed into
    the handle's log list.
    """
    server_url: str | None = request.config.getoption("--server-url", default=None)
    if server_url:
        yield _ServerHandle(base_url=server_url.rstrip("/"))
        return

    port = _find_free_port()
    # Intentionally omit API_KEY so the server runs in no-auth mode.
    # Codex sends no Authorization header when using a custom provider, so
    # requiring auth would block all requests.  Remove all case variants since
    # conftest.py sets lowercase ``api_key`` via os.environ.update().
    env = {**os.environ}
    for key in list(env):
        if key.lower() in {
            "api_key",
            "api_key_ssm_parameter",
            "api_key_secretsmanager_secret",
        }:
            del env[key]

    proc = subprocess.Popen(  # noqa: S603
        [  # noqa: S607
            "uv",
            "run",
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

    deadline = time.monotonic() + 30
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


def _assert_model_identity(logs: list[str], expected_model_id: str) -> None:
    """Verify that Bedrock requests in *logs* targeted *expected_model_id*.

    Parses each stdout log line as JSON and inspects the ``model_id`` field emitted
    by ``validate_model``.  Skips silently when *logs* is empty (external server).

    **Trailing-entry tolerance**: when a previous test's codex process is killed by
    a timeout, its orphaned SSE connections can deliver their final server log entries
    *after* that test's fixture teardown — appearing at the start of the *next* test's
    log window.  To avoid false failures, entries that arrive before the first
    occurrence of the expected model are treated as "trailing from previous test" and
    are ignored.  Only entries from the first expected-model entry onwards are checked.

    Raises:
        AssertionError: If any request *after* the first correct-model entry used a
            different model than expected.
    """
    if not logs:
        return

    model_ids_seen: list[str] = []
    for line in logs:
        if not line.startswith("{"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid = entry.get("model_id")
        if mid:
            model_ids_seen.append(mid)

    if not model_ids_seen:
        return

    prefix = expected_model_id.split(":", maxsplit=1)[0]
    # Find the first log entry that belongs to this test (expected model present).
    # Entries before it are "trailing" from a previously timed-out test and ignored.
    first_expected_idx = next(
        (i for i, m in enumerate(model_ids_seen) if m.startswith(prefix)), None
    )
    if first_expected_idx is None:
        # The expected model never appeared — test likely timed out before making any
        # requests, or all entries are trailing contamination.  Nothing to assert.
        return

    unexpected = [
        m for m in model_ids_seen[first_expected_idx:] if not m.startswith(prefix)
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
    yield
    new_logs = _stdapi_server_session.logs[snapshot:]
    _assert_model_identity(new_logs, model_config["model_env"])


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def codex_base_url(_stdapi_server_session: _ServerHandle) -> str:
    """OpenAI /v1 endpoint URL for Codex (OPENAI_BASE_URL value)."""
    return f"{_stdapi_server_session.base_url}/v1"


# ---------------------------------------------------------------------------
# Core subprocess helper
# ---------------------------------------------------------------------------


def _codex_provider_overrides(base_url: str) -> list[str]:
    """Build the ``-c`` overrides pointing Codex at the stdapi.ai server.

    Newer Codex CLIs reject overrides of the reserved built-in ``openai``
    provider, so a dedicated ``stdapi`` provider is declared instead, using
    the OpenAI Responses API wire format.

    Args:
        base_url: The stdapi.ai ``/v1`` endpoint URL.

    Returns:
        ``-c`` flag/value pairs for ``codex exec``.
    """
    return [
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
    ]


def _run_codex(
    base_url: str,
    prompt: str,
    model_env: str,
    extra_env: dict[str, str] | None = None,
    workdir: Path | None = None,
    timeout: int = 600,
) -> list[dict]:  # type: ignore[type-arg]
    """Run ``codex exec --json`` and return the parsed JSONL event list.

    Args:
        base_url:   Value for ``OPENAI_BASE_URL`` (the stdapi.ai ``/v1`` endpoint).
        prompt:     Natural-language task for Codex.
        model_env:  Bedrock model ID passed via ``-m``.
        extra_env:  Additional environment overrides.
        workdir:    Working directory for Codex (``-C`` flag); defaults to stdapi source.
        timeout:    Subprocess timeout in seconds.

    Returns:
        List of parsed JSONL event dicts emitted by ``codex exec --json``.

    Raises:
        AssertionError: If the process exits non-zero or output contains no events.
    """
    assert _CODEX_BIN is not None

    env: dict[str, str] = {
        **os.environ,
        # Route all Responses API calls through our test server.
        "OPENAI_BASE_URL": base_url,
        # The server runs without an API key; any non-empty value avoids SDK errors.
        "OPENAI_API_KEY": "test-key",
    }
    if extra_env:
        env.update(extra_env)

    cwd = workdir or _STDAPI_SRC

    cmd: list[str] = [
        _CODEX_BIN,
        "exec",
        *_codex_provider_overrides(base_url),
        "-m",
        model_env,
        "--json",
        "--ephemeral",
        "--dangerously-bypass-approvals-and-sandbox",
        "-s",
        "read-only",
        "-C",
        str(cwd),
        prompt,
    ]

    proc = subprocess.run(  # noqa: S603, PLW1510
        cmd, env=env, cwd=str(cwd), capture_output=True, text=True, timeout=timeout
    )
    assert proc.returncode == 0, (
        f"codex exited with code {proc.returncode}\n"
        f"stdout: {proc.stdout[:2000]}\n"
        f"stderr: {proc.stderr[:500]}"
    )

    events: list[dict] = []  # type: ignore[type-arg]
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        with contextlib.suppress(json.JSONDecodeError):
            events.append(json.loads(line))
    assert events, f"codex produced no JSONL events\nstdout: {proc.stdout[:500]}"
    return events


def _assert_result(
    events: list[dict],  # type: ignore[type-arg]
    *,
    contains: str | None = None,
    min_tool_calls: int = 0,
) -> str:
    """Assert the Codex JSONL event list represents a successful response.

    Args:
        events:         Parsed JSONL events from ``_run_codex``.
        contains:       Optional substring that must appear in the final agent message
                        (case-insensitive).
        min_tool_calls: Minimum number of ``command_execution`` items expected.

    Returns:
        The text of the last ``agent_message`` item.
    """
    # Must not have a turn.failed event
    failed = [e for e in events if e.get("type") == "turn.failed"]
    assert not failed, f"codex turn failed: {failed[0]}"

    # Extract the final agent message text
    agent_messages = [
        e["item"]["text"]
        for e in events
        if e.get("type") == "item.completed"
        and e.get("item", {}).get("type") == "agent_message"
    ]
    assert agent_messages, f"No agent_message found in events:\n{events}"
    result: str = agent_messages[-1]
    assert len(result) > 5, f"agent_message is unexpectedly short: {result!r}"

    if contains:
        assert contains.lower() in result.lower(), (
            f"Expected {contains!r} in result:\n{result}"
        )

    if min_tool_calls > 0:
        tool_calls = [
            e
            for e in events
            if e.get("type") == "item.completed"
            and e.get("item", {}).get("type") == "command_execution"
        ]
        assert len(tool_calls) >= min_tool_calls, (
            f"Expected at least {min_tool_calls} command_execution items, "
            f"got {len(tool_calls)}"
        )

    return result


def _log_metrics(
    events: list[dict],  # type: ignore[type-arg]
    model_id: str,
    test_name: str,
) -> None:
    """Print benchmark metrics to stdout in a grep-friendly format.

    Output is visible with ``pytest -s`` or in the failure output.
    Collect all results with::

        pytest --agentic -s 2>&1 | grep CO-METRICS
    """
    tool_calls = sum(
        1
        for e in events
        if e.get("type") == "item.completed"
        and e.get("item", {}).get("type") == "command_execution"
    )
    completed = next((e for e in events if e.get("type") == "turn.completed"), {})
    usage = completed.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cached = usage.get("cached_input_tokens", 0)
    cache_info = f" [cached={cached}]" if cached else ""
    print(  # noqa: T201
        f"\nCO-METRICS | {model_id:<30} | {test_name:<40} "
        f"| tool_calls={tool_calls:>2} "
        f"| in={input_tokens:>6} out={output_tokens:>5}{cache_info}"
    )


# ---------------------------------------------------------------------------
# Prompts
#
# All prompts require Codex to use shell commands to explore source files,
# exercising multi-turn tool use via the /v1/responses function_call pattern.
# ---------------------------------------------------------------------------

_PROMPT_REQUEST_PIPELINE = f"""\
You are working in the stdapi.ai source tree at {_STDAPI_SRC}.

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
You are working in the stdapi.ai source tree at {_STDAPI_SRC}.

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
You are working in the stdapi.ai source tree at {_STDAPI_SRC}.

Produce a precise, code-backed mapping of Responses API parameters to AWS Bedrock
Converse API fields as implemented in stdapi.ai.

Use shell commands to read:
  1. stdapi/types/openai_responses.py — find ResponseCreateParams fields
  2. The translate_request function in the adapter — read its full body
  3. The map_input function — read how input items map to Bedrock messages

For each parameter you find, quote the EXACT line(s) of code that handle it and state:
  OpenAI param name  →  Bedrock field name  (with code quote)

Document at least 6 parameter mappings. Include: model, instructions, input,
temperature/inferenceConfig, tools/toolConfig, and stream handling.
"""

_PROMPT_MODEL_OVERRIDES = f"""\
You are working in the stdapi.ai source tree at {_STDAPI_SRC}.

Find ALL model-specific behavior override files in the chat module and document
the custom logic each one implements.

You MUST:
  1. List ALL Python files inside stdapi/models/chat/ using shell commands
  2. Read _default.py briefly to understand what the base class provides
  3. For each model-specific file:
     - Read the file
     - State which model family it targets
     - Quote at least one overridden method signature
     - Explain in 1 sentence what custom behavior it adds

Document at least 4 model-specific files with real code quotes.
"""


# ---------------------------------------------------------------------------
# Tests: request pipeline tracing  (target ~2-4 tool calls)
# ---------------------------------------------------------------------------


@pytest.mark.agentic
@_SKIP_NO_CODEX
@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
class TestCodexPipeline:
    """Codex traces multi-file execution paths via /v1/responses with shell tools."""

    def test_trace_request_pipeline(
        self,
        request: pytest.FixtureRequest,
        model_config: dict,  # type: ignore[type-arg]
        codex_base_url: str,
    ) -> None:
        """Trace POST /v1/responses from route handler to Bedrock converse().

        Requires multi-file shell exploration: route → handler → adapter → Bedrock.
        Target: ≥2 shell tool calls.
        """
        events = _run_codex(
            base_url=codex_base_url,
            prompt=_PROMPT_REQUEST_PIPELINE,
            model_env=model_config["model_env"],
            extra_env=model_config["extra_env"],
            timeout=model_config.get("timeout", 600),
        )
        _log_metrics(events, model_config["model_env"], "test_trace_request_pipeline")
        result = _assert_result(events, contains="converse", min_tool_calls=2)
        assert any(
            kw in result.lower()
            for kw in ("create_response", "translate_request", "map_input", "_default")
        ), f"Expected core function names in result:\n{result}"

    def test_trace_streaming_path(
        self,
        request: pytest.FixtureRequest,
        model_config: dict,  # type: ignore[type-arg]
        codex_base_url: str,
    ) -> None:
        """Trace the streaming code path from stream=True to SSE output.

        Requires reading the adapter and SSE formatting code.
        Target: ≥2 shell tool calls.
        """
        events = _run_codex(
            base_url=codex_base_url,
            prompt=_PROMPT_STREAMING_PATH,
            model_env=model_config["model_env"],
            extra_env=model_config["extra_env"],
            timeout=model_config.get("timeout", 600),
        )
        _log_metrics(events, model_config["model_env"], "test_trace_streaming_path")
        result = _assert_result(events, min_tool_calls=2)
        assert any(
            kw in result.lower()
            for kw in ("stream", "sse", "event", "format_stream", "converse_stream")
        ), f"Expected streaming-related keywords in result:\n{result}"


# ---------------------------------------------------------------------------
# Tests: code analysis  (target ~3-5 tool calls)
# ---------------------------------------------------------------------------


@pytest.mark.agentic
@_SKIP_NO_CODEX
@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
class TestCodexAnalysis:
    """Codex performs multi-file analysis tasks via /v1/responses with shell tools."""

    def test_audit_parameter_mapping(
        self,
        request: pytest.FixtureRequest,
        model_config: dict,  # type: ignore[type-arg]
        codex_base_url: str,
    ) -> None:
        """Audit the Responses API → Bedrock parameter translation across source files.

        Requires reading types, adapter, and map_input.
        Target: ≥2 shell tool calls.
        """
        events = _run_codex(
            base_url=codex_base_url,
            prompt=_PROMPT_PARAMETER_MAPPING,
            model_env=model_config["model_env"],
            extra_env=model_config["extra_env"],
            timeout=model_config.get("timeout", 600),
        )
        _log_metrics(events, model_config["model_env"], "test_audit_parameter_mapping")
        result = _assert_result(events, min_tool_calls=2)
        assert any(
            kw in result.lower()
            for kw in (
                "instructions",
                "inferencecconfig",
                "temperature",
                "toolconfig",
                "messages",
            )
        ), f"Expected parameter names in result:\n{result}"

    def test_enumerate_model_overrides(
        self,
        request: pytest.FixtureRequest,
        model_config: dict,  # type: ignore[type-arg]
        codex_base_url: str,
    ) -> None:
        """Find and explain all model-specific override files in stdapi/models/chat/.

        Requires glob + reading multiple files.
        Target: ≥3 shell tool calls.
        """
        events = _run_codex(
            base_url=codex_base_url,
            prompt=_PROMPT_MODEL_OVERRIDES,
            model_env=model_config["model_env"],
            extra_env=model_config["extra_env"],
            timeout=model_config.get("timeout", 600),
        )
        _log_metrics(
            events, model_config["model_env"], "test_enumerate_model_overrides"
        )
        result = _assert_result(events, min_tool_calls=3)
        assert any(
            kw in result.lower()
            for kw in (
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
        ), f"Expected model family names or override mentions in result:\n{result}"
