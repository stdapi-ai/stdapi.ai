"""The stdapi.ai server the agentic CLIs are pointed at.

One authenticated server is spawned per test session and shared by every agentic
tool, since the gateway serves the Anthropic and OpenAI routes from the same app.
It binds 127.0.0.1 only; the containers reach it through pasta's loopback
forwarding, so nothing is exposed off the machine.

Its stdout is captured line by line because that JSON request log is the only place
the resolved Bedrock ``model_id`` is observable, which is what lets the tests prove a
CLI's traffic actually reached the model under test.

Ref: stdapi/main.py:app
     stdapi/monitoring.py
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from secrets import token_hex
from typing import TYPE_CHECKING

import httpx

from ._podman import _redacted

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Repository root, whose ``stdapi`` package the server is started from.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Seconds allowed for the server to answer /health after launch.
_STARTUP_TIMEOUT = 60

#: Settings this module sets itself, dropped from the inherited environment first.
#:
#: ``strict_input_validation`` is one of them because this lane runs the gateway
#: as a deployment does, and its default is off. Real clients send fields the
#: OpenAI schema does not define -- Hermes puts ``name`` on a ``tool`` message,
#: which the legacy ``function`` role carried -- and rejecting those here would
#: report a working client as a gateway failure. What strict mode rejects is
#: asserted in the unit suite, against the schemas rather than against a CLI.
_OVERRIDDEN_SETTINGS = frozenset({"api_key", "strict_input_validation"})


def find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class AgenticServer:
    """A running stdapi.ai server, or a handle to an external one."""

    base_url: str
    api_key: str
    #: Loopback port to forward into the container; None for an external server.
    forward_port: int | None = None
    #: stdout lines appended in real time by a background reader thread.
    logs: list[str] = field(default_factory=list)
    #: stderr lines, kept for diagnosing startup failures.
    stderr_lines: list[str] = field(default_factory=list)
    process: subprocess.Popen[str] | None = field(default=None, repr=False)

    def url(self, route: str) -> str:
        """Return the full URL of *route* (e.g. ``/anthropic``) on this server."""
        return f"{self.base_url}{route}"

    def log_entries(self, start: int) -> Iterator[dict[str, object]]:
        """Yield the JSON log events recorded from index *start* onwards.

        Args:
            start: Index into :attr:`logs` taken before the test ran.

        Yields:
            Each parsable JSON log line as a dict; non-JSON lines are skipped.
        """
        for line in self.logs[start:]:
            if not line.startswith("{"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _stream(pipe: object, sink: list[str]) -> None:
    """Append each line of *pipe* to *sink* until the pipe closes."""
    assert pipe is not None
    # extend() consumes the iterator lazily, so lines still land in *sink* as the
    # server writes them rather than only when the pipe closes.
    sink.extend(line.rstrip() for line in pipe)  # type: ignore[attr-defined]


def start_server() -> AgenticServer:
    """Spawn an authenticated stdapi.ai server on a free loopback port.

    Runs on the current interpreter rather than through ``uv run``, because the
    wrapper does not forward the kill signal to the server it spawns -- which
    would leak a listening process -- and it serialises startup on the uv
    environment lock.

    Returns:
        A handle to the running server, already answering /health.

    Raises:
        RuntimeError: If the server does not become healthy in time.
    """
    port = find_free_port()
    api_key = token_hex(16)

    # The suite's conftest sets a lowercase ``api_key`` in os.environ, which wins
    # the settings' case-insensitive lookup; drop every variant so the key below
    # is the one the server actually enforces.
    from os import environ  # noqa: PLC0415

    env = {k: v for k, v in environ.items() if k.lower() not in _OVERRIDDEN_SETTINGS}
    env["API_KEY"] = api_key
    env["STRICT_INPUT_VALIDATION"] = "false"

    process = subprocess.Popen(  # noqa: S603
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
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    server = AgenticServer(
        base_url=f"http://127.0.0.1:{port}",
        api_key=api_key,
        forward_port=port,
        process=process,
    )
    threading.Thread(
        target=_stream, args=(process.stdout, server.logs), daemon=True
    ).start()
    threading.Thread(
        target=_stream, args=(process.stderr, server.stderr_lines), daemon=True
    ).start()

    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{server.base_url}/health", timeout=2.0).status_code == 200:
                return server
        except httpx.HTTPError:
            pass
        time.sleep(0.5)

    process.kill()
    # A failing startup often dumps the settings, so the key is blanked out.
    startup_log = _redacted(
        "\n".join(server.stderr_lines[-30:] + server.logs[-10:]), {"api_key": api_key}
    )
    msg = f"stdapi server failed to start on port {port}.\nLast output:\n{startup_log}"
    raise RuntimeError(msg)


def stop_server(server: AgenticServer) -> None:
    """Terminate a server started by :func:`start_server`, killing it if it hangs."""
    if server.process is None:
        return
    server.process.terminate()
    try:
        server.process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.process.kill()
