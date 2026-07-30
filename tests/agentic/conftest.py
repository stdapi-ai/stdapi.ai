"""Fixtures shared by every agentic test module.

The whole lane is opt-in (``--agentic``) and container-only: the CLIs are
third-party binaries driven by a model, so they are never executed on the host.
A module joins the lane by exporting a module-level ``TOOL`` and parametrizing on
``model_config``; the identity check below then applies to it automatically.

Ref: tests/agentic/_podman.py
     tests/agentic/_server.py
     tests/agentic/_tools.py:AGENTIC_TOOLS
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ._podman import build_image, installed_versions, podman_argv
from ._runner import (
    ModelConfig,
    any_run_completed,
    assert_model_identity,
    reset_run_tracking,
)
from ._server import AgenticServer, start_server, stop_server
from ._tools import npm_packages

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture(scope="session")
def agentic_image(request: pytest.FixtureRequest) -> str:
    """Container image holding the agentic CLIs, built on first use.

    The tag is derived from the package list and the Containerfile, so editing
    either rebuilds automatically. ``--agentic-rebuild`` forces a fresh install
    to pick up new CLI releases that ``@latest`` would otherwise not re-resolve.
    """
    packages = npm_packages()
    tag = build_image(packages, refresh=request.config.getoption("--agentic-rebuild"))
    versions = installed_versions(tag)
    if versions:
        reporter = request.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                "agentic CLIs: "
                + ", ".join(f"{name}=={ver}" for name, ver in sorted(versions.items()))
            )
    return tag


@pytest.fixture(scope="session")
def agentic_server(request: pytest.FixtureRequest) -> Generator[AgenticServer]:
    """The stdapi.ai server every agentic CLI in the session is pointed at.

    One authenticated server serves both the Anthropic and OpenAI routes, so the
    lane costs a single process regardless of how many tools it drives. With
    ``--server-url`` an external deployment is used instead and the model-identity
    assertions are skipped, since its logs are not observable here.
    """
    external: str | None = request.config.getoption("--server-url", default=None)
    if external:
        from os import getenv  # noqa: PLC0415

        yield AgenticServer(
            base_url=external.rstrip("/"), api_key=getenv("OPENAI_API_KEY", "")
        )
        return
    server = start_server()
    yield server
    stop_server(server)


@pytest.fixture
def agentic_workdir(tmp_path: Path) -> Path:
    """Per-test writable directory, the only writable mount inside the container."""
    return tmp_path


@pytest.fixture(autouse=True)
def _model_identity_check(request: pytest.FixtureRequest) -> Generator[None]:
    """Assert every request the test produced targeted the parametrized model.

    Autouse so no test can forget it: without this check a CLI silently falling
    back to its own default model would still pass, and the test would prove
    nothing about the gateway routing the model it names.

    The podman check comes first, and the server is resolved lazily, so a machine
    without podman skips the lane without paying for a server startup.
    """
    if podman_argv() is None:
        pytest.skip(
            "podman is required for agentic tests: they run the CLIs in a "
            "container so no third-party binary executes on the host"
        )
    agentic_server: AgenticServer = request.getfixturevalue("agentic_server")
    tool = getattr(request.module, "TOOL", None)
    config = (
        request.getfixturevalue("model_config")
        if "model_config" in request.fixturenames
        else None
    )
    if tool is None or not isinstance(config, ModelConfig):
        yield
        return
    log_start = len(agentic_server.logs)
    reset_run_tracking()
    yield
    assert_model_identity(
        tool=tool,
        server=agentic_server,
        log_start=log_start,
        config=config,
        test_name=request.node.originalname or request.node.name,
        ran=any_run_completed(),
    )
