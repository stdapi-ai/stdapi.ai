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
from ._tools import DEFAULT_IMAGE_GROUP, IMAGE_GROUPS, npm_packages

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from ._tools import AgenticTool


def tool_under_test(request: pytest.FixtureRequest) -> AgenticTool | None:
    """Return the CLI the current test drives, if it declares one.

    A module driving one CLI exports a module-level ``TOOL``; one driving the
    same CLI over several gateway routes parametrizes an ``agentic_tool``
    fixture instead.

    Args:
        request: Fixture request of the test.

    Returns:
        The tool, or None for a test that drives no registered CLI.
    """
    if "agentic_tool" in request.fixturenames:
        tool: AgenticTool = request.getfixturevalue("agentic_tool")
        return tool
    return getattr(request.module, "TOOL", None)


@pytest.fixture(scope="session")
def _agentic_images() -> dict[str, str]:
    """Image tag of every group built so far in this session, keyed by group."""
    return {}


@pytest.fixture
def agentic_image(
    request: pytest.FixtureRequest, _agentic_images: dict[str, str]
) -> str:
    """Container image holding the CLI under test, built on first use.

    Each image group is built at most once per session: the tools sharing the
    shared Node.js image never pay for a group whose install tree is large or
    whose base image is another language's.

    The tag is derived from the group's package list and its Containerfile, so
    editing either rebuilds automatically. ``--agentic-rebuild`` forces a fresh
    install to pick up new CLI releases that ``@latest`` would otherwise not
    re-resolve.
    """
    tool = tool_under_test(request)
    group = IMAGE_GROUPS[tool.image_group if tool else DEFAULT_IMAGE_GROUP]
    if (tag := _agentic_images.get(group.name)) is not None:
        return tag
    tag = build_image(
        npm_packages(group.name),
        group.containerfile,
        refresh=request.config.getoption("--agentic-rebuild"),
    )
    _agentic_images[group.name] = tag
    versions = installed_versions(tag)
    if versions:
        reporter = request.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                f"agentic CLIs [{group.name}]: "
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
    tool = tool_under_test(request)
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
