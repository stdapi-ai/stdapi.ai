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

from ._podman import _redacted, build_image, installed_versions, podman_argv
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
    lane costs a single process regardless of how many tools it drives.

    ``--server-url`` is refused rather than honored: the CLIs are handed
    ``http://127.0.0.1:<forwarded port>`` by :mod:`._tools`, which pasta maps to
    a server on the host's loopback, and an external deployment has no such port
    to forward. Pointing them at the URL instead would also make the lane's
    model-identity assertions unobservable, since they read the server's own log.
    """
    if request.config.getoption("--server-url", default=None):
        pytest.skip(
            "the agentic lane drives its CLIs through a loopback port forwarded "
            "into their container, which no external deployment provides"
        )
    server = start_server()
    yield server
    stop_server(server)


@pytest.fixture
def agentic_workdir(tmp_path: Path) -> Path:
    """Per-test writable directory, the only writable mount inside the container."""
    return tmp_path


#: Line the gateway's stderr starts an unhandled-exception report with.
_TRACEBACK_MARKER = "Traceback (most recent call last)"


@pytest.fixture(autouse=True)
def _no_gateway_traceback(request: pytest.FixtureRequest) -> Generator[None]:
    """Fail the test when the gateway logged an unhandled exception during it.

    A CLI reports a 500 as an opaque "API call failed", and several of them retry
    it silently, so without this the only symptom of a gateway crash is a client
    that gave up. The traceback itself is on the server's stderr, which nothing
    else in the lane reads.
    """
    if podman_argv() is None or "agentic_server" not in request.fixturenames:
        yield
        return
    server: AgenticServer = request.getfixturevalue("agentic_server")
    start = len(server.stderr_lines)
    yield
    produced = server.stderr_lines[start:]
    if any(_TRACEBACK_MARKER in line for line in produced):
        # A traceback can interpolate request headers, so the key is blanked.
        report = _redacted("\n".join(produced), {"api_key": server.api_key})
        pytest.fail(f"the gateway raised while serving this test:\n{report}")


@pytest.fixture(autouse=True)
def _model_identity_check(request: pytest.FixtureRequest) -> Generator[None]:
    """Assert every request the test produced targeted the parametrized model.

    Autouse so no test can forget it: without this check a CLI silently falling
    back to its own default model would still pass, and the test would prove
    nothing about the gateway routing the model it names.

    The podman check comes first, and the server is resolved only for a test
    that drives a registered CLI, so a machine without podman skips the lane --
    and a harness unit test runs -- without paying for a server startup.
    """
    if podman_argv() is None:
        pytest.skip(
            "podman is required for agentic tests: they run the CLIs in a "
            "container so no third-party binary executes on the host"
        )
    tool = tool_under_test(request)
    config = (
        request.getfixturevalue("model_config")
        if "model_config" in request.fixturenames
        else None
    )
    if tool is None or not isinstance(config, ModelConfig):
        yield
        return
    agentic_server: AgenticServer = request.getfixturevalue("agentic_server")
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
