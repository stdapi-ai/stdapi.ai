"""The agentic lane's own primitives: image groups and service containers.

Both are shared plumbing every tool module leans on, and both fail in ways that
look like the tool's fault: a mis-named image group silently lands a CLI in the
wrong image, and a service container that never becomes reachable reports as a
client error rather than as a harness one. These tests pin the plumbing itself,
so a tool module's failure means the tool.

Nothing here calls a model, but the lane is opt-in as a whole because it needs
podman and a gateway process.

Ref: https://docs.podman.io/en/latest/markdown/podman-run.1.html
     https://passt.top/passt/about/#pasta
     tests/agentic/_tools.py:IMAGE_GROUPS
     tests/agentic/_podman.py:start_service_container
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from ._podman import image_tag, start_service_container, stop_service_container
from ._server import find_free_port
from ._tools import (
    AGENTIC_TOOLS,
    DEFAULT_IMAGE_GROUP,
    IMAGE_GROUPS,
    AgenticTool,
    npm_packages,
)

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.agentic

#: Value handed to the probe service, to prove the environment reaches it.
_PROBE_VALUE = "stdapi-service-probe"

#: Seconds allowed for the probe service to answer; it is a one-line server.
_PROBE_STARTUP_TIMEOUT = 60

#: Seconds a probe of the published port waits before giving up.
_PROBE_TIMEOUT = 10.0


def _http_probe_argv(port: int) -> tuple[str, ...]:
    """Return the command answering the probe value over HTTP on *port*."""
    script = (
        "require('http')"
        ".createServer((req, res) => res.end(process.env.STDAPI_PROBE))"
        f".listen({port}, '0.0.0.0', () => console.log('probe listening'))"
    )
    return ("node", "-e", script)


def _tcp_probe_argv(port: int) -> tuple[str, ...]:
    """Return the command writing the probe value to any TCP client on *port*."""
    script = (
        "require('net')"
        ".createServer(connection => connection.end(process.env.STDAPI_PROBE))"
        f".listen({port}, '0.0.0.0', () => console.log('probe listening'))"
    )
    return ("node", "-e", script)


def _read_tcp(port: int) -> str:
    """Return what the service writes to a bare TCP client on the host loopback."""
    with socket.create_connection(("127.0.0.1", port), timeout=_PROBE_TIMEOUT) as sock:
        return sock.recv(1024).decode()


def _port_answers(port: int) -> bool:
    """True while something accepts connections on the host's loopback *port*."""
    with socket.socket() as probe:
        probe.settimeout(_PROBE_TIMEOUT)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _unused(*args: object) -> Any:  # noqa: ANN401
    """Stand in for a fake tool's callables, which no test invokes.

    Raises:
        NotImplementedError: Always, if a test ever does invoke one.
    """
    raise NotImplementedError


def _fake_tool(
    *, tool_id: str, npm_package: str | None, image_group: str = DEFAULT_IMAGE_GROUP
) -> AgenticTool:
    """Return a registry entry standing in for a real tool.

    Args:
        tool_id: Tool identifier.
        npm_package: npm specifier, or None for a tool its image already ships.
        image_group: Image group the tool runs in.

    Returns:
        A tool usable by the pure registry helpers.
    """
    return AgenticTool(
        id=tool_id,
        npm_package=npm_package,
        binary=tool_id,
        route="/v1",
        metrics_prefix="TEST-METRICS",
        build=_unused,
        parse=_unused,
        prepare_workdir=_unused,
        attributes_sessions=False,
        image_group=image_group,
    )


class TestImageGroups:
    """Every tool resolves to an image the lane can actually build.

    Ref: tests/agentic/_tools.py:ImageGroup
    """

    def test_every_tool_names_a_registered_group(self) -> None:
        """Each tool's image group exists and names a real Containerfile.

        A typo in a tool's ``image_group`` would otherwise only surface as a
        KeyError inside the first test that runs it.

        Ref: tests/agentic/_tools.py:AGENTIC_TOOLS
        """
        for tool in AGENTIC_TOOLS:
            group = IMAGE_GROUPS.get(tool.image_group)
            assert group is not None, (
                f"{tool.id} names unregistered image group {tool.image_group!r}"
            )
            containerfile = Path(__file__).parent / group.containerfile
            assert containerfile.is_file(), (
                f"image group {group.name!r} names a missing build file: "
                f"{group.containerfile}"
            )

    def test_npm_packages_are_scoped_to_one_group(self) -> None:
        """A group installs its own tools' packages and nobody else's.

        That scoping is the point of the grouping: a heavyweight install must
        not reach the image every other tool waits for.

        Ref: tests/agentic/_tools.py:npm_packages
        """
        tools = (
            _fake_tool(tool_id="shared", npm_package="shared@latest"),
            _fake_tool(
                tool_id="heavy", npm_package="heavy@latest", image_group="heavy"
            ),
        )
        assert npm_packages(DEFAULT_IMAGE_GROUP, tools) == ("shared@latest",)
        assert npm_packages("heavy", tools) == ("heavy@latest",)

    def test_a_tool_without_an_npm_package_installs_nothing(self) -> None:
        """A tool its image already ships contributes no npm specifier.

        Ref: tests/agentic/_tools.py:AgenticTool
        """
        preinstalled = _fake_tool(tool_id="preinstalled", npm_package=None)
        assert npm_packages(DEFAULT_IMAGE_GROUP, (preinstalled,)) == ()

    def test_image_tag_tracks_the_package_list(self) -> None:
        """Two package sets never share a tag, so an image is never stale.

        Ref: tests/agentic/_podman.py:image_tag
        """
        containerfile = IMAGE_GROUPS[DEFAULT_IMAGE_GROUP].containerfile
        assert image_tag(("a@1",), containerfile) != image_tag(("a@2",), containerfile)
        assert image_tag(("a@1", "b@1"), containerfile) == image_tag(
            ("b@1", "a@1"), containerfile
        )


class TestServiceContainer:
    """A detached container's port reaches the tests, and only them.

    The clients shaped as servers (Open WebUI, wyoming-openai) are unusable
    without this: the one-shot runner only returns once the process exits.

    Ref: tests/agentic/_podman.py:start_service_container
    """

    @pytest.mark.parametrize(
        ("probe_argv", "health_path"),
        [
            pytest.param(_http_probe_argv, "/", id="http-health-poll"),
            pytest.param(_tcp_probe_argv, None, id="tcp-health-poll"),
        ],
    )
    def test_published_port_answers_then_stops(
        self,
        probe_argv: Callable[[int], tuple[str, ...]],
        health_path: str | None,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """The service is reachable while up, and gone once stopped.

        Both health probes are covered: an HTTP endpoint for a web service, a
        bare TCP connect for a protocol exposing no HTTP at all. The payload is
        read back from the environment the container was given, so a service
        started without its configuration cannot pass.

        Ref: tests/agentic/_podman.py:ServiceContainer
        """
        port = find_free_port()
        container = start_service_container(
            image=agentic_image,
            port=port,
            workdir=agentic_workdir,
            env={"STDAPI_PROBE": _PROBE_VALUE, "HOME": "/work/home"},
            forward_port=None,
            argv=probe_argv(port),
            data_dirs=("home", "data"),
            health_path=health_path,
            startup_timeout=_PROBE_STARTUP_TIMEOUT,
        )
        try:
            assert (agentic_workdir / "home").is_dir()
            assert (agentic_workdir / "data").is_dir()
            assert "probe listening" in container.logs()
            if health_path is None:
                assert _read_tcp(port) == _PROBE_VALUE
            else:
                response = httpx.get(container.base_url, timeout=_PROBE_TIMEOUT)
                assert response.text == _PROBE_VALUE
        finally:
            stop_service_container(container)

        assert not _port_answers(port), (
            "the published port still answers after the service was stopped"
        )
