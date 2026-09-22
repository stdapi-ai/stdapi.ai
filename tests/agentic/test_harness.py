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

import json
import socket
import subprocess
from pathlib import Path
from secrets import token_hex
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from . import _podman
from ._podman import (
    _engine_user,
    _env_flags,
    _redacted,
    image_tag,
    start_service_container,
    stop_service_container,
)
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


class TestSecretHygiene:
    """The gateway's live API key never reaches a report or another host user.

    Pure filesystem and string checks: nothing here starts a container.

    Ref: tests/agentic/_podman.py:_redacted
         tests/agentic/_podman.py:_env_flags
    """

    def test_redacted_blanks_every_long_env_value(self) -> None:
        """Each env value long enough to be a secret is replaced everywhere.

        Every failure path that embeds container or CLI output routes through
        this helper, so a value it misses would land verbatim in CI output.

        Ref: tests/agentic/_runner.py:run_agent
        """
        env = {"API_KEY": "k" * 32, "OPENAI_API_KEY": "sk-agentic-000111222333"}
        text = f"auth={env['API_KEY']} again={env['API_KEY']} {env['OPENAI_API_KEY']}"
        redacted = _redacted(text, env)
        assert env["API_KEY"] not in redacted
        assert env["OPENAI_API_KEY"] not in redacted
        assert redacted == "auth=*** again=*** ***"

    def test_redacted_leaves_short_values_alone(self) -> None:
        """A short value ("1", "true") matching by coincidence is not blanked.

        Ref: tests/agentic/_podman.py:_MIN_SECRET_LENGTH
        """
        env = {"DEBUG": "true", "PORT": "8080"}
        assert _redacted("true output on 8080", env) == "true output on 8080"

    def test_env_flags_serves_a_private_file_and_removes_it(self) -> None:
        """Secrets travel via a 0600 file in a 0700 directory, never the argv.

        The argv is world-readable through ``/proc``, so the flags may name the
        file but must not carry a value; the file itself must be closed to
        other users and gone once podman has read it.

        Ref: tests/agentic/_podman.py:_env_flags
        """
        env = {"API_KEY": "k" * 32}
        with _env_flags(env) as flags:
            assert flags[0] == "--env-file"
            assert all(env["API_KEY"] not in flag for flag in flags)
            path = Path(flags[1])
            assert path.parent.stat().st_mode & 0o777 == 0o700
            assert path.stat().st_mode & 0o777 == 0o600
            assert path.read_text(encoding="utf-8") == f"API_KEY={env['API_KEY']}\n"
        assert not path.parent.exists(), "the env file must not outlive the run"

    def test_env_flags_yields_nothing_for_an_empty_environment(self) -> None:
        """No environment means no flags and no file to clean up.

        Ref: tests/agentic/_podman.py:_env_flags
        """
        with _env_flags({}) as flags:
            assert flags == []


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

    def test_a_failed_start_leaves_no_container_behind(
        self, agentic_image: str, agentic_workdir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A container whose start fails is removed, not left in "Created".

        ``podman run --detach`` creates the container before the runtime starts
        it, so an entry point that cannot run leaves one behind for every failed
        attempt unless the harness removes it.

        Ref: https://docs.podman.io/en/latest/markdown/podman-run.1.html
             tests/agentic/_podman.py:start_service_container
        """
        token = token_hex(6)
        monkeypatch.setattr(_podman, "token_hex", lambda _: token)
        with pytest.raises(RuntimeError, match="podman run failed"):
            start_service_container(
                image=agentic_image,
                port=find_free_port(),
                workdir=agentic_workdir,
                env={},
                forward_port=None,
                entrypoint="/nonexistent-stdapi-probe",
                startup_timeout=_PROBE_STARTUP_TIMEOUT,
            )
        podman = _podman.podman_argv()
        assert podman is not None
        exists = subprocess.run(  # noqa: S603
            [*podman, "container", "exists", f"stdapi-agentic-svc-{token}"],
            capture_output=True,
            timeout=60,
            check=False,
        )
        assert exists.returncode == 1, "the failed container was left behind"

    def test_a_start_that_times_out_is_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``podman run`` killed by its timeout still has its container removed.

        The timeout kills the podman client, not the container it was creating,
        so without the removal a hung start leaves one behind. Podman is stubbed:
        a real hang cannot be produced on demand.

        Ref: https://docs.python.org/3/library/subprocess.html#subprocess.run
             tests/agentic/_podman.py:start_service_container
        """
        removed: list[str] = []

        def hang(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:  # noqa: ANN401
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

        monkeypatch.setattr(_podman, "token_hex", lambda _: "hung")
        monkeypatch.setattr(_podman, "podman_argv", lambda: ("podman",))
        monkeypatch.setattr(_podman, "pull_image", lambda *_, **__: None)
        monkeypatch.setattr(_podman, "_user_flags", lambda *_: [])
        monkeypatch.setattr(_podman, "_remove_container", removed.append)
        monkeypatch.setattr(subprocess, "run", hang)
        with pytest.raises(subprocess.TimeoutExpired):
            start_service_container(
                image="stdapi-agentic:unused",
                port=find_free_port(),
                workdir=tmp_path,
                env={},
                forward_port=None,
                startup_timeout=_PROBE_STARTUP_TIMEOUT,
            )
        assert removed == ["stdapi-agentic-svc-hung"]


class TestKeepIdMapping:
    """The explicit ID maps standing in for ``--userns=keep-id``.

    Ref: https://docs.podman.io/en/latest/markdown/podman-run.1.html#userns-mode
         tests/agentic/_podman.py:_engine_user
    """

    @staticmethod
    def _engine_user_for(
        monkeypatch: pytest.MonkeyPatch,
        uidmap: list[dict[str, int]],
        gidmap: list[dict[str, int]],
    ) -> tuple[tuple[str, ...], str]:
        """Run the uncached ``_engine_user`` against a stubbed ``podman info``.

        Args:
            monkeypatch: Fixture used to stub podman.
            uidmap: ``Host.IDMappings.uidmap`` the stub reports.
            gidmap: ``Host.IDMappings.gidmap`` the stub reports.

        Returns:
            What ``_engine_user`` returns for that mapping.
        """
        info = json.dumps({"uidmap": uidmap, "gidmap": gidmap})
        monkeypatch.setattr(_podman, "podman_argv", lambda: ("podman",))
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *_, **__: subprocess.CompletedProcess((), 0, info, ""),
        )
        return _engine_user.__wrapped__()

    def test_no_subordinate_range_is_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An engine with no subordinate IDs cannot reproduce keep-id, and says so.

        Without the range there is nothing to map the container's other IDs
        onto; building flags anyway would hand podman a zero-sized range and
        fail later with an error naming nothing the reader can act on.

        Ref: https://docs.podman.io/en/latest/markdown/podman-info.1.html
             tests/agentic/_podman.py:_engine_user
        """
        own_only = [{"container_id": 0, "host_id": 1000, "size": 1}]
        with pytest.raises(RuntimeError, match="not a rootless mapping"):
            self._engine_user_for(monkeypatch, own_only, own_only)

    def test_maps_the_engine_user_onto_itself(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A typical rootless mapping yields keep-id's three ranges per ID kind.

        Distinct UID and GID catch a swapped loop or ``UID:GID`` order that a
        host with ``1000:1000`` would hide.

        Ref: https://docs.podman.io/en/latest/markdown/podman-run.1.html#userns-mode
             tests/agentic/_podman.py:_engine_user
        """
        flags, user = self._engine_user_for(
            monkeypatch,
            [
                {"container_id": 0, "host_id": 1000, "size": 1},
                {"container_id": 1, "host_id": 524288, "size": 65536},
            ],
            [
                {"container_id": 0, "host_id": 1001, "size": 1},
                {"container_id": 1, "host_id": 524288, "size": 65536},
            ],
        )
        assert flags == (
            "--uidmap=0:1:1000",
            "--uidmap=1000:0:1",
            "--uidmap=1001:1001:64536",
            "--gidmap=0:1:1001",
            "--gidmap=1001:0:1",
            "--gidmap=1002:1002:64535",
        )
        assert user == "1000:1001"

    def test_an_id_above_the_range_is_clipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ID at or above the subordinate range size clips instead of failing.

        keep-id maps only ``min(id, size)`` IDs below the user and drops the
        range above it; a directory-service UID such as 1234567 with the
        default 65536 subordinate IDs must still run the lane.

        Ref: https://docs.podman.io/en/latest/markdown/podman-run.1.html#userns-mode
             tests/agentic/_podman.py:_engine_user
        """
        flags, user = self._engine_user_for(
            monkeypatch,
            [
                {"container_id": 0, "host_id": 1234567, "size": 1},
                {"container_id": 1, "host_id": 524288, "size": 65536},
            ],
            [
                {"container_id": 0, "host_id": 65536, "size": 1},
                {"container_id": 1, "host_id": 524288, "size": 65536},
            ],
        )
        assert flags == (
            "--uidmap=0:1:65536",
            "--uidmap=1234567:0:1",
            "--gidmap=0:1:65536",
            "--gidmap=65536:0:1",
        )
        assert user == "1234567:65536"
