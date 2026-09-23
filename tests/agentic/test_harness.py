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
from ._runner import ModelConfig, assert_result
from ._server import find_free_port
from ._tools import (
    AGENTIC_TOOLS,
    DEFAULT_IMAGE_GROUP,
    IMAGE_GROUPS,
    AgenticResult,
    AgenticTool,
    _codex_parse,
    npm_packages,
)
from .test_inspect_ai import _anthropic_batch_processed, _openai_batch_processed

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


def _codex_jsonl(*events: dict[str, object]) -> str:
    """Render *events* as the JSONL ``codex exec --json`` normally emits."""
    return "\n".join(json.dumps(event) for event in events)


class TestCodexTrace:
    """``_codex_parse`` only grades commands that actually ran and read something.

    Nothing here starts a container: these are unit tests of the pure JSONL
    parser, added because the trace-grading path (``assert_result``'s
    ``trace_any_of``, ``_codex_parse``'s ``commands``) had no offline coverage.

    Ref: tests/agentic/_tools.py:_codex_parse
         https://github.com/stdapi-ai/stdapi.ai/issues/299
    """

    def test_a_failed_command_is_dropped_from_commands_but_still_counted_as_a_step(
        self,
    ) -> None:
        """A nonzero ``exit_code`` excludes a command from grading, not from the step count.

        A command that named the right file but failed to run (a missing mount, a
        wrong path) is not evidence the agent read anything, even though the CLI
        still spent a turn on it.
        """
        stdout = _codex_jsonl(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "sed -n '1,5p' _default.py",
                    "exit_code": 2,
                    "aggregated_output": "",
                },
            },
            {"type": "turn.completed", "usage": {}},
        )
        result = _codex_parse(stdout)
        assert result.commands == ()
        assert result.steps == 1

    def test_a_command_with_no_output_is_dropped_from_commands(self) -> None:
        """A command that exits 0 but prints nothing is not evidence of a real read."""
        stdout = _codex_jsonl(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "grep -c _prepare_converse_request _default.py",
                    "exit_code": 0,
                    "aggregated_output": "",
                },
            },
            {"type": "turn.completed", "usage": {}},
        )
        assert _codex_parse(stdout).commands == ()

    def test_a_command_that_ran_and_produced_output_is_kept(self) -> None:
        """A command exiting 0 with output is graded, in the order it ran."""
        stdout = _codex_jsonl(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "grep -n _prepare_converse_request _default.py",
                    "exit_code": 0,
                    "aggregated_output": "42:def _prepare_converse_request(...):",
                },
            },
            {"type": "turn.completed", "usage": {}},
        )
        assert _codex_parse(stdout).commands == (
            "grep -n _prepare_converse_request _default.py",
        )

    def test_a_command_carrying_neither_field_is_kept(self) -> None:
        """A trace missing ``exit_code``/``aggregated_output`` is graded as before.

        The gate only applies when the trace actually carries the field, so an
        older or differently-shaped Codex build is not newly broken by it.
        """
        stdout = _codex_jsonl(
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "command": "cat _default.py"},
            },
            {"type": "turn.completed", "usage": {}},
        )
        assert _codex_parse(stdout).commands == ("cat _default.py",)


class TestTraceAnyOf:
    """``assert_result``'s ``trace_any_of`` grades the tool trace, not the prose.

    Ref: tests/agentic/_runner.py:assert_result
    """

    @staticmethod
    def _result(*, commands: tuple[str, ...]) -> AgenticResult:
        return AgenticResult(
            text="a long enough answer to pass the length floor",
            steps=len(commands) or 1,
            input_tokens=1,
            output_tokens=1,
            commands=commands,
        )

    def test_a_matching_command_passes(self) -> None:
        """A command containing the keyword, case-insensitively, satisfies the check."""
        result = self._result(commands=("GREP -n _prepare_converse_request x.py",))
        assert_result(
            result,
            config=ModelConfig(model="test"),
            trace_any_of=("_prepare_converse_request",),
        )

    def test_no_matching_command_fails(self) -> None:
        """A trace that never names the keyword fails, even with a good answer."""
        result = self._result(commands=("ls .",))
        with pytest.raises(pytest.fail.Exception):
            assert_result(
                result,
                config=ModelConfig(model="test"),
                trace_any_of=("_prepare_converse_request",),
            )

    def test_empty_commands_fails_rather_than_passing_vacuously(self) -> None:
        """A tool that exposes no trace fails this check instead of skipping it."""
        result = self._result(commands=())
        with pytest.raises(pytest.fail.Exception):
            assert_result(
                result,
                config=ModelConfig(model="test"),
                trace_any_of=("_prepare_converse_request",),
            )


class TestBatchProcessedPredicates:
    """The predicates deciding whether a timed-out batch ever started.

    Ref: tests/agentic/test_inspect_ai.py:_skip_or_fail_unstarted_batch
    """

    @pytest.mark.parametrize(
        ("status", "expected"),
        [("validating", False), ("in_progress", True), ("completed", True)],
    )
    def test_openai_batch_processed_reads_the_status(
        self, status: str, expected: bool
    ) -> None:
        """Only `validating` counts as not yet started; every other status has."""
        assert _openai_batch_processed({"status": status}) is expected

    @pytest.mark.parametrize(
        ("counts", "expected"),
        [
            ({"succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}, False),
            ({"succeeded": 1, "errored": 0, "canceled": 0, "expired": 0}, True),
            ({"errored": 0, "canceled": 0, "expired": 1}, True),
        ],
    )
    def test_anthropic_batch_processed_reads_the_counts(
        self, counts: dict[str, int], expected: bool
    ) -> None:
        """At least one settled request of any kind counts as started."""
        assert _anthropic_batch_processed({"request_counts": counts}) is expected

    def test_anthropic_batch_processed_treats_a_missing_counts_dict_as_unstarted(
        self,
    ) -> None:
        """A batch whose response carries no `request_counts` at all is not started."""
        assert _anthropic_batch_processed({}) is False
