"""Podman discovery, image building and sandboxed execution for the agentic lane.

The agentic CLIs are third-party binaries driven by a language model, so they run in a
container rather than on the host: locked-down (no capabilities, read-only root
filesystem, no new privileges), given only the gateway source read-only, and reachable
to nothing but the one port the test server listens on.

Two execution models share that sandbox: :func:`run_in_container` runs one command to
completion and returns its output, while :func:`start_service_container` leaves a
long-running server up for the duration of a test module and publishes its port back
to the host's loopback.

Networking: the container keeps its own network namespace and pasta forwards its
loopback port to the host's, so the stdapi.ai server under test stays bound to
127.0.0.1 and is never exposed beyond the machine. A service container adds the
reverse direction, again on 127.0.0.1 only.

Ref: https://docs.podman.io/en/latest/markdown/podman-run.1.html
     https://passt.top/passt/about/#pasta
     tests/agentic/Containerfile
"""

from __future__ import annotations

import atexit
import shutil
import socket
import subprocess
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from functools import cache
from hashlib import sha256
from pathlib import Path
from secrets import token_hex
from tempfile import mkdtemp
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

#: Mount point of the host root inside a toolbox container.
_HOST_ROOT_PREFIX = "/run/host"

#: Names left out of a staged source copy; bytecode is noise the CLI never reads.
_STAGING_EXCLUDES = ("__pycache__", "*.pyc")

#: Staging copy of each host source tree, keyed by the tree it was made from.
_staged_sources: dict[Path, Path] = {}

#: Memory ceiling per agentic CLI run; a runaway agent must not swap the host.
_MEMORY_LIMIT = "4g"

#: Process ceiling per run, so a fork bomb in a tool cannot exhaust the host.
_PIDS_LIMIT = "512"

#: Size of the writable tmpfs mounted at /tmp inside the container.
_TMPFS_SIZE = "512m"

#: Seconds between two health probes of a starting service container.
_HEALTH_POLL_INTERVAL = 1.0

#: Seconds a health probe waits for an answer before retrying.
_HEALTH_PROBE_TIMEOUT = 5.0

#: Seconds a container gets to stop cleanly before it is killed.
_SERVICE_STOP_TIMEOUT = "10"

#: Seconds allowed for pulling an image that is not in the local store.
_PULL_TIMEOUT = 1800

#: Marker of a run whose container never started, so the CLI never executed.
#:
#: Under a parallel session the runtime intermittently fails to set up the new
#: network namespace ("write to /proc/sys/net/ipv4/ping_group_range ... OCI
#: runtime error"), and podman reports it as an exit code from the container.
#: Nothing was tested, so it is retried rather than reported as a client failure --
#: which is what it looked like, two per pass, spread across unrelated clients.
_OCI_START_ERROR = "OCI runtime error"

#: Attempts allowed for a container that fails to start.
_START_ATTEMPTS = 3

#: Seconds waited before starting a container again.
_START_RETRY_DELAY = 2.0

#: Names of the service containers this process started and has not stopped.
_running_services: set[str] = set()


@cache
def podman_argv() -> tuple[str, ...] | None:
    """Return the argv prefix that invokes podman, or None when it is unavailable.

    Inside a container the local engine cannot create the user namespace a
    rootless container needs, but the host's podman service is reachable over its
    socket; ``--remote`` drives that instead. The engine then resolves paths in
    the host's namespace, which is why :func:`host_path` exists.

    Returns:
        The argv prefix, or ``None`` if no usable podman was found.
    """
    podman = shutil.which("podman")
    if podman is None:
        return None
    if not Path("/run/.containerenv").is_file():
        return (podman,)
    probe = subprocess.run(  # noqa: S603
        [podman, "--remote", "version"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return (podman, "--remote") if probe.returncode == 0 else (podman,)


def uses_remote_engine() -> bool:
    """True when the container engine runs outside this process's mount namespace."""
    argv = podman_argv()
    return argv is not None and "--remote" in argv


def host_path(path: Path) -> str:
    """Translate *path* into the namespace of the engine that will resolve it.

    A toolbox container sees the host root under ``/run/host``; the host's engine
    does not, so that prefix has to be stripped from any path handed to it.
    Paths outside the prefix (``/tmp`` is shared as-is) are returned unchanged.

    Args:
        path: Path as seen by the test process.

    Returns:
        The equivalent path as a string in the engine's own namespace.
    """
    resolved = str(path)
    if uses_remote_engine() and resolved.startswith(f"{_HOST_ROOT_PREFIX}/"):
        return resolved[len(_HOST_ROOT_PREFIX) :]
    return resolved


#: Shortest env value treated as a secret; below it, a match is coincidence.
_MIN_SECRET_LENGTH = 12


def _redacted(text: str, env: Mapping[str, str]) -> str:
    """Blank out any secret from *env* a container echoed into *text*.

    A misconfigured image often prints its own configuration at startup, and
    that text ends up in a pytest failure and from there in CI output.

    Args:
        text: Container output about to be reported.
        env: Environment the container was started with.

    Returns:
        The text with every long env value replaced.
    """
    for value in env.values():
        if len(value) >= _MIN_SECRET_LENGTH:
            text = text.replace(value, "***")
    return text


@contextmanager
def _env_flags(env: Mapping[str, str]) -> Iterator[list[str]]:
    """Yield the podman flags carrying *env*, keeping its values off the argv.

    A process command line is readable through ``/proc`` by any user on the
    machine, and these values include the gateway's live API key, so they travel
    through a private file removed as soon as podman has read it.  The file lives
    outside every mount, so no container ever sees it.

    Args:
        env: Environment variables to set inside the container.

    Yields:
        Flags to append to the podman command; empty when *env* is.
    """
    if not env:
        yield []
        return
    directory = Path(mkdtemp(prefix="agentic-env-"))
    directory.chmod(0o700)
    path = directory / "env"
    path.write_text("".join(f"{k}={v}\n" for k, v in env.items()), encoding="utf-8")
    path.chmod(0o600)
    try:
        yield ["--env-file", host_path(path)]
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@cache
def _staging_root() -> Path:
    """Directory holding the read-only source copies handed to containers.

    The repository is never bind-mounted directly. Under SELinux a project tree
    carries whatever label it was last given, and a tree stamped with one
    container's private MCS categories is unreadable from every other container --
    the CLI then reports an empty source tree and the run fails for a reason that
    looks nothing like a labelling problem. Copying into a directory this process
    owns makes the mount readable under any policy without ever relabelling the
    repository, which is a host-wide side effect a test suite must not have.
    """
    root = Path(mkdtemp(prefix="stdapi-agentic-src-"))
    atexit.register(shutil.rmtree, root, True)  # noqa: FBT003
    return root


def _staged(source: Path) -> Path:
    """Return the staging copy of *source*, making it on first use.

    Args:
        source: Host directory to expose to a container read-only.

    Returns:
        Path of the copy, inside :func:`_staging_root`.
    """
    staged = _staged_sources.get(source)
    if staged is None:
        staged = _staging_root() / str(len(_staged_sources)) / source.name
        shutil.copytree(
            source, staged, ignore=shutil.ignore_patterns(*_STAGING_EXCLUDES)
        )
        _staged_sources[source] = staged
    return staged


def image_tag(packages: Sequence[str], containerfile: str) -> str:
    """Return the image tag for a package set, keyed on its content.

    The digest covers both the package list and the Containerfile, so editing
    either invalidates the tag and the next run rebuilds instead of silently
    reusing a stale image. Two image groups therefore never share a tag: they
    differ in their packages, in their build file, or in both.

    Args:
        packages: npm specifiers baked into the image.
        containerfile: Build file name, relative to this directory.

    Returns:
        A ``stdapi-agentic:<digest>`` tag.
    """
    path = Path(__file__).parent / containerfile
    digest = sha256(
        f"{' '.join(sorted(packages))}\n{path.read_text()}".encode()
    ).hexdigest()[:12]
    return f"stdapi-agentic:{digest}"


def image_exists(tag: str) -> bool:
    """True when *tag* is already present in the local podman image store."""
    argv = podman_argv()
    assert argv is not None
    return (
        subprocess.run(  # noqa: S603
            [*argv, "image", "exists", tag],
            capture_output=True,
            timeout=60,
            check=False,
        ).returncode
        == 0
    )


def build_image(
    packages: Sequence[str], containerfile: str, *, refresh: bool = False
) -> str:
    """Build the agentic image for *packages* if it is not already present.

    Args:
        packages: npm specifiers to install into the image.
        containerfile: Build file name, relative to this directory.
        refresh: Rebuild from scratch, re-resolving ``@latest`` to pick up new
            CLI releases, instead of reusing the cached image.

    Returns:
        The tag of the usable image.

    Raises:
        RuntimeError: If podman is unavailable or the build fails.
    """
    argv = podman_argv()
    if argv is None:
        msg = "podman is required to build the agentic image"
        raise RuntimeError(msg)

    tag = image_tag(packages, containerfile)
    if image_exists(tag) and not refresh:
        return tag

    context = Path(__file__).parent
    cmd = [
        *argv,
        "build",
        "-f",
        host_path(context / containerfile),
        "--build-arg",
        f"PACKAGES={' '.join(packages)}",
        "-t",
        tag,
    ]
    if refresh:
        # Re-resolve "@latest" and the base image instead of reusing layer cache.
        cmd += ["--no-cache", "--pull=always"]
    cmd.append(host_path(context))

    result = subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, timeout=1800, check=False
    )
    if result.returncode != 0:
        msg = (
            f"podman build failed ({result.returncode}) for {tag}\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
        raise RuntimeError(msg)
    return tag


def installed_versions(tag: str) -> dict[str, str]:
    """Return ``{npm package: version}`` recorded in the image at build time.

    Args:
        tag: Image to inspect.

    Returns:
        Mapping of package name to installed version; empty when unreadable.
    """
    argv = podman_argv()
    if argv is None:
        return {}
    result = subprocess.run(  # noqa: S603
        [
            *argv,
            "run",
            "--rm",
            "--network=none",
            tag,
            "cat",
            "/opt/agentic-packages.json",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        return {}
    from json import JSONDecodeError, loads  # noqa: PLC0415

    try:
        dependencies = loads(result.stdout).get("dependencies", {})
    except JSONDecodeError:
        return {}
    return {name: info.get("version", "?") for name, info in dependencies.items()}


def _remove_container(name: str) -> None:
    """Force-remove the named container, best effort.

    Args:
        name: Container name to remove.
    """
    argv = podman_argv()
    if argv is None:
        return
    with suppress(OSError, subprocess.SubprocessError):
        subprocess.run(  # noqa: S603
            [*argv, "rm", "--force", "--time", _SERVICE_STOP_TIMEOUT, name],
            capture_output=True,
            timeout=120,
            check=False,
        )


def run_in_container(
    *,
    image: str,
    argv: Sequence[str],
    workdir: Path,
    mounts: Mapping[Path, str],
    env: Mapping[str, str],
    forward_port: int | None,
    timeout: int,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run *argv* inside the agentic image and return the completed process.

    The container gets no capabilities, no new privileges, a read-only root
    filesystem and a memory/PID ceiling. *workdir* is the only writable bind
    mount; *mounts* are all read-only, and are served from a staging copy rather
    than the tree itself (see :func:`_staging_root`). Nothing outside them is
    visible, which is what keeps ``tests/.env`` and the host's credentials away
    from the CLI.

    ``--userns=keep-id`` maps the host user to the same UID inside, so files the
    CLI writes into *workdir* stay owned by the test runner.

    A run that exceeds *timeout* is removed by name: the timeout kills this
    podman client and leaves the container up, and an orphaned agent keeps
    running -- and keeps billing whatever it is calling.

    Args:
        image: Image tag to run.
        argv: Command and arguments to execute in the container.
        workdir: Host directory bind-mounted read-write at ``/work``.
        mounts: Host path to read-only container path.
        env: Environment variables to set in the container.
        forward_port: Host loopback port to expose on the container's loopback;
            None when the target is an external URL the container can route to.
        timeout: Seconds before the run is killed.
        stdin: Optional text piped to the process.

    Returns:
        The completed process, with text stdout/stderr.

    Raises:
        RuntimeError: If podman is unavailable.
        subprocess.TimeoutExpired: If the run exceeds *timeout*.
    """
    podman = podman_argv()
    if podman is None:
        msg = "podman is required to run agentic tests"
        raise RuntimeError(msg)

    name = f"stdapi-agentic-run-{token_hex(6)}"
    cmd = [
        *podman,
        "run",
        "--rm",
        # Named so the timeout path below can remove it; "--rm" covers nothing
        # there, because the signal reaches this client and not the container.
        "--name",
        name,
        # Own netns; pasta forwards the container's loopback port to the host's,
        # so the server under test never binds anything but 127.0.0.1.
        f"--network=pasta:-T,{forward_port}" if forward_port else "--network=pasta",
        "--read-only",
        f"--tmpfs=/tmp:rw,size={_TMPFS_SIZE},mode=1777",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--userns=keep-id",
        f"--memory={_MEMORY_LIMIT}",
        f"--pids-limit={_PIDS_LIMIT}",
        # ",Z" relabels this mount for SELinux; without it every write is denied
        # under an enforcing policy. Only the per-test directory is relabelled --
        # the read-only source mounts below are readable as-is, so the repository's
        # own labels are never touched.
        "-v",
        f"{host_path(workdir)}:/work:rw,Z",
        "-w",
        "/work",
    ]
    # Only attach stdin for tools that read their prompt from it. Codex treats an
    # open stdin as extra input to append and stalls waiting on it.
    if stdin is not None:
        cmd.append("--interactive")
    for source, target in mounts.items():
        # ",z" relabels the staging copy as shared so any container may read it.
        # Only the copy is touched; the repository keeps the label it had.
        cmd += ["-v", f"{host_path(_staged(source))}:{target}:ro,z"]

    with _env_flags(env) as env_flags:
        cmd += env_flags
        cmd.append(image)
        cmd.extend(argv)

        for attempt in range(_START_ATTEMPTS):
            try:
                if stdin is None:
                    process = subprocess.run(  # noqa: S603, PLW1510
                        cmd,
                        stdin=subprocess.DEVNULL,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                else:
                    process = subprocess.run(  # noqa: S603, PLW1510
                        cmd,
                        input=stdin,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
            except subprocess.TimeoutExpired:
                _remove_container(name)
                raise
            if process.returncode == 0 or _OCI_START_ERROR not in process.stderr:
                return process
            if attempt < _START_ATTEMPTS - 1:
                _remove_container(name)
                time.sleep(_START_RETRY_DELAY)
        return process


# ---------------------------------------------------------------------------
# Service containers — a server left running for the duration of a test module
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceContainer:
    """A detached container serving one port back to the tests.

    Attributes:
        name: Container name, unique per start.
        image: Image reference it was started from.
        port: Port the service listens on inside the container, published on the
            same port of the host's loopback.
        workdir: Host directory bind-mounted read-write at ``/work``.
        env: Environment the container was started with, holding the secrets
            :meth:`logs` redacts; kept out of the repr for the same reason.
    """

    name: str
    image: str
    port: int
    workdir: Path
    env: Mapping[str, str] = field(default_factory=dict, repr=False)

    @property
    def base_url(self) -> str:
        """HTTP base URL the tests reach this service at."""
        return f"http://127.0.0.1:{self.port}"

    def logs(self) -> str:
        """Return everything the container has written to stdout and stderr.

        A misconfigured service often prints its own configuration -- API key
        included -- so the output is redacted here, before any caller can embed
        it in a failure report.

        Returns:
            The combined redacted output, or an empty string when podman cannot
            report it.
        """
        argv = podman_argv()
        if argv is None:
            return ""
        result = subprocess.run(  # noqa: S603
            [*argv, "logs", self.name],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return _redacted(f"{result.stdout}{result.stderr}", self.env)


def pull_image(image: str, *, refresh: bool = False) -> None:
    """Fetch *image* into the local store unless it is already there.

    Args:
        image: Fully qualified image reference.
        refresh: Pull again even when the reference resolves locally, to
            re-resolve a moving tag such as ``:main`` or ``:latest``.

    Raises:
        RuntimeError: If podman is unavailable or the pull fails.
    """
    argv = podman_argv()
    if argv is None:
        msg = "podman is required to pull a service image"
        raise RuntimeError(msg)
    if image_exists(image) and not refresh:
        return
    result = subprocess.run(  # noqa: S603
        [*argv, "pull", image],
        capture_output=True,
        text=True,
        timeout=_PULL_TIMEOUT,
        check=False,
    )
    if result.returncode != 0:
        msg = f"podman pull failed ({result.returncode}) for {image}\n{result.stderr[-2000:]}"
        raise RuntimeError(msg)


def _is_running(name: str) -> bool:
    """True while the named container is still up."""
    argv = podman_argv()
    assert argv is not None
    result = subprocess.run(  # noqa: S603
        [*argv, "inspect", "-f", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result.stdout.strip() == "true"


def _is_healthy(container: ServiceContainer, health_path: str | None) -> bool:
    """True once the service answers on its published port.

    Args:
        container: Service being probed.
        health_path: Path of an HTTP endpoint to poll; None probes the TCP port
            only, which is all a non-HTTP protocol exposes.

    Returns:
        Whether the service is accepting requests.
    """
    if health_path is None:
        with socket.socket() as probe:
            probe.settimeout(_HEALTH_PROBE_TIMEOUT)
            return probe.connect_ex(("127.0.0.1", container.port)) == 0
    try:
        response = httpx.get(
            f"{container.base_url}{health_path}", timeout=_HEALTH_PROBE_TIMEOUT
        )
    except httpx.HTTPError:
        return False
    # Any answered request proves the service is serving; the endpoint's own
    # status is the caller's business, not the harness's.
    return response.status_code < 500


@cache
def _register_service_cleanup() -> None:
    """Arrange for service containers still up at interpreter exit to be removed."""
    atexit.register(_stop_all_services)


def _stop_all_services() -> None:
    """Force-remove every service container this process still has running."""
    for name in tuple(_running_services):
        _remove_container(name)
        _running_services.discard(name)


def start_service_container(
    *,
    image: str,
    port: int,
    workdir: Path,
    env: Mapping[str, str],
    forward_port: int | None,
    argv: Sequence[str] = (),
    data_dirs: Sequence[str] = (),
    health_path: str | None = None,
    startup_timeout: int,
    read_only: bool = True,
    user: str | None = None,
    refresh: bool = False,
) -> ServiceContainer:
    """Start a pulled image detached and wait until its port answers.

    The counterpart of :func:`run_in_container` for clients shaped as servers
    rather than as one-shot commands: the image is pulled instead of built, the
    container outlives the call, and pasta forwards traffic both ways -- outbound
    to the gateway under test (``-T``) and inbound to the service (``-t``), the
    latter bound to the host's loopback so the service is never exposed off the
    machine. The sandbox is otherwise the one-shot runner's: no capabilities, no
    new privileges, a read-only root and ``/work`` the only writable mount.

    The service must listen on all interfaces inside the container: pasta
    forwards an inbound connection to the container's own address, so a server
    bound to the container's loopback is unreachable and the health poll fails.

    Args:
        image: Fully qualified image reference, pulled if absent locally.
        port: Port the service listens on, published on the same host port.
        workdir: Host directory bind-mounted read-write at ``/work``.
        env: Environment variables to set in the container.
        forward_port: Host loopback port to expose on the container's loopback,
            typically the gateway under test; None when it needs none.
        argv: Command overriding the image's own, if any.
        data_dirs: Directory names created under *workdir* before the start, for
            state the service writes (``DATA_DIR``, ``HOME``, caches). Created
            here so they belong to the test runner rather than to the container.
        health_path: HTTP path polled until the service answers; None probes the
            TCP port instead, for a service that speaks no HTTP.
        startup_timeout: Seconds allowed for the service to become reachable.
        read_only: Keep the container's root filesystem read-only. Turn it off
            only for an image that writes outside ``/work`` and ``/tmp``, and say
            in the caller why.
        user: ``UID:GID`` to run as, overriding the image's own ``USER``. An
            image running as root needs it: under ``--userns=keep-id`` container
            root maps to a subordinate host UID, which -- with no capabilities --
            cannot write into *workdir* and would leave the test runner unable to
            delete whatever it did write. Pass the owner of *workdir*.
        refresh: Re-pull the image before starting, for a moving tag.

    Returns:
        A handle to the running container.

    Raises:
        RuntimeError: If podman is unavailable, the run fails, or the service
            does not answer within *startup_timeout*.
    """
    podman = podman_argv()
    if podman is None:
        msg = "podman is required to run agentic tests"
        raise RuntimeError(msg)
    pull_image(image, refresh=refresh)
    for name in data_dirs:
        (workdir / name).mkdir(parents=True, exist_ok=True)

    container = ServiceContainer(
        name=f"stdapi-agentic-svc-{token_hex(6)}",
        image=image,
        port=port,
        workdir=workdir,
        env=dict(env),
    )
    pasta = [f"-t,127.0.0.1/{port}"]
    if forward_port:
        pasta.insert(0, f"-T,{forward_port}")
    cmd = [
        *podman,
        "run",
        "--detach",
        "--name",
        container.name,
        f"--network=pasta:{','.join(pasta)}",
        f"--tmpfs=/tmp:rw,size={_TMPFS_SIZE},mode=1777",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--userns=keep-id",
        f"--memory={_MEMORY_LIMIT}",
        f"--pids-limit={_PIDS_LIMIT}",
        # ",Z" relabels the per-test directory for SELinux, as in the one-shot
        # runner; nothing else is mounted, so no other label is ever touched.
        "-v",
        f"{host_path(workdir)}:/work:rw,Z",
        "-w",
        "/work",
    ]
    if read_only:
        cmd.append("--read-only")
    if user is not None:
        cmd += ["--user", user]

    with _env_flags(env) as env_flags:
        cmd += env_flags
        cmd.append(image)
        cmd.extend(argv)

        started = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, timeout=300, check=False
        )
    if started.returncode != 0:
        msg = (
            f"podman run failed ({started.returncode}) for {image}\n"
            f"{_redacted(started.stderr, env)[-2000:]}"
        )
        raise RuntimeError(msg)
    _running_services.add(container.name)
    _register_service_cleanup()

    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if _is_healthy(container, health_path):
            return container
        if not _is_running(container.name):
            break
        time.sleep(_HEALTH_POLL_INTERVAL)

    logs = container.logs()[-3000:]
    stop_service_container(container)
    msg = (
        f"service container {image} did not answer on port {port} within "
        f"{startup_timeout}s.\nLast output:\n{logs}"
    )
    raise RuntimeError(msg)


def stop_service_container(container: ServiceContainer) -> None:
    """Remove a container started by :func:`start_service_container`.

    Args:
        container: Handle returned when it was started.
    """
    argv = podman_argv()
    if argv is None:
        return
    subprocess.run(  # noqa: S603
        [*argv, "rm", "--force", "--time", _SERVICE_STOP_TIMEOUT, container.name],
        capture_output=True,
        timeout=120,
        check=False,
    )
    _running_services.discard(container.name)
