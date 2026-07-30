"""Podman discovery, image building and sandboxed execution for the agentic lane.

The agentic CLIs are third-party binaries driven by a language model, so they run in a
container rather than on the host: locked-down (no capabilities, read-only root
filesystem, no new privileges), given only the gateway source read-only, and reachable
to nothing but the one port the test server listens on.

Networking: the container keeps its own network namespace and pasta forwards its
loopback port to the host's, so the stdapi.ai server under test stays bound to
127.0.0.1 and is never exposed beyond the machine.

Ref: https://docs.podman.io/en/latest/markdown/podman-run.1.html
     https://passt.top/passt/about/#pasta
     tests/agentic/Containerfile
"""

from __future__ import annotations

import shutil
import subprocess
from functools import cache
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Mount point of the host root inside a toolbox container.
_HOST_ROOT_PREFIX = "/run/host"

#: Memory ceiling per agentic CLI run; a runaway agent must not swap the host.
_MEMORY_LIMIT = "4g"

#: Process ceiling per run, so a fork bomb in a tool cannot exhaust the host.
_PIDS_LIMIT = "512"

#: Size of the writable tmpfs mounted at /tmp inside the container.
_TMPFS_SIZE = "512m"


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


def image_tag(packages: Sequence[str]) -> str:
    """Return the image tag for a package set, keyed on its content.

    The digest covers both the package list and the Containerfile, so editing
    either invalidates the tag and the next run rebuilds instead of silently
    reusing a stale image.

    Args:
        packages: npm specifiers baked into the image.

    Returns:
        A ``stdapi-agentic:<digest>`` tag.
    """
    containerfile = Path(__file__).parent / "Containerfile"
    digest = sha256(
        f"{' '.join(sorted(packages))}\n{containerfile.read_text()}".encode()
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


def build_image(packages: Sequence[str], *, refresh: bool = False) -> str:
    """Build the agentic image for *packages* if it is not already present.

    Args:
        packages: npm specifiers to install into the image.
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

    tag = image_tag(packages)
    if image_exists(tag) and not refresh:
        return tag

    context = Path(__file__).parent
    cmd = [
        *argv,
        "build",
        "-f",
        host_path(context / "Containerfile"),
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
    mount; *mounts* are all read-only. Nothing outside them is visible, which is
    what keeps ``tests/.env`` and the host's credentials away from the CLI.

    ``--userns=keep-id`` maps the host user to the same UID inside, so files the
    CLI writes into *workdir* stay owned by the test runner.

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

    cmd = [
        *podman,
        "run",
        "--rm",
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
        cmd += ["-v", f"{host_path(source)}:{target}:ro"]
    for key, value in env.items():
        cmd += ["-e", f"{key}={value}"]
    cmd.append(image)
    cmd.extend(argv)

    if stdin is None:
        return subprocess.run(  # noqa: S603, PLW1510
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    return subprocess.run(  # noqa: S603, PLW1510
        cmd, input=stdin, capture_output=True, text=True, timeout=timeout
    )
