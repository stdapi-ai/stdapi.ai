"""Which dotenv profile a run loads, and which one its subprocesses load.

``tests/conftest.py`` picks the profile at import time from ``sys.argv``, and
publishes the choice so that a pytest-xdist worker -- whose ``sys.argv`` is
execnet's bootstrap code rather than the command line -- can reuse it. Both
halves of that have failed silently in the past: a worker that could not read
the choice loaded ``tests/.env`` over values it had correctly inherited, and a
marker honoured by every child made a nested run load the outer run's profile
instead of the one its own arguments name.

Neither failure is visible in the run that causes it. What surfaces is a distant
test asserting against the wrong gateway or the wrong region, which reads as a
product defect, so the selection is pinned here rather than left to be inferred.

Ref: tests/conftest.py
     https://pytest-xdist.readthedocs.io/en/stable/how-it-works.html
"""

import subprocess
import sys
from os import environ
from pathlib import Path

import pytest

#: Application repository root, which is what a profile path is relative to.
_REPO_ROOT = Path(__file__).parents[1]

#: Environment variable ``conftest`` publishes the master's choice in.
_MARKER = "_STDAPI_TESTS_ENV_FILE"

#: A collection small enough that the subprocesses below cost a second each.
_A_SMALL_MODULE = "tests/test_healthcheck.py"

#: Seconds one subprocess pytest gets to collect that module.
_TIMEOUT = 300.0


def _reported_env_file(*args: str, env_extra: dict[str, str] | None = None) -> str:
    """Collect in a subprocess and return the profile its session header names.

    Args:
        args: Extra arguments the subprocess pytest is given.
        env_extra: Environment variables set on top of this process' own.

    Returns:
        The path the ``envfile:`` header reports, or an empty string if the run
        reported no header at all.
    """
    result = subprocess.run(  # noqa: S603
        # Not -q: the header the profile is reported in is what -q suppresses.
        [sys.executable, "-m", "pytest", _A_SMALL_MODULE, "--collect-only", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=_TIMEOUT,
        env={**environ, **(env_extra or {})},
    )
    for line in result.stdout.splitlines():
        if line.startswith("envfile:"):
            return line.partition(":")[2].strip()
    return ""


@pytest.fixture(scope="module")
def server_url_profile() -> Path:
    """The ``--server-url`` profile, without which none of this is observable."""
    profile = _REPO_ROOT / "tests" / ".env.server-url"
    if not profile.is_file():
        pytest.skip(f"{profile.name} is not present; no profile to select between")
    return profile


class TestProfileSelection:
    """The file a run loads is the one its own arguments name."""

    def test_the_server_url_profile_is_selected_from_the_command_line(
        self, server_url_profile: Path
    ) -> None:
        """A run given ``--server-url`` loads the profile that names that lane.

        Ref: tests/conftest.py
        """
        reported = _reported_env_file("--server-url", "http://127.0.0.1:9")
        assert reported == f"tests/{server_url_profile.name}"

    def test_a_nested_run_ignores_the_profile_its_parent_chose(
        self, server_url_profile: Path
    ) -> None:
        """A child with a command line of its own selects from it, not from the marker.

        The container suite is the real case: it runs inside a session that
        loaded ``tests/.env`` and starts a pytest of its own against the port the
        image is published on. Honouring the inherited marker there pointed every
        client at the outer run's gateway, which failed as fourteen setup errors.

        Ref: tests/test_container_image.py::TestServerBoot
        """
        reported = _reported_env_file(
            "--server-url", "http://127.0.0.1:9", env_extra={_MARKER: "tests/.env"}
        )
        assert reported == f"tests/{server_url_profile.name}"

    def test_a_process_with_no_command_line_reuses_the_marker(
        self, server_url_profile: Path
    ) -> None:
        """A worker's ``sys.argv`` is unusable, so only there is the marker read.

        Run with ``-c``, the interpreter reports exactly the ``sys.argv`` an
        execnet bootstrap gives a worker, which is what the selection tests for.
        Nothing else in the tree can distinguish the two cases.

        Ref: tests/conftest.py
        """
        code = (
            "import sys, os;"
            "sys.path.insert(0, 'tests');"
            "sys.argv = ['-c'];"
            f"os.environ['{_MARKER}'] = 'tests/{server_url_profile.name}';"
            "import conftest;"
            "print(conftest._loaded_env_file)"
        )
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=_TIMEOUT,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == f"tests/{server_url_profile.name}"
