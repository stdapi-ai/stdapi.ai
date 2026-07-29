"""Tests keeping the package metadata version in sync with the running server."""

import tomllib
from pathlib import Path

from stdapi.server import SERVER_VERSION


def test_pyproject_version_matches_server_version() -> None:
    """pyproject.toml's version must match stdapi.server.SERVER_VERSION.

    Guards against the package metadata (pip/PyPI, sdist/wheel filenames)
    silently drifting behind the version reported by the running server.
    """
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    assert data["project"]["version"] == SERVER_VERSION
