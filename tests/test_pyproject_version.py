"""Package metadata version stays in sync with the version the server reports.

Ref: stdapi/server.py:SERVER_VERSION
"""

import tomllib
from pathlib import Path

from stdapi.server import SERVER_VERSION


def test_pyproject_version_matches_server_version() -> None:
    """pyproject.toml's version equals stdapi.server.SERVER_VERSION.

    Guards against the package metadata (pip/PyPI, sdist/wheel filenames)
    silently drifting behind the version reported by the running server, which is
    also the version echoed in the ``server_version`` log field and the user agent.

    Ref: stdapi/server.py:USER_AGENT
    """
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    assert data["project"]["version"] == SERVER_VERSION
