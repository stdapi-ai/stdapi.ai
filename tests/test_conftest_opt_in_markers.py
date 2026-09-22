"""The opt-in lanes, each skipped unless its own command line flag is passed.

A lane is wired in three places -- the flag, the registered marker and the skip
loop that joins them -- and losing any one of them either breaks collection or
silently runs a billed lane on every invocation, so each is checked here against
the real option parser of this session.

Ref: https://docs.pytest.org/en/stable/example/simple.html#control-skipping-of-tests-according-to-command-line-option
     tests/conftest.py:pytest_collection_modifyitems
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._helpers import FakeCollectedItem
from tests.conftest import (
    _OPT_IN_MARKERS,
    pytest_addoption,
    pytest_collection_modifyitems,
)


class _FakeConfig:
    """A config resolving option names like the real one, with chosen lanes on."""

    def __init__(self, real: pytest.Config, enabled: frozenset[str]) -> None:
        self._real = real
        self._enabled = enabled

    def getoption(self, name: str, default: object = None) -> object:
        """Return True for an enabled lane flag and False for any other option.

        Args:
            name: Option name, as passed to ``pytest.Config.getoption``.
            default: Accepted for signature parity and ignored.

        Returns:
            Whether *name* is one of the enabled lane flags.

        Raises:
            ValueError: If *name* is not an option this session registered.
        """
        self._real.getoption(name)
        return name in self._enabled


class _RecordingParser:
    """A parser recording the keyword arguments of every option registered on it."""

    def __init__(self) -> None:
        self.options: dict[str, dict[str, Any]] = {}

    def addoption(self, *names: str, **kwargs: Any) -> None:  # noqa: ANN401
        """Record *kwargs* under each of the option's *names*.

        Args:
            *names: Flag spellings of the option, such as ``--agentic``.
            **kwargs: Keyword arguments the option was registered with.
        """
        for name in names:
            self.options[name] = kwargs


def _skip_reasons(config: pytest.Config, lane: str, *flags: str) -> list[str]:
    """Return the skip reasons the hook gives an item carrying the *lane* marker.

    Args:
        config: This session's config, whose parser validates every option name.
        lane: Opt-in marker to put on the item.
        flags: Command line flags to treat as passed.

    Returns:
        The reason of every ``skip`` mark the hook added.
    """
    item = FakeCollectedItem(getattr(pytest.mark, lane))
    pytest_collection_modifyitems(
        _FakeConfig(config, frozenset(flags)),  # type: ignore[arg-type]
        [item],  # type: ignore[list-item]
    )
    return [
        str(mark.mark.kwargs["reason"])
        for mark in item.added
        if mark.mark.name == "skip"
    ]


class TestImageGenerationLane:
    """``image_generation`` runs only under ``--image-generation``.

    Ref: tests/conftest.py:pytest_collection_modifyitems
    """

    def test_the_flag_is_registered(self, pytestconfig: pytest.Config) -> None:
        """The hyphenated flag resolves to a boolean option in this session.

        Ref: tests/conftest.py:pytest_addoption
        """
        assert isinstance(pytestconfig.getoption("--image-generation"), bool)

    def test_the_flag_defaults_to_off(self) -> None:
        """A run that does not pass the flag leaves the billed lane off.

        Ref: tests/conftest.py:pytest_addoption
        """
        parser = _RecordingParser()
        pytest_addoption(parser)  # type: ignore[arg-type]
        option = parser.options["--image-generation"]
        assert option["action"] == "store_true"
        assert option["default"] is False

    def test_the_marker_is_registered(self, pytestconfig: pytest.Config) -> None:
        """``--strict-markers`` would reject the marker if it were not declared.

        Ref: pyproject.toml ([tool.pytest.ini_options] markers)
        """
        declared = {line.split(":")[0] for line in pytestconfig.getini("markers")}
        assert "image_generation" in declared

    def test_is_skipped_without_the_flag(self, pytestconfig: pytest.Config) -> None:
        """The skip names the flag to pass, hyphenated as it is typed.

        Ref: tests/conftest.py:pytest_collection_modifyitems
        """
        assert _skip_reasons(pytestconfig, "image_generation") == [
            "Need --image-generation option to run this test"
        ]

    def test_runs_with_the_flag(self, pytestconfig: pytest.Config) -> None:
        """Passing the flag lifts the skip.

        Ref: tests/conftest.py:pytest_collection_modifyitems
        """
        reasons = _skip_reasons(pytestconfig, "image_generation", "--image-generation")
        assert reasons == []

    def test_expensive_does_not_open_it(self, pytestconfig: pytest.Config) -> None:
        """The lane is split out of ``--expensive``, not a part of it.

        Ref: tests/conftest.py:pytest_collection_modifyitems
        """
        reasons = _skip_reasons(pytestconfig, "image_generation", "--expensive")
        assert reasons == ["Need --image-generation option to run this test"]


@pytest.mark.parametrize("lane", _OPT_IN_MARKERS)
def test_every_lane_flag_resolves(pytestconfig: pytest.Config, lane: str) -> None:
    """Each opt-in marker maps to a registered flag, so collection cannot crash.

    Ref: tests/conftest.py:pytest_addoption
    """
    assert _skip_reasons(pytestconfig, lane) == [
        f"Need --{lane.replace('_', '-')} option to run this test"
    ]
