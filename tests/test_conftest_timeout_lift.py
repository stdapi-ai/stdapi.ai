"""Lifting the global per-test timeout from the lanes that time themselves.

``timeout = 900`` in ``pyproject.toml`` exists to end a deadlock, but the container
and agentic lanes drive builds and third-party clients on subprocess budgets that
legitimately run past it and report their own failure when one expires. Killing
them at 900 s would turn a clean, named failure into a dead xdist worker.

Ref: https://pytest-timeout.readthedocs.io/en/latest/#timeout
     tests/conftest.py:pytest_collection_modifyitems
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.conftest import _SELF_TIMED_MARKERS, pytest_collection_modifyitems

if TYPE_CHECKING:
    from collections.abc import Iterator


class _FakeConfig:
    """A config whose every option is off, so no target-specific branch acts."""

    @staticmethod
    def getoption(name: str, default: object = None) -> object:  # noqa: ARG004
        """Report every command line option as unset."""
        return False


class _FakeItem:
    """A collected item exposing just the marker API the hook uses."""

    def __init__(self, *markers: pytest.MarkDecorator) -> None:
        self.nodeid = "tests/test_x.py::test_y"
        self._marks = [marker.mark for marker in markers]
        self.fixturenames: tuple[str, ...] = ()
        self.added: list[Any] = []

    def get_closest_marker(self, name: str) -> Any | None:  # noqa: ANN401
        """Return the first of the item's own markers that *name* matches."""
        return next((mark for mark in self._marks if mark.name == name), None)

    def add_marker(self, marker: Any) -> None:  # noqa: ANN401
        """Record a marker the hook applied."""
        self.added.append(marker)


def _timeouts(*markers: pytest.MarkDecorator) -> Iterator[Any]:
    """Yield every ``timeout`` mark the hook puts on an item carrying *markers*."""
    item = _FakeItem(*markers)
    pytest_collection_modifyitems(
        _FakeConfig(),  # type: ignore[arg-type]
        [item],  # type: ignore[list-item]
    )
    return (mark.mark for mark in item.added if mark.mark.name == "timeout")


class TestSelfTimedLanes:
    """Only the self-timed lanes lose the global timeout, and never an explicit one."""

    @pytest.mark.parametrize("lane", _SELF_TIMED_MARKERS)
    def test_a_self_timed_lane_loses_the_global_timeout(self, lane: str) -> None:
        """Zero disables pytest-timeout, leaving the lane's own budgets in charge."""
        (mark,) = _timeouts(getattr(pytest.mark, lane))
        assert mark.args == (0,)

    @pytest.mark.parametrize("lane", _SELF_TIMED_MARKERS)
    def test_an_explicit_timeout_survives_the_lift(self, lane: str) -> None:
        """A test that states its own budget keeps it rather than losing the bound."""
        assert not list(_timeouts(getattr(pytest.mark, lane), pytest.mark.timeout(30)))

    def test_an_ordinary_test_keeps_the_global_timeout(self) -> None:
        """Nothing is added outside the self-timed lanes, so the default applies."""
        assert not list(_timeouts(pytest.mark.expensive))
