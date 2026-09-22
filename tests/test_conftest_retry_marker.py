"""The ``retry`` marker, which re-runs a test a stated non-determinism can fail.

A retry is the one mechanism in this suite that can hide a real regression, so the
translation into the rerun plugin's own marker refuses any use that does not say
why the test is allowed to fail.

Ref: https://pytest-rerunfailures.readthedocs.io/en/latest/#re-run-individual-failing-tests
     tests/conftest.py:pytest_collection_modifyitems
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests._helpers import FakeCollectedItem
from tests.conftest import (
    _DEFAULT_RERUN_DELAY,
    _DEFAULT_RERUNS,
    pytest_collection_modifyitems,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class _FakeConfig:
    """A config whose every option is off, so only the retry branch acts."""

    @staticmethod
    def getoption(name: str, default: object = None) -> object:  # noqa: ARG004
        """Report every command line option as unset."""
        return False


def _run(marker: pytest.MarkDecorator | None) -> FakeCollectedItem:
    """Collect a single item carrying *marker* and return it afterwards.

    Args:
        marker: Marker to put on the item, or None for a bare item.

    Returns:
        The item, with whatever markers the hook added recorded on it.
    """
    item = FakeCollectedItem() if marker is None else FakeCollectedItem(marker)
    pytest_collection_modifyitems(
        _FakeConfig(),  # type: ignore[arg-type]
        [item],  # type: ignore[list-item]
    )
    return item


def _flaky_marks(item: FakeCollectedItem) -> Iterator[Any]:
    """Yield every ``flaky`` mark the hook put on *item*."""
    return (mark.mark for mark in item.added if mark.mark.name == "flaky")


class TestRetryMarker:
    """``retry`` becomes the rerun plugin's ``flaky``, and only with a reason."""

    def test_reason_only_gets_the_default_attempts(self) -> None:
        """A reason is enough: the count and delay fall back to the defaults."""
        item = _run(pytest.mark.retry("a cold cache reports no cached tokens"))
        (mark,) = _flaky_marks(item)
        assert mark.kwargs == {
            "reruns": _DEFAULT_RERUNS,
            "reruns_delay": _DEFAULT_RERUN_DELAY,
        }

    def test_explicit_counts_win_over_the_defaults(self) -> None:
        """``reruns`` and ``delay`` are forwarded under the plugin's own names."""
        item = _run(pytest.mark.retry("a stated reason", reruns=5, delay=0.5))
        (mark,) = _flaky_marks(item)
        assert mark.kwargs == {"reruns": 5, "reruns_delay": 0.5}

    @pytest.mark.parametrize("args", [(), ("",), ("   ",)])
    def test_a_missing_reason_fails_collection(self, args: tuple[str, ...]) -> None:
        """Retrying without saying why is a collection error, not a silent retry."""
        with pytest.raises(pytest.UsageError, match="must state a reason"):
            _run(pytest.mark.retry(*args))

    def test_an_unmarked_test_is_never_retried(self) -> None:
        """Nothing is added to an item that does not ask to be retried."""
        assert not list(_flaky_marks(_run(None)))
