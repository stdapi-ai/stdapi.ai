"""Independent leaderboards the page is allowed to republish.

Each module here turns one upstream leaderboard into :class:`RawScore` rows. A
source only belongs here when its licence permits redistribution — the licence
of record lives in :data:`~docs_gen.model_catalog.config.SOURCES`.
"""

from __future__ import annotations

import json
import time
from contextlib import suppress
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING

from docs_gen.model_catalog.config import SNAPSHOT_DIR

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

#: How long a cached upstream snapshot stays usable, in seconds.
SNAPSHOT_MAX_AGE: float = 24 * 3600.0


@dataclass(frozen=True, slots=True)
class RawScore:
    """One leaderboard row, before it is matched to a model.

    Attributes:
        source: Key of the source that published it.
        board: Sub-leaderboard the row belongs to.
        metric: Metric name as the source reports it.
        label: Short display label for the table column.
        value: The score itself.
        name: Model name exactly as the source spells it.
        organization: Publisher the source attributes the model to.
        as_of: Date of the snapshot the row came from.
        unit: Display unit, empty when unitless.
        higher_is_better: Whether a larger value is a better result.
        rank: Rank within the sub-leaderboard, when published.
        ci_low: Lower bound of the published confidence interval, when any.
        ci_high: Upper bound of the published confidence interval, when any.
        samples: Vote or sample count behind the score, when published.
        licence: How the source classifies the model's weights licence, when it
            says: an open-weight Apache-2.0 model and a research-only one are
            not the same proposition.
    """

    source: str
    board: str
    metric: str
    label: str
    value: float
    name: str
    organization: str
    as_of: str
    unit: str = ""
    higher_is_better: bool = True
    rank: int | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    samples: int | None = None
    licence: str = ""


@dataclass(frozen=True, slots=True)
class RawReference:
    """A third-party evaluation entry that publishes no comparable score.

    Attributes:
        source: Key of the source that published it.
        name: Model name exactly as the source spells it.
        organization: Publisher the source attributes the model to.
        label: Short display label for the link.
        detail: What the reader will find there.
        url: Page the link points at.
    """

    source: str
    name: str
    organization: str
    label: str
    detail: str
    url: str


@dataclass(frozen=True, slots=True)
class SourceResult:
    """Everything one source contributed to a run.

    Attributes:
        key: Source key.
        as_of: Date of the snapshot used.
        scores: Comparable rows read from the source.
        references: Entries the source publishes without a comparable score.
        notes: Anything degraded during collection, reported by the CLI.
    """

    key: str
    as_of: str
    scores: list[RawScore]
    references: list[RawReference] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def snapshot(
    name: str, produce: Callable[[], object], *, refresh: bool = False, key: str = ""
) -> object:
    """Return a cached upstream snapshot, fetching it only when stale.

    Args:
        name: Snapshot file stem.
        produce: Callable that fetches the snapshot from upstream.
        refresh: Ignore any cached copy and fetch again.
        key: What was asked for. A snapshot taken of one set of boards must not
            be served for another, or adding a board yields no rows for a day.

    Returns:
        The decoded snapshot.

    Raises:
        Exception: Upstream could not be read and no cached copy exists.
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    # mkdir's mode only applies at creation; a directory left over from before
    # this restriction existed would otherwise keep its looser permissions.
    SNAPSHOT_DIR.chmod(0o700)
    digest = sha256(key.encode()).hexdigest()[:8] if key else "all"
    path: Path = SNAPSHOT_DIR / f"{name}.{digest}.json"
    fresh_enough = (
        path.is_file() and time.time() - path.stat().st_mtime < SNAPSHOT_MAX_AGE
    )
    if not refresh and fresh_enough:
        with suppress(ValueError, OSError):
            return json.loads(path.read_text())
    try:
        data = produce()
    except Exception:
        # A yesterday's snapshot is better information than none: fall back to
        # it rather than let one unreachable source blank its column.
        if not path.is_file():
            raise
        with suppress(ValueError, OSError):
            return json.loads(path.read_text())
        raise
    # Written whole then moved: a run interrupted mid-write would otherwise
    # leave a truncated file that the next day's run reads as the truth.
    scratch = path.with_suffix(".partial")
    scratch.write_text(json.dumps(data))
    scratch.replace(path)
    return data
