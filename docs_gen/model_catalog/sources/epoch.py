"""Epoch AI's independently-run benchmark results, published under CC BY 4.0.

Unlike an arena rating these are objective, reproducible task scores, so they
answer a different question and are shown alongside rather than blended in.
Only benchmarks with a stable public definition are published.
"""

from __future__ import annotations

import csv
import io
import zipfile
from typing import Any, NamedTuple

from docs_gen.model_catalog.http import get_bytes
from docs_gen.model_catalog.sources import RawScore, SourceResult, snapshot

#: Archive of every benchmark table Epoch AI publishes.
_ARCHIVE_URL: str = "https://epoch.ai/data/benchmark_data.zip"


class _Benchmark(NamedTuple):
    """One benchmark table inside the archive.

    Attributes:
        member: File name within the archive.
        label: Short display label for the table column.
        column: CSV column holding the score.
        scale: Multiplier turning the raw score into a percentage.
    """

    member: str
    label: str
    column: str
    scale: float


#: Benchmark tables published on the page, in display order.
BENCHMARKS: tuple[_Benchmark, ...] = (
    _Benchmark("gpqa_diamond.csv", "GPQA Diamond", "mean_score", 100.0),
    _Benchmark("swe_bench_verified.csv", "SWE-bench Verified", "mean_score", 100.0),
    _Benchmark("frontiermath.csv", "FrontierMath", "mean_score", 100.0),
    _Benchmark("math_level_5.csv", "MATH Level 5", "mean_score", 100.0),
    _Benchmark("simpleqa_verified.csv", "SimpleQA Verified", "mean_score", 100.0),
    _Benchmark("aider_polyglot_external.csv", "Aider Polyglot", "Percent correct", 1.0),
)


def _collect() -> dict[str, list[dict[str, str]]]:
    """Download the archive and decode the published benchmark tables.

    Returns:
        Archive member name to its rows.
    """
    archive = zipfile.ZipFile(io.BytesIO(get_bytes(_ARCHIVE_URL)))
    tables: dict[str, list[dict[str, str]]] = {}
    for benchmark in BENCHMARKS:
        if benchmark.member not in archive.namelist():
            continue
        text = archive.read(benchmark.member).decode("utf-8", "replace")
        tables[benchmark.member] = [
            dict(row) for row in csv.DictReader(io.StringIO(text))
        ]
    return tables


def _row_date(row: dict[str, Any]) -> str:
    """Return the date one benchmark row was evaluated.

    Args:
        row: A benchmark table row.

    Returns:
        An ISO date, or an empty string when the row carries none.
    """
    for key in ("Started at", "Date of evaluation"):
        value = str(row.get(key) or "")[:10]
        if value:
            return value
    return ""


def _score(row: dict[str, Any], column: str, scale: float) -> float | None:
    """Read one row's score.

    Args:
        row: A benchmark table row.
        column: Column holding the score.
        scale: Multiplier turning it into a percentage.

    Returns:
        The score, or ``None`` when the cell is blank or not a number.
    """
    raw = str(row.get(column) or "").strip().rstrip("%")
    try:
        return float(raw) * scale
    except ValueError:
        return None


def _best_runs(
    rows: list[dict[str, Any]], benchmark: _Benchmark
) -> dict[str, dict[str, Any]]:
    """Keep one run per model.

    Epoch publishes several runs of a benchmark per model — different harnesses,
    different dates. Taking whichever came first in the CSV means the published
    number changes when the file is re-sorted, so the most recent run wins and
    ties break on the row's own identifier.

    Args:
        rows: One benchmark table's rows.
        benchmark: The table being read.

    Returns:
        Model name to its chosen row.
    """
    chosen: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("Model version") or "").strip()
        if not name or _score(row, benchmark.column, benchmark.scale) is None:
            continue
        current = chosen.get(name)
        if current is None or (_row_date(row), str(row.get("id", ""))) > (
            _row_date(current),
            str(current.get("id", "")),
        ):
            chosen[name] = row
    return chosen


def fetch(*, refresh: bool = False) -> SourceResult:
    """Read the published benchmark results.

    Args:
        refresh: Ignore any cached snapshot.

    Returns:
        One score per model and benchmark, dated from the row it came from.
    """
    raw = snapshot(
        "epoch",
        _collect,
        refresh=refresh,
        key="|".join(item.member for item in BENCHMARKS),
    )
    assert isinstance(raw, dict)  # noqa: S101 -- snapshot round-trips its own JSON
    scores: list[RawScore] = []
    notes: list[str] = []
    as_of = ""
    for benchmark in BENCHMARKS:
        rows: list[dict[str, Any]] = raw.get(benchmark.member, [])
        if not rows:
            notes.append(f"{benchmark.member}: missing from the archive")
            continue
        if benchmark.column not in rows[0]:
            notes.append(
                f"{benchmark.member}: no {benchmark.column!r} column — "
                f"upstream renamed it, so this benchmark is not published"
            )
            continue
        for name, row in sorted(_best_runs(rows, benchmark).items()):
            value = _score(row, benchmark.column, benchmark.scale)
            if value is None:
                continue
            published = _row_date(row)
            as_of = max(as_of, published)
            scores.append(
                RawScore(
                    source="epoch",
                    board=benchmark.member.removesuffix(".csv"),
                    metric="accuracy",
                    label=benchmark.label,
                    value=value,
                    name=name,
                    organization=str(row.get("Organization") or ""),
                    as_of=published,
                    unit="%",
                )
            )
    return SourceResult(key="epoch", as_of=as_of, scores=scores, notes=notes)
