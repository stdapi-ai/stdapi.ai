"""Word error rates from the Open ASR Leaderboard, published under Apache-2.0.

Only the scores are read. The audio corpora behind them carry their own,
sometimes non-commercial, licences and are never fetched or redistributed here.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from docs_gen.model_catalog.http import get_bytes
from docs_gen.model_catalog.sources import RawScore, SourceResult, snapshot

#: Results file the leaderboard application itself reads.
_RESULTS_URL: str = (
    "https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results"
    "/resolve/main/english_short_latest.csv"
)

#: Column holding the average word error rate across the English short-form set.
_SCORE_COLUMN: str = "avg"


def _collect() -> list[dict[str, str]]:
    """Download the published results table.

    Returns:
        The decoded rows.
    """
    text = get_bytes(_RESULTS_URL).decode("utf-8", "replace")
    return [dict(row) for row in csv.DictReader(io.StringIO(text))]


def fetch(*, refresh: bool = False) -> SourceResult:
    """Read the published word error rates.

    Args:
        refresh: Ignore any cached snapshot.

    Returns:
        One score per model, lower being better.
    """
    raw = snapshot("open_asr", _collect, refresh=refresh)
    assert isinstance(raw, list)  # noqa: S101 -- snapshot round-trips its own JSON
    scores: list[RawScore] = []
    notes: list[str] = []
    if raw and _SCORE_COLUMN not in raw[0]:
        notes.append(f"no {_SCORE_COLUMN!r} column — upstream renamed it")
        raw = []
    for row in raw:
        record: dict[str, Any] = row
        name = str(record.get("model") or "").strip()
        value = str(record.get(_SCORE_COLUMN) or "").strip()
        if not name or not value:
            continue
        try:
            rate = float(value)
        except ValueError:
            notes.append(f"{name}: unreadable word error rate {value!r}")
            continue
        scores.append(
            RawScore(
                source="open_asr",
                board="english_short",
                metric="wer",
                label="Open ASR (WER)",
                value=rate,
                name=name,
                organization=name.split("/", 1)[0] if "/" in name else "",
                as_of="",
                unit="%",
                higher_is_better=False,
            )
        )
    return SourceResult(key="open_asr", as_of="", scores=scores, notes=notes)
