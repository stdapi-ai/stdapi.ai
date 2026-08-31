"""Word error rates from the Open ASR Leaderboard, published under Apache-2.0.

Only the scores are read. The audio corpora behind them carry their own,
sometimes non-commercial, licences and are never fetched or redistributed here.

The results dataset the leaderboard used to publish went private in August 2026
and now answers ``401`` to anyone outside its owning organisation, so the rates
are read from the table the leaderboard application itself serves.
"""

from __future__ import annotations

import html
import re
from typing import Any

from docs_gen.model_catalog.http import FetchError, get_json
from docs_gen.model_catalog.sources import RawScore, SourceResult, snapshot

#: Layout of the running leaderboard application, values included.
_CONFIG_URL: str = "https://hf-audio-open-asr-leaderboard.hf.space/config"

#: Identifier the application gives its English short-form results table.
_TABLE_ELEMENT_ID: str = "leaderboard-table"

#: Table column holding the model name, as a link the application renders.
_NAME_HEADER: str = "model"

#: Table column holding the average word error rate over the English short set.
_SCORE_HEADER: str = "Average WER ⬇️"

#: Column holding the average word error rate, in the collected snapshot.
_SCORE_COLUMN: str = "avg"

#: Matches the markup the application wraps a model name in.
_TAG = re.compile(r"<[^>]*>")


def _table(config: Any) -> dict[str, Any]:  # noqa: ANN401 -- an upstream JSON document
    """Return the English short-form results table from the application layout.

    Args:
        config: The decoded application configuration.

    Returns:
        The table, as headers and rows.

    Raises:
        FetchError: The application serves no such table.
    """
    components = config.get("components") if isinstance(config, dict) else None
    for component in components or ():
        props = component.get("props") or {}
        value = props.get("value")
        if props.get("elem_id") == _TABLE_ELEMENT_ID and isinstance(value, dict):
            return value
    msg = f"no {_TABLE_ELEMENT_ID!r} table at {_CONFIG_URL}"
    raise FetchError(msg)


def _collect() -> list[dict[str, str]]:
    """Download the published results table.

    Returns:
        The decoded rows, keyed as the retired results file keyed them.

    Raises:
        FetchError: The table does not carry the columns the rates are read from.
    """
    table = _table(get_json(_CONFIG_URL))
    headers = [str(header) for header in table.get("headers") or ()]
    missing = {_NAME_HEADER, _SCORE_HEADER} - set(headers)
    if missing:
        msg = f"{_CONFIG_URL} publishes no {sorted(missing)} column"
        raise FetchError(msg)
    name_at, score_at = headers.index(_NAME_HEADER), headers.index(_SCORE_HEADER)
    rows = []
    for row in table.get("data") or ():
        if not isinstance(row, list) or len(row) <= max(name_at, score_at):
            continue
        # The name cell is a link the application renders; only its text is data.
        name = html.unescape(_TAG.sub("", str(row[name_at] or ""))).strip()
        rows.append({"model": name, _SCORE_COLUMN: str(row[score_at] or "").strip()})
    return rows


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
