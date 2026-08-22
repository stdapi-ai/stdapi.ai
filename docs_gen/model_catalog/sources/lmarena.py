"""LMArena arena ratings, published under CC BY 4.0.

One arena covers each modality the gateway serves except embeddings and
reranking: text and vision for chat, search for grounded answers, and the image
and video arenas for generation. Only the ``overall`` category of each arena is
published — the sub-categories multiply the columns without adding a distinct
answer to "is this model good".

The snapshot is read from the dataset's Parquet files rather than the paginated
rows API: one request per arena instead of a hundred and fifty, which is both
faster and the difference between a run that completes and one that is rate
limited half way through.
"""

from __future__ import annotations

import io
from typing import Any

import pyarrow.parquet as pq

from docs_gen.model_catalog.http import get_bytes
from docs_gen.model_catalog.sources import RawScore, SourceResult, snapshot

#: Parquet file holding one arena's latest published snapshot.
_PARQUET_URL: str = (
    "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset"
    "/resolve/main/{board}/latest-00000-of-00001.parquet"
)

#: Arena configurations published on the page, mapped to their display label.
BOARDS: dict[str, str] = {
    "text": "Text Arena",
    "vision": "Vision Arena",
    "search": "Search Arena",
    "text_to_image": "Text-to-Image Arena",
    "image_edit": "Image Edit Arena",
    "text_to_video": "Text-to-Video Arena",
    "image_to_video": "Image-to-Video Arena",
}

#: Category kept from each arena; the others slice the same votes by prompt type.
_CATEGORY: str = "overall"

#: Columns read from each arena; anything else is left behind.
_COLUMNS: tuple[str, ...] = (
    "model_name",
    "organization",
    "license",
    "rating",
    "rating_lower",
    "rating_upper",
    "vote_count",
    "rank",
    "category",
    "leaderboard_publish_date",
)


def _collect() -> dict[str, list[dict[str, Any]]]:
    """Fetch every published arena.

    Returns:
        Arena configuration name to its ``overall`` rows.
    """
    collected: dict[str, list[dict[str, Any]]] = {}
    for board in BOARDS:
        try:
            raw = get_bytes(_PARQUET_URL.format(board=board))
            table = pq.read_table(io.BytesIO(raw))
        except Exception:  # noqa: BLE001 -- one arena must not cost the others
            collected[board] = []
            continue
        columns = [name for name in _COLUMNS if name in table.column_names]
        rows = table.select(columns).to_pylist()
        collected[board] = [
            {key: _plain(value) for key, value in row.items()}
            for row in rows
            if row.get("category") == _CATEGORY
        ]
    return collected


def _plain(value: object) -> object:
    """Return a JSON-serialisable form of a Parquet value.

    Args:
        value: Value read from the table.

    Returns:
        The value, dates rendered as ISO strings.
    """
    return (
        value if value is None or isinstance(value, (str, int, float)) else str(value)
    )


def fetch(*, refresh: bool = False) -> SourceResult:
    """Read the arena ratings.

    Args:
        refresh: Ignore any cached snapshot.

    Returns:
        Every published arena rating, with the snapshot date the dataset carries.
    """
    raw = snapshot("lmarena", _collect, refresh=refresh, key="|".join(BOARDS))
    assert isinstance(raw, dict)  # noqa: S101 -- snapshot round-trips its own JSON
    scores: list[RawScore] = []
    notes: list[str] = []
    as_of = ""
    for board, label in BOARDS.items():
        rows: list[dict[str, Any]] = raw.get(board, [])
        if not rows:
            notes.append(f"{board}: no rows published")
            continue
        for row in rows:
            published = str(row.get("leaderboard_publish_date") or "")[:10]
            as_of = max(as_of, published)
            scores.append(
                RawScore(
                    source="lmarena",
                    board=board,
                    metric="elo",
                    label=label,
                    value=float(row["rating"]),
                    name=str(row["model_name"]),
                    organization=str(row.get("organization") or ""),
                    as_of=published,
                    rank=_optional_int(row.get("rank")),
                    ci_low=_optional_float(row.get("rating_lower")),
                    ci_high=_optional_float(row.get("rating_upper")),
                    samples=_optional_int(row.get("vote_count")),
                    licence=str(row.get("license") or ""),
                )
            )
    return SourceResult(key="lmarena", as_of=as_of, scores=scores, notes=notes)


def _optional_float(value: object) -> float | None:
    """Return *value* as a float, or ``None`` when it is absent.

    Args:
        value: Raw dataset value.

    Returns:
        The float, or ``None``.
    """
    return None if value is None else float(value)  # type: ignore[arg-type]


def _optional_int(value: object) -> int | None:
    """Return *value* as an integer, or ``None`` when it is absent.

    Args:
        value: Raw dataset value.

    Returns:
        The integer, or ``None``.
    """
    return None if value is None else int(float(value))  # type: ignore[arg-type]
