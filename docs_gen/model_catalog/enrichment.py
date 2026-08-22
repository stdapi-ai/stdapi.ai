"""Facts filled in by hand from a vendor's own documentation.

Some fields have no machine-readable source at all. AWS does not return a
context window to an ordinary caller, and the open databases only cover the
models someone bothered to add. Because a regeneration now *updates* the
published data set rather than replacing it, a fact only has to be established
once — so the ones that matter are looked up at the source and recorded here.

Every entry carries where it came from and when it was checked. The page shows
the value; the citation lives with the generator, in
``state/provenance.json``, so a reader of this repository can audit any number
on the page back to the vendor page it came from.

An entry only ever fills a field the automatic sources left empty: the moment
AWS or an open database starts publishing one, that wins.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import ValidationError

from docs_gen.model_catalog.config import ENRICHMENT_PATH, PROVENANCE_PATH
from docs_gen.model_catalog.tokens import format_tokens, parse_tokens

if TYPE_CHECKING:
    from collections.abc import Iterable

    from docs_gen.model_catalog.schema import ModelRow

#: Fields written as a date, where a month and a full day can agree.
_DATE_FIELDS: frozenset[str] = frozenset({"knowledge_cutoff"})

#: Fields quoted in tokens, where a rounded value and an exact one can agree.
_TOKEN_FIELDS: frozenset[str] = frozenset({"context_window", "max_output_tokens"})

#: Fields an entry is allowed to set; anything else is a mistake, not a fact.
ALLOWED_FIELDS: frozenset[str] = frozenset(
    {
        "context_window",
        "max_output_tokens",
        "knowledge_cutoff",
        "reasoning",
        "tool_call",
        "open_weights",
        "family",
        "parameters",
        "active_parameters",
    }
)


class Applied(NamedTuple):
    """What the overlay contributed to one run.

    Attributes:
        filled: Field values written, by field name.
        refined: Values replaced because the curated one is the same figure at
            better precision, by field name.
        skipped: Values ignored because an automatic source already had one.
        unknown: Entries naming a model or field the catalogue does not have.
        disputed: Fields where a curated, cited value disagrees with what an
            automatic source published, and the automatic source won.
    """

    filled: dict[str, int]
    refined: dict[str, int]
    skipped: int
    unknown: list[str]
    disputed: list[str]


def load() -> dict[str, dict[str, Any]]:
    """Read the curated overlay.

    Returns:
        Model ID to field name to ``{value, source, source_name, checked}``.
    """
    if not ENRICHMENT_PATH.is_file():
        return {}
    loaded = json.loads(ENRICHMENT_PATH.read_text())
    return {key: value for key, value in loaded.items() if not key.startswith("_")}


def apply(rows: Iterable[ModelRow], overlay: dict[str, dict[str, Any]]) -> Applied:
    """Fill empty fields from the overlay.

    Args:
        rows: Rows being assembled.
        overlay: The curated overlay.

    Returns:
        A record of what was written, for the CLI to report.
    """
    filled: dict[str, int] = {}
    refined: dict[str, int] = {}
    skipped = 0
    unknown: list[str] = []
    disputed: list[str] = []
    by_id = {row.id: row for row in rows}

    for model_id, fields in sorted(overlay.items()):
        row = by_id.get(model_id)
        if row is None:
            unknown.append(model_id)
            continue
        for name, entry in sorted(fields.items()):
            if name not in ALLOWED_FIELDS or not isinstance(entry, dict):
                unknown.append(f"{model_id}.{name}")
                continue
            value = _normalised(name, entry.get("value"))
            if value is None:
                unknown.append(f"{model_id}.{name}")
                continue
            collected = getattr(row, name, None)
            already = collected not in (None, "", [])
            if already and not _same_figure(name, collected, value):
                if collected != value:
                    disputed.append(f"{model_id}.{name} {collected} vs {value}")
                skipped += 1
                continue
            if already:
                skipped += 1
            try:
                setattr(row, name, value)
            except ValidationError:
                # Shaped like a value but not one this field accepts, e.g. a
                # string in a boolean field: reject the entry, do not publish it.
                unknown.append(f"{model_id}.{name}")
                continue
            counter = refined if already else filled
            counter[name] = counter.get(name, 0) + 1

    return Applied(
        filled=filled,
        refined=refined,
        skipped=skipped,
        unknown=unknown,
        disputed=disputed,
    )


def _same_figure(name: str, collected: object, curated: object) -> bool:
    """Report whether two values state one fact at different precision.

    The model cards round both kinds of figure they carry. A card saying ``4K``
    and a vendor page saying 4096 are one fact, and publishing 4,000 as if it
    were exact claims a precision the card never had; a cutoff of ``2024-03``
    and one of ``2024-03-05`` are likewise one fact. Where the coarser value is
    the finer one rounded, the curated value is the better of the two.

    Args:
        name: Field being set.
        collected: What an automatic source published.
        curated: What the overlay states.

    Returns:
        True when the two agree and the curated value is the more precise.
    """
    if collected == curated:
        return False
    if name in _TOKEN_FIELDS:
        return format_tokens(collected) == format_tokens(curated)
    if name in _DATE_FIELDS:
        return (
            isinstance(collected, str)
            and isinstance(curated, str)
            and (curated.startswith(collected))
        )
    return False


def record_provenance(
    rows: Iterable[ModelRow], overlay: dict[str, dict[str, Any]]
) -> int:
    """Write the citation for every published value this overlay is the source of.

    Run against the finished catalogue rather than against one run's rows: a
    value established here is carried forward by later runs, and a model AWS
    has retired keeps both its value and its citation. The test is what the
    catalogue actually publishes — a citation is only written where the
    published value is still the one this entry states.

    Args:
        rows: The published rows, after the merge.
        overlay: The curated overlay.

    Returns:
        How many citations were written.
    """
    provenance: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        for name, entry in sorted((overlay.get(row.id) or {}).items()):
            if name not in ALLOWED_FIELDS or not isinstance(entry, dict):
                continue
            value = _normalised(name, entry.get("value"))
            if value is None or getattr(row, name, None) != value:
                continue
            provenance.setdefault(row.id, {})[name] = {
                "value": str(value),
                "source": str(entry.get("source", "")),
                "source_name": str(entry.get("source_name", "")),
                "checked": str(entry.get("checked", "")),
            }
    _write_provenance(provenance)
    return sum(len(fields) for fields in provenance.values())


def _normalised(name: str, value: object) -> object | None:
    """Put a hand-written value into the shape every other source uses.

    A curator reads ``1,047,576`` off a vendor page and writes it however that
    page wrote it; the column has to read the same for all 142 models. This
    only reshapes the value — whether it actually fits the field is for
    :meth:`~pydantic.BaseModel.__setattr__` to decide, once ``validate_assignment``
    is on.

    Args:
        name: Field being set.
        value: The value as the overlay spells it.

    Returns:
        The value to store, or ``None`` when it cannot be read as one.
    """
    if name == "context_window":
        return format_tokens(value)
    if name == "max_output_tokens":
        return parse_tokens(value)
    return value if value not in (None, "") else None


def _write_provenance(provenance: dict[str, dict[str, dict[str, str]]]) -> None:
    """Record which vendor page each hand-filled value came from.

    Args:
        provenance: Model ID to field name to its citation.
    """
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "_readme": [
            "Vendor pages backing the hand-checked values on the Models page.",
            "Written by the generator from state/enrichment.json; do not edit.",
            "One entry per published value an overlay entry states, whether the",
            "overlay filled it or an automatic source arrived at the same value.",
            "Values no overlay entry covers are sourced by the page's own table.",
        ],
        "models": provenance,
    }
    PROVENANCE_PATH.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
