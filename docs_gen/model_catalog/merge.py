"""Folds a fresh collection into the data set already published.

A regeneration is an update, not a replacement, for two reasons.

The richest fields on this page come from parts of ``ListFoundationModels`` AWS
does not document — context window, inference APIs, media types, the capability
flags. Undocumented fields are beta by nature: they can stop being returned
without notice, to everyone or to one caller. Replacing the data set wholesale
would blank those columns the first time that happens, silently and everywhere.
Merging keeps the last known value instead.

AWS also stops listing a model when it retires it. Replacing would delete it
from the catalogue; merging keeps it, marked, so a reader can still look up a
model they have running and find its dates.
"""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from docs_gen.model_catalog.schema import Catalog, Manifest, ModelDetail, ModelRow

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

#: Fields carried forward from the previous data set when a run returns nothing.
#:
#: The undocumented ``ListFoundationModels`` fields, the vendor prose that
#: travels with them, and the fields the AWS model cards state that no API
#: returns. Everything else — availability, prices, scores — has an
#: authoritative source every run, so a fresh empty value is the truth.
CARRY_FORWARD: tuple[str, ...] = (
    "family",
    "knowledge_cutoff",
    "reasoning",
    "tool_call",
    "open_weights",
    "licence",
    "parameters",
    "active_parameters",
    "apis",
    "max_output_tokens",
    "context_window",
    "prompt_caching",
    "guardrails",
    "latency_optimized",
    "provisioned",
    "count_tokens",
    "prompt_routing",
    "batch_in_region",
    "batch_cross_region",
    "image_types",
    "document_types",
    "video_types",
    "start_of_life",
    "end_of_life",
)

#: Long-form detail fields carried forward on the same reasoning.
CARRY_FORWARD_DETAIL: tuple[str, ...] = (
    "description",
    "summary",
    "attributes",
    "languages",
    "use_cases",
    "context_window",
    "policy_url",
)


@dataclass(slots=True)
class MergeReport:
    """What folding the new collection into the old one changed.

    Attributes:
        added: Models this run saw for the first time.
        retired: Models the previous data set had that this run did not see.
        returned: Models that had been retired and are listed again.
        carried: Field values kept from the previous data set, by field name.
        previous_live: Models the previous data set counted as live, before
            this run's retirements were applied — the merge's own denominator.
    """

    added: list[str] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)
    returned: list[str] = field(default_factory=list)
    carried: dict[str, int] = field(default_factory=dict)
    previous_live: int = 0


def _empty(value: object) -> bool:
    """Report whether a collected value carries no information.

    Args:
        value: A field value from the fresh collection.

    Returns:
        ``True`` when the field was not returned this run.
    """
    return value is None or value in ("", [])


def merge_models(
    previous: Iterable[ModelRow], current: Iterable[ModelRow], *, generated: str
) -> tuple[list[ModelRow], MergeReport]:
    """Fold a fresh set of rows into the published one.

    Args:
        previous: Rows from the published data set.
        current: Rows this run collected.
        generated: Date stamp of this run.

    Returns:
        The merged rows, sorted by model ID, and a report of what changed.
    """
    report = MergeReport()
    old = {row.id: row for row in previous}
    # Taken before any row below is mutated: merging retires rows in place, so
    # counting live rows afterwards would count survivors, not the prior total.
    report.previous_live = sum(1 for row in old.values() if not row.retired)
    merged: dict[str, ModelRow] = {}

    for row in current:
        before = old.get(row.id)
        if before is None:
            row.first_seen = generated
            report.added.append(row.id)
        else:
            row.first_seen = before.first_seen or generated
            if before.retired:
                report.returned.append(row.id)
            for name in CARRY_FORWARD:
                if _empty(getattr(row, name)) and not _empty(getattr(before, name)):
                    try:
                        setattr(row, name, getattr(before, name))
                    except ValidationError:
                        # The published value no longer fits this field's type;
                        # carrying it forward would only fail again next run.
                        continue
                    report.carried[name] = report.carried.get(name, 0) + 1
        row.last_seen = generated
        row.retired = False
        merged[row.id] = row

    for model_id, before in old.items():
        if model_id in merged:
            continue
        if not before.retired:
            report.retired.append(model_id)
        before.retired = True
        merged[model_id] = before

    return sorted(merged.values(), key=lambda row: row.id), report


def merge_detail(previous: dict[str, Any] | None, current: ModelDetail) -> ModelDetail:
    """Fold a fresh detail document into the published one.

    Args:
        previous: The published document, or ``None`` when there was none.
        current: The document this run collected.

    Returns:
        The merged document.
    """
    if not previous:
        return current
    for name in CARRY_FORWARD_DETAIL:
        if _empty(getattr(current, name)) and not _empty(previous.get(name)):
            setattr(current, name, previous[name])
    # Prices are not carried forward. They have an authoritative source every
    # run, so a missing one means AWS stopped publishing it — and a card
    # quoting last month's rates beside a table showing none is worse than a
    # card showing none.
    return current


#: Share of the published catalogue a run may lose before it is refused. A run
#: that loses this share *or more* is refused; only a smaller loss goes through.
RETIREMENT_CEILING: float = 0.2


class UnsafeUpdateError(RuntimeError):
    """A run would delete more of the catalogue than a real change could."""


def _unreadable(path: Path, detail: object) -> str:
    """Build the message for a previous file that cannot be used at all.

    Args:
        path: Path it was read from.
        detail: What went wrong, or a reason phrase.

    Returns:
        The message to raise :class:`UnsafeUpdateError` with.
    """
    return (
        f"{path} exists but could not be read ({detail}). Refusing to "
        f"regenerate from nothing — fix or delete it, or pass --fresh."
    )


def load_previous(path: Path) -> Catalog | None:
    """Read the published data set, when there is one.

    Args:
        path: Path of the published ``catalog.json``.

    Returns:
        The published catalogue, or ``None`` when there is none.

    Raises:
        UnsafeUpdateError: The published file exists but cannot be read as a
            catalogue with at least one model. Treating either as "no
            previous data" would silently drop every retired model and reset
            every ``first_seen``.
    """
    if not path.is_file():
        return None
    text = path.read_text()
    try:
        catalog = Catalog.model_validate_json(text)
    except ValidationError:
        # The strict schema rejected the file outright — most often a field
        # this version added or dropped since the file was written. The merge
        # only needs each model's own identity and history, so a looser read
        # recovers that instead of discarding it over an unrelated field.
        catalog = _lenient_catalog(text, path)
    if not catalog.models:
        msg = (
            f"{path} exists but lists no models. Treating that as no previous "
            f"data would reset every first_seen date and drop every retired "
            f"model. Refusing to regenerate from nothing — fix or delete it, "
            f"or pass --fresh."
        )
        raise UnsafeUpdateError(msg)
    return catalog


#: Values for the manifest fields a merge reads, when the old file lacks them.
#:
#: The merge only needs the region list, to keep every stored region index
#: pointing at the region it pointed at when it was written.
_MANIFEST_FLOOR: dict[str, Any] = {
    "generated": "",
    "gateway_version": "",
    "partitions": [],
    "currencies": [],
    "reference_region": "",
    "regions": [],
    "region_buckets": {},
}


def _lenient_catalog(text: str, path: Path) -> Catalog:
    """Rebuild the previous catalogue when the current schema rejects its shape.

    Any field the current schema no longer knows is dropped; any field it now
    requires that the file lacks falls back to that field's own default. A
    model missing here still needs its ``id``: that is the one fact a merge
    cannot proceed without.

    Args:
        text: The file's raw JSON text.
        path: Path it was read from, for the error message.

    Returns:
        The previous catalogue, rebuilt field by field.

    Raises:
        UnsafeUpdateError: The text is not JSON, or not shaped like a
            catalogue at all.
    """
    try:
        raw = json.loads(text)
    except ValueError as error:
        raise UnsafeUpdateError(_unreadable(path, error)) from error
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise UnsafeUpdateError(_unreadable(path, "not shaped like a catalogue"))
    manifest_data = raw.get("manifest")
    # model_construct leaves a required field simply unset, and reading one
    # then raises AttributeError far from here — the merge reads
    # manifest.regions. Validate what survives and let the rest default.
    known = {
        key: value
        for key, value in (
            manifest_data if isinstance(manifest_data, dict) else {}
        ).items()
        if key in Manifest.model_fields
    }
    try:
        manifest = Manifest(**known)
    except ValidationError:
        manifest = Manifest.model_construct(**{**_MANIFEST_FLOOR, **known})
    models = [
        _lenient_row(entry)
        for entry in raw["models"]
        if isinstance(entry, dict) and entry.get("id")
    ]
    return Catalog.model_construct(manifest=manifest, models=models)


def _lenient_row(entry: dict[str, Any]) -> ModelRow:
    """Rebuild one previously published row, keeping whatever still validates.

    A value the current schema rejects is dropped rather than carried: pydantic
    does not revalidate a model instance, so an unchecked one would be
    republished as-is and only fail the *next* run's schema check.

    Args:
        entry: One model as the previous file spells it.

    Returns:
        The row, with any unusable field left at its default.
    """
    known = {key: value for key, value in entry.items() if key in ModelRow.model_fields}
    try:
        return ModelRow(**known)
    except ValidationError:
        row = ModelRow.model_construct(id=str(entry["id"]))
        for name, value in known.items():
            with suppress(ValidationError, ValueError):
                setattr(row, name, value)
        return row


def check_sane(
    previous: Catalog | None,
    report: MergeReport,
    total: int,
    *,
    accept_retirements: bool = False,
) -> None:
    """Refuse an update that looks like a collection failure.

    An empty ``search_models`` or a half-answered one retires most of the
    catalogue in a single run, and the result is a green build and a gutted
    page. Nothing legitimate does that — except a real deprecation wave, which
    the operator can confirm with ``accept_retirements``.

    Args:
        previous: The published catalogue, when there was one.
        report: What the merge changed.
        total: How many models the run collected.
        accept_retirements: Skip the retirement-ceiling refusal. Set by an
            operator who has confirmed the loss is real, not a collection
            failure — unlike ``--fresh``, this keeps every model's history.

    Raises:
        UnsafeUpdateError: The run collected nothing, or retired too much and
            ``accept_retirements`` was not set.
    """
    if not total:
        msg = "the run collected no models at all; refusing to publish"
        raise UnsafeUpdateError(msg)
    if previous is None or not previous.models or accept_retirements:
        return
    live = report.previous_live
    if live and len(report.retired) / live >= RETIREMENT_CEILING:
        msg = (
            f"{len(report.retired)} of {live} models disappeared in one run, at "
            f"or over the {RETIREMENT_CEILING:.0%} ceiling. That is a collection "
            f"failure, not a release. Re-run, or pass --accept-retirements if "
            f"it is real."
        )
        raise UnsafeUpdateError(msg)
