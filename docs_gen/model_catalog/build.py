"""Assembles the published data set from the collected sources."""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from docs_gen.model_catalog import bedrock, gateway
from docs_gen.model_catalog import enrichment as enrichment_module
from docs_gen.model_catalog import merge as merge_module
from docs_gen.model_catalog.config import (
    DATA_DIR,
    DEFAULT_PARTITION,
    GLOBAL_PROFILE_PREFIXES,
    HEADLINE_DIMENSIONS,
    INDEX_GZIP_BUDGET,
    PROVIDER_LOGOS,
    REFERENCE_REGION,
    REGION_COUNTRIES,
    REPO_ROOT,
    SERVICE_LOGOS,
    SNAPSHOT_DIR,
    SOURCES,
    TIERS_NEEDING_A_REWRITE,
    UNMATCHED_PATH,
    board_fits_model,
    region_bucket,
)
from docs_gen.model_catalog.matching import CatalogModel, Matcher, loose_keys
from docs_gen.model_catalog.schema import (
    Catalog,
    Manifest,
    ModelDetail,
    ModelRow,
    PriceGroup,
    Reference,
    Score,
    ServiceVariant,
    SourceReport,
)
from docs_gen.model_catalog.sources import (
    RawReference,
    RawScore,
    SourceResult,
    aws_model_cards,
    lmarena,
    models_dev,
    mteb,
    open_asr,
)
from docs_gen.model_catalog.sources import epoch as epoch_source
from docs_gen.model_catalog.tokens import parse_tokens

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

#: Characters allowed in the filename of a per-model price document.
_UNSAFE_IN_SLUG: re.Pattern[str] = re.compile(r"[^A-Za-z0-9._-]")

#: Every source module the generator reads, keyed by source.
COLLECTORS: dict[str, Callable[..., SourceResult]] = {
    "lmarena": lmarena.fetch,
    "epoch": epoch_source.fetch,
    "mteb": mteb.fetch,
    "open_asr": open_asr.fetch,
}


@dataclass(slots=True)
class BuildReport:
    """What one run produced, for the CLI to print and for the release gate.

    Attributes:
        models: Models written to the index.
        price_documents: Per-model price documents written.
        matched: Scores and references attached to a model.
        llm_calls: Chat completions the matcher issued.
        index_bytes: Size of the gzipped index.
        merge: What folding this run into the published data set changed.
        enrichment: What the hand-curated overlay contributed.
        citations: Published values the overlay is the recorded source of.
        notes: Anything that degraded during the run.
    """

    models: int = 0
    price_documents: int = 0
    matched: int = 0
    llm_calls: int = 0
    index_bytes: int = 0
    merge: merge_module.MergeReport = field(default_factory=merge_module.MergeReport)
    enrichment: enrichment_module.Applied = field(
        default_factory=lambda: enrichment_module.Applied({}, {}, 0, [], [])
    )
    citations: int = 0
    notes: list[str] = field(default_factory=list)


#: Fill colours declared in a brand logo, ignoring gradients and "none".
_SVG_FILLS: re.Pattern[str] = re.compile(r'fill\s*[:=]\s*"?(#[0-9A-Fa-f]{3,8})')


def logo_backdrop(stem: str) -> str:
    """Decide what a brand logo needs behind it to stay visible.

    A logo is used unmodified — recolouring someone's mark is not ours to do —
    so the page adapts around it. Marks drawn in black ink disappear on a dark
    page and need a light chip; OpenAI's white mark needs the opposite; anything
    with real brand colour needs nothing at all.

    Args:
        stem: Logo asset stem under ``docs/styles``.

    Returns:
        ``light``, ``dark`` or an empty string.
    """
    if not stem:
        return ""
    path = REPO_ROOT / "docs" / "styles" / f"logo_{stem}.svg"
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    measured = (_luminance(value) for value in _SVG_FILLS.findall(text))
    inks = [ink for ink in measured if ink is not None]
    if not inks:
        # No fill at all means the SVG paints in the default black.
        return "light"
    # Judged on the darkest ink, not the average: a mark drawn in black with
    # white counter-shapes was designed for a light background, and the black is
    # the part that vanishes here.
    if min(inks) < 0.25:
        return "light"
    if min(inks) > 0.85:
        return "dark"
    return ""


def _luminance(colour: str) -> float | None:
    """Return the relative brightness of a hex colour.

    Args:
        colour: A ``#rgb`` or ``#rrggbb`` value.

    Returns:
        Brightness from 0 to 1, or ``None`` when the value cannot be read.
    """
    value = colour.lstrip("#")
    if len(value) == 3:
        value = "".join(channel * 2 for channel in value)
    if len(value) < 6:
        return None
    try:
        red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        return None
    return (0.299 * red + 0.587 * green + 0.114 * blue) / 255


def slug_for(model_id: str) -> str:
    """Return the filename stem of a model's price document.

    Args:
        model_id: Amazon Bedrock model ID.

    Returns:
        A filename-safe stem.
    """
    return _UNSAFE_IN_SLUG.sub("_", model_id)


def serving_geographies(
    inference_profiles: dict[str, str], regions: Sequence[str]
) -> list[str]:
    """Return where inference actually runs for a model.

    A model reachable from Frankfurt through a global inference profile is not
    the same answer as a model running in Frankfurt, and a reader filtering for
    sovereignty needs the second one.

    Args:
        inference_profiles: Region to cross-region inference profile ID.
        regions: Regions the model is callable from.

    Returns:
        Sorted geography markers: ``global``, a geography prefix, or a region.
    """
    geographies: set[str] = set()
    for region, profile in inference_profiles.items():
        if profile.startswith(GLOBAL_PROFILE_PREFIXES):
            geographies.add("global")
            continue
        prefix = profile.split(".", 1)[0]
        # A cross-region profile is named for the geography it runs in: us, eu,
        # apac, jp. Anything else is an opaque profile ID — showing one raw is
        # noise, so fall back to the region it is reached from.
        geographies.add(prefix if prefix.isalpha() and len(prefix) <= 4 else region)
    geographies.update(region for region in regions if region not in inference_profiles)
    return sorted(geographies)


def headline_prices(
    card: dict[str, Any],
) -> tuple[
    dict[tuple[str, str], dict[str, str]],
    dict[tuple[str, str], dict[str, str]],
    dict[tuple[str, str], str],
]:
    """Reduce a price card to the rates the table can show and sort on.

    Two rates for every way a region can reach the model: the standard tier,
    and the cheapest *whole* tier AWS publishes alongside it.

    Both of those words matter. ``routing`` names how the request reaches the
    model — in-region, through a geography profile, or globally — and those are
    different products at different prices, so the caller picks one and is
    quoted that one. Taking the cheapest per dimension across them would quote
    a bill nobody is ever sent. The same is true across tiers: ``flex`` and
    ``batch`` price some dimensions and not others, so a per-dimension minimum
    blends them into a tier that cannot be bought.

    The latency-optimised product is left out: it is a different thing to buy,
    not another way of running this one.

    Args:
        card: A ``model_pricing`` entry.

    Returns:
        Keyed by region and routing kind: the standard prices, the
        cheapest-tier prices, and which tier that was.
    """
    rows = [
        row
        for row in gateway.iter_price_rows(card)
        if not (row.get("cache_ttl") or row.get("context"))
        and str(row.get("dimension")) in HEADLINE_DIMENSIONS
    ]
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_region[str(row["region"])].append(row)

    standard: dict[tuple[str, str], dict[str, str]] = {}
    cheapest: dict[tuple[str, str], dict[str, str]] = {}
    tiers: dict[tuple[str, str], str] = {}
    for region, regional in by_region.items():
        for routing in sorted({str(row.get("routing") or "") for row in regional}):
            kind = _routing_kind(region, routing)
            if kind == "latency":
                # A separate, dearer product, not a way of running the same one.
                continue
            eligible = [
                row for row in regional if str(row.get("routing") or "") == routing
            ]
            plain = _tier_prices(eligible, "standard")
            if not plain:
                continue
            standard[region, kind] = plain
            discounted, tier = _cheapest_tier(eligible, plain)
            if discounted:
                cheapest[region, kind] = discounted
                tiers[region, kind] = tier
    return standard, cheapest, tiers


def _routing_kind(region: str, routing: str) -> str:
    """Name what kind of product a region's quoted routing is.

    Args:
        region: AWS region the price belongs to.
        routing: The ``routing`` value the price was taken from.

    Returns:
        ``region``, ``geography``, ``global``, ``latency``, or ``""`` when AWS
        publishes the price without a routing at all.
    """
    if not routing:
        return ""
    if routing == region:
        return "region"
    if routing in {"global", "latency"}:
        return routing
    return "geography"


def _tier_prices(rows: list[dict[str, Any]], tier: str) -> dict[str, str]:
    """Return one tier's price per billed dimension.

    A dimension priced both plainly and per media spec keeps the plain rate;
    image and video models publish nothing but spec rows, so for those the
    cheapest spec is the only price there is and quoting it beats a dash.

    Args:
        rows: Price rows already narrowed to one region and routing.
        tier: Service tier to read.

    Returns:
        Billed dimension to exact unit price.
    """
    prices: dict[str, str] = {}
    specified: dict[str, str] = {}
    for row in rows:
        if row.get("tier") != tier:
            continue
        dimension = str(row["dimension"])
        price = str(row["unit_price"])
        target = specified if row.get("spec") else prices
        current = target.get(dimension)
        if current is None or float(price) < float(current):
            target[dimension] = price
    for dimension, price in specified.items():
        prices.setdefault(dimension, price)
    return prices


def _cheapest_tier(
    rows: list[dict[str, Any]], standard: dict[str, str]
) -> tuple[dict[str, str], str]:
    """Return the cheapest whole tier published for a region.

    Only tiers a caller reaches by changing a request are eligible — see
    :data:`~docs_gen.model_catalog.config.TIERS_NEEDING_A_REWRITE`.

    Comparable only on the dimensions a tier shares with standard: crediting a
    tier for pricing fewer dimensions would reward it for selling less, not for
    costing less. A tier priced above standard on any shared dimension is not a
    discount and is skipped. Among the tiers left, each of which is cheaper or
    equal to standard on every shared dimension and strictly cheaper on at
    least one, the lowest weighted total wins; a tie goes to the tier sharing
    more dimensions with standard, so a broader discount is not lost to a
    narrower one that happens to match it on the dimensions the total weighs.

    Args:
        rows: Price rows already narrowed to one region and routing.
        standard: That region's standard-tier prices.

    Returns:
        The tier's full published prices — every dimension it prices, not only
        the ones shared with standard — and its name; an empty mapping and ""
        when no tier undercuts standard.
    """
    best: dict[str, str] = {}
    name = ""
    lowest: float | None = None
    breadth = 0
    candidates = {str(row.get("tier") or "") for row in rows}
    for tier in sorted(candidates - {"standard"} - TIERS_NEEDING_A_REWRITE):
        prices = _tier_prices(rows, tier)
        shared = {
            dimension: price
            for dimension, price in prices.items()
            if dimension in standard
        }
        if not shared:
            continue
        if any(
            float(price) > float(standard[dimension])
            for dimension, price in shared.items()
        ):
            continue
        if not any(
            float(price) < float(standard[dimension])
            for dimension, price in shared.items()
        ):
            continue
        total = _tier_total(shared)
        coverage = len(shared)
        if lowest is None or total < lowest or (total == lowest and coverage > breadth):
            best, name, lowest, breadth = prices, tier, total, coverage
    return best, name


def _tier_total(shared: dict[str, str]) -> float:
    """Score a tier by what a mixed request costs on the dimensions it shares with standard.

    Args:
        shared: The tier's prices, restricted to dimensions standard also prices.

    Returns:
        A comparable total: input and output tokens dominate a typical bill and
        are weighted accordingly; a tier pricing neither falls back to its
        cheapest shared dimension.
    """
    weights = {"input_tokens": 3.0, "output_tokens": 1.0}
    total = sum(
        float(shared[dimension]) * weight
        for dimension, weight in weights.items()
        if dimension in shared
    )
    return total or min(float(price) for price in shared.values())


def group_prices(
    standard: dict[tuple[str, str], dict[str, str]],
    cheapest: dict[tuple[str, str], dict[str, str]],
    tiers: dict[tuple[str, str], str],
    region_index: dict[str, int],
    available: set[str],
) -> list[PriceGroup]:
    """Collapse regions that publish identical prices.

    Emitted shape, one entry per distinct price set: ``regions`` (manifest
    region indices), ``prices`` (standard-tier unit price per billed
    dimension), ``cheapest`` (the cheapest tier's own unit price per
    dimension it prices — every one of them, not only those that differ from
    ``prices``) and ``cheapest_tier`` (that tier's name, ``""`` when no tier
    undercuts standard). A dimension present in ``prices`` but absent from
    ``cheapest`` is one the cheap tier does not sell at all; the page must show
    a dash for it, not the standard rate.

    A region appears once per routing AWS publishes for it, because routing is
    a choice the caller makes and the products are priced differently: staying
    inside a geography frequently costs more than letting AWS route globally.
    ``routing`` names which product the group prices.

    Args:
        standard: Region and routing kind, to billed dimension to standard-tier
            unit price.
        cheapest: The same, for the cheapest-tier unit price, already limited
            to the dimensions that tier actually prices.
        tiers: The same, for the name of that cheapest tier.
        region_index: Region to its index in the manifest.
        available: Regions the model can actually be called from. AWS publishes
            a price for regions that do not serve the model, and quoting one
            would price a call the reader cannot make.

    Returns:
        One group per distinct price set, regions sorted by index.
    """
    grouped: dict[str, list[int]] = defaultdict(list)
    for region, kind in sorted(set(standard) | set(cheapest)):
        if region not in region_index or region not in available:
            continue
        key = json.dumps(
            [
                standard.get((region, kind), {}),
                cheapest.get((region, kind), {}),
                tiers.get((region, kind), ""),
                kind,
            ],
            sort_keys=True,
        )
        grouped[key].append(region_index[region])
    groups = []
    for key, indices in sorted(grouped.items(), key=lambda item: sorted(item[1])):
        plain, discounted, tier, kind = json.loads(key)
        groups.append(
            PriceGroup(
                regions=sorted(indices),
                prices=plain,
                cheapest=discounted,
                cheapest_tier=tier,
                routing=kind,
            )
        )
    return groups


def collect_sources(
    *, refresh: bool, only: Iterable[str] | None = None
) -> dict[str, SourceResult]:
    """Read every published leaderboard.

    A source that cannot be read degrades its columns instead of failing the
    run, and says so in the returned notes.

    Args:
        refresh: Ignore cached snapshots.
        only: Restrict collection to these source keys.

    Returns:
        Source key to what it contributed.
    """
    wanted = set(only) if only is not None else set(COLLECTORS)
    results: dict[str, SourceResult] = {}
    for key, collector in COLLECTORS.items():
        if key not in wanted:
            continue
        try:
            results[key] = collector(refresh=refresh)
        except Exception as error:  # noqa: BLE001 -- one bad source must not fail the build
            results[key] = SourceResult(
                key=key,
                as_of="",
                scores=[],
                notes=[f"unreadable: {type(error).__name__}: {error}"],
            )
    return results


def _boards(result: SourceResult) -> dict[str, list[RawScore | RawReference]]:
    """Group one source's rows by sub-leaderboard.

    Args:
        result: What the source contributed.

    Returns:
        Board key to its rows; references share the pseudo-board ``reference``.
    """
    boards: dict[str, list[RawScore | RawReference]] = defaultdict(list)
    for score in result.scores:
        boards[score.board].append(score)
    for reference in result.references:
        boards["reference"].append(reference)
    return dict(boards)


def ambiguous_keys(models: Iterable[dict[str, Any]]) -> frozenset[str]:
    """Return the loose match keys more than one catalogue model answers to.

    Two Bedrock models can differ only by a release stamp. Their loosened keys
    collide, so those keys must never decide a match on their own.

    Args:
        models: ``search_models`` records.

    Returns:
        The colliding keys.
    """
    seen: Counter[str] = Counter()
    for model in models:
        candidate = CatalogModel(
            id=str(model["id"]),
            name=str(model.get("name") or model["id"]),
            provider=str(model.get("provider") or ""),
            aliases=tuple(str(value) for value in model.get("aliases", ())),
        )
        seen.update(loose_keys(candidate))
    return frozenset(key for key, count in seen.items() if count > 1)


def build(
    instance: gateway.Gateway,
    *,
    generated: str,
    refresh: bool = False,
    sources: Iterable[str] | None = None,
    matcher: Matcher | None = None,
    reuse: bool = True,
    accept_retirements: bool = False,
) -> tuple[
    Catalog,
    BuildReport,
    dict[str, dict[str, Any]],
    dict[str, SourceResult],
    dict[str, bedrock.ModelFacts],
]:
    """Assemble the published catalogue.

    Args:
        instance: Running stdapi.ai instance to read the catalogue from.
        generated: Date stamp recorded in the manifest.
        refresh: Ignore cached upstream snapshots.
        sources: Restrict collection to these source keys.
        matcher: Matcher to use; a rules-only matcher is built when omitted.
        reuse: Fold the run into the published data set instead of replacing it.
        accept_retirements: Publish a run that retires more models than the
            ceiling allows, keeping the history a fresh run would discard.

    Returns:
        The catalogue, a report of what the run produced, the full price cards
        and Amazon Bedrock facts the detail documents are written from, and what
        each source contributed.
    """
    report = BuildReport()
    matcher = matcher or Matcher()

    models = instance.models()
    matcher.set_ambiguous_keys(ambiguous_keys(models))
    price_cards = {str(card["id"]): card for card in instance.prices()}

    bedrock_regions = bedrock.commercial_bedrock_regions()
    regional = bedrock.list_foundation_models(bedrock_regions)
    unreachable = {entry.region: entry.error for entry in regional if entry.error}
    bedrock_facts = bedrock.index_by_model(regional)

    previous = merge_module.load_previous(DATA_DIR / "catalog.json") if reuse else None
    # Indices are stored in the artefact, including in rows this run did not
    # see. Appending keeps every existing index pointing at the same region;
    # re-sorting would silently repoint every retired model's availability.
    regions = list(previous.manifest.regions) if previous else []
    for region in sorted(
        {region for model in models for region in model.get("regions", ())}
    ):
        if region not in regions:
            regions.append(region)
    region_index = {region: index for index, region in enumerate(regions)}

    card_facts, published_facts = _stated_facts(models, report, refresh=refresh)

    collected = collect_sources(refresh=refresh, only=sources)
    for result in collected.values():
        report.notes.extend(f"{result.key}: {note}" for note in result.notes)
    boards_by_source = {key: _boards(result) for key, result in collected.items()}

    matched_names: dict[str, set[str]] = defaultdict(set)
    rows: list[ModelRow] = []
    for model in models:
        row = _build_row(model, price_cards, bedrock_facts, region_index)
        _attach_results(row, model, boards_by_source, collected, matcher, matched_names)
        report.matched += len(row.scores) + len(row.references)
        rows.append(row)

    rows, absorbed, fold_notes = fold_service_variants(rows)
    report.notes.extend(f"service variants: {note}" for note in fold_notes)
    apply_stated_facts(rows, card_facts, published_facts)
    # Last, so that a card can only ever fill a date Amazon Bedrock left unsaid.
    apply_card_lifecycle_fallback(rows, card_facts)

    overlay = enrichment_module.load()
    report.enrichment = enrichment_module.apply(rows, overlay)

    # A row folded into another was not withdrawn by AWS, so it must not be
    # carried forward and tagged retired: it is the same model, under the ID
    # the surviving row now lists as a variant.
    history = [
        row for row in (previous.models if previous else []) if row.id not in absorbed
    ]
    rows, merged = merge_module.merge_models(history, rows, generated=generated)
    merge_module.check_sane(
        previous, merged, len(models), accept_retirements=accept_retirements
    )
    _drop_impossible(rows)
    unpriced = [row.id for row in rows if not row.price_groups and not row.retired]
    if unpriced:
        # AWS serves a model before its rate reaches the Price List API, so an
        # empty price is news about AWS, not about the collection.
        report.notes.append(
            f"AWS publishes no price for {len(unpriced)} served model(s): "
            + ", ".join(sorted(unpriced))
        )
    # Citations are written against the merged catalogue, so a value a previous
    # run established and this one carried forward — including for a model AWS
    # has since retired — keeps the vendor page it came from.
    report.citations = enrichment_module.record_provenance(rows, overlay)
    report.models = len(rows)
    report.merge = merged
    report.llm_calls = matcher.llm_calls
    if matcher.llm_failures:
        report.notes.append(
            f"matching model: {matcher.llm_failures} of {matcher.llm_calls} calls "
            f"failed; last error: {matcher.last_error}"
        )
    matcher.save_cache()

    # A source that published nothing this run is credited with nothing, and a
    # row reading "0 of 0 entries" reads as a broken page rather than as the
    # unreachable source it is: the run's notes already report why it is missing.
    source_reports = [
        report_
        for report_ in (
            SourceReport(
                key=info.key,
                name=info.name,
                url=info.url,
                licence=info.licence,
                licence_url=info.licence_url,
                attribution=info.attribution,
                as_of=collected[info.key].as_of if info.key in collected else "",
                rows=_source_rows(info.key, collected, published_facts, card_facts),
                matched=_source_matched(
                    info.key, matched_names, rows, published_facts, card_facts
                ),
            )
            for info in SOURCES
            if info.key in collected or info.key in ("models_dev", "aws_model_cards")
        )
        if report_.rows
    ]

    manifest = Manifest(
        generated=generated,
        headline_dimensions=list(HEADLINE_DIMENSIONS),
        carried_fields=merged.carried,
        retired_models=sum(1 for row in rows if row.retired),
        gateway_version=instance.version(),
        partitions=[DEFAULT_PARTITION],
        currencies=sorted({row.currency for row in rows}),
        reference_region=REFERENCE_REGION,
        regions=regions,
        region_buckets={region: region_bucket(region) for region in regions},
        region_countries={
            region: REGION_COUNTRIES[region]
            for region in regions
            if region in REGION_COUNTRIES
        },
        unreachable_regions=unreachable,
        sources=source_reports,
    )
    return (
        Catalog(manifest=manifest, models=rows),
        report,
        price_cards,
        collected,
        bedrock_facts,
    )


def _source_rows(
    key: str,
    collected: dict[str, SourceResult],
    published_facts: dict[str, dict[str, Any]],
    card_facts: dict[str, dict[str, Any]],
) -> int:
    """Return how many entries a source published this run.

    ``card_facts`` and ``published_facts`` are already joined to this
    catalogue, so their length is a match count, not a publication count — it
    is read instead from the source's own cached raw snapshot, which predates
    any join. Falls back to the joined count when that snapshot is missing.

    Args:
        key: Source key.
        collected: Source key to what each leaderboard contributed.
        published_facts: Model ID to the facts models.dev publishes for it.
        card_facts: Model ID to the facts its AWS model card states.

    Returns:
        The entry count.
    """
    if key == "aws_model_cards":
        return _snapshot_length(key) or len(card_facts)
    if key == "models_dev":
        return _snapshot_length(key) or len(published_facts)
    result = collected.get(key)
    return len(result.scores) + len(result.references) if result else 0


def _snapshot_length(name: str) -> int:
    """Return how many entries a source's cached raw snapshot holds.

    Args:
        name: Snapshot key, matching :func:`sources.snapshot`'s ``name``
            argument as called with no ``key``.

    Returns:
        The entry count, or ``0`` when the snapshot cannot be read.
    """
    path = SNAPSHOT_DIR / f"{name}.all.json"
    try:
        raw = json.loads(path.read_text())
    except OSError, ValueError:
        return 0
    return len(raw) if isinstance(raw, list | dict) else 0


def _source_matched(
    key: str,
    matched_names: dict[str, set[str]],
    rows: list[ModelRow],
    published_facts: dict[str, dict[str, Any]],
    card_facts: dict[str, dict[str, Any]],
) -> int:
    """Return how many of a source's entries reached a model in the catalogue.

    Args:
        key: Source key.
        matched_names: Accumulator of matched source names, per source.
        rows: The assembled rows.
        published_facts: Model ID to the facts models.dev publishes for it.
        card_facts: Model ID to the facts its AWS model card states.

    Returns:
        The matched count.
    """
    if key == "aws_model_cards":
        return sum(1 for row in rows if row.id in card_facts)
    if key == "models_dev":
        return sum(1 for row in rows if row.id in published_facts)
    return len(matched_names[key])


def _build_row(
    model: dict[str, Any],
    price_cards: dict[str, dict[str, Any]],
    bedrock_facts: dict[str, bedrock.ModelFacts],
    region_index: dict[str, int],
) -> ModelRow:
    """Turn one catalogue entry into a table row.

    Args:
        model: A ``search_models`` record.
        price_cards: Model ID to its ``model_pricing`` card.
        bedrock_facts: Model ID to its per-region Bedrock capabilities.
        region_index: Region to its index in the manifest.

    Returns:
        The assembled row, without scores.
    """
    model_id = str(model["id"])
    profiles: dict[str, str] = {
        str(k): str(v) for k, v in (model.get("inference_profiles") or {}).items()
    }
    regions = [str(region) for region in model.get("regions", ())]
    card = price_cards.get(model_id)
    by_routing, cheaper, cheap_tiers = headline_prices(card) if card else ({}, {}, {})
    facts_entry = bedrock_facts.get(model_id) or bedrock.ModelFacts({}, {})
    facts = facts_entry.table
    return ModelRow(
        id=model_id,
        slug=slug_for(model_id),
        name=str(model.get("name") or model_id),
        provider=str(model.get("provider") or ""),
        service=str(model.get("service") or ""),
        input_modalities=[str(value) for value in model.get("input_modalities", ())],
        output_modalities=[str(value) for value in model.get("output_modalities", ())],
        aliases=[str(value) for value in model.get("aliases", ())],
        routes=[str(value) for value in model.get("supported_routes", ())],
        mcp_tools=[str(value) for value in model.get("supported_mcp_tools", ())],
        regions=sorted(
            region_index[region] for region in regions if region in region_index
        ),
        buckets=sorted({region_bucket(region) for region in regions}),
        serving=serving_geographies(profiles, regions),
        streaming=model.get("response_streaming"),
        batch=model.get("batch"),
        legacy=bool(model.get("legacy")),
        start_of_life=_date(model.get("start_of_life_time")),
        end_of_life=_date(model.get("end_of_life_time")),
        legacy_time=_date(model.get("legacy_time")),
        public_extended_access=_date(model.get("public_extended_access_time")),
        inference_types=facts.get("inference_types", []),
        customizations=facts.get("customizations", []),
        logo=PROVIDER_LOGOS.get(str(model.get("provider") or ""), ""),
        service_logo=SERVICE_LOGOS.get(str(model.get("service") or ""), ""),
        logo_backdrop=logo_backdrop(
            PROVIDER_LOGOS.get(str(model.get("provider") or ""), "")
        ),
        family=(facts.get("families") or [""])[0],
        apis=facts.get("apis", []),
        max_output_tokens=facts.get("max_output_tokens"),
        context_window=facts_entry.detail.get("context_window"),
        prompt_caching=facts.get("prompt_caching"),
        guardrails=facts.get("guardrails"),
        latency_optimized=facts.get("latency_optimized"),
        provisioned=facts.get("provisioned"),
        count_tokens=facts.get("count_tokens"),
        prompt_routing=facts.get("prompt_routing"),
        batch_in_region=facts.get("batch_in_region"),
        batch_cross_region=facts.get("batch_cross_region"),
        image_types=facts.get("image_types", []),
        document_types=facts.get("document_types", []),
        video_types=facts.get("video_types", []),
        currency=_currency(card),
        default_routing=_default_routing(card),
        price_groups=group_prices(
            by_routing, cheaper, cheap_tiers, region_index, set(regions)
        ),
        has_prices=bool(card and card.get("prices")),
    )


def _default_routing(card: dict[str, Any] | None) -> str:
    """Return the routing this gateway would bill, when it is not the region.

    ``model_pricing`` reports the routings a request would take. When that is
    exactly ``global`` the gateway bills the global rate, so quoting the
    in-region one would show a price the caller is never charged.

    Args:
        card: The model's price card, or ``None``.

    Returns:
        ``global``, or an empty string when the region's own rate applies.

    Ref: stdapi/routes/core_models.py
    """
    routings = (card or {}).get("default_routings")
    return "global" if routings == ["global"] else ""


def _drop_impossible(rows: Iterable[ModelRow]) -> None:
    """Unset values no model can have, whichever source or run produced them.

    An output ceiling equal to the context window means a model that can answer
    with its entire input, which none can. Both the raw API and models.dev
    publish that shape as a stand-in for "unknown", and the merge would carry
    one forward for good, so the invariant is enforced on the finished rows
    rather than at any single source.

    Args:
        rows: The published rows, after the merge.
    """
    for row in rows:
        context = parse_tokens(row.context_window)
        if context and row.max_output_tokens and row.max_output_tokens >= context:
            row.max_output_tokens = None


#: A trailing API-version tag, as Bedrock Runtime appends it to a model ID.
_API_VERSION_TAG: re.Pattern[str] = re.compile(r"-v?\d+:\d+$")

#: Row fields that are the union of what every service variant offers.
_VARIANT_UNION: tuple[str, ...] = (
    "regions",
    "buckets",
    "serving",
    "routes",
    "mcp_tools",
    "aliases",
    "apis",
    "inference_types",
    "customizations",
    "image_types",
    "document_types",
    "video_types",
    "input_modalities",
    "output_modalities",
)

#: Row fields holding a model's lifecycle dates.
#:
#: Amazon Bedrock states these itself, in ``ListFoundationModels``'s
#: ``modelLifecycle``, and that is the catalogue's reference for them. They are
#: named here so that the precedence is a rule the code states rather than a
#: consequence of the order two functions happen to run in.
_LIFECYCLE_FIELDS: tuple[str, ...] = ("start_of_life", "end_of_life")

#: Row fields stating a fact about the model rather than a capability of one
#: service, which a folded row therefore takes from whichever variant states it.
#:
#: This is the fact half of the split ``_absorb_variant`` makes; the capability
#: half is ``_VARIANT_EITHER``, which is true when *either* service offers it and
#: so may only ever be turned on by a variant, never off. Keep a field in exactly
#: one of the two: reading a capability off an absorbed variant would let a third
#: party stating ``reasoning: false`` publish a denial where the catalogue had
#: honestly said nothing.
_VARIANT_FACTS: tuple[str, ...] = (
    "context_window",
    "max_output_tokens",
    "knowledge_cutoff",
    "licence",
    "start_of_life",
    "end_of_life",
)

#: Row fields true for the model when either service offers them.
_VARIANT_EITHER: tuple[str, ...] = (
    "streaming",
    "batch",
    "batch_in_region",
    "batch_cross_region",
    "prompt_caching",
    "guardrails",
    "latency_optimized",
    "provisioned",
    "count_tokens",
    "prompt_routing",
    "reasoning",
    "tool_call",
)


def _stated_facts(
    models: list[dict[str, Any]], report: BuildReport, *, refresh: bool
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Read the facts no AWS API returns, from the sources that state them.

    Either source failing degrades the columns it fills rather than the run.

    Args:
        models: The catalogue as the gateway reports it.
        report: Run report, for the notes.
        refresh: Ignore cached snapshots.

    Returns:
        What the AWS model cards state, and what models.dev publishes.
    """
    try:
        cards, notes = aws_model_cards.fetch(
            (str(model["id"]) for model in models), refresh=refresh
        )
        report.notes.extend(f"aws model cards: {note}" for note in notes)
    except Exception as error:  # noqa: BLE001 -- one bad source must not fail the build
        cards = {}
        report.notes.append(
            f"aws model cards unreadable: {type(error).__name__}: {error}"
        )

    try:
        published = models_dev.fetch(refresh=refresh)
    except Exception as error:  # noqa: BLE001 -- one bad source must not fail the build
        published = {}
        report.notes.append(f"models.dev unreadable: {type(error).__name__}: {error}")
    return cards, published


def fold_service_variants(
    rows: list[ModelRow],
) -> tuple[list[ModelRow], set[str], list[str]]:
    """Merge the rows that are one model reached through two AWS services.

    Amazon serves several models through both Bedrock Runtime and Bedrock
    Mantle, under IDs differing only by the Runtime API-version tag. They are
    one model reached two ways, and listing them twice doubles the row and
    makes the reader compare a model against itself. Their rates are usually
    identical but not always — in ``ap-southeast-2`` Mantle is about 14%
    cheaper — so the prices are merged rather than assumed equal, and each
    keeps the service it belongs to wherever the two disagree.

    Args:
        rows: Every row this run assembled.

    Returns:
        The rows with each such pair folded into one, the IDs that were folded
        away, and any notes worth reporting.
    """
    families: dict[tuple[str, str], list[ModelRow]] = defaultdict(list)
    for row in rows:
        families[(row.provider, _API_VERSION_TAG.sub("", row.id))].append(row)

    folded: list[ModelRow] = []
    notes: list[str] = []
    absorbed: set[str] = set()
    for members in families.values():
        services = {row.service for row in members}
        if len(members) < 2 or len(services) < 2:
            continue
        primary = _headline_variant(members)
        for other in members:
            if other is not primary:
                _absorb_variant(primary, other)
                absorbed.add(other.id)
        primary.price_groups = _merge_variant_prices(members)
        primary.variants = sorted(
            (
                ServiceVariant(
                    id=row.id,
                    service=row.service,
                    service_logo=row.service_logo,
                    regions=list(row.regions),
                )
                for row in members
            ),
            key=lambda variant: variant.service,
        )
        folded.append(primary)

    if folded:
        notes.append(
            f"{len(folded)} model(s) folded from two AWS services into one row"
        )
    return [row for row in rows if row.id not in absorbed], absorbed, notes


def _merge_variant_prices(members: list[ModelRow]) -> list[PriceGroup]:
    """Combine the price groups of a model served through several services.

    The two surfaces usually charge the same, but not always: in
    ``ap-southeast-2`` Bedrock Mantle is about 14% cheaper than Bedrock Runtime
    for the models served on both. Every group therefore keeps the service it
    belongs to, and the page quotes the one the reader would pay. Where the
    services agree on a region, the duplicate is dropped and the group is left
    unattributed, because naming a service would suggest a choice that has no
    consequence.

    Args:
        members: The rows being folded together.

    Returns:
        The union of their price groups, each tagged with its service only
        where the services actually differ.
    """
    by_key: dict[tuple[int, str], list[tuple[str, PriceGroup]]] = defaultdict(list)
    for row in members:
        for group in row.price_groups:
            for index in group.regions:
                by_key[index, group.routing].append((row.service, group))

    merged: list[PriceGroup] = []
    for (index, routing), entries in sorted(by_key.items()):
        rates = {
            _dump([group.prices, group.cheapest, group.cheapest_tier])
            for _, group in entries
        }
        for service, group in entries:
            merged.append(
                group.model_copy(
                    update={
                        "regions": [index],
                        "routing": routing,
                        "service": service if len(rates) > 1 else "",
                    }
                )
            )
            if len(rates) == 1:
                break
    return _collapse_groups(merged)


def _collapse_groups(groups: list[PriceGroup]) -> list[PriceGroup]:
    """Put back together the regions that share a price.

    Args:
        groups: One-region groups.

    Returns:
        One group per distinct price set, regions sorted by index.
    """
    grouped: dict[str, list[int]] = defaultdict(list)
    shapes: dict[str, PriceGroup] = {}
    for group in groups:
        key = _dump(
            [
                group.prices,
                group.cheapest,
                group.cheapest_tier,
                group.routing,
                group.service,
            ]
        )
        grouped[key].extend(group.regions)
        shapes[key] = group
    return [
        shapes[key].model_copy(update={"regions": sorted(set(indices))})
        for key, indices in sorted(grouped.items(), key=lambda item: sorted(item[1]))
    ]


def _headline_variant(members: list[ModelRow]) -> ModelRow:
    """Pick the variant whose ID and name the folded row carries.

    Args:
        members: The rows being folded together.

    Returns:
        The row to keep, chosen by reach so the choice is stable between runs.
    """
    return min(members, key=lambda row: (-len(row.regions), row.id))


def _absorb_variant(primary: ModelRow, other: ModelRow) -> None:
    """Fold one service's row into the row the catalogue will publish.

    Args:
        primary: The row being kept.
        other: The row being folded into it.
    """
    for name in _VARIANT_UNION:
        merged = set(getattr(primary, name)) | set(getattr(other, name))
        setattr(primary, name, sorted(merged))
    for name in _VARIANT_EITHER:
        if getattr(other, name) and not getattr(primary, name):
            setattr(primary, name, getattr(other, name))
    # A fact one surface reports and the other does not is still the model's.
    # The lifecycle dates are here because Bedrock Mantle's listing carries no
    # ``modelLifecycle`` at all, so only the Runtime twin ever states them.
    for name in _VARIANT_FACTS:
        if getattr(primary, name) in (None, "", []):
            setattr(primary, name, getattr(other, name))
    # AWS spells the same model two ways; the prose name reads better than the
    # bare identifier the OpenAI-compatible surface reports.
    if (" " in other.name) > (" " in primary.name):
        primary.name = other.name
    if other.id not in primary.aliases:
        primary.aliases = sorted({*primary.aliases, other.id})


def apply_stated_facts(
    rows: Iterable[ModelRow],
    card_facts: dict[str, dict[str, Any]],
    published_facts: dict[str, dict[str, Any]],
) -> None:
    """Fill the facts no AWS API returns, once the service variants are folded.

    Read after the fold, so that the row being filled is the one the catalogue
    publishes. The lifecycle dates are not read here at all:
    :func:`apply_card_lifecycle_fallback` resolves those against Amazon Bedrock's
    own answer, which outranks what a card states.

    A folded row is one model reached through several IDs, so a fact stated
    against any of them is the model's own — models.dev keys its entries to the
    Runtime ID, which the fold absorbs. Only ``_VARIANT_FACTS`` are read that
    way, mirroring the fact-versus-capability split ``_absorb_variant`` already
    makes: a capability an absorbed variant denies must not overwrite what the
    surviving row leaves unknown.

    The loops run in precedence order and ``_apply_published_facts`` never
    overwrites, so the first source to state a fact wins: AWS's own card before
    the open database, and a row's own ID before one it absorbed.

    Args:
        rows: The rows the catalogue will publish, after folding.
        card_facts: Model ID to the facts its AWS model card states.
        published_facts: Model ID to the facts models.dev publishes for it.
    """
    for row in rows:
        absorbed = [item.id for item in row.variants if item.id != row.id]
        for source in (card_facts, published_facts):
            _apply_published_facts(row, _model_wide(source.get(row.id, {})))
            for identity in absorbed:
                _apply_published_facts(
                    row,
                    {
                        name: value
                        for name, value in _model_wide(source.get(identity, {})).items()
                        if name in _VARIANT_FACTS
                    },
                )


def _model_wide(stated: dict[str, Any]) -> dict[str, Any]:
    """Drop the facts a source may not decide on its own.

    Args:
        stated: What one source publishes for one model ID.

    Returns:
        The same facts without the lifecycle dates, which
        :func:`apply_card_lifecycle_fallback` resolves against Amazon Bedrock's
        own answer instead.
    """
    return {
        name: value for name, value in stated.items() if name not in _LIFECYCLE_FIELDS
    }


def apply_card_lifecycle_fallback(
    rows: Iterable[ModelRow], card_facts: dict[str, dict[str, Any]]
) -> None:
    """Fill a lifecycle date only where no Amazon Bedrock API stated one.

    Amazon Bedrock's ``modelLifecycle`` is the reference for these dates, and a
    model card is a fallback for the models it says nothing about. The order is
    stated here rather than left to whichever function happens to run first,
    because a reordering elsewhere would otherwise invert it silently.

    The fallback is an approximation, knowingly accepted. A card's "Launch date"
    is frequently the vendor's announcement of the model rather than the date it
    reached Amazon Bedrock — the Qwen cards state the upstream Alibaba release —
    so it answers a different question from the one this column asks. It is
    published where nothing better exists because a date close to the truth
    serves a reader better than an empty cell.

    Args:
        rows: The rows the catalogue will publish, after folding.
        card_facts: Model ID to the facts its AWS model card states.
    """
    for row in rows:
        for identity in (row.id, *(item.id for item in row.variants)):
            stated = card_facts.get(identity, {})
            for name in _LIFECYCLE_FIELDS:
                if getattr(row, name) in (None, "", []) and stated.get(name):
                    setattr(row, name, stated[name])


def _apply_published_facts(row: ModelRow, facts: dict[str, Any]) -> None:
    """Fill the facts AWS's API does not return, from a source that states them.

    First source to state a fact wins, and the model cards are read before the
    open database: AWS's own page for a model outranks a third party's entry
    for it. Nothing here overwrites a value already collected.

    Args:
        row: Row being assembled.
        facts: What the source publishes for this model ID. A value the schema
            rejects is dropped.
    """
    for name, value in facts.items():
        if getattr(row, name, None) not in (None, "", []):
            continue
        try:
            setattr(row, name, value)
        except ValidationError, ValueError:
            # A source that starts returning a different shape loses that one
            # fact, rather than taking the whole catalogue down with it.
            continue


def _attach_results(
    row: ModelRow,
    model: dict[str, Any],
    boards_by_source: dict[str, dict[str, list[RawScore | RawReference]]],
    collected: dict[str, SourceResult],
    matcher: Matcher,
    matched_names: dict[str, set[str]],
) -> None:
    """Match every board onto one model and attach what it found.

    Args:
        row: Row being assembled.
        model: The ``search_models`` record behind it.
        boards_by_source: Source key to board key to rows.
        collected: Source key to what it contributed.
        matcher: Matcher deciding the mappings.
        matched_names: Accumulator of matched source names, per source.
    """
    candidate = CatalogModel(
        id=row.id,
        name=row.name,
        provider=row.provider,
        aliases=tuple(str(value) for value in model.get("aliases", ())),
    )
    for source_key, boards in boards_by_source.items():
        for board, board_rows in boards.items():
            if not board_fits_model(
                f"{source_key}/{board}", row.input_modalities, row.output_modalities
            ):
                continue
            decision, chosen = matcher.match_board(
                candidate, source_key, board, board_rows
            )
            if chosen is None or decision.matched_name is None:
                continue
            matched_names[source_key].add(chosen.name)
            if isinstance(chosen, RawScore):
                if chosen.licence and not row.licence:
                    row.licence = chosen.licence
                row.scores.append(
                    Score(
                        source=source_key,
                        board=board,
                        metric=chosen.metric,
                        label=chosen.label,
                        value=chosen.value,
                        unit=chosen.unit,
                        higher_is_better=chosen.higher_is_better,
                        rank=chosen.rank,
                        ci_low=chosen.ci_low,
                        ci_high=chosen.ci_high,
                        samples=chosen.samples,
                        as_of=chosen.as_of or collected[source_key].as_of,
                        matched_name=chosen.name,
                        match_method=decision.method,
                    )
                )
            else:
                row.references.append(
                    Reference(
                        source=source_key,
                        label=chosen.label,
                        detail=chosen.detail,
                        url=chosen.url,
                        matched_name=chosen.name,
                        match_method=decision.method,
                    )
                )


def _date(value: object) -> str | None:
    """Return the date part of a timestamp.

    Args:
        value: A date or timestamp string, or ``None``.

    Returns:
        An ISO date, or ``None``.
    """
    return str(value)[:10] if value else None


def _currency(card: dict[str, Any] | None) -> str:
    """Return the currency a price card publishes.

    Args:
        card: A ``model_pricing`` entry, or ``None``.

    Returns:
        The ISO currency code, defaulting to ``USD``.
    """
    if not card:
        return "USD"
    for row in card.get("prices", ()):
        return str(row.get("currency") or "USD")
    return "USD"


def write(
    catalog: Catalog,
    price_cards: dict[str, dict[str, Any]],
    bedrock_facts: dict[str, bedrock.ModelFacts],
    report: BuildReport,
) -> None:
    """Write the published artefacts.

    One document per model carries everything the table does not: the full
    published price matrix and the vendor's own prose. Both are an order of
    magnitude larger than the row they belong to, and neither is needed until a
    reader opens that one model.

    Args:
        catalog: The assembled catalogue.
        price_cards: Model ID to its full ``model_pricing`` card.
        bedrock_facts: Model ID to its collected Amazon Bedrock metadata.
        report: Report updated with what was written.

    Raises:
        ValueError: The index exceeds its first-paint budget.
    """
    detail_dir = DATA_DIR / "detail"
    detail_dir.mkdir(parents=True, exist_ok=True)
    published: dict[str, dict[str, Any]] = {}
    for existing in detail_dir.glob("*.json"):
        with suppress(ValueError):
            published[existing.stem] = json.loads(existing.read_text())
        existing.unlink()

    regions = catalog.manifest.regions
    for row in catalog.models:
        facts = bedrock_facts.get(row.id)
        detail = merge_module.merge_detail(
            published.get(row.slug),
            ModelDetail(
                id=row.id,
                prices=_priced_where_served(
                    price_cards.get(row.id) or {},
                    {regions[index] for index in row.regions if index < len(regions)},
                ),
                **(facts.detail if facts else {}),
            ),
        )
        (detail_dir / f"{row.slug}.json").write_text(
            _dump(detail.model_dump(exclude_none=True, mode="json"))
        )
        report.price_documents += 1

    index = _dump(catalog.model_dump(exclude_none=True, mode="json"))
    (DATA_DIR / "catalog.json").write_text(index)
    report.index_bytes = len(gzip.compress(index.encode()))
    if report.index_bytes > INDEX_GZIP_BUDGET:
        msg = (
            f"index is {report.index_bytes} bytes gzipped, over the "
            f"{INDEX_GZIP_BUDGET} byte first-paint budget"
        )
        raise ValueError(msg)


def _priced_where_served(card: dict[str, Any], available: set[str]) -> dict[str, Any]:
    """Drop the price rows for regions the model cannot be called from.

    AWS publishes a rate for every region whether or not the model is served
    there: Claude 3.7 Sonnet is callable from two regions and priced in
    thirty-one. Quoting the other twenty-nine would offer a price nobody can
    buy, and would contradict the table, which already counts only the regions
    the model runs in.

    Args:
        card: The model's full price card as the gateway returned it.
        available: Regions the model is callable from.

    Returns:
        The card with its price rows narrowed, or unchanged when it has none.
    """
    rows = card.get("prices")
    if not isinstance(rows, list):
        return card
    return {
        **card,
        "prices": [
            row
            for row in rows
            if not isinstance(row, dict) or row.get("region") in available
        ],
    }


def write_unmatched(
    catalog: Catalog, collected: dict[str, SourceResult]
) -> dict[str, Any]:
    """Record what neither pass could map, in both directions.

    Args:
        catalog: The assembled catalogue.
        collected: Source key to what it contributed.

    Returns:
        The report that was written.
    """
    matched: dict[str, set[str]] = defaultdict(set)
    claims: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in catalog.models:
        for score in row.scores:
            matched[score.source].add(score.matched_name)
            claims[score.source, score.board, score.matched_name].append(row.id)
        for reference in row.references:
            matched[reference.source].add(reference.matched_name)
            claims[reference.source, "reference", reference.matched_name].append(row.id)

    published: dict[str, set[str]] = {
        key: {row.name for row in result.scores}
        | {row.name for row in result.references}
        for key, result in collected.items()
    }
    report = {
        "rows_without_a_model": {
            key: sorted(names - matched[key]) for key, names in published.items()
        },
        "models_without_a_score": sorted(
            row.id for row in catalog.models if not row.scores and not row.references
        ),
        # Two Bedrock IDs often serve the same model, so sharing an entry is
        # usually right — but it is also what a wrong match looks like, so every
        # shared entry is listed here for review against the override file.
        "entries_shared_by_several_models": {
            f"{source}/{board}/{name}": sorted(ids)
            for (source, board, name), ids in sorted(claims.items())
            if len(ids) > 1
        },
    }
    UNMATCHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    UNMATCHED_PATH.write_text(_dump(report))
    return report


def _dump(value: object) -> str:
    """Serialise an artefact deterministically.

    Args:
        value: The document to serialise.

    Returns:
        Compact JSON with a trailing newline, so a regeneration diffs cleanly.
    """
    return json.dumps(value, separators=(",", ":"), sort_keys=False, default=str) + "\n"
