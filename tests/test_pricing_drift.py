"""Drift detection for ``DEFAULT_MODEL_PRICES`` against the sources it was copied from.

``stdapi/models/pricing_overrides.py:DEFAULT_MODEL_PRICES`` and its Global
cross-Region twin ``DEFAULT_MODEL_GLOBAL_PRICES`` are hand-copied tables of
rates the AWS Price List API does not publish. Nothing in the running gateway
can notice when AWS changes one of them: the figure is simply reported to the
operator as fact. It has already happened -- GPT-5.6 Luna shipped at 5x the
real rate for two releases -- so the tables need a check that reads the vendor
source and says so.

Two sources, and they are not interchangeable:

- **Bedrock model cards** (``docs.aws.amazon.com``) for the OpenAI Mantle
  models. Server-rendered documentation with a labelled ``Pricing`` section and
  a per-1M-token table, one row per inference option. The AWS Bedrock pricing
  page no longer carries per-1M rates for these models at all -- it links out to
  the cards -- so the card is the *only* AWS source for them. The ``In-Region``
  row prices ``DEFAULT_MODEL_PRICES``; the ``Global CRIS`` row of the *same*
  table prices ``DEFAULT_MODEL_GLOBAL_PRICES``, and most cards carry no such row
  at all.
- **The AWS Bedrock pricing page** for the Stability AI image services, whose
  per-generation rates live in one table keyed by display name.

Both are HTML, so the detector's first duty is to tell "the price changed" from
"I could not read the page". They are different events with different answers,
and conflating them produces the false alarms that get a detector switched off:

===========  =============================================  ================
Outcome      Meaning                                        Effect
===========  =============================================  ================
MATCH        the table agrees with the source               none
DRIFT        the source publishes a different rate          **fails**
VANISHED     the source no longer publishes this rate       reported only
NEW          the source publishes a rate the table lacks    reported only
UNREACHABLE  the source could not be fetched or parsed      reported only
===========  =============================================  ================

**A vanished price is never removed and never fails.** Usage recorded against a
delisted, renamed or enrollment-gated model still has to be priced, so the entry
stays and the run says so out loud. That covers a Global rate a card stops
quoting as much as a delisted model: the ``global.`` inference profile keeps
serving calls that have to be priced somehow.

A model whose card never published a Global rate is **silent**, not vanished:
most models publish only In-Region, so its absence from
``DEFAULT_MODEL_GLOBAL_PRICES`` is the correct state rather than a finding.

A hand-copied rate also stops being the right answer the moment AWS starts
publishing one, so a second live check reads the Price List itself and fails
when any table entry's model has gained a row -- in either form, a native
Bedrock row or an ``(Amazon Bedrock Edition)`` Marketplace listing.
``_apply_default_prices`` guards per *model*, not per dimension: one published
row discards the whole hand-copied entry, silently unpricing every dimension
AWS did not publish.

The live check is opt-in (``--drift``): a vendor changing a price is not a
regression in this repository, and it must never turn an unrelated run red. It
fails only inside its own lane, where a hard failure is the point. The
classifier itself is exercised offline against recorded fixtures, including a
deliberately wrong expected value, because a detector nobody has seen fail is
not known to work.

Ref: stdapi/models/pricing_overrides.py:DEFAULT_MODEL_PRICES
     stdapi/models/pricing_overrides.py:DEFAULT_MODEL_GLOBAL_PRICES
     stdapi/pricing.py:register_default_prices
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-openai.html
     https://aws.amazon.com/bedrock/pricing/
"""

import re
import warnings
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from html import unescape
from typing import TYPE_CHECKING, Final

import httpx
import pytest
from botocore.exceptions import BotoCoreError, ClientError

from stdapi import pricing
from stdapi.aws import AWSConnectionManager, get_client
from stdapi.config import SETTINGS
from stdapi.models.pricing_overrides import (
    DEFAULT_MODEL_GLOBAL_LONG_CONTEXT_PRICES,
    DEFAULT_MODEL_GLOBAL_PRICES,
    DEFAULT_MODEL_LONG_CONTEXT_PRICES,
    DEFAULT_MODEL_PRICES,
    MODEL_LONG_CONTEXT_THRESHOLDS,
)
from stdapi.pricing import ContextLength, Dimension
from tests.conftest import REPO_ROOT

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Recorded source excerpts backing the offline classifier tests.
FIXTURES_DIR: Final = REPO_ROOT / "tests" / "fixtures" / "pricing"

#: Seconds a vendor page gets to answer; slower than this counts as unreachable.
_FETCH_TIMEOUT: Final[float] = 30.0

#: Identifies these requests to the vendor's CDN.
_USER_AGENT: Final[str] = "stdapi.ai-tests price-drift detector"

#: Bedrock model card per OpenAI Mantle model, the sole AWS source for its rates.
_MODEL_CARD_URLS: Final[dict[str, str]] = {
    "openai.gpt-5.4": "model-card-openai-gpt-54",
    "openai.gpt-5.5": "model-card-openai-gpt-55",
    "openai.gpt-5.6-cyber": "model-card-openai-gpt-56-cyber",
    "openai.gpt-5.6-luna": "model-card-openai-gpt-56-luna",
    "openai.gpt-5.6-sol": "model-card-openai-gpt-56-sol",
    "openai.gpt-5.6-terra": "model-card-openai-gpt-56-terra",
    "openai.gpt-daybreak-blue-5.6-sol": "model-card-openai-gpt-daybreak-blue-56-sol",
}

#: Where a model card lives, given its slug.
_USER_GUIDE: Final[str] = "https://docs.aws.amazon.com/bedrock/latest/userguide/"

#: The OpenAI card index, scanned for GPT-5.x cards no table entry prices.
_OPENAI_CARD_INDEX: Final[str] = f"{_USER_GUIDE}model-cards-openai.html"

#: The page carrying the Stability AI Image Services per-generation table.
_BEDROCK_PRICING_PAGE: Final[str] = "https://aws.amazon.com/bedrock/pricing/"

#: Pricing-page display name per Stability image-service model.
_STABILITY_PAGE_NAMES: Final[dict[str, str]] = {
    "stability.stable-image-remove-background-v1:0": "Stable Image Remove Background",
    "stability.stable-image-erase-object-v1:0": "Stable Image Erase Object",
    "stability.stable-image-control-structure-v1:0": "Stable Image Control Structure",
    "stability.stable-image-control-sketch-v1:0": "Stable Image Control Sketch",
    "stability.stable-image-style-guide-v1:0": "Stable Image Style Guide",
    "stability.stable-image-search-replace-v1:0": "Stable Image Search and Replace",
    "stability.stable-image-inpaint-v1:0": "Stable Image Inpaint",
    "stability.stable-image-search-recolor-v1:0": "Stable Image Search and Recolor",
    "stability.stable-style-transfer-v1:0": "Stable Image Style Transfer",
    "stability.stable-conservative-upscale-v1:0": "Stable Image Conservative Upscale",
    "stability.stable-creative-upscale-v1:0": "Stable Image Creative upscale",
    "stability.stable-fast-upscale-v1:0": "Stable Image Fast Upscale",
    "stability.stable-outpaint-v1:0": "Stable Image Outpaint",
}

#: Card links whose model belongs to the family DEFAULT_MODEL_PRICES prices.
_OPENAI_CARD_LINK: Final[re.Pattern[str]] = re.compile(
    r"(model-card-openai-gpt-(?!oss)[a-z0-9.-]+)\.html"
)

#: Divisor turning a card's per-1M-token rate into a per-token one.
_PER_MILLION: Final[Decimal] = Decimal(1_000_000)

#: The card note stating the unit; a change to it invalidates the division above.
_PER_MILLION_NOTE: Final[str] = "per 1 million tokens"

#: The card row naming the rate that applies in the model's own region.
_IN_REGION_ROW: Final[str] = "in-region"

#: The card row naming the rate the ``global.`` inference profile is billed at.
_GLOBAL_ROW: Final[str] = "global cris"

#: Caption fragment labelling the short-context table; an uncaptioned table is one.
_SHORT_CONTEXT_CAPTION: Final[str] = "short context"

#: Caption fragment labelling the long-context table, absent from most cards.
_LONG_CONTEXT_CAPTION: Final[str] = "long context"

#: How a card states the boundary between its two context tables, e.g. "(272K)".
_CONTEXT_WINDOW_SIZE: Final[re.Pattern[str]] = re.compile(
    r"\((\d+)\s*K\)", re.IGNORECASE
)

#: Multiplier turning a card's "K" context-window size into prompt tokens.
_THOUSAND: Final[int] = 1_000

#: Header of the pricing-page table holding the Stability rates.
_STABILITY_HEADING: Final[str] = "stability ai image services"

#: The unit that table states; a change to it invalidates a direct comparison.
_PER_GENERATION_NOTE: Final[str] = "price per generation"

#: Card column header fragment to dimension, longest-qualified fragment first.
_CARD_COLUMNS: Final[tuple[tuple[str, Dimension], ...]] = (
    ("cache write", Dimension.CACHE_WRITE_TOKENS),
    ("cache read", Dimension.CACHE_READ_TOKENS),
    ("input", Dimension.INPUT_TOKENS),
    ("output", Dimension.OUTPUT_TOKENS),
)

_TAG: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")
_SPACES: Final[re.Pattern[str]] = re.compile(r"\s+")
_TABLE: Final[re.Pattern[str]] = re.compile(r"<table.*?</table>", re.DOTALL)
_ROW: Final[re.Pattern[str]] = re.compile(r"<tr.*?</tr>", re.DOTALL)
_CELL: Final[re.Pattern[str]] = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.DOTALL)
_MONEY: Final[re.Pattern[str]] = re.compile(r"^\$([0-9]+(?:\.[0-9]+)?)$")
_H1: Final[re.Pattern[str]] = re.compile(r"<h1[\s>]")
_PRICING_SECTION: Final[re.Pattern[str]] = re.compile(
    r'<h2[^>]*id="[^"]*-pricing"[^>]*>.*?</h2>(.*?)(?=<h2|\Z)', re.DOTALL
)
#: A bold caption labelling the table that follows it, or a table itself.
_CAPTION_OR_TABLE: Final[re.Pattern[str]] = re.compile(
    r"<p[^>]*>\s*<b>(.*?)</b>\s*</p>|<table.*?</table>", re.DOTALL
)


class UnreadableSourceError(Exception):
    """The source was served but its pricing could not be read with confidence.

    Raised instead of guessing whenever the structure the parser depends on is
    absent or ambiguous, so a redesigned page reports as unreachable rather than
    as a drift or a vanished rate.
    """


class PriceSourceWarning(UserWarning):
    """A vendor source said something the run reports but must not fail on."""


class Outcome(StrEnum):
    """What comparing one table entry against its source established.

    Declared worst first: the report is grouped in this order, so the one
    outcome that needs a person shows above the many that do not.
    """

    DRIFT = "DRIFT"
    VANISHED = "VANISHED"
    NEW = "NEW"
    UNREACHABLE = "UNREACHABLE"
    MATCH = "MATCH"


@dataclass(frozen=True, slots=True)
class Finding:
    """One outcome, for one model, in enough detail to act on."""

    outcome: Outcome
    model_id: str
    detail: str


@dataclass(frozen=True, slots=True)
class SourceReading:
    """What a source had to say about one model, in exactly one of three states.

    ``rates`` set means the source published them. ``rates`` and ``problem``
    both unset means the source was read and no longer lists the model -- a
    vanished rate, never removed and never a failure. ``problem`` set means the
    source could not be read or understood at all.
    """

    url: str
    rates: dict[Dimension, Decimal] | None = None
    problem: str | None = None


@dataclass(frozen=True, slots=True)
class ThresholdReading:
    """What a card had to say about the prompt size its rates split at.

    The same three states as :class:`SourceReading`: ``tokens`` set means the
    card states a context window, both unset means it prices a single tier, and
    ``problem`` set means the card could not be read.
    """

    url: str
    tokens: int | None = None
    problem: str | None = None


@dataclass(frozen=True, slots=True)
class CardReadings:
    """Everything one model card publishes, as separately classifiable readings.

    A card prices up to two context tiers, each with an In-Region and a Global
    row, and states the boundary between them. Each is compared against its own
    table, so one withdrawn rate never reads as another's.
    """

    in_region: SourceReading
    cross_region: SourceReading
    long_in_region: SourceReading
    long_cross_region: SourceReading
    threshold: ThresholdReading


def _text(fragment: str) -> str:
    """Return *fragment*'s visible text, unescaped and whitespace-collapsed."""
    return _SPACES.sub(" ", unescape(_TAG.sub(" ", fragment))).strip()


def _rows(table: str) -> list[list[str]]:
    """Return *table*'s rows as lists of cell texts."""
    return [[_text(cell) for cell in _CELL.findall(row)] for row in _ROW.findall(table)]


def _money(cell: str) -> Decimal | None:
    """Return the USD amount *cell* states, or None when it states no rate.

    A card writes an em dash where a rate does not apply, which is a published
    absence rather than an unreadable cell.
    """
    match = _MONEY.match(cell)
    return Decimal(match.group(1)) if match else None


def _card_dimension(header: str) -> Dimension | None:
    """Return the dimension a card column header names, or None for the others."""
    lowered = header.casefold()
    return next(
        (dimension for fragment, dimension in _CARD_COLUMNS if fragment in lowered),
        None,
    )


def _captioned_tables(section: str) -> list[tuple[str | None, str]]:
    """Return the section's pricing tables, each with the caption labelling it.

    A card carries one unlabelled table, or several captioned ones: a short and
    a long context window, or a separate AWS GovCloud block. Pairing each table
    with its own caption is what keeps one tier's rate from being read as
    another's.
    """
    tables: list[tuple[str | None, str]] = []
    caption: str | None = None
    for match in _CAPTION_OR_TABLE.finditer(section):
        if match.group(0).startswith("<table"):
            tables.append((caption, match.group(0)))
            caption = None
        else:
            caption = _text(match.group(1))
    return tables


def _short_context_table(section: str) -> str:
    """Return the card's commercial short-context pricing table.

    Only an absent caption or one naming the short context window prices what
    ``DEFAULT_MODEL_PRICES`` registers, and anything else is a different rate
    that must not be guessed at.

    Raises:
        UnreadableSourceError: If the section holds no single such table.
    """
    candidates = [
        table
        for caption, table in _captioned_tables(section)
        if caption is None or _SHORT_CONTEXT_CAPTION in caption.casefold()
    ]
    if len(candidates) != 1:
        msg = (
            f"expected exactly one uncaptioned or short-context pricing table, "
            f"found {len(candidates)}"
        )
        raise UnreadableSourceError(msg)
    return candidates[0]


def _long_context_table(section: str) -> str | None:
    """Return the card's long-context pricing table, or None when it has none.

    Most cards price a single context window and carry no such table at all,
    which is a published absence rather than a fault. Two would mean the
    caption stopped naming one table, which must not be guessed at either.

    Raises:
        UnreadableSourceError: If the section holds more than one.
    """
    candidates = [
        table
        for caption, table in _captioned_tables(section)
        if caption is not None and _LONG_CONTEXT_CAPTION in caption.casefold()
    ]
    if len(candidates) > 1:
        msg = (
            f"expected at most one long-context pricing table, found {len(candidates)}"
        )
        raise UnreadableSourceError(msg)
    return candidates[0] if candidates else None


def _pricing_section(page: str) -> str:
    """Return the card's Pricing section, checked to still quote per-1M rates.

    Raises:
        UnreadableSourceError: If the section or its unit note is absent.
    """
    section = _PRICING_SECTION.search(page)
    if section is None:
        msg = "the card has no Pricing section"
        raise UnreadableSourceError(msg)
    body = section.group(1)
    if _PER_MILLION_NOTE not in _text(body).casefold():
        msg = f"the Pricing section no longer states rates {_PER_MILLION_NOTE!r}"
        raise UnreadableSourceError(msg)
    return body


def _pricing_rows(page: str, context: ContextLength = "") -> list[list[str]] | None:
    """Return the rows of one of a card's commercial pricing tables.

    Every inference option is a row of one table, so the In-Region and the
    Global rate of a context tier are both read from it -- and both are
    therefore protected from the other tier's block and from GovCloud by the
    same table selection.

    Args:
        page: The model card's HTML.
        context: "" for the short-context table, "long" for the long-context one.

    Returns:
        The header row followed by one row per inference option, or None when
        *context* is "long" and the card prices a single context window.

    Raises:
        UnreadableSourceError: If the pricing section, its unit note or the
            wanted table cannot be identified.
    """
    body = _pricing_section(page)
    table = _long_context_table(body) if context else _short_context_table(body)
    return _rows(table) if table is not None else None


def parse_context_window(page: str) -> int | None:
    """Return the prompt size at which a card leaves its short-context rate.

    A split card captions its first table with the window it prices, e.g.
    "Short Context Window (272K)". That figure is the boundary
    ``MODEL_LONG_CONTEXT_THRESHOLDS`` has to carry: registering a different one
    prices real calls from the wrong tier in whichever direction it errs.

    Args:
        page: The model card's HTML.

    Returns:
        The boundary in prompt tokens, or None when the card captions no
        context window -- the ordinary case of a card pricing a single tier.

    Raises:
        UnreadableSourceError: If the pricing section or its unit note is
            absent, or a short-context caption states no readable size.
    """
    for caption, _ in _captioned_tables(_pricing_section(page)):
        if caption is None or _SHORT_CONTEXT_CAPTION not in caption.casefold():
            continue
        if (size := _CONTEXT_WINDOW_SIZE.search(caption)) is None:
            msg = f"the short-context caption {caption!r} states no window size"
            raise UnreadableSourceError(msg)
        return int(size.group(1)) * _THOUSAND
    return None


def _row_rates(rows: list[list[str]], label: str) -> dict[Dimension, Decimal] | None:
    """Return the per-token rates the row labelled *label* states.

    Args:
        rows: A pricing table, header row first.
        label: The casefolded inference option naming the wanted row.

    Returns:
        The rate per token, per dimension, or None when the table has no such
        row -- which is a published absence for every option but In-Region.
    """
    labelled = next(
        (row for row in rows[1:] if row and row[0].casefold() == label), None
    )
    if labelled is None:
        return None
    return {
        dimension: amount / _PER_MILLION
        for header, cell in zip(rows[0], labelled, strict=False)
        if (dimension := _card_dimension(header)) is not None
        and (amount := _money(cell)) is not None
    }


def parse_model_card(
    page: str, context: ContextLength = ""
) -> dict[Dimension, Decimal] | None:
    """Return the per-token In-Region rates a Bedrock model card publishes.

    Args:
        page: The model card's HTML.
        context: Which context tier's table to read -- "" for the short-context
            rates ``DEFAULT_MODEL_PRICES`` carries, "long" for the
            ``DEFAULT_MODEL_LONG_CONTEXT_PRICES`` ones.

    Returns:
        The rate per token, per dimension, for the model's own region, or None
        when *context* is "long" and the card prices a single context window.

    Raises:
        UnreadableSourceError: If the pricing section, its unit note, the
            wanted table or its In-Region row cannot be identified.
    """
    rows = _pricing_rows(page, context)
    if rows is None:
        return None
    rates = _row_rates(rows, _IN_REGION_ROW)
    if rates is None:
        msg = "the pricing table has no In-Region row"
        raise UnreadableSourceError(msg)
    if not rates:
        msg = "the In-Region row states no rate"
        raise UnreadableSourceError(msg)
    return rates


def parse_model_card_global(
    page: str, context: ContextLength = ""
) -> dict[Dimension, Decimal] | None:
    """Return the per-token Global cross-Region rates a model card publishes.

    Args:
        page: The model card's HTML.
        context: Which context tier's table to read, as in
            :func:`parse_model_card`.

    Returns:
        The rate per token, per dimension, for the ``global.`` inference
        profile, or None when the card quotes no Global row in that tier --
        the ordinary case, since most models are priced In-Region only.

    Raises:
        UnreadableSourceError: If the pricing section, its unit note or the
            wanted table cannot be identified, or if a Global row is present
            but prices nothing, which reads as changed columns rather than as
            a withdrawn rate.
    """
    rows = _pricing_rows(page, context)
    if rows is None:
        return None
    rates = _row_rates(rows, _GLOBAL_ROW)
    if rates is not None and not rates:
        msg = "the Global CRIS row states no rate"
        raise UnreadableSourceError(msg)
    return rates


def parse_stability_prices(page: str) -> dict[str, Decimal]:
    """Return the per-generation rates the Bedrock pricing page publishes.

    Args:
        page: The AWS Bedrock pricing page's HTML.

    Returns:
        The rate per generation, keyed by casefolded display name.

    Raises:
        UnreadableSourceError: If the Stability table is absent, no longer
            prices per generation, or states no rate.
    """
    for table in _TABLE.findall(page):
        rows = _rows(table)
        heading = " ".join(rows[0]).casefold() if rows else ""
        if _STABILITY_HEADING not in heading:
            continue
        if _PER_GENERATION_NOTE not in heading:
            msg = "the Stability table no longer prices per generation"
            raise UnreadableSourceError(msg)
        prices = {
            row[0].casefold(): amount
            for row in rows[1:]
            if len(row) > 1 and (amount := _money(row[1])) is not None
        }
        if not prices:
            msg = "the Stability table states no rate"
            raise UnreadableSourceError(msg)
        return prices
    msg = f"no table headed {_STABILITY_HEADING!r}"
    raise UnreadableSourceError(msg)


def classify(
    model_id: str, expected: Mapping[Dimension, str], reading: SourceReading
) -> list[Finding]:
    """Compare one table entry against what its source published.

    An unreadable source and a rate the source no longer publishes are reported
    and never failed on; only a rate the source states differently is a drift.
    A rate published for a dimension the table does not carry is reported as
    new, since the table under-reporting a dimension is the vendor's change
    rather than a regression here.

    Args:
        model_id: The model the entry prices.
        expected: The entry's rates, as ``DEFAULT_MODEL_PRICES`` states them.
        reading: What the source had to say about *model_id*.

    Returns:
        One finding per dimension compared, plus one per dimension only the
        source carries; a single finding when the source was unusable or the
        model is no longer listed at all.
    """
    if reading.problem is not None:
        detail = f"{reading.url}: {reading.problem}"
        return [Finding(Outcome.UNREACHABLE, model_id, detail)]
    if reading.rates is None:
        detail = (
            f"{reading.url} no longer publishes a rate for this model. "
            f"Keep the entry: usage recorded against it still has to be priced."
        )
        return [Finding(Outcome.VANISHED, model_id, detail)]
    findings = [
        _compare(model_id, dimension, Decimal(rate), reading)
        for dimension, rate in expected.items()
    ]
    findings.extend(
        Finding(
            Outcome.NEW,
            model_id,
            f"{dimension.value}: {reading.url} publishes {rate}, the table has none",
        )
        for dimension, rate in reading.rates.items()
        if dimension not in expected
    )
    return findings


def _compare(
    model_id: str, dimension: Dimension, expected: Decimal, reading: SourceReading
) -> Finding:
    """Return the finding comparing one dimension's rate against the source."""
    published = (reading.rates or {}).get(dimension)
    if published is None:
        detail = (
            f"{dimension.value}: {reading.url} no longer publishes this rate. "
            f"Keep the entry: usage recorded against it still has to be priced."
        )
        return Finding(Outcome.VANISHED, model_id, detail)
    if published != expected:
        detail = (
            f"{dimension.value}: table has {expected:f}, {reading.url} "
            f"publishes {published:f}"
        )
        return Finding(Outcome.DRIFT, model_id, detail)
    return Finding(Outcome.MATCH, model_id, f"{dimension.value}: {expected:f}")


def _qualified_key(model_id: str, label: str) -> str:
    """Return how a model's entry in a qualified table is named in the report.

    Four tables price the same model, so the report has to say which rate a
    finding is about for it to be actionable.
    """
    return f"{model_id} ({label})"


def _global_key(model_id: str) -> str:
    """Return how a model's Global cross-Region entry is named in the report."""
    return _qualified_key(model_id, "Global")


def classify_qualified(
    model_id: str,
    table: Mapping[str, Mapping[Dimension, str]],
    label: str,
    constant: str,
    reading: SourceReading,
) -> list[Finding]:
    """Compare one model's entry in an optional rate table against its card.

    The Global, long-context and long-context Global tables all name only the
    models AWS publishes that rate for, so all three share one rule: a model
    absent from both the table and the card reports nothing (that is the state
    of most models, and reporting it would bury the findings that matter), a
    card that stops quoting a rate the table carries is a vanished rate like
    any other, and a rate only the card carries is new.

    Args:
        model_id: The model the entry prices.
        table: The table holding the qualified rates.
        label: Which rate this is, as the report names it.
        constant: The table's name in the source, so a finding says what to edit.
        reading: What the model's card had to say about this rate.

    Returns:
        The findings for this rate, empty when neither side carries one.
    """
    expected = table.get(model_id)
    if expected is not None:
        return classify(_qualified_key(model_id, label), expected, reading)
    if reading.rates is None:
        return []
    return [
        Finding(
            Outcome.NEW,
            _qualified_key(model_id, label),
            f"{dimension.value}: {reading.url} publishes {rate}, "
            f"{constant} has no entry",
        )
        for dimension, rate in reading.rates.items()
    ]


def classify_global(model_id: str, reading: SourceReading) -> list[Finding]:
    """Compare one model's Global cross-Region entry against what its card said.

    A card that stops quoting a Global rate the table carries is a vanished
    rate like any other -- the entry is kept, since the ``global.`` inference
    profile still serves calls that have to be priced.

    Args:
        model_id: The model the entry prices.
        reading: What the model's card had to say about its Global rate.

    Returns:
        The findings for *model_id*'s Global rate, empty when neither the table
        nor the card carries one.
    """
    return classify_qualified(
        model_id,
        DEFAULT_MODEL_GLOBAL_PRICES,
        "Global",
        "DEFAULT_MODEL_GLOBAL_PRICES",
        reading,
    )


def classify_threshold(model_id: str, reading: ThresholdReading) -> list[Finding]:
    """Compare a model's registered long-context boundary against its card.

    The boundary decides which of two published tiers prices a real call, so a
    stale one mis-bills exactly as a stale rate does -- and more quietly, since
    both rates it selects between are correct. A card that stops splitting its
    rates is reported, never failed: the registered boundary keeps whatever
    calls it already priced.

    Args:
        model_id: The model the boundary applies to.
        reading: What the model's card had to say about its context window.

    Returns:
        One finding, or none when neither the card nor the table states a
        boundary -- the ordinary case of a model priced at a single tier.
    """
    key = _qualified_key(model_id, "context window")
    expected = MODEL_LONG_CONTEXT_THRESHOLDS.get(model_id)
    if reading.problem is not None:
        return [Finding(Outcome.UNREACHABLE, key, f"{reading.url}: {reading.problem}")]
    if reading.tokens is None:
        if expected is None:
            return []
        detail = (
            f"{reading.url} no longer states a context window. Keep the entry: "
            f"calls are still priced from whichever tier it selects."
        )
        return [Finding(Outcome.VANISHED, key, detail)]
    if expected is None:
        detail = (
            f"{reading.url} splits its rates at {reading.tokens} prompt tokens, "
            f"MODEL_LONG_CONTEXT_THRESHOLDS has no entry"
        )
        return [Finding(Outcome.NEW, key, detail)]
    if expected != reading.tokens:
        detail = (
            f"table has {expected} prompt tokens, {reading.url} states {reading.tokens}"
        )
        return [Finding(Outcome.DRIFT, key, detail)]
    return [Finding(Outcome.MATCH, key, f"{expected} prompt tokens")]


def _card_url(slug: str) -> str:
    """Return the user guide URL of the model card named *slug*."""
    return f"{_USER_GUIDE}{slug}.html"


def card_is_withdrawn(slug: str, page: str) -> bool:
    """Whether *page* is the stub the user guide serves for a card that is gone.

    ``docs.aws.amazon.com`` answers an unknown page with 200 and a near-empty
    document, so the HTTP status cannot tell a withdrawn card from a served
    one. A served card carries an ``<h1>`` and derives its section anchors from
    its own slug; the stub has neither. Both are required, because reading a
    served card as withdrawn would stop checking that model without failing.
    """
    return slug not in page and not _H1.search(page)


def _withdrawn_readings(url: str) -> CardReadings:
    """Return the readings of a card the user guide no longer serves."""
    return CardReadings(
        SourceReading(url),
        SourceReading(url),
        SourceReading(url),
        SourceReading(url),
        ThresholdReading(url),
    )


def _read_model_card(client: httpx.Client, slug: str) -> CardReadings:
    """Fetch a model card once, never raising on a source-side problem.

    Returns:
        Everything the card publishes. A card that cannot be read at all makes
        every reading unreachable, so a redesigned card never reads as a
        withdrawn Global or long-context rate.
    """
    url = _card_url(slug)
    try:
        response = client.get(url)
        if response.status_code == httpx.codes.NOT_FOUND:
            return _withdrawn_readings(url)
        response.raise_for_status()
        page = response.text
        if card_is_withdrawn(slug, page):
            return _withdrawn_readings(url)
        return CardReadings(
            SourceReading(url, rates=parse_model_card(page)),
            SourceReading(url, rates=parse_model_card_global(page)),
            SourceReading(url, rates=parse_model_card(page, "long")),
            SourceReading(url, rates=parse_model_card_global(page, "long")),
            ThresholdReading(url, tokens=parse_context_window(page)),
        )
    except (httpx.HTTPError, UnreadableSourceError) as exc:
        problem = f"{type(exc).__name__}: {exc}"
        reading = SourceReading(url, problem=problem)
        return CardReadings(
            reading, reading, reading, reading, ThresholdReading(url, problem=problem)
        )


def stability_readings(
    page: str | None, problem: str | None
) -> dict[str, SourceReading]:
    """Return one reading per Stability model from a single fetch of the page.

    A page whose Stability table cannot be located at all reports thirteen
    unreachable sources rather than thirteen vanished rates: the difference is
    what keeps a redesigned page from reading as a mass delisting.

    Args:
        page: The pricing page's HTML, or None when the fetch failed.
        problem: Why the fetch failed, or None when it did not.

    Returns:
        A reading per model in ``_STABILITY_PAGE_NAMES``.
    """
    if page is None:
        return {
            model_id: SourceReading(_BEDROCK_PRICING_PAGE, problem=problem)
            for model_id in _STABILITY_PAGE_NAMES
        }
    try:
        published = parse_stability_prices(page)
    except UnreadableSourceError as exc:
        return {
            model_id: SourceReading(_BEDROCK_PRICING_PAGE, problem=str(exc))
            for model_id in _STABILITY_PAGE_NAMES
        }
    return {
        model_id: (
            SourceReading(
                _BEDROCK_PRICING_PAGE, rates={Dimension.OUTPUT_IMAGES: amount}
            )
            if (amount := published.get(name.casefold())) is not None
            else SourceReading(_BEDROCK_PRICING_PAGE)
        )
        for model_id, name in _STABILITY_PAGE_NAMES.items()
    }


def unpriced_stability_rows(page: str) -> list[Finding]:
    """Return a finding per pricing-page row no table entry prices."""
    try:
        published = parse_stability_prices(page)
    except UnreadableSourceError:
        return []
    known = {name.casefold() for name in _STABILITY_PAGE_NAMES.values()}
    return [
        Finding(
            Outcome.NEW,
            name,
            f"{_BEDROCK_PRICING_PAGE} prices it at {amount:f} per generation, "
            f"DEFAULT_MODEL_PRICES has no entry",
        )
        for name, amount in sorted(published.items())
        if name not in known
    ]


def unpriced_openai_cards(index: str) -> list[Finding]:
    """Return a finding per GPT-5.x model card no table entry prices."""
    known = set(_MODEL_CARD_URLS.values())
    return [
        Finding(
            Outcome.NEW,
            slug,
            f"{_card_url(slug)} is a GPT-5.x model card, DEFAULT_MODEL_PRICES "
            f"has no entry",
        )
        for slug in sorted(set(_OPENAI_CARD_LINK.findall(index)) - known)
    ]


def format_report(findings: list[Finding]) -> str:
    """Render *findings* grouped by outcome, worst first."""
    lines = ["The hand-copied price tables vs. the sources they were copied from:", ""]
    for outcome in Outcome:
        selected = [finding for finding in findings if finding.outcome is outcome]
        if not selected:
            continue
        lines.append(f"{outcome.value} ({len(selected)}):")
        lines.extend(f"  {finding.model_id}: {finding.detail}" for finding in selected)
        lines.append("")
    return "\n".join(lines)


#: What to do about a drift, printed with the failure that reports one.
_FIX: Final[str] = (
    "FIX: the vendor changed a published rate. Update the matching entry in "
    "stdapi/models/pricing_overrides.py -- DEFAULT_MODEL_PRICES, or the table "
    "the parenthesis after the model names: '(Global)' is "
    "DEFAULT_MODEL_GLOBAL_PRICES, '(long context)' is "
    "DEFAULT_MODEL_LONG_CONTEXT_PRICES, '(long context, Global)' is "
    "DEFAULT_MODEL_GLOBAL_LONG_CONTEXT_PRICES and '(context window)' is "
    "MODEL_LONG_CONTEXT_THRESHOLDS -- to the "
    "value the source now publishes (a model card states per-1M-token rates: "
    "divide by 1e6), re-run this test, and note the change in the release "
    "entry. Never delete an entry whose rate merely stopped being published: "
    "usage recorded against that model still has to be priced."
)


def _collect(client: httpx.Client) -> list[Finding]:
    """Read every source once and classify the whole table against it."""
    findings: list[Finding] = []
    for model_id, slug in _MODEL_CARD_URLS.items():
        card = _read_model_card(client, slug)
        findings.extend(
            classify(model_id, DEFAULT_MODEL_PRICES[model_id], card.in_region)
        )
        findings.extend(classify_global(model_id, card.cross_region))
        findings.extend(
            classify_qualified(
                model_id,
                DEFAULT_MODEL_LONG_CONTEXT_PRICES,
                "long context",
                "DEFAULT_MODEL_LONG_CONTEXT_PRICES",
                card.long_in_region,
            )
        )
        findings.extend(
            classify_qualified(
                model_id,
                DEFAULT_MODEL_GLOBAL_LONG_CONTEXT_PRICES,
                "long context, Global",
                "DEFAULT_MODEL_GLOBAL_LONG_CONTEXT_PRICES",
                card.long_cross_region,
            )
        )
        findings.extend(classify_threshold(model_id, card.threshold))

    page: str | None = None
    problem: str | None = None
    try:
        response = client.get(_BEDROCK_PRICING_PAGE)
        response.raise_for_status()
        page = response.text
    except httpx.HTTPError as exc:
        problem = f"{type(exc).__name__}: {exc}"
    for model_id, reading in stability_readings(page, problem).items():
        findings.extend(classify(model_id, DEFAULT_MODEL_PRICES[model_id], reading))
    if page is not None:
        findings.extend(unpriced_stability_rows(page))

    try:
        response = client.get(_OPENAI_CARD_INDEX)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        findings.append(
            Finding(
                Outcome.UNREACHABLE, _OPENAI_CARD_INDEX, f"{type(exc).__name__}: {exc}"
            )
        )
    else:
        findings.extend(unpriced_openai_cards(response.text))
    return findings


@pytest.mark.drift
def test_default_model_prices_match_their_published_source() -> None:
    """Every hand-copied rate must still be the rate its vendor source publishes.

    The gateway reports these figures to the operator as cost, and nothing at
    runtime can contradict them, so a vendor-side change is invisible until a
    customer reads a wrong number. This lane is opt-in precisely so that a
    vendor's change never reddens an unrelated run, and fails hard inside it
    because a drift is worth a person's attention.

    A rate that vanished from its source, a model the source newly publishes and
    a source that could not be read are reported instead of failed: only the
    first is even about our table, and none of them is a defect here. A run that
    could compare nothing at all skips, so a green result never means "not
    checked".

    Ref: stdapi/models/pricing_overrides.py:DEFAULT_MODEL_PRICES
         stdapi/models/pricing_overrides.py:DEFAULT_MODEL_GLOBAL_PRICES
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-openai.html
         https://aws.amazon.com/bedrock/pricing/
    """
    priced = {*_MODEL_CARD_URLS, *_STABILITY_PAGE_NAMES}
    assert priced == set(DEFAULT_MODEL_PRICES), (
        "DEFAULT_MODEL_PRICES gained or lost an entry: give it a source above, "
        "or this detector silently stops covering it"
    )
    for name, table in (
        ("DEFAULT_MODEL_GLOBAL_PRICES", DEFAULT_MODEL_GLOBAL_PRICES),
        ("DEFAULT_MODEL_LONG_CONTEXT_PRICES", DEFAULT_MODEL_LONG_CONTEXT_PRICES),
        (
            "DEFAULT_MODEL_GLOBAL_LONG_CONTEXT_PRICES",
            DEFAULT_MODEL_GLOBAL_LONG_CONTEXT_PRICES,
        ),
        ("MODEL_LONG_CONTEXT_THRESHOLDS", MODEL_LONG_CONTEXT_THRESHOLDS),
    ):
        assert set(table) <= set(_MODEL_CARD_URLS), (
            f"{name} carries a model with no model card above, "
            f"or this detector silently stops covering it"
        )
    assert set(DEFAULT_MODEL_LONG_CONTEXT_PRICES) <= set(
        MODEL_LONG_CONTEXT_THRESHOLDS
    ), (
        "a long-context rate with no registered boundary is unreachable: the "
        "default 200K would select it for prompts the model still bills short"
    )

    with httpx.Client(
        timeout=_FETCH_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        findings = _collect(client)

    report = format_report(findings)
    print(report)  # noqa: T201 -- shown by pytest on failure, and with -s
    compared = [
        finding
        for finding in findings
        if finding.outcome in {Outcome.MATCH, Outcome.DRIFT}
    ]
    if not compared:
        pytest.skip(f"No source published a rate to compare against.\n{report}")
    reported = [
        finding
        for finding in findings
        if finding.outcome not in {Outcome.MATCH, Outcome.DRIFT}
    ]
    if reported:
        warnings.warn(format_report(reported), PriceSourceWarning, stacklevel=2)
    if any(finding.outcome is Outcome.DRIFT for finding in findings):
        pytest.fail(f"{report}\n{_FIX}")


#: The Price List service codes a Bedrock row arrives under, native or Marketplace.
_BEDROCK_SERVICE_CODES: Final[tuple[str, ...]] = (
    "AmazonBedrock",
    "AmazonBedrockService",
    "AmazonBedrockFoundationModels",
)

#: What to do when AWS starts publishing a rate one of the tables hand-copies.
_FIX_PUBLISHED: Final[str] = (
    "FIX: AWS now publishes Price List rows for a model the hand-copied tables "
    "price. _apply_default_prices guards per model, so those rows already "
    "discarded the whole table entry and every dimension AWS does not publish "
    "is unpriced right now. Check which dimensions the published rows cover: "
    "drop the entry when they cover all of them, and when they do not, treat "
    "the per-model guard as the defect it becomes and fix that first."
)


async def _published_bedrock_model_keys() -> dict[str, set[str]]:
    """Fetch the Bedrock rows AWS publishes today, keyed as the price catalog keys them.

    Drives ``_fetch_service_pricing``, the same ingestion the running gateway
    loads its catalog with, so a change to key normalization or to the
    Marketplace listing parser moves this check with it instead of past it.

    Returns:
        Price-catalog model key to the claims that produced its rows -- a
        usagetype for a native row, ``"<listing name>:<usagetype>"`` for a
        Marketplace one.

    Raises:
        BotoCoreError: When the Price List API is unreachable.
        ClientError: When the Price List API refuses the request.
    """
    endpoint = pricing.pricing_endpoint_region()
    claimed: dict[str, set[str]] = {}
    # type-ignore: the RegionName stub Literal lags EUSC/China (works live).
    async with AWSConnectionManager(("pricing", endpoint)):  # type: ignore[arg-type]
        client = get_client("pricing", endpoint)  # type: ignore[arg-type]
        for region in SETTINGS.aws_bedrock_regions:
            for service_code in _BEDROCK_SERVICE_CODES:
                rows, claims = await pricing._fetch_service_pricing(  # noqa: SLF001
                    client, service_code, str(region), []
                )
                for key in rows:
                    claimed.setdefault(key.model, set()).add(claims[key])
    return claimed


@pytest.mark.drift
async def test_the_price_list_publishes_no_rate_the_tables_hand_copy() -> None:
    """No model the hand-copied tables price may have gained a Price List row.

    The tables exist only because AWS publishes these rates nowhere the gateway
    can read them: the OpenAI hosted models are absent from the Price List
    outside GovCloud -- they are not AWS Marketplace listings either, no
    ``(Amazon Bedrock Edition)`` product names one -- and the Stability image
    services are on the pricing page alone. That absence is a vendor fact, and
    the day it changes the hand-copied entry stops being a fallback and starts
    being a second, unreconciled answer.

    Worse, it fails closed on the wrong side: ``_apply_default_prices`` guards
    per model, so a single published row -- one dimension, one region, either
    listing form -- discards the whole entry, and every dimension AWS did not
    publish goes unpriced with no test noticing. A run that fetched nothing
    fails rather than passing vacuously.

    Ref: stdapi/models/pricing_overrides.py:DEFAULT_MODEL_PRICES
         stdapi/pricing.py:_apply_default_prices
         stdapi/pricing.py:_ingest_marketplace_item
    """
    if pricing.pricing_endpoint_region() is None:
        pytest.skip("this partition has no AWS Price List API endpoint")
    try:
        claimed = await _published_bedrock_model_keys()
    except (BotoCoreError, ClientError) as exc:
        pytest.skip(f"the AWS Price List API is not reachable: {exc}")

    assert claimed, "the Price List published no Bedrock row at all -- nothing was read"
    published = [
        f"{model_id} (catalog key {key!r}): {', '.join(sorted(claimed[key]))}"
        for model_id in sorted({*DEFAULT_MODEL_PRICES, *DEFAULT_MODEL_GLOBAL_PRICES})
        if (key := pricing.resolve_model_key(model_id)) in claimed
    ]
    if published:
        pytest.fail(
            "The AWS Price List now publishes rows for hand-copied models:\n"
            + "\n".join(f"  {line}" for line in published)
            + f"\n{_FIX_PUBLISHED}"
        )


@pytest.fixture(scope="module")
def gpt_56_cyber_card() -> str:
    """The recorded Pricing section of the GPT-5.6 Cyber model card."""
    return (FIXTURES_DIR / "model_card_openai_gpt_56_cyber_pricing.html").read_text()


@pytest.fixture(scope="module")
def gpt_56_sol_card() -> str:
    """The recorded Pricing section of the GPT-5.6 Sol model card."""
    return (FIXTURES_DIR / "model_card_openai_gpt_56_sol_pricing.html").read_text()


@pytest.fixture(scope="module")
def gpt_54_card() -> str:
    """The recorded Pricing section of the GPT-5.4 model card."""
    return (FIXTURES_DIR / "model_card_openai_gpt_54_pricing.html").read_text()


@pytest.fixture(scope="module")
def daybreak_blue_card() -> str:
    """The recorded Pricing section of the Daybreak Blue GPT-5.6 Sol model card."""
    path = FIXTURES_DIR / "model_card_openai_gpt_daybreak_blue_56_sol_pricing.html"
    return path.read_text()


@pytest.fixture(scope="module")
def stability_table() -> str:
    """The recorded Stability AI Image Services table of the Bedrock pricing page."""
    return (FIXTURES_DIR / "bedrock_pricing_page_stability_table.html").read_text()


class TestModelCardParsing:
    """The model card parser reads the commercial short-context In-Region rates.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-cyber.html
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-daybreak-blue-56-sol.html
    """

    def test_single_table_card_yields_every_published_dimension(
        self, gpt_56_cyber_card: str
    ) -> None:
        """A card with one pricing table yields its four per-1M rates, per token."""
        assert parse_model_card(gpt_56_cyber_card) == {
            Dimension.INPUT_TOKENS: Decimal("0.00001375"),
            Dimension.CACHE_WRITE_TOKENS: Decimal("0.0000171875"),
            Dimension.CACHE_READ_TOKENS: Decimal("0.000001375"),
            Dimension.OUTPUT_TOKENS: Decimal("0.0000825"),
        }

    def test_two_table_card_takes_the_short_context_window(
        self, daybreak_blue_card: str
    ) -> None:
        """The long-context table must not be mistaken for the registered rate.

        ``DEFAULT_MODEL_PRICES`` has no context axis and carries the 272K
        short-context tier, so a parser taking the last table would report a
        drift on every run against a rate the table never claimed.
        """
        rates = parse_model_card(daybreak_blue_card)
        assert rates is not None
        assert rates[Dimension.INPUT_TOKENS] == Decimal("0.0000055")
        assert rates[Dimension.OUTPUT_TOKENS] == Decimal("0.000033")

    def test_the_global_row_is_read_from_the_short_context_table(
        self, gpt_56_sol_card: str
    ) -> None:
        """A Global rate is the one in the table In-Region was taken from.

        Both context windows carry a Global CRIS row, so a parser reaching for
        the last matching row anywhere in the section would compare the 1M-tier
        rate against a table that only ever held the 272K one -- permanent false
        drift, on four dimensions, on every run.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html
        """
        assert parse_model_card_global(gpt_56_sol_card) == {
            Dimension.INPUT_TOKENS: Decimal("0.000004"),
            Dimension.CACHE_WRITE_TOKENS: Decimal("0.000005"),
            Dimension.CACHE_READ_TOKENS: Decimal("0.0000004"),
            Dimension.OUTPUT_TOKENS: Decimal("0.00002"),
        }

    def test_a_card_without_a_global_row_publishes_no_global_rate(
        self, gpt_56_cyber_card: str
    ) -> None:
        """Most models are priced In-Region only, which is an absence not a fault."""
        assert parse_model_card_global(gpt_56_cyber_card) is None

    def test_the_govcloud_block_is_neither_the_commercial_nor_a_global_rate(
        self, gpt_54_card: str
    ) -> None:
        """A card's third table shape must not leak into either rate.

        GPT-5.4 prices an uncaptioned commercial table and a captioned AWS
        GovCloud one; the GovCloud block is a different partition's rate, and
        its In-Region row is not a Global rate either.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-54.html
        """
        rates = parse_model_card(gpt_54_card)
        assert rates is not None
        assert rates[Dimension.INPUT_TOKENS] == Decimal("0.00000275")
        assert rates[Dimension.OUTPUT_TOKENS] == Decimal("0.0000165")
        assert parse_model_card_global(gpt_54_card) is None

    def test_a_global_row_pricing_nothing_is_unreadable(
        self, gpt_56_sol_card: str
    ) -> None:
        """A Global row whose every cell stopped being money reads as changed columns.

        Guessing here would report four dimensions as vanished on a card that
        merely renamed its headers, so the parser refuses instead.
        """
        card = gpt_56_sol_card.replace(
            '<tr><td tabindex="-1">Global CRIS</td><td tabindex="-1">$4.00</td>'
            '<td tabindex="-1">$5.00</td><td tabindex="-1">$0.40</td>'
            '<td tabindex="-1">$20.00</td></tr>',
            '<tr><td tabindex="-1">Global CRIS</td><td tabindex="-1">n/a</td>'
            '<td tabindex="-1">n/a</td><td tabindex="-1">n/a</td>'
            '<td tabindex="-1">n/a</td></tr>',
        )
        with pytest.raises(UnreadableSourceError, match="Global CRIS row states"):
            parse_model_card_global(card)

    def test_an_ambiguous_section_is_unreadable_for_the_global_rate_too(
        self, gpt_56_sol_card: str
    ) -> None:
        """The Global parser inherits the In-Region parser's table strictness."""
        card = gpt_56_sol_card.replace("Long Context Window (1M)", "Short Context")
        with pytest.raises(UnreadableSourceError, match="exactly one"):
            parse_model_card_global(card)

    def test_an_em_dash_is_a_published_absence_not_a_rate(
        self, gpt_56_cyber_card: str
    ) -> None:
        """A dimension a card prices with an em dash is absent, not unreadable."""
        card = gpt_56_cyber_card.replace(">$17.1875<", ">—<")
        rates = parse_model_card(card)
        assert rates is not None
        assert Dimension.CACHE_WRITE_TOKENS not in rates

    def test_an_ambiguous_pricing_section_is_unreadable(
        self, daybreak_blue_card: str
    ) -> None:
        """Two candidate tables must raise rather than pick one."""
        card = daybreak_blue_card.replace("Long Context Window (1M)", "Short Context")
        with pytest.raises(UnreadableSourceError, match="exactly one"):
            parse_model_card(card)

    def test_a_changed_unit_note_is_unreadable(self, gpt_56_cyber_card: str) -> None:
        """Rates stated in another unit must not be divided by a million."""
        card = gpt_56_cyber_card.replace(_PER_MILLION_NOTE, "per 1 thousand tokens")
        with pytest.raises(UnreadableSourceError, match="no longer states"):
            parse_model_card(card)

    def test_a_page_without_a_pricing_section_is_unreadable(self) -> None:
        """A redesigned card reports as unreadable, never as a vanished rate."""
        with pytest.raises(UnreadableSourceError, match="no Pricing section"):
            parse_model_card("<html><body><h1>GPT-5.6 Cyber</h1></body></html>")

    def test_the_user_guide_soft_404_reads_as_a_withdrawn_card(self) -> None:
        """The 200-with-a-stub answer for an unknown page means the card is gone.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-cyber.html
        """
        stub = "<html><head><title>Amazon Bedrock</title></head><body></body></html>"
        assert card_is_withdrawn("model-card-openai-gpt-56-cyber", stub)


class TestLongContextParsing:
    """The parser reads the second context tier, and the boundary between them.

    A card that splits its rates publishes two tables and names the window each
    prices. Both halves are needed: the rates without the boundary would be
    selected for the wrong calls, and the boundary without the rates would
    select a tier that is not there.

    Ref: stdapi/models/pricing_overrides.py:DEFAULT_MODEL_LONG_CONTEXT_PRICES
         stdapi/models/pricing_overrides.py:MODEL_LONG_CONTEXT_THRESHOLDS
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html
    """

    def test_the_long_context_table_yields_the_1m_tier_rates(
        self, gpt_56_sol_card: str
    ) -> None:
        """The long reading is the 1M table's In-Region row, not the 272K one."""
        assert parse_model_card(gpt_56_sol_card, "long") == {
            Dimension.INPUT_TOKENS: Decimal("0.0000088"),
            Dimension.CACHE_WRITE_TOKENS: Decimal("0.000011"),
            Dimension.CACHE_READ_TOKENS: Decimal("0.00000088"),
            Dimension.OUTPUT_TOKENS: Decimal("0.000033"),
        }

    def test_the_long_context_global_row_comes_from_the_same_table(
        self, gpt_56_sol_card: str
    ) -> None:
        """The 1M Global rate is the one beside the 1M In-Region rate."""
        assert parse_model_card_global(gpt_56_sol_card, "long") == {
            Dimension.INPUT_TOKENS: Decimal("0.000008"),
            Dimension.CACHE_WRITE_TOKENS: Decimal("0.00001"),
            Dimension.CACHE_READ_TOKENS: Decimal("0.0000008"),
            Dimension.OUTPUT_TOKENS: Decimal("0.00003"),
        }

    def test_a_single_tier_card_publishes_no_long_context_rate(
        self, gpt_56_cyber_card: str
    ) -> None:
        """One context window is an absence, not a fault: most cards have one."""
        assert parse_model_card(gpt_56_cyber_card, "long") is None
        assert parse_model_card_global(gpt_56_cyber_card, "long") is None

    def test_the_govcloud_block_is_not_mistaken_for_a_long_context_table(
        self, gpt_54_card: str
    ) -> None:
        """A captioned table that is not the long-context one yields no long rate."""
        assert parse_model_card(gpt_54_card, "long") is None

    def test_the_boundary_is_read_from_the_short_context_caption(
        self, gpt_56_sol_card: str
    ) -> None:
        """The short-context caption states where the long-context rate starts."""
        assert parse_context_window(gpt_56_sol_card) == 272_000

    def test_a_single_tier_card_states_a_window_when_it_captions_one(
        self, gpt_56_cyber_card: str
    ) -> None:
        """A card can name its window without pricing a second tier.

        Cyber's window is 272K and it publishes no 1M rate, so the boundary is
        still the point below which its one rate applies -- registering it is
        what keeps a 250K prompt from being recorded as long-context.
        """
        assert parse_context_window(gpt_56_cyber_card) == 272_000
        assert parse_model_card(gpt_56_cyber_card, "long") is None

    def test_an_uncaptioned_card_states_no_boundary(self, gpt_54_card: str) -> None:
        """A card pricing one unlabelled table leaves the model on the default."""
        assert parse_context_window(gpt_54_card) is None

    def test_a_caption_without_a_size_is_unreadable(self, gpt_56_sol_card: str) -> None:
        """A reworded caption must raise rather than yield a guessed boundary."""
        card = gpt_56_sol_card.replace(
            "Short Context Window (272K)", "Short Context Window"
        )
        with pytest.raises(UnreadableSourceError, match="no window size"):
            parse_context_window(card)

    def test_two_long_context_tables_are_unreadable(self, gpt_56_sol_card: str) -> None:
        """A caption that stops naming one table must not have one picked for it."""
        card = gpt_56_sol_card.replace(
            "Short Context Window (272K)", "Long Context Window (1M)"
        )
        with pytest.raises(UnreadableSourceError, match="at most one"):
            parse_model_card(card, "long")

    def test_a_changed_boundary_is_reported_as_drift(self) -> None:
        """A card moving its window must fail, naming both figures.

        Silent here is the worst case: both rates it selects between stay
        correct, so every cost the gateway reports looks plausible while half
        of them come from the wrong tier.
        """
        url = _card_url("model-card-openai-gpt-56-sol")
        reading = ThresholdReading(url, tokens=400_000)
        findings = classify_threshold("openai.gpt-5.6-sol", reading)
        assert [finding.outcome for finding in findings] == [Outcome.DRIFT]
        assert findings[0].model_id == "openai.gpt-5.6-sol (context window)"
        assert findings[0].detail == (
            f"table has 272000 prompt tokens, {url} states 400000"
        )

    def test_a_withdrawn_boundary_keeps_its_entry_without_failing(self) -> None:
        """A card that stops splitting its rates keeps the registered boundary."""
        reading = ThresholdReading(_card_url("model-card-openai-gpt-56-sol"))
        findings = classify_threshold("openai.gpt-5.6-sol", reading)
        assert [finding.outcome for finding in findings] == [Outcome.VANISHED]
        assert "Keep the entry" in findings[0].detail

    def test_a_model_with_no_boundary_on_either_side_reports_nothing(self) -> None:
        """A single-tier model with no entry is silent, like an In-Region-only one."""
        reading = ThresholdReading(_card_url("model-card-openai-gpt-54"))
        assert classify_threshold("openai.gpt-5.4", reading) == []

    def test_a_newly_split_card_is_reported_not_failed(self) -> None:
        """A card that gains a context window is actionable, not our regression."""
        url = _card_url("model-card-openai-gpt-54")
        findings = classify_threshold(
            "openai.gpt-5.4", ThresholdReading(url, tokens=200_000)
        )
        assert [finding.outcome for finding in findings] == [Outcome.NEW]
        assert "MODEL_LONG_CONTEXT_THRESHOLDS has no entry" in findings[0].detail

    def test_a_served_card_is_never_read_as_withdrawn(
        self, gpt_56_cyber_card: str
    ) -> None:
        """A card carrying its own slug is served, however else it changed."""
        assert not card_is_withdrawn(
            "model-card-openai-gpt-56-cyber", gpt_56_cyber_card
        )

    def test_a_redesigned_card_without_its_slug_is_not_withdrawn(self) -> None:
        """An ``<h1>`` alone keeps a restructured card under comparison.

        Reading a served card as withdrawn would stop checking that model
        silently, since a withdrawn rate never fails.
        """
        page = "<html><body><h1>GPT-5.6 Cyber</h1><p>Prices moved.</p></body></html>"
        assert not card_is_withdrawn("model-card-openai-gpt-56-cyber", page)


class TestStabilityPageParsing:
    """The pricing page parser reads the per-generation image service rates.

    Ref: https://aws.amazon.com/bedrock/pricing/
    """

    def test_every_listed_image_service_is_read(self, stability_table: str) -> None:
        """The table's thirteen rows each yield a per-generation rate."""
        prices = parse_stability_prices(stability_table)
        assert len(prices) == 13
        assert prices["stable image erase object"] == Decimal("0.07")
        assert prices["stable image creative upscale"] == Decimal("0.60")

    def test_a_missing_table_is_unreadable(self) -> None:
        """A page without the table reports unreachable, not thirteen delistings.

        Reading a redesigned page as a mass delisting is the false alarm that
        gets a detector switched off, so the two are told apart here.
        """
        readings = stability_readings("<html><body>No prices here.</body></html>", None)
        assert {reading.problem is not None for reading in readings.values()} == {True}

    def test_a_changed_unit_is_unreadable(self, stability_table: str) -> None:
        """A per-image or per-step column must not be compared per generation."""
        page = stability_table.replace("Price per generation for each model", "Price")
        with pytest.raises(UnreadableSourceError, match="per generation"):
            parse_stability_prices(page)


class TestGpt56Detection:
    """The detector's two directions, proved on the GPT-5.6 family.

    The two newest entries were copied from these cards, and the family is the
    one that actually drifted in production, so it is the worked example: the
    same classifier is shown reporting a match against the shipped table and a
    drift against a deliberately wrong expected value.

    Ref: stdapi/models/pricing_overrides.py:DEFAULT_MODEL_PRICES
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-cyber.html
    """

    @staticmethod
    def _outcomes(findings: list[Finding]) -> set[Outcome]:
        """Return the distinct outcomes *findings* reports."""
        return {finding.outcome for finding in findings}

    @pytest.mark.parametrize(
        ("model_id", "fixture_name"),
        [
            ("openai.gpt-5.6-cyber", "gpt_56_cyber_card"),
            ("openai.gpt-daybreak-blue-5.6-sol", "daybreak_blue_card"),
        ],
    )
    def test_the_shipped_table_matches_its_card(
        self, request: pytest.FixtureRequest, model_id: str, fixture_name: str
    ) -> None:
        """Every rate the entry carries must be the rate the card publishes."""
        card: str = request.getfixturevalue(fixture_name)
        reading = SourceReading(_card_url("x"), rates=parse_model_card(card))
        findings = classify(model_id, DEFAULT_MODEL_PRICES[model_id], reading)
        assert findings, "nothing was compared"
        assert self._outcomes(findings) == {Outcome.MATCH}

    def test_a_wrong_expected_value_is_reported_as_drift(
        self, gpt_56_cyber_card: str
    ) -> None:
        """A deliberately wrong output rate must be reported, with both values.

        This is the failure the detector exists for -- the shape GPT-5.6 Luna's
        5x error had -- so it is constructed here rather than waited for.
        """
        drifted = {
            **DEFAULT_MODEL_PRICES["openai.gpt-5.6-cyber"],
            Dimension.OUTPUT_TOKENS: "0.0004125",  # 5x the published rate
        }
        reading = SourceReading(
            _card_url("x"), rates=parse_model_card(gpt_56_cyber_card)
        )
        findings = classify("openai.gpt-5.6-cyber", drifted, reading)

        drift = [f for f in findings if f.outcome is Outcome.DRIFT]
        assert len(drift) == 1
        assert "0.0004125" in drift[0].detail
        assert "0.0000825" in drift[0].detail
        assert self._outcomes(findings) == {Outcome.MATCH, Outcome.DRIFT}

    def test_a_delisted_model_keeps_its_entry_without_failing(self) -> None:
        """A card that 404s reports vanished, and says to keep the entry."""
        model_id = "openai.gpt-5.6-cyber"
        reading = SourceReading(_card_url("model-card-openai-gpt-56-cyber"))
        findings = classify(model_id, DEFAULT_MODEL_PRICES[model_id], reading)
        assert self._outcomes(findings) == {Outcome.VANISHED}
        assert "Keep the entry" in findings[0].detail

    def test_a_withdrawn_rate_is_vanished_rather_than_drift(
        self, gpt_56_cyber_card: str
    ) -> None:
        """A card that stops pricing one dimension keeps that rate in the table."""
        card = gpt_56_cyber_card.replace(">$17.1875<", ">—<")
        reading = SourceReading(_card_url("x"), rates=parse_model_card(card))
        findings = classify(
            "openai.gpt-5.6-cyber",
            DEFAULT_MODEL_PRICES["openai.gpt-5.6-cyber"],
            reading,
        )
        vanished = [f for f in findings if f.outcome is Outcome.VANISHED]
        assert [f.model_id for f in vanished] == ["openai.gpt-5.6-cyber"]
        assert Dimension.CACHE_WRITE_TOKENS.value in vanished[0].detail
        assert Outcome.DRIFT not in self._outcomes(findings)

    def test_an_unreachable_source_is_never_a_drift(self) -> None:
        """A source that could not be fetched reports unreachable and nothing else."""
        model_id = "openai.gpt-5.6-cyber"
        reading = SourceReading(_card_url("x"), problem="ConnectTimeout")
        findings = classify(model_id, DEFAULT_MODEL_PRICES[model_id], reading)
        assert self._outcomes(findings) == {Outcome.UNREACHABLE}

    def test_a_newly_published_dimension_is_reported_not_failed(self) -> None:
        """A rate the card gained is actionable, but it is not our regression."""
        reading = SourceReading(
            _card_url("x"),
            rates={
                Dimension.INPUT_TOKENS: Decimal("0.00001375"),
                Dimension.INPUT_IMAGES: Decimal("0.002"),
            },
        )
        findings = classify(
            "openai.gpt-5.6-cyber", {Dimension.INPUT_TOKENS: "0.00001375"}, reading
        )
        assert self._outcomes(findings) == {Outcome.MATCH, Outcome.NEW}


class TestGlobalDetection:
    """The same four outcomes, proved on the Global cross-Region table.

    ``DEFAULT_MODEL_GLOBAL_PRICES`` is hand-copied from the same cards and has
    exactly the staleness the In-Region table has, so it is held to the same
    standard: the shipped entry is shown matching its card, and a deliberately
    wrong value is shown reported as a drift naming both values and the source.

    Ref: stdapi/models/pricing_overrides.py:DEFAULT_MODEL_GLOBAL_PRICES
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html
    """

    #: The model both directions are proved on, one of the three priced Globally.
    MODEL_ID: Final[str] = "openai.gpt-5.6-sol"

    @staticmethod
    def _reading(card: str) -> SourceReading:
        """Return the Global reading a served card yields."""
        return SourceReading(
            _card_url("model-card-openai-gpt-56-sol"),
            rates=parse_model_card_global(card),
        )

    def test_the_shipped_global_table_matches_its_card(
        self, gpt_56_sol_card: str
    ) -> None:
        """Every Global rate the entry carries must be the rate the card publishes."""
        findings = classify_global(self.MODEL_ID, self._reading(gpt_56_sol_card))
        assert len(findings) == len(DEFAULT_MODEL_GLOBAL_PRICES[self.MODEL_ID])
        assert {finding.outcome for finding in findings} == {Outcome.MATCH}

    def test_a_wrong_expected_global_value_is_reported_as_drift(
        self, gpt_56_sol_card: str
    ) -> None:
        """A deliberately wrong Global output rate must be reported, with both values.

        The Global rate is ~9% under In-Region, so the drift this guards against
        is quiet: copying the In-Region figure into the Global table is invisible
        in every response the gateway serves.
        """
        drifted = {
            **DEFAULT_MODEL_GLOBAL_PRICES[self.MODEL_ID],
            Dimension.OUTPUT_TOKENS: "0.000022",  # the In-Region rate, not Global
        }
        findings = classify(
            _global_key(self.MODEL_ID), drifted, self._reading(gpt_56_sol_card)
        )

        drift = [f for f in findings if f.outcome is Outcome.DRIFT]
        assert len(drift) == 1
        assert drift[0].model_id == "openai.gpt-5.6-sol (Global)"
        url = _card_url("model-card-openai-gpt-56-sol")
        assert drift[0].detail == (
            f"output_tokens: table has 0.000022, {url} publishes 0.00002"
        )

    def test_a_model_priced_in_region_only_reports_nothing(self) -> None:
        """A card with no Global row, for a model with no entry, is silent.

        Four of the seven priced models are In-Region only; reporting each of
        them on every run would bury the findings that need a person.
        """
        reading = SourceReading(_card_url("model-card-openai-gpt-56-cyber"))
        assert classify_global("openai.gpt-5.6-cyber", reading) == []

    def test_a_withdrawn_global_rate_keeps_its_entry_without_failing(self) -> None:
        """A card that stops quoting a Global rate the table carries is vanished.

        The ``global.`` inference profile keeps serving calls whose usage has to
        be priced, so the entry stays and the run says so.
        """
        reading = SourceReading(_card_url("model-card-openai-gpt-56-sol"))
        findings = classify_global(self.MODEL_ID, reading)
        assert [finding.outcome for finding in findings] == [Outcome.VANISHED]
        assert findings[0].model_id == "openai.gpt-5.6-sol (Global)"
        assert "Keep the entry" in findings[0].detail

    def test_one_withdrawn_global_dimension_is_vanished_rather_than_drift(
        self, gpt_56_sol_card: str
    ) -> None:
        """An em dash in the Global row withdraws that rate, and keeps the entry."""
        card = gpt_56_sol_card.replace(
            '<td tabindex="-1">$5.00</td>', '<td tabindex="-1">—</td>', 1
        )
        findings = classify_global(self.MODEL_ID, self._reading(card))
        vanished = [f for f in findings if f.outcome is Outcome.VANISHED]
        assert len(vanished) == 1
        assert Dimension.CACHE_WRITE_TOKENS.value in vanished[0].detail
        assert Outcome.DRIFT not in {finding.outcome for finding in findings}

    def test_an_unreachable_card_is_never_a_global_drift(self) -> None:
        """A card that could not be read reports unreachable and nothing else."""
        reading = SourceReading(_card_url("x"), problem="ConnectTimeout")
        findings = classify_global(self.MODEL_ID, reading)
        assert [finding.outcome for finding in findings] == [Outcome.UNREACHABLE]

    def test_an_unreachable_card_is_silent_for_a_model_priced_in_region_only(
        self,
    ) -> None:
        """Its In-Region reading already reports the source, so this one must not."""
        reading = SourceReading(_card_url("x"), problem="ConnectTimeout")
        assert classify_global("openai.gpt-5.6-cyber", reading) == []

    def test_a_newly_published_global_rate_is_reported(
        self, gpt_56_sol_card: str
    ) -> None:
        """The next Global rate AWS adds must be noticed, not silently unpriced.

        A globally-routed call to a model the table does not cover is billed at
        the pricier In-Region rate, so the gateway over-reports its cost.
        """
        findings = classify_global(
            "openai.gpt-5.6-cyber", self._reading(gpt_56_sol_card)
        )
        assert {finding.outcome for finding in findings} == {Outcome.NEW}
        assert {finding.model_id for finding in findings} == {
            "openai.gpt-5.6-cyber (Global)"
        }
        assert "DEFAULT_MODEL_GLOBAL_PRICES has no entry" in findings[0].detail


class TestNewAtTheSource:
    """Models a source publishes that no table entry prices are reported.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-openai.html
    """

    def test_a_new_gpt_card_is_reported(self) -> None:
        """A GPT-5.x card the table does not price shows up as new."""
        index = (
            '<a href="./model-card-openai-gpt-56-cyber.html">Cyber</a>'
            '<a href="./model-card-openai-gpt-57-nova.html">Nova</a>'
        )
        findings = unpriced_openai_cards(index)
        assert [finding.model_id for finding in findings] == [
            "model-card-openai-gpt-57-nova"
        ]
        assert findings[0].outcome is Outcome.NEW

    def test_the_gpt_oss_cards_are_not_reported(self) -> None:
        """gpt-oss is carried by the Price List API and must not be noise."""
        index = '<a href="./model-card-openai-gpt-oss-120b.html">gpt-oss</a>'
        assert unpriced_openai_cards(index) == []

    def test_a_new_image_service_row_is_reported(self, stability_table: str) -> None:
        """A pricing-page row the table does not price shows up as new."""
        page = stability_table.replace(
            "<td>Stable Image Outpaint</td>", "<td>Stable Image Relight</td>"
        )
        findings = unpriced_stability_rows(page)
        assert [finding.model_id for finding in findings] == ["stable image relight"]

    def test_the_listed_image_services_are_not_reported(
        self, stability_table: str
    ) -> None:
        """The rows the table already prices must not be reported on every run."""
        assert unpriced_stability_rows(stability_table) == []


class TestReport:
    """The report groups findings so the actionable ones are read first.

    Ref: tests/test_pricing_drift.py:format_report
    """

    def test_every_outcome_is_grouped_and_counted(self) -> None:
        """Each outcome present gets a counted heading and its findings."""
        report = format_report(
            [
                Finding(Outcome.MATCH, "a", "ok"),
                Finding(Outcome.DRIFT, "b", "moved"),
                Finding(Outcome.DRIFT, "c", "moved"),
            ]
        )
        assert "MATCH (1):" in report
        assert "DRIFT (2):" in report
        assert "VANISHED" not in report
        assert report.index("DRIFT") < report.index("MATCH")
