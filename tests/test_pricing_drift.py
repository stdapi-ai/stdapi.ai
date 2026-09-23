"""Drift detection for the gateway's prices against the AWS sources that publish them.

Two lanes. The first checks the hand-copied tables against the pages they were
copied from. The second, the **card lane**, checks every AWS model card that
publishes a rate table -- found through the provider index pages, whichever
way the gateway gets that model's rate -- against the rate the gateway
actually bills. It runs the gateway's own catalog load, then resolves each
endpoint, region, serving option and context tier the card offers. The card
parser both lanes use is shared with the Models page generator
(``docs_gen/model_catalog/sources/model_card_prices.py``).

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

============  =============================================  ================
Outcome       Meaning                                        Effect
============  =============================================  ================
MATCH         the table agrees with the source               none
DRIFT         the source publishes a different rate          **fails**
CARD-ONLY     the gateway bills nothing for a card's rate    **fails**
VANISHED      the source no longer publishes this rate       reported only
GATEWAY-ONLY  the gateway bills a rate no card publishes     reported only
NEW           the source publishes a rate the table lacks    reported only
UNREACHABLE   the source could not be fetched or parsed      reported only
============  =============================================  ================

CARD-ONLY and GATEWAY-ONLY come from the card lane alone. A card-only rate
fails because it is reported as zero cost.

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
     stdapi/pricing.py:resolve_price
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-openai.html
     https://aws.amazon.com/bedrock/pricing/
"""

import asyncio
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import httpx
import pytest
from botocore.exceptions import BotoCoreError, ClientError

from docs_gen.model_catalog.sources import model_card_prices as card_prices
from docs_gen.model_catalog.sources.model_card_prices import (
    PER_MILLION_NOTE,
    UnreadableSourceError,
    card_is_withdrawn,
    parse_context_window,
)
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
from stdapi.pricing import ContextLength, Dimension, Routing, Service
from tests.conftest import REPO_ROOT

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Mapping

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
    "openai.gpt-6-astra": "model-card-openai-gpt-6-astra",
}

#: Where a model card lives, given its slug.
_USER_GUIDE: Final[str] = "https://docs.aws.amazon.com/bedrock/latest/userguide/"

#: The user guide's table of contents, which lists every provider's card index.
_USER_GUIDE_TOC: Final[str] = f"{_USER_GUIDE}toc-contents.json"

#: Card pages fetched at once by the card lane.
_CARD_FETCH_CONCURRENCY: Final[int] = 8

#: Commercial regions the card lane resolves the gateway's rates in, one or more per geography.
_CARD_LANE_REGIONS: Final[frozenset[str]] = frozenset(
    {
        "us-east-1",
        "us-east-2",
        "us-west-2",
        "ca-central-1",
        "eu-west-1",
        "eu-central-1",
        "ap-northeast-1",
        "ap-south-1",
        "ap-southeast-2",
        "sa-east-1",
    }
)

#: Price-catalog service billing each endpoint a card names.
_ENDPOINT_SERVICE: Final[dict[str, Service]] = {
    "bedrock-runtime": Service.BEDROCK,
    "bedrock-mantle": Service.BEDROCK_MANTLE,
}

#: Routing the gateway records for a call served through each card option.
_OPTION_ROUTING: Final[dict[card_prices.Option, Routing]] = {
    "in_region": "",
    "geo": "",
    "global": "global",
}

#: How the report names each card option.
_OPTION_LABEL: Final[dict[card_prices.Option, str]] = {
    "in_region": "In-Region",
    "geo": "Geo",
    "global": "Global",
}

#: Token dimensions compared whether or not a card prices them.
_CARD_DIMENSIONS: Final[tuple[str, ...]] = (
    Dimension.INPUT_TOKENS,
    Dimension.CACHE_WRITE_TOKENS,
    Dimension.CACHE_READ_TOKENS,
    Dimension.OUTPUT_TOKENS,
)

#: The OpenAI card index, scanned for frontier GPT cards no table entry prices.
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

#: Header of the pricing-page table holding the Stability rates.
_STABILITY_HEADING: Final[str] = "stability ai image services"

#: The unit that table states; a change to it invalidates a direct comparison.
_PER_GENERATION_NOTE: Final[str] = "price per generation"

#: A table on the Bedrock pricing page.
_TABLE: Final[re.Pattern[str]] = re.compile(r"<table.*?</table>", re.DOTALL)


class PriceSourceWarning(UserWarning):
    """A vendor source said something the run reports but must not fail on."""


class Outcome(StrEnum):
    """What comparing one rate against its source established.

    Declared worst first: the report is grouped in this order, so the one
    outcome that needs a person shows above the many that do not. ``CARD_ONLY``
    and ``GATEWAY_ONLY`` come from the card lane, which compares what the
    gateway actually bills rather than a hand-copied table.
    """

    DRIFT = "DRIFT"
    CARD_ONLY = "CARD-ONLY"
    VANISHED = "VANISHED"
    GATEWAY_ONLY = "GATEWAY-ONLY"
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


def _dimensions(rates: Mapping[str, Decimal] | None) -> dict[Dimension, Decimal] | None:
    """Key a shared-parser reading by the gateway's own dimension type.

    Args:
        rates: Per-token rates keyed by dimension value, or None.

    Returns:
        The same rates keyed by :class:`Dimension`, or None.
    """
    return (
        None if rates is None else {Dimension(key): rate for key, rate in rates.items()}
    )


def parse_model_card(
    page: str, context: ContextLength = ""
) -> dict[Dimension, Decimal] | None:
    """Return the per-token regional rates a Bedrock model card publishes.

    That is the In-Region row, or the Geo row of a card with none (Kimi K3 is
    served only through cross-Region profiles, and calls it "US CRIS").

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
            wanted table or its regional row cannot be identified.
    """
    rows = card_prices.tier_rows(page, context)
    if rows is None:
        return None
    rates = card_prices.option_rates(rows, "in_region")
    if rates is None:
        rates = card_prices.option_rates(rows, "geo")
    if rates is None:
        msg = "the pricing table has no In-Region or Geo CRIS row"
        raise UnreadableSourceError(msg)
    return _dimensions(rates)


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
    rows = card_prices.tier_rows(page, context)
    return (
        None if rows is None else _dimensions(card_prices.option_rates(rows, "global"))
    )


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
        rows = card_prices.rows(table)
        heading = " ".join(rows[0]).casefold() if rows else ""
        if _STABILITY_HEADING not in heading:
            continue
        if _PER_GENERATION_NOTE not in heading:
            msg = "the Stability table no longer prices per generation"
            raise UnreadableSourceError(msg)
        prices = {
            row[0].casefold(): amount
            for row in rows[1:]
            if len(row) > 1 and (amount := card_prices.money(row[1])) is not None
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


def classify_card(model_id: str, card: CardReadings) -> list[Finding]:
    """Compare every table carrying *model_id* against what its card publishes.

    Args:
        model_id: The model the entries price.
        card: Everything the model's card had to say.

    Returns:
        The findings for all four rate tables and the context-window boundary.
    """
    return [
        *classify(model_id, DEFAULT_MODEL_PRICES[model_id], card.in_region),
        *classify_global(model_id, card.cross_region),
        *classify_qualified(
            model_id,
            DEFAULT_MODEL_LONG_CONTEXT_PRICES,
            "long context",
            "DEFAULT_MODEL_LONG_CONTEXT_PRICES",
            card.long_in_region,
        ),
        *classify_qualified(
            model_id,
            DEFAULT_MODEL_GLOBAL_LONG_CONTEXT_PRICES,
            "long context, Global",
            "DEFAULT_MODEL_GLOBAL_LONG_CONTEXT_PRICES",
            card.long_cross_region,
        ),
        *classify_threshold(model_id, card.threshold),
    ]


def _card_url(slug: str) -> str:
    """Return the user guide URL of the model card named *slug*."""
    return f"{_USER_GUIDE}{slug}.html"


def _withdrawn_readings(url: str) -> CardReadings:
    """Return the readings of a card the user guide no longer serves."""
    return CardReadings(
        SourceReading(url),
        SourceReading(url),
        SourceReading(url),
        SourceReading(url),
        ThresholdReading(url),
    )


def parse_card_readings(url: str, page: str) -> CardReadings:
    """Return everything a served model card publishes.

    Args:
        url: Where the card was read from, quoted by every finding.
        page: The model card's HTML.

    Returns:
        The card's four rate readings and its context-window boundary.

    Raises:
        UnreadableSourceError: If any of them cannot be read with confidence.
    """
    return CardReadings(
        SourceReading(url, rates=parse_model_card(page)),
        SourceReading(url, rates=parse_model_card_global(page)),
        SourceReading(url, rates=parse_model_card(page, "long")),
        SourceReading(url, rates=parse_model_card_global(page, "long")),
        ThresholdReading(url, tokens=parse_context_window(page)),
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
        return parse_card_readings(url, page)
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
    """Return a finding per frontier GPT model card no table entry prices."""
    known = set(_MODEL_CARD_URLS.values())
    return [
        Finding(
            Outcome.NEW,
            slug,
            f"{_card_url(slug)} is a frontier GPT model card, DEFAULT_MODEL_PRICES "
            f"has no entry",
        )
        for slug in sorted(set(_OPENAI_CARD_LINK.findall(index)) - known)
    ]


def format_report(
    findings: list[Finding],
    title: str = "The hand-copied price tables vs. the sources they were copied from:",
    *,
    counted: frozenset[Outcome] = frozenset(),
) -> str:
    """Render *findings* grouped by outcome, worst first.

    Args:
        findings: What the run established.
        title: The report's first line.
        counted: Outcomes shown as a count only, for a lane whose matches
            would otherwise bury what needs a person.

    Returns:
        The report.
    """
    lines = [title, ""]
    for outcome in Outcome:
        selected = [finding for finding in findings if finding.outcome is outcome]
        if not selected:
            continue
        lines.append(f"{outcome.value} ({len(selected)}):")
        if outcome not in counted:
            lines.extend(
                f"  {finding.model_id}: {finding.detail}" for finding in selected
            )
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
        findings.extend(classify_card(model_id, _read_model_card(client, slug)))

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


#: The recorded cards' short-context table heading.
_SHORT_CAPTION: Final[str] = (
    "Commercial Regions — short context (272K input tokens or fewer)"
)

#: The recorded cards' long-context table heading.
_LONG_CAPTION: Final[str] = (
    "Commercial Regions — long context (more than 272K input tokens)"
)


#: What the card lane does about a finding that fails it.
_FIX_CARD_LANE: Final[str] = (
    "FIX: the gateway bills a different rate than the model card publishes, or "
    "none at all. A card rate AWS also publishes in the Price List points at "
    "the ingestion in stdapi/pricing.py; a card-only rate (the frontier OpenAI "
    "models) at the tables in stdapi/models/pricing_overrides.py; a context "
    "window at MODEL_LONG_CONTEXT_THRESHOLDS. Divide a card's per-1M rate by 1e6."
)

#: Resolves the gateway's standard-tier unit price for one call shape, or None.
type Resolver = Callable[
    [Service, str, str, Dimension, Routing, ContextLength], Decimal | None
]


@dataclass(frozen=True, slots=True)
class PricedCard:
    """One model card that publishes a rate table, read in full.

    Attributes:
        url: Where the card was read from, quoted by every finding.
        prices: The commercial rates and tier boundary it publishes.
        serving: The model IDs and per-endpoint availability it states.
    """

    url: str
    prices: card_prices.CardPrices
    serving: card_prices.CardServing


def _expected_rates(
    prices: card_prices.CardPrices, option: card_prices.Option, context: ContextLength
) -> tuple[Mapping[str, Decimal] | None, bool]:
    """Return what a card says one option bills in one context tier.

    A card that prices a single tier states that rate for every prompt size,
    so a long prompt is expected at the short rate; a gateway rate that differs
    there is one the card does not publish, not a changed one.

    Args:
        prices: What the card publishes.
        option: The serving option.
        context: The context tier.

    Returns:
        The card's rates, or None when it prices nothing there, and whether
        they were published for this tier rather than carried from the short one.
    """
    split = any(tier == "long" for _, tier in prices.rates)
    if context and not split:
        return prices.rates.get((option, "")), False
    return prices.rates.get((option, context)), True


def _rate_findings(
    card: PricedCard,
    endpoint: str,
    region: str,
    option: card_prices.Option,
    resolve: Resolver,
) -> list[tuple[Outcome, str, str]]:
    """Compare one endpoint, region and option of a card against the gateway.

    Args:
        card: The card.
        endpoint: The endpoint serving the call.
        region: The region the call is sent to.
        option: The serving option the card offers there.
        resolve: The gateway's rate resolver.

    Returns:
        (outcome, subject, detail) per dimension and context tier compared.
    """
    model_id = card.serving.model_ids[endpoint]
    service = _ENDPOINT_SERVICE[endpoint]
    contexts: tuple[ContextLength, ...] = ("", "long")
    results: list[tuple[Outcome, str, str]] = []
    for context in contexts:
        expected, published = _expected_rates(card.prices, option, context)
        subject = (
            f"{model_id} ({endpoint}, {_OPTION_LABEL[option]}"
            f"{', long context' if context else ''})"
        )
        billed = {
            dimension: rate
            for dimension in _CARD_DIMENSIONS
            if (
                rate := resolve(
                    service,
                    model_id,
                    region,
                    Dimension(dimension),
                    _OPTION_ROUTING[option],
                    context,
                )
            )
            is not None
        }
        diffs = {
            diff.dimension: diff for diff in card_prices.diff_rates(expected, billed)
        }
        for dimension in sorted({*(expected or {}), *billed}):
            diff = diffs.get(dimension)
            if diff is None:
                results.append(
                    (Outcome.MATCH, subject, f"{dimension}: {billed[dimension]:f}")
                )
                continue
            outcome = (
                Outcome.GATEWAY_ONLY
                if diff.kind == "gateway-only" or not published
                else Outcome.CARD_ONLY
                if diff.kind == "card-only"
                else Outcome.DRIFT
            )
            card_side = "none" if diff.card is None else f"{diff.card:f}"
            gateway_side = "none" if diff.gateway is None else f"{diff.gateway:f}"
            detail = (
                f"{dimension}: gateway bills {gateway_side}, {card.url} publishes "
                f"{card_side}"
            )
            results.append((outcome, subject, detail))
    return results


def classify_card_against_gateway(
    card: PricedCard,
    regions: Collection[str],
    resolve: Resolver,
    threshold: Callable[[str], int],
) -> list[Finding]:
    """Compare every rate a card publishes against what the gateway bills.

    Each endpoint the card names is checked in every region its availability
    table offers it in, among *regions*, for every serving option offered
    there -- the call shapes a real request takes -- and in both context
    tiers. A finding identical across regions is reported once, with its
    regions, so a wrong rate reads as one line rather than thirty.

    Args:
        card: The card.
        regions: The regions the gateway's rates were loaded for.
        resolve: The gateway's rate resolver.
        threshold: The gateway's long-context boundary per model ID.

    Returns:
        The findings, including a context-window one when the card splits its
        rates.
    """
    grouped: dict[tuple[Outcome, str, str], list[str]] = defaultdict(list)
    for endpoint in card.serving.model_ids:
        offered = card.serving.availability.get(endpoint, {})
        for region in sorted(set(offered) & set(regions)):
            for option in sorted(offered[region], key=card_prices.OPTIONS.index):
                for key in _rate_findings(card, endpoint, region, option, resolve):
                    grouped[key].append(region)
    findings = [
        Finding(outcome, subject, f"{detail} [{', '.join(where)}]")
        for (outcome, subject, detail), where in grouped.items()
    ]
    if card.prices.threshold is not None:
        for model_id in sorted(set(card.serving.model_ids.values())):
            billed = threshold(model_id)
            outcome = (
                Outcome.MATCH if billed == card.prices.threshold else Outcome.DRIFT
            )
            findings.append(
                Finding(
                    outcome,
                    _qualified_key(model_id, "context window"),
                    f"gateway switches past {billed} prompt tokens, {card.url} "
                    f"past {card.prices.threshold}",
                )
            )
    return findings


async def _fetch_text(client: httpx.AsyncClient, url: str) -> str:
    """Return the body of *url*.

    Args:
        client: The HTTP client.
        url: The page.

    Returns:
        The page text.

    Raises:
        httpx.HTTPError: When the page cannot be fetched.
    """
    response = await client.get(url)
    response.raise_for_status()
    return response.text


async def _discover_card_slugs(client: httpx.AsyncClient) -> set[str]:
    """Return every model card the provider index pages link to.

    Args:
        client: The HTTP client.

    Returns:
        Card page names without ``.html``.

    Raises:
        httpx.HTTPError: When the table of contents or an index page cannot
            be fetched.
    """
    toc = await _fetch_text(client, _USER_GUIDE_TOC)
    indexes = await asyncio.gather(
        *(
            _fetch_text(client, f"{_USER_GUIDE}{page}")
            for page in card_prices.index_pages(toc)
        )
    )
    return set().union(*(card_prices.card_slugs(index) for index in indexes))


def read_priced_card(url: str, slug: str, page: str) -> PricedCard | None:
    """Read a served card, or None when it publishes no rate table.

    Args:
        url: Where the card was read from.
        slug: The card's page name without ``.html``.
        page: The card's HTML.

    Returns:
        The card, or None for a withdrawn card or one that only links to the
        pricing page.

    Raises:
        UnreadableSourceError: If its rates or serving tables cannot be read.
    """
    if card_prices.card_is_withdrawn(slug, page) or not card_prices.has_price_table(
        page
    ):
        return None
    return PricedCard(
        url, card_prices.parse_card_prices(page), card_prices.parse_serving(page)
    )


async def _read_priced_cards(
    client: httpx.AsyncClient, slugs: Iterable[str]
) -> tuple[list[PricedCard], list[Finding]]:
    """Fetch every card and keep the ones publishing a rate table.

    Args:
        client: The HTTP client.
        slugs: The cards to read.

    Returns:
        The priced cards, and an unreachable finding per card that could not
        be fetched or read.
    """
    semaphore = asyncio.Semaphore(_CARD_FETCH_CONCURRENCY)

    async def read(slug: str) -> PricedCard | Finding | None:
        url = _card_url(slug)
        async with semaphore:
            try:
                return read_priced_card(url, slug, await _fetch_text(client, url))
            except (httpx.HTTPError, UnreadableSourceError) as exc:
                return Finding(Outcome.UNREACHABLE, url, f"{type(exc).__name__}: {exc}")

    outcomes = await asyncio.gather(*(read(slug) for slug in sorted(slugs)))
    return (
        [outcome for outcome in outcomes if isinstance(outcome, PricedCard)],
        [outcome for outcome in outcomes if isinstance(outcome, Finding)],
    )


async def _load_gateway_catalog(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Load the gateway's own price catalog for the card lane's regions.

    Runs ``_load_price_catalog`` itself -- Price List ingestion, Mantle
    fallback, hand-copied defaults, regional fallback -- into a fresh state,
    restricted to the Bedrock service codes, so a resolved rate is exactly
    what a call in that region would be billed.

    Args:
        monkeypatch: Scopes the catalog to this test.

    Returns:
        Whether every fetch succeeded; a partial catalog would report rates it
        never loaded as missing.

    Raises:
        BotoCoreError: When the Price List API is unreachable.
        ClientError: When the Price List API refuses the request.
    """
    monkeypatch.setattr(pricing, "_state", pricing._PriceCatalogState())  # noqa: SLF001
    monkeypatch.setattr(pricing, "_catalog_regions", lambda: set(_CARD_LANE_REGIONS))
    monkeypatch.setattr(
        pricing,
        "_SERVICE_CODE_TO_SERVICE",
        dict.fromkeys(_BEDROCK_SERVICE_CODES, Service.BEDROCK),
    )
    endpoint = pricing.pricing_endpoint_region()
    # type-ignore: the RegionName stub Literal lags EUSC/China (works live).
    async with AWSConnectionManager(("pricing", endpoint)):  # type: ignore[arg-type]
        await pricing._load_price_catalog([])  # noqa: SLF001
    return pricing._state.catalog_complete  # noqa: SLF001


def _resolve_standard(
    service: Service,
    model_id: str,
    region: str,
    dimension: Dimension,
    routing: Routing,
    context: ContextLength,
) -> Decimal | None:
    """Resolve the gateway's standard-tier unit price, as a call is billed.

    Args:
        service: The service billing the call.
        model_id: The model invoked.
        region: The region the call is sent to.
        dimension: The billed dimension.
        routing: The serving profile.
        context: The context tier.

    Returns:
        The per-unit amount, or None when the gateway prices nothing.
    """
    price = pricing.resolve_price(
        service, model_id, region, dimension, routing=routing, context=context
    )
    return None if price is None else price.amount


@pytest.mark.drift
async def test_every_priced_model_card_matches_what_the_gateway_bills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every rate an AWS model card publishes must be the rate the gateway bills.

    A card is an authoritative AWS price source whichever way the gateway gets
    the rate -- the Price List, or a hand-copied table -- so the comparison is
    against the gateway's effective rate: its own catalog load, resolved for
    each endpoint, region and serving option the card offers the model on, in
    both context tiers. That catches what reading a table cannot: an ingestion
    that drops or misreads a row (Kimi K3 resolved nothing on bedrock-runtime),
    a regional fallback carrying the wrong rate, a routing priced at the
    wrong option, a boundary registered at the wrong size.

    Cards are discovered from the provider index pages, so a new priced card
    is checked the day AWS publishes it. A rate the gateway bills differently
    or not at all fails; a rate only the gateway has, and a card that cannot
    be read, are reported.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html
         stdapi/pricing.py:resolve_price
         stdapi/pricing.py:_load_price_catalog
    """
    if pricing.pricing_endpoint_region() is None:
        pytest.skip("this partition has no AWS Price List API endpoint")
    async with httpx.AsyncClient(
        timeout=_FETCH_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        try:
            slugs = await _discover_card_slugs(client)
        except httpx.HTTPError as exc:
            pytest.skip(f"the user guide's card index is not reachable: {exc}")
        cards, findings = await _read_priced_cards(client, slugs)
    try:
        complete = await _load_gateway_catalog(monkeypatch)
    except (BotoCoreError, ClientError) as exc:
        pytest.skip(f"the AWS Price List API is not reachable: {exc}")
    if not complete:
        pytest.skip("the AWS Price List API answered only part of the catalog")
    for card in cards:
        findings.extend(
            classify_card_against_gateway(
                card,
                _CARD_LANE_REGIONS,
                _resolve_standard,
                pricing.long_context_threshold,
            )
        )

    report = format_report(
        findings,
        f"What the gateway bills vs. the {len(cards)} AWS model card(s) publishing "
        f"a rate table, of {len(slugs)} discovered:",
        counted=frozenset({Outcome.MATCH}),
    )
    print(report)  # noqa: T201 -- shown by pytest on failure, and with -s
    if not any(finding.outcome is Outcome.MATCH for finding in findings):
        pytest.skip(f"No card published a rate to compare against.\n{report}")
    reported = [
        finding
        for finding in findings
        if finding.outcome in {Outcome.GATEWAY_ONLY, Outcome.UNREACHABLE}
    ]
    if reported:
        warnings.warn(format_report(reported), PriceSourceWarning, stacklevel=2)
    if any(
        finding.outcome in {Outcome.DRIFT, Outcome.CARD_ONLY} for finding in findings
    ):
        pytest.fail(f"{report}\n{_FIX_CARD_LANE}")


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
def gpt_56_luna_card() -> str:
    """The recorded Pricing section of the GPT-5.6 Luna model card."""
    return (FIXTURES_DIR / "model_card_openai_gpt_56_luna_pricing.html").read_text()


@pytest.fixture(scope="module")
def daybreak_blue_card() -> str:
    """The recorded Pricing section of the Daybreak Blue GPT-5.6 Sol model card."""
    path = FIXTURES_DIR / "model_card_openai_gpt_daybreak_blue_56_sol_pricing.html"
    return path.read_text()


@pytest.fixture(scope="module")
def gpt_6_astra_card() -> str:
    """The recorded Pricing section of the GPT-6 Astra model card."""
    return (FIXTURES_DIR / "model_card_openai_gpt_6_astra_pricing.html").read_text()


@pytest.fixture(scope="module")
def kimi_k3_card() -> str:
    """The recorded Pricing section of the Kimi K3 model card."""
    return (FIXTURES_DIR / "model_card_moonshot_ai_kimi_k3_pricing.html").read_text()


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
        assert rates[Dimension.INPUT_TOKENS] == Decimal("0.0000044")
        assert rates[Dimension.OUTPUT_TOKENS] == Decimal("0.000022")

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
        """A card's GovCloud table must not leak into either rate.

        GPT-5.4 heads its GovCloud table "AWS GovCloud (US-West) — short
        context", so it names the short context window as the commercial one
        does; the GovCloud block is a different partition's rate, and its
        In-Region row is not a Global rate either.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-54.html
        """
        rates = parse_model_card(gpt_54_card)
        assert rates is not None
        assert rates[Dimension.INPUT_TOKENS] == Decimal("0.00000275")
        assert rates[Dimension.OUTPUT_TOKENS] == Decimal("0.0000165")
        assert parse_model_card_global(gpt_54_card) is None

    def test_a_govcloud_block_repeating_the_commercial_captions_is_excluded(
        self, gpt_56_luna_card: str
    ) -> None:
        """A GovCloud block captioned exactly like the commercial one is still skipped.

        GPT-5.6 Luna heads its GovCloud tables "Short context" and "Long
        context" too, so only the GovCloud heading above them separates the two
        blocks. Reading by caption alone found two short-context tables and
        reported the card unreadable, which left the Luna and Terra rates
        unverified against their source.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-luna.html
        """
        rates = parse_model_card(gpt_56_luna_card)
        assert rates is not None
        assert rates[Dimension.INPUT_TOKENS] == Decimal("0.00000022")
        assert rates[Dimension.OUTPUT_TOKENS] == Decimal("0.00000132")

        long_rates = parse_model_card(gpt_56_luna_card, "long")
        assert long_rates is not None
        assert long_rates[Dimension.INPUT_TOKENS] == Decimal("0.00000044")
        assert long_rates[Dimension.OUTPUT_TOKENS] == Decimal("0.00000198")

    def test_the_govcloud_block_does_not_supply_a_global_rate_either(
        self, gpt_56_luna_card: str
    ) -> None:
        """The Global rate still comes from the commercial table above the heading.

        The GovCloud tables carry an In-Region row and no Global one, so reading
        them would have dropped the Global rate rather than misstating it.
        """
        rates = parse_model_card_global(gpt_56_luna_card)
        assert rates is not None
        assert rates[Dimension.INPUT_TOKENS] == Decimal("0.0000002")
        assert rates[Dimension.OUTPUT_TOKENS] == Decimal("0.0000012")

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
        card = gpt_56_sol_card.replace(_LONG_CAPTION, _SHORT_CAPTION)
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
        card = daybreak_blue_card.replace(_LONG_CAPTION, _SHORT_CAPTION)
        with pytest.raises(UnreadableSourceError, match="exactly one"):
            parse_model_card(card)

    def test_a_changed_unit_note_is_unreadable(self, gpt_56_cyber_card: str) -> None:
        """Rates stated in another unit must not be divided by a million."""
        card = gpt_56_cyber_card.replace(PER_MILLION_NOTE, "per 1 thousand tokens")
        with pytest.raises(UnreadableSourceError, match="no longer states"):
            parse_model_card(card)

    def test_a_page_without_a_pricing_section_is_unreadable(self) -> None:
        """A redesigned card reports as unreadable, never as a vanished rate."""
        with pytest.raises(UnreadableSourceError, match="no Pricing section"):
            parse_model_card("<html><body><h1>GPT-5.6 Cyber</h1></body></html>")

    def test_a_bold_paragraph_caption_still_labels_its_table(
        self, gpt_56_sol_card: str
    ) -> None:
        """A card captioning its tables in bold paragraphs reads like a headed one.

        The cards captioned their tables "<p><b>Short Context Window (272K)</b></p>"
        before moving to sub-headings, and a card still in that form must not
        read as one ambiguous uncaptioned section.
        """
        card = re.sub(
            r"<h3[^>]*>Commercial Regions — (short|long) context[^<]*</h3>",
            lambda match: (
                "<p><b>Short Context Window (272K)</b></p>"
                if match.group(1) == "short"
                else "<p><b>Long Context Window (1M)</b></p>"
            ),
            gpt_56_sol_card,
        )
        assert "<h3" not in card
        rates = parse_model_card(card, "long")
        assert rates is not None
        assert rates[Dimension.INPUT_TOKENS] == Decimal("0.0000088")
        assert parse_context_window(card) == 272_000

    def test_a_card_without_an_in_region_row_is_read_from_us_cris(
        self, kimi_k3_card: str
    ) -> None:
        """Kimi K3 prices only its cross-Region profiles: US CRIS is its regional rate.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-moonshot-ai-kimi-k3.html
        """
        assert parse_model_card(kimi_k3_card) == {
            Dimension.INPUT_TOKENS: Decimal("0.0000033"),
            Dimension.OUTPUT_TOKENS: Decimal("0.0000165"),
            Dimension.CACHE_READ_TOKENS: Decimal("0.00000033"),
            Dimension.CACHE_WRITE_TOKENS: Decimal("0.000004125"),
        }
        assert parse_model_card_global(kimi_k3_card) == {
            Dimension.INPUT_TOKENS: Decimal("0.000003"),
            Dimension.OUTPUT_TOKENS: Decimal("0.000015"),
            Dimension.CACHE_READ_TOKENS: Decimal("0.0000003"),
            Dimension.CACHE_WRITE_TOKENS: Decimal("0.00000375"),
        }
        assert parse_context_window(kimi_k3_card) is None

    def test_the_in_region_row_wins_over_us_cris(self, kimi_k3_card: str) -> None:
        """A card quoting both prices the model's own region at In-Region."""
        card = kimi_k3_card.replace(
            '<tr><td tabindex="-1">Global CRIS</td>',
            '<tr><td tabindex="-1">In-Region</td><td tabindex="-1">$9.00</td>'
            '<td tabindex="-1">$9.00</td><td tabindex="-1">$9.00</td>'
            '<td tabindex="-1">$9.00</td></tr>'
            '<tr><td tabindex="-1">Global CRIS</td>',
        )
        rates = parse_model_card(card)
        assert rates is not None
        assert rates[Dimension.INPUT_TOKENS] == Decimal("0.000009")

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
        """The long rate is the commercial one, never the GovCloud block below it."""
        rates = parse_model_card(gpt_54_card, "long")
        assert rates is not None
        assert rates[Dimension.INPUT_TOKENS] == Decimal("0.0000055")
        assert rates[Dimension.OUTPUT_TOKENS] == Decimal("0.00002475")

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

    def test_an_uncaptioned_card_states_no_boundary(
        self, gpt_56_cyber_card: str
    ) -> None:
        """A card pricing one unlabelled table leaves the model on the default."""
        card = re.sub(r"<h3[^>]*>.*?</h3>", "", gpt_56_cyber_card, flags=re.DOTALL)
        assert parse_context_window(card) is None

    def test_a_caption_without_a_size_is_unreadable(self, gpt_56_sol_card: str) -> None:
        """A reworded caption must raise rather than yield a guessed boundary."""
        card = gpt_56_sol_card.replace(
            _SHORT_CAPTION, "Commercial Regions — short context"
        )
        with pytest.raises(UnreadableSourceError, match="no window size"):
            parse_context_window(card)

    def test_two_long_context_tables_are_unreadable(self, gpt_56_sol_card: str) -> None:
        """A caption that stops naming one table must not have one picked for it."""
        card = gpt_56_sol_card.replace(_SHORT_CAPTION, _LONG_CAPTION)
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
        reading = ThresholdReading(_card_url("model-card-openai-gpt-oss-120b"))
        assert classify_threshold("openai.gpt-oss-120b-1:0", reading) == []

    def test_a_newly_split_card_is_reported_not_failed(self) -> None:
        """A card that gains a context window is actionable, not our regression."""
        url = _card_url("model-card-openai-gpt-oss-120b")
        findings = classify_threshold(
            "openai.gpt-oss-120b-1:0", ThresholdReading(url, tokens=200_000)
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
            ("openai.gpt-6-astra", "gpt_6_astra_card"),
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

    @pytest.mark.parametrize(
        ("model_id", "fixture_name"),
        [
            ("openai.gpt-5.4", "gpt_54_card"),
            ("openai.gpt-5.6-luna", "gpt_56_luna_card"),
            ("openai.gpt-5.6-sol", "gpt_56_sol_card"),
            ("openai.gpt-daybreak-blue-5.6-sol", "daybreak_blue_card"),
            ("openai.gpt-6-astra", "gpt_6_astra_card"),
        ],
    )
    def test_every_table_matches_a_split_card(
        self, request: pytest.FixtureRequest, model_id: str, fixture_name: str
    ) -> None:
        """Each table a split card prices, and its boundary, matches the card.

        A split card feeds up to four rate tables and the boundary between
        them, so a model entered in only some of them is billed from the wrong
        tier or routing on the rest.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-astra.html
             https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-54.html
        """
        card: str = request.getfixturevalue(fixture_name)
        findings = classify_card(model_id, parse_card_readings(_card_url("x"), card))
        assert findings, "nothing was compared"
        assert self._outcomes(findings) == {Outcome.MATCH}
        assert any("context window" in finding.model_id for finding in findings)
        assert any("long context" in finding.model_id for finding in findings)

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
        """A frontier GPT card the table does not price shows up as new."""
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

    def test_counted_outcomes_are_reported_as_a_count_only(self) -> None:
        """A lane with hundreds of matches lists only the findings that need a person."""
        report = format_report(
            [Finding(Outcome.MATCH, "a", "ok"), Finding(Outcome.DRIFT, "b", "moved")],
            counted=frozenset({Outcome.MATCH}),
        )
        assert "MATCH (1):" in report
        assert "a: ok" not in report
        assert "b: moved" in report


@pytest.fixture(scope="module")
def sol_card_page() -> str:
    """The recorded GPT-5.6 Sol card: Pricing, Programmatic Access and availability."""
    return (FIXTURES_DIR / "model_card_openai_gpt_56_sol.html").read_text()


@pytest.fixture(scope="module")
def grok_43_card_page() -> str:
    """The recorded Grok 4.3 card: a GovCloud block and one availability table."""
    return (FIXTURES_DIR / "model_card_xai_grok_4_3.html").read_text()


@pytest.fixture(scope="module")
def kimi_k3_card_page() -> str:
    """The recorded Kimi K3 card: US CRIS and Global rows, bedrock-runtime only."""
    return (FIXTURES_DIR / "model_card_moonshot_ai_kimi_k3.html").read_text()


def _per_token(*per_million: str) -> dict[str, Decimal]:
    """Build a card rate set from per-1M figures, in card column order.

    Args:
        *per_million: Input, cache write, cache read and output, "" for none.

    Returns:
        Per-token rates keyed by dimension value.
    """
    return {
        dimension: Decimal(rate) / 1_000_000
        for dimension, rate in zip(_CARD_DIMENSIONS, per_million, strict=True)
        if rate
    }


class TestCardServingParsing:
    """The shared parser reads every rate, endpoint and region a priced card states.

    Ref: docs_gen/model_catalog/sources/model_card_prices.py
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-xai-grok-4-3.html
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-moonshot-ai-kimi-k3.html
    """

    def test_a_split_card_yields_every_option_in_both_tiers(
        self, sol_card_page: str
    ) -> None:
        """In-Region, Geo and Global are each read from both context tables."""
        prices = card_prices.parse_card_prices(sol_card_page)
        assert prices.threshold == 272_000
        assert prices.rates == {
            ("in_region", ""): _per_token("4.40", "5.50", "0.44", "22.00"),
            ("geo", ""): _per_token("4.40", "5.50", "0.44", "22.00"),
            ("global", ""): _per_token("4.00", "5.00", "0.40", "20.00"),
            ("in_region", "long"): _per_token("8.80", "11.00", "0.88", "33.00"),
            ("geo", "long"): _per_token("8.80", "11.00", "0.88", "33.00"),
            ("global", "long"): _per_token("8.00", "10.00", "0.80", "30.00"),
        }

    def test_availability_is_read_per_endpoint(self, sol_card_page: str) -> None:
        """Mantle serves In-Region only and bedrock-runtime Geo and Global only."""
        serving = card_prices.parse_serving(sol_card_page)
        assert serving.model_ids == {
            "bedrock-runtime": "openai.gpt-5.6-sol",
            "bedrock-mantle": "openai.gpt-5.6-sol",
        }
        assert serving.availability["bedrock-mantle"] == {
            "us-east-1": frozenset({"in_region"}),
            "us-east-2": frozenset({"in_region"}),
        }
        runtime = serving.availability["bedrock-runtime"]
        assert runtime["us-east-1"] == frozenset({"geo", "global"})
        assert runtime["eu-west-1"] == frozenset({"global"})

    def test_a_govcloud_block_after_the_commercial_table_is_excluded(
        self, grok_43_card_page: str
    ) -> None:
        """A bold-paragraph GovCloud heading ends the commercial rates."""
        prices = card_prices.parse_card_prices(grok_43_card_page)
        assert prices.rates == {
            ("in_region", ""): _per_token("1.25", "", "0.20", "2.50")
        }
        assert prices.threshold is None

    def test_one_availability_table_applies_to_the_only_endpoint(
        self, grok_43_card_page: str
    ) -> None:
        """An unsplit table is the one endpoint's, and GovCloud rows are left out."""
        serving = card_prices.parse_serving(grok_43_card_page)
        assert serving.model_ids == {"bedrock-mantle": "xai.grok-4.3"}
        assert set(serving.availability["bedrock-mantle"]) == {
            "us-east-1",
            "us-east-2",
            "us-west-2",
        }

    def test_a_us_cris_row_is_the_geo_rate(self, kimi_k3_card_page: str) -> None:
        """Kimi K3 prices its Geo profile under "US CRIS", with no In-Region row."""
        prices = card_prices.parse_card_prices(kimi_k3_card_page)
        assert set(prices.rates) == {("geo", ""), ("global", "")}
        assert prices.rates["geo", ""] == _per_token("3.30", "4.125", "0.33", "16.50")

    def test_supported_icons_mark_availability_whatever_their_alt_text(
        self, kimi_k3_card_page: str
    ) -> None:
        """The "supported" alt text reads as the same yes as the green-circle one."""
        serving = card_prices.parse_serving(kimi_k3_card_page)
        runtime = serving.availability["bedrock-runtime"]
        assert runtime["us-east-1"] == frozenset({"geo", "global"})

    def test_a_card_linking_to_the_pricing_page_has_no_price_table(self) -> None:
        """Most cards only link out, which is not a card to compare."""
        page = (
            '<h1>Nova</h1><h2 id="model-card-nova-pricing">Pricing</h2>'
            '<p>See the <a href="https://aws.amazon.com/bedrock/pricing/">page</a>.</p>'
        )
        assert not card_prices.has_price_table(page)
        assert read_priced_card("u", "model-card-nova", page) is None

    def test_cards_are_discovered_from_the_provider_index(self) -> None:
        """Every card an index page links is found, the contents naming the index."""
        index = (FIXTURES_DIR / "model_cards_openai_index.html").read_text()
        slugs = card_prices.card_slugs(index)
        assert {
            "model-card-openai-gpt-56-sol",
            "model-card-openai-gpt-oss-20b",
        } <= slugs
        toc = '{"href":"model-cards-openai.html"},{"href":"model-card-openai-gpt-54.html"}'
        assert card_prices.index_pages(toc) == ["model-cards-openai.html"]

    def test_card_prices_survive_a_snapshot_round_trip(
        self, sol_card_page: str
    ) -> None:
        """What the generator caches reads back as exactly what was parsed."""
        prices = card_prices.parse_card_prices(sol_card_page)
        restored = card_prices.prices_from_json(card_prices.prices_to_json(prices))
        assert restored == prices


class TestCardAgainstGateway:
    """A priced card is compared with the rate the gateway bills per call shape.

    Ref: tests/test_pricing_drift.py:classify_card_against_gateway
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html
    """

    @staticmethod
    def _card(slug: str, page: str) -> PricedCard:
        """Read a recorded card as the lane reads a served one."""
        card = read_priced_card("card", slug, page)
        assert card is not None
        return card

    @staticmethod
    def _resolver(
        prices: card_prices.CardPrices,
        overrides: Mapping[tuple[str, Routing, ContextLength, str], Decimal | None]
        | None = None,
    ) -> Resolver:
        """Return a resolver billing exactly the card's rates, bar *overrides*.

        Args:
            prices: The card whose rates the gateway is to bill.
            overrides: (service, routing, context, dimension) to a different
                rate, or None to bill nothing.

        Returns:
            The resolver.
        """
        options: dict[Routing, card_prices.Option] = {
            "": "in_region",
            "global": "global",
        }

        def resolve(
            service: Service,
            _model_id: str,
            _region: str,
            dimension: Dimension,
            routing: Routing,
            context: ContextLength,
        ) -> Decimal | None:
            key = (service.value, routing, context, dimension.value)
            if overrides and key in overrides:
                return overrides[key]
            tier = context if (options[routing], context) in prices.rates else ""
            return prices.rates.get((options[routing], tier), {}).get(dimension.value)

        return resolve

    @staticmethod
    def _grouped(findings: list[Finding]) -> dict[Outcome, list[str]]:
        """Group finding subjects by outcome."""
        grouped: dict[Outcome, list[str]] = defaultdict(list)
        for finding in findings:
            grouped[finding.outcome].append(finding.model_id)
        return grouped

    def test_a_gateway_billing_the_card_matches_everywhere(
        self, sol_card_page: str
    ) -> None:
        """Every endpoint, region, option and tier the card offers compares equal."""
        card = self._card("model-card-openai-gpt-56-sol", sol_card_page)
        findings = classify_card_against_gateway(
            card,
            {"us-east-1", "eu-west-1"},
            self._resolver(card.prices),
            lambda _: 272_000,
        )
        grouped = self._grouped(findings)
        assert set(grouped) == {Outcome.MATCH}
        subjects = set(grouped[Outcome.MATCH])
        assert (
            "openai.gpt-5.6-sol (bedrock-mantle, In-Region, long context)" in subjects
        )
        assert "openai.gpt-5.6-sol (bedrock-runtime, Global)" in subjects
        assert "openai.gpt-5.6-sol (bedrock-mantle, Global)" not in subjects

    def test_a_different_global_rate_is_one_drift_across_regions(
        self, sol_card_page: str
    ) -> None:
        """A wrong rate reads as one finding naming every region it is wrong in."""
        card = self._card("model-card-openai-gpt-56-sol", sol_card_page)
        resolve = self._resolver(
            card.prices,
            {("bedrock-runtime", "global", "", "input_tokens"): Decimal("0.0000044")},
        )
        drifts = [
            finding
            for finding in classify_card_against_gateway(
                card, {"us-east-1", "eu-west-1"}, resolve, lambda _: 272_000
            )
            if finding.outcome is Outcome.DRIFT
        ]
        assert len(drifts) == 1
        assert drifts[0].model_id == "openai.gpt-5.6-sol (bedrock-runtime, Global)"
        assert "gateway bills 0.0000044" in drifts[0].detail
        assert drifts[0].detail.endswith("[eu-west-1, us-east-1]")

    def test_a_rate_the_gateway_cannot_price_is_card_only(
        self, sol_card_page: str
    ) -> None:
        """A dropped row is reported at zero cost, so it fails like a drift."""
        card = self._card("model-card-openai-gpt-56-sol", sol_card_page)
        resolve = self._resolver(
            card.prices, {("bedrock-mantle", "", "long", "cache_write_tokens"): None}
        )
        grouped = self._grouped(
            classify_card_against_gateway(
                card, {"us-east-1"}, resolve, lambda _: 272_000
            )
        )
        assert grouped[Outcome.CARD_ONLY] == [
            "openai.gpt-5.6-sol (bedrock-mantle, In-Region, long context)"
        ]

    def test_a_long_rate_on_a_single_tier_card_is_gateway_only(
        self, grok_43_card_page: str
    ) -> None:
        """A single-tier card bills every prompt size at one rate; a second is ours."""
        card = self._card("model-card-xai-grok-4-3", grok_43_card_page)
        resolve = self._resolver(
            card.prices,
            {("bedrock-mantle", "", "long", "input_tokens"): Decimal("0.0000025")},
        )
        grouped = self._grouped(
            classify_card_against_gateway(
                card, {"us-west-2"}, resolve, lambda _: 200_000
            )
        )
        assert Outcome.DRIFT not in grouped
        assert grouped[Outcome.GATEWAY_ONLY] == [
            "xai.grok-4.3 (bedrock-mantle, In-Region, long context)"
        ]

    def test_a_different_context_window_is_a_drift(self, sol_card_page: str) -> None:
        """The boundary selects between two correct rates, so a wrong one mis-bills."""
        card = self._card("model-card-openai-gpt-56-sol", sol_card_page)
        findings = classify_card_against_gateway(
            card, set(), self._resolver(card.prices), lambda _: 200_000
        )
        assert [(finding.outcome, finding.model_id) for finding in findings] == [
            (Outcome.DRIFT, "openai.gpt-5.6-sol (context window)")
        ]
