"""Drift detection for ``DEFAULT_MODEL_PRICES`` against the sources it was copied from.

``stdapi/models/pricing_overrides.py:DEFAULT_MODEL_PRICES`` is a hand-copied
table of rates the AWS Price List API does not publish. Nothing in the running
gateway can notice when AWS changes one of them: the figure is simply reported
to the operator as fact. It has already happened -- GPT-5.6 Luna shipped at 5x
the real rate for two releases -- so the table needs a check that reads the
vendor source and says so.

Two sources, and they are not interchangeable:

- **Bedrock model cards** (``docs.aws.amazon.com``) for the OpenAI Mantle
  models. Server-rendered documentation with a labelled ``Pricing`` section and
  a per-1M-token table. The AWS Bedrock pricing page no longer carries per-1M
  rates for these models at all -- it links out to the cards -- so the card is
  the *only* AWS source for them.
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
stays and the run says so out loud.

The live check is opt-in (``--drift``): a vendor changing a price is not a
regression in this repository, and it must never turn an unrelated run red. It
fails only inside its own lane, where a hard failure is the point. The
classifier itself is exercised offline against recorded fixtures, including a
deliberately wrong expected value, because a detector nobody has seen fail is
not known to work.

Ref: stdapi/models/pricing_overrides.py:DEFAULT_MODEL_PRICES
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

from stdapi.models.pricing_overrides import DEFAULT_MODEL_PRICES
from stdapi.pricing import Dimension
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


def _short_context_table(section: str) -> str:
    """Return the card's commercial short-context pricing table.

    A card carries one unlabelled table, or several captioned ones: a short and
    a long context window, or a separate AWS GovCloud block. Only an absent
    caption or one naming the short context window prices what the table
    registers, and anything else is a different rate that must not be guessed
    at.

    Raises:
        UnreadableSourceError: If the section holds no single such table.
    """
    candidates: list[str] = []
    caption: str | None = None
    for match in _CAPTION_OR_TABLE.finditer(section):
        if match.group(0).startswith("<table"):
            if caption is None or "short context" in caption.casefold():
                candidates.append(match.group(0))
            caption = None
        else:
            caption = _text(match.group(1))
    if len(candidates) != 1:
        msg = (
            f"expected exactly one uncaptioned or short-context pricing table, "
            f"found {len(candidates)}"
        )
        raise UnreadableSourceError(msg)
    return candidates[0]


def parse_model_card(page: str) -> dict[Dimension, Decimal]:
    """Return the per-token In-Region rates a Bedrock model card publishes.

    Args:
        page: The model card's HTML.

    Returns:
        The rate per token, per dimension, for the model's own region.

    Raises:
        UnreadableSourceError: If the pricing section, its unit note, its
            short-context table or its In-Region row cannot be identified.
    """
    section = _PRICING_SECTION.search(page)
    if section is None:
        msg = "the card has no Pricing section"
        raise UnreadableSourceError(msg)
    body = section.group(1)
    if _PER_MILLION_NOTE not in _text(body).casefold():
        msg = f"the Pricing section no longer states rates {_PER_MILLION_NOTE!r}"
        raise UnreadableSourceError(msg)
    rows = _rows(_short_context_table(body))
    in_region = next(
        (row for row in rows[1:] if row and row[0].casefold() == _IN_REGION_ROW), None
    )
    if not rows or in_region is None:
        msg = "the pricing table has no In-Region row"
        raise UnreadableSourceError(msg)
    rates = {
        dimension: amount / _PER_MILLION
        for header, cell in zip(rows[0], in_region, strict=False)
        if (dimension := _card_dimension(header)) is not None
        and (amount := _money(cell)) is not None
    }
    if not rates:
        msg = "the In-Region row states no rate"
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


def _read_model_card(client: httpx.Client, slug: str) -> SourceReading:
    """Fetch and parse a model card, never raising on a source-side problem."""
    url = _card_url(slug)
    try:
        response = client.get(url)
        if response.status_code == httpx.codes.NOT_FOUND:
            return SourceReading(url)
        response.raise_for_status()
        if card_is_withdrawn(slug, response.text):
            return SourceReading(url)
        return SourceReading(url, rates=parse_model_card(response.text))
    except (httpx.HTTPError, UnreadableSourceError) as exc:
        return SourceReading(url, problem=f"{type(exc).__name__}: {exc}")


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
    lines = ["DEFAULT_MODEL_PRICES vs. the sources it was copied from:", ""]
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
    "DEFAULT_MODEL_PRICES in stdapi/models/pricing_overrides.py to the value "
    "the source now publishes (a model card states per-1M-token rates: divide "
    "by 1e6), re-run this test, and note the change in the release entry. "
    "Never delete an entry whose rate merely stopped being published: usage "
    "recorded against that model still has to be priced."
)


def _collect(client: httpx.Client) -> list[Finding]:
    """Read every source once and classify the whole table against it."""
    findings: list[Finding] = []
    for model_id, slug in _MODEL_CARD_URLS.items():
        reading = _read_model_card(client, slug)
        findings.extend(classify(model_id, DEFAULT_MODEL_PRICES[model_id], reading))

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
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards-openai.html
         https://aws.amazon.com/bedrock/pricing/
    """
    priced = {*_MODEL_CARD_URLS, *_STABILITY_PAGE_NAMES}
    assert priced == set(DEFAULT_MODEL_PRICES), (
        "DEFAULT_MODEL_PRICES gained or lost an entry: give it a source above, "
        "or this detector silently stops covering it"
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


@pytest.fixture(scope="module")
def gpt_56_cyber_card() -> str:
    """The recorded Pricing section of the GPT-5.6 Cyber model card."""
    return (FIXTURES_DIR / "model_card_openai_gpt_56_cyber_pricing.html").read_text()


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
        assert rates[Dimension.INPUT_TOKENS] == Decimal("0.0000055")
        assert rates[Dimension.OUTPUT_TOKENS] == Decimal("0.000033")

    def test_an_em_dash_is_a_published_absence_not_a_rate(
        self, gpt_56_cyber_card: str
    ) -> None:
        """A dimension a card prices with an em dash is absent, not unreadable."""
        card = gpt_56_cyber_card.replace(">$17.1875<", ">—<")
        assert Dimension.CACHE_WRITE_TOKENS not in parse_model_card(card)

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
