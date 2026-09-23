"""The prices, endpoints and regions an Amazon Bedrock model card publishes.

A handful of cards carry a ``Pricing`` section with a per-1M-token table: one row
per inference option (In-Region, Geo CRIS, Global CRIS), sometimes one table per
context tier, sometimes an AWS GovCloud block below the commercial one. For the
frontier OpenAI models the card is the *only* AWS source of a rate; for the
others it restates what the Price List API publishes. Either way it is an
authoritative statement of what AWS bills, so two readers compare against it:
the ``--drift`` lane (``tests/test_pricing_drift.py``) against the rate the
gateway actually resolves, and the Models page generator against the rates the
gateway publishes. Neither ever publishes a card price itself.

The pages are HTML written for people, so the parser's first duty is to tell
"the price changed" from "I could not read the page": anything ambiguous raises
:class:`UnreadableSourceError` instead of being guessed at.

Only the standard library is used, so the generator can read cards without the
gateway installed; dimensions are the gateway's own ``Dimension`` values as
plain strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from html import unescape
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

#: A card's serving option: its own Region, a geography's profile, or the global one.
type Option = Literal["in_region", "geo", "global"]

#: A context tier: "" for the short (or only) one, "long" past the card's boundary.
type Context = Literal["", "long"]

#: Every serving option, in the order a card lists its rows.
OPTIONS: Final[tuple[Option, ...]] = ("in_region", "geo", "global")

#: Table row labels naming each option; a card with no Geo row may say "US CRIS".
_OPTION_ROWS: Final[dict[Option, tuple[str, ...]]] = {
    "in_region": ("in-region",),
    "geo": ("geo cris", "us cris"),
    "global": ("global cris",),
}

#: Availability-table column header naming each option.
_OPTION_COLUMNS: Final[dict[str, Option]] = {
    "in-region": "in_region",
    "geo": "geo",
    "global": "global",
}

#: The two Bedrock inference endpoints a card distinguishes.
ENDPOINTS: Final[tuple[str, ...]] = ("bedrock-runtime", "bedrock-mantle")

#: Divisor turning a card's per-1M-token rate into a per-token one.
_PER_MILLION: Final[Decimal] = Decimal(1_000_000)

#: The card note stating the unit; a change to it invalidates the division above.
PER_MILLION_NOTE: Final[str] = "per 1 million tokens"

#: Caption fragment labelling the short-context table; an uncaptioned table is one.
_SHORT_CONTEXT_CAPTION: Final[str] = "short context"

#: Caption fragment labelling the long-context table, absent from most cards.
_LONG_CONTEXT_CAPTION: Final[str] = "long context"

#: Heading opening the AWS GovCloud rates, which are not the commercial ones.
_GOVCLOUD_HEADING: Final[str] = "aws govcloud"

#: How a card states its context tables' boundary, e.g. "(272K input tokens or fewer)".
_CONTEXT_WINDOW_SIZE: Final[re.Pattern[str]] = re.compile(
    r"\((\d+)\s*K\b", re.IGNORECASE
)

#: Multiplier turning a card's "K" context-window size into prompt tokens.
_THOUSAND: Final[int] = 1_000

#: Card column header fragment to billed dimension, longest-qualified fragment first.
CARD_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("cache write", "cache_write_tokens"),
    ("cache read", "cache_read_tokens"),
    ("input", "input_tokens"),
    ("output", "output_tokens"),
)

#: Regions outside the commercial partition, whose rates a card prices apart.
_NON_COMMERCIAL_PREFIXES: Final[tuple[str, ...]] = ("us-gov-", "cn-", "eusc-")

#: Any tag, so a fragment reads as its visible text.
_TAG: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")

#: A run of whitespace, collapsed to one space in visible text.
_SPACES: Final[re.Pattern[str]] = re.compile(r"\s+")

#: One table row.
_ROW: Final[re.Pattern[str]] = re.compile(r"<tr.*?</tr>", re.DOTALL)

#: One header or data cell, capturing its inner HTML.
_CELL: Final[re.Pattern[str]] = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.DOTALL)

#: A cell stating a USD amount.
_MONEY: Final[re.Pattern[str]] = re.compile(r"^\$([0-9]+(?:\.[0-9]+)?)$")

#: A page title, which the user guide's not-found stub lacks.
_H1: Final[re.Pattern[str]] = re.compile(r"<h1[\s>]")

#: A paragraph or sub-heading, scanned for the GovCloud heading.
_PARAGRAPH_OR_HEADING: Final[re.Pattern[str]] = re.compile(
    r"<(p|h[3-6])[\s>].*?</\1>", re.DOTALL
)

#: A bold caption or sub-heading labelling the table that follows it, or a table.
_CAPTION_OR_TABLE: Final[re.Pattern[str]] = re.compile(
    r"<p[^>]*>\s*<b>(.*?)</b>\s*</p>|<h[3-6][^>]*>(.*?)</h[3-6]>|<table.*?</table>",
    re.DOTALL,
)

#: The icon a card's availability table marks a supported option with.
_YES_ICON: Final[str] = "icon-yes"

#: A card link, as a provider index page writes it.
_CARD_LINK: Final[re.Pattern[str]] = re.compile(r"(model-card-[a-z0-9.-]+)\.html")

#: A provider index page, as the user guide's table of contents names it.
_INDEX_PAGE: Final[re.Pattern[str]] = re.compile(r"\"(model-cards-[a-z0-9.-]+\.html)\"")


class UnreadableSourceError(Exception):
    """The card was served but its pricing could not be read with confidence.

    Raised instead of guessing whenever the structure the parser depends on is
    absent or ambiguous, so a redesigned page reports as unreadable rather than
    as a changed or withdrawn rate.
    """


@dataclass(frozen=True, slots=True)
class CardPrices:
    """Everything one card's commercial Pricing section publishes.

    Attributes:
        rates: Per-token rate per dimension, keyed by (option, context tier).
            An option or tier the card does not price is absent.
        threshold: Prompt tokens past which the long tier applies, or None when
            the card prices a single tier.
    """

    rates: dict[tuple[Option, Context], dict[str, Decimal]]
    threshold: int | None = None


@dataclass(frozen=True, slots=True)
class CardServing:
    """Where and how one card says its model is invoked.

    Attributes:
        model_ids: Model ID per endpoint, from the Programmatic Access table.
        availability: Options each endpoint offers, per commercial region.
    """

    model_ids: dict[str, str]
    availability: dict[str, dict[str, frozenset[Option]]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RateDiff:
    """One dimension on which a card and the gateway do not say the same.

    Attributes:
        dimension: The billed dimension.
        card: The card's per-token rate, or None when it publishes none.
        gateway: The gateway's per-token rate, or None when it has none.
    """

    dimension: str
    card: Decimal | None
    gateway: Decimal | None

    @property
    def kind(self) -> Literal["drift", "card-only", "gateway-only"]:
        """Name the disagreement.

        Returns:
            "drift" when both publish different rates, else which side alone does.
        """
        if self.card is None:
            return "gateway-only"
        return "card-only" if self.gateway is None else "drift"


def _text(fragment: str) -> str:
    """Return *fragment*'s visible text, unescaped and whitespace-collapsed.

    Args:
        fragment: HTML.

    Returns:
        The visible text.
    """
    return _SPACES.sub(" ", unescape(_TAG.sub(" ", fragment))).strip()


def _raw_rows(table: str) -> list[list[str]]:
    """Return *table*'s rows as lists of raw cell HTML.

    Args:
        table: A table's HTML.

    Returns:
        The rows, header row first.
    """
    return [_CELL.findall(row) for row in _ROW.findall(table)]


def rows(table: str) -> list[list[str]]:
    """Return *table*'s rows as lists of cell texts.

    Args:
        table: A table's HTML.

    Returns:
        The rows, header row first.
    """
    return [[_text(cell) for cell in row] for row in _raw_rows(table)]


def money(cell: str) -> Decimal | None:
    """Return the USD amount *cell* states, or None when it states no rate.

    A card writes an em dash where a rate does not apply, which is a published
    absence rather than an unreadable cell.

    Args:
        cell: A cell's visible text.

    Returns:
        The amount, or None.
    """
    match = _MONEY.match(cell)
    return Decimal(match.group(1)) if match else None


def _card_dimension(header: str) -> str | None:
    """Return the dimension a card column header names, or None for the others.

    Args:
        header: A column header's visible text.

    Returns:
        The dimension, or None.
    """
    lowered = header.casefold()
    return next(
        (dimension for fragment, dimension in CARD_COLUMNS if fragment in lowered), None
    )


def _section(page: str, anchor: str) -> str | None:
    """Return the body of the card section whose heading ID ends in *anchor*.

    Args:
        page: The card's HTML.
        anchor: The section's anchor suffix, e.g. "pricing".

    Returns:
        The section's HTML up to the next ``<h2>``, or None when absent.
    """
    match = re.search(
        rf'<h2[^>]*id="[^"]*-{anchor}"[^>]*>.*?</h2>(.*?)(?=<h2|\Z)', page, re.DOTALL
    )
    return match.group(1) if match else None


def _captioned_tables(section: str) -> list[tuple[str | None, str]]:
    """Return a section's tables, each with the caption labelling it.

    A card carries one unlabelled table, or several captioned ones -- by a bold
    paragraph or a sub-heading: a short and a long context window, a separate
    AWS GovCloud block, or one availability table per endpoint. Pairing each
    table with its own caption is what keeps one table's figures from being
    read as another's.

    Args:
        section: A card section's HTML.

    Returns:
        (caption or None, table HTML) pairs, in page order.
    """
    tables: list[tuple[str | None, str]] = []
    caption: str | None = None
    for match in _CAPTION_OR_TABLE.finditer(section):
        if match.group(0).startswith("<table"):
            tables.append((caption, match.group(0)))
            caption = None
        else:
            bold, heading = match.group(1, 2)
            caption = _text(bold if bold is not None else heading)
    return tables


def has_price_table(page: str) -> bool:
    """Whether the card's Pricing section carries a rate table at all.

    Most cards only link to the Amazon Bedrock pricing page, which is a
    published absence rather than something to read.

    Args:
        page: The card's HTML.

    Returns:
        True when the Pricing section holds a table.
    """
    section = _section(page, "pricing")
    return section is not None and "<table" in section


def pricing_section(page: str) -> str:
    """Return the card's commercial Pricing block, checked to still quote per-1M rates.

    Args:
        page: The card's HTML.

    Returns:
        The Pricing section, truncated before any AWS GovCloud rates.

    Raises:
        UnreadableSourceError: If the section or its unit note is absent.
    """
    body = _section(page, "pricing")
    if body is None:
        msg = "the card has no Pricing section"
        raise UnreadableSourceError(msg)
    if PER_MILLION_NOTE not in _text(body).casefold():
        msg = f"the Pricing section no longer states rates {PER_MILLION_NOTE!r}"
        raise UnreadableSourceError(msg)
    # A GovCloud block repeats the commercial captions below its own heading,
    # so only what precedes that heading is the commercial rate.
    for block in _PARAGRAPH_OR_HEADING.finditer(body):
        if _GOVCLOUD_HEADING in _text(block.group(0)).casefold():
            return body[: block.start()]
    return body


def _tier_table(section: str, context: Context) -> str | None:
    """Return the commercial pricing table of one context tier.

    Only an absent caption or one naming the short context window prices the
    short tier; only one naming the long window prices the long tier, which
    most cards do not carry at all.

    Args:
        section: The commercial Pricing block.
        context: The tier wanted.

    Returns:
        The table, or None when *context* is "long" and the card has none.

    Raises:
        UnreadableSourceError: If the short tier is not exactly one table, or
            the long tier is more than one.
    """
    fragment = _LONG_CONTEXT_CAPTION if context else _SHORT_CONTEXT_CAPTION
    candidates = [
        table
        for caption, table in _captioned_tables(section)
        if (caption is None and not context)
        or (caption is not None and fragment in caption.casefold())
    ]
    if context:
        if len(candidates) > 1:
            msg = f"expected at most one long-context pricing table, found {len(candidates)}"
            raise UnreadableSourceError(msg)
        return candidates[0] if candidates else None
    if len(candidates) != 1:
        msg = (
            "expected exactly one uncaptioned or short-context pricing table, "
            f"found {len(candidates)}"
        )
        raise UnreadableSourceError(msg)
    return candidates[0]


def tier_rows(page: str, context: Context = "") -> list[list[str]] | None:
    """Return the rows of one of a card's commercial pricing tables.

    Every inference option is a row of one table, so all options of a context
    tier are read from it and share the same protection from the other tier's
    block and from GovCloud.

    Args:
        page: The card's HTML.
        context: The tier wanted.

    Returns:
        The header row followed by one row per option, or None when *context*
        is "long" and the card prices a single tier.

    Raises:
        UnreadableSourceError: If the Pricing section, its unit note or the
            wanted table cannot be identified.
    """
    table = _tier_table(pricing_section(page), context)
    return rows(table) if table is not None else None


def option_rates(
    table_rows: list[list[str]], option: Option
) -> dict[str, Decimal] | None:
    """Return the per-token rates the row of *option* states.

    Args:
        table_rows: A pricing table, header row first.
        option: The option wanted.

    Returns:
        The rate per token per dimension, or None when the table has no such
        row -- a published absence.

    Raises:
        UnreadableSourceError: If the row is present but states no rate, which
            reads as changed columns rather than as a withdrawn rate.
    """
    labels = _OPTION_ROWS[option]
    labelled = next(
        (
            row
            for label in labels
            for row in table_rows[1:]
            if row and row[0].casefold() == label
        ),
        None,
    )
    if labelled is None:
        return None
    rates = {
        dimension: amount / _PER_MILLION
        for header, cell in zip(table_rows[0], labelled, strict=False)
        if (dimension := _card_dimension(header)) is not None
        and (amount := money(cell)) is not None
    }
    if not rates:
        msg = f"the {labelled[0]} row states no rate"
        raise UnreadableSourceError(msg)
    return rates


def parse_context_window(page: str) -> int | None:
    """Return the prompt size at which a card leaves its short-context rate.

    A split card captions its first table with the window it prices, e.g.
    "short context (272K input tokens or fewer)".

    Args:
        page: The card's HTML.

    Returns:
        The boundary in prompt tokens, or None when the card captions no
        context window -- the ordinary case of a card pricing a single tier.

    Raises:
        UnreadableSourceError: If the Pricing section or its unit note is
            absent, or a short-context caption states no readable size.
    """
    for caption, _ in _captioned_tables(pricing_section(page)):
        if caption is None or _SHORT_CONTEXT_CAPTION not in caption.casefold():
            continue
        if (size := _CONTEXT_WINDOW_SIZE.search(caption)) is None:
            msg = f"the short-context caption {caption!r} states no window size"
            raise UnreadableSourceError(msg)
        return int(size.group(1)) * _THOUSAND
    return None


def parse_card_prices(page: str) -> CardPrices:
    """Return every commercial rate and the tier boundary a card publishes.

    Args:
        page: The card's HTML.

    Returns:
        The card's prices.

    Raises:
        UnreadableSourceError: If the Pricing section cannot be read with
            confidence, or publishes no rate at all.
    """
    rates: dict[tuple[Option, Context], dict[str, Decimal]] = {}
    contexts: tuple[Context, ...] = ("", "long")
    for context in contexts:
        if (table_rows := tier_rows(page, context)) is None:
            continue
        for option in OPTIONS:
            if (published := option_rates(table_rows, option)) is not None:
                rates[option, context] = published
    if not rates:
        msg = "the pricing table has no In-Region, Geo CRIS or Global CRIS row"
        raise UnreadableSourceError(msg)
    return CardPrices(rates, parse_context_window(page))


def is_commercial(region: str) -> bool:
    """Whether *region* is in the commercial partition the card's rates apply to.

    Args:
        region: An AWS region code.

    Returns:
        True for a commercial region.
    """
    return not region.startswith(_NON_COMMERCIAL_PREFIXES)


def _endpoint_of(caption: str | None) -> str | None:
    """Return the endpoint an availability-table caption names, if any.

    Args:
        caption: The caption, or None.

    Returns:
        The endpoint, or None when the caption names none.
    """
    lowered = (caption or "").casefold()
    return next((endpoint for endpoint in ENDPOINTS if endpoint in lowered), None)


def _availability_table(table: str) -> dict[str, frozenset[Option]] | None:
    """Read one availability table, or None when *table* is not one.

    Args:
        table: A table's HTML.

    Returns:
        Options offered per commercial region, or None when the header is not
        Region / In-Region / Geo / Global (e.g. a CRIS destination table).
    """
    raw = _raw_rows(table)
    if not raw:
        return None
    header = [_text(cell).casefold() for cell in raw[0]]
    if not header or header[0] != "region":
        return None
    columns = [_OPTION_COLUMNS.get(name) for name in header[1:]]
    if None in columns:
        return None
    offered: dict[str, frozenset[Option]] = {}
    for row in raw[1:]:
        if not row:
            continue
        region = _text(row[0]).split(" ", 1)[0]
        if is_commercial(region):
            offered[region] = frozenset(
                option
                for option, cell in zip(columns, row[1:], strict=False)
                if option is not None and _YES_ICON in cell
            )
    return offered


def parse_serving(page: str) -> CardServing:
    """Return the model IDs and per-endpoint availability a card states.

    A card whose availability differs by endpoint captions one table per
    endpoint; one that does not carries a single table, which applies to every
    endpoint the Programmatic Access table lists.

    Args:
        page: The card's HTML.

    Returns:
        What the card says about where its model is invoked.

    Raises:
        UnreadableSourceError: If the Programmatic Access table names no model.
    """
    model_ids: dict[str, str] = {}
    for _, table in _captioned_tables(_section(page, "programmatic-access") or ""):
        for row in rows(table)[1:]:
            if len(row) > 1 and row[0] in ENDPOINTS:
                model_ids.setdefault(row[0], row[1])
    if not model_ids:
        msg = "the Programmatic Access table names no model"
        raise UnreadableSourceError(msg)
    availability: dict[str, dict[str, frozenset[Option]]] = {}
    for caption, table in _captioned_tables(
        _section(page, "regional-availability") or ""
    ):
        if (offered := _availability_table(table)) is None:
            continue
        endpoint = _endpoint_of(caption)
        for target in (endpoint,) if endpoint else tuple(model_ids):
            availability.setdefault(target, {}).update(offered)
    return CardServing(model_ids, availability)


def card_is_withdrawn(slug: str, page: str) -> bool:
    """Whether *page* is the stub the user guide serves for a card that is gone.

    ``docs.aws.amazon.com`` answers an unknown page with 200 and a near-empty
    document, so the HTTP status cannot tell a withdrawn card from a served
    one. A served card carries an ``<h1>`` and derives its section anchors from
    its own slug; the stub has neither. Both are required, because reading a
    served card as withdrawn would stop checking that model without failing.

    Args:
        slug: The card's page name, without ``.html``.
        page: What the user guide served for it.

    Returns:
        True for the not-found stub.
    """
    return slug not in page and not _H1.search(page)


def index_pages(toc: str) -> list[str]:
    """Return the provider index pages the user guide's table of contents lists.

    Args:
        toc: ``toc-contents.json`` as served.

    Returns:
        Page names, sorted.
    """
    return sorted(set(_INDEX_PAGE.findall(toc)))


def card_slugs(index: str) -> set[str]:
    """Return the model cards a provider index page links to.

    Args:
        index: A ``model-cards-<provider>.html`` page.

    Returns:
        Card page names without ``.html``.
    """
    return set(_CARD_LINK.findall(index))


def diff_rates(
    card: Mapping[str, Decimal] | None,
    gateway: Mapping[str, Decimal],
    dimensions: Iterable[str] = (),
) -> list[RateDiff]:
    """Return every dimension on which a card and the gateway disagree.

    Args:
        card: The card's rates for one option and tier, or None when it
            publishes none there.
        gateway: The gateway's rates for the same option and tier.
        dimensions: Further dimensions to compare beyond both sides' own.

    Returns:
        One entry per disagreeing dimension, sorted by dimension.
    """
    published = card or {}
    return [
        RateDiff(dimension, published.get(dimension), gateway.get(dimension))
        for dimension in sorted({*published, *gateway, *dimensions})
        if published.get(dimension) != gateway.get(dimension)
    ]


def prices_to_json(prices: CardPrices) -> dict[str, object]:
    """Serialise card prices for a snapshot, rates as exact decimal strings.

    Args:
        prices: What the card publishes.

    Returns:
        A JSON-compatible mapping :func:`prices_from_json` reads back.
    """
    return {
        "rates": {
            f"{option}:{context}": {dim: f"{rate:f}" for dim, rate in dims.items()}
            for (option, context), dims in prices.rates.items()
        },
        "threshold": prices.threshold,
    }


def prices_from_json(data: Mapping[str, object]) -> CardPrices:
    """Read card prices back from :func:`prices_to_json`'s output.

    Args:
        data: The serialised prices.

    Returns:
        The card prices.

    Raises:
        UnreadableSourceError: If *data* is not what the serialiser writes.
    """
    raw_rates = data.get("rates")
    threshold = data.get("threshold")
    if not isinstance(raw_rates, dict) or not (
        threshold is None or isinstance(threshold, int)
    ):
        msg = "malformed card-price snapshot"
        raise UnreadableSourceError(msg)
    rates: dict[tuple[Option, Context], dict[str, Decimal]] = {}
    for key, dims in raw_rates.items():
        option, _, context = str(key).partition(":")
        if (
            option not in OPTIONS
            or context not in ("", "long")
            or not isinstance(dims, dict)
        ):
            msg = f"malformed card-price snapshot entry {key!r}"
            raise UnreadableSourceError(msg)
        rates[option, context] = {  # type: ignore[index]
            str(dim): Decimal(str(rate)) for dim, rate in dims.items()
        }
    return CardPrices(rates, threshold)
