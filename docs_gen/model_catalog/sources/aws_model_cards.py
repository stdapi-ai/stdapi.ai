"""Facts from the Amazon Bedrock model cards in the AWS user guide.

Each model AWS serves has a documentation page stating its context window, its
output ceiling, its knowledge cutoff, its launch date and its end-of-life date.
None of that is in any API response an ordinary caller can make, and it is the
authoritative statement for the models AWS itself hosts.

Only *facts* are read: numbers, dates and lifecycle states. The prose on those
pages is AWS's copy under the AWS Site Terms, and this generator does not
republish it — the model descriptions the page shows come from the Bedrock API
instead.

The join is exact. Every card prints the model IDs it describes, and a card's
facts are attached only to IDs the catalogue already has, so a page cannot be
matched to a model by guesswork.

A card's rate table is read too, but never published: the gateway stays the
one source of every price on the page. What the card publishes is compared with
what the gateway publishes, and any disagreement goes to the run report
(:func:`price_disagreements`) for a person to settle.
"""

from __future__ import annotations

import html
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from docs_gen.model_catalog.http import get_bytes, map_concurrent
from docs_gen.model_catalog.sources import model_card_prices, snapshot
from docs_gen.model_catalog.tokens import format_tokens, parse_tokens

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

#: Where the user guide lives.
_BASE: str = "https://docs.aws.amazon.com/bedrock/latest/userguide/"

#: The guide's table of contents, which lists every model card page.
_TOC: str = _BASE + "toc-contents.json"

#: Card page names, as they appear in the table of contents.
_CARD_PAGE: re.Pattern[str] = re.compile(r"\"(model-card-[a-z0-9\-.]+\.html)\"")

#: Every tag, so a page can be read as the list of visible lines it renders to.
_TAG: re.Pattern[str] = re.compile(r"<[^>]+>")

#: A Bedrock-shaped model ID, before it is checked against the catalogue.
_MODEL_ID: re.Pattern[str] = re.compile(
    r"\b([a-z0-9-]+\.[a-z0-9][a-z0-9\-.]{3,}(?::\d+)?)\b"
)

#: Cross-region prefixes a card prints in front of the plain model ID.
_GEOGRAPHY: frozenset[str] = frozenset({"us", "eu", "apac", "jp", "au", "ca", "global"})

#: The labelled facts worth reading, and the field each one fills.
_FIELDS: dict[str, str] = {
    "Context window": "context_window",
    "Max output tokens": "max_output_tokens",
    "Knowledge cutoff": "knowledge_cutoff",
    "Model launch date": "launch_date",
    "Model EOL date": "eol_date",
}

#: A label line, which is followed by its value on the next visible line.
_LABEL: re.Pattern[str] = re.compile(rf"^({'|'.join(map(re.escape, _FIELDS))}):$")

#: A label and its value rendered as one visible line, e.g. one table cell.
_LABEL_VALUE: re.Pattern[str] = re.compile(
    rf"^({'|'.join(map(re.escape, _FIELDS))}):\s*(\S.*)$"
)

#: Cards a run must still read a context window from, or its layout has changed.
#:
#: 102 of the 124 cards published as of 2026-08 state one. Well under half of
#: that means the parser stopped finding the label, not that AWS wrote less.
_CONTEXT_WINDOW_FLOOR: int = 50

#: Cards a run must still read their APIs from; 127 of 132 marked them as of 2026-09.
_CAPABILITIES_FLOOR: int = 50

#: A card's API column names, and the catalogue APIs each one stands for.
_CARD_APIS: dict[str, tuple[str, ...]] = {
    "Responses": ("OpenAI Responses",),
    "Chat Completions": ("OpenAI Chat Completions",),
    "Converse": ("Converse", "ConverseStream"),
    "Invoke": ("InvokeModel", "InvokeModelStream"),
}

#: API column names a card prints that the catalogue has no API for.
_UNLISTED_CARD_APIS: frozenset[str] = frozenset({"Messages"})

#: One HTML table.
_TABLE: re.Pattern[str] = re.compile(r"<table\b.*?</table>", re.DOTALL)

#: One table row.
_ROW: re.Pattern[str] = re.compile(r"<tr\b.*?</tr>", re.DOTALL)

#: One table cell, header or data.
_CELL: re.Pattern[str] = re.compile(r"<t[dh]\b.*?</t[dh]>", re.DOTALL)

#: The supported ("yes") or not-supported ("no") icon a card marks an item with.
_ICON: re.Pattern[str] = re.compile(r"<img\b[^>]*\bicon-(yes|no)\.png[^>]*>")

#: A feature line stating prompt caching, e.g. "Explicit Prompt Caching".
_CACHING: re.Pattern[str] = re.compile(r"\bprompt caching\b", re.IGNORECASE)

#: Values that mean "not applicable" rather than a fact.
_ABSENT: frozenset[str] = frozenset({"n/a", "na", "none", "-", "—", ""})

#: Months as the cards abbreviate them, for reading "Aug 2023".
_MONTHS: dict[str, str] = {
    name.lower(): f"{index:02d}"
    for index, name in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}


def _lines(body: str) -> list[str]:
    """Render one documentation page down to its visible lines.

    Args:
        body: The page's HTML.

    Returns:
        Non-empty visible lines, in order.
    """
    return [
        line.strip()
        for line in html.unescape(_TAG.sub("\n", body)).splitlines()
        if line.strip()
    ]


def _marked_items(cell: str) -> list[tuple[bool | None, str]]:
    """Split one table cell into the items its icons mark.

    Args:
        cell: The cell's HTML.

    Returns:
        Each item's mark (``None`` for text before the first icon) and its text.
    """
    parts = _ICON.split(cell)
    items: list[tuple[bool | None, str]] = [(None, parts[0])]
    items.extend(
        (mark == "yes", text)
        for mark, text in zip(parts[1::2], parts[2::2], strict=True)
    )
    return [
        (mark, " ".join(html.unescape(_TAG.sub(" ", text)).split()))
        for mark, text in items
    ]


def parse_capabilities(body: str) -> dict[str, Any]:
    """Read the APIs and the prompt caching a model card marks supported.

    Cards print APIs in two layouts: one table per endpoint, with the API names
    as headers and an icon under each; or an icon beside each API name. Either
    way an API counts as supported when any endpoint supports it.

    Args:
        body: The card page's HTML.

    Returns:
        ``apis``, card API name to whether it is supported, and
        ``prompt_caching``, when the card states them.
    """
    apis: dict[str, bool] = {}
    caching: list[bool] = []
    known = _CARD_APIS.keys() | _UNLISTED_CARD_APIS
    for table in _TABLE.findall(body):
        rows = [
            [_marked_items(cell) for cell in _CELL.findall(row)]
            for row in _ROW.findall(table)
        ]
        headers = (
            [" ".join(item[1] for item in cell).strip() for cell in rows[0]]
            if rows
            else []
        )
        per_column = len(rows) == 2 and headers and set(headers) <= known
        for index, row in enumerate(rows):
            for column, cell in enumerate(row):
                for mark, text in cell:
                    if mark is None:
                        continue
                    name = headers[column] if per_column and index == 1 else text
                    if name in known:
                        apis[name] = apis.get(name, False) or mark
                    elif _CACHING.search(text):
                        caching.append(mark)
    stated: dict[str, Any] = {}
    # A table marking every API unsupported is about an API it does not list,
    # such as bidirectional streaming or asynchronous invocation.
    if any(apis.values()):
        stated["apis"] = apis
    if caching:
        stated["prompt_caching"] = any(caching)
    return stated


def apply_card_apis(listed: Iterable[str], card: Mapping[str, bool]) -> list[str]:
    """Correct the APIs Amazon Bedrock lists for a model with what its card states.

    The card's Converse and Invoke columns do not tell a call from its streamed
    form, so they only remove an API or add the plain call where the listing
    has neither form; the OpenAI columns name one API each and are taken as is.

    Args:
        listed: The APIs ``ListFoundationModels`` reports.
        card: Card API name to whether the card marks it supported.

    Returns:
        The corrected APIs, sorted.
    """
    apis = set(listed)
    for name, supported in card.items():
        labels = _CARD_APIS.get(name, ())
        if not labels:
            continue
        if not supported:
            apis.difference_update(labels)
        elif not apis.intersection(labels):
            apis.add(labels[0])
    return sorted(apis)


def _card_pages() -> list[str]:
    """List every model card page the user guide publishes.

    Returns:
        Page names, sorted.
    """
    toc = get_bytes(_TOC).decode("utf-8", "replace")
    return sorted(set(_CARD_PAGE.findall(toc)))


def _read_card(page: str) -> dict[str, Any]:
    """Read one model card.

    Args:
        page: Page name within the user guide.

    Returns:
        The IDs the card describes and the facts it states, or an empty mapping
        when the page cannot be read.
    """
    try:
        body = get_bytes(_BASE + page).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 -- one unreachable card must not fail the run
        return {}
    lines = _lines(body)
    facts: dict[str, str] = {}
    for index, line in enumerate(lines):
        one_line = _LABEL_VALUE.match(line)
        if one_line:
            facts.setdefault(_FIELDS[one_line.group(1)], one_line.group(2).strip())
            continue
        if index == len(lines) - 1:
            continue
        label = _LABEL.match(line)
        if not label:
            continue
        value_line = lines[index + 1]
        # An empty value cell leaves the next label as the next visible line;
        # that label is not this field's value.
        if _LABEL.match(value_line) or _LABEL_VALUE.match(value_line):
            continue
        facts.setdefault(_FIELDS[label.group(1)], value_line)
    card: dict[str, Any] = {
        "page": page,
        "ids": sorted(set(_MODEL_ID.findall(body))),
        "facts": facts,
    }
    if capabilities := parse_capabilities(body):
        card["capabilities"] = capabilities
    if model_card_prices.has_price_table(body):
        try:
            card["prices"] = model_card_prices.prices_to_json(
                model_card_prices.parse_card_prices(body)
            )
        except model_card_prices.UnreadableSourceError as error:
            card["price_problem"] = str(error)
    return card


def _collect() -> list[dict[str, Any]]:
    """Read every model card.

    Returns:
        One entry per card that stated anything.

    Raises:
        RuntimeError: Too few cards yielded a context window or their APIs,
            meaning the user guide's layout changed under the parser rather than
            AWS writing less.
    """
    cards = [card for card in map_concurrent(_read_card, _card_pages()) if card]
    for stated, found, floor in (
        (
            "a context window",
            sum(1 for card in cards if card.get("facts", {}).get("context_window")),
            _CONTEXT_WINDOW_FLOOR,
        ),
        (
            "their APIs",
            sum(1 for card in cards if "apis" in card.get("capabilities", {})),
            _CAPABILITIES_FLOOR,
        ),
    ):
        if found < floor:
            msg = (
                f"only {found} of {len(cards)} model cards state {stated}, under "
                f"the floor of {floor}; the user guide's layout likely changed"
            )
            raise RuntimeError(msg)
    return [
        card
        for card in cards
        if card.get("facts")
        or "capabilities" in card
        or "prices" in card
        or "price_problem" in card
    ]


def _plain_id(model_id: str) -> str:
    """Strip a cross-region prefix from a model ID printed on a card.

    Args:
        model_id: An ID as the card prints it.

    Returns:
        The ID without its geography prefix.
    """
    head, _, tail = model_id.partition(".")
    return tail if tail and head in _GEOGRAPHY else model_id


#: The API-version tag AWS appends to a model's own slug in its page name,
#: A trailing API-version tag: ``-v1:0``, optionally after a release date.
#:
#: Anchored on the colon, so a model whose own name carries a version — Kimi
#: ``v3.1``, GLM ``4.7`` — keeps it.
_VERSION_TAG: re.Pattern[str] = re.compile(r"(-\d{8})?-v\d+:\d+$")


def _page_slug(page: str) -> str:
    """Return a page's own identity, as its file name spells it.

    Args:
        page: Page name within the user guide.

    Returns:
        The page name with the common prefix and suffix stripped.
    """
    return page.removeprefix("model-card-").removesuffix(".html")


def _model_part(model_id: str) -> str:
    """Render the model half of a catalogue ID the way a page name spells it.

    A page is named ``model-card-<vendor>-<model>``, where the vendor is AWS's
    own slug for the publisher and does not always match the ID's namespace —
    ``moonshot-ai`` against ``moonshot.``, and DeepSeek repeats itself. The
    vendor is therefore dropped from both sides and only the model is compared.

    Args:
        model_id: A plain (geography-stripped) catalogue ID.

    Returns:
        The model half, hyphenated, without its trailing API-version tag.
    """
    _, _, model = model_id.lower().partition(".")
    return _VERSION_TAG.sub("", (model or model_id.lower())).replace(".", "-")


def _confirms(page: str, model_id: str) -> bool:
    """Report whether a page names this model as the one it describes.

    Args:
        page: Page name within the user guide.
        model_id: A catalogue ID the page mentions.

    Returns:
        True when the page's own name ends with this model's name. A two-token
        tail is required, so a bare version like ``v3`` cannot claim a page.
    """
    tail = _model_part(model_id)
    if tail.count("-") < 1:
        return False
    slug = _page_slug(page)
    return slug == tail or slug.endswith(f"-{tail}")


def _date(value: str) -> str | None:
    """Read a date the way the cards write it.

    Args:
        value: ``Mar 13, 2024``, ``September 10, 2026`` or ``Aug 2023``,
            optionally prefixed ``Legacy:``.

    Returns:
        An ISO date or year-month, or ``None``.
    """
    text = value.strip().rstrip(".").removeprefix("Legacy:").strip()
    # "No sooner than 10/1/2026" is a floor AWS may move, not a retirement date.
    if text.lower() in _ABSENT or text.lower().startswith("no sooner"):
        return None
    full = re.match(r"^([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})$", text)
    if full:
        month = _month_of(full.group(1))
        return f"{full.group(3)}-{month}-{int(full.group(2)):02d}" if month else None
    partial = re.match(r"^([A-Za-z]+)\s+(\d{4})$", text)
    if partial:
        month = _month_of(partial.group(1))
        return f"{partial.group(2)}-{month}" if month else None
    return text if re.match(r"^\d{4}(-\d{2}){0,2}$", text) else None


def _month_of(name: str) -> str | None:
    """Return the two-digit month for a month name or abbreviation.

    Args:
        name: ``Aug``, ``August``, or anything else.

    Returns:
        ``08``, or ``None``.
    """
    lowered = name.lower()
    for full, number in _MONTHS.items():
        if full.startswith(lowered[:3]):
            return number
    return None


def _claimed(card: Mapping[str, Any], catalogue: set[str]) -> tuple[set[str], str]:
    """Return the catalogue models a card describes, and why when it is not one.

    Args:
        card: One card entry of the snapshot.
        catalogue: Every Bedrock model ID in the catalogue.

    Returns:
        The IDs the card's facts and prices belong to, and "" or the problem
        with the join: "unmatched" (no catalogue ID), "ambiguous" (several and
        none confirmed, so none returned) or "multi" (several confirmed).
    """
    page = str(card.get("page"))
    matched = {
        plain
        for raw_id in card.get("ids", ())
        if (plain := _plain_id(str(raw_id))) in catalogue
    }
    if not matched:
        return set(), "unmatched"
    if len(matched) == 1:
        return matched, ""
    # A card that names more than one catalogue model may be a table of the
    # model it describes plus a cross-referenced sibling; only the IDs the
    # page's own slug confirms are the card's own identity.
    confirmed = {model_id for model_id in matched if _confirms(page, model_id)}
    if not confirmed:
        return matched, "ambiguous"
    return confirmed, "multi" if len(confirmed) > 1 else ""


def _snapshot_cards(*, refresh: bool) -> list[dict[str, Any]]:
    """Return every card the user guide publishes, from the day's snapshot.

    Args:
        refresh: Ignore any cached snapshot.

    Returns:
        One entry per card that stated anything.
    """
    raw = snapshot("aws_model_cards", _collect, refresh=refresh)
    assert isinstance(raw, list)  # noqa: S101 -- snapshot round-trips its own JSON
    # A snapshot taken before the cards were read for their APIs carries none.
    if not refresh and not any("capabilities" in card for card in raw):
        return _snapshot_cards(refresh=True)
    return raw


def fetch(
    known: Iterable[str], *, refresh: bool = False
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Read the facts the model cards state, for models the catalogue has.

    Args:
        known: Every Bedrock model ID in the catalogue.
        refresh: Ignore any cached snapshot.

    Returns:
        Model ID to the facts its card states, and any notes worth reporting.
    """
    catalogue = set(known)
    facts: dict[str, dict[str, Any]] = {}
    problems: dict[str, list[str]] = defaultdict(list)

    cards = _snapshot_cards(refresh=refresh)
    for card in cards:
        stated = card.get("facts") or {}
        matched, problem = _claimed(card, catalogue)
        if problem:
            page = str(card.get("page"))
            problems[problem].append(
                page
                if problem == "unmatched"
                else f"{page} ({', '.join(sorted(matched))})"
            )
            if problem != "multi":
                continue
        contributed = {
            "context_window": format_tokens(stated.get("context_window")),
            "max_output_tokens": parse_tokens(stated.get("max_output_tokens")),
            "knowledge_cutoff": _date(stated.get("knowledge_cutoff", "")),
            "start_of_life": _date(stated.get("launch_date", "")),
            "end_of_life": _date(stated.get("eol_date", "")),
            "card_apis": card.get("capabilities", {}).get("apis"),
            "prompt_caching": card.get("capabilities", {}).get("prompt_caching"),
        }
        usable = {key: value for key, value in contributed.items() if value is not None}
        for model_id in matched:
            facts.setdefault(model_id, {}).update(usable)

    notes = []
    if not any("capabilities" in card for card in cards):
        notes.append("no model card states its APIs, so none corrects the listing")
    if unmatched := problems["unmatched"]:
        notes.append(
            f"{len(unmatched)} model card(s) describe no model this gateway serves"
        )
    if ambiguous := problems["ambiguous"]:
        notes.append(
            "model card(s) name more than one catalogue model with no way to "
            f"tell which the page describes, so no facts were attached: "
            f"{'; '.join(ambiguous)}"
        )
    if still_multi := problems["multi"]:
        notes.append(
            f"model card(s) confirm more than one catalogue model as their own: "
            f"{'; '.join(still_multi)}"
        )
    return facts, notes


#: How the report names each card option; the gateway does not tell In-Region from Geo.
_OPTION_LABEL: dict[str, str] = {"regional": "In-Region/Geo", "global": "Global"}

#: Price-row routings that are neither a region's own rate nor the global one.
_OTHER_ROUTINGS: frozenset[str] = frozenset({"latency"})


def _gateway_rates(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, dict[Decimal, list[str]]]]:
    """Group a gateway price card's standard token rates the way a card prices them.

    The gateway names a regional row by its region or by its geography's
    profile prefix, and both are what a card's In-Region and Geo rows price --
    the same rate on every card that publishes both -- so they are one class
    here.

    Args:
        rows: The price rows of one ``model_pricing`` card.

    Returns:
        ("regional" or "global", context tier) to dimension to rate to the
        commercial regions publishing it.
    """
    dimensions = {dimension for _, dimension in model_card_prices.CARD_COLUMNS}
    grouped: dict[tuple[str, str], dict[str, dict[Decimal, list[str]]]] = {}
    for row in rows:
        region = str(row.get("region", ""))
        routing = str(row.get("routing", ""))
        if (
            row.get("tier") != "standard"
            or row.get("currency", "USD") != "USD"
            or row.get("cache_ttl")
            or row.get("spec")
            or row.get("dimension") not in dimensions
            or routing in _OTHER_ROUTINGS
            or not model_card_prices.is_commercial(region)
        ):
            continue
        try:
            rate = Decimal(str(row.get("unit_price")))
        except InvalidOperation:
            continue
        kind = "global" if routing == "global" else "regional"
        tier = str(row.get("context") or "")
        grouped.setdefault((kind, tier), {}).setdefault(
            str(row["dimension"]), {}
        ).setdefault(rate, []).append(region)
    return grouped


def _card_rates(
    prices: model_card_prices.CardPrices,
) -> dict[tuple[str, str], dict[str, Decimal]]:
    """Group a card's rates the way :func:`_gateway_rates` groups the gateway's.

    Args:
        prices: What the card publishes.

    Returns:
        ("regional" or "global", context tier) to dimension to rate. The
        regional rate is the In-Region row, or the Geo row of a card with none.
    """
    grouped: dict[tuple[str, str], dict[str, Decimal]] = {}
    for context in ("", "long"):
        regional = prices.rates.get(("in_region", context)) or prices.rates.get(
            ("geo", context)
        )
        if regional:
            grouped["regional", context] = dict(regional)
        if global_rates := prices.rates.get(("global", context)):
            grouped["global", context] = dict(global_rates)
    return grouped


def _per_million(rate: Decimal) -> str:
    """Render a per-token rate as the per-1M figure a card prints.

    Args:
        rate: The per-token rate.

    Returns:
        The per-1M figure, without trailing zeros.
    """
    return f"{(rate * 1_000_000).normalize():f}"


def _disagreements(
    model_id: str, page: str, prices: model_card_prices.CardPrices, rows: list[Any]
) -> list[str]:
    """Describe every rate on which a card and the gateway's price card differ.

    A card pricing a single context tier states that rate for every prompt
    size, so a long-context rate the gateway publishes is compared with the
    card's only one.

    Args:
        model_id: The catalogue model.
        page: The card's page name.
        prices: What the card publishes.
        rows: The gateway's price rows for the model.

    Returns:
        One line per disagreeing option, tier and dimension.
    """
    card = _card_rates(prices)
    gateway = _gateway_rates(row for row in rows if isinstance(row, dict))
    if not gateway:
        return [f"{model_id}: gateway publishes no price, {page} prices it"]
    split = any(tier == "long" for _, tier in card)
    lines = []
    for kind, tier in sorted({*card, *gateway}):
        expected = card.get((kind, tier if split else ""), {})
        published = gateway.get((kind, tier), {})
        for dimension in sorted({*expected, *published}):
            want = expected.get(dimension)
            found = published.get(dimension, {})
            if set(found) == ({want} if want is not None else set()):
                continue
            label = f"{_OPTION_LABEL[kind]}{', long context' if tier else ''}"
            gateway_side = (
                "; ".join(
                    f"{_per_million(rate)}/1M in {', '.join(sorted(regions))}"
                    for rate, regions in sorted(found.items())
                )
                or "nothing"
            )
            card_side = "nothing" if want is None else f"{_per_million(want)}/1M"
            lines.append(
                f"{model_id} ({label}) {dimension}: gateway publishes {gateway_side}, "
                f"{page} states {card_side}"
            )
    return lines


def price_disagreements(
    price_cards: Mapping[str, Mapping[str, Any]], *, refresh: bool = False
) -> list[str]:
    """Compare every priced model card with what the gateway publishes.

    Nothing here is published: the gateway stays the source of every price
    on the page, and a card that disagrees with it is news for a person --
    either AWS changed a rate, or the gateway bills one wrongly.

    Args:
        price_cards: Model ID to its full ``model_pricing`` card.
        refresh: Ignore any cached snapshot.

    Returns:
        One line per disagreement, and one per priced card that could not be
        read.
    """
    catalogue = set(price_cards)
    lines: list[str] = []
    for card in _snapshot_cards(refresh=refresh):
        page = str(card.get("page"))
        matched, problem = _claimed(card, catalogue)
        if problem in ("unmatched", "ambiguous"):
            continue
        if "price_problem" in card:
            lines.append(f"{page}: pricing unreadable: {card['price_problem']}")
            continue
        if "prices" not in card:
            continue
        try:
            prices = model_card_prices.prices_from_json(card["prices"])
        except model_card_prices.UnreadableSourceError as error:
            lines.append(f"{page}: pricing unreadable: {error}")
            continue
        for model_id in sorted(matched):
            rows = price_cards[model_id].get("prices")
            lines.extend(
                _disagreements(
                    model_id, page, prices, rows if isinstance(rows, list) else []
                )
            )
    return lines
