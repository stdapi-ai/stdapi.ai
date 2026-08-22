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
"""

from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING, Any

from docs_gen.model_catalog.http import get_bytes, map_concurrent
from docs_gen.model_catalog.sources import snapshot
from docs_gen.model_catalog.tokens import format_tokens, parse_tokens

if TYPE_CHECKING:
    from collections.abc import Iterable

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
    return {"page": page, "ids": sorted(set(_MODEL_ID.findall(body))), "facts": facts}


def _collect() -> list[dict[str, Any]]:
    """Read every model card.

    Returns:
        One entry per card that stated anything.

    Raises:
        RuntimeError: Too few cards yielded a context window, meaning the user
            guide's layout changed under the parser rather than AWS writing less.
    """
    cards = [card for card in map_concurrent(_read_card, _card_pages()) if card]
    found = sum(1 for card in cards if card.get("facts", {}).get("context_window"))
    if found < _CONTEXT_WINDOW_FLOOR:
        msg = (
            f"only {found} of {len(cards)} model cards state a context window, "
            f"under the floor of {_CONTEXT_WINDOW_FLOOR}; the user guide's layout "
            "likely changed"
        )
        raise RuntimeError(msg)
    return [card for card in cards if card.get("facts")]


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
    raw = snapshot("aws_model_cards", _collect, refresh=refresh)
    assert isinstance(raw, list)  # noqa: S101 -- snapshot round-trips its own JSON
    catalogue = set(known)
    facts: dict[str, dict[str, Any]] = {}
    unmatched: list[str] = []
    ambiguous: list[str] = []
    still_multi: list[str] = []

    for card in raw:
        page = str(card.get("page"))
        stated = card.get("facts") or {}
        matched = {
            plain
            for raw_id in card.get("ids", ())
            if (plain := _plain_id(str(raw_id))) in catalogue
        }
        if not matched:
            unmatched.append(page)
            continue
        if len(matched) > 1:
            # A card that names more than one catalogue model may be a table
            # of the model it describes plus a cross-referenced sibling; only
            # the IDs the page's own slug confirms are the card's own identity.
            confirmed = {model_id for model_id in matched if _confirms(page, model_id)}
            if not confirmed:
                ambiguous.append(f"{page} ({', '.join(sorted(matched))})")
                continue
            matched = confirmed
            if len(matched) > 1:
                still_multi.append(f"{page} ({', '.join(sorted(matched))})")
        contributed = {
            "context_window": format_tokens(stated.get("context_window")),
            "max_output_tokens": parse_tokens(stated.get("max_output_tokens")),
            "knowledge_cutoff": _date(stated.get("knowledge_cutoff", "")),
            "start_of_life": _date(stated.get("launch_date", "")),
            "end_of_life": _date(stated.get("eol_date", "")),
        }
        usable = {key: value for key, value in contributed.items() if value is not None}
        for model_id in matched:
            facts.setdefault(model_id, {}).update(usable)

    notes = []
    if unmatched:
        notes.append(
            f"{len(unmatched)} model card(s) describe no model this gateway serves"
        )
    if ambiguous:
        notes.append(
            "model card(s) name more than one catalogue model with no way to "
            f"tell which the page describes, so no facts were attached: "
            f"{'; '.join(ambiguous)}"
        )
    if still_multi:
        notes.append(
            f"model card(s) confirm more than one catalogue model as their own: "
            f"{'; '.join(still_multi)}"
        )
    return facts, notes
