"""Writes the generated blocks of ``docs/models.md``.

Three things on the page have to be Markdown rather than data the table script
renders. The snapshot stamp and the sources section must be readable without
JavaScript, because they carry the licence attributions the sources require. The
model list must exist in the page source at all, because MkDocs' search index
and ``docs_hooks/trademarks.py`` both read the Markdown and neither can see a
row that only exists inside a JSON file.
"""

from __future__ import annotations

import re
from html import escape
from typing import TYPE_CHECKING

from docs_gen.model_catalog.config import REPO_ROOT
from docs_gen.model_catalog.gateway import UNKNOWN_VERSION

if TYPE_CHECKING:
    from pathlib import Path

    from docs_gen.model_catalog.schema import Catalog

#: Page whose generated blocks this module owns.
PAGE_PATH: Path = REPO_ROOT / "docs" / "models.md"

#: Marker pairs delimiting each generated block.
_BLOCKS: tuple[str, ...] = ("generated", "noscript", "sources", "providers")


class TrademarkCoverageError(RuntimeError):
    """A provider in the catalogue has no trademark registry entry."""


def _replace(text: str, block: str, body: str) -> str:
    """Replace one marked block of the page.

    Args:
        text: Current page source.
        block: Block name, without the ``catalog:`` prefix.
        body: Replacement body.

    Returns:
        The page with that block replaced.

    Raises:
        ValueError: The markers are missing from the page.
    """
    pattern = re.compile(
        rf"(<!-- catalog:{block} -->).*?(<!-- /catalog:{block} -->)", re.DOTALL
    )
    if not pattern.search(text):
        msg = f"docs/models.md has no <!-- catalog:{block} --> block"
        raise ValueError(msg)
    return pattern.sub(
        lambda match: f"{match.group(1)}\n{body}\n{match.group(2)}", text
    )


def _generated_block(catalog: Catalog) -> str:
    """Render the snapshot stamp.

    Args:
        catalog: The generated catalogue.

    Returns:
        Markdown for the stamp admonition.
    """
    manifest = catalog.manifest
    retired = (
        f" {manifest.retired_models} model(s) AWS no longer lists are kept, tagged"
        " `delisted`."
        if manifest.retired_models
        else ""
    )
    source = (
        f"stdapi.ai {manifest.gateway_version}"
        if manifest.gateway_version and manifest.gateway_version != UNKNOWN_VERSION
        else "a stdapi.ai instance"
    )
    return (
        f"Snapshot taken on **{manifest.generated}** from "
        f"{source}, covering {len(catalog.models)} models across "
        f"{len(manifest.regions)} AWS regions.{retired} Prices and availability "
        "move — before you commit to a number, confirm it against the "
        "[Amazon Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) "
        "and your own [`search_models`](api_search_models.md)."
    )


#: What one row of a source counts, where "entries" would mislead.
_UNITS: dict[str, str] = {"aws_model_cards": "model cards", "models_dev": "models"}


def _sources_block(catalog: Catalog) -> str:
    """Render the sources section, including the attributions the licences require.

    Args:
        catalog: The generated catalogue.

    Returns:
        Markdown for the sources section.
    """
    lines = [
        (
            "Every number on this page comes from one of the sources below, "
            "reproduced unmodified. Mapping each entry onto an Amazon Bedrock "
            "model ID is our own work, and any error in that mapping is ours, "
            "not the source's."
        ),
        "",
        "| Source | Licence | Read on | Used here |",
        "| --- | --- | --- | --- |",
        (
            "| The gateway's own [`search_models`](api_search_models.md) and "
            "[`model_pricing`](api_model_pricing.md) | — | "
            f"{catalog.manifest.generated} | {len(catalog.models)} |"
        ),
        (
            "| [Amazon Bedrock `ListFoundationModels`]"
            "(https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListFoundationModels.html), "
            "read raw so its undocumented fields survive "
            f"| — | {catalog.manifest.generated} "
            "| capabilities, APIs, media types, limits |"
        ),
    ]
    # "Read on" is when the source last published, when it says; otherwise the
    # day this snapshot fetched it, which is the honest answer for a source
    # that dates nothing.
    lines.extend(
        f"| [{source.name}]({source.url}) "
        f"| [{source.licence}]({source.licence_url}) "
        f"| {source.as_of or catalog.manifest.generated} "
        f"| {source.matched} of {source.rows} {_UNITS.get(source.key, 'entries')} |"
        for source in catalog.manifest.sources
    )
    lines.append("")
    lines.extend(
        f"- **{source.name}** — {source.attribution}"
        for source in catalog.manifest.sources
    )
    lines.extend(
        [
            "",
            (
                "Some of what the table shows comes from parts of "
                "`ListFoundationModels` AWS does not document, so AWS may stop "
                "returning them at any time. A regeneration updates the published "
                "data set rather than replacing it: the last known value of such a "
                "field is kept, and a model AWS stops listing stays in the table, "
                "tagged `delisted`, with the date it was last seen."
            ),
        ]
    )
    return "\n".join(lines)


def _providers_block(catalog: Catalog) -> str:
    """Render the line that attributes the marks the table displays.

    The table shows a hundred model names and twenty brand logos, all rendered
    from JSON. ``docs_hooks/trademarks.py`` derives a page's attributions by
    matching its registry against the page *Markdown*, so without this line the
    page would show every vendor's mark and credit none of them.

    Args:
        catalog: The generated catalogue.

    Returns:
        One Markdown paragraph naming every mark the table can show.
    """
    from docs_hooks.trademarks import _COMPILED  # noqa: PLC0415

    providers = {model.provider.casefold() for model in catalog.models}
    labels = sorted(
        {
            entry.label
            for entry, _ in _COMPILED
            if providers & {mark.casefold() for mark in entry.marks}
        }
    )
    return (
        "The model names and brand logos in the table above are trademarks of "
        "their respective owners: " + "; ".join(labels) + "."
    )


def _noscript_block(catalog: Catalog) -> str:
    """Render the catalogue as plain HTML for readers and crawlers without JS.

    The interactive table is built from JSON, so without this the page has no
    content at all: nothing for MkDocs' own search to index, nothing for a
    crawler, and a blank gap for anyone whose script failed to load. It sits
    inside ``<noscript>``, so it costs a reader with JavaScript nothing.

    Args:
        catalog: The generated catalogue.

    Returns:
        An HTML table of every model.
    """
    rows = [
        (
            "<table><thead><tr>"
            "<th>Model</th><th>ID</th><th>Provider</th>"
            "<th>Input</th><th>Output</th><th>Regions</th>"
            "</tr></thead><tbody>"
        )
    ]
    rows.extend(
        "<tr>"
        f"<td>{escape(model.name)}{' (legacy)' if model.legacy else ''}</td>"
        f"<td><code>{escape(model.id)}</code></td>"
        f"<td>{escape(model.provider)}</td>"
        f"<td>{escape(', '.join(model.input_modalities))}</td>"
        f"<td>{escape(', '.join(model.output_modalities))}</td>"
        f"<td>{len(model.regions)}</td>"
        "</tr>"
        for model in catalog.models
    )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def check_trademarks(catalog: Catalog, markdown: str) -> list[str]:
    """Report catalogue providers the trademark registry does not cover.

    The site attributes marks by matching the registry's patterns against a
    page's Markdown, so a provider that appears only inside the JSON artefact
    would ship unattributed. This runs at generation time so the omission
    surfaces before the build.

    Args:
        catalog: The generated catalogue.
        markdown: The page source the generator is about to write.

    Returns:
        Providers the page would show without attributing them, sorted. A
        provider is covered when some registry entry both claims its name as a
        mark and has a pattern the page actually triggers.

    Raises:
        TrademarkCoverageError: The registry could not be read.
    """
    try:
        from docs_hooks.trademarks import _COMPILED  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - the hook is part of the repo
        msg = f"the trademark registry could not be imported: {error}"
        raise TrademarkCoverageError(msg) from error

    triggered = {
        mark.casefold()
        for entry, pattern in _COMPILED
        if pattern.search(markdown)
        for mark in entry.marks
    }
    return [
        provider
        for provider in sorted({model.provider for model in catalog.models})
        if provider.casefold() not in triggered
    ]


def write(catalog: Catalog) -> None:
    """Fill every generated block of the Models page.

    Args:
        catalog: The generated catalogue.

    Raises:
        TrademarkCoverageError: A provider would ship unattributed.
    """
    text = PAGE_PATH.read_text(encoding="utf-8")
    bodies = {
        "generated": _generated_block(catalog),
        "noscript": _noscript_block(catalog),
        "sources": _sources_block(catalog),
        "providers": _providers_block(catalog),
    }
    for block in _BLOCKS:
        text = _replace(text, block, bodies[block])
    missing = check_trademarks(catalog, text)
    if missing:
        msg = (
            "no trademark registry entry covers "
            + ", ".join(missing)
            + "; add one to docs_hooks/trademarks.py"
        )
        raise TrademarkCoverageError(msg)
    PAGE_PATH.write_text(text, encoding="utf-8")
