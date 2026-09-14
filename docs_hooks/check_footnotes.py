"""MkDocs hook failing the build on a footnote that does not render.

Footnotes are matched by hand, page by page, and neither half of a mismatch is
an error anywhere else. A marker whose definition is missing is left as literal
text, so ``[^39]`` reaches the reader in the middle of a sentence; a definition
nobody cites is still rendered at the foot of the page, with a back-reference
pointing at an anchor that was never written. ``--strict`` catches neither: it
promotes broken links and missing nav entries.

The comparison page is where this bites, because every competitor claim hangs
off a numbered source and the numbers are renumbered by hand as rows move.
"""

import logging
import re
from typing import Any

from mkdocs.exceptions import PluginError

#: A footnote marker, wherever the page cites one.
_REFERENCE_RE = re.compile(r"\[\^([^\]\s]+)\]")

#: A footnote definition, which opens its own line and ends in a colon.
_DEFINITION_RE = re.compile(r"^\[\^([^\]\s]+)\]:", re.MULTILINE)

#: Fenced and inline code, where a negated character class such as ``[^>]`` is
#: spelled exactly like a marker and means something else entirely.
_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]*`", re.DOTALL)

_log = logging.getLogger("mkdocs.hooks.check_footnotes")

#: One ``page: finding`` per footnote that would not render, across the build.
_UNMATCHED: list[str] = []


def unmatched(markdown: str) -> tuple[list[str], list[str]]:
    """Find the footnotes of one page that no counterpart completes.

    Args:
        markdown: The page's Markdown source.

    Returns:
        The identifiers cited with no definition, and those defined and never
        cited, each in the order the page writes them.
    """
    prose = _CODE_RE.sub("", markdown)
    defined = [match[1] for match in _DEFINITION_RE.finditer(prose)]
    # Dropping the opening marker keeps a definition from citing itself, while
    # a source pointing at a neighbouring one still counts.
    cited = [
        match[1] for match in _REFERENCE_RE.finditer(_DEFINITION_RE.sub("", prose))
    ]
    return (
        [name for name in dict.fromkeys(cited) if name not in defined],
        [name for name in defined if name not in cited],
    )


def on_page_markdown(
    markdown: str,
    page: Any,  # noqa: ANN401
    config: Any = None,  # noqa: ARG001,ANN401
    files: Any = None,  # noqa: ARG001,ANN401
) -> None:
    """Record every footnote on one page that would not render.

    Args:
        markdown: The page's Markdown source.
        page: The MkDocs page being rendered, named in the failure.
        config: MkDocs configuration object.
        files: The MkDocs file collection.
    """
    dangling, orphan = unmatched(markdown)
    _UNMATCHED.extend(
        f"{page.file.src_uri}: [^{name}] has no definition" for name in dangling
    )
    _UNMATCHED.extend(
        f"{page.file.src_uri}: [^{name}] is defined but never cited" for name in orphan
    )


def on_post_build(config: Any = None) -> None:  # noqa: ARG001,ANN401
    """Fail the build if any page carries a footnote that would not render.

    Args:
        config: MkDocs configuration object.

    Raises:
        PluginError: If at least one footnote went unmatched. Deferred to the
            end so one build reports every one of them rather than the first.
    """
    if not _UNMATCHED:
        _log.info("Every footnote marker resolves.")
        return
    listed = "\n  ".join(_UNMATCHED)
    count = len(_UNMATCHED)
    _UNMATCHED.clear()
    msg = (
        f"{count} footnote(s) would not render, and no other check sees them:"
        f"\n  {listed}\n"
        "Cite every definition once, define every marker on its own page, and "
        "wrap a literal bracket-caret in backticks so it reads as code."
    )
    raise PluginError(msg)
