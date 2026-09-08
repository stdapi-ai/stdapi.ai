"""MkDocs hook failing the build on an icon shortcode that does not resolve.

``pymdownx.emoji`` leaves a shortcode it cannot resolve as literal text, so
``:material-key-multiple:`` reaches the reader spelled out in the middle of a
heading. Nothing else notices: the name is well-formed, the page builds, and
``--strict`` stays green -- it promotes broken links and missing nav entries,
not unknown icons. Six of them were published this way.

The available names are read from the icon sets mkdocs-material ships, which is
the same source the emoji index resolves against, so a set added or renamed
upstream needs no change here.
"""

import logging
import re
from pathlib import Path
from typing import Any

import material
from mkdocs.exceptions import PluginError

#: Where mkdocs-material keeps one SVG per icon, one directory per set.
_ICONS_DIR = Path(material.__file__).parent / "templates" / ".icons"

#: Shortcodes of the icon sets, which are the ones resolved from a file on disk.
#: Every other shortcode is an emoji, which the Twemoji index resolves instead
#: and which this hook leaves alone.
_ICON_RE = re.compile(r":(material|fontawesome|octicons|simple)-[a-z0-9-]+:")

#: Fenced and inline code, where a shortcode is quoted rather than rendered --
#: the pages documenting this convention name icons that need not exist.
_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]*`", re.DOTALL)

_log = logging.getLogger("mkdocs.hooks.check_icons")

#: Every icon name the installed sets provide, as its shortcode spelling.
_AVAILABLE = {
    str(svg.relative_to(_ICONS_DIR).with_suffix("")).replace("/", "-")
    for svg in _ICONS_DIR.rglob("*.svg")
}

#: One ``page: shortcode`` per unresolved icon, collected across the build.
_UNRESOLVED: list[str] = []


def on_page_markdown(
    markdown: str,
    page: Any,  # noqa: ANN401
    config: Any = None,  # noqa: ARG001,ANN401
    files: Any = None,  # noqa: ARG001,ANN401
) -> None:
    """Record every icon shortcode on one page that no installed set provides.

    Args:
        markdown: The page's Markdown source.
        page: The MkDocs page being rendered, named in the failure.
        config: MkDocs configuration object.
        files: The MkDocs file collection.
    """
    for match in _ICON_RE.finditer(_CODE_RE.sub("", markdown)):
        name = match.group().strip(":")
        if name not in _AVAILABLE:
            _UNRESOLVED.append(f"{page.file.src_uri}: {match.group()}")


def on_post_build(config: Any = None) -> None:  # noqa: ARG001,ANN401
    """Fail the build if any page named an icon that does not exist.

    Args:
        config: MkDocs configuration object.

    Raises:
        PluginError: If at least one shortcode went unresolved. Deferred to the
            end so one build reports every one of them rather than the first.
    """
    if not _UNRESOLVED:
        _log.info("Every icon shortcode resolves (%d available).", len(_AVAILABLE))
        return
    listed = "\n  ".join(sorted(set(_UNRESOLVED)))
    _UNRESOLVED.clear()
    msg = (
        f"{len(set(listed.splitlines()))} icon shortcode(s) name no installed "
        f"icon and would render as literal text:\n  {listed}\n"
        "Search the name at https://squidfunk.github.io/mkdocs-material/reference/icons-emojis/"
    )
    raise PluginError(msg)
