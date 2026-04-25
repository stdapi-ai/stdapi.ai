"""MkDocs hooks for Markdown for Agents support.

Generates agent-friendly markdown versions of documentation pages,
llms.txt index, and llms-full.txt concatenated content during the build.
"""

import re
from pathlib import Path
from typing import Any

from yaml import dump, safe_load

#: Frontmatter keys that are MkDocs-specific and meaningless to agents.
_MKDOCS_ONLY_KEYS: frozenset[str] = frozenset({"hide"})

#: Pages excluded from markdown output (no useful textual content).
_EXCLUDED_PAGES: frozenset[str] = frozenset({"api_reference.md"})

#: Regex matching a YAML frontmatter block at the start of a file.
_FRONTMATTER_RE: re.Pattern[str] = re.compile(r"\A---\n(.*?\n)---\n", re.DOTALL)

#: Type alias for a nav section: (section_name, [(title, url, src_path), ...]).
type _Section = tuple[str, list[tuple[str, str, str]]]


def _clean_frontmatter(content: str) -> tuple[str, str, dict[str, Any]]:
    """Strip MkDocs-only keys from YAML frontmatter.

    Args:
        content: Raw markdown file content with optional frontmatter.

    Returns:
        Tuple of (cleaned file content, body without frontmatter, metadata).
    """
    if not (match := _FRONTMATTER_RE.match(content)):
        return content, content, {}

    meta: dict[str, Any] = safe_load(match.group(1)) or {}
    body = content[match.end() :]

    if agent_meta := {k: v for k, v in meta.items() if k not in _MKDOCS_ONLY_KEYS}:
        fm = dump(agent_meta, sort_keys=False, allow_unicode=True).rstrip("\n")
        return f"---\n{fm}\n---\n{body}", body, meta
    return body, body, meta


def _walk_nav(
    nav: list[Any], site_url: str, *, section_name: str = ""
) -> list[_Section]:
    """Recursively walk MkDocs nav to extract flat sections with page entries.

    Args:
        nav: The ``nav`` list (or sub-list) from MkDocs config.
        site_url: Site base URL with trailing slash.
        section_name: Current section name (empty for top-level).

    Returns:
        Flat list of ``(section_name, [(title, url, src_path), ...])`` tuples.
    """
    sections: list[_Section] = []
    pages: list[tuple[str, str, str]] = []

    for item in nav:
        if not isinstance(item, dict):
            continue
        for title, value in item.items():
            match value:
                case str() if value in _EXCLUDED_PAGES and section_name:
                    pages.append(
                        ("API Reference — OpenAPI Spec", f"{site_url}openapi.yml", "")
                    )
                case str() if value not in _EXCLUDED_PAGES:
                    pages.append((title, f"{site_url}md/{value}", value))
                case list():
                    sections.extend(_walk_nav(value, site_url, section_name=title))

    if pages:
        sections.insert(0, (section_name, pages))
    return sections


def _generate_llms_txt(
    site_name: str,
    site_description: str,
    sections: list[_Section],
    meta_by_src: dict[str, dict[str, Any]],
) -> str:
    """Generate llms.txt content following the llmstxt.org specification.

    Args:
        site_name: Site display name.
        site_description: Site description for the blockquote.
        sections: Nav sections from ``_walk_nav``.
        meta_by_src: Mapping of source path to parsed frontmatter metadata.

    Returns:
        Complete llms.txt content.
    """
    lines = [f"# {site_name}", "", f"> {site_description}", ""]

    for section_name, pages in sections:
        if section_name:
            lines.extend((f"## {section_name}", ""))
        for title, url, src in pages:
            entry = f"- [{title}]({url})"
            if desc := meta_by_src.get(src, {}).get("description", ""):
                entry += f": {desc}"
            lines.append(entry)
        lines.append("")

    return "\n".join(lines)


def _generate_llms_full_txt(
    site_name: str,
    site_description: str,
    sections: list[_Section],
    bodies_by_src: dict[str, str],
) -> str:
    """Generate llms-full.txt with all markdown content concatenated.

    Args:
        site_name: Site display name.
        site_description: Site description for the blockquote.
        sections: Nav sections from ``_walk_nav``.
        bodies_by_src: Mapping of source path to body content (no frontmatter).

    Returns:
        Concatenated markdown content.
    """
    parts = [f"# {site_name}", "", f"> {site_description}", ""]

    for _section_name, pages in sections:
        for title, url, src in pages:
            if body := bodies_by_src.get(src, ""):
                parts.extend(
                    ("---", "", f"## {title}", f"Source: {url}", "", body.strip(), "")
                )

    return "\n".join(parts)


def on_post_build(config: Any) -> None:  # noqa: ANN401
    """Generate agent-friendly markdown files, llms.txt, and llms-full.txt.

    Copies source markdown files with cleaned frontmatter to ``site/md/``,
    then generates index files for agent discovery.

    Args:
        config: MkDocs configuration object.
    """
    docs_dir = Path(config["docs_dir"])
    site_dir = Path(config["site_dir"])
    md_dir = site_dir / "md"
    md_dir.mkdir(exist_ok=True)

    meta_by_src: dict[str, dict[str, Any]] = {}
    bodies_by_src: dict[str, str] = {}

    for src_file in sorted(docs_dir.glob("*.md")):
        if (name := src_file.name) not in _EXCLUDED_PAGES:
            cleaned, body, meta = _clean_frontmatter(
                src_file.read_text(encoding="utf-8")
            )
            meta_by_src[name] = meta
            bodies_by_src[name] = body
            (md_dir / name).write_text(cleaned, encoding="utf-8")

    site_url = config["site_url"].rstrip("/") + "/"
    sections = _walk_nav(config["nav"], site_url)
    site_name: str = config["site_name"]
    site_description: str = config["site_description"]

    (site_dir / "llms.txt").write_text(
        _generate_llms_txt(site_name, site_description, sections, meta_by_src),
        encoding="utf-8",
    )
    (site_dir / "llms-full.txt").write_text(
        _generate_llms_full_txt(site_name, site_description, sections, bodies_by_src),
        encoding="utf-8",
    )
