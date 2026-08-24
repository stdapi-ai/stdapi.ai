"""MkDocs hook emitting an RSS 2.0 feed for the blog.

Material's blog plugin does not ship a feed. This writes one from the post
sources at build time, so syndication targets that import by feed receive each
post with its canonical URL already pointing at this site.
"""

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from yaml import safe_load

#: Feed path, matching the name mkdocs-rss-plugin publishes.
_FEED_NAME = "feed_rss_created.xml"

#: Most recent posts to include.
_MAX_ITEMS = 20

#: Day and month names, spelled out rather than taken from the build locale,
#: because RFC 822 requires English regardless of where the build runs.
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)  # fmt: skip


def _rfc822(moment: datetime) -> str:
    """Format a datetime as an RFC 822 date, as RSS requires.

    Args:
        moment: The datetime to format. Naive values are read as UTC.

    Returns:
        The formatted date, for example ``Fri, 19 Dec 2025 00:00:00 +0000``.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    moment = moment.astimezone(UTC)
    return (
        f"{_DAYS[moment.weekday()]}, {moment.day:02d} {_MONTHS[moment.month - 1]} "
        f"{moment.year} {moment:%H:%M:%S} +0000"
    )


def _created(meta: dict[str, Any]) -> datetime | None:
    """Read a post's creation date from its frontmatter.

    Args:
        meta: Parsed frontmatter. ``date`` may be a scalar or a mapping
            carrying ``created``, both of which Material accepts.

    Returns:
        The creation datetime, or ``None`` if the frontmatter carries none.
    """
    value = meta.get("date")
    if isinstance(value, dict):
        value = value.get("created")
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    return None


def _item(post: dict[str, Any], site_url: str) -> str:
    """Render one post as an RSS ``<item>``.

    Args:
        post: Post metadata carrying ``slug``, ``title``, ``description``,
            ``categories`` and ``created``.
        site_url: Site base URL with a trailing slash.

    Returns:
        The serialised ``<item>`` element.
    """
    url = f"{site_url}blog/{post['slug']}/"
    parts = [
        "    <item>",
        f"      <title>{escape(post['title'])}</title>",
        f"      <link>{escape(url)}</link>",
        f'      <guid isPermaLink="true">{escape(url)}</guid>',
        f"      <pubDate>{_rfc822(post['created'])}</pubDate>",
    ]
    if description := post.get("description"):
        parts.append(f"      <description>{escape(description)}</description>")
    parts.extend(
        f"      <category>{escape(category)}</category>"
        for category in post.get("categories") or ()
    )
    parts.append("    </item>")
    return "\n".join(parts)


def on_post_build(config: Any) -> None:  # noqa: ANN401
    """Write the blog RSS feed into the built site.

    Posts dated in the future are skipped, matching the blog plugin's
    ``draft_if_future_date`` behaviour — they are not in the build either.

    Args:
        config: MkDocs configuration object.
    """
    posts_dir = Path(config["docs_dir"]) / "blog" / "posts"
    if not posts_dir.is_dir():
        return

    site_url = config["site_url"].rstrip("/") + "/"
    now = datetime.now(UTC)
    posts: list[dict[str, Any]] = []

    for source in posts_dir.glob("*.md"):
        text = source.read_text(encoding="utf-8")
        if not text.startswith("---\n") or "\n---\n" not in text:
            continue
        meta = safe_load(text.split("\n---\n", 1)[0][4:]) or {}
        created = _created(meta)
        if created is None or created > now:
            continue
        posts.append(
            {
                "slug": meta.get("slug") or source.stem,
                "title": meta.get("title") or source.stem,
                "description": meta.get("description"),
                "categories": meta.get("categories"),
                "created": created,
            }
        )

    posts.sort(key=lambda post: post["created"], reverse=True)
    items = "\n".join(_item(post, site_url) for post in posts[:_MAX_ITEMS])
    feed_url = f"{site_url}{_FEED_NAME}"
    built = _rfc822(posts[0]["created"] if posts else now)

    feed = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>{escape(config["site_name"])} Blog</title>
    <link>{escape(site_url)}blog/</link>
    <description>{escape(config["site_description"])}</description>
    <language>en</language>
    <lastBuildDate>{built}</lastBuildDate>
    <atom:link href="{escape(feed_url)}" rel="self" type="application/rss+xml"/>
{items}
  </channel>
</rss>
"""
    (Path(config["site_dir"]) / _FEED_NAME).write_text(feed, encoding="utf-8")
