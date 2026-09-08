"""Tests for the icon-shortcode check that runs during the documentation build.

An icon name that no installed set provides is not an error anywhere else: the
shortcode is well-formed, the page builds, and ``mkdocs build --strict`` stays
green because it promotes broken links and missing nav entries, not unknown
icons. ``pymdownx.emoji`` simply leaves the text as it found it, so the reader
sees ``:material-key-multiple:`` spelled out. Six shortcodes reached the
published site that way.

Ref: docs_hooks/check_icons.py
     https://squidfunk.github.io/mkdocs-material/reference/icons-emojis/
"""

from types import SimpleNamespace

import pytest

# The icon sets ship with mkdocs-material, which is the `docs` group rather than
# the `test` one, so the offline lane has nothing to check against.
pytest.importorskip("material", reason="mkdocs-material is not installed")

from mkdocs.exceptions import PluginError

from docs_hooks.check_icons import (
    _AVAILABLE,
    _UNRESOLVED,
    on_page_markdown,
    on_post_build,
)


def _page(src_uri: str = "page.md") -> SimpleNamespace:
    """Build the minimum of a MkDocs page that the hook reads.

    Args:
        src_uri: Source path the hook names in its failure.

    Returns:
        An object exposing ``file.src_uri``.
    """
    return SimpleNamespace(file=SimpleNamespace(src_uri=src_uri))


@pytest.fixture(autouse=True)
def _clean_collection() -> None:
    """Keep one test's findings out of the next one's, and out of a real build."""
    _UNRESOLVED.clear()


class TestAvailableIcons:
    """What the hook checks against comes from the installed sets."""

    def test_every_set_the_shortcode_pattern_accepts_is_present(self) -> None:
        """A set the pattern matches but that ships no icon would fail every page.

        Ref: docs_hooks/check_icons.py
        """
        prefixes = {name.split("-")[0] for name in _AVAILABLE}
        assert {"material", "fontawesome", "octicons", "simple"} <= prefixes

    def test_the_icon_directory_was_actually_found(self) -> None:
        """An empty set would silently pass every page instead of failing it.

        Ref: docs_hooks/check_icons.py
        """
        assert len(_AVAILABLE) > 1000


class TestUnresolvedIcons:
    """A name no set provides fails the build, and says where it is."""

    def test_an_unknown_icon_is_reported(self) -> None:
        """The exact shortcode that shipped is the one this pins.

        Ref: docs/operations_iam_permissions.md
        """
        on_page_markdown(
            "## :material-key-multiple: Tenant API Key Delivery",
            _page("operations_iam_permissions.md"),
        )
        assert _UNRESOLVED == ["operations_iam_permissions.md: :material-key-multiple:"]

    def test_a_known_icon_is_not_reported(self) -> None:
        """The replacement resolves, so the same heading passes.

        Ref: docs_hooks/check_icons.py
        """
        on_page_markdown("## :material-account-key: Tenant API Key Delivery", _page())
        assert _UNRESOLVED == []

    @pytest.mark.parametrize(
        "markdown",
        [
            "`:material-key-multiple:`",
            "```\n:material-key-multiple:\n```",
            "~~~\n:material-key-multiple:\n~~~",
        ],
        ids=["inline", "fenced-backticks", "fenced-tildes"],
    )
    def test_a_quoted_shortcode_is_left_alone(self, markdown: str) -> None:
        """Code spans name icons to document the convention, not to render them.

        Ref: docs_hooks/check_icons.py
        """
        on_page_markdown(markdown, _page())
        assert _UNRESOLVED == []

    def test_an_emoji_shortcode_is_left_alone(self) -> None:
        """Twemoji resolves these from an index, not from a file in the icon sets.

        Ref: https://squidfunk.github.io/mkdocs-material/reference/icons-emojis/
        """
        on_page_markdown(":smile: :+1: :warning:", _page())
        assert _UNRESOLVED == []


class TestBuildOutcome:
    """The findings reach the operator as a failed build, not a log line."""

    def test_the_build_fails_and_names_every_page(self) -> None:
        """One build reports them all, rather than stopping at the first.

        Ref: docs_hooks/check_icons.py
        """
        on_page_markdown(":material-firewall:", _page("a.md"))
        on_page_markdown(":material-loopback:", _page("b.md"))
        with pytest.raises(PluginError) as failure:
            on_post_build()
        message = str(failure.value)
        assert "a.md: :material-firewall:" in message
        assert "b.md: :material-loopback:" in message

    def test_a_clean_build_raises_nothing(self) -> None:
        """A build with nothing to report raises nothing.

        Ref: docs_hooks/check_icons.py
        """
        on_page_markdown(":material-account-key:", _page())
        on_post_build()
