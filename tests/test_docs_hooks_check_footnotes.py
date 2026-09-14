"""Tests for the footnote check that runs during the documentation build.

A footnote marker whose definition is missing is not an error anywhere else:
``mkdocs build --strict`` promotes broken links and missing nav entries, and
python-markdown simply leaves the marker as it found it, so the reader sees
``[^39]`` spelled out in the middle of a sentence. The mirror case is as quiet:
a definition nobody references is still rendered at the foot of the page, with a
back-reference pointing at an anchor that does not exist.

Both matter most on the comparison page, where every competitor claim hangs off
a numbered source and the numbers are edited by hand.

Ref: docs_hooks/check_footnotes.py
     https://python-markdown.github.io/extensions/footnotes/
"""

from types import SimpleNamespace

import pytest
from mkdocs.exceptions import PluginError

from docs_hooks.check_footnotes import (
    _UNMATCHED,
    on_page_markdown,
    on_post_build,
    unmatched,
)
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.local


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
    _UNMATCHED.clear()


class TestMatching:
    """What a page has to hold for every footnote on it to render."""

    def test_a_marker_without_a_definition_is_reported(self) -> None:
        """It reaches the reader as literal text, which no other check catches.

        Ref: https://python-markdown.github.io/extensions/footnotes/
        """
        assert unmatched("Served on the proxy only. [^39]\n") == (["39"], [])

    def test_a_definition_without_a_marker_is_reported(self) -> None:
        """It renders a numbered note whose back-reference anchor does not exist.

        Ref: https://python-markdown.github.io/extensions/footnotes/
        """
        assert unmatched("Nothing points here.\n\n[^39]: An orphan source\n") == (
            [],
            ["39"],
        )

    def test_a_marker_and_its_definition_pass(self) -> None:
        """The shape every footnote on a published page has.

        Ref: docs/compare.md
        """
        assert unmatched("Beta, and needs a database. [^39]\n\n[^39]: Source\n") == (
            [],
            [],
        )

    def test_a_marker_used_twice_passes(self) -> None:
        """One source is cited from a table cell and from the prose below it.

        Ref: docs/compare.md
        """
        assert unmatched("Cell [^40] and prose [^40].\n\n[^40]: Source\n") == ([], [])

    def test_a_definition_may_cite_another_footnote(self) -> None:
        """A source that refers to a neighbouring one is still a reference.

        Ref: docs/compare.md
        """
        assert unmatched("Text [^1]\n\n[^1]: see [^2]\n[^2]: Source\n") == ([], [])

    @pytest.mark.parametrize(
        "markdown",
        [
            "A character class `[^>]` matches anything but a bracket.",
            "```\nre.compile(r'[^A-Za-z0-9]')\n```",
            "~~~\nre.compile(r'[^A-Za-z0-9]')\n~~~",
        ],
        ids=["inline", "fenced-backticks", "fenced-tildes"],
    )
    def test_a_character_class_in_code_is_left_alone(self, markdown: str) -> None:
        """A negated character class is spelled exactly like a footnote marker.

        Ref: docs_hooks/check_claims.py
        """
        assert unmatched(markdown) == ([], [])

    def test_every_finding_is_reported_in_page_order(self) -> None:
        """One pass reports them all, so a page is fixed in one edit.

        Ref: docs_hooks/check_footnotes.py
        """
        assert unmatched("[^9] [^7]\n\n[^3]: a\n[^1]: b\n") == (["9", "7"], ["3", "1"])


class TestPublishedPages:
    """The pages as they ship, which is what the build renders."""

    def test_no_published_page_carries_an_unmatched_footnote(self) -> None:
        """The comparison page's forty sources are numbered and edited by hand.

        Ref: docs/compare.md
        """
        findings = {
            path.relative_to(REPO_ROOT).as_posix(): unmatched(
                path.read_text(encoding="utf-8")
            )
            for path in sorted((REPO_ROOT / "docs").rglob("*.md"))
        }
        assert {
            page: found for page, found in findings.items() if found != ([], [])
        } == {}


class TestBuildOutcome:
    """The findings reach the author as a failed build, not a log line."""

    def test_the_build_fails_and_names_every_page(self) -> None:
        """One build reports them all, rather than stopping at the first.

        Ref: docs_hooks/check_footnotes.py
        """
        on_page_markdown("A claim [^39]", _page("compare.md"))
        on_page_markdown("[^12]: An orphan source", _page("features.md"))
        with pytest.raises(PluginError) as failure:
            on_post_build()
        message = str(failure.value)
        assert "compare.md: [^39] has no definition" in message
        assert "features.md: [^12] is defined but never cited" in message

    def test_a_clean_build_raises_nothing(self) -> None:
        """A build with nothing to report raises nothing.

        Ref: docs_hooks/check_footnotes.py
        """
        on_page_markdown("A claim [^39]\n\n[^39]: Source", _page())
        on_post_build()
