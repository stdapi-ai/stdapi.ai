"""Tests for the documentation claim check and the trademark registry it guards.

Ref: docs_hooks/check_claims.py
     docs_hooks/trademarks.py
     .claude/skills/stdapi-marketing/SKILL.md (banned claims)
"""

import re

from docs_hooks.check_claims import _EXCEPTIONS, _FLOORS, _RULES, _surfaces, scan
from docs_hooks.trademarks import _ASSET_RE, _ENTRY_BY_ASSET, _REGISTRY


def _rules(text: str, relative: str = "docs/page.md") -> set[str]:
    """Names of the rules that fire on a piece of Markdown.

    Args:
        text: The Markdown to scan.
        relative: Path the text is attributed to, as the rules see it.

    Returns:
        The rule names reported for the text.
    """
    return {
        detail.split("]")[0].removeprefix("[") for _, detail in scan(text, relative)
    }


class TestStalePhrasings:
    """Phrasings that were true at one release and stopped being true.

    Ref: docs_hooks/check_claims.py:_RULES
    """

    def test_a_dialect_count_of_three_is_caught(self) -> None:
        """The gateway serves four dialects, so "three dialects" is stale.

        Ref: docs/roadmap.md (v1.17.0 Ollama routes)
        """
        assert "dialect-count" in _rules("It speaks three API dialects today.")

    def test_a_dialect_list_without_ollama_is_caught(self) -> None:
        """An enumeration used as the dialect list must name Ollama.

        Ref: docs_hooks/check_claims.py:_RULES (dialect-list)
        """
        assert "dialect-list" in _rules(
            "the OpenAI, Anthropic and Cohere APIs your tools already speak"
        )

    def test_the_vendor_cross_validation_lane_is_not_a_dialect_list(self) -> None:
        """Three vendor APIs really are cross-validated, so that sentence stands.

        Ref: .github/workflows/test.yml:official-api
        """
        assert "dialect-list" not in _rules(
            "The same tests run against the real OpenAI, Anthropic and Cohere APIs."
        )

    def test_the_latest_version_line_is_caught_off_the_roadmap(self) -> None:
        """Exactly one page names the latest version.

        Ref: docs_hooks/check_claims.py:_RULES (latest-version)
        """
        assert "latest-version" in _rules("**Latest: v1.17.0** - what shipped.")

    def test_the_latest_version_line_is_allowed_on_the_roadmap(self) -> None:
        """The roadmap is the page that names the latest version.

        Ref: docs/roadmap.md
        """
        assert "latest-version" not in _rules(
            "**Latest: v1.17.0** - what shipped.", "docs/roadmap.md"
        )

    def test_a_release_entry_keeps_the_count_it_shipped_with(self) -> None:
        """Release entries are historical records and are never rewritten.

        Ref: docs_hooks/check_claims.py:_EXCEPTIONS
        """
        assert ("dialect-count", "docs/roadmap.md") in _EXCEPTIONS
        assert "dialect-count" not in _rules(
            "closed gaps across all three API dialects", "docs/roadmap.md"
        )


class TestMarkupSurfaces:
    """The HTML overrides and SVG assets a crawler reads are scanned like prose.

    Ref: docs_hooks/check_claims.py:_SURFACES
    """

    def test_the_html_overrides_and_svg_assets_are_scanned(self) -> None:
        """The JSON-LD and the social card carry claims Markdown never sees.

        Ref: docs/overrides/main.html (schema.org JSON-LD)
        """
        scanned = {path.name for path in _surfaces()}
        assert {"main.html", "og-image.svg", "README.md"} <= scanned

    def test_a_stale_dialect_list_in_the_json_ld_is_caught(self) -> None:
        """The structured data is what a search engine and an LLM read.

        Ref: docs/overrides/main.html (SoftwareApplication description)
        """
        assert "dialect-list" in _rules(
            '<script type="application/ld+json">{"description": '
            '"Self-hosted OpenAI, Anthropic and Cohere compatible API gateway"}'
            "</script>",
            "docs/overrides/main.html",
        )

    def test_an_entity_escaped_dialect_list_is_caught(self) -> None:
        """The social card spells the conjunction ``&amp;``, not "and".

        Ref: docs/styles/og-image.svg
        """
        assert "dialect-list" in _rules(
            '<text x="595" y="348">OpenAI, Anthropic &amp; Cohere compatible</text>',
            "docs/styles/og-image.svg",
        )

    def test_the_four_dialect_card_passes_through_the_escape(self) -> None:
        """Naming Ollama clears the rule whichever conjunction is spelled.

        Ref: docs/styles/og-image.svg (subtitle)
        """
        assert not _rules(
            '<text x="595" y="348">OpenAI, Anthropic, Cohere &amp; Ollama compatible'
            "</text>",
            "docs/styles/og-image.svg",
        )

    def test_markup_is_not_a_claim(self) -> None:
        """An id, a class, a CSS property or a URL is not something a reader reads.

        Ref: docs_hooks/check_claims.py:_text_content
        """
        assert not _rules(
            "<!-- seamless header -->\n"
            '<div id="powerful-grid" class="enterprise-grade">\n'
            '  <a href="https://example.test/zero-configuration">Read the docs</a>\n'
            "  <style>.hero { transition: all 0.3s; --seamless: 1 }</style>\n"
            "</div>",
            "docs/overrides/partials/hero.html",
        )

    def test_a_markup_hit_reports_the_line_it_sits_on(self) -> None:
        """Blanking markup keeps every line number the reader can act on.

        Ref: docs_hooks/check_claims.py:scan
        """
        text = (
            "<svg>\n  <title>x</title>\n  <text>100+ models,\n  0 errors</text>\n</svg>"
        )
        assert scan(text, "docs/styles/card.svg")[0][0] == 4


class TestBannedClaims:
    """The "never write" column of the marketing skill, enforced mechanically.

    Ref: .claude/skills/stdapi-marketing/SKILL.md (banned claims)
    """

    def test_a_stale_model_count_is_caught(self) -> None:
        """Counts drift; the published floor does not.

        Ref: .claude/skills/stdapi-marketing/SKILL.md (stale counts)
        """
        assert "stale-count" in _rules("Access 80+ LLM models from one endpoint.")

    def test_a_stale_count_is_caught_whatever_noun_follows_it(self) -> None:
        """The claim is the number, not the noun the copy happens to pick.

        Ref: .claude/skills/stdapi-marketing/SKILL.md (canonical facts)
        """
        assert "stale-count" in _rules("**80+ API operations** exposed as MCP tools")
        assert "stale-count" in _rules("80+ MCP API tools on every deployment")
        assert "stale-count" in _rules("Backed by 4,400+ tests.")

    def test_a_current_floor_is_not_a_stale_count(self) -> None:
        """A floor at or above the canonical value is what copy must publish.

        Ref: .claude/skills/stdapi-marketing/SKILL.md (canonical facts)
        """
        assert "stale-count" not in _rules(
            "100+ endpoints across four protocols, 100+ models, 8,000+ test cases"
        )

    def test_a_stale_count_in_markdown_bold_is_caught(self) -> None:
        """Markdown bold wraps a claim in underscores, which a word boundary reads as text.

        Ref: docs/index.md (``__6,000+ test cases__``)
        """
        assert "stale-count" in _rules("- :material-test-tube: __6,000+ test cases__")
        assert "stale-count" in _rules("__90+ endpoints__ across four protocols")

    def test_a_countable_noun_elsewhere_is_not_a_published_floor(self) -> None:
        """Only the counted claims of the marketing skill carry a floor.

        Ref: docs/features.md (60+ voices, 30+ languages)
        """
        assert "stale-count" not in _rules("60+ voices across 30+ languages")
        assert "stale-count" not in _rules("Agents that make 10+ tool calls per task")

    def test_every_counted_noun_has_a_floor(self) -> None:
        """A noun the pattern matches with no floor would crash the check.

        Ref: docs_hooks/check_claims.py:_below_floor
        """
        pattern = next(rule.pattern for rule in _RULES if rule.name == "stale-count")
        assert set(pattern.groupindex) - {"value"} == set(_FLOORS)

    def test_a_time_to_production_promise_is_caught(self) -> None:
        """Time to production depends on domain, certificate and model access.

        Ref: .claude/skills/stdapi-marketing/SKILL.md ("5 minutes to production")
        """
        assert "time-to-production" in _rules("Deploy in 5-15 minutes using Terraform.")

    def test_a_data_residency_absolute_is_caught(self) -> None:
        """S3 I/O, web grounding and Nova system tools egress.

        Ref: .claude/skills/stdapi-marketing/SKILL.md ("data never leaves")
        """
        assert "data-never-leaves" in _rules("Your data never leaves your AWS account.")

    def test_an_inherited_certification_is_caught(self) -> None:
        """Certifications belong to the AWS services, not to stdapi.ai.

        Ref: docs/operations_compliance.md (certifications are not inherited)
        """
        assert "inherited-compliance" in _rules(
            "The gateway inherits the AWS compliance certifications."
        )

    def test_a_zero_code_change_absolute_is_caught(self) -> None:
        """Adoption changes the base URL, and the model name where it differs.

        Ref: .claude/skills/stdapi-marketing/SKILL.md ("your code never changes")
        """
        assert "no-code-changes" in _rules("Zero code changes: your SDK works as-is.")

    def test_technical_prose_about_tools_is_not_a_compatibility_promise(self) -> None:
        """A restrictive clause about tools is not an absolute compatibility claim.

        Ref: docs/use_cases_coding_assistants.md
        """
        assert "absolute-compatibility" not in _rules(
            "Any tool using the Anthropic SDK can be configured the same way."
        )

    def test_every_rule_states_what_to_write_instead(self) -> None:
        """A failure that only forbids leaves the writer stuck.

        Ref: docs_hooks/check_claims.py:_Rule
        """
        assert all(rule.message and rule.name for rule in _RULES)


class TestTrademarkDetection:
    """Marks are derived from what a page names and from the logos it embeds.

    Ref: docs_hooks/trademarks.py:on_page_markdown
    """

    def test_the_coding_assistant_marks_are_registered(self) -> None:
        """The blog post names Cline, Cursor and Continue.dev.

        Ref: docs/blog/posts/bedrock-with-your-openai-code.md
        """
        owners = {entry.label: entry.owner for entry in _REGISTRY}
        assert owners["Cline"] == "Cline Bot Inc."
        assert owners["Cursor"] == "Anysphere, Inc."
        assert owners["Continue"] == "Continue Dev, Inc."

    def test_a_pagination_cursor_is_not_the_cursor_mark(self) -> None:
        """The API reference uses "Cursor" for pagination on pages Cursor is absent from.

        Ref: docs/api_anthropic_models.md (cursor-based pagination)
        """
        pattern = re.compile(
            next(entry.pattern for entry in _REGISTRY if entry.label == "Cursor")
        )
        assert not pattern.search("**Cursor-based pagination**: use `after_id`")
        assert not pattern.search("Cursor from the `next_page` field.")
        assert pattern.search("IDE coding agents (Cline, Cursor)")

    def test_a_raster_logo_is_checked_like_an_svg_one(self) -> None:
        """A logo shipped as PNG must be attributed too.

        Ref: docs/use_cases_home_assistant.md (styles/logo_wyoming.png)
        """
        assert _ASSET_RE.findall("![Wyoming](styles/logo_wyoming.png)") == ["wyoming"]
        assert "wyoming" in _ENTRY_BY_ASSET
