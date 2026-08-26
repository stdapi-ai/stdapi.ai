"""Tests for the public Models page generator.

Everything here is offline: the matcher, the dataset shaping and the page
blocks are pure functions, and the committed artefact is checked as it stands
in the repository. Nothing contacts AWS, a gateway or a leaderboard.
"""

from __future__ import annotations

import gzip
import json
import os
import re
from typing import TYPE_CHECKING, Any
from urllib.request import Request

import pytest

from docs_gen.model_catalog import build, enrichment, http, merge, page, sources
from docs_gen.model_catalog.config import (
    DATA_DIR,
    ENRICHMENT_PATH,
    HEADLINE_DIMENSIONS,
    INDEX_GZIP_BUDGET,
    PROVENANCE_PATH,
    REPO_ROOT,
    SOURCES,
    region_bucket,
)
from docs_gen.model_catalog.gateway import UNKNOWN_VERSION
from docs_gen.model_catalog.matching import (
    CatalogModel,
    Matcher,
    _parse_answer,
    loose_forms,
    loose_keys,
    normalise,
    split_variant,
    strict_forms,
)
from docs_gen.model_catalog.page import PAGE_PATH
from docs_gen.model_catalog.schema import Catalog, Manifest, ModelRow, PriceGroup
from docs_gen.model_catalog.sources import (
    RawScore,
    SourceResult,
    aws_model_cards,
    models_dev,
)
from docs_gen.model_catalog.tokens import format_tokens, parse_tokens

if TYPE_CHECKING:
    from pathlib import Path


def score(
    name: str,
    *,
    organization: str = "",
    board: str = "text",
    **kwargs: Any,  # noqa: ANN401
) -> RawScore:
    """Build a leaderboard row for a test.

    Args:
        name: Model name as the source spells it.
        organization: Publisher the source attributes it to.
        board: Sub-leaderboard key.
        **kwargs: Any other :class:`RawScore` field.

    Returns:
        The row.
    """
    return RawScore(
        source="lmarena",
        board=board,
        metric="elo",
        label="Text Arena",
        value=kwargs.pop("value", 1200.0),
        name=name,
        organization=organization,
        as_of="2026-08-19",
        **kwargs,
    )


# --- normalisation --------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "claude-sonnet-4-5-20250929-v1-0",
        ),
        ("Claude Sonnet 4.5", "claude-sonnet-4-5"),
        ("mistralai/Voxtral-Mini-3B-2507", "voxtral-mini-3b-2507"),
        ("gpt-image-2 (medium)", "gpt-image-2"),
    ],
)
def test_normalise_reduces_a_name_to_its_core(value: str, expected: str) -> None:
    """Vendor namespaces and bracketed qualifiers are removed, identity is kept.

    Ref: docs_gen/model_catalog/matching.py
    """
    assert normalise(value) == expected


def test_normalise_keeps_the_release_stamp() -> None:
    """Two models differing only by release date must stay distinguishable.

    Dropping the stamp is what made Claude 3.5 Sonnet's two Bedrock IDs take
    each other's benchmark score.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
    """
    june = strict_forms("anthropic.claude-3-5-sonnet-20240620-v1:0")
    october = strict_forms("anthropic.claude-3-5-sonnet-20241022-v2:0")
    assert not june & october


def test_stripping_a_version_never_leaves_a_bare_vendor_name() -> None:
    """``deepseek-v3.2`` must not reduce to ``deepseek``.

    Every DeepSeek model would otherwise answer to the same key.

    Ref: docs_gen/model_catalog/matching.py
    """
    assert "deepseek" not in strict_forms("deepseek-v3.2")
    assert "deepseek" not in loose_forms("DeepSeek-V3-0324")
    assert "deepseek-v3-2" in strict_forms("deepseek-v3.2")


def test_loose_forms_drop_a_leading_vendor_token() -> None:
    """A source that prefixes the vendor still meets a Bedrock ID that does not.

    MTEB spells the model ``cohere-embed-english-v3``; Bedrock spells it
    ``cohere.embed-english-v3``. The two only meet once the vendor token can be
    dropped from either side.

    Ref: https://github.com/embeddings-benchmark/results
    """
    assert loose_forms("cohere-embed-english-v3") & loose_forms(
        "cohere.embed-english-v3"
    )
    assert "embed-english" in loose_forms("cohere-embed-english-v3")


@pytest.mark.parametrize(
    ("value", "identity", "variant"),
    [
        ("claude-opus-5-high", "claude-opus-5", ("high",)),
        ("deepseek-v3-1-thinking", "deepseek-v3-1", ("thinking",)),
        ("claude-opus-5", "claude-opus-5", ()),
    ],
)
def test_split_variant_separates_a_configuration_from_a_model(
    value: str, identity: str, variant: tuple[str, ...]
) -> None:
    """A reasoning effort names a configuration, not a different model.

    Ref: https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset
    """
    assert split_variant(value) == (identity, variant)


# --- rule matching --------------------------------------------------------


def test_dated_models_match_their_own_release() -> None:
    """Each Claude 3.5 Sonnet ID takes the score of its own release stamp.

    Ref: docs_gen/model_catalog/matching.py
    """
    matcher = Matcher()
    rows = [score("claude-3-5-sonnet-20240620"), score("claude-3-5-sonnet-20241022")]
    for model_id, expected in (
        ("anthropic.claude-3-5-sonnet-20240620-v1:0", "claude-3-5-sonnet-20240620"),
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", "claude-3-5-sonnet-20241022"),
    ):
        model = CatalogModel(
            id=model_id, name="Claude 3.5 Sonnet", provider="Anthropic"
        )
        decision, chosen = matcher.match_board(model, "lmarena", "text", rows)
        assert chosen is not None
        assert decision.matched_name == expected
        assert decision.method == "rule"


def test_an_ambiguous_loose_key_never_decides_a_match() -> None:
    """A loose key two catalogue models share may not match on its own.

    Only the loosened keys are guarded: two Bedrock IDs sharing a strict key are
    two IDs for one model, and both legitimately take its score.

    Ref: docs_gen/model_catalog/matching.py
    """
    model = CatalogModel(
        id="acme.thing-20250101", name="acme.thing-20250101", provider="Amazon"
    )
    rows = [score("thing")]
    assert Matcher().match_board(model, "lmarena", "text", rows)[1] is not None

    guarded = Matcher(ambiguous_keys=frozenset(loose_keys(model)))
    decision, chosen = guarded.match_board(model, "lmarena", "text", rows)
    assert chosen is None
    assert decision.matched_name is None


def test_a_plain_entry_beats_a_configuration_entry() -> None:
    """When both exist, the row naming no configuration wins.

    Ref: https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset
    """
    rows = [
        score("claude-opus-5-high", samples=99999),
        score("claude-opus-5", samples=10),
    ]
    model = CatalogModel(
        id="anthropic.claude-opus-5", name="Claude Opus 5", provider="Anthropic"
    )
    decision, _ = Matcher().match_board(model, "lmarena", "text", rows)
    assert decision.matched_name == "claude-opus-5"


def test_the_most_voted_configuration_wins_when_no_plain_entry_exists() -> None:
    """A disclosed configuration is published rather than nothing at all.

    Ref: https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset
    """
    rows = [
        score("claude-opus-5-high", samples=5000),
        score("claude-opus-5-max", samples=100),
    ]
    model = CatalogModel(
        id="anthropic.claude-opus-5", name="Claude Opus 5", provider="Anthropic"
    )
    decision, _ = Matcher().match_board(model, "lmarena", "text", rows)
    assert decision.matched_name == "claude-opus-5-high"


def test_a_different_publisher_is_never_matched() -> None:
    """An identically-named model from another vendor is not the same model.

    Ref: docs_gen/model_catalog/matching.py
    """
    rows = [score("nova-2-lite", organization="google")]
    model = CatalogModel(
        id="amazon.nova-2-lite-v1:0", name="Nova 2 Lite", provider="Amazon"
    )
    assert Matcher().match_board(model, "lmarena", "text", rows)[1] is None


def test_a_platform_publisher_does_not_block_a_match() -> None:
    """MTEB attributes its Bedrock entries to ``bedrock``, not to the vendor.

    Ref: https://github.com/embeddings-benchmark/results
    """
    rows = [score("cohere-embed-english-v3", organization="bedrock")]
    model = CatalogModel(
        id="cohere.embed-english-v3", name="Embed English", provider="Cohere"
    )
    assert Matcher().match_board(model, "lmarena", "text", rows)[1] is not None


# --- overrides ------------------------------------------------------------


def test_an_override_pins_a_match_and_a_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A human decision outranks both the rules and the model.

    Ref: docs_gen/model_catalog/matching.py
    """
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps(
            {
                "a.model": {
                    "lmarena/text": {"match": "other-entry", "reason": "checked"}
                },
                "b.model": {
                    "lmarena/text": {"match": None, "reason": "not the same model"}
                },
            }
        )
    )
    monkeypatch.setattr("docs_gen.model_catalog.matching.OVERRIDES_PATH", overrides)
    monkeypatch.setattr(
        "docs_gen.model_catalog.matching.MATCH_CACHE_PATH", tmp_path / "cache.json"
    )
    matcher = Matcher()

    rows = [score("other-entry"), score("model")]
    pinned, chosen = matcher.match_board(
        CatalogModel(id="a.model", name="Model", provider="Amazon"),
        "lmarena",
        "text",
        rows,
    )
    assert pinned.method == "override"
    assert chosen is not None
    assert chosen.name == "other-entry"

    rejected, nothing = matcher.match_board(
        CatalogModel(id="b.model", name="Model", provider="Amazon"),
        "lmarena",
        "text",
        rows,
    )
    assert rejected.method == "override"
    assert nothing is None


# --- the model that resolves the leftovers --------------------------------


def test_a_disagreement_between_the_two_directions_blocks_the_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both directions must land on the same model, or nothing is published.

    Ref: docs_gen/model_catalog/matching.py
    """
    monkeypatch.setattr(
        "docs_gen.model_catalog.matching.MATCH_CACHE_PATH", tmp_path / "cache.json"
    )
    matcher = Matcher(gateway_url="http://127.0.0.1:8000")
    answers = iter(
        [
            {"candidate": "some-entry", "confidence": 0.99, "reason": "same family"},
            {"candidate": None, "confidence": 0.1, "reason": "unsure"},
        ]
    )
    monkeypatch.setattr(matcher, "_ask", lambda *_: next(answers))

    model = CatalogModel(id="acme.some-thing", name="Some Thing", provider="Amazon")
    decision, chosen = matcher.match_board(
        model, "lmarena", "text", [score("some-entry")]
    )
    assert chosen is None
    assert "disagreed" in decision.reason


def test_low_confidence_blocks_the_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unsure answer is treated as no answer.

    Ref: docs_gen/model_catalog/matching.py
    """
    monkeypatch.setattr(
        "docs_gen.model_catalog.matching.MATCH_CACHE_PATH", tmp_path / "cache.json"
    )
    matcher = Matcher(gateway_url="http://127.0.0.1:8000")
    monkeypatch.setattr(
        matcher,
        "_ask",
        lambda *_: {"candidate": "some-entry", "confidence": 0.4, "reason": "maybe"},
    )
    model = CatalogModel(id="acme.some-thing", name="Some Thing", provider="Amazon")
    assert (
        matcher.match_board(model, "lmarena", "text", [score("some-entry")])[1] is None
    )


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"candidate": "x", "confidence": 1, "reason": "y"}', "x"),
        (
            'Sure!\n```json\n{"candidate": "x", "confidence": 1, "reason": "y"}\n```',
            "x",
        ),
    ],
)
def test_an_answer_is_parsed_even_when_the_model_wraps_it(
    content: str, expected: str
) -> None:
    """Not every model can be held to a JSON schema, so the reply is parsed leniently.

    Ref: docs_gen/model_catalog/matching.py
    """
    parsed = _parse_answer(content)
    assert parsed is not None
    assert parsed["candidate"] == expected


def test_prose_without_json_is_not_an_answer() -> None:
    """A reply with no JSON object yields nothing rather than a guess.

    Ref: docs_gen/model_catalog/matching.py
    """
    assert _parse_answer("I am not sure about this one.") is None


# --- dataset shaping ------------------------------------------------------


@pytest.mark.parametrize(
    ("region", "bucket"),
    [
        ("us-east-1", "americas"),
        ("sa-east-1", "americas"),
        ("eu-west-3", "europe"),
        ("ap-northeast-1", "asia_pacific"),
        ("il-central-1", "middle_east"),
        ("af-south-1", "africa"),
        ("xx-nowhere-1", "other"),
    ],
)
def test_region_buckets_are_one_editorial_mapping(region: str, bucket: str) -> None:
    """The four filter buttons resolve through a single table.

    Ref: docs_gen/model_catalog/config.py
    """
    assert region_bucket(region) == bucket


def test_serving_geographies_separate_where_a_model_runs_from_where_it_is_callable() -> (
    None
):
    """A global profile is not the same answer as a regional deployment.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
    """
    geographies = build.serving_geographies(
        {"eu-west-1": "eu.anthropic.claude", "us-east-1": "global.anthropic.claude"},
        ["eu-west-1", "us-east-1", "ap-south-1"],
    )
    assert geographies == ["ap-south-1", "eu", "global"]


def test_headline_prices_keep_only_the_plain_standard_rate() -> None:
    """Cache, long-context and media rows never become the headline price.

    Ref: docs/api_model_pricing.md
    """
    card = {
        "prices": [
            {
                "region": "us-east-1",
                "dimension": "input_tokens",
                "tier": "standard",
                "unit_price": "0.003",
            },
            {
                "region": "us-east-1",
                "dimension": "input_tokens",
                "tier": "batch",
                "unit_price": "0.001",
            },
            {
                "region": "us-east-1",
                "dimension": "input_tokens",
                "tier": "standard",
                "context": "long",
                "unit_price": "0.006",
            },
            {
                "region": "us-east-1",
                "dimension": "input_tokens",
                "tier": "standard",
                "routing": "global",
                "unit_price": "0.002",
            },
        ]
    }
    standard, cheapest, tiers = build.headline_prices(card)
    # Each routing is priced apart: they are different products, and the reader
    # picks one rather than being quoted whichever is cheaper.
    assert standard == {
        ("us-east-1", ""): {"input_tokens": "0.003"},
        ("us-east-1", "global"): {"input_tokens": "0.002"},
    }
    # Batch needs a different API, so it is never the cheaper tier on offer.
    assert cheapest == {}
    assert tiers == {}


def test_the_cheapest_tier_is_offered_alongside_the_standard_one() -> None:
    """A workload that can wait pays the batch rate, and the page can show it.

    Ref: docs/api_model_pricing.md
    """
    card = {
        "prices": [
            {
                "region": "eu-west-1",
                "dimension": "output_tokens",
                "tier": "standard",
                "unit_price": "0.015",
            },
            {
                "region": "eu-west-1",
                "dimension": "output_tokens",
                "tier": "flex",
                "unit_price": "0.0075",
            },
        ]
    }
    groups = build.group_prices(
        *build.headline_prices(card), {"eu-west-1": 0}, {"eu-west-1"}
    )
    assert groups[0].prices == {"output_tokens": "0.015"}
    assert groups[0].cheapest == {"output_tokens": "0.0075"}
    assert groups[0].cheapest_tier == "flex"


def test_a_model_sold_on_one_tier_advertises_no_cheaper_one() -> None:
    """An empty cheapest map is what tells the page there is no cheaper tier.

    Ref: docs_gen/model_catalog/build.py
    """
    card = {
        "prices": [
            {
                "region": "eu-west-1",
                "dimension": "input_tokens",
                "tier": "standard",
                "unit_price": "0.5",
            }
        ]
    }
    groups = build.group_prices(
        *build.headline_prices(card), {"eu-west-1": 0}, {"eu-west-1"}
    )
    assert groups[0].cheapest == {}
    assert not groups[0].cheapest_tier


def test_a_cheaper_tier_carries_every_dimension_it_sells() -> None:
    """A tier's map lists every dimension it sells, not only the cheaper ones.

    A dimension missing from the map means that tier does not sell it, so the
    page shows a dash instead of quoting the on-demand rate beside a batch one.

    Ref: docs_gen/model_catalog/build.py
    """

    def row(dimension: str, tier: str, price: str) -> dict[str, str]:
        return {
            "region": "eu-west-1",
            "dimension": dimension,
            "tier": tier,
            "unit_price": price,
        }

    card = {
        "prices": [
            row("input_tokens", "standard", "1"),
            row("output_tokens", "standard", "4"),
            row("cache_read_tokens", "standard", "0.5"),
            # Flex halves the token rates but does not sell cache reads.
            row("input_tokens", "flex", "0.5"),
            row("output_tokens", "flex", "2"),
        ]
    }
    group = build.group_prices(
        *build.headline_prices(card), {"eu-west-1": 0}, {"eu-west-1"}
    )[0]
    assert group.cheapest_tier == "flex"
    assert group.cheapest == {"input_tokens": "0.5", "output_tokens": "2"}
    assert "cache_read_tokens" in group.prices


def test_a_tier_dearer_on_any_shared_dimension_is_not_the_cheaper_one() -> None:
    """A dearer tier never wins by pricing fewer dimensions.

    Priority costs more than standard, so scoring each tier only on the
    dimensions it happens to publish would let it win on a shorter list.

    Ref: docs_gen/model_catalog/build.py
    """

    def row(dimension: str, tier: str, price: str) -> dict[str, str]:
        return {
            "region": "eu-west-1",
            "dimension": dimension,
            "tier": tier,
            "unit_price": price,
        }

    card = {
        "prices": [
            row("input_tokens", "standard", "1"),
            row("output_tokens", "standard", "4"),
            row("output_tokens", "priority", "6"),
        ]
    }
    group = build.group_prices(
        *build.headline_prices(card), {"eu-west-1": 0}, {"eu-west-1"}
    )[0]
    assert group.cheapest == {}
    assert not group.cheapest_tier


def test_regions_with_identical_prices_collapse_into_one_group() -> None:
    """The price matrix is the artefact's long tail, so equal regions share a row.

    Ref: docs_gen/model_catalog/build.py
    """
    groups = build.group_prices(
        {
            ("us-east-1", "region"): {"input_tokens": "1"},
            ("us-west-2", "region"): {"input_tokens": "1"},
            ("eu-west-1", "region"): {"input_tokens": "2"},
        },
        {},
        {},
        {"us-east-1": 0, "us-west-2": 1, "eu-west-1": 2},
        {"us-east-1", "us-west-2", "eu-west-1"},
    )
    assert sorted(
        (tuple(group.regions), group.prices["input_tokens"]) for group in groups
    ) == [((0, 1), "1"), ((2,), "2")]


def test_a_model_id_becomes_a_safe_filename() -> None:
    """A detail document is named after its model without escaping its directory.

    Ref: docs_gen/model_catalog/build.py
    """
    assert build.slug_for("anthropic.claude-sonnet-4-5-20250929-v1:0") == (
        "anthropic.claude-sonnet-4-5-20250929-v1_0"
    )
    assert "/" not in build.slug_for("a/b:c")


def test_an_unreadable_source_degrades_its_columns_instead_of_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One broken leaderboard must not cost the whole catalogue.

    Ref: docs_gen/model_catalog/build.py
    """

    def broken(*, refresh: bool) -> SourceResult:  # noqa: ARG001
        message = "upstream is down"
        raise RuntimeError(message)

    monkeypatch.setitem(build.COLLECTORS, "lmarena", broken)
    collected = build.collect_sources(refresh=False, only=["lmarena"])
    assert collected["lmarena"].scores == []
    assert "upstream is down" in collected["lmarena"].notes[0]


# --- the committed artefact -----------------------------------------------


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    """Load the committed catalogue.

    Returns:
        The published data set, validated against its schema.
    """
    path = DATA_DIR / "catalog.json"
    if not path.is_file():
        pytest.skip("the Models page data set has not been generated")
    return Catalog.model_validate_json(path.read_text())


def test_the_committed_index_matches_its_schema(catalog: Catalog) -> None:
    """A hand-edited or half-written artefact fails here rather than in the browser.

    Ref: docs_gen/model_catalog/schema.py
    """
    assert catalog.models
    assert catalog.manifest.regions
    assert catalog.manifest.reference_region in catalog.manifest.regions


def test_the_committed_index_stays_within_its_first_paint_budget() -> None:
    """The page must not silently grow into a multi-megabyte download.

    Ref: docs_gen/model_catalog/config.py
    """
    path = DATA_DIR / "catalog.json"
    if not path.is_file():
        pytest.skip("the Models page data set has not been generated")
    assert len(gzip.compress(path.read_bytes())) <= INDEX_GZIP_BUDGET


def test_deprecated_models_are_published_too(catalog: Catalog) -> None:
    """A retired model stays findable, tagged rather than removed.

    Ref: docs/api_search_models.md
    """
    assert any(model.legacy for model in catalog.models)


def test_every_published_score_names_its_source_and_its_entry(catalog: Catalog) -> None:
    """A score with no traceable origin cannot be checked by a reader.

    Ref: docs/models.md
    """
    known = {info.key for info in SOURCES}
    for model in catalog.models:
        for result in model.scores:
            assert result.source in known
            assert result.matched_name
            assert result.match_method in {"rule", "llm", "override"}


def test_every_published_source_is_attributed_on_the_page(catalog: Catalog) -> None:
    """Each licence's attribution requirement is discharged in the page source.

    Ref: https://creativecommons.org/licenses/by/4.0/
    """
    markdown = page.PAGE_PATH.read_text(encoding="utf-8")
    for source in catalog.manifest.sources:
        assert source.name in markdown
        assert source.licence_url in markdown


@pytest.mark.parametrize(
    "asset", ["js/models-table.min.js", "styles/models-table.min.css"]
)
def test_the_page_loads_the_asset_the_build_writes(asset: str) -> None:
    """The minifier renames what it minifies, and the page must name the artefact.

    The page injects its own script and stylesheet, so nothing rewrites those two
    references for it: naming the source file here is a 404 on the built site.

    Ref: mkdocs.yml
    """
    source = asset.replace(".min.", ".")
    assert asset.rsplit("/", 1)[1] in PAGE_PATH.read_text(encoding="utf-8")
    assert (REPO_ROOT / "docs" / source).is_file()
    assert source in (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")


def test_the_snapshot_stamp_names_the_version_the_instance_reported(
    catalog: Catalog,
) -> None:
    """The stamp is the page's own provenance, so it says which build it read.

    Ref: docs_gen/model_catalog/gateway.py
    """
    stamped = catalog.model_copy(
        update={
            "manifest": catalog.manifest.model_copy(
                update={"gateway_version": "1.16.0"}
            )
        }
    )
    assert "from stdapi.ai 1.16.0," in page._generated_block(stamped)  # noqa: SLF001


@pytest.mark.parametrize("version", [UNKNOWN_VERSION, ""])
def test_a_missing_version_never_reaches_the_page(
    catalog: Catalog, version: str
) -> None:
    """An instance serving no OpenAPI document must not read as a broken build.

    Ref: docs_gen/model_catalog/gateway.py
    """
    stamped = catalog.model_copy(
        update={
            "manifest": catalog.manifest.model_copy(update={"gateway_version": version})
        }
    )
    stamp = page._generated_block(stamped)  # noqa: SLF001
    assert "from a stdapi.ai instance," in stamp
    assert UNKNOWN_VERSION not in stamp


def test_every_provider_shown_is_covered_by_the_trademark_registry(
    catalog: Catalog,
) -> None:
    """A mark the page displays without attribution fails the build, so catch it here.

    Ref: docs_hooks/trademarks.py
    """
    markdown = page.PAGE_PATH.read_text(encoding="utf-8")
    assert page.check_trademarks(catalog, markdown) == []


def test_no_leaderboard_entry_is_claimed_by_unrelated_models(catalog: Catalog) -> None:
    """Two models may share an entry only when they are two IDs for one model.

    Ref: docs_gen/model_catalog/state/unmatched.json
    """
    claims: dict[tuple[str, str, str], set[str]] = {}
    for model in catalog.models:
        for result in model.scores:
            claims.setdefault(
                (result.source, result.board, result.matched_name), set()
            ).add(model.provider)
    across_vendors = {
        key: providers for key, providers in claims.items() if len(providers) > 1
    }
    assert not across_vendors


# --- updating rather than replacing ---------------------------------------


def a_row(model_id: str, **fields: Any) -> ModelRow:  # noqa: ANN401
    """Build a catalogue row for a test.

    Args:
        model_id: Amazon Bedrock model ID.
        **fields: Any other :class:`ModelRow` field.

    Returns:
        The row.
    """
    defaults: dict[str, Any] = {
        "id": model_id,
        "slug": build.slug_for(model_id),
        "name": model_id,
        "provider": "Amazon",
        "service": "AWS Bedrock Runtime",
        "input_modalities": ["TEXT"],
        "output_modalities": ["TEXT"],
    }
    return ModelRow(**{**defaults, **fields})


def test_a_value_aws_stops_returning_is_kept() -> None:
    """An undocumented field that disappears must not blank the column.

    Ref: docs_gen/model_catalog/merge.py
    """
    before = [a_row("a.model", context_window="200K", family="Nova")]
    merged, report = merge.merge_models(
        before, [a_row("a.model")], generated="2026-08-22"
    )
    assert merged[0].context_window == "200K"
    assert merged[0].family == "Nova"
    assert report.carried["context_window"] == 1


def test_a_value_aws_now_returns_wins() -> None:
    """A fresh value always beats the one carried forward.

    Ref: docs_gen/model_catalog/merge.py
    """
    before = [a_row("a.model", context_window="128K")]
    merged, report = merge.merge_models(
        before, [a_row("a.model", context_window="200K")], generated="2026-08-22"
    )
    assert merged[0].context_window == "200K"
    assert not report.carried


def test_a_model_aws_stops_listing_is_kept_and_marked() -> None:
    """A retired model stays findable instead of vanishing from the catalogue.

    Ref: docs_gen/model_catalog/merge.py
    """
    before = [a_row("gone.model", last_seen="2026-08-01"), a_row("here.model")]
    merged, report = merge.merge_models(
        before, [a_row("here.model")], generated="2026-08-22"
    )
    gone = next(row for row in merged if row.id == "gone.model")
    assert gone.retired
    assert gone.last_seen == "2026-08-01"
    assert report.retired == ["gone.model"]
    assert next(row for row in merged if row.id == "here.model").retired is False


def test_a_model_that_comes_back_is_unmarked() -> None:
    """A model AWS lists again is current again.

    Ref: docs_gen/model_catalog/merge.py
    """
    before = [a_row("a.model", retired=True, first_seen="2026-01-01")]
    merged, report = merge.merge_models(
        before, [a_row("a.model")], generated="2026-08-22"
    )
    assert merged[0].retired is False
    assert merged[0].first_seen == "2026-01-01"
    assert report.returned == ["a.model"]


@pytest.mark.parametrize(
    ("tokens", "shown"),
    [
        (1_000_000, "1M"),
        (204800, "200K"),
        (131072, "128K"),
        (500000, "500K"),
        (163840, "160K"),
        ("163,840", "160K"),
        ("200K", "200K"),
        (0, None),
    ],
)
def test_a_context_window_is_shown_the_way_vendors_quote_it(
    tokens: int, shown: str | None
) -> None:
    """131072 is quoted as 128K, and printing the raw number helps nobody.

    Ref: https://models.dev/
    """
    assert format_tokens(tokens) == shown


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("us.anthropic.claude-opus-5", "anthropic.claude-opus-5"),
        ("meta.llama3-8b", "meta.llama3-8b"),
    ],
)
def test_a_geography_prefix_is_not_a_different_model(
    model_id: str, expected: str
) -> None:
    """models.dev lists one entry per inference profile; they are one model.

    Ref: https://models.dev/
    """
    assert models_dev._base_id(model_id) == expected  # noqa: SLF001


@pytest.mark.parametrize(
    ("stem", "expected"), [("anthropic", "light"), ("openai", "dark"), ("google", "")]
)
def test_a_logo_gets_only_the_backdrop_it_needs(stem: str, expected: str) -> None:
    """A mark is never recoloured, so the page adapts around it instead.

    Ref: docs_gen/model_catalog/build.py
    """
    assert build.logo_backdrop(stem) == expected


def test_the_committed_catalogue_carries_context_windows(catalog: Catalog) -> None:
    """Context window has no AWS source, so its external one must be working.

    Ref: https://models.dev/
    """
    assert len([model for model in catalog.models if model.context_window]) > 30


# --- hand-filled facts must stay auditable --------------------------------


def test_every_hand_filled_fact_cites_a_page() -> None:
    """A number with no source is a number nobody can check.

    Ref: docs_gen/model_catalog/enrichment.py
    """
    if not PROVENANCE_PATH.is_file():
        pytest.skip("the catalogue has not been generated")
    used = json.loads(PROVENANCE_PATH.read_text())["models"]
    uncited = [
        f"{model}.{field}"
        for model, fields in used.items()
        for field, entry in fields.items()
        if not entry.get("source", "").startswith("https://")
        or not entry.get("checked")
    ]
    assert not uncited


def test_the_overlay_only_sets_fields_it_is_allowed_to() -> None:
    """An entry naming a field the catalogue does not have is a typo, not a fact.

    Ref: docs_gen/model_catalog/enrichment.py
    """
    if not ENRICHMENT_PATH.is_file():
        pytest.skip("no overlay")
    overlay = enrichment.load()
    stray = [
        f"{model}.{field}"
        for model, fields in overlay.items()
        for field in fields
        if field not in enrichment.ALLOWED_FIELDS
    ]
    assert not stray


def test_the_overlay_never_overwrites_a_collected_value() -> None:
    """AWS and the open databases win; the overlay only fills what they left.

    Ref: docs_gen/model_catalog/enrichment.py
    """
    row = a_row("a.model", context_window="200K")
    applied = enrichment.apply(
        [row],
        {
            "a.model": {
                "context_window": {
                    "value": "1M",
                    "source": "https://example.invalid/",
                    "checked": "2026-08-22",
                }
            }
        },
    )
    assert row.context_window == "200K"
    assert applied.skipped == 1
    assert not applied.filled


# --- the AWS model cards ---------------------------------------------------


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("Mar 13, 2024", "2024-03-13"),
        ("September 10, 2026", "2026-09-10"),
        ("Aug 2023", "2023-08"),
        ("Legacy: July 7, 2026", "2026-07-07"),
        # A floor AWS may move is not a retirement date.
        ("No sooner than 10/1/2026", None),
        ("N/A", None),
        ("", None),
    ],
)
def test_a_model_card_date_is_read_in_every_form_aws_writes_it(
    written: str, expected: str | None
) -> None:
    """The cards mix three date formats and a placeholder.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html
    """
    assert aws_model_cards._date(written) == expected  # noqa: SLF001


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("us.anthropic.claude-opus-5", "anthropic.claude-opus-5"),
        ("global.amazon.nova-2-lite-v1:0", "amazon.nova-2-lite-v1:0"),
        ("meta.llama3-8b-instruct-v1:0", "meta.llama3-8b-instruct-v1:0"),
    ],
)
def test_a_card_id_is_reduced_to_the_model_it_names(
    written: str, expected: str
) -> None:
    """Cards print the cross-region variants beside the plain ID.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html
    """
    assert aws_model_cards._plain_id(written) == expected  # noqa: SLF001


def test_a_card_is_only_joined_to_a_model_the_catalogue_has(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The join is by printed ID, so a card can never be matched by guesswork.

    Ref: docs_gen/model_catalog/sources/aws_model_cards.py
    """
    cards = [
        {
            "page": "model-card-known.html",
            "ids": ["us.acme.known-v1:0", "boto3.client"],
            "facts": {"context_window": "200K tokens", "max_output_tokens": "4K"},
        },
        {
            "page": "model-card-stranger.html",
            "ids": ["other.stranger-v1:0"],
            "facts": {"context_window": "1M tokens"},
        },
    ]
    monkeypatch.setattr(
        "docs_gen.model_catalog.sources.aws_model_cards.snapshot",
        lambda *_args, **_kwargs: cards,
    )
    facts, notes = aws_model_cards.fetch(["acme.known-v1:0"])
    assert facts == {
        "acme.known-v1:0": {"context_window": "200K", "max_output_tokens": 4000}
    }
    assert notes == ["1 model card(s) describe no model this gateway serves"]


# --- refusing to publish a collection failure ------------------------------


def test_losing_most_of_the_catalogue_in_one_run_is_refused() -> None:
    """An empty or half-answered collection is a failure, not a release.

    Ref: docs_gen/model_catalog/merge.py
    """
    previous = Catalog(
        manifest=Manifest(
            generated="2026-08-01",
            gateway_version="1.16.0",
            partitions=["aws"],
            currencies=["USD"],
            reference_region="us-east-1",
            regions=["us-east-1"],
            region_buckets={"us-east-1": "americas"},
        ),
        models=[a_row(f"a.model-{index}") for index in range(10)],
    )
    _, report = merge.merge_models(
        previous.models, [a_row("a.model-0")], generated="2026-08-22"
    )
    with pytest.raises(merge.UnsafeUpdateError):
        merge.check_sane(previous, report, total=1)


def test_a_normal_retirement_is_allowed() -> None:
    """One model going away is a Tuesday, not a catastrophe.

    Ref: docs_gen/model_catalog/merge.py
    """
    previous = Catalog(
        manifest=Manifest(
            generated="2026-08-01",
            gateway_version="1.16.0",
            partitions=["aws"],
            currencies=["USD"],
            reference_region="us-east-1",
            regions=["us-east-1"],
            region_buckets={"us-east-1": "americas"},
        ),
        models=[a_row(f"a.model-{index}") for index in range(10)],
    )
    current = [a_row(f"a.model-{index}") for index in range(1, 10)]
    _, report = merge.merge_models(previous.models, current, generated="2026-08-22")
    merge.check_sane(previous, report, total=9)


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        # Every AWS model card names the unit.
        ("128K tokens", 128000),
        ("1M tokens", 1_000_000),
        ("512 tokens", 512),
        ("163,840", 163840),
        ("N/A", None),
    ],
)
def test_a_token_count_is_read_with_the_unit_spelled_out(
    written: str, expected: int | None
) -> None:
    """A count nobody can parse is a column of dashes.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html
    """
    assert parse_tokens(written) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (128000, "128K"),  # decimal: the vendor wrote 128K
        (131072, "128K"),  # binary: the vendor also wrote 128K
        (204800, "200K"),  # binary only
        (200000, "200K"),
        (1_000_000, "1M"),
        (1_048_576, "1M"),
        (512, "512"),
        (163840, "160K"),
    ],
)
def test_a_token_count_is_written_the_way_its_vendor_writes_it(
    count: int, expected: str
) -> None:
    """128000 is 128K, not 125K — the wrong base is a wrong number.

    Ref: docs_gen/model_catalog/tokens.py
    """
    assert format_tokens(count) == expected


@pytest.mark.parametrize(
    "count", [128000, 131072, 204800, 1_000_000, 1_048_576, 163840, 262140, 1_050_000]
)
def test_reading_a_rendered_count_gives_the_same_rendering_back(count: int) -> None:
    """A value must survive a merge that re-reads what the last run wrote.

    Ref: docs_gen/model_catalog/merge.py
    """
    once = format_tokens(count)
    assert format_tokens(once) == once


# --- the page and the generator must agree ---------------------------------


def _price_columns_block() -> str:
    """Return the page's ``PRICE_COLUMNS`` literal as source text.

    Returns:
        Everything between the opening bracket and its close.
    """
    script = (REPO_ROOT / "docs" / "js" / "models-table.js").read_text()
    return script.split("var PRICE_COLUMNS = [", 1)[1].split("\n  ];", 1)[0]


def test_the_page_prices_exactly_the_dimensions_the_generator_emits() -> None:
    """A dimension in one list and not the other is a price nobody sees.

    The guardrail model billed on ``text_units`` showed a dash for months
    because the generator did not headline the dimension the page could render.

    Ref: docs_gen/model_catalog/config.py
    """
    block = _price_columns_block()
    on_page = re.findall(r'key: "([a-z_]+)"', block)
    assert on_page == list(HEADLINE_DIMENSIONS)


def test_every_price_column_carries_a_scale_and_a_unit() -> None:
    """A column with no unit renders a bare number nobody can act on.

    Ref: docs/js/models-table.js
    """
    block = _price_columns_block()
    for field in ("label:", "unit:", "scale:", "dimensionHelp:"):
        assert block.count(field) == len(HEADLINE_DIMENSIONS), field
    # Each dimension explains its own billed unit; one shared sentence for all
    # thirteen said nothing about any of them.
    helps = re.findall(r'dimensionHelp: "([^"]+)"', block)
    assert len(set(helps)) == len(HEADLINE_DIMENSIONS)


# --- which source states which fact ----------------------------------------


@pytest.mark.parametrize(
    ("page", "model_id", "confirmed"),
    [
        # The page name repeats the vendor; the model half still decides.
        ("model-card-deepseek-deepseek-v3-1.html", "deepseek.v3.1", True),
        # A sibling the page only cross-references must not take its facts.
        ("model-card-deepseek-deepseek-v3-1.html", "deepseek.v3-v1:0", False),
        # AWS spells the vendor two ways; both IDs are the same model.
        (
            "model-card-moonshot-ai-kimi-k2-thinking.html",
            "moonshot.kimi-k2-thinking",
            True,
        ),
        (
            "model-card-moonshot-ai-kimi-k2-thinking.html",
            "moonshotai.kimi-k2-thinking",
            True,
        ),
        (
            "model-card-anthropic-claude-3-sonnet.html",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            True,
        ),
    ],
)
def test_a_card_is_claimed_only_by_the_model_it_is_named_for(
    page: str, model_id: str, confirmed: bool
) -> None:
    """A card that cross-references a sibling must not overwrite that sibling.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html
    """
    assert aws_model_cards._confirms(page, model_id) is confirmed  # noqa: SLF001


def test_a_models_own_version_is_not_read_as_an_api_version() -> None:
    """``deepseek.v3.1`` reduced to ``deepseek`` matched every DeepSeek page.

    Ref: docs_gen/model_catalog/sources/aws_model_cards.py
    """
    part = aws_model_cards._model_part  # noqa: SLF001 -- the unit under test
    assert part("deepseek.v3.1") == "v3-1"
    assert part("zai.glm-4.7") == "glm-4-7"
    # The API version tag is anchored on its colon, so only it is stripped.
    assert part("anthropic.claude-3-sonnet-20240229-v1:0") == "claude-3-sonnet"


def test_the_output_ceiling_is_never_the_context_window(catalog: Catalog) -> None:
    """``converse.maxTokensMaximum`` is a request ceiling, not an output limit.

    Reading it as the output limit published Gemma 3's whole 128K context
    window as its largest response.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html
    """
    same = [
        model.id
        for model in catalog.models
        if model.max_output_tokens
        and model.context_window
        and parse_tokens(model.context_window) == model.max_output_tokens
    ]
    assert not same


# --- a curated value that is the same fact, stated better -------------------


@pytest.mark.parametrize(
    ("name", "collected", "curated", "same"),
    [
        # The card rounds 4096 to "4K", which reads back as 4000.
        ("max_output_tokens", 4000, 4096, True),
        ("max_output_tokens", 8000, 8192, True),
        # A genuinely different ceiling is a disagreement, not a rounding.
        ("max_output_tokens", 4000, 64000, False),
        ("context_window", "200K", "200K", False),
        # The card states the month; the vendor page states the day.
        ("knowledge_cutoff", "2024-03", "2024-03-05", True),
        ("knowledge_cutoff", "2023-12", "2023-03", False),
        # Nothing else is ever a rounding of anything.
        ("family", "Nova", "Amazon Nova 2", False),
    ],
)
def test_a_rounded_value_and_an_exact_one_are_recognised_as_one_fact(
    name: str, collected: object, curated: object, same: bool
) -> None:
    """Publishing a card's "4K" as an exact 4,000 claims precision it never had.

    Ref: docs_gen/model_catalog/enrichment.py
    """
    assert enrichment._same_figure(name, collected, curated) is same  # noqa: SLF001


def test_a_curated_value_that_really_disagrees_is_reported_not_published() -> None:
    """The automatic source wins, but a cited disagreement must not be silent.

    Ref: docs_gen/model_catalog/enrichment.py
    """
    row = a_row("a.model", context_window="128K")
    applied = enrichment.apply(
        [row],
        {
            "a.model": {
                "context_window": {
                    "value": "256K",
                    "source": "https://example.invalid/",
                    "checked": "2026-08-22",
                }
            }
        },
    )
    assert row.context_window == "128K"
    assert applied.disputed == ["a.model.context_window 128K vs 256K"]


def test_a_citation_survives_the_run_that_retires_its_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retired model keeps its value, so it must keep its source too.

    Ref: docs_gen/model_catalog/enrichment.py
    """
    # Redirected: the real path holds the committed citations of a real run.
    written_to = tmp_path / "provenance.json"
    monkeypatch.setattr(enrichment, "PROVENANCE_PATH", written_to)
    retired = a_row("a.gone", parameters="7B")
    retired.retired = True
    overlay = {
        "a.gone": {
            "parameters": {
                "value": "7B",
                "source": "https://example.invalid/card",
                "source_name": "Vendor model card",
                "checked": "2026-08-22",
            }
        }
    }
    assert enrichment.record_provenance([retired], overlay) == 1
    written = json.loads(written_to.read_text())["models"]
    assert written["a.gone"]["parameters"]["source"] == "https://example.invalid/card"


# --- reading a previous data set the current schema does not recognise ------


def _previous(tmp_path: Path, document: dict[str, Any]) -> Path:
    """Write a previous catalogue for a merge to read.

    Args:
        tmp_path: Directory to write into.
        document: The raw JSON document.

    Returns:
        Path of the written file.
    """
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document))
    return path


_A_ROW: dict[str, Any] = {
    "id": "a.model",
    "slug": "a_model",
    "name": "A",
    "provider": "P",
    "service": "S",
    "first_seen": "2026-01-01",
}


def test_a_previous_file_written_by_another_schema_still_merges(tmp_path: Path) -> None:
    """A field added or dropped since must not cost every model its history.

    Ref: docs_gen/model_catalog/merge.py
    """
    path = _previous(
        tmp_path,
        {"manifest": {"generated": "2026-01-01", "gone_since": 1}, "models": [_A_ROW]},
    )
    previous = merge.load_previous(path)
    assert previous is not None
    assert [row.id for row in previous.models] == ["a.model"]
    # The merge reads this; model_construct used to leave it unset and raise.
    assert list(previous.manifest.regions) == []


@pytest.mark.parametrize(
    "document",
    [{"manifest": {}, "models": []}, {"models": "not a list"}, ["not a catalogue"]],
)
def test_a_previous_file_that_is_not_a_catalogue_is_refused(
    tmp_path: Path, document: object
) -> None:
    """Reading it as "no previous data" would reset every first_seen date.

    Ref: docs_gen/model_catalog/merge.py
    """
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(document))
    with pytest.raises(merge.UnsafeUpdateError):
        merge.load_previous(path)


def test_a_previous_file_that_is_not_json_is_refused(tmp_path: Path) -> None:
    """A truncated write must fail the run, not silently start over.

    Ref: docs_gen/model_catalog/merge.py
    """
    path = tmp_path / "catalog.json"
    path.write_text('{"models": [')
    with pytest.raises(merge.UnsafeUpdateError):
        merge.load_previous(path)


def test_an_operator_can_accept_a_real_deprecation_wave() -> None:
    """Otherwise the only way past the ceiling is --fresh, which drops history.

    Ref: docs_gen/model_catalog/merge.py
    """
    previous = Catalog(
        manifest=Manifest(
            generated="2026-08-01",
            gateway_version="1.16.0",
            partitions=["aws"],
            currencies=["USD"],
            reference_region="us-east-1",
            regions=["us-east-1"],
            region_buckets={"us-east-1": "americas"},
        ),
        models=[a_row(f"a.model-{index}") for index in range(10)],
    )
    _, report = merge.merge_models(
        previous.models, [a_row("a.model-0")], generated="2026-08-22"
    )
    with pytest.raises(merge.UnsafeUpdateError):
        merge.check_sane(previous, report, total=1)
    merge.check_sane(previous, report, total=1, accept_retirements=True)


# --- the detail card and the table must agree -------------------------------


def test_no_model_is_priced_in_a_region_it_cannot_be_called_from(
    catalog: Catalog,
) -> None:
    """AWS prices Claude 3.7 Sonnet in 31 regions and serves it in two.

    Quoting the other 29 offers a price nobody can buy, and contradicts the
    table, which counts only the regions the model runs in.

    Ref: docs_gen/model_catalog/build.py
    """
    names = catalog.manifest.regions
    offenders = []
    for model in catalog.models:
        path = DATA_DIR / "detail" / f"{model.slug}.json"
        if not path.is_file():
            continue
        served = {names[index] for index in model.regions if index < len(names)}
        rows = (json.loads(path.read_text()).get("prices") or {}).get("prices") or []
        stray = {str(row.get("region")) for row in rows} - served
        if stray:
            offenders.append((model.id, sorted(stray)[:3]))
    assert not offenders


def test_the_page_assets_load_only_on_the_page_that_uses_them() -> None:
    """58 pages shipping 41 KiB of a table script none of them render is waste.

    Ref: https://www.mkdocs.org/user-guide/configuration/#extra_css
    """
    config = (REPO_ROOT / "mkdocs.yml").read_text()
    # The minifier is configured with both files by name, so it is the
    # site-wide asset lists that must not carry them, not the whole config.
    site_wide = config.split("extra_css:", 1)[1].split("markdown_extensions:", 1)[0]
    assert "models-table.css" not in site_wide
    assert "models-table.js" not in site_wide
    page_source = PAGE_PATH.read_text()
    assert (
        '<link rel="stylesheet" href="../styles/models-table.min.css">' in page_source
    )
    assert '<script src="../js/models-table.min.js" defer></script>' in page_source
    # The relative hrefs above assume a page served as a directory.
    assert "use_directory_urls: false" not in config


# --- what the generator is allowed to fetch --------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.invalid/data.json",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "http://example.invalid/data.json",
    ],
)
def test_the_generator_refuses_a_url_it_should_not_fetch(url: str) -> None:
    """A source URL is data; the scheme rules are the control on it.

    Ref: docs_gen/model_catalog/http.py
    """
    with pytest.raises(ValueError, match="refusing"):
        http._validated(url)  # noqa: SLF001


@pytest.mark.parametrize(
    "url", ["https://example.invalid/data.json", "http://127.0.0.1:8000/v1/models"]
)
def test_the_generator_fetches_https_and_a_local_gateway(url: str) -> None:
    """Plain HTTP is allowed only to the loopback instance being read.

    Ref: docs_gen/model_catalog/http.py
    """
    assert http._validated(url) == url  # noqa: SLF001


def test_a_redirect_is_checked_against_the_same_rules() -> None:
    """Urlopen follows redirects itself, and its default handler allows ftp.

    Ref: https://docs.python.org/3/library/urllib.request.html#urllib.request.HTTPRedirectHandler
    """
    handler = http._CheckedRedirects()  # noqa: SLF001
    with pytest.raises(ValueError, match="refusing"):
        handler.redirect_request(
            Request("https://example.invalid/"),
            None,
            302,
            "Found",
            {},
            "ftp://elsewhere.invalid/x",
        )


# --- the snapshot cache -----------------------------------------------------


def test_a_snapshot_of_one_question_is_not_served_for_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding a board would otherwise yield no rows until the cache expired.

    Ref: docs_gen/model_catalog/sources/__init__.py
    """
    monkeypatch.setattr("docs_gen.model_catalog.sources.SNAPSHOT_DIR", tmp_path)
    assert sources.snapshot("board", lambda: ["a"], key="one") == ["a"]
    assert sources.snapshot("board", lambda: ["b"], key="two") == ["b"]
    assert sources.snapshot("board", lambda: ["c"], key="one") == ["a"]


def test_a_snapshot_directory_is_readable_only_by_its_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A world-writable cache lets anyone pre-plant what the page publishes.

    Ref: docs_gen/model_catalog/sources/__init__.py
    """
    loose = tmp_path / "cache"
    loose.mkdir(mode=0o755)
    monkeypatch.setattr("docs_gen.model_catalog.sources.SNAPSHOT_DIR", loose)
    sources.snapshot("thing", lambda: {"v": 1})
    assert loose.stat().st_mode & 0o777 == 0o700


def test_yesterdays_snapshot_beats_no_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One unreachable source must degrade its column, not blank it.

    Ref: docs_gen/model_catalog/sources/__init__.py
    """
    monkeypatch.setattr("docs_gen.model_catalog.sources.SNAPSHOT_DIR", tmp_path)

    def unreachable() -> object:
        msg = "upstream is down"
        raise RuntimeError(msg)

    assert sources.snapshot("thing", lambda: {"v": 1}) == {"v": 1}
    os.utime(tmp_path / "thing.all.json", (0, 0))
    assert sources.snapshot("thing", unreachable) == {"v": 1}
    (tmp_path / "thing.all.json").unlink()
    with pytest.raises(RuntimeError, match="upstream is down"):
        sources.snapshot("thing", unreachable)


def test_a_previous_row_keeps_its_history_and_drops_only_what_no_longer_fits(
    tmp_path: Path,
) -> None:
    """A value the schema now rejects is dropped, not carried.

    Pydantic does not revalidate an instance, so an unchecked value would be
    republished as-is and only fail the *next* run's schema check.

    Ref: docs_gen/model_catalog/merge.py
    """
    path = _previous(
        tmp_path,
        {
            "manifest": {"generated": "2026-01-01"},
            # An int where the schema wants a rendered string like "128K".
            "models": [{**_A_ROW, "context_window": 12345, "gone_since": 1}],
        },
    )
    previous = merge.load_previous(path)
    assert previous is not None
    row = previous.models[0]
    assert row.context_window is None
    assert row.first_seen == "2026-01-01"
    assert row.name == "A"


# --- a region is priced once per way of reaching it -------------------------


def _price_row(region: str, routing: str, price: str) -> dict[str, Any]:
    """Build one standard-tier input-token price row.

    Args:
        region: AWS region.
        routing: The routing the price belongs to.
        price: Unit price as a decimal string.

    Returns:
        A price row shaped like the gateway's.
    """
    return {
        "region": region,
        "tier": "standard",
        "dimension": "input_tokens",
        "routing": routing,
        "unit_price": price,
    }


@pytest.mark.parametrize(
    ("region", "routing", "kind"),
    [
        ("eu-central-1", "eu-central-1", "region"),
        ("eu-central-1", "eu", "geography"),
        ("ap-northeast-1", "apac", "geography"),
        ("us-east-1", "global", "global"),
        ("us-east-1", "latency", "latency"),
        ("us-east-1", "", ""),
    ],
)
def test_a_routing_is_named_by_what_kind_of_product_it_is(
    region: str, routing: str, kind: str
) -> None:
    """The reader chooses a product, so the page has to know which one it is.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
    """
    assert build._routing_kind(region, routing) == kind  # noqa: SLF001


def test_every_way_of_reaching_a_region_is_priced_separately() -> None:
    """A region is priced once per routing AWS publishes for it.

    Staying in a geography can cost more than routing globally, and the reader
    picks between them, so both prices are published.

    Ref: https://aws.amazon.com/bedrock/pricing/
    """
    card = {
        "prices": [
            _price_row("eu-central-1", "eu-central-1", "0.000000429"),
            _price_row("eu-central-1", "global", "0.00000039"),
        ]
    }
    standard, _, _ = build.headline_prices(card)
    assert standard == {
        ("eu-central-1", "region"): {"input_tokens": "0.000000429"},
        ("eu-central-1", "global"): {"input_tokens": "0.00000039"},
    }


def test_the_latency_optimised_product_is_not_a_way_of_running_this_one() -> None:
    """It is a different thing to buy, priced apart from the model's own rate.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/latency-optimized-inference.html
    """
    card = {
        "prices": [
            _price_row("us-east-1", "us-east-1", "0.001"),
            _price_row("us-east-1", "latency", "0.004"),
        ]
    }
    standard, _, _ = build.headline_prices(card)
    assert set(standard) == {("us-east-1", "region")}


def test_a_region_offering_two_routings_yields_two_groups() -> None:
    """The page picks the group matching the routing the reader chose.

    Ref: docs_gen/model_catalog/build.py
    """
    card = {
        "prices": [
            _price_row("eu-central-1", "eu-central-1", "0.000000429"),
            _price_row("eu-central-1", "global", "0.00000039"),
        ]
    }
    groups = build.group_prices(
        *build.headline_prices(card), {"eu-central-1": 0}, {"eu-central-1"}
    )
    assert {group.routing for group in groups} == {"region", "global"}
    priced = {group.routing: group.prices["input_tokens"] for group in groups}
    assert priced["region"] > priced["global"]


# --- one model, two AWS services --------------------------------------------


def _served(model_id: str, service: str, **fields: Any) -> ModelRow:  # noqa: ANN401
    """Build a row as one AWS service reports it.

    Args:
        model_id: The ``model`` value that service accepts.
        service: The AWS service serving it.
        **fields: Any other :class:`ModelRow` field.

    Returns:
        The row.
    """
    return a_row(model_id, service=service, provider="OpenAI", **fields)


def test_one_model_on_two_services_becomes_one_row() -> None:
    """Listing it twice makes the reader compare a model against itself.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/mantle.html
    """
    rows, absorbed, notes = build.fold_service_variants(
        [
            _served("openai.gpt-oss-120b", "AWS Bedrock Mantle", regions=[0, 1, 2]),
            _served("openai.gpt-oss-120b-1:0", "AWS Bedrock Runtime", regions=[0, 1]),
        ]
    )
    assert [row.id for row in rows] == ["openai.gpt-oss-120b"]
    assert absorbed == {"openai.gpt-oss-120b-1:0"}
    assert notes
    folded = rows[0]
    # Reach decides which ID survives, so the choice is the same every run.
    assert [variant.service for variant in folded.variants] == [
        "AWS Bedrock Mantle",
        "AWS Bedrock Runtime",
    ]
    # The absorbed ID stays callable, so it has to stay findable.
    assert "openai.gpt-oss-120b-1:0" in folded.aliases
    assert folded.regions == [0, 1, 2]


def test_two_versions_of_a_model_are_not_one_model() -> None:
    """Nova Reel v1:0 and v1:1 are different models on the same service.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html
    """
    rows, absorbed, _ = build.fold_service_variants(
        [
            a_row("amazon.nova-reel-v1:0", service="AWS Bedrock Runtime"),
            a_row("amazon.nova-reel-v1:1", service="AWS Bedrock Runtime"),
        ]
    )
    assert len(rows) == 2
    assert not absorbed


def test_the_folded_row_takes_the_name_a_reader_can_read() -> None:
    """One surface reports ``qwen3-32b``, the other ``Qwen3 32B (dense)``.

    Ref: docs_gen/model_catalog/build.py
    """
    rows, _, _ = build.fold_service_variants(
        [
            _served(
                "qwen.qwen3-32b", "AWS Bedrock Mantle", name="qwen3-32b", regions=[0, 1]
            ),
            _served(
                "qwen.qwen3-32b-v1:0",
                "AWS Bedrock Runtime",
                name="Qwen3 32B (dense)",
                regions=[0],
            ),
        ]
    )
    assert rows[0].name == "Qwen3 32B (dense)"


def test_a_service_is_named_on_a_price_only_where_the_services_differ() -> None:
    """A service is named on a price only where the two services disagree.

    Mantle is about 14% cheaper than Runtime in ap-southeast-2 and the same
    everywhere else, so naming a service everywhere would imply a choice that
    usually has no consequence.

    Ref: https://aws.amazon.com/bedrock/pricing/
    """
    mantle = _served(
        "acme.model",
        "AWS Bedrock Mantle",
        regions=[0, 1],
        price_groups=[
            PriceGroup(regions=[0], prices={"input_tokens": "1"}, routing="region"),
            PriceGroup(regions=[1], prices={"input_tokens": "0.86"}, routing="region"),
        ],
    )
    runtime = _served(
        "acme.model-v1:0",
        "AWS Bedrock Runtime",
        regions=[0, 1],
        price_groups=[
            PriceGroup(regions=[0], prices={"input_tokens": "1"}, routing="region"),
            PriceGroup(regions=[1], prices={"input_tokens": "1"}, routing="region"),
        ],
    )
    rows, _, _ = build.fold_service_variants([mantle, runtime])
    by_region = {
        index: [group for group in rows[0].price_groups if index in group.regions]
        for index in (0, 1)
    }
    # Region 0: the two agree, so one unattributed price.
    assert [group.service for group in by_region[0]] == [""]
    # Region 1: they do not, so both, each naming who charges it.
    assert sorted(group.service for group in by_region[1]) == [
        "AWS Bedrock Mantle",
        "AWS Bedrock Runtime",
    ]


# --- the price the gateway would actually bill -------------------------------


@pytest.mark.parametrize(
    ("routings", "expected"),
    [
        (["global"], "global"),
        (["us-east-1"], ""),
        (["apac", "eu", "us"], ""),
        ([], ""),
        (None, ""),
    ],
)
def test_a_model_routed_globally_is_priced_globally(
    routings: list[str] | None, expected: str
) -> None:
    """A model the gateway routes globally is priced globally.

    Quoting the in-region rate for one of those shows a price the caller is
    never charged.

    Ref: stdapi/routes/core_models.py
    """
    card = {"default_routings": routings} if routings is not None else {}
    assert build._default_routing(card) == expected  # noqa: SLF001


def test_the_committed_catalogue_knows_where_the_gateway_routes(
    catalog: Catalog,
) -> None:
    """The page defaults to this, so it has to survive into the artefact.

    Ref: docs_gen/model_catalog/schema.py
    """
    routed = [model for model in catalog.models if model.default_routing == "global"]
    assert routed
    # Every routing named must be one the page can actually select.
    assert {model.default_routing for model in catalog.models} <= {"", "global"}
