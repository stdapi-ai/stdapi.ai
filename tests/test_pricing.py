"""Unit tests for stdapi.pricing's AWS Price List parsing and resolution.

These exercise the pure parsing/normalization logic directly (no network),
using recorded AWS Price List API samples in tests/fixtures/pricing/.

Regression content: fixtures catch bugs in pricePerUnit/unit JSON level handling,
model-key normalization against live Bedrock IDs, and unvalidated operator
price overrides (the 3 confirmed mispricing incidents that prompted this module).
"""

import asyncio
import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from stdapi import models, pricing
from stdapi.aws import AWSConnectionManager, get_client
from stdapi.config import SETTINGS
from stdapi.models.deprecation import DEPRECATED_MODELS
from stdapi.pricing import (
    Dimension,
    Price,
    PriceKey,
    Service,
    _apply_price_overrides,
    _apply_regional_fallback,
    _ingest_price_list_item,
    _region_family,
    _resolve_dimension,
    _resolve_tier,
    inference_type_to_dimension,
    is_model_priced,
    normalize_model_key,
    normalize_usagetype_model,
    parse_unit_scale,
    refresh_price_catalog_for_new_models,
    resolve_price,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "pricing"


def _ingest_fixture(
    name: str, service: Service
) -> tuple[dict[PriceKey, Price], list[str]]:
    """Ingest every item in a fixture file and return the resulting price index.

    Args:
        name: Fixture file name.
        service: AWS service (member of Service enum).

    Returns:
        (price index, diagnostics) -- diagnostics is shared across all
        items so intra-fixture collisions are recorded; tests assert it
        stays empty.
    """
    items = json.loads((_FIXTURES_DIR / name).read_text())
    results: dict[PriceKey, Price] = {}
    claims: dict[PriceKey, str] = {}  # Shared across items to detect collisions.
    diagnostics: list[str] = []
    for item in items:
        region = item["product"]["attributes"]["regionCode"]
        _ingest_price_list_item(
            json.dumps(item), service, region, "USD", results, claims, diagnostics
        )
    return results, diagnostics


class TestIngestPriceListItem:
    """Parsing of real-shaped AWS Price List API items."""

    def test_bedrock_fixture_produces_one_entry_per_item(self) -> None:
        """Every fixture item must yield exactly one price entry (none silently dropped)."""
        items = json.loads((_FIXTURES_DIR / "bedrock_sample.json").read_text())
        results, diagnostics = _ingest_fixture("bedrock_sample.json", Service.BEDROCK)
        assert diagnostics == []
        assert len(results) == len(items)

    def test_per_1k_token_price_is_divided_by_1000(self) -> None:
        """Regression: `unit: "1K tokens"` must scale the stored price, not be ignored."""
        results, diagnostics = _ingest_fixture("bedrock_sample.json", Service.BEDROCK)
        assert diagnostics == []
        key = PriceKey(
            Service.BEDROCK, "novalite", "us-east-1", Dimension.INPUT_TOKENS, "standard"
        )
        # Fixture raw price is 0.00008 USD per 1K tokens.
        assert results[key].amount == Decimal("0.00008") / 1000
        assert results[key].currency == "USD"

    def test_service_tier_attribute_becomes_price_key_tier(self) -> None:
        """A flex-tier row must be keyed under tier="flex", not merged into another row."""
        results, diagnostics = _ingest_fixture("bedrock_sample.json", Service.BEDROCK)
        assert diagnostics == []
        flex_key = PriceKey(
            Service.BEDROCK,
            "claudesonnet45",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "flex",
        )
        # The fixture's other claudesonnet45 row is global-routed, standard-tier.
        global_key = PriceKey(
            Service.BEDROCK,
            "claudesonnet45",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "global",
        )
        assert flex_key in results
        assert global_key in results
        assert results[flex_key] != results[global_key]

    def test_native_cross_region_global_usagetype_becomes_routing_global(self) -> None:
        """Regression: "-cross-region-global" usagetype must map to routing="global".

        These rows incorrectly modeled routing via a fabricated service_tier
        attribute before the fix. The actual signal is the usagetype suffix.
        """
        results, diagnostics = _ingest_fixture("bedrock_sample.json", Service.BEDROCK)
        assert diagnostics == []
        global_key = PriceKey(
            Service.BEDROCK,
            "claudesonnet45",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "global",
        )
        assert global_key in results
        assert results[global_key].amount == Decimal("0.003") / 1000

    def test_polly_fixture_matches_engine_based_key(self) -> None:
        """Polly rows (no `model` attribute) must key the same way record_polly_usage does."""
        results, diagnostics = _ingest_fixture("polly_sample.json", Service.POLLY)
        assert diagnostics == []
        key = PriceKey(
            Service.POLLY,
            normalize_model_key("amazon.polly-standard"),
            "us-east-1",
            Dimension.INPUT_CHARACTERS,
            "standard",
        )
        assert key in results
        assert results[key].amount == Decimal("0.000004")

    def test_transcribe_fixture_matches_synthetic_key(self) -> None:
        """Transcribe rows must key the same way record_transcribe_usage does."""
        results, diagnostics = _ingest_fixture(
            "transcribe_sample.json", Service.TRANSCRIBE
        )
        assert diagnostics == []
        key = PriceKey(
            Service.TRANSCRIBE,
            normalize_model_key("amazon.transcribe"),
            "us-east-1",
            Dimension.INPUT_SECONDS,
            "standard",
        )
        assert key in results
        assert results[key].amount == Decimal("0.0024")

    def test_marketplace_global_usagetype_becomes_routing_global(self) -> None:
        """Regression: bare "_Global" usagetype must map to routing="global".

        Before the fix these rows were skipped entirely (routing wasn't modeled),
        so models priced only via Global routing had no price at all.
        """
        results, diagnostics = _ingest_fixture("bedrock_sample.json", Service.BEDROCK)
        assert diagnostics == []
        key = PriceKey(
            Service.BEDROCK,
            normalize_model_key("Claude Opus 4.5"),
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "global",
        )
        assert key in results
        assert results[key].amount == Decimal("3.00") / 1_000_000

    def test_model_customization_rows_are_not_ingested(self) -> None:
        """Regression: "Model Customization" rows must be skipped (not billed as inference).

        This app only bills inference usage. Before the fix these rows weren't
        excluded, so they silently collided with the model's on-demand PriceKey.
        """
        results: dict[PriceKey, Price] = {}
        item = json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "us-east-1",
                        "usagetype": "USE1-NovaMicro-Model-Customization-input-tokens",
                        "inferenceType": "Input tokens",
                        "model": "Nova Micro",
                        "feature": "Model Customization",
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "Tokens",
                                    "pricePerUnit": {"USD": "0.001"},
                                }
                            }
                        }
                    }
                },
            }
        )
        _ingest_price_list_item(item, Service.BEDROCK, "us-east-1", "USD", results)
        assert results == {}

    def test_malformed_json_is_skipped_not_raised(self) -> None:
        """A single bad item must not raise -- callers rely on this to isolate failures."""
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            "{not valid json", Service.BEDROCK, "us-east-1", "USD", results
        )
        assert results == {}

    @pytest.mark.parametrize("bad_price", ["", "N/A", "NaN", "Infinity"])
    def test_invalid_or_non_finite_price_per_unit_is_skipped_not_raised(
        self, bad_price: str
    ) -> None:
        """Regression: an invalid/non-finite pricePerUnit must be skipped, not raise.

        ``Decimal("")``/``Decimal("N/A")`` raise ``decimal.InvalidOperation``;
        ``Decimal("NaN")``/``Decimal("Infinity")`` parse but aren't valid prices.
        """
        results: dict[PriceKey, Price] = {}
        item = json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "us-east-1",
                        "usagetype": "USE1-SomeModel-input-tokens",
                        "inferenceType": "Input tokens",
                        "model": "Some Model",
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "1K tokens",
                                    "pricePerUnit": {"USD": bad_price},
                                }
                            }
                        }
                    }
                },
            }
        )
        _ingest_price_list_item(item, Service.BEDROCK, "us-east-1", "USD", results)
        assert results == {}


class TestResolveTier:
    """AWS signals tier via 3 incompatible schemas -- _resolve_tier() must handle all 3."""

    def test_service_tier_attribute_wins_when_present(self) -> None:
        """Newer models (GPT-OSS, Nemotron, ...): explicit `service_tier` attribute."""
        assert _resolve_tier({"service_tier": "Flex"}) == "flex"

    def test_inference_type_suffix_when_no_service_tier_attribute(self) -> None:
        """Regression: tier suffix on inferenceType must be parsed when no service_tier attr."""
        assert _resolve_tier({"inferenceType": "Output tokens flex"}) == "flex"
        assert _resolve_tier({"inferenceType": "Input tokens priority"}) == "priority"

    def test_batch_feature_with_no_tier_hint_on_inference_type(self) -> None:
        """Some older models' batch rows have no inferenceType suffix at all.

        Batch is only signaled by `feature` == "Batch Inference".
        """
        assert (
            _resolve_tier(
                {"inferenceType": "Input tokens", "feature": "Batch Inference"}
            )
            == "batch"
        )

    def test_defaults_to_standard_with_no_tier_signal_at_all(self) -> None:
        """No service_tier, no inferenceType suffix, no Batch Inference feature -> standard."""
        attrs = {"inferenceType": "Input tokens", "feature": "On-demand Inference"}
        assert _resolve_tier(attrs) == "standard"
        assert _resolve_tier({}) == "standard"

    def test_batch_feature_without_a_space_is_also_recognized(self) -> None:
        """Regression: "BatchInference" (no space) must also match."""
        assert (
            _resolve_tier(
                {"inferenceType": "Input tokens", "feature": "BatchInference"}
            )
            == "batch"
        )

    def test_usagetype_only_batch_signal_with_no_other_hint(self) -> None:
        """Regression: batch suffix in usagetype must be detected when no other signal exists.

        Confirmed live on Nova 2.0 Lite/Pro global-batch rows -- no service_tier,
        no inferenceType suffix, no distinct feature. Before the fix, these rows
        collided with global-standard on the same PriceKey.
        """
        attrs = {
            "inferenceType": "Output tokens",
            "feature": "On-demand Inference",
            "usagetype": "USE1-Nova2.0Lite-output-tokens-cross-region-global-batch",
        }
        assert _resolve_tier(attrs) == "batch"


class TestPriceCollisionDetection:
    """Detecting when two different price-list rows claim the same PriceKey.

    This is a systemic guard that catches the exact bug class behind several
    confirmed mispricing incidents (BatchInference spelling, Model Customization,
    Nova Canvas, Nova 2.0 global-batch): two unrelated rows silently resolve
    to identical PriceKeys, with the wrong one winning depending on ingestion order.
    """

    def test_different_usagetype_same_key_different_price_warns(self) -> None:
        """Two different rows resolving to the same PriceKey with different prices must warn."""
        results: dict[PriceKey, Price] = {}
        claims: dict[PriceKey, str] = {}
        diagnostics: list[str] = []
        key = PriceKey(
            Service.BEDROCK, "modelc", "us-east-1", Dimension.INPUT_TOKENS, "standard"
        )
        pricing._store_price(  # noqa: SLF001
            results,
            claims,
            key,
            Price(Decimal("0.003"), "USD"),
            "usagetype-a",
            diagnostics,
        )
        pricing._store_price(  # noqa: SLF001
            results,
            claims,
            key,
            Price(Decimal("0.0015"), "USD"),
            "usagetype-b",
            diagnostics,
        )
        assert any("collision" in d.lower() for d in diagnostics)
        assert results[key].amount == Decimal("0.0015")  # last write still wins

    def test_same_usagetype_repeated_price_band_does_not_warn(self) -> None:
        """Multiple price-band terms from the SAME row (tiered volume pricing) must not warn."""
        results: dict[PriceKey, Price] = {}
        claims: dict[PriceKey, str] = {}
        diagnostics: list[str] = []
        key = PriceKey(
            Service.BEDROCK, "modeld", "us-east-1", Dimension.INPUT_TOKENS, "standard"
        )
        pricing._store_price(  # noqa: SLF001
            results,
            claims,
            key,
            Price(Decimal("0.003"), "USD"),
            "usagetype-a",
            diagnostics,
        )
        pricing._store_price(  # noqa: SLF001
            results,
            claims,
            key,
            Price(Decimal("0.0025"), "USD"),
            "usagetype-a",
            diagnostics,
        )
        assert diagnostics == []

    def test_same_usagetype_same_price_does_not_warn(self) -> None:
        """A byte-identical re-ingestion (e.g. AWS pagination overlap) must not warn."""
        results: dict[PriceKey, Price] = {}
        claims: dict[PriceKey, str] = {}
        diagnostics: list[str] = []
        key = PriceKey(
            Service.BEDROCK, "modele", "us-east-1", Dimension.INPUT_TOKENS, "standard"
        )
        price = Price(Decimal("0.003"), "USD")
        pricing._store_price(results, claims, key, price, "usagetype-a", diagnostics)  # noqa: SLF001
        pricing._store_price(results, claims, key, price, "usagetype-a", diagnostics)  # noqa: SLF001
        assert diagnostics == []


class _FakePricingPaginator:
    """Minimal fake aiobotocore pricing paginator serving items per ServiceCode."""

    def __init__(
        self,
        items_by_service_code: dict[str, list[dict[str, object]]],
        delay_by_service_code: dict[str, float],
    ) -> None:
        self._items = items_by_service_code
        self._delays = delay_by_service_code

    async def paginate(self, **kwargs: object) -> AsyncIterator[dict[str, object]]:
        """Yield one page of the ServiceCode's items, after its optional delay."""
        service_code = str(kwargs["ServiceCode"])
        if delay := self._delays.get(service_code, 0.0):
            await asyncio.sleep(delay)
        items = self._items.get(service_code, [])
        yield {"PriceList": [json.dumps(item) for item in items]}


class _FakePricingClient:
    """Minimal fake aiobotocore pricing client serving items per ServiceCode."""

    def __init__(
        self,
        items_by_service_code: dict[str, list[dict[str, object]]],
        delay_by_service_code: dict[str, float] | None = None,
    ) -> None:
        self._paginator = _FakePricingPaginator(
            items_by_service_code, delay_by_service_code or {}
        )

    def get_paginator(self, _name: str) -> _FakePricingPaginator:
        """Return the paginator serving this client's registered items."""
        return self._paginator


class TestCrossFetchCollisionDetection:
    """Collisions between two fetches in the same catalog load.

    Same-region Bedrock service codes claiming one PriceKey must warn and
    resolve deterministically (fetch generation order, not completion order).
    """

    #: The PriceKey both fetches' items below resolve to.
    _KEY = PriceKey(
        Service.BEDROCK, "somemodel", "us-east-1", Dimension.INPUT_TOKENS, "standard"
    )

    @staticmethod
    def _bedrock_item(usagetype: str, price: str) -> dict[str, object]:
        """Build one Bedrock price-list item resolving to ``_KEY``."""
        return {
            "product": {
                "attributes": {
                    "regionCode": "us-east-1",
                    "usagetype": usagetype,
                    "inferenceType": "Input tokens",
                    "model": "Some Model",
                }
            },
            "terms": {
                "OnDemand": {
                    "SKU1": {
                        "priceDimensions": {
                            "SKU1.A": {
                                "unit": "1K tokens",
                                "pricePerUnit": {"USD": price},
                            }
                        }
                    }
                }
            },
        }

    async def _load_catalog(
        self,
        monkeypatch: pytest.MonkeyPatch,
        items_by_service_code: dict[str, list[dict[str, object]]],
        delay_by_service_code: dict[str, float] | None = None,
    ) -> tuple[dict[PriceKey, Price], list[str]]:
        """Run a full _load_price_catalog against a fake pricing client."""
        client = _FakePricingClient(items_by_service_code, delay_by_service_code)
        monkeypatch.setattr(pricing, "get_client", lambda *_a, **_k: client)
        monkeypatch.setattr(pricing, "_catalog_regions", lambda: {"us-east-1"})
        monkeypatch.setattr(SETTINGS, "cost_price_overrides", {})
        monkeypatch.setattr(pricing._state, "price_index", {})  # noqa: SLF001
        diagnostics: list[str] = []
        await pricing._load_price_catalog(diagnostics)  # noqa: SLF001
        return pricing._state.price_index, diagnostics  # noqa: SLF001

    async def test_two_service_codes_claiming_the_same_price_key_warns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two fetches resolving to the same PriceKey must warn, later fetch winning."""
        index, diagnostics = await self._load_catalog(
            monkeypatch,
            {
                "AmazonBedrock": [
                    self._bedrock_item("USE1-SomeModel-input-tokens", "0.001")
                ],
                "AmazonBedrockService": [
                    self._bedrock_item("USW2-SomeModel-input-tokens", "0.002")
                ],
            },
        )
        assert any("collision" in d.lower() for d in diagnostics)
        assert index[self._KEY].amount == Decimal("0.002") / 1000

    async def test_same_usagetype_different_price_across_fetches_warns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Identical usagetype text across two fetches must still warn.

        The merge tags claims with the service code, so same-usagetype
        cross-fetch stores aren't mistaken for repeated price bands.
        """
        index, diagnostics = await self._load_catalog(
            monkeypatch,
            {
                "AmazonBedrock": [
                    self._bedrock_item("USE1-SomeModel-input-tokens", "0.001")
                ],
                "AmazonBedrockService": [
                    self._bedrock_item("USE1-SomeModel-input-tokens", "0.002")
                ],
            },
        )
        assert any("collision" in d.lower() for d in diagnostics)
        assert index[self._KEY].amount == Decimal("0.002") / 1000

    async def test_winner_is_deterministic_regardless_of_completion_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The colliding winner must follow fetch generation order.

        AmazonBedrock is delayed to complete last; AmazonBedrockService
        (later in ``_SERVICE_CODE_TO_SERVICE`` order) must still win.
        """
        index, diagnostics = await self._load_catalog(
            monkeypatch,
            {
                "AmazonBedrock": [
                    self._bedrock_item("USE1-SomeModel-input-tokens", "0.001")
                ],
                "AmazonBedrockService": [
                    self._bedrock_item("USW2-SomeModel-input-tokens", "0.002")
                ],
            },
            delay_by_service_code={"AmazonBedrock": 0.05},
        )
        assert any("collision" in d.lower() for d in diagnostics)
        assert index[self._KEY].amount == Decimal("0.002") / 1000


class TestNativeCacheTtl:
    """Native (non-Marketplace) 1-hour prompt-cache-write pricing."""

    def test_one_hour_usagetype_becomes_cache_ttl_1h(self) -> None:
        """A "1 hour" usagetype row must be keyed under cache_ttl="1h", not merged."""
        results: dict[PriceKey, Price] = {}
        item = json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "us-east-1",
                        "usagetype": "USE1-Claude4Sonnet-cache-write-input-token-count-1-hour",
                        "inferenceType": "Prompt cache write input tokens",
                        "model": "Claude Sonnet 4.5",
                        "feature": "On-demand Inference",
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "1K tokens",
                                    "pricePerUnit": {"USD": "0.0075"},
                                }
                            }
                        }
                    }
                },
            }
        )
        _ingest_price_list_item(item, Service.BEDROCK, "us-east-1", "USD", results)
        key = PriceKey(
            Service.BEDROCK,
            "claudesonnet45",
            "us-east-1",
            Dimension.CACHE_WRITE_TOKENS,
            "standard",
            "1h",
        )
        assert key in results
        assert results[key].amount == Decimal("0.0075") / 1000


class TestMarketplaceCacheTtl:
    """Marketplace-listed models' 5-minute (default) vs 1-hour prompt-cache-write pricing."""

    @staticmethod
    def _ingest_marketplace_cache_row(
        usagetype: str, price: str
    ) -> dict[PriceKey, Price]:
        results: dict[PriceKey, Price] = {}
        item = json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "us-east-1",
                        "usagetype": usagetype,
                        "servicename": "Claude Opus 4.5 (Amazon Bedrock Edition)",
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "Units",
                                    "pricePerUnit": {"USD": price},
                                }
                            }
                        }
                    }
                },
            }
        )
        _ingest_price_list_item(item, Service.BEDROCK, "us-east-1", "USD", results)
        return results

    def test_1h_and_5m_default_are_distinct_keys(self) -> None:
        """A 1h cache-write row and the default (5m) row must not collide."""
        one_hour = self._ingest_marketplace_cache_row(
            "USE1-MP:USE1_CacheWrite1hInputTokenCount-Units", "6.60"
        )
        default = self._ingest_marketplace_cache_row(
            "USE1-MP:USE1_CacheWriteInputTokenCount-Units", "4.125"
        )
        model = normalize_model_key("Claude Opus 4.5")
        one_hour_key = PriceKey(
            Service.BEDROCK,
            model,
            "us-east-1",
            Dimension.CACHE_WRITE_TOKENS,
            "standard",
            "1h",
        )
        default_key = PriceKey(
            Service.BEDROCK,
            model,
            "us-east-1",
            Dimension.CACHE_WRITE_TOKENS,
            "standard",
            "",
        )
        assert one_hour[one_hour_key].amount == Decimal("6.60") / 1_000_000
        assert default[default_key].amount == Decimal("4.125") / 1_000_000


class TestTitanImageGeneratorIngestion:
    """Titan Image Generator's titanModel/T2I-I2I/resolution/quality pricing rows."""

    def test_t2i_and_i2i_share_the_same_image_spec_key(self) -> None:
        """T2I and I2I are priced identically at each resolution/quality (live-confirmed)."""
        results, diagnostics = _ingest_fixture(
            "titan_image_generator_sample.json", Service.BEDROCK
        )
        assert diagnostics == []
        key = PriceKey(
            Service.BEDROCK,
            "titanimagegeneratorg1",
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
            "standard",
            "",
            "",
            "1024:standard",
        )
        assert key in results
        assert results[key].amount == Decimal("0.01")

    def test_premium_quality_is_a_distinct_key(self) -> None:
        """Standard and Premium at the same resolution must not collide."""
        results, diagnostics = _ingest_fixture(
            "titan_image_generator_sample.json", Service.BEDROCK
        )
        assert diagnostics == []
        standard_key = PriceKey(
            Service.BEDROCK,
            "titanimagegeneratorg1",
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
            "standard",
            "",
            "",
            "1024:standard",
        )
        premium_key = PriceKey(
            Service.BEDROCK,
            "titanimagegeneratorg1",
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
            "standard",
            "",
            "",
            "1024:premium",
        )
        assert results[standard_key] != results[premium_key]

    def test_custom_fine_tuned_variant_is_skipped(self) -> None:
        """ "Custom " rows (fine-tuned model invocation) must not be ingested at all.

        This app never invokes customized/fine-tuned Titan Image Generator
        models -- ingesting this row under the base model's key would
        silently overprice (or, depending on dict ordering, underprice) it.
        """  # noqa: D210
        results, diagnostics = _ingest_fixture(
            "titan_image_generator_sample.json", Service.BEDROCK
        )
        assert diagnostics == []
        assert not any(
            key.model == "titanimagegeneratorg1" and key.spec == "" for key in results
        )
        # T2I standard and I2I standard collapse into one key (same price); plus T2I
        # premium -- 2 distinct keys total, Custom excluded.
        assert len(results) == 2


class TestNovaCanvasIngestion:
    """Nova Canvas T2I/I2I pricing with image_spec resolution/quality keys.

    Regression: Nova Canvas carries a `model` attribute (unlike Titan Image
    Generator), so it took the wrong branch in _resolve_native_model and its
    inferenceType wasn't recognized -- real on-demand rows were silently dropped.
    Meanwhile Provisioned Throughput rows matched the usagetype fallback and
    silently overwrote the real image prices. Confirmed live on Nova 2.0 Omni/Pro.
    """

    def test_on_demand_rows_are_ingested_under_their_image_spec(self) -> None:
        """T2I/I2I on-demand rows must resolve to distinct, correctly-priced image_spec keys."""
        results, diagnostics = _ingest_fixture(
            "nova_canvas_sample.json", Service.BEDROCK
        )
        assert diagnostics == []
        t2i_key = PriceKey(
            Service.BEDROCK,
            "amazonnovacanvas",
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
            "standard",
            "",
            "",
            "1024:standard",
        )
        i2i_key = PriceKey(
            Service.BEDROCK,
            "amazonnovacanvas",
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
            "standard",
            "",
            "",
            "2048:standard",
        )
        assert results[t2i_key].amount == Decimal("0.04")
        assert results[i2i_key].amount == Decimal("0.06")

    def test_provisioned_throughput_row_is_not_ingested(self) -> None:
        """The Provisioned Throughput row must not collide with (or replace) the real image price."""
        results, diagnostics = _ingest_fixture(
            "nova_canvas_sample.json", Service.BEDROCK
        )
        assert diagnostics == []
        # Only the 2 real on-demand rows -- the Provisioned Throughput
        # "ModelUnits" row must not appear at all, under any key.
        assert len(results) == 2
        assert Decimal("30.25") not in {price.amount for price in results.values()}


class TestLumaRayIngestion:
    """Luma Ray rows: bare "Video" inferenceType, "Ray v2" model, HDRes/StandardRes."""

    @staticmethod
    def _item(usagetype: str, price: str) -> str:
        return json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "us-west-2",
                        "usagetype": usagetype,
                        "inferenceType": "Video",
                        "model": "Ray v2",
                        "provider": "Luma AI",
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "Second",
                                    "pricePerUnit": {"USD": price},
                                }
                            }
                        }
                    }
                },
            }
        )

    def test_bare_video_inference_type_maps_to_output_seconds(self) -> None:
        """Luma's bare "Video" inferenceType must resolve to Dimension.OUTPUT_SECONDS."""
        assert (
            inference_type_to_dimension(
                "Video", "USW2-Ray-V2-Medfps-HDRes", Service.BEDROCK
            )
            == Dimension.OUTPUT_SECONDS
        )

    def test_model_id_and_price_row_normalize_to_the_same_key(self) -> None:
        """The model ID and the price row's "Ray v2" name must share one price key."""
        assert normalize_model_key("luma.ray-v2:0") == normalize_model_key("Ray v2")

    def test_hd_and_standard_rows_ingest_under_distinct_spec_buckets(self) -> None:
        """HDRes ingests under spec="hd" and StandardRes under the flat bucket."""
        results: dict[PriceKey, Price] = {}
        claims: dict[PriceKey, str] = {}
        diagnostics: list[str] = []
        for usagetype, price in (
            ("USW2-Ray-V2-Medfps-StandardRes", "0.75"),
            ("USW2-Ray-V2-Medfps-HDRes", "1.50"),
        ):
            _ingest_price_list_item(
                self._item(usagetype, price),
                Service.BEDROCK,
                "us-west-2",
                "USD",
                results,
                claims,
                diagnostics,
            )
        assert diagnostics == []
        model = normalize_model_key("Ray v2")
        standard_key = PriceKey(
            Service.BEDROCK, model, "us-west-2", Dimension.OUTPUT_SECONDS, "standard"
        )
        hd_key = PriceKey(
            Service.BEDROCK,
            model,
            "us-west-2",
            Dimension.OUTPUT_SECONDS,
            "standard",
            "",
            "",
            "hd",
        )
        assert results[standard_key].amount == Decimal("0.75")
        assert results[hd_key].amount == Decimal("1.50")

    def test_resolve_price_round_trip_for_the_luma_model_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """resolve_price() must find the ingested HD row via the Bedrock model ID."""
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item("USW2-Ray-V2-Medfps-HDRes", "1.50"),
            Service.BEDROCK,
            "us-west-2",
            "USD",
            results,
        )
        monkeypatch.setattr(pricing._state, "price_index", results)  # noqa: SLF001
        price = resolve_price(
            Service.BEDROCK,
            "luma.ray-v2:0",
            "us-west-2",
            Dimension.OUTPUT_SECONDS,
            spec="hd",
        )
        assert price is not None
        assert price.amount == Decimal("1.50")


class TestImageTokenCountIsNotAnImageDimension:
    """Multimodal "image token count" must not resolve to OUTPUT_IMAGES."""

    def test_image_token_count_usagetype_does_not_match_output_images(self) -> None:
        """ "Image Token Count" inferenceType/usagetype pairs must resolve to no dimension."""  # noqa: D210
        assert (
            inference_type_to_dimension(
                "Input Image Token Count",
                "USE1-Nova2.0Omni-input-image-token-count",
                Service.BEDROCK,
            )
            is None
        )
        assert (
            inference_type_to_dimension(
                "Output Image Token Count",
                "USE1-Nova2.0Omni-output-image-token-count",
                Service.BEDROCK,
            )
            is None
        )

    def test_unit_and_image_fallback_still_matches_when_no_token_present(self) -> None:
        """The "token" exclusion must not affect usagetypes with no "token" substring."""
        assert (
            inference_type_to_dimension(
                "",
                "USE1-TitanImageGeneratorG1-ProvisionedThroughput-NoCommit-ModelUnits",
                Service.BEDROCK,
            )
            == Dimension.OUTPUT_IMAGES
        )

    @pytest.mark.parametrize(
        ("service", "expected_dimension"),
        [
            (Service.TRANSLATE, Dimension.INPUT_CHARACTERS),
            (Service.COMPREHEND, Dimension.COMPREHEND_UNITS),
            (Service.POLLY, Dimension.INPUT_CHARACTERS),
        ],
    )
    def test_non_bedrock_unit_image_usagetype_resolves_to_service_fallback(
        self, service: Service, expected_dimension: Dimension
    ) -> None:
        """Regression: non-Bedrock "unit"/"image" usagetypes must use the service fallback.

        The "unit"/"image" substring heuristic is Bedrock-only; other
        services' rows must resolve to their fallback dimension, not
        OUTPUT_IMAGES.
        """
        assert (
            inference_type_to_dimension("", "USE1-SomeUnitOrImageUsage", service)
            is None
        )
        assert (
            _resolve_dimension(service, "", "", "USE1-SomeUnitOrImageUsage")
            == expected_dimension
        )


class TestNova2GlobalBatchIngestion:
    """Nova 2.0 global-routed batch rows, signaled only by usagetype."""

    def test_global_standard_and_global_batch_are_distinct_keys(self) -> None:
        """Both rows must resolve to their own price, not collapse onto one key."""
        results, diagnostics = _ingest_fixture(
            "nova2_global_batch_sample.json", Service.BEDROCK
        )
        assert diagnostics == []
        standard_key = PriceKey(
            Service.BEDROCK,
            "nova20lite",
            "us-east-1",
            Dimension.OUTPUT_TOKENS,
            "standard",
            "",
            "global",
        )
        batch_key = PriceKey(
            Service.BEDROCK,
            "nova20lite",
            "us-east-1",
            Dimension.OUTPUT_TOKENS,
            "batch",
            "",
            "global",
        )
        assert results[standard_key].amount == Decimal("0.0000025")
        assert results[batch_key].amount == Decimal("0.00000125")


class TestMarketplaceCacheWriteOneHourNewGeneration:
    """Marketplace's newer "cache_write_tokens_1h" usagetype generation.

    Older listings signal 1h cache-write via "CacheWrite1hInputTokenCount"
    (see TestMarketplaceCacheTtl); newer listings use "cache_write_tokens_1h"
    instead, which the older pattern doesn't match.
    """

    @staticmethod
    def _item(
        usagetype: str,
        price: str,
        servicename: str = "Claude Opus 4.5 (Amazon Bedrock Edition)",
    ) -> str:
        return json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "eu-west-1",
                        "usagetype": usagetype,
                        "servicename": servicename,
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "Units",
                                    "pricePerUnit": {"USD": price},
                                }
                            }
                        }
                    }
                },
            }
        )

    def test_1h_new_generation_usagetype_maps_to_cache_write_1h(self) -> None:
        """ "cache_write_tokens_1h_standard" must resolve to cache_ttl="1h"."""  # noqa: D210
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item("EU-MP:EU_cache_write_tokens_1h_standard-Units", "6.60"),
            Service.BEDROCK,
            "eu-west-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.BEDROCK,
            normalize_model_key("Claude Opus 4.5"),
            "eu-west-1",
            Dimension.CACHE_WRITE_TOKENS,
            "standard",
            "1h",
        )
        assert key in results
        assert results[key].amount == Decimal("6.60") / 1_000_000

    def test_base_new_generation_usagetype_maps_to_cache_write_default(self) -> None:
        """ "cache_write_tokens_standard" (no "1h") must resolve to cache_ttl=""."""  # noqa: D210
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item("EU-MP:EU_cache_write_tokens_standard-Units", "4.125"),
            Service.BEDROCK,
            "eu-west-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.BEDROCK,
            normalize_model_key("Claude Opus 4.5"),
            "eu-west-1",
            Dimension.CACHE_WRITE_TOKENS,
            "standard",
            "",
        )
        assert key in results
        assert results[key].amount == Decimal("4.125") / 1_000_000

    def test_global_1h_variant_maps_to_global_routing_and_1h_ttl(self) -> None:
        """The global-routed 1h variant must combine routing="global" with cache_ttl="1h"."""
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item("EU-MP:EU_cache_write_tokens_1h_global_standard-Units", "6.00"),
            Service.BEDROCK,
            "eu-west-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.BEDROCK,
            normalize_model_key("Claude Opus 4.5"),
            "eu-west-1",
            Dimension.CACHE_WRITE_TOKENS,
            "standard",
            "1h",
            "global",
        )
        assert key in results
        assert results[key].amount == Decimal("6.00") / 1_000_000

    def test_base_and_1h_rows_do_not_collide(self) -> None:
        """Ingesting both the base and 1h rows together must produce 2 distinct keys, no warning."""
        results: dict[PriceKey, Price] = {}
        claims: dict[PriceKey, str] = {}
        diagnostics: list[str] = []
        _ingest_price_list_item(
            self._item("EU-MP:EU_cache_write_tokens_1h_standard-Units", "6.60"),
            Service.BEDROCK,
            "eu-west-1",
            "USD",
            results,
            claims,
            diagnostics,
        )
        _ingest_price_list_item(
            self._item("EU-MP:EU_cache_write_tokens_standard-Units", "4.125"),
            Service.BEDROCK,
            "eu-west-1",
            "USD",
            results,
            claims,
            diagnostics,
        )
        assert diagnostics == []
        model = normalize_model_key("Claude Opus 4.5")
        one_hour_key = PriceKey(
            Service.BEDROCK,
            model,
            "eu-west-1",
            Dimension.CACHE_WRITE_TOKENS,
            "standard",
            "1h",
        )
        default_key = PriceKey(
            Service.BEDROCK,
            model,
            "eu-west-1",
            Dimension.CACHE_WRITE_TOKENS,
            "standard",
            "",
        )
        assert results[one_hour_key] != results[default_key]


class TestMarketplaceLegacyContextWindowListingsSkipped:
    """Legacy "(100K)"-suffixed Marketplace listings must not be ingested at all."""

    @staticmethod
    def _item(servicename: str, price: str = "3.26") -> str:
        return json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "us-east-1",
                        "usagetype": "USE1-MP:USE1_InputTokenCount-Units",
                        "servicename": servicename,
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "Units",
                                    "pricePerUnit": {"USD": price},
                                }
                            }
                        }
                    }
                },
            }
        )

    def test_100k_context_window_listing_is_not_ingested(self) -> None:
        """ "Claude Instant (100K) (Amazon Bedrock Edition)" must yield no price entry."""  # noqa: D210
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item("Claude Instant (100K) (Amazon Bedrock Edition)"),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
        )
        assert results == {}

    def test_base_listing_without_context_window_suffix_ingests_normally(self) -> None:
        """The base "Claude Instant" listing (no "(100K)" suffix) must ingest as usual."""
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item("Claude Instant (Amazon Bedrock Edition)", "0.80"),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.BEDROCK,
            normalize_model_key("Claude Instant"),
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        assert key in results
        assert results[key].amount == Decimal("0.80") / 1_000_000


class TestMarketplaceClaimIncludesListingName:
    """Marketplace collision claims are tagged with the listing name, not just the usagetype.

    Distinct products can share an identical usagetype string; two listings
    that normalize to the SAME model key but carry different prices must now
    be caught as a collision (previously silent, since the claim was keyed
    on usagetype text alone).
    """

    @staticmethod
    def _item(servicename: str, price: str) -> str:
        return json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "us-west-2",
                        "usagetype": "USW2-MP:USW2_InputTokenCount-Units",
                        "servicename": servicename,
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "Units",
                                    "pricePerUnit": {"USD": price},
                                }
                            }
                        }
                    }
                },
            }
        )

    def test_same_model_key_different_listing_same_usagetype_warns(self) -> None:
        """ "Claude 3.5 Sonnet" and "... v2" both normalize to "claude35sonnet"."""  # noqa: D210
        results: dict[PriceKey, Price] = {}
        claims: dict[PriceKey, str] = {}
        diagnostics: list[str] = []
        _ingest_price_list_item(
            self._item("Claude 3.5 Sonnet (Amazon Bedrock Edition)", "3.00"),
            Service.BEDROCK,
            "us-west-2",
            "USD",
            results,
            claims,
            diagnostics,
        )
        _ingest_price_list_item(
            self._item("Claude 3.5 Sonnet v2 (Amazon Bedrock Edition)", "3.30"),
            Service.BEDROCK,
            "us-west-2",
            "USD",
            results,
            claims,
            diagnostics,
        )
        assert any("collision" in d.lower() for d in diagnostics)


class TestMarketplaceLatencyOptimizedRouting:
    """Marketplace "_LatencyOptimized" usagetype rows -- a distinct, pricier serving profile."""

    @staticmethod
    def _item(usagetype: str, price: str) -> str:
        return json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "us-west-2",
                        "usagetype": usagetype,
                        "servicename": "Claude Opus 4.5 (Amazon Bedrock Edition)",
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "Units",
                                    "pricePerUnit": {"USD": price},
                                }
                            }
                        }
                    }
                },
            }
        )

    def test_latency_optimized_usagetype_ingests_with_routing_latency(self) -> None:
        """The "_LatencyOptimized" marker must resolve to routing="latency"."""
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item("USW2-MP:USW2_InputTokenCount_LatencyOptimized-Units", "3.60"),
            Service.BEDROCK,
            "us-west-2",
            "USD",
            results,
        )
        key = PriceKey(
            Service.BEDROCK,
            normalize_model_key("Claude Opus 4.5"),
            "us-west-2",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "latency",
        )
        assert key in results
        assert results[key].amount == Decimal("3.60") / 1_000_000

    def test_latency_optimized_and_plain_rows_do_not_collide(self) -> None:
        """The latency-optimized row and the plain row must resolve to distinct keys."""
        results: dict[PriceKey, Price] = {}
        claims: dict[PriceKey, str] = {}
        diagnostics: list[str] = []
        _ingest_price_list_item(
            self._item("USW2-MP:USW2_InputTokenCount_LatencyOptimized-Units", "3.60"),
            Service.BEDROCK,
            "us-west-2",
            "USD",
            results,
            claims,
            diagnostics,
        )
        _ingest_price_list_item(
            self._item("USW2-MP:USW2_InputTokenCount-Units", "3.00"),
            Service.BEDROCK,
            "us-west-2",
            "USD",
            results,
            claims,
            diagnostics,
        )
        assert diagnostics == []
        model = normalize_model_key("Claude Opus 4.5")
        latency_key = PriceKey(
            Service.BEDROCK,
            model,
            "us-west-2",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "latency",
        )
        plain_key = PriceKey(
            Service.BEDROCK, model, "us-west-2", Dimension.INPUT_TOKENS, "standard"
        )
        assert results[latency_key].amount == Decimal("3.60") / 1_000_000
        assert results[plain_key].amount == Decimal("3.00") / 1_000_000


class TestLongContextAxis:
    """Long-context (>200K prompt) pricing bucket, signaled by a "long-context" usagetype segment."""

    def test_long_context_input_tokens_row_does_not_collide_with_standard_row(
        self,
    ) -> None:
        """The long-context row must index separately from the standard global row."""
        results: dict[PriceKey, Price] = {}
        claims: dict[PriceKey, str] = {}
        diagnostics: list[str] = []
        standard_item = json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "us-east-1",
                        "usagetype": "USE1-Claude4Sonnet-input-tokens-cross-region-global",
                        "inferenceType": "Input tokens",
                        "model": "Claude Sonnet 4",
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "1K tokens",
                                    "pricePerUnit": {"USD": "0.003"},
                                }
                            }
                        }
                    }
                },
            }
        )
        long_item = json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "us-east-1",
                        "usagetype": (
                            "USE1-Claude4Sonnet-input-tokens-long-context"
                            "-cross-region-global"
                        ),
                        "inferenceType": "Input tokens long context",
                        "model": "Claude Sonnet 4",
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "1K tokens",
                                    "pricePerUnit": {"USD": "0.006"},
                                }
                            }
                        }
                    }
                },
            }
        )
        _ingest_price_list_item(
            standard_item,
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
            claims,
            diagnostics,
        )
        _ingest_price_list_item(
            long_item, Service.BEDROCK, "us-east-1", "USD", results, claims, diagnostics
        )
        assert diagnostics == []
        model = normalize_model_key("Claude Sonnet 4")
        standard_key = PriceKey(
            Service.BEDROCK,
            model,
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "global",
        )
        long_key = PriceKey(
            Service.BEDROCK,
            model,
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "global",
            "",
            "long",
        )
        assert results[standard_key].amount == Decimal("0.003") / 1000
        assert results[long_key].amount == Decimal("0.006") / 1000

    def test_long_context_cache_read_row_does_not_collide_with_standard_row(
        self,
    ) -> None:
        """Cache-variant rows (featuretype-driven) must also respect the context axis."""
        results: dict[PriceKey, Price] = {}
        claims: dict[PriceKey, str] = {}
        diagnostics: list[str] = []
        standard_item = json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "us-east-1",
                        "usagetype": (
                            "USE1-Claude4Sonnet-cache-read-input-token-count"
                            "-cross-region-global"
                        ),
                        "featuretype": "Prompt cache read",
                        "model": "Claude Sonnet 4",
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "1K tokens",
                                    "pricePerUnit": {"USD": "0.0003"},
                                }
                            }
                        }
                    }
                },
            }
        )
        long_item = json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "us-east-1",
                        "usagetype": (
                            "USE1-Claude4Sonnet-cache-read-input-token-count"
                            "-long-context-cross-region-global"
                        ),
                        "featuretype": "Prompt cache read",
                        "model": "Claude Sonnet 4",
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "1K tokens",
                                    "pricePerUnit": {"USD": "0.0006"},
                                }
                            }
                        }
                    }
                },
            }
        )
        _ingest_price_list_item(
            standard_item,
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
            claims,
            diagnostics,
        )
        _ingest_price_list_item(
            long_item, Service.BEDROCK, "us-east-1", "USD", results, claims, diagnostics
        )
        assert diagnostics == []
        model = normalize_model_key("Claude Sonnet 4")
        standard_key = PriceKey(
            Service.BEDROCK,
            model,
            "us-east-1",
            Dimension.CACHE_READ_TOKENS,
            "standard",
            "",
            "global",
        )
        long_key = PriceKey(
            Service.BEDROCK,
            model,
            "us-east-1",
            Dimension.CACHE_READ_TOKENS,
            "standard",
            "",
            "global",
            "",
            "long",
        )
        assert results[standard_key].amount == Decimal("0.0003") / 1000
        assert results[long_key].amount == Decimal("0.0006") / 1000


class TestNovaMultiModalEmbeddingsInputMediaExclusion:
    """Nova Multimodal Embeddings' per-input-image SKUs are not billed usage; its per-input-tokens SKU is."""

    @staticmethod
    def _item(usagetype: str, unit: str, price: str) -> str:
        return json.dumps(
            {
                "product": {
                    "attributes": {"regionCode": "us-east-1", "usagetype": usagetype}
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {"unit": unit, "pricePerUnit": {"USD": price}}
                            }
                        }
                    }
                },
            }
        )

    @pytest.mark.parametrize(
        ("usagetype", "dimension", "spec"),
        [
            (
                "USE1-NovaMultiModalEmbeddings-input-document-image",
                Dimension.INPUT_IMAGES,
                "document",
            ),
            (
                "USE1-NovaMultiModalEmbeddings-input-standard-image",
                Dimension.INPUT_IMAGES,
                "",
            ),
            (
                "USE1-NovaMultiModalEmbeddings-input-audio-second",
                Dimension.INPUT_SECONDS,
                "audio",
            ),
            (
                "USE1-NovaMultiModalEmbeddings-input-video-second",
                Dimension.INPUT_SECONDS,
                "video",
            ),
        ],
    )
    def test_input_media_rows_ingest_with_spec(
        self, usagetype: str, dimension: Dimension, spec: str
    ) -> None:
        """Per-input-media rows index under INPUT_IMAGES/INPUT_SECONDS spec buckets."""
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item(usagetype, "Images Processed", "0.0008"),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.BEDROCK,
            "novamultimodalembeddings",
            "us-east-1",
            dimension,
            "standard",
            "",
            "",
            spec,
        )
        assert set(results) == {key}

    def test_input_tokens_row_ingests_via_usagetype_token_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ "input-tokens" (no inferenceType) must ingest as INPUT_TOKENS."""  # noqa: D210
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item(
                "USE1-NovaMultiModalEmbeddings-input-tokens", "1K tokens", "0.00002"
            ),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
        )
        monkeypatch.setattr(pricing._state, "price_index", results)  # noqa: SLF001
        price = resolve_price(
            Service.BEDROCK,
            "amazon.nova-2-multimodal-embeddings-v1:0",
            "us-east-1",
            Dimension.INPUT_TOKENS,
        )
        assert price is not None
        assert price.amount == Decimal("0.00002") / 1000

    def test_batch_suffix_on_input_tokens_row_becomes_batch_tier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ "input-tokens-batch" must be keyed under tier="batch"."""  # noqa: D210
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item(
                "USE1-NovaMultiModalEmbeddings-input-tokens-batch",
                "1K tokens",
                "0.00001",
            ),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
        )
        monkeypatch.setattr(pricing._state, "price_index", results)  # noqa: SLF001
        price = resolve_price(
            Service.BEDROCK,
            "amazon.nova-2-multimodal-embeddings-v1:0",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            tier="batch",
        )
        assert price is not None
        assert price.amount == Decimal("0.00001") / 1000


class TestNovaGroundingRequests:
    """Nova Grounding (built-in web-grounding tool) $/request pricing."""

    def test_nova_grounding_usagetype_maps_to_grounding_requests_dimension(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The "nova-grounding" usagetype marker must resolve to Dimension.GROUNDING_REQUESTS."""
        results: dict[PriceKey, Price] = {}
        item = json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "us-east-1",
                        "usagetype": "USE1-Nova2.0Lite-nova-grounding",
                        "model": "Nova 2.0 Lite",
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "Requests",
                                    "pricePerUnit": {"USD": "0.03"},
                                }
                            }
                        }
                    }
                },
            }
        )
        _ingest_price_list_item(item, Service.BEDROCK, "us-east-1", "USD", results)
        monkeypatch.setattr(pricing._state, "price_index", results)  # noqa: SLF001
        price = resolve_price(
            Service.BEDROCK,
            "amazon.nova-2-lite-v1:0",
            "us-east-1",
            Dimension.GROUNDING_REQUESTS,
        )
        assert price is not None
        assert price.amount == Decimal("0.03")


class TestReservedCapacityExclusion:
    """Reserved-capacity ("Reserved - N Month") rows are a commitment rate, not billed by this app."""

    def test_reserved_capacity_row_is_not_ingested(self) -> None:
        """A "Reserved - 1 Month" feature row must be skipped, even with no inferenceType."""
        results: dict[PriceKey, Price] = {}
        item = json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "eu-west-1",
                        "usagetype": (
                            "EU-Claude4.5Sonnet-reserved-1-month-input-tokens"
                            "-per-minute-cross-region-global"
                        ),
                        "feature": "Reserved - 1 Month",
                        "model": "Claude Sonnet 4.5",
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "1K tokens",
                                    "pricePerUnit": {"USD": "1.5"},
                                }
                            }
                        }
                    }
                },
            }
        )
        _ingest_price_list_item(item, Service.BEDROCK, "eu-west-1", "USD", results)
        assert results == {}


class TestUsagetypeTokenFallbackTierSuffixes:
    """Native rows with no inferenceType/featuretype at all (xai.grok's "mantle" usagetype schema).

    "mantle" rows are the bedrock-mantle API's rates: they must key under
    ``Service.BEDROCK_MANTLE``, never mixing with bedrock-runtime rates.
    """

    @staticmethod
    def _item(usagetype: str, model: str = "xai.grok-4.3", price: str = "0.002") -> str:
        return json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "us-east-1",
                        "usagetype": usagetype,
                        "model": model,
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "1K tokens",
                                    "pricePerUnit": {"USD": price},
                                }
                            }
                        }
                    }
                },
            }
        )

    @pytest.mark.parametrize(
        ("usagetype", "expected_tier"),
        [
            ("USE1-xai.grok-4.3-mantle-input-tokens-flex", "flex"),
            ("USE1-xai.grok-4.3-mantle-input-tokens-priority", "priority"),
            ("USE1-xai.grok-4.3-mantle-input-tokens-standard", "standard"),
            ("USE1-xai.grok-4.3-mantle-input-tokens-batch", "batch"),
        ],
    )
    def test_mantle_input_tokens_usagetype_maps_to_input_tokens_with_tier_suffix(
        self, usagetype: str, expected_tier: str
    ) -> None:
        """Each tier suffix must resolve to INPUT_TOKENS under its own tier."""
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item(usagetype), Service.BEDROCK, "us-east-1", "USD", results
        )
        key = PriceKey(
            Service.BEDROCK_MANTLE,
            normalize_model_key("xai.grok-4.3"),
            "us-east-1",
            Dimension.INPUT_TOKENS,
            expected_tier,
        )
        assert key in results

    def test_mantle_cache_read_tokens_usagetype_maps_to_cache_read_tokens(self) -> None:
        """ "cache-read-tokens" must resolve to CACHE_READ_TOKENS."""  # noqa: D210
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item("USE1-xai.grok-4.3-mantle-cache-read-tokens-standard"),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.BEDROCK_MANTLE,
            normalize_model_key("xai.grok-4.3"),
            "us-east-1",
            Dimension.CACHE_READ_TOKENS,
            "standard",
        )
        assert key in results

    def test_cache_read_pattern_takes_precedence_over_inputtoken_substring(
        self,
    ) -> None:
        """ "cache-read-input-token-count" contains "inputtoken" too -- cache patterns must win."""  # noqa: D210
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item(
                "USE1-xai.grok-4.3-mantle-cache-read-input-token-count-standard"
            ),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.BEDROCK_MANTLE,
            normalize_model_key("xai.grok-4.3"),
            "us-east-1",
            Dimension.CACHE_READ_TOKENS,
            "standard",
        )
        assert key in results
        assert not any(k.dimension == Dimension.INPUT_TOKENS for k in results)


class TestNovaSonicModality:
    """Nova Sonic (speech-to-speech) rows: text-modality tokens billed, speech-modality rows unmapped."""

    @staticmethod
    def _item(inference_type: str, usagetype: str, price: str = "0.0034") -> str:
        return json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "us-east-1",
                        "usagetype": usagetype,
                        "inferenceType": inference_type,
                        "model": "Nova Sonic",
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "1K tokens",
                                    "pricePerUnit": {"USD": price},
                                }
                            }
                        }
                    }
                },
            }
        )

    def test_text_input_token_maps_to_input_tokens(self) -> None:
        """ "Text Input Token" must resolve to INPUT_TOKENS."""  # noqa: D210
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item("Text Input Token", "USE1-NovaSonic-text-input-tokens"),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.BEDROCK,
            normalize_model_key("Nova Sonic"),
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        assert key in results

    def test_text_output_token_maps_to_output_tokens(self) -> None:
        """ "Text output token" must resolve to OUTPUT_TOKENS."""  # noqa: D210
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item("Text output token", "USE1-NovaSonic-text-output-tokens"),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.BEDROCK,
            normalize_model_key("Nova Sonic"),
            "us-east-1",
            Dimension.OUTPUT_TOKENS,
            "standard",
        )
        assert key in results

    def test_speech_understanding_tokens_ingest_with_speech_spec(self) -> None:
        """Speech-modality rows must index under the "speech" spec bucket."""
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item(
                "Speech Understanding input token", "USE1-NovaSonic-speech-input-tokens"
            ),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.BEDROCK,
            normalize_model_key("Nova Sonic"),
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "",
            "speech",
        )
        assert set(results) == {key}

    def test_unrecognized_inference_type_is_not_ingested(self) -> None:
        """An unrecognized non-empty inferenceType must not fall through to the usagetype fallback."""
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item("Some Future Modality token", "USE1-NovaSonic-input-tokens"),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
        )
        assert results == {}


class TestNativeLatencyOptimizedRouting:
    """Native "<Model> Latency Optimized" `model` attribute rows -- a distinct serving profile.

    Confirmed live: Nova Pro publishes a separate "Nova Pro Latency Optimized"
    price-list `model` value, pricier than the plain "Nova Pro" rows, that
    must key under the same normalized model ("novapro") with routing="latency".
    """

    @staticmethod
    def _item(model: str, usagetype: str, price: str) -> str:
        return json.dumps(
            {
                "product": {
                    "attributes": {
                        "regionCode": "us-east-1",
                        "usagetype": usagetype,
                        "inferenceType": "Input tokens",
                        "model": model,
                    }
                },
                "terms": {
                    "OnDemand": {
                        "SKU1": {
                            "priceDimensions": {
                                "SKU1.A": {
                                    "unit": "1K tokens",
                                    "pricePerUnit": {"USD": price},
                                }
                            }
                        }
                    }
                },
            }
        )

    def test_latency_optimized_model_attribute_ingests_under_the_base_key_with_routing_latency(
        self,
    ) -> None:
        """ "Nova Pro Latency Optimized" must key under "novapro" with routing="latency"."""  # noqa: D210
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item(
                "Nova Pro Latency Optimized", "USE1-NovaPro-input-tokens", "0.0012"
            ),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.BEDROCK,
            "novapro",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "latency",
        )
        assert key in results
        assert results[key].amount == Decimal("0.0012") / 1000

    def test_latency_optimized_and_plain_native_rows_do_not_collide(self) -> None:
        """The latency-optimized and plain "Nova Pro" rows must resolve to distinct keys."""
        results: dict[PriceKey, Price] = {}
        claims: dict[PriceKey, str] = {}
        diagnostics: list[str] = []
        _ingest_price_list_item(
            self._item(
                "Nova Pro Latency Optimized", "USE1-NovaPro-input-tokens", "0.0012"
            ),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
            claims,
            diagnostics,
        )
        _ingest_price_list_item(
            self._item("Nova Pro", "USE1-NovaPro-input-tokens", "0.0008"),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
            claims,
            diagnostics,
        )
        assert diagnostics == []
        latency_key = PriceKey(
            Service.BEDROCK,
            "novapro",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "latency",
        )
        plain_key = PriceKey(
            Service.BEDROCK, "novapro", "us-east-1", Dimension.INPUT_TOKENS, "standard"
        )
        assert results[latency_key].amount == Decimal("0.0012") / 1000
        assert results[plain_key].amount == Decimal("0.0008") / 1000


class TestNormalizeModelKey:
    """Matching Bedrock model IDs to AWS Price List `model` display names."""

    @pytest.mark.parametrize(
        ("model_id", "display_name"),
        [
            ("anthropic.claude-3-5-sonnet-20241022-v2:0", "Claude 3.5 Sonnet"),
            ("us.anthropic.claude-3-5-sonnet-20241022-v2:0", "Claude 3.5 Sonnet"),
            ("amazon.nova-lite-v1:0", "Nova Lite"),
            ("amazon.nova-pro-v1:0:24k", "Nova Pro"),
        ],
    )
    def test_model_id_matches_display_name(
        self, model_id: str, display_name: str
    ) -> None:
        """A decorated model ID must normalize to the same key as its display name.

        Covers dated snapshots, cross-region prefixes, and context-window
        suffixes -- a mismatch here means cost silently comes out as $0.
        """
        assert normalize_model_key(model_id) == normalize_model_key(display_name)

    def test_distinct_models_do_not_collide(self) -> None:
        """Different models/snapshots must not normalize to the same key."""
        assert normalize_model_key(
            "anthropic.claude-3-opus-20240229-v1:0"
        ) != normalize_model_key("anthropic.claude-3-sonnet-20240229-v1:0")


class TestRegisterModelKeyOverrides:
    """register_model_key_overrides() merges entries into the override registry."""

    @pytest.fixture(autouse=True)
    def _restore_overrides(self) -> Generator[None]:
        """Snapshot/restore `_MODEL_KEY_OVERRIDES` so this class's mutations don't leak."""
        original = dict(pricing._MODEL_KEY_OVERRIDES)  # noqa: SLF001
        yield
        pricing._MODEL_KEY_OVERRIDES.clear()  # noqa: SLF001
        pricing._MODEL_KEY_OVERRIDES.update(original)  # noqa: SLF001

    def test_registered_override_resolves_via_resolve_model_key(self) -> None:
        """A newly registered override must be used by resolve_model_key()."""
        pricing.register_model_key_overrides({"vendor.x-v1:0": "vendorx"})
        assert pricing.resolve_model_key("vendor.x-v1:0") == "vendorx"

    def test_unregistered_model_id_falls_back_to_normalize_model_key(self) -> None:
        """A model ID with no registered override must resolve via normalize_model_key()."""
        model_id = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert pricing.resolve_model_key(model_id) == normalize_model_key(model_id)


class TestModelsPackageRegistersPricingOverrides:
    """`stdapi.models` registers its `pricing_overrides.MODEL_KEY_OVERRIDES` at import time."""

    def test_nova_2_lite_resolves_via_the_registered_override(self) -> None:
        """ "amazon.nova-2-lite-v1:0" must resolve to "nova20lite" via the registered table."""  # noqa: D210
        assert pricing.resolve_model_key("amazon.nova-2-lite-v1:0") == "nova20lite"


class TestDefaultModelPrices:
    """Built-in pricing-page defaults: applied only to models with no published row."""

    def test_gap_model_resolves_at_the_pricing_page_rate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model absent from the catalog must price at its registered default."""
        index: dict[PriceKey, Price] = {}
        pricing._apply_default_prices(index)  # noqa: SLF001
        monkeypatch.setattr(pricing._state, "price_index", index)  # noqa: SLF001
        price = resolve_price(
            Service.BEDROCK,
            "stability.stable-image-inpaint-v1:0",
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
        )
        assert price is not None
        assert price.amount == Decimal("0.07")
        assert price.currency == "USD"

    def test_published_row_disables_defaults_for_that_model(self) -> None:
        """Any published row for a model must suppress all its default prices."""
        published_key = PriceKey(
            Service.BEDROCK,
            pricing.resolve_model_key("stability.stable-image-inpaint-v1:0"),
            "us-west-2",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        index = {published_key: Price(Decimal("0.001"), "USD")}
        pricing._apply_default_prices(index)  # noqa: SLF001
        inpaint_keys = {key for key in index if key.model == published_key.model}
        assert inpaint_keys == {published_key}
        # Other gap models still get their defaults.
        assert any(
            key.model == pricing.resolve_model_key("stability.stable-outpaint-v1:0")
            for key in index
        )

    def test_operator_override_still_wins_over_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cost_price_overrides must overwrite a built-in default price."""
        monkeypatch.setattr(
            SETTINGS,
            "cost_price_overrides",
            {"stability.stable-image-inpaint-v1:0": {"output_images": 0.05}},
        )
        index: dict[PriceKey, Price] = {}
        pricing._apply_default_prices(index)  # noqa: SLF001
        _apply_price_overrides(index, {"us-east-1"})
        monkeypatch.setattr(pricing._state, "price_index", index)  # noqa: SLF001
        price = resolve_price(
            Service.BEDROCK,
            "stability.stable-image-inpaint-v1:0",
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
        )
        assert price is not None
        assert price.amount == Decimal("0.05")


class TestParseUnitScale:
    """Scale-multiplier parsing from AWS Price List `unit` strings."""

    @pytest.mark.parametrize(
        ("unit", "expected_scale"),
        [
            ("1K tokens", 1000),
            ("1M Characters", 1_000_000),
            ("Tokens", 1),
            # Regression: the digit run was captured but never used, silently
            # assuming any K/M-prefixed unit meant exactly 1000x/1_000_000x.
            ("10K tokens", 10_000),
            ("5M characters", 5_000_000),
        ],
    )
    def test_scale_multiplier_uses_the_actual_leading_digit(
        self, unit: str, expected_scale: int
    ) -> None:
        """The leading digit run must multiply the K/M scale, not be discarded."""
        assert parse_unit_scale(unit) == expected_scale


class TestNormalizeUsagetypeModel:
    """Model-key extraction from usagetype text for `model`-less price-list rows."""

    def test_strips_the_mp_marker_and_its_duplicated_region_code(self) -> None:
        """Regression: "MP:<REGION>_..." usagetypes must strip both marker and region code."""
        assert (
            normalize_usagetype_model(
                "USE1-MP:USE1_created_image_stable_image_core-Units"
            )
            == "createdimagestableimagecore"
        )

    def test_plain_usagetype_without_mp_marker_is_unaffected(self) -> None:
        """A normal (non-Marketplace) usagetype must still extract the model token as before."""
        assert normalize_usagetype_model("USE1-NovaLite-input-tokens") == "novalite"


class TestApplyPriceOverrides:
    """Operator-supplied COST_PRICE_OVERRIDES validation."""

    def test_valid_override_is_applied_per_region_currency(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A valid override must resolve currency from each region's own partition."""
        monkeypatch.setattr(
            SETTINGS,
            "cost_price_overrides",
            {"anthropic.claude-3-5-sonnet-20241022": {"input_tokens": 0.000003}},
        )
        index: dict[PriceKey, Price] = {}
        _apply_price_overrides(index, {"us-east-1", "eusc-de-east-1"})
        us_key = PriceKey(
            Service.BEDROCK,
            normalize_model_key("anthropic.claude-3-5-sonnet-20241022"),
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        eusc_key = PriceKey(
            Service.BEDROCK,
            normalize_model_key("anthropic.claude-3-5-sonnet-20241022"),
            "eusc-de-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        assert index[us_key].currency == "USD"
        assert index[eusc_key].currency == "EUR"

    @pytest.mark.parametrize("bad_price", [float("nan"), float("inf"), -1.0, 0.0])
    def test_non_finite_or_non_positive_price_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, bad_price: float
    ) -> None:
        """Regression: NaN/Infinity must never enter the index (crashes compute_costs)."""
        monkeypatch.setattr(
            SETTINGS,
            "cost_price_overrides",
            {"some-model": {"input_tokens": bad_price}},
        )
        index: dict[PriceKey, Price] = {}
        _apply_price_overrides(index, {"us-east-1"})
        assert index == {}

    def test_unknown_dimension_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unknown dimension name must be skipped, not raise."""
        monkeypatch.setattr(
            SETTINGS,
            "cost_price_overrides",
            {"some-model": {"not_a_real_dimension": 1.0}},
        )
        index: dict[PriceKey, Price] = {}
        _apply_price_overrides(index, {"us-east-1"})
        assert index == {}

    def test_override_honors_registered_model_key_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: an override for a key-overridden model must be resolvable."""
        model_id = "override.priced-model-v1:0"
        pricing.register_model_key_overrides({model_id: "customkey"})
        try:
            monkeypatch.setattr(
                SETTINGS, "cost_price_overrides", {model_id: {"input_tokens": 0.5}}
            )
            index: dict[PriceKey, Price] = {}
            _apply_price_overrides(index, {"us-east-1"})
            monkeypatch.setattr(pricing._state, "price_index", index)  # noqa: SLF001
            price = resolve_price(
                Service.BEDROCK, model_id, "us-east-1", Dimension.INPUT_TOKENS
            )
            assert price is not None
            assert price.amount == Decimal("0.5")
        finally:
            pricing._MODEL_KEY_OVERRIDES.pop(model_id, None)  # noqa: SLF001


class TestApplyRegionalFallback:
    """Near-region-first backfill for models AWS hasn't priced in every region."""

    def test_same_geography_region_is_preferred(self) -> None:
        """eu-west-3 must fall back to another eu-* region before us-east-1."""
        key = PriceKey(
            Service.BEDROCK, "modely", "us-east-1", Dimension.INPUT_TOKENS, "standard"
        )
        eu_key = PriceKey(
            Service.BEDROCK, "modely", "eu-west-1", Dimension.INPUT_TOKENS, "standard"
        )
        index = {
            key: Price(Decimal("0.001"), "USD"),
            eu_key: Price(Decimal("0.0011"), "USD"),
        }
        _apply_regional_fallback(index, {"us-east-1", "eu-west-1", "eu-west-3"})
        fallback_key = PriceKey(
            Service.BEDROCK, "modely", "eu-west-3", Dimension.INPUT_TOKENS, "standard"
        )
        assert index[fallback_key] == index[eu_key]

    def test_falls_back_to_anchor_region_when_no_geography_match(self) -> None:
        """No ap-* region priced at all must still fall back to the us-east-1 anchor."""
        us_key = PriceKey(
            Service.BEDROCK, "modelz", "us-east-1", Dimension.INPUT_TOKENS, "standard"
        )
        index = {us_key: Price(Decimal("0.002"), "USD")}
        _apply_regional_fallback(index, {"us-east-1", "ap-southeast-2"})
        fallback_key = PriceKey(
            Service.BEDROCK,
            "modelz",
            "ap-southeast-2",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        assert index[fallback_key] == index[us_key]

    def test_never_backfills_across_a_partition_boundary(self) -> None:
        """Regression: must not backfill USD prices into EUR-partition regions."""
        us_key = PriceKey(
            Service.BEDROCK, "modelp", "us-east-1", Dimension.INPUT_TOKENS, "standard"
        )
        index = {us_key: Price(Decimal("0.001"), "USD")}
        _apply_regional_fallback(index, {"us-east-1", "eusc-de-east-1"})
        fallback_key = PriceKey(
            Service.BEDROCK,
            "modelp",
            "eusc-de-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        assert fallback_key not in index

    def test_backfills_within_the_same_non_standard_partition(self) -> None:
        """A second eusc-* region's price must still be used for a eusc-* region missing one."""
        us_key = PriceKey(
            Service.BEDROCK, "modelq", "us-east-1", Dimension.INPUT_TOKENS, "standard"
        )
        eusc_key = PriceKey(
            Service.BEDROCK,
            "modelq",
            "eusc-de-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        index = {
            us_key: Price(Decimal("0.001"), "USD"),
            eusc_key: Price(Decimal("0.0012"), "EUR"),
        }
        _apply_regional_fallback(
            index, {"us-east-1", "eusc-de-east-1", "eusc-fr-east-1"}
        )
        fallback_key = PriceKey(
            Service.BEDROCK,
            "modelq",
            "eusc-fr-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        assert index[fallback_key] == index[eusc_key]

    def test_existing_region_price_is_never_overwritten(self) -> None:
        """A region that already has its own price must keep it, not get backfilled over."""
        us_key = PriceKey(
            Service.BEDROCK, "modelw", "us-east-1", Dimension.INPUT_TOKENS, "standard"
        )
        eu_key = PriceKey(
            Service.BEDROCK, "modelw", "eu-west-1", Dimension.INPUT_TOKENS, "standard"
        )
        index = {
            us_key: Price(Decimal("0.001"), "USD"),
            eu_key: Price(Decimal("0.0099"), "USD"),
        }
        _apply_regional_fallback(index, {"us-east-1", "eu-west-1"})
        assert index[eu_key].amount == Decimal("0.0099")

    @pytest.mark.parametrize(
        ("region", "family"),
        [
            ("eu-west-3", "eu"),
            ("us-east-1", "us"),
            ("ap-southeast-2", "ap"),
            ("us-gov-west-1", "us"),
            ("cn-north-1", "cn"),
            ("eusc-de-east-1", "eusc"),
        ],
    )
    def test_region_family_extracts_geo_prefix(self, region: str, family: str) -> None:
        """Geo prefix is everything before the first hyphen."""
        assert _region_family(region) == family

    @pytest.mark.parametrize(
        ("bedrock_region", "endpoint"),
        [
            ("us-east-2", "us-east-1"),
            ("eu-west-3", "eu-central-1"),
            ("ap-northeast-1", "ap-south-1"),
            # EUSC has its own in-partition endpoint: the only source of its
            # rows, and commercial endpoints are cross-partition anyway.
            ("eusc-de-east-1", "eusc-de-east-1"),
            ("cn-northwest-1", "cn-north-1"),
            # GovCloud has no Price List endpoint at all: pricing is skipped
            # instead of crash-looping startup on an unreachable call.
            ("us-gov-west-1", None),
            ("xx-unknown-1", "us-east-1"),
        ],
    )
    def test_pricing_endpoint_follows_first_bedrock_region(
        self, monkeypatch: pytest.MonkeyPatch, bedrock_region: str, endpoint: str | None
    ) -> None:
        """The endpoint matches the first configured Bedrock region's geography."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", [bedrock_region])
        assert pricing.pricing_endpoint_region() == endpoint

    async def test_catalog_load_skips_partitions_without_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GovCloud: the load no-ops with a diagnostic instead of calling AWS."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-gov-west-1"])

        def _no_client(*_args: object, **_kwargs: object) -> object:
            message = "must not create a pricing client"
            raise AssertionError(message)

        monkeypatch.setattr(pricing, "get_client", _no_client)
        diagnostics: list[str] = []
        await pricing._load_price_catalog(diagnostics)  # noqa: SLF001
        assert pricing._state.price_index == {}  # noqa: SLF001
        assert any("no endpoint" in d for d in diagnostics)

    async def test_overrides_still_apply_without_a_pricing_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GovCloud: operator cost_price_overrides remain the sole price source."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-gov-west-1"])
        monkeypatch.setattr(
            SETTINGS,
            "cost_price_overrides",
            {"anthropic.claude-gov-model-v1:0": {"input_tokens": 0.000003}},
        )
        await pricing._load_price_catalog([])  # noqa: SLF001
        price = resolve_price(
            Service.BEDROCK,
            "anthropic.claude-gov-model-v1:0",
            "us-gov-west-1",
            Dimension.INPUT_TOKENS,
        )
        assert price is not None
        assert price.amount == Decimal("0.000003")
        assert price.currency == "USD"

    def test_plain_and_global_routing_backfill_independently(self) -> None:
        """Regression: plain and global routing must backfill independently."""
        plain_key = PriceKey(
            Service.BEDROCK,
            "routedmodel",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        global_key = PriceKey(
            Service.BEDROCK,
            "routedmodel",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "global",
        )
        index = {
            plain_key: Price(Decimal("0.0033"), "USD"),
            global_key: Price(Decimal("0.0030"), "USD"),
        }
        _apply_regional_fallback(index, {"us-east-1", "eu-west-1"})
        plain_fallback = PriceKey(
            Service.BEDROCK,
            "routedmodel",
            "eu-west-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        global_fallback = PriceKey(
            Service.BEDROCK,
            "routedmodel",
            "eu-west-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "global",
        )
        assert index[plain_fallback].amount == Decimal("0.0033")
        assert index[global_fallback].amount == Decimal("0.0030")

    def test_long_and_standard_context_backfill_independently(self) -> None:
        """Regression: a long-context price must backfill another region's long-context key, not its standard one."""
        standard_key = PriceKey(
            Service.BEDROCK, "ctxmodel", "us-east-1", Dimension.INPUT_TOKENS, "standard"
        )
        long_key = PriceKey(
            Service.BEDROCK,
            "ctxmodel",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "",
            "",
            "long",
        )
        index = {
            standard_key: Price(Decimal("0.003"), "USD"),
            long_key: Price(Decimal("0.006"), "USD"),
        }
        _apply_regional_fallback(index, {"us-east-1", "eu-west-1"})
        standard_fallback = PriceKey(
            Service.BEDROCK, "ctxmodel", "eu-west-1", Dimension.INPUT_TOKENS, "standard"
        )
        long_fallback = PriceKey(
            Service.BEDROCK,
            "ctxmodel",
            "eu-west-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "",
            "",
            "long",
        )
        assert index[standard_fallback].amount == Decimal("0.003")
        assert index[long_fallback].amount == Decimal("0.006")


class TestResolvePrice:
    """Tier-aware price resolution."""

    def test_missing_tier_falls_back_to_standard_scaled_by_aws_ratio(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tier miss falls back to standard * AWS tier ratio (flex=50%, priority=175%)."""
        standard_key = PriceKey(
            Service.BEDROCK, "modelx", "us-east-1", Dimension.INPUT_TOKENS, "standard"
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            standard_key,
            Price(Decimal("0.001"), "USD"),
        )
        flex_price = resolve_price(
            Service.BEDROCK, "modelx", "us-east-1", Dimension.INPUT_TOKENS, tier="flex"
        )
        assert flex_price is not None
        assert flex_price.amount == Decimal("0.0005")

        priority_price = resolve_price(
            Service.BEDROCK,
            "modelx",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            tier="priority",
        )
        assert priority_price is not None
        assert priority_price.amount == Decimal("0.00175")

    def test_tier_ratio_not_applied_to_per_request_dimensions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-request fees (grounding, search units, images) are tier-flat."""
        standard_key = PriceKey(
            Service.BEDROCK,
            "modelx",
            "us-east-1",
            Dimension.GROUNDING_REQUESTS,
            "standard",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            standard_key,
            Price(Decimal("0.035"), "USD"),
        )
        flex_price = resolve_price(
            Service.BEDROCK,
            "modelx",
            "us-east-1",
            Dimension.GROUNDING_REQUESTS,
            tier="flex",
        )
        assert flex_price is not None
        assert flex_price.amount == Decimal("0.035")

    def test_long_context_premium_beats_exact_tier_at_standard_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A flex long-context call without a flex long row bills standard-long * ratio."""
        flex_short_key = PriceKey(
            Service.BEDROCK, "modelx", "us-east-1", Dimension.INPUT_TOKENS, "flex"
        )
        standard_long_key = PriceKey(
            Service.BEDROCK,
            "modelx",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "",
            "",
            "long",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            flex_short_key,
            Price(Decimal("0.001"), "USD"),
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            standard_long_key,
            Price(Decimal("0.004"), "USD"),
        )
        long_flex_price = resolve_price(
            Service.BEDROCK,
            "modelx",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            tier="flex",
            context="long",
        )
        assert long_flex_price is not None
        assert long_flex_price.amount == Decimal("0.002")

    def test_tier_ratio_fallback_preserves_routing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: tier fallback must preserve routing axis."""
        plain_key = PriceKey(
            Service.BEDROCK, "modely", "us-east-1", Dimension.INPUT_TOKENS, "standard"
        )
        global_key = PriceKey(
            Service.BEDROCK,
            "modely",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "global",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            plain_key,
            Price(Decimal("0.0033"), "USD"),
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            global_key,
            Price(Decimal("0.003"), "USD"),
        )

        flex_price = resolve_price(
            Service.BEDROCK,
            "modely",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            tier="flex",
            routing="global",
        )
        assert flex_price is not None
        assert flex_price.amount == Decimal("0.0015")  # 0.003 * 0.5, not 0.0033 * 0.5

    def test_tier_ratio_fallback_preserves_cache_ttl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: tier fallback must use the same cache_ttl bucket."""
        flat_key = PriceKey(
            Service.BEDROCK,
            "modelz",
            "us-east-1",
            Dimension.CACHE_WRITE_TOKENS,
            "standard",
        )
        one_hour_key = PriceKey(
            Service.BEDROCK,
            "modelz",
            "us-east-1",
            Dimension.CACHE_WRITE_TOKENS,
            "standard",
            "1h",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            flat_key,
            Price(Decimal("0.002"), "USD"),
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            one_hour_key,
            Price(Decimal("0.006"), "USD"),
        )

        flex_price = resolve_price(
            Service.BEDROCK,
            "modelz",
            "us-east-1",
            Dimension.CACHE_WRITE_TOKENS,
            tier="flex",
            cache_ttl="1h",
        )
        assert flex_price is not None
        # 0.006 * 0.5 -- distinct from both the flat price (0.002) and 0.002 * 0.5.
        assert flex_price.amount == Decimal("0.003")

    def test_reserved_tier_falls_back_to_unscaled_standard_price(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reserved tier (no AWS ratio) defaults to 1x standard rate."""
        standard_key = PriceKey(
            Service.BEDROCK, "modelw", "us-east-1", Dimension.INPUT_TOKENS, "standard"
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            standard_key,
            Price(Decimal("0.001"), "USD"),
        )

        reserved_price = resolve_price(
            Service.BEDROCK,
            "modelw",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            tier="reserved",
        )
        assert reserved_price is not None
        assert reserved_price.amount == Decimal("0.001")

    def test_model_key_override_matches_a_known_naming_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model in _MODEL_KEY_OVERRIDES must resolve via its overridden key."""
        model_id, override_key = next(iter(pricing._MODEL_KEY_OVERRIDES.items()))  # noqa: SLF001
        key = PriceKey(
            Service.BEDROCK,
            override_key,
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            key,
            Price(Decimal("0.002"), "USD"),
        )
        price = resolve_price(
            Service.BEDROCK, model_id, "us-east-1", Dimension.INPUT_TOKENS
        )
        assert price is not None
        assert price.amount == Decimal("0.002")

    def test_missing_model_returns_none(self) -> None:
        """No matching entry at all must return None, not raise."""
        assert (
            resolve_price(
                Service.BEDROCK, "no-such-model", "us-east-1", Dimension.INPUT_TOKENS
            )
            is None
        )

    def test_global_routing_resolves_a_distinctly_priced_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model with distinct plain/global prices must resolve each independently."""
        plain_key = PriceKey(
            Service.BEDROCK,
            "modelroute",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        global_key = PriceKey(
            Service.BEDROCK,
            "modelroute",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "global",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            plain_key,
            Price(Decimal("0.0033"), "USD"),
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            global_key,
            Price(Decimal("0.003"), "USD"),
        )
        plain_price = resolve_price(
            Service.BEDROCK, "modelroute", "us-east-1", Dimension.INPUT_TOKENS
        )
        global_price = resolve_price(
            Service.BEDROCK,
            "modelroute",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            routing="global",
        )
        assert plain_price is not None
        assert global_price is not None
        assert plain_price.amount == Decimal("0.0033")
        assert global_price.amount == Decimal("0.003")

    def test_global_routing_falls_back_to_plain_when_not_indexed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Requesting global routing for a model with only a plain price must not miss."""
        plain_key = PriceKey(
            Service.BEDROCK,
            "modelnoroute",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            plain_key,
            Price(Decimal("0.001"), "USD"),
        )
        price = resolve_price(
            Service.BEDROCK,
            "modelnoroute",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            routing="global",
        )
        assert price is not None
        assert price.amount == Decimal("0.001")

    def test_latency_routing_resolves_a_distinctly_priced_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model with distinct plain/latency prices must resolve each independently."""
        plain_key = PriceKey(
            Service.BEDROCK,
            "modellatency",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        latency_key = PriceKey(
            Service.BEDROCK,
            "modellatency",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "latency",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            plain_key,
            Price(Decimal("0.003"), "USD"),
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            latency_key,
            Price(Decimal("0.0036"), "USD"),
        )
        plain_price = resolve_price(
            Service.BEDROCK, "modellatency", "us-east-1", Dimension.INPUT_TOKENS
        )
        latency_price = resolve_price(
            Service.BEDROCK,
            "modellatency",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            routing="latency",
        )
        assert plain_price is not None
        assert latency_price is not None
        assert plain_price.amount == Decimal("0.003")
        assert latency_price.amount == Decimal("0.0036")

    def test_latency_routing_falls_back_to_plain_when_not_indexed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Requesting latency routing for a model with only a plain price must not miss."""
        plain_key = PriceKey(
            Service.BEDROCK,
            "modelnolatency",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            plain_key,
            Price(Decimal("0.001"), "USD"),
        )
        price = resolve_price(
            Service.BEDROCK,
            "modelnolatency",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            routing="latency",
        )
        assert price is not None
        assert price.amount == Decimal("0.001")

    def test_image_spec_resolves_a_distinctly_priced_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Different resolution/quality buckets for one model must resolve independently."""
        small_key = PriceKey(
            Service.BEDROCK,
            "imagemodel",
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
            "standard",
            "",
            "",
            "512:standard",
        )
        large_key = PriceKey(
            Service.BEDROCK,
            "imagemodel",
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
            "standard",
            "",
            "",
            "1024:premium",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            small_key,
            Price(Decimal("0.008"), "USD"),
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            large_key,
            Price(Decimal("0.012"), "USD"),
        )
        small_price = resolve_price(
            Service.BEDROCK,
            "imagemodel",
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
            spec="512:standard",
        )
        large_price = resolve_price(
            Service.BEDROCK,
            "imagemodel",
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
            spec="1024:premium",
        )
        assert small_price is not None
        assert large_price is not None
        assert small_price.amount == Decimal("0.008")
        assert large_price.amount == Decimal("0.012")

    def test_image_spec_falls_back_to_flat_price_when_not_indexed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model with only a flat per-image price must still resolve for any spec."""
        flat_key = PriceKey(
            Service.BEDROCK,
            "flatimagemodel",
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
            "standard",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            flat_key,
            Price(Decimal("0.0036"), "USD"),
        )
        price = resolve_price(
            Service.BEDROCK,
            "flatimagemodel",
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
            spec="1024:standard",
        )
        assert price is not None
        assert price.amount == Decimal("0.0036")


class TestResolvePriceContext:
    """Long-context resolve_price behavior: exact match, fallback, and tier-ratio interaction."""

    def test_long_context_price_is_returned_when_indexed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """context="long" must resolve the long-context entry, not the standard one."""
        standard_key = PriceKey(
            Service.BEDROCK,
            "modellong",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        long_key = PriceKey(
            Service.BEDROCK,
            "modellong",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "",
            "",
            "long",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            standard_key,
            Price(Decimal("0.003"), "USD"),
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            long_key,
            Price(Decimal("0.006"), "USD"),
        )
        price = resolve_price(
            Service.BEDROCK,
            "modellong",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            context="long",
        )
        assert price is not None
        assert price.amount == Decimal("0.006")

    def test_falls_back_to_standard_context_when_no_long_price_indexed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model with only a standard-context price must still resolve for context="long"."""
        standard_key = PriceKey(
            Service.BEDROCK,
            "modelnolong",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            standard_key,
            Price(Decimal("0.003"), "USD"),
        )
        price = resolve_price(
            Service.BEDROCK,
            "modelnolong",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            context="long",
        )
        assert price is not None
        assert price.amount == Decimal("0.003")

    def test_tier_ratio_fallback_respects_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: tier fallback must scale the long-context standard price, not the plain one."""
        plain_key = PriceKey(
            Service.BEDROCK,
            "modelctxtier",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        long_key = PriceKey(
            Service.BEDROCK,
            "modelctxtier",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "",
            "",
            "long",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            plain_key,
            Price(Decimal("0.003"), "USD"),
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            long_key,
            Price(Decimal("0.006"), "USD"),
        )
        flex_price = resolve_price(
            Service.BEDROCK,
            "modelctxtier",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            tier="flex",
            context="long",
        )
        assert flex_price is not None
        # 0.006 * 0.5 -- distinct from the plain price's 0.003 * 0.5 = 0.0015.
        assert flex_price.amount == Decimal("0.003")


class TestRefreshPriceCatalogForNewModels:
    """On-demand price-catalog refresh triggered by newly discovered Bedrock models."""

    def test_is_model_priced_reflects_current_index(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """is_model_priced() must check the model's normalized key against the live index."""
        assert is_model_priced("amazon.brand-new-model-v1:0") is False
        key = PriceKey(
            Service.BEDROCK,
            "brandnewmodel",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            key,
            Price(Decimal("0.001"), "USD"),
        )
        assert is_model_priced("amazon.brand-new-model-v1:0") is True

    def test_is_model_priced_uses_model_key_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model listed in _MODEL_KEY_OVERRIDES must be checked under its override key."""
        override_model_id, override_key = next(
            iter(pricing._MODEL_KEY_OVERRIDES.items())  # noqa: SLF001
        )
        key = PriceKey(
            Service.BEDROCK,
            override_key,
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            key,
            Price(Decimal("0.001"), "USD"),
        )
        assert is_model_priced(override_model_id) is True

    async def test_no_reload_when_cost_tracking_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Must never reload the catalog when cost tracking is disabled."""
        monkeypatch.setattr(SETTINGS, "cost_tracking", False)

        async def _fail(_diagnostics: list[str]) -> None:
            pytest.fail(
                "_load_price_catalog must not be called when cost_tracking is disabled"
            )

        monkeypatch.setattr(pricing, "_load_price_catalog", _fail)
        await refresh_price_catalog_for_new_models(["amazon.some-model-v1:0"])

    async def test_no_reload_when_every_model_already_priced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Must never reload the catalog when every given model already has a price."""
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        key = PriceKey(
            Service.BEDROCK,
            "alreadypricedmodel",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            key,
            Price(Decimal("0.001"), "USD"),
        )

        async def _fail(_diagnostics: list[str]) -> None:
            pytest.fail(
                "_load_price_catalog must not be called when all models are priced"
            )

        monkeypatch.setattr(pricing, "_load_price_catalog", _fail)
        await refresh_price_catalog_for_new_models(["amazon.already-priced-model-v1:0"])

    async def test_reload_triggered_for_an_unpriced_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A newly discovered, unpriced model must trigger exactly one immediate reload."""
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        calls = 0

        async def _fake_load(_diagnostics: list[str]) -> None:
            nonlocal calls
            calls += 1

        monkeypatch.setattr(pricing, "_load_price_catalog", _fake_load)
        await refresh_price_catalog_for_new_models(["amazon.unreleased-model-v1:0"])
        assert calls == 1

    async def test_no_reload_when_no_model_ids_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty model_ids iterable must never trigger a reload."""

        async def _fail(_diagnostics: list[str]) -> None:
            pytest.fail("_load_price_catalog must not be called with no model IDs")

        monkeypatch.setattr(pricing, "_load_price_catalog", _fail)
        await refresh_price_catalog_for_new_models([])


@pytest.mark.expensive
async def test_bedrock_model_pricing_coverage() -> None:
    """Every registered + deprecated-but-still-live Bedrock model must have pricing.

    Talks to the real Bedrock and Pricing APIs (skips cleanly if AWS isn't
    reachable) -- fixtures can't catch this because the whole point is to
    catch models AWS actually serves *today* that our model-key matching
    doesn't handle yet. Coverage is checked per (model, region) pair: for
    currently-registered models, using each model's own resolved
    ``.regions`` (a model is not necessarily available in every configured
    region); for ``DEPRECATED_MODELS`` entries not currently registered
    (this account may have lost access, but others may not have), across
    every configured ``aws_bedrock_regions`` instead, since we have no
    per-region availability info for them anymore.

    On failure, prints two lists (see ``stdapi/pricing.py``'s
    ``_MODEL_KEY_OVERRIDES`` docstring for the exact steps this maps to):

    1. "UNPRICED MODELS": registered model IDs (with the regions they're
       missing pricing in) that resolve to no price for any billed
       dimension/tier.
    2. "CANDIDATE MATCHES": distinct AWS Price List `model` attribute values
       that no registered model (after applying ``_MODEL_KEY_OVERRIDES``)
       claims, shown as ``"AWS display name" -> 'normalized_key'``. Pair an
       unpriced model ID with the normalized key of its obvious match and
       add that one line to ``_MODEL_KEY_OVERRIDES``.

    This only covers models with a real Price List `model` attribute. A few
    (Titan, some marketplace/Foundation-Models rows) have none at all and
    are keyed from `usagetype` text instead (see
    ``normalize_usagetype_model``) -- those aren't surfaced here since the
    signal is too noisy to present usefully; if one of those shows up in
    UNPRICED MODELS, it needs manual investigation, not just a quick pairing.
    """
    try:
        async with AWSConnectionManager(("sts", None)):
            pass
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"AWS is not reachable: {exc}")

    async with AWSConnectionManager(
        *(("bedrock", region) for region in SETTINGS.aws_bedrock_regions),
        *(("bedrock-runtime", region) for region in SETTINGS.aws_bedrock_regions),
        # type-ignore: the RegionName stub Literal lags EUSC/China (works live).
        ("pricing", pricing.pricing_endpoint_region()),  # type: ignore[arg-type]
    ):
        await models.initialize_bedrock_models()
        await pricing.start_price_catalog()
        registered = dict(models._MODELS)  # noqa: SLF001
        configured_regions = [str(r) for r in SETTINGS.aws_bedrock_regions]

        # Some dimensions are only priced per spec bucket (image resolution/
        # quality, media modality) -- a spec-less lookup alone would wrongly
        # report those models as unpriced.
        spec_probes: dict[Dimension, tuple[str, ...]] = {
            Dimension.OUTPUT_IMAGES: (
                "",
                "512:standard",
                "512:premium",
                "1024:standard",
                "1024:premium",
            ),
            Dimension.INPUT_IMAGES: ("", "document"),
            Dimension.INPUT_SECONDS: ("", "audio", "video"),
            Dimension.OUTPUT_SECONDS: ("", "hd"),
        }

        def _missing_regions(model_id: str, check_regions: list[str]) -> list[str]:
            return [
                region
                for region in check_regions
                if not any(
                    resolve_price(
                        Service.BEDROCK, model_id, region, dim, tier, "", "", spec
                    )
                    for dim in Dimension
                    for tier in ("standard", "flex", "priority", "batch")
                    for spec in spec_probes.get(dim, ("",))
                )
            ]

        unpriced: dict[str, list[str]] = {}
        for model_id, details in registered.items():
            if model_id in _KNOWN_PRICING_GAPS:
                continue
            if missing := _missing_regions(model_id, [str(r) for r in details.regions]):
                unpriced[model_id] = missing
        for model_id in DEPRECATED_MODELS:
            if model_id in registered or model_id in _KNOWN_PRICING_GAPS:
                continue  # already checked above with precise per-model regions
            if missing := _missing_regions(model_id, configured_regions):
                unpriced[model_id] = missing

        candidates = (
            await _unclaimed_bedrock_price_list_models(
                {*registered, *DEPRECATED_MODELS}
            )
            if unpriced
            else []
        )

    if not unpriced:
        return

    lines = ["Bedrock pricing coverage gap found.", ""]
    lines.append(f"UNPRICED MODELS ({len(unpriced)}):")
    lines.extend(f"  {mid}: missing in {regions}" for mid, regions in unpriced.items())
    lines.append("")
    if candidates:
        lines.append(f"CANDIDATE MATCHES ({len(candidates)}):")
        lines.extend(f"  {name!r} -> {key!r}" for name, key in candidates)
        lines.append("")
    lines.append(
        "FIX (agent-actionable): for each unpriced model with an obvious "
        "candidate match, add one line to MODEL_KEY_OVERRIDES in "
        "stdapi/models/pricing_overrides.py:\n"
        '    "<unpriced-model-id>": "<candidate-normalized-key>",\n'
        "then re-run this test (see that module's docstring). If no candidate "
        "matches, AWS hasn't published pricing yet (or withdrew it): either "
        "wait, or -- once confirmed upstream against live Price List data -- "
        "add the model ID to _KNOWN_PRICING_GAPS in this file. Never remove a "
        "model's implementation because its pricing disappeared: keep it in "
        "case pricing returns or users retain model access."
    )
    pytest.fail("\n".join(lines))


#: Live-confirmed upstream pricing gaps (2026-07), excluded from the coverage check.
# SDXL: AWS publishes no price rows at all, and its pricing page only says
# legacy Stability models are priced per step count and resolution, with no
# figures (the Image Services page rates ship as DEFAULT_MODEL_PRICES). Re-check
# by removing the entry. Never remove a model's implementation for a pricing
# gap: keep it in case pricing returns or users retain model access.
_KNOWN_PRICING_GAPS: Final[frozenset[str]] = frozenset(
    {"stability.stable-diffusion-xl-v1"}
)


async def _unclaimed_bedrock_price_list_models(
    known_model_ids: set[str],
) -> list[tuple[str, str]]:
    """Fetch live Bedrock Price List ``model`` names unclaimed by any known model ID.

    Args:
        known_model_ids: Registered and/or deprecated Bedrock model IDs to
            treat as already claimed.

    Returns:
        Sorted ``(price_list_model_name, normalized_key)`` pairs for every
        distinct `model` attribute value in the live catalog whose normalized
        key doesn't match any known model ID (after applying
        ``pricing._MODEL_KEY_OVERRIDES``).
    """
    claimed_keys = {
        pricing._MODEL_KEY_OVERRIDES.get(model_id) or normalize_model_key(model_id)  # noqa: SLF001
        for model_id in known_model_ids
    }
    model_names: set[str] = set()
    pricing_region = pricing.pricing_endpoint_region()
    # type-ignore: the RegionName stub Literal lags EUSC/China (works live).
    client = get_client("pricing", pricing_region)  # type: ignore[arg-type]
    for service_code in pricing._SERVICE_CODE_TO_SERVICE:  # noqa: SLF001
        if pricing._SERVICE_CODE_TO_SERVICE[service_code] != Service.BEDROCK:  # noqa: SLF001
            continue
        paginator = client.get_paginator("get_products")
        async for page in paginator.paginate(
            ServiceCode=service_code,
            Filters=[
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": pricing_region}
            ],
        ):
            for raw in page.get("PriceList", []):
                model_name = json.loads(raw)["product"]["attributes"].get("model")
                if model_name:
                    model_names.add(model_name)

    return sorted(
        (name, key)
        for name in model_names
        if (key := normalize_model_key(name)) not in claimed_keys
    )


class TestModelPrices:
    """model_prices(): filtered, sorted read of one model's price rows."""

    @staticmethod
    def _seed(monkeypatch: pytest.MonkeyPatch) -> dict[PriceKey, Price]:
        """Seed a small multi-axis card for one model plus a decoy model."""
        rows = {
            PriceKey(
                Service.BEDROCK,
                "cardmodel",
                "us-east-1",
                Dimension.INPUT_TOKENS,
                "standard",
            ): Price(Decimal("0.000003"), "USD"),
            PriceKey(
                Service.BEDROCK,
                "cardmodel",
                "us-east-1",
                Dimension.INPUT_TOKENS,
                "flex",
            ): Price(Decimal("0.0000015"), "USD"),
            PriceKey(
                Service.BEDROCK,
                "cardmodel",
                "us-east-1",
                Dimension.INPUT_TOKENS,
                "standard",
                "",
                "",
                "",
                "long",
            ): Price(Decimal("0.000006"), "USD"),
            PriceKey(
                Service.BEDROCK,
                "cardmodel",
                "us-east-1",
                Dimension.CACHE_WRITE_TOKENS,
                "standard",
                "5m",
            ): Price(Decimal("0.00000375"), "USD"),
            PriceKey(
                Service.BEDROCK,
                "cardmodel",
                "us-east-1",
                Dimension.OUTPUT_TOKENS,
                "standard",
                "",
                "global",
            ): Price(Decimal("0.000015"), "USD"),
            PriceKey(
                Service.BEDROCK,
                "cardmodel",
                "eu-west-1",
                Dimension.INPUT_TOKENS,
                "standard",
            ): Price(Decimal("0.0000033"), "USD"),
            PriceKey(
                Service.BEDROCK,
                "cardmodel",
                "us-east-1",
                Dimension.OUTPUT_IMAGES,
                "standard",
                "",
                "",
                "1024:standard",
            ): Price(Decimal("0.04"), "USD"),
            PriceKey(
                Service.BEDROCK,
                "othermodel",
                "us-east-1",
                Dimension.INPUT_TOKENS,
                "standard",
            ): Price(Decimal("0.5"), "USD"),
        }
        for key, price in rows.items():
            monkeypatch.setitem(pricing._state.price_index, key, price)  # noqa: SLF001
        return rows

    def test_returns_only_the_requested_model_sorted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the model's rows are returned, region-major sorted."""
        self._seed(monkeypatch)
        rows = pricing.model_prices("amazon.cardmodel-v1:0")
        assert len(rows) == 7
        assert all(key.model == "cardmodel" for key, _ in rows)
        assert [key.region for key, _ in rows] == ["eu-west-1"] + ["us-east-1"] * 6

    def test_axis_filters_combine_with_and(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Region, tier, and dimension filters intersect."""
        self._seed(monkeypatch)
        rows = pricing.model_prices(
            "amazon.cardmodel-v1:0",
            region="us-east-1",
            tier="standard",
            dimensions={Dimension.INPUT_TOKENS},
        )
        assert [(key.dimension, key.context) for key, _ in rows] == [
            (Dimension.INPUT_TOKENS, ""),
            (Dimension.INPUT_TOKENS, "long"),
        ]

    def test_variants_false_keeps_only_base_rows_and_specs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The base card drops tier/TTL/routing/context variants, keeps specs."""
        self._seed(monkeypatch)
        rows = pricing.model_prices("amazon.cardmodel-v1:0", variants=False)
        assert {(key.dimension, key.spec, key.region) for key, _ in rows} == {
            (Dimension.INPUT_TOKENS, "", "us-east-1"),
            (Dimension.INPUT_TOKENS, "", "eu-west-1"),
            (Dimension.OUTPUT_IMAGES, "1024:standard", "us-east-1"),
        }

    def test_routing_and_context_filters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Routing and context filters match their exact rows."""
        self._seed(monkeypatch)
        (global_row,) = pricing.model_prices("amazon.cardmodel-v1:0", routing="global")
        assert global_row[0].dimension is Dimension.OUTPUT_TOKENS
        (long_row,) = pricing.model_prices("amazon.cardmodel-v1:0", context="long")
        assert long_row[1].amount == Decimal("0.000006")

    def test_model_key_overrides_are_resolved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An aliased/overridden model ID reads the same rows."""
        self._seed(monkeypatch)
        pricing.register_model_key_overrides({"vendor.other-alias-v9:9": "cardmodel"})
        try:
            assert len(pricing.model_prices("vendor.other-alias-v9:9")) == 7
        finally:
            pricing._MODEL_KEY_OVERRIDES.pop("vendor.other-alias-v9:9", None)  # noqa: SLF001

    def test_unknown_model_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A model with no indexed rows yields an empty list, not an error."""
        self._seed(monkeypatch)
        assert pricing.model_prices("vendor.unknown-v1:0") == []
