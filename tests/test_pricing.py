"""Unit tests for stdapi.pricing's AWS Price List parsing and resolution.

These exercise the pure parsing/normalization logic directly (no network),
using recorded AWS Price List API samples in tests/fixtures/pricing/.

The fixtures guard the three mispricing classes confirmed in production:
pricePerUnit/unit JSON level handling, model-key normalization against live
Bedrock IDs, and unvalidated operator price overrides.

Ref: stdapi/pricing.py:_ingest_price_list_item
     stdapi/pricing.py:resolve_price
     botocore/data/pricing/2017-10-15/service-2.json
"""

import asyncio
import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
from botocore.exceptions import BotoCoreError, ClientError

from stdapi import models, pricing
from stdapi.aws import AWSConnectionManager, get_client
from stdapi.config import SETTINGS
from stdapi.models.deprecation import DEPRECATED_MODELS
from stdapi.models.moderation import GUARDRAIL_CHECKS_MODERATION_MODEL
from stdapi.pricing import (
    KNOWLEDGE_BASE_MODEL,
    WEB_SEARCH_MODEL,
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
    guardrail_policy_model,
    inference_type_to_dimension,
    is_model_priced,
    normalize_model_key,
    normalize_usagetype_model,
    parse_unit_scale,
    refresh_price_catalog_for_new_models,
    resolve_price,
)
from tests.conftest import set_test_price

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator, Mapping

    from stdapi.monitoring import EventLog


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


@pytest.fixture(autouse=True)
def _reset_catalog_load_carry_state() -> Generator[None]:
    """Reset partial-load retry/cooldown state so one test can't leak into another."""
    yield
    pricing._state.pending_fetch_specs = None  # noqa: SLF001
    pricing._state.pending_index = {}  # noqa: SLF001
    pricing._state.pending_claims = {}  # noqa: SLF001
    pricing._state.unpriced_cooldown = {}  # noqa: SLF001
    pricing._state.catalog_complete = False  # noqa: SLF001


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


def _use_fake_catalog(
    monkeypatch: pytest.MonkeyPatch, client: object, region: str = "us-east-1"
) -> None:
    """Point the catalog loader at *client* and at a single region.

    The price index itself needs no reset: conftest's autouse ``_clean_price_index``
    already empties it before every test.
    """
    monkeypatch.setattr(pricing, "get_client", lambda *_a, **_k: client)
    monkeypatch.setattr(pricing, "_catalog_regions", lambda: {region})
    monkeypatch.setattr(SETTINGS, "cost_price_overrides", {})


def _price_item(
    attrs: Mapping[str, object], *, unit: str, price: str
) -> dict[str, object]:
    """Build a minimal AWS Price List item dict with one OnDemand price dimension.

    Args:
        attrs: The ``product.attributes`` mapping (must include ``regionCode``).
        unit: The ``priceDimensions`` ``unit`` string.
        price: The raw USD ``pricePerUnit`` string.

    Returns:
        A price-list item dict, ready for ``json.dumps()`` or direct use.
    """
    return {
        "product": {"attributes": dict(attrs)},
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


class TestIngestPriceListItem:
    """Parsing of real-shaped AWS Price List API items.

    Ref: stdapi/pricing.py:_ingest_price_list_item
    """

    def test_bedrock_fixture_produces_one_entry_per_item(self) -> None:
        """Every fixture item must yield exactly one price entry (none silently dropped)."""
        items = json.loads((_FIXTURES_DIR / "bedrock_sample.json").read_text())
        results, diagnostics = _ingest_fixture("bedrock_sample.json", Service.BEDROCK)
        assert diagnostics == []
        assert len(results) == len(items)

    def test_per_1k_token_price_is_divided_by_1000(self) -> None:
        """Regression: `unit: "1K tokens"` must scale the stored price, not be ignored.

        Ref: stdapi/pricing.py:parse_unit_scale
             stdapi/pricing.py:_parse_price
        """
        results, diagnostics = _ingest_fixture("bedrock_sample.json", Service.BEDROCK)
        assert diagnostics == []
        key = PriceKey(
            Service.BEDROCK, "novalite", "us-east-1", Dimension.INPUT_TOKENS, "standard"
        )
        # Fixture raw price is 0.00008 USD per 1K tokens.
        assert results[key].amount == Decimal("0.00008") / 1000
        assert results[key].currency == "USD"

    def test_service_tier_attribute_becomes_price_key_tier(self) -> None:
        """A flex-tier row must be keyed under tier="flex", not merged into another row.

        Ref: stdapi/pricing.py:_resolve_tier
        """
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

        The usagetype suffix is the only signal: these rows carry no
        service_tier attribute of their own.

        Ref: stdapi/pricing.py:_native_routing
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
        """Polly rows (no `model` attribute) must key the same way record_polly_usage does.

        Ref: stdapi/pricing.py:_synthesize_service_model_key
             stdapi/usage.py:record_polly_usage
        """
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
        """Transcribe rows must key the same way record_transcribe_usage does.

        Ref: stdapi/pricing.py:_synthesize_service_model_key
             stdapi/usage.py:record_transcribe_usage
        """
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

        Skipping them leaves a model priced only via Global routing with no
        price at all.

        Ref: stdapi/pricing.py:_marketplace_routing
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

        This app only bills inference usage, and an ingested customization row
        silently collides with the model's on-demand PriceKey.

        Ref: stdapi/pricing.py:_ingest_native_item
        """
        results: dict[PriceKey, Price] = {}
        item = json.dumps(
            _price_item(
                {
                    "regionCode": "us-east-1",
                    "usagetype": "USE1-NovaMicro-Model-Customization-input-tokens",
                    "inferenceType": "Input tokens",
                    "model": "Nova Micro",
                    "feature": "Model Customization",
                },
                unit="Tokens",
                price="0.001",
            )
        )
        _ingest_price_list_item(item, Service.BEDROCK, "us-east-1", "USD", results)
        assert results == {}

    def test_malformed_json_is_skipped_not_raised(self) -> None:
        """A single bad item must not raise -- callers rely on this to isolate failures.

        A valid item ingested into the same accumulator afterwards proves the
        bad one was skipped rather than the whole call being inert.
        """
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            "{not valid json", Service.BEDROCK, "us-east-1", "USD", results
        )
        assert results == {}

        good_item = json.dumps(
            _price_item(
                {
                    "regionCode": "us-east-1",
                    "usagetype": "USE1-SomeModel-input-tokens",
                    "inferenceType": "Input tokens",
                    "model": "Some Model",
                },
                unit="1K tokens",
                price="0.001",
            )
        )
        _ingest_price_list_item(good_item, Service.BEDROCK, "us-east-1", "USD", results)
        key = PriceKey(
            Service.BEDROCK,
            normalize_model_key("Some Model"),
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        assert set(results) == {key}
        assert results[key].amount == Decimal("0.001") / 1000

    @pytest.mark.parametrize("bad_price", ["", "N/A", "NaN", "Infinity"])
    def test_invalid_or_non_finite_price_per_unit_is_skipped_not_raised(
        self, bad_price: str
    ) -> None:
        """Regression: an invalid/non-finite pricePerUnit must be skipped, not raise.

        ``Decimal("")``/``Decimal("N/A")`` raise ``decimal.InvalidOperation``;
        ``Decimal("NaN")``/``Decimal("Infinity")`` parse but aren't valid prices.

        Ref: stdapi/pricing.py:_parse_price
        """
        results: dict[PriceKey, Price] = {}
        item = json.dumps(
            _price_item(
                {
                    "regionCode": "us-east-1",
                    "usagetype": "USE1-SomeModel-input-tokens",
                    "inferenceType": "Input tokens",
                    "model": "Some Model",
                },
                unit="1K tokens",
                price=bad_price,
            )
        )
        _ingest_price_list_item(item, Service.BEDROCK, "us-east-1", "USD", results)
        assert results == {}


class TestResolveTier:
    """AWS signals tier via 3 incompatible schemas -- _resolve_tier() must handle all 3.

    Ref: stdapi/pricing.py:_resolve_tier
    """

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
        no inferenceType suffix, no distinct feature -- so an undetected batch
        suffix collides with global-standard on the same PriceKey.
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

    Ref: stdapi/pricing.py:_store_price
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
    """Minimal fake aiobotocore pricing paginator serving items per ServiceCode.

    Ref: stdapi/pricing.py:_fetch_service_pricing
         botocore/data/pricing/2017-10-15/service-2.json
    """

    def __init__(
        self,
        items_by_service_code: dict[str, list[dict[str, object]]],
        delay_by_service_code: dict[str, float],
        raise_by_service_code: dict[str, Exception] | None = None,
        completed: list[str] | None = None,
    ) -> None:
        self._items = items_by_service_code
        self._delays = delay_by_service_code
        self._raises = raise_by_service_code or {}
        self._completed = completed

    async def paginate(self, **kwargs: object) -> AsyncIterator[dict[str, object]]:
        """Yield one page of the ServiceCode's items, after its optional delay/error."""
        service_code = str(kwargs["ServiceCode"])
        if delay := self._delays.get(service_code, 0.0):
            await asyncio.sleep(delay)
        if error := self._raises.get(service_code):
            raise error
        items = self._items.get(service_code, [])
        yield {"PriceList": [json.dumps(item) for item in items]}
        if self._completed is not None:
            self._completed.append(service_code)


class _FakePricingClient:
    """Minimal fake aiobotocore pricing client serving items per ServiceCode.

    Ref: stdapi/pricing.py:_fetch_service_pricing
         botocore/data/pricing/2017-10-15/service-2.json
    """

    def __init__(
        self,
        items_by_service_code: dict[str, list[dict[str, object]]],
        delay_by_service_code: dict[str, float] | None = None,
        raise_by_service_code: dict[str, Exception] | None = None,
        completed: list[str] | None = None,
    ) -> None:
        self._paginator = _FakePricingPaginator(
            items_by_service_code,
            delay_by_service_code or {},
            raise_by_service_code,
            completed,
        )

    def get_paginator(self, _name: str) -> _FakePricingPaginator:
        """Return the paginator serving this client's registered items."""
        return self._paginator


class TestCrossFetchCollisionDetection:
    """Collisions between two fetches in the same catalog load.

    Same-region Bedrock service codes claiming one PriceKey must warn and
    resolve deterministically (fetch generation order, not completion order).

    Ref: stdapi/pricing.py:_load_price_catalog
         stdapi/pricing.py:_store_price
    """

    #: The PriceKey both fetches' items below resolve to.
    _KEY = PriceKey(
        Service.BEDROCK, "somemodel", "us-east-1", Dimension.INPUT_TOKENS, "standard"
    )

    @staticmethod
    def _bedrock_item(usagetype: str, price: str) -> dict[str, object]:
        """Build one Bedrock price-list item resolving to ``_KEY``."""
        return _price_item(
            {
                "regionCode": "us-east-1",
                "usagetype": usagetype,
                "inferenceType": "Input tokens",
                "model": "Some Model",
            },
            unit="1K tokens",
            price=price,
        )

    async def _load_catalog(
        self,
        monkeypatch: pytest.MonkeyPatch,
        items_by_service_code: dict[str, list[dict[str, object]]],
        delay_by_service_code: dict[str, float] | None = None,
    ) -> tuple[dict[PriceKey, Price], list[str]]:
        """Run a full _load_price_catalog against a fake pricing client."""
        client = _FakePricingClient(items_by_service_code, delay_by_service_code)
        _use_fake_catalog(monkeypatch, client)
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


class TestPartialFetchFailureIsTolerated:
    """One fetch failing must not cancel or discard its sibling fetches.

    Ref: stdapi/pricing.py:_load_price_catalog
         stdapi/pricing.py:_fetch_or_capture
    """

    @staticmethod
    def _bedrock_item(usagetype: str, price: str) -> dict[str, object]:
        """Build one Bedrock price-list item resolving to a fixed PriceKey."""
        return _price_item(
            {
                "regionCode": "us-east-1",
                "usagetype": usagetype,
                "inferenceType": "Input tokens",
                "model": "Some Model",
            },
            unit="1K tokens",
            price=price,
        )

    #: The PriceKey both service codes' items below resolve to.
    _KEY = PriceKey(
        Service.BEDROCK, "somemodel", "us-east-1", Dimension.INPUT_TOKENS, "standard"
    )

    async def test_sibling_fetch_completes_and_partial_catalog_is_published(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A slow, throttled sibling must still complete and its price get published."""
        error_response = {"Error": {"Code": "ThrottlingException", "Message": "x"}}
        completed: list[str] = []
        client = _FakePricingClient(
            {
                "AmazonBedrock": [],
                "AmazonBedrockService": [
                    self._bedrock_item("USE1-SomeModel-input-tokens", "0.001")
                ],
            },
            delay_by_service_code={"AmazonBedrockService": 0.05},
            raise_by_service_code={
                "AmazonBedrock": ClientError(error_response, "GetProducts")  # type: ignore[arg-type]
            },
            completed=completed,
        )
        _use_fake_catalog(monkeypatch, client)

        await pricing._load_price_catalog([])  # noqa: SLF001

        # The throttled fetch didn't cancel its slower sibling.
        assert "AmazonBedrockService" in completed
        # The sibling's price is published even though AmazonBedrock failed.
        assert pricing._state.price_index[self._KEY].amount == Decimal("0.001") / 1000  # noqa: SLF001

    async def test_retry_refetches_only_the_failed_pair(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After a partial failure, only the failed (region, service_code) is queued to retry."""
        error_response = {"Error": {"Code": "ThrottlingException", "Message": "x"}}
        client = _FakePricingClient(
            {"AmazonBedrock": [], "AmazonBedrockService": []},
            raise_by_service_code={
                "AmazonBedrock": ClientError(error_response, "GetProducts")  # type: ignore[arg-type]
            },
        )
        _use_fake_catalog(monkeypatch, client)

        await pricing._load_price_catalog([])  # noqa: SLF001

        assert pricing._state.pending_fetch_specs == [("us-east-1", "AmazonBedrock")]  # noqa: SLF001

    async def test_final_success_after_retry_merges_everything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once the previously-failed fetch succeeds, its prices join the earlier successes."""
        error_response = {"Error": {"Code": "ThrottlingException", "Message": "x"}}
        completed: list[str] = []
        raises: dict[str, Exception] = {
            "AmazonBedrock": ClientError(error_response, "GetProducts")  # type: ignore[arg-type]
        }
        client = _FakePricingClient(
            {
                "AmazonBedrock": [
                    self._bedrock_item("USE1-SomeModel-input-tokens", "0.001")
                ],
                "AmazonBedrockService": [
                    self._bedrock_item("USE1-SomeModel-input-tokens", "0.001")
                ],
            },
            raise_by_service_code=raises,
            completed=completed,
        )
        _use_fake_catalog(monkeypatch, client)

        diagnostics: list[str] = []
        await pricing._load_price_catalog(diagnostics)  # noqa: SLF001
        assert pricing._state.pending_fetch_specs == [("us-east-1", "AmazonBedrock")]  # noqa: SLF001

        completed.clear()
        del raises["AmazonBedrock"]  # AWS throttling clears up.
        diagnostics = []
        await pricing._load_price_catalog(diagnostics)  # noqa: SLF001

        # Only the previously-failed pair was refetched, not the already-succeeded one.
        assert completed == ["AmazonBedrock"]
        assert pricing._state.pending_fetch_specs is None  # noqa: SLF001
        assert not any("collision" in d.lower() for d in diagnostics)
        assert pricing._state.price_index[self._KEY].amount == Decimal("0.001") / 1000  # noqa: SLF001


class TestNativeCacheTtl:
    """Native (non-Marketplace) 1-hour prompt-cache-write pricing.

    AWS spells the marker "-1-hour" on Bedrock's own rows and "-1h-" on the
    bedrock-mantle ones. Missing either folds the 1-hour rate onto the
    5-minute key, where the surviving rate is whichever row the Price List
    happened to return last.

    Ref: stdapi/pricing.py:_ingest_native_item
         stdapi/pricing.py:_CACHE_WRITE_1H_PATTERN
    """

    @staticmethod
    def _ingest(usagetype: str, price: str) -> tuple[dict[PriceKey, Price], list[str]]:
        """Ingest one native cache-write row, returning its results and diagnostics."""
        results: dict[PriceKey, Price] = {}
        diagnostics: list[str] = []
        item = json.dumps(
            _price_item(
                {
                    "regionCode": "us-east-1",
                    "usagetype": usagetype,
                    "inferenceType": "Prompt cache write input tokens",
                    "model": "Claude Sonnet 4.5",
                    "feature": "On-demand Inference",
                },
                unit="1K tokens",
                price=price,
            )
        )
        _ingest_price_list_item(
            item, Service.BEDROCK, "us-east-1", "USD", results, {}, diagnostics
        )
        return results, diagnostics

    @pytest.mark.parametrize(
        ("usagetype", "service"),
        [
            (
                "USE1-Claude4Sonnet-cache-write-input-token-count-1-hour",
                Service.BEDROCK,
            ),
            (
                "USE1-anthropic.claude-sonnet-4-5-mantle-cache-write-tokens-1h-standard",
                Service.BEDROCK_MANTLE,
            ),
        ],
    )
    def test_one_hour_usagetype_becomes_cache_ttl_1h(
        self, usagetype: str, service: Service
    ) -> None:
        """Either spelling of the 1-hour marker must key under cache_ttl="1h"."""
        results, _ = self._ingest(usagetype, "0.0075")
        key = PriceKey(
            service,
            "claudesonnet45",
            "us-east-1",
            Dimension.CACHE_WRITE_TOKENS,
            "standard",
            "1h",
        )
        assert key in results
        assert results[key].amount == Decimal("0.0075") / 1000

    def test_the_default_ttl_row_keeps_its_own_key(self) -> None:
        """A row with no 1-hour marker must not be bucketed as 1h."""
        results, _ = self._ingest(
            "USE1-anthropic.claude-sonnet-4-5-mantle-cache-write-tokens-standard",
            "0.00375",
        )
        assert [key.cache_ttl for key in results] == [""]

    def test_the_two_ttls_do_not_collide(self) -> None:
        """The 1-hour and default rows must survive as two separately-priced keys."""
        results: dict[PriceKey, Price] = {}
        claims: dict[PriceKey, str] = {}
        diagnostics: list[str] = []
        for usagetype, price in (
            (
                "USE1-anthropic.claude-sonnet-4-5-mantle-cache-write-tokens-1h-standard",
                "0.0075",
            ),
            (
                "USE1-anthropic.claude-sonnet-4-5-mantle-cache-write-tokens-standard",
                "0.00375",
            ),
        ):
            item = json.dumps(
                _price_item(
                    {
                        "regionCode": "us-east-1",
                        "usagetype": usagetype,
                        "inferenceType": "Prompt cache write input tokens",
                        "model": "Claude Sonnet 4.5",
                        "feature": "On-demand Inference",
                    },
                    unit="1K tokens",
                    price=price,
                )
            )
            _ingest_price_list_item(
                item, Service.BEDROCK, "us-east-1", "USD", results, claims, diagnostics
            )
        assert diagnostics == []
        assert {key.cache_ttl: price.amount for key, price in results.items()} == {
            "1h": Decimal("0.0075") / 1000,
            "": Decimal("0.00375") / 1000,
        }


class TestMarketplaceCacheTtl:
    """Marketplace-listed models' 5-minute (default) vs 1-hour prompt-cache-write pricing.

    Ref: stdapi/pricing.py:_marketplace_dimension_tier_ttl
    """

    @staticmethod
    def _ingest_marketplace_cache_row(
        usagetype: str, price: str
    ) -> dict[PriceKey, Price]:
        results: dict[PriceKey, Price] = {}
        item = json.dumps(
            _price_item(
                {
                    "regionCode": "us-east-1",
                    "usagetype": usagetype,
                    "servicename": "Claude Opus 4.5 (Amazon Bedrock Edition)",
                },
                unit="Units",
                price=price,
            )
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
    """Titan Image Generator's titanModel/T2I-I2I/resolution/quality pricing rows.

    Ref: stdapi/pricing.py:_image_generation_spec
         stdapi/pricing.py:_resolve_native_model
    """

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

    Nova Canvas carries a `model` attribute (unlike Titan Image Generator), so
    the wrong _resolve_native_model branch drops its real on-demand rows while
    Provisioned Throughput rows match the usagetype fallback and overwrite the
    image prices. Confirmed live on Nova 2.0 Omni/Pro.

    Ref: stdapi/pricing.py:_resolve_native_model
         stdapi/pricing.py:_native_price_spec
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


class TestVideoGenerationIngestion:
    """Nova Reel-like T2V/I2V rows: OUTPUT_SECONDS dimension, "hdres" usagetype -> spec="hd".

    Ref: stdapi/pricing.py:_native_price_spec
         stdapi/pricing.py:_VIDEO_GENERATION_PATTERN
    """

    @staticmethod
    def _item(inference_type: str, usagetype: str, price: str) -> str:
        return json.dumps(
            _price_item(
                {
                    "regionCode": "us-east-1",
                    "usagetype": usagetype,
                    "inferenceType": inference_type,
                    "model": "Nova Reel",
                },
                unit="Seconds",
                price=price,
            )
        )

    def test_t2v_row_maps_to_output_seconds_dimension(self) -> None:
        """A "T2V ..." inferenceType must resolve to Dimension.OUTPUT_SECONDS."""
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item(
                "T2V Standard fps Standard Resolution",
                "USE1-NovaReel-output-seconds",
                "0.06",
            ),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.BEDROCK,
            "novareel",
            "us-east-1",
            Dimension.OUTPUT_SECONDS,
            "standard",
        )
        assert key in results
        assert results[key].amount == Decimal("0.06")

    def test_i2v_row_also_maps_to_output_seconds(self) -> None:
        """An "I2V ..." (image-to-video) inferenceType must resolve the same as T2V."""
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item(
                "I2V Standard fps Standard Resolution",
                "USE1-NovaReel-output-seconds",
                "0.06",
            ),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.BEDROCK,
            "novareel",
            "us-east-1",
            Dimension.OUTPUT_SECONDS,
            "standard",
        )
        assert set(results) == {key}
        assert results[key].amount == Decimal("0.06")

    def test_hdres_usagetype_ingests_under_the_hd_spec_bucket(self) -> None:
        """A "HDResolution" usagetype segment must resolve to spec="hd", distinct from standard."""
        results: dict[PriceKey, Price] = {}
        claims: dict[PriceKey, str] = {}
        diagnostics: list[str] = []
        _ingest_price_list_item(
            self._item(
                "T2V Standard fps Standard Resolution",
                "USE1-NovaReel-output-seconds",
                "0.06",
            ),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
            claims,
            diagnostics,
        )
        _ingest_price_list_item(
            self._item(
                "T2V Standard fps HD Resolution",
                "USE1-NovaReel-HDResolution-output-seconds",
                "0.08",
            ),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
            claims,
            diagnostics,
        )
        assert diagnostics == []
        standard_key = PriceKey(
            Service.BEDROCK,
            "novareel",
            "us-east-1",
            Dimension.OUTPUT_SECONDS,
            "standard",
        )
        hd_key = PriceKey(
            Service.BEDROCK,
            "novareel",
            "us-east-1",
            Dimension.OUTPUT_SECONDS,
            "standard",
            "",
            "",
            "hd",
        )
        assert results[standard_key].amount == Decimal("0.06")
        assert results[hd_key].amount == Decimal("0.08")

    def test_resolve_price_round_trip_for_hd_spec(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """resolve_price() must find the ingested HD row via a real Bedrock model ID.

        Ref: stdapi/pricing.py:resolve_price
        """
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item(
                "T2V Standard fps HD Resolution",
                "USE1-NovaReel-HDResolution-output-seconds",
                "0.08",
            ),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
        )
        monkeypatch.setattr(pricing._state, "price_index", results)  # noqa: SLF001
        price = resolve_price(
            Service.BEDROCK,
            "amazon.nova-reel-v1:1",
            "us-east-1",
            Dimension.OUTPUT_SECONDS,
            spec="hd",
        )
        assert price is not None
        assert price.amount == Decimal("0.08")


class TestLumaRayIngestion:
    """Luma Ray rows: bare "Video" inferenceType, "Ray v2" model, HDRes/StandardRes.

    Ref: stdapi/pricing.py:inference_type_to_dimension
         stdapi/pricing.py:_native_price_spec
    """

    @staticmethod
    def _item(usagetype: str, price: str) -> str:
        return json.dumps(
            _price_item(
                {
                    "regionCode": "us-west-2",
                    "usagetype": usagetype,
                    "inferenceType": "Video",
                    "model": "Ray v2",
                    "provider": "Luma AI",
                },
                unit="Second",
                price=price,
            )
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
        """The model ID and the price row's "Ray v2" name must share one price key.

        Ref: stdapi/pricing.py:normalize_model_key
        """
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
        """resolve_price() must find the ingested HD row via the Bedrock model ID.

        Ref: stdapi/pricing.py:resolve_price
        """
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
    """Multimodal "image token count" must not resolve to OUTPUT_IMAGES.

    Ref: stdapi/pricing.py:inference_type_to_dimension
         stdapi/pricing.py:_resolve_dimension
    """

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
    """Nova 2.0 global-routed batch rows, signaled only by usagetype.

    Ref: stdapi/pricing.py:_resolve_tier
         stdapi/pricing.py:_native_routing
    """

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

    Ref: stdapi/pricing.py:_marketplace_dimension_tier_ttl
    """

    @staticmethod
    def _item(
        usagetype: str,
        price: str,
        servicename: str = "Claude Opus 4.5 (Amazon Bedrock Edition)",
    ) -> str:
        return json.dumps(
            _price_item(
                {
                    "regionCode": "eu-west-1",
                    "usagetype": usagetype,
                    "servicename": servicename,
                },
                unit="Units",
                price=price,
            )
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
    """Legacy "(100K)"-suffixed Marketplace listings must not be ingested at all.

    Ref: stdapi/pricing.py:_ingest_marketplace_item
         stdapi/pricing.py:_MARKETPLACE_CONTEXT_WINDOW_LISTING_PATTERN
    """

    @staticmethod
    def _item(servicename: str, price: str = "3.26") -> str:
        return json.dumps(
            _price_item(
                {
                    "regionCode": "us-east-1",
                    "usagetype": "USE1-MP:USE1_InputTokenCount-Units",
                    "servicename": servicename,
                },
                unit="Units",
                price=price,
            )
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

    Distinct products can share an identical usagetype string, so a claim keyed
    on that text alone misses two listings that normalize to the SAME model key
    while carrying different prices.

    Ref: stdapi/pricing.py:_ingest_marketplace_item
         stdapi/pricing.py:_store_price
    """

    @staticmethod
    def _item(servicename: str, price: str) -> str:
        return json.dumps(
            _price_item(
                {
                    "regionCode": "us-west-2",
                    "usagetype": "USW2-MP:USW2_InputTokenCount-Units",
                    "servicename": servicename,
                },
                unit="Units",
                price=price,
            )
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
    """Marketplace "_LatencyOptimized" usagetype rows -- a distinct, pricier serving profile.

    Ref: stdapi/pricing.py:_marketplace_routing
    """

    @staticmethod
    def _item(usagetype: str, price: str) -> str:
        return json.dumps(
            _price_item(
                {
                    "regionCode": "us-west-2",
                    "usagetype": usagetype,
                    "servicename": "Claude Opus 4.5 (Amazon Bedrock Edition)",
                },
                unit="Units",
                price=price,
            )
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


class TestMarketplaceScaleOneDimensions:
    """Marketplace usagetypes billed flat per-unit, not per-1M (createdimage/searchunit/...).

    Unlike token dimensions (see _MARKETPLACE_PER_MILLION_DIMENSIONS), these
    quote a per-unit price directly -- no /1_000_000 scaling applies.

    Ref: stdapi/pricing.py:_ingest_marketplace_item
         stdapi/pricing.py:_MARKETPLACE_PER_MILLION_DIMENSIONS
    """

    @staticmethod
    def _item(usagetype: str, price: str) -> str:
        return json.dumps(
            _price_item(
                {
                    "regionCode": "us-east-1",
                    "usagetype": usagetype,
                    "servicename": "Test Image Model (Amazon Bedrock Edition)",
                },
                unit="Units",
                price=price,
            )
        )

    def test_created_image_usagetype_resolves_the_unscaled_per_image_price(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A "CreatedImage" usagetype must price per image, not divided by 1M."""
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item("USE1-MP:USE1_CreatedImage-Units", "0.04"),
            Service.BEDROCK,
            "us-east-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.BEDROCK,
            normalize_model_key("Test Image Model"),
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
            "standard",
        )
        assert key in results
        assert results[key].amount == Decimal("0.04")

        monkeypatch.setattr(pricing._state, "price_index", results)  # noqa: SLF001
        price = resolve_price(
            Service.BEDROCK, "Test Image Model", "us-east-1", Dimension.OUTPUT_IMAGES
        )
        assert price is not None
        assert price.amount == Decimal("0.04")


class TestLongContextAxis:
    """Long-context (>200K prompt) pricing bucket, signaled by a "long-context" usagetype segment.

    Ref: stdapi/pricing.py:_price_context
    """

    def test_long_context_input_tokens_row_does_not_collide_with_standard_row(
        self,
    ) -> None:
        """The long-context row must index separately from the standard global row."""
        results: dict[PriceKey, Price] = {}
        claims: dict[PriceKey, str] = {}
        diagnostics: list[str] = []
        standard_item = json.dumps(
            _price_item(
                {
                    "regionCode": "us-east-1",
                    "usagetype": "USE1-Claude4Sonnet-input-tokens-cross-region-global",
                    "inferenceType": "Input tokens",
                    "model": "Claude Sonnet 4",
                },
                unit="1K tokens",
                price="0.003",
            )
        )
        long_item = json.dumps(
            _price_item(
                {
                    "regionCode": "us-east-1",
                    "usagetype": (
                        "USE1-Claude4Sonnet-input-tokens-long-context"
                        "-cross-region-global"
                    ),
                    "inferenceType": "Input tokens long context",
                    "model": "Claude Sonnet 4",
                },
                unit="1K tokens",
                price="0.006",
            )
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
            _price_item(
                {
                    "regionCode": "us-east-1",
                    "usagetype": (
                        "USE1-Claude4Sonnet-cache-read-input-token-count"
                        "-cross-region-global"
                    ),
                    "featuretype": "Prompt cache read",
                    "model": "Claude Sonnet 4",
                },
                unit="1K tokens",
                price="0.0003",
            )
        )
        long_item = json.dumps(
            _price_item(
                {
                    "regionCode": "us-east-1",
                    "usagetype": (
                        "USE1-Claude4Sonnet-cache-read-input-token-count"
                        "-long-context-cross-region-global"
                    ),
                    "featuretype": "Prompt cache read",
                    "model": "Claude Sonnet 4",
                },
                unit="1K tokens",
                price="0.0006",
            )
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
    """Nova Multimodal Embeddings' per-input-image SKUs are not billed usage; its per-input-tokens SKU is.

    Ref: stdapi/pricing.py:_native_price_spec
    """

    @staticmethod
    def _item(usagetype: str, unit: str, price: str) -> str:
        return json.dumps(
            _price_item(
                {"regionCode": "us-east-1", "usagetype": usagetype},
                unit=unit,
                price=price,
            )
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
        # "Images Processed" carries no K/M multiplier: the price is per unit.
        assert results[key].amount == Decimal("0.0008")

    def test_input_tokens_row_ingests_via_usagetype_token_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ "input-tokens" (no inferenceType) must ingest as INPUT_TOKENS.

        Ref: stdapi/pricing.py:_generic_usagetype_dimension
        """  # noqa: D210
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
        """ "input-tokens-batch" must be keyed under tier="batch".

        Ref: stdapi/pricing.py:_resolve_tier
        """  # noqa: D210
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
    """Nova Grounding (built-in web-grounding tool) $/request pricing.

    Ref: stdapi/pricing.py:_generic_usagetype_dimension
         stdapi/pricing.py:_USAGETYPE_FALLBACK_DIMENSIONS
    """

    def test_nova_grounding_usagetype_maps_to_grounding_requests_dimension(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The "nova-grounding" usagetype marker must resolve to Dimension.GROUNDING_REQUESTS."""
        results: dict[PriceKey, Price] = {}
        item = json.dumps(
            _price_item(
                {
                    "regionCode": "us-east-1",
                    "usagetype": "USE1-Nova2.0Lite-nova-grounding",
                    "model": "Nova 2.0 Lite",
                },
                unit="Requests",
                price="0.03",
            )
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
    """Reserved-capacity ("Reserved - N Month") rows are a commitment rate, not billed by this app.

    Ref: stdapi/pricing.py:_ingest_native_item
    """

    def test_reserved_capacity_row_is_not_ingested(self) -> None:
        """A "Reserved - 1 Month" feature row must be skipped, even with no inferenceType."""
        results: dict[PriceKey, Price] = {}
        item = json.dumps(
            _price_item(
                {
                    "regionCode": "eu-west-1",
                    "usagetype": (
                        "EU-Claude4.5Sonnet-reserved-1-month-input-tokens"
                        "-per-minute-cross-region-global"
                    ),
                    "feature": "Reserved - 1 Month",
                    "model": "Claude Sonnet 4.5",
                },
                unit="1K tokens",
                price="1.5",
            )
        )
        _ingest_price_list_item(item, Service.BEDROCK, "eu-west-1", "USD", results)
        assert results == {}


class TestTranslateAndComprehendIngestion:
    """TranslateText/DetectDominantLanguage/DetectToxicContent synthetic model keys.

    These services carry no `model` attribute; _synthesize_service_model_key()
    reconstructs the exact synthetic model string record_translate_usage()/
    record_comprehend_usage() bill against (see stdapi/usage.py).

    Ref: stdapi/pricing.py:_synthesize_service_model_key
         stdapi/usage.py:record_translate_usage
         stdapi/usage.py:record_comprehend_usage
    """

    @staticmethod
    def _item(operation: str, usagetype: str, unit: str, price: str) -> str:
        return json.dumps(
            _price_item(
                {
                    "regionCode": "us-east-1",
                    "usagetype": usagetype,
                    "operation": operation,
                },
                unit=unit,
                price=price,
            )
        )

    def test_translate_text_ingests_under_the_synthetic_translate_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A TranslateText row must key under "amazon.translate", billed via INPUT_CHARACTERS."""
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item("TranslateText", "USE1-TranslateText", "Characters", "0.000015"),
            Service.TRANSLATE,
            "us-east-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.TRANSLATE,
            normalize_model_key("amazon.translate"),
            "us-east-1",
            Dimension.INPUT_CHARACTERS,
            "standard",
        )
        assert key in results
        assert results[key].amount == Decimal("0.000015")

        monkeypatch.setattr(pricing._state, "price_index", results)  # noqa: SLF001
        price = resolve_price(
            Service.TRANSLATE,
            "amazon.translate",
            "us-east-1",
            Dimension.INPUT_CHARACTERS,
        )
        assert price is not None
        assert price.amount == Decimal("0.000015")

    def test_detect_dominant_language_ingests_under_the_language_detection_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A DetectDominantLanguage row must key under "amazon.comprehend-language-detection"."""
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item(
                "DetectDominantLanguage",
                "USE1-DetectDominantLanguage",
                "Units",
                "0.0001",
            ),
            Service.COMPREHEND,
            "us-east-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.COMPREHEND,
            normalize_model_key("amazon.comprehend-language-detection"),
            "us-east-1",
            Dimension.COMPREHEND_UNITS,
            "standard",
        )
        assert key in results
        assert results[key].amount == Decimal("0.0001")

        monkeypatch.setattr(pricing._state, "price_index", results)  # noqa: SLF001
        price = resolve_price(
            Service.COMPREHEND,
            "amazon.comprehend-language-detection",
            "us-east-1",
            Dimension.COMPREHEND_UNITS,
        )
        assert price is not None
        assert price.amount == Decimal("0.0001")

    def test_detect_toxic_content_ingests_under_the_toxicity_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A DetectToxicContent row must key under "amazon.comprehend-toxicity"."""
        results: dict[PriceKey, Price] = {}
        _ingest_price_list_item(
            self._item(
                "DetectToxicContent", "USE1-DetectToxicContent", "Units", "0.0004"
            ),
            Service.COMPREHEND,
            "us-east-1",
            "USD",
            results,
        )
        key = PriceKey(
            Service.COMPREHEND,
            normalize_model_key("amazon.comprehend-toxicity"),
            "us-east-1",
            Dimension.COMPREHEND_UNITS,
            "standard",
        )
        assert key in results
        assert results[key].amount == Decimal("0.0004")

        monkeypatch.setattr(pricing._state, "price_index", results)  # noqa: SLF001
        price = resolve_price(
            Service.COMPREHEND,
            "amazon.comprehend-toxicity",
            "us-east-1",
            Dimension.COMPREHEND_UNITS,
        )
        assert price is not None
        assert price.amount == Decimal("0.0004")


class TestGuardrailsIngestion:
    """Bedrock Guardrails rows and their synthetic model keys.

    Guardrails rows carry no `model`, `inferenceType` or `featuretype`
    attribute, so both the key and the dimension are reconstructed from the
    usagetype -- otherwise guardrail spend can never be priced. The
    usagetypes below are the ones the Price List API publishes verbatim.

    Ref: https://aws.amazon.com/bedrock/pricing/
         stdapi/pricing.py:_guardrail_model
         stdapi/usage.py:record_guardrail_policy_usage
    """

    @staticmethod
    def _item(usagetype: str, price: str, unit: str = "TextUnit") -> str:
        return json.dumps(
            _price_item(
                {"regionCode": "us-east-1", "usagetype": usagetype},
                unit=unit,
                price=price,
            )
        )

    def _ingest(self, *items: str) -> tuple[dict[PriceKey, Price], list[str]]:
        """Ingest *items* into one shared batch, returning its results and diagnostics."""
        results: dict[PriceKey, Price] = {}
        claims: dict[PriceKey, str] = {}
        diagnostics: list[str] = []
        for item in items:
            _ingest_price_list_item(
                item, Service.BEDROCK, "us-east-1", "USD", results, claims, diagnostics
            )
        return results, diagnostics

    def test_guardrail_policy_row_prices_its_own_policy_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A policy row must key under that policy's model, in text units.

        The usagetype ends in "UnitsConsumed", which the generic Bedrock
        fallback reads as generated images -- a dimension nothing records
        against a guardrail, so the rate would never be found.
        """
        results, _ = self._ingest(
            self._item("USE1-Guardrail-ContentPolicyUnitsConsumed", "0.00015")
        )
        key = PriceKey(
            Service.BEDROCK,
            normalize_model_key(guardrail_policy_model("content")),
            "us-east-1",
            Dimension.TEXT_UNITS,
            "standard",
        )
        assert key in results

        monkeypatch.setattr(pricing._state, "price_index", results)  # noqa: SLF001
        price = resolve_price(
            Service.BEDROCK,
            guardrail_policy_model("content"),
            "us-east-1",
            Dimension.TEXT_UNITS,
        )
        assert price is not None
        assert price.amount == Decimal("0.00015")

    def test_image_content_policy_row_prices_input_images(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The content policy's image rate shares its model, split off by dimension.

        AWS prices the same policy per text unit and per image; the two rows
        stay distinct because image moderation records INPUT_IMAGES.
        """
        results, _ = self._ingest(
            self._item("USE1-Guardrail-ContentPolicyUnitsConsumed", "0.00015"),
            self._item(
                "USE1-Guardrail-ContentPolicyImageUnitsConsumed",
                "0.00075",
                unit="Images Processed",
            ),
        )
        monkeypatch.setattr(pricing._state, "price_index", results)  # noqa: SLF001
        price = resolve_price(
            Service.BEDROCK,
            guardrail_policy_model("content"),
            "us-east-1",
            Dimension.INPUT_IMAGES,
        )
        assert price is not None
        assert price.amount == Decimal("0.00075")

    def test_each_policy_keeps_its_own_rate(self) -> None:
        """Policies must not share a PriceKey: a guardrail pays every one it applies.

        Folding them onto one model could only ever charge a single policy's
        rate, so a multi-policy guardrail would be billed a fraction of its
        real cost no matter which row won.
        """
        results, diagnostics = self._ingest(
            self._item(
                "USE1-Guardrail-ContextualGroundingPolicyUnitsConsumed", "0.0001"
            ),
            self._item(
                "USE1-Guardrail-AutomatedReasoningPolicyUnitsConsumed", "0.00017"
            ),
            self._item("USE1-Guardrail-TopicPolicyUnitsConsumed", "0.00015"),
        )
        assert diagnostics == []
        assert {key.model: price.amount for key, price in results.items()} == {
            normalize_model_key(
                guardrail_policy_model("contextual-grounding")
            ): Decimal("0.0001"),
            normalize_model_key(guardrail_policy_model("automated-reasoning")): Decimal(
                "0.00017"
            ),
            normalize_model_key(guardrail_policy_model("topic")): Decimal("0.00015"),
        }

    def test_an_unmodeled_policy_is_reported_rather_than_billed_at_nothing(
        self,
    ) -> None:
        """A policy AWS adds later must surface, not silently cost nothing.

        Its units would otherwise be recorded against no price key at all,
        and TEXT_UNITS misses are deliberately silent at request time.
        """
        results, diagnostics = self._ingest(
            self._item("USE1-Guardrail-QuantumPolicyUnitsConsumed", "0.00042")
        )
        assert results == {}
        assert len(diagnostics) == 1
        assert "QuantumPolicyUnitsConsumed" in diagnostics[0]

    def test_guardrail_checks_row_prices_the_checks_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A guardrail-checks row must key under the distinct checks model.

        The two APIs are billed separately, so their rows must never fold
        onto one key.
        """
        results, _ = self._ingest(
            self._item(
                "USE1-GuardrailChecks-ContentFilterCheckUnitsConsumed", "0.00007"
            )
        )
        assert [key.model for key in results] == [
            normalize_model_key(GUARDRAIL_CHECKS_MODERATION_MODEL)
        ]

        monkeypatch.setattr(pricing._state, "price_index", results)  # noqa: SLF001
        price = resolve_price(
            Service.BEDROCK,
            GUARDRAIL_CHECKS_MODERATION_MODEL,
            "us-east-1",
            Dimension.TEXT_UNITS,
        )
        assert price is not None
        assert price.amount == Decimal("0.00007")

    def test_an_uninvoked_check_is_not_ingested_at_all(self) -> None:
        """Only the content filter is ever requested, so the other checks' rates are dropped.

        Keeping them would either overwrite the content-filter rate or mint a
        model key per check that nothing prices against.
        """
        results, diagnostics = self._ingest(
            self._item(
                "USE1-GuardrailChecks-ContentFilterCheckUnitsConsumed", "0.00007"
            ),
            self._item(
                "USE1-GuardrailChecks-PromptAttackCheckUnitsConsumed", "0.00008"
            ),
            self._item(
                "USE1-GuardrailChecks-SensitiveInformationCheckUnitsConsumed", "0.0001"
            ),
        )
        assert diagnostics == []
        assert [(key.model, price.amount) for key, price in results.items()] == [
            (normalize_model_key(GUARDRAIL_CHECKS_MODERATION_MODEL), Decimal("0.00007"))
        ]


class TestWebSearchIngestion:
    """The built-in web search row and its synthetic model key.

    AWS publishes one flat per-query rate that carries no ``model``
    attribute, because the same rate applies to every model that can call the
    tool. Without a synthetic key the row resolves to no model at all and web
    search spend is reported as free.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html
         https://aws.amazon.com/bedrock/pricing/
         stdapi/pricing.py:_bedrock_synthetic_model
         stdapi/usage.py:record_web_search_usage
    """

    @staticmethod
    def _ingest(usagetype: str, region: str) -> dict[PriceKey, Price]:
        """Ingest one published web search row and return the price index."""
        results: dict[PriceKey, Price] = {}
        item = json.dumps(
            _price_item(
                {"regionCode": region, "usagetype": usagetype, "operation": ""},
                unit="Queries",
                price="0.0120000000",
            )
        )
        _ingest_price_list_item(item, Service.BEDROCK, region, "USD", results, {}, [])
        return results

    def test_web_search_row_prices_the_web_search_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The published row resolves to the key the recorder bills against."""
        region = "us-east-1"
        results = self._ingest("USE1-Bedrock-Websearch-Queries", region)
        assert list(results) == [
            PriceKey(
                Service.BEDROCK,
                normalize_model_key(WEB_SEARCH_MODEL),
                region,
                Dimension.GROUNDING_REQUESTS,
                "standard",
            )
        ]

        monkeypatch.setattr(pricing._state, "price_index", results)  # noqa: SLF001
        price = resolve_price(
            Service.BEDROCK, WEB_SEARCH_MODEL, region, Dimension.GROUNDING_REQUESTS
        )
        assert price is not None
        assert price.amount == Decimal("0.012")

    def test_a_mantle_segment_would_put_the_rate_out_of_reach(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A "mantle" usagetype segment keys the rate under the other service.

        The tool is only reachable through Mantle-served models, so a row named
        ``...-Mantle-Websearch-Queries`` is the plausible alternative spelling
        -- and it would key under the Mantle service while
        ``record_web_search_usage`` records under Bedrock's, leaving every query
        unpriced with nothing failing. The live catalog test is the authority on
        the string AWS actually publishes.

        Ref: stdapi/pricing.py:_bedrock_api_service
             stdapi/pricing.py:_bedrock_synthetic_model
             tests/test_pricing.py:test_live_price_catalog_ingests_cleanly
        """
        results = self._ingest("USE1-Bedrock-Mantle-Websearch-Queries", "us-east-1")
        assert list(results) == [
            PriceKey(
                Service.BEDROCK_MANTLE,
                normalize_model_key(WEB_SEARCH_MODEL),
                "us-east-1",
                Dimension.GROUNDING_REQUESTS,
                "standard",
            )
        ]

        monkeypatch.setattr(pricing._state, "price_index", results)  # noqa: SLF001
        assert (
            resolve_price(
                Service.BEDROCK,
                WEB_SEARCH_MODEL,
                "us-east-1",
                Dimension.GROUNDING_REQUESTS,
            )
            is None
        )


class TestUsagetypeTokenFallbackTierSuffixes:
    """Native rows with no inferenceType/featuretype at all (xai.grok's "mantle" usagetype schema).

    "mantle" rows are the bedrock-mantle API's rates: they must key under
    ``Service.BEDROCK_MANTLE``, never mixing with bedrock-runtime rates.

    Ref: stdapi/pricing.py:_bedrock_api_service
         stdapi/pricing.py:_generic_usagetype_dimension
    """

    @staticmethod
    def _item(usagetype: str, model: str = "xai.grok-4.3", price: str = "0.002") -> str:
        return json.dumps(
            _price_item(
                {"regionCode": "us-east-1", "usagetype": usagetype, "model": model},
                unit="1K tokens",
                price=price,
            )
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
        assert set(results) == {key}
        assert results[key].amount == Decimal("0.002") / 1000

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
        assert set(results) == {key}
        assert results[key].amount == Decimal("0.002") / 1000

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
        assert set(results) == {key}
        assert results[key].amount == Decimal("0.002") / 1000


class TestNovaSonicModality:
    """Nova Sonic (speech-to-speech) rows: text-modality tokens billed, speech-modality rows unmapped.

    Ref: stdapi/pricing.py:inference_type_to_dimension
         stdapi/pricing.py:_native_price_spec
    """

    @staticmethod
    def _item(inference_type: str, usagetype: str, price: str = "0.0034") -> str:
        return json.dumps(
            _price_item(
                {
                    "regionCode": "us-east-1",
                    "usagetype": usagetype,
                    "inferenceType": inference_type,
                    "model": "Nova Sonic",
                },
                unit="1K tokens",
                price=price,
            )
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
        assert set(results) == {key}
        assert results[key].amount == Decimal("0.0034") / 1000

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
        assert set(results) == {key}
        assert results[key].amount == Decimal("0.0034") / 1000

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
        assert results[key].amount == Decimal("0.0034") / 1000

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

    Ref: stdapi/pricing.py:_resolve_native_model
         stdapi/pricing.py:_LATENCY_OPTIMIZED_SUFFIX_PATTERN
    """

    @staticmethod
    def _item(model: str, usagetype: str, price: str) -> str:
        return json.dumps(
            _price_item(
                {
                    "regionCode": "us-east-1",
                    "usagetype": usagetype,
                    "inferenceType": "Input tokens",
                    "model": model,
                },
                unit="1K tokens",
                price=price,
            )
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
    """Matching Bedrock model IDs to AWS Price List `model` display names.

    Ref: stdapi/pricing.py:normalize_model_key
    """

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
    """register_model_key_overrides() merges entries into the override registry.

    Ref: stdapi/pricing.py:register_model_key_overrides
         stdapi/pricing.py:resolve_model_key
    """

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
    """`stdapi.models` registers its `pricing_overrides.MODEL_KEY_OVERRIDES` at import time.

    Ref: stdapi/models/pricing_overrides.py:MODEL_KEY_OVERRIDES
         stdapi/pricing.py:resolve_model_key
    """

    def test_nova_2_lite_resolves_via_the_registered_override(self) -> None:
        """ "amazon.nova-2-lite-v1:0" must resolve to "nova20lite" via the registered table."""  # noqa: D210
        assert pricing.resolve_model_key("amazon.nova-2-lite-v1:0") == "nova20lite"


class TestDefaultModelPrices:
    """Built-in pricing-page defaults: applied only to models with no published row.

    Ref: stdapi/pricing.py:_apply_default_prices
         stdapi/models/pricing_overrides.py:DEFAULT_MODEL_PRICES
    """

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

    @pytest.mark.parametrize(
        ("model_id", "input_rate", "output_rate"),
        [
            ("openai.gpt-5.6-cyber", "0.00001375", "0.0000825"),
            ("openai.gpt-daybreak-blue-5.6-sol", "0.0000055", "0.000033"),
        ],
    )
    def test_daybreak_models_price_at_their_model_card_rate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_id: str,
        input_rate: str,
        output_rate: str,
    ) -> None:
        """Daybreak Red and Blue resolve a price instead of billing at zero.

        Both are Mantle-only and in-Region in us-east-2, and the OpenAI Mantle
        rows are absent from the Price List API, so the model-card rate is the
        only source a deployment has.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-cyber.html
             https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-daybreak-blue-56-sol.html
        """
        index: dict[PriceKey, Price] = {}
        pricing._apply_default_prices(index)  # noqa: SLF001
        monkeypatch.setattr(pricing._state, "price_index", index)  # noqa: SLF001
        prices = {
            dimension: resolve_price(
                Service.BEDROCK_MANTLE, model_id, "us-east-2", dimension
            )
            for dimension in (Dimension.INPUT_TOKENS, Dimension.OUTPUT_TOKENS)
        }
        assert all(price is not None for price in prices.values())
        assert prices[Dimension.INPUT_TOKENS].amount == Decimal(input_rate)  # type: ignore[union-attr]
        assert prices[Dimension.OUTPUT_TOKENS].amount == Decimal(output_rate)  # type: ignore[union-attr]

    def test_gap_model_prices_identically_on_the_mantle_service(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same default rate resolves for Bedrock and for Bedrock Mantle.

        The pricing page does not distinguish invocation APIs, so a Mantle-served
        model would otherwise bill at zero while its runtime twin bills correctly.

        Ref: stdapi/pricing.py:register_default_prices
        """
        index: dict[PriceKey, Price] = {}
        pricing._apply_default_prices(index)  # noqa: SLF001
        monkeypatch.setattr(pricing._state, "price_index", index)  # noqa: SLF001
        prices = [
            resolve_price(
                service,
                "stability.stable-image-inpaint-v1:0",
                "us-east-1",
                Dimension.OUTPUT_IMAGES,
            )
            for service in (Service.BEDROCK, Service.BEDROCK_MANTLE)
        ]
        assert all(price is not None for price in prices)
        assert {price.amount for price in prices if price} == {Decimal("0.07")}

    def test_registration_keys_every_service_region_and_dimension(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Registering one model fans out over both services, every region and dimension.

        The registry is a plain dict keyed by ``PriceKey``, so the fan-out at
        registration time is the only thing that makes a default reachable; the
        tier is pinned to ``standard`` so the tier-ratio fallback can scale it.

        Ref: stdapi/pricing.py:register_default_prices
        """
        monkeypatch.setattr(pricing, "_DEFAULT_PRICES", {})
        pricing.register_default_prices(
            {
                "test.gap-model": {
                    Dimension.INPUT_TOKENS: "0.000123",
                    Dimension.OUTPUT_TOKENS: "0.000456",
                }
            },
            ["us-east-1", "eu-west-1"],
        )

        registry = pricing._DEFAULT_PRICES  # noqa: SLF001
        model = pricing.resolve_model_key("test.gap-model")
        assert len(registry) == 8, "2 services x 2 regions x 2 dimensions"
        assert {key.service for key in registry} == {
            Service.BEDROCK,
            Service.BEDROCK_MANTLE,
        }
        assert {key.region for key in registry} == {"us-east-1", "eu-west-1"}
        assert {key.tier for key in registry} == {"standard"}
        key = PriceKey(
            Service.BEDROCK_MANTLE,
            model,
            "eu-west-1",
            Dimension.OUTPUT_TOKENS,
            "standard",
        )
        assert registry[key] == Price(Decimal("0.000456"), "USD")

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
        """cost_price_overrides must overwrite a built-in default price.

        Ref: stdapi/pricing.py:_apply_price_overrides
        """
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


class TestKnowledgeBaseRetrievalPrice:
    """The managed knowledge base's per-retrieval rate, published only on the page.

    AWS charges $1.00 per 1,000 standard retrieval calls of a managed
    knowledge base, and the Price List API carries no row for it at all -- no
    usagetype of any Bedrock service code mentions knowledge bases. The rate
    therefore reaches the catalog only as a built-in default, and a search
    would otherwise be reported as free.

    Ref: https://aws.amazon.com/bedrock/pricing/
         stdapi/usage.py:record_knowledge_base_usage
    """

    def test_a_retrieval_resolves_at_the_published_rate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The key the recorder bills against prices at $0.001 per retrieval."""
        index: dict[PriceKey, Price] = {}
        pricing._apply_default_prices(index)  # noqa: SLF001
        monkeypatch.setattr(pricing._state, "price_index", index)  # noqa: SLF001

        price = resolve_price(
            Service.BEDROCK, KNOWLEDGE_BASE_MODEL, "us-east-1", Dimension.SEARCH_UNITS
        )

        assert price is not None
        assert price.amount == Decimal("0.001")
        assert price.currency == "USD"

    def test_the_rate_reaches_a_region_the_page_does_not_quote(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deployment outside the quoted regions still prices its retrievals.

        The page quotes three US regions; the regional fallback is what carries
        the rate to the rest of the partition, and without it a European
        deployment would report every search as free.

        Ref: stdapi/pricing.py:_apply_regional_fallback
        """
        index: dict[PriceKey, Price] = {}
        pricing._apply_default_prices(index)  # noqa: SLF001
        pricing._apply_regional_fallback(index, {"eu-west-1"})  # noqa: SLF001
        monkeypatch.setattr(pricing._state, "price_index", index)  # noqa: SLF001

        price = resolve_price(
            Service.BEDROCK, KNOWLEDGE_BASE_MODEL, "eu-west-1", Dimension.SEARCH_UNITS
        )

        assert price is not None
        assert price.amount == Decimal("0.001")


class TestParseUnitScale:
    """Scale-multiplier parsing from AWS Price List `unit` strings.

    Ref: stdapi/pricing.py:parse_unit_scale
    """

    @pytest.mark.parametrize(
        ("unit", "expected_scale"),
        [
            ("1K tokens", 1000),
            ("1M Characters", 1_000_000),
            ("Tokens", 1),
            # The digit run scales the multiplier: a K/M prefix is not always 1000x/1e6x.
            ("10K tokens", 10_000),
            ("5M characters", 5_000_000),
        ],
    )
    def test_scale_multiplier_uses_the_actual_leading_digit(
        self, unit: str, expected_scale: int
    ) -> None:
        """The leading digit run must multiply the K/M scale, not be discarded."""
        assert parse_unit_scale(unit) == expected_scale

    @pytest.mark.parametrize(
        ("unit", "expected_scale"),
        [
            ("1K tokens", 1000),
            ("1M", 1_000_000),
            ("1000 tokens", 1000),
            ("1 image", 1),
            ("not a real unit", 1),
        ],
    )
    def test_scale_without_trailing_whitespace_or_km_multiplier(
        self, unit: str, expected_scale: int
    ) -> None:
        """A bare "1M" and a plain numeric multiplier ("1000 tokens") must both parse."""
        assert parse_unit_scale(unit) == expected_scale


class TestNormalizeUsagetypeModel:
    """Model-key extraction from usagetype text for `model`-less price-list rows.

    Ref: stdapi/pricing.py:normalize_usagetype_model
    """

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
    """Operator-supplied COST_PRICE_OVERRIDES validation.

    Ref: stdapi/pricing.py:_apply_price_overrides
    """

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
        assert index[us_key].amount == Decimal("0.000003")
        assert index[eusc_key].currency == "EUR"
        assert index[eusc_key].amount == Decimal("0.000003")

    def test_override_also_resolves_under_bedrock_mantle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An override must win for Mantle-routed requests too, not just bedrock-runtime."""
        monkeypatch.setattr(
            SETTINGS,
            "cost_price_overrides",
            {"some-mantle-model": {"input_tokens": 0.5}},
        )
        index: dict[PriceKey, Price] = {}
        _apply_price_overrides(index, {"us-east-1"})
        monkeypatch.setattr(pricing._state, "price_index", index)  # noqa: SLF001
        price = resolve_price(
            Service.BEDROCK_MANTLE,
            "some-mantle-model",
            "us-east-1",
            Dimension.INPUT_TOKENS,
        )
        assert price is not None
        assert price.amount == Decimal("0.5")
        assert price.currency == "USD"

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


class TestCatalogRegions:
    """The region set the catalog is fetched, backfilled and overridden for.

    Mantle usage is recorded under the region that served the call, so a
    Mantle region missing from this set leaves that traffic unpriced with no
    regional fallback and no override reachable.

    Ref: stdapi/pricing.py:_catalog_regions
         stdapi/config.py:aws_bedrock_mantle_regions
    """

    def test_mantle_only_region_is_part_of_the_catalog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Mantle region absent from aws_bedrock_regions must still be fetched."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1"])
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_regions", ["ap-south-1"])
        assert "ap-south-1" in pricing._catalog_regions()  # noqa: SLF001


class TestApplyRegionalFallback:
    """Near-region-first backfill for models AWS hasn't priced in every region.

    Ref: stdapi/pricing.py:_apply_regional_fallback
         stdapi/pricing.py:_FALLBACK_ANCHOR_REGIONS
    """

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
        """Geo prefix is everything before the first hyphen.

        Ref: stdapi/pricing.py:_region_family
        """
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
        """The endpoint matches the first configured Bedrock region's geography.

        Ref: stdapi/pricing.py:pricing_endpoint_region
             stdapi/pricing.py:_PRICING_ENDPOINT_BY_GEOGRAPHY
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", [bedrock_region])
        assert pricing.pricing_endpoint_region() == endpoint

    async def test_catalog_load_skips_partitions_without_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GovCloud: the load no-ops with a diagnostic instead of calling AWS.

        Ref: stdapi/pricing.py:_load_price_catalog
             stdapi/pricing.py:_UNPRICED_PARTITIONS
        """
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
        """GovCloud: operator cost_price_overrides remain the sole price source.

        Ref: stdapi/pricing.py:_apply_price_overrides
        """
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
    """Tier-aware price resolution.

    Ref: stdapi/pricing.py:resolve_price
         stdapi/pricing.py:_TIER_PRICE_RATIO
         stdapi/pricing.py:_TIER_SCALED_DIMENSIONS
    """

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
        model_id, override_key = "openai.gpt-oss-120b-1:0", "gptoss120b"
        assert pricing._MODEL_KEY_OVERRIDES[model_id] == override_key  # noqa: SLF001
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
    """Long-context resolve_price behavior: exact match, fallback, and tier-ratio interaction.

    Ref: stdapi/pricing.py:resolve_price
         stdapi/pricing.py:_price_context
    """

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
    """On-demand price-catalog refresh triggered by newly discovered Bedrock models.

    Ref: stdapi/pricing.py:refresh_price_catalog_for_new_models
         stdapi/pricing.py:is_model_priced
    """

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
        override_model_id, override_key = "openai.gpt-oss-120b-1:0", "gptoss120b"
        assert pricing._MODEL_KEY_OVERRIDES[override_model_id] == override_key  # noqa: SLF001
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
        """Must never reload the catalog when cost tracking is disabled.

        The model ID is deliberately unpriced, so only the ``cost_tracking``
        guard can suppress the reload.
        """
        monkeypatch.setattr(SETTINGS, "cost_tracking", False)
        calls = 0

        async def _counting_load(_diagnostics: list[str]) -> None:
            nonlocal calls
            calls += 1

        monkeypatch.setattr(pricing, "_load_price_catalog", _counting_load)
        assert is_model_priced("amazon.some-model-v1:0") is False
        await refresh_price_catalog_for_new_models(["amazon.some-model-v1:0"])
        assert calls == 0, "_load_price_catalog ran with cost tracking disabled"

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
        calls = 0

        async def _counting_load(_diagnostics: list[str]) -> None:
            nonlocal calls
            calls += 1

        monkeypatch.setattr(pricing, "_load_price_catalog", _counting_load)
        assert is_model_priced("amazon.already-priced-model-v1:0") is True
        await refresh_price_catalog_for_new_models(["amazon.already-priced-model-v1:0"])
        assert calls == 0, "_load_price_catalog ran for an already-priced model"

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

    async def test_no_reload_for_a_permanently_unpriced_model_within_the_cooldown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model a completed reload still couldn't price must not retrigger one right away.

        Ref: stdapi/pricing.py:_UNPRICED_MODEL_COOLDOWN_NS
        """
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        calls = 0

        async def _fake_load(_diagnostics: list[str]) -> None:
            nonlocal calls
            calls += 1  # Never actually prices the model -- simulates a dead model ID.

        monkeypatch.setattr(pricing, "_load_price_catalog", _fake_load)
        await refresh_price_catalog_for_new_models(["amazon.dead-model-v1:0"])
        assert calls == 1

        # Same permanently-unpriced model reappears -- must not reload again.
        await refresh_price_catalog_for_new_models(["amazon.dead-model-v1:0"])
        assert calls == 1

        # A genuinely new, never-seen model must still trigger a reload.
        await refresh_price_catalog_for_new_models(["amazon.another-new-model-v1:0"])
        assert calls == 2

    async def test_no_reload_when_no_model_ids_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty model_ids iterable must never trigger a reload."""
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        calls = 0

        async def _counting_load(_diagnostics: list[str]) -> None:
            nonlocal calls
            calls += 1

        monkeypatch.setattr(pricing, "_load_price_catalog", _counting_load)
        await refresh_price_catalog_for_new_models([])
        assert calls == 0, "_load_price_catalog ran with no model IDs"


class TestStartPriceCatalogBackgroundLoad:
    """start/stop_price_catalog: background load task lifecycle and logging.

    Ref: stdapi/pricing.py:start_price_catalog
         stdapi/pricing.py:stop_price_catalog
         stdapi/pricing.py:_load_price_catalog_with_retry
    """

    @pytest.fixture(autouse=True)
    async def _stop_task_after_test(self) -> AsyncIterator[None]:
        """Always stop a leftover background load task after each test."""
        yield
        await pricing.stop_price_catalog()

    @pytest.fixture
    def events(self, monkeypatch: pytest.MonkeyPatch) -> list[EventLog]:
        """Capture the background events written by the load task."""
        from stdapi import monitoring  # noqa: PLC0415

        written: list[EventLog] = []
        # Works only because _log_price_catalog_event imports write_log_event at
        # call time (circular-import workaround); an import move fails loudly here.
        monkeypatch.setattr(monitoring, "write_log_event", written.append)
        return written

    def test_disabled_cost_tracking_spawns_no_task(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No task must be spawned when cost tracking is disabled."""
        monkeypatch.setattr(SETTINGS, "cost_tracking", False)
        pricing.start_price_catalog()
        assert pricing._state.load_task is None  # noqa: SLF001

    async def test_successful_load_logs_an_info_background_event(
        self, monkeypatch: pytest.MonkeyPatch, events: list[EventLog]
    ) -> None:
        """A clean load completes the task and logs one info background event.

        Ref: stdapi/pricing.py:_log_price_catalog_event
        """
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)

        async def _load(_diagnostics: list[str]) -> None:
            return

        monkeypatch.setattr(pricing, "_load_price_catalog", _load)
        pricing.start_price_catalog()
        task = pricing._state.load_task  # noqa: SLF001
        assert task is not None
        await task

        (event,) = events
        assert event["type"] == "background"
        assert event["event"] == "price_catalog_load"
        assert event["level"] == "info"
        assert "error_detail" not in event
        assert isinstance(event["execution_time_ms"], int)

    async def test_diagnostics_downgrade_the_event_to_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, events: list[EventLog]
    ) -> None:
        """Load diagnostics surface as the event's error detail at warning level."""
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)

        async def _load(diagnostics: list[str]) -> None:
            diagnostics.append("Price catalog collision on X")

        monkeypatch.setattr(pricing, "_load_price_catalog", _load)
        pricing.start_price_catalog()
        assert pricing._state.load_task is not None  # noqa: SLF001
        await pricing._state.load_task  # noqa: SLF001

        (event,) = events
        assert event["level"] == "warning"
        assert event["error_detail"] == ["Price catalog collision on X"]

    async def test_aws_error_is_retried_until_success(
        self, monkeypatch: pytest.MonkeyPatch, events: list[EventLog]
    ) -> None:
        """An AWS error is logged and retried after a backoff, not fatal."""
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        monkeypatch.setattr(pricing, "_LOAD_RETRY_INITIAL_SECONDS", 0)
        attempts = 0

        async def _load(_diagnostics: list[str]) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                error_response = {"Error": {"Code": "Throttling", "Message": "x"}}
                raise ClientError(error_response, "GetProducts")  # type: ignore[arg-type]

        monkeypatch.setattr(pricing, "_load_price_catalog", _load)
        pricing.start_price_catalog()
        assert pricing._state.load_task is not None  # noqa: SLF001
        await pricing._state.load_task  # noqa: SLF001

        assert attempts == 2
        assert [event["level"] for event in events] == ["warning", "info"]
        assert "retrying in 0s" in str(events[0]["error_detail"])

    async def test_stop_cancels_an_inflight_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """stop_price_catalog cancels the task and clears the state reference."""
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        started = asyncio.Event()

        async def _load(_diagnostics: list[str]) -> None:
            started.set()
            await asyncio.sleep(3600)

        monkeypatch.setattr(pricing, "_load_price_catalog", _load)
        pricing.start_price_catalog()
        task = pricing._state.load_task  # noqa: SLF001
        assert task is not None
        await asyncio.wait_for(started.wait(), timeout=5)

        await pricing.stop_price_catalog()

        assert pricing._state.load_task is None  # noqa: SLF001
        assert task.cancelled()

    async def test_unexpected_error_is_logged_and_retried(
        self, monkeypatch: pytest.MonkeyPatch, events: list[EventLog]
    ) -> None:
        """A non-AWS bug (e.g. AttributeError) is logged at error level and retried."""
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        monkeypatch.setattr(pricing, "_LOAD_RETRY_INITIAL_SECONDS", 0)
        attempts = 0

        async def _load(_diagnostics: list[str]) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                msg = "unexpected price-row shape"
                raise AttributeError(msg)

        monkeypatch.setattr(pricing, "_load_price_catalog", _load)
        pricing.start_price_catalog()
        task = pricing._state.load_task  # noqa: SLF001
        assert task is not None
        await task

        assert attempts == 2
        assert [event["level"] for event in events] == ["error", "info"]
        assert "unexpected price-row shape" in str(events[0]["error_detail"])
        assert "retrying in 0s" in str(events[0]["error_detail"])

    async def test_stop_during_unexpected_error_backoff_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch, events: list[EventLog]
    ) -> None:
        """stop_price_catalog must not raise while backing off from an unexpected error."""
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        monkeypatch.setattr(pricing, "_LOAD_RETRY_INITIAL_SECONDS", 3600)
        attempted = asyncio.Event()

        async def _load(_diagnostics: list[str]) -> None:
            attempted.set()
            msg = "unexpected price-row shape"
            raise AttributeError(msg)

        monkeypatch.setattr(pricing, "_load_price_catalog", _load)
        pricing.start_price_catalog()
        task = pricing._state.load_task  # noqa: SLF001
        assert task is not None
        await asyncio.wait_for(attempted.wait(), timeout=5)

        await pricing.stop_price_catalog()  # must not raise, even mid-backoff

        assert pricing._state.load_task is None  # noqa: SLF001
        assert task.cancelled()
        (event,) = events
        assert event["level"] == "error"

    async def test_second_start_keeps_the_first_task(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A duplicate start_price_catalog call must not replace the running task."""
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        started = asyncio.Event()

        async def _load(_diagnostics: list[str]) -> None:
            started.set()
            await asyncio.sleep(3600)

        monkeypatch.setattr(pricing, "_load_price_catalog", _load)
        pricing.start_price_catalog()
        first = pricing._state.load_task  # noqa: SLF001
        assert first is not None
        await asyncio.wait_for(started.wait(), timeout=5)

        pricing.start_price_catalog()

        assert pricing._state.load_task is first  # noqa: SLF001

    async def test_backoff_delays_double_and_cap_at_the_maximum(
        self, monkeypatch: pytest.MonkeyPatch, events: list[EventLog]
    ) -> None:
        """Retry delays double from the initial value and cap at the maximum.

        Ref: stdapi/pricing.py:_LOAD_RETRY_INITIAL_SECONDS
             stdapi/pricing.py:_LOAD_RETRY_MAX_SECONDS
        """
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        real_sleep = asyncio.sleep  # Captured before the monkeypatch below.
        delays: list[float] = []

        async def _sleep(delay: float) -> None:
            delays.append(delay)
            await real_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", _sleep)
        attempts = 0

        async def _load(_diagnostics: list[str]) -> None:
            nonlocal attempts
            attempts += 1
            if attempts <= 6:
                error_response = {"Error": {"Code": "Throttling", "Message": "x"}}
                raise ClientError(error_response, "GetProducts")  # type: ignore[arg-type]

        monkeypatch.setattr(pricing, "_load_price_catalog", _load)
        pricing.start_price_catalog()
        task = pricing._state.load_task  # noqa: SLF001
        assert task is not None
        await asyncio.wait_for(task, timeout=5)

        assert delays == [60, 120, 240, 480, 900, 900]
        assert attempts == 7
        assert events[-1]["level"] == "info"

    async def test_shutdown_error_event_has_no_execution_time(
        self, events: list[EventLog]
    ) -> None:
        """A task that already failed is logged at shutdown without execution_time_ms.

        Ref: stdapi/pricing.py:_log_price_catalog_event
        """

        async def _boom() -> None:
            msg = "boom"
            raise RuntimeError(msg)

        task = asyncio.get_running_loop().create_task(_boom())
        await asyncio.sleep(0)  # Let the task finish before shutdown handles it.
        assert task.done()
        pricing._state.load_task = task  # noqa: SLF001

        await pricing.stop_price_catalog()  # Must not raise.

        assert pricing._state.load_task is None  # noqa: SLF001
        (event,) = events
        assert event["level"] == "error"
        assert "raised during shutdown" in str(event["error_detail"])
        assert "execution_time_ms" not in event

    async def test_backoff_exits_when_a_refresh_completed_the_catalog(
        self, monkeypatch: pytest.MonkeyPatch, events: list[EventLog]
    ) -> None:
        """The loop must not reload after an on-demand refresh completed the catalog.

        Ref: stdapi/pricing.py:_PriceCatalogState
             stdapi/pricing.py:refresh_price_catalog_for_new_models
        """
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        error_response = {"Error": {"Code": "ThrottlingException", "Message": "x"}}
        client = _FakePricingClient(
            {"AmazonBedrock": [], "AmazonBedrockService": []},
            raise_by_service_code={
                "AmazonBedrock": ClientError(error_response, "GetProducts")  # type: ignore[arg-type]
            },
        )
        _use_fake_catalog(monkeypatch, client)

        real_load = pricing._load_price_catalog  # noqa: SLF001
        load_calls = 0

        async def _counting_load(diagnostics: list[str]) -> None:
            nonlocal load_calls
            load_calls += 1
            await real_load(diagnostics)

        monkeypatch.setattr(pricing, "_load_price_catalog", _counting_load)
        real_sleep = asyncio.sleep  # Captured before the monkeypatch below.

        async def _sleep(_delay: float) -> None:
            # Simulate refresh_price_catalog_for_new_models finishing the
            # pending fetches while the retry loop is backing off.
            pricing._state.catalog_complete = True  # noqa: SLF001
            pricing._state.pending_fetch_specs = None  # noqa: SLF001
            await real_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", _sleep)
        pricing.start_price_catalog()
        task = pricing._state.load_task  # noqa: SLF001
        assert task is not None
        await asyncio.wait_for(task, timeout=5)

        assert load_calls == 1  # The partial-failure load, never re-run.
        assert [event["level"] for event in events] == ["warning", "info"]
        assert "already completed" in str(events[-1]["error_detail"])


@pytest.mark.slow
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

    Ref: stdapi/models/pricing_overrides.py:MODEL_KEY_OVERRIDES
         stdapi/models/pricing_overrides.py:DEFAULT_MODEL_PRICES
         stdapi/pricing.py:_MODEL_KEY_OVERRIDES
    """
    try:
        async with AWSConnectionManager(("sts", None)):
            pass
    except (BotoCoreError, ClientError) as exc:
        pytest.skip(f"AWS is not reachable: {exc}")

    async with AWSConnectionManager(
        *(("bedrock", region) for region in SETTINGS.aws_bedrock_regions),
        *(("bedrock-runtime", region) for region in SETTINGS.aws_bedrock_regions),
        # type-ignore: the RegionName stub Literal lags EUSC/China (works live).
        ("pricing", pricing.pricing_endpoint_region()),  # type: ignore[arg-type]
    ):
        await models.initialize_bedrock_models()
        await pricing._load_price_catalog([])  # noqa: SLF001
        registered = dict(models._MODELS)  # noqa: SLF001
        configured_regions = [str(r) for r in SETTINGS.aws_bedrock_regions]
        # Without these two, an empty catalog or an empty model registry would
        # make the whole coverage check below pass vacuously.
        assert registered, "no Bedrock model was registered -- nothing was checked"
        assert pricing._state.price_index, "the price catalog loaded no rows at all"  # noqa: SLF001

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
            # Mantle-only models are keyed under BEDROCK_MANTLE; a price on
            # either invocation API means the model is priced.
            return [
                region
                for region in check_regions
                if not any(
                    resolve_price(service, model_id, region, dim, tier, "", "", spec)
                    for service in (Service.BEDROCK, Service.BEDROCK_MANTLE)
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
        "matches, check the AWS Bedrock pricing page "
        "(https://aws.amazon.com/bedrock/pricing/): when it publishes a rate "
        "the Price List lacks, add a per-single-unit entry to "
        "DEFAULT_MODEL_PRICES in stdapi/models/pricing_overrides.py (divide "
        "the page's per-1M rate by 1e6). If the page has no rate either, AWS "
        "hasn't published pricing yet (or withdrew it): either wait, or -- "
        "once confirmed upstream -- add the model ID to _KNOWN_PRICING_GAPS "
        "in this file. Never remove a model's implementation because its "
        "pricing disappeared: keep it in case pricing returns or users "
        "retain model access."
    )
    pytest.fail("\n".join(lines))


#: Live-confirmed upstream pricing gaps (2026-07), excluded from the coverage check.
# SDXL: AWS publishes no price rows at all, and its pricing page only says
# legacy Stability models are priced per step count and resolution, with no
# figures (the Image Services page rates ship as DEFAULT_MODEL_PRICES). Re-check
# by removing the entry. Never remove a model's implementation for a pricing
# gap: keep it in case pricing returns or users retain model access.
_KNOWN_PRICING_GAPS: Final[frozenset[str]] = frozenset(
    {
        "stability.stable-diffusion-xl-v1",
        # No Price List rows and no pricing-page rate (only GLM 4.7/5 listed).
        "zai.glm-4.6",
    }
)


#: Keys this app bills that no Price List row names: (service, model, dimension).
_SYNTHETIC_MODEL_PROBES: Final[tuple[tuple[Service, str, Dimension], ...]] = (
    # Every guardrail policy AWS charges for; the word and free
    # sensitive-information policies publish $0 rows, which are not ingested.
    (Service.BEDROCK, guardrail_policy_model("content"), Dimension.TEXT_UNITS),
    (Service.BEDROCK, guardrail_policy_model("content"), Dimension.INPUT_IMAGES),
    (Service.BEDROCK, guardrail_policy_model("topic"), Dimension.TEXT_UNITS),
    (
        Service.BEDROCK,
        guardrail_policy_model("sensitive-information"),
        Dimension.TEXT_UNITS,
    ),
    (
        Service.BEDROCK,
        guardrail_policy_model("contextual-grounding"),
        Dimension.TEXT_UNITS,
    ),
    (
        Service.BEDROCK,
        guardrail_policy_model("automated-reasoning"),
        Dimension.TEXT_UNITS,
    ),
    (Service.BEDROCK, GUARDRAIL_CHECKS_MODERATION_MODEL, Dimension.TEXT_UNITS),
    # The built-in web search tool's one flat per-query rate.
    (Service.BEDROCK, WEB_SEARCH_MODEL, Dimension.GROUNDING_REQUESTS),
    # A managed knowledge base's per-retrieval rate, which reaches the loaded
    # catalog as a built-in default: the Price List API publishes no row for it.
    (Service.BEDROCK, KNOWLEDGE_BASE_MODEL, Dimension.SEARCH_UNITS),
    (Service.POLLY, "amazon.polly-standard", Dimension.INPUT_CHARACTERS),
    (Service.POLLY, "amazon.polly-neural", Dimension.INPUT_CHARACTERS),
    (Service.POLLY, "amazon.polly-long-form", Dimension.INPUT_CHARACTERS),
    (Service.POLLY, "amazon.polly-generative", Dimension.INPUT_CHARACTERS),
    (Service.TRANSLATE, "amazon.translate", Dimension.INPUT_CHARACTERS),
    (Service.TRANSCRIBE, "amazon.transcribe", Dimension.INPUT_SECONDS),
    (
        Service.COMPREHEND,
        "amazon.comprehend-language-detection",
        Dimension.COMPREHEND_UNITS,
    ),
    (Service.COMPREHEND, "amazon.comprehend-toxicity", Dimension.COMPREHEND_UNITS),
)


@pytest.mark.slow
async def test_live_price_catalog_ingests_cleanly() -> None:
    """The live catalog must load without diagnostics and price every synthetic key.

    Both halves cover what fixtures structurally cannot: a hand-written item
    can only assert on a usagetype someone thought to write down, and AWS
    names them nothing like one would guess. Diagnostics catch an unmodeled
    pricing axis folding distinct rates onto one PriceKey -- at runtime only
    a startup warning, but it prices a model by whichever row AWS returned
    last. The probes catch the reverse: a key ``usage.record_*_usage()``
    bills against that no live row lands on, which costs that usage at
    nothing, silently.

    Ref: stdapi/pricing.py:_store_price
         stdapi/pricing.py:_synthesize_service_model_key
         stdapi/usage.py:_BEST_EFFORT_PRICED_DIMENSIONS
    """
    try:
        async with AWSConnectionManager(("sts", None)):
            pass
    except (BotoCoreError, ClientError) as exc:
        pytest.skip(f"AWS is not reachable: {exc}")

    diagnostics: list[str] = []
    async with AWSConnectionManager(
        # type-ignore: the RegionName stub Literal lags EUSC/China (works live).
        ("pricing", pricing.pricing_endpoint_region())  # type: ignore[arg-type]
    ):
        await pricing._load_price_catalog(diagnostics)  # noqa: SLF001

    index = pricing._state.price_index  # noqa: SLF001
    assert index, "the price catalog loaded no rows at all"
    assert diagnostics == []

    # Any region the key is published in proves the reconstruction: which
    # regions carry a service's rows depends on where that service is
    # configured, not on whether its key is built correctly.
    regions = {key.region for key in index}
    unpriced = [
        f"{model} {dimension}"
        for service, model, dimension in _SYNTHETIC_MODEL_PROBES
        if not any(
            resolve_price(service, model, region, dimension) for region in regions
        )
    ]
    assert not unpriced, f"synthetic keys with no live price: {unpriced}"


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
    """model_prices(): filtered, sorted read of one model's price rows.

    Ref: stdapi/pricing.py:model_prices
         stdapi/pricing.py:_row_sort_key
    """

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
        # Swap (don't mutate): model_prices caches a per-index grouping by identity.
        monkeypatch.setattr(
            pricing._state,  # noqa: SLF001
            "price_index",
            {**pricing._state.price_index, **rows},  # noqa: SLF001
        )
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
        """An aliased/overridden model ID reads the same rows.

        The alias shares no normalized key with "cardmodel", so identical rows
        can only come from the override registry.

        Ref: stdapi/pricing.py:resolve_model_key
        """
        self._seed(monkeypatch)
        assert normalize_model_key("vendor.other-alias-v9:9") != "cardmodel"
        pricing.register_model_key_overrides({"vendor.other-alias-v9:9": "cardmodel"})
        try:
            aliased = pricing.model_prices("vendor.other-alias-v9:9")
            assert len(aliased) == 7
            assert all(key.model == "cardmodel" for key, _ in aliased)
            assert aliased == pricing.model_prices("amazon.cardmodel-v1:0")
        finally:
            pricing._MODEL_KEY_OVERRIDES.pop("vendor.other-alias-v9:9", None)  # noqa: SLF001

    def test_unknown_model_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A model with no indexed rows yields an empty list, not an error."""
        self._seed(monkeypatch)
        assert pricing.model_prices("vendor.unknown-v1:0") == []


class TestModelPricesServiceDedupe:
    """Rows registered under both Bedrock services collapse to one.

    Ref: stdapi/pricing.py:_dedupe_service_rows
    """

    @staticmethod
    def _seed_dual(monkeypatch: pytest.MonkeyPatch) -> None:
        """Seed one identical price under both Bedrock services."""
        rows = {
            PriceKey(
                service, "dualmodel", "us-east-1", Dimension.INPUT_TOKENS, "standard"
            ): Price(Decimal("0.000002"), "USD")
            for service in (Service.BEDROCK, Service.BEDROCK_MANTLE)
        }
        # Swap (don't mutate): model_prices caches a per-index grouping by identity.
        monkeypatch.setattr(
            pricing._state,  # noqa: SLF001
            "price_index",
            {**pricing._state.price_index, **rows},  # noqa: SLF001
        )

    def test_dual_service_rows_collapse_to_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A price registered under both services yields a single row."""
        self._seed_dual(monkeypatch)
        (row,) = pricing.model_prices("dualmodel")
        assert row[0].service is Service.BEDROCK

    def test_preferred_service_row_is_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """preferred_service selects which duplicate survives."""
        self._seed_dual(monkeypatch)
        (row,) = pricing.model_prices(
            "dualmodel", preferred_service=Service.BEDROCK_MANTLE
        )
        assert row[0].service is Service.BEDROCK_MANTLE

    def test_single_service_row_kept_regardless_of_preference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A row published under one service only is never dropped."""
        set_test_price("solomodel", "us-east-1", Dimension.INPUT_TOKENS, "0.01", "USD")
        (row,) = pricing.model_prices(
            "solomodel", preferred_service=Service.BEDROCK_MANTLE
        )
        assert row[0].service is Service.BEDROCK


class TestModelPricesGroupingCache:
    """The per-model row grouping tracks price-index swaps.

    Ref: stdapi/pricing.py:_rows_by_model
    """

    def test_prices_seeded_after_a_read_are_visible(self) -> None:
        """A price added (index swap) after a first read appears in the next.

        The first read populates the grouping cache; a stale cache would keep
        returning only the input-token row.
        """
        set_test_price("cachedmodel", "us-east-1", Dimension.INPUT_TOKENS, "1", "USD")
        ((first_key, first_price),) = pricing.model_prices("cachedmodel")
        assert first_key.dimension is Dimension.INPUT_TOKENS
        assert first_price.amount == Decimal(1)

        set_test_price("cachedmodel", "us-east-1", Dimension.OUTPUT_TOKENS, "2", "USD")
        rows = pricing.model_prices("cachedmodel")
        assert {(key.dimension, price.amount) for key, price in rows} == {
            (Dimension.INPUT_TOKENS, Decimal(1)),
            (Dimension.OUTPUT_TOKENS, Decimal(2)),
        }
