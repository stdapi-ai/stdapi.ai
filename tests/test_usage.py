"""Unit tests for stdapi.usage's cost computation, focused on multi-currency handling.

Regression: a record's dimensions can resolve to more than one currency (e.g.
regional price fallback pulling from a differently-partitioned region). Costs
must never be summed across currencies -- records always surface a full
per-currency breakdown instead of collapsing to a single cost/currency pair.
"""

from decimal import Decimal
from json import dumps, loads
from typing import TYPE_CHECKING, Any

import pytest

from stdapi import usage
from stdapi.config import SETTINGS
from stdapi.pricing import Dimension, Service
from stdapi.usage import (
    IMAGE_SPEC,
    UsageKey,
    UsageRecord,
    compute_costs,
    emit_usage_metrics,
    get_model_state,
    record_bedrock_usage,
    record_comprehend_usage,
    record_polly_usage,
    record_transcribe_usage,
    record_translate_usage,
)
from tests.conftest import set_test_price

if TYPE_CHECKING:
    from collections.abc import Generator


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


@pytest.fixture(autouse=True)
def _reset_context_vars() -> Generator[None]:
    """Reset request-scoped context vars so tests don't leak state via execution order."""
    usage_token = usage.init_usage()
    model_state_token = usage.init_model_state()
    image_spec_token = IMAGE_SPEC.set("")
    yield
    usage.USAGE.reset(usage_token)
    usage.MODEL_STATE.reset(model_state_token)
    IMAGE_SPEC.reset(image_spec_token)


class TestComputeCostsMultiCurrency:
    """A record whose dimensions resolve to more than one currency."""

    def _record_mixed_currency_usage(self) -> None:
        usage.init_usage()
        get_model_state("mixedmodel").region = "us-east-1"
        set_test_price(
            "mixedmodel", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD"
        )
        set_test_price(
            "mixedmodel", "us-east-1", Dimension.OUTPUT_TOKENS, "0.000015", "EUR"
        )
        record_bedrock_usage("mixedmodel", input_tokens=1000, output_tokens=1000)

    def test_mixed_currency_surfaces_full_per_currency_breakdown(self) -> None:
        """A record spanning multiple currencies: costs dict has both, cost/currency left unset."""
        self._record_mixed_currency_usage()
        compute_costs()
        record = next(iter(usage.USAGE.get().values()))
        assert record.currency == ""
        assert record.cost == Decimal(0)
        assert record.costs == {"USD": Decimal("0.003000"), "EUR": Decimal("0.015000")}

    def test_mixed_currency_log_entry_has_costs_not_cost(self) -> None:
        """usage_log_entries() must expose `costs`, not a misleading single `cost`."""
        self._record_mixed_currency_usage()
        compute_costs()
        entry = next(iter(usage.usage_log_entries()))
        assert "cost" not in entry
        assert "currency" not in entry
        assert entry["costs"] == {"USD": "0.003", "EUR": "0.015"}

    def test_multi_currency_return_value_warns_naming_both_currencies(self) -> None:
        """compute_costs()'s return value must surface a warning naming both currencies."""
        self._record_mixed_currency_usage()
        warnings = compute_costs()
        assert warnings == [
            "Multiple currencies resolved for bedrock-runtime/mixedmodel "
            "in us-east-1: ['EUR', 'USD']"
        ]

    def test_single_currency_still_uses_cost_and_currency(self) -> None:
        """A normal, single-currency record must still use cost/currency, never costs."""
        usage.init_usage()
        get_model_state("onemodel").region = "us-east-1"
        set_test_price(
            "onemodel", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD"
        )
        set_test_price(
            "onemodel", "us-east-1", Dimension.OUTPUT_TOKENS, "0.000015", "USD"
        )
        record_bedrock_usage("onemodel", input_tokens=1000, output_tokens=1000)
        compute_costs()
        record = next(iter(usage.USAGE.get().values()))
        assert record.currency == "USD"
        assert record.cost == Decimal("0.018000")
        assert record.costs == {}


class TestRoutingTierPricing:
    """Model-state routing -> UsageRecord.routing -> resolve_price plumbing."""

    def _set_routed_prices(self) -> None:
        set_test_price(
            "routedmodel", "us-east-1", Dimension.INPUT_TOKENS, "0.0033", "USD"
        )
        set_test_price(
            "routedmodel",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "0.003",
            "USD",
            routing="global",
        )

    def test_global_routing_used_prices_at_global_rate(self) -> None:
        """A global-routed call must use the cheaper global price."""
        usage.init_usage()
        state = get_model_state("routedmodel")
        state.region = "us-east-1"
        state.routing = "global"
        self._set_routed_prices()
        record_bedrock_usage("routedmodel", input_tokens=1000)
        record = next(iter(usage.USAGE.get().values()))
        assert record.routing == "global"
        compute_costs()
        assert record.cost == Decimal("3.000000")

    def test_no_effective_routing_uses_regional_rate(self) -> None:
        """When no routing profile is tracked, cost must use the plain (regional) rate."""
        usage.init_usage()
        get_model_state("routedmodel").region = "us-east-1"
        self._set_routed_prices()
        record_bedrock_usage("routedmodel", input_tokens=1000)
        record = next(iter(usage.USAGE.get().values()))
        assert record.routing == ""
        compute_costs()
        # The plain (regional) rate, not the cheaper global one.
        assert record.cost == Decimal("3.300000")


class TestImageSpecPricing:
    """IMAGE_SPEC -> UsageRecord.output_images_by_spec -> resolve_price plumbing."""

    def test_mixed_spec_images_price_each_bucket_independently(self) -> None:
        """Two calls with different specs in one record must each use their own price."""
        usage.init_usage()
        get_model_state("imagemodel").region = "us-east-1"
        set_test_price(
            "imagemodel",
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
            "0.008",
            "USD",
            spec="512:standard",
        )
        set_test_price(
            "imagemodel",
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
            "0.012",
            "USD",
            spec="1024:premium",
        )

        IMAGE_SPEC.set("512:standard")
        record_bedrock_usage("imagemodel", output_images=2)
        IMAGE_SPEC.set("1024:premium")
        record_bedrock_usage("imagemodel", output_images=3)

        record = next(iter(usage.USAGE.get().values()))
        assert record.output_images_by_spec == {"512:standard": 2, "1024:premium": 3}
        compute_costs()
        # 2 * 0.008 + 3 * 0.012 = 0.016 + 0.036 = 0.052
        assert record.cost == Decimal("0.052000")

    def test_partial_spec_breakdown_prices_the_remainder_at_the_flat_rate(self) -> None:
        """Regression: flat-only portion must not be mispriced as $0 when mixed with spec-bearing.

        Prior bug: _dimension_price_buckets returned ONLY breakdown buckets when
        output_images_by_spec was non-empty, silently pricing flat portion as $0.
        """
        usage.init_usage()
        get_model_state("mixedimagemodel").region = "us-east-1"
        set_test_price(
            "mixedimagemodel",
            "us-east-1",
            Dimension.OUTPUT_IMAGES,
            "0.01",
            "USD",
            spec="1024:standard",
        )
        set_test_price(
            "mixedimagemodel", "us-east-1", Dimension.OUTPUT_IMAGES, "0.0036", "USD"
        )

        IMAGE_SPEC.set("1024:standard")
        record_bedrock_usage(
            "mixedimagemodel", output_images=2
        )  # goes into the breakdown
        IMAGE_SPEC.set("")
        record_bedrock_usage(
            "mixedimagemodel", output_images=3
        )  # flat-only, no breakdown

        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.OUTPUT_IMAGES] == 5
        assert record.output_images_by_spec == {"1024:standard": 2}
        compute_costs()
        # 2 * 0.01 (spec) + 3 * 0.0036 (flat remainder) = 0.02 + 0.0108
        assert record.cost == Decimal("0.030800")

    def test_no_spec_falls_back_to_flat_price(self) -> None:
        """A model with no IMAGE_SPEC set must still price via the flat per-image rate."""
        usage.init_usage()
        get_model_state("flatimagemodel").region = "us-east-1"
        set_test_price(
            "flatimagemodel", "us-east-1", Dimension.OUTPUT_IMAGES, "0.0036", "USD"
        )
        record_bedrock_usage("flatimagemodel", output_images=2)
        record = next(iter(usage.USAGE.get().values()))
        assert record.output_images_by_spec == {}
        compute_costs()
        assert record.cost == Decimal("0.007200")

    def test_resetting_image_spec_prevents_leakage_to_a_later_flat_priced_call(
        self,
    ) -> None:
        """Regression: stale IMAGE_SPEC must not leak from one model to a later unrelated model.

        Only Titan Image Generator sets IMAGE_SPEC; nothing cleared it afterwards.
        """
        usage.init_usage()
        get_model_state("titanimagegeneratorv2").region = "us-east-1"
        get_model_state("flatimagemodel").region = "us-east-1"
        set_test_price(
            "flatimagemodel", "us-east-1", Dimension.OUTPUT_IMAGES, "0.0036", "USD"
        )

        IMAGE_SPEC.set("1024:premium")
        record_bedrock_usage("titanimagegeneratorv2", output_images=1)
        IMAGE_SPEC.set(
            ""
        )  # What ImageModelBase._record_invoke_usage now does after every call.

        record_bedrock_usage("flatimagemodel", output_images=2)
        flat_record = usage.USAGE.get()[
            next(k for k in usage.USAGE.get() if k.model == "flatimagemodel")
        ]
        assert flat_record.output_images_by_spec == {}
        compute_costs()
        assert flat_record.cost == Decimal("0.007200")


class TestOutputSecondsSpecPricing:
    """output_seconds_spec -> UsageRecord.output_seconds_by_spec -> resolve_price plumbing."""

    def test_mixed_spec_seconds_price_each_bucket_independently(self) -> None:
        """A flat call and an "hd"-spec call in one record must each use their own price."""
        usage.init_usage()
        get_model_state("videomodel").region = "us-east-1"
        set_test_price(
            "videomodel", "us-east-1", Dimension.OUTPUT_SECONDS, "0.06", "USD"
        )
        set_test_price(
            "videomodel",
            "us-east-1",
            Dimension.OUTPUT_SECONDS,
            "0.08",
            "USD",
            spec="hd",
        )

        record_bedrock_usage("videomodel", output_seconds=5)
        record_bedrock_usage("videomodel", output_seconds=5, output_seconds_spec="hd")

        record = next(iter(usage.USAGE.get().values()))
        assert record.output_seconds_by_spec == {"hd": 5}
        assert record.quantities[Dimension.OUTPUT_SECONDS] == 10
        compute_costs()
        # 5 * 0.06 (flat remainder) + 5 * 0.08 (hd) = 0.3 + 0.4 = 0.7
        assert record.cost == Decimal("0.700000")

    def test_no_spec_falls_back_to_flat_price(self) -> None:
        """A model with no spec bucket must still price via the flat per-second rate."""
        usage.init_usage()
        get_model_state("flatvideomodel").region = "us-east-1"
        set_test_price(
            "flatvideomodel", "us-east-1", Dimension.OUTPUT_SECONDS, "0.05", "USD"
        )
        record_bedrock_usage("flatvideomodel", output_seconds=6)
        record = next(iter(usage.USAGE.get().values()))
        assert record.output_seconds_by_spec == {}
        compute_costs()
        assert record.cost == Decimal("0.300000")

    def test_usage_log_entry_reports_output_seconds_by_spec(self) -> None:
        """usage_log_entries() must surface the accumulated output_seconds_by_spec breakdown."""
        usage.init_usage()
        get_model_state("videomodel").region = "us-east-1"
        record_bedrock_usage("videomodel", output_seconds=5, output_seconds_spec="hd")
        entry = next(iter(usage.usage_log_entries()))
        assert entry["output_seconds_by_spec"] == {"hd": 5}


class TestCacheTtlPricing:
    """cache_write_tokens_by_ttl -> UsageRecord -> resolve_price plumbing.

    Regression: mirrors TestImageSpecPricing scenarios for cache-write tokens;
    previously untested end-to-end.
    """

    def test_mixed_ttl_cache_writes_price_each_bucket_independently(self) -> None:
        """Two TTL buckets in one record must each use their own price."""
        usage.init_usage()
        get_model_state("cachemodel").region = "us-east-1"
        set_test_price(
            "cachemodel",
            "us-east-1",
            Dimension.CACHE_WRITE_TOKENS,
            "0.000004",
            "USD",
            cache_ttl="5m",
        )
        set_test_price(
            "cachemodel",
            "us-east-1",
            Dimension.CACHE_WRITE_TOKENS,
            "0.000008",
            "USD",
            cache_ttl="1h",
        )

        record_bedrock_usage(
            "cachemodel",
            cache_write_tokens=1500,
            cache_write_tokens_by_ttl={"5m": 500, "1h": 1000},
        )
        record = next(iter(usage.USAGE.get().values()))
        assert record.cache_write_tokens_by_ttl == {"5m": 500, "1h": 1000}
        compute_costs()
        # 500 * 0.000004 + 1000 * 0.000008 = 0.002 + 0.008 = 0.010
        assert record.cost == Decimal("0.010000")

    def test_partial_ttl_breakdown_prices_the_remainder_at_the_flat_rate(self) -> None:
        """A flat-only call mixed with a TTL-bearing call must not lose the flat portion's cost."""
        usage.init_usage()
        get_model_state("mixedcachemodel").region = "us-east-1"
        set_test_price(
            "mixedcachemodel",
            "us-east-1",
            Dimension.CACHE_WRITE_TOKENS,
            "0.000008",
            "USD",
            cache_ttl="1h",
        )
        set_test_price(
            "mixedcachemodel",
            "us-east-1",
            Dimension.CACHE_WRITE_TOKENS,
            "0.000004",
            "USD",
        )

        record_bedrock_usage(
            "mixedcachemodel",
            cache_write_tokens=1000,
            cache_write_tokens_by_ttl={"1h": 400},
        )
        record_bedrock_usage(
            "mixedcachemodel", cache_write_tokens=600
        )  # flat-only, no breakdown

        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.CACHE_WRITE_TOKENS] == 1600
        assert record.cache_write_tokens_by_ttl == {"1h": 400}
        compute_costs()
        # 400 * 0.000008 (1h) + 1200 * 0.000004 (flat remainder) = 0.0032 + 0.0048
        assert record.cost == Decimal("0.008000")

    def test_ttl_breakdown_without_a_flat_total_is_still_priced(self) -> None:
        """A by-TTL breakdown with no matching flat quantity must still be priced."""
        usage.init_usage()
        get_model_state("noflatcachemodel").region = "us-east-1"
        set_test_price(
            "noflatcachemodel",
            "us-east-1",
            Dimension.CACHE_WRITE_TOKENS,
            "0.000004",
            "USD",
        )

        # No flat cache_write_tokens passed at all.
        record_bedrock_usage("noflatcachemodel", cache_write_tokens_by_ttl={"5m": 500})

        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.CACHE_WRITE_TOKENS] == 500
        assert record.cache_write_tokens_by_ttl == {"5m": 500}
        compute_costs()
        # 500 * 0.000004
        assert record.cost == Decimal("0.002000")

    @pytest.mark.parametrize(
        "flat_call_first", [True, False], ids=["flat-first", "breakdown-first"]
    )
    def test_flat_only_and_breakdown_only_calls_merge_order_independently(
        self, flat_call_first: bool
    ) -> None:
        """A flat-only call and a breakdown-only call must merge to the same total in either order."""
        usage.init_usage()
        get_model_state("ordercachemodel").region = "us-east-1"
        set_test_price(
            "ordercachemodel",
            "us-east-1",
            Dimension.CACHE_WRITE_TOKENS,
            "0.000004",
            "USD",
        )

        def _flat_only_call() -> None:
            record_bedrock_usage("ordercachemodel", cache_write_tokens=100)

        def _breakdown_only_call() -> None:
            record_bedrock_usage(
                "ordercachemodel", cache_write_tokens_by_ttl={"5m": 150}
            )

        calls = (_flat_only_call, _breakdown_only_call)
        for call in calls if flat_call_first else reversed(calls):
            call()

        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.CACHE_WRITE_TOKENS] == 250
        assert record.cache_write_tokens_by_ttl == {"5m": 150}
        compute_costs()
        # (150 in "5m" + 100 undifferentiated) * 0.000004
        assert record.cost == Decimal("0.001000")


class TestRecordBedrockUsageTierResolution:
    """record_bedrock_usage's tier precedence: explicit arg > this model's invocation state."""

    def test_explicit_tier_overrides_context_var(self) -> None:
        """An explicit `tier=` argument must win over a conflicting model-state tier."""
        usage.init_usage()
        state = get_model_state("tiermodel")
        state.region = "us-east-1"
        state.service_tier = "priority"
        record_bedrock_usage("tiermodel", tier="flex", input_tokens=100)
        record = next(iter(usage.USAGE.get().values()))
        assert record.tier == "flex"

    def test_falls_back_to_service_tier_context_var_when_not_given(self) -> None:
        """With no explicit tier, the record must pick up the model-state tier (set by models/__init__.py)."""
        usage.init_usage()
        state = get_model_state("tiermodel")
        state.region = "us-east-1"
        state.service_tier = "priority"
        record_bedrock_usage("tiermodel", input_tokens=100)
        record = next(iter(usage.USAGE.get().values()))
        assert record.tier == "priority"

    def test_falls_back_to_standard_when_neither_is_set(self) -> None:
        """With no explicit tier and a default (never-overridden) model-state tier, default to standard."""
        usage.init_usage()
        get_model_state("tiermodel").region = "us-east-1"
        record_bedrock_usage("tiermodel", input_tokens=100)
        record = next(iter(usage.USAGE.get().values()))
        assert record.tier == "standard"


class TestInitUsageTokenReset:
    """init_usage() must return a token that restores the previous USAGE dict.

    Regression: nested calls permanently replaced the outer dict because
    init_usage() didn't capture a reset token.
    """

    def test_nested_init_usage_does_not_clobber_the_outer_dict(self) -> None:
        """A reset token must restore the exact prior dict object, not a copy."""
        outer_token = usage.init_usage()
        outer_records = usage.USAGE.get()
        inner_token = usage.init_usage()
        assert usage.USAGE.get() is not outer_records

        usage.USAGE.reset(inner_token)
        assert usage.USAGE.get() is outer_records

        usage.USAGE.reset(outer_token)


class TestComputeCostsUnpricedDimension:
    """A record with a dimension for which no price could be resolved."""

    def test_fully_unpriced_model_returns_a_warning_naming_the_dimension(self) -> None:
        """A model with no price entry at all: warning names it, cost stays zero."""
        usage.init_usage()
        # Seed an unrelated price so the catalog counts as ready -- the point
        # of this test is a genuine per-model miss, not an unloaded catalog.
        set_test_price(
            "othermodel", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD"
        )
        get_model_state("nopricemodel").region = "us-east-1"
        record_bedrock_usage("nopricemodel", input_tokens=1000)
        warnings = compute_costs()
        assert warnings == [
            "No price found for bedrock-runtime/nopricemodel in "
            "us-east-1: ['input_tokens']"
        ]
        record = next(iter(usage.USAGE.get().values()))
        assert record.cost == Decimal(0)
        assert record.currency == ""

    def test_partially_priced_model_still_prices_the_resolvable_dimension(self) -> None:
        """One priced and one unpriced dimension: cost reflects only the priced one."""
        usage.init_usage()
        get_model_state("partialpricemodel").region = "us-east-1"
        set_test_price(
            "partialpricemodel", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD"
        )
        record_bedrock_usage("partialpricemodel", input_tokens=1000, output_tokens=1000)
        warnings = compute_costs()
        assert warnings == [
            "No price found for bedrock-runtime/partialpricemodel in "
            "us-east-1: ['output_tokens']"
        ]
        record = next(iter(usage.USAGE.get().values()))
        assert record.cost == Decimal("0.003000")
        assert record.currency == "USD"


class TestComputeCostsRegionSkip:
    """Records with no region attributed must never be priced."""

    def test_record_without_region_is_skipped(self) -> None:
        """compute_costs() must leave cost/currency untouched when region is empty."""
        usage.init_usage()
        key = UsageKey(Service.BEDROCK, "modelnoregion", "", "", "standard")
        usage.USAGE.get()[key] = UsageRecord(
            Service.BEDROCK,
            "modelnoregion",
            "",
            region="",
            quantities={Dimension.INPUT_TOKENS: 100},
        )
        compute_costs()
        record = usage.USAGE.get()[key]
        assert record.cost == Decimal(0)
        assert record.currency == ""


class TestEmitUsageMetrics:
    """CloudWatch EMF line emission -- previously had zero test coverage."""

    def test_disabled_setting_emits_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cloudwatch_metrics off must suppress emission entirely."""
        monkeypatch.setattr(SETTINGS, "cloudwatch_metrics", False)
        usage.init_usage()
        get_model_state("emfmodel").region = "us-east-1"
        record_bedrock_usage("emfmodel", input_tokens=1000)
        written: list[dict[str, Any]] = []
        monkeypatch.setattr(usage, "stdout_write", written.append)
        emit_usage_metrics()
        assert written == []

    def test_single_currency_emits_one_line_with_quantities_and_cost(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A priced record emits one EMF line with both quantity metrics and Cost/Currency."""
        monkeypatch.setattr(SETTINGS, "cloudwatch_metrics", True)
        usage.init_usage()
        get_model_state("emfmodel").region = "us-east-1"
        set_test_price(
            "emfmodel", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD"
        )
        record_bedrock_usage("emfmodel", input_tokens=1000)
        compute_costs()

        written: list[dict[str, Any]] = []
        monkeypatch.setattr(usage, "stdout_write", written.append)
        emit_usage_metrics()

        assert len(written) == 1
        payload = written[0]
        assert payload["InputTokens"] == 1000
        assert payload["Cost"] == pytest.approx(0.003)
        assert payload["Currency"] == "USD"
        # Cost and quantities must be scoped to separate directives -- a
        # single directive spanning both ["Model"] and ["Model", "Currency"]
        # would also publish Cost bare-by-Model, summing across currencies on
        # a multi-currency record (see the multi-currency test below).
        quantity_directive, cost_directive = payload["_aws"]["CloudWatchMetrics"]
        assert quantity_directive["Dimensions"] == [["Model"]]
        assert {m["Name"] for m in quantity_directive["Metrics"]} == {"InputTokens"}
        assert cost_directive["Dimensions"] == [["Model", "Currency"]]
        assert {m["Name"] for m in cost_directive["Metrics"]} == {"Cost"}

    def test_no_cost_resolved_emits_quantities_only_with_model_dimension(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No price indexed at all: quantities still emit, but with no Cost/Currency."""
        monkeypatch.setattr(SETTINGS, "cloudwatch_metrics", True)
        usage.init_usage()
        get_model_state("emfmodel").region = "us-east-1"
        record_bedrock_usage("emfmodel", input_tokens=1000)
        compute_costs()

        written: list[dict[str, Any]] = []
        monkeypatch.setattr(usage, "stdout_write", written.append)
        emit_usage_metrics()

        assert len(written) == 1
        payload = written[0]
        assert payload["InputTokens"] == 1000
        assert "Cost" not in payload
        assert "Currency" not in payload
        metrics = payload["_aws"]["CloudWatchMetrics"][0]
        assert metrics["Dimensions"] == [["Model"]]

    def test_multi_currency_emits_one_extra_line_without_duplicating_quantities(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression guard: quantity metrics must only appear on the first per-currency line."""
        monkeypatch.setattr(SETTINGS, "cloudwatch_metrics", True)
        usage.init_usage()
        get_model_state("emfmodel").region = "us-east-1"
        set_test_price(
            "emfmodel", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD"
        )
        set_test_price(
            "emfmodel", "us-east-1", Dimension.OUTPUT_TOKENS, "0.000015", "EUR"
        )
        record_bedrock_usage("emfmodel", input_tokens=1000, output_tokens=1000)
        compute_costs()

        written: list[dict[str, Any]] = []
        monkeypatch.setattr(usage, "stdout_write", written.append)
        emit_usage_metrics()

        assert len(written) == 2
        first, second = written
        assert first["InputTokens"] == 1000
        assert first["OutputTokens"] == 1000
        assert {"Cost", "Currency"} <= first.keys()
        # The second (extra-currency) line must NOT re-report quantities.
        assert "InputTokens" not in second
        assert "OutputTokens" not in second
        assert {"Cost", "Currency"} <= second.keys()
        assert {first["Currency"], second["Currency"]} == {"USD", "EUR"}

    def test_cost_is_a_json_serializable_float(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EMF requires JSON numbers: Cost must be a float and survive json.dumps."""
        monkeypatch.setattr(SETTINGS, "cloudwatch_metrics", True)
        usage.init_usage()
        get_model_state("emfmodel").region = "us-east-1"
        set_test_price(
            "emfmodel", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD"
        )
        record_bedrock_usage("emfmodel", input_tokens=1000)
        compute_costs()

        written: list[dict[str, Any]] = []
        monkeypatch.setattr(usage, "stdout_write", written.append)
        emit_usage_metrics()

        payload = written[0]
        # A float, not a Decimal (which would raise TypeError in dumps below).
        assert isinstance(payload["Cost"], float)
        assert loads(dumps(payload))["Cost"] == pytest.approx(0.003)


class TestLongContextDetection:
    """record_bedrock_usage's context="long" detection (prompt > 200K tokens)."""

    def test_exactly_threshold_is_not_long(self) -> None:
        """A prompt of exactly 200_000 tokens must stay at the standard rate (boundary)."""
        usage.init_usage()
        get_model_state("longmodel").region = "us-east-1"
        record_bedrock_usage("longmodel", input_tokens=200_000)
        record = next(iter(usage.USAGE.get().values()))
        assert record.context == ""

    def test_one_token_over_threshold_via_mixed_dimensions_is_long(self) -> None:
        """Input + cached + cache_write tokens combine to cross the threshold."""
        usage.init_usage()
        get_model_state("longmodel").region = "us-east-1"
        record_bedrock_usage(
            "longmodel",
            input_tokens=100_001,
            cached_tokens=50_000,
            cache_write_tokens=50_000,
        )
        record = next(iter(usage.USAGE.get().values()))
        assert record.context == "long"

    def test_small_and_large_calls_produce_separate_records(self) -> None:
        """A long-context call and a standard call for the same model must not merge."""
        usage.init_usage()
        get_model_state("longmodel").region = "us-east-1"
        record_bedrock_usage("longmodel", input_tokens=1_000)
        record_bedrock_usage("longmodel", input_tokens=200_001)
        records = list(usage.USAGE.get().values())
        assert len(records) == 2
        contexts = {record.context for record in records}
        assert contexts == {"", "long"}

    def test_usage_log_entries_include_context_only_for_the_long_record(self) -> None:
        """usage_log_entries() must surface "context": "long" only on the long-context record."""
        usage.init_usage()
        get_model_state("longmodel").region = "us-east-1"
        record_bedrock_usage("longmodel", input_tokens=1_000)
        record_bedrock_usage("longmodel", input_tokens=200_001)
        entries = usage.usage_log_entries()
        assert len(entries) == 2
        long_entries = [entry for entry in entries if entry.get("context") == "long"]
        standard_entries = [entry for entry in entries if "context" not in entry]
        assert len(long_entries) == 1
        assert len(standard_entries) == 1

    def test_compute_costs_uses_the_long_context_price_for_the_long_record(
        self,
    ) -> None:
        """compute_costs() must pass record.context through to resolve_price."""
        usage.init_usage()
        get_model_state("longmodel").region = "us-east-1"
        set_test_price(
            "longmodel", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD"
        )
        set_test_price(
            "longmodel",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "0.000006",
            "USD",
            context="long",
        )

        record_bedrock_usage("longmodel", input_tokens=1_000)  # standard
        record_bedrock_usage("longmodel", input_tokens=200_001)  # long

        records = {r.context: r for r in usage.USAGE.get().values()}
        compute_costs()
        # Standard record: 1_000 * 0.000003
        assert records[""].cost == Decimal("0.003000")
        # Long record: 200_001 * 0.000006
        assert records["long"].cost == Decimal("1.200006")


class TestGroundingRequests:
    """grounding_requests -> Dimension.GROUNDING_REQUESTS plumbing."""

    def test_zero_or_none_grounding_requests_omit_the_field(self) -> None:
        """No grounding tool calls must not add a grounding_requests entry."""
        usage.init_usage()
        get_model_state("groundedmodel").region = "us-east-1"
        record_bedrock_usage("groundedmodel", input_tokens=100, grounding_requests=0)
        record_bedrock_usage("groundedmodel", input_tokens=100, grounding_requests=None)
        record = next(iter(usage.USAGE.get().values()))
        assert Dimension.GROUNDING_REQUESTS not in record.quantities
        entry = next(iter(usage.usage_log_entries()))
        assert "grounding_requests" not in entry

    def test_usage_log_entries_reports_grounding_requests_count(self) -> None:
        """usage_log_entries() must surface the accumulated grounding_requests count."""
        usage.init_usage()
        get_model_state("groundedmodel").region = "us-east-1"
        record_bedrock_usage("groundedmodel", grounding_requests=2)
        entry = next(iter(usage.usage_log_entries()))
        assert entry["grounding_requests"] == 2

    def test_grounding_requests_are_priced_per_request(self) -> None:
        """Cost is computed from an indexed GROUNDING_REQUESTS price."""
        usage.init_usage()
        get_model_state("groundedmodel").region = "us-east-1"
        set_test_price(
            "groundedmodel", "us-east-1", Dimension.GROUNDING_REQUESTS, "0.03", "USD"
        )
        record_bedrock_usage("groundedmodel", grounding_requests=2)
        record = next(iter(usage.USAGE.get().values()))
        compute_costs()
        assert record.cost == Decimal("0.060000")


class TestSearchUnitsUsage:
    """search_units -> Dimension.SEARCH_UNITS plumbing."""

    def test_search_units_are_recorded_logged_and_priced(self) -> None:
        """search_units must accumulate in quantities, the log entry, and cost."""
        usage.init_usage()
        get_model_state("searchmodel").region = "us-east-1"
        set_test_price(
            "searchmodel", "us-east-1", Dimension.SEARCH_UNITS, "0.001", "USD"
        )
        record_bedrock_usage("searchmodel", search_units=3)
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.SEARCH_UNITS] == 3
        entry = next(iter(usage.usage_log_entries()))
        assert entry["search_units"] == 3
        compute_costs()
        assert record.cost == Decimal("0.003000")


class TestInputMediaSpecUsage:
    """input_images/input_seconds + media_spec -> *_by_spec breakdown plumbing."""

    def test_input_images_with_media_spec_price_and_log_by_spec(self) -> None:
        """input_images with media_spec="document" must price from the "document" spec bucket."""
        usage.init_usage()
        get_model_state("mediaimagemodel").region = "us-east-1"
        set_test_price(
            "mediaimagemodel",
            "us-east-1",
            Dimension.INPUT_IMAGES,
            "0.0008",
            "USD",
            spec="document",
        )
        record_bedrock_usage("mediaimagemodel", input_images=2, media_spec="document")
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.INPUT_IMAGES] == 2
        assert record.input_images_by_spec == {"document": 2}
        entry = next(iter(usage.usage_log_entries()))
        assert entry["input_images_by_spec"] == {"document": 2}
        compute_costs()
        assert record.cost == Decimal("0.001600")

    def test_input_seconds_with_media_spec_price_and_log_by_spec(self) -> None:
        """input_seconds with media_spec="audio" must price from the "audio" spec bucket."""
        usage.init_usage()
        get_model_state("mediaaudiomodel").region = "us-east-1"
        set_test_price(
            "mediaaudiomodel",
            "us-east-1",
            Dimension.INPUT_SECONDS,
            "0.0001",
            "USD",
            spec="audio",
        )
        record_bedrock_usage("mediaaudiomodel", input_seconds=45, media_spec="audio")
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.INPUT_SECONDS] == 45
        assert record.input_seconds_by_spec == {"audio": 45}
        entry = next(iter(usage.usage_log_entries()))
        assert entry["input_seconds_by_spec"] == {"audio": 45}
        compute_costs()
        assert record.cost == Decimal("0.004500")

    def test_mixed_audio_and_video_input_seconds_price_each_bucket_independently(
        self,
    ) -> None:
        """Two calls with different media_spec values must each use their own rate."""
        usage.init_usage()
        get_model_state("mixedaudiovideomodel").region = "us-east-1"
        set_test_price(
            "mixedaudiovideomodel",
            "us-east-1",
            Dimension.INPUT_SECONDS,
            "0.0001",
            "USD",
            spec="audio",
        )
        set_test_price(
            "mixedaudiovideomodel",
            "us-east-1",
            Dimension.INPUT_SECONDS,
            "0.0005",
            "USD",
            spec="video",
        )
        record_bedrock_usage(
            "mixedaudiovideomodel", input_seconds=30, media_spec="audio"
        )
        record_bedrock_usage(
            "mixedaudiovideomodel", input_seconds=10, media_spec="video"
        )
        record = next(iter(usage.USAGE.get().values()))
        assert record.input_seconds_by_spec == {"audio": 30, "video": 10}
        compute_costs()
        # 30 * 0.0001 + 10 * 0.0005 = 0.003 + 0.005 = 0.008
        assert record.cost == Decimal("0.008000")


class TestLatencyRoutingUsage:
    """ModelInvocationState.routing == "latency" -> UsageKey/UsageRecord/log plumbing."""

    def test_latency_routing_state_produces_latency_key_log_and_price(self) -> None:
        """A "latency"-routed call must key/log as "latency" and price from its own rate."""
        usage.init_usage()
        state = get_model_state("latencymodel")
        state.region = "us-east-1"
        state.routing = "latency"
        set_test_price(
            "latencymodel", "us-east-1", Dimension.INPUT_TOKENS, "0.0033", "USD"
        )
        set_test_price(
            "latencymodel",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "0.004",
            "USD",
            routing="latency",
        )
        record_bedrock_usage("latencymodel", input_tokens=1000)
        key, record = next(iter(usage.USAGE.get().items()))
        assert key.routing == "latency"
        assert record.routing == "latency"
        entry = next(iter(usage.usage_log_entries()))
        assert entry["routing"] == "latency"
        compute_costs()
        # The latency rate (0.004), not the plain fallback (0.0033).
        assert record.cost == Decimal("4.000000")

    def test_latency_routing_falls_back_to_the_plain_rate_when_not_indexed(
        self,
    ) -> None:
        """A "latency"-routed call with no latency price indexed must use the plain rate."""
        usage.init_usage()
        state = get_model_state("latencyfallbackmodel")
        state.region = "us-east-1"
        state.routing = "latency"
        set_test_price(
            "latencyfallbackmodel", "us-east-1", Dimension.INPUT_TOKENS, "0.0033", "USD"
        )
        record_bedrock_usage("latencyfallbackmodel", input_tokens=1000)
        record = next(iter(usage.USAGE.get().values()))
        assert record.routing == "latency"
        compute_costs()
        assert record.cost == Decimal("3.300000")


class TestNonBedrockRecordUsageHelpers:
    """record_polly_usage/record_transcribe_usage/record_translate_usage/record_comprehend_usage."""

    def test_record_polly_usage_bills_exact_character_count(self) -> None:
        """Polly bills per character, no minimum."""
        usage.init_usage()
        billed = record_polly_usage(42, "neural")
        assert billed == 42
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.INPUT_CHARACTERS] == 42

    def test_record_translate_usage_bills_exact_character_count(self) -> None:
        """Translate bills per character, no minimum."""
        usage.init_usage()
        billed = record_translate_usage(100)
        assert billed == 100
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.INPUT_CHARACTERS] == 100

    @pytest.mark.parametrize(
        ("duration", "expected"), [(5.0, 15), (15.0, 15), (15.4, 16), (30.0, 30)]
    )
    def test_record_transcribe_usage_applies_15_second_minimum(
        self, duration: float, expected: int
    ) -> None:
        """Transcribe bills per second, rounded up, with a 15-second minimum."""
        usage.init_usage()
        billed = record_transcribe_usage(duration)
        assert billed == expected
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.INPUT_SECONDS] == expected

    @pytest.mark.parametrize(("text_length", "expected"), [(50, 3), (300, 3), (500, 5)])
    def test_record_comprehend_usage_applies_3_unit_minimum(
        self, text_length: int, expected: int
    ) -> None:
        """Comprehend bills in 100-character units, rounded up, with a 3-unit minimum."""
        usage.init_usage()
        billed = record_comprehend_usage(text_length, "language-detection")
        assert billed == expected
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.COMPREHEND_UNITS] == expected

    def test_record_polly_usage_with_zero_characters_records_nothing(self) -> None:
        """Zero characters must not create a usage record."""
        usage.init_usage()
        billed = record_polly_usage(0, "neural")
        assert billed == 0
        assert usage.USAGE.get() == {}

    def test_record_translate_usage_with_zero_characters_records_nothing(self) -> None:
        """Zero characters must not create a usage record."""
        usage.init_usage()
        billed = record_translate_usage(0)
        assert billed == 0
        assert usage.USAGE.get() == {}

    def test_record_transcribe_usage_with_zero_duration_bills_the_minimum(self) -> None:
        """Zero duration must still bill the 15-second minimum, not record nothing."""
        usage.init_usage()
        billed = record_transcribe_usage(0)
        assert billed == 15
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.INPUT_SECONDS] == 15

    def test_record_comprehend_usage_with_zero_length_bills_the_minimum(self) -> None:
        """Zero text length must still bill the 3-unit minimum, not record nothing."""
        usage.init_usage()
        billed = record_comprehend_usage(0, "language-detection")
        assert billed == 3
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.COMPREHEND_UNITS] == 3

    def test_non_bedrock_usage_does_not_inherit_a_prior_bedrock_region(self) -> None:
        """Regression: Polly/Transcribe/Translate/Comprehend must not inherit a Bedrock model's region.

        Old bug: shared-context-var region fallback let non-Bedrock services
        silently inherit a prior Bedrock call's region, mispricing it.
        """
        usage.init_usage()
        get_model_state(
            "priorbedrockmodel"
        ).region = "us-east-1"  # Simulates a prior Bedrock call in this context.
        record_polly_usage(42, "neural")
        record = next(iter(usage.USAGE.get().values()))
        assert record.region == ""


class TestFormatCost:
    """format_cost: exact plain-decimal text with no exponent or trailing zeros."""

    @pytest.mark.parametrize(
        ("amount", "expected"),
        [
            ("0.000001", "0.000001"),  # float JSON rendering would give "1e-6"
            ("0.000015000", "0.000015"),
            ("1E-7", "0.0000001"),
            ("12.500000", "12.5"),
            ("2.000000", "2"),
            ("120", "120"),  # Decimal.normalize alone would give "1.2E+2"
            ("0", "0"),  # A zero amount renders as plain "0"
        ],
    )
    def test_plain_decimal_rendering(self, amount: str, expected: str) -> None:
        """Every amount renders as plain decimal text."""
        assert usage.format_cost(Decimal(amount)) == expected
