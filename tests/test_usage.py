"""Usage recording, cost computation, log entries and EMF emission in stdapi.usage.

Usage aggregates into one record per (service, model, operation, region, tier,
routing, context) key, then ``compute_costs()`` resolves a price per dimension
and per TTL/spec bucket. Costs are never summed across currencies: a record
whose dimensions resolve to more than one currency (e.g. regional price
fallback pulling from a differently-partitioned Region) surfaces a full
per-currency breakdown instead of a single cost/currency pair.

Ref: stdapi/usage.py:record_bedrock_usage
     stdapi/usage.py:compute_costs
     stdapi/usage.py:usage_log_entries
     stdapi/pricing.py:resolve_price
"""

import asyncio
from decimal import Decimal
from json import dumps, loads
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from stdapi import usage
from stdapi.config import SETTINGS, _Settings
from stdapi.pricing import (
    Dimension,
    Service,
    guardrail_policy_model,
    normalize_model_key,
)
from stdapi.usage import (
    IMAGE_SPEC,
    UsageKey,
    UsageRecord,
    compute_costs,
    emit_usage_metrics,
    get_model_state,
    record_bedrock_usage,
    record_comprehend_usage,
    record_guardrail_policy_usage,
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
    """A record whose dimensions resolve to more than one currency.

    Ref: stdapi/usage.py:_apply_record_cost
    """

    def _record_mixed_currency_usage(self) -> None:
        get_model_state("mixedmodel").region = "us-east-1"
        set_test_price(
            "mixedmodel", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD"
        )
        set_test_price(
            "mixedmodel", "us-east-1", Dimension.OUTPUT_TOKENS, "0.000015", "EUR"
        )
        record_bedrock_usage("mixedmodel", input_tokens=1000, output_tokens=1000)

    def test_mixed_currency_surfaces_full_per_currency_breakdown(self) -> None:
        """A record spanning two currencies fills ``costs`` and leaves cost/currency unset.

        1000 input tokens at 0.000003 USD and 1000 output tokens at 0.000015
        EUR: each currency keeps its own subtotal, quantized to 6 decimals.
        """
        self._record_mixed_currency_usage()
        compute_costs()
        record = next(iter(usage.USAGE.get().values()))
        assert record.currency == ""
        assert record.cost == Decimal(0)
        assert record.costs == {"USD": Decimal("0.003000"), "EUR": Decimal("0.015000")}

    def test_mixed_currency_log_entry_has_costs_not_cost(self) -> None:
        """A multi-currency log entry carries ``costs`` and neither ``cost`` nor ``currency``.

        Ref: stdapi/usage.py:_add_cost_fields
        """
        self._record_mixed_currency_usage()
        compute_costs()
        entry = next(iter(usage.usage_log_entries()))
        assert "cost" not in entry
        assert "currency" not in entry
        assert entry["costs"] == {"USD": "0.003", "EUR": "0.015"}

    def test_multi_currency_return_value_warns_naming_both_currencies(self) -> None:
        """compute_costs() returns one warning naming the service, model, Region and currencies."""
        self._record_mixed_currency_usage()
        warnings = compute_costs()
        assert warnings == [
            (
                "Multiple currencies resolved for bedrock-runtime/mixedmodel "
                "in us-east-1: ['EUR', 'USD']"
            )
        ]

    def test_single_currency_still_uses_cost_and_currency(self) -> None:
        """A single-currency record fills cost/currency and leaves ``costs`` empty.

        1000 * 0.000003 + 1000 * 0.000015 = 0.018 in one currency.
        """
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
    """Model-state routing -> UsageRecord.routing -> resolve_price plumbing.

    AWS publishes a separate, cheaper price for global cross-Region inference
    profiles, so the serving profile is part of the price lookup key.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
         stdapi/usage.py:record_bedrock_usage
         stdapi/pricing.py:resolve_price
    """

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
        """A global-routed call is billed at the global rate (0.003), not the regional one."""
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
        """With no routing profile tracked, the record is billed at the plain regional rate."""
        get_model_state("routedmodel").region = "us-east-1"
        self._set_routed_prices()
        record_bedrock_usage("routedmodel", input_tokens=1000)
        record = next(iter(usage.USAGE.get().values()))
        assert record.routing == ""
        compute_costs()
        # The plain (regional) rate, not the cheaper global one.
        assert record.cost == Decimal("3.300000")


class TestImageSpecPricing:
    """IMAGE_SPEC -> UsageRecord.output_images_by_spec -> resolve_price plumbing.

    Image models publish a price per "<resolution>:<quality>" spec, so images
    generated at different specs within one request must be priced per bucket
    rather than at a single flat rate.

    Ref: stdapi/usage.py:_dimension_price_buckets
         stdapi/pricing.py:resolve_price
    """

    def test_mixed_spec_images_price_each_bucket_independently(self) -> None:
        """Two specs in one record bill per bucket: 2 * 0.008 + 3 * 0.012 = 0.052."""
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
        """A spec-bearing and a flat-only call in one record bill 2 * 0.01 + 3 * 0.0036.

        A non-empty ``output_images_by_spec`` must not narrow the record to its
        breakdown buckets, which would price the flat-only portion at $0.

        Ref: stdapi/usage.py:_reconcile_buckets
        """
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
        """With no IMAGE_SPEC set, images bill at the flat per-image rate: 2 * 0.0036."""
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
        """A cleared IMAGE_SPEC keeps a later model's images on its flat rate.

        Only Titan Image Generator sets IMAGE_SPEC, so it is cleared after each
        call: a stale spec would send the next model's images into a spec
        bucket that model has no price for.

        Ref: stdapi/models/image/__init__.py:ImageModelBase._record_invoke_usage
        """
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
    """output_seconds_spec -> UsageRecord.output_seconds_by_spec -> resolve_price plumbing.

    Video models publish a per-second price per resolution bucket ("hd"), so
    generated seconds are priced per bucket with the unbucketed remainder
    falling back to the flat rate.

    Ref: stdapi/usage.py:_dimension_price_buckets
         stdapi/pricing.py:resolve_price
    """

    def test_mixed_spec_seconds_price_each_bucket_independently(self) -> None:
        """A flat call and an "hd" call in one record bill 5 * 0.06 + 5 * 0.08 = 0.7."""
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
        """With no spec bucket recorded, seconds bill at the flat rate: 6 * 0.05 = 0.3."""
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
        """The log entry carries both the flat output_seconds total and its per-spec breakdown."""
        get_model_state("videomodel").region = "us-east-1"
        record_bedrock_usage("videomodel", output_seconds=5, output_seconds_spec="hd")
        entry = next(iter(usage.usage_log_entries()))
        assert entry["output_seconds_by_spec"] == {"hd": 5}
        assert entry["output_seconds"] == 5


class TestCacheTtlPricing:
    """cache_write_tokens_by_ttl -> UsageRecord -> resolve_price plumbing.

    Bedrock reports cache writes per TTL bucket (``usage.cacheDetails``) and
    AWS charges a 1h write more than a 5m one, so each bucket is priced from
    its own rate; anything the breakdown does not cover falls back to the flat
    cache-write rate.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
         stdapi/usage.py:_add_cache_ttl_breakdown
         stdapi/usage.py:_dimension_price_buckets
    """

    def test_mixed_ttl_cache_writes_price_each_bucket_independently(self) -> None:
        """Two TTL buckets bill per bucket: 500 * 0.000004 + 1000 * 0.000008 = 0.01."""
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
        """The portion no TTL bucket covers bills at the flat rate: 400 * 0.000008 + 1200 * 0.000004.

        Ref: stdapi/usage.py:_reconcile_buckets
        """
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
        """A by-TTL breakdown with no flat total tops the flat quantity up and bills it.

        Ref: stdapi/usage.py:_add_cache_ttl_breakdown
        """
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
        """A flat-only and a breakdown-only call merge to 250 tokens and 0.001 in either order."""
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
    """record_bedrock_usage's tier precedence: explicit arg > this model's invocation state.

    Tiers are priced differently, and the shared model state can be overwritten
    by a sibling call to the same model, so an explicitly threaded tier wins.
    Bedrock's wire value for the Standard tier is "default", normalized here to
    "standard" (the pricing catalog's tier name).

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
         stdapi/usage.py:record_bedrock_usage
    """

    def test_explicit_tier_overrides_context_var(self) -> None:
        """An explicit ``tier=`` argument wins over a conflicting model-state tier."""
        state = get_model_state("tiermodel")
        state.region = "us-east-1"
        state.service_tier = "priority"
        record_bedrock_usage("tiermodel", tier="flex", input_tokens=100)
        record = next(iter(usage.USAGE.get().values()))
        assert record.tier == "flex"

    def test_falls_back_to_service_tier_context_var_when_not_given(self) -> None:
        """With no explicit tier, the record picks up the model-state tier."""
        state = get_model_state("tiermodel")
        state.region = "us-east-1"
        state.service_tier = "priority"
        record_bedrock_usage("tiermodel", input_tokens=100)
        record = next(iter(usage.USAGE.get().values()))
        assert record.tier == "priority"

    def test_falls_back_to_standard_when_neither_is_set(self) -> None:
        """A never-overridden model-state tier ("default") is normalized to "standard"."""
        get_model_state("tiermodel").region = "us-east-1"
        record_bedrock_usage("tiermodel", input_tokens=100)
        record = next(iter(usage.USAGE.get().values()))
        assert record.tier == "standard"


class TestInitUsageTokenReset:
    """init_usage() returns a token that restores the previous USAGE dict.

    Without the token, a nested in-process call permanently replaces the outer
    request's dict.

    Ref: stdapi/usage.py:init_usage
    """

    def test_nested_init_usage_does_not_clobber_the_outer_dict(self) -> None:
        """Resetting the inner token restores the exact prior dict object, not a copy."""
        outer_token = usage.init_usage()
        outer_records = usage.USAGE.get()
        inner_token = usage.init_usage()
        assert usage.USAGE.get() is not outer_records

        usage.USAGE.reset(inner_token)
        assert usage.USAGE.get() is outer_records

        usage.USAGE.reset(outer_token)


class TestComputeCostsUnpricedDimension:
    """A record with a dimension for which no price could be resolved.

    A pricing miss must never block the request: the cost is omitted and the
    dimension is named in a warning for the request log.

    Ref: stdapi/usage.py:_apply_record_cost
    """

    def test_fully_unpriced_model_returns_a_warning_naming_the_dimension(self) -> None:
        """A model with no price entry at all is warned about and left at zero cost."""
        # Seed an unrelated price so the catalog counts as ready -- the point
        # of this test is a genuine per-model miss, not an unloaded catalog.
        set_test_price(
            "othermodel", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD"
        )
        get_model_state("nopricemodel").region = "us-east-1"
        record_bedrock_usage("nopricemodel", input_tokens=1000)
        warnings = compute_costs()
        assert warnings == [
            (
                "No price found for bedrock-runtime/nopricemodel in "
                "us-east-1: ['input_tokens']"
            )
        ]
        record = next(iter(usage.USAGE.get().values()))
        assert record.cost == Decimal(0)
        assert record.currency == ""

    def test_partially_priced_model_still_prices_the_resolvable_dimension(self) -> None:
        """With one dimension priced and one not, the cost covers only the priced one."""
        get_model_state("partialpricemodel").region = "us-east-1"
        set_test_price(
            "partialpricemodel", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD"
        )
        record_bedrock_usage("partialpricemodel", input_tokens=1000, output_tokens=1000)
        warnings = compute_costs()
        assert warnings == [
            (
                "No price found for bedrock-runtime/partialpricemodel in "
                "us-east-1: ['output_tokens']"
            )
        ]
        record = next(iter(usage.USAGE.get().values()))
        assert record.cost == Decimal("0.003000")
        assert record.currency == "USD"


class TestComputeCostsRegionSkip:
    """Records with no Region attributed are never priced.

    Prices are per Region, so an unattributed record has no defensible rate to
    bill at and is skipped without a pricing-miss warning.

    Ref: stdapi/usage.py:compute_costs
    """

    def test_record_without_region_is_skipped(self) -> None:
        """An empty region leaves cost/currency untouched and emits no warning."""
        key = UsageKey(Service.BEDROCK, "modelnoregion", "", "", "standard")
        usage.USAGE.get()[key] = UsageRecord(
            Service.BEDROCK,
            "modelnoregion",
            "",
            region="",
            quantities={Dimension.INPUT_TOKENS: 100},
        )
        warnings = compute_costs()
        record = usage.USAGE.get()[key]
        assert record.cost == Decimal(0)
        assert record.currency == ""
        assert warnings == [], "an unattributed record must not report a pricing miss"


class TestEmitUsageMetrics:
    """One CloudWatch EMF log line per usage record, written to stdout.

    Each line is an EMF document: ``_aws.CloudWatchMetrics`` holds the metric
    directives (Namespace, Dimensions, Metrics) and every metric named in a
    directive must exist as a root member of the same line.

    Ref: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Embedded_Metric_Format_Specification.html
         stdapi/usage.py:emit_usage_metrics
    """

    def test_disabled_setting_emits_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With cloudwatch_metrics off, a priced record emits no line at all."""
        monkeypatch.setattr(SETTINGS, "cloudwatch_metrics", False)
        get_model_state("emfmodel").region = "us-east-1"
        record_bedrock_usage("emfmodel", input_tokens=1000)
        written: list[dict[str, Any]] = []
        monkeypatch.setattr(usage, "stdout_write", written.append)
        emit_usage_metrics()
        assert written == []

    def test_single_currency_emits_one_line_with_quantities_and_cost(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A priced record emits one EMF line carrying its quantities and Cost/Currency.

        Cost sits in its own directive dimensioned by ["Model", "Currency"]:
        EMF publishes every metric of a directive under each of that
        directive's dimension sets, so a single directive spanning ["Model"]
        and ["Model", "Currency"] would also publish Cost bare-by-Model,
        silently summing across currencies.
        """
        monkeypatch.setattr(SETTINGS, "cloudwatch_metrics", True)
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
        # The declared "Model" dimension needs a matching root member.
        assert payload["Model"] == "emfmodel"
        quantity_directive, cost_directive = payload["_aws"]["CloudWatchMetrics"]
        assert quantity_directive["Dimensions"] == [["Model"]]
        assert {m["Name"] for m in quantity_directive["Metrics"]} == {"InputTokens"}
        assert cost_directive["Dimensions"] == [["Model", "Currency"]]
        assert {m["Name"] for m in cost_directive["Metrics"]} == {"Cost"}
        assert {
            directive["Namespace"] for directive in (quantity_directive, cost_directive)
        } == {SETTINGS.cloudwatch_metrics_namespace}

    def test_no_cost_resolved_emits_quantities_only_with_model_dimension(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no price resolved, the line keeps its quantities and omits Cost/Currency.

        Currency is a declared EMF dimension, so it must be absent whenever no
        Cost member is emitted.
        """
        monkeypatch.setattr(SETTINGS, "cloudwatch_metrics", True)
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

    def test_a_record_with_nothing_to_report_emits_no_line(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A record with no billed quantity and no cost is skipped, not emitted empty.

        ``total_tokens`` is a reporting total rather than one of the billed
        ``Dimension`` entries in ``_DIMENSION_INFO``, so a record carrying only
        it has empty ``quantities`` and nothing to meter. An EMF directive that
        declares dimensions but names no metric is an invalid document
        CloudWatch rejects, so the whole line is dropped.
        """
        monkeypatch.setattr(SETTINGS, "cloudwatch_metrics", True)
        get_model_state("emfmodel").region = "us-east-1"
        record_bedrock_usage("emfmodel", total_tokens=1000)
        compute_costs()

        written: list[dict[str, Any]] = []
        monkeypatch.setattr(usage, "stdout_write", written.append)
        emit_usage_metrics()

        assert usage.USAGE.get(), "the record itself must still exist"
        assert written == []

    def test_multi_currency_emits_one_extra_line_without_duplicating_quantities(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A two-currency record emits one line per currency, quantities only on the first.

        Repeating the quantity metrics on the second line would double-count
        them, since both lines carry the same ["Model"] dimension value.
        """
        monkeypatch.setattr(SETTINGS, "cloudwatch_metrics", True)
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
        cost_by_currency = {line["Currency"]: line["Cost"] for line in written}
        assert cost_by_currency["USD"] == pytest.approx(0.003)  # 1000 * 0.000003
        assert cost_by_currency["EUR"] == pytest.approx(0.015)  # 1000 * 0.000015
        # The extra-currency line declares the cost directive only.
        assert [
            directive["Dimensions"] for directive in second["_aws"]["CloudWatchMetrics"]
        ] == [[["Model", "Currency"]]]

    def test_cost_is_a_json_serializable_float(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cost is emitted as a JSON float, not a Decimal.

        EMF metric values must be JSON numbers, and ``json.dumps`` raises
        TypeError on the Decimal the cost is computed in.
        """
        monkeypatch.setattr(SETTINGS, "cloudwatch_metrics", True)
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
    """record_bedrock_usage's context="long" detection (prompt > 200K tokens).

    AWS bills a whole call at the long-context rate once the prompt exceeds
    200K tokens. Bedrock reports fresh, cache-read and cache-write tokens
    separately (inputTokens excludes both cache counts), so the threshold is
    evaluated on their sum, and the bucket is part of the record key so long
    and standard calls to one model never merge.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
         stdapi/usage.py:record_bedrock_usage
    """

    def test_exactly_threshold_is_not_long(self) -> None:
        """A prompt of exactly 200_000 tokens stays in the standard bucket (boundary)."""
        get_model_state("longmodel").region = "us-east-1"
        record_bedrock_usage("longmodel", input_tokens=200_000)
        record = next(iter(usage.USAGE.get().values()))
        assert record.context == ""

    def test_one_token_over_threshold_via_mixed_dimensions_is_long(self) -> None:
        """Fresh, cached and cache-write tokens sum to 200_001 and mark the record long."""
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
        """A long-context call and a standard call for the same model stay separate records."""
        get_model_state("longmodel").region = "us-east-1"
        record_bedrock_usage("longmodel", input_tokens=1_000)
        record_bedrock_usage("longmodel", input_tokens=200_001)
        records = list(usage.USAGE.get().values())
        assert len(records) == 2
        contexts = {record.context for record in records}
        assert contexts == {"", "long"}

    def test_usage_log_entries_include_context_only_for_the_long_record(self) -> None:
        """Only the long-context entry carries "context": "long", with its own token total."""
        get_model_state("longmodel").region = "us-east-1"
        record_bedrock_usage("longmodel", input_tokens=1_000)
        record_bedrock_usage("longmodel", input_tokens=200_001)
        entries = usage.usage_log_entries()
        assert len(entries) == 2
        (long_entry,) = [entry for entry in entries if entry.get("context") == "long"]
        (standard_entry,) = [entry for entry in entries if "context" not in entry]
        assert long_entry["input_tokens"] == 200_001
        assert standard_entry["input_tokens"] == 1_000

    def test_compute_costs_uses_the_long_context_price_for_the_long_record(
        self,
    ) -> None:
        """The long record bills at the long-context rate while the standard one keeps its own.

        Ref: stdapi/pricing.py:resolve_price
        """
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
    """grounding_requests -> Dimension.GROUNDING_REQUESTS plumbing.

    Amazon Nova's ``nova_grounding`` system tool bills per invocation on top of
    inference, so grounding calls are a dimension of their own.

    Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
         stdapi/usage.py:record_bedrock_usage
    """

    def test_zero_or_none_grounding_requests_omit_the_field(self) -> None:
        """Zero or None grounding calls add neither the dimension nor the log key."""
        get_model_state("groundedmodel").region = "us-east-1"
        record_bedrock_usage("groundedmodel", input_tokens=100, grounding_requests=0)
        record_bedrock_usage("groundedmodel", input_tokens=100, grounding_requests=None)
        record = next(iter(usage.USAGE.get().values()))
        assert Dimension.GROUNDING_REQUESTS not in record.quantities
        entry = next(iter(usage.usage_log_entries()))
        assert "grounding_requests" not in entry

    def test_usage_log_entries_reports_grounding_requests_count(self) -> None:
        """The log entry reports the accumulated grounding_requests count."""
        get_model_state("groundedmodel").region = "us-east-1"
        record_bedrock_usage("groundedmodel", grounding_requests=2)
        entry = next(iter(usage.usage_log_entries()))
        assert entry["grounding_requests"] == 2

    def test_grounding_requests_are_priced_per_request(self) -> None:
        """Grounding calls bill per request: 2 * 0.03 = 0.06."""
        get_model_state("groundedmodel").region = "us-east-1"
        set_test_price(
            "groundedmodel", "us-east-1", Dimension.GROUNDING_REQUESTS, "0.03", "USD"
        )
        record_bedrock_usage("groundedmodel", grounding_requests=2)
        record = next(iter(usage.USAGE.get().values()))
        compute_costs()
        assert record.cost == Decimal("0.060000")


class TestSearchUnitsUsage:
    """search_units -> Dimension.SEARCH_UNITS plumbing.

    Bedrock rerank is billed in search units rather than tokens.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html
         stdapi/usage.py:record_bedrock_usage
    """

    def test_search_units_are_recorded_logged_and_priced(self) -> None:
        """search_units reach the record, the log entry and the cost: 3 * 0.001 = 0.003."""
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
    """input_images/input_seconds + media_spec -> *_by_spec breakdown plumbing.

    Multimodal models publish separate per-item prices for document pages,
    audio seconds and video seconds, so each call's single media kind is
    recorded as its own spec bucket.

    Ref: stdapi/usage.py:record_bedrock_usage
         stdapi/usage.py:_dimension_price_buckets
    """

    def test_input_images_with_media_spec_price_and_log_by_spec(self) -> None:
        """input_images with media_spec="document" bills from the "document" bucket."""
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
        """input_seconds with media_spec="audio" bills from the "audio" bucket: 45 * 0.0001."""
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
        """Audio and video seconds in one record bill 30 * 0.0001 + 10 * 0.0005 = 0.008."""
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
    """ModelInvocationState.routing == "latency" -> UsageKey/UsageRecord/log plumbing.

    Latency-optimized inference carries its own published price, so the routing
    label is part of the price lookup and falls back to the plain rate when AWS
    publishes no latency-optimized price for the model.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/latency-optimized-inference.html
         stdapi/models/__init__.py:_request_routing
         stdapi/pricing.py:resolve_price
    """

    def test_latency_routing_state_produces_latency_key_log_and_price(self) -> None:
        """A "latency"-routed call keys and logs as "latency" and bills 1000 * 0.004."""
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
        """With no latency price indexed, a "latency"-routed call bills the plain rate."""
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
    """Non-Bedrock billed quantities: Polly and Translate characters, Transcribe seconds, Comprehend units.

    Each helper returns the quantity AWS actually bills -- which is not the
    input size when a service applies a minimum -- and records it under a model
    key that encodes the priced variant (Polly's engine, Comprehend's feature).

    Ref: https://docs.aws.amazon.com/transcribe/latest/dg/what-is.html
         stdapi/usage.py:record_polly_usage
         stdapi/usage.py:record_transcribe_usage
         stdapi/usage.py:record_translate_usage
         stdapi/usage.py:record_comprehend_usage
    """

    def test_record_polly_usage_bills_exact_character_count(self) -> None:
        """Polly bills the exact character count, with no minimum, per engine."""
        billed = record_polly_usage(42, "neural")
        assert billed == 42
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.INPUT_CHARACTERS] == 42
        assert record.service == Service.POLLY
        # The engine is part of the priced model key: Polly rates differ by engine.
        assert record.model == "amazon.polly-neural"

    def test_record_translate_usage_bills_exact_character_count(self) -> None:
        """Translate bills the exact character count, with no minimum."""
        billed = record_translate_usage(100)
        assert billed == 100
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.INPUT_CHARACTERS] == 100
        assert record.service == Service.TRANSLATE
        assert record.model == "amazon.translate"

    @pytest.mark.parametrize(
        ("duration", "expected"), [(5.0, 15), (15.0, 15), (15.4, 16), (30.0, 30)]
    )
    def test_record_transcribe_usage_applies_15_second_minimum(
        self, duration: float, expected: int
    ) -> None:
        """Transcribe bills per second, rounded up, with a 15-second per-request minimum."""
        billed = record_transcribe_usage(duration)
        assert billed == expected
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.INPUT_SECONDS] == expected
        assert record.service == Service.TRANSCRIBE
        assert record.model == "amazon.transcribe"

    @pytest.mark.parametrize(("text_length", "expected"), [(50, 3), (300, 3), (500, 5)])
    def test_record_comprehend_usage_applies_3_unit_minimum(
        self, text_length: int, expected: int
    ) -> None:
        """Comprehend bills in 100-character units, rounded up, with a 3-unit minimum."""
        billed = record_comprehend_usage(text_length, "language-detection")
        assert billed == expected
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.COMPREHEND_UNITS] == expected
        assert record.service == Service.COMPREHEND
        # The feature is part of the priced model key: rates differ per feature.
        assert record.model == "amazon.comprehend-language-detection"

    def test_record_polly_usage_with_zero_characters_records_nothing(self) -> None:
        """Zero characters bill nothing and create no usage record."""
        billed = record_polly_usage(0, "neural")
        assert billed == 0
        assert usage.USAGE.get() == {}

    def test_record_translate_usage_with_zero_characters_records_nothing(self) -> None:
        """Zero characters bill nothing and create no usage record."""
        billed = record_translate_usage(0)
        assert billed == 0
        assert usage.USAGE.get() == {}

    def test_record_transcribe_usage_with_zero_duration_bills_the_minimum(self) -> None:
        """Zero duration still bills the 15-second minimum rather than recording nothing."""
        billed = record_transcribe_usage(0)
        assert billed == 15
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.INPUT_SECONDS] == 15

    def test_record_comprehend_usage_with_zero_length_bills_the_minimum(self) -> None:
        """Zero text length still bills the 3-unit minimum rather than recording nothing."""
        billed = record_comprehend_usage(0, "language-detection")
        assert billed == 3
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.COMPREHEND_UNITS] == 3

    def test_non_bedrock_usage_does_not_inherit_a_prior_bedrock_region(self) -> None:
        """A prior Bedrock call's Region does not leak into a Polly record.

        The MODEL_STATE region fallback is Bedrock-only: a shared one would let
        Polly/Transcribe/Translate/Comprehend be priced against the Bedrock
        Region of an earlier call in the same request.

        Ref: stdapi/usage.py:_record_usage
        """
        get_model_state(
            "priorbedrockmodel"
        ).region = "us-east-1"  # Simulates a prior Bedrock call in this context.
        record_polly_usage(42, "neural")
        record = next(iter(usage.USAGE.get().values()))
        assert record.region == ""


class TestGuardrailPolicyPricing:
    """ApplyGuardrail usage: every applied policy billed at its own rate.

    A guardrail applies every policy the operator configured to the same
    content, and AWS prices each policy separately, so their rates sum.
    Recording one model per guardrail could only ever charge a single
    policy's rate, under-billing every multi-policy guardrail.

    Ref: https://aws.amazon.com/bedrock/pricing/
         stdapi/usage.py:record_guardrail_policy_usage
         stdapi/monitoring.py:_add_warnings
    """

    @staticmethod
    def _seed(policy: str, dimension: Dimension, amount: str) -> None:
        """Publish one policy's rate in the test price index."""
        set_test_price(
            normalize_model_key(guardrail_policy_model(policy)),
            "us-east-1",
            dimension,
            amount,
            "USD",
        )

    def test_unpriced_text_units_emit_no_pricing_miss_warning(self) -> None:
        """With no Guardrails rate in the catalog, the record is silent, not warned about.

        A configured guardrail applies to every route, so a warned miss would
        raise the level of every request log to ``warning``.
        """
        # Seed an unrelated price so the catalog counts as ready.
        set_test_price(
            "othermodel", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD"
        )
        record_guardrail_policy_usage({"contentPolicyUnits": 3}, region="us-east-1")
        assert compute_costs() == []
        record = next(iter(usage.USAGE.get().values()))
        assert record.cost == Decimal(0)

    def test_every_applied_policy_is_billed_at_its_own_rate(self) -> None:
        """Three policies over the same text cost the sum of three rates, not one."""
        self._seed("content", Dimension.TEXT_UNITS, "0.00015")
        self._seed("topic", Dimension.TEXT_UNITS, "0.00015")
        self._seed("automated-reasoning", Dimension.TEXT_UNITS, "0.00017")
        record_guardrail_policy_usage(
            {
                "contentPolicyUnits": 3,
                "topicPolicyUnits": 3,
                "automatedReasoningPolicyUnits": 3,
            },
            region="us-east-1",
        )
        assert compute_costs() == []
        records = list(usage.USAGE.get().values())
        assert len(records) == 3
        # 3 text units x (0.00015 + 0.00015 + 0.00017).
        assert sum(record.cost for record in records) == Decimal("0.001410")
        assert {record.currency for record in records} == {"USD"}

    def test_the_content_policy_image_rate_bills_separately_from_its_text_rate(
        self,
    ) -> None:
        """One policy, two dimensions: images must not be charged the text rate."""
        self._seed("content", Dimension.TEXT_UNITS, "0.00015")
        self._seed("content", Dimension.INPUT_IMAGES, "0.00075")
        record_guardrail_policy_usage(
            {"contentPolicyUnits": 2, "contentPolicyImageUnits": 1}, region="us-east-1"
        )
        assert compute_costs() == []
        record = next(iter(usage.USAGE.get().values()))
        # Both dimensions aggregate onto the one content-policy model.
        assert record.cost == Decimal("0.001050")

    def test_unapplied_policies_record_nothing(self) -> None:
        """Absent, zero and non-integer counts must not mint empty usage records.

        ``usage`` reports every policy field, zeroed for the ones the
        guardrail does not apply.
        """
        self._seed("content", Dimension.TEXT_UNITS, "0.00015")
        record_guardrail_policy_usage(
            {
                "contentPolicyUnits": 2,
                "topicPolicyUnits": 0,
                "wordPolicyUnits": None,
                "automatedReasoningPolicies": 4,
            },
            region="us-east-1",
        )
        assert [key.model for key in usage.USAGE.get()] == [
            guardrail_policy_model("content")
        ]

    def test_the_policy_count_field_is_never_billed(self) -> None:
        """``automatedReasoningPolicies`` counts policies, not billable units."""
        self._seed("automated-reasoning", Dimension.TEXT_UNITS, "0.00017")
        record_guardrail_policy_usage(
            {"automatedReasoningPolicies": 7}, region="us-east-1"
        )
        assert usage.USAGE.get() == {}


class TestFormatCost:
    """format_cost: exact plain-decimal text with no exponent or trailing zeros.

    Costs go into JSON logs as strings: float rendering would turn small
    amounts into exponent notation and lose exactness.

    Ref: stdapi/usage.py:format_cost
    """

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


class TestCloudWatchNamespaceValidation:
    """cloudwatch_metrics_namespace is validated at settings load, not at emission time.

    An invalid namespace makes CloudWatch silently drop EMF metric extraction
    (the log line is still written), so the gateway rejects it up front. Its
    charset is stricter than CloudWatch's, which also accepts the space
    character, and the reserved ``AWS/`` prefix is refused outright.

    Ref: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/cloudwatch_concepts.html
         stdapi/config.py:_Settings._validate_cloudwatch_namespace
    """

    @staticmethod
    def _settings(namespace: str) -> _Settings:
        return _Settings(
            aws_bedrock_regions=["us-east-1"], cloudwatch_metrics_namespace=namespace
        )

    @pytest.mark.parametrize(
        "namespace", ["stdapi", "my-app_metrics.prod/v1#2:3", "a" * 255]
    )
    def test_valid_namespace_accepted(self, namespace: str) -> None:
        """Every allowed character and the 255-character limit are accepted unchanged."""
        assert self._settings(namespace).cloudwatch_metrics_namespace == namespace

    def test_invalid_character_rejected(self) -> None:
        """A space in the namespace fails validation on the charset rule."""
        with pytest.raises(
            ValidationError, match="cloudwatch_metrics_namespace"
        ) as exc:
            self._settings("invalid namespace")
        (error,) = exc.value.errors()
        assert error["loc"] == ("cloudwatch_metrics_namespace",)
        assert "must be 1-255 characters" in error["msg"]

    def test_reserved_aws_prefix_rejected(self) -> None:
        """A namespace starting with the reserved "AWS/" prefix fails validation."""
        with pytest.raises(ValidationError, match="reserved") as exc:
            self._settings("AWS/MyNamespace")
        (error,) = exc.value.errors()
        assert error["loc"] == ("cloudwatch_metrics_namespace",)
        assert 'must not start with the reserved "AWS/" prefix' in error["msg"]

    def test_too_long_namespace_rejected(self) -> None:
        """A 256-character namespace fails validation on the length rule."""
        with pytest.raises(
            ValidationError, match="cloudwatch_metrics_namespace"
        ) as exc:
            self._settings("a" * 256)
        (error,) = exc.value.errors()
        assert error["loc"] == ("cloudwatch_metrics_namespace",)
        assert "must be 1-255 characters" in error["msg"]


class TestConcurrentSameModelUsageAttribution:
    """MODEL_STATE is one shared entry per model, not per call.

    A sibling call to the same model can overwrite region/tier/routing between
    invocation and recording, so callers that may run concurrently with
    differing values pass them explicitly to record_bedrock_usage.

    Ref: stdapi/usage.py:ModelInvocationState
    """

    async def test_concurrent_calls_with_explicit_region_attribute_correctly(
        self,
    ) -> None:
        """Concurrent same-model calls passing region explicitly bill each Region separately."""

        async def call(region: str, tokens: int) -> None:
            # Simulate a sibling call's write to the shared model state.
            get_model_state("concurrentmodel").region = region
            await asyncio.sleep(0)
            record_bedrock_usage("concurrentmodel", region=region, input_tokens=tokens)

        await asyncio.gather(call("us-east-1", 1000), call("us-west-2", 2000))

        by_region = {
            key.region: record.quantities[Dimension.INPUT_TOKENS]
            for key, record in usage.USAGE.get().items()
        }
        assert by_region == {"us-east-1": 1000, "us-west-2": 2000}

    async def test_concurrent_calls_without_explicit_region_share_last_written_state(
        self,
    ) -> None:
        """Omitting region collapses concurrent calls onto the last-written Region.

        Documented, accepted behavior of the shared-state fallback: both calls
        merge into one record keyed to whichever Region was written last.

        The interleaving is driven by explicit events rather than by ``gather``'s
        scheduling order, so which write is "last" does not depend on how many
        times ``record_bedrock_usage`` happens to await.
        """
        first_written = asyncio.Event()
        second_written = asyncio.Event()

        async def early() -> None:
            get_model_state("fallbackmodel").region = "us-east-1"
            first_written.set()
            await second_written.wait()
            record_bedrock_usage("fallbackmodel", input_tokens=1000)

        async def late() -> None:
            await first_written.wait()
            get_model_state("fallbackmodel").region = "us-west-2"
            second_written.set()
            record_bedrock_usage("fallbackmodel", input_tokens=2000)

        await asyncio.gather(early(), late())

        records = usage.USAGE.get()
        assert len(records) == 1
        key, record = next(iter(records.items()))
        assert key.region == "us-west-2"  # Last write wins.
        assert record.quantities[Dimension.INPUT_TOKENS] == 3000
