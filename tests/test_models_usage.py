"""Unit tests for grounding-tool usage counting in stdapi.models.

Covers ``_count_grounding_tool_uses`` (non-streaming Converse responses) and
``_capture_stream_usage`` (ConverseStream events), which feed the
``grounding_requests`` billed dimension (see tests/test_usage.py for its
pricing/log-entry coverage). Also covers ``_record_converse_usage`` billing
details (cacheDetails TTL breakdown, effective-tier pricing, concurrent-call
region attribution) and exactly-once recording across ``converse()`` failover.
"""

from asyncio import Event, create_task, gather, wait_for
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

import pytest
from botocore.exceptions import ClientError
from pydantic_core import to_json

import stdapi.models
from stdapi import usage
from stdapi.config import SETTINGS
from stdapi.models import ModelBase, _count_grounding_tool_uses, _iter_invoke_stream
from stdapi.monitoring import REQUEST_LOG, EventLog
from stdapi.pricing import Dimension, Price, PriceKey, Service, _state
from stdapi.region_routing import RegionRouter
from stdapi.usage import compute_costs
from tests.conftest import set_test_price

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime.type_defs import ConverseResponseTypeDef


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


@pytest.fixture(autouse=True)
def _usage_scope() -> Generator[None]:
    """Install fresh per-request usage/model-state scopes for each test."""
    usage_token = usage.init_usage()
    state_token = usage.init_model_state()
    yield
    usage.USAGE.reset(usage_token)
    usage.MODEL_STATE.reset(state_token)


class TestCountGroundingToolUses:
    """_count_grounding_tool_uses: counts billed-grounding toolUse blocks in a Converse response."""

    def test_counts_only_billed_grounding_tool_blocks(self) -> None:
        """Two nova_grounding toolUse blocks count; text and other tool blocks don't."""
        response: dict[str, Any] = {
            "output": {
                "message": {
                    "content": [
                        {"toolUse": {"name": "nova_grounding", "toolUseId": "1"}},
                        {"text": "some text"},
                        {"toolUse": {"name": "nova_grounding", "toolUseId": "2"}},
                        {"toolUse": {"name": "other_tool", "toolUseId": "3"}},
                    ]
                }
            }
        }
        assert _count_grounding_tool_uses(response) == 2  # type: ignore[arg-type]

    def test_missing_output_returns_zero(self) -> None:
        """A response with no `output` key must count as zero, not raise."""
        assert _count_grounding_tool_uses(cast("ConverseResponseTypeDef", {})) == 0

    def test_empty_content_returns_zero(self) -> None:
        """A response with empty output/message content must count as zero."""
        response: dict[str, Any] = {"output": {"message": {"content": []}}}
        assert _count_grounding_tool_uses(response) == 0  # type: ignore[arg-type]


class TestCaptureStreamUsage:
    """_capture_stream_usage: counts contentBlockStart grounding-tool events across a stream."""

    async def test_grounding_tool_uses_are_recorded_at_the_metadata_event(self) -> None:
        """Two nova_grounding contentBlockStart events feed GROUNDING_REQUESTS == 2."""

        async def fake_stream() -> AsyncIterator[dict[str, Any]]:
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"name": "nova_grounding", "toolUseId": "1"}},
                }
            }
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 1,
                    "start": {"toolUse": {"name": "nova_grounding", "toolUseId": "2"}},
                }
            }
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 2,
                    "start": {"toolUse": {"name": "other_tool", "toolUseId": "3"}},
                }
            }
            yield {
                "metadata": {
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
                }
            }

        model: ModelBase[Any, Any] = ModelBase("streaminggroundedmodel")
        events = [
            event
            async for event in model._capture_stream_usage(fake_stream())  # noqa: SLF001
        ]

        assert len(events) == 4  # All events are yielded unmodified.
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.GROUNDING_REQUESTS] == 2

    async def test_grounding_counter_resets_after_each_metadata_event(self) -> None:
        """A second metadata event must not re-bill earlier grounding calls.

        Live-verified (2026-07): AWS emits a single trailing metadata event
        even for multi-round server-tool streams; this guards the defensive
        counter reset should that cardinality ever change.
        """

        async def fake_stream() -> AsyncIterator[dict[str, Any]]:
            yield {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"name": "nova_grounding", "toolUseId": "1"}},
                }
            }
            yield {
                "metadata": {
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
                }
            }
            yield {
                "metadata": {
                    "usage": {"inputTokens": 3, "outputTokens": 4, "totalTokens": 7}
                }
            }

        model: ModelBase[Any, Any] = ModelBase("multimetadatamodel")
        _ = [e async for e in model._capture_stream_usage(fake_stream())]  # noqa: SLF001

        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.GROUNDING_REQUESTS] == 1
        assert record.quantities[Dimension.INPUT_TOKENS] == 4

    async def test_metadata_service_tier_is_billed(self) -> None:
        """The serviceTier reported on the metadata event drives the billed tier."""

        async def fake_stream() -> AsyncIterator[dict[str, Any]]:
            yield {
                "metadata": {
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                    "serviceTier": {"type": "flex"},
                }
            }

        model: ModelBase[Any, Any] = ModelBase("streamtieredmodel")
        _ = [e async for e in model._capture_stream_usage(fake_stream())]  # noqa: SLF001

        assert next(iter(usage.USAGE.get())).tier == "flex"


class TestIterInvokeStream:
    """_iter_invoke_stream: the usage callback fires only for metrics-carrying chunks."""

    async def test_usage_callback_skipped_for_chunks_without_invocation_metrics(
        self,
    ) -> None:
        """Chunks lacking 'amazon-bedrock-invocationMetrics' don't invoke the callback."""

        async def fake_body() -> AsyncIterator[dict[str, Any]]:
            yield {"chunk": {"bytes": to_json({"text": "hello"})}}
            yield {
                "chunk": {
                    "bytes": to_json(
                        {"amazon-bedrock-invocationMetrics": {"inputTokenCount": 1}}
                    )
                }
            }
            yield {"chunk": {"bytes": to_json({"text": "world"})}}

        calls: list[Any] = []
        chunks = [
            chunk
            async for chunk in _iter_invoke_stream(fake_body(), calls.append)  # type: ignore[arg-type]
        ]

        assert len(chunks) == 3
        assert len(calls) == 1
        assert "amazon-bedrock-invocationMetrics" in calls[0]


class TestRecordConverseUsageEffectiveTier:
    """_record_converse_usage: the response's serviceTier overrides the requested one."""

    def test_response_service_tier_overrides_requested_tier(self) -> None:
        """A flex request served at standard must be billed at standard."""
        usage.get_model_state("tieredmodel").service_tier = "flex"
        ModelBase("tieredmodel")._record_converse_usage(  # noqa: SLF001
            {  # type: ignore[typeddict-item]
                "usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6},
                "serviceTier": {"type": "standard"},
            }
        )
        assert next(iter(usage.USAGE.get())).tier == "standard"

    def test_requested_tier_is_used_when_response_reports_none(self) -> None:
        """Without a response serviceTier, the requested tier still applies."""
        usage.get_model_state("tieredmodel").service_tier = "priority"
        ModelBase("tieredmodel")._record_converse_usage(  # noqa: SLF001
            {"usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6}}  # type: ignore[typeddict-item]
        )
        assert next(iter(usage.USAGE.get())).tier == "priority"


class TestPerCallAttribution:
    """Explicit per-call region/routing must win over the shared model state.

    Concurrent same-model calls share one ModelInvocationState: a sibling
    call may overwrite it between invocation and recording, so recording
    relies on the explicitly threaded per-call values instead.
    """

    def test_explicit_region_and_routing_win_over_model_state(self) -> None:
        """A record carries its own call's region/routing, not the state's."""
        state = usage.get_model_state("racymodel")
        state.region = "eu-west-3"
        state.routing = "global"
        ModelBase("racymodel")._record_converse_usage(  # noqa: SLF001
            {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}},  # type: ignore[typeddict-item]
            region="us-east-1",
            routing="",
        )
        key = next(iter(usage.USAGE.get()))
        assert key.region == "us-east-1"
        assert key.routing == ""


async def _noop_prepare(_self: object, _request: object, _region: str) -> None:
    """No-op stand-in for ModelBase._prepare_converse_request_for_region."""


class TestConcurrentConverseAttribution:
    """Two in-flight _converse calls must each bill their own serving region."""

    async def test_concurrent_calls_bill_two_distinct_regions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sibling overwrite of the shared model state must not leak into either record."""
        started: dict[str, Event] = {"us-east-1": Event(), "eu-west-1": Event()}
        release = Event()
        tokens: dict[str, tuple[int, int]] = {
            "us-east-1": (11, 1),
            "eu-west-1": (22, 2),
        }

        class _BlockingClient:
            """Fake Bedrock client whose converse blocks until released."""

            def __init__(self, region: str) -> None:
                """Bind the fake client to its target region."""
                self._region = region

            async def converse(self, **_kwargs: object) -> dict[str, Any]:
                """Signal the call started, wait for release, report per-region usage."""
                started[self._region].set()
                await release.wait()
                input_tokens, output_tokens = tokens[self._region]
                return {
                    "usage": {
                        "inputTokens": input_tokens,
                        "outputTokens": output_tokens,
                        "totalTokens": input_tokens + output_tokens,
                    }
                }

        monkeypatch.setattr(
            ModelBase, "_prepare_converse_request_for_region", _noop_prepare
        )
        monkeypatch.setattr(
            stdapi.models,
            "bedrock_client",
            lambda region, **_kwargs: _BlockingClient(region),
        )

        model: ModelBase[Any, Any] = ModelBase("racemodel")
        regions: tuple[RegionName, ...] = ("us-east-1", "eu-west-1")
        calls = [
            create_task(
                model._converse(  # noqa: SLF001
                    {"modelId": "racemodel"}, region, single_region=True
                )
            )
            for region in regions
        ]
        await wait_for(
            gather(started["us-east-1"].wait(), started["eu-west-1"].wait()), timeout=5
        )
        # Both calls are in flight: mimic a sibling overwriting the shared state.
        usage.get_model_state("racemodel").region = "ap-south-1"
        release.set()
        await gather(*calls)

        records = usage.USAGE.get()
        assert {key.region for key in records} == {"us-east-1", "eu-west-1"}
        for key, record in records.items():
            expected_input, expected_output = tokens[key.region]
            assert record.quantities[Dimension.INPUT_TOKENS] == expected_input
            assert record.quantities[Dimension.OUTPUT_TOKENS] == expected_output


class TestRecordConverseUsageCacheDetails:
    """_record_converse_usage: cacheDetails feeds the per-TTL cache-write breakdown."""

    def test_cache_details_populate_the_ttl_breakdown(self) -> None:
        """Each cacheDetails entry maps its ttl to its inputTokens."""
        ModelBase("cachedetailmodel")._record_converse_usage(  # noqa: SLF001
            {  # type: ignore[typeddict-item]
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 1,
                    "totalTokens": 11,
                    "cacheWriteInputTokens": 700,
                    "cacheDetails": [
                        {"ttl": "5m", "inputTokens": 500},
                        {"ttl": "1h", "inputTokens": 200},
                    ],
                }
            }
        )
        record = next(iter(usage.USAGE.get().values()))
        assert record.cache_write_tokens_by_ttl == {"5m": 500, "1h": 200}
        # The breakdown exactly covers the flat total: no deficit top-up.
        assert record.quantities[Dimension.CACHE_WRITE_TOKENS] == 700


class TestEffectiveTierPricing:
    """The response-reported tier must drive pricing, not the requested one."""

    def test_flex_request_served_standard_is_priced_at_the_standard_rate(self) -> None:
        """With both tiers priced, a flex request served at standard bills the standard rate."""
        state = usage.get_model_state("tierpricemodel")
        state.region = "us-east-1"
        state.service_tier = "flex"
        set_test_price(
            "tierpricemodel", "us-east-1", Dimension.INPUT_TOKENS, "0.000004", "USD"
        )
        _state.price_index[
            PriceKey(
                Service.BEDROCK,
                "tierpricemodel",
                "us-east-1",
                Dimension.INPUT_TOKENS,
                "flex",
            )
        ] = Price(Decimal("0.000001"), "USD")

        ModelBase("tierpricemodel")._record_converse_usage(  # noqa: SLF001
            {  # type: ignore[typeddict-item]
                "usage": {"inputTokens": 1000, "outputTokens": 1, "totalTokens": 1001},
                "serviceTier": {"type": "standard"},
            }
        )
        record = next(iter(usage.USAGE.get().values()))
        assert record.tier == "standard"
        compute_costs()
        # 1000 * 0.000004 (standard rate), not 1000 * 0.000001 (flex rate).
        assert record.cost == Decimal("0.004000")


class TestConverseFailoverUsage:
    """converse() failover must bill exactly once, keyed to the serving region."""

    async def test_throttled_region_is_not_billed_and_failover_records_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A throttled first region leaves one record keyed to the failover region."""
        calls: list[str] = []

        class _FlakyClient:
            """Fake Bedrock client: throttles the first call, succeeds after."""

            def __init__(self, region: str) -> None:
                """Bind the fake client to its target region."""
                self._region = region

            async def converse(self, **_kwargs: object) -> dict[str, Any]:
                """Raise ThrottlingException on the first call, then return usage."""
                calls.append(self._region)
                if len(calls) == 1:
                    raise ClientError(
                        {"Error": {"Code": "ThrottlingException", "Message": "x"}},
                        "Converse",
                    )
                return {
                    "usage": {"inputTokens": 7, "outputTokens": 3, "totalTokens": 10}
                }

        async def _fake_candidates(_model_id: str, **_kwargs: object) -> list[str]:
            return ["us-east-1", "eu-west-1"]

        monkeypatch.setattr(SETTINGS, "aws_bedrock_region_routing", "ordered")
        monkeypatch.setattr(SETTINGS, "aws_bedrock_max_retries", 1)
        monkeypatch.setattr(stdapi.models, "REGION_ROUTER", RegionRouter())
        monkeypatch.setattr(
            stdapi.models, "compute_candidate_regions", _fake_candidates
        )
        monkeypatch.setattr(
            ModelBase, "_prepare_converse_request_for_region", _noop_prepare
        )
        monkeypatch.setattr(
            stdapi.models,
            "bedrock_client",
            lambda region, **_kwargs: _FlakyClient(region),
        )

        # mark_error logs the throttling warning into the request log.
        log_token = REQUEST_LOG.set(
            EventLog(
                type="request",
                level="info",
                date=datetime.now(UTC),
                server_id="test",
                server_version="0.0.0",
            )
        )
        try:
            response = await ModelBase("failovermodel").converse(
                {"modelId": "failovermodel"}
            )
        finally:
            REQUEST_LOG.reset(log_token)

        assert calls == ["us-east-1", "eu-west-1"]
        assert response["usage"]["inputTokens"] == 7
        records = usage.USAGE.get()
        assert len(records) == 1
        key, record = next(iter(records.items()))
        assert key.region == "eu-west-1"
        assert record.quantities[Dimension.INPUT_TOKENS] == 7
        assert record.quantities[Dimension.OUTPUT_TOKENS] == 3
