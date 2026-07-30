"""Billing side of stdapi.models: turning Converse/ConverseStream responses into usage records.

Covers grounding-tool counting (``_count_grounding_tool_uses`` on non-streaming
responses, ``_capture_stream_usage`` on stream events) feeding the
``grounding_requests`` dimension, ``_record_converse_usage`` billing details
(cacheDetails TTL breakdown, effective-tier pricing, per-call region/routing
attribution, prompt-router invoked-model attribution), exactly-once recording
across ``converse()`` failover, and ``_converse`` stripping the
ConverseStream-only guardrail ``streamProcessingMode`` field that non-streaming
Converse rejects. Pricing and log-entry coverage of those dimensions lives in
tests/test_usage.py.

All AWS calls are replaced by fakes: no Bedrock client is contacted.

Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
     stdapi/models/__init__.py:ModelBase._record_converse_usage
"""

from asyncio import Event, create_task, gather, wait_for
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

import pytest
from botocore.exceptions import ClientError
from pydantic_core import to_json

import stdapi.models
from stdapi import usage
from stdapi.config import SETTINGS
from stdapi.models import ModelBase, _count_grounding_tool_uses, _iter_invoke_stream
from stdapi.pricing import Dimension, Price, PriceKey, Service, _state
from stdapi.region_routing import RegionRouter
from stdapi.usage import compute_costs
from tests.conftest import set_test_price

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Generator

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
    )


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
    """_count_grounding_tool_uses: counts billed-grounding toolUse blocks in a Converse response.

    Amazon Nova's ``nova_grounding`` system tool bills per invocation on top of
    inference, so its ``toolUse`` output blocks are counted while ordinary
    client-side tool calls are not.

    Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
         stdapi/models/__init__.py:_count_grounding_tool_uses
    """

    def test_counts_only_billed_grounding_tool_blocks(self) -> None:
        """Two nova_grounding blocks count; a text block and another tool's block do not."""
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
        """A response with no ``output`` key counts zero instead of raising KeyError."""
        assert _count_grounding_tool_uses(cast("ConverseResponseTypeDef", {})) == 0

    def test_empty_content_returns_zero(self) -> None:
        """A response with empty output/message content counts zero."""
        response: dict[str, Any] = {"output": {"message": {"content": []}}}
        assert _count_grounding_tool_uses(response) == 0  # type: ignore[arg-type]


class TestCaptureStreamUsage:
    """_capture_stream_usage: usage is recorded from the stream's trailing metadata event.

    ConverseStream reports usage and the serving tier only on the metadata
    event, so grounding-tool calls seen in earlier contentBlockStart events are
    accumulated and attached to that event's record. Events are passed through
    unchanged.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStreamMetadataEvent.html
         stdapi/models/__init__.py:ModelBase._capture_stream_usage
    """

    async def test_grounding_tool_uses_are_recorded_at_the_metadata_event(self) -> None:
        """Two nova_grounding contentBlockStart events bill 2 grounding requests."""

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
            async for event in model._capture_stream_usage(  # noqa: SLF001
                cast("AsyncIterable[ConverseStreamOutputTypeDef]", fake_stream())
            )
        ]

        # Every upstream event is passed through, in order and unmodified.
        assert events == [
            {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"name": "nova_grounding", "toolUseId": "1"}},
                }
            },
            {
                "contentBlockStart": {
                    "contentBlockIndex": 1,
                    "start": {"toolUse": {"name": "nova_grounding", "toolUseId": "2"}},
                }
            },
            {
                "contentBlockStart": {
                    "contentBlockIndex": 2,
                    "start": {"toolUse": {"name": "other_tool", "toolUseId": "3"}},
                }
            },
            {
                "metadata": {
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
                }
            },
        ]
        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.GROUNDING_REQUESTS] == 2
        assert record.quantities[Dimension.INPUT_TOKENS] == 1

    async def test_grounding_counter_resets_after_each_metadata_event(self) -> None:
        """A second metadata event bills its own tokens without re-billing earlier grounding calls.

        Live-verified (2026-07): AWS emits a single trailing metadata event
        even for multi-round server-tool streams; this guards the defensive
        counter reset should that cardinality ever change. Both events'
        token counts still accumulate into the one record (1 + 3 input tokens).
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
        _ = [
            e
            async for e in model._capture_stream_usage(  # noqa: SLF001
                cast("AsyncIterable[ConverseStreamOutputTypeDef]", fake_stream())
            )
        ]

        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.GROUNDING_REQUESTS] == 1
        assert record.quantities[Dimension.INPUT_TOKENS] == 4
        assert record.quantities[Dimension.OUTPUT_TOKENS] == 5

    async def test_metadata_service_tier_is_billed(self) -> None:
        """The serviceTier reported on the metadata event becomes the record's billed tier.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
        """

        async def fake_stream() -> AsyncIterator[dict[str, Any]]:
            yield {
                "metadata": {
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                    "serviceTier": {"type": "flex"},
                }
            }

        model: ModelBase[Any, Any] = ModelBase("streamtieredmodel")
        _ = [
            e
            async for e in model._capture_stream_usage(  # noqa: SLF001
                cast("AsyncIterable[ConverseStreamOutputTypeDef]", fake_stream())
            )
        ]

        key, record = next(iter(usage.USAGE.get().items()))
        assert key.tier == "flex"
        assert record.tier == "flex"
        assert record.quantities[Dimension.INPUT_TOKENS] == 1

    async def test_closing_stream_early_still_records_trailing_metadata_usage(
        self,
    ) -> None:
        """Closing the wrapper before the metadata event still drains and bills it.

        A client disconnect must not lose the usage AWS already charged for.

        Ref: stdapi/models/__init__.py:_DISCONNECT_DRAIN_MAX_EVENTS
        """

        async def fake_stream() -> AsyncIterator[dict[str, Any]]:
            yield {"contentBlockDelta": {"delta": {"text": "hello"}}}
            yield {"contentBlockDelta": {"delta": {"text": " world"}}}
            yield {
                "metadata": {
                    "usage": {"inputTokens": 3, "outputTokens": 4, "totalTokens": 7}
                }
            }

        model: ModelBase[Any, Any] = ModelBase("disconnectmodel")
        wrapped = model._capture_stream_usage(  # noqa: SLF001
            cast("AsyncIterable[ConverseStreamOutputTypeDef]", fake_stream())
        )
        # Simulate a client disconnect: only the first event is consumed.
        await anext(wrapped)
        await wrapped.aclose()

        record = next(iter(usage.USAGE.get().values()))
        assert record.quantities[Dimension.INPUT_TOKENS] == 3
        assert record.quantities[Dimension.OUTPUT_TOKENS] == 4


class TestIterInvokeStream:
    """_iter_invoke_stream: the usage callback fires only for metrics-carrying chunks.

    InvokeModelWithResponseStream reports token counts in a single chunk
    carrying ``amazon-bedrock-invocationMetrics``; parsing every other chunk for
    usage would bill repeatedly or on absent counters.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html
         stdapi/models/__init__.py:_iter_invoke_stream
    """

    async def test_usage_callback_skipped_for_chunks_without_invocation_metrics(
        self,
    ) -> None:
        """Only the chunk carrying invocationMetrics invokes the callback; all chunks pass through."""

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
        assert calls[0]["amazon-bedrock-invocationMetrics"] == {"inputTokenCount": 1}


class TestRecordConverseUsageEffectiveTier:
    """_record_converse_usage: the response's serviceTier overrides the requested one.

    AWS echoes the tier that actually served the call, and a request can be
    served on a different tier than asked for (Reserved overflows to Standard),
    so billing follows the response rather than the request.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
         stdapi/models/__init__.py:ModelBase._record_converse_usage
    """

    def test_response_service_tier_overrides_requested_tier(self) -> None:
        """A flex request served at standard is billed as standard."""
        usage.get_model_state("tieredmodel").service_tier = "flex"
        ModelBase("tieredmodel")._record_converse_usage(  # noqa: SLF001
            {  # type: ignore[typeddict-item]
                "usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6},
                "serviceTier": {"type": "standard"},
            }
        )
        assert next(iter(usage.USAGE.get())).tier == "standard"

    def test_requested_tier_is_used_when_response_reports_none(self) -> None:
        """With no serviceTier in the response, the requested tier is billed."""
        usage.get_model_state("tieredmodel").service_tier = "priority"
        ModelBase("tieredmodel")._record_converse_usage(  # noqa: SLF001
            {"usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6}}  # type: ignore[typeddict-item]
        )
        assert next(iter(usage.USAGE.get())).tier == "priority"


class TestRecordConverseUsagePromptRouterAttribution:
    """_record_converse_usage: bill the prompt router's actually-invoked model.

    A prompt router picks a target model per request and reports it in
    ``trace.promptRouter.invokedModelId``; the models differ in price, so usage
    is keyed to the invoked model whenever it maps to a known catalog entry.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         stdapi/models/__init__.py:_invoked_model_id
    """

    def test_bills_the_trace_invoked_model_when_known(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Usage is keyed to promptRouter.invokedModelId, not to the router's own ID."""
        monkeypatch.setitem(
            stdapi.models._ALL_MODELS,  # noqa: SLF001
            "anthropic.claude-3-haiku-20240307-v1:0",
            cast("Any", object()),
        )
        ModelBase("my-router")._record_converse_usage(  # noqa: SLF001
            {  # type: ignore[typeddict-item]
                "usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6},
                "trace": {
                    "promptRouter": {
                        "invokedModelId": (
                            "arn:aws:bedrock:us-east-1::foundation-model/"
                            "anthropic.claude-3-haiku-20240307-v1:0"
                        )
                    }
                },
            }
        )
        (key,) = usage.USAGE.get()
        assert key.model == "anthropic.claude-3-haiku-20240307-v1:0"
        assert usage.USAGE.get()[key].quantities[Dimension.INPUT_TOKENS] == 5

    def test_bills_the_base_model_of_an_invoked_inference_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An invoked inference-profile ARN bills its base model, geography prefix stripped.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
        """
        monkeypatch.setitem(
            stdapi.models._ALL_MODELS,  # noqa: SLF001
            "anthropic.claude-3-haiku-20240307-v1:0",
            cast("Any", object()),
        )
        ModelBase("my-router")._record_converse_usage(  # noqa: SLF001
            {  # type: ignore[typeddict-item]
                "usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6},
                "trace": {
                    "promptRouter": {
                        "invokedModelId": (
                            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/"
                            "us.anthropic.claude-3-haiku-20240307-v1:0"
                        )
                    }
                },
            }
        )
        (key,) = usage.USAGE.get()
        assert key.model == "anthropic.claude-3-haiku-20240307-v1:0"
        assert usage.USAGE.get()[key].quantities[Dimension.INPUT_TOKENS] == 5

    def test_falls_back_to_static_model_id_when_invoked_model_unknown(self) -> None:
        """An invokedModelId naming an unknown model bills the requested model instead."""
        ModelBase("my-router")._record_converse_usage(  # noqa: SLF001
            {  # type: ignore[typeddict-item]
                "usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6},
                "trace": {
                    "promptRouter": {
                        "invokedModelId": (
                            "arn:aws:bedrock:us-east-1::foundation-model/"
                            "unregistered-fake-model"
                        )
                    }
                },
            }
        )
        (key,) = usage.USAGE.get()
        assert key.model == "my-router"
        assert usage.USAGE.get()[key].quantities[Dimension.INPUT_TOKENS] == 5

    def test_falls_back_to_static_model_id_without_trace(self) -> None:
        """A plain (non-router) Converse response bills the requested model."""
        ModelBase("plain-model")._record_converse_usage(  # noqa: SLF001
            {"usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6}}  # type: ignore[typeddict-item]
        )
        (key,) = usage.USAGE.get()
        assert key.model == "plain-model"
        assert usage.USAGE.get()[key].quantities[Dimension.INPUT_TOKENS] == 5


class TestPerCallAttribution:
    """Explicit per-call region/routing wins over the shared model state.

    Concurrent same-model calls share one ModelInvocationState: a sibling
    call may overwrite it between invocation and recording, so recording
    relies on the explicitly threaded per-call values instead.

    Ref: stdapi/usage.py:ModelInvocationState
         stdapi/models/__init__.py:ModelBase._record_converse_usage
    """

    def test_explicit_region_and_routing_win_over_model_state(self) -> None:
        """A record carries its own call's region and routing, not the shared state's."""
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
    """Two in-flight _converse calls each bill their own serving Region.

    Ref: stdapi/models/__init__.py:ModelBase._converse
         stdapi/usage.py:ModelInvocationState
    """

    async def test_concurrent_calls_bill_two_distinct_regions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sibling overwrite of the shared model state reaches neither in-flight record.

        Both fake calls are held open until a third Region is written into the
        shared state, so a record built from that state would be misattributed.
        """
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


class TestConverseGuardrailStreamProcessingMode:
    """_converse strips streamProcessingMode, which the non-streaming Converse API rejects.

    streamProcessingMode exists only on ConverseStream's
    GuardrailStreamConfiguration, so the shared guardrail config must be
    trimmed before a non-streaming call.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html
         stdapi/models/__init__.py:ModelBase._converse
    """

    async def test_stream_processing_mode_is_stripped_before_the_converse_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """StreamProcessingMode is dropped while the rest of guardrailConfig is passed through."""
        captured: dict[str, Any] = {}

        class _CapturingClient:
            """Fake Bedrock client recording the kwargs passed to converse()."""

            async def converse(self, **kwargs: object) -> dict[str, Any]:
                """Record kwargs and return a minimal usage-bearing response."""
                captured.update(kwargs)
                return {
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
                }

        monkeypatch.setattr(
            ModelBase, "_prepare_converse_request_for_region", _noop_prepare
        )
        monkeypatch.setattr(
            stdapi.models,
            "bedrock_client",
            lambda _region, **_kwargs: _CapturingClient(),
        )

        model: ModelBase[Any, Any] = ModelBase("guardrailmodel")
        await model._converse(  # noqa: SLF001
            {
                "modelId": "guardrailmodel",
                "guardrailConfig": {
                    "guardrailIdentifier": "gr1",
                    "guardrailVersion": "1",
                    "streamProcessingMode": "async",
                },
            },
            "us-east-1",
            single_region=True,
        )

        assert captured["guardrailConfig"] == {
            "guardrailIdentifier": "gr1",
            "guardrailVersion": "1",
        }


class TestRecordConverseUsageCacheDetails:
    """_record_converse_usage: cacheDetails feeds the per-TTL cache-write breakdown.

    Bedrock reports cache reads and cache writes separately from
    ``inputTokens`` (which excludes both), so each lands in its own dimension
    and the ``cacheDetails`` TTL split becomes the cache-write breakdown.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
         stdapi/models/__init__.py:ModelBase._record_converse_usage
    """

    def test_cache_details_populate_the_ttl_breakdown(self) -> None:
        """Each cacheDetails entry maps its ttl to its inputTokens, leaving inputTokens alone."""
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
        # inputTokens excludes the cache writes: it is billed as-is, and no
        # cache-read dimension appears when AWS reported none.
        assert record.quantities[Dimension.INPUT_TOKENS] == 10
        assert Dimension.CACHE_READ_TOKENS not in record.quantities

    def test_malformed_cache_details_entries_are_skipped(self) -> None:
        """CacheDetails entries missing "ttl" or "inputTokens" are dropped, not raised on."""
        ModelBase("cachedetailmalformedmodel")._record_converse_usage(  # noqa: SLF001
            {  # type: ignore[typeddict-item]
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 1,
                    "totalTokens": 11,
                    "cacheWriteInputTokens": 500,
                    "cacheDetails": [
                        {"ttl": "5m", "inputTokens": 500},
                        {"ttl": "1h"},
                        {"inputTokens": 200},
                    ],
                }
            }
        )
        record = next(iter(usage.USAGE.get().values()))
        assert record.cache_write_tokens_by_ttl == {"5m": 500}
        # The dropped entries leave the flat cache-write total unchanged.
        assert record.quantities[Dimension.CACHE_WRITE_TOKENS] == 500


class TestEffectiveTierPricing:
    """The response-reported tier drives pricing, not the requested one.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
         stdapi/pricing.py:resolve_price
    """

    def test_flex_request_served_standard_is_priced_at_the_standard_rate(self) -> None:
        """With both tiers priced, a flex request served at standard bills 1000 * 0.000004."""
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
    """converse() failover bills exactly once, keyed to the serving Region.

    A ThrottlingException (HTTP 429) is retried in the next candidate Region;
    the throttled attempt produced no tokens and must leave no record behind.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html
         stdapi/models/__init__.py:ModelBase.converse
    """

    @pytest.mark.usefixtures("request_log")
    async def test_throttled_region_is_not_billed_and_failover_records_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A throttled first Region leaves a single record keyed to the failover Region."""
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
        response = await ModelBase("failovermodel").converse(
            {"modelId": "failovermodel"}
        )

        assert calls == ["us-east-1", "eu-west-1"]
        assert response["usage"]["inputTokens"] == 7
        records = usage.USAGE.get()
        assert len(records) == 1
        key, record = next(iter(records.items()))
        assert key.region == "eu-west-1"
        assert record.quantities[Dimension.INPUT_TOKENS] == 7
        assert record.quantities[Dimension.OUTPUT_TOKENS] == 3
