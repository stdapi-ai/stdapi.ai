"""Request log in stdapi.monitoring: usage, cost, error capture and caller identity.

Everything here runs in-process against a seeded price index and patched
Bedrock seams, so the token counts, currencies and log levels asserted below are
exact rather than indicative.

Ref: stdapi/monitoring.py:_finalize_usage
     stdapi/monitoring.py:log_request_event
     stdapi/monitoring.py:_rebuild_and_log_stream
     stdapi/monitoring.py:resolve_request_identity
"""

from asyncio import create_task, sleep
from contextvars import copy_context
from gc import collect
from json import loads
from logging import ERROR
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from botocore.exceptions import ClientError
from botocore.exceptions import ConnectionError as BotocoreConnectionError
from opentelemetry.trace import get_current_span
from pydantic import SecretStr
from sse_starlette import ServerSentEvent
from starlette.requests import Request as StarletteRequest

import stdapi.auth
from stdapi import monitoring, usage
from stdapi.api_errors import ApiError
from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import AuthenticationHandler, authenticate
from stdapi.config import SETTINGS
from stdapi.models import ModelBase
from stdapi.monitoring import (
    PRINCIPAL,
    REQUEST,
    REQUEST_ID,
    REQUEST_LOG,
    EventLog,
    Principal,
    SseHandledStreamError,
    _finalize_usage,
    build_metadata,
    log_background_event,
    log_request_event,
    otel_manager,
    resolve_request_identity,
)
from stdapi.pricing import Dimension
from stdapi.usage import get_model_state, record_bedrock_usage
from tests._helpers import make_event_log
from tests.conftest import set_test_price

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterable,
        Callable,
        Coroutine,
        Generator,
    )
    from typing import Any

    from opentelemetry.trace.span import Span
    from types_aiobotocore_bedrock_runtime.type_defs import ConverseStreamOutputTypeDef

    from stdapi.config import LogLevel


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


def _make_request(method: str = "GET", path: str = "/test") -> StarletteRequest:
    """Build a minimal Starlette request for testing log_request_event."""
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": [],
    }
    return StarletteRequest(scope)


def _openai_tagged_request() -> StarletteRequest:
    """Build a request whose matched route is tagged OpenAI.

    The error envelope is picked from the resolved route's tags, so a terminal
    SSE error event only takes the OpenAI shape when a tagged route is active.
    """
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "query_string": b"",
        "headers": [],
        "route": SimpleNamespace(tags=[TAG_OPENAI]),
    }
    return StarletteRequest(scope)


def _boom(log: EventLog) -> None:
    """Stand-in for a raising ``_finalize_usage``."""
    message = f"boom for {log.get('type')}"
    raise RuntimeError(message)


def _capture_costed_logs(monkeypatch: pytest.MonkeyPatch) -> list[EventLog]:
    """Enable cost tracking, seed a standard-tier price, and capture written logs."""
    written: list[EventLog] = []
    monkeypatch.setattr(monitoring, "write_log_event", written.append)
    monkeypatch.setattr(SETTINGS, "cost_tracking", True)
    set_test_price("modela", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD")
    return written


def _install_converse_seams(
    monkeypatch: pytest.MonkeyPatch, response: dict[str, Any]
) -> list[EventLog]:
    """Patch the region/prepare/client seams so ModelBase.converse runs offline."""
    written = _capture_costed_logs(monkeypatch)

    async def candidates(_model_id: str, **_kwargs: object) -> list[str]:
        """Pin routing to a single region, bypassing the region router."""
        return ["us-east-1"]

    async def prepare(
        _self: ModelBase[Any, Any], _request: object, _region: str
    ) -> None:
        """No-op replacement for the region-specific request preparation."""

    class FakeBedrockClient:
        """Minimal Bedrock runtime client exposing only converse()."""

        async def converse(self, **_kwargs: object) -> dict[str, Any]:
            """Return the canned Converse response."""
            return response

    monkeypatch.setattr("stdapi.models.compute_candidate_regions", candidates)
    monkeypatch.setattr(ModelBase, "_prepare_converse_request_for_region", prepare)
    monkeypatch.setattr(
        "stdapi.models.bedrock_client", lambda _region, **_kwargs: FakeBedrockClient()
    )
    return written


async def _wait_for_flag(flag: list[bool]) -> None:
    """Let the event loop run asyncio's async-generator finalizers until *flag* is set."""
    for _ in range(100):
        if flag:
            return
        collect()
        await sleep(0)


@pytest.fixture(autouse=True)
def _reset_model_state() -> Generator[None]:
    """Reset the per-model invocation state so tests don't leak it via execution order."""
    token = usage.init_model_state()
    yield
    usage.MODEL_STATE.reset(token)


class TestPublishedLogLevels:
    """_published_log_levels() turns a minimum severity into the publishable set.

    Ref: stdapi/monitoring.py:_published_log_levels
    """

    def test_disabled_publishes_nothing(self) -> None:
        """``"disabled"`` yields an empty set instead of failing the index lookup.

        ``"disabled"`` is not a member of ``LogLevel``, so it has to be
        special-cased before the severity slice.
        """
        assert monitoring._published_log_levels("disabled") == set()  # noqa: SLF001

    def test_warning_publishes_warning_and_above(self) -> None:
        """A minimum level keeps itself and every more severe level, and drops the rest."""
        levels = monitoring._published_log_levels("warning")  # noqa: SLF001
        assert levels == {"warning", "error", "critical"}
        assert "info" not in levels


@pytest.mark.usefixtures("usage_scope")
class TestFinalizeUsage:
    """_finalize_usage() turns the request's usage accumulator into log usage/cost fields.

    Ref: stdapi/monitoring.py:_finalize_usage
         stdapi/usage.py:record_bedrock_usage
    """

    def test_populates_usage_and_cost_from_a_single_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single billed record yields a per-entry cost and an identical request total.

        1000 input tokens at 0.000003 per token is 0.003, formatted as an exact
        decimal string rather than a float.
        """
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        get_model_state("modela").region = "us-east-1"
        set_test_price("modela", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD")
        record_bedrock_usage("modela", input_tokens=1000)

        log = make_event_log()
        _finalize_usage(log)

        assert log["usage"][0]["cost"] == "0.003"
        assert log["cost"] == {"USD": "0.003"}

    def test_aggregates_cost_across_multiple_records_in_one_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two models billed in one request keep separate entries but one summed total.

        0.003 + 0.005 must be added in decimal: the total is the sum of the
        per-entry costs, not a re-derivation from one model's rate.
        """
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        get_model_state("modela").region = "us-east-1"
        get_model_state("modelb").region = "us-east-1"
        set_test_price("modela", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD")
        set_test_price("modelb", "us-east-1", Dimension.INPUT_TOKENS, "0.000005", "USD")
        record_bedrock_usage("modela", input_tokens=1000)
        record_bedrock_usage("modelb", input_tokens=1000)

        log = make_event_log()
        _finalize_usage(log)

        assert len(log["usage"]) == 2
        assert {entry["cost"] for entry in log["usage"]} == {"0.003", "0.005"}
        assert log["cost"] == {"USD": "0.008"}

    def test_multi_currency_costs_entries_fold_into_the_request_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A record priced in two currencies reports a ``costs`` breakdown and both totals.

        Nothing is converted: a single entry whose dimensions are published in USD
        and EUR keeps each currency separate all the way to the request total, so
        no single ``cost`` string can represent it.
        """
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        get_model_state("modela").region = "us-east-1"
        set_test_price("modela", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD")
        set_test_price(
            "modela", "us-east-1", Dimension.OUTPUT_TOKENS, "0.000015", "EUR"
        )
        record_bedrock_usage("modela", input_tokens=1000, output_tokens=1000)

        log = make_event_log()
        _finalize_usage(log)

        assert log["usage"][0]["costs"] == {"USD": "0.003", "EUR": "0.015"}
        assert "cost" not in log["usage"][0]
        assert log["cost"] == {"USD": "0.003", "EUR": "0.015"}
        assert log["level"] == "warning"
        assert any(
            "Multiple currencies resolved" in str(detail)
            for detail in log["error_detail"]
        )

    def test_no_usage_leaves_log_fields_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request with no billed usage adds neither ``usage`` nor ``cost`` to the log.

        The fields are set only when there are entries, so non-model routes keep a
        minimal log record.
        """
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)

        log = make_event_log()
        _finalize_usage(log)

        assert "usage" not in log
        assert "cost" not in log

    def test_cost_tracking_disabled_still_logs_usage_without_cost(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With cost_tracking off, token quantities are still logged but no cost is computed.

        Usage accounting is independent of pricing: the entry keeps its counters
        and simply carries no cost field, at either level.
        """
        monkeypatch.setattr(SETTINGS, "cost_tracking", False)
        get_model_state("modela").region = "us-east-1"
        record_bedrock_usage("modela", input_tokens=1000)

        log = make_event_log()
        _finalize_usage(log)

        assert log["usage"][0]["input_tokens"] == 1000
        assert "cost" not in log["usage"][0]
        assert "cost" not in log

    def test_catalog_not_ready_omits_cost_without_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Before the price catalog has loaded, cost is skipped and the log stays ``info``.

        The (autouse) _clean_price_index fixture leaves the index empty, so
        price_catalog_ready() is False -- the exact state during the startup
        window before the background catalog load completes. Warning on every
        record served in that window would be pure noise.

        Ref: stdapi/usage.py:compute_costs
             stdapi/pricing.py:price_catalog_ready
        """
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        get_model_state("modela").region = "us-east-1"
        record_bedrock_usage("modela", input_tokens=1000)

        log = make_event_log()
        _finalize_usage(log)

        assert log["usage"][0]["input_tokens"] == 1000
        assert "cost" not in log["usage"][0]
        assert "cost" not in log
        assert "error_detail" not in log
        assert log["level"] == "info"

    def test_catalog_ready_genuine_miss_still_warns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once the catalog is ready, a real per-model pricing miss warns and names the model.

        Readiness is global, not per model: seeding any price makes the catalog
        ready, so a second model with no rows is a genuine miss rather than a
        startup artefact.

        Ref: stdapi/usage.py:_apply_record_cost
        """
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        get_model_state("modela").region = "us-east-1"
        get_model_state("modelb").region = "us-east-1"
        # Seeds an unrelated price so the catalog counts as ready.
        set_test_price("modela", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD")
        record_bedrock_usage("modelb", input_tokens=1000)

        log = make_event_log()
        _finalize_usage(log)

        assert log["level"] == "warning"
        assert "No price found for bedrock-runtime/modelb" in str(log["error_detail"])
        assert "cost" not in log["usage"][0]
        assert "cost" not in log


class TestFinalizeUsageFailureIsolation:
    """A raising _finalize_usage never fails the request, drops its log, or leaks context.

    Cost accounting runs in the middleware's ``finally`` block, after the
    response body has been produced, so a pricing bug there must stay contained.

    Ref: stdapi/monitoring.py:_finalize_usage_safely
         stdapi/monitoring.py:log_request_event
    """

    def test_finalize_usage_failure_does_not_fail_the_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A finalize exception does not propagate out of log_request_event.

        The status code recorded inside the scope survives, so the client still
        gets its successful response.
        """
        monkeypatch.setattr(monitoring, "_finalize_usage", _boom)
        written: list[EventLog] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)

        request = _make_request()
        with log_request_event(request) as log:
            log["status_code"] = 200

        assert log["status_code"] == 200

    def test_finalize_usage_failure_is_recorded_and_log_still_written(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure is recorded in error_detail, raises the level, and the log is written.

        Swallowing the exception silently would make cost-tracking regressions
        invisible, so the traceback is folded into the request log instead.
        """
        monkeypatch.setattr(monitoring, "_finalize_usage", _boom)
        written: list[EventLog] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)

        request = _make_request()
        with log_request_event(request) as log:
            pass

        assert written == [log]
        assert log["level"] in ("error", "critical")
        assert any("boom" in str(detail) for detail in log["error_detail"])

    def test_finalize_usage_failure_still_resets_request_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-request ContextVars are reset even when _finalize_usage raises.

        ``_reset_request_context`` runs after the finalize in the same ``finally``
        block; a leak here would let the next request inherit this one's
        correlation ID and usage accumulator.

        Restoration is asserted against the values captured before the request
        rather than against ``LookupError``: resetting a token restores whatever
        was set beforehand, so an ambient value from an earlier test in the same
        worker would make an unconditional ``LookupError`` expectation fail.

        Ref: stdapi/monitoring.py:_reset_request_context
        """
        monkeypatch.setattr(monitoring, "_finalize_usage", _boom)
        outer_usage = usage.USAGE.get(None)
        outer_request_id = REQUEST_ID.get(None)
        outer_request_log = REQUEST_LOG.get(None)

        request = _make_request()
        with log_request_event(request):
            assert REQUEST_ID.get()
            assert REQUEST_LOG.get()["type"] == "request"
            inner_request_id = REQUEST_ID.get()

        assert inner_request_id != outer_request_id
        assert REQUEST_ID.get(None) == outer_request_id
        assert REQUEST_LOG.get(None) is outer_request_log
        assert usage.USAGE.get(None) is outer_usage


class TestStreamUsageContinuity:
    """The stream log picks up usage recorded after the request log finalized.

    Drain-style tasks (streamed /v1/completions, parallel image jobs) are
    created before the stream log scope exists and capture the request
    context: the request finalize drains what it logged, and the shared
    accumulator carries later records into the ``request_stream`` log.

    Ref: stdapi/monitoring.py:_finalize_usage
         stdapi/monitoring.py:_rebuild_and_log_stream
    """

    async def test_request_finalize_drains_records(self) -> None:
        """The request finalize empties the accumulator so its records cannot be billed twice."""
        token = usage.init_usage()
        try:
            record_bedrock_usage("drainedmodel", input_tokens=5, total_tokens=5)
            log = make_event_log()
            _finalize_usage(log)
            assert log["usage"]
            assert not usage.USAGE.get()
        finally:
            usage.USAGE.reset(token)

    async def test_drain_survives_a_failing_finalize(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A finalize failure still drains, so a later stream log cannot re-bill the records.

        The drain lives in ``_finalize_usage``'s ``finally``, past the metrics
        emission: a failing ``emit_usage_metrics`` propagates but must not leave
        the accumulator populated.
        """
        token = usage.init_usage()
        try:
            record_bedrock_usage("failfinalizemodel", input_tokens=5, total_tokens=5)

            def _boom() -> None:
                message = "emit failed"
                raise RuntimeError(message)

            monkeypatch.setattr(monitoring, "emit_usage_metrics", _boom)
            with pytest.raises(RuntimeError, match=r"^emit failed$"):
                _finalize_usage(make_event_log())
            assert not usage.USAGE.get()
        finally:
            usage.USAGE.reset(token)

    async def test_late_task_usage_lands_in_stream_log(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Usage recorded by a pre-stream-scope task lands in the stream log, not the request log.

        The recorder task is created while the first chunk is produced — before
        the ``request_stream`` scope exists — so the request finalize sees nothing
        and the shared accumulator carries the record forward.
        """
        written: list[EventLog] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)
        usage_token = usage.init_usage()
        id_token = REQUEST_ID.set("test-request-id")
        try:

            async def recorder() -> None:
                record_bedrock_usage(
                    "latemodel", input_tokens=7, output_tokens=3, total_tokens=10
                )

            first_event = ServerSentEvent(data="first")
            second_event = ServerSentEvent(data="second")

            async def source() -> AsyncGenerator[ServerSentEvent]:
                # Mimics format_stream: the task is created while producing
                # the first chunk, before the stream log scope exists.
                task = create_task(recorder())
                yield first_event
                await task
                yield second_event

            stream = monitoring.log_request_sse_stream_event(source())
            assert await stream.__anext__() is first_event
            # Middleware finalizes (and drains) the request log at this point.
            request_log = make_event_log()
            _finalize_usage(request_log)
            assert "usage" not in request_log
            _ = [chunk async for chunk in stream]

            stream_logs = [w for w in written if w["type"] == "request_stream"]
            assert len(stream_logs) == 1
            (entry,) = stream_logs[0]["usage"]
            assert entry["input_tokens"] == 7
            assert entry["output_tokens"] == 3
        finally:
            REQUEST_ID.reset(id_token)
            usage.USAGE.reset(usage_token)


class TestRequestPipelineCost:
    """ModelBase.converse inside log_request_event lands billed cost in the request log.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
         stdapi/models/__init__.py:ModelBase.converse
    """

    async def test_converse_usage_costed_in_request_log(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Converse response's billed tokens surface as a costed entry with region and tier.

        The response reports ``serviceTier.type = standard``, and the region comes
        from the routing decision rather than the response, so both are recorded
        on the usage entry that pricing keys off.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
        """
        response: dict[str, Any] = {
            "usage": {"inputTokens": 1000, "outputTokens": 0, "totalTokens": 1000},
            "output": {"message": {"content": [{"text": "hi"}]}},
            "serviceTier": {"type": "standard"},
        }
        written = _install_converse_seams(monkeypatch, response)

        with log_request_event(_make_request()) as log:
            await ModelBase("modela").converse({"modelId": "modela"})

        assert written == [log]
        (entry,) = log["usage"]
        assert entry["cost"] == "0.003"
        assert entry["region"] == "us-east-1"
        assert entry["tier"] == "standard"
        assert log["cost"] == {"USD": "0.003"}

    async def test_requested_flex_tier_billed_when_response_omits_tier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no serviceTier echoed back, the tier requested on the Converse call is billed.

        Bedrock may omit ``serviceTier`` from the response, so the gateway falls
        back to the tier it asked for instead of silently billing standard.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
        """
        response: dict[str, Any] = {
            "usage": {"inputTokens": 1000, "outputTokens": 0, "totalTokens": 1000},
            "output": {"message": {"content": [{"text": "hi"}]}},
        }
        written = _install_converse_seams(monkeypatch, response)

        with log_request_event(_make_request()) as log:
            await ModelBase("modela").converse(
                {"modelId": "modela", "serviceTier": {"type": "flex"}}
            )

        assert written == [log]
        (entry,) = log["usage"]
        assert entry["tier"] == "flex"
        assert entry["input_tokens"] == 1000


class TestStreamLogCost:
    """Usage captured by _capture_stream_usage is costed in the request_stream log.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStreamMetadataEvent.html
         stdapi/models/__init__.py:ModelBase._capture_stream_usage
         stdapi/monitoring.py:log_request_stream_event
    """

    async def test_stream_metadata_usage_costed_in_stream_log(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Usage from the stream's trailing metadata event lands costed in the stream log.

        ConverseStream reports token usage only in its final ``metadata`` event,
        so the wrapper must keep accumulating past the content deltas — and pass
        every chunk through unchanged while doing so.
        """
        written = _capture_costed_logs(monkeypatch)
        usage_token = usage.init_usage()
        id_token = REQUEST_ID.set("test-request-id")
        try:

            async def source() -> AsyncGenerator[dict[str, Any]]:
                yield {"contentBlockDelta": {"delta": {"text": "a"}}}
                yield {"contentBlockDelta": {"delta": {"text": "b"}}}
                yield {
                    "metadata": {
                        "usage": {
                            "inputTokens": 1000,
                            "outputTokens": 0,
                            "totalTokens": 1000,
                        }
                    }
                }

            wrapped = ModelBase("modela")._capture_stream_usage(  # noqa: SLF001
                cast("AsyncIterable[ConverseStreamOutputTypeDef]", source()),
                region="us-east-1",
            )
            stream = await monitoring.log_request_stream_event(wrapped)
            chunks = [chunk async for chunk in stream]

            assert len(chunks) == 3
            (stream_log,) = [w for w in written if w["type"] == "request_stream"]
            (entry,) = stream_log["usage"]
            assert entry["input_tokens"] == 1000
            assert entry["cost"] == "0.003"
            assert entry["region"] == "us-east-1"
            assert stream_log["cost"] == {"USD": "0.003"}
        finally:
            REQUEST_ID.reset(id_token)
            usage.USAGE.reset(usage_token)


class TestStreamClientDisconnect:
    """A client disconnect mid-stream never raises and always closes the Bedrock source.

    ``aclose()`` on the wrapper stands in for the client going away; the
    underlying event stream must be closed so the connection is released.

    Ref: stdapi/monitoring.py:_rebuild_and_log_stream
    """

    @staticmethod
    def _source(finalized: list[bool]) -> AsyncGenerator[dict[str, Any]]:
        """Fake Bedrock event stream flagging *finalized* when its finally runs."""

        async def generate() -> AsyncGenerator[dict[str, Any]]:
            try:
                yield {"contentBlockDelta": {"delta": {"text": "a"}}}
                yield {"contentBlockDelta": {"delta": {"text": "b"}}}
                yield {"metadata": {"usage": {"inputTokens": 1000}}}
            finally:
                finalized.append(True)

        return generate()

    async def test_disconnect_mid_stream_logs_the_usage_it_drained(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closing after the stream log scope opened bills what the drain recovered.

        The consumer left before the metadata event, but Bedrock had already
        produced and billed it, so the wrapper drains it on close. That only
        reaches the client's accounting if the source is closed before the log
        entry is finalised, and the source generator's ``finally`` still runs.
        """
        written = _capture_costed_logs(monkeypatch)
        usage_token = usage.init_usage()
        id_token = REQUEST_ID.set("test-request-id")
        finalized: list[bool] = []
        try:
            wrapped = ModelBase("modela")._capture_stream_usage(  # noqa: SLF001
                cast(
                    "AsyncIterable[ConverseStreamOutputTypeDef]",
                    self._source(finalized),
                ),
                region="us-east-1",
            )
            stream = await monitoring.log_request_stream_event(wrapped)
            assert await stream.__anext__()
            assert await stream.__anext__()  # Enters the logged streaming loop.
            await stream.aclose()

            (stream_log,) = [w for w in written if w["type"] == "request_stream"]
            (entry,) = stream_log["usage"]
            assert entry["input_tokens"] == 1000
            assert stream_log["level"] == "info"
            await _wait_for_flag(finalized)
            assert finalized == [True]
        finally:
            REQUEST_ID.reset(id_token)
            usage.USAGE.reset(usage_token)

    async def test_disconnect_on_first_chunk_closes_source_without_stream_log(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closing right after the first chunk closes the source without opening a stream log.

        The first chunk is yielded before any logging setup runs, so a client that
        leaves immediately produces no ``request_stream`` entry at all — while the
        source generator is still finalised.
        """
        written = _capture_costed_logs(monkeypatch)
        usage_token = usage.init_usage()
        id_token = REQUEST_ID.set("test-request-id")
        finalized: list[bool] = []
        try:
            wrapped = ModelBase("modela")._capture_stream_usage(  # noqa: SLF001
                cast(
                    "AsyncIterable[ConverseStreamOutputTypeDef]",
                    self._source(finalized),
                ),
                region="us-east-1",
            )
            stream = await monitoring.log_request_stream_event(wrapped)
            assert await stream.__anext__()
            await stream.aclose()

            assert [w for w in written if w["type"] == "request_stream"] == []
            await _wait_for_flag(finalized)
            assert finalized == [True]
        finally:
            REQUEST_ID.reset(id_token)
            usage.USAGE.reset(usage_token)


class TestSseHandledStreamErrorLevel:
    """SseHandledStreamError.__init__: level defaults from status; explicit override wins.

    The marker exception is raised after the adapter already emitted spec-
    compliant error events, so it is never "critical": a 4xx is the client's
    fault (warning) and anything else is an error.

    Ref: stdapi/monitoring.py:SseHandledStreamError
    """

    @pytest.mark.parametrize(
        ("status", "level", "expected"),
        [
            (400, None, "warning"),
            (500, None, "error"),
            (None, None, "error"),
            (500, "warning", "warning"),
        ],
    )
    def test_level_defaults_from_status_unless_overridden(
        self, status: int | None, level: LogLevel | None, expected: LogLevel
    ) -> None:
        """The resolved level follows the status-based default unless *level* is given."""
        exc = SseHandledStreamError("boom", status=status, level=level)
        assert exc.level == expected
        assert exc.status == status
        assert exc.args == ("boom",)


class TestMidStreamErrorLoggedInStreamEvent:
    """A mid-stream error lands in the request_stream log instead of the request log.

    Only the ``SseHandledStreamError`` branch is covered here; the branches that
    additionally yield a terminal ``error`` SSE event live in
    :class:`TestMidStreamTerminalErrorEvent`.

    Ref: stdapi/monitoring.py:log_request_sse_stream_event
         stdapi/monitoring.py:_stream_exception_detail
    """

    @pytest.mark.usefixtures("request_log")
    async def test_sse_handled_stream_error_logs_message_and_warning_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mid-stream SseHandledStreamError(status=400) is logged as a warning and swallowed.

        The adapter already emitted its own protocol-compliant error events, so
        the wrapper must not append a REST-envelope ``error`` event on top: the
        consumer sees only the chunk produced before the failure.
        """
        written: list[EventLog] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)
        id_token = REQUEST_ID.set("test-request-id")
        try:

            async def source() -> AsyncGenerator[ServerSentEvent]:
                yield ServerSentEvent(data="first")
                message = "bad request"
                raise SseHandledStreamError(message, status=400)

            stream = monitoring.log_request_sse_stream_event(source())
            events = [chunk async for chunk in stream]

            assert len(events) == 1
            assert events[0].data == "first"
            (stream_log,) = [w for w in written if w["type"] == "request_stream"]
            assert stream_log["level"] == "warning"
            assert any(
                "bad request" in str(detail) for detail in stream_log["error_detail"]
            )
        finally:
            REQUEST_ID.reset(id_token)


#: An ARN and an account ID, the two identifiers AWS messages routinely embed.
_LEAKY_DETAIL = (
    "Operation not allowed on "
    "arn:aws:bedrock:us-east-1:123456789012:inference-profile/secret in 123456789012"
)


class TestMidStreamTerminalErrorEvent:
    """A mid-stream failure closes the SSE stream with a provider-formatted error event.

    Once the response headers are on the wire the gateway can no longer answer
    with an HTTP error, so the wrapper turns the exception into a terminal
    ``error`` event in the envelope the matched route's provider uses. What that
    event may say is the security contract: an AWS message reaches the client
    only after ``hide_security_details`` redacts it, and a 5xx is replaced by a
    fixed sentence so backend internals never leave the process.

    Ref: stdapi/monitoring.py:log_request_sse_stream_event
         stdapi/utils.py:hide_security_details
         stdapi/aws_bedrock.py:AWS_ERROR_MAP
    """

    @staticmethod
    async def _run(
        monkeypatch: pytest.MonkeyPatch, exc: BaseException
    ) -> tuple[list[ServerSentEvent], EventLog]:
        """Fail a one-chunk SSE stream with *exc* and return its events and stream log.

        Args:
            monkeypatch: Fixture used to capture the written log events.
            exc: Exception raised after the first chunk was produced.

        Returns:
            Every event the consumer saw, and the ``request_stream`` log entry.
        """
        written: list[EventLog] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)

        async def source() -> AsyncGenerator[ServerSentEvent]:
            yield ServerSentEvent(data="first")
            raise exc

        id_token = REQUEST_ID.set("test-request-id")
        request_token = REQUEST.set(cast("Any", _openai_tagged_request()))
        try:
            events = [
                chunk
                async for chunk in monitoring.log_request_sse_stream_event(source())
            ]
        finally:
            REQUEST.reset(request_token)
            REQUEST_ID.reset(id_token)
        (stream_log,) = [w for w in written if w["type"] == "request_stream"]
        return events, stream_log

    async def test_api_error_keeps_its_message_param_and_code(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """An ApiError becomes an OpenAI error envelope carrying its param and code.

        ``param`` and ``code`` are what an SDK matches on, so they must survive
        the trip through the SSE boundary rather than collapse into a bare
        message the way a generic failure does.
        """
        error = ApiError("Unsupported voice", status=400)
        error.param = "voice"
        error.code = "invalid_value"

        events, stream_log = await self._run(monkeypatch, error)

        assert [event.event for event in events] == [None, "error"]
        assert loads(str(events[-1].data)) == {
            "error": {
                "message": "Unsupported voice",
                "type": "invalid_request_error",
                "param": "voice",
                "code": "invalid_value",
            }
        }
        assert stream_log["level"] == "warning"
        assert request_log["level"] == "warning"

    async def test_a_denied_call_ends_the_stream_the_way_it_would_end_a_request(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A denial mid-stream reports the deployment's missing permission, not a 403.

        A stream that opened before the guardrail, tool or storage call it
        needed was refused must not answer differently from the same
        misconfiguration caught before the headers went out.

        Ref: stdapi/api_errors.py:denied_feature_unavailable
        """
        error = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": _LEAKY_DETAIL}},
            "ApplyGuardrail",
        )

        events, stream_log = await self._run(monkeypatch, error)

        body = loads(str(events[-1].data))["error"]
        assert body["code"] == "feature_unavailable"
        assert body["message"].startswith("The requested feature is not available")
        assert "arn:aws" not in str(events[-1].data)
        assert any(_LEAKY_DETAIL in str(d) for d in stream_log["error_detail"])

    async def test_client_error_below_500_is_relayed_redacted(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A 4xx AWS message reaches the client only with its ARN and account ID masked.

        The client needs to know what it did wrong, so the message is relayed --
        but ``ValidationException`` messages quote the resource, and neither the
        inference-profile ARN nor the account ID may be disclosed.
        """
        error = ClientError(
            {"Error": {"Code": "ValidationException", "Message": _LEAKY_DETAIL}},
            "ConverseStream",
        )

        events, stream_log = await self._run(monkeypatch, error)

        body = loads(str(events[-1].data))["error"]
        assert body["type"] == "invalid_request_error"
        assert body["message"] == ("Operation not allowed on <arn> in <account-id>")
        assert "arn:aws" not in body["message"]
        assert "123456789012" not in body["message"]
        # The unredacted original stays server-side, in the operator's log.
        assert any(_LEAKY_DETAIL in str(d) for d in stream_log["error_detail"])
        assert request_log["level"] == "warning"

    async def test_client_error_at_500_or_above_is_replaced_wholesale(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A 5xx AWS message is swapped for a fixed sentence, not merely redacted.

        Redaction only masks identifiers it recognises; a backend fault message
        can describe internals in prose, so nothing of it is relayed at all.
        """
        error = ClientError(
            {
                "Error": {
                    "Code": "InternalServerException",
                    "Message": f"backend pool exhausted: {_LEAKY_DETAIL}",
                }
            },
            "ConverseStream",
        )

        events, stream_log = await self._run(monkeypatch, error)

        body = loads(str(events[-1].data))["error"]
        assert (
            body["message"] == "The request could not be completed. Retry the request."
        )
        assert body["type"] == "server_error"
        assert "backend pool" not in str(events[-1].data)
        assert any("backend pool" in str(d) for d in stream_log["error_detail"])
        assert request_log["level"] == "error"

    async def test_connection_failure_is_a_service_unavailable_event(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A botocore connection failure closes the stream as a retryable 503.

        The error is not a ``ClientError`` and carries no AWS error code, so it
        falls to the transport branch and its default 503 rather than the 502
        an unmapped service code would get.
        """
        events, stream_log = await self._run(
            monkeypatch, BotocoreConnectionError(error=OSError("connection refused"))
        )

        body = loads(str(events[-1].data))["error"]
        assert body["message"] == (
            "The service is temporarily unavailable. Retry the request."
        )
        assert body["type"] == "server_error"
        # An unexpected transport failure is a server fault, logged as critical.
        assert stream_log["level"] == "critical"
        assert request_log["level"] == "error"

    async def test_unexpected_exception_yields_a_bare_500(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """An unforeseen exception yields "Internal Server Error" and a critical log.

        The exception text is never echoed: it is arbitrary Python detail. The
        traceback is kept in the request log instead, which is the only place it
        remains diagnosable.
        """
        events, _ = await self._run(monkeypatch, RuntimeError(_LEAKY_DETAIL))

        body = loads(str(events[-1].data))["error"]
        assert body["message"] == "Internal Server Error"
        assert body["type"] == "server_error"
        assert "arn:aws" not in str(events[-1].data)
        assert request_log["level"] == "critical"
        assert any("RuntimeError" in str(d) for d in request_log["error_detail"])


class TestBackgroundEventLog:
    """log_background_event writes one ``background`` entry, critical on failure.

    Scheduled cleanups run after the response was sent, so a failure there has
    no HTTP status to surface it: the log entry is the only signal, and it must
    carry the traceback.

    Ref: stdapi/monitoring.py:log_background_event
    """

    def test_failure_is_logged_critical_with_traceback_and_reraised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exception inside the scope is re-raised after being logged as critical."""
        written: list[EventLog] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)

        def fail() -> None:
            """Fail inside the background scope."""
            message = "cleanup failed"
            raise RuntimeError(message)

        with (
            pytest.raises(RuntimeError, match=r"^cleanup failed$"),
            log_background_event("cleanup", "rid-background"),
        ):
            fail()

        (log,) = written
        assert log["type"] == "background"
        assert log["id"] == "rid-background"
        assert log["level"] == "critical"
        assert any(
            "RuntimeError: cleanup failed" in str(d) for d in log["error_detail"]
        )
        assert "execution_time_ms" in log

    def test_success_stays_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A clean scope writes a single info entry, so failures stand out."""
        written: list[EventLog] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)

        with log_background_event("cleanup", "rid-ok") as log:
            assert log["level"] == "info"

        (written_log,) = written
        assert written_log["level"] == "info"
        assert "error_detail" not in written_log

    def test_scope_receives_what_the_work_logs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``log_error_details`` inside the scope writes into the background entry.

        ``log_error_details`` is a no-op outside a request-log context, so
        background work whose errors are its only report would log them nowhere
        without the scope installing itself as that context.

        Ref: stdapi/monitoring.py:log_background_event
        """
        written: list[EventLog] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)
        assert REQUEST_LOG.get(None) is None

        with log_background_event("vector_store_indexing", "rid-detail"):
            monitoring.log_error_details("indexing failed", level="error")

        (log,) = written
        assert log["error_detail"] == ["indexing failed"]
        assert log["level"] == "error"
        # The context is the scope's own, and only for its duration.
        assert REQUEST_LOG.get(None) is None

    def test_scope_restores_the_request_log_it_replaced(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """Work scheduled by a request logs into its own entry, not the request's.

        A background task inherits the request's context, whose log entry is
        already written by the time the task runs.

        Ref: stdapi/monitoring.py:log_background_event
        """
        written: list[EventLog] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)

        with log_background_event("cleanup", "rid-nested"):
            monitoring.log_error_details("cleanup warning", level="warning")

        (log,) = written
        assert log["error_detail"] == ["cleanup warning"]
        assert "error_detail" not in request_log
        assert REQUEST_LOG.get(None) is request_log


@pytest.fixture
def bind_principal() -> Generator[Callable[[Principal], None]]:
    """Bind a verified caller to the request context for the test's duration.

    The binding is undone by value rather than by token: an async test body runs
    in its own copy of the context, so a token taken there cannot be reset here.

    Yields:
        Callable binding its argument as the current request's principal.
    """

    def _bind(principal: Principal) -> None:
        PRINCIPAL.set(principal)

    yield _bind
    PRINCIPAL.set(None)


@pytest.fixture
def request_id() -> Generator[str]:
    """Bind a request ID for code that stamps it onto an outgoing call.

    Yields:
        The bound request ID.
    """
    token = REQUEST_ID.set("req-identity")
    yield "req-identity"
    REQUEST_ID.reset(token)


@pytest.fixture
async def api_key_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a global authentication handler accepting only ``good-key``."""
    monkeypatch.setattr(SETTINGS, "api_key", SecretStr("good-key"))
    monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
    monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
    handler = AuthenticationHandler()
    assert await handler.initialize()
    monkeypatch.setattr(stdapi.auth, "_auth_handler", handler)


@pytest.mark.usefixtures("request_log")
class TestRequestIdentity:
    """Which identity a request is attributed to, and where that identity may appear.

    A verified caller outranks the identifier the request body declares, because
    the first is what the gateway checked and the second is what the caller
    chose. The record itself stays in the request context: a log event is
    JSON-encoded to stdout field by field, so anything bound onto it is
    published to the operator's log stream.

    Ref: stdapi/monitoring.py:resolve_request_identity
         stdapi/monitoring.py:build_metadata
         stdapi/auth.py:authenticate
    """

    async def test_no_identity_when_the_request_declares_none(self) -> None:
        """A request with neither a verified caller nor a declared user has no identity."""
        assert resolve_request_identity() is None

    async def test_falls_back_to_the_identifier_the_client_declared(
        self, request_log: dict[str, Any]
    ) -> None:
        """Without a verified caller, the client-supplied user identifier is used."""
        request_log["request_user_id"] = "user-42"

        assert resolve_request_identity() == "user-42"

    async def test_verified_caller_outranks_the_declared_identifier(
        self, request_log: dict[str, Any], bind_principal: Callable[[Principal], None]
    ) -> None:
        """A verified caller is attributed over the identifier the request declares."""
        request_log["request_user_id"] = "user-42"
        bind_principal(Principal(subject="sub-123", username="alice"))

        assert resolve_request_identity() == "sub-123"

    @pytest.mark.usefixtures("request_id")
    async def test_metadata_attributes_the_invocation_to_the_verified_caller(
        self, request_log: dict[str, Any], bind_principal: Callable[[Principal], None]
    ) -> None:
        """The verified caller reaches request metadata, sanitized like any identity.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html
        """
        request_log["request_user_id"] = "user-42"
        bind_principal(Principal(subject="sub<123>"))

        metadata = build_metadata({"caller": "app"})

        assert metadata["stdapi-ai.user_id"] == "sub123"
        assert metadata["caller"] == "app"

    @pytest.mark.usefixtures("request_id")
    async def test_metadata_keeps_the_declared_identifier_without_a_verified_caller(
        self, request_log: dict[str, Any]
    ) -> None:
        """An unauthenticated deployment attributes invocations exactly as before."""
        request_log["request_user_id"] = "user-42"

        assert build_metadata()["stdapi-ai.user_id"] == "user-42"

    def test_the_verified_caller_never_reaches_the_emitted_log_event(
        self,
        bind_principal: Callable[[Principal], None],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """No part of the caller record is published to the log stream.

        The record carries credential details -- a subject, a user name, the
        client it authenticated with -- that the request log is not the place
        for; only the identity fields already published stay published.
        """
        bind_principal(
            Principal(
                subject="sub-123",
                username="alice",
                client_id="client-abc",
                scopes=frozenset({"chat.write"}),
            )
        )
        capsys.readouterr()

        with log_request_event(_make_request()) as log:
            log["request_user_id"] = "user-42"

        written = capsys.readouterr().out
        assert "sub-123" not in written
        assert "alice" not in written
        assert "client-abc" not in written
        assert "chat.write" not in written
        assert loads(written.strip().splitlines()[-1])["request_user_id"] == "user-42"

    @pytest.mark.usefixtures("api_key_authentication")
    async def test_an_api_key_authenticates_without_identifying_a_caller(
        self, request_log: dict[str, Any], bind_principal: Callable[[Principal], None]
    ) -> None:
        """A valid API key is the deployment's credential, not a person's.

        Nothing is attributed to it: the identity stays the one the request
        itself declares.
        """
        request_log["request_user_id"] = "user-42"
        bind_principal(Principal(subject="outer-user"))

        await authenticate(credentials=None, x_api_key="good-key")

        assert resolve_request_identity() == "user-42"

    async def test_authentication_drops_a_caller_from_an_enclosing_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bind_principal: Callable[[Principal], None],
    ) -> None:
        """A request served inside another request's context never inherits its caller.

        A tool call reaches the API as a second request running in the first
        one's context, so each request derives its own identity instead of
        inheriting one it never verified.
        """
        monkeypatch.setattr(stdapi.auth, "_auth_handler", AuthenticationHandler())
        bind_principal(Principal(subject="outer-user"))

        await authenticate(credentials=None, x_api_key=None)

        assert resolve_request_identity() is None

    @pytest.mark.usefixtures("api_key_authentication")
    async def test_a_rejected_credential_leaves_no_identity(
        self, bind_principal: Callable[[Principal], None]
    ) -> None:
        """A request that fails authentication carries no identity at all."""
        bind_principal(Principal(subject="outer-user"))

        with pytest.raises(ApiError) as exc_info:
            await authenticate(credentials=None, x_api_key="wrong-key")

        assert exc_info.value.status == 401
        assert resolve_request_identity() is None


class _NeverResumed:
    """Awaitable that suspends whoever awaits it and is never resumed."""

    def __await__(self) -> Generator[None]:
        """Suspend the awaiting coroutine.

        Yields:
            None
        """
        yield


def _suspended_request(span: Span) -> Coroutine[None, None, None]:
    """Build a request coroutine suspended inside *span*'s activation.

    This is the shape a request has while it waits on a backend call: the
    activation is installed, and the frame holding it is parked on an await.

    Args:
        span: Span the coroutine activates.

    Returns:
        The coroutine, not yet started.
    """

    async def request() -> None:
        with otel_manager.use_span(span):
            await _NeverResumed()

    return request()


@pytest.mark.skipif(not SETTINGS.otel_enabled, reason="tracing is disabled")
class TestAbandonedRequestSpan:
    """Deactivation of a request span when the request is abandoned, not unwound.

    A request coroutine that is collected while suspended is closed with
    ``GeneratorExit`` by whatever finalizes it, in whatever context that
    finalizer holds -- never necessarily the one the span was activated in.
    """

    def test_the_activating_context_keeps_no_span_of_an_abandoned_request(self) -> None:
        """The context the activation lives in is restored wherever it is closed.

        ``BaseHTTPMiddleware`` runs the downstream call in a child task holding
        a copy of the middleware's context, so an abandoned request is routinely
        finalized against a copy rather than the original. Restoring the
        previous context by token cannot cross that boundary, and what it leaves
        behind is the ended request span, still current: every span started
        there afterwards joins the finished request's trace.

        Ref: stdapi/monitoring_otel.py:OpenTelemetryManager.use_span
        """
        span = otel_manager.start_span("GET /abandoned", attributes={})
        middleware = copy_context()
        request = _suspended_request(span)
        middleware.run(request.send, None)
        child = middleware.run(copy_context)

        child.run(request.close)

        assert not child.run(get_current_span).get_span_context().is_valid
        following = child.run(
            lambda: otel_manager.start_span("GET /next", attributes={})
        )
        assert following.get_span_context().trace_id != span.get_span_context().trace_id

    def test_an_abandoned_request_leaves_an_unrelated_context_alone(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Closing the activation elsewhere writes nothing and reports nothing.

        This is what the agentic lane caught: the finalizer ran in a context
        that had never seen the activation, and resetting the token there fails,
        which OpenTelemetry reports as an error traceback on the server's
        stderr long after the client was answered. Nothing may be written into
        that context either -- it is tracing work of its own.

        Ref: stdapi/monitoring_otel.py:OpenTelemetryManager.use_span
        """
        span = otel_manager.start_span("GET /abandoned", attributes={})
        request = _suspended_request(span)
        copy_context().run(request.send, None)
        unrelated = copy_context()

        with caplog.at_level(ERROR, logger="opentelemetry.context"):
            unrelated.run(request.close)

        assert not caplog.records
        assert not unrelated.run(get_current_span).get_span_context().is_valid
