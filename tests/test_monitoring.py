"""Request/stream log finalization in stdapi.monitoring: usage, cost and error capture.

Everything here runs in-process against a seeded price index and patched
Bedrock seams, so the token counts, currencies and log levels asserted below are
exact rather than indicative.

Ref: stdapi/monitoring.py:_finalize_usage
     stdapi/monitoring.py:log_request_event
     stdapi/monitoring.py:_rebuild_and_log_stream
"""

from asyncio import create_task, sleep
from datetime import UTC, datetime
from gc import collect
from typing import TYPE_CHECKING, cast

import pytest
from sse_starlette import ServerSentEvent
from starlette.requests import Request as StarletteRequest

from stdapi import monitoring, usage
from stdapi.config import SETTINGS
from stdapi.models import ModelBase
from stdapi.monitoring import (
    REQUEST_ID,
    REQUEST_LOG,
    EventLog,
    SseHandledStreamError,
    _finalize_usage,
    log_request_event,
)
from stdapi.pricing import Dimension
from stdapi.usage import get_model_state, record_bedrock_usage
from tests.conftest import set_test_price

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable, Generator
    from typing import Any

    from types_aiobotocore_bedrock_runtime.type_defs import ConverseStreamOutputTypeDef

    from stdapi.config import LogLevel


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


def _new_log() -> EventLog:
    return EventLog(
        type="request",
        level="info",
        date=datetime.now(UTC),
        server_id="test",
        server_version="0.0.0",
    )


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


class TestFinalizeUsage:
    """_finalize_usage() turns the request's usage accumulator into log usage/cost fields.

    Ref: stdapi/monitoring.py:_finalize_usage
         stdapi/usage.py:record_bedrock_usage
    """

    @pytest.fixture(autouse=True)
    def _usage_scope(self) -> Generator[None]:
        """Install a fresh usage accumulator and reset it after each test."""
        token = usage.init_usage()
        yield
        usage.USAGE.reset(token)

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

        log = _new_log()
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

        log = _new_log()
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

        log = _new_log()
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

        log = _new_log()
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

        log = _new_log()
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

        log = _new_log()
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

        log = _new_log()
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
            log = _new_log()
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
                _finalize_usage(_new_log())
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
            request_log = _new_log()
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

    async def test_disconnect_mid_stream_writes_stream_log_without_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closing after the stream log scope opened still writes that log, without usage.

        The metadata event never arrived, so nothing was billed — but the timing
        entry is still emitted and the source generator's ``finally`` runs.
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
            assert "usage" not in stream_log
            assert "cost" not in stream_log
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

    Only the ``SseHandledStreamError`` branch is covered here; the ``ApiError`` and
    ``ClientError`` branches, which additionally yield a terminal ``error`` SSE
    event, are not.

    Ref: stdapi/monitoring.py:log_request_sse_stream_event
         stdapi/monitoring.py:_stream_exception_detail
    """

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
        log_token = REQUEST_LOG.set(_new_log())
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
            REQUEST_LOG.reset(log_token)
