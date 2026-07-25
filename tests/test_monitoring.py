"""Unit tests for stdapi.monitoring's usage/cost finalization (_finalize_usage).

Regression: previously untested end-to-end; existing tests exercised compute_costs()
directly against a manually-seeded price index, bypassing this function entirely.
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
    """_published_log_levels() severity filtering."""

    def test_disabled_publishes_nothing(self) -> None:
        """LOG_LEVEL=disabled must yield no published levels (and not crash)."""
        assert monitoring._published_log_levels("disabled") == set()  # noqa: SLF001

    def test_warning_publishes_warning_and_above(self) -> None:
        """A minimum level keeps itself and every more severe level."""
        assert monitoring._published_log_levels("warning") == {  # noqa: SLF001
            "warning",
            "error",
            "critical",
        }


class TestFinalizeUsage:
    """_finalize_usage() end-to-end tests."""

    @pytest.fixture(autouse=True)
    def _usage_scope(self) -> Generator[None]:
        """Install a fresh usage accumulator and reset it after each test."""
        token = usage.init_usage()
        yield
        usage.USAGE.reset(token)

    def test_populates_usage_and_cost_from_a_single_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Single record appears in log["usage"] with matching log["cost"]."""
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
        """Two different models billed within one request must sum into one request-level total."""
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
        assert log["cost"] == {"USD": "0.008"}

    def test_multi_currency_costs_entries_fold_into_the_request_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A record surfacing a `costs` (plural) breakdown must still roll into log["cost"]."""
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        get_model_state("modela").region = "us-east-1"
        set_test_price("modela", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD")
        set_test_price(
            "modela", "us-east-1", Dimension.OUTPUT_TOKENS, "0.000015", "EUR"
        )
        record_bedrock_usage("modela", input_tokens=1000, output_tokens=1000)

        log = _new_log()
        _finalize_usage(log)

        assert "costs" in log["usage"][0]
        assert log["cost"] == {"USD": "0.003", "EUR": "0.015"}

    def test_no_usage_leaves_log_fields_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request with no billed usage must not add empty usage/cost fields to the log."""
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)

        log = _new_log()
        _finalize_usage(log)

        assert "usage" not in log
        assert "cost" not in log

    def test_cost_tracking_disabled_still_logs_usage_without_cost(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With cost_tracking off, quantities are still logged but no cost is computed."""
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
        """Before the price catalog has loaded, cost is skipped and no warning fires.

        The (autouse) _clean_price_index fixture leaves the index empty, so
        price_catalog_ready() is False -- the exact state during the startup
        window before the background catalog load completes.
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
        """Once the catalog is ready, a real per-model pricing miss still warns."""
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


class TestFinalizeUsageFailureIsolation:
    """A raising _finalize_usage must not fail the request, drop its log, or leak context."""

    def test_finalize_usage_failure_does_not_fail_the_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exception raised by _finalize_usage must not propagate out of log_request_event."""
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
        """The failure must be recorded in error_detail/level, and the log still written."""
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
        """Per-request ContextVars must still be reset even when _finalize_usage raises."""
        monkeypatch.setattr(monitoring, "_finalize_usage", _boom)

        request = _make_request()
        with log_request_event(request):
            pass

        with pytest.raises(LookupError):
            REQUEST_ID.get()


class TestStreamUsageContinuity:
    """The stream log picks up usage recorded after the request log finalized.

    Drain-style tasks (streamed /v1/completions, parallel image jobs) are
    created before the stream log scope exists and capture the request
    context: the request finalize drains what it logged, and the shared
    accumulator carries later records into the ``request_stream`` log.
    """

    async def test_request_finalize_drains_records(self) -> None:
        """A record logged by the request finalize must not be re-logged later."""
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
        """A finalize failure must still drain, or the stream log re-logs the records."""
        token = usage.init_usage()
        try:
            record_bedrock_usage("failfinalizemodel", input_tokens=5, total_tokens=5)

            def _boom() -> None:
                message = "emit failed"
                raise RuntimeError(message)

            monkeypatch.setattr(monitoring, "emit_usage_metrics", _boom)
            with pytest.raises(RuntimeError):
                _finalize_usage(_new_log())
            assert not usage.USAGE.get()
        finally:
            usage.USAGE.reset(token)

    async def test_late_task_usage_lands_in_stream_log(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Usage recorded by a pre-stream-scope task appears in the stream log only."""
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
    """ModelBase.converse inside log_request_event lands billed cost in the request log."""

    async def test_converse_usage_costed_in_request_log(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Converse response's billed tokens must surface as usage and cost in the request log."""
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
        """When the response reports no serviceTier, the requested flex tier must be billed."""
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
    """Usage captured by _capture_stream_usage is costed in the request_stream log."""

    async def test_stream_metadata_usage_costed_in_stream_log(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Usage from the stream's metadata event must land, costed, in the stream log."""
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
            assert entry["cost"] == "0.003"
            assert entry["region"] == "us-east-1"
        finally:
            REQUEST_ID.reset(id_token)
            usage.USAGE.reset(usage_token)


class TestStreamClientDisconnect:
    """A client disconnect mid-stream must not raise and must still close the source."""

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
        """Closing mid-stream still writes the stream log, with no usage (no metadata seen)."""
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
            await _wait_for_flag(finalized)
            assert finalized == [True]
        finally:
            REQUEST_ID.reset(id_token)
            usage.USAGE.reset(usage_token)

    async def test_disconnect_on_first_chunk_closes_source_without_stream_log(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closing right after the first chunk closes the source; no stream log scope opened."""
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
    """SseHandledStreamError.__init__: level defaults from status; explicit override wins."""

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


class TestMidStreamErrorLoggedInStreamEvent:
    """Mid-stream ApiError/ClientError/SseHandledStreamError land in the request_stream log."""

    async def test_sse_handled_stream_error_logs_message_and_warning_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mid-stream SseHandledStreamError(status=400) is recorded as a warning."""
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
            (stream_log,) = [w for w in written if w["type"] == "request_stream"]
            assert stream_log["level"] == "warning"
            assert any(
                "bad request" in str(detail) for detail in stream_log["error_detail"]
            )
        finally:
            REQUEST_ID.reset(id_token)
            REQUEST_LOG.reset(log_token)
