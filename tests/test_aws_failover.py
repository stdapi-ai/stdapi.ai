"""Region candidates, cross-region failover and client pooling for auxiliary AWS services.

Polly, Comprehend, Translate and Transcribe have no inference-profile layer, so
rotating across the configured Bedrock regions is their only resilience mechanism.
That makes the failover/fatal classification of an AWS error the load-bearing
decision here: a per-region quota error must move on, a caller error must not.

Ref: https://docs.aws.amazon.com/transcribe/latest/dg/what-is.html
     https://docs.aws.amazon.com/comprehend/latest/dg/guidelines-and-limits.html
     stdapi/aws.py:call_with_region_failover
"""

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Self
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

import stdapi.aws
import stdapi.monitoring
from stdapi.aws import (
    CONFIG,
    AWSConnectionManager,
    call_with_region_failover,
    is_failover_error,
    raise_first_exception,
    service_regions,
)
from stdapi.config import AWS_SESSION, SETTINGS
from stdapi.monitoring import REQUEST_LOG, EventLog

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from types import TracebackType

    from aiobotocore.config import AioConfig
    from types_aiobotocore_bedrock.literals import RegionName


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


def _client_error(code: str, status: int = 400, message: str = "") -> ClientError:
    response: Any = {
        "Error": {"Code": code, "Message": message or code},
        "ResponseMetadata": {"HTTPStatusCode": status},
    }
    return ClientError(response, "SomeOperation")


def _patch_get_client(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Route get_client to a stub and log the requested regions."""
    requested: list[str] = []

    def _get_client(_service: str, region: RegionName | None = None) -> str:
        requested.append(str(region))
        return f"client-{region}"

    monkeypatch.setattr(stdapi.aws, "get_client", _get_client)
    return requested


class TestServiceRegions:
    """service_regions: the per-service region setting overrides the Bedrock region list.

    Ref: stdapi/aws.py:service_regions
    """

    def test_unset_setting_yields_all_bedrock_regions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no service region, every Bedrock region is a candidate, in settings order."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1", "eu-west-1"])
        assert service_regions(None) == ["us-east-1", "eu-west-1"]

    def test_explicit_setting_yields_that_region_only(self) -> None:
        """A configured service region pins the service to it and disables failover."""
        assert service_regions("ap-southeast-2") == ["ap-southeast-2"]

    def test_returned_list_is_a_copy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The returned candidates are a copy, so a caller cannot corrupt the settings.

        Callers reorder and truncate the candidate list per request; the settings
        list is process-wide and would otherwise leak those mutations.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1", "eu-west-1"])
        candidates = service_regions(None)
        candidates.append("ap-southeast-2")
        assert SETTINGS.aws_bedrock_regions == ["us-east-1", "eu-west-1"]
        assert candidates == ["us-east-1", "eu-west-1", "ap-southeast-2"]


class TestIsFailoverError:
    """is_failover_error: region-level errors fail over, caller errors do not.

    Ref: stdapi/aws.py:is_failover_error
    """

    def test_network_errors_fail_over(self) -> None:
        """Any ``BotoCoreError`` (network, DNS, endpoint resolution) fails over.

        The whole ``BotoCoreError`` hierarchy is classified without inspecting the
        instance: it can never carry a per-region AWS error code to reason about.
        """
        error = EndpointConnectionError(endpoint_url="https://x.invalid")
        assert is_failover_error(error) is True

    @pytest.mark.parametrize(
        "code",
        [
            "InternalFailure",
            "InternalServerError",
            "InternalServerException",
            "LimitExceededException",
            "RequestTimeout",
            "ServiceQuotaExceededException",
            "ServiceUnavailable",
            "ServiceUnavailableException",
            "ThrottlingException",
            "TooManyRequestsException",
        ],
    )
    def test_failover_codes_are_pinned(self, code: str) -> None:
        """Each code in the failover set is classified as retryable on an HTTP 400.

        The status is deliberately 400 so the verdict comes from the code alone and
        not from the 5xx shortcut. ``LimitExceededException`` is Transcribe's
        per-region concurrent-job quota — the reason this failover layer exists.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/what-is.html
        """
        assert is_failover_error(_client_error(code, status=400)) is True

    def test_http_5xx_fails_over(self) -> None:
        """An unlisted error code still fails over when the HTTP status is 5xx.

        A 5xx is server-side by definition, so the code allow-list is only the fast
        path: an unknown backend failure must not become a fatal caller error.
        """
        assert is_failover_error(_client_error("WeirdBackendError", status=503)) is True

    def test_service_missing_from_the_region_fails_over(self) -> None:
        """Comprehend's ``NotAuthorizedException`` outside its regions fails over.

        Comprehend exists in roughly 13 regions only, and a region that does not host
        it answers this 4xx — indistinguishable from a policy denial by status, hence
        the dedicated code allow-list.

        Ref: https://docs.aws.amazon.com/comprehend/latest/dg/guidelines-and-limits.html
        """
        error = _client_error(
            "NotAuthorizedException",
            message="Your account is not authorized to make this call.",
        )
        assert is_failover_error(error) is True

    def test_operation_unsupported_in_the_region_fails_over(self) -> None:
        """An ``UNSUPPORTED_OPERATION`` message fails over despite a caller-error code.

        The same ``InvalidRequestException`` code is used for genuine bad input, so
        only the message prefix distinguishes a region gap from a fatal error.
        """
        error = _client_error(
            "InvalidRequestException",
            message="UNSUPPORTED_OPERATION: This operation is not supported in this region",
        )
        assert is_failover_error(error) is True

    @pytest.mark.parametrize(
        "code",
        [
            "AccessDeniedException",
            "ValidationException",
            "UnrecognizedClientException",
            "ExpiredTokenException",
        ],
    )
    def test_account_global_and_caller_errors_do_not_fail_over(self, code: str) -> None:
        """Auth, policy and validation errors are fatal: they fail identically everywhere.

        Retrying these in every candidate region only multiplies the latency and the
        CloudTrail noise of a request that can never succeed.
        """
        assert is_failover_error(_client_error(code, status=400)) is False

    def test_plain_invalid_request_does_not_fail_over(self) -> None:
        """``InvalidRequestException`` without the ``UNSUPPORTED_OPERATION`` prefix stays fatal.

        The counterpart of the region-gap case: the code is shared, so a plain
        validation message must not be mistaken for a regional capability gap.
        """
        error = _client_error("InvalidRequestException", message="Invalid text segment")
        assert is_failover_error(error) is False


class TestCallWithRegionFailover:
    """call_with_region_failover: ordered attempts, and the raise semantics per error class.

    Ref: stdapi/aws.py:call_with_region_failover
    """

    async def test_first_region_success_stops_there(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful first call returns its result and the serving region, no second try.

        The serving region is returned because callers need it to build region-scoped
        follow-up calls (an S3 staging bucket, a job lookup) against the same region.
        """
        requested = _patch_get_client(monkeypatch)

        async def _call(client: object, region: RegionName) -> str:
            return f"ok-{client}-{region}"

        result, region = await call_with_region_failover(
            "comprehend", ["us-east-1", "eu-west-1"], _call
        )
        assert result == "ok-client-us-east-1-us-east-1"
        assert region == "us-east-1"
        assert requested == ["us-east-1"]

    async def test_region_error_falls_over_to_next(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A throttling error in region 1 is retried in region 2, which serves the call.

        Both regions get a client from the pool, in candidate order.
        """
        requested = _patch_get_client(monkeypatch)

        async def _call(_client: object, region: RegionName) -> str:
            if region == "us-east-1":
                error = _client_error("ThrottlingException")
                raise error
            return "served"

        result, region = await call_with_region_failover(
            "comprehend", ["us-east-1", "eu-west-1"], _call
        )
        assert (result, region) == ("served", "eu-west-1")
        assert requested == ["us-east-1", "eu-west-1"]

    async def test_botocore_error_falls_over_to_next(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A network-level ``BotoCoreError`` in region 1 is retried in region 2.

        A ``BotoCoreError`` carries no AWS error code, so it is only reachable
        through the ``BotoCoreError`` branch of the classifier, not the code table.
        """
        requested = _patch_get_client(monkeypatch)

        async def _call(_client: object, region: RegionName) -> str:
            if region == "us-east-1":
                error = EndpointConnectionError(endpoint_url="https://x.invalid")
                raise error
            return "served"

        result, region = await call_with_region_failover(
            "comprehend", ["us-east-1", "eu-west-1"], _call
        )
        assert (result, region) == ("served", "eu-west-1")
        assert requested == ["us-east-1", "eu-west-1"]

    async def test_caller_error_raises_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``ValidationException`` propagates unchanged without touching region 2."""
        requested = _patch_get_client(monkeypatch)

        async def _call(_client: object, _region: RegionName) -> str:
            error = _client_error("ValidationException")
            raise error

        with pytest.raises(ClientError, match="ValidationException") as exc_info:
            await call_with_region_failover(
                "comprehend", ["us-east-1", "eu-west-1"], _call
            )
        assert exc_info.value.response["Error"]["Code"] == "ValidationException"
        assert requested == ["us-east-1"]

    async def test_last_region_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every region fails, the LAST region's error is the one raised.

        The two errors are given different HTTP statuses so the assertion can tell
        them apart: reporting the first failure would hide the outcome of the region
        that actually decided the request.
        """
        _patch_get_client(monkeypatch)

        async def _call(_client: object, region: RegionName) -> str:
            error = _client_error(
                "ThrottlingException", status=400 if region == "us-east-1" else 503
            )
            raise error

        with pytest.raises(ClientError) as exc_info:
            await call_with_region_failover(
                "comprehend", ["us-east-1", "eu-west-1"], _call
            )
        assert exc_info.value.response["ResponseMetadata"]["HTTPStatusCode"] == 503

    async def test_single_region_has_no_failover_layer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a single candidate, even a failover-class error propagates directly.

        A single-region service has nowhere to fail over to, so botocore's in-region
        retries own the whole retry budget and the error reaches the caller as-is.
        """
        requested = _patch_get_client(monkeypatch)

        async def _call(_client: object, _region: RegionName) -> str:
            error = _client_error("ThrottlingException")
            raise error

        with pytest.raises(ClientError) as exc_info:
            await call_with_region_failover("comprehend", ["us-east-1"], _call)
        assert exc_info.value.response["Error"]["Code"] == "ThrottlingException"
        assert requested == ["us-east-1"]

    async def test_empty_regions_raises_value_error(self) -> None:
        """An empty candidate list fails on the unpacking, before any call is attempted.

        ``regions`` is documented as holding at least one region; the starred unpack
        that splits off the last candidate is what enforces it.

        Ref: stdapi/aws.py:call_with_region_failover
        """
        calls: list[tuple[object, str]] = []

        async def _call(client: object, region: RegionName) -> str:
            calls.append((client, region))
            return "never"

        with pytest.raises(ValueError, match="not enough values") as exc_info:
            await call_with_region_failover("comprehend", [], _call)
        assert "expected at least 1" in str(exc_info.value)
        assert calls == [], "no region was attempted"


class TestOnFailedRegionHook:
    """call_with_region_failover: the best-effort cleanup hook on failover-class failures.

    The hook exists for calls whose server-side effect may already have been applied
    when the call errored (e.g. a Transcribe job accepted then throttled), so the
    abandoned region gets a chance to clean up before the next one is tried.

    Ref: stdapi/aws.py:_cleanup_failed_region
    """

    @staticmethod
    def _record_hook(
        calls: list[tuple[object, str]],
    ) -> Callable[[object, RegionName], Awaitable[None]]:
        async def _hook(client: object, region: RegionName) -> None:
            calls.append((client, str(region)))

        return _hook

    async def test_hook_called_on_failover_class_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hook receives the failed region's own client, before the next region runs."""
        _patch_get_client(monkeypatch)
        calls: list[tuple[object, str]] = []

        async def _call(_client: object, region: RegionName) -> str:
            if region == "us-east-1":
                error = _client_error("ThrottlingException")
                raise error
            return "served"

        result, region = await call_with_region_failover(
            "comprehend",
            ["us-east-1", "eu-west-1"],
            _call,
            on_failed_region=self._record_hook(calls),
        )
        assert (result, region) == ("served", "eu-west-1")
        assert calls == [("client-us-east-1", "us-east-1")]

    async def test_hook_called_when_last_region_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hook also runs for the last region, before its error propagates.

        The last candidate is served outside the failover loop, so it needs its own
        cleanup call; otherwise the final region would leak its side effect.
        """
        _patch_get_client(monkeypatch)
        calls: list[tuple[object, str]] = []

        async def _call(_client: object, _region: RegionName) -> str:
            error = _client_error("ThrottlingException")
            raise error

        with pytest.raises(ClientError, match="ThrottlingException"):
            await call_with_region_failover(
                "comprehend",
                ["us-east-1", "eu-west-1"],
                _call,
                on_failed_region=self._record_hook(calls),
            )
        assert calls == [
            ("client-us-east-1", "us-east-1"),
            ("client-eu-west-1", "eu-west-1"),
        ]

    async def test_hook_not_called_on_caller_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fatal caller error propagates without invoking the hook.

        A request rejected before it took effect has nothing to clean up.
        """
        _patch_get_client(monkeypatch)
        calls: list[tuple[object, str]] = []

        async def _call(_client: object, _region: RegionName) -> str:
            error = _client_error("ValidationException")
            raise error

        with pytest.raises(ClientError, match="ValidationException"):
            await call_with_region_failover(
                "comprehend",
                ["us-east-1", "eu-west-1"],
                _call,
                on_failed_region=self._record_hook(calls),
            )
        assert calls == []

    async def test_hook_own_client_error_is_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``ClientError`` raised by the hook itself is suppressed, not surfaced.

        Cleanup is best-effort: the region being abandoned is already unhealthy, so
        its cleanup failing must not mask the successful failover.
        """
        _patch_get_client(monkeypatch)
        hook_calls: list[str] = []

        async def _call(_client: object, region: RegionName) -> str:
            if region == "us-east-1":
                error = _client_error("ThrottlingException")
                raise error
            return "served"

        async def _hook(_client: object, region: RegionName) -> None:
            hook_calls.append(region)
            error = _client_error("BadRequestException")
            raise error

        result, region = await call_with_region_failover(
            "comprehend", ["us-east-1", "eu-west-1"], _call, on_failed_region=_hook
        )
        assert (result, region) == ("served", "eu-west-1")
        assert hook_calls == ["us-east-1"], "the raising hook did run"


class TestFailoverWarningLog:
    """call_with_region_failover: a silent failover is still reported on the request log.

    A request that succeeded after failing over returns a 200, so the degraded region
    is only observable through the warning raised on the request log.

    Ref: stdapi/aws.py:call_with_region_failover
         stdapi/monitoring.py:log_error_details
    """

    @staticmethod
    async def _failing_then_serving_call(_client: object, region: RegionName) -> str:
        if region == "us-east-1":
            error = _client_error("ThrottlingException")
            raise error
        return "served"

    async def test_failover_logs_warning_naming_service_and_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failover raises the request log to ``warning`` and names service and region.

        The entry also names the error class so the operator can tell a throttled
        region from an unreachable one without a second log source.
        """
        _patch_get_client(monkeypatch)
        log: EventLog = EventLog(
            type="request",
            level="info",
            date=MagicMock(),
            server_id="test-server",
            server_version="0.0.0",
        )
        token = REQUEST_LOG.set(log)
        try:
            await call_with_region_failover(
                "comprehend",
                ["us-east-1", "eu-west-1"],
                self._failing_then_serving_call,
            )
        finally:
            REQUEST_LOG.reset(token)
        assert log["level"] == "warning"
        assert any(
            "comprehend" in str(detail)
            and "us-east-1" in str(detail)
            and "ClientError" in str(detail)
            for detail in log["error_detail"]
        ), log.get("error_detail")

    async def test_no_request_log_does_not_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without an active request log, the failover still succeeds and logs nothing.

        Failover also runs outside a request (startup discovery, background refresh),
        where the request-log context var is unset.
        """
        _patch_get_client(monkeypatch)
        monkeypatch.setattr(stdapi.monitoring, "REQUEST_LOG", ContextVar("request_log"))
        result, region = await call_with_region_failover(
            "comprehend", ["us-east-1", "eu-west-1"], self._failing_then_serving_call
        )
        assert (result, region) == ("served", "eu-west-1")


class TestGetClient:
    """get_client: the single-client fallback, and per-(service, region) pool lookup.

    Clients are long-lived and pooled at startup, so lookup is a pure dict access;
    the only special case is a service pinned to one region, whose single client
    answers for any requested region.

    Ref: stdapi/aws.py:get_client
    """

    def test_single_registered_client_is_returned_for_any_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With one client registered for a service, any requested region resolves to it.

        This is how a region-pinned service (``aws_polly_region`` and friends) keeps
        working when a caller passes the region it happens to be routing in.
        """
        client = object()
        monkeypatch.setattr(
            stdapi.aws, "_CLIENTS", {"comprehend": {"us-east-1": client}}
        )
        assert stdapi.aws.get_client("comprehend", "eu-west-1") is client
        assert stdapi.aws.get_client("comprehend", "us-east-1") is client

    def test_repeated_calls_return_the_same_cached_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pooled client is returned by identity, and a sibling region never leaks in.

        Returning a fresh client per call would leak connections, so identity — not
        equality — is the property under test.
        """
        client = object()
        other = object()
        monkeypatch.setattr(
            stdapi.aws,
            "_CLIENTS",
            {"comprehend": {"us-east-1": client, "eu-west-1": other}},
        )
        assert stdapi.aws.get_client("comprehend", "us-east-1") is client
        assert stdapi.aws.get_client("comprehend", "us-east-1") is client
        assert stdapi.aws.get_client("comprehend", "eu-west-1") is other


class _FakeClientCM:
    """Stand-in for an aiobotocore ``create_client`` async context manager."""

    def __init__(self, *, fail: bool) -> None:
        self._fail = fail
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> Self:
        if self._fail:
            error = _client_error("ThrottlingException")
            raise error
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.exited = True


class _FailingExitClientCM(_FakeClientCM):
    """Fake client whose cleanup itself raises."""

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        message = "close failed"
        raise RuntimeError(message)


class TestRaiseFirstException:
    """raise_first_exception: re-raise the first exception, note the suppressed siblings.

    ``gather(..., return_exceptions=True)`` turns concurrent failures into ordinary
    result values; without this helper only one of them would ever be diagnosable.

    Ref: stdapi/aws.py:raise_first_exception
    """

    def test_no_exceptions_returns_none(self) -> None:
        """A results sequence holding no exception instance leaves the sequence untouched.

        An exception *class* among the results is ordinary data: the scan matches
        ``BaseException`` instances only. ``AWSConnectionManager.__aenter__`` zips the
        very same sequence with its client specs right after this call, so the results
        must survive it unchanged.

        Ref: stdapi/aws.py:AWSConnectionManager.__aenter__
        """
        results: list[object] = ["ok", 42, ValueError]

        raise_first_exception(results)

        assert results == ["ok", 42, ValueError]

    def test_single_exception_raised_without_note(self) -> None:
        """A lone exception is re-raised by identity, with no note attached.

        A single failure has no sibling to report, so annotating it would only add
        noise to the traceback.
        """
        error = ValueError("boom")
        with pytest.raises(ValueError, match="boom") as exc_info:
            raise_first_exception(["ok", error])
        assert exc_info.value is error
        assert getattr(exc_info.value, "__notes__", []) == []

    def test_multiple_exceptions_first_raised_with_note_on_others(self) -> None:
        """The first exception is raised and carries the siblings as a PEP 678 note.

        The note is the only trace left of the suppressed failures: their class and
        message are inlined so a single traceback explains the whole fan-out.
        """
        first = ValueError("first error")
        second = _client_error("ThrottlingException")
        with pytest.raises(ValueError, match="first error") as exc_info:
            raise_first_exception(["ok", first, second])
        assert exc_info.value is first
        (note,) = first.__notes__
        assert "1 concurrent failure(s) suppressed" in note
        assert "ClientError" in note
        assert "ThrottlingException" in note
        assert getattr(second, "__notes__", []) == [], (
            "the suppressed sibling itself is not annotated"
        )


class TestAWSConnectionManagerAenter:
    """AWSConnectionManager.__aenter__: cleanup when client creation partially fails.

    ``__aenter__`` raising means the context manager protocol never calls
    ``__aexit__``, so the failing attempt has to unwind its own exit stack or the
    clients it already entered leak for the life of the process.

    Ref: stdapi/aws.py:AWSConnectionManager.__aenter__
    """

    async def test_failing_client_creation_closes_already_created_clients(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One failing client creation closes every sibling already entered, and empties the pool.

        The registry must come back empty: a half-populated ``_CLIENTS`` would hand
        out closed clients to every later ``get_client`` call.
        """
        monkeypatch.setattr(stdapi.aws, "_CLIENTS", {})
        created: dict[tuple[str, str], _FakeClientCM] = {}

        def _fake_create_client(
            service: str, *, region_name: str, **_kwargs: object
        ) -> _FakeClientCM:
            cm = _FakeClientCM(fail=service == "comprehend")
            created[(service, region_name)] = cm
            return cm

        monkeypatch.setattr(AWS_SESSION, "create_client", _fake_create_client)

        manager = AWSConnectionManager(
            ("polly", "us-east-1"), ("comprehend", "us-east-1")
        )
        with pytest.raises(ClientError, match="ThrottlingException") as exc_info:
            await manager.__aenter__()

        assert exc_info.value.response["Error"]["Code"] == "ThrottlingException"
        polly_cm = created[("polly", "us-east-1")]
        comprehend_cm = created[("comprehend", "us-east-1")]
        assert polly_cm.entered is True
        assert polly_cm.exited is True
        assert comprehend_cm.entered is False
        assert stdapi.aws._CLIENTS == {}  # noqa: SLF001

    async def test_failing_cleanup_is_noted_on_the_original_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cleanup failure becomes a note on the original error instead of masking it.

        Raising from the cleanup path would replace the startup failure the operator
        actually needs to see with a bare ``RuntimeError`` from the exit stack.
        """
        monkeypatch.setattr(stdapi.aws, "_CLIENTS", {})
        created: dict[tuple[str, str], _FakeClientCM] = {}

        def _fake_create_client(
            service: str, *, region_name: str, **_kwargs: object
        ) -> _FakeClientCM:
            cm: _FakeClientCM = (
                _FakeClientCM(fail=True)
                if service == "comprehend"
                else _FailingExitClientCM(fail=False)
            )
            created[(service, region_name)] = cm
            return cm

        monkeypatch.setattr(AWS_SESSION, "create_client", _fake_create_client)

        manager = AWSConnectionManager(
            ("polly", "us-east-1"), ("comprehend", "us-east-1")
        )
        with pytest.raises(ClientError, match="ThrottlingException") as exc_info:
            await manager.__aenter__()

        assert exc_info.value.response["Error"]["Code"] == "ThrottlingException"
        (note,) = exc_info.value.__notes__
        assert "Client cleanup also failed" in note
        assert "RuntimeError" in note
        assert "close failed" in note
        assert stdapi.aws._CLIENTS == {}  # noqa: SLF001


class TestFailoverRetryConfig:
    """__aenter__: multi-region failover services get the reduced retry budget.

    Deep in-region botocore retries and cross-region failover both consume the same
    wall-clock budget, so a service with several candidate regions trades retry depth
    for a faster hop to the next region.

    Ref: stdapi/aws.py:AWSConnectionManager.__aenter__
         stdapi/aws.py:service_regions
    """

    @staticmethod
    def _record_configs(
        monkeypatch: pytest.MonkeyPatch,
    ) -> dict[tuple[str, str], AioConfig]:
        """Stub client creation and record the config passed per (service, region)."""
        monkeypatch.setattr(stdapi.aws, "_CLIENTS", {})
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_enabled", False)
        recorded: dict[tuple[str, str], AioConfig] = {}

        def _fake_create_client(
            service: str, *, region_name: str, config: AioConfig
        ) -> _FakeClientCM:
            recorded[(service, region_name)] = config
            return _FakeClientCM(fail=False)

        monkeypatch.setattr(AWS_SESSION, "create_client", _fake_create_client)
        return recorded

    async def test_multi_region_services_use_failover_retry_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With several candidate regions, every failover service shares one reduced-retry config.

        The config object is shared by identity, not merely equal: it is built once
        per ``__aenter__`` and handed to each failover service's client.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1", "eu-west-1"])
        monkeypatch.setattr(SETTINGS, "aws_polly_region", None)
        monkeypatch.setattr(SETTINGS, "aws_translate_region", None)
        recorded = self._record_configs(monkeypatch)

        manager = AWSConnectionManager(("polly", None), ("translate", None))
        await manager.__aenter__()
        try:
            polly_config = recorded[("polly", "us-east-1")]
            expected = SETTINGS.aws_failover_max_retries + 1
            # AioConfig stores `retries` as a runtime attribute the stubs don't declare.
            assert polly_config.retries["max_attempts"] == expected  # type: ignore[attr-defined]
            assert polly_config is not CONFIG, (
                "a failover service must not reuse the standard retry budget"
            )
            assert recorded[("translate", "us-east-1")] is polly_config
        finally:
            await manager.__aexit__(None, None, None)

    async def test_pinned_region_keeps_standard_retry_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a single pinned region, the shared standard-retry ``CONFIG`` applies.

        A pinned service has nowhere to fail over to, so shortening its in-region
        retry budget would only make it less resilient.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1", "eu-west-1"])
        monkeypatch.setattr(SETTINGS, "aws_polly_region", "eu-west-1")
        recorded = self._record_configs(monkeypatch)

        manager = AWSConnectionManager(("polly", "eu-west-1"))
        await manager.__aenter__()
        try:
            # Identity, not contents: botocore renames CONFIG.retries keys in
            # place once a real client is created elsewhere in the test run.
            assert recorded[("polly", "eu-west-1")] is CONFIG
        finally:
            await manager.__aexit__(None, None, None)
