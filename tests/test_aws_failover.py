"""Unit tests for auxiliary-service region candidates and failover helpers."""

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
    """service_regions: service setting overrides the Bedrock regions default."""

    def test_unset_setting_yields_all_bedrock_regions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no service region, every Bedrock region is a candidate."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1", "eu-west-1"])
        assert service_regions(None) == ["us-east-1", "eu-west-1"]

    def test_explicit_setting_yields_that_region_only(self) -> None:
        """A configured service region disables multi-region candidates."""
        assert service_regions("ap-southeast-2") == ["ap-southeast-2"]

    def test_returned_list_is_a_copy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutating the returned candidates must not alter the settings list."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1", "eu-west-1"])
        service_regions(None).append("ap-southeast-2")
        assert SETTINGS.aws_bedrock_regions == ["us-east-1", "eu-west-1"]


class TestIsFailoverError:
    """is_failover_error: region-level errors yes, caller errors no."""

    def test_network_errors_fail_over(self) -> None:
        """BotoCoreError (network/DNS) always fails over."""
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
        """Every failover-class code (incl. Transcribe's LimitExceededException)."""
        assert is_failover_error(_client_error(code)) is True

    def test_http_5xx_fails_over(self) -> None:
        """An unlisted code still fails over when the HTTP status is 5xx."""
        assert is_failover_error(_client_error("WeirdBackendError", status=503)) is True

    def test_service_missing_from_the_region_fails_over(self) -> None:
        """Comprehend's answer outside its regions moves on to the next one."""
        error = _client_error(
            "NotAuthorizedException",
            message="Your account is not authorized to make this call.",
        )
        assert is_failover_error(error) is True

    def test_operation_unsupported_in_the_region_fails_over(self) -> None:
        """An operation the region does not offer moves on to the next one."""
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
        """Auth/policy/validation errors fail identically in every region."""
        assert is_failover_error(_client_error(code)) is False

    def test_plain_invalid_request_does_not_fail_over(self) -> None:
        """A validation error keeping the InvalidRequestException code stays fatal."""
        error = _client_error("InvalidRequestException", message="Invalid text segment")
        assert is_failover_error(error) is False


class TestCallWithRegionFailover:
    """call_with_region_failover: ordered attempts, correct raise semantics."""

    async def test_first_region_success_stops_there(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful first call returns immediately with its region."""
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
        """A throttling error in region 1 is retried in region 2."""
        _patch_get_client(monkeypatch)

        async def _call(_client: object, region: RegionName) -> str:
            if region == "us-east-1":
                error = _client_error("ThrottlingException")
                raise error
            return "served"

        result, region = await call_with_region_failover(
            "comprehend", ["us-east-1", "eu-west-1"], _call
        )
        assert (result, region) == ("served", "eu-west-1")

    async def test_botocore_error_falls_over_to_next(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A network-level BotoCoreError in region 1 is retried in region 2."""
        _patch_get_client(monkeypatch)

        async def _call(_client: object, region: RegionName) -> str:
            if region == "us-east-1":
                error = EndpointConnectionError(endpoint_url="https://x.invalid")
                raise error
            return "served"

        result, region = await call_with_region_failover(
            "comprehend", ["us-east-1", "eu-west-1"], _call
        )
        assert (result, region) == ("served", "eu-west-1")

    async def test_caller_error_raises_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A validation error is not retried in other regions."""
        requested = _patch_get_client(monkeypatch)

        async def _call(_client: object, _region: RegionName) -> str:
            error = _client_error("ValidationException")
            raise error

        with pytest.raises(ClientError, match="ValidationException"):
            await call_with_region_failover(
                "comprehend", ["us-east-1", "eu-west-1"], _call
            )
        assert requested == ["us-east-1"]

    async def test_last_region_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every region fails, the last region's error is raised."""
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
        """With one candidate, even region-level errors propagate directly."""
        requested = _patch_get_client(monkeypatch)

        async def _call(_client: object, _region: RegionName) -> str:
            error = _client_error("ThrottlingException")
            raise error

        with pytest.raises(ClientError):
            await call_with_region_failover("comprehend", ["us-east-1"], _call)
        assert requested == ["us-east-1"]

    async def test_empty_regions_raises_value_error(self) -> None:
        """An empty candidate list violates the documented "at least one" contract."""

        async def _call(_client: object, _region: RegionName) -> str:
            return "never"

        with pytest.raises(ValueError, match="not enough values"):
            await call_with_region_failover("comprehend", [], _call)


class TestOnFailedRegionHook:
    """call_with_region_failover: best-effort cleanup hook on failed regions."""

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
        """The hook receives the failed region's client before failing over."""
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
        """The hook also runs for the last region before its error propagates."""
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
        """A non-failover caller error propagates without invoking the hook."""
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
        """A ClientError raised by the hook itself does not break failover."""
        _patch_get_client(monkeypatch)

        async def _call(_client: object, region: RegionName) -> str:
            if region == "us-east-1":
                error = _client_error("ThrottlingException")
                raise error
            return "served"

        async def _hook(_client: object, _region: RegionName) -> None:
            error = _client_error("BadRequestException")
            raise error

        result, region = await call_with_region_failover(
            "comprehend", ["us-east-1", "eu-west-1"], _call, on_failed_region=_hook
        )
        assert (result, region) == ("served", "eu-west-1")


class TestFailoverWarningLog:
    """call_with_region_failover: per-region warning in the request log."""

    @staticmethod
    async def _failing_then_serving_call(_client: object, region: RegionName) -> str:
        if region == "us-east-1":
            error = _client_error("ThrottlingException")
            raise error
        return "served"

    async def test_failover_logs_warning_naming_service_and_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a seeded request log, a failover writes a warning entry."""
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
            "comprehend" in str(detail) and "us-east-1" in str(detail)
            for detail in log["error_detail"]
        )

    async def test_no_request_log_does_not_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without an active request log, the failover path still succeeds."""
        _patch_get_client(monkeypatch)
        monkeypatch.setattr(stdapi.monitoring, "REQUEST_LOG", ContextVar("request_log"))
        result, region = await call_with_region_failover(
            "comprehend", ["us-east-1", "eu-west-1"], self._failing_then_serving_call
        )
        assert (result, region) == ("served", "eu-west-1")


class TestGetClient:
    """get_client: single-client fallback and per-(service, region) caching."""

    def test_single_registered_client_is_returned_for_any_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With only one client registered for a service, any requested region gets it."""
        client = object()
        monkeypatch.setattr(
            stdapi.aws, "_CLIENTS", {"comprehend": {"us-east-1": client}}
        )
        assert stdapi.aws.get_client("comprehend", "eu-west-1") is client

    def test_repeated_calls_return_the_same_cached_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two calls for the same (service, region) return the identical client object."""
        client = object()
        monkeypatch.setattr(
            stdapi.aws,
            "_CLIENTS",
            {"comprehend": {"us-east-1": client, "eu-west-1": object()}},
        )
        assert stdapi.aws.get_client("comprehend", "us-east-1") is client
        assert stdapi.aws.get_client("comprehend", "us-east-1") is client


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
    """raise_first_exception: re-raises the first exception, noting any others."""

    def test_no_exceptions_returns_none(self) -> None:
        """With no exceptions in results, nothing is raised."""
        raise_first_exception(["ok", 42])

    def test_single_exception_raised_without_note(self) -> None:
        """A lone exception is raised as-is, with no note attached."""
        error = ValueError("boom")
        with pytest.raises(ValueError, match="boom") as exc_info:
            raise_first_exception(["ok", error])
        assert getattr(exc_info.value, "__notes__", []) == []

    def test_multiple_exceptions_first_raised_with_note_on_others(self) -> None:
        """The first exception is raised and notes the suppressed siblings."""
        first = ValueError("first error")
        second = _client_error("ThrottlingException")
        with pytest.raises(ValueError, match="first error") as exc_info:
            raise_first_exception(["ok", first, second])
        assert exc_info.value is first
        (note,) = first.__notes__
        assert "1 concurrent failure(s) suppressed" in note
        assert "ClientError" in note
        assert "ThrottlingException" in note


class TestAWSConnectionManagerAenter:
    """AWSConnectionManager.__aenter__: cleanup on partial client-creation failure."""

    async def test_failing_client_creation_closes_already_created_clients(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One failing client creation closes every sibling client already entered."""
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
        with pytest.raises(ClientError, match="ThrottlingException"):
            await manager.__aenter__()

        polly_cm = created[("polly", "us-east-1")]
        comprehend_cm = created[("comprehend", "us-east-1")]
        assert polly_cm.entered is True
        assert polly_cm.exited is True
        assert comprehend_cm.entered is False
        assert stdapi.aws._CLIENTS == {}  # noqa: SLF001

    async def test_failing_cleanup_is_noted_on_the_original_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cleanup failure becomes a note on the original exception, not a mask."""
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

        (note,) = exc_info.value.__notes__
        assert "Client cleanup also failed" in note
        assert "RuntimeError" in note
        assert "close failed" in note
        assert stdapi.aws._CLIENTS == {}  # noqa: SLF001


class TestFailoverRetryConfig:
    """__aenter__: multi-region failover services get the reduced retry budget."""

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
        """With several candidate regions, one shared reduced-retry config is used."""
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
            assert recorded[("translate", "us-east-1")] is polly_config
        finally:
            await manager.__aexit__(None, None, None)

    async def test_pinned_region_keeps_standard_retry_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a single pinned region, the standard retry budget applies."""
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
