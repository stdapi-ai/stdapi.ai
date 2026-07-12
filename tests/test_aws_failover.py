"""Unit tests for auxiliary-service region candidates and failover helpers."""

from typing import TYPE_CHECKING, Any, Self

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

import stdapi.aws
from stdapi.aws import (
    AWSConnectionManager,
    call_with_region_failover,
    is_failover_error,
    service_regions,
)
from stdapi.config import AWS_SESSION, SETTINGS

if TYPE_CHECKING:
    from types import TracebackType

    from types_aiobotocore_bedrock.literals import RegionName


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


def _client_error(code: str, status: int = 400) -> ClientError:
    response: Any = {
        "Error": {"Code": code, "Message": code},
        "ResponseMetadata": {"HTTPStatusCode": status},
    }
    return ClientError(response, "SomeOperation")


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


class TestIsFailoverError:
    """is_failover_error: region-level errors yes, caller errors no."""

    def test_network_errors_fail_over(self) -> None:
        """BotoCoreError (network/DNS) always fails over."""
        error = EndpointConnectionError(endpoint_url="https://x.invalid")
        assert is_failover_error(error) is True

    @pytest.mark.parametrize(
        "code", ["ThrottlingException", "ServiceUnavailableException"]
    )
    def test_region_level_client_errors_fail_over(self, code: str) -> None:
        """Throttling/availability codes fail over."""
        assert is_failover_error(_client_error(code)) is True

    def test_http_5xx_fails_over(self) -> None:
        """An unlisted code still fails over when the HTTP status is 5xx."""
        assert is_failover_error(_client_error("WeirdBackendError", status=503)) is True

    def test_caller_errors_do_not_fail_over(self) -> None:
        """Validation errors would fail in every region: no failover."""
        assert is_failover_error(_client_error("ValidationException")) is False

    def test_access_denied_does_not_fail_over(self) -> None:
        """AccessDeniedException is a caller-permission error: no failover."""
        assert is_failover_error(_client_error("AccessDeniedException")) is False

    def test_not_authorized_fails_over(self) -> None:
        """NotAuthorizedException is treated as a region-level issue: failover."""
        assert is_failover_error(_client_error("NotAuthorizedException")) is True


class TestCallWithRegionFailover:
    """call_with_region_failover: ordered attempts, correct raise semantics."""

    @staticmethod
    def _patch_clients(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Route get_client to a stub and log the requested regions."""
        requested: list[str] = []

        def _get_client(_service: str, region: RegionName | None = None) -> str:
            requested.append(str(region))
            return f"client-{region}"

        monkeypatch.setattr(stdapi.aws, "get_client", _get_client)
        return requested

    async def test_first_region_success_stops_there(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful first call returns immediately with its region."""
        requested = self._patch_clients(monkeypatch)

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
        self._patch_clients(monkeypatch)

        async def _call(_client: object, region: RegionName) -> str:
            if region == "us-east-1":
                error = _client_error("ThrottlingException")
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
        requested = self._patch_clients(monkeypatch)

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
        self._patch_clients(monkeypatch)

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
        requested = self._patch_clients(monkeypatch)

        async def _call(_client: object, _region: RegionName) -> str:
            error = _client_error("ThrottlingException")
            raise error

        with pytest.raises(ClientError):
            await call_with_region_failover("comprehend", ["us-east-1"], _call)
        assert requested == ["us-east-1"]


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
