"""Unit tests for application-level helpers (:mod:`stdapi.main`)."""

from __future__ import annotations

import pytest

from stdapi.main import _upstream_service_name


class _FakeConnectionError:
    """Minimal stand-in for a botocore connection error carrying ``kwargs``."""

    def __init__(self, endpoint_url: str) -> None:
        self.kwargs = {"endpoint_url": endpoint_url}


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("https://bedrock-runtime.us-east-1.amazonaws.com", "Bedrock"),
        ("https://s3.eu-west-1.amazonaws.com", "S3"),
        ("https://polly.us-east-1.amazonaws.com", "Polly"),
    ],
)
def test_upstream_service_name_maps_known_services(
    endpoint: str, expected: str
) -> None:
    """A recognised AWS endpoint host yields its friendly service name."""
    assert _upstream_service_name(_FakeConnectionError(endpoint)) == expected  # type: ignore[arg-type]


def test_upstream_service_name_hides_unknown_endpoint() -> None:
    """A custom/VPC endpoint host is not disclosed; a generic name is returned."""
    exc = _FakeConnectionError("https://vpce-0abc123def.execute-api.example")
    assert _upstream_service_name(exc) == "The upstream service"  # type: ignore[arg-type]


def test_upstream_service_name_handles_missing_endpoint() -> None:
    """An error without an endpoint URL yields the generic name."""
    assert _upstream_service_name(_FakeConnectionError("")) == "The upstream service"  # type: ignore[arg-type]


class TestBotocoreConnectionErrorHandler:
    """The 503 connection-error handler hides internal endpoint detail."""

    async def test_generic_body_while_endpoint_stays_in_logs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Clients see a generic per-service 503; the endpoint URL is logged only."""
        from botocore.exceptions import EndpointConnectionError  # noqa: PLC0415

        from stdapi import main  # noqa: PLC0415

        endpoint = "https://bedrock-runtime.us-east-1.amazonaws.com"
        logged: list[str] = []
        monkeypatch.setattr(
            main,
            "log_error_details",
            lambda message, **_kwargs: logged.append(str(message)),
        )

        response = await main.handle_botocore_connection_error(
            _request(), EndpointConnectionError(endpoint_url=endpoint)
        )

        assert response.status_code == 503
        body = bytes(response.body).decode()
        assert "Bedrock is temporarily unavailable." in body
        assert endpoint not in body
        assert any(endpoint in entry for entry in logged)

    async def test_unknown_endpoint_is_not_disclosed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A custom/VPC endpoint is never named or disclosed in the 503 body."""
        from botocore.exceptions import EndpointConnectionError  # noqa: PLC0415

        from stdapi import main  # noqa: PLC0415

        endpoint = "https://vpce-0abc123.execute-api.internal"
        monkeypatch.setattr(main, "log_error_details", lambda *_a, **_k: None)

        response = await main.handle_botocore_connection_error(
            _request(), EndpointConnectionError(endpoint_url=endpoint)
        )

        assert response.status_code == 503
        body = bytes(response.body).decode()
        assert "The upstream service is temporarily unavailable." in body
        assert "vpce-0abc123" not in body
