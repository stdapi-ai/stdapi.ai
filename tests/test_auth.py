"""Unit tests for API-key authentication (:mod:`stdapi.auth`)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from stdapi.api_errors import ApiError
from stdapi.auth import AuthenticationHandler
from stdapi.config import SETTINGS
from stdapi.monitoring import REQUEST_LOG, EventLog


async def test_empty_api_key_disables_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty API key (e.g. from an empty secret) is treated as auth disabled."""
    monkeypatch.setattr(SETTINGS, "api_key", SecretStr(""))
    monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
    monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
    assert await AuthenticationHandler().initialize() is False


async def test_empty_api_key_from_ssm_disables_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty value retrieved from SSM is treated as auth disabled."""

    async def _empty_ssm_value() -> SecretStr:
        return SecretStr("")

    monkeypatch.setattr(SETTINGS, "api_key", None)
    monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", "/stdapi/api-key")
    monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
    monkeypatch.setattr(
        AuthenticationHandler, "_get_api_key_from_ssm", staticmethod(_empty_ssm_value)
    )
    assert await AuthenticationHandler().initialize() is False


async def test_nonempty_api_key_enables_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-empty API key enables authentication and rejects bad credentials."""
    monkeypatch.setattr(SETTINGS, "api_key", SecretStr("a-real-secret"))
    monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
    monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
    handler = AuthenticationHandler()
    assert await handler.initialize() is True
    handler.verify_credentials(SecretStr("a-real-secret"))
    # Rejections log into the request context; provide one for the unit test.
    log_token = REQUEST_LOG.set(
        EventLog(
            type="request",
            level="info",
            date=MagicMock(),
            server_id="test-server",
            server_version="0.0.0",
        )
    )
    try:
        with pytest.raises(ApiError):
            handler.verify_credentials(SecretStr("wrong-secret"))
        with pytest.raises(ApiError):
            handler.verify_credentials(None)
    finally:
        REQUEST_LOG.reset(log_token)
