"""API-key authentication in :mod:`stdapi.auth`.

``AuthenticationHandler.initialize`` returns whether authentication ends up
enabled, and that boolean is what decides between a locked-down gateway and an
open one, so an empty value from any source must not be mistaken for a key.

Ref: stdapi/auth.py:AuthenticationHandler
"""

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
    """An empty API key (e.g. from an empty secret) is treated as auth disabled.

    With no key hashed, ``verify_credentials`` short-circuits and accepts even a
    request that carries no credentials at all.
    """
    monkeypatch.setattr(SETTINGS, "api_key", SecretStr(""))
    monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
    monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
    handler = AuthenticationHandler()
    assert await handler.initialize() is False
    handler.verify_credentials(None)


async def test_empty_api_key_from_ssm_disables_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty value retrieved from SSM is treated as auth disabled.

    The emptiness check runs after the source is read, so an existing but blank
    SSM parameter behaves exactly like an unset ``api_key``.
    """

    async def _empty_ssm_value() -> SecretStr:
        return SecretStr("")

    monkeypatch.setattr(SETTINGS, "api_key", None)
    monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", "/stdapi/api-key")
    monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
    monkeypatch.setattr(
        AuthenticationHandler, "_get_api_key_from_ssm", staticmethod(_empty_ssm_value)
    )
    handler = AuthenticationHandler()
    assert await handler.initialize() is False
    handler.verify_credentials(None)


async def test_nonempty_api_key_enables_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-empty API key enables authentication and rejects bad credentials.

    The plaintext key is dropped from ``SETTINGS`` once hashed, and both a wrong
    token and a missing token are rejected with the same detail-free 401 so the
    response cannot distinguish the two.

    Ref: stdapi/auth.py:verify_credentials
    """
    monkeypatch.setattr(SETTINGS, "api_key", SecretStr("a-real-secret"))
    monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
    monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
    handler = AuthenticationHandler()
    assert await handler.initialize() is True
    assert SETTINGS.api_key is None, "the plaintext key must not stay in SETTINGS"
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
        with pytest.raises(ApiError) as wrong_key:
            handler.verify_credentials(SecretStr("wrong-secret"))
        assert wrong_key.value.status == 401
        assert str(wrong_key.value) == "Unauthorized"

        with pytest.raises(ApiError) as no_key:
            handler.verify_credentials(None)
        assert no_key.value.status == 401
        assert str(no_key.value) == "Unauthorized"
    finally:
        REQUEST_LOG.reset(log_token)
