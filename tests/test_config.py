"""Unit tests for configuration settings (:mod:`stdapi.config`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from stdapi.config import SETTINGS, _Settings

if TYPE_CHECKING:
    import pytest


def test_proxy_trusted_hosts_defaults_to_wildcard() -> None:
    """proxy_trusted_hosts defaults to '*' for backward compatibility."""
    assert SETTINGS.proxy_trusted_hosts == "*"


def test_proxy_trusted_hosts_parses_json_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """A JSON array in the environment is parsed into a list of trusted hosts."""
    monkeypatch.setenv("PROXY_TRUSTED_HOSTS", '["10.0.0.0/8", "127.0.0.1"]')
    assert _Settings().proxy_trusted_hosts == ["10.0.0.0/8", "127.0.0.1"]
