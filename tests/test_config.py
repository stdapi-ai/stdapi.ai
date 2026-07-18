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


def test_mantle_regions_comma_list_strips_and_drops_empty_items() -> None:
    """A comma-separated region string strips whitespace and drops empty items."""
    settings = _Settings(
        aws_bedrock_regions=["us-east-1"],
        aws_bedrock_mantle_regions="us-east-1, ,eu-west-1,",  # type: ignore[arg-type]
    )
    assert settings.aws_bedrock_mantle_regions == ["us-east-1", "eu-west-1"]


def test_mantle_regions_empty_string_falls_back_to_bedrock_regions() -> None:
    """An empty Mantle regions string falls back to aws_bedrock_regions."""
    settings = _Settings(
        aws_bedrock_regions=["us-east-1", "us-west-2"],
        aws_bedrock_mantle_regions="",  # type: ignore[arg-type]
    )
    assert settings.aws_bedrock_mantle_regions == ["us-east-1", "us-west-2"]


def test_mantle_preferred_models_comma_list_strips_whitespace() -> None:
    """A comma-separated preferred-models string strips whitespace per item."""
    settings = _Settings(
        aws_bedrock_mantle_preferred_models="anthropic.claude-haiku-4-5, openai.gpt-oss "  # type: ignore[arg-type]
    )
    assert settings.aws_bedrock_mantle_preferred_models == [
        "anthropic.claude-haiku-4-5",
        "openai.gpt-oss",
    ]
