"""Unit tests for configuration settings (:mod:`stdapi.config`)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stdapi.config import SETTINGS, _Settings


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


def test_bedrock_regions_duplicates_are_removed_preserving_order() -> None:
    """A duplicated region is dropped, keeping the first-occurrence order."""
    settings = _Settings(
        aws_bedrock_regions="us-east-1,eu-west-1,eu-central-1,eu-west-1"  # type: ignore[arg-type]
    )
    assert settings.aws_bedrock_regions == ["us-east-1", "eu-west-1", "eu-central-1"]


def test_mantle_regions_duplicates_are_removed_preserving_order() -> None:
    """A duplicated Mantle region is dropped, keeping the first-occurrence order."""
    settings = _Settings(
        aws_bedrock_regions=["us-east-1"],
        aws_bedrock_mantle_regions="eu-west-1,us-east-1,eu-west-1",  # type: ignore[arg-type]
    )
    assert settings.aws_bedrock_mantle_regions == ["eu-west-1", "us-east-1"]


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


def test_routes_prefixes_default_to_valid_values() -> None:
    """The default routes prefixes (empty OpenAI, /anthropic, /cohere) are accepted."""
    settings = _Settings()
    assert settings.openai_routes_prefix == ""
    assert settings.anthropic_routes_prefix == "/anthropic"
    assert settings.cohere_routes_prefix == "/cohere"


@pytest.mark.parametrize(
    "field_name",
    ["openai_routes_prefix", "anthropic_routes_prefix", "cohere_routes_prefix"],
)
def test_routes_prefix_accepts_custom_valid_value(field_name: str) -> None:
    """A well-formed custom prefix (leading slash, no trailing slash) is accepted."""
    settings = _Settings(**{field_name: "/x"})  # type: ignore[arg-type]
    assert getattr(settings, field_name) == "/x"


@pytest.mark.parametrize(
    "field_name",
    ["openai_routes_prefix", "anthropic_routes_prefix", "cohere_routes_prefix"],
)
def test_routes_prefix_rejects_missing_leading_slash(field_name: str) -> None:
    """A prefix without a leading slash is rejected."""
    with pytest.raises(ValidationError, match=field_name):
        _Settings(**{field_name: "cohere"})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name",
    ["openai_routes_prefix", "anthropic_routes_prefix", "cohere_routes_prefix"],
)
def test_routes_prefix_rejects_trailing_slash(field_name: str) -> None:
    """A prefix with a trailing slash is rejected."""
    with pytest.raises(ValidationError, match=field_name):
        _Settings(**{field_name: "/cohere/"})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field_name", ["anthropic_routes_prefix", "cohere_routes_prefix"]
)
def test_routes_prefix_rejects_empty_when_default_is_non_empty(field_name: str) -> None:
    """An empty prefix is rejected for settings whose default is non-empty."""
    with pytest.raises(ValidationError, match=field_name):
        _Settings(**{field_name: ""})  # type: ignore[arg-type]


def test_openai_routes_prefix_allows_empty_value() -> None:
    """An empty prefix is accepted for openai_routes_prefix, whose default is empty."""
    settings = _Settings(openai_routes_prefix="")
    assert settings.openai_routes_prefix == ""


def test_duplicate_non_empty_routes_prefixes_are_rejected() -> None:
    """Two providers sharing the same non-empty routes prefix are rejected."""
    with pytest.raises(ValidationError, match="routes prefixes must be unique"):
        _Settings(anthropic_routes_prefix="/dup", cohere_routes_prefix="/dup")


def test_videos_prefix_defaults_to_valid_value() -> None:
    """The default videos prefix ("videos/") is accepted."""
    assert _Settings().aws_s3_videos_prefix == "videos/"


@pytest.mark.parametrize("value", ["videos/", "prod/videos/", "a-b_c.d*e'f(g)/"])
def test_videos_prefix_accepts_well_formed_values(value: str) -> None:
    """A trailing-slash-terminated prefix using S3-safe characters is accepted."""
    settings = _Settings(aws_s3_videos_prefix=value)
    assert settings.aws_s3_videos_prefix == value


def test_videos_prefix_rejects_empty_value() -> None:
    """An empty prefix is rejected: it would widen the ownership check to the whole bucket."""
    with pytest.raises(ValidationError, match="aws_s3_videos_prefix"):
        _Settings(aws_s3_videos_prefix="")


def test_videos_prefix_rejects_leading_slash() -> None:
    """A prefix starting with "/" is rejected."""
    with pytest.raises(ValidationError, match="aws_s3_videos_prefix"):
        _Settings(aws_s3_videos_prefix="/videos/")


def test_videos_prefix_rejects_double_slash() -> None:
    """A prefix containing "//" is rejected."""
    with pytest.raises(ValidationError, match="aws_s3_videos_prefix"):
        _Settings(aws_s3_videos_prefix="videos//nested/")


def test_videos_prefix_rejects_missing_trailing_slash() -> None:
    """A prefix without a trailing "/" is rejected.

    Without it, the ownership check in ``_region_videos_uri_prefix`` (which
    appends a "/" before comparing) would no longer match the literal prefix
    used to build the Bedrock output ``s3Uri``.
    """
    with pytest.raises(ValidationError, match="aws_s3_videos_prefix"):
        _Settings(aws_s3_videos_prefix="videos")


def test_videos_prefix_rejects_unsafe_characters() -> None:
    """A prefix using characters outside S3's safe set is rejected."""
    with pytest.raises(ValidationError, match="aws_s3_videos_prefix"):
        _Settings(aws_s3_videos_prefix="videos with spaces/")


def test_session_encryption_key_arn_defaults_to_none() -> None:
    """The default session encryption key ARN (unset, AWS-managed key) is accepted."""
    assert _Settings().aws_bedrock_session_encryption_key_arn is None


@pytest.mark.parametrize(
    "value",
    [
        "arn:aws:kms:us-east-1:123456789012:key/1234abcd-12ab-34cd-56ef-1234567890ab",
        "arn:aws-cn:kms:cn-north-1:123456789012:key/1234abcd-12ab-34cd-56ef-1234567890ab",
        "arn:aws-us-gov:kms:us-gov-west-1:123456789012:key/1234abcd-12ab-34cd-56ef-1234567890ab",
        "arn:aws-eusc:kms:eusc-de-east-1:123456789012:key/1234abcd-12ab-34cd-56ef-1234567890ab",
    ],
)
def test_session_encryption_key_arn_accepts_valid_kms_key_arn(value: str) -> None:
    """A well-formed KMS key ARN is accepted for any AWS partition."""
    settings = _Settings(aws_bedrock_session_encryption_key_arn=value)
    assert settings.aws_bedrock_session_encryption_key_arn == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not-an-arn",
        "arn:aws:s3:::my-bucket/key",
        "arn:aws:kms:us-east-1:123456789012:alias/my-key",
        "arn:aws:kms:us-east-1:123456789012:key/too-short",
        "arn:aws:kms:us-east-1:12345:key/1234abcd-12ab-34cd-56ef-1234567890ab",
        "arn:awsx:kms:us-east-1:123456789012:key/1234abcd-12ab-34cd-56ef-1234567890ab",
    ],
)
def test_session_encryption_key_arn_rejects_invalid_values(value: str) -> None:
    """A value that is not a KMS key ARN is rejected."""
    with pytest.raises(ValidationError, match="aws_bedrock_session_encryption_key_arn"):
        _Settings(aws_bedrock_session_encryption_key_arn=value)


def test_mantle_endpoint_url_defaults_to_none() -> None:
    """The Mantle endpoint URL template is unset by default."""
    assert _Settings().aws_bedrock_mantle_endpoint_url is None


def test_mantle_endpoint_url_accepts_valid_https_template() -> None:
    """A well-formed https:// template with a "{region}" placeholder is accepted."""
    value = "https://bedrock-mantle.{region}.example.com"
    settings = _Settings(aws_bedrock_mantle_endpoint_url=value)
    assert settings.aws_bedrock_mantle_endpoint_url == value


def test_mantle_endpoint_url_rejects_non_https_scheme() -> None:
    """A non-"https://" endpoint URL is rejected: it would disable transport encryption."""
    with pytest.raises(ValidationError, match="aws_bedrock_mantle_endpoint_url"):
        _Settings(aws_bedrock_mantle_endpoint_url="http://bedrock-mantle.{region}.aws")


@pytest.mark.parametrize(
    "value",
    [
        "https://bedrock-mantle.{regoin}.example.com",
        "https://bedrock-mantle.{}.example.com",
        "https://bedrock-mantle.{.example.com",
    ],
)
def test_mantle_endpoint_url_rejects_malformed_placeholder(value: str) -> None:
    """A malformed "{region}" placeholder is rejected at startup, not per-request."""
    with pytest.raises(ValidationError, match="aws_bedrock_mantle_endpoint_url"):
        _Settings(aws_bedrock_mantle_endpoint_url=value)
