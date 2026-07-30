"""Settings parsing and validation in :mod:`stdapi.config`.

``_Settings`` is instantiated once at import time, so every rule here is
enforced at startup rather than per request. Each rejection test asserts the
validator's own message (which embeds the offending value) so an unrelated
validation failure cannot make the test pass.

Ref: stdapi/config.py:_Settings
"""

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
    """A comma-separated region string strips whitespace and drops empty items.

    Trailing commas and stray spaces are common in container env vars; an empty
    item would otherwise become an empty region name and break endpoint building.
    """
    settings = _Settings(
        aws_bedrock_regions=["us-east-1"],
        aws_bedrock_mantle_regions="us-east-1, ,eu-west-1,",  # type: ignore[arg-type]
    )
    assert settings.aws_bedrock_mantle_regions == ["us-east-1", "eu-west-1"]


def test_bedrock_regions_duplicates_are_removed_preserving_order() -> None:
    """A duplicated region is dropped, keeping the first-occurrence order.

    Order is the routing preference order, so de-duplication must not reshuffle it.
    """
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
    """An empty Mantle regions string falls back to aws_bedrock_regions.

    Mantle hosts a subset of the catalogue, but with no explicit list the gateway
    reuses the Bedrock region preference rather than serving no region at all.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
    """
    settings = _Settings(
        aws_bedrock_regions=["us-east-1", "us-west-2"],
        aws_bedrock_mantle_regions="",  # type: ignore[arg-type]
    )
    assert settings.aws_bedrock_mantle_regions == ["us-east-1", "us-west-2"]


def test_mantle_preferred_models_comma_list_strips_whitespace() -> None:
    """A comma-separated preferred-models string strips whitespace per item.

    A model ID with a stray leading/trailing space would never match a catalogue
    entry, silently disabling the Mantle preference.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions-mantle.html
    """
    settings = _Settings(
        aws_bedrock_mantle_preferred_models="anthropic.claude-haiku-4-5, openai.gpt-oss "  # type: ignore[arg-type]
    )
    assert settings.aws_bedrock_mantle_preferred_models == [
        "anthropic.claude-haiku-4-5",
        "openai.gpt-oss",
    ]


def test_routes_prefixes_default_to_valid_values() -> None:
    """The default routes prefixes (empty OpenAI, /anthropic, /cohere) are accepted.

    OpenAI is root-mounted so ``/v1/...`` works unprefixed; the other two are
    namespaced to avoid colliding with it.
    """
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
    """A prefix without a leading slash is rejected.

    Ref: stdapi/config.py:_check_routes_prefix
    """
    with pytest.raises(ValidationError, match=field_name) as excinfo:
        _Settings(**{field_name: "cohere"})  # type: ignore[arg-type]
    (error,) = excinfo.value.errors()
    assert error["loc"] == (field_name,)
    assert f'Invalid {field_name} "cohere"' in error["msg"]
    assert 'must start with "/"' in error["msg"]


@pytest.mark.parametrize(
    "field_name",
    ["openai_routes_prefix", "anthropic_routes_prefix", "cohere_routes_prefix"],
)
def test_routes_prefix_rejects_trailing_slash(field_name: str) -> None:
    """A prefix with a trailing slash is rejected.

    Routers are mounted as ``prefix + "/v1/..."``, so a trailing slash would
    produce a double slash in every path.

    Ref: stdapi/config.py:_check_routes_prefix
    """
    with pytest.raises(ValidationError, match=field_name) as excinfo:
        _Settings(**{field_name: "/cohere/"})  # type: ignore[arg-type]
    (error,) = excinfo.value.errors()
    assert error["loc"] == (field_name,)
    assert f'Invalid {field_name} "/cohere/"' in error["msg"]
    assert 'must not end with "/"' in error["msg"]


@pytest.mark.parametrize(
    "field_name", ["anthropic_routes_prefix", "cohere_routes_prefix"]
)
def test_routes_prefix_rejects_empty_when_default_is_non_empty(field_name: str) -> None:
    """An empty prefix is rejected for settings whose default is non-empty.

    Only one provider may be root-mounted, and OpenAI holds that slot; emptying
    another provider's prefix would shadow OpenAI's ``/v1`` routes.

    Ref: stdapi/config.py:_validate_required_routes_prefix
    """
    with pytest.raises(ValidationError, match=field_name) as excinfo:
        _Settings(**{field_name: ""})  # type: ignore[arg-type]
    (error,) = excinfo.value.errors()
    assert error["loc"] == (field_name,)
    assert f'Invalid {field_name} ""' in error["msg"]


def test_openai_routes_prefix_allows_empty_value() -> None:
    """An empty prefix is accepted for openai_routes_prefix, whose default is empty.

    Ref: stdapi/config.py:_validate_openai_routes_prefix
    """
    settings = _Settings(openai_routes_prefix="")
    assert settings.openai_routes_prefix == ""


def test_duplicate_non_empty_routes_prefixes_are_rejected() -> None:
    """Two providers sharing the same non-empty routes prefix are rejected.

    This is a whole-model rule, so the error is reported without a field
    location and names both colliding settings.

    Ref: stdapi/config.py:_validate_unique_routes_prefixes
    """
    expected = "routes prefixes must be unique"
    with pytest.raises(ValidationError, match=expected) as excinfo:
        _Settings(anthropic_routes_prefix="/dup", cohere_routes_prefix="/dup")
    (error,) = excinfo.value.errors()
    assert error["loc"] == ()
    assert (
        'anthropic_routes_prefix and cohere_routes_prefix must not both be set to "/dup"'
        in error["msg"]
    )


def test_videos_prefix_defaults_to_valid_value() -> None:
    """The default videos prefix ("videos/") is accepted."""
    assert _Settings().aws_s3_videos_prefix == "videos/"


@pytest.mark.parametrize("value", ["videos/", "prod/videos/", "a-b_c.d*e'f(g)/"])
def test_videos_prefix_accepts_well_formed_values(value: str) -> None:
    """A trailing-slash-terminated prefix using S3-safe characters is accepted.

    Ref: stdapi/config.py:_validate_videos_prefix
    """
    settings = _Settings(aws_s3_videos_prefix=value)
    assert settings.aws_s3_videos_prefix == value


def test_videos_prefix_rejects_empty_value() -> None:
    """An empty prefix is rejected: it would widen the ownership check to the whole bucket.

    Ref: stdapi/config.py:_validate_videos_prefix
    """
    with pytest.raises(ValidationError, match="aws_s3_videos_prefix") as excinfo:
        _Settings(aws_s3_videos_prefix="")
    (error,) = excinfo.value.errors()
    assert error["loc"] == ("aws_s3_videos_prefix",)
    assert 'Invalid aws_s3_videos_prefix ""' in error["msg"]
    assert "must be non-empty" in error["msg"]


def test_videos_prefix_rejects_leading_slash() -> None:
    """A prefix starting with "/" is rejected.

    Ref: stdapi/config.py:_validate_videos_prefix
    """
    with pytest.raises(ValidationError, match="aws_s3_videos_prefix") as excinfo:
        _Settings(aws_s3_videos_prefix="/videos/")
    (error,) = excinfo.value.errors()
    assert error["loc"] == ("aws_s3_videos_prefix",)
    assert 'Invalid aws_s3_videos_prefix "/videos/"' in error["msg"]


def test_videos_prefix_rejects_double_slash() -> None:
    """A prefix containing "//" is rejected.

    Ref: stdapi/config.py:_validate_videos_prefix
    """
    with pytest.raises(ValidationError, match="aws_s3_videos_prefix") as excinfo:
        _Settings(aws_s3_videos_prefix="videos//nested/")
    (error,) = excinfo.value.errors()
    assert error["loc"] == ("aws_s3_videos_prefix",)
    assert 'Invalid aws_s3_videos_prefix "videos//nested/"' in error["msg"]


def test_videos_prefix_rejects_missing_trailing_slash() -> None:
    """A prefix without a trailing "/" is rejected.

    Without it, the ownership check in ``_region_videos_uri_prefix`` (which
    appends a "/" before comparing) would no longer match the literal prefix
    used to build the Bedrock output ``s3Uri``.

    Ref: stdapi/config.py:_validate_videos_prefix
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_StartAsyncInvoke.html
    """
    with pytest.raises(ValidationError, match="aws_s3_videos_prefix") as excinfo:
        _Settings(aws_s3_videos_prefix="videos")
    (error,) = excinfo.value.errors()
    assert error["loc"] == ("aws_s3_videos_prefix",)
    assert 'Invalid aws_s3_videos_prefix "videos"' in error["msg"]
    assert 'must end with a trailing "/"' in error["msg"]


def test_videos_prefix_rejects_unsafe_characters() -> None:
    """A prefix using characters outside S3's safe set is rejected.

    Ref: stdapi/config.py:_validate_videos_prefix
    """
    with pytest.raises(ValidationError, match="aws_s3_videos_prefix") as excinfo:
        _Settings(aws_s3_videos_prefix="videos with spaces/")
    (error,) = excinfo.value.errors()
    assert error["loc"] == ("aws_s3_videos_prefix",)
    assert 'Invalid aws_s3_videos_prefix "videos with spaces/"' in error["msg"]
    assert "S3-safe" in error["msg"]


def test_session_encryption_key_arn_defaults_to_none() -> None:
    """The default session encryption key ARN (unset, AWS-managed key) is accepted.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/sessions.html
    """
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
    """A well-formed KMS key ARN is accepted for any AWS partition.

    The gateway also runs in the China, GovCloud and EU Sovereign Cloud
    partitions, so the pattern must not hardcode ``arn:aws``.

    Ref: stdapi/config.py:_validate_session_encryption_key_arn
    """
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
    """A value that is not a KMS key ARN is rejected.

    Bedrock ``CreateSession`` needs a key ARN specifically: an alias ARN, a
    non-KMS ARN or a short key ID would only fail once a session is created.

    Ref: stdapi/config.py:_validate_session_encryption_key_arn
    """
    with pytest.raises(
        ValidationError, match="aws_bedrock_session_encryption_key_arn"
    ) as excinfo:
        _Settings(aws_bedrock_session_encryption_key_arn=value)
    (error,) = excinfo.value.errors()
    assert error["loc"] == ("aws_bedrock_session_encryption_key_arn",)
    assert f'Invalid aws_bedrock_session_encryption_key_arn "{value}"' in error["msg"]
    assert "must be a KMS key ARN" in error["msg"]


def test_mantle_endpoint_url_defaults_to_none() -> None:
    """The Mantle endpoint URL template is unset by default.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
    """
    assert _Settings().aws_bedrock_mantle_endpoint_url is None


def test_mantle_endpoint_url_accepts_valid_https_template() -> None:
    """A well-formed https:// template with a "{region}" placeholder is accepted.

    Ref: stdapi/config.py:_validate_mantle_endpoint_url
    """
    value = "https://bedrock-mantle.{region}.example.com"
    settings = _Settings(aws_bedrock_mantle_endpoint_url=value)
    assert settings.aws_bedrock_mantle_endpoint_url == value


def test_mantle_endpoint_url_rejects_non_https_scheme() -> None:
    """A non-"https://" endpoint URL is rejected: it would disable transport encryption.

    Ref: stdapi/config.py:_validate_mantle_endpoint_url
    """
    value = "http://bedrock-mantle.{region}.aws"
    with pytest.raises(
        ValidationError, match="aws_bedrock_mantle_endpoint_url"
    ) as excinfo:
        _Settings(aws_bedrock_mantle_endpoint_url=value)
    (error,) = excinfo.value.errors()
    assert error["loc"] == ("aws_bedrock_mantle_endpoint_url",)
    assert f'Invalid aws_bedrock_mantle_endpoint_url "{value}"' in error["msg"]
    assert '"https://" scheme' in error["msg"]


@pytest.mark.parametrize(
    "value",
    [
        "https://bedrock-mantle.{regoin}.example.com",
        "https://bedrock-mantle.{}.example.com",
        "https://bedrock-mantle.{.example.com",
    ],
)
def test_mantle_endpoint_url_rejects_malformed_placeholder(value: str) -> None:
    """A malformed "{region}" placeholder is rejected at startup, not per-request.

    The template is expanded with ``str.format(region=...)``, so a misspelled
    key, a positional ``{}`` and an unbalanced brace all break endpoint building.

    Ref: stdapi/config.py:_validate_mantle_endpoint_url
    """
    with pytest.raises(
        ValidationError, match="aws_bedrock_mantle_endpoint_url"
    ) as excinfo:
        _Settings(aws_bedrock_mantle_endpoint_url=value)
    (error,) = excinfo.value.errors()
    assert error["loc"] == ("aws_bedrock_mantle_endpoint_url",)
    assert f'Invalid aws_bedrock_mantle_endpoint_url "{value}"' in error["msg"]
    assert "malformed" in error["msg"]
