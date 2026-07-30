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

pytestmark = pytest.mark.local


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


@pytest.mark.parametrize(
    ("value", "expected_fragment"),
    [
        pytest.param("", "must be non-empty", id="empty"),
        pytest.param("/videos/", "", id="leading-slash"),
        pytest.param("videos//nested/", "", id="double-slash"),
        pytest.param("videos", 'must end with a trailing "/"', id="no-trailing-slash"),
        pytest.param("videos with spaces/", "S3-safe", id="unsafe-characters"),
    ],
)
def test_videos_prefix_rejects_malformed_values(
    value: str, expected_fragment: str
) -> None:
    """A malformed videos prefix is rejected, and the message echoes the offending value.

    An empty prefix would widen the video-ownership check to the whole bucket, and a
    missing trailing "/" would stop ``_region_videos_uri_prefix`` (which appends one
    before comparing) from matching the literal prefix used to build Bedrock's output
    ``s3Uri``.

    Ref: stdapi/config.py:_validate_videos_prefix
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_StartAsyncInvoke.html
    """
    with pytest.raises(ValidationError, match="aws_s3_videos_prefix") as excinfo:
        _Settings(aws_s3_videos_prefix=value)
    (error,) = excinfo.value.errors()
    assert error["loc"] == ("aws_s3_videos_prefix",)
    assert f'Invalid aws_s3_videos_prefix "{value}"' in error["msg"]
    assert expected_fragment in error["msg"]


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


class TestBedrockModelArnMapping:
    """aws_bedrock_model_arn_mapping: only inference-profile and prompt-router ARNs.

    The mapping silently redirects every invocation of a model ID to another ARN, so
    a typo has to fail at startup rather than once per request.

    Ref: stdapi/config.py:_validate_arn_mapping
         https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
    """

    @pytest.mark.parametrize(
        "arn",
        [
            pytest.param(
                "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc",
                id="application-inference-profile",
            ),
            pytest.param(
                "arn:aws:bedrock:us-east-1:123456789012:inference-profile/abc",
                id="inference-profile",
            ),
            pytest.param(
                "arn:aws:bedrock:us-east-1:123456789012:prompt-router/abc",
                id="prompt-router",
            ),
            pytest.param(
                "arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:default-prompt-router/abc",
                id="default-prompt-router-govcloud",
            ),
        ],
    )
    def test_profile_and_router_arns_are_accepted(self, arn: str) -> None:
        """A profile or prompt-router ARN, in any AWS partition, is kept verbatim."""
        settings = _Settings(aws_bedrock_model_arn_mapping={"my-model": arn})
        assert settings.aws_bedrock_model_arn_mapping == {"my-model": arn}

    @pytest.mark.parametrize(
        "arn",
        [
            pytest.param(
                "arn:aws:bedrock:us-east-1:123456789012:foundation-model/x",
                id="foundation-model",
            ),
            pytest.param("not-an-arn", id="not-an-arn"),
            pytest.param(
                "arn:aws:bedrock:us-east-1:123456789012:inference-profile/abc extra",
                id="trailing-data",
            ),
        ],
    )
    def test_other_arns_are_rejected_naming_the_model_id(self, arn: str) -> None:
        """A non-profile, non-router ARN is rejected, naming both model ID and ARN."""
        with pytest.raises(ValidationError, match="Invalid ARN format") as excinfo:
            _Settings(aws_bedrock_model_arn_mapping={"my-model": arn})
        (error,) = excinfo.value.errors()
        assert error["loc"] == ("aws_bedrock_model_arn_mapping",)
        assert f"my-model: {arn}" in error["msg"]

    def test_default_mapping_is_empty(self) -> None:
        """No mapping is configured by default, so model IDs reach Bedrock unchanged."""
        assert _Settings().aws_bedrock_model_arn_mapping == {}


class TestGuardrailSettingsCrossValidation:
    """Guardrail identifier/version pairing and the override default it drives.

    A guardrail configured without a version would be silently inert, and the
    per-request override flag is forced on when there is no server guardrail to
    protect, so both rules decide whether content filtering applies at all.

    Ref: stdapi/config.py:_Settings._validate
         https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param(
                {"aws_bedrock_guardrail_identifier": "gr-123"}, id="identifier-only"
            ),
            pytest.param({"aws_bedrock_guardrail_version": "1"}, id="version-only"),
        ],
    )
    def test_identifier_and_version_must_both_be_set(
        self, kwargs: dict[str, str]
    ) -> None:
        """Half a guardrail pair fails startup with a message naming both settings."""
        with pytest.raises(ValidationError, match="Both") as excinfo:
            _Settings(**kwargs)  # type: ignore[arg-type]
        (error,) = excinfo.value.errors()
        assert error["loc"] == ()
        assert "aws_bedrock_guardrail_identifier" in error["msg"]
        assert "aws_bedrock_guardrail_version" in error["msg"]

    def test_both_set_is_accepted(self) -> None:
        """A complete identifier/version pair is accepted."""
        settings = _Settings(
            aws_bedrock_guardrail_identifier="gr-123", aws_bedrock_guardrail_version="1"
        )
        assert settings.aws_bedrock_guardrail_identifier == "gr-123"
        assert settings.aws_bedrock_guardrail_version == "1"

    def test_override_is_forced_on_without_a_server_guardrail(self) -> None:
        """Disabling the per-request override does nothing when no guardrail is set.

        With no server guardrail to protect, refusing a caller-supplied guardrail
        would only remove a capability, so the flag is reset to True.
        """
        settings = _Settings(aws_bedrock_allow_guardrail_override=False)
        assert settings.aws_bedrock_allow_guardrail_override is True

    def test_override_stays_off_with_a_server_guardrail(self) -> None:
        """With a server guardrail configured, an explicit False override is honoured."""
        settings = _Settings(
            aws_bedrock_allow_guardrail_override=False,
            aws_bedrock_guardrail_identifier="gr-123",
            aws_bedrock_guardrail_version="1",
        )
        assert settings.aws_bedrock_allow_guardrail_override is False


class TestApiKeySourceExclusivity:
    """API-key sources: the direct key and a fully-configured secret are exclusive.

    Ambiguous configuration means the priority chain in ``initialize`` silently picks
    a source and the operator's intended key is never read.

    Ref: stdapi/config.py:_Settings._validate
         stdapi/auth.py:AuthenticationHandler.initialize
    """

    def test_direct_key_with_a_full_secret_source_is_rejected(self) -> None:
        """A direct key plus both secret settings fails startup naming all three."""
        with pytest.raises(ValidationError, match="Only one of api_key") as excinfo:
            _Settings(
                api_key="direct",  # type: ignore[arg-type]
                api_key_secretsmanager_secret="stdapi/api-key",  # noqa: S106
                api_key_secretsmanager_key="api_key",
            )
        (error,) = excinfo.value.errors()
        assert error["loc"] == ()
        assert "api_key_secretsmanager_secret" in error["msg"]
        assert "api_key_secretsmanager_key" in error["msg"]

    def test_secret_field_selector_defaults_to_a_truthy_value(self) -> None:
        """The secret-field selector defaults to ``api_key``, so the guard needs only two.

        The rule reads as three settings but the third is never empty by default: naming
        a secret alongside a direct key is therefore always ambiguous.
        """
        assert _Settings().api_key_secretsmanager_key == "api_key"
        with pytest.raises(ValidationError, match="Only one of api_key"):
            _Settings(
                api_key="direct",  # type: ignore[arg-type]
                api_key_secretsmanager_secret="stdapi/api-key",  # noqa: S106
            )

    def test_a_direct_key_alone_is_accepted(self) -> None:
        """With no secret configured, the direct key is the single unambiguous source."""
        settings = _Settings(api_key="direct")  # type: ignore[arg-type]
        assert settings.api_key is not None
        assert settings.api_key.get_secret_value() == "direct"


class TestMcpToolFiltering:
    """mcp_include_tools / mcp_exclude_tools parsing and reconciliation.

    These two settings decide which gateway operations an MCP client may invoke, so
    the reconciliation rule is an attack-surface control rather than a convenience.

    Ref: stdapi/config.py:_parse_mcp_tools_list
         stdapi/config.py:_Settings._validate
    """

    def test_comma_separated_string_becomes_a_list(self) -> None:
        """A comma-separated env value is parsed into de-duplicated tool names.

        The parser routes through ``set()``, so order is not preserved.
        """
        settings = _Settings(mcp_exclude_tools="a, b ,a")  # type: ignore[arg-type]
        assert settings.mcp_exclude_tools is not None
        assert set(settings.mcp_exclude_tools) == {"a", "b"}

    def test_empty_value_becomes_none(self) -> None:
        """An empty list collapses to None: no filtering, rather than "expose nothing"."""
        settings = _Settings(mcp_include_tools="")  # type: ignore[arg-type]
        assert settings.mcp_include_tools is None

    def test_exclude_is_subtracted_from_include(self) -> None:
        """With both set, exclude is subtracted from include and then dropped.

        Leaving both populated would hand the MCP server an overlapping pair; folding
        them into a single allow-list keeps the exposed set unambiguous.
        """
        settings = _Settings(mcp_include_tools="a,b", mcp_exclude_tools="b")  # type: ignore[arg-type]
        assert settings.mcp_include_tools == ["a"]
        assert settings.mcp_exclude_tools is None

    def test_exclude_alone_is_left_untouched(self) -> None:
        """Without an include list, the exclude list survives as a deny-list."""
        settings = _Settings(mcp_exclude_tools="b")  # type: ignore[arg-type]
        assert settings.mcp_exclude_tools == ["b"]
        assert settings.mcp_include_tools is None


class TestOpenApiJsonImplication:
    """enable_openapi_json is implied by enable_docs and enable_redoc.

    Swagger UI and ReDoc both fetch ``/openapi.json``; without the implication they
    would render an empty page instead of failing visibly.

    Ref: stdapi/config.py:_Settings._validate
    """

    @pytest.mark.parametrize(
        "field",
        [
            pytest.param("enable_docs", id="docs"),
            pytest.param("enable_redoc", id="redoc"),
        ],
    )
    def test_docs_or_redoc_forces_openapi_json_on(self, field: str) -> None:
        """Enabling either docs UI turns on the OpenAPI schema route it depends on."""
        settings = _Settings(enable_openapi_json=False, **{field: True})  # type: ignore[arg-type]
        assert settings.enable_openapi_json is True

    def test_all_three_off_stays_off(self) -> None:
        """With no docs UI enabled, an explicit False is honoured."""
        settings = _Settings(
            enable_openapi_json=False, enable_docs=False, enable_redoc=False
        )
        assert settings.enable_openapi_json is False
