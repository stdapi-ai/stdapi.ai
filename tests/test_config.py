"""Settings parsing and validation in :mod:`stdapi.config`.

``_Settings`` is instantiated once at import time, so every rule here is
enforced at startup rather than per request. Each rejection test asserts the
validator's own message (which embeds the offending value) so an unrelated
validation failure cannot make the test pass.

Ref: stdapi/config.py:_Settings
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from stdapi.config import AWS_SESSION, SETTINGS, _Settings

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


class TestExtraModelParamsDropSettings:
    """extra_model_params_denylist / extra_model_params_drop_all parsing.

    These settings customize the LiteLLM client-control parameter filter shared
    by every "extra model parameters" funnel.

    Ref: stdapi/config.py:_parse_extra_model_params_denylist
         stdapi/aws_bedrock.py:filter_extra_model_parameters
    """

    def test_denylist_defaults_to_the_built_in_names(self) -> None:
        """With nothing configured, the setting already holds the built-in names."""
        settings = _Settings(aws_bedrock_regions=["us-east-1"])
        assert {"drop_params", "api_key", "custom_llm_provider"} <= (
            settings.extra_model_params_denylist
        )

    def test_denylist_comma_separated_string_extends_the_built_in_names(self) -> None:
        """A comma-separated env value is merged into the built-in names, not swapped for them.

        The merge happens here so the request path is one membership test; an
        operator adding a project-specific field must not silently disable the
        built-in protection.
        """
        settings = _Settings(
            aws_bedrock_regions=["us-east-1"],
            extra_model_params_denylist="x_flag, x_trace_id ,x_flag",  # type: ignore[arg-type]
        )
        assert {"x_flag", "x_trace_id", "drop_params"} <= (
            settings.extra_model_params_denylist
        )

    def test_denylist_is_not_built_when_the_passthrough_is_disabled(self) -> None:
        """extra_model_params_drop_all leaves nothing to filter, so nothing is retained.

        The filter short-circuits on the flag, so keeping the merged set would
        hold names no request path ever reads.
        """
        settings = _Settings(
            aws_bedrock_regions=["us-east-1"],
            extra_model_params_drop_all=True,
            extra_model_params_denylist="x_flag",  # type: ignore[arg-type]
        )
        assert settings.extra_model_params_denylist == frozenset()

    def test_drop_all_defaults_to_false(self) -> None:
        """The passthrough stays enabled unless explicitly disabled."""
        settings = _Settings(aws_bedrock_regions=["us-east-1"])
        assert settings.extra_model_params_drop_all is False


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


class TestBedrockRegionDetection:
    """aws_bedrock_regions falls back to the SDK's own region, or refuses to start.

    The region list drives every downstream default -- routing, S3 buckets,
    pricing -- so an unset ``AWS_BEDROCK_REGIONS`` may only be filled from the
    SDK's resolved region. When the SDK has none either, startup must fail loudly
    rather than leave the gateway with an empty region list.

    Ref: stdapi/config.py:_Settings._parse_bedrock_regions
    """

    @staticmethod
    def _pin_sdk_region(monkeypatch: pytest.MonkeyPatch, region: str | None) -> None:
        """Make the shared botocore session resolve (or fail to resolve) *region*."""
        monkeypatch.setattr(AWS_SESSION, "get_config_variable", lambda _name: region)

    def test_empty_value_falls_back_to_the_sdk_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no configured region, the SDK's resolved region becomes the only region."""
        self._pin_sdk_region(monkeypatch, "eu-west-3")
        settings = _Settings(aws_bedrock_regions="")  # type: ignore[arg-type]
        assert settings.aws_bedrock_regions == ["eu-west-3"]

    def test_no_region_anywhere_is_rejected_at_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With nothing configured and nothing detected, the settings refuse to build."""
        self._pin_sdk_region(monkeypatch, None)
        with pytest.raises(
            ValidationError,
            match=r"No AWS region specified in environment or configuration\.",
        ):
            _Settings(aws_bedrock_regions="")  # type: ignore[arg-type]


class TestOAuthDiscoverySettings:
    """The three settings publishing OAuth 2.0 protected resource metadata.

    The resource identifier is compared character by character by clients and
    is embedded in a quoted ``WWW-Authenticate`` parameter, so its shape is
    validated at startup rather than repaired per request.

    Ref: https://www.rfc-editor.org/rfc/rfc9728#section-3.3
         stdapi/config.py:_Settings._validate_oauth
    """

    def test_resource_identifier_keeps_its_exact_form(self) -> None:
        """A bare origin is published unchanged, and a trailing slash is dropped.

        RFC 9728 section 3.3 compares the published value against the URL the
        client inserted the well-known suffix into, so a normalised form (the
        slash a URL type would append) would be rejected by a strict client.
        """
        settings = _Settings(
            oauth_resource_identifier="https://api.example.com",
            oauth_authorization_servers="https://issuer.example.com/pool",  # type: ignore[arg-type]
        )
        assert settings.oauth_resource_identifier == "https://api.example.com"

        slashed = _Settings(
            oauth_resource_identifier="https://api.example.com/",
            oauth_authorization_servers="https://issuer.example.com/pool",  # type: ignore[arg-type]
        )
        assert slashed.oauth_resource_identifier == "https://api.example.com"

    @pytest.mark.parametrize(
        "value",
        [
            "http://localhost:8000",
            "https://[2001:db8::1]:8443",
            "https://api.example.com",
        ],
    )
    def test_resource_identifier_accepts_every_origin_form(self, value: str) -> None:
        """A port, a plain-HTTP local deployment and an IP literal are all origins."""
        settings = _Settings(
            oauth_resource_identifier=value,
            oauth_authorization_servers="https://issuer.example.com/pool",  # type: ignore[arg-type]
        )
        assert settings.oauth_resource_identifier == value

    @pytest.mark.parametrize(
        "value",
        [
            "api.example.com",
            "https://api.example.com/v1",
            "https://api.example.com?x=1",
            'https://api.example.com" ,realm="x',
            "ftp://api.example.com",
        ],
    )
    def test_resource_identifier_rejects_anything_but_an_origin(
        self, value: str
    ) -> None:
        """A relative URL, a path, a query or a quote in the value fails startup.

        The last two matter beyond tidiness: the value is interpolated into a
        quoted header parameter, so a double quote would let the setting inject
        further challenge parameters.
        """
        with pytest.raises(ValidationError, match="Invalid oauth_resource_identifier"):
            _Settings(
                oauth_resource_identifier=value,
                oauth_authorization_servers="https://issuer.example.com/pool",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "value",
        [
            "http://issuer.example.com/pool",
            "https://issuer.example.com/pool?x=1",
            "https://issuer.example.com/pool#f",
            "issuer.example.com",
        ],
    )
    def test_authorization_server_must_be_an_https_issuer(self, value: str) -> None:
        """RFC 8414 issuer identifiers use "https" and carry no query or fragment."""
        with pytest.raises(
            ValidationError, match="Invalid oauth_authorization_servers entry"
        ):
            _Settings(
                oauth_resource_identifier="https://api.example.com",
                oauth_authorization_servers=value,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("value", ['sco"pe', "scope with space", "back\\slash"])
    def test_scope_rejects_characters_a_challenge_cannot_carry(
        self, value: str
    ) -> None:
        """A scope is an RFC 6749 token, so it cannot close the quoted parameter."""
        with pytest.raises(
            ValidationError, match="Invalid oauth_scopes_supported entry"
        ):
            _Settings(
                oauth_resource_identifier="https://api.example.com",
                oauth_authorization_servers="https://issuer.example.com/pool",  # type: ignore[arg-type]
                oauth_scopes_supported=value,  # type: ignore[arg-type]
            )

    def test_resource_identifier_requires_an_authorization_server(self) -> None:
        """Publishing a document that names no authorization server fails startup.

        Such a document leaves a client unable to obtain a token, and the
        TypeScript MCP client reads the omission as this server being its own
        authorization server. With no user pool configured there is nothing to
        derive the issuer from, so the value has to be given.
        """
        with pytest.raises(
            ValidationError, match="oauth_authorization_servers is required"
        ):
            _Settings(
                oauth_resource_identifier="https://api.example.com",
                oauth_authorization_servers="",  # type: ignore[arg-type]
                aws_cognito_user_pool_id=None,
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("oauth_authorization_servers", "https://issuer.example.com/pool"),
            ("oauth_scopes_supported", "stdapi/invoke"),
        ],
    )
    def test_the_other_settings_require_a_resource_identifier(
        self, field: str, value: str
    ) -> None:
        """Describing a document that is never published fails startup instead."""
        with pytest.raises(
            ValidationError, match="oauth_resource_identifier is required"
        ):
            _Settings(oauth_resource_identifier=None, **{field: value})  # type: ignore[arg-type]

    def test_unset_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With nothing configured no document is described, which is the default."""
        for name in (
            "oauth_resource_identifier",
            "oauth_authorization_servers",
            "oauth_scopes_supported",
        ):
            monkeypatch.delenv(name, raising=False)
            monkeypatch.delenv(name.upper(), raising=False)
        settings = _Settings()
        assert settings.oauth_resource_identifier is None
        assert settings.oauth_authorization_servers == []
        assert settings.oauth_scopes_supported == []


#: User pool the derivation cases publish discovery for.
_POOL_ID = "eu-west-3_a1b2c3d4e"

#: Issuer that pool puts in every token it signs.
_POOL_ISSUER = f"https://cognito-idp.eu-west-3.amazonaws.com/{_POOL_ID}"


class TestOAuthDerivedFromCognito:
    """Discovery settings the configured user pool already answers.

    The pool issues the tokens this deployment accepts, so it is the
    authorization server clients must be sent to and the scopes it requires are
    the scopes they must ask for: both are derived rather than typed twice. An
    explicit value stays in force, and one the pool contradicts fails startup
    instead of publishing an issuer whose tokens every request refuses.

    Ref: https://docs.aws.amazon.com/cognito/latest/developerguide/amazon-cognito-user-pools-using-tokens-verifying-a-jwt.html
         stdapi/config.py:_Settings._validate_oauth
    """

    @staticmethod
    def _settings(**overrides: Any) -> _Settings:  # noqa: ANN401
        """Build the settings of a deployment publishing discovery for a pool.

        The discovery settings are empty unless a case sets them: the suite's
        own environment carries all three, which would otherwise stand in for
        every derived value.
        """
        return _Settings(
            aws_bedrock_regions=["us-east-1"],
            **{
                "aws_cognito_user_pool_id": _POOL_ID,
                "aws_cognito_client_ids": ["1example23456789abcdefghij"],
                "oauth_resource_identifier": "https://api.example.com",
                "oauth_authorization_servers": [],
                "oauth_scopes_supported": [],
                **overrides,
            },
        )

    @pytest.mark.parametrize(
        ("issuer_type", "expected"),
        [
            ("original", _POOL_ISSUER),
            (
                "updated",
                f"https://issuer-cognito-idp.eu-west-3.amazonaws.com/{_POOL_ID}",
            ),
        ],
    )
    def test_authorization_server_defaults_to_the_pool_issuer(
        self, issuer_type: str, expected: str
    ) -> None:
        """The published issuer is the one the pool's own tokens carry.

        Both issuer configurations are covered because the two hosts differ,
        and a client sent to the wrong one obtains a token this deployment
        rejects on every request.
        """
        settings = self._settings(aws_cognito_issuer_type=issuer_type)
        assert settings.oauth_authorization_servers == [expected]

    def test_the_pool_issuer_host_follows_its_partition(self) -> None:
        """A pool outside the commercial partition publishes its own host.

        The host is resolved from the pool's Region rather than assembled from
        ``amazonaws.com``, so a sovereign or China deployment publishes an
        issuer that resolves.
        """
        settings = self._settings(aws_cognito_user_pool_id="cn-north-1_a1b2c3d4e")
        assert settings.oauth_authorization_servers == [
            "https://cognito-idp.cn-north-1.amazonaws.com.cn/cn-north-1_a1b2c3d4e"
        ]

    def test_configured_authorization_servers_win(self) -> None:
        """An explicit list is published as given, in its own order.

        A deployment also fronted by an identity-aware proxy publishes both
        issuers, which no derivation can guess.
        """
        edge_issuer = "https://login.example.com"
        settings = self._settings(
            oauth_authorization_servers=[edge_issuer, _POOL_ISSUER]
        )
        assert settings.oauth_authorization_servers == [edge_issuer, _POOL_ISSUER]

    def test_an_authorization_server_contradicting_the_pool_is_rejected(self) -> None:
        """Publishing only an issuer the pool did not sign fails startup.

        Every token obtained from it is refused, which reads to the client as
        an unusable deployment rather than a configuration mistake.
        """
        with pytest.raises(
            ValidationError, match="oauth_authorization_servers must name the issuer"
        ):
            self._settings(
                oauth_authorization_servers=[
                    "https://cognito-idp.eu-west-3.amazonaws.com/eu-west-3_9z8y7x6w5"
                ]
            )

    def test_scopes_default_to_the_required_scopes(self) -> None:
        """The advertised scopes are the ones a token must carry to be accepted."""
        settings = self._settings(aws_cognito_required_scopes=["stdapi/invoke"])
        assert settings.oauth_scopes_supported == ["stdapi/invoke"]

    def test_configured_scopes_win(self) -> None:
        """An explicit scope list is published even when the pool requires others."""
        settings = self._settings(
            aws_cognito_required_scopes=["stdapi/invoke"],
            oauth_scopes_supported=["stdapi/read"],
        )
        assert settings.oauth_scopes_supported == ["stdapi/read"]

    def test_a_required_scope_a_challenge_cannot_carry_is_rejected(self) -> None:
        """A required scope is validated too, since it becomes a published one.

        The published value is interpolated into a quoted ``WWW-Authenticate``
        parameter, so a double quote would let the setting inject further
        challenge parameters.
        """
        with pytest.raises(
            ValidationError, match="Invalid aws_cognito_required_scopes entry"
        ):
            self._settings(aws_cognito_required_scopes=['sco"pe'])

    def test_nothing_is_derived_without_a_resource_identifier(self) -> None:
        """A pool alone publishes no document, so it fills no discovery setting."""
        settings = self._settings(
            oauth_resource_identifier=None,
            aws_cognito_required_scopes=["stdapi/invoke"],
        )
        assert settings.oauth_authorization_servers == []
        assert settings.oauth_scopes_supported == []
