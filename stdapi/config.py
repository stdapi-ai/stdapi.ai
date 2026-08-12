"""Configuration management using Pydantic models.

Centralizes every environment variable the server reads, with type conversion
and validation.

Key Components:
- _DefaultModelParameters: Defines reusable model inference parameters
- _Settings: Main configuration class loaded from environment variables
- SETTINGS: Global configuration instance used throughout the application
"""

import re
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from aiobotocore.session import get_session
from aiohttp import ClientTimeout
from pydantic import (
    AliasChoices,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    JsonValue,
    SecretStr,
    Tag,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from stdapi.server import SERVER_NAME, SERVER_VERSION
from stdapi.utils import (
    match_bedrock_app_profile_arn,
    match_bedrock_prompt_router_arn,
    stdout_write,
)

if TYPE_CHECKING:
    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime.literals import ServiceTierTypeType
else:
    type RegionName = str
    type ServiceTierTypeType = str

#: HTTP download timeout
DOWNLOAD_TIMEOUT = ClientTimeout(total=20, connect=5)

#: Logging levels
LogLevel = Literal["info", "warning", "error", "critical"]

#: AWS Session
AWS_SESSION = get_session()

#: CloudWatch namespace charset (alphanumeric plus . - _ / # :), 1-255 characters.
_CLOUDWATCH_NAMESPACE_PATTERN = re.compile(r"^[A-Za-z0-9._\-/#:]{1,255}$")

#: Route prefix format: one or more "/segment" parts, no trailing slash.
_ROUTES_PREFIX_PATTERN = re.compile(r"^(/[A-Za-z0-9._~-]+)+$")

#: S3 prefix format: one or more "segment/" parts (S3-safe charset), no leading slash.
_S3_PREFIX_PATTERN = re.compile(r"^([A-Za-z0-9!_.*'()-]+/)+$")

#: KMS key ARN format for Bedrock CreateSession's ``encryptionKeyArn``, any AWS partition.
_KMS_KEY_ARN_PATTERN = re.compile(
    r"^arn:aws(?:-[a-z]+)*:kms:[a-zA-Z0-9-]*:[0-9]{12}:key/[a-zA-Z0-9-]{36}$"
)

#: Amazon Cognito user pool ID format: the pool's AWS Region, then its identifier.
_COGNITO_USER_POOL_ID_PATTERN = re.compile(r"^[a-z]{2}[a-z-]*-[0-9]+_[A-Za-z0-9]+$")

#: URL host: a registered name or a bracketed IP literal, with an optional port.
_URL_HOST = r"(?:\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9._~%!$&'()*+,;=-]+)(?::[0-9]+)?"

#: Origin URL format: scheme and host only, the form an OAuth resource identifier takes.
_ORIGIN_URL_PATTERN = re.compile(rf"^https?://{_URL_HOST}$")

#: OAuth 2.0 issuer identifier format: an "https" URL with no query or fragment (RFC 8414).
_ISSUER_URL_PATTERN = re.compile(
    rf"^https://{_URL_HOST}(?:/[A-Za-z0-9._~%!$&'()*+,;=:@/-]*)?$"
)

#: OAuth 2.0 scope token format: printable ASCII without space, quote or backslash (RFC 6749).
_OAUTH_SCOPE_PATTERN = re.compile(r"^[\x21\x23-\x5B\x5D-\x7E]+$")

#: IAM role ARN format, any AWS partition, with an optional path before the role name.
_IAM_ROLE_ARN_PATTERN = re.compile(
    r"^arn:aws(?:-[a-z]+)*:iam::[0-9]{12}:role(?:/[\w+=,.@-]+)*/[\w+=,.@-]+$"
)

#: Session tag key charset accepted by AWS STS AssumeRole (1-128 characters).
_SESSION_TAG_KEY_PATTERN = re.compile(r"^[\w.:/=+\-@ ]{1,128}$")

#: Built-in set of ``anthropic_beta`` flags known to be supported by AWS Bedrock.
_ANTHROPIC_BETA_BEDROCK_FLAGS: frozenset[str] = frozenset(
    {
        "computer-use-2024-10-22",
        "computer-use-2025-01-24",
        "computer-use-2025-11-24",
        "token-efficient-tools-2025-02-19",
        "Interleaved-thinking-2025-05-14",
        "output-128k-2025-02-19",
        "dev-full-thinking-2025-05-14",
        "context-1m-2025-08-07",
        "context-management-2025-06-27",
        "effort-2025-11-24",
        "tool-search-tool-2025-10-19",
        "tool-examples-2025-10-29",
    }
)

#: Guardrail trace levels accepted wherever a guardrail is configured.
GuardrailTrace = Literal["disabled", "enabled", "enabled_full"]


class ModelAliasConfig(BaseModel):
    """A model alias that carries configuration alongside its target model.

    Every field except ``model`` is optional and overrides the equivalent
    general configuration for requests naming the alias, while a value sent
    with the request still wins unless its override gate is closed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(
        description=(
            "Model ID or ARN this alias resolves to, as it would be written "
            "in the plain string form of the alias."
        )
    )
    service_tier: ServiceTierTypeType | None = Field(
        default=None,
        description=(
            "Service tier applied to requests naming this alias, overriding "
            "default_model_service_tiers for them. A request may still select "
            "another tier unless aws_bedrock_allow_service_tier_override is "
            "disabled.\n\n"
            "Like default_model_service_tiers, applies to models served by the "
            "Bedrock Converse and InvokeModel APIs; a Mantle-served model runs "
            "on the tier its own request names.\n\n"
            "Example: 'flex'"
        ),
    )
    guardrail_identifier: str | None = Field(
        default=None,
        validation_alias=AliasChoices("guardrail_identifier", "guardrail_id"),
        description=(
            "ID of the Amazon Bedrock Guardrail applied to requests naming "
            "this alias, overriding aws_bedrock_guardrail_identifier for them. "
            "Requires guardrail_version. May also be written 'guardrail_id'.\n\n"
            "Example: 'abcd1234efgh'"
        ),
    )
    guardrail_version: str | None = Field(
        default=None,
        description=(
            "Version of the Amazon Bedrock Guardrail applied to requests "
            "naming this alias. Requires guardrail_identifier.\n\n"
            "Example: 'DRAFT'"
        ),
    )
    guardrail_trace: GuardrailTrace | None = Field(
        default=None,
        description=(
            "Whether the guardrail trace is enabled for requests naming this "
            "alias. Requires guardrail_identifier."
        ),
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description=(
            "Key-value metadata attached to requests naming this alias, for "
            "audit reporting: it reaches Amazon Bedrock model invocation logs "
            "only, never Cost Explorer or CUR 2.0. A key also sent with the "
            "request wins.\n\n"
            'Example: {"team": "research"}'
        ),
    )
    extra_params: dict[str, JsonValue] | None = Field(
        default=None,
        description=(
            "Model parameters applied to requests naming this alias, in the "
            "same format as default_model_params, whose entry for the target "
            "model they override. A parameter sent with the request wins.\n\n"
            'Example: {"temperature": 0.2}'
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """Reject a guardrail that could never be applied.

        Returns:
            Self, once the guardrail fields agree.

        Raises:
            ValueError: When the guardrail configuration is incomplete.
        """
        if bool(self.guardrail_identifier) != bool(self.guardrail_version):
            msg = (
                "Both guardrail_identifier (guardrail_id) and guardrail_version "
                "are required to apply an Amazon Bedrock Guardrail from a model alias."
            )
            raise ValueError(msg)
        if self.guardrail_trace and not self.guardrail_identifier:
            msg = "guardrail_trace requires guardrail_identifier (guardrail_id)."
            raise ValueError(msg)
        return self


def _model_alias_kind(value: object) -> str:
    """Select the alias form a configured value uses.

    Args:
        value: Raw or already-validated ``MODEL_ALIASES`` entry.

    Returns:
        ``"configuration"`` for the object form, ``"model"`` for the plain
        target model ID, so an invalid object reports its own error instead of
        one error per union member.
    """
    return "configuration" if isinstance(value, dict | ModelAliasConfig) else "model"


#: A ``MODEL_ALIASES`` entry: a target model ID, or that model plus its configuration.
type ModelAliasTarget = Annotated[
    Annotated[str, Tag("model")] | Annotated[ModelAliasConfig, Tag("configuration")],
    Discriminator(_model_alias_kind),
]


class _Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Service region settings are optional and fall back to the default boto3
    session region; an S3 bucket must be in the same region as the service
    using it.

    Validation Rules:
    - Bedrock Guardrails require both identifier and version
    - API key sources are mutually exclusive
    - S3 buckets default to shared usage when not specified
    - Timezone must be a valid IANA timezone identifier

    See individual field descriptions for detailed parameter documentation.
    """

    model_config = SettingsConfigDict(env_ignore_empty=True)

    aws_adaptive_retry: bool = Field(
        default=False,
        description=(
            "Enable adaptive retry mode for all AWS service calls. "
            "When enabled, the client dynamically adjusts its retry behavior based on observed error rates: "
            "it slows down and spreads out retries when a service appears congested, "
            "and resumes normal pacing once conditions improve. "
            "This reduces the risk of amplifying load on an already-stressed endpoint, "
            "but may add latency to individual requests when throttling is detected. "
            "When disabled, retries follow a standard exponential backoff strategy. "
            "Default: false."
        ),
    )

    aws_max_pool_connections: int = Field(
        default=50,
        gt=0,
        description=(
            "Maximum number of concurrent HTTP connections per AWS service client. "
            "Each AWS service client (per region) maintains its own connection pool up to this limit. "
            "Increase this value if you observe connection pool exhaustion under high concurrency. "
            "Default: 50."
        ),
    )

    aws_connect_timeout: int = Field(
        default=5,
        gt=0,
        description=(
            "Timeout in seconds for establishing a connection to an AWS service endpoint, "
            "and for a real-time audio session to become ready in one region (connection, "
            "initial handshake and the first response). "
            "Keeping this value short allows fast failover to another region when a connection "
            "cannot be established. Increase it only if you experience spurious connection timeouts "
            "on high-latency networks, or real-time audio requests failing shortly after they start. "
            "Default: 5."
        ),
    )

    aws_s3_bucket: str | None = Field(
        default=None,
        description=(
            "AWS S3 bucket name for storing generated files and application data. "
            "This is the primary storage location for:\n"
            "- Generated images, audio, and documents\n"
            "- Uploaded user files for processing\n"
            "- Temporary files during multi-step operations\n\n"
            "Files are served via presigned URLs for secure, time-limited access. "
            "The bucket must be in the first region specified in aws_bedrock_regions "
            "(the primary region where your server should be hosted) for optimal "
            "performance and to avoid cross-region data transfer costs.\n\n"
            "Example: 'my-llm-storage-us-east-1'\n\n"
            "If not specified, some features will be disabled."
        ),
    )

    aws_s3_regional_buckets: dict[RegionName, str] = Field(
        default={},
        description=(
            "Region-specific S3 buckets for temporary file storage during Bedrock operations. "
            "Some models require S3 buckets in the same region for async and batch inference.\n\n"
            "Define buckets for regions where you need to use these features. "
            "Keys are AWS region identifiers, values are bucket names.\n\n"
            "Example: {'us-east-1': 'my-bedrock-temp-us-east-1', 'eu-west-1': 'my-bedrock-temp-eu-west-1'}\n\n"
            "If not specified for a region, operations requiring regional buckets may fail."
        ),
    )

    aws_s3_accelerate: bool = Field(
        default=False,
        description=(
            "Enable S3 Transfer Acceleration for presigned URLs to improve download "
            "performance for generated images. Transfer Acceleration uses CloudFront's "
            "globally distributed edge locations to optimize data transfer speeds, "
            "especially beneficial for geographically distributed users downloading "
            "high-resolution images.\n\n"
            "Requirements:\n"
            "- Transfer Acceleration must be enabled on the bucket specified by 'aws_s3_bucket'\n"
            "- Additional data transfer costs apply (see AWS S3 Transfer Acceleration pricing)\n\n"
            "Currently applies to: Image generation API (presigned URLs for generated images)."
        ),
    )

    aws_polly_region: RegionName | None = Field(
        default=None,
        description=(
            "AWS region for Polly text-to-speech service. When unset, every "
            "aws_bedrock_regions entry is a candidate: engine availability is "
            "discovered per region, and synthesis fails over across the "
            "regions offering the requested engine and voice."
        ),
    )

    aws_comprehend_region: RegionName | None = Field(
        default=None,
        description=(
            "AWS region for Comprehend language detection service. When unset, "
            "every aws_bedrock_regions entry is a candidate, tried in order "
            "with automatic failover on region-level errors."
        ),
    )

    aws_bedrock_regions: Annotated[list[RegionName], NoDecode] = Field(
        default=[],
        description=(
            "List of AWS regions where Bedrock AI models are available for use. "
            "The first region is the primary region where your server should be hosted "
            "on AWS for optimal performance. Your S3 bucket (aws_s3_bucket) must also "
            "be located in this region to minimize latency and data transfer costs.\n\n"
            "The server will attempt to use models in the order of regions specified, "
            "falling back to subsequent regions if a model is not available in "
            "the primary region. This enables access to region-specific models "
            "and provides redundancy.\n\n"
            "Environment variable format: Comma-separated string\n"
            "Example: 'us-east-1,us-west-2,eu-west-1'\n\n"
            "Common model availability by region:\n"
            "- us-east-1: Widest model selection, including latest releases\n"
            "- us-west-2: Good selection, often first for new models\n"
            "- eu-west-1: European compliance, subset of US models\n\n"
            "If not specified, the current region detected by the AWS SDK will be used."
        ),
    )

    aws_bedrock_region_routing: Literal[
        "disabled", "ordered", "lowest_latency", "round_robin"
    ] = Field(
        default="ordered",
        description=(
            "Automatic region routing strategy for Bedrock invocations.\n"
            "Distributes requests across configured regions to handle quota limits "
            "and regional unavailability.\n\n"
            "Strategies:\n"
            "- 'disabled': No routing, uses single region per model\n"
            "- 'ordered': Try regions in configured order, skip blocked ones (default). "
            "Best for prompt caching compatibility.\n"
            "- 'lowest_latency': Prefer the region with lowest measured latency.\n"
            "- 'round_robin': Distribute evenly across regions. "
            "Incompatible with prompt caching.\n\n"
            "Requires at least 2 regions in aws_bedrock_regions to take effect."
        ),
    )

    aws_bedrock_region_routing_quota_backoff_seconds: int = Field(
        default=60,
        gt=0,
        description=(
            "Seconds to avoid a region after receiving a quota/throttling error. "
            "Only effective when aws_bedrock_region_routing is not 'disabled'."
        ),
    )

    aws_bedrock_region_routing_unavailable_backoff_seconds: int = Field(
        default=30,
        gt=0,
        description=(
            "Seconds to avoid a region after receiving an unavailability error. "
            "Only effective when aws_bedrock_region_routing is not 'disabled'."
        ),
    )

    aws_bedrock_region_routing_max_quota_backoff_seconds: int = Field(
        default=3600,
        gt=0,
        description=(
            "Hard ceiling in seconds on the exponential quota backoff for a single region. "
            "Quota backoff doubles on each consecutive error; this value caps how high it can grow. "
            "Only effective when aws_bedrock_region_routing is not 'disabled'. "
            "Default: 3600 (1 hour)."
        ),
    )

    aws_bedrock_region_routing_quota_stale_factor: int = Field(
        default=2,
        gt=0,
        description=(
            "Multiplier applied to the max quota backoff to compute the stale-error threshold. "
            "If the most recent quota error for a region is older than "
            "(max_quota_backoff * factor) seconds, the consecutive-error counter is reset "
            "and the next error is treated as a fresh start rather than an escalation. "
            "A higher value keeps memory of past errors for longer before resetting. "
            "Only effective when aws_bedrock_region_routing is not 'disabled'. "
            "Default: 2 (threshold = 2 * max_quota_backoff)."
        ),
    )

    aws_bedrock_max_retries: int = Field(
        default=9,
        ge=0,
        description=(
            "Maximum number of retries for Bedrock invocations. "
            "When region routing is enabled, each retry escalates to the next available region "
            "and every region is tried at most once, so the attempts are also bounded by the "
            "number of candidate regions. "
            "When region routing is disabled, retries are performed against the single configured region."
        ),
    )

    aws_failover_max_retries: int = Field(
        default=2,
        ge=0,
        description=(
            "Maximum number of SDK retry attempts per candidate region for the "
            "multi-region failover services (Polly, Transcribe, Translate, Comprehend). "
            "Only applied when the service has several candidate regions "
            "(no explicit region setting); failover across regions replaces "
            "deep in-region retrying. "
            "Default: 2."
        ),
    )

    aws_bedrock_mantle_enabled: bool = Field(
        default=True,
        description=(
            "If true (default), expose models served by the Amazon Bedrock Mantle "
            "endpoint (OpenAI/Anthropic-compatible APIs) in addition to the classic "
            "Bedrock Converse models. Mantle-only models (e.g. OpenAI GPT, xAI "
            "Grok, Google Gemma 4) become available on the chat completions, "
            "responses, messages and completions routes. Models available on both "
            "the classic bedrock-runtime endpoint and Mantle are served by "
            "bedrock-runtime unless listed in aws_bedrock_mantle_preferred_models.\n\n"
            "When Bedrock Mantle is unreachable or the IAM role lacks "
            "bedrock-mantle permissions, Mantle models are simply not listed and "
            "a warning is logged at startup.\n\n"
            "Note: Amazon Bedrock Guardrails are not supported on Mantle-served "
            "requests."
        ),
    )

    aws_bedrock_mantle_regions: Annotated[list[RegionName], NoDecode] = Field(
        default=[],
        description=(
            "List of AWS regions used for Amazon Bedrock Mantle, in failover "
            "priority order. Defaults to aws_bedrock_regions when unset. "
            "Model availability differs per region; the served model catalog is "
            "the union of all listed regions.\n\n"
            "Environment variable format: Comma-separated string"
        ),
    )

    aws_bedrock_mantle_endpoint_url: str | None = Field(
        default=None,
        description=(
            "Override the Amazon Bedrock Mantle endpoint URL template. "
            "The '{region}' placeholder is substituted with the target region. "
            "Default: 'https://bedrock-mantle.{region}.api.aws'."
        ),
    )

    aws_bedrock_mantle_preferred_models: Annotated[list[str], NoDecode] = Field(
        default=[],
        description=(
            "Model IDs (or ID prefixes) served by Amazon Bedrock Mantle even when "
            "also available on the classic bedrock-runtime endpoint. Useful to "
            "leverage Mantle's independent throughput quotas or native response "
            "storage for selected models.\n\n"
            "Environment variable format: Comma-separated string\n"
            "Example: 'anthropic.claude-haiku-4-5,openai.gpt-oss'"
        ),
    )

    aws_bedrock_mantle_service_header: bool = Field(
        default=False,
        description=(
            "If true, honor the 'x-stdapi-service: bedrock-mantle' request "
            "header to route a model available on both endpoints through "
            "Bedrock Mantle for that request instead of the default "
            "bedrock-runtime serving.\n\n"
            "Cannot be enabled together with Amazon Bedrock Guardrails: guardrails "
            "do not apply to Mantle-served requests, so a per-request header would "
            "allow clients to bypass them."
        ),
    )

    aws_bedrock_mantle_project: str | None = Field(
        default=None,
        description=(
            "Default Amazon Bedrock Mantle project (workspace) ID used to attribute "
            "Mantle inference requests for cost tracking and observability. Applies "
            "only to models served by the Bedrock Mantle endpoint.\n\n"
            "The same project ID is sent as the 'OpenAI-Project' header on the "
            "OpenAI-compatible APIs and as the 'anthropic-workspace' header on the "
            "Anthropic Messages API. Use the bare project ID (e.g. 'proj_abc123' or "
            "'default'), not the ARN. When unset, requests fall to the account's "
            "default project."
        ),
    )

    aws_bedrock_allow_mantle_project_override: bool = Field(
        default=False,
        description=(
            "Allow users to override the configured Bedrock Mantle project at request "
            "level using the 'OpenAI-Project' or 'anthropic-workspace' header. When "
            "disabled and a default project is configured, request headers are ignored. "
            "When no default project is configured, request headers are always honored "
            "regardless of this setting. Defaults to False."
        ),
    )

    aws_bedrock_user_role_arn: str | None = Field(
        default=None,
        description=(
            "ARN of an AWS IAM role the server assumes, once per end user, to run "
            "that user's model invocations under an identity of their own. AWS then "
            "reports Amazon Bedrock model usage per end user in AWS Cost Explorer and "
            "in the Cost and Usage Report, instead of reporting it all under the "
            "server's own identity.\n\n"
            "The role must grant the Amazon Bedrock model invocation actions, and its "
            "trust policy must allow the server's own role to call both "
            "'sts:AssumeRole' and 'sts:TagSession' on it. The server's role needs the "
            "same two actions on this role ARN.\n\n"
            "Example: 'arn:aws:iam::123456789012:role/stdapi-ai-end-user'\n\n"
            "Only model invocations are covered. When unset (default), every request "
            "runs under the server's own identity and usage is reported under it."
        ),
    )

    aws_bedrock_user_role_session_duration: int = Field(
        default=3600,
        ge=900,
        le=3600,
        description=(
            "Lifetime in seconds of the per-end-user role session obtained with "
            "aws_bedrock_user_role_arn. Sessions are cached and reused until they "
            "approach expiry, so a longer lifetime means fewer AWS STS calls.\n\n"
            "Accepted range: 900 to 3600 seconds. The 3600-second ceiling is imposed "
            "by AWS: the server itself runs under an assumed role, and a role session "
            "obtained from another role session cannot last longer than one hour.\n\n"
            "Example: 3600\n\n"
            "Defaults to 3600."
        ),
    )

    aws_bedrock_user_role_tag_key: str | None = Field(
        default="user",
        description=(
            "Session tag key carrying the end user identity on the role sessions "
            "obtained with aws_bedrock_user_role_arn. Activate it as a cost "
            "allocation tag, under the 'IAM principal' type, to group Amazon Bedrock "
            "costs by end user in AWS Cost Explorer, and to write access policies "
            "conditioned on 'aws:PrincipalTag/<key>'.\n\n"
            "Keys beginning with 'aws:' are reserved by AWS and rejected.\n\n"
            "Example: 'user'\n\n"
            "Set to null to send no session tag: end users are then still "
            "distinguished by the role session name. Defaults to 'user'."
        ),
    )

    aws_bedrock_user_role_require_identity: bool = Field(
        default=False,
        description=(
            "Reject a model invocation that identifies no end user, instead of "
            "running it under the server's own identity, when "
            "aws_bedrock_user_role_arn is configured.\n\n"
            "Enable it so that no model usage can escape per-user attribution. "
            "Requests must then carry an authenticated caller, or name the end user "
            "with 'safety_identifier' or 'user' on the OpenAI-compatible APIs, or "
            "with 'metadata.user_id' on the Anthropic Messages API. APIs that have "
            "no such field, such as audio transcription, then require an "
            "authenticated caller.\n\n"
            "A real-time speech-to-speech session is refused outright while this is "
            "enabled: it keeps for its whole life the identity it opened with, so it "
            "can only ever be attributed to the server.\n\n"
            "Defaults to False, which attributes such requests to the server itself."
        ),
    )

    aws_bedrock_external_web_access: bool = Field(
        default=False,
        description=(
            "Allow the built-in web search tool to reach the external web. Searches "
            "are answered from the Amazon Bedrock web index and cache either way, "
            "and answers are current and carry source citations.\n\n"
            "AWS documents that retrieval is served entirely from that index and "
            "cache today, so no request data leaves the AWS boundary even when this "
            "is enabled, and that a future release may allow live external "
            "retrieval, at which point request data may leave it. Enabling this is "
            "therefore a decision taken in advance about behaviour that can change.\n\n"
            "Enabling it also requires the 'bedrock-websearch:ExternalWebAccess' IAM "
            "permission on the credentials this server uses. Each action is "
            "authorized only when a model actually attempts it, and a denied call "
            "does not fail the request. "
            "See https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html\n\n"
            "Defaults to False."
        ),
    )

    aws_bedrock_allow_external_web_access_override: bool = Field(
        default=False,
        description=(
            "Allow users to override aws_bedrock_external_web_access at request level "
            "with the web search tool's 'external_web_access' field. When disabled, a "
            "request that sets the field to anything other than the configured value is "
            "rejected instead of being silently overridden. Defaults to False."
        ),
    )

    aws_s3_accepted_buckets: dict[str, RegionName] = Field(
        default={},
        description=(
            "S3 buckets that the application has read access to, mapped to their region. "
            "These buckets can be used as input S3 data sources, and S3 HTTP URLs "
            "(including presigned URLs) for these buckets will be automatically converted "
            "to S3 URIs for direct access.\n\n"
            "Keys are bucket names, values are AWS region identifiers.\n\n"
            "Example: {'my-data-bucket': 'us-east-1', 'my-eu-bucket': 'eu-west-1'}\n\n"
            "If not specified, only the application's own S3 buckets (aws_s3_bucket and "
            "aws_s3_regional_buckets) are recognized for S3 URI conversion."
        ),
    )

    aws_bedrock_model_region_restrict: dict[str, tuple[RegionName, ...]] = Field(
        default={},
        description=(
            "Restrict a model to specific region(s) only. Can be used when a model "
            "provides important features only in certain regions.\n\n"
            "Keys are Bedrock model IDs (or prefixes), values are ordered lists of "
            "allowed regions. When set, the model will only be available in the listed "
            "regions (intersected with the regions where it is actually available), "
            "and the list order defines the routing priority when the default "
            '"ordered" routing strategy is used. '
            "No fallback to other regions occurs.\n\n"
            "Example: {'amazon.nova-pro-v1:0': ['us-east-1']}\n\n"
            "Use case: Nova grounding is only available in us-east-1, so restricting "
            "nova-pro to us-east-1 ensures grounding always works."
        ),
    )

    aws_bedrock_cross_region_inference: bool = Field(
        default=True, description="If true, allow cross region inference to be used."
    )

    aws_bedrock_cross_region_inference_global: bool = Field(
        default=True,
        description='If True, allow "global" cross region inference to be used that can route requests to any region, worldwide.\n'
        'Can be set to False if you want to restrict to regional inference only (Example: "eu", "us", ...), '
        "this can be useful to comply with regulations like EU GDPR.\n"
        "If set to True, global cross region inference is preferred over regional inference if both are available.",
    )

    aws_bedrock_legacy: bool = Field(
        default=False, description="If true, allow legacy Bedrock models to be used."
    )

    aws_bedrock_deprecated_model_fallback: bool = Field(
        default=True,
        description=(
            "If true, requests that use a deprecated model ID are transparently retried "
            "with the recommended replacement model instead of returning a 404 error.\n\n"
            "The replacement mapping is defined in stdapi/models/deprecation.py. "
            "Disable this if you want deprecated model IDs to fail explicitly so "
            "clients are forced to migrate."
        ),
    )

    aws_bedrock_deprecated_models: dict[str, str] = Field(
        default={},
        description=(
            "Additional deprecated model ID mappings, merged with the built-in deprecation registry at startup. "
            "User-provided entries take precedence over built-in ones, so this can also be used to override "
            "the fallback target of an already-defined deprecated model.\n\n"
            "Keys are deprecated model IDs, values are the recommended replacement model IDs.\n\n"
            "Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html\n\n"
            "Example: {'my-old-model-v1': 'my-new-model-v2'}"
        ),
    )

    aws_bedrock_marketplace_auto_subscribe: bool = Field(
        default=True,
        description=(
            "If true, allow the server to automatically subscribe to new models in the AWS Marketplace. "
            "When set to false, models that do not have marketplace entitlements will be hidden in the server. "
            "This provides control over which models are accessible through the API.\n\n"
            "Required IAM permissions when set to true:\n"
            "- aws-marketplace:Subscribe\n"
            "- aws-marketplace:ViewSubscriptions"
        ),
    )

    aws_bedrock_allow_cross_region_inference_profile_arn: bool = Field(
        default=False,
        description=(
            "If True, allow users to pass cross-region inference profile ARNs directly as model IDs. "
            "Cross-region inference profiles enable routing to multiple regions for better availability. "
            "When disabled, only standard model IDs and configured profiles are accepted.\n\n"
            "Example ARN: arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        ),
    )

    aws_bedrock_allow_application_inference_profile_arn: bool = Field(
        default=False,
        description=(
            "If True, allow users to pass application inference profile ARNs directly as model IDs. "
            "Application inference profiles are custom routing configurations for specific use cases. "
            "When disabled, only standard model IDs and configured profiles are accepted.\n\n"
            "Example ARN: arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123xyz"
        ),
    )

    aws_bedrock_allow_prompt_router_arn: bool = Field(
        default=False,
        description=(
            "If True, allow users to pass prompt router ARNs directly as model IDs. "
            "Prompt routers enable dynamic model selection based on prompt characteristics. "
            "When disabled, only standard model IDs and configured profiles are accepted.\n\n"
            "Example ARN: arn:aws:bedrock:us-east-1:123456789012:default-prompt-router/my-router"
        ),
    )

    aws_bedrock_allow_prompt_arn: bool = Field(
        default=False,
        description=(
            "If True, allow users to reference an Amazon Bedrock Prompt Management prompt ARN "
            "in the OpenAI Responses API `prompt.id` parameter. "
            "The prompt template is rendered by Amazon Bedrock and its variables are filled from "
            "`prompt.variables`. When disabled, any `prompt` parameter is rejected with a 400 error.\n\n"
            "Required IAM permissions when set to true:\n"
            "- bedrock:GetPrompt\n"
            "- bedrock:RenderPrompt\n\n"
            "Example ARN: arn:aws:bedrock:us-east-1:123456789012:prompt/ABCDE12345:1"
        ),
    )

    aws_bedrock_model_arn_mapping: dict[str, str] = Field(
        default={},
        description=(
            "Map standard model IDs to custom inference profile or prompt router ARNs. "
            "This allows server administrators to override the default cross-region inference profiles "
            "with custom application inference profiles, cross-region inference profiles, or prompt routers.\n\n"
            "The mapped ARN will be used instead of the default profile when clients request the model by its ID. "
            "This provides centralized control over model routing without requiring client changes.\n\n"
            "Supported ARN types:\n"
            "- Cross-region inference profile: arn:aws:bedrock:REGION:ACCOUNT:inference-profile/ID\n"
            "- Application inference profile: arn:aws:bedrock:REGION:ACCOUNT:application-inference-profile/ID\n"
            "- Prompt router: arn:aws:bedrock:REGION:ACCOUNT:default-prompt-router/ID\n\n"
            "Example: {\n"
            '  "anthropic.claude-3-5-sonnet-20241022-v2:0": "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-custom-profile",\n'
            '  "anthropic.claude-3-5-haiku-20241022-v1:0": "arn:aws:bedrock:us-east-1:123456789012:default-prompt-router/my-router"\n'
            "}"
        ),
    )

    aws_bedrock_guardrail_identifier: str | None = Field(
        default=None,
        description=(
            "Configure Amazon Bedrock Guardrails to include safeguards in model input and responses. "
            "The ID of the guardrail. Version is also required."
        ),
    )

    aws_bedrock_guardrail_version: str | None = Field(
        default=None,
        description=(
            "Configure Amazon Bedrock Guardrails to include safeguards in model input and responses. "
            "The version of the guardrail. ID is also required."
        ),
    )

    aws_bedrock_guardrail_trace: GuardrailTrace | None = Field(
        default=None,
        description=(
            "Configure Amazon Bedrock Guardrails to include safeguards in model input and responses. "
            "Whether or not to enable the guardrail trace."
        ),
    )

    aws_bedrock_allow_guardrail_override: bool = Field(
        default=False,
        description=(
            "Allow users to override the global guardrail configuration at request level using headers. "
            "When enabled, users can specify per-request guardrail settings via:\n"
            "- X-Amzn-Bedrock-GuardrailIdentifier\n"
            "- X-Amzn-Bedrock-GuardrailVersion\n"
            "- X-Amzn-Bedrock-Trace\n\n"
            "When disabled and a global guardrail is configured, request headers are ignored for security. "
            "Note: If no global guardrail is configured, request headers are always allowed regardless of this setting. "
            "Defaults to False for security to prevent users from bypassing configured safety controls."
        ),
    )

    aws_bedrock_allow_service_tier_override: bool = Field(
        default=True,
        description=(
            "Allow users to select the service tier at request level, through the "
            "'service_tier' request parameter or the X-Amzn-Bedrock-Service-Tier header. "
            "When disabled, a request cannot change the tier configured for the model "
            "by default_model_service_tiers or by the model alias it names, which keeps "
            "the cost profile of a shared deployment under the administrator's control. "
            "A model with no configured tier still honors the request in either case. "
            "Applies to models served by the Bedrock Converse and InvokeModel APIs. "
            "Defaults to True, matching the behavior of previous versions."
        ),
    )

    aws_bedrock_session_encryption_key_arn: str | None = Field(
        default=None,
        description=(
            "KMS key ARN encrypting AWS Bedrock session storage backing "
            "stored responses and chat completions (store=true).\n\n"
            "Unset (default): sessions are encrypted with the AWS-managed key."
        ),
    )

    aws_transcribe_region: RegionName | None = Field(
        default=None,
        description=(
            "AWS region for Transcribe speech-to-text service. When unset, "
            "every aws_bedrock_regions entry with a co-located S3 bucket "
            "(aws_transcribe_s3_bucket/aws_s3_bucket for the primary region, "
            "aws_s3_regional_buckets for the others) is a candidate, tried "
            "in order with automatic failover on region-level errors."
        ),
    )

    aws_transcribe_s3_bucket: str | None = Field(
        default=None,
        description=(
            "AWS S3 bucket name for temporary file storage during transcription. "
            "Must be in the same region as aws_transcribe_region. "
            "Defaults to aws_s3_bucket if not specified."
        ),
    )

    aws_s3_tmp_prefix: str = Field(
        default="tmp/",
        description=(
            "S3 prefix (folder path) for temporary files used during job processing. "
            "This prefix is used for all temporary files including:\n"
            "- Generated images, audio, and documents (in aws_s3_bucket)\n"
            "- Transcription workflow files (in aws_transcribe_s3_bucket)\n\n"
            "Configure S3 lifecycle policies to automatically delete objects under "
            "this prefix after 1 day to minimize storage costs.\n\n"
            "Example: 'tmp/' stores files under s3://bucket/tmp/\n"
            "Example: 'temporary/' stores files under s3://bucket/temporary/\n"
            "Example: '' (empty string) stores files at bucket root (not recommended)"
        ),
    )

    aws_s3_files_prefix: str = Field(
        default="files/",
        description=(
            "S3 prefix (folder path) for Files API objects.\n\n"
            "Configure S3 Lifecycle rules on this prefix to:\n"
            "- Automatically delete expired files (tag-based, see Terraform module)\n"
            "- Apply S3 Intelligent-Tiering for cost optimisation\n\n"
            "Example: 'files/' stores objects under s3://bucket/files/\n"
            "Example: 'uploads/files/' stores objects under s3://bucket/uploads/files/\n"
            "Example: '' (empty string) stores files at bucket root (not recommended)"
        ),
    )

    aws_s3_videos_prefix: str = Field(
        default="videos/",
        description=(
            "S3 prefix (folder path) for generated videos (Videos API).\n\n"
            "AWS Bedrock writes each video generation job's output under this "
            "prefix, in a folder named after the job (in the regional bucket "
            "of the region that served the job). Videos persist until deleted "
            "through the API; configure an S3 Lifecycle rule on this prefix "
            "to cap storage costs.\n\n"
            "Example: 'videos/' stores objects under s3://bucket/videos/\n"
            "Example: '' (empty string) stores videos at bucket root (not recommended)"
        ),
    )

    aws_s3_videos_expires_after: int | None = Field(
        default=None,
        ge=3600,
        description=(
            "Retention period in seconds for generated videos (Videos API).\n\n"
            "When set, `Video.expires_at` reports the job completion time plus "
            "this value and expired video content can no longer be downloaded "
            "(404). The server does not delete objects itself: pair this "
            "setting with an S3 Lifecycle expiration rule on "
            "AWS_S3_VIDEOS_PREFIX covering the same duration (rounded up to "
            "whole days).\n\n"
            "Unset (default): videos never expire and persist until deleted "
            "through the API."
        ),
    )

    aws_s3_batches_prefix: str = Field(
        default="batches/",
        description=(
            "S3 prefix (folder path) for the Batch API's own data — the "
            "submitted requests, the results, and the batch records "
            "themselves.\n\n"
            "Each batch stores its data under a folder of its own below this "
            "prefix, in the regional bucket that served it. Configure an S3 "
            "Lifecycle rule on this prefix to cap storage costs; results stay "
            "readable for as long as the objects exist.\n\n"
            "Example: 'batches/' stores objects under s3://bucket/batches/\n"
            "Example: '' (empty string) is rejected: the prefix must not be "
            "the bucket root."
        ),
    )

    aws_bedrock_batch_role_arn: str | None = Field(
        default=None,
        description=(
            "ARN of the AWS IAM service role that Amazon Bedrock assumes to run "
            "batch inference jobs. Required to enable the Batch API; the batch "
            "endpoints answer 503 while it is unset.\n\n"
            "The role's trust policy must allow 'bedrock.amazonaws.com' to "
            "assume it, and the role must be able to read from and write to "
            "every bucket configured with aws_s3_bucket / "
            "aws_s3_bucket_<region>, under aws_s3_batches_prefix. The server's "
            "own role needs 'iam:PassRole' on this ARN. See "
            "https://docs.aws.amazon.com/bedrock/latest/userguide/batch-iam-sr.html\n\n"
            "Example: 'arn:aws:iam::123456789012:role/stdapi-ai-batch'\n\n"
            "Unset (default): the Batch API is disabled."
        ),
    )

    aws_translate_region: RegionName | None = Field(
        default=None,
        description=(
            "AWS region for Translate text translation service. When unset, "
            "every aws_bedrock_regions entry is a candidate, tried in order "
            "with automatic failover on region-level errors."
        ),
    )

    timezone: ZoneInfo = Field(
        default=ZoneInfo("UTC"), description="Timezone for request date & time"
    )

    openai_routes_prefix: str = Field(
        default="", description="OpenAI API compatible routes prefix"
    )

    anthropic_routes_prefix: str = Field(
        default="/anthropic", description="Anthropic API compatible routes prefix"
    )

    cohere_routes_prefix: str = Field(
        default="/cohere", description="Cohere API compatible routes prefix"
    )

    api_key: SecretStr | None = Field(
        default=None,
        description=(
            "API key for client authentication. When specified, all API requests "
            "must include this key in the Authorization header as 'Bearer <key>' "
            "or in the 'X-API-Key' header.\n\n"
            "If not specified, authentication is disabled and the API accepts "
            "all requests (suitable for internal/private deployments only).\n\n"
            "Security note: Use environment variable or secure parameter store "
            "rather than hardcoding in configuration files.\n"
            "Example: 'sk-1234567890abcdef...'"
        ),
    )

    api_key_ssm_parameter: str | None = Field(
        default=None,
        description=(
            "AWS Systems Manager Parameter Store parameter name containing the API key. "
            "This is the recommended approach for secure API key storage in AWS "
            "environments as it supports encryption, access control, and auditing.\n\n"
            "Takes precedence over other API key sources if multiple are specified. "
            "The parameter should be of type 'SecureString' for encryption at rest.\n\n"
            "Example: '/llm/prod/api-key' or '/myapp/secrets/auth-token'\n\n"
            "Required IAM permissions: ssm:GetParameter, kms:Decrypt (if encrypted)"
        ),
    )

    api_key_secretsmanager_secret: str | None = Field(
        default=None,
        description=(
            "AWS Secrets Manager secret name containing the API key. "
            "Used for secure key storage with automatic rotation support "
            "and fine-grained access control.\n\n"
            "Only used if api_key_ssm_parameter is not specified. The secret "
            "can be a simple string or JSON object (use api_key_secretsmanager_key "
            "to specify the JSON key name).\n\n"
            "Example: 'llm-api-credentials' or 'prod/llm/auth'\n\n"
            "Required IAM permissions: secretsmanager:GetSecretValue"
        ),
    )

    api_key_secretsmanager_key: str = Field(
        default="api_key",
        description=(
            "Key name within the AWS Secrets Manager secret containing the API key. "
            "Used only with api_key_secretsmanager_secret. Defaults to 'api_key' if not specified."
        ),
    )

    authentication_mode: Literal["any", "api_key", "cognito"] = Field(
        default="any",
        description=(
            "Which client authentication methods this deployment accepts:\n"
            "- 'any': every method that is configured (default)\n"
            "- 'api_key': the API key only\n"
            "- 'cognito': Amazon Cognito user pool tokens only\n\n"
            "The value asserts the intended security posture: startup fails when "
            "the selected method is not configured, and when a method that would "
            "be ignored is configured anyway, so a credential is never accepted "
            "or silently refused by accident.\n"
            "Example: 'cognito'"
        ),
    )

    aws_cognito_user_pool_id: str | None = Field(
        default=None,
        description=(
            "Identifier of the Amazon Cognito user pool whose tokens authenticate "
            "clients. Clients send a pool access token in the "
            "'Authorization: Bearer <token>' header; the token signature, issuer, "
            "expiry, application and scopes are validated on every request.\n\n"
            "The identifier is prefixed by the pool's AWS Region, which is where "
            "the signing keys are read from. Requires aws_cognito_client_ids.\n\n"
            "If not specified, user pool authentication is disabled.\n"
            "Example: 'eu-west-3_a1b2c3d4e'"
        ),
    )

    aws_cognito_client_ids: Annotated[list[str], NoDecode] = Field(
        default=[],
        description=(
            "Amazon Cognito user pool application client IDs whose tokens are "
            "accepted, as a comma-separated list. A token issued to any other "
            "application is rejected.\n\n"
            "Required when aws_cognito_user_pool_id is set.\n"
            "Example: '1example23456789abcdefghij,2example3456789abcdefghijk'"
        ),
    )

    aws_cognito_required_scopes: Annotated[list[str], NoDecode] = Field(
        default=[],
        description=(
            "OAuth 2.0 scopes a token must all carry to be accepted, as a "
            "comma-separated list.\n\n"
            "Custom scopes exist only on tokens obtained from the user pool's "
            "OAuth 2.0 token endpoint, which requires a resource server and a "
            "pool domain. Tokens obtained by signing in with a username and "
            "password carry only 'aws.cognito.signin.user.admin', so requiring a "
            "custom scope rejects them.\n\n"
            "If not specified, any scope set is accepted.\n"
            "Example: 'stdapi/invoke'"
        ),
    )

    aws_cognito_accept_id_token: bool = Field(
        default=False,
        description=(
            "Accept Amazon Cognito identity tokens in addition to access tokens.\n\n"
            "Identity tokens describe the signed-in user rather than granting API "
            "access, and carry no scopes. Enable only for clients that cannot "
            "obtain an access token.\n"
            "Example: 'true'"
        ),
    )

    aws_cognito_issuer_type: Literal["original", "updated"] = Field(
        default="original",
        description=(
            "Issuer configuration of the Amazon Cognito user pool, which decides "
            "the issuer URL its tokens carry:\n"
            "- 'original': 'https://cognito-idp.<region>.amazonaws.com/<pool-id>' "
            "(default)\n"
            "- 'updated': 'https://issuer-cognito-idp.<region>.amazonaws.com/"
            "<pool-id>', available on the Essentials and Plus pool tiers\n\n"
            "Tokens whose issuer does not match are rejected, so this must match "
            "the pool's own setting.\n"
            "Example: 'updated'"
        ),
    )

    oauth_resource_identifier: str | None = Field(
        default=None,
        description=(
            "Public URL clients use to reach this API. Setting it publishes an "
            "OAuth 2.0 protected resource metadata document at "
            "'<url>/.well-known/oauth-protected-resource' and puts that address "
            "in the challenge every 401 response carries, so an AI agent can "
            "discover where to obtain a token instead of being configured for "
            "this deployment beforehand.\n\n"
            "Clients compare the value against the URL they dialled character "
            "by character, so it must be the exact origin they use: scheme and "
            "host, an explicit port only when it is not the default one for the "
            "scheme, and no path, query or fragment.\n\n"
            "Requires oauth_authorization_servers. If not specified, no "
            "metadata document is published and 401 responses only state that a "
            "bearer token is expected.\n"
            "Example: 'https://api.example.com'"
        ),
    )

    oauth_authorization_servers: Annotated[list[str], NoDecode] = Field(
        default=[],
        description=(
            "Issuer URLs of the OAuth 2.0 authorization servers that issue "
            "tokens for this API, as a comma-separated list, published in its "
            "protected resource metadata. A client reads each issuer's own "
            "metadata to find where to sign in, so this deployment never has to "
            "describe the sign-in flow itself.\n\n"
            "An Amazon Cognito user pool issues "
            "'https://cognito-idp.<region>.amazonaws.com/<pool-id>', or "
            "'https://issuer-cognito-idp.<region>.amazonaws.com/<pool-id>' when "
            "the pool uses the updated issuer (see aws_cognito_issuer_type). A "
            "load balancer or API gateway authenticating in front of this "
            "server publishes the issuer of whichever provider it uses.\n\n"
            "Required when oauth_resource_identifier is set.\n"
            "Example: 'https://cognito-idp.eu-west-3.amazonaws.com/eu-west-3_a1b2c3d4e'"
        ),
    )

    oauth_scopes_supported: Annotated[list[str], NoDecode] = Field(
        default=[],
        description=(
            "OAuth 2.0 scopes a token needs to call this API, as a "
            "comma-separated list. Published in the protected resource metadata "
            "and requested in the 401 challenge, so a client asks its "
            "authorization server for the right scopes on its first attempt.\n\n"
            "Set it to the same value as aws_cognito_required_scopes when an "
            "Amazon Cognito user pool authenticates clients.\n\n"
            "Requires oauth_resource_identifier. If not specified, no scope is "
            "advertised and the client asks for whatever its own configuration "
            "names.\n"
            "Example: 'stdapi/invoke'"
        ),
    )

    otel_enabled: bool = Field(
        default=False,
        description=(
            "Enable OpenTelemetry distributed tracing for observability and debugging. "
            "When enabled, the server will instrument HTTP requests, AWS service calls, "
            "and internal operations to provide detailed performance insights.\n\n"
            "Integrates seamlessly with AWS X-Ray for end-to-end trace visualization "
            "and with other OTEL-compatible systems like Jaeger or DataDog.\n\n"
            "Set to false to disable all tracing overhead (recommended for "
            "performance-critical deployments where observability is not needed)."
        ),
    )

    otel_service_name: str = Field(
        default="stdapi.ai",
        description=(
            "Service name identifier for OpenTelemetry traces. This name appears "
            "in trace visualizations to distinguish this service from others in "
            "your distributed system.\n\n"
            "Use descriptive names that include environment information for clarity:\n"
            "- 'llm-production'\n"
            "- 'llm-staging-us-east-1'\n"
            "- 'my-ai-service-v2'\n\n"
            "This helps identify traces in complex microservice architectures "
            "and multi-environment deployments."
        ),
    )

    otel_exporter_endpoint: str = Field(
        default="http://127.0.0.1:4318/v1/traces",
        description=(
            "OpenTelemetry traces export endpoint URL. Traces are sent here "
            "in OTLP (OpenTelemetry Protocol) format for processing and storage.\n\n"
            "Common configurations:\n"
            "- AWS ADOT Collector: 'http://127.0.0.1:4318/v1/traces' (default)\n"
            "- Jaeger: 'http://jaeger:14268/api/traces'\n"
            "- Direct X-Ray: Use ADOT collector as intermediary\n"
            "- Cloud services: Provider-specific OTLP endpoints\n\n"
            "The endpoint must support OTLP HTTP protocol. For AWS X-Ray "
            "integration, use the AWS Distro for OpenTelemetry (ADOT) collector "
            "as an intermediary."
        ),
    )

    otel_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "OpenTelemetry trace sampling rate (0.0 to 1.0). Controls what "
            "percentage of requests generate traces to balance observability "
            "with performance and storage costs.\n\n"
            "Sampling recommendations:\n"
            "- 1.0 (100%): Development, debugging, low-traffic services\n"
            "- 0.1 (10%): Production services with moderate traffic\n"
            "- 0.01 (1%): High-traffic production services\n"
            "- 0.0 (0%): Equivalent to disabling tracing\n\n"
            "Higher sampling rates provide more detailed insights but increase "
            "system overhead and storage costs. Adjust based on your traffic "
            "volume and monitoring budget."
        ),
    )

    log_request_params: bool = Field(
        default=False,
        description=(
            "If True, add requests and responses parameters "
            "(JSON body, form, query) to logs. "
            "Can be used to debug integration, but should not be enabled in production."
            "Defaults to False."
        ),
    )

    log_client_ip: bool = Field(
        default=False,
        description=(
            "If True, log the client IP address for each request and add it to OpenTelemetry spans. "
            "When enable_proxy_headers is True, logs the real client IP from X-Forwarded-For header. "
            "When enable_proxy_headers is False, logs the direct connection IP (typically the proxy IP). "
            "The IP is added as 'client.address' attribute to OTEL spans when otel_enabled is True. "
            "Defaults to False for privacy considerations."
        ),
    )

    log_level: LogLevel | Literal["disabled"] = Field(
        default="info",
        description=(
            "Minimum logging level to output. Only log events at or above this level "
            "will be written to STDOUT. Log levels in order of severity: info < warning < error < critical. "
            "Set to 'disabled' to suppress all log output (not recommended). "
            "Example: Setting 'warning' will output warning, error, and critical events, but suppress info events."
        ),
    )

    strict_input_validation: bool = Field(
        default=False,
        description="If True, raise error on extra fields in input request.",
    )

    chat_completions_reasoning_field: Literal[
        "reasoning_content", "reasoning", "none"
    ] = Field(
        default="reasoning_content",
        description=(
            "Field carrying a reasoning model's thinking text on "
            "'/v1/chat/completions'. The OpenAI API itself returns no thinking "
            "text at all, so vendors differ: 'reasoning_content' (default) is the "
            "DeepSeek spelling most clients read, 'reasoning' is the one OpenRouter "
            "and vLLM use, and 'none' emits neither and keeps the responses "
            "strictly OpenAI-shaped. Clients can also suppress it per request with "
            "'include_reasoning' or 'reasoning.exclude'."
        ),
    )

    max_input_file_size: int = Field(
        default=0,
        ge=0,
        description=(
            "Maximum size in bytes of an inline input file loaded into memory "
            "(base64, data URI, or a downloaded HTTP(S)/S3 source read for model "
            "input). Requests exceeding this size are rejected with HTTP 413 "
            "before the content is fully decoded or downloaded, protecting the "
            "server against memory-exhaustion. Streaming uploads to S3 (multipart "
            "form uploads, Files API ingest from URLs, and S3-to-S3 copies) are "
            "not affected, so large file transfers remain possible.\n\n"
            "Set to 0 (default) to disable the limit. Example: 26214400 (25 MiB)."
        ),
    )

    max_concurrent_input_downloads: int = Field(
        default=8,
        gt=0,
        description=(
            "Maximum number of input files fetched or resolved concurrently within "
            "a single request. Bounds the number of simultaneous outbound downloads "
            "so a request carrying many remote inputs (images, documents, audio) "
            "cannot exhaust sockets/memory or amplify SSRF against a target. "
            "Excess inputs queue and run as slots free up. Default: 8."
        ),
    )

    default_model_params: dict[str, dict[str, JsonValue]] = Field(
        default={},
        description=(
            "Default inference parameters applied to specific models automatically. "
            "This allows you to configure model-specific behavior globally without "
            "requiring clients to specify parameters in every request.\n\n"
            "Use cases:\n"
            "- Set consistent temperature/creativity levels per model\n"
            "- Enable provider-specific features (e.g., Anthropic beta features)\n"
            "- Configure default token limits for cost control\n"
            "- Set up model-specific stop sequences\n\n"
            "Parameters are merged with request parameters, with request values "
            "taking precedence when both are specified.\n\n"
            "Environment variable format: JSON string\n"
            "Example configurations:\n\n"
            "Basic parameters:\n"
            '{"amazon.nova-micro-v1:0": {"temperature": 0.7, "max_tokens": 1000}}\n\n'
            "Provider-specific features:\n"
            '{"anthropic.claude-sonnet-4-5-20250929-v1:0": {\n'
            '  "anthropic_beta": ["Interleaved-thinking-2025-05-14"]\n'
            "}}\n\n"
            "Multiple models:\n"
            '{"amazon.nova-micro-v1:0": {"temperature": 0.3},\n'
            ' "amazon.nova-lite-v1:0": {"temperature": 0.7}}'
        ),
    )

    default_model_service_tiers: dict[str, ServiceTierTypeType] = Field(
        default={},
        description=(
            "Default service tier for specific Bedrock models. "
            "When a model is invoked without an explicit service tier "
            "(via header or request parameter), the configured default will be used. "
            "This allows you to set model-specific service tier preferences "
            "globally without requiring clients to specify them per request.\n\n"
            "Service tiers:\n"
            "- 'default': Standard compute tier (default)\n"
            "- 'flex': Flexible compute tier for cost optimization\n"
            "- 'priority': Priority compute tier for lower latency\n"
            "- 'reserved': Reserved capacity for dedicated resources (requires AWS contact)\n\n"
            "Important: Not all models support all service tiers. "
            "Check the official AWS documentation for each model.\n\n"
            "Examples:\n"
            "- amazon.nova-pro-v1:0 supports: default, flex, priority (not reserved)\n"
            "- amazon.nova-premier-v1:0 supports: default, flex, priority, reserved\n\n"
            "Environment variable format: JSON string\n"
            "Example: {'amazon.nova-pro-v1:0': 'flex'}"
        ),
    )

    image_generation_model: str | None = Field(
        default=None,
        description=(
            "Default model ID for image generation (e.g. `amazon.nova-canvas-v1:0`). "
            "Required unless the client or the LLM specifies a model per call."
        ),
    )

    default_tts_model: Literal[
        "amazon.polly-standard",
        "amazon.polly-neural",
        "amazon.polly-long-form",
        "amazon.polly-generative",
    ] = Field(
        default="amazon.polly-standard",
        description=(
            "Default text-to-speech model to use if not specified in the request."
        ),
    )

    default_tts_language: str | None = Field(
        default=None,
        description=(
            "Default language code for text-to-speech synthesis. When specified, "
            "this language will be used instead of automatic language detection via AWS Comprehend. "
            "Must be a valid AWS Polly language code (e.g., 'en-US', 'fr-FR', 'es-ES'). "
            "If not specified, the system will automatically detect the language using AWS Comprehend. "
            "Setting a default language can improve performance by avoiding Comprehend API calls."
        ),
    )

    tokens_estimation: bool = Field(
        default=False,
        deprecated=True,
        description="Deprecated and ignored: token estimation has been removed; "
        "only real AWS-billed usage is reported.",
    )

    tokens_estimation_default_encoding: str | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated and ignored: token estimation has been removed.",
    )

    cloudwatch_metrics: bool = Field(
        default=False,
        description="If True, emit per-request AWS-billed usage as CloudWatch "
        "Embedded Metric Format (EMF) log lines (extracted as metrics on ECS).",
    )

    cloudwatch_metrics_namespace: str = Field(
        default="stdapi",
        description=(
            "CloudWatch namespace for the emitted usage metrics. Must be "
            "1-255 characters, using only alphanumeric characters and "
            "'. - _ / # :', and must not start with the reserved 'AWS/' prefix."
        ),
    )

    # Cost tracking settings
    cost_tracking: bool = Field(
        default=False,
        description=(
            "Enable real-time cost tracking from live AWS pricing. Disabled by "
            "default: it requires the extra pricing:GetProducts IAM permission, "
            "which existing deployments may not grant. When enabled, "
            "request logs include per-entry cost/currency and request-level totals. "
            "Pricing data is fetched from AWS Price List API in a background task "
            "at startup (never delaying readiness) and cached in memory, then "
            "refreshed on demand whenever a newly available Bedrock model has no "
            "catalog entry yet. Costs are computed from actual "
            "AWS-billed quantities (tokens, characters, seconds, etc.) multiplied "
            "by the resolved unit price. Silently omits the cost on a pricing miss "
            "rather than blocking request processing."
        ),
    )

    cost_price_overrides: dict[str, dict[str, float]] = Field(
        default={},
        description=(
            "Operator-supplied unit price overrides for Bedrock models not covered "
            "by the AWS Price List API (other services always use the catalog). "
            "The key is the Bedrock model ID (as used by the API), and the inner "
            "dict maps dimension name to price per ONE unit.\n\n"
            "Example (for a model with missing pricing):\n"
            'COST_PRICE_OVERRIDES=\'{"anthropic.claude-3-5-sonnet-20241022": '
            '{"input_tokens": 0.000003, "output_tokens": 0.000015}}\'\n\n'
            "Prices are per one unit (token, character, second, etc.) in the "
            "deployment's partition currency (USD for standard AWS, EUR for EUSC)."
        ),
    )

    enable_docs: bool = Field(
        default=False,
        description=(
            "Enable interactive API documentation UI at /docs. "
            "Disabled by default for security in production environments."
        ),
    )

    enable_redoc: bool = Field(
        default=False,
        description=(
            "Enable ReDoc API documentation UI at /redoc. "
            "Disabled by default for security in production environments."
        ),
    )

    enable_openapi_json: bool = Field(
        default=False,
        description=(
            "Enable OpenAPI JSON schema endpoint at /openapi.json. "
            "Disabled by default for security in production environments. "
            "This endpoint is automatically enabled when enable_docs or enable_redoc is true, "
            "since both documentation UIs require access to the OpenAPI schema."
        ),
    )

    cors_allow_origins: list[str] | None = Field(
        default=None,
        description=(
            "List of origins allowed to make cross-origin requests (CORS). "
            "When set, enables CORS middleware to handle browser cross-origin requests. "
            "Use ['*'] to allow all origins (development), or specify exact origins for production. "
            "If not specified, CORS middleware is not enabled and cross-origin requests from browsers will be blocked. "
            "Example: ['https://myapp.com', 'https://app.example.com']"
        ),
    )

    trusted_hosts: list[str] | None = Field(
        default=None,
        description=(
            "List of trusted host header values for Host header validation. "
            "When set, requests with Host headers not matching any value in this list "
            "will be rejected with HTTP 400. This protects against Host header injection attacks. "
            "Supports wildcard subdomains (e.g., '*.example.com'). "
            "If not specified, no Host header validation is performed. "
            "Example: ['api.example.com', '*.myapp.com', 'localhost']"
        ),
    )

    enable_proxy_headers: bool = Field(
        default=False,
        description=(
            "Enable ProxyHeadersMiddleware to trust X-Forwarded-* headers from reverse proxies. "
            "When enabled, the server will use X-Forwarded-For, X-Forwarded-Proto, and X-Forwarded-Port "
            "headers to determine the client's real IP address and the original request scheme/port. "
            "Only enable this when running behind a trusted reverse proxy (nginx, ALB, CloudFront, etc.). "
            "WARNING: Enabling this without a trusted proxy allows clients to spoof their IP address "
            "and other connection details. Default: false"
        ),
    )

    proxy_trusted_hosts: list[str] | Literal["*"] = Field(
        default="*",
        description=(
            "Trusted proxy hosts/IPs whose X-Forwarded-* headers are honored when "
            "enable_proxy_headers is True. Only requests whose immediate peer IP is "
            "in this list have their forwarded client IP and scheme trusted; any "
            "other peer cannot spoof X-Forwarded-For.\n\n"
            "Defaults to '*' (trust every peer) for backward compatibility. For "
            "defense-in-depth, restrict this to your reverse proxy's IP range so "
            "direct clients cannot forge their source IP.\n\n"
            "Environment variable format: JSON array or '*'.\n"
            'Example: ["10.0.0.0/8"] or ["127.0.0.1"]'
        ),
    )

    enable_gzip: bool = Field(
        default=False,
        description=(
            "Enable GZip compression middleware for HTTP responses. "
            "When enabled, responses larger than 1 KiB will be compressed "
            "using gzip if the client supports it (Accept-Encoding: gzip header). "
            "This reduces bandwidth usage and improves response times for large payloads. "
            "Default: false. Note: AWS services like ALB and CloudFront can handle compression, "
            "so enabling this may be redundant in some deployment scenarios. "
            "Prefer using ALB or CloudFront compression when available."
        ),
    )

    enable_mcp_streamable_http: bool = Field(
        default=False,
        description=(
            "Enable the MCP (Model Context Protocol) server using Streamable HTTP transport. "
            "When enabled, exposes an MCP-compatible endpoint at /mcp that AI clients can connect to. "
            "This is the recommended transport: it implements the latest MCP Streamable HTTP "
            "specification, offering better session management and more robust connection handling. "
            "Default: false."
        ),
    )

    mcp_stateless_http: bool = Field(
        default=False,
        description=(
            "Serve the MCP Streamable HTTP transport in stateless mode. "
            "Each request is then handled by a fresh transport that keeps no session "
            "state, so any client may call /mcp without initializing a session first "
            "and any replica may serve any request. "
            "Required by hosts that provide their own session isolation and inject an "
            "Mcp-Session-Id header the server never issued, such as Amazon Bedrock "
            "AgentCore Runtime. "
            "Requires enable_mcp_streamable_http. Default: false."
        ),
    )

    enable_mcp_sse: bool = Field(
        default=False,
        description=(
            "Enable the MCP (Model Context Protocol) server using Server-Sent Events (SSE) transport. "
            "When enabled, exposes MCP endpoints at /sse for AI clients that require SSE transport. "
            "SSE transport is maintained for backwards compatibility with older MCP client implementations. "
            "Prefer enable_mcp_streamable_http for new deployments. "
            "Default: false."
        ),
    )

    mcp_include_tools: list[str] | None = Field(
        default=None,
        description=(
            "Comma-separated list of MCP tool names to expose exclusively. "
            "Only the listed tools will be available to MCP clients; all others are hidden. "
            "When both mcp_include_tools and mcp_exclude_tools are specified, "
            "tools in mcp_exclude_tools are removed from mcp_include_tools.\n\n"
            "Example: 'openai_chat_completion,openai_embedding,openai_model_list'"
        ),
    )

    mcp_exclude_tools: list[str] | None = Field(
        default=None,
        description=(
            "Comma-separated list of MCP tool names to hide from MCP clients. "
            "All other tools remain exposed. When mcp_include_tools is also specified, "
            "mcp_exclude_tools values are removed from mcp_include_tools.\n\n"
            "Example: 'openai_files_delete,anthropic_files_delete'"
        ),
    )

    ssrf_protection_block_private_networks: bool = Field(
        default=True,
        description=(
            "Enable SSRF protection by blocking requests to private networks. "
            "When enabled, the server will reject requests to private/local networks, "
            "including RFC 1918 private addresses (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16). "
            "This protects against Server-Side Request Forgery (SSRF) attacks by preventing "
            "access to internal network resources. Default: true (recommended for security). "
            "Disable only in controlled environments where accessing local networks is required."
        ),
    )

    ai_response_timeout: int = Field(
        default=600,
        gt=0,
        description=(
            "Maximum time in seconds to wait for an AI model to complete a response. "
            "This applies to both streaming and non-streaming requests, from the moment "
            "the model starts generating until the last token is received.\n\n"
            "The default of 600 seconds (10 minutes) accommodates models with extended "
            "reasoning or thinking capabilities, which may take longer to generate "
            "complex responses. For standard models without extended thinking, responses "
            "typically complete well within 60 seconds.\n\n"
            "Increase this value if you experience timeout errors with long-running "
            "requests (e.g., large document analysis, complex reasoning tasks). "
            "Decrease it to fail fast on unexpectedly slow responses.\n\n"
            "Example: 300 (5 minutes), 600 (10 minutes, default), 900 (15 minutes)"
        ),
    )

    model_cache_seconds: int = Field(
        default=900,
        description=(
            "Cache lifetime in seconds for the Bedrock models list. "
            "When a request needs the model list (e.g., model lookup, /models endpoint) "
            "and the cache has expired, the server queries AWS Bedrock (in parallel across "
            "all configured regions) to discover newly available models, check for model "
            "access changes, and update inference profiles. This is a lazy refresh that "
            "occurs on-demand, not a background task, which can temporarily increase request "
            "latency when the cache is being refreshed. All AWS API requests are executed "
            "concurrently to minimize the latency penalty. Lower values provide faster "
            "detection of new models but increase AWS API calls and may cause more requests "
            "to experience refresh latency. Default: 900 seconds (15 minutes) for a balance "
            "between freshness and performance."
        ),
    )

    model_aliases: dict[str, ModelAliasTarget] = Field(
        default={},
        description=(
            "Map of model aliases to actual model IDs. "
            "Allows users to reference models using custom alias names. "
            "This is merged with default system aliases at startup. "
            "User-provided aliases take precedence over system defaults.\n\n"
            "An alias maps either to a model ID, or to an object carrying that "
            "model plus the configuration to apply to requests naming the "
            "alias: 'service_tier', 'guardrail_id' with 'guardrail_version' "
            "(and optionally 'guardrail_trace'), 'metadata' and 'extra_params'. "
            "Those values override the equivalent server-wide configuration, "
            "and a value sent with the request still wins unless its override "
            "setting (aws_bedrock_allow_guardrail_override, "
            "aws_bedrock_allow_service_tier_override) is disabled.\n\n"
            "Example: {\n"
            '  "my-tts": "amazon.polly-neural",\n'
            '  "my-stt": "amazon.transcribe",\n'
            '  "my-chat": {\n'
            '    "model": "amazon.nova-lite-v1:0",\n'
            '    "service_tier": "flex",\n'
            '    "metadata": {"team": "research"},\n'
            '    "extra_params": {"temperature": 0.2}\n'
            "  }\n"
            "}"
        ),
    )

    drop_unsupported_system_prompt: bool = Field(
        default=True,
        description=(
            "If True, system prompts are silently dropped when models don't support them. "
            "If False, an error is returned when a system prompt is passed to a model "
            "that doesn't support system prompts (e.g., mistral.mistral-7b models). "
            "Default: True for backward compatibility."
        ),
    )

    anthropic_beta_filter: bool = Field(
        default=True,
        description=(
            "Enable filtering of unsupported anthropic_beta flags for Anthropic "
            "Claude models. When enabled, flags not in the allowlist are silently "
            "removed to prevent Bedrock ValidationException errors.\n\n"
            "Set to false to pass all flags through.\n"
            "Default: true"
        ),
    )

    anthropic_beta_allowlist: Annotated[frozenset[str], NoDecode] = Field(
        default=frozenset(),
        description=(
            "Additional anthropic_beta flags to allow beyond the built-in defaults. "
            "This is merged with the built-in set of Bedrock-supported flags, so you "
            "only need to specify extra flags here (e.g., newly added Bedrock flags).\n\n"
            "Only effective when anthropic_beta_filter is true.\n\n"
            "Environment variable format: Comma-separated string\n"
            "Example: 'new-feature-2026-03-01,another-flag-2026-04-01'"
        ),
    )

    extra_model_params_drop_all: bool = Field(
        default=False,
        description=(
            "If true, disable the 'extra model parameters' passthrough entirely: "
            "no undeclared request field (a request's model_extra) is ever "
            "forwarded to Bedrock as a provider-specific inference parameter, on "
            "every route that supports it (the chat completions/responses/messages "
            "inference extras, and embeddings/images/audio/rerank/etc. through "
            "get_extra_model_parameters()). Overrides extra_model_params_denylist: "
            "with this enabled, denylist filtering no longer matters because "
            "nothing is forwarded.\n\n"
            "Set to false (default) to keep the passthrough, filtered by the "
            "built-in default denylist and extra_model_params_denylist."
        ),
    )

    extra_model_params_denylist: Annotated[frozenset[str], NoDecode] = Field(
        default=frozenset(),
        description=(
            "Additional parameter names to strip from the 'extra model "
            "parameters' passthrough, merged with the built-in default denylist "
            "of LiteLLM client-control parameters (e.g. 'drop_params', "
            "'api_key', 'custom_llm_provider') that some OpenAI-SDK-based "
            "clients leak into extra_body and that are never legitimate "
            "Bedrock model parameters.\n\n"
            "Only effective when extra_model_params_drop_all is false.\n\n"
            "Environment variable format: Comma-separated string\n"
            "Example: 'x_internal_debug_flag,x_proxy_trace_id'"
        ),
    )

    @field_validator("mcp_include_tools", "mcp_exclude_tools", mode="before")
    @classmethod
    def _parse_mcp_tools_list(cls, value: list[str] | str | None) -> list[str] | None:
        """Parse MCP tool names from a comma-separated string or list.

        Args:
            value: A comma-separated string of tool names, a list, or None.

        Returns:
            A list of unique tool name strings, or None if the input is empty.
        """
        items = cls._parse_comma_list(value or [])
        return list(set(items)) if items else None

    @field_validator("anthropic_beta_allowlist", mode="before")
    @classmethod
    def _parse_anthropic_beta_allowlist(
        cls, value: frozenset[str] | str
    ) -> frozenset[str]:
        """Parse anthropic_beta_allowlist and merge with built-in defaults.

        Args:
            value: A comma-separated string of extra flags or a frozenset.

        Returns:
            A frozenset of all allowed flags (built-in + user-specified).
        """
        extra = cls._parse_comma_list(value) if isinstance(value, str) else value
        return frozenset(_ANTHROPIC_BETA_BEDROCK_FLAGS | set(extra))

    @field_validator("extra_model_params_denylist", mode="before")
    @classmethod
    def _parse_extra_model_params_denylist(
        cls, value: frozenset[str] | str, info: ValidationInfo
    ) -> frozenset[str]:
        """Parse extra_model_params_denylist and merge with built-in defaults.

        Merging here rather than per request keeps the filter a single membership
        test against an already-built frozenset. Reads ``extra_model_params_drop_all``
        from *info*, which requires that field to stay declared before this one.

        Args:
            value: A comma-separated string of parameter names or a frozenset.
            info: Validation context, holding the fields validated so far.

        Returns:
            Every parameter name to drop, empty when nothing is forwarded at all.
        """
        if info.data.get("extra_model_params_drop_all"):
            return frozenset()
        # LiteLLM client-side parameters, never forwarded as Bedrock model parameters.
        default_dropped = frozenset(
            {
                "acompletion",
                "additional_drop_params",
                "aimg_generation",
                "api_base",
                "api_key",
                "api_version",
                "atext_completion",
                "caching",
                "context_window_fallback_dict",
                "custom_llm_provider",
                "drop_params",
                "fallbacks",
                "force_timeout",
                "litellm_call_id",
                "litellm_credential_name",
                "litellm_logging_obj",
                "litellm_metadata",
                "litellm_request_debug",
                "litellm_session_id",
                "litellm_system_prompt",
                "litellm_trace_id",
                "logger_fn",
                "max_retries",
                "mock_response",
                "mock_timeout",
                "model_info",
                "model_list",
                "num_retries",
                "proxy_server_request",
                "request_timeout",
                "stream_timeout",
                "use_client",
                "verbose",
            }
        )
        extra = cls._parse_comma_list(value) if isinstance(value, str) else value
        return default_dropped | frozenset(extra)

    @field_validator(
        "aws_bedrock_mantle_regions",
        "aws_bedrock_mantle_preferred_models",
        "aws_cognito_client_ids",
        "aws_cognito_required_scopes",
        "oauth_authorization_servers",
        "oauth_scopes_supported",
        mode="before",
    )
    @classmethod
    def _parse_comma_list(cls, value: str | list[str]) -> list[str]:
        """Parse a comma-separated environment string into a list.

        Args:
            value: A comma-separated string or a pre-parsed list.

        Returns:
            List of stripped, non-empty items, first-occurrence order
            preserved, duplicates removed (a duplicated region would cascade
            into every downstream region list, e.g. duplicated model card
            regions and redundant failover attempts).
        """
        if isinstance(value, str):
            value = [item for item in (v.strip() for v in value.split(",")) if item]
        return list(dict.fromkeys(value))

    @field_validator("aws_bedrock_regions", mode="before")
    @classmethod
    def _parse_bedrock_regions(cls, value: str | list[str]) -> list[str]:
        """Parse AWS Bedrock regions from environment variable or list input.

        Falls back to the AWS SDK session region when no region is given.

        Args:
            value: Either a comma-separated string of regions (e.g., "us-east-1, us-west-2")
                  or a pre-parsed list of region strings.

        Returns:
            List of AWS region identifiers with whitespace stripped.

        Raises:
            ValueError: When no region is given and none can be detected.
        """
        value = cls._parse_comma_list(value)
        if not value:
            region = AWS_SESSION.get_config_variable("region")
            if region:
                value.append(region)
            else:
                msg = "No AWS region specified in environment or configuration."
                raise ValueError(msg)
        return value

    @field_validator("timezone", mode="before")
    @classmethod
    def _parse_timezone(cls, value: ZoneInfo | str) -> ZoneInfo:
        """Parse and validate timezone from environment variable or ZoneInfo object.

        Args:
            value: Either a timezone string (e.g., "America/New_York", "UTC")
                  or an existing ZoneInfo object.

        Returns:
            Validated ZoneInfo object for the specified timezone.

        Raises:
            ValueError: When the timezone string is not a valid IANA identifier.
                       Includes a list of available timezones in the error message.
        """
        if isinstance(value, str):
            try:
                return ZoneInfo(value)
            except ZoneInfoNotFoundError, ValueError:
                msg = f'Invalid timezone "{value}", possible values: {", ".join(available_timezones())}.'
                raise ValueError(msg) from None
        return value

    @field_validator("cloudwatch_metrics_namespace")
    @classmethod
    def _validate_cloudwatch_namespace(cls, value: str) -> str:
        """Validate the CloudWatch namespace against AWS naming constraints.

        An invalid namespace makes CloudWatch silently skip EMF metric
        extraction (log lines are still emitted, but no metric appears).

        Args:
            value: CloudWatch metrics namespace.

        Returns:
            The validated namespace.

        Raises:
            ValueError: If the namespace violates CloudWatch naming rules.
        """
        if not _CLOUDWATCH_NAMESPACE_PATTERN.fullmatch(value):
            msg = (
                f'Invalid cloudwatch_metrics_namespace "{value}": must be 1-255 '
                "characters using only alphanumeric characters and . - _ / # :"
            )
            raise ValueError(msg)
        if value.startswith("AWS/"):
            msg = (
                f'Invalid cloudwatch_metrics_namespace "{value}": must not start '
                'with the reserved "AWS/" prefix.'
            )
            raise ValueError(msg)
        return value

    @field_validator("aws_s3_videos_prefix")
    @classmethod
    def _validate_videos_prefix(cls, value: str) -> str:
        """Validate the S3 prefix used for generated videos.

        The Bedrock output ``s3Uri`` uses this value verbatim, while the
        ownership check gating ``get_video_job``/listing appends a missing "/"
        so it cannot match a sibling prefix: without the trailing "/" a prefix
        would be written one way and checked another, hence it is required. An
        empty value would widen that ownership check to the whole bucket, so it
        is rejected too.

        Args:
            value: S3 prefix for generated videos.

        Returns:
            The validated prefix.

        Raises:
            ValueError: If the value is empty, starts with "/", contains "//",
                uses characters outside S3's safe set, or has no trailing "/".
        """
        if not _S3_PREFIX_PATTERN.fullmatch(value):
            msg = (
                f'Invalid aws_s3_videos_prefix "{value}": must be non-empty, must '
                'not start with "/" or contain "//", must use only S3-safe '
                "characters (alphanumerics plus ! _ . * ' ( ) -) per path segment, "
                'and must end with a trailing "/"'
            )
            raise ValueError(msg)
        return value

    @field_validator("aws_s3_batches_prefix")
    @classmethod
    def _validate_batches_prefix(cls, value: str) -> str:
        """Validate the S3 prefix used for Batch API data.

        Batch data is addressed by prefix, so an empty value would put batch
        records at the bucket root and make every stray object a candidate.

        Args:
            value: S3 prefix for Batch API data.

        Returns:
            The validated prefix.

        Raises:
            ValueError: If the value is empty, starts with "/", contains "//",
                uses characters outside S3's safe set, or has no trailing "/".
        """
        if not _S3_PREFIX_PATTERN.fullmatch(value):
            msg = (
                f'Invalid aws_s3_batches_prefix "{value}": must be non-empty, must '
                'not start with "/" or contain "//", must use only S3-safe '
                "characters (alphanumerics plus ! _ . * ' ( ) -) per path segment, "
                'and must end with a trailing "/"'
            )
            raise ValueError(msg)
        return value

    @field_validator("aws_bedrock_batch_role_arn")
    @classmethod
    def _validate_batch_role_arn(cls, value: str | None) -> str | None:
        """Validate the batch service role ARN against the IAM role format.

        A malformed ARN would otherwise only surface once the first batch
        submission is rejected by AWS.

        Args:
            value: IAM role ARN, or None to keep the Batch API disabled.

        Returns:
            The validated ARN.

        Raises:
            ValueError: If the value is set and is not an IAM role ARN.
        """
        if value is not None and not _IAM_ROLE_ARN_PATTERN.fullmatch(value):
            msg = (
                f'Invalid aws_bedrock_batch_role_arn "{value}": must be an IAM role '
                'ARN "arn:<partition>:iam::<account-id>:role/<name>".'
            )
            raise ValueError(msg)
        return value

    @staticmethod
    def _check_routes_prefix(field_name: str, value: str, *, allow_empty: bool) -> str:
        """Validate a routes-prefix value against the shared prefix format.

        Args:
            field_name: Name of the setting being validated, used in the error message.
            value: The routes-prefix value to validate.
            allow_empty: Whether an empty string is a legal value for this setting.

        Returns:
            The validated prefix.

        Raises:
            ValueError: If the value is not empty (when allowed) and does not match
                the required "/segment" format.
        """
        if allow_empty and value == "":
            return value
        if not _ROUTES_PREFIX_PATTERN.fullmatch(value):
            msg = (
                f'Invalid {field_name} "{value}": must start with "/", must not end '
                'with "/", and use only alphanumeric characters and . _ ~ - per '
                "path segment"
            )
            raise ValueError(msg)
        return value

    @field_validator("openai_routes_prefix")
    @classmethod
    def _validate_openai_routes_prefix(cls, value: str) -> str:
        """Validate the OpenAI routes prefix, allowing the empty (root-mounted) default.

        Args:
            value: OpenAI routes prefix.

        Returns:
            The validated prefix.

        Raises:
            ValueError: If the value is non-empty and does not match the required format.
        """
        return cls._check_routes_prefix("openai_routes_prefix", value, allow_empty=True)

    @field_validator("anthropic_routes_prefix", "cohere_routes_prefix")
    @classmethod
    def _validate_required_routes_prefix(cls, value: str, info: ValidationInfo) -> str:
        """Validate the Anthropic and Cohere routes prefixes.

        Args:
            value: Routes prefix value.
            info: Validation info, used to identify the field being validated.

        Returns:
            The validated prefix.

        Raises:
            ValueError: If the value does not match the required format.
        """
        return cls._check_routes_prefix(str(info.field_name), value, allow_empty=False)

    @field_validator("aws_bedrock_model_arn_mapping")
    @classmethod
    def _validate_arn_mapping(cls, value: dict[str, str]) -> dict[str, str]:
        """Validate that all ARN mappings have valid ARN formats.

        Args:
            value: Dictionary mapping model IDs to ARNs.

        Returns:
            The validated dictionary.

        Raises:
            ValueError: If any ARN has an invalid format.
        """
        invalid_arns = []
        for model_id, arn in value.items():
            if not (
                match_bedrock_app_profile_arn(arn)
                or match_bedrock_prompt_router_arn(arn)
            ):
                invalid_arns.append(f"  - {model_id}: {arn}")
        if invalid_arns:
            msg = (
                "Invalid ARN format(s) in aws_bedrock_model_arn_mapping. "
                "ARNs must be inference profiles or prompt routers:\n"
                + "\n".join(invalid_arns)
            )
            raise ValueError(msg)
        return value

    @field_validator("aws_bedrock_session_encryption_key_arn")
    @classmethod
    def _validate_session_encryption_key_arn(cls, value: str | None) -> str | None:
        """Validate the KMS key ARN against Bedrock CreateSession's ``encryptionKeyArn`` format.

        A typo'd ARN would otherwise only surface as a per-request AWS error
        once a stored response or chat completion is created.

        Args:
            value: KMS key ARN, or None to use the AWS-managed key.

        Returns:
            The validated ARN.

        Raises:
            ValueError: If the value is set and is not a valid KMS key ARN.
        """
        if value is not None and not _KMS_KEY_ARN_PATTERN.fullmatch(value):
            msg = (
                f'Invalid aws_bedrock_session_encryption_key_arn "{value}": must be '
                'a KMS key ARN "arn:<partition>:kms:<region>:<account-id>:key/<key-id>".'
            )
            raise ValueError(msg)
        return value

    @field_validator("aws_bedrock_user_role_arn")
    @classmethod
    def _validate_user_role_arn(cls, value: str | None) -> str | None:
        """Validate the per-end-user role ARN against the STS AssumeRole format.

        A malformed ARN would otherwise only surface once the first model
        invocation fails to obtain the end user's credentials.

        Args:
            value: IAM role ARN, or None to keep the server's own identity.

        Returns:
            The validated ARN.

        Raises:
            ValueError: If the value is set and is not an IAM role ARN.
        """
        if value is not None and not _IAM_ROLE_ARN_PATTERN.fullmatch(value):
            msg = (
                f'Invalid aws_bedrock_user_role_arn "{value}": must be an IAM role '
                'ARN "arn:<partition>:iam::<account-id>:role/<name>".'
            )
            raise ValueError(msg)
        return value

    @field_validator("aws_bedrock_user_role_tag_key")
    @classmethod
    def _validate_user_role_tag_key(cls, value: str | None) -> str | None:
        """Validate the end user session tag key against the STS AssumeRole charset.

        Args:
            value: Session tag key, or None to send no session tag.

        Returns:
            The validated key.

        Raises:
            ValueError: If the key is empty, too long, uses characters AWS STS
                rejects, or starts with the reserved "aws:" prefix.
        """
        if value is None:
            return value
        if not _SESSION_TAG_KEY_PATTERN.fullmatch(value):
            msg = (
                f'Invalid aws_bedrock_user_role_tag_key "{value}": must be 1 to 128 '
                "characters, letters, digits, spaces or _ . : / = + - @"
            )
            raise ValueError(msg)
        if value.lower().startswith("aws:"):
            msg = (
                f'Invalid aws_bedrock_user_role_tag_key "{value}": keys beginning '
                'with "aws:" are reserved by AWS.'
            )
            raise ValueError(msg)
        return value

    @field_validator("aws_bedrock_mantle_endpoint_url")
    @classmethod
    def _validate_mantle_endpoint_url(cls, value: str | None) -> str | None:
        """Validate the Bedrock Mantle endpoint URL template.

        A bad "{region}" placeholder would otherwise only surface as a
        per-request formatting error, and a non-HTTPS scheme would silently
        disable transport encryption toward Mantle.

        Args:
            value: Endpoint URL template, or None to use the default.

        Returns:
            The validated template.

        Raises:
            ValueError: If the value does not use "https://" or its
                "{region}" placeholder is malformed.
        """
        if value is None:
            return value
        if not value.startswith("https://"):
            msg = (
                f'Invalid aws_bedrock_mantle_endpoint_url "{value}": must use the '
                '"https://" scheme.'
            )
            raise ValueError(msg)
        try:
            value.format(region="us-east-1")
        except (KeyError, IndexError, ValueError) as error:
            msg = (
                f'Invalid aws_bedrock_mantle_endpoint_url "{value}": malformed '
                '"{region}" placeholder.'
            )
            raise ValueError(msg) from error
        return value

    @field_validator("oauth_resource_identifier")
    @classmethod
    def _validate_oauth_resource_identifier(cls, value: str | None) -> str | None:
        """Validate the public URL published as the OAuth resource identifier.

        The value is compared character by character by clients and is embedded
        in a quoted header parameter, so a path, a query, or any character that
        could close the quoted string is rejected rather than escaped.

        Args:
            value: Public origin URL of this deployment, or None.

        Returns:
            The URL without its trailing slash, or None.

        Raises:
            ValueError: If the value is not an absolute "http"/"https" origin.
        """
        if value is None:
            return None
        origin = value.rstrip("/")
        if not _ORIGIN_URL_PATTERN.fullmatch(origin):
            msg = (
                f'Invalid oauth_resource_identifier "{value}": must be the '
                'absolute URL clients dial, with no path, e.g. "https://api.example.com".'
            )
            raise ValueError(msg)
        return origin

    @field_validator("oauth_authorization_servers")
    @classmethod
    def _validate_oauth_authorization_servers(cls, value: list[str]) -> list[str]:
        """Validate the published authorization server issuer URLs.

        Args:
            value: Issuer URLs of the authorization servers.

        Returns:
            The issuer URLs without their trailing slash.

        Raises:
            ValueError: If an entry is not an "https" URL free of query and fragment.
        """
        issuers = [item.rstrip("/") for item in value]
        for issuer in issuers:
            if not _ISSUER_URL_PATTERN.fullmatch(issuer):
                msg = (
                    f'Invalid oauth_authorization_servers entry "{issuer}": must '
                    'be an "https" issuer URL with no query or fragment, e.g. '
                    '"https://cognito-idp.eu-west-3.amazonaws.com/eu-west-3_a1b2c3d4e".'
                )
                raise ValueError(msg)
        return issuers

    @field_validator("oauth_scopes_supported")
    @classmethod
    def _validate_oauth_scopes_supported(cls, value: list[str]) -> list[str]:
        """Validate the published OAuth 2.0 scopes.

        Args:
            value: Scopes a token needs to call this API.

        Returns:
            The scopes, unchanged.

        Raises:
            ValueError: If a scope is not an RFC 6749 scope token.
        """
        for scope in value:
            if not _OAUTH_SCOPE_PATTERN.fullmatch(scope):
                msg = (
                    f'Invalid oauth_scopes_supported entry "{scope}": an OAuth '
                    "2.0 scope carries no space, double quote or backslash."
                )
                raise ValueError(msg)
        return value

    def _validate_oauth(self) -> None:
        """Ensure the OAuth 2.0 discovery configuration is complete.

        A resource identifier with no authorization server would publish a
        document telling clients nothing about where to obtain a token, and the
        remaining settings describe a document that is not published at all.

        Raises:
            ValueError: If the discovery configuration is incomplete.
        """
        if self.oauth_resource_identifier:
            if not self.oauth_authorization_servers:
                msg = (
                    "oauth_authorization_servers is required with "
                    "oauth_resource_identifier: metadata naming no authorization "
                    "server leaves a client unable to obtain a token, and is read "
                    "by some clients as this server being its own authorization "
                    "server."
                )
                raise ValueError(msg)
        elif self.oauth_authorization_servers or self.oauth_scopes_supported:
            msg = (
                "oauth_resource_identifier is required to publish OAuth 2.0 "
                "protected resource metadata."
            )
            raise ValueError(msg)

    def _validate_cognito(self) -> None:
        """Ensure the Amazon Cognito user pool configuration is complete.

        A half-configured pool would leave the deployment authenticated by
        whatever else happens to be set, so an incomplete combination fails
        startup instead.

        Raises:
            ValueError: If the pool configuration is incomplete or malformed.
        """
        if self.aws_cognito_user_pool_id:
            if not _COGNITO_USER_POOL_ID_PATTERN.fullmatch(
                self.aws_cognito_user_pool_id
            ):
                msg = (
                    f'Invalid aws_cognito_user_pool_id "{self.aws_cognito_user_pool_id}"'
                    ': must be an Amazon Cognito user pool ID, "<region>_<identifier>".'
                )
                raise ValueError(msg)
            if not self.aws_cognito_client_ids:
                msg = (
                    "aws_cognito_client_ids is required with "
                    "aws_cognito_user_pool_id: without an application allowlist, "
                    "a token issued to any application of the pool would be accepted."
                )
                raise ValueError(msg)
        elif (
            self.aws_cognito_client_ids
            or self.aws_cognito_required_scopes
            or self.aws_cognito_accept_id_token
            or self.aws_cognito_issuer_type != "original"
        ):
            msg = (
                "aws_cognito_user_pool_id is required to configure Amazon Cognito "
                "authentication."
            )
            raise ValueError(msg)

    def _validate_authentication_mode(self) -> None:
        """Ensure the accepted authentication methods are the configured ones.

        The mode states the intended security posture, so a method it demands
        must exist and a method it would ignore must not be configured.

        Raises:
            ValueError: If a configured method contradicts the mode.
        """
        api_key_configured = bool(
            self.api_key
            or self.api_key_ssm_parameter
            or self.api_key_secretsmanager_secret
        )
        if self.authentication_mode == "api_key":
            if not api_key_configured:
                msg = (
                    'authentication_mode "api_key" requires an API key source '
                    "(api_key, api_key_ssm_parameter or api_key_secretsmanager_secret)."
                )
                raise ValueError(msg)
            if self.aws_cognito_user_pool_id:
                msg = (
                    'authentication_mode "api_key" ignores aws_cognito_user_pool_id, '
                    'which is configured. Use authentication_mode "any" to accept both.'
                )
                raise ValueError(msg)
        elif self.authentication_mode == "cognito":
            if not self.aws_cognito_user_pool_id:
                msg = 'authentication_mode "cognito" requires aws_cognito_user_pool_id.'
                raise ValueError(msg)
            if api_key_configured:
                msg = (
                    'authentication_mode "cognito" ignores the configured API key '
                    'source. Use authentication_mode "any" to accept both.'
                )
                raise ValueError(msg)

    def _validate_unique_routes_prefixes(self) -> None:
        """Ensure non-empty API routes prefixes do not collide across providers.

        Raises:
            ValueError: If two providers share the same non-empty routes prefix.
        """
        seen_prefixes: dict[str, str] = {}
        for field_name, prefix in (
            ("openai_routes_prefix", self.openai_routes_prefix),
            ("anthropic_routes_prefix", self.anthropic_routes_prefix),
            ("cohere_routes_prefix", self.cohere_routes_prefix),
        ):
            if not prefix:
                continue
            if prefix in seen_prefixes:
                msg = (
                    f"{seen_prefixes[prefix]} and {field_name} must not both be set "
                    f'to "{prefix}": routes prefixes must be unique.'
                )
                raise ValueError(msg)
            seen_prefixes[prefix] = field_name

    @model_validator(mode="after")
    def _validate(self) -> Self:
        """Perform cross-field validation and apply configuration defaults.

        Validation rules:
        1. Bedrock Guardrails require both identifier and version
        2. The Mantle service header requires Mantle and excludes Guardrails
        3. API key sources are mutually exclusive
        4. Transcribe S3 bucket defaults to main S3 bucket if not specified
        5. When both MCP include/exclude are specified, include is filtered by exclude
        6. Non-empty API routes prefixes must be unique across providers
        7. The Amazon Cognito configuration is complete
        8. The accepted authentication methods are the configured ones
        9. The OAuth 2.0 discovery configuration is complete

        Returns:
            Self with validated and defaulted configuration.

        Raises:
            ValueError: When configuration combinations are invalid or conflicting.
        """
        self.enable_openapi_json = (
            self.enable_openapi_json or self.enable_docs or self.enable_redoc
        )
        self._validate_unique_routes_prefixes()
        if (
            self.aws_bedrock_guardrail_identifier
            and not self.aws_bedrock_guardrail_version
        ) or (
            self.aws_bedrock_guardrail_version
            and not self.aws_bedrock_guardrail_identifier
        ):
            msg = (
                "Both aws_bedrock_guardrail_identifier & aws_bedrock_guardrail_version "
                "are required to configure Amazon Bedrock Guardrails."
            )
            raise ValueError(msg)

        # An alias-borne guardrail is operator configuration too: it must not
        # open the request-level override gate, and must not be silently
        # dropped by a Mantle-served deployment.
        alias_guardrail = any(
            isinstance(alias, ModelAliasConfig) and alias.guardrail_identifier
            for alias in self.model_aliases.values()
        )
        if (
            not self.aws_bedrock_guardrail_identifier
            and not self.aws_bedrock_guardrail_version
            and not alias_guardrail
        ):
            self.aws_bedrock_allow_guardrail_override = True

        if self.aws_bedrock_mantle_service_header and (
            self.aws_bedrock_guardrail_identifier
            or alias_guardrail
            or not self.aws_bedrock_mantle_enabled
        ):
            msg = (
                "aws_bedrock_mantle_service_header requires aws_bedrock_mantle_enabled "
                "and is incompatible with Amazon Bedrock Guardrails "
                "(guardrails do not apply to Mantle-served requests)."
            )
            raise ValueError(msg)
        if not self.aws_bedrock_mantle_regions:
            self.aws_bedrock_mantle_regions = self.aws_bedrock_regions

        if (
            self.aws_bedrock_user_role_require_identity
            and not self.aws_bedrock_user_role_arn
        ):
            msg = (
                "aws_bedrock_user_role_require_identity requires "
                "aws_bedrock_user_role_arn: without a per-end-user role, rejecting "
                "requests that identify no end user would attribute nothing."
            )
            raise ValueError(msg)

        if not self.aws_transcribe_s3_bucket:
            self.aws_transcribe_s3_bucket = self.aws_s3_bucket
        if (
            self.api_key
            and self.api_key_secretsmanager_secret
            and self.api_key_secretsmanager_key
        ):
            msg = (
                "Only one of api_key, api_key_secretsmanager_secret "
                "and api_key_secretsmanager_key must be specified."
            )
            raise ValueError(msg)
        if self.mcp_include_tools and self.mcp_exclude_tools:
            self.mcp_include_tools = list(
                set(self.mcp_include_tools) - set(self.mcp_exclude_tools)
            )
            self.mcp_exclude_tools = None
        self._validate_cognito()
        self._validate_authentication_mode()
        self._validate_oauth()
        return self

    def now(self) -> AwareDatetime:
        """Return the current date and time in the configured timezone.

        Returns:
            Current timezone-aware datetime.
        """
        return datetime.now(self.timezone)

    def deprecated(self) -> set[str]:
        """Return the deprecated settings that were explicitly set.

        Returns:
            Names of settings marked with ``Field(deprecated=...)`` found in
            ``model_fields_set``.
        """
        fields = type(self).model_fields
        return {name for name in self.model_fields_set if fields[name].deprecated}


try:
    SETTINGS = _Settings()
except ValidationError as error:
    import sys

    stdout_write(
        {
            "type": "start",
            "level": "error",
            "date": datetime.now(ZoneInfo("UTC")).isoformat(),
            "server_id": SERVER_NAME,
            "server_version": SERVER_VERSION,
            "error_detail": [
                {
                    "message": "Configuration validation failed. Verify your environment variables and try again.",
                    "details": error.errors(  # type: ignore[dict-item]
                        include_url=False, include_context=False, include_input=False
                    ),
                }
            ],
        }
    )
    sys.exit(1)

#: Current detected region
AWS_REGION: str = (
    AWS_SESSION.get_config_variable("region") or SETTINGS.aws_bedrock_regions[0]
)
