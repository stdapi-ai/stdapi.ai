"""Configuration management using Pydantic models.

This module centralizes all environment variable configuration for the server.
It provides comprehensive validation, type safety, and clear documentation of all configuration parameters.

The configuration system supports:
- Environment variable loading with type conversion
- AWS service configuration across multiple regions
- OpenAI API compatibility settings
- Authentication and security options
- OpenTelemetry tracing configuration
- Model parameter defaults and customization

Key Components:
- _DefaultModelParameters: Defines reusable model inference parameters
- _Settings: Main configuration class loaded from environment variables
- SETTINGS: Global configuration instance used throughout the application

Environment Variable Examples:
    AWS_S3_BUCKET=my-stdapi-bucket
    AWS_BEDROCK_REGIONS=us-east-1,us-west-2
    API_KEY=your-secret-api-key
    TIMEZONE=America/New_York
    OTEL_ENABLED=true

For detailed configuration options, see the _Settings class documentation.
"""

import re
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from aiobotocore.session import get_session
from aiohttp import ClientTimeout
from pydantic import (
    AwareDatetime,
    Field,
    JsonValue,
    SecretStr,
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


class _Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    This class manages all application configuration through environment variables,
    providing type safety, validation, and comprehensive AWS service integration.

    Configuration Categories:
    1. **AWS Storage**: S3 buckets for file storage and temporary data
    2. **AWS AI Services**: Bedrock, Polly, Comprehend, Transcribe, Translate
    3. **Authentication**: API keys from environment, SSM, or Secrets Manager
    4. **OpenAI Compatibility**: Route prefixes and API emulation
    5. **Observability**: OpenTelemetry tracing and logging controls
    6. **Model Defaults**: Per-model inference parameter overrides

    AWS Service Regions:
    - Most AWS services use optional region settings, falling back to the
      default boto3 session region if not specified
    - S3 buckets must be in the same region as their associated services
    - Bedrock supports multi-region configuration for model availability

    Environment Variable Examples:
        # Required AWS configuration
        AWS_S3_BUCKET=my-stdapi-files
        AWS_BEDROCK_REGIONS=us-east-1,us-west-2,eu-west-1

        # Optional service regions (fallback to default AWS region)
        AWS_POLLY_REGION=us-east-1
        AWS_TRANSCRIBE_REGION=us-east-1
        AWS_TRANSCRIBE_S3_BUCKET=my-transcribe-temp

        # Authentication options (choose one)
        API_KEY=your-secret-key
        API_KEY_SSM_PARAMETER=/stdapi/api-key
        API_KEY_SECRETSMANAGER_SECRET=stdapi-secrets

        # OpenAI compatibility
        OPENAI_ROUTES_PREFIX=/v1

        # Observability
        OTEL_ENABLED=true
        OTEL_SERVICE_NAME=stdapi-prod
        OTEL_SAMPLE_RATE=0.1

        # Model configuration
        DEFAULT_MODEL_PARAMS={"anthropic.claude-3-sonnet": {"temperature": 0.7}}

        # Application behavior
        TIMEZONE=America/New_York
        STRICT_INPUT_VALIDATION=false
        LOG_REQUEST_PARAMS=false

    Validation Rules:
    - Bedrock Guardrails require both identifier and version
    - API key sources are mutually exclusive
    - S3 buckets default to shared usage when not specified
    - Timezone must be a valid IANA timezone identifier

    See individual field documentation for detailed parameter descriptions.
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
            "Timeout in seconds for establishing a connection to an AWS service endpoint. "
            "Keeping this value short allows fast failover to another region when a connection "
            "cannot be established. Increase it only if you experience spurious connection timeouts "
            "on high-latency networks. "
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
            "When region routing is enabled, retries cycle through all available regions in order. "
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

    aws_bedrock_guardrail_trace: (
        Literal["disabled", "enabled", "enabled_full"] | None
    ) = Field(
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

    model_aliases: dict[str, str] = Field(
        default={},
        description=(
            "Map of model aliases to actual model IDs. "
            "Allows users to reference models using custom alias names. "
            "This is merged with default system aliases at startup. "
            "User-provided aliases take precedence over system defaults.\n\n"
            "Example: {\n"
            '  "my-tts": "amazon.polly-neural",\n'
            '  "my-stt": "amazon.transcribe"\n'
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

        Converts a comma-separated string to a frozenset and merges with
        `ANTHROPIC_BETA_BEDROCK_FLAGS`.

        Args:
            value: A comma-separated string of extra flags or a frozenset.

        Returns:
            A frozenset of all allowed flags (built-in + user-specified).
        """
        extra = cls._parse_comma_list(value) if isinstance(value, str) else value
        return frozenset(_ANTHROPIC_BETA_BEDROCK_FLAGS | set(extra))

    @field_validator(
        "aws_bedrock_mantle_regions",
        "aws_bedrock_mantle_preferred_models",
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
            # Let AWS SDK try to detect the region if not specified
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

        Converts IANA timezone identifier strings to ZoneInfo objects with
        comprehensive error handling and helpful error messages.

        Args:
            value: Either a timezone string (e.g., "America/New_York", "UTC")
                  or an existing ZoneInfo object.

        Returns:
            Validated ZoneInfo object for the specified timezone.

        Raises:
            ValueError: When the timezone string is not a valid IANA identifier.
                       Includes a list of available timezones in the error message.

        Examples:
            "UTC" -> ZoneInfo("UTC")
            "America/New_York" -> ZoneInfo("America/New_York")
            "Invalid/Zone" -> ValueError with available options
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

        The Bedrock output ``s3Uri`` is built from this value verbatim
        (``stdapi.models.video.AsyncVideoModel.start_video_generation``),
        while the ownership check that gates ``get_video_job``/listing
        (``_region_videos_uri_prefix``) appends a "/" when one is missing so
        it cannot match an unrelated sibling prefix. A prefix without a
        trailing "/" would therefore be written one way and checked another,
        so the trailing "/" is required rather than merely recommended. An
        empty value would widen the ownership check to the whole bucket,
        which is security-relevant, so empty prefixes are rejected too.

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

        Validates configuration combinations that span multiple fields and
        applies intelligent defaults where appropriate. This ensures the
        configuration is internally consistent and usable.

        Validation rules:
        1. Bedrock Guardrails require both identifier and version
        2. API key sources are mutually exclusive
        3. Transcribe S3 bucket defaults to main S3 bucket if not specified
        4. API key configuration options cannot conflict
        5. When both MCP include/exclude are specified, include is filtered by exclude
        6. Non-empty API routes prefixes must be unique across providers

        Returns:
            Self with validated and defaulted configuration.

        Raises:
            ValueError: When configuration combinations are invalid or conflicting.

        Examples of invalid configurations:
        - Guardrail ID without version (or vice versa)
        - Multiple API key sources specified simultaneously
        - Two routes prefixes set to the same non-empty value
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

        if (
            not self.aws_bedrock_guardrail_identifier
            and not self.aws_bedrock_guardrail_version
        ):
            self.aws_bedrock_allow_guardrail_override = True

        if self.aws_bedrock_mantle_service_header and (
            self.aws_bedrock_guardrail_identifier or not self.aws_bedrock_mantle_enabled
        ):
            msg = (
                "aws_bedrock_mantle_service_header requires aws_bedrock_mantle_enabled "
                "and is incompatible with Amazon Bedrock Guardrails "
                "(guardrails do not apply to Mantle-served requests)."
            )
            raise ValueError(msg)
        if not self.aws_bedrock_mantle_regions:
            self.aws_bedrock_mantle_regions = self.aws_bedrock_regions

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
