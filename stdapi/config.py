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

from datetime import datetime
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from aioboto3 import Session
from aiohttp import ClientTimeout
from pydantic import (
    AwareDatetime,
    Field,
    JsonValue,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from stdapi.server import SERVER_NAME, SERVER_VERSION
from stdapi.utils import stdout_write

#: HTTP download timeout
DOWNLOAD_TIMEOUT = ClientTimeout(total=20, connect=5)

#: Logging levels
LogLevel = Literal["info", "warning", "error", "critical"]


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

    aws_s3_regional_buckets: dict[str, str] = Field(
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

    aws_polly_region: str | None = Field(
        default=None, description="AWS region for Polly text-to-speech service"
    )

    aws_comprehend_region: str | None = Field(
        default=None, description="AWS region for Comprehend language detection service"
    )

    aws_bedrock_regions: Annotated[list[str], NoDecode] = Field(
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
        default=True, description="If true, allow legacy Bedrock models to be used."
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

    aws_transcribe_region: str | None = Field(
        default=None, description="AWS region for Transcribe speech-to-text service"
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

    aws_translate_region: str | None = Field(
        default=None, description="AWS region for Translate text translation service"
    )

    timezone: ZoneInfo = Field(
        default=ZoneInfo("UTC"), description="Timezone for request date & time"
    )

    openai_routes_prefix: str = Field(
        default="", description="OpenAI API compatible routes prefix"
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

    tokens_estimation: bool = Field(
        default=False,
        description=(
            "If True, estimate the number of tokens using a tokenizer based on the "
            "request input/output text when not directly returned by the model itself."
        ),
    )

    tokens_estimation_default_encoding: str = Field(
        default="o200k_base",
        description="Tiktoken Tokenizer encoding to use for token count estimation.",
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

    @field_validator("aws_bedrock_regions", mode="before")
    @classmethod
    def _parse_bedrock_regions(cls, value: str | list[str]) -> list[str]:
        """Parse AWS Bedrock regions from environment variable or list input.

        Converts comma-separated region strings into a list of region identifiers,
        stripping whitespace and filtering out empty values for robust parsing.

        Args:
            value: Either a comma-separated string of regions (e.g., "us-east-1, us-west-2")
                  or a pre-parsed list of region strings.

        Returns:
            List of AWS region identifiers with whitespace stripped.

        Example:
            "us-east-1, us-west-2 , eu-west-1" -> ["us-east-1", "us-west-2", "eu-west-1"]
        """
        if isinstance(value, str):
            value = [
                region for region in (v.strip() for v in value.split(",")) if region
            ]
        if not value:
            # Let Boto3 try to detect the region if not specified
            session = Session()
            if session.region_name:
                value.append(session.region_name)
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
            except (ZoneInfoNotFoundError, ValueError):
                msg = f'Invalid timezone "{value}", possible values: {", ".join(available_timezones())}.'
                raise ValueError(msg) from None
        return value

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

        Returns:
            Self with validated and defaulted configuration.

        Raises:
            ValueError: When configuration combinations are invalid or conflicting.

        Examples of invalid configurations:
        - Guardrail ID without version (or vice versa)
        - Multiple API key sources specified simultaneously
        """
        self.enable_openapi_json = (
            self.enable_openapi_json or self.enable_docs or self.enable_redoc
        )
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
        return self

    def now(self) -> AwareDatetime:
        """Returns the current date and time based on the specified timezone.

        Returns:
            datetime: The current date and time adjusted to the configured timezone.
        """
        return datetime.now(self.timezone)


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
