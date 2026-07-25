"""Common AWS Bedrock utilities."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

from botocore.exceptions import ClientError
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.config import SETTINGS
from stdapi.types import JsonMapping
from stdapi.utils import validation_error_handler

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

    from starlette.datastructures import Headers
    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime import BedrockRuntimeClient
    from types_aiobotocore_bedrock_runtime.literals import (
        AudioFormatType,
        DocumentFormatType,
        GuardrailTraceType,
        PerformanceConfigLatencyType,
        ServiceTierTypeType,
        VideoFormatType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockTypeDef,
        GuardrailStreamConfigurationTypeDef,
        InferenceConfigurationTypeDef,
        MessageUnionTypeDef,
        OutputConfigTypeDef,
        PerformanceConfigurationTypeDef,
        PromptVariableValuesTypeDef,
        ResponseStreamTypeDef,
        ServiceTierTypeDef,
        SystemContentBlockTypeDef,
        ToolConfigurationTypeDef,
    )

    from stdapi.types import BaseModelRequestWithExtra

    class ConverseRequestBaseTypeDef(TypedDict):
        """Converse request base type definition.

        Common fields from "ConverseRequestTypeDef" and
        "ConverseStreamRequestTypeDef".
        """

        modelId: str
        messages: NotRequired[Sequence[MessageUnionTypeDef]]
        system: NotRequired[Sequence[SystemContentBlockTypeDef]]
        inferenceConfig: NotRequired[InferenceConfigurationTypeDef]
        toolConfig: NotRequired[ToolConfigurationTypeDef]
        additionalModelRequestFields: NotRequired[Mapping[str, Any]]
        promptVariables: NotRequired[Mapping[str, PromptVariableValuesTypeDef]]
        additionalModelResponseFieldPaths: NotRequired[Sequence[str]]
        requestMetadata: NotRequired[Mapping[str, str]]
        performanceConfig: NotRequired[PerformanceConfigurationTypeDef]
        guardrailConfig: NotRequired[GuardrailStreamConfigurationTypeDef]
        serviceTier: NotRequired[ServiceTierTypeDef]
        outputConfig: NotRequired[OutputConfigTypeDef]

    class AmazonBedrockInvocationMetrics(TypedDict):
        """amazon-bedrock-invocationMetrics field in responses."""

        inputTokenCount: int
        outputTokenCount: int
        invocationLatency: int
        firstByteLatency: int

    BedrockInvocationTypeDef = TypedDict(
        "BedrockInvocationTypeDef",
        {
            "amazon-bedrock-invocationMetrics": NotRequired[
                AmazonBedrockInvocationMetrics
            ]
        },
    )

#: Bedrock documents types with the matching MIME type
MIME_TYPES_TO_DOCUMENT_TYPE: dict[str, DocumentFormatType] = {
    "csv": "csv",
    "html": "html",
    "pdf": "pdf",
    "msword": "doc",
    "vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "vnd.ms-excel": "xls",
    "vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "plain": "txt",
    "markdown": "md",
}

#: Bedrock videos types with the matching MIME type
MIME_TYPES_TO_VIDEO_TYPE: dict[str, VideoFormatType] = {
    "x-matroska": "mkv",
    "quicktime": "mov",
    "x-flv": "flv",
    "x-ms-wmv": "wmv",
    "3gpp": "three_gp",
}

#: Bedrock audio types with the matching MIME type
MIME_TYPES_TO_AUDIO_TYPE: dict[str, AudioFormatType] = {
    "x-aac": "aac",
    "x-flac": "flac",
    "x-m4a": "mp4",
    "mpeg": "mp3",
    "x-wav": "wav",
}

#: Bedrock limit for sync body size (25MB), here with a little margin
BEDROCK_BODY_SIZE_LIMIT = 24_990_000

#: Bedrock error codes on model error
_BEDROCK_MODEL_ERROR_CODES = {
    "ModelErrorException",
    "ModelStreamErrorException",
    "ModelTimeoutException",
}

#: Guardrail configuration for the request.
GUARDRAIL_CONFIG_VAR: ContextVar[GuardrailStreamConfigurationTypeDef] = ContextVar(
    "guardrail_configuration"
)
_GUARDRAIL_IDENTIFIER_HEADER = "X-Amzn-Bedrock-GuardrailIdentifier"
_GUARDRAIL_VERSION_HEADER = "X-Amzn-Bedrock-GuardrailVersion"
_GUARDRAIL_TRACE_HEADER = "X-Amzn-Bedrock-Trace"
_GUARDRAIL_TRACE_VALUES = {"disabled", "enabled", "enabled_full"}

#: Performance configuration for the request
PERFORMANCE_CONFIG_VAR: ContextVar[
    tuple[PerformanceConfigLatencyType | None, ServiceTierTypeType | None]
] = ContextVar("performance_configuration")
_PERFORMANCE_CONFIG_LATENCY_HEADER = "X-Amzn-Bedrock-PerformanceConfig-Latency"
_SERVICE_TIER_HEADER = "X-Amzn-Bedrock-Service-Tier"

#: Prompt caching type
PromptCaching = Literal["system", "messages", "tools"]

#: Available prompt caching
PROMPT_CACHING: frozenset[PromptCaching] = frozenset(("system", "messages", "tools"))

#: Minimal prompt caching
PROMPT_CACHING_BASIC: frozenset[PromptCaching] = frozenset(("system", "messages"))

#: Default prompt caching configuration
PROMPT_CACHING_DEFAULT: ContentBlockTypeDef = {"cachePoint": {"type": "default"}}

#: AWS error codes to HTTP status + error type mapping
AWS_ERROR_MAP: dict[str, tuple[int, str]] = {
    **dict.fromkeys(
        {
            "ThrottlingException",
            "TooManyRequestsException",
            "ServiceQuotaExceededException",
        },
        (429, "rate_limit_error"),
    ),
    **dict.fromkeys({"AccessDeniedException"}, (403, "permission_error")),
    **dict.fromkeys(
        {
            "UnrecognizedClientException",
            "InvalidSignatureException",
            "ExpiredTokenException",
        },
        (401, "authentication_error"),
    ),
    **dict.fromkeys({"ResourceNotFoundException"}, (404, "not_found_error")),
    **dict.fromkeys(
        {"ValidationException", "BadRequestException", "EntityTooSmall", "InvalidPart"},
        (400, "invalid_request_error"),
    ),
    **dict.fromkeys(
        {
            "ServiceUnavailableException",
            "InternalServerException",
            "ServiceFailureException",
            "ReadTimeoutError",
        },
        (503, "server_error"),
    ),
}

#: Stream error event keys → canonical AWS error code.
INVOKE_STREAM_ERRORS: dict[str, str] = {
    "internalServerException": "InternalServerException",
    "modelStreamErrorException": "ModelStreamErrorException",
    "validationException": "ValidationException",
    "throttlingException": "ThrottlingException",
    "modelTimeoutException": "ModelTimeoutException",
    "serviceUnavailableException": "ServiceUnavailableException",
}


class _DefaultModelParameters(BaseModel):
    """Default inference parameters for a model, merged with per-request overrides.

    Accepts the standard Bedrock inference parameters plus any provider-specific
    extras (``extra="allow"``); extras are forwarded as ``additionalModelRequestFields``.
    """

    model_config = ConfigDict(extra="allow", frozen=True)
    __pydantic_extra__: JsonMapping = {}

    temperature: float | None = Field(
        default=None, ge=0, description="Default sampling temperature to use."
    )
    top_p: float | None = Field(
        validation_alias=AliasChoices("top_p", "topP"),
        default=None,
        ge=0,
        description="Default nucleus sampling.",
    )
    stop_sequences: str | list[str] | None = Field(
        validation_alias=AliasChoices("stop_sequences", "stopSequences"),
        default=None,
        description="Default sequences where the API will stop generating further tokens.",
    )
    max_tokens: int | None = Field(
        validation_alias=AliasChoices("max_tokens", "maxTokens"),
        default=None,
        ge=1,
        description="The default maximum number of tokens that can be generated by the model",
    )


def set_guardrail_configuration(headers: Headers) -> None:
    """Set the guardrail context var for the current request.

    Falls back to global settings when header override is absent or not allowed.
    Request-level headers (``X-Amzn-Bedrock-GuardrailIdentifier``,
    ``X-Amzn-Bedrock-GuardrailVersion``, ``X-Amzn-Bedrock-Trace``) are accepted
    when ``aws_bedrock_allow_guardrail_override`` is enabled.

    Args:
        headers: Incoming request headers.
    """
    if (
        SETTINGS.aws_bedrock_allow_guardrail_override
        and _GUARDRAIL_IDENTIFIER_HEADER in headers
        and _GUARDRAIL_VERSION_HEADER in headers
    ):
        config: GuardrailStreamConfigurationTypeDef = {
            "guardrailIdentifier": headers[_GUARDRAIL_IDENTIFIER_HEADER].strip(),
            "guardrailVersion": headers[_GUARDRAIL_VERSION_HEADER].strip(),
        }
        trace: GuardrailTraceType = (
            headers.get(_GUARDRAIL_TRACE_HEADER, "").strip().lower()  # type: ignore[assignment]
        )
        if trace in _GUARDRAIL_TRACE_VALUES:
            config["trace"] = trace
    elif (
        SETTINGS.aws_bedrock_guardrail_identifier
        and SETTINGS.aws_bedrock_guardrail_version
    ):
        config = {
            "guardrailIdentifier": SETTINGS.aws_bedrock_guardrail_identifier,
            "guardrailVersion": SETTINGS.aws_bedrock_guardrail_version,
        }
        if SETTINGS.aws_bedrock_guardrail_trace:
            config["trace"] = SETTINGS.aws_bedrock_guardrail_trace
    else:
        return
    GUARDRAIL_CONFIG_VAR.set(config)


#: OpenAI moderation model name prefix aliasing the default guardrail model.
_OMNI_MODERATION_PREFIX = "omni-moderation"

#: OpenAI moderation model name prefix aliasing the Comprehend moderation model.
_TEXT_MODERATION_PREFIX = "text-moderation"

#: Moderation model ID selecting the Amazon Comprehend toxicity backend.
COMPREHEND_MODERATION_MODEL: str = "amazon.comprehend-toxicity"

#: Moderation model ID selecting the server's default Bedrock guardrail.
GUARDRAIL_MODERATION_MODEL: str = "amazon.bedrock-runtime-guardrail"

#: Bedrock guardrail content filter types mapped to OpenAI moderation categories.
_GUARDRAIL_FILTER_CATEGORIES: dict[str, str] = {
    "HATE": "hate",
    "INSULTS": "harassment",
    "SEXUAL": "sexual",
    "VIOLENCE": "violence",
    "MISCONDUCT": "illicit",
}

#: Bedrock guardrail confidence levels mapped to OpenAI-style scores.
_GUARDRAIL_CONFIDENCE_SCORES: dict[str, float] = {
    "NONE": 0.0,
    "LOW": 0.25,
    "MEDIUM": 0.5,
    "HIGH": 0.75,
}

#: Guardrail policies scanned for interventions, as (policy, entry list) keys.
_GUARDRAIL_POLICY_ENTRIES: tuple[tuple[str, str], ...] = (
    ("contentPolicy", "filters"),
    ("topicPolicy", "topics"),
    ("wordPolicy", "customWords"),
    ("wordPolicy", "managedWordLists"),
    ("sensitiveInformationPolicy", "piiEntities"),
    ("sensitiveInformationPolicy", "regexes"),
    ("contextualGroundingPolicy", "filters"),
)

#: Per-request holder for Bedrock Converse guardrail trace assessments, shared across the request's tasks.
GUARDRAIL_TRACE_VAR: ContextVar[dict[str, Any]] = ContextVar("guardrail_trace")


def is_comprehend_moderation_model(model: str) -> bool:
    """Whether *model* selects (or aliases) Amazon Comprehend toxicity detection.

    Args:
        model: The moderation ``model`` value.

    Returns:
        True for ``amazon.comprehend-toxicity`` and its ``text-moderation-*``
        OpenAI aliases.
    """
    return model == COMPREHEND_MODERATION_MODEL or model.startswith(
        _TEXT_MODERATION_PREFIX
    )


def _is_default_guardrail_model(model: str) -> bool:
    """Whether *model* selects (or aliases) the server's default guardrail.

    Args:
        model: The moderation ``model`` value.

    Returns:
        True for ``amazon.bedrock-runtime-guardrail`` and its
        ``omni-moderation-*`` OpenAI aliases.
    """
    return model == GUARDRAIL_MODERATION_MODEL or model.startswith(
        _OMNI_MODERATION_PREFIX
    )


def resolve_guardrail_model(model: str | None) -> tuple[str, str]:
    """Resolve a moderation ``model`` value to a guardrail (identifier, version).

    ``amazon.bedrock-runtime-guardrail``, its ``omni-moderation-*`` OpenAI
    aliases, and ``None`` resolve to the request's configured guardrail; an
    explicit guardrail ``<id>``, ``<id>:<version>``, or ARN is honored when
    guardrail override is allowed.

    Args:
        model: The moderation ``model`` value, if any.

    Returns:
        Tuple of (guardrail identifier, guardrail version).

    Raises:
        ApiError: When no guardrail is configured or the override is not allowed.
    """
    configured = GUARDRAIL_CONFIG_VAR.get(None)
    if model is None or _is_default_guardrail_model(model):
        if configured is None:
            # Imported here because stdapi.monitoring imports this module.
            from stdapi.monitoring import log_error_details  # noqa: PLC0415

            log_error_details(
                "No moderation guardrail is configured "
                "(aws_bedrock_guardrail_identifier): 'moderation' rejected.",
                level="warning",
            )
            msg = (
                "Moderation is not enabled on the current server. Please "
                "contact the administrator to enable it, or pass an AWS "
                "Bedrock guardrail as the moderation model."
            )
            raise ApiError(msg)
        return configured["guardrailIdentifier"], configured["guardrailVersion"]
    identifier, version = model, "DRAFT"
    head, sep, tail = model.rpartition(":")
    version_given = bool(sep and (tail == "DRAFT" or tail.isdigit()))
    if version_given:
        identifier, version = head, tail
    if (
        configured
        and identifier == configured["guardrailIdentifier"]
        and (not version_given or version == configured["guardrailVersion"])
    ):
        return configured["guardrailIdentifier"], configured["guardrailVersion"]
    if not SETTINGS.aws_bedrock_allow_guardrail_override:
        msg = "Selecting a guardrail via 'model' is not allowed on this server."
        raise ApiError(msg)
    return identifier, version


def resolve_moderation_model(model: str | None) -> tuple[str, str] | None:
    """Resolve a moderation ``model`` to a guardrail or the Comprehend backend.

    ``amazon.comprehend-toxicity`` and its ``text-moderation-*`` aliases select
    Amazon Comprehend toxicity detection. An omitted model or an
    ``omni-moderation-*`` alias resolves to the configured guardrail when one
    is set and falls back to Comprehend otherwise, while
    ``amazon.bedrock-runtime-guardrail`` requires a configured guardrail.

    Args:
        model: The moderation ``model`` value, if any.

    Returns:
        Tuple of (guardrail identifier, guardrail version), or ``None`` for
        the Amazon Comprehend toxicity backend.

    Raises:
        ApiError: When no guardrail is available or the override is not allowed.
    """
    if model is not None and is_comprehend_moderation_model(model):
        return None
    if (
        model is None or model.startswith(_OMNI_MODERATION_PREFIX)
    ) and GUARDRAIL_CONFIG_VAR.get(None) is None:
        return None
    return resolve_guardrail_model(model)


def guardrail_region(identifier: str) -> RegionName:
    """Return the AWS region hosting the guardrail.

    Args:
        identifier: Guardrail identifier or ARN.

    Returns:
        The region embedded in the ARN, or the primary Bedrock region.
    """
    if identifier.startswith("arn:"):
        return identifier.split(":")[3]  # type: ignore[return-value]
    return SETTINGS.aws_bedrock_regions[0]


def map_guardrail_filters(
    assessments: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, bool], dict[str, float], bool]:
    """Map guardrail assessments to OpenAI moderation categories and scores.

    Only content policy filters map to categories; every other guardrail
    policy (topics, words, sensitive information, grounding) contributes to
    the returned intervention flag.

    Args:
        assessments: Guardrail assessments (ApplyGuardrail or Converse trace).

    Returns:
        Tuple of (category flags, category scores, whether any policy hit).
    """
    categories: dict[str, bool] = {}
    scores: dict[str, float] = {}
    intervened = False
    for assessment in assessments:
        for policy, entries in _GUARDRAIL_POLICY_ENTRIES:
            for entry in assessment.get(policy, {}).get(entries, ()):
                detected = entry.get("detected") is True or entry.get("action") in (
                    "BLOCKED",
                    "ANONYMIZED",
                )
                intervened = intervened or detected
                category = (
                    _GUARDRAIL_FILTER_CATEGORIES.get(entry.get("type", ""))
                    if policy == "contentPolicy"
                    else None
                )
                if category is None:
                    continue
                categories[category] = categories.get(category, False) or detected
                scores[category] = max(
                    scores.get(category, 0.0),
                    _GUARDRAIL_CONFIDENCE_SCORES.get(
                        entry.get("confidence", "NONE"), 0.0
                    ),
                )
    return categories, scores, intervened


def set_performance_configuration(headers: Headers) -> None:
    """Set the performance context var for the current request.

    Reads ``X-Amzn-Bedrock-PerformanceConfig-Latency`` and
    ``X-Amzn-Bedrock-Service-Tier`` from the request headers.

    Args:
        headers: Incoming request headers.
    """
    latency: PerformanceConfigLatencyType | None = (
        headers[_PERFORMANCE_CONFIG_LATENCY_HEADER].strip()  # type: ignore[assignment]
        if _PERFORMANCE_CONFIG_LATENCY_HEADER in headers
        else None
    )
    service_tier: ServiceTierTypeType | None = (
        headers[_SERVICE_TIER_HEADER].strip()  # type: ignore[assignment]
        if _SERVICE_TIER_HEADER in headers
        else None
    )
    PERFORMANCE_CONFIG_VAR.set((latency, service_tier))


def set_inference_configuration(
    model_id: str,
    additional_request_fields: JsonMapping,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop_sequences: list[str] | str | None = None,
    **extra_params: JsonValue,
) -> InferenceConfigurationTypeDef:
    """Build a Bedrock ``inferenceConfig`` dict, applying model defaults from settings.

    Parameters not provided are filled from ``SETTINGS.default_model_params[model_id]``
    when present.  Provider-specific extras (``extra_params`` and model-default extras)
    are merged into *additional_request_fields* in-place.

    Args:
        model_id: Bedrock model identifier; used to look up defaults.
        additional_request_fields: Mutable dict updated with provider-specific extras.
        temperature: Sampling temperature (0-1).
        top_p: Nucleus sampling threshold (0-1).
        max_tokens: Maximum tokens to generate.
        stop_sequences: Sequences that halt generation.
        **extra_params: Additional provider-specific fields.

    Returns:
        Populated ``InferenceConfigurationTypeDef`` dict.
    """
    config: InferenceConfigurationTypeDef = {}
    with validation_error_handler():
        default = _DefaultModelParameters(
            **SETTINGS.default_model_params.get(model_id, {})  # type: ignore[arg-type]
        )

    temperature = temperature or default.temperature
    if temperature is not None:
        config["temperature"] = temperature

    top_p = top_p or default.top_p
    if top_p is not None:
        config["topP"] = top_p

    max_tokens = max_tokens or default.max_tokens
    if max_tokens is not None:
        config["maxTokens"] = max_tokens

    stop_sequences = stop_sequences or default.stop_sequences
    if stop_sequences is not None:
        config["stopSequences"] = (
            [stop_sequences] if isinstance(stop_sequences, str) else stop_sequences
        )

    additional_request_fields.update(
        {
            key: value
            for key, value in ((default.model_extra or {}) | extra_params).items()
            if value is not None
        }
    )
    return config


def get_extra_model_parameters(
    model_id: str, request: BaseModelRequestWithExtra
) -> JsonMapping:
    """Return merged model parameters: defaults from settings overridden by request extras.

    Args:
        model_id: Bedrock model identifier; used to look up ``SETTINGS.default_model_params``.
        request: Request object whose ``model_extra`` dict takes precedence over defaults.

    Returns:
        Merged parameter dict.
    """
    return {
        **SETTINGS.default_model_params.get(model_id, {}),
        **(request.model_extra or {}),
    }


@contextmanager
def handle_bedrock_client_error() -> Generator[None]:
    """Translate Bedrock ``ClientError`` to an :class:`ApiError` with an appropriate HTTP status.

    Yields:
        None

    Raises:
        ApiError: For recognised error codes (model errors, S3 credential issues, etc.).
            Unrecognised errors are re-raised as-is.
    """
    try:
        yield
    except ClientError as error:
        error_message = error.response["Error"]["Message"]
        match error.response["Error"]["Code"]:
            case "ValidationException" if "Invalid S3 credentials" in error_message:
                msg = (
                    "Unable to access the S3 bucket. "
                    "Ensure the S3 bucket is in the same region as the Bedrock model that is called."
                )
                raise ApiError(msg) from error
            case code if code in _BEDROCK_MODEL_ERROR_CODES:  # pragma: no cover
                raise ApiError(error_message, status=500) from error
            case "ModelNotReadyException":  # pragma: no cover
                raise ApiError(error_message, status=503) from error
            case _:  # pragma: no cover
                raise


def build_system_blocks(*content: str) -> list[SystemContentBlockTypeDef]:
    """Build system content blocks, filtering empty strings.

    Args:
        *content: Text parts to include as system blocks.

    Returns:
        List of ``{"text": part}`` dicts for each non-empty part.
    """
    return [{"text": part} for part in content if part]


def bedrock_client(region: RegionName, *, single_region: bool) -> BedrockRuntimeClient:
    """Return the Bedrock runtime client appropriate for the routing mode.

    Args:
        region: AWS region to target.
        single_region: Whether the call is locked to a single region for its lifetime.

    Returns:
        A botocore Bedrock runtime async client.
    """
    return get_client(  # type: ignore[no-any-return]
        (
            "bedrock-runtime"
            if single_region or SETTINGS.aws_bedrock_region_routing == "disabled"
            else "bedrock-runtime.no-retry"
        ),
        region,
    )


def check_stream_event(event: ResponseStreamTypeDef) -> None:
    """Raise a ``ClientError`` for a recognised stream error event, or ``RuntimeError`` if unknown.

    Args:
        event: A non-chunk event from ``invoke_model_with_response_stream``.

    Raises:
        ClientError: For any recognised error event key in :data:`INVOKE_STREAM_ERRORS`.
        RuntimeError: For unrecognised event types.
    """
    for error_key, error_code in INVOKE_STREAM_ERRORS.items():
        if error_key in event:
            error_data = event[error_key]  # type: ignore[literal-required]
            error_response: dict[str, Any] = {
                "Error": {
                    "Code": error_code,
                    "Message": error_data.get("originalMessage", error_data["message"]),
                }
            }
            if "originalStatusCode" in error_data:
                error_response["ResponseMetadata"] = {
                    "HTTPStatusCode": error_data["originalStatusCode"]
                }
            raise ClientError(
                error_response,  # type: ignore[arg-type]
                "invoke_model_with_response_stream",
            )
    msg = f"Received unexpected streaming event type: {list(event.keys())}"  # pragma: no cover
    raise RuntimeError(msg)  # pragma: no cover


@dataclass(frozen=True, slots=True)
class BedrockTokenUsage:
    """Token counts extracted from a Bedrock streaming invocation-metrics event."""

    input_tokens: int
    output_tokens: int


def usage_from_amazon_bedrock_invocation_metrics(
    data: Mapping[str, Any],
) -> BedrockTokenUsage:
    """Get usage tokens from "amazon-bedrock-invocationMetrics" if present.

    Args:
        data: A mapping including potential Amazon Bedrock invocation metrics.

    Returns:
        Extracted token usage, zeroed when the metrics field is absent.
    """
    if metrics := data.get("amazon-bedrock-invocationMetrics"):
        return BedrockTokenUsage(
            metrics.get("inputTokenCount", 0), metrics.get("outputTokenCount", 0)
        )
    return BedrockTokenUsage(0, 0)
