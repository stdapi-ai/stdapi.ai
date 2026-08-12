"""Common AWS Bedrock utilities."""

from asyncio import gather
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import batched
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

from botocore.exceptions import ClientError
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
)

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.config import SETTINGS, ModelAliasConfig
from stdapi.exceptions import ServerError
from stdapi.types import JsonMapping
from stdapi.usage import record_guardrail_policy_usage
from stdapi.utils import validation_error_handler

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

    from starlette.datastructures import Headers
    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime import BedrockRuntimeClient
    from types_aiobotocore_bedrock_runtime.literals import (
        AudioFormatType,
        DocumentFormatType,
        GuardrailContentSourceType,
        GuardrailStreamProcessingModeType,
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
    # libmagic reports raw ADTS AAC streams as "audio/x-hx-aac-adts".
    "x-hx-aac-adts": "aac",
    "x-flac": "flac",
    "x-m4a": "mp4",
    # libmagic reports Matroska as "x-matroska" for both .mka and .mkv.
    "x-matroska": "mkv",
    "mpeg": "mp3",
    "x-wav": "wav",
}

#: Bedrock limit for sync body size (25MB), here with a little margin
BEDROCK_BODY_SIZE_LIMIT = 24_990_000

#: Bedrock error codes on model error
_BEDROCK_MODEL_ERROR_CODES: frozenset[str] = frozenset(
    ("ModelErrorException", "ModelStreamErrorException", "ModelTimeoutException")
)

#: Guardrail configuration for the request.
GUARDRAIL_CONFIG_VAR: ContextVar[GuardrailStreamConfigurationTypeDef] = ContextVar(
    "guardrail_configuration"
)
_GUARDRAIL_IDENTIFIER_HEADER = "X-Amzn-Bedrock-GuardrailIdentifier"
_GUARDRAIL_VERSION_HEADER = "X-Amzn-Bedrock-GuardrailVersion"
_GUARDRAIL_TRACE_HEADER = "X-Amzn-Bedrock-Trace"
_GUARDRAIL_TRACE_VALUES: frozenset[str] = frozenset(
    ("disabled", "enabled", "enabled_full")
)
#: Header selecting ConverseStream guardrail assessment timing (stream-only; stripped for Converse).
_GUARDRAIL_STREAM_PROCESSING_MODE_HEADER = (
    "X-Amzn-Bedrock-GuardrailStreamProcessingMode"
)
#: Valid values for the guardrail stream-processing-mode header/field.
_GUARDRAIL_STREAM_PROCESSING_MODE_VALUES: frozenset[str] = frozenset(("sync", "async"))

#: Whether the request itself selected the guardrail held by GUARDRAIL_CONFIG_VAR.
GUARDRAIL_REQUEST_OVERRIDE_VAR: ContextVar[bool] = ContextVar(
    "guardrail_request_override"
)

#: Performance configuration for the request
PERFORMANCE_CONFIG_VAR: ContextVar[
    tuple[PerformanceConfigLatencyType | None, ServiceTierTypeType | None]
] = ContextVar("performance_configuration")
_PERFORMANCE_CONFIG_LATENCY_HEADER = "X-Amzn-Bedrock-PerformanceConfig-Latency"
_SERVICE_TIER_HEADER = "X-Amzn-Bedrock-Service-Tier"


@dataclass(frozen=True, slots=True)
class BedrockPrompt:
    """Resolved Amazon Bedrock Prompt Management prompt used as a Converse ``modelId``.

    Attributes:
        arn: Prompt ARN, including the ``:<version>`` suffix when a version is selected.
        region: AWS region owning the prompt; the Converse call is pinned to it.
        model_id: Catalog model ID configured on the prompt variant, used for
            dispatch, response formatting and cost attribution.
    """

    arn: str
    region: RegionName
    model_id: str


#: Prompt Management prompt serving the request, set only on Responses `prompt` requests
BEDROCK_PROMPT_VAR: ContextVar[BedrockPrompt] = ContextVar("bedrock_prompt")


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


@dataclass(frozen=True, slots=True)
class AliasOverlay:
    """Configuration a model alias applies to the requests naming it.

    Every field is resolved once, when the alias table is built, so serving a
    request costs a lookup instead of a merge.

    Attributes:
        service_tier: Service tier the alias selects, if any.
        guardrail: Guardrail configuration the alias applies, if any.
        metadata: Request metadata the alias attaches, if any.
        model_params: The target model's default parameters, overridden by the
            alias' own; ``None`` when the alias sets none.
        inference_defaults: ``model_params`` validated as inference defaults.
    """

    service_tier: ServiceTierTypeType | None = None
    guardrail: GuardrailStreamConfigurationTypeDef | None = None
    metadata: Mapping[str, str] | None = None
    model_params: JsonMapping | None = None
    inference_defaults: _DefaultModelParameters | None = None


#: Configuration of the model alias the current request named, if any.
MODEL_ALIAS_OVERLAY_VAR: ContextVar[AliasOverlay | None] = ContextVar(
    "model_alias_overlay", default=None
)

#: Whether the request's alias overlay has already been resolved.
MODEL_ALIAS_OVERLAY_RESOLVED_VAR: ContextVar[bool] = ContextVar(
    "model_alias_overlay_resolved", default=False
)


def build_alias_overlay(alias: str, config: ModelAliasConfig) -> AliasOverlay:
    """Resolve a configured model alias into its ready-to-apply overlay.

    The alias' parameters are validated here rather than in the settings model:
    the inference-parameter schema lives with the request builder, which reads
    the settings and so cannot be read by them. Startup still fails, naming the
    alias, instead of every request naming it.

    Args:
        alias: Name the configuration is attached to, used in the error.
        config: Alias configuration, as validated from ``model_aliases``.

    Returns:
        The overlay applied to every request naming that alias.

    Raises:
        ServerError: When the alias' extra parameters are not valid model parameters.
    """
    guardrail: GuardrailStreamConfigurationTypeDef | None = None
    if config.guardrail_identifier and config.guardrail_version:
        guardrail = {
            "guardrailIdentifier": config.guardrail_identifier,
            "guardrailVersion": config.guardrail_version,
        }
        if config.guardrail_trace:
            guardrail["trace"] = config.guardrail_trace
    model_params: JsonMapping | None = None
    inference_defaults: _DefaultModelParameters | None = None
    if config.extra_params:
        model_params = (
            SETTINGS.default_model_params.get(config.model, {}) | config.extra_params
        )
        try:
            inference_defaults = _DefaultModelParameters(**model_params)  # type: ignore[arg-type]
        except ValidationError as error:
            msg = f"Invalid 'extra_params' on the model alias '{alias}': {error}"
            raise ServerError(msg) from error
    return AliasOverlay(
        service_tier=config.service_tier,
        guardrail=guardrail,
        metadata=config.metadata,
        model_params=model_params,
        inference_defaults=inference_defaults,
    )


def apply_alias_overlay(overlay: AliasOverlay | None) -> None:
    """Apply the overlay of the model alias the request named, if any.

    The alias sits between the request and the server-wide configuration, so
    its guardrail replaces the configured one unless the request selected a
    guardrail of its own and was allowed to.

    Only the model the request itself named installs an overlay: a secondary
    resolution within the same request (a prompt template's model, a built-in
    tool's model) must neither clear it nor replace it with its own.

    Args:
        overlay: Overlay of the named alias, or ``None`` when the request named
            a model directly.
    """
    if MODEL_ALIAS_OVERLAY_RESOLVED_VAR.get():
        return
    MODEL_ALIAS_OVERLAY_RESOLVED_VAR.set(True)
    MODEL_ALIAS_OVERLAY_VAR.set(overlay)
    if (
        overlay is not None
        and overlay.guardrail is not None
        and not GUARDRAIL_REQUEST_OVERRIDE_VAR.get(False)
    ):
        GUARDRAIL_CONFIG_VAR.set(overlay.guardrail)


def resolve_service_tier(
    model_id: str, requested: ServiceTierTypeType | None
) -> ServiceTierTypeType | None:
    """Resolve the service tier to serve a request with.

    Applies the request value, then the named alias', then
    ``default_model_service_tiers``. A request cannot displace a configured
    tier while ``aws_bedrock_allow_service_tier_override`` is disabled.

    Args:
        model_id: Bedrock model identifier serving the request.
        requested: Tier the request selected, by parameter or header.

    Returns:
        The tier to send to Bedrock, or ``None`` to let it apply its own default.
    """
    overlay = MODEL_ALIAS_OVERLAY_VAR.get()
    configured = (
        overlay.service_tier if overlay is not None else None
    ) or SETTINGS.default_model_service_tiers.get(model_id)
    if requested and (
        SETTINGS.aws_bedrock_allow_service_tier_override or not configured
    ):
        return requested
    return configured


def alias_request_metadata(
    existing: Mapping[str, str] | None,
) -> Mapping[str, str] | None:
    """Merge the named alias' metadata under the request's own.

    Args:
        existing: Metadata carried by the request, if any.

    Returns:
        The metadata to attach to the model call.
    """
    overlay = MODEL_ALIAS_OVERLAY_VAR.get()
    if overlay is None or overlay.metadata is None:
        return existing
    if not existing:
        return overlay.metadata
    return {**overlay.metadata, **existing}


def set_guardrail_configuration(headers: Headers) -> None:
    """Set the guardrail context var for the current request.

    Falls back to global settings when header override is absent or not allowed.
    Request-level headers (``X-Amzn-Bedrock-GuardrailIdentifier``,
    ``X-Amzn-Bedrock-GuardrailVersion``, ``X-Amzn-Bedrock-Trace``,
    ``X-Amzn-Bedrock-GuardrailStreamProcessingMode``) are accepted when
    ``aws_bedrock_allow_guardrail_override`` is enabled, and are recorded as
    such so a model alias does not override them.

    Also arms the model-alias overlay for the request, since resolving one
    rewrites the guardrail this sets.

    Args:
        headers: Incoming request headers.
    """
    GUARDRAIL_REQUEST_OVERRIDE_VAR.set(False)
    MODEL_ALIAS_OVERLAY_RESOLVED_VAR.set(False)
    MODEL_ALIAS_OVERLAY_VAR.set(None)
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
        # ConverseStream-only; _converse strips it back out for non-streaming requests.
        stream_processing_mode: GuardrailStreamProcessingModeType = (
            headers.get(_GUARDRAIL_STREAM_PROCESSING_MODE_HEADER, "")  # type: ignore[assignment]
            .strip()
            .lower()
        )
        if stream_processing_mode in _GUARDRAIL_STREAM_PROCESSING_MODE_VALUES:
            config["streamProcessingMode"] = stream_processing_mode
        GUARDRAIL_REQUEST_OVERRIDE_VAR.set(True)
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
    ``amazon.bedrock-runtime-guardrail`` requires a configured guardrail. The
    moderations route selects the ``amazon.bedrock-runtime-guardrail-checks``
    backend (and its no-guardrail fallback chain) before calling this
    resolver.

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


def validate_bedrock_region(region: str, *, label: str = "Region") -> None:
    """Ensure *region* is configured, so a lookup by region cannot raise an unhandled ``KeyError``.

    Args:
        region: AWS region to validate.
        label: Noun phrase describing the region's origin, prefixed to the error message.

    Raises:
        ApiError: If *region* is not in ``SETTINGS.aws_bedrock_regions``.
    """
    if region not in SETTINGS.aws_bedrock_regions:
        msg = f"{label} '{region}' is not a configured Bedrock region."
        raise ApiError(msg)


def guardrail_region(identifier: str) -> RegionName:
    """Return the AWS region hosting the guardrail.

    Args:
        identifier: Guardrail identifier or ARN.

    Returns:
        The region embedded in the ARN, or the primary Bedrock region.

    Raises:
        ApiError: If the ARN's region is not a configured Bedrock region.
    """
    if identifier.startswith("arn:"):
        region = identifier.split(":")[3]
        validate_bedrock_region(region, label="Guardrail ARN region")
        # Membership was just checked against SETTINGS.aws_bedrock_regions.
        return region  # type: ignore[return-value]
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


class GuardrailInterventionError(ApiError):
    """Content blocked by the request's configured Bedrock guardrail."""

    code = "content_filter"


#: Guardrail calls run concurrently per batch when guarding multiple texts.
_GUARDRAIL_TEXT_BATCH_SIZE: int = 10


async def apply_guardrail_to_text(
    text: str, *, source: GuardrailContentSourceType, maskable: bool = True
) -> str:
    """Apply the request's configured guardrail to *text* via ApplyGuardrail.

    Used on routes whose AWS backend has no native guardrail integration.
    No-op returning *text* unchanged when no guardrail is configured for the
    request or *text* is empty.

    Args:
        text: Text content to evaluate.
        source: ``INPUT`` for client-supplied text, ``OUTPUT`` for text
            generated by the backend.
        maskable: Whether the guardrail's masked output may replace *text*
            when the intervention only anonymized content.

    Returns:
        *text* unchanged, or the guardrail's masked output text.

    Raises:
        GuardrailInterventionError: When the guardrail blocks the content, or
            masks it while *maskable* is false.
    """
    config = GUARDRAIL_CONFIG_VAR.get(None)
    if config is None or not text:
        return text
    identifier = config["guardrailIdentifier"]
    region = guardrail_region(identifier)
    client: BedrockRuntimeClient = get_client("bedrock-runtime", region)
    with handle_bedrock_client_error():
        response = await client.apply_guardrail(
            guardrailIdentifier=identifier,
            guardrailVersion=config["guardrailVersion"],
            source=source,
            content=[{"text": {"text": text}}],
        )
    record_guardrail_policy_usage(response.get("usage", {}), region=region)
    if response.get("action") != "GUARDRAIL_INTERVENED":
        return text
    assessments: Sequence[Mapping[str, Any]] = response.get("assessments", ())
    blocked = any(
        entry.get("action") == "BLOCKED"
        for assessment in assessments
        for policy, entries in _GUARDRAIL_POLICY_ENTRIES
        for entry in assessment.get(policy, {}).get(entries, ())
    )
    outputs = response.get("outputs", ())
    if not blocked and maskable:
        # Masking-only intervention: honor it by substituting the masked text.
        return outputs[0].get("text", text) if outputs else text
    if blocked:
        # On a block, `outputs` carries the guardrail's configured messaging.
        msg = (outputs[0].get("text") if outputs else None) or (
            "The content was blocked by the configured Amazon Bedrock guardrail."
        )
    else:
        msg = (
            "The configured Amazon Bedrock guardrail masked content that "
            "cannot be substituted in this response format."
        )
    raise GuardrailInterventionError(msg)


async def apply_guardrail_to_texts[T](
    items: Sequence[T | str], *, source: GuardrailContentSourceType
) -> list[T | str]:
    """Apply the request's configured guardrail to every string in *items*.

    Each string is checked with one ApplyGuardrail call (run concurrently in
    batches of :data:`_GUARDRAIL_TEXT_BATCH_SIZE`, like the moderations
    route); non-string items pass through unchanged. No-op returning the
    items unchanged when no guardrail is configured for the request.

    Args:
        items: Request elements; only strings are guarded.
        source: ``INPUT`` for client-supplied text, ``OUTPUT`` for text
            generated by the backend.

    Returns:
        The items, with strings possibly replaced by masked guardrail output.

    Raises:
        GuardrailInterventionError: When the guardrail blocks any string.
    """
    results: list[T | str] = list(items)
    if GUARDRAIL_CONFIG_VAR.get(None) is None:
        return results
    indexed_texts = [
        (index, item) for index, item in enumerate(results) if isinstance(item, str)
    ]
    for batch in batched(indexed_texts, _GUARDRAIL_TEXT_BATCH_SIZE, strict=False):
        guarded = await gather(
            *(apply_guardrail_to_text(item, source=source) for _, item in batch)
        )
        for (index, _), new_text in zip(batch, guarded, strict=True):
            results[index] = new_text
    return results


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


#: Per-model_id validated-defaults cache: model_id -> (settings sub-dict, validated instance).
_DEFAULT_MODEL_PARAMETERS_CACHE: dict[
    str, tuple[JsonMapping | None, _DefaultModelParameters]
] = {}


def _get_default_model_parameters(model_id: str) -> _DefaultModelParameters:
    """Return the validated default parameters for *model_id*, cached by settings identity.

    A model alias carrying parameters of its own supplies them already merged
    and validated. Otherwise, settings are immutable after startup, so the
    entry is reused while ``SETTINGS.default_model_params[model_id]`` is the
    same object; replacing that mapping (e.g. in a test) invalidates it. A
    model with no configured defaults is keyed by ``None``, which compares
    identical across calls.

    Args:
        model_id: Bedrock model identifier; used to look up
            ``SETTINGS.default_model_params``.

    Returns:
        Validated default parameters for *model_id*.

    Raises:
        ApiError: When the configured defaults for *model_id* fail validation.
    """
    overlay = MODEL_ALIAS_OVERLAY_VAR.get()
    if overlay is not None and overlay.inference_defaults is not None:
        return overlay.inference_defaults
    raw = SETTINGS.default_model_params.get(model_id)
    cached = _DEFAULT_MODEL_PARAMETERS_CACHE.get(model_id)
    if cached is not None and cached[0] is raw:
        return cached[1]
    with validation_error_handler():
        parameters = _DefaultModelParameters(**(raw or {}))  # type: ignore[arg-type]
    _DEFAULT_MODEL_PARAMETERS_CACHE[model_id] = (raw, parameters)
    return parameters


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
    default = _get_default_model_parameters(model_id)

    temperature = temperature if temperature is not None else default.temperature
    if temperature is not None:
        config["temperature"] = temperature

    top_p = top_p if top_p is not None else default.top_p
    if top_p is not None:
        config["topP"] = top_p

    max_tokens = max_tokens if max_tokens is not None else default.max_tokens
    if max_tokens is not None:
        config["maxTokens"] = max_tokens

    stop_sequences = (
        stop_sequences if stop_sequences is not None else default.stop_sequences
    )
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


def filter_extra_model_parameters(extra: JsonMapping | None) -> JsonMapping:
    """Strip LiteLLM client-control parameters leaked into request extras.

    Shared by every route that forwards a request's ``model_extra`` to Bedrock as
    provider-specific inference fields: :func:`get_extra_model_parameters` for the
    non-chat routes, and each chat adapter's ``_inference_extras``.

    Args:
        extra: Candidate extra parameters keyed by name, or ``None`` when the
            request carries none.

    Returns:
        ``extra`` with dropped keys removed. Empty when it holds nothing or when
        ``SETTINGS.extra_model_params_drop_all`` is true.
    """
    if extra is None or SETTINGS.extra_model_params_drop_all:
        return {}
    dropped = SETTINGS.extra_model_params_denylist
    return {key: value for key, value in extra.items() if key not in dropped}


def get_extra_model_parameters(
    model_id: str, request: BaseModelRequestWithExtra
) -> JsonMapping:
    """Return merged model parameters: defaults from settings overridden by request extras.

    A model alias carrying parameters of its own supplies the defaults instead,
    already merged over the target model's.

    Args:
        model_id: Bedrock model identifier; used to look up ``SETTINGS.default_model_params``.
        request: Request object whose ``model_extra`` dict takes precedence over defaults,
            filtered through :func:`filter_extra_model_parameters`.

    Returns:
        Merged parameter dict.
    """
    overlay = MODEL_ALIAS_OVERLAY_VAR.get()
    defaults = (
        overlay.model_params
        if overlay is not None and overlay.model_params is not None
        else SETTINGS.default_model_params.get(model_id, {})
    )
    return {**defaults, **filter_extra_model_parameters(request.model_extra)}


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
        # Imported here: stdapi.monitoring imports stdapi.aws_bedrock (import cycle).
        from stdapi.monitoring import log_error_details  # noqa: PLC0415

        error_message = error.response["Error"]["Message"]
        match error.response["Error"]["Code"]:
            case "ValidationException" if "Invalid S3 credentials" in error_message:
                msg = (
                    "Unable to access the S3 bucket. "
                    "Ensure the S3 bucket is in the same region as the Bedrock model that is called."
                )
                raise ApiError(msg) from error
            case code if code in _BEDROCK_MODEL_ERROR_CODES:  # pragma: no cover
                log_error_details(error_message, status=500)
                msg = "The model failed to process the request."
                raise ApiError(msg, status=500) from error
            case "ModelNotReadyException":  # pragma: no cover
                log_error_details(error_message, status=503)
                msg = "The model is not ready yet. Retry the request in a few moments."
                raise ApiError(msg, status=503) from error
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
