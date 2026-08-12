"""Models."""

from asyncio import CancelledError, Lock, create_task, gather, sleep
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from importlib import import_module
from pkgutil import iter_modules
from re import Pattern
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Final, Never, TypedDict

from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    HTTPClientError,
    ReadTimeoutError,
)
from botocore.exceptions import ConnectionError as BotocoreConnectionError
from pydantic import AwareDatetime, BaseModel, Field, JsonValue
from pydantic_core import from_json, to_json

import stdapi.region_routing as _region_routing
from stdapi.api_errors import ApiError, UnsupportedModelError
from stdapi.aws import get_client
from stdapi.aws_bedrock import (
    BEDROCK_PROMPT_VAR,
    GUARDRAIL_CONFIG_VAR,
    GUARDRAIL_TRACE_VAR,
    PERFORMANCE_CONFIG_VAR,
    AliasOverlay,
    BedrockPrompt,
    alias_request_metadata,
    apply_alias_overlay,
    bedrock_client,
    build_alias_overlay,
    check_stream_event,
    handle_bedrock_client_error,
    resolve_service_tier,
    usage_from_amazon_bedrock_invocation_metrics,
    validate_bedrock_region,
)
from stdapi.aws_bedrock_mantle import MantleError
from stdapi.aws_bedrock_mantle import request_json as mantle_request_json
from stdapi.aws_s3 import (
    get_s3_bucket_for_region,
    require_s3_bucket_for_region,
    track_temporary_s3_objects,
)
from stdapi.config import SETTINGS, ModelAliasConfig
from stdapi.exceptions import ServerError
from stdapi.input_file import (
    InlineMediaLimits,
    get_s3_input_regions,
    inline_media_storage_error,
    pin_bedrock_upload_region,
    plan_bedrock_media_transport,
    resolve_all_bedrock_content_blocks,
)
from stdapi.models.capabilities import ROUTE_CAPABILITIES, Capability
from stdapi.models.deprecation import DEPRECATED_MODELS
from stdapi.models.pricing_overrides import (
    DEFAULT_MODEL_PRICE_REGIONS,
    DEFAULT_MODEL_PRICES,
    MODEL_KEY_OVERRIDES,
)
from stdapi.monitoring import (
    REQUEST_ID,
    REQUEST_LOG,
    EventLog,
    add_server_warning,
    build_metadata,
    log_error_details,
)
from stdapi.pricing import (
    Routing,
    refresh_price_catalog_for_new_models,
    register_default_prices,
    register_model_key_overrides,
)
from stdapi.region_routing import REGION_ROUTER, ROUTING_RETRYABLE_CODES
from stdapi.usage import get_model_state, record_bedrock_usage
from stdapi.utils import (
    match_bedrock_app_profile_arn,
    match_bedrock_prompt_arn,
    match_bedrock_prompt_router_arn,
)

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterable,
        Awaitable,
        Callable,
        Sequence,
    )

    from aiobotocore.eventstream import AioEventStream
    from types_aiobotocore_bedrock.client import BedrockClient
    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock.type_defs import (
        InferenceProfileModelTypeDef,
        ListInferenceProfilesRequestTypeDef,
        ListProvisionedModelThroughputsRequestTypeDef,
        PromptRouterTargetModelTypeDef,
    )
    from types_aiobotocore_bedrock_runtime import BedrockRuntimeClient
    from types_aiobotocore_bedrock_runtime.literals import ServiceTierTypeType
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
        ConverseStreamResponseTypeDef,
        GuardrailStreamConfigurationTypeDef,
        InvokeModelRequestTypeDef,
        ResponseStreamTypeDef,
    )

    from stdapi.aws_bedrock import BedrockTokenUsage, ConverseRequestBaseTypeDef
    from stdapi.input_file import BedrockMediaType

    class _ModelCache(TypedDict):
        """Model cache configuration."""

        update_next: AwareDatetime | None
        update_interval: timedelta
        update_lock: Lock
        access_lock: Lock
        user_profiles_access_lock: Lock
        prompts_access_lock: Lock

else:
    type RegionName = str


@dataclass(slots=True)
class InvokeResult[ResponseT]:
    """Result of a model invocation with optional token counts.

    Attributes:
        response: The parsed JSON response body from the model.
        input_tokens: Number of input tokens consumed, or None if not available.
        output_tokens: Number of output tokens consumed, or None if not available.
        region: Region that served the call, for usage attribution.
        tier: Service tier that served the call (AWS-reported when available).
        routing: Serving profile of the call, for usage attribution.
    """

    response: ResponseT
    input_tokens: int | None = None
    output_tokens: int | None = None
    region: str = ""
    tier: ServiceTierTypeType | None = None
    routing: Routing | None = None


#: Prefix Bedrock uses for its global cross-region inference profile IDs.
_GLOBAL_INFERENCE_PROFILE_PREFIX: Final = "global."

#: Built-in Bedrock tool names billed per invocation (counted from toolUse blocks).
_BILLED_GROUNDING_TOOLS: Final[frozenset[str]] = frozenset({"nova_grounding"})

#: Max upstream events drained after a client disconnect to still capture the trailing usage event.
_DISCONNECT_DRAIN_MAX_EVENTS: Final[int] = 50

#: Output modality advertised by rerank models (relevance rankings, not text).
RERANKING_MODALITY: str = "RERANKING"

# Keep stdapi.pricing model-agnostic: its model-key table is owned here.
register_model_key_overrides(MODEL_KEY_OVERRIDES)
register_default_prices(DEFAULT_MODEL_PRICES, DEFAULT_MODEL_PRICE_REGIONS)


def _request_routing(resolved_model_id: str, latency: str | None) -> Routing:
    """Serving profile of one prepared request, from its resolved values.

    Args:
        resolved_model_id: The model/profile ID sent to Bedrock.
        latency: The request's ``performanceConfig`` latency value, if any.

    Returns:
        "latency", "global" or "" (plain/regional).
    """
    if latency == "optimized":
        return "latency"
    if resolved_model_id.startswith(_GLOBAL_INFERENCE_PROFILE_PREFIX):
        return "global"
    return ""


def _request_uses_system_tool(request: ConverseRequestBaseTypeDef) -> bool:
    """Whether *request*'s tool configuration promotes a Bedrock system tool.

    Some system tools (e.g. ``nova_grounding``) are rejected on the ``global.``
    inference profile even when the model otherwise supports cross-region routing.

    Args:
        request: Converse request payload.

    Returns:
        True if any ``toolConfig.tools`` entry is a ``systemTool``.
    """
    tool_config = request.get("toolConfig")
    if not tool_config:
        return False
    return any("systemTool" in tool for tool in tool_config.get("tools", ()))


def _count_grounding_tool_uses(response: ConverseResponseTypeDef) -> int:
    """Count per-invocation-billed grounding-tool calls in a Converse response.

    Args:
        response: Converse API response.

    Returns:
        Number of ``toolUse`` output content blocks naming a
        :data:`_BILLED_GROUNDING_TOOLS` tool.
    """
    content = response.get("output", {}).get("message", {}).get("content", ())
    return sum(
        1
        for block in content
        if block.get("toolUse", {}).get("name") in _BILLED_GROUNDING_TOOLS
    )


def _catalog_model_id(reference: str) -> str | None:
    """Return the catalog model ID *reference* designates, if it is a known model.

    Accepts a bare model ID, a foundation-model or inference-profile ARN, and the
    ``<geography>.`` prefixed IDs of cross-region inference profiles.

    Args:
        reference: Model ID, inference profile ID, or Bedrock ARN.

    Returns:
        The matching :data:`_ALL_MODELS` key, or None when none matches.
    """
    model_id = reference.rsplit("/", 1)[-1]
    if model_id in _ALL_MODELS:
        return model_id
    stripped = model_id.split(".", 1)[-1]
    return stripped if stripped in _ALL_MODELS else None


def _invoked_model_id(response: Mapping[str, Any]) -> str | None:
    """Return the prompt router's actually-invoked model ID from *response*'s trace.

    A prompt router resolves to one of its target models per invocation; the
    caller's static model ID (the router's configured/first target) may not be
    the model that actually served the request and should be billed for it.

    Args:
        response: Converse response or ConverseStream metadata event.

    Returns:
        The invoked model's catalog ID, or None if the trace has none or it does
        not name a configured model.
    """
    invoked = response.get("trace", {}).get("promptRouter", {}).get("invokedModelId")
    return _catalog_model_id(invoked) if invoked else None


#: Bedrock models details
_MODELS: dict[str, ModelDetails] = {}

#: Service label for models served by the Amazon Bedrock Mantle endpoint.
MANTLE_SERVICE = "AWS Bedrock Mantle"

#: SPEECH-input model ID prefixes without Bedrock Converse support (bidirectional streaming only).
NON_CONVERSE_SPEECH_MODEL_PREFIXES: tuple[str, ...] = ("amazon.nova-2-sonic",)

#: Every Mantle-discovered model, including ones served by bedrock-runtime.
MANTLE_MODELS: dict[str, ModelDetails] = {}

#: Mantle model ID provider prefix to display name.
_MANTLE_PROVIDERS = {
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "google": "Google",
    "minimax": "MiniMax",
    "mistral": "Mistral AI",
    "moonshotai": "Moonshot AI",
    "nvidia": "NVIDIA",
    "openai": "OpenAI",
    "qwen": "Qwen",
    "writer": "Writer",
    "xai": "xAI",
    "zai": "Zhipu AI",
}

#: Non-Bedrock models details
EXTRA_MODELS: dict[str, ModelDetails] = {}

#: All models (Bedrock + extra services)
_ALL_MODELS: dict[str, ModelDetails] = {}

#: Bedrock models by output modality
_MODELS_OUTPUT_MODALITY: dict[str, set[str]] = {}

#: Non-Bedrock models by output modality
EXTRA_MODELS_OUTPUT_MODALITY: dict[str, set[str]] = {}

#: All models by output modality
_ALL_MODELS_OUTPUT_MODALITY: dict[str, set[str]] = {}

#: Bedrock models by input modality
_MODELS_INPUT_MODALITY: dict[str, set[str]] = {}

#: Non-Bedrock models by input modality
EXTRA_MODELS_INPUT_MODALITY: dict[str, set[str]] = {}

#: All models by input modality
_ALL_MODELS_INPUT_MODALITY: dict[str, set[str]] = {}

#: All models by supported route path or MCP tool name (search_models filter index)
_ALL_MODELS_BY_ROUTE_OR_TOOL: dict[str, set[str]] = {}

#: All models by AWS region (search_models filter index)
_ALL_MODELS_BY_REGION: dict[str, set[str]] = {}

#: Model IDs with response_streaming is True (search_models filter index)
_ALL_MODELS_STREAMING: set[str] = set()

#: Model IDs with response_streaming is False (search_models filter index)
_ALL_MODELS_NON_STREAMING: set[str] = set()

#: Model IDs with legacy is True (search_models filter index)
_ALL_MODELS_LEGACY: set[str] = set()

#: Model IDs with legacy is not True (search_models filter index)
_ALL_MODELS_NON_LEGACY: set[str] = set()

#: Model cache state
_CACHE: _ModelCache = {
    "update_next": None,
    "update_lock": Lock(),
    "update_interval": timedelta(seconds=SETTINGS.model_cache_seconds),
    "access_lock": Lock(),
    "user_profiles_access_lock": Lock(),
    "prompts_access_lock": Lock(),
}

#: Always-allowed inference types
_INFERENCE_TYPES = {"INFERENCE_PROFILE", "ON_DEMAND"}

#: TTL cache for application inference profiles and prompt routers
_USER_PROFILES: dict[str, tuple[ModelDetails, AwareDatetime]] = {}

#: TTL cache for Prompt Management prompts, keyed by versioned ARN
_PROMPTS: dict[str, tuple[str, AwareDatetime]] = {}

#: Model aliases (populated on import, merged with user settings at startup)
MODEL_ALIASES: dict[str, str] = {}

#: Configuration carried by the aliases that declare any, keyed by alias name
MODEL_ALIAS_OVERLAYS: dict[str, AliasOverlay] = {}

#: Registered model classes for all model families
_GLOBAL_MODEL_REGISTRY: set[type[ModelBase[Any, Any]]] = set()

#: Fallback model class per package (populated by load_model_plugins)
_DEFAULT: dict[str, type[ModelBase[Any, Any]]] = {}


class ModelRegionUnavailableError(Exception):
    """Raised when a model has no valid identifier for a specific region.

    Signals the region-routing layer to skip the region and try another instead
    of sending a geographically-scoped inference profile that AWS Bedrock would
    reject with ``ValidationException``: "The provided model identifier is
    invalid." Happens transiently when a region is
    discovered as offering a model before its profile has propagated there.
    """

    def __init__(self, message: str, *, region: RegionName) -> None:
        """Store the offending region alongside the message.

        Args:
            message: Human-readable error description.
            region: AWS region that cannot serve the model.
        """
        super().__init__(message)
        self.region = region


class ModelDetails(BaseModel):
    """Metadata and capability flags for a single Bedrock model.

    Attributes:
        id: Bedrock model identifier.
        name: Human-readable model name.
        provider: Model provider name (e.g. Anthropic, Amazon).
        service: AWS service hosting the model.
        input_modalities: Accepted input types (e.g. TEXT, IMAGE).
        output_modalities: Produced output types (e.g. TEXT, IMAGE).
        response_streaming: Whether the model supports streaming responses.
        legacy: Whether the model is deprecated.
        start_of_life_time: GA date, if known.
        end_of_life_time: Deprecation date, if known.
        legacy_time: Date the model was marked legacy, if known.
        public_extended_access_time: Extended public-access end date, if known.
        aliases: Alternative model IDs that resolve to this model.
        regions: All regions where the model is accessible.
        inference_profiles: Per-region inference profile ARNs.
        supported_routes: Route paths this model can serve (e.g. /v1/chat/completions).
        supported_mcp_tools: MCP tool names (operation_ids) this model can serve.
    """

    id: str
    name: str
    provider: str
    service: str = "AWS Bedrock Runtime"
    input_modalities: list[str]
    output_modalities: list[str]
    response_streaming: bool | None = None
    legacy: bool | None = None
    start_of_life_time: AwareDatetime | None = None
    end_of_life_time: AwareDatetime | None = None
    legacy_time: AwareDatetime | None = None
    public_extended_access_time: AwareDatetime | None = None
    aliases: list[str] | None = None
    regions: list[RegionName]
    inference_profiles: dict[RegionName, str] | None = None
    #: Per-region non-``global.`` inference profile cache, gateway-internal routing
    #: state, never part of the public /search_models response or OpenAPI schema.
    inference_profiles_regional: dict[RegionName, str] | None = Field(
        default=None, exclude=True
    )
    supported_routes: list[str] = []
    supported_mcp_tools: list[str] = []

    def get_id(
        self,
        region: RegionName | None = None,
        *,
        inference_profile: bool = False,
        prefer_regional: bool = False,
    ) -> str:
        """Return the model ID or inference profile ID valid for a specific region.

        A ``global.`` inference profile is valid in every region, so it is a safe
        substitute when the target region has no profile of its own. A geo-scoped
        profile (``us.``, ``eu.``, …) is only valid within its geography and is
        never returned for a different region: doing so makes Bedrock reject the
        request with ``ValidationException``.

        Args:
            region: Target AWS region. If ``None``, returns any available profile.
            inference_profile: If True, prefer the inference profile for that region.
            prefer_regional: If True, return the geo-scoped profile cached for
                *region* instead of ``global.`` when one is available (e.g. a
                system tool the model rejects on the global profile).

        Returns:
            The appropriate model identifier for the given region.

        Raises:
            ModelRegionUnavailableError: When *inference_profile* is requested for a
                specific region that has no profile of its own and no ``global.``
                profile exists as a safe fallback.
        """
        if not inference_profile:
            return self.id
        if (
            prefer_regional
            and region is not None
            and (
                regional_profile := (self.inference_profiles_regional or {}).get(region)
            )
        ):
            return regional_profile
        profiles = self.inference_profiles or {}
        if not profiles:
            # On-demand model: the bare foundation-model ID is the correct identifier.
            return self.id
        if region is not None and (profile := profiles.get(region)):
            return profile
        if global_profile := next(
            (
                pid
                for pid in profiles.values()
                if pid.startswith(_GLOBAL_INFERENCE_PROFILE_PREFIX)
            ),
            None,
        ):
            return global_profile
        if region is None:
            return next(iter(profiles.values()))
        msg = f"Model '{self.id}' has no inference profile available in region '{region}'."
        raise ModelRegionUnavailableError(msg, region=region)

    def set_inference_profile(self, region: RegionName, name: str) -> None:
        """Sets the inference profile for a specified region.

        Args:
            region: The region for which the inference profile needs
                to be set.
            name: The name of the inference profile to associate with the region.
        """
        if self.inference_profiles is None:
            self.inference_profiles = {}
        self.inference_profiles[region] = name

    def set_inference_profile_regional(self, region: RegionName, name: str) -> None:
        """Sets the geo-scoped (non-``global.``) inference profile for a specified region.

        Args:
            region: The region for which the inference profile needs to be set.
            name: The name of the geo-scoped inference profile to associate with
                the region.
        """
        if self.inference_profiles_regional is None:
            self.inference_profiles_regional = {}
        self.inference_profiles_regional[region] = name


class ModelBase[RequestT, ResponseT]:
    """Base class for provider-specific models."""

    __slots__ = ("_model_id",)

    #: Model ID matcher, regex pattern or string prefix
    MATCHER: ClassVar[str | Pattern[str]] = ""

    #: Whether this model class targets the Amazon Bedrock Mantle endpoint.
    IS_MANTLE: ClassVar[bool] = False

    #: Maps HTTP header name (lowercase) to a (field_key, transform) tuple.
    PASSTHROUGH_HEADERS: ClassVar[
        MappingProxyType[str, tuple[str, Callable[[str], Any]]]
    ] = MappingProxyType({})

    #: Regex to extract model alias from model ID
    ALIAS_MATCHER: ClassVar[Pattern[str] | None] = None

    #: Media kinds the model reads from an ``s3Location`` in a Bedrock Converse content block.
    S3_LOCATION_MEDIA_TYPES: ClassVar[frozenset[BedrockMediaType]] = frozenset()

    #: How much media the model accepts inline in one Bedrock Converse request.
    INLINE_MEDIA_LIMITS: ClassVar[InlineMediaLimits] = InlineMediaLimits()

    #: Whether InvokeModel natively accepts the guardrail configuration kwargs.
    NATIVE_GUARDRAIL_SUPPORTED: ClassVar[bool] = True

    @classmethod
    def get_supported_operations(cls) -> Capability:
        """Return capability flags for route-based model matching.

        Returns:
            Capability flags. Override in model family base classes
            to enable auto-detection of specific operation support.
        """
        return Capability(0)

    def __init__(self, model_id: str) -> None:
        """Initialize the model with its Bedrock model identifier.

        Args:
            model_id: The AWS Bedrock model identifier.
        """
        self._model_id = model_id

    @classmethod
    def get_aliases(cls, all_models: dict[str, ModelDetails]) -> dict[str, str]:
        """Return API model name aliases mapped to model IDs.

        IDs are visited Mantle-first so a bedrock-runtime model wins when
        both services derive the same alias.

        Args:
            all_models: All available models keyed by Bedrock model ID.

        Returns:
            A dict mapping model alias to model ID.
        """
        if not cls.ALIAS_MATCHER:
            return {}
        return {
            match.group(1): model_id
            for model_id in _order_ids_mantle_first(all_models)
            if (match := cls.ALIAS_MATCHER.match(model_id))
        }

    @property
    def model(self) -> ModelDetails:
        """Model details for this instance.

        A plain registry lookup: caching it per instance would require
        ``__dict__`` (the hierarchy is fully slotted) and could serve details
        made stale by a catalog refresh.

        Returns:
            Model details including region, provider, and capabilities.

        Raises:
            KeyError: If the model is not found in the registry.
        """
        try:
            return _MODELS[self._model_id]
        except KeyError:
            return EXTRA_MODELS[self._model_id]

    async def select_region(self, *, s3_required: bool = False) -> RegionName:
        """Select and lock the best invocation region for this model.

        Call this only when the region must be known *before* building the
        request body — for example to upload S3 inputs to the correct bucket
        first.  Pass the result to :meth:`invoke` / :meth:`invoke_stream` as
        ``region=`` to lock the retry loop to that region. Otherwise skip it
        and let :meth:`invoke` handle selection and multi-region retry itself.

        Args:
            s3_required: When ``True``, only regions with a configured S3
                bucket are considered as candidates.

        Returns:
            AWS region string.
        """
        candidates = await compute_candidate_regions(
            self._model_id, s3_required=s3_required
        )
        return (
            REGION_ROUTER.ordered_regions(self._model_id, candidates)
            if REGION_ROUTER
            else candidates
        )[0]

    async def invoke(
        self,
        body: RequestT | bytes,
        *,
        inference_profile: bool = True,
        region: RegionName | None = None,
        s3_required: bool = False,
        service_tier: ServiceTierTypeType | None = None,
        guardrail: GuardrailStreamConfigurationTypeDef | None = None,
    ) -> InvokeResult[ResponseT]:
        """Invoke the model via ``InvokeModel``.

        Args:
            body: JSON request payload, or an already JSON-encoded body (e.g.
                to reuse one serialization across a fan-out of identical
                invokes) -- skips re-serializing it for every call.
            inference_profile: Use the cross-region inference profile ID when available.
            region: Pin the retry loop to this region. Use with :meth:`select_region`
                when S3 inputs have already been placed in a specific region. When
                ``None``, the router selects freely across all candidate regions.
            s3_required: Restrict candidate regions to those with a configured S3
                bucket. Ignored when *region* is provided.
            service_tier: Service tier configuration. When provided, takes precedence
                over context variable and settings. Defaults to None (uses fallback).
            guardrail: Guardrail configuration. When provided, takes precedence
                over context variable. Defaults to None (uses the context var only
                when the model class supports native InvokeModel guardrails).

        Returns:
            InvokeResult containing the parsed JSON response body and token counts.
        """
        candidates = await compute_candidate_regions(
            self._model_id, region=region, s3_required=s3_required
        )
        resp = await route_and_execute(
            self._model_id,
            candidates,
            partial(
                _invoke,
                self._model_id,
                body,  # type: ignore[arg-type]
                inference_profile=inference_profile,
                single_region=len(candidates) == 1,
                service_tier=service_tier,
                guardrail=guardrail
                or (
                    GUARDRAIL_CONFIG_VAR.get(None)
                    if self.NATIVE_GUARDRAIL_SUPPORTED
                    else None
                ),
            ),
        )
        self._record_invoke_usage(
            resp.input_tokens,
            resp.output_tokens,
            resp.response,
            region=resp.region,
            tier=resp.tier,
            routing=resp.routing,
        )
        return resp  # type: ignore[return-value]

    async def invoke_stream(
        self,
        body: RequestT,
        *,
        inference_profile: bool = True,
        region: RegionName | None = None,
        s3_required: bool = False,
        service_tier: ServiceTierTypeType | None = None,
        guardrail: GuardrailStreamConfigurationTypeDef | None = None,
    ) -> AsyncGenerator[JsonValue]:
        """Invoke the model via ``InvokeModelWithResponseStream``.

        Args:
            body: JSON request payload.
            inference_profile: Use the cross-region inference profile ID when available.
            region: Pin the retry loop to this region. Use with :meth:`select_region`
                when S3 inputs have already been placed in a specific region. When
                ``None``, the router selects freely across all candidate regions.
            s3_required: Restrict candidate regions to those with a configured S3
                bucket. Ignored when *region* is provided.
            service_tier: Service tier configuration. When provided, takes precedence
                over context variable and settings. Defaults to None (uses fallback).
            guardrail: Guardrail configuration. When provided, takes precedence
                over context variable. Defaults to None (uses the context var only
                when the model class supports native InvokeModel guardrails).

        Yields:
            Parsed JSON chunks from the streaming response.
        """
        candidates = await compute_candidate_regions(
            self._model_id, region=region, s3_required=s3_required
        )
        async for chunk in await route_and_execute(
            self._model_id,
            candidates,
            partial(
                _open_invoke_stream,
                self._model_id,
                body,  # type: ignore[arg-type]
                inference_profile=inference_profile,
                single_region=len(candidates) == 1,
                service_tier=service_tier,
                guardrail=guardrail
                or (
                    GUARDRAIL_CONFIG_VAR.get(None)
                    if self.NATIVE_GUARDRAIL_SUPPORTED
                    else None
                ),
                record_usage_callback=self._record_invocation_metrics_usage,
            ),
        ):
            yield chunk

    async def invoke_async(
        self,
        body: RequestT,
        *,
        inference_profile: bool = True,
        output_file: str = "output.json",
    ) -> InvokeResult[ResponseT]:
        """Invoke the model via ``StartAsyncInvoke`` and wait for the result.

        Args:
            body: JSON request payload.
            inference_profile: Use the cross-region inference profile ID when available.
            output_file: Output file name to retrieve from S3.

        Returns:
            InvokeResult wrapping the parsed response body and billing
            attribution (async invocations report no token usage).

        Raises:
            ApiError: When invocation fails or results cannot be retrieved.
        """
        candidates = await compute_candidate_regions(self._model_id, s3_required=True)
        effective_region = (
            REGION_ROUTER.ordered_regions(self._model_id, candidates)
            if REGION_ROUTER
            else candidates
        )[0]
        s3_bucket_name = require_s3_bucket_for_region(effective_region)
        bedrock: BedrockRuntimeClient = get_client("bedrock-runtime", effective_region)
        resolved_model_id = await resolve_routed_model_id(
            self._model_id, effective_region, inference_profile=inference_profile
        )
        with handle_bedrock_client_error():
            invocation_arn = (
                await bedrock.start_async_invoke(
                    modelId=resolved_model_id,
                    modelInput=body,  # type: ignore[arg-type]
                    outputDataConfig={
                        "s3OutputDataConfig": {
                            "s3Uri": f"s3://{s3_bucket_name}/{SETTINGS.aws_s3_tmp_prefix}{REQUEST_ID.get()}/"
                        }
                    },
                    tags=[
                        {"key": k, "value": v}
                        for k, v in build_metadata(apn=True).items()
                    ],
                )
            )["invocationArn"]

        # Region locked from here — poll and retrieve in the same region
        s3_key = await _wait_for_async_invocation_completion(bedrock, invocation_arn)
        s3_output_path = f"{s3_key}/{output_file}"
        track_temporary_s3_objects(
            s3_bucket_name, s3_output_path, f"{s3_key}/manifest.json"
        )
        return InvokeResult(
            response=from_json(
                await (
                    await get_client("s3", effective_region).get_object(
                        Bucket=s3_bucket_name, Key=s3_output_path
                    )
                )["Body"].read()
            ),
            region=effective_region,
            routing=_request_routing(resolved_model_id, None),
        )

    def _record_converse_usage(
        self,
        response: ConverseResponseTypeDef,
        grounding_requests: int = 0,
        *,
        region: str = "",
        requested_tier: ServiceTierTypeType | None = None,
        routing: Routing | None = None,
    ) -> None:
        """Record token usage from a Converse API response.

        Args:
            response: Converse API response with usage metrics.
            grounding_requests: Per-invocation-billed built-in grounding-tool
                calls observed in the response/stream content
                (see :data:`_BILLED_GROUNDING_TOOLS`).
            region: Region that served the call (per-call, race-free
                attribution -- see :func:`stdapi.usage.record_bedrock_usage`).
            requested_tier: The request's tier, used when the response
                doesn't report the tier that actually served it.
            routing: Serving profile of the call.

        Note:
            When ``self._model_id`` names a prompt router, usage is billed to the
            trace's ``promptRouter.invokedModelId`` (the model the router actually
            selected) instead of the router's static first target model, when known
            (see :func:`_invoked_model_id`).
        """
        # Bill counted grounding calls even on a missing/empty usage block.
        usage = response.get("usage") or {}
        if not usage and not grounding_requests:
            return
        record_bedrock_usage(
            _invoked_model_id(response) or self._model_id,
            # AWS reports the tier that actually served the call; it takes
            # precedence over the requested one.
            tier=(response.get("serviceTier") or {}).get("type") or requested_tier,
            region=region,
            routing=routing,
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
            total_tokens=int(usage.get("totalTokens", 0)),
            cached_tokens=int(usage.get("cacheReadInputTokens", 0)),
            cache_write_tokens=int(usage.get("cacheWriteInputTokens", 0)),
            grounding_requests=grounding_requests,
            cache_write_tokens_by_ttl={
                d["ttl"]: int(d["inputTokens"])
                for d in usage.get("cacheDetails") or ()
                if "ttl" in d and "inputTokens" in d
            },
        )

    def _record_invoke_usage(
        self,
        input_tokens: int | None,
        output_tokens: int | None,
        response: Mapping[str, Any],  # noqa: ARG002
        *,
        region: str = "",
        tier: ServiceTierTypeType | None = None,
        routing: Routing | None = None,
    ) -> None:
        """Record usage from a non-Converse API invocation (e.g. InvokeModel).

        Args:
            input_tokens: Input tokens, or None.
            output_tokens: Output tokens, or None.
            response: Response body for subclass extension (e.g. counting images).
            region: Region that served the call.
            tier: Service tier that served the call.
            routing: Serving profile of the call.
        """
        record_bedrock_usage(
            self._model_id,
            tier=tier,
            region=region,
            routing=routing,
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
        )

    def _record_invocation_metrics_usage(
        self,
        data: Mapping[str, Any],
        *,
        region: str = "",
        tier: ServiceTierTypeType | None = None,
        routing: Routing | None = None,
    ) -> BedrockTokenUsage:
        """Record usage from "amazon-bedrock-invocationMetrics" if present.

        Args:
            data: A mapping including potential Amazon Bedrock invocation metrics.
            region: Region that served the call.
            tier: Service tier that served the call.
            routing: Serving profile of the call.

        Returns:
            The extracted token usage (also used by callers that need the raw
            counts, e.g. to synthesize a Converse-format usage block).
        """
        usage = usage_from_amazon_bedrock_invocation_metrics(data)
        record_bedrock_usage(
            self._model_id,
            tier=tier,
            region=region,
            routing=routing,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        return usage

    async def _prepare_converse_request_for_region(
        self, request: ConverseRequestBaseTypeDef, region: RegionName
    ) -> None:
        """Prepare a Converse request in-place for *region*.

        Folds in performance/service-tier context-var values when present,
        mirroring :func:`_build_invoke_kwargs` for the ``InvokeModel`` path.
        A Prompt Management request keeps the prompt ARN as ``modelId``: Bedrock
        materializes the conversation from the stored prompt.

        Args:
            request: Converse request payload to mutate.
            region: AWS region to target.
        """
        await resolve_all_bedrock_content_blocks(
            region, s3_location_media_types=self.S3_LOCATION_MEDIA_TYPES
        )
        latency, perf_service_tier = PERFORMANCE_CONFIG_VAR.get((None, None))
        if prompt := BEDROCK_PROMPT_VAR.get(None):
            set_effective_region(self._model_id, region)
            get_model_state(self._model_id).routing = _request_routing(
                prompt.arn, latency
            )
            request["modelId"] = prompt.arn
        else:
            request["modelId"] = await resolve_routed_model_id(
                self._model_id,
                region,
                inference_profile=True,
                latency=latency,
                prefer_regional=_request_uses_system_tool(request),
            )
        request["requestMetadata"] = build_metadata(
            alias_request_metadata(request.get("requestMetadata"))
        )

        if latency:
            request["performanceConfig"] = {"latency": latency}

        if service_tier := resolve_service_tier(
            self._model_id,
            request.get("serviceTier", {}).get("type") or perf_service_tier,
        ):
            request["serviceTier"] = {"type": service_tier}
            get_model_state(self._model_id).service_tier = service_tier

    async def _converse(
        self,
        request: ConverseRequestBaseTypeDef,
        region: RegionName,
        *,
        single_region: bool,
    ) -> ConverseResponseTypeDef:
        """Call the Bedrock Converse API for *region*.

        Args:
            request: Converse request payload (``modelId`` is injected here).
            region: AWS region to target.
            single_region: Selects the botocore client (see :func:`bedrock_client`).

        Returns:
            Bedrock Converse response.
        """
        # Fan-out callers (e.g. n > 1 choices) reuse one request dict across
        # concurrent calls: a per-call copy keeps its mutation isolated.
        request = dict(request)  # type: ignore[assignment]
        await self._prepare_converse_request_for_region(request, region)
        if guardrail_config := request.get("guardrailConfig"):
            # streamProcessingMode is ConverseStream-only; Converse rejects it as
            # an unknown parameter.
            request["guardrailConfig"] = {  # type: ignore[typeddict-item]
                key: value
                for key, value in guardrail_config.items()
                if key != "streamProcessingMode"
            }
        with handle_bedrock_client_error():
            response = await bedrock_client(
                region, single_region=single_region
            ).converse(**request)
        self._record_converse_usage(
            response,
            grounding_requests=_count_grounding_tool_uses(response),
            region=region,
            requested_tier=request.get("serviceTier", {}).get("type"),
            routing=_request_routing(
                request["modelId"], request.get("performanceConfig", {}).get("latency")
            ),
        )
        return response

    async def _converse_stream(
        self,
        request: ConverseRequestBaseTypeDef,
        region: RegionName,
        *,
        single_region: bool,
    ) -> ConverseStreamResponseTypeDef:
        """Open the Bedrock ConverseStream API for *region*.

        Failover is possible until the stream opens; once open, the region is locked.

        Args:
            request: Converse request payload (``modelId`` is injected here).
            region: AWS region to target.
            single_region: Selects the botocore client (see :func:`bedrock_client`).

        Returns:
            Bedrock ConverseStream response containing the event stream.
        """
        # Fan-out callers (e.g. n > 1 choices) reuse one request dict across
        # concurrent calls: a per-call copy keeps its mutation isolated.
        request = dict(request)  # type: ignore[assignment]
        await self._prepare_converse_request_for_region(request, region)
        with handle_bedrock_client_error():
            response = await bedrock_client(
                region, single_region=single_region
            ).converse_stream(**request)

        response["stream"] = self._capture_stream_usage(  # type: ignore[typeddict-item]
            response["stream"],
            region=region,
            requested_tier=request.get("serviceTier", {}).get("type"),
            routing=_request_routing(
                request["modelId"], request.get("performanceConfig", {}).get("latency")
            ),
        )
        return response

    def _handle_stream_event(
        self,
        event: Mapping[str, Any],
        grounding_requests: int,
        *,
        region: str,
        requested_tier: ServiceTierTypeType | None,
        routing: Routing | None,
    ) -> tuple[int, bool]:
        """Count grounding-tool calls in *event* and record usage if it carries any.

        Also captures the metadata event's guardrail trace into the request's
        shared :data:`GUARDRAIL_TRACE_VAR` holder, like non-streaming
        :meth:`converse`.

        Args:
            event: A single ConverseStream event.
            grounding_requests: Grounding-tool calls counted so far.
            region: Region that served the call.
            requested_tier: The request's tier, used when the metadata event
                doesn't report the tier that actually served the call.
            routing: Serving profile of the call.

        Returns:
            Tuple of (updated grounding-tool count, whether usage was recorded).
        """
        if (block_start := event.get("contentBlockStart")) and (
            block_start["start"].get("toolUse", {}).get("name")
            in _BILLED_GROUNDING_TOOLS
        ):
            grounding_requests += 1
        if metadata := event.get("metadata"):
            if (trace_holder := GUARDRAIL_TRACE_VAR.get(None)) is not None and (
                trace := metadata.get("trace", {}).get("guardrail")
            ):
                trace_holder.update(trace)
            if metadata.get("usage"):
                # The metadata event carries the same usage/serviceTier keys
                # as a non-streaming Converse response.
                self._record_converse_usage(
                    metadata,
                    grounding_requests=grounding_requests,
                    region=region,
                    requested_tier=requested_tier,
                    routing=routing,
                )
                return 0, True
        return grounding_requests, False

    async def _capture_stream_usage(
        self,
        stream: AsyncIterable[ConverseStreamOutputTypeDef],
        *,
        region: str = "",
        requested_tier: ServiceTierTypeType | None = None,
        routing: Routing | None = None,
    ) -> AsyncGenerator[ConverseStreamOutputTypeDef]:
        """Yield ConverseStream events, recording billed usage from metadata events.

        Per-invocation-billed grounding-tool calls (``contentBlockStart``
        events naming a :data:`_BILLED_GROUNDING_TOOLS` tool) are counted
        and attached to the usage recorded at the trailing metadata event.

        If the consumer disconnects (``GeneratorExit``/cancellation) before
        that event arrives, the remaining upstream events are drained --
        bounded by :data:`_DISCONNECT_DRAIN_MAX_EVENTS` -- so the trailing
        usage event, already billed by Bedrock, is still recorded. Not when
        the cancellation lands while a read is in flight: it closes the
        Bedrock stream first, leaving the drain nothing to read.

        Args:
            stream: Original Bedrock event stream.
            region: Region that served the call.
            requested_tier: The request's tier, used when the metadata event
                doesn't report the tier that actually served the call.
            routing: Serving profile of the call.

        Yields:
            Unmodified stream events.
        """
        grounding_requests = 0
        stream_iter = aiter(stream)
        try:
            async for event in stream_iter:
                grounding_requests, _ = self._handle_stream_event(
                    event,
                    grounding_requests,
                    region=region,
                    requested_tier=requested_tier,
                    routing=routing,
                )
                yield event
        except GeneratorExit, CancelledError:
            for _ in range(_DISCONNECT_DRAIN_MAX_EVENTS):
                try:
                    event = await anext(stream_iter)
                except StopAsyncIteration:
                    break
                grounding_requests, recorded = self._handle_stream_event(
                    event,
                    grounding_requests,
                    region=region,
                    requested_tier=requested_tier,
                    routing=routing,
                )
                if recorded:
                    break
            raise

    async def _converse_candidate_regions(self) -> list[RegionName]:
        """Return the candidate regions of a Converse call.

        Media too large to travel inline is uploaded while the request is
        prepared, and Bedrock only reads an object stored in its own region, so
        a request carrying any is pinned to a single region with storage
        configured — failover would hand the next region a reference it cannot
        read.

        Returns:
            The model's candidate regions, or a single region when the request
            is served by a Prompt Management prompt (its ARN is region-bound, so
            failover elsewhere cannot work) or carries media to upload.

        Raises:
            ApiError: When media must be uploaded but no candidate region has
                storage configured (413).
        """
        uploads = await plan_bedrock_media_transport(
            self.INLINE_MEDIA_LIMITS,
            s3_location_media_types=self.S3_LOCATION_MEDIA_TYPES,
        )
        if prompt := BEDROCK_PROMPT_VAR.get(None):
            if uploads and get_s3_bucket_for_region(prompt.region) is None:
                raise inline_media_storage_error(self.INLINE_MEDIA_LIMITS)
            return [prompt.region]
        if uploads:
            model = await get_model_details(self._model_id)
            if not any(map(get_s3_bucket_for_region, model.regions)):
                raise inline_media_storage_error(self.INLINE_MEDIA_LIMITS)
            return [
                pin_bedrock_upload_region(await self.select_region(s3_required=True))
            ]
        return await compute_candidate_regions(self._model_id)

    async def converse(
        self, request: ConverseRequestBaseTypeDef
    ) -> ConverseResponseTypeDef:
        """Invoke the model via the Bedrock Converse API.

        Args:
            request: Bedrock Converse request payload (without ``modelId``).

        Returns:
            Bedrock Converse response.
        """
        candidates = await self._converse_candidate_regions()
        response = await route_and_execute(
            self._model_id,
            candidates,
            lambda r: self._converse(request, r, single_region=len(candidates) == 1),
        )
        if (trace_holder := GUARDRAIL_TRACE_VAR.get(None)) is not None and (
            trace := response.get("trace", {}).get("guardrail")
        ):
            trace_holder.update(trace)
        return response

    async def converse_stream(
        self, request: ConverseRequestBaseTypeDef
    ) -> ConverseStreamResponseTypeDef:
        """Invoke the model via the Bedrock ConverseStream API.

        Failover is possible until the stream opens; once the stream is open the region
        is locked for the duration.

        Args:
            request: Bedrock Converse request payload (without ``modelId``).

        Returns:
            Bedrock ConverseStream response containing the event stream.
        """
        candidates = await self._converse_candidate_regions()
        return await route_and_execute(
            self._model_id,
            candidates,
            lambda r: self._converse_stream(
                request, r, single_region=len(candidates) == 1
            ),
        )


async def get_model_details(model_id: str) -> ModelDetails:
    """Return Bedrock model details by ID.

    Args:
        model_id: Bedrock model identifier.

    Returns:
        Model details.

    Raises:
        KeyError: If the model is not found.
    """
    async with _CACHE["access_lock"]:
        return _MODELS[model_id]


async def get_all_models_details() -> dict[str, ModelDetails]:
    """Return all models (Bedrock + extra services).

    Returns:
        All models keyed by model ID.
    """
    async with _CACHE["access_lock"]:
        return _ALL_MODELS


async def get_all_models_details_and_modalities() -> tuple[
    dict[str, ModelDetails], dict[str, set[str]], dict[str, set[str]]
]:
    """Return all models with their output and input modality sets.

    Returns:
        Tuple of (models dict, output-modality index, input-modality index).
    """
    async with _CACHE["access_lock"]:
        return _ALL_MODELS, _ALL_MODELS_OUTPUT_MODALITY, _ALL_MODELS_INPUT_MODALITY


async def get_all_models_search_indexes() -> tuple[
    dict[str, set[str]], dict[str, set[str]], set[str], set[str], set[str], set[str]
]:
    """Return inverted indexes over the catalogue used by the search_models filters.

    Rebuilt alongside the catalogue in :func:`update_unified_models_collections`,
    so filtering by route/MCP tool, region, streaming, or legacy status is a
    O(1) set lookup instead of a full-catalog scan.

    Returns:
        Tuple of (route path/MCP tool name to model IDs, AWS region to model
        IDs, streaming model IDs, non-streaming model IDs, legacy model IDs,
        non-legacy model IDs).
    """
    async with _CACHE["access_lock"]:
        return (
            _ALL_MODELS_BY_ROUTE_OR_TOOL,
            _ALL_MODELS_BY_REGION,
            _ALL_MODELS_STREAMING,
            _ALL_MODELS_NON_STREAMING,
            _ALL_MODELS_LEGACY,
            _ALL_MODELS_NON_LEGACY,
        )


def resolve_model_alias(model_id: str) -> str:
    """Resolve a model alias to its canonical model ID, or return *model_id* unchanged.

    Args:
        model_id: Model ID or alias.

    Returns:
        Canonical model ID.
    """
    return MODEL_ALIASES.get(model_id, model_id)


def _find_model_class(
    model_id: str, *, mantle: bool = False
) -> type[ModelBase[Any, Any]] | None:
    """Find the most specific registered model class matching *model_id*.

    Args:
        model_id: Bedrock model identifier to look up.
        mantle: When True, only consider Mantle model classes; when False,
            only classic Converse model classes.

    Returns:
        The matching model class, or ``None`` if no class is registered for this ID.
    """
    best: type[ModelBase[Any, Any]] | None = None
    best_score = -1
    for cls in _GLOBAL_MODEL_REGISTRY:
        if cls.IS_MANTLE is not mantle:
            continue
        matcher = getattr(cls, "MATCHER", "")
        if isinstance(matcher, Pattern):
            if matcher.match(model_id) and best_score < 0:
                best, best_score = cls, 0
        elif (
            matcher
            and model_id.startswith(matcher)
            and (score := len(matcher)) > best_score
        ):
            best, best_score = cls, score
    return best


def _advertised_output_modalities(
    model_id: str, output_modalities: list[str]
) -> list[str]:
    """Return the output modalities to advertise for a Bedrock model.

    Bedrock lists rerank models with a TEXT output modality; they are
    advertised with the dedicated RERANKING modality instead, since they only
    produce relevance rankings.

    Args:
        model_id: Bedrock model identifier.
        output_modalities: Output modalities from the Bedrock listing.

    Returns:
        Output modalities to advertise.
    """
    model_class = _find_model_class(model_id)
    if model_class is not None and (
        Capability.RERANK & model_class.get_supported_operations()
    ):
        return [RERANKING_MODALITY]
    return output_modalities


def _compute_model_capabilities(
    model_id: str, model: ModelDetails
) -> tuple[list[str], list[str]]:
    """Derive the routes and MCP tools a model supports from its modalities and class capabilities.

    Args:
        model_id: Bedrock model identifier.
        model: Model details containing modality information.

    Returns:
        Tuple of (sorted list of route paths, sorted list of operation_ids / MCP tool names).
    """
    model_class = _find_model_class(model_id, mantle=model.service == MANTLE_SERVICE)
    capability_flags = (
        model_class.get_supported_operations()
        if model_class is not None
        else Capability(0)
    )
    # SPEECH-input Converse models transcribe through the generic Converse STT
    # default even without a dedicated audio model class.
    if (
        "SPEECH" in model.input_modalities
        and "TEXT" in model.output_modalities
        and model.service != MANTLE_SERVICE
        and not (capability_flags & Capability.STT)
        and not model_id.startswith(NON_CONVERSE_SPEECH_MODEL_PREFIXES)
    ):
        capability_flags |= Capability.STT | Capability.STT_TRANSLATE
    input_mods = model.input_modalities
    output_mods = model.output_modalities
    routes: list[str] = []
    tools: list[str] = []
    for op_id, cap in ROUTE_CAPABILITIES.items():
        if cap.required_input_modality not in input_mods:
            continue
        if cap.required_output_modality not in output_mods:
            continue
        if cap.required_capability and not (cap.required_capability & capability_flags):
            continue
        routes.append(cap.path)
        tools.append(op_id)
    return sorted(routes), sorted(tools)


def update_unified_models_collections() -> None:
    """Merge Bedrock and extra-service model collections into the unified ``_ALL_*`` dicts.

    Rebuilds ``_ALL_MODELS``, ``_ALL_MODELS_OUTPUT_MODALITY``,
    ``_ALL_MODELS_INPUT_MODALITY``, the model-alias index, and the
    search_models inverted indexes (route/MCP tool, region, streaming, legacy).
    """
    _ALL_MODELS.clear()
    _ALL_MODELS.update(_MODELS | EXTRA_MODELS)

    _ALL_MODELS_OUTPUT_MODALITY.clear()
    _ALL_MODELS_OUTPUT_MODALITY.update(_MODELS_OUTPUT_MODALITY)
    for key, value in EXTRA_MODELS_OUTPUT_MODALITY.items():
        _ALL_MODELS_OUTPUT_MODALITY[key] = (
            _ALL_MODELS_OUTPUT_MODALITY.get(key, set()) | value
        )

    _ALL_MODELS_INPUT_MODALITY.clear()
    _ALL_MODELS_INPUT_MODALITY.update(_MODELS_INPUT_MODALITY)
    for key, value in EXTRA_MODELS_INPUT_MODALITY.items():
        _ALL_MODELS_INPUT_MODALITY[key] = (
            _ALL_MODELS_INPUT_MODALITY.get(key, set()) | value
        )

    _populate_model_aliases(_ALL_MODELS)

    _ALL_MODELS_BY_ROUTE_OR_TOOL.clear()
    _ALL_MODELS_BY_REGION.clear()
    _ALL_MODELS_STREAMING.clear()
    _ALL_MODELS_NON_STREAMING.clear()
    _ALL_MODELS_LEGACY.clear()
    _ALL_MODELS_NON_LEGACY.clear()
    for model_id, model in _ALL_MODELS.items():
        model.supported_routes, model.supported_mcp_tools = _compute_model_capabilities(
            model_id, model
        )
        for route_or_tool in (*model.supported_routes, *model.supported_mcp_tools):
            _ALL_MODELS_BY_ROUTE_OR_TOOL.setdefault(route_or_tool, set()).add(model_id)
        for region in model.regions:
            _ALL_MODELS_BY_REGION.setdefault(region, set()).add(model_id)
        if model.response_streaming is True:
            _ALL_MODELS_STREAMING.add(model_id)
        elif model.response_streaming is False:
            _ALL_MODELS_NON_STREAMING.add(model_id)
        if model.legacy is True:
            _ALL_MODELS_LEGACY.add(model_id)
        else:
            _ALL_MODELS_NON_LEGACY.add(model_id)


async def _get_provisioned_models(bedrock_client: BedrockClient) -> set[str]:
    """Return the set of provisioned model IDs available in this region.

    Args:
        bedrock_client: Bedrock control-plane client for the region.

    Returns:
        Set of provisioned model IDs (ARN suffix only).
    """
    models_ids: set[str] = set()
    params: ListProvisionedModelThroughputsRequestTypeDef = {}
    while True:
        try:
            response = await bedrock_client.list_provisioned_model_throughputs(**params)
        except ClientError as exc:  # pragma: no cover
            error = exc.response["Error"]
            if (
                error["Code"] == "AccessDeniedException"
                and "not supported" in error["Message"]
            ):
                break
            raise
        for model in response["provisionedModelSummaries"]:
            models_ids.add(model["modelArn"].rsplit("/", 1)[-1])
        if not (next_token := response.get("nextToken")):
            break
        params["nextToken"] = next_token
    return models_ids


async def _get_inference_profiles(
    bedrock_client: BedrockClient,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return mappings of model ID → inference profile ID for this region.

    Fetches active system-defined cross-region inference profiles when
    ``aws_bedrock_cross_region_inference`` is enabled.

    Args:
        bedrock_client: Bedrock control-plane client for the region.

    Returns:
        Tuple of (model ID → preferred inference profile ID,
        model ID → geo-scoped inference profile ID).
    """
    result: dict[str, str] = {}
    regional_result: dict[str, str] = {}
    if SETTINGS.aws_bedrock_cross_region_inference:
        params: ListInferenceProfilesRequestTypeDef = {
            "maxResults": 1000,
            "typeEquals": "SYSTEM_DEFINED",
        }
        profiles_all: dict[str, list[str]] = {}
        while True:
            response = await bedrock_client.list_inference_profiles(**params)
            for profile in response["inferenceProfileSummaries"]:
                if profile["status"] == "ACTIVE":
                    profiles_all.setdefault(
                        profile["models"][0]["modelArn"].rsplit("/", 1)[-1], []
                    ).append(profile["inferenceProfileId"])
            if not (next_token := response.get("nextToken")):
                break
            params["nextToken"] = next_token
        _filter_inference_profiles(result, regional_result, profiles_all)
    return result, regional_result


def _region_restriction_for(model_id: str) -> tuple[RegionName, ...] | None:
    """Return the region restriction configured for *model_id*, if any.

    Matches an exact ``aws_bedrock_model_region_restrict`` key first, then the
    first key that is a prefix of *model_id*.

    Args:
        model_id: Bedrock model ID.

    Returns:
        Allowed regions in priority order, or ``None`` when unrestricted.
    """
    restrictions = SETTINGS.aws_bedrock_model_region_restrict
    entry = restrictions.get(model_id)
    if entry is None:
        entry = next(
            (v for k, v in restrictions.items() if model_id.startswith(k)), None
        )
    return entry


def _filter_inference_profiles(
    profiles: dict[str, str],
    regional_profiles: dict[str, str],
    profiles_all: dict[str, list[str]],
) -> None:
    """Populate *profiles* with the best inference profile per model.

    Prefers the ``global.`` prefix profile when ``aws_bedrock_cross_region_inference_global``
    is enabled; otherwise picks the first non-global candidate. *regional_profiles*
    always records the first non-global candidate when one exists, so callers that
    must avoid the ``global.`` profile (e.g. a system tool unsupported on it) have
    a geo-scoped alternative to fall back on.

    Models that have an ``aws_bedrock_model_region_restrict`` entry are always
    assigned a non-global profile: a global profile would route requests
    worldwide, bypassing the configured region restriction.

    Args:
        profiles: Output dict (model ID → profile ID) updated in-place.
        regional_profiles: Output dict (model ID → non-global profile ID) updated
            in-place.
        profiles_all: All discovered profile IDs per model ID.
    """
    use_global = SETTINGS.aws_bedrock_cross_region_inference_global
    for model_id, profile_ids in profiles_all.items():
        model_restricted = _region_restriction_for(model_id) is not None
        regional_candidate = next(
            (
                pid
                for pid in profile_ids
                if not pid.startswith(_GLOBAL_INFERENCE_PROFILE_PREFIX)
            ),
            None,
        )
        if regional_candidate is not None:
            regional_profiles[model_id] = regional_candidate
        if (
            use_global
            and not model_restricted
            and (
                profile := next(
                    (
                        pid
                        for pid in profile_ids
                        if pid.startswith(_GLOBAL_INFERENCE_PROFILE_PREFIX)
                    ),
                    None,
                )
            )
        ) or (profile := regional_candidate):
            profiles[model_id] = profile


async def _get_bedrock_models_from_region(region: RegionName) -> list[ModelDetails]:
    """Fetch available foundation models from *region* and return filtered ``ModelDetails``.

    Models restricted via ``aws_bedrock_model_region_restrict`` to regions that
    exclude *region* are dropped, as are models whose ``end_of_life_time`` falls
    before the next scheduled cache refresh (a model going EOL between two cache
    updates is dropped proactively rather than served until the next refresh).

    Args:
        region: AWS region to query.

    Returns:
        List of available model details for the given region.
    """
    bedrock_client: BedrockClient = get_client("bedrock", region)

    foundation_models, provisioned_models, (profiles, regional_profiles) = await gather(
        bedrock_client.list_foundation_models(),
        _get_provisioned_models(bedrock_client),
        _get_inference_profiles(bedrock_client),
    )
    next_refresh = SETTINGS.now() + _CACHE["update_interval"]
    models: list[ModelDetails] = []
    for model in foundation_models["modelSummaries"]:
        # A legacy time reached before the next refresh counts as legacy.
        legacy_time = model["modelLifecycle"].get("legacyTime")
        if not (
            (
                legacy_time is None
                or legacy_time > next_refresh
                or SETTINGS.aws_bedrock_legacy
            )
            and (
                SETTINGS.aws_bedrock_legacy
                or (model["modelLifecycle"]["status"] != "LEGACY")
            )
            and (
                (eol := model["modelLifecycle"].get("endOfLifeTime")) is None
                or eol > next_refresh
            )
            and (
                (set(model["inferenceTypesSupported"]) & _INFERENCE_TYPES)
                or (
                    "PROVISIONED" in model["inferenceTypesSupported"]
                    and model["modelId"] in provisioned_models
                )
            )
            and (
                (allowed := _region_restriction_for(model["modelId"])) is None
                or region in allowed
            )
        ):
            continue
        models.append(
            ModelDetails(
                id=model["modelId"],
                name=model["modelName"],
                provider=model["providerName"],
                regions=[region],
                input_modalities=model["inputModalities"],  # type: ignore[arg-type]
                output_modalities=_advertised_output_modalities(
                    model["modelId"],
                    model["outputModalities"],  # type: ignore[arg-type]
                ),
                response_streaming=model.get("responseStreamingSupported"),
                inference_profiles={region: inference_profile}
                if (inference_profile := profiles.get(model["modelId"]))
                else None,
                inference_profiles_regional={region: regional_profile}
                if (regional_profile := regional_profiles.get(model["modelId"]))
                else None,
                legacy=(
                    model["modelLifecycle"]["status"] == "LEGACY"
                    or (legacy_time is not None and legacy_time <= next_refresh)
                    or None
                ),
                start_of_life_time=model["modelLifecycle"].get("startOfLifeTime"),
                end_of_life_time=model["modelLifecycle"].get("endOfLifeTime"),
                legacy_time=model["modelLifecycle"].get("legacyTime"),
                public_extended_access_time=model["modelLifecycle"].get(
                    "publicExtendedAccessTime"
                ),
            )
        )
    return models


def is_mantle_preferred(model_id: str) -> bool:
    """Whether *model_id* is configured to be served by Mantle over bedrock-runtime.

    Args:
        model_id: Model identifier.

    Returns:
        True when the ID matches an ``aws_bedrock_mantle_preferred_models``
        entry (exact or prefix).
    """
    return any(
        model_id == entry or model_id.startswith(entry)
        for entry in SETTINGS.aws_bedrock_mantle_preferred_models
    )


def is_mantle_served(model_id: str) -> bool:
    """Whether *model_id* is served by the Mantle endpoint by default.

    Args:
        model_id: Model identifier.

    Returns:
        True when the registered model's service is Mantle.
    """
    model = _ALL_MODELS.get(model_id)
    return model is not None and model.service == MANTLE_SERVICE


async def _get_mantle_models_from_region(region: RegionName) -> list[ModelDetails]:
    """Fetch the Mantle model catalog from *region*.

    Args:
        region: AWS region to query.

    Returns:
        List of available Mantle model details for the given region.
    """
    catalog = await mantle_request_json(region, "GET", "/v1/models")
    models: list[ModelDetails] = []
    for entry in catalog.get("data") or ():
        if not isinstance(entry, Mapping) or not (model_id := entry.get("id")):
            continue
        if entry.get("status") not in (None, "available"):
            continue
        provider_key, _, name = model_id.partition(".")
        model_class = _find_model_class(model_id, mantle=True)
        models.append(
            ModelDetails(
                id=model_id,
                name=name or model_id,
                provider=_MANTLE_PROVIDERS.get(provider_key, provider_key.capitalize()),
                service=MANTLE_SERVICE,
                regions=[region],
                input_modalities=list(
                    getattr(model_class, "INPUT_MODALITIES", None) or ["TEXT"]
                ),
                output_modalities=["TEXT"],
                response_streaming=True,
            )
        )
    return models


async def _collect_mantle_models(
    failed_regions: dict[str, str],
) -> dict[str, ModelDetails]:
    """Fetch the Mantle model catalog from every configured region, in parallel.

    A region whose fetch fails is skipped (recorded in *failed_regions*) and
    retried on the next cache refresh, mirroring the bedrock-runtime behavior.

    Args:
        failed_regions: Accumulator mapping unreachable regions to the error.

    Returns:
        Mantle models keyed by model ID, with per-region data merged in
        region priority order.
    """
    regions = SETTINGS.aws_bedrock_mantle_regions
    region_models = await gather(
        *(_get_mantle_models_from_region(region) for region in regions),
        return_exceptions=True,
    )
    models: dict[str, ModelDetails] = {}
    for region, result in zip(regions, region_models, strict=True):
        if isinstance(result, BaseException):
            if not isinstance(result, Exception):
                raise result
            # Any per-region failure (including credential-chain errors)
            # degrades gracefully: Mantle models are skipped with a warning.
            failed_regions[f"{region} (Mantle)"] = f"{type(result).__name__}: {result}"
            continue
        for model in result:
            if existing := models.get(model.id):
                existing.regions.extend(model.regions)
            else:
                models[model.id] = model
    return models


def _merge_mantle_models(
    all_models: dict[str, ModelDetails], mantle_models: dict[str, ModelDetails]
) -> None:
    """Merge previously-collected Mantle models into *all_models*.

    bedrock-runtime keeps priority for dual-homed models unless the model is
    explicitly preferred on Mantle.

    Args:
        all_models: Resolved bedrock-runtime models, updated in-place.
        mantle_models: Mantle models collected by :func:`_collect_mantle_models`.
    """
    MANTLE_MODELS.clear()
    MANTLE_MODELS.update(mantle_models)
    for model_id, mantle_model in mantle_models.items():
        if model_id not in all_models or is_mantle_preferred(model_id):
            all_models[model_id] = mantle_model


async def _collect_region_candidates(
    failed_regions: dict[str, str],
) -> dict[str, list[ModelDetails]]:
    """Fetch candidate models from every configured region, in parallel.

    A region whose fetch fails with an AWS error is skipped (recorded in
    *failed_regions*) instead of failing the whole refresh; it is retried
    on the next cache refresh. Models only available in a skipped region
    drop out of the cache until that region recovers.

    Args:
        failed_regions: Accumulator mapping unreachable regions to the error.

    Returns:
        Per model ID, its per-region candidates in region priority order;
        models with an ``aws_bedrock_model_region_restrict`` entry follow
        that entry's order instead.

    Raises:
        BotoCoreError: When every configured region fails (first error).
        ClientError: When every configured region fails (first error).
    """
    regions = list(_region_routing.ORDERED_BEDROCK_REGIONS)
    region_models = await gather(
        *(_get_bedrock_models_from_region(region) for region in regions),
        return_exceptions=True,
    )
    candidates: dict[str, list[ModelDetails]] = {}
    errors: list[BaseException] = []
    for region, result in zip(regions, region_models, strict=True):
        if isinstance(result, BaseException):
            if not isinstance(result, (BotoCoreError, ClientError)):
                raise result
            errors.append(result)
            failed_regions[region] = f"{type(result).__name__}: {result}"
            continue
        for model in result:
            candidates.setdefault(model.id, []).append(model)
    for model_id, model_candidates in candidates.items():
        if allowed := _region_restriction_for(model_id):
            model_candidates.sort(key=lambda model: allowed.index(model.regions[0]))
    if errors and len(errors) == len(regions):
        raise errors[0]
    return candidates


async def _check_candidates(
    candidates: dict[str, list[ModelDetails]],
    unavailable_models: dict[str, dict[str, list[str]]],
) -> dict[str, ModelDetails]:
    """Resolve candidates through availability checks, in parallel rounds.

    Each round checks every still-unresolved model against its next candidate
    region, all models concurrently; nearly all resolve in the first round. A
    model failing in one region falls through to its next candidate region.
    Once a model passes, its remaining candidate regions are merged unchecked
    (later regions are trusted). A check failing with an AWS error is recorded
    as an issue for that region, so one degraded region cannot fail a whole
    refresh.

    Args:
        candidates: Per model ID, per-region candidates in priority order.
        unavailable_models: Accumulator for failed availability checks.

    Returns:
        Available models keyed by model ID, with region data merged.

    Raises:
        BotoCoreError: When every availability check errors (first error).
        ClientError: When every availability check errors (first error).
    """
    all_models: dict[str, ModelDetails] = {}
    pending: dict[str, int] = dict.fromkeys(candidates, 0)
    errors: list[BaseException] = []
    checks = 0
    while pending:
        batch = list(pending.items())
        results = await gather(
            *(
                _check_model_availability(candidates[model_id][index])
                for model_id, index in batch
            ),
            return_exceptions=True,
        )
        checks += len(batch)
        pending = {}
        for (model_id, index), result in zip(batch, results, strict=True):
            model = candidates[model_id][index]
            if isinstance(result, BaseException):
                if not isinstance(result, (BotoCoreError, ClientError)):
                    raise result
                errors.append(result)
                issues = [f"availability check failed: {type(result).__name__}"]
            else:
                issues = result
            if not issues:
                for later in candidates[model_id][index + 1 :]:
                    _merge_candidate(model, later)
                all_models[model_id] = model
                continue
            if issues != ["unavailable"]:
                # "unavailable" alone is an AWS catalog inconsistency (listed
                # but region-unavailable, e.g. amazon.titan-embed-g1-text-02):
                # not operator-actionable, skip silently.
                unavailable_models.setdefault(model_id, {})[model.regions[0]] = issues
            if index + 1 < len(candidates[model_id]):
                pending[model_id] = index + 1
    if checks and len(errors) == checks:
        raise errors[0]
    return all_models


async def _collect_all_models(
    failed_regions: dict[str, str], unavailable_models: dict[str, dict[str, list[str]]]
) -> tuple[dict[str, ModelDetails], dict[str, str]]:
    """Collect bedrock-runtime and Mantle models concurrently and merge them.

    Mantle is a separate endpoint, so its discovery runs alongside the
    bedrock-runtime region-candidate collection and availability checks. If the
    bedrock-runtime path raises first, or the caller is cancelled, the
    still-running Mantle task is cancelled and awaited so its outcome is never
    left as an un-retrieved task warning.

    Args:
        failed_regions: Accumulator mapping unreachable regions to the error.
        unavailable_models: Accumulator for failed availability checks.

    Returns:
        Tuple of (bedrock-runtime models merged with Mantle models, invalid
        ARN mappings from :func:`_apply_user_profiles`).

    Raises:
        BotoCoreError: When every configured region fails (first error).
        ClientError: When every configured region fails (first error).
    """
    mantle_task = (
        create_task(_collect_mantle_models(failed_regions))
        if SETTINGS.aws_bedrock_mantle_enabled
        else None
    )
    try:
        candidates = await _collect_region_candidates(failed_regions)
        all_models = await _check_candidates(candidates, unavailable_models)
        invalid_arn_mappings = _apply_user_profiles(all_models)
        if mantle_task is not None:
            _merge_mantle_models(all_models, await mantle_task)
    except BaseException:
        if mantle_task is not None:
            mantle_task.cancel()
            with suppress(BaseException):
                await mantle_task
        raise
    return all_models, invalid_arn_mappings


async def initialize_bedrock_models(start_event: EventLog | None = None) -> bool:
    """Refresh the Bedrock model cache from all configured regions if stale.

    Individual unreachable regions are tolerated (warned about, retried on
    the next refresh); the refresh only fails when every region fails or
    every availability check errors.

    A lazy on-demand refresh (``start_event`` is None) that discovers
    previously unregistered model IDs triggers an immediate price-catalog
    refresh for them (see
    :func:`stdapi.pricing.refresh_price_catalog_for_new_models`), so newly
    released models get cost tracking without a background poll.

    Args:
        start_event: Optional startup event log to record warnings on for
            unreachable regions, unavailable models, invalid ARN mappings,
            and unmatched ``aws_bedrock_model_region_restrict`` keys.

    Returns:
        ``True`` if the cache was refreshed, ``False`` otherwise.
    """
    updated = False
    unavailable_models: dict[str, dict[str, list[str]]] = {}
    failed_regions: dict[str, str] = {}
    new_model_ids: set[str] = set()

    async with _CACHE["update_lock"]:
        if _CACHE["update_next"] is None or _CACHE["update_next"] <= SETTINGS.now():
            all_models, invalid_arn_mappings = await _collect_all_models(
                failed_regions, unavailable_models
            )

            mantle_guardrail_aliases = _mantle_guardrail_aliases(all_models)
            mantle_guardrail_models = (
                sum(1 for m in all_models.values() if m.service == MANTLE_SERVICE)
                if SETTINGS.aws_bedrock_guardrail_identifier
                else 0
            )

            unmatched_restrict_keys = {
                restrict_key
                for restrict_key in SETTINGS.aws_bedrock_model_region_restrict
                if not any(
                    model_id == restrict_key or model_id.startswith(restrict_key)
                    for model_id in all_models
                )
            }

            models_input: dict[str, set[str]] = {}
            models_output: dict[str, set[str]] = {}
            for model_id in sorted(all_models):
                for modality in all_models[model_id].output_modalities:
                    models_output.setdefault(modality.upper(), set()).add(model_id)
                for modality in all_models[model_id].input_modalities:
                    models_input.setdefault(modality.upper(), set()).add(model_id)

            async with _CACHE["access_lock"]:
                if all_models != _MODELS:
                    new_model_ids = set(all_models) - set(_MODELS)
                    _MODELS.clear()
                    _MODELS.update(all_models)
                    updated = True
                if models_output != _MODELS_OUTPUT_MODALITY:
                    _MODELS_OUTPUT_MODALITY.clear()
                    _MODELS_OUTPUT_MODALITY.update(models_output)
                    updated = True
                if models_input != _MODELS_INPUT_MODALITY:
                    _MODELS_INPUT_MODALITY.clear()
                    _MODELS_INPUT_MODALITY.update(models_input)
                    updated = True
                if updated:
                    update_unified_models_collections()
            _CACHE["update_next"] = SETTINGS.now() + _CACHE["update_interval"]
        else:
            invalid_arn_mappings = {}
            unmatched_restrict_keys = set()
            mantle_guardrail_models = 0
            mantle_guardrail_aliases = []
    if mantle_guardrail_aliases:
        _reject_mantle_guardrail_aliases(mantle_guardrail_aliases, start_event)
    _warn_bedrock_refresh_issues(
        start_event,
        failed_regions,
        unavailable_models,
        invalid_arn_mappings,
        unmatched_restrict_keys,
        mantle_guardrail_models,
    )
    await _trigger_price_catalog_refresh(start_event, new_model_ids)
    return updated


async def _trigger_price_catalog_refresh(
    start_event: EventLog | None, new_model_ids: set[str]
) -> None:
    """Trigger an on-demand price-catalog refresh for newly discovered Bedrock models.

    No-op for the initial startup call (``start_event`` is not None) -- that
    one already gets a full catalog from ``start_price_catalog()``.

    A refresh failure (Pricing API throttling/permission errors) is warned
    about instead of propagating: it must not fail model listing.

    Args:
        start_event: The event log passed to ``initialize_bedrock_models()``.
        new_model_ids: Model IDs discovered by this refresh that weren't
            previously registered.
    """
    if new_model_ids and start_event is None:
        try:
            await refresh_price_catalog_for_new_models(new_model_ids)
        except (BotoCoreError, ClientError) as exc:
            if REQUEST_LOG.get(None) is not None:
                log_error_details(
                    f"Price-catalog refresh for new models failed: {exc}",
                    level="warning",
                )


def _mantle_guardrail_aliases(all_models: dict[str, ModelDetails]) -> list[str]:
    """Return the guardrail-bearing aliases whose target model cannot apply it.

    Amazon Bedrock Guardrails are a bedrock-runtime feature, so a Mantle-served
    model silently ignores one. Unlike the server-wide guardrail, an alias names
    exactly one model, so the mismatch is decidable.

    Args:
        all_models: All available models keyed by model ID.

    Returns:
        Names of the offending aliases, sorted.
    """
    return sorted(
        alias
        for alias, target in SETTINGS.model_aliases.items()
        if isinstance(target, ModelAliasConfig)
        and target.guardrail_identifier
        and (model := all_models.get(target.model)) is not None
        and model.service == MANTLE_SERVICE
    )


def _reject_mantle_guardrail_aliases(
    aliases: list[str], start_event: EventLog | None
) -> None:
    """Fail startup when an alias guardrail targets a model that cannot apply it.

    Serving unfiltered content while the operator believes a guardrail applies
    is not an acceptable default, so this is fatal at startup. A later refresh
    that turns a model Mantle-served only warns: the deployment is already
    running, and stopping it would be a worse outcome than reporting it.

    Args:
        aliases: Offending alias names.
        start_event: Startup event log, or ``None`` on a lazy refresh.

    Raises:
        ServerError: At startup, naming the offending aliases.
    """
    detail = (
        f"Amazon Bedrock Guardrails configured by the model aliases "
        f"{', '.join(aliases)} cannot apply to their target model, which is "
        "served by Bedrock Mantle. Point the alias at another model, or, if "
        "the model is also available on the classic endpoint, remove it from "
        "AWS_BEDROCK_MANTLE_PREFERRED_MODELS."
    )
    if start_event is None:
        if REQUEST_LOG.get(None) is not None:
            log_error_details(detail, level="warning")
        return
    raise ServerError(detail)


def _warn_bedrock_refresh_issues(
    start_event: EventLog | None,
    failed_regions: dict[str, str],
    unavailable_models: dict[str, dict[str, list[str]]],
    invalid_arn_mappings: dict[str, str],
    unmatched_restrict_keys: set[str],
    mantle_guardrail_models: int = 0,
) -> None:
    """Record warnings for Bedrock model availability/configuration issues.

    On lazy refreshes (no *start_event*), unreachable regions are still
    surfaced as a warning on the current request log, if any.

    Args:
        start_event: Startup event log to record warnings on, if any.
        failed_regions: Regions whose fetch failed, mapped to the error.
        unavailable_models: Model IDs mapped to per-region availability issues.
        invalid_arn_mappings: Model IDs mapped to ARN-mapping error messages.
        unmatched_restrict_keys: ``aws_bedrock_model_region_restrict`` keys that
            did not match any available model.
        mantle_guardrail_models: Mantle-served models exposed while Amazon
            Bedrock Guardrails are configured (guardrails do not apply to them).
    """
    if start_event is None:
        if failed_regions and REQUEST_LOG.get(None) is not None:
            log_error_details(
                {"unreachable_bedrock_regions": failed_regions},  # type: ignore[dict-item]
                level="warning",
            )
        return
    if failed_regions:
        add_server_warning(
            start_event,
            {"unreachable_bedrock_regions": failed_regions},  # type: ignore[dict-item]
        )
    if unavailable_models:
        add_server_warning(
            start_event,
            {"unavailable_bedrock_models": unavailable_models},  # type: ignore[dict-item]
        )
    if invalid_arn_mappings:
        add_server_warning(
            start_event,
            {"invalid_bedrock_model_arn_mappings": invalid_arn_mappings},  # type: ignore[dict-item]
        )
    if unmatched_restrict_keys:
        add_server_warning(
            start_event,
            "'aws_bedrock_model_region_restrict' has no matching available model "
            f"for: {', '.join(sorted(unmatched_restrict_keys))}. Check for unknown "
            "model IDs/prefixes or models not available in the configured regions.",
        )
    if mantle_guardrail_models:
        add_server_warning(
            start_event,
            "Amazon Bedrock Guardrails do not apply to Bedrock Mantle-served "
            f"models ({mantle_guardrail_models} models affected); set "
            "AWS_BEDROCK_MANTLE_ENABLED=false to disable them.",
        )


def _apply_user_profiles(all_models: dict[str, ModelDetails]) -> dict[str, str]:
    """Apply ``aws_bedrock_model_arn_mapping`` settings to *all_models* in-place.

    Args:
        all_models: Models dict updated in-place with user-configured ARNs.

    Returns:
        Dict of model IDs → error messages for entries that could not be applied.
    """
    invalid_arn_mappings: dict[str, str] = {}
    for model_id, arn in SETTINGS.aws_bedrock_model_arn_mapping.items():
        try:
            model = all_models[model_id]
        except KeyError:
            invalid_arn_mappings[model_id] = (
                "Model not found in available Bedrock models"
            )
            continue
        if arn_match := (
            match_bedrock_app_profile_arn(arn) or match_bedrock_prompt_router_arn(arn)
        ):
            inferred_region: RegionName = arn_match.group("region")  # type: ignore[assignment]
            if inferred_region not in model.regions:
                model.regions.append(inferred_region)
            model.set_inference_profile(inferred_region, arn)
    return invalid_arn_mappings


def _resolve_deprecated(
    models: dict[str, ModelDetails], model_id: str
) -> tuple[ModelDetails | None, str]:
    """Follow the deprecation chain in *DEPRECATED_MODELS* until a live model is found.

    Args:
        models: The active models dict to look up against.
        model_id: Starting (deprecated) model ID.

    Returns:
        ``(model, effective_id)`` — *model* is ``None`` if the chain is exhausted
        without finding a live model. *effective_id* is the last ID tried.
    """
    seen = {model_id}
    while replacement := DEPRECATED_MODELS.get(model_id):
        if replacement in seen:
            break  # cycle guard
        seen.add(replacement)
        model_id = replacement
        if model := models.get(model_id):
            return model, model_id
    return None, model_id


def _warn_model_lifecycle(model: ModelDetails, original_id: str, model_id: str) -> None:
    """Emit a warning log entry if the resolved model is deprecated or legacy.

    Args:
        model: The resolved ``ModelDetails``.
        original_id: The model ID originally requested by the caller.
        model_id: The effective model ID after deprecation chain resolution.
    """
    if model_id != original_id:
        warning = (
            f"Model '{original_id}' is deprecated, routed to replacement '{model_id}'."
        )
    elif model.legacy:
        eol = f" on {model.end_of_life_time.date()}" if model.end_of_life_time else ""
        warning = f"Model '{model_id}' is legacy and will reach end-of-life{eol}. Please migrate to a supported model."
    else:
        return
    log_error_details(warning, level="warning")


def _order_ids_mantle_first(all_models: dict[str, ModelDetails]) -> list[str]:
    """Return model IDs sorted with Mantle-served models first.

    Mantle-served IDs come first so a bedrock-runtime model overwrites them
    on alias collision (runtime has serving priority).

    Args:
        all_models: All available models keyed by model ID.

    Returns:
        Model IDs, Mantle-served first.
    """
    return sorted(all_models, key=lambda mid: all_models[mid].service != MANTLE_SERVICE)


def _populate_model_aliases(all_models: dict[str, ModelDetails]) -> None:
    """Rebuild ``MODEL_ALIASES`` from all registered model classes and user settings.

    An operator alias carrying configuration contributes its target model to
    ``MODEL_ALIASES``, like the plain form, and its resolved configuration to
    ``MODEL_ALIAS_OVERLAYS``. Also sets ``ModelDetails.aliases`` for each model
    that has aliases.

    Args:
        all_models: All available models keyed by model ID.
    """
    MODEL_ALIASES.clear()
    MODEL_ALIAS_OVERLAYS.clear()
    for cls in _GLOBAL_MODEL_REGISTRY:
        MODEL_ALIASES.update(cls.get_aliases(all_models))
    for alias, target in SETTINGS.model_aliases.items():
        if isinstance(target, str):
            MODEL_ALIASES[alias] = target
        else:
            MODEL_ALIASES[alias] = target.model
            MODEL_ALIAS_OVERLAYS[alias] = build_alias_overlay(alias, target)

    aliases_by_model: dict[str, set[str]] = {}
    for alias, model_id in MODEL_ALIASES.items():
        if model_id in all_models:
            aliases_by_model.setdefault(model_id, set()).add(alias)

    for model_id, aliases in aliases_by_model.items():
        all_models[model_id].aliases = sorted(aliases)


def _merge_candidate(existing: ModelDetails, candidate: ModelDetails) -> None:
    """Append *candidate*'s region and inference profile to *existing*.

    Args:
        existing: Model already confirmed available in an earlier region.
        candidate: The same model as listed by another region.
    """
    region = candidate.regions[0]
    if region not in existing.regions:
        existing.regions.append(region)
        if profile := (candidate.inference_profiles or {}).get(region):
            existing.set_inference_profile(region, profile)
        if regional_profile := (candidate.inference_profiles_regional or {}).get(
            region
        ):
            existing.set_inference_profile_regional(region, regional_profile)


async def _check_model_availability(model: ModelDetails) -> list[str]:
    """Check one candidate model's availability in its listing region.

    Args:
        model: Candidate model from ``_get_bedrock_models_from_region``.

    Returns:
        Issue labels; empty when the model is fully available.

    Raises:
        BotoCoreError: When the availability call fails.
        ClientError: When the availability call fails.
    """
    bedrock_client: BedrockClient = get_client("bedrock", model.regions[0])
    availability = await bedrock_client.get_foundation_model_availability(
        modelId=model.id
    )
    if (
        availability["authorizationStatus"] == "AUTHORIZED"
        and availability["entitlementAvailability"] == "AVAILABLE"
        and availability["regionAvailability"] == "AVAILABLE"
        and (
            SETTINGS.aws_bedrock_marketplace_auto_subscribe
            or availability["agreementAvailability"]["status"] == "AVAILABLE"
        )
    ):
        return []
    return [
        issue
        for issue, value, expected in (
            ("unauthorized", availability["authorizationStatus"], "AUTHORIZED"),
            ("unentitled", availability["entitlementAvailability"], "AVAILABLE"),
            ("unavailable", availability["regionAvailability"], "AVAILABLE"),
            (
                "no_agreement",
                availability["agreementAvailability"]["status"],
                "AVAILABLE",
            ),
        )
        if value != expected
    ]


def load_model_plugins[ModelT: ModelBase[Any, Any]](
    package_name: str,
    class_type: type[ModelT],
    registry: list[tuple[str | Pattern[str], type[ModelT]]],
) -> None:
    """Import all modules in *package_name* and register model classes into *registry*.

    Skips private modules (``_``-prefixed), except ``_default`` which is stored
    in ``_DEFAULT`` as the fallback class.

    Args:
        package_name: Dotted package path to scan (e.g. ``stdapi.models.chat``).
        class_type: Expected model class whose name is derived from ``class_type.__name__``.
        registry: List of ``(matcher, class)`` tuples sorted by specificity after loading.
    """
    class_name = class_type.__name__.removesuffix("Base")
    for module_info in iter_modules(import_module(package_name).__path__):
        name = module_info.name
        if name == "_default":
            _DEFAULT[package_name] = getattr(
                import_module(f"{package_name}._default"), class_name
            )
            continue
        if name.startswith("_"):
            continue
        module = import_module(f"{package_name}.{name}")

        try:
            cls: type[ModelT] = getattr(module, class_name)
        except AttributeError:  # pragma: no cover
            msg = f"Module {module} does not define {class_name}"
            raise ImportError(msg) from None

        matcher = getattr(cls, "MATCHER", None)
        if not matcher:  # pragma: no cover
            msg = f"{class_name} {cls} has no MATCHER"
            raise ImportError(msg) from None

        registry.append((matcher, cls))
        _GLOBAL_MODEL_REGISTRY.add(cls)

    registry.sort(
        key=lambda item: (
            isinstance(item[0], str),
            -len(item[0] if isinstance(item[0], str) else item[0].pattern),
        )
    )


def get_model[ModelT: ModelBase[Any, Any]](
    model_id: str,
    cache: dict[str, ModelT],
    registry: list[tuple[str | Pattern[str], type[ModelT]]],
    package_name: str,
) -> ModelT:
    """Return the model instance for *model_id*, consulting *registry* on a cache miss.

    Falls back to the ``_default`` class registered for *package_name* when no matcher
    matches; raises :class:`UnsupportedModelError` if no default is registered.

    Args:
        model_id: Provider model identifier.
        cache: Per-model instance cache (updated on miss).
        registry: Ordered ``(matcher, class)`` pairs from :func:`load_model_plugins`.
        package_name: Package name used to look up the fallback default class.

    Returns:
        Cached or newly created model instance.

    Raises:
        UnsupportedModelError: If no matcher and no default class match *model_id*.
    """
    try:
        return cache[model_id]
    except KeyError:
        for matcher, model_cls in registry:
            match matcher:
                case str() if model_id.startswith(matcher):
                    cache[model_id] = model_cls(model_id)
                    return cache[model_id]
                case Pattern() if matcher.match(model_id):
                    cache[model_id] = model_cls(model_id)
                    return cache[model_id]
    try:
        cls: type[ModelT] = _DEFAULT[package_name]  # type: ignore[assignment]
    except KeyError:
        raise UnsupportedModelError(model_id) from None
    else:
        cache[model_id] = cls(model_id)
        return cache[model_id]


async def compute_candidate_regions(
    model_id: str, *, region: RegionName | None = None, s3_required: bool = False
) -> list[RegionName]:
    """Return ordered candidate regions for routing, taking S3 input locality into account.

    The priority rules are:

    1. **S3 inputs present** — regions are ranked by descending total S3 input
       data volume, and only the single best one is returned: S3 content blocks
       are resolved as a terminal operation (the ``s3Location`` URI is written
       into the request body and ``_bedrock_source`` is deleted), so retrying
       elsewhere would send a cross-region S3 reference Bedrock cannot access.
       When ``s3_required`` is also set, the ranking only considers
       bucket-configured regions (the others cannot serve an async invocation).
       When no S3 input region overlaps with the model's available regions,
       falls back to the first model region that has a configured S3 bucket
       (the object will be copied there).  Raises :class:`ApiError` when no
       such bucket region exists either.

    2. **S3 required, no S3 inputs** — restrict candidates to model regions
       that have a configured S3 bucket.  Raises :class:`ApiError` when no
       such region exists.

    3. **No S3 constraint** — return ``model.available_regions`` as-is.

    Args:
        model_id: Model ID.
        region: Region to enforce if any.
        s3_required: When ``True``, only regions with a configured S3 bucket
            are considered (used for async invocation).

    Returns:
        Ordered list of candidate region strings.  Always non-empty.

    Raises:
        ApiError: When S3 inputs are present but the model has no available
            region with a configured S3 bucket, or when S3 is required but no
            model region has a configured bucket.
    """
    if region:
        return [region]
    model = await get_model_details(model_id)
    regions = model.regions

    if s3_input_regions := get_s3_input_regions():
        sorted_input_regions = sorted(
            s3_input_regions, key=s3_input_regions.__getitem__, reverse=True
        )
        if s3_required:
            # An unbucketed region cannot serve an async invocation regardless of
            # S3 input locality, so it is not a valid pin candidate here.
            sorted_input_regions = [
                r
                for r in sorted_input_regions
                if get_s3_bucket_for_region(r) is not None
            ]
        if priority := [r for r in sorted_input_regions if r in regions]:
            # Single region only — S3 content blocks are terminal and cannot be re-resolved.
            return priority[:1]
        if bucketed := [r for r in regions if get_s3_bucket_for_region(r) is not None]:
            # No overlap: fall back to the first model region that has a bucket so the
            # S3 object can be copied there before invocation.
            return bucketed[:1]
        msg = (
            f"S3 input data is located in {sorted(s3_input_regions)} but model "
            f"'{model_id}' is only available in {sorted(regions)}."
        )
        raise ApiError(msg)

    if s3_required:
        if s3_capable := [
            r for r in regions if get_s3_bucket_for_region(r) is not None
        ]:
            return s3_capable
        msg = (
            f"Model '{model_id}' requires an S3 bucket but none is configured for its "
            f"available regions {sorted(regions)}."
        )
        raise ApiError(msg)

    return regions


async def route_and_execute[T](
    model_id: str,
    candidates: list[RegionName],
    fn: Callable[[RegionName], Awaitable[T]],
) -> T:
    """Route ``fn`` across candidate regions with retry and health bookkeeping.

    With a single candidate (region lock or S3-constrained), delegates directly to
    botocore's adaptive retries within that region. With routing disabled, uses
    the first candidate. Otherwise walks router-ordered regions for up to
    ``SETTINGS.aws_bedrock_max_retries + 1`` attempts — each candidate tried at most
    once — escalating to the next region on any retryable error and marking the
    region healthy on success.

    A region that cannot serve the model — because it has no valid inference
    profile (:class:`ModelRegionUnavailableError`) or because Bedrock reports the
    model identifier as invalid — is skipped like any other retryable error, so a
    transiently incomplete discovery snapshot self-heals across regions instead of
    surfacing to the caller.

    ``ReadTimeoutError`` is never retried across regions: the request already
    reached Bedrock and is billed regardless, so failing over would double-bill
    the invocation instead of recovering it.

    Args:
        model_id: Used for router health bookkeeping.
        candidates: From ``compute_candidate_regions``. Single-element means region-locked.
        fn: Must call ``set_effective_region`` before the AWS call.

    Returns:
        Result of the first successful ``fn`` call.

    Raises:
        ApiError: When no candidate region can serve the model, or (wrapping a
            retryable AWS error code such as ``ModelNotReadyException``) as the last
            error when all attempts exhausted.
        ClientError: Last non-retryable error, or last retryable error when all attempts exhausted.
        BotocoreConnectionError: Last connection error when all attempts exhausted.
        HTTPClientError: Last HTTP client error when all attempts exhausted, or immediately
            for a ``ReadTimeoutError`` (never retried).
    """
    if not REGION_ROUTER or len(candidates) == 1:
        return await _execute_pinned(candidates[0], fn)

    last_exc: (
        ClientError
        | ApiError
        | BotocoreConnectionError
        | HTTPClientError
        | ModelRegionUnavailableError
        | MantleError
    )
    # A region is attempted at most once per request: every failed attempt marks its
    # region, so once all candidates are blocked the router leads with the same one
    # again — and re-invoking it back-to-back on the single-attempt no-retry client
    # cannot succeed, while a second quota error would double its backoff, letting a
    # single request escalate one region to the hour-long ceiling that the
    # ``retry-after`` the client receives never reflects.
    remaining = list(candidates)
    for _ in range(min(SETTINGS.aws_bedrock_max_retries + 1, len(candidates))):
        region = REGION_ROUTER.ordered_regions(model_id, remaining)[0]
        remaining.remove(region)
        try:
            result = await fn(region)
        except MantleError as exc:
            if not exc.failover:
                raise
            last_exc = exc
            REGION_ROUTER.mark_error(model_id, region, _mantle_failover_label(exc))
        except (ClientError, ApiError) as exc:
            if (code := _retryable_error_code(exc)) is None:
                raise
            last_exc = exc
            REGION_ROUTER.mark_error(model_id, region, code)
        except (
            ModelRegionUnavailableError,
            BotocoreConnectionError,
            HTTPClientError,
        ) as exc:
            last_exc = exc
            REGION_ROUTER.mark_error(model_id, region, _region_failover_label(exc))
        else:
            REGION_ROUTER.mark_success(model_id, region)
            return result

    if isinstance(last_exc, ModelRegionUnavailableError):
        raise _model_unavailable_api_error(last_exc) from last_exc
    raise last_exc


async def _execute_pinned[T](
    region: RegionName, fn: Callable[[RegionName], Awaitable[T]]
) -> T:
    """Invoke *fn* on *region* with no other region to fail over to.

    Args:
        region: The only region able to serve the request.
        fn: Must call ``set_effective_region`` before the AWS call.

    Returns:
        Result of the ``fn`` call.

    Raises:
        ApiError: When *region* cannot serve the model.
    """
    try:
        return await fn(region)
    except ModelRegionUnavailableError as exc:
        raise _model_unavailable_api_error(exc) from exc


def _mantle_failover_label(exc: MantleError) -> str:
    """Return the router health-bookkeeping label for a failover *exc*.

    Args:
        exc: Mantle error that allows failover.

    Returns:
        The Converse error code whose backoff matches *exc*'s HTTP status, so the
        router applies the quota escalation only to a throttle.
    """
    return "ThrottlingException" if exc.status == 429 else "ServiceUnavailableException"


def _retryable_error_code(exc: ClientError | ApiError) -> str | None:
    """Return the AWS error code of *exc* when another region should be tried.

    A ``ClientError`` always carries a code, so only a bare ``ApiError`` can yield
    None, and that never satisfies the invalid-model-identifier check.

    Args:
        exc: Error raised by a Bedrock invocation.

    Returns:
        The AWS error code to record via ``mark_error``, or None when the error is
        fatal in every region and must be re-raised.
    """
    if (code := _client_error_code(exc)) is None:
        return None
    retryable = code in ROUTING_RETRYABLE_CODES or (
        isinstance(exc, ClientError) and _is_invalid_model_identifier(exc)
    )
    return code if retryable else None


def _model_unavailable_api_error(exc: ModelRegionUnavailableError) -> ApiError:
    """Return the client-facing error for *exc*, logging its routing diagnostic.

    Args:
        exc: Signal that a region has no valid identifier for the model.

    Returns:
        ``ApiError`` to raise in place of *exc*, which is an internal routing
        signal that Starlette would otherwise answer as a bare 500.
    """
    # The exception text is a routing diagnostic naming regions and profiles;
    # it belongs in the log, not in the client's response.
    log_error_details(str(exc), level="warning")
    msg = (
        "The requested model is not available. Select a different model, "
        "or check that access to this one has been granted."
    )
    return ApiError(msg)


def _region_failover_label(
    exc: ModelRegionUnavailableError | BotocoreConnectionError | HTTPClientError,
) -> str:
    """Return the router health-bookkeeping label for *exc*.

    Args:
        exc: Error raised while invoking a Bedrock region.

    Returns:
        *exc*'s class name, to record via ``REGION_ROUTER.mark_error``.

    Raises:
        ReadTimeoutError: Re-raises *exc* unchanged instead of returning a label. The
            request already reached Bedrock and is billed regardless of this
            client-side timeout, so retrying in another region would double-bill the
            invocation instead of recovering it.
    """
    if isinstance(exc, ReadTimeoutError):
        raise exc
    return exc.__class__.__name__


def _client_error_code(exc: ClientError | ApiError) -> str | None:
    """Return the AWS error code carried by *exc*, directly or via its wrapped cause.

    ``handle_bedrock_client_error`` converts some ``ClientError`` instances (e.g.
    ``ModelNotReadyException``) to ``ApiError`` via ``raise ... from client_error``,
    so the original code must be recovered from ``__cause__`` in that case.

    Args:
        exc: Error raised by a Bedrock invocation.

    Returns:
        The AWS error code, or None if unavailable.
    """
    if isinstance(exc, ClientError):
        return exc.response["Error"]["Code"]
    cause = exc.__cause__
    return cause.response["Error"]["Code"] if isinstance(cause, ClientError) else None


def _is_invalid_model_identifier(exc: ClientError) -> bool:
    """Return True when *exc* is a Bedrock "invalid model identifier" validation error.

    Such an error means the selected region cannot serve the model (e.g. its
    inference profile has not propagated there yet), so the request should be
    retried in another region rather than failing outright.

    Args:
        exc: Bedrock ``ClientError`` to classify.

    Returns:
        True if the error is a ``ValidationException`` reporting an invalid model identifier.
    """
    error = exc.response.get("Error", {})
    return error.get("Code") == "ValidationException" and (
        "model identifier is invalid" in error.get("Message", "")
    )


def set_effective_region(model_id: str, region: RegionName) -> None:
    """Record *region* in the request log's ``model_regions`` set and *model_id*'s invocation state.

    Args:
        model_id: Bedrock model identifier the region applies to.
        region: AWS region selected for this request.
    """
    log = REQUEST_LOG.get()
    if "model_regions" not in log:
        log["model_regions"] = set()
    log["model_regions"].add(region)
    get_model_state(model_id).region = region


async def resolve_routed_model_id(
    model_id: str,
    region: RegionName,
    *,
    inference_profile: bool,
    latency: str | None = None,
    prefer_regional: bool = False,
) -> str:
    """Resolve the model/profile ID to send to Bedrock for *region* and record its routing.

    Args:
        model_id: Bedrock model identifier.
        region: Target AWS region.
        inference_profile: Use the cross-region inference profile ID when available.
        latency: The request's ``performanceConfig`` latency value, if any.
        prefer_regional: Prefer a cached geo-scoped inference profile over
            ``global.`` for *region* (e.g. a system tool unsupported on it).

    Returns:
        The resolved model or inference-profile ID to send to Bedrock.
    """
    set_effective_region(model_id, region)
    resolved_model_id = (await get_model_details(model_id)).get_id(
        region, inference_profile=inference_profile, prefer_regional=prefer_regional
    )
    get_model_state(model_id).routing = _request_routing(resolved_model_id, latency)
    return resolved_model_id


async def _build_invoke_kwargs(
    model_id: str,
    body: Mapping[str, Any] | bytes,
    region: RegionName,
    *,
    inference_profile: bool,
    service_tier: ServiceTierTypeType | None = None,
    guardrail: GuardrailStreamConfigurationTypeDef | None = None,
) -> InvokeModelRequestTypeDef:
    """Build the kwargs dict for ``InvokeModel`` / ``InvokeModelWithResponseStream``.

    Folds in the performance context-var values when present; the guardrail
    configuration is applied only when explicitly provided.

    Args:
        model_id: Bedrock model identifier.
        body: JSON request payload, or an already JSON-encoded body (skips
            re-serializing it).
        region: Target AWS region (used to resolve the model/profile ID).
        inference_profile: Use cross-region inference profile ID when available.
        service_tier: Service tier configuration. When provided, takes precedence
            over context variable and settings. When None, falls back to
            PERFORMANCE_CONFIG_VAR and SETTINGS.default_model_service_tiers.
        guardrail: Guardrail configuration, already resolved by the caller.

    Returns:
        Fully populated request kwargs dict.
    """
    latency, perf_service_tier = PERFORMANCE_CONFIG_VAR.get((None, None))
    resolved_model_id = await resolve_routed_model_id(
        model_id, region, inference_profile=inference_profile, latency=latency
    )
    kwargs: InvokeModelRequestTypeDef = {
        "modelId": resolved_model_id,
        "contentType": "application/json",
        "accept": "application/json",
        "body": body if isinstance(body, bytes) else to_json(body),
        # InvokeModel carries the metadata as a JSON string header, unlike
        # Converse which takes a map.
        "requestMetadata": to_json(
            build_metadata(alias_request_metadata(None))
        ).decode(),
    }

    if guardrail is not None:
        kwargs["guardrailIdentifier"] = guardrail["guardrailIdentifier"]
        kwargs["guardrailVersion"] = guardrail["guardrailVersion"]
        if trace := guardrail.get("trace"):
            kwargs["trace"] = trace.upper()  # type: ignore[typeddict-item]

    if latency:
        kwargs["performanceConfigLatency"] = latency
    if service_tier := resolve_service_tier(
        model_id, service_tier or perf_service_tier
    ):
        get_model_state(model_id).service_tier = kwargs["serviceTier"] = service_tier
    return kwargs


async def _invoke(
    model_id: str,
    body: Mapping[str, Any] | bytes,
    region: RegionName,
    *,
    inference_profile: bool,
    single_region: bool,
    service_tier: ServiceTierTypeType | None = None,
    guardrail: GuardrailStreamConfigurationTypeDef | None = None,
) -> InvokeResult[Mapping[str, Any]]:
    """Call ``InvokeModel`` and return the parsed JSON response with token counts.

    Args:
        model_id: Bedrock model identifier.
        body: JSON request payload, or an already JSON-encoded body (skips
            re-serializing it).
        region: AWS region to target.
        inference_profile: Use cross-region inference profile ID when available.
        single_region: Selects the botocore client (see :func:`bedrock_client`).
        service_tier: Service tier configuration (takes precedence over context var).
        guardrail: Guardrail configuration, already resolved by the caller.

    Returns:
        InvokeResult containing the response body and token counts.
    """
    kwargs = await _build_invoke_kwargs(
        model_id,
        body,
        region,
        inference_profile=inference_profile,
        service_tier=service_tier,
        guardrail=guardrail,
    )
    with handle_bedrock_client_error():
        response = await bedrock_client(
            region, single_region=single_region
        ).invoke_model(**kwargs)
    headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    return InvokeResult(
        response=from_json(await response["body"].read()),
        input_tokens=(
            int(v) if (v := headers.get("x-amzn-bedrock-input-token-count")) else None
        ),
        output_tokens=(
            int(v) if (v := headers.get("x-amzn-bedrock-output-token-count")) else None
        ),
        region=region,
        # AWS reports the tier that actually served the call; bill with it.
        tier=response.get("serviceTier") or kwargs.get("serviceTier"),
        routing=_request_routing(
            kwargs["modelId"], kwargs.get("performanceConfigLatency")
        ),
    )


async def _iter_invoke_stream(
    body: AioEventStream[ResponseStreamTypeDef],
    record_usage_callback: Callable[..., object] | None = None,
) -> AsyncGenerator[JsonValue]:
    """Iterate over an open ``InvokeModelWithResponseStream`` event stream.

    Args:
        body: Open event stream from ``invoke_model_with_response_stream``.
        record_usage_callback: Optional callback to record usage from each event.

    Yields:
        Parsed JSON chunks.
    """
    async for event in body:
        if "chunk" in event:
            chunk = from_json(event["chunk"]["bytes"])
            if (
                record_usage_callback is not None
                and isinstance(chunk, Mapping)
                and "amazon-bedrock-invocationMetrics" in chunk
            ):
                record_usage_callback(chunk)
            yield chunk
        else:
            check_stream_event(event)


async def _open_invoke_stream(
    model_id: str,
    body: Mapping[str, Any],
    region: RegionName,
    *,
    inference_profile: bool,
    single_region: bool,
    service_tier: ServiceTierTypeType | None = None,
    guardrail: GuardrailStreamConfigurationTypeDef | None = None,
    record_usage_callback: Callable[..., object] | None = None,
) -> AsyncGenerator[JsonValue]:
    """Open an ``InvokeModelWithResponseStream`` connection and return a JSON chunk generator.

    The HTTP connection is established before the first ``yield`` so
    :func:`route_and_execute` can retry on a different region if the open fails.
    Once the stream is open, retrying is no longer possible.

    Args:
        model_id: Bedrock model identifier.
        body: JSON request payload.
        region: AWS region to target.
        inference_profile: Use cross-region inference profile ID when available.
        single_region: Selects the botocore client (see :func:`bedrock_client`).
        service_tier: Service tier configuration (takes precedence over context var).
        guardrail: Guardrail configuration, already resolved by the caller.
        record_usage_callback: Optional callback to record usage from streaming events.

    Returns:
        Async generator yielding parsed JSON chunks.
    """
    kwargs = await _build_invoke_kwargs(
        model_id,
        body,
        region,
        inference_profile=inference_profile,
        service_tier=service_tier,
        guardrail=guardrail,
    )
    with handle_bedrock_client_error():
        response = await bedrock_client(
            region, single_region=single_region
        ).invoke_model_with_response_stream(**kwargs)
    if record_usage_callback is not None:
        # Bind this call's attribution (see record_bedrock_usage's `region`).
        record_usage_callback = partial(
            record_usage_callback,
            region=region,
            # AWS reports the tier that actually served the call.
            tier=response.get("serviceTier") or kwargs.get("serviceTier"),
            routing=_request_routing(
                kwargs["modelId"], kwargs.get("performanceConfigLatency")
            ),
        )

    return _iter_invoke_stream(response["body"], record_usage_callback)


def _raise_model_not_found(
    models: dict[str, ModelDetails],
    original_id: str,
    model_id: str,
    input_modality: str | None,
    output_modality: str | None,
    error_status: int | None,
    *,
    bedrock_only: bool,
) -> Never:
    """Raise :exc:`UnsupportedModelError` with an appropriate message.

    Args:
        models: Active models dict, used to build the set of available model IDs.
        original_id: The model ID originally requested by the caller.
        model_id: The effective model ID after deprecation chain resolution.
        input_modality: Required input modality to filter available model IDs.
        output_modality: Required output modality to filter available model IDs.
        error_status: HTTP status code override for :exc:`UnsupportedModelError`.
        bedrock_only: Whether to restrict modality index lookups to Bedrock models.

    Raises:
        UnsupportedModelError: Always.
    """
    detail: str | None = None
    if model_id != original_id:
        if SETTINGS.aws_bedrock_deprecated_model_fallback:
            detail = f"Model '{original_id}' is deprecated; replacement model '{model_id}' is also not found."
        else:
            detail = f"This model is deprecated or pending deprecation, please use '{model_id}' instead."
    model_ids = set(models)
    if input_modality:
        model_ids &= (
            _MODELS_INPUT_MODALITY if bedrock_only else _ALL_MODELS_INPUT_MODALITY
        ).get(input_modality, set())
    if output_modality:
        model_ids &= (
            _MODELS_OUTPUT_MODALITY if bedrock_only else _ALL_MODELS_OUTPUT_MODALITY
        ).get(output_modality, set())
    raise UnsupportedModelError(
        original_id, available_models=model_ids, detail=detail, status=error_status
    ) from None


async def validate_model(
    model_id: str,
    output_modality: str | None = None,
    input_modality: str | None = None,
    *,
    bedrock_only: bool = True,
    error_status: int | None = None,
) -> ModelDetails:
    """Validate *model_id* and return its ``ModelDetails``.

    Resolves aliases and ARNs, refreshes the cache on a miss, checks modality support,
    and records the model ID in the request log. An alias carrying configuration
    applies it to the rest of the request.

    If the model is not found and is listed in :data:`~stdapi.models.deprecation.DEPRECATED_MODELS`,
    and :attr:`~stdapi.config.Settings.aws_bedrock_deprecated_model_fallback` is enabled,
    the lookup is transparently retried with the replacement model ID.

    Args:
        model_id: Model ID, alias, or ARN to validate.
        output_modality: Required output modality (e.g. ``"TEXT"``).
        input_modality: Required input modality (e.g. ``"IMAGE"``).
        bedrock_only: Restrict lookup to Bedrock models (default ``True``).
        error_status: HTTP status code override for ``UnsupportedModelError``.

    Returns:
        Validated :class:`ModelDetails`.

    Raises:
        UnsupportedModelError: If the model is not found.
        ApiError: If the model does not support the requested modality.
    """
    if MODEL_ALIAS_OVERLAYS:
        apply_alias_overlay(MODEL_ALIAS_OVERLAYS.get(model_id))
    model_id = resolve_model_alias(model_id)
    original_id = model_id
    models = _MODELS if bedrock_only else _ALL_MODELS
    if model_id.startswith("arn:"):
        model = await _validate_model_from_arn(model_id)
    else:
        async with _CACHE["access_lock"]:
            model = models.get(model_id)

    if model is None:
        await initialize_bedrock_models()
        async with _CACHE["access_lock"]:
            model = models.get(model_id)
            if model is None:
                fallback_model, model_id = _resolve_deprecated(models, model_id)
                if SETTINGS.aws_bedrock_deprecated_model_fallback:
                    model = fallback_model
        if model is None:
            _raise_model_not_found(
                models,
                original_id,
                model_id,
                input_modality,
                output_modality,
                error_status,
                bedrock_only=bedrock_only,
            )

    if output_modality and output_modality not in model.output_modalities:
        msg = f"Model '{model_id}' does not support {output_modality.lower()} output modality."
        raise ApiError(msg)
    if input_modality and input_modality not in model.input_modalities:
        msg = f"Model '{model_id}' does not support {input_modality.lower()} input modality."
        raise ApiError(msg)
    log = REQUEST_LOG.get()
    log["model_id"] = model_id
    _warn_model_lifecycle(model, original_id, model_id)
    return model


async def _validate_model_from_arn(arn: str) -> ModelDetails | None:
    """Resolve an ARN to a ``ModelDetails`` instance, using a TTL cache.

    Supports application inference profiles, cross-region inference profiles, and
    prompt router ARNs. Caches validated results for one model cache interval.

    Args:
        arn: Bedrock ARN of an inference profile or prompt router.

    Returns:
        Resolved :class:`ModelDetails` with ``inference_profile`` set to *arn*.

    Raises:
        ApiError: If *arn* does not match a valid inference profile or prompt router,
            or if the ARN type is disabled by server configuration.
    """
    models: (
        Sequence[InferenceProfileModelTypeDef]
        | Sequence[PromptRouterTargetModelTypeDef]
        | None
    ) = None
    async with _CACHE["user_profiles_access_lock"]:
        if cached := _USER_PROFILES.get(arn):
            model, expiration = cached
            if expiration > SETTINGS.now():
                return model
            del _USER_PROFILES[arn]

        try:
            models, region = (
                await _get_application_inference_profile_models(arn)
                or await _get_prompt_router_models(arn)
                or (None, None)
            )
        except ClientError as error:
            if (
                error.response["Error"]["Code"] != "ResourceNotFoundException"
            ):  # pragma: no cover
                raise

        model_arn = None
        while True:
            if not models or not region:
                msg = f"ARN does not match a valid inference profile or prompt router: {model_arn or arn}"
                raise ApiError(msg)

            model_arn = models[0]["modelArn"]
            if "inference-profile" in model_arn:
                models = (
                    await get_client("bedrock", region).get_inference_profile(
                        inferenceProfileIdentifier=model_arn
                    )
                ).get("models") or ()
                continue

            model_id = model_arn.rsplit("/", 1)[1]
            break

        async with _CACHE["access_lock"]:
            try:
                base_model = _MODELS[model_id].model_copy()
            except KeyError:
                msg = f"model {model_id} not found for ARN: {arn}"
                raise ApiError(msg) from None
        model = base_model.model_copy()
        if region not in model.regions:
            model.regions.append(region)
        model.set_inference_profile(region, arn)
        _USER_PROFILES[arn] = (model, SETTINGS.now() + _CACHE["update_interval"])
        return model


async def _get_prompt_router_models(
    arn: str,
) -> tuple[Sequence[PromptRouterTargetModelTypeDef], RegionName] | None:
    """Return target models and region for a prompt-router ARN, or ``None`` if *arn* doesn't match.

    Args:
        arn: Bedrock ARN to test.

    Returns:
        ``(models, region)`` when *arn* is a prompt router, ``None`` otherwise.

    Raises:
        ApiError: If prompt routers are disabled by server configuration.
    """
    if result := match_bedrock_prompt_router_arn(arn):
        if not SETTINGS.aws_bedrock_allow_prompt_router_arn:
            msg = "Prompt router are not allowed by server configuration."
            raise ApiError(msg)
        region: RegionName = result.group("region")  # type: ignore[assignment]
        validate_bedrock_region(region)
        return (
            await get_client("bedrock", region).get_prompt_router(promptRouterArn=arn)
        ).get("models") or (), region
    return None


async def _get_application_inference_profile_models(
    arn: str,
) -> tuple[Sequence[InferenceProfileModelTypeDef], RegionName] | None:
    """Return member models and region for an inference-profile ARN, or ``None`` if *arn* doesn't match.

    Handles both application inference profiles and cross-region inference profiles.

    Args:
        arn: Bedrock ARN to test.

    Returns:
        ``(models, region)`` when *arn* is an inference profile, ``None`` otherwise.

    Raises:
        ApiError: If the profile type is disabled by server configuration.
    """
    if result := match_bedrock_app_profile_arn(arn):
        if "application-inference-profile" in arn:
            if not SETTINGS.aws_bedrock_allow_application_inference_profile_arn:
                msg = "Application inference profile are not allowed by server configuration."
                raise ApiError(msg)
        elif not SETTINGS.aws_bedrock_allow_cross_region_inference_profile_arn:
            msg = "Cross-region inference profile are not allowed by server configuration."
            raise ApiError(msg)
        region: RegionName = result.group("region")  # type: ignore[assignment]
        validate_bedrock_region(region)
        return (
            await get_client("bedrock", region).get_inference_profile(
                inferenceProfileIdentifier=arn
            )
        ).get("models") or (), region
    return None


async def _get_prompt_model_id(
    arn: str, version: str | None, region: RegionName
) -> str:
    """Return the model ID configured on a Prompt Management prompt, using a TTL cache.

    Args:
        arn: Prompt ARN without version suffix.
        version: Prompt version to read, or ``None`` for the working draft.
        region: AWS region owning the prompt.

    Returns:
        Bedrock model ID configured on the prompt variant.

    Raises:
        ApiError: If the prompt has no TEXT variant bound to a model.
    """
    cache_key = f"{arn}:{version}" if version else arn
    async with _CACHE["prompts_access_lock"]:
        if cached := _PROMPTS.get(cache_key):
            model_id, expiration = cached
            if expiration > SETTINGS.now():
                return model_id
            del _PROMPTS[cache_key]

        kwargs = {"promptVersion": version} if version else {}
        with handle_bedrock_client_error():
            prompt = await get_client("bedrock-agent", region).get_prompt(
                promptIdentifier=arn, **kwargs
            )
        # PromptVariantList is capped at one entry, so defaultVariant is redundant.
        variant = (prompt.get("variants") or (None,))[0]
        if variant is None or variant.get("templateType") != "TEXT":
            msg = f"Prompt '{cache_key}' is not a TEXT prompt: only TEXT prompts are supported."
            raise ApiError(msg)
        variant_model_id: str = variant.get("modelId") or ""
        if not variant_model_id:
            msg = f"Prompt '{cache_key}' is not bound to a model."
            raise ApiError(msg)
        # A prompt may name an inference profile instead of the model itself.
        variant_model_id = _catalog_model_id(variant_model_id) or variant_model_id
        _PROMPTS[cache_key] = (
            variant_model_id,
            SETTINGS.now() + _CACHE["update_interval"],
        )
        return variant_model_id


async def resolve_bedrock_prompt(prompt_id: str, version: str | None) -> BedrockPrompt:
    """Resolve an OpenAI Responses ``prompt`` reference to a Bedrock prompt resource.

    Args:
        prompt_id: Value of the request's ``prompt.id`` field.
        version: Value of the request's ``prompt.version`` field, if any.

    Returns:
        The resolved prompt, with the model ID that serves it.

    Raises:
        ApiError: If prompts are disabled by server configuration, *prompt_id* is
            not a Bedrock prompt ARN, *version* is invalid or conflicts with the
            ARN suffix, or the prompt's model cannot be served.
    """
    if not SETTINGS.aws_bedrock_allow_prompt_arn:
        msg = "Prompt templates are not allowed by server configuration."
        raise ApiError(msg)
    result = match_bedrock_prompt_arn(prompt_id)
    if result is None:
        msg = (
            f"'{prompt_id}' is not an Amazon Bedrock Prompt Management prompt ARN: "
            "hosted prompt templates do not exist on this server."
        )
        raise ApiError(msg)
    arn_version = result.group("version")
    if version is not None:
        if not (version.isascii() and version.isdigit()) or len(version) > 5:
            msg = f"Invalid prompt version '{version}': Amazon Bedrock prompt versions are numbers."
            raise ApiError(msg)
        if arn_version is not None and arn_version != version:
            msg = f"Prompt version '{version}' conflicts with the version in the prompt ARN ('{arn_version}')."
            raise ApiError(msg)
    else:
        version = arn_version
    base_arn: str = result.group("base")
    region: RegionName = result.group("region")  # type: ignore[assignment]
    validate_bedrock_region(region)
    model = await validate_model(
        await _get_prompt_model_id(base_arn, version, region),
        input_modality="TEXT",
        output_modality="TEXT",
    )
    return BedrockPrompt(
        arn=f"{base_arn}:{version}" if version else base_arn,
        region=region,
        model_id=model.id,
    )


#: Initial delay between ``GetAsyncInvoke`` polls.
_ASYNC_INVOKE_POLL_INITIAL_DELAY: Final[float] = 0.5

#: Multiplier applied to the poll delay after each unfinished check.
_ASYNC_INVOKE_POLL_BACKOFF: Final[float] = 2.0

#: Maximum delay between ``GetAsyncInvoke`` polls.
_ASYNC_INVOKE_POLL_MAX_DELAY: Final[float] = 2.0


async def _wait_for_async_invocation_completion(
    bedrock_client: BedrockRuntimeClient, invocation_arn: str
) -> str:
    """Poll ``GetAsyncInvoke`` until the job completes and return the S3 output key.

    Poll delay backs off exponentially from :data:`_ASYNC_INVOKE_POLL_INITIAL_DELAY`
    to :data:`_ASYNC_INVOKE_POLL_MAX_DELAY`, easing load for jobs that take longer
    than a couple seconds to complete.

    Args:
        bedrock_client: Bedrock runtime client for the job's region.
        invocation_arn: ARN returned by ``StartAsyncInvoke``.

    Returns:
        S3 key prefix of the output (bucket stripped from the ``s3Uri``).

    Raises:
        ApiError: If the invocation status is ``Failed``.
    """
    delay = _ASYNC_INVOKE_POLL_INITIAL_DELAY
    while True:
        response = await bedrock_client.get_async_invoke(invocationArn=invocation_arn)
        match response["status"]:
            case "Completed":
                return (
                    response["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
                    .removeprefix("s3://")
                    .split("/", 1)[1]
                )
            case "Failed":
                log_error_details(response["failureMessage"], status=502)
                msg = "The request could not be completed. Retry the request."
                raise ApiError(msg, status=502)
        await sleep(delay)
        delay = min(delay * _ASYNC_INVOKE_POLL_BACKOFF, _ASYNC_INVOKE_POLL_MAX_DELAY)
