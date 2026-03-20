"""Models."""

from asyncio import Lock, gather, sleep
from datetime import timedelta
from functools import cached_property
from importlib import import_module
from pkgutil import iter_modules
from re import Pattern
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, TypeVar

from botocore.exceptions import ClientError
from botocore.exceptions import ConnectionError as BotocoreConnectionError
from pydantic import AwareDatetime, BaseModel, JsonValue
from pydantic_core import from_json, to_json

import stdapi.region_routing as _region_routing
from stdapi.api_errors import ApiError, UnsupportedModelError
from stdapi.aws import get_client
from stdapi.aws_bedrock import (
    GUARDTRAIL_CONFIG_VAR,
    PERFORMANCE_CONFIG_VAR,
    bedrock_client,
    check_stream_event,
    handle_bedrock_client_error,
)
from stdapi.aws_s3 import (
    get_s3_bucket_for_region,
    require_s3_bucket_for_region,
    track_temporary_s3_objects,
)
from stdapi.config import SETTINGS
from stdapi.input_file import get_s3_input_regions, resolve_all_bedrock_content_blocks
from stdapi.models.deprecation import DEPRECATED_MODELS
from stdapi.monitoring import REQUEST_ID, REQUEST_LOG, log_error_details
from stdapi.region_routing import REGION_ROUTER, ROUTING_RETRYABLE_CODES
from stdapi.utils import match_bedrock_app_profile_arn, match_bedrock_prompt_router_arn

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        Awaitable,
        Callable,
        Iterable,
        Mapping,
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
    from types_aiobotocore_bedrock_runtime.literals import TraceType
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ConverseResponseTypeDef,
        ConverseStreamResponseTypeDef,
        InvokeModelRequestTypeDef,
        ResponseStreamTypeDef,
    )

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef

    class _ModelCache(TypedDict):
        """Model cache configuration."""

        update_next: AwareDatetime | None
        update_interval: timedelta
        update_lock: Lock
        access_lock: Lock
        user_profiles_access_lock: Lock

else:
    type RegionName = str

#: Bedrock models details
_MODELS: dict[str, ModelDetails] = {}

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

#: Model cache state
_CACHE: _ModelCache = {
    "update_next": None,
    "update_lock": Lock(),
    "update_interval": timedelta(seconds=SETTINGS.model_cache_seconds),
    "access_lock": Lock(),
    "user_profiles_access_lock": Lock(),
}

#: Always-allowed inference types
_INFERENCE_TYPES = {"INFERENCE_PROFILE", "ON_DEMAND"}

#: TTL cache for application inference profiles and prompt routers
_USER_PROFILES: dict[str, tuple[ModelDetails, AwareDatetime]] = {}

#: Model aliases (populated on import, merged with user settings at startup)
MODEL_ALIASES: dict[str, str] = {}

#: Registered model classes for all model families
_GLOBAL_MODEL_REGISTRY: set[type[ModelBase[Any, Any]]] = set()

#: Fallback model class per package (populated by load_model_plugins)
_DEFAULT: dict[str, type[ModelBase[Any, Any]]] = {}


class ModelDetails(BaseModel):
    """Model details and features."""

    id: str
    name: str
    provider: str
    region: RegionName
    service: str = "AWS Bedrock"
    input_modalities: list[str]
    output_modalities: list[str]
    response_streaming: bool = False
    legacy: bool = False
    start_of_life_time: AwareDatetime | None = None
    end_of_life_time: AwareDatetime | None = None
    legacy_time: AwareDatetime | None = None
    public_extended_access_time: AwareDatetime | None = None
    inference_profile: str | None = None
    aliases: list[str] | None = None
    available_regions: list[RegionName] = []
    inference_profiles_by_region: dict[RegionName, str] = {}

    def get_id(self, *, inference_profile: bool = False) -> str:
        """Return the model ID, preferring the inference profile when requested.

        Args:
            inference_profile: When ``True``, return ``self.inference_profile`` if set.

        Returns:
            Model or inference-profile identifier.
        """
        return (self.inference_profile or self.id) if inference_profile else self.id

    def get_id_for_region(
        self, region: RegionName, *, inference_profile: bool = False
    ) -> str:
        """Return the model ID or inference profile ARN for a specific region.

        Args:
            region: Target AWS region.
            inference_profile: If True, prefer the inference profile for that region.

        Returns:
            The appropriate model identifier for the given region.
        """
        if inference_profile:
            if profile := self.inference_profiles_by_region.get(region):
                return profile
            return self.inference_profile or self.id
        return self.id


RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class ModelBase[RequestT, ResponseT]:
    """Base class for provider-specific models."""

    __slots__ = ("_model_id",)

    #: Model ID matcher, regex pattern or string prefix
    MATCHER: ClassVar[str | Pattern[str]] = ""

    #: Maps HTTP header name (lowercase) to a (field_key, transform) tuple.
    #: The transform callable converts the raw header string to the expected value type.
    PASSTHROUGH_HEADERS: ClassVar[
        MappingProxyType[str, tuple[str, Callable[[str], Any]]]
    ] = MappingProxyType({})

    #: Regex to extract model alias from model ID
    ALIAS_MATCHER: ClassVar[Pattern[str] | None] = None

    def __init__(self, model_id: str) -> None:
        """Initialize the model with its Bedrock model identifier.

        Args:
            model_id: The AWS Bedrock model identifier.
        """
        self._model_id = model_id

    @classmethod
    def get_aliases(cls, all_models: dict[str, ModelDetails]) -> dict[str, str]:
        """Return API model name aliases mapped to model IDs.

        Args:
            all_models: All available models keyed by Bedrock model ID.

        Returns:
            A dict mapping model alias to model ID.
        """
        return (
            {
                match.group(1): model_id
                for model_id in all_models
                if (match := cls.ALIAS_MATCHER.match(model_id))
            }
            if cls.ALIAS_MATCHER
            else {}
        )

    @cached_property
    def model(self) -> ModelDetails:
        """Model details for this instance.

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
        ``region=`` to lock the retry loop to that region.

        When no S3 is involved, skip this call entirely and let :meth:`invoke`
        handle region selection and multi-region retry on its own.

        Args:
            s3_required: When ``True``, only regions with a configured S3
                bucket are considered as candidates.

        Returns:
            AWS region string.
        """
        candidates = await _compute_candidate_regions(
            self._model_id, s3_required=s3_required
        )
        return (
            REGION_ROUTER.ordered_regions(self._model_id, candidates)
            if REGION_ROUTER
            else candidates
        )[0]

    async def invoke(
        self,
        body: RequestT,
        *,
        inference_profile: bool = True,
        region: RegionName | None = None,
        s3_required: bool = False,
    ) -> ResponseT:
        """Invoke the model via ``InvokeModel``.

        Args:
            body: JSON request payload.
            inference_profile: Use the cross-region inference profile ID when available.
            region: Pin the retry loop to this region. Use with :meth:`select_region`
                when S3 inputs have already been placed in a specific region. When
                ``None``, the router selects freely across all candidate regions.
            s3_required: Restrict candidate regions to those with a configured S3
                bucket. Ignored when *region* is provided.

        Returns:
            Parsed JSON response body.
        """
        candidates = await _compute_candidate_regions(
            self._model_id, region=region, s3_required=s3_required
        )
        return await _route_and_execute(
            self._model_id,
            candidates,
            lambda r: _invoke(  # type: ignore[return-value,arg-type]
                self._model_id,
                body,  # type: ignore[arg-type]
                r,
                inference_profile=inference_profile,
                single_region=len(candidates) == 1,
            ),
        )

    async def batch_invoke(
        self, bodies: Iterable[RequestT], *, inference_profile: bool = True
    ) -> list[ResponseT]:
        """Invoke the model concurrently for multiple request bodies.

        Args:
            bodies: Iterable of JSON request payloads.
            inference_profile: Use the cross-region inference profile ID when available.

        Returns:
            List of parsed JSON response bodies, in the same order as *bodies*.
        """
        return await gather(
            *(self.invoke(body, inference_profile=inference_profile) for body in bodies)
        )

    async def invoke_stream(
        self,
        body: RequestT,
        *,
        inference_profile: bool = True,
        region: RegionName | None = None,
        s3_required: bool = False,
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

        Yields:
            Parsed JSON chunks from the streaming response.
        """
        candidates = await _compute_candidate_regions(
            self._model_id, region=region, s3_required=s3_required
        )
        async for chunk in await _route_and_execute(
            self._model_id,
            candidates,
            lambda r: _open_invoke_stream(
                self._model_id,
                body,  # type: ignore[arg-type]
                r,
                inference_profile=inference_profile,
                single_region=len(candidates) == 1,
            ),
        ):
            yield chunk

    async def invoke_async(
        self,
        body: RequestT,
        *,
        inference_profile: bool = True,
        output_file: str = "output.json",
    ) -> ResponseT:
        """Invoke the model via ``StartAsyncInvoke`` and wait for the result.

        Args:
            body: JSON request payload.
            inference_profile: Use the cross-region inference profile ID when available.
            output_file: Output file name to retrieve from S3.

        Returns:
            Parsed JSON response body.

        Raises:
            ApiError: When invocation fails or results cannot be retrieved.
        """
        candidates = await _compute_candidate_regions(self._model_id, s3_required=True)
        effective_region = (
            REGION_ROUTER.ordered_regions(self._model_id, candidates)
            if REGION_ROUTER
            else candidates
        )[0]
        _set_effective_region(effective_region)

        s3_bucket_name = require_s3_bucket_for_region(effective_region)
        bedrock: BedrockRuntimeClient = get_client("bedrock-runtime", effective_region)
        with handle_bedrock_client_error():
            invocation_arn = (
                await bedrock.start_async_invoke(
                    modelId=(await get_model_details(self._model_id)).get_id_for_region(
                        effective_region, inference_profile=inference_profile
                    ),
                    modelInput=body,  # type: ignore[arg-type]
                    outputDataConfig={
                        "s3OutputDataConfig": {
                            "s3Uri": f"s3://{s3_bucket_name}/{SETTINGS.aws_s3_tmp_prefix}{REQUEST_ID.get()}/"
                        }
                    },
                )
            )["invocationArn"]

        # Region locked from here — poll and retrieve in the same region
        s3_key = await _wait_for_async_invocation_completion(bedrock, invocation_arn)
        s3_output_path = f"{s3_key}/{output_file}"
        track_temporary_s3_objects(
            s3_bucket_name, s3_output_path, f"{s3_key}/manifest.json"
        )
        return from_json(  # type: ignore[no-any-return]
            await (
                await get_client("s3", effective_region).get_object(
                    Bucket=s3_bucket_name, Key=s3_output_path
                )
            )["Body"].read()
        )

    async def converse(
        self, request: ConverseRequestBaseTypeDef
    ) -> ConverseResponseTypeDef:
        """Invoke the model via the Bedrock Converse API.

        Args:
            request: Bedrock Converse request payload (without ``modelId``).

        Returns:
            Bedrock Converse response.
        """
        candidates = await _compute_candidate_regions(self._model_id)
        return await _route_and_execute(
            self._model_id,
            candidates,
            lambda r: _converse(
                self._model_id, request, r, single_region=len(candidates) == 1
            ),
        )

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
        candidates = await _compute_candidate_regions(self._model_id)
        return await _route_and_execute(
            self._model_id,
            candidates,
            lambda r: _converse_stream(
                self._model_id, request, r, single_region=len(candidates) == 1
            ),
        )


ModelT = TypeVar("ModelT", bound=ModelBase[Any, Any])


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


def resolve_model_alias(model_id: str) -> str:
    """Resolve a model alias to its canonical model ID, or return *model_id* unchanged.

    Args:
        model_id: Model ID or alias.

    Returns:
        Canonical model ID.
    """
    return MODEL_ALIASES.get(model_id, model_id)


def update_unified_models_collections() -> None:
    """Merge Bedrock and extra-service model collections into the unified ``_ALL_*`` dicts.

    Rebuilds ``_ALL_MODELS``, ``_ALL_MODELS_OUTPUT_MODALITY``,
    ``_ALL_MODELS_INPUT_MODALITY``, and the model-alias index.
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


async def _get_inference_profiles(bedrock_client: BedrockClient) -> dict[str, str]:
    """Return a mapping of model ID → inference profile ID for this region.

    Fetches active system-defined cross-region inference profiles when
    ``aws_bedrock_cross_region_inference`` is enabled.

    Args:
        bedrock_client: Bedrock control-plane client for the region.

    Returns:
        Dict mapping model ID to its preferred inference profile ID.
    """
    result: dict[str, str] = {}
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
        _filter_inference_profiles(result, profiles_all)
    return result


def _filter_inference_profiles(
    profiles: dict[str, str], profiles_all: dict[str, list[str]]
) -> None:
    """Populate *profiles* with the best inference profile per model.

    Prefers the ``global.`` prefix profile when ``aws_bedrock_cross_region_inference_global``
    is enabled; otherwise picks the first non-global candidate.

    Args:
        profiles: Output dict (model ID → profile ID) updated in-place.
        profiles_all: All discovered profile IDs per model ID.
    """
    for model_id, profile_ids in profiles_all.items():
        candidate_profile = ""
        for profile_id in profile_ids:
            if profile_id.startswith("global."):
                if SETTINGS.aws_bedrock_cross_region_inference_global:
                    profiles[model_id] = profile_id
                    break
                continue
            candidate_profile = profile_id
        else:
            if candidate_profile:
                profiles[model_id] = candidate_profile


async def _get_bedrock_models_from_region(region: RegionName) -> list[ModelDetails]:
    """Fetch available foundation models from *region* and return filtered ``ModelDetails``.

    Models restricted via ``aws_bedrock_model_region_restrict`` to regions that exclude
    *region* are dropped immediately. Models whose ``end_of_life_time`` falls before the
    next scheduled cache refresh are also excluded, so a model that goes EOL between two
    cache updates is proactively dropped rather than served until the next refresh.

    Args:
        region: AWS region to query.
    """
    bedrock_client: BedrockClient = get_client("bedrock", region)

    foundation_models, provisioned_models, profiles = await gather(
        bedrock_client.list_foundation_models(),
        _get_provisioned_models(bedrock_client),
        _get_inference_profiles(bedrock_client),
    )
    restrictions = SETTINGS.aws_bedrock_model_region_restrict
    next_refresh = SETTINGS.now() + _CACHE["update_interval"]
    return [
        ModelDetails(
            id=model["modelId"],
            name=model["modelName"],
            provider=model["providerName"],
            region=region,
            input_modalities=model["inputModalities"],  # type: ignore[arg-type]
            output_modalities=model["outputModalities"],  # type: ignore[arg-type]
            response_streaming=model.get("responseStreamingSupported", False),
            inference_profile=profiles.get(model["modelId"]),
            legacy=model["modelLifecycle"]["status"] == "LEGACY",
            start_of_life_time=model["modelLifecycle"].get("startOfLifeTime"),
            end_of_life_time=model["modelLifecycle"].get("endOfLifeTime"),
            legacy_time=model["modelLifecycle"].get("legacyTime"),
            public_extended_access_time=model["modelLifecycle"].get(
                "publicExtendedAccessTime"
            ),
        )
        for model in foundation_models["modelSummaries"]
        if (
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
            (
                allowed := restrictions.get(model["modelId"])
                or next(
                    (
                        v
                        for k, v in restrictions.items()
                        if model["modelId"].startswith(k)
                    ),
                    None,
                )
            )
            is None
            or region in allowed
        )
    ]


async def initialize_bedrock_models() -> tuple[
    bool, dict[str, dict[str, list[str]]], dict[str, str], set[str]
]:
    """Refresh the Bedrock model cache from all configured regions if stale.

    Returns:
        Tuple of (updated, unavailable_models, invalid_arn_mappings, unmatched_restrict_keys) where
        *updated* is ``True`` if the cache was refreshed, *unavailable_models* maps
        model IDs to per-region availability issues, *invalid_arn_mappings* maps
        model IDs to ARN-mapping error messages, and *unmatched_restrict_keys* is the
        set of ``aws_bedrock_model_region_restrict`` keys that did not match any
        available model.
    """
    updated = False
    unavailable_models: dict[str, dict[str, list[str]]] = {}

    async with _CACHE["update_lock"]:
        if _CACHE["update_next"] is None or _CACHE["update_next"] <= SETTINGS.now():
            region_models = await gather(
                *(
                    _get_bedrock_models_from_region(region)
                    for region in _region_routing.ORDERED_BEDROCK_REGIONS
                )
            )
            all_models: dict[str, ModelDetails] = {}
            for region, models in zip(
                _region_routing.ORDERED_BEDROCK_REGIONS, region_models, strict=False
            ):
                bedrock_client: BedrockClient = get_client("bedrock", region)
                await gather(
                    *(
                        _filter_model(
                            bedrock_client, model, all_models, unavailable_models
                        )
                        for model in models
                    )
                )

            invalid_arn_mappings = _apply_user_profiles(all_models)

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
                if updated and _CACHE["update_next"] is not None:
                    update_unified_models_collections()
            _CACHE["update_next"] = SETTINGS.now() + _CACHE["update_interval"]
        else:
            invalid_arn_mappings = {}
            unmatched_restrict_keys = set()
    return updated, unavailable_models, invalid_arn_mappings, unmatched_restrict_keys


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
        model.inference_profile = arn
        if arn_match := (
            match_bedrock_app_profile_arn(arn) or match_bedrock_prompt_router_arn(arn)
        ):
            model.region = arn_match.group("region")  # type: ignore[assignment]
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


def _populate_model_aliases(all_models: dict[str, ModelDetails]) -> None:
    """Rebuild ``MODEL_ALIASES`` from all registered model classes and user settings.

    Also sets ``ModelDetails.aliases`` for each model that has aliases.

    Args:
        all_models: All available models keyed by model ID.
    """
    MODEL_ALIASES.clear()
    for cls in _GLOBAL_MODEL_REGISTRY:
        MODEL_ALIASES.update(cls.get_aliases(all_models))
    MODEL_ALIASES.update(SETTINGS.model_aliases)

    aliases_by_model: dict[str, set[str]] = {}
    for alias, model_id in MODEL_ALIASES.items():
        if model_id in all_models:
            aliases_by_model.setdefault(model_id, set()).add(alias)

    for model_id, aliases in aliases_by_model.items():
        all_models[model_id].aliases = sorted(aliases)


async def _filter_model(
    bedrock_client: BedrockClient,
    model: ModelDetails,
    models: dict[str, ModelDetails],
    unavailable_models: dict[str, dict[str, list[str]]],
) -> None:
    """Check *model* availability and add it to *models*, or append region data if already known.

    When *model* is already present from another region, its ``available_regions`` and
    ``inference_profiles_by_region`` are extended rather than creating a duplicate entry.

    Args:
        bedrock_client: Bedrock control-plane client for the model's region.
        model: Candidate model from ``_get_bedrock_models_from_region``.
        models: Accumulator dict updated in-place.
        unavailable_models: Accumulator for models that fail the availability check.
    """
    if model.id in models:
        existing = models[model.id]
        if model.region not in existing.available_regions:
            existing.available_regions.append(model.region)
            if model.inference_profile:
                existing.inference_profiles_by_region[model.region] = (
                    model.inference_profile
                )
        return

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
        model.available_regions = [model.region]
        if model.inference_profile:
            model.inference_profiles_by_region = {model.region: model.inference_profile}
        models[model.id] = model
    else:
        unavailable_models.setdefault(model.id, {})[model.region] = [
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


async def _compute_candidate_regions(
    model_id: str, *, region: RegionName | None = None, s3_required: bool = False
) -> list[RegionName]:
    """Return ordered candidate regions for routing, taking S3 input locality into account.

    The priority rules are:

    1. **S3 inputs present** — regions are ranked by descending total S3 input
       data volume.  Only the single best region is returned so that the retry
       loop stays pinned to it.  S3 content blocks are resolved as a terminal
       operation (the ``s3Location`` URI is written into the request body and
       ``_bedrock_source`` is deleted); retrying on a different region would
       send a cross-region S3 reference that Bedrock cannot access.
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
    available = model.available_regions or [model.region]

    if s3_input_regions := get_s3_input_regions():
        if priority := [
            r
            for r in sorted(
                s3_input_regions, key=s3_input_regions.__getitem__, reverse=True
            )
            if r in available
        ]:
            # Single region only — S3 content blocks are terminal and cannot be re-resolved.
            return priority[:1]
        if bucketed := [
            r for r in available if get_s3_bucket_for_region(r) is not None
        ]:
            # No overlap: fall back to the first model region that has a bucket so the
            # S3 object can be copied there before invocation.
            return bucketed[:1]
        msg = (
            f"S3 input data is located in {sorted(s3_input_regions)} but model "
            f"'{model_id}' is only available in {sorted(available)}."
        )
        raise ApiError(msg)

    if s3_required:
        if s3_capable := [
            r for r in available if get_s3_bucket_for_region(r) is not None
        ]:
            return s3_capable
        msg = (
            f"Model '{model_id}' requires an S3 bucket but none is configured for its "
            f"available regions {sorted(available)}."
        )
        raise ApiError(msg)

    return available


async def _route_and_execute[T](
    model_id: str,
    candidates: list[RegionName],
    fn: Callable[[RegionName], Awaitable[T]],
) -> T:
    """Route ``fn`` across candidate regions with retry and health bookkeeping.

    With a single candidate (region lock or S3-constrained), delegates directly to
    botocore's adaptive retries within that region. With routing disabled, uses
    the first candidate. Otherwise cycles through router-ordered regions up to
    ``SETTINGS.aws_bedrock_max_retries + 1`` attempts, escalating to the next region
    on any retryable error and marking the region healthy on success.

    Args:
        model_id: Used for router health bookkeeping.
        candidates: From ``_compute_candidate_regions``. Single-element means region-locked.
        fn: Must call ``_set_effective_region`` before the AWS call.

    Returns:
        Result of the first successful ``fn`` call.

    Raises:
        ClientError: Last non-retryable error, or last retryable error when all attempts exhausted.
        ConnectionError: Last connection error when all attempts exhausted.
    """
    if not REGION_ROUTER or len(candidates) == 1:
        return await fn(candidates[0])

    for _ in range(SETTINGS.aws_bedrock_max_retries + 1):
        region = REGION_ROUTER.ordered_regions(model_id, candidates)[0]
        try:
            result = await fn(region)
        except ClientError as exc:
            if (code := exc.response["Error"]["Code"]) not in ROUTING_RETRYABLE_CODES:
                raise
            last_exc: ClientError | BotocoreConnectionError = exc
            REGION_ROUTER.mark_error(model_id, region, code)
        except BotocoreConnectionError as exc:
            last_exc = exc
            REGION_ROUTER.mark_error(model_id, region, exc.__class__.__name__)
        else:
            REGION_ROUTER.mark_success(model_id, region)
            return result

    raise last_exc


def _set_effective_region(region: RegionName) -> None:
    """Record *region* in the request log's ``model_regions`` set.

    Args:
        region: AWS region selected for this request.
    """
    log = REQUEST_LOG.get()
    if "model_regions" not in log:
        log["model_regions"] = set()
    log["model_regions"].add(region)


async def _build_invoke_kwargs(
    model_id: str,
    body: Mapping[str, Any],
    region: RegionName,
    *,
    inference_profile: bool,
) -> InvokeModelRequestTypeDef:
    """Build the kwargs dict for ``InvokeModel`` / ``InvokeModelWithResponseStream``.

    Folds in guardrail and performance context-var values when present.

    Args:
        model_id: Bedrock model identifier.
        body: JSON request payload.
        region: Target AWS region (used to resolve the model/profile ID).
        inference_profile: Use cross-region inference profile ID when available.

    Returns:
        Fully populated request kwargs dict.
    """
    kwargs: InvokeModelRequestTypeDef = {
        "modelId": (await get_model_details(model_id)).get_id_for_region(
            region, inference_profile=inference_profile
        ),
        "contentType": "application/json",
        "accept": "application/json",
        "body": to_json(body),
    }
    try:
        guardtrail_config = GUARDTRAIL_CONFIG_VAR.get()
    except LookupError:
        pass
    else:
        kwargs["guardrailIdentifier"] = guardtrail_config["guardrailIdentifier"]
        kwargs["guardrailVersion"] = guardtrail_config["guardrailVersion"]
        try:
            # The format differs (Uppercase instead of lowercase)
            trace: TraceType = guardtrail_config["trace"].upper()  # type: ignore[assignment]
        except KeyError:
            pass
        else:
            kwargs["trace"] = trace
    latency, service_tier = PERFORMANCE_CONFIG_VAR.get()
    if latency:
        kwargs["performanceConfigLatency"] = latency
    if service_tier:
        kwargs["serviceTier"] = service_tier
    return kwargs


async def _invoke(
    model_id: str,
    body: Mapping[str, Any],
    region: RegionName,
    *,
    inference_profile: bool,
    single_region: bool,
) -> Mapping[str, Any]:
    """Call ``InvokeModel`` and return the parsed JSON response.

    Args:
        model_id: Bedrock model identifier.
        body: JSON request payload.
        region: AWS region to target.
        inference_profile: Use cross-region inference profile ID when available.
        single_region: Selects the botocore client (see :func:`bedrock_client`).

    Returns:
        Parsed JSON response body.
    """
    _set_effective_region(region)
    with handle_bedrock_client_error():
        response = await bedrock_client(
            region, single_region=single_region
        ).invoke_model(
            **(
                await _build_invoke_kwargs(
                    model_id, body, region, inference_profile=inference_profile
                )
            )
        )
    return from_json(await response["body"].read())  # type: ignore[no-any-return]


async def _iter_invoke_stream(
    body: AioEventStream[ResponseStreamTypeDef],
) -> AsyncGenerator[JsonValue]:
    """Iterate over an open ``InvokeModelWithResponseStream`` event stream.

    Args:
        body: Open event stream from ``invoke_model_with_response_stream``.

    Yields:
        Parsed JSON chunks.
    """
    async for event in body:
        if "chunk" in event:
            yield from_json(event["chunk"]["bytes"])
        else:
            check_stream_event(event)


async def _open_invoke_stream(
    model_id: str,
    body: Mapping[str, Any],
    region: RegionName,
    *,
    inference_profile: bool,
    single_region: bool,
) -> AsyncGenerator[JsonValue]:
    """Open an ``InvokeModelWithResponseStream`` connection and return a JSON chunk generator.

    The HTTP connection is established before the first ``yield`` so
    :func:`_route_and_execute` can retry on a different region if the open fails.
    Once the stream is open, retrying is no longer possible.

    Args:
        model_id: Bedrock model identifier.
        body: JSON request payload.
        region: AWS region to target.
        inference_profile: Use cross-region inference profile ID when available.
        single_region: Selects the botocore client (see :func:`bedrock_client`).

    Returns:
        Async generator yielding parsed JSON chunks.
    """
    _set_effective_region(region)
    with handle_bedrock_client_error():
        response = await bedrock_client(
            region, single_region=single_region
        ).invoke_model_with_response_stream(
            **(
                await _build_invoke_kwargs(
                    model_id, body, region, inference_profile=inference_profile
                )
            )
        )

    return _iter_invoke_stream(response["body"])


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
    and records the model ID in the request log.

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
    model_id = resolve_model_alias(model_id)
    original_id = model_id
    models = _MODELS if bedrock_only else _ALL_MODELS
    if model_id.startswith("arn:"):
        model = await _validate_model_from_arn(model_id)
    else:
        async with _CACHE["access_lock"]:
            try:
                model = models[model_id]
            except KeyError:
                model = None

    if model is None:
        await initialize_bedrock_models()
        async with _CACHE["access_lock"]:
            model = models.get(model_id)
            if model is None and SETTINGS.aws_bedrock_deprecated_model_fallback:
                model, model_id = _resolve_deprecated(models, model_id)

        if model is None:
            msg = (
                f"Model '{original_id}' is deprecated; replacement model '{model_id}' not found."
                if model_id != original_id
                else f"Model '{model_id}' not found. This model is deprecated or pending deprecation, please use '{hint}' instead."
                if (hint := DEPRECATED_MODELS.get(model_id))
                else f"Model '{model_id}' not found."
            )
            model_ids = set(models)
            if input_modality:
                model_ids &= (
                    _MODELS_INPUT_MODALITY
                    if bedrock_only
                    else _ALL_MODELS_INPUT_MODALITY
                )[input_modality]
            if output_modality:
                model_ids &= (
                    _MODELS_OUTPUT_MODALITY
                    if bedrock_only
                    else _ALL_MODELS_OUTPUT_MODALITY
                )[output_modality]
            raise UnsupportedModelError(
                msg, available_models=model_ids, status=error_status
            ) from None

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
        model.inference_profile = arn
        model.region = region
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
        return (
            await get_client("bedrock", region).get_inference_profile(
                inferenceProfileIdentifier=arn
            )
        ).get("models") or (), region
    return None


async def _wait_for_async_invocation_completion(
    bedrock_client: BedrockRuntimeClient, invocation_arn: str
) -> str:
    """Poll ``GetAsyncInvoke`` until the job completes and return the S3 output key.

    Args:
        bedrock_client: Bedrock runtime client for the job's region.
        invocation_arn: ARN returned by ``StartAsyncInvoke``.

    Returns:
        S3 key prefix of the output (bucket stripped from the ``s3Uri``).

    Raises:
        ApiError: If the invocation status is ``Failed``.
    """
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
                raise ApiError(response["failureMessage"])
        await sleep(0.5)


async def _converse(
    model_id: str,
    request: ConverseRequestBaseTypeDef,
    region: RegionName,
    *,
    single_region: bool,
) -> ConverseResponseTypeDef:
    """Call the Bedrock Converse API and return the response.

    Resolves pending S3 content blocks for *region* and injects ``modelId`` before calling.

    Args:
        model_id: Bedrock model identifier.
        request: Converse request payload (``modelId`` is injected here).
        region: AWS region to target.
        single_region: Selects the botocore client (see :func:`bedrock_client`).

    Returns:
        Bedrock Converse response.
    """
    _set_effective_region(region)
    await resolve_all_bedrock_content_blocks(region)
    request["modelId"] = (await get_model_details(model_id)).get_id_for_region(
        region, inference_profile=True
    )
    with handle_bedrock_client_error():
        return await bedrock_client(region, single_region=single_region).converse(
            **request
        )


async def _converse_stream(
    model_id: str,
    request: ConverseRequestBaseTypeDef,
    region: RegionName,
    *,
    single_region: bool,
) -> ConverseStreamResponseTypeDef:
    """Open the Bedrock ConverseStream API and return the event-stream response.

    Resolves pending S3 content blocks for *region* and injects ``modelId`` before calling.
    Once the stream is open, no further failover is possible.

    Args:
        model_id: Bedrock model identifier.
        request: Converse request payload (``modelId`` is injected here).
        region: AWS region to target.
        single_region: Selects the botocore client (see :func:`bedrock_client`).

    Returns:
        Bedrock ConverseStream response containing the event stream.
    """
    _set_effective_region(region)
    await resolve_all_bedrock_content_blocks(region)
    request["modelId"] = (await get_model_details(model_id)).get_id_for_region(
        region, inference_profile=True
    )
    with handle_bedrock_client_error():
        return await bedrock_client(
            region, single_region=single_region
        ).converse_stream(**request)
