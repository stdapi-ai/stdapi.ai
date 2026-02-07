"""Models."""

from asyncio import Lock, gather, sleep
from contextlib import suppress
from datetime import timedelta
from functools import cached_property
from importlib import import_module
from pkgutil import iter_modules
from re import Pattern
from secrets import token_hex
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypedDict, TypeVar

from botocore.exceptions import ClientError
from fastapi import BackgroundTasks, HTTPException
from pydantic import AwareDatetime, BaseModel
from pydantic_core import from_json, to_json

from stdapi.aws import get_client
from stdapi.aws_bedrock import (
    GUARDTRAIL_CONFIG_VAR,
    PERFORMANCE_CONFIG_VAR,
    PROMPT_CACHING_SUPPORTED,
    PROMPT_CACHING_TOOL_SUPPORTED,
    PromptCaching,
    handle_bedrock_client_error,
)
from stdapi.aws_s3 import aws_s3_cleanup
from stdapi.config import SETTINGS
from stdapi.models.deprecation import DEPRECATED_MODELS
from stdapi.monitoring import REQUEST_ID, REQUEST_LOG, log_error_details
from stdapi.openai_exceptions import OpenaiError, OpenaiUnsupportedModelError
from stdapi.utils import (
    b64decode_data_uri,
    get_base64_decoded_size,
    get_data_uri_type,
    match_bedrock_app_profile_arn,
    match_bedrock_prompt_router_arn,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from types_aiobotocore_bedrock.client import BedrockClient
    from types_aiobotocore_bedrock.type_defs import (
        InferenceProfileModelTypeDef,
        ListInferenceProfilesRequestTypeDef,
        ListProvisionedModelThroughputsRequestTypeDef,
        PromptRouterTargetModelTypeDef,
    )
    from types_aiobotocore_bedrock_runtime import BedrockRuntimeClient
    from types_aiobotocore_bedrock_runtime.literals import (
        ServiceTierTypeType,
        TraceType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        CachePointBlockTypeDef,
        InferenceConfigurationTypeDef,
        InvokeModelRequestTypeDef,
        MessageTypeDef,
        SystemContentBlockTypeDef,
        ToolConfigurationTypeDef,
    )
    from types_aiobotocore_s3.client import S3Client
    from types_aiobotocore_s3.type_defs import BlobTypeDef, PutObjectRequestTypeDef

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef

    class _ModelCache(TypedDict):
        """Model cache configuration."""

        update_next: AwareDatetime | None
        update_interval: timedelta
        update_lock: Lock
        access_lock: Lock
        user_profiles_access_lock: Lock


#: Models details
_MODELS: dict[str, ModelDetails] = {}

#: Non Bedrock models details
EXTRA_MODELS: dict[str, ModelDetails] = {}

#: All models
_ALL_MODELS: dict[str, ModelDetails] = {}

#: Models by output modality
_MODELS_OUTPUT_MODALITY: dict[str, set[str]] = {}

#: Non Bedrock models by output modality
EXTRA_MODELS_OUTPUT_MODALITY: dict[str, set[str]] = {}

#: All models by output modality
_ALL_MODELS_OUTPUT_MODALITY: dict[str, set[str]] = {}

#: Models by input modality
_MODELS_INPUT_MODALITY: dict[str, set[str]] = {}

#: Non Bedrock models by input modality
EXTRA_MODELS_INPUT_MODALITY: dict[str, set[str]] = {}

#: All models by input modality
_ALL_MODELS_INPUT_MODALITY: dict[str, set[str]] = {}

#: Model cache configuration
_CACHE: _ModelCache = {
    "update_next": None,
    "update_lock": Lock(),
    "update_interval": timedelta(seconds=SETTINGS.model_cache_seconds),
    "access_lock": Lock(),
    "user_profiles_access_lock": Lock(),
}

#: Always allowed inference types
_INFERENCE_TYPES = {"INFERENCE_PROFILE", "ON_DEMAND"}

#: Cached cache point for prompt caching
_CACHE_POINT: dict[Literal["cachePoint"], CachePointBlockTypeDef] = {
    "cachePoint": {"type": "default"}
}

#: Cached application inference profiles & prompt routers
_USER_PROFILES: dict[str, tuple[ModelDetails, AwareDatetime]] = {}

#: Model aliases (Populated by models implementation on import then merged at startup with user settings)
MODEL_ALIASES: dict[str, str] = {}


class ModelDetails(BaseModel):
    """Model details and features."""

    id: str
    name: str
    provider: str
    region: str
    service: str = "AWS Bedrock"
    input_modalities: list[str]
    output_modalities: list[str]
    response_streaming: bool = False
    legacy: bool = False
    inference_profile: str | None = None
    aliases: list[str] | None = None

    def get_id(self, *, inference_profile: bool = False) -> str:
        """Returns the identifier of the object based on the inference profile flag.

        If `inference_profile` is True, the method retrieves the identifier
        based on the active inference profile. Otherwise, it retrieves the
        standard identifier.

        Args:
            inference_profile: Indicates whether to use the inference
                profile identifier or the standard identifier.

        Returns:
            The identifier based on the provided inference profile flag.
        """
        return (self.inference_profile or self.id) if inference_profile else self.id


RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class ModelBase[RequestT, ResponseT]:
    """Base class for provider-specific models."""

    __slots__ = ("_model_id",)

    MATCHER: ClassVar[str | Pattern[str]] = ""

    def __init__(self, model_id: str) -> None:
        """Initialize the model with a specific model identifier.

        Args:
            model_id: The AWS Bedrock model identifier.
        """
        self._model_id = model_id

    @cached_property
    def model(self) -> ModelDetails:
        """Get the model details for this model.

        Returns:
            Model details including region, provider, and capabilities.

        Raises:
            KeyError: If the model is not found in the registry.
        """
        try:
            return _MODELS[self._model_id]
        except KeyError:
            return EXTRA_MODELS[self._model_id]

    async def invoke(
        self, body: RequestT, *, inference_profile: bool = True
    ) -> ResponseT:
        """Invoke the model through AWS Bedrock.

        Args:
            body: The input data to invoke the operation.
            inference_profile: If True, use the inference profile. Otherwise, use the model ID.

        Returns:
            The result of the invoked operation.
        """
        return await invoke_json(
            self._model_id,
            body,  # type: ignore[return-value,arg-type]
            inference_profile=inference_profile,
        )

    async def batch_invoke(
        self, bodies: Iterable[RequestT], *, inference_profile: bool = True
    ) -> list[ResponseT]:
        """Invoke the model multiple times through AWS Bedrock.

        Args:
            bodies: The input data to invoke the operation.
            inference_profile: If True, use the inference profile. Otherwise, use the model ID.

        Returns:
            The result of the invoked operation.
        """
        return await gather(
            *(self.invoke(body, inference_profile=inference_profile) for body in bodies)
        )

    async def invoke_async(
        self,
        body: RequestT,
        background_tasks: BackgroundTasks,
        *,
        inference_profile: bool = True,
        output_file: str = "output.json",
    ) -> ResponseT:
        """Invoke the model through AWS Bedrock asynchronous API.

        Args:
            body: The input data to invoke the operation.
            background_tasks: FastAPI background tasks for cleanup.
            inference_profile: If True, use the inference profile. Otherwise, use the model ID.
            output_file: Output file name to retrieve from S3.
                Defaults to "output.json".

        Returns:
            The result of the invoked operation.
        """
        return await invoke_json_async(
            self._model_id,
            body,  # type: ignore[return-value,arg-type]
            background_tasks,
            inference_profile=inference_profile,
            output_file=output_file,
        )


ModelT = TypeVar("ModelT", bound=ModelBase[Any, Any])


async def get_model_details(model_id: str) -> ModelDetails:
    """Get a Bedrock model by its ID.

    Args:
        model_id: The model ID.

    Returns:
        Model details.

    Raises:
        KeyError: If the model is not found.
    """
    async with _CACHE["access_lock"]:
        return _MODELS[model_id]


async def get_all_models_details() -> dict[str, ModelDetails]:
    """Get all models (Bedrock + other AWS services).

    Returns:
        All models details.
    """
    async with _CACHE["access_lock"]:
        return _ALL_MODELS


async def get_all_models_details_and_modalities() -> tuple[
    dict[str, ModelDetails], dict[str, set[str]], dict[str, set[str]]
]:
    """Get all models (Bedrock + other AWS services) with input and output modalities..

    Returns:
        All models details.
    """
    async with _CACHE["access_lock"]:
        return _ALL_MODELS, _ALL_MODELS_OUTPUT_MODALITY, _ALL_MODELS_INPUT_MODALITY


def resolve_model_alias(model_id: str) -> str:
    """Resolve a model alias to its actual model ID.

    Args:
        model_id: The model ID or alias to resolve.

    Returns:
        The resolved model ID. If no alias is found, returns the original model_id.
    """
    return MODEL_ALIASES.get(model_id, model_id)


def update_unified_models_collections() -> None:
    """Updates all model-related collections with additional model data.

    This function takes into account the base models and additional models
    along with their respective input and output modalities, and combines
    them into unified collections. The resulting collections are prepared
    to be accessed and utilized by other parts of the application.

    Raises:
        KeyError: If there is a mismatch of modalities between base models and
                  extra models.
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
    """Retrieve provisioned models from AWS Bedrock.

    Args:
        bedrock_client: AWS Bedrock client for the specific region

    Returns:
        Models IDs.
    """
    next_token = None
    models_ids: set[str] = set()
    params: ListProvisionedModelThroughputsRequestTypeDef = {}
    while True:
        if next_token:
            params["nextToken"] = next_token
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
        next_token = response.get("nextToken")
        if not next_token:
            break
    return models_ids


async def _get_inference_profiles(bedrock_client: BedrockClient) -> dict[str, str]:
    """Retrieve cross region inference profiles from AWS Bedrock.

    Args:
        bedrock_client: AWS Bedrock client for the specific region

    Returns:
        Inference profiles IDs.
    """
    profiles: dict[str, str] = {}
    if SETTINGS.aws_bedrock_cross_region_inference:
        params: ListInferenceProfilesRequestTypeDef = {
            "maxResults": 1000,
            "typeEquals": "SYSTEM_DEFINED",
        }
        next_token = None
        profiles_all: dict[str, list[str]] = {}
        while True:
            if next_token:
                params["nextToken"] = next_token
            response = await bedrock_client.list_inference_profiles(**params)
            for profile in response["inferenceProfileSummaries"]:
                if profile["status"] == "ACTIVE":
                    profiles_all.setdefault(
                        profile["models"][0]["modelArn"].rsplit("/", 1)[-1], []
                    ).append(profile["inferenceProfileId"])
            next_token = response.get("nextToken")
            if not next_token:
                break
        _filter_inference_profiles(profiles, profiles_all)
    return profiles


def _filter_inference_profiles(
    profiles: dict[str, str], profiles_all: dict[str, list[str]]
) -> None:
    """Filters and assigns the appropriate inference profile to the given model.

    Args:
        profiles: A dictionary to store selected profiles for
            each model ID.
        profiles_all: A dictionary containing all profiles for each model ID.
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


async def _get_bedrock_models_from_region(region: str) -> list[ModelDetails]:
    """Get available models from a specific AWS Bedrock region and populate cache.

    Args:
        region: AWS region to query for models
    """
    bedrock_client: BedrockClient = get_client("bedrock", region)

    foundation_models, provisioned_models, profiles = await gather(
        bedrock_client.list_foundation_models(),
        _get_provisioned_models(bedrock_client),
        _get_inference_profiles(bedrock_client),
    )
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
        )
        for model in foundation_models["modelSummaries"]
        if (
            SETTINGS.aws_bedrock_legacy
            or (model["modelLifecycle"]["status"] != "LEGACY")
        )
        and (
            (set(model["inferenceTypesSupported"]) & _INFERENCE_TYPES)
            or (
                "PROVISIONED" in model["inferenceTypesSupported"]
                and model["modelId"] in provisioned_models
            )
        )
    ]


async def initialize_bedrock_models() -> tuple[
    bool, dict[str, dict[str, list[str]]], dict[str, str]
]:
    """Get all available Bedrock models from all configured regions.

    Returns:
        Tuple of (True if the model list was updated, map of unavailable models, map of invalid ARN mappings).
    """
    updated = False
    unavailable_models: dict[str, dict[str, list[str]]] = {}

    async with _CACHE["update_lock"]:
        if _CACHE["update_next"] is None or _CACHE["update_next"] <= SETTINGS.now():
            regions = SETTINGS.aws_bedrock_regions
            region_models = await gather(
                *(_get_bedrock_models_from_region(region) for region in regions)
            )
            all_models: dict[str, ModelDetails] = {}
            for region, models in zip(regions, region_models, strict=False):
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
    return updated, unavailable_models, invalid_arn_mappings


def _apply_user_profiles(all_models: dict[str, ModelDetails]) -> dict[str, str]:
    """Applies user-specified AWS Bedrock ARNs as inference profiles to the respective models.

    Args:
        all_models: A dictionary where keys are model IDs and
            values are `ModelDetails` objects representing the available models.

    Returns:
        A dictionary containing invalid ARN mappings where keys are model IDs and
        values are error messages indicating the issue.
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
            model.region = arn_match.group("region")
    return invalid_arn_mappings


def _populate_model_aliases(all_models: dict[str, ModelDetails]) -> None:
    """Populates the aliases field for each model based on the MODEL_ALIASES mapping.

    Args:
        all_models: A dictionary where keys are model IDs and
            values are `ModelDetails` objects representing the available models.
    """
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
    """Filter and validate a Bedrock model for availability and authorization.

    Checks if a model meets the configured criteria and is available in the region.
    Only models that pass all checks are added to the global model cache.

    Args:
        bedrock_client: AWS Bedrock client for the specific region
        model: Foundation model summary from AWS Bedrock
        models: All models.
        unavailable_models: Map of model IDs to region availability status.

    Returns:
        None: Models are added to global cache dictionaries as side effect
    """
    if model.id not in models:
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
            models[model.id] = model
        else:
            unavailable_models.setdefault(model.id, {})[model.region] = [
                issue
                for issue, value, expected in (
                    ("unauthorized", availability["authorizationStatus"], "AUTHORIZED"),
                    (
                        "unentitled",
                        availability["entitlementAvailability"],
                        "AVAILABLE",
                    ),
                    ("unavailable", availability["regionAvailability"], "AVAILABLE"),
                    (
                        "no_agreement",
                        availability["agreementAvailability"]["status"],
                        "AVAILABLE",
                    ),
                )
                if value != expected
            ]


def load_model_plugins(
    package_name: str,
    class_type: type[ModelT],
    registry: list[tuple[str | Pattern[str], type[ModelT]]],
) -> None:
    """Import all modules in the specified package and auto-register model classes.

    Args:
        package_name: Package name under which to import the model
        class_type: Class name under which to import the model
        registry: Models classes registry with string prefix or regex pattern matchers.
    """
    class_name = class_type.__name__.removesuffix("Base")
    for module_info in iter_modules(import_module(package_name).__path__):
        name = module_info.name
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


def get_model(
    model_id: str,
    cache: dict[str, ModelT],
    registry: list[tuple[str | Pattern[str], type[ModelT]]],
) -> ModelT:
    """Resolve the model class matching the provided identifier.

    Args:
        model_id: The provider model identifier.
        cache: Model cache dictionary.
        registry: Models classes registry with string prefix or regex pattern matchers.

    Returns:
        The model associated with the ``model_id``.

    Raises:
        LookupError: If no registered model matches ``model_id``.
    """
    try:
        return cache[model_id]
    except KeyError:
        for matcher, model_cls in registry:
            # Support both string prefix and compiled regex patterns
            matched = (
                model_id.startswith(matcher)
                if isinstance(matcher, str)
                else matcher.match(model_id)
                if isinstance(matcher, Pattern)
                else False
            )
            if matched:
                cache[model_id] = model_cls(model_id)
                return cache[model_id]
    raise OpenaiUnsupportedModelError(model_id)


async def _prepare_bedrock_request(
    model_id: str, body: Mapping[str, Any], *, inference_profile: bool = True
) -> tuple[BedrockRuntimeClient, InvokeModelRequestTypeDef]:
    """Prepare a Bedrock request with common setup logic.

    Args:
        model_id: Model ID.
        body: JSON payload.
        inference_profile: If True, use the inference profile. Otherwise, use the model ID.

    Returns:
        A tuple of (BedrockRuntimeClient, request kwargs).
    """
    model = await get_model_details(model_id)
    bedrock_client: BedrockRuntimeClient = get_client("bedrock-runtime", model.region)
    kwargs: InvokeModelRequestTypeDef = {
        "modelId": model.get_id(inference_profile=inference_profile),
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
    return bedrock_client, kwargs


async def invoke_json(
    model_id: str, body: Mapping[str, Any], *, inference_profile: bool = True
) -> Mapping[str, Any]:
    """Invoke a Bedrock model from a JSON payload and return the JSON response.

    Args:
        model_id: Model ID.
        body: JSON payload.
        inference_profile: If True, use the inference profile. Otherwise, use the model ID.

    Returns:
        JSON response.
    """
    bedrock_client, kwargs = await _prepare_bedrock_request(
        model_id, body, inference_profile=inference_profile
    )
    with handle_bedrock_client_error():
        response = await bedrock_client.invoke_model(**kwargs)
    return from_json(await response["body"].read())  # type: ignore[no-any-return]


async def prepare_converse_request(
    model: ModelDetails,
    bedrock_messages: list[MessageTypeDef],
    inference_cfg: InferenceConfigurationTypeDef,
    system_blocks: list[SystemContentBlockTypeDef],
    tool_config: ToolConfigurationTypeDef | None,
    additional_request_fields: Mapping[str, Any],
    service_tier: ServiceTierTypeType | None,
    prompt_caching: set[PromptCaching],
    *,
    inference_profile: bool = True,
) -> tuple[BedrockRuntimeClient, ConverseRequestBaseTypeDef]:
    """Prepare a Bedrock Converse request payload and client.

    Args:
        model: Model details.
        bedrock_messages: Converted Bedrock message list.
        inference_cfg: Bedrock inference configuration.
        system_blocks: Optional top-level system instruction blocks.
        tool_config: Optional Bedrock tool configuration.
        additional_request_fields: Additional request fields.
        service_tier: Service tier configuration.
        prompt_caching: Prompt caching configuration.
        inference_profile: If True, use the inference profile. Otherwise, use the model ID.

    Returns:
        A tuple of (BedrockRuntimeClient, request payload dict).
    """
    latency, default_service_tier = PERFORMANCE_CONFIG_VAR.get()
    service_tier = service_tier or default_service_tier
    request: ConverseRequestBaseTypeDef = {
        "modelId": model.get_id(inference_profile=inference_profile),
        "messages": bedrock_messages,
        "inferenceConfig": inference_cfg,
    }
    if system_blocks:
        request["system"] = system_blocks
    if tool_config:
        request["toolConfig"] = tool_config
    if additional_request_fields:
        request["additionalModelRequestFields"] = additional_request_fields
    if service_tier:
        request["serviceTier"] = {"type": service_tier}
    if latency:
        request["performanceConfig"] = {"latency": latency}

    with suppress(LookupError):
        request["guardrailConfig"] = GUARDTRAIL_CONFIG_VAR.get()
    _enable_converse_prompt_caching(
        model, system_blocks, tool_config, bedrock_messages, prompt_caching
    )
    return get_client("bedrock-runtime", model.region), request


def _enable_converse_prompt_caching(
    model: ModelDetails,
    system_blocks: list[SystemContentBlockTypeDef],
    tool_config: ToolConfigurationTypeDef | None,
    bedrock_messages: list[MessageTypeDef],
    prompt_caching: set[PromptCaching],
) -> None:
    """Enables prompt caching for specified components including system blocks, tools, and messages.

    Args:
        model: Model details.
        bedrock_messages: A list of message objects of type MessageTypeDef,
            on which caching will be applied if "messages" is in the prompt_caching set.
        system_blocks: A list of system content blocks of type SystemContentBlockTypeDef to
            which the cache point will be appended if "system" is in the prompt_caching set.
        tool_config: An optional tool configuration of type ToolConfigurationTypeDef.
            If "tools" is in prompt_caching and the configuration is provided,
            the cache point is appended to its tools attribute.
        prompt_caching: A set of PromptCaching values that specifies the components
            (e.g., "system", "tools", "messages") for which caching should be enabled.
    """
    if model.id.startswith(PROMPT_CACHING_SUPPORTED):
        if "system" in prompt_caching and system_blocks:
            system_blocks.append(_CACHE_POINT)  # type: ignore[arg-type]
        if "messages" in prompt_caching and bedrock_messages:
            for message in bedrock_messages:
                message["content"].append(_CACHE_POINT)  # type: ignore[attr-defined]
        if (
            "tools" in prompt_caching
            and tool_config
            and model.id.startswith(PROMPT_CACHING_TOOL_SUPPORTED)
        ):
            tool_config["tools"].append(_CACHE_POINT)  # type: ignore[attr-defined]


async def validate_model(
    model_id: str,
    output_modality: str | None = None,
    input_modality: str | None = None,
    *,
    bedrock_only: bool = True,
) -> ModelDetails:
    """Validate and return the model details for a given model ID.

    Args:
        model_id: Model ID or alias to validate
        output_modality: Expected output modality.
        input_modality: Expected input modality.
        bedrock_only: If True, only allow Bedrock models.

    Returns:
        Returns the model details.

    Raises:
        HTTPException: If model is not found or not supported
    """
    model_id = resolve_model_alias(model_id)
    if model_id.startswith("arn:"):
        model = await _validate_model_from_arn(model_id)
    else:
        # First, try to get the model from the cache
        models = _MODELS if bedrock_only else _ALL_MODELS
        async with _CACHE["access_lock"]:
            try:
                model = models[model_id]
            except KeyError:
                model = None

    # If not found, update the cache and retry, if still not found, raise an error
    if model is None:
        await initialize_bedrock_models()
        async with _CACHE["access_lock"]:
            try:
                model = models[model_id]
            except KeyError:
                try:
                    msg = (
                        f"Model '{model_id}' not found. "
                        f"This model is deprecated or pending deprecation, "
                        f"please use '{DEPRECATED_MODELS[model_id]}' instead."
                    )
                except KeyError:
                    msg = f"Model '{model_id}' not found."
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
                raise OpenaiUnsupportedModelError(
                    msg, available_models=model_ids
                ) from None

    # Check model modalities
    if output_modality and output_modality not in model.output_modalities:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' does not support {output_modality.lower()} output modality.",
        )
    if input_modality and input_modality not in model.input_modalities:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' does not support {input_modality.lower()} input modality.",
        )
    log = REQUEST_LOG.get()
    log["model_id"] = model_id
    return model


async def _validate_model_from_arn(arn: str) -> ModelDetails | None:
    """Validates the model associated with a given ARN and returns its details.

    This function internally checks the cached user profiles and their expiration timestamps.
    If the model data is not available in the cache, it delegates the retrieval to external
    services based on the ARN. If a valid model is fetched, its details are cached for
    future use.

    Args:
        arn: The Amazon Resource Name (ARN) of the model to be validated and retrieved.

    Returns:
        The details of the validated model if available.

    Raises:
        OpenaiError: If the ARN does not correspond to a valid application inference profile
        or prompt router, or if the model cannot be found for the provided ARN.
        HTTPException: If the ARN type is not allowed by the server configuration.
    """
    models: (
        Sequence[InferenceProfileModelTypeDef]
        | Sequence[PromptRouterTargetModelTypeDef]
        | None
    ) = None
    async with _CACHE["user_profiles_access_lock"]:
        try:
            model, expiration = _USER_PROFILES[arn]
            if expiration > SETTINGS.now():
                return model
            del _USER_PROFILES[arn]
        except KeyError:
            pass

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
                raise OpenaiError(msg)

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
                raise OpenaiError(msg) from None
        model = base_model.model_copy()
        model.inference_profile = arn
        model.region = region
        _USER_PROFILES[arn] = (model, SETTINGS.now() + _CACHE["update_interval"])
        return model


async def _get_prompt_router_models(
    arn: str,
) -> tuple[Sequence[PromptRouterTargetModelTypeDef], str] | None:
    """Retrieves a list of models associated with a given Prompt Router ARN from AWS Bedrock.

    Args:
        arn: The ARN of the Prompt Router for which the associated models are to be
            retrieved.

    Returns:
        A sequence of target model definitions associated with the provided Prompt
        Router ARN, the region of the provided Prompt
    """
    if result := match_bedrock_prompt_router_arn(arn):
        if not SETTINGS.aws_bedrock_allow_prompt_router_arn:
            msg = "Prompt router are not allowed by server configuration."
            raise OpenaiError(msg)
        region = result.group("region")
        return (
            await get_client("bedrock", region).get_prompt_router(promptRouterArn=arn)
        ).get("models") or (), region
    return None


async def _get_application_inference_profile_models(
    arn: str,
) -> tuple[Sequence[InferenceProfileModelTypeDef], str] | None:
    """Fetches inference profile models associated with the specified ARN.

    Args:
        arn: The Amazon Resource Name (ARN) of the application inference profile.

    Returns:
        A sequence of models associated with the application inference profile,
        the region of the application inference profile
    """
    if result := match_bedrock_app_profile_arn(arn):
        if "application-inference-profile" in arn:
            if not SETTINGS.aws_bedrock_allow_application_inference_profile_arn:
                msg = "Application inference profile are not allowed by server configuration."
                raise OpenaiError(msg)
        elif not SETTINGS.aws_bedrock_allow_cross_region_inference_profile_arn:
            msg = "Cross-region inference profile are not allowed by server configuration."
            raise OpenaiError(msg)
        region = result.group("region")
        return (
            await get_client("bedrock", region).get_inference_profile(
                inferenceProfileIdentifier=arn
            )
        ).get("models") or (), region
    return None


async def _wait_for_async_invocation_completion(
    bedrock_client: BedrockRuntimeClient, invocation_arn: str
) -> str:
    """Wait for async invocation to complete.

    Args:
        bedrock_client: Bedrock Runtime client
        invocation_arn: Async invocation ARN

    Returns:
        S3 object key.

    Raises:
        HTTPException: If invocation fails
    """
    while True:  # Timeout at FastAPI level
        response = await bedrock_client.get_async_invoke(invocationArn=invocation_arn)
        status = response["status"]
        if status == "Completed":
            return (
                response["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
                .removeprefix("s3://")
                .split("/", 1)[1]
            )
        if status == "Failed":
            raise HTTPException(status_code=400, detail=response["failureMessage"])
        await sleep(0.5)


async def invoke_json_async(
    model_id: str,
    body: Mapping[str, Any],
    background_tasks: BackgroundTasks,
    *,
    inference_profile: bool = True,
    output_file: str = "output.json",
) -> Mapping[str, Any]:
    """Invoke a Bedrock model asynchronously from a JSON payload and return the JSON response.

    This function handles the entire async invocation workflow from starting the
    async invocation through AWS Bedrock processing to result retrieval from S3,
    including AWS client initialization and cleanup management.

    Args:
        model_id: Model ID.
        body: JSON payload.
        background_tasks: FastAPI background tasks for cleanup.
        inference_profile: If True, use the inference profile. Otherwise, use the model ID.
        output_file: Output JSON file name to retrieve from S3 (e.g., "output.json").
            Defaults to "output.json".

    Returns:
        JSON response from the output file.

    Raises:
        HTTPException: When invocation configuration is missing, invocation fails,
            or results cannot be retrieved.
    """
    model = await get_model_details(model_id)
    s3_bucket, s3_client = get_model_s3_bucket(model)
    bedrock_client: BedrockRuntimeClient = get_client("bedrock-runtime", model.region)
    s3_tmp_objects: list[tuple[str, str]] = []
    request_id = REQUEST_ID.get()
    try:
        with handle_bedrock_client_error():
            invocation_arn = (
                await bedrock_client.start_async_invoke(
                    modelId=model.get_id(inference_profile=inference_profile),
                    modelInput=body,
                    outputDataConfig={
                        "s3OutputDataConfig": {
                            "s3Uri": f"s3://{s3_bucket}/{SETTINGS.aws_s3_tmp_prefix}{request_id}/"
                        }
                    },
                )
            )["invocationArn"]

        s3_key = await _wait_for_async_invocation_completion(
            bedrock_client, invocation_arn
        )
        s3_output_path = f"{s3_key}/{output_file}"
        s3_tmp_objects.append((s3_bucket, s3_output_path))
        s3_tmp_objects.append((s3_bucket, f"{s3_key}/manifest.json"))
        return from_json(  # type: ignore[no-any-return]
            await (await s3_client.get_object(Bucket=s3_bucket, Key=s3_output_path))[
                "Body"
            ].read()
        )

    finally:
        if s3_tmp_objects:
            background_tasks.add_task(
                aws_s3_cleanup, s3_client, s3_tmp_objects, request_id
            )


def get_model_s3_bucket(model: ModelDetails) -> tuple[str, S3Client]:
    """Retrieve the S3 bucket and S3 client for a given model's region.

    This function determines the appropriate S3 bucket and initializes the S3 client
    based on the model's associated region. If the region-specific bucket is
    not configured and the default region matches, it uses the globally configured
    S3 bucket. If no valid configuration exists, the function logs the error details
    and raises an HTTPException indicating the unavailability of async invocation.

    Args:
        model (ModelDetails): The model details containing a region attribute.

    Returns:
        tuple[str, S3Client]: A tuple containing the S3 bucket name and the initialized
        S3 client for the given region.

    Raises:
        HTTPException: If the required S3 bucket configurations for the region or
        default context are missing, an HTTPException is raised with a relevant
        error message.
    """
    try:
        s3_bucket = SETTINGS.aws_s3_regional_buckets[model.region]
    except KeyError as error:
        if model.region == SETTINGS.aws_bedrock_regions[0]:
            if SETTINGS.aws_s3_bucket:
                return SETTINGS.aws_s3_bucket, get_client("s3")
            log_error_details(
                "S3 bucket not configured (aws_s3_bucket): some features are disabled"
            )
        else:
            log_error_details(
                f"S3 {model.region} regional bucket not configured (aws_s3_regional_buckets): some features are disabled"
            )
        raise HTTPException(
            status_code=400,
            detail="Async invocation is not available on the current server. "
            "Please contact the administrator to enable it.",
        ) from error
    return s3_bucket, get_client("s3", model.region)


async def put_to_s3(
    content: BlobTypeDef | str, model: ModelDetails, content_type: str = ""
) -> tuple[str, str]:
    """Uploads content to an S3 bucket under a temporary key.

    The key is derived from the request ID and model details.
    This function handles both raw binary content and base64 encoded data URIs.

    Args:
        content: The content to upload. Can be a raw byte string or a base64 encoded
            data URI.
        model: Details of the model for which the content is being uploaded. Helps
            determine the appropriate S3 storage location.
        content_type: The MIME type of the content being uploaded. Used to set the
            `ContentType` in the S3 object metadata. Defaults to an empty string,
            indicating no specific content type.

    Returns:
        A tuple containing the S3 bucket name and the key of the uploaded object.
    """
    if isinstance(content, str):
        content = (
            await b64decode_data_uri(content)
            if content.startswith("data:")
            else content.encode()
        )
    ext = f".{content_type.split('/')[-1]}" if content_type else ""
    s3_key = f"{SETTINGS.aws_s3_tmp_prefix}{REQUEST_ID.get()}/{token_hex(4)}{ext}"
    s3_bucket, s3_client = get_model_s3_bucket(model)
    kwargs: PutObjectRequestTypeDef = {
        "Bucket": s3_bucket,
        "Key": s3_key,
        "Body": content,
    }
    if content_type:
        kwargs["ContentType"] = content_type
    await s3_client.put_object(**kwargs)
    return s3_bucket, s3_key


async def get_s3_content_type_and_size(
    uri: str, model: ModelDetails
) -> tuple[str, int]:
    """Get information about the S3 file behind an S3 URI.

    Args:
        uri: plain text, base64 encoded as string or URI, or S3 URI
        model: Model details containing region information.

    Returns:
        File size and content type.

    Raises:
        HTTPException: If the required S3 bucket for the region is not configured.
    """
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    result = await get_client("s3", region_name=model.region).head_object(
        Bucket=bucket, Key=key
    )
    return result["ContentLength"], result["ContentType"]


async def get_content_type_and_size(value: str, model: ModelDetails) -> tuple[str, int]:
    """Determines the content type and size of the given data based on its format.

    This function analyzes a given input value to determine its content type and size. The input
    can be in the form of a data URI, an S3 resource path, or plain text. Based on the format,
    the function identifies the content type and computes the size of the data.

    Args:
        value: The input data to be evaluated. It can be in a data URI format,
            an S3 path, or a plain string.
        model: The model object responsible for handling metadata
            when processing S3 resources.

    Returns:
        A tuple containing the content type as a string and the size of the data as an integer.
    """
    if value.startswith("data:"):
        return get_data_uri_type(value), get_base64_decoded_size(value)
    if value.startswith("s3://"):
        return await get_s3_content_type_and_size(value, model)
    return "text/plain", len(value)
