"""Models."""

from asyncio import Lock, Queue, create_task, gather, sleep
from collections.abc import AsyncGenerator, Iterable, Mapping
from contextlib import suppress
from datetime import timedelta
from importlib import import_module
from pkgutil import iter_modules
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, TypeVar

from botocore.exceptions import ClientError
from fastapi import BackgroundTasks, HTTPException
from pydantic import AwareDatetime, BaseModel
from pydantic_core import from_json, to_json

from stdapi.aws import get_client
from stdapi.aws_bedrock import GUARDTRAIL_CONFIG_VAR, handle_bedrock_client_error
from stdapi.aws_s3 import aws_s3_cleanup
from stdapi.config import SETTINGS
from stdapi.models.deprecation import DEPRECATED_MODELS
from stdapi.monitoring import REQUEST_ID, REQUEST_LOG, log_error_details
from stdapi.openai_exceptions import OpenaiUnsupportedModelError

if TYPE_CHECKING:
    from types_aiobotocore_bedrock.client import BedrockClient
    from types_aiobotocore_bedrock.type_defs import (
        ListInferenceProfilesRequestTypeDef,
        ListProvisionedModelThroughputsRequestTypeDef,
    )
    from types_aiobotocore_bedrock_runtime import BedrockRuntimeClient
    from types_aiobotocore_bedrock_runtime.literals import TraceType
    from types_aiobotocore_bedrock_runtime.type_defs import (
        InferenceConfigurationTypeDef,
        InvokeModelRequestTypeDef,
        MessageTypeDef,
        PerformanceConfigurationTypeDef,
        SystemContentBlockTypeDef,
        ToolConfigurationTypeDef,
    )
    from types_aiobotocore_s3.client import S3Client

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef

    class _ModelCache(TypedDict):
        """Model cache configuration."""

        update_next: AwareDatetime | None
        update_interval: timedelta
        update_lock: Lock
        access_lock: Lock


#: Models details
_MODELS: dict[str, "ModelDetails"] = {}

#: Non Bedrock models details
EXTRA_MODELS: dict[str, "ModelDetails"] = {}

#: All models
_ALL_MODELS: dict[str, "ModelDetails"] = {}

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
_CACHE: "_ModelCache" = {
    "update_next": None,
    "update_lock": Lock(),
    "update_interval": timedelta(seconds=SETTINGS.model_cache_seconds),
    "access_lock": Lock(),
}

#: Always allowed inference types
_INFERENCE_TYPES = {"INFERENCE_PROFILE", "ON_DEMAND"}


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

    MATCHER: ClassVar[str] = ""

    def __init__(self, model_id: str) -> None:
        """Initialize the model with a specific model identifier.

        Args:
            model_id: The AWS Bedrock model identifier.
        """
        self._model_id = model_id

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

    async def invoke_stream(
        self, body: RequestT, *, inference_profile: bool = True
    ) -> AsyncGenerator[ResponseT]:
        """Invoke the model through AWS Bedrock with streaming response.

        Args:
            body: The input data to invoke the operation.
            inference_profile: If True, use the inference profile. Otherwise, use the model ID.

        Yields:
            Streaming response chunks from the invoked operation.
        """
        async for chunk in invoke_json_stream(
            self._model_id,
            body,  # type: ignore[arg-type]
            inference_profile=inference_profile,
        ):
            yield chunk  # type: ignore[misc]

    async def invoke_async(
        self,
        body: RequestT,
        background_tasks: BackgroundTasks,
        *,
        inference_profile: bool = True,
    ) -> ResponseT:
        """Invoke the model through AWS Bedrock asynchronous API.

        Args:
            body: The input data to invoke the operation.
            background_tasks: FastAPI background tasks for cleanup.
            inference_profile: If True, use the inference profile. Otherwise, use the model ID.

        Returns:
            The result of the invoked operation.
        """
        return await invoke_json_async(
            self._model_id,
            body,  # type: ignore[return-value,arg-type]
            background_tasks,
            inference_profile=inference_profile,
        )

    async def batch_invoke_async(
        self,
        bodies: Iterable[RequestT],
        background_tasks: BackgroundTasks,
        *,
        inference_profile: bool = True,
    ) -> list[ResponseT]:
        """Invoke the model multiple times through AWS Bedrock asynchronously.

        Args:
            bodies: The input data to invoke the operation.
            background_tasks: FastAPI background tasks for cleanup.
            inference_profile: If True, use the inference profile. Otherwise, use the model ID.

        Returns:
            The results of the invoked operations.
        """
        return await gather(
            *(
                self.invoke_async(
                    body, background_tasks, inference_profile=inference_profile
                )
                for body in bodies
            )
        )

    async def batch_invoke_stream(
        self, bodies: Iterable[RequestT], *, inference_profile: bool = True
    ) -> AsyncGenerator[tuple[int, ResponseT]]:
        """Invoke the model multiple times through AWS Bedrock with streaming responses.

        Args:
            bodies: The input data to invoke the operation.
            inference_profile: If True, use the inference profile. Otherwise, use the model ID.

        Yields:
            Tuples of (generator_index, response_chunk) where generator_index indicates
            which input body the response chunk corresponds to.
        """
        generators = [
            self.invoke_stream(body, inference_profile=inference_profile)
            for body in bodies
        ]
        if generators:
            queue: Queue[tuple[int, ResponseT] | None] = Queue()
            tasks = [
                create_task(self._generator_to_queue(gen, queue, index))
                for index, gen in enumerate(generators)
            ]
            completed_generators = 0
            try:
                while completed_generators < len(generators):
                    item = await queue.get()
                    if item is None:
                        completed_generators += 1
                    else:
                        yield item
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await gather(*tasks)

    @staticmethod
    async def _generator_to_queue(
        gen: AsyncGenerator[ResponseT],
        queue: Queue[tuple[int, ResponseT] | None],
        index: int,
    ) -> None:
        """Converts an asynchronous generator into an asyncio queue and signals completion.

        This static method consumes items from an asynchronous generator and places them into the
        provided asyncio queue along with their associated index. After consuming all items, it
        places a completion signal (None) into the queue.

        Args:
            gen: An asynchronous generator yielding responses.
            queue: The asyncio queue where
                the generator items will be placed. The queue also receives a completion
                signal (None) after processing all items.
            index: An index associated with the generator, used to track which
                generator the items originate from.
        """
        try:
            async for item in gen:
                await queue.put((index, item))
        finally:
            await queue.put(None)  # Signal completion
            await gen.aclose()


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


async def _get_provisioned_models(bedrock_client: "BedrockClient") -> set[str]:
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
        except ClientError as exc:
            error = exc.response["Error"]
            if (
                error["Code"] == "AccessDeniedException"
                and "not supported" in error["Message"]
            ):
                break
            raise  # pragma: no cover
        for model in response["provisionedModelSummaries"]:
            models_ids.add(model["modelArn"].rsplit("/", 1)[-1])
        next_token = response.get("nextToken")
        if not next_token:
            break
    return models_ids


async def _get_inference_profiles(bedrock_client: "BedrockClient") -> dict[str, str]:
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


async def initialize_bedrock_models() -> tuple[bool, dict[str, dict[str, list[str]]]]:
    """Get all available Bedrock models from all configured regions.

    Returns:
        Tuple of (True if the model list was updated, map of unavailable models).
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
    return updated, unavailable_models


async def _filter_model(
    bedrock_client: "BedrockClient",
    model: ModelDetails,
    models: dict[str, "ModelDetails"],
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
    registry: list[tuple[str, type[ModelT]]],
) -> None:
    """Import all modules in the specified package and auto-register model classes.

    Args:
        package_name: Package name under which to import the model
        class_type: Class name under which to import the model
        registry: Models classes registry.
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
    model_id: str, cache: dict[str, ModelT], registry: list[tuple[str, type[ModelT]]]
) -> ModelT:
    """Resolve the model class matching the provided identifier.

    Args:
        model_id: The provider model identifier.
        cache: Model cache dictionary.
        registry: Models classes registry.

    Returns:
        The model associated with the ``model_id``.

    Raises:
        LookupError: If no registered model matches ``model_id``.
    """
    try:
        return cache[model_id]
    except KeyError:
        for matcher, model_cls in registry:
            if model_id.startswith(matcher):
                cache[model_id] = model_cls(model_id)
                return cache[model_id]
    raise OpenaiUnsupportedModelError(model_id)


async def _prepare_bedrock_request(
    model_id: str, body: Mapping[str, Any], *, inference_profile: bool = True
) -> "tuple[BedrockRuntimeClient, InvokeModelRequestTypeDef]":
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


async def invoke_json_stream(
    model_id: str, body: Mapping[str, Any], *, inference_profile: bool = True
) -> AsyncGenerator[Mapping[str, Any]]:
    """Invoke a Bedrock model from a JSON payload and return a streaming JSON response.

    Args:
        model_id: Model ID.
        body: JSON payload.
        inference_profile: If True, use the inference profile. Otherwise, use the model ID.

    Yields:
        JSON response chunks from the streaming response.
    """
    bedrock_client, kwargs = await _prepare_bedrock_request(
        model_id, body, inference_profile=inference_profile
    )
    with handle_bedrock_client_error():
        response = await bedrock_client.invoke_model_with_response_stream(**kwargs)
        async for event in response["body"]:
            if "chunk" in event:
                yield from_json(event["chunk"]["bytes"])
                continue
            for key, value in event:  # type: ignore[misc]
                if key.endswith("Exception"):  # type: ignore[has-type]
                    raise ClientError(
                        error_response={
                            "Error": {
                                "Code": f"{key[0]}{key[1:]}",  # type: ignore[has-type]
                                "Message": value["message"],  # type: ignore[has-type]
                            }
                        },
                        operation_name="InvokeModelWithResponseStream",
                    )


async def prepare_converse_request(
    model: ModelDetails,
    bedrock_messages: list["MessageTypeDef"],
    inference_cfg: "InferenceConfigurationTypeDef",
    system_blocks: list["SystemContentBlockTypeDef"],
    tool_config: "ToolConfigurationTypeDef | None",
    additional_request_fields: Mapping[str, Any],
    performance_config: "PerformanceConfigurationTypeDef",
    *,
    inference_profile: bool = True,
) -> "tuple[BedrockRuntimeClient, ConverseRequestBaseTypeDef]":
    """Prepare a Bedrock Converse request payload and client.

    Args:
        model: Model details.
        bedrock_messages: Converted Bedrock message list.
        inference_cfg: Bedrock inference configuration.
        system_blocks: Optional top-level system instruction blocks.
        tool_config: Optional Bedrock tool configuration.
        additional_request_fields: Additional request fields.
        performance_config: Prefered performance configuration.
        inference_profile: If True, use the inference profile. Otherwise, use the model ID.

    Returns:
        A tuple of (BedrockRuntimeClient, request payload dict).
    """
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
    if performance_config:
        request["performanceConfig"] = performance_config
    with suppress(LookupError):
        request["guardrailConfig"] = GUARDTRAIL_CONFIG_VAR.get()
    return get_client("bedrock-runtime", model.region), request


async def validate_model(
    model_id: str,
    output_modality: str | None = None,
    input_modality: str | None = None,
    *,
    bedrock_only: bool = True,
) -> ModelDetails:
    """Validate and return the model details for a given model ID.

    Args:
        model_id: Model ID to validate
        output_modality: Expected output modality.
        input_modality: Expected input modality.
        bedrock_only: If True, only allow Bedrock models.

    Returns:
        Returns the model details.

    Raises:
        HTTPException: If model is not found or not supported
    """
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


async def _wait_for_async_invocation_completion(
    bedrock_client: "BedrockRuntimeClient", invocation_arn: str
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

    Returns:
        JSON response.

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
                            "s3Uri": f"s3://{s3_bucket}/{request_id}"
                        }
                    },
                )
            )["invocationArn"]

        s3_key = await _wait_for_async_invocation_completion(
            bedrock_client, invocation_arn
        )
        s3_output = f"{s3_key}/output.json"
        s3_tmp_objects.extend(
            ((s3_bucket, s3_output), (s3_bucket, f"{s3_key}/manifest.json"))
        )
        return from_json(  # type: ignore[no-any-return]
            await (await s3_client.get_object(Bucket=s3_bucket, Key=s3_output))[
                "Body"
            ].read()
        )

    finally:
        if s3_tmp_objects:
            background_tasks.add_task(
                aws_s3_cleanup, s3_client, s3_tmp_objects, request_id
            )


def get_model_s3_bucket(model: ModelDetails) -> "tuple[str, S3Client]":
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
