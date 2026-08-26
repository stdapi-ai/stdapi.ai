"""OpenAI-compatible Models API implementation using AWS Bedrock."""

from asyncio import Lock
from typing import Annotated

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel

from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.models import (
    ModelDetails,
    catalog_generation,
    get_all_models_details,
    initialize_bedrock_models,
    validate_model,
)
from stdapi.monitoring import log_request_params, log_response_params
from stdapi.types.openai_models import Model

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=["Models", TAG_OPENAI]
)


class ModelsResponse(BaseModel):
    """Response for the /v1/models endpoint following OpenAI API specification."""

    object: str = "list"
    data: list[Model]


#: /v1/models route response cache
_ALL_MODELS: list[Model] = []
#: Cached ModelsResponse, rebuilt alongside `_ALL_MODELS`.
_MODELS_RESPONSE = ModelsResponse(data=[])
#: Guards concurrent rebuilds of the model caches above.
_ALL_MODELS_LOCK = Lock()
#: Catalog generation the caches above were built from; -1 until they are.
_CATALOG_GENERATION = -1


def format_bedrock_model_to_openai(model: ModelDetails) -> Model:
    """Format a Bedrock model to OpenAI API model format.

    Args:
        model: Bedrock foundation model summary object

    Returns:
        Model object formatted according to OpenAI API specification
    """
    return Model(
        id=model.id,
        object="model",
        created=(
            int(model.start_of_life_time.timestamp()) if model.start_of_life_time else 0
        ),
        owned_by=model.provider,
    )


@router.get(
    "/models",
    summary="List available models (OpenAI format)",
    operation_id="openai_model_list",
    description=(
        "Lists all currently available models with basic metadata (owner, creation date) "
        "(OpenAI Models API).\n\n"
        "**Agent note:** For richer filtering — by modality, route, MCP tool, region, or "
        "legacy status — use `search_models` instead."
    ),
    response_description="Describes model offerings that can be used with the API",
    response_model_exclude_none=True,
    responses={
        200: {
            "description": "List of available models.",
            "content": {
                "application/json": {
                    "examples": {
                        "list": {
                            "summary": "Example list",
                            "value": {
                                "object": "list",
                                "data": [
                                    {
                                        "id": "amazon.nova-micro-v1:0",
                                        "object": "model",
                                        "created": 1640995200,
                                        "owned_by": "Amazon",
                                    },
                                    {
                                        "id": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                                        "object": "model",
                                        "created": 1640995200,
                                        "owned_by": "Anthropic",
                                    },
                                ],
                            },
                        }
                    }
                }
            },
        }
    },
)
async def list_models(_: Annotated[None, Depends(authenticate)]) -> ModelsResponse:
    """Lists the currently available models.

    Returns:
        ModelsResponse containing list of all available models with metadata

    Raises:
        ApiError: When unable to retrieve models from backend services (500)
    """
    global _MODELS_RESPONSE, _CATALOG_GENERATION  # noqa: PLW0603
    await initialize_bedrock_models()
    # Compared rather than taken from the call above: a refresh that ran in the
    # background, or that another listing route triggered, changes the catalog
    # without this call ever seeing it.
    generation = catalog_generation()
    async with _ALL_MODELS_LOCK:
        if generation != _CATALOG_GENERATION or not _ALL_MODELS:
            models = await get_all_models_details()
            _ALL_MODELS.clear()
            _ALL_MODELS.extend(
                format_bedrock_model_to_openai(models[model_id])
                for model_id in sorted(models)
            )
            _MODELS_RESPONSE = ModelsResponse(data=list(_ALL_MODELS))
            _CATALOG_GENERATION = generation
    return log_response_params(_MODELS_RESPONSE)


@router.get(
    "/models/{model}",
    summary="Retrieve details for a specific model by ID (OpenAI format)",
    operation_id="openai_model_get",
    description=(
        "Retrieves basic metadata (owner, creation date) for a single model by ID (OpenAI Models API).\n\n"
        "**Agent note:** Use `search_models` to look up modalities, supported routes, regions, "
        "and other extended metadata not available here."
    ),
    response_description="Describes a model offering that can be used with the API.",
    responses={
        200: {
            "description": "Model retrieved successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "model": {
                            "summary": "Example model",
                            "value": {
                                "id": "amazon.nova-micro-v1:0",
                                "object": "model",
                                "created": 1640995200,
                                "owned_by": "Amazon",
                            },
                        }
                    }
                }
            },
        },
        404: {
            "description": "Model not found",
            "content": {
                "application/json": {
                    "examples": {
                        "not_found": {
                            "summary": "Model not found",
                            "value": {
                                "error": {
                                    "message": "The model `unknown` does not exist or you do not have access to it.",
                                    "type": "invalid_request_error",
                                    "param": None,
                                    "code": "model_not_found",
                                }
                            },
                        }
                    }
                }
            },
        },
    },
    response_model_exclude_none=True,
)
async def retrieve_model(
    model: str = Path(  # noqa: FAST002
        ...,
        description="The ID of the model to use for this request",
        examples=["amazon.nova-micro-v1:0"],
        min_length=1,
        max_length=255,
        str_strip_whitespace=True,
    ),
    _: Annotated[None, Depends(authenticate)] = None,
) -> Model:
    """Retrieve a specific model by its ID from AWS Bedrock.

    Args:
        model: The ID of the model to retrieve

    Returns:
        Model object with details about the specified model

    Raises:
        ApiError: When the model is not found (404)
    """
    log_request_params({"model": model})
    return log_response_params(
        format_bedrock_model_to_openai(await validate_model(model, bedrock_only=False))
    )
