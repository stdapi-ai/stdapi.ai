"""OpenAI-compatible Models API implementation using AWS Bedrock."""

from asyncio import Lock
from time import time
from typing import Annotated

from fastapi import APIRouter, Depends, Path
from pydantic import BaseModel

from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.models import (
    ModelDetails,
    get_all_models_details,
    initialize_bedrock_models,
    validate_model,
)
from stdapi.monitoring import log_request_params, log_response_params
from stdapi.types.openai_models import Model

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=["models", "openai"]
)

#: /v1/models route response cache
_ALL_MODELS: list["Model"] = []
_ALL_MODELS_LOCK = Lock()

#: Default model timestamp
_TIMESTAMP = int(time())


class ModelsResponse(BaseModel):
    """Response for the /v1/models endpoint following OpenAI API specification.

    Contains a list of available models and response metadata.
    """

    object: str = "list"
    data: list[Model]


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
        created=_TIMESTAMP,
        owned_by=f"{model.provider} ({model.service} {model.region})",
    )


@router.get(
    "/models",
    summary="OpenAI - /v1/models",
    description="Lists the currently available models",
    response_description="A list of model objects",
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
                                        "owned_by": "Amazon (AWS Bedrock us-east-1)",
                                    },
                                    {
                                        "id": "amazon.titan-embed-text-v2:0",
                                        "object": "model",
                                        "created": 1640995200,
                                        "owned_by": "Amazon (AWS Bedrock eu-west-3)",
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

    Returns a list of currently available models, and provides basic information
    about each one such as the owner and availability. The models are sourced
    from AWS Bedrock across all configured regions and cached for performance.

    Returns:
        ModelsResponse containing list of all available models with metadata

    Raises:
        HTTPException: When unable to retrieve models from backend services (500)
    """
    updated = (await initialize_bedrock_models())[0]
    async with _ALL_MODELS_LOCK:
        if updated or not _ALL_MODELS:
            models = await get_all_models_details()
            _ALL_MODELS.clear()
            _ALL_MODELS.extend(
                format_bedrock_model_to_openai(models[model_id])
                for model_id in sorted(models)
            )
    return log_response_params(ModelsResponse(data=_ALL_MODELS))


@router.get(
    "/models/{model}",
    summary="OpenAI - /v1/models/{model}",
    description="Retrieves a model instance",
    response_description="The model object matching the specified ID",
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
                                "owned_by": "Amazon (AWS Bedrock us-east-1)",
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
        example="amazon.nova-micro-v1:0",
        min_length=1,
        max_length=255,
        str_strip_whitespace=True,
    ),
    _: Annotated[None, Depends(authenticate)] = None,
) -> Model:
    """Retrieve a specific model by its ID from AWS Bedrock.

    Gets detailed information about a specific model from the cached model data.
    If the model cache is empty, it will be populated by querying all configured
    AWS Bedrock regions.

    Args:
        model: The ID of the model to retrieve

    Returns:
        Model object with details about the specified model

    Raises:
        HTTPException: When the model is not found (404)
    """
    log_request_params({"model": model})
    return log_response_params(
        format_bedrock_model_to_openai(await validate_model(model, bedrock_only=False))
    )
