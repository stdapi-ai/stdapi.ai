"""Anthropic Models API endpoint implementation.

This module implements Anthropic-compatible endpoints for the Models API,
providing AWS Bedrock integration while maintaining API compatibility.

Functions:
    list_models: List available models.
    retrieve_model: Retrieve a specific model by ID.
"""

from asyncio import Lock
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from stdapi.api_providers.anthropic import TAG_ANTHROPIC
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.models import (
    ModelDetails,
    get_all_models_details,
    initialize_bedrock_models,
    validate_model,
)
from stdapi.monitoring import log_request_params, log_response_params
from stdapi.types.anthropic_messages import ModelInfo, ModelListResponse

if SETTINGS.anthropic_routes_prefix != SETTINGS.openai_routes_prefix:
    router: APIRouter = APIRouter(
        prefix=f"{SETTINGS.anthropic_routes_prefix}/v1", tags=["Models", TAG_ANTHROPIC]
    )

    #: /v1/models route response cache
    _ALL_MODELS: list[ModelInfo] = []
    _ALL_MODELS_LOCK = Lock()

    def format_bedrock_model_to_anthropic(model: ModelDetails) -> ModelInfo:
        """Format a Bedrock model to Anthropic API model format.

        Args:
            model: Bedrock foundation model summary object

        Returns:
            ModelInfo object formatted according to Anthropic API specification
        """
        return ModelInfo(
            id=model.id,
            created_at=(
                model.start_of_life_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                if model.start_of_life_time
                else "1970-01-01T00:00:00Z"
            ),
            display_name=model.name,
            type="model",
        )

    @router.get(
        "/models",
        summary="List available text generation models (Anthropic format)",
        operation_id="anthropic_model_list",
        description=(
            "Lists all available text generation models with display name and creation date "
            "(Anthropic Models API). Only models that support text input and output are included.\n\n"
            "**Agent note:** For richer filtering — by modality, route, MCP tool, region, or "
            "legacy status — use `search_models` instead."
        ),
        response_description="A paginated list of available models.",
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
                                    "data": [
                                        {
                                            "id": "amazon.nova-micro-v1:0",
                                            "type": "model",
                                            "display_name": "Amazon Nova Micro",
                                            "created_at": "2025-01-01T00:00:00Z",
                                        }
                                    ],
                                    "has_more": False,
                                    "first_id": "amazon.nova-micro-v1:0",
                                    "last_id": "amazon.nova-micro-v1:0",
                                },
                            }
                        }
                    }
                },
            }
        },
    )
    async def list_models(
        _: Annotated[None, Depends(authenticate)] = None,
        limit: Annotated[
            int, Query(ge=1, le=1000, description="Number of items to return per page.")
        ] = 1000,
        after_id: Annotated[
            str | None,
            Query(
                alias="after_id",
                description="ID of the object to use as a cursor for pagination. When provided, returns the page of results immediately after this object.",
            ),
        ] = None,
        before_id: Annotated[
            str | None,
            Query(
                alias="before_id",
                description="ID of the object to use as a cursor for pagination. When provided, returns the page of results immediately before this object.",
            ),
        ] = None,
    ) -> ModelListResponse:
        """List available models.

        Returns a paginated list of available models with metadata.

        Returns:
            ModelListResponse containing list of available models.

        Raises:
            ApiError: When unable to retrieve models from backend services (500)
        """
        updated = await initialize_bedrock_models()
        async with _ALL_MODELS_LOCK:
            if updated or not _ALL_MODELS:
                models = await get_all_models_details()
                _ALL_MODELS.clear()
                _ALL_MODELS.extend(
                    format_bedrock_model_to_anthropic(models[model_id])
                    for model_id in sorted(models)
                    if "TEXT" in models[model_id].input_modalities
                    and "TEXT" in models[model_id].output_modalities
                )

        data = _ALL_MODELS
        match (after_id, before_id):
            case (str() as aid, None):
                if (
                    idx := next((i for i, m in enumerate(data) if m.id == aid), None)
                ) is not None:
                    data = data[idx + 1 :]
            case (None, str() as bid):
                if (
                    idx := next((i for i, m in enumerate(data) if m.id == bid), None)
                ) is not None:
                    data = data[:idx]
            case _:
                pass
        has_more = len(data) > limit
        data = data[:limit]

        return log_response_params(
            ModelListResponse(
                data=data,
                has_more=has_more,
                first_id=data[0].id if data else None,
                last_id=data[-1].id if data else None,
            )
        )

    @router.get(
        "/models/{model_id}",
        summary="Retrieve details for a specific model by ID (Anthropic format)",
        operation_id="anthropic_model_get",
        description=(
            "Retrieves metadata (display name, creation date) for a single model by ID "
            "(Anthropic Models API).\n\n"
            "**Agent note:** Use `search_models` to look up modalities, supported routes, regions, "
            "and other extended metadata not available here."
        ),
        response_description="Model information.",
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
                                    "type": "model",
                                    "display_name": "Amazon Nova Micro",
                                    "created_at": "2025-01-01T00:00:00Z",
                                },
                            }
                        }
                    }
                },
            },
            404: {"description": "Model not found"},
        },
        response_model_exclude_none=True,
    )
    async def retrieve_model(
        model_id: str = Path(  # noqa: FAST002
            ...,
            description="The ID of the model to retrieve.",
            examples=["amazon.nova-micro-v1:0"],
            min_length=1,
            max_length=255,
            str_strip_whitespace=True,
        ),
        _: Annotated[None, Depends(authenticate)] = None,
    ) -> ModelInfo:
        """Retrieve a specific model by its ID.

        Gets detailed information about a specific model.

        Args:
            model_id: The ID of the model to retrieve.

        Returns:
            ModelInfo object with details about the specified model.

        Raises:
            ApiError: When the model is not found (404)
        """
        log_request_params({"model_id": model_id})
        return log_response_params(
            format_bedrock_model_to_anthropic(
                await validate_model(model_id, bedrock_only=False)
            )
        )
