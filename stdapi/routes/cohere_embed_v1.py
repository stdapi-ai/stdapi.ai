"""Cohere-compatible legacy Embed API implementation using AWS Bedrock.

This module implements the /v1/embed endpoint following the Cohere v1 API
specification shape, calling AWS Bedrock embedding models (e.g., Cohere Embed,
Amazon Titan Embeddings) to compute embedding vectors.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from stdapi.api_providers.cohere import TAG_COHERE
from stdapi.auth import authenticate
from stdapi.aws_bedrock import get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.capabilities import register_route_capability
from stdapi.models.embedding import get_embedding_model
from stdapi.monitoring import REQUEST_ID, log_request_params, log_response_params
from stdapi.types.cohere import ApiMeta, ApiVersion, BilledUnits
from stdapi.types.cohere_embed import (
    EmbeddingsByType,
    EmbedResponse,
    EmbedV1FloatsResponse,
    EmbedV1Request,
    ImageDescription,
)

register_route_capability(
    "cohere_embed_v1", f"{SETTINGS.cohere_routes_prefix}/v1/embed", "TEXT", "EMBEDDING"
)

router = APIRouter(
    prefix=f"{SETTINGS.cohere_routes_prefix}/v1", tags=["Embeddings", TAG_COHERE]
)


@router.post(
    "/embed",
    summary="Generate text embeddings as numeric vectors (Cohere v1 format)",
    operation_id="cohere_embed_v1",
    description=(
        "Creates embedding vector(s) for the input texts or images "
        "(legacy Cohere v1 Embed API).\n\n"
        "Provided for compatibility with older Cohere SDKs and integrations; "
        "new clients should prefer the v2 `cohere_embed` endpoint. Returns the "
        "legacy `embeddings_floats` shape (a plain list of float vectors) "
        "unless `embedding_types` is provided, in which case embeddings are "
        "grouped by type. The optional `input_type` and `truncate` parameters "
        "are applied to Cohere models and ignored for providers without an "
        "equivalent.\n\n"
        "**Find compatible models:** Call `search_models` with "
        "`mcp_tool=cohere_embed_v1` to discover model IDs that support "
        "embeddings."
    ),
    response_description="Embed response.",
    responses={
        200: {"description": "Embeddings successfully created."},
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "texts": {
                            "summary": "Embed texts",
                            "value": {
                                "model": "cohere.embed-multilingual-v3",
                                "input_type": "search_document",
                                "texts": ["Hello world", "Bonjour le monde"],
                            },
                        }
                    }
                }
            }
        }
    },
    response_model_exclude_none=True,
)
async def embed_v1(
    request: EmbedV1Request, _: Annotated[None, Depends(authenticate)] = None
) -> EmbedV1FloatsResponse | EmbedResponse:
    """Create embeddings for the provided texts and/or images (legacy v1 format).

    Args:
        request: Embed parameters following the Cohere v1 API.

    Returns:
        EmbedV1FloatsResponse (legacy `embeddings_floats` shape) when
        `embedding_types` is unset; EmbedResponse grouped by embedding type
        otherwise. Both contain float embedding vectors, one per input item.

    Raises:
        ApiError: With 404 if the model does not exist; 400 on unsupported
            options or invalid values.
    """
    log_request_params(request)
    model_id = (await validate_model(request.model, "EMBEDDING")).id
    extra_params = get_extra_model_parameters(model_id, request)
    if model_id.startswith("cohere."):
        # Cohere-specific body fields; other providers have no equivalent.
        if request.input_type is not None:
            extra_params["input_type"] = request.input_type
        if request.truncate is not None:
            extra_params["truncate"] = request.truncate
    response = await get_embedding_model(model_id).embed_text(
        [*(request.texts or ()), *(request.images or ())],
        dimensions=None,
        extra_params=extra_params,
    )
    meta = ApiMeta(
        api_version=ApiVersion(version="1"),
        billed_units=BilledUnits(
            input_tokens=response.prompt_tokens,
            images=len(request.images) if request.images else None,
        ),
    )
    images = (
        [ImageDescription(**image.model_dump()) for image in response.images]
        if response.images
        else None
    )
    if request.embedding_types is None:
        return log_response_params(
            EmbedV1FloatsResponse(
                id=REQUEST_ID.get(),
                embeddings=response.embeddings,
                texts=request.texts,
                images=images,
                meta=meta,
            ),
            exclude={"embeddings"},
        )
    return log_response_params(
        EmbedResponse(
            id=REQUEST_ID.get(),
            embeddings=EmbeddingsByType(float_=response.embeddings),
            texts=request.texts,
            images=images,
            meta=meta,
        ),
        exclude={"embeddings"},
    )
