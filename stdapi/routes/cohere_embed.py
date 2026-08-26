"""Cohere-compatible ``/v2/embed`` endpoint using AWS Bedrock embedding models."""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends

from stdapi.api_providers.cohere import TAG_COHERE
from stdapi.auth import authenticate
from stdapi.aws_bedrock import apply_guardrail_to_texts, get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.capabilities import register_route_capability
from stdapi.models.embedding import get_embedding_model
from stdapi.monitoring import REQUEST_ID, log_request_params, log_response_params
from stdapi.types.cohere import ApiMeta, BilledUnits
from stdapi.types.cohere_embed import (
    TITAN_EMBED_V2_PREFIX,
    EmbedRequest,
    EmbedResponse,
    ImageDescription,
    build_embeddings_by_type,
    resolve_embedding_types,
)

if TYPE_CHECKING:
    from stdapi.models.embedding import EmbedInputValue

register_route_capability(
    "cohere_embed", f"{SETTINGS.cohere_routes_prefix}/v2/embed", "TEXT", "EMBEDDING"
)

router = APIRouter(
    prefix=f"{SETTINGS.cohere_routes_prefix}/v2", tags=["Embeddings", TAG_COHERE]
)


@router.post(
    "/embed",
    summary="Generate text embeddings as numeric vectors (Cohere format)",
    operation_id="cohere_embed",
    description=(
        "Creates embedding vector(s) for the input texts, images, or mixed "
        "text and image `inputs` (Cohere v2 Embed API).\n\n"
        "Returns fixed-dimensional float vectors suitable for semantic search, "
        "clustering, and retrieval-augmented generation, one per input and in "
        "request order. Works with every available embedding model; the "
        "Cohere-specific `input_type`, `truncate`, and `max_tokens` parameters "
        "are applied to Cohere models and ignored for providers without an "
        "equivalent.\n\n"
        "**Find compatible models:** Call `search_models` with "
        "`route=cohere_embed` to discover model IDs that support embeddings."
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
async def embed(
    request: EmbedRequest, _: Annotated[None, Depends(authenticate)] = None
) -> EmbedResponse:
    """Create embeddings for the provided texts, images and fused inputs.

    Args:
        request: Embed parameters following the Cohere v2 API.

    Returns:
        EmbedResponse containing float embedding vectors, one per input item,
        in request order: texts, then images, then `inputs` entries.

    Raises:
        ApiError: With 404 if the model does not exist; 400 on unsupported
            options or invalid values.
    """
    log_request_params(request)
    model_id = (
        await validate_model(request.model, "EMBEDDING", route="cohere_embed")
    ).id
    extra_params = get_extra_model_parameters(model_id, request)
    native_embedding_types = resolve_embedding_types(model_id, request.embedding_types)
    if model_id.startswith("cohere."):
        # Cohere-specific body fields; other providers have no equivalent.
        extra_params["input_type"] = request.input_type
        if request.truncate is not None:
            extra_params["truncate"] = request.truncate
        if request.max_tokens is not None:
            extra_params["max_tokens"] = request.max_tokens
        if native_embedding_types is not None:
            extra_params["embedding_types"] = list(native_embedding_types)
    elif (
        model_id.startswith(TITAN_EMBED_V2_PREFIX)
        and native_embedding_types is not None
    ):
        extra_params["embeddingTypes"] = list(native_embedding_types)
    entries = request.embed_inputs()
    # Regrouped after screening, so a fused input keeps the grouping it was sent with.
    screened = iter(
        await apply_guardrail_to_texts(
            [
                part
                for entry in entries
                for part in (entry if isinstance(entry, list) else (entry,))
            ],
            source="INPUT",
        )
    )
    inputs: list[EmbedInputValue] = [
        [next(screened) for _ in entry] if isinstance(entry, list) else next(screened)
        for entry in entries
    ]
    response = await get_embedding_model(model_id).embed_text(
        inputs, dimensions=request.output_dimension, extra_params=extra_params
    )
    return log_response_params(
        EmbedResponse(
            id=REQUEST_ID.get(),
            embeddings=build_embeddings_by_type(response, request.embedding_types),
            texts=request.texts,
            images=(
                [ImageDescription(**image.model_dump()) for image in response.images]
                if response.images
                else None
            ),
            meta=ApiMeta(
                billed_units=BilledUnits(
                    input_tokens=response.prompt_tokens,
                    images=request.count_images() or None,
                )
            ),
        ),
        exclude={"embeddings"},
    )
