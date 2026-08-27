"""Ollama-compatible embedding endpoints using AWS Bedrock embedding models.

- POST /api/embed — embed one or several inputs
- POST /api/embeddings — embed one prompt (deprecated upstream, still served)
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from stdapi.api_providers.ollama import TAG_OLLAMA
from stdapi.auth import authenticate
from stdapi.aws_bedrock import apply_guardrail_to_texts, get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.capabilities import register_route_capability
from stdapi.models.embedding import get_embedding_model
from stdapi.monitoring import log_request_params, log_response_params
from stdapi.types.ollama import (
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbedRequest,
    EmbedResponse,
    total_duration,
)

register_route_capability(
    "ollama_embed",
    f"{SETTINGS.ollama_routes_prefix}/api/embed",
    "TEXT",
    "EMBEDDING",
    mcp_tool=False,
)
register_route_capability(
    "ollama_embeddings",
    f"{SETTINGS.ollama_routes_prefix}/api/embeddings",
    "TEXT",
    "EMBEDDING",
    mcp_tool=False,
)

router = APIRouter(
    prefix=f"{SETTINGS.ollama_routes_prefix}/api", tags=["Embeddings", TAG_OLLAMA]
)


@router.post(
    "/embed",
    summary="Generate text embeddings as numeric vectors (Ollama format)",
    operation_id="ollama_embed",
    description=(
        "Creates embedding vector(s) for the input text (Ollama Embed API).\n\n"
        "Accepts a single string or an array of strings and returns one vector "
        "per input, in request order. `dimensions` selects the vector width on "
        "models that support it. `truncate`, `keep_alive` and `options` have no "
        "effect and are accepted and ignored.\n\n"
        "Use the model names `ollama_tags` publishes."
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
                        "batch": {
                            "summary": "Embed texts",
                            "value": {
                                "model": "amazon.titan-embed-text-v2:0",
                                "input": ["first", "second"],
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
    """Create embeddings for the provided input.

    Args:
        request: Embed parameters following the Ollama API.

    Returns:
        EmbedResponse holding one vector per input, in request order.

    Raises:
        ApiError: With 404 if the model does not exist; 400 on unsupported
            options or invalid values.
    """
    log_request_params(request)
    model_id = (
        await validate_model(request.model, "EMBEDDING", route="ollama_embed")
    ).id
    inputs = request.input if isinstance(request.input, list) else [request.input]
    response = await get_embedding_model(model_id).embed_text(
        await apply_guardrail_to_texts(inputs, source="INPUT"),
        dimensions=request.dimensions,
        extra_params=get_extra_model_parameters(model_id, request),
    )
    return log_response_params(
        EmbedResponse(
            model=request.model,
            embeddings=response.embeddings,
            total_duration=total_duration(),
            prompt_eval_count=response.prompt_tokens,
        ),
        exclude={"embeddings"},
    )


@router.post(
    "/embeddings",
    summary="Generate a text embedding as a numeric vector (Ollama legacy format)",
    operation_id="ollama_embeddings",
    description=(
        "Creates one embedding vector for `prompt` (Ollama legacy Embeddings "
        "API).\n\nSuperseded upstream by `ollama_embed`, which embeds several "
        "inputs at once and reports token usage. Kept for clients that still "
        "call it."
    ),
    response_description="Embedding response.",
    responses={
        200: {"description": "The embedding was created."},
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
    },
    deprecated=True,
    response_model_exclude_none=True,
)
async def embeddings(
    request: EmbeddingsRequest, _: Annotated[None, Depends(authenticate)] = None
) -> EmbeddingsResponse:
    """Create a single embedding for the provided prompt.

    Args:
        request: Legacy embedding parameters following the Ollama API.

    Returns:
        EmbeddingsResponse holding the vector for the prompt.

    Raises:
        ApiError: With 404 if the model does not exist; 400 on unsupported
            options or invalid values.
    """
    log_request_params(request)
    model_id = (
        await validate_model(request.model, "EMBEDDING", route="ollama_embeddings")
    ).id
    response = await get_embedding_model(model_id).embed_text(
        await apply_guardrail_to_texts([request.prompt], source="INPUT"),
        dimensions=None,
        extra_params=get_extra_model_parameters(model_id, request),
    )
    return log_response_params(
        EmbeddingsResponse(embedding=response.embeddings[0]), exclude={"embedding"}
    )
