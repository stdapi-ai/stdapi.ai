"""OpenAI-compatible Embeddings API implementation using AWS Bedrock.

This module implements the /v1/embeddings endpoint following the OpenAI API
specification shape, calling AWS Bedrock embedding models (e.g., Amazon Titan
Embeddings, Cohere Embed v3) to compute embedding vectors.
"""

from array import array
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from stdapi.auth import authenticate
from stdapi.aws_bedrock import get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.embedding import get_embedding_model
from stdapi.monitoring import REQUEST_LOG, log_request_params, log_response_params
from stdapi.types.openai_embeddings import (
    CreateEmbeddingResponse,
    Embedding,
    EmbeddingCreateParams,
    Usage,
)
from stdapi.utils import b64encode

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=["embeddings", "openai"]
)


@router.post(
    "/embeddings",
    summary="OpenAI - /v1/embeddings",
    description="Creates an embedding vector representing the input text.",
    response_description="Embedding list response in OpenAI format",
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
                        "single": {
                            "summary": "Single input",
                            "value": {
                                "model": "amazon.titan-embed-text-v2:0",
                                "input": "Hello world",
                            },
                        },
                        "batch": {
                            "summary": "Batch input",
                            "value": {
                                "model": "amazon.titan-embed-text-v2:0",
                                "input": ["first", "second"],
                            },
                        },
                    }
                }
            }
        }
    },
    response_model_exclude_none=True,
)
async def create_embeddings(
    request: EmbeddingCreateParams,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: Annotated[None, Depends(authenticate)] = None,
) -> CreateEmbeddingResponse:
    """Create embeddings for the provided input strings.

    Args:
        request: Embedding creation parameters following OpenAI API.
        background_tasks: FastAPI background tasks.

    Returns:
        EmbeddingListResponse containing embedding vectors, one per input item.

    Raises:
        HTTPException: With 404 if the model does not exist; 400 on unsupported
            options or invalid values.
    """
    log_request_params(request)
    if request.user:
        log = REQUEST_LOG.get()
        log["request_user_id"] = request.user

    await validate_model(request.model, "EMBEDDING")
    if isinstance(request.input, str):
        input_texts: list[str] = [request.input]
    elif isinstance(request.input[0], str):
        input_texts = list(request.input)
    else:  # pragma: no cover
        raise HTTPException(status_code=400, detail="Unsupported input type.")

    response = await get_embedding_model(request.model).embed_text(
        input_texts,
        dimensions=request.dimensions,
        extra_params=get_extra_model_parameters(request.model, request),
        background_tasks=background_tasks,
    )
    b64_embedding = request.encoding_format == "base64"
    return log_response_params(
        CreateEmbeddingResponse(
            object="list",
            data=[
                Embedding(
                    object="embedding",
                    index=index,
                    embedding=(
                        await b64encode(array("f", vector).tobytes())
                        if b64_embedding
                        else vector
                    ),
                )
                for index, vector in enumerate(response.embeddings)
            ],
            model=request.model,
            usage=Usage(
                prompt_tokens=response.prompt_tokens,
                total_tokens=response.total_tokens or response.prompt_tokens,
            ),
        ),
        exclude={"data"},
    )
