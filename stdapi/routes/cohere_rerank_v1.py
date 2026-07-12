"""Cohere-compatible legacy Rerank API implementation using AWS Bedrock.

This module implements the /v1/rerank endpoint following the Cohere v1 API
specification shape, calling AWS Bedrock rerank models (e.g., Amazon Rerank,
Cohere Rerank) through the Bedrock Rerank API.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from stdapi.api_errors import ApiError
from stdapi.api_providers.cohere import TAG_COHERE
from stdapi.auth import authenticate
from stdapi.aws_bedrock import get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.models import RERANKING_MODALITY, validate_model
from stdapi.models.capabilities import Capability, register_route_capability
from stdapi.models.rerank import get_rerank_model
from stdapi.monitoring import REQUEST_ID, log_request_params, log_response_params
from stdapi.types.cohere import ApiMeta, ApiVersion, BilledUnits
from stdapi.types.cohere_rerank import (
    RerankV1Request,
    RerankV1Response,
    RerankV1Result,
    RerankV1ResultDocument,
)

register_route_capability(
    "cohere_rerank_v1",
    f"{SETTINGS.cohere_routes_prefix}/v1/rerank",
    "TEXT",
    RERANKING_MODALITY,
    Capability.RERANK,
)

router = APIRouter(
    prefix=f"{SETTINGS.cohere_routes_prefix}/v1", tags=["Rerank", TAG_COHERE]
)


@router.post(
    "/rerank",
    summary="Rank documents by relevance to a query (Cohere v1 format)",
    operation_id="cohere_rerank_v1",
    description=(
        "Ranks the provided documents by semantic relevance to the query "
        "(legacy Cohere v1 Rerank API).\n\n"
        "Provided for compatibility with older Cohere SDKs and integrations; "
        "new clients should prefer the v2 `cohere_rerank` endpoint. Documents "
        "may be plain strings or objects with a `text` field, and "
        "`return_documents` echoes the documents back in the results.\n\n"
        "**Find compatible models:** Call `search_models` with "
        "`mcp_tool=cohere_rerank_v1` to discover model IDs that support "
        "reranking."
    ),
    response_description="Rerank response.",
    responses={
        200: {"description": "Documents successfully reranked."},
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "rerank": {
                            "summary": "Rerank documents",
                            "value": {
                                "model": "cohere.rerank-v3-5:0",
                                "query": "What is the capital of the United States?",
                                "documents": [
                                    "Carson City is the capital city of Nevada.",
                                    "Washington, D.C. is the capital of the United States.",
                                ],
                                "top_n": 1,
                            },
                        }
                    }
                }
            }
        }
    },
    response_model_exclude_none=True,
)
async def rerank_v1(
    request: RerankV1Request, _: Annotated[None, Depends(authenticate)] = None
) -> RerankV1Response:
    """Rank the provided documents by semantic relevance to the query.

    Args:
        request: Rerank parameters following the Cohere v1 API.

    Returns:
        RerankV1Response with results ordered by decreasing relevance.

    Raises:
        ApiError: With 404 if the model does not exist; 400 on unsupported
            options or invalid values.
    """
    log_request_params(request)
    if request.rank_fields is not None and request.rank_fields != ["text"]:
        msg = (
            "Only the default ['text'] rank_fields is supported on this implementation."
        )
        raise ApiError(msg)
    if request.max_chunks_per_doc is not None:
        msg = "'max_chunks_per_doc' is not supported on this implementation."
        raise ApiError(msg)
    model_id = (await validate_model(request.model, RERANKING_MODALITY)).id
    documents = [
        document if isinstance(document, str) else document.text
        for document in request.documents
    ]
    response = await get_rerank_model(model_id).rerank(
        request.query,
        documents,
        top_n=request.top_n,
        extra_params=get_extra_model_parameters(model_id, request),
    )
    return log_response_params(
        RerankV1Response(
            id=REQUEST_ID.get(),
            results=[
                RerankV1Result(
                    document=(
                        RerankV1ResultDocument(text=documents[result.index])
                        if request.return_documents
                        else None
                    ),
                    index=result.index,
                    relevance_score=result.relevance_score,
                )
                for result in response.results
            ],
            meta=ApiMeta(
                api_version=ApiVersion(version="1"),
                billed_units=BilledUnits(search_units=response.search_units),
            ),
        ),
        exclude={"results"},
    )
