"""Cohere-compatible Rerank API implementation using AWS Bedrock.

This module implements the /v2/rerank endpoint following the Cohere API
specification shape, calling AWS Bedrock rerank models (e.g., Amazon Rerank,
Cohere Rerank) through the Bedrock Rerank API.
"""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends

from stdapi.api_providers.cohere import TAG_COHERE
from stdapi.auth import authenticate
from stdapi.aws_bedrock import get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.models import RERANKING_MODALITY, validate_model
from stdapi.models.capabilities import Capability, register_route_capability
from stdapi.models.rerank import get_rerank_model
from stdapi.monitoring import REQUEST_ID, log_request_params, log_response_params
from stdapi.types.cohere import ApiMeta, BilledUnits
from stdapi.types.cohere_rerank import RerankRequest, RerankResponse, RerankResult

if TYPE_CHECKING:
    from stdapi.types import JsonMapping

register_route_capability(
    "cohere_rerank",
    f"{SETTINGS.cohere_routes_prefix}/v2/rerank",
    "TEXT",
    RERANKING_MODALITY,
    Capability.RERANK,
)

router = APIRouter(
    prefix=f"{SETTINGS.cohere_routes_prefix}/v2", tags=["Rerank", TAG_COHERE]
)


@router.post(
    "/rerank",
    summary="Rank documents by relevance to a query (Cohere format)",
    operation_id="cohere_rerank",
    description=(
        "Ranks the provided documents by semantic relevance to the query "
        "(Cohere v2 Rerank API).\n\n"
        "Returns one result per document (or the `top_n` most relevant ones), "
        "ordered by decreasing `relevance_score`. Ideal as the second stage of "
        "a retrieval pipeline, after a vector or keyword search.\n\n"
        "**Find compatible models:** Call `search_models` with "
        "`mcp_tool=cohere_rerank` to discover model IDs that support reranking."
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
async def rerank(
    request: RerankRequest, _: Annotated[None, Depends(authenticate)] = None
) -> RerankResponse:
    """Rank the provided documents by semantic relevance to the query.

    Args:
        request: Rerank parameters following the Cohere v2 API.

    Returns:
        RerankResponse with results ordered by decreasing relevance.

    Raises:
        ApiError: With 404 if the model does not exist; 400 on unsupported
            options or invalid values.
    """
    log_request_params(request)
    model_id = (await validate_model(request.model, RERANKING_MODALITY)).id
    extra_params = get_extra_model_parameters(model_id, request)
    if request.max_tokens_per_doc is not None:
        extra_params["max_tokens_per_doc"] = request.max_tokens_per_doc
    # A fresh list: `rerank` accepts a mix of str/mapping documents, but
    # `request.documents` is invariantly typed as `list[str]`.
    documents: list[str | JsonMapping] = list(request.documents)
    response = await get_rerank_model(model_id).rerank(
        request.query, documents, top_n=request.top_n, extra_params=extra_params
    )
    return log_response_params(
        RerankResponse(
            id=REQUEST_ID.get(),
            results=[
                RerankResult(index=result.index, relevance_score=result.relevance_score)
                for result in response.results
            ],
            meta=ApiMeta(billed_units=BilledUnits(search_units=response.search_units)),
        ),
        exclude={"results"},
    )
