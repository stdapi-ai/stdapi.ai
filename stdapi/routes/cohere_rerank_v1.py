"""Cohere-compatible legacy Rerank API implementation using AWS Bedrock.

This module implements the /v1/rerank endpoint following the Cohere v1 API
specification shape, calling AWS Bedrock rerank models (e.g., Amazon Rerank,
Cohere Rerank) through the Bedrock Rerank API.
"""

from asyncio import gather
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends

from stdapi.api_errors import ApiError
from stdapi.api_providers.cohere import TAG_COHERE
from stdapi.auth import authenticate
from stdapi.aws_bedrock import (
    apply_guardrail_to_text,
    apply_guardrail_to_texts,
    get_extra_model_parameters,
)
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
from stdapi.utils import to_json_str

if TYPE_CHECKING:
    from stdapi.types import JsonMapping

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


def _project_document(
    document: str | JsonMapping, rank_fields: list[str] | None
) -> str | JsonMapping:
    """Reduce an object document to its requested `rank_fields`.

    Args:
        document: A document, either free text or a field->value object.
        rank_fields: Object fields to rank on, or None to rank on every field.

    Returns:
        `document` unchanged (strings, or objects when `rank_fields` is unset),
        or an object containing only the requested fields.

    Raises:
        ApiError: If `rank_fields` matches none of the document's fields.
    """
    if isinstance(document, str) or rank_fields is None:
        return document
    projected = {field: document[field] for field in rank_fields if field in document}
    if not projected:
        msg = f"'rank_fields' {rank_fields} matched none of this document's fields."
        raise ApiError(msg)
    return projected


def _echo_document_text(document: str | JsonMapping) -> str:
    """Render the `text` Cohere echoes back for a `return_documents` result.

    A single-key `{"text": ...}` object echoes its text value verbatim;
    any other object is rendered as one `key: value` line per field (Cohere's
    own join algorithm for object documents is not publicly documented).
    Non-string values are JSON-encoded rather than passed through `str()`, so
    e.g. `True`/`None` render as `true`/`null` instead of Python literals.

    Args:
        document: A document, either free text or a field->value object.

    Returns:
        The text to echo back for this document.
    """
    if isinstance(document, str):
        return document
    text = document.get("text")
    if document.keys() == {"text"} and isinstance(text, str):
        return text
    return "\n".join(
        f"{key}: {value if isinstance(value, str) else to_json_str(value)}"
        for key, value in document.items()
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
        "may be plain strings or field->value objects; `rank_fields` selects "
        "which object fields are ranked on, and `return_documents` echoes the "
        "documents back in the results.\n\n"
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
    if request.max_chunks_per_doc is not None:
        msg = "'max_chunks_per_doc' is not supported on this implementation."
        raise ApiError(msg)
    model_id = (await validate_model(request.model, RERANKING_MODALITY)).id
    # The query and documents are independent AWS guardrail calls; run them concurrently.
    documents, query = await gather(
        apply_guardrail_to_texts(
            [
                _project_document(document, request.rank_fields)
                for document in request.documents
            ],
            source="INPUT",
        ),
        apply_guardrail_to_text(request.query, source="INPUT"),
    )
    response = await get_rerank_model(model_id).rerank(
        query,
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
                        RerankV1ResultDocument(
                            text=_echo_document_text(request.documents[result.index])
                        )
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
