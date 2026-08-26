"""Local Cohere-compatible rerank types (Cohere v1 and v2 Rerank APIs)."""

from pydantic import Field

from stdapi.types import BaseModelRequestWithExtra, BaseModelResponse, JsonMapping
from stdapi.types.cohere import ApiMeta


class RerankRequest(BaseModelRequestWithExtra):
    """Request body for reranking documents against a query."""

    model: str = Field(
        description="ID of the model to use. Wildcard patterns are accepted and "
        "select the most recent matching model.",
        min_length=1,
        max_length=255,
    )
    query: str = Field(description="The search query.", min_length=1)
    documents: list[str] = Field(
        description="A list of texts that will be compared to the query.", min_length=1
    )
    top_n: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Limits the number of returned rerank results. "
            "When unset, all rerank results are returned."
        ),
    )
    max_tokens_per_doc: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Truncate long documents to this number of tokens. "
            "Supported by some models only."
        ),
    )
    priority: int | None = Field(
        default=None,
        description=(
            "Accepted for compatibility and ignored. Cohere API request "
            "scheduling priority is not applicable on this implementation."
        ),
    )
    return_documents: bool | None = Field(
        default=None,
        description=(
            "Accepted for compatibility with the v1 API and ignored: v2 "
            "results always reference the input documents by `index` only."
        ),
    )


class RerankV1Request(BaseModelRequestWithExtra):
    """Request body for reranking documents against a query (Cohere v1 Rerank API)."""

    model: str = Field(
        description="ID of the model to use. Wildcard patterns are accepted and "
        "select the most recent matching model.",
        min_length=1,
        max_length=255,
    )
    query: str = Field(description="The search query.", min_length=1)
    documents: list[str | JsonMapping] = Field(
        description=(
            "A list of texts, or field->value objects, that will be compared "
            "to the query. `rank_fields` selects which object fields are "
            "ranked on."
        ),
        min_length=1,
    )
    top_n: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Limits the number of returned rerank results. "
            "When unset, all rerank results are returned."
        ),
    )
    rank_fields: list[str] | None = Field(
        default=None,
        description=(
            "Object document fields to rank on. When unset, object documents "
            "are ranked on all of their fields."
        ),
    )
    return_documents: bool = Field(
        default=False,
        description=(
            "When true, each result echoes back the text of the corresponding "
            "input document."
        ),
    )
    max_chunks_per_doc: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum number of chunks to produce per document. "
            "Not supported on this implementation."
        ),
    )


class RerankV1ResultDocument(BaseModelResponse):
    """Echoed input document of a v1 rerank result."""

    text: str = Field(description="The text of the reranked document.")


class RerankV1Result(BaseModelResponse):
    """A single reranked document reference (v1 format)."""

    document: RerankV1ResultDocument | None = Field(
        default=None,
        description="The input document, echoed back when `return_documents` is true.",
    )
    index: int = Field(
        description="Index of the document in the request's `documents` list."
    )
    relevance_score: float = Field(
        description="Relevance score of the document for the query, in [0, 1]."
    )


class RerankV1Response(BaseModelResponse):
    """Rerank response model (v1 format)."""

    id: str = Field(description="Unique identifier of the request.")
    results: list[RerankV1Result] = Field(
        description="Documents ordered by decreasing relevance to the query."
    )
    meta: ApiMeta = Field(description="Response metadata.")


class RerankResult(BaseModelResponse):
    """A single reranked document reference."""

    index: int = Field(
        description="Index of the document in the request's `documents` list."
    )
    relevance_score: float = Field(
        description="Relevance score of the document for the query, in [0, 1]."
    )


class RerankResponse(BaseModelResponse):
    """Rerank response model."""

    id: str = Field(description="Unique identifier of the request.")
    results: list[RerankResult] = Field(
        description="Documents ordered by decreasing relevance to the query."
    )
    meta: ApiMeta = Field(description="Response metadata.")
