"""Local OpenAI-compatible Vector Stores API types."""

from typing import Annotated, Literal, Self

from pydantic import (
    Discriminator,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from stdapi.types import (
    FILE_ID_PATTERN,
    BaseModelRequest,
    BaseModelResponse,
    JsonMapping,
)

#: Value types an ``attributes`` entry accepts.
AttributeValue = bool | float | str

#: Smallest ``max_chunk_size_tokens`` the upstream API accepts.
CHUNK_SIZE_TOKENS_MIN: int = 100

#: Largest ``max_chunk_size_tokens`` the upstream API accepts.
CHUNK_SIZE_TOKENS_MAX: int = 4096

#: Default ``max_chunk_size_tokens`` of the ``auto`` chunking strategy.
CHUNK_SIZE_TOKENS_DEFAULT: int = 800

#: Default ``chunk_overlap_tokens`` of the ``auto`` chunking strategy.
CHUNK_OVERLAP_TOKENS_DEFAULT: int = 400

#: Largest number of files one file batch accepts.
FILE_BATCH_MAX_FILES: int = 2000

#: Largest number of key-value pairs an ``attributes`` or ``metadata`` mapping holds.
MAPPING_MAX_PAIRS: int = 16

#: Largest number of queries one search request embeds and runs.
SEARCH_QUERIES_MAX: int = 16

#: Comparison operators the vector index applies to numbers only.
_ORDERING_OPERATORS: frozenset[str] = frozenset({"gt", "gte", "lt", "lte"})

#: Description shared by every ``attributes`` field.
_ATTRIBUTES_DESCRIPTION = (
    "Up to 16 key-value pairs stored with the file and usable in search `filters`. "
    "Keys are at most 64 characters; values are strings of at most 512 characters, "
    "numbers, or booleans."
)

#: Description shared by every ``metadata`` field.
_METADATA_DESCRIPTION = (
    "Up to 16 key-value pairs stored with the vector store. "
    "Keys are at most 64 characters and values at most 512 characters."
)

#: Description shared by the ``file_id`` fields.
_FILE_ID_DESCRIPTION = "The ID of an uploaded file the vector store should index."

#: ``attributes`` mapping, with the upstream count, key and value bounds enforced.
Attributes = Annotated[
    dict[
        Annotated[str, Field(max_length=64)],
        bool | float | Annotated[str, Field(max_length=512)],
    ],
    Field(max_length=MAPPING_MAX_PAIRS),
]

#: ``metadata`` mapping, with the upstream count, key and value bounds enforced.
Metadata = Annotated[
    dict[Annotated[str, Field(max_length=64)], Annotated[str, Field(max_length=512)]],
    Field(max_length=MAPPING_MAX_PAIRS),
]


class StaticChunkingConfig(BaseModelRequest):
    """Chunk size and overlap, in tokens."""

    max_chunk_size_tokens: int = Field(
        default=CHUNK_SIZE_TOKENS_DEFAULT,
        ge=CHUNK_SIZE_TOKENS_MIN,
        le=CHUNK_SIZE_TOKENS_MAX,
        description="Maximum number of tokens in each chunk (100 to 4096).",
    )
    chunk_overlap_tokens: int = Field(
        default=CHUNK_OVERLAP_TOKENS_DEFAULT,
        ge=0,
        description=(
            "Number of tokens shared between consecutive chunks; "
            "must not exceed half of `max_chunk_size_tokens`."
        ),
    )

    @model_validator(mode="after")
    def _check_overlap(self) -> Self:
        """Enforce the upstream rule that the overlap is at most half a chunk.

        Returns:
            The validated configuration.

        Raises:
            ValueError: When the overlap exceeds half of ``max_chunk_size_tokens``.
        """
        if self.chunk_overlap_tokens > self.max_chunk_size_tokens // 2:
            msg = "chunk_overlap_tokens must not exceed half of max_chunk_size_tokens"
            raise ValueError(msg)
        return self


class AutoChunkingStrategyParam(BaseModelRequest):
    """Default chunking strategy: 800-token chunks overlapping by 400 tokens."""

    type: Literal["auto"] = Field(description="Selects the default chunking strategy.")


class StaticChunkingStrategyParam(BaseModelRequest):
    """Chunking strategy with a caller-chosen chunk size and overlap."""

    type: Literal["static"] = Field(description="Selects an explicit chunk size.")
    static: StaticChunkingConfig = Field(description="Chunk size and overlap.")


#: Chunking strategy accepted on a request.
ChunkingStrategyParam = Annotated[
    AutoChunkingStrategyParam | StaticChunkingStrategyParam, Discriminator("type")
]


class StaticChunkingStrategy(BaseModelResponse):
    """The chunk size and overlap a file was indexed with."""

    type: Literal["static"] = Field(default="static", description="Always `static`.")
    static: StaticChunkingConfig = Field(description="Chunk size and overlap.")


class ExpiresAfter(BaseModelResponse):
    """Expiration policy of a vector store."""

    anchor: Literal["last_active_at"] = Field(
        default="last_active_at",
        description="Timestamp the expiration is counted from.",
    )
    days: int = Field(
        ge=1, description="Number of days after the anchor before the store expires."
    )


class FileCounts(BaseModelResponse):
    """Per-status counts of the files attached to a vector store."""

    in_progress: int = Field(description="Files still being processed.")
    completed: int = Field(description="Files successfully processed.")
    failed: int = Field(description="Files that failed to process.")
    cancelled: int = Field(description="Files whose processing was cancelled.")
    total: int = Field(description="Total number of files.")


class VectorStore(BaseModelResponse):
    """A searchable collection of indexed files."""

    id: str = Field(description="The vector store identifier.")
    object: Literal["vector_store"] = Field(
        default="vector_store", description="The object type, always `vector_store`."
    )
    created_at: int = Field(description="Unix timestamp (seconds) of creation.")
    name: str = Field(description="The name of the vector store.")
    description: str | None = Field(
        default=None, description="The description of the vector store."
    )
    usage_bytes: int = Field(description="Bytes of indexed content in the store.")
    file_counts: FileCounts = Field(description="Per-status file counts.")
    status: Literal["expired", "in_progress", "completed"] = Field(
        description="`completed` once every attached file finished processing."
    )
    expires_after: ExpiresAfter | None = Field(
        default=None, description="The expiration policy, when one is set."
    )
    expires_at: int | None = Field(
        default=None, description="Unix timestamp (seconds) the store expires at."
    )
    last_active_at: int | None = Field(
        default=None, description="Unix timestamp (seconds) of the last activity."
    )
    metadata: Metadata | None = Field(
        default=None, description="Caller-supplied key-value pairs."
    )


class VectorStoreDeleted(BaseModelResponse):
    """Confirmation that a vector store was deleted."""

    id: str = Field(description="The vector store identifier.")
    object: Literal["vector_store.deleted"] = Field(
        default="vector_store.deleted", description="The object type."
    )
    deleted: bool = Field(description="Whether the vector store was deleted.")


class ListVectorStoresResponse(BaseModelResponse):
    """One page of vector stores."""

    object: Literal["list"] = Field(default="list", description="Always `list`.")
    data: list[VectorStore] = Field(description="The vector stores in this page.")
    first_id: str | None = Field(
        default=None, description="ID of the first store in the page."
    )
    last_id: str | None = Field(
        default=None, description="ID of the last store in the page."
    )
    has_more: bool = Field(description="Whether more stores follow this page.")


class LastError(BaseModelResponse):
    """Why a file failed to be indexed."""

    code: Literal["server_error", "unsupported_file", "invalid_file"] = Field(
        description="The error category."
    )
    message: str = Field(description="Human-readable description of the failure.")


class VectorStoreFile(BaseModelResponse):
    """A file attached to a vector store."""

    id: str = Field(description="The file identifier.")
    object: Literal["vector_store.file"] = Field(
        default="vector_store.file", description="The object type."
    )
    created_at: int = Field(description="Unix timestamp (seconds) of the attachment.")
    usage_bytes: int = Field(description="Bytes of indexed content for this file.")
    vector_store_id: str = Field(description="The vector store the file belongs to.")
    status: Literal["in_progress", "completed", "cancelled", "failed"] = Field(
        description="`completed` once the file is searchable."
    )
    last_error: LastError | None = Field(
        default=None, description="The failure that stopped processing, if any."
    )
    chunking_strategy: StaticChunkingStrategy | None = Field(
        default=None, description="The chunk size and overlap used for this file."
    )
    attributes: Attributes | None = Field(
        default=None, description="Caller-supplied key-value pairs."
    )


class VectorStoreFileDeleted(BaseModelResponse):
    """Confirmation that a file was detached from a vector store."""

    id: str = Field(description="The file identifier.")
    object: Literal["vector_store.file.deleted"] = Field(
        default="vector_store.file.deleted", description="The object type."
    )
    deleted: bool = Field(description="Whether the file was detached.")


class ListVectorStoreFilesResponse(BaseModelResponse):
    """One page of vector store files."""

    object: Literal["list"] = Field(default="list", description="Always `list`.")
    data: list[VectorStoreFile] = Field(description="The files in this page.")
    first_id: str | None = Field(
        default=None, description="ID of the first file in the page."
    )
    last_id: str | None = Field(
        default=None, description="ID of the last file in the page."
    )
    has_more: bool = Field(description="Whether more files follow this page.")


class VectorStoreFileBatch(BaseModelResponse):
    """A group of files attached to a vector store in one request."""

    id: str = Field(description="The file batch identifier.")
    object: Literal["vector_store.file_batch"] = Field(
        default="vector_store.file_batch", description="The object type."
    )
    created_at: int = Field(description="Unix timestamp (seconds) of creation.")
    vector_store_id: str = Field(description="The vector store the batch belongs to.")
    status: Literal["in_progress", "completed", "cancelled", "failed"] = Field(
        description="`completed` once every file in the batch finished processing."
    )
    file_counts: FileCounts = Field(description="Per-status file counts.")


class VectorStoreCreateParams(BaseModelRequest):
    """Request body of `POST /v1/vector_stores`."""

    name: str | None = Field(
        default=None, max_length=256, description="A name for the vector store."
    )
    description: str | None = Field(
        default=None,
        max_length=512,
        description="A description of what the vector store holds.",
    )
    file_ids: list[Annotated[str, Field(pattern=FILE_ID_PATTERN)]] | None = Field(
        default=None,
        max_length=FILE_BATCH_MAX_FILES,
        description=(
            "IDs of uploaded files to index into the new store. "
            "Indexing runs in the background; poll the store until its `status` "
            "is `completed`."
        ),
    )
    chunking_strategy: ChunkingStrategyParam | None = Field(
        default=None,
        description=(
            "How the files are split before indexing. Applies to `file_ids` and "
            "to every file later attached without its own strategy."
        ),
    )
    expires_after: ExpiresAfter | None = Field(
        default=None,
        description="Expire the store this many days after its last activity.",
    )
    metadata: Metadata | None = Field(default=None, description=_METADATA_DESCRIPTION)


class VectorStoreUpdateParams(BaseModelRequest):
    """Request body of `POST /v1/vector_stores/{vector_store_id}`."""

    name: str | None = Field(
        default=None, max_length=256, description="A new name for the vector store."
    )
    expires_after: ExpiresAfter | None = Field(
        default=None,
        description="A new expiration policy; `null` removes the current one.",
    )
    metadata: Metadata | None = Field(default=None, description=_METADATA_DESCRIPTION)


class VectorStoreFileCreateParams(BaseModelRequest):
    """Request body of `POST /v1/vector_stores/{vector_store_id}/files`."""

    file_id: str = Field(pattern=FILE_ID_PATTERN, description=_FILE_ID_DESCRIPTION)
    attributes: Attributes | None = Field(
        default=None, description=_ATTRIBUTES_DESCRIPTION
    )
    chunking_strategy: ChunkingStrategyParam | None = Field(
        default=None, description="How this file is split before indexing."
    )


class VectorStoreFileUpdateParams(BaseModelRequest):
    """Request body of `POST /v1/vector_stores/{vector_store_id}/files/{file_id}`."""

    attributes: Attributes | None = Field(description=_ATTRIBUTES_DESCRIPTION)


class FileBatchFile(BaseModelRequest):
    """One entry of a file batch, with its own attributes and chunking."""

    file_id: str = Field(pattern=FILE_ID_PATTERN, description=_FILE_ID_DESCRIPTION)
    attributes: Attributes | None = Field(
        default=None, description=_ATTRIBUTES_DESCRIPTION
    )
    chunking_strategy: ChunkingStrategyParam | None = Field(
        default=None, description="How this file is split before indexing."
    )


class FileBatchCreateParams(BaseModelRequest):
    """Request body of `POST /v1/vector_stores/{vector_store_id}/file_batches`."""

    file_ids: list[Annotated[str, Field(pattern=FILE_ID_PATTERN)]] | None = Field(
        default=None,
        max_length=FILE_BATCH_MAX_FILES,
        description=(
            "IDs of uploaded files to index, at most 2000. "
            "Mutually exclusive with `files`."
        ),
    )
    files: list[FileBatchFile] | None = Field(
        default=None,
        max_length=FILE_BATCH_MAX_FILES,
        description=(
            "Files to index with per-file `attributes` or `chunking_strategy`, "
            "at most 2000. Mutually exclusive with `file_ids`."
        ),
    )
    attributes: Attributes | None = Field(
        default=None,
        description=f"{_ATTRIBUTES_DESCRIPTION} Applied to every file in `file_ids`.",
    )
    chunking_strategy: ChunkingStrategyParam | None = Field(
        default=None,
        description="How every file in `file_ids` is split before indexing.",
    )

    @model_validator(mode="after")
    def _check_files(self) -> Self:
        """Enforce that exactly one of ``file_ids`` and ``files`` is supplied.

        Returns:
            The validated parameters.

        Raises:
            ValueError: When both or neither are supplied.
        """
        if bool(self.file_ids) == bool(self.files):
            msg = "Exactly one of 'file_ids' and 'files' must be provided"
            raise ValueError(msg)
        return self


class ComparisonFilter(BaseModelRequest):
    """Compares one attribute key against a value."""

    key: str = Field(description="The attribute key to compare.")
    type: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"] = Field(
        description="The comparison operator."
    )
    value: AttributeValue | list[float | str] = Field(
        description=(
            "The value compared against; a number for `gt`, `gte`, `lt` and "
            "`lte`, a list for `in` and `nin`."
        )
    )

    @field_validator("value")
    @classmethod
    def _check_value(
        cls, value: AttributeValue | list[float | str], info: ValidationInfo
    ) -> AttributeValue | list[float | str]:
        """Enforce that the ordering operators compare numbers.

        The index orders numbers only, so a string bound would fail in the
        backend instead of being reported as the invalid request it is. Placed
        on the field rather than the model so the reported error names it.

        Args:
            value: The value the filter compares against.
            info: Validation context, carrying the already-validated operator.

        Returns:
            The validated value.

        Raises:
            ValueError: When an ordering operator is given a non-numeric value.
        """
        numeric = isinstance(value, float | int) and not isinstance(value, bool)
        if info.data.get("type") in _ORDERING_OPERATORS and not numeric:
            msg = f"'{info.data['type']}' compares numbers, so it needs a number"
            raise ValueError(msg)
        return value


class CompoundFilter(BaseModelRequest):
    """Combines several filters."""

    type: Literal["and", "or"] = Field(description="How the filters are combined.")
    # One level deep: a self-referential request schema cannot be published.
    filters: list[ComparisonFilter | JsonMapping] = Field(
        min_length=1,
        description=(
            "The filters to combine; an entry may itself be a compound filter."
        ),
    )


#: A search filter: one comparison, or a combination of filters.
SearchFilter = ComparisonFilter | CompoundFilter


class RankingOptions(BaseModelRequest):
    """Result ranking options."""

    ranker: str | None = Field(
        default=None,
        description=(
            "Reserved for a re-ranking model. Results are always ranked by "
            "semantic similarity, so this value does not change the ranking."
        ),
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Drop results scoring below this value. Rejected by a vector store "
            "whose scores are not comparable between searches; use "
            "`max_num_results` there."
        ),
    )


class VectorStoreSearchParams(BaseModelRequest):
    """Request body of `POST /v1/vector_stores/{vector_store_id}/search`."""

    query: (
        Annotated[str, Field(min_length=1)]
        | Annotated[
            list[Annotated[str, Field(min_length=1)]],
            Field(min_length=1, max_length=SEARCH_QUERIES_MAX),
        ]
    ) = Field(
        description="The text to search for, or up to 16 texts searched together."
    )
    max_num_results: int = Field(
        default=10, ge=1, le=50, description="Maximum number of results to return."
    )
    filters: SearchFilter | None = Field(
        default=None, description="Restrict results by file `attributes`."
    )
    ranking_options: RankingOptions | None = Field(
        default=None, description="Result ranking options."
    )
    rewrite_query: bool = Field(
        default=False,
        description=(
            "Reserved for query rewriting. The query is searched as written, "
            "so this value does not change the results."
        ),
    )


class SearchResultContent(BaseModelResponse):
    """One piece of content returned by a search."""

    type: Literal["text"] = Field(default="text", description="Always `text`.")
    text: str = Field(description="The matching text.")


class VectorStoreSearchResult(BaseModelResponse):
    """One search hit."""

    file_id: str = Field(description="The file the content comes from.")
    filename: str = Field(description="The name of that file.")
    score: float = Field(
        description=(
            "Relevance of the result, best match first. A similarity between 0 "
            "and 1, or, on a vector store that measures its own relevance, that "
            "measure reported unchanged — comparable within this page only."
        )
    )
    attributes: Attributes | None = Field(
        default=None, description="The attributes stored with the file."
    )
    content: list[SearchResultContent] = Field(description="The matching content.")


class VectorStoreSearchResultsPage(BaseModelResponse):
    """One page of search results."""

    object: Literal["vector_store.search_results.page"] = Field(
        default="vector_store.search_results.page", description="The object type."
    )
    search_query: list[str] = Field(description="The query that was searched.")
    data: list[VectorStoreSearchResult] = Field(description="The results.")
    has_more: bool = Field(default=False, description="Always `false`.")
    next_page: str | None = Field(default=None, description="Always `null`.")


class FileContent(BaseModelResponse):
    """One indexed chunk of a file."""

    type: Literal["text"] = Field(default="text", description="Always `text`.")
    text: str = Field(description="The chunk text.")


class FileContentPage(BaseModelResponse):
    """One page of a file's indexed content."""

    object: Literal["vector_store.file_content.page"] = Field(
        default="vector_store.file_content.page", description="The object type."
    )
    data: list[FileContent] = Field(description="The chunks, in document order.")
    has_more: bool = Field(default=False, description="Always `false`.")
    next_page: str | None = Field(default=None, description="Always `null`.")
