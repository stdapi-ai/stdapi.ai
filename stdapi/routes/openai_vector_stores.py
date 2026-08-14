"""OpenAI-compatible Vector Stores API routes."""

from contextlib import suppress
from typing import TYPE_CHECKING, Annotated, Literal

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import APIRouter, Depends, Path, Query

from stdapi.api_errors import ApiError
from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.cleanup import schedule_cleanup
from stdapi.config import SETTINGS
from stdapi.monitoring import log_request_params, log_response_params
from stdapi.types import FILE_ID_PATTERN
from stdapi.types.openai_vector_stores import (
    ExpiresAfter,
    FileBatchCreateParams,
    FileContent,
    FileContentPage,
    FileCounts,
    LastError,
    ListVectorStoreFilesResponse,
    ListVectorStoresResponse,
    SearchResultContent,
    StaticChunkingConfig,
    StaticChunkingStrategy,
    VectorStore,
    VectorStoreCreateParams,
    VectorStoreDeleted,
    VectorStoreFile,
    VectorStoreFileBatch,
    VectorStoreFileCreateParams,
    VectorStoreFileDeleted,
    VectorStoreFileUpdateParams,
    VectorStoreSearchParams,
    VectorStoreSearchResult,
    VectorStoreSearchResultsPage,
    VectorStoreUpdateParams,
)
from stdapi.vector_stores import (
    FILE_BATCH_ID_PATTERN,
    VECTOR_STORE_ID_PATTERN,
    BatchRecord,
    FileRecord,
    PendingFile,
    StoreRecord,
    attach_files,
    cancel_batch,
    check_attributes,
    create_store,
    delete_store,
    detach_file,
    list_batch_files,
    list_store_files,
    list_stores,
    new_batch_id,
    parse_batch_id,
    parse_store_id,
    read_batch,
    read_file,
    read_file_chunks,
    read_store,
    search,
    touch_store,
    update_file_attributes,
    update_store,
)

if TYPE_CHECKING:
    from enum import Enum

    from stdapi.types.openai_vector_stores import Attributes, ChunkingStrategyParam

#: OpenAI vector stores router tags
OPENAI_VECTOR_STORES_TAGS: list[str | Enum] = ["Vector Stores", TAG_OPENAI]

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=OPENAI_VECTOR_STORES_TAGS
)

#: Reusable path annotation for the ``vector_store_id`` path parameter.
_StoreId = Annotated[
    str,
    Path(description="The ID of the vector store.", pattern=VECTOR_STORE_ID_PATTERN),
]

#: Reusable path annotation for the ``file_id`` path parameter.
_FileId = Annotated[
    str, Path(description="The ID of the attached file.", pattern=FILE_ID_PATTERN)
]

#: Reusable path annotation for the ``batch_id`` path parameter.
_BatchId = Annotated[
    str, Path(description="The ID of the file batch.", pattern=FILE_BATCH_ID_PATTERN)
]

#: Reusable query annotation for the listing page size.
_Limit = Annotated[
    int, Query(ge=1, le=100, description="Number of objects to return, from 1 to 100.")
]

#: Reusable query annotation for the listing sort order.
_Order = Annotated[
    Literal["asc", "desc"],
    Query(description="Sort order by the `created_at` timestamp of the objects."),
]

#: Reusable query annotation for the per-status listing filter.
_StatusFilter = Annotated[
    Literal["in_progress", "completed", "failed", "cancelled"] | None,
    Query(alias="filter", description="Return only files with this status."),
]


def _to_vector_store(record: StoreRecord) -> VectorStore:
    """Convert a store record into its API representation.

    Args:
        record: The stored bookkeeping of the vector store.

    Returns:
        The API object.
    """
    counts = record.file_counts
    return VectorStore(
        id=record.id,
        created_at=record.created_at,
        name=record.name,
        description=record.description or None,
        usage_bytes=record.usage_bytes,
        file_counts=FileCounts(
            in_progress=counts.in_progress,
            completed=counts.completed,
            failed=counts.failed,
            cancelled=counts.cancelled,
            total=counts.total,
        ),
        status=record.status,
        expires_after=(
            ExpiresAfter(days=record.expires_after_days)
            if record.expires_after_days is not None
            else None
        ),
        expires_at=record.expires_at,
        last_active_at=record.last_active_at,
        metadata=record.metadata or None,
    )


def _to_file(store_id: str, record: FileRecord) -> VectorStoreFile:
    """Convert a file record into its API representation.

    Args:
        store_id: The vector store the file belongs to.
        record: The stored bookkeeping of the file.

    Returns:
        The API object.
    """
    return VectorStoreFile(
        id=record.id,
        created_at=record.created_at,
        usage_bytes=record.usage_bytes,
        vector_store_id=store_id,
        status=record.status,
        last_error=(
            LastError(code=record.last_error.code, message=record.last_error.message)
            if record.last_error
            else None
        ),
        chunking_strategy=StaticChunkingStrategy(
            static=StaticChunkingConfig(
                max_chunk_size_tokens=record.max_chunk_size_tokens,
                chunk_overlap_tokens=record.chunk_overlap_tokens,
            )
        ),
        attributes=record.attributes,
    )


def _to_batch(store_id: str, record: BatchRecord) -> VectorStoreFileBatch:
    """Convert a batch record into its API representation.

    Args:
        store_id: The vector store the batch belongs to.
        record: The stored bookkeeping of the batch.

    Returns:
        The API object.
    """
    counts = record.file_counts
    return VectorStoreFileBatch(
        id=record.id,
        created_at=record.created_at,
        vector_store_id=store_id,
        status=record.status,
        file_counts=FileCounts(
            in_progress=counts.in_progress,
            completed=counts.completed,
            failed=counts.failed,
            cancelled=counts.cancelled,
            total=counts.total,
        ),
    )


def _chunking(
    strategy: ChunkingStrategyParam | None, store: StoreRecord | None = None
) -> tuple[int, int]:
    """Resolve the chunk size and overlap of a request.

    Args:
        strategy: The requested chunking strategy, if any.
        store: The store whose defaults apply when the request carries none.

    Returns:
        ``(max_chunk_size_tokens, chunk_overlap_tokens)``.
    """
    if strategy is not None and strategy.type == "static":
        return (
            strategy.static.max_chunk_size_tokens,
            strategy.static.chunk_overlap_tokens,
        )
    if store is not None:
        return store.max_chunk_size_tokens, store.chunk_overlap_tokens
    return (
        SETTINGS.vector_store_chunk_size_tokens,
        SETTINGS.vector_store_chunk_overlap_tokens,
    )


def _pending(
    store: StoreRecord,
    file_id: str,
    attributes: Attributes | None,
    strategy: ChunkingStrategyParam | None,
) -> PendingFile:
    """Build one indexing request, validating its attributes.

    Args:
        store: The store the file is attached to.
        file_id: The uploaded file to index.
        attributes: Caller-supplied attributes, if any.
        strategy: The requested chunking strategy, if any.

    Returns:
        The pending file.

    Raises:
        ApiError: When the attributes do not fit the per-file budget (400).
    """
    resolved = attributes or {}
    check_attributes(resolved)
    size, overlap = _chunking(strategy, store)
    return PendingFile(
        file_id=file_id,
        attributes=resolved,
        max_chunk_size_tokens=size,
        chunk_overlap_tokens=overlap,
    )


@router.post(
    "/vector_stores",
    summary="Create a vector store for semantic file search (OpenAI format)",
    operation_id="openai_vector_store_create",
    description=(
        "Creates a vector store, an indexed collection of files that can be "
        "searched by meaning rather than by keyword (OpenAI Vector Stores API).\n\n"
        "Pass `file_ids` to index files already uploaded with `openai_file`. "
        "Indexing runs in the background: the store is returned immediately with "
        "`status=in_progress` and becomes `completed` once every file is indexed. "
        "Poll `openai_vector_store_get`, or `openai_vector_store_file_get` for one file.\n\n"
        "Only text files can be indexed; any other file is reported with "
        "`status=failed` and `last_error.code=unsupported_file`."
    ),
    response_description="The created vector store.",
    response_model_exclude_none=True,
)
async def create_vector_store(
    request: VectorStoreCreateParams, _: Annotated[None, Depends(authenticate)] = None
) -> VectorStore:
    """Create a vector store and start indexing the files it was given.

    Args:
        request: Vector store creation parameters.

    Returns:
        The created vector store.

    Raises:
        ApiError: With 503 when vector storage is not configured; 404 when one
            of the files does not exist.
    """
    log_request_params(request)
    size, overlap = _chunking(request.chunking_strategy)
    store = await create_store(
        name=request.name or "",
        description=request.description or "",
        metadata=dict(request.metadata or {}),
        expires_after_days=(
            request.expires_after.days if request.expires_after else None
        ),
        max_chunk_size_tokens=size,
        chunk_overlap_tokens=overlap,
    )
    if request.file_ids:
        try:
            await attach_files(
                store,
                [
                    _pending(store, file_id, None, request.chunking_strategy)
                    for file_id in request.file_ids
                ],
                batch_id="",
            )
        except ApiError:
            # The caller never learns this id, and a cleanup cannot schedule cleanups.
            with suppress(ApiError, BotoCoreError, ClientError):
                await delete_store(store.id)
            raise
        store = await read_store(store.id)
    return log_response_params(_to_vector_store(store))


@router.get(
    "/vector_stores",
    summary="List vector stores (OpenAI format)",
    operation_id="openai_vector_store_list",
    description="Returns a paginated list of vector stores (OpenAI Vector Stores API).",
    response_description="A list of vector stores.",
    response_model_exclude_none=True,
)
async def list_vector_stores(
    after: Annotated[
        str | None,
        Query(
            description="Cursor: the object ID to start after.",
            pattern=VECTOR_STORE_ID_PATTERN,
        ),
    ] = None,
    before: Annotated[
        str | None,
        Query(
            description="Cursor: the object ID to stop before.",
            pattern=VECTOR_STORE_ID_PATTERN,
        ),
    ] = None,
    limit: _Limit = 20,
    order: _Order = "desc",
    _: Annotated[None, Depends(authenticate)] = None,
) -> ListVectorStoresResponse:
    """List vector stores with cursor-based pagination.

    Args:
        after: Return stores created strictly after this ID.
        before: Return stores created strictly before this ID.
        limit: Maximum number of stores to return.
        order: Sort order by creation time.

    Returns:
        One page of vector stores.

    Raises:
        ApiError: With 503 when vector storage is not configured.
    """
    log_request_params(
        {"after": after, "before": before, "limit": limit, "order": order}
    )
    records, has_more = await list_stores(
        after=after or "", before=before or "", limit=limit, order=order
    )
    stores = [_to_vector_store(record) for record in records]
    return log_response_params(
        ListVectorStoresResponse(
            data=stores,
            has_more=has_more,
            first_id=stores[0].id if stores else None,
            last_id=stores[-1].id if stores else None,
        )
    )


@router.get(
    "/vector_stores/{vector_store_id}",
    summary="Retrieve a vector store (OpenAI format)",
    operation_id="openai_vector_store_get",
    description=(
        "Returns a vector store, including its indexing progress in "
        "`file_counts` and its `status` (OpenAI Vector Stores API)."
    ),
    response_description="The vector store.",
    response_model_exclude_none=True,
)
async def retrieve_vector_store(
    vector_store_id: _StoreId, _: Annotated[None, Depends(authenticate)] = None
) -> VectorStore:
    """Retrieve one vector store.

    Args:
        vector_store_id: The vector store to read.

    Returns:
        The vector store.

    Raises:
        ApiError: With 404 when the vector store does not exist.
    """
    log_request_params({"vector_store_id": vector_store_id})
    return log_response_params(
        _to_vector_store(await read_store(parse_store_id(vector_store_id)))
    )


@router.post(
    "/vector_stores/{vector_store_id}",
    summary="Update a vector store (OpenAI format)",
    operation_id="openai_vector_store_update",
    description=(
        "Updates a vector store's `name`, `metadata` or expiration policy "
        "(OpenAI Vector Stores API). Send `expires_after` as `null` to remove "
        "the expiration."
    ),
    response_description="The updated vector store.",
    response_model_exclude_none=True,
)
async def update_vector_store(
    vector_store_id: _StoreId,
    request: VectorStoreUpdateParams,
    _: Annotated[None, Depends(authenticate)] = None,
) -> VectorStore:
    """Update one vector store.

    Args:
        vector_store_id: The vector store to update.
        request: The fields to change.

    Returns:
        The updated vector store.

    Raises:
        ApiError: With 404 when the vector store does not exist.
    """
    log_request_params(request)
    record = await update_store(
        parse_store_id(vector_store_id),
        name=request.name,
        metadata=dict(request.metadata) if request.metadata is not None else None,
        expires_after_days=(
            request.expires_after.days if request.expires_after else None
        ),
        clear_expiry="expires_after" in request.model_fields_set
        and request.expires_after is None,
    )
    return log_response_params(_to_vector_store(record))


@router.delete(
    "/vector_stores/{vector_store_id}",
    summary="Delete a vector store (OpenAI format)",
    operation_id="openai_vector_store_delete",
    description=(
        "Permanently deletes a vector store and everything indexed in it "
        "(OpenAI Vector Stores API). The uploaded files themselves are kept."
    ),
    response_description="Deletion status.",
    response_model_exclude_none=True,
)
async def delete_vector_store(
    vector_store_id: _StoreId, _: Annotated[None, Depends(authenticate)] = None
) -> VectorStoreDeleted:
    """Delete one vector store.

    Args:
        vector_store_id: The vector store to delete.

    Returns:
        The deletion confirmation.

    Raises:
        ApiError: With 404 when the vector store does not exist.
    """
    log_request_params({"vector_store_id": vector_store_id})
    await delete_store(parse_store_id(vector_store_id))
    return log_response_params(VectorStoreDeleted(id=vector_store_id, deleted=True))


@router.post(
    "/vector_stores/{vector_store_id}/search",
    summary="Search a vector store by meaning (OpenAI format)",
    operation_id="openai_vector_store_search",
    description=(
        "Returns the passages of the indexed files closest in meaning to the "
        "query (OpenAI Vector Stores API).\n\n"
        "Each result carries its source `file_id`, `filename`, the matching "
        "text and a `score` between 0 and 1, best match first. Use `filters` to "
        "restrict the search to files carrying given `attributes`, and "
        "`ranking_options.score_threshold` to drop weak matches."
    ),
    response_description="The search results.",
    response_model_exclude_none=True,
)
async def search_vector_store(
    vector_store_id: _StoreId,
    request: VectorStoreSearchParams,
    _: Annotated[None, Depends(authenticate)] = None,
) -> VectorStoreSearchResultsPage:
    """Search one vector store.

    Args:
        vector_store_id: The vector store to search.
        request: The query and its options.

    Returns:
        One page of search results.

    Raises:
        ApiError: With 404 when the vector store does not exist.
    """
    log_request_params(request)
    store = await read_store(parse_store_id(vector_store_id))
    queries = [request.query] if isinstance(request.query, str) else request.query
    results = await search(
        store,
        queries,
        max_num_results=request.max_num_results,
        filters=request.filters,
        score_threshold=(
            request.ranking_options.score_threshold if request.ranking_options else None
        ),
    )
    schedule_cleanup(touch_store(store))
    return log_response_params(
        VectorStoreSearchResultsPage(
            search_query=list(queries),
            data=[
                VectorStoreSearchResult(
                    file_id=result.file_id,
                    filename=result.filename,
                    score=result.score,
                    attributes=result.attributes or None,
                    content=[SearchResultContent(text=result.text)],
                )
                for result in results
            ],
        ),
        exclude={"data"},
    )


@router.post(
    "/vector_stores/{vector_store_id}/files",
    summary="Attach a file to a vector store (OpenAI format)",
    operation_id="openai_vector_store_file_create",
    description=(
        "Indexes an already-uploaded file into a vector store (OpenAI Vector "
        "Stores API). Upload the file first with `openai_file`.\n\n"
        "Indexing runs in the background: the file is returned with "
        "`status=in_progress` and becomes `completed` once searchable. Use "
        "`openai_vector_store_file_batch_create` to attach several files at once."
    ),
    response_description="The attached file.",
    response_model_exclude_none=True,
)
async def create_vector_store_file(
    vector_store_id: _StoreId,
    request: VectorStoreFileCreateParams,
    _: Annotated[None, Depends(authenticate)] = None,
) -> VectorStoreFile:
    """Attach one file to a vector store.

    Args:
        vector_store_id: The vector store to attach the file to.
        request: The file and its indexing options.

    Returns:
        The attached file.

    Raises:
        ApiError: With 404 when the vector store or the file does not exist;
            400 when the attributes exceed the per-file budget.
    """
    log_request_params(request)
    store = await read_store(parse_store_id(vector_store_id))
    records = await attach_files(
        store,
        [
            _pending(
                store, request.file_id, request.attributes, request.chunking_strategy
            )
        ],
        batch_id="",
    )
    return log_response_params(_to_file(store.id, records[0]))


@router.get(
    "/vector_stores/{vector_store_id}/files",
    summary="List the files attached to a vector store (OpenAI format)",
    operation_id="openai_vector_store_file_list",
    description=(
        "Returns a paginated list of the files attached to a vector store, "
        "with their indexing status (OpenAI Vector Stores API)."
    ),
    response_description="A list of vector store files.",
    response_model_exclude_none=True,
)
async def list_vector_store_files(
    vector_store_id: _StoreId,
    after: Annotated[
        str | None,
        Query(
            description="Cursor: the object ID to start after.", pattern=FILE_ID_PATTERN
        ),
    ] = None,
    before: Annotated[
        str | None,
        Query(
            description="Cursor: the object ID to stop before.", pattern=FILE_ID_PATTERN
        ),
    ] = None,
    limit: _Limit = 20,
    order: _Order = "desc",
    status_filter: _StatusFilter = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> ListVectorStoreFilesResponse:
    """List the files attached to a vector store.

    Args:
        vector_store_id: The vector store to list.
        after: Return files created strictly after this ID.
        before: Return files created strictly before this ID.
        limit: Maximum number of files to return.
        order: Sort order by creation time.
        status_filter: Return only files with this status.

    Returns:
        One page of vector store files.

    Raises:
        ApiError: With 404 when the vector store does not exist.
    """
    log_request_params(
        {
            "vector_store_id": vector_store_id,
            "after": after,
            "before": before,
            "limit": limit,
            "order": order,
            "filter": status_filter,
        }
    )
    store_id = parse_store_id(vector_store_id)
    await read_store(store_id)
    records, has_more = await list_store_files(
        store_id,
        after=after or "",
        before=before or "",
        limit=limit,
        order=order,
        status=status_filter or "",
    )
    files = [_to_file(store_id, record) for record in records]
    return log_response_params(
        ListVectorStoreFilesResponse(
            data=files,
            has_more=has_more,
            first_id=files[0].id if files else None,
            last_id=files[-1].id if files else None,
        )
    )


@router.get(
    "/vector_stores/{vector_store_id}/files/{file_id}",
    summary="Retrieve a file attached to a vector store (OpenAI format)",
    operation_id="openai_vector_store_file_get",
    description=(
        "Returns one attached file with its indexing `status` and, when it "
        "failed, `last_error` (OpenAI Vector Stores API)."
    ),
    response_description="The vector store file.",
    response_model_exclude_none=True,
)
async def retrieve_vector_store_file(
    vector_store_id: _StoreId,
    file_id: _FileId,
    _: Annotated[None, Depends(authenticate)] = None,
) -> VectorStoreFile:
    """Retrieve one attached file.

    Args:
        vector_store_id: The vector store the file belongs to.
        file_id: The attached file to read.

    Returns:
        The vector store file.

    Raises:
        ApiError: With 404 when the vector store or the file does not exist.
    """
    log_request_params({"vector_store_id": vector_store_id, "file_id": file_id})
    store_id = parse_store_id(vector_store_id)
    return log_response_params(_to_file(store_id, await read_file(store_id, file_id)))


@router.post(
    "/vector_stores/{vector_store_id}/files/{file_id}",
    summary="Update the attributes of an attached file (OpenAI format)",
    operation_id="openai_vector_store_file_update",
    description=(
        "Replaces the `attributes` stored with an attached file (OpenAI Vector "
        "Stores API). The new attributes apply to later searches."
    ),
    response_description="The updated vector store file.",
    response_model_exclude_none=True,
)
async def update_vector_store_file(
    vector_store_id: _StoreId,
    file_id: _FileId,
    request: VectorStoreFileUpdateParams,
    _: Annotated[None, Depends(authenticate)] = None,
) -> VectorStoreFile:
    """Replace an attached file's attributes.

    Args:
        vector_store_id: The vector store the file belongs to.
        file_id: The attached file to update.
        request: The new attributes.

    Returns:
        The updated vector store file.

    Raises:
        ApiError: With 404 when the vector store or the file does not exist;
            400 when the attributes exceed the per-file budget.
    """
    log_request_params(request)
    store_id = parse_store_id(vector_store_id)
    attributes = request.attributes or {}
    check_attributes(attributes)
    return log_response_params(
        _to_file(store_id, await update_file_attributes(store_id, file_id, attributes))
    )


@router.delete(
    "/vector_stores/{vector_store_id}/files/{file_id}",
    summary="Detach a file from a vector store (OpenAI format)",
    operation_id="openai_vector_store_file_delete",
    description=(
        "Removes a file from a vector store so it is no longer searchable "
        "(OpenAI Vector Stores API). The uploaded file itself is kept."
    ),
    response_description="Deletion status.",
    response_model_exclude_none=True,
)
async def delete_vector_store_file(
    vector_store_id: _StoreId,
    file_id: _FileId,
    _: Annotated[None, Depends(authenticate)] = None,
) -> VectorStoreFileDeleted:
    """Detach one file from a vector store.

    Args:
        vector_store_id: The vector store the file belongs to.
        file_id: The attached file to remove.

    Returns:
        The deletion confirmation.

    Raises:
        ApiError: With 404 when the vector store or the file does not exist.
    """
    log_request_params({"vector_store_id": vector_store_id, "file_id": file_id})
    await detach_file(parse_store_id(vector_store_id), file_id)
    return log_response_params(VectorStoreFileDeleted(id=file_id, deleted=True))


@router.get(
    "/vector_stores/{vector_store_id}/files/{file_id}/content",
    summary="Read the indexed content of an attached file (OpenAI format)",
    operation_id="openai_vector_store_file_content",
    description=(
        "Returns the passages a file was indexed as, in document order "
        "(OpenAI Vector Stores API). Use `openai_file_content` to download the "
        "original file instead."
    ),
    response_description="The indexed content of the file.",
    response_model_exclude_none=True,
)
async def retrieve_vector_store_file_content(
    vector_store_id: _StoreId,
    file_id: _FileId,
    _: Annotated[None, Depends(authenticate)] = None,
) -> FileContentPage:
    """Return one attached file's indexed passages.

    Args:
        vector_store_id: The vector store the file belongs to.
        file_id: The attached file to read.

    Returns:
        The indexed passages, in document order.

    Raises:
        ApiError: With 404 when the vector store or the file does not exist.
    """
    log_request_params({"vector_store_id": vector_store_id, "file_id": file_id})
    store_id = parse_store_id(vector_store_id)
    record = await read_file(store_id, file_id)
    chunks = await read_file_chunks(store_id, record)
    return log_response_params(
        FileContentPage(data=[FileContent(text=chunk) for chunk in chunks]),
        exclude={"data"},
    )


@router.post(
    "/vector_stores/{vector_store_id}/file_batches",
    summary="Attach several files to a vector store at once (OpenAI format)",
    operation_id="openai_vector_store_file_batch_create",
    description=(
        "Indexes several already-uploaded files into a vector store in one "
        "request (OpenAI Vector Stores API).\n\n"
        "Pass `file_ids` to share one set of `attributes` and one "
        "`chunking_strategy` across the batch, or `files` to give each file its "
        "own. Indexing runs in the background; poll "
        "`openai_vector_store_file_batch_get` until `status` is `completed`."
    ),
    response_description="The created file batch.",
    response_model_exclude_none=True,
)
async def create_vector_store_file_batch(
    vector_store_id: _StoreId,
    request: FileBatchCreateParams,
    _: Annotated[None, Depends(authenticate)] = None,
) -> VectorStoreFileBatch:
    """Attach a batch of files to a vector store.

    Args:
        vector_store_id: The vector store to attach the files to.
        request: The files and their indexing options.

    Returns:
        The created file batch.

    Raises:
        ApiError: With 404 when the vector store or one of the files does not
            exist; 400 when the attributes exceed the per-file budget.
    """
    log_request_params(request)
    store = await read_store(parse_store_id(vector_store_id))
    if request.files:
        pending = [
            _pending(store, entry.file_id, entry.attributes, entry.chunking_strategy)
            for entry in request.files
        ]
    else:
        pending = [
            _pending(store, file_id, request.attributes, request.chunking_strategy)
            for file_id in request.file_ids or ()
        ]
    batch_id = new_batch_id()
    await attach_files(store, pending, batch_id=batch_id)
    return log_response_params(
        _to_batch(store.id, await read_batch(store.id, batch_id))
    )


@router.get(
    "/vector_stores/{vector_store_id}/file_batches/{batch_id}",
    summary="Retrieve a file batch (OpenAI format)",
    operation_id="openai_vector_store_file_batch_get",
    description=(
        "Returns a file batch with its indexing progress in `file_counts` "
        "(OpenAI Vector Stores API)."
    ),
    response_description="The file batch.",
    response_model_exclude_none=True,
)
async def retrieve_vector_store_file_batch(
    vector_store_id: _StoreId,
    batch_id: _BatchId,
    _: Annotated[None, Depends(authenticate)] = None,
) -> VectorStoreFileBatch:
    """Retrieve one file batch.

    Args:
        vector_store_id: The vector store the batch belongs to.
        batch_id: The file batch to read.

    Returns:
        The file batch.

    Raises:
        ApiError: With 404 when the vector store or the batch does not exist.
    """
    log_request_params({"vector_store_id": vector_store_id, "batch_id": batch_id})
    store_id = parse_store_id(vector_store_id)
    return log_response_params(
        _to_batch(store_id, await read_batch(store_id, parse_batch_id(batch_id)))
    )


@router.post(
    "/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel",
    summary="Cancel a file batch (OpenAI format)",
    operation_id="openai_vector_store_file_batch_cancel",
    description=(
        "Stops indexing the files of a batch that have not started yet "
        "(OpenAI Vector Stores API). Files already indexed stay searchable and "
        "keep their `completed` status."
    ),
    response_description="The cancelled file batch.",
    response_model_exclude_none=True,
)
async def cancel_vector_store_file_batch(
    vector_store_id: _StoreId,
    batch_id: _BatchId,
    _: Annotated[None, Depends(authenticate)] = None,
) -> VectorStoreFileBatch:
    """Cancel one file batch.

    Args:
        vector_store_id: The vector store the batch belongs to.
        batch_id: The file batch to cancel.

    Returns:
        The file batch.

    Raises:
        ApiError: With 404 when the vector store or the batch does not exist.
    """
    log_request_params({"vector_store_id": vector_store_id, "batch_id": batch_id})
    store_id = parse_store_id(vector_store_id)
    return log_response_params(
        _to_batch(store_id, await cancel_batch(store_id, parse_batch_id(batch_id)))
    )


@router.get(
    "/vector_stores/{vector_store_id}/file_batches/{batch_id}/files",
    summary="List the files of a file batch (OpenAI format)",
    operation_id="openai_vector_store_file_batch_file_list",
    description=(
        "Returns a paginated list of the files of one batch, with their "
        "indexing status (OpenAI Vector Stores API)."
    ),
    response_description="A list of vector store files.",
    response_model_exclude_none=True,
)
async def list_vector_store_file_batch_files(
    vector_store_id: _StoreId,
    batch_id: _BatchId,
    after: Annotated[
        str | None,
        Query(
            description="Cursor: the object ID to start after.", pattern=FILE_ID_PATTERN
        ),
    ] = None,
    before: Annotated[
        str | None,
        Query(
            description="Cursor: the object ID to stop before.", pattern=FILE_ID_PATTERN
        ),
    ] = None,
    limit: _Limit = 20,
    order: _Order = "desc",
    status_filter: _StatusFilter = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> ListVectorStoreFilesResponse:
    """List the files of one batch.

    Args:
        vector_store_id: The vector store the batch belongs to.
        batch_id: The file batch to list.
        after: Return files created strictly after this ID.
        before: Return files created strictly before this ID.
        limit: Maximum number of files to return.
        order: Sort order by creation time.
        status_filter: Return only files with this status.

    Returns:
        One page of vector store files.

    Raises:
        ApiError: With 404 when the vector store or the batch does not exist.
    """
    log_request_params(
        {
            "vector_store_id": vector_store_id,
            "batch_id": batch_id,
            "after": after,
            "before": before,
            "limit": limit,
            "order": order,
            "filter": status_filter,
        }
    )
    store_id = parse_store_id(vector_store_id)
    await read_batch(store_id, parse_batch_id(batch_id))
    records, has_more = await list_batch_files(
        store_id,
        batch_id,
        after=after or "",
        before=before or "",
        limit=limit,
        order=order,
        status=status_filter or "",
    )
    files = [_to_file(store_id, record) for record in records]
    return log_response_params(
        ListVectorStoreFilesResponse(
            data=files,
            has_more=has_more,
            first_id=files[0].id if files else None,
            last_id=files[-1].id if files else None,
        )
    )
