"""Vector Stores engine — identifiers, chunking, indexing and semantic search.

Everything here is backend-neutral: the engine reads a file, cuts it into
passages, embeds them, keeps the bookkeeping honest and merges what a search
answered. Which index holds the vectors, how a filter is expressed and what a
score means belong to the backend behind
:class:`~stdapi.vector_stores.backend.VectorIndex`, and the engine refuses or
degrades against what that backend declares rather than discovering a gap
mid-request.
"""

from asyncio import Semaphore, Task, create_task, wait_for
from base64 import b32hexencode
from contextlib import suppress
from dataclasses import dataclass
from re import compile as re_compile
from typing import TYPE_CHECKING, Final, Literal
from uuid import uuid7

from botocore.exceptions import BotoCoreError, ClientError

from stdapi.api_errors import ApiError
from stdapi.cleanup import drain_tasks, schedule_cleanup
from stdapi.config import SETTINGS
from stdapi.files import get_file, get_file_content, parse_file_id
from stdapi.models import validate_model
from stdapi.models.embedding import get_embedding_model
from stdapi.monitoring import REQUEST_ID, log_background_event, log_error_details
from stdapi.types import FILE_ID_PATTERN
from stdapi.utils import now_utc_timestamp
from stdapi.vector_stores._concurrency import gather_bounded
from stdapi.vector_stores._paging import page_records
from stdapi.vector_stores.backend import (
    IndexVector,
    as_stream,
    check_filter,
    raise_not_found,
    unsupported_file_message,
)
from stdapi.vector_stores.knowledge_base import (
    DOCUMENT_ID_PATTERN,
    KNOWLEDGE_BASE_ID_PATTERN,
    STORE_ID_PREFIX,
    check_allowlisted,
    is_knowledge_base_store,
)
from stdapi.vector_stores.models import (
    BatchRecord,
    FileCountsRecord,
    FileErrorRecord,
    FileRecord,
    PendingFile,
    SearchResult,
    StoreRecord,
)
from stdapi.vector_stores.records import (
    RECORD_WAVE,
    all_record_keys,
    batch_key,
    delete_record,
    file_key,
    gather_records,
    read_record,
    records_bucket,
    store_file_records,
    store_key,
    store_prefix,
    update_record,
    write_if_unchanged,
    write_record,
)
from stdapi.vector_stores.records import list_batch_files as _list_batch_file_records
from stdapi.vector_stores.records import list_store_files as _list_store_file_records
from stdapi.vector_stores.records import list_stores as _list_store_records
from stdapi.vector_stores.records import read_batch as _read_batch_record
from stdapi.vector_stores.records import read_file as _read_file_record
from stdapi.vector_stores.registry import (
    alternative_for,
    backend_for,
    default_backend,
    external_store_for,
    external_stores,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from stdapi.input_file import InputFileUrl
    from stdapi.types.openai_vector_stores import (
        Attributes,
        ChunkingStrategyParam,
        SearchFilter,
    )
    from stdapi.vector_stores.backend import VectorIndex

#: Regex pattern a vector store identifier must match on input.
VECTOR_STORE_ID_PATTERN: str = (
    rf"^(vs_[0-9a-v]{{26}}|{STORE_ID_PREFIX}{KNOWLEDGE_BASE_ID_PATTERN})$"
)

#: Regex pattern the identifier of a file attached to a store must match on input.
VECTOR_STORE_FILE_ID_PATTERN: str = (
    rf"^({FILE_ID_PATTERN[1:-1]}|{DOCUMENT_ID_PATTERN})$"
)

#: Regex pattern a file batch identifier must match on input.
FILE_BATCH_ID_PATTERN: str = r"^vsfb_[0-9a-v]{26}$"

#: Compiled matcher for vector store identifiers.
_STORE_ID_RE = re_compile(VECTOR_STORE_ID_PATTERN).match

#: Compiled matcher for file batch identifiers.
_BATCH_ID_RE = re_compile(FILE_BATCH_ID_PATTERN).match

#: Approximate characters per token used to turn a token budget into a text slice.
_CHARACTERS_PER_TOKEN: Final[int] = 4

#: Fraction of a chunk a cut may be moved back over to reach a word boundary.
_CUT_SEARCH_FRACTION: Final[int] = 5

#: Bytes of one file indexing buffers, whatever ``max_input_file_size`` allows.
_MAX_INDEXABLE_BYTES: Final[int] = 100 * 1024 * 1024

#: Chunks embedded concurrently while indexing one file.
_EMBED_WAVE: Final[int] = 16

#: Vectors the attribute rewrite holds in memory at once.
_REWRITE_WINDOW: Final[int] = 100

#: Seconds of inactivity before a search refreshes ``last_active_at``.
_LAST_ACTIVE_REFRESH_SECONDS: Final[int] = 3600

#: Files read, chunked and embedded at once, however many requests ask for it.
_INDEXING_SLOTS: Final[int] = 2

#: Bounds concurrent indexing to :data:`_INDEXING_SLOTS` files server-wide.
_INDEXING_SEMAPHORE: Final[Semaphore] = Semaphore(_INDEXING_SLOTS)

#: Seconds a file may stay ``in_progress`` with nothing renewing its store's lease.
_INDEXING_LEASE_SECONDS: Final[int] = 900

#: Seconds before its lease runs out at which an indexing wave renews it.
_LEASE_MARGIN_SECONDS: Final[int] = 300

#: Why a file whose indexing lost the task running it reports having failed.
_INTERRUPTED_MESSAGE: Final[str] = (
    "The file could not be indexed: the indexing was interrupted before it "
    "finished. Attach the file again to index it."
)

#: Why a file the server could not index reports having failed.
_FAILED_MESSAGE: Final[str] = "The file could not be indexed."

#: Records a merged listing reads before paging in memory.
_LIST_ALL: Final[int] = 1000


@dataclass(slots=True, frozen=True)
class _IndexingWave:
    """The files one background indexing task answers for.

    Attributes:
        store_id: The store the files are attached to.
        file_ids: The files the task indexes, in order.
    """

    store_id: str
    file_ids: tuple[str, ...]


#: Running indexing tasks and the files they own, held until they finish.
_INDEXING_TASKS: Final[dict[Task[None], _IndexingWave]] = {}

#: Why a store held elsewhere answers no file batch.
_NO_BATCH: Final[str] = (
    "it keeps no file batches. Attach and follow its files one at a time instead."
)


def new_store_id() -> str:
    """Return a new vector store identifier that sorts by creation time."""
    return f"vs_{b32hexencode(uuid7().bytes).decode().rstrip('=').lower()}"


def new_batch_id() -> str:
    """Return a new file batch identifier that sorts by creation time."""
    return f"vsfb_{b32hexencode(uuid7().bytes).decode().rstrip('=').lower()}"


def parse_store_id(store_id: str) -> str:
    """Validate a vector store identifier.

    A store served from elsewhere is only addressable when the deployment was
    given it. One it was not is answered exactly as a malformed or unknown
    identifier is, so the configuration cannot be probed through the API.

    Args:
        store_id: The identifier to validate.

    Returns:
        The identifier, unchanged.

    Raises:
        ApiError: When the identifier is malformed, or names a store this
            deployment may not address (404, as upstream reports an unknown one).
    """
    if not _STORE_ID_RE(store_id):
        raise_not_found("vector store", store_id)
    if is_knowledge_base_store(store_id):
        check_allowlisted(store_id)
    return store_id


def parse_batch_id(batch_id: str) -> str:
    """Validate a file batch identifier.

    Args:
        batch_id: The identifier to validate.

    Returns:
        The identifier, unchanged.

    Raises:
        ApiError: When the identifier is malformed (404, as upstream reports it).
    """
    if not _BATCH_ID_RE(batch_id):
        raise_not_found("file batch", batch_id)
    return batch_id


def vector_key(file_id: str, chunk_index: int) -> str:
    """Return the index key of one chunk.

    Args:
        file_id: The file the chunk comes from.
        chunk_index: Position of the chunk within the file.

    Returns:
        The vector key.
    """
    return f"{file_id}#{chunk_index}"


def check_attributes(attributes: Attributes, store: StoreRecord | None = None) -> None:
    """Reject attributes the backend serving *store* cannot keep searchable.

    Args:
        attributes: The caller-supplied attributes.
        store: The store the file is attached to, or ``None`` when the caller
            does not hold the record.

    Raises:
        ApiError: When the attributes do not fit the per-file budget (400).
    """
    backend = default_backend() if store is None else backend_for(store)
    backend.check_attributes(attributes)


def check_chunking_strategy(
    strategy: ChunkingStrategyParam | None, store: StoreRecord | None = None
) -> None:
    """Reject a chunking strategy the backend serving *store* cannot honour.

    A backend that cuts the passages itself has no chunk size to be told, and
    silently ignoring the request would answer with passages the caller did not
    ask for.

    Args:
        strategy: The requested chunking strategy, if any.
        store: The store the file is attached to, or ``None`` when it is being
            created.

    Raises:
        ApiError: When the backend chooses the passage boundaries itself (400).
    """
    if strategy is None or strategy.type != "static":
        return
    backend = default_backend() if store is None else backend_for(store)
    if backend.capabilities.chunks_on_ingestion:
        msg = (
            "This vector store chooses its own passage boundaries, so "
            "'chunking_strategy' cannot be set on it. Send the request without it."
        )
        raise ApiError(msg)


def chunk_text(
    text: str,
    max_chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    max_characters: int,
    max_bytes: int,
) -> list[str]:
    """Split *text* into overlapping chunks.

    The token budget is applied as an approximate character budget and the cut
    is moved back to the nearest line or word boundary, so a chunk never ends
    mid-word when one is within reach.

    Args:
        text: The text to split.
        max_chunk_size_tokens: Chunk size, in tokens.
        chunk_overlap_tokens: Overlap between consecutive chunks, in tokens.
        max_characters: Hard character ceiling per chunk, or ``0`` for none.
        max_bytes: Hard byte ceiling per chunk, or ``0`` for none.

    Returns:
        The chunks, in document order; empty when *text* holds no content.
    """
    size = max_chunk_size_tokens * _CHARACTERS_PER_TOKEN
    if max_characters:
        size = min(size, max_characters)
    overlap = min(chunk_overlap_tokens * _CHARACTERS_PER_TOKEN, size // 2)
    # How far a cut may move back to a separator, bounded so the loop progresses.
    reach = max(1, size - size // _CUT_SEARCH_FRACTION)
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            for separator in ("\n", " "):
                cut = text.rfind(separator, start + reach, end)
                if cut > start:
                    end = cut
                    break
        if chunk := text[start:end].strip():
            chunks.extend(_split_on_bytes(chunk, max_bytes))
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _split_on_bytes(chunk: str, max_bytes: int) -> list[str]:
    """Split *chunk* so every piece fits the per-vector text budget.

    Args:
        chunk: A chunk already within its character budget.
        max_bytes: Bytes one piece may take, or ``0`` for none.

    Returns:
        The chunk alone, or the pieces it was split into.
    """
    if not max_bytes or len(chunk.encode()) <= max_bytes:
        return [chunk]
    pieces: list[str] = []
    remaining = chunk
    while remaining:
        encoded = remaining.encode()[:max_bytes]
        piece = encoded.decode(errors="ignore")
        pieces.append(piece)
        remaining = remaining[len(piece) :]
    return pieces


async def _embed(model_id: str, texts: Sequence[str]) -> list[list[float]]:
    """Embed *texts* in bounded waves.

    Args:
        model_id: The embedding model identifier.
        texts: The texts to embed.

    Returns:
        One vector per text, in order.
    """
    model = get_embedding_model(model_id)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_WAVE):
        wave: list[InputFileUrl | str] = list(texts[start : start + _EMBED_WAVE])
        response = await model.embed_text(wave, dimensions=None, extra_params={})
        vectors.extend(response.embeddings)
    return vectors


async def resolve_embedding_model() -> tuple[str, int]:
    """Resolve the configured embedding model and measure its vector dimension.

    The dimension is measured rather than configured: an index's dimension is
    immutable, so a wrong value would only surface when the first file is
    indexed.

    Returns:
        ``(model_id, dimensions)``.

    Raises:
        ApiError: When the configured model cannot produce an embedding.
    """
    model_id = (
        await validate_model(SETTINGS.vector_store_embedding_model, "EMBEDDING")
    ).id
    vectors = await _embed(model_id, ["."])
    if not vectors or not vectors[0]:
        msg = (
            "The vector store could not be created because the configured "
            "embedding model returned no embedding."
        )
        raise ApiError(msg, status=503)
    return model_id, len(vectors[0])


def _max_input_characters(model_id: str) -> int:
    """Return the per-input character ceiling of *model_id*, or ``0`` when unbounded."""
    return get_embedding_model(model_id).max_input_characters


async def read_store(store_id: str) -> StoreRecord:
    """Return the store record of *store_id*, deleting its index once expired.

    Args:
        store_id: A validated vector store identifier.

    Returns:
        The store record.

    Raises:
        ApiError: When the store does not exist (404).
    """
    external = external_store_for(store_id)
    if external is not None:
        return await external.read_store(store_id)
    current = await read_record(StoreRecord, store_key(store_id))
    if current is None:
        raise_not_found("vector store", store_id)
    record = current[0]
    if record.expired and not record.index_deleted:
        schedule_cleanup(_release_expired(record))
    return await _recover_store(record)


async def _recover_store(store: StoreRecord) -> StoreRecord:
    """Finish what a task that stopped mid-flight left behind on *store*.

    Nothing sweeps a vector store on a schedule, so the two states a task can
    be killed in are settled the next time the store is read: the vectors of a
    detached file that were not all reclaimed, and a file left ``in_progress``
    by a task that no longer exists. Both are recognised without a read of
    their own — the store record names them — so a store with neither pays two
    comparisons.

    Args:
        store: The store record just read.

    Returns:
        The store record, re-read when the recovery changed it.
    """
    recovered = False
    if store.detaching:
        recovered = await _reclaim_detached(store)
    if (
        store.file_counts.in_progress
        and now_utc_timestamp() >= store.indexing_expires_at
    ):
        recovered = await _settle_abandoned_files(store.id) or recovered
    if not recovered:
        return store
    current = await read_record(StoreRecord, store_key(store.id))
    return store if current is None else current[0]


async def _release_expired(store: StoreRecord) -> None:
    """Delete the index of an expired store, once.

    The store itself stays readable with an ``expired`` status, as upstream
    reports it; only the storage behind it is reclaimed.

    Args:
        store: The expired store record.
    """
    # Recorded after the delete, so a failed delete is retried by a later read.
    with suppress(ApiError, BotoCoreError, ClientError):
        await backend_for(store).delete_index(store.id)
        await update_record(
            StoreRecord,
            store_key(store.id),
            lambda r: setattr(r, "index_deleted", True),
        )


async def create_store(
    *,
    name: str,
    description: str,
    metadata: dict[str, str],
    expires_after_days: int | None,
    max_chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> StoreRecord:
    """Create the index and the store record backing a new vector store.

    The index is created first: a failure afterwards leaves an unreferenced
    index that is reclaimed, never a store record pointing at nothing.

    Args:
        name: The store name.
        description: The store description.
        metadata: Caller-supplied key-value pairs.
        expires_after_days: Days of inactivity before the store expires.
        max_chunk_size_tokens: Default chunk size for files attached later.
        chunk_overlap_tokens: Default chunk overlap for files attached later.

    Returns:
        The created store record.

    Raises:
        ApiError: When vector storage is not configured (503).
    """
    records_bucket()
    backend = default_backend()
    model_id, dimensions = await resolve_embedding_model()
    store_id = new_store_id()
    now = now_utc_timestamp()
    await backend.create_index(store_id, dimensions=dimensions)
    record = StoreRecord(
        id=store_id,
        created_at=now,
        last_active_at=now,
        name=name,
        description=description,
        metadata=metadata,
        expires_after_days=expires_after_days,
        embedding_model=model_id,
        dimensions=dimensions,
        max_chunk_size_tokens=max_chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )
    try:
        await write_record(store_key(store_id), record, etag="*")
    except BotoCoreError, ClientError:
        schedule_cleanup(backend.delete_index(store_id))
        raise
    return record


async def delete_store(store_id: str) -> None:
    """Delete a vector store, its records and its index.

    The records are deleted first, so a partial failure leaves an unreferenced
    index rather than a store the client can still see.

    Args:
        store_id: A validated vector store identifier.

    Raises:
        ApiError: When the store does not exist (404), or is held elsewhere and
            is therefore not this deployment's to delete (400).
    """
    external = external_store_for(store_id)
    if external is not None:
        await external.read_store(store_id)
        external.refuse("it cannot be deleted here. Detach its files instead.")
    store = await read_store(store_id)
    prefix = store_prefix(store_id)
    await delete_record(store_key(store_id))
    schedule_cleanup(
        *(delete_record(key) for key in await all_record_keys(prefix)),
        backend_for(store).delete_index(store_id),
    )


async def update_store(
    store_id: str,
    *,
    name: str | None,
    metadata: dict[str, str] | None,
    expires_after_days: int | None,
    clear_expiry: bool,
) -> StoreRecord:
    """Apply an update to a store record.

    Args:
        store_id: A validated vector store identifier.
        name: A new name, or ``None`` to keep the current one.
        metadata: New metadata, or ``None`` to keep the current one.
        expires_after_days: A new expiration, or ``None`` to keep the current one.
        clear_expiry: Whether the caller explicitly removed the expiration.

    Returns:
        The updated store record.

    Raises:
        ApiError: When the store does not exist (404), or is held elsewhere and
            therefore describes itself (400).
    """
    external = external_store_for(store_id)
    if external is not None:
        await external.read_store(store_id)
        external.refuse(
            "its name, metadata and expiration are read from it and cannot be "
            "changed here."
        )
    await read_store(store_id)

    def mutate(record: StoreRecord) -> None:
        """Apply the requested changes to *record*."""
        if name is not None:
            record.name = name
        if metadata is not None:
            record.metadata = metadata
        if clear_expiry:
            record.expires_after_days = None
        elif expires_after_days is not None:
            record.expires_after_days = expires_after_days

    return await update_record(StoreRecord, store_key(store_id), mutate)


async def touch_store(record: StoreRecord) -> None:
    """Refresh ``last_active_at`` when it has gone stale.

    Written coarsely on purpose: it anchors the expiration, which is measured in
    days, so a conditional write per search would only add contention.

    An expired store is never refreshed: its index has already been released,
    so a refresh would leave a store reporting ``completed`` over nothing.

    Args:
        record: The store record a request just used.
    """
    # A store held elsewhere has no expiration here, so nothing anchors.
    if record.expired or record.index_deleted or external_store_for(record) is not None:
        return
    now = now_utc_timestamp()
    if now - record.last_active_at < _LAST_ACTIVE_REFRESH_SECONDS:
        return
    with suppress(ApiError, BotoCoreError, ClientError):
        await update_record(
            StoreRecord,
            store_key(record.id),
            lambda stored: setattr(stored, "last_active_at", now),
        )
        record.last_active_at = now


async def attach_files(
    store: StoreRecord, pending: Sequence[PendingFile], *, batch_id: str
) -> list[FileRecord]:
    """Register *pending* files on *store* and start indexing them.

    The records are written before the response so the files are immediately
    visible as ``in_progress``; the indexing itself runs off the request path.

    Args:
        store: The store record the files are attached to.
        pending: The files to index.
        batch_id: The batch the files belong to, or ``""``.

    Returns:
        The created file records, in the order of *pending*.

    Raises:
        ApiError: When one of the files does not exist (404).
    """
    external = external_store_for(store)
    if external is not None:
        if batch_id:
            external.refuse(
                "files cannot be attached to it in batches. Attach them one at "
                "a time instead."
            )
        return await external.attach_documents(store.id, pending)
    now = now_utc_timestamp()
    # One file is one record: a repeated id would inflate the totals for good.
    unique: dict[str, PendingFile] = {}
    for entry in pending:
        unique.setdefault(entry.file_id, entry)
    pending = list(unique.values())
    sources = await gather_bounded(
        [get_file(parse_file_id(entry.file_id)) for entry in pending], RECORD_WAVE
    )
    records = [
        FileRecord(
            id=entry.file_id,
            created_at=now,
            filename=source.filename,
            attributes=entry.attributes,
            max_chunk_size_tokens=entry.max_chunk_size_tokens,
            chunk_overlap_tokens=entry.chunk_overlap_tokens,
            batch_id=batch_id,
        )
        for entry, source in zip(pending, sources, strict=True)
    ]
    # Re-attaching replaces the record, so its outcome and chunks move with it.
    existing_records = [
        existing[0]
        for existing in await gather_records(
            FileRecord, [file_key(store.id, record.id) for record in records]
        )
        if existing is not None
    ]
    # A file still being reclaimed would take the record replacing it with it.
    for existing in existing_records:
        if existing.detaching:
            await _reclaim_file(store.id, existing.id)
    replaced = [existing for existing in existing_records if not existing.detaching]
    # An unfinished replacement still owns the older vectors: take the larger.
    previous = {
        record.id: max(record.chunk_count, record.previous_chunk_count)
        for record in replaced
    }
    for record in records:
        record.previous_chunk_count = previous.get(record.id, 0)
    await gather_bounded(
        [write_record(file_key(store.id, record.id), record) for record in records],
        RECORD_WAVE,
    )

    def add_pending(stored: StoreRecord) -> None:
        """Count the newly attached files, releasing the replaced ones."""
        counts = stored.file_counts
        for old in replaced:
            setattr(counts, old.status, max(0, getattr(counts, old.status) - 1))
            stored.usage_bytes = max(0, stored.usage_bytes - old.usage_bytes)
        counts.in_progress += len(records)
        # The lease the files are indexed under, renewed as the wave advances.
        stored.indexing_expires_at = max(
            stored.indexing_expires_at, now + _INDEXING_LEASE_SECONDS
        )

    await update_record(StoreRecord, store_key(store.id), add_pending)
    if batch_id:
        await write_record(
            batch_key(store.id, batch_id),
            BatchRecord(
                id=batch_id,
                created_at=now,
                file_counts=FileCountsRecord(in_progress=len(records)),
            ),
        )
    # Last, and still inside the request: every record the job names is durable
    # by this point, and so are the file bytes, so the job it hands over is
    # replayable by whichever server picks it up.
    await _hand_over_indexing(store.id, [r.id for r in records], batch_id)
    return records


async def _hand_over_indexing(
    store_id: str, file_ids: list[str], batch_id: str
) -> None:
    """Give the indexing of *file_ids* to the queue, or run it here.

    Args:
        store_id: A validated vector store identifier.
        file_ids: The files to index, in order.
        batch_id: The batch the files belong to, or ``""``.
    """
    # Imported here: the queue's handlers call back into this module.
    from stdapi.vector_stores.jobs import enqueue_indexing  # noqa: PLC0415

    if not await enqueue_indexing(store_id, file_ids, batch_id):
        start_indexing(store_id, file_ids, batch_id)


def start_indexing(store_id: str, file_ids: list[str], batch_id: str) -> None:
    """Run the indexing of *file_ids* in a task that outlives the request.

    Args:
        store_id: A validated vector store identifier.
        file_ids: The files to index, in order.
        batch_id: The batch the files belong to, or ``""``.
    """
    task = create_task(
        index_files(store_id, file_ids, batch_id, REQUEST_ID.get("vector_store"))
    )
    _INDEXING_TASKS[task] = _IndexingWave(store_id, tuple(file_ids))
    task.add_done_callback(lambda done: _INDEXING_TASKS.pop(done, None))


async def drain_indexing(timeout: float) -> int:  # noqa: ASYNC109 -- shared drain contract
    """Await the indexing still running, settling what the deadline leaves.

    A file whose task is cut short at the deadline is settled exactly as one
    whose task was killed is, so a shutdown leaves no record claiming an
    indexing nobody is doing any more.

    Args:
        timeout: Seconds allowed before the unfinished indexing is cancelled.

    Returns:
        Number of indexing tasks that had not finished at the deadline.
    """
    if not (waves := dict(_INDEXING_TASKS)):
        return 0
    unfinished = await drain_tasks(set(waves), timeout)
    for task, wave in waves.items():
        if task.cancelled() or not task.done():
            await gather_bounded(
                [
                    _settle_abandoned(wave.store_id, file_id)
                    for file_id in wave.file_ids
                ],
                RECORD_WAVE,
            )
    return unfinished


async def index_files(
    store_id: str, file_ids: list[str], batch_id: str, request_id: str
) -> None:
    """Index every file of a wave, updating the counters as each one settles.

    Runs its own usage scope so the embeddings it bills are recorded: a usage
    entry written after the originating request's log was finalized is dropped.

    One file is read, chunked and embedded at a time, and the wave holds one of
    the server's indexing slots while it does: the fan-out is caller-controlled,
    so what it costs in memory and in embedding calls may not grow with the
    request rate.

    Args:
        store_id: A validated vector store identifier.
        file_ids: The files to index, in order.
        batch_id: The batch the files belong to, or ``""``.
        request_id: Identifier correlating the work with its request.
    """
    with log_background_event("vector_store_indexing", request_id, record_usage=True):
        try:
            store = await read_store(store_id)
        except (ApiError, BotoCoreError, ClientError, OSError) as exc:
            log_error_details(f"Vector store indexing failed: {exc!r}", level="error")
            return
        # The lease attaching the files wrote; renewed for as long as this runs.
        lease = now_utc_timestamp() + _INDEXING_LEASE_SECONDS
        for file_id in file_ids:
            lease = await _hold_slot(store_id, lease)
            try:
                status, usage_bytes = await _index_one_file(store, file_id, batch_id)
            except (ApiError, BotoCoreError, ClientError, OSError) as exc:
                # Nothing else settles these files: an escape strands them in progress.
                log_error_details(
                    f"Vector store indexing failed: {exc!r}", level="error"
                )
                status, usage_bytes = await _fail_file(
                    store_id, file_id, "server_error", _FAILED_MESSAGE
                )
            finally:
                _INDEXING_SEMAPHORE.release()
            if status:
                await _settle_counters(store_id, batch_id, status, usage_bytes)


async def _hold_slot(store_id: str, lease: int) -> int:
    """Take an indexing slot, holding the store's lease while waiting for one.

    A file queued behind the slots is still being indexed, so the lease that
    tells a reader whether anything owns it is renewed for as long as the wait
    lasts.

    Args:
        store_id: A validated vector store identifier.
        lease: When the lease currently held on the store runs out.

    Returns:
        When the lease held once the slot was taken runs out.
    """
    while True:
        if now_utc_timestamp() + _LEASE_MARGIN_SECONDS >= lease:
            lease = await _renew_lease(store_id)
        try:
            await wait_for(_INDEXING_SEMAPHORE.acquire(), _LEASE_MARGIN_SECONDS)
        except TimeoutError:
            continue
        return lease


async def _renew_lease(store_id: str) -> int:
    """Push back the moment a store's files count as owned by nobody.

    Args:
        store_id: A validated vector store identifier.

    Returns:
        When the renewed lease runs out.
    """
    lease = now_utc_timestamp() + _INDEXING_LEASE_SECONDS

    def mutate(record: StoreRecord) -> None:
        """Hold the lease at least until *lease*."""
        record.indexing_expires_at = max(record.indexing_expires_at, lease)

    with suppress(ApiError, BotoCoreError, ClientError, OSError):
        await update_record(StoreRecord, store_key(store_id), mutate)
    return lease


async def _write_owned(
    store_id: str, file_id: str, mutate: Callable[[FileRecord], None]
) -> FileRecord | None:
    """Apply *mutate* to a file record the indexing that read it still owns.

    Written only while the stored record is unchanged, so a file detached,
    cancelled or already settled by another writer is left as it stands rather
    than resurrected by work that started before.

    Args:
        store_id: A validated vector store identifier.
        file_id: The file to write.
        mutate: Callable applying the change in place.

    Returns:
        The record as written, or ``None`` when the file is no longer this
        indexing's to settle.
    """
    key = file_key(store_id, file_id)
    current = await read_record(FileRecord, key)
    if current is None:
        return None
    record, etag = current
    if record.detaching or record.status != "in_progress":
        return None
    mutate(record)
    return record if await write_if_unchanged(key, record, etag) else None


async def _fail_file(
    store_id: str,
    file_id: str,
    code: Literal["server_error", "unsupported_file", "invalid_file"],
    message: str,
) -> tuple[str, int]:
    """Record an error on a file whose indexing could not finish.

    Args:
        store_id: A validated vector store identifier.
        file_id: The file that could not be indexed.
        code: The upstream error code the record reports.
        message: What the caller reads as the file's error.

    Returns:
        ``("failed", 0)``, or ``("", 0)`` when the record was not this
        indexing's to settle and its counters must stay untouched.
    """

    def mutate(record: FileRecord) -> None:
        """Move the file to its failed terminal state."""
        record.status = "failed"
        record.usage_bytes = 0
        record.last_error = FileErrorRecord(code=code, message=message)

    try:
        settled = await _write_owned(store_id, file_id, mutate)
    except (ApiError, BotoCoreError, ClientError, OSError) as exc:
        log_error_details(f"Vector store indexing failed: {exc!r}", level="error")
        return "", 0
    return ("failed", 0) if settled is not None else ("", 0)


async def _settle_abandoned(store_id: str, file_id: str) -> FileRecord | None:
    """Fail a file left ``in_progress`` by an indexing nothing is running.

    Args:
        store_id: A validated vector store identifier.
        file_id: The file left in progress.

    Returns:
        The settled record, or ``None`` when it was not this reader's to
        settle.
    """

    def mutate(record: FileRecord) -> None:
        """Move the file to its failed terminal state."""
        record.status = "failed"
        record.usage_bytes = 0
        record.last_error = FileErrorRecord(
            code="server_error", message=_INTERRUPTED_MESSAGE
        )

    settled = None
    with suppress(ApiError, BotoCoreError, ClientError, OSError):
        settled = await _write_owned(store_id, file_id, mutate)
        if settled is not None:
            await _settle_counters(store_id, settled.batch_id, "failed", 0)
    return settled


async def settle_interrupted(store_id: str, file_ids: Sequence[str]) -> None:
    """Fail the files of an indexing job that will not be attempted again.

    What a queued job's last delivery owes the caller: a file left
    ``in_progress`` by a job nobody will run again reads as work still in
    flight for ever, and re-attaching it is the only way forward.

    Args:
        store_id: A validated vector store identifier.
        file_ids: The files the job named.
    """
    await gather_bounded(
        [_settle_abandoned(store_id, file_id) for file_id in file_ids], RECORD_WAVE
    )


async def renew_indexing_lease(store_id: str) -> None:
    """Hold a store's indexing lease before a queued job starts running.

    A job may wait in the queue longer than the lease attaching its files
    wrote, and the first thing indexing does is read the store — which is
    where an expired lease settles those same files as abandoned.

    Args:
        store_id: A validated vector store identifier.
    """
    await _renew_lease(store_id)


async def _settle_abandoned_files(store_id: str) -> bool:
    """Fail every file of a store whose indexing no longer has a task.

    Args:
        store_id: A validated vector store identifier.

    Returns:
        Whether at least one file was settled.
    """
    horizon = now_utc_timestamp() - _INDEXING_LEASE_SECONDS
    stale = [
        record
        for record in await store_file_records(store_id, "in_progress")
        if record.created_at <= horizon
    ]
    if not stale:
        return False
    settled = await gather_bounded(
        [_settle_abandoned(store_id, record.id) for record in stale], RECORD_WAVE
    )
    return any(record is not None for record in settled)


async def _index_one_file(
    store: StoreRecord, file_id: str, batch_id: str
) -> tuple[str, int]:
    """Index one file and write its terminal record.

    Every write of the record goes through :func:`_write_owned`, so a file
    detached or settled while this ran is left alone and its counters are not
    moved twice.

    Args:
        store: The store record the file belongs to.
        file_id: The file to index.
        batch_id: The batch the file belongs to, or ``""``.

    Returns:
        ``(status, usage_bytes)`` of the settled file, or ``("", 0)`` when the
        file was not this indexing's to settle and its counters are answered
        for elsewhere.
    """
    backend = backend_for(store)
    try:
        record = await _read_file_record(store.id, file_id)
    except ApiError:
        return "", 0
    if record.status != "in_progress":
        # A replayed job: the file already settled, and re-indexing it would
        # embed and bill its chunks a second time for the same outcome.
        return "", 0
    if batch_id and (await _read_batch_record(store.id, batch_id)).cancel_requested:
        cancelled = await _write_owned(
            store.id, file_id, lambda stored: setattr(stored, "status", "cancelled")
        )
        return ("cancelled", 0) if cancelled is not None else ("", 0)
    try:
        chunks = await _load_chunks(store, record, backend)
    except _FileIndexingError as exc:
        return await _fail_file(store.id, file_id, exc.code, str(exc))
    except (ApiError, BotoCoreError, ClientError, OSError) as exc:
        log_error_details(f"Vector store indexing failed: {exc!r}", level="error")
        return await _fail_file(store.id, file_id, "server_error", _FAILED_MESSAGE)
    return await _store_chunks(store, file_id, chunks, backend)


async def _store_chunks(
    store: StoreRecord, file_id: str, chunks: list[str], backend: VectorIndex
) -> tuple[str, int]:
    """Embed a file's chunks into the index and complete its record.

    Args:
        store: The store record the file belongs to.
        file_id: The file being indexed.
        chunks: Its chunks, in document order.
        backend: The backend serving the store.

    Returns:
        ``(status, usage_bytes)`` of the settled file, or ``("", 0)`` when the
        file was not this indexing's to settle.
    """

    def count(stored: FileRecord) -> None:
        """Record what the vectors about to be written hold."""
        stored.chunk_count = len(chunks)
        stored.usage_bytes = sum(len(chunk.encode()) for chunk in chunks)

    # Stored before the vectors exist, so a mid-write failure orphans none.
    counted = await _write_owned(store.id, file_id, count)
    if counted is None:
        return "", 0
    try:
        await backend.put_vectors(store.id, _embedded_chunks(store, counted, chunks))
    except (ApiError, BotoCoreError, ClientError) as exc:
        log_error_details(f"Vector store indexing failed: {exc!r}", level="error")
        return await _fail_file(store.id, file_id, "server_error", _FAILED_MESSAGE)

    def complete(stored: FileRecord) -> None:
        """Move the file to its completed terminal state."""
        stored.previous_chunk_count = 0
        stored.status = "completed"

    settled = await _write_owned(store.id, file_id, complete)
    if settled is None:
        return "", 0
    # Chunks beyond the new count would stay searchable with stale text.
    stale = counted.previous_chunk_count
    if stale > len(chunks):
        await backend.delete_vectors(
            store.id,
            [vector_key(file_id, index) for index in range(len(chunks), stale)],
        )
    return settled.status, settled.usage_bytes


class _FileIndexingError(Exception):
    """A file the API cannot index, reported as the file's own error."""

    #: The upstream error code the file record reports.
    code: Literal["server_error", "unsupported_file", "invalid_file"] = "server_error"


class _UnsupportedFileError(_FileIndexingError):
    """The file's bytes are not indexable text."""

    code = "unsupported_file"


class _InvalidFileError(_FileIndexingError):
    """The file is text, but holds nothing this server can index."""

    code = "invalid_file"


def _unsupported_message(media_type: str, backend: VectorIndex) -> str:
    """Return how a file the store cannot index is explained to the caller.

    Args:
        media_type: The media type the file was uploaded with.
        backend: The backend serving the store the file is attached to.

    Returns:
        The message the file's ``last_error`` reports.
    """
    return unsupported_file_message(
        backend.capabilities, alternative=alternative_for(media_type, backend)
    )


async def _load_chunks(
    store: StoreRecord, record: FileRecord, backend: VectorIndex
) -> list[str]:
    """Read a file's text and split it into chunks.

    Args:
        store: The store the file is attached to.
        record: The file record.
        backend: The backend serving the store, which declares what it ingests.

    Returns:
        The chunks, in document order.

    Raises:
        _FileIndexingError: When the file holds no indexable text.
    """
    capabilities = backend.capabilities
    stream, content_type = await get_file_content(parse_file_id(record.id))
    media_type = content_type.split(";", 1)[0].strip()
    if not capabilities.may_ingest(media_type):
        raise _UnsupportedFileError(_unsupported_message(media_type, backend))
    # Zero means no limit, but indexing buffers the file, so it caps its own.
    limit = SETTINGS.max_input_file_size
    limit = min(limit, _MAX_INDEXABLE_BYTES) if limit else _MAX_INDEXABLE_BYTES
    body = bytearray()
    async for part in stream:
        body.extend(part)
        if len(body) > limit:
            msg = "The file is larger than this server accepts for indexing."
            raise _InvalidFileError(msg)
    try:
        text = body.decode()
    except UnicodeDecodeError:
        raise _UnsupportedFileError(_unsupported_message(media_type, backend)) from None
    del body
    if "\x00" in text:
        raise _UnsupportedFileError(_unsupported_message(media_type, backend))
    chunks = chunk_text(
        text,
        record.max_chunk_size_tokens,
        record.chunk_overlap_tokens,
        _max_input_characters(store.embedding_model),
        capabilities.max_chunk_bytes,
    )
    if not chunks:
        msg = "The file holds no text to index."
        raise _InvalidFileError(msg)
    return chunks


async def _embedded_chunks(
    store: StoreRecord, record: FileRecord, chunks: Sequence[str]
) -> AsyncIterator[IndexVector]:
    """Embed *chunks* in waves and yield them as the backend writes them.

    Args:
        store: The store the file is attached to.
        record: The file record.
        chunks: The chunks, in document order.

    Yields:
        One vector per chunk, in document order.
    """
    for start in range(0, len(chunks), _EMBED_WAVE):
        window = chunks[start : start + _EMBED_WAVE]
        vectors = await _embed(store.embedding_model, window)
        for offset, (chunk, vector) in enumerate(zip(window, vectors, strict=True)):
            yield IndexVector(
                key=vector_key(record.id, start + offset),
                file_id=record.id,
                filename=record.filename,
                chunk_index=start + offset,
                text=chunk,
                attributes=record.attributes or {},
                embedding=vector,
            )


async def _settle_counters(
    store_id: str, batch_id: str, status: str, usage_bytes: int
) -> None:
    """Move one file out of ``in_progress`` in the store and batch counters.

    Args:
        store_id: A validated vector store identifier.
        batch_id: The batch the file belongs to, or ``""``.
        status: The terminal status the file reached.
        usage_bytes: Indexed bytes the file contributes.
    """

    def mutate(counts: FileCountsRecord) -> None:
        """Move one file from ``in_progress`` to *status*."""
        counts.in_progress = max(0, counts.in_progress - 1)
        setattr(counts, status, getattr(counts, status) + 1)

    lease = now_utc_timestamp() + _INDEXING_LEASE_SECONDS

    def mutate_store(record: StoreRecord) -> None:
        """Apply the file's outcome to the store counters, and hold the lease."""
        mutate(record.file_counts)
        record.usage_bytes += usage_bytes
        # A file settling is the wave saying it is alive, for whatever is left.
        record.indexing_expires_at = max(record.indexing_expires_at, lease)

    with suppress(ApiError, BotoCoreError, ClientError):
        await update_record(StoreRecord, store_key(store_id), mutate_store)
    if batch_id:
        with suppress(ApiError, BotoCoreError, ClientError):
            await update_record(
                BatchRecord,
                batch_key(store_id, batch_id),
                lambda record: mutate(record.file_counts),
            )


async def detach_file(store_id: str, file_id: str) -> None:
    """Remove a file and its vectors from a store.

    The record naming the vectors is the only thing that can reclaim them, so
    it is the last thing to go: the file is marked as detaching, which takes it
    out of every listing and out of every search, and is deleted once its
    vectors are. A task killed in between leaves a record to finish from
    instead of vectors nothing points at, and the next read of the store
    finishes it.

    Args:
        store_id: A validated vector store identifier.
        file_id: The file to remove.

    Raises:
        ApiError: When the file is not attached to the store (404).
    """
    external = external_store_for(store_id)
    if external is not None:
        await external.delete_document(store_id, file_id)
        return
    record = await _read_file_record(store_id, file_id)
    reclaim = bool(record.chunk_count or record.previous_chunk_count)
    if reclaim:
        record = await update_record(
            FileRecord,
            file_key(store_id, file_id),
            lambda stored: setattr(stored, "detaching", True),
            resource="file",
        )

    def mutate(store: StoreRecord) -> None:
        """Release the file from the store counters, and name what it leaves."""
        counts = store.file_counts
        setattr(counts, record.status, max(0, getattr(counts, record.status) - 1))
        store.usage_bytes = max(0, store.usage_bytes - record.usage_bytes)
        if reclaim and record.id not in store.detaching:
            store.detaching.append(record.id)

    with suppress(ApiError, BotoCoreError, ClientError):
        await update_record(StoreRecord, store_key(store_id), mutate)
    if not reclaim:
        await delete_record(file_key(store_id, file_id))
        return
    schedule_cleanup(_reclaim_file(store_id, file_id))


async def _reclaim_detached(store: StoreRecord) -> bool:
    """Reclaim what the files detached from *store* have not finished leaving.

    Args:
        store: The store record naming them.

    Returns:
        Whether at least one file was reclaimed.
    """
    reclaimed = False
    for file_id in list(store.detaching):
        with suppress(ApiError, BotoCoreError, ClientError, OSError):
            await _reclaim_file(store.id, file_id)
            reclaimed = True
    return reclaimed


async def _reclaim_file(store_id: str, file_id: str) -> None:
    """Delete a detached file's vectors, then the record naming them.

    A file whose re-indexing failed still owns the vectors of the version
    before it, so both counts are reclaimed. Deleting more keys than the index
    holds is harmless, which is what makes a second attempt safe.

    Args:
        store_id: A validated vector store identifier.
        file_id: The detached file to reclaim.
    """
    key = file_key(store_id, file_id)
    current = await read_record(FileRecord, key)
    if current is not None:
        record = current[0]
        count = max(record.chunk_count, record.previous_chunk_count)
        if count:
            await backend_for(store_id).delete_vectors(
                store_id, [vector_key(file_id, index) for index in range(count)]
            )
        await delete_record(key)

    def mutate(store: StoreRecord) -> None:
        """Forget a file whose vectors are gone."""
        store.detaching = [entry for entry in store.detaching if entry != file_id]

    with suppress(ApiError, BotoCoreError, ClientError):
        await update_record(StoreRecord, store_key(store_id), mutate)


async def update_file_attributes(
    store_id: str, file_id: str, attributes: Attributes
) -> FileRecord:
    """Replace a file's attributes and re-write them onto its vectors.

    Args:
        store_id: A validated vector store identifier.
        file_id: The file to update.
        attributes: The new attributes.

    Returns:
        The updated file record.

    Raises:
        ApiError: When the file is not attached to the store (404), or the
            store keeps the attributes it was given at attachment (400).
    """
    external = external_store_for(store_id)
    if external is not None:
        await external.read_document(store_id, file_id)
        external.refuse(
            "the attributes of an attached file cannot be replaced. Attach the "
            "file again with the attributes it should carry."
        )
    await _read_file_record(store_id, file_id)
    record = await update_record(
        FileRecord,
        file_key(store_id, file_id),
        lambda stored: setattr(stored, "attributes", attributes),
        resource="file",
    )
    if record.chunk_count and record.status == "completed":
        schedule_cleanup(_rewrite_attributes(store_id, record))
    return record


async def _rewrite_attributes(store_id: str, record: FileRecord) -> None:
    """Re-write a file's vectors so they carry its new attributes."""
    backend = backend_for(store_id)
    keys = [vector_key(record.id, index) for index in range(record.chunk_count)]
    for start in range(0, len(keys), _REWRITE_WINDOW):
        stored = await backend.get_vectors(
            store_id, keys[start : start + _REWRITE_WINDOW], with_embeddings=True
        )
        rewritten = [vector for vector in stored if vector.embedding]
        for vector in rewritten:
            vector.attributes = record.attributes or {}
        if rewritten:
            await backend.put_vectors(store_id, as_stream(rewritten))


async def read_file_chunks(store_id: str, record: FileRecord) -> list[str]:
    """Return a file's indexed chunks, in document order.

    Args:
        store_id: A validated vector store identifier.
        record: The file record.

    Returns:
        The chunk texts.

    Raises:
        ApiError: When the store cuts its own passages and does not address
            them individually (400).
    """
    external = external_store_for(store_id)
    if external is not None:
        external.refuse(
            "the passages a file was indexed as cannot be listed. Download the "
            "file itself instead."
        )
    stored = await backend_for(store_id).get_vectors(
        store_id,
        [vector_key(record.id, index) for index in range(record.chunk_count)],
        with_embeddings=False,
    )
    texts = {vector.chunk_index: vector.text for vector in stored}
    return [texts[index] for index in sorted(texts)]


async def search(
    store: StoreRecord,
    queries: Sequence[str],
    *,
    max_num_results: int,
    filters: SearchFilter | None,
    score_threshold: float | None,
) -> list[SearchResult]:
    """Return the chunks of *store* closest to *queries*.

    Args:
        store: The store to search.
        queries: One or more query texts.
        max_num_results: Maximum results to return.
        filters: Restriction over the files' attributes, if any.
        score_threshold: Minimum score a result must reach, if any.

    Returns:
        The results, best score first.

    Raises:
        ApiError: When the backend cannot express the filter or the threshold
            (400).
    """
    if store.expired:
        return []
    backend = backend_for(store)
    capabilities = backend.capabilities
    if filters is not None:
        check_filter(filters, capabilities)
    if score_threshold is not None and not capabilities.normalised_score:
        msg = (
            "'ranking_options.score_threshold' is not available on this vector "
            "store: its relevance scores are not comparable between searches. "
            "Search it with 'max_num_results' instead."
        )
        raise ApiError(msg)
    external = external_store_for(store)
    if external is not None:
        matches = await external.query_text(
            store.id, queries, max_results=max_num_results, search_filter=filters
        )
    else:
        vectors = await _embed(store.embedding_model, list(queries))
        matches = await backend.query(
            store.id, vectors, max_results=max_num_results, search_filter=filters
        )
    # A file whose vectors are still being reclaimed is already gone: whatever
    # the index still answers for it must not reach the caller.
    detaching = frozenset(store.detaching)
    best: dict[str, SearchResult] = {}
    for match in matches:
        if match.file_id in detaching:
            continue
        if score_threshold is not None and match.score < score_threshold:
            continue
        current = best.get(match.key)
        if current is None or match.score > current.score:
            best[match.key] = SearchResult(
                file_id=match.file_id,
                filename=match.filename,
                score=match.score,
                text=match.text,
                attributes=match.attributes,
            )
    results = list(best.values())
    results.sort(key=lambda result: result.score, reverse=True)
    return results[:max_num_results]


def _native_configured() -> bool:
    """Whether this deployment can serve stores of its own.

    Returns:
        Whether both the record bucket and the vector index are configured. A
        deployment given only external stores serves those and nothing else.
    """
    if not SETTINGS.aws_s3_bucket:
        return False
    try:
        default_backend().check_configured()
    except ApiError:
        return False
    return True


async def list_stores(
    *, after: str, before: str, limit: int, order: str
) -> tuple[list[StoreRecord], bool]:
    """List every vector store, this deployment's own and the ones it addresses.

    A store held elsewhere is addressed by an identifier naming it rather than
    minted here, so it carries no creation time: the two sources are merged on
    the ``created_at`` every store reports, and the cursors name positions in
    that order.

    Args:
        after: Return the stores following this identifier.
        before: Return the page ending immediately before this identifier.
        limit: Maximum records to return.
        order: ``"asc"`` or ``"desc"``.

    Returns:
        ``(records, has_more)``.

    Raises:
        FeatureUnavailableError: When the deployment serves no vector store at
            all (503).
    """
    external: list[StoreRecord] = []
    for backend in external_stores():
        external.extend(await backend.list_stores())
    if not external:
        # Also where a deployment serving no store at all reports what it lacks.
        return await _list_store_records(
            after=after, before=before, limit=limit, order=order
        )
    records: list[StoreRecord] = []
    if _native_configured():
        # Read whole, so the two sources page as one listing.
        records, _ = await _list_store_records(
            after="", before="", limit=_LIST_ALL, order="asc"
        )
    return page_records(
        records + external, after=after, before=before, limit=limit, order=order
    )


async def list_store_files(
    store_id: str, *, after: str, before: str, limit: int, order: str, status: str
) -> tuple[list[FileRecord], bool]:
    """List the files attached to *store_id*.

    Args:
        store_id: A validated vector store identifier.
        after: Return files created strictly after this identifier.
        before: Return files created strictly before this identifier.
        limit: Maximum records to return.
        order: ``"asc"`` or ``"desc"``.
        status: Keep only files with this status, or ``""`` for all.

    Returns:
        ``(records, has_more)``.
    """
    external = external_store_for(store_id)
    if external is not None:
        return await external.list_documents(
            store_id,
            after=after,
            before=before,
            limit=limit,
            order=order,
            status=status,
        )
    return await _list_store_file_records(
        store_id, after=after, before=before, limit=limit, order=order, status=status
    )


async def read_file(store_id: str, file_id: str) -> FileRecord:
    """Return the record of *file_id* in *store_id*.

    Args:
        store_id: A validated vector store identifier.
        file_id: A file identifier.

    Returns:
        The file record.

    Raises:
        ApiError: When the file is not attached to the store (404).
    """
    external = external_store_for(store_id)
    if external is not None:
        return await external.read_document(store_id, file_id)
    return await _read_file_record(store_id, file_id)


async def read_batch(store_id: str, batch_id: str) -> BatchRecord:
    """Return the batch record of *batch_id* in *store_id*.

    Args:
        store_id: A validated vector store identifier.
        batch_id: A validated file batch identifier.

    Returns:
        The batch record.

    Raises:
        ApiError: When the batch does not exist (404), or the store takes no
            batch at all (400).
    """
    external = external_store_for(store_id)
    if external is not None:
        external.refuse(_NO_BATCH)
    return await _read_batch_record(store_id, batch_id)


async def list_batch_files(
    store_id: str,
    batch_id: str,
    *,
    after: str,
    before: str,
    limit: int,
    order: str,
    status: str,
) -> tuple[list[FileRecord], bool]:
    """List the files of one batch.

    Args:
        store_id: A validated vector store identifier.
        batch_id: A validated file batch identifier.
        after: Return files created strictly after this identifier.
        before: Return files created strictly before this identifier.
        limit: Maximum records to return.
        order: ``"asc"`` or ``"desc"``.
        status: Keep only files with this status, or ``""`` for all.

    Returns:
        ``(records, has_more)``.

    Raises:
        ApiError: When the store takes no batch at all (400).
    """
    external = external_store_for(store_id)
    if external is not None:
        external.refuse(_NO_BATCH)
    return await _list_batch_file_records(
        store_id,
        batch_id,
        after=after,
        before=before,
        limit=limit,
        order=order,
        status=status,
    )


async def cancel_batch(store_id: str, batch_id: str) -> BatchRecord:
    """Ask a file batch to stop indexing the files it has not started.

    Files that already finished keep their outcome; only the ones still waiting
    are reported as cancelled.

    Args:
        store_id: A validated vector store identifier.
        batch_id: A validated file batch identifier.

    Returns:
        The updated batch record.

    Raises:
        ApiError: When the batch does not exist (404).
    """
    await read_batch(store_id, batch_id)
    return await update_record(
        BatchRecord,
        batch_key(store_id, batch_id),
        lambda record: setattr(record, "cancel_requested", True),
        resource="file batch",
    )
