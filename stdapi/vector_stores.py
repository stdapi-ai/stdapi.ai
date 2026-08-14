"""Vector Stores engine — indexing, bookkeeping and semantic search.

A vector store is a vector index plus the bookkeeping the API answers with:
per-store counters, per-file status and per-batch progress. The index holds no
such state, so it lives in JSON objects in the application bucket, one per
store, per attached file and per batch, mutated with conditional writes.

Consistency rule: a file record reaches its terminal state **before** the store
counters that summarise it. The counters may lag the file listing, never lead
it — a store still reporting ``in_progress`` for a file already ``completed``
converges, while counters claiming a completion the listing cannot show does not.
"""

from asyncio import Task, create_task, gather
from base64 import b32hexencode
from contextlib import suppress
from dataclasses import dataclass, field
from re import compile as re_compile
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn
from uuid import uuid7

from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, Field, JsonValue

from stdapi.api_errors import (
    ApiError,
    FeatureUnavailableError,
    feature_unavailable_guard,
)
from stdapi.aws import get_client
from stdapi.aws_s3 import BUCKET_TO_REGION, S3_TAGGING
from stdapi.cleanup import schedule_cleanup
from stdapi.config import SETTINGS
from stdapi.files import get_file, get_file_content, parse_file_id
from stdapi.models import validate_model
from stdapi.models.embedding import get_embedding_model
from stdapi.monitoring import REQUEST_ID, log_background_event, log_error_details
from stdapi.types.openai_vector_stores import (
    CHUNK_OVERLAP_TOKENS_DEFAULT,
    CHUNK_SIZE_TOKENS_DEFAULT,
    Attributes,
    AttributeValue,
    ComparisonFilter,
    CompoundFilter,
    SearchFilter,
)
from stdapi.utils import now_utc_timestamp, to_json_bytes, validation_error_handler

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from contextlib import AbstractContextManager

    from types_aiobotocore_s3.client import S3Client

    from stdapi.input_file import InputFileUrl
    from stdapi.types import JsonMapping

#: The feature name a caller reads when the deployment cannot serve vector stores.
_FEATURE: str = "The Vector Stores API"

#: What an unreachable vector endpoint means, for the operator.
_UNREACHABLE_DETAIL: str = (
    "The S3 Vectors endpoint is unreachable or timed out: S3 Vectors is offered "
    "in fewer regions than model inference; set 'aws_s3_vectors_region' to a "
    "region that provides it."
)

#: Regex pattern a vector store identifier must match on input.
VECTOR_STORE_ID_PATTERN: str = r"^vs_[0-9a-v]{26}$"

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

#: Distance metric every index is created with.
_DISTANCE_METRIC: Final = "cosine"

#: Vector element type every index is created with.
_DATA_TYPE: Final = "float32"

#: Metadata key holding the chunk text; never usable in a filter.
_TEXT_KEY: Final = "_text"

#: Metadata key holding the source file name.
_FILENAME_KEY: Final = "_filename"

#: Metadata key holding the source file identifier.
_FILE_ID_KEY: Final = "_file_id"

#: Metadata key holding the chunk position within its file.
_CHUNK_INDEX_KEY: Final = "_chunk_index"

#: Metadata keys the index stores but never filters on; fixed at index creation.
_NON_FILTERABLE_KEYS: Final[tuple[str, ...]] = (
    _TEXT_KEY,
    _FILENAME_KEY,
    _FILE_ID_KEY,
    _CHUNK_INDEX_KEY,
)

#: Prefix isolating caller attribute keys from the keys above.
_ATTRIBUTE_PREFIX: Final = "a_"

#: Bytes of filterable metadata one vector accepts.
_MAX_FILTERABLE_BYTES: Final[int] = 2048

#: Bytes of chunk text one vector holds, leaving room for the other metadata.
_MAX_CHUNK_BYTES: Final[int] = 32768

#: Bytes of one file indexing buffers, whatever ``max_input_file_size`` allows.
_MAX_INDEXABLE_BYTES: Final[int] = 100 * 1024 * 1024

#: Vectors written per index write.
_PUT_VECTORS_BATCH: Final[int] = 500

#: Vector keys read per index read.
_GET_VECTORS_BATCH: Final[int] = 100

#: Vector keys deleted per index delete.
_DELETE_VECTORS_BATCH: Final[int] = 500

#: Chunks embedded concurrently while indexing one file.
_EMBED_WAVE: Final[int] = 16

#: Record reads or writes issued concurrently.
_RECORD_WAVE: Final[int] = 32

#: Records one listing call scans before paging in memory.
_LIST_SCAN_MAX: Final[int] = 1000

#: Attempts a conditional record update makes before giving up.
_CAS_ATTEMPTS: Final[int] = 8

#: Seconds of inactivity before a search refreshes ``last_active_at``.
_LAST_ACTIVE_REFRESH_SECONDS: Final[int] = 3600

#: Seconds in a day, for the ``expires_after`` anchor arithmetic.
_SECONDS_PER_DAY: Final[int] = 86400

#: S3 error codes meaning the object is absent.
_MISSING_CODES: Final[frozenset[str]] = frozenset({"404", "NoSuchKey"})

#: S3 error codes a conditional write returns when another writer won.
_CONFLICT_CODES: Final[frozenset[str]] = frozenset(
    {"PreconditionFailed", "ConditionalRequestConflict"}
)

#: Content types whose bytes are never text, rejected before decoding is attempted.
_BINARY_CONTENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "application/gzip",
        "application/msword",
        "application/octet-stream",
        "application/pdf",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
    }
)

#: OpenAI comparison operator → index filter operator.
_FILTER_OPERATORS: Final[dict[str, str]] = {
    "eq": "$eq",
    "ne": "$ne",
    "gt": "$gt",
    "gte": "$gte",
    "lt": "$lt",
    "lte": "$lte",
    "in": "$in",
    "nin": "$nin",
}

#: Strong references to running indexing tasks, held until they finish.
_INDEXING_TASKS: Final[set[Task[None]]] = set()


class FileCountsRecord(BaseModel):
    """Per-status file counts of a store or a batch."""

    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0

    @property
    def total(self) -> int:
        """Total number of files counted."""
        return self.in_progress + self.completed + self.failed + self.cancelled


class StoreRecord(BaseModel):
    """The bookkeeping of one vector store.

    ``embedding_model`` and ``dimensions`` are frozen at creation: an index's
    dimension cannot be changed, so a later change to the configured default
    must not reach an existing store.
    """

    id: str
    created_at: int
    last_active_at: int
    name: str = ""
    description: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)
    expires_after_days: int | None = None
    embedding_model: str
    dimensions: int
    max_chunk_size_tokens: int = CHUNK_SIZE_TOKENS_DEFAULT
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS_DEFAULT
    file_counts: FileCountsRecord = Field(default_factory=FileCountsRecord)
    usage_bytes: int = 0
    index_deleted: bool = False

    @property
    def expires_at(self) -> int | None:
        """Unix timestamp the store expires at, or ``None`` when it never does."""
        if self.expires_after_days is None:
            return None
        return self.last_active_at + self.expires_after_days * _SECONDS_PER_DAY

    @property
    def expired(self) -> bool:
        """Whether the store has reached its expiration."""
        expires_at = self.expires_at
        return expires_at is not None and now_utc_timestamp() >= expires_at

    @property
    def status(self) -> Literal["expired", "in_progress", "completed"]:
        """The store status the API reports."""
        if self.expired:
            return "expired"
        return "in_progress" if self.file_counts.in_progress else "completed"


class FileErrorRecord(BaseModel):
    """Why a file failed to be indexed."""

    code: Literal["server_error", "unsupported_file", "invalid_file"]
    message: str


class FileRecord(BaseModel):
    """The bookkeeping of one file attached to a vector store."""

    id: str
    created_at: int
    filename: str = ""
    status: Literal["in_progress", "completed", "cancelled", "failed"] = "in_progress"
    last_error: FileErrorRecord | None = None
    usage_bytes: int = 0
    chunk_count: int = 0
    previous_chunk_count: int = 0
    attributes: Attributes = Field(default_factory=dict)
    max_chunk_size_tokens: int = CHUNK_SIZE_TOKENS_DEFAULT
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS_DEFAULT
    batch_id: str = ""


class BatchRecord(BaseModel):
    """The bookkeeping of one file batch."""

    id: str
    created_at: int
    file_counts: FileCountsRecord = Field(default_factory=FileCountsRecord)
    cancel_requested: bool = False

    @property
    def status(self) -> Literal["in_progress", "completed", "cancelled", "failed"]:
        """The batch status the API reports."""
        if self.file_counts.in_progress:
            return "in_progress"
        if self.file_counts.cancelled:
            return "cancelled"
        return "failed" if self.file_counts.failed else "completed"


@dataclass(slots=True)
class SearchResult:
    """One search hit, as the search route and the retrieval tool both consume it.

    Attributes:
        file_id: Identifier of the file the chunk comes from.
        filename: Name of that file.
        score: Similarity in ``[0, 1]``, 1 being an exact match.
        text: The matching chunk text.
        attributes: The attributes stored with the file.
    """

    file_id: str
    filename: str
    score: float
    text: str
    attributes: Attributes


@dataclass(slots=True)
class PendingFile:
    """One file waiting to be indexed.

    Attributes:
        file_id: Identifier of the uploaded file.
        attributes: Attributes to store with every chunk.
        max_chunk_size_tokens: Chunk size for this file.
        chunk_overlap_tokens: Chunk overlap for this file.
    """

    file_id: str
    attributes: Attributes = field(default_factory=dict)
    max_chunk_size_tokens: int = CHUNK_SIZE_TOKENS_DEFAULT
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS_DEFAULT


def raise_not_found(resource: str, identifier: str) -> NoReturn:
    """Raise the 404 answering an unknown store, file or batch.

    Args:
        resource: Human-readable resource name.
        identifier: The identifier that was not found.

    Raises:
        ApiError: Always (404).
    """
    msg = f"No {resource} found with id '{identifier}'."
    raise ApiError(msg, status=404)


def _raise_unavailable() -> NoReturn:
    """Raise the 503 answering a deployment with no vector storage configured.

    Raises:
        FeatureUnavailableError: Always (503).
    """
    raise FeatureUnavailableError(
        _FEATURE,
        "Vector storage not configured (aws_s3_vectors_bucket, aws_s3_bucket): "
        "the Vector Stores API is disabled.",
    )


def _vectors_guard(*actions: str) -> AbstractContextManager[None]:
    """Answer a denied index call as the Vector Stores API being unavailable.

    Args:
        *actions: The ``s3vectors`` actions the guarded call needs.

    Returns:
        The guard wrapping the call.
    """
    permissions = ", ".join(f"s3vectors:{action}" for action in actions)
    return feature_unavailable_guard(
        _FEATURE,
        missing=f"{permissions} on the vector bucket set in 'aws_s3_vectors_bucket'",
        unreachable=_UNREACHABLE_DETAIL,
    )


def records_bucket() -> str:
    """Return the application bucket holding the vector store bookkeeping.

    Returns:
        The bucket name.

    Raises:
        ApiError: When the deployment has no vector storage configured (503).
    """
    if SETTINGS.aws_s3_bucket and SETTINGS.aws_s3_vectors_bucket:
        return SETTINGS.aws_s3_bucket
    _raise_unavailable()


def _vector_bucket() -> str:
    """Return the configured vector bucket.

    Returns:
        The vector bucket name.

    Raises:
        ApiError: When the deployment has no vector storage configured (503).
    """
    if SETTINGS.aws_s3_bucket and SETTINGS.aws_s3_vectors_bucket:
        return SETTINGS.aws_s3_vectors_bucket
    _raise_unavailable()


def _records_client() -> S3Client:
    """Return the S3 client for the bucket holding the bookkeeping records."""
    client: S3Client = get_client("s3", BUCKET_TO_REGION.get(records_bucket()))
    return client


def _vectors_client() -> Any:  # noqa: ANN401 - no published stubs for this client
    """Return the client pinned to the region holding the vector bucket."""
    return get_client("s3vectors", SETTINGS.aws_s3_vectors_region)


def new_store_id() -> str:
    """Return a new vector store identifier that sorts by creation time."""
    return f"vs_{b32hexencode(uuid7().bytes).decode().rstrip('=').lower()}"


def new_batch_id() -> str:
    """Return a new file batch identifier that sorts by creation time."""
    return f"vsfb_{b32hexencode(uuid7().bytes).decode().rstrip('=').lower()}"


def index_name(store_id: str) -> str:
    """Return the index name backing *store_id*.

    An index name accepts neither underscores nor uppercase, so the identifier's
    separator is substituted; the mapping stays total and reversible.

    Args:
        store_id: A vector store identifier.

    Returns:
        The index name.
    """
    return store_id.replace("_", "-", 1)


def parse_store_id(store_id: str) -> str:
    """Validate a vector store identifier.

    Args:
        store_id: The identifier to validate.

    Returns:
        The identifier, unchanged.

    Raises:
        ApiError: When the identifier is malformed (404, as upstream reports it).
    """
    if not _STORE_ID_RE(store_id):
        raise_not_found("vector store", store_id)
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


def _store_prefix(store_id: str) -> str:
    """Return the key prefix of every record of *store_id*."""
    return f"{SETTINGS.aws_s3_vector_stores_prefix}{store_id}/"


def _store_key(store_id: str) -> str:
    """Return the key of the store record of *store_id*."""
    return f"{_store_prefix(store_id)}store.json"


def _file_key(store_id: str, file_id: str) -> str:
    """Return the key of the file record of *file_id* in *store_id*."""
    return f"{_store_prefix(store_id)}files/{file_id}.json"


def _batch_key(store_id: str, batch_id: str) -> str:
    """Return the key of the batch record of *batch_id* in *store_id*."""
    return f"{_store_prefix(store_id)}batches/{batch_id}.json"


async def _read[RecordT: BaseModel](
    model: type[RecordT], key: str
) -> tuple[RecordT, str] | None:
    """Read a record and its entity tag.

    Args:
        model: The record model to parse the object into.
        key: Object key of the record.

    Returns:
        ``(record, etag)``, or ``None`` when the record does not exist.
    """
    try:
        response = await _records_client().get_object(Bucket=records_bucket(), Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in _MISSING_CODES:
            return None
        raise
    return model.model_validate_json(await response["Body"].read()), response["ETag"]


async def _write(key: str, record: BaseModel, *, etag: str | None = None) -> None:
    """Write a record, conditionally when *etag* is given.

    Args:
        key: Object key of the record.
        record: The record to serialise.
        etag: Entity tag the stored object must still carry; ``"*"`` requires
            that no object exists yet.

    Raises:
        ClientError: When the condition failed, or the write itself did.
    """
    conditions: dict[str, str] = {}
    if etag == "*":
        conditions["IfNoneMatch"] = "*"
    elif etag is not None:
        conditions["IfMatch"] = etag
    await _records_client().put_object(
        Bucket=records_bucket(),
        Key=key,
        Body=record.model_dump_json().encode(),
        ContentType="application/json",
        Tagging=S3_TAGGING,
        **conditions,  # type: ignore[arg-type]
    )


async def update_record[RecordT: BaseModel](
    model: type[RecordT],
    key: str,
    mutate: Callable[[RecordT], None],
    resource: str = "vector store",
) -> RecordT:
    """Apply *mutate* to the record at *key* under a compare-and-swap loop.

    Args:
        model: The record model to parse the object into.
        key: Object key of the record.
        mutate: Callable applying the change in place.
        resource: Human-readable name of the record, for the error messages.

    Returns:
        The updated record.

    Raises:
        ApiError: When the record was deleted meanwhile (404) or is too
            contended (409).
    """
    for _ in range(_CAS_ATTEMPTS):
        current = await _read(model, key)
        if current is None:
            # Deleted between the read and this write; the key stays out of the message.
            msg = f"The {resource} was deleted while this request ran."
            raise ApiError(msg, status=404)
        record, etag = current
        mutate(record)
        try:
            await _write(key, record, etag=etag)
        except ClientError as exc:
            if exc.response["Error"]["Code"] in _CONFLICT_CODES:
                continue
            raise
        return record
    msg = f"The {resource} is being updated by another request. Retry the request."
    raise ApiError(msg, status=409)


async def read_store(store_id: str) -> StoreRecord:
    """Return the store record of *store_id*, deleting its index once expired.

    Args:
        store_id: A validated vector store identifier.

    Returns:
        The store record.

    Raises:
        ApiError: When the store does not exist (404).
    """
    current = await _read(StoreRecord, _store_key(store_id))
    if current is None:
        raise_not_found("vector store", store_id)
    record = current[0]
    if record.expired and not record.index_deleted:
        schedule_cleanup(_release_expired(store_id))
    return record


async def _release_expired(store_id: str) -> None:
    """Delete the index of an expired store, once.

    The store itself stays readable with an ``expired`` status, as upstream
    reports it; only the storage behind it is reclaimed.

    Args:
        store_id: A validated vector store identifier.
    """
    # Recorded after the delete, so a failed delete is retried by a later read.
    with suppress(ApiError, BotoCoreError, ClientError):
        await _delete_index(store_id)
        await update_record(
            StoreRecord,
            _store_key(store_id),
            lambda r: setattr(r, "index_deleted", True),
        )


async def _delete_index(store_id: str) -> None:
    """Delete the index backing *store_id*, ignoring an already-deleted one."""
    with _vectors_guard("DeleteIndex"):
        try:
            await _vectors_client().delete_index(
                vectorBucketName=_vector_bucket(), indexName=index_name(store_id)
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "NotFoundException":
                raise


async def read_file(store_id: str, file_id: str) -> FileRecord:
    """Return the file record of *file_id* in *store_id*.

    Args:
        store_id: A validated vector store identifier.
        file_id: A file identifier.

    Returns:
        The file record.

    Raises:
        ApiError: When the file is not attached to the store (404).
    """
    current = await _read(FileRecord, _file_key(store_id, file_id))
    if current is None:
        raise_not_found("file", file_id)
    return current[0]


async def read_batch(store_id: str, batch_id: str) -> BatchRecord:
    """Return the batch record of *batch_id* in *store_id*.

    Args:
        store_id: A validated vector store identifier.
        batch_id: A validated file batch identifier.

    Returns:
        The batch record.

    Raises:
        ApiError: When the batch does not exist (404).
    """
    current = await _read(BatchRecord, _batch_key(store_id, batch_id))
    if current is None:
        raise_not_found("file batch", batch_id)
    return current[0]


async def _list_ids(
    prefix: str, delimiter: str, *, after: str, limit: int
) -> tuple[list[str], bool]:
    """List the identifiers under *prefix* in ascending order.

    Args:
        prefix: Key prefix to list under.
        delimiter: ``"/"`` to list sub-prefixes, ``""`` to list objects.
        after: Identifier to start strictly after, or ``""``.
        limit: Maximum identifiers to return.

    Returns:
        ``(identifiers, has_more)``.
    """
    arguments: dict[str, str | int] = {
        "Bucket": records_bucket(),
        "Prefix": prefix,
        "MaxKeys": limit + 1,
    }
    if delimiter:
        arguments["Delimiter"] = delimiter
    if after:
        # ``StartAfter`` excludes the key, not the record; the extras drop below.
        arguments["StartAfter"] = f"{prefix}{after}"
        arguments["MaxKeys"] = limit + 2
    response = await _records_client().list_objects_v2(**arguments)  # type: ignore[arg-type]
    if delimiter:
        identifiers = [
            entry["Prefix"][len(prefix) : -1]
            for entry in response.get("CommonPrefixes", ())
            if "Prefix" in entry
        ]
    else:
        identifiers = [
            entry["Key"][len(prefix) :].removesuffix(".json")
            for entry in response.get("Contents", ())
        ]
    if after:
        identifiers = [identifier for identifier in identifiers if identifier > after]
    return identifiers[:limit], len(identifiers) > limit


async def list_stores(
    *, after: str, before: str, limit: int, order: str
) -> tuple[list[StoreRecord], bool]:
    """List vector stores, newest first by default.

    Args:
        after: Return stores created strictly after this identifier.
        before: Return stores created strictly before this identifier.
        limit: Maximum records to return.
        order: ``"asc"`` or ``"desc"``.

    Returns:
        ``(records, has_more)``.
    """
    prefix = SETTINGS.aws_s3_vector_stores_prefix
    if order == "asc" and not before:
        ids, has_more = await _list_ids(prefix, "/", after=after, limit=limit)
    else:
        ids, has_more = await _list_ids(prefix, "/", after="", limit=_LIST_SCAN_MAX)
        if after:
            ids = (
                [i for i in ids if i > after]
                if order == "asc"
                else [i for i in ids if i < after]
            )
        if before:
            ids = [i for i in ids if i < before]
        if order == "desc":
            ids = ids[::-1]
        has_more = len(ids) > limit
        ids = ids[:limit]
    records = [
        record[0]
        for record in (await _gather_records(StoreRecord, [_store_key(i) for i in ids]))
        if record is not None
    ]
    return records, has_more


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
    prefix = f"{_store_prefix(store_id)}files/"
    ids, _ = await _list_ids(prefix, "", after="", limit=_LIST_SCAN_MAX)
    if after:
        ids = (
            [i for i in ids if i > after]
            if order == "asc"
            else [i for i in ids if i < after]
        )
    if before:
        ids = [i for i in ids if i < before]
    if order == "desc":
        ids = ids[::-1]
    records = [
        record[0]
        for record in (
            await _gather_records(FileRecord, [_file_key(store_id, i) for i in ids])
        )
        if record is not None
    ]
    if status:
        records = [r for r in records if r.status == status]
    return records[:limit], len(records) > limit


async def _bounded[ResultT](awaitables: list[Awaitable[ResultT]]) -> list[ResultT]:
    """Await *awaitables* concurrently, in waves, keeping their order.

    Args:
        awaitables: The coroutines to run.

    Returns:
        Their results, in the order they were given.
    """
    results: list[ResultT] = []
    for start in range(0, len(awaitables), _RECORD_WAVE):
        results.extend(await gather(*awaitables[start : start + _RECORD_WAVE]))
    return results


async def _gather_records[RecordT: BaseModel](
    model: type[RecordT], keys: Sequence[str]
) -> list[tuple[RecordT, str] | None]:
    """Read several records concurrently, in the order of *keys*."""
    return await _bounded([_read(model, key) for key in keys])


def chunk_text(
    text: str,
    max_chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    max_characters: int,
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
            chunks.extend(_split_on_bytes(chunk))
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _split_on_bytes(chunk: str) -> list[str]:
    """Split *chunk* so every piece fits the per-vector text budget.

    Args:
        chunk: A chunk already within its character budget.

    Returns:
        The chunk alone, or the pieces it was split into.
    """
    if len(chunk.encode()) <= _MAX_CHUNK_BYTES:
        return [chunk]
    pieces: list[str] = []
    remaining = chunk
    while remaining:
        encoded = remaining.encode()[:_MAX_CHUNK_BYTES]
        piece = encoded.decode(errors="ignore")
        pieces.append(piece)
        remaining = remaining[len(piece) :]
    return pieces


def attribute_key(key: str) -> str:
    """Return the metadata key storing the caller attribute *key*."""
    return f"{_ATTRIBUTE_PREFIX}{key}"


def check_attributes(attributes: Attributes) -> None:
    """Reject attributes that exceed the searchable-attribute budget.

    Args:
        attributes: The caller-supplied attributes.

    Raises:
        ApiError: When the attributes do not fit the per-file budget (400).
    """
    if not attributes:
        return
    size = len(to_json_bytes({attribute_key(k): v for k, v in attributes.items()}))
    if size > _MAX_FILTERABLE_BYTES:
        msg = (
            f"The 'attributes' of this file take {size} bytes, above the "
            f"{_MAX_FILTERABLE_BYTES}-byte limit for searchable attributes. "
            "Use fewer keys, or shorter values."
        )
        raise ApiError(msg)


def translate_filter(search_filter: SearchFilter | JsonMapping) -> JsonMapping:
    """Translate an upstream search filter into an index filter.

    Args:
        search_filter: The filter as the API received it. A filter nested more
            than one level deep arrives as a plain mapping and is validated here.

    Returns:
        The equivalent index filter.

    Raises:
        RequestValidationError: When a nested filter is not a valid filter.
    """
    if isinstance(search_filter, dict):
        with validation_error_handler():
            search_filter = (
                CompoundFilter.model_validate(search_filter)
                if search_filter.get("type") in ("and", "or")
                else ComparisonFilter.model_validate(search_filter)
            )
    if isinstance(search_filter, CompoundFilter):
        return {
            f"${search_filter.type}": [
                translate_filter(inner) for inner in search_filter.filters
            ]
        }
    value: JsonValue = search_filter.value  # type: ignore[assignment]
    return {
        attribute_key(search_filter.key): {_FILTER_OPERATORS[search_filter.type]: value}
    }


def score_from_distance(distance: float) -> float:
    """Convert a cosine distance into the similarity score the API reports.

    The distance is ``1 - cosine_similarity``, so an exact match scores 1 and an
    orthogonal one scores 0; opposite vectors clamp to 0 rather than reporting a
    negative score.

    Args:
        distance: The distance the index returned.

    Returns:
        A score in ``[0, 1]``.
    """
    return max(0.0, min(1.0, 1.0 - distance))


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
    model_id, dimensions = await resolve_embedding_model()
    store_id = new_store_id()
    now = now_utc_timestamp()
    with _vectors_guard("CreateIndex"):
        await _vectors_client().create_index(
            vectorBucketName=_vector_bucket(),
            indexName=index_name(store_id),
            dataType=_DATA_TYPE,
            dimension=dimensions,
            distanceMetric=_DISTANCE_METRIC,
            metadataConfiguration={
                "nonFilterableMetadataKeys": list(_NON_FILTERABLE_KEYS)
            },
        )
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
        await _write(_store_key(store_id), record, etag="*")
    except BotoCoreError, ClientError:
        schedule_cleanup(_delete_index(store_id))
        raise
    return record


async def delete_store(store_id: str) -> None:
    """Delete a vector store, its records and its index.

    The records are deleted first, so a partial failure leaves an unreferenced
    index rather than a store the client can still see.

    Args:
        store_id: A validated vector store identifier.

    Raises:
        ApiError: When the store does not exist (404).
    """
    await read_store(store_id)
    prefix = _store_prefix(store_id)
    await _delete_record(_store_key(store_id))
    schedule_cleanup(
        *(_delete_record(key) for key in await _all_record_keys(prefix)),
        _delete_index(store_id),
    )


async def _delete_record(key: str) -> None:
    """Delete one bookkeeping record."""
    await _records_client().delete_object(Bucket=records_bucket(), Key=key)


async def _all_record_keys(prefix: str) -> list[str]:
    """Return every record key under *prefix*, however many pages it takes.

    Deletion is the one listing that may not stop at a page: a record left
    behind holds caller data a deleted store promised to remove.
    """
    keys: list[str] = []
    arguments: dict[str, str | int] = {
        "Bucket": records_bucket(),
        "Prefix": prefix,
        "MaxKeys": _LIST_SCAN_MAX,
    }
    while True:
        response = await _records_client().list_objects_v2(**arguments)  # type: ignore[arg-type]
        keys.extend(entry["Key"] for entry in response.get("Contents", ()))
        token = response.get("NextContinuationToken")
        if not response.get("IsTruncated") or not token:
            return keys
        arguments["ContinuationToken"] = token


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
        ApiError: When the store does not exist (404).
    """
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

    return await update_record(StoreRecord, _store_key(store_id), mutate)


async def touch_store(record: StoreRecord) -> None:
    """Refresh ``last_active_at`` when it has gone stale.

    Written coarsely on purpose: it anchors the expiration, which is measured in
    days, so a conditional write per search would only add contention.

    An expired store is never refreshed: its index has already been released,
    so a refresh would leave a store reporting ``completed`` over nothing.

    Args:
        record: The store record a request just used.
    """
    if record.expired or record.index_deleted:
        return
    now = now_utc_timestamp()
    if now - record.last_active_at < _LAST_ACTIVE_REFRESH_SECONDS:
        return
    with suppress(ApiError, BotoCoreError, ClientError):
        await update_record(
            StoreRecord,
            _store_key(record.id),
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
    now = now_utc_timestamp()
    # One file is one record: a repeated id would inflate the totals for good.
    unique: dict[str, PendingFile] = {}
    for entry in pending:
        unique.setdefault(entry.file_id, entry)
    pending = list(unique.values())
    sources = await _bounded(
        [get_file(parse_file_id(entry.file_id)) for entry in pending]
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
    replaced = [
        existing[0]
        for existing in await _gather_records(
            FileRecord, [_file_key(store.id, record.id) for record in records]
        )
        if existing is not None
    ]
    # An unfinished replacement still owns the older vectors: take the larger.
    previous = {
        record.id: max(record.chunk_count, record.previous_chunk_count)
        for record in replaced
    }
    for record in records:
        record.previous_chunk_count = previous.get(record.id, 0)
    await _bounded(
        [_write(_file_key(store.id, record.id), record) for record in records]
    )

    def add_pending(stored: StoreRecord) -> None:
        """Count the newly attached files, releasing the replaced ones."""
        counts = stored.file_counts
        for old in replaced:
            setattr(counts, old.status, max(0, getattr(counts, old.status) - 1))
            stored.usage_bytes = max(0, stored.usage_bytes - old.usage_bytes)
        counts.in_progress += len(records)

    await update_record(StoreRecord, _store_key(store.id), add_pending)
    if batch_id:
        await _write(
            _batch_key(store.id, batch_id),
            BatchRecord(
                id=batch_id,
                created_at=now,
                file_counts=FileCountsRecord(in_progress=len(records)),
            ),
        )
    _start_indexing(store.id, [r.id for r in records], batch_id)
    return records


def _start_indexing(store_id: str, file_ids: list[str], batch_id: str) -> None:
    """Run the indexing of *file_ids* in a task that outlives the request.

    Args:
        store_id: A validated vector store identifier.
        file_ids: The files to index, in order.
        batch_id: The batch the files belong to, or ``""``.
    """
    task = create_task(
        _index_files(store_id, file_ids, batch_id, REQUEST_ID.get("vector_store"))
    )
    _INDEXING_TASKS.add(task)
    task.add_done_callback(_INDEXING_TASKS.discard)


async def _index_files(
    store_id: str, file_ids: list[str], batch_id: str, request_id: str
) -> None:
    """Index every file of a wave, updating the counters as each one settles.

    Runs its own usage scope so the embeddings it bills are recorded: a usage
    entry written after the originating request's log was finalized is dropped.

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
        for file_id in file_ids:
            try:
                status, usage_bytes = await _index_one_file(store, file_id, batch_id)
            except (ApiError, BotoCoreError, ClientError, OSError) as exc:
                # Nothing else settles these files: an escape strands them in progress.
                log_error_details(
                    f"Vector store indexing failed: {exc!r}", level="error"
                )
                status, usage_bytes = await _fail_file(store_id, file_id), 0
            if status:
                await _settle_counters(store_id, batch_id, status, usage_bytes)


async def _fail_file(store_id: str, file_id: str) -> str:
    """Record a server error on a file whose indexing could not finish.

    Args:
        store_id: A validated vector store identifier.
        file_id: The file that could not be indexed.

    Returns:
        ``"failed"``, or ``""`` when the record could not be written and its
        counters must therefore stay untouched.
    """

    def mutate(record: FileRecord) -> None:
        """Move the file to its failed terminal state."""
        record.status = "failed"
        record.usage_bytes = 0
        record.last_error = FileErrorRecord(
            code="server_error", message="The file could not be indexed."
        )

    try:
        await update_record(
            FileRecord, _file_key(store_id, file_id), mutate, resource="file"
        )
    except (ApiError, BotoCoreError, ClientError, OSError) as exc:
        log_error_details(f"Vector store indexing failed: {exc!r}", level="error")
        return ""
    return "failed"


async def _index_one_file(
    store: StoreRecord, file_id: str, batch_id: str
) -> tuple[str, int]:
    """Index one file and write its terminal record.

    Args:
        store: The store record the file belongs to.
        file_id: The file to index.
        batch_id: The batch the file belongs to, or ``""``.

    Returns:
        ``(status, usage_bytes)`` of the settled file, or ``("", 0)`` when the
        file was detached meanwhile and its counters already released.
    """
    try:
        record = await read_file(store.id, file_id)
    except ApiError:
        return "", 0
    if batch_id and (await read_batch(store.id, batch_id)).cancel_requested:
        record.status = "cancelled"
        await _write(_file_key(store.id, file_id), record)
        return record.status, 0
    try:
        chunks = await _load_chunks(store, record)
    except _FileIndexingError as exc:
        record.status = "failed"
        record.last_error = FileErrorRecord(code=exc.code, message=str(exc))
        await _write(_file_key(store.id, file_id), record)
        return record.status, 0
    except (ApiError, BotoCoreError, ClientError, OSError) as exc:
        log_error_details(f"Vector store indexing failed: {exc!r}", level="error")
        record.status = "failed"
        record.last_error = FileErrorRecord(
            code="server_error", message="The file could not be indexed."
        )
        await _write(_file_key(store.id, file_id), record)
        return record.status, 0

    # Stored before the vectors exist, so a mid-write failure orphans none.
    record.chunk_count = len(chunks)
    record.usage_bytes = sum(len(chunk.encode()) for chunk in chunks)
    await _write(_file_key(store.id, file_id), record)
    try:
        await _write_vectors(store, record, chunks)
    except (ApiError, BotoCoreError, ClientError) as exc:
        log_error_details(f"Vector store indexing failed: {exc!r}", level="error")
        record.status = "failed"
        record.usage_bytes = 0
        record.last_error = FileErrorRecord(
            code="server_error", message="The file could not be indexed."
        )
        await _write(_file_key(store.id, file_id), record)
        return record.status, 0
    # Chunks beyond the new count would stay searchable with stale text.
    stale = record.previous_chunk_count
    record.previous_chunk_count = 0
    record.status = "completed"
    await _write(_file_key(store.id, file_id), record)
    if stale > len(chunks):
        await _delete_vector_keys(
            store.id,
            [vector_key(record.id, index) for index in range(len(chunks), stale)],
        )
    return record.status, record.usage_bytes


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


async def _load_chunks(store: StoreRecord, record: FileRecord) -> list[str]:
    """Read a file's text and split it into chunks.

    Args:
        store: The store the file is attached to.
        record: The file record.

    Returns:
        The chunks, in document order.

    Raises:
        _FileIndexingError: When the file holds no indexable text.
    """
    stream, content_type = await get_file_content(parse_file_id(record.id))
    if content_type.split(";", 1)[0].strip() in _BINARY_CONTENT_TYPES:
        raise _UnsupportedFileError(_UNSUPPORTED_MESSAGE)
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
        raise _UnsupportedFileError(_UNSUPPORTED_MESSAGE) from None
    del body
    if "\x00" in text:
        raise _UnsupportedFileError(_UNSUPPORTED_MESSAGE)
    chunks = chunk_text(
        text,
        record.max_chunk_size_tokens,
        record.chunk_overlap_tokens,
        _max_input_characters(store.embedding_model),
    )
    if not chunks:
        msg = "The file holds no text to index."
        raise _InvalidFileError(msg)
    return chunks


#: Message reported for a file whose bytes are not indexable text.
_UNSUPPORTED_MESSAGE: Final = (
    "This file type cannot be indexed. Provide the content as a text file."
)


def vector_key(file_id: str, chunk_index: int) -> str:
    """Return the index key of one chunk.

    Args:
        file_id: The file the chunk comes from.
        chunk_index: Position of the chunk within the file.

    Returns:
        The vector key.
    """
    return f"{file_id}#{chunk_index}"


async def _write_vectors(
    store: StoreRecord, record: FileRecord, chunks: Sequence[str]
) -> None:
    """Embed *chunks* and write them into the store's index.

    Args:
        store: The store the file is attached to.
        record: The file record.
        chunks: The chunks, in document order.
    """
    attributes = {attribute_key(k): v for k, v in record.attributes.items()}
    client = _vectors_client()
    bucket = _vector_bucket()
    name = index_name(store.id)
    for start in range(0, len(chunks), _PUT_VECTORS_BATCH):
        window = chunks[start : start + _PUT_VECTORS_BATCH]
        vectors = await _embed(store.embedding_model, window)
        with _vectors_guard("PutVectors"):
            await client.put_vectors(
                vectorBucketName=bucket,
                indexName=name,
                vectors=[
                    {
                        "key": vector_key(record.id, start + offset),
                        "data": {"float32": vector},
                        "metadata": {
                            _TEXT_KEY: chunk,
                            _FILENAME_KEY: record.filename,
                            _FILE_ID_KEY: record.id,
                            _CHUNK_INDEX_KEY: start + offset,
                            **attributes,
                        },
                    }
                    for offset, (chunk, vector) in enumerate(
                        zip(window, vectors, strict=True)
                    )
                ],
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

    def mutate_store(record: StoreRecord) -> None:
        """Apply the file's outcome to the store counters."""
        mutate(record.file_counts)
        record.usage_bytes += usage_bytes

    with suppress(ApiError, BotoCoreError, ClientError):
        await update_record(StoreRecord, _store_key(store_id), mutate_store)
    if batch_id:
        with suppress(ApiError, BotoCoreError, ClientError):
            await update_record(
                BatchRecord,
                _batch_key(store_id, batch_id),
                lambda record: mutate(record.file_counts),
            )


async def detach_file(store_id: str, file_id: str) -> None:
    """Remove a file and its vectors from a store.

    Args:
        store_id: A validated vector store identifier.
        file_id: The file to remove.

    Raises:
        ApiError: When the file is not attached to the store (404).
    """
    record = await read_file(store_id, file_id)
    await _delete_record(_file_key(store_id, file_id))

    def mutate(store: StoreRecord) -> None:
        """Remove the file from the store counters."""
        counts = store.file_counts
        setattr(counts, record.status, max(0, getattr(counts, record.status) - 1))
        store.usage_bytes = max(0, store.usage_bytes - record.usage_bytes)

    with suppress(ApiError, BotoCoreError, ClientError):
        await update_record(StoreRecord, _store_key(store_id), mutate)
    if record.chunk_count or record.previous_chunk_count:
        schedule_cleanup(_delete_vectors(store_id, record))


async def _delete_vectors(store_id: str, record: FileRecord) -> None:
    """Delete every vector of *record* from the store's index.

    A file whose re-indexing failed still owns the vectors of the version
    before it, so both counts are reclaimed.
    """
    count = max(record.chunk_count, record.previous_chunk_count)
    await _delete_vector_keys(
        store_id, [vector_key(record.id, index) for index in range(count)]
    )


async def _delete_vector_keys(store_id: str, keys: list[str]) -> None:
    """Delete *keys* from the store's index, ignoring what is already gone."""
    client = _vectors_client()
    for start in range(0, len(keys), _DELETE_VECTORS_BATCH):
        # The guard's own error is suppressed too: its warning is the report.
        with (
            suppress(ApiError, BotoCoreError, ClientError),
            _vectors_guard("DeleteVectors"),
        ):
            await client.delete_vectors(
                vectorBucketName=_vector_bucket(),
                indexName=index_name(store_id),
                keys=keys[start : start + _DELETE_VECTORS_BATCH],
            )


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
        ApiError: When the file is not attached to the store (404).
    """
    await read_file(store_id, file_id)
    record = await update_record(
        FileRecord,
        _file_key(store_id, file_id),
        lambda stored: setattr(stored, "attributes", attributes),
        resource="file",
    )
    if record.chunk_count and record.status == "completed":
        schedule_cleanup(_rewrite_attributes(store_id, record))
    return record


async def _rewrite_attributes(store_id: str, record: FileRecord) -> None:
    """Re-write a file's vectors so they carry its new attributes."""
    client = _vectors_client()
    bucket = _vector_bucket()
    name = index_name(store_id)
    attributes = {attribute_key(k): v for k, v in record.attributes.items()}
    keys = [vector_key(record.id, index) for index in range(record.chunk_count)]
    for start in range(0, len(keys), _GET_VECTORS_BATCH):
        with _vectors_guard("GetVectors"):
            existing = await client.get_vectors(
                vectorBucketName=bucket,
                indexName=name,
                keys=keys[start : start + _GET_VECTORS_BATCH],
                returnData=True,
                returnMetadata=True,
            )
        vectors = [
            {
                "key": vector["key"],
                "data": vector["data"],
                "metadata": {
                    **{
                        key: value
                        for key, value in vector.get("metadata", {}).items()
                        if not key.startswith(_ATTRIBUTE_PREFIX)
                    },
                    **attributes,
                },
            }
            for vector in existing.get("vectors", ())
            if "data" in vector
        ]
        if vectors:
            with _vectors_guard("PutVectors"):
                await client.put_vectors(
                    vectorBucketName=bucket, indexName=name, vectors=vectors
                )


async def read_file_chunks(store_id: str, record: FileRecord) -> list[str]:
    """Return a file's indexed chunks, in document order.

    Args:
        store_id: A validated vector store identifier.
        record: The file record.

    Returns:
        The chunk texts.
    """
    client = _vectors_client()
    keys = [vector_key(record.id, index) for index in range(record.chunk_count)]
    texts: dict[int, str] = {}
    for start in range(0, len(keys), _GET_VECTORS_BATCH):
        with _vectors_guard("GetVectors"):
            response = await client.get_vectors(
                vectorBucketName=_vector_bucket(),
                indexName=index_name(store_id),
                keys=keys[start : start + _GET_VECTORS_BATCH],
                returnData=False,
                returnMetadata=True,
            )
        for vector in response.get("vectors", ()):
            metadata = vector.get("metadata", {})
            texts[int(metadata.get(_CHUNK_INDEX_KEY, 0))] = str(
                metadata.get(_TEXT_KEY, "")
            )
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
    """
    if store.expired:
        return []
    index_filter = translate_filter(filters) if filters is not None else None
    vectors = await _embed(store.embedding_model, list(queries))
    client = _vectors_client()
    best: dict[str, tuple[float, JsonMapping]] = {}
    queries_arguments: list[dict[str, Any]] = []
    for vector in vectors:
        arguments: dict[str, Any] = {
            "vectorBucketName": _vector_bucket(),
            "indexName": index_name(store.id),
            "topK": max_num_results,
            "queryVector": {"float32": vector},
            "returnMetadata": True,
            "returnDistance": True,
        }
        if index_filter is not None:
            arguments["filter"] = index_filter
        queries_arguments.append(arguments)
    with _vectors_guard("QueryVectors"):
        responses = await _bounded(
            [client.query_vectors(**arguments) for arguments in queries_arguments]
        )
    for response in responses:
        for hit in response.get("vectors", ()):
            score = score_from_distance(hit.get("distance", 1.0))
            current = best.get(hit["key"])
            if current is None or score > current[0]:
                best[hit["key"]] = (score, hit.get("metadata", {}))
    results = [
        SearchResult(
            file_id=str(metadata.get(_FILE_ID_KEY, "")),
            filename=str(metadata.get(_FILENAME_KEY, "")),
            score=score,
            text=str(metadata.get(_TEXT_KEY, "")),
            attributes=_caller_attributes(metadata),
        )
        for score, metadata in best.values()
        if score_threshold is None or score >= score_threshold
    ]
    results.sort(key=lambda result: result.score, reverse=True)
    return results[:max_num_results]


def _caller_attributes(metadata: JsonMapping) -> Attributes:
    """Return the caller attributes carried by a vector's metadata."""
    attributes: Attributes = {}
    for key, value in metadata.items():
        if key.startswith(_ATTRIBUTE_PREFIX) and isinstance(
            value, bool | float | int | str
        ):
            attributes[key[len(_ATTRIBUTE_PREFIX) :]] = _attribute_value(value)
    return attributes


def _attribute_value(value: JsonValue) -> AttributeValue:
    """Return *value* as one of the types an attribute may hold."""
    return value if isinstance(value, bool | str) else float(value)  # type: ignore[arg-type]


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
        _batch_key(store_id, batch_id),
        lambda record: setattr(record, "cancel_requested", True),
        resource="file batch",
    )


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
    """
    records, _ = await list_store_files(
        store_id,
        after=after,
        before=before,
        limit=_LIST_SCAN_MAX,
        order=order,
        status=status,
    )
    batch_records = [record for record in records if record.batch_id == batch_id]
    return batch_records[:limit], len(batch_records) > limit
