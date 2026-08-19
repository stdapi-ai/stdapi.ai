"""The bookkeeping a vector store answers with, in the application bucket.

A vector index holds no per-store counters, no per-file status and no batch
progress, so the API's own state lives in JSON objects — one per store, per
attached file and per batch — mutated with conditional writes.

Consistency rule: a file record reaches its terminal state **before** the store
counters that summarise it. The counters may lag the file listing, never lead
it — a store still reporting ``in_progress`` for a file already ``completed``
converges, while counters claiming a completion the listing cannot show does not.
"""

from typing import TYPE_CHECKING, Final

from botocore.exceptions import ClientError
from pydantic import BaseModel

from stdapi.api_errors import ApiError, FeatureUnavailableError
from stdapi.aws import get_client
from stdapi.aws_s3 import BUCKET_TO_REGION, S3_TAGGING
from stdapi.config import SETTINGS
from stdapi.vector_stores._concurrency import gather_bounded
from stdapi.vector_stores._paging import page_identifiers, page_records
from stdapi.vector_stores.backend import FEATURE, raise_not_found
from stdapi.vector_stores.models import BatchRecord, FileRecord, StoreRecord
from stdapi.vector_stores.registry import default_backend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from types_aiobotocore_s3.client import S3Client

#: Record reads or writes issued concurrently.
RECORD_WAVE: Final[int] = 32

#: Records one listing call scans before paging in memory.
_LIST_SCAN_MAX: Final[int] = 1000

#: Attempts a conditional record update makes before giving up.
_CAS_ATTEMPTS: Final[int] = 8

#: S3 error codes meaning the object is absent.
_MISSING_CODES: Final[frozenset[str]] = frozenset({"404", "NoSuchKey"})

#: S3 error codes a conditional write returns when another writer won.
_CONFLICT_CODES: Final[frozenset[str]] = frozenset(
    {"PreconditionFailed", "ConditionalRequestConflict"}
)


def records_bucket() -> str:
    """Return the application bucket holding the vector store bookkeeping.

    The bookkeeping is only served when a backend is: a deployment missing
    either half cannot answer the API at all.

    Returns:
        The bucket name.

    Raises:
        FeatureUnavailableError: When the deployment has no vector storage
            configured (503).
    """
    if not SETTINGS.aws_s3_bucket:
        raise FeatureUnavailableError(
            FEATURE,
            "Vector storage not configured (aws_s3_bucket): "
            "the Vector Stores API is disabled.",
        )
    default_backend().check_configured()
    return SETTINGS.aws_s3_bucket


def records_client() -> S3Client:
    """Return the S3 client for the bucket holding the bookkeeping records."""
    client: S3Client = get_client("s3", BUCKET_TO_REGION.get(records_bucket()))
    return client


def store_prefix(store_id: str) -> str:
    """Return the key prefix of every record of *store_id*."""
    return f"{SETTINGS.aws_s3_vector_stores_prefix}{store_id}/"


def store_key(store_id: str) -> str:
    """Return the key of the store record of *store_id*."""
    return f"{store_prefix(store_id)}store.json"


def file_key(store_id: str, file_id: str) -> str:
    """Return the key of the file record of *file_id* in *store_id*."""
    return f"{store_prefix(store_id)}files/{file_id}.json"


def batch_key(store_id: str, batch_id: str) -> str:
    """Return the key of the batch record of *batch_id* in *store_id*."""
    return f"{store_prefix(store_id)}batches/{batch_id}.json"


async def read_record[RecordT: BaseModel](
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
        response = await records_client().get_object(Bucket=records_bucket(), Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in _MISSING_CODES:
            return None
        raise
    return model.model_validate_json(await response["Body"].read()), response["ETag"]


async def write_record(key: str, record: BaseModel, *, etag: str | None = None) -> None:
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
    await records_client().put_object(
        Bucket=records_bucket(),
        Key=key,
        Body=record.model_dump_json().encode(),
        ContentType="application/json",
        Tagging=S3_TAGGING,
        **conditions,  # type: ignore[arg-type]
    )


async def write_if_unchanged(key: str, record: BaseModel, etag: str) -> bool:
    """Write a record only while the stored object still carries *etag*.

    Args:
        key: Object key of the record.
        record: The record to serialise.
        etag: Entity tag the stored object must still carry.

    Returns:
        Whether the record was written; ``False`` when another writer moved the
        record on, or deleted it, first.
    """
    try:
        await write_record(key, record, etag=etag)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in _CONFLICT_CODES | _MISSING_CODES:
            return False
        raise
    return True


async def delete_record(key: str) -> None:
    """Delete one bookkeeping record."""
    await records_client().delete_object(Bucket=records_bucket(), Key=key)


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
        current = await read_record(model, key)
        if current is None:
            # Deleted between the read and this write; the key stays out of the message.
            msg = f"The {resource} was deleted while this request ran."
            raise ApiError(msg, status=404)
        record, etag = current
        mutate(record)
        try:
            await write_record(key, record, etag=etag)
        except ClientError as exc:
            if exc.response["Error"]["Code"] in _CONFLICT_CODES:
                continue
            raise
        return record
    msg = f"The {resource} is being updated by another request. Retry the request."
    raise ApiError(msg, status=409)


async def gather_records[RecordT: BaseModel](
    model: type[RecordT], keys: Sequence[str]
) -> list[tuple[RecordT, str] | None]:
    """Read several records concurrently, in the order of *keys*."""
    return await gather_bounded([read_record(model, key) for key in keys], RECORD_WAVE)


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
    current = await read_record(FileRecord, file_key(store_id, file_id))
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
    current = await read_record(BatchRecord, batch_key(store_id, batch_id))
    if current is None:
        raise_not_found("file batch", batch_id)
    return current[0]


async def list_ids(
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
    response = await records_client().list_objects_v2(**arguments)  # type: ignore[arg-type]
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


async def all_record_keys(prefix: str) -> list[str]:
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
        response = await records_client().list_objects_v2(**arguments)  # type: ignore[arg-type]
        keys.extend(entry["Key"] for entry in response.get("Contents", ()))
        token = response.get("NextContinuationToken")
        if not response.get("IsTruncated") or not token:
            return keys
        arguments["ContinuationToken"] = token


async def list_stores(
    *, after: str, before: str, limit: int, order: str
) -> tuple[list[StoreRecord], bool]:
    """List vector stores, newest first by default.

    A store identifier is minted from the instant the record reports as its
    creation, so ascending key order is ascending creation order and the page is
    cut on the identifiers rather than on a thousand records read to sort them.

    Args:
        after: Return the stores following this identifier.
        before: Return the page ending immediately before this identifier.
        limit: Maximum records to return.
        order: ``"asc"`` or ``"desc"``.

    Returns:
        ``(records, has_more)``.
    """
    prefix = SETTINGS.aws_s3_vector_stores_prefix
    if order == "asc" and not before:
        ids, has_more = await list_ids(prefix, "/", after=after, limit=limit)
    else:
        ids, _ = await list_ids(prefix, "/", after="", limit=_LIST_SCAN_MAX)
        ids, has_more = page_identifiers(
            ids, after=after, before=before, limit=limit, order=order
        )
    records = [
        record[0]
        for record in (await gather_records(StoreRecord, [store_key(i) for i in ids]))
        if record is not None
    ]
    return records, has_more


async def store_file_records(store_id: str, status: str) -> list[FileRecord]:
    """Return every file attached to *store_id*, in no particular order.

    Args:
        store_id: A validated vector store identifier.
        status: Keep only files with this status, or ``""`` for all.

    Returns:
        The file records.
    """
    prefix = f"{store_prefix(store_id)}files/"
    ids, _ = await list_ids(prefix, "", after="", limit=_LIST_SCAN_MAX)
    records = [
        record[0]
        for record in (
            await gather_records(FileRecord, [file_key(store_id, i) for i in ids])
        )
        if record is not None
    ]
    return (
        [record for record in records if record.status == status] if status else records
    )


async def list_store_files(
    store_id: str, *, after: str, before: str, limit: int, order: str, status: str
) -> tuple[list[FileRecord], bool]:
    """List the files attached to *store_id*, most recently attached first.

    A file keeps the identifier it was uploaded under, which names the moment of
    the *upload* — a file uploaded last week and attached today would sort as an
    old one — and re-attaching it rewrites ``created_at`` where the identifier
    cannot move.  The listing therefore orders on the attachment time it
    reports, and the cursors name positions in that order.

    Args:
        store_id: A validated vector store identifier.
        after: Return the files following this identifier.
        before: Return the page ending immediately before this identifier.
        limit: Maximum records to return.
        order: ``"asc"`` or ``"desc"``.
        status: Keep only files with this status, or ``""`` for all.

    Returns:
        ``(records, has_more)``.
    """
    return page_records(
        await store_file_records(store_id, status),
        after=after,
        before=before,
        limit=limit,
        order=order,
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
    """List the files of one batch, most recently attached first.

    Ordered on the attachment time it reports, exactly as the store's own file
    listing is.

    Args:
        store_id: A validated vector store identifier.
        batch_id: A validated file batch identifier.
        after: Return the files following this identifier.
        before: Return the page ending immediately before this identifier.
        limit: Maximum records to return.
        order: ``"asc"`` or ``"desc"``.
        status: Keep only files with this status, or ``""`` for all.

    Returns:
        ``(records, has_more)``.
    """
    records = await store_file_records(store_id, status)
    return page_records(
        [record for record in records if record.batch_id == batch_id],
        after=after,
        before=before,
        limit=limit,
        order=order,
    )
