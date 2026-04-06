"""Files API core storage — CRUD operations backed by S3.

Each uploaded file is stored as a single S3 object. File metadata is
encoded entirely in native S3 object attributes (ContentType,
ContentDisposition, user Metadata, LastModified, ContentLength),
so no external database is required.

File payload format: ``base32(uuid7_bytes + crc32_bytes)`` — 20 bytes encoded
as 32 base32 characters (no padding).  The first 16 bytes are a UUIDv7 (millisecond-
precision timestamp in the high bits, so payloads remain lexicographically ordered by
creation time), and the last 4 bytes are the CRC32 of the S3 bucket name.  This
fingerprint lets any caller resolve the correct bucket without lookup tables or
hardcoded region mappings.

S3 keys store the bare 32-char payload (no API-level prefix).  The ``file-``
or ``file_`` prefix is added at the API response boundary only.
"""

from asyncio import gather
from base64 import b32decode, b32encode
from binascii import crc32 as _crc32
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import chain
from re import compile as re_compile
from typing import TYPE_CHECKING
from uuid import uuid7

from botocore.exceptions import ClientError

from stdapi.api_errors import ApiError
from stdapi.api_errors import FileNotExistError as _FileNotFoundError
from stdapi.aws import get_client
from stdapi.aws_s3 import (
    BUCKET_TO_REGION,
    require_s3_bucket_for_region,
    track_temporary_s3_objects,
)
from stdapi.config import SETTINGS
from stdapi.utils import now_utc_timestamp, parse_content_disposition_filename

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import NotRequired, TypedDict

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_s3.client import S3Client
    from types_aiobotocore_s3.type_defs import HeadObjectOutputTypeDef

    from stdapi.input_file import InputFile

    #: S3 object metadata fields for stored files.
    _FileMetadata = TypedDict(
        "_FileMetadata", {"expires-at": str, "purpose": NotRequired[str]}
    )

#: Characters forbidden in filenames per the Anthropic Files API specification.
_FILENAME_FORBIDDEN_RE = re_compile(r'[<>:"|?*\\/]|[\x00-\x1f]')

#: Maximum filename length per the Anthropic spec.
_FILENAME_MAX_LEN: int = 500

#: CRC32 fingerprint → bucket name, for O(1) file-ID bucket resolution.
_BUCKET_CRC32: dict[int, str] = {_crc32(b.encode()): b for b in BUCKET_TO_REGION}


@dataclass(slots=True)
class FileRecord:
    """Internal canonical representation of a stored file.

    Attributes:
        file_id: Bare 32-char base32 payload (no ``file-``/``file_`` prefix).
        filename: Original filename extracted from ``Content-Disposition``.
        content_type: MIME type string, e.g. ``application/pdf``.
        purpose: OpenAI purpose string, or ``""`` if not set.
        size: File size in bytes from S3 ``ContentLength``.
        created_at: UTC creation time from S3 ``LastModified``.
        expires_at: Unix timestamp (seconds) of expiry, or ``None`` if no expiry.
    """

    file_id: str
    filename: str
    content_type: str
    purpose: str
    size: int
    created_at: datetime
    expires_at: int | None


def _validate_filename(filename: str) -> str:
    """Validate filename characters and length.

    Raises:
        ApiError: Forbidden characters or exceeds the 500-character limit.

    Returns:
        Validated filename.
    """
    if len(filename) > _FILENAME_MAX_LEN:
        msg = f"Filename exceeds maximum length of {_FILENAME_MAX_LEN} characters."
        raise ApiError(msg)
    if _FILENAME_FORBIDDEN_RE.search(filename):
        msg = 'Filename contains forbidden characters (< > : " | ? * \\ / or control chars).'
        raise ApiError(msg)
    return filename


def _require_bucket() -> str:
    """Return the configured S3 bucket.

    Raises:
        ApiError: ``aws_s3_bucket`` not configured (503).
    """
    if bucket := SETTINGS.aws_s3_bucket:
        return bucket
    msg = "Files API is not available: aws_s3_bucket is not configured."
    raise ApiError(msg, status=503)


def _s3_key(payload: str) -> str:
    """Return the S3 object key for the bare 32-char *payload*, e.g. ``{prefix}{payload}``."""
    return f"{SETTINGS.aws_s3_files_prefix}{payload}"


def _file_id_from_bucket(bucket: str) -> str:
    """Generate a bare 32-char base32 payload embedding a CRC32 fingerprint of the bucket name.

    Payload: ``uuid7_bytes (16) + crc32_bytes (4)`` = 20 bytes → 32 lowercase
    base32 characters without padding.

    Args:
        bucket: S3 bucket name the file will be stored in.
    """
    return (
        b32encode(uuid7().bytes + _crc32(bucket.encode()).to_bytes(4, "big"))
        .lower()
        .decode()
    )


def resolve_file_bucket(payload: str) -> str:
    """Resolve the S3 bucket for a bare file payload by matching its embedded CRC32 fingerprint.

    Decodes the last 4 bytes of the payload and looks up the matching bucket in
    :data:`_BUCKET_CRC32`.  Falls back to the primary bucket when no bucket
    matches the fingerprint (e.g. a decommissioned bucket).

    The caller must supply a well-formed 32-char base32 payload; format
    validation is enforced at the API boundary.

    Args:
        payload: Bare 32-char base32 file payload.

    Returns:
        S3 bucket name.

    Raises:
        ApiError: If no bucket is configured at all (503).
    """
    return (
        _BUCKET_CRC32.get(int.from_bytes(b32decode(payload.upper())[16:], "big"))
        or _require_bucket()
    )


def _record_from_head(payload: str, head: HeadObjectOutputTypeDef) -> FileRecord:
    """Build a ``FileRecord`` from a ``HeadObject`` response.

    Args:
        payload: Bare 32-char base32 file payload.
        head: ``HeadObject`` response dict.
    """
    meta = head.get("Metadata", {})
    return FileRecord(
        file_id=payload,
        filename=parse_content_disposition_filename(head.get("ContentDisposition", ""))
        or payload,
        content_type=head.get("ContentType", "application/octet-stream"),
        purpose=meta.get("purpose", ""),
        size=head["ContentLength"],
        created_at=head["LastModified"].replace(tzinfo=UTC),
        expires_at=int(v) if (v := meta.get("expires-at", "")) else None,
    )


async def upload_file(
    file: InputFile,
    purpose: str | None = None,
    expires_after: int | None = None,
    region: RegionName | None = None,
) -> FileRecord:
    """Stream *file* to S3 and return its metadata record.

    Metadata is encoded in native S3 object attributes — no database needed.
    When *expires_after* is set, an S3 object tag ``expires=true`` is added
    for Lifecycle rule cleanup; code-level expiry is enforced in
    :func:`get_file`.

    Args:
        file: Input file with content and optional filename / content type.
        purpose: OpenAI purpose string (stored as metadata; not validated).
        expires_after: Seconds from now until expiry, or ``None``.
        region: AWS region for bucket selection; ``None`` uses the default bucket.

    Raises:
        ApiError: ``aws_s3_bucket`` not configured (503) or invalid filename.
    """
    bucket = require_s3_bucket_for_region(region) if region else _require_bucket()
    payload = _file_id_from_bucket(bucket)
    s3_key = _s3_key(payload)
    expires_at = (
        now_utc_timestamp() + expires_after if expires_after is not None else None
    )

    metadata: _FileMetadata = {
        "expires-at": str(expires_at) if expires_at is not None else ""
    }
    if purpose:
        metadata["purpose"] = purpose
    await file.to_s3(
        BUCKET_TO_REGION[bucket],
        bucket=bucket,
        key=s3_key,
        temporary=False,
        content_disposition=f'attachment; filename="{_validate_filename(await file.get_filename() or "upload")}"',
        metadata=metadata,  # type: ignore[arg-type]
    )
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(bucket))
    if expires_at is not None:
        await s3.put_object_tagging(
            Bucket=bucket,
            Key=s3_key,
            Tagging={"TagSet": [{"Key": "expires", "Value": "true"}]},
        )
    return _record_from_head(payload, await s3.head_object(Bucket=bucket, Key=s3_key))


async def _get_file_impl(payload: str) -> tuple[FileRecord, str, S3Client]:
    """Resolve bucket, ``HeadObject``, and enforce expiry for *payload*.

    Returns:
        ``(record, bucket, s3_client)`` for further operations.

    Raises:
        _FileNotFoundError: Not found or expired.
    """
    bucket = resolve_file_bucket(payload)
    key = _s3_key(payload)
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(bucket))
    try:
        record = _record_from_head(
            payload, await s3.head_object(Bucket=bucket, Key=key)
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            msg = f"File '{payload}' not found."
            raise _FileNotFoundError(msg) from exc
        raise  # pragma: no cover
    if record.expires_at is not None and now_utc_timestamp() >= record.expires_at:
        track_temporary_s3_objects(bucket, key)
        msg = f"File '{payload}' has expired."
        raise _FileNotFoundError(msg)
    return record, bucket, s3


async def get_file(payload: str) -> FileRecord:
    """Retrieve file metadata from S3, enforcing expiry.

    Expired files schedule background deletion and raise ``_FileNotFoundError``.

    Raises:
        _FileNotFoundError: Not found or expired.
    """
    return (await _get_file_impl(payload))[0]


async def delete_file(payload: str) -> None:
    """Delete *payload* from S3; expired files are treated as non-existent.

    Raises:
        _FileNotFoundError: Not found or expired.
    """
    _, bucket, s3 = await _get_file_impl(payload)
    await s3.delete_object(Bucket=bucket, Key=_s3_key(payload))


async def get_file_content(payload: str) -> tuple[AsyncIterator[bytes], str]:
    """Return a streaming body iterator and content type for *payload*.

    Raises:
        _FileNotFoundError: Not found or expired.
    """
    record, bucket, s3 = await _get_file_impl(payload)
    return (
        (await s3.get_object(Bucket=bucket, Key=_s3_key(payload)))[
            "Body"
        ].iter_chunks(),
        record.content_type,
    )


async def _head_record(s3: S3Client, bucket: str, key: str) -> FileRecord | None:
    """``HeadObject`` *key* and return a ``FileRecord``, or ``None`` if absent."""
    try:
        head = await s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise  # pragma: no cover
    return _record_from_head(key.removeprefix(SETTINGS.aws_s3_files_prefix), head)


async def _scan_bucket_page(
    bucket: str, prefix: str, start_after: str | None, max_keys: int
) -> list[str]:
    """Return up to *max_keys* S3 keys from *bucket* starting after *start_after*."""
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(bucket))
    return [
        obj["Key"]
        for obj in (
            await (
                s3.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix,
                    MaxKeys=max_keys,
                    StartAfter=start_after,
                )
                if start_after
                else s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)
            )
        ).get("Contents", [])
    ]


async def _scan_all_keys(
    s3: S3Client, bucket: str, prefix: str, start_after: str | None = None
) -> list[str]:
    """Scan all S3 keys under *prefix* in *bucket*, returning them in ascending order."""
    resp = await (
        s3.list_objects_v2(
            Bucket=bucket, Prefix=prefix, MaxKeys=1000, StartAfter=start_after
        )
        if start_after
        else s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
    )
    all_keys: list[str] = []
    while True:
        all_keys.extend(obj["Key"] for obj in resp.get("Contents", ()))
        if not resp.get("IsTruncated"):
            break
        resp = await s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=1000,
            ContinuationToken=resp["NextContinuationToken"],
        )
    return all_keys


async def _head_key(key: str) -> FileRecord | None:
    """Resolve bucket via embedded CRC32 and fetch the ``FileRecord`` for *key*, or ``None``."""
    payload = key.removeprefix(SETTINGS.aws_s3_files_prefix)
    if len(payload) != 32:  # skip non-file objects (markers, stray keys)
        return None
    bucket = resolve_file_bucket(payload)
    return await _head_record(
        get_client("s3", BUCKET_TO_REGION.get(bucket)), bucket, key
    )


async def _records_for_keys(keys: list[str]) -> list[FileRecord]:
    """Fan-out ``HeadObject`` calls for *keys*, returning records for existing objects."""
    return [r for r in await gather(*(_head_key(k) for k in keys)) if r is not None]


async def list_files(
    after: str | None, before: str | None, limit: int, order: str, purpose: str | None
) -> tuple[list[FileRecord], bool]:
    """List files from all configured S3 buckets with cursor-based pagination.

    S3 ``ListObjectsV2`` returns keys in ascending lexicographic order; because
    file keys use UUIDv7 (timestamp-ordered), ascending key order equals
    ascending creation-time order.  Keys from multiple buckets are merged and
    sorted globally before paging.

    Args:
        after: Return files created strictly after this bare payload (exclusive).
        before: Return files created strictly before this bare payload (Anthropic ``before_id``).
        limit: Maximum records to return.
        order: ``"asc"`` or ``"desc"`` (OpenAI default is ``"desc"``).
        purpose: Filter by purpose; triggers a ``HeadObject`` fan-out to read metadata.

    Returns:
        ``(records, has_more)`` tuple.

    Raises:
        ApiError: ``aws_s3_bucket`` not configured (503).
    """
    _require_bucket()
    prefix = SETTINGS.aws_s3_files_prefix
    prefix_len = len(prefix)
    buckets = list(BUCKET_TO_REGION.keys())

    # Efficient path: ascending without purpose filter or before cursor
    if order == "asc" and before is None and purpose is None:
        start_after = _s3_key(after) if after else None
        per_bucket = await gather(
            *[_scan_bucket_page(b, prefix, start_after, limit + 1) for b in buckets]
        )
        keys = [
            k
            for k in sorted(chain.from_iterable(per_bucket))
            if len(k) - prefix_len == 32
        ][: limit + 1]
        return await _records_for_keys(keys[:limit]), len(keys) > limit

    # Slow path: full scan needed for desc order, before cursor, or purpose filter
    start_after = _s3_key(after) if order == "asc" and after else None
    all_keys = [
        k
        for k in sorted(
            chain.from_iterable(
                await gather(
                    *[
                        _scan_all_keys(
                            get_client("s3", BUCKET_TO_REGION[b]),
                            b,
                            prefix,
                            start_after,
                        )
                        for b in buckets
                    ]
                )
            )
        )
        if len(k) - prefix_len == 32
    ]

    if order == "desc" and after:
        all_keys = [k for k in all_keys if k < _s3_key(after)]
    if before:
        all_keys = [k for k in all_keys if k < _s3_key(before)]

    if purpose is not None:
        filtered = [
            r for r in (await _records_for_keys(all_keys)) if r.purpose == purpose
        ]
        return (
            filtered[-limit:][::-1]
            if order == "desc"
            else filtered[-limit:]
            if before
            else filtered[:limit]
        ), len(filtered) > limit

    if order == "desc":
        return await _records_for_keys(all_keys[-limit:][::-1]), len(all_keys) > limit

    # asc + before_id: ascending slice ending just before the cursor
    return await _records_for_keys(all_keys[-limit:]), len(all_keys) > limit
