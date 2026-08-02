"""Files API core storage — CRUD operations backed by S3.

Each uploaded file is stored as a single S3 object. File metadata is
encoded entirely in native S3 object attributes (ContentType,
ContentDisposition, user Metadata, LastModified, ContentLength),
so no external database is required.

File payload format: ``base32hex(uuid7_bytes + crc32_bytes)`` — 20 bytes encoded
as 32 base32hex characters (no padding).  The first 16 bytes are a UUIDv7 (millisecond-
precision timestamp in the high bits, so payloads remain lexicographically ordered by
creation time), and the last 4 bytes are the CRC32 of the S3 bucket name.  This
fingerprint lets any caller resolve the correct bucket without lookup tables or
hardcoded region mappings.

The base32hex alphabet (``0-9a-v``) is used rather than standard base32
(``a-z2-7``) because only the former sorts in the same order as the bytes it
encodes: standard base32 maps the six highest values to ``2-7``, which sort
below ``a-z``, breaking the creation-time ordering the listing relies on.
Standard-alphabet payloads remain accepted, told apart by the bucket
fingerprint they decode to.

S3 keys store the bare 32-char payload (no API-level prefix).  The ``file-``
or ``file_`` prefix is added at the API response boundary only.
"""

from asyncio import gather
from base64 import b32decode, b32hexdecode, b32hexencode
from binascii import Error as _BinasciiError
from binascii import crc32 as _crc32
from contextlib import suppress
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
    S3_TAGGING,
    multipart_copy_parts,
    require_s3_bucket_for_region,
    track_temporary_s3_objects,
)
from stdapi.config import SETTINGS
from stdapi.monitoring import log_error_details
from stdapi.server import AWS_APN_ID
from stdapi.types import FILE_ID_PATTERN
from stdapi.utils import now_utc_timestamp, parse_content_disposition_filename

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from typing import TypedDict

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_s3.client import S3Client
    from types_aiobotocore_s3.type_defs import (
        CopySourceTypeDef,
        HeadObjectOutputTypeDef,
    )

    from stdapi.input_file import InputFile

    #: S3 object metadata fields for stored files.
    _FileMetadata = TypedDict("_FileMetadata", {"expires-at": str, "purpose": str})

#: Characters forbidden in filenames per the Anthropic Files API specification.
_FILENAME_FORBIDDEN_RE = re_compile(r'[<>:"|?*\\/]|[\x00-\x1f]')

#: Maximum filename length per the Anthropic spec.
_FILENAME_MAX_LEN: int = 500

#: OpenAI purpose stored for files uploaded without a purpose (OpenAI's "any purpose" value).
DEFAULT_PURPOSE = "user_data"

#: S3's single-request ``CopyObject`` size cap; larger objects need ``UploadPartCopy``.
_COPY_OBJECT_MAX_BYTES = 5 * 1024**3

#: Part size used when correcting metadata on an object above :data:`_COPY_OBJECT_MAX_BYTES`.
_METADATA_FIX_PART_SIZE = 512 * 1024**2

#: CRC32 fingerprint → bucket name, for O(1) file-ID bucket resolution.
_BUCKET_CRC32: dict[int, str] = {_crc32(b.encode()): b for b in BUCKET_TO_REGION}

#: Compiled matcher for prefixed Files API identifiers.
_FILE_ID_RE = re_compile(FILE_ID_PATTERN).match


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
    log_error_details(
        "S3 bucket not configured (aws_s3_bucket): the Files API is disabled.",
        level="warning",
    )
    msg = (
        "The Files API is not available on the current server. "
        "Please contact the administrator to enable it."
    )
    raise ApiError(msg, status=503)


def file_id_s3_key(payload: str) -> str:
    """Return the S3 object key for the bare 32-char *payload*, e.g. ``{prefix}{payload}``."""
    return f"{SETTINGS.aws_s3_files_prefix}{payload}"


def parse_file_id(fid: str) -> str:
    """Validate a prefixed Files API identifier and return its bare 32-char payload.

    Args:
        fid: Files API identifier prefixed with ``file-`` or ``file_``.

    Returns:
        Bare 32-char base32 payload (no prefix).

    Raises:
        ApiError: When *fid* does not match :data:`stdapi.types.FILE_ID_PATTERN`.
    """
    if not _FILE_ID_RE(fid):
        msg = "Invalid file ID."
        raise ApiError(msg, status=400)
    return fid[5:]


def encode_id_payload(bucket: str) -> str:
    """Generate a bare 32-char payload embedding a CRC32 fingerprint of the bucket name.

    Payload: ``uuid7_bytes (16) + crc32_bytes (4)`` = 20 bytes → 32 lowercase
    base32hex characters without padding, which sort by creation time.

    Args:
        bucket: S3 bucket name the file will be stored in.

    Returns:
        Bare 32-char payload.
    """
    return (
        b32hexencode(uuid7().bytes + _crc32(bucket.encode()).to_bytes(4, "big"))
        .lower()
        .decode()
    )


def _decode_or_none(decode: Callable[[str], bytes], payload: str) -> bytes | None:
    """Decode *payload* with *decode*, returning ``None`` on a wrong alphabet.

    Args:
        decode: ``b32hexdecode`` or ``b32decode``.
        payload: Upper-cased bare payload.

    Returns:
        Decoded bytes, or ``None`` when the payload is not valid in that alphabet.
    """
    try:
        return decode(payload)
    except _BinasciiError, KeyError:
        return None


def decode_id_payload(payload: str) -> bytes:
    """Decode a bare 32-char payload to its 20 raw bytes, accepting both alphabets.

    A payload using ``w-z`` can only be a standard-alphabet one; otherwise the
    alphabets overlap, and the decoding whose trailing CRC32 names a configured
    bucket wins.

    Args:
        payload: Bare 32-char file or upload payload.

    Returns:
        The 20 decoded bytes: ``uuid7_bytes (16) + crc32_bytes (4)``.
    """
    upper = payload.upper()
    candidates = [
        decoded
        for decode in (b32hexdecode, b32decode)
        # A payload valid in one alphabet is usually invalid in the other;
        # b32decode reports that as KeyError rather than binascii.Error.
        if (decoded := _decode_or_none(decode, upper)) is not None
    ]
    for decoded in candidates:
        if int.from_bytes(decoded[16:], "big") in _BUCKET_CRC32:
            return decoded
    # No fingerprint matched, e.g. a decommissioned bucket: the caller falls
    # back to the primary bucket, so any successful decoding will do.
    return candidates[0] if candidates else b""


def resolve_file_bucket(payload: str) -> str:
    """Resolve the S3 bucket for a bare file payload by matching its embedded CRC32 fingerprint.

    Falls back to the primary bucket when no bucket matches the fingerprint
    (e.g. a decommissioned bucket).  The caller must supply a well-formed
    32-char base32 payload; format validation is enforced at the API boundary.

    Args:
        payload: Bare 32-char base32 file payload.

    Returns:
        S3 bucket name.

    Raises:
        ApiError: If no bucket is configured at all (503).
    """
    return (
        _BUCKET_CRC32.get(int.from_bytes(decode_id_payload(payload)[16:], "big"))
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


async def _force_s3_metadata(
    s3: S3Client,
    bucket: str,
    key: str,
    size: int,
    content_type: str,
    content_disposition: str,
    metadata: _FileMetadata,
) -> None:
    """Force *content_disposition*/*metadata* onto an existing S3 object via self-copy.

    A server-side S3-to-S3 copy (used by :meth:`InputFile.to_s3` for
    ``s3://``/``file-id:`` sources) always inherits the source object's own
    ``Content-Disposition`` and metadata and ignores whatever the caller
    requested, so :func:`upload_file` calls this to reconcile the two when
    they differ.

    Args:
        s3: Authenticated S3 client.
        bucket: Bucket holding the object.
        key: Key of the object to fix in place.
        size: Current object size, to pick single-shot vs. multipart copy.
        content_type: Content type to preserve on the object.
        content_disposition: Desired ``Content-Disposition`` header value.
        metadata: Desired user metadata.

    Raises:
        BotoCoreError: If the AWS SDK fails.
        ClientError: If S3 returns an error.
    """
    copy_source: CopySourceTypeDef = {"Bucket": bucket, "Key": key}
    if size <= _COPY_OBJECT_MAX_BYTES:
        await s3.copy_object(
            Bucket=bucket,
            Key=key,
            CopySource=copy_source,
            ContentType=content_type,
            ContentDisposition=content_disposition,
            Metadata=metadata,  # type: ignore[arg-type]
            MetadataDirective="REPLACE",
        )
        return
    upload_id = (
        await s3.create_multipart_upload(
            Bucket=bucket,
            Key=key,
            ContentType=content_type,
            ContentDisposition=content_disposition,
            Metadata=metadata,  # type: ignore[arg-type]
            Tagging=S3_TAGGING,
        )
    )["UploadId"]
    try:
        parts = await multipart_copy_parts(
            s3,
            bucket=bucket,
            key=key,
            upload_id=upload_id,
            copy_source=copy_source,
            size=size,
            part_size=_METADATA_FIX_PART_SIZE,
        )
        await s3.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},  # type: ignore[typeddict-item]
        )
    except Exception:
        with suppress(ClientError):
            await s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        raise


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
            ``None``/falsy (e.g. from the Anthropic Files API) is stored as
            :data:`DEFAULT_PURPOSE`, so the stored value always matches what
            is displayed and matched by the ``purpose`` list filter.
        expires_after: Seconds from now until expiry, or ``None``.
        region: AWS region for bucket selection; ``None`` uses the default bucket.

    Raises:
        ApiError: ``aws_s3_bucket`` not configured (503) or invalid filename.
    """
    bucket = require_s3_bucket_for_region(region) if region else _require_bucket()
    payload = encode_id_payload(bucket)
    s3_key = file_id_s3_key(payload)
    expires_at = (
        now_utc_timestamp() + expires_after if expires_after is not None else None
    )

    metadata: _FileMetadata = {
        "expires-at": str(expires_at) if expires_at is not None else "",
        "purpose": purpose or DEFAULT_PURPOSE,
    }
    content_disposition = f'attachment; filename="{_validate_filename(await file.get_filename() or "upload")}"'
    await file.to_s3(
        BUCKET_TO_REGION[bucket],
        bucket=bucket,
        key=s3_key,
        temporary=False,
        content_disposition=content_disposition,
        metadata=metadata,  # type: ignore[arg-type]
    )
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(bucket))
    head = await s3.head_object(Bucket=bucket, Key=s3_key)
    if head.get("ContentDisposition", "") != content_disposition or head.get(
        "Metadata", {}
    ) != dict(metadata):
        # Only S3-to-S3 copy sources reach this: they ignore the requested
        # content-disposition/metadata, so force them in place.
        await _force_s3_metadata(
            s3,
            bucket,
            s3_key,
            head["ContentLength"],
            head.get("ContentType", "application/octet-stream"),
            content_disposition,
            metadata,
        )
        head = await s3.head_object(Bucket=bucket, Key=s3_key)
    if expires_at is not None:
        await s3.put_object_tagging(
            Bucket=bucket,
            Key=s3_key,
            Tagging={
                "TagSet": [
                    {"Key": "stdapi-ai.expires", "Value": "true"},
                    {"Key": "aws-apn-id", "Value": AWS_APN_ID},
                ]
            },
        )
    return _record_from_head(payload, head)


async def _get_file_impl(payload: str) -> tuple[FileRecord, str, S3Client]:
    """Resolve bucket, ``HeadObject``, and enforce expiry for *payload*.

    Returns:
        ``(record, bucket, s3_client)`` for further operations.

    Raises:
        _FileNotFoundError: Not found or expired.
    """
    bucket = resolve_file_bucket(payload)
    key = file_id_s3_key(payload)
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
    await s3.delete_object(Bucket=bucket, Key=file_id_s3_key(payload))


async def get_file_content(payload: str) -> tuple[AsyncIterator[bytes], str]:
    """Return a streaming body iterator and content type for *payload*.

    Raises:
        _FileNotFoundError: Not found or expired.
    """
    record, bucket, s3 = await _get_file_impl(payload)
    return (
        (await s3.get_object(Bucket=bucket, Key=file_id_s3_key(payload)))[
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
        start_after = file_id_s3_key(after) if after else None
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
    start_after = file_id_s3_key(after) if order == "asc" and after else None
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
        all_keys = [k for k in all_keys if k < file_id_s3_key(after)]
    if before:
        all_keys = [k for k in all_keys if k < file_id_s3_key(before)]

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
