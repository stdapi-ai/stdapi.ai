"""Files API core storage — CRUD operations backed by S3.

Each uploaded file is stored as a single S3 object. File metadata is
encoded entirely in native S3 object attributes (ContentType,
ContentDisposition, user Metadata, ContentLength) plus the identifier
itself, so no external database is required.

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
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import chain
from mimetypes import guess_extension
from re import compile as re_compile
from typing import TYPE_CHECKING
from uuid import uuid7

from botocore.exceptions import ClientError

from stdapi.api_errors import ApiError, FeatureUnavailableError
from stdapi.api_errors import FileNotExistError as _FileNotFoundError
from stdapi.aws import get_client
from stdapi.aws_s3 import (
    BUCKET_TO_REGION,
    EXPIRING_S3_TAG_SET,
    put_s3_object,
    require_s3_bucket_for_region,
    track_temporary_s3_objects,
)
from stdapi.config import SETTINGS
from stdapi.types import FILE_ID_PATTERN
from stdapi.utils import now_utc_timestamp, parse_content_disposition_filename

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from typing import TypedDict

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_s3.client import S3Client
    from types_aiobotocore_s3.type_defs import HeadObjectOutputTypeDef

    from stdapi.input_file import InputFile

    #: S3 object metadata fields for stored files.
    _FileMetadata = TypedDict("_FileMetadata", {"expires-at": str, "purpose": str})

#: Characters a quoted ``Content-Disposition`` filename cannot carry: a quote ends it, controls are not header text.
_FILENAME_FORBIDDEN_RE = re_compile(r'["\x00-\x1f\x7f]')

#: Maximum filename length the Files API reports.
_FILENAME_MAX_LEN: int = 500

#: Filename given to an upload that carries none, before its media type's extension.
_UNNAMED_FILENAME = "unnamed"

#: OpenAI purpose stored for files uploaded without a purpose (OpenAI's "any purpose" value).
DEFAULT_PURPOSE = "user_data"

#: CRC32 fingerprint → bucket name, for O(1) file-ID bucket resolution.
_BUCKET_CRC32: dict[int, str] = {_crc32(b.encode()): b for b in BUCKET_TO_REGION}

#: Compiled matcher for prefixed Files API identifiers.
_FILE_ID_RE = re_compile(FILE_ID_PATTERN).match

#: The feature name a caller reads when no bucket backs the Files API.
_FEATURE: str = "The Files API"


@dataclass(slots=True)
class FileRecord:
    """Internal canonical representation of a stored file.

    Attributes:
        file_id: Bare 32-char base32 payload (no ``file-``/``file_`` prefix).
        filename: Original filename extracted from ``Content-Disposition``.
        content_type: MIME type string, e.g. ``application/pdf``.
        purpose: OpenAI purpose string, or ``""`` if not set.
        size: File size in bytes from S3 ``ContentLength``.
        created_at: UTC creation time, read from the UUIDv7 in ``file_id``.
        expires_at: Unix timestamp (seconds) of expiry, or ``None`` if no expiry.
    """

    file_id: str
    filename: str
    content_type: str
    purpose: str
    size: int
    created_at: datetime
    expires_at: int | None


def _sanitize_filename(filename: str, mime_type: str) -> str:
    """Return the name a file is stored and reported under.

    Only the final component of *filename* is kept, so a client sending a path
    names the file it ends with instead of being refused.  A name left empty by
    that becomes ``unnamed`` plus the extension *mime_type* implies, when it
    implies one.

    Args:
        filename: Filename as the client sent it, possibly a path or empty.
        mime_type: Media type stored with the file, read for the fallback extension.

    Returns:
        The filename to report and to write to ``Content-Disposition``.

    Raises:
        ApiError: The kept name exceeds 500 characters, or carries a character
            a ``Content-Disposition`` header cannot hold.
    """
    name = filename.rpartition("/")[2].rpartition("\\")[2]
    if not name:
        name = _UNNAMED_FILENAME + (guess_extension(mime_type) or "")
    if len(name) > _FILENAME_MAX_LEN:
        msg = f"Filename exceeds maximum length of {_FILENAME_MAX_LEN} characters."
        raise ApiError(msg)
    if _FILENAME_FORBIDDEN_RE.search(name):
        msg = 'Filename cannot be stored: remove any quote (") or control character.'
        raise ApiError(msg)
    return name


def _require_bucket() -> str:
    """Return the configured S3 bucket.

    Raises:
        FeatureUnavailableError: ``aws_s3_bucket`` not configured (503).
    """
    if bucket := SETTINGS.aws_s3_bucket:
        return bucket
    raise FeatureUnavailableError(
        _FEATURE, "S3 bucket not configured (aws_s3_bucket): the Files API is disabled."
    )


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


def _payload_created_ms(payload: str) -> int:
    """Return the creation time encoded in *payload*'s UUIDv7, in epoch milliseconds.

    ``0`` when *payload* carries no decodable UUIDv7, which only a stray object
    stored under the files prefix can produce.

    Args:
        payload: Bare 32-char base32 file or upload payload.
    """
    return int.from_bytes(decode_id_payload(payload)[:6], "big")


def payload_created_at(payload: str) -> int:
    """Return the creation time encoded in *payload*'s UUIDv7, in Unix seconds.

    Identifiers sort by this time, so anything listed by identifier reports
    its creation time from here rather than from a clock read later.

    Args:
        payload: Bare 32-char base32 identifier payload.
    """
    return _payload_created_ms(payload) // 1000


def _record_from_head(payload: str, head: HeadObjectOutputTypeDef) -> FileRecord:
    """Build a ``FileRecord`` from a ``HeadObject`` response.

    ``created_at`` is read from the identifier rather than from S3
    ``LastModified``: listing and cursors both order on the identifier, and
    ``LastModified`` is when the bytes landed, which is later for content whose
    identifier is minted first — a multipart upload completed part by part, or a
    batch's result file written when the batch ends.

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
        created_at=datetime.fromtimestamp(created_ms / 1000, UTC)
        if (created_ms := _payload_created_ms(payload))
        else head["LastModified"].replace(tzinfo=UTC),
        expires_at=int(v) if (v := meta.get("expires-at", "")) else None,
    )


def _is_expired(record: FileRecord) -> bool:
    """Return whether *record* has reached its expiry."""
    return record.expires_at is not None and now_utc_timestamp() >= record.expires_at


def head_is_expired(payload: str, head: HeadObjectOutputTypeDef) -> bool:
    """Return whether the file *head* describes has reached its expiry.

    Every read of a stored file goes through it, so that an input referencing a
    file expires at the same instant retrieving that file does.

    Args:
        payload: Bare 32-char base32 file payload.
        head: ``HeadObject`` response dict.
    """
    return _is_expired(_record_from_head(payload, head))


async def upload_file(
    file: InputFile,
    purpose: str | None = None,
    expires_after: int | None = None,
    region: RegionName | None = None,
) -> FileRecord:
    """Stream *file* to S3 and return its metadata record.

    Metadata is encoded in native S3 object attributes — no database needed.
    When *expires_after* is set, the object is tagged with
    :data:`EXPIRING_S3_TAG_KEY` for Lifecycle rule cleanup; code-level expiry
    is enforced by both :func:`get_file` and :func:`list_files`.

    Args:
        file: Input file with content and optional filename / content type.
        purpose: OpenAI purpose string (stored as metadata; not validated).
            ``None``/falsy (e.g. from the Anthropic Files API) is stored as
            :data:`DEFAULT_PURPOSE`, so the stored value always matches what
            is displayed and matched by the ``purpose`` list filter.
        expires_after: Seconds from now until expiry, or ``None``.
        region: AWS region for bucket selection; ``None`` uses the default bucket.

    Raises:
        ApiError: ``aws_s3_bucket`` not configured (503), or a filename the
            stored metadata cannot hold (400).
    """
    bucket = (
        require_s3_bucket_for_region(region, feature=_FEATURE)
        if region
        else _require_bucket()
    )
    payload = encode_id_payload(bucket)
    s3_key = file_id_s3_key(payload)
    expires_at = (
        now_utc_timestamp() + expires_after if expires_after is not None else None
    )

    metadata: _FileMetadata = {
        "expires-at": str(expires_at) if expires_at is not None else "",
        "purpose": purpose or DEFAULT_PURPOSE,
    }
    # Both are resolved from the same metadata lookup, so reading the media
    # type for the fallback name costs nothing beyond the filename itself.
    filename = _sanitize_filename(
        await file.get_filename() or "", await file.get_content_type()
    )
    content_disposition = f'attachment; filename="{filename}"'
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
    if expires_at is not None:
        await s3.put_object_tagging(
            Bucket=bucket, Key=s3_key, Tagging={"TagSet": EXPIRING_S3_TAG_SET}
        )
    return _record_from_head(payload, head)


async def put_file_content(
    payload: str,
    bucket: str,
    data: bytes | AsyncIterator[bytes],
    *,
    filename: str,
    purpose: str,
    content_type: str = "application/jsonl",
    expires_after: int | None = None,
) -> None:
    """Store *data* as the Files API object named by *payload*.

    Unlike :func:`upload_file`, the identifier is chosen by the caller, so a
    payload derived from something stable names the same object every time and
    writing it again is harmless.

    Args:
        payload: Bare 32-char base32 file payload to store the content under.
        bucket: Bucket the payload's fingerprint names.
        data: File content as bytes or an async byte-chunk iterator.
        filename: Filename reported by the Files API.
        purpose: OpenAI purpose string stored with the object.
        content_type: MIME type of the content.
        expires_after: Seconds the content stays readable, counted from now.
            ``None`` keeps it until it is deleted.
    """
    expires_at = (
        now_utc_timestamp() + expires_after if expires_after is not None else None
    )
    await put_s3_object(
        data,
        content_type,
        bucket=bucket,
        key=file_id_s3_key(payload),
        content_disposition=f'attachment; filename="{_sanitize_filename(filename, content_type)}"',
        metadata={
            "expires-at": str(expires_at) if expires_at is not None else "",
            "purpose": purpose,
        },
        expiring=expires_at is not None,
    )


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
    if _is_expired(record):
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
    """``HeadObject`` *key* and return a ``FileRecord``, or ``None`` if absent or expired.

    Deletion of an expired object is left to the retrieve path and to the
    bucket's lifecycle rule: a listing sees as many objects as it scans, and
    scheduling one delete per expired key would fan out with the bucket.
    """
    try:
        head = await s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise  # pragma: no cover
    record = _record_from_head(key.removeprefix(SETTINGS.aws_s3_files_prefix), head)
    return None if _is_expired(record) else record


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


async def _fill_page(
    keys: list[str], limit: int, purpose: str | None = None
) -> tuple[list[FileRecord], bool]:
    """Load *keys* in order until *limit* records exist, reporting whether more follow.

    Deleted and expired keys resolve to no record, so loading only the first
    *limit* of them could answer a short -- or empty -- page while live files
    remain behind them. An empty page with more to come stalls the SDK pagers,
    which stop there and have no cursor to resume from.

    Args:
        keys: Candidate keys, already in the answer's order.
        limit: Maximum records the page holds.
        purpose: Keep only records carrying this purpose, when given.

    Returns:
        ``(records, has_more)`` tuple.
    """
    records: list[FileRecord] = []
    for start in range(0, len(keys), limit + 1):
        batch = await _records_for_keys(keys[start : start + limit + 1])
        records.extend(r for r in batch if purpose is None or r.purpose == purpose)
        if len(records) > limit:
            return records[:limit], True
    return records[:limit], False


async def list_files(
    after: str | None, before: str | None, limit: int, order: str, purpose: str | None
) -> tuple[list[FileRecord], bool]:
    """List files from all configured S3 buckets with cursor-based pagination.

    S3 ``ListObjectsV2`` returns keys in ascending lexicographic order; because
    file keys open with the UUIDv7 that also names each record's ``created_at``,
    ascending key order is ascending creation-time order.  Keys from multiple
    buckets are merged and sorted globally before paging.  Expired files are left
    out of the page, so a page may hold fewer records than *limit*.

    Args:
        after: Return files created strictly after this bare payload (exclusive).
        before: Return the page ending immediately before this bare payload
            (Anthropic ``before_id``), so the page is taken from the records
            nearest the cursor rather than from the start of the listing.
        limit: Maximum records to return.
        order: ``"asc"`` or ``"desc"`` (OpenAI default is ``"desc"``).
        purpose: Filter by purpose, read from the ``HeadObject`` metadata of one
            page-sized batch of candidates at a time.

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
        records: list[FileRecord] = []
        while len(records) <= limit:
            per_bucket = await gather(
                *[_scan_bucket_page(b, prefix, start_after, limit + 1) for b in buckets]
            )
            keys = [
                k
                for k in sorted(chain.from_iterable(per_bucket))
                if len(k) - prefix_len == 32
            ][: limit + 1]
            if not keys or keys[-1] == start_after:
                break
            # Keys that resolve to no record (deleted or expired) are skipped, so
            # the scan continues from the last one seen until the page is full.
            start_after = keys[-1]
            records.extend(await _records_for_keys(keys))
        return records[:limit], len(records) > limit

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
        # The cursor's page is the one the listing reaches just ahead of it, so
        # which side of the cursor to keep follows the listing's direction.
        cursor = file_id_s3_key(before)
        all_keys = [
            k for k in all_keys if (k > cursor if order == "desc" else k < cursor)
        ]

    ordered = all_keys[::-1] if order == "desc" else all_keys
    if before:
        # The page ends at the cursor, so it is filled from the far end.
        page, has_more = await _fill_page(ordered[::-1], limit, purpose)
        return page[::-1], has_more
    return await _fill_page(ordered, limit, purpose)
