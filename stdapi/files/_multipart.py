"""Multipart upload sessions backed by S3 native multipart uploads.

A lean zero-byte marker (``{aws_s3_tmp_prefix}{upload_id}``) stores only
``filename``, ``mime-type``, ``purpose``, and ``total-bytes`` — the fields
unreachable from an in-progress S3 multipart upload (``list_multipart_uploads``
does not expose ``ContentType``, ``ContentDisposition``, or ``Metadata``).
``created_at``/``expires_at`` are derived from the uuid7 timestamp in the
``upload_id``; ``expires_at`` reflects the S3 lifecycle cleanup window (1 day).
The marker is deleted in the background on completion or cancellation.

The S3 multipart upload ID is resolved via a per-process LRU cache
(``upload_id → (s3_upload_id, part_count)``), making the happy path for
:func:`add_part` a single S3 ``upload_part`` call.  ALB sticky sessions
maximise cache hits across sequential parts.  Concurrent calls from different
pods may race on the part number (last writer wins); sequential use is safe.

ID formats
----------
- Session: ``upload_{base32(uuid7_bytes(16) + crc32_bytes(4))}`` — swapping
  ``upload_`` for ``file-`` gives the final file ID (O(1) bucket resolution).
- Part: ``part_{fingerprint(16 hex)}{part_number(4 hex)}{random(12 hex)}``
"""

from asyncio import gather
from base64 import b32decode, b32encode
from binascii import crc32 as _crc32
from contextlib import suppress
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Never
from uuid import uuid4, uuid7

from botocore.exceptions import ClientError

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.aws_s3 import BUCKET_TO_REGION, S3_TAGGING, track_temporary_s3_objects
from stdapi.config import SETTINGS
from stdapi.files._core import (
    FileRecord,
    _record_from_head,
    _require_bucket,
    file_id_s3_key,
    resolve_file_bucket,
)
from stdapi.utils import now_utc_timestamp

if TYPE_CHECKING:
    from types_aiobotocore_s3.client import S3Client

#: TTL in seconds for a pending multipart session (1 day, matching the S3 lifecycle cleanup window).
_MULTIPART_EXPIRY_SECONDS: int = 86400

#: Per-process cache: upload_id → (s3_upload_id, part_count, expires_monotonic).
_cache: dict[str, tuple[str, int, float]] = {}


def _cache_get(upload_id: str) -> tuple[str, int] | None:
    """Return cached ``(s3_upload_id, part_count)`` for *upload_id*, or ``None`` if absent/expired."""
    if entry := _cache.get(upload_id):
        s3_upload_id, part_count, expires = entry
        if monotonic() < expires:
            return s3_upload_id, part_count
        del _cache[upload_id]
    return None


def _cache_set(upload_id: str, s3_upload_id: str, part_count: int) -> None:
    """Store *s3_upload_id* and *part_count* in the per-process cache for the session TTL."""
    _cache[upload_id] = (
        s3_upload_id,
        part_count,
        monotonic() + _MULTIPART_EXPIRY_SECONDS,
    )


def _cache_del(upload_id: str) -> None:
    """Evict *upload_id* from the per-process cache."""
    _cache.pop(upload_id, None)


def _file_id_from_upload_id(upload_id: str) -> str:
    """Return the bare file payload for *upload_id* by stripping the ``upload_`` prefix."""
    return upload_id[7:]


def _multipart_ids_from_bucket(bucket: str) -> tuple[str, str]:
    """Generate a linked ``(upload_id, file_payload)`` pair sharing one base32 payload.

    Sharing the payload means :func:`~stdapi.files._core.resolve_file_bucket` works
    on both, and :func:`_file_id_from_upload_id` is a pure string prefix strip.

    Args:
        bucket: S3 bucket name the session will be stored in.
    """
    payload = (
        b32encode(uuid7().bytes + _crc32(bucket.encode()).to_bytes(4, "big"))
        .lower()
        .decode()
    )
    return f"upload_{payload}", payload


def _upload_fingerprint(upload_id: str) -> str:
    """Return the 16-hex-char fingerprint for *upload_id* (first 8 bytes of its base32 payload)."""
    return b32decode(upload_id[7:].upper())[:8].hex()


def _multipart_meta_key(upload_id: str) -> str:
    """Return the S3 key for the session metadata marker, e.g. ``tmp/upload_{...}``."""
    return f"{SETTINGS.aws_s3_tmp_prefix}{upload_id}"


def _created_at_from_upload_id(upload_id: str) -> int:
    """Extract the Unix creation timestamp (seconds) from the uuid7 in *upload_id*.

    UUID7's first 48 bits are the millisecond Unix timestamp; the upload_id
    payload is ``uuid7_bytes (16) + crc32_bytes (4)`` base32-encoded.

    Returns:
        Unix timestamp in seconds.
    """
    return int.from_bytes(b32decode(upload_id[7:].upper())[:6], "big") // 1000


def _make_part_id(upload_id: str, part_number: int) -> str:
    """Generate a part ID from the upload fingerprint and 1-based part number.

    Format: ``part_{fingerprint(16 hex)}{part_number(4 hex)}{random(12 hex)}``.

    Args:
        upload_id: Source of the fingerprint (first 8 bytes of the base32 payload).
        part_number: 1-based S3 part number.

    Returns:
        Part ID string.
    """
    return f"part_{_upload_fingerprint(upload_id)}{part_number:04x}{uuid4().hex[:12]}"


def _extract_part_number(part_id: str, upload_id: str) -> int:
    """Extract the 1-based S3 part number from *part_id*, validating the fingerprint.

    Args:
        part_id: Part identifier to parse.
        upload_id: Upload session identifier used to verify the fingerprint.

    Returns:
        1-based S3 part number.

    Raises:
        ApiError: Fingerprint mismatch — part does not belong to this upload (400).
    """
    if part_id[5:21] != _upload_fingerprint(upload_id):
        msg = f"Part '{part_id}' does not belong to upload '{upload_id}'."
        raise ApiError(msg)
    return int(part_id[21:25], 16)


@dataclass(slots=True)
class MultipartSession:
    """Immutable creation-time metadata for a multipart upload session.

    Read from the S3 marker object; never updated after creation.

    Attributes:
        upload_id: Session identifier.
        file_id: Pre-derived bare file payload for the completed object.
        s3_bucket: S3 bucket.
        s3_key: S3 key for the assembled file (``{prefix}{payload}``).
        filename: Original filename.
        mime_type: MIME type declared at session creation.
        purpose: OpenAI purpose string.
        total_bytes: Declared total size in bytes (validated at completion).
        expires_at: Unix timestamp (seconds) when the session expires.
        created_at: Unix timestamp (seconds) when the session was created.
    """

    upload_id: str
    file_id: str
    s3_bucket: str
    s3_key: str
    filename: str
    mime_type: str
    purpose: str
    total_bytes: int
    expires_at: int
    created_at: int


async def _load_multipart_session(
    upload_id: str, bucket: str, s3: S3Client
) -> MultipartSession:
    """Read session metadata from the S3 marker via ``HeadObject``.

    Args:
        upload_id: Session identifier.
        bucket: S3 bucket containing the marker.
        s3: Authenticated S3 client.

    Returns:
        Populated ``MultipartSession``.

    Raises:
        ApiError: Marker not found (404).
    """
    try:
        head = await s3.head_object(Bucket=bucket, Key=_multipart_meta_key(upload_id))
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            msg = f"Upload '{upload_id}' not found."
            raise ApiError(msg, status=404) from exc
        raise  # pragma: no cover
    meta = head["Metadata"]
    file_id = _file_id_from_upload_id(upload_id)
    created_at = _created_at_from_upload_id(upload_id)
    return MultipartSession(
        upload_id=upload_id,
        file_id=file_id,
        s3_bucket=bucket,
        s3_key=file_id_s3_key(file_id),
        filename=meta["filename"],
        mime_type=meta["mime-type"],
        purpose=meta["purpose"],
        total_bytes=int(meta["total-bytes"]),
        expires_at=created_at + _MULTIPART_EXPIRY_SECONDS,
        created_at=created_at,
    )


async def _check_not_pending(upload_id: str, bucket: str, s3: S3Client) -> Never:
    """Distinguish 404 (session never existed) from 400 (session exists but not pending).

    Args:
        upload_id: Session identifier.
        bucket: S3 bucket containing the marker.
        s3: Authenticated S3 client.

    Raises:
        ApiError: 404 if the marker is absent; 400 if present (not pending).
    """
    try:
        await s3.head_object(Bucket=bucket, Key=_multipart_meta_key(upload_id))
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            msg = f"Upload '{upload_id}' not found."
            raise ApiError(msg, status=404) from exc
        raise  # pragma: no cover
    msg = f"Upload '{upload_id}' is not pending."
    raise ApiError(msg)


async def _require_s3_upload_id(
    upload_id: str, bucket: str, s3_key: str, s3: S3Client
) -> str:
    """Return the S3 multipart upload ID for a pending session.

    Checks the per-process cache first; on a cache miss falls back to
    ``list_multipart_uploads`` + ``list_parts`` to populate the cache.

    Args:
        upload_id: Session identifier.
        bucket: S3 bucket.
        s3_key: S3 key of the in-progress multipart upload.
        s3: Authenticated S3 client.

    Raises:
        ApiError: 404 if the session does not exist; 400 if not pending.
    """
    if cached := _cache_get(upload_id):
        return cached[0]

    for upload in (await s3.list_multipart_uploads(Bucket=bucket, Prefix=s3_key)).get(
        "Uploads", []
    ):
        if upload["Key"] == s3_key:
            s3_upload_id = upload["UploadId"]
            _cache_set(
                upload_id,
                s3_upload_id,
                len(await _list_all_parts(s3, bucket, s3_key, s3_upload_id)),
            )
            return s3_upload_id

    return await _check_not_pending(upload_id, bucket, s3)


async def _list_all_parts(
    s3: S3Client, bucket: str, key: str, s3_upload_id: str
) -> dict[int, tuple[str, int]]:
    """Paginate ``list_parts`` and return all parts as ``{part_number: (etag, size)}``."""
    parts: dict[int, tuple[str, int]] = {}
    marker = 0
    while True:
        resp = await s3.list_parts(
            Bucket=bucket, Key=key, UploadId=s3_upload_id, PartNumberMarker=marker
        )
        parts |= {
            p["PartNumber"]: (p["ETag"], p["Size"]) for p in resp.get("Parts", [])
        }
        if not resp.get("IsTruncated"):
            break
        marker = resp["NextPartNumberMarker"]
    return parts


async def create_multipart_session(
    filename: str, mime_type: str, purpose: str, total_bytes: int
) -> MultipartSession:
    """Create an S3 native multipart upload and write a metadata marker in parallel.

    Args:
        filename: Original filename for the final file.
        mime_type: MIME type declared for the final file.
        purpose: OpenAI purpose string.
        total_bytes: Declared total size in bytes (validated at completion).

    Raises:
        ApiError: ``aws_s3_bucket`` not configured (503).
    """
    bucket = _require_bucket()
    upload_id, file_id = _multipart_ids_from_bucket(bucket)
    s3_key = file_id_s3_key(file_id)
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(bucket))
    created_at = _created_at_from_upload_id(upload_id)

    multipart_resp, _ = await gather(
        s3.create_multipart_upload(
            Bucket=bucket,
            Key=s3_key,
            ContentType=mime_type,
            ContentDisposition=f'attachment; filename="{filename}"',
            Metadata={"purpose": purpose, "expires-at": ""},
            Tagging=S3_TAGGING,
        ),
        s3.put_object(
            Bucket=bucket,
            Key=_multipart_meta_key(upload_id),
            Body=b"",
            ContentType="application/octet-stream",
            Metadata={
                "filename": filename,
                "mime-type": mime_type,
                "purpose": purpose,
                "total-bytes": str(total_bytes),
            },
            Tagging=S3_TAGGING,
        ),
    )
    _cache_set(upload_id, multipart_resp["UploadId"], 0)

    return MultipartSession(
        upload_id=upload_id,
        file_id=file_id,
        s3_bucket=bucket,
        s3_key=s3_key,
        filename=filename,
        mime_type=mime_type,
        purpose=purpose,
        total_bytes=total_bytes,
        expires_at=created_at + _MULTIPART_EXPIRY_SECONDS,
        created_at=created_at,
    )


async def add_part(upload_id: str, data: bytes) -> tuple[str, int]:
    """Add a part to an existing multipart session.

    On a cache hit (same pod, sequential uploads) only a single S3
    ``upload_part`` call is made.  On a cache miss the session is
    rediscovered via ``list_multipart_uploads`` + ``list_parts``.

    Args:
        upload_id: Session identifier.
        data: Raw bytes for this part (max 64 MiB per OpenAI spec).

    Returns:
        ``(part_id, created_at)`` — part_id encodes the 1-based part number.

    Raises:
        ApiError: 404 if the session does not exist; 400 if not pending.
    """
    file_id = _file_id_from_upload_id(upload_id)
    bucket = resolve_file_bucket(file_id)
    s3_key = file_id_s3_key(file_id)
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(bucket))

    if not (cached := _cache_get(upload_id)):
        await _require_s3_upload_id(upload_id, bucket, s3_key, s3)
        cached = _cache_get(upload_id) or ("", 0)
    s3_upload_id, part_count = cached

    part_number = part_count + 1

    try:
        await s3.upload_part(
            Bucket=bucket,
            Key=s3_key,
            UploadId=s3_upload_id,
            PartNumber=part_number,
            Body=data,
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchUpload":
            _cache_del(upload_id)
            await _check_not_pending(upload_id, bucket, s3)
        raise  # pragma: no cover

    _cache_set(upload_id, s3_upload_id, part_number)
    return _make_part_id(upload_id, part_number), now_utc_timestamp()


async def complete_multipart_session(
    upload_id: str, part_ids: list[str]
) -> tuple[MultipartSession, FileRecord]:
    """Assemble parts and produce a file record, validating fingerprints and total size.

    Args:
        upload_id: Session identifier.
        part_ids: Ordered part IDs to include in the final file.

    Returns:
        ``(session, file_record)`` — session metadata and the assembled file.

    Raises:
        ApiError: 404 not found; 400 not pending, bad part fingerprint,
            unknown part, or size mismatch.
    """
    file_id = _file_id_from_upload_id(upload_id)
    bucket = resolve_file_bucket(file_id)
    s3_key = file_id_s3_key(file_id)
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(bucket))

    session, s3_upload_id = await gather(
        _load_multipart_session(upload_id, bucket, s3),
        _require_s3_upload_id(upload_id, bucket, s3_key, s3),
    )
    parts_info = await _list_all_parts(s3, bucket, s3_key, s3_upload_id)

    s3_parts: list[dict[str, int | str]] = []
    assembled_size = 0
    for pid in part_ids:
        pn = _extract_part_number(pid, upload_id)
        if (part := parts_info.get(pn)) is None:
            msg = f"Part '{pid}' (number {pn}) was not uploaded."
            raise ApiError(msg)
        etag, size = part
        s3_parts.append({"PartNumber": pn, "ETag": etag})
        assembled_size += size

    if assembled_size != session.total_bytes:
        msg = (
            f"Assembled size {assembled_size} does not match "
            f"declared bytes {session.total_bytes}."
        )
        raise ApiError(msg)

    await s3.complete_multipart_upload(
        Bucket=bucket,
        Key=s3_key,
        UploadId=s3_upload_id,
        MultipartUpload={"Parts": s3_parts},  # type: ignore[typeddict-item]
    )
    _cache_del(upload_id)
    track_temporary_s3_objects(bucket, _multipart_meta_key(upload_id))
    return session, _record_from_head(
        file_id, await s3.head_object(Bucket=bucket, Key=s3_key)
    )


async def cancel_multipart_session(upload_id: str) -> MultipartSession:
    """Cancel a pending session and abort the S3 multipart upload.

    Raises:
        ApiError: 404 if the session does not exist; 400 if not pending.
    """
    file_id = _file_id_from_upload_id(upload_id)
    bucket = resolve_file_bucket(file_id)
    s3_key = file_id_s3_key(file_id)
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(bucket))

    session, s3_upload_id = await gather(
        _load_multipart_session(upload_id, bucket, s3),
        _require_s3_upload_id(upload_id, bucket, s3_key, s3),
    )
    with suppress(ClientError):
        await s3.abort_multipart_upload(
            Bucket=bucket, Key=s3_key, UploadId=s3_upload_id
        )
    _cache_del(upload_id)
    track_temporary_s3_objects(bucket, _multipart_meta_key(upload_id))
    return session
