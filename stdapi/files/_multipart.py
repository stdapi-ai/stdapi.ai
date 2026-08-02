"""Multipart upload sessions backed by S3 native multipart uploads.

A lean zero-byte marker (``{aws_s3_tmp_prefix}{upload_id}``) stores only
``filename``, ``mime-type``, ``purpose``, and ``total-bytes`` — the fields
unreachable from an in-progress S3 multipart upload (``list_multipart_uploads``
does not expose ``ContentType``, ``ContentDisposition``, or ``Metadata``).
``created_at``/``expires_at`` are derived from the uuid7 timestamp in the
``upload_id``; ``expires_at`` reflects the S3 lifecycle cleanup window (1 day).
The marker is deleted in the background on completion or cancellation.

The S3 multipart upload ID is resolved via a bounded per-process LRU cache
(``upload_id → (s3_upload_id, expiry)``), sparing :func:`add_part` a
``list_multipart_uploads`` call; ALB sticky sessions maximise cache hits across
sequential parts.  Concurrent calls from different pods may race on the part
number (last writer wins); sequential use is safe.

ID formats
----------
- Session: ``upload_{base32hex(uuid7_bytes(16) + crc32_bytes(4))}`` — swapping
  ``upload_`` for ``file-`` gives the final file ID (O(1) bucket resolution).
- Part: ``part_{fingerprint(16 hex)}{part_number(4 hex)}{random(12 hex)}``
"""

from asyncio import gather
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from time import monotonic
from typing import TYPE_CHECKING, Never
from uuid import uuid4

from botocore.exceptions import ClientError

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.aws_s3 import BUCKET_TO_REGION, S3_TAGGING, track_temporary_s3_objects
from stdapi.config import SETTINGS
from stdapi.files._core import (
    FileRecord,
    _record_from_head,
    _require_bucket,
    _validate_filename,
    decode_id_payload,
    encode_id_payload,
    file_id_s3_key,
    resolve_file_bucket,
)
from stdapi.utils import now_utc_timestamp

if TYPE_CHECKING:
    from types_aiobotocore_s3.client import S3Client

#: TTL in seconds for a pending multipart session (1 day, matching the S3 lifecycle cleanup window).
_MULTIPART_EXPIRY_SECONDS: int = 86400

#: Maximum number of parts a multipart upload session can hold (S3 limit).
_MAX_PART_NUMBER: int = 10000

#: Minimum size in bytes for a part that is not the last one (S3 limit, 5 MiB).
_MIN_PART_SIZE: int = 5 * 1024 * 1024

#: S3 tagging query string marking an object for Lifecycle expiry cleanup.
_EXPIRING_S3_TAGGING: str = f"{S3_TAGGING}&stdapi-ai.expires=true"

#: Per-process cache: upload_id → (s3_upload_id, expires_monotonic) (bounded LRU).
_cache: dict[str, tuple[str, float]] = {}

#: Maximum entries retained in the multipart session cache.
_CACHE_MAX: int = 4096


def _cache_get(upload_id: str) -> str | None:
    """Return the cached ``s3_upload_id`` for *upload_id*, or ``None`` if absent/expired."""
    if entry := _cache.pop(upload_id, None):
        s3_upload_id, expires = entry
        if monotonic() < expires:
            _cache[upload_id] = entry
            return s3_upload_id
    return None


def _cache_set(upload_id: str, s3_upload_id: str) -> None:
    """Store *s3_upload_id* in the per-process cache for the session TTL.

    Sessions abandoned without a completion or cancellation are never looked up
    again, so the least recently used entry is dropped once the cache is full.
    """
    if upload_id not in _cache and len(_cache) >= _CACHE_MAX:
        del _cache[next(iter(_cache))]
    _cache[upload_id] = (s3_upload_id, monotonic() + _MULTIPART_EXPIRY_SECONDS)


def _cache_del(upload_id: str) -> None:
    """Evict *upload_id* from the per-process cache."""
    _cache.pop(upload_id, None)


def _file_id_from_upload_id(upload_id: str) -> str:
    """Return the bare file payload for *upload_id* by stripping the ``upload_`` prefix."""
    return upload_id[7:]


def _multipart_ids_from_bucket(bucket: str) -> tuple[str, str]:
    """Generate a linked ``(upload_id, file_payload)`` pair sharing one payload.

    Sharing the payload means :func:`~stdapi.files._core.resolve_file_bucket` works
    on both, and :func:`_file_id_from_upload_id` is a pure string prefix strip.

    Args:
        bucket: S3 bucket name the session will be stored in.
    """
    payload = encode_id_payload(bucket)
    return f"upload_{payload}", payload


def _upload_fingerprint(upload_id: str) -> str:
    """Return the 16-hex-char fingerprint for *upload_id*.

    Digested from the whole payload rather than sliced out of it: the payload
    opens with the uuid7 millisecond timestamp, so two sessions created in the
    same millisecond share every leading byte and would share a fingerprint,
    letting a part addressed to one session pass the ownership check of the other.
    """
    return sha256(decode_id_payload(upload_id[7:])).digest()[:8].hex()


def _multipart_meta_key(upload_id: str) -> str:
    """Return the S3 key for the session metadata marker, e.g. ``tmp/upload_{...}``."""
    return f"{SETTINGS.aws_s3_tmp_prefix}{upload_id}"


def _created_at_from_upload_id(upload_id: str) -> int:
    """Extract the Unix creation timestamp (seconds) from the uuid7 in *upload_id*.

    UUID7's first 48 bits are the millisecond Unix timestamp; the upload_id
    payload is ``uuid7_bytes (16) + crc32_bytes (4)`` encoded.

    Returns:
        Unix timestamp in seconds.
    """
    return int.from_bytes(decode_id_payload(upload_id[7:])[:6], "big") // 1000


def _make_part_id(upload_id: str, part_number: int) -> str:
    """Generate a part ID from the upload fingerprint and 1-based part number.

    Format: ``part_{fingerprint(16 hex)}{part_number(4 hex)}{random(12 hex)}``.

    Args:
        upload_id: Source of the fingerprint (first 8 bytes of the payload).
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
    ``list_multipart_uploads`` to populate the cache. The upload ID is fixed
    for the session's lifetime, so caching it never goes stale.

    Args:
        upload_id: Session identifier.
        bucket: S3 bucket.
        s3_key: S3 key of the in-progress multipart upload.
        s3: Authenticated S3 client.

    Raises:
        ApiError: 404 if the session does not exist; 400 if not pending.
    """
    if cached := _cache_get(upload_id):
        return cached

    for upload in (await s3.list_multipart_uploads(Bucket=bucket, Prefix=s3_key)).get(
        "Uploads", []
    ):
        if upload["Key"] == s3_key:
            s3_upload_id = upload["UploadId"]
            _cache_set(upload_id, s3_upload_id)
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
    filename: str,
    mime_type: str,
    purpose: str,
    total_bytes: int,
    expires_after: int | None = None,
) -> MultipartSession:
    """Create an S3 native multipart upload and write a metadata marker in parallel.

    The final file's expiry is stamped on the S3 multipart upload itself
    (metadata and Lifecycle tag), so the assembled object inherits it without
    any extra call at completion.

    Args:
        filename: Original filename for the final file.
        mime_type: MIME type declared for the final file.
        purpose: OpenAI purpose string.
        total_bytes: Declared total size in bytes (validated at completion).
        expires_after: Seconds from creation until the final file expires, or ``None``.

    Raises:
        ApiError: ``aws_s3_bucket`` not configured (503) or invalid filename.
    """
    filename = _validate_filename(filename)
    bucket = _require_bucket()
    upload_id, file_id = _multipart_ids_from_bucket(bucket)
    s3_key = file_id_s3_key(file_id)
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(bucket))
    created_at = _created_at_from_upload_id(upload_id)
    file_expires_at = created_at + expires_after if expires_after is not None else None

    multipart_resp, _ = await gather(
        s3.create_multipart_upload(
            Bucket=bucket,
            Key=s3_key,
            ContentType=mime_type,
            ContentDisposition=f'attachment; filename="{filename}"',
            Metadata={
                "purpose": purpose,
                "expires-at": str(file_expires_at)
                if file_expires_at is not None
                else "",
            },
            Tagging=_EXPIRING_S3_TAGGING if file_expires_at is not None else S3_TAGGING,
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
    _cache_set(upload_id, multipart_resp["UploadId"])

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

    The part number continues the parts already stored in S3, so consecutive
    parts of one upload keep their order even when a load balancer spreads them
    over several server instances.

    Args:
        upload_id: Session identifier.
        data: Raw bytes for this part (max 64 MiB per OpenAI spec).

    Returns:
        ``(part_id, created_at)`` — part_id encodes the 1-based part number.

    Raises:
        ApiError: 404 if the session does not exist; 400 if not pending or the
            session already holds the maximum number of parts.
    """
    file_id = _file_id_from_upload_id(upload_id)
    bucket = resolve_file_bucket(file_id)
    s3_key = file_id_s3_key(file_id)
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(bucket))

    s3_upload_id = await _require_s3_upload_id(upload_id, bucket, s3_key, s3)
    # Numbering from the parts S3 holds: another instance may have served the
    # previous part, and reusing its number would overwrite it.
    parts = await _list_all_parts(s3, bucket, s3_key, s3_upload_id)
    part_number = max(parts, default=0) + 1
    if part_number > _MAX_PART_NUMBER:
        msg = f"This upload already has the maximum of {_MAX_PART_NUMBER} parts."
        raise ApiError(msg)

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

    return _make_part_id(upload_id, part_number), now_utc_timestamp()


async def complete_multipart_session(
    upload_id: str, part_ids: list[str]
) -> tuple[MultipartSession, FileRecord]:
    """Assemble parts and produce a file record, validating fingerprints and total size.

    S3 cannot reassemble multipart parts out of order, so *part_ids* must be
    strictly ascending by part number; this is checked before any S3 call.

    Args:
        upload_id: Session identifier.
        part_ids: Ordered part IDs to include in the final file.

    Returns:
        ``(session, file_record)`` — session metadata and the assembled file.

    Raises:
        ApiError: 404 not found; 400 not pending, out-of-order part_ids, bad
            part fingerprint, unknown part, undersized non-last part, or size
            mismatch.
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

    # Ordering is validated over the whole list first: only once it is known to be
    # strictly ascending does "last element" reliably mean "highest part number",
    # which the size check below relies on.
    part_numbers: list[int] = []
    previous_pn = 0
    for pid in part_ids:
        pn = _extract_part_number(pid, upload_id)
        if pn <= previous_pn:
            msg = (
                f"Part '{pid}' (number {pn}) is out of order: part_ids must be "
                "listed in ascending upload order."
            )
            raise ApiError(msg)
        previous_pn = pn
        part_numbers.append(pn)

    s3_parts: list[dict[str, int | str]] = []
    assembled_size = 0
    last_index = len(part_ids) - 1
    for i, (pid, pn) in enumerate(zip(part_ids, part_numbers, strict=True)):
        if (part := parts_info.get(pn)) is None:
            msg = f"Part '{pid}' (number {pn}) was not uploaded."
            raise ApiError(msg)
        etag, size = part
        if i != last_index and size < _MIN_PART_SIZE:
            msg = (
                f"Part '{pid}' (number {pn}) is too small: every part except "
                f"the last must be at least {_MIN_PART_SIZE} bytes."
            )
            raise ApiError(msg)
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
