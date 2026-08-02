"""AWS S3 utilities."""

import contextlib
from asyncio import Semaphore, Task, create_task, gather
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from urllib.parse import urlencode
from uuid import uuid4

from botocore.exceptions import ClientError

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.cleanup import schedule_cleanup
from stdapi.config import SETTINGS
from stdapi.monitoring import log_error_details
from stdapi.server import AWS_APN_ID
from stdapi.utils import async_iter, buffered_chunks, chain_async_iterators

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_s3.client import S3Client
    from types_aiobotocore_s3.type_defs import CopySourceTypeDef

#: Default S3 ``Tagging`` query string
S3_TAGGING: Final[str] = urlencode({"aws-apn-id": AWS_APN_ID})

#: Maximum object size supported by S3 ``copy_object`` in a single request (5 GiB).
_COPY_OBJECT_MAX_BYTES: Final[int] = 5 * 1024 * 1024 * 1024

#: Multipart copy chunk size in bytes (512 MiB).
_MULTIPART_COPY_PART_SIZE: Final[int] = 512 * 1024 * 1024

#: Multipart upload chunk size (8 MiB).
UPLOAD_CHUNK_SIZE: Final[int] = 8 * 1024 * 1024

#: Maximum concurrent part copies during a multipart server-side copy.
MULTIPART_COPY_CONCURRENCY: Final[int] = 8

#: Maximum multipart-upload parts in flight (memory bound: this x chunk size).
_UPLOAD_PARTS_IN_FLIGHT: Final[int] = 2


def _bucket_to_region() -> dict[str, RegionName]:
    """Build the reverse bucket → region mapping from the settings.

    Returns:
        Bucket name mapped to its region, covering the regional buckets, the
        default bucket, and the Transcribe bucket.
    """
    mapping: dict[str, RegionName] = {
        bucket_name: region
        for region, bucket_name in SETTINGS.aws_s3_regional_buckets.items()
    }
    if SETTINGS.aws_s3_bucket:
        mapping.setdefault(SETTINGS.aws_s3_bucket, SETTINGS.aws_bedrock_regions[0])
    if SETTINGS.aws_transcribe_s3_bucket:
        # Existing mappings win: a bucket shared with the above keeps its region.
        mapping.setdefault(
            SETTINGS.aws_transcribe_s3_bucket,
            SETTINGS.aws_transcribe_region or SETTINGS.aws_bedrock_regions[0],
        )
    return mapping


#: Reverse bucket → region mapping (includes the default and Transcribe buckets).
BUCKET_TO_REGION: dict[str, RegionName] = _bucket_to_region()


@dataclass(slots=True)
class S3Object:
    """Reference to an S3 object.

    Attributes:
        bucket: S3 bucket name.
        key: S3 object key.
    """

    bucket: str
    key: str

    @property
    def uri(self) -> str:
        """The ``s3://bucket/key`` URI."""
        return f"s3://{self.bucket}/{self.key}"


def track_temporary_s3_objects(bucket: str, *keys: str) -> None:
    """Record a newly created S3 object for later cleanup.

    Args:
        bucket: S3 bucket name.
        *keys: S3 object key(s).
    """
    s3 = get_client("s3", BUCKET_TO_REGION.get(bucket))
    schedule_cleanup(*(s3.delete_object(Bucket=bucket, Key=key) for key in keys))


def _get_tmp_key(content_type: str | None = None) -> str:
    """Generates a temporary key for an object in AWS S3.

    Args:
        content_type: Content type of the object.

    Returns:
        str: The generated temporary key.
    """
    ext = f".{content_type.split('/', 1)[1].split(';', 1)[0]}" if content_type else ""
    return f"{SETTINGS.aws_s3_tmp_prefix}{uuid4().hex}{ext}"


def get_s3_bucket_for_region(region: RegionName) -> str | None:
    """Return S3 bucket for the given region, or None if not available.

    Checks regional buckets first, then falls back to the default bucket
    if the region is the primary Bedrock region.

    Args:
        region: AWS region identifier.

    Returns:
        S3 bucket name or None if no bucket is available for this region.
    """
    if bucket := SETTINGS.aws_s3_regional_buckets.get(region):
        return bucket
    if region == SETTINGS.aws_bedrock_regions[0]:
        return SETTINGS.aws_s3_bucket or None
    return None


def require_s3_bucket_for_region(region: RegionName) -> str:
    """Return S3 bucket for the region, raising :class:`ApiError` if missing.

    Args:
        region: AWS region identifier.

    Returns:
        S3 bucket name.

    Raises:
        ApiError: If no S3 bucket is configured for the region.
    """
    if bucket := get_s3_bucket_for_region(region):
        return bucket
    if region == SETTINGS.aws_bedrock_regions[0]:
        log_error_details(
            "S3 bucket not configured (aws_s3_bucket): some features are disabled"
        )
    else:
        log_error_details(
            f"S3 {region} regional bucket not configured "
            "(aws_s3_regional_buckets): some features are disabled"
        )
    msg = (
        "Async invocation is not available on the current server. "
        "Please contact the administrator to enable it."
    )
    raise ApiError(msg)


async def put_object_and_get_url(body: bytes, content_type: str, filename: str) -> str:
    """Uploads an object to an AWS S3 bucket and retrieves the pre-signed URL to access it.

    The URL is valid for 3600 seconds. The S3 prefix is added automatically.

    Args:
        body: The binary content of the object to be uploaded.
        content_type: The MIME type of the object being uploaded.
        filename: The name of the file to be stored in the S3 bucket.

    Returns:
        A pre-signed URL for accessing the uploaded object in the S3 bucket.
    """
    s3_bucket = SETTINGS.aws_s3_bucket
    if not s3_bucket:  # pragma: no cover
        log_error_details(
            "No S3 bucket configured for presigned URLs. "
            "AWS_S3_BUCKET environment variable is not set."
        )
        msg = (
            "The url response format is not enabled on this server. "
            "Please contact the administrator to enabled it."
        )
        raise ApiError(msg)

    s3_accelerate_client: S3Client = get_client("s3.accelerate")
    s3_key = f"{SETTINGS.aws_s3_tmp_prefix}{filename}"
    return (
        await gather(
            s3_accelerate_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": s3_bucket, "Key": s3_key},
                ExpiresIn=3600,
            ),
            put_s3_object(body, content_type, bucket=s3_bucket, key=s3_key),
        )
    )[0]


async def get_bytes_from_s3(s3_bucket: str, s3_key: str) -> bytes:
    """Retrieve raw bytes from an S3 object.

    Args:
        s3_bucket: Name of the S3 bucket containing the object.
        s3_key: Key (path) of the object within the S3 bucket.

    Returns:
        Raw bytes content of the S3 object.
    """
    s3_client: S3Client = get_client("s3", BUCKET_TO_REGION.get(s3_bucket))
    return await (await s3_client.get_object(Bucket=s3_bucket, Key=s3_key))[
        "Body"
    ].read()


async def get_text_from_s3(s3_bucket: str, s3_key: str) -> str:
    """Retrieve and decode S3 object content as a string.

    Args:
        s3_bucket: Name of the S3 bucket containing the object
        s3_key: Key (path) of the object within the S3 bucket

    Returns:
        Decoded string content of the S3 object
    """
    return (await get_bytes_from_s3(s3_bucket, s3_key)).decode()


async def multipart_copy_parts(
    s3: S3Client,
    *,
    bucket: str,
    key: str,
    upload_id: str,
    copy_source: CopySourceTypeDef,
    size: int,
    part_size: int,
) -> list[dict[str, int | str]]:
    """Server-side copy all byte ranges of a multipart copy concurrently.

    Runs the ranged ``upload_part_copy`` calls with bounded concurrency
    (``MULTIPART_COPY_CONCURRENCY``); part numbers follow the byte ranges.

    Args:
        s3: S3 client.
        bucket: Destination bucket.
        key: Destination object key.
        upload_id: Multipart upload ID.
        copy_source: Source object reference.
        size: Source object size in bytes.
        part_size: Bytes per copied part.

    Returns:
        Completed parts, ordered by part number.
    """
    semaphore = Semaphore(MULTIPART_COPY_CONCURRENCY)

    async def _copy_part(part_number: int, start: int) -> dict[str, int | str]:
        """Copy one byte range under the part-copy concurrency bound."""
        async with semaphore:
            end = min(start + part_size, size) - 1
            etag = (
                await s3.upload_part_copy(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    CopySource=copy_source,
                    CopySourceRange=f"bytes={start}-{end}",
                )
            )["CopyPartResult"]["ETag"]
            return {"PartNumber": part_number, "ETag": etag}

    return list(
        await gather(
            *(
                _copy_part(part_number, start)
                for part_number, start in enumerate(range(0, size, part_size), start=1)
            )
        )
    )


async def copy_s3_object(
    source_bucket: str,
    source_key: str,
    *,
    dest_bucket: str | None = None,
    dest_key: str | None = None,
    dest_region: RegionName | None = None,
    content_type: str | None = None,
    temporary: bool = False,
) -> S3Object:
    """Copy an S3 object between buckets using server-side copy.

    Uses a single-request copy for objects up to 5 GiB and multipart copy for
    larger objects.

    Args:
        source_bucket: Source S3 bucket name.
        source_key: Source S3 object key.
        dest_bucket: Destination S3 bucket name. When ``None``, the
            regional bucket for *dest_region* is used.
        dest_key: Destination S3 object key. When ``None``, a temporary
            key is auto-generated via :func:`_get_tmp_key`.
        dest_region: Region for the destination bucket/client.
        content_type: Optional MIME type used to derive the file extension
            when auto-generating *dest_key*.
        temporary: If ``True``, the object will be deleted when the request ends.

    Returns:
        An :class:`S3Object` referencing the copied object.

    Raises:
        BotoCoreError: If the AWS SDK fails.
        ClientError: If S3 returns an error.
        ValueError: If the source object has an invalid size.
    """
    s3 = get_client("s3", dest_region)
    size = (await s3.head_object(Bucket=source_bucket, Key=source_key))["ContentLength"]

    dest_bucket = dest_bucket or (
        require_s3_bucket_for_region(dest_region) if dest_region else source_bucket
    )
    dest_key = dest_key or _get_tmp_key(content_type)
    copy_source: CopySourceTypeDef = {"Bucket": source_bucket, "Key": source_key}
    if size <= _COPY_OBJECT_MAX_BYTES:
        await s3.copy_object(
            Bucket=dest_bucket,
            Key=dest_key,
            CopySource=copy_source,
            Tagging=S3_TAGGING,
            TaggingDirective="REPLACE",
        )
    else:
        upload_id: str | None = None
        try:
            upload_id = (
                await s3.create_multipart_upload(
                    Bucket=dest_bucket, Key=dest_key, Tagging=S3_TAGGING
                )
            )["UploadId"]

            parts = await multipart_copy_parts(
                s3,
                bucket=dest_bucket,
                key=dest_key,
                upload_id=upload_id,
                copy_source=copy_source,
                size=size,
                part_size=_MULTIPART_COPY_PART_SIZE,
            )

            await s3.complete_multipart_upload(
                Bucket=dest_bucket,
                Key=dest_key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
        except Exception:
            if upload_id is not None:
                with contextlib.suppress(ClientError):
                    await s3.abort_multipart_upload(
                        Bucket=dest_bucket, Key=dest_key, UploadId=upload_id
                    )
            raise

    if temporary:
        track_temporary_s3_objects(dest_bucket, dest_key)

    return S3Object(bucket=dest_bucket, key=dest_key)


async def put_s3_object(
    data: bytes | AsyncIterator[bytes],
    content_type: str | None = None,
    *,
    bucket: str | None = None,
    key: str | None = None,
    region: RegionName | None = None,
    temporary: bool = False,
    content_disposition: str | None = None,
    metadata: dict[str, str] | None = None,
) -> S3Object:
    """Upload data to S3, choosing the most efficient strategy.

    The S3 client is resolved automatically from :data:`BUCKET_TO_REGION`.

    - **bytes** objects smaller than 5 GiB use a single ``PutObject``.
    - **bytes** objects >= 5 GiB are uploaded via multipart.
    - **async iterators** are buffered into ``UPLOAD_CHUNK_SIZE`` chunks;
      a single ``PutObject`` is used when everything fits in one chunk,
      otherwise multipart upload is started.

    Args:
        data: File content as bytes or async byte-chunk iterator.
        content_type: Optional MIME type for the ``ContentType`` header.
        bucket: Destination bucket. When ``None``, the regional bucket
            for *region* is used.
        key: Destination object key. When ``None``, a temporary key is
            auto-generated via :func:`_get_tmp_key`.
        region: AWS region for bucket resolution when *bucket* is not
            specified.
        temporary: If ``True``, the object will be deleted when the request ends.
        content_disposition: Optional ``Content-Disposition`` header value.
        metadata: Optional user-defined S3 object metadata key/value pairs.

    Returns:
        An :class:`S3Object` referencing the uploaded object.
    """
    bucket = bucket or (require_s3_bucket_for_region(region) if region else None)
    if not bucket:
        msg = "Either 'bucket' or 'region' must be specified"
        raise ValueError(msg)
    key = key or _get_tmp_key(content_type)
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(bucket))
    kwargs: dict[str, str | dict[str, str]] = {"Tagging": S3_TAGGING}
    if content_type:
        kwargs["ContentType"] = content_type
    if content_disposition:
        kwargs["ContentDisposition"] = content_disposition
    if metadata is not None:
        kwargs["Metadata"] = metadata

    if isinstance(data, bytes):
        if len(data) < _COPY_OBJECT_MAX_BYTES:
            await s3.put_object(Bucket=bucket, Key=key, Body=data, **kwargs)  # type: ignore[arg-type]
        else:
            # Slicing skips `buffered_chunks`' re-chunking copies, and parts stay
            # real bytes -- the only buffer botocore's flexible-checksum body
            # wrapper accepts.
            await _multipart_upload(
                s3, bucket, key, _bytes_chunks(data, UPLOAD_CHUNK_SIZE), **kwargs
            )
        if temporary:
            track_temporary_s3_objects(bucket, key)
        return S3Object(bucket=bucket, key=key)

    sized = buffered_chunks(data, UPLOAD_CHUNK_SIZE)
    first = await anext(sized, b"")
    second = await anext(sized, b"")

    if not second:
        await s3.put_object(Bucket=bucket, Key=key, Body=first, **kwargs)  # type: ignore[arg-type]
    else:
        await _multipart_upload(
            s3,
            bucket,
            key,
            chain_async_iterators(async_iter(first, second), sized),
            **kwargs,
        )
    if temporary:
        track_temporary_s3_objects(bucket, key)
    return S3Object(bucket=bucket, key=key)


async def _bytes_chunks(data: bytes, chunk_size: int) -> AsyncIterator[bytes]:
    """Split *data* into slices, without ``buffered_chunks``' re-chunking.

    Args:
        data: Source bytes, already fully in memory.
        chunk_size: Size in bytes of each yielded slice, except possibly the last.

    Yields:
        ``bytes`` slices of *data*.
    """
    for start in range(0, len(data), chunk_size):
        yield data[start : start + chunk_size]


async def _multipart_upload(
    s3: S3Client,
    bucket: str,
    key: str,
    chunks: AsyncIterator[bytes],
    **kwargs: str | dict[str, str],
) -> None:
    """Perform a multipart upload from an async chunk iterator.

    Reading and uploading are pipelined: the next chunk is read while up to
    ``_UPLOAD_PARTS_IN_FLIGHT`` parts upload, bounding the extra memory to
    ``_UPLOAD_PARTS_IN_FLIGHT * UPLOAD_CHUNK_SIZE`` bytes per upload. Parts are
    collected in flight order, which keeps the completion list ordered by part
    number as CompleteMultipartUpload requires.

    Args:
        s3: S3 client.
        bucket: Destination bucket.
        key: Destination object key.
        chunks: Async iterator yielding ``bytes`` chunks (each at least
            ``UPLOAD_CHUNK_SIZE`` except possibly the last).
        **kwargs: Extra arguments forwarded to ``create_multipart_upload``
            (e.g. ``ContentType``).
    """
    upload_id: str | None = None
    in_flight: deque[Task[dict[str, int | str]]] = deque()
    try:
        upload_id = multipart_id = (
            await s3.create_multipart_upload(Bucket=bucket, Key=key, **kwargs)  # type: ignore[arg-type]
        )["UploadId"]

        async def _upload_part(part_number: int, body: bytes) -> dict[str, int | str]:
            """Upload one part and return its completion-list entry."""
            etag = (
                await s3.upload_part(
                    Bucket=bucket,
                    Key=key,
                    UploadId=multipart_id,
                    PartNumber=part_number,
                    Body=body,
                )
            )["ETag"]
            return {"PartNumber": part_number, "ETag": etag}

        parts: list[dict[str, int | str]] = []
        part_number = 0
        async for chunk in chunks:
            part_number += 1
            in_flight.append(create_task(_upload_part(part_number, chunk)))
            if len(in_flight) >= _UPLOAD_PARTS_IN_FLIGHT:
                parts.append(await in_flight.popleft())
        while in_flight:
            parts.append(await in_flight.popleft())
        await s3.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},  # type: ignore[typeddict-item]
        )
    except Exception:
        for task in in_flight:
            task.cancel()
        if in_flight:
            await gather(*in_flight, return_exceptions=True)
        if upload_id is not None:
            with contextlib.suppress(ClientError):
                await s3.abort_multipart_upload(
                    Bucket=bucket, Key=key, UploadId=upload_id
                )
        raise
