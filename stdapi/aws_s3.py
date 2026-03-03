"""AWS S3 utilities."""

import contextlib
from asyncio import gather
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from uuid import uuid4

from botocore.exceptions import ClientError

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.cleanup import schedule_cleanup
from stdapi.config import SETTINGS
from stdapi.monitoring import log_error_details
from stdapi.utils import async_iter, buffered_chunks, chain_async_iterators

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_s3.client import S3Client

#: Maximum object size supported by S3 ``copy_object`` in a single request (5 GiB).
_COPY_OBJECT_MAX_BYTES: Final[int] = 5 * 1024 * 1024 * 1024

#: Multipart copy chunk size in bytes (512 MiB).
_MULTIPART_COPY_PART_SIZE: Final[int] = 512 * 1024 * 1024

#: Multipart upload chunk size (8 MiB).
UPLOAD_CHUNK_SIZE: Final[int] = 8 * 1024 * 1024

#: Reverse bucket → region mapping (includes the default bucket).
BUCKET_TO_REGION: dict[str, RegionName] = {
    bucket_name: region
    for region, bucket_name in SETTINGS.aws_s3_regional_buckets.items()
}
if SETTINGS.aws_s3_bucket:
    BUCKET_TO_REGION.setdefault(SETTINGS.aws_s3_bucket, SETTINGS.aws_bedrock_regions[0])


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


async def _get_tmp_key(content_type: str | None = None) -> str:
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


async def _delete_s3_objects(s3_objects_to_delete: list[tuple[str, str]]) -> None:
    """Delete S3 temporary objects.

    The S3 client is resolved automatically from :data:`BUCKET_TO_REGION`.

    Args:
        s3_objects_to_delete: List of (bucket, key) tuples to delete.
    """
    await gather(
        *(
            get_client("s3", BUCKET_TO_REGION.get(bucket)).delete_object(
                Bucket=bucket, Key=key
            )
            for bucket, key in s3_objects_to_delete
        )
    )


async def put_object_and_get_url(body: bytes, content_type: str, filename: str) -> str:
    """Uploads an object to an AWS S3 bucket and retrieves the pre-signed URL to access it.

    This function asynchronously uploads the provided object to the specified S3 bucket and
    returns a pre-signed URL for accessing the uploaded object. The URL is valid for 3600 seconds.

    S3 prefix is added automatically.

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
    """Retrieve and decode S3 object content as a string.

    Args:
        s3_bucket: Name of the S3 bucket containing the object
        s3_key: Key (path) of the object within the S3 bucket

    Returns:
        Decoded string content of the S3 object
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
    dest_key = dest_key or await _get_tmp_key(content_type)
    copy_source = {"Bucket": source_bucket, "Key": source_key}
    if size <= _COPY_OBJECT_MAX_BYTES:
        await s3.copy_object(Bucket=dest_bucket, Key=dest_key, CopySource=copy_source)
    else:
        upload_id: str | None = None
        try:
            upload_id = (
                await s3.create_multipart_upload(Bucket=dest_bucket, Key=dest_key)
            )["UploadId"]

            parts: list[dict[str, int | str]] = []
            for part_number, start in enumerate(
                range(0, size, _MULTIPART_COPY_PART_SIZE), start=1
            ):
                end = min(start + _MULTIPART_COPY_PART_SIZE, size) - 1
                etag = (
                    await s3.upload_part_copy(
                        Bucket=dest_bucket,
                        Key=dest_key,
                        UploadId=upload_id,
                        PartNumber=part_number,
                        CopySource=copy_source,
                        CopySourceRange=f"bytes={start}-{end}",
                    )
                )["CopyPartResult"]["ETag"]
                parts.append({"PartNumber": part_number, "ETag": etag})

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

    Returns:
        An :class:`S3Object` referencing the uploaded object.
    """
    bucket = bucket or (require_s3_bucket_for_region(region) if region else None)
    if not bucket:
        msg = "Either 'bucket' or 'region' must be specified"
        raise ValueError(msg)
    key = key or await _get_tmp_key(content_type)
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(bucket))
    kwargs: dict[str, str] = {}
    if content_type:
        kwargs["ContentType"] = content_type

    if isinstance(data, bytes):
        if len(data) < _COPY_OBJECT_MAX_BYTES:
            await s3.put_object(Bucket=bucket, Key=key, Body=data, **kwargs)  # type: ignore[arg-type]
            if temporary:
                track_temporary_s3_objects(bucket, key)
            return S3Object(bucket=bucket, key=key)
        data = async_iter(data)

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


async def _multipart_upload(
    s3: S3Client, bucket: str, key: str, chunks: AsyncIterator[bytes], **kwargs: str
) -> None:
    """Perform a multipart upload from an async chunk iterator.

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
    try:
        upload_id = (
            await s3.create_multipart_upload(Bucket=bucket, Key=key, **kwargs)  # type: ignore[arg-type]
        )["UploadId"]
        parts: list[dict[str, int | str]] = []
        part_number = 0
        async for chunk in chunks:
            part_number += 1
            etag = (
                await s3.upload_part(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=chunk,
                )
            )["ETag"]
            parts.append({"PartNumber": part_number, "ETag": etag})
        await s3.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},  # type: ignore[typeddict-item]
        )
    except Exception:
        if upload_id is not None:
            with contextlib.suppress(ClientError):
                await s3.abort_multipart_upload(
                    Bucket=bucket, Key=key, UploadId=upload_id
                )
        raise
