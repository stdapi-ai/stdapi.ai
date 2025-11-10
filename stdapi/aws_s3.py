"""AWS S3 utilities."""

from asyncio import gather
from typing import TYPE_CHECKING

from stdapi.aws import get_client
from stdapi.config import SETTINGS
from stdapi.monitoring import log_background_event, log_error_details
from stdapi.openai_exceptions import OpenaiError

if TYPE_CHECKING:
    from types_aiobotocore_s3.client import S3Client


async def aws_s3_cleanup(
    s3_client: "S3Client", s3_objects_to_delete: list[tuple[str, str]], request_id: str
) -> None:
    """Cleanup tasks for S3 temporary resources.

    To execute with FastAPI BackgroundTasks.

    Args:
        s3_client: S3 client
        s3_objects_to_delete: List of (bucket, key) tuples to delete
        request_id: Request ID
    """
    with log_background_event("aws_s3_cleanup", request_id):
        await gather(
            *(
                s3_client.delete_object(Bucket=bucket, Key=key)
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
        raise OpenaiError(msg)

    s3_client: S3Client = get_client("s3")
    s3_accelerate_client: S3Client = get_client("s3.accelerate")
    s3_key = f"{SETTINGS.aws_s3_tmp_prefix}{filename}"
    return (
        await gather(
            s3_accelerate_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": s3_bucket, "Key": s3_key},
                ExpiresIn=3600,
            ),
            s3_client.put_object(
                Bucket=s3_bucket, Key=s3_key, Body=body, ContentType=content_type
            ),
        )
    )[0]
