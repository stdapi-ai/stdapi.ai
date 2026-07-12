"""Local OpenAI-compatible Uploads API types."""

from typing import Literal

from pydantic import Field

from stdapi.input_file import IngestInputFile
from stdapi.types import (
    PART_ID_PATTERN,
    UPLOAD_ID_PATTERN,
    BaseModelRequest,
    BaseModelResponse,
)
from stdapi.types.openai_files import FileObject, FilePurpose

#: Valid status values for an in-progress or finished upload.
UploadStatus = Literal["pending", "completed", "cancelled", "expired"]


class UploadExpiresAfter(BaseModelRequest):
    """Expiration policy applied to the file created by an upload."""

    anchor: Literal["created_at"] = Field(
        description="Anchor timestamp after which the expiration policy "
        "applies. Supported anchors: `created_at`."
    )
    seconds: int = Field(
        ge=3600,
        le=2592000,
        description="Seconds after the anchor time until the file expires "
        "(1 hour to 30 days).",
    )


class CreateUploadBody(BaseModelRequest):
    """Request body for ``POST /v1/uploads``."""

    bytes: int = Field(
        gt=0,
        le=8 * 1024 * 1024 * 1024,
        description="The number of bytes in the file you are uploading.",
    )
    filename: str = Field(description="The name of the file to upload.")
    mime_type: str = Field(description="The MIME type of the file.")
    purpose: FilePurpose = Field(
        description="The intended purpose of the uploaded file."
    )
    expires_after: UploadExpiresAfter | None = Field(
        default=None,
        description="Expiration policy for the resulting file. By default the "
        "file persists until manually deleted.",
    )


class AddUploadPartJsonBody(BaseModelRequest):
    """Request body for ``POST /v1/uploads/{upload_id}/parts`` via ``application/json``.

    Alternative to ``multipart/form-data`` for MCP tools and HTTP clients that
    cannot construct multipart requests.  The ``data`` field accepts a base64
    string, a data URI, an HTTPS URL, or an S3 URI.
    """

    data: IngestInputFile = Field(
        description=(
            "The chunk of bytes as a base64 string, data URI "
            "(``data:<mime>;base64,<data>``), HTTPS URL, or S3 URI (``s3://bucket/key``)."
        )
    )


class CompleteUploadBody(BaseModelRequest):
    """Request body for ``POST /v1/uploads/{upload_id}/complete``."""

    part_ids: list[str] = Field(description="The ordered list of Part IDs.")
    md5: str | None = Field(
        default=None,
        description="Optional md5 checksum for the file contents. Accepted but not validated.",
    )


# Ref: openai.types.uploads.upload_part.UploadPart
class UploadPart(BaseModelResponse):
    """A part of a multipart upload."""

    id: str = Field(
        description="The upload Part unique identifier.", pattern=PART_ID_PATTERN
    )
    object: Literal["upload.part"] = Field(
        default="upload.part",
        description="The object type, which is always `upload.part`.",
    )
    created_at: int = Field(
        description="The Unix timestamp (in seconds) for when the Part was created."
    )
    upload_id: str = Field(
        description="The ID of the Upload object that this Part was added to.",
        pattern=UPLOAD_ID_PATTERN,
    )


# Ref: openai.types.uploads.upload.Upload
class Upload(BaseModelResponse):
    """The Upload object can accept byte chunks in the form of Parts."""

    id: str = Field(
        description="The Upload unique identifier.", pattern=UPLOAD_ID_PATTERN
    )
    object: Literal["upload"] = Field(
        default="upload", description="The object type, which is always 'upload'."
    )
    bytes: int = Field(description="The intended number of bytes to be uploaded.")
    created_at: int = Field(
        description="The Unix timestamp (in seconds) for when the Upload was created."
    )
    expires_at: int = Field(
        description="The Unix timestamp (in seconds) for when the Upload will expire."
    )
    filename: str = Field(description="The name of the file to be uploaded.")
    purpose: FilePurpose = Field(description="The intended purpose of the file.")
    status: UploadStatus = Field(description="The status of the Upload.")
    file: FileObject | None = Field(
        default=None, description="The ready File object after the Upload is completed."
    )
