"""Local OpenAI-compatible Files API types."""

from typing import Literal

from pydantic import Field

from stdapi.input_file import IngestInputFile  # noqa: TC001
from stdapi.types import BaseModelRequest, BaseModelResponse

#: Valid values for the ``purpose`` field on a file upload or filter.
FilePurpose = Literal[
    "assistants", "batch", "fine-tune", "vision", "user_data", "evals"
]

#: Description shared by all JSON-body ``file`` fields.
_FILE_FIELD_DESCRIPTION = (
    "The file content as a base64 string, data URI (``data:<mime>;base64,<data>``), "
    "HTTPS URL, or S3 URI (``s3://bucket/key``). "
    "The server auto-detects the encoding and MIME type."
)


class FileUploadJsonBody(BaseModelRequest):
    """Request body for file upload via ``application/json``.

    Alternative to ``multipart/form-data`` for MCP tools and HTTP clients that
    cannot construct multipart requests. The ``file`` field accepts a base64
    string, a data URI, an HTTPS URL, or an S3 URI — the same sources that
    ``InputFile`` accepts from strings.
    """

    file: IngestInputFile = Field(description=_FILE_FIELD_DESCRIPTION)
    purpose: FilePurpose = Field(
        default="assistants",
        description=(
            "Intended purpose of the file: `assistants` (Assistants API), "
            "`batch` (Batch API), `fine-tune` (fine-tuning), `vision` (vision "
            "fine-tuning images), `user_data` (any purpose), or `evals` "
            "(eval datasets)."
        ),
    )
    expires_after_anchor: Literal["created_at"] | None = Field(
        default=None,
        description="Anchor timestamp after which the expiration policy applies. Supported anchors: `created_at`.",
    )
    expires_after_seconds: int | None = Field(
        default=None,
        ge=3600,
        le=2592000,
        description=(
            "Seconds after the anchor time until the file expires (1 hour to "
            "30 days). By default, `purpose=batch` files expire after 30 days; "
            "all other files persist until manually deleted."
        ),
    )


# Ref: openai.types.file_object.FileObject
class FileObject(BaseModelResponse):
    """The `File` object represents a document that has been uploaded to the API."""

    id: str = Field(
        description="The file identifier, which can be referenced in the API endpoints."
    )
    object: Literal["file"] = Field(
        default="file", description="The object type, which is always `file`."
    )
    bytes: int = Field(description="The file size in bytes.")
    created_at: int = Field(
        description="Unix timestamp (in seconds) when the file was created."
    )
    filename: str = Field(description="The name of the file.")
    purpose: str = Field(description="The intended purpose of the file.")
    status: Literal["uploaded", "processed", "error"] = Field(
        default="processed", description="Deprecated. The current status of the file."
    )
    expires_at: int | None = Field(
        default=None,
        description="The Unix timestamp (in seconds) for when the file will expire.",
    )
    status_details: str | None = Field(
        default=None,
        description="Deprecated. Details on why a fine-tuning training file failed validation.",
    )


# Ref: openai.types.file_deleted.FileDeleted
class FileDeleted(BaseModelResponse):
    """Response returned when a file is deleted."""

    id: str = Field(description="The file identifier.")
    object: Literal["file"] = Field(
        default="file", description="The object type, which is always `file`."
    )
    deleted: bool = Field(description="Whether the file was deleted.")


# Ref: openai.types.list_files_response.ListFilesResponse
class ListFilesResponse(BaseModelResponse):
    """Paginated list of files returned by GET /v1/files."""

    object: Literal["list"] = Field(
        default="list", description="The object type, which is always `list`."
    )
    data: list[FileObject] = Field(description="List of File objects.")
    has_more: bool = Field(description="Whether more results exist after this page.")
    first_id: str = Field(
        description="ID of the first file in the list, or '' when empty."
    )
    last_id: str = Field(
        description="ID of the last file in the list, or '' when empty."
    )
