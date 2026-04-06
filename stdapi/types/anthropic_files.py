"""Local Anthropic-compatible Files API types."""

from typing import Literal

from pydantic import Field

from stdapi.types import BaseModelResponse


# Ref: anthropic.types.beta.FileMetadata
class FileMetadata(BaseModelResponse):
    """The `FileMetadata` object represents a file that has been uploaded to the API."""

    id: str = Field(
        description="Unique object identifier. The format and length of IDs may change over time."
    )
    type: Literal["file"] = Field(
        default="file", description='Object type. For files, this is always `"file"`.'
    )
    filename: str = Field(description="Original filename of the uploaded file.")
    mime_type: str = Field(description="MIME type of the file.")
    size_bytes: int = Field(description="Size of the file in bytes.")
    created_at: str = Field(
        description="RFC 3339 datetime string representing when the file was created."
    )
    downloadable: bool = Field(
        default=True, description="Whether the file can be downloaded."
    )


# Ref: anthropic.types.beta.DeletedFile
class DeletedFile(BaseModelResponse):
    """Response returned when a file is deleted via the Files API."""

    id: str = Field(description="ID of the deleted file.")
    type: Literal["file_deleted"] = Field(
        default="file_deleted",
        description='Deleted object type. For file deletion, this is always `"file_deleted"`.',
    )


class FileListResponse(BaseModelResponse):
    """Paginated list of files returned by GET /v1/files."""

    data: list[FileMetadata] = Field(description="List of file metadata objects.")
    has_more: bool = Field(
        default=False, description="Whether there are more results available."
    )
    first_id: str | None = Field(
        default=None, description="ID of the first file in this page of results."
    )
    last_id: str | None = Field(
        default=None, description="ID of the last file in this page of results."
    )
