"""Local OpenAI-compatible Files API types."""

from typing import Literal

from pydantic import Field

from stdapi.types import BaseModelResponse

#: Valid values for the ``purpose`` field on a file upload or filter.
FilePurpose = Literal[
    "assistants", "batch", "fine-tune", "vision", "user_data", "evals"
]


# Ref: openai.types.file_object.FileObject
class FileObject(BaseModelResponse):
    """The `File` object represents a document that has been uploaded to the API."""

    id: str = Field(
        description="The file identifier, which can be referenced in the API endpoints."
    )
    object: Literal["file"] = Field(
        default="file", description="The object type, which is always `file`."
    )
    bytes: int = Field(description="The size of the file, in bytes.")
    created_at: int = Field(
        description="The Unix timestamp (in seconds) for when the file was created."
    )
    filename: str = Field(description="The name of the file.")
    purpose: str = Field(
        description=(
            "The intended purpose of the file. Supported values are `assistants`, `assistants_output`, "
            "`batch`, `batch_output`, `fine-tune`, `fine-tune-results`, `vision`, and `user_data`."
        )
    )
    status: Literal["uploaded", "processed", "error"] = Field(
        default="processed",
        description="Deprecated. The current status of the file, which can be either `uploaded`, `processed`, or `error`.",
    )
    expires_at: int | None = Field(
        default=None,
        description="The Unix timestamp (in seconds) for when the file will expire.",
    )
    status_details: str | None = Field(
        default=None,
        description="Deprecated. For details on why a fine-tuning training file failed validation, see the `error` field on `fine_tuning.job`.",
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
