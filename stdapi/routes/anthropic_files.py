"""Anthropic-compatible Files API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Path, Query, UploadFile
from fastapi.responses import StreamingResponse

from stdapi.api_providers.anthropic import TAG_ANTHROPIC
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.files import (
    FileRecord,
    delete_file,
    get_file,
    get_file_content,
    list_files,
    upload_file,
)
from stdapi.input_file import InputFile
from stdapi.monitoring import log_request_params, log_response_params
from stdapi.types import FILE_ID_PATTERN
from stdapi.types.anthropic_files import DeletedFile, FileListResponse, FileMetadata


def _strip(fid: str) -> str:
    """Return the bare 32-char payload for *fid* by stripping the ``file-``/``file_`` prefix."""
    return fid[5:]


router = APIRouter(
    prefix=f"{SETTINGS.anthropic_routes_prefix}/v1", tags=["Files", TAG_ANTHROPIC]
)

#: Reusable path annotation for the ``file_id`` path parameter.
_FileId = Annotated[str, Path(description="ID of the File.", pattern=FILE_ID_PATTERN)]


def _to_file_metadata(record: FileRecord) -> FileMetadata:
    """Convert a ``_FileRecord`` to an Anthropic ``FileMetadata`` response.

    Args:
        record: Internal ``_FileRecord`` instance.

    Returns:
        Serialisable ``FileMetadata``.
    """
    return FileMetadata(
        id=f"file_{record.file_id}",
        filename=record.filename,
        mime_type=record.content_type,
        size_bytes=record.size,
        created_at=record.created_at.isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        downloadable=True,
    )


@router.post(
    "/files",
    summary="Upload a file for use in other API endpoints (Anthropic format)",
    operation_id="anthropic_file",
    description=(
        "Uploads a file and returns `FileMetadata` with the assigned file ID (Anthropic Files API).\n\n"
        "The returned `id` (format: `file_<32 hex chars>`) can be referenced in `anthropic_message` "
        "requests to supply documents or images without re-uploading them.\n\n"
        "**File expiry:** Files persist until manually deleted unless an expiry is configured."
    ),
    response_description="The file metadata.",
    response_model_exclude_none=True,
)
async def upload(
    file: Annotated[UploadFile, File(..., description="The file to upload.")],
    _: Annotated[None, Depends(authenticate)] = None,
) -> FileMetadata:
    """Upload a file.

    Returns:
        FileMetadata for the uploaded file.

    Raises:
        ApiError: If S3 is not configured.
    """
    log_request_params({"filename": file.filename})
    return log_response_params(_to_file_metadata(await upload_file(InputFile(file))))


@router.get(
    "/files",
    summary="List uploaded files (Anthropic format)",
    operation_id="anthropic_file_list",
    description="Returns a paginated list of uploaded files with metadata (Anthropic Files API).",
    response_description="A list of file metadata objects.",
    response_model_exclude_none=True,
)
async def list_files_endpoint(
    after_id: Annotated[
        str | None,
        Query(
            description=(
                "ID of the object to use as a cursor for pagination. "
                "When provided, returns the page of results immediately after this object."
            ),
            pattern=FILE_ID_PATTERN,
        ),
    ] = None,
    before_id: Annotated[
        str | None,
        Query(
            description=(
                "ID of the object to use as a cursor for pagination. "
                "When provided, returns the page of results immediately before this object."
            ),
            pattern=FILE_ID_PATTERN,
        ),
    ] = None,
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=1000,
            description="Number of items to return per page. Defaults to `20`. Ranges from `1` to `1000`.",
        ),
    ] = 20,
    _: Annotated[None, Depends(authenticate)] = None,
) -> FileListResponse:
    """List files with cursor-based pagination.

    Returns:
        FileListResponse with paginated file metadata.

    Raises:
        ApiError: If S3 is not configured.
    """
    log_request_params({"after_id": after_id, "before_id": before_id, "limit": limit})
    records, has_more = await list_files(
        _strip(after_id) if after_id else None,
        _strip(before_id) if before_id else None,
        limit,
        "asc",
        None,
    )
    files = [_to_file_metadata(r) for r in records]
    return log_response_params(
        FileListResponse(
            data=files,
            has_more=has_more,
            first_id=files[0].id if files else None,
            last_id=files[-1].id if files else None,
        )
    )


@router.get(
    "/files/{file_id}",
    summary="Retrieve metadata for an uploaded file (Anthropic format)",
    operation_id="anthropic_files_get",
    description="Returns metadata (name, size, MIME type, creation date) for a specific file by ID (Anthropic Files API).",
    response_description="The file metadata.",
    response_model_exclude_none=True,
)
async def retrieve_file(
    file_id: _FileId, _: Annotated[None, Depends(authenticate)] = None
) -> FileMetadata:
    """Retrieve metadata for a specific file.

    Args:
        file_id: Unique file identifier.

    Returns:
        FileMetadata with file details.

    Raises:
        FileNotExistError: If the file does not exist or has expired (404).
    """
    log_request_params({"file_id": file_id})
    return log_response_params(_to_file_metadata(await get_file(_strip(file_id))))


@router.delete(
    "/files/{file_id}",
    summary="Delete an uploaded file (Anthropic format)",
    operation_id="anthropic_files_delete",
    description="Permanently deletes a file by ID and returns a deletion confirmation (Anthropic Files API).",
    response_description="Deletion status.",
    response_model_exclude_none=True,
)
async def delete_file_endpoint(
    file_id: _FileId, _: Annotated[None, Depends(authenticate)] = None
) -> DeletedFile:
    """Delete a file by ID.

    Args:
        file_id: Unique file identifier.

    Returns:
        DeletedFile confirmation.

    Raises:
        FileNotExistError: If the file does not exist (404).
    """
    log_request_params({"file_id": file_id})
    payload = _strip(file_id)
    await delete_file(payload)
    return log_response_params(DeletedFile(id=f"file_{payload}"))


@router.get(
    "/files/{file_id}/content",
    summary="Download the raw content of an uploaded file (Anthropic format)",
    operation_id="anthropic_file_content",
    description="Returns the raw binary content of a file as a streaming download (Anthropic Files API).",
    response_description="The raw file content.",
)
async def get_content(
    file_id: _FileId, _: Annotated[None, Depends(authenticate)] = None
) -> StreamingResponse:
    """Stream the raw content of a file.

    Args:
        file_id: Unique file identifier.

    Returns:
        StreamingResponse with raw file bytes.

    Raises:
        FileNotExistError: If the file does not exist or has expired (404).
    """
    log_request_params({"file_id": file_id})
    stream, content_type = await get_file_content(_strip(file_id))
    return StreamingResponse(stream, media_type=content_type)
