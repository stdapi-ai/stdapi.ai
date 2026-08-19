"""Anthropic-compatible Files API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Path, Query, Request, UploadFile
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
from stdapi.types.anthropic_files import (
    AnthropicFileUploadJsonBody,
    DeletedFile,
    FileListResponse,
    FileMetadata,
)
from stdapi.utils import missing_file_error, validation_error_handler

#: Download hardening: the stored content type is client-controlled, so the
#: response must never be rendered inline nor content-type-sniffed by a browser.
_CONTENT_DOWNLOAD_HEADERS = {
    "Content-Disposition": "attachment",
    "X-Content-Type-Options": "nosniff",
}


def _strip(fid: str) -> str:
    """Return the bare 32-char payload for *fid* by stripping the ``file-``/``file_`` prefix."""
    return fid[5:]


_router = APIRouter(
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


@_router.post(
    "/files",
    summary="Upload a file for use in other API endpoints (Anthropic format)",
    operation_id="anthropic_file",
    description=(
        "Uploads a file and returns `FileMetadata` with the assigned file ID (Anthropic Files API).\n\n"
        "The returned `id` (format: `file_<32 hex chars>`) can be referenced in `anthropic_message` "
        "requests to supply documents or images without re-uploading them.\n\n"
        "**Providing the file:** Two request formats are accepted:\n"
        "- `multipart/form-data`: standard binary file upload via the `file` field.\n"
        "- `application/json`: pass `file` as a base64 string, data URI "
        "(`data:<mime>;base64,<data>`), HTTPS URL, or S3 URI — preferred for **MCP tools** and "
        "AI agents that cannot construct multipart requests.\n\n"
        "**MCP / AI agent usage:** Call this tool with a JSON body. "
        "To upload inline content use a data URI: "
        '`{"file": "data:text/plain;base64,SGVsbG8h"}`. '
        "To ingest a remote file pass its URL: "
        '`{"file": "https://example.com/document.pdf"}`.\n\n'
        "**File expiry:** Files persist until manually deleted unless an expiry is configured."
    ),
    response_description="The file metadata.",
    response_model_exclude_none=True,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "data_uri": {
                            "summary": "Inline content via data URI (MCP / AI agent)",
                            "value": {
                                "file": "data:text/plain;base64,SGVsbG8gV29ybGQ="
                            },
                        },
                        "url": {
                            "summary": "Fetch from URL",
                            "value": {"file": "https://example.com/document.pdf"},
                        },
                    }
                }
            }
        }
    },
)
async def upload(
    http_request: Request,
    file: Annotated[
        UploadFile | None,
        File(
            description=(
                "The file to upload. "
                "Use an ``application/json`` body to pass a base64 string, data URI, or URL instead."
            )
        ),
    ] = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> FileMetadata:
    """Upload a file.

    Accepts ``multipart/form-data`` (binary upload) or ``application/json``
    (base64, data URI, HTTPS URL, or S3 URI in the ``file`` field).

    Returns:
        FileMetadata for the uploaded file.

    Raises:
        ApiError: If S3 is not configured.
    """
    if "application/json" in http_request.headers.get("content-type", ""):
        with validation_error_handler():
            body = AnthropicFileUploadJsonBody.model_validate_json(
                await http_request.body()
            )
        return log_response_params(_to_file_metadata(await upload_file(body.file)))
    if file is None:
        missing_file_error()
    log_request_params({"filename": file.filename})
    return log_response_params(_to_file_metadata(await upload_file(InputFile(file))))


@_router.get(
    "/files",
    summary="List uploaded files (Anthropic format)",
    operation_id="anthropic_file_list",
    description=(
        "Returns a paginated list of uploaded files with metadata, most recently "
        "created first (Anthropic Files API)."
    ),
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
    """List files with cursor-based pagination, most recently created first.

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
        "desc",
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


@_router.get(
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


@_router.delete(
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


@_router.get(
    "/files/{file_id}/content",
    summary="Download the raw content of an uploaded file (Anthropic format)",
    operation_id="anthropic_file_content",
    description=(
        "Returns the raw binary content of a file as a streaming download "
        "(Anthropic Files API).\n\n"
        "**MCP / AI agent usage:** text files come back as text and images as "
        "an image; any other content, and anything too large to carry, comes "
        "back as a JSON reference holding the URL to download it from."
    ),
    response_description="The raw file content.",
)
async def get_content(
    file_id: _FileId, _: Annotated[None, Depends(authenticate)] = None
) -> StreamingResponse:
    """Stream the raw content of a file.

    Args:
        file_id: Unique file identifier.

    Returns:
        StreamingResponse with raw file bytes, served as a non-sniffable download.

    Raises:
        FileNotExistError: If the file does not exist or has expired (404).
    """
    log_request_params({"file_id": file_id})
    stream, content_type = await get_file_content(_strip(file_id))
    return StreamingResponse(
        stream, media_type=content_type, headers=_CONTENT_DOWNLOAD_HEADERS
    )


#: Disabled when sharing the OpenAI base path: the /v1/files paths would collide.
router: APIRouter | None = (
    None
    if SETTINGS.anthropic_routes_prefix == SETTINGS.openai_routes_prefix
    else _router
)
