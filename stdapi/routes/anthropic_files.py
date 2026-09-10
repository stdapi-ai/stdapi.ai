"""Anthropic-compatible Files API routes."""

from asyncio import gather
from contextlib import suppress
from typing import Annotated

from fastapi import APIRouter, Depends, File, Path, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import AfterValidator, StringConstraints

from stdapi.api_errors import ApiError, FileNotExistError
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

#: Content type is client-controlled: force download and disable content-type sniffing.
_CONTENT_DOWNLOAD_HEADERS = {
    "Content-Disposition": "attachment",
    "X-Content-Type-Options": "nosniff",
}


#: Most unique IDs one listing request may name, per the Anthropic Files API.
_MAX_LIST_IDS: int = 100


def _strip(fid: str) -> str:
    """Return the bare 32-char payload for *fid* by stripping the ``file-``/``file_`` prefix."""
    return fid[5:]


def _unique_ids(ids: list[str]) -> list[str]:
    """Return *ids* de-duplicated in the order given, refusing more than the cap.

    Both accepted prefixes name the same file, so IDs are compared on the payload
    they carry rather than as written.

    Args:
        ids: File IDs the listing request named.

    Returns:
        The unique IDs.

    Raises:
        ValueError: More unique IDs were named than one request may select.
    """
    unique = list({_strip(fid): fid for fid in ids}.values())
    if len(unique) > _MAX_LIST_IDS:
        msg = f"At most {_MAX_LIST_IDS} unique `ids` may be requested."
        raise ValueError(msg)
    return unique


#: A listing's ``ids`` filter: each entry a file ID, de-duplicated and capped.
_SelectedIds = Annotated[
    list[Annotated[str, StringConstraints(pattern=FILE_ID_PATTERN)]],
    AfterValidator(_unique_ids),
]

#: Query key the Anthropic client writes a list under: its serialiser uses brackets.
_IDS_BRACKET_KEY = "ids[]"


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


async def _visible_file(payload: str) -> FileRecord | None:
    """Return the record for *payload*, or ``None`` when it names no readable file.

    Args:
        payload: Bare 32-char file payload.

    Returns:
        The file's record, or ``None`` when it is unknown, deleted or expired.
    """
    with suppress(FileNotExistError):
        return await get_file(payload)
    return None


def _list_response(files: list[FileMetadata], *, has_more: bool) -> FileListResponse:
    """Wrap *files* in the listing envelope, reporting the page edges as cursors.

    Args:
        files: The page's file metadata, in the order it is served.
        has_more: Whether further pages follow this one.

    Returns:
        Serialisable ``FileListResponse``.
    """
    return FileListResponse(
        data=files,
        has_more=has_more,
        first_id=files[0].id if files else None,
        last_id=files[-1].id if files else None,
    )


@_router.get(
    "/files",
    summary="List uploaded files (Anthropic format)",
    operation_id="anthropic_file_list",
    description=(
        "Returns a paginated list of uploaded files with metadata, most recently "
        "created first (Anthropic Files API).\n\n"
        "Pass `ids` to fetch a known set of files in one call instead of paging "
        "through the whole list."
    ),
    response_description="A list of file metadata objects.",
    response_model_exclude_none=True,
)
async def list_files_endpoint(
    ids: Annotated[
        _SelectedIds | None,
        Query(
            description=(
                "Restrict the result to the files whose ID is in this list, at most "
                "100 after de-duplication. The whole selection is returned as a single "
                "page, so `after_id`, `before_id` and `limit` are ignored; IDs naming "
                "no readable file are omitted instead of reported."
            )
        ),
    ] = None,
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
    scope_id: Annotated[
        str | None,
        Query(
            description=(
                "Not available: files are not associated with a scope, so a request "
                "naming this parameter is refused."
            )
        ),
    ] = None,
    bracketed_ids: Annotated[
        _SelectedIds | None,
        Query(
            alias=_IDS_BRACKET_KEY,
            description=(
                "The same filter as `ids`, under the key the Anthropic client writes "
                "a list to. Sending both merges them."
            ),
        ),
    ] = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> FileListResponse:
    """List files, most recently created first, by ID selection or cursor pagination.

    Returns:
        FileListResponse with the selected or paginated file metadata.

    Raises:
        ApiError: If a scope filter is requested, or if S3 is not configured.
    """
    if bracketed_ids is not None:
        try:
            ids = _unique_ids((ids or []) + bracketed_ids)
        except ValueError as error:
            raise ApiError(str(error)) from error
    if scope_id is not None:
        msg = (
            "Filtering by `scope_id` is not available: files are not associated with "
            "a scope. Omit it to list files, or name the ones you want in `ids`."
        )
        raise ApiError(msg)
    if ids is not None:
        log_request_params({"ids": ids})
        selected = await gather(*(_visible_file(_strip(fid)) for fid in ids))
        return log_response_params(
            _list_response(
                sorted(
                    (_to_file_metadata(r) for r in selected if r is not None),
                    key=lambda metadata: metadata.id,
                    reverse=True,
                ),
                has_more=False,
            )
        )
    log_request_params({"after_id": after_id, "before_id": before_id, "limit": limit})
    records, has_more = await list_files(
        _strip(after_id) if after_id else None,
        _strip(before_id) if before_id else None,
        limit,
        "desc",
        None,
    )
    return log_response_params(
        _list_response([_to_file_metadata(r) for r in records], has_more=has_more)
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
