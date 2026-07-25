"""OpenAI-compatible Files API routes."""

from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, Path, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import AliasChoices

from stdapi.api_errors import ApiError
from stdapi.api_providers.openai import TAG_OPENAI
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
from stdapi.types.openai_files import (
    FileDeleted,
    FileObject,
    FilePurpose,
    FileUploadJsonBody,
    ListFilesResponse,
)
from stdapi.utils import missing_file_error, validation_error_handler

#: Minimum accepted value (seconds) for ``expires_after[seconds]`` (1 hour).
_EXPIRES_AFTER_SECONDS_MIN = 3600
#: Maximum accepted value (seconds) for ``expires_after[seconds]`` (30 days).
_EXPIRES_AFTER_SECONDS_MAX = 2592000


def _strip(fid: str) -> str:
    """Return the bare 32-char payload for *fid* by stripping the ``file-``/``file_`` prefix."""
    return fid[5:]


if TYPE_CHECKING:
    from enum import Enum

#: OpenAI files router tags
OPENAI_FILES_TAGS: list[str | Enum] = ["Files", TAG_OPENAI]

router = APIRouter(prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=OPENAI_FILES_TAGS)

#: Reusable path annotation for the ``file_id`` path parameter.
_FileId = Annotated[
    str,
    Path(
        description="The ID of the file to use for this request.",
        pattern=FILE_ID_PATTERN,
    ),
]


def _to_file_object(record: FileRecord) -> FileObject:
    """Convert a ``_FileRecord`` to an OpenAI ``FileObject`` response.

    Args:
        record: Internal ``_FileRecord`` instance.

    Returns:
        Serialisable ``FileObject``.
    """
    return FileObject(
        id=f"file-{record.file_id}",
        bytes=record.size,
        created_at=int(record.created_at.timestamp()),
        filename=record.filename,
        purpose=record.purpose or "user_data",
        status="processed",
        expires_at=record.expires_at,
    )


@router.post(
    "/files",
    summary="Upload a file for use in other API endpoints (OpenAI format)",
    operation_id="openai_file",
    description=(
        "Uploads a file and returns a `FileObject` with the assigned file ID (OpenAI Files API).\n\n"
        "The returned `file_id` (format: `file-<32 hex chars>`) can be referenced in other tools "
        "such as `openai_image_edit` and `openai_image_variation` to supply images without re-uploading.\n\n"
        "**Providing the file:** Two request formats are accepted:\n"
        "- `multipart/form-data`: standard binary file upload via the `file` field.\n"
        "- `application/json`: pass `file` as a base64 string, data URI "
        "(`data:<mime>;base64,<data>`), HTTPS URL, or S3 URI — preferred for **MCP tools** and "
        "AI agents that cannot construct multipart requests.\n\n"
        "**MCP / AI agent usage:** Call this tool with a JSON body. "
        "To upload inline content use a data URI: "
        '`{"file": "data:text/plain;base64,SGVsbG8h", "purpose": "user_data"}`. '
        "To ingest a remote file pass its URL: "
        '`{"file": "https://example.com/document.pdf", "purpose": "assistants"}`.\n\n'
        "**File expiry:** Files with `purpose=batch` expire after 30 days by default. "
        "All other files persist until manually deleted. "
        "Use `expires_after[seconds]` (1 hour-30 days) to set a custom TTL.\n\n"
        "For files larger than a few MB, prefer the multipart upload workflow: "
        "create a session with `openai_upload`, add parts with `openai_upload_part`, "
        "then finalise with `openai_upload_complete`."
    ),
    response_description="The uploaded File object.",
    response_model_exclude_none=True,
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "examples": {
                        "assistants": {
                            "summary": "Upload for Assistants API",
                            "value": {"purpose": "assistants"},
                        },
                        "batch": {
                            "summary": "Upload for Batch API (expires after 30 days)",
                            "value": {"purpose": "batch"},
                        },
                        "fine-tune": {
                            "summary": "Upload for fine-tuning",
                            "value": {"purpose": "fine-tune"},
                        },
                    }
                },
                "application/json": {
                    "examples": {
                        "data_uri": {
                            "summary": "Inline content via data URI (MCP / AI agent)",
                            "value": {
                                "file": "data:text/plain;base64,SGVsbG8gV29ybGQ=",
                                "purpose": "user_data",
                            },
                        },
                        "url": {
                            "summary": "Fetch from URL",
                            "value": {
                                "file": "https://example.com/document.pdf",
                                "purpose": "assistants",
                            },
                        },
                        "base64": {
                            "summary": "Raw base64",
                            "value": {
                                "file": "SGVsbG8gV29ybGQ=",
                                "purpose": "user_data",
                            },
                        },
                    }
                },
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
                "The File object (not file name) to be uploaded. "
                "Use an ``application/json`` body to pass a base64 string, data URI, or URL instead."
            )
        ),
    ] = None,
    purpose: Annotated[
        FilePurpose,
        Form(
            description=(
                "Intended purpose of the file: `assistants` (Assistants API), "
                "`batch` (Batch API), `fine-tune` (fine-tuning), `vision` "
                "(vision fine-tuning images), `user_data` (any purpose), or "
                "`evals` (eval datasets)."
            )
        ),
    ] = "assistants",
    expires_after_anchor: Annotated[  # noqa: ARG001
        Literal["created_at"] | None,
        Form(
            validation_alias=AliasChoices(
                "expires_after_anchor", "expires_after[anchor]"
            ),
            description="Anchor timestamp after which the expiration policy applies. Supported anchors: `created_at`.",
        ),
    ] = None,
    expires_after_seconds: Annotated[
        int | None,
        Form(
            validation_alias=AliasChoices(
                "expires_after_seconds", "expires_after[seconds]"
            ),
            ge=_EXPIRES_AFTER_SECONDS_MIN,
            le=_EXPIRES_AFTER_SECONDS_MAX,
            description=(
                "Seconds after the anchor time until the file expires (1 hour "
                "to 30 days). By default, `purpose=batch` files expire after "
                "30 days; all other files persist until manually deleted."
            ),
        ),
    ] = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> FileObject:
    """Upload a file.

    Accepts ``multipart/form-data`` (binary upload) or ``application/json``
    (base64, data URI, HTTPS URL, or S3 URI in the ``file`` field).

    Returns:
        FileObject with metadata for the uploaded file.

    Raises:
        ApiError: If the upload parameters are invalid or S3 is not configured.
    """
    if "application/json" in http_request.headers.get("content-type", ""):
        with validation_error_handler():
            body = FileUploadJsonBody.model_validate(await http_request.json())
        log_request_params({"purpose": body.purpose})
        return log_response_params(
            _to_file_object(
                await upload_file(body.file, body.purpose, body.expires_after_seconds)
            )
        )
    if file is None:
        missing_file_error()
    if expires_after_seconds is None and (
        raw := (await http_request.form()).get("expires_after[seconds]")
    ):
        try:
            expires_after_seconds = int(str(raw))
        except ValueError:
            msg = (
                "Input should be a valid integer, unable to parse string as an integer"
            )
            raise ApiError(msg) from None
        if expires_after_seconds < _EXPIRES_AFTER_SECONDS_MIN:
            msg = (
                f"Input should be greater than or equal to {_EXPIRES_AFTER_SECONDS_MIN}"
            )
            raise ApiError(msg)
        if expires_after_seconds > _EXPIRES_AFTER_SECONDS_MAX:
            msg = f"Input should be less than or equal to {_EXPIRES_AFTER_SECONDS_MAX}"
            raise ApiError(msg)
    log_request_params({"purpose": purpose, "filename": file.filename})
    return log_response_params(
        _to_file_object(
            await upload_file(InputFile(file), purpose, expires_after_seconds)
        )
    )


@router.get(
    "/files",
    summary="List uploaded files (OpenAI format)",
    operation_id="openai_file_list",
    description="Returns a paginated list of uploaded files, optionally filtered by purpose (OpenAI Files API).",
    response_description="A list of File objects.",
    response_model_exclude_none=True,
)
async def list_files_endpoint(
    purpose: Annotated[
        str | None, Query(description="Only return files with the given purpose.")
    ] = None,
    after: Annotated[
        str | None,
        Query(
            description=(
                "Cursor for pagination: the object ID to start after "
                "(the last ID from a previous page)."
            ),
            pattern=FILE_ID_PATTERN,
        ),
    ] = None,
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=10000,
            description="A limit on the number of objects to be returned.",
        ),
    ] = 10000,
    order: Annotated[
        Literal["asc", "desc"],
        Query(description="Sort order by the `created_at` timestamp of the objects."),
    ] = "desc",
    _: Annotated[None, Depends(authenticate)] = None,
) -> ListFilesResponse:
    """List files with cursor-based pagination.

    Returns:
        ListFilesResponse with paginated file objects.

    Raises:
        ApiError: If S3 is not configured.
    """
    log_request_params(
        {"purpose": purpose, "after": after, "limit": limit, "order": order}
    )
    records, has_more = await list_files(
        _strip(after) if after else None, None, limit, order, purpose
    )
    files = [_to_file_object(r) for r in records]
    return log_response_params(
        ListFilesResponse(
            data=files,
            has_more=has_more,
            first_id=files[0].id if files else "",
            last_id=files[-1].id if files else "",
        )
    )


@router.get(
    "/files/{file_id}",
    summary="Retrieve metadata for an uploaded file (OpenAI format)",
    operation_id="openai_files_get",
    description="Returns metadata (name, size, purpose, creation time) for a specific file by ID (OpenAI Files API).",
    response_description="The File object.",
    response_model_exclude_none=True,
)
async def retrieve_file(
    file_id: _FileId, _: Annotated[None, Depends(authenticate)] = None
) -> FileObject:
    """Retrieve metadata for a specific file.

    Args:
        file_id: Unique file identifier.

    Returns:
        FileObject with file metadata.

    Raises:
        FileNotExistError: If the file does not exist or has expired (404).
    """
    log_request_params({"file_id": file_id})
    return log_response_params(_to_file_object(await get_file(_strip(file_id))))


@router.delete(
    "/files/{file_id}",
    summary="Delete an uploaded file (OpenAI format)",
    operation_id="openai_files_delete",
    description="Permanently deletes a file by ID and returns a deletion confirmation (OpenAI Files API).",
    response_description="Deletion status.",
    response_model_exclude_none=True,
)
async def delete_file_endpoint(
    file_id: _FileId, _: Annotated[None, Depends(authenticate)] = None
) -> FileDeleted:
    """Delete a file by ID.

    Args:
        file_id: Unique file identifier.

    Returns:
        FileDeleted confirmation.

    Raises:
        FileNotExistError: If the file does not exist (404).
    """
    log_request_params({"file_id": file_id})
    payload = _strip(file_id)
    await delete_file(payload)
    return log_response_params(FileDeleted(id=f"file-{payload}", deleted=True))


@router.get(
    "/files/{file_id}/content",
    summary="Download the raw content of an uploaded file (OpenAI format)",
    operation_id="openai_file_content",
    description="Returns the raw binary content of a file as a streaming download (OpenAI Files API).",
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
