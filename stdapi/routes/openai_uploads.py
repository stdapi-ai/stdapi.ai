"""OpenAI-compatible Uploads API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Path, Request, UploadFile

from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.files import (
    MultipartSession,
    add_part,
    cancel_multipart_session,
    complete_multipart_session,
    create_multipart_session,
)
from stdapi.monitoring import log_request_params, log_response_params
from stdapi.routes.openai_files import (
    OPENAI_FILES_TAGS,
    _resolve_expires_after_seconds,
    _to_file_object,
)
from stdapi.types import UPLOAD_ID_PATTERN
from stdapi.types.openai_files import FileObject  # noqa: TC001
from stdapi.types.openai_uploads import (
    AddUploadPartJsonBody,
    CompleteUploadBody,
    CreateUploadBody,
    Upload,
    UploadPart,
)
from stdapi.utils import missing_file_error, validation_error_handler

router = APIRouter(prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=OPENAI_FILES_TAGS)

#: Reusable path annotation for the ``upload_id`` path parameter.
_UploadId = Annotated[
    str, Path(description="The ID of the upload.", pattern=UPLOAD_ID_PATTERN)
]


def _to_upload(
    session: MultipartSession, status: str, file: FileObject | None = None
) -> Upload:
    """Convert *session* to an OpenAI ``Upload`` response with *status*.

    Args:
        session: Internal multipart session.
        status: ``"pending"``, ``"completed"``, or ``"cancelled"``.
        file: Completed file object, if any.
    """
    return Upload(
        id=session.upload_id,
        bytes=session.total_bytes,
        created_at=session.created_at,
        expires_at=session.expires_at,
        filename=session.filename,
        purpose=session.purpose,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        file=file,
    )


@router.post(
    "/uploads",
    summary="Create a multipart upload session for large files (OpenAI format)",
    operation_id="openai_upload",
    description=(
        "Creates a multipart upload session for uploading large files in chunks (OpenAI Uploads API).\n\n"
        "**Multipart upload workflow:**\n"
        "1. Call `openai_upload` to create a session and get an `upload_id`.\n"
        "2. Upload file chunks with `openai_upload_part` (one call per chunk).\n"
        "3. Call `openai_upload_complete` with the ordered list of part IDs to assemble the final file.\n"
        "4. Optionally call `openai_upload_cancel` to abort a pending session.\n\n"
        "**File expiry:** Files with `purpose=batch` expire after 30 days by default. "
        "All other files persist until manually deleted. "
        "Use `expires_after.seconds` (1 hour-30 days) to set a custom TTL.\n\n"
        "For small files, use `openai_file` instead — it uploads in a single request."
    ),
    response_description="The Upload object.",
    response_model_exclude_none=True,
)
async def create_upload_endpoint(
    body: CreateUploadBody, _: Annotated[None, Depends(authenticate)] = None
) -> Upload:
    """Create a new multipart upload session.

    Returns:
        Upload object in ``pending`` status.

    Raises:
        ApiError: If S3 is not configured (503).
    """
    log_request_params(
        {
            "filename": body.filename,
            "mime_type": body.mime_type,
            "purpose": body.purpose,
            "expires_after": body.expires_after,
        }
    )
    return log_response_params(
        _to_upload(
            await create_multipart_session(
                body.filename,
                body.mime_type,
                body.purpose,
                body.bytes,
                _resolve_expires_after_seconds(
                    body.purpose,
                    body.expires_after.seconds if body.expires_after else None,
                ),
            ),
            "pending",
        )
    )


@router.post(
    "/uploads/{upload_id}/parts",
    summary="Upload a chunk in a multipart upload session (OpenAI format)",
    operation_id="openai_upload_part",
    description=(
        "Adds a binary chunk (Part) to an existing multipart upload session (OpenAI Uploads API).\n\n"
        "**Prerequisite:** Create an upload session first with `openai_upload`. "
        "Call this endpoint once per chunk, then finalise with `openai_upload_complete`.\n\n"
        "**MCP / AI agent usage:** Pass the chunk as a JSON body with ``data`` set to a base64 string, "
        "data URI (``data:<mime>;base64,<data>``), HTTPS URL, or S3 URI."
    ),
    response_description="The upload Part object.",
    response_model_exclude_none=True,
)
async def add_upload_part(
    http_request: Request,
    upload_id: _UploadId,
    data: Annotated[
        UploadFile | None, File(description="The chunk of bytes for this Part.")
    ] = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> UploadPart:
    """Upload a part (chunk of bytes) to an existing upload session.

    Accepts ``multipart/form-data`` (binary upload) or ``application/json``
    (base64, data URI, HTTPS URL, or S3 URI in the ``data`` field).

    Args:
        http_request: FastAPI request object used to detect content-type.
        upload_id: Upload session identifier.
        data: File chunk to upload (multipart only).

    Returns:
        UploadPart object for the uploaded chunk.

    Raises:
        ApiError: If the upload is not found (404) or not pending (400).
    """
    log_request_params({"upload_id": upload_id})
    if "application/json" in http_request.headers.get("content-type", ""):
        with validation_error_handler():
            body = AddUploadPartJsonBody.model_validate_json(await http_request.body())
        chunk = await body.data.to_bytes()
    elif data is None:
        missing_file_error()
    else:
        chunk = await data.read()
    part_id, created_at = await add_part(upload_id, chunk)
    return log_response_params(
        UploadPart(id=part_id, created_at=created_at, upload_id=upload_id)
    )


@router.post(
    "/uploads/{upload_id}/complete",
    summary="Complete a multipart upload and create the final file (OpenAI format)",
    operation_id="openai_upload_complete",
    description=(
        "Assembles all uploaded parts into a final `File` object and marks the session as completed (OpenAI Uploads API).\n\n"
        "**Prerequisite:** All parts must have been uploaded via `openai_upload_part`. "
        "Provide the ordered list of part IDs returned by those calls. "
        "The resulting file behaves like a file uploaded with `openai_file`.\n\n"
        "**Integrity:** pass `md5` to have the assembled file checked against the "
        "digest of the bytes you sent; a mismatch is refused and leaves no file behind."
    ),
    response_description="The completed Upload object with a nested File object.",
    response_model_exclude_none=True,
)
async def complete_upload_endpoint(
    upload_id: _UploadId,
    body: CompleteUploadBody,
    _: Annotated[None, Depends(authenticate)] = None,
) -> Upload:
    """Complete a multipart upload and produce the final file.

    Args:
        upload_id: Upload session identifier.
        body: Ordered list of part IDs and the optional content checksum.

    Returns:
        Upload object with ``status="completed"`` and the ``file`` field populated.

    Raises:
        ApiError: If the upload is not found (404), not pending (400), a part ID
            is unknown (400), the assembled size does not match (400), or the
            contents do not match the declared ``md5`` (400).
    """
    log_request_params({"upload_id": upload_id, "part_ids": body.part_ids})
    session, file_record = await complete_multipart_session(
        upload_id, body.part_ids, body.md5
    )
    return log_response_params(
        _to_upload(session, "completed", _to_file_object(file_record))
    )


@router.post(
    "/uploads/{upload_id}/cancel",
    summary="Cancel a pending multipart upload session (OpenAI format)",
    operation_id="openai_upload_cancel",
    description=(
        "Cancels a pending multipart upload session; no further parts can be added (OpenAI Uploads API).\n\n"
        "**Prerequisite:** The session must have been created with `openai_upload` and not yet completed or cancelled."
    ),
    response_description="The cancelled Upload object.",
    response_model_exclude_none=True,
)
async def cancel_upload_endpoint(
    upload_id: _UploadId, _: Annotated[None, Depends(authenticate)] = None
) -> Upload:
    """Cancel a pending upload session.

    Args:
        upload_id: Upload session identifier.

    Returns:
        Upload object with ``status="cancelled"``.

    Raises:
        ApiError: If the upload is not found (404) or not pending (400).
    """
    log_request_params({"upload_id": upload_id})
    return log_response_params(
        _to_upload(await cancel_multipart_session(upload_id), "cancelled")
    )
