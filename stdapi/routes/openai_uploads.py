"""OpenAI-compatible Uploads API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Path, UploadFile

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
from stdapi.routes.openai_files import OPENAI_FILES_TAGS, _to_file_object
from stdapi.types import UPLOAD_ID_PATTERN
from stdapi.types.openai_files import FileObject  # noqa: TC001
from stdapi.types.openai_uploads import (
    CompleteUploadBody,
    CreateUploadBody,
    Upload,
    UploadPart,
)

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
    summary="OpenAI - POST /v1/uploads",
    operation_id="openai_upload",
    description=(
        "Creates an intermediate Upload object that you can add Parts to.\n\n"
        "Once you complete the Upload, we will create a File object that contains all the parts you uploaded."
        "This File is usable in the rest of our platform as a regular File object."
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
        }
    )
    return log_response_params(
        _to_upload(
            await create_multipart_session(
                body.filename, body.mime_type, body.purpose, body.bytes
            ),
            "pending",
        )
    )


@router.post(
    "/uploads/{upload_id}/parts",
    summary="OpenAI - POST /v1/uploads/{upload_id}/parts",
    operation_id="openai_upload_part",
    description="Adds a Part to an Upload object.",
    response_description="The upload Part object.",
    response_model_exclude_none=True,
)
async def add_upload_part(
    upload_id: _UploadId,
    data: Annotated[
        UploadFile, File(..., description="The chunk of bytes for this Part.")
    ],
    _: Annotated[None, Depends(authenticate)] = None,
) -> UploadPart:
    """Upload a part (chunk of bytes) to an existing upload session.

    Args:
        upload_id: Upload session identifier.
        data: File chunk to upload.

    Returns:
        UploadPart object for the uploaded chunk.

    Raises:
        ApiError: If the upload is not found (404) or not pending (400).
    """
    log_request_params({"upload_id": upload_id})
    part_id, created_at = await add_part(upload_id, await data.read())
    return log_response_params(
        UploadPart(id=part_id, created_at=created_at, upload_id=upload_id)
    )


@router.post(
    "/uploads/{upload_id}/complete",
    summary="OpenAI - POST /v1/uploads/{upload_id}/complete",
    operation_id="openai_upload_complete",
    description="Completes the Upload. Only call this when all parts have been uploaded.",
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
        body: Ordered list of part IDs.

    Returns:
        Upload object with ``status="completed"`` and the ``file`` field populated.

    Raises:
        ApiError: If the upload is not found (404), not pending (400), a part ID
            is unknown (400), or the assembled size does not match (400).
    """
    log_request_params({"upload_id": upload_id, "part_ids": body.part_ids})
    session, file_record = await complete_multipart_session(upload_id, body.part_ids)
    return log_response_params(
        _to_upload(session, "completed", _to_file_object(file_record))
    )


@router.post(
    "/uploads/{upload_id}/cancel",
    summary="OpenAI - POST /v1/uploads/{upload_id}/cancel",
    operation_id="openai_upload_cancel",
    description="Cancels the Upload. No Parts may be added after an Upload is cancelled.",
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
