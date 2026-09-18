"""Ollama-compatible model management endpoints.

Every model this server offers is already available, and none of them is stored
here, so this module holds the one verb whose post-condition can be met and the
five that would have to change a model store this server does not have, plus
the blob existence probe, which is mounted only so that the upload verb beside
it does not turn into a 405.

- POST   /api/pull — make a model available (already true, reports success)
- POST   /api/create — refused
- POST   /api/copy — refused
- POST   /api/push — refused
- DELETE /api/delete — refused
- POST   /api/blobs/{digest} — refused
- HEAD   /api/blobs/{digest} — never present, so always 404
"""

from typing import TYPE_CHECKING, Annotated, Never

from fastapi import APIRouter, Depends, Path, Request
from starlette.responses import Response, StreamingResponse

from stdapi.api_errors import NoModelStoreError
from stdapi.api_providers.ollama import NDJSON_MEDIA_TYPE, TAG_OLLAMA
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.monitoring import (
    guard_ndjson_stream_errors,
    log_request_params,
    log_response_params,
)
from stdapi.types.ollama import (
    CopyRequest,
    CreateRequest,
    DeleteRequest,
    PullRequest,
    PushRequest,
    StatusResponse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from stdapi.types import JsonMapping

router = APIRouter(
    prefix=f"{SETTINGS.ollama_routes_prefix}/api", tags=["Models", TAG_OLLAMA]
)

#: Status Ollama's model management operations report on completion.
_SUCCESS: str = "success"

#: Refusal shared by the verbs that would have to write to a model store.
_NO_MODEL_STORE: str = (
    "This server does not store models: the models it offers are hosted and "
    "already available, so they cannot be created, copied, published or "
    "deleted. Call the model list endpoint to see what is available."
)

#: Refusal of a blob upload, which only feeds the model creation this server refuses.
_NO_BLOB_STORE: str = (
    "This server does not store models: a blob is uploaded only to build a "
    "model from it, which this server cannot do, so there is nothing to upload "
    "it into. Call the model list endpoint to see what is available."
)


def _refuse(message: str = _NO_MODEL_STORE) -> Never:
    """Refuse an operation that would have to change a model store.

    The request is well-formed; the server will not perform it, because models
    live in Amazon Bedrock and cannot be written from here.

    Args:
        message: The refusal to report.

    Raises:
        NoModelStoreError: Always.
    """
    raise NoModelStoreError(message)


async def _pull_status() -> AsyncGenerator[JsonMapping]:
    """Report the outcome of a pull as a status stream.

    Yields:
        The single terminal status: nothing is transferred, so there is no
        progress to report and inventing one would be a fiction.
    """
    yield {"status": _SUCCESS}


@router.post(
    "/pull",
    summary="Make a model available (Ollama format)",
    operation_id="ollama_pull",
    description=(
        "Makes a model available for use (Ollama Pull API).\n\n"
        "Models are hosted and available as soon as they appear in "
        "`ollama_tags`, so nothing is transferred and the call reports success "
        "immediately; a model this server does not offer answers 404. `insecure` "
        "is accepted and ignored."
    ),
    response_description="The status of the operation.",
    response_model=StatusResponse,
    responses={
        200: {"description": "The model is available."},
        404: {"description": "Model not found."},
    },
    response_model_exclude_none=True,
)
async def pull(
    request: PullRequest, _: Annotated[None, Depends(authenticate)] = None
) -> StatusResponse | StreamingResponse:
    """Confirm that a model is available for use.

    Args:
        request: Pull parameters following the Ollama API.

    Returns:
        The success status, as a stream when ``stream`` is true.

    Raises:
        ApiError: With 404 if the model does not exist.
    """
    log_request_params(request)
    await validate_model(request.model, bedrock_only=False)
    if request.stream:
        return StreamingResponse(
            guard_ndjson_stream_errors(_pull_status()), media_type=NDJSON_MEDIA_TYPE
        )
    return log_response_params(StatusResponse(status=_SUCCESS))


@router.post(
    "/create",
    summary="Create a model (Ollama format)",
    operation_id="ollama_create",
    description=(
        "UNSUPPORTED (Ollama Create API). This server offers hosted models and "
        "stores none of its own, so a model cannot be created here."
    ),
    response_description="Always an error.",
    response_model=None,
    responses={403: {"description": "Models cannot be created on this server."}},
)
async def create(
    request: CreateRequest, _: Annotated[None, Depends(authenticate)] = None
) -> Never:
    """Refuse to create a model.

    Args:
        request: Create parameters following the Ollama API.

    Raises:
        NoModelStoreError: Always.
    """
    log_request_params(request)
    _refuse()


@router.post(
    "/copy",
    summary="Copy a model (Ollama format)",
    operation_id="ollama_copy",
    description=(
        "UNSUPPORTED (Ollama Copy API). This server offers hosted models and "
        "stores none of its own, so a model cannot be copied here."
    ),
    response_description="Always an error.",
    response_model=None,
    responses={403: {"description": "Models cannot be copied on this server."}},
)
async def copy(
    request: CopyRequest, _: Annotated[None, Depends(authenticate)] = None
) -> Never:
    """Refuse to copy a model.

    Args:
        request: Copy parameters following the Ollama API.

    Raises:
        NoModelStoreError: Always.
    """
    log_request_params(request)
    _refuse()


@router.post(
    "/push",
    summary="Publish a model (Ollama format)",
    operation_id="ollama_push",
    description=(
        "UNSUPPORTED (Ollama Push API). This server offers hosted models and "
        "publishes none, so a model cannot be pushed from here."
    ),
    response_description="Always an error.",
    response_model=None,
    responses={403: {"description": "Models cannot be published from this server."}},
)
async def push(
    request: PushRequest, _: Annotated[None, Depends(authenticate)] = None
) -> Never:
    """Refuse to publish a model.

    Args:
        request: Push parameters following the Ollama API.

    Raises:
        NoModelStoreError: Always.
    """
    log_request_params(request)
    _refuse()


@router.delete(
    "/delete",
    summary="Delete a model (Ollama format)",
    operation_id="ollama_delete",
    description=(
        "UNSUPPORTED (Ollama Delete API). This server offers hosted models and "
        "stores none of its own, so a model cannot be deleted here. Reporting "
        "success would tell the caller a model went away when it did not."
    ),
    response_description="Always an error.",
    response_model=None,
    responses={403: {"description": "Models cannot be deleted on this server."}},
)
async def delete(
    request: DeleteRequest, _: Annotated[None, Depends(authenticate)] = None
) -> Never:
    """Refuse to delete a model.

    Args:
        request: Delete parameters following the Ollama API.

    Raises:
        NoModelStoreError: Always.
    """
    log_request_params(request)
    _refuse()


async def _discard_body(http_request: Request) -> None:
    """Read a request body to its end without holding any of it.

    The refusal is the answer to a client that may already be streaming a
    multi-gigabyte file: reading the body out in chunks and dropping each one
    keeps the allocation constant, and letting it finish is what makes the
    ``403`` arrive as a response rather than as a broken pipe.

    Args:
        http_request: The incoming request.
    """
    async for _chunk in http_request.stream():
        pass


@router.post(
    "/blobs/{digest}",
    summary="Push a blob (Ollama format)",
    operation_id="ollama_blob_push",
    description=(
        "UNSUPPORTED (Ollama Push a Blob API). A blob is uploaded only to build "
        "a model from it, which this server offers no way to do, so there is "
        "nothing to upload it into."
    ),
    response_description="Always an error.",
    response_model=None,
    responses={403: {"description": "Blobs cannot be stored on this server."}},
)
async def push_blob(
    http_request: Request,
    digest: Annotated[str, Path(description="Digest of the blob, as `sha256:<hex>`.")],
    _: Annotated[None, Depends(authenticate)] = None,
) -> Never:
    """Refuse to store a blob.

    Args:
        http_request: The incoming request, whose body is drained and dropped.
        digest: Digest the client named the blob by.

    Raises:
        NoModelStoreError: Always.
    """
    log_request_params({"digest": digest})
    await _discard_body(http_request)
    _refuse(_NO_BLOB_STORE)


# Mounted only so that the upload verb above does not turn an absent blob into
# a 405. No blob is ever present here, which is the answer Ollama gives for one.
@router.head("/blobs/{digest}", include_in_schema=False)
async def head_blob(_: Annotated[None, Depends(authenticate)] = None) -> Response:
    """Report that a blob is not present, which is always true here.

    The digest is left undeclared: nothing is looked up under it.

    Returns:
        An empty 404 response.
    """
    return Response(status_code=404)
