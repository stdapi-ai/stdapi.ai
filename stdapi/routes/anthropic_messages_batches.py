"""Anthropic-compatible ``/v1/messages/batches`` endpoints.

A Message Batch runs many Messages API requests asynchronously, at the
discounted batch price. The requests are sent inline, the batch is polled until
it ends, and its results are streamed back as JSONL.
"""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Path, Query, Request
from fastapi.responses import StreamingResponse

from stdapi.api_errors import ApiError
from stdapi.api_providers.anthropic import TAG_ANTHROPIC
from stdapi.auth import authenticate
from stdapi.batches import (
    MAX_MODELS_PER_BATCH,
    MIN_REQUESTS_PER_MODEL,
    BatchState,
    cancel_batch,
    create_batch,
    delete_batch,
    get_batch,
    iter_anthropic_results,
    list_batches,
    prepare_anthropic_requests,
    rfc3339,
    settle,
)
from stdapi.config import SETTINGS
from stdapi.monitoring import log_request_params, log_response_params
from stdapi.types.anthropic_batches import (
    MESSAGE_BATCH_ID_PATTERN,
    DeletedMessageBatch,
    MessageBatch,
    MessageBatchCreateParams,
    MessageBatchList,
    MessageBatchRequestCounts,
    MessageBatchStatus,
)

if TYPE_CHECKING:
    from stdapi.batches import BatchRecord

router = APIRouter(
    prefix=f"{SETTINGS.anthropic_routes_prefix}/v1", tags=["Batches", TAG_ANTHROPIC]
)

#: Endpoint the batched requests are run against.
_ENDPOINT = "/v1/messages"

#: Reusable path annotation for the ``message_batch_id`` path parameter.
_BatchId = Annotated[
    str, Path(description="ID of the Message Batch.", pattern=MESSAGE_BATCH_ID_PATTERN)
]


def _results_url(http_request: Request, batch_id: str) -> str:
    """Return the address the batch's results are readable at.

    Args:
        http_request: The incoming request, whose origin the URL is built on.
        batch_id: Prefixed Message Batch identifier.

    Returns:
        The absolute URL, on the origin the client dialled — the forwarded
        origin behind a reverse proxy — since a client fetches it verbatim.
    """
    origin = str(http_request.base_url).rstrip("/")
    return (
        f"{origin}{SETTINGS.anthropic_routes_prefix}"
        f"/v1/messages/batches/{batch_id}/results"
    )


def _status(state: BatchState) -> MessageBatchStatus:
    """Map the state of a batch's jobs to the Message Batch processing status.

    Args:
        state: The batch state.

    Returns:
        The processing status reported to the client.
    """
    if state.ended:
        return "ended"
    return (
        "canceling" if state.record.cancel_initiated_at is not None else "in_progress"
    )


def _to_message_batch(http_request: Request, state: BatchState) -> MessageBatch:
    """Convert a batch state to an Anthropic ``MessageBatch`` response.

    Args:
        http_request: The incoming request, whose origin ``results_url`` is built on.
        state: The batch state.

    Returns:
        Serialisable ``MessageBatch``.
    """
    record: BatchRecord = state.record
    batch_id = f"msgbatch_{record.batch_id}"
    status = _status(state)
    ended = status == "ended"
    cancelled = state.cancelled
    settled = state.succeeded + state.errored
    return MessageBatch(
        id=batch_id,
        processing_status=status,
        request_counts=MessageBatchRequestCounts(
            processing=0 if ended else record.requests - settled,
            succeeded=state.succeeded,
            errored=state.errored,
            canceled=(record.requests - settled) if cancelled else 0,
            expired=(record.requests - settled) if state.expired else 0,
        ),
        created_at=rfc3339(record.created_at) or "",
        expires_at=rfc3339(record.expires_at) or "",
        ended_at=rfc3339(state.ended_at),
        cancel_initiated_at=rfc3339(record.cancel_initiated_at),
        results_url=_results_url(http_request, batch_id) if ended else None,
    )


@router.post(
    "/messages/batches",
    summary="Create a batch of message requests to run asynchronously (Anthropic format)",
    operation_id="anthropic_message_batch",
    description=(
        "Creates a Message Batch that runs many `anthropic_message` requests "
        "asynchronously, at the discounted batch price (Anthropic Message "
        "Batches API).\n\n"
        f"A batch must carry at least {MIN_REQUESTS_PER_MODEL} requests for "
        f"each model it names, and may name at most {MAX_MODELS_PER_BATCH} "
        "different models. Tool use, structured outputs and streaming are not "
        "available in a batch.\n\n"
        "Returns the `MessageBatch` immediately. Poll "
        "`anthropic_message_batch_get` until its `processing_status` is "
        "`ended`, then read `results_url` with `anthropic_message_batch_results`. "
        "Results may come back in any order — match them to the requests by "
        "`custom_id`."
    ),
    response_description="The created Message Batch.",
    responses={
        200: {"description": "Message Batch created."},
        400: {"description": "Invalid request or unsupported parameters."},
        503: {"description": "The Message Batches API is not enabled on this server."},
    },
    response_model_exclude_none=True,
)
async def create(
    http_request: Request,
    request: MessageBatchCreateParams,
    _: Annotated[None, Depends(authenticate)] = None,
) -> MessageBatch:
    """Create a Message Batch from inline requests.

    Args:
        http_request: The incoming request.
        request: Message Batch creation parameters.

    Returns:
        The created Message Batch.

    Raises:
        ApiError: With 400 when a request cannot be batched; 503 when the
            Message Batches API is not enabled.
    """
    log_request_params({"requests": len(request.requests)})
    prepared = await prepare_anthropic_requests(request.requests)
    return log_response_params(
        _to_message_batch(
            http_request,
            await create_batch(
                surface="anthropic",
                endpoint=_ENDPOINT,
                completion_window="24h",
                prepared=prepared,
            ),
        )
    )


@router.get(
    "/messages/batches",
    summary="List Message Batches (Anthropic format)",
    operation_id="anthropic_message_batch_list",
    description=(
        "Returns a paginated list of Message Batches, newest first (Anthropic "
        "Message Batches API)."
    ),
    response_description="A paginated list of Message Batch objects.",
    response_model_exclude_none=True,
)
async def list_all(
    http_request: Request,
    before_id: Annotated[
        str | None,
        Query(
            description="Cursor for pagination: return the page immediately "
            "before this Message Batch ID.",
            pattern=MESSAGE_BATCH_ID_PATTERN,
        ),
    ] = None,
    after_id: Annotated[
        str | None,
        Query(
            description="Cursor for pagination: return the page immediately "
            "after this Message Batch ID.",
            pattern=MESSAGE_BATCH_ID_PATTERN,
        ),
    ] = None,
    limit: Annotated[
        int, Query(ge=1, le=1000, description="Number of items to return per page.")
    ] = 20,
    _: Annotated[None, Depends(authenticate)] = None,
) -> MessageBatchList:
    """List Message Batches, newest first.

    Args:
        http_request: The incoming request.
        before_id: Message Batch ID cursor for the preceding page.
        after_id: Message Batch ID cursor for the following page.
        limit: Maximum number of Message Batches to return.

    Returns:
        Paginated list of Message Batch objects.
    """
    log_request_params({"before_id": before_id, "after_id": after_id, "limit": limit})
    states, has_more = await list_batches(
        "anthropic",
        after=after_id[9:] if after_id else None,
        before=before_id[9:] if before_id else None,
        limit=limit,
    )
    batches = [_to_message_batch(http_request, state) for state in states]
    return log_response_params(
        MessageBatchList(
            data=batches,
            has_more=has_more,
            first_id=batches[0].id if batches else None,
            last_id=batches[-1].id if batches else None,
        )
    )


@router.get(
    "/messages/batches/{message_batch_id}",
    summary="Retrieve a Message Batch (Anthropic format)",
    operation_id="anthropic_message_batch_get",
    description=(
        "Returns the current state of a Message Batch (Anthropic Message "
        "Batches API). Poll this endpoint until `processing_status` is `ended`, "
        "then read the results with `anthropic_message_batch_results`."
    ),
    response_description="The Message Batch.",
    responses={
        200: {"description": "The Message Batch state."},
        404: {"description": "Message Batch not found."},
    },
    response_model_exclude_none=True,
)
async def retrieve(
    http_request: Request,
    message_batch_id: _BatchId,
    _: Annotated[None, Depends(authenticate)] = None,
) -> MessageBatch:
    """Retrieve the current state of a Message Batch.

    Args:
        http_request: The incoming request.
        message_batch_id: Message Batch identifier.

    Returns:
        The Message Batch.

    Raises:
        ApiError: With 404 if the Message Batch does not exist.
    """
    log_request_params({"message_batch_id": message_batch_id})
    return log_response_params(
        _to_message_batch(
            http_request,
            await settle(await get_batch(message_batch_id[9:], "anthropic")),
        )
    )


@router.get(
    "/messages/batches/{message_batch_id}/results",
    summary="Stream the results of a Message Batch (Anthropic format)",
    operation_id="anthropic_message_batch_results",
    description=(
        "Streams the results of a Message Batch as JSONL, one result per line "
        "(Anthropic Message Batches API). Available only once "
        "`processing_status` is `ended`. Results may come back in any order — "
        "match them to the requests by `custom_id`."
    ),
    response_description="The results, as a JSONL stream.",
    responses={
        200: {"description": "The results."},
        404: {"description": "Message Batch not found or not ended yet."},
    },
)
async def results(
    message_batch_id: _BatchId, _: Annotated[None, Depends(authenticate)] = None
) -> StreamingResponse:
    """Stream the results of an ended Message Batch.

    Args:
        message_batch_id: Message Batch identifier.

    Returns:
        StreamingResponse yielding one JSON result per line.

    Raises:
        ApiError: With 404 if the Message Batch does not exist or has not
            ended yet.
    """
    log_request_params({"message_batch_id": message_batch_id})
    state = await settle(await get_batch(message_batch_id[9:], "anthropic"))
    if not state.ended:
        msg = (
            "The results of this Message Batch are not available yet. "
            "Retrieve the batch until its processing_status is 'ended'."
        )
        raise ApiError(msg, status=404)
    return StreamingResponse(
        iter_anthropic_results(state.record, canceled=state.cancelled),
        media_type="application/x-jsonlines",
    )


@router.post(
    "/messages/batches/{message_batch_id}/cancel",
    summary="Cancel a Message Batch (Anthropic format)",
    operation_id="anthropic_message_batch_cancel",
    description=(
        "Cancels a Message Batch that is still processing (Anthropic Message "
        "Batches API). The batch moves to `canceling` and then to `ended`; "
        "requests that already produced a Message keep it and stay readable in "
        "the results, and the ones that never ran are reported `canceled`. "
        "Cancelling a batch that has already ended changes nothing."
    ),
    response_description="The Message Batch being cancelled.",
    responses={
        200: {"description": "Cancellation started."},
        404: {"description": "Message Batch not found."},
    },
    response_model_exclude_none=True,
)
async def cancel(
    http_request: Request,
    message_batch_id: _BatchId,
    _: Annotated[None, Depends(authenticate)] = None,
) -> MessageBatch:
    """Cancel a processing Message Batch.

    Args:
        http_request: The incoming request.
        message_batch_id: Message Batch identifier.

    Returns:
        The Message Batch, now cancelling.

    Raises:
        ApiError: With 404 if the Message Batch does not exist.
    """
    log_request_params({"message_batch_id": message_batch_id})
    return log_response_params(
        _to_message_batch(
            http_request, await cancel_batch(message_batch_id[9:], "anthropic")
        )
    )


@router.delete(
    "/messages/batches/{message_batch_id}",
    summary="Delete a Message Batch (Anthropic format)",
    operation_id="anthropic_message_batch_delete",
    description=(
        "Deletes a Message Batch and its results (Anthropic Message Batches "
        "API). Only a batch whose `processing_status` is `ended` can be "
        "deleted; cancel it first with `anthropic_message_batch_cancel` if it "
        "is still processing."
    ),
    response_description="Deletion confirmation.",
    responses={
        200: {"description": "Message Batch deleted."},
        400: {"description": "The Message Batch is still processing."},
        404: {"description": "Message Batch not found."},
    },
    response_model_exclude_none=True,
)
async def delete(
    message_batch_id: _BatchId, _: Annotated[None, Depends(authenticate)] = None
) -> DeletedMessageBatch:
    """Delete an ended Message Batch.

    Args:
        message_batch_id: Message Batch identifier.

    Returns:
        Deletion confirmation.

    Raises:
        ApiError: With 404 if the Message Batch does not exist; 400 while it
            is still processing.
    """
    log_request_params({"message_batch_id": message_batch_id})
    await delete_batch(message_batch_id[9:], "anthropic")
    return log_response_params(DeletedMessageBatch(id=message_batch_id))
