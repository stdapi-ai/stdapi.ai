"""OpenAI-compatible ``/v1/batches`` endpoints.

A batch runs a file of requests asynchronously, at the discounted batch price
and without holding a connection open. The batch is created from a file
uploaded with ``purpose='batch'``, polled until it ends, and read back from the
result files it names.
"""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Path, Query

from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.batches import (
    MIN_REQUESTS_PER_MODEL,
    BatchState,
    cancel_batch,
    create_batch,
    finish_listed,
    get_batch,
    list_batches,
    materialize_openai_results,
    prepare_openai_requests,
    read_input_requests,
    require_batches_enabled,
    settle,
)
from stdapi.config import SETTINGS
from stdapi.monitoring import log_request_params, log_response_params
from stdapi.types.openai_batches import (
    BATCH_ID_PATTERN,
    Batch,
    BatchCreateParams,
    BatchError,
    BatchErrors,
    BatchList,
    BatchRequestCounts,
    BatchStatus,
    BatchUsage,
    BatchUsageInputTokensDetails,
)

if TYPE_CHECKING:
    from stdapi.batches import BatchRecord

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=["Batches", TAG_OPENAI]
)

#: What a client is told about a batch that stopped without running.
_FAILED_BATCH_ERROR = BatchError(
    code="batch_failed",
    message="The batch stopped before its requests could run and produced no "
    "result. Create it again, or send the requests without batching.",
)

#: Reusable path annotation for the ``batch_id`` path parameter.
_BatchId = Annotated[
    str, Path(description="The identifier of the batch.", pattern=BATCH_ID_PATTERN)
]


def _status(state: BatchState) -> BatchStatus:
    """Map the state of a batch's jobs to the OpenAI batch status.

    A batch is `completed` only once its results are readable: between the
    moment the requests stop running and the moment the result files are
    published, it is `finalizing`.

    Args:
        state: The batch state.

    Returns:
        The status reported to the client.
    """
    if state.record.cancel_initiated_at is not None:
        return "cancelled" if state.ended else "cancelling"
    if state.failed:
        return "failed"
    if state.expired:
        return "expired"
    if state.ended:
        return "completed" if state.record.output_file_id is not None else "finalizing"
    return "validating" if state.pending else "in_progress"


def _to_batch(state: BatchState) -> Batch:
    """Convert a batch state to an OpenAI ``Batch`` response.

    Args:
        state: The batch state.

    Returns:
        Serialisable ``Batch``.
    """
    record: BatchRecord = state.record
    status = _status(state)
    ended_at = state.ended_at
    started = min((job.submitted_at for job in state.jobs), default=record.created_at)
    usage = record.usage
    return Batch(
        id=f"batch_{record.batch_id}",
        endpoint=record.endpoint,
        input_file_id=record.input_file_id or "",
        completion_window=record.completion_window,
        status=status,
        output_file_id=record.output_file_id,
        error_file_id=record.error_file_id,
        created_at=record.created_at,
        in_progress_at=None if state.pending else started,
        expires_at=record.expires_at,
        finalizing_at=ended_at if status in ("finalizing", "completed") else None,
        completed_at=ended_at if status == "completed" else None,
        failed_at=ended_at if status == "failed" else None,
        expired_at=ended_at if status == "expired" else None,
        cancelling_at=record.cancel_initiated_at,
        cancelled_at=ended_at if status == "cancelled" else None,
        request_counts=BatchRequestCounts(
            total=record.requests, completed=state.succeeded, failed=state.errored
        ),
        usage=BatchUsage(
            input_tokens=usage.input_tokens,
            input_tokens_details=BatchUsageInputTokensDetails(
                cached_tokens=usage.cached_tokens
            ),
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )
        if usage is not None
        else None,
        model=record.jobs[0].model if record.jobs else None,
        metadata=record.metadata,
        errors=BatchErrors(data=[_FAILED_BATCH_ERROR]) if status == "failed" else None,
    )


async def _finish(state: BatchState) -> BatchState:
    """Total and publish a batch's results once it has ended.

    Args:
        state: The batch state.

    Returns:
        The batch state, with its usage and its result files named.
    """
    return await materialize_openai_results(await settle(state))


@router.post(
    "/batches",
    summary="Create a batch of requests to run asynchronously (OpenAI format)",
    operation_id="openai_batch",
    description=(
        "Creates a batch that runs every request of an uploaded file "
        "asynchronously, at the discounted batch price (OpenAI Batch API).\n\n"
        "Upload the requests first with `openai_file` and `purpose='batch'`: one "
        "JSON object per line, each with a unique `custom_id`, `method` `POST`, "
        "`url` equal to `endpoint`, and `body` holding the request itself. "
        "Every line must name the same model.\n\n"
        f"A batch must carry at least {MIN_REQUESTS_PER_MODEL} requests. Tool "
        "use, structured outputs (`response_format` of type `json_schema`), "
        "streaming and `n` above 1 are not available in a batch.\n\n"
        "Returns the `Batch` immediately. Poll `openai_batch_get` until its "
        "`status` is `completed`, then download `output_file_id` with "
        "`openai_file_content`. Results may come back in any order — match them "
        "to the requests by `custom_id`.\n\n"
        "Set `output_expires_after` to have the result files deleted "
        "automatically once they are no longer needed."
    ),
    response_description="The created batch.",
    responses={
        200: {"description": "Batch created."},
        400: {"description": "Invalid request or unsupported parameters."},
        503: {"description": "The Batch API is not enabled on this server."},
    },
    response_model_exclude_none=True,
)
async def create(
    request: BatchCreateParams, _: Annotated[None, Depends(authenticate)] = None
) -> Batch:
    """Create a batch from an uploaded file of requests.

    Args:
        request: Batch creation parameters.

    Returns:
        The created batch.

    Raises:
        ApiError: With 400 when the input file or one of its lines cannot be
            batched; 503 when the Batch API is not enabled.
    """
    log_request_params(request)
    # Before the input file is read, so a 503 costs no 200 MB parse.
    require_batches_enabled()
    lines = await read_input_requests(request.input_file_id)
    prepared = await prepare_openai_requests(lines, request.endpoint)
    del lines
    return log_response_params(
        _to_batch(
            await create_batch(
                surface="openai",
                endpoint=request.endpoint,
                completion_window=request.completion_window,
                prepared=prepared,
                input_file_id=request.input_file_id,
                metadata=request.metadata,
                output_expires_after=(
                    request.output_expires_after.seconds
                    if request.output_expires_after
                    else None
                ),
            )
        )
    )


@router.get(
    "/batches",
    summary="List batches (OpenAI format)",
    operation_id="openai_batch_list",
    description=(
        "Returns a paginated list of batches, newest first (OpenAI Batch API)."
    ),
    response_description="A paginated list of Batch objects.",
    response_model_exclude_none=True,
)
async def list_all(
    after: Annotated[
        str | None,
        Query(
            description="Cursor for pagination: the batch ID to start after "
            "(the last ID from a previous page).",
            pattern=BATCH_ID_PATTERN,
        ),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=100, description="A limit on the number of objects returned."),
    ] = 20,
    _: Annotated[None, Depends(authenticate)] = None,
) -> BatchList:
    """List batches, newest first.

    Args:
        after: Batch ID cursor; only batches created before it are returned.
        limit: Maximum number of batches to return.

    Returns:
        Paginated list of Batch objects.
    """
    log_request_params({"after": after, "limit": limit})
    states, has_more = await list_batches(
        "openai", after=after[6:] if after else None, limit=limit
    )
    batches = [_to_batch(state) for state in await finish_listed(states, _finish)]
    return log_response_params(
        BatchList(
            data=batches,
            has_more=has_more,
            first_id=batches[0].id if batches else None,
            last_id=batches[-1].id if batches else None,
        )
    )


@router.get(
    "/batches/{batch_id}",
    summary="Retrieve a batch (OpenAI format)",
    operation_id="openai_batch_get",
    description=(
        "Returns the current state of a batch (OpenAI Batch API). Poll this "
        "endpoint until `status` is `completed`, then download `output_file_id` "
        "with `openai_file_content`; requests that failed are collected in "
        "`error_file_id`."
    ),
    response_description="The batch.",
    responses={
        200: {"description": "The batch state."},
        404: {"description": "Batch not found."},
    },
    response_model_exclude_none=True,
)
async def retrieve(
    batch_id: _BatchId, _: Annotated[None, Depends(authenticate)] = None
) -> Batch:
    """Retrieve the current state of a batch.

    Args:
        batch_id: Batch identifier.

    Returns:
        The batch.

    Raises:
        ApiError: With 404 if the batch does not exist.
    """
    log_request_params({"batch_id": batch_id})
    return log_response_params(
        _to_batch(await _finish(await get_batch(batch_id[6:], "openai")))
    )


@router.post(
    "/batches/{batch_id}/cancel",
    summary="Cancel a batch (OpenAI format)",
    operation_id="openai_batch_cancel",
    description=(
        "Cancels a batch that is still running (OpenAI Batch API). The batch "
        "moves to `cancelling` and then to `cancelled`; requests that already "
        "produced an answer keep it and stay readable in `output_file_id`. "
        "Cancelling a batch that has already ended changes nothing."
    ),
    response_description="The batch being cancelled.",
    responses={
        200: {"description": "Cancellation started."},
        404: {"description": "Batch not found."},
    },
    response_model_exclude_none=True,
)
async def cancel(
    batch_id: _BatchId, _: Annotated[None, Depends(authenticate)] = None
) -> Batch:
    """Cancel a running batch.

    Args:
        batch_id: Batch identifier.

    Returns:
        The batch, now cancelling.

    Raises:
        ApiError: With 404 if the batch does not exist.
    """
    log_request_params({"batch_id": batch_id})
    # A cancel that arrives after the batch ended settles it, as a read does:
    # otherwise the same batch is `finalizing` here and `completed` on a poll.
    return log_response_params(
        _to_batch(await _finish(await cancel_batch(batch_id[6:], "openai")))
    )
