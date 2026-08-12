"""Anthropic Message Batches API types."""

from typing import Annotated, Literal

from pydantic import Field

from stdapi.types import BaseModelRequest, BaseModelResponse
from stdapi.types.anthropic_messages import Message, MessageCreateParams

#: Regex pattern that a valid Message Batches API batch ID must match.
MESSAGE_BATCH_ID_PATTERN: str = r"^msgbatch_[a-z0-9]{32}$"

#: Regex pattern a `custom_id` must match, per the Message Batches API.
CUSTOM_ID_PATTERN: str = r"^[a-zA-Z0-9_-]{1,64}$"

#: Message Batch processing status.
MessageBatchStatus = Literal["in_progress", "canceling", "ended"]


class MessageBatchRequest(BaseModelRequest):
    """One request within a Message Batch."""

    custom_id: str = Field(
        description="Identifier you choose for this request, unique within the "
        "batch. Results may come back in any order, so this is how a result is "
        "matched back to its request.",
        pattern=CUSTOM_ID_PATTERN,
    )
    params: MessageCreateParams = Field(
        description="The Messages API parameters for this request, exactly as "
        "they would be sent to `anthropic_message`."
    )


class MessageBatchCreateParams(BaseModelRequest):
    """Message Batch creation parameters (Anthropic Message Batches API)."""

    requests: list[MessageBatchRequest] = Field(
        description="The requests to run. Each is an individual Messages API "
        "request with its own `custom_id` and parameters.",
        min_length=1,
    )


class MessageBatchRequestCounts(BaseModelResponse):
    """Counts of requests in the Message Batch, by status."""

    processing: int = Field(description="Number of requests still being processed.")
    succeeded: int = Field(
        description="Number of requests that completed successfully."
    )
    errored: int = Field(description="Number of requests that errored.")
    canceled: int = Field(
        description="Number of requests that were canceled. Known only once "
        "processing of the whole batch has ended."
    )
    expired: int = Field(
        description="Number of requests that expired. Known only once "
        "processing of the whole batch has ended."
    )


class MessageBatch(BaseModelResponse):
    """A batch of Messages API requests processed asynchronously."""

    id: str = Field(description="Unique object identifier of the Message Batch.")
    type: Literal["message_batch"] = Field(
        default="message_batch", description="Object type. Always `message_batch`."
    )
    processing_status: MessageBatchStatus = Field(
        description="Processing status of the Message Batch."
    )
    request_counts: MessageBatchRequestCounts = Field(
        description="Tallies of the requests in the batch, by status."
    )
    created_at: str = Field(
        description="RFC 3339 datetime string of the Message Batch creation time."
    )
    expires_at: str = Field(
        description="RFC 3339 datetime string of the time at which the Message "
        "Batch stops being processed, 24 hours after creation."
    )
    ended_at: str | None = Field(
        default=None,
        description="RFC 3339 datetime string of the time processing ended. "
        "Set only once every request has succeeded, errored, been canceled or "
        "expired.",
    )
    archived_at: str | None = Field(
        default=None,
        description="RFC 3339 datetime string of the time the Message Batch was "
        "archived and its results became unavailable.",
    )
    cancel_initiated_at: str | None = Field(
        default=None,
        description="RFC 3339 datetime string of the time cancellation was "
        "initiated. Set only if cancellation was initiated.",
    )
    results_url: str | None = Field(
        default=None,
        description="URL of the `.jsonl` file holding the results of the "
        "batch's requests. Set only once processing has ended. Results are not "
        "guaranteed to be in request order — match them with `custom_id`.",
    )


class MessageBatchList(BaseModelResponse):
    """Paginated list of Message Batches."""

    data: list[MessageBatch] = Field(description="List of Message Batch objects.")
    has_more: bool = Field(
        description="Whether more Message Batches exist after this page."
    )
    first_id: str | None = Field(
        default=None,
        description="ID of the first Message Batch in this page, or null when empty.",
    )
    last_id: str | None = Field(
        default=None,
        description="ID of the last Message Batch in this page, or null when empty.",
    )


class DeletedMessageBatch(BaseModelResponse):
    """Message Batch deletion confirmation."""

    id: str = Field(description="ID of the deleted Message Batch.")
    type: Literal["message_batch_deleted"] = Field(
        default="message_batch_deleted",
        description="Deleted object type. Always `message_batch_deleted`.",
    )


class MessageBatchSucceededResult(BaseModelResponse):
    """Result of a request that completed successfully."""

    type: Literal["succeeded"] = Field(
        default="succeeded", description="Result type. Always `succeeded`."
    )
    message: Message = Field(description="The Message the request produced.")


class MessageBatchErrorDetail(BaseModelResponse):
    """The error that made a batched request fail."""

    type: str = Field(description="Machine-readable error type.")
    message: str = Field(description="Human-readable description of the error.")


class MessageBatchErrorEnvelope(BaseModelResponse):
    """Anthropic error envelope carried by an errored batch result."""

    type: Literal["error"] = Field(
        default="error", description="Object type. Always `error`."
    )
    error: MessageBatchErrorDetail = Field(description="The error.")


class MessageBatchErroredResult(BaseModelResponse):
    """Result of a request that errored."""

    type: Literal["errored"] = Field(
        default="errored", description="Result type. Always `errored`."
    )
    error: MessageBatchErrorEnvelope = Field(
        description="The error the request failed with."
    )


class MessageBatchCanceledResult(BaseModelResponse):
    """Result of a request that was canceled before it ran."""

    type: Literal["canceled"] = Field(
        default="canceled", description="Result type. Always `canceled`."
    )


class MessageBatchExpiredResult(BaseModelResponse):
    """Result of a request that expired before it ran."""

    type: Literal["expired"] = Field(
        default="expired", description="Result type. Always `expired`."
    )


#: Processing result of one batched request.
MessageBatchResult = Annotated[
    MessageBatchSucceededResult
    | MessageBatchErroredResult
    | MessageBatchCanceledResult
    | MessageBatchExpiredResult,
    Field(discriminator="type"),
]


class MessageBatchIndividualResponse(BaseModelResponse):
    """One line of a Message Batch results file."""

    custom_id: str = Field(
        description="The `custom_id` of the request this result answers."
    )
    result: MessageBatchResult = Field(
        description="Processing result for this request."
    )
