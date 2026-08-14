"""OpenAI Batch API types."""

from typing import Literal

from pydantic import Field, JsonValue, field_validator

from stdapi.types import BaseModelRequest, BaseModelResponse
from stdapi.types.openai import Metadata, PaginatedListEnvelope

#: Regex pattern that a valid Batch API batch ID must match.
BATCH_ID_PATTERN: str = r"^batch_[a-z0-9]{32}$"

#: Maximum length of a metadata key, as upstream imposes.
_METADATA_KEY_MAX_LEN: int = 64

#: Maximum length of a metadata value, as upstream imposes.
_METADATA_VALUE_MAX_LEN: int = 512

#: Shortest accepted lifetime (seconds) for a batch's result files (1 hour).
_OUTPUT_EXPIRES_AFTER_SECONDS_MIN: int = 3600

#: Longest accepted lifetime (seconds) for a batch's result files (30 days).
_OUTPUT_EXPIRES_AFTER_SECONDS_MAX: int = 2592000

#: Batch lifecycle status.
BatchStatus = Literal[
    "validating",
    "failed",
    "in_progress",
    "finalizing",
    "completed",
    "expired",
    "cancelling",
    "cancelled",
]

#: API endpoints a batch may target.
BatchEndpoint = Literal["/v1/chat/completions"]


class BatchOutputExpiresAfter(BaseModelRequest):
    """Expiration policy for the files a batch produces."""

    anchor: Literal["created_at"] = Field(
        description="Timestamp the expiration is counted from: the creation of "
        "the result files, not of the batch. `created_at` is the only accepted "
        "value."
    )
    seconds: int = Field(
        ge=_OUTPUT_EXPIRES_AFTER_SECONDS_MIN,
        le=_OUTPUT_EXPIRES_AFTER_SECONDS_MAX,
        description="Number of seconds the result files stay readable, between "
        "3600 (1 hour) and 2592000 (30 days).",
    )


class BatchCreateParams(BaseModelRequest):
    """Batch creation parameters (OpenAI Batch API)."""

    input_file_id: str = Field(
        description="ID of an uploaded file holding the requests to run, one "
        "JSON object per line. The file must have been uploaded with "
        "`purpose='batch'`, and every line must name the same model."
    )
    endpoint: BatchEndpoint = Field(
        description="The API endpoint every request in the file targets. "
        "`/v1/chat/completions` is the endpoint available for batches."
    )
    completion_window: Literal["24h"] = Field(
        description="The time frame within which the batch is processed. "
        "`24h` is the only accepted value."
    )
    metadata: Metadata | None = Field(
        default=None,
        description="Up to 16 key-value pairs stored with the batch and "
        "returned on every read. Keys are at most 64 characters, values at "
        "most 512.",
        max_length=16,
    )
    output_expires_after: BatchOutputExpiresAfter | None = Field(
        default=None,
        description="Expiration policy for the result files. Once expired, "
        "they are no longer readable and are deleted. Omit it to keep them "
        "until they are deleted with `openai_file_delete`.",
    )

    @field_validator("metadata")
    @classmethod
    def _check_metadata_lengths(cls, value: Metadata | None) -> Metadata | None:
        """Enforce the per-key and per-value caps the batch record is stored with.

        Args:
            value: The submitted metadata.

        Returns:
            The metadata, unchanged.

        Raises:
            ValueError: When a key or a value is too long.
        """
        for key, item in (value or {}).items():
            if len(key) > _METADATA_KEY_MAX_LEN or len(item) > _METADATA_VALUE_MAX_LEN:
                msg = (
                    f"'metadata' keys are at most {_METADATA_KEY_MAX_LEN} "
                    f"characters and values at most {_METADATA_VALUE_MAX_LEN}."
                )
                raise ValueError(msg)
        return value


class BatchError(BaseModelResponse):
    """A validation error that prevented a batch from being created or run."""

    code: str | None = Field(default=None, description="Machine-readable error code.")
    message: str | None = Field(
        default=None, description="Human-readable description of the error."
    )
    param: str | None = Field(
        default=None,
        description="Name of the request parameter that caused the error, if any.",
    )
    line: int | None = Field(
        default=None,
        description="Line number in the input file where the error occurred, if any.",
    )


class BatchErrors(BaseModelResponse):
    """List of batch-level errors."""

    object: Literal["list"] = Field(
        default="list", description="The object type, which is always `list`."
    )
    data: list[BatchError] = Field(
        default_factory=list, description="The batch-level errors."
    )


class BatchRequestCounts(BaseModelResponse):
    """Counts of requests in the batch, by outcome."""

    total: int = Field(description="Total number of requests in the batch.")
    completed: int = Field(
        description="Number of requests that completed successfully."
    )
    failed: int = Field(description="Number of requests that failed.")


class BatchUsageInputTokensDetails(BaseModelResponse):
    """Breakdown of the batch's input tokens."""

    cached_tokens: int = Field(
        default=0, description="Number of input tokens served from the prompt cache."
    )


class BatchUsageOutputTokensDetails(BaseModelResponse):
    """Breakdown of the batch's output tokens."""

    reasoning_tokens: int = Field(
        default=0, description="Number of output tokens spent on reasoning."
    )


class BatchUsage(BaseModelResponse):
    """Token usage totalled over every request in the batch."""

    input_tokens: int = Field(description="Number of input tokens.")
    input_tokens_details: BatchUsageInputTokensDetails = Field(
        default_factory=BatchUsageInputTokensDetails,
        description="Breakdown of the input tokens.",
    )
    output_tokens: int = Field(description="Number of output tokens.")
    output_tokens_details: BatchUsageOutputTokensDetails = Field(
        default_factory=BatchUsageOutputTokensDetails,
        description="Breakdown of the output tokens.",
    )
    total_tokens: int = Field(description="Total number of tokens used.")


class Batch(BaseModelResponse):
    """A batch of requests processed asynchronously (OpenAI Batch API)."""

    id: str = Field(description="Unique identifier of the batch.")
    object: Literal["batch"] = Field(
        default="batch", description="The object type, which is always `batch`."
    )
    endpoint: str = Field(description="The API endpoint the batch targets.")
    errors: BatchErrors | None = Field(
        default=None, description="Batch-level errors, when the batch failed."
    )
    input_file_id: str = Field(description="ID of the batch's input file.")
    completion_window: str = Field(
        description="The time frame within which the batch is processed."
    )
    status: BatchStatus = Field(description="Current status of the batch.")
    output_file_id: str | None = Field(
        default=None,
        description="ID of the file holding the results of the requests that "
        "completed successfully. Available once the batch has ended.",
    )
    error_file_id: str | None = Field(
        default=None,
        description="ID of the file holding the results of the requests that "
        "failed. Available once the batch has ended and at least one request "
        "failed.",
    )
    created_at: int = Field(
        description="Unix timestamp (in seconds) when the batch was created."
    )
    in_progress_at: int | None = Field(
        default=None,
        description="Unix timestamp (in seconds) when the batch started running.",
    )
    expires_at: int | None = Field(
        default=None,
        description="Unix timestamp (in seconds) when the batch stops being processed.",
    )
    finalizing_at: int | None = Field(
        default=None,
        description="Unix timestamp (in seconds) when the batch started finalizing.",
    )
    completed_at: int | None = Field(
        default=None,
        description="Unix timestamp (in seconds) when the batch completed.",
    )
    failed_at: int | None = Field(
        default=None, description="Unix timestamp (in seconds) when the batch failed."
    )
    expired_at: int | None = Field(
        default=None, description="Unix timestamp (in seconds) when the batch expired."
    )
    cancelling_at: int | None = Field(
        default=None,
        description="Unix timestamp (in seconds) when cancellation started.",
    )
    cancelled_at: int | None = Field(
        default=None,
        description="Unix timestamp (in seconds) when the batch was cancelled.",
    )
    request_counts: BatchRequestCounts | None = Field(
        default=None, description="Counts of requests in the batch, by outcome."
    )
    usage: BatchUsage | None = Field(
        default=None,
        description="Token usage totalled over the batch, once it has ended.",
    )
    model: str | None = Field(
        default=None, description="The model every request in the batch used."
    )
    metadata: Metadata | None = Field(
        default=None, description="The key-value pairs attached to the batch."
    )


class BatchList(PaginatedListEnvelope):
    """Paginated list of batches."""

    object: Literal["list"] = Field(
        default="list", description="The object type, which is always `list`."
    )
    data: list[Batch] = Field(description="List of Batch objects.")


class BatchOutputResponse(BaseModelResponse):
    """The response of one request, as written in a batch output file."""

    status_code: int = Field(description="HTTP status code of the request.")
    request_id: str = Field(description="Identifier of the individual request.")
    body: JsonValue = Field(description="The response body of the request.")


class BatchOutputLine(BaseModelResponse):
    """One line of a batch output or error file."""

    id: str = Field(description="Identifier of this result line.")
    custom_id: str = Field(
        description="The `custom_id` of the input line this result answers."
    )
    response: BatchOutputResponse | None = Field(
        default=None, description="The response, when the request succeeded."
    )
    error: BatchError | None = Field(
        default=None, description="The error, when the request failed."
    )
