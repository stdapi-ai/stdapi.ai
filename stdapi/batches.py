"""Batch API engine — many model requests, run asynchronously for a lower price.

A batch is submitted once, runs without a connection held open, and is read
back later. Its state lives entirely in object storage and in the backing
inference jobs, so the server itself stays stateless: one small record object
per batch holds what the client sent, and every status figure is read live.

Requests are grouped by the model they name, one inference job per model, and
the batch reports the aggregate. Results are written per job and translated to
the calling API's dialect on read.
"""

from asyncio import TaskGroup, gather
from base64 import b32hexencode
from binascii import crc32 as _crc32
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import blake2b
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal

from botocore.exceptions import ClientError
from pydantic_core import from_json

from stdapi.api_errors import (
    ApiError,
    FeatureUnavailableError,
    feature_unavailable_guard,
)
from stdapi.aws import get_client
from stdapi.aws_bedrock import GUARDRAIL_CONFIG_VAR, get_extra_model_parameters
from stdapi.aws_s3 import BUCKET_TO_REGION, put_s3_object, require_s3_bucket_for_region
from stdapi.cleanup import schedule_cleanup
from stdapi.config import SETTINGS
from stdapi.files import (
    decode_id_payload,
    encode_id_payload,
    get_file,
    get_file_content,
    parse_file_id,
    payload_created_at,
    put_file_content,
    resolve_file_bucket,
)
from stdapi.models import (
    ModelBase,
    ModelRegionUnavailableError,
    runtime_twin,
    validate_model,
)
from stdapi.models.chat import get_chat_model, serves_via_mantle
from stdapi.models.chat._adapters import _anthropic_message as anthropic_adapter
from stdapi.models.chat._adapters import _openai_chat_completion as openai_adapter
from stdapi.models.chat._default import ChatModel
from stdapi.models.embedding import EmbeddingModelBase, get_embedding_model
from stdapi.monitoring import log_error_details, tenant_aws_credential
from stdapi.routes.openai_embeddings import build_embedding_response
from stdapi.types.openai_chat_completions import CompletionCreateParams
from stdapi.types.openai_embeddings import EmbeddingCreateParams
from stdapi.usage import record_bedrock_usage
from stdapi.utils import now_utc_timestamp, to_json_bytes, validation_error_handler

if TYPE_CHECKING:
    from asyncio import Task
    from collections.abc import (
        AsyncIterator,
        Awaitable,
        Callable,
        Coroutine,
        Generator,
        Sequence,
    )

    from types_aiobotocore_bedrock.client import BedrockClient
    from types_aiobotocore_bedrock.literals import ModelInvocationTypeType, RegionName
    from types_aiobotocore_bedrock.type_defs import GetModelInvocationJobResponseTypeDef
    from types_aiobotocore_bedrock_runtime.type_defs import ConverseResponseTypeDef
    from types_aiobotocore_s3.client import S3Client
    from types_aiobotocore_s3.type_defs import ObjectIdentifierTypeDef

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef
    from stdapi.input_file import InputFileUrl
    from stdapi.types import JsonMapping
    from stdapi.types.anthropic_batches import MessageBatchRequest

#: Which API a batch was created through; its results speak that API's dialect.
BatchSurface = Literal["openai", "anthropic"]

#: The feature name a caller reads when the deployment cannot run a batch.
_FEATURE: str = "The Batch API"

#: Permissions a denied job submission names to the operator.
_CREATE_JOB_PERMISSIONS: str = (
    "bedrock:CreateModelInvocationJob, or iam:PassRole on the batch service "
    "role set in 'aws_bedrock_batch_role_arn' with the condition "
    "iam:PassedToService=bedrock.amazonaws.com"
)

#: How a refused submission reads when the model, not the deployment, is the cause.
_MODEL_REFUSALS: tuple[str, ...] = (
    "The provided model identifier is invalid",
    "Batch inference is not supported",
)

#: What the operator reads when a job submission is refused as invalid.
_REFUSED_JOB_DETAIL: str = (
    "Batch inference job refused: {message}. Check the account's batch quotas "
    "for this model, the model's own batch availability, and the batch service "
    "role set in 'aws_bedrock_batch_role_arn'."
)

#: Endpoint whose requests a batch runs as embeddings rather than as completions.
_EMBEDDINGS_ENDPOINT: str = "/v1/embeddings"

#: Default minimum requests one model must carry, set by an adjustable backend quota.
MIN_REQUESTS_PER_MODEL: int = 100

#: Maximum number of distinct models one batch may fan out to.
MAX_MODELS_PER_BATCH: int = 8

#: Maximum number of requests one batch may carry, per calling API.
MAX_REQUESTS: dict[str, int] = {"openai": 50_000, "anthropic": 100_000}

#: Maximum accepted size of a batch input file, matching the upstream limit.
_MAX_INPUT_FILE_BYTES: int = 200 * 1024**2

#: Maximum accepted length of a `custom_id`.
_CUSTOM_ID_MAX_LEN: int = 64

#: How long a batch is processed before it stops, matching both APIs' 24 hours.
_BATCH_WINDOW_SECONDS: int = 24 * 3600

#: Backend job states in which no further work will happen.
_TERMINAL_STATUSES = frozenset(
    {"Completed", "PartiallyCompleted", "Failed", "Stopped", "Expired"}
)

#: Backend job states in which the requests have not started running yet.
_PENDING_STATUSES = frozenset({"Submitted", "Validating", "Scheduled"})

#: Errors meaning another read already recorded this batch's usage.
_ALREADY_CLAIMED_ERRORS = frozenset(
    {"PreconditionFailed", "ConditionalRequestConflict"}
)

#: Job errors meaning the batch is gone or was never ours.
_JOB_NOT_FOUND_ERRORS = frozenset(
    {"ValidationException", "AccessDeniedException", "ResourceNotFoundException"}
)

#: Job errors meaning the job was already over when the stop reached it.
_ALREADY_STOPPED_ERRORS = frozenset({"ValidationException", "ConflictException"})

#: Storage errors meaning the object was never written.
_NO_SUCH_OBJECT_ERRORS = frozenset({"404", "NoSuchKey"})

#: Image counters a job reports, as (counter, price bucket).
_IMAGE_COUNTERS: tuple[tuple[str, str], ...] = (
    ("inputStandardImageCount", ""),
    ("inputDocumentImageCount", "document"),
)

#: Media-duration counters a job reports, as (counter, price bucket).
_SECOND_COUNTERS: tuple[tuple[str, str], ...] = (
    ("inputAudioSecond", "audio"),
    ("inputVideoSecond", "video"),
)

#: Name of the object each job writes its aggregate counters to.
_MANIFEST_SUFFIX = "manifest.json.out"

#: Name of the file each job's requests are written to.
_INPUT_FILE_NAME = "input.jsonl"

#: Maximum batch records returned by one listing scan.
_LIST_SCAN_LIMIT: int = 1000

#: Maximum storage requests one bucket's listing scan makes.
_LIST_SCAN_REQUESTS: int = 20

#: Storage pages one probe of a listing scan walks before giving its instant up.
_LIST_PROBE_PAGES: int = 4

#: Time span a listing scan's first tail probe reaches back over, in milliseconds.
_LIST_SEEK_SPAN_MS: int = 3600 * 1000

#: Factor a tail probe's span grows by while it reaches too few records.
_LIST_SEEK_GROWTH: int = 16

#: Requests translated concurrently while a batch is being prepared.
_BUILD_CONCURRENCY: int = 32

#: Models the batch being prepared resolved, keyed by the name its lines wrote.
_PINNED_MODELS: ContextVar[dict[str, ModelBase[Any, Any]] | None] = ContextVar(
    "batch_pinned_models", default=None
)

#: Batches whose outcome is stored concurrently while a listing is answered.
_FINISH_CONCURRENCY: int = 8

#: Last payload byte marking a batch's results file.
_OUTPUT_FILE_MARKER: int = 1

#: Last payload byte marking a batch's errors file.
_ERROR_FILE_MARKER: int = 2

#: Client-facing message for a request the backend rejected as invalid.
_RECORD_ERRORS: dict[int, tuple[str, str]] = {
    400: (
        "invalid_request_error",
        "The request was rejected as invalid and produced no output.",
    ),
    429: (
        "rate_limit_exceeded",
        "The request was not run because the rate limit was exceeded.",
    ),
}

#: Client-facing message for a request that failed for any other reason.
_RECORD_ERROR_DEFAULT = (
    "server_error",
    "The request could not be completed and produced no output.",
)


class BatchNotFoundError(ApiError):
    """The requested batch does not exist."""

    status = 404


@dataclass(slots=True)
class BatchJobRef:
    """One backing inference job of a batch.

    Attributes:
        model: Model name exactly as the client wrote it.
        model_id: Resolved backend model identifier.
        region: AWS region running the job.
        bucket: S3 bucket holding the job's input and output.
        job_arn: ARN of the inference job.
        job_id: Identifier segment of the job ARN, naming its output folder.
        requests: Number of requests submitted to the job.
        prefix: S3 key prefix holding the job's data.
    """

    model: str
    model_id: str
    region: str
    bucket: str
    job_arn: str
    job_id: str
    requests: int
    prefix: str


@dataclass(slots=True)
class BatchUsageTotals:
    """Token counters totalled over a batch.

    Attributes:
        input_tokens: Prompt tokens, cache buckets included.
        output_tokens: Generated tokens.
        cached_tokens: Prompt tokens served from the cache.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


@dataclass(slots=True)
class BatchRecord:
    """What the client asked for, stored once and never recomputed.

    Attributes:
        batch_id: Bare 32-char payload identifying the batch.
        bucket: S3 bucket holding this record.
        surface: API the batch was created through.
        created_at: Unix timestamp read from ``batch_id``; the completion
            window runs from it.
        endpoint: API endpoint every request targets.
        completion_window: Time frame within which the batch is processed.
        input_file_id: Files API identifier of the submitted requests.
        metadata: Key-value pairs the client attached to the batch.
        jobs: The backing inference jobs.
        cancel_initiated_at: Unix timestamp cancellation was asked for.
        deleted: Whether the batch was deleted by the client.
        output_file_id: Files API identifier of the results.
        error_file_id: Files API identifier of the failed requests' results.
        usage: Token counters, once the batch has ended.
        output_expires_after: Seconds the result files stay readable once
            written, or ``None`` to keep them until they are deleted.
    """

    batch_id: str
    bucket: str
    surface: BatchSurface
    created_at: int
    endpoint: str
    completion_window: str
    input_file_id: str | None = None
    metadata: dict[str, str] | None = None
    output_expires_after: int | None = None
    jobs: list[BatchJobRef] = field(default_factory=list)
    cancel_initiated_at: int | None = None
    deleted: bool = False
    output_file_id: str | None = None
    error_file_id: str | None = None
    usage: BatchUsageTotals | None = None

    @property
    def expires_at(self) -> int:
        """Unix timestamp at which the batch stops being processed."""
        return self.created_at + _BATCH_WINDOW_SECONDS

    @property
    def requests(self) -> int:
        """Total number of requests in the batch."""
        return sum(job.requests for job in self.jobs)


@dataclass(slots=True)
class JobState:
    """Live state of one backing job.

    Attributes:
        ref: The job this state belongs to.
        status: Backend job status.
        submitted_at: Unix timestamp the job was submitted.
        ended_at: Unix timestamp the job stopped, once it has.
        total: Number of requests the job holds.
        succeeded: Number of requests that produced an answer.
        errored: Number of requests that failed.
    """

    ref: BatchJobRef
    status: str
    submitted_at: int
    ended_at: int | None
    total: int
    succeeded: int
    errored: int


@dataclass(slots=True)
class BatchState:
    """A batch and the live state of every job behind it.

    Attributes:
        record: The stored batch record.
        jobs: Live state of each backing job.
    """

    record: BatchRecord
    jobs: list[JobState]

    @property
    def ended(self) -> bool:
        """Whether every backing job has stopped."""
        return all(job.status in _TERMINAL_STATUSES for job in self.jobs)

    @property
    def pending(self) -> bool:
        """Whether no backing job has started running yet."""
        return all(job.status in _PENDING_STATUSES for job in self.jobs)

    @property
    def failed(self) -> bool:
        """Whether any backing job stopped without running its requests."""
        return any(job.status == "Failed" for job in self.jobs)

    @property
    def expired(self) -> bool:
        """Whether any backing job ran out of time."""
        return any(job.status == "Expired" for job in self.jobs)

    @property
    def cancelled(self) -> bool:
        """Whether cancellation was asked for and every job has stopped."""
        return self.record.cancel_initiated_at is not None and self.ended

    @property
    def ended_at(self) -> int | None:
        """Unix timestamp the last job stopped, or ``None`` while any runs."""
        if not self.ended:
            return None
        return max((job.ended_at or job.submitted_at) for job in self.jobs)

    @property
    def succeeded(self) -> int:
        """Number of requests that produced an answer."""
        return sum(job.succeeded for job in self.jobs)

    @property
    def errored(self) -> int:
        """Number of requests that failed."""
        return sum(job.errored for job in self.jobs)


def batch_s3_key(payload: str) -> str:
    """Return the S3 object key holding the batch record for *payload*."""
    return f"{SETTINGS.aws_s3_batches_prefix}{payload}"


def require_batches_enabled() -> tuple[str, str]:
    """Return the batch service role and the bucket holding batch records.

    Returns:
        Tuple of (service role ARN, S3 bucket name).

    Raises:
        ApiError: 503 when the Batch API is not configured on this server.
    """
    role_arn = SETTINGS.aws_bedrock_batch_role_arn
    bucket = SETTINGS.aws_s3_bucket
    if role_arn and bucket:
        return role_arn, bucket
    raise FeatureUnavailableError(
        _FEATURE,
        "Batch API disabled: 'aws_bedrock_batch_role_arn' and 'aws_s3_bucket' "
        "must both be set.",
    )


def _derive_file_payload(payload: str, bucket: str, marker: int) -> str:
    """Derive a stable file payload from a batch payload.

    The derived payload keeps the batch's creation-time ordering and carries
    the fingerprint of *bucket*, so the same batch always names the same file
    and writing it twice is harmless.

    Args:
        payload: Bare 32-char batch payload.
        bucket: Bucket the file is written to.
        marker: Byte distinguishing this file from the batch's other files.

    Returns:
        Bare 32-char file payload.
    """
    raw = decode_id_payload(payload)[:15].ljust(15, b"\0")
    return (
        b32hexencode(
            raw + bytes((marker,)) + _crc32(bucket.encode()).to_bytes(4, "big")
        )
        .lower()
        .decode()
    )


def _result_line_id(payload: str, custom_id: str) -> str:
    """Return a stable identifier for one result line of a batch."""
    digest = blake2b(f"{payload}:{custom_id}".encode(), digest_size=12).hexdigest()
    return f"batch_req_{digest}"


def _strip_nulls(value: Any) -> Any:  # noqa: ANN401
    """Return *value* without the keys whose value is null.

    Results echo every alternative of a union with a null value; the response
    adapters read a present key as a present value, so the nulls must go.

    Args:
        value: Decoded JSON value.

    Returns:
        The value with null-valued mapping keys removed, recursively.
    """
    if isinstance(value, dict):
        return {k: _strip_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_nulls(v) for v in value]
    return value


async def _iter_jsonl(stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Yield the non-empty lines of a byte stream, without their separator.

    Args:
        stream: Byte chunks, split at arbitrary boundaries.

    Yields:
        One line at a time.
    """
    buffer = bytearray()
    async for chunk in stream:
        buffer.extend(chunk)
        start = 0
        while (end := buffer.find(b"\n", start)) != -1:
            if line := bytes(buffer[start:end]).strip():
                yield line
            start = end + 1
        del buffer[:start]
    if line := bytes(buffer).strip():
        yield line


def _validate_custom_ids(custom_ids: Sequence[str], label: str) -> None:
    """Check that every ``custom_id`` is unique and short enough to round-trip.

    Args:
        custom_ids: The identifiers, in input order.
        label: How a request's position is named to the client.

    Raises:
        ApiError: When one is empty, too long, or used twice.
    """
    seen: set[str] = set()
    for index, custom_id in enumerate(custom_ids):
        if not custom_id or len(custom_id) > _CUSTOM_ID_MAX_LEN:
            msg = (
                f"{label} {index + 1}: 'custom_id' must be between 1 and "
                f"{_CUSTOM_ID_MAX_LEN} characters."
            )
            raise ApiError(msg)
        if custom_id in seen:
            msg = f"{label} {index + 1}: 'custom_id' {custom_id!r} is used more than once."
            raise ApiError(msg)
        seen.add(custom_id)


def _check_batchable(
    request: ConverseRequestBaseTypeDef, index: int, label: str
) -> None:
    """Reject a request asking for a capability batches cannot serve.

    Args:
        request: The translated request.
        index: Zero-based position of the request in the batch.
        label: How a request's position is named to the client.

    Raises:
        ApiError: When the request asks for tools, a structured output, or is
            covered by a content guardrail.
    """
    fields = request.get("additionalModelRequestFields") or {}
    if "toolConfig" in request or (
        isinstance(fields, dict) and fields.get("tools") is not None
    ):
        msg = (
            f"{label} {index + 1}: tool use is not available for batched requests. "
            "Remove 'tools', 'tool_choice' and any built-in tool option, or send "
            "the request without batching."
        )
        raise ApiError(msg)
    if "outputConfig" in request:
        msg = (
            f"{label} {index + 1}: a structured output schema is not available for "
            "batched requests. Remove 'response_format' of type 'json_schema', or "
            "send the request without batching."
        )
        raise ApiError(msg)
    if "guardrailConfig" in request:
        msg = (
            f"{label} {index + 1}: content guardrails are not available for batched "
            "requests. Send this request without batching to keep it guarded."
        )
        raise ApiError(msg)


def _drop_cache_points(blocks: Sequence[Any]) -> list[Any]:
    """Return *blocks* without the cache points it carries."""
    return [block for block in blocks if "cachePoint" not in block]


def _to_model_input(request: ConverseRequestBaseTypeDef) -> dict[str, Any]:
    """Return the batched form of a translated request.

    Prompt caching is dropped rather than refused. A batched request never
    reads or writes a cache, and one carrying a cache point fails outright, so
    keeping the hint would cost the whole batch an answer it can otherwise
    give — at the batch price, which is already the lower one.

    Args:
        request: The translated request.

    Returns:
        The request body a batched invocation accepts.
    """
    model_input: dict[str, Any] = {
        key: value
        for key, value in request.items()
        if key not in ("modelId", "serviceTier", "requestMetadata")
    }
    if system := model_input.get("system"):
        if kept := _drop_cache_points(system):
            model_input["system"] = kept
        else:
            del model_input["system"]
    if messages := model_input.get("messages"):
        model_input["messages"] = [
            {**message, "content": _drop_cache_points(message["content"])}
            for message in messages
        ]
    return model_input


def _batch_model_id(model: str, model_id: str) -> str:
    """Return the identifier a batched job runs *model_id* under.

    Batched requests run on one backend only, so a model this server normally
    reaches through another one runs under the identifier that backend knows
    it by. A model that backend knows under no name at all cannot be batched.

    Args:
        model: Model name as written by the client.
        model_id: Resolved model identifier.

    Returns:
        The identifier to batch, which may name the same model differently.

    Raises:
        ApiError: When no backend that runs batches serves the model.
    """
    if not serves_via_mantle(model_id):
        return model_id
    if twin := runtime_twin(model_id):
        return twin
    msg = (
        f"The model `{model}` cannot run batched requests. Send its requests "
        "without batching, or batch another model."
    )
    raise ApiError(msg)


@contextmanager
def _pinned_models() -> Generator[None]:
    """Pin every model name of the batch being prepared to one resolution.

    A name is written once per request but resolved per request line, and a
    pattern is matched against the whole catalogue on each of those: memoising
    the resolution turns up to one scan per line into one per distinct name,
    and stops a catalogue refresh landing mid-file from letting one pattern
    resolve to two models within a single batch.
    """
    token = _PINNED_MODELS.set({})
    try:
        yield
    finally:
        _PINNED_MODELS.reset(token)


async def _resolve_model(model: str) -> ChatModel:
    """Resolve a model name to the model class a batch can run it with.

    Args:
        model: Model name as written by the client.

    Returns:
        The resolved chat model.

    Raises:
        ApiError: When the model cannot serve batched requests.
    """
    pinned = _PINNED_MODELS.get()
    if isinstance(already := (pinned or {}).get(model), ChatModel):
        return already
    model_id = _batch_model_id(
        model,
        (
            await validate_model(
                model,
                input_modality="TEXT",
                output_modality="TEXT",
                # Every chat route indexes the same models; a pattern in a batch
                # is scoped to them rather than to the whole catalogue.
                route="openai_chat_completion",
            )
        ).id,
    )
    resolved = get_chat_model(model_id)
    if not isinstance(resolved, ChatModel):
        msg = f"The model `{model}` is not available for batched requests."
        raise ApiError(msg)
    if pinned is not None:
        pinned[model] = resolved
    return resolved


async def _resolve_embedding_model(model: str) -> EmbeddingModelBase[Any, Any]:
    """Resolve a model name to the embedding model class a batch can run it with.

    Args:
        model: Model name as written by the client.

    Returns:
        The resolved embedding model.

    Raises:
        ApiError: When the model cannot serve batched requests.
    """
    pinned = _PINNED_MODELS.get()
    if isinstance(already := (pinned or {}).get(model), EmbeddingModelBase):
        return already
    model_id = _batch_model_id(
        model, (await validate_model(model, "EMBEDDING", route="openai_embedding")).id
    )
    resolved = get_embedding_model(model_id)
    if pinned is not None:
        pinned[model] = resolved
    return resolved


@dataclass(slots=True)
class PreparedRequest:
    """One translated request, ready to be written to a job's input.

    Attributes:
        custom_id: Client-chosen identifier of the request.
        model: Model name as written by the client.
        model_id: Resolved backend model identifier.
        model_input: The request body a batched invocation accepts.
    """

    custom_id: str
    model: str
    model_id: str
    model_input: JsonMapping


async def _prepare_openai_request(
    custom_id: str, body: CompletionCreateParams, index: int
) -> PreparedRequest:
    """Translate one chat completion request into its batched form.

    Args:
        custom_id: Client-chosen identifier of the request.
        body: The chat completion parameters.
        index: Zero-based position of the request in the batch.

    Returns:
        The translated request.

    Raises:
        ApiError: When the request asks for something batches cannot serve.
    """
    if body.stream:
        msg = f"Line {index + 1}: 'stream' is not available for batched requests."
        raise ApiError(msg)
    resolved = await _resolve_model(body.model)
    request, _, choices = await resolved.build_completion_request(body)
    if choices > 1:
        msg = (
            f"Line {index + 1}: 'n' must be 1 for batched requests; ask for one "
            "completion per request instead."
        )
        raise ApiError(msg)
    _check_batchable(request, index, "Line")
    return PreparedRequest(
        custom_id, body.model, resolved.model.id, _to_model_input(request)
    )


async def _prepare_embedding_request(
    custom_id: str, body: EmbeddingCreateParams, index: int
) -> PreparedRequest:
    """Translate one embeddings request into its batched form.

    Args:
        custom_id: Client-chosen identifier of the request.
        body: The embedding parameters.
        index: Zero-based position of the request in the batch.

    Returns:
        The translated request.

    Raises:
        ApiError: When the request asks for something batches cannot serve.
    """
    if body.encoding_format == "base64":
        msg = (
            f"Line {index + 1}: 'encoding_format' must be 'float' for batched "
            "requests; ask for the vectors as numbers, or send the request "
            "without batching."
        )
        raise ApiError(msg)
    if GUARDRAIL_CONFIG_VAR.get(None) is not None:
        msg = (
            f"Line {index + 1}: content guardrails are not available for batched "
            "requests. Send this request without batching to keep it guarded."
        )
        raise ApiError(msg)
    model = await _resolve_embedding_model(body.model)
    raw_inputs = body.input if isinstance(body.input, list) else [body.input]
    # `EmbeddingCreateParams._unsupported` rejects any non-str/InputFileUrl item.
    inputs: list[InputFileUrl | str] = raw_inputs  # type: ignore[assignment]
    try:
        model_input = await model.build_batch_request(
            inputs, body.dimensions, get_extra_model_parameters(model.model.id, body)
        )
    except ApiError as exc:
        # The model knows nothing of the file it came from; the position does.
        msg = f"Line {index + 1}: {exc}"
        raise ApiError(msg, status=exc.status) from exc
    return PreparedRequest(custom_id, body.model, model.model.id, model_input)


async def _prepare_anthropic_request(
    entry: MessageBatchRequest, index: int
) -> PreparedRequest:
    """Translate one message request into its batched form.

    Args:
        entry: The batch entry holding the request.
        index: Zero-based position of the request in the batch.

    Returns:
        The translated request.

    Raises:
        ApiError: When the request asks for something batches cannot serve.
    """
    if entry.params.stream:
        msg = f"Request {index + 1}: 'stream' is not available for batched requests."
        raise ApiError(msg)
    resolved = await _resolve_model(entry.params.model)
    request, _ = await resolved.build_message_request(entry.params)
    _check_batchable(request, index, "Request")
    return PreparedRequest(
        entry.custom_id, entry.params.model, resolved.model.id, _to_model_input(request)
    )


async def _resolve_distinct(
    names: set[str], resolve: Callable[[str], Coroutine[Any, Any, ModelBase[Any, Any]]]
) -> list[ModelBase[Any, Any]]:
    """Resolve every model name a batch input writes, a bounded number at a time.

    An input file may name one model per request, so the resolutions run in the
    same waves the translation does rather than all at once, and under a task
    group whose first refusal cancels the wave: a refused file costs one wave
    of lookups, not one per name it holds.

    Args:
        names: The distinct model names the input writes.
        resolve: Coroutine function resolving one name.

    Returns:
        The resolved models.

    Raises:
        ApiError: When a name resolves to no model a batch can run.
    """
    resolved: list[ModelBase[Any, Any]] = []
    ordered = sorted(names)
    for start in range(0, len(ordered), _BUILD_CONCURRENCY):
        try:
            async with TaskGroup() as wave:
                tasks: list[Task[ModelBase[Any, Any]]] = [
                    wave.create_task(resolve(name))
                    for name in ordered[start : start + _BUILD_CONCURRENCY]
                ]
        except BaseExceptionGroup as failures:
            # The refusal itself, not the group the wave wrapped it in: this is
            # the message the client reads.
            first: BaseException = failures
            while isinstance(first, BaseExceptionGroup):
                first = first.exceptions[0]
            raise first from None
        resolved.extend(task.result() for task in tasks)
    return resolved


async def _prepare_all[T](
    items: Sequence[T], prepare: Callable[[T, int], Awaitable[PreparedRequest]]
) -> list[PreparedRequest]:
    """Translate every request, a bounded number at a time.

    Args:
        items: The requests to translate, in input order.
        prepare: Coroutine function translating one request.

    Returns:
        The translated requests, in input order.
    """
    prepared: list[PreparedRequest] = []
    for start in range(0, len(items), _BUILD_CONCURRENCY):
        wave = items[start : start + _BUILD_CONCURRENCY]
        prepared.extend(
            await gather(
                *(prepare(item, start + offset) for offset, item in enumerate(wave))
            )
        )
    return prepared


def _group_by_model(
    prepared: Sequence[PreparedRequest],
) -> dict[str, list[PreparedRequest]]:
    """Group translated requests by the model they resolved to.

    Grouping on the resolved model rather than on the name each request wrote
    keeps two names of one model — an alias, a pattern, the ID itself — in a
    single job instead of two.

    Args:
        prepared: The translated requests, in input order.

    Returns:
        Requests keyed by resolved model ID, each list in input order.

    Raises:
        ApiError: When the batch fans out to more models than allowed, or a
            model carries fewer requests than the backend accepts.
    """
    groups: dict[str, list[PreparedRequest]] = {}
    for item in prepared:
        groups.setdefault(item.model_id, []).append(item)
    if len(groups) > MAX_MODELS_PER_BATCH:
        msg = (
            f"A batch may name at most {MAX_MODELS_PER_BATCH} different models; "
            f"this one names {len(groups)}. Split it into several batches."
        )
        raise ApiError(msg)
    if short := sorted(
        model for model, items in groups.items() if len(items) < MIN_REQUESTS_PER_MODEL
    ):
        counts = ", ".join(f"{model} ({len(groups[model])})" for model in short)
        msg = (
            f"A batch must carry at least {MIN_REQUESTS_PER_MODEL} requests for "
            f"each model it names. Below the minimum: {counts}."
        )
        raise ApiError(msg)
    return groups


async def _delete_object(bucket: str, key: str) -> None:
    """Delete one stored object, best effort."""
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(bucket))
    with suppress(ClientError):
        await s3.delete_object(Bucket=bucket, Key=key)


def _job_name(payload: str, index: int) -> str:
    """Return the backend job name for the *index*-th job of a batch."""
    return f"stdapi-{payload}-{index}"


async def _resolve_job_model(endpoint: str, model: str) -> ModelBase[Any, Any]:
    """Resolve the model class the job serving *endpoint* runs its requests with.

    Args:
        endpoint: API endpoint every request of the job targets.
        model: Model name as written by the client.

    Returns:
        The resolved model.

    Raises:
        ApiError: When the model cannot serve batched requests.
    """
    if endpoint == _EMBEDDINGS_ENDPOINT:
        return await _resolve_embedding_model(model)
    return await _resolve_model(model)


def _invocation_type(endpoint: str) -> ModelInvocationTypeType:
    """Return the invocation a job's requests are written for.

    The requests of each endpoint are written in the shape of one invocation,
    and the two shapes are not interchangeable: a job started for the wrong one
    is accepted, runs its whole window, and fails every single request.

    Args:
        endpoint: API endpoint every request of the job targets.

    Returns:
        The invocation the backend runs each request through.
    """
    return "InvokeModel" if endpoint == _EMBEDDINGS_ENDPOINT else "Converse"


async def _submit_job(
    *,
    payload: str,
    index: int,
    endpoint: str,
    model_id: str,
    items: Sequence[PreparedRequest],
    role_arn: str,
) -> BatchJobRef:
    """Write one model's requests to storage and start its inference job.

    Args:
        payload: Bare 32-char batch payload.
        index: Zero-based position of the job within the batch.
        endpoint: API endpoint every request of the batch targets.
        model_id: Resolved identifier of the model running the job.
        items: The model's translated requests, in input order.
        role_arn: Service role the backend assumes to read and write storage.

    Returns:
        Reference to the started job.

    Raises:
        ApiError: When the model cannot run batched requests.
    """
    # What the caller wrote, to name in anything they read back.
    model = items[0].model
    job_model = await _resolve_job_model(endpoint, model_id)
    region = await job_model.select_region(s3_required=True)
    bucket = require_s3_bucket_for_region(region, feature=_FEATURE)
    try:
        # A model with no inference profile answers its own identifier, which
        # is the only form the backend accepts for it.
        invocation_id = job_model.model.get_id(region, inference_profile=True)
    except ModelRegionUnavailableError as exc:
        msg = f"The model `{model}` is not available for batched requests."
        raise ApiError(msg) from exc
    prefix = f"{SETTINGS.aws_s3_batches_prefix}{payload}/{index}/"
    input_key = f"{prefix}{_INPUT_FILE_NAME}"
    body = b"".join(
        to_json_bytes({"recordId": item.custom_id, "modelInput": item.model_input})
        + b"\n"
        for item in items
    )
    await put_s3_object(body, "application/jsonl", bucket=bucket, key=input_key)
    del body
    client: BedrockClient = get_client("bedrock", region)
    with feature_unavailable_guard(_FEATURE, missing=_CREATE_JOB_PERMISSIONS):
        try:
            response = await client.create_model_invocation_job(
                jobName=_job_name(payload, index),
                roleArn=role_arn,
                modelId=invocation_id,
                modelInvocationType=_invocation_type(endpoint),
                inputDataConfig={
                    "s3InputDataConfig": {
                        "s3Uri": f"s3://{bucket}/{input_key}",
                        "s3InputFormat": "JSONL",
                    }
                },
                outputDataConfig={
                    "s3OutputDataConfig": {"s3Uri": f"s3://{bucket}/{prefix}out/"}
                },
                timeoutDurationInHours=_BATCH_WINDOW_SECONDS // 3600,
            )
        except ClientError as exc:
            # The requests were written before the job could refuse them: drop them.
            schedule_cleanup(_delete_object(bucket, input_key))
            error = exc.response["Error"]
            if error["Code"] == "ValidationException":
                raise _refused_job(
                    model, (invocation_id, job_model.model.id), error["Message"]
                ) from exc
            raise
    job_arn = response["jobArn"]
    return BatchJobRef(
        model=model,
        region=region,
        bucket=bucket,
        job_arn=job_arn,
        job_id=job_arn.rsplit("/", 1)[-1],
        model_id=job_model.model.id,
        requests=len(items),
        prefix=prefix,
    )


def _refused_job(model: str, model_ids: tuple[str, ...], message: str) -> ApiError:
    """Return the error a submission the backend refused as invalid answers with.

    The backend reports every invalid submission as one flat validation
    failure, whatever it was about, so a refusal is read as being about the
    *model* from two signs: it names one of the identifiers the submission
    carried, or it is one of the wordings the backend reserves for a model it
    will not batch. Anything else — a quota, the service role, the storage
    paths — is a deployment problem the caller can neither see nor fix.

    Args:
        model: Model name as written by the client.
        model_ids: Backend model identifiers the submission carried.
        message: The backend's own refusal message.

    Returns:
        An unsupported-model error naming the model, or a feature this
        deployment cannot run, with the cause left in the server log.
    """
    if message.startswith(_MODEL_REFUSALS) or any(
        model_id in message for model_id in model_ids
    ):
        log_error_details(message, level="warning")
        msg = f"The model `{model}` is not available for batched requests."
        return ApiError(msg)
    return FeatureUnavailableError(
        _FEATURE, _REFUSED_JOB_DETAIL.format(message=message)
    )


def _job_client(ref: BatchJobRef) -> BedrockClient:
    """Return the backend client serving *ref*'s region."""
    region: RegionName = ref.region  # type: ignore[assignment]
    client: BedrockClient = get_client("bedrock", region)
    return client


async def _stop_job(ref: BatchJobRef) -> None:
    """Ask the backend to stop one job, ignoring a job that already stopped.

    Any other failure propagates: a cancellation that reached no job must not
    be answered as a success, or the requests run to completion and are billed
    while every poll says the batch is stopping.

    Args:
        ref: The job to stop.

    Raises:
        ClientError: When the backend refused the stop for any reason other
            than the job already being over.
    """
    client = _job_client(ref)
    try:
        await client.stop_model_invocation_job(jobIdentifier=ref.job_arn)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in _ALREADY_STOPPED_ERRORS:
            raise
        log_error_details(exc.response["Error"]["Message"], level="warning")


async def _abandon_jobs(refs: Sequence[BatchJobRef]) -> None:
    """Stop the jobs of a batch that will never exist, and drop their data.

    A job nothing points at runs for its whole window and is billed for it, so
    one that cannot be stopped is named to the operator: it is the only trace
    left of it.

    Args:
        refs: The jobs that were started.
    """
    stops = await gather(*(_stop_job(ref) for ref in refs), return_exceptions=True)
    for ref, stop in zip(refs, stops, strict=True):
        if isinstance(stop, BaseException):
            log_error_details(
                f"Batch inference job {ref.job_id} was started in {ref.region} but "
                f"could not be stopped ({stop}), and no batch names it. Stop it "
                "manually, or it runs and bills until its window ends.",
                level="warning",
            )
    schedule_cleanup(*(_delete_job_data(ref) for ref in refs))


async def _write_record(record: BatchRecord) -> None:
    """Store *record* as the batch's own object."""
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(record.bucket))
    await s3.put_object(
        Bucket=record.bucket,
        Key=batch_s3_key(record.batch_id),
        Body=to_json_bytes(record),
        ContentType="application/json",
    )


async def create_batch(
    *,
    surface: BatchSurface,
    endpoint: str,
    completion_window: str,
    prepared: Sequence[PreparedRequest],
    input_file_id: str | None = None,
    metadata: dict[str, str] | None = None,
    output_expires_after: int | None = None,
) -> BatchState:
    """Group translated requests by model, start a job per model, and store the batch.

    A job that starts while a sibling fails is stopped again, and so is every
    job of a batch whose record cannot be stored: nothing that could not be
    found, cancelled or settled afterwards is left running and billing.

    Args:
        surface: API the batch is created through.
        endpoint: API endpoint every request targets.
        completion_window: Time frame within which the batch is processed.
        prepared: The translated requests, in input order.
        input_file_id: Files API identifier of the submitted requests.
        metadata: Key-value pairs to attach to the batch.
        output_expires_after: Seconds the result files stay readable once
            written, or ``None`` to keep them until they are deleted.

    Returns:
        The created batch and the state of its jobs.

    Raises:
        ApiError: When the requests cannot be grouped into runnable jobs, or
            when the API key carries a tenant AWS credential a batch job
            cannot run under.
    """
    if tenant_aws_credential() is not None:
        # Refused rather than run on this deployment's account: a batch job
        # outlives the one-hour role session role chaining caps a tenant
        # credential at, and would land hours of spend on someone else's bill.
        msg = (
            "The Batch API is not available for API keys that carry an AWS "
            "credential of their own: a batch job cannot run under it. "
            "Use the non-batch endpoints instead."
        )
        raise ApiError(msg)
    role_arn, bucket = require_batches_enabled()
    groups = _group_by_model(prepared)
    payload = encode_id_payload(bucket)
    results = await gather(
        *(
            _submit_job(
                payload=payload,
                index=index,
                endpoint=endpoint,
                model_id=model_id,
                items=items,
                role_arn=role_arn,
            )
            for index, (model_id, items) in enumerate(groups.items())
        ),
        return_exceptions=True,
    )
    started = [item for item in results if isinstance(item, BatchJobRef)]
    if failures := [item for item in results if isinstance(item, BaseException)]:
        await _abandon_jobs(started)
        raise failures[0]

    record = BatchRecord(
        batch_id=payload,
        bucket=bucket,
        surface=surface,
        created_at=payload_created_at(payload),
        endpoint=endpoint,
        completion_window=completion_window,
        input_file_id=input_file_id,
        metadata=metadata,
        output_expires_after=output_expires_after,
        jobs=started,
    )
    try:
        await _write_record(record)
    except BaseException:
        # Unstored, the batch cannot be found, cancelled or settled by anyone.
        await _abandon_jobs(started)
        raise
    return BatchState(
        record, [_pending_state(ref, record.created_at) for ref in started]
    )


def _pending_state(ref: BatchJobRef, created_at: int) -> JobState:
    """Return the state of a job that was just submitted."""
    return JobState(
        ref=ref,
        status="Submitted",
        submitted_at=created_at,
        ended_at=None,
        total=ref.requests,
        succeeded=0,
        errored=0,
    )


async def _read_record(payload: str, surface: BatchSurface) -> BatchRecord:
    """Load the batch record for *payload*.

    Args:
        payload: Bare 32-char batch payload.
        surface: API the batch must have been created through.

    Returns:
        The stored batch record.

    Raises:
        BatchNotFoundError: When no such batch exists on this API.
    """
    require_batches_enabled()
    bucket = resolve_file_bucket(payload)
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(bucket))
    try:
        body = await (await s3.get_object(Bucket=bucket, Key=batch_s3_key(payload)))[
            "Body"
        ].read()
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            raise _not_found() from exc
        raise  # pragma: no cover - defensive
    data = from_json(body)
    if not isinstance(data, dict) or data.get("surface") != surface:
        raise _not_found()
    record = BatchRecord(
        batch_id=payload,
        bucket=bucket,
        surface=surface,
        created_at=int(data["created_at"]),
        endpoint=str(data["endpoint"]),
        completion_window=str(data["completion_window"]),
        input_file_id=data.get("input_file_id"),
        metadata=data.get("metadata"),
        output_expires_after=data.get("output_expires_after"),
        jobs=[BatchJobRef(**job) for job in data.get("jobs", ())],
        cancel_initiated_at=data.get("cancel_initiated_at"),
        deleted=bool(data.get("deleted")),
        output_file_id=data.get("output_file_id"),
        error_file_id=data.get("error_file_id"),
        usage=BatchUsageTotals(**usage) if (usage := data.get("usage")) else None,
    )
    if record.deleted:
        raise _not_found()
    return record


def _not_found() -> BatchNotFoundError:
    """Return the error raised for an unknown batch."""
    return BatchNotFoundError("No batch found with that identifier.")


def _to_job_state(
    ref: BatchJobRef, response: GetModelInvocationJobResponseTypeDef
) -> JobState:
    """Map a backend job description to its live state.

    Args:
        ref: The job reference the description belongs to.
        response: Job description returned by the backend.

    Returns:
        The job's live state.
    """
    total = int(response.get("totalRecordCount") or ref.requests)
    # An unknown status is read as still running rather than as finished.
    status = response.get("status") or "InProgress"
    if status == "Failed" and (message := response.get("message")):
        # The client only ever sees the failed count: the reason lands here.
        log_error_details(
            f"Batch inference job {ref.job_id} failed: {message}. Check the "
            "batch service role set in 'aws_bedrock_batch_role_arn' and its "
            "access to the batch bucket.",
            level="warning",
        )
    return JobState(
        ref=ref,
        status=status,
        submitted_at=int(response["submitTime"].timestamp()),
        ended_at=int(end.timestamp()) if (end := response.get("endTime")) else None,
        total=total,
        succeeded=int(response.get("successRecordCount") or 0),
        errored=int(response.get("errorRecordCount") or 0),
    )


async def _read_job(ref: BatchJobRef) -> JobState:
    """Read one job's live state from the backend.

    Args:
        ref: The job to read.

    Returns:
        The job's live state.

    Raises:
        BatchNotFoundError: When the backend no longer knows the job.
    """
    client = _job_client(ref)
    try:
        response = await client.get_model_invocation_job(jobIdentifier=ref.job_arn)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in _JOB_NOT_FOUND_ERRORS:
            raise _not_found() from exc
        raise  # pragma: no cover - defensive
    return _to_job_state(ref, response)


async def _load_state(record: BatchRecord) -> BatchState:
    """Read the live state of every job behind *record*."""
    return BatchState(record, list(await gather(*(_read_job(r) for r in record.jobs))))


async def get_batch(payload: str, surface: BatchSurface) -> BatchState:
    """Return a batch and the live state of its jobs.

    Args:
        payload: Bare 32-char batch payload.
        surface: API the batch must have been created through.

    Returns:
        The batch state.

    Raises:
        BatchNotFoundError: When no such batch exists on this API.
    """
    return await _load_state(await _read_record(payload, surface))


async def cancel_batch(payload: str, surface: BatchSurface) -> BatchState:
    """Ask every job behind a batch to stop.

    Asking twice is accepted and changes nothing, and a batch that has already
    ended keeps the state it ended in: cancellation never rewrites an outcome
    the requests already reached.

    Args:
        payload: Bare 32-char batch payload.
        surface: API the batch must have been created through.

    Returns:
        The batch state after cancellation was asked for.

    Raises:
        BatchNotFoundError: When no such batch exists on this API.
    """
    record = await _read_record(payload, surface)
    state = await _load_state(record)
    if state.ended:
        return state
    await gather(*(_stop_job(ref) for ref in record.jobs))
    if record.cancel_initiated_at is None:
        record.cancel_initiated_at = now_utc_timestamp()
        await _write_record(record)
    return await _load_state(record)


async def delete_batch(payload: str, surface: BatchSurface) -> None:
    """Delete a batch that has ended, and the data it holds.

    Args:
        payload: Bare 32-char batch payload.
        surface: API the batch must have been created through.

    Raises:
        ApiError: When the batch has not ended yet.
        BatchNotFoundError: When no such batch exists on this API.
    """
    state = await get_batch(payload, surface)
    if not state.ended:
        msg = (
            "Only a batch that has ended can be deleted. Cancel it first, then "
            "delete it once it has ended."
        )
        raise ApiError(msg)
    record = state.record
    record.deleted = True
    await _write_record(record)
    await gather(
        *(_delete_job_data(ref) for ref in record.jobs), return_exceptions=True
    )


async def _delete_job_data(ref: BatchJobRef) -> None:
    """Delete the stored requests and results of one job."""
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(ref.bucket))
    token: str | None = None
    while True:
        page = await s3.list_objects_v2(
            Bucket=ref.bucket,
            Prefix=ref.prefix,
            MaxKeys=1000,
            **({"ContinuationToken": token} if token else {}),  # type: ignore[arg-type]
        )
        keys: list[ObjectIdentifierTypeDef] = [
            {"Key": obj["Key"]} for obj in page.get("Contents", ())
        ]
        if keys:
            await s3.delete_objects(Bucket=ref.bucket, Delete={"Objects": keys})
        token = page.get("NextContinuationToken")
        if not token:
            return


def _instant_key(created_ms: int) -> str:
    """Return the lowest record key a batch created at *created_ms* can hold.

    A payload opens with the 48-bit millisecond timestamp of its UUIDv7 and is
    encoded in the order-preserving base32hex alphabet, so zero-filling what
    follows that timestamp names where an instant starts in key order.

    Args:
        created_ms: Creation instant, in milliseconds since the epoch.

    Returns:
        The key every batch created at *created_ms* or later sorts at or after.
    """
    return batch_s3_key(
        b32hexencode(created_ms.to_bytes(6, "big") + bytes(14)).lower().decode()
    )


async def _walk_tail(
    s3: S3Client, bucket: str, start_after: str | None, budget: int
) -> tuple[list[str], bool, int]:
    """Walk the records stored after *start_after*, keeping the newest of them.

    The delimiter rolls each batch's own data up under its own key prefix, so
    what a page carries is batch records rather than the objects one batch
    stores.

    Args:
        s3: Client for the bucket's region.
        bucket: The bucket to read.
        start_after: Key to resume after, or ``None`` to walk from the start.
        budget: Maximum storage requests this walk may make.

    Returns:
        Tuple of (at most ``_LIST_SCAN_LIMIT`` payloads oldest first, whether
        the end of the listing was reached, storage requests made).
    """
    prefix = SETTINGS.aws_s3_batches_prefix
    payloads: list[str] = []
    token: str | None = None
    for request in range(1, budget + 1):
        resume: dict[str, str] = {}
        if token:
            resume = {"ContinuationToken": token}
        elif start_after:
            resume = {"StartAfter": start_after}
        response = await s3.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            Delimiter="/",
            MaxKeys=_LIST_SCAN_LIMIT,
            **resume,  # type: ignore[arg-type]
        )
        payloads.extend(
            payload
            for obj in response.get("Contents", ())
            if len(payload := obj["Key"].removeprefix(prefix)) == 32
        )
        del payloads[:-_LIST_SCAN_LIMIT]
        token = response.get("NextContinuationToken")
        if not token:
            return payloads, True, request
    return payloads, False, budget


async def _scan_bucket(bucket: str) -> list[str]:
    """Return the newest batch payloads stored in *bucket*.

    Payloads sort by creation time and storage only ever walks them oldest
    first, so reaching the newest ones means seeking into the key space rather
    than paging to the end of it: paging forward answers from the oldest
    records as soon as the bucket outgrows the budget that paging is given,
    which is when a client most needs the newest.

    A bucket holding no more than one page is answered by that one request.
    Past that, each probe resumes after the key an instant starts at, and the
    instant moves until one probe reaches the end of the listing while still
    holding records — later when the probe was left with more to walk than its
    pages cover, earlier when it reached too few records. A probe that reached
    the end holds *every* record newer than its instant, which is what makes
    the answer a tail rather than a prefix of the oldest keys.

    The first instant probed is a recent one, and reaching further back costs
    one request per step where giving an instant up costs a walk: a scan pays
    for guessing too far back, never for guessing too close.

    Args:
        bucket: The bucket to scan.

    Returns:
        At most ``_LIST_SCAN_LIMIT`` payloads, oldest first.
    """
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(bucket))
    budget = _LIST_SCAN_REQUESTS
    reached, complete, used = await _walk_tail(s3, bucket, None, 1)
    if complete:
        return reached
    budget -= used
    # A record created this second sorts after a key built from this second, so
    # the seek's upper bound is the next one; the listing overflows what one
    # walk covers, so its start is already known to be too far back.
    now_ms = (now_utc_timestamp() + 1) * 1000
    span = _LIST_SEEK_SPAN_MS
    crowded, sparse = 0, now_ms
    newest: list[str] = []
    while budget > 0:
        if crowded:
            instant = (crowded + sparse) // 2
            if instant in (crowded, sparse):
                break
        else:
            # Until one probe comes back crowded the span grows instead of
            # being halved: a quiet recent past is otherwise walked back to the
            # epoch one halving at a time.
            instant = now_ms - span
            span *= _LIST_SEEK_GROWTH
            if instant <= 0:
                break
        payloads, complete, used = await _walk_tail(
            s3, bucket, _instant_key(instant), min(_LIST_PROBE_PAGES, budget)
        )
        budget -= used
        if not complete:
            # Not a tail, but newer than anything walked before it: the answer
            # if no probe reaches the end of the listing within the budget. The
            # bound moves to the last record the walk reached rather than to
            # the instant it started from, so a burst denser than one probe
            # covers is crossed by the records each probe reads instead of by
            # halvings alone -- which the budget runs out of first.
            if payloads:
                instant = min(
                    max(instant, payload_created_at(payloads[-1]) * 1000), sparse
                )
            crowded, reached = instant, payloads
            continue
        if len(payloads) > len(newest):
            newest = payloads
        if len(newest) >= _LIST_SCAN_LIMIT:
            # A complete probe already answers with the most this scan can
            # ever return, so seeking further back would only spend budget.
            break
        sparse = instant
    return newest or reached


async def list_batches(
    surface: BatchSurface,
    *,
    after: str | None = None,
    before: str | None = None,
    limit: int = 20,
) -> tuple[list[BatchState], bool]:
    """List this server's batches, newest first.

    Batches order on their identifier, which is also where each one reads the
    creation time it reports, so paging order and reported times agree.

    Args:
        surface: API the batches must have been created through.
        after: Return the page of batches created just before this payload.
        before: Return the page of batches created just after this payload.
        limit: Maximum number of batches to return.

    Returns:
        Tuple of (page of batches, whether more batches remain).
    """
    require_batches_enabled()
    payloads = sorted(
        chain.from_iterable(await gather(*(_scan_bucket(b) for b in BUCKET_TO_REGION))),
        reverse=True,
    )
    if after:
        payloads = payloads[payloads.index(after) + 1 :] if after in payloads else []
    # A `before` cursor pages adjacent to it, so newer batches walk outwards.
    backwards = False
    if before is not None and before in payloads:
        backwards = True
        payloads = payloads[: payloads.index(before)][::-1]
    states: list[BatchState] = []
    has_more = False
    for start in range(0, len(payloads), limit + 1):
        wave = payloads[start : start + limit + 1]
        records = await gather(
            *(_read_record(p, surface) for p in wave), return_exceptions=True
        )
        for record in records:
            if isinstance(record, BaseException) and not isinstance(
                record, BatchNotFoundError
            ):
                raise record
        states.extend(
            await gather(
                *(_load_state(r) for r in records if isinstance(r, BatchRecord))
            )
        )
        if len(states) > limit:
            has_more = True
            break
    page = states[:limit]
    return (page[::-1] if backwards else page), has_more


async def finish_listed(
    states: Sequence[BatchState], finish: Callable[[BatchState], Awaitable[BatchState]]
) -> list[BatchState]:
    """Store the outcome of every listed batch that has just ended.

    A listing is the only view a client that never retrieves a batch has of
    it, so it settles and publishes exactly as a retrieval does — otherwise
    the batch's usage is never recorded. A batch still running, or whose
    outcome is already stored, costs no backend call at all: *finish* returns
    it untouched, so the work is proportional to the batches that ended since
    the last read rather than to the size of the page.

    One batch that cannot be settled is reported to the operator and listed as
    it stands rather than failing the whole page: nothing has been claimed, so
    the next read settles it.

    Args:
        states: The batches of the page, in listing order.
        finish: Coroutine function storing one batch's outcome.

    Returns:
        The batches, in listing order.
    """
    finished: list[BatchState] = []
    for start in range(0, len(states), _FINISH_CONCURRENCY):
        wave = states[start : start + _FINISH_CONCURRENCY]
        results = await gather(*map(finish, wave), return_exceptions=True)
        for state, result in zip(wave, results, strict=True):
            if isinstance(result, BaseException):
                log_error_details(
                    f"Batch {state.record.batch_id} could not be settled while "
                    f"listing: {result}.",
                    level="warning",
                )
            finished.append(state if isinstance(result, BaseException) else result)
    return finished


async def _read_manifest(ref: BatchJobRef) -> JsonMapping:
    """Read one job's aggregate counters.

    Args:
        ref: The job to read the counters of.

    Returns:
        The counters, or an empty mapping when the job wrote none.

    Raises:
        ClientError: When the counters exist but could not be read — settling
            on a failed read would burn the one-shot billing claim on zeros.
    """
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(ref.bucket))
    key = f"{ref.prefix}out/{ref.job_id}/{_MANIFEST_SUFFIX}"
    try:
        body = await (await s3.get_object(Bucket=ref.bucket, Key=key))["Body"].read()
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in _NO_SUCH_OBJECT_ERRORS:
            raise
        log_error_details(
            f"Batch job {ref.job_id} wrote no usage counters.", level="warning"
        )
        return {}
    data = from_json(body)
    return data if isinstance(data, dict) else {}


async def _claim_billing(record: BatchRecord) -> bool:
    """Claim the right to record this batch's usage exactly once.

    Args:
        record: The batch record.

    Returns:
        Whether this caller won the claim.
    """
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(record.bucket))
    try:
        await s3.put_object(
            Bucket=record.bucket,
            Key=f"{batch_s3_key(record.batch_id)}.billed",
            Body=b"",
            IfNoneMatch="*",
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] in _ALREADY_CLAIMED_ERRORS:
            return False
        # Recording usage twice beats losing the entry to a lost claim.
        log_error_details(exc.response["Error"]["Message"], level="warning")
    return True


def _record_media_usage(ref: BatchJobRef, manifest: JsonMapping) -> None:
    """Record the media one embedding job consumed, beyond its tokens.

    A multimodal embedding model is billed per image and per second of media,
    exactly as the synchronous embeddings path records them; a batch of them
    recording tokens alone would report no cost at all. Only embedding jobs
    report media this way — a conversation's images are billed as tokens, and
    counting them again would bill them twice.

    Args:
        ref: The job the counters belong to.
        manifest: The job's aggregate counters.
    """
    for key, spec in _IMAGE_COUNTERS:
        if count := _counter(manifest, key):
            record_bedrock_usage(
                ref.model_id,
                tier="batch",
                region=ref.region,
                input_images=count,
                media_spec=spec,
                billed_externally=False,
            )
    for key, spec in _SECOND_COUNTERS:
        if count := _counter(manifest, key):
            record_bedrock_usage(
                ref.model_id,
                tier="batch",
                region=ref.region,
                input_seconds=count,
                media_spec=spec,
                billed_externally=False,
            )


def _counter(manifest: JsonMapping, key: str) -> int:
    """Return one counter of a job's manifest, or zero when it reports none."""
    value = manifest.get(key)
    return int(value) if isinstance(value, (int, float)) else 0


async def settle(state: BatchState) -> BatchState:
    """Total a finished batch's usage, record it, and store the totals.

    Does nothing while the batch is still running, or once it has been
    settled. The usage is recorded exactly once, by whichever read observes
    the end of the batch first.

    Args:
        state: The batch state.

    Returns:
        The batch state, with its usage totals filled in.
    """
    record = state.record
    if not state.ended or record.usage is not None:
        return state
    manifests = await gather(*(_read_manifest(ref) for ref in record.jobs))
    totals = BatchUsageTotals()
    billed = await _claim_billing(record)
    embeddings = record.endpoint == _EMBEDDINGS_ENDPOINT
    for ref, manifest in zip(record.jobs, manifests, strict=True):
        counts = {
            key: int(value)
            for key in (
                "inputTokenCount",
                "outputTokenCount",
                "cacheReadInputTokenCount",
                "cacheWriteInputTokenCount",
            )
            if isinstance(value := manifest.get(key), (int, float))
        }
        totals.input_tokens += counts.get("inputTokenCount", 0)
        totals.output_tokens += counts.get("outputTokenCount", 0)
        totals.cached_tokens += counts.get("cacheReadInputTokenCount", 0)
        if billed:
            # Always this deployment's own spend: the job ran on its account,
            # whatever key reads the results back.
            record_bedrock_usage(
                ref.model_id,
                tier="batch",
                region=ref.region,
                input_tokens=counts.get("inputTokenCount", 0),
                output_tokens=counts.get("outputTokenCount", 0),
                cached_tokens=counts.get("cacheReadInputTokenCount", 0),
                cache_write_tokens=counts.get("cacheWriteInputTokenCount", 0),
                billed_externally=False,
            )
            if embeddings:
                _record_media_usage(ref, manifest)
    record.usage = totals
    await _write_record(record)
    return state


async def _iter_job_output(ref: BatchJobRef) -> AsyncIterator[JsonMapping]:
    """Yield one job's result lines, in the order the backend wrote them.

    Args:
        ref: The job to read the results of.

    Yields:
        One decoded result line at a time.

    Raises:
        ClientError: When the results exist but could not be read — publishing
            a truncated results file would be permanent.
    """
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(ref.bucket))
    key = f"{ref.prefix}out/{ref.job_id}/{_INPUT_FILE_NAME}.out"
    try:
        body = (await s3.get_object(Bucket=ref.bucket, Key=key))["Body"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in _NO_SUCH_OBJECT_ERRORS:
            raise
        return
    async for line in _iter_jsonl(body.iter_chunks()):
        decoded = from_json(line)
        if isinstance(decoded, dict):
            yield decoded


async def _iter_job_input_ids(ref: BatchJobRef) -> AsyncIterator[str]:
    """Yield the ``custom_id`` of every request submitted to one job.

    Args:
        ref: The job to read the requests of.

    Yields:
        One identifier at a time, in input order.
    """
    s3: S3Client = get_client("s3", BUCKET_TO_REGION.get(ref.bucket))
    key = f"{ref.prefix}{_INPUT_FILE_NAME}"
    try:
        body = (await s3.get_object(Bucket=ref.bucket, Key=key))["Body"]
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in _NO_SUCH_OBJECT_ERRORS:
            raise
        return
    async for line in _iter_jsonl(body.iter_chunks()):
        decoded = from_json(line)
        if isinstance(decoded, dict) and (custom_id := decoded.get("recordId")):
            yield str(custom_id)


def _record_error(line: JsonMapping) -> tuple[str, str] | None:
    """Return the client-facing error of a failed result line.

    The backend's own wording names its internals, so only its status class
    reaches the client; the original is kept in the server log.

    Args:
        line: One decoded result line.

    Returns:
        Tuple of (error code, message), or ``None`` when the line succeeded.
    """
    error = line.get("error")
    if not isinstance(error, dict):
        return None
    if detail := error.get("errorMessage"):
        log_error_details(str(detail), level="warning")
    if error.get("expired"):
        return "expired", "The request expired before it could be run."
    code = error.get("errorCode")
    return _RECORD_ERRORS.get(
        int(code) if isinstance(code, (int, str)) and str(code).isdigit() else 0,
        _RECORD_ERROR_DEFAULT,
    )


async def _to_completion(
    line: JsonMapping, model_id: str, custom_id: str, created: int
) -> JsonMapping:
    """Translate one successful result line into a chat completion body.

    Args:
        line: One decoded result line.
        model_id: Model that produced the answer.
        custom_id: Client-chosen identifier of the request.
        created: Unix timestamp to report as the completion time.

    Returns:
        The chat completion, as a JSON mapping.
    """
    output: ConverseResponseTypeDef = _strip_nulls(line.get("modelOutput") or {})
    completion = await openai_adapter.format_response(
        f"chatcmpl-{custom_id}",
        created,
        model_id,
        [output],
        None,
        None,
        openai_adapter.DEFAULT_OUTPUT_MODALITIES,  # type: ignore[arg-type]
    )
    return completion.model_dump(exclude_none=True)


async def _to_embeddings(line: JsonMapping, model_id: str) -> JsonMapping:
    """Translate one successful result line into an embeddings body.

    Args:
        line: One decoded result line.
        model_id: Model that produced the vectors.

    Returns:
        The embeddings response, as a JSON mapping.
    """
    output = line.get("modelOutput")
    response = get_embedding_model(model_id).read_batch_response(
        output if isinstance(output, dict) else {}
    )
    embeddings = await build_embedding_response(response, model_id, b64_embedding=False)
    return embeddings.model_dump(exclude_none=True)


async def _to_message(line: JsonMapping, model: str, custom_id: str) -> JsonMapping:
    """Translate one successful result line into a message body.

    Args:
        line: One decoded result line.
        model: Resolved backend model identifier the message ran on.
        custom_id: Client-chosen identifier of the request.

    Returns:
        The message, as a JSON mapping.
    """
    output: ConverseResponseTypeDef = _strip_nulls(line.get("modelOutput") or {})
    message = await anthropic_adapter.format_response(
        output.get("output", {}).get("message", {}).get("content", []),
        output.get("stopReason"),
        output.get("usage", {}),
        f"msg_{custom_id}",
        model,
        None,
        lambda *_: None,
        service_tier="batch",
    )
    return message.model_dump(exclude_none=True)


async def iter_openai_results(
    record: BatchRecord, *, errors: bool
) -> AsyncIterator[bytes]:
    """Yield a batch's results as the lines of an OpenAI results file.

    Args:
        record: The batch record.
        errors: Yield the requests that failed instead of those that succeeded.

    Yields:
        One JSONL line at a time.
    """
    embeddings = record.endpoint == _EMBEDDINGS_ENDPOINT
    for ref in record.jobs:
        async for line in _iter_job_output(ref):
            custom_id = str(line.get("recordId", ""))
            error = _record_error(line)
            if (error is not None) != errors:
                continue
            line_id = _result_line_id(record.batch_id, custom_id)
            if error is None:
                body = (
                    await _to_embeddings(line, ref.model_id)
                    if embeddings
                    else await _to_completion(
                        line, ref.model_id, custom_id, record.created_at
                    )
                )
                payload: JsonMapping = {
                    "id": line_id,
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 200,
                        "request_id": line_id,
                        "body": body,
                    },
                    "error": None,
                }
            else:
                payload = {
                    "id": line_id,
                    "custom_id": custom_id,
                    "response": None,
                    "error": {"code": error[0], "message": error[1]},
                }
            yield to_json_bytes(payload) + b"\n"


async def iter_anthropic_results(
    record: BatchRecord, *, canceled: bool
) -> AsyncIterator[bytes]:
    """Yield a batch's results as the lines of a Message Batch results file.

    Every request that produced an outcome keeps it, cancelled batch or not.

    Args:
        record: The batch record.
        canceled: Also report the requests that produced no outcome as
            canceled, for a batch that was stopped before they ran.

    Yields:
        One JSONL line at a time.
    """
    for ref in record.jobs:
        answered: set[str] = set()
        async for line in _iter_job_output(ref):
            custom_id = str(line.get("recordId", ""))
            error = _record_error(line)
            if error is None:
                result: JsonMapping = {
                    "type": "succeeded",
                    "message": await _to_message(line, ref.model_id, custom_id),
                }
            elif error[0] == "expired":
                result = {"type": "expired"}
            else:
                result = {
                    "type": "errored",
                    "error": {
                        "type": "error",
                        "error": {"type": error[0], "message": error[1]},
                    },
                }
            if canceled:
                answered.add(custom_id)
            yield to_json_bytes({"custom_id": custom_id, "result": result}) + b"\n"
        if not canceled:
            continue
        async for custom_id in _iter_job_input_ids(ref):
            if custom_id not in answered:
                yield (
                    to_json_bytes(
                        {"custom_id": custom_id, "result": {"type": "canceled"}}
                    )
                    + b"\n"
                )


async def materialize_openai_results(state: BatchState) -> BatchState:
    """Write a finished batch's results to the Files API and name them on the batch.

    Writing the same batch's results twice writes the same objects, so a
    second reader observing the end of the batch changes nothing.  The
    expiration the batch was created with is applied to both files, counted
    from the moment they are written.

    Args:
        state: The batch state.

    Returns:
        The batch state, with its results named.
    """
    record = state.record
    if not state.ended or state.failed or record.output_file_id is not None:
        # A failed batch has no results; an empty file would mislead the client.
        return state
    bucket = record.bucket
    expires_after = record.output_expires_after
    output_payload = _derive_file_payload(record.batch_id, bucket, _OUTPUT_FILE_MARKER)
    await put_file_content(
        output_payload,
        bucket,
        iter_openai_results(record, errors=False),
        filename=f"batch_{record.batch_id}_output.jsonl",
        purpose="batch_output",
        expires_after=expires_after,
    )
    record.output_file_id = f"file-{output_payload}"
    if state.errored:
        error_payload = _derive_file_payload(
            record.batch_id, bucket, _ERROR_FILE_MARKER
        )
        await put_file_content(
            error_payload,
            bucket,
            iter_openai_results(record, errors=True),
            filename=f"batch_{record.batch_id}_error.jsonl",
            purpose="batch_output",
            expires_after=expires_after,
        )
        record.error_file_id = f"file-{error_payload}"
    await _write_record(record)
    return state


async def read_input_requests(file_id: str) -> list[JsonMapping]:
    """Read and decode the requests of a batch input file.

    Args:
        file_id: Files API identifier of the input file.

    Returns:
        The decoded request lines, in input order.

    Raises:
        ApiError: When the file was not uploaded for batching, holds no
            request, or holds a line that is not a JSON object.
    """
    payload = parse_file_id(file_id)
    record = await get_file(payload)
    if record.purpose != "batch":
        msg = (
            f"File '{file_id}' was uploaded with purpose '{record.purpose}'. "
            "Upload the requests with purpose 'batch'."
        )
        raise ApiError(msg)
    if record.size > _MAX_INPUT_FILE_BYTES:
        msg = (
            f"A batch input file may be at most "
            f"{_MAX_INPUT_FILE_BYTES // 1024**2} MB; '{file_id}' is larger."
        )
        raise ApiError(msg)
    content, _ = await get_file_content(payload)
    maximum = MAX_REQUESTS["openai"]
    lines: list[JsonMapping] = []
    async for index, line in _enumerate(_iter_jsonl(content)):
        if index >= maximum:
            # Refused mid-decode, so an oversized file is never materialised in memory.
            msg = (
                f"A batch may carry at most {maximum} requests; '{file_id}' "
                f"carries more."
            )
            raise ApiError(msg)
        try:
            decoded = from_json(line)
        except ValueError as exc:
            msg = f"Line {index + 1}: each line must be a JSON object."
            raise ApiError(msg) from exc
        if not isinstance(decoded, dict):
            msg = f"Line {index + 1}: each line must be a JSON object."
            raise ApiError(msg)
        lines.append(decoded)
    if not lines:
        msg = f"File '{file_id}' holds no request."
        raise ApiError(msg)
    return lines


async def _enumerate[T](iterator: AsyncIterator[T]) -> AsyncIterator[tuple[int, T]]:
    """Yield ``(index, item)`` pairs from an asynchronous iterator."""
    index = 0
    async for item in iterator:
        yield index, item
        index += 1


async def prepare_openai_requests(
    lines: Sequence[JsonMapping], endpoint: str
) -> list[PreparedRequest]:
    """Validate and translate the request lines of a batch input file.

    Args:
        lines: The decoded request lines, in input order.
        endpoint: API endpoint every request must target.

    Returns:
        The translated requests, in input order.

    Raises:
        ApiError: When a line is malformed, targets another endpoint, or names
            a model another line does not.
    """
    if len(lines) > MAX_REQUESTS["openai"]:
        msg = (
            f"A batch may carry at most {MAX_REQUESTS['openai']} requests; this "
            f"one carries {len(lines)}."
        )
        raise ApiError(msg)
    embeddings = endpoint == _EMBEDDINGS_ENDPOINT
    params = EmbeddingCreateParams if embeddings else CompletionCreateParams
    bodies: list[CompletionCreateParams | EmbeddingCreateParams] = []
    custom_ids: list[str] = []
    for index, line in enumerate(lines):
        url = line.get("url")
        if url != endpoint:
            msg = (
                f"Line {index + 1}: 'url' must be '{endpoint}', the endpoint the "
                f"batch targets."
            )
            raise ApiError(msg)
        if (method := line.get("method", "POST")) != "POST":
            msg = f"Line {index + 1}: 'method' must be 'POST', not {method!r}."
            raise ApiError(msg)
        body = line.get("body")
        if not isinstance(body, dict):
            msg = f"Line {index + 1}: 'body' must be a JSON object."
            raise ApiError(msg)
        with validation_error_handler():
            bodies.append(params.model_validate(body))
        custom_ids.append(str(line.get("custom_id", "")))
    _validate_custom_ids(custom_ids, "Line")
    # Resolved once per distinct name — never the full file — and compared
    # before translating a single request: two names of one model are one
    # model, and a file naming more than one is refused before it pays for
    # translating any of its lines.
    resolve = _resolve_embedding_model if embeddings else _resolve_model
    with _pinned_models():
        resolved = await _resolve_distinct({body.model for body in bodies}, resolve)
        if len({model.model.id for model in resolved}) > 1:
            msg = (
                "Every request in a batch input file must name the same model. "
                "Split the file into one file per model."
            )
            raise ApiError(msg)
        return await _prepare_all(
            list(zip(custom_ids, bodies, strict=True)),
            (lambda item, index: _prepare_embedding_request(item[0], item[1], index))  # type: ignore[arg-type]
            if embeddings
            else (lambda item, index: _prepare_openai_request(item[0], item[1], index)),  # type: ignore[arg-type]
        )


async def prepare_anthropic_requests(
    requests: Sequence[MessageBatchRequest],
) -> list[PreparedRequest]:
    """Validate and translate the requests of a Message Batch.

    Args:
        requests: The requests, in input order.

    Returns:
        The translated requests, in input order.

    Raises:
        ApiError: When the batch carries too many requests, or a duplicate
            ``custom_id``.
    """
    if len(requests) > MAX_REQUESTS["anthropic"]:
        msg = (
            f"A batch may carry at most {MAX_REQUESTS['anthropic']} requests; "
            f"this one carries {len(requests)}."
        )
        raise ApiError(msg)
    _validate_custom_ids([entry.custom_id for entry in requests], "Request")
    with _pinned_models():
        return await _prepare_all(list(requests), _prepare_anthropic_request)


def rfc3339(timestamp: int | None) -> str | None:
    """Format a Unix timestamp as an RFC 3339 string, or return ``None``."""
    if timestamp is None:
        return None
    return (
        datetime.fromtimestamp(timestamp, UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


__all__ = [
    "MAX_MODELS_PER_BATCH",
    "MAX_REQUESTS",
    "MIN_REQUESTS_PER_MODEL",
    "BatchNotFoundError",
    "BatchRecord",
    "BatchState",
    "BatchSurface",
    "cancel_batch",
    "create_batch",
    "delete_batch",
    "finish_listed",
    "get_batch",
    "iter_anthropic_results",
    "iter_openai_results",
    "list_batches",
    "materialize_openai_results",
    "prepare_anthropic_requests",
    "prepare_openai_requests",
    "read_input_requests",
    "require_batches_enabled",
    "rfc3339",
    "settle",
]
