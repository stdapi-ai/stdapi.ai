"""Vector store indexing handed to a queue, so another server can finish it.

Indexing runs in the task that served the attach request. That task can be
killed at any moment — a platform sends ``SIGKILL`` a fixed delay after
``SIGTERM``, and a deployment need not use the published module at all — and
what it was embedding dies with it: the file settles as ``failed`` on the next
read and the caller has to attach it again. The shutdown drain makes that
outcome less likely; nothing makes it recoverable.

A queue does. The job is a message, so a server that never saw the request can
run it, and the work survives the instance that accepted it.

What the design rests on, and what breaks if it is changed:

- **The send is the last thing the request does.** Every file record, the store
  counters and the batch record are already written under conditional writes,
  and the file bytes have been durable since they were uploaded — so the job is
  replayable from the moment it is sent. Sending from a background task would
  reintroduce exactly the window this closes.
- **The message is a pointer**: a store, its files, its batch, the request they
  came from. No caller content is ever copied into a second store at rest, and
  there is nothing in the message to inject with.
- **The consumer runs in this process**, alongside the request handlers, on
  every instance. A dedicated consumer service would bill an extra container
  per deployment for a feature most never enable, and would have to be made
  redundant on its own.
- **Redelivery is the recovery mechanism**, not an accident of the transport.
  Standard queues are at-least-once, and a job replayed after a kill is how the
  work gets finished, so every handler here has to be idempotent.
- **Whatever comes back off the queue is data, never instruction.** Only this
  server's own role can write to it, and it is still parsed into a model that
  forbids unknown fields, dispatched through an allowlist fixed at import, and
  re-validated identifier by identifier before any of it names an object key.
"""

from asyncio import CancelledError, Task, create_task, sleep, wait
from contextlib import suppress
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal

from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.cleanup import drain_tasks
from stdapi.config import SETTINGS, SQS_QUEUE_URL_RE
from stdapi.files import parse_file_id
from stdapi.monitoring import (
    REQUEST_ID,
    add_server_warning,
    log_error_details,
    requests_in_flight,
)
from stdapi.utils import try_parse_json
from stdapi.vector_stores.engine import (
    index_files,
    parse_batch_id,
    parse_store_id,
    renew_indexing_lease,
    settle_interrupted,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.monitoring import EventLog

#: The job type an indexing wave is published under.
_INDEX_FILES: Final = "vector_store.index_files"

#: Files one job may name, so a single message cannot fan out without bound.
MAX_JOB_FILES: Final[int] = 500

#: Characters an identifier carried by a job may take before it is even parsed.
_MAX_ID_LENGTH: Final[int] = 128

#: Seconds a receive waits for a message before answering empty.
_RECEIVE_WAIT_SECONDS: Final[int] = 20

#: Messages one receive asks for; the work behind one is minutes long.
_RECEIVE_BATCH: Final[int] = 1

#: Seconds a message stays invisible, renewed for as long as the job runs.
_VISIBILITY_SECONDS: Final[int] = 60

#: Seconds between two renewals, short enough to survive a slow renewal.
_HEARTBEAT_SECONDS: Final[int] = 20

#: Requests in flight above which the consumer stops asking for work.
_BUSY_REQUESTS: Final[int] = 8

#: Seconds the consumer waits before testing the load again.
_BUSY_WAIT_SECONDS: Final[float] = 1.0

#: Seconds the consumer waits after a queue error, so a broken queue cannot spin.
_ERROR_WAIT_SECONDS: Final[float] = 5.0

#: Deliveries a job gets when the queue names no dead-letter queue.
_DEFAULT_MAX_RECEIVES: Final[int] = 3

#: Deliveries the queue's own redrive policy allows, read at startup.
_MAX_RECEIVES: int = _DEFAULT_MAX_RECEIVES

#: Whether the queue names a dead-letter queue to move an exhausted job to.
_HAS_DEAD_LETTER_QUEUE: bool = False

#: The receive loop, while one is running.
_CONSUMER: Task[None] | None = None

#: Jobs being run right now, awaited by the shutdown drain.
_JOB_TASKS: Final[set[Task[None]]] = set()

#: Whether the consumer has been asked to stop receiving.
_STOPPING: bool = False


def queue_region() -> RegionName | None:
    """Return the region of the configured indexing queue.

    Derived rather than configured: a queue URL names the endpoint that serves
    it, so a second setting could only disagree with it.

    Returns:
        The region, or ``None`` when no queue is configured.
    """
    if not (url := SETTINGS.aws_sqs_vector_store_queue_url):
        return None
    # The URL was matched by the settings validator, so this one matches too.
    match = SQS_QUEUE_URL_RE(url)
    region: RegionName = match["region"]  # type: ignore[index, assignment]
    return region


def _client() -> Any:  # noqa: ANN401 -- the pooled botocore client is untyped
    """Return the pooled client of the queue's own region."""
    return get_client("sqs", queue_region())


def _validated(parse: Callable[[str], str], value: str) -> str:
    """Return *value* as *parse* accepts it, as a value error when it does not.

    Args:
        parse: The identifier validator the API itself uses.
        value: The identifier read out of a message.

    Returns:
        The identifier, unchanged.

    Raises:
        ValueError: When the identifier is not one this server would mint.
    """
    if len(value) > _MAX_ID_LENGTH:
        msg = "identifier is too long"
        raise ValueError(msg)
    try:
        parse(value)
    except ApiError as exc:
        raise ValueError(str(exc)) from exc
    return value


class IndexFilesJob(BaseModel):
    """One indexing wave, as it travels between two servers.

    Identifiers only: which store, which of its files, which batch they were
    attached in, and the request that attached them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["vector_store.index_files"]
    store_id: str
    file_ids: list[str] = Field(min_length=1, max_length=MAX_JOB_FILES)
    batch_id: str = ""
    #: Correlates the work with the request that asked for it, across servers.
    request_id: str = Field(default="", max_length=64, pattern=r"^[A-Za-z0-9_.:-]*$")

    @field_validator("store_id")
    @classmethod
    def _check_store_id(cls, value: str) -> str:
        """Re-validate the store identifier, which becomes an object key prefix."""
        return _validated(parse_store_id, value)

    @field_validator("file_ids")
    @classmethod
    def _check_file_ids(cls, value: list[str]) -> list[str]:
        """Re-validate every file identifier, which becomes an object key."""
        return [_validated(parse_file_id, entry) for entry in value]

    @field_validator("batch_id")
    @classmethod
    def _check_batch_id(cls, value: str) -> str:
        """Re-validate the batch identifier, which becomes an object key."""
        return _validated(parse_batch_id, value) if value else value


async def _run_index_files(job: IndexFilesJob) -> None:
    """Index the files of *job*, whichever server originally accepted them.

    The lease is renewed first: a job that waited in the queue longer than the
    lease its attach wrote would otherwise be settled as abandoned by the very
    store read that starts it.

    Args:
        job: The indexing wave to run.
    """
    await renew_indexing_lease(job.store_id)
    await index_files(job.store_id, list(job.file_ids), job.batch_id, job.request_id)


async def _settle_index_files(job: IndexFilesJob) -> None:
    """Fail the files of an indexing wave that will not be attempted again.

    Args:
        job: The indexing wave being given up on.
    """
    await settle_interrupted(job.store_id, job.file_ids)


@dataclass(slots=True, frozen=True)
class _JobHandler:
    """What one job type is parsed into, run by, and given up on with.

    Attributes:
        model: The model the message body must validate against.
        run: What performs the job.
        settle: What answers for the job's records when it is abandoned.
    """

    model: type[Any]
    run: Callable[[Any], Awaitable[None]]
    settle: Callable[[Any], Awaitable[None]]


#: The job types this server runs, fixed at import: a message names one or none.
_HANDLERS: Final[Mapping[str, _JobHandler]] = MappingProxyType(
    {
        _INDEX_FILES: _JobHandler(
            model=IndexFilesJob, run=_run_index_files, settle=_settle_index_files
        )
    }
)


def _sanitized_request_id(request_id: str) -> str:
    """Return *request_id* reduced to what a job may carry.

    Args:
        request_id: The current request's identifier.

    Returns:
        The identifier, stripped of anything outside the job model's charset
        and truncated, so a correlation value can never fail a send.
    """
    return "".join(
        character
        for character in request_id
        if character.isascii() and (character.isalnum() or character in "_.:-")
    )[:64]


async def enqueue_indexing(
    store_id: str, file_ids: Sequence[str], batch_id: str
) -> bool:
    """Publish an indexing wave, so any server can run it.

    Args:
        store_id: A validated vector store identifier.
        file_ids: The files to index, in order.
        batch_id: The batch the files belong to, or ``""``.

    Returns:
        Whether the job was handed over. ``False`` when no queue is configured,
        when the wave is larger than one message may name, or when the send
        failed — in every case the caller indexes the wave itself, which is
        what a deployment without a queue always does.
    """
    if not (url := SETTINGS.aws_sqs_vector_store_queue_url):
        return False
    if len(file_ids) > MAX_JOB_FILES:
        log_error_details(
            f"Vector store indexing of {len(file_ids)} files exceeds the "
            f"{MAX_JOB_FILES} a job may carry: indexing it in this server, "
            "where it is lost if the server stops first.",
            level="warning",
        )
        return False
    job = IndexFilesJob(
        type=_INDEX_FILES,
        store_id=store_id,
        file_ids=list(file_ids),
        batch_id=batch_id,
        request_id=_sanitized_request_id(REQUEST_ID.get("vector_store")),
    )
    try:
        await _client().send_message(QueueUrl=url, MessageBody=job.model_dump_json())
    except (BotoCoreError, ClientError, OSError) as exc:
        # Never the caller's failure: the files are attached either way, and
        # indexing them here is exactly what an unqueued deployment does.
        log_error_details(
            "Vector store indexing job could not be queued "
            f"({type(exc).__name__}): check the server role's sqs:SendMessage "
            "permission on aws_sqs_vector_store_queue_url. Indexing in this "
            "server instead, where it is lost if the server stops first.",
            level="error",
        )
        return False
    return True


async def initialize_job_queue(start_event: EventLog) -> None:
    """Read what the indexing queue promises, and report what it does not.

    Reported and never fatal, exactly as the other startup probes are: a queue
    this server cannot describe still accepts and delivers messages, and a
    server refusing to start would turn one missing permission into an outage.

    Args:
        start_event: Startup log event a shortcoming is reported on.
    """
    global _MAX_RECEIVES, _HAS_DEAD_LETTER_QUEUE  # noqa: PLW0603

    _MAX_RECEIVES = _DEFAULT_MAX_RECEIVES
    _HAS_DEAD_LETTER_QUEUE = False
    if not (url := SETTINGS.aws_sqs_vector_store_queue_url):
        return
    try:
        attributes = (
            await _client().get_queue_attributes(
                QueueUrl=url, AttributeNames=["RedrivePolicy"]
            )
        ).get("Attributes", {})
    except (BotoCoreError, ClientError, OSError) as exc:
        add_server_warning(
            start_event,
            "The vector store indexing queue could not be described "
            f"({type(exc).__name__}): check that "
            "'aws_sqs_vector_store_queue_url' names an existing queue and that "
            "the server role has sqs:GetQueueAttributes on it",
        )
        return
    policy = try_parse_json(attributes.get("RedrivePolicy", ""))
    if not isinstance(policy, dict) or not policy.get("deadLetterTargetArn"):
        add_server_warning(
            start_event,
            "The vector store indexing queue has no dead-letter queue: a file "
            f"that cannot be indexed is given up on after {_DEFAULT_MAX_RECEIVES} "
            "attempts and its message dropped. Give the queue a redrive policy "
            "to keep those messages for inspection",
        )
        return
    _HAS_DEAD_LETTER_QUEUE = True
    count = policy.get("maxReceiveCount")
    if isinstance(count, int) and count > 0:
        _MAX_RECEIVES = count


def open_job_consumer() -> None:
    """Start consuming indexing jobs, when a queue is configured.

    No-op without one, and idempotent, so a second lifespan in one process
    neither leaves the previous loop running nor refuses to start a new one.
    """
    global _CONSUMER, _STOPPING  # noqa: PLW0603

    _STOPPING = False
    if _CONSUMER is not None or not SETTINGS.aws_sqs_vector_store_queue_url:
        return
    _CONSUMER = create_task(_consume())


def close_job_consumer() -> None:
    """Stop asking the queue for work, leaving the jobs already running.

    The receive it is sitting in lasts as long as the long poll, which is
    longer than a shutdown has, so the loop is cancelled rather than asked.
    """
    global _CONSUMER, _STOPPING  # noqa: PLW0603

    _STOPPING = True
    if _CONSUMER is not None:
        _CONSUMER.cancel()
        _CONSUMER = None


async def drain_indexing_jobs(timeout: float) -> int:  # noqa: ASYNC109 -- shared drain contract
    """Await the queued jobs still running, and report what the deadline left.

    Nothing is settled here, unlike the drain of the indexing this server owns:
    a job cut short was never deleted from the queue, so it comes back to
    whichever server is still up. That is the whole point of the queue.

    Args:
        timeout: Seconds allowed before the unfinished jobs are cancelled.

    Returns:
        Number of jobs that had not finished at the deadline.
    """
    return await drain_tasks(_JOB_TASKS, timeout)


async def _consume() -> None:
    """Receive jobs and run them, one at a time, until asked to stop."""
    url = SETTINGS.aws_sqs_vector_store_queue_url
    while not _STOPPING:
        # The only priority asyncio offers is declining to compete: a task busy
        # answering clients does not also ask the queue for minutes of work.
        if requests_in_flight() > _BUSY_REQUESTS:
            await sleep(_BUSY_WAIT_SECONDS)
            continue
        try:
            messages = (
                await _client().receive_message(
                    QueueUrl=url,
                    MaxNumberOfMessages=_RECEIVE_BATCH,
                    WaitTimeSeconds=_RECEIVE_WAIT_SECONDS,
                    VisibilityTimeout=_VISIBILITY_SECONDS,
                    MessageSystemAttributeNames=["ApproximateReceiveCount"],
                )
            ).get("Messages", ())
        except (BotoCoreError, ClientError, OSError) as exc:
            log_error_details(
                f"Vector store indexing queue could not be read ({exc!r}).",
                level="error",
            )
            await sleep(_ERROR_WAIT_SECONDS)
            continue
        for message in messages:
            task = create_task(_run_job(message))
            _JOB_TASKS.add(task)
            task.add_done_callback(_JOB_TASKS.discard)
            # Waited on rather than awaited: a cancellation here must stop the
            # loop, never the job, which the drain still has to finish.
            await wait({task})


async def _run_job(message: dict[str, Any]) -> None:
    """Run one message, and decide what the queue does with it next.

    Args:
        message: The message as received.
    """
    receipt: str = message.get("ReceiptHandle", "")
    deliveries = _delivery_count(message)
    handler, job = _parse_job(message.get("Body", ""))
    if handler is None or job is None:
        # Nothing will ever make this message valid; keeping it only costs
        # another delivery of the same rejection.
        await _delete(receipt)
        return
    if deliveries > _MAX_RECEIVES:
        # Only reachable on a queue with no dead-letter queue to move it to.
        await _give_up(handler, job, deliveries)
        await _delete(receipt)
        return
    heartbeat = create_task(_keep_invisible(receipt))
    try:
        await handler.run(job)
    except Exception as exc:  # noqa: BLE001
        # Deliberately every exception, not the expected few: one that escaped
        # here left the job neither retried to a conclusion nor given up, so
        # the file stayed "in_progress" for ever while the message worked its
        # way to the dead-letter queue -- the one outcome this job type exists
        # to prevent. A file reported failed is recoverable; one pending for
        # ever is not.
        log_error_details(
            f"Vector store indexing job failed ({exc!r}); delivery "
            f"{deliveries} of {_MAX_RECEIVES}.",
            level="error",
        )
        if deliveries >= _MAX_RECEIVES:
            await _give_up(handler, job, deliveries)
        # Left on the queue: another delivery, or the dead-letter queue.
        return
    finally:
        heartbeat.cancel()
        with suppress(CancelledError):
            await heartbeat
    await _delete(receipt)


def _delivery_count(message: dict[str, Any]) -> int:
    """Return how many times *message* has been delivered, at least once."""
    try:
        return max(1, int(message.get("Attributes", {})["ApproximateReceiveCount"]))
    except KeyError, TypeError, ValueError:
        return 1


def _parse_job(body: str) -> tuple[_JobHandler | None, Any]:
    """Resolve a message body to a handler and the job it validates as.

    The type names an entry of a mapping built at import; it never selects
    code by any other means, so a message can only ask for what is already
    there.

    Args:
        body: The raw message body.

    Returns:
        ``(handler, job)``, or ``(None, None)`` when the body is not a job this
        server runs.
    """
    payload = try_parse_json(body)
    kind = payload.get("type") if isinstance(payload, dict) else None
    handler = _HANDLERS.get(kind) if isinstance(kind, str) else None
    if handler is None:
        log_error_details(
            "Vector store indexing queue delivered a message that is not a job "
            "this server runs; dropping it.",
            level="error",
        )
        return None, None
    try:
        return handler, handler.model.model_validate(payload)
    except ValidationError as exc:
        log_error_details(
            f"Vector store indexing job is malformed ({exc.error_count()} "
            "invalid fields); dropping it.",
            level="error",
        )
        return None, None


async def _give_up(handler: _JobHandler, job: Any, deliveries: int) -> None:  # noqa: ANN401 -- one of the job models
    """Settle the records of a job that has run out of deliveries.

    Args:
        handler: The handler that answers for the job.
        job: The job being abandoned.
        deliveries: How many times it had been delivered.
    """
    log_error_details(
        f"Vector store indexing job abandoned after {deliveries} deliveries: "
        "its files are reported as failed and must be attached again."
        + (
            " Its message is kept in the dead-letter queue."
            if _HAS_DEAD_LETTER_QUEUE
            else " Give the queue a dead-letter queue to keep its message."
        ),
        level="error",
    )
    with suppress(ApiError, BotoCoreError, ClientError, OSError):
        await handler.settle(job)


async def _keep_invisible(receipt: str) -> None:
    """Hold a message invisible for as long as its job runs.

    The base visibility is short on purpose: a server killed seconds into a job
    must not hide it for the length of the work it never did.

    Args:
        receipt: Receipt handle of the message being worked on.
    """
    url = SETTINGS.aws_sqs_vector_store_queue_url
    while True:
        await sleep(_HEARTBEAT_SECONDS)
        with suppress(BotoCoreError, ClientError, OSError):
            await _client().change_message_visibility(
                QueueUrl=url,
                ReceiptHandle=receipt,
                VisibilityTimeout=_VISIBILITY_SECONDS,
            )


async def _delete(receipt: str) -> None:
    """Take a finished message off the queue.

    A failure here is a redelivery, which every handler is written to survive.

    Args:
        receipt: Receipt handle of the message to delete.
    """
    with suppress(BotoCoreError, ClientError, OSError):
        await _client().delete_message(
            QueueUrl=SETTINGS.aws_sqs_vector_store_queue_url, ReceiptHandle=receipt
        )
