"""In-memory doubles for the Batch API's storage and job backend.

The Batch API keeps a record object per batch and reads every status figure
back from the inference jobs, so a unit test needs both an object store and a
job service. Both are faked here, faithfully enough that the record round-trip,
the ownership scoping and the results translation are the code under test.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from botocore.exceptions import ClientError

from stdapi import batches
from stdapi.models.chat._default import ChatModel

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import pytest

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef

#: Region every fake job runs in.
REGION = "us-east-1"

#: Bucket every fake object is stored in.
BUCKET = "test-bucket"

#: Batch service role the fake backend accepts.
ROLE_ARN = "arn:aws:iam::123456789012:role/stdapi-ai-batch"


class _Body:
    """The streaming body an S3 ``GetObject`` answers with."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        """Return the whole object."""
        return self._data

    async def iter_chunks(self) -> AsyncIterator[bytes]:
        """Yield the object in two chunks, so line splitting is exercised."""
        middle = len(self._data) // 2
        yield self._data[:middle]
        yield self._data[middle:]


def _client_error(code: str, operation: str) -> ClientError:
    """Build the ``ClientError`` botocore raises for *code*."""
    return ClientError({"Error": {"Code": code, "Message": code}}, operation)  # type: ignore[arg-type]


#: Error codes the fakes raise, named so the raise sites stay one line.
_PRECONDITION = "PreconditionFailed"
_NO_SUCH_KEY = "NoSuchKey"
_VALIDATION = "ValidationException"
_NOT_FOUND = "ResourceNotFoundException"


class FakeS3:
    """An object store holding every written object in memory."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    async def put_object(
        self,
        *,
        Bucket: str,  # noqa: N803
        Key: str,  # noqa: N803
        Body: bytes = b"",  # noqa: N803
        IfNoneMatch: str | None = None,  # noqa: N803
        **_kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        """Store an object, honouring the create-only precondition."""
        if IfNoneMatch == "*" and (Bucket, Key) in self.objects:
            raise _client_error(_PRECONDITION, "PutObject")
        self.objects[Bucket, Key] = Body
        return {}

    async def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        """Return an object's body, or raise the not-found error."""
        try:
            data = self.objects[Bucket, Key]
        except KeyError:
            raise _client_error(_NO_SUCH_KEY, "GetObject") from None
        return {"Body": _Body(data)}

    async def list_objects_v2(
        self,
        *,
        Bucket: str,  # noqa: N803
        Prefix: str = "",  # noqa: N803
        MaxKeys: int = 1000,  # noqa: N803
        ContinuationToken: str | None = None,  # noqa: N803
        **_kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        """List one page of *Bucket* under *Prefix*, in key order.

        Paging is honoured: ``MaxKeys`` bounds the page and the continuation
        token resumes after the last key returned, so a caller's paging loop
        runs more than once.
        """
        keys = [
            key
            for (bucket, key) in sorted(self.objects)
            if bucket == Bucket
            and key.startswith(Prefix)
            and (ContinuationToken is None or key > ContinuationToken)
        ]
        page = keys[:MaxKeys]
        response: dict[str, Any] = {"Contents": [{"Key": key} for key in page]}
        if page and len(keys) > MaxKeys:
            response["NextContinuationToken"] = page[-1]
        return response

    async def delete_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        """Delete one object."""
        self.objects.pop((Bucket, Key), None)
        return {}

    async def delete_objects(
        self,
        *,
        Bucket: str,  # noqa: N803
        Delete: dict[str, Any],  # noqa: N803
    ) -> dict[str, Any]:
        """Delete the listed objects."""
        for entry in Delete["Objects"]:
            self.objects.pop((Bucket, entry["Key"]), None)
        return {}


class FakeBedrock:
    """A job service recording submissions and reporting a settable status."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.created: list[dict[str, Any]] = []
        self.stopped: list[str] = []
        self.reject_models: set[str] = set()
        self.stop_error: str | None = None

    async def create_model_invocation_job(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Register a job and return its ARN."""
        self.created.append(kwargs)
        if kwargs["modelId"] in self.reject_models:
            raise _client_error(_VALIDATION, "CreateModelInvocationJob")
        job_id = f"job{len(self.jobs)}"
        arn = f"arn:aws:bedrock:{REGION}:123456789012:model-invocation-job/{job_id}"
        self.jobs[arn] = {
            "jobArn": arn,
            "modelId": kwargs["modelId"],
            "roleArn": kwargs["roleArn"],
            "status": "Submitted",
            "submitTime": datetime(2026, 8, 12, tzinfo=UTC),
            "inputDataConfig": kwargs["inputDataConfig"],
            "outputDataConfig": kwargs["outputDataConfig"],
        }
        return {"jobArn": arn}

    async def get_model_invocation_job(self, *, jobIdentifier: str) -> dict[str, Any]:  # noqa: N803
        """Return a job's description, or raise the not-found error."""
        try:
            return self.jobs[jobIdentifier]
        except KeyError:
            raise _client_error(_NOT_FOUND, "GetModelInvocationJob") from None

    async def stop_model_invocation_job(self, *, jobIdentifier: str) -> dict[str, Any]:  # noqa: N803
        """Record a stop request and move the job to ``Stopping``.

        Stopping is asynchronous, so the job does not reach ``Stopped`` here;
        and a job that already reached a terminal state is refused, which is
        what keeps a finished job's outcome from being rewritten.
        """
        self.stopped.append(jobIdentifier)
        if self.stop_error:
            raise _client_error(self.stop_error, "StopModelInvocationJob")
        job = self.jobs.get(jobIdentifier)
        if job is None:
            raise _client_error(_NOT_FOUND, "StopModelInvocationJob")
        if job["status"] in batches._TERMINAL_STATUSES:  # noqa: SLF001
            raise _client_error(_VALIDATION, "StopModelInvocationJob")
        job["status"] = "Stopping"
        return {}

    def start(self, *, arn: str | None = None) -> None:
        """Move one job — or every job, by default — to ``InProgress``.

        Args:
            arn: The job to move; every job when omitted.
        """
        for job_arn, job in self.jobs.items():
            if arn in (None, job_arn):
                job["status"] = "InProgress"

    def finish(
        self,
        *,
        arn: str | None = None,
        status: str = "Completed",
        succeeded: int = 100,
        errored: int = 0,
    ) -> None:
        """Move one job — or every job, by default — to a terminal state.

        Args:
            arn: The job to move; every job when omitted.
            status: Terminal status the job reports.
            succeeded: Requests the job answered.
            errored: Requests the job failed.
        """
        for job_arn, job in self.jobs.items():
            if arn not in (None, job_arn):
                continue
            job["status"] = status
            job["endTime"] = datetime(2026, 8, 12, 1, tzinfo=UTC)
            job["totalRecordCount"] = succeeded + errored
            job["successRecordCount"] = succeeded
            job["errorRecordCount"] = errored


class _OfflineChatModel(ChatModel):
    """Chat model with the two calls that would reach AWS replaced."""

    __slots__ = ()

    async def select_region(self, *, s3_required: bool = False) -> Any:  # noqa: ANN401
        """Return the single fake region."""
        return REGION

    @property
    def model(self) -> Any:  # noqa: ANN401
        """Return canned details, so no live catalog is needed."""
        from tests._helpers import make_model_details  # noqa: PLC0415

        return make_model_details(self._model_id)


class StubChatModel(_OfflineChatModel):
    """Chat model translating a request into a minimal, recognisable payload."""

    __slots__ = ()

    async def build_completion_request(
        self,
        request: Any,  # noqa: ANN401
    ) -> tuple[ConverseRequestBaseTypeDef, None, int]:
        """Return a Converse request carrying the first message's text."""
        return _stub_request(request), None, request.n or 1

    async def build_message_request(
        self,
        request: Any,  # noqa: ANN401
    ) -> tuple[ConverseRequestBaseTypeDef, None]:
        """Return a Converse request carrying the first message's text."""
        return _stub_request(request), None


class TranslatingChatModel(_OfflineChatModel):
    """Chat model running the real request translation, prompt caching included.

    A batch built through it carries the blocks a real request produces, which
    is what a test asserting on the submitted body needs.
    """

    __slots__ = ()

    #: Honor the cache hints a client sends, as a caching-capable model does.
    PROMPT_CACHING_SUPPORTED: ClassVar[bool] = True

    #: Cache the tool definitions too, so every cache-point site is reachable.
    PROMPT_CACHING_TOOL_SUPPORTED: ClassVar[bool] = True


def _stub_request(request: Any) -> ConverseRequestBaseTypeDef:  # noqa: ANN401
    """Build the Converse request a stubbed translation produces."""
    built: dict[str, Any] = {
        "modelId": "",
        "messages": [{"role": "user", "content": [{"text": str(request.messages)}]}],
        "inferenceConfig": {},
    }
    if getattr(request, "tools", None):
        built["toolConfig"] = {"tools": []}
    if getattr(request, "response_format", None) is not None and (
        getattr(request.response_format, "type", "") == "json_schema"
    ):
        built["outputConfig"] = {"textFormat": {}}
    return built  # type: ignore[return-value]


def install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    models: dict[str, list[str]] | None = None,
    translate: bool = False,
) -> tuple[FakeS3, FakeBedrock]:
    """Point the Batch API at the in-memory store and job service.

    Args:
        monkeypatch: The test's patcher.
        models: Optional map of model name to output modalities, for a model
            the batch must refuse; every other name resolves to a text model.
        translate: Run the real request translation instead of the stub one,
            for a test asserting on the body a batch submits.

    Returns:
        Tuple of (object store, job service).
    """
    from stdapi.config import SETTINGS  # noqa: PLC0415
    from tests._helpers import make_model_details  # noqa: PLC0415

    s3 = FakeS3()
    bedrock = FakeBedrock()
    monkeypatch.setattr(SETTINGS, "aws_s3_bucket", BUCKET)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_batch_role_arn", ROLE_ARN)
    monkeypatch.setattr(batches, "BUCKET_TO_REGION", {BUCKET: REGION})
    monkeypatch.setattr(batches, "resolve_file_bucket", lambda _payload: BUCKET)
    monkeypatch.setattr(batches, "require_s3_bucket_for_region", lambda _region: BUCKET)
    monkeypatch.setattr(
        batches,
        "get_client",
        lambda service, _region=None: s3 if service == "s3" else bedrock,
    )

    async def _validate_model(model_id: str, *_args: object, **_kwargs: object) -> Any:  # noqa: ANN401
        return make_model_details(
            model_id, output_modalities=(models or {}).get(model_id, ["TEXT"])
        )

    async def _put_s3_object(
        data: bytes, _content_type: str | None = None, **kwargs: str
    ) -> None:
        await s3.put_object(Bucket=kwargs["bucket"], Key=kwargs["key"], Body=data)

    async def _put_file_content(
        payload: str, bucket: str, data: bytes | AsyncIterator[bytes], **_kwargs: object
    ) -> None:
        body = data if isinstance(data, bytes) else b"".join([c async for c in data])
        await s3.put_object(
            Bucket=bucket, Key=f"{SETTINGS.aws_s3_files_prefix}{payload}", Body=body
        )

    monkeypatch.setattr(batches, "validate_model", _validate_model)
    monkeypatch.setattr(
        batches, "get_chat_model", TranslatingChatModel if translate else StubChatModel
    )
    monkeypatch.setattr(batches, "serves_via_mantle", lambda _model_id: False)
    monkeypatch.setattr(batches, "put_s3_object", _put_s3_object)
    monkeypatch.setattr(batches, "put_file_content", _put_file_content)
    return s3, bedrock


def capture_translations(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every translated request on its way into the batched form.

    Args:
        monkeypatch: The test's patcher.

    Returns:
        The requests as translated, filled in as the batch is prepared.
    """
    seen: list[Any] = []
    original = batches._to_model_input  # noqa: SLF001

    def _record(request: Any) -> Any:  # noqa: ANN401
        seen.append(request)
        return original(request)

    monkeypatch.setattr(batches, "_to_model_input", _record)
    return seen


def install_input_file(
    monkeypatch: pytest.MonkeyPatch, lines: list[str], *, purpose: str = "batch"
) -> str:
    """Make the Files API answer a batch input file holding *lines*.

    Args:
        monkeypatch: The test's patcher.
        lines: The JSONL lines of the file.
        purpose: The purpose the file was uploaded with.

    Returns:
        The file identifier to submit the batch with.
    """
    from stdapi.files import FileRecord  # noqa: PLC0415

    file_id = "file-0123456789abcdefghijklmnopqrstuv"
    body = ("\n".join(lines)).encode()

    async def _get_file(_payload: str) -> FileRecord:
        return FileRecord(
            file_id=file_id[5:],
            filename="requests.jsonl",
            content_type="application/jsonl",
            purpose=purpose,
            size=len(body),
            created_at=datetime(2026, 8, 12, tzinfo=UTC),
            expires_at=None,
        )

    async def _get_file_content(_payload: str) -> tuple[AsyncIterator[bytes], str]:
        async def _stream() -> AsyncIterator[bytes]:
            yield body

        return _stream(), "application/jsonl"

    monkeypatch.setattr(batches, "get_file", _get_file)
    monkeypatch.setattr(batches, "get_file_content", _get_file_content)
    return file_id


def write_job_output(
    s3: FakeS3,
    bedrock: FakeBedrock,
    lines: list[dict[str, Any]],
    manifest: dict[str, int],
    *,
    arn: str | None = None,
) -> None:
    """Write the results and the counters one — or every — fake job produced.

    Args:
        s3: The object store.
        bedrock: The job service holding the jobs.
        lines: The result lines, as the backend writes them.
        manifest: The aggregate counters the job reports.
        arn: The job to write for; every job when omitted.
    """
    from stdapi.utils import to_json_bytes  # noqa: PLC0415

    for job_arn, job in bedrock.jobs.items():
        if arn not in (None, job_arn):
            continue
        prefix = job["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"].split("/", 3)[3]
        job_id = job_arn.rsplit("/", 1)[-1]
        s3.objects[BUCKET, f"{prefix}{job_id}/input.jsonl.out"] = b"".join(
            to_json_bytes(line) + b"\n" for line in lines
        )
        s3.objects[BUCKET, f"{prefix}{job_id}/manifest.json.out"] = to_json_bytes(
            manifest
        )


def converse_output(text: str) -> dict[str, Any]:
    """Return a result line's ``modelOutput``, with the unions the backend nulls out."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "text": text,
                        "image": None,
                        "toolUse": None,
                        "toolResult": None,
                        "reasoningContent": None,
                        "citationsContent": None,
                    }
                ],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 5,
            "outputTokens": 3,
            "totalTokens": 8,
            "cacheReadInputTokens": None,
            "cacheWriteInputTokens": None,
        },
        "metrics": {"latencyMs": 120},
    }


def chat_lines(
    count: int,
    *,
    model: str = "amazon.nova-micro-v1:0",
    prefix: str = "req",
    body: str = "",
) -> list[str]:
    """Return *count* well-formed chat completion request lines.

    Args:
        count: Number of lines.
        model: Model every line names.
        prefix: Prefix of each line's ``custom_id``.
        body: Replacement request body, for a line carrying more than the text.

    Returns:
        The JSONL lines.
    """
    body = body or (
        f'{{"model": "{model}", "messages": [{{"role": "user", "content": "hi"}}]}}'
    )
    return [
        f'{{"custom_id": "{prefix}-{index}", "method": "POST", '
        f'"url": "/v1/chat/completions", "body": {body}}}'
        for index in range(count)
    ]


def read_result_file(s3: FakeS3, file_id: str) -> list[dict[str, Any]]:
    """Return the decoded lines of the stored Files API object named by *file_id*."""
    from pydantic_core import from_json  # noqa: PLC0415

    from stdapi.config import SETTINGS  # noqa: PLC0415

    body = s3.objects[BUCKET, f"{SETTINGS.aws_s3_files_prefix}{file_id[5:]}"]
    return [from_json(line) for line in body.splitlines()]
