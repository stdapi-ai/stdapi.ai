"""OpenAI-compatible Audio Transcription API implementation using AWS Transcribe."""

from asyncio import gather, sleep
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from botocore.exceptions import ClientError
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import Response
from pydantic_core import from_json
from sse_starlette import EventSourceResponse, JSONServerSentEvent

from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.models import (
    EXTRA_MODELS,
    EXTRA_MODELS_INPUT_MODALITY,
    EXTRA_MODELS_OUTPUT_MODALITY,
    ModelDetails,
)
from stdapi.monitoring import (
    REQUEST_ID,
    REQUEST_LOG,
    log_background_event,
    log_error_details,
    log_request_params,
    log_request_stream_event,
    log_response_params,
)
from stdapi.openai_exceptions import OpenaiError, OpenaiUnsupportedModelError
from stdapi.tokenizer import estimate_token_count
from stdapi.types.openai_audio import (
    AudioResponseFormat,
    AudioTimestampGranularities,
    ChunkingStrategy,
    Transcription,
    TranscriptionCreateParams,
    TranscriptionCreateResponse,
    TranscriptionSegment,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
    TranscriptionVerbose,
    TranscriptionWord,
    UsageDuration,
    UsageInputTokenDetails,
    UsageTokens,
)
from stdapi.utils import (
    format_language_code,
    language_code_to_name,
    validation_error_handler,
)

if TYPE_CHECKING:
    from typing import NotRequired

    from types_aiobotocore_s3.client import S3Client
    from types_aiobotocore_transcribe.client import TranscribeServiceClient
    from types_aiobotocore_transcribe.type_defs import (
        StartTranscriptionJobRequestTypeDef,
    )
    from typing_extensions import TypedDict

    class TranscriptionJobItem(TypedDict):
        """AWS Transcribe transcript item structure."""

        type: str
        alternatives: list[dict[str, str]]
        start_time: str
        end_time: str

    class TranscriptionJobAudioSegment(TypedDict):
        """AWS Transcribe audio segment structure."""

        id: int
        start_time: str
        end_time: str
        transcript: str

    class TranscriptionJobTranscript(TypedDict):
        """AWS Transcribe transcript result structure."""

        transcript: str

    class TranscriptionJobData(TypedDict, total=False):
        """AWS Transcribe job result data structure."""

        transcripts: list[TranscriptionJobTranscript]
        audio_segments: list[TranscriptionJobAudioSegment]
        items: list[TranscriptionJobItem]
        language_code: str
        subtitle_content: NotRequired[str]


from stdapi.aws import get_client

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1/audio", tags=["audio", "openai"]
)

# Subtitles formats
SUBTITLE_FORMATS: set[Literal["srt", "vtt"]] = {"srt", "vtt"}

#: Transcribe model ID
TRANSCRIBE_MODEL_ID = "amazon.transcribe"


async def initialize_transcribe_models() -> None:
    """Initialize extra models."""
    transcribe: TranscribeServiceClient = get_client("transcribe")
    EXTRA_MODELS_INPUT_MODALITY.setdefault("SPEECH", set()).add(TRANSCRIBE_MODEL_ID)
    EXTRA_MODELS_OUTPUT_MODALITY.setdefault("TEXT", set()).add(TRANSCRIBE_MODEL_ID)
    EXTRA_MODELS[TRANSCRIBE_MODEL_ID] = ModelDetails(
        id=TRANSCRIBE_MODEL_ID,
        name="Transcribe",
        provider="Amazon",
        region=transcribe.meta.region_name,
        service="AWS Transcribe",
        input_modalities=["SPEECH"],
        output_modalities=["TEXT"],
    )


class InvalidLanguageFormatError(OpenaiError):
    """Exception raised when language format is invalid."""

    code = "invalid_language_format"


class TranscriptionError(Exception):
    """Exception raised when transcription fails."""


def _build_transcription_job_params(
    job_id: str,
    s3_bucket: str,
    language: str | None,
    response_format: AudioResponseFormat,
) -> "StartTranscriptionJobRequestTypeDef":
    """Build transcription job parameters.

    Args:
        job_id: Unique job identifier
        s3_bucket: S3 bucket name
        language: Optional language code
        response_format: Response format for transcription

    Returns:
        Job parameters for AWS Transcribe
    """
    s3_prefix = SETTINGS.aws_s3_tmp_prefix
    job_params: StartTranscriptionJobRequestTypeDef = {
        "TranscriptionJobName": job_id,
        "Media": {"MediaFileUri": f"s3://{s3_bucket}/{s3_prefix}{job_id}/input"},
        "OutputBucketName": s3_bucket,
        "OutputKey": f"{s3_prefix}{job_id}/output.json",
    }

    if language:
        job_params["LanguageCode"] = format_language_code(language)  # type: ignore[typeddict-item]
    else:
        job_params["IdentifyLanguage"] = True

    if response_format in SUBTITLE_FORMATS:
        job_params["Subtitles"] = {
            "Formats": [response_format],  # type: ignore[list-item]
            "OutputStartIndex": 1,
        }
        # AWS Transcribe will create subtitle file at: {s3_prefix}{job_id}/output.{format}

    return job_params


@contextmanager
def _handle_transcription_error(language: str | None) -> Generator[None]:
    """Context manager to handle transcription job start errors.

    Args:
        language: Language code that may have caused the error

    Raises:
        HTTPException: With appropriate error message

    Usage:
        with _handle_transcription_error(language):
            await transcribe.start_transcription_job(**job_params)
    """
    try:
        yield
    except ClientError as error:
        if error.response["Error"]["Code"] == "BadRequestException":
            error_message = error.response["Error"]["Message"]
            if "languageCode" in error_message:
                msg = (f"Language '{language}' is not supported by the model",)
                raise InvalidLanguageFormatError(msg) from error
            if "file" in error_message:
                raise HTTPException(status_code=400, detail=error_message) from error
        raise  # pragma: no cover


async def _wait_for_transcription_completion(
    transcribe: "TranscribeServiceClient", job_id: str
) -> None:
    """Wait for transcription job to complete.

    Args:
        transcribe: Transcribe service client
        job_id: Transcription job ID

    Raises:
        HTTPException: If transcription fails
    """
    while True:  # Timeout at FastAPI level
        job = (await transcribe.get_transcription_job(TranscriptionJobName=job_id))[
            "TranscriptionJob"
        ]
        if job["TranscriptionJobStatus"] == "COMPLETED":
            break
        if job["TranscriptionJobStatus"] == "FAILED":
            raise HTTPException(status_code=400, detail=job["FailureReason"])
        await sleep(0.5)


async def _get_transcription_results(
    s3_client: "S3Client",
    s3_bucket: str,
    job_id: str,
    response_format: AudioResponseFormat,
) -> "TranscriptionJobData":
    """Get transcription results from S3.

    Args:
        s3_client: S3 client
        s3_bucket: S3 bucket name
        job_id: Job identifier
        response_format: Response format

    Returns:
        Transcription data
    """
    s3_prefix = SETTINGS.aws_s3_tmp_prefix
    s3_output_key = f"{s3_prefix}{job_id}/output.json"

    if response_format in SUBTITLE_FORMATS:
        data, subtitle = await gather(
            _get_result_from_s3(s3_client, s3_bucket, s3_output_key),
            _get_result_from_s3(
                s3_client, s3_bucket, f"{s3_prefix}{job_id}/output.{response_format}"
            ),
        )
        transcription_data: TranscriptionJobData = from_json(data)["results"]
        transcription_data["subtitle_content"] = subtitle
        return transcription_data

    return from_json(await _get_result_from_s3(s3_client, s3_bucket, s3_output_key))[  # type: ignore[no-any-return]
        "results"
    ]


async def _delete_transcription_job(
    transcribe: "TranscribeServiceClient", job_name: str
) -> None:
    """Deletes a transcription job with the specified job name.

    Args:
        transcribe: Transcribe client
        job_name: The name of the transcription job to be deleted.
    """
    try:
        await transcribe.delete_transcription_job(TranscriptionJobName=job_name)
    except ClientError as error:
        if (
            error.response["Error"]["Code"] == "BadRequestException"
            and "couldn't be deleted" in error.response["Error"]["Message"]
        ):
            return
        raise


async def _transcribe_cleanup(
    s3_client: "S3Client",
    transcribe: "TranscribeServiceClient",
    s3_bucket: str,
    s3_tmp_objects: set[str],
    transcribe_tmp_jobs: set[str],
    request_id: str,
) -> None:
    """Cleanup tasks for temporary resources.

    Args:
        s3_client: S3 client
        transcribe: Transcribe client
        s3_bucket: S3 bucket name
        s3_tmp_objects: Set of S3 objects to delete
        transcribe_tmp_jobs: Set of transcription jobs to delete
        request_id: request id.
    """
    with log_background_event("aws_transcribe_cleanup", request_id):
        await gather(
            *(
                s3_client.delete_object(Bucket=s3_bucket, Key=key)
                for key in s3_tmp_objects
            ),
            *(
                _delete_transcription_job(transcribe, job_name)
                for job_name in transcribe_tmp_jobs
            ),
        )


async def perform_transcription_task(
    audio_content: bytes,
    background_tasks: BackgroundTasks,
    language: str | None = None,
    response_format: AudioResponseFormat = "json",
) -> "TranscriptionJobData":
    """Perform complete transcription task using AWS Transcribe with integrated cleanup.

    This function handles the entire transcription workflow from audio upload
    through AWS Transcribe processing to result retrieval, including AWS client
    initialization and cleanup management. It supports both transcription
    and translation workflows by automatically detecting or using specified languages.

    Args:
        audio_content: Audio file content as bytes
        background_tasks: FastAPI background tasks for cleanup
        language: Optional language code for the input audio (ISO-639-1 format)
        response_format: Format for the output response (json, text, srt, vtt, verbose_json)

    Returns:
        Transcript data dictionary with results, or dict with subtitle_content for subtitle formats

    Raises:
        HTTPException: When transcription fails, validation errors occur, or
            unsupported file formats are provided
    """
    s3_bucket = SETTINGS.aws_transcribe_s3_bucket
    if not s3_bucket:
        log_error_details(
            "No S3 bucket configured for AWS Transcribe. "
            "AWS_S3_BUCKET and AWS_TRANSCRIBE_S3_BUCKET environment variable are not set."
        )
        raise HTTPException(
            status_code=404,
            detail="This endpoint is not available on the current server. "
            "Please contact the administrator to enabled it.",
        )
    transcribe: TranscribeServiceClient = get_client("transcribe")
    s3_client: S3Client = get_client("s3", transcribe.meta.region_name)
    s3_tmp_objects: set[str] = set()
    transcribe_tmp_jobs: set[str] = set()
    request_id = REQUEST_ID.get()

    try:
        # Upload audio to S3
        s3_prefix = SETTINGS.aws_s3_tmp_prefix
        s3_input_key = f"{s3_prefix}{request_id}/input"
        await s3_client.put_object(
            Bucket=s3_bucket, Key=s3_input_key, Body=audio_content
        )
        s3_tmp_objects.add(s3_input_key)

        # Build job parameters and start transcription
        job_params = _build_transcription_job_params(
            request_id, s3_bucket, language, response_format
        )

        with _handle_transcription_error(language):
            await transcribe.start_transcription_job(**job_params)

        # Track resources for cleanup
        transcribe_tmp_jobs.add(request_id)
        s3_tmp_objects.add(f"{s3_prefix}{request_id}/output.json")
        s3_tmp_objects.add(f"{s3_prefix}{request_id}/.write_access_check_file.temp")
        if response_format in SUBTITLE_FORMATS:
            s3_tmp_objects.add(f"{s3_prefix}{request_id}/output.{response_format}")

        # Wait for completion and get results
        await _wait_for_transcription_completion(transcribe, request_id)
        return await _get_transcription_results(
            s3_client, s3_bucket, request_id, response_format
        )

    finally:
        if s3_tmp_objects or transcribe_tmp_jobs:
            background_tasks.add_task(
                _transcribe_cleanup,
                s3_client,
                transcribe,
                s3_bucket,
                s3_tmp_objects,
                transcribe_tmp_jobs,
                request_id,
            )


async def _get_result_from_s3(
    s3_client: "S3Client", s3_bucket: str, s3_key: str
) -> str:
    """Retrieve and decode S3 object content as a string.

    Downloads the specified object from S3 and decodes its binary content
    to a UTF-8 string. Used primarily to fetch transcription results and
    subtitle files generated by AWS Transcribe.

    Args:
        s3_client: Initialized AWS S3 client for performing operations
        s3_bucket: Name of the S3 bucket containing the object
        s3_key: Key (path) of the object within the S3 bucket

    Returns:
        Decoded string content of the S3 object

    Raises:
        ClientError: When S3 object retrieval fails or object doesn't exist
        UnicodeDecodeError: When object content cannot be decoded as UTF-8
    """
    return (
        await (await s3_client.get_object(Bucket=s3_bucket, Key=s3_key))["Body"].read()
    ).decode()


async def _transcript_audio_sse(
    stream: Generator[str],
) -> AsyncGenerator[JSONServerSentEvent]:
    """Generate Server-Sent Events for real-time audio transcription streaming.

    Converts a text generator stream into OpenAI-compatible Server-Sent Events
    for real-time transcription delivery. Emits delta events for incremental
    text updates and a final done event with complete transcript and usage data.

    Args:
        stream: Generator yielding transcript text chunks from AWS Transcribe

    Yields:
        JSONServerSentEvent: SSE events with transcript.text.delta and
            transcript.text.done event types following OpenAI streaming format
    """
    deltas: list[str] = []
    try:
        for delta in stream:
            if deltas:
                delta = f" {delta}"
            deltas.append(delta)
            yield JSONServerSentEvent(
                data=TranscriptionTextDeltaEvent(
                    delta=delta, type="transcript.text.delta"
                ).model_dump(mode="json", exclude_none=True)
            )
    finally:
        full_text = "".join(deltas)
        estimated_tokens = await estimate_token_count(full_text) or 0
        yield JSONServerSentEvent(
            data=TranscriptionTextDoneEvent(
                text=full_text,
                usage=UsageTokens(
                    # Estimated token count for transcribed text
                    input_tokens=0,
                    output_tokens=estimated_tokens,
                    total_tokens=estimated_tokens,
                    type="tokens",
                    input_token_details=UsageInputTokenDetails(
                        text_tokens=0, audio_tokens=0
                    ),
                ),
                type="transcript.text.done",
            ).model_dump(mode="json", exclude_none=True)
        )


def get_transcript_text(transcript_data: "TranscriptionJobData") -> str:
    """Extract and concatenate transcript text from AWS Transcribe response data.

    Args:
        transcript_data: Parsed transcription results from AWS Transcribe

    Returns:
        Concatenated transcript text as a single string
    """
    return " ".join(
        transcript["transcript"] for transcript in transcript_data["transcripts"]
    ).strip()


def format_text_or_json_response(
    transcript_data: "TranscriptionJobData",
    text: str,
    response_format: AudioResponseFormat,
    timestamp_granularities: list[AudioTimestampGranularities] | None = None,
) -> str | TranscriptionCreateResponse:
    """Format transcription response based on requested output format.

    Converts transcript data into the appropriate response format following
    OpenAI API specification. Supports plain text, JSON, and verbose JSON
    with optional timestamp granularity information.

    Args:
        transcript_data: Parsed transcription results from AWS Transcribe
        text: Processed transcript text content
        response_format: Desired output format (text, json, verbose_json)
        timestamp_granularities: Optional list of timestamp types to include

    Returns:
        Formatted response as string for text format or OpenAI types for JSON formats
    """
    if response_format == "text":
        return text

    duration = get_audio_duration(transcript_data)
    usage_duration = UsageDuration(
        type="duration",
        # Minimum AWS Transcribe billed duration is 15s
        seconds=max(ceil(duration), 15),
    )
    if response_format == "verbose_json":
        segments = None
        words = None

        if timestamp_granularities:
            if "segment" in timestamp_granularities:
                segments = [
                    TranscriptionSegment(
                        id=segment["id"],
                        end=float(segment["end_time"]),
                        start=float(segment["start_time"]),
                        text=segment["transcript"],
                        # Not supported
                        no_speech_prob=0.0 if len(segment["transcript"]) else 1.0,
                        avg_logprob=0.0,
                        compression_ratio=0.0,
                        seek=0,
                        temperature=0.0,
                        tokens=[],
                    )
                    for segment in transcript_data["audio_segments"]
                ]
            if "word" in timestamp_granularities:
                words = [
                    TranscriptionWord(
                        word=item["alternatives"][0]["content"],
                        end=float(item["end_time"]),
                        start=float(item["start_time"]),
                    )
                    for item in transcript_data["items"]
                    if item["type"] == "pronunciation"
                ]

        return log_response_params(
            TranscriptionVerbose(
                duration=duration,
                language=language_code_to_name(transcript_data["language_code"]),
                text=text,
                segments=segments,
                words=words,
                usage=usage_duration,
            )
        )
    return log_response_params(Transcription(text=text, usage=usage_duration))


def get_audio_duration(transcript_data: "TranscriptionJobData") -> float:
    """Get audio duration from AWS Transcribe response data.

    Args:
        transcript_data: Parsed transcription results from AWS Transcribe

    Returns:
        Duration in seconds
    """
    return float(transcript_data["audio_segments"][-1]["end_time"])


def format_subtitle_response(
    response_format: AudioResponseFormat, subtitle_content: str, file: UploadFile
) -> Response:
    """Format subtitle response with proper content type and disposition headers.

    Creates a FastAPI Response object for subtitle format downloads (SRT/VTT)
    with appropriate MIME type and filename in Content-Disposition header.

    Args:
        response_format: The subtitle response format (SRT or VTT)
        subtitle_content: The subtitle content as a string
        file: The original uploaded file for filename extraction

    Returns:
        FastAPI Response object with subtitle content and proper headers
    """
    return Response(
        content=subtitle_content.encode(),
        media_type=f"text/{response_format}; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename={Path(file.filename or 'audio').stem}.{response_format}"
        },
    )


@router.post(
    "/transcriptions",
    response_model=None,
    summary="OpenAI - /v1/audio/transcriptions",
    description=(
        "Transcribes audio files into text.\n"
        "The model uses the specified language or automatically detects it when not provided."
        "Supports multiple output formats including plain text, "
        "JSON, verbose JSON with segments/words, and subtitle formats (SRT/VTT)."
    ),
    response_description="Returns transcription in the specified format",
    responses={
        200: {"description": "Transcription completed (or streaming)."},
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "examples": {
                        "json": {
                            "summary": "JSON response",
                            "value": {
                                "model": "amazon.transcribe",
                                "response_format": "json",
                            },
                        },
                        "vtt": {
                            "summary": "Subtitle (VTT)",
                            "value": {"response_format": "vtt"},
                        },
                        "stream": {
                            "summary": "Streaming SSE",
                            "value": {"stream": True},
                        },
                    }
                }
            }
        }
    },
    response_model_exclude_none=True,
)
async def create_transcription(
    file: Annotated[
        UploadFile,
        File(
            ...,
            description="The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.",
        ),
    ],
    *,
    model: Annotated[
        str,
        Form(
            description=(
                "The transcription model to use.\nAvailable models: amazon.transcribe"
            )
        ),
    ] = TRANSCRIBE_MODEL_ID,
    language: Annotated[
        str | None,
        Form(
            description=(
                "The language of the input audio.\n"
                "Supplying the input language in ISO-639-1 (e.g. `en`) format will improve accuracy and latency."
            )
        ),
    ] = None,
    prompt: Annotated[
        str | None,
        Form(
            description=(
                "An optional text to guide the model's style or continue a previous audio segment.\n"
                "The prompt should match the audio language.\nUNSUPPORTED on this implementation."
            )
        ),
    ] = None,
    chunking_strategy: Annotated[
        ChunkingStrategy,
        Form(
            description=(
                "Controls how the audio is cut into chunks.\n"
                "When set to `auto`, the server first normalizes loudness and then uses voice activity detection (VAD) to choose boundaries. "
                "`server_vad` object can be provided to tweak VAD detection parameters manually. "
                "If unset, the audio is transcribed as a single block.\nUNSUPPORTED on this implementation."
            )
        ),
    ] = "auto",
    response_format: Annotated[
        AudioResponseFormat,
        Form(
            description=(
                "The format of the transcript output.\n"
                "Supported formats: `json`, `text`, `srt`, `verbose_json`, `vtt`"
            )
        ),
    ] = "json",
    timestamp_granularities: Annotated[
        str,
        Form(
            description=(
                "Comma-separated list of timestamp granularities to populate for this transcription (e.g. `word,segment`).\n"
                "`response_format` must be set to `verbose_json` to use timestamp granularities.\n"
                "Either or both of these options are supported: `word`, or `segment`."
            )
        ),
    ] = "",
    temperature: Annotated[
        float,
        Form(
            description=(
                "The sampling temperature, between `0` and `1`.\n"
                "Higher values like `0.8` will make the output more random, while lower values like `0.2` will make it more focused and deterministic. "
                "If set to `0`, the model will use log probability to automatically increase the temperature until certain thresholds are hit.\n"
                "UNSUPPORTED on this implementation."
            )
        ),
    ] = 0.0,
    stream: Annotated[
        bool | None,
        Form(
            description=(
                "If set to true, the model response data will be streamed to the client as it is generated using "
                "server-sent events."
            )
        ),
    ] = False,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: Annotated[None, Depends(authenticate)] = None,
) -> str | TranscriptionCreateResponse | EventSourceResponse | Response:
    """Transcribes audio into the input language.

    Converts audio to text in the same language as the source audio. The model will
    use the specified language or automatically detect the language if not provided.
    Supports multiple output formats including plain text, JSON with metadata, and
    subtitle formats for video content.

    Args:
        file: The audio file to transcribe.
        model: The transcription model to use. Available models: amazon.transcribe.
        language: The language of the input audio (ISO-639-1 code, e.g. `en`). Improves accuracy and latency when provided.
        prompt: Optional style guidance for the model. UNSUPPORTED on this implementation.
        chunking_strategy: Controls how the audio is cut into chunks. `auto` only is supported on this implementation.
        response_format: Output format: `json`, `text`, `srt`, `verbose_json`, or `vtt`.
        timestamp_granularities: For `verbose_json` only; comma-separated values among `word` and `segment` (e.g. `word,segment`).
        temperature: Sampling temperature. UNSUPPORTED on this implementation (must be 0.0).
        stream: Whether to stream partial results via Server-Sent Events.
        background_tasks: FastAPI background tasks for cleanup.

    Returns:
        The transcribed text in the requested format.

    Raises:
        HTTPException: When transcription fails or invalid parameters are provided.
    """
    with validation_error_handler():
        request = TranscriptionCreateParams(
            model=model,
            language=language,
            prompt=prompt,
            chunking_strategy=chunking_strategy,
            response_format=response_format,
            timestamp_granularities=(
                timestamp_granularities.split(",") if timestamp_granularities else []  # type: ignore[arg-type]
            ),
            temperature=temperature,
            stream=stream,
        )
    log_request_params(request)

    if model != TRANSCRIBE_MODEL_ID:
        raise OpenaiUnsupportedModelError(model)
    log = REQUEST_LOG.get()
    log["model_id"] = model

    transcript_data = await perform_transcription_task(
        audio_content=await file.read(),
        background_tasks=background_tasks,
        language=request.language,
        response_format=request.response_format,
    )

    # Handle subtitle formats (SRT/VTT)
    if request.response_format in SUBTITLE_FORMATS:
        return format_subtitle_response(
            response_format, transcript_data["subtitle_content"], file
        )

    # Handle streaming
    if request.stream:
        return EventSourceResponse(
            await log_request_stream_event(
                _transcript_audio_sse(
                    transcript["transcript"]
                    for transcript in transcript_data["transcripts"]
                )
            )
        )

    # Handle text, json, and verbose_json formats
    return format_text_or_json_response(
        transcript_data,
        get_transcript_text(transcript_data),
        request.response_format,
        request.timestamp_granularities,
    )
