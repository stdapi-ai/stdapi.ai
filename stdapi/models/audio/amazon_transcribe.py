"""Amazon Transcribe model implementation."""

from asyncio import gather, sleep
from contextlib import contextmanager
from math import ceil
from typing import TYPE_CHECKING, NotRequired

from botocore.exceptions import ClientError
from fastapi import HTTPException, Response
from fastapi import UploadFile as FastAPIUploadFile
from pydantic_core import from_json
from typing_extensions import TypedDict

from stdapi.aws import get_client
from stdapi.aws_s3 import get_text_from_s3, put_upload_file_to_s3
from stdapi.aws_translate import translate, translate_subtitle
from stdapi.config import SETTINGS
from stdapi.models import (
    EXTRA_MODELS,
    EXTRA_MODELS_INPUT_MODALITY,
    EXTRA_MODELS_OUTPUT_MODALITY,
    MODEL_ALIASES,
    ModelDetails,
)
from stdapi.models.audio import AudioModelBase
from stdapi.monitoring import (
    REQUEST_ID,
    log_background_event,
    log_error_details,
    log_response_params,
)
from stdapi.openai_exceptions import OpenaiInvalidLanguageFormatError
from stdapi.tokenizer import estimate_token_count
from stdapi.types.openai_audio import (
    SUBTITLE_FORMATS,
    AudioResponseFormat,
    AudioTimestampGranularities,
    Transcription,
    TranscriptionCreateResponse,
    TranscriptionSegment,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
    TranscriptionVerbose,
    TranscriptionWord,
    Translation,
    TranslationCreateResponse,
    TranslationVerbose,
    UsageDuration,
    UsageInputTokenDetails,
    UsageTokens,
)
from stdapi.utils import format_language_code, language_code_to_name

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from fastapi import BackgroundTasks, UploadFile
    from types_aiobotocore_s3.client import S3Client
    from types_aiobotocore_transcribe.client import TranscribeServiceClient
    from types_aiobotocore_transcribe.type_defs import (
        StartTranscriptionJobRequestTypeDef,
    )

#: Transcribe model ID
AWS_TRANSCRIBE_MODEL_ID = "amazon.transcribe"

# Use AWS transcribe as the default STT engine
MODEL_ALIASES["whisper-1"] = AWS_TRANSCRIBE_MODEL_ID


# AWS Transcribe-specific data structures
class TranscribeJobItem(TypedDict, total=False):
    """AWS Transcribe transcript item structure."""

    type: str
    alternatives: list[dict[str, str]]
    start_time: NotRequired[str]
    end_time: NotRequired[str]


class TranscribeJobAudioSegment(TypedDict):
    """AWS Transcribe audio segment structure."""

    id: int
    start_time: str
    end_time: str
    transcript: str


class TranscribeJobTranscript(TypedDict):
    """AWS Transcribe transcript result structure."""

    transcript: str


class TranscribeJobData(TypedDict, total=False):
    """AWS Transcribe job result data structure."""

    transcripts: list[TranscribeJobTranscript]
    audio_segments: list[TranscribeJobAudioSegment]
    items: list[TranscribeJobItem]
    language_code: str
    subtitle_content: NotRequired[str]


async def initialize_transcribe_models() -> None:
    """Initialize extra models."""
    transcribe: TranscribeServiceClient = get_client("transcribe")
    EXTRA_MODELS_INPUT_MODALITY.setdefault("SPEECH", set()).add(AWS_TRANSCRIBE_MODEL_ID)
    EXTRA_MODELS_OUTPUT_MODALITY.setdefault("TEXT", set()).add(AWS_TRANSCRIBE_MODEL_ID)
    EXTRA_MODELS[AWS_TRANSCRIBE_MODEL_ID] = ModelDetails(
        id=AWS_TRANSCRIBE_MODEL_ID,
        name="Transcribe",
        provider="Amazon",
        region=transcribe.meta.region_name,
        service="AWS Transcribe",
        input_modalities=["SPEECH"],
        output_modalities=["TEXT"],
    )


def _get_transcript_text(transcript_data: TranscribeJobData) -> str:
    """Extract and concatenate transcript text from AWS Transcribe response data.

    Args:
        transcript_data: Parsed transcription results from AWS Transcribe

    Returns:
        Concatenated transcript text as a single string
    """
    return " ".join(
        transcript["transcript"] for transcript in transcript_data["transcripts"]
    ).strip()


def _get_audio_duration(transcript_data: TranscribeJobData) -> float:
    """Get audio duration from AWS Transcribe response data.

    Args:
        transcript_data: Parsed transcription results from AWS Transcribe

    Returns:
        Duration in seconds
    """
    try:
        segment = transcript_data["audio_segments"][-1]
    except IndexError:
        return 0.0
    return float(segment["end_time"])


def _build_transcription_job_params(
    job_id: str, s3_bucket: str, language: str | None, response_format: str
) -> StartTranscriptionJobRequestTypeDef:
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
        # AWS Transcribe will create subtitle file at: {s3_prefix}{job_id}/output.{format}
        job_params["Subtitles"] = {
            "Formats": [response_format],  # type: ignore[list-item]
            "OutputStartIndex": 1,
        }

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
                raise OpenaiInvalidLanguageFormatError(msg) from error
            if "file" in error_message:
                raise HTTPException(status_code=400, detail=error_message) from error
        raise  # pragma: no cover


async def _wait_for_transcription_completion(
    transcribe: TranscribeServiceClient, job_id: str
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
    s3_client: S3Client, s3_bucket: str, job_id: str, response_format: str
) -> TranscribeJobData:
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
            get_text_from_s3(s3_client, s3_bucket, s3_output_key),
            get_text_from_s3(
                s3_client, s3_bucket, f"{s3_prefix}{job_id}/output.{response_format}"
            ),
        )
        transcription_data: TranscribeJobData = from_json(data)["results"]
        transcription_data["subtitle_content"] = subtitle
        return transcription_data

    return from_json(  # type: ignore[no-any-return]
        await get_text_from_s3(s3_client, s3_bucket, s3_output_key)
    )["results"]


async def _delete_transcription_job(
    transcribe: TranscribeServiceClient, job_name: str
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
    s3_client: S3Client,
    transcribe: TranscribeServiceClient,
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


def _format_json_response(
    transcript_data: TranscribeJobData,
    text: str,
    duration: float,
    usage_duration: UsageDuration,
    timestamp_granularities: list[AudioTimestampGranularities] | None = None,
) -> TranscriptionCreateResponse:
    """Format transcription response based on requested output format.

    Converts transcript data into the appropriate response format following
    OpenAI API specification. Supports plain text, JSON, and verbose JSON
    with optional timestamp granularity information.

    Args:
        transcript_data: Parsed transcription results from AWS Transcribe
        text: Processed transcript text content
        duration: Audio duration in seconds
        usage_duration: Usage duration for transcription
        timestamp_granularities: Optional list of timestamp types to include

    Returns:
        Formatted response as string for text format or OpenAI types for JSON formats
    """
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


class AudioModel(AudioModelBase[None, None]):
    """Amazon Transcribe audio model implementation (transcription only)."""

    MATCHER = AWS_TRANSCRIBE_MODEL_ID

    @staticmethod
    async def _transcribe(
        audio_content: UploadFile,
        background_tasks: BackgroundTasks,
        response_format: AudioResponseFormat,
        language: str | None = None,
    ) -> TranscribeJobData:
        """Perform transcription task using AWS Transcribe and returns row result.

        This function handles the entire transcription workflow from audio upload
        through AWS Transcribe processing to result retrieval, including AWS client
        initialization and cleanup management.

        Args:
            audio_content: Audio file content file
            background_tasks: FastAPI background tasks for cleanup
            response_format: Format for the output response (json, text, srt, vtt, verbose_json)
            language: Optional language code for the input audio (ISO-639-1 format)

        Returns:
            Raw transcription response

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
            await put_upload_file_to_s3(
                audio_content, s3_client, s3_bucket, s3_input_key
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

    @classmethod
    def _format_transcription_response(
        cls,
        transcript_data: TranscribeJobData,
        file: FastAPIUploadFile,
        response_format: AudioResponseFormat,
        timestamp_granularities: list[AudioTimestampGranularities] | None = None,
    ) -> str | TranscriptionCreateResponse | Response:
        """Format transcription response for the route.

        Args:
            transcript_data: AWS Transcribe job data
            file: Original uploaded file
            response_format: Requested response format
            timestamp_granularities: Optional timestamp granularities for verbose_json

        Returns:
            Formatted response in the requested format
        """
        if response_format in SUBTITLE_FORMATS:
            return cls._format_subtitle_response(
                response_format, transcript_data["subtitle_content"], file
            )

        text = _get_transcript_text(transcript_data)

        if response_format == "text":
            return text

        duration = _get_audio_duration(transcript_data)
        usage_duration = UsageDuration(
            type="duration",
            # Minimum AWS Transcribe billed duration is 15s
            seconds=max(ceil(duration), 15),
        )

        if response_format == "verbose_json":
            return _format_json_response(
                transcript_data, text, duration, usage_duration, timestamp_granularities
            )

        return log_response_params(Transcription(text=text, usage=usage_duration))

    @classmethod
    def _format_translation_response(
        cls,
        transcript_data: TranscribeJobData,
        translated_content: str,
        file: FastAPIUploadFile,
        response_format: AudioResponseFormat,
    ) -> str | TranslationCreateResponse | Response:
        """Format translation response for the route.

        Args:
            transcript_data: AWS Transcribe job data
            translated_content: Pre-translated text or subtitle content
            file: Original uploaded file
            response_format: Requested response format

        Returns:
            Formatted response in the requested format
        """
        if "subtitle_content" in transcript_data:
            return cls._format_subtitle_response(
                response_format, translated_content, file
            )

        if response_format == "text":
            return translated_content

        if response_format == "verbose_json":
            return TranslationVerbose(
                duration=_get_audio_duration(transcript_data),
                language="english",  # Translation output is always English
                text=translated_content,
                segments=[
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
                ],
            )

        return Translation(text=translated_content)

    async def stt(
        self,
        audio_content: UploadFile,
        background_tasks: BackgroundTasks,
        response_format: AudioResponseFormat,
        language: str | None = None,
        timestamp_granularities: list[AudioTimestampGranularities] | None = None,
    ) -> str | TranscriptionCreateResponse | Response:
        """Perform transcription task using AWS Transcribe.

        This function handles the entire transcription workflow from audio upload
        through AWS Transcribe processing to result retrieval, including AWS client
        initialization and cleanup management.

        Args:
            audio_content: Audio file content file
            background_tasks: FastAPI background tasks for cleanup
            response_format: Format for the output response (json, text, srt, vtt, verbose_json)
            language: Optional language code for the input audio (ISO-639-1 format)
            timestamp_granularities: Optional timestamp granularities for verbose_json

        Returns:
            Formatted transcription response

        Raises:
            HTTPException: When transcription fails, validation errors occur, or
                unsupported file formats are provided
        """
        return self._format_transcription_response(
            await self._transcribe(
                audio_content, background_tasks, response_format, language
            ),
            audio_content,
            response_format,
            timestamp_granularities,
        )

    async def stt_stream(
        self,
        audio_content: UploadFile,
        background_tasks: BackgroundTasks,
        response_format: AudioResponseFormat,
        language: str | None = None,
    ) -> AsyncGenerator[TranscriptionTextDeltaEvent | TranscriptionTextDoneEvent]:
        """Transcribe audio to text with streaming response.

        Args:
            audio_content: Audio file to transcribe
            background_tasks: FastAPI background tasks for cleanup
            response_format: Format for output
            language: Optional language code

        Yields:
            TranscriptionTextDeltaEvent or TranscriptionTextDoneEvent objects

        Raises:
            HTTPException: When transcription fails
        """
        transcript_data = await self._transcribe(
            audio_content, background_tasks, response_format, language
        )

        full_text_parts: list[str] = []
        for transcript in transcript_data["transcripts"]:
            text = transcript["transcript"]
            full_text_parts.append(text)
            yield TranscriptionTextDeltaEvent(delta=text, type="transcript.text.delta")

        text = " ".join(full_text_parts)
        estimated_tokens = await estimate_token_count(text) or 0
        yield TranscriptionTextDoneEvent(
            text=text,
            type="transcript.text.done",
            usage=UsageTokens(
                # Estimated token count for transcribed text
                input_tokens=0,
                output_tokens=estimated_tokens,
                total_tokens=estimated_tokens,
                type="tokens",
                input_token_details=UsageInputTokenDetails(
                    text_tokens=0, audio_tokens=0
                ),
            )
            if estimated_tokens
            else None,
        )

    async def stt_translate(
        self,
        audio_content: UploadFile,
        background_tasks: BackgroundTasks,
        response_format: AudioResponseFormat,
    ) -> str | TranslationCreateResponse | Response:
        """Transcribe and translate audio to English.

        This method performs transcription using AWS Transcribe, detects the source
        language, and translates the transcribed text to English using AWS Translate.

        Args:
            audio_content: Audio file to transcribe and translate
            background_tasks: FastAPI background tasks for cleanup
            response_format: Format for output (json, text, srt, vtt, verbose_json)

        Returns:
            Formatted translation response with translated text in English

        Raises:
            HTTPException: When transcription or translation fails
        """
        transcript_data = await self._transcribe(
            audio_content, background_tasks, response_format
        )

        language = transcript_data["language_code"]
        if "subtitle_content" in transcript_data:
            translated_content = await translate_subtitle(
                transcript_data["subtitle_content"], language
            )
        else:
            translated_content = await translate(
                _get_transcript_text(transcript_data), language
            )

        return self._format_translation_response(
            transcript_data, translated_content, audio_content, response_format
        )
