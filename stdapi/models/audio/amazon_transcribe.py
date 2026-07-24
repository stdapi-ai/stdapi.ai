"""Amazon Transcribe model implementation."""

from asyncio import gather, sleep
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, NotRequired

from botocore.exceptions import ClientError
from pydantic_core import from_json
from typing_extensions import TypedDict

from stdapi.api_errors import ApiError, InvalidLanguageFormatError
from stdapi.aws import call_with_region_failover, get_client
from stdapi.aws_s3 import copy_s3_object, get_text_from_s3, track_temporary_s3_objects
from stdapi.aws_translate import translate, translate_subtitle
from stdapi.cleanup import schedule_cleanup
from stdapi.config import SETTINGS
from stdapi.models import (
    EXTRA_MODELS,
    EXTRA_MODELS_INPUT_MODALITY,
    EXTRA_MODELS_OUTPUT_MODALITY,
    ModelDetails,
)
from stdapi.models.audio import AudioModelBase
from stdapi.monitoring import (
    REQUEST_ID,
    build_metadata,
    log_error_details,
    log_response_params,
)
from stdapi.types.openai_audio import (
    SUBTITLE_FORMATS,
    AudioResponseFormat,
    AudioTimestampGranularities,
    Transcription,
    TranscriptionCreateResponse,
    TranscriptionDiarized,
    TranscriptionDiarizedSegment,
    TranscriptionSegment,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
    TranscriptionVerbose,
    TranscriptionWord,
    Translation,
    TranslationCreateResponse,
    TranslationVerbose,
    UsageDuration,
)
from stdapi.usage import record_transcribe_usage
from stdapi.utils import format_language_code, language_code_to_name

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from fastapi import Response
    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_transcribe.client import TranscribeServiceClient
    from types_aiobotocore_transcribe.type_defs import (
        StartTranscriptionJobRequestTypeDef,
    )

    from stdapi.input_file import InputFile

#: Transcribe model ID
AWS_TRANSCRIBE_MODEL_ID = "amazon.transcribe"

#: Ordinal value of the "A" letter used as speaker label
_A_ORDINAL_VALUE = ord("A")


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
    speaker_label: NotRequired[str]


class TranscribeJobTranscript(TypedDict):
    """AWS Transcribe transcript result structure."""

    transcript: str


class TranscribeJobSpeakerLabelItem(TypedDict):
    """AWS Transcribe speaker label item structure."""

    speaker_label: str
    content: str


class TranscribeJobSpeakerSegment(TypedDict):
    """AWS Transcribe speaker segment structure."""

    speaker_label: str
    start_time: str
    end_time: str
    items: list[TranscribeJobSpeakerLabelItem]


class TranscribeJobData(TypedDict, total=False):
    """AWS Transcribe job result data structure."""

    transcripts: list[TranscribeJobTranscript]
    audio_segments: list[TranscribeJobAudioSegment]
    items: list[TranscribeJobItem]
    language_code: str
    subtitle_content: NotRequired[str]


#: Region that served the current request's transcription job.
_SERVED_REGION: ContextVar[str] = ContextVar("transcribe_served_region", default="")


def transcribe_job_candidates() -> list[tuple[RegionName, str]]:
    """Return the candidate (region, S3 bucket) pairs for transcription jobs.

    Transcribe reads and writes through an S3 bucket co-located with the
    job's region: the primary region is served by ``aws_transcribe_s3_bucket``
    (itself defaulting to ``aws_s3_bucket``), the other candidates by their
    ``aws_s3_regional_buckets`` entry.

    Returns:
        (region, bucket) pairs in priority order; empty when no candidate
        region has a usable bucket.
    """
    if region := SETTINGS.aws_transcribe_region:
        bucket = (
            SETTINGS.aws_transcribe_s3_bucket
            or SETTINGS.aws_s3_regional_buckets.get(region)
        )
        return [(region, bucket)] if bucket else []
    primary = SETTINGS.aws_bedrock_regions[0]
    candidates: list[tuple[RegionName, str]] = []
    for candidate in SETTINGS.aws_bedrock_regions:
        bucket = SETTINGS.aws_s3_regional_buckets.get(candidate)
        if candidate == primary:
            bucket = SETTINGS.aws_transcribe_s3_bucket or bucket
        if bucket:
            candidates.append((candidate, bucket))
    return candidates


async def initialize_transcribe_models() -> None:
    """Initialize extra models.

    The advertised regions are the bucket-equipped serving candidates; the
    model stays registered with an empty list when there is none (requests
    then get the 404 guard's operator-actionable error message).
    """
    EXTRA_MODELS_INPUT_MODALITY.setdefault("SPEECH", set()).add(AWS_TRANSCRIBE_MODEL_ID)
    EXTRA_MODELS_OUTPUT_MODALITY.setdefault("TEXT", set()).add(AWS_TRANSCRIBE_MODEL_ID)
    EXTRA_MODELS[AWS_TRANSCRIBE_MODEL_ID] = ModelDetails(
        id=AWS_TRANSCRIBE_MODEL_ID,
        name="Transcribe",
        provider="Amazon",
        regions=[region for region, _ in transcribe_job_candidates()],
        service="AWS Transcribe",
        input_modalities=["SPEECH"],
        output_modalities=["TEXT"],
    )


async def _start_transcription_with_failover(
    candidates: list[tuple[RegionName, str]],
    job_id: str,
    language: str | None,
    response_format: str,
) -> tuple[RegionName, str]:
    """Start the transcription job, failing over across candidate regions.

    The audio must already be uploaded to the first candidate's bucket;
    later attempts server-side copy it into their own region's bucket
    (Transcribe requires the media bucket in the job's region). A failed
    region gets a best-effort job deletion so no orphaned job keeps
    running (and billing) there.

    Args:
        candidates: (region, bucket) pairs in priority order (at least one).
        job_id: Transcription job name (also the input key's directory).
        language: Optional language code, for caller-error translation.
        response_format: Requested response format.

    Returns:
        The (region, bucket) pair that accepted the job.

    Raises:
        ApiError: For caller errors (unsupported language, bad file).
        BotoCoreError: When every candidate region fails (last error).
        ClientError: Same as above.
    """
    input_key = f"{SETTINGS.aws_s3_tmp_prefix}{job_id}/input"
    buckets = dict(candidates)
    first_bucket = candidates[0][1]

    async def _attempt(transcribe: TranscribeServiceClient, region: RegionName) -> str:
        """Run one region's transcription job against its co-located bucket."""
        bucket = buckets[region]
        if bucket != first_bucket:
            await copy_s3_object(
                first_bucket,
                input_key,
                dest_bucket=bucket,
                dest_key=input_key,
                dest_region=region,
                temporary=True,
            )
        with _handle_transcription_error(language):
            await transcribe.start_transcription_job(
                **_build_transcription_job_params(
                    job_id, bucket, language, response_format
                )
            )
        return bucket

    async def _cleanup(
        transcribe: TranscribeServiceClient, _region: RegionName
    ) -> None:
        """Best-effort delete: the start may have been accepted despite the error."""
        await transcribe.delete_transcription_job(TranscriptionJobName=job_id)

    bucket, region = await call_with_region_failover(
        "transcribe",
        [region for region, _ in candidates],
        _attempt,
        on_failed_region=_cleanup,
    )
    return region, bucket


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
        Duration in seconds, 0.0 when the response reports no segments
        (usage then falls back to the AWS 15-second billing minimum).
    """
    try:
        return float(transcript_data["audio_segments"][-1]["end_time"])
    except KeyError, IndexError:
        log_error_details(
            "Transcribe response reports no audio segments;"
            " billing the 15-second minimum.",
            level="warning",
        )
        return 0.0


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
        "Tags": [{"Key": k, "Value": v} for k, v in build_metadata(apn=True).items()],
    }

    if language:
        job_params["LanguageCode"] = format_language_code(language)  # type: ignore[typeddict-item]
    else:
        job_params["IdentifyLanguage"] = True

    if response_format in SUBTITLE_FORMATS:
        # AWS Transcribe will create subtitle file at: {s3_prefix}{job_id}/output.{format}
        job_params["Subtitles"] = {"Formats": [response_format], "OutputStartIndex": 1}

    elif response_format == "diarized_json":
        job_params["Settings"] = {"ShowSpeakerLabels": True, "MaxSpeakerLabels": 10}

    return job_params


@contextmanager
def _handle_transcription_error(language: str | None) -> Generator[None]:
    """Context manager to handle transcription job start errors.

    Args:
        language: Language code that may have caused the error.

    Yields:
        None

    Raises:
        ApiError: With appropriate error message.

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
                msg = f"Language '{language}' is not supported by the model"
                raise InvalidLanguageFormatError(msg) from error
            if "file" in error_message:
                raise ApiError(error_message) from error
        raise  # pragma: no cover


async def _wait_for_transcription_completion(
    transcribe: TranscribeServiceClient, job_id: str
) -> None:
    """Wait for transcription job to complete.

    Args:
        transcribe: Transcribe service client
        job_id: Transcription job ID

    Raises:
        ApiError: If transcription fails
    """
    while True:  # Timeout at FastAPI level
        job = (await transcribe.get_transcription_job(TranscriptionJobName=job_id))[
            "TranscriptionJob"
        ]
        if job["TranscriptionJobStatus"] == "COMPLETED":
            break
        if job["TranscriptionJobStatus"] == "FAILED":
            raise ApiError(job["FailureReason"])
        await sleep(0.5)


async def _get_transcription_results(
    s3_bucket: str, job_id: str, response_format: str
) -> TranscribeJobData:
    """Get transcription results from S3.

    Args:
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
            get_text_from_s3(s3_bucket, s3_output_key),
            get_text_from_s3(
                s3_bucket, f"{s3_prefix}{job_id}/output.{response_format}"
            ),
        )
        transcription_data: TranscribeJobData = from_json(data)["results"]
        transcription_data["subtitle_content"] = subtitle
        return transcription_data

    return from_json(  # type: ignore[no-any-return]
        await get_text_from_s3(s3_bucket, s3_output_key)
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


def _format_diarized_json_response(
    transcript_data: TranscribeJobData,
    text: str,
    duration: float,
    usage_duration: UsageDuration,
) -> TranscriptionDiarized:
    """Format transcription response as diarized JSON with speaker segments.

    Converts AWS Transcribe speaker-labeled results into OpenAI-compatible
    diarized JSON format with speaker segments.

    Args:
        transcript_data: Parsed transcription results from AWS Transcribe
        text: Processed transcript text content
        duration: Audio duration in seconds
        usage_duration: Usage duration for transcription

    Returns:
        Formatted diarized response with speaker segments
    """
    speakers: dict[str, str] = {}
    return log_response_params(
        TranscriptionDiarized(
            duration=duration,
            segments=[
                TranscriptionDiarizedSegment(
                    id=f"seg_{segment['id']}",
                    start=float(segment["start_time"]),
                    end=float(segment["end_time"]),
                    speaker=speakers.setdefault(
                        segment["speaker_label"], chr(_A_ORDINAL_VALUE + len(speakers))
                    ),
                    text=segment["transcript"],
                    type="transcript.text.segment",
                )
                for segment in transcript_data["audio_segments"]
            ],
            task="transcribe",
            text=text,
            usage=usage_duration,
        )
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

    SUPPORTED_RESPONSES_FORMATS = frozenset(
        {"json", "text", "srt", "verbose_json", "vtt", "diarized_json"}
    )
    SUPPORTED_TIMESTAMP_GRANULARITIES = frozenset({"word", "segment"})

    @classmethod
    def get_aliases(
        cls,
        all_models: dict[str, ModelDetails],  # noqa: ARG003
    ) -> dict[str, str]:
        """Return dynamic aliases specific to this model class.

        Override in subclasses to provide model-specific aliases.
        Each alias maps an alternative name to a Bedrock model ID.

        Args:
            all_models: All available models keyed by model ID.

        Returns:
            A dict mapping alias to model ID.
        """
        return {"whisper-1": "amazon.transcribe"}

    async def _transcribe(
        self,
        audio_content: InputFile,
        response_format: AudioResponseFormat,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        *,
        logprobs: bool = False,
    ) -> TranscribeJobData:
        """Perform transcription task using AWS Transcribe and returns row result.

        This function handles the entire transcription workflow from audio upload
        through AWS Transcribe processing to result retrieval, including AWS client
        initialization and cleanup management.

        Args:
            audio_content: Audio file content file
            response_format: Format for the output response (json, text, srt, vtt, verbose_json, diarized_json)
            language: Optional language code for the input audio (ISO-639-1 format)
            prompt: Optional prompt for transcription.
            temperature: Optional temperature for transcription.
            logprobs: If true, return log probabilities.

        Returns:
            Raw transcription response

        Raises:
            ApiError: When transcription fails, validation errors occur, or
                unsupported file formats are provided
        """
        self._validate_no_prompt(prompt)
        self._validate_no_temperature(temperature)
        self._validate_no_logprobs(logprobs)

        candidates = transcribe_job_candidates()
        if not candidates:
            log_error_details(
                "No S3 bucket configured for AWS Transcribe: set AWS_S3_BUCKET, "
                "AWS_TRANSCRIBE_S3_BUCKET, or an AWS_S3_REGIONAL_BUCKETS entry "
                "for a candidate region."
            )
            msg = (
                "This model is not available on the current server. "
                "Please contact the administrator to enabled it."
            )
            raise ApiError(msg, status=404)

        to_cleanup: tuple[TranscribeServiceClient, str] | None = None
        request_id = REQUEST_ID.get()
        s3_prefix = SETTINGS.aws_s3_tmp_prefix

        try:
            # Upload audio once, to the first candidate's bucket (the source
            # is consumed); failover attempts server-side copy it from there.
            await audio_content.to_s3(
                candidates[0][0],
                bucket=candidates[0][1],
                key=f"{s3_prefix}{request_id}/input",
            )
            region, s3_bucket = await _start_transcription_with_failover(
                candidates, request_id, language, response_format
            )
            _SERVED_REGION.set(region)
            transcribe: TranscribeServiceClient = get_client("transcribe", region)

            # Track resources for cleanup
            to_cleanup = (transcribe, request_id)
            track_temporary_s3_objects(
                s3_bucket,
                f"{s3_prefix}{request_id}/output.json",
                f"{s3_prefix}{request_id}/.write_access_check_file.temp",
            )
            if response_format in SUBTITLE_FORMATS:
                track_temporary_s3_objects(
                    s3_bucket, f"{s3_prefix}{request_id}/output.{response_format}"
                )

            # Wait for completion and get results
            await _wait_for_transcription_completion(transcribe, request_id)
            return await _get_transcription_results(
                s3_bucket, request_id, response_format
            )

        finally:
            if to_cleanup:
                schedule_cleanup(_delete_transcription_job(*to_cleanup))

    @classmethod
    async def _format_transcription_response(
        cls,
        transcript_data: TranscribeJobData,
        response_format: AudioResponseFormat,
        billed_seconds: int,
        duration: float,
        timestamp_granularities: list[AudioTimestampGranularities] | None = None,
        filename: str | None = None,
    ) -> str | TranscriptionCreateResponse | TranscriptionDiarized | Response:
        """Format transcription response for the route.

        Args:
            transcript_data: AWS Transcribe job data
            response_format: Requested response format
            billed_seconds: Pre-computed billed seconds.
            duration: Pre-computed audio duration in seconds.
            timestamp_granularities: Optional timestamp granularities for verbose_json
            filename: Original filename of the audio file

        Returns:
            Formatted response in the requested format
        """
        if response_format in SUBTITLE_FORMATS:
            return await cls._format_subtitle_response(
                response_format, transcript_data["subtitle_content"], filename
            )

        text = _get_transcript_text(transcript_data)

        if response_format == "text":
            return text

        usage_duration = UsageDuration(type="duration", seconds=billed_seconds)

        if response_format == "verbose_json":
            return _format_json_response(
                transcript_data, text, duration, usage_duration, timestamp_granularities
            )
        if response_format == "diarized_json":
            return _format_diarized_json_response(
                transcript_data, text, duration, usage_duration
            )
        return log_response_params(Transcription(text=text, usage=usage_duration))

    @classmethod
    async def _format_translation_response(
        cls,
        transcript_data: TranscribeJobData,
        translated_content: str,
        response_format: AudioResponseFormat,
        filename: str | None = None,
    ) -> str | TranslationCreateResponse | Response:
        """Format translation response for the route.

        Args:
            transcript_data: AWS Transcribe job data
            translated_content: Pre-translated text or subtitle content
            response_format: Requested response format
            filename: Original filename of the audio file

        Returns:
            Formatted response in the requested format
        """
        if "subtitle_content" in transcript_data:
            return await cls._format_subtitle_response(
                response_format, translated_content, filename
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
        audio_content: InputFile,
        response_format: AudioResponseFormat,
        language: str | None = None,
        timestamp_granularities: list[AudioTimestampGranularities] | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        *,
        logprobs: bool,
    ) -> str | TranscriptionCreateResponse | TranscriptionDiarized | Response:
        """Perform transcription task using AWS Transcribe.

        This function handles the entire transcription workflow from audio upload
        through AWS Transcribe processing to result retrieval, including AWS client
        initialization and cleanup management.

        Args:
            audio_content: Audio file content file
            response_format: Format for the output response (json, text, srt, vtt, verbose_json, diarized_json)
            language: Optional language code for the input audio (ISO-639-1 format)
            timestamp_granularities: Optional timestamp granularities for verbose_json
            prompt: Optional prompt for transcription.
            temperature: Optional temperature for transcription.
            logprobs: If true, return log probabilities.

        Returns:
            Formatted transcription response

        Raises:
            ApiError: When transcription fails, validation errors occur, or
                unsupported file formats are provided
        """
        self._validate_response_formats(response_format, timestamp_granularities)
        transcript_data = await self._transcribe(
            audio_content,
            response_format,
            language,
            prompt,
            temperature,
            logprobs=logprobs,
        )
        duration = _get_audio_duration(transcript_data)
        return await self._format_transcription_response(
            transcript_data,
            response_format,
            record_transcribe_usage(duration, region=_SERVED_REGION.get()),
            duration,
            timestamp_granularities,
            await audio_content.get_filename(),
        )

    async def stt_stream(
        self,
        audio_content: InputFile,
        response_format: AudioResponseFormat,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        *,
        logprobs: bool,  # noqa: ARG002
    ) -> AsyncGenerator[TranscriptionTextDeltaEvent | TranscriptionTextDoneEvent]:
        """Transcribe audio to text with streaming response.

        Args:
            audio_content: Audio file to transcribe
            response_format: Format for output
            language: Optional language code
            prompt: Optional prompt for transcription.
            temperature: Optional temperature for transcription.
            logprobs: If true, return log probabilities.

        Yields:
            TranscriptionTextDeltaEvent or TranscriptionTextDoneEvent objects

        Raises:
            ApiError: When transcription fails
        """
        transcript_data = await self._transcribe(
            audio_content, response_format, language, prompt, temperature
        )
        record_transcribe_usage(
            _get_audio_duration(transcript_data), region=_SERVED_REGION.get()
        )
        full_text_parts: list[str] = []
        for transcript in transcript_data["transcripts"]:
            text = transcript["transcript"]
            full_text_parts.append(text)
            yield TranscriptionTextDeltaEvent(delta=text, type="transcript.text.delta")

        # UsageDuration not supported in streaming mode
        yield TranscriptionTextDoneEvent(
            text=" ".join(full_text_parts), type="transcript.text.done"
        )

    async def stt_translate(
        self,
        audio_content: InputFile,
        response_format: AudioResponseFormat,
        prompt: str | None,
        temperature: float | None = None,
    ) -> str | TranslationCreateResponse | Response:
        """Transcribe and translate audio to English.

        This method performs transcription using AWS Transcribe, detects the source
        language, and translates the transcribed text to English using AWS Translate.

        Args:
            audio_content: Audio file to transcribe and translate
            response_format: Format for output (json, text, srt, vtt, verbose_json)
            prompt: Optional prompt for translation.
            temperature: Optional temperature for transcription.

        Returns:
            Formatted translation response with translated text in English

        Raises:
            ApiError: When transcription or translation fails
        """
        transcript_data = await self._transcribe(
            audio_content, response_format, prompt=prompt, temperature=temperature
        )
        record_transcribe_usage(
            _get_audio_duration(transcript_data), region=_SERVED_REGION.get()
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

        return await self._format_translation_response(
            transcript_data,
            translated_content,
            response_format,
            await audio_content.get_filename(),
        )
