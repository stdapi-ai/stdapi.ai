"""Amazon Transcribe model implementation."""

from asyncio import gather, sleep
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Literal, NotRequired, cast
from zlib import compress

from botocore.exceptions import ClientError, ParamValidationError
from fastapi import Response
from pydantic_core import from_json
from typing_extensions import TypedDict

from stdapi.api_errors import (
    ApiError,
    InvalidLanguageFormatError,
    UnsupportedParameterError,
)
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
from stdapi.types import BaseModelResponse, JsonMapping
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
from stdapi.utils import (
    format_language_code,
    language_code_to_name,
    validation_error_handler,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_transcribe.client import TranscribeServiceClient
    from types_aiobotocore_transcribe.type_defs import (
        StartTranscriptionJobRequestTypeDef,
        ToxicityDetectionSettingsTypeDef,
    )

    from stdapi.input_file import InputFile

#: Transcribe model ID
AWS_TRANSCRIBE_MODEL_ID = "amazon.transcribe"

#: Ordinal value of the "A" letter used as speaker label
_A_ORDINAL_VALUE = ord("A")


class _TranscribeContentRedaction(BaseModelResponse):
    """PII redaction configuration; only the single-output mode is supported (see #82)."""

    RedactionType: Literal["PII"] = "PII"
    RedactionOutput: Literal["redacted"] = "redacted"
    # PII entity type values (e.g. "NAME", "SSN") are not re-validated here;
    # AWS Transcribe rejects an unknown one with its own 400 error.
    PiiEntityTypes: list[str] | None = None


class _TranscribeToxicityDetectionSetting(BaseModelResponse):
    """One entry of the ToxicityDetection list."""

    # Only "ALL" is documented today, but the category value is forwarded
    # as-is: AWS Transcribe rejects an unknown one with its own 400 error.
    ToxicityCategories: list[str]


class _TranscribeModelSettings(BaseModelResponse):
    """Custom language model selection."""

    LanguageModelName: str


class _TranscribeExtraParams(BaseModelResponse):
    """Supported extra parameters for AWS Transcribe's StartTranscriptionJob.

    ``Settings``' sub-fields are flattened to the top level (mirroring Polly's
    extra-params convention) so the ``Settings``/``TerminologyNames`` keys stay
    free for AWS Translate's own extra parameters on the translation route.
    """

    # MaxAlternatives/MaxSpeakerLabels/VocabularyFilterMethod are forwarded
    # as-is (not range/enum-checked here): AWS Transcribe rejects an
    # out-of-range or unknown value with its own 400 error.
    ChannelIdentification: bool | None = None
    ContentRedaction: _TranscribeContentRedaction | None = None
    IdentifyMultipleLanguages: bool | None = None
    LanguageOptions: list[str] | None = None
    MaxAlternatives: int | None = None
    MaxSpeakerLabels: int | None = None
    ModelSettings: _TranscribeModelSettings | None = None
    ShowAlternatives: bool | None = None
    ShowSpeakerLabels: bool | None = None
    ToxicityDetection: list[_TranscribeToxicityDetectionSetting] | None = None
    VocabularyFilterMethod: str | None = None
    VocabularyFilterName: str | None = None
    VocabularyName: str | None = None


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
    extra: _TranscribeExtraParams | None = None,
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
        extra: Optional extra StartTranscriptionJob parameters.

    Returns:
        The (region, bucket) pair that accepted the job.

    Raises:
        ApiError: For caller errors (unsupported language, bad file, or
            incompatible extra parameters).
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
                    job_id, bucket, language, response_format, extra
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


#: Default speaker cap for diarized_json output, overridable via extra_params.MaxSpeakerLabels
_DEFAULT_MAX_SPEAKER_LABELS = 10

#: Settings sub-fields flattened onto _TranscribeExtraParams (see its docstring)
_SETTINGS_FIELDS: set[str] = {
    "ChannelIdentification",
    "MaxAlternatives",
    "MaxSpeakerLabels",
    "ShowAlternatives",
    "ShowSpeakerLabels",
    "VocabularyFilterMethod",
    "VocabularyFilterName",
    "VocabularyName",
}


def _apply_language_params(
    job_params: StartTranscriptionJobRequestTypeDef,
    language: str | None,
    extra: _TranscribeExtraParams | None,
) -> None:
    """Set the job's language-identification fields (mutually exclusive per AWS).

    Args:
        job_params: Job parameters to update in place.
        language: Optional explicit language code.
        extra: Optional extra StartTranscriptionJob parameters.
    """
    if extra is not None and extra.IdentifyMultipleLanguages:
        job_params["IdentifyMultipleLanguages"] = True
        if extra.LanguageOptions:
            job_params["LanguageOptions"] = extra.LanguageOptions  # type: ignore[typeddict-item]
    elif language:
        job_params["LanguageCode"] = format_language_code(language)  # type: ignore[typeddict-item]
    else:
        job_params["IdentifyLanguage"] = True
        if extra is not None and extra.LanguageOptions:
            job_params["LanguageOptions"] = extra.LanguageOptions  # type: ignore[typeddict-item]


#: Parameter name reported when ChannelIdentification conflicts with diarized_json
_CHANNEL_IDENTIFICATION_PARAM = "ChannelIdentification"


def _apply_extra_settings(
    job_params: StartTranscriptionJobRequestTypeDef,
    response_format: str,
    extra: _TranscribeExtraParams | None,
) -> None:
    """Merge Settings/ContentRedaction/ModelSettings/ToxicityDetection into the job.

    Args:
        job_params: Job parameters to update in place.
        response_format: Response format for transcription.
        extra: Optional extra StartTranscriptionJob parameters.

    Raises:
        UnsupportedParameterError: If ``extra.ChannelIdentification`` is combined
            with ``diarized_json`` output (AWS rejects that combination; Transcribe
            already forces ``ShowSpeakerLabels`` for diarization).
    """
    settings: dict[str, object] = (
        {"ShowSpeakerLabels": True, "MaxSpeakerLabels": _DEFAULT_MAX_SPEAKER_LABELS}
        if response_format == "diarized_json"
        else {}
    )
    if extra is not None:
        if extra.ChannelIdentification and response_format == "diarized_json":
            raise UnsupportedParameterError(_CHANNEL_IDENTIFICATION_PARAM)
        settings.update(extra.model_dump(include=_SETTINGS_FIELDS, exclude_none=True))
        if extra.ContentRedaction is not None:
            job_params["ContentRedaction"] = extra.ContentRedaction.model_dump()  # type: ignore[typeddict-item]
        if extra.ModelSettings is not None:
            job_params["ModelSettings"] = extra.ModelSettings.model_dump()  # type: ignore[typeddict-item]
        if extra.ToxicityDetection is not None:
            job_params["ToxicityDetection"] = cast(
                "list[ToxicityDetectionSettingsTypeDef]",
                [setting.model_dump() for setting in extra.ToxicityDetection],
            )
    if settings:
        job_params["Settings"] = settings  # type: ignore[typeddict-item]


def _build_transcription_job_params(
    job_id: str,
    s3_bucket: str,
    language: str | None,
    response_format: str,
    extra: _TranscribeExtraParams | None = None,
) -> StartTranscriptionJobRequestTypeDef:
    """Build transcription job parameters.

    Args:
        job_id: Unique job identifier
        s3_bucket: S3 bucket name
        language: Optional language code
        response_format: Response format for transcription
        extra: Optional extra StartTranscriptionJob parameters

    Returns:
        Job parameters for AWS Transcribe

    Raises:
        UnsupportedParameterError: If ``extra.ChannelIdentification`` is combined
            with ``diarized_json`` output (AWS rejects that combination; Transcribe
            already forces ``ShowSpeakerLabels`` for diarization).
    """
    s3_prefix = SETTINGS.aws_s3_tmp_prefix
    job_params: StartTranscriptionJobRequestTypeDef = {
        "TranscriptionJobName": job_id,
        "Media": {"MediaFileUri": f"s3://{s3_bucket}/{s3_prefix}{job_id}/input"},
        "OutputBucketName": s3_bucket,
        "OutputKey": f"{s3_prefix}{job_id}/output.json",
        "Tags": [{"Key": k, "Value": v} for k, v in build_metadata(apn=True).items()],
    }

    _apply_language_params(job_params, language, extra)

    if response_format in SUBTITLE_FORMATS:
        # AWS Transcribe will create subtitle file at: {s3_prefix}{job_id}/output.{format}
        job_params["Subtitles"] = {"Formats": [response_format], "OutputStartIndex": 1}

    _apply_extra_settings(job_params, response_format, extra)

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
    except ParamValidationError as error:
        # botocore validates some Settings/ContentRedaction/ToxicityDetection
        # fields client-side (e.g. MaxAlternatives below its minimum); surface
        # it as a caller 400 instead of an unhandled 500.
        raise ApiError(str(error)) from error


def _s3_key_from_uri(uri: str, s3_bucket: str) -> str:
    """Return the S3 object key from a Transcribe output URI.

    Args:
        uri: ``https://s3.<region>.amazonaws.com/<bucket>/<key>`` output URI.
        s3_bucket: Bucket the job wrote to.

    Returns:
        The object key.
    """
    return uri.split(f"/{s3_bucket}/", 1)[-1]


async def _wait_for_transcription_completion(
    transcribe: TranscribeServiceClient, job_id: str, s3_bucket: str
) -> tuple[str, str | None]:
    """Wait for transcription job to complete and return its output keys.

    Content redaction renames the output: Transcribe prepends ``redacted-`` to the
    requested ``OutputKey`` file name and reports it as ``RedactedTranscriptFileUri``
    instead of ``TranscriptFileUri``. Reading the keys back from the job description
    keeps this independent of the naming convention.

    Args:
        transcribe: Transcribe service client
        job_id: Transcription job ID
        s3_bucket: Bucket the job writes its output to

    Returns:
        The transcript object key, and the subtitle object key when one was requested.

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

    transcript = job["Transcript"]
    subtitle_uris = job.get("Subtitles", {}).get("SubtitleFileUris")
    return (
        _s3_key_from_uri(
            transcript.get("RedactedTranscriptFileUri")
            or transcript["TranscriptFileUri"],
            s3_bucket,
        ),
        _s3_key_from_uri(subtitle_uris[0], s3_bucket) if subtitle_uris else None,
    )


async def _get_transcription_results(
    s3_bucket: str, s3_output_key: str, subtitle_key: str | None
) -> TranscribeJobData:
    """Get transcription results from S3.

    Args:
        s3_bucket: S3 bucket name
        s3_output_key: Transcript object key
        subtitle_key: Subtitle object key, when a subtitle format was requested

    Returns:
        Transcription data
    """
    if subtitle_key:
        data, subtitle = await gather(
            get_text_from_s3(s3_bucket, s3_output_key),
            get_text_from_s3(s3_bucket, subtitle_key),
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


#: avg_logprob reported for silent segments, below Whisper's "-1 suggests failed" threshold
_SILENT_SEGMENT_AVG_LOGPROB = -2.0


def _text_compression_ratio(text: str) -> float:
    """Compute the zlib-based compression ratio, mirroring Whisper's own metric.

    AWS Transcribe exposes no decoder-internal compression ratio; a high
    ratio (compressed size much smaller than the original) signals
    repetitive or hallucinated text, computed here directly from the text.

    Args:
        text: Segment text to measure.

    Returns:
        Ratio of the raw UTF-8 byte length to its zlib-compressed length, or
        0.0 for empty text.
    """
    if not text:
        return 0.0
    raw = text.encode("utf-8")
    return len(raw) / len(compress(raw))


def _build_transcription_segment(
    segment: TranscribeJobAudioSegment, text: str
) -> TranscriptionSegment:
    """Build a verbose-JSON segment, approximating Whisper-only confidence stats.

    AWS Transcribe exposes no per-token log-probabilities or sampling
    parameters: avg_logprob/no_speech_prob are derived from whether the
    segment carries a transcript (so the documented combined silence signal
    still fires), and compression_ratio is computed on the returned text.

    Args:
        segment: Raw AWS Transcribe audio segment.
        text: Segment text to report (source or translated).

    Returns:
        The formatted verbose-JSON segment.
    """
    has_transcript = bool(segment["transcript"])
    return TranscriptionSegment(
        id=segment["id"],
        end=float(segment["end_time"]),
        start=float(segment["start_time"]),
        text=text,
        no_speech_prob=0.0 if has_transcript else 1.0,
        avg_logprob=0.0 if has_transcript else _SILENT_SEGMENT_AVG_LOGPROB,
        compression_ratio=_text_compression_ratio(text),
        # Not supported: Transcribe has no seek offset, sampling temperature, or tokens
        seek=0,
        temperature=0.0,
        tokens=[],
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

    # OpenAI defaults to segment-level timestamps when the parameter is omitted.
    if not timestamp_granularities or "segment" in timestamp_granularities:
        segments = [
            _build_transcription_segment(segment, segment["transcript"])
            for segment in transcript_data["audio_segments"]
        ]
    if timestamp_granularities and "word" in timestamp_granularities:
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


def _pop_translate_extra_params(
    extra_params: JsonMapping | None,
) -> tuple[JsonMapping | None, JsonMapping | None, list[str] | None]:
    """Split AWS Translate's Settings/TerminologyNames out of the extra params.

    The remainder stays reserved for the underlying StartTranscriptionJob call
    (see ``_TranscribeExtraParams``); ``Settings``/``TerminologyNames`` never
    collide with it since Transcribe's own ``Settings`` sub-fields are
    flattened onto the top level there.

    Args:
        extra_params: Raw extra parameters from the request.

    Returns:
        (remaining extra params for the Transcribe job, Translate ``Settings``,
        Translate ``TerminologyNames``).
    """
    if not extra_params:
        return extra_params, None, None
    remaining = dict(extra_params)
    settings = remaining.pop("Settings", None)
    terminology_names = remaining.pop("TerminologyNames", None)
    # Both are handed to AWS Translate, which validates their shape and returns
    # a clean 4xx; narrowing them here would only duplicate that check.
    return (remaining or None, settings, terminology_names)  # type: ignore[return-value]


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
        extra_params: JsonMapping | None = None,
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
            extra_params: Optional extra StartTranscriptionJob parameters.
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

        extra: _TranscribeExtraParams | None = None
        if extra_params:
            with validation_error_handler():
                extra = _TranscribeExtraParams(**extra_params)  # type: ignore[arg-type]

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
                candidates, request_id, language, response_format, extra
            )
            _SERVED_REGION.set(region)
            transcribe: TranscribeServiceClient = get_client("transcribe", region)

            # Track resources for cleanup
            to_cleanup = (transcribe, request_id)
            track_temporary_s3_objects(
                s3_bucket, f"{s3_prefix}{request_id}/.write_access_check_file.temp"
            )

            # Wait for completion and get results
            s3_output_key, subtitle_key = await _wait_for_transcription_completion(
                transcribe, request_id, s3_bucket
            )
            track_temporary_s3_objects(
                s3_bucket, s3_output_key, *filter(None, (subtitle_key,))
            )
            return await _get_transcription_results(
                s3_bucket, s3_output_key, subtitle_key
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
            return Response(content=text, media_type="text/plain; charset=utf-8")

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
        settings: JsonMapping | None = None,
        terminology_names: list[str] | None = None,
    ) -> str | TranslationCreateResponse | Response:
        """Format translation response for the route.

        Args:
            transcript_data: AWS Transcribe job data
            translated_content: Pre-translated text or subtitle content
            response_format: Requested response format
            filename: Original filename of the audio file
            settings: Optional AWS Translate ``Settings`` for the per-segment
                translate calls used by ``verbose_json``.
            terminology_names: Optional AWS Translate custom terminology names
                for the per-segment translate calls used by ``verbose_json``.

        Returns:
            Formatted response in the requested format
        """
        if "subtitle_content" in transcript_data:
            return await cls._format_subtitle_response(
                response_format, translated_content, filename
            )

        if response_format == "text":
            return Response(
                content=translated_content, media_type="text/plain; charset=utf-8"
            )

        if response_format == "verbose_json":
            audio_segments = transcript_data["audio_segments"]
            translated_segments = await gather(
                *(
                    translate(
                        segment["transcript"],
                        transcript_data["language_code"],
                        settings=settings,
                        terminology_names=terminology_names,
                    )
                    for segment in audio_segments
                )
            )
            return TranslationVerbose(
                duration=_get_audio_duration(transcript_data),
                language="english",  # Translation output is always English
                text=translated_content,
                segments=[
                    _build_transcription_segment(segment, segment_text)
                    for segment, segment_text in zip(
                        audio_segments, translated_segments, strict=True
                    )
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
        extra_params: JsonMapping | None = None,
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
            extra_params: Optional extra StartTranscriptionJob parameters.
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
            extra_params,
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
        extra_params: JsonMapping | None = None,
        *,
        logprobs: bool,
    ) -> AsyncGenerator[TranscriptionTextDeltaEvent | TranscriptionTextDoneEvent]:
        """Transcribe audio to text with streaming response.

        Args:
            audio_content: Audio file to transcribe
            response_format: Format for output
            language: Optional language code
            prompt: Optional prompt for transcription.
            temperature: Optional temperature for transcription.
            extra_params: Optional extra StartTranscriptionJob parameters.
            logprobs: If true, return log probabilities.

        Yields:
            TranscriptionTextDeltaEvent or TranscriptionTextDoneEvent objects

        Raises:
            ApiError: When transcription fails
            UnsupportedParameterError: When ``logprobs`` is requested, as on the
                non-streaming path -- Transcribe returns no log probabilities.
        """
        self._validate_no_logprobs(logprobs)
        transcript_data = await self._transcribe(
            audio_content, response_format, language, prompt, temperature, extra_params
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
        extra_params: JsonMapping | None = None,
    ) -> str | TranslationCreateResponse | Response:
        """Transcribe and translate audio to English.

        This method performs transcription using AWS Transcribe, detects the source
        language, and translates the transcribed text to English using AWS Translate.

        Args:
            audio_content: Audio file to transcribe and translate
            response_format: Format for output (json, text, srt, vtt, verbose_json)
            prompt: Optional prompt for translation.
            temperature: Optional temperature for transcription.
            extra_params: Optional extra StartTranscriptionJob parameters.

        Returns:
            Formatted translation response with translated text in English

        Raises:
            ApiError: When transcription or translation fails
        """
        job_extra_params, settings, terminology_names = _pop_translate_extra_params(
            extra_params
        )
        transcript_data = await self._transcribe(
            audio_content,
            response_format,
            prompt=prompt,
            temperature=temperature,
            extra_params=job_extra_params,
        )
        record_transcribe_usage(
            _get_audio_duration(transcript_data), region=_SERVED_REGION.get()
        )
        language = transcript_data["language_code"]
        if "subtitle_content" in transcript_data:
            translated_content = await translate_subtitle(
                transcript_data["subtitle_content"],
                language,
                settings=settings,
                terminology_names=terminology_names,
            )
        else:
            translated_content = await translate(
                _get_transcript_text(transcript_data),
                language,
                settings=settings,
                terminology_names=terminology_names,
            )

        return await self._format_translation_response(
            transcript_data,
            translated_content,
            response_format,
            await audio_content.get_filename(),
            settings,
            terminology_names,
        )
