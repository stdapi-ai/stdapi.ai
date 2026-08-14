"""Amazon Polly TTS model implementation."""

from asyncio import CancelledError, create_task, gather, sleep
from contextlib import contextmanager, suppress
from re import compile as re_compile
from time import monotonic
from typing import TYPE_CHECKING, Literal

from aws_sdk_polly.models import (
    CloseStreamEvent,
    Engine,
    LanguageCode,
    OutputFormat,
    StartSpeechSynthesisStreamActionStreamCloseStreamEvent,
    StartSpeechSynthesisStreamActionStreamTextEvent,
    StartSpeechSynthesisStreamEventStreamAudioEvent,
    StartSpeechSynthesisStreamEventStreamStreamClosedEvent,
    StartSpeechSynthesisStreamInput,
    TextEvent,
    TextType,
    VoiceId,
)
from botocore.exceptions import BotoCoreError, ClientError, ParamValidationError

from stdapi.api_errors import ApiError, UnsupportedModelError
from stdapi.aws import call_with_region_failover, get_client, service_regions
from stdapi.aws_bidi import BidiSession, open_bidi_stream
from stdapi.aws_s3 import (
    get_s3_bucket_for_region,
    s3_key_from_uri,
    track_temporary_s3_objects,
)
from stdapi.config import SETTINGS
from stdapi.media import encode_audio_stream, stream_body
from stdapi.models import (
    EXTRA_MODELS,
    EXTRA_MODELS_INPUT_MODALITY,
    EXTRA_MODELS_OUTPUT_MODALITY,
    ModelDetails,
)
from stdapi.models.audio import AudioModelBase, TTSResponse
from stdapi.monitoring import (
    REQUEST_LOG,
    EventLog,
    add_server_warning,
    log_error_details,
)
from stdapi.types import BaseModelResponse, JsonMapping
from stdapi.types.openai_audio import OPENAI_VOICES_FEMALE
from stdapi.usage import record_comprehend_usage, record_polly_usage
from stdapi.utils import format_language_code, validation_error_handler

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Generator, Sequence
    from typing import Any

    from smithy_core.aio.eventstream import DuplexEventStream
    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_comprehend.client import ComprehendClient
    from types_aiobotocore_comprehend.type_defs import (
        DetectDominantLanguageResponseTypeDef,
    )
    from types_aiobotocore_polly.client import PollyClient
    from types_aiobotocore_polly.literals import (
        EngineType,
        GenderType,
        LanguageCodeType,
        OutputFormatType,
        TextTypeType,
        VoiceIdType,
    )
    from types_aiobotocore_polly.type_defs import (
        DescribeVoicesInputTypeDef,
        StartSpeechSynthesisTaskInputTypeDef,
        SynthesisTaskTypeDef,
        SynthesizeSpeechInputTypeDef,
        SynthesizeSpeechOutputTypeDef,
        VoiceTypeDef,
    )
    from types_aiobotocore_s3.client import S3Client

    from stdapi.types.openai_audio import AudioFileFormat

#: Model prefix for AWS Polly (Mimic Bedrock AWS models)
_PREFIX = "amazon.polly-"

#: Polly output format name if different from the response format name
_FORMAT: dict[str, OutputFormatType] = {"ogg": "ogg_vorbis", "opus": "ogg_opus"}

#: Formats to encode from PCM using ffmpeg (Not supported by Polly natively)
_FORMAT_ENCODE = {"wav", "flac", "aac"}

#: Polly's default sample rate for pcm output when none is requested
_POLLY_DEFAULT_PCM_SAMPLE_RATE = 16000

#: Polly pcm supported sample rates
_PCM_SAMPLE_RATES = {8000, 16000}

#: OpenAI's documented pcm response_format contract: 24 kHz, 16-bit, mono, little-endian
_OPENAI_PCM_SAMPLE_RATE = 24000

#: Content type of Polly's speech marks output (a stream of JSON lines)
_SPEECH_MARKS_CONTENT_TYPE = "application/x-json-stream"

#: Sample size for language detection
_LANG_DETECT_SAMPLE_SIZE = 500

#: Polly errors to return as 400 errors
_CLIENT_VALIDATION_ERRORS = {
    "TextLengthExceededException",
    "InvalidSampleRateException",
    "InvalidSsmlException",
    "LexiconNotFoundException",
    "MarksNotSupportedForFormatException",
    "SsmlMarksNotSupportedForTextTypeException",
    "LanguageNotSupportedException",
    "EngineNotSupportedException",
}

#: Polly errors reporting that the audio cannot be stored where it was requested
_STORAGE_ERRORS = frozenset({"InvalidS3BucketException", "InvalidS3KeyException"})

#: Billed characters (SSML markup excluded) a single synthesis call accepts
_MAX_BILLED_CHARACTERS = 3000

#: Total characters (SSML markup included) a single synthesis call accepts
_MAX_CHARACTERS = 6000

#: Billed characters one ten-minute incremental synthesis renders, with margin
_STREAM_MAX_BILLED_CHARACTERS = 20_000

#: Engine synthesizing text incrementally; the others reject the operation
_STREAM_ENGINE: EngineType = "generative"

#: Opening tag of a document the caller wrote, which is synthesized unchanged
_SSML_DOCUMENT_PREFIX = "<speak>"

#: Event ending the text a stream carries, before its input half is closed
_CLOSE_STREAM_EVENT = StartSpeechSynthesisStreamActionStreamCloseStreamEvent(
    CloseStreamEvent()
)

#: Billed characters (SSML markup excluded) a synthesis job accepts
_JOB_MAX_BILLED_CHARACTERS = 100_000

#: Total characters (SSML markup included) a synthesis job accepts
_JOB_MAX_CHARACTERS = 200_000

#: SSML markup, which Polly excludes from its billed character count
_SSML_TAG = re_compile(r"<[^>]*>")

#: Key prefix of the audio objects a synthesis job writes
_JOB_KEY_PREFIX = f"{SETTINGS.aws_s3_tmp_prefix}speech"

#: Initial synthesis job status poll interval, in seconds
_POLL_INTERVAL_INITIAL = 0.5

#: Maximum synthesis job status poll interval after exponential backoff, in seconds
_POLL_INTERVAL_MAX = 5.0

#: Answer to a long input this server cannot synthesize
_LONG_INPUT_UNAVAILABLE = (
    f"'input' is limited to {_MAX_BILLED_CHARACTERS:,} characters "
    f"({_MAX_CHARACTERS:,} including SSML markup) on this server. Split the text "
    "into shorter requests, or contact the administrator to enable longer inputs."
)

#: Answer to an input longer than one incremental synthesis renders
_STREAM_INPUT_TOO_LONG = (
    f"'input' is limited to {_STREAM_MAX_BILLED_CHARACTERS:,} characters for this "
    "voice on this server. Split the text into shorter requests, or contact the "
    "administrator to enable longer inputs."
)

#: Supported Polly models
_SUPPORTED_SPEECH_MODELS: set[str] = {
    f"{_PREFIX}standard",
    f"{_PREFIX}neural",
    f"{_PREFIX}long-form",
    f"{_PREFIX}generative",
}

_VOICES_DESCRIPTIONS: dict[VoiceIdType, str] = {}
_VOICES_BY_GENDERS: dict[GenderType, set[VoiceIdType]] = {}
_VOICES_BY_LANGUAGE: dict[LanguageCodeType, set[VoiceIdType]] = {}
_VOICES_BY_ENGINE: dict[EngineType, set[VoiceIdType]] = {}
_VOICES_BY_NAME_LOWER: dict[str, VoiceIdType] = {}

#: Voices per engine and region, in candidate priority order (non-empty only).
_VOICES_BY_ENGINE_REGION: dict[EngineType, dict[RegionName, set[VoiceIdType]]] = {}


class _PollyExtraParams(BaseModelResponse):
    """Supported extra parameters for Polly."""

    LanguageCode: str | None = None
    LexiconNames: list[str] | None = None
    SampleRate: int | None = None
    # Not value-constrained: Polly's own MarksNotSupportedForFormatException
    # already maps an unknown/unsupported mark type to a 400 error.
    SpeechMarkTypes: list[str] | None = None


def _engine_from_model(model: str) -> EngineType:
    """Retrieve engine from model name.

    Args:
        model: Model name.

    Returns:
        Engine name.
    """
    if model not in _SUPPORTED_SPEECH_MODELS:
        raise UnsupportedModelError(model)
    return model.removeprefix(_PREFIX)  # type: ignore[return-value]


async def _get_voices_per_engine(
    engine: EngineType, region: RegionName
) -> set[VoiceIdType]:
    """Retrieve one region's voices for an engine from Polly.

    The voices' metadata is merged into the region-independent lookup tables
    only once every page succeeded, so a failure mid-pagination leaves no
    partial metadata behind.

    Args:
        engine: The engine to filter voices for.
        region: The region to query.

    Returns:
        Voice IDs the engine supports in the region; empty when the engine
        is not available there (Polly then returns no voices, not an error).
    """
    next_token = None
    polly: PollyClient = get_client("polly", region)
    engine_voices: set[VoiceIdType] = set()
    voices: list[VoiceTypeDef] = []
    params: DescribeVoicesInputTypeDef = {"Engine": engine}
    while True:
        if next_token:
            params["NextToken"] = next_token
        response = await polly.describe_voices(**params)
        voices.extend(response["Voices"])
        next_token = response.get("NextToken")
        if not next_token:
            break
    for voice in voices:
        voice_id = voice["Id"]
        gender = voice["Gender"]
        engine_voices.add(voice_id)
        _VOICES_DESCRIPTIONS[voice_id] = f"{gender}, {voice['LanguageName']}"
        _VOICES_BY_GENDERS.setdefault(gender, set()).add(voice_id)
        _VOICES_BY_LANGUAGE.setdefault(voice["LanguageCode"], set()).add(voice_id)
        _VOICES_BY_NAME_LOWER[voice_id.lower()] = voice_id
    return engine_voices


async def initialize_polly_models(start_event: EventLog | None = None) -> None:
    """Initialize voices for all models across every candidate region.

    Engine availability is discovered per region (Polly engines are not
    offered everywhere): an engine is registered as a model when at least
    one candidate region has voices for it, and synthesis later routes to
    those regions only. A (engine, region) pair whose voice retrieval fails
    with an AWS error is skipped (warned about on *start_event*) instead of
    failing startup.

    Args:
        start_event: Optional startup event log to record warnings on for
            engine/region pairs whose voices could not be retrieved.
    """
    _VOICES_DESCRIPTIONS.clear()
    _VOICES_BY_GENDERS.clear()
    _VOICES_BY_LANGUAGE.clear()
    _VOICES_BY_ENGINE.clear()
    _VOICES_BY_NAME_LOWER.clear()
    _VOICES_BY_ENGINE_REGION.clear()
    regions = service_regions(SETTINGS.aws_polly_region)
    pairs = [
        (_engine_from_model(model), region)
        for model in _SUPPORTED_SPEECH_MODELS
        for region in regions
    ]
    results = await gather(
        *(_get_voices_per_engine(engine, region) for engine, region in pairs),
        return_exceptions=True,
    )
    failed: dict[str, str] = {}
    for (engine, region), result in zip(pairs, results, strict=True):
        if isinstance(result, BaseException):
            if not isinstance(result, (BotoCoreError, ClientError)):
                raise result
            failed[f"{engine}@{region}"] = f"{type(result).__name__}: {result}"
        elif result:
            _VOICES_BY_ENGINE_REGION.setdefault(engine, {})[region] = result
    if failed and start_event is not None:
        add_server_warning(
            start_event,
            {"unavailable_polly_engines": failed},  # type: ignore[dict-item]
        )
    # Intentionally lenient: if every region failed, this registers zero models
    # instead of raising, since Polly is an optional service.
    for engine, voices_by_region in _VOICES_BY_ENGINE_REGION.items():
        _VOICES_BY_ENGINE[engine] = set().union(*voices_by_region.values())
        model_id = f"amazon.polly-{engine}"
        EXTRA_MODELS_INPUT_MODALITY.setdefault("TEXT", set()).add(model_id)
        EXTRA_MODELS_OUTPUT_MODALITY.setdefault("SPEECH", set()).add(model_id)
        EXTRA_MODELS[model_id] = ModelDetails(
            id=model_id,
            name=f"Polly {engine.capitalize()}",
            provider="Amazon",
            regions=[*voices_by_region],
            service="AWS Polly",
            input_modalities=["TEXT"],
            output_modalities=["SPEECH"],
            response_streaming=True,
        )


def _engine_voice_regions(engine: EngineType, voice_id: str) -> list[RegionName]:
    """Return the candidate regions able to synthesize a voice with an engine.

    Args:
        engine: Polly engine.
        voice_id: Selected voice ID (possibly a raw, undiscovered name).

    Returns:
        Regions offering the engine with this voice, falling back to all
        regions offering the engine, then to every candidate region (the
        voice/engine is then left to Polly to accept or reject).
    """
    voices_by_region = _VOICES_BY_ENGINE_REGION.get(engine, {})
    if regions := [
        region for region, voices in voices_by_region.items() if voice_id in voices
    ]:
        return regions
    return [*voices_by_region] or service_regions(SETTINGS.aws_polly_region)


async def _select_voice(
    text: str, voice: str, engine: EngineType
) -> tuple[VoiceIdType, LanguageCodeType | None]:
    """Select a voice based on OpenAI compatibility.

    Args:
        text: Input text for language detection.
        voice: OpenAI voice name.
        engine: AWS Polly engine.

    Returns:
        Voice ID and optional language code.
    """
    if (voice_lower := voice.lower()) in _VOICES_BY_NAME_LOWER:
        return _VOICES_BY_NAME_LOWER[voice_lower], None

    try:
        gender: GenderType = "Female" if OPENAI_VOICES_FEMALE[voice] else "Male"
    except KeyError:
        return voice, None  # type: ignore[return-value]
    # Ordered, deduplicated: try the detected language before the en-US fallback.
    for language in dict.fromkeys((await _detect_language(text), "en-US")):
        candidates = (
            _VOICES_BY_GENDERS[gender]
            & _VOICES_BY_LANGUAGE[language]
            & _VOICES_BY_ENGINE[engine]
        )
        if candidates:
            return min(candidates), language
    return voice, None  # type: ignore[return-value]


async def _detect_language(text: str) -> LanguageCodeType:
    """Detect language from a short sample of the full text.

    Uses the configured default language when set, else AWS Comprehend,
    falling back to English.

    Args:
        text: Text to detect language from.

    Returns:
        Language code.
    """
    if SETTINGS.default_tts_language is None:
        sample_text = (
            text
            if len(text) <= _LANG_DETECT_SAMPLE_SIZE
            else text[
                : (
                    pos
                    if (pos := text.rfind(" ", 0, _LANG_DETECT_SAMPLE_SIZE)) != -1
                    else _LANG_DETECT_SAMPLE_SIZE
                )
            ]
        )

        def _detect(
            comprehend: ComprehendClient, _region: RegionName
        ) -> Awaitable[DetectDominantLanguageResponseTypeDef]:
            """Start the language detection call on one region's client."""
            return comprehend.detect_dominant_language(Text=sample_text)

        response, used_region = await call_with_region_failover(
            "comprehend", service_regions(SETTINGS.aws_comprehend_region), _detect
        )
        record_comprehend_usage(
            len(sample_text), "language-detection", region=used_region
        )
        if response.get("Languages"):
            language = format_language_code(
                max(response["Languages"], key=lambda x: x["Score"])["LanguageCode"]
            )
            if language in _VOICES_BY_LANGUAGE:
                return language
        return "en-US"
    return SETTINGS.default_tts_language  # type: ignore[return-value]


def _prosody_document(text: str, speed: float) -> str:
    """Wrap text in the SSML document carrying a non-default speaking rate.

    Args:
        text: Text to speak at that rate.
        speed: Speed multiplier for speech.

    Returns:
        A self-contained SSML document.
    """
    return f'<speak><prosody rate="{int(speed * 100)}%">{text}</prosody></speak>'


def _prepare_text_for_speech(input_text: str, speed: float) -> tuple[str, TextTypeType]:
    """Prepare text for speech synthesis with speed adjustment.

    Args:
        input_text: Original input text
        speed: Speed multiplier for speech

    Returns:
        Tuple of (processed_text, text_type)
    """
    if input_text.startswith(_SSML_DOCUMENT_PREFIX):
        return input_text, "ssml"
    if speed != 1.0:
        return _prosody_document(input_text, speed), "ssml"
    return input_text, "text"


@contextmanager
def _handle_polly_error(
    model_id: str, voice_id: str, engine: EngineType
) -> Generator[None]:
    """Context manager to handle Polly service errors and raise appropriate HTTP exceptions.

    Args:
        model_id: model ID.
        voice_id: voice ID.
        engine: Polly engine being used.

    Yields:
        None

    Raises:
        ApiError: With the appropriate error message and status code.
    """
    try:
        yield
    except ClientError as error:
        if (
            error.response["Error"]["Code"] == "ValidationException"
            and "voice" in error.response["Error"]["Message"]
        ):
            voices = tuple(
                f"{voice_id} ({details})"
                for voice_id, details in _VOICES_DESCRIPTIONS.items()
                if voice_id in _VOICES_BY_ENGINE[engine]
            )
            message = (
                f"Available voices: {'; '.join(voices)}"
                if voices
                else "Ensure this model is available for your region"
            )
            msg = f"Voice '{voice_id}' not found for model '{model_id}'. {message}."
            raise ApiError(msg) from error
        if error.response["Error"]["Code"] in _CLIENT_VALIDATION_ERRORS:
            raise ApiError(error.response["Error"]["Message"]) from error
        raise  # pragma: no cover
    except ParamValidationError as error:
        # botocore validates some request fields client-side (e.g. SampleRate's
        # type); surface it as a caller 400 instead of an unhandled 500.
        log_error_details(str(error))
        msg = "Invalid speech synthesis settings."
        raise ApiError(msg) from error


def _synthesis_transport(
    text: str, text_type: TextTypeType, *, streamable: bool, job_available: bool
) -> Literal["call", "stream", "job"]:
    """Return how the text is synthesized: in one call, incrementally, or as a job.

    Polly limits a single call to 3,000 billed characters and 6,000 in total,
    counting SSML markup towards the second only, so the decision is made on
    the final text and both limits are checked. Beyond them, incremental
    synthesis comes first: it returns audio as it is produced and needs no
    storage, where a job returns nothing until it completes and writes to a
    bucket. Only its ten-minute duration sends the longest inputs back to a job.

    Args:
        text: Text as it is sent for synthesis.
        text_type: Whether that text is plain text or an SSML document.
        streamable: Whether this voice and this text can be synthesized
            incrementally.
        job_available: Whether a region able to serve the request can also
            store a job's audio.

    Returns:
        The transport to synthesize the text with.

    Raises:
        ApiError: The text is longer than this server synthesizes, the
            rejection naming the limit actually enforced.
    """
    # Counted by subtracting the markup, which never copies the document itself.
    billed = (
        len(text) - sum(map(len, _SSML_TAG.findall(text)))
        if text_type == "ssml"
        else len(text)
    )
    if billed <= _MAX_BILLED_CHARACTERS and len(text) <= _MAX_CHARACTERS:
        return "call"
    if streamable and billed <= _STREAM_MAX_BILLED_CHARACTERS:
        return "stream"
    if not job_available:
        if streamable:
            raise ApiError(_STREAM_INPUT_TOO_LONG)
        log_error_details(
            "No S3 bucket configured (aws_s3_bucket, aws_s3_regional_buckets) for "
            f"the speech synthesis regions: inputs over {_MAX_BILLED_CHARACTERS} "
            "characters are rejected"
        )
        raise ApiError(_LONG_INPUT_UNAVAILABLE)
    if billed > _JOB_MAX_BILLED_CHARACTERS or len(text) > _JOB_MAX_CHARACTERS:
        msg = (
            f"'input' is limited to {_JOB_MAX_BILLED_CHARACTERS:,} characters "
            f"({_JOB_MAX_CHARACTERS:,} including SSML markup)."
        )
        raise ApiError(msg)
    return "job"


def _synthesis_job_candidates(
    engine: EngineType, voice_id: str
) -> list[tuple[RegionName, str]]:
    """Return the candidate (region, S3 bucket) pairs for a synthesis job.

    A job writes its audio to a bucket in its own region, so a region without
    one cannot serve it.

    Args:
        engine: Polly engine.
        voice_id: Selected voice ID.

    Returns:
        (region, bucket) pairs in priority order; empty when no region able to
        synthesize the voice has a bucket.
    """
    return [
        (region, bucket)
        for region in _engine_voice_regions(engine, voice_id)
        if (bucket := get_s3_bucket_for_region(region))
    ]


async def _start_synthesis_job(
    request: SynthesizeSpeechInputTypeDef, candidates: list[tuple[RegionName, str]]
) -> tuple[SynthesisTaskTypeDef, RegionName, str]:
    """Start a synthesis job, failing over across the candidate regions.

    No failed-region cleanup: a start that errors returns no job identifier,
    and Polly offers no way to stop a job that may have been created anyway.

    Args:
        request: Synthesis request, as built for a single call.
        candidates: (region, bucket) pairs in priority order (at least one).

    Returns:
        The started job, and the region and bucket that accepted it: the
        status polling and the download both belong there.

    Raises:
        BotoCoreError: When every candidate region fails (last error).
        ClientError: Same as above.
    """
    buckets = dict(candidates)

    async def _start(polly: PollyClient, region: RegionName) -> SynthesisTaskTypeDef:
        """Start the job writing to one region's co-located bucket."""
        params: StartSpeechSynthesisTaskInputTypeDef = {
            **request,
            "OutputS3BucketName": buckets[region],
            "OutputS3KeyPrefix": _JOB_KEY_PREFIX,
        }
        return (await polly.start_speech_synthesis_task(**params))["SynthesisTask"]

    job, region = await call_with_region_failover(
        "polly", [region for region, _ in candidates], _start
    )
    return job, region, buckets[region]


async def _wait_for_synthesis_job(region: RegionName, job_id: str) -> None:
    """Poll a synthesis job in its own region until its audio is available.

    Args:
        region: Region that accepted the job.
        job_id: Synthesis job identifier.

    Raises:
        ApiError: The job failed, or is still running after
            ``ai_response_timeout`` seconds.
    """
    polly: PollyClient = get_client("polly", region)
    deadline = monotonic() + SETTINGS.ai_response_timeout
    interval = _POLL_INTERVAL_INITIAL
    while True:
        job = (await polly.get_speech_synthesis_task(TaskId=job_id))["SynthesisTask"]
        status = job["TaskStatus"]
        if status == "completed":
            return
        if status == "failed":
            # The reason names the backend and its storage, and stays in the log.
            log_error_details(job.get("TaskStatusReason", status), status=503)
            msg = "The speech could not be synthesized. Retry the request."
            raise ApiError(msg, status=503)
        if monotonic() >= deadline:
            log_error_details(
                f"Speech synthesis still '{status}' after "
                f"{SETTINGS.ai_response_timeout}s (ai_response_timeout)",
                status=503,
            )
            msg = "The speech synthesis timed out. Retry with a shorter 'input'."
            raise ApiError(msg, status=503)
        await sleep(interval)
        interval = min(interval * 2, _POLL_INTERVAL_MAX)


async def _synthesis_job_audio(
    region: RegionName, bucket: str, key: str
) -> AsyncGenerator[bytes]:
    """Stream a completed job's audio from the region that served it.

    Args:
        region: Region that served the job.
        bucket: Bucket the job wrote its audio to.
        key: Key of the audio object.

    Returns:
        The audio stream.
    """
    s3: S3Client = get_client("s3", region)
    return stream_body((await s3.get_object(Bucket=bucket, Key=key))["Body"])


async def _synthesize_long_text(
    request: SynthesizeSpeechInputTypeDef,
    model_id: str,
    engine: EngineType,
    voice_id: str,
    candidates: list[tuple[RegionName, str]],
) -> tuple[AsyncGenerator[bytes], int]:
    """Synthesize text too long for a single call, as a job.

    Args:
        request: Synthesis request, as built for a single call.
        model_id: Model ID, for error messages.
        engine: Polly engine.
        voice_id: Selected voice ID.
        candidates: (region, bucket) pairs in priority order (at least one).

    Returns:
        The audio stream, and the billed character count.

    Raises:
        ApiError: No candidate bucket accepts the audio, or the job failed or
            timed out.
    """
    try:
        with _handle_polly_error(model_id, voice_id, engine):
            job, region, bucket = await _start_synthesis_job(request, candidates)
    except ClientError as error:
        if error.response["Error"]["Code"] not in _STORAGE_ERRORS:
            raise
        log_error_details(error.response["Error"]["Message"])
        raise ApiError(_LONG_INPUT_UNAVAILABLE) from error

    # Tracked as soon as the job is accepted, so no path leaves the object behind.
    key = s3_key_from_uri(job["OutputUri"], bucket)
    track_temporary_s3_objects(bucket, key)

    # Billed as soon as Polly accepts the text, whatever the job then does.
    input_tokens = record_polly_usage(
        job.get("RequestCharacters", 0), engine, region=region
    )
    await _wait_for_synthesis_job(region, job["TaskId"])
    return await _synthesis_job_audio(region, bucket, key), input_tokens


def _stream_text_events(text: str, speed: float) -> list[TextEvent]:
    """Split text into the events one incremental synthesis carries.

    An event takes what a single call takes, so longer text is cut on a space
    to keep words whole -- Polly reassembles the text itself, so a cut needs no
    sentence boundary. A speed envelope is rebuilt around every chunk, because
    an SSML document may not span events. Only text no caller wrote as a
    document reaches here, so a chunk is spoken as written whatever it contains.

    Args:
        text: Plain text to synthesize, as the caller sent it.
        speed: Speech speed multiplier.

    Returns:
        The text events to send, in order.
    """
    events = []
    spoken_as_written = speed == 1.0
    text_type = TextType("text" if spoken_as_written else "ssml")
    start = 0
    while start < len(text):
        end = start + _MAX_BILLED_CHARACTERS
        if end >= len(text):
            chunk = text[start:]
        else:
            cut = text.rfind(" ", start, end)
            chunk = text[start : end if cut == -1 else cut + 1]
        start += len(chunk)
        events.append(
            TextEvent(
                text=chunk if spoken_as_written else _prosody_document(chunk, speed),
                text_type=text_type,
            )
        )
    return events


async def _send_speech_text(
    session: BidiSession[Any, Any], events: Sequence[TextEvent]
) -> None:
    """Send the whole text, then end the input half of the stream.

    Polly ends a session five seconds after the last input event, and its close
    event does not stop that timer: closing the input half is what tells the
    service the text is complete, and what lets it return the rest of the audio.

    Args:
        session: The open synthesis session.
        events: The text events to send, in order.
    """
    for event in events:
        await session.send(StartSpeechSynthesisStreamActionStreamTextEvent(event))
    await session.send(_CLOSE_STREAM_EVENT)
    await session.close_input()


async def _stream_speech_audio(
    stream_input: StartSpeechSynthesisStreamInput,
    events: Sequence[TextEvent],
    characters: int,
    engine: EngineType,
    regions: list[RegionName],
) -> AsyncGenerator[bytes]:
    """Synthesize text incrementally, yielding audio as it is produced.

    Args:
        stream_input: Synthesis parameters the stream is opened with.
        events: The text events to send, in order.
        characters: Billed characters the events carry, billed when the session
            ends without reporting its own count.
        engine: Polly engine, for usage attribution.
        regions: Candidate regions, in priority order.

    Yields:
        An empty chunk once the service accepted the request, then every audio
        chunk it produces. The marker is what makes a rejected request a caller
        error instead of a response body that stops after one byte.

    Raises:
        ApiError: No region accepted the request, or the session failed before
            the whole text had been spoken.
    """

    async def _open(
        client: Any,  # noqa: ANN401
        _region: RegionName,
    ) -> DuplexEventStream[Any, Any, Any]:
        """Open the synthesis stream on one region's client."""
        stream: DuplexEventStream[
            Any, Any, Any
        ] = await client.start_speech_synthesis_stream(input=stream_input)
        return stream

    async with open_bidi_stream("polly", regions, _open) as session:
        yield b""
        billed: int | None = None
        sender = create_task(_send_speech_text(session, events))
        try:
            async for event in session:
                if isinstance(event, StartSpeechSynthesisStreamEventStreamAudioEvent):
                    if chunk := event.value.audio_chunk:
                        yield chunk
                elif isinstance(
                    event, StartSpeechSynthesisStreamEventStreamStreamClosedEvent
                ):
                    billed = event.value.request_characters
            # An error here truncated the audio, so it must not be swallowed.
            await sender
        finally:
            if not sender.done():
                sender.cancel()
            # Awaited on every path: an unretrieved error surfaces late, in asyncio.
            with suppress(CancelledError, Exception):
                await sender
            if billed is None:
                log_error_details(
                    "The speech synthesis stream ended without reporting its "
                    f"usage: billing the {characters} characters sent.",
                    level="warning",
                )
            record_polly_usage(
                characters if billed is None else billed, engine, region=session.region
            )


async def _synthesize_streamed_text(
    request: SynthesizeSpeechInputTypeDef,
    text: str,
    speed: float,
    sample_rate: int | None,
    model_id: str,
    engine: EngineType,
    voice_id: str,
    candidates: list[tuple[RegionName, str]],
) -> tuple[AsyncGenerator[bytes], int]:
    """Synthesize text incrementally, as a job where no stream can be opened.

    The fallback covers a deployment whose permissions or region do not offer
    the operation: the request is then served the way it was before it existed,
    and only a deployment that cannot run a job either sees the failure.

    Args:
        request: Synthesis request, as built for a single call.
        text: Text to synthesize, as the caller sent it.
        speed: Speech speed multiplier.
        sample_rate: Requested sample rate in Hz, None for the format default.
        model_id: Model ID, for error messages.
        engine: Polly engine.
        voice_id: Selected voice ID.
        candidates: (region, bucket) pairs a job could run in, possibly empty.

    Returns:
        The audio stream, and the billed character count.

    Raises:
        ApiError: The request is not one Polly accepts, or no stream could be
            opened and no job can serve the request.
    """
    audio = _stream_speech_audio(
        StartSpeechSynthesisStreamInput(
            engine=Engine(engine),
            voice_id=VoiceId(voice_id),
            output_format=OutputFormat(request["OutputFormat"]),
            sample_rate=None if sample_rate is None else str(sample_rate),
            language_code=(
                LanguageCode(language)
                if (language := request.get("LanguageCode"))
                else None
            ),
            lexicon_names=request.get("LexiconNames"),
        ),
        _stream_text_events(text, speed),
        len(text),
        engine,
        _engine_voice_regions(engine, voice_id),
    )
    try:
        # Last point a failure can still be a clean error and be retried elsewhere.
        await anext(audio)
    except ApiError as error:
        if not candidates:
            raise
        log_error_details(
            f"Incremental speech synthesis unavailable ({error}): "
            "synthesizing as a job instead.",
            level="warning",
        )
        return await _synthesize_long_text(
            request, model_id, engine, voice_id, candidates
        )
    # The count the response reports; Polly bills its own normalisation, later.
    return audio, len(text)


async def _synthesize_text(
    request: SynthesizeSpeechInputTypeDef,
    model_id: str,
    engine: EngineType,
    voice_id: str,
) -> tuple[AsyncGenerator[bytes], int]:
    """Synthesize text in a single call, failing over across candidate regions.

    Args:
        request: Synthesis request.
        model_id: Model ID, for error messages.
        engine: Polly engine.
        voice_id: Selected voice ID.

    Returns:
        The audio stream, and the billed character count.

    Raises:
        ApiError: The request is not one Polly accepts.
    """

    def _synthesize(
        polly: PollyClient, _region: RegionName
    ) -> Awaitable[SynthesizeSpeechOutputTypeDef]:
        """Start the speech synthesis call on one region's client."""
        return polly.synthesize_speech(**request)

    with _handle_polly_error(model_id, voice_id, engine):
        response, region = await call_with_region_failover(
            "polly", _engine_voice_regions(engine, voice_id), _synthesize
        )

    input_tokens = record_polly_usage(
        int(response["RequestCharacters"]), engine, region=region
    )
    return stream_body(response["AudioStream"]), input_tokens


class AudioModel(AudioModelBase[None, None]):
    """Amazon Polly audio model implementation (TTS only)."""

    __slots__ = ()

    MATCHER = _PREFIX

    @classmethod
    def get_aliases(
        cls,
        all_models: dict[str, ModelDetails],  # noqa: ARG003
    ) -> dict[str, str]:
        """Return the OpenAI TTS model names mapped to Polly engines.

        Args:
            all_models: All available models keyed by model ID.

        Returns:
            A dict mapping alias to model ID.
        """
        return {"tts-1": "amazon.polly-standard", "tts-1-hd": "amazon.polly-neural"}

    async def tts(
        self,
        text: str,
        voice: str,
        resp_format: AudioFileFormat,
        speed: float = 1.0,
        extra_params: JsonMapping | None = None,
    ) -> TTSResponse:
        """Generate audio from text using AWS Polly.

        Text longer than a single synthesis call accepts is synthesized as a
        job instead, which needs an S3 bucket in the serving region.

        Args:
            text: Text to convert to speech.
            voice: Voice to use.
            resp_format: Audio format.
            speed: Speech speed multiplier.
            extra_params: Extra model parameters.

        Returns:
            TTS response with audio stream, or with a JSON speech marks stream
            when the "SpeechMarkTypes" extra parameter is used.
        """
        log = REQUEST_LOG.get()
        extra: _PollyExtraParams | None = None
        if extra_params:
            with validation_error_handler():
                extra = _PollyExtraParams(
                    **extra_params  # type: ignore[arg-type]
                )
        # Polly returns speech marks as JSON lines under "OutputFormat=json",
        # so "response_format" is ignored and nothing is re-encoded.
        speech_marks = extra is not None and bool(extra.SpeechMarkTypes)
        # pcm without an explicit SampleRate extra param is also routed through
        # ffmpeg, so it can be resampled to OpenAI's documented 24 kHz contract.
        explicit_sample_rate = bool(extra and extra.SampleRate)
        encoding = not speech_marks and (
            resp_format in _FORMAT_ENCODE
            or (resp_format == "pcm" and not explicit_sample_rate)
        )
        output_format: OutputFormatType = (
            "json"
            if speech_marks
            else ("pcm" if encoding else _FORMAT.get(resp_format, resp_format))  # type: ignore[arg-type]
        )
        # Polly's own default PCM sample rate, used unless the caller overrides it.
        sample_rate = _POLLY_DEFAULT_PCM_SAMPLE_RATE if encoding else None
        # Force OpenAI's 24 kHz pcm contract, unless the caller pinned Polly's rate.
        output_sample_rate = (
            _OPENAI_PCM_SAMPLE_RATE if encoding and resp_format == "pcm" else None
        )

        engine = _engine_from_model(self.model.id)
        voice_id, language = await _select_voice(text, voice, engine)
        log["voice_id"] = voice_id
        streamable = (
            engine == _STREAM_ENGINE
            # A stream produces no timing marks, and an SSML document must stay whole.
            and not speech_marks
            and not text.startswith(_SSML_DOCUMENT_PREFIX)
            # A stream's rejection would not name the voices that do exist.
            and voice_id in _VOICES_BY_ENGINE.get(engine, ())
        )
        plain_text = text
        text, text_type = _prepare_text_for_speech(text, speed)

        request: SynthesizeSpeechInputTypeDef = {
            "Engine": engine,
            "Text": text,
            "TextType": text_type,
            "OutputFormat": output_format,
            "VoiceId": voice_id,
        }
        if language:
            request["LanguageCode"] = language
        if extra is not None:
            request.update(  # type: ignore[call-arg]
                **extra.model_dump(exclude_none=True)
            )
            if extra.SampleRate is not None:
                sample_rate = extra.SampleRate
                request["SampleRate"] = str(extra.SampleRate)
                if encoding and sample_rate not in _PCM_SAMPLE_RATES:
                    # Polly pcm caps at 16 kHz; encode from Ogg Vorbis above it.
                    output_format = request["OutputFormat"] = "ogg_vorbis"

        # Before the length check, so the rejection names the limit actually enforced.
        candidates = _synthesis_job_candidates(engine, voice_id)
        match _synthesis_transport(
            text, text_type, streamable=streamable, job_available=bool(candidates)
        ):
            case "stream":
                body, input_tokens = await _synthesize_streamed_text(
                    request,
                    plain_text,
                    speed,
                    sample_rate,
                    self.model.id,
                    engine,
                    voice_id,
                    candidates,
                )
            case "job":
                body, input_tokens = await _synthesize_long_text(
                    request, self.model.id, engine, voice_id, candidates
                )
            case _:
                body, input_tokens = await _synthesize_text(
                    request, self.model.id, engine, voice_id
                )

        if encoding:
            # channels/sample_rate only apply to the raw-pcm source: _ffmpeg_args
            # ignores both when input_format is unset (encoded source, autodetected).
            is_pcm_source = output_format == "pcm"
            audio_stream = encode_audio_stream(
                body,
                resp_format,
                input_format="s16le" if is_pcm_source else None,
                channels=1 if is_pcm_source else None,
                sample_rate=sample_rate if is_pcm_source else None,
                output_sample_rate=output_sample_rate,
            )
        else:
            audio_stream = body

        return TTSResponse(
            audio_stream=audio_stream,
            input_tokens=input_tokens,
            output_tokens=0,
            content_type=_SPEECH_MARKS_CONTENT_TYPE if speech_marks else None,
        )
