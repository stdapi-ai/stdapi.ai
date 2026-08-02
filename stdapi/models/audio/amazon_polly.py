"""Amazon Polly TTS model implementation."""

from asyncio import gather
from contextlib import contextmanager
from typing import TYPE_CHECKING

from botocore.exceptions import BotoCoreError, ClientError, ParamValidationError

from stdapi.api_errors import ApiError, UnsupportedModelError
from stdapi.aws import call_with_region_failover, get_client, service_regions
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
    from collections.abc import Awaitable, Generator

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
        SynthesizeSpeechInputTypeDef,
        SynthesizeSpeechOutputTypeDef,
        VoiceTypeDef,
    )

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

    Also merges the voices' metadata (description, gender, language, name)
    into the region-independent lookup tables, but only once the full
    listing (all pages) has succeeded, so a failure part-way through
    pagination leaves no partial metadata behind.

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

    If a default language is configured in settings, use it instead of auto-detection.
    Otherwise, use AWS Comprehend to detect the language.
    Fallback to English if no language is detected.

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


def _prepare_text_for_speech(input_text: str, speed: float) -> tuple[str, TextTypeType]:
    """Prepare text for speech synthesis with speed adjustment.

    Args:
        input_text: Original input text
        speed: Speed multiplier for speech

    Returns:
        Tuple of (processed_text, text_type)
    """
    if input_text.startswith("<speak>"):
        return input_text, "ssml"
    if speed != 1.0:
        return (
            f'<speak><prosody rate="{int(speed * 100)}%">{input_text}</prosody></speak>'
        ), "ssml"
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

    Usage:
        with _handle_polly_error(model_id, voice_id, engine):
            response = await polly.synthesize_speech(**request)
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


class AudioModel(AudioModelBase[None, None]):
    """Amazon Polly audio model implementation (TTS only)."""

    MATCHER = _PREFIX

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
        # Speech marks are timing metadata, not audio: Polly requires
        # "OutputFormat=json" and returns JSON lines instead of an audio
        # stream, so "response_format" is ignored and nothing is re-encoded.
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

        def _synthesize(
            polly: PollyClient, _region: RegionName
        ) -> Awaitable[SynthesizeSpeechOutputTypeDef]:
            """Start the speech synthesis call on one region's client."""
            return polly.synthesize_speech(**request)

        with _handle_polly_error(self.model.id, voice_id, engine):
            response, used_region = await call_with_region_failover(
                "polly", _engine_voice_regions(engine, voice_id), _synthesize
            )

        input_tokens = record_polly_usage(
            int(response["RequestCharacters"]), engine, region=used_region
        )

        body = stream_body(response["AudioStream"])
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
