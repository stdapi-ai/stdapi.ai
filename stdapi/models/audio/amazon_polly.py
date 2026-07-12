"""Amazon Polly TTS model implementation."""

from asyncio import gather
from contextlib import contextmanager
from typing import TYPE_CHECKING

from botocore.exceptions import BotoCoreError, ClientError

from stdapi.api_errors import ApiError, UnsupportedModelError
from stdapi.aws import get_client
from stdapi.config import SETTINGS
from stdapi.media import encode_audio_stream, stream_body
from stdapi.models import (
    EXTRA_MODELS,
    EXTRA_MODELS_INPUT_MODALITY,
    EXTRA_MODELS_OUTPUT_MODALITY,
    ModelDetails,
)
from stdapi.models.audio import AudioModelBase, TTSResponse
from stdapi.monitoring import REQUEST_LOG, EventLog, add_server_warning
from stdapi.types import BaseModelResponse, JsonMapping
from stdapi.types.openai_audio import OPENAI_VOICES_FEMALE
from stdapi.usage import record_comprehend_usage, record_polly_usage
from stdapi.utils import format_language_code, validation_error_handler

if TYPE_CHECKING:
    from collections.abc import Generator

    from types_aiobotocore_comprehend.client import ComprehendClient
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
    )

    from stdapi.types.openai_audio import AudioFileFormat

#: Model prefix for AWS Polly (Mimic Bedrock AWS models)
_PREFIX = "amazon.polly-"

#: Polly output format name if different from the response format name
_FORMAT: dict[str, OutputFormatType] = {"ogg": "ogg_vorbis", "opus": "ogg_opus"}

#: Formats to encode from PCM using ffmpeg (Not supported by Polly natively)
_FORMAT_ENCODE = {"wav", "flac", "aac"}

#: Polly PCM supported sample rates
_PCM_SAMPLE_RATES = {8000, 16000}

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


class _PollyExtraParams(BaseModelResponse):
    """Supported extra parameters for Polly."""

    LanguageCode: str | None = None
    LexiconNames: str | None = None
    SampleRate: int | None = None


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


async def _get_voices_per_engine(engine: EngineType) -> None:
    """Retrieve voices from Polly.

    Args:
        engine: The engine to filter voices for.
    """
    next_token = None
    polly: PollyClient = get_client("polly")
    engine_voices = _VOICES_BY_ENGINE[engine] = set()
    params: DescribeVoicesInputTypeDef = {"Engine": engine}
    while True:
        if next_token:
            params["NextToken"] = next_token
        response = await polly.describe_voices(**params)
        for voice in response["Voices"]:
            voice_id = voice["Id"]
            gender = voice["Gender"]
            engine_voices.add(voice_id)
            _VOICES_DESCRIPTIONS[voice_id] = f"{gender}, {voice['LanguageName']}"
            _VOICES_BY_GENDERS.setdefault(gender, set()).add(voice_id)
            _VOICES_BY_LANGUAGE.setdefault(voice["LanguageCode"], set()).add(voice_id)
            _VOICES_BY_NAME_LOWER[voice_id.lower()] = voice_id
        next_token = response.get("NextToken")
        if not next_token:
            break


async def initialize_polly_models(start_event: EventLog | None = None) -> None:
    """Initialize voices for all models.

    An engine whose voice retrieval fails with an AWS error is disabled
    (warned about on *start_event*) instead of failing startup; text-to-
    speech keeps working for the other engines.

    Args:
        start_event: Optional startup event log to record warnings on for
            engines whose voices could not be retrieved.
    """
    _VOICES_DESCRIPTIONS.clear()
    _VOICES_BY_GENDERS.clear()
    _VOICES_BY_LANGUAGE.clear()
    _VOICES_BY_ENGINE.clear()
    _VOICES_BY_NAME_LOWER.clear()
    engines = [_engine_from_model(model) for model in _SUPPORTED_SPEECH_MODELS]
    results = await gather(
        *(_get_voices_per_engine(engine) for engine in engines), return_exceptions=True
    )
    failed: dict[str, str] = {}
    for engine, result in zip(engines, results, strict=True):
        if isinstance(result, BaseException):
            if not isinstance(result, (BotoCoreError, ClientError)):
                raise result
            failed[engine] = f"{type(result).__name__}: {result}"
            # Drop any partially paginated voice set for the failed engine.
            _VOICES_BY_ENGINE.pop(engine, None)
    if failed and start_event is not None:
        add_server_warning(
            start_event,
            {"unavailable_polly_engines": failed},  # type: ignore[dict-item]
        )
    polly: PollyClient = get_client("polly")
    for engine, voices in _VOICES_BY_ENGINE.items():
        if voices:
            model_id = f"amazon.polly-{engine}"
            EXTRA_MODELS_INPUT_MODALITY.setdefault("TEXT", set()).add(model_id)
            EXTRA_MODELS_OUTPUT_MODALITY.setdefault("SPEECH", set()).add(model_id)
            EXTRA_MODELS[model_id] = ModelDetails(
                id=model_id,
                name=f"Polly {engine.capitalize()}",
                provider="Amazon",
                regions=[polly.meta.region_name],  # type: ignore[list-item]
                service="AWS Polly",
                input_modalities=["TEXT"],
                output_modalities=["SPEECH"],
                response_streaming=True,
            )


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
    for language in {await _detect_language(text), "en-US"}:
        candidates = (
            _VOICES_BY_GENDERS[gender]
            & _VOICES_BY_LANGUAGE[language]
            & _VOICES_BY_ENGINE[engine]
        )
        if candidates:
            return sorted(candidates)[0], language
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
        comprehend: ComprehendClient = get_client("comprehend")
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
        response = await comprehend.detect_dominant_language(Text=sample_text)
        record_comprehend_usage(len(sample_text), "language-detection")
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


class AudioModel(AudioModelBase[None, None]):
    """Amazon Polly audio model implementation (TTS only)."""

    MATCHER = _PREFIX

    @classmethod
    def get_aliases(cls, all_models: dict[str, ModelDetails]) -> dict[str, str]:  # noqa: ARG003
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
            TTS response with audio stream.
        """
        log = REQUEST_LOG.get()
        encoding = resp_format in _FORMAT_ENCODE
        output_format: OutputFormatType = (
            "ogg_vorbis" if encoding else _FORMAT.get(resp_format, resp_format)  # type: ignore[arg-type]
        )
        sample_rate = None

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
        if extra_params:
            with validation_error_handler():
                extra = _PollyExtraParams(
                    **extra_params  # type: ignore[arg-type]
                )
            request.update(  # type: ignore[call-arg]
                **extra.model_dump(exclude_none=True)
            )
            if extra.SampleRate:
                sample_rate = extra.SampleRate
                request["SampleRate"] = str(extra.SampleRate)
                if encoding and sample_rate in _PCM_SAMPLE_RATES:
                    # Use lossless PCM if supported instead of a lossy Vorbis
                    output_format = request["OutputFormat"] = "pcm"

        polly: PollyClient = get_client("polly")
        with _handle_polly_error(self.model.id, voice_id, engine):
            response = await polly.synthesize_speech(**request)

        input_tokens = record_polly_usage((int(response["RequestCharacters"])), engine)

        body = stream_body(response["AudioStream"])
        if encoding:
            audio_stream = encode_audio_stream(
                body,
                resp_format,
                input_format="s16le" if output_format == "pcm" else None,
                channels=1,
                sample_rate=sample_rate,
            )
        else:
            audio_stream = body

        return TTSResponse(
            audio_stream=audio_stream, input_tokens=input_tokens, output_tokens=0
        )
