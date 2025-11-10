"""OpenAI-compatible Text-to-Speech API implementation using AWS Polly."""

from asyncio import gather
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Annotated

from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, Response
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import JsonValue
from sse_starlette import EventSourceResponse, JSONServerSentEvent

from stdapi.auth import authenticate
from stdapi.aws_bedrock import get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.media import encode_audio_stream, stream_body
from stdapi.models import (
    EXTRA_MODELS,
    EXTRA_MODELS_INPUT_MODALITY,
    EXTRA_MODELS_OUTPUT_MODALITY,
    ModelDetails,
)
from stdapi.monitoring import (
    REQUEST_LOG,
    log_request_params,
    log_request_stream_event,
    log_response_params,
)
from stdapi.openai_exceptions import OpenaiUnsupportedModelError
from stdapi.types import BaseModelResponse
from stdapi.types.openai_audio import (
    AudioFileFormat,
    SpeechAudioDeltaEvent,
    SpeechAudioDoneEvent,
    SpeechCreateParams,
    SpeechUsage,
)
from stdapi.utils import b64encode, format_language_code, validation_error_handler

if TYPE_CHECKING:
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


from stdapi.aws import get_client

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1/audio", tags=["audio", "openai"]
)

#: Model prefix for AWS Polly (Mimic Bedrock AWS models)
_POLLY_PREFIX = "amazon.polly-"

#: Polly output format name if different from the response format name
_FORMAT_POLLY: "dict[str, OutputFormatType]" = {"ogg": "ogg_vorbis", "opus": "ogg_opus"}

#: Content-type format name if different from the response format name
_FORMAT_CONTENT_TYPE = {"mp3": "mpeg"}

#: Formats to encode from PCM using ffmpeg (Not supported by Polly natively)
_FORMAT_ENCODE = {"wav", "flac", "aac"}

#: Polly PCM supported sample rates
_POLLY_PCM_SAMPLE_RATES = {8000, 16000}

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
    f"{_POLLY_PREFIX}standard",
    f"{_POLLY_PREFIX}neural",
    f"{_POLLY_PREFIX}long-form",
    f"{_POLLY_PREFIX}generative",
}

_VOICES_DESCRIPTIONS: "dict[VoiceIdType, str]" = {}
_VOICES_BY_GENDERS: "dict[GenderType, set[VoiceIdType]]" = {}
_VOICES_BY_LANGUAGE: "dict[LanguageCodeType, set[VoiceIdType]]" = {}
_VOICES_BY_ENGINE: "dict[EngineType, set[VoiceIdType]]" = {}

#: OpenAI voices and matching gender (No Neutral support with Polly, fallback to Female)
_OPENAI_VOICES_GENDER: "dict[str, GenderType]" = {
    "alloy": "Female",
    "ash": "Male",
    "ballad": "Female",
    "coral": "Female",
    "echo": "Male",
    "fable": "Female",
    "nova": "Female",
    "onyx": "Male",
    "sage": "Female",
    "shimmer": "Female",
    "verse": "Male",
}


class _PollyExtraParams(BaseModelResponse):
    """Supported extra parameters for Polly."""

    LanguageCode: str | None = None
    LexiconNames: str | None = None
    SampleRate: int | None = None


def _engine_from_model(model: str) -> "EngineType":
    """Retrieve engine from model name.

    Args:
        model: Model name.

    Returns:
        Engine name.
    """
    if model not in _SUPPORTED_SPEECH_MODELS:
        raise OpenaiUnsupportedModelError(model)
    return model.removeprefix(_POLLY_PREFIX)  # type: ignore[return-value]


async def _get_voices_per_engine(engine: "EngineType") -> None:
    """Retrieve voices from Poly.

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
        next_token = response.get("NextToken")
        if not next_token:
            break


async def initialize_polly_models() -> None:
    """Initialize voices for all models."""
    _VOICES_DESCRIPTIONS.clear()
    _VOICES_BY_GENDERS.clear()
    _VOICES_BY_LANGUAGE.clear()
    _VOICES_BY_ENGINE.clear()
    await gather(
        *(
            _get_voices_per_engine(_engine_from_model(model))
            for model in _SUPPORTED_SPEECH_MODELS
        )
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
                region=polly.meta.region_name,
                service="AWS Polly",
                input_modalities=["TEXT"],
                output_modalities=["SPEECH"],
                response_streaming=True,
            )


async def _speech_audio_bytestream(
    stream: AsyncGenerator[bytes],
) -> AsyncGenerator[bytes]:
    """Generate real-time audio streaming, with logging.

    Args:
        stream: Audio stream yielding audio bytes chunks

    Yields:
        Audio stream yielding audio bytes chunks
    """
    try:
        async for chunk in stream:
            yield chunk
    finally:
        await stream.aclose()


async def _speech_audio_sse(
    stream: AsyncGenerator[bytes], characters_count: int
) -> AsyncGenerator[JSONServerSentEvent]:
    """Generate Server-Sent Events for real-time audio streaming.

    Converts audio stream into OpenAI-compatible Server-Sent Events
    for real-time audio delivery. Emits delta events for incremental
    audio chunks and a final done event with usage statistics.

    Args:
        stream: Audio stream yielding audio bytes chunks
        characters_count: Input text characters count for usage tracking

    Yields:
        JSONServerSentEvent: SSE events with speech.audio.delta and
            speech.audio.done event types following OpenAI streaming format
    """
    try:
        async for chunk in stream:
            yield JSONServerSentEvent(
                data=SpeechAudioDeltaEvent(audio=await b64encode(chunk)).model_dump(
                    mode="json", exclude_none=True
                )
            )
    finally:
        await stream.aclose()
        yield JSONServerSentEvent(
            data=log_response_params(
                SpeechAudioDoneEvent(
                    usage=SpeechUsage(
                        # Polly is billed by character count, not token
                        input_tokens=characters_count,
                        output_tokens=0,
                        total_tokens=characters_count,
                    )
                ).model_dump(mode="json", exclude_none=True)
            )
        )


async def _select_voice(
    text: str, voice: str, engine: "EngineType"
) -> "tuple[VoiceIdType, LanguageCodeType | None]":
    """Select a voice based on OpenAI compatibility.

    Args:
        text: Input text for language detection.
        voice: OpenAI voice name.
        engine: AWS Polly engine.

    Returns:
        Voice ID.
    """
    if voice in _VOICES_DESCRIPTIONS:
        return voice, None  # type: ignore[return-value]

    try:
        gender = _OPENAI_VOICES_GENDER[voice]
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


async def _detect_language(text: str) -> "LanguageCodeType":
    """Detect language from a short sample of the full text.

    Fallback to English if no language is detected.

    Args:
        text: Text to detect language from.

    Returns:
        Language code.
    """
    comprehend: ComprehendClient = get_client("comprehend")
    response = await comprehend.detect_dominant_language(
        Text=text
        if len(text) <= _LANG_DETECT_SAMPLE_SIZE
        else text[
            : (
                pos
                if (pos := text.rfind(" ", 0, _LANG_DETECT_SAMPLE_SIZE)) != -1
                else _LANG_DETECT_SAMPLE_SIZE
            )
        ]
    )
    if response.get("Languages"):
        language = format_language_code(
            max(response["Languages"], key=lambda x: x["Score"])["LanguageCode"]
        )
        if language in _VOICES_BY_LANGUAGE:
            return language  # type: ignore[return-value]
    return "en-US"


def _prepare_text_for_speech(
    input_text: str, speed: float
) -> tuple[str, "TextTypeType"]:
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
    model_id: str, voice_id: str, engine: "EngineType"
) -> Generator[None]:
    """Context manager to handle Polly service errors and raise appropriate HTTP exceptions.

    Args:
        model_id: model ID.
        voice_id: voice ID.
        engine: Polly engine being used

    Raises:
        HTTPException: With the appropriate error message and status code

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
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{voice_id}' not found for model '{model_id}'. {message}.",
            ) from error
        if error.response["Error"]["Code"] in _CLIENT_VALIDATION_ERRORS:
            raise HTTPException(
                status_code=400, detail=error.response["Error"]["Message"]
            ) from error
        raise  # pragma: no cover


async def generate_audio(
    text: str,
    model_id: str = SETTINGS.default_tts_model,
    voice: str = "alloy",
    resp_format: AudioFileFormat = "mp3",
    speed: float = 1.0,
    extra_params: dict[str, JsonValue] | None = None,
) -> AsyncGenerator[bytes]:
    """Generates audio from text using AWS Polly or Polly-compatible services.

    This method synthesizes speech from a given input text using a specific model,
    voice, and response format. It provides options to adjust playback speed and
    encodes the output if required. The synthesized audio is returned as a stream
    or formatted output.

    Args:
        text: The text to convert to speech.
        model_id: The ID of the speech synthesis model to use.
        voice: The desired voice for the synthesized speech.
        resp_format: The output audio file format.
        speed: The speed multiplier for the speech output. Defaults to 1.0.
        extra_params: Extra model parameters.

    Returns:
        A stream of the generated audio encoded in the specified format.
    """
    log = REQUEST_LOG.get()
    encoding = resp_format in _FORMAT_ENCODE
    output_format: OutputFormatType = (
        "ogg_vorbis" if encoding else _FORMAT_POLLY.get(resp_format, resp_format)  # type: ignore[arg-type]
    )
    sample_rate = None

    engine = _engine_from_model(model_id)
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
            if encoding and sample_rate in _POLLY_PCM_SAMPLE_RATES:
                # Use lossless PCM if supported instead of a lossy Vorbis
                output_format = request["OutputFormat"] = "pcm"

    polly: PollyClient = get_client("polly")
    with _handle_polly_error(model_id, voice_id, engine):
        response = await polly.synthesize_speech(**request)

    body = stream_body(response["AudioStream"])
    if encoding:
        return encode_audio_stream(
            body,
            resp_format,
            input_format="s16le" if output_format == "pcm" else None,
            channels=1,
            sample_rate=sample_rate,
        )
    return body


async def _create_speech_response(
    audio_stream: AsyncGenerator[bytes],
    request: SpeechCreateParams,
    resp_format: str,
    characters_count: int,
) -> Response:
    """Create the final speech response.

    Args:
        audio_stream: Audio stream from Polly
        request: Original speech request
        resp_format: Response format string
        characters_count: Length of input text

    Returns:
        Appropriate response (streaming or SSE)
    """
    audio_stream = await log_request_stream_event(audio_stream)
    if request.stream_format == "sse":
        return EventSourceResponse(_speech_audio_sse(audio_stream, characters_count))

    return StreamingResponse(
        content=_speech_audio_bytestream(audio_stream),
        media_type=f"audio/{_FORMAT_CONTENT_TYPE.get(resp_format, resp_format)}",
        headers={"Content-Disposition": f"attachment; filename=speech.{resp_format}"},
    )


@router.post(
    "/speech",
    summary="OpenAI - /v1/audio/speech",
    description=(
        "Generates audio from the input text using advanced text-to-speech models.\n"
        "Supports multiple voices, output formats, and playback speeds. Can stream using SSE."
    ),
    response_description="Returns audio file in the specified format",
    responses={
        200: {"description": "Audio generated (or streaming)."},
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "file": {
                            "summary": "Generate MP3 file",
                            "value": {
                                "model": "amazon.polly-standard",
                                "voice": "Amy",
                                "input": "Hello, I'am Amy",
                                "response_format": "mp3",
                            },
                        },
                        "sse": {
                            "summary": "Stream using SSE",
                            "value": {
                                "model": "amazon.polly-standard",
                                "voice": "Amy",
                                "input": "Hello, I'am Amy",
                                "response_format": "mp3",
                                "stream_format": "sse",
                            },
                        },
                    }
                }
            }
        }
    },
)
async def create_speech(
    request: SpeechCreateParams, _: Annotated[None, Depends(authenticate)] = None
) -> Response:
    """Generates audio from the input text.

    Converts input text to audio using advanced text-to-speech models. Supports
    multiple voices, output formats, and playback speeds. Provides both standard
    audio file responses and real-time streaming capabilities.

    Args:
        request: The text-to-speech request containing model, voice, and format parameters.

    Returns:
        Response: Audio file in the specified format or streaming response.

    Raises:
        HTTPException: When audio generation fails, validation errors occur, or
            unsupported voice/model combinations are provided.
    """
    log_request_params(request)
    log = REQUEST_LOG.get()
    log["model_id"] = model_id = request.model

    resp_format = request.response_format
    return await _create_speech_response(
        await generate_audio(
            request.input,
            model_id,
            request.voice,
            resp_format,
            request.speed,
            extra_params=get_extra_model_parameters(model_id, request),
        ),
        request,
        resp_format,
        len(request.input),
    )
