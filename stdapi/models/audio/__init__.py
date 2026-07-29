"""Audio models base classes and dynamic registry.

This package exposes the base interfaces for audio models (TTS and transcription)
and provides a minimal plugin/registry system that auto-loads model implementations
located in this package directory and resolves them by matching the model identifier.

Design:
- Audio model modules expose a class named `AudioModel` with a class variable `MATCHER`
  containing a string prefix or compiled regex matching model identifiers.
- Models can support TTS, transcription, or both capabilities.
- The package auto-loads and registers these classes once on import.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from fastapi import Response

from stdapi.api_errors import ApiError, UnsupportedParameterError
from stdapi.config import SETTINGS
from stdapi.models import ModelBase, get_model, load_model_plugins
from stdapi.models.capabilities import Capability
from stdapi.utils import language_code_to_name

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from re import Pattern

    from stdapi.input_file import InputFile
    from stdapi.types import JsonMapping
    from stdapi.types.openai_audio import (
        AudioFileFormat,
        AudioResponseFormat,
        AudioTimestampGranularities,
        TranscriptionCreateResponse,
        TranscriptionDiarized,
        TranscriptionTextDeltaEvent,
        TranscriptionTextDoneEvent,
        TranslationCreateResponse,
    )


class TTSResponse(TypedDict):
    """Text-to-speech response.

    Attributes:
        audio_stream: Async generator yielding audio bytes.
        input_tokens: Billed characters reported as tokens for OpenAI parity.
        output_tokens: 0 when the backend reports no output metric.
        content_type: Content type of the stream when it is not the audio
            format requested by the caller (e.g. a JSON speech marks stream).
            None or absent to use the requested audio format.
    """

    audio_stream: AsyncGenerator[bytes]
    input_tokens: int
    output_tokens: int
    content_type: NotRequired[str | None]


class AudioModelBase[RequestT, ResponseT](ModelBase[RequestT, ResponseT]):
    """Base class for provider-specific audio models supporting TTS and/or transcription."""

    #: Set of supported response formats. Used with _validate_response_formats
    SUPPORTED_RESPONSES_FORMATS: frozenset[AudioResponseFormat] = frozenset()

    #: Set of supported timestamp granularities. Used with _validate_response_formats
    SUPPORTED_TIMESTAMP_GRANULARITIES: frozenset[AudioTimestampGranularities] = (
        frozenset()
    )

    #: Transcription prompt
    TRANSCRIPTION_PROMPT = "Transcribe the audio."

    #: Translation prompt
    TRANSLATION_PROMPT = "Translate into english."

    async def tts(
        self,
        text: str,  # noqa: ARG002
        voice: str,  # noqa: ARG002
        resp_format: AudioFileFormat,  # noqa: ARG002
        speed: float = 1.0,  # noqa: ARG002
        extra_params: JsonMapping | None = None,  # noqa: ARG002
    ) -> TTSResponse:
        """Generate audio from text.

        Args:
            text: Text to convert to speech.
            voice: Voice to use.
            resp_format: Audio format.
            speed: Speech speed multiplier.
            extra_params: Extra model parameters.

        Returns:
            TTS response with audio stream.

        Raises:
            ApiError: If TTS is not supported by this model.
        """
        msg = f"Text-to-speech is not supported by {self.model.id}"
        raise ApiError(msg)

    async def stt(
        self,
        audio_content: InputFile,  # noqa: ARG002
        response_format: AudioResponseFormat,  # noqa: ARG002
        language: str | None = None,  # noqa: ARG002
        timestamp_granularities: list[AudioTimestampGranularities] | None = None,  # noqa: ARG002
        prompt: str | None = None,  # noqa: ARG002
        temperature: float | None = None,  # noqa: ARG002
        extra_params: JsonMapping | None = None,  # noqa: ARG002
        *,
        logprobs: bool,  # noqa: ARG002
    ) -> str | TranscriptionCreateResponse | TranscriptionDiarized | Response:
        """Transcribe audio to text.

        Args:
            audio_content: Audio file to transcribe.
            response_format: Format for output (json, text, srt, vtt, verbose_json, diarized_json).
            language: Optional language code.
            timestamp_granularities: Optional timestamp granularities for verbose_json.
            prompt: Optional prompt for transcription.
            temperature: Optional temperature for transcription.
            extra_params: Extra model parameters.
            logprobs: If true, return log probabilities.

        Returns:
            Formatted transcription response (str | TranscriptionCreateResponse | TranscriptionDiarized | Response).

        Raises:
            ApiError: If transcription is not supported by this model.
        """
        msg = f"Audio transcription is not supported by {self.model.id}"
        raise ApiError(msg)

    async def stt_stream(
        self,
        audio_content: InputFile,  # noqa: ARG002
        response_format: AudioResponseFormat,  # noqa: ARG002
        language: str | None = None,  # noqa: ARG002
        prompt: str | None = None,  # noqa: ARG002
        temperature: float | None = None,  # noqa: ARG002
        extra_params: JsonMapping | None = None,  # noqa: ARG002
        *,
        logprobs: bool,  # noqa: ARG002
    ) -> AsyncGenerator[TranscriptionTextDeltaEvent | TranscriptionTextDoneEvent]:
        """Transcribe audio to text with streaming response.

        Args:
            audio_content: Audio file to transcribe.
            response_format: Format for output.
            language: Optional language code.
            prompt: Optional prompt for transcription.
            temperature: Optional temperature for transcription.
            extra_params: Extra model parameters.
            logprobs: If true, return log probabilities.

        Yields:
            TranscriptionTextDeltaEvent or TranscriptionTextDoneEvent objects.

        Raises:
            ApiError: If streaming transcription is not supported by this model.
        """
        msg = f"Streaming audio transcription is not supported by {self.model.id}"
        raise ApiError(msg)
        yield

    async def stt_translate(
        self,
        audio_content: InputFile,  # noqa: ARG002
        response_format: AudioResponseFormat,  # noqa: ARG002
        prompt: str | None,  # noqa: ARG002
        temperature: float | None = None,  # noqa: ARG002
        extra_params: JsonMapping | None = None,  # noqa: ARG002
    ) -> str | TranslationCreateResponse | Response:
        """Transcribe and translate audio to English.

        Args:
            audio_content: Audio file to transcribe and translate.
            response_format: Format for output (json, text, srt, vtt, verbose_json).
            prompt: Optional prompt for translation.
            temperature: Optional temperature for transcription.
            extra_params: Extra model parameters.

        Returns:
            Formatted translation response (str | TranslationCreateResponse | Response).

        Raises:
            ApiError: If translation is not supported by this model.
        """
        msg = f"Audio transcription and translation is not supported by {self.model.id}"
        raise ApiError(msg)

    @classmethod
    def get_supported_operations(cls) -> Capability:
        """Auto-detect supported audio operations from method override presence.

        Returns:
            Capability flags for operations this audio model implements.
        """
        ops = Capability(0)
        if cls.tts is not AudioModelBase.tts:
            ops |= Capability.TTS
        if cls.stt is not AudioModelBase.stt:
            ops |= Capability.STT
        if cls.stt_translate is not AudioModelBase.stt_translate:
            ops |= Capability.STT_TRANSLATE
        return ops

    @staticmethod
    async def _format_subtitle_response(
        response_format: AudioResponseFormat,
        subtitle_content: str,
        filename: str | None,
    ) -> Response:
        """Format subtitle response with proper content type and disposition headers.

        Creates a FastAPI Response object for subtitle format downloads (SRT/VTT)
        with appropriate MIME type and filename in Content-Disposition header.

        Args:
            response_format: The subtitle response format (SRT or VTT)
            subtitle_content: The subtitle content as a string
            filename: The original filename of the audio file

        Returns:
            FastAPI Response with subtitle content and appropriate headers
        """
        return Response(
            content=subtitle_content,
            media_type=(
                "application/x-subrip" if response_format == "srt" else "text/vtt"
            ),
            headers={
                "Content-Disposition": f'attachment; filename="{Path(filename or "audio").stem}.{response_format}"'
            },
        )

    @staticmethod
    def _validate_no_temperature(value: float | None) -> None:
        """Validate that temperature parameter is not provided.

        Args:
            value: The value to validate

        Raises:
            ApiError: If a parameter is provided.
        """
        if value:
            param = "temperature"
            raise UnsupportedParameterError(param)

    @staticmethod
    def _validate_no_prompt(value: str | None) -> None:
        """Validate that prompt parameter is not provided.

        Args:
            value: The value to validate

        Raises:
            ApiError: If a parameter is provided.
        """
        if value is not None:
            param = "prompt"
            raise UnsupportedParameterError(param)

    @staticmethod
    def _validate_no_language(value: str | None) -> None:
        """Validate that language parameter is not provided.

        Args:
            value: The value to validate

        Raises:
            ApiError: If a parameter is provided.
        """
        if value is not None:
            param = "language"
            raise UnsupportedParameterError(param)

    @staticmethod
    def _validate_no_logprobs(value: bool) -> None:  # noqa: FBT001
        """Validate that logprobs parameter is not provided.

        Args:
            value: The value to validate

        Raises:
            ApiError: If a parameter is provided.
        """
        if value:
            param = "include.logprobs"
            raise UnsupportedParameterError(param)

    @classmethod
    def _validate_response_formats(
        cls,
        value: AudioResponseFormat,
        timestamp_granularities: list[AudioTimestampGranularities] | None = None,
    ) -> None:
        """Validate that response_format parameter value.

        Args:
            value: The value to validate
            timestamp_granularities: Timestamp granularities

        Raises:
            ApiError: If a parameter is provided.
        """
        if value not in cls.SUPPORTED_RESPONSES_FORMATS:
            msg = f"Response format '{value}' is not supported by this model."
            raise ApiError(msg)

        if value == "verbose_json" and timestamp_granularities:
            for granularity in timestamp_granularities:
                if granularity not in cls.SUPPORTED_TIMESTAMP_GRANULARITIES:
                    msg = f"'verbose_json' with '{granularity}' timestamp granularity is not supported by this model."
                    raise ApiError(msg)

    @classmethod
    def _built_prompt(
        cls, prompt: str | None, language: str | None, *, translate: bool = False
    ) -> str:
        """Builds a comprehensive prompt string based on various optional parameters.

        This class method constructs a detailed prompt to be used for transcription
        and/or translation tasks. It combines static prompts with additional context
        such as the expected language, translation requirement, and any user-provided
        customization to produce a well-rounded string.

        Args:
            prompt: A custom prompt string provided by the user. If None,
                this section of the string will be omitted.
            language: Specifies the language of the audio by providing its
                language code. If None, no language information will be appended to the
                prompt.
            translate: If set to True, the method will include a translation
                directive in the prompt. Defaults to False.

        Returns:
            str: A concatenated string containing the full prompt with all the relevant
                context.
        """
        prompt_items = [cls.TRANSCRIPTION_PROMPT]
        if language:
            prompt_items.append(
                f"The audio is excepted to be {language_code_to_name(language)} language."
            )
        if translate:
            prompt_items.append(cls.TRANSLATION_PROMPT)
        if prompt:
            prompt_items.append(prompt)
        return "\n".join(prompt_items)


# Audio Model Registry
_AUDIO_MODEL_REGISTRY: list[
    tuple[str | Pattern[str], type[AudioModelBase[Any, Any]]]
] = []
_AUDIO_MODEL_CACHE: dict[str, AudioModelBase[Any, Any]] = {}


def get_audio_model(model_id: str) -> AudioModelBase[Any, Any]:
    """Resolve the audio model class matching the provided identifier.

    Args:
        model_id: The provider model identifier (e.g., "amazon.polly-standard", "amazon.transcribe").

    Returns:
        The audio model associated to the ``model_id``.

    Raises:
        LookupError: If no registered audio model matches ``model_id``.
    """
    return get_model(model_id, _AUDIO_MODEL_CACHE, _AUDIO_MODEL_REGISTRY, __name__)


load_model_plugins(
    class_type=AudioModelBase, package_name=__name__, registry=_AUDIO_MODEL_REGISTRY
)


async def synthesize_speech(
    text: str, voice: str = "alloy", resp_format: AudioFileFormat = "mp3"
) -> AsyncGenerator[bytes]:
    """Asynchronously synthesizes speech from text using the default TTS model.

    The function interacts with the default TTS model to convert the input text into audio
    in the specified voice and format. The result is provided as an asynchronous
    generator that yields chunks of audio data.

    Args:
        text: The text to be converted into speech.
        voice: The desired voice configuration for the speech synthesis.
        resp_format: The audio file format for the synthesized speech.

    Yields:
        bytes: Chunks of audio data representing the synthesized speech in the
        specified format.
    """
    return (
        await get_audio_model(SETTINGS.default_tts_model).tts(
            text=text, voice=voice, resp_format=resp_format
        )
    )["audio_stream"]
