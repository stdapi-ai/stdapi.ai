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
from typing import TYPE_CHECKING, Any, TypedDict

from fastapi import Response

from stdapi.config import SETTINGS
from stdapi.models import ModelBase, get_model, load_model_plugins
from stdapi.openai_exceptions import OpenaiError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from re import Pattern

    from fastapi import BackgroundTasks, UploadFile
    from pydantic import JsonValue

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
        characters_count: Number of characters in input text.
    """

    audio_stream: AsyncGenerator[bytes]
    characters_count: int


class AudioModelBase[RequestT, ResponseT](ModelBase[RequestT, ResponseT]):
    """Base class for provider-specific audio models supporting TTS and/or transcription."""

    async def tts(
        self,
        text: str,  # noqa: ARG002
        voice: str,  # noqa: ARG002
        resp_format: AudioFileFormat,  # noqa: ARG002
        speed: float = 1.0,  # noqa: ARG002
        extra_params: dict[str, JsonValue] | None = None,  # noqa: ARG002
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
            OpenaiError: If TTS is not supported by this model.
        """
        msg = f"Text-to-speech is not supported by {self.model.id}"
        raise OpenaiError(msg)

    async def stt(
        self,
        audio_content: UploadFile,  # noqa: ARG002
        background_tasks: BackgroundTasks,  # noqa: ARG002
        response_format: AudioResponseFormat,  # noqa: ARG002
        language: str | None = None,  # noqa: ARG002
        timestamp_granularities: list[AudioTimestampGranularities] | None = None,  # noqa: ARG002
    ) -> str | TranscriptionCreateResponse | TranscriptionDiarized | Response:
        """Transcribe audio to text.

        Args:
            audio_content: Audio file to transcribe.
            background_tasks: FastAPI background tasks for cleanup.
            response_format: Format for output (json, text, srt, vtt, verbose_json, diarized_json).
            language: Optional language code.
            timestamp_granularities: Optional timestamp granularities for verbose_json.

        Returns:
            Formatted transcription response (str | TranscriptionCreateResponse | TranscriptionDiarized | Response).

        Raises:
            OpenaiError: If transcription is not supported by this model.
        """
        msg = f"Audio transcription is not supported by {self.model.id}"
        raise OpenaiError(msg)

    async def stt_stream(
        self,
        audio_content: UploadFile,  # noqa: ARG002
        background_tasks: BackgroundTasks,  # noqa: ARG002
        response_format: AudioResponseFormat,  # noqa: ARG002
        language: str | None = None,  # noqa: ARG002
    ) -> AsyncGenerator[TranscriptionTextDeltaEvent | TranscriptionTextDoneEvent]:
        """Transcribe audio to text with streaming response.

        Args:
            audio_content: Audio file to transcribe.
            background_tasks: FastAPI background tasks for cleanup.
            response_format: Format for output.
            language: Optional language code.

        Yields:
            TranscriptionTextDeltaEvent or TranscriptionTextDoneEvent objects.

        Raises:
            OpenaiError: If streaming transcription is not supported by this model.
        """
        msg = f"Streaming audio transcription is not supported by {self.model.id}"
        raise OpenaiError(msg)
        yield

    async def stt_translate(
        self,
        audio_content: UploadFile,  # noqa: ARG002
        background_tasks: BackgroundTasks,  # noqa: ARG002
        response_format: AudioResponseFormat,  # noqa: ARG002
    ) -> str | TranslationCreateResponse | Response:
        """Transcribe and translate audio to English.

        Args:
            audio_content: Audio file to transcribe and translate.
            background_tasks: FastAPI background tasks for cleanup.
            response_format: Format for output (json, text, srt, vtt, verbose_json).

        Returns:
            Formatted translation response (str | TranslationCreateResponse | Response).

        Raises:
            OpenaiError: If translation is not supported by this model.
        """
        msg = f"Audio transcription and translation is not supported by {self.model.id}"
        raise OpenaiError(msg)

    @staticmethod
    def _format_subtitle_response(
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
            FastAPI Response with subtitle content and appropriate headers
        """
        content_type = (
            "application/x-subrip" if response_format == "srt" else "text/vtt"
        )
        original_filename = Path(file.filename or "audio").stem
        filename = f"{original_filename}.{response_format}"

        return Response(
            content=subtitle_content,
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )


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
    return get_model(model_id, _AUDIO_MODEL_CACHE, _AUDIO_MODEL_REGISTRY)


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
