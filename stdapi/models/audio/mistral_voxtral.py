"""Mistral Voxtral model implementation."""

from typing import TYPE_CHECKING, TypedDict

from stdapi.aws_bedrock import MIME_TYPES_TO_AUDIO_TYPE
from stdapi.models.audio import AudioModelBase
from stdapi.types.openai_audio import (
    Transcription,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
    Translation,
    TranslationCreateResponse,
    UsageInputTokenDetails,
    UsageTokens,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fastapi import Response

    from stdapi.input_file import InputFile
    from stdapi.types import JsonMapping
    from stdapi.types.openai_audio import (
        AudioResponseFormat,
        AudioTimestampGranularities,
        TranscriptionCreateResponse,
        TranscriptionDiarized,
    )


class _AudioContent(TypedDict):
    """Audio content with base64-encoded data and format."""

    type: str
    input_audio: dict[str, str]


class _TextContent(TypedDict):
    """Text content with transcription prompt."""

    type: str
    text: str


class _Message(TypedDict):
    """Chat message containing audio and text content."""

    role: str
    content: list[_AudioContent | _TextContent]


class _Request(TypedDict):
    """Request payload for Mistral Voxtral transcription."""

    messages: list[_Message]
    temperature: float


class _PromptTokensDetails(TypedDict, total=False):
    """Detailed breakdown of prompt tokens."""

    audio_tokens: int
    cached_tokens: int


class _ResponseUsage(TypedDict):
    """Token usage information from response."""

    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    prompt_tokens_details: _PromptTokensDetails


class _ResponseMessage(TypedDict):
    """Message content from response."""

    content: str
    refusal: str | None
    role: str


class _ResponseChoice(TypedDict):
    """Individual response choice."""

    finish_reason: str
    index: int
    logprobs: None
    message: _ResponseMessage


class _Response(TypedDict):
    """Response from Mistral Voxtral API."""

    choices: list[_ResponseChoice]
    created: int
    id: str
    model: str
    object: str
    service_tier: str
    usage: _ResponseUsage


class _StreamDelta(TypedDict, total=False):
    """Delta content in streaming chunk."""

    content: str
    role: str


class _StreamChoice(TypedDict, total=False):
    """Individual streaming choice."""

    delta: _StreamDelta
    finish_reason: str | None
    index: int
    logprobs: None


class _BedrockInvocationMetrics(TypedDict):
    """AWS Bedrock invocation metrics."""

    inputTokenCount: int
    outputTokenCount: int
    invocationLatency: int
    firstByteLatency: int


class _StreamChunk(TypedDict, total=False):
    """Streaming chunk from Mistral Voxtral API."""

    choices: list[_StreamChoice]
    created: int
    id: str
    model: str
    object: str
    service_tier: str


class AudioModel(AudioModelBase[_Request, _Response]):
    """Mistral Voxtral audio model implementation."""

    MATCHER = "mistral.voxtral-"

    SUPPORTED_RESPONSES_FORMATS = frozenset({"json", "text"})

    async def stt(
        self,
        audio_content: InputFile,
        response_format: AudioResponseFormat,
        language: str | None = None,
        timestamp_granularities: list[AudioTimestampGranularities] | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        extra_params: JsonMapping | None = None,  # noqa: ARG002
        *,
        logprobs: bool,
    ) -> str | TranscriptionCreateResponse | TranscriptionDiarized | Response:
        """Transcribe audio to text.

        This method uses Mistral Voxtral to transcribe audio content by encoding
        it to base64 and sending it through the chat completions interface.

        Args:
            audio_content: Audio file to transcribe
            response_format: Format for output (json, text, verbose_json)
            language: Optional language code (not currently used)
            timestamp_granularities: Optional timestamp granularities (not currently supported)
            prompt: Optional prompt for transcription.
            temperature: Optional temperature for transcription.
            extra_params: Unused; not supported by this model.
            logprobs: If true, return log probabilities.

        Returns:
            Formatted transcription response with text and token usage

        Raises:
            ApiError: When unsupported format is requested
            ApiError: When transcription fails
        """
        self._validate_response_formats(response_format, timestamp_granularities)

        result = await self.invoke(
            await self._build_request(audio_content, prompt, temperature, language)
        )
        response = result.response
        choice = response["choices"][0]
        content = choice["message"]["content"]

        if response_format == "text":
            return content

        usage = response["usage"]
        return Transcription(
            text=content,
            logprobs=choice.get("logprobs") if logprobs else None,
            usage=UsageTokens(
                input_tokens=usage["prompt_tokens"],
                output_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
                input_token_details=UsageInputTokenDetails(
                    audio_tokens=usage["prompt_tokens_details"]["audio_tokens"],
                    text_tokens=usage["prompt_tokens_details"]["cached_tokens"],
                )
                if "prompt_tokens_details" in usage
                else None,
            ),
        )

    async def stt_stream(
        self,
        audio_content: InputFile,
        response_format: AudioResponseFormat,  # noqa: ARG002
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        extra_params: JsonMapping | None = None,  # noqa: ARG002
        *,
        logprobs: bool,
    ) -> AsyncGenerator[TranscriptionTextDeltaEvent | TranscriptionTextDoneEvent]:
        """Transcribe audio to text with streaming response.

        Args:
            audio_content: Audio file to transcribe
            response_format: Format for output (only "text" is supported for streaming)
            language: Optional language code
            prompt: Optional prompt for transcription.
            temperature: Optional temperature for transcription.
            extra_params: Unused; not supported by this model.
            logprobs: If true, return log probabilities.

        Yields:
            TranscriptionTextDeltaEvent or TranscriptionTextDoneEvent objects

        Raises:
            ApiError: When unsupported format is requested
        """
        full_text_parts: list[str] = []
        metrics: _BedrockInvocationMetrics | None = None

        chunk: _StreamChunk
        async for chunk in self.invoke_stream(  # type: ignore[assignment]
            await self._build_request(audio_content, prompt, temperature, language)
        ):
            choice = chunk["choices"][0]
            if "delta" in choice:
                delta = choice["delta"]
                if delta.get("content"):
                    content = delta["content"]
                    full_text_parts.append(content)
                    yield TranscriptionTextDeltaEvent(
                        delta=content,
                        type="transcript.text.delta",
                        logprobs=choice.get("logprobs") if logprobs else None,
                    )

            if (
                choice.get("finish_reason")
                and "amazon-bedrock-invocationMetrics" in chunk
            ):
                metrics = chunk["amazon-bedrock-invocationMetrics"]  # type: ignore[typeddict-item]

        yield TranscriptionTextDoneEvent(
            text="".join(full_text_parts),
            type="transcript.text.done",
            logprobs=None,
            usage=UsageTokens(
                input_tokens=metrics["inputTokenCount"],
                output_tokens=metrics["outputTokenCount"],
                total_tokens=metrics["inputTokenCount"] + metrics["outputTokenCount"],
                type="tokens",
                input_token_details=UsageInputTokenDetails(
                    audio_tokens=metrics["inputTokenCount"], text_tokens=0
                ),
            )
            if metrics
            else None,
        )

    async def stt_translate(
        self,
        audio_content: InputFile,
        response_format: AudioResponseFormat,
        prompt: str | None,
        temperature: float | None = None,
        extra_params: JsonMapping | None = None,  # noqa: ARG002
    ) -> str | TranslationCreateResponse | Response:
        """Transcribe and translate audio to English.

        This method performs transcription using AWS Transcribe, detects the source
        language, and translates the transcribed text to English using AWS Translate.

        Args:
            audio_content: Audio file to transcribe and translate
            response_format: Format for output (json, text, srt, vtt, verbose_json)
            prompt: Optional prompt for translation.
            temperature: Optional temperature for transcription.
            extra_params: Unused; not supported by this model (Voxtral translates natively).

        Returns:
            Formatted translation response with translated text in English

        Raises:
            ApiError: When transcription or translation fails
        """
        self._validate_response_formats(response_format)
        result = await self.invoke(
            await self._build_request(
                audio_content, prompt, temperature, translate=True
            )
        )
        content = result.response["choices"][0]["message"]["content"]

        if response_format == "text":
            return content
        return Translation(text=content)

    async def _build_request(
        self,
        audio_content: InputFile,
        prompt: str | None,
        temperature: float | None,
        language: str | None = None,
        *,
        translate: bool = False,
    ) -> _Request:
        """Build a transcription request object asynchronously using given audio content.

        This method constructs the request body for audio transcription by reading
        the provided audio file, determining its MIME type, validating that it is
        an audio format, and encoding it in base64. It includes instructions for
        transcription within the request payload.

        Args:
            audio_content: The audio file to be read and included in
                the transcription request. The file format is validated to ensure
                it's an audio type.
            prompt: Optional prompt for transcription.
            temperature: Optional temperature for transcription.
            language: Optional language code for the input audio (ISO-639-1 format)
            translate: Optional flag to enable translation to English.

        Returns:
            A structured request object ready to be sent for audio
                transcription. This object contains the encoded audio, its format,
                and related instructions.

        Raises:
            ApiError: If the provided file is not in a supported audio format.
        """
        file_format = (await audio_content.get_content_type_tuple())[1]
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": await audio_content.to_base64(),
                                "format": MIME_TYPES_TO_AUDIO_TYPE.get(
                                    file_format, file_format
                                ),
                            },
                        },
                        {
                            "type": "text",
                            "text": self._built_prompt(
                                prompt, language, translate=translate
                            ),
                        },
                    ],
                }
            ],
            "temperature": temperature or 0.0,
        }
