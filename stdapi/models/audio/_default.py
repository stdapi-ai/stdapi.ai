"""Default speech-to-text model implementation using AWS Bedrock Converse API.

For Bedrock models accepting SPEECH input, the audio is sent as a Converse
audio content block followed by a text transcription prompt, and the model's
text output is returned as the transcript.
"""

from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from fastapi import Response

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import (
    BEDROCK_BODY_SIZE_LIMIT,
    MIME_TYPES_TO_AUDIO_TYPE,
    apply_guardrail_to_text,
)
from stdapi.media import encode_audio_stream
from stdapi.models import NON_CONVERSE_SPEECH_MODEL_PREFIXES
from stdapi.models.audio import AudioModelBase
from stdapi.types.openai_audio import (
    Transcription,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
    Translation,
    UsageTokens,
)
from stdapi.utils import b64_encoded_len

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Mapping

    from types_aiobotocore_bedrock_runtime.type_defs import (
        AudioBlockTypeDef,
        ContentBlockTypeDef,
        TokenUsageTypeDef,
    )

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef
    from stdapi.input_file import InputFile
    from stdapi.types import JsonMapping
    from stdapi.types.openai_audio import (
        AudioResponseFormat,
        AudioTimestampGranularities,
        TranscriptionCreateResponse,
        TranscriptionDiarized,
        TranslationCreateResponse,
    )

#: Audio formats accepted by the Bedrock Converse audio content block ("AudioFormat" enum)
CONVERSE_AUDIO_FORMATS: frozenset[str] = frozenset(
    {
        "aac",
        "flac",
        "m4a",
        "mka",
        "mkv",
        "mp3",
        "mp4",
        "mpeg",
        "mpga",
        "ogg",
        "opus",
        "pcm",
        "wav",
        "webm",
        "x-aac",
    }
)

#: Media types holding an audio track ffmpeg can extract and transcode.
_TRANSCODABLE_MEDIA_TYPES: frozenset[str] = frozenset({"audio", "video"})

#: ApiError status returned by the ffmpeg pipeline when the encode itself failed.
_ENCODE_FAILURE_STATUS = 500

#: Largest audio payload whose encoded form still fits in the request body.
_MAX_INLINE_AUDIO_BYTES = BEDROCK_BODY_SIZE_LIMIT // 4 * 3


async def _single_chunk_stream(data: bytes) -> AsyncGenerator[bytes]:
    """Yield *data* as a single chunk for the ffmpeg encoding pipeline.

    Args:
        data: Complete audio content.

    Yields:
        The audio content as one chunk.
    """
    yield data


class AudioModel(AudioModelBase[Any, Any]):
    """Default speech-to-text model using AWS Bedrock Converse API."""

    __slots__ = ()

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
        keywords: list[str] | None = None,
        languages: list[str] | None = None,
        *,
        logprobs: bool,  # noqa: ARG002
    ) -> str | TranscriptionCreateResponse | TranscriptionDiarized | Response:
        """Transcribe audio to text via the Bedrock Converse API.

        Args:
            audio_content: Audio file to transcribe.
            response_format: Format for output (json, text).
            language: Optional language code for the input audio (ISO-639-1 format).
            timestamp_granularities: Optional timestamp granularities (not supported).
            prompt: Optional prompt for transcription.
            temperature: Optional temperature for transcription.
            extra_params: Unused; not supported by this model.
            keywords: Optional literal terms folded into the transcription context.
            languages: Optional expected input language codes folded into the
                transcription context.
            logprobs: Accepted but ignored; Bedrock reports no log probabilities
                on the Converse API.

        Returns:
            Formatted transcription response with text and token usage.

        Raises:
            ApiError: When an unsupported format is requested or transcription fails.
        """
        self._validate_converse_supported()
        self._validate_response_formats(response_format, timestamp_granularities)

        response = await self.converse(
            await self._build_request(
                audio_content, prompt, temperature, language, keywords, languages
            )
        )
        content = await apply_guardrail_to_text(
            self._output_text(response), source="OUTPUT"
        )

        if response_format == "text":
            return Response(content=content, media_type="text/plain; charset=utf-8")

        return Transcription(
            text=content,
            # Bedrock reports no log probabilities on the Converse API.
            logprobs=None,
            # Converse reports no audio/text input split, so
            # input_token_details is left unset (issue #95).
            usage=self._usage_tokens(response.get("usage")),
        )

    async def stt_stream(
        self,
        audio_content: InputFile,
        response_format: AudioResponseFormat,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        extra_params: JsonMapping | None = None,  # noqa: ARG002
        keywords: list[str] | None = None,
        languages: list[str] | None = None,
        *,
        logprobs: bool,  # noqa: ARG002
    ) -> AsyncGenerator[TranscriptionTextDeltaEvent | TranscriptionTextDoneEvent]:
        """Transcribe audio to text with streaming response.

        Args:
            audio_content: Audio file to transcribe.
            response_format: Format for output; ``diarized_json`` is refused,
                the model reporting no speakers.
            language: Optional language code for the input audio (ISO-639-1 format).
            prompt: Optional prompt for transcription.
            temperature: Optional temperature for transcription.
            extra_params: Unused; not supported by this model.
            keywords: Optional literal terms folded into the transcription context.
            languages: Optional expected input language codes folded into the
                transcription context.
            logprobs: Accepted but ignored; Bedrock reports no log probabilities
                on the Converse API.

        Yields:
            TranscriptionTextDeltaEvent or TranscriptionTextDoneEvent objects.

        Raises:
            ApiError: When an unsupported format is requested or transcription fails.
        """
        self._validate_streamed_diarization(response_format)
        self._validate_converse_supported()
        full_text_parts: list[str] = []
        usage: TokenUsageTypeDef | None = None

        stream = (
            await self.converse_stream(
                await self._build_request(
                    audio_content, prompt, temperature, language, keywords, languages
                )
            )
        )["stream"]
        async for event in stream:
            if delta := event.get("contentBlockDelta", {}).get("delta", {}).get("text"):
                full_text_parts.append(delta)
                yield TranscriptionTextDeltaEvent(
                    delta=delta,
                    type="transcript.text.delta",
                    # Bedrock reports no log probabilities on the Converse API.
                    logprobs=None,
                )
            if metadata_usage := event.get("metadata", {}).get("usage"):
                usage = metadata_usage

        # Converse reports no audio/text input split, so input_token_details is
        # left unset (issue #95).
        yield TranscriptionTextDoneEvent(
            text="".join(full_text_parts),
            type="transcript.text.done",
            logprobs=None,
            usage=self._usage_tokens(usage),
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

        The translation directive is folded into the transcription prompt: the
        model transcribes and translates natively in a single Converse call.

        Args:
            audio_content: Audio file to transcribe and translate.
            response_format: Format for output (json, text).
            prompt: Optional prompt for translation.
            temperature: Optional temperature for transcription.
            extra_params: Unused; not supported by this model.

        Returns:
            Formatted translation response with translated text in English.

        Raises:
            ApiError: When an unsupported format is requested or transcription fails.
        """
        self._validate_converse_supported()
        self._validate_response_formats(response_format)
        response = await self.converse(
            await self._build_request(
                audio_content, prompt, temperature, translate=True
            )
        )
        content = await apply_guardrail_to_text(
            self._output_text(response), source="OUTPUT"
        )

        if response_format == "text":
            return Response(content=content, media_type="text/plain; charset=utf-8")
        return Translation(text=content)

    async def _build_request(
        self,
        audio_content: InputFile,
        prompt: str | None,
        temperature: float | None,
        language: str | None = None,
        keywords: list[str] | None = None,
        languages: list[str] | None = None,
        *,
        translate: bool = False,
    ) -> ConverseRequestBaseTypeDef:
        """Build the Converse request carrying the audio and the transcription prompt.

        Args:
            audio_content: The audio file to be read and included in the
                transcription request.
            prompt: Optional prompt for transcription.
            temperature: Optional temperature for transcription.
            language: Optional language code for the input audio (ISO-639-1 format).
            keywords: Optional literal terms that may appear in the audio,
                folded into the transcription context.
            languages: Optional expected input language codes (ISO-639-1
                format), folded into the transcription context.
            translate: Optional flag to enable translation to English.

        Returns:
            Bedrock Converse request payload with ``modelId`` set to an empty
            string placeholder; ``converse()``/``converse_stream()`` fill in
            the real region-specific value.

        Raises:
            ApiError: If the provided file is not in a supported audio format.
        """
        return {
            "modelId": "",  # placeholder — overwritten by converse()/converse_stream()
            "messages": [
                {
                    "role": "user",
                    # The audio block must precede the text prompt: speech
                    # models (e.g. Voxtral) ignore audio placed after the text.
                    "content": [
                        await self._audio_content_block(audio_content),
                        {
                            "text": self._built_prompt(
                                prompt,
                                language,
                                keywords,
                                languages,
                                translate=translate,
                            )
                        },
                    ],
                }
            ],
            "inferenceConfig": {"temperature": temperature or 0.0},
        }

    async def _audio_content_block(
        self, audio_content: InputFile
    ) -> ContentBlockTypeDef:
        """Build the Converse audio content block, transcoding when needed.

        Audio whose format is outside the Converse ``AudioFormat`` enum is
        normalized to FLAC (lossless, and encodable by minimal ffmpeg builds)
        through the bounded ffmpeg pipeline. Video containers take that path
        too: they are legitimate transcription inputs (the OpenAI API accepts
        mp4/mpeg/webm) and ffmpeg's default stream selection picks their audio
        track. Uploads that are neither audio nor video (images, PDFs, text,
        ...) are rejected with the accepted format list, and audio too large to
        embed inline is refused before the Converse call.

        Args:
            audio_content: The audio file to embed as inline bytes.

        Returns:
            Bedrock Converse audio content block with an inline bytes source.

        Raises:
            ApiError: If the file is not in a supported audio format, holds no
                decodable audio, or is too large to embed inline.
        """
        media_type, file_format = await audio_content.get_content_type_tuple()
        audio_format = MIME_TYPES_TO_AUDIO_TYPE.get(file_format, file_format)
        if audio_format in CONVERSE_AUDIO_FORMATS:
            # Checked twice: the reported size avoids reading a body that
            # cannot fit, and the read length is what actually travels, since a
            # chunked response or an upload without a length reports zero.
            self._check_inline_size(
                b64_encoded_len(await audio_content.get_size()),
                audio_format,
                transcoded=False,
            )
            data = await audio_content.to_bytes()
            self._check_inline_size(
                b64_encoded_len(len(data)), audio_format, transcoded=False
            )
            audio_block: AudioBlockTypeDef = {
                "format": audio_format,  # type: ignore[typeddict-item]
                "source": {"bytes": data},
            }
            return {"audio": audio_block}
        if media_type not in _TRANSCODABLE_MEDIA_TYPES:
            msg = (
                f"Unsupported audio format '{file_format}'. Accepted formats: "
                f"{', '.join(sorted(CONVERSE_AUDIO_FORMATS))}."
            )
            raise ApiError(msg)
        buf = bytearray()
        source = await audio_content.to_bytes()
        try:
            async with aclosing(
                encode_audio_stream(_single_chunk_stream(source), "flac")
            ) as encoded:
                async for chunk in encoded:
                    buf.extend(chunk)
                    if b64_encoded_len(len(buf)) > BEDROCK_BODY_SIZE_LIMIT:
                        # Closing the stream kills ffmpeg: a long audio track
                        # would otherwise buffer far past what Converse accepts.
                        break
        except ApiError as exception:
            # Only a failed encode names the upload: a stall (504) and a server
            # missing ffmpeg keep their own status and message.
            if exception.status != _ENCODE_FAILURE_STATUS:
                raise
            msg = (
                f"The uploaded '{file_format}' file could not be decoded as "
                "audio. Upload a file carrying a decodable audio track."
            )
            raise ApiError(msg) from exception
        self._check_inline_size(b64_encoded_len(len(buf)), file_format, transcoded=True)
        transcoded_block: AudioBlockTypeDef = {
            "format": "flac",
            "source": {"bytes": bytes(buf)},
        }
        return {"audio": transcoded_block}

    @staticmethod
    def _check_inline_size(
        encoded_size: int, file_format: str, *, transcoded: bool
    ) -> None:
        """Reject audio too large for the Converse inline bytes source.

        Args:
            encoded_size: Size the audio occupies once encoded into the request
                body, which is the dimension the limit applies to. The
                transcode stops as soon as it passes the limit, so it is a
                lower bound there.
            file_format: Format of the uploaded file, for the error message.
            transcoded: Whether *encoded_size* is the FLAC conversion of that
                upload.

        Raises:
            ApiError: If the audio exceeds the Bedrock inline body limit.
        """
        if encoded_size <= BEDROCK_BODY_SIZE_LIMIT:
            return
        origin = (
            f"the '{file_format}' file converted to FLAC exceeds"
            if transcoded
            else f"the '{file_format}' file exceeds"
        )
        msg = (
            f"The audio to transcribe is too large: {origin} the "
            f"{_MAX_INLINE_AUDIO_BYTES} bytes accepted inline. Upload a shorter "
            "file, or one already in a natively supported format: "
            f"{', '.join(sorted(CONVERSE_AUDIO_FORMATS))}."
        )
        raise ApiError(msg)

    def _validate_converse_supported(self) -> None:
        """Reject speech models that the Bedrock Converse API cannot serve.

        Raises:
            ApiError: If the model has no Converse API support.
        """
        if self._model_id.startswith(NON_CONVERSE_SPEECH_MODEL_PREFIXES):
            msg = f"Audio transcription is not supported by {self._model_id}"
            raise ApiError(msg)

    @staticmethod
    def _output_text(response: Mapping[str, Any]) -> str:
        """Extract the text output from a Converse response.

        Args:
            response: Bedrock Converse response.

        Returns:
            Concatenated text content blocks of the output message.
        """
        return "".join(
            block["text"]
            for block in response["output"]["message"]["content"]
            if "text" in block
        )

    @staticmethod
    def _usage_tokens(usage: TokenUsageTypeDef | None) -> UsageTokens | None:
        """Map Converse token usage to the OpenAI usage payload.

        Args:
            usage: Converse ``usage`` block, or None when not reported.

        Returns:
            OpenAI-format token usage, or None when unavailable.
        """
        if not usage:
            return None
        return UsageTokens(
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            total_tokens=usage.get("totalTokens", 0),
            type="tokens",
        )
