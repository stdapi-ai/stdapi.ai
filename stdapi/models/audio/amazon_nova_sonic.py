"""Amazon Nova Sonic transcription and translation.

This family answers on a bidirectional model stream instead of a
request/response API: the transcript of an upload is a by-product of a live
session in which the model listens, and the translation is the reply it gives
back. Nothing about that session is forgiving -- an audio format it does not
expect, a missing handshake block or a teardown sent one event too early
produce no error and no transcript, only a connection the service drops about a
minute later. Every constant and every ordering choice here is therefore load
bearing, and the session is bounded on three axes: per event, in total, and in
how much audio it will carry.
"""

from asyncio import CancelledError, create_task, get_running_loop
from asyncio import timeout_at as async_timeout_at
from contextlib import aclosing, suppress
from typing import TYPE_CHECKING, Any, Final
from uuid import uuid4

from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from fastapi import Response
from pydantic_core import from_json

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import apply_guardrail_to_text
from stdapi.aws_bidi import open_bidi_stream
from stdapi.media import encode_audio_stream
from stdapi.models import compute_candidate_regions, set_effective_region
from stdapi.models.audio import AudioModelBase
from stdapi.types.openai_audio import (
    Transcription,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
    Translation,
    UsageInputTokenDetails,
    UsageTokens,
)
from stdapi.usage import record_bedrock_usage
from stdapi.utils import b64encode, to_json_bytes

if TYPE_CHECKING:
    from asyncio import Timeout
    from collections.abc import AsyncGenerator, Callable, Coroutine

    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.aws_bidi import BidiSession
    from stdapi.input_file import InputFile
    from stdapi.types import JsonMapping
    from stdapi.types.openai_audio import (
        AudioResponseFormat,
        AudioTimestampGranularities,
        TranscriptionCreateResponse,
        TranscriptionDiarized,
        TranslationCreateResponse,
    )

#: Sample rate the session declares and the upload is conditioned to, in hertz.
SAMPLE_RATE: Final = 16000

#: Bytes of conditioned audio per second (mono, 16-bit).
BYTES_PER_SECOND: Final = SAMPLE_RATE * 2

#: Conditioned audio carried by one input event (~32 ms).
FRAME_BYTES: Final = BYTES_PER_SECOND * 32 // 1000

#: Silence appended to the upload; without it the closing utterance is never transcribed.
TRAILING_SILENCE: Final = bytes(BYTES_PER_SECOND * 2)

#: Longest audio this family accepts, in seconds.
MAX_AUDIO_SECONDS: Final = 600

#: Conditioned audio bytes that ceiling allows.
_MAX_AUDIO_BYTES: Final = MAX_AUDIO_SECONDS * BYTES_PER_SECOND

#: Seconds the session may go without an event before it is abandoned.
_EVENT_TIMEOUT: Final = 30.0

#: Seconds a whole session may last, under the service's own connection limit.
_SESSION_TIMEOUT: Final = 420.0

#: Model whose transcripts carry the timestamps and subtitles this one cannot.
_TIMESTAMPED_MODEL: Final = "amazon.transcribe"

#: ApiError status the ffmpeg pipeline uses when the encode itself failed.
_ENCODE_FAILURE_STATUS: Final = 500

#: Generation stage of a text block that restates speech already emitted.
_FINAL_STAGE: Final = "FINAL"

#: Upload formats named in the rejection message; anything decodable is accepted.
_EXAMPLE_FORMATS: Final = "flac, m4a, mp3, mp4, mpeg, ogg, wav, webm"


class _Transcript:
    """What one session produced: its text, and what it billed."""

    __slots__ = (
        "assistant",
        "input_speech_tokens",
        "input_text_tokens",
        "output_speech_tokens",
        "output_text_tokens",
        "total_tokens",
        "user",
    )

    def __init__(self) -> None:
        """Start with no text and nothing metered."""
        self.user: list[str] = []
        self.assistant: list[str] = []
        self.input_speech_tokens = 0
        self.input_text_tokens = 0
        self.output_speech_tokens = 0
        self.output_text_tokens = 0
        self.total_tokens = 0

    def usage_tokens(self) -> UsageTokens:
        """Return the session's metering in the OpenAI usage shape.

        Returns:
            Input and output token counts, with the audio/text split the
            protocol reports.
        """
        input_tokens = self.input_speech_tokens + self.input_text_tokens
        output_tokens = self.output_speech_tokens + self.output_text_tokens
        return UsageTokens(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=self.total_tokens or (input_tokens + output_tokens),
            type="tokens",
            input_token_details=UsageInputTokenDetails(
                audio_tokens=self.input_speech_tokens,
                text_tokens=self.input_text_tokens,
            ),
        )


class AudioModel(AudioModelBase[Any, Any]):
    """Amazon Nova Sonic speech-to-text and speech-to-English-text."""

    __slots__ = ()

    # A string, not a pattern: the catalog ranks every pattern below every string
    # prefix, so a pattern here would lose this model's routes to the chat family.
    MATCHER = "amazon.nova-2-sonic"

    SUPPORTED_RESPONSES_FORMATS = frozenset({"json", "text"})

    TRANSCRIPTION_PROMPT = "Listen to the user's speech."

    TRANSLATION_PROMPT = (
        "Reply with the English translation of what the user said, and nothing else."
    )

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
        """Transcribe audio in the language it was spoken in.

        Args:
            audio_content: Audio file to transcribe.
            response_format: Format for output (json, text).
            language: Optional language code for the input audio (ISO-639-1 format).
            timestamp_granularities: Rejected; this model reports no timestamps.
            prompt: Optional prompt folded into the session's instructions.
            temperature: Optional sampling temperature.
            extra_params: Unused; not supported by this model.
            keywords: Optional literal terms folded into the session's instructions.
            languages: Optional expected input language codes folded into the
                session's instructions.
            logprobs: Accepted but ignored; no log probabilities are reported.

        Returns:
            The transcript, as plain text or as a transcription object.

        Raises:
            ApiError: When an unsupported format is requested or the session fails.
        """
        self._validate_response_formats(response_format, timestamp_granularities)
        transcript = await self._run_session(
            audio_content,
            self._built_prompt(prompt, language, keywords, languages),
            temperature,
            translate=False,
        )
        text = await apply_guardrail_to_text(" ".join(transcript.user), source="OUTPUT")
        if response_format == "text":
            return Response(content=text, media_type="text/plain; charset=utf-8")
        return Transcription(
            text=text,
            # This model reports no log probabilities.
            logprobs=None,
            usage=transcript.usage_tokens(),
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
        """Transcribe audio, emitting each utterance as it is recognized.

        Args:
            audio_content: Audio file to transcribe.
            response_format: Format for output (json, text).
            language: Optional language code for the input audio (ISO-639-1 format).
            prompt: Optional prompt folded into the session's instructions.
            temperature: Optional sampling temperature.
            extra_params: Unused; not supported by this model.
            keywords: Optional literal terms folded into the session's instructions.
            languages: Optional expected input language codes folded into the
                session's instructions.
            logprobs: Accepted but ignored; no log probabilities are reported.

        Yields:
            One delta per recognized utterance, then the terminal done event.

        Raises:
            ApiError: When an unsupported format is requested or the session fails.
        """
        self._validate_response_formats(response_format)
        transcript = _Transcript()
        parts: list[str] = []
        async with aclosing(
            self._session_events(
                audio_content,
                self._built_prompt(prompt, language, keywords, languages),
                temperature,
                transcript,
                translate=False,
            )
        ) as utterances:
            async for utterance in utterances:
                # The separator travels with the delta, so concatenating the
                # deltas rebuilds exactly the non-streamed transcript.
                delta = utterance if not parts else f" {utterance}"
                parts.append(delta)
                yield TranscriptionTextDeltaEvent(
                    delta=delta, type="transcript.text.delta", logprobs=None
                )
        yield TranscriptionTextDoneEvent(
            text="".join(parts),
            type="transcript.text.done",
            logprobs=None,
            usage=transcript.usage_tokens(),
        )

    async def stt_translate(
        self,
        audio_content: InputFile,
        response_format: AudioResponseFormat,
        prompt: str | None,
        temperature: float | None = None,
        extra_params: JsonMapping | None = None,  # noqa: ARG002
    ) -> str | TranslationCreateResponse | Response:
        """Translate spoken audio into English text.

        Args:
            audio_content: Audio file to translate.
            response_format: Format for output (json, text).
            prompt: Optional prompt folded into the session's instructions.
            temperature: Optional sampling temperature.
            extra_params: Unused; not supported by this model.

        Returns:
            The English translation, as plain text or as a translation object.

        Raises:
            ApiError: When an unsupported format is requested or the session fails.
        """
        self._validate_response_formats(response_format)
        transcript = await self._run_session(
            audio_content,
            self._built_prompt(prompt, None, translate=True),
            temperature,
            translate=True,
        )
        text = await apply_guardrail_to_text(
            " ".join(transcript.assistant), source="OUTPUT"
        )
        if response_format == "text":
            return Response(content=text, media_type="text/plain; charset=utf-8")
        return Translation(text=text)

    @classmethod
    def _validate_response_formats(
        cls,
        value: AudioResponseFormat,
        timestamp_granularities: list[AudioTimestampGranularities] | None = None,
    ) -> None:
        """Reject the formats and granularities that would need timestamps.

        Args:
            value: The requested response format.
            timestamp_granularities: The requested timestamp granularities.

        Raises:
            ApiError: The format or the granularities cannot be served.
        """
        if value not in cls.SUPPORTED_RESPONSES_FORMATS:
            served = "', '".join(sorted(cls.SUPPORTED_RESPONSES_FORMATS))
            msg = (
                f"Response format '{value}' is not available with this model, which "
                "returns no timestamps and does not report a detected language. "
                f"Request '{served}', or use `{_TIMESTAMPED_MODEL}` for timestamps "
                "and subtitles."
            )
            raise ApiError(msg)
        if timestamp_granularities:
            msg = (
                "Timestamps are not available with this model. Request the "
                f"transcript without `timestamp_granularities`, or use "
                f"`{_TIMESTAMPED_MODEL}`, which timestamps every word and segment."
            )
            raise ApiError(msg)

    async def _run_session(
        self,
        audio_content: InputFile,
        instructions: str,
        temperature: float | None,
        *,
        translate: bool,
    ) -> _Transcript:
        """Run one session to completion and return everything it produced.

        Args:
            audio_content: Audio file to send.
            instructions: System instructions opening the session.
            temperature: Optional sampling temperature.
            translate: Whether to read the model's reply instead of its
                transcription of the upload.

        Returns:
            The session's text and metering.

        Raises:
            ApiError: The session could not be opened or completed.
        """
        transcript = _Transcript()
        async with aclosing(
            self._session_events(
                audio_content,
                instructions,
                temperature,
                transcript,
                translate=translate,
            )
        ) as events:
            async for _ in events:
                pass
        return transcript

    async def _session_events(
        self,
        audio_content: InputFile,
        instructions: str,
        temperature: float | None,
        transcript: _Transcript,
        *,
        translate: bool,
    ) -> AsyncGenerator[str]:
        """Drive one session, yielding each utterance the model recognizes.

        The audio is conditioned and size-checked first, so an upload this
        family cannot carry is refused before anything is opened. Metering is
        written from the last frame seen even when the caller walks away
        mid-session.

        Args:
            audio_content: Audio file to send.
            instructions: System instructions opening the session.
            temperature: Optional sampling temperature.
            transcript: Accumulator receiving the text and the metering.
            translate: Whether to read the model's reply instead of its
                transcription of the upload.

        Yields:
            Each recognized utterance, in order.

        Raises:
            ApiError: The session could not be opened or completed.
        """
        audio = await self._conditioned_audio(audio_content)
        regions = await compute_candidate_regions(self._model_id)
        names = _SessionNames()
        region: RegionName | None = None
        try:
            async with open_bidi_stream(
                "bedrock-runtime",
                regions,
                self._open_stream,
                prime=_priming(names, instructions, temperature),
            ) as session:
                region = session.region
                set_effective_region(self._model_id, region)
                sender = create_task(_send_audio(session, names, audio))
                del audio
                try:
                    async for utterance in self._read_session(
                        session, transcript, translate=translate
                    ):
                        yield utterance
                    await _end_session(session, names)
                finally:
                    sender.cancel()
                    with suppress(CancelledError, Exception):
                        await sender
        finally:
            _record_usage(self._model_id, transcript, region)

    async def _open_stream(self, client: Any, _region: RegionName) -> Any:  # noqa: ANN401
        """Open the bidirectional stream for this model in one region.

        Args:
            client: The region's bidirectional client.
            _region: The region it serves (unused: the model is region-agnostic).

        Returns:
            The SDK's duplex event stream.
        """
        return await client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self._model_id)
        )

    async def _read_session(
        self,
        session: BidiSession[Any, Any],
        transcript: _Transcript,
        *,
        translate: bool,
    ) -> AsyncGenerator[str]:
        """Consume the session's events until its text is complete.

        Transcription stops at the model's first spoken turn, which is both the
        point at which the transcript is whole and the last point before any
        speech is generated. Translation is that speech, so it runs to the end
        of the turn instead.

        Args:
            session: The open session.
            transcript: Accumulator receiving the text and the metering.
            translate: Whether the model's reply is what is wanted.

        Yields:
            Each recognized utterance, in order.

        Raises:
            ApiError: The session stopped answering.
        """
        reader = _EventReader(transcript, translate=translate)
        try:
            deadline = get_running_loop().time() + _SESSION_TIMEOUT
            async with async_timeout_at(deadline) as limit:
                # Bounds the first event too: a session given audio it cannot
                # read never answers at all.
                _extend(limit, deadline)
                async for event in session:
                    _extend(limit, deadline)
                    if (payload := _event_body(event)) is None:
                        continue
                    utterance, complete = reader.read(*payload)
                    if utterance is not None:
                        yield utterance
                    if complete:
                        return
        except TimeoutError as exception:
            msg = "The audio could not be processed in time. Retry with a shorter recording."
            raise ApiError(msg, status=504) from exception

    async def _conditioned_audio(self, audio_content: InputFile) -> bytearray:
        """Decode an upload into the mono 16 kHz samples the session accepts.

        Args:
            audio_content: The uploaded file.

        Returns:
            Little-endian 16-bit mono samples, unconverted: copying them into
            ``bytes`` would briefly double a buffer that reaches tens of megabytes.

        Raises:
            ApiError: The upload holds no audio, is longer than this model
                accepts, or could not be decoded.
        """
        media_type, file_format = await audio_content.get_content_type_tuple()
        if media_type not in {"audio", "video"}:
            msg = (
                f"Unsupported audio format '{file_format}'. Upload an audio or "
                f"video file, for example: {_EXAMPLE_FORMATS}."
            )
            raise ApiError(msg)
        source = await audio_content.to_bytes()
        buffer = bytearray()
        try:
            async with aclosing(
                encode_audio_stream(
                    _one_chunk(source),
                    "pcm",
                    output_sample_rate=SAMPLE_RATE,
                    output_channels=1,
                )
            ) as encoded:
                del source
                async for chunk in encoded:
                    buffer.extend(chunk)
                    if len(buffer) > _MAX_AUDIO_BYTES:
                        # Closing the stream kills ffmpeg: a long recording
                        # would otherwise be decoded in full before being refused.
                        break
        except ApiError as exception:
            if exception.status != _ENCODE_FAILURE_STATUS:
                raise
            msg = (
                f"The uploaded '{file_format}' file could not be decoded as "
                "audio. Upload a file carrying a decodable audio track."
            )
            raise ApiError(msg) from exception
        if len(buffer) > _MAX_AUDIO_BYTES:
            msg = (
                f"The audio is too long: this model accepts at most "
                f"{MAX_AUDIO_SECONDS // 60} minutes per request. Upload a shorter "
                f"recording, or use `{_TIMESTAMPED_MODEL}`, which has no such limit."
            )
            raise ApiError(msg)
        return buffer


class _SessionNames:
    """The identifiers one session's events refer to each other by."""

    __slots__ = ("audio", "instructions", "prompt")

    def __init__(self) -> None:
        """Mint one identifier per content block of the session."""
        self.prompt = uuid4().hex
        self.instructions = uuid4().hex
        self.audio = uuid4().hex


async def _one_chunk(data: bytes) -> AsyncGenerator[bytes]:
    """Yield *data* as the single chunk the ffmpeg pipeline reads.

    Args:
        data: Complete upload content.

    Yields:
        The upload as one chunk.
    """
    yield data


def _extend(limit: Timeout, deadline: float) -> None:
    """Give the session one more event's worth of patience, up to its deadline.

    Args:
        limit: The session's timeout object.
        deadline: The absolute moment the session must be over.
    """
    limit.reschedule(min(deadline, get_running_loop().time() + _EVENT_TIMEOUT))


def _priming(
    names: _SessionNames, instructions: str, temperature: float | None
) -> Callable[[BidiSession[Any, Any]], Coroutine[Any, Any, None]]:
    """Build the handshake the service needs before it answers at all.

    Args:
        names: The session's content identifiers.
        instructions: System instructions opening the session.
        temperature: Optional sampling temperature.

    Returns:
        A coroutine function sending the handshake on a session.
    """

    async def prime(session: BidiSession[Any, Any]) -> None:
        """Send the handshake events, in the only order the service accepts."""
        inference: dict[str, Any] = {}
        if temperature is not None:
            inference["temperature"] = temperature
        await _send(session, {"sessionStart": {"inferenceConfiguration": inference}})
        await _send(
            session,
            {
                "promptStart": {
                    "promptName": names.prompt,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    # Not optional in practice: without it the session stays
                    # silent, even though nothing here reads the audio back.
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 24000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "matthew",
                    },
                }
            },
        )
        # The session's first content block must be the system one.
        await _send(
            session,
            {
                "contentStart": {
                    "promptName": names.prompt,
                    "contentName": names.instructions,
                    "type": "TEXT",
                    "interactive": True,
                    "role": "SYSTEM",
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            },
        )
        await _send(
            session,
            {
                "textInput": {
                    "promptName": names.prompt,
                    "contentName": names.instructions,
                    "content": instructions,
                }
            },
        )
        await _send(
            session,
            {
                "contentEnd": {
                    "promptName": names.prompt,
                    "contentName": names.instructions,
                }
            },
        )
        await _send(
            session,
            {
                "contentStart": {
                    "promptName": names.prompt,
                    "contentName": names.audio,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                    },
                }
            },
        )

    return prime


async def _send(session: BidiSession[Any, Any], event: dict[str, Any]) -> None:
    """Send one protocol event on a session.

    Args:
        session: The open session.
        event: The event body, without its ``event`` wrapper.
    """
    await session.send(
        InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=to_json_bytes({"event": event}))
        )
    )


async def _send_audio(
    session: BidiSession[Any, Any], names: _SessionNames, audio: bytes | bytearray
) -> None:
    """Stream the conditioned audio into the session and close its content block.

    The trailing silence is not padding: without it the model never treats the
    last utterance as finished, and never transcribes it.

    Args:
        session: The open session.
        names: The session's content identifiers.
        audio: Little-endian 16-bit mono samples at :data:`SAMPLE_RATE`.
    """
    view = memoryview(audio)
    for start in range(0, len(view), FRAME_BYTES):
        await _send_audio_frame(session, names, view[start : start + FRAME_BYTES])
    for start in range(0, len(TRAILING_SILENCE), FRAME_BYTES):
        await _send_audio_frame(
            session, names, memoryview(TRAILING_SILENCE)[start : start + FRAME_BYTES]
        )
    await _send(
        session,
        {"contentEnd": {"promptName": names.prompt, "contentName": names.audio}},
    )


async def _send_audio_frame(
    session: BidiSession[Any, Any], names: _SessionNames, frame: memoryview
) -> None:
    """Send one audio frame.

    Args:
        session: The open session.
        names: The session's content identifiers.
        frame: The samples this frame carries.
    """
    await _send(
        session,
        {
            "audioInput": {
                "promptName": names.prompt,
                "contentName": names.audio,
                "content": await b64encode(frame),
            }
        },
    )


async def _end_session(session: BidiSession[Any, Any], names: _SessionNames) -> None:
    """Close the prompt and the session, best effort.

    Sent only once the text is complete: sending either one earlier discards
    everything the session produced.

    Args:
        session: The open session.
        names: The session's content identifiers.
    """
    with suppress(ApiError):
        await _send(session, {"promptEnd": {"promptName": names.prompt}})
        await _send(session, {"sessionEnd": {}})


def _event_body(event: Any) -> tuple[str, dict[str, Any]] | None:  # noqa: ANN401
    """Decode one output event into its name and body.

    Args:
        event: The event the stream yielded.

    Returns:
        The event name and body, or None for anything this family does not read.
    """
    payload = getattr(getattr(event, "value", None), "bytes_", None)
    if not payload:
        return None
    decoded = from_json(payload)
    body = decoded.get("event") if isinstance(decoded, dict) else None
    if not isinstance(body, dict) or not body:
        return None
    name = next(iter(body))
    inner = body[name]
    return (name, inner) if isinstance(inner, dict) else None


class _EventReader:
    """Turns one session's output events into the text a route returns.

    Content blocks announce their speaker and generation stage in a
    ``contentStart``, and the text arrives in later events that only reference
    the block, so both have to be remembered to attribute the text at all.
    """

    __slots__ = ("_roles", "_stages", "_transcript", "_translate")

    def __init__(self, transcript: _Transcript, *, translate: bool) -> None:
        """Read into *transcript*.

        Args:
            transcript: Accumulator receiving the text and the metering.
            translate: Whether the model's reply is what is wanted, rather than
                its transcription of the upload.
        """
        self._transcript = transcript
        self._translate = translate
        self._roles: dict[str, str] = {}
        self._stages: dict[str, str] = {}

    def read(self, name: str, body: dict[str, Any]) -> tuple[str | None, bool]:
        """Read one event.

        Args:
            name: The event name.
            body: The event body.

        Returns:
            The utterance to emit, if any, and whether the text is now complete.
        """
        match name:
            case "contentStart":
                self._note_block(body)
                # The model's own turn starts only once it has heard everything,
                # and stopping here is what keeps it from generating speech.
                return None, not self._translate and body.get("role") == "ASSISTANT"
            case "textOutput":
                return self._read_text(body), False
            case "usageEvent":
                _record_metering(body, self._transcript)
            case "contentEnd" if self._translate:
                return None, body.get("stopReason") == "END_TURN"
        return None, False

    def _note_block(self, body: dict[str, Any]) -> None:
        """Remember a content block's speaker and generation stage.

        Args:
            body: The ``contentStart`` body.
        """
        content_id = body.get("contentId")
        if not content_id:
            return
        if role := body.get("role"):
            self._roles[content_id] = role
        fields = body.get("additionalModelFields")
        if isinstance(fields, str) and fields:
            with suppress(ValueError):
                decoded = from_json(fields.encode())
                if isinstance(decoded, dict):
                    self._stages[content_id] = decoded.get("generationStage", "")

    def _read_text(self, body: dict[str, Any]) -> str | None:
        """Attribute one text event to a speaker and keep it.

        Args:
            body: The ``textOutput`` body.

        Returns:
            The utterance to emit, or None when it is not the caller's.
        """
        text = (body.get("content") or "").strip()
        if not text:
            return None
        content_id = body.get("contentId", "")
        role = self._roles.get(content_id, body.get("role", ""))
        if role == "USER":
            self._transcript.user.append(text)
            return text
        if (
            self._translate
            and role == "ASSISTANT"
            # A FINAL block restates speech the preview already carried; keeping
            # both duplicates the whole reply.
            and self._stages.get(content_id) != _FINAL_STAGE
        ):
            self._transcript.assistant.append(text)
        return None


def _record_metering(body: dict[str, Any], transcript: _Transcript) -> None:
    """Take the session's running totals from one metering frame.

    The frames carry cumulative totals as well as deltas; summing the deltas
    would double-count every frame that repeats a total.

    Args:
        body: The ``usageEvent`` body.
        transcript: Accumulator receiving the metering.
    """
    total = body.get("details", {}).get("total", {})
    inputs = total.get("input", {})
    outputs = total.get("output", {})
    transcript.input_speech_tokens = inputs.get("speechTokens", 0)
    transcript.input_text_tokens = inputs.get("textTokens", 0)
    transcript.output_speech_tokens = outputs.get("speechTokens", 0)
    transcript.output_text_tokens = outputs.get("textTokens", 0)
    transcript.total_tokens = body.get("totalTokens", 0)


def _record_usage(
    model_id: str, transcript: _Transcript, region: RegionName | None
) -> None:
    """Record what the session billed, whether or not it ran to completion.

    Args:
        model_id: The Bedrock model identifier.
        transcript: The session's metering.
        region: The region that served it, or None if none ever did.
    """
    if region is None:
        return
    input_speech = transcript.input_speech_tokens
    output_speech = transcript.output_speech_tokens
    record_bedrock_usage(
        model_id,
        region=region,
        input_tokens=input_speech + transcript.input_text_tokens,
        output_tokens=output_speech + transcript.output_text_tokens,
        total_tokens=transcript.total_tokens,
        # Speech tokens are priced an order of magnitude above text ones.
        input_tokens_by_spec={"speech": input_speech} if input_speech else None,
        output_tokens_by_spec={"speech": output_speech} if output_speech else None,
    )
