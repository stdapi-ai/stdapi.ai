"""Amazon Nova Sonic live speech-to-speech conversations.

The backend is the same bidirectional model stream the transcription family
uses, driven as an open-ended conversation instead of one upload: the caller's
speech flows in continuously and the model speaks back over it. Three of its
rules shape everything here -- the first content block must be the system one,
the audio block that is open is what the model listens to, and a gap of about a
minute with nothing arriving ends the session with a validation error, so the
session sends silence rather than nothing.
"""

from asyncio import CancelledError, create_task, get_running_loop, sleep
from asyncio import timeout as async_timeout
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any, ClassVar, Final
from uuid import uuid4

from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from pydantic_core import from_json

from stdapi.aws_bidi import open_bidi_stream
from stdapi.models import compute_candidate_regions, set_effective_region
from stdapi.models.realtime import (
    InputTranscript,
    OutputAudio,
    OutputTranscript,
    RealtimeBackendSession,
    RealtimeModelBase,
    ResponseFinished,
    ResponseStarted,
    SpeechStarted,
    SpeechStopped,
    UsageReport,
)
from stdapi.utils import b64decode, b64encode, to_json_bytes

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Buffer,
        Callable,
        Coroutine,
    )

    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.aws_bidi import BidiSession
    from stdapi.models.realtime import BackendEvent

#: Sample rates the session's audio configurations accept, in hertz.
_SAMPLE_RATES: Final = frozenset({8000, 16000, 24000})

#: Milliseconds of conditioned audio carried by one input event.
_FRAME_MS: Final = 32

#: Seconds a whole session may last, under the service's own connection limit.
_MAX_SESSION_SECONDS: Final = 480.0

#: Seconds of silence appended to a turn the caller ended itself.
_TRAILING_SILENCE_SECONDS: Final = 1.0

#: Seconds of quiet after which silence is sent, under the service's ~55 s.
_KEEPALIVE_SECONDS: Final = 20.0

#: Seconds the goodbye gets before the session is abandoned.
_CLOSE_TIMEOUT: Final = 5.0

#: Stop reason of a content block the caller spoke over.
_INTERRUPTED: Final = "INTERRUPTED"

#: Stop reason of the content block that completes the model's answer.
_END_TURN: Final = "END_TURN"

#: Generation stage of a text block restating speech already reported.
_FINAL_STAGE: Final = "FINAL"

#: Voice each OpenAI voice name is served by; anything else is passed through.
_VOICES: Final[dict[str, str]] = {
    "alloy": "tiffany",
    "ash": "matthew",
    "ballad": "amy",
    "cedar": "stephen",
    "coral": "danielle",
    "echo": "matthew",
    "marin": "joanna",
    "sage": "ruth",
    "shimmer": "olivia",
    "verse": "stephen",
}


class _SessionNames:
    """The identifiers one session's events refer to each other by."""

    __slots__ = ("audio", "instructions", "prompt")

    def __init__(self) -> None:
        """Mint one identifier per content block of the session."""
        self.prompt = uuid4().hex
        self.instructions = uuid4().hex
        self.audio = uuid4().hex


class _NovaSonicSession(RealtimeBackendSession):
    """A live Nova Sonic conversation, driven event by event."""

    __slots__ = (
        "_audio_open",
        "_frame_bytes",
        "_last_sent",
        "_names",
        "_pending_text",
        "_roles",
        "_sample_rate",
        "_session",
        "_silence",
        "_speech_output",
        "_stages",
        "_trailing_silence",
        "region",
    )

    def __init__(
        self,
        session: BidiSession[Any, Any],
        names: _SessionNames,
        input_sample_rate: int,
        region: RegionName,
        *,
        speech_output: bool,
    ) -> None:
        """Wrap an opened stream whose handshake already ran.

        Args:
            session: The opened bidirectional stream.
            names: The session's content identifiers.
            input_sample_rate: Rate, in hertz, of the audio the caller sends.
            region: Region serving the session.
            speech_output: Whether the model's speech is reported at all.
        """
        self._session = session
        self._names = names
        self._sample_rate = input_sample_rate
        self._frame_bytes = input_sample_rate * 2 * _FRAME_MS // 1000
        self._silence = bytes(self._frame_bytes)
        self._trailing_silence = bytes(
            int(input_sample_rate * 2 * _TRAILING_SILENCE_SECONDS)
        )
        self._audio_open = False
        self._pending_text = False
        self._speech_output = speech_output
        self._last_sent = get_running_loop().time()
        self._roles: dict[str, str] = {}
        self._stages: dict[str, str] = {}
        self.region = region

    async def send_audio(self, audio: Buffer) -> None:
        """Send one chunk of the caller's speech, opening a turn if none is open.

        Args:
            audio: 16-bit mono PCM at the session's input sample rate.
        """
        if not self._audio_open:
            await self._open_audio()
        view = memoryview(audio)
        for start in range(0, len(view), self._frame_bytes):
            await self._send(
                {
                    "audioInput": {
                        "promptName": self._names.prompt,
                        "contentName": self._names.audio,
                        "content": await b64encode(
                            view[start : start + self._frame_bytes]
                        ),
                    }
                }
            )
        self._last_sent = get_running_loop().time()

    async def send_text(self, text: str) -> None:
        """Add a written message from the caller to the conversation.

        Args:
            text: What the caller wrote.
        """
        content = uuid4().hex
        await self._send(
            {
                "contentStart": {
                    "promptName": self._names.prompt,
                    "contentName": content,
                    "type": "TEXT",
                    "interactive": True,
                    "role": "USER",
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            }
        )
        await self._send(
            {
                "textInput": {
                    "promptName": self._names.prompt,
                    "contentName": content,
                    "content": text,
                }
            }
        )
        await self._send(
            {"contentEnd": {"promptName": self._names.prompt, "contentName": content}}
        )
        self._pending_text = True
        self._last_sent = get_running_loop().time()

    async def end_turn(self) -> None:
        """End the caller's turn, which is what starts the model answering.

        The trailing silence is not padding: the model finishes an utterance
        when it hears the speaker stop, and a block that ends on the last spoken
        sample is never finished at all -- no answer comes, and the session dies
        on its own idle timeout about a minute later. A written turn is answered
        the same way, through an audio block carrying only that silence: the
        model answers a turn it heard end, whatever the turn was made of.
        """
        if not self._audio_open and not self._pending_text:
            return
        self._pending_text = False
        await self.send_audio(self._trailing_silence)
        await self._close_audio()

    async def _close_audio(self) -> None:
        """Close the audio content block the model is listening to, if any."""
        if not self._audio_open:
            return
        # Closed before the send: keepalive silence past a contentEnd is fatal.
        self._audio_open = False
        await self._send(
            {
                "contentEnd": {
                    "promptName": self._names.prompt,
                    "contentName": self._names.audio,
                }
            }
        )
        # Every turn needs its own name; the closed block is already ended.
        self._names.audio = uuid4().hex

    async def events(self) -> AsyncGenerator[BackendEvent]:
        """Yield what the model reports, as neutral events.

        Yields:
            One event per backend event that carries meaning.
        """
        async for event in self._session:
            payload = getattr(getattr(event, "value", None), "bytes_", None)
            if not payload:
                continue
            decoded = from_json(payload)
            body = decoded.get("event") if isinstance(decoded, dict) else None
            if not isinstance(body, dict) or not body:
                continue
            name = next(iter(body))
            inner = body[name]
            if not isinstance(inner, dict):
                continue
            for translated in await self._translate(name, inner):
                yield translated

    async def keepalive(self) -> None:
        """Send silence whenever the caller has been quiet for too long.

        The service ends a session that receives nothing for about a minute,
        which a caller pausing between two questions reaches easily.
        """
        while True:
            await sleep(_KEEPALIVE_SECONDS)
            if get_running_loop().time() - self._last_sent < _KEEPALIVE_SECONDS:
                continue
            await self.send_audio(self._silence)

    async def _translate(  # noqa: PLR0911 - one branch per backend event name
        self, name: str, body: dict[str, Any]
    ) -> list[BackendEvent]:
        """Turn one backend event into the neutral events it stands for.

        Args:
            name: The backend event name.
            body: The backend event body.

        Returns:
            The neutral events, in order; empty for anything not reported.
        """
        match name:
            case "userSpeechStart":
                return [SpeechStarted(int(body.get("inputAudioOffsetMs", 0)))]
            case "userSpeechEnd":
                return [SpeechStopped(int(body.get("inputAudioOffsetMs", 0)))]
            case "completionStart":
                return [ResponseStarted()]
            case "contentStart":
                self._note_block(body)
            case "textOutput":
                return self._read_text(body)
            case "audioOutput" if self._speech_output:
                if content := body.get("content"):
                    return [OutputAudio(await b64decode(content))]
            case "contentEnd":
                return _read_content_end(body)
            case "usageEvent":
                return [_read_usage(body)]
        return []

    def _note_block(self, body: dict[str, Any]) -> None:
        """Remember a content block's speaker and generation stage.

        Args:
            body: The ``contentStart`` body.
        """
        if not (content_id := body.get("contentId")):
            return
        if role := body.get("role"):
            self._roles[content_id] = role
        fields = body.get("additionalModelFields")
        if isinstance(fields, str) and fields:
            with suppress(ValueError):
                decoded = from_json(fields.encode())
                if isinstance(decoded, dict):
                    self._stages[content_id] = decoded.get("generationStage", "")

    def _read_text(self, body: dict[str, Any]) -> list[BackendEvent]:
        """Attribute one text event to its speaker.

        Args:
            body: The ``textOutput`` body.

        Returns:
            The neutral event it stands for, or nothing.
        """
        text = body.get("content") or ""
        if not text.strip():
            return []
        content_id = body.get("contentId", "")
        role = self._roles.get(content_id, body.get("role", ""))
        if role == "USER":
            return [InputTranscript(text)]
        if role != "ASSISTANT":
            return []
        # A FINAL block restates the speculative one; both duplicate the answer.
        if self._stages.get(content_id) == _FINAL_STAGE:
            return []
        return [OutputTranscript(text)]

    async def _open_audio(self) -> None:
        """Open the audio content block the model listens to."""
        # Marked open before the send, so the keepalive tick opens no second block.
        self._audio_open = True
        await self._send(
            {
                "contentStart": {
                    "promptName": self._names.prompt,
                    "contentName": self._names.audio,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": _audio_input_configuration(
                        self._sample_rate
                    ),
                }
            }
        )

    async def _send(self, event: dict[str, Any]) -> None:
        """Send one protocol event.

        Args:
            event: The event body, without its ``event`` wrapper.
        """
        await self._session.send(
            InvokeModelWithBidirectionalStreamInputChunk(
                value=BidirectionalInputPayloadPart(
                    bytes_=to_json_bytes({"event": event})
                )
            )
        )

    async def close(self) -> None:
        """End the prompt and the session, best effort.

        The open audio block is closed rather than the turn ended: a session on
        its way out has no use for one more answer, and would be billed for it.

        Bounded: a goodbye sent on a stalled transport would otherwise hold the
        whole session's teardown open for as long as the connection is dead.
        """
        with suppress(Exception):
            async with async_timeout(_CLOSE_TIMEOUT):
                await self._close_audio()
                await self._send({"promptEnd": {"promptName": self._names.prompt}})
                await self._send({"sessionEnd": {}})


def _audio_input_configuration(sample_rate: int) -> dict[str, Any]:
    """Build the audio configuration one input content block declares.

    Args:
        sample_rate: Rate, in hertz, of the audio the caller sends.

    Returns:
        The configuration block.
    """
    return {
        "mediaType": "audio/lpcm",
        "sampleRateHertz": sample_rate,
        "sampleSizeBits": 16,
        "channelCount": 1,
        "audioType": "SPEECH",
    }


def _read_content_end(body: dict[str, Any]) -> list[BackendEvent]:
    """Report the end of the model's answer, and nothing else.

    Args:
        body: The ``contentEnd`` body.

    Returns:
        The neutral event it stands for, or nothing.
    """
    stop_reason = body.get("stopReason")
    if stop_reason == _INTERRUPTED:
        return [ResponseFinished(interrupted=True)]
    if stop_reason == _END_TURN:
        return [ResponseFinished()]
    return []


def _read_usage(body: dict[str, Any]) -> UsageReport:
    """Read the session's running totals from one metering event.

    Totals are reported rather than the deltas beside them: a lost event then
    costs nothing, because the next one restates everything billed so far.

    Args:
        body: The ``usageEvent`` body.

    Returns:
        Everything the session has billed so far.
    """
    total = body.get("details", {}).get("total", {})
    inputs = total.get("input", {})
    outputs = total.get("output", {})
    return UsageReport(
        input_speech_tokens=inputs.get("speechTokens", 0),
        input_text_tokens=inputs.get("textTokens", 0),
        output_speech_tokens=outputs.get("speechTokens", 0),
        output_text_tokens=outputs.get("textTokens", 0),
        total_tokens=body.get("totalTokens", 0),
    )


class RealtimeModel(RealtimeModelBase[Any, Any]):
    """Amazon Nova Sonic live speech-to-speech conversations."""

    __slots__ = ()

    # A string, not a pattern: a pattern would lose these routes to the chat family.
    MATCHER = "amazon.nova-2-sonic"

    INPUT_SAMPLE_RATES: ClassVar[frozenset[int]] = _SAMPLE_RATES

    OUTPUT_SAMPLE_RATES: ClassVar[frozenset[int]] = _SAMPLE_RATES

    DEFAULT_VOICE: ClassVar[str] = "matthew"

    MAX_SESSION_SECONDS: ClassVar[float] = _MAX_SESSION_SECONDS

    @asynccontextmanager
    async def open_session(
        self,
        *,
        instructions: str,
        input_sample_rate: int,
        output_sample_rate: int,
        voice: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        speech_output: bool = True,
    ) -> AsyncIterator[RealtimeBackendSession]:
        """Open one live conversation and close it when the caller is done.

        Args:
            instructions: System instructions opening the session.
            input_sample_rate: Rate, in hertz, of the audio the caller sends.
            output_sample_rate: Rate, in hertz, the model should speak at.
            voice: Voice the model answers with, or None for the default one.
            temperature: Optional sampling temperature.
            max_output_tokens: Optional cap on the tokens one answer may use.
            speech_output: Whether the model should speak its answers.

        Yields:
            The open session, whose serving region is on ``region``.

        Raises:
            ApiError: No candidate region could open the session.
        """
        names = _SessionNames()
        regions = await compute_candidate_regions(self._model_id)
        async with open_bidi_stream(
            "bedrock-runtime",
            regions,
            self._open_stream,
            prime=_priming(
                names,
                instructions,
                output_sample_rate,
                _VOICES.get(voice or "", voice) or self.DEFAULT_VOICE,
                temperature,
                max_output_tokens,
            ),
        ) as stream:
            set_effective_region(self._model_id, stream.region)
            session = _NovaSonicSession(
                stream,
                names,
                input_sample_rate,
                stream.region,
                speech_output=speech_output,
            )
            keepalive = create_task(session.keepalive())
            try:
                yield session
            finally:
                keepalive.cancel()
                with suppress(CancelledError, Exception):
                    await keepalive
                await session.close()

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


def _priming(
    names: _SessionNames,
    instructions: str,
    output_sample_rate: int,
    voice: str,
    temperature: float | None,
    max_output_tokens: int | None,
) -> Callable[[BidiSession[Any, Any]], Coroutine[Any, Any, None]]:
    """Build the handshake the service needs before it answers at all.

    Args:
        names: The session's content identifiers.
        instructions: System instructions opening the session.
        output_sample_rate: Rate, in hertz, the model should speak at.
        voice: Voice the model answers with.
        temperature: Optional sampling temperature.
        max_output_tokens: Optional cap on the tokens one answer may use.

    Returns:
        A coroutine function sending the handshake on a session.
    """

    async def prime(session: BidiSession[Any, Any]) -> None:
        """Send the handshake events, in the only order the service accepts."""
        inference: dict[str, Any] = {}
        if temperature is not None:
            inference["temperature"] = temperature
        if max_output_tokens is not None:
            inference["maxTokens"] = max_output_tokens
        prompt_start: dict[str, Any] = {
            "promptName": names.prompt,
            "textOutputConfiguration": {"mediaType": "text/plain"},
        }
        # Not optional: without it the session stays silent, transcript or not.
        prompt_start["audioOutputConfiguration"] = {
            "mediaType": "audio/lpcm",
            "sampleRateHertz": output_sample_rate,
            "sampleSizeBits": 16,
            "channelCount": 1,
            "voiceId": voice,
        }
        events: list[dict[str, Any]] = [
            {"sessionStart": {"inferenceConfiguration": inference}},
            {"promptStart": prompt_start},
            # The session's first content block must be the system one.
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
            {
                "textInput": {
                    "promptName": names.prompt,
                    "contentName": names.instructions,
                    "content": instructions,
                }
            },
            {
                "contentEnd": {
                    "promptName": names.prompt,
                    "contentName": names.instructions,
                }
            },
        ]
        for event in events:
            await session.send(
                InvokeModelWithBidirectionalStreamInputChunk(
                    value=BidirectionalInputPayloadPart(
                        bytes_=to_json_bytes({"event": event})
                    )
                )
            )

    return prime
