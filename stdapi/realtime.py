"""Realtime sessions: ephemeral secrets, audio conversion and one live socket.

A realtime session is the one request shape the HTTP middleware never sees: a
WebSocket scope short-circuits ``BaseHTTPMiddleware``, so the request log, the
per-request header configuration, the usage accounting and the scheduled
cleanups are all opened here instead, per session, and the bill is flushed as
the session runs rather than once it ends.
"""

from array import array
from asyncio import CancelledError, create_task, get_running_loop, shield, to_thread
from asyncio import timeout as async_timeout
from binascii import Error as BinasciiError
from contextlib import AsyncExitStack, suppress
from hashlib import blake2b
from hmac import compare_digest, digest
from secrets import token_bytes
from sys import byteorder
from time import time
from traceback import format_exception
from typing import TYPE_CHECKING, Any, Final, NamedTuple, TypeIs
from uuid import uuid4

from pybase64 import urlsafe_b64decode, urlsafe_b64encode
from pydantic import ValidationError
from pydantic_core import from_json
from starlette.websockets import WebSocketDisconnect, WebSocketState

from stdapi.api_errors import ApiError
from stdapi.auth import (
    enforce_tenant_endpoint_scope,
    realtime_signing_key,
    verify_websocket_credentials,
)
from stdapi.aws_bedrock import (
    GuardrailInterventionError,
    apply_guardrail_to_text,
    set_guardrail_configuration,
    set_performance_configuration,
)
from stdapi.aws_bedrock_mantle import set_mantle_project
from stdapi.cleanup import CLEANUPS, drain_tasks, run_cleanups_detached
from stdapi.config import SETTINGS
from stdapi.input_file import reset_current_input_files
from stdapi.models import validate_model
from stdapi.models.realtime import (
    InputTranscript,
    OutputAudio,
    OutputTranscript,
    RealtimeModelBase,
    ResponseFinished,
    ResponseStarted,
    SpeechStarted,
    SpeechStopped,
    UsageReport,
    get_realtime_model,
)
from stdapi.monitoring import (
    PRINCIPAL,
    TENANT,
    flush_usage_log_event,
    log_error_details,
    log_request_event,
)
from stdapi.tenant_keys import resume_tenant
from stdapi.types.openai_realtime import (
    FORMAT_SAMPLE_RATES,
    PCM_SAMPLE_RATE,
    RealtimeSessionConfig,
    SessionConfig,
    TranscriptionSessionConfig,
)
from stdapi.usage import record_bedrock_usage
from stdapi.utils import b64decode, b64encode, to_json_bytes, to_json_str, webuuid

if TYPE_CHECKING:
    from asyncio import Task
    from collections.abc import Iterator

    from fastapi import WebSocket
    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.models.realtime import BackendEvent, RealtimeBackendSession
    from stdapi.types import JsonList, JsonMapping

#: Prefix every ephemeral client secret carries, as upstream mints them.
CLIENT_SECRET_PREFIX: Final = "ek_"  # noqa: S105 - a prefix, not a secret

#: Close code a session ended by an error carries, as upstream sends it.
ERROR_CLOSE_CODE: Final = 3000

#: Subprotocol echoed back when a browser client offers it.
_SUBPROTOCOL: Final = "realtime"

#: Subprotocol prefix carrying an API key from a browser client.
_KEY_SUBPROTOCOL_PREFIX: Final = "openai-insecure-api-key."

#: Personalisation of the key the client secrets are signed with.
_SIGNING_PERSON: Final = b"stdapi-rt"

#: Signature length, in bytes, of a minted client secret.
_SIGNATURE_SIZE: Final = 32

#: Instructions opening a session that asked for none.
_DEFAULT_INSTRUCTIONS: Final = (
    "You are a helpful, concise voice assistant. Reply in spoken sentences."
)

#: Instructions opening a transcription session, which must not converse.
_TRANSCRIPTION_INSTRUCTIONS: Final = (
    "Listen to the user's speech. Do not reply and do not ask questions."
)

#: Sessions currently open, closed together on shutdown.
_OPEN_SESSIONS: Final[set[RealtimeSession]] = set()

#: Whether the server is shutting down and must open no further session.
_SHUTTING_DOWN = False

#: Seconds a closing session gets to send its goodbye before it is dropped.
_CLOSE_TIMEOUT: Final = 2.0

#: Seconds the backend reader gets to end itself once its stream is closed.
_BACKEND_STOP_TIMEOUT: Final = 2.0

#: Reader stops still finishing after their session's teardown was cancelled.
_STOP_TASKS: Final[set[Task[None]]] = set()

#: Close code and reason sent to every session still open at shutdown.
_SHUTDOWN_CLOSE: Final = (1001, "server_shutdown")

#: Message closing a session that failed in a way nothing else answered for.
_UNEXPECTED_ERROR: Final = "The request could not be completed. Retry the request."

#: Bytes of 16-bit audio the client may buffer before a commit is required.
_MAX_BUFFERED_AUDIO_BYTES: Final = 24000 * 2 * 120

#: Largest single client event accepted, in bytes.
_MAX_EVENT_BYTES: Final = 4 * 1024 * 1024

#: Conversation items a session keeps addressable, oldest dropped past it.
_MAX_TRACKED_ITEMS: Final = 200

#: Statuses of an answer nothing cut short, which carry no status details.
_CLEAN_RESPONSE_STATUSES: Final = frozenset({"in_progress", "completed"})

#: Why an answer ended early, in the upstream vocabulary, keyed by its status.
_RESPONSE_END_REASONS: Final = {
    "cancelled": "client_cancelled",
    "incomplete": "turn_detected",
}


def _ulaw_decode_table() -> array[int]:
    """Build the G.711 mu-law byte to 16-bit sample table.

    Returns:
        One signed 16-bit sample per encoded byte value.
    """
    table = array("h", bytes(512))
    for encoded in range(256):
        value = ~encoded & 0xFF
        magnitude = (((value & 0x0F) << 3) + 0x84) << ((value & 0x70) >> 4)
        sample = magnitude - 0x84
        table[encoded] = -sample if value & 0x80 else sample
    return table


def _alaw_decode_table() -> array[int]:
    """Build the G.711 A-law byte to 16-bit sample table.

    Returns:
        One signed 16-bit sample per encoded byte value.
    """
    table = array("h", bytes(512))
    for encoded in range(256):
        value = encoded ^ 0x55
        exponent = (value & 0x70) >> 4
        mantissa = value & 0x0F
        sample = (
            (mantissa << 4) + 8
            if exponent == 0
            else ((mantissa << 4) + 0x108) << (exponent - 1)
        )
        table[encoded] = -sample if value & 0x80 else sample
    return table


def _encode_table(decoded: array[int]) -> bytes:
    """Invert a decode table into a sample to byte table.

    Args:
        decoded: The codec's byte-to-sample table.

    Returns:
        One encoded byte per signed 16-bit sample, indexed by ``sample + 32768``.
    """
    pairs = sorted((value, code) for code, value in enumerate(decoded))
    table = bytearray(65536)
    index = 0
    for sample in range(-32768, 32768):
        while index + 1 < len(pairs) and abs(pairs[index + 1][0] - sample) <= abs(
            pairs[index][0] - sample
        ):
            index += 1
        table[sample + 32768] = pairs[index][1]
    return bytes(table)


#: G.711 mu-law and A-law conversion tables, built once at import.
_ULAW_DECODE: Final = _ulaw_decode_table()
_ALAW_DECODE: Final = _alaw_decode_table()
_ULAW_ENCODE: Final = _encode_table(_ULAW_DECODE)
_ALAW_ENCODE: Final = _encode_table(_ALAW_DECODE)

#: Decode and encode tables of each companded format, keyed by its media type.
_COMPANDED: Final[dict[str, tuple[array[int], bytes]]] = {
    "audio/pcmu": (_ULAW_DECODE, _ULAW_ENCODE),
    "audio/pcma": (_ALAW_DECODE, _ALAW_ENCODE),
}


def _samples(pcm: bytes) -> array[int]:
    """Read little-endian 16-bit samples out of a PCM buffer.

    Args:
        pcm: Little-endian 16-bit mono samples.

    Returns:
        The samples, in host order.
    """
    values = array("h")
    values.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    if byteorder != "little":
        values.byteswap()
    return values


def _pcm_bytes(values: array[int]) -> bytes:
    """Write samples back out as little-endian 16-bit PCM.

    Args:
        values: Signed 16-bit samples in host order.

    Returns:
        The little-endian bytes.
    """
    if byteorder != "little":
        values = array("h", values)
        values.byteswap()
    return values.tobytes()


def decode_client_audio(audio: bytes, media_type: str) -> bytes:
    """Turn one chunk of the caller's audio into 16-bit mono PCM.

    Args:
        audio: The chunk as the caller sent it.
        media_type: The session's input format.

    Returns:
        Little-endian 16-bit mono samples at the format's own sample rate.
    """
    if (tables := _COMPANDED.get(media_type)) is None:
        return audio
    decode = tables[0]
    return _pcm_bytes(array("h", (decode[byte] for byte in audio)))


def encode_client_audio(pcm: bytes, media_type: str) -> bytes:
    """Turn one chunk of the model's speech into the caller's own format.

    Args:
        pcm: Little-endian 16-bit mono samples at the format's sample rate.
        media_type: The session's output format.

    Returns:
        The chunk in the format the caller asked for.
    """
    if (tables := _COMPANDED.get(media_type)) is None:
        return pcm
    encode = tables[1]
    return bytes(encode[sample + 32768] for sample in _samples(pcm))


#: Bytes below which a G.711 conversion runs inline instead of on a thread.
_COMPANDED_INLINE_MAX_BYTES: Final = 16 * 1024


async def _decoded(audio: bytes, media_type: str) -> bytes:
    """Turn the caller's audio into PCM, off the event loop when it is large.

    Args:
        audio: The chunk as the caller sent it.
        media_type: The session's input format.

    Returns:
        Little-endian 16-bit mono samples at the format's own sample rate.
    """
    if media_type not in _COMPANDED or len(audio) <= _COMPANDED_INLINE_MAX_BYTES:
        return decode_client_audio(audio, media_type)
    return await to_thread(decode_client_audio, audio, media_type)


async def _encoded(pcm: bytes, media_type: str) -> bytes:
    """Turn the model's speech into the caller's format, off the loop when large.

    Args:
        pcm: Little-endian 16-bit mono samples at the format's sample rate.
        media_type: The session's output format.

    Returns:
        The chunk in the format the caller asked for.
    """
    if media_type not in _COMPANDED or len(pcm) <= _COMPANDED_INLINE_MAX_BYTES:
        return encode_client_audio(pcm, media_type)
    return await to_thread(encode_client_audio, pcm, media_type)


def _signing_key() -> bytes:
    """Return the key ephemeral client secrets are signed with.

    Derived from the deployment's own credential so every instance behind a load
    balancer verifies what any other minted, with a configured key overriding it
    and a per-process key as the last resort.

    Returns:
        The 32-byte signing key.
    """
    global _RANDOM_SIGNING_KEY  # noqa: PLW0603
    if (configured := SETTINGS.realtime_client_secret_key) is not None:
        return blake2b(
            configured.get_secret_value().encode(),
            digest_size=_SIGNATURE_SIZE,
            person=_SIGNING_PERSON,
        ).digest()
    if (derived := realtime_signing_key(_SIGNING_PERSON, _SIGNATURE_SIZE)) is not None:
        return derived
    if _RANDOM_SIGNING_KEY is None:
        _RANDOM_SIGNING_KEY = token_bytes(_SIGNATURE_SIZE)
    return _RANDOM_SIGNING_KEY


#: Key used when the deployment has no credential to derive one from.
_RANDOM_SIGNING_KEY: bytes | None = None


class ClientSecret(NamedTuple):
    """What a verified ephemeral client secret carries.

    Attributes:
        session: The session configuration the secret opens.
        tenant_key_id: Key ID of the tenant whose API key minted the secret,
            binding the session to that tenant's scopes; None when the mint
            was not tenant-authenticated.
    """

    session: SessionConfig
    tenant_key_id: str | None


def mint_client_secret(session: SessionConfig, ttl: int) -> tuple[str, int]:
    """Mint a signed, short-lived secret carrying *session*.

    Nothing is stored: the secret is the session configuration plus a signature,
    so any instance can verify one minted by any other. A mint authorized by a
    tenant API key embeds the tenant's key ID, so the session it opens carries
    the tenant's scopes rather than escaping them.

    Args:
        session: Session configuration the secret opens.
        ttl: Seconds the secret stays usable.

    Returns:
        The secret, and the Unix time in seconds after which it is refused.
    """
    expires_at = int(time()) + ttl
    claims: dict[str, Any] = {
        "exp": expires_at,
        "session": session.model_dump(mode="json"),
    }
    if (tenant := TENANT.get()) is not None:
        claims["tenant"] = tenant.key_id
    payload = to_json_bytes(claims)
    signature = digest(_signing_key(), payload, "sha256")
    return (
        f"{CLIENT_SECRET_PREFIX}{_urlsafe(payload)}.{_urlsafe(signature)}",
        expires_at,
    )


def read_client_secret(  # noqa: PLR0911 - every branch is one way to be invalid
    value: str,
) -> ClientSecret | None:
    """Verify a client secret and return what it carries.

    Args:
        value: The credential the caller presented.

    Returns:
        The session configuration and the minting tenant's key ID, or None
        when *value* is not a valid, unexpired secret this deployment minted.
    """
    if not value.startswith(CLIENT_SECRET_PREFIX):
        return None
    encoded, _, signature = value[len(CLIENT_SECRET_PREFIX) :].partition(".")
    if not signature:
        return None
    try:
        payload = _unurlsafe(encoded)
        provided = _unurlsafe(signature)
    except ValueError:
        return None
    if not compare_digest(digest(_signing_key(), payload, "sha256"), provided):
        return None
    try:
        decoded = from_json(payload)
    except ValueError:
        return None
    if not isinstance(decoded, dict) or decoded.get("exp", 0) < time():
        return None
    tenant_key_id = decoded.get("tenant")
    if tenant_key_id is not None and not isinstance(tenant_key_id, str):
        return None
    session = _parse_session(decoded.get("session"))
    if session is None:
        return None
    return ClientSecret(session, tenant_key_id)


def _parse_session(value: Any) -> SessionConfig | None:  # noqa: ANN401
    """Validate a session configuration mapping.

    Args:
        value: The mapping to validate.

    Returns:
        The parsed configuration, or None when it is not one.
    """
    if not isinstance(value, dict):
        return None
    try:
        if value.get("type") == "transcription":
            return TranscriptionSessionConfig.model_validate(value)
        return RealtimeSessionConfig.model_validate(value)
    except ValidationError:
        return None


def _urlsafe(value: bytes) -> str:
    """Encode *value* as unpadded URL-safe base64.

    Args:
        value: The bytes to encode.

    Returns:
        The encoded text.
    """
    return urlsafe_b64encode(value).decode().rstrip("=")


def _unurlsafe(value: str) -> bytes:
    """Decode unpadded URL-safe base64.

    Args:
        value: The encoded text.

    Returns:
        The decoded bytes.

    Raises:
        ValueError: *value* is not valid base64.
    """
    try:
        return urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except BinasciiError as error:
        raise ValueError(str(error)) from None


def websocket_credential(websocket: WebSocket) -> str | None:
    """Read the credential a WebSocket client presented.

    Three carriers, because three kinds of client exist: the SDK sends an
    ``Authorization`` header, other gateway clients send ``x-api-key``, and a
    browser -- which cannot set headers on a WebSocket at all -- smuggles it
    through the subprotocol list.

    Args:
        websocket: The connection being opened.

    Returns:
        The credential, or None when the client presented none.
    """
    headers = websocket.headers
    if credential := headers.get("x-api-key"):
        return credential
    authorization = headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token
    for offered in _offered_subprotocols(websocket):
        if offered.startswith(_KEY_SUBPROTOCOL_PREFIX):
            return offered[len(_KEY_SUBPROTOCOL_PREFIX) :]
    return None


def _offered_subprotocols(websocket: WebSocket) -> Iterator[str]:
    """Yield each subprotocol the client offered.

    Args:
        websocket: The connection being opened.

    Yields:
        One offered subprotocol per entry, stripped.
    """
    for value in websocket.headers.getlist("sec-websocket-protocol"):
        for offered in value.split(","):
            if stripped := offered.strip():
                yield stripped


def open_realtime_sessions() -> None:
    """Accept sessions again, for a deployment whose lifespan starts anew.

    The shutdown latch is a module global, so a second lifespan in one process
    would otherwise refuse every session it is ever asked to open.
    """
    global _SHUTTING_DOWN  # noqa: PLW0603
    _SHUTTING_DOWN = False


def close_realtime_sessions() -> None:
    """Ask every open session to close, and refuse new ones.

    A termination signal kills an open socket in milliseconds without a close
    frame, and the server offers no graceful WebSocket drain, so the goodbye has
    to be sent from here.
    """
    global _SHUTTING_DOWN  # noqa: PLW0603
    _SHUTTING_DOWN = True
    for session in tuple(_OPEN_SESSIONS):
        session.request_close(*_SHUTDOWN_CLOSE)


async def drain_session_stops(timeout: float) -> int:  # noqa: ASYNC109 -- shared drain contract
    """Await the backend readers still ending after their session's teardown left.

    Args:
        timeout: Seconds allowed before the unfinished readers are cancelled.

    Returns:
        Number of readers that had not finished at the deadline.
    """
    return await drain_tasks(_STOP_TASKS, timeout)


class _Item:
    """One conversation item, as the client sees it and may address it."""

    __slots__ = (
        "audio_ms",
        "content",
        "id",
        "previous_id",
        "role",
        "status",
        "truncated",
    )

    def __init__(
        self, item_id: str, role: str, content: JsonList, status: str = "completed"
    ) -> None:
        """Hold one item of the conversation.

        Args:
            item_id: Identifier the item is reported under.
            role: Who the item belongs to.
            content: The item's content parts.
            status: Status to report.
        """
        self.id = item_id
        self.role = role
        self.content = content
        self.status = status
        self.audio_ms = 0
        self.truncated = False
        self.previous_id: str | None = None


def _item_body(item: _Item) -> JsonMapping:
    """Render one conversation item, wherever it is reported.

    Args:
        item: The item to render.

    Returns:
        The item, in the shape the client expects.
    """
    return {
        "id": item.id,
        "object": "realtime.item",
        "type": "message",
        "status": item.status,
        "role": item.role,
        "content": item.content,
    }


def _is_offset(value: Any) -> TypeIs[int]:  # noqa: ANN401
    """Whether *value* is an index or a millisecond offset a client may send.

    Args:
        value: The field as it arrived, unvalidated.

    Returns:
        True for a non-negative integer, which ``True`` is not.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class _Response:
    """The answer being spoken, and what it has produced so far."""

    __slots__ = ("audio_bytes", "cancelled", "id", "item_id", "started", "transcript")

    def __init__(self) -> None:
        """Start an answer that has produced nothing yet."""
        self.id = f"resp_{uuid4().hex}"
        self.item_id = f"item_{uuid4().hex}"
        self.transcript: list[str] = []
        self.audio_bytes = 0
        self.started = False
        self.cancelled = False


class _Metering:
    """Everything the session billed, and everything already flushed."""

    __slots__ = ("recorded", "totals", "transcribed")

    def __init__(self) -> None:
        """Start with nothing billed and nothing recorded."""
        self.totals = UsageReport()
        self.recorded = UsageReport()
        self.transcribed = UsageReport()

    def pending(self) -> tuple[int, int, int, int, int]:
        """Return what has been billed but not yet recorded.

        Returns:
            Input speech, input text, output speech, output text and total
            tokens accumulated since the last record.
        """
        return (
            self.totals.input_speech_tokens - self.recorded.input_speech_tokens,
            self.totals.input_text_tokens - self.recorded.input_text_tokens,
            self.totals.output_speech_tokens - self.recorded.output_speech_tokens,
            self.totals.output_text_tokens - self.recorded.output_text_tokens,
            self.totals.total_tokens - self.recorded.total_tokens,
        )

    def read_input(self) -> tuple[int, int]:
        """Take what reading the caller's turn cost, and start counting again.

        Returns:
            Speech and text tokens billed for the caller's own input since the
            last transcript was reported.
        """
        totals = self.totals
        counted = (
            totals.input_speech_tokens - self.transcribed.input_speech_tokens,
            totals.input_text_tokens - self.transcribed.input_text_tokens,
        )
        self.transcribed = totals
        return counted


class RealtimeSession:
    """One client socket, its backend conversation, and everything between.

    The two directions run as sibling tasks: what the client sends is applied to
    the backend, and what the backend reports is rendered as the client's own
    events. Neither side buffers, so a slow consumer slows its producer instead
    of growing a queue.
    """

    __slots__ = (
        "_backend",
        "_backend_task",
        "_buffered",
        "_client_task",
        "_closing",
        "_config",
        "_conversation_id",
        "_items",
        "_last_item_id",
        "_locked",
        "_metering",
        "_model",
        "_model_id",
        "_pending_item",
        "_response",
        "_session_id",
        "_stack",
        "_started",
        "_stopping",
        "_suppressed",
        "_websocket",
    )

    def __init__(
        self,
        websocket: WebSocket,
        model: RealtimeModelBase[Any, Any],
        model_id: str,
        config: SessionConfig,
        *,
        locked: bool = False,
    ) -> None:
        """Bind one client socket to the model that will serve it.

        Args:
            websocket: The accepted connection.
            model: The model class serving the session.
            model_id: Identifier of the model serving the session.
            config: Session configuration the client connected with.
            locked: Whether the client may not change what its credential
                pinned: the model, the instructions and the output token cap.
        """
        self._websocket = websocket
        self._model = model
        self._model_id = model_id
        self._config = config
        self._locked = locked
        self._session_id = f"sess_{uuid4().hex}"
        self._conversation_id = f"conv_{uuid4().hex}"
        self._backend: RealtimeBackendSession | None = None
        self._backend_task: Task[None] | None = None
        self._client_task: Task[None] | None = None
        self._stack = AsyncExitStack()
        self._response: _Response | None = None
        self._metering = _Metering()
        self._buffered = bytearray()
        self._items: dict[str, _Item] = {}
        self._last_item_id: str | None = None
        self._pending_item: str | None = None
        self._suppressed = False
        self._stopping = False
        self._closing: tuple[int, str] | None = None
        self._started = get_running_loop().time()

    def request_close(self, code: int, reason: str) -> None:
        """Ask the session to stop, from outside its own tasks.

        Args:
            code: WebSocket close code to send.
            reason: Close reason to send.
        """
        self._closing = (code, reason)
        if self._client_task is not None:
            self._client_task.cancel()

    async def run(self) -> None:
        """Serve the session until either side ends it."""
        _OPEN_SESSIONS.add(self)
        try:
            await self._send_event(
                {"type": "session.created", "session": self._session_view()}
            )
            await self._serve()
        except WebSocketDisconnect:
            pass
        except TimeoutError:
            self._closing = (1000, "session_expired")
        except ApiError as exception:
            await self._fail(exception)
        finally:
            _OPEN_SESSIONS.discard(self)
            self._record_usage()
            await self._close(*(self._closing or (1000, "")))

    async def _serve(self) -> None:
        """Run the client and backend directions until one of them ends."""
        try:
            self._client_task = create_task(self._pump_client())
            try:
                async with async_timeout(self._model.MAX_SESSION_SECONDS):
                    await self._client_task
            except CancelledError:
                # Only a cancellation from the backend half ends the session.
                if self._closing is None:
                    raise
            finally:
                await self._stop_client()
        finally:
            self._stopping = True
            try:
                await self._stack.aclose()
            finally:
                await self._stop_backend()

    async def _stop_client(self) -> None:
        """Stop reading the client, which a timed-out session leaves pending."""
        if (task := self._client_task) is None or task.done():
            return
        task.cancel()
        with suppress(CancelledError, Exception):
            await task

    async def _stop_backend(self) -> None:
        """Stop reading the backend, whose stream is already closed.

        Awaited rather than cancelled: a reader cancelled while a chunk of the
        backend's answer is still outstanding leaves the transport holding a
        cancelled result it completes anyway, which surfaces as an unhandled
        failure in whatever the process runs next. The wait runs in its own
        task and is awaited through a shield: a teardown that is itself being
        cancelled would otherwise cancel the reader after all, mid-read.
        """
        if (task := self._backend_task) is None:
            return
        self._backend_task = None
        stop = create_task(_finish_reader(task))
        _STOP_TASKS.add(stop)
        stop.add_done_callback(_STOP_TASKS.discard)
        # Only this second delivery is dropped; the one that triggered it propagates.
        with suppress(CancelledError):
            await shield(stop)

    async def _pump_client(self) -> None:
        """Apply everything the client sends, until it stops or the socket does."""
        while self._closing is None:
            message = await self._websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            payload = message.get("text") or message.get("bytes") or ""
            if len(payload) > _MAX_EVENT_BYTES:
                await self._error("invalid_request_error", "Event payload too large.")
                continue
            try:
                event = from_json(payload)
            except ValueError:
                await self._error("invalid_request_error", "Event is not valid JSON.")
                continue
            if not isinstance(event, dict):
                await self._error("invalid_request_error", "Event is not an object.")
                continue
            await self._apply(event)

    async def _apply(self, event: JsonMapping) -> None:  # noqa: C901 - one arm per client event
        """Apply one client event.

        Args:
            event: The event the client sent.
        """
        match event.get("type"):
            case "session.update":
                await self._update_session(event.get("session"))
            case "input_audio_buffer.append":
                await self._append_audio(event.get("audio"))
            case "input_audio_buffer.commit":
                await self._commit_audio()
            case "input_audio_buffer.clear":
                self._buffered.clear()
                await self._send_event({"type": "input_audio_buffer.cleared"})
            case "conversation.item.create":
                await self._create_item(event.get("item"))
            case "conversation.item.truncate":
                await self._truncate_item(event)
            case "conversation.item.retrieve":
                await self._retrieve_item(event.get("item_id"))
            case "conversation.item.delete":
                await self._delete_item(event.get("item_id"))
            case "response.create":
                await self._create_response()
            case "response.cancel":
                await self._cancel_response()
            case "output_audio_buffer.clear":
                await self._send_event({"type": "output_audio_buffer.cleared"})
            case _ as kind:
                await self._error(
                    "invalid_request_error",
                    f"Unknown or unsupported event type '{kind}'.",
                )

    async def _update_session(self, session: Any) -> None:  # noqa: ANN401
        """Replace the session configuration and acknowledge it.

        Args:
            session: The configuration the client sent.
        """
        if session is not None and not isinstance(session, dict):
            await self._error("invalid_request_error", "'session' must be an object.")
            return
        merged = _deep_merge(self._config.model_dump(mode="json"), session or {})
        merged["type"] = self._config.type
        if (parsed := _parse_session(merged)) is None:
            await self._error(
                "invalid_request_error", "The session configuration is not valid."
            )
            return
        if self._locked and not self._same_pinned_settings(parsed):
            await self._error(
                "invalid_request_error",
                "The model, the instructions and the output token cap this "
                "credential was issued for cannot be changed.",
            )
            return
        if self._backend is not None and not self._same_backend_settings(parsed):
            await self._error(
                "invalid_request_error",
                "The instructions, voice and audio formats cannot be changed once "
                "the model has answered. Open a new session to change them.",
            )
            return
        self._config = parsed
        await self._send_event(
            {"type": "session.updated", "session": self._session_view()}
        )

    def _same_pinned_settings(self, other: SessionConfig) -> bool:
        """Whether *other* keeps everything the credential pinned.

        Args:
            other: The configuration the client asked for.

        Returns:
            True when the model, the instructions and the output token cap the
            session was opened with are all unchanged.
        """
        current = self._config
        return (
            _instructions(current) == _instructions(other)
            and current.model == other.model
            and _max_output_tokens(current) == _max_output_tokens(other)
        )

    def _same_backend_settings(self, other: SessionConfig) -> bool:
        """Whether *other* keeps everything the open conversation fixed.

        Args:
            other: The configuration the client asked for.

        Returns:
            True when nothing the backend session cannot change was changed.
        """
        current = self._config
        return (
            _instructions(current) == _instructions(other)
            and current.audio.input.format.type == other.audio.input.format.type
            and current.audio.output.format.type == other.audio.output.format.type
            and current.audio.output.voice == other.audio.output.voice
        )

    async def _append_audio(self, audio: Any) -> None:  # noqa: ANN401
        """Send one chunk of the caller's speech to the model.

        Args:
            audio: The base64 chunk the client sent.
        """
        if not isinstance(audio, str) or not audio:
            await self._error("invalid_request_error", "'audio' must be base64 audio.")
            return
        try:
            decoded = await b64decode(audio)
        except ValueError:
            await self._error("invalid_request_error", "'audio' is not valid base64.")
            return
        pcm = await _decoded(decoded, self._config.audio.input.format.type)
        if self._manual_turns():
            if len(self._buffered) + len(pcm) > _MAX_BUFFERED_AUDIO_BYTES:
                await self._error(
                    "invalid_request_error",
                    "Too much audio was buffered without a commit.",
                )
                return
            self._buffered.extend(pcm)
            return
        await (await self._ensure_backend()).send_audio(pcm)

    async def _commit_audio(self) -> None:
        """End the caller's turn, which is what starts the model answering."""
        backend = await self._ensure_backend()
        if self._buffered:
            await backend.send_audio(self._buffered)
            self._buffered.clear()
        await backend.end_turn()
        self._pending_item = item_id = f"item_{uuid4().hex}"
        await self._send_event(
            {"type": "input_audio_buffer.committed", "item_id": item_id}
        )
        item = _Item(item_id, "user", [{"type": "input_audio", "transcript": None}])
        await self._add_item(item)
        await self._finish_item(item)

    async def _create_item(self, item: Any) -> None:  # noqa: ANN401
        """Add a conversation item the client wrote.

        Args:
            item: The item the client sent.
        """
        if not isinstance(item, dict):
            await self._error("invalid_request_error", "'item' must be an object.")
            return
        content = item.get("content") or []
        if not isinstance(content, list):
            await self._error("invalid_request_error", "'content' must be an array.")
            return
        texts = [
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") in {"input_text", "text"}
            and isinstance(part.get("text"), str)
            and part["text"]
        ]
        if not texts:
            await self._error(
                "invalid_request_error",
                "Only text conversation items can be added to a session; send "
                "speech with input_audio_buffer.append.",
            )
            return
        text = await apply_guardrail_to_text("\n".join(texts), source="INPUT")
        await (await self._ensure_backend()).send_text(text)
        role = item.get("role", "user")
        tracked = _Item(
            item.get("id") or f"item_{uuid4().hex}",
            role if isinstance(role, str) else "user",
            [{"type": "input_text", "text": text}],
        )
        await self._add_item(tracked, created=True)
        await self._finish_item(tracked)

    async def _add_item(self, item: _Item, *, created: bool = False) -> None:
        """Track *item* and announce that the conversation now holds it.

        Args:
            item: The item being added.
            created: Whether to also send ``conversation.item.created``, which
                clients written before the added/done pair listen for.
        """
        item.previous_id = self._last_item_id
        self._last_item_id = item.id
        self._items[item.id] = item
        while len(self._items) > _MAX_TRACKED_ITEMS:
            del self._items[next(iter(self._items))]
        body = _item_body(item)
        if created:
            await self._send_event(
                {
                    "type": "conversation.item.created",
                    "previous_item_id": item.previous_id,
                    "item": body,
                }
            )
        await self._send_event(
            {
                "type": "conversation.item.added",
                "previous_item_id": item.previous_id,
                "item": body,
            }
        )

    async def _finish_item(self, item: _Item) -> None:
        """Announce that *item* has settled and will not change further.

        Args:
            item: The item that settled.
        """
        await self._send_event(
            {
                "type": "conversation.item.done",
                "previous_item_id": item.previous_id,
                "item": _item_body(item),
            }
        )

    async def _truncate_item(self, event: JsonMapping) -> None:
        """Cut an answer down to what the caller actually heard.

        A caller who speaks over the answer has heard only its beginning, so
        the rest of it is dropped from the session's own record of the item:
        what is left is what a later ``conversation.item.retrieve`` reports.

        Args:
            event: The ``conversation.item.truncate`` event the client sent.
        """
        item_id = event.get("item_id")
        content_index = event.get("content_index")
        audio_end_ms = event.get("audio_end_ms")
        if (
            not isinstance(item_id, str)
            or not _is_offset(content_index)
            or not _is_offset(audio_end_ms)
        ):
            await self._error(
                "invalid_request_error",
                "'item_id', 'content_index' and 'audio_end_ms' are required to "
                "truncate a conversation item.",
            )
            return
        item = self._items.get(item_id)
        if item is None or item.role != "assistant":
            await self._error(
                "invalid_request_error",
                f"No assistant message item '{item_id}' is available to truncate.",
            )
            return
        if content_index >= len(item.content) or audio_end_ms > item.audio_ms:
            await self._error(
                "invalid_request_error",
                f"The item '{item_id}' has no audio at content index "
                f"{content_index} lasting {audio_end_ms} ms.",
            )
            return
        item.audio_ms = audio_end_ms
        item.truncated = True
        part = item.content[content_index]
        if isinstance(part, dict):
            # Nothing aligns a transcript to the audio, so all of it is dropped.
            item.content[content_index] = {**part, "transcript": ""}
        await self._send_event(
            {
                "type": "conversation.item.truncated",
                "item_id": item_id,
                "content_index": content_index,
                "audio_end_ms": audio_end_ms,
            }
        )

    async def _retrieve_item(self, item_id: Any) -> None:  # noqa: ANN401
        """Report the session's own record of one conversation item.

        Args:
            item_id: Identifier of the item the client asked for.
        """
        if not isinstance(item_id, str) or (item := self._items.get(item_id)) is None:
            await self._unknown_item(item_id)
            return
        await self._send_event(
            {"type": "conversation.item.retrieved", "item": _item_body(item)}
        )

    async def _delete_item(self, item_id: Any) -> None:  # noqa: ANN401
        """Drop one conversation item from the session's own record.

        Args:
            item_id: Identifier of the item the client asked to remove.
        """
        if not isinstance(item_id, str) or self._items.pop(item_id, None) is None:
            await self._unknown_item(item_id)
            return
        if self._last_item_id == item_id:
            self._last_item_id = next(reversed(self._items), None)
        await self._send_event(
            {"type": "conversation.item.deleted", "item_id": item_id}
        )

    async def _unknown_item(self, item_id: Any) -> None:  # noqa: ANN401
        """Report that no item answers to what the client addressed.

        Args:
            item_id: Identifier the client sent, of any shape.
        """
        named = f" '{item_id}'" if isinstance(item_id, str) else ""
        await self._error(
            "invalid_request_error",
            f"No conversation item{named} is available in this session.",
        )

    async def _create_response(self) -> None:
        """Start the model answering, ending the caller's turn if one is open."""
        backend = await self._ensure_backend()
        if self._buffered:
            await backend.send_audio(self._buffered)
            self._buffered.clear()
        await backend.end_turn()

    async def _cancel_response(self) -> None:
        """Stop reporting the answer in progress.

        The model is not told to stop, so what it keeps speaking is dropped
        rather than reported as a second answer the caller never asked for.
        """
        if (response := self._response) is None:
            return
        response.cancelled = True
        self._suppressed = True
        await self._finish_response(interrupted=True)

    def _manual_turns(self) -> bool:
        """Whether the caller ends every turn itself.

        Returns:
            True when turn detection is off, so audio is held until a commit.
        """
        return self._config.audio.input.turn_detection is None

    async def _ensure_backend(self) -> RealtimeBackendSession:
        """Open the model conversation on first use.

        Opening is deferred because a client configures its session before
        speaking, and the instructions, voice and audio formats are fixed for
        the whole conversation once it is open.

        Returns:
            The open backend session.

        Raises:
            ApiError: The conversation could not be opened.
        """
        if self._backend is not None:
            return self._backend
        audio = self._config.audio
        self._check_format(audio.input.format.type, self._model.INPUT_SAMPLE_RATES)
        self._check_format(audio.output.format.type, self._model.OUTPUT_SAMPLE_RATES)
        self._backend = await self._stack.enter_async_context(
            self._model.open_session(
                instructions=_instructions(self._config),
                input_sample_rate=sample_rate(audio.input.format.type),
                output_sample_rate=sample_rate(audio.output.format.type),
                voice=audio.output.voice,
                max_output_tokens=_max_output_tokens(self._config),
                speech_output=self._speech_output(),
            )
        )
        self._backend_task = create_task(self._drive_backend())
        return self._backend

    @staticmethod
    def _check_format(media_type: str, rates: frozenset[int]) -> None:
        """Refuse an audio format this model cannot carry, before it opens.

        Args:
            media_type: The format the session is configured with.
            rates: Sample rates the model accepts, empty when it declares none.

        Raises:
            ApiError: The model cannot serve that format.
        """
        if not rates or sample_rate(media_type) in rates:
            return
        served = "', '".join(
            sorted(name for name, rate in FORMAT_SAMPLE_RATES.items() if rate in rates)
        )
        message = (
            f"The audio format '{media_type}' is not available with this model. "
            f"Open the session with '{served}' instead."
        )
        raise ApiError(message)

    async def _drive_backend(self) -> None:
        """Render everything the backend reports as the client's own events."""
        if (backend := self._backend) is None:  # pragma: no cover - never opened
            return
        try:
            async for event in backend.events():
                await self._report(event)
                if self._closing is not None:
                    break
        except CancelledError:
            raise
        except ApiError as exception:
            # A stream closed by the teardown fails by design; nothing owes for it.
            if not self._stopping:
                await self._fail(exception)
        else:
            # The backend ended the conversation, so the client cannot send into it.
            self._closing = self._closing or (1000, "session_ended")
        if self._client_task is not None:
            self._client_task.cancel()

    async def _report(self, event: BackendEvent) -> None:
        """Render one backend event.

        Args:
            event: What the backend reported.
        """
        match event:
            case SpeechStarted():
                await self._send_event(
                    {
                        "type": "input_audio_buffer.speech_started",
                        "audio_start_ms": event.offset_ms,
                        "item_id": self._pending_item or "",
                    }
                )
            case SpeechStopped():
                await self._send_event(
                    {
                        "type": "input_audio_buffer.speech_stopped",
                        "audio_end_ms": event.offset_ms,
                        "item_id": self._pending_item or "",
                    }
                )
            case InputTranscript():
                await self._report_input_transcript(event.text)
            case ResponseStarted():
                await self._start_response()
            case OutputTranscript():
                await self._report_answer(event.text)
            case OutputAudio():
                await self._report_audio(event.audio)
            case ResponseFinished() if self._suppressed:
                # The cancelled answer still ran: report nothing, bill everything.
                self._suppressed = False
                self._record_usage()
            case ResponseFinished():
                await self._finish_response(interrupted=event.interrupted)
            case UsageReport():
                self._metering.totals = event

    async def _report_input_transcript(self, text: str) -> None:
        """Check a transcript of the caller's speech, and report it if asked for.

        The guardrail runs on every transcript, including the ones the caller
        never asked to receive: it is the only reading of the caller's speech
        this route has.

        Args:
            text: What the caller said.

        Raises:
            ApiError: The transcript could not be checked.
            GuardrailInterventionError: The guardrail blocked what was said.
        """
        try:
            text = await apply_guardrail_to_text(text, source="INPUT")
        except GuardrailInterventionError:
            raise
        except ApiError as exception:
            await self._report_transcription_failed(exception)
            raise
        if self._config.audio.input.transcription is None:
            return
        item_id = self._pending_item or f"item_{uuid4().hex}"
        self._pending_item = item_id
        if (item := self._items.get(item_id)) is None:
            item = _Item(item_id, "user", [{"type": "input_audio", "transcript": text}])
            await self._add_item(item)
            await self._finish_item(item)
        else:
            item.content = [{"type": "input_audio", "transcript": text}]
        await self._send_event(
            {
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": item_id,
                "content_index": 0,
                "delta": text,
            }
        )
        speech_tokens, text_tokens = self._metering.read_input()
        await self._send_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": item_id,
                "content_index": 0,
                "transcript": text,
                "usage": {
                    "type": "tokens",
                    "input_tokens": speech_tokens + text_tokens,
                    "output_tokens": 0,
                    "total_tokens": speech_tokens + text_tokens,
                    "input_token_details": {
                        "audio_tokens": speech_tokens,
                        "text_tokens": text_tokens,
                    },
                },
            }
        )
        self._pending_item = None

    async def _report_transcription_failed(self, exception: ApiError) -> None:
        """Report that a caller turn could not be transcribed.

        Without it, a failed transcription and a caller who said nothing are
        the same event stream, and a client waiting on the transcript of an
        item has no way to tell them apart.

        Args:
            exception: What went wrong; only its own message reaches the client.
        """
        if self._config.audio.input.transcription is None:
            return
        await self._send_event(
            {
                "type": "conversation.item.input_audio_transcription.failed",
                "item_id": self._pending_item or "",
                "content_index": 0,
                "error": {
                    "type": "server_error"
                    if exception.status >= 500
                    else "invalid_request_error",
                    "code": exception.code,
                    "message": exception.args[0],
                    "param": None,
                },
            }
        )

    async def _start_response(self) -> None:
        """Announce a new answer."""
        if self._response is not None or self._suppressed:
            return
        response = self._response = _Response()
        await self._send_event(
            {
                "type": "response.created",
                "response": self._response_view(response, "in_progress"),
            }
        )

    async def _open_output_item(self) -> _Response | None:
        """Announce the item and content part the answer is written into.

        Returns:
            The answer in progress, or None when there is none to write into:
            the client cancelled it while this was announcing it.
        """
        if self._response is None:
            await self._start_response()
        if (response := self._response) is None:
            return None
        if response.started:
            return response
        response.started = True
        item = _Item(
            response.item_id,
            "assistant",
            self._assistant_content(""),
            status="in_progress",
        )
        await self._send_event(
            {
                "type": "response.output_item.added",
                "response_id": response.id,
                "output_index": 0,
                "item": _item_body(item),
            }
        )
        await self._add_item(item)
        await self._send_event(
            {
                "type": "response.content_part.added",
                "response_id": response.id,
                "item_id": response.item_id,
                "output_index": 0,
                "content_index": 0,
                "part": {"type": self._part_type(), "transcript": ""},
            }
        )
        return response

    async def _report_answer(self, text: str) -> None:
        """Report what the model answered, as it answers it.

        The same text is the transcript of the speech being generated, or the
        answer itself when the session produces no speech.

        Args:
            text: What the model answered.
        """
        if self._suppressed or (response := await self._open_output_item()) is None:
            return
        response.transcript.append(text)
        await self._send_event(
            {
                "type": "response.output_audio_transcript.delta"
                if self._speech_output()
                else "response.output_text.delta",
                "response_id": response.id,
                "item_id": response.item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            }
        )

    async def _report_audio(self, audio: bytes) -> None:
        """Report one chunk of the model's speech.

        Args:
            audio: 16-bit mono samples at the session's output rate.
        """
        if not self._speech_output() or self._suppressed:
            return
        if (response := await self._open_output_item()) is None or response.cancelled:
            return
        response.audio_bytes += len(audio)
        # Tracked as it is spoken: a barge-in truncates long before the answer ends.
        if (item := self._items.get(response.item_id)) is not None:
            item.audio_ms = self._audio_ms(response.audio_bytes)
        await self._send_event(
            {
                "type": "response.output_audio.delta",
                "response_id": response.id,
                "item_id": response.item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": await b64encode(
                    await _encoded(audio, self._config.audio.output.format.type)
                ),
            }
        )

    async def _finish_response(self, *, interrupted: bool) -> None:
        """Close out the answer, check it, and bill what it consumed.

        The check runs once the answer is complete, which is after its speech
        has been streamed: an intervention therefore ends the session rather
        than the answer.

        Args:
            interrupted: Whether the caller spoke over it.

        Raises:
            GuardrailInterventionError: The guardrail blocked the answer.
        """
        if (response := self._response) is None:
            return
        self._response = None
        try:
            transcript = await apply_guardrail_to_text(
                "".join(response.transcript), source="OUTPUT"
            )
            await self._report_finished(response, transcript, interrupted=interrupted)
        finally:
            # Billed per answer: a dropped socket would take the whole session with it.
            self._record_usage()

    async def _report_finished(
        self, response: _Response, transcript: str, *, interrupted: bool
    ) -> None:
        """Report the end of one answer.

        Args:
            response: The answer that ended.
            transcript: Everything it said, checked.
            interrupted: Whether the caller spoke over it.
        """
        status = (
            "cancelled"
            if response.cancelled
            else ("incomplete" if interrupted else "completed")
        )
        if response.started:
            done_type = (
                "response.output_audio.done"
                if self._speech_output()
                else "response.output_text.done"
            )
            await self._send_event(
                {
                    "type": done_type,
                    "response_id": response.id,
                    "item_id": response.item_id,
                    "output_index": 0,
                    "content_index": 0,
                    **({} if self._speech_output() else {"text": transcript}),
                }
            )
            if self._speech_output():
                await self._send_event(
                    {
                        "type": "response.output_audio_transcript.done",
                        "response_id": response.id,
                        "item_id": response.item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "transcript": transcript,
                    }
                )
            await self._send_event(
                {
                    "type": "response.content_part.done",
                    "response_id": response.id,
                    "item_id": response.item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": self._part_type(), "transcript": transcript},
                }
            )
            await self._send_event(
                {
                    "type": "response.output_item.done",
                    "response_id": response.id,
                    "output_index": 0,
                    "item": self._item_view(response, status, transcript),
                }
            )
            if (item := self._items.get(response.item_id)) is not None:
                item.status = status
                if not item.truncated:
                    item.content = self._assistant_content(transcript)
                    item.audio_ms = self._audio_ms(response.audio_bytes)
                await self._finish_item(item)
        await self._send_event(
            {
                "type": "response.done",
                "response": self._response_view(response, status, transcript),
            }
        )

    def _record_usage(self) -> None:
        """Record what the backend billed since the last record, and flush it."""
        if (backend := self._backend) is None or backend.region is None:
            return
        input_speech, input_text, output_speech, output_text, total = (
            self._metering.pending()
        )
        if not any((input_speech, input_text, output_speech, output_text)):
            return
        self._metering.recorded = self._metering.totals
        region: RegionName = backend.region
        record_bedrock_usage(
            self._model_id,
            region=region,
            input_tokens=input_speech + input_text,
            output_tokens=output_speech + output_text,
            total_tokens=total,
            # Speech tokens are priced an order of magnitude above text ones.
            input_tokens_by_spec={"speech": input_speech} if input_speech else None,
            output_tokens_by_spec={"speech": output_speech} if output_speech else None,
        )
        flush_usage_log_event(int((get_running_loop().time() - self._started) * 1000))

    def _speech_output(self) -> bool:
        """Whether the session answers with speech.

        Returns:
            True unless the caller asked for text-only answers.
        """
        if isinstance(self._config, TranscriptionSessionConfig):
            return False
        modalities = self._config.output_modalities
        return modalities is None or "audio" in modalities

    def _audio_ms(self, audio_bytes: int) -> int:
        """Return how long an answer's speech lasts, in milliseconds.

        Args:
            audio_bytes: 16-bit mono samples at the session's output rate.

        Returns:
            The duration a client would have played, rounded down.
        """
        return audio_bytes * 500 // sample_rate(self._config.audio.output.format.type)

    def _part_type(self) -> str:
        """Return the content part type the answer is streamed into.

        Returns:
            ``audio`` for a spoken answer, ``text`` otherwise.
        """
        return "audio" if self._speech_output() else "text"

    def _assistant_content(self, transcript: str) -> JsonList:
        """Render the content of the conversation item an answer settles into.

        A conversation item names its parts differently from the content part
        events streaming the same answer, and carries what was said under a
        different field in each of its two forms.

        Args:
            transcript: What was said, once it is known.

        Returns:
            The item's content parts, in the shape the client expects.
        """
        if self._speech_output():
            return [{"type": "output_audio", "transcript": transcript}]
        return [{"type": "output_text", "text": transcript}]

    def _item_view(
        self, response: _Response, status: str, transcript: str = ""
    ) -> JsonMapping:
        """Render the conversation item an answer is written into.

        Args:
            response: The answer in progress.
            status: Status to report.
            transcript: What was said, once it is known.

        Returns:
            The item, in the shape the client expects.
        """
        return _item_body(
            _Item(
                response.item_id,
                "assistant",
                self._assistant_content(transcript),
                status=status,
            )
        )

    def _response_view(
        self, response: _Response, status: str, transcript: str = ""
    ) -> JsonMapping:
        """Render one answer.

        Every field upstream always sends is present, whatever its value: a
        client validating the frame refuses one whose model declares a field
        without a default, and the answer never reaches its pipeline.

        Args:
            response: The answer.
            status: Status to report.
            transcript: What was said, once it is known.

        Returns:
            The response object, in the shape the client expects.
        """
        totals = self._metering.totals
        input_tokens = totals.input_speech_tokens + totals.input_text_tokens
        output_tokens = totals.output_speech_tokens + totals.output_text_tokens
        audio = self._config.audio.output
        return {
            "id": response.id,
            "object": "realtime.response",
            "status": status,
            "status_details": _status_details(status),
            "conversation_id": self._conversation_id,
            "output_modalities": ["audio"] if self._speech_output() else ["text"],
            "max_output_tokens": _reported_token_cap(self._config),
            "audio": {
                "output": {
                    "format": audio.format.model_dump(mode="json", exclude_none=True),
                    "voice": audio.voice,
                }
            },
            # Always null: a response carries no metadata to attach any to.
            "metadata": None,
            "output": [self._item_view(response, status, transcript)]
            if response.started
            else [],
            "usage": {
                "total_tokens": totals.total_tokens or input_tokens + output_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_token_details": {
                    "text_tokens": totals.input_text_tokens,
                    "audio_tokens": totals.input_speech_tokens,
                    "cached_tokens": 0,
                },
                "output_token_details": {
                    "text_tokens": totals.output_text_tokens,
                    "audio_tokens": totals.output_speech_tokens,
                },
            },
        }

    def _session_view(self) -> JsonMapping:
        """Render the effective session configuration.

        Returns:
            The session object, in the shape the client expects.
        """
        view: JsonMapping = self._config.model_dump(mode="json", exclude_none=True)
        view["id"] = self._session_id
        view["object"] = "realtime.session"
        view["model"] = self._model_id
        return view

    async def _send_event(self, event: JsonMapping) -> None:
        """Send one server event.

        Args:
            event: The event body, which is given its own identifier here.
        """
        if self._websocket.client_state is not WebSocketState.CONNECTED:
            return
        await self._websocket.send_text(
            to_json_str({"event_id": f"event_{webuuid()}", **event})
        )

    async def _error(self, kind: str, message: str, code: str | None = None) -> None:
        """Report a non-fatal error to the client.

        Args:
            kind: Error type, in the upstream vocabulary.
            message: What the caller can act on.
            code: Machine-readable code, when one applies.
        """
        log_error_details(message, level="warning")
        await self._send_event(
            {
                "type": "error",
                "error": {
                    "type": kind,
                    "code": code,
                    "message": message,
                    "param": None,
                    "event_id": None,
                },
            }
        )

    async def _fail(self, exception: ApiError) -> None:
        """Report a fatal error and end the session.

        Args:
            exception: What went wrong; only its own message reaches the client.
        """
        kind = "invalid_request_error" if exception.status < 500 else "server_error"
        code = exception.code
        await self._error(kind, exception.args[0], code)
        self._closing = (ERROR_CLOSE_CODE, f"{kind}.{code}" if code else kind)

    async def _close(self, code: int, reason: str) -> None:
        """Close the socket, tolerating one already closed by the client.

        Args:
            code: WebSocket close code.
            reason: Close reason.
        """
        if self._websocket.client_state is not WebSocketState.CONNECTED:
            return
        with suppress(Exception):
            async with async_timeout(_CLOSE_TIMEOUT):
                await self._websocket.close(code=code, reason=reason)


async def _finish_reader(task: Task[None]) -> None:
    """Let a backend reader end on its closed stream, cancelling as last resort.

    A closed stream ends the reader on its own; the timeout bounds the wait
    when it does not, and only then is the reader cancelled mid-read.

    Args:
        task: The reader task.
    """
    with suppress(CancelledError, Exception):
        async with async_timeout(_BACKEND_STOP_TIMEOUT):
            await task
    task.cancel()
    with suppress(CancelledError, Exception):
        await task


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Merge *update* into *base*, one nesting level at a time.

    A top-level merge would drop every sibling of the field a client updated:
    sending ``audio.input.turn_detection`` alone must not reset the voice the
    session was configured with.

    Args:
        base: The configuration in force, as a mapping.
        update: What the client sent.

    Returns:
        The merged mapping.
    """
    merged = dict(base)
    for key, value in update.items():
        current = merged.get(key)
        merged[key] = (
            _deep_merge(current, value)
            if isinstance(current, dict) and isinstance(value, dict)
            else value
        )
    return merged


def _instructions(config: SessionConfig) -> str:
    """Return the system instructions a session opens with.

    Args:
        config: The session configuration.

    Returns:
        The caller's instructions, or the default ones for its kind of session.
    """
    if isinstance(config, TranscriptionSessionConfig):
        return _TRANSCRIPTION_INSTRUCTIONS
    return config.instructions or _DEFAULT_INSTRUCTIONS


async def serve_realtime_session(websocket: WebSocket, model: str | None) -> None:
    """Serve one Realtime WebSocket, from the upgrade to the close frame.

    Everything the HTTP middleware would have done is done here instead: the
    request log entry, the per-request header configuration, the usage scope and
    the scheduled cleanups. A failure that would have been a status code becomes
    a terminal ``error`` event, because the upgrade has already succeeded by the
    time anything can fail -- which is also how the upstream API answers a
    rejected credential.

    Args:
        websocket: The connection being opened.
        model: Model named on the query string, if any.
    """
    subprotocol = next(
        (
            offered
            for offered in _offered_subprotocols(websocket)
            if offered == _SUBPROTOCOL
        ),
        None,
    )
    with log_request_event(websocket) as log:
        CLEANUPS.set([])
        reset_current_input_files()
        try:
            await websocket.accept(
                subprotocol=subprotocol, headers=[(b"x-request-id", log["id"].encode())]
            )
            log["status_code"] = 101
            await _open_session(websocket, model)
        except WebSocketDisconnect:
            log["status_code"] = 499
        except ApiError as exception:
            log["status_code"] = exception.status
            await _refuse(websocket, exception)
        except Exception as exception:  # noqa: BLE001
            # No error middleware runs on a WebSocket: an escape reports nothing.
            log["status_code"] = 500
            log["level"] = "critical"
            log.setdefault("error_detail", []).append(
                "\n".join(format_exception(exception))
            )
            await _refuse(websocket, ApiError(_UNEXPECTED_ERROR, status=500))
        finally:
            run_cleanups_detached(log["id"])


async def _open_session(websocket: WebSocket, model: str | None) -> None:
    """Authenticate, configure and run one accepted connection.

    Args:
        websocket: The accepted connection.
        model: Model named on the query string, if any.

    Raises:
        ApiError: The credential was refused, the deployment is stopping, the
            model cannot serve a live conversation, the model asked for is not
            the one the credential was issued for, or a request header is not
            valid.
    """
    credential = websocket_credential(websocket)
    secret = read_client_secret(credential) if credential else None
    minted = secret is not None
    config: SessionConfig
    if secret is None:
        await verify_websocket_credentials(credential, websocket.scope)
        config = RealtimeSessionConfig()
    else:
        config = secret.session
        # Cleared per connection: a session must not inherit another's identity.
        PRINCIPAL.set(None)
        TENANT.set(None)
        if secret.tenant_key_id is not None:
            # The mint was tenant-authorized; the session keeps the tenant's
            # scopes and drops with the key, so revocation reaches it too.
            TENANT.set(await resume_tenant(secret.tenant_key_id))
            enforce_tenant_endpoint_scope(websocket.scope)
    if _SHUTTING_DOWN:
        message = "The server is shutting down. Reconnect to start a new session."
        raise ApiError(message, status=503)
    if not minted:
        # Ephemeral secrets are client-held: these headers need the deployment's key.
        set_guardrail_configuration(websocket.headers)
        set_performance_configuration(websocket.headers)
        set_mantle_project(websocket.headers)
    locked = minted and not SETTINGS.realtime_allow_session_override
    requested = model or config.model
    if locked and config.model:
        if model and model != config.model:
            message = (
                "The 'model' query parameter is not the model this credential "
                "was issued for."
            )
            raise ApiError(message)
        requested = config.model
    if not requested:
        message = "The 'model' query parameter is required to open a realtime session."
        raise ApiError(message)
    model_id = (
        await validate_model(
            requested,
            input_modality="SPEECH",
            output_modality="SPEECH",
            route="openai_realtime",
        )
    ).id
    realtime_model = get_realtime_model(model_id)
    if not isinstance(realtime_model, RealtimeModelBase):  # pragma: no cover
        message = "This model cannot serve a live conversation."
        raise ApiError(message, status=404)
    await RealtimeSession(
        websocket, realtime_model, model_id, config, locked=locked
    ).run()


async def _refuse(websocket: WebSocket, exception: ApiError) -> None:
    """Report a fatal error on an accepted connection and close it.

    Args:
        websocket: The accepted connection.
        exception: What went wrong; only its own message reaches the client.
    """
    if websocket.client_state is not WebSocketState.CONNECTED:
        return
    kind = "invalid_request_error" if exception.status < 500 else "server_error"
    code = exception.code or ("invalid_api_key" if exception.status == 401 else None)
    log_error_details(exception.args[0], status=exception.status)
    with suppress(Exception):
        async with async_timeout(_CLOSE_TIMEOUT):
            await _refuse_events(websocket, kind, code, exception)


async def _refuse_events(
    websocket: WebSocket, kind: str, code: str | None, exception: ApiError
) -> None:
    """Send the terminal error event and the close frame.

    Args:
        websocket: The accepted connection.
        kind: Error type, in the upstream vocabulary.
        code: Machine-readable code, when one applies.
        exception: What went wrong.
    """
    await websocket.send_text(
        to_json_str(
            {
                "event_id": f"event_{webuuid()}",
                "type": "error",
                "error": {
                    "type": kind,
                    "code": code,
                    "message": exception.args[0],
                    "param": exception.param,
                    "event_id": None,
                },
            }
        )
    )
    await websocket.close(
        code=ERROR_CLOSE_CODE, reason=f"{kind}.{code}" if code else kind
    )


def _status_details(status: str) -> JsonMapping | None:
    """Return why an answer did not run to its end.

    Args:
        status: Status the answer is reported with.

    Returns:
        None for an answer still running or completed, as upstream sends it,
        and what ended it otherwise.
    """
    if status in _CLEAN_RESPONSE_STATUSES:
        return None
    return {"type": status, "reason": _RESPONSE_END_REASONS.get(status)}


def _reported_token_cap(config: SessionConfig) -> int | str:
    """Return the cap one answer may use, as a response reports it.

    Args:
        config: The session configuration.

    Returns:
        The numeric cap, or ``inf`` when the caller left the answer unbounded.
    """
    cap = _max_output_tokens(config)
    return "inf" if cap is None else cap


def _max_output_tokens(config: SessionConfig) -> int | None:
    """Return the cap one answer may use, when the caller set a numeric one.

    Args:
        config: The session configuration.

    Returns:
        The cap, or None when the caller left it unbounded.
    """
    if isinstance(config, TranscriptionSessionConfig):
        return None
    value = config.max_output_tokens
    return value if isinstance(value, int) else None


def sample_rate(media_type: str) -> int:
    """Return the sample rate a client audio format is defined at.

    Args:
        media_type: The format's media type.

    Returns:
        The rate in hertz.
    """
    return FORMAT_SAMPLE_RATES.get(media_type, PCM_SAMPLE_RATE)
