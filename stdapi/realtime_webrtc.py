"""WebRTC call transport for the Realtime API.

``POST /v1/realtime/calls`` trades an SDP offer for an answer, and the media
path negotiated by that exchange terminates here: ICE, DTLS-SRTP and Opus are
handled by aiortc in this process, events ride the ``oai-events`` data channel
in the same JSON vocabulary as the WebSocket, and audio rides the media tracks
instead of base64 events. Everything behind the transport -- the session, the
backend conversation, guardrails, usage -- is the same machinery the WebSocket
uses.

This module imports the ``webrtc`` optional dependencies at import time, so it
is only imported once ``realtime_webrtc_enabled`` proved them installed. The
call registry is per-process: control requests must reach the instance that
answered the SDP, which a deployment of more than one instance cannot
guarantee behind a load balancer.
"""

from asyncio import (
    Event,
    Queue,
    QueueFull,
    create_task,
    get_running_loop,
    sleep,
    wait_for,
)
from contextlib import suppress
from fractions import Fraction
from ipaddress import ip_address
from traceback import format_exception
from typing import TYPE_CHECKING, Any, Final, NamedTuple

from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av.audio.frame import AudioFrame
from av.audio.resampler import AudioResampler
from pybase64 import b64decode, b64encode
from starlette.websockets import WebSocketState

from stdapi import realtime
from stdapi.api_errors import ApiError
from stdapi.cleanup import CLEANUPS, drain_tasks, run_cleanups_detached
from stdapi.config import SETTINGS
from stdapi.input_file import reset_current_input_files
from stdapi.models.realtime import RealtimeModelBase
from stdapi.monitoring import TENANT, log_error_details, log_request_event
from stdapi.realtime import _MAX_EVENT_BYTES, RealtimeSession, sample_rate
from stdapi.types.openai_realtime import AudioFormat
from stdapi.utils import to_json_str, webuuid

if TYPE_CHECKING:
    from asyncio import Task

    from aiortc import RTCDataChannel
    from fastapi import Request, WebSocket

    from stdapi.types import JsonMapping
    from stdapi.types.openai_realtime import SessionConfig

#: Prefix every call identifier carries, as upstream mints them.
CALL_ID_PREFIX: Final = "rtc_"

#: Data channel label the events ride, fixed by the upstream protocol.
_DATA_CHANNEL_LABEL: Final = "oai-events"

#: Seconds of audio each outgoing media frame carries.
_FRAME_SECONDS: Final = 0.02

#: Milliseconds of caller audio batched into one synthesized append event.
_INPUT_CHUNK_MS: Final = 100

#: Events buffered for a data channel that has not opened yet, before giving up.
_MAX_PENDING_EVENTS: Final = 1024

#: Client messages queued for the session before the call is dropped.
_MAX_QUEUED_MESSAGES: Final = 1024

#: Seconds of not-yet-spoken speech buffered before the oldest is dropped.
_MAX_BUFFERED_SECONDS: Final = 480

#: Media sections an SDP offer may carry before it is refused unparsed.
_MAX_OFFER_MEDIA_SECTIONS: Final = 16

#: SDP attribute prefix of one ICE candidate the caller offers.
_CANDIDATE_ATTRIBUTE: Final = "a=candidate:"

#: Bytes queued on the data channel before the session's sends wait for it.
_MAX_CHANNEL_BUFFERED_BYTES: Final = 1024 * 1024

#: Seconds a full data channel is given to drain before the call is dropped.
_CHANNEL_DRAIN_SECONDS: Final = 30.0

#: Seconds one wait on a full data channel lasts before the call is rechecked.
_CHANNEL_DRAIN_POLL: Final = 1.0

#: Calls this instance holds, keyed by call identifier.
_CALLS: Final[dict[str, RealtimeCall]] = {}

#: Tasks serving a call after its creating request returned.
_CALL_TASKS: Final[set[Task[None]]] = set()

#: The ASGI message a transport delivers once its peer connection is gone.
_DISCONNECT: Final = {"type": "websocket.disconnect"}


class InvalidOfferError(ApiError):
    """The SDP offer could not be parsed, in the upstream error shape."""

    code = "invalid_offer"


class RealtimeCall(NamedTuple):
    """One call held by this instance.

    Attributes:
        transport: The media transport terminating the call.
        session: The realtime session serving it.
        tenant_key_id: Key ID of the tenant that opened the call, or None when
            it was opened under the deployment's own credentials.
    """

    transport: WebRTCCallTransport
    session: RealtimeSession
    tenant_key_id: str | None


def get_call(call_id: str) -> RealtimeCall:
    """Return the call *call_id* names, which the caller must own.

    A call opened under a tenant API key is addressable by that tenant and by
    the deployment's own credentials; another tenant is answered the same 404
    as an unknown call, so the registry is not enumerable.

    Args:
        call_id: Identifier of the call, from the creation's Location header.

    Returns:
        The call.

    Raises:
        ApiError: No call under that identifier is held by this instance, or
            the caller's tenant is not the one that opened it.
    """
    call = _CALLS.get(call_id)
    if (
        call is not None
        and (tenant := TENANT.get()) is not None
        and tenant.key_id != call.tenant_key_id
    ):
        log_error_details(
            f"Tenant API key '{tenant.key_id}' addressed call "
            f"'{call_id}', which it did not open: refused as unknown."
        )
        call = None
    if call is None:
        log_error_details(
            f"No call '{call_id}' is held by this instance. The call may have "
            "ended -- or, in a deployment of more than one instance, this "
            "request reached an instance other than the one that answered "
            "the SDP offer, which is where call control must be routed.",
            level="warning",
        )
        message = f"No call '{call_id}' is active."
        raise ApiError(message, status=404)
    return call


def _ice_servers() -> list[RTCIceServer]:
    """Build the ICE servers the deployment is configured to use.

    Returns:
        The STUN and TURN servers, possibly empty.
    """
    servers = []
    if (stun := SETTINGS.realtime_webrtc_stun_server) is not None:
        servers.append(RTCIceServer(urls=[stun]))
    if (turn := SETTINGS.realtime_webrtc_turn_server) is not None and (
        password := SETTINGS.realtime_webrtc_turn_password
    ) is not None:
        servers.append(
            RTCIceServer(
                urls=[turn],
                username=SETTINGS.realtime_webrtc_turn_username,
                credential=password.get_secret_value(),
            )
        )
    return servers


def _may_probe(address: str, *, allow_private: bool) -> bool:
    """Whether the gateway may send ICE checks to *address*.

    Args:
        address: Connection address of one offered candidate.
        allow_private: Whether the deployment opted into candidates that are
            not globally routable.

    Returns:
        True when the address may be probed.
    """
    try:
        parsed = ip_address(address)
    except ValueError:
        # A hostname or an mDNS ".local" name: resolving one is itself a
        # lookup this process would make on an untrusted caller's behalf.
        return False
    return parsed.is_global or allow_private


def _screen_candidates(offer_sdp: str) -> str:
    """Drop the offered ICE candidates the gateway must not probe.

    The offer names the addresses this process sends STUN checks to, so an
    unscreened one turns the gateway into a probe of the network it runs in.
    Only globally routable literals survive, unless the deployment opted into
    private ones for callers that share its network.

    Args:
        offer_sdp: The offer, as the caller sent it.

    Returns:
        The offer with the refused candidate attributes removed.

    Raises:
        InvalidOfferError: The offer carried candidates and none survived,
            which aiortc could not have connected either way.
    """
    allow_private = SETTINGS.realtime_webrtc_allow_private_candidates
    kept: list[str] = []
    offered = refused = 0
    for line in offer_sdp.splitlines(keepends=True):
        attribute = line.strip()
        if not attribute.startswith(_CANDIDATE_ATTRIBUTE):
            kept.append(line)
            continue
        offered += 1
        fields = attribute[len(_CANDIDATE_ATTRIBUTE) :].split()
        if len(fields) > 4 and _may_probe(fields[4], allow_private=allow_private):
            kept.append(line)
        else:
            refused += 1
    if offered and offered == refused:
        log_error_details(
            f"Every one of the {offered} ICE candidates the SDP offer carries "
            "names an address this gateway may not probe. Set "
            "realtime_webrtc_allow_private_candidates when callers "
            "legitimately share the deployment's network.",
            level="warning",
        )
        message = "The SDP offer carries no ICE candidate this gateway can reach."
        raise InvalidOfferError(message)
    return "".join(kept)


class _OutgoingTrack(MediaStreamTrack):
    """The model's speech as a paced media track.

    Frames are emitted at wall-clock rate whether or not speech is buffered:
    an answering machine that goes quiet, not a stream that stalls. The model
    produces faster than realtime, so the buffer holds what has not been
    spoken yet and a barge-in clears it.
    """

    kind = "audio"

    def __init__(self, rate: int) -> None:
        """Start with nothing buffered.

        Args:
            rate: Sample rate of the buffered PCM, in hertz; aiortc's Opus
                encoder resamples whatever rate the frames carry.
        """
        super().__init__()
        self._rate = rate
        self._samples = int(_FRAME_SECONDS * rate)
        self._limit = _MAX_BUFFERED_SECONDS * rate * 2
        self._buffer = bytearray()
        self._timestamp = 0
        self._start: float | None = None

    def write(self, pcm: bytes) -> None:
        """Queue speech to be sent, keeping the newest past the buffer cap.

        Args:
            pcm: 16-bit mono samples at the track's rate.
        """
        self._buffer.extend(pcm)
        if (excess := len(self._buffer) - self._limit) > 0:
            del self._buffer[:excess]

    def clear(self) -> bool:
        """Drop the speech not yet sent, as a barge-in does.

        Returns:
            True when speech was still waiting to be sent.
        """
        sounding = bool(self._buffer)
        del self._buffer[:]
        return sounding

    async def recv(self) -> AudioFrame:
        """Emit the next 20 ms frame, of buffered speech or of silence.

        Returns:
            The frame, paced to wall-clock time.

        Raises:
            MediaStreamError: The track was stopped.
        """
        if self.readyState != "live":
            raise MediaStreamError
        loop_time = get_running_loop().time
        if self._start is None:
            self._start = loop_time()
        else:
            self._timestamp += self._samples
            wait = self._start + (self._timestamp / self._rate) - loop_time()
            if wait > 0:
                await sleep(wait)
        frame = AudioFrame(format="s16", layout="mono", samples=self._samples)
        plane = frame.planes[0]
        data = bytearray(plane.buffer_size)
        take = min(len(self._buffer), len(data))
        if take:
            data[:take] = self._buffer[:take]
            del self._buffer[:take]
        plane.update(bytes(data))
        frame.pts = self._timestamp
        frame.sample_rate = self._rate
        frame.time_base = Fraction(1, self._rate)
        return frame


class WebRTCCallTransport:
    """One WebRTC peer connection, as the session's client transport.

    Client events arrive from the data channel and from any attached sideband
    WebSocket; the caller's audio arrives from the media track and is
    synthesized into the same ``input_audio_buffer.append`` events a WebSocket
    client sends, so the session serves both transports identically. The
    model's audio events are intercepted and fed to the outgoing track instead
    of the channel, as upstream does.
    """

    __slots__ = (
        "_channel",
        "_closed",
        "_input_rate",
        "_pc",
        "_pending",
        "_queue",
        "_sidebands",
        "_tasks",
        "_track",
        "call_id",
    )

    def __init__(self, call_id: str, input_rate: int, output_rate: int) -> None:
        """Prepare the transport of one call.

        Args:
            call_id: Identifier the call is addressed by.
            input_rate: Sample rate the session reads the caller's speech at.
            output_rate: Sample rate the session writes the model's speech at.
        """
        self.call_id = call_id
        self._input_rate = input_rate
        self._track = _OutgoingTrack(output_rate)
        self._pc: RTCPeerConnection | None = None
        self._channel: RTCDataChannel | None = None
        self._queue: Queue[dict[str, Any]] = Queue(maxsize=_MAX_QUEUED_MESSAGES)
        self._pending: list[str] = []
        self._sidebands: list[WebSocket] = []
        self._tasks: set[Task[None]] = set()
        self._closed = False

    @property
    def connected(self) -> bool:
        """Whether events can still be sent to the caller."""
        return not self._closed

    async def answer(self, offer_sdp: str) -> str:
        """Answer the caller's SDP offer and start the media path.

        Args:
            offer_sdp: The offer, as the caller generated it.

        Returns:
            The SDP answer, ICE candidates included.

        Raises:
            InvalidOfferError: The offer could not be parsed or answered, or
                it named no ICE candidate the gateway may reach.
        """
        # Screened before the synchronous parse: the offer is untrusted, and
        # negotiating a media section is not O(1).
        media = [line for line in offer_sdp.splitlines() if line.startswith("m=")]
        if len(media) > _MAX_OFFER_MEDIA_SECTIONS:
            message = "The SDP offer carries too many media sections."
            raise InvalidOfferError(message)
        if not any(line.startswith("m=audio") for line in media):
            message = "The SDP offer must carry an audio media section."
            raise InvalidOfferError(message)
        offer_sdp = _screen_candidates(offer_sdp)
        pc = self._pc = RTCPeerConnection(RTCConfiguration(iceServers=_ice_servers()))
        pc.on("datachannel", self._on_datachannel)
        pc.on("track", self._on_track)
        pc.on("connectionstatechange", self._on_connectionstatechange)
        try:
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=offer_sdp, type="offer")
            )
            pc.addTrack(self._track)
            await pc.setLocalDescription(await pc.createAnswer())
        # Anything the stack refuses -- parse, negotiate, gather -- is one
        # invalid offer to the caller; the detail stays in the operator's log.
        except Exception as error:  # noqa: BLE001
            with suppress(Exception):
                await pc.close()
            log_error_details(f"The SDP offer could not be answered: {error!r}")
            message = "The SDP offer could not be parsed or answered."
            raise InvalidOfferError(message) from None
        return pc.localDescription.sdp

    def _on_datachannel(self, channel: RTCDataChannel) -> None:
        """Adopt the caller's event channel and flush what awaited it.

        Args:
            channel: The channel the caller opened.
        """
        if channel.label != _DATA_CHANNEL_LABEL or self._closed:
            return
        self._channel = channel
        # aiortc signals the drain only on the way down through this
        # threshold, which is the one the sends wait on.
        channel.bufferedAmountLowThreshold = _MAX_CHANNEL_BUFFERED_BYTES
        channel.on("message", self._on_channel_message)
        pending, self._pending = self._pending, []
        for text in pending:
            channel.send(text)

    def _on_channel_message(self, message: str | bytes) -> None:
        """Deliver one caller event to the session.

        Args:
            message: The event, as the channel carried it.
        """
        if self._closed:
            return
        if len(message) > _MAX_EVENT_BYTES:
            # Refused before it is held: the channel has no backpressure.
            self._disconnect()
            return
        self._deliver({"type": "websocket.receive", "text": message})

    def _deliver(self, message: dict[str, Any]) -> None:
        """Queue one client message, ending the call when the bound is hit.

        Args:
            message: The ``websocket.receive`` message to deliver.
        """
        try:
            self._queue.put_nowait(message)
        except QueueFull:
            self._disconnect()

    def _on_track(self, track: MediaStreamTrack) -> None:
        """Start reading the caller's audio.

        Args:
            track: The media track the caller sends.
        """
        if track.kind != "audio":
            return
        task = create_task(self._read_track(track))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _on_connectionstatechange(self) -> None:
        """End the session once the peer connection is gone."""
        if self._pc is not None and self._pc.connectionState in {"failed", "closed"}:
            self._disconnect()

    def _disconnect(self) -> None:
        """Report the caller gone, which is what ends the session."""
        if not self._closed:
            self._closed = True
            # A full queue may drop the sentinel: receive() checks the flag
            # before every read, so the session still learns.
            with suppress(QueueFull):
                self._queue.put_nowait(_DISCONNECT)

    async def _read_track(self, track: MediaStreamTrack) -> None:
        """Feed the caller's audio to the session as synthesized append events.

        The decoder hands out 48 kHz frames; they are resampled to the
        session's input rate and batched so the event rate stays low.

        Args:
            track: The media track the caller sends.
        """
        resampler = AudioResampler(format="s16", layout="mono", rate=self._input_rate)
        chunk_bytes = self._input_rate * _INPUT_CHUNK_MS // 1000 * 2
        pending = bytearray()
        while not self._closed:
            try:
                frame: AudioFrame = await track.recv()  # type: ignore[assignment]
            except MediaStreamError:
                break
            for resampled in resampler.resample(frame):
                pending.extend(bytes(resampled.planes[0])[: resampled.samples * 2])
            while len(pending) >= chunk_bytes:
                self._push_audio(bytes(pending[:chunk_bytes]))
                del pending[:chunk_bytes]
        if pending:
            self._push_audio(bytes(pending))

    def _push_audio(self, pcm: bytes) -> None:
        """Deliver one chunk of the caller's audio to the session.

        Args:
            pcm: 16-bit mono samples at the session's input rate.
        """
        if self._closed:
            return
        event = to_json_str(
            {"type": "input_audio_buffer.append", "audio": b64encode(pcm).decode()}
        )
        self._deliver({"type": "websocket.receive", "text": event})

    async def receive(self) -> dict[str, Any]:
        """Return the next client message, whichever path delivered it.

        Returns:
            A ``websocket.receive`` message, or ``websocket.disconnect`` once
            the peer connection is gone.
        """
        if self._closed:
            return _DISCONNECT
        return await self._queue.get()

    async def send_event(self, event: JsonMapping) -> None:
        """Send one server event, routing audio to the media track.

        Args:
            event: The event body, identifier included.
        """
        kind = event.get("type")
        if kind == "response.output_audio.delta":
            # Audio rides the media track; upstream sends no audio events.
            if isinstance(delta := event.get("delta"), str):
                with suppress(ValueError):
                    self._track.write(b64decode(delta, validate=True))
            return
        barged_in = False
        if kind == "input_audio_buffer.speech_started":
            # Playback is the gateway's here, so a caller who speaks over the
            # tail of a finished answer must stop hearing it, and upstream
            # reports that drop as output_audio_buffer.cleared.
            barged_in = self._track.clear()
        elif kind == "output_audio_buffer.cleared" or (
            kind == "response.done"
            and isinstance(response := event.get("response"), dict)
            and response.get("status") != "completed"
        ):
            # An answer that did not run to its end must also stop sounding: a
            # barge-in reports "incomplete", an explicit cancel "cancelled".
            self._track.clear()
        await self._write(to_json_str(event))
        if barged_in:
            await self._write(
                to_json_str(
                    {
                        "event_id": f"event_{webuuid()}",
                        "type": "output_audio_buffer.cleared",
                    }
                )
            )

    async def _write(self, text: str) -> None:
        """Write one serialized event to the caller and every sideband.

        Args:
            text: The event, as it goes on the wire.
        """
        if (channel := self._channel) is not None:
            if await self._drain_channel(channel):
                try:
                    channel.send(text)
                except Exception:  # noqa: BLE001 - a closing channel ends the session instead
                    self._disconnect()
        elif len(self._pending) < _MAX_PENDING_EVENTS:
            self._pending.append(text)
        else:
            self._disconnect()
        for websocket in tuple(self._sidebands):
            try:
                await websocket.send_text(text)
            except Exception:  # noqa: BLE001 - a dead sideband must not end the call
                self._forget_sideband(websocket)

    async def _drain_channel(self, channel: RTCDataChannel) -> bool:
        """Wait until the data channel has room for one more event.

        aiortc's ``send`` never blocks and queues without limit, so this is
        where the session meets the backpressure the WebSocket transport gets
        from TCP; a caller that never drains the queue is dropped rather than
        held in memory.

        Args:
            channel: The channel the events are written to.

        Returns:
            True when the channel has room, False once the call ended -- for
            not reading what it was sent, or for any other reason while the
            wait was in progress.
        """
        if channel.bufferedAmount <= _MAX_CHANNEL_BUFFERED_BYTES:
            return True
        loop_time = get_running_loop().time
        deadline = loop_time() + _CHANNEL_DRAIN_SECONDS
        drained = Event()
        channel.on("bufferedamountlow", drained.set)
        try:
            while True:
                drained.clear()
                if (
                    self._closed
                    or channel.readyState != "open"
                    or channel.bufferedAmount <= _MAX_CHANNEL_BUFFERED_BYTES
                ):
                    return not self._closed
                if (remaining := deadline - loop_time()) <= 0:
                    log_error_details(
                        f"Call '{self.call_id}' left "
                        f"{channel.bufferedAmount} bytes of server events "
                        "unread: dropped rather than queueing more.",
                        level="warning",
                    )
                    self._disconnect()
                    return False
                # Sliced: a hangup while the wait is in progress must not
                # hold the session for the whole drain allowance.
                with suppress(TimeoutError):
                    await wait_for(drained.wait(), min(remaining, _CHANNEL_DRAIN_POLL))
        finally:
            channel.remove_listener("bufferedamountlow", drained.set)

    async def serve_sideband(self, websocket: WebSocket) -> None:
        """Serve one sideband WebSocket until either side ends it.

        The connection observes the call's server events and may send client
        events into its session, exactly as upstream's sideband does.

        Args:
            websocket: The accepted connection.
        """
        self._sidebands.append(websocket)
        try:
            while not self._closed:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if payload := (message.get("text") or message.get("bytes")):
                    self._deliver({"type": "websocket.receive", "text": payload})
        finally:
            self._forget_sideband(websocket)

    def _forget_sideband(self, websocket: WebSocket) -> None:
        """Stop mirroring events to one sideband connection.

        Args:
            websocket: The connection to forget.
        """
        with suppress(ValueError):
            self._sidebands.remove(websocket)

    async def close(self, code: int, reason: str) -> None:
        """Tear the whole call down, idempotently.

        Args:
            code: WebSocket close code, forwarded to the sidebands.
            reason: Close reason, forwarded to the sidebands.
        """
        self._disconnect()
        self._closed = True
        for task in tuple(self._tasks):
            task.cancel()
        self._track.stop()
        if self._channel is not None:
            with suppress(Exception):
                self._channel.close()
        if self._pc is not None:
            with suppress(Exception):
                await self._pc.close()
        for websocket in tuple(self._sidebands):
            self._forget_sideband(websocket)
            with suppress(Exception):
                if websocket.client_state is WebSocketState.CONNECTED:
                    await websocket.close(code=code, reason=reason)


async def open_call(
    request: Request,
    model: str | None,
    config: SessionConfig,
    offer_sdp: str,
    *,
    locked: bool = False,
) -> tuple[str, str]:
    """Answer one SDP offer and start serving the call it opens.

    The session outlives this request: it runs as a background task holding
    its own request log scope, flushing usage per answer like the WebSocket
    does, and is registered so ``hangup`` and the sideband can address it.

    Args:
        request: The creating request, whose identity and headers the call
            keeps for its whole lifetime.
        model: Model named on the query string, if any.
        config: Session configuration the call opens with.
        offer_sdp: The caller's SDP offer.
        locked: Whether the client may not change what its credential pinned.

    Returns:
        The call identifier and the SDP answer.

    Raises:
        ApiError: No model was named, the model cannot serve a live
            conversation, or the offer could not be answered.
    """
    requested = model or config.model
    if locked and config.model:
        # The secret pinned its model: a differing query parameter is refused
        # rather than silently escaping the pin, as on the WebSocket.
        if model and model != config.model:
            message = (
                "The 'model' query parameter is not the model this credential "
                "was issued for."
            )
            raise ApiError(message)
        requested = config.model
    if not requested:
        message = (
            "The 'model' query parameter or the session configuration must "
            "name the model answering the call."
        )
        raise ApiError(message)
    # The media is Opus either way: the session's own formats stay PCM.
    config = config.model_copy(deep=True)
    config.audio.input.format = AudioFormat()
    config.audio.output.format = AudioFormat()
    call_id = f"{CALL_ID_PREFIX}{webuuid()}"
    transport = WebRTCCallTransport(
        call_id,
        sample_rate(config.audio.input.format.type),
        sample_rate(config.audio.output.format.type),
    )
    # Resolved through the realtime module so both transports share one seam,
    # concurrently with the SDP answer: neither await feeds the other.
    model_task = create_task(
        realtime.validate_model(  # type: ignore[attr-defined]
            requested,
            input_modality="SPEECH",
            output_modality="SPEECH",
            route="openai_realtime",
        )
    )
    try:
        answer_sdp = await transport.answer(offer_sdp)
        model_id = (await model_task).id
    except BaseException:
        model_task.cancel()
        with suppress(BaseException):
            await model_task
        with suppress(Exception):
            await transport.close(1000, "")
        raise
    realtime_model = realtime.get_realtime_model(model_id)  # type: ignore[attr-defined]
    if not isinstance(realtime_model, RealtimeModelBase):  # pragma: no cover
        with suppress(Exception):
            await transport.close(1000, "")
        message = "This model cannot serve a live conversation."
        raise ApiError(message, status=404)
    session = RealtimeSession(
        transport, realtime_model, model_id, config, locked=locked, fixed_formats=True
    )
    tenant = TENANT.get()
    _CALLS[call_id] = RealtimeCall(
        transport, session, tenant.key_id if tenant is not None else None
    )
    # The task copies this request's context: principal, tenant and headers.
    task = create_task(_serve_call(call_id, transport, session, request))
    _CALL_TASKS.add(task)
    task.add_done_callback(_CALL_TASKS.discard)
    return call_id, answer_sdp


async def _serve_call(
    call_id: str,
    transport: WebRTCCallTransport,
    session: RealtimeSession,
    request: Request,
) -> None:
    """Serve one call from the SDP answer to the teardown.

    Args:
        call_id: Identifier the call is registered under.
        transport: The call's media transport.
        session: The session serving it.
        request: The creating request, logged as the session's own scope.
    """
    with log_request_event(request) as log:
        log["status_code"] = 200
        CLEANUPS.set([])
        reset_current_input_files()
        try:
            await session.run()
        except Exception as exception:  # noqa: BLE001 - nothing above this reports a detached task
            log["status_code"] = 500
            log["level"] = "critical"
            log.setdefault("error_detail", []).append(
                "\n".join(format_exception(exception))
            )
        finally:
            _CALLS.pop(call_id, None)
            # The session only closes a connected transport; a vanished peer
            # still needs its connection and tasks torn down.
            with suppress(Exception):
                await transport.close(1000, "")
            run_cleanups_detached(log["id"])


async def serve_sideband(websocket: WebSocket, call_id: str) -> None:
    """Attach one authenticated WebSocket to a call as its sideband.

    Args:
        websocket: The accepted connection.
        call_id: Identifier of the call to attach to.

    Raises:
        ApiError: No call under that identifier is held by this instance, or
            the caller's tenant did not open it.
    """
    await get_call(call_id).transport.serve_sideband(websocket)


def hangup_call(call_id: str) -> None:
    """End one call, as upstream's ``hangup`` verb does.

    Args:
        call_id: Identifier of the call to end.

    Raises:
        ApiError: No call under that identifier is held by this instance, or
            the caller's tenant did not open it.
    """
    get_call(call_id).session.request_close(1000, "client_hangup")


async def drain_realtime_calls(timeout: float) -> int:  # noqa: ASYNC109 -- shared drain contract
    """Await the call tasks still ending after shutdown asked them to close.

    Args:
        timeout: Seconds allowed before the unfinished tasks are cancelled.

    Returns:
        Number of tasks that had not finished at the deadline.
    """
    return await drain_tasks(_CALL_TASKS, timeout)
