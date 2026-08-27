"""WebRTC call transport: SDP exchange, media bridge, call control, routes.

Everything here is offline: aiortc plays both peers over loopback UDP inside
one process, and the model conversation is a scripted fake. That proves the
SDP exchange, the DTLS/SRTP handshake, the data-channel event bridge, the
audio resampling in both directions and the call registry -- and proves
nothing about NAT traversal, security groups, STUN/TURN or a real browser,
which only a deployed gateway exercises.

The upstream contract was probed against api.openai.com on 2026-08-27: raw
``application/sdp`` and ``multipart/form-data`` both answer 201 with the SDP
answer as ``text/plain`` and the call id in ``Location``; a JSON body answers
400 ``unsupported_content_type``; ``hangup`` answers 200.

Ref: https://developers.openai.com/api/docs/guides/realtime-webrtc
     stdapi/realtime_webrtc.py:WebRTCCallTransport
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack, MediaStreamError

from stdapi.aws_bedrock import GUARDRAIL_CONFIG_VAR
from stdapi.aws_bedrock_mantle import MANTLE_PROJECT_VAR
from stdapi.config import SETTINGS
from stdapi.models.realtime import (
    OutputAudio,
    OutputTranscript,
    RealtimeBackendSession,
    RealtimeModelBase,
    ResponseFinished,
    ResponseStarted,
    UsageReport,
)
from stdapi.monitoring import TENANT, Tenant, log_request_event
from stdapi.realtime import _MAX_EVENT_BYTES, mint_client_secret
from stdapi.realtime_webrtc import (
    _CALLS,
    _MAX_CHANNEL_BUFFERED_BYTES,
    _MAX_QUEUED_MESSAGES,
    InvalidOfferError,
    RealtimeCall,
    WebRTCCallTransport,
    _screen_candidates,
    drain_realtime_calls,
    get_call,
    hangup_call,
    open_call,
)
from stdapi.types.openai_realtime import RealtimeSessionConfig
from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator

    from av.audio.frame import AudioFrame
    from starlette.testclient import TestClient

    from stdapi.models.realtime import BackendEvent
    from stdapi.realtime import RealtimeSession

pytestmark = pytest.mark.local

#: Model identifier the fake backend answers for, never resolved against AWS.
_MODEL = "fake.realtime-v1:0"

#: Seconds a loopback negotiation step may take before the test fails.
_STEP_TIMEOUT = 10.0

#: One loud 100 ms answer chunk (24 kHz square wave), audible after Opus.
_LOUD_CHUNK = (b"\x00\x40" * 24 + b"\x00\xc0" * 24) * 50

#: The scripted answer every fake conversation replays after a turn ends.
_ANSWER_SCRIPT: tuple[BackendEvent, ...] = (
    ResponseStarted(),
    OutputTranscript("Sure thing."),
    OutputAudio(_LOUD_CHUNK),
    ResponseFinished(),
)


class _FakeSession(RealtimeBackendSession):
    """A backend that records what it was sent and replays a script."""

    def __init__(self, script: list[BackendEvent]) -> None:
        """Hold *script* to replay once a turn ends.

        Args:
            script: Events to report once the caller ends a turn.
        """
        self._script = script
        self.audio = bytearray()
        self.texts: list[str] = []
        self.ended = 0
        self.region = "us-east-1"
        self.closed = asyncio.Event()

    async def send_audio(self, audio: Any) -> None:  # noqa: ANN401
        """Record one chunk of the caller's speech."""
        self.audio.extend(bytes(audio))

    async def send_text(self, text: str) -> None:
        """Record one written message from the caller."""
        self.texts.append(text)

    async def end_turn(self) -> None:
        """Record that the caller ended a turn."""
        self.ended += 1

    async def events(self) -> AsyncGenerator[BackendEvent]:
        """Replay the script once a turn ended, then wait for the close.

        Yields:
            Each scripted event, in order.
        """
        # Polled: end_turn is a plain call, not an event the reader owns.
        while not self.ended and not self.closed.is_set():  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        for event in self._script:
            yield event
        await self.closed.wait()


class _FakeModel(RealtimeModelBase[Any, Any]):
    """A realtime model whose sessions are scripted fakes."""

    MATCHER = _MODEL
    INPUT_SAMPLE_RATES = frozenset({8000, 16000, 24000})
    OUTPUT_SAMPLE_RATES = frozenset({8000, 16000, 24000})
    DEFAULT_VOICE = "fake"
    MAX_SESSION_SECONDS = 60.0

    script: ClassVar[list[BackendEvent]] = []
    opened: ClassVar[list[_FakeSession]] = []

    def open_session(self, **_: Any) -> Any:  # noqa: ANN401
        """Open one scripted conversation.

        Returns:
            An async context manager yielding the fake session.
        """
        from contextlib import asynccontextmanager  # noqa: PLC0415

        @asynccontextmanager
        async def _session() -> AsyncGenerator[_FakeSession]:
            session = _FakeSession(list(type(self).script))
            type(self).opened.append(session)
            try:
                yield session
            finally:
                session.closed.set()

        return _session()


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_FakeModel]]:
    """Serve every realtime session from :class:`_FakeModel`.

    Yields:
        The fake model class, whose ``script`` the test sets and whose
        ``opened`` sessions it asserts on.
    """
    from stdapi import realtime  # noqa: PLC0415

    class _Details:
        id = _MODEL

    async def _validate_model(model_id: str, **_: Any) -> Any:  # noqa: ANN401
        from stdapi.api_errors import UnsupportedModelError  # noqa: PLC0415
        from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

        if model_id != _MODEL:
            raise UnsupportedModelError(model_id)
        REQUEST_LOG.get()["model_id"] = _MODEL
        return _Details()

    monkeypatch.setattr(realtime, "validate_model", _validate_model)
    monkeypatch.setattr(realtime, "get_realtime_model", lambda _id: _FakeModel(_MODEL))
    _FakeModel.opened = []
    _FakeModel.script = list(_ANSWER_SCRIPT)
    yield _FakeModel
    _FakeModel.opened = []
    _FakeModel.script = []


class _ToneTrack(AudioStreamTrack):
    """A caller's microphone: a 440 Hz tone at 48 kHz, paced like the base."""

    async def recv(self) -> Any:  # noqa: ANN401
        """Emit the next silence frame from the base, filled with a tone.

        Returns:
            The frame.
        """
        frame = cast("AudioFrame", await super().recv())
        samples = frame.samples
        rate = frame.sample_rate
        start = frame.pts or 0
        tone = bytearray()
        for index in range(samples):
            value = int(12000 * math.sin(2 * math.pi * 440 * (start + index) / rate))
            tone += value.to_bytes(2, "little", signed=True)
        frame.planes[0].update(bytes(tone))
        return frame


def _fake_request() -> Any:  # noqa: ANN401
    """Build the creating request the call keeps for its log scope.

    Returns:
        A minimal HTTP request, as the middleware would have seen it.
    """
    from starlette.requests import Request  # noqa: PLC0415

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/realtime/calls",
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
        }
    )


class _Client:
    """The caller's side of one loopback call."""

    def __init__(self) -> None:
        """Prepare the peer, its tone track and its event channel."""
        self.pc = RTCPeerConnection()
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.audio = bytearray()
        self.channel = self.pc.createDataChannel("oai-events")
        self.channel.on("message", self._on_message)
        self.pc.addTrack(_ToneTrack())
        self._reader: asyncio.Task[None] | None = None
        self.pc.on("track", self._on_track)

    def _on_message(self, message: str | bytes) -> None:
        self.events.put_nowait(json.loads(message))

    def _on_track(self, track: Any) -> None:  # noqa: ANN401
        self._reader = asyncio.get_running_loop().create_task(self._read(track))

    async def _read(self, track: Any) -> None:  # noqa: ANN401
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                return
            self.audio.extend(bytes(frame.planes[0])[: frame.samples * 4])

    async def offer(self) -> str:
        """Create the SDP offer.

        Returns:
            The offer.
        """
        await self.pc.setLocalDescription(await self.pc.createOffer())
        return self.pc.localDescription.sdp

    async def connect(self, answer_sdp: str) -> None:
        """Apply the answer and wait for the event channel to open."""
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type="answer")
        )
        while self.channel.readyState != "open":  # noqa: ASYNC110 - readyState has no waiter
            await asyncio.sleep(0.01)

    async def next_event(self, kind: str) -> dict[str, Any]:
        """Return the next event of *kind*, recording nothing it skips.

        Args:
            kind: The event type awaited.

        Returns:
            The event.
        """
        async with asyncio.timeout(_STEP_TIMEOUT):
            while True:
                event = await self.events.get()
                if event.get("type") == kind:
                    return event

    async def close(self) -> None:
        """Tear the caller down."""
        if self._reader is not None:
            self._reader.cancel()
        await self.pc.close()


async def _open_loopback_call(
    config: RealtimeSessionConfig | None = None,
) -> tuple[_Client, str]:
    """Open one call between an in-process caller and the gateway side.

    Args:
        config: Session configuration the call opens with.

    Returns:
        The connected caller and the call identifier.
    """
    client = _Client()
    offer = await client.offer()
    with log_request_event(_fake_request()):
        call_id, answer = await open_call(
            _fake_request(), _MODEL, config or RealtimeSessionConfig(), offer
        )
    assert call_id.startswith("rtc_")
    assert "m=audio" in answer
    await client.connect(answer)
    return client, call_id


async def _drain_call_tasks() -> None:
    """Wait for every background call task the test started."""
    assert await drain_realtime_calls(_STEP_TIMEOUT) == 0


@pytest.fixture
def allow_private_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept the private host candidates an in-process caller offers.

    The screen the gateway applies to an untrusted offer would drop every
    candidate a loopback caller has, so these tests deploy as an on-network
    one does.
    """
    monkeypatch.setattr(SETTINGS, "realtime_webrtc_allow_private_candidates", True)


@pytest.mark.usefixtures("fake_backend", "allow_private_candidates")
class TestLoopbackCall:
    """The full media path, aiortc to aiortc over loopback UDP.

    Ref: https://developers.openai.com/api/docs/guides/realtime-webrtc
         stdapi/realtime_webrtc.py:open_call
    """

    async def test_session_events_ride_the_data_channel(self) -> None:
        """The session opens on the channel exactly as it does on the socket."""
        client, call_id = await _open_loopback_call()
        try:
            created = await client.next_event("session.created")
            assert created["session"]["model"] == _MODEL
        finally:
            hangup_call(call_id)
            await _drain_call_tasks()
            await client.close()

    async def test_audio_flows_both_ways_and_stays_off_the_channel(
        self, fake_backend: type[_FakeModel]
    ) -> None:
        """Caller audio reaches the backend resampled; answers ride the track.

        The caller's 48 kHz Opus tone must arrive at the fake backend as
        24 kHz PCM through the synthesized append events, the scripted answer
        must come back as audible track audio, and no
        ``response.output_audio.delta`` event may appear on the data channel.
        """
        started = asyncio.get_running_loop().time()
        client, call_id = await _open_loopback_call()
        try:
            await client.next_event("session.created")
            # Let the tone flow long enough for a few append events.
            await asyncio.sleep(1.0)
            client.channel.send(json.dumps({"type": "input_audio_buffer.commit"}))
            done = await client.next_event("response.done")
            assert done["response"]["status"] == "completed"
            transcript = done["response"]["output"][0]["content"][0]["transcript"]
            assert transcript == "Sure thing."
            # The answer's audio keeps playing after response.done: the track
            # paces at wall-clock rate while the model produced it instantly.
            await asyncio.sleep(1.0)
            elapsed = asyncio.get_running_loop().time() - started
            audio = bytes(fake_backend.opened[0].audio)
            # Resampled to the session's 24 kHz, the byte count tracks wall
            # time at 48000 B/s: a tone left at its 48 kHz capture rate would
            # arrive at double that and break the upper bound.
            assert len(audio) >= 48000 * 0.4
            assert len(audio) <= 48000 * elapsed * 1.25
            # The tone survives Opus and the resampling audibly: a peak well
            # above the noise floor, read as the 16-bit samples it carries.
            peak = max(
                abs(int.from_bytes(audio[index : index + 2], "little", signed=True))
                for index in range(0, len(audio) - 1, 2)
            )
            assert peak > 4000
            # The loud scripted answer survives Opus audibly.
            received = client.audio
            assert len(received) > 0
            peak = max(
                abs(int.from_bytes(received[index : index + 2], "little", signed=True))
                for index in range(0, min(len(received), 200000), 2)
            )
            assert peak > 1000
            drained: list[dict[str, Any]] = []
            while not client.events.empty():
                drained.append(client.events.get_nowait())
            assert not [
                event
                for event in drained
                if event.get("type") == "response.output_audio.delta"
            ]
        finally:
            hangup_call(call_id)
            await _drain_call_tasks()
            await client.close()

    async def test_hangup_tears_the_call_down(self) -> None:
        """Hangup ends the session, closes the peer and empties the registry."""
        client, call_id = await _open_loopback_call()
        try:
            await client.next_event("session.created")
            hangup_call(call_id)
            await _drain_call_tasks()
            assert call_id not in _CALLS
            async with asyncio.timeout(_STEP_TIMEOUT):
                # Polled: aiortc exposes no waiter for the caller-side state.
                while self._alive(client):  # noqa: ASYNC110
                    await asyncio.sleep(0.05)
        finally:
            await client.close()

    @staticmethod
    def _alive(client: _Client) -> bool:
        """Whether the caller still sees the connection up."""
        return client.pc.connectionState not in {"closed", "failed"}


class TestLockedSecret:
    """The model pin a locked ephemeral secret places on a call.

    Ref: stdapi/realtime_webrtc.py:open_call
    """

    @pytest.mark.usefixtures("fake_backend")
    async def test_a_differing_model_query_is_refused(self) -> None:
        """A locked secret's model wins over the query string, as on the socket."""
        from stdapi.api_errors import ApiError  # noqa: PLC0415

        with log_request_event(_fake_request()), pytest.raises(ApiError) as excinfo:
            await open_call(
                _fake_request(),
                "another.model-v1:0",
                RealtimeSessionConfig(model=_MODEL),
                "v=0",
                locked=True,
            )
        assert "issued for" in str(excinfo.value)


class TestCallRegistry:
    """The per-instance call registry and its upstream-shaped failure mode.

    Ref: stdapi/realtime_webrtc.py:get_call
    """

    def test_an_unknown_call_answers_an_upstream_shaped_404(self) -> None:
        """The 404 does not describe the deployment's instance topology."""
        from stdapi.api_errors import ApiError  # noqa: PLC0415

        with pytest.raises(ApiError) as excinfo:
            get_call("rtc_unknown")
        assert excinfo.value.status == 404
        assert "No call 'rtc_unknown' is active" in str(excinfo.value)
        assert "instance" not in str(excinfo.value)


class _StubCloseSession:
    """Just enough of a session for the call-control surface."""

    def __init__(self) -> None:
        """Start with no close requested."""
        self.closed: list[tuple[int, str]] = []

    def request_close(self, code: int, reason: str) -> None:
        """Record the close request."""
        self.closed.append((code, reason))


class TestCallOwnership:
    """A call is addressable only by the credential domain that opened it.

    Ref: stdapi/realtime_webrtc.py:get_call
    """

    @pytest.fixture
    async def owned_call(
        self,
    ) -> AsyncGenerator[tuple[WebRTCCallTransport, _StubCloseSession]]:
        """Register one call opened under tenant ``tk_owner``."""
        transport = WebRTCCallTransport("rtc_owned", 24000, 24000)
        session = _StubCloseSession()
        _CALLS["rtc_owned"] = RealtimeCall(
            transport, cast("RealtimeSession", session), "tk_owner"
        )
        yield transport, session
        _CALLS.pop("rtc_owned", None)
        await transport.close(1000, "")

    async def test_a_foreign_tenant_is_answered_the_unknown_call_404(
        self, owned_call: tuple[WebRTCCallTransport, _StubCloseSession]
    ) -> None:
        """Another tenant cannot end, observe or drive the call."""
        del owned_call
        from stdapi.api_errors import ApiError  # noqa: PLC0415

        token = TENANT.set(Tenant(key_id="tk_other", name="other"))
        try:
            with pytest.raises(ApiError) as excinfo:
                hangup_call("rtc_owned")
        finally:
            TENANT.reset(token)
        assert excinfo.value.status == 404
        # The same answer as an unknown call: the registry is not enumerable.
        assert "No call 'rtc_owned' is active" in str(excinfo.value)

    async def test_the_owning_tenant_and_the_deployment_reach_the_call(
        self, owned_call: tuple[WebRTCCallTransport, _StubCloseSession]
    ) -> None:
        """The opening tenant and the deployment's own credentials both do."""
        transport, session = owned_call
        token = TENANT.set(Tenant(key_id="tk_owner", name="owner"))
        try:
            assert get_call("rtc_owned").transport is transport
        finally:
            TENANT.reset(token)
        # No tenant in context is the deployment's own credential domain.
        hangup_call("rtc_owned")
        assert session.closed == [(1000, "client_hangup")]

    async def test_a_tenant_cannot_reach_a_deployment_owned_call(self) -> None:
        """A tenant key does not address a call the deployment opened."""
        from stdapi.api_errors import ApiError  # noqa: PLC0415

        transport = WebRTCCallTransport("rtc_deploy", 24000, 24000)
        _CALLS["rtc_deploy"] = RealtimeCall(
            transport, cast("RealtimeSession", _StubCloseSession()), None
        )
        token = TENANT.set(Tenant(key_id="tk_any", name="any"))
        try:
            with pytest.raises(ApiError) as excinfo:
                get_call("rtc_deploy")
        finally:
            TENANT.reset(token)
            _CALLS.pop("rtc_deploy", None)
            await transport.close(1000, "")
        assert excinfo.value.status == 404


class TestSideband:
    """The sideband WebSocket surface of a call transport.

    Ref: stdapi/realtime_webrtc.py:WebRTCCallTransport.serve_sideband
    """

    async def test_events_mirror_and_control_flows_in(self) -> None:
        """A sideband sees server events and injects client events."""
        transport = WebRTCCallTransport("rtc_test", 24000, 24000)
        socket = _StubSocket()
        task = asyncio.get_running_loop().create_task(
            transport.serve_sideband(socket)  # type: ignore[arg-type]
        )
        await asyncio.sleep(0)
        await transport.send_event({"type": "session.created", "session": {}})
        assert json.loads(socket.sent[0])["type"] == "session.created"
        socket.incoming.put_nowait(
            {"type": "websocket.receive", "text": '{"type": "response.create"}'}
        )
        message = await transport.receive()
        assert message["text"] == '{"type": "response.create"}'
        socket.incoming.put_nowait({"type": "websocket.disconnect"})
        async with asyncio.timeout(_STEP_TIMEOUT):
            await task
        await transport.close(1000, "")

    async def test_close_reaches_the_sideband(self) -> None:
        """Closing the call closes an attached sideband with it."""
        transport = WebRTCCallTransport("rtc_test", 24000, 24000)
        socket = _StubSocket()
        task = asyncio.get_running_loop().create_task(
            transport.serve_sideband(socket)  # type: ignore[arg-type]
        )
        await asyncio.sleep(0)
        await transport.close(1000, "hangup")
        assert socket.closed == (1000, "hangup")
        task.cancel()


class _StubSocket:
    """Just enough of a WebSocket for the sideband surface."""

    def __init__(self) -> None:
        """Start connected, with nothing exchanged."""
        from starlette.websockets import WebSocketState  # noqa: PLC0415

        self.client_state = WebSocketState.CONNECTED
        self.incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sent: list[str] = []
        self.closed: tuple[int, str] | None = None

    async def receive(self) -> dict[str, Any]:
        """Return the next queued message."""
        return await self.incoming.get()

    async def send_text(self, text: str) -> None:
        """Record one mirrored event."""
        self.sent.append(text)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Record the close."""
        from starlette.websockets import WebSocketState  # noqa: PLC0415

        self.closed = (code, reason)
        self.client_state = WebSocketState.DISCONNECTED


@pytest.fixture
def webrtc_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the WebRTC transport for the routes under test."""
    monkeypatch.setattr(SETTINGS, "realtime_webrtc_enabled", True)


@pytest.fixture
def stub_open_call(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the media stack under the route with a recording stub.

    Returns:
        The record of the one call the stub answered.
    """
    import stdapi.realtime_webrtc  # noqa: PLC0415

    record: dict[str, Any] = {}

    async def _open_call(
        request: Any,  # noqa: ANN401, ARG001
        model: str | None,
        config: Any,  # noqa: ANN401
        offer_sdp: str,
        *,
        locked: bool = False,
    ) -> tuple[str, str]:
        record.update(
            model=model,
            config=config,
            offer=offer_sdp,
            locked=locked,
            # Captured at call time: what the route left in the request
            # context is what the whole call would run under.
            tenant=TENANT.get(),
            guardrail=GUARDRAIL_CONFIG_VAR.get(None),
            mantle_project=MANTLE_PROJECT_VAR.get(),
        )
        return "rtc_stub", "v=0\r\nfake-answer"

    monkeypatch.setattr(stdapi.realtime_webrtc, "open_call", _open_call)
    return record


class TestCallCreationRoute:
    """POST /v1/realtime/calls: encodings, auth and the disabled default.

    Ref: https://developers.openai.com/api/docs/guides/realtime-webrtc
         stdapi/routes/openai_realtime.py:create_realtime_call
    """

    def test_disabled_by_default_answers_404(
        self, app_client: TestClient, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Without the setting the endpoint answers 404, as before the feature.

        The caller's message stays generic; the setting that enables the
        transport is named in the operator's log only.
        """
        response = app_client.post(
            "/v1/realtime/calls",
            content="v=0",
            headers={"Content-Type": "application/sdp"},
        )
        assert response.status_code == 404
        message = response.json()["error"]["message"]
        assert "not available on this deployment" in message
        assert "realtime_webrtc_enabled" not in message
        assert "realtime_webrtc_enabled" in capsys.readouterr().out

    @pytest.mark.usefixtures("webrtc_enabled")
    def test_a_raw_sdp_offer_answers_201_with_location(
        self, app_client: TestClient, stub_open_call: dict[str, Any]
    ) -> None:
        """The application/sdp encoding, probed against upstream 2026-08-27."""
        response = app_client.post(
            "/v1/realtime/calls?model=fake.realtime-v1:0",
            content="v=0\r\noffer",
            headers={"Content-Type": "application/sdp"},
        )
        assert response.status_code == 201
        assert response.headers["location"] == "/v1/realtime/calls/rtc_stub"
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text == "v=0\r\nfake-answer"
        assert stub_open_call["offer"] == "v=0\r\noffer"
        assert stub_open_call["model"] == "fake.realtime-v1:0"

    @pytest.mark.usefixtures("webrtc_enabled")
    def test_a_multipart_offer_carries_a_session(
        self, app_client: TestClient, stub_open_call: dict[str, Any]
    ) -> None:
        """The multipart encoding, probed against upstream 2026-08-27."""
        response = app_client.post(
            "/v1/realtime/calls",
            files={
                "sdp": (None, "v=0\r\noffer"),
                "session": (
                    None,
                    '{"type": "realtime", "model": "fake.realtime-v1:0"}',
                ),
            },
        )
        assert response.status_code == 201
        assert stub_open_call["config"].model == "fake.realtime-v1:0"

    @pytest.mark.usefixtures("webrtc_enabled", "stub_open_call")
    def test_a_json_body_is_refused_like_upstream(self, app_client: TestClient) -> None:
        """Upstream refuses JSON despite the SDK typing it; so does this."""
        response = app_client.post(
            "/v1/realtime/calls", json={"sdp": "v=0", "session": {}}
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "unsupported_content_type"

    @pytest.mark.usefixtures("webrtc_enabled", "stub_open_call")
    def test_an_invalid_session_field_is_refused(self, app_client: TestClient) -> None:
        """A session field that is not a session configuration answers 400."""
        response = app_client.post(
            "/v1/realtime/calls",
            files={"sdp": (None, "v=0"), "session": (None, "not-json")},
        )
        assert response.status_code == 400
        assert "'session' form field" in response.json()["error"]["message"]

    @pytest.mark.usefixtures("webrtc_enabled", "stub_open_call")
    def test_a_multipart_body_without_an_offer_is_refused(
        self, app_client: TestClient
    ) -> None:
        """A multipart body must carry the offer in its sdp field."""
        response = app_client.post(
            "/v1/realtime/calls", files={"session": (None, "{}")}
        )
        assert response.status_code == 400
        assert "'sdp' form field" in response.json()["error"]["message"]

    @pytest.mark.usefixtures("webrtc_enabled", "stub_open_call")
    def test_an_oversized_offer_is_refused_unread(self, app_client: TestClient) -> None:
        """An offer past the size bound answers 413 before any parsing."""
        response = app_client.post(
            "/v1/realtime/calls",
            content="v=0\r\n" + "a" * (64 * 1024),
            headers={"Content-Type": "application/sdp"},
        )
        assert response.status_code == 413
        assert "too large" in response.json()["error"]["message"]

    @pytest.mark.usefixtures("webrtc_enabled")
    @pytest.mark.parametrize("override", [True, False])
    def test_an_ephemeral_secret_carries_its_session(
        self,
        app_client: TestClient,
        stub_open_call: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        override: bool,
    ) -> None:
        """A browser-held secret authenticates the offer and pins its session.

        The session is locked against overrides exactly when the deployment
        disabled ``realtime_allow_session_override``.
        """
        monkeypatch.setattr(SETTINGS, "realtime_allow_session_override", override)
        secret, _ = mint_client_secret(
            RealtimeSessionConfig(model=_MODEL, instructions="Be brief."), 60
        )
        response = app_client.post(
            "/v1/realtime/calls",
            content="v=0\r\noffer",
            headers={
                "Content-Type": "application/sdp",
                "Authorization": f"Bearer {secret}",
            },
        )
        assert response.status_code == 201
        assert stub_open_call["config"].instructions == "Be brief."
        assert stub_open_call["config"].model == _MODEL
        assert stub_open_call["locked"] is (not override)

    @pytest.mark.usefixtures("webrtc_enabled", "stub_open_call")
    def test_a_locked_secret_refuses_a_replacement_session(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With overrides disabled, a multipart session field answers 400."""
        monkeypatch.setattr(SETTINGS, "realtime_allow_session_override", False)
        secret, _ = mint_client_secret(RealtimeSessionConfig(model=_MODEL), 60)
        response = app_client.post(
            "/v1/realtime/calls",
            files={"sdp": (None, "v=0"), "session": (None, "{}")},
            headers={"Authorization": f"Bearer {secret}"},
        )
        assert response.status_code == 400
        assert "cannot be replaced" in response.json()["error"]["message"]

    @pytest.mark.usefixtures("webrtc_enabled")
    def test_a_secret_cannot_carry_the_deployment_key_headers(
        self,
        app_client: TestClient,
        stub_open_call: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The per-request override headers are reset on a secret-held offer.

        With the override allowed and no deployment default configured, a
        guardrail or Mantle-project header sent beside the secret must not
        reach the call: those headers need the deployment's own key.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", True)
        secret, _ = mint_client_secret(RealtimeSessionConfig(model=_MODEL), 60)
        response = app_client.post(
            "/v1/realtime/calls",
            content="v=0\r\noffer",
            headers={
                "Content-Type": "application/sdp",
                "Authorization": f"Bearer {secret}",
                "X-Amzn-Bedrock-GuardrailIdentifier": "attacker-guardrail",
                "X-Amzn-Bedrock-GuardrailVersion": "1",
                "OpenAI-Project": "proj_attacker",
            },
        )
        assert response.status_code == 201
        assert stub_open_call["guardrail"] is None
        assert stub_open_call["mantle_project"] == ""

    @pytest.mark.usefixtures("webrtc_enabled")
    def test_a_tenant_minted_secret_resumes_its_tenant(
        self,
        app_client: TestClient,
        stub_open_call: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The call runs under the tenant whose key authorized the mint."""
        tenant = Tenant(key_id="tk_owner0000000001", name="owner")
        token = TENANT.set(tenant)
        try:
            secret, _ = mint_client_secret(RealtimeSessionConfig(model=_MODEL), 60)
        finally:
            TENANT.reset(token)
        resumed: list[str] = []

        async def _resume(key_id: str) -> Tenant:
            resumed.append(key_id)
            return tenant

        monkeypatch.setattr("stdapi.routes.openai_realtime.resume_tenant", _resume)
        response = app_client.post(
            "/v1/realtime/calls",
            content="v=0\r\noffer",
            headers={
                "Content-Type": "application/sdp",
                "Authorization": f"Bearer {secret}",
            },
        )
        assert response.status_code == 201
        assert resumed == [tenant.key_id]
        assert stub_open_call["tenant"] is tenant

    @pytest.mark.usefixtures("webrtc_enabled", "stub_open_call")
    def test_an_out_of_scope_tenant_secret_is_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The resumed tenant's endpoint scope applies to the calls route."""
        restricted = Tenant(
            key_id="tk_scoped000000001", name="scoped", endpoints_allow=()
        )
        token = TENANT.set(restricted)
        try:
            secret, _ = mint_client_secret(RealtimeSessionConfig(model=_MODEL), 60)
        finally:
            TENANT.reset(token)

        async def _resume(key_id: str) -> Tenant:
            del key_id
            return restricted

        monkeypatch.setattr("stdapi.routes.openai_realtime.resume_tenant", _resume)
        response = app_client.post(
            "/v1/realtime/calls",
            content="v=0\r\noffer",
            headers={
                "Content-Type": "application/sdp",
                "Authorization": f"Bearer {secret}",
            },
        )
        assert response.status_code == 401


class TestCallCreationAuthentication:
    """POST /v1/realtime/calls requires a verified credential.

    The route authenticates in-body -- an ephemeral secret or the standard
    credentials -- so it is allowlisted out of the suite-wide route guard;
    these tests pin its credential path directly, against an app whose API
    key check is armed.

    Ref: stdapi/routes/openai_realtime.py:_authenticate_call
    """

    @pytest.mark.usefixtures("webrtc_enabled", "stub_open_call")
    def test_no_credential_is_refused(self, enforced_auth_client: TestClient) -> None:
        """An anonymous offer answers 401."""
        response = enforced_auth_client.post(
            "/v1/realtime/calls",
            content="v=0",
            headers={"Content-Type": "application/sdp"},
        )
        assert response.status_code == 401

    @pytest.mark.usefixtures("webrtc_enabled", "stub_open_call")
    def test_a_wrong_bearer_is_refused(self, enforced_auth_client: TestClient) -> None:
        """A credential that is not the deployment's key answers 401."""
        response = enforced_auth_client.post(
            "/v1/realtime/calls",
            content="v=0",
            headers={
                "Content-Type": "application/sdp",
                "Authorization": "Bearer wrong-key",
            },
        )
        assert response.status_code == 401

    @pytest.mark.usefixtures("webrtc_enabled", "stub_open_call")
    def test_a_tampered_secret_is_refused(
        self, enforced_auth_client: TestClient
    ) -> None:
        """A forged signature is neither a secret nor a deployment key."""
        secret, _ = mint_client_secret(RealtimeSessionConfig(model=_MODEL), 60)
        # Eight signature characters flipped: a single trailing base64 char
        # can decode to the same bytes through the discarded padding bits.
        suffix = "ABABABAB" if not secret.endswith("ABABABAB") else "BABABABA"
        tampered = secret[:-8] + suffix
        response = enforced_auth_client.post(
            "/v1/realtime/calls",
            content="v=0",
            headers={
                "Content-Type": "application/sdp",
                "Authorization": f"Bearer {tampered}",
            },
        )
        assert response.status_code == 401

    @pytest.mark.usefixtures("webrtc_enabled", "stub_open_call")
    def test_an_expired_secret_is_refused(
        self, enforced_auth_client: TestClient
    ) -> None:
        """A secret past its expiry answers 401 rather than falling through."""
        secret, _ = mint_client_secret(RealtimeSessionConfig(model=_MODEL), -10)
        response = enforced_auth_client.post(
            "/v1/realtime/calls",
            content="v=0",
            headers={
                "Content-Type": "application/sdp",
                "Authorization": f"Bearer {secret}",
            },
        )
        assert response.status_code == 401

    @pytest.mark.usefixtures("webrtc_enabled")
    def test_a_valid_secret_authenticates(
        self, enforced_auth_client: TestClient, stub_open_call: dict[str, Any]
    ) -> None:
        """The positive control: an unexpired secret opens the call."""
        secret, _ = mint_client_secret(RealtimeSessionConfig(model=_MODEL), 60)
        response = enforced_auth_client.post(
            "/v1/realtime/calls",
            content="v=0\r\noffer",
            headers={
                "Content-Type": "application/sdp",
                "Authorization": f"Bearer {secret}",
            },
        )
        assert response.status_code == 201
        assert stub_open_call["config"].model == _MODEL

    @pytest.mark.usefixtures("stub_open_call")
    def test_the_credential_is_checked_before_the_feature_gate(
        self, enforced_auth_client: TestClient
    ) -> None:
        """With WebRTC disabled, an anonymous caller still reads 401, not 404.

        Answering the feature's 404 first would make the endpoint an
        unauthenticated oracle for the deployment's configuration.
        """
        response = enforced_auth_client.post(
            "/v1/realtime/calls",
            content="v=0",
            headers={"Content-Type": "application/sdp"},
        )
        assert response.status_code == 401


@pytest.mark.usefixtures("webrtc_enabled", "fake_backend")
class TestInvalidOffer:
    """Hostile or malformed SDP offers, through the real media stack.

    Ref: stdapi/realtime_webrtc.py:WebRTCCallTransport.answer
    """

    @staticmethod
    def _post_offer(app_client: TestClient, offer: str) -> Any:  # noqa: ANN401
        """Post *offer* through the route with the fake model named."""
        return app_client.post(
            f"/v1/realtime/calls?model={_MODEL}",
            content=offer,
            headers={"Content-Type": "application/sdp"},
        )

    @staticmethod
    def _assert_refused_cleanly(response: Any) -> None:  # noqa: ANN401
        """Assert a 400 ``invalid_offer`` that leaks nothing and leaks no call."""
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "invalid_offer"
        assert _CALLS == {}
        assert asyncio.run(drain_realtime_calls(_STEP_TIMEOUT)) == 0

    def test_an_empty_offer_is_refused(self, app_client: TestClient) -> None:
        """An empty body is one invalid offer, not a 500."""
        self._assert_refused_cleanly(self._post_offer(app_client, ""))

    def test_an_offer_without_audio_is_refused(self, app_client: TestClient) -> None:
        """An offer negotiating no audio section cannot become a call."""
        response = self._post_offer(app_client, "v=0\r\no=- 0 0 IN IP4 0.0.0.0")
        self._assert_refused_cleanly(response)
        assert "audio media section" in response.json()["error"]["message"]

    def test_a_garbage_offer_answers_a_fixed_message(
        self, app_client: TestClient
    ) -> None:
        """The parse failure detail stays out of the caller's error body."""
        response = self._post_offer(
            app_client, "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
        )
        self._assert_refused_cleanly(response)
        message = response.json()["error"]["message"]
        assert message == "The SDP offer could not be parsed or answered."

    def test_too_many_media_sections_are_refused_unparsed(
        self, app_client: TestClient
    ) -> None:
        """A section-flooded offer is refused before aiortc parses it."""
        offer = "v=0\r\n" + "m=audio 9 UDP/TLS/RTP/SAVPF 0\r\n" * 17
        response = self._post_offer(app_client, offer)
        self._assert_refused_cleanly(response)
        assert "too many media sections" in response.json()["error"]["message"]

    def test_an_offer_of_only_internal_candidates_is_refused(
        self, app_client: TestClient
    ) -> None:
        """An offer naming only addresses inside the network is refused."""
        offer = (
            "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"
            "a=candidate:1 1 udp 2130706431 10.0.0.5 4000 typ host\r\n"
            "a=candidate:2 1 udp 2130706431 127.0.0.1 4001 typ host\r\n"
        )
        response = self._post_offer(app_client, offer)
        self._assert_refused_cleanly(response)
        assert "no ICE candidate" in response.json()["error"]["message"]


class TestOfferCandidateScreen:
    """Which offered ICE candidates the gateway is willing to probe.

    An offer names the addresses this process sends STUN checks to, and the
    caller holding an ephemeral secret is untrusted, so the screen decides
    what a call may make the gateway reach for.

    Ref: stdapi/realtime_webrtc.py:_screen_candidates
         https://developers.openai.com/api/docs/guides/realtime-webrtc
    """

    @staticmethod
    def _offer(*addresses: str) -> str:
        """Build an audio offer whose candidates name *addresses*.

        Args:
            addresses: Connection address of each candidate, in order.

        Returns:
            The offer.
        """
        lines = ["v=0", "m=audio 9 UDP/TLS/RTP/SAVPF 111"]
        lines += [
            f"a=candidate:{index} 1 udp 2130706431 {address} {4000 + index} typ host"
            for index, address in enumerate(addresses)
        ]
        return "\r\n".join([*lines, ""])

    @staticmethod
    def _kept(screened: str) -> list[str]:
        """Return the connection address of every surviving candidate.

        Args:
            screened: The screened offer.

        Returns:
            One address per kept candidate.
        """
        return [
            line.split()[4]
            for line in screened.splitlines()
            if line.startswith("a=candidate:")
        ]

    def test_only_globally_routable_candidates_survive(self) -> None:
        """Private, shared, loopback and link-local addresses are dropped."""
        screened = _screen_candidates(
            self._offer(
                "10.0.0.5",
                "192.168.1.9",
                "172.16.0.1",
                "100.64.0.1",
                "127.0.0.1",
                "169.254.169.254",
                "fe80::1",
                "8.8.8.8",
                "2001:4860:4860::8888",
            )
        )
        assert self._kept(screened) == ["8.8.8.8", "2001:4860:4860::8888"]

    def test_hostname_and_mdns_candidates_are_always_dropped(self) -> None:
        """A name would be resolved on the gateway's own network."""
        offer = self._offer("abc-123.local", "internal.example.com", "8.8.8.8")
        for allowed in (False, True):
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(
                    SETTINGS, "realtime_webrtc_allow_private_candidates", allowed
                )
                assert self._kept(_screen_candidates(offer)) == ["8.8.8.8"]

    @pytest.mark.usefixtures("allow_private_candidates")
    def test_the_opt_in_keeps_the_candidates_an_on_network_caller_has(self) -> None:
        """A same-network deployment accepts what its callers can offer."""
        screened = _screen_candidates(self._offer("10.0.0.5", "127.0.0.1"))
        assert self._kept(screened) == ["10.0.0.5", "127.0.0.1"]

    def test_an_offer_carrying_no_candidate_is_left_alone(self) -> None:
        """Nothing to screen is not a refusal: aiortc still answers it."""
        offer = self._offer()
        assert _screen_candidates(offer) == offer

    def test_a_malformed_candidate_line_is_dropped(self) -> None:
        """A candidate attribute with no address names nothing to probe."""
        offer = "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=candidate:1 1 udp\r\n"
        with pytest.raises(InvalidOfferError):
            _screen_candidates(offer)


class TestCallControlRoutes:
    """Hangup and the SIP-only verbs.

    Ref: https://developers.openai.com/api/docs/guides/realtime-webrtc
         stdapi/routes/openai_realtime.py:hangup_realtime_call
    """

    @pytest.mark.usefixtures("webrtc_enabled")
    def test_hangup_answers_200(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hangup reaches the registry and answers an empty 200."""
        import stdapi.realtime_webrtc  # noqa: PLC0415

        ended: list[str] = []
        monkeypatch.setattr(stdapi.realtime_webrtc, "hangup_call", ended.append)
        response = app_client.post("/v1/realtime/calls/rtc_stub/hangup")
        assert response.status_code == 200
        assert ended == ["rtc_stub"]

    @pytest.mark.usefixtures("webrtc_enabled")
    def test_hangup_of_an_unknown_call_answers_404(
        self, app_client: TestClient
    ) -> None:
        """An unknown call answers an upstream-shaped 404."""
        response = app_client.post("/v1/realtime/calls/rtc_gone/hangup")
        assert response.status_code == 404
        message = response.json()["error"]["message"]
        assert "No call 'rtc_gone' is active" in message
        assert "instance" not in message

    @pytest.mark.parametrize("verb", ["accept", "reject", "refer"])
    def test_sip_verbs_are_refused_with_the_documented_answer(
        self, app_client: TestClient, verb: str
    ) -> None:
        """The SIP-only verbs answer a clean 400 naming what serves telephony.

        Refused whether or not WebRTC is enabled: the refusal is the
        documented limitation, not a configuration gap.
        """
        response = app_client.post(f"/v1/realtime/calls/rtc_x/{verb}")
        assert response.status_code == 400
        message = response.json()["error"]["message"]
        assert "SIP" in message
        assert "WebSocket" in message


class TestSidebandRoute:
    """GET /v1/realtime?call_id=... dispatching to a held call.

    Ref: stdapi/realtime.py:_open_sideband
    """

    @pytest.mark.usefixtures("webrtc_enabled")
    def test_an_unknown_call_is_refused_on_the_socket(
        self, app_client: TestClient
    ) -> None:
        """The sideband of a call this instance does not hold answers an error."""
        with app_client.websocket_connect("/v1/realtime?call_id=rtc_gone") as websocket:
            event = json.loads(websocket.receive_text())
            assert event["type"] == "error"
            message = event["error"]["message"]
            assert "No call 'rtc_gone' is active" in message
            assert "instance" not in message

    def test_disabled_webrtc_stays_generic_on_the_socket(
        self, app_client: TestClient, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Without the feature, the setting is named in the log, not the error."""
        with app_client.websocket_connect("/v1/realtime?call_id=rtc_gone") as websocket:
            event = json.loads(websocket.receive_text())
            assert event["type"] == "error"
            message = event["error"]["message"]
            assert "not available on this deployment" in message
            assert "realtime_webrtc_enabled" not in message
        assert "realtime_webrtc_enabled" in capsys.readouterr().out


class TestOutgoingTrack:
    """The paced track carrying the model's speech.

    Ref: stdapi/realtime_webrtc.py:_OutgoingTrack
    """

    async def test_the_buffer_keeps_the_newest_speech_past_its_cap(self) -> None:
        """The not-yet-spoken buffer is bounded, dropping the oldest bytes."""
        from stdapi.realtime_webrtc import (  # noqa: PLC0415
            _MAX_BUFFERED_SECONDS,
            _OutgoingTrack,
        )

        track = _OutgoingTrack(8000)
        limit = _MAX_BUFFERED_SECONDS * 8000 * 2
        track.write(b"\x00" * limit)
        track.write(b"\x01" * 2)
        assert len(track._buffer) == limit  # noqa: SLF001
        assert track._buffer[-2:] == b"\x01\x01"  # noqa: SLF001
        track.stop()

    async def test_buffered_speech_is_framed_then_silence_follows(self) -> None:
        """Written PCM comes back frame by frame, then frames go quiet."""
        from stdapi.realtime_webrtc import _OutgoingTrack  # noqa: PLC0415

        track = _OutgoingTrack(24000)
        track.write(b"\x01\x00" * 480)
        first = await track.recv()
        assert first.sample_rate == 24000
        assert bytes(first.planes[0])[:960] == b"\x01\x00" * 480
        second = await track.recv()
        assert bytes(second.planes[0])[:960] == bytes(960)
        track.write(b"\x02\x00" * 480)
        track.clear()
        third = await track.recv()
        assert bytes(third.planes[0])[:960] == bytes(960)
        track.stop()
        with pytest.raises(MediaStreamError):
            await track.recv()


class TestBargeIn:
    """The outgoing track is cleared whenever an answer did not finish.

    Ref: stdapi/realtime_webrtc.py:WebRTCCallTransport.send_event
    """

    @pytest.mark.parametrize("status", ["incomplete", "cancelled"])
    async def test_an_unfinished_answer_stops_sounding(self, status: str) -> None:
        """A barge-in ("incomplete") or a cancel drops the unsent speech."""
        transport = WebRTCCallTransport("rtc_test", 24000, 24000)
        await transport.send_event(
            {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(b"\x01\x00" * 480).decode(),
            }
        )
        assert transport._track._buffer  # noqa: SLF001
        await transport.send_event(
            {"type": "response.done", "response": {"status": status}}
        )
        assert not transport._track._buffer  # noqa: SLF001
        await transport.close(1000, "")

    async def test_a_completed_answer_keeps_playing(self) -> None:
        """A finished answer's backlog keeps draining at wall-clock rate."""
        transport = WebRTCCallTransport("rtc_test", 24000, 24000)
        await transport.send_event(
            {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(b"\x01\x00" * 480).decode(),
            }
        )
        await transport.send_event(
            {"type": "response.done", "response": {"status": "completed"}}
        )
        assert transport._track._buffer  # noqa: SLF001
        await transport.close(1000, "")

    async def test_speech_started_stops_a_finished_answer_still_sounding(self) -> None:
        """The caller talking over the played tail clears it and says so.

        Playback belongs to the gateway on a call, so ``speech_started`` is
        what stops it: the answer already reported ``completed``, and nothing
        else would drop the seconds of speech still queued.

        Ref: https://developers.openai.com/api/docs/guides/realtime-webrtc
        """
        transport = WebRTCCallTransport("rtc_test", 24000, 24000)
        socket = _StubSocket()
        task = asyncio.get_running_loop().create_task(
            transport.serve_sideband(socket)  # type: ignore[arg-type]
        )
        await asyncio.sleep(0)
        await transport.send_event(
            {
                "type": "response.output_audio.delta",
                "delta": base64.b64encode(b"\x01\x00" * 480).decode(),
            }
        )
        await transport.send_event(
            {"type": "response.done", "response": {"status": "completed"}}
        )
        assert transport._track._buffer  # noqa: SLF001
        await transport.send_event(
            {"type": "input_audio_buffer.speech_started", "audio_start_ms": 0}
        )
        assert not transport._track._buffer  # noqa: SLF001
        kinds = [json.loads(text)["type"] for text in socket.sent]
        assert kinds[-2:] == [
            "input_audio_buffer.speech_started",
            "output_audio_buffer.cleared",
        ]
        assert json.loads(socket.sent[-1])["event_id"].startswith("event_")
        task.cancel()
        await transport.close(1000, "")

    async def test_speech_started_over_silence_reports_nothing_cleared(self) -> None:
        """With nothing left to play there is no output buffer to clear."""
        transport = WebRTCCallTransport("rtc_test", 24000, 24000)
        socket = _StubSocket()
        task = asyncio.get_running_loop().create_task(
            transport.serve_sideband(socket)  # type: ignore[arg-type]
        )
        await asyncio.sleep(0)
        await transport.send_event(
            {"type": "input_audio_buffer.speech_started", "audio_start_ms": 0}
        )
        assert [json.loads(text)["type"] for text in socket.sent] == [
            "input_audio_buffer.speech_started"
        ]
        task.cancel()
        await transport.close(1000, "")


class TestTransportBackpressure:
    """The event queue is bounded, and hitting the bound ends the call.

    Ref: stdapi/realtime_webrtc.py:WebRTCCallTransport._on_channel_message
    """

    async def test_an_oversized_channel_message_ends_the_call(self) -> None:
        """A message past the event size cap is refused before it is held."""
        transport = WebRTCCallTransport("rtc_test", 24000, 24000)
        transport._on_channel_message("x" * (_MAX_EVENT_BYTES + 1))  # noqa: SLF001
        assert not transport.connected
        assert (await transport.receive())["type"] == "websocket.disconnect"
        await transport.close(1000, "")

    async def test_a_flooded_event_queue_ends_the_call(self) -> None:
        """A caller outrunning the session's pump is dropped, not buffered."""
        transport = WebRTCCallTransport("rtc_test", 24000, 24000)
        for _ in range(_MAX_QUEUED_MESSAGES):
            transport._on_channel_message("{}")  # noqa: SLF001
        assert transport.connected
        transport._on_channel_message("{}")  # noqa: SLF001
        assert not transport.connected
        assert (await transport.receive())["type"] == "websocket.disconnect"
        await transport.close(1000, "")


class _StubChannel:
    """Just enough of an aiortc data channel to drive the send path.

    ``send`` never blocks in aiortc either: what it queues is only reported
    through ``bufferedAmount``, which this stub lets the test move.
    """

    label = "oai-events"

    def __init__(self, buffered: int = 0) -> None:
        """Start open, with *buffered* bytes already queued.

        Args:
            buffered: Bytes the channel reports as not yet on the wire.
        """
        self.readyState = "open"
        self.bufferedAmount = buffered
        self.bufferedAmountLowThreshold = 0
        self.sent: list[str] = []
        self.handlers: dict[str, list[Any]] = {}

    def on(self, event: str, handler: Any) -> Any:  # noqa: ANN401
        """Register *handler* for *event*, as pyee does.

        Args:
            event: Event name.
            handler: What to call.

        Returns:
            The handler, as pyee returns it.
        """
        self.handlers.setdefault(event, []).append(handler)
        return handler

    def remove_listener(self, event: str, handler: Any) -> None:  # noqa: ANN401
        """Forget *handler*, raising if it was never registered."""
        self.handlers[event].remove(handler)

    def emit(self, event: str) -> None:
        """Fire *event* at everything registered for it."""
        for handler in list(self.handlers.get(event, ())):
            handler()

    def send(self, text: str) -> None:
        """Record one event written to the channel."""
        self.sent.append(text)

    def close(self) -> None:
        """Close the channel, as the transport teardown does."""
        self.readyState = "closed"


class TestChannelBackpressure:
    """Outbound events wait for the data channel instead of piling up.

    Ref: stdapi/realtime_webrtc.py:WebRTCCallTransport._drain_channel
    """

    @staticmethod
    def _adopt(buffered: int) -> tuple[WebRTCCallTransport, _StubChannel]:
        """Open one transport whose channel already holds *buffered* bytes.

        Args:
            buffered: Bytes the channel reports as not yet on the wire.

        Returns:
            The transport and its channel.
        """
        transport = WebRTCCallTransport("rtc_test", 24000, 24000)
        channel = _StubChannel(buffered)
        transport._on_datachannel(cast("Any", channel))  # noqa: SLF001
        return transport, channel

    async def test_the_drain_threshold_is_armed_when_the_channel_opens(self) -> None:
        """Aiortc only signals a drain across the threshold it is given."""
        transport, channel = self._adopt(0)
        assert channel.bufferedAmountLowThreshold == _MAX_CHANNEL_BUFFERED_BYTES
        await transport.send_event({"type": "session.created", "session": {}})
        assert json.loads(channel.sent[0])["type"] == "session.created"
        await transport.close(1000, "")

    async def test_a_full_channel_holds_the_event_until_it_drains(self) -> None:
        """A caller that stops reading stalls the sender, not the memory."""
        transport, channel = self._adopt(_MAX_CHANNEL_BUFFERED_BYTES + 1)
        task = asyncio.get_running_loop().create_task(
            transport.send_event({"type": "session.created", "session": {}})
        )
        async with asyncio.timeout(_STEP_TIMEOUT):
            # Polled: the wait registers its listener on the channel itself.
            while not channel.handlers.get("bufferedamountlow"):  # noqa: ASYNC110
                await asyncio.sleep(0.01)
        assert channel.sent == []
        assert not task.done()
        channel.bufferedAmount = 0
        channel.emit("bufferedamountlow")
        async with asyncio.timeout(_STEP_TIMEOUT):
            await task
        assert json.loads(channel.sent[0])["type"] == "session.created"
        assert transport.connected
        assert not channel.handlers["bufferedamountlow"]
        await transport.close(1000, "")

    async def test_a_hangup_releases_a_send_waiting_on_the_channel(self) -> None:
        """Tearing the call down does not wait out the drain allowance."""
        transport, channel = self._adopt(_MAX_CHANNEL_BUFFERED_BYTES + 1)
        task = asyncio.get_running_loop().create_task(
            transport.send_event({"type": "session.created", "session": {}})
        )
        async with asyncio.timeout(_STEP_TIMEOUT):
            # Polled: the wait registers its listener on the channel itself.
            while not channel.handlers.get("bufferedamountlow"):  # noqa: ASYNC110
                await asyncio.sleep(0.01)
        await transport.close(1000, "hangup")
        async with asyncio.timeout(_STEP_TIMEOUT):
            await task
        assert channel.sent == []

    async def test_a_channel_that_never_drains_ends_the_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bound is a bound: an unread channel drops the call.

        Without it a caller could answer nothing and make the gateway hold
        every event it was ever sent.
        """
        import stdapi.realtime_webrtc  # noqa: PLC0415

        monkeypatch.setattr(stdapi.realtime_webrtc, "_CHANNEL_DRAIN_SECONDS", 0.01)
        transport, channel = self._adopt(_MAX_CHANNEL_BUFFERED_BYTES + 1)
        async with asyncio.timeout(_STEP_TIMEOUT):
            await transport.send_event({"type": "session.created", "session": {}})
        assert channel.sent == []
        assert not transport.connected
        assert (await transport.receive())["type"] == "websocket.disconnect"
        await transport.close(1000, "")


@pytest.mark.usefixtures("fake_backend", "allow_private_candidates")
class TestFixedCallFormats:
    """The PCM formats a call pins cannot be changed by the client.

    Ref: stdapi/realtime.py:RealtimeSession._update_session
    """

    async def test_a_format_change_is_refused_on_a_call(self) -> None:
        """A ``session.update`` naming another audio format answers an error."""
        client, call_id = await _open_loopback_call()
        try:
            await client.next_event("session.created")
            client.channel.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "audio": {"output": {"format": {"type": "audio/pcmu"}}}
                        },
                    }
                )
            )
            error = await client.next_event("error")
            assert "media negotiation" in error["error"]["message"]
        finally:
            hangup_call(call_id)
            await _drain_call_tasks()
            await client.close()

    async def test_the_pinned_format_may_be_echoed_with_its_rate(self) -> None:
        """Echoing ``audio/pcm`` with its explicit rate is not a change.

        The rate a client sends beside the pinned encoding is the one that
        encoding is defined at, and the SDKs send it: refusing it would drop
        the rest of the update with it.

        Ref: https://developers.openai.com/api/docs/guides/realtime-webrtc
             stdapi/realtime.py:RealtimeSession._update_session
        """
        client, call_id = await _open_loopback_call()
        try:
            await client.next_event("session.created")
            client.channel.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "instructions": "Be brief.",
                            "audio": {
                                "input": {
                                    "format": {"type": "audio/pcm", "rate": 24000}
                                },
                                "output": {
                                    "format": {"type": "audio/pcm", "rate": 24000}
                                },
                            },
                        },
                    }
                )
            )
            updated = await client.next_event("session.updated")
            assert updated["session"]["instructions"] == "Be brief."
        finally:
            hangup_call(call_id)
            await _drain_call_tasks()
            await client.close()


@pytest.mark.usefixtures("fake_backend", "allow_private_candidates")
class TestCallBilling:
    """A call bills and logs through its own request scope.

    Ref: stdapi/realtime_webrtc.py:_serve_call
    """

    async def test_an_answer_records_its_usage(
        self, fake_backend: type[_FakeModel], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The answer's usage reaches the request log, as on the WebSocket."""
        fake_backend.script = [
            ResponseStarted(),
            OutputTranscript("Sure thing."),
            OutputAudio(_LOUD_CHUNK),
            UsageReport(
                input_speech_tokens=10,
                input_text_tokens=5,
                output_speech_tokens=20,
                output_text_tokens=8,
                total_tokens=43,
            ),
            ResponseFinished(),
        ]
        client, call_id = await _open_loopback_call()
        try:
            await client.next_event("session.created")
            client.channel.send(json.dumps({"type": "response.create"}))
            await client.next_event("response.done")
        finally:
            hangup_call(call_id)
            await _drain_call_tasks()
            await client.close()
        recorded = logged_usage_entries(capsys.readouterr().out, model=_MODEL)
        assert len(recorded) == 1
        assert recorded[0]["total_tokens"] == 43

    async def test_a_failing_session_logs_a_critical_500(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An escape from a detached call task is logged, not swallowed."""
        from stdapi.realtime_webrtc import _serve_call  # noqa: PLC0415

        class _BoomSession:
            async def run(self) -> None:
                msg = "boom"
                raise RuntimeError(msg)

        transport = WebRTCCallTransport("rtc_fail", 24000, 24000)
        _CALLS["rtc_fail"] = RealtimeCall(
            transport, cast("RealtimeSession", _BoomSession()), None
        )
        await _serve_call(
            "rtc_fail",
            transport,
            cast("RealtimeSession", _BoomSession()),
            _fake_request(),
        )
        assert "rtc_fail" not in _CALLS
        events = []
        for line in capsys.readouterr().out.splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        entry = next(
            event
            for event in events
            if isinstance(event, dict) and event.get("status_code") == 500
        )
        assert entry["level"] == "critical"
        assert "RuntimeError" in "".join(entry["error_detail"])


class TestSecretAudioEvent:
    """The synthesized append events the caller's audio becomes.

    Ref: stdapi/realtime_webrtc.py:WebRTCCallTransport._push_audio
    """

    async def test_audio_is_batched_and_base64_encoded(self) -> None:
        """One pushed chunk is one well-formed append event."""
        transport = WebRTCCallTransport("rtc_test", 24000, 24000)
        transport._push_audio(b"\x01\x02" * 100)  # noqa: SLF001
        message = await transport.receive()
        event = json.loads(message["text"])
        assert event["type"] == "input_audio_buffer.append"
        assert base64.b64decode(event["audio"]) == b"\x01\x02" * 100
        await transport.close(1000, "")
