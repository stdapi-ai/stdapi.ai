"""Realtime session internals: ephemeral secrets, audio codecs, event table.

Everything here runs against a fake backend, so the whole event translation is
exercised without opening a model conversation: the table is what a live session
is built out of, and it is the part a live test can only sample.

Ref: stdapi/realtime.py:RealtimeSession
     https://developers.openai.com/api/reference/resources/realtime
"""

from __future__ import annotations

import asyncio
import base64
import threading
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

import pytest
from openai.types.realtime.realtime_server_event import RealtimeServerEvent
from pydantic import TypeAdapter
from starlette.websockets import WebSocketDisconnect

from stdapi.config import SETTINGS
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
from stdapi.realtime import (
    CLIENT_SECRET_PREFIX,
    RealtimeSession,
    close_realtime_sessions,
    decode_client_audio,
    encode_client_audio,
    mint_client_secret,
    open_realtime_sessions,
    read_client_secret,
    websocket_credential,
)
from stdapi.types.openai_realtime import (
    RealtimeSessionConfig,
    TranscriptionSessionConfig,
)
from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterator

    from starlette.testclient import TestClient

    from stdapi.models.realtime import BackendEvent

pytestmark = pytest.mark.local

#: Model identifier the fake backend answers for, never resolved against AWS.
_MODEL = "fake.realtime-v1:0"

#: 16-bit samples covering the range both G.711 codecs have to carry.
_SAMPLE_SWEEP = (-32000, -8000, -1000, -100, 0, 100, 1000, 8000, 32000)

#: Largest error a G.711 round trip may introduce, in 16-bit sample units.
_G711_TOLERANCE = 1024

#: One frame of silence, in bytes, appended by a client that must send audio.
_SILENT_FRAME = bytes(1600)

#: That same frame, as a client sends it.
_FRAME = base64.b64encode(_SILENT_FRAME).decode()

#: One answer carrying 100 ms of speech at the session's default 24 kHz output.
_ANSWER_SCRIPT = (
    ResponseStarted(),
    OutputTranscript("Sure thing."),
    OutputAudio(bytes(4800)),
    ResponseFinished(),
)

#: Seconds a test waits for the fake backend to reach a point in its script.
_GATE_TIMEOUT = 5.0

#: The GA item lifecycle pair, which every tracked item is announced with.
_ITEM_LIFECYCLE_KINDS = frozenset({"conversation.item.added", "conversation.item.done"})

#: The official client's own server event union, which it validates every frame against.
_SERVER_EVENT: TypeAdapter[RealtimeServerEvent] = TypeAdapter(RealtimeServerEvent)


class _Gate:
    """A script marker holding the backend until the test releases it.

    The fake backend otherwise replays its whole script before the client can
    react, which is exactly the window barge-in happens in.
    """

    __slots__ = ("_event", "reached")

    def __init__(self) -> None:
        """Start closed, and unreached."""
        self._event = threading.Event()
        self.reached = threading.Event()

    def release(self) -> None:
        """Let the backend replay the rest of its script."""
        self._event.set()

    async def wait(self) -> None:
        """Block the backend until the test thread releases the gate."""
        self.reached.set()
        # Polled: the release comes from the test thread, not the server's loop.
        while not self._event.is_set():  # noqa: ASYNC110
            await asyncio.sleep(0.005)


class _FakeSession(RealtimeBackendSession):
    """A backend that records what it was sent and replays a script."""

    __slots__ = (
        "_script",
        "audio",
        "closed",
        "ended",
        "reader_cancelled",
        "region",
        "texts",
    )

    def __init__(self, script: list[Any]) -> None:
        """Record nothing yet and hold *script* to replay.

        Args:
            script: Events to report once the caller ends a turn, with optional
                ``_Gate`` markers holding the replay.
        """
        self._script = script
        self.audio = bytearray()
        self.texts: list[str] = []
        self.ended = 0
        self.region = "us-east-1"
        self.closed = asyncio.Event()
        self.reader_cancelled = False

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
        """Replay the script, then wait until the stream is closed, as one does.

        Yields:
            Each scripted event, in order; a ``_Gate`` entry reports nothing and
            holds the replay until the test releases it.
        """
        for event in self._script:
            if isinstance(event, _Gate):
                await event.wait()
                continue
            yield event
        try:
            await self.closed.wait()
        except asyncio.CancelledError:
            self.reader_cancelled = True
            raise


class _FakeModel(RealtimeModelBase[Any, Any]):
    """A realtime model whose conversations are :class:`_FakeSession`."""

    __slots__ = ()

    MATCHER = _MODEL

    MAX_SESSION_SECONDS = 30.0

    #: The session every conversation of this model opened, for assertions.
    opened: list[_FakeSession] = []  # noqa: RUF012 - a test double's recorder

    #: Script every conversation replays.
    script: list[Any] = []  # noqa: RUF012 - a test double's recorder

    @asynccontextmanager
    async def open_session(self, **_: Any) -> AsyncIterator[RealtimeBackendSession]:  # noqa: ANN401
        """Open one fake conversation, whose stream ends when it is closed.

        Yields:
            The fake session, also appended to ``opened``.
        """
        session = _FakeSession(list(_FakeModel.script))
        _FakeModel.opened.append(session)
        try:
            yield session
        finally:
            session.closed.set()


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_FakeModel]]:
    """Serve every realtime session from :class:`_FakeModel`.

    Both the model lookup and the catalog validation are replaced: the point of
    this module is the translation, not the model registry.

    Yields:
        The fake model class, whose ``script`` the test sets and whose
        ``opened`` sessions it asserts on.
    """
    from stdapi import realtime  # noqa: PLC0415

    class _Details:
        """The one field the session reads off a validated model."""

        id = _MODEL

    async def _validate_model(model_id: str, **_: Any) -> Any:  # noqa: ANN401
        """Accept the fake model and refuse everything else."""
        from stdapi.api_errors import UnsupportedModelError  # noqa: PLC0415
        from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

        if model_id != _MODEL:
            raise UnsupportedModelError(model_id)
        REQUEST_LOG.get()["model_id"] = _MODEL
        return _Details()

    monkeypatch.setattr(realtime, "validate_model", _validate_model)
    monkeypatch.setattr(realtime, "get_realtime_model", lambda _id: _FakeModel(_MODEL))
    _FakeModel.opened = []
    _FakeModel.script = []
    yield _FakeModel
    _FakeModel.opened = []
    _FakeModel.script = []


def _connect(client: TestClient) -> Any:  # noqa: ANN401
    """Open the realtime WebSocket of the in-process app.

    Args:
        client: The pre-authenticated test client.

    Returns:
        The WebSocket context manager.
    """
    return client.websocket_connect(f"/v1/realtime?model={_MODEL}")


def _read_until_closed(websocket: Any) -> None:  # noqa: ANN401
    """Read events until the server closes the connection.

    Args:
        websocket: The open connection.

    Raises:
        WebSocketDisconnect: Always, once the server closes.
    """
    while True:
        websocket.receive_json()


def _arm_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a handler accepting one key, which no test presents.

    The lifespan is what arms authentication, and ``app_client`` skips it; this
    is how a unit test gets a deployment that refuses a wrong credential.

    Args:
        monkeypatch: The test's patcher.
    """
    from pydantic import SecretStr  # noqa: PLC0415

    import stdapi.auth  # noqa: PLC0415
    from stdapi.auth import AuthenticationHandler  # noqa: PLC0415

    monkeypatch.setattr(SETTINGS, "api_key", SecretStr("the-deployment-key"))
    monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
    monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
    handler = AuthenticationHandler()
    handler._hash_api_key(SecretStr("the-deployment-key"))  # noqa: SLF001
    monkeypatch.setattr(stdapi.auth, "_auth_handler", handler)


def _drain(websocket: Any, terminal: str) -> list[dict[str, Any]]:  # noqa: ANN401
    """Read events up to and including the first one of type *terminal*.

    Args:
        websocket: The open connection.
        terminal: Event type ending the collection.

    Returns:
        Every event received, in order.
    """
    events: list[dict[str, Any]] = []
    while True:
        event = websocket.receive_json()
        events.append(event)
        if event["type"] in {terminal, "error"}:
            return events


class TestClientSecrets:
    """A minted secret carries its session and stops working when it expires.

    Ref: stdapi/realtime.py:mint_client_secret
    """

    def test_a_minted_secret_reads_back_the_session_it_carries(self) -> None:
        """The configuration survives the round trip, with no server-side store."""
        session = RealtimeSessionConfig(instructions="Be brief.", model="a.model")

        value, expires_at = mint_client_secret(session, 600)
        read = read_client_secret(value)

        assert value.startswith(CLIENT_SECRET_PREFIX)
        assert expires_at > 0
        assert isinstance(read, RealtimeSessionConfig)
        assert read.instructions == "Be brief."
        assert read.model == "a.model"

    def test_a_transcription_session_survives_the_round_trip(self) -> None:
        """The two session shapes are told apart by their own ``type``."""
        value, _ = mint_client_secret(TranscriptionSessionConfig(), 600)

        assert isinstance(read_client_secret(value), TranscriptionSessionConfig)

    def test_an_expired_secret_is_refused(self) -> None:
        """A secret past its expiry reads as no session at all."""
        value, _ = mint_client_secret(RealtimeSessionConfig(), -1)

        assert read_client_secret(value) is None

    def test_a_tampered_secret_is_refused(self) -> None:
        """Editing the payload invalidates the signature over it."""
        value, _ = mint_client_secret(
            RealtimeSessionConfig(instructions="Be brief."), 600
        )
        payload, _, signature = value.partition(".")

        assert read_client_secret(f"{payload}x.{signature}") is None
        assert read_client_secret(f"{payload}.{signature[:-2]}xy") is None

    @pytest.mark.parametrize(
        "value", ["", "sk-something", "ek_", "ek_nodot", "ek_!!!.!!!"]
    )
    def test_anything_that_is_not_a_secret_is_refused(self, value: str) -> None:
        """A credential of another kind is not mistaken for a session."""
        assert read_client_secret(value) is None


class TestSigningKeyDerivation:
    """The key a secret is signed with is the same on every instance.

    A per-process key would make an ephemeral secret minted by one task fail on
    whichever task the client's WebSocket happens to reach, which is the whole
    failure mode a stateless secret exists to avoid.

    Ref: stdapi/auth.py:AuthenticationHandler.derived_key
    """

    def test_the_same_api_key_derives_the_same_signing_key(self) -> None:
        """Two independent handlers holding one API key agree on the key."""
        from pydantic import SecretStr  # noqa: PLC0415

        from stdapi.auth import AuthenticationHandler  # noqa: PLC0415

        first, second = AuthenticationHandler(), AuthenticationHandler()
        first._hash_api_key(SecretStr("the-deployment-key"))  # noqa: SLF001
        second._hash_api_key(SecretStr("the-deployment-key"))  # noqa: SLF001

        derived = first.derived_key(b"stdapi-rt", 32)
        assert derived is not None
        assert derived == second.derived_key(b"stdapi-rt", 32)

    def test_a_different_api_key_derives_a_different_signing_key(self) -> None:
        """A deployment cannot verify a secret another deployment minted."""
        from pydantic import SecretStr  # noqa: PLC0415

        from stdapi.auth import AuthenticationHandler  # noqa: PLC0415

        first, second = AuthenticationHandler(), AuthenticationHandler()
        first._hash_api_key(SecretStr("one-key"))  # noqa: SLF001
        second._hash_api_key(SecretStr("another-key"))  # noqa: SLF001

        assert first.derived_key(b"stdapi-rt", 32) != second.derived_key(
            b"stdapi-rt", 32
        )

    def test_no_api_key_derives_nothing(self) -> None:
        """Without a credential to derive from there is no deployment-wide key."""
        from stdapi.auth import AuthenticationHandler  # noqa: PLC0415

        assert AuthenticationHandler().derived_key(b"stdapi-rt", 32) is None


class TestClientSecretRoute:
    """The minting route answers the upstream envelope and honours its lifetime.

    Ref: https://developers.openai.com/api/reference/resources/realtime/subresources/client_secrets/methods/create
         stdapi/routes/openai_realtime.py:create_realtime_client_secret
    """

    def test_an_empty_body_mints_a_default_realtime_session(
        self, app_client: TestClient
    ) -> None:
        """No body at all is a valid request, as it is upstream."""
        response = app_client.post("/v1/realtime/client_secrets")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["value"].startswith(CLIENT_SECRET_PREFIX)
        assert body["session"]["type"] == "realtime"
        assert read_client_secret(body["value"]) is not None

    def test_the_requested_lifetime_is_applied(self, app_client: TestClient) -> None:
        """``expires_after.seconds`` decides when the secret stops working."""
        import time  # noqa: PLC0415

        response = app_client.post(
            "/v1/realtime/client_secrets",
            json={
                "expires_after": {"anchor": "created_at", "seconds": 60},
                "session": {"type": "realtime", "instructions": "Be brief."},
            },
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert 0 < body["expires_at"] - int(time.time()) <= 60
        assert body["session"]["instructions"] == "Be brief."

    @pytest.mark.parametrize("seconds", [1, 100000])
    def test_a_lifetime_outside_the_accepted_range_is_refused(
        self, app_client: TestClient, seconds: int
    ) -> None:
        """The bounds are enforced at parse time, not after minting."""
        response = app_client.post(
            "/v1/realtime/client_secrets",
            json={"expires_after": {"anchor": "created_at", "seconds": seconds}},
        )

        assert response.status_code == 400, response.text
        assert "expires_after.seconds" in response.json()["error"]["message"]

    def test_the_minted_value_stays_out_of_the_request_log(
        self, app_client: TestClient, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A credential is never written to the log that records the response."""
        response = app_client.post(
            "/v1/realtime/client_secrets", json={"session": {"type": "transcription"}}
        )

        assert response.status_code == 200, response.text
        assert response.json()["value"] not in capsys.readouterr().out


class TestAudioConversion:
    """The client's audio formats convert both ways without a resample.

    Ref: https://developers.openai.com/api/reference/resources/realtime
         stdapi/realtime.py:decode_client_audio
    """

    def test_pcm_is_carried_through_untouched(self) -> None:
        """The default format is the backend's own, so nothing is converted."""
        pcm = b"\x01\x02\x03\x04"

        assert decode_client_audio(pcm, "audio/pcm") == pcm
        assert encode_client_audio(pcm, "audio/pcm") == pcm

    @pytest.mark.parametrize("media_type", ["audio/pcmu", "audio/pcma"])
    def test_a_g711_round_trip_keeps_the_signal(self, media_type: str) -> None:
        """Every sample survives encoding and decoding within codec tolerance."""
        import struct  # noqa: PLC0415

        pcm = struct.pack(f"<{len(_SAMPLE_SWEEP)}h", *_SAMPLE_SWEEP)

        companded = encode_client_audio(pcm, media_type)
        restored = struct.unpack(
            f"<{len(_SAMPLE_SWEEP)}h", decode_client_audio(companded, media_type)
        )

        assert len(companded) == len(_SAMPLE_SWEEP)
        for original, result in zip(_SAMPLE_SWEEP, restored, strict=True):
            assert abs(original - result) <= _G711_TOLERANCE, (original, result)

    @pytest.mark.parametrize("media_type", ["audio/pcmu", "audio/pcma"])
    def test_every_encoded_byte_survives_a_decode_and_re_encode(
        self, media_type: str
    ) -> None:
        """Re-encoding a decoded byte lands on a code carrying the same sample.

        Not the same byte: both codecs give positive and negative zero their own
        code, so one decoded sample legitimately re-encodes to the other one.
        """
        encoded = bytes(range(256))

        decoded = decode_client_audio(encoded, media_type)
        again = encode_client_audio(decoded, media_type)

        assert decode_client_audio(again, media_type) == decoded


class TestWebsocketCredential:
    """Every carrier a realtime client may present its credential in.

    Ref: https://developers.openai.com/api/docs/guides/realtime
         stdapi/realtime.py:websocket_credential
    """

    @staticmethod
    def _websocket(*headers: tuple[bytes, bytes]) -> Any:  # noqa: ANN401
        """Build a WebSocket carrying *headers* and nothing else."""
        from starlette.websockets import WebSocket  # noqa: PLC0415

        return WebSocket(
            {
                "type": "websocket",
                "path": "/v1/realtime",
                "headers": list(headers),
                "query_string": b"",
            },
            receive=None,  # type: ignore[arg-type]
            send=None,  # type: ignore[arg-type]
        )

    def test_the_authorization_header_is_read(self) -> None:
        """The SDK sends the credential as a bearer token."""
        websocket = self._websocket((b"authorization", b"Bearer secret-value"))

        assert websocket_credential(websocket) == "secret-value"

    def test_the_api_key_header_wins_over_the_bearer_token(self) -> None:
        """``x-api-key`` takes precedence, as it does on the HTTP routes."""
        websocket = self._websocket(
            (b"authorization", b"Bearer bearer-value"), (b"x-api-key", b"header-value")
        )

        assert websocket_credential(websocket) == "header-value"

    def test_the_subprotocol_carries_a_browser_client_credential(self) -> None:
        """A browser cannot set headers on a WebSocket, so it uses this instead."""
        websocket = self._websocket(
            (
                b"sec-websocket-protocol",
                b"realtime, openai-insecure-api-key.browser-value",
            )
        )

        assert websocket_credential(websocket) == "browser-value"

    def test_no_credential_reads_as_none(self) -> None:
        """An anonymous connection presents nothing, rather than an empty string."""
        assert websocket_credential(self._websocket()) is None


class TestSessionEvents:
    """The client and server event tables, over a fake backend.

    Ref: https://developers.openai.com/api/reference/resources/realtime
         stdapi/realtime.py:RealtimeSession
    """

    @pytest.mark.usefixtures("fake_backend")
    def test_the_session_opens_on_a_created_event(self, app_client: TestClient) -> None:
        """``session.created`` names the session and the model serving it."""
        with _connect(app_client) as websocket:
            created = websocket.receive_json()

        assert created["type"] == "session.created"
        assert created["session"]["id"].startswith("sess_")
        assert created["session"]["model"] == _MODEL
        assert created["event_id"].startswith("event_")

    @pytest.mark.usefixtures("fake_backend")
    def test_an_unknown_event_type_is_reported_and_the_session_continues(
        self, app_client: TestClient
    ) -> None:
        """An unsupported event answers ``error`` rather than closing the socket."""
        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "conversation.item.retrieve"})
            error = websocket.receive_json()
            websocket.send_json({"type": "input_audio_buffer.clear"})
            cleared = websocket.receive_json()

        assert error["type"] == "error"
        assert error["error"]["type"] == "invalid_request_error"
        assert cleared["type"] == "input_audio_buffer.cleared"

    @pytest.mark.usefixtures("fake_backend")
    def test_a_malformed_payload_is_reported(self, app_client: TestClient) -> None:
        """Text that is not a JSON object answers ``error``."""
        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_text("not json at all")
            error = websocket.receive_json()

        assert error["type"] == "error"
        assert "JSON" in error["error"]["message"]

    @pytest.mark.usefixtures("fake_backend")
    def test_the_updated_session_is_echoed_back(self, app_client: TestClient) -> None:
        """``session.update`` answers the whole effective configuration."""
        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {"type": "realtime", "instructions": "Be brief."},
                }
            )
            updated = websocket.receive_json()

        assert updated["type"] == "session.updated"
        assert updated["session"]["instructions"] == "Be brief."
        assert updated["session"]["model"] == _MODEL

    def test_committed_audio_reaches_the_model_and_ends_its_turn(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """Appended audio is held until the commit, which ends the turn once."""
        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "audio": {"input": {"turn_detection": None}},
                    },
                }
            )
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(_SILENT_FRAME).decode(),
                }
            )
            websocket.send_json({"type": "input_audio_buffer.commit"})
            committed = websocket.receive_json()

        assert committed["type"] == "input_audio_buffer.committed"
        assert committed["item_id"].startswith("item_")
        session = fake_backend.opened[0]
        assert bytes(session.audio) == _SILENT_FRAME
        assert session.ended == 1

    def test_cleared_audio_never_reaches_the_model(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """``input_audio_buffer.clear`` drops what was appended and not committed."""
        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "audio": {"input": {"turn_detection": None}},
                    },
                }
            )
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(_SILENT_FRAME).decode(),
                }
            )
            websocket.send_json({"type": "input_audio_buffer.clear"})
            websocket.receive_json()
            websocket.send_json({"type": "input_audio_buffer.commit"})
            websocket.receive_json()

        assert not bytes(fake_backend.opened[0].audio)

    def test_a_text_item_reaches_the_model(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """A written message is added to the conversation and acknowledged."""
        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello there."}],
                    },
                }
            )
            created = websocket.receive_json()

        assert created["type"] == "conversation.item.created"
        assert created["item"]["content"][0]["text"] == "Hello there."
        assert fake_backend.opened[0].texts == ["Hello there."]

    @pytest.mark.usefixtures("fake_backend")
    def test_an_audio_item_is_refused(self, app_client: TestClient) -> None:
        """Speech goes through the audio buffer, never through an item."""
        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_audio", "audio": "AAAA"}],
                    },
                }
            )
            error = websocket.receive_json()

        assert error["type"] == "error"
        assert "input_audio_buffer.append" in error["error"]["message"]

    def test_a_backend_turn_is_reported_as_the_whole_response_sequence(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """One backend answer becomes the full ordered client event sequence."""
        fake_backend.script = [
            SpeechStarted(120),
            SpeechStopped(2480),
            InputTranscript("this is a test"),
            ResponseStarted(),
            OutputTranscript("Sure thing."),
            OutputAudio(b"\x01\x00\x02\x00"),
            UsageReport(
                input_speech_tokens=197,
                input_text_tokens=348,
                output_speech_tokens=52,
                output_text_tokens=26,
                total_tokens=623,
            ),
            ResponseFinished(),
        ]

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "audio": {
                            "input": {
                                "turn_detection": None,
                                "transcription": {"model": "whisper-1"},
                            }
                        },
                    },
                }
            )
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(_SILENT_FRAME).decode(),
                }
            )
            websocket.send_json({"type": "input_audio_buffer.commit"})
            events = _drain(websocket, "response.done")

        # Dropped: the client half produces them, so they interleave with these.
        kinds = [
            event["type"]
            for event in events
            if event["type"] not in _ITEM_LIFECYCLE_KINDS
        ]
        assert kinds == [
            "input_audio_buffer.committed",
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.completed",
            "response.created",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_audio_transcript.delta",
            "response.output_audio.delta",
            "response.output_audio.done",
            "response.output_audio_transcript.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.done",
        ], kinds
        by_kind = {event["type"]: event for event in events}
        assert by_kind["input_audio_buffer.speech_started"]["audio_start_ms"] == 120
        assert (
            base64.b64decode(by_kind["response.output_audio.delta"]["delta"])
            == b"\x01\x00\x02\x00"
        )
        assert (
            by_kind["response.output_audio_transcript.done"]["transcript"]
            == "Sure thing."
        )
        done = by_kind["response.done"]["response"]
        assert done["status"] == "completed"
        assert done["usage"]["total_tokens"] == 623
        assert done["usage"]["input_token_details"]["audio_tokens"] == 197
        assert done["usage"]["output_token_details"]["audio_tokens"] == 52

    def test_a_completed_transcript_carries_what_reading_the_turn_cost(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """``usage`` is required on the event, and is the caller's own input.

        A client built on the official event models drops the whole transcript
        when it is missing, so the caller's turn never reaches the conversation.
        """
        fake_backend.script = [
            UsageReport(
                input_speech_tokens=197, input_text_tokens=12, total_tokens=209
            ),
            InputTranscript("this is a test"),
        ]

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "audio": {"input": {"transcription": {"model": "whisper-1"}}},
                    },
                }
            )
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            events = _drain(
                websocket, "conversation.item.input_audio_transcription.completed"
            )

        completed = events[-1]
        assert completed["usage"] == {
            "type": "tokens",
            "input_tokens": 209,
            "output_tokens": 0,
            "total_tokens": 209,
            "input_token_details": {"audio_tokens": 197, "text_tokens": 12},
        }, completed

    def test_a_text_only_session_reports_text_instead_of_speech(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """``output_modalities: ['text']`` suppresses every audio event."""
        fake_backend.script = [
            ResponseStarted(),
            OutputTranscript("Sure thing."),
            OutputAudio(b"\x01\x00"),
            ResponseFinished(),
        ]

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "output_modalities": ["text"],
                        "audio": {"input": {"turn_detection": None}},
                    },
                }
            )
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            events = _drain(websocket, "response.done")

        kinds = [event["type"] for event in events]
        assert "response.output_text.delta" in kinds, kinds
        assert not [kind for kind in kinds if "audio" in kind], kinds

    @pytest.mark.usefixtures("fake_backend")
    def test_the_backend_settings_cannot_change_once_the_model_answers(
        self, app_client: TestClient
    ) -> None:
        """Changing the voice mid-conversation is refused, not silently dropped."""
        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "audio": {"output": {"voice": "verse"}},
                    },
                }
            )
            error = websocket.receive_json()

        assert error["type"] == "error"
        assert "voice" in error["error"]["message"]

    @pytest.mark.usefixtures("fake_backend")
    def test_an_unknown_model_ends_the_session_on_an_error_event(
        self, app_client: TestClient
    ) -> None:
        """A model that cannot serve the route is refused after the upgrade."""
        with app_client.websocket_connect("/v1/realtime?model=nope") as websocket:
            error = websocket.receive_json()

        assert error["type"] == "error"
        assert error["error"]["code"] == "model_not_found"

    @pytest.mark.usefixtures("fake_backend")
    def test_the_model_is_required(self, app_client: TestClient) -> None:
        """A connection naming no model is refused before anything is opened."""
        with app_client.websocket_connect("/v1/realtime") as websocket:
            error = websocket.receive_json()

        assert error["type"] == "error"
        assert "model" in error["error"]["message"]

    @pytest.mark.usefixtures("fake_backend")
    def test_an_unexpected_failure_closes_the_socket_cleanly(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A failure nothing else answers ends the session, not the connection.

        No error middleware runs on a WebSocket, so anything escaping the
        session handler leaves the client holding an open socket that never
        answers, and the failure unreported.

        Ref: stdapi/realtime.py:serve_realtime_session
        """
        from stdapi import realtime  # noqa: PLC0415

        def _fail(_model_id: str) -> Any:  # noqa: ANN401
            message = "backend exploded"
            raise RuntimeError(message)

        monkeypatch.setattr(realtime, "get_realtime_model", _fail)
        capfd.readouterr()
        with _connect(app_client) as websocket:
            error = websocket.receive_json()

        assert error["type"] == "error"
        assert error["error"]["type"] == "server_error"
        assert "backend exploded" not in error["error"]["message"]
        logged = capfd.readouterr().out
        assert '"level":"critical"' in logged
        assert "backend exploded" in logged


class TestConversationItems:
    """The item lifecycle and the events addressing one item by identifier.

    A voice framework waits on ``conversation.item.done`` before advancing a
    turn, and sends ``conversation.item.truncate`` on every barge-in: both are
    the vocabulary the current API is written in, and neither has a fallback.

    Ref: https://developers.openai.com/api/reference/resources/realtime/server-events
         stdapi/realtime.py:RealtimeSession._add_item
    """

    @staticmethod
    def _answered(websocket: Any) -> tuple[str, list[dict[str, Any]]]:  # noqa: ANN401
        """Run one answered turn and return its item id and every event."""
        websocket.receive_json()
        websocket.send_json({"type": "response.create"})
        events = _drain(websocket, "response.done")
        added = next(
            event for event in events if event["type"] == "conversation.item.added"
        )
        return added["item"]["id"], events

    @pytest.mark.usefixtures("fake_backend")
    def test_a_written_item_is_announced_in_both_vocabularies(
        self, app_client: TestClient
    ) -> None:
        """The added/done pair is sent beside the event it superseded."""
        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello there."}],
                    },
                }
            )
            events = _drain(websocket, "conversation.item.done")

        by_kind = {event["type"]: event for event in events}
        assert [event["type"] for event in events] == [
            "conversation.item.created",
            "conversation.item.added",
            "conversation.item.done",
        ]
        assert by_kind["conversation.item.added"]["item"]["role"] == "user"
        assert by_kind["conversation.item.done"]["item"]["status"] == "completed"
        assert (
            by_kind["conversation.item.done"]["item"]["id"]
            == by_kind["conversation.item.created"]["item"]["id"]
        )

    def test_an_answer_settles_before_the_response_is_done(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """The answer's item is added in progress and done with its transcript."""
        fake_backend.script = list(_ANSWER_SCRIPT)

        with _connect(app_client) as websocket:
            _, events = self._answered(websocket)

        kinds = [event["type"] for event in events]
        assert (
            kinds.index("conversation.item.added")
            < kinds.index("conversation.item.done")
            < kinds.index("response.done")
        ), kinds
        by_kind = {event["type"]: event for event in events}
        added = by_kind["conversation.item.added"]["item"]
        done = by_kind["conversation.item.done"]["item"]
        assert added["role"] == "assistant"
        assert added["status"] == "in_progress"
        assert done["status"] == "completed"
        assert done["content"][0]["transcript"] == "Sure thing."

    def test_an_interrupted_answer_settles_as_incomplete(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """Every view of the item agrees on the status the answer ended with."""
        fake_backend.script = [*_ANSWER_SCRIPT[:-1], ResponseFinished(interrupted=True)]

        with _connect(app_client) as websocket:
            _, events = self._answered(websocket)

        by_kind = {event["type"]: event for event in events}
        assert by_kind["conversation.item.done"]["item"]["status"] == "incomplete"
        assert by_kind["response.output_item.done"]["item"]["status"] == "incomplete"
        assert by_kind["response.done"]["response"]["status"] == "incomplete"

    @pytest.mark.parametrize("modalities", [None, ["text"]])
    def test_one_item_is_rendered_the_same_way_on_every_event_carrying_it(
        self,
        app_client: TestClient,
        fake_backend: type[_FakeModel],
        modalities: list[str] | None,
    ) -> None:
        """The answer's item reaches the client six times, as one object.

        A client keyed on the item identifier merges what the response events
        and the conversation events say about it, so a field rendered one way
        under ``response.output_item.done`` and another under
        ``conversation.item.done`` is the same message read as two.

        Ref: stdapi/realtime.py:_item_body
        """
        fake_backend.script = list(_ANSWER_SCRIPT)

        with _connect(app_client) as websocket:
            websocket.receive_json()
            if modalities is not None:
                websocket.send_json(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "output_modalities": modalities,
                        },
                    }
                )
                websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            events = _drain(websocket, "response.done")
            item_id = next(
                event for event in events if event["type"] == "conversation.item.added"
            )["item"]["id"]
            websocket.send_json(
                {"type": "conversation.item.retrieve", "item_id": item_id}
            )
            events.append(websocket.receive_json())

        rendered: dict[str, list[Any]] = {"in_progress": [], "completed": []}
        for event in events:
            item = (
                event["response"]["output"][0]
                if event["type"] == "response.done"
                else event.get("item")
            )
            if isinstance(item, dict) and item["id"] == item_id:
                rendered[item["status"]].append((event["type"], item))
        assert [kind for kind, _ in rendered["in_progress"]] == [
            "response.output_item.added",
            "conversation.item.added",
        ], rendered
        assert [kind for kind, _ in rendered["completed"]] == [
            "response.output_item.done",
            "conversation.item.done",
            "response.done",
            "conversation.item.retrieved",
        ], rendered
        for stage in rendered.values():
            first = stage[0][1]
            assert all(item == first for _, item in stage), stage

    def test_truncating_an_answer_drops_what_the_caller_never_heard(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """The item's transcript is removed, and its audio cut to what was played."""
        fake_backend.script = list(_ANSWER_SCRIPT)

        with _connect(app_client) as websocket:
            item_id, _ = self._answered(websocket)
            websocket.send_json(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": 40,
                }
            )
            truncated = websocket.receive_json()
            websocket.send_json(
                {"type": "conversation.item.retrieve", "item_id": item_id}
            )
            retrieved = websocket.receive_json()

        assert truncated["type"] == "conversation.item.truncated", truncated
        assert truncated["item_id"] == item_id
        assert truncated["audio_end_ms"] == 40
        assert retrieved["type"] == "conversation.item.retrieved", retrieved
        assert retrieved["item"]["content"][0]["transcript"] == ""

    def test_truncating_past_the_end_of_the_audio_is_refused(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """100 ms of speech cannot be truncated at 5 seconds."""
        fake_backend.script = list(_ANSWER_SCRIPT)

        with _connect(app_client) as websocket:
            item_id, _ = self._answered(websocket)
            websocket.send_json(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": 5000,
                }
            )
            error = websocket.receive_json()

        assert error["type"] == "error", error
        assert error["error"]["type"] == "invalid_request_error"
        assert "5000 ms" in error["error"]["message"]

    @pytest.mark.parametrize(
        ("event", "message"),
        [
            (
                {"type": "conversation.item.truncate", "item_id": "item_nope"},
                "'audio_end_ms'",
            ),
            (
                {
                    "type": "conversation.item.truncate",
                    "item_id": 5,
                    "content_index": 0,
                    "audio_end_ms": 0,
                },
                "'audio_end_ms'",
            ),
            (
                {
                    "type": "conversation.item.truncate",
                    "item_id": "item_nope",
                    "audio_end_ms": 0,
                },
                "'content_index'",
            ),
            (
                {
                    "type": "conversation.item.truncate",
                    "item_id": "item_nope",
                    "content_index": 0,
                    "audio_end_ms": 0,
                },
                "No assistant message item 'item_nope'",
            ),
            (
                {"type": "conversation.item.retrieve", "item_id": "item_nope"},
                "No conversation item 'item_nope'",
            ),
            ({"type": "conversation.item.delete"}, "No conversation item is"),
        ],
    )
    @pytest.mark.usefixtures("fake_backend")
    def test_an_item_event_that_addresses_nothing_is_refused(
        self, app_client: TestClient, event: dict[str, Any], message: str
    ) -> None:
        """An unusable item event answers ``error`` and keeps the session open."""
        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(event)
            error = websocket.receive_json()
            websocket.send_json({"type": "input_audio_buffer.clear"})
            cleared = websocket.receive_json()

        assert error["type"] == "error", error
        assert error["error"]["type"] == "invalid_request_error"
        assert message in error["error"]["message"], error
        assert cleared["type"] == "input_audio_buffer.cleared"

    def test_a_deleted_item_is_gone_from_the_session(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """``delete`` is acknowledged, and the item can no longer be retrieved."""
        fake_backend.script = list(_ANSWER_SCRIPT)

        with _connect(app_client) as websocket:
            item_id, _ = self._answered(websocket)
            websocket.send_json(
                {"type": "conversation.item.delete", "item_id": item_id}
            )
            deleted = websocket.receive_json()
            websocket.send_json(
                {"type": "conversation.item.retrieve", "item_id": item_id}
            )
            error = websocket.receive_json()

        assert deleted["type"] == "conversation.item.deleted", deleted
        assert deleted["item_id"] == item_id
        assert error["type"] == "error", error

    @pytest.mark.usefixtures("fake_backend")
    def test_a_committed_turn_is_added_as_the_caller_own_item(
        self, app_client: TestClient
    ) -> None:
        """The committed audio becomes an item under the announced identifier."""
        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "audio": {"input": {"turn_detection": None}},
                    },
                }
            )
            websocket.receive_json()
            websocket.send_json({"type": "input_audio_buffer.append", "audio": _FRAME})
            websocket.send_json({"type": "input_audio_buffer.commit"})
            events = _drain(websocket, "conversation.item.done")

        by_kind = {event["type"]: event for event in events}
        committed = by_kind["input_audio_buffer.committed"]["item_id"]
        assert by_kind["conversation.item.added"]["item"]["id"] == committed
        assert by_kind["conversation.item.added"]["item"]["role"] == "user"
        assert by_kind["conversation.item.done"]["item"]["id"] == committed

    def test_an_answer_is_truncated_while_it_is_still_being_spoken(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """Barge-in truncates the answer in flight, not once it is over.

        A caller speaks over the answer while it is being streamed, which is the
        only moment truncation is asked for: the item's audio is what has been
        sent so far, and what survives the answer's own end is what was heard.
        """
        gate = _Gate()
        fake_backend.script = [
            ResponseStarted(),
            OutputTranscript("Sure thing."),
            # 100 ms of speech at the session's default 24 kHz output.
            OutputAudio(bytes(4800)),
            gate,
            ResponseFinished(interrupted=True),
        ]

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            events = _drain(websocket, "response.output_audio.delta")
            item_id = next(
                event for event in events if event["type"] == "conversation.item.added"
            )["item"]["id"]
            websocket.send_json(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": 40,
                }
            )
            truncated = websocket.receive_json()
            gate.release()
            _drain(websocket, "response.done")
            websocket.send_json(
                {"type": "conversation.item.retrieve", "item_id": item_id}
            )
            retrieved = websocket.receive_json()

        assert truncated["type"] == "conversation.item.truncated", truncated
        assert truncated["audio_end_ms"] == 40
        assert retrieved["item"]["content"][0]["transcript"] == "", retrieved
        assert retrieved["item"]["status"] == "incomplete", retrieved


class TestTranscriptionFailure:
    """A transcript that could not be produced is its own event, not silence.

    Silence and a failed transcription are otherwise the same event stream, so
    a client waiting on the transcript of a turn cannot tell a caller who said
    nothing from one whose speech was never read.

    Ref: https://developers.openai.com/api/reference/resources/realtime/server-events
         stdapi/realtime.py:RealtimeSession._report_transcription_failed
    """

    def test_a_failed_transcript_names_the_item_it_belongs_to(
        self,
        app_client: TestClient,
        fake_backend: type[_FakeModel],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The failure is reported against the item, before the session ends."""
        from stdapi import realtime  # noqa: PLC0415
        from stdapi.api_errors import ApiError  # noqa: PLC0415

        async def _apply(text: str, *, source: str, **_: object) -> str:
            """Fail every check of the caller's speech."""
            if source == "INPUT":
                message = "The check could not be completed."
                raise ApiError(message, status=503)
            return text

        monkeypatch.setattr(realtime, "apply_guardrail_to_text", _apply)
        fake_backend.script = [InputTranscript("this is a test")]

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "audio": {"input": {"transcription": {"model": "whisper-1"}}},
                    },
                }
            )
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            events = _drain(
                websocket, "conversation.item.input_audio_transcription.failed"
            )

        failed = events[-1]
        assert failed["type"] == "conversation.item.input_audio_transcription.failed"
        assert failed["content_index"] == 0
        assert failed["error"]["message"] == "The check could not be completed."

    def test_a_caller_who_asked_for_no_transcript_is_told_nothing(
        self,
        app_client: TestClient,
        fake_backend: type[_FakeModel],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The event only exists for a session that asked for transcription."""
        from stdapi import realtime  # noqa: PLC0415
        from stdapi.api_errors import ApiError  # noqa: PLC0415

        async def _apply(text: str, *, source: str, **_: object) -> str:
            """Fail every check of the caller's speech."""
            if source == "INPUT":
                message = "The check could not be completed."
                raise ApiError(message, status=503)
            return text

        monkeypatch.setattr(realtime, "apply_guardrail_to_text", _apply)
        fake_backend.script = [InputTranscript("this is a test")]

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            events = _drain(websocket, "error")

        kinds = [event["type"] for event in events]
        assert not [kind for kind in kinds if "transcription" in kind], kinds
        assert events[-1]["type"] == "error", events


class TestSessionBilling:
    """Every answer bills what it consumed, before the session is over.

    A session bills continuously while it is open, so the record cannot wait for
    the close: a connection dropped mid-conversation would take the whole bill
    with it. What ``response.done`` reports to the client is a different thing
    from what the deployment is billed for, and only the second one is money.

    Ref: stdapi/realtime.py:RealtimeSession._record_usage
         stdapi/monitoring.py:flush_usage_log_event
    """

    @staticmethod
    def _turn(report: UsageReport) -> list[Any]:
        """Script one answered turn ending on *report*'s running totals."""
        return [
            ResponseStarted(),
            OutputTranscript("Sure thing."),
            OutputAudio(b"\x01\x00\x02\x00"),
            report,
            ResponseFinished(),
        ]

    def test_each_answer_is_billed_while_the_session_is_still_open(
        self,
        app_client: TestClient,
        fake_backend: type[_FakeModel],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Two answers record two usage entries, the second one a delta only."""
        fake_backend.script = [
            *self._turn(
                UsageReport(
                    input_speech_tokens=197,
                    input_text_tokens=348,
                    output_speech_tokens=52,
                    output_text_tokens=26,
                    total_tokens=623,
                )
            ),
            *self._turn(
                UsageReport(
                    input_speech_tokens=297,
                    input_text_tokens=349,
                    output_speech_tokens=152,
                    output_text_tokens=27,
                    total_tokens=825,
                )
            ),
        ]

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            _drain(websocket, "response.done")
            _drain(websocket, "response.done")
            # Round-trips past the flush, which the answer's own events precede.
            websocket.send_json({"type": "input_audio_buffer.clear"})
            websocket.receive_json()
            recorded = logged_usage_entries(capsys.readouterr().out, model=_MODEL)

        assert len(recorded) == 2, recorded
        first, second = recorded
        assert first["region"] == "us-east-1"
        assert first["input_tokens"] == 545
        assert first["output_tokens"] == 78
        assert first["total_tokens"] == 623
        assert first["input_tokens_by_spec"] == {"speech": 197}
        assert first["output_tokens_by_spec"] == {"speech": 52}
        assert second["input_tokens"] == 101
        assert second["output_tokens"] == 101
        assert second["input_tokens_by_spec"] == {"speech": 100}
        assert second["output_tokens_by_spec"] == {"speech": 100}

    def test_a_session_that_reports_no_usage_records_nothing(
        self,
        app_client: TestClient,
        fake_backend: type[_FakeModel],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """An answer the backend never metered is not invented a bill for."""
        fake_backend.script = [
            ResponseStarted(),
            OutputTranscript("Hi."),
            ResponseFinished(),
        ]

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            _drain(websocket, "response.done")
            websocket.send_json({"type": "input_audio_buffer.clear"})
            websocket.receive_json()
            recorded = logged_usage_entries(capsys.readouterr().out, model=_MODEL)

        assert recorded == []


class TestBargeInAndCancellation:
    """What ``response.done`` says when an answer did not run to its end.

    Ref: https://developers.openai.com/api/reference/resources/realtime
         stdapi/realtime.py:RealtimeSession._cancel_response
    """

    def test_an_answer_the_caller_spoke_over_reports_incomplete(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """An interrupted answer is not reported as a completed one."""
        fake_backend.script = [
            ResponseStarted(),
            OutputTranscript("Sure thi"),
            ResponseFinished(interrupted=True),
        ]

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            events = _drain(websocket, "response.done")

        assert events[-1]["response"]["status"] == "incomplete", events[-1]

    def test_a_cancelled_answer_stops_being_reported(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """What the model keeps speaking is dropped, not reported as a new answer.

        The backend is not told to stop, so audio keeps arriving after the
        cancellation; reporting it would open a second response the client never
        asked for, and replay the answer it just cancelled.
        """
        gate, replayed = _Gate(), _Gate()
        fake_backend.script = [
            ResponseStarted(),
            OutputTranscript("Sure thing,"),
            gate,
            OutputAudio(b"\x01\x00\x02\x00"),
            OutputTranscript(" and also this."),
            ResponseFinished(),
            replayed,
        ]

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            _drain(websocket, "response.output_audio_transcript.delta")
            websocket.send_json({"type": "response.cancel"})
            done = _drain(websocket, "response.done")[-1]
            gate.release()
            assert replayed.reached.wait(_GATE_TIMEOUT), "the backend never replayed"
            websocket.send_json({"type": "input_audio_buffer.clear"})
            after = websocket.receive_json()

        assert done["response"]["status"] == "cancelled", done
        assert after["type"] == "input_audio_buffer.cleared", after


class TestResponseObject:
    """The response object, on both of the events carrying it.

    A voice application validates the frame it was given, and a model field
    declared without a default is required: Pipecat declares ``status_details``
    that way, so a missing key kills its reader task on the very first
    ``response.created`` and the session connects without ever speaking. Every
    field upstream always sends is therefore present, whatever its value.

    Ref: https://developers.openai.com/api/reference/resources/realtime/server-events
         openai.types.realtime.realtime_response.RealtimeResponse
         stdapi/realtime.py:RealtimeSession._response_view
    """

    #: Every field upstream puts on a response object, whichever event carries it.
    _FIELDS = frozenset(
        {
            "id",
            "object",
            "status",
            "status_details",
            "conversation_id",
            "output_modalities",
            "max_output_tokens",
            "audio",
            "metadata",
            "output",
            "usage",
        }
    )

    @staticmethod
    def _both(websocket: Any) -> tuple[dict[str, Any], dict[str, Any]]:  # noqa: ANN401
        """Answer one turn and return what its two response events reported.

        Args:
            websocket: The open connection, already past ``session.created``.

        Returns:
            The response object of ``response.created`` and of ``response.done``.
        """
        websocket.send_json({"type": "response.create"})
        events = _drain(websocket, "response.done")
        by_kind = {event["type"]: event for event in events}
        kinds = [event["type"] for event in events]
        assert "response.created" in by_kind, kinds
        assert "response.done" in by_kind, kinds
        return by_kind["response.created"]["response"], by_kind["response.done"][
            "response"
        ]

    def test_both_response_events_carry_every_field_upstream_sends(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """One answer is described by one object, rendered the same way twice.

        Only the status, the output and the usage move between the two events;
        a field present on one and absent from the other is the same answer read
        as two different ones.
        """
        fake_backend.script = list(_ANSWER_SCRIPT)

        with _connect(app_client) as websocket:
            websocket.receive_json()
            created, done = self._both(websocket)

        assert set(created) == self._FIELDS, sorted(created)
        assert set(done) == self._FIELDS, sorted(done)
        assert created["id"] == done["id"]
        assert created["status"] == "in_progress"
        assert done["status"] == "completed"
        assert {
            field: value
            for field, value in created.items()
            if field not in {"status", "output", "usage"}
        } == {
            field: value
            for field, value in done.items()
            if field not in {"status", "output", "usage"}
        }

    def test_an_answer_that_ran_to_its_end_says_so_with_an_explicit_null(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """``status_details`` is null while in progress and once completed.

        Measured against the upstream API: the key is sent on both events with
        a null value, never omitted.
        """
        fake_backend.script = list(_ANSWER_SCRIPT)

        with _connect(app_client) as websocket:
            websocket.receive_json()
            created, done = self._both(websocket)

        assert "status_details" in created, created
        assert created["status_details"] is None, created
        assert "status_details" in done, done
        assert done["status_details"] is None, done

    def test_an_answer_the_caller_spoke_over_reports_why_it_stopped(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """A barge-in names the turn that ended the answer, not just its status."""
        fake_backend.script = [*_ANSWER_SCRIPT[:-1], ResponseFinished(interrupted=True)]

        with _connect(app_client) as websocket:
            websocket.receive_json()
            _, done = self._both(websocket)

        assert done["status"] == "incomplete", done
        assert done["status_details"] == {
            "type": "incomplete",
            "reason": "turn_detected",
        }, done

    def test_a_cancelled_answer_reports_the_client_that_cancelled_it(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """``response.cancel`` is the reason the answer ended, and is named."""
        gate = _Gate()
        fake_backend.script = [
            ResponseStarted(),
            OutputTranscript("Sure thing,"),
            gate,
            ResponseFinished(),
        ]

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            _drain(websocket, "response.output_audio_transcript.delta")
            websocket.send_json({"type": "response.cancel"})
            done = _drain(websocket, "response.done")[-1]["response"]
            gate.release()

        assert done["status"] == "cancelled", done
        assert done["status_details"] == {
            "type": "cancelled",
            "reason": "client_cancelled",
        }, done

    def test_an_answer_reports_the_session_it_was_produced_under(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """The audio, the modalities and the token cap are the effective ones.

        A client reads these back to know what it is holding -- which format the
        audio it receives is in, and whether speech is coming at all.
        """
        fake_backend.script = list(_ANSWER_SCRIPT)

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "max_output_tokens": 256,
                        "audio": {
                            "output": {
                                "format": {"type": "audio/pcmu"},
                                "voice": "matthew",
                            }
                        },
                    },
                }
            )
            websocket.receive_json()
            created, done = self._both(websocket)

        assert done["output_modalities"] == ["audio"], done
        assert done["max_output_tokens"] == 256, done
        assert done["audio"] == {
            "output": {"format": {"type": "audio/pcmu"}, "voice": "matthew"}
        }, done
        assert created["audio"] == done["audio"], created

    def test_a_text_only_session_reports_the_modality_it_answered_in(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """``output_modalities`` is what the answer used, not what was defaulted."""
        fake_backend.script = list(_ANSWER_SCRIPT)

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {"type": "realtime", "output_modalities": ["text"]},
                }
            )
            websocket.receive_json()
            _, done = self._both(websocket)

        assert done["output_modalities"] == ["text"], done

    def test_an_unbounded_answer_reports_the_cap_upstream_reports(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """A session naming no cap answers ``inf``, the value upstream sends."""
        fake_backend.script = list(_ANSWER_SCRIPT)

        with _connect(app_client) as websocket:
            websocket.receive_json()
            _, done = self._both(websocket)

        assert done["max_output_tokens"] == "inf", done

    def test_every_answer_of_a_session_names_the_same_conversation(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """The session holds one conversation, and each answer is added to it.

        A client grouping answers by ``conversation_id`` otherwise reads two
        turns of one call as two unrelated conversations.
        """
        fake_backend.script = [*_ANSWER_SCRIPT, *_ANSWER_SCRIPT]

        with _connect(app_client) as websocket:
            websocket.receive_json()
            _, first = self._both(websocket)
            second = _drain(websocket, "response.done")[-1]["response"]
        with _connect(app_client) as other:
            other.receive_json()
            _, elsewhere = self._both(other)

        assert first["conversation_id"].startswith("conv_"), first
        assert second["conversation_id"] == first["conversation_id"], second
        assert elsewhere["conversation_id"] != first["conversation_id"], elsewhere

    def test_no_metadata_is_reported_because_none_can_be_attached(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """``metadata`` is always null: a response carries none to echo back."""
        fake_backend.script = list(_ANSWER_SCRIPT)

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {"type": "response.create", "response": {"metadata": {"call": "42"}}}
            )
            events = _drain(websocket, "response.done")

        done = events[-1]["response"]
        assert "metadata" in done, done
        assert done["metadata"] is None, done


class TestSecretCarriedSessions:
    """What an ephemeral client secret opens, and what it cannot change.

    The browser flow is the reason the secret exists: it is presented on the
    socket by a client that holds nothing else, so what it grants is the whole
    security boundary of the route.

    Ref: https://developers.openai.com/api/reference/resources/realtime/subresources/client_secrets/methods/create
         stdapi/realtime.py:_open_session
    """

    @staticmethod
    def _bearer(secret: str) -> dict[str, str]:
        """Present *secret* the way a server-side SDK does."""
        return {"Authorization": f"Bearer {secret}"}

    @pytest.mark.usefixtures("fake_backend")
    def test_a_secret_opens_the_session_it_carries_without_a_model_query(
        self, app_client: TestClient
    ) -> None:
        """The carried configuration is the session, model included."""
        value, _ = mint_client_secret(
            RealtimeSessionConfig(instructions="Be brief.", model=_MODEL), 600
        )

        with app_client.websocket_connect(
            "/v1/realtime", headers=self._bearer(value)
        ) as websocket:
            created = websocket.receive_json()

        assert created["type"] == "session.created"
        assert created["session"]["instructions"] == "Be brief."
        assert created["session"]["model"] == _MODEL

    def test_an_expired_secret_is_refused_as_a_credential(
        self,
        app_client: TestClient,
        fake_backend: type[_FakeModel],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unusable secret falls through to authentication, which refuses it."""
        assert fake_backend is not None
        _arm_authentication(monkeypatch)
        value, _ = mint_client_secret(RealtimeSessionConfig(model=_MODEL), -1)

        with app_client.websocket_connect(
            f"/v1/realtime?model={_MODEL}", headers=self._bearer(value)
        ) as websocket:
            error = websocket.receive_json()

        assert error["type"] == "error"
        assert error["error"]["code"] == "invalid_api_key"
        assert fake_backend.opened == [], "a refused connection opened a session"

    def test_a_wrong_credential_never_opens_a_backend_session(
        self,
        app_client: TestClient,
        fake_backend: type[_FakeModel],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Refusal happens before anything billable is opened."""
        _arm_authentication(monkeypatch)

        with app_client.websocket_connect(
            f"/v1/realtime?model={_MODEL}", headers=self._bearer("not-the-key")
        ) as websocket:
            error = websocket.receive_json()

        assert error["error"]["code"] == "invalid_api_key", error
        assert fake_backend.opened == [], "a refused connection opened a session"

    @pytest.mark.usefixtures("fake_backend")
    def test_the_query_model_overrides_the_secret_by_default(
        self, app_client: TestClient
    ) -> None:
        """Upstream behaviour: the carried session is a default, not a constraint."""
        value, _ = mint_client_secret(
            RealtimeSessionConfig(model="another.model-v1:0"), 600
        )

        with app_client.websocket_connect(
            f"/v1/realtime?model={_MODEL}", headers=self._bearer(value)
        ) as websocket:
            created = websocket.receive_json()

        assert created["type"] == "session.created"
        assert created["session"]["model"] == _MODEL

    def test_a_pinned_secret_refuses_another_model(
        self,
        app_client: TestClient,
        fake_backend: type[_FakeModel],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With overrides off, the query string cannot repoint a minted secret."""
        monkeypatch.setattr(SETTINGS, "realtime_allow_session_override", False)
        value, _ = mint_client_secret(
            RealtimeSessionConfig(model="another.model-v1:0"), 600
        )

        with app_client.websocket_connect(
            f"/v1/realtime?model={_MODEL}", headers=self._bearer(value)
        ) as websocket:
            error = websocket.receive_json()

        assert error["type"] == "error", error
        assert "model" in error["error"]["message"]
        assert fake_backend.opened == []

    @pytest.mark.usefixtures("fake_backend")
    def test_a_pinned_secret_refuses_new_instructions(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With overrides off, the operator's instructions survive session.update."""
        monkeypatch.setattr(SETTINGS, "realtime_allow_session_override", False)
        value, _ = mint_client_secret(
            RealtimeSessionConfig(
                model=_MODEL, instructions="Only answer about the product."
            ),
            600,
        )

        with app_client.websocket_connect(
            "/v1/realtime", headers=self._bearer(value)
        ) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {"type": "realtime", "instructions": "Anything goes."},
                }
            )
            error = websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "audio": {"output": {"voice": "sage"}},
                    },
                }
            )
            updated = websocket.receive_json()

        assert error["type"] == "error", error
        assert updated["type"] == "session.updated", updated
        assert updated["session"]["instructions"] == "Only answer about the product."

    @pytest.mark.usefixtures("fake_backend")
    def test_the_configuration_headers_are_ignored_for_a_secret(
        self, app_client: TestClient
    ) -> None:
        """A credential held by an untrusted client cannot redirect the guardrail."""
        applied: list[Any] = []
        from stdapi import realtime  # noqa: PLC0415

        value, _ = mint_client_secret(RealtimeSessionConfig(model=_MODEL), 600)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(realtime, "set_guardrail_configuration", applied.append)
            with app_client.websocket_connect(
                "/v1/realtime",
                headers={
                    **self._bearer(value),
                    "X-Amzn-Bedrock-GuardrailIdentifier": "permissive",
                },
            ) as websocket:
                created = websocket.receive_json()

        assert created["type"] == "session.created"
        assert applied == [], "request headers were honoured for a client secret"


class TestSessionBounds:
    """A slow or hostile client must not grow the server's memory.

    Ref: stdapi/realtime.py:RealtimeSession._append_audio
    """

    @pytest.mark.usefixtures("fake_backend")
    def test_an_oversized_event_is_refused_and_the_session_survives(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One event past the cap answers an error rather than being buffered."""
        from stdapi import realtime  # noqa: PLC0415

        monkeypatch.setattr(realtime, "_MAX_EVENT_BYTES", 256)

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(bytes(512)).decode(),
                }
            )
            error = websocket.receive_json()
            websocket.send_json({"type": "input_audio_buffer.clear"})
            cleared = websocket.receive_json()

        assert error["type"] == "error"
        assert "too large" in error["error"]["message"]
        assert cleared["type"] == "input_audio_buffer.cleared"

    def test_audio_buffered_without_a_commit_is_bounded(
        self,
        app_client: TestClient,
        fake_backend: type[_FakeModel],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Past the cap the append is refused, and clearing lets the caller on."""
        from stdapi import realtime  # noqa: PLC0415

        monkeypatch.setattr(realtime, "_MAX_BUFFERED_AUDIO_BYTES", len(_SILENT_FRAME))
        chunk = base64.b64encode(_SILENT_FRAME).decode()

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "audio": {"input": {"turn_detection": None}},
                    },
                }
            )
            websocket.receive_json()
            websocket.send_json({"type": "input_audio_buffer.append", "audio": chunk})
            websocket.send_json({"type": "input_audio_buffer.append", "audio": chunk})
            error = websocket.receive_json()
            websocket.send_json({"type": "input_audio_buffer.clear"})
            websocket.receive_json()
            websocket.send_json({"type": "input_audio_buffer.append", "audio": chunk})
            websocket.send_json({"type": "input_audio_buffer.commit"})
            committed = websocket.receive_json()

        assert error["type"] == "error"
        assert "commit" in error["error"]["message"]
        assert committed["type"] == "input_audio_buffer.committed"
        assert bytes(fake_backend.opened[0].audio) == _SILENT_FRAME


class TestMalformedEvents:
    """A valid-JSON event of the wrong shape is answered, not fatal.

    Every one of these closed the socket with an unhandled ``TypeError`` and a
    critical log entry, repeatable at will by any authenticated client.

    Ref: stdapi/realtime.py:RealtimeSession._apply
    """

    @pytest.mark.parametrize(
        "event",
        [
            {"type": "session.update", "session": "x"},
            {"type": "conversation.item.create", "item": {"content": 5}},
            {
                "type": "conversation.item.create",
                "item": {"content": [{"type": "input_text", "text": 123}]},
            },
        ],
    )
    @pytest.mark.usefixtures("fake_backend")
    def test_a_wrongly_shaped_event_answers_an_error(
        self, app_client: TestClient, event: dict[str, Any]
    ) -> None:
        """The session answers ``error`` and keeps serving."""
        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(event)
            error = websocket.receive_json()
            websocket.send_json({"type": "input_audio_buffer.clear"})
            cleared = websocket.receive_json()

        assert error["type"] == "error", error
        assert error["error"]["type"] == "invalid_request_error"
        assert cleared["type"] == "input_audio_buffer.cleared"

    @pytest.mark.usefixtures("fake_backend")
    def test_one_nested_setting_does_not_reset_its_siblings(
        self, app_client: TestClient
    ) -> None:
        """``session.update`` merges into the configuration instead of replacing it."""
        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "audio": {"output": {"voice": "sage"}},
                    },
                }
            )
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "audio": {"input": {"turn_detection": None}},
                    },
                }
            )
            updated = websocket.receive_json()

        assert updated["type"] == "session.updated", updated
        assert updated["session"]["audio"]["output"]["voice"] == "sage"
        assert updated["session"]["audio"]["input"].get("turn_detection") is None


class TestGuardrailedSession:
    """A configured guardrail reads everything the session carries.

    The route has no native guardrail integration, so every turn is checked
    through ApplyGuardrail: what the caller said, what was written, and what was
    answered. Nothing else on the session is text.

    Ref: stdapi/aws_bedrock.py:apply_guardrail_to_text
         stdapi/realtime.py:RealtimeSession._finish_response
    """

    @staticmethod
    def _record(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
        """Replace the guardrail with a recorder returning masked text."""
        from stdapi import realtime  # noqa: PLC0415

        checked: list[tuple[str, str]] = []

        async def _apply(text: str, *, source: str, **_: object) -> str:
            """Record one check and mask what a guardrail would mask."""
            checked.append((source, text))
            return text.replace("secret", "***")

        monkeypatch.setattr(realtime, "apply_guardrail_to_text", _apply)
        return checked

    def test_the_caller_speech_and_the_answer_are_both_checked(
        self,
        app_client: TestClient,
        fake_backend: type[_FakeModel],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The transcript goes in as INPUT, the answer out as OUTPUT."""
        checked = self._record(monkeypatch)
        fake_backend.script = [
            InputTranscript("tell me a secret"),
            ResponseStarted(),
            OutputTranscript("a secret then"),
            ResponseFinished(),
        ]

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            events = _drain(websocket, "response.done")

        assert ("INPUT", "tell me a secret") in checked, checked
        assert ("OUTPUT", "a secret then") in checked, checked
        done = events[-1]["response"]
        assert done["output"][0]["content"][0]["transcript"] == "a *** then"

    def test_the_caller_speech_is_checked_without_transcription_events(
        self,
        app_client: TestClient,
        fake_backend: type[_FakeModel],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A caller who asked for no transcript is still not unchecked."""
        checked = self._record(monkeypatch)
        fake_backend.script = [
            InputTranscript("tell me a secret"),
            ResponseStarted(),
            OutputTranscript("no"),
            ResponseFinished(),
        ]

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            events = _drain(websocket, "response.done")

        kinds = [event["type"] for event in events]
        assert not [kind for kind in kinds if "transcription" in kind], kinds
        assert ("INPUT", "tell me a secret") in checked, checked

    def test_a_blocked_answer_ends_the_session(
        self,
        app_client: TestClient,
        fake_backend: type[_FakeModel],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An intervention on the answer is terminal, as it cannot be undone."""
        from stdapi import realtime  # noqa: PLC0415
        from stdapi.aws_bedrock import GuardrailInterventionError  # noqa: PLC0415

        blocked = "The content was blocked by the configured guardrail."

        async def _apply(text: str, *, source: str, **_: object) -> str:
            """Block every answer, and let everything the caller says through."""
            if source == "OUTPUT":
                raise GuardrailInterventionError(blocked)
            return text

        monkeypatch.setattr(realtime, "apply_guardrail_to_text", _apply)
        fake_backend.script = [
            ResponseStarted(),
            OutputTranscript("something blocked"),
            ResponseFinished(),
        ]

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            events = _drain(websocket, "response.done")

        assert events[-1]["type"] == "error", events
        assert "guardrail" in events[-1]["error"]["message"].lower()


class TestSessionExpiry:
    """The duration cap closes the session rather than leaking its tasks.

    Ref: stdapi/realtime.py:RealtimeSession._serve
    """

    def test_a_session_past_its_cap_is_closed(
        self,
        app_client: TestClient,
        fake_backend: type[_FakeModel],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The socket closes on ``session_expired`` once the cap is reached."""
        monkeypatch.setattr(fake_backend, "MAX_SESSION_SECONDS", 0.2)

        with _connect(app_client) as websocket:
            websocket.receive_json()
            with pytest.raises(WebSocketDisconnect) as raised:
                _read_until_closed(websocket)

        assert raised.value.reason == "session_expired", raised.value


class TestAudioFormatSupport:
    """A format the model cannot carry is refused before the conversation opens.

    Ref: stdapi/realtime.py:RealtimeSession._check_format
    """

    def test_a_rate_the_model_does_not_accept_is_refused(
        self,
        app_client: TestClient,
        fake_backend: type[_FakeModel],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """G.711 on a model that only speaks 24 kHz fails at once, not mid-turn."""
        monkeypatch.setattr(fake_backend, "INPUT_SAMPLE_RATES", frozenset({24000}))

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "audio": {"input": {"format": {"type": "audio/pcmu"}}},
                    },
                }
            )
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            error = websocket.receive_json()

        assert error["type"] == "error", error
        assert "audio/pcmu" in error["error"]["message"]
        assert fake_backend.opened == []

    def test_a_model_declaring_no_rates_accepts_every_format(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """An empty declaration is "unknown", not "nothing is supported"."""
        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "audio": {"input": {"format": {"type": "audio/pcma"}}},
                    },
                }
            )
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            websocket.send_json({"type": "input_audio_buffer.clear"})
            cleared = websocket.receive_json()

        assert cleared["type"] == "input_audio_buffer.cleared"
        assert len(fake_backend.opened) == 1


class TestShutdownDrain:
    """A deployment on its way out opens no new session, and says so.

    The latch is a module global, so a lifespan that starts after one that
    stopped -- a second app in one process -- must not inherit it.

    Ref: stdapi/realtime.py:close_realtime_sessions
         stdapi/main.py:lifespan
    """

    @pytest.fixture(autouse=True)
    def _restore(self) -> Iterator[None]:
        """Leave the module global as the test found it."""
        yield
        open_realtime_sessions()

    def test_a_stopping_deployment_refuses_a_new_session(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """The refusal is an error event on the accepted socket, not a status."""
        close_realtime_sessions()

        with _connect(app_client) as websocket:
            error = websocket.receive_json()

        assert error["type"] == "error", error
        assert "shutting down" in error["error"]["message"]
        assert fake_backend.opened == []

    @pytest.mark.usefixtures("fake_backend")
    def test_a_lifespan_that_starts_again_serves_sessions(
        self, app_client: TestClient
    ) -> None:
        """Starting the app again clears the latch the last stop set."""
        close_realtime_sessions()
        open_realtime_sessions()

        with _connect(app_client) as websocket:
            created = websocket.receive_json()

        assert created["type"] == "session.created"


class TestOfficialEventTypes:
    """Every server event parses against the official client's own event models.

    A voice application does not read these events itself: its SDK validates
    each frame against the ``openai`` package's Realtime models and surfaces a
    rejected one as an error instead of as the event it was. A shape invented
    here is therefore not a cosmetic difference but the event never arriving --
    the assistant turn missing from the conversation, or the caller's transcript
    dropped whole.

    Ref: https://developers.openai.com/api/reference/resources/realtime/server-events
         stdapi/realtime.py:_item_body
    """

    @staticmethod
    def _validate(events: list[dict[str, Any]]) -> None:
        """Fail on the first event the official models refuse.

        Args:
            events: Every event the session sent, in order.
        """
        from pydantic import ValidationError  # noqa: PLC0415

        for event in events:
            try:
                _SERVER_EVENT.validate_python(event)
            except ValidationError as error:
                # One error per member of a 45-way union; only ours is readable.
                refused = [
                    line
                    for line in str(error).splitlines()
                    if event["type"].replace(".", "-") in line.lower()
                ]
                pytest.fail(
                    f"{event['type']} was refused: {event}\n" + "\n".join(refused)
                )

    def test_a_spoken_turn_parses_whole(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """Every event of a spoken turn, its item lifecycle included, parses."""
        fake_backend.script = [
            SpeechStarted(120),
            SpeechStopped(2480),
            UsageReport(input_speech_tokens=197, total_tokens=197),
            InputTranscript("this is a test"),
            *_ANSWER_SCRIPT,
        ]

        with _connect(app_client) as websocket:
            events = [websocket.receive_json()]
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        "audio": {
                            "input": {
                                "turn_detection": None,
                                "transcription": {"model": "whisper-1"},
                            }
                        },
                    },
                }
            )
            events.append(websocket.receive_json())
            websocket.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Hello there."}],
                    },
                }
            )
            events += _drain(websocket, "conversation.item.done")
            websocket.send_json({"type": "input_audio_buffer.append", "audio": _FRAME})
            websocket.send_json({"type": "input_audio_buffer.commit"})
            events += _drain(websocket, "response.done")
            item_id = next(
                event["item"]["id"]
                for event in events
                if event["type"] == "response.output_item.added"
            )
            # Drained per answer rather than one frame each: the two halves
            # interleave, so what the commit produced may still be unread.
            for sent, answer in (
                (
                    {
                        "type": "conversation.item.truncate",
                        "item_id": item_id,
                        "content_index": 0,
                        "audio_end_ms": 40,
                    },
                    "conversation.item.truncated",
                ),
                (
                    {"type": "conversation.item.retrieve", "item_id": item_id},
                    "conversation.item.retrieved",
                ),
                (
                    {"type": "conversation.item.delete", "item_id": item_id},
                    "conversation.item.deleted",
                ),
                ({"type": "conversation.item.retrieve", "item_id": item_id}, "error"),
            ):
                websocket.send_json(sent)
                events += _drain(websocket, answer)

        kinds = {event["type"] for event in events}
        assert "conversation.item.input_audio_transcription.completed" in kinds, kinds
        assert "conversation.item.truncated" in kinds, kinds
        assert "error" in kinds, kinds
        self._validate(events)

    def test_a_text_only_turn_parses_whole(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """A session answering in text writes its item as ``output_text``."""
        fake_backend.script = list(_ANSWER_SCRIPT)

        with _connect(app_client) as websocket:
            events = [websocket.receive_json()]
            websocket.send_json(
                {
                    "type": "session.update",
                    "session": {"type": "realtime", "output_modalities": ["text"]},
                }
            )
            events.append(websocket.receive_json())
            websocket.send_json({"type": "response.create"})
            events += _drain(websocket, "response.done")

        answered = next(
            event for event in events if event["type"] == "conversation.item.done"
        )
        assert answered["item"]["content"] == [
            {"type": "output_text", "text": "Sure thing."}
        ], answered
        self._validate(events)


class TestBackendTeardown:
    """A closed session leaves nothing behind for the next thing to trip over.

    The backend's stream is closed before its reader is stopped: a reader
    cancelled while a chunk is still outstanding leaves the transport
    completing a result nobody holds any more, which is reported half a second
    after the socket closed -- in whatever the process runs next, never in the
    session that caused it.

    Ref: stdapi/realtime.py:RealtimeSession._stop_backend
    """

    def test_the_backend_reader_ends_on_its_closed_stream(
        self, app_client: TestClient, fake_backend: type[_FakeModel]
    ) -> None:
        """The reader observes the close instead of being cancelled mid-read."""
        fake_backend.script = list(_ANSWER_SCRIPT)

        with _connect(app_client) as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "response.create"})
            _drain(websocket, "response.done")

        session = fake_backend.opened[0]
        assert not session.reader_cancelled, (
            "the backend reader was cancelled instead of ending on its closed stream"
        )

    async def test_a_cancelled_teardown_still_lets_the_reader_end(self) -> None:
        """A teardown cancelled mid-stop leaves the reader to end on the close.

        The stop is awaited through a shield: when the session is torn down
        because something upstream was cancelled (the client vanished and the
        server cancelled the connection's scope), the cancellation must not
        propagate into the reader, which its closed stream ends on its own.

        Ref: stdapi/realtime.py:RealtimeSession._stop_backend
        """
        session = RealtimeSession.__new__(RealtimeSession)
        closed = asyncio.Event()
        cancelled = False

        async def _read() -> None:
            """Wait for the stream to close, recording a cancellation instead."""
            nonlocal cancelled
            try:
                await closed.wait()
            except asyncio.CancelledError:
                cancelled = True
                raise

        reader = asyncio.ensure_future(_read())
        session._backend_task = reader  # noqa: SLF001
        teardown = asyncio.ensure_future(session._stop_backend())  # noqa: SLF001
        await asyncio.sleep(0)  # The teardown reaches its shielded wait.
        teardown.cancel()
        closed.set()
        with suppress(asyncio.CancelledError):
            await teardown
        await reader
        assert not cancelled, "the outer cancellation reached the backend reader"
