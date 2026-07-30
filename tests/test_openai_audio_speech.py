"""Tests for the OpenAI /v1/audio/speech route served by Amazon Polly.

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://stdapi.ai/api_openai_audio_speech/
     stdapi/routes/openai_audio_speech.py:create_speech
"""

import json
from base64 import b64decode
from datetime import UTC, datetime
from inspect import AGEN_CLOSED, getasyncgenstate
from typing import TYPE_CHECKING, Any

import magic
import pytest
from openai import BadRequestError, NotFoundError, OpenAI
from starlette.requests import Request

from stdapi.api_errors import ApiError
from stdapi.models.audio import TTSResponse
from stdapi.monitoring import REQUEST, REQUEST_ID, REQUEST_LOG, EventLog
from stdapi.routes import openai_audio_speech
from stdapi.types.openai_audio import SpeechCreateParams
from tests.conftest import SAMPLES_DIR, logged_usage_entries

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from starlette.testclient import TestClient


def _assert_is_mp3(audio: bytes) -> None:
    """Assert the payload is a bare MPEG layer III elementary stream."""
    signature = str(magic.from_buffer(audio))
    assert "MPEG ADTS, layer III" in signature, f"not mp3 audio: {signature}"


def _flac_sample_rate(audio: bytes) -> int:
    """Return the sample rate advertised by a FLAC stream's STREAMINFO block."""
    assert audio.startswith(b"fLaC"), "not a FLAC stream"
    # STREAMINFO starts at byte 8; its sample rate is the 20 bits at bit offset 80.
    return (audio[18] << 12) | (audio[19] << 4) | (audio[20] >> 4)


def _sse_events(body: bytes) -> list[dict[str, Any]]:
    """Return the JSON payloads of the ``data:`` lines of an SSE body."""
    return [
        json.loads(line.removeprefix("data:"))
        for line in body.decode().splitlines()
        if line.startswith("data:")
    ]


class TestAudioSpeech:
    """/v1/audio/speech: the OpenAI speech contract mapped onto Polly SynthesizeSpeech.

    Ref: https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html
         stdapi/models/audio/amazon_polly.py:AudioModel.tts
    """

    def test_basic_speech_generation(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """Speech generated with only the required parameters is an mp3 stream.

        ``response_format`` defaults to ``mp3``, which Polly synthesizes
        natively, so the body is served straight through with the
        ``audio/mpeg`` content type and no in-process re-encode.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_audio_speech.py:create_speech
        """
        response = openai_client.audio.speech.create(
            model=speech_standard_model, voice="alloy", input="Test."
        )

        audio_data = response.content
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0
        assert response.response.headers.get("content-type") == "audio/mpeg"
        _assert_is_mp3(audio_data)

    def test_basic_speech_long_generation(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """A 3000-character input is synthesized as a single mp3 stream.

        3000 billed characters is Polly's SynthesizeSpeech maximum, and the
        Latin sample exercises the language-detection path: Comprehend
        detects a language with no Polly voice, so the en-US fallback voice
        is used instead of failing the request.

        Ref: https://docs.aws.amazon.com/polly/latest/dg/limits.html
             stdapi/models/audio/amazon_polly.py:_select_voice
        """
        with (SAMPLES_DIR / "lorem_ipsum.txt").open() as file:
            input_text = file.read(3000)  # SynthesizeSpeech characters limit

        response = openai_client.audio.speech.create(
            model=speech_standard_model, voice="alloy", input=input_text
        )

        audio_data = response.content
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0
        assert response.response.headers.get("content-type") == "audio/mpeg"
        _assert_is_mp3(audio_data)

    @pytest.mark.parametrize("sample_rate", ["8000", "24000"])
    def test_speech_with_extra_polly_sample_rate(
        self,
        openai_client: OpenAI,
        speech_standard_model: str,
        use_official_api: bool,
        sample_rate: str,
    ) -> None:
        """The Polly ``SampleRate`` extra parameter sets the encoded output rate.

        Polly has no FLAC output, so the gateway re-encodes with ffmpeg: at
        8 kHz the source stays pcm, while 24 kHz exceeds Polly's 16 kHz pcm
        cap and is synthesized as Ogg Vorbis instead. Either way the FLAC
        STREAMINFO must advertise the requested rate.

        Ref: https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html
             stdapi/models/audio/amazon_polly.py:AudioModel.tts
        """
        if use_official_api:
            pytest.skip("Amazon Polly is not available on the official OpenAI API")
        response = openai_client.audio.speech.create(
            model=speech_standard_model,
            voice="alloy",
            input="Test.",
            extra_body={"SampleRate": sample_rate},
            response_format="flac",  # To enforce FFMPEG encoding
        )

        audio_data = response.content
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0
        assert response.response.headers.get("content-type") == "audio/flac"
        assert _flac_sample_rate(audio_data) == int(sample_rate)

    def test_speech_pcm_default_resamples_to_24khz(
        self, openai_client: OpenAI, speech_standard_model: str, use_official_api: bool
    ) -> None:
        """Default pcm output follows OpenAI's 24 kHz contract, not Polly's 16 kHz.

        Polly pcm accepts only 8 kHz and 16 kHz, so the gateway resamples its
        16 kHz output to 24 kHz with ffmpeg: the default body carries ~1.5x
        the samples of the same text pinned to Polly's native rate.

        Ref: https://stdapi.ai/api_openai_audio_speech/
             stdapi/models/audio/amazon_polly.py:AudioModel.tts
        """
        if use_official_api:
            pytest.skip("Amazon Polly is not available on the official OpenAI API")

        default_response = openai_client.audio.speech.create(
            model=speech_standard_model,
            voice="alloy",
            input="Testing the default pcm sample rate.",
            response_format="pcm",
        )
        native_response = openai_client.audio.speech.create(
            model=speech_standard_model,
            voice="alloy",
            input="Testing the default pcm sample rate.",
            response_format="pcm",
            extra_body={"SampleRate": "16000"},
        )

        default_len = len(default_response.content)
        native_len = len(native_response.content)
        assert default_len > 0
        assert native_len > 0
        # 24 kHz output carries ~1.5x the samples of Polly's native 16 kHz output.
        assert 1.3 < default_len / native_len < 1.7
        assert default_response.response.headers.get("content-type") == "audio/pcm"

    def test_speech_marks_returns_json_lines(
        self, openai_client: OpenAI, speech_standard_model: str, use_official_api: bool
    ) -> None:
        """``SpeechMarkTypes`` returns ordered timing marks instead of audio.

        Polly generates no audio for a speech-marks request: the gateway
        forwards ``OutputFormat=json`` and streams the JSON lines through
        unchanged, labelled ``application/x-json-stream``.

        Ref: https://docs.aws.amazon.com/polly/latest/dg/speechmarks.html
             stdapi/models/audio/amazon_polly.py:AudioModel.tts
        """
        if use_official_api:
            pytest.skip("Amazon Polly is not available on the official OpenAI API")

        response = openai_client.audio.speech.create(
            model=speech_standard_model,
            voice="alloy",
            input="Hello, how are you?",
            extra_body={"SpeechMarkTypes": ["word"]},
        )

        assert (
            response.response.headers.get("content-type") == "application/x-json-stream"
        )
        lines = [line for line in response.content.decode().splitlines() if line]
        assert lines
        marks = [json.loads(line) for line in lines]
        for mark in marks:
            assert mark["type"] == "word"
            assert mark.keys() >= {"time", "type", "value"}
        times = [mark["time"] for mark in marks]
        assert times == sorted(times), "word marks must be ordered by time"
        values = [str(mark["value"]).lower() for mark in marks]
        assert "hello" in values
        assert "you" in values

    def test_speech_with_extra_invalid_parameter(
        self, openai_client: OpenAI, speech_standard_model: str, use_official_api: bool
    ) -> None:
        """Polly extra parameters are validated: bad type and unknown name are 400s.

        Extra body fields are parsed by ``_PollyExtraParams``, which forbids
        unknown keys, so both failures surface as the gateway's
        ``invalid_request_error`` naming the offending field.

        Ref: https://stdapi.ai/api_openai_audio_speech/
             stdapi/models/audio/amazon_polly.py:_PollyExtraParams
             stdapi/main.py:handle_validation_exception
        """
        if use_official_api:
            pytest.skip("Amazon Polly is not available on the official OpenAI API")

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.audio.speech.create(
                model=speech_standard_model,
                voice="alloy",
                input="Test.",
                extra_body={"SampleRate": "invalid_value"},
            )

        error_body = exc_info.value.body
        assert exc_info.value.status_code == 400
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert "SampleRate" in str(error_body["message"])
        assert "integer" in str(error_body["message"]).lower()

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.audio.speech.create(
                model=speech_standard_model,
                voice="alloy",
                input="Test.",
                extra_body={"Invalid": "invalid_value"},
            )

        error_body = exc_info.value.body
        assert exc_info.value.status_code == 400
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert "Invalid" in str(error_body["message"])
        assert "permitted" in str(error_body["message"]), (
            "unknown extra parameters must be rejected, not ignored"
        )

    @pytest.mark.parametrize(
        "voice",
        ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash", "sage", "coral"],
    )
    def test_all_voices_compatibility(
        self, openai_client: OpenAI, speech_standard_model: str, voice: str
    ) -> None:
        """Every OpenAI built-in voice name resolves to a usable Polly voice.

        Polly has no ``alloy``-style voices: the gateway maps each OpenAI name
        to a Polly voice of the matching gender for the detected language, so
        an unmapped name would be forwarded verbatim and rejected by Polly.

        Ref: https://developers.openai.com/api/docs/guides/text-to-speech#voice-options
             stdapi/models/audio/amazon_polly.py:_select_voice
        """
        response = openai_client.audio.speech.create(
            model=speech_standard_model, voice=voice, input="Test."
        )

        audio_data = response.content
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0
        assert response.response.headers.get("content-type") == "audio/mpeg"
        _assert_is_mp3(audio_data)

    @pytest.mark.parametrize("voice", ["Amy", "amy"])
    def test_polly_voices_compatibility(
        self,
        openai_client: OpenAI,
        speech_standard_model: str,
        voice: str,
        use_official_api: bool,
    ) -> None:
        """A native Polly voice ID is accepted case-insensitively.

        Voices are indexed by lowercase name, so ``amy`` resolves to the
        ``Amy`` voice ID; an unresolved name would reach Polly verbatim and
        come back as a 400 invalid-voice error.

        Ref: https://docs.aws.amazon.com/polly/latest/dg/available-voices.html
             stdapi/models/audio/amazon_polly.py:_select_voice
        """
        if use_official_api:
            pytest.skip(
                "Amazon Polly models are not available on the official OpenAI API"
            )
        response = openai_client.audio.speech.create(
            model=speech_standard_model, voice=voice, input="Test."
        )

        assert isinstance(response.content, bytes)
        assert len(response.content) > 0
        assert response.response.headers.get("content-type") == "audio/mpeg"
        _assert_is_mp3(response.content)

    @pytest.mark.parametrize("speed", [0.25, 1.0, 2.0])
    def test_speed_parameter_validation(
        self, openai_client: OpenAI, speech_standard_model: str, speed: float
    ) -> None:
        """In-range ``speed`` values are accepted and still yield valid mp3 audio.

        Polly has no speed parameter: any value other than 1.0 is applied by
        wrapping the text in ``<speak><prosody rate="N%">``, so a malformed
        rate would come back as Polly's InvalidSsmlException instead of audio.
        The gateway accepts 0.2 to 2.0, narrower than OpenAI's 0.25 to 4.0,
        because ``<prosody rate>`` is only partially supported per engine.

        Ref: https://docs.aws.amazon.com/polly/latest/dg/supportedtags.html
             stdapi/models/audio/amazon_polly.py:_prepare_text_for_speech
        """
        response = openai_client.audio.speech.create(
            model=speech_standard_model, voice="alloy", input="Test.", speed=speed
        )

        audio_data = response.content
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0
        assert response.response.headers.get("content-type") == "audio/mpeg"
        _assert_is_mp3(audio_data)

    @pytest.mark.parametrize(
        ("format_name", "content_type", "signature_check"),
        [
            ("mp3", "audio/mpeg", "MPEG ADTS, layer III"),
            ("opus", "audio/opus", "Opus audio"),
            ("aac", "audio/aac", "ADTS, AAC"),
            ("flac", "audio/flac", "FLAC"),
            ("wav", "audio/wav", "WAVE audio"),
            ("pcm", "audio/pcm", None),  # PCM may not have clear signature
        ],
    )
    def test_all_response_formats(
        self,
        openai_client: OpenAI,
        speech_standard_model: str,
        format_name: str,
        content_type: str,
        signature_check: str | None,
    ) -> None:
        """Each OpenAI response_format is returned with its own container and type.

        Polly emits only mp3, ogg_vorbis, ogg_opus and pcm, so wav, flac and
        aac are transcoded in-process from pcm; ``pcm`` itself is raw signed
        16-bit mono little-endian with no container header.

        Ref: https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html
             stdapi/media.py:encode_audio_stream
        """
        response = openai_client.audio.speech.create(
            model=speech_standard_model,
            voice="alloy",
            input="Test.",
            response_format=format_name,  # type: ignore[arg-type]
        )

        audio_data = response.content
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0
        assert response.response.headers.get("content-type") == content_type

        # Validate audio format signature when available
        if signature_check:
            magic_result = magic.from_buffer(audio_data)
            assert signature_check in str(magic_result)
        else:
            assert not audio_data.startswith((b"RIFF", b"fLaC", b"OggS", b"ID3")), (
                "pcm must be raw samples, without a container header"
            )
            assert len(audio_data) % 2 == 0, "16-bit samples come in byte pairs"

    def test_text_length_boundaries(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """Audio length scales with the input, down to a single character.

        The long case must be real words: Polly renders a repeated single letter
        ("A" * 128) as *less* audio than one "A" (1454 vs 2708 bytes), so a degenerate
        input cannot demonstrate scaling.

        Ref: https://docs.aws.amazon.com/polly/latest/dg/limits.html
             stdapi/types/openai_audio.py:SpeechCreateParams
        """
        # Test minimum length (1 character)
        response = openai_client.audio.speech.create(
            model=speech_standard_model, voice="alloy", input="A"
        )
        assert isinstance(response.content, bytes)
        assert len(response.content) > 0
        shortest = response.content
        _assert_is_mp3(shortest)

        # Test longer length
        response = openai_client.audio.speech.create(
            model=speech_standard_model,
            voice="alloy",
            input=(
                "This is a considerably longer sentence, spoken aloud, so that the "
                "synthesized audio is measurably longer than a single letter."
            ),
        )
        assert isinstance(response.content, bytes)
        assert len(response.content) > 0
        _assert_is_mp3(response.content)
        assert len(response.content) > 2 * len(shortest), (
            "a full sentence must synthesize measurably more audio than one letter"
        )

    def test_empty_input_error(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """An empty ``input`` is rejected as an ``invalid_request_error``.

        ``input`` has a minimum length of one character, so the request never
        reaches Polly: the failure comes from request validation and carries
        no OpenAI error ``code``.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_audio.py:SpeechCreateParams
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.audio.speech.create(
                model=speech_standard_model, voice="alloy", input=""
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert error_body["code"] is None
        # Validate error message mentions input validation
        error_message = str(error).lower()
        assert any(
            word in error_message
            for word in ["input", "required", "empty", "character"]
        )
        assert "input" in error_message, "the error must name the offending field"

    def test_invalid_model_error(self, openai_client: OpenAI) -> None:
        """An unknown model is rejected as a 404 ``model_not_found``.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             stdapi/api_errors.py:UnsupportedModelError
        """
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.audio.speech.create(
                model="invalid-nonexistent-model", voice="alloy", input="Test message."
            )

        error = exc_info.value
        assert error.status_code == 404
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert error_body["code"] == "model_not_found"
        error_message = str(error).lower()
        assert any(
            word in error_message
            for word in ["model", "invalid", "supported", "exist", "access"]
        )
        assert "invalid-nonexistent-model" in error_message, (
            "the error must echo the rejected model ID"
        )

    def test_invalid_voice_error(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """An unknown voice is rejected as an ``invalid_request_error``.

        The voice is a free-form string (Polly voice IDs are accepted too), so
        the rejection comes from Polly's ValidationException, which the
        gateway rewrites into a 400 listing the engine's available voices.

        Ref: https://docs.aws.amazon.com/polly/latest/dg/available-voices.html
             stdapi/models/audio/amazon_polly.py:_handle_polly_error
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.audio.speech.create(
                model=speech_standard_model,
                voice="invalid_voice_name",
                input="Test message.",
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert error_body["code"] is None
        error_message = str(error).lower()
        assert any(
            word in error_message
            for word in ["voice", "invalid", "supported", "input", "should"]
        )
        assert "invalid_voice_name" in error_message, (
            "the error must echo the rejected voice name"
        )

    @pytest.mark.parametrize("speed", [0.0, -1.0, 10.0])
    def test_invalid_speed_error(
        self, openai_client: OpenAI, speech_standard_model: str, speed: float
    ) -> None:
        """Out-of-range ``speed`` values are rejected as ``invalid_request_error``.

        The gateway bounds ``speed`` to 0.2 to 2.0, so all three values fail
        request validation before any Polly call and carry no error ``code``.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_audio.py:SpeechCreateParams
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.audio.speech.create(
                model=speech_standard_model,
                voice="alloy",
                input="Test message.",
                speed=speed,
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert error_body["code"] is None
        error_message = str(error).lower()
        assert any(
            word in error_message
            for word in ["speed", "range", "0.25", "4.0", "greater", "less"]
        )
        assert "speed" in error_message, "the error must name the offending field"

    def test_invalid_response_format_error(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """An unsupported ``response_format`` is rejected, listing the valid values.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_audio.py:SpeechCreateParams
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.audio.speech.create(
                model=speech_standard_model,
                voice="alloy",
                input="Test message.",
                response_format="invalid_format",  # type: ignore[arg-type]
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert error_body["code"] is None
        error_message = str(error).lower()
        assert any(
            word in error_message
            for word in ["format", "supported", "response", "input", "should"]
        )
        assert "mp3" in error_message, "the error must enumerate the accepted formats"
        assert "pcm" in error_message

    def test_missing_required_parameters(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """A null ``model``, ``voice`` or ``input`` fails validation naming that field.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/main.py:handle_validation_exception
        """
        # Test missing model
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.audio.speech.create(
                voice="alloy",
                input="Test message.",
                model=None,  # type: ignore[arg-type]
            )
        error_body = exc_info.value.body
        assert exc_info.value.status_code == 400
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert "model" in str(error_body["message"]).lower()

        # Test missing voice
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.audio.speech.create(
                model=speech_standard_model,
                input="Test message.",
                voice=None,  # type: ignore[arg-type]
            )
        error_body = exc_info.value.body
        assert exc_info.value.status_code == 400
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert "voice" in str(error_body["message"]).lower()

        # Test missing input
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.audio.speech.create(
                model=speech_standard_model,
                voice="alloy",
                input=None,  # type: ignore[arg-type]
            )
        error_body = exc_info.value.body
        assert exc_info.value.status_code == 400
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert "input" in str(error_body["message"]).lower()

    def test_stream_format_functionality(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """``stream_format`` selects between a raw audio body and SSE audio events.

        With ``sse`` the same mp3 bytes are delivered as base64
        ``speech.audio.delta`` events, closed by a single
        ``speech.audio.done`` event carrying the usage totals.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_audio_speech.py:_speech_audio_sse
        """
        # Test default "audio" stream format
        response_audio = openai_client.audio.speech.create(
            model=speech_standard_model,
            voice="alloy",
            input="Test..",
            stream_format="audio",
        )

        assert isinstance(response_audio.content, bytes)
        assert len(response_audio.content) > 0
        assert response_audio.response.headers.get("content-type") == "audio/mpeg"
        _assert_is_mp3(response_audio.content)

        # Test "sse" stream format for Server-Sent Events
        response_sse = openai_client.audio.speech.create(
            model=speech_standard_model,
            voice="alloy",
            input="Testing audio stream.",
            stream_format="sse",
        )

        assert isinstance(response_sse.content, bytes)
        assert len(response_sse.content) > 0
        content_type = response_sse.response.headers.get("content-type", "")
        assert content_type.startswith("text/event-stream")

        events = _sse_events(response_sse.content)
        deltas = [event for event in events if event["type"] == "speech.audio.delta"]
        assert deltas, "sse streaming must emit speech.audio.delta events"
        _assert_is_mp3(b"".join(b64decode(delta["audio"]) for delta in deltas))

        (done,) = [event for event in events if event["type"] == "speech.audio.done"]
        usage = done["usage"]
        assert usage["input_tokens"] > 0
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]

    def test_speech_usage_logged(
        self,
        test_client: TestClient | None,
        speech_standard_model: str,
        api_key: str,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """Synthesis logs Polly character usage plus the Comprehend detection call.

        An OpenAI voice name has no Polly equivalent, so the gateway detects
        the language with Comprehend first; that call is billed separately and
        is charged a 3-unit minimum whatever the sample length.

        Ref: stdapi/models/audio/amazon_polly.py:_detect_language
             stdapi/usage.py:record_polly_usage
        """
        if test_client is None:
            pytest.skip("Requires local test server")

        text = "Hello world, this is a test."

        capfd.readouterr()

        response = test_client.post(
            "/v1/audio/speech",
            json={"model": speech_standard_model, "voice": "alloy", "input": text},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert response.status_code == 200
        assert response.content is not None
        assert len(response.content) > 0

        captured_out = capfd.readouterr().out

        polly_entries = logged_usage_entries(
            captured_out, service="polly", operation="/v1/audio/speech"
        )
        assert polly_entries, "Expected polly service in usage"
        polly_entry = polly_entries[0]
        assert polly_entry["model"] == "amazon.polly-standard"
        assert "input_characters" in polly_entry
        assert polly_entry["input_characters"] == len(text)

        # Comprehend has a 3-unit minimum for language detection
        comprehend_entries = logged_usage_entries(
            captured_out, service="comprehend", operation="/v1/audio/speech"
        )
        assert comprehend_entries, "Expected comprehend service in usage"
        comprehend_entry = comprehend_entries[0]
        assert comprehend_entry["model"] == "amazon.comprehend-language-detection"
        assert "comprehend_units" in comprehend_entry
        assert comprehend_entry["comprehend_units"] == 3


class TestAudioSpeechMCP:
    """MCP tool behavior of the /v1/audio/speech endpoint.

    Ref: https://stdapi.ai/api_openai_audio_speech/
         stdapi/routes/openai_audio_speech.py:create_speech
    """

    def test_mcp_default_uses_sse(
        self, test_client: TestClient, api_key: str, speech_standard_model: str
    ) -> None:
        """An MCP tool call without ``stream_format`` streams SSE audio events.

        Over HTTP the default is a binary audio body; MCP clients cannot
        consume that, so the route switches to the SSE framing whenever
        ``stream_format`` was not set explicitly by the caller.

        Ref: stdapi/routes/openai_audio_speech.py:create_speech
             stdapi/mcp.py:is_mcp
        """
        if test_client is None:
            pytest.skip("Local only")

        # Initialize MCP session
        init_response = test_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "1.0"},
                },
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json, text/event-stream",
            },
        )
        assert init_response.status_code == 200
        mcp_session_id = init_response.headers["mcp-session-id"]

        # Call the speech tool with default parameters
        response = test_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "openai_audio_speech",
                    "arguments": {
                        "model": speech_standard_model,
                        "voice": "alloy",
                        "input": "Test MCP audio.",
                    },
                },
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json, text/event-stream",
                "mcp-session-id": mcp_session_id,
            },
        )
        assert response.status_code == 200

        # For MCP calls without explicit stream_format, should return SSE
        response_text = response.text
        assert "speech.audio.delta" in response_text
        assert "speech.audio.done" in response_text
        assert "total_tokens" in response_text, (
            "the terminating done event must carry the usage totals"
        )


async def _byte_stream(*chunks: bytes) -> AsyncGenerator[bytes]:
    """Yield the given chunks as a TTS-like audio stream."""
    for chunk in chunks:
        yield chunk


class _StubSpeechModel:
    """Audio model stub returning a fixed TTS response."""

    def __init__(self, response: TTSResponse) -> None:
        self._response = response

    async def tts(self, **_kwargs: object) -> TTSResponse:
        """Return the fixed TTS response, ignoring the request parameters."""
        return self._response


@pytest.mark.local
class TestAudioSpeechContentType:
    """create_speech: the model's content type override wins over response_format.

    Ref: https://stdapi.ai/api_openai_audio_speech/
         stdapi/routes/openai_audio_speech.py:create_speech
    """

    @pytest.fixture(autouse=True)
    def _request_context(self) -> Generator[None]:
        """Provide the request-scoped context vars the route logs into."""
        log_token = REQUEST_LOG.set(
            EventLog(
                type="start",
                level="info",
                date=datetime.now(UTC),
                server_id="test",
                server_version="0.0.0",
            )
        )
        id_token = REQUEST_ID.set("test-request")
        request_token = REQUEST.set(
            Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/audio/speech",
                    "headers": [],
                }
            )
        )
        try:
            yield
        finally:
            REQUEST.reset(request_token)
            REQUEST_ID.reset(id_token)
            REQUEST_LOG.reset(log_token)

    @staticmethod
    def _patch_model(
        monkeypatch: pytest.MonkeyPatch, tts_response: TTSResponse
    ) -> None:
        """Route model resolution to a stub returning ``tts_response``."""

        async def _validate_model(model: str, **_kwargs: object) -> object:
            """Accept any model ID without calling AWS."""
            return type("_Model", (), {"id": model})

        monkeypatch.setattr(openai_audio_speech, "validate_model", _validate_model)
        monkeypatch.setattr(
            openai_audio_speech,
            "get_audio_model",
            lambda _model_id: _StubSpeechModel(tts_response),
        )

    async def test_content_type_override_labels_the_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A JSON speech marks payload is served as-is with its own content type.

        ``response_format="mp3"`` is ignored for a non-audio payload, and no
        download filename is attached since the body is not an audio file.

        Ref: https://docs.aws.amazon.com/polly/latest/dg/speechmarks.html
             stdapi/routes/openai_audio_speech.py:create_speech
        """
        marks = b'{"time":0,"type":"word","value":"Hello"}\n'
        self._patch_model(
            monkeypatch,
            TTSResponse(
                audio_stream=_byte_stream(marks),
                input_tokens=5,
                output_tokens=0,
                content_type="application/x-json-stream",
            ),
        )

        response = await openai_audio_speech.create_speech(
            SpeechCreateParams(
                model="amazon.polly-neural",
                voice="Joanna",
                input="Hello",
                response_format="mp3",
            )
        )
        body = b"".join([chunk async for chunk in response.body_iterator])  # type: ignore[attr-defined]

        assert response.media_type == "application/x-json-stream"
        assert "content-disposition" not in response.headers
        assert body == marks

    async def test_content_type_override_rejects_sse(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SSE audio events cannot carry a non-audio payload, so it is rejected.

        The route fails with a 400 before streaming anything and closes the
        backend stream, so no Polly response is left dangling.

        Ref: https://stdapi.ai/api_openai_audio_speech/
             stdapi/routes/openai_audio_speech.py:create_speech
        """
        stream = _byte_stream(b"{}\n")
        self._patch_model(
            monkeypatch,
            TTSResponse(
                audio_stream=stream,
                input_tokens=5,
                output_tokens=0,
                content_type="application/x-json-stream",
            ),
        )

        with pytest.raises(ApiError, match="sse") as exc_info:
            await openai_audio_speech.create_speech(
                SpeechCreateParams(
                    model="amazon.polly-neural",
                    voice="Joanna",
                    input="Hello",
                    stream_format="sse",
                )
            )

        error = exc_info.value
        assert error.status == 400
        assert error.code is None
        assert "speech marks" in str(error)
        assert getasyncgenstate(stream) == AGEN_CLOSED, (
            "the rejected stream must be closed"
        )

    async def test_without_override_the_response_format_still_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An audio response keeps its response_format derived content type.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_audio_speech.py:create_speech
        """
        self._patch_model(
            monkeypatch,
            TTSResponse(
                audio_stream=_byte_stream(b"audio"), input_tokens=5, output_tokens=0
            ),
        )

        response = await openai_audio_speech.create_speech(
            SpeechCreateParams(
                model="amazon.polly-neural",
                voice="Joanna",
                input="Hello",
                response_format="mp3",
            )
        )
        body = b"".join([chunk async for chunk in response.body_iterator])  # type: ignore[attr-defined]

        assert response.media_type == "audio/mpeg"
        assert response.headers["content-disposition"] == (
            "attachment; filename=speech.mp3"
        )
        assert body == b"audio"
