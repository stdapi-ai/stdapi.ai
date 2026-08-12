"""Tests for the OpenAI /v1/audio/speech route served by Amazon Polly.

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://stdapi.ai/api_openai_audio_speech/
     stdapi/routes/openai_audio_speech.py:create_speech
"""

import json
from base64 import b64decode
from inspect import AGEN_CLOSED, getasyncgenstate
from typing import TYPE_CHECKING, Any

import magic
import pytest
from openai import BadRequestError, NotFoundError, OpenAI
from pydantic import ValidationError
from starlette.requests import Request

from stdapi.api_errors import ApiError
from stdapi.models.audio import TTSResponse
from stdapi.monitoring import REQUEST, REQUEST_ID, REQUEST_LOG
from stdapi.routes import openai_audio_speech
from stdapi.types.openai_audio import SpeechCreateParams
from tests._helpers import make_event_log
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

    @pytest.mark.image
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

    @pytest.mark.slow
    def test_speech_above_the_synchronous_character_limit(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """A 4000-character input is still returned as one complete mp3 file.

        4000 characters is under the 4096 the OpenAI API accepts, and over the
        3000 billed characters Polly synthesizes in a single call, so the
        gateway must serve it without the caller splitting the text.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             https://docs.aws.amazon.com/polly/latest/dg/limits.html
             stdapi/models/audio/amazon_polly.py:AudioModel.tts
        """
        with (SAMPLES_DIR / "lorem_ipsum.txt").open() as file:
            input_text = file.read(4000)  # over the single-call limit, under OpenAI's

        response = openai_client.audio.speech.create(
            model=speech_standard_model, voice="alloy", input=input_text
        )
        truncated = openai_client.audio.speech.create(
            model=speech_standard_model, voice="alloy", input=input_text[:3000]
        )

        audio_data = response.content
        assert isinstance(audio_data, bytes)
        assert response.response.headers.get("content-type") == "audio/mpeg"
        _assert_is_mp3(audio_data)
        # Text silently cut at the single-call limit would still be valid mp3:
        # only the same text cut at 3000 characters says how long it should be.
        assert len(audio_data) > len(truncated.content) * 1.2, (
            "the audio is no longer than the first 3000 characters spoken"
        )

    @pytest.mark.image
    @pytest.mark.gateway("Amazon Polly is not available on the official OpenAI API")
    @pytest.mark.retry(
        "the first ffmpeg encode in a long-lived process can miss its stdout EOF: "
        "ffmpeg exits 0 with its input fully consumed, yet the parent's read blocks "
        "until the encode timeout. Only ever the first one -- the 24000 case that "
        "follows seconds later always passes -- and never when this file runs alone"
    )
    @pytest.mark.parametrize("sample_rate", ["8000", "24000"])
    def test_speech_with_extra_polly_sample_rate(
        self, openai_client: OpenAI, speech_standard_model: str, sample_rate: str
    ) -> None:
        """The Polly ``SampleRate`` extra parameter sets the encoded output rate.

        Polly has no FLAC output, so the gateway re-encodes with ffmpeg: at
        8 kHz the source stays pcm, while 24 kHz exceeds Polly's 16 kHz pcm
        cap and is synthesized as Ogg Vorbis instead. Either way the FLAC
        STREAMINFO must advertise the requested rate.

        Ref: https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html
             stdapi/models/audio/amazon_polly.py:AudioModel.tts
        """
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

    @pytest.mark.image
    @pytest.mark.gateway("Amazon Polly is not available on the official OpenAI API")
    def test_speech_pcm_default_resamples_to_24khz(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """Default pcm output follows OpenAI's 24 kHz contract, not Polly's 16 kHz.

        Polly pcm accepts only 8 kHz and 16 kHz, so the gateway resamples its
        16 kHz output to 24 kHz with ffmpeg: the default body carries ~1.5x
        the samples of the same text pinned to Polly's native rate.

        Ref: https://stdapi.ai/api_openai_audio_speech/
             stdapi/models/audio/amazon_polly.py:AudioModel.tts
        """
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
        assert default_len / native_len == pytest.approx(1.5, rel=0.15)
        assert default_response.response.headers.get("content-type") == "audio/pcm"

    @pytest.mark.gateway("Amazon Polly is not available on the official OpenAI API")
    def test_speech_marks_returns_json_lines(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """``SpeechMarkTypes`` returns ordered timing marks instead of audio.

        Polly generates no audio for a speech-marks request: the gateway
        forwards ``OutputFormat=json`` and streams the JSON lines through
        unchanged, labelled ``application/x-json-stream``.

        Ref: https://docs.aws.amazon.com/polly/latest/dg/speechmarks.html
             stdapi/models/audio/amazon_polly.py:AudioModel.tts
        """
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

    @pytest.mark.gateway("Amazon Polly is not available on the official OpenAI API")
    def test_speech_with_extra_invalid_parameter(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """Polly extra parameters are validated: bad type and unknown name are 400s.

        Extra body fields are parsed by ``_PollyExtraParams``, which forbids
        unknown keys, so both failures surface as the gateway's
        ``invalid_request_error`` naming the offending field.

        Ref: https://stdapi.ai/api_openai_audio_speech/
             stdapi/models/audio/amazon_polly.py:_PollyExtraParams
             stdapi/main.py:handle_validation_exception
        """
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

    # "alloy" is covered by test_basic_speech_generation; each extra voice is a
    # billed Polly synthesis plus a Comprehend language detection.
    @pytest.mark.parametrize(
        "voice", ["echo", "fable", "onyx", "nova", "shimmer", "ash", "sage", "coral"]
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

    @pytest.mark.gateway(
        "Amazon Polly models are not available on the official OpenAI API"
    )
    @pytest.mark.parametrize("voice", ["Amy", "amy"])
    def test_polly_voices_compatibility(
        self, openai_client: OpenAI, speech_standard_model: str, voice: str
    ) -> None:
        """A native Polly voice ID is accepted case-insensitively.

        Voices are indexed by lowercase name, so ``amy`` resolves to the
        ``Amy`` voice ID; an unresolved name would reach Polly verbatim and
        come back as a 400 invalid-voice error.

        Ref: https://docs.aws.amazon.com/polly/latest/dg/available-voices.html
             stdapi/models/audio/amazon_polly.py:_select_voice
        """
        response = openai_client.audio.speech.create(
            model=speech_standard_model, voice=voice, input="Test."
        )

        assert isinstance(response.content, bytes)
        assert len(response.content) > 0
        assert response.response.headers.get("content-type") == "audio/mpeg"
        _assert_is_mp3(response.content)

    # 1.0 is the no-op default, already covered by test_basic_speech_generation.
    @pytest.mark.parametrize("speed", [0.25, 2.0])
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

    @pytest.mark.image
    @pytest.mark.parametrize(
        ("format_name", "content_type", "signature_check"),
        [
            # mp3 is the default format, covered by test_basic_speech_generation.
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
        response = openai_client.audio.speech.create(
            model=speech_standard_model, voice="alloy", input="A"
        )
        assert isinstance(response.content, bytes)
        assert len(response.content) > 0
        shortest = response.content
        _assert_is_mp3(shortest)

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
        self, openai_client: OpenAI, speech_standard_model: str, use_official_api: bool
    ) -> None:
        """An unknown voice is rejected as an ``invalid_request_error``.

        OpenAI validates ``voice`` against its built-in enum, so the message
        enumerates the accepted names and never repeats the rejected one. The
        gateway takes a free-form string (Polly voice IDs are accepted too), so
        the rejection comes from Polly's ValidationException, which the gateway
        rewrites into a 400 quoting the offending name.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             https://docs.aws.amazon.com/polly/latest/dg/available-voices.html
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
        if use_official_api:
            # Enum validation: the accepted voices are listed, the input is not.
            assert "alloy" in error_message, (
                "the error must enumerate the accepted built-in voices"
            )
            assert "coral" in error_message
        else:
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
        self, openai_client: OpenAI, speech_standard_model: str, use_official_api: bool
    ) -> None:
        """An unsupported ``response_format`` is rejected, naming the offending field.

        OpenAI points at the field through ``param`` and tags the failure with
        the ``unsupported_value`` code, its message staying terse. The gateway
        surfaces the pydantic validation error instead: it enumerates the
        accepted formats in the message but leaves ``param`` and ``code`` null.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             https://developers.openai.com/api/docs/guides/error-codes
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
        error_message = str(error).lower()
        assert "response_format" in error_message, (
            "the error must name the offending field"
        )
        if use_official_api:
            assert error_body["code"] == "unsupported_value"
            assert error_body["param"] == "response_format"
        else:
            assert error_body["code"] is None
            assert "mp3" in error_message, (
                "the error must enumerate the accepted formats"
            )
            assert "pcm" in error_message

    def test_missing_required_parameters(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """A null ``model``, ``voice`` or ``input`` fails validation naming that field.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/main.py:handle_validation_exception
        """
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

    @pytest.mark.gateway(
        "CreateSpeechRequest: stream_format='sse' is not supported for tts-1"
    )
    def test_stream_format_functionality(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """``stream_format`` selects between a raw audio body and SSE audio events.

        With ``sse`` the same mp3 bytes are delivered as base64
        ``speech.audio.delta`` events, closed by a single
        ``speech.audio.done`` event carrying the usage totals. OpenAI restricts
        that framing to its ``gpt-4o-mini-tts`` family, and silently returns an
        ``audio/mpeg`` body for ``tts-1``.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_audio_speech.py:_speech_audio_sse
        """
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

    @pytest.mark.local
    def test_speech_usage_logged(
        self,
        test_client: TestClient,
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

    @pytest.mark.local
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


@pytest.fixture
def _speech_request_context() -> Generator[None]:
    """Provide the request-scoped context vars the route logs into."""
    log_token = REQUEST_LOG.set(make_event_log(type="start"))
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


def _patch_speech_model(
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


@pytest.mark.local
@pytest.mark.usefixtures("_speech_request_context")
class TestAudioSpeechContentType:
    """create_speech: the model's content type override wins over response_format.

    Ref: https://stdapi.ai/api_openai_audio_speech/
         stdapi/routes/openai_audio_speech.py:create_speech
    """

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
        _patch_speech_model(
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
        _patch_speech_model(
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
        _patch_speech_model(
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


#: Message of the mid-stream failure raised by the ffmpeg encoder
_ENCODE_ERROR = "Failed to encode the audio to 'mp3'."


@pytest.mark.local
@pytest.mark.usefixtures("_speech_request_context")
class TestAudioSpeechSseTermination:
    """``speech.audio.done`` terminates a completed SSE stream, and nothing else.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/routes/openai_audio_speech.py:_speech_audio_sse
    """

    async def test_broken_stream_ends_on_an_error_not_on_done(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stream failing mid-audio emits an error event and no done event.

        The encoder raises once ffmpeg fails or stalls; a client that stops
        reading on ``speech.audio.done`` would otherwise take the truncated
        audio for a complete one and record its usage.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             stdapi/media.py:encode_audio_stream
        """

        async def _failing_stream() -> AsyncGenerator[bytes]:
            """Yield one chunk, then fail like a broken ffmpeg encode."""
            yield b"audio"
            raise ApiError(_ENCODE_ERROR, status=500)

        _patch_speech_model(
            monkeypatch,
            TTSResponse(
                audio_stream=_failing_stream(), input_tokens=5, output_tokens=0
            ),
        )

        response = await openai_audio_speech.create_speech(
            SpeechCreateParams(
                model="amazon.polly-neural",
                voice="Joanna",
                input="Hello",
                stream_format="sse",
            )
        )
        events = [event async for event in response.body_iterator]  # type: ignore[attr-defined]

        assert json.loads(events[0].data)["type"] == "speech.audio.delta"
        assert not [
            event for event in events if "speech.audio.done" in str(event.data)
        ], "a truncated stream must not be reported as done"
        assert events[-1].event == "error"
        assert "Failed to encode the audio" in str(events[-1].data)

    async def test_disconnect_closes_the_stream_without_a_done_event(self) -> None:
        """Closing the event generator mid-audio must not raise.

        ``aclose`` throws ``GeneratorExit`` into the generator, and yielding
        while it is in flight makes Python raise ``RuntimeError: async
        generator ignored GeneratorExit`` on every client disconnect.

        Ref: https://docs.python.org/3/reference/expressions.html#asynchronous-generator-functions
             stdapi/routes/openai_audio_speech.py:create_speech
        """
        audio_stream = _byte_stream(b"audio", b"more")
        events = openai_audio_speech._speech_audio_sse(audio_stream, 5, 0)  # noqa: SLF001

        first = await anext(events)
        await events.aclose()

        assert json.loads(str(first.data))["type"] == "speech.audio.delta"
        assert getasyncgenstate(events) == AGEN_CLOSED
        assert getasyncgenstate(audio_stream) == AGEN_CLOSED, (
            "the backend stream must be closed with the client connection"
        )


@pytest.mark.local
class TestSpeechCreateParamsSsml:
    """``input`` starting with ``<speak>`` is treated as an SSML document.

    Polly is told ``TextType=ssml`` for such an input, and the gateway applies
    ``speed`` by wrapping the text in its own ``<speak><prosody>`` envelope --
    which would nest a second ``<speak>`` root -- so the two are incompatible.

    Ref: https://docs.aws.amazon.com/polly/latest/dg/ssml.html
         stdapi/types/openai_audio.py:SpeechCreateParams._unsupported
    """

    def test_ssml_input_without_speed_is_accepted(self) -> None:
        """An SSML document alone is accepted and left untouched."""
        params = SpeechCreateParams(
            model="amazon.polly-neural", voice="Joanna", input="<speak>Hello</speak>"
        )

        assert params.input == "<speak>Hello</speak>"
        assert params.speed == 1.0

    @pytest.mark.parametrize("speed", [1.0, 1.5])
    def test_ssml_input_with_an_explicit_speed_is_rejected(self, speed: float) -> None:
        """Any explicitly set ``speed`` is rejected, including the default 1.0.

        The check keys off ``model_fields_set``, not off the value, so sending
        ``speed=1.0`` next to SSML fails even though it would have been a no-op.
        """
        with pytest.raises(ValidationError) as exc_info:
            SpeechCreateParams(
                model="amazon.polly-neural",
                voice="Joanna",
                input="<speak>Hello</speak>",
                speed=speed,
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "value_error"
        assert "speed is not supported for SSML input" in errors[0]["msg"]

    def test_plain_text_input_still_accepts_speed(self) -> None:
        """Plain text keeps the documented ``speed`` support."""
        params = SpeechCreateParams(
            model="amazon.polly-neural", voice="Joanna", input="Hello", speed=1.5
        )

        assert params.speed == 1.5


@pytest.mark.local
class TestSpeechCreateParamsCompatibility:
    """Fields kept for OpenAI compatibility: accepted, and never forwarded to Polly.

    Only the declared fields are read by the route; anything else lands in
    ``model_extra`` and is forwarded to SynthesizeSpeech, where an unknown
    parameter is rejected. A field that is dropped from the request model would
    therefore turn a working request into a 400.

    Ref: https://stdapi.ai/api_openai_audio_speech/
         stdapi/types/openai_audio.py:SpeechCreateParams
         stdapi/routes/openai_audio_speech.py:create_speech
    """

    def test_instructions_is_accepted_and_not_forwarded(self) -> None:
        """``instructions`` is parsed as a known field, so Polly never sees it.

        Polly has no equivalent of OpenAI's voice instructions, so the value is
        accepted and ignored rather than rejected.
        """
        params = SpeechCreateParams(
            model="amazon.polly-neural",
            voice="Joanna",
            input="Hello",
            instructions="Speak in a cheerful tone",
        )

        assert params.instructions == "Speak in a cheerful tone"
        assert params.model_extra == {}

    def test_unknown_fields_are_kept_as_polly_extras(self) -> None:
        """An undeclared field is collected as a Polly SynthesizeSpeech extra."""
        params = SpeechCreateParams.model_validate(
            {
                "model": "amazon.polly-neural",
                "voice": "Joanna",
                "input": "Hello",
                "LexiconNames": ["MyLexicon"],
            }
        )

        assert params.model_extra == {"LexiconNames": ["MyLexicon"]}


@pytest.mark.local
class TestSpeechCreateParamsPollyAliases:
    """A raw Polly SynthesizeSpeech body is accepted through the field aliases.

    ``Text``/``Engine``/``VoiceId``/``OutputFormat`` are validation aliases for
    the OpenAI field names. A broken alias would not fail loudly: the key would
    fall through to ``model_extra`` and be forwarded to Polly as a duplicate
    parameter.

    Ref: https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html
         stdapi/types/openai_audio.py:SpeechCreateParams
    """

    def test_polly_field_names_populate_the_openai_fields(self) -> None:
        """Every Polly alias maps onto its OpenAI counterpart, leaving no extras."""
        params = SpeechCreateParams.model_validate(
            {
                "Text": "Hello",
                "Engine": "amazon.polly-neural",
                "VoiceId": "Joanna",
                "OutputFormat": "wav",
            }
        )

        assert params.input == "Hello"
        assert params.model == "amazon.polly-neural"
        assert params.voice == "Joanna"
        assert params.response_format == "wav"
        assert params.model_extra == {}

    def test_openai_field_names_are_still_accepted(self) -> None:
        """The OpenAI names keep working alongside the aliases."""
        params = SpeechCreateParams.model_validate(
            {
                "input": "Hello",
                "model": "amazon.polly-neural",
                "voice": "Joanna",
                "response_format": "wav",
            }
        )

        assert params.input == "Hello"
        assert params.model == "amazon.polly-neural"
        assert params.voice == "Joanna"
        assert params.response_format == "wav"
        assert params.model_extra == {}
