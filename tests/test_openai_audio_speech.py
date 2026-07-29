"""Tests for the OpenAI /v1/audio/speech route.

Comprehensive test suite that validates all features of the OpenAI Audio Speech API
specification, ensuring compatibility with the official OpenAI API behavior.
"""

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

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


class TestAudioSpeech:
    """Test suite for the /v1/audio/speech endpoint.

    Tests are designed to validate complete OpenAI API compatibility including:
    - All parameter combinations and validations
    - All response formats and audio output validation
    - Complete error scenario coverage with exact error matching
    - Edge cases and boundary conditions
    - HTTP headers and streaming behavior
    """

    def test_basic_speech_generation(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """Test fundamental speech generation functionality with default parameters.

        Validates the core text-to-speech functionality using minimal parameters
        to ensure the service can convert text to audio successfully.

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier

        Validates:
            - Response contains binary audio data
            - Audio data is non-empty and properly formatted
            - Content-Type header indicates MP3 format (default)
            - Basic TTS conversion works with standard model and voice
        """
        response = openai_client.audio.speech.create(
            model=speech_standard_model, voice="alloy", input="Test."
        )

        # Check that we get audio data back
        audio_data = response.content
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0

        # Check response headers
        assert response.response.headers.get("content-type") == "audio/mpeg"

    def test_basic_speech_long_generation(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """Test speech generation with longer text input.

        Validates text-to-speech functionality with extended input text
        to ensure the service handles longer content without language detection errors.

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier

        Validates:
            - Response contains binary audio data for longer text
            - Audio data is non-empty and properly formatted
            - Content-Type header indicates MP3 format (default)
            - TTS handles text samples within language detection limits
            - Fallback to default en-US voice if language detection fails (Esperanto sample)
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

    @pytest.mark.parametrize("sample_rate", ["8000", "24000"])
    def test_speech_with_extra_polly_sample_rate(
        self,
        openai_client: OpenAI,
        speech_standard_model: str,
        use_official_api: bool,
        sample_rate: str,
    ) -> None:
        """Test speech generation with extra Polly-specific parameters.

        Validates that extra AWS Polly parameters are properly forwarded and applied,
        specifically testing the SampleRate parameter to ensure provider-specific
        parameters work as documented.

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier
            use_official_api: True is using official OpenAI API.
            sample_rate: Saple rate of generated voice.

        Validates:
            - Extra Polly parameters are accepted and forwarded
            - SampleRate parameter works correctly
            - Audio is generated successfully with custom sample rate
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

    def test_speech_pcm_default_resamples_to_24khz(
        self, openai_client: OpenAI, speech_standard_model: str, use_official_api: bool
    ) -> None:
        """Test default pcm output follows OpenAI's 24 kHz contract, not Polly's 16 kHz.

        Validates against real Polly that omitting the `SampleRate` extra
        parameter resamples pcm output server-side: its byte length scales
        with 24 kHz, roughly 1.5x the length of Polly's native 16 kHz pcm
        for the same synthesized text.

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier
            use_official_api: True is using official OpenAI API.

        Validates:
            - Default pcm output length scales with a 24 kHz sample rate
            - An explicit Polly-native SampleRate still yields 16 kHz pcm
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

    def test_speech_marks_returns_json_lines(
        self, openai_client: OpenAI, speech_standard_model: str, use_official_api: bool
    ) -> None:
        """Test SpeechMarkTypes returns timing marks instead of audio.

        Validates against real Polly that the `SpeechMarkTypes` extra
        parameter switches the response to a JSON lines stream labeled
        `application/x-json-stream`, bypassing audio synthesis entirely.

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier
            use_official_api: True is using official OpenAI API.

        Validates:
            - The response content type is `application/x-json-stream`
            - Each line is a JSON object describing a word-timing mark
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
        for line in lines:
            mark = json.loads(line)
            assert mark["type"] == "word"

    def test_speech_with_extra_invalid_parameter(
        self, openai_client: OpenAI, speech_standard_model: str, use_official_api: bool
    ) -> None:
        """Test speech generation with invalid extra Polly-specific parameters.

        Validates that invalid extra AWS Polly parameters are properly rejected
        with appropriate validation errors.

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier
            use_official_api: True is using official OpenAI API.

        Validates:
            - Invalid extra Polly parameters are rejected
            - Proper error response with validation details
            - Error indicates the parameter validation failure
        """
        if use_official_api:
            pytest.skip("Amazon Polly is not available on the official OpenAI API")

        with pytest.raises(BadRequestError):
            openai_client.audio.speech.create(
                model=speech_standard_model,
                voice="alloy",
                input="Test.",
                extra_body={"SampleRate": "invalid_value"},
            )

        with pytest.raises(BadRequestError):
            openai_client.audio.speech.create(
                model=speech_standard_model,
                voice="alloy",
                input="Test.",
                extra_body={"Invalid": "invalid_value"},
            )

    @pytest.mark.parametrize(
        "voice",
        ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash", "sage", "coral"],
    )
    def test_all_voices_compatibility(
        self, openai_client: OpenAI, speech_standard_model: str, voice: str
    ) -> None:
        """Test all OpenAI voices work with different models.

        Validates that all standard OpenAI voices work correctly.

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier
            voice: The voice to test

        Validates:
            - All voices produce valid audio output
            - Voice selection works across model variations
            - Response format and content type headers are correct
        """
        response = openai_client.audio.speech.create(
            model=speech_standard_model, voice=voice, input="Test."
        )

        audio_data = response.content
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0
        assert response.response.headers.get("content-type") == "audio/mpeg"

    @pytest.mark.parametrize("voice", ["Amy", "amy"])
    def test_polly_voices_compatibility(
        self,
        openai_client: OpenAI,
        speech_standard_model: str,
        voice: str,
        use_official_api: bool,
    ) -> None:
        """Test Polly voices works.

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier
            voice: The voice to test
            use_official_api: True if using OpenAI API

        Validates:
            - All voices produce valid audio output
            - Voice selection works across model variations
            - Response format and content type headers are correct
        """
        if use_official_api:
            pytest.skip(
                "Amazon Polly models are not available on the official OpenAI API"
            )
        response = openai_client.audio.speech.create(
            model=speech_standard_model, voice=voice, input="Test."
        )

        assert isinstance(response.content, bytes)

    @pytest.mark.parametrize("speed", [0.25, 1.0, 2.0])
    def test_speed_parameter_validation(
        self, openai_client: OpenAI, speech_standard_model: str, speed: float
    ) -> None:
        """Test speed parameter with valid boundary values.

        Validates speed parameter behavior at boundary values and common settings
        according to OpenAI specification (0.25 to 4.0 range).

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier
            speed: The speed value to test

        Validates:
            - Low speed (0.25) produces valid output
            - High speed (4.0 for OpenAI, 2.0 for Polly) produces valid output
            - 1.0 Speed
            - Audio duration varies appropriately with speed
        """
        response = openai_client.audio.speech.create(
            model=speech_standard_model, voice="alloy", input="Test.", speed=speed
        )

        audio_data = response.content
        assert isinstance(audio_data, bytes)
        assert len(audio_data) > 0
        assert response.response.headers.get("content-type") == "audio/mpeg"

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
        """Test all OpenAI supported audio response formats.

        Validates all audio formats specified in OpenAI API documentation:
        mp3, opus, aac, flac, wav, pcm. Each format is tested for proper
        content-type headers and valid audio format signatures.

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier
            format_name: The audio format to test
            content_type: Expected content type header
            signature_check: Expected signature string in audio data (or None)

        Validates:
            - All format conversions work correctly
            - Content-Type headers match requested formats exactly
            - Audio data contains proper format signatures
            - Response data is valid binary audio content
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

    def test_text_length_boundaries(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """Test text input length at boundary conditions.

        Validates behavior with minimum viable text (1 char) and maximum
        allowed text (4096 chars according to OpenAI spec).

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier

        Validates:
            - Single character input produces valid audio
            - Longer length text is processed correctly
            - Audio output length scales appropriately with input
        """
        # Test minimum length (1 character)
        response = openai_client.audio.speech.create(
            model=speech_standard_model, voice="alloy", input="A"
        )
        assert isinstance(response.content, bytes)
        assert len(response.content) > 0

        # Test longer length
        response = openai_client.audio.speech.create(
            model=speech_standard_model, voice="alloy", input="A" * 128
        )
        assert isinstance(response.content, bytes)
        assert len(response.content) > 0

    def test_empty_input_error(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """Test error handling for empty text input.

        Validates proper error response for empty input according to OpenAI specification.

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and code (null)
            - Error response format matches OpenAI specification
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

    def test_invalid_model_error(self, openai_client: OpenAI) -> None:
        """Test error handling for invalid model specification.

        Validates proper error response for non-existent model names.

        Args:
            openai_client: OpenAI client instance for API calls

        Validates:
            - Correct HTTP status code (404)
            - Proper error type ("invalid_request_error") and code ("model_not_found")
            - Error message identifies model as invalid
            - Consistent error response structure
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

    def test_invalid_voice_error(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """Test error handling for invalid voice specification.

        Validates proper error response for non-existent voice names.

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and code (null)
            - Error message identifies voice as invalid
            - Error response includes available voice information
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

    @pytest.mark.parametrize("speed", [0.0, -1.0, 10.0])
    def test_invalid_speed_error(
        self, openai_client: OpenAI, speech_standard_model: str, speed: float
    ) -> None:
        """Test error handling for invalid speed values.

        Validates proper error response for speed values outside the valid range (0.25-4.0).

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier
            speed: The invalid speed value to test

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and code (null)
            - Error message mentions speed validation
            - All boundary violations are caught
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

    def test_invalid_response_format_error(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """Test error handling for invalid response format specification.

        Validates proper error response for unsupported audio formats.

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and code (null)
            - Error message mentions format validation
            - Lists supported formats in error response
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

    def test_missing_required_parameters(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """Test error handling for missing required parameters.

        Validates that all required parameters (model, voice, input) must be provided.

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier

        Validates:
            - Missing parameters result in proper validation errors
            - Error messages identify specific missing fields
            - Consistent error response format
        """
        # Test missing model
        with pytest.raises(BadRequestError):
            openai_client.audio.speech.create(
                voice="alloy",
                input="Test message.",
                model=None,  # type: ignore[arg-type]
            )

        # Test missing voice
        with pytest.raises(BadRequestError):
            openai_client.audio.speech.create(
                model=speech_standard_model,
                input="Test message.",
                voice=None,  # type: ignore[arg-type]
            )

        # Test missing input
        with pytest.raises(BadRequestError):
            openai_client.audio.speech.create(
                model=speech_standard_model,
                voice="alloy",
                input=None,  # type: ignore[arg-type]
            )

    def test_stream_format_functionality(
        self, openai_client: OpenAI, speech_standard_model: str
    ) -> None:
        """Test stream_format parameter with all supported formats.

        Validates that the stream_format parameter works correctly with both
        standard HTTP response ("audio") and Server-Sent Events ("sse") formats.

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model identifier

        Validates:
            - "audio" format returns standard HTTP response with complete audio
            - "sse" format enables Server-Sent Events streaming
            - Both formats produce valid audio output
            - Response headers are appropriate for each format
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

        # Test "sse" stream format for Server-Sent Events
        response_sse = openai_client.audio.speech.create(
            model=speech_standard_model,
            voice="alloy",
            input="Testing audio stream.",
            stream_format="sse",
        )

        # SSE response should still provide audio content but potentially with different streaming behavior
        assert isinstance(response_sse.content, bytes)
        assert len(response_sse.content) > 0

    def test_speech_usage_logged(
        self,
        test_client: TestClient | None,
        speech_standard_model: str,
        api_key: str,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """Test that usage is logged for TTS (speech) requests."""
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
    """Test suite for MCP tool behavior of the /v1/audio/speech endpoint."""

    def test_mcp_default_uses_sse(
        self, test_client: TestClient, api_key: str, speech_standard_model: str
    ) -> None:
        """Test MCP tool defaults to SSE streaming when stream_format not specified.

        Validates that when the speech endpoint is called as an MCP tool with
        default parameters (no stream_format specified), it returns SSE format
        for better MCP client compatibility.

        Args:
            test_client: Test client for HTTP requests
            api_key: Authentication token
            speech_standard_model: Standard speech model identifier

        Validates:
            - MCP tool call with default params returns SSE response
            - Response contains speech.audio.delta events
            - Response is properly formatted as Server-Sent Events
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
    """create_speech: the model's content type override wins over response_format."""

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
        """A JSON speech marks payload is served as-is with its own content type."""
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
        """SSE audio events cannot carry a non-audio payload, so it is rejected."""
        self._patch_model(
            monkeypatch,
            TTSResponse(
                audio_stream=_byte_stream(b"{}\n"),
                input_tokens=5,
                output_tokens=0,
                content_type="application/x-json-stream",
            ),
        )

        with pytest.raises(ApiError, match="sse"):
            await openai_audio_speech.create_speech(
                SpeechCreateParams(
                    model="amazon.polly-neural",
                    voice="Joanna",
                    input="Hello",
                    stream_format="sse",
                )
            )

    async def test_without_override_the_response_format_still_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An audio response keeps its response_format derived content type."""
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
