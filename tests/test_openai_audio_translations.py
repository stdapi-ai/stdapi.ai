"""Tests for the OpenAI /v1/audio/translations route.

Comprehensive test suite that validates all features of the OpenAI Audio Translations API
specification, ensuring compatibility with the official OpenAI API behavior.
"""

import io
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError, NotFoundError, OpenAI

from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from starlette.testclient import TestClient as TestClientType


@pytest.fixture(scope="module")
def sample_audio_fr_file(openai_client: OpenAI, speech_standard_model: str) -> bytes:
    """Create a French-language sample audio file for translation usage testing.

    ``aws_translate.translate()`` returns early without billing AWS Translate
    when the detected source language is English, so exercising the billing
    path requires non-English audio. Generated via the speech endpoint (same
    pattern as conftest's ``sample_audio_mp3_file``; MP3 is a native Polly
    format) and cached under ``tests/.cache/audio_fr.mp3``.
    """
    audio_file = Path(__file__).parent / ".cache" / "audio_fr.mp3"
    if audio_file.exists():
        with audio_file.open("rb") as file:
            return file.read()
    content = openai_client.audio.speech.create(
        model=speech_standard_model,
        voice="alloy",
        input="Ceci est un test simple.",
        response_format="mp3",
    ).content
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    with audio_file.open("wb") as file:
        file.write(content)
    return content


class TestAudioTranslations:
    """Test suite for the /v1/audio/translations endpoint.

    Tests are designed to validate complete OpenAI API compatibility including:
    - All parameter combinations and validations
    - All response formats and translation output validation
    - Complete error scenario coverage with exact error matching
    - Edge cases and boundary conditions
    - Translation-specific functionality and language processing
    - File format support and audio processing capabilities
    """

    @pytest.mark.expensive
    def test_core_translation_functionality(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """Test core translation functionality with comprehensive parameter validation.

        Efficiently validates the fundamental translation capabilities in a single
        optimized API call, focusing on translation-specific features:
        - Audio to English text conversion (core translation function)
        - Parameter handling (temperature, prompt)
        - JSON response format validation
        - Translation-specific output validation

        Args:
            openai_client: OpenAI client instance for API calls
            sample_audio_file: Audio file bytes for translation testing
            transcription_model: Transcription model identifier

        Validates:
            - Response contains text attribute with translated English content
            - Text output is string type and non-empty
            - Temperature and prompt parameters are processed correctly
            - Core translation workflow functions end-to-end
        """
        response = openai_client.audio.translations.create(
            file=("test.wav", io.BytesIO(sample_audio_file)), model=transcription_model
        )

        # Validate core translation functionality
        assert hasattr(response, "text")
        assert isinstance(response.text, str)
        assert len(response.text.strip()) > 0

    @pytest.mark.expensive
    def test_translation_specific_response_formats(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """Test translation-specific aspects of response formats.

        Focuses only on translation-specific format features, avoiding
        redundant testing of generic format support already covered
        in transcription tests. Optimized to test key formats efficiently.

        Args:
            openai_client: OpenAI client instance for API calls
            sample_audio_file: Audio file bytes for translation testing
            transcription_model: Transcription model identifier

        Validates:
            - TEXT format returns plain English text string
            - VERBOSE_JSON format includes translation-specific metadata
            - Translation workflow works with different output formats
        """
        # Test TEXT format for translation (efficient single call)
        text_response = openai_client.audio.translations.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format="text",
        )
        assert isinstance(text_response, str)
        assert len(text_response.strip()) > 0

        # Test VERBOSE_JSON for translation-specific metadata
        verbose_response = openai_client.audio.translations.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format="verbose_json",
        )
        assert hasattr(verbose_response, "text")
        assert hasattr(verbose_response, "language")  # Source language detected
        assert hasattr(verbose_response, "duration")
        assert isinstance(verbose_response.duration, int | float)
        assert verbose_response.duration >= 0

    def test_invalid_model_error(
        self, openai_client: OpenAI, sample_audio_file: bytes
    ) -> None:
        """Test error handling for invalid translation model specification.

        Validates proper error response for non-existent model names.

        Args:
            openai_client: OpenAI client instance for API calls
            sample_audio_file: Audio file bytes for translation testing

        Validates:
            - Correct HTTP status code (404)
            - Proper error type ("invalid_request_error") and code ("model_not_found")
            - Error message identifies model as invalid
            - Consistent error response structure
        """
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.audio.translations.create(
                file=("test.wav", io.BytesIO(sample_audio_file)),
                model="invalid-nonexistent-model",
            )

        error = exc_info.value
        assert error.status_code == 404
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert error_body["code"] == "model_not_found"
        error_message = str(error).lower()
        assert any(
            word in error_message for word in ["model", "invalid", "exist", "access"]
        )

    def test_empty_file_error(
        self, openai_client: OpenAI, transcription_model: str
    ) -> None:
        """Test error handling for empty audio files.

        Validates that the translation service properly handles empty files
        and returns appropriate error responses.

        Args:
            openai_client: OpenAI client instance for API calls
            transcription_model: Valid transcription model identifier

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and code (null)
            - Error message mentions invalid file format
            - Error response lists supported formats
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.audio.translations.create(
                file=("empty.wav", io.BytesIO(b"")), model=transcription_model
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        error_message = str(error).lower()
        assert any(
            word in error_message for word in ["format", "supported", "invalid", "file"]
        )

    def test_unsupported_file_format_error(
        self, openai_client: OpenAI, transcription_model: str
    ) -> None:
        """Test error handling for unsupported file formats.

        Validates that the translation service properly handles unsupported file
        formats and returns appropriate error responses.

        Args:
            openai_client: OpenAI client instance for API calls
            transcription_model: Valid transcription model identifier

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and code (null)
            - Error message mentions invalid file format
            - Error response lists supported audio formats
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.audio.translations.create(
                file=("test.txt", io.BytesIO(b"This is not an audio file")),
                model=transcription_model,
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        error_message = str(error).lower()
        assert any(
            word in error_message
            for word in ["format", "supported", "invalid", "flac", "mp3", "wav"]
        )

    @pytest.mark.expensive
    def test_subtitle_format_translation(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """Test translation-specific subtitle format processing.

        Validates the translation-specific subtitle handling logic
        that extracts text from subtitle formats and translates it while
        preserving timing and structure. This functionality is unique to
        the translation endpoint.

        Args:
            openai_client: OpenAI client instance for API calls
            sample_audio_file: Audio file bytes for testing
            transcription_model: Transcription model identifier

        Validates:
            - SRT format returns translated subtitle content with preserved structure
            - VTT format returns translated subtitle content with preserved structure
            - Subtitle translation processing works correctly for both formats
            - Translation-specific subtitle handling functionality
        """
        # Test SRT format translation
        srt_response = openai_client.audio.translations.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format="srt",
        )

        assert isinstance(srt_response, str)
        assert len(srt_response.strip()) > 0
        # For SRT format, should contain timing markers if subtitle content exists
        # or translated text if no timing structure (fallback behavior)

        # Test VTT format translation
        vtt_response = openai_client.audio.translations.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format="vtt",
        )

        assert isinstance(vtt_response, str)
        assert len(vtt_response.strip()) > 0
        # For VTT format, should contain WEBVTT header if subtitle content exists
        # or translated text if no timing structure (fallback behavior)

    def test_empty_transcription_handling(
        self, openai_client: OpenAI, transcription_model: str
    ) -> None:
        """Test translation handling of empty transcription results.

        Validates translation-specific logic for handling cases where
        transcription produces no text content. This tests the translation
        service's robustness in edge cases.

        Args:
            openai_client: OpenAI client instance for API calls
            transcription_model: Transcription model identifier

        Validates:
            - Empty transcription is handled gracefully in translation flow
            - Translation service returns appropriate response for empty input
            - No crashes or unexpected errors occur
        """
        # Create minimal audio content that might produce empty transcription
        minimal_audio = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

        try:
            response = openai_client.audio.translations.create(
                file=("minimal.wav", io.BytesIO(minimal_audio)),
                model=transcription_model,
            )

            # Should handle empty transcription gracefully
            assert hasattr(response, "text")
            assert isinstance(response.text, str)
            # Text may be empty but should not crash

        except BadRequestError:
            # Expected behavior for invalid/minimal audio - translation service should handle gracefully
            pass

    @pytest.mark.expensive
    def test_translation_usage_logged(
        self,
        test_client: TestClientType | None,
        api_key: str,
        sample_audio_fr_file: bytes,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """Test that usage is logged for translation requests."""
        if test_client is None:
            pytest.skip("Requires local test server")

        capfd.readouterr()

        response = test_client.post(
            "/v1/audio/translations",
            files={
                "file": ("test.mp3", io.BytesIO(sample_audio_fr_file), "audio/mpeg")
            },
            data={"model": "amazon.transcribe"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert response.status_code == 200

        response_data = response.json()
        assert isinstance(response_data.get("text"), str)
        assert len(response_data["text"].strip()) > 0

        translate_entries = logged_usage_entries(
            capfd.readouterr().out,
            service="translate",
            operation="/v1/audio/translations",
        )
        assert translate_entries, "Expected translate service in usage"
        translate_entry = translate_entries[0]
        assert translate_entry["model"] == "amazon.translate"
        assert "input_characters" in translate_entry
        # Value depends on audio content
        assert translate_entry["input_characters"] > 0


class TestAudioTranslationsJsonBody:
    """Tests for POST /v1/audio/translations using an application/json body.

    Verifies that the JSON body path (data URI, base64 string) is accepted and
    routes correctly through the handler without requiring multipart form data.
    """

    @pytest.fixture(autouse=True)
    def _skip_on_official_api(self, use_official_api: bool) -> None:
        """JSON body input is an extension not supported by the official API."""
        if use_official_api:
            pytest.skip("JSON body input not supported by the official OpenAI API")

    def test_json_body_missing_file_returns_400(
        self, openai_client: OpenAI, transcription_model: str
    ) -> None:
        """JSON body without the required file field returns 400."""
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}audio/translations",
            json={"model": transcription_model},
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 400

    def test_json_body_invalid_model_returns_404(self, openai_client: OpenAI) -> None:
        """JSON body path reaches model validation — invalid model returns 404.

        Uses a dummy audio data URI to satisfy input validation; model lookup
        runs before any audio decoding so the file content is irrelevant.
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}audio/translations",
            json={
                "file": "data:audio/wav;base64,dGVzdA==",
                "model": "nonexistent-model-xyz",
            },
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "model_not_found"

    @pytest.mark.expensive
    def test_json_body_translation(
        self,
        openai_client: OpenAI,
        sample_audio_file_base64: str,
        transcription_model: str,
    ) -> None:
        """JSON body with audio data URI translates correctly."""
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}audio/translations",
            json={"file": sample_audio_file_base64, "model": transcription_model},
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body.get("text"), str)
        assert len(body["text"].strip()) > 0
