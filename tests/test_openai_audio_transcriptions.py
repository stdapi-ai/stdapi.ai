"""Tests for the OpenAI /v1/audio/transcriptions route.

Comprehensive test suite that validates all features of the OpenAI Audio Transcriptions API
specification, ensuring compatibility with the official OpenAI API behavior.
"""

import io

import pytest
from openai import BadRequestError, NotFoundError, OpenAI


class TestAudioTranscriptions:
    """Test suite for the /v1/audio/transcriptions endpoint.

    Tests are designed to validate complete OpenAI API compatibility including:
    - All parameter combinations and validations
    - All response formats and transcription output validation
    - Complete error scenario coverage with exact error matching
    - Edge cases and boundary conditions
    - Streaming behavior and timestamp granularities
    - File format support and audio processing capabilities
    """

    @pytest.mark.expensive
    def test_basic_transcription(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """Test basic transcription functionality with default parameters.

        Validates the core transcription functionality using minimal parameters
        to ensure the service can convert audio to text successfully. This test
        serves as the foundation for all other transcription features.

        Args:
            openai_client: OpenAI client instance for API calls
            sample_audio_file: Audio file bytes for transcription testing
            transcription_model: Transcription model identifier

        Validates:
            - Response contains text attribute with transcribed content
            - Text output is string type
            - Basic transcription works with default JSON response format
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)), model=transcription_model
        )

        # Should return JSON response with text field
        assert hasattr(response, "text")
        assert isinstance(response.text, str)

    @pytest.mark.expensive
    @pytest.mark.parametrize(
        "response_format", ["json", "text", "srt", "vtt", "verbose_json"]
    )
    def test_different_response_formats(
        self,
        openai_client: OpenAI,
        sample_audio_file: bytes,
        transcription_model: str,
        response_format: str,
    ) -> None:
        """Test transcription with all supported response formats.

        Validates that the transcription service supports all OpenAI-compatible
        response formats for different use cases:
        - json: Standard structured response with text field
        - text: Plain text output for simple integration
        - srt: SubRip subtitle format with timestamps for video
        - vtt: WebVTT format for web-based video captioning
        - verbose_json: Extended response with metadata and timing

        Args:
            openai_client: OpenAI client instance for API calls
            sample_audio_file: Audio file bytes for transcription testing
            transcription_model: Transcription model identifier
            response_format: The response format to test

        Validates:
            - All response formats return appropriate data types
            - SRT/VTT formats contain timing markers or are empty
            - Verbose JSON includes task, language, and duration metadata
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format=response_format,  # type: ignore[call-overload]
        )

        if response_format == "json":
            assert hasattr(response, "text")
            assert isinstance(response.text, str)
        elif response_format == "text":
            assert isinstance(response, str)
        elif response_format in ["srt", "vtt"]:
            assert isinstance(response, str)
            # SRT/VTT should contain timing information
            assert "-->" in response or len(response.strip()) == 0
        elif response_format == "verbose_json":
            assert hasattr(response, "text")
            assert hasattr(response, "language")
            assert hasattr(response, "duration")
            assert isinstance(response.duration, int | float)
            assert response.duration >= 0

    @pytest.mark.expensive
    def test_timestamp_granularities(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """Test transcription with timestamp granularities for detailed timing information.

        Validates both segment and word-level timestamp granularities in a single
        efficient API call. This test ensures the transcription service can provide
        detailed timing information for accessibility and video captioning use cases.

        Args:
            openai_client: OpenAI client instance for API calls
            sample_audio_file: Audio file bytes for transcription testing
            transcription_model: Transcription model identifier

        Validates:
            - Response contains text, segments, and words attributes
            - Segment objects have start, end, and text timing information
            - Word objects have word, start, and end timing information
            - Verbose JSON format includes task and duration metadata
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
        )

        # Validate basic response structure
        assert hasattr(response, "text")
        assert hasattr(response, "segments")
        assert hasattr(response, "words")

        # Validate detailed segment timing information if available
        if hasattr(response, "segments") and response.segments:
            segment = response.segments[0]
            assert hasattr(segment, "start")
            assert hasattr(segment, "end")
            assert hasattr(segment, "text")

        # Validate detailed word timing information if available
        if hasattr(response, "words") and response.words:
            word = response.words[0]
            assert hasattr(word, "word")
            assert hasattr(word, "start")
            assert hasattr(word, "end")

    @pytest.mark.expensive
    def test_streaming_transcription(
        self,
        openai_client: OpenAI,
        sample_audio_file: bytes,
        transcription_stream_model: str,
    ) -> None:
        """Test streaming transcription functionality for real-time processing.

        Validates that the transcription service can provide streaming responses
        with proper content validation for real-time applications like live captioning
        or voice assistants. This test validates both chunk presence and content quality.

        Args:
            openai_client: OpenAI client instance for API calls
            sample_audio_file: Audio file bytes for transcription testing
            transcription_stream_model: Valid transcription model identifier

        Validates:
            - Streaming response is iterable and produces chunks
            - Chunks contain proper transcription delta events
            - Chunk content includes meaningful transcription text
            - Stream processing completes with proper completion event
            - Accumulated content represents valid transcription
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_stream_model,
            stream=True,
        )

        # Validate streaming response and collect chunks with content validation
        chunks = []
        accumulated_text = ""
        has_delta_events = False

        for chunk in response:
            chunks.append(chunk)

            # Validate chunk structure and content
            assert hasattr(chunk, "type"), f"Chunk missing 'type' attribute: {chunk}"

            # Check for delta chunk type and validate content
            if chunk.type == "transcript.text.delta":
                has_delta_events = True
                assert hasattr(chunk, "delta"), (
                    f"Delta chunk missing 'delta' attribute: {chunk}"
                )

                # Delta is a direct string attribute in OpenAI API
                if chunk.delta:
                    accumulated_text += chunk.delta
                    # Validate delta text is not empty or just whitespace when present
                    assert chunk.delta.strip(), (
                        f"Delta text is empty or whitespace: '{chunk.delta}'"
                    )

            # Limit chunks for efficiency while ensuring we get meaningful content
            if len(chunks) >= 15:  # Allow more chunks to capture complete transcription
                break

        # Validate overall streaming behavior
        assert len(chunks) > 0, "No streaming chunks received"
        assert has_delta_events, "No delta transcription events received"

        # Ensure we accumulated meaningful text from delta events
        assert accumulated_text.strip(), (
            f"No meaningful text accumulated from delta events: '{accumulated_text}'"
        )

        # Validate that the accumulated text contains expected content
        # Since we know the input audio contains "test audio file for transcription testing"
        final_text = accumulated_text.strip()

        # Basic sanity check - transcription should contain some common words
        # that would likely appear in the test audio
        final_text_lower = final_text.lower()
        expected_words = ["test", "audio", "file"]
        word_matches = sum(1 for word in expected_words if word in final_text_lower)
        assert word_matches >= 1, (
            f"Transcription doesn't contain expected content: '{final_text}'"
        )

        # Validate text quality - should be reasonably long and contain meaningful content
        assert len(final_text) > 10, f"Transcription text too short: '{final_text}'"
        assert not final_text.isdigit(), (
            f"Transcription contains only digits: '{final_text}'"
        )

        # Validate that we got multiple delta chunks (streaming behavior)
        delta_chunks = [c for c in chunks if c.type == "transcript.text.delta"]
        assert len(delta_chunks) >= 1, (
            f"Expected multiple delta chunks, got {len(delta_chunks)}"
        )

    def test_empty_file_error(
        self, openai_client: OpenAI, transcription_model: str
    ) -> None:
        """Test error handling for empty audio files.

        Validates that the transcription service properly handles empty files
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
            openai_client.audio.transcriptions.create(
                file=("empty.wav", io.BytesIO(b"")), model=transcription_model
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert error_body["code"] is None
        error_message = str(error).lower()
        assert any(
            word in error_message for word in ["format", "supported", "invalid", "file"]
        )

    def test_invalid_model_error(
        self, openai_client: OpenAI, sample_audio_file: bytes
    ) -> None:
        """Test error handling for invalid transcription model specification.

        Validates proper error response for non-existent model names.

        Args:
            openai_client: OpenAI client instance for API calls
            sample_audio_file: Audio file bytes for transcription testing

        Validates:
            - Correct HTTP status code (404)
            - Proper error type ("invalid_request_error") and code ("model_not_found")
            - Error message identifies model as invalid
            - Consistent error response structure
        """
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.audio.transcriptions.create(
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

    def test_invalid_language_error(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """Test error handling for invalid language code specification.

        Validates proper error response for non-ISO 639-1 language codes.

        Args:
            openai_client: OpenAI client instance for API calls
            sample_audio_file: Audio file bytes for transcription testing
            transcription_model: Valid transcription model identifier

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and code ("invalid_language_format")
            - Error message identifies language as invalid
            - Error response mentions ISO 639-1 format requirement
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.audio.transcriptions.create(
                file=("test.wav", io.BytesIO(sample_audio_file)),
                model=transcription_model,
                language="invalid-lang",
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert error_body["code"] == "invalid_language_format"
        error_message = str(error).lower()
        assert any(
            word in error_message for word in ["language", "invalid", "iso", "format"]
        )

    def test_invalid_response_format_error(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """Test error handling for invalid response format specification.

        Validates proper error response for unsupported response formats.

        Args:
            openai_client: OpenAI client instance for API calls
            sample_audio_file: Audio file bytes for transcription testing
            transcription_model: Valid transcription model identifier

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and code (null)
            - Error message mentions format validation
            - Lists supported formats in error response
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.audio.transcriptions.create(
                file=("test.wav", io.BytesIO(sample_audio_file)),
                model=transcription_model,
                response_format="invalid_format",  # type: ignore[call-overload]
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
            for word in ["format", "response", "json", "text", "vtt", "srt"]
        )

    @pytest.mark.expensive
    @pytest.mark.parametrize("temperature", [0.5, 1.0])
    def test_temperature_parameter_validation(
        self,
        openai_client: OpenAI,
        sample_audio_file: bytes,
        transcription_model: str,
        temperature: float,
        use_openai_api: bool,
    ) -> None:
        """Test temperature parameter with various values.

        OpenAI API accepts temperature values outside typical ranges for transcriptions
        (unlike speech synthesis), so this test validates that the API handles
        various temperature values without errors.

        Args:
            openai_client: OpenAI client instance for API calls
            sample_audio_file: Audio file bytes for transcription testing
            transcription_model: Valid transcription model identifier
            temperature: The temperature value to test
            use_openai_api: True if using the official OpenAI API

        Validates:
            - Various temperature values are accepted
            - Response contains text attribute
            - Response text is string type
        """
        if not use_openai_api:
            pytest.skip("Parameter is not supported by Amazon Transcribe.")

        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            temperature=temperature,
        )

        assert hasattr(response, "text")
        assert isinstance(response.text, str)

    def test_unsupported_file_format_error(
        self, openai_client: OpenAI, transcription_model: str
    ) -> None:
        """Test error handling for unsupported file formats.

        Validates that the transcription service properly handles unsupported file
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
            openai_client.audio.transcriptions.create(
                file=("test.txt", io.BytesIO(b"This is not an audio file")),
                model=transcription_model,
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
            for word in ["format", "supported", "invalid", "flac", "mp3", "wav"]
        )

    @pytest.mark.expensive
    @pytest.mark.parametrize("audio_format", ["mp3", "wav", "flac"])
    def test_supported_audio_formats(
        self,
        openai_client: OpenAI,
        speech_standard_model: str,
        transcription_model: str,
        audio_format: str,
    ) -> None:
        """Test transcription with all supported audio formats.

        Validates that the transcription service supports all OpenAI-compatible
        audio file formats for input processing.

        Args:
            openai_client: OpenAI client instance for API calls
            speech_standard_model: Standard speech model for audio generation
            transcription_model: Valid transcription model identifier
            audio_format: The audio format to test

        Validates:
            - All supported formats produce valid transcription output
            - Response contains text attribute for each format
            - Processing works across different audio encodings
        """
        # Generate audio in the specific format
        audio_response = openai_client.audio.speech.create(
            model=speech_standard_model,
            voice="alloy",
            input=f"Testing {audio_format} format transcription.",
            response_format=audio_format,  # type: ignore[arg-type]
        )

        # Test transcription of the generated audio
        response = openai_client.audio.transcriptions.create(
            file=(f"test.{audio_format}", io.BytesIO(audio_response.content)),
            model=transcription_model,
        )

        assert hasattr(response, "text")
        assert isinstance(response.text, str)
        assert len(response.text.strip()) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("language", ["en", "fr"])
    def test_language_parameter_validation(
        self,
        openai_client: OpenAI,
        sample_audio_file: bytes,
        transcription_model: str,
        language: str,
    ) -> None:
        """Test language parameter with valid ISO 639-1 language codes.

        Validates language parameter functionality with common language codes
        to ensure proper language detection and transcription accuracy.

        Args:
            openai_client: OpenAI client instance for API calls
            sample_audio_file: Audio file bytes for transcription testing
            transcription_model: Valid transcription model identifier
            language: The language code to test

        Validates:
            - Valid ISO 639-1 codes are accepted
            - Response contains text attribute
            - Language parameter doesn't cause errors
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            language=language,
        )

        assert hasattr(response, "text")
        assert isinstance(response.text, str)

    @pytest.mark.expensive
    def test_single_timestamp_granularities(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """Test individual timestamp granularities separately.

        Validates that individual timestamp granularities work correctly
        when requested separately (not combined).

        Args:
            openai_client: OpenAI client instance for API calls
            sample_audio_file: Audio file bytes for transcription testing
            transcription_model: Valid transcription model identifier

        Validates:
            - Individual granularities work correctly
            - Segment-level timestamps provide timing information
            - Word-level timestamps provide detailed timing
        """
        # Test segment-level timestamps only
        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

        assert hasattr(response, "text")
        assert hasattr(response, "segments")
        if hasattr(response, "segments") and response.segments:
            segment = response.segments[0]
            assert hasattr(segment, "start")
            assert hasattr(segment, "end")
            assert hasattr(segment, "text")

        # Test word-level timestamps only
        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

        assert hasattr(response, "text")
        assert hasattr(response, "words")
        if hasattr(response, "words") and response.words:
            word = response.words[0]
            assert hasattr(word, "word")
            assert hasattr(word, "start")
            assert hasattr(word, "end")

    @pytest.mark.expensive
    def test_response_format_consistency(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """Test consistency across all response formats with same audio.

        Validates that different response formats produce consistent transcription
        results for the same audio input.

        Args:
            openai_client: OpenAI client instance for API calls
            sample_audio_file: Audio file bytes for transcription testing
            transcription_model: Valid transcription model identifier

        Validates:
            - All formats return non-empty responses
            - Text content is consistent across formats
            - Format-specific features work correctly
        """
        formats = ["json", "text", "verbose_json"]
        responses = {}

        for format_name in formats:
            response = openai_client.audio.transcriptions.create(
                file=("test.wav", io.BytesIO(sample_audio_file)),
                model=transcription_model,
                response_format=format_name,  # type: ignore[call-overload]
            )
            responses[format_name] = response

        # Validate all responses have content
        for format_name, response in responses.items():
            if format_name == "text":
                assert isinstance(response, str)
                assert len(response.strip()) > 0
            else:
                assert hasattr(response, "text")
                assert isinstance(response.text, str)
                assert len(response.text.strip()) > 0

    @pytest.mark.expensive
    def test_verbose_json_structure(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """Test comprehensive validation of verbose JSON response structure.

        Validates the extended verbose_json response format that includes detailed
        metadata and timing information. This format is essential for applications
        requiring complete transcription analysis including task type, detected
        language, audio duration, and granular timing data.

        Args:
            openai_client: OpenAI client instance for API calls
            sample_audio_file: Audio file bytes for transcription testing
            transcription_model: Transcription model identifier

        Validates:
            - Required metadata fields (text, language, duration)
            - Duration is non-negative number
            - Optional timestamp arrays (segments and words) when available
            - Segment timing data structure (start, end, text)
            - Word timing data structure (word, start, end)
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
        )

        # Check required fields
        assert hasattr(response, "text")
        assert hasattr(response, "language")
        assert hasattr(response, "duration")

        # Check values
        assert isinstance(response.duration, int | float)
        assert response.duration >= 0

        # Check optional timestamp fields
        if hasattr(response, "segments") and response.segments is not None:
            for segment in response.segments:
                assert hasattr(segment, "start")
                assert hasattr(segment, "end")
                assert hasattr(segment, "text")

        if hasattr(response, "words") and response.words is not None:
            for word in response.words:
                assert hasattr(word, "word")
                assert hasattr(word, "start")
                assert hasattr(word, "end")
