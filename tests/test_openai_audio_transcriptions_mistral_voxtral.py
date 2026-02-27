"""Basic tests for Mistral Voxtral audio transcription models via OpenAI-compatible API."""

from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError

if TYPE_CHECKING:
    from openai import OpenAI

VOXTRAL_MINI = "mistral.voxtral-mini-3b-2507"

VOXTRAL_ALL = (VOXTRAL_MINI,)
VOXTRAL_SAMPLE = (VOXTRAL_MINI,)


class TestMistralVoxtralTranscriptions:
    """Basic behavior checks for Mistral Voxtral transcription models."""

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", VOXTRAL_ALL)
    def test_basic_transcription_json(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_audio_mp3_file: bytes,
        model_id: str,
    ) -> None:
        """Basic transcription returns JSON with text and usage tokens."""
        if use_official_api:
            pytest.skip(
                "Mistral Voxtral models are not available on the official OpenAI API"
            )

        response = openai_client.audio.transcriptions.create(
            file=("test.mp3", sample_audio_mp3_file),
            model=model_id,
            response_format="json",
        )

        assert hasattr(response, "text")
        assert isinstance(response.text, str)
        assert len(response.text.strip()) > 0

        # Validate usage information
        assert hasattr(response, "usage")
        assert response.usage is not None
        assert hasattr(response.usage, "input_tokens")
        assert hasattr(response.usage, "output_tokens")
        assert hasattr(response.usage, "total_tokens")
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert (
            response.usage.total_tokens
            == response.usage.input_tokens + response.usage.output_tokens
        )

        # Validate input token details
        if (
            hasattr(response.usage, "input_token_details")
            and response.usage.input_token_details
        ):
            assert hasattr(response.usage.input_token_details, "audio_tokens")
            audio_tokens = response.usage.input_token_details.audio_tokens
            assert audio_tokens is not None

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", VOXTRAL_ALL)
    def test_transcription_text_format(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_audio_mp3_file: bytes,
        model_id: str,
    ) -> None:
        """Text format returns plain string without usage metadata."""
        if use_official_api:
            pytest.skip(
                "Mistral Voxtral models are not available on the official OpenAI API"
            )

        response = openai_client.audio.transcriptions.create(
            file=("test.mp3", sample_audio_mp3_file),
            model=model_id,
            response_format="text",
        )

        assert isinstance(response, str)
        assert len(response.strip()) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", VOXTRAL_SAMPLE)
    def test_transcription_with_temperature(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_audio_mp3_file: bytes,
        model_id: str,
    ) -> None:
        """Temperature parameter is accepted and processed correctly."""
        if use_official_api:
            pytest.skip(
                "Mistral Voxtral models are not available on the official OpenAI API"
            )

        response = openai_client.audio.transcriptions.create(
            file=("test.mp3", sample_audio_mp3_file),
            model=model_id,
            response_format="json",
            temperature=0.7,
        )

        assert hasattr(response, "text")
        assert isinstance(response.text, str)
        assert len(response.text.strip()) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", VOXTRAL_SAMPLE)
    def test_transcription_with_prompt(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_audio_mp3_file: bytes,
        model_id: str,
    ) -> None:
        """Prompt parameter guides transcription style."""
        if use_official_api:
            pytest.skip(
                "Mistral Voxtral models are not available on the official OpenAI API"
            )

        response = openai_client.audio.transcriptions.create(
            file=("test.mp3", sample_audio_mp3_file),
            model=model_id,
            response_format="json",
            prompt="This is a test audio file for transcription.",
        )

        assert hasattr(response, "text")
        assert isinstance(response.text, str)
        assert len(response.text.strip()) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", VOXTRAL_SAMPLE)
    def test_transcription_with_language(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_audio_mp3_file: bytes,
        model_id: str,
    ) -> None:
        """Language parameter specifies input audio language."""
        if use_official_api:
            pytest.skip(
                "Mistral Voxtral models are not available on the official OpenAI API"
            )

        response = openai_client.audio.transcriptions.create(
            file=("test.mp3", sample_audio_mp3_file),
            model=model_id,
            response_format="json",
            language="en",
        )

        assert hasattr(response, "text")
        assert isinstance(response.text, str)
        assert len(response.text.strip()) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", VOXTRAL_SAMPLE)
    def test_transcription_all_parameters(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_audio_mp3_file: bytes,
        model_id: str,
    ) -> None:
        """All supported parameters work together correctly."""
        if use_official_api:
            pytest.skip(
                "Mistral Voxtral models are not available on the official OpenAI API"
            )

        response = openai_client.audio.transcriptions.create(
            file=("test.mp3", sample_audio_mp3_file),
            model=model_id,
            response_format="json",
            language="en",
            prompt="Test audio transcription.",
            temperature=0.5,
        )

        assert hasattr(response, "text")
        assert isinstance(response.text, str)
        assert len(response.text.strip()) > 0
        assert hasattr(response, "usage")
        assert response.usage is not None

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", VOXTRAL_ALL)
    def test_streaming_transcription(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_audio_mp3_file: bytes,
        model_id: str,
    ) -> None:
        """Streaming produces delta and done events with usage tokens."""
        if use_official_api:
            pytest.skip(
                "Mistral Voxtral models are not available on the official OpenAI API"
            )

        response = openai_client.audio.transcriptions.create(
            file=("test.mp3", sample_audio_mp3_file), model=model_id, stream=True
        )

        chunks = []
        accumulated_text = ""
        has_delta_events = False
        has_done_event = False

        for chunk in response:
            chunks.append(chunk)

            if chunk.type == "transcript.text.delta":
                has_delta_events = True
                if chunk.delta:
                    accumulated_text += chunk.delta
                    assert chunk.delta.strip()

            elif chunk.type == "transcript.text.done":
                has_done_event = True
                assert hasattr(chunk, "text")
                assert hasattr(chunk, "usage")

                if chunk.usage:
                    assert hasattr(chunk.usage, "input_tokens")
                    assert hasattr(chunk.usage, "output_tokens")
                    assert chunk.usage.input_tokens > 0
                    assert chunk.usage.output_tokens > 0
                    assert hasattr(chunk.usage, "input_token_details")
                    assert chunk.usage.input_token_details is not None
                    assert hasattr(chunk.usage.input_token_details, "audio_tokens")
                    assert chunk.usage.input_token_details.audio_tokens is not None
                    assert chunk.usage.input_token_details.audio_tokens > 0

        assert len(chunks) > 0
        assert has_delta_events
        assert has_done_event
        assert accumulated_text.strip()

    @pytest.mark.parametrize("model_id", VOXTRAL_SAMPLE)
    def test_invalid_audio_file(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """Non-audio file format raises BadRequestError."""
        if use_official_api:
            pytest.skip(
                "Mistral Voxtral models are not available on the official OpenAI API"
            )

        with pytest.raises(BadRequestError):
            openai_client.audio.transcriptions.create(
                file=("test.txt", b"This is not an audio file"), model=model_id
            )

    @pytest.mark.parametrize("model_id", VOXTRAL_SAMPLE)
    def test_verbose_json_unsupported(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_audio_mp3_file: bytes,
        model_id: str,
    ) -> None:
        """Verbose JSON format is not supported by Voxtral models."""
        if use_official_api:
            pytest.skip(
                "Mistral Voxtral models are not available on the official OpenAI API"
            )

        with pytest.raises(BadRequestError):
            openai_client.audio.transcriptions.create(
                file=("test.mp3", sample_audio_mp3_file),
                model=model_id,
                response_format="verbose_json",
            )

    @pytest.mark.parametrize("model_id", VOXTRAL_SAMPLE)
    def test_srt_format_unsupported(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_audio_mp3_file: bytes,
        model_id: str,
    ) -> None:
        """SRT subtitle format is not supported by Voxtral models."""
        if use_official_api:
            pytest.skip(
                "Mistral Voxtral models are not available on the official OpenAI API"
            )

        with pytest.raises(BadRequestError):
            openai_client.audio.transcriptions.create(
                file=("test.mp3", sample_audio_mp3_file),
                model=model_id,
                response_format="srt",
            )
