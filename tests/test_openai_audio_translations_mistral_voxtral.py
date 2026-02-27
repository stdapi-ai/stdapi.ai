"""Basic tests for Mistral Voxtral audio translation models via OpenAI-compatible API."""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

VOXTRAL_MINI = "mistral.voxtral-mini-3b-2507"

VOXTRAL_ALL = (VOXTRAL_MINI,)
VOXTRAL_SAMPLE = (VOXTRAL_MINI,)


class TestMistralVoxtralTranslations:
    """Basic behavior checks for Mistral Voxtral translation models."""

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", VOXTRAL_ALL)
    def test_basic_translation_json(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_audio_mp3_file: bytes,
        model_id: str,
    ) -> None:
        """Basic translation returns JSON with translated English text."""
        if use_official_api:
            pytest.skip(
                "Mistral Voxtral models are not available on the official OpenAI API"
            )

        response = openai_client.audio.translations.create(
            file=("test.mp3", sample_audio_mp3_file),
            model=model_id,
            response_format="json",
        )

        assert hasattr(response, "text")
        assert isinstance(response.text, str)
        assert len(response.text.strip()) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", VOXTRAL_ALL)
    def test_translation_text_format(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_audio_mp3_file: bytes,
        model_id: str,
    ) -> None:
        """Text format returns plain English string."""
        if use_official_api:
            pytest.skip(
                "Mistral Voxtral models are not available on the official OpenAI API"
            )

        response = openai_client.audio.translations.create(
            file=("test.mp3", sample_audio_mp3_file),
            model=model_id,
            response_format="text",
        )

        assert isinstance(response, str)
        assert len(response.strip()) > 0
