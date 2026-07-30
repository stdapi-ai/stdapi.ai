"""Tests for /v1/audio/translations on Mistral Voxtral (native Bedrock translation).

Voxtral translates inside the model instead of chaining Amazon Transcribe and
Amazon Translate, so it supports only the ``json`` and ``text`` response formats.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-voxtral-mini-3b-2507.html
     stdapi/models/audio/mistral_voxtral.py:AudioModel.stt_translate
"""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

VOXTRAL_MINI = "mistral.voxtral-mini-3b-2507"

VOXTRAL_ALL = (VOXTRAL_MINI,)
VOXTRAL_SAMPLE = (VOXTRAL_MINI,)


class TestMistralVoxtralTranslations:
    """Basic behavior checks for Mistral Voxtral translation models.

    The sample audio says "This is a test." in English, so the translated output
    is expected to keep that wording; assertions stay tolerant because the
    wording comes from a generative model.

    Ref: https://stdapi.ai/api_openai_audio_translations/
    """

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
        assert "test" in response.text.lower(), (
            f"translated text does not reflect the sample audio: {response.text!r}"
        )

    @pytest.mark.parametrize("model_id", VOXTRAL_ALL)
    def test_translation_text_format(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_audio_mp3_file: bytes,
        model_id: str,
    ) -> None:
        """Text format returns the plain English string, not a JSON envelope."""
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
        assert not response.lstrip().startswith(("{", '"')), (
            f"text format must not be JSON-encoded: {response[:40]!r}"
        )
        assert "test" in response.lower(), (
            f"translated text does not reflect the sample audio: {response!r}"
        )
