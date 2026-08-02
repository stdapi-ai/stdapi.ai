"""Tests for /v1/audio/translations on Mistral Voxtral (native Bedrock translation).

Voxtral translates inside the model instead of chaining Amazon Transcribe and
Amazon Translate, so it supports only the ``json`` and ``text`` response formats.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-voxtral-mini-3b-2507.html
     stdapi/models/audio/_default.py:AudioModel.stt_translate
"""

from typing import TYPE_CHECKING

import pytest

from stdapi.api_errors import ApiError
from stdapi.input_file import InputFile
from stdapi.models.audio.mistral_voxtral import AudioModel

if TYPE_CHECKING:
    from openai import OpenAI

VOXTRAL_MINI = "mistral.voxtral-mini-3b-2507"

VOXTRAL_ALL = (VOXTRAL_MINI,)
VOXTRAL_SAMPLE = (VOXTRAL_MINI,)

#: Every test needs the gateway: Voxtral has no official OpenAI equivalent.
pytestmark = pytest.mark.gateway(
    "Mistral Voxtral models are not available on the official OpenAI API"
)


class TestMistralVoxtralTranslations:
    """Basic behavior checks for Mistral Voxtral translation models.

    The sample audio says "This is a test." in English, so the translated output
    is expected to keep that wording; assertions stay tolerant because the
    wording comes from a generative model.

    Ref: https://stdapi.ai/api_openai_audio_translations/
    """

    @pytest.mark.parametrize("model_id", VOXTRAL_ALL)
    def test_basic_translation_json(
        self, openai_client: OpenAI, sample_audio_mp3_file: bytes, model_id: str
    ) -> None:
        """Basic translation returns JSON with translated English text."""
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
        self, openai_client: OpenAI, sample_audio_mp3_file: bytes, model_id: str
    ) -> None:
        """Text format returns the plain English string, not a JSON envelope."""
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


@pytest.mark.local
class TestMistralVoxtralTranslationsResponseFormats:
    """Voxtral translation accepts only ``json`` and ``text``.

    ``stt_translate`` runs its own ``_validate_response_formats`` check, so the
    subtitle and verbose formats are refused there independently of the
    transcription surface, before any Bedrock invocation.

    Ref: https://stdapi.ai/api_openai_audio_translations/
         stdapi/models/audio/_default.py:AudioModel.stt_translate
         stdapi/models/audio/__init__.py:AudioModelBase._validate_response_formats
    """

    @pytest.mark.parametrize("response_format", ["verbose_json", "srt", "vtt"])
    async def test_unsupported_response_format_is_rejected(
        self, response_format: str
    ) -> None:
        """A format Voxtral cannot produce fails with a 400 naming it."""
        with pytest.raises(ApiError) as exc_info:
            await AudioModel(VOXTRAL_MINI).stt_translate(
                InputFile("data:audio/mp3;base64,AAAA"),
                response_format,  # type: ignore[arg-type]
                None,
            )

        assert exc_info.value.status == 400
        assert response_format in str(exc_info.value)
        assert "not supported" in str(exc_info.value)

    def test_supported_response_formats_are_json_and_text(self) -> None:
        """The model declares exactly the two formats it can serve."""
        assert frozenset({"json", "text"}) == AudioModel.SUPPORTED_RESPONSES_FORMATS
