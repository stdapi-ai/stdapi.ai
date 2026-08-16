"""Tests for the OpenAI /v1/audio/translations route.

With ``amazon.transcribe`` the route is an Amazon Transcribe batch job followed by
Amazon Translate into English, so the accepted ``response_format`` set is OpenAI's
``CreateTranslationRequest`` one (json, text, srt, verbose_json, vtt) and both AWS
services are billed within a single request.

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://stdapi.ai/api_openai_audio_translations/
     stdapi/routes/openai_audio_translations.py:create_translation
"""

import io
from base64 import b64encode
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError, NotFoundError, OpenAI
from starlette.responses import Response

import stdapi.aws
from stdapi import aws_translate
from stdapi.api_errors import ApiError, UnsupportedParameterError
from stdapi.config import SETTINGS
from stdapi.input_file import InputFile
from stdapi.models.audio.amazon_transcribe import AudioModel
from tests._helpers import make_client_error
from tests.conftest import _sample_cache_file, logged_usage_entries
from tests.test_openai_audio_transcriptions import _stub_transcribe

if TYPE_CHECKING:
    from openai.types.audio import Translation
    from starlette.testclient import TestClient as TestClientType


@pytest.fixture(scope="module")
def sample_audio_fr_file(openai_client: OpenAI, speech_standard_model: str) -> bytes:
    """Create a French-language sample audio file for translation usage testing.

    ``aws_translate.translate()`` returns early without billing AWS Translate
    when the detected source language is English, so exercising the billing
    path requires non-English audio. Generated via the speech endpoint (same
    pattern as conftest's ``sample_audio_mp3_file``; MP3 is a native Polly
    format) and cached under ``tests/.cache`` per producing model.
    """
    audio_file = _sample_cache_file("audio_fr", speech_standard_model, "mp3")
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
    """End-to-end behavior of /v1/audio/translations with the default STT model.

    The sample audio says "This is a test." in English, so content assertions stay
    tolerant (a single expected word) rather than matching an exact transcript.

    Ref: https://stdapi.ai/api_openai_audio_translations/
         stdapi/models/audio/amazon_transcribe.py:AudioModel.stt_translate
    """

    @pytest.mark.slow
    def test_core_translation_functionality(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """A minimal translation request returns the English text as ``text``.

        Neither ``prompt`` nor ``temperature`` is sent: ``amazon.transcribe``
        rejects both with a 400 rather than ignoring them, so a default request is
        the only shape valid for every model this fixture can resolve to.

        Ref: https://stdapi.ai/api_openai_audio_translations/
             stdapi/models/audio/__init__.py:AudioModelBase._validate_no_prompt
        """
        response = openai_client.audio.translations.create(
            file=("test.wav", io.BytesIO(sample_audio_file)), model=transcription_model
        )

        assert hasattr(response, "text")
        assert isinstance(response.text, str)
        assert len(response.text.strip()) > 0
        assert "test" in response.text.lower(), (
            f"translation does not reflect the sample audio: {response.text!r}"
        )

    @pytest.mark.slow
    def test_translation_specific_response_formats(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """``text`` returns a bare string and ``verbose_json`` reports English output.

        The verbose translation object's ``language`` is the *output* language and is
        always English — unlike the transcription surface, where it is the detected
        source language.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_audio.py:TranslationVerbose
        """
        text_response = openai_client.audio.translations.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format="text",
        )
        assert isinstance(text_response, str)
        assert len(text_response.strip()) > 0
        assert not text_response.lstrip().startswith(("{", '"')), (
            f"text format must not be JSON-encoded: {text_response[:40]!r}"
        )
        assert "test" in text_response.lower()

        verbose_response = openai_client.audio.translations.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format="verbose_json",
        )
        assert hasattr(verbose_response, "text")
        assert hasattr(verbose_response, "language")
        assert hasattr(verbose_response, "duration")
        assert isinstance(verbose_response.duration, int | float)
        assert verbose_response.duration >= 0
        assert verbose_response.language.lower() in {"english", "en"}, (
            f"translation output language must be English: {verbose_response.language!r}"
        )
        assert "test" in verbose_response.text.lower()

    def test_invalid_model_error(
        self, openai_client: OpenAI, sample_audio_file: bytes
    ) -> None:
        """An unknown model is a 404 ``model_not_found`` naming the requested model.

        Model resolution happens before the audio is staged in S3, so the file
        content is irrelevant to this outcome.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             stdapi/api_errors.py:UnsupportedModelError
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
        assert "invalid-nonexistent-model" in error_message, (
            f"error must name the rejected model: {error_message}"
        )
        assert any(
            word in error_message for word in ["model", "invalid", "exist", "access"]
        )

    def test_empty_file_error(
        self, openai_client: OpenAI, transcription_model: str
    ) -> None:
        """A zero-byte audio file is a 400 ``invalid_request_error`` with no error code.

        There is no local media validation for ``amazon.transcribe``: the file is
        staged in S3 and Transcribe's own ``BadRequestException`` is re-wrapped, which
        yields the plain base error (``code`` null) rather than a typed one.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/how-input.html
             stdapi/models/audio/amazon_transcribe.py:_handle_transcription_error
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
        assert error_body.get("code") is None
        error_message = str(error).lower()
        assert any(
            word in error_message for word in ["format", "supported", "invalid", "file"]
        )

    def test_unsupported_file_format_error(
        self, openai_client: OpenAI, transcription_model: str
    ) -> None:
        """A non-audio payload is a 400 ``invalid_request_error`` with no error code.

        Amazon Transcribe batch accepts only AMR/FLAC/M4A/MP3/MP4/Ogg/WebM/WAV and
        detects the media format itself, so a text file is rejected by the service,
        not by a local extension check.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/how-input.html
             stdapi/models/audio/amazon_transcribe.py:_handle_transcription_error
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
        assert error_body.get("code") is None
        error_message = str(error).lower()
        assert any(
            word in error_message
            for word in ["format", "supported", "invalid", "flac", "mp3", "wav"]
        )

    @pytest.mark.slow
    def test_subtitle_format_translation(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """srt/vtt translation keeps cue numbering, timings and the WEBVTT header.

        Only the cue text is replaced: the subtitle file produced by Transcribe's
        Subtitles feature is re-emitted with each text block swapped for its
        translation. Cue numbering starts at 1 because the gateway always sends
        ``OutputStartIndex=1``, overriding Transcribe's AWS-specific default of 0.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/subtitles.html
             stdapi/aws_translate.py:translate_subtitle
        """
        srt_response = openai_client.audio.translations.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format="srt",
        )

        assert isinstance(srt_response, str)
        assert len(srt_response.strip()) > 0
        assert "-->" in srt_response, f"SRT lost its cue timings: {srt_response!r}"
        assert srt_response.strip().startswith("1"), (
            f"SRT cue numbering must start at 1: {srt_response[:20]!r}"
        )
        assert "test" in srt_response.lower()

        vtt_response = openai_client.audio.translations.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format="vtt",
        )

        assert isinstance(vtt_response, str)
        assert len(vtt_response.strip()) > 0
        assert vtt_response.lstrip().startswith("WEBVTT"), (
            f"VTT lost its header: {vtt_response[:20]!r}"
        )
        assert "-->" in vtt_response, f"VTT lost its cue timings: {vtt_response!r}"
        assert "test" in vtt_response.lower()

    def test_empty_transcription_handling(
        self, openai_client: OpenAI, transcription_model: str
    ) -> None:
        """Speechless audio yields either a ``text`` payload or a clean 400, never a 500.

        A header-only WAV carries no samples: Transcribe may reject the media
        outright or complete with an empty transcript, and both outcomes must stay on
        the caller-error side of the contract. AWS Translate is skipped either way
        (empty or English text short-circuits it).

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/how-input.html
             stdapi/aws_translate.py:translate
        """
        # Create minimal audio content that might produce empty transcription
        minimal_audio = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

        response: Translation | None = None
        rejection: BadRequestError | None = None
        try:
            response = openai_client.audio.translations.create(
                file=("minimal.wav", io.BytesIO(minimal_audio)),
                model=transcription_model,
            )
        except BadRequestError as error:
            rejection = error

        if rejection is not None:
            assert rejection.status_code == 400
            error_body = rejection.body
            assert isinstance(error_body, dict)
            assert error_body["type"] == "invalid_request_error"
        else:
            assert response is not None
            assert isinstance(response.text, str)
            assert getattr(response, "duration", None) is None, (
                "the default json format must not carry verbose_json fields"
            )

    @pytest.mark.slow
    @pytest.mark.local
    def test_translation_usage_logged(
        self,
        test_client: TestClientType,
        api_key: str,
        sample_audio_fr_file: bytes,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A translation request bills both Amazon Transcribe and Amazon Translate.

        French audio is required: ``translate()`` short-circuits on an English source
        and would never produce a Translate usage record. Transcribe is billed per
        second with a 15-second per-request minimum, so the logged seconds are at
        least 15 even for a short clip.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/what-is.html
             stdapi/usage.py:record_translate_usage
        """
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

        captured = capfd.readouterr().out
        translate_entries = logged_usage_entries(
            captured, service="translate", operation="/v1/audio/translations"
        )
        assert translate_entries, "Expected translate service in usage"
        translate_entry = translate_entries[0]
        assert translate_entry["model"] == "amazon.translate"
        assert "input_characters" in translate_entry
        # Value depends on audio content
        assert translate_entry["input_characters"] > 0

        transcribe_entries = logged_usage_entries(
            captured, service="transcribe", operation="/v1/audio/translations"
        )
        assert transcribe_entries, "Expected transcribe service in usage"
        transcribe_entry = transcribe_entries[0]
        assert transcribe_entry["model"] == "amazon.transcribe"
        assert transcribe_entry["input_seconds"] >= 15, (
            "Transcribe bills a 15-second minimum per request"
        )


@pytest.mark.gateway("JSON body input not supported by the official OpenAI API")
class TestAudioTranslationsJsonBody:
    """POST /v1/audio/translations with an application/json body (gateway extension).

    The JSON path accepts the audio as a base64 string, data URI, HTTPS URL or S3
    URI, and is the only path through which provider extra parameters such as AWS
    Translate's ``Settings`` can be sent.

    Ref: https://stdapi.ai/api_openai_audio_translations/
         stdapi/types/openai_audio.py:AudioTranslationJsonBody
    """

    def test_json_body_missing_file_returns_400(
        self, openai_client: OpenAI, transcription_model: str
    ) -> None:
        """JSON body without the required file field returns 400 naming the field.

        The JSON path validates ``AudioTranslationJsonBody`` directly rather than the
        route signature, so the reported location is ``file`` — not the ``body.file``
        prefix that route-level (multipart) validation produces.

        Ref: stdapi/main.py:handle_validation_exception
             stdapi/routes/openai_audio_translations.py:AudioTranslationJsonBody
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}audio/translations",
            json={"model": transcription_model},
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        message = error["message"]
        assert "file" in message, f"error must point at the missing field: {message}"
        assert "required" in message.lower(), (
            f"error must say the field is required: {message}"
        )

    def test_json_body_invalid_model_returns_404(self, openai_client: OpenAI) -> None:
        """JSON body path reaches model validation — invalid model returns 404.

        Uses a dummy audio data URI to satisfy input validation; model lookup
        runs before any audio decoding so the file content is irrelevant.

        Ref: stdapi/api_errors.py:UnsupportedModelError
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

    @pytest.mark.slow
    def test_json_body_translation(
        self,
        openai_client: OpenAI,
        sample_audio_file_base64: str,
        transcription_model: str,
    ) -> None:
        """JSON body with an audio data URI translates and returns the json payload.

        Ref: stdapi/input_file.py:InputFile
        """
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
        assert "test" in body["text"].lower(), (
            f"translation does not reflect the sample audio: {body['text']!r}"
        )

    @pytest.mark.slow
    def test_json_body_translation_with_translate_settings(
        self,
        openai_client: OpenAI,
        sample_audio_fr_file: bytes,
        transcription_model: str,
    ) -> None:
        """AWS Translate ``Settings`` reach the real TranslateText call.

        Non-English audio is required: ``translate()`` skips AWS Translate
        entirely for English source text, which would never exercise the
        ``Settings`` plumbing. English is not a formality-capable target language,
        and Amazon Translate ignores ``Formality`` there instead of failing, so the
        request must still succeed with a translated payload.

        Ref: https://docs.aws.amazon.com/translate/latest/dg/customizing-translations-formality.html
             stdapi/models/audio/amazon_transcribe.py:_pop_translate_extra_params
        """
        http_client = openai_client._client  # noqa: SLF001
        data_uri = f"data:audio/mpeg;base64,{b64encode(sample_audio_fr_file).decode()}"
        response = http_client.post(
            f"{openai_client.base_url}audio/translations",
            json={
                "file": data_uri,
                "model": transcription_model,
                "Settings": {"Formality": "FORMAL"},
            },
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body.get("text"), str)
        assert len(body["text"].strip()) > 0
        assert any(word in body["text"].lower() for word in ("test", "simple")), (
            f"French audio was not translated into English: {body['text']!r}"
        )


@pytest.mark.local
class TestAudioTranslationsResponseFormatBugs:
    """Local stub tests for response-format regressions (no AWS calls made).

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/models/audio/amazon_transcribe.py:AudioModel._format_translation_response
    """

    def test_text_format_returns_raw_plain_text(
        self, test_client: TestClientType, api_key: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """response_format=text returns raw text/plain, not a JSON-quoted string.

        The stubbed transcript is English, so AWS Translate is skipped and the
        returned body is exactly the concatenated transcript.
        """
        _stub_transcribe(monkeypatch)

        response = test_client.post(
            "/v1/audio/translations",
            files={"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")},
            data={"model": "amazon.transcribe", "response_format": "text"},
            headers={"Authorization": f"Bearer {api_key}"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text == "hello world"


@pytest.mark.local
class TestTranslateUnsupportedParameters:
    """Amazon Transcribe rejects the translation parameters it cannot honour.

    ``prompt`` and ``temperature`` are valid OpenAI translation fields with no
    Transcribe equivalent: forwarding them would change nothing, so the backend
    fails the request instead of transcribing while ignoring them. The
    translation path runs its own validation, so it can regress independently
    of the transcription one.

    Ref: https://stdapi.ai/api_openai_audio_translations/
         stdapi/models/audio/amazon_transcribe.py:AudioModel.stt_translate
         stdapi/models/audio/__init__.py:AudioModelBase._validate_no_prompt
    """

    @staticmethod
    def _audio() -> InputFile:
        """Return a tiny data-URI audio input; it is never read.

        Returns:
            An ``InputFile`` pointing at inline base64 audio.
        """
        return InputFile("data:audio/wav;base64,AAAA")

    async def test_prompt_is_rejected(self) -> None:
        """A ``prompt`` fails with 400 before the transcription job is started."""
        with pytest.raises(UnsupportedParameterError) as exc_info:
            await AudioModel("amazon.transcribe").stt_translate(
                self._audio(), "json", prompt="Translate carefully"
            )

        assert exc_info.value.status == 400
        assert exc_info.value.code == "unsupported_parameter"
        assert exc_info.value.param == "prompt"
        assert "prompt" in str(exc_info.value)

    async def test_temperature_is_rejected(self) -> None:
        """A non-zero ``temperature`` fails with 400: Transcribe has no sampling knob."""
        with pytest.raises(UnsupportedParameterError) as exc_info:
            await AudioModel("amazon.transcribe").stt_translate(
                self._audio(), "json", prompt=None, temperature=0.5
            )

        assert exc_info.value.status == 400
        assert exc_info.value.code == "unsupported_parameter"
        assert exc_info.value.param == "temperature"
        assert "temperature" in str(exc_info.value)

    @pytest.mark.usefixtures("request_log")
    async def test_no_prompt_and_zero_temperature_are_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The documented defaults reach the backend and return the transcript.

        The stubbed transcript is English, so AWS Translate short-circuits and
        the text is returned unchanged.
        """
        _stub_transcribe(monkeypatch)

        response = await AudioModel("amazon.transcribe").stt_translate(
            self._audio(), "json", prompt=None, temperature=0.0
        )

        assert not isinstance(response, str | Response)
        assert response.text == "hello world"


class _StubLanguagesClient:
    """Translate client stub answering ``ListLanguages`` from fixed pages."""

    def __init__(self, *pages: list[str], error: Exception | None = None) -> None:
        self._pages = pages
        self._error = error
        self.translate_calls = 0

    async def list_languages(self, **kwargs: object) -> dict[str, object]:
        """Return the page addressed by ``NextToken``, or raise the fixed error.

        Args:
            **kwargs: ``ListLanguages`` request fields.

        Returns:
            One page of supported languages, with a ``NextToken`` while more
            pages remain.

        Raises:
            Exception: The error this stub was built with, when there is one.
        """
        if self._error is not None:
            raise self._error
        index = int(str(kwargs.get("NextToken", 0)))
        page: dict[str, object] = {
            "Languages": [
                {"LanguageCode": code, "LanguageName": code}
                for code in self._pages[index]
            ]
        }
        if index + 1 < len(self._pages):
            page["NextToken"] = str(index + 1)
        return page

    async def translate_text(self, **_kwargs: object) -> dict[str, str]:
        """Record a translation call and return a fixed payload.

        Returns:
            A fixed ``TranslateText`` response.
        """
        self.translate_calls += 1
        return {"TranslatedText": "translated"}


@pytest.mark.local
class TestTranslateLanguagePairValidation:
    """An unsupported language pair is refused before AWS Translate is called.

    The translations route always targets English, so only the transcript's
    source language can be unsupported — and it is only known after the audio
    has been transcribed and billed. The supported set is read once at startup
    and held, never per request.

    Ref: https://docs.aws.amazon.com/translate/latest/APIReference/API_ListLanguages.html
         stdapi/aws_translate.py:load_supported_languages
         stdapi/aws_translate.py:translate
    """

    @pytest.fixture(autouse=True)
    def _translate_regions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Serve Translate from a single known region."""
        monkeypatch.setattr(SETTINGS, "aws_translate_region", "us-east-1")

    @staticmethod
    def _patch_client(
        monkeypatch: pytest.MonkeyPatch, client: _StubLanguagesClient
    ) -> None:
        """Route every Translate client lookup to *client*."""
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, _region=None: client
        )

    async def test_unsupported_source_is_refused_without_calling_translate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A language Translate does not support answers 400 listing the codes.

        Ref: stdapi/aws_translate.py:_unsupported_pair_error
        """
        client = _StubLanguagesClient()
        self._patch_client(monkeypatch, client)
        monkeypatch.setattr(aws_translate, "_SUPPORTED_LANGUAGE_CODES", {"en", "es"})

        with pytest.raises(ApiError) as exc_info:
            await aws_translate.translate("dumela", "tn-ZA")

        assert exc_info.value.status == 400
        assert "en, es" in str(exc_info.value)
        assert client.translate_calls == 0

    @pytest.mark.usefixtures("request_log")
    async def test_supported_source_still_reaches_translate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A supported pair is translated as before.

        Ref: stdapi/aws_translate.py:translate
        """
        client = _StubLanguagesClient()
        self._patch_client(monkeypatch, client)
        monkeypatch.setattr(aws_translate, "_SUPPORTED_LANGUAGE_CODES", {"en", "es"})

        assert await aws_translate.translate("hola", "es-US") == "translated"
        assert client.translate_calls == 1

    @pytest.mark.usefixtures("request_log")
    async def test_validation_is_off_while_the_catalog_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the catalog the pair is left to AWS, as it was before.

        A deployment whose role lacks ``translate:ListLanguages`` must keep
        translating rather than refuse every request.

        Ref: stdapi/aws_translate.py:translate
        """
        client = _StubLanguagesClient()
        self._patch_client(monkeypatch, client)
        monkeypatch.setattr(aws_translate, "_SUPPORTED_LANGUAGE_CODES", set())

        assert await aws_translate.translate("dumela", "tn-ZA") == "translated"
        assert client.translate_calls == 1

    async def test_load_reads_every_page_of_the_catalog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``ListLanguages`` is paginated, so every page reaches the cache.

        Ref: stdapi/aws_translate.py:load_supported_languages
        """
        codes: set[str] = set()
        monkeypatch.setattr(aws_translate, "_SUPPORTED_LANGUAGE_CODES", codes)
        self._patch_client(
            monkeypatch, _StubLanguagesClient(["en", "es"], ["fr", "de"])
        )

        await aws_translate.load_supported_languages()

        assert codes == {"en", "es", "fr", "de"}

    async def test_load_failure_leaves_validation_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refused ``ListLanguages`` never fails startup nor blocks requests.

        Ref: stdapi/aws_translate.py:load_supported_languages
        """
        codes: set[str] = set()
        monkeypatch.setattr(aws_translate, "_SUPPORTED_LANGUAGE_CODES", codes)
        self._patch_client(
            monkeypatch,
            _StubLanguagesClient(
                error=make_client_error("AccessDeniedException", "ListLanguages")
            ),
        )

        await aws_translate.load_supported_languages()

        assert not codes
