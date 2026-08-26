"""Tests for the OpenAI-compatible ``/v1/audio/transcriptions`` route.

Ref: https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
     https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     stdapi/routes/openai_audio_transcriptions.py:create_transcription
"""

import io
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from aws_sdk_transcribe_streaming.models import (
    Alternative,
    Item,
    ItemType,
    Result,
    Transcript,
    TranscriptEvent,
)
from openai import BadRequestError, NotFoundError, OpenAI
from pydantic import ValidationError
from starlette.responses import Response

from stdapi import aws_bedrock, usage
from stdapi.api_errors import ApiError, UnsupportedModelError, UnsupportedParameterError
from stdapi.config import SETTINGS, _Settings
from stdapi.input_file import InputFile
from stdapi.models.audio._default import AudioModel as DefaultAudioModel
from stdapi.models.audio.amazon_nova_sonic import AudioModel as NovaSonicAudioModel
from stdapi.models.audio.amazon_transcribe import (
    AudioModel,
    _build_transcription_job_params,
    _stream_input,
    _StreamedTranscript,
    _TranscribeExtraParams,
    _TranscribeLanguageIdSetting,
)
from stdapi.routes import openai_audio_transcriptions
from stdapi.types.openai_audio import (
    AudioResponseFormat,
    TranscriptionCreateParams,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
    TranscriptionTextSegmentEvent,
)
from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator

    from starlette.testclient import TestClient as TestClientType

    from stdapi.types.openai_audio import TranscriptionStreamEvent
    from stdapi.usage import UsageLogEntry

from stdapi.models.audio.amazon_transcribe import TranscribeJobData

#: Stubbed AWS Transcribe job result used by response-format regression tests.
_STUB_TRANSCRIPT_DATA: TranscribeJobData = {
    "transcripts": [{"transcript": "hello world"}],
    "audio_segments": [
        {"id": 0, "start_time": "0.0", "end_time": "1.0", "transcript": "hello"},
        {"id": 1, "start_time": "1.0", "end_time": "2.0", "transcript": "world"},
    ],
    "items": [
        {
            "type": "pronunciation",
            "alternatives": [{"content": "hello"}],
            "start_time": "0.0",
            "end_time": "0.5",
        },
        {
            "type": "pronunciation",
            "alternatives": [{"content": "world"}],
            "start_time": "1.0",
            "end_time": "1.5",
        },
    ],
    "language_code": "en-US",
}

#: Words spoken by the ``sample_audio_file`` fixture ("This is a test.").
_SAMPLE_AUDIO_WORDS = ("test", "this")


def _stub_transcribe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace AWS Transcribe with the fixed ``_STUB_TRANSCRIPT_DATA`` job result.

    Shared with ``test_openai_audio_translations.py``: both exercise
    ``AudioModel._transcribe`` against the same amazon_transcribe backend.
    """

    async def _fake_transcribe(
        _self: AudioModel, *_args: object, **_kwargs: object
    ) -> TranscribeJobData:
        return _STUB_TRANSCRIPT_DATA

    monkeypatch.setattr(AudioModel, "_transcribe", _fake_transcribe)


def _matches_sample_audio(transcript: str) -> bool:
    """Return whether *transcript* is a plausible transcript of the sample audio.

    Args:
        transcript: The transcript returned by the endpoint.

    Returns:
        True when the transcript carries at least one of the spoken words.
    """
    text = transcript.strip()
    assert text, "Transcription returned an empty transcript"
    return any(word in text.lower() for word in _SAMPLE_AUDIO_WORDS)


class TestAudioTranscriptions:
    """Transcription behavior shared by every transcription-capable model.

    On this gateway the route is served by Amazon Transcribe: ``srt``/``vtt``
    come from an S3-staged ``StartTranscriptionJob``'s Subtitles feature, while
    ``stream=true`` runs a live ``StartStreamTranscription`` session whenever the
    request names a language it can be opened with.

    Ref: https://stdapi.ai/api_openai_audio_transcriptions/
         https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartTranscriptionJob.html
         stdapi/models/audio/amazon_transcribe.py:AudioModel
    """

    @pytest.mark.slow
    def test_basic_transcription(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """Transcription with default parameters returns the transcript as JSON.

        ``response_format`` defaults to ``json``, whose only required field is
        ``text``. The ``sample_audio_file`` fixture speaks "This is a test.", so
        the transcript is matched tolerantly against those words.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/models/audio/amazon_transcribe.py:AudioModel.stt
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)), model=transcription_model
        )

        assert isinstance(response.text, str)
        text = response.text.strip()
        assert text, "Transcription returned an empty transcript"
        assert any(word in text.lower() for word in _SAMPLE_AUDIO_WORDS), (
            f"Transcript does not match the sample audio: {text!r}"
        )

    @pytest.mark.slow
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
        """Every ``response_format`` value returns its documented payload shape.

        ``json``/``verbose_json`` deserialize to objects, ``text``/``srt``/``vtt``
        to raw strings. Subtitle output is produced by Transcribe's Subtitles
        feature with ``OutputStartIndex=1`` (AWS defaults to 0), so the first SubRip
        cue is numbered 1 and WebVTT output starts with its mandatory signature.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             https://docs.aws.amazon.com/transcribe/latest/dg/subtitles.html
             stdapi/models/audio/amazon_transcribe.py:_build_transcription_job_params
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format=response_format,  # type: ignore[call-overload]
        )

        if response_format == "json":
            assert isinstance(response.text, str)
            assert _matches_sample_audio(response.text)
        elif response_format == "text":
            assert isinstance(response, str)
            assert _matches_sample_audio(response)
        elif response_format in ("srt", "vtt"):
            assert isinstance(response, str)
            subtitles = response.strip()
            assert "-->" in subtitles, (
                f"No cue timings in {response_format} output: {subtitles!r}"
            )
            if response_format == "vtt":
                assert subtitles.startswith("WEBVTT"), (
                    f"WebVTT output misses its signature: {subtitles[:32]!r}"
                )
            else:
                assert subtitles.startswith("1"), (
                    f"SubRip cues must start at index 1: {subtitles[:32]!r}"
                )
        elif response_format == "verbose_json":
            assert _matches_sample_audio(response.text)
            assert response.language, "verbose_json must report the audio language"
            assert isinstance(response.duration, int | float)
            assert response.duration > 0
            assert response.segments, (
                "verbose_json must default to segment-level timestamps"
            )

    @pytest.mark.slow
    def test_timestamp_granularities(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """Requesting both granularities populates ``segments`` and ``words``.

        Word timings are derived from Transcribe's ``items`` (pronunciation entries
        only) and segment timings from its ``audio_segments``, so both arrays are
        ordered by start time and every interval is non-negative.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/models/audio/amazon_transcribe.py:_format_json_response
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
        )

        assert response.text.strip()

        segments = response.segments
        assert segments, "segment granularity requested but no segments returned"
        for segment in segments:
            assert segment.start >= 0
            assert segment.end >= segment.start

        words = response.words
        assert words, "word granularity requested but no words returned"
        for word in words:
            assert word.word.strip(), "word entry without text"
            assert word.start >= 0
            assert word.end >= word.start
        word_starts = [word.start for word in words]
        assert word_starts == sorted(word_starts), (
            f"Word timestamps are not chronological: {word_starts}"
        )

    @pytest.mark.slow
    def test_streaming_transcription(
        self,
        openai_client: OpenAI,
        sample_audio_file: bytes,
        transcription_stream_model: str,
    ) -> None:
        """``stream=true`` emits ``transcript.text.delta`` events carrying the transcript.

        How many deltas a transcript is cut into is the recognizer's decision, on
        either target, so the sequence is asserted rather than the split: delta
        events carrying text, and the ``transcript.text.done`` event closing them.

        ``language`` is sent because it is what lets Amazon Transcribe stream the
        recording live instead of once it has been read whole.

        Ref: https://developers.openai.com/api/docs/guides/speech-to-text
             https://docs.aws.amazon.com/transcribe/latest/APIReference/API_streaming_StartStreamTranscription.html
             stdapi/models/audio/amazon_transcribe.py:AudioModel.stt_stream
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_stream_model,
            language="en",
            stream=True,
        )

        chunks = []
        accumulated_text = ""
        has_delta_events = False

        for chunk in response:
            chunks.append(chunk)

            assert hasattr(chunk, "type"), f"Chunk missing 'type' attribute: {chunk}"

            if chunk.type == "transcript.text.delta":
                has_delta_events = True
                assert hasattr(chunk, "delta"), (
                    f"Delta chunk missing 'delta' attribute: {chunk}"
                )

                if chunk.delta:
                    accumulated_text += chunk.delta
                    assert chunk.delta.strip(), (
                        f"Delta text is empty or whitespace: '{chunk.delta}'"
                    )

            # Cap the chunks read so a long transcript cannot stall the test.
            if len(chunks) >= 15:
                break

        assert len(chunks) > 0, "No streaming chunks received"
        assert has_delta_events, "No delta transcription events received"
        assert accumulated_text.strip(), (
            f"No meaningful text accumulated from delta events: '{accumulated_text}'"
        )

        # Match the sample audio's words tolerantly rather than the exact ASR output.
        final_text = accumulated_text.strip()

        final_text_lower = final_text.lower()
        expected_words = ["test", "audio", "file"]
        word_matches = sum(1 for word in expected_words if word in final_text_lower)
        assert word_matches >= 1, (
            f"Transcription doesn't contain expected content: '{final_text}'"
        )

        assert len(final_text) > 10, f"Transcription text too short: '{final_text}'"
        assert not final_text.isdigit(), (
            f"Transcription contains only digits: '{final_text}'"
        )

        delta_chunks = [c for c in chunks if c.type == "transcript.text.delta"]
        assert len(delta_chunks) >= 1, (
            f"Expected multiple delta chunks, got {len(delta_chunks)}"
        )

        # The terminating done event is only reached when the chunk cap did not truncate.
        if len(chunks) < 15:
            done_chunks = [c for c in chunks if c.type == "transcript.text.done"]
            assert len(done_chunks) == 1, (
                f"Expected exactly one done event, got {len(done_chunks)}"
            )
            done_text = done_chunks[0].text
            assert done_text.strip(), "the done event must carry the transcript"
            assert _matches_sample_audio(done_text), (
                f"Done event text does not match the sample audio: {done_text!r}"
            )

    def test_empty_file_error(
        self, openai_client: OpenAI, transcription_model: str
    ) -> None:
        """An empty audio file is rejected as a 400 ``invalid_request_error``.

        A zero-byte object is staged to S3 like any other upload, so the rejection
        comes from Transcribe (``BadRequestException`` at job start, or the job's
        ``FailureReason``) and is re-emitted in OpenAI's error envelope.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/how-input.html
             stdapi/models/audio/amazon_transcribe.py:_handle_transcription_error
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
        assert error_body["message"].strip(), "Error envelope carries no message"
        error_message = str(error).lower()
        assert any(
            word in error_message for word in ["format", "supported", "invalid", "file"]
        )

    def test_invalid_model_error(
        self, openai_client: OpenAI, sample_audio_file: bytes
    ) -> None:
        """An unknown model is rejected with 404 ``model_not_found``.

        Model resolution runs before any audio is read, and the message names the
        rejected identifier ("The model `X` does not exist or you do not have
        access to it.").

        Ref: https://stdapi.ai/api_openai_audio_transcriptions/
             stdapi/api_errors.py:UnsupportedModelError
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
        assert "invalid-nonexistent-model" in error_body["message"], (
            f"Error does not name the rejected model: {error_body['message']!r}"
        )
        error_message = str(error).lower()
        assert any(
            word in error_message for word in ["model", "invalid", "exist", "access"]
        )

    def test_invalid_language_error(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """An unsupported ``language`` yields 400 ``invalid_language_format``.

        The gateway expands the OpenAI ISO-639-1 code into a Transcribe locale, so an
        unknown code only fails at ``StartTranscriptionJob``; a ``BadRequestException``
        mentioning ``languageCode`` is remapped to this dedicated error code, which
        echoes the code exactly as the caller sent it.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/supported-languages.html
             stdapi/api_errors.py:InvalidLanguageFormatError
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
        assert "invalid-lang" in error_body["message"], (
            f"Error does not echo the rejected language: {error_body['message']!r}"
        )
        error_message = str(error).lower()
        assert any(
            word in error_message for word in ["language", "invalid", "iso", "format"]
        )

    def test_invalid_response_format_error(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """A ``response_format`` outside the enum is rejected with 400 and no error code.

        ``AudioResponseFormat`` is a ``Literal``, so the request fails in request-body
        validation; the gateway reports such failures as ``invalid_request_error``
        with ``code`` unset and lists the accepted values in the message.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/main.py:handle_validation_exception
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
        use_official_api: bool,
    ) -> None:
        """``temperature`` is accepted and still returns a transcript.

        Amazon Transcribe exposes no sampling temperature, so the gateway rejects the
        parameter outright and only the official API exercises this path.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/models/audio/__init__.py:AudioModelBase._validate_no_temperature
        """
        if not use_official_api:
            pytest.skip("Parameter is not supported by Amazon Transcribe.")

        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            temperature=temperature,
        )

        assert isinstance(response.text, str)
        text = response.text.strip()
        assert text, f"Empty transcript with temperature={temperature}"
        assert any(word in text.lower() for word in _SAMPLE_AUDIO_WORDS), (
            f"Transcript does not match the sample audio: {text!r}"
        )

    def test_unsupported_file_format_error(
        self, openai_client: OpenAI, transcription_model: str
    ) -> None:
        """A non-audio payload is rejected as a 400 ``invalid_request_error``.

        The gateway performs no local media sniffing: the text file is staged to S3
        and Transcribe, which accepts only AMR/FLAC/M4A/MP3/MP4/Ogg/WebM/WAV in batch
        mode, fails the job.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/how-input.html
             stdapi/models/audio/amazon_transcribe.py:AudioModel._transcribe
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
        assert error_body["message"].strip(), "Error envelope carries no message"
        error_message = str(error).lower()
        assert any(
            word in error_message
            for word in ["format", "supported", "invalid", "flac", "mp3", "wav"]
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("audio_format", ["mp3", "wav", "flac"])
    def test_supported_audio_formats(
        self,
        openai_client: OpenAI,
        speech_standard_model: str,
        transcription_model: str,
        sample_audio_file: bytes,
        sample_audio_mp3_file: bytes,
        audio_format: str,
    ) -> None:
        """mp3, wav and flac uploads are all transcribed.

        All three are in Transcribe's batch media-format set, and every sample
        speaks the same sentence, so the transcript pins that the container was
        really decoded. wav and mp3 reuse the session-cached samples; only flac
        has no cached equivalent and is synthesized here.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/how-input.html
             stdapi/models/audio/amazon_transcribe.py:AudioModel._transcribe
        """
        if audio_format == "wav":
            audio = sample_audio_file
        elif audio_format == "mp3":
            audio = sample_audio_mp3_file
        else:
            audio = openai_client.audio.speech.create(
                model=speech_standard_model,
                voice="alloy",
                input="This is a test.",
                response_format=audio_format,  # type: ignore[arg-type]
            ).content

        response = openai_client.audio.transcriptions.create(
            file=(f"test.{audio_format}", io.BytesIO(audio)), model=transcription_model
        )

        assert isinstance(response.text, str)
        assert _matches_sample_audio(response.text), (
            f"Transcript does not match the {audio_format} audio: {response.text!r}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("language", ["en", "fr"])
    def test_language_parameter_validation(
        self,
        openai_client: OpenAI,
        sample_audio_file: bytes,
        transcription_model: str,
        language: str,
    ) -> None:
        """ISO-639-1 ``language`` codes are accepted and expanded to Transcribe locales.

        Transcribe requires a full locale, so the gateway maximizes the code
        (``en`` → ``en-US``, ``fr`` → ``fr-FR``) before starting the job. Only the
        matching language can be checked against the transcript: forcing ``fr`` on
        English audio legitimately yields unrelated — possibly empty — text.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/supported-languages.html
             stdapi/utils.py:format_language_code
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            language=language,
        )

        assert isinstance(response.text, str)
        if language == "en":
            text = response.text.strip()
            assert text, "Empty transcript for the audio's own language"
            assert any(word in text.lower() for word in _SAMPLE_AUDIO_WORDS), (
                f"Transcript does not match the sample audio: {text!r}"
            )

    @pytest.mark.slow
    def test_single_timestamp_granularities(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """Each granularity populates only its own array and leaves the other unset.

        Requesting ``["segment"]`` must not return ``words`` and requesting
        ``["word"]`` must not return ``segments``: the gateway builds each array
        strictly from the requested granularities.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/models/audio/amazon_transcribe.py:_format_json_response
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

        assert response.text.strip()
        segments = response.segments
        assert segments, "segment granularity requested but no segments returned"
        for segment in segments:
            assert segment.start >= 0
            assert segment.end >= segment.start
        assert response.words is None, "words returned without word granularity"

        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

        assert response.text.strip()
        words = response.words
        assert words, "word granularity requested but no words returned"
        for word in words:
            assert word.word.strip(), "word entry without text"
            assert word.start >= 0
            assert word.end >= word.start
        assert response.segments is None, (
            "segments returned without segment granularity"
        )

    @pytest.mark.slow
    def test_verbose_json_structure(
        self, openai_client: OpenAI, sample_audio_file: bytes, transcription_model: str
    ) -> None:
        """``verbose_json`` segments carry the full Whisper-shaped confidence fields.

        ``avg_logprob``, ``no_speech_prob`` and ``compression_ratio`` have no Amazon
        Transcribe equivalent and are synthesized by the gateway, so only their
        documented ranges are asserted (log-probabilities are never positive,
        probabilities stay within 0..1) rather than specific values.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/models/audio/amazon_transcribe.py:_build_transcription_segment
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_model,
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
        )

        assert response.text.strip()
        assert response.language, "verbose_json must report the audio language"
        assert isinstance(response.duration, int | float)
        assert response.duration > 0

        segments = response.segments
        assert segments, "segment granularity requested but no segments returned"
        segment_ids = [segment.id for segment in segments]
        assert segment_ids == sorted(segment_ids), (
            f"Segment ids are not increasing: {segment_ids}"
        )
        assert len(set(segment_ids)) == len(segment_ids), (
            f"Duplicate segment ids: {segment_ids}"
        )
        for segment in segments:
            assert segment.start >= 0
            assert segment.end >= segment.start
            assert segment.seek >= 0
            assert segment.avg_logprob <= 0.0, (
                f"avg_logprob must be a log probability: {segment.avg_logprob}"
            )
            assert 0.0 <= segment.no_speech_prob <= 1.0
            assert segment.compression_ratio >= 0.0
            assert isinstance(segment.tokens, list)

        words = response.words
        assert words, "word granularity requested but no words returned"
        for word in words:
            assert word.word.strip(), "word entry without text"
            assert word.start >= 0
            assert word.end >= word.start

    @pytest.mark.slow
    def test_diarized_json_response(
        self,
        openai_client: OpenAI,
        sample_audio_file: bytes,
        transcription_diarize_model: str,
    ) -> None:
        """``diarized_json`` returns speaker-labelled segments and ``task="transcribe"``.

        Transcribe labels speakers ``spk_0``…``spk_29``; the gateway renumbers them in
        first-appearance order to the sequential capital letters OpenAI documents, so
        the first speaker encountered is always ``A``.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/diarization.html
             stdapi/models/audio/amazon_transcribe.py:_format_diarized_json_response
        """
        response = openai_client.audio.transcriptions.create(
            file=("test.wav", io.BytesIO(sample_audio_file)),
            model=transcription_diarize_model,
            response_format="diarized_json",
        )

        assert isinstance(response.text, str)
        assert response.text.strip(), "Diarized response carries no transcript"
        assert response.task == "transcribe"
        assert response.duration > 0

        segments = response.segments
        assert segments, "Diarization returned no speaker segments"
        for segment in segments:
            assert isinstance(segment.id, str)
            assert segment.id.startswith("seg_")
            assert isinstance(segment.start, int | float)
            assert isinstance(segment.end, int | float)
            assert segment.end > 0
            assert isinstance(segment.speaker, str)
            assert isinstance(segment.text, str)
            assert segment.type == "transcript.text.segment"

            assert segment.start >= 0
            assert segment.end >= segment.start

            assert segment.speaker.isalpha()
            assert segment.speaker.isupper()
            assert len(segment.speaker) == 1

        assert "A" in {segment.speaker for segment in segments}, (
            "Speaker labels must start at 'A': "
            f"{sorted({segment.speaker for segment in segments})}"
        )

    @pytest.mark.slow
    def test_streaming_diarized_json_emits_segment_events(
        self,
        openai_client: OpenAI,
        sample_audio_file: bytes,
        transcription_diarize_model: str,
    ) -> None:
        """``diarized_json`` + ``stream=true`` emits ``transcript.text.segment`` events.

        A speaker is attached only once a segment is finalized, so the segments
        are asserted as a sequence -- each one before the terminal
        ``transcript.text.done`` -- rather than against a fixed count, which is
        the recognizer's decision. Speaker labels are the sequential capital
        letters the API publishes, in first-appearance order, so the first
        speaker heard is always ``A``.

        Ref: https://developers.openai.com/api/docs/guides/speech-to-text
             openai.types.audio.transcription_text_segment_event.TranscriptionTextSegmentEvent
             stdapi/models/audio/amazon_transcribe.py:AudioModel.stt_stream
        """
        events = list(
            openai_client.audio.transcriptions.create(
                file=("test.wav", io.BytesIO(sample_audio_file)),
                model=transcription_diarize_model,
                response_format="diarized_json",
                language="en",
                stream=True,
            )
        )

        assert events, "No streaming events received"
        assert events[-1].type == "transcript.text.done", (
            f"Stream does not end with the done event: {[e.type for e in events]}"
        )
        segments = [
            event for event in events if event.type == "transcript.text.segment"
        ]
        assert segments, (
            "diarized_json streamed no speaker segments: "
            f"{[event.type for event in events]}"
        )

        speakers: list[str] = []
        starts: list[float] = []
        for segment in segments:
            assert segment.id, "Segment carries no identifier"
            assert segment.text.strip(), f"Empty segment text: {segment!r}"
            assert segment.start >= 0
            assert segment.end >= segment.start
            assert segment.speaker.isalpha(), (
                f"Speaker label is not a letter label: {segment.speaker!r}"
            )
            assert segment.speaker.isupper(), (
                f"Speaker label is not capitalised: {segment.speaker!r}"
            )
            speakers.append(segment.speaker)
            starts.append(segment.start)

        segment_ids = [segment.id for segment in segments]
        assert len(set(segment_ids)) == len(segment_ids), (
            f"Duplicate segment ids: {segment_ids}"
        )
        assert starts == sorted(starts), f"Segment starts are not ordered: {starts}"
        assert "A" in speakers, f"Speaker labels must start at 'A': {sorted(speakers)}"
        assert speakers[0] == "A", (
            f"The first speaker heard must be labelled 'A': {speakers}"
        )

    @pytest.mark.slow
    @pytest.mark.local
    def test_transcription_usage_logged(
        self,
        test_client: TestClientType,
        transcription_model: str,
        api_key: str,
        sample_audio_file: bytes,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A transcription logs one ``transcribe`` usage entry billing 15 seconds.

        Amazon Transcribe bills per second with a 15-second minimum per request, so
        the sample clip — well under 15 seconds — is always billed as 15.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/what-is.html
             stdapi/usage.py:record_transcribe_usage
        """
        capfd.readouterr()

        response = test_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(sample_audio_file), "audio/wav")},
            data={"model": transcription_model},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert response.status_code == 200

        response_data = response.json()
        assert isinstance(response_data.get("text"), str)
        assert len(response_data["text"].strip()) > 0

        transcribe_entries = logged_usage_entries(
            capfd.readouterr().out,
            service="transcribe",
            operation="/v1/audio/transcriptions",
        )
        assert transcribe_entries, "Expected transcribe service in usage"
        transcribe_entry = transcribe_entries[0]
        assert transcribe_entry["model"] == "amazon.transcribe"
        assert "input_seconds" in transcribe_entry
        # Transcribe uses 15-second minimum billing
        assert transcribe_entry["input_seconds"] == 15


@pytest.mark.gateway("JSON body input not supported by the official OpenAI API")
class TestAudioTranscriptionsJsonBody:
    """``application/json`` request bodies for POST /v1/audio/transcriptions.

    A gateway extension for MCP tools and clients that cannot build multipart
    requests: ``file`` accepts a base64 string, data URI, HTTPS URL or S3 URI. It is
    also the only path through which provider-specific extra parameters are reachable.

    Ref: https://stdapi.ai/api_openai_audio_transcriptions/
         stdapi/types/openai_audio.py:AudioTranscriptionJsonBody
    """

    def test_json_body_missing_file_returns_400(
        self, openai_client: OpenAI, transcription_model: str
    ) -> None:
        """A JSON body without ``file`` returns 400 and names the missing field.

        ``file`` is required on the JSON model (it has no multipart counterpart to
        fall back on), so Pydantic's ``missing`` error is surfaced as a body
        validation failure pointing at ``body.file``.

        Ref: stdapi/utils.py:validation_error_handler
             stdapi/main.py:handle_validation_exception
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}audio/transcriptions",
            json={"model": transcription_model},
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "file" in error["message"], (
            f"Error does not point at the missing field: {error['message']!r}"
        )

    def test_json_body_invalid_model_returns_404(self, openai_client: OpenAI) -> None:
        """JSON body path reaches model validation — invalid model returns 404.

        Uses a dummy audio data URI to satisfy input validation; model lookup
        runs before any audio decoding so the file content is irrelevant.

        Ref: stdapi/api_errors.py:UnsupportedModelError
             stdapi/routes/openai_audio_transcriptions.py:create_transcription
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}audio/transcriptions",
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
        assert "nonexistent-model-xyz" in error["message"], (
            f"Error does not name the rejected model: {error['message']!r}"
        )

    @pytest.mark.slow
    def test_json_body_transcription(
        self,
        openai_client: OpenAI,
        sample_audio_file_base64: str,
        transcription_model: str,
    ) -> None:
        """A ``data:audio/wav;base64`` file in the JSON body is transcribed.

        The response is the same ``json`` payload as the multipart path, including the
        duration-based usage block (15 seconds minimum billing).

        Ref: https://stdapi.ai/api_openai_audio_transcriptions/
             stdapi/input_file.py:InputFile
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}audio/transcriptions",
            json={"file": sample_audio_file_base64, "model": transcription_model},
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body.get("text"), str)
        assert len(body["text"].strip()) > 0
        assert any(word in body["text"].lower() for word in _SAMPLE_AUDIO_WORDS), (
            f"Transcript does not match the sample audio: {body['text']!r}"
        )
        assert body["usage"]["type"] == "duration"
        assert body["usage"]["seconds"] >= 15

    @pytest.mark.slow
    def test_json_body_transcription_with_transcribe_extra_params(
        self,
        openai_client: OpenAI,
        sample_audio_file_base64: str,
        transcription_model: str,
    ) -> None:
        """``ContentRedaction`` extra params are accepted on the JSON body path.

        Extra provider parameters are only reachable through the JSON body; they are
        validated against ``_TranscribeExtraParams`` (which pins
        ``RedactionOutput="redacted"``) and merged into ``StartTranscriptionJob``. The
        sample audio contains no PII, so only acceptance and a normal transcript are
        asserted here.

        Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_ContentRedaction.html
             stdapi/models/audio/amazon_transcribe.py:_apply_extra_settings
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}audio/transcriptions",
            json={
                "file": sample_audio_file_base64,
                "model": transcription_model,
                "ContentRedaction": {
                    "RedactionType": "PII",
                    "PiiEntityTypes": ["NAME", "SSN"],
                },
            },
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body.get("text"), str)
        assert len(body["text"].strip()) > 0


@pytest.mark.local
class TestAudioTranscriptionsResponseFormatBugs:
    """Response-format formatting checked against a fixed stubbed Transcribe result.

    ``AudioModel._transcribe`` is replaced by ``_STUB_TRANSCRIPT_DATA``, so every
    field below is fully deterministic and no AWS call is made.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/models/audio/amazon_transcribe.py:AudioModel._format_transcription_response
    """

    def test_text_format_returns_raw_plain_text(
        self, test_client: TestClientType, api_key: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``response_format=text`` returns raw ``text/plain``, not a JSON-quoted string.

        The route returns a bare ``Response`` for this format so FastAPI does not
        serialize the transcript as a JSON string.

        Ref: stdapi/models/audio/amazon_transcribe.py:AudioModel._format_transcription_response
        """
        _stub_transcribe(monkeypatch)

        response = test_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")},
            data={"model": "amazon.transcribe", "response_format": "text"},
            headers={"Authorization": f"Bearer {api_key}"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text == "hello world"

    def test_verbose_json_without_granularities_defaults_to_segments(
        self, test_client: TestClientType, api_key: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``verbose_json`` with no ``timestamp_granularities`` still populates segments.

        OpenAI defaults to segment-level timestamps when the parameter is omitted,
        while ``words`` stays unset because word timing was never requested.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/models/audio/amazon_transcribe.py:_format_json_response
        """
        _stub_transcribe(monkeypatch)

        response = test_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")},
            data={"model": "amazon.transcribe", "response_format": "verbose_json"},
            headers={"Authorization": f"Bearer {api_key}"},
        )

        assert response.status_code == 200
        body = response.json()
        assert isinstance(body.get("segments"), list)
        assert len(body["segments"]) > 0
        assert body.get("words") is None
        assert body["text"] == "hello world"
        assert body["language"] == "english"
        assert body["duration"] == 2.0
        assert body["usage"]["type"] == "duration"
        assert body["usage"]["seconds"] == 15
        assert [segment["text"] for segment in body["segments"]] == ["hello", "world"]
        assert [segment["start"] for segment in body["segments"]] == [0.0, 1.0]
        assert [segment["end"] for segment in body["segments"]] == [1.0, 2.0]


@pytest.mark.local
class TestTranscriptionCreateParamsValidation:
    """Request-level rules enforced before any Amazon Transcribe job is started.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/types/openai_audio.py:TranscriptionCreateParams._unsupported
    """

    def test_auto_chunking_strategy_is_the_default(self) -> None:
        """``chunking_strategy`` defaults to ``auto`` and is accepted explicitly."""
        assert TranscriptionCreateParams(model="m").chunking_strategy == "auto"
        assert (
            TranscriptionCreateParams(model="m", chunking_strategy="auto")
        ).chunking_strategy == "auto"

    def test_server_vad_literal_is_rejected(self) -> None:
        """The bare ``server_vad`` literal is outside the accepted field union."""
        with pytest.raises(ValidationError) as exc_info:
            TranscriptionCreateParams.model_validate(
                {"model": "m", "chunking_strategy": "server_vad"}
            )

        assert "chunking_strategy" in str(exc_info.value)

    def test_vad_config_object_is_rejected(self) -> None:
        """A full VAD config parses, then is refused as an unsupported parameter.

        Transcribe segments the audio itself, so accepting the thresholds would
        let a caller believe their voice-activity tuning was applied. The
        refusal is an ``UnsupportedParameterError``: Pydantic passes it through
        instead of wrapping it in a ``ValidationError``.
        """
        with pytest.raises(UnsupportedParameterError) as exc_info:
            TranscriptionCreateParams.model_validate(
                {
                    "model": "m",
                    "chunking_strategy": {
                        "type": "server_vad",
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 200,
                        "threshold": 0.5,
                    },
                }
            )

        assert exc_info.value.status == 400
        assert "chunking_strategy" in str(exc_info.value)

    def test_timestamp_granularities_require_verbose_json(self) -> None:
        """Granularities outside ``verbose_json`` are rejected, not silently dropped.

        Only ``verbose_json`` carries ``words``/``segments``, so any other
        response format would discard the requested timings without a signal.
        """
        with pytest.raises(ValidationError) as exc_info:
            TranscriptionCreateParams(
                model="m", response_format="json", timestamp_granularities=["word"]
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "value_error"
        assert (
            "timestamp_granularities requires response_format='verbose_json'"
            in errors[0]["msg"]
        )

    def test_timestamp_granularities_accepted_with_verbose_json(self) -> None:
        """Both granularities are kept as requested for ``verbose_json``."""
        params = TranscriptionCreateParams(
            model="m",
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

        assert params.timestamp_granularities == ["word", "segment"]

    def test_no_granularities_is_accepted_for_any_format(self) -> None:
        """An empty granularity list leaves every response format valid."""
        assert (
            TranscriptionCreateParams(
                model="m", response_format="text"
            ).timestamp_granularities
            == []
        )


@pytest.mark.local
class TestTranscriptionMultipartFormParsing:
    """Form-encoded fields the multipart path has to decode itself.

    ``timestamp_granularities`` arrives as one comma-separated form value and is
    split by the route before the request model sees it, an encoding that only
    exists on this gateway.

    Ref: https://stdapi.ai/api_openai_audio_transcriptions/
         stdapi/routes/openai_audio_transcriptions.py:create_transcription
    """

    @pytest.mark.parametrize("granularities", ["word", "word,segment"])
    def test_comma_separated_granularities_reach_the_request_model(
        self, app_client: TestClientType, granularities: str
    ) -> None:
        """The comma-separated form value is parsed and cross-checked against the format.

        The rejection can only come from the request model, so seeing it proves
        the form string was split into a non-empty granularity list.
        """
        response = app_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")},
            data={
                "model": "amazon.transcribe",
                "response_format": "json",
                "timestamp_granularities": granularities,
            },
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "timestamp_granularities" in error["message"]

    def test_omitted_granularities_are_an_empty_list(
        self, app_client: TestClientType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the field the request passes validation and reaches model resolution."""

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> None:
            raise UnsupportedModelError(model_id, status=400)

        monkeypatch.setattr(
            openai_audio_transcriptions, "validate_model", _validate_model
        )
        response = app_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")},
            data={"model": "probe-model-id", "response_format": "json"},
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"

    def test_bracket_suffixed_fields_reach_the_request_model(
        self, app_client: TestClientType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeated ``name[]`` fields — the official SDKs' actual wire format — bind.

        The official ``openai`` Python SDK posts list-valued multipart fields under
        a ``name[]`` suffix (verified on the wire for ``timestamp_granularities[]``,
        ``include[]``, ``known_speaker_names[]`` and ``known_speaker_references[]``),
        never the bare name alone. ``log_request_params`` is patched to capture the
        constructed ``TranscriptionCreateParams`` before model resolution, proving the
        values reached the request model instead of being silently dropped.
        """
        captured: dict[str, TranscriptionCreateParams] = {}

        def _capture_log_request_params(
            request: TranscriptionCreateParams, *_args: object, **_kwargs: object
        ) -> TranscriptionCreateParams:
            captured["request"] = request
            return request

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> None:
            raise UnsupportedModelError(model_id, status=400)

        monkeypatch.setattr(
            openai_audio_transcriptions,
            "log_request_params",
            _capture_log_request_params,
        )
        monkeypatch.setattr(
            openai_audio_transcriptions, "validate_model", _validate_model
        )

        response = app_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")},
            data={
                "model": "probe-model-id",
                "response_format": "verbose_json",
                "timestamp_granularities[]": ["word", "segment"],
                "include[]": ["logprobs"],
                "known_speaker_names[]": ["agent", "customer"],
                "known_speaker_references[]": ["data:audio/wav;base64,AAA"],
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"

        request = captured["request"]
        assert request.timestamp_granularities == ["word", "segment"]
        assert request.include == ["logprobs"]
        assert request.known_speaker_names == ["agent", "customer"]
        assert request.known_speaker_references == ["data:audio/wav;base64,AAA"]

    def test_languages_and_keywords_bracket_fields_reach_the_request_model(
        self, app_client: TestClientType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeated ``languages[]``/``keywords[]`` fields bind like the other lists.

        ``log_request_params`` is patched to capture the constructed request
        model before model resolution, proving the values reached it instead of
        being silently dropped.

        Ref: https://developers.openai.com/api/docs/guides/transcription
             stdapi/routes/openai_audio_transcriptions.py:create_transcription
        """
        captured: dict[str, TranscriptionCreateParams] = {}

        def _capture_log_request_params(
            request: TranscriptionCreateParams, *_args: object, **_kwargs: object
        ) -> TranscriptionCreateParams:
            captured["request"] = request
            return request

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> None:
            raise UnsupportedModelError(model_id, status=400)

        monkeypatch.setattr(
            openai_audio_transcriptions,
            "log_request_params",
            _capture_log_request_params,
        )
        monkeypatch.setattr(
            openai_audio_transcriptions, "validate_model", _validate_model
        )

        response = app_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")},
            data={
                "model": "probe-model-id",
                "response_format": "json",
                "languages[]": ["en", "fr"],
                "keywords[]": ["Amoxicillin", "EBITDA"],
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"

        request = captured["request"]
        assert request.languages == ["en", "fr"]
        assert request.keywords == ["Amoxicillin", "EBITDA"]

    def test_language_with_languages_is_rejected_by_the_route(
        self, app_client: TestClientType
    ) -> None:
        """Both language fields together return 400 before model resolution.

        Ref: https://developers.openai.com/api/docs/guides/transcription
             stdapi/types/openai_audio.py:TranscriptionCreateParams._unsupported
        """
        response = app_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")},
            data={
                "model": "amazon.transcribe",
                "language": "en",
                "languages[]": ["en", "fr"],
            },
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "cannot be combined" in error["message"]

    def test_bare_include_still_binds_as_a_list(
        self, app_client: TestClientType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bare ``include`` field keeps binding for non-SDK multipart clients.

        The bare field name is this gateway's own convenience path, not the SDK's
        ``include[]`` wire format, and binds as a single-item list.
        """
        captured: dict[str, TranscriptionCreateParams] = {}

        def _capture_log_request_params(
            request: TranscriptionCreateParams, *_args: object, **_kwargs: object
        ) -> TranscriptionCreateParams:
            captured["request"] = request
            return request

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> None:
            raise UnsupportedModelError(model_id, status=400)

        monkeypatch.setattr(
            openai_audio_transcriptions,
            "log_request_params",
            _capture_log_request_params,
        )
        monkeypatch.setattr(
            openai_audio_transcriptions, "validate_model", _validate_model
        )

        response = app_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")},
            data={
                "model": "probe-model-id",
                "response_format": "json",
                "include": "logprobs",
            },
        )

        assert response.status_code == 400
        assert captured["request"].include == ["logprobs"]


@pytest.mark.local
class TestTranscriptionAdvertisedRequestSchema:
    """The published request schema describes what a JSON caller can send.

    The schema OpenAPI publishes — and that fastapi_mcp copies into the MCP tool
    definition — is generated from the multipart form fields, while a JSON body
    is validated by ``AudioTranscriptionJsonBody``. A property that model has no
    field for, or a scalar where it requires a list, is a call an agent
    following the schema cannot get right.

    Ref: https://spec.openapis.org/oas/v3.1.0#request-body-object
         stdapi/routes/openai_audio_transcriptions.py:create_transcription
    """

    @staticmethod
    def _form_properties() -> dict[str, Any]:
        """Return the published request-body properties of the transcription route.

        Returns:
            The property schemas of the generated request-body model.
        """
        from stdapi.main import app  # noqa: PLC0415

        spec = app.openapi()
        content = spec["paths"]["/v1/audio/transcriptions"]["post"]["requestBody"][
            "content"
        ]
        ref: str = content["multipart/form-data"]["schema"]["$ref"]
        return spec["components"]["schemas"][ref.rpartition("/")[2]]["properties"]  # type: ignore[no-any-return]

    def test_no_bracket_alias_is_advertised(self) -> None:
        """The ``name[]`` aliases stay out of the schema: they are multipart-only.

        Sent as JSON, a bracketed key is not merged into its bare field; it lands
        in the request model's extras and is forwarded to the backend as a
        provider parameter, so advertising it invites a silently wrong call.
        """
        properties = self._form_properties()

        assert properties
        assert not [name for name in properties if name.endswith("[]")]

    def test_timestamp_granularities_is_advertised_as_a_list(self) -> None:
        """The advertised type matches the list the JSON body model requires."""
        granularities = self._form_properties()["timestamp_granularities"]

        assert all(
            option.get("type") in {"array", "null"}
            for option in granularities.get("anyOf", [granularities])
        )

    def test_advertised_granularity_list_is_accepted_as_json(
        self, app_client: TestClientType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A JSON body built from the advertised list type passes validation.

        Model resolution is stubbed out to fail, so reaching a ``model_not_found``
        error proves the body itself was accepted.
        """

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> None:
            raise UnsupportedModelError(model_id, status=400)

        monkeypatch.setattr(
            openai_audio_transcriptions, "validate_model", _validate_model
        )
        response = app_client.post(
            "/v1/audio/transcriptions",
            json={
                "file": "data:audio/mp3;base64,AAAA",
                "model": "probe-model-id",
                "response_format": "verbose_json",
                "timestamp_granularities": ["word", "segment"],
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"


@pytest.mark.local
class TestTranscribeUnsupportedParameters:
    """Amazon Transcribe rejects the OpenAI parameters it has no equivalent for.

    ``prompt``, ``temperature``, ``keywords`` and ``logprobs`` are accepted by
    the request model (they are valid OpenAI fields) and refused by the backend,
    so the caller gets a 400 instead of a transcript produced while ignoring them.
    Each rejection is an ``unsupported_parameter`` error naming the offending
    field, which is what lets a client tell it apart from a malformed request.

    Ref: https://stdapi.ai/api_openai_audio_transcriptions/
         stdapi/models/audio/amazon_transcribe.py:AudioModel._transcribe
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
            await AudioModel("amazon.transcribe").stt(
                self._audio(), "json", prompt="Transcribe carefully", logprobs=False
            )

        assert exc_info.value.status == 400
        assert exc_info.value.code == "unsupported_parameter"
        assert exc_info.value.param == "prompt"
        assert "prompt" in str(exc_info.value)

    async def test_temperature_is_rejected(self) -> None:
        """A non-zero ``temperature`` fails with 400: Transcribe has no sampling knob."""
        with pytest.raises(UnsupportedParameterError) as exc_info:
            await AudioModel("amazon.transcribe").stt(
                self._audio(), "json", temperature=0.5, logprobs=False
            )

        assert exc_info.value.status == 400
        assert exc_info.value.code == "unsupported_parameter"
        assert exc_info.value.param == "temperature"
        assert "temperature" in str(exc_info.value)

    async def test_logprobs_is_rejected(self) -> None:
        """``include=["logprobs"]`` fails with 400: token confidences are not available.

        The blamed parameter is the ``include`` entry the caller sent, not the
        internal flag it was parsed into.
        """
        with pytest.raises(UnsupportedParameterError) as exc_info:
            await AudioModel("amazon.transcribe").stt(
                self._audio(), "json", logprobs=True
            )

        assert exc_info.value.status == 400
        assert exc_info.value.code == "unsupported_parameter"
        assert exc_info.value.param == "include.logprobs"
        assert "logprobs" in str(exc_info.value)

    async def test_keywords_is_rejected_with_vocabulary_pointer(self) -> None:
        """``keywords`` fails with 400 pointing at pre-created custom vocabularies.

        Transcribe has no inline keyword-list equivalent, so the rejection names
        the ``VocabularyName`` extra parameter as the working alternative instead
        of silently dropping the caller's recognition hints.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/custom-vocabulary.html
             stdapi/models/audio/amazon_transcribe.py:_validate_no_keywords
        """
        with pytest.raises(ApiError) as exc_info:
            await AudioModel("amazon.transcribe").stt(
                self._audio(), "json", keywords=["Amoxicillin"], logprobs=False
            )

        assert exc_info.value.status == 400
        assert exc_info.value.code == "unsupported_parameter"
        assert exc_info.value.param == "keywords"
        assert "VocabularyName" in str(exc_info.value)

    async def test_keywords_is_rejected_when_streaming(self) -> None:
        """The streaming path refuses ``keywords`` with the same 400.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/custom-vocabulary.html
             stdapi/models/audio/amazon_transcribe.py:AudioModel.stt_stream
        """
        stream = AudioModel("amazon.transcribe").stt_stream(
            self._audio(), "json", keywords=["Amoxicillin"], logprobs=False
        )

        with pytest.raises(ApiError) as exc_info:
            await anext(stream)

        assert exc_info.value.status == 400
        assert exc_info.value.code == "unsupported_parameter"
        assert exc_info.value.param == "keywords"

    @pytest.mark.usefixtures("request_log")
    async def test_zero_temperature_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``temperature=0`` is the only accepted value, being Transcribe's behaviour.

        The response params are logged into the request context, which the
        ``request_log`` fixture provides outside a real request.
        """
        _stub_transcribe(monkeypatch)

        response = await AudioModel("amazon.transcribe").stt(
            self._audio(), "json", temperature=0.0, logprobs=False
        )

        assert not isinstance(response, str | Response)
        assert response.text == "hello world"


@pytest.mark.local
class TestTranscribeStreamTermination:
    """AudioModel.stt_stream ends with a done event carrying the whole transcript.

    Exercised on the path a request naming no language takes, where no live
    session can be opened: one delta per transcript of the finished job,
    followed by the terminating done event whose ``text`` is those deltas
    concatenated -- each delta after the first carries the space that separates
    it from the previous one, so a client appending them gets the done text.

    Ref: https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
         stdapi/models/audio/amazon_transcribe.py:AudioModel.stt_stream
    """

    async def test_done_event_concatenates_every_delta(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two transcripts produce two deltas and one done event holding both.

        Nothing here is diarized, so no delta may name a segment: a
        ``segment_id`` would point at an event the stream never sends.
        """

        async def _fake_transcribe(
            _self: AudioModel, *_args: object, **_kwargs: object
        ) -> dict[str, Any]:
            return {
                **_STUB_TRANSCRIPT_DATA,
                "transcripts": [{"transcript": "hello"}, {"transcript": "world"}],
            }

        monkeypatch.setattr(AudioModel, "_transcribe", _fake_transcribe)
        usage_token = usage.init_usage()
        try:
            events = [
                event
                async for event in AudioModel("amazon.transcribe").stt_stream(
                    InputFile("data:audio/wav;base64,AAAA"), "json", logprobs=False
                )
            ]
        finally:
            usage.USAGE.reset(usage_token)

        *deltas, done = events
        assert all(
            isinstance(event, TranscriptionTextDeltaEvent) for event in deltas
        ), events
        assert [event.delta for event in deltas] == ["hello", " world"]  # type: ignore[union-attr]
        assert isinstance(done, TranscriptionTextDoneEvent)
        assert done.type == "transcript.text.done"
        assert done.text == "hello world"
        assert "".join(event.delta for event in deltas) == done.text  # type: ignore[union-attr]
        assert [event.segment_id for event in deltas] == [None, None]  # type: ignore[union-attr]


def _stream_result(
    *,
    result_id: str,
    partial: bool,
    transcript: str,
    items: list[tuple[str, str | None, float, float]] = [],  # noqa: B006
) -> TranscriptEvent:
    """Build one ``TranscriptEvent`` the way a live session delivers it.

    Args:
        result_id: Identifier Transcribe restates a result under.
        partial: Whether the result may still change.
        transcript: The alternative's whole transcript.
        items: ``(content, speaker, start, end)`` tuples; a content of ``"."``,
            ``","`` or ``"?"`` is sent as a punctuation item.

    Returns:
        The event, shaped as the streaming SDK models it.
    """
    return TranscriptEvent(
        transcript=Transcript(
            results=[
                Result(
                    result_id=result_id,
                    is_partial=partial,
                    start_time=items[0][2] if items else 0.0,
                    end_time=items[-1][3] if items else 0.0,
                    alternatives=[
                        Alternative(
                            transcript=transcript,
                            items=[
                                Item(
                                    content=content,
                                    speaker=speaker,
                                    start_time=start,
                                    end_time=end,
                                    type=(
                                        ItemType.PUNCTUATION
                                        if content in {".", ",", "?"}
                                        else ItemType.PRONUNCIATION
                                    ),
                                )
                                for content, speaker, start, end in items
                            ],
                        )
                    ],
                )
            ]
        )
    )


@pytest.mark.local
class TestStreamedDiarization:
    """A live session's finalized results become speaker-labelled segments.

    Amazon Transcribe labels the speaker of each word rather than of a result,
    and a single finalized result routinely spans two speakers, so a result is
    cut into runs of consecutive words sharing a speaker. Punctuation carries no
    speaker of its own and stays with the run it terminates.

    Ref: https://docs.aws.amazon.com/transcribe/latest/dg/diarization.html
         stdapi/models/audio/amazon_transcribe.py:_StreamedTranscript
    """

    def test_one_result_is_cut_into_one_run_per_speaker(self) -> None:
        """Consecutive words of one speaker become one part, ending at the last word."""
        transcript = _StreamedTranscript(diarize=True)

        parts = list(
            transcript.read(
                _stream_result(
                    result_id="r1",
                    partial=False,
                    transcript="Hello there. Hi",
                    items=[
                        ("Hello", "0", 0.0, 0.5),
                        ("there", "0", 0.5, 1.0),
                        (".", None, 1.0, 1.0),
                        ("Hi", "2", 1.2, 1.6),
                    ],
                )
            )
        )

        assert [(part.speaker, part.text) for part in parts] == [
            ("0", "Hello there."),
            ("2", "Hi"),
        ]
        assert parts[0].start == 0.0
        assert parts[0].end == 1.0
        assert parts[1].start == 1.2

    def test_a_language_written_without_spaces_keeps_its_spacing(self) -> None:
        """Words are cut out of the transcript, so none of them gains a space.

        Japanese is written without word separators while Transcribe still
        attributes one word at a time, so a segment rebuilt by joining those
        words would read differently from the same recording transcribed
        without ``stream=true``.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/supported-languages.html
        """
        transcript = _StreamedTranscript(diarize=True)

        parts = list(
            transcript.read(
                _stream_result(
                    result_id="r1",
                    partial=False,
                    transcript="こんにちは元気ですか",
                    items=[
                        ("こんにちは", "0", 0.0, 0.6),
                        ("元気", "0", 0.6, 1.0),
                        ("ですか", "1", 1.0, 1.4),
                    ],
                )
            )
        )

        assert [(part.speaker, part.text) for part in parts] == [
            ("0", "こんにちは元気"),
            ("1", "ですか"),
        ]

    def test_words_the_transcript_does_not_restate_are_still_joined(self) -> None:
        """When the two disagree, the words are spaced rather than dropped.

        The transcript is the spacing reference, so a result whose words cannot
        be found in it falls back to separating them, which is what every
        language the segments are asked for in practice does.
        """
        transcript = _StreamedTranscript(diarize=True)

        parts = list(
            transcript.read(
                _stream_result(
                    result_id="r1",
                    partial=False,
                    transcript="[redacted]",
                    items=[
                        ("Hello", "0", 0.0, 0.5),
                        ("there", "0", 0.5, 1.0),
                        (".", None, 1.0, 1.0),
                    ],
                )
            )
        )

        assert [(part.speaker, part.text) for part in parts] == [("0", "Hello there.")]

    def test_a_partial_result_yields_nothing(self) -> None:
        """Nothing is emitted until Transcribe marks the result final.

        The speaker of a segment cannot be revised once sent, which holds only
        because a result is read out exactly once, when it stops changing.
        """
        transcript = _StreamedTranscript(diarize=True)

        assert not list(
            transcript.read(
                _stream_result(
                    result_id="r1",
                    partial=True,
                    transcript="Hello",
                    items=[("Hello", "0", 0.0, 0.5)],
                )
            )
        )

    def test_a_result_without_speakers_still_yields_its_text(self) -> None:
        """A finalized result Transcribe labelled nobody in stays a plain part."""
        transcript = _StreamedTranscript(diarize=True)

        parts = list(
            transcript.read(
                _stream_result(result_id="r1", partial=False, transcript="Hello there")
            )
        )

        assert [(part.speaker, part.text) for part in parts] == [(None, "Hello there")]

    def test_speaker_labels_are_off_unless_diarization_was_asked_for(self) -> None:
        """Without diarization a finalized result stays one unlabelled part."""
        transcript = _StreamedTranscript()

        parts = list(
            transcript.read(
                _stream_result(
                    result_id="r1",
                    partial=False,
                    transcript="Hello there",
                    items=[("Hello", "0", 0.0, 0.5), ("there", "2", 0.5, 1.0)],
                )
            )
        )

        assert [(part.speaker, part.text) for part in parts] == [(None, "Hello there")]

    def test_diarized_json_opens_the_session_with_speaker_labels(self) -> None:
        """``diarized_json`` is what turns speaker partitioning on, nothing else.

        Ref: stdapi/models/audio/amazon_transcribe.py:_stream_input
        """
        assert (
            _stream_input({"language_code": "en-US"}, None, diarize=True)
        ).show_speaker_label is True
        assert (
            _stream_input({"language_code": "en-US"}, None, diarize=False)
        ).show_speaker_label is False


@pytest.mark.local
class TestJobFallbackDiarization:
    """A streamed request no live session can serve still emits its segments.

    Whether a live session is available depends on the deployment and on the
    language the caller named, which the caller cannot see, so both paths answer
    ``diarized_json`` the same way: the job path emits every segment it already
    holds before the terminal done event.

    Ref: https://developers.openai.com/api/docs/guides/speech-to-text
         stdapi/models/audio/amazon_transcribe.py:AudioModel._job_transcript
    """

    @staticmethod
    async def _events(
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[list[TranscriptionStreamEvent], list[UsageLogEntry]]:
        """Stream ``diarized_json`` from a job whose speakers are already known.

        The conversation goes back to its first speaker, which is the only way
        to tell a label that is reused from one handed out again, and it runs
        past the 15-second billing minimum so the recorded seconds are the
        job's own rather than the floor.

        Returns:
            The events the stream produced, and the usage it recorded.
        """

        async def _fake_transcribe(
            _self: AudioModel, *_args: object, **_kwargs: object
        ) -> dict[str, Any]:
            return {
                "transcripts": [{"transcript": "hello world again"}],
                "audio_segments": [
                    {
                        "id": 0,
                        "start_time": "0.0",
                        "end_time": "1.0",
                        "transcript": "hello",
                        "speaker_label": "spk_1",
                    },
                    {
                        "id": 1,
                        "start_time": "1.0",
                        "end_time": "2.0",
                        "transcript": "world",
                        "speaker_label": "spk_0",
                    },
                    {
                        "id": 2,
                        "start_time": "2.0",
                        "end_time": "20.0",
                        "transcript": "again",
                        "speaker_label": "spk_1",
                    },
                ],
            }

        monkeypatch.setattr(AudioModel, "_transcribe", _fake_transcribe)
        usage_token = usage.init_usage()
        try:
            events = [
                event
                async for event in AudioModel("amazon.transcribe").stt_stream(
                    InputFile("data:audio/wav;base64,AAAA"),
                    "diarized_json",
                    logprobs=False,
                )
            ]
            return events, usage.usage_log_entries()
        finally:
            usage.USAGE.reset(usage_token)

    async def test_every_segment_is_emitted_before_the_done_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each audio segment becomes one delta and one segment event."""
        events, _ = await self._events(monkeypatch)

        assert [event.type for event in events] == [
            "transcript.text.delta",
            "transcript.text.segment",
            "transcript.text.delta",
            "transcript.text.segment",
            "transcript.text.delta",
            "transcript.text.segment",
            "transcript.text.done",
        ]
        assert events[-1].text == "hello world again"  # type: ignore[union-attr]

    async def test_the_deltas_rebuild_the_final_transcript(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concatenating the deltas gives the done text, separators included.

        Ref: openai.types.audio.transcription_text_delta_event.TranscriptionTextDeltaEvent
        """
        events, _ = await self._events(monkeypatch)

        deltas = [
            event for event in events if isinstance(event, TranscriptionTextDeltaEvent)
        ]
        done = events[-1]
        assert isinstance(done, TranscriptionTextDoneEvent)
        assert [event.delta for event in deltas] == ["hello", " world", " again"]
        assert "".join(event.delta for event in deltas) == done.text
        assert [event.segment_id for event in deltas] == ["seg_0", "seg_1", "seg_2"]

    async def test_speaker_labels_follow_first_appearance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The first speaker heard is ``A``, and hearing it again reuses ``A``."""
        events, _ = await self._events(monkeypatch)
        segments = [
            event
            for event in events
            if isinstance(event, TranscriptionTextSegmentEvent)
        ]

        assert [segment.speaker for segment in segments] == ["A", "B", "A"]
        assert [segment.id for segment in segments] == ["seg_0", "seg_1", "seg_2"]
        assert [(segment.start, segment.end) for segment in segments] == [
            (0.0, 1.0),
            (1.0, 2.0),
            (2.0, 20.0),
        ]

    async def test_the_transcribed_audio_is_billed_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A streamed diarized request records the job's seconds exactly once.

        The segments are read out of the same job payload the duration comes
        from, so emitting them must not skip -- nor repeat -- the recording.
        The 20 seconds are above the 15-second billing minimum, so a lost
        duration reads as a different number rather than as the floor, and a
        second recording doubles it (repeated recordings sum into one entry).

        Ref: stdapi/usage.py:record_transcribe_usage
        """
        _, entries = await self._events(monkeypatch)

        transcribe = [entry for entry in entries if entry["service"] == "transcribe"]
        assert len(transcribe) == 1, f"expected one transcribe entry: {entries}"
        assert transcribe[0]["model"] == "amazon.transcribe"
        assert transcribe[0]["input_seconds"] == 20


@pytest.mark.local
@pytest.mark.usefixtures("request_log")
class TestNonStreamedDiarizationRefusesMaskedText:
    """Without ``stream=true``, a masking guardrail fails ``diarized_json``.

    The segments are built from the raw job payload rather than from the
    guarded text, so answering with a masked transcript would hand back exactly
    what the guardrail took out. Only the streamed request can carry the masked
    text, because it drops the segments to do it.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
         stdapi/models/audio/amazon_transcribe.py:AudioModel._format_transcription_response
    """

    @staticmethod
    def _stub_masking_guardrail(monkeypatch: pytest.MonkeyPatch) -> None:
        """Configure a guardrail whose only intervention anonymizes a word."""

        class _GuardrailClient:
            @staticmethod
            async def apply_guardrail(**_params: object) -> dict[str, Any]:
                return {
                    "action": "GUARDRAIL_INTERVENED",
                    "assessments": [
                        {
                            "sensitiveInformationPolicy": {
                                "piiEntities": [
                                    {"type": "EMAIL", "action": "ANONYMIZED"}
                                ]
                            }
                        }
                    ],
                    "outputs": [{"text": "hello {EMAIL}"}],
                }

        monkeypatch.setattr(
            aws_bedrock, "get_client", lambda _service, _region: _GuardrailClient()
        )
        monkeypatch.setattr(
            aws_bedrock,
            "GUARDRAIL_CONFIG_VAR",
            SimpleNamespace(
                get=lambda _default=None: {
                    "guardrailIdentifier": "gr123",
                    "guardrailVersion": "1",
                }
            ),
        )

    @staticmethod
    async def _transcribe(
        monkeypatch: pytest.MonkeyPatch, response_format: AudioResponseFormat
    ) -> Any:  # noqa: ANN401
        """Transcribe the stubbed job through the masking guardrail."""
        _stub_transcribe(monkeypatch)
        TestNonStreamedDiarizationRefusesMaskedText._stub_masking_guardrail(monkeypatch)
        usage_token = usage.init_usage()
        try:
            return await AudioModel("amazon.transcribe").stt(
                InputFile("data:audio/wav;base64,AAAA"), response_format, logprobs=False
            )
        finally:
            usage.USAGE.reset(usage_token)

    async def test_diarized_json_fails_with_content_filter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The masked transcript is refused rather than returned with segments."""
        with pytest.raises(ApiError) as exc_info:
            await self._transcribe(monkeypatch, "diarized_json")

        assert exc_info.value.status == 400
        assert exc_info.value.code == "content_filter"

    async def test_json_still_answers_with_the_masked_transcript(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same guardrail masks the plain format, which carries no segments.

        Asserted beside the refusal so the difference is pinned as the format's,
        not as the guardrail declining to mask at all.
        """
        response = await self._transcribe(monkeypatch, "json")

        assert not isinstance(response, str | Response)
        assert response.text == "hello {EMAIL}"


@pytest.mark.local
class TestStreamedDiarizationIsRefusedWithoutSpeakers:
    """A model that cannot label speakers refuses ``diarized_json`` when streaming.

    Speaker labels are the whole point of the format, so a model with none to
    give answers a clean 400 instead of a 200 carrying an unlabelled transcript.

    Ref: https://developers.openai.com/api/docs/guides/speech-to-text
         stdapi/models/audio/__init__.py:AudioModelBase._validate_streamed_diarization
    """

    @staticmethod
    def _audio() -> InputFile:
        """Return a minimal upload; no model reaches it before refusing."""
        return InputFile("data:audio/wav;base64,AAAA")

    async def test_a_bedrock_speech_model_refuses(self) -> None:
        """The Converse-backed default model refuses before any backend call.

        The refusal names the model that does label speakers, so the caller can
        act on it without going back to the catalog.
        """
        stream = DefaultAudioModel("mistral.voxtral-mini-3b-2507").stt_stream(
            self._audio(), "diarized_json", logprobs=False
        )

        with pytest.raises(ApiError) as exc_info:
            await anext(stream)

        assert exc_info.value.status == 400
        assert "diarized_json" in str(exc_info.value)
        assert "amazon.transcribe" in str(exc_info.value)

    async def test_amazon_nova_sonic_refuses(self) -> None:
        """Nova Sonic serves ``json``/``text`` only and says so."""
        stream = NovaSonicAudioModel("amazon.nova-2-sonic-v1:0").stt_stream(
            self._audio(), "diarized_json", logprobs=False
        )

        with pytest.raises(ApiError) as exc_info:
            await anext(stream)

        assert exc_info.value.status == 400
        assert "diarized_json" in str(exc_info.value)


@pytest.mark.local
class TestStreamingTranscriptionRoute:
    """The route serialises the model's events as ``text/event-stream``.

    The SSE framing is what an OpenAI client parses, so it is asserted on the
    raw body rather than through the SDK: one data-only event per model event,
    each carrying its own ``type``.

    Ref: https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
         stdapi/routes/openai_audio_transcriptions.py:_transcript_audio_sse
    """

    @staticmethod
    def _stub_model(monkeypatch: pytest.MonkeyPatch) -> None:
        """Serve the route from a model streaming two deltas and a done event."""

        class _StubModel:
            @staticmethod
            async def stt_stream(
                **_kwargs: object,
            ) -> AsyncGenerator[
                TranscriptionTextDeltaEvent | TranscriptionTextDoneEvent
            ]:
                yield TranscriptionTextDeltaEvent(
                    delta="hello ", type="transcript.text.delta"
                )
                yield TranscriptionTextDeltaEvent(
                    delta="world", type="transcript.text.delta"
                )
                yield TranscriptionTextDoneEvent(
                    text="hello world", type="transcript.text.done"
                )

        async def _validate_model(*_args: object, **_kwargs: object) -> Any:  # noqa: ANN401
            return SimpleNamespace(id="stub-speech-model")

        monkeypatch.setattr(
            openai_audio_transcriptions, "validate_model", _validate_model
        )
        monkeypatch.setattr(
            openai_audio_transcriptions,
            "get_audio_model",
            lambda _model_id: _StubModel(),
        )

    @staticmethod
    def _stub_diarized_model(monkeypatch: pytest.MonkeyPatch) -> None:
        """Serve the route from a model streaming a delta, a segment and a done."""

        class _StubModel:
            @staticmethod
            async def stt_stream(
                **_kwargs: object,
            ) -> AsyncGenerator[
                TranscriptionTextDeltaEvent
                | TranscriptionTextSegmentEvent
                | TranscriptionTextDoneEvent
            ]:
                yield TranscriptionTextDeltaEvent(
                    delta="hello world",
                    type="transcript.text.delta",
                    segment_id="seg_0",
                )
                yield TranscriptionTextSegmentEvent(
                    id="seg_0",
                    start=0.0,
                    end=1.5,
                    speaker="A",
                    text="hello world",
                    type="transcript.text.segment",
                )
                yield TranscriptionTextDoneEvent(
                    text="hello world", type="transcript.text.done"
                )

        async def _validate_model(*_args: object, **_kwargs: object) -> Any:  # noqa: ANN401
            return SimpleNamespace(id="stub-speech-model")

        monkeypatch.setattr(
            openai_audio_transcriptions, "validate_model", _validate_model
        )
        monkeypatch.setattr(
            openai_audio_transcriptions,
            "get_audio_model",
            lambda _model_id: _StubModel(),
        )

    @staticmethod
    def _post_stream(app_client: TestClientType, response_format: str = "json") -> Any:  # noqa: ANN401
        """Post a streaming transcription request through the multipart path."""
        return app_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")},
            data={
                "model": "stub-speech-model",
                "response_format": response_format,
                "stream": "true",
            },
        )

    @staticmethod
    def _sse_payloads(body: str) -> list[dict[str, Any]]:
        """Return the JSON payload of every ``data:`` line in an SSE body."""
        return [
            json.loads(line.removeprefix("data:").strip())
            for line in body.splitlines()
            if line.startswith("data:") and "[DONE]" not in line
        ]

    def test_events_are_streamed_as_server_sent_events(
        self, app_client: TestClientType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each model event becomes one SSE data payload keeping its ``type``."""
        self._stub_model(monkeypatch)

        response = self._post_stream(app_client)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payloads = self._sse_payloads(response.text)
        assert [payload["type"] for payload in payloads] == [
            "transcript.text.delta",
            "transcript.text.delta",
            "transcript.text.done",
        ]
        assert [payload["delta"] for payload in payloads[:-1]] == ["hello ", "world"]
        assert payloads[-1]["text"] == "hello world"
        assert not any("segment_id" in payload for payload in payloads), (
            "segment_id belongs to a diarized stream only"
        )

    def test_a_configured_guardrail_masks_the_streamed_transcript(
        self, app_client: TestClientType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a guardrail configured, the stream carries the guarded transcript only.

        The deltas are withheld until the transcript is complete, so a masked
        transcript replaces them instead of being emitted after the raw text
        has already reached the client.
        """
        self._stub_model(monkeypatch)

        async def _mask(text: str, **_kwargs: object) -> str:
            return text.replace("world", "*****")

        monkeypatch.setattr(
            openai_audio_transcriptions, "apply_guardrail_to_text", _mask
        )
        monkeypatch.setattr(
            openai_audio_transcriptions,
            "GUARDRAIL_CONFIG_VAR",
            SimpleNamespace(get=lambda _default=None: {"guardrailIdentifier": "gr"}),
        )

        response = self._post_stream(app_client)

        assert response.status_code == 200
        payloads = self._sse_payloads(response.text)
        assert [payload["type"] for payload in payloads] == [
            "transcript.text.delta",
            "transcript.text.done",
        ]
        assert payloads[0]["delta"] == "hello *****"
        assert payloads[-1]["text"] == "hello *****"
        assert "world" not in response.text

    def test_segment_events_reach_the_client(
        self, app_client: TestClientType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A segment event is serialised with its speaker and timestamps intact.

        The delta that belongs to it names the same ``segment_id``, which is how
        a client correlates the two without matching their text.

        Ref: openai.types.audio.transcription_text_segment_event.TranscriptionTextSegmentEvent
             openai.types.audio.transcription_text_delta_event.TranscriptionTextDeltaEvent
        """
        self._stub_diarized_model(monkeypatch)

        response = self._post_stream(app_client, "diarized_json")

        assert response.status_code == 200
        payloads = self._sse_payloads(response.text)
        assert [payload["type"] for payload in payloads] == [
            "transcript.text.delta",
            "transcript.text.segment",
            "transcript.text.done",
        ]
        assert payloads[0] == {
            "delta": "hello world",
            "type": "transcript.text.delta",
            "segment_id": "seg_0",
        }
        assert payloads[1] == {
            "id": "seg_0",
            "start": 0.0,
            "end": 1.5,
            "speaker": "A",
            "text": "hello world",
            "type": "transcript.text.segment",
        }

    def test_a_speakerless_model_answers_an_error_rather_than_a_stream(
        self, app_client: TestClientType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refusing ``diarized_json`` reaches the client as HTTP 400, not as SSE.

        The refusal is raised on the stream's first event, so it only stays an
        HTTP status while that event is pulled before the response starts.

        Ref: stdapi/models/audio/__init__.py:AudioModelBase._validate_streamed_diarization
        """

        async def _validate_model(*_args: object, **_kwargs: object) -> Any:  # noqa: ANN401
            return SimpleNamespace(id="stub-speech-model")

        monkeypatch.setattr(
            openai_audio_transcriptions, "validate_model", _validate_model
        )
        monkeypatch.setattr(
            openai_audio_transcriptions,
            "get_audio_model",
            lambda _model_id: DefaultAudioModel("mistral.voxtral-mini-3b-2507"),
        )

        response = self._post_stream(app_client, "diarized_json")

        assert response.status_code == 400
        assert response.headers["content-type"].startswith("application/json")
        error = response.json()["error"]
        assert "diarized_json" in error["message"]
        assert "amazon.transcribe" in error["message"]

    def test_a_masking_guardrail_withholds_the_segments(
        self, app_client: TestClientType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A masked transcript is never followed by segments carrying the raw text.

        Segments repeat the transcript, so a stream that masked the deltas and
        kept the segments would hand the client exactly what the guardrail took
        out.
        """
        self._stub_diarized_model(monkeypatch)

        async def _mask(text: str, **_kwargs: object) -> str:
            return text.replace("world", "*****")

        monkeypatch.setattr(
            openai_audio_transcriptions, "apply_guardrail_to_text", _mask
        )
        monkeypatch.setattr(
            openai_audio_transcriptions,
            "GUARDRAIL_CONFIG_VAR",
            SimpleNamespace(get=lambda _default=None: {"guardrailIdentifier": "gr"}),
        )

        response = self._post_stream(app_client, "diarized_json")

        assert response.status_code == 200
        payloads = self._sse_payloads(response.text)
        assert [payload["type"] for payload in payloads] == [
            "transcript.text.delta",
            "transcript.text.done",
        ]
        assert "world" not in response.text


@pytest.fixture
def _request_id() -> Iterator[None]:
    """Bind a request id for code calling ``build_metadata`` outside a request."""
    from stdapi.monitoring import REQUEST_ID  # noqa: PLC0415

    token = REQUEST_ID.set("test-request-id")
    yield
    REQUEST_ID.reset(token)


@pytest.mark.local
@pytest.mark.usefixtures("_request_id", "request_log")
class TestTranscribeLanguagesMapping:
    """OpenAI ``languages`` maps onto Transcribe's multi-language identification.

    The request-level list is translated into the same
    ``IdentifyMultipleLanguages``/``LanguageOptions`` job fields the
    provider-specific extras already drive, so the standard OpenAI name
    reaches existing AWS plumbing instead of duplicating it.

    Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartTranscriptionJob.html
         https://developers.openai.com/api/docs/guides/transcription
         stdapi/models/audio/amazon_transcribe.py:_apply_language_params
    """

    def test_multiple_languages_enable_multi_language_identification(self) -> None:
        """Two expected languages become ``LanguageOptions`` locale candidates.

        Ref: stdapi/models/audio/amazon_transcribe.py:_apply_language_params
        """
        job_params = _build_transcription_job_params(
            "job", "bucket", None, "json", languages=["en", "fr"]
        )

        assert job_params["IdentifyMultipleLanguages"] is True
        assert job_params["LanguageOptions"] == ["en-US", "fr-FR"]
        assert "LanguageCode" not in job_params
        assert "IdentifyLanguage" not in job_params

    def test_single_entry_languages_behaves_like_language(self) -> None:
        """A one-entry list produces the same language fields as ``language``.

        Ref: stdapi/models/audio/amazon_transcribe.py:_apply_language_params
        """
        plural = _build_transcription_job_params(
            "job", "bucket", None, "json", languages=["en"]
        )
        singular = _build_transcription_job_params("job", "bucket", "en", "json")

        assert plural["LanguageCode"] == singular["LanguageCode"] == "en-US"
        assert "IdentifyMultipleLanguages" not in plural
        assert "LanguageOptions" not in plural

    def test_no_language_hint_keeps_auto_detection(self) -> None:
        """Without any hint the job still auto-detects a single language.

        Ref: stdapi/models/audio/amazon_transcribe.py:_apply_language_params
        """
        job_params = _build_transcription_job_params("job", "bucket", None, "json")

        assert job_params["IdentifyLanguage"] is True
        assert "IdentifyMultipleLanguages" not in job_params

    @pytest.mark.parametrize(
        "extra",
        [
            _TranscribeExtraParams(IdentifyMultipleLanguages=True),
            _TranscribeExtraParams(LanguageOptions=["en-US"]),
        ],
        ids=["IdentifyMultipleLanguages", "LanguageOptions"],
    )
    def test_languages_with_aws_language_extras_is_rejected(
        self, extra: _TranscribeExtraParams
    ) -> None:
        """``languages`` with the equivalent AWS extras is refused, not merged.

        Both spellings express the same intent; picking one would silently
        ignore the other, so the request fails naming the OpenAI parameter.

        Ref: stdapi/models/audio/amazon_transcribe.py:_apply_language_params
        """
        with pytest.raises(UnsupportedParameterError) as exc_info:
            _build_transcription_job_params(
                "job", "bucket", None, "json", extra=extra, languages=["en", "fr"]
            )

        assert exc_info.value.status == 400
        assert exc_info.value.param == "languages"


@pytest.mark.local
class TestTranscribeKeywordsRouteRejection:
    """The route surfaces the ``keywords`` rejection in the OpenAI error envelope.

    Ref: https://docs.aws.amazon.com/transcribe/latest/dg/custom-vocabulary.html
         stdapi/models/audio/amazon_transcribe.py:_validate_no_keywords
    """

    def test_keywords_returns_unsupported_parameter_envelope(
        self, test_client: TestClientType, api_key: str
    ) -> None:
        """``keywords[]`` with ``amazon.transcribe`` answers a 400 naming the fix.

        The rejection runs before any AWS call, and the envelope carries the
        ``unsupported_parameter`` code, the ``keywords`` param and the
        ``VocabularyName`` pointer a migrating client needs.

        Ref: stdapi/main.py:handle_api_error
        """
        response = test_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")},
            data={"model": "amazon.transcribe", "keywords[]": ["Amoxicillin"]},
            headers={"Authorization": f"Bearer {api_key}"},
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "unsupported_parameter"
        assert error["param"] == "keywords"
        assert "VocabularyName" in error["message"]


@pytest.mark.local
class TestTranscriptionDetectedLanguages:
    """The ``json`` response reports the detected language(s) as ``languages``.

    ``AudioModel._transcribe`` is stubbed, so the mapping from Transcribe's
    locale codes to the OpenAI response entries is fully deterministic and no
    AWS call is made.

    Ref: https://platform.openai.com/docs/api-reference/audio/createTranscription
         stdapi/models/audio/amazon_transcribe.py:_detected_languages
    """

    def test_single_detected_language_is_reported(
        self, test_client: TestClientType, api_key: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The job's ``language_code`` locale surfaces as one bare ISO entry.

        Ref: stdapi/models/audio/amazon_transcribe.py:_detected_languages
        """
        _stub_transcribe(monkeypatch)

        response = test_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")},
            data={"model": "amazon.transcribe", "response_format": "json"},
            headers={"Authorization": f"Bearer {api_key}"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["text"] == "hello world"
        assert body["languages"] == [{"code": "en"}]

    def test_multi_language_result_lists_each_language_once(
        self, test_client: TestClientType, api_key: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``language_codes`` entries map to de-duplicated bare codes in order.

        Two English locales collapse into a single ``en`` entry while the
        reported order is preserved.

        Ref: stdapi/models/audio/amazon_transcribe.py:_detected_languages
        """
        multi_language_data = {
            key: value
            for key, value in _STUB_TRANSCRIPT_DATA.items()
            if key != "language_code"
        }
        multi_language_data["language_codes"] = [
            {"language_code": "en-US", "duration_in_seconds": 5.0},
            {"language_code": "fr-FR", "duration_in_seconds": 2.0},
            {"language_code": "en-GB", "duration_in_seconds": 1.0},
        ]

        async def _fake_transcribe(
            _self: AudioModel, *_args: object, **_kwargs: object
        ) -> dict[str, Any]:
            return multi_language_data

        monkeypatch.setattr(AudioModel, "_transcribe", _fake_transcribe)

        response = test_client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(b"fake"), "audio/wav")},
            data={"model": "amazon.transcribe", "response_format": "json"},
            headers={"Authorization": f"Bearer {api_key}"},
        )

        assert response.status_code == 200
        assert response.json()["languages"] == [{"code": "en"}, {"code": "fr"}]


@pytest.mark.local
@pytest.mark.usefixtures("_request_id", "request_log")
class TestTranscribeLanguageIdSettings:
    """``LanguageIdSettings`` attaches custom resources to identified languages.

    AWS requires this parameter (rather than the flat ``VocabularyName`` /
    ``VocabularyFilterName`` / ``ModelSettings`` ones) whenever the language is
    auto-identified, which is exactly the combination a multilingual caller
    asks for.

    Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartTranscriptionJob.html
         stdapi/models/audio/amazon_transcribe.py:_apply_language_params
    """

    def test_extra_params_accept_language_id_settings(self) -> None:
        """The extra-params model parses the AWS-shaped per-language map.

        The strict model is what the route validates the JSON body against, so
        an absent field is a 400 for a documented Transcribe feature.

        Ref: stdapi/models/audio/amazon_transcribe.py:_TranscribeExtraParams
        """
        extra = _TranscribeExtraParams.model_validate(
            {
                "IdentifyMultipleLanguages": True,
                "LanguageOptions": ["en-US", "es-US"],
                "LanguageIdSettings": {
                    "en-US": {"VocabularyName": "Medical", "VocabularyFilterName": "F"},
                    "es-US": {"VocabularyName": "Medico"},
                },
            }
        )

        assert extra.LanguageIdSettings is not None
        assert extra.LanguageIdSettings["en-US"].VocabularyName == "Medical"
        assert extra.LanguageIdSettings["en-US"].VocabularyFilterName == "F"
        assert extra.LanguageIdSettings["es-US"].VocabularyName == "Medico"

    def test_unknown_sub_field_is_still_rejected(self) -> None:
        """A per-language entry only accepts the three AWS sub-parameters.

        Ref: stdapi/models/audio/amazon_transcribe.py:_TranscribeLanguageIdSetting
        """
        with pytest.raises(ValidationError):
            _TranscribeExtraParams.model_validate(
                {"LanguageIdSettings": {"en-US": {"NotARealField": "x"}}}
            )

    def test_language_id_settings_reach_a_multi_language_job(self) -> None:
        """With multi-language identification the map is sent, omitting unset keys.

        ``None`` sub-fields must not reach AWS: botocore rejects a null where a
        string is expected, which would surface as a 400 for a valid request.

        Ref: stdapi/models/audio/amazon_transcribe.py:_apply_language_params
        """
        job_params = _build_transcription_job_params(
            "job",
            "bucket",
            None,
            "json",
            extra=_TranscribeExtraParams(
                IdentifyMultipleLanguages=True,
                LanguageOptions=["en-US", "es-US"],
                LanguageIdSettings={
                    "en-US": _TranscribeLanguageIdSetting(VocabularyName="Medical")
                },
            ),
        )

        assert job_params["IdentifyMultipleLanguages"] is True
        assert job_params["LanguageIdSettings"] == {
            "en-US": {"VocabularyName": "Medical"}
        }

    def test_language_id_settings_reach_a_single_language_job(self) -> None:
        """Plain automatic identification carries the map too.

        Ref: stdapi/models/audio/amazon_transcribe.py:_apply_language_params
        """
        job_params = _build_transcription_job_params(
            "job",
            "bucket",
            None,
            "json",
            extra=_TranscribeExtraParams(
                LanguageIdSettings={
                    "en-US": _TranscribeLanguageIdSetting(LanguageModelName="MyModel")
                }
            ),
        )

        assert job_params["IdentifyLanguage"] is True
        assert job_params["LanguageIdSettings"] == {
            "en-US": {"LanguageModelName": "MyModel"}
        }

    def test_language_id_settings_with_a_fixed_language_is_rejected(self) -> None:
        """A fixed ``language`` turns identification off, so the map cannot apply.

        AWS would ignore it silently; the request is refused instead, naming
        the parameter that has no effect.

        Ref: stdapi/models/audio/amazon_transcribe.py:_apply_language_params
        """
        with pytest.raises(UnsupportedParameterError) as exc_info:
            _build_transcription_job_params(
                "job",
                "bucket",
                "en",
                "json",
                extra=_TranscribeExtraParams(
                    LanguageIdSettings={
                        "en-US": _TranscribeLanguageIdSetting(VocabularyName="Medical")
                    }
                ),
            )

        assert exc_info.value.status == 400
        assert exc_info.value.param == "LanguageIdSettings"


@pytest.mark.local
@pytest.mark.usefixtures("_request_id", "request_log")
class TestTranscribeOutputEncryption:
    """The transcription output is encrypted with the configured KMS key.

    Ref: https://docs.aws.amazon.com/transcribe/latest/dg/encryption.html
         stdapi/models/audio/amazon_transcribe.py:_build_transcription_job_params
    """

    def test_output_is_unencrypted_by_default(self) -> None:
        """Without the setting the job carries no encryption fields.

        Ref: stdapi/config.py:aws_transcribe_output_encryption_key_arn
        """
        job_params = _build_transcription_job_params("job", "bucket", "en", "json")

        assert "OutputEncryptionKMSKeyId" not in job_params
        assert "KMSEncryptionContext" not in job_params

    def test_configured_key_encrypts_the_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The configured key ARN and the request's encryption context are sent.

        The context pairs are the same non-secret request identifiers the job
        is tagged with, so a key policy can be conditioned on them.

        Ref: stdapi/config.py:aws_transcribe_output_encryption_key_arn
        """
        key_arn = (
            "arn:aws:kms:us-east-1:123456789012:key/"
            "1234abcd-12ab-34cd-56ef-1234567890ab"
        )
        monkeypatch.setattr(
            SETTINGS, "aws_transcribe_output_encryption_key_arn", key_arn
        )

        job_params = _build_transcription_job_params("job", "bucket", "en", "json")

        assert job_params["OutputEncryptionKMSKeyId"] == key_arn
        context = job_params["KMSEncryptionContext"]
        assert context["stdapi-ai.request_id"] == "test-request-id"
        assert all(key and value for key, value in context.items())

    @pytest.mark.parametrize(
        "value", ["not-an-arn", "arn:aws:kms:us-east-1:123456789012:alias/my-key"]
    )
    def test_invalid_key_arn_is_refused_at_startup(self, value: str) -> None:
        """A value that is not a KMS key ARN fails settings validation.

        It would otherwise only surface once a job is started, after the audio
        has already been uploaded.

        Ref: stdapi/config.py:_validate_transcribe_output_key_arn
        """
        with pytest.raises(ValidationError) as exc_info:
            _Settings(aws_transcribe_output_encryption_key_arn=value)

        (error,) = exc_info.value.errors()
        assert error["loc"] == ("aws_transcribe_output_encryption_key_arn",)
        assert "must be a KMS key ARN" in error["msg"]
