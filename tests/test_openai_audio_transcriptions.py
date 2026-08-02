"""Tests for the OpenAI-compatible ``/v1/audio/transcriptions`` route.

Ref: https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
     https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     stdapi/routes/openai_audio_transcriptions.py:create_transcription
"""

import io
from typing import TYPE_CHECKING, Any

import pytest
from openai import BadRequestError, NotFoundError, OpenAI
from pydantic import ValidationError
from starlette.responses import Response

from stdapi import usage
from stdapi.api_errors import ApiError, UnsupportedModelError, UnsupportedParameterError
from stdapi.input_file import InputFile
from stdapi.models.audio.amazon_transcribe import (
    AudioModel,
    _build_transcription_job_params,
    _TranscribeExtraParams,
)
from stdapi.routes import openai_audio_transcriptions
from stdapi.types.openai_audio import (
    TranscriptionCreateParams,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
)
from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from collections.abc import Iterator

    from starlette.testclient import TestClient as TestClientType

#: Stubbed AWS Transcribe job result used by response-format regression tests.
_STUB_TRANSCRIPT_DATA: dict[str, Any] = {
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
    ) -> dict[str, Any]:
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

    On this gateway the route is served by Amazon Transcribe *batch* jobs
    (S3-staged ``StartTranscriptionJob``): ``srt``/``vtt`` come from Transcribe's
    Subtitles feature and ``stream=true`` is synthesized from a completed job
    instead of being genuinely incremental.

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

        With Amazon Transcribe the stream is synthesized from a finished batch job —
        one delta per transcript then ``transcript.text.done`` — so only the presence
        and content of delta events is asserted, never incremental partial results.

        Ref: https://developers.openai.com/api/docs/guides/speech-to-text#prompting
             https://docs.aws.amazon.com/transcribe/latest/APIReference/API_streaming_StartStreamTranscription.html
             stdapi/models/audio/amazon_transcribe.py:AudioModel.stt_stream
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

        # The sample audio speaks "This is a test.": match its words tolerantly
        # rather than the exact ASR output.
        final_text = accumulated_text.strip()

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

        # The stream terminates with a single done event carrying the full
        # transcript; only assert it when the chunk cap did not truncate it.
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
        # Test segment-level timestamps only
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

        # Test word-level timestamps only
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
            # Validate segment types
            assert isinstance(segment.id, str)
            assert segment.id.startswith("seg_")
            assert isinstance(segment.start, int | float)
            assert isinstance(segment.end, int | float)
            assert segment.end > 0
            assert isinstance(segment.speaker, str)
            assert isinstance(segment.text, str)
            assert segment.type == "transcript.text.segment"

            # Validate timing
            assert segment.start >= 0
            assert segment.end >= segment.start

            # Validate speaker label format (should be A, B, C, etc.)
            assert segment.speaker.isalpha()
            assert segment.speaker.isupper()
            assert len(segment.speaker) == 1

        assert "A" in {segment.speaker for segment in segments}, (
            "Speaker labels must start at 'A': "
            f"{sorted({segment.speaker for segment in segments})}"
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

        ``include`` moved from a scalar to a list to match upstream; this checks
        the bare field name (this gateway's own convenience path, not the SDK's
        wire format) still reaches the request model as a single-item list.
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

    The batch job is finished before the first event is emitted, so the stream
    is one delta per transcript followed by the terminating done event whose
    ``text`` is those deltas joined by a space.

    Ref: https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
         stdapi/models/audio/amazon_transcribe.py:AudioModel.stt_stream
    """

    async def test_done_event_concatenates_every_delta(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two transcripts produce two deltas and one done event holding both."""

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
        assert [event.delta for event in deltas] == ["hello", "world"]  # type: ignore[union-attr]
        assert isinstance(done, TranscriptionTextDoneEvent)
        assert done.type == "transcript.text.done"
        assert done.text == "hello world"


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
