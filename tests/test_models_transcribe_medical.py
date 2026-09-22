"""Medical transcription (``amazon.transcribe-medical``): resolution, jobs and sessions.

The model runs on the standard Transcribe plumbing through its own operations:
``StartMedicalTranscriptionJob`` for a job and ``StartMedicalStreamTranscription``
for a live session. A job accepts US English and the primary care specialty only,
writes no subtitles and reports no language; a live session takes six
specialties. The recorded job outputs under ``tests/fixtures/transcribe/`` are
verbatim from real jobs run on 2026-09-22 (account ID replaced).

Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartMedicalTranscriptionJob.html
     https://docs.aws.amazon.com/transcribe/latest/APIReference/API_streaming_StartMedicalStreamTranscription.html
     stdapi/models/audio/amazon_transcribe_medical.py:AudioModel
"""

from asyncio import Event
from contextlib import asynccontextmanager
from json import loads
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from aws_sdk_transcribe_streaming.models import (
    MedicalAlternative,
    MedicalResult,
    MedicalTranscript,
    MedicalTranscriptEvent,
    MedicalTranscriptResultStreamTranscriptEvent,
    StartMedicalStreamTranscriptionInput,
)
from fastapi.exceptions import RequestValidationError

import stdapi.aws
from stdapi.api_errors import ApiError, FeatureUnavailableError
from stdapi.config import SETTINGS
from stdapi.models import (
    _GLOBAL_MODEL_REGISTRY,
    EXTRA_MODELS,
    _compute_model_capabilities,
    _find_model_class,
)
from stdapi.models.audio import amazon_transcribe, get_audio_model
from stdapi.models.audio.amazon_transcribe import (
    _STREAM_FRAME_BYTES,
    AWS_TRANSCRIBE_MEDICAL_MODEL_ID,
    AWS_TRANSCRIBE_MODEL_ID,
    _start_transcription_with_failover,
)
from stdapi.models.audio.amazon_transcribe import AudioModel as TranscribeAudioModel
from stdapi.models.audio.amazon_transcribe_medical import (
    _MEDICAL_TRANSCRIPTION_JOB,
    AudioModel,
    _build_medical_job_params,
    _TranscribeMedicalExtraParams,
)
from stdapi.models.capabilities import Capability
from stdapi.monitoring import REQUEST_ID
from stdapi.types.openai_audio import (
    Transcription,
    TranscriptionDiarized,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
    TranscriptionVerbose,
)
from tests._helpers import make_client_error, make_model_details

if TYPE_CHECKING:
    from collections.abc import Generator

    from stdapi.types import JsonMapping

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Recorded medical job outputs.
_FIXTURES = Path(__file__).parent / "fixtures" / "transcribe"

#: Transcribe's refusal of an operation the region does not offer, verbatim.
_REGION_REFUSAL = (
    "Your account isn't authorized to call this operation. Check your account "
    "permissions and try your request again."
)


def _recorded_results(name: str) -> dict[str, Any]:
    """Return the ``results`` of a recorded medical job output.

    Args:
        name: Fixture file name.

    Returns:
        The job's results, as the transcript file holds them.
    """
    return loads((_FIXTURES / name).read_text())["results"]  # type: ignore[no-any-return]


class _FakeAudioContent:
    """``InputFile`` stand-in recording whether the audio was staged at all."""

    def __init__(self) -> None:
        self.uploaded = False

    async def get_filename(self) -> str | None:
        """Return no filename."""
        return None

    async def to_s3(self, region: str, *, bucket: str, key: str) -> None:
        """Record the upload without doing anything."""
        self.uploaded = True


class _MedicalTranscribeClient:
    """Stub Transcribe client serving the medical job operations."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.started: list[dict[str, Any]] = []
        self.polled: list[str] = []
        self.deleted: list[str] = []

    async def start_medical_transcription_job(self, **params: Any) -> None:  # noqa: ANN401
        """Record the job params or raise the configured error."""
        if self._error is not None:
            raise self._error
        self.started.append(params)

    async def get_medical_transcription_job(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        """Report the job as completed, its transcript in the job's bucket."""
        name = params["MedicalTranscriptionJobName"]
        self.polled.append(name)
        return {
            "MedicalTranscriptionJob": {
                "TranscriptionJobStatus": "COMPLETED",
                "Transcript": {
                    "TranscriptFileUri": (
                        "https://s3.us-east-1.amazonaws.com/us-bucket/"
                        f"tmp/{name}/output.json"
                    )
                },
            }
        }

    async def delete_medical_transcription_job(self, **params: Any) -> None:  # noqa: ANN401
        """Record the deleted job name."""
        self.deleted.append(params["MedicalTranscriptionJobName"])


@pytest.fixture
def _request_id(request_log: dict[str, Any]) -> Generator[dict[str, Any]]:
    """Bind a request ID, which names the job, and the request log.

    Yields:
        The request log.
    """
    token = REQUEST_ID.set("job1")
    yield request_log
    REQUEST_ID.reset(token)


def _serve_jobs(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> tuple[_MedicalTranscribeClient, list[dict[str, Any]]]:
    """Run medical jobs against a stub client answering with *results*.

    Args:
        monkeypatch: Patcher applying the replacements.
        results: The job's transcript results.

    Returns:
        The stub client, and every usage recording as its keyword arguments.
    """
    client = _MedicalTranscribeClient()
    monkeypatch.setattr(
        amazon_transcribe,
        "transcribe_job_candidates",
        lambda: [("us-east-1", "us-bucket")],
    )
    monkeypatch.setattr(stdapi.aws, "get_client", lambda _service, _region=None: client)
    monkeypatch.setattr(
        amazon_transcribe, "get_client", lambda _service, _region=None: client
    )
    monkeypatch.setattr(
        amazon_transcribe, "track_temporary_s3_objects", lambda *_: None
    )
    monkeypatch.setattr(
        amazon_transcribe, "schedule_cleanup", lambda coro: coro.close()
    )

    async def _results(*_args: object) -> dict[str, Any]:
        return dict(results)

    monkeypatch.setattr(amazon_transcribe, "_get_transcription_results", _results)
    recorded: list[dict[str, Any]] = []

    def _record(duration: float, **kwargs: Any) -> int:  # noqa: ANN401
        recorded.append({"duration": duration, **kwargs})
        return 9

    monkeypatch.setattr(amazon_transcribe, "record_transcribe_usage", _record)
    return client, recorded


class TestModelResolution:
    """``amazon.transcribe-medical`` resolves to its own class everywhere.

    The standard class matches the string prefix ``amazon.transcribe``, which
    the medical ID starts with; only a longer string matcher wins, in the audio
    registry and in the catalog alike.

    Ref: stdapi/models/__init__.py:_find_model_class
         stdapi/models/__init__.py:_compute_model_capabilities
    """

    def test_the_audio_registry_resolves_each_id_to_its_class(self) -> None:
        """Neither ID captures the other.

        Ref: stdapi/models/audio/__init__.py:get_audio_model
        """
        assert type(get_audio_model(AWS_TRANSCRIBE_MEDICAL_MODEL_ID)) is AudioModel
        assert type(get_audio_model(AWS_TRANSCRIBE_MODEL_ID)) is TranscribeAudioModel

    def test_the_catalog_resolves_the_medical_class(self) -> None:
        """The catalog picks the medical class over every family sharing its prefix.

        Ref: stdapi/models/__init__.py:_find_model_class
        """
        import stdapi.main  # noqa: F401, PLC0415

        assert _find_model_class(AWS_TRANSCRIBE_MEDICAL_MODEL_ID) is AudioModel

    def test_only_the_transcription_route_is_advertised(self) -> None:
        """Transcription is listed, translation is not: the source is English.

        Ref: stdapi/models/__init__.py:_compute_model_capabilities
        """
        import stdapi.main  # noqa: F401, PLC0415

        routes, tools = _compute_model_capabilities(
            AWS_TRANSCRIBE_MEDICAL_MODEL_ID,
            make_model_details(
                AWS_TRANSCRIBE_MEDICAL_MODEL_ID,
                input_modalities=["SPEECH"],
                output_modalities=["TEXT"],
                service="AWS Transcribe",
            ),
        )

        assert routes == [f"{SETTINGS.openai_routes_prefix}/v1/audio/transcriptions"]
        assert tools == ["openai_audio_transcription"]
        assert not AudioModel.get_supported_operations() & Capability.STT_TRANSLATE

    def test_no_alias_points_at_the_medical_model(self) -> None:
        """No upstream model transcribes medical audio, so nothing aliases to it.

        The standard model's OpenAI names must keep pointing at the standard one.

        Ref: stdapi/models/__init__.py:_populate_model_aliases
        """
        import stdapi.main  # noqa: F401, PLC0415

        all_models = {
            model_id: make_model_details(model_id)
            for model_id in (AWS_TRANSCRIBE_MODEL_ID, AWS_TRANSCRIBE_MEDICAL_MODEL_ID)
        }
        aliases: dict[str, str] = {}
        for cls in _GLOBAL_MODEL_REGISTRY:
            aliases.update(cls.get_aliases(all_models))

        assert AWS_TRANSCRIBE_MEDICAL_MODEL_ID not in aliases.values()
        assert aliases["whisper-1"] == AWS_TRANSCRIBE_MODEL_ID

    async def test_translation_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The translation route refuses the model instead of running a standard job.

        Ref: stdapi/models/audio/__init__.py:AudioModelBase.stt_translate
        """
        monkeypatch.setitem(
            EXTRA_MODELS,
            AWS_TRANSCRIBE_MEDICAL_MODEL_ID,
            make_model_details(AWS_TRANSCRIBE_MEDICAL_MODEL_ID),
        )
        with pytest.raises(ApiError, match="not supported"):
            await AudioModel(AWS_TRANSCRIBE_MEDICAL_MODEL_ID).stt_translate(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "json",
                prompt=None,
            )


@pytest.mark.usefixtures("_request_id")
class TestJobParameters:
    """The job is started as AWS Transcribe Medical accepts it.

    ``Specialty`` must be ``PRIMARYCARE`` on a job and ``LanguageCode`` ``en-US``;
    ``MediaSampleRateHertz`` is never declared, since AWS refuses a declared rate
    below 16 kHz but transcribes 8 kHz audio it detects itself.

    Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartMedicalTranscriptionJob.html
         stdapi/models/audio/amazon_transcribe_medical.py:_build_medical_job_params
    """

    def test_defaults(self) -> None:
        """A request with no extra parameter is a primary care conversation.

        Ref: stdapi/models/audio/amazon_transcribe_medical.py:_build_medical_job_params
        """
        params = _build_medical_job_params("job1", "bucket", None, "json", None, None)

        assert params["MedicalTranscriptionJobName"] == "job1"
        assert params["LanguageCode"] == "en-US"
        assert params["Specialty"] == "PRIMARYCARE"
        assert params["Type"] == "CONVERSATION"
        assert params["OutputBucketName"] == "bucket"
        prefix = SETTINGS.aws_s3_tmp_prefix
        assert params["OutputKey"] == f"{prefix}job1/output.json"
        assert params["Media"]["MediaFileUri"] == f"s3://bucket/{prefix}job1/input"
        assert params["Tags"]
        assert "MediaSampleRateHertz" not in params
        assert "Settings" not in params

    def test_extra_parameters_and_diarization(self) -> None:
        """``Type`` and the settings reach the job; diarized output labels speakers.

        Ref: stdapi/models/audio/amazon_transcribe.py:_job_settings
        """
        extra = _TranscribeMedicalExtraParams(Type="DICTATION", VocabularyName="meds")

        params = _build_medical_job_params(
            "job1", "bucket", None, "diarized_json", extra, None
        )

        assert params["Type"] == "DICTATION"
        assert params["Settings"] == {
            "ShowSpeakerLabels": True,
            "MaxSpeakerLabels": 10,
            "VocabularyName": "meds",
        }

    def test_the_output_encryption_key_applies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deployment's transcript key encrypts a medical transcript too.

        Ref: stdapi/models/audio/amazon_transcribe_medical.py:_build_medical_job_params
        """
        key_arn = "arn:aws:kms:us-east-1:123456789012:key/abcd"
        monkeypatch.setattr(
            SETTINGS, "aws_transcribe_output_encryption_key_arn", key_arn
        )

        params = _build_medical_job_params("job1", "bucket", None, "json", None, None)

        assert params["OutputEncryptionKMSKeyId"] == key_arn
        assert params["KMSEncryptionContext"]

    @pytest.mark.parametrize(
        "extra",
        [
            {"ContentRedaction": {"RedactionType": "PII"}},
            {"ContentIdentificationType": "PHI"},
            {"IdentifyMultipleLanguages": True},
            {"Type": "MONOLOGUE"},
            {"Specialty": "DERMATOLOGY"},
        ],
    )
    def test_a_parameter_the_model_lacks_is_refused(
        self, extra: dict[str, Any]
    ) -> None:
        """A standard-only, unknown or out-of-range parameter is a validation error.

        Content identification is not exposed: the OpenAI response has nowhere to
        carry the entities it finds.

        Ref: stdapi/models/audio/amazon_transcribe_medical.py:_TranscribeMedicalExtraParams
        """
        with pytest.raises(RequestValidationError):
            AudioModel._live_request(None, None, extra, diarize=False)  # noqa: SLF001


@pytest.mark.usefixtures("_request_id")
class TestRequestValidation:
    """What a medical job cannot serve is refused before any audio is staged.

    Ref: stdapi/models/audio/amazon_transcribe_medical.py:AudioModel._transcribe
    """

    @pytest.mark.parametrize(
        ("language", "languages", "param"),
        [
            ("fr", None, "language"),
            ("en-GB", None, "language"),
            (None, ["es"], "languages"),
        ],
    )
    async def test_a_language_other_than_us_english_is_refused(
        self, language: str | None, languages: list[str] | None, param: str
    ) -> None:
        """AWS transcribes medical audio in US English only.

        Its own refusal ("Medical transcription isn't supported in this
        language") carries no parameter name, so the model answers first.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/transcribe-medical.html
        """
        audio = _FakeAudioContent()

        with pytest.raises(ApiError) as raised:
            await AudioModel(AWS_TRANSCRIBE_MEDICAL_MODEL_ID).stt(
                audio,  # type: ignore[arg-type]
                "json",
                language=language,
                languages=languages,
                logprobs=False,
            )

        assert raised.value.code == "invalid_language_format"
        assert raised.value.param == param
        assert not audio.uploaded

    @pytest.mark.parametrize("response_format", ["srt", "vtt"])
    async def test_subtitles_are_refused_with_the_alternative(
        self, response_format: str
    ) -> None:
        """No subtitle file exists for a medical job: timed segments are the way forward.

        Ref: stdapi/models/audio/amazon_transcribe_medical.py:AudioModel._validate_response_formats
        """
        audio = _FakeAudioContent()

        with pytest.raises(ApiError, match="verbose_json"):
            await AudioModel(AWS_TRANSCRIBE_MEDICAL_MODEL_ID).stt(
                audio,  # type: ignore[arg-type]
                response_format,  # type: ignore[arg-type]
                logprobs=False,
            )

        assert not audio.uploaded

    async def test_a_specialty_is_refused_on_a_job(self) -> None:
        """A job takes primary care only; the other specialties need a live session.

        Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartMedicalTranscriptionJob.html
        """
        audio = _FakeAudioContent()

        with pytest.raises(ApiError, match="stream=true") as raised:
            await AudioModel(AWS_TRANSCRIBE_MEDICAL_MODEL_ID).stt(
                audio,  # type: ignore[arg-type]
                "json",
                extra_params={"Specialty": "CARDIOLOGY"},
                logprobs=False,
            )

        assert raised.value.status == 400
        assert raised.value.param == "Specialty"
        assert not audio.uploaded


@pytest.mark.usefixtures("_request_id")
class TestJobTranscription:
    """A finished medical job answers every supported format, in English.

    A medical transcript names no language, so the model sets US English on it.

    Ref: stdapi/models/audio/amazon_transcribe_medical.py:AudioModel._transcribe
    """

    async def test_the_job_runs_through_the_medical_operations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Start, poll, delete and usage all go to the medical product.

        Ref: stdapi/models/audio/amazon_transcribe.py:AudioModel._transcribe
        """
        client, recorded = _serve_jobs(
            monkeypatch, _recorded_results("medical_dictation_phi.json")
        )
        cleanups: list[Any] = []
        monkeypatch.setattr(amazon_transcribe, "schedule_cleanup", cleanups.append)

        response = await AudioModel(AWS_TRANSCRIBE_MEDICAL_MODEL_ID).stt(
            _FakeAudioContent(),  # type: ignore[arg-type]
            "json",
            language="en",
            extra_params={"Type": "DICTATION"},
            logprobs=False,
        )

        assert isinstance(response, Transcription)
        assert "metformin" in response.text
        assert response.languages is not None
        assert [language.code for language in response.languages] == ["en"]
        (params,) = client.started
        assert params["Type"] == "DICTATION"
        assert client.polled == ["job1"]
        (cleanup,) = cleanups
        await cleanup
        assert client.deleted == ["job1"]
        assert recorded == [
            {
                "duration": 8.63,
                "region": "us-east-1",
                "model": AWS_TRANSCRIBE_MEDICAL_MODEL_ID,
            }
        ]

    async def test_verbose_json_reports_english_and_timed_segments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Segments and words come from the recorded output, the language is set.

        Ref: stdapi/models/audio/amazon_transcribe.py:_format_json_response
        """
        _serve_jobs(monkeypatch, _recorded_results("medical_dictation_phi.json"))

        response = await AudioModel(AWS_TRANSCRIBE_MEDICAL_MODEL_ID).stt(
            _FakeAudioContent(),  # type: ignore[arg-type]
            "verbose_json",
            timestamp_granularities=["word", "segment"],
            logprobs=False,
        )

        assert isinstance(response, TranscriptionVerbose)
        assert response.language == "english"
        assert response.segments is not None
        assert len(response.segments) == 1
        assert response.words is not None
        assert response.words[0].word == "Patient"

    async def test_diarized_json_labels_the_speakers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A diarized request gets speaker segments out of the recorded conversation.

        The recorded two-voice clip came back as one speaker, so only the shape
        is asserted, never the number of speakers.

        Ref: stdapi/models/audio/amazon_transcribe.py:_format_diarized_json_response
        """
        client, _ = _serve_jobs(
            monkeypatch, _recorded_results("medical_conversation_speakers.json")
        )

        response = await AudioModel(AWS_TRANSCRIBE_MEDICAL_MODEL_ID).stt(
            _FakeAudioContent(),  # type: ignore[arg-type]
            "diarized_json",
            logprobs=False,
        )

        assert isinstance(response, TranscriptionDiarized)
        assert response.segments
        assert response.segments[0].speaker == "A"
        assert "chest pain" in response.text
        assert client.started[0]["Settings"]["ShowSpeakerLabels"] is True


@pytest.mark.usefixtures("_request_id")
class TestRegionRefusal:
    """A region without medical transcription hands the job to the next one.

    ``StartMedicalTranscriptionJob`` outside its regions answers a
    ``BadRequestException`` saying the account "isn't authorized", verbatim
    below; refused everywhere, the feature is the deployment's gap.

    Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartMedicalTranscriptionJob.html
         stdapi/models/audio/amazon_transcribe.py:_start_transcription_with_failover
    """

    @staticmethod
    def _patch(
        monkeypatch: pytest.MonkeyPatch, clients: dict[str, _MedicalTranscribeClient]
    ) -> None:
        """Serve each region from its stub and make the S3 copy a no-op."""
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, region=None: clients[region]
        )

        async def _copy(*_args: object, **_kwargs: object) -> None:
            return None

        monkeypatch.setattr(amazon_transcribe, "copy_s3_object", _copy)

    @staticmethod
    def _refusal() -> Exception:
        """Return the verbatim refusal of a region without medical transcription."""
        return make_client_error(
            "BadRequestException",
            "StartMedicalTranscriptionJob",
            message=_REGION_REFUSAL,
            status=400,
        )

    async def test_the_next_region_starts_the_job(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """eu-central-2 refuses, us-east-1 starts the job.

        Ref: stdapi/aws.py:is_region_unavailable_error
        """
        clients = {
            "eu-central-2": _MedicalTranscribeClient(self._refusal()),
            "us-east-1": _MedicalTranscribeClient(),
        }
        self._patch(monkeypatch, clients)

        region, bucket = await _start_transcription_with_failover(
            [("eu-central-2", "zurich-bucket"), ("us-east-1", "us-bucket")],
            "job1",
            None,
            "json",
            operations=_MEDICAL_TRANSCRIPTION_JOB,
        )

        assert (region, bucket) == ("us-east-1", "us-bucket")
        assert clients["us-east-1"].started[0]["OutputBucketName"] == "us-bucket"

    async def test_refused_everywhere_is_a_feature_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """The caller reads the generic 503; the operator reads the action refused.

        Ref: stdapi/api_errors.py:FeatureUnavailableError
        """
        self._patch(
            monkeypatch, {"eu-central-2": _MedicalTranscribeClient(self._refusal())}
        )

        with pytest.raises(FeatureUnavailableError) as raised:
            await _start_transcription_with_failover(
                [("eu-central-2", "zurich-bucket")],
                "job1",
                None,
                "json",
                operations=_MEDICAL_TRANSCRIPTION_JOB,
            )

        assert raised.value.status == 503
        assert "Medical transcription is not available" in str(raised.value)
        assert "StartMedicalTranscriptionJob" not in str(raised.value)
        assert any(
            "transcribe:StartMedicalTranscriptionJob" in str(detail)
            for detail in request_log["error_detail"]
        )


def _medical_events(*texts: str) -> list[MedicalTranscriptResultStreamTranscriptEvent]:
    """Build the finalized results a live medical session delivers.

    Args:
        *texts: One finalized result per text.

    Returns:
        The events, in order.
    """
    return [
        MedicalTranscriptResultStreamTranscriptEvent(
            MedicalTranscriptEvent(
                transcript=MedicalTranscript(
                    results=[
                        MedicalResult(
                            result_id=f"r{index}",
                            is_partial=False,
                            alternatives=[MedicalAlternative(transcript=text)],
                        )
                    ]
                )
            )
        )
        for index, text in enumerate(texts)
    ]


class TestLiveSession:
    """A streamed medical transcription opens a live medical session.

    The session takes the six specialties, which is what a job cannot; a
    parameter only a job honours sends the request to a job instead.

    Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_streaming_StartMedicalStreamTranscription.html
         stdapi/models/audio/amazon_transcribe_medical.py:AudioModel._live_request
    """

    def test_the_session_request(self) -> None:
        """Specialty and type reach the session, the language is US English.

        Ref: stdapi/models/audio/amazon_transcribe_medical.py:AudioModel._live_request
        """
        extra: JsonMapping = {"Specialty": "CARDIOLOGY", "Type": "DICTATION"}

        request = AudioModel._live_request("en", None, extra, diarize=True)  # noqa: SLF001

        assert isinstance(request, StartMedicalStreamTranscriptionInput)
        assert request.language_code == "en-US"
        assert request.specialty == "CARDIOLOGY"
        assert request.type == "DICTATION"
        assert request.show_speaker_label is True
        assert request.media_sample_rate_hertz == 16000

    def test_a_job_only_parameter_needs_a_job(self) -> None:
        """Alternatives exist on a job only, so no session is opened for them.

        Ref: stdapi/models/audio/amazon_transcribe_medical.py:AudioModel._live_request
        """
        extra: JsonMapping = {"ShowAlternatives": True, "MaxAlternatives": 2}

        assert AudioModel._live_request(None, None, extra, diarize=False) is None  # noqa: SLF001

    async def test_the_stream_is_served_and_billed_as_medical_streaming(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The medical operation is opened and its seconds billed at the streaming rate.

        Ref: stdapi/models/audio/amazon_transcribe.py:AudioModel._live_transcript
        """
        opened: list[object] = []
        closed = Event()

        class _Client:
            @staticmethod
            def start_medical_stream_transcription(request: object) -> None:
                opened.append(request)

        class _Session:
            region = "us-east-1"

            async def send(self, _event: object) -> None:
                return None

            async def close_input(self) -> None:
                closed.set()

            async def __aiter__(self) -> Any:  # noqa: ANN401
                await closed.wait()
                for event in _medical_events(
                    "Echocardiogram shows", "mild regurgitation."
                ):
                    yield event

        @asynccontextmanager
        async def _open(*args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
            args[2](_Client(), "us-east-1")
            yield _Session()

        async def _frames(*_args: object, **_kwargs: object) -> Any:  # noqa: ANN401
            yield bytes(_STREAM_FRAME_BYTES)

        recorded: list[dict[str, Any]] = []

        def _record(duration: float, **kwargs: Any) -> int:  # noqa: ANN401
            recorded.append({"duration": duration, **kwargs})
            return 1

        monkeypatch.setattr(amazon_transcribe, "open_bidi_stream", _open)
        monkeypatch.setattr(amazon_transcribe, "_stream_audio_frames", _frames)
        monkeypatch.setattr(
            amazon_transcribe, "transcribe_stream_regions", lambda: ["us-east-1"]
        )
        monkeypatch.setattr(amazon_transcribe, "record_transcribe_usage", _record)

        events = [
            event
            async for event in AudioModel(AWS_TRANSCRIBE_MEDICAL_MODEL_ID).stt_stream(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "json",
                extra_params={"Specialty": "CARDIOLOGY"},
                logprobs=False,
            )
        ]

        (request,) = opened
        assert isinstance(request, StartMedicalStreamTranscriptionInput)
        assert request.specialty == "CARDIOLOGY"
        deltas = [
            event for event in events if isinstance(event, TranscriptionTextDeltaEvent)
        ]
        assert [event.delta for event in deltas] == [
            "Echocardiogram shows",
            " mild regurgitation.",
        ]
        assert isinstance(events[-1], TranscriptionTextDoneEvent)
        assert recorded == [
            {
                "duration": pytest.approx(_STREAM_FRAME_BYTES / 2 / 16000),
                "region": "us-east-1",
                "streaming": True,
                "model": AWS_TRANSCRIBE_MEDICAL_MODEL_ID,
            }
        ]

    async def test_a_language_other_than_us_english_is_refused(self) -> None:
        """A stream refuses the language before any session is opened.

        Ref: stdapi/models/audio/amazon_transcribe_medical.py:_validate_language
        """
        with pytest.raises(ApiError) as raised:
            async for _ in AudioModel(AWS_TRANSCRIBE_MEDICAL_MODEL_ID).stt_stream(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "json",
                language="de",
                logprobs=False,
            ):
                pass

        assert raised.value.code == "invalid_language_format"

    @pytest.mark.usefixtures("_request_id")
    async def test_a_stream_a_job_serves_is_billed_as_a_medical_job(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job-only parameter streams from a medical job, billed at the batch rate.

        Ref: stdapi/models/audio/amazon_transcribe.py:AudioModel._job_transcript
        """
        client, recorded = _serve_jobs(
            monkeypatch, _recorded_results("medical_dictation_phi.json")
        )
        monkeypatch.setattr(
            amazon_transcribe, "transcribe_stream_regions", lambda: ["us-east-1"]
        )

        events = [
            event
            async for event in AudioModel(AWS_TRANSCRIBE_MEDICAL_MODEL_ID).stt_stream(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "json",
                extra_params={"ShowAlternatives": True, "MaxAlternatives": 2},
                logprobs=False,
            )
        ]

        assert isinstance(events[-1], TranscriptionTextDoneEvent)
        assert "metformin" in events[-1].text
        assert client.started[0]["Settings"]["ShowAlternatives"] is True
        assert recorded == [
            {
                "duration": 8.63,
                "region": "us-east-1",
                "model": AWS_TRANSCRIBE_MEDICAL_MODEL_ID,
            }
        ]


@asynccontextmanager
async def _refused_everywhere(*_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
    """Stand in for a live session every candidate region refuses to open."""
    feature, detail = "Live transcription", "no region offers it"
    raise FeatureUnavailableError(feature, detail)
    yield  # pragma: no cover - the context never opens


@pytest.mark.usefixtures("_request_id")
class TestLiveSessionUnavailable:
    """A stream no region can open live falls back to a job when a job can serve it.

    A region without medical streaming refuses the session before any audio is
    sent, so a primary care request is still served whole by a medical job; a
    specialty only a live session takes is the deployment's gap, a 503, never a
    400 telling a caller who streamed to stream.

    Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_streaming_StartMedicalStreamTranscription.html
         stdapi/models/audio/amazon_transcribe.py:_live_or_job
    """

    async def test_primary_care_falls_back_to_a_medical_job(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refused in every region, the stream is served by a job, billed as batch.

        Ref: stdapi/models/audio/amazon_transcribe.py:AudioModel.stt_stream
        """
        client, recorded = _serve_jobs(
            monkeypatch, _recorded_results("medical_dictation_phi.json")
        )
        monkeypatch.setattr(amazon_transcribe, "open_bidi_stream", _refused_everywhere)
        monkeypatch.setattr(
            amazon_transcribe, "transcribe_stream_regions", lambda: ["eu-west-3"]
        )

        events = [
            event
            async for event in AudioModel(AWS_TRANSCRIBE_MEDICAL_MODEL_ID).stt_stream(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "json",
                logprobs=False,
            )
        ]

        assert isinstance(events[-1], TranscriptionTextDoneEvent)
        assert "metformin" in events[-1].text
        assert len(client.started) == 1
        assert "streaming" not in recorded[0]

    @pytest.mark.parametrize("regions", [["eu-west-3"], []])
    async def test_a_specialty_no_region_streams_is_the_deployment(
        self, monkeypatch: pytest.MonkeyPatch, regions: list[str]
    ) -> None:
        """Refused everywhere, or with no live region at all, a specialty is a 503.

        Ref: stdapi/models/audio/amazon_transcribe.py:AudioModel.stt_stream
        """
        client, _ = _serve_jobs(
            monkeypatch, _recorded_results("medical_dictation_phi.json")
        )
        monkeypatch.setattr(amazon_transcribe, "open_bidi_stream", _refused_everywhere)
        monkeypatch.setattr(
            amazon_transcribe, "transcribe_stream_regions", lambda: regions
        )
        audio = _FakeAudioContent()

        with pytest.raises(FeatureUnavailableError) as raised:
            async for _ in AudioModel(AWS_TRANSCRIBE_MEDICAL_MODEL_ID).stt_stream(
                audio,  # type: ignore[arg-type]
                "json",
                extra_params={"Specialty": "NEUROLOGY"},
                logprobs=False,
            ):
                pass

        assert raised.value.status == 503
        assert "stream=true" not in str(raised.value)
        assert not client.started
        assert not audio.uploaded

    async def test_a_standard_stream_falls_back_to_a_job_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fallback is the standard model's as well: a job serves any request.

        Ref: stdapi/models/audio/amazon_transcribe.py:AudioModel._job_serves
        """
        monkeypatch.setattr(amazon_transcribe, "open_bidi_stream", _refused_everywhere)
        monkeypatch.setattr(
            amazon_transcribe, "transcribe_stream_regions", lambda: ["us-east-1"]
        )

        async def _job(*_args: object, **_kwargs: object) -> dict[str, Any]:
            return {
                "transcripts": [{"transcript": "from the job"}],
                "audio_segments": [
                    {
                        "id": 0,
                        "start_time": "0.0",
                        "end_time": "1.0",
                        "transcript": "from the job",
                    }
                ],
                "items": [],
                "language_code": "en-US",
            }

        monkeypatch.setattr(TranscribeAudioModel, "_transcribe", _job)
        monkeypatch.setattr(
            amazon_transcribe, "record_transcribe_usage", lambda *_a, **_k: 1
        )

        events = [
            event
            async for event in TranscribeAudioModel(AWS_TRANSCRIBE_MODEL_ID).stt_stream(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "text",
                language="en",
                logprobs=False,
            )
        ]

        assert isinstance(events[-1], TranscriptionTextDoneEvent)
        assert events[-1].text == "from the job"
