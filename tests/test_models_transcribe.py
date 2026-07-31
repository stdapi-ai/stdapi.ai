"""Unit tests for AWS Transcribe: audio duration, region candidates, failover.

Transcription runs as a batch ``StartTranscriptionJob`` staged through S3, so a job is
pinned to a (region, bucket) pair and every candidate region needs a co-located bucket.
Several verbose_json fields have no AWS equivalent and are synthesized here, so tests
covering them assert gateway behavior rather than an AWS contract.

Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartTranscriptionJob.html
     stdapi/models/audio/amazon_transcribe.py:AudioModel
"""

from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError, ParamValidationError
from pydantic import ValidationError

import stdapi.aws
from stdapi.api_errors import (
    ApiError,
    InvalidLanguageFormatError,
    UnsupportedParameterError,
)
from stdapi.aws_s3 import _bucket_to_region
from stdapi.config import SETTINGS
from stdapi.models import EXTRA_MODELS
from stdapi.models.audio import amazon_transcribe
from stdapi.models.audio.amazon_transcribe import (
    AWS_TRANSCRIBE_MODEL_ID,
    AudioModel,
    _build_transcription_job_params,
    _build_transcription_segment,
    _get_audio_duration,
    _start_transcription_with_failover,
    _text_compression_ratio,
    _TranscribeContentRedaction,
    _TranscribeExtraParams,
    _TranscribeModelSettings,
    _TranscribeToxicityDetectionSetting,
    initialize_transcribe_models,
    transcribe_job_candidates,
)
from stdapi.monitoring import REQUEST_ID, REQUEST_LOG
from stdapi.types.openai_audio import (
    TranscriptionVerbose,
    TranslationVerbose,
    UsageDuration,
)
from tests._helpers import make_client_error

if TYPE_CHECKING:
    from collections.abc import Generator

    from types_aiobotocore_bedrock.literals import RegionName


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


class TestGetAudioDuration:
    """_get_audio_duration: the last segment's end time is the billed duration.

    Amazon Transcribe reports no media duration on the job, so it is recovered from
    the transcript's audio segments; a missing duration still bills the 15-second
    per-request minimum, which is why it warns instead of failing.

    Ref: https://docs.aws.amazon.com/transcribe/latest/dg/what-is.html
         stdapi/models/audio/amazon_transcribe.py:_get_audio_duration
    """

    def test_returns_last_segment_end_time(self) -> None:
        """The duration is the end time of the final audio segment."""
        data: dict[str, Any] = {
            "audio_segments": [{"end_time": "1.5"}, {"end_time": "42.75"}]
        }
        assert _get_audio_duration(data) == 42.75  # type: ignore[arg-type]

    def test_missing_segments_warns_and_returns_zero(
        self, request_log: dict[str, Any]
    ) -> None:
        """No segments: return 0.0 (15s minimum billed) and warn in the request log."""
        assert _get_audio_duration({}) == 0.0
        assert request_log["level"] == "warning"
        assert any("15-second minimum" in str(d) for d in request_log["error_detail"])

    def test_empty_segments_list_warns_and_returns_zero(
        self, request_log: dict[str, Any]
    ) -> None:
        """An empty segment list must behave like a missing one."""
        data: dict[str, Any] = {"audio_segments": []}
        assert _get_audio_duration(data) == 0.0  # type: ignore[arg-type]
        assert request_log["level"] == "warning"
        assert any("15-second minimum" in str(d) for d in request_log["error_detail"])


class TestTranscribeJobCandidates:
    """transcribe_job_candidates: per-region bucket pairing rules.

    Transcribe reads the media from S3 and writes its output there, and the bucket
    must live in the job's region, so a region without a usable bucket can never be
    a candidate.

    Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartTranscriptionJob.html
         stdapi/models/audio/amazon_transcribe.py:transcribe_job_candidates
    """

    def test_explicit_region_uses_the_transcribe_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A configured region pairs with the dedicated Transcribe bucket."""
        monkeypatch.setattr(SETTINGS, "aws_transcribe_region", "us-west-2")
        monkeypatch.setattr(SETTINGS, "aws_transcribe_s3_bucket", "transcribe-bucket")
        assert transcribe_job_candidates() == [("us-west-2", "transcribe-bucket")]

    def test_explicit_region_without_any_bucket_yields_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A configured region with no usable bucket disables transcription."""
        monkeypatch.setattr(SETTINGS, "aws_transcribe_region", "us-west-2")
        monkeypatch.setattr(SETTINGS, "aws_transcribe_s3_bucket", None)
        monkeypatch.setattr(SETTINGS, "aws_s3_regional_buckets", {})
        assert transcribe_job_candidates() == []

    def test_default_pairs_bedrock_regions_with_their_buckets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unset region: primary uses the Transcribe bucket, others regional ones."""
        monkeypatch.setattr(SETTINGS, "aws_transcribe_region", None)
        monkeypatch.setattr(SETTINGS, "aws_transcribe_s3_bucket", "primary-bucket")
        monkeypatch.setattr(
            SETTINGS,
            "aws_bedrock_regions",
            ["us-east-1", "eu-west-1", "ap-southeast-2"],
        )
        monkeypatch.setattr(
            SETTINGS, "aws_s3_regional_buckets", {"eu-west-1": "eu-bucket"}
        )
        # ap-southeast-2 has no bucket: it is not a candidate.
        assert transcribe_job_candidates() == [
            ("us-east-1", "primary-bucket"),
            ("eu-west-1", "eu-bucket"),
        ]

    def test_primary_region_falls_back_to_its_regional_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a dedicated bucket, the primary region uses its regional one."""
        monkeypatch.setattr(SETTINGS, "aws_transcribe_region", None)
        monkeypatch.setattr(SETTINGS, "aws_transcribe_s3_bucket", None)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1"])
        monkeypatch.setattr(
            SETTINGS, "aws_s3_regional_buckets", {"us-east-1": "us-bucket"}
        )
        assert transcribe_job_candidates() == [("us-east-1", "us-bucket")]


class _StubTranscribeClient:
    """Stub Transcribe client with a fixed per-region start outcome."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.started: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.polled: list[str] = []

    async def start_transcription_job(self, **params: Any) -> None:  # noqa: ANN401
        """Record the job params or raise the configured error."""
        if self._error is not None:
            raise self._error
        self.started.append(params)

    async def delete_transcription_job(self, **params: Any) -> None:  # noqa: ANN401
        """Record the deleted job name."""
        self.deleted.append(params["TranscriptionJobName"])

    async def get_transcription_job(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        """Record the poll and report the job as completed."""
        job_name = params["TranscriptionJobName"]
        self.polled.append(job_name)
        return {
            "TranscriptionJob": {
                "TranscriptionJobStatus": "COMPLETED",
                "Transcript": {
                    "TranscriptFileUri": (
                        f"https://s3.eu-west-1.amazonaws.com/eu-bucket/{job_name}/output.json"
                    )
                },
            }
        }


class TestStartTranscriptionWithFailover:
    """_start_transcription_with_failover: whole-job region failover.

    The media is uploaded once, to the first candidate's bucket; a later candidate
    needs a server-side copy into its own region first. Caller errors must abort the
    loop, and a region that may have accepted the job despite erroring gets a
    best-effort deletion so it stops billing.

    Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartTranscriptionJob.html
         stdapi/aws.py:call_with_region_failover
    """

    @pytest.fixture(autouse=True)
    def _request_context(self, request_log: dict[str, Any]) -> Generator[None]:
        """Provide the request ID the job tags are built from."""
        id_token = REQUEST_ID.set("job1")
        yield
        REQUEST_ID.reset(id_token)

    @staticmethod
    def _patch_infra(
        monkeypatch: pytest.MonkeyPatch, clients: dict[str, _StubTranscribeClient]
    ) -> list[tuple[str, str, str]]:
        """Stub regional clients and S3 copy; return the copy call log."""
        monkeypatch.setattr(
            stdapi.aws, "get_client", lambda _service, region=None: clients[region]
        )
        copies: list[tuple[str, str, str]] = []

        async def _fake_copy(
            source_bucket: str,
            source_key: str,
            *,
            dest_bucket: str,
            dest_key: str,
            dest_region: RegionName,
            temporary: bool,
        ) -> None:
            assert temporary is True
            assert dest_key == source_key
            copies.append((source_bucket, dest_bucket, dest_region))

        monkeypatch.setattr(amazon_transcribe, "copy_s3_object", _fake_copy)
        return copies

    async def test_first_region_success_starts_without_copy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A healthy first region starts the job with no S3 copy."""
        clients = {
            "us-east-1": _StubTranscribeClient(),
            "eu-west-1": _StubTranscribeClient(),
        }
        copies = self._patch_infra(monkeypatch, clients)

        region, bucket = await _start_transcription_with_failover(
            [("us-east-1", "us-bucket"), ("eu-west-1", "eu-bucket")],
            "job1",
            None,
            "json",
        )

        assert (region, bucket) == ("us-east-1", "us-bucket")
        assert copies == []
        (params,) = clients["us-east-1"].started
        assert params["OutputBucketName"] == "us-bucket"

    async def test_region_error_copies_input_and_fails_over(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A throttled first region copies the audio and starts in the second."""
        clients = {
            "us-east-1": _StubTranscribeClient(
                make_client_error("ThrottlingException", "StartTranscriptionJob")
            ),
            "eu-west-1": _StubTranscribeClient(),
        }
        copies = self._patch_infra(monkeypatch, clients)

        region, bucket = await _start_transcription_with_failover(
            [("us-east-1", "us-bucket"), ("eu-west-1", "eu-bucket")],
            "job1",
            None,
            "json",
        )

        assert (region, bucket) == ("eu-west-1", "eu-bucket")
        assert copies == [("us-bucket", "eu-bucket", "eu-west-1")]
        (params,) = clients["eu-west-1"].started
        assert params["Media"]["MediaFileUri"].startswith("s3://eu-bucket/")
        log = REQUEST_LOG.get()
        assert log["level"] == "warning"
        assert any(
            "transcribe" in str(detail) and "us-east-1" in str(detail)
            for detail in log["error_detail"]
        )

    async def test_caller_error_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bad language code raises immediately without trying region 2.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/supported-languages.html
             stdapi/models/audio/amazon_transcribe.py:_handle_transcription_error
        """
        clients = {
            "us-east-1": _StubTranscribeClient(
                make_client_error(
                    "BadRequestException",
                    "StartTranscriptionJob",
                    message="unsupported languageCode",
                )
            ),
            "eu-west-1": _StubTranscribeClient(),
        }
        self._patch_infra(monkeypatch, clients)

        with pytest.raises(InvalidLanguageFormatError) as excinfo:
            await _start_transcription_with_failover(
                [("us-east-1", "us-bucket"), ("eu-west-1", "eu-bucket")],
                "job1",
                "xx",
                "json",
            )
        assert excinfo.value.status == 400
        assert excinfo.value.code == "invalid_language_format"
        assert "'xx'" in str(excinfo.value), (
            "the error must name the rejected language code"
        )
        assert clients["eu-west-1"].started == []

    async def test_last_region_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every candidate fails, the last region's error is raised."""
        clients = {
            "us-east-1": _StubTranscribeClient(
                make_client_error("ThrottlingException", "StartTranscriptionJob")
            ),
            "eu-west-1": _StubTranscribeClient(
                make_client_error("ThrottlingException", "StartTranscriptionJob")
            ),
        }
        copies = self._patch_infra(monkeypatch, clients)

        with pytest.raises(ClientError) as excinfo:
            await _start_transcription_with_failover(
                [("us-east-1", "us-bucket"), ("eu-west-1", "eu-bucket")],
                "job1",
                None,
                "json",
            )
        assert excinfo.value.response["Error"]["Code"] == "ThrottlingException"
        # The input is still copied into the second region's bucket before
        # its (also failing) start attempt.
        assert copies == [("us-bucket", "eu-bucket", "eu-west-1")]

    async def test_botocore_param_validation_error_becomes_a_caller_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A client-side botocore rejection (e.g. MaxAlternatives<2) surfaces as ApiError, not a 500.

        botocore, not Transcribe, rejects an out-of-range ``Settings`` value, so the
        request never reaches AWS; the botocore report is kept verbatim as the
        message of a plain 400 (no error ``code``).

        Ref: botocore/data/transcribe/2017-10-26/service-2.json
             stdapi/models/audio/amazon_transcribe.py:_handle_transcription_error
        """
        clients = {
            "us-east-1": _StubTranscribeClient(
                ParamValidationError(
                    report="Invalid range for parameter Settings.MaxAlternatives"
                )
            )
        }
        self._patch_infra(monkeypatch, clients)

        with pytest.raises(ApiError) as excinfo:
            await _start_transcription_with_failover(
                [("us-east-1", "us-bucket")], "job1", None, "json"
            )

        assert excinfo.value.status == 400, "a client-side rejection must not be a 500"
        assert excinfo.value.code is None
        assert "Settings.MaxAlternatives" in str(excinfo.value), (
            "the botocore validation report must reach the caller"
        )

    async def test_failed_region_job_is_deleted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A timed-out first region gets a best-effort job deletion."""
        clients = {
            "us-east-1": _StubTranscribeClient(
                make_client_error("RequestTimeout", "StartTranscriptionJob")
            ),
            "eu-west-1": _StubTranscribeClient(),
        }
        self._patch_infra(monkeypatch, clients)

        region, bucket = await _start_transcription_with_failover(
            [("us-east-1", "us-bucket"), ("eu-west-1", "eu-bucket")],
            "job1",
            None,
            "json",
        )

        assert (region, bucket) == ("eu-west-1", "eu-bucket")
        assert clients["us-east-1"].deleted == ["job1"]
        assert clients["eu-west-1"].deleted == []

    async def test_all_regions_failed_deletes_the_job_everywhere(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every failed region, including the last one, gets the job deletion."""
        clients = {
            "us-east-1": _StubTranscribeClient(
                make_client_error("RequestTimeout", "StartTranscriptionJob")
            ),
            "eu-west-1": _StubTranscribeClient(
                make_client_error("RequestTimeout", "StartTranscriptionJob")
            ),
        }
        self._patch_infra(monkeypatch, clients)

        with pytest.raises(ClientError) as excinfo:
            await _start_transcription_with_failover(
                [("us-east-1", "us-bucket"), ("eu-west-1", "eu-bucket")],
                "job1",
                None,
                "json",
            )
        assert excinfo.value.response["Error"]["Code"] == "RequestTimeout"
        assert clients["us-east-1"].deleted == ["job1"]
        assert clients["eu-west-1"].deleted == ["job1"]


class _FakeAudioContent:
    """Minimal ``InputFile`` stand-in exposing only what ``stt`` needs."""

    async def get_filename(self) -> str | None:
        """Return no filename."""
        return None

    async def to_s3(self, region: str, *, bucket: str, key: str) -> None:
        """Accept the upload without doing anything."""


class TestSttDurationComputedOnce:
    """stt: the audio duration is computed once and reused for usage and formatting.

    Ref: stdapi/models/audio/amazon_transcribe.py:AudioModel.stt
    """

    async def test_empty_segments_warn_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A transcript with no audio segments emits the missing-duration warning once.

        The same computed duration feeds the usage record and the verbose_json
        payload, so a segment-less transcript must still yield a complete response
        (duration 0.0, empty segment list, billed seconds from the usage helper)
        with a single warning rather than one per consumer.
        """
        transcript_data: dict[str, Any] = {
            "transcripts": [{"transcript": "hello"}],
            "audio_segments": [],
            "items": [],
            "language_code": "en-US",
        }

        async def _fake_transcribe(
            _self: AudioModel, *_args: object, **_kwargs: object
        ) -> dict[str, Any]:
            return transcript_data

        monkeypatch.setattr(AudioModel, "_transcribe", _fake_transcribe)
        monkeypatch.setattr(
            amazon_transcribe, "record_transcribe_usage", lambda *_a, **_k: 15
        )

        response = await AudioModel(AWS_TRANSCRIBE_MODEL_ID).stt(
            _FakeAudioContent(),  # type: ignore[arg-type]
            "verbose_json",
            logprobs=False,
        )

        warnings = [
            d for d in request_log["error_detail"] if "15-second minimum" in str(d)
        ]
        assert len(warnings) == 1

        assert isinstance(response, TranscriptionVerbose)
        assert response.duration == 0.0
        assert response.text == "hello"
        assert response.language == "english"
        assert response.segments == []
        assert isinstance(response.usage, UsageDuration)
        assert response.usage.seconds == 15


class TestSttStreamRejectsLogprobs:
    """stt_stream: ``logprobs`` is rejected, matching the non-streaming ``stt`` path.

    Ref: stdapi/models/audio/amazon_transcribe.py:AudioModel.stt_stream
         stdapi/models/audio/__init__.py:AudioModelBase._validate_no_logprobs
    """

    async def test_logprobs_is_rejected_before_any_transcription_job_starts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``include=["logprobs"]`` fails with 400 without calling Transcribe.

        Amazon Transcribe returns no log probabilities on either path, and the
        non-streaming ``stt`` already rejects the parameter (see
        ``TestTranscribeUnsupportedParameters`` in test_openai_audio_transcriptions.py);
        streaming must reject it too instead of silently ignoring it.
        """

        async def _unexpected_transcribe(
            _self: AudioModel, *_args: object, **_kwargs: object
        ) -> dict[str, Any]:
            pytest.fail("_transcribe must not run once logprobs is rejected")

        monkeypatch.setattr(AudioModel, "_transcribe", _unexpected_transcribe)

        with pytest.raises(UnsupportedParameterError) as excinfo:
            async for _ in AudioModel(AWS_TRANSCRIBE_MODEL_ID).stt_stream(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "text",
                logprobs=True,
            ):
                pass

        assert excinfo.value.status == 400
        assert "logprobs" in str(excinfo.value)


class TestServedRegionStickiness:
    """stt: polling, results, and usage stay in the region that started the job.

    A Transcribe job only exists in the region that accepted it and writes its output
    to that region's bucket, so polling, result fetching and billing must all follow
    the served region rather than the first candidate.

    Ref: https://docs.aws.amazon.com/transcribe/latest/dg/what-is.html
         stdapi/models/audio/amazon_transcribe.py:_SERVED_REGION
    """

    @pytest.mark.usefixtures("request_log")
    async def test_polling_and_usage_use_the_serving_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After a failover start, everything runs against the serving region."""
        candidates = [("us-east-1", "us-bucket"), ("eu-west-1", "eu-bucket")]
        monkeypatch.setattr(
            amazon_transcribe, "transcribe_job_candidates", lambda: candidates
        )

        async def _fake_start(*_args: object, **_kwargs: object) -> tuple[str, str]:
            return "eu-west-1", "eu-bucket"

        monkeypatch.setattr(
            amazon_transcribe, "_start_transcription_with_failover", _fake_start
        )

        client = _StubTranscribeClient()
        client_regions: list[str | None] = []

        def _fake_get_client(
            _service: str, region: str | None = None
        ) -> _StubTranscribeClient:
            client_regions.append(region)
            return client

        monkeypatch.setattr(amazon_transcribe, "get_client", _fake_get_client)
        monkeypatch.setattr(
            amazon_transcribe, "track_temporary_s3_objects", lambda *_a: None
        )
        monkeypatch.setattr(
            amazon_transcribe, "schedule_cleanup", lambda coro: coro.close()
        )

        async def _fake_results(
            s3_bucket: str, _output_key: str, _subtitle_key: str | None
        ) -> dict[str, Any]:
            assert s3_bucket == "eu-bucket"
            return {
                "transcripts": [{"transcript": "hello"}],
                "audio_segments": [{"end_time": "1.0"}],
            }

        monkeypatch.setattr(
            amazon_transcribe, "_get_transcription_results", _fake_results
        )

        usage_regions: list[str] = []

        def _fake_usage(_duration: float, region: str = "") -> int:
            usage_regions.append(region)
            return 15

        monkeypatch.setattr(amazon_transcribe, "record_transcribe_usage", _fake_usage)

        id_token = REQUEST_ID.set("job2")
        try:
            await AudioModel(AWS_TRANSCRIBE_MODEL_ID).stt(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "json",
                logprobs=False,
            )
        finally:
            REQUEST_ID.reset(id_token)

        assert client_regions == ["eu-west-1"]
        assert client.polled == ["job2"]
        assert usage_regions == ["eu-west-1"]


class TestTranscriptOutputKeys:
    """_wait_for_transcription_completion: output keys come from the job description.

    Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_ContentRedaction.html
         stdapi/models/audio/amazon_transcribe.py:_wait_for_transcription_completion
    """

    @staticmethod
    def _client(transcript: dict[str, str], **extra: Any) -> Any:  # noqa: ANN401
        """Return a stub client reporting a completed job with *transcript*."""

        class _Client:
            @staticmethod
            async def get_transcription_job(**_params: Any) -> dict[str, Any]:  # noqa: ANN401
                return {
                    "TranscriptionJob": {
                        "TranscriptionJobStatus": "COMPLETED",
                        "Transcript": transcript,
                        **extra,
                    }
                }

        return _Client()

    async def test_redacted_uri_wins_over_plain_transcript(self) -> None:
        """Content redaction renames the output; the redacted key must be read.

        Transcribe prepends ``redacted-`` to the requested ``OutputKey`` file name and
        reports it as ``RedactedTranscriptFileUri``, so rebuilding the key from the job
        ID would read a non-existent object. Only ``RedactionOutput=redacted`` is
        supported, so there is no unredacted twin to fall back on.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/pii-redaction-output.html
        """
        client = self._client(
            {
                "RedactedTranscriptFileUri": (
                    "https://s3.us-east-1.amazonaws.com/b/tmp/job/redacted-output.json"
                )
            }
        )

        (
            output_key,
            subtitle_key,
        ) = await amazon_transcribe._wait_for_transcription_completion(  # noqa: SLF001
            client, "job", "b"
        )

        assert output_key == "tmp/job/redacted-output.json"
        assert subtitle_key is None

    async def test_subtitle_uri_is_returned(self) -> None:
        """A requested subtitle file is located from ``Subtitles.SubtitleFileUris``.

        Subtitle files land beside the transcript in the same S3 location, and the job
        description is the only place their final names are reported.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/subtitles.html
        """
        client = self._client(
            {
                "TranscriptFileUri": "https://s3.eu-west-1.amazonaws.com/b/tmp/job/out.json"
            },
            Subtitles={
                "SubtitleFileUris": [
                    "https://s3.eu-west-1.amazonaws.com/b/tmp/job/out.srt"
                ]
            },
        )

        (
            output_key,
            subtitle_key,
        ) = await amazon_transcribe._wait_for_transcription_completion(  # noqa: SLF001
            client, "job", "b"
        )

        assert output_key == "tmp/job/out.json"
        assert subtitle_key == "tmp/job/out.srt"


class TestNoCandidateRegions:
    """No usable bucket anywhere: documented 404 on requests.

    Ref: stdapi/models/audio/amazon_transcribe.py:AudioModel._transcribe
    """

    @pytest.mark.usefixtures("request_log")
    async def test_request_raises_documented_404(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request without any candidate region fails with the 404 guard."""
        monkeypatch.setattr(amazon_transcribe, "transcribe_job_candidates", list)
        with pytest.raises(ApiError) as excinfo:
            await AudioModel(AWS_TRANSCRIBE_MODEL_ID).stt(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "json",
                logprobs=False,
            )
        assert excinfo.value.status == 404
        assert "not available" in str(excinfo.value)


class TestInitializeTranscribeModels:
    """initialize_transcribe_models: regions metadata mirrors the candidates.

    Ref: stdapi/models/audio/amazon_transcribe.py:initialize_transcribe_models
    """

    @pytest.fixture(autouse=True)
    def _restore_extra_models(self) -> Generator[None]:
        """Restore the shared model registry entry after each test."""
        saved = EXTRA_MODELS.get(AWS_TRANSCRIBE_MODEL_ID)
        yield
        if saved is None:
            EXTRA_MODELS.pop(AWS_TRANSCRIBE_MODEL_ID, None)
        else:
            EXTRA_MODELS[AWS_TRANSCRIBE_MODEL_ID] = saved

    async def test_registers_the_candidate_regions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The advertised regions are the bucket-equipped candidates."""
        monkeypatch.setattr(SETTINGS, "aws_transcribe_region", None)
        monkeypatch.setattr(SETTINGS, "aws_transcribe_s3_bucket", "primary-bucket")
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1", "eu-west-1"])
        monkeypatch.setattr(
            SETTINGS, "aws_s3_regional_buckets", {"eu-west-1": "eu-bucket"}
        )
        await initialize_transcribe_models()
        model = EXTRA_MODELS[AWS_TRANSCRIBE_MODEL_ID]
        assert model.regions == ["us-east-1", "eu-west-1"]

    async def test_registers_empty_regions_without_any_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without any usable bucket the model stays registered, regions empty."""
        monkeypatch.setattr(SETTINGS, "aws_transcribe_region", "us-west-2")
        monkeypatch.setattr(SETTINGS, "aws_transcribe_s3_bucket", None)
        monkeypatch.setattr(SETTINGS, "aws_s3_regional_buckets", {})
        await initialize_transcribe_models()
        assert EXTRA_MODELS[AWS_TRANSCRIBE_MODEL_ID].regions == []


class TestBucketToRegionMapping:
    """_bucket_to_region: the Transcribe bucket resolves to the transcribe region.

    Temporary object cleanup needs the region each bucket lives in, and the dedicated
    Transcribe bucket may sit outside the Bedrock region set.

    Ref: stdapi/aws_s3.py:_bucket_to_region
    """

    def test_dedicated_transcribe_bucket_maps_to_the_transcribe_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dedicated Transcribe bucket maps to the configured transcribe region."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1"])
        monkeypatch.setattr(SETTINGS, "aws_s3_regional_buckets", {})
        monkeypatch.setattr(SETTINGS, "aws_s3_bucket", "primary-bucket")
        monkeypatch.setattr(SETTINGS, "aws_transcribe_region", "eu-west-1")
        monkeypatch.setattr(SETTINGS, "aws_transcribe_s3_bucket", "transcribe-bucket")
        mapping = _bucket_to_region()
        assert mapping["transcribe-bucket"] == "eu-west-1"
        assert mapping["primary-bucket"] == "us-east-1"

    def test_bucket_shared_with_the_primary_keeps_its_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The config-defaulted (shared) Transcribe bucket keeps its mapping."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1"])
        monkeypatch.setattr(SETTINGS, "aws_s3_regional_buckets", {})
        monkeypatch.setattr(SETTINGS, "aws_s3_bucket", "primary-bucket")
        monkeypatch.setattr(SETTINGS, "aws_transcribe_region", "eu-west-1")
        monkeypatch.setattr(SETTINGS, "aws_transcribe_s3_bucket", "primary-bucket")
        assert _bucket_to_region()["primary-bucket"] == "us-east-1"

    def test_transcribe_bucket_without_transcribe_region_uses_the_primary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a transcribe region the bucket maps to the primary region."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1"])
        monkeypatch.setattr(SETTINGS, "aws_s3_regional_buckets", {})
        monkeypatch.setattr(SETTINGS, "aws_s3_bucket", None)
        monkeypatch.setattr(SETTINGS, "aws_transcribe_region", None)
        monkeypatch.setattr(SETTINGS, "aws_transcribe_s3_bucket", "transcribe-bucket")
        assert _bucket_to_region()["transcribe-bucket"] == "us-east-1"


class TestFormatTranslationResponseVerboseJson:
    """_format_translation_response: verbose_json segments are translated too.

    The top-level text comes from one TranslateText call over the whole transcript;
    the per-segment texts need their own calls, which must carry the same Translate
    ``Settings``/``TerminologyNames`` as the main one.

    Ref: https://docs.aws.amazon.com/translate/latest/APIReference/API_TranslateText.html
         stdapi/models/audio/amazon_transcribe.py:AudioModel._format_translation_response
    """

    async def test_segments_are_translated_not_left_in_source_language(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each segment's text is translated, not just the top-level text."""
        transcript_data: dict[str, Any] = {
            "language_code": "fr-FR",
            "audio_segments": [
                {
                    "id": 0,
                    "start_time": "0.0",
                    "end_time": "1.0",
                    "transcript": "Bonjour",
                },
                {
                    "id": 1,
                    "start_time": "1.0",
                    "end_time": "2.0",
                    "transcript": "le monde",
                },
            ],
        }
        translations = {"Bonjour": "Hello", "le monde": "the world"}

        async def _fake_translate(
            text: str,
            language: str,
            target_language_code: str = "en",  # noqa: ARG001
            settings: dict[str, str] | None = None,
            terminology_names: list[str] | None = None,
        ) -> str:
            assert language == "fr-FR"
            assert settings is None
            assert terminology_names is None
            return translations[text]

        monkeypatch.setattr(amazon_transcribe, "translate", _fake_translate)

        response = await AudioModel._format_translation_response(  # noqa: SLF001
            transcript_data,  # type: ignore[arg-type]
            "Hello the world",
            "verbose_json",
        )

        assert isinstance(response, TranslationVerbose)
        assert response.segments is not None
        assert [segment.text for segment in response.segments] == ["Hello", "the world"]

    async def test_settings_and_terminology_names_reach_each_segment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Settings/TerminologyNames are forwarded to every per-segment translate call."""
        transcript_data: dict[str, Any] = {
            "language_code": "fr-FR",
            "audio_segments": [
                {
                    "id": 0,
                    "start_time": "0.0",
                    "end_time": "1.0",
                    "transcript": "Bonjour",
                }
            ],
        }
        calls: list[tuple[dict[str, str] | None, list[str] | None]] = []

        async def _fake_translate(
            text: str,  # noqa: ARG001
            language: str,  # noqa: ARG001
            target_language_code: str = "en",  # noqa: ARG001
            settings: dict[str, str] | None = None,
            terminology_names: list[str] | None = None,
        ) -> str:
            calls.append((settings, terminology_names))
            return "Hello"

        monkeypatch.setattr(amazon_transcribe, "translate", _fake_translate)

        await AudioModel._format_translation_response(  # noqa: SLF001
            transcript_data,  # type: ignore[arg-type]
            "Hello",
            "verbose_json",
            settings={"Formality": "FORMAL"},
            terminology_names=["MyGlossary"],
        )

        assert calls == [({"Formality": "FORMAL"}, ["MyGlossary"])]


class TestPopTranslateExtraParams:
    """_pop_translate_extra_params: split Translate's Settings/TerminologyNames out.

    On the translation route the extra-params bag is shared by two services, so
    Translate's own keys must be removed before the rest is validated as
    StartTranscriptionJob parameters.

    Ref: stdapi/models/audio/amazon_transcribe.py:_pop_translate_extra_params
    """

    def test_none_input_passes_through(self) -> None:
        """A missing extra_params dict stays None for all three outputs."""
        assert amazon_transcribe._pop_translate_extra_params(None) == (  # noqa: SLF001
            None,
            None,
            None,
        )

    def test_only_translate_keys_leaves_nothing_for_the_job(self) -> None:
        """A dict with only Settings/TerminologyNames leaves no job extra_params."""
        remaining, settings, terminology_names = (
            amazon_transcribe._pop_translate_extra_params(  # noqa: SLF001
                {
                    "Settings": {"Formality": "FORMAL"},
                    "TerminologyNames": ["MyGlossary"],
                }
            )
        )
        assert remaining is None
        assert settings == {"Formality": "FORMAL"}
        assert terminology_names == ["MyGlossary"]

    def test_mixed_keys_keep_the_transcribe_fields_for_the_job(self) -> None:
        """Transcribe-only keys stay in the remaining dict, untouched."""
        remaining, settings, terminology_names = (
            amazon_transcribe._pop_translate_extra_params(  # noqa: SLF001
                {"Settings": {"Profanity": "MASK"}, "VocabularyName": "MyVocabulary"}
            )
        )
        assert remaining == {"VocabularyName": "MyVocabulary"}
        assert settings == {"Profanity": "MASK"}
        assert terminology_names is None


class TestTextCompressionRatio:
    """_text_compression_ratio: a real, text-based repetition signal.

    Amazon Transcribe exposes no decoder-internal metric, so verbose_json's
    ``compression_ratio`` is computed from the returned text here; the threshold
    mirrors OpenAI's "above 2.4 suggests compression failed" wording.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/models/audio/amazon_transcribe.py:_text_compression_ratio
    """

    def test_empty_text_is_zero(self) -> None:
        """An empty segment reports a ratio of 0.0, not a false-confident value."""
        assert _text_compression_ratio("") == 0.0

    def test_repetitive_text_yields_a_high_ratio(self) -> None:
        """Highly repetitive text compresses well: a high ratio, as Whisper defines it."""
        repetitive = "the quick brown fox " * 50
        assert _text_compression_ratio(repetitive) > 2.4


class TestBuildTranscriptionSegment:
    """_build_transcription_segment: derives confidence stats Transcribe lacks.

    ``avg_logprob`` and ``no_speech_prob`` have no AWS equivalent; they are set so
    that OpenAI's documented silence signal (no_speech_prob at 1.0 together with
    avg_logprob below -1) still fires on a segment with no text.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/models/audio/amazon_transcribe.py:_build_transcription_segment
    """

    def test_segment_with_transcript_reports_confident_values(self) -> None:
        """A segment with text reports the confident (non-silent) defaults."""
        segment: dict[str, Any] = {
            "id": 0,
            "start_time": "0.0",
            "end_time": "1.0",
            "transcript": "hello",
        }
        result = _build_transcription_segment(segment, "hello")  # type: ignore[arg-type]

        assert result.no_speech_prob == 0.0
        assert result.avg_logprob == 0.0
        assert result.compression_ratio == _text_compression_ratio("hello")

    def test_segment_without_transcript_reports_silence_consistently(self) -> None:
        """An empty segment reports the documented combined silence signal."""
        segment: dict[str, Any] = {
            "id": 0,
            "start_time": "0.0",
            "end_time": "1.0",
            "transcript": "",
        }
        result = _build_transcription_segment(segment, "")  # type: ignore[arg-type]

        assert result.no_speech_prob == 1.0
        assert result.avg_logprob < -1.0


class TestTranscribeExtraParamsValueConstraints:
    """_TranscribeExtraParams: values are forwarded to AWS verbatim, not re-validated.

    Amazon Transcribe already rejects an unsupported enum/range value with its
    own 400 error (surfaced via ``_handle_transcription_error``); duplicating
    that validation here would only reject clients AWS itself would accept
    (e.g. a newly added PII entity type). Only field *names* and the redaction
    output mode are validated locally.

    Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_ContentRedaction.html
         stdapi/models/audio/amazon_transcribe.py:_TranscribeExtraParams
    """

    def test_unknown_pii_entity_type_is_accepted(self) -> None:
        """A PiiEntityTypes value outside the documented set is forwarded as-is."""
        extra = _TranscribeExtraParams(
            ContentRedaction=_TranscribeContentRedaction(
                RedactionType="PII", PiiEntityTypes=["NOT_A_REAL_ENTITY"]
            )
        )
        assert extra.ContentRedaction is not None
        assert extra.ContentRedaction.PiiEntityTypes == ["NOT_A_REAL_ENTITY"]

    def test_redacted_and_unredacted_output_is_rejected(self) -> None:
        """The dual-output redaction mode is rejected (no tracked-cleanup path yet).

        ``redacted_and_unredacted`` makes Transcribe write two objects, only one of
        which the gateway tracks for cleanup, so the field is pinned to ``redacted``.
        """
        with pytest.raises(ValidationError) as excinfo:
            _TranscribeExtraParams(
                ContentRedaction={  # type: ignore[arg-type]
                    "RedactionType": "PII",
                    "RedactionOutput": "redacted_and_unredacted",
                }
            )

        errors = excinfo.value.errors()
        assert [error["type"] for error in errors] == ["literal_error"]
        assert "RedactionOutput" in errors[0]["loc"]

    def test_unknown_vocabulary_filter_method_is_accepted(self) -> None:
        """A VocabularyFilterMethod outside mask/remove/tag is forwarded as-is."""
        extra = _TranscribeExtraParams(VocabularyFilterMethod="delete")
        assert extra.VocabularyFilterMethod == "delete"

    def test_max_alternatives_out_of_range_is_accepted(self) -> None:
        """MaxAlternatives outside AWS's documented 2-10 range is forwarded as-is."""
        assert _TranscribeExtraParams(MaxAlternatives=1).MaxAlternatives == 1

    def test_max_speaker_labels_out_of_range_is_accepted(self) -> None:
        """MaxSpeakerLabels outside AWS's documented 2-30 range is forwarded as-is."""
        assert _TranscribeExtraParams(MaxSpeakerLabels=31).MaxSpeakerLabels == 31

    def test_unknown_top_level_field_is_rejected(self) -> None:
        """An unsupported field name (extra="forbid") fails validation.

        Values are forwarded to AWS unchecked, so the field name is the only local
        guard against a typo silently reaching StartTranscriptionJob.
        """
        with pytest.raises(ValidationError) as excinfo:
            _TranscribeExtraParams(NotARealField=True)  # type: ignore[call-arg]

        errors = excinfo.value.errors()
        assert [error["type"] for error in errors] == ["extra_forbidden"]
        assert "NotARealField" in errors[0]["loc"]

    def test_documented_values_round_trip(self) -> None:
        """A fully populated, valid extra-params payload validates and dumps back."""
        extra = _TranscribeExtraParams(
            ContentRedaction=_TranscribeContentRedaction(
                RedactionType="PII", PiiEntityTypes=["NAME", "SSN"]
            ),
            VocabularyFilterName="myfilter",
            VocabularyFilterMethod="mask",
            ShowAlternatives=True,
            MaxAlternatives=3,
            ToxicityDetection=[
                _TranscribeToxicityDetectionSetting(ToxicityCategories=["ALL"])
            ],
            ModelSettings=_TranscribeModelSettings(LanguageModelName="my-clm"),
            LanguageOptions=["en-US", "es-US"],
            IdentifyMultipleLanguages=True,
        )
        dumped = extra.model_dump(exclude_none=True)
        assert dumped["VocabularyFilterMethod"] == "mask"
        assert dumped["ContentRedaction"] == {
            "RedactionType": "PII",
            "RedactionOutput": "redacted",
            "PiiEntityTypes": ["NAME", "SSN"],
        }


class TestBuildTranscriptionJobParamsExtra:
    """_build_transcription_job_params: extra_params merge into StartTranscriptionJob.

    AWS requires exactly one of LanguageCode / IdentifyLanguage /
    IdentifyMultipleLanguages, and nests several of these fields under ``Settings``,
    so the flat extra-params bag has to be redistributed rather than merged as-is.

    Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartTranscriptionJob.html
         stdapi/models/audio/amazon_transcribe.py:_build_transcription_job_params
    """

    @pytest.fixture(autouse=True)
    def _request_context(self, request_log: dict[str, Any]) -> Generator[None]:
        """Provide the request ID the job tags are built from."""
        id_token = REQUEST_ID.set("job1")
        yield
        REQUEST_ID.reset(id_token)

    def test_content_redaction_is_forwarded(self) -> None:
        """ContentRedaction is passed through to the job params unchanged."""
        extra = _TranscribeExtraParams(
            ContentRedaction=_TranscribeContentRedaction(
                RedactionType="PII", PiiEntityTypes=["NAME"]
            )
        )
        params = _build_transcription_job_params("job1", "bucket", "en", "json", extra)
        assert params["ContentRedaction"] == {
            "RedactionType": "PII",
            "RedactionOutput": "redacted",
            "PiiEntityTypes": ["NAME"],
        }

    def test_vocabulary_and_alternatives_settings_are_merged(self) -> None:
        """VocabularyFilterName/Method and ShowAlternatives/MaxAlternatives land under Settings."""
        extra = _TranscribeExtraParams(
            VocabularyFilterName="myfilter",
            VocabularyFilterMethod="mask",
            ShowAlternatives=True,
            MaxAlternatives=4,
        )
        params = _build_transcription_job_params("job1", "bucket", "en", "json", extra)
        assert params["Settings"] == {
            "VocabularyFilterName": "myfilter",
            "VocabularyFilterMethod": "mask",
            "ShowAlternatives": True,
            "MaxAlternatives": 4,
        }

    def test_max_speaker_labels_overrides_the_diarized_json_default(self) -> None:
        """extra.MaxSpeakerLabels overrides the hardcoded default of 10."""
        extra = _TranscribeExtraParams(MaxSpeakerLabels=25)
        params = _build_transcription_job_params(
            "job1", "bucket", "en", "diarized_json", extra
        )
        assert params["Settings"] == {"ShowSpeakerLabels": True, "MaxSpeakerLabels": 25}

    def test_toxicity_detection_and_model_settings_are_forwarded(self) -> None:
        """ToxicityDetection and ModelSettings.LanguageModelName reach the job params."""
        extra = _TranscribeExtraParams(
            ToxicityDetection=[
                _TranscribeToxicityDetectionSetting(ToxicityCategories=["ALL"])
            ],
            ModelSettings=_TranscribeModelSettings(LanguageModelName="my-clm"),
        )
        params = _build_transcription_job_params("job1", "bucket", "en", "json", extra)
        assert params["ToxicityDetection"] == [{"ToxicityCategories": ["ALL"]}]
        assert params["ModelSettings"] == {"LanguageModelName": "my-clm"}

    def test_identify_multiple_languages_supersedes_language_code(self) -> None:
        """IdentifyMultipleLanguages wins over an explicit language, per AWS's mutual exclusion."""
        extra = _TranscribeExtraParams(
            IdentifyMultipleLanguages=True, LanguageOptions=["en-US", "fr-FR"]
        )
        params = _build_transcription_job_params("job1", "bucket", "en", "json", extra)
        assert params["IdentifyMultipleLanguages"] is True
        assert params["LanguageOptions"] == ["en-US", "fr-FR"]
        assert "LanguageCode" not in params
        assert "IdentifyLanguage" not in params

    def test_language_options_apply_with_auto_identification(self) -> None:
        """LanguageOptions without an explicit language narrows auto-identification."""
        extra = _TranscribeExtraParams(LanguageOptions=["en-US", "fr-FR"])
        params = _build_transcription_job_params("job1", "bucket", None, "json", extra)
        assert params["IdentifyLanguage"] is True
        assert params["LanguageOptions"] == ["en-US", "fr-FR"]

    def test_channel_identification_conflicts_with_diarized_json(self) -> None:
        """ChannelIdentification with diarized_json is rejected with a clean 400.

        diarized_json forces ``ShowSpeakerLabels``, which AWS refuses to combine with
        channel identification; the conflict is reported as the offending parameter
        instead of being forwarded and failing the job.

        Ref: https://docs.aws.amazon.com/transcribe/latest/dg/diarization.html
        """
        extra = _TranscribeExtraParams(ChannelIdentification=True)
        with pytest.raises(UnsupportedParameterError) as excinfo:
            _build_transcription_job_params(
                "job1", "bucket", "en", "diarized_json", extra
            )

        assert excinfo.value.status == 400
        assert excinfo.value.code == "unsupported_parameter"
        assert excinfo.value.param == "ChannelIdentification"
        assert "ChannelIdentification" in str(excinfo.value)

    def test_channel_identification_alone_is_accepted(self) -> None:
        """ChannelIdentification without diarized_json is accepted normally."""
        extra = _TranscribeExtraParams(ChannelIdentification=True)
        params = _build_transcription_job_params("job1", "bucket", "en", "json", extra)
        assert params["Settings"] == {"ChannelIdentification": True}

    def test_no_extra_params_matches_prior_behavior(self) -> None:
        """Without extra_params, the job params are unchanged from before #82."""
        params = _build_transcription_job_params("job1", "bucket", "en", "json")
        assert "Settings" not in params
        assert "ContentRedaction" not in params
