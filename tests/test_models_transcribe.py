"""Unit tests for AWS Transcribe: audio duration, region candidates, failover."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError

import stdapi.aws
from stdapi.api_errors import ApiError, InvalidLanguageFormatError
from stdapi.aws_s3 import _bucket_to_region
from stdapi.config import SETTINGS
from stdapi.models import EXTRA_MODELS
from stdapi.models.audio import amazon_transcribe
from stdapi.models.audio.amazon_transcribe import (
    AWS_TRANSCRIBE_MODEL_ID,
    AudioModel,
    _get_audio_duration,
    _start_transcription_with_failover,
    initialize_transcribe_models,
    transcribe_job_candidates,
)
from stdapi.monitoring import REQUEST_ID, REQUEST_LOG, EventLog

if TYPE_CHECKING:
    from collections.abc import Generator

    from types_aiobotocore_bedrock.literals import RegionName


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


def _new_log() -> EventLog:
    return EventLog(
        type="request",
        level="info",
        date=datetime.now(UTC),
        server_id="test",
        server_version="0.0.0",
    )


class TestGetAudioDuration:
    """_get_audio_duration: the last segment's end time is the billed duration."""

    def test_returns_last_segment_end_time(self) -> None:
        """The duration is the end time of the final audio segment."""
        data: dict[str, Any] = {
            "audio_segments": [{"end_time": "1.5"}, {"end_time": "42.75"}]
        }
        assert _get_audio_duration(data) == 42.75  # type: ignore[arg-type]

    def test_missing_segments_warns_and_returns_zero(self) -> None:
        """No segments: return 0.0 (15s minimum billed) and warn in the request log."""
        log = _new_log()
        token = REQUEST_LOG.set(log)
        try:
            assert _get_audio_duration({}) == 0.0
        finally:
            REQUEST_LOG.reset(token)
        assert log["level"] == "warning"
        assert any("15-second minimum" in str(d) for d in log["error_detail"])

    def test_empty_segments_list_warns_and_returns_zero(self) -> None:
        """An empty segment list must behave like a missing one."""
        log = _new_log()
        token = REQUEST_LOG.set(log)
        try:
            data: dict[str, Any] = {"audio_segments": []}
            assert _get_audio_duration(data) == 0.0  # type: ignore[arg-type]
        finally:
            REQUEST_LOG.reset(token)
        assert log["level"] == "warning"


class TestTranscribeJobCandidates:
    """transcribe_job_candidates: per-region bucket pairing rules."""

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
        self.polled.append(params["TranscriptionJobName"])
        return {"TranscriptionJob": {"TranscriptionJobStatus": "COMPLETED"}}


def _client_error(code: str, message: str = "x") -> ClientError:
    response: Any = {"Error": {"Code": code, "Message": message}}
    return ClientError(response, "StartTranscriptionJob")


class TestStartTranscriptionWithFailover:
    """_start_transcription_with_failover: whole-job region failover."""

    @pytest.fixture(autouse=True)
    def _request_context(self) -> Generator[None]:
        """Provide the request ID and log the job tags are built from."""
        id_token = REQUEST_ID.set("job1")
        log_token = REQUEST_LOG.set(_new_log())
        yield
        REQUEST_LOG.reset(log_token)
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
            "us-east-1": _StubTranscribeClient(_client_error("ThrottlingException")),
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
        """A bad language code raises immediately without trying region 2."""
        clients = {
            "us-east-1": _StubTranscribeClient(
                _client_error("BadRequestException", "unsupported languageCode")
            ),
            "eu-west-1": _StubTranscribeClient(),
        }
        self._patch_infra(monkeypatch, clients)

        with pytest.raises(InvalidLanguageFormatError):
            await _start_transcription_with_failover(
                [("us-east-1", "us-bucket"), ("eu-west-1", "eu-bucket")],
                "job1",
                "xx",
                "json",
            )
        assert clients["eu-west-1"].started == []

    async def test_last_region_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every candidate fails, the last region's error is raised."""
        clients = {
            "us-east-1": _StubTranscribeClient(_client_error("ThrottlingException")),
            "eu-west-1": _StubTranscribeClient(_client_error("ThrottlingException")),
        }
        copies = self._patch_infra(monkeypatch, clients)

        with pytest.raises(ClientError):
            await _start_transcription_with_failover(
                [("us-east-1", "us-bucket"), ("eu-west-1", "eu-bucket")],
                "job1",
                None,
                "json",
            )
        # The input is still copied into the second region's bucket before
        # its (also failing) start attempt.
        assert copies == [("us-bucket", "eu-bucket", "eu-west-1")]

    async def test_failed_region_job_is_deleted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A timed-out first region gets a best-effort job deletion."""
        clients = {
            "us-east-1": _StubTranscribeClient(_client_error("RequestTimeout")),
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
            "us-east-1": _StubTranscribeClient(_client_error("RequestTimeout")),
            "eu-west-1": _StubTranscribeClient(_client_error("RequestTimeout")),
        }
        self._patch_infra(monkeypatch, clients)

        with pytest.raises(ClientError):
            await _start_transcription_with_failover(
                [("us-east-1", "us-bucket"), ("eu-west-1", "eu-bucket")],
                "job1",
                None,
                "json",
            )
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
    """stt: the audio duration is computed once and reused for usage and formatting."""

    async def test_empty_segments_warn_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A transcript with no audio segments emits the missing-duration warning once."""
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

        log = _new_log()
        token = REQUEST_LOG.set(log)
        try:
            await AudioModel(AWS_TRANSCRIBE_MODEL_ID).stt(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "verbose_json",
                logprobs=False,
            )
        finally:
            REQUEST_LOG.reset(token)

        warnings = [d for d in log["error_detail"] if "15-second minimum" in str(d)]
        assert len(warnings) == 1


class TestServedRegionStickiness:
    """stt: polling, results, and usage stay in the region that started the job."""

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
            s3_bucket: str, _job_id: str, _response_format: str
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
        log_token = REQUEST_LOG.set(_new_log())
        try:
            await AudioModel(AWS_TRANSCRIBE_MODEL_ID).stt(
                _FakeAudioContent(),  # type: ignore[arg-type]
                "json",
                logprobs=False,
            )
        finally:
            REQUEST_LOG.reset(log_token)
            REQUEST_ID.reset(id_token)

        assert client_regions == ["eu-west-1"]
        assert client.polled == ["job2"]
        assert usage_regions == ["eu-west-1"]


class TestNoCandidateRegions:
    """No usable bucket anywhere: documented 404 on requests."""

    async def test_request_raises_documented_404(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request without any candidate region fails with the 404 guard."""
        monkeypatch.setattr(amazon_transcribe, "transcribe_job_candidates", list)
        log_token = REQUEST_LOG.set(_new_log())
        try:
            with pytest.raises(ApiError) as excinfo:
                await AudioModel(AWS_TRANSCRIBE_MODEL_ID).stt(
                    _FakeAudioContent(),  # type: ignore[arg-type]
                    "json",
                    logprobs=False,
                )
        finally:
            REQUEST_LOG.reset(log_token)
        assert excinfo.value.status == 404
        assert "not available" in str(excinfo.value)


class TestInitializeTranscribeModels:
    """initialize_transcribe_models: regions metadata mirrors the candidates."""

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
    """_bucket_to_region: the Transcribe bucket resolves to the transcribe region."""

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
