"""Unit tests for AWS Transcribe: audio duration, region candidates, failover."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError

import stdapi.aws
from stdapi.api_errors import InvalidLanguageFormatError
from stdapi.config import SETTINGS
from stdapi.models.audio import amazon_transcribe
from stdapi.models.audio.amazon_transcribe import (
    AWS_TRANSCRIBE_MODEL_ID,
    AudioModel,
    _get_audio_duration,
    _start_transcription_with_failover,
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

    async def start_transcription_job(self, **params: Any) -> None:  # noqa: ANN401
        """Record the job params or raise the configured error."""
        if self._error is not None:
            raise self._error
        self.started.append(params)


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


class _FakeAudioContent:
    """Minimal ``InputFile`` stand-in exposing only what ``stt`` needs."""

    async def get_filename(self) -> str | None:
        """Return no filename."""
        return None


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
