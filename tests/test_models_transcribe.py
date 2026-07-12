"""Unit tests for AWS Transcribe audio-duration extraction feeding billed usage."""

from datetime import UTC, datetime
from typing import Any

from stdapi.models.audio.amazon_transcribe import _get_audio_duration
from stdapi.monitoring import REQUEST_LOG, EventLog


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
