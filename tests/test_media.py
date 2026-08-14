"""ffmpeg audio transcoding: the pipe handling that keeps a request bounded.

The encoder feeds ffmpeg over stdin and streams stdout back to the client, which
makes the request's lifetime depend on a subprocess. Both hazards of that are
covered here: stderr filling its pipe buffer, and ffmpeg producing nothing at
all. A gateway serving long-lived deployments must turn either into an error
rather than a request that never ends.

These drive a real subprocess, but a shell rather than ffmpeg, so they stay
offline and deterministic.

Ref: https://docs.python.org/3/library/asyncio-subprocess.html
     stdapi/media.py:encode_audio_stream
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from stdapi.api_errors import ApiError, FeatureUnavailableError
from stdapi.media import _drain, _ffmpeg_args, encode_audio_stream

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from typing import Any

pytestmark = pytest.mark.local

#: Stderr volume that deadlocks an undrained pipe.
#:
#: asyncio buffers a subprocess pipe into memory on its own, so a small burst is
#: harmless; past its stream limit it pauses reading and the OS pipe fills behind
#: it. Measured on this platform: 256 KiB still completes, 1 MiB does not.
_DEADLOCK_STDERR = 1024 * 1024


async def _feed(*chunks: bytes) -> AsyncGenerator[bytes]:
    """Yield each chunk as an input stream would."""
    for chunk in chunks:
        yield chunk


class TestFfmpegArguments:
    """The command line built for one encode.

    Ref: https://ffmpeg.org/ffmpeg.html
         stdapi/media.py:_ffmpeg_args
    """

    def test_encoded_input_is_autodetected(self) -> None:
        """Without an input format, no ``-f`` precedes the input."""
        args = _ffmpeg_args("mp3", None, None, None, None)
        assert args[:3] == ["ffmpeg", "-i", "pipe:0"]
        assert args[-3:] == ["-f", "mp3", "pipe:1"]

    def test_raw_pcm_declares_its_rate_and_channels(self) -> None:
        """Raw PCM input carries the format, rate and channel count ffmpeg needs."""
        args = _ffmpeg_args("flac", "pcm", 16000, 1, None)
        assert args[1:7] == ["-f", "s16le", "-ar", "16000", "-ac", "1"]

    def test_raw_pcm_without_rate_or_channels_is_refused(self) -> None:
        """Raw PCM with neither rate nor channels cannot be decoded, so it raises."""
        with pytest.raises(ValueError, match="sample_rate or channels"):
            _ffmpeg_args("flac", "pcm", None, None, None)

    def test_output_resampling_is_requested_after_the_input(self) -> None:
        """``output_sample_rate`` adds an ``-ar`` on the output side.

        The input side may carry its own ``-ar``; the resample must be the later
        one, since ffmpeg applies options to the next file named after them.
        """
        args = _ffmpeg_args("pcm", "pcm", 16000, 1, 24000)
        assert args.index("pipe:0") < len(args) - 1 - args[::-1].index("-ar")

    def test_output_downmix_is_requested_after_the_input(self) -> None:
        """``output_channels`` adds an ``-ac`` on the output side.

        Backends that only accept mono need the downmix applied to the output;
        an ``-ac`` before the input declares the *source* layout instead, which
        silently misreads a stereo upload.
        """
        args = _ffmpeg_args("pcm", None, None, None, 16000, 1)
        downmix = len(args) - 1 - args[::-1].index("-ac")
        assert args.index("pipe:0") < downmix
        assert args[downmix + 1] == "1"

    def test_output_channels_are_left_alone_by_default(self) -> None:
        """Without ``output_channels`` no channel option reaches the output side."""
        args = _ffmpeg_args("pcm", None, None, None, 16000)
        assert "-ac" not in args


class TestStderrIsDrained:
    """ffmpeg's stderr is consumed so its pipe buffer cannot fill.

    ffmpeg reports progress and warnings on stderr for the whole encode. asyncio
    buffers that pipe only up to its stream limit; past it, reading pauses, the
    OS pipe fills, and ffmpeg blocks writing to stderr -- so it stops consuming
    stdin and producing stdout, and the request ends only when the client gives
    up. A malformed input that makes ffmpeg warn per frame reaches that volume
    quickly.

    Ref: https://docs.python.org/3/library/asyncio-subprocess.html#asyncio.subprocess.Process.communicate
         stdapi/media.py:_drain
    """

    async def test_output_survives_more_stderr_than_a_pipe_holds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A process writing 1 MiB to stderr still streams its stdout.

        Verified to deadlock without the drain: the child blocks writing stderr
        before it has written any stdout, and the encode never finishes.
        """
        noise = _DEADLOCK_STDERR
        monkeypatch.setattr(
            "stdapi.media._ffmpeg_args",
            lambda *_args: [
                "sh",
                "-c",
                f"cat >/dev/null; head -c {noise} /dev/zero >&2; echo -n done",
            ],
        )
        chunks = [chunk async for chunk in encode_audio_stream(_feed(b"audio"), "mp3")]
        assert b"".join(chunks) == b"done"

    async def test_drain_keeps_only_the_head(self) -> None:
        """The drain keeps a bounded head, not everything ffmpeg ever wrote.

        The head is what matters: ffmpeg names the option it rejected first, then
        may print its whole format table, which would evict the message itself
        from a tail-shaped buffer. The head lands in the caller's buffer as it is
        read, so it stays loggable even while the drain is still running.
        """

        class _Stream:
            def __init__(self) -> None:
                self.remaining = 40

            async def read(self, _size: int) -> bytes:
                if not self.remaining:
                    return b""
                self.remaining -= 1
                return b"first" if self.remaining == 39 else b"x" * 1000

        captured = bytearray()
        await _drain(_Stream(), captured)  # type: ignore[arg-type]
        assert len(captured) <= 4096
        assert captured.startswith(b"first"), "the head is kept, not the tail"


class TestStalledEncodeIsBounded:
    """An ffmpeg that never produces output fails the request instead of hanging.

    Ref: https://developers.openai.com/api/reference/resources/audio
         stdapi/media.py:encode_audio_stream
    """

    async def test_silent_process_raises_a_gateway_timeout(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A process that emits nothing is terminated and reported as a 504.

        The timeout is shortened here; what is asserted is that the wait is
        bounded at all, and that the error names the encode rather than leaking
        a subprocess detail.
        """
        monkeypatch.setattr("stdapi.media._ENCODE_CHUNK_TIMEOUT", 0.5)
        monkeypatch.setattr(
            "stdapi.media._ffmpeg_args", lambda *_args: ["sh", "-c", "sleep 60"]
        )
        with pytest.raises(ApiError) as excinfo:
            async for _chunk in encode_audio_stream(_feed(b"audio"), "flac"):
                pass
        assert excinfo.value.status == 504
        assert "flac" in str(excinfo.value)

    async def test_the_timeout_log_names_exit_code_input_state_and_stderr(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """The 504 log carries the exit code, the input task's state and stderr.

        Those three fields say whether the stall was ffmpeg's fault or its
        input's without a second reproduction, and the stderr head must be
        readable even though the drain task is still blocked on a live pipe.
        """
        monkeypatch.setattr("stdapi.media._ENCODE_CHUNK_TIMEOUT", 0.5)
        monkeypatch.setattr(
            "stdapi.media._ffmpeg_args",
            lambda *_args: ["sh", "-c", "echo oops >&2; sleep 60"],
        )
        with pytest.raises(ApiError):
            async for _chunk in encode_audio_stream(_feed(b"audio"), "flac"):
                pass
        details = "".join(map(str, request_log["error_detail"]))
        assert "produced no output" in details
        assert "Exit code: None" in details
        assert "Input: finished, stdin closed" in details
        assert "ffmpeg said: oops" in details

    async def test_a_missing_encoder_is_reported_as_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """No ffmpeg on the server is the deployment's fault, not the caller's.

        The caller reads the same 503 refusal every unavailable feature gives,
        naming the encoding it asked for; the operator reads what is missing in
        the request log, at ``warning`` -- the ``critical`` an unlabelled detail
        resolves to would page for a request nobody can act on.

        Ref: stdapi/api_errors.py:FeatureUnavailableError
             stdapi/media.py:encode_audio_stream
        """
        monkeypatch.setattr(
            "stdapi.media._ffmpeg_args", lambda *_args: ["stdapi-no-such-binary-exists"]
        )
        with pytest.raises(FeatureUnavailableError) as excinfo:
            async for _chunk in encode_audio_stream(_feed(b"audio"), "mp3"):
                pass
        assert excinfo.value.status == 503
        assert str(excinfo.value) == (
            "The 'mp3' encoding is not available on the current server. "
            "Please contact the administrator to enable it."
        )
        assert request_log["level"] == "warning"
        assert "ffmpeg is not installed" in "".join(
            map(str, request_log["error_detail"])
        )


class TestFailedEncodeIsReported:
    """An ffmpeg that exits nonzero fails the request instead of returning 200.

    ffmpeg closing stdout ends the streaming loop whether the encode succeeded
    or died on a bad input; only the exit code tells them apart. Without the
    check the client would receive a truncated or empty body with an OK status.

    Ref: https://ffmpeg.org/ffmpeg.html
         stdapi/media.py:encode_audio_stream
    """

    async def test_nonzero_exit_raises_and_logs_stderr(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A nonzero exit after EOF raises a 500 and logs the stderr head.

        Bytes already streamed cannot be unsent; raising still aborts the
        chunked response so the client sees a failure, not a short success.
        """
        monkeypatch.setattr(
            "stdapi.media._ffmpeg_args",
            lambda *_args: [
                "sh",
                "-c",
                "cat >/dev/null; printf partial; echo bad input >&2; exit 3",
            ],
        )
        streamed = bytearray()

        async def _consume() -> None:
            async for chunk in encode_audio_stream(_feed(b"audio"), "mp3"):
                streamed.extend(chunk)

        with pytest.raises(ApiError) as excinfo:
            await _consume()
        assert excinfo.value.status == 500
        assert "mp3" in str(excinfo.value)
        assert bytes(streamed) == b"partial", "bytes before the failure streamed"
        details = "".join(map(str, request_log["error_detail"]))
        assert "exited with code 3" in details
        assert "ffmpeg said: bad input" in details

    async def test_a_broken_input_stream_is_named_in_the_failure_log(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A source that dies mid-upload is reported as the cause, not as ffmpeg's fault.

        The encode fails either way; only the input task's state says whether
        the payload stopped arriving or the encoder rejected it, and the two
        need opposite fixes. The caller still gets the mapped ``ApiError``: the
        cleanup reaps the failed input task, and re-raising its exception there
        would replace the answer with an unmapped 500.
        """
        monkeypatch.setattr(
            "stdapi.media._ffmpeg_args",
            lambda *_args: ["sh", "-c", "cat >/dev/null; exit 3"],
        )

        async def _failing_source() -> AsyncGenerator[bytes]:
            yield b"audio"
            msg = "upload aborted"
            raise ValueError(msg)

        with pytest.raises(ApiError) as excinfo:
            async for _chunk in encode_audio_stream(_failing_source(), "mp3"):
                pass

        assert excinfo.value.status == 500
        details = "".join(map(str, request_log["error_detail"]))
        assert "exited with code 3" in details
        assert "Input: failed with ValueError('upload aborted')" in details

    async def test_a_zero_exit_is_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A clean exit streams the full body and logs nothing."""
        monkeypatch.setattr(
            "stdapi.media._ffmpeg_args",
            lambda *_args: ["sh", "-c", "cat >/dev/null; echo -n done"],
        )
        chunks = [chunk async for chunk in encode_audio_stream(_feed(b"audio"), "mp3")]
        assert b"".join(chunks) == b"done"
        assert "error_detail" not in request_log
