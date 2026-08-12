"""Media-related utilities."""

from asyncio import CancelledError, create_subprocess_exec, create_task, wait_for
from contextlib import suppress
from subprocess import PIPE
from typing import TYPE_CHECKING

from stdapi.api_errors import ApiError
from stdapi.monitoring import log_error_details

if TYPE_CHECKING:
    from asyncio import Task
    from asyncio.streams import StreamReader, StreamWriter
    from collections.abc import AsyncGenerator

#: Format aliases for ffmpeg (only when output format differs from requested format)
_FFMPEG_FORMAT_ALIASES = {"aac": "adts", "pcm": "s16le", "vorbis": "ogg"}

#: Streaming chunk size (64KB optimal for network streaming with encoding)
_CHUNK_SIZE = 65536

#: Seconds ffmpeg may go without producing output before the encode is abandoned.
_ENCODE_CHUNK_TIMEOUT = 120

#: Seconds a killed ffmpeg is given to be reaped.
_PROCESS_EXIT_TIMEOUT = 10

#: Bytes of ffmpeg stderr kept for diagnostics; the head, where ffmpeg errors first.
_STDERR_KEPT = 4096


async def _drain(stderr: StreamReader | None, seen: bytearray) -> None:
    """Consume ffmpeg's stderr so its pipe buffer can never fill.

    Args:
        stderr: The process's stderr stream.
        seen: Buffer receiving the first :data:`_STDERR_KEPT` bytes, for diagnostics.
    """
    if stderr is None:  # pragma: no cover
        return
    while chunk := await stderr.read(_CHUNK_SIZE):
        if len(seen) < _STDERR_KEPT:
            seen.extend(chunk[: _STDERR_KEPT - len(seen)])


def _input_state(task: Task[None]) -> str:
    """Describe what became of the task feeding ffmpeg's stdin.

    A stalled encode is either ffmpeg's fault or its input's, and those need
    opposite fixes; this says which without a second reproduction.

    Args:
        task: The task running :func:`_process_input_stream`.

    Returns:
        A short phrase naming the task's state, and its error if it had one.
    """
    if not task.done():
        return "still feeding"
    if task.cancelled():  # pragma: no cover
        return "cancelled"
    if (error := task.exception()) is not None:
        return f"failed with {error!r}"
    return "finished, stdin closed"


def _decode_stderr(seen: bytearray) -> str:
    """Return ffmpeg's captured stderr as a loggable suffix.

    Args:
        seen: Stderr head captured so far, even while the drain is still running.

    Returns:
        A prefixed one-line suffix, or an empty string when nothing was captured.
    """
    if captured := bytes(seen).decode("utf-8", "replace").strip():
        return f" ffmpeg said: {captured}"
    return ""


async def _process_input_stream(
    stream: AsyncGenerator[bytes], stdin: StreamWriter | None
) -> None:
    """Process input stream and feed to process.

    Args:
        stream: StreamReader from AWS Polly.
        stdin: Process stdin to feed audio to.
    """
    if stdin is not None:
        try:
            async for chunk in stream:
                stdin.write(chunk)
                await stdin.drain()
        except OSError:  # pragma: no cover
            return
        finally:
            stdin.close()
            await stream.aclose()
            with suppress(OSError):
                await stdin.wait_closed()


def _ffmpeg_args(
    output_format: str,
    input_format: str | None,
    sample_rate: int | None,
    channels: int | None,
    output_sample_rate: int | None,
    output_channels: int | None = None,
) -> list[str]:
    """Build the ffmpeg command line for one encode.

    Args:
        output_format: Target audio format.
        input_format: Input audio format, or None to let ffmpeg autodetect.
        sample_rate: Input sample rate in Hz.
        channels: Input channel count.
        output_sample_rate: Rate to resample the output to, if any.
        output_channels: Channel count to mix the output down to, if any.

    Returns:
        The full argument vector, reading stdin and writing stdout.

    Raises:
        ValueError: If raw PCM is specified without sample_rate or channels.
    """
    args = ["ffmpeg"]

    if input_format:
        # -f: input format (e.g., s16le for raw PCM)
        args.extend(("-f", _FFMPEG_FORMAT_ALIASES.get(input_format, input_format)))
        if sample_rate:
            # -ar: audio sample rate in Hz
            args.extend(("-ar", str(sample_rate)))
        elif not channels:
            msg = "sample_rate or channels must be specified for raw PCM"
            raise ValueError(msg)
        if channels:
            # -ac: audio channels (1=mono, 2=stereo)
            args.extend(("-ac", str(channels)))

    args.extend(
        (
            "-i",  # Input from stdin
            "pipe:0",
            "-q:a",  # Audio quality (0=highest)
            "0",
        )
    )
    if output_sample_rate:
        # -ar: resample the output to a specific sample rate in Hz
        args.extend(("-ar", str(output_sample_rate)))
    if output_channels:
        # -ac after the input: mix the output down to this channel count
        args.extend(("-ac", str(output_channels)))
    args.extend(
        (
            "-f",  # Output format
            _FFMPEG_FORMAT_ALIASES.get(output_format, output_format),
            "pipe:1",  # Output to stdout
        )
    )
    return args


async def encode_audio_stream(
    stream: AsyncGenerator[bytes],
    output_format: str,
    input_format: str | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    output_sample_rate: int | None = None,
    output_channels: int | None = None,
) -> AsyncGenerator[bytes]:
    """Encode audio stream using ffmpeg with highest quality settings.

    Supports both raw PCM and encoded formats (mp3, ogg, flac, etc.) as input.

    Args:
        stream: Async generator yielding audio bytes from input source.
        output_format: Target audio format (mp3, wav, flac, aac, opus, pcm, vorbis).
        input_format: Input audio format. Required for raw PCM (e.g., s16le).
            Set to None for encoded formats to enable autodetection.
        sample_rate: Sample rate in Hz (e.g., 16000, 44100, 48000).
            At least one of sample_rate or channels is required for raw PCM input.
            Optional for encoded formats.
        channels: Number of audio channels (1=mono, 2=stereo).
            At least one of sample_rate or channels is required for raw PCM input.
            Optional for encoded formats.
        output_sample_rate: Resample the output to this rate in Hz.
            Optional; leaves the source rate untouched when not set.
        output_channels: Mix the output down to this channel count (1=mono).
            Optional; leaves the source layout untouched when not set.

    Yields:
        Encoded audio bytes in the specified output format.

    Raises:
        ValueError: If raw PCM is specified without sample_rate or channels.
        ApiError: If ffmpeg is not installed on the server.
    """
    ffmpeg_args = _ffmpeg_args(
        output_format,
        input_format,
        sample_rate,
        channels,
        output_sample_rate,
        output_channels,
    )

    try:
        process = await create_subprocess_exec(
            *ffmpeg_args, stdin=PIPE, stdout=PIPE, stderr=PIPE
        )
    except FileNotFoundError as exception:
        log_error_details(
            "ffmpeg is not installed on the server. It is required for audio encoding."
        )
        msg = (
            f"The '{output_format}' encoding is not supported by the server. "
            "Please contact the administrator to enabled it."
        )
        raise ApiError(msg) from exception

    stderr_head = bytearray()
    input_task = create_task(_process_input_stream(stream, process.stdin))
    # ffmpeg writes progress and warnings to stderr for the whole encode: past
    # asyncio's stream limit the OS pipe fills, ffmpeg blocks writing to stderr,
    # and it stops consuming stdin, so the encode never completes.
    stderr_task = create_task(_drain(process.stderr, stderr_head))

    try:
        while True:
            if process.stdout:
                chunk = await wait_for(
                    process.stdout.read(_CHUNK_SIZE), _ENCODE_CHUNK_TIMEOUT
                )
                if not chunk:
                    break
                yield chunk
            else:  # pragma: no cover
                break
        # ffmpeg closed stdout; a nonzero exit means the body is incomplete.
        with suppress(TimeoutError):
            await wait_for(process.wait(), _PROCESS_EXIT_TIMEOUT)
            await wait_for(stderr_task, _PROCESS_EXIT_TIMEOUT)
        if process.returncode != 0:
            log_error_details(
                f"ffmpeg exited with code {process.returncode} after closing its "
                f"output. Command: {' '.join(ffmpeg_args)}. "
                f"Input: {_input_state(input_task)}.{_decode_stderr(stderr_head)}"
            )
            msg = f"Failed to encode the audio to '{output_format}'."
            raise ApiError(msg, status=500)
    except TimeoutError as exception:
        details = _decode_stderr(stderr_head)
        log_error_details(
            f"ffmpeg produced no output for {_ENCODE_CHUNK_TIMEOUT}s and was "
            f"terminated. Command: {' '.join(ffmpeg_args)}. "
            f"Exit code: {process.returncode}. Input: {_input_state(input_task)}."
            f"{details}"
        )
        msg = f"Timed out encoding the audio to '{output_format}'."
        raise ApiError(msg, status=504) from exception
    finally:
        # Killing first breaks the pipes, so no task can stay blocked on one.
        with suppress(ProcessLookupError):
            process.kill()
        for task in (input_task, stderr_task):
            task.cancel()
            # A task that failed on its own must not replace the error being
            # raised: the log above already reports it through _input_state().
            with suppress(CancelledError, Exception):
                await wait_for(task, _PROCESS_EXIT_TIMEOUT)
        # Reaping is bounded too: a process that ignores the kill must not strand
        # the request that spawned it.
        with suppress(TimeoutError):
            await wait_for(process.wait(), _PROCESS_EXIT_TIMEOUT)


async def stream_body(stream: StreamReader) -> AsyncGenerator[bytes]:
    """Convert a StreamReader to an async generator of bytes.

    Args:
        stream: Stream reader to consume.

    Yields:
        Raw bytes chunks read from the stream.
    """
    while True:
        chunk = await stream.read(_CHUNK_SIZE)
        if not chunk:
            break
        yield chunk
