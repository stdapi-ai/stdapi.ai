"""Media-related utilities."""

from asyncio import CancelledError, create_subprocess_exec, create_task
from collections.abc import AsyncGenerator
from contextlib import suppress
from subprocess import PIPE
from typing import TYPE_CHECKING

from stdapi.monitoring import log_error_details
from stdapi.openai_exceptions import OpenaiError

if TYPE_CHECKING:
    from asyncio.streams import StreamReader, StreamWriter

#: Format aliases for ffmpeg (only when output format differs from requested format)
_FFMPEG_FORMAT_ALIASES = {"aac": "adts", "pcm": "s16le", "vorbis": "ogg"}

#: Streaming chunk size (64KB optimal for network streaming with encoding)
_CHUNK_SIZE = 65536


async def _process_input_stream(
    stream: AsyncGenerator[bytes], stdin: "StreamWriter | None"
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
            await stdin.wait_closed()


async def encode_audio_stream(
    stream: AsyncGenerator[bytes],
    output_format: str,
    input_format: str | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> AsyncGenerator[bytes]:
    """Encode audio stream using ffmpeg with highest quality settings.

    Supports both raw PCM and encoded formats (mp3, ogg, flac, etc.) as input.
    Automatically handles format conversion and applies maximum quality encoding.

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

    Yields:
        Encoded audio bytes in the specified output format.

    Raises:
        ValueError: If raw PCM is specified without sample_rate or channels.
        OpenaiError: If ffmpeg is not installed on the server.
    """
    ffmpeg_args = ["ffmpeg"]

    # Input format specification (required for raw PCM, optional for encoded formats)
    if input_format:
        # -f: input format (e.g., s16le for raw PCM)
        ffmpeg_args.extend(
            ("-f", _FFMPEG_FORMAT_ALIASES.get(input_format, input_format))
        )
        if sample_rate:
            # -ar: audio sample rate in Hz
            ffmpeg_args.extend(("-ar", str(sample_rate)))
        elif not channels:  # pragma: no cover
            msg = "sample_rate or channels must be specified for raw PCM"
            raise ValueError(msg)
        if channels:
            # -ac: audio channels (1=mono, 2=stereo)
            ffmpeg_args.extend(("-ac", str(channels)))

    ffmpeg_args.extend(
        (
            "-i",  # Input from stdin
            "pipe:0",
            "-q:a",  # Audio quality (0=highest)
            "0",
            "-f",  # Output format
            _FFMPEG_FORMAT_ALIASES.get(output_format, output_format),
            "pipe:1",  # Output to stdout
        )
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
        raise OpenaiError(msg) from exception

    input_task = create_task(_process_input_stream(stream, process.stdin))

    try:
        while True:
            if process.stdout:
                chunk = await process.stdout.read(_CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
            else:  # pragma: no cover
                break
    finally:
        input_task.cancel()
        with suppress(CancelledError):
            await input_task
        with suppress(ProcessLookupError):
            process.terminate()
            await process.wait()


async def stream_body(stream: "StreamReader") -> AsyncGenerator[bytes]:
    """Convert Stream reader to async generator of bytes.

    Args:
        stream: Stream reader.
    """
    while True:
        chunk = await stream.read(_CHUNK_SIZE)
        if not chunk:
            break
        yield chunk
