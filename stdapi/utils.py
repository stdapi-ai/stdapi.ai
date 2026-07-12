"""Common utilities."""

import sys
import warnings
from asyncio import to_thread
from base64 import b32encode
from binascii import Error as BinasciiError
from contextlib import contextmanager
from datetime import UTC, datetime
from io import BytesIO
from json import JSONDecodeError
from re import ASCII
from re import compile as compile_regex
from typing import (
    TYPE_CHECKING,
    Literal,
    LiteralString,
    Never,
    NotRequired,
    Protocol,
    TypedDict,
    TypeVar,
)
from urllib.parse import unquote
from uuid import uuid7 as uuid

from fastapi.exceptions import RequestValidationError
from langcodes import Language
from PIL import Image
from pybase64 import b64decode as _b64decode
from pybase64 import b64encode as _b64encode
from pydantic import BaseModel, JsonValue, ValidationError
from pydantic_core import from_json, to_json
from sse_starlette import JSONServerSentEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Buffer, Generator

    class _AsyncReader(Protocol):
        """Protocol for objects with an async ``read(size)`` method."""

        async def read(self, size: int, /) -> bytes: ...


T = TypeVar("T")

#: Maximum decoded image size in pixels, guarding Pillow against decompression bombs.
_MAX_IMAGE_PIXELS: int = 50_000_000
# Pillow only raises above 2x its threshold, so halve it to make the cap the hard limit.
Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS // 2
# Silence the warning Pillow emits in the [cap/2, cap] band; those images are allowed.
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


#: Application inference profile ARN regex matcher (end-anchored to reject trailing data)
match_bedrock_app_profile_arn = compile_regex(
    "arn:aws(?:-[^:]+)?:bedrock:(?P<region>[a-z0-9-]{1,20}):[0-9]{12}:(?:inference-profile|application-inference-profile)/[a-zA-Z0-9_.:-]+\\Z"
).match

#: Prompt router ARN regex matcher (end-anchored to reject trailing data)
match_bedrock_prompt_router_arn = compile_regex(
    "arn:aws(?:-[^:]+)?:bedrock:(?P<region>[a-z0-9-]{1,20}):[0-9]{12}:(?:prompt-router|default-prompt-router)/[a-zA-Z0-9_.:-]+\\Z"
).match


def json_sse(event: LiteralString | None, payload: BaseModel) -> JSONServerSentEvent:
    """Build a ``JSONServerSentEvent`` from a pydantic payload.

    Args:
        event: SSE event name (always a literal in callers), or ``None`` for a
            data-only event.
        payload: Pydantic model to serialise as the event data.

    Returns:
        A ready-to-yield ``JSONServerSentEvent``.
    """
    data = payload.model_dump(mode="json", exclude_none=True)
    return (
        JSONServerSentEvent(event=event, data=data)
        if event
        else JSONServerSentEvent(data=data)
    )


def now_utc_timestamp() -> int:
    """Return the current UTC time as a Unix timestamp (seconds).

    Returns:
        Current Unix timestamp in seconds.
    """
    return int(datetime.now(UTC).timestamp())


@contextmanager
def validation_error_handler() -> Generator[None]:
    """Context manager to convert body-parsing errors to FastAPI RequestValidationError.

    Converts a Pydantic ``ValidationError`` or a ``JSONDecodeError`` (raised by
    ``Request.json()`` on a malformed body) into a ``RequestValidationError``,
    matching how FastAPI itself reports an invalid request body.

    Yields:
        None

    Raises:
        RequestValidationError: Converted from ValidationError or JSONDecodeError,
            with error details preserved.

    Usage:
        with validation_error_handler():
            model = MyModel(**data)
    """
    try:
        yield
    except ValidationError as error:
        raise RequestValidationError(error.errors()) from error
    except JSONDecodeError as error:
        raise RequestValidationError(
            [
                {
                    "type": "json_invalid",
                    "loc": ("body", error.pos),
                    "msg": "JSON decode error",
                    "input": {},
                    "ctx": {"error": error.msg},
                }
            ]
        ) from error


def missing_file_error() -> Never:
    """Raise a validation error for a missing required ``file`` body field."""
    msg = "ValidationError"
    raise ValidationError.from_exception_data(
        msg, [{"type": "missing", "loc": ("body", "file"), "input": None, "ctx": {}}]
    )


def format_language_code(language: str) -> str:
    """Format language code to ISO-639-1 format (Like en-US).

    Args:
        language: Language code in any format.

    Returns:
        language code in ISO-639-1 format
    """
    return Language(language).maximize().simplify_script().to_tag()


def language_code_to_name(language_code: str) -> str:
    """Convert language code to language name.

    Args:
        language_code: Language code in ISO-639-1 format.

    Returns:
        language name
    """
    return Language(language_code.split("-", 1)[0]).language_name().lower()


def try_parse_json(text: str) -> JsonValue:
    """Try to parse JSON.

    Args:
        text: Input text.

    Returns:
        JSON or initial text if parsing failed.
    """
    try:
        return from_json(text.strip().encode(), allow_partial=True)  # type: ignore[no-any-return]
    except ValueError:
        return text


async def b64decode(
    value: str | Buffer, *, altchars: str | Buffer | None = None, validate: bool = False
) -> bytes:
    """Decode a base64 encoded string or buffer into bytes using the base64 algorithm.

    Args:
        value: The base64 encoded string or buffer to decode.
        altchars: Optional string or buffer containing two
            characters to replace '+' and '/' in the standard base64 alphabet.
        validate: When set to True, input will be validated to ensure it
            conforms to base64 encoding rules. Defaults to False.

    Returns:
        bytes: The decoded data in bytes.
    """
    try:
        return await to_thread(_b64decode, value, altchars=altchars, validate=validate)
    except BinasciiError as error:
        msg = f"Invalid base64 data: {error}"
        raise ValueError(msg) from None


async def b64encode(value: Buffer, altchars: str | Buffer | None = None) -> str:
    """Encodes a given binary data into a base64 encoded string.

    This function operates asynchronously, allowing the calling code to run
    other tasks while the encoding is handled in a separate thread. The function
    takes an optional `altchars` argument to replace the default `+` and `/`
    characters used in the base64 alphabet with user-specified characters.

    Args:
        value: A buffer containing binary data that will be base64 encoded.
        altchars: An optional argument specifying alternative characters to replace
            the `+` and `/` characters in the standard base64 alphabet. Can be a string
            or another buffer. Defaults to None.

    Returns:
        The base64 encoded representation of the input binary data.
    """
    return (await to_thread(_b64encode, value, altchars=altchars)).decode()


def b64_decoded_len(value: str, prefix_len: int = 0) -> int:
    """Calculate the length of the decoded base64 string.

    Args:
        value: The base64-encoded string whose decoded length needs to be calculated.
        prefix_len: Prefix length to not count in the length calculation.

    Returns:
        The length of the decoded string.
    """
    return (
        (n := len(value) - prefix_len) * 3 // 4
        - (value[-1] == "=")
        - (value[-2] == "=" if n > 1 else 0)
    )


#: PIL image formats
_PilImageFormats = Literal["JPEG", "WEBP", "PNG"]


class _PilImageParams(TypedDict):
    """PIL image parameters."""

    format: _PilImageFormats
    optimize: bool
    compress_level: NotRequired[int]
    quality: NotRequired[int]


def _convert_image(
    content: bytes, output_format: _PilImageFormats, compression: int = 100
) -> tuple[bytes, int, int]:
    """Converts an input image to the specified format.

    Args:
        content: The binary content of the input image to be converted.
        output_format: The desired output format for the image.
        compression: The level of compression or quality for the output image.
            For "JPEG" and "WEBP", this represents the "quality" (1-100).
            For "PNG", this represents the compression level (0-9), calculated
            proportionally based on the input quality.

    Returns:
        The binary content of the converted image in the specified format
        and compression level, width, height.

    Raises:
        ValueError: If compression is not between 0 and 100
    """
    if not 0 <= compression <= 100:
        msg = f"Compression must be between 0 and 100, got {compression}"
        raise ValueError(msg)
    try:
        with (
            BytesIO() as output_buffer,
            BytesIO(content) as input_buffer,
            Image.open(input_buffer) as image,
        ):
            save_kwargs: _PilImageParams = {"format": output_format, "optimize": True}

            if output_format in ("JPEG", "WEBP"):
                save_kwargs["quality"] = compression

                if output_format == "JPEG" and image.mode in ("RGBA", "LA"):
                    # Convert transparent to blank background
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    if image.mode == "RGBA":
                        background.paste(image, mask=image.split()[-1])
                    else:
                        background.paste(image)
                    image = background  # type: ignore[assignment]

            elif output_format == "PNG":
                # PNG uses compress_level (0-9)
                save_kwargs["compress_level"] = int((100 - compression) / 100 * 9)

            image.save(output_buffer, **save_kwargs)
            width, height = image.size
            return output_buffer.getvalue(), width, height
    except Image.DecompressionBombError as error:
        msg = "Image exceeds the maximum allowed pixel size."
        raise ValueError(msg) from error


def _convert_base64_image(
    content: str | Buffer, output_format: _PilImageFormats, compression: int = 100
) -> tuple[str, int, int]:
    """Converts a base64-encoded image into a specified format.

    This function decodes the base64-encoded content, converts the image to the desired
    format, applies the specified compression level, and re-encodes it back into a base64 string.

    Args:
        content: Base64-encoded string or Buffer containing the image data to be converted.
        output_format: Desired format for the output image (e.g., JPEG, PNG).
        compression: Compression level for the output image. Defaults to 100.

    Returns:
        Base64-encoded string representing the image in the specified format, width, height.
    """
    image, width, height = _convert_image(
        _b64decode(content), output_format, compression
    )
    return _b64encode(image).decode(), width, height


async def convert_base64_image(
    content: str | Buffer,
    output_format: Literal["jpeg", "webp", "png"],
    compression: int = 100,
) -> tuple[str, int, int]:
    """Convert image encoded in Base64 from one format to another asynchronously.

    Args:
        content: Image data as bytes
        output_format: Output format ("jpeg", "webp", "png")
        compression: Compression quality (0-100, where 100 is highest quality)

    Returns:
        Converted image data as base64 string, width, height.
    """
    return await to_thread(
        _convert_base64_image,
        content=content,
        output_format=output_format.upper(),  # type: ignore[arg-type]
        compression=compression,
    )


async def get_base64_image_size(content: str | Buffer) -> tuple[int, int]:
    """Calculates the dimensions of an image from its Base64-encoded content.

    The function takes a Base64-encoded string or a Buffer containing an image,
    decodes it, and opens it as an image to retrieve its width and height.

    Args:
        content: The Base64-encoded image content as a string or Buffer.

    Returns:
        A tuple containing the width and height of the image as integers.
    """
    try:
        with BytesIO(await b64decode(content)) as buffer, Image.open(buffer) as image:
            return image.size
    except Image.DecompressionBombError as error:
        msg = "Image exceeds the maximum allowed pixel size."
        raise ValueError(msg) from error


def webuuid() -> str:
    """Generates a base32 encoded UUID string.

    Returns:
        str: The base32 encoded and formatted UUID string.
    """
    return b32encode(uuid().bytes).rstrip(b"=").lower().decode()


def stdout_write(value: JsonValue) -> None:
    """Writes a JSON-encoded value to the standard output.

    Args:
        value: The value to be JSON-encoded and written to standard output.
    """
    msg = f"{to_json(value).decode()}\n"
    try:
        sys.stdout.write(msg)
        sys.stdout.flush()
    except ValueError as error:  # pragma: no cover
        if "closed" in str(error):
            return
        raise


# data:[<mediatype>][;base64],<data>
_DATA_URI_PATTERN = compile_regex(
    r"^data:"
    r"([a-zA-Z0-9][a-zA-Z0-9\-\+\.]*\/[a-zA-Z0-9][a-zA-Z0-9\-\+\.]*)?"  # optional mediatype
    r"(?:;[a-zA-Z0-9\-]+=[^;,]+)*"  # optional parameters
    r"(?:;base64)?"  # optional base64 indicator
    r",",  # required comma, before data block
    ASCII,  # Use ASCII-only matching for speed
)
_data_uri_matcher = _DATA_URI_PATTERN.match


def get_data_uri_data(string: str) -> str:
    """Return the base64 data of data URI or a base64 encoded string.

    Args:
        string: The string to check.

    Returns:
        Raw base64 content with data URI prefix stripped if present.
    """
    return string.split(",", 1)[-1] if _data_uri_matcher(string) else string


#: Matches AWS ARNs, redacted from client-facing error messages.
_ARN_RE = compile_regex(r"arn:aws[\w:/.-]*")

#: Matches bare 12-digit AWS account IDs.
_ACCOUNT_ID_RE = compile_regex(r"\b\d{12}\b")


def hide_security_details(status: int, message: str) -> str:
    """Hide sensitive information from client response in case of HTTP errors.

    AWS error messages routinely embed ARNs and account IDs; these are redacted
    from every message regardless of status so internal identifiers are not
    disclosed to clients.

    Args:
        status: HTTP status code.
        message: Message body.

    Returns:
        Message body.
    """
    if status == 401:
        return "Unauthorized"
    if status == 403:
        return "Forbidden"
    return _ACCOUNT_ID_RE.sub("<account-id>", _ARN_RE.sub("<arn>", message))


def strip_url_query(url: str) -> str:
    """Return *url* with its query string replaced by a redaction marker.

    Used before logging or tracing user-supplied URLs so presigned URL
    signatures (``X-Amz-Signature`` and friends) are not persisted. When a
    query string is present it is replaced by ``?<redacted>`` so the censoring
    is explicit to log readers; a URL without a query is returned unchanged.

    Args:
        url: The URL to sanitise.

    Returns:
        The URL with any query string replaced by ``?<redacted>``.
    """
    base, sep, _query = url.partition("?")
    return f"{base}?<redacted>" if sep else base


#: RFC 6266 / RFC 5987 Content-Disposition filename patterns.
_CD_FILENAME_STAR_RE = compile_regex(
    r"""filename\*\s*=\s*(?:[A-Za-z0-9_-]+'')?([^;\s]+)""", ASCII
)
_CD_FILENAME_RE = compile_regex(
    r"""filename\s*=\s*"([^"\\]*)"|filename\s*=\s*([^\s;]+)"""
)


def parse_content_disposition_filename(header: str) -> str:
    """Extract a filename from a ``Content-Disposition`` header value.

    Prefers ``filename*`` (RFC 5987 extended notation) over plain ``filename``.
    URL-decodes the ``filename*`` value if present.

    Args:
        header: The raw ``Content-Disposition`` header value.

    Returns:
        The extracted filename, or an empty string if none is found.
    """
    if match := _CD_FILENAME_STAR_RE.search(header):
        return unquote(match.group(1))
    if match := _CD_FILENAME_RE.search(header):
        return match.group(1) or match.group(2) or ""
    return ""


async def buffered_chunks(
    chunks: AsyncIterator[bytes], chunk_size: int
) -> AsyncIterator[bytes]:
    """Buffer an async byte stream into chunks of at least *chunk_size*.

    Accumulates incoming bytes in a ``bytearray`` and yields a ``bytes``
    object each time the buffer reaches *chunk_size*.  Any leftover bytes
    are yielded at the end.

    Args:
        chunks: Async iterator of arbitrarily-sized byte fragments.
        chunk_size: Minimum number of bytes per yielded chunk (except
            possibly the last).

    Yields:
        ``bytes`` objects of at least *chunk_size* (last may be smaller).
    """
    buf = bytearray()
    async for chunk in chunks:
        buf += chunk
        if len(buf) >= chunk_size:
            yield bytes(buf)
            buf = bytearray()
    if buf:
        yield bytes(buf)


async def chain_async_iterators[T](*iterators: AsyncIterator[T]) -> AsyncIterator[T]:
    """Chain multiple async iterators into a single async iterator.

    Args:
        *iterators: Async iterators to chain together.

    Yields:
        Items from each iterator in order.
    """
    for it in iterators:
        async for item in it:
            yield item


async def async_iter[T](*values: T) -> AsyncIterator[T]:
    """Wrap one or more values into an async iterator.

    Args:
        *values: Values to yield.

    Yields:
        Each value in order.
    """
    for value in values:
        yield value


async def read_chunks(reader: _AsyncReader, chunk_size: int) -> AsyncIterator[bytes]:
    """Yield byte chunks from an async reader.

    Reads from *reader* by calling its ``.read(chunk_size)`` method
    until an empty ``bytes`` result is returned.

    Args:
        reader: Any object with an async ``read(size)`` method
            (e.g. ``aiohttp.StreamReader``, ``starlette.UploadFile``).
        chunk_size: Number of bytes to request per read.

    Yields:
        Non-empty ``bytes`` chunks.
    """
    while chunk := await reader.read(chunk_size):
        yield chunk
