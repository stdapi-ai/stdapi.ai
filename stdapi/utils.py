"""Common utilities."""

from asyncio import to_thread
from base64 import b32encode
from binascii import Error as BinasciiError
from contextlib import contextmanager
from io import BytesIO
from os.path import splitext
from re import ASCII, Pattern
from re import compile as compile_regex
from sys import stdout
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, TypeVar
from uuid import uuid7 as uuid

from fastapi.exceptions import RequestValidationError
from langcodes import Language
from magic import from_buffer
from PIL import Image
from pybase64 import b64decode as _b64decode
from pybase64 import b64encode as _b64encode
from pydantic import JsonValue, ValidationError
from pydantic_core import from_json, to_json

from stdapi.api_errors import ApiError

if TYPE_CHECKING:
    from collections.abc import Buffer, Generator

    from fastapi import UploadFile

T = TypeVar("T")

#: Audio MIME type regex pattern
AUDIO_MIME_PATTERN = compile_regex(r"^audio/")

#: Application inference profile ARN regex matcher
match_bedrock_app_profile_arn = compile_regex(
    "arn:aws(?:-[^:]+)?:bedrock:(?P<region>[a-z0-9-]{1,20}):[0-9]{12}:(?:inference-profile|application-inference-profile)/[a-zA-Z0-9-:.]+"
).match

#: Prompt router ARN regex matcher
match_bedrock_prompt_router_arn = compile_regex(
    "arn:aws(?:-[^:]+)?:bedrock:(?P<region>[a-z0-9-]{1,20}):[0-9]{12}:(?:prompt-router|default-prompt-router)/[a-zA-Z0-9-:.]+"
).match


@contextmanager
def validation_error_handler() -> Generator[None]:
    """Context manager to convert Pydantic ValidationError to FastAPI RequestValidationError.

    Usage:
        with validation_error_handler():
            model = MyModel(**data)

    Raises:
        RequestValidationError: Converted from ValidationError with error details preserved.
    """
    try:
        yield
    except ValidationError as error:
        raise RequestValidationError(error.errors()) from error


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


async def b64decode_data_uri(
    value: str, *, altchars: str | Buffer | None = None, validate: bool = False
) -> bytes:
    """Decode a base64 encoded data URI into bytes.

    Args:
        value: The base64 encoded URI to decode.
        altchars: Optional string or buffer containing two
            characters to replace '+' and '/' in the standard base64 alphabet.
        validate: When set to True, input will be validated to ensure it
            conforms to base64 encoding rules. Defaults to False.

    Returns:
        bytes: The decoded data in bytes.
    """
    view = memoryview(raise_if_empty(value).encode())
    try:
        return await b64decode(
            view[view.tobytes().find(b",") + 1 :], altchars=altchars, validate=validate
        )
    finally:
        view.release()


async def b64decode_data_or_uri_with_mime(
    value: str | None, *, altchars: str | Buffer | None = None, validate: bool = False
) -> tuple[bytes, str]:
    """Decodes a Base64 encoded string or a data URI with its MIME type.

    This function handles decoding of standard Base64 encoded strings as well as
    data URIs that include MIME type information. If the input is a data URI, the
    MIME type is extracted and returned along with the decoded data. Otherwise,
    the MIME type is derived from the decoded data.

    Args:
        value (str): The Base64 encoded string or data URI to decode.
        altchars (str | Buffer | None, optional): A string or buffer containing the
            alternative characters for '+' and '/' in the Base64 alphabet. Defaults
            to None.
        validate (bool, optional): If True, checks if the Base64 input is valid
            before decoding. Defaults to False.

    Returns:
        tuple[bytes, str]: A tuple where the first element is the decoded data as
        bytes, and the second element is the MIME type of the decoded data.
    """
    value = raise_if_empty(value)
    match = _data_uri_matcher(value)
    if match and match.group(1):
        return await b64decode_data_uri(
            value, altchars=altchars, validate=validate
        ), match.group(1)
    data = raise_if_empty(await b64decode(value, altchars=altchars, validate=validate))
    return data, from_buffer(data, mime=True)


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


async def read_and_b64encode_file(file: UploadFile) -> str:
    """Read an uploaded file and encode it to base64.

    Args:
        file: The uploaded file to read and encode.

    Returns:
        Base64-encoded string representation of the file contents.
    """
    content = await file.read()
    close_task = file.close()
    try:
        return await b64encode(content)
    finally:
        await close_task


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


async def convert_image(
    content: bytes,
    output_format: Literal["jpeg", "webp", "png"],
    compression: int = 100,
) -> tuple[bytes, int, int]:
    """Convert image from one format to another asynchronously.

    Args:
        content: Image data as bytes
        output_format: Output format ("jpeg", "webp", "png")
        compression: Compression quality (0-100, where 100 is highest quality)

    Returns:
        Converted image data as bytes, width, height.
    """
    return await to_thread(
        _convert_image,
        content=content,
        output_format=output_format.upper(),  # type: ignore[arg-type]
        compression=compression,
    )


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
    with BytesIO(await b64decode(content)) as buffer, Image.open(buffer) as image:
        return image.size


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
        stdout.write(msg)
        stdout.flush()
    except ValueError as error:  # pragma: no cover
        if "closed" in str(error):
            return
        raise


def raise_if_empty[T](value: T | None) -> T:
    """Raise if value is empty else return value.

    Raises:
        ValueError: If the provided `value` is None or evaluates to False.

    Args:
        value: The value to check for emptiness. Must not be None or an empty
            value (e.g., empty list, string, etc.).

    Returns:
        The same `value` if it is not None or empty.
    """
    if not value:
        msg = "Empty data"
        raise ValueError(msg)
    return value


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


def is_data_uri(string: str) -> bool:
    """Check if a string is a data URI.

    Args:
        string: The string to check

    Returns:
        bool: True if the string is a data URI.
    """
    return _data_uri_matcher(string) is not None


def get_data_uri_type(string: str) -> str:
    """Return the media type of data URI or plain text.

    Args:
        string: The string to check

    Returns:
        Media type.
    """
    match = _data_uri_matcher(string)
    if match and match.group(1):
        return match.group(1)
    return "text/plain"


def get_data_uri_data(string: str) -> str:
    """Return the base64 data of data URI or a base64 encoded string.

    Args:
        string: The string to check

    Returns:
        Media type.
    """
    return string.split(",", 1)[-1] if is_data_uri(string) else string


def hide_security_details(status: int, message: str) -> str:
    """Hide sensitive information from client response in case of HTTP errors.

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
    return message


def guess_media_type(filepath: str) -> tuple[Literal["video", "audio", "image"], str]:
    """Guess the media type from a file path based on its extension.

    Can be useful if the file cannot be accessed directly and only the file path is known.
    Guess based on a common list of media types.

    Args:
        filepath: Path to the media file

    Returns:
        str: "video", "audio", or "image", file extension.

    Raises:
        ValueError: If the file type cannot be determined
    """
    extension = splitext(filepath)[1].lower()  # noqa: PTH122
    if not extension:
        msg = f"A file extension is required for file: {filepath}"
        raise ValueError(msg)
    if extension in {
        "mp4",
        "avi",
        "mov",
        "mkv",
        "flv",
        "wmv",
        "webm",
        "m4v",
        "mpg",
        "mpeg",
        "3gp",
        "ogv",
        "mts",
        "m2ts",
        "ts",
        "vob",
    }:
        return "video", extension
    if extension in {
        "mp3",
        "wav",
        "flac",
        "aac",
        "ogg",
        "wma",
        "m4a",
        "opus",
        "aiff",
        "ape",
        "alac",
        "pcm",
        "dsd",
        "amr",
        "au",
        "mid",
        "midi",
    }:
        return "audio", extension
    if extension in {
        "jpg",
        "jpeg",
        "png",
        "gif",
        "bmp",
        "tiff",
        "tif",
        "webp",
        "svg",
        "ico",
        "heic",
        "heif",
        "raw",
        "cr2",
        "nef",
        "arw",
        "dng",
    }:
        return "image", extension
    msg = f"Unsupported media type for file: {filepath}"
    raise ValueError(msg)


def get_base64_decoded_size(value: str) -> int:
    """Compute the decoded size of base64 data without actually decoding it.

    This function calculates the size of the decoded data by analyzing the base64
    string length and padding, avoiding the memory overhead of actual decoding.

    Args:
        value: Base64-encoded string or URI.

    Returns:
        The size in bytes of the decoded data.
    """
    prefix_length = value.find(",") if value.startswith("data:") else 0
    padding = 0
    for i in range(len(value) - 1, -1, -1):
        if value[i] == "=":
            padding += 1
        else:
            break
    return (len(value) * 3) // 4 - padding - prefix_length


def get_and_validate_mime(content: bytes, mime_pattern: Pattern[str]) -> str:
    """Validate MIME type of content against a regex pattern.

    Args:
        content: Binary content to check.
        mime_pattern: Compiled regex pattern (e.g., AUDIO_MIME_PATTERN).

    Raises:
        ApiError: If MIME type doesn't match the pattern.
    """
    mime = from_buffer(content, mime=True)
    if not mime_pattern.match(mime):
        msg = f"Unsupported input file format: {mime}"
        raise ApiError(msg)
    return mime
