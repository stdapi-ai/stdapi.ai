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
from json import dumps as _std_dumps
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
)
from urllib.parse import unquote
from uuid import uuid7 as uuid

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse as _JSONResponseBase
from langcodes import Language
from PIL import Image, UnidentifiedImageError
from pybase64 import b64decode as _b64decode
from pybase64 import b64encode as _b64encode
from pydantic import BaseModel, JsonValue, ValidationError
from pydantic_core import from_json, to_json
from sse_starlette import JSONServerSentEvent, ServerSentEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Buffer, Generator

    class _AsyncReader(Protocol):
        """Protocol for objects with an async ``read(size)`` method."""

        async def read(self, size: int, /) -> bytes: ...


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

#: Marketplace model endpoint ARN matcher, in the shape bedrock-runtime takes as a model ID
match_marketplace_endpoint_arn = compile_regex(
    "arn:aws(?:-[^:]+)?:sagemaker:(?P<region>[a-z0-9-]{1,20}):[0-9]{12}:endpoint/(?P<name>[a-zA-Z0-9-]+)\\Z"
).match

#: Public-hub content ARN matcher, whose last two segments name a listing and its version
match_sagemaker_hub_content_arn = compile_regex(
    "arn:aws(?:-[^:]+)?:sagemaker:[a-z0-9-]{1,20}:[^:]*:hub-content/SageMakerPublicHub/Model/(?P<name>[^/]+)/(?P<version>[^/]+)\\Z"
).match

#: Prompt Management prompt ARN matcher, with the optional version suffix Bedrock accepts as a model ID
match_bedrock_prompt_arn = compile_regex(
    "(?P<base>arn:aws(?:-[^:]+)?:bedrock:(?P<region>[a-z0-9-]{1,20}):[0-9]{12}:prompt/[0-9a-zA-Z]{10})(?::(?P<version>[0-9]{1,5}))?\\Z"
).match


def to_json_str(value: object) -> str:
    """Encode a value as a compact JSON string via pydantic_core.

    Falls back to the stdlib encoder (with its ASCII escaping) for input
    pydantic_core rejects, such as strings carrying lone surrogates.

    Args:
        value: Value to encode.

    Returns:
        Compact JSON string (no spaces after separators).
    """
    try:
        return to_json(value).decode()
    except ValueError:
        return _std_dumps(value, separators=(",", ":"))


def to_json_bytes(value: object) -> bytes:
    """Encode a value as compact UTF-8 JSON bytes via pydantic_core.

    Falls back to the stdlib encoder (with its ASCII escaping) for input
    pydantic_core rejects, such as strings carrying lone surrogates.

    Args:
        value: Value to encode.

    Returns:
        Compact JSON bytes (no spaces after separators).
    """
    try:
        return to_json(value)
    except ValueError:
        return _std_dumps(value, separators=(",", ":")).encode()


class _PreEncodedJSONServerSentEvent(JSONServerSentEvent):
    """``JSONServerSentEvent`` fed pre-serialized JSON, skipping the re-encode."""

    def __init__(self, data: str, *, event: str | None = None) -> None:
        """Store the already-encoded JSON payload as the event data.

        Args:
            data: JSON-encoded event data.
            event: SSE event name, or ``None`` for a data-only event.
        """
        ServerSentEvent.__init__(self, data, event=event)


def json_sse(event: LiteralString | None, payload: BaseModel) -> JSONServerSentEvent:
    """Build a ``JSONServerSentEvent`` from a pydantic payload.

    The payload is encoded in a single pydantic_core pass; the wire format is
    identical to ``JSONServerSentEvent``'s stdlib encoding (compact separators,
    raw UTF-8).

    Args:
        event: SSE event name (always a literal in callers), or ``None`` for a
            data-only event.
        payload: Pydantic model to serialise as the event data.

    Returns:
        A ready-to-yield ``JSONServerSentEvent``.
    """
    return _PreEncodedJSONServerSentEvent(
        payload.model_dump_json(exclude_none=True), event=event
    )


class JSONResponse(_JSONResponseBase):
    """``JSONResponse`` rendered with pydantic_core instead of the stdlib encoder.

    Emits the same compact, raw-UTF-8 JSON as the FastAPI default, falling back
    to an escaping encoder for input pydantic_core rejects (such as strings
    carrying lone surrogates).
    """

    def render(self, content: object) -> bytes:
        """Render the response body as compact JSON bytes.

        Args:
            content: JSON-serializable response content.

        Returns:
            UTF-8 encoded JSON body.
        """
        try:
            return to_json(content)
        except ValueError:
            # A lone surrogate has no UTF-8 encoding, so the fallback has to
            # escape it: FastAPI's own renderer would fail on the very input
            # this branch exists for.
            return _std_dumps(content, ensure_ascii=True, separators=(",", ":")).encode(
                "ascii"
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


#: ISO-639-1 base language codes mapped to lowercase English language names.
_LANGUAGE_NAMES: dict[str, str] = {
    "aa": "afar",
    "ab": "abkhazian",
    "ae": "avestan",
    "af": "afrikaans",
    "ak": "akan",
    "am": "amharic",
    "an": "aragonese",
    "ar": "arabic",
    "as": "assamese",
    "av": "avaric",
    "ay": "aymara",
    "az": "azerbaijani",
    "ba": "bashkir",
    "be": "belarusian",
    "bg": "bulgarian",
    "bh": "bihari languages",
    "bi": "bislama",
    "bm": "bambara",
    "bn": "bangla",
    "bo": "tibetan",
    "br": "breton",
    "bs": "bosnian",
    "ca": "catalan",
    "ce": "chechen",
    "ch": "chamorro",
    "co": "corsican",
    "cr": "cree",
    "cs": "czech",
    "cu": "church slavic",
    "cv": "chuvash",
    "cy": "welsh",
    "da": "danish",
    "de": "german",
    "dv": "divehi",
    "dz": "dzongkha",
    "ee": "ewe",
    "el": "greek",
    "en": "english",
    "eo": "esperanto",
    "es": "spanish",
    "et": "estonian",
    "eu": "basque",
    "fa": "persian",
    "ff": "fula",
    "fi": "finnish",
    "fj": "fijian",
    "fo": "faroese",
    "fr": "french",
    "fy": "western frisian",
    "ga": "irish",
    "gd": "scottish gaelic",
    "gl": "galician",
    "gn": "guarani",
    "gu": "gujarati",
    "gv": "manx",
    "ha": "hausa",
    "he": "hebrew",
    "hi": "hindi",
    "ho": "hiri motu",
    "hr": "croatian",
    "ht": "haitian creole",
    "hu": "hungarian",
    "hy": "armenian",
    "hz": "herero",
    "ia": "interlingua",
    "id": "indonesian",
    "ie": "interlingue",
    "ig": "igbo",
    "ii": "sichuan yi",
    "ik": "inupiaq",
    "in": "indonesian",
    "io": "ido",
    "is": "icelandic",
    "it": "italian",
    "iu": "inuktitut",
    "iw": "hebrew",
    "ja": "japanese",
    "ji": "yiddish",
    "jv": "javanese",
    "jw": "javanese",
    "ka": "georgian",
    "kg": "kongo",
    "ki": "kikuyu",
    "kj": "kuanyama",
    "kk": "kazakh",
    "kl": "kalaallisut",
    "km": "khmer",
    "kn": "kannada",
    "ko": "korean",
    "kr": "kanuri",
    "ks": "kashmiri",
    "ku": "kurdish",
    "kv": "komi",
    "kw": "cornish",
    "ky": "kyrgyz",
    "la": "latin",
    "lb": "luxembourgish",
    "lg": "ganda",
    "li": "limburgish",
    "ln": "lingala",
    "lo": "lao",
    "lt": "lithuanian",
    "lu": "luba-katanga",
    "lv": "latvian",
    "mg": "malagasy",
    "mh": "marshallese",
    "mi": "māori",
    "mk": "macedonian",
    "ml": "malayalam",
    "mn": "mongolian",
    "mo": "moldavian",
    "mr": "marathi",
    "ms": "malay",
    "mt": "maltese",
    "my": "burmese",
    "na": "nauru",
    "nb": "norwegian bokmål",
    "nd": "north ndebele",
    "ne": "nepali",
    "ng": "ndonga",
    "nl": "dutch",
    "nn": "norwegian nynorsk",
    "no": "norwegian",
    "nr": "south ndebele",
    "nv": "navajo",
    "ny": "nyanja",
    "oc": "occitan",
    "oj": "ojibwa",
    "om": "oromo",
    "or": "odia",
    "os": "ossetic",
    "pa": "punjabi",
    "pi": "pali",
    "pl": "polish",
    "ps": "pashto",
    "pt": "portuguese",
    "qu": "quechua",
    "rm": "romansh",
    "rn": "rundi",
    "ro": "romanian",
    "ru": "russian",
    "rw": "kinyarwanda",
    "sa": "sanskrit",
    "sc": "sardinian",
    "sd": "sindhi",
    "se": "northern sami",
    "sg": "sango",
    "sh": "serbo-croatian",
    "si": "sinhala",
    "sk": "slovak",
    "sl": "slovenian",
    "sm": "samoan",
    "sn": "shona",
    "so": "somali",
    "sq": "albanian",
    "sr": "serbian",
    "ss": "swati",
    "st": "southern sotho",
    "su": "sundanese",
    "sv": "swedish",
    "sw": "swahili",
    "ta": "tamil",
    "te": "telugu",
    "tg": "tajik",
    "th": "thai",
    "ti": "tigrinya",
    "tk": "turkmen",
    "tl": "tagalog",
    "tn": "tswana",
    "to": "tongan",
    "tr": "turkish",
    "ts": "tsonga",
    "tt": "tatar",
    "tw": "twi",
    "ty": "tahitian",
    "ug": "uyghur",
    "uk": "ukrainian",
    "ur": "urdu",
    "uz": "uzbek",
    "ve": "venda",
    "vi": "vietnamese",
    "vo": "volapük",
    "wa": "walloon",
    "wo": "wolof",
    "xh": "xhosa",
    "yi": "yiddish",
    "yo": "yoruba",
    "za": "zhuang",
    "zh": "chinese",
    "zu": "zulu",
}


def language_code_to_name(language_code: str) -> str:
    """Convert language code to language name.

    Args:
        language_code: Language code in ISO-639-1 format.

    Returns:
        language name
    """
    code = language_code.split("-", 1)[0].lower()
    return _LANGUAGE_NAMES.get(code, f"unknown language [{code}]")


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


#: Size below which base64 codec calls run inline instead of hopping to a thread.
_B64_INLINE_MAX_BYTES = 256 * 1024


async def b64decode(
    value: str | Buffer, *, altchars: str | Buffer | None = None, validate: bool = False
) -> bytes:
    """Decode a base64 encoded string or buffer into bytes using the base64 algorithm.

    Small inputs are decoded inline: the executor round-trip costs more than
    the decode itself below ``_B64_INLINE_MAX_BYTES``.

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
        if len(value) <= _B64_INLINE_MAX_BYTES:  # type: ignore[arg-type]
            return _b64decode(value, altchars=altchars, validate=validate)
        return await to_thread(_b64decode, value, altchars=altchars, validate=validate)
    except BinasciiError as error:
        msg = f"Invalid base64 data: {error}"
        raise ValueError(msg) from None


async def b64encode(value: Buffer, altchars: str | Buffer | None = None) -> str:
    """Encodes a given binary data into a base64 encoded string.

    Small inputs are encoded inline: the executor round-trip costs more than
    the encode itself below ``_B64_INLINE_MAX_BYTES``.

    Args:
        value: A buffer containing binary data that will be base64 encoded.
        altchars: An optional argument specifying alternative characters to replace
            the `+` and `/` characters in the standard base64 alphabet. Can be a string
            or another buffer. Defaults to None.

    Returns:
        The base64 encoded representation of the input binary data.
    """
    if len(value) <= _B64_INLINE_MAX_BYTES:  # type: ignore[arg-type]
        return _b64encode(value, altchars=altchars).decode()
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


def b64_encoded_len(size: int) -> int:
    """Calculate the length of the base64 encoding of *size* bytes.

    Args:
        size: Length in bytes of the content to encode.

    Returns:
        The length of the padded base64 representation.
    """
    return (size + 2) // 3 * 4


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


#: Alpha threshold below which a pixel is treated as the OpenAI-style "edit" region.
_MASK_ALPHA_THRESHOLD = 127


def _alpha_mask_to_bw(content: bytes, threshold: int, *, invert: bool) -> bytes:
    """Convert an OpenAI-style alpha-transparency mask to a black/white RGB mask.

    Nova Canvas/Titan require a black-and-white ``maskImage`` (no alpha channel).
    For inpainting, pure black marks the region to edit and pure white the
    region to preserve; outpainting uses the opposite polarity. OpenAI-style
    masks use a transparent (low-alpha) pixel to mark the region to edit,
    whether stored as RGBA/LA/PA or as a palette ("P") image with a ``tRNS``
    transparency chunk (e.g. pngquant/"PNG-8" exports). Masks with no alpha
    information are assumed to already use the target task's black/white format.

    Args:
        content: Decoded mask image bytes.
        threshold: Alpha value below which a pixel is treated as the edit region.
        invert: If True, render the edit region white and the preserve region
            black (outpainting polarity) instead of the inpainting default.

    Returns:
        PNG-encoded black/white RGB mask bytes, or *content* unchanged if the
        mask has no alpha channel or is not a decodable image (downstream
        validation then rejects it).
    """
    try:
        image = Image.open(BytesIO(content))
    except UnidentifiedImageError, Image.DecompressionBombError:
        return content
    with image:
        has_palette_alpha = image.mode == "P" and "transparency" in image.info
        if image.mode not in ("RGBA", "LA", "PA") and not has_palette_alpha:
            return content
        edit_color, preserve_color = (255, 0) if invert else (0, 255)
        alpha = image.convert("RGBA").split()[-1]
        bw = alpha.point(
            lambda value: edit_color if value < threshold else preserve_color
        )
        with BytesIO() as output_buffer:
            bw.convert("RGB").save(output_buffer, format="PNG")
            return output_buffer.getvalue()


async def alpha_mask_to_bw(
    content: str, threshold: int = _MASK_ALPHA_THRESHOLD, *, invert: bool = False
) -> str:
    """Convert a base64-encoded alpha-transparency mask to a black/white RGB mask.

    A no-op passthrough for masks that have no alpha channel.

    Args:
        content: Base64-encoded mask image.
        threshold: Alpha value below which a pixel is treated as the edit region.
        invert: If True, render the edit region white and the preserve region
            black — the outpainting mask polarity, opposite of inpainting's.

    Returns:
        Base64-encoded black/white PNG mask, or *content* unchanged if the
        mask has no alpha channel.
    """
    decoded = await b64decode(content)
    converted = await to_thread(_alpha_mask_to_bw, decoded, threshold, invert=invert)
    return content if converted == decoded else await b64encode(converted)


async def get_base64_image_size(content: str | Buffer) -> tuple[int, int]:
    """Calculates the dimensions of an image from its Base64-encoded content.

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

    Any ``BaseModel`` nested in *value* is serialized natively by
    ``pydantic_core``, dropping its ``None`` fields and using field names
    (not aliases) -- matching ``model_dump(mode="json", exclude_none=True)``.

    Args:
        value: The value to be JSON-encoded and written to standard output.
    """
    msg = f"{to_json(value, exclude_none=True, by_alias=False).decode()}\n"
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


def hide_security_details(status: int, message: str, *, disclosed: bool = False) -> str:
    """Hide sensitive information from client response in case of HTTP errors.

    AWS error messages routinely embed ARNs and account IDs; these are redacted
    from every message regardless of status so internal identifiers are not
    disclosed to clients.

    Args:
        status: HTTP status code.
        message: Message body.
        disclosed: Whether *message* is a fixed text written for the client,
            which a refusal status then keeps instead of flattening it. The
            redaction below still runs over it.

    Returns:
        Message body.
    """
    if not disclosed:
        if status == 401:
            return "Unauthorized"
        if status == 403:
            return "Forbidden"
    return _ACCOUNT_ID_RE.sub("<account-id>", _ARN_RE.sub("<arn>", message))


def strip_url_query(url: str) -> str:
    """Return *url* with its query string replaced by a redaction marker.

    Used before logging or tracing user-supplied URLs so presigned URL
    signatures (``X-Amz-Signature`` and friends) are not persisted; the marker
    keeps the censoring explicit to log readers.

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

    Args:
        reader: Any object with an async ``read(size)`` method
            (e.g. ``aiohttp.StreamReader``, ``starlette.UploadFile``).
        chunk_size: Number of bytes to request per read.

    Yields:
        Non-empty ``bytes`` chunks.
    """
    while chunk := await reader.read(chunk_size):
        yield chunk
