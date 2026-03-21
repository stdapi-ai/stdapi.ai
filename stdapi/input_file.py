"""Unified input file handling for all source types."""

from abc import ABC, abstractmethod
from asyncio import TaskGroup
from contextvars import ContextVar
from enum import IntEnum
from re import IGNORECASE
from re import compile as compile_regex
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Literal, Self
from urllib.parse import urlparse

from aiohttp import ClientError as AIOHTTPClientError
from aiohttp import ClientSession
from magic import from_buffer, from_file
from pydantic_core.core_schema import (
    no_info_plain_validator_function,
    plain_serializer_function_ser_schema,
    str_schema,
)
from starlette.datastructures import UploadFile

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.aws_bedrock import (
    MIME_TYPES_TO_AUDIO_TYPE,
    MIME_TYPES_TO_DOCUMENT_TYPE,
    MIME_TYPES_TO_VIDEO_TYPE,
)
from stdapi.aws_s3 import (
    BUCKET_TO_REGION,
    UPLOAD_CHUNK_SIZE,
    S3Object,
    copy_s3_object,
    get_bytes_from_s3,
    put_s3_object,
)
from stdapi.config import DOWNLOAD_TIMEOUT, SETTINGS
from stdapi.security import validate_url_ssrf
from stdapi.server import HTTP_CLIENT_HEADERS
from stdapi.utils import (
    b64_decoded_len,
    b64decode,
    b64encode,
    parse_content_disposition_filename,
    read_chunks,
)

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import CoreSchema
    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime.type_defs import (
        AudioBlockTypeDef,
        ContentBlockTypeDef,
        DocumentBlockTypeDef,
        DocumentSourceTypeDef,
        ImageBlockTypeDef,
        ImageSourceTypeDef,
        VideoBlockTypeDef,
    )

    #: Media type literal for Bedrock content blocks.
    _BedrockMediaType = Literal["image", "document", "video", "audio"]

#: Truncation length when representing base64 content in logs / repr.
_B64_REPR_LIMIT: int = 24

#: Number of bytes to read for magic-based MIME detection.
_MAGIC_PREFIX_SIZE: int = 8192


def _magic_detect(data: bytes) -> str:
    """Detect the MIME type of *data*.

    Prefers ``magic_buffer`` for performance.  When it returns the generic
    ``application/octet-stream``, ``libmagic`` may have silently failed to
    identify the content via its in-memory code path; the result is verified
    by writing *data* to a temporary file and calling ``magic_file`` instead,
    which uses a different internal path and is more reliable.  Both code
    paths return the same value for genuinely unrecognised content.

    Args:
        data: Raw bytes to identify.

    Returns:
        MIME type string (e.g. ``"audio/mpeg"``).
    """
    if (mime := from_buffer(data, mime=True)) != "application/octet-stream":
        return mime
    with NamedTemporaryFile() as tmp:
        tmp.write(data)
        tmp.flush()
        return from_file(tmp.name, mime=True)


#: Number of base64 characters needed for magic-based MIME detection.
_MAGIC_PREFIX_SIZE_BASE64 = ((_MAGIC_PREFIX_SIZE + 2) // 3) * 4

#: S3 virtual-hosted style URL pattern.
_S3_VIRTUAL_HOST_RE = compile_regex(
    r"^https?://"
    r"(?P<bucket>[a-z0-9](?:[a-z0-9.-]{1,61}[a-z0-9])?)\."
    r"(?:"
    r"s3(?:[.-](?P<region>[a-z0-9-]+))?"
    r"|s3\.dualstack\.(?P<region_dual>[a-z0-9-]+)"
    r"|s3-accelerate(?:\.dualstack)?"
    r")"
    r"\.amazonaws\.com/"
    r"(?P<key>[^?#]+)",
    IGNORECASE,
).match

#: S3 path-style URL pattern.
_S3_PATH_STYLE_RE = compile_regex(
    r"^https?://"
    r"(?:"
    r"s3(?:[.-](?P<region>[a-z0-9-]+))?"
    r"|s3\.dualstack\.(?P<region_dual>[a-z0-9-]+)"
    r")"
    r"\.amazonaws\.com/"
    r"(?P<bucket>[^/?#]+)/"
    r"(?P<key>[^?#]+)",
    IGNORECASE,
).match

#: S3 URI pattern.
_S3_URI_RE = compile_regex(
    r"^s3://"
    r"(?P<bucket>[^/]+)/"
    r"(?P<key>.+)"
    r"$",
    IGNORECASE,
).match

#: HTTP URI pattern.
_HTTP_URI_RE = compile_regex(r"^https?://", IGNORECASE).match

#: Data-URI header pattern (captures content type).
_DATA_URI_RE = compile_regex(
    r"^data:([a-zA-Z0-9][a-zA-Z0-9\-+.]*/"
    r"[a-zA-Z0-9][a-zA-Z0-9\-+.]*)"
    r"(?:;[a-zA-Z0-9\-]+=[^;,]+)*"
    r"(?:;base64)?,"
).match

#: JSON-schema pattern for URL-only InputFile variants
_URL_ONLY_SCHEMA_PATTERN: str = r"^(?:https?://|s3://|data:)"

#: Accepted S3 buckets
_ACCEPTED_BUCKETS: frozenset[str] = frozenset(
    bucket
    for bucket in (
        SETTINGS.aws_s3_bucket,
        *SETTINGS.aws_s3_regional_buckets.values(),
        *SETTINGS.aws_s3_accepted_buckets,
    )
    if bucket
)

#: Document formats accepted by Bedrock Converse.
_BEDROCK_DOCUMENT_FORMATS: frozenset[str] = frozenset(
    {"csv", "doc", "docx", "html", "md", "pdf", "txt", "xls", "xlsx"}
)

#: Regex to sanitize document names for Bedrock (only [a-zA-Z0-9_\- ] allowed).
_BEDROCK_DOC_NAME_RE = compile_regex(r"[^a-zA-Z0-9_\- ]+")

#: Tracks InputFile instances created during the current request context.
_CURRENT_INPUT_FILES: ContextVar[list[InputFile]] = ContextVar("_current_input_files")


def _track_current_input_files(file: InputFile) -> None:
    """Record a newly created InputFiles object for later cleanup."""
    if (files := _CURRENT_INPUT_FILES.get(None)) is None:
        _CURRENT_INPUT_FILES.set(files := [])
    files.append(file)


class _FileOrigin(IntEnum):
    """How the file was originally provided."""

    BASE64 = 0
    DATA_URI = 1
    HTTP_URL = 2
    S3_URI = 3
    UPLOAD = 4


# URL only origins
_URL_ONLY_ORIGINS: frozenset[_FileOrigin] = frozenset(
    {_FileOrigin.HTTP_URL, _FileOrigin.S3_URI, _FileOrigin.DATA_URI}
)


class _FileSource(ABC):
    """Abstract base class for file source backends."""

    __slots__ = ("_content_type", "_filename", "_region", "_repr", "_size")

    _filename: str | None
    _content_type: str
    _size: int
    _region: RegionName
    _repr: str

    @abstractmethod
    async def _resolve_metadata(self) -> None:
        """Populate *meta* with content type and size minimizing data handling."""

    @abstractmethod
    async def _read(self) -> bytes:
        """Return the full file content as bytes.

        Returns:
            The complete file bytes.
        """

    async def get_content_type(self) -> str:
        """Return the content type.

        Returns:
            The content type string.
        """
        if not hasattr(self, "_content_type"):
            await self._resolve_metadata()
        return self._content_type

    async def get_size(self) -> int:
        """Return the file size.

        Returns:
            The file size in bytes.
        """
        if not hasattr(self, "_size"):
            await self._resolve_metadata()
        return self._size

    async def get_filename(self) -> str | None:
        """Return a filename derived from the source.

        Returns:
            A filename string.
        """
        if not hasattr(self, "_filename"):
            await self._resolve_metadata()
        return self._filename

    def region(self) -> RegionName | None:
        """The S3 region, if known.

        Returns:
            S3 region or None.
        """
        try:
            return self._region
        except AttributeError:
            return None

    def is_s3(self) -> bool:
        """Returns if the file comes from S3.

        Returns:
            True if a S3 file.
        """
        return hasattr(self, "_region")

    def __repr__(self) -> str:
        """String representation of the object.

        Returns:
            string.
        """
        return self._repr

    async def to_bytes(self) -> bytes:
        """Return the full file content as bytes.

        This is a **terminal method** — calling it consumes the source.

        Returns:
            The complete file content.
        """
        data = await self._read()
        self._metadata_from_bytes(data)
        return data

    async def to_base64(self) -> str:
        """Return file content as a base64-encoded string.

        This is a **terminal method** — calling it consumes the source.

        Returns:
            The base64-encoded file content.
        """
        return await b64encode(await self.to_bytes())

    async def to_data_uri(self) -> str:
        """Return file content as a ``data:`` URI with content type.

        This is a **terminal method** — calling it consumes the source.

        Returns:
            A data URI string like ``data:image/png;base64,...``.
        """
        return f"data:{await self.get_content_type() or 'application/octet-stream'};base64,{await self.to_base64()}"

    async def to_s3(
        self, region: RegionName, *, bucket: str | None = None, key: str | None = None
    ) -> S3Object:
        """Ensure the file is on S3 in *region* and return an ``S3Object``.

        This is a **terminal method** — calling it consumes the source.

        Args:
            region: Target AWS region.
            bucket: Explicit destination bucket.  When ``None``, the
                regional bucket from settings is used.
            key: Explicit destination object key.  When ``None``, a
                temporary key is auto-generated.

        Returns:
            An ``S3Object`` pointing to the S3 object.
        """
        return await put_s3_object(
            await self.to_bytes(),
            await self.get_content_type(),
            bucket=bucket,
            key=key,
            region=region,
            temporary=True,
        )

    def _metadata_from_bytes(self, data: bytes) -> None:
        """Processes metadata information derived from a byte stream.

        Args:
            data: The raw byte data used to calculate metadata properties.
        """
        self._size = len(data)
        if not hasattr(self, "_content_type"):
            self._content_type = _magic_detect(data)


class _S3Source(_FileSource):
    """Source backend for ``s3://`` URIs."""

    __slots__ = ("_bucket", "_key", "_uri")

    _uri: str
    _bucket: str
    _key: str

    def __init__(self, uri: str, bucket: str, key: str) -> None:
        """Initialize with the S3 file.

        Args:
            uri: S3 URI.
            bucket: S3 bucket name.
            key: S3 object key.
        """
        self._bucket = bucket
        self._key = key
        self._region = BUCKET_TO_REGION[bucket]
        self._uri = self._repr = uri

    async def _resolve_metadata(self) -> None:
        """Resolve content type and size via S3 ``HeadObject``.

        Raises:
            ApiError: When the S3 object cannot be found or accessed.
        """
        head = await get_client("s3", self._region).head_object(
            Bucket=self._bucket, Key=self._key
        )
        self._size = head["ContentLength"]
        self._content_type = head["ContentType"]
        self._filename = (
            filename
            if (
                filename := parse_content_disposition_filename(
                    head.get("ContentDisposition", "")
                )
            )
            else self._key.rsplit("/", 1)[-1] or None
        )

    async def _read(self) -> bytes:
        """Download the S3 object body.

        Returns:
            The complete file bytes.
        """
        return await get_bytes_from_s3(self._bucket, self._key)

    async def to_s3(
        self, region: RegionName, *, bucket: str | None = None, key: str | None = None
    ) -> S3Object:
        """Ensure the file is on S3 in *region* and return an ``S3Object``.

        - S3 files already in *region* are returned as-is.
        - S3 files in another region are copied via server-side copy.

        This is a **terminal method** — calling it consumes the source.

        Args:
            region: Target AWS region.
            bucket: Explicit destination bucket.  When ``None``, the
                regional bucket from settings is used.
            key: Explicit destination object key.  When ``None``, a
                temporary key is auto-generated.

        Returns:
            An ``S3Object`` pointing to the S3 object.

        Raises:
            ApiError: When the file cannot be uploaded or copied.
        """
        if self._region == region and bucket is None and key is None:
            return S3Object(bucket=self._bucket, key=self._key)
        return await copy_s3_object(
            self._bucket,
            self._key,
            dest_bucket=bucket,
            dest_key=key,
            dest_region=region,
            temporary=True,
        )


class _HttpSource(_FileSource):
    """Source backend for ``http(s)://`` URLs."""

    __slots__ = ("_ssrf_validated", "_url")

    def __init__(self, url: str) -> None:
        """Initialise with the target URL.

        Args:
            url: The HTTP(S) URL to fetch.
        """
        self._repr = self._url = url
        self._ssrf_validated = False

    async def _resolve_metadata(self) -> None:
        """Resolve content type and size via HTTP ``HEAD`` request.

        Falls back to a partial-read probe using ``python-magic`` when the
        server does not supply a ``Content-Type`` header.

        Raises:
            ApiError: When the HTTP request fails.
        """
        await self._validate_ssrf()
        async with ClientSession(
            headers=HTTP_CLIENT_HEADERS, timeout=DOWNLOAD_TIMEOUT
        ) as session:
            try:
                async with session.head(self._url) as resp:
                    resp.raise_for_status()
                    if content_type := resp.headers.get("Content-Type"):
                        self._content_type = content_type.split(";", 1)[0].strip()
                    self._size = (
                        int(content_length)
                        if (content_length := resp.headers.get("Content-Length"))
                        else 0
                    )
                    self._filename = (
                        filename
                        if (
                            filename := parse_content_disposition_filename(
                                resp.headers.get("Content-Disposition", "")
                            )
                        )
                        else (urlparse(self._url).path.rsplit("/", 1)[-1] or None)
                    )
            except AIOHTTPClientError as error:
                msg = f"Error downloading {self._url}: {error}"
                raise ApiError(msg) from error

        if not hasattr(self, "_content_type"):
            await self._content_type_from_partial()

    async def _content_type_from_partial(self) -> None:
        """Download the first bytes to detect content type via magic.

        Raises:
            ApiError: When the HTTP range request fails.
        """
        async with ClientSession(
            headers={
                **HTTP_CLIENT_HEADERS,
                "Range": f"bytes=0-{_MAGIC_PREFIX_SIZE - 1}",
            },
            timeout=DOWNLOAD_TIMEOUT,
        ) as session:
            try:
                async with session.get(self._url) as resp:
                    if resp.status not in (200, 206):
                        resp.raise_for_status()
                    self._content_type = _magic_detect(
                        await resp.content.read(_MAGIC_PREFIX_SIZE)
                    )
            except AIOHTTPClientError as error:
                msg = f"Error downloading {self._url}: {error}"
                raise ApiError(msg) from error

    async def _read(self) -> bytes:
        """Download the full HTTP response body.

        Returns:
            The complete file bytes.

        Raises:
            ApiError: When the HTTP download fails or returns an empty body.
        """
        await self._validate_ssrf()
        async with ClientSession(
            headers=HTTP_CLIENT_HEADERS, timeout=DOWNLOAD_TIMEOUT
        ) as session:
            try:
                async with session.get(self._url) as resp:
                    resp.raise_for_status()
                    if not (body := await resp.read()):
                        msg = f"Error downloading {self._url}: Empty body"
                        raise ApiError(msg)
            except AIOHTTPClientError as error:
                msg = f"Error downloading {self._url}: {error}"
                raise ApiError(msg) from error
        return body

    async def to_s3(
        self, region: RegionName, *, bucket: str | None = None, key: str | None = None
    ) -> S3Object:
        """Ensure the file is on S3 in *region* and return an ``S3Object``.

        This is a **terminal method** — calling it consumes the source.

        Args:
            region: Target AWS region.
            bucket: Explicit destination bucket.  When ``None``, the
                regional bucket from settings is used.
            key: Explicit destination object key.  When ``None``, a
                temporary key is auto-generated.

        Returns:
            An ``S3Object`` pointing to the S3 object.
        """
        await self._validate_ssrf()
        content_type = self._content_type if hasattr(self, "_content_type") else None
        async with (
            ClientSession(
                headers=HTTP_CLIENT_HEADERS, timeout=DOWNLOAD_TIMEOUT
            ) as session,
            session.get(self._url) as resp,
        ):
            resp.raise_for_status()
            return await put_s3_object(
                read_chunks(resp.content, UPLOAD_CHUNK_SIZE),
                content_type,
                region=region,
                bucket=bucket,
                key=key,
                temporary=True,
            )

    async def _validate_ssrf(self) -> None:
        """Validates the URL against SSRF (Server-Side Request Forgery)."""
        if not self._ssrf_validated:
            await validate_url_ssrf(self._url)
            self._ssrf_validated = True


class _DataUriSource(_FileSource):
    """Source backend for ``data:`` URIs."""

    __slots__ = ("_data_start", "_value")

    _value: str
    _data_start: int

    def __init__(self, value: str) -> None:
        """Initialise with the data URI string.

        Args:
            value: The full ``data:`` URI.
        """
        self._value = value
        self._data_start = value.index(",") + 1
        self._repr = f"{value[: self._data_start + _B64_REPR_LIMIT]}..."

    async def _resolve_metadata(self) -> None:
        """Extract content type from the data URI header and calculate size."""
        if match := _DATA_URI_RE(self._value):
            self._content_type = match.group(1)
        else:  # pragma: no cover
            msg = f"Invalid data URI: {self._value}"
            raise ApiError(msg)
        self._filename = None
        self._size = b64_decoded_len(self._value, self._data_start)

    async def _read(self) -> bytes:
        """Decode the base64 payload of the data URI.

        Returns:
            The decoded file bytes.

        Raises:
            ApiError: When the data URI payload is invalid base64.
        """
        view = memoryview(self._value.encode())
        try:
            return await b64decode(view[self._data_start :])
        except ValueError as error:
            msg = f"Invalid data URI starting with {self._repr!r}: {error.args[0]}"
            raise ApiError(msg) from None
        finally:
            view.release()
            del self._value

    async def to_base64(self) -> str:
        """Return file content as a base64-encoded string.

        This is a **terminal method** — calling it consumes the source.

        Returns:
            The base64-encoded file content.
        """
        try:
            return self._value[self._value.index(",") + 1 :]
        finally:
            del self._value

    async def to_data_uri(self) -> str:
        """Return file content as a ``data:`` URI with content type.

        This is a **terminal method** — calling it consumes the source.

        Returns:
            A data URI string like ``data:image/png;base64,...``.
        """
        try:
            return self._value
        finally:
            del self._value


class _Base64Source(_FileSource):
    """Source backend for raw base64 strings."""

    __slots__ = ("_value",)

    _value: str

    def __init__(self, value: str) -> None:
        """Initialise with the raw base64 string.

        Args:
            value: The base64-encoded content.
        """
        self._value = value
        self._repr = f"{value[:_B64_REPR_LIMIT]}..."

    async def _resolve_metadata(self) -> None:
        """Decode a prefix of the base64 string to detect content via magic."""
        self._content_type = _magic_detect(
            await b64decode(self._value[:_MAGIC_PREFIX_SIZE_BASE64])
        )
        self._filename = None
        self._size = b64_decoded_len(self._value)

    async def _read(self) -> bytes:
        """Decode the full base64 string.

        Returns:
            The decoded file bytes.

        Raises:
            ApiError: When the base64 string is invalid.
        """
        try:
            data = await b64decode(self._value, validate=True)
        except ValueError as error:
            raise ApiError(str(error)) from None
        finally:
            del self._value
        self._metadata_from_bytes(data)
        return data

    async def to_base64(self) -> str:
        """Return file content as a base64-encoded string.

        This is a **terminal method** — calling it consumes the source.

        Returns:
            The base64-encoded file content.
        """
        try:
            return self._value
        finally:
            del self._value

    async def to_data_uri(self) -> str:
        """Return file content as a ``data:`` URI with content type.

        This is a **terminal method** — calling it consumes the source.

        Returns:
            A data URI string like ``data:image/png;base64,...``.
        """
        try:
            return f"data:{await self.get_content_type() or 'application/octet-stream'};base64,{self._value}"
        finally:
            del self._value


class _UploadSource(_FileSource):
    """Source backend for ``starlette.datastructures.UploadFile``."""

    __slots__ = ("_upload",)

    _upload: UploadFile

    def __init__(self, upload: UploadFile) -> None:
        """Initialise with an ``UploadFile`` instance.

        Args:
            upload: The uploaded file to wrap.
        """
        self._upload = upload
        self._repr = repr(upload)

    async def _resolve_metadata(self) -> None:
        """Read a prefix of the upload to detect content via magic.

        The file pointer is reset to the beginning afterwards so the full
        content can still be read.
        """
        self._content_type = _magic_detect(await self._upload.read(_MAGIC_PREFIX_SIZE))
        self._filename = self._upload.filename or (
            parse_content_disposition_filename(
                self._upload.headers.get("content-disposition", "")
            )
            or None
        )
        self._size = self._upload.size or 0
        await self._upload.seek(0)

    async def _read(self) -> bytes:
        """Read the full upload content and close the file.

        Returns:
            The complete file bytes.

        Raises:
            ApiError: When the upload reference is missing.
        """
        if not hasattr(self, "_content_type"):
            await self._resolve_metadata()
        try:
            return await self._upload.read()
        finally:
            await self._upload.close()
            del self._upload

    async def to_s3(
        self, region: RegionName, *, bucket: str | None = None, key: str | None = None
    ) -> S3Object:
        """Ensure the file is on S3 in *region* and return an ``S3Object``.

        This is a **terminal method** — calling it consumes the source.

        Args:
            region: Target AWS region.
            bucket: Explicit destination bucket.  When ``None``, the
                regional bucket from settings is used.
            key: Explicit destination object key.  When ``None``, a
                temporary key is auto-generated.

        Returns:
            An ``S3Object`` pointing to the S3 object.
        """
        if not hasattr(self, "_content_type"):
            await self._resolve_metadata()
        content_type = self._content_type
        try:
            return await put_s3_object(
                read_chunks(self._upload, UPLOAD_CHUNK_SIZE),
                content_type,
                region=region,
                bucket=bucket,
                key=key,
                temporary=True,
            )
        finally:
            await self._upload.close()
            del self._upload


class InputFile:
    """Represents an input file, supporting various origins like URLs, base64, S3 URIs, and data URIs."""

    #: Origins accepted by this class when used as a Pydantic field.
    ALLOWED_ORIGINS: frozenset[_FileOrigin] = frozenset(
        {
            _FileOrigin.BASE64,
            _FileOrigin.DATA_URI,
            _FileOrigin.HTTP_URL,
            _FileOrigin.S3_URI,
        }
    )

    __slots__ = ("_bedrock_source", "_origin", "_source")

    _origin: _FileOrigin
    _source: _FileSource
    _bedrock_source: ImageSourceTypeDef

    def __new__(
        cls, value: UploadFile | str, *, content_type: str | None = None
    ) -> Self:
        """Create a new InputFile, normalizing the URI and detecting origin.

        Args:
            value: Raw string (URL, data URI, base64, or S3 URI), or an
                ``UploadFile`` instance.
            content_type: Content type of the file, if already known.

        Returns:
            A new ``InputFile`` instance.

        Raises:
            ValueError: When *value* is a string whose detected origin is not
                in ``ALLOWED_ORIGINS``.
        """
        if isinstance(value, str):
            origin, normalised, bucket, key = cls._normalize_and_detect_origin(value)
            instance = super().__new__(cls)
            instance._origin = origin
            instance._source = cls._build_source(origin, normalised, bucket, key)
            if content_type:
                instance._source._content_type = content_type  # noqa: SLF001
            _track_current_input_files(instance)
            return instance
        if isinstance(value, UploadFile):
            instance = super().__new__(cls)
            instance._origin = _FileOrigin.UPLOAD
            instance._source = _UploadSource(value)
            _track_current_input_files(instance)
            return instance
        msg = f"Unsupported input type: {type(value).__name__!r}"
        raise ValueError(msg)

    @classmethod
    def _normalize_and_detect_origin(
        cls, value: str
    ) -> tuple[_FileOrigin, str, str, str]:
        """Detect origin and normalise value.

        Args:
            value: Raw input string (URL, data URI, base64, etc.).

        Returns:
            A ``(origin, normalised_value, bucket, key)`` tuple.

        Raises:
            ValueError: When the detected origin is not in ``ALLOWED_ORIGINS``.
        """
        if value.startswith("data:"):
            origin, normalised, bucket, key = _FileOrigin.DATA_URI, value, "", ""
        elif match := _S3_URI_RE(value):
            origin, normalised, bucket, key = (
                _FileOrigin.S3_URI,
                value,
                match["bucket"],
                match["key"],
            )
        elif _HTTP_URI_RE(value):
            if (match := (_S3_VIRTUAL_HOST_RE(value) or _S3_PATH_STYLE_RE(value))) and (
                bucket := match["bucket"]
            ) in _ACCEPTED_BUCKETS:
                origin, normalised, key = (
                    _FileOrigin.S3_URI,
                    f"s3://{bucket}/{match['key']}",
                    match["key"],
                )
            else:
                origin, normalised, bucket, key = _FileOrigin.HTTP_URL, value, "", ""
        else:
            origin, normalised, bucket, key = _FileOrigin.BASE64, value, "", ""

        if origin not in cls.ALLOWED_ORIGINS:
            allowed = ", ".join(o.name for o in sorted(cls.ALLOWED_ORIGINS))
            msg = f"Expected one of [{allowed}], got {origin.name}."
            raise ValueError(msg)
        return origin, normalised, bucket, key

    @staticmethod
    def _build_source(
        origin: _FileOrigin, normalised: str, bucket: str, key: str
    ) -> _FileSource:
        """Construct the appropriate ``FileSource`` for *origin*.

        Args:
            origin: Detected file origin.
            normalised: Normalised string value.
            bucket: S3 bucket name (empty for non-S3 origins).
            key: S3 object key (empty for non-S3 origins).

        Returns:
            A concrete ``FileSource`` instance.
        """
        match origin:
            case _FileOrigin.S3_URI:
                return _S3Source(uri=normalised, bucket=bucket, key=key)
            case _FileOrigin.HTTP_URL:
                return _HttpSource(url=normalised)
            case _FileOrigin.DATA_URI:
                return _DataUriSource(value=normalised)
            case _FileOrigin.BASE64:
                return _Base64Source(value=normalised)
            case _:  # pragma: no cover
                msg = f"Cannot create source for origin {origin}"
                raise ValueError(msg)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: type, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Return the Pydantic v2 core schema for this type.

        Args:
            source_type: The Python type being processed.
            handler: Pydantic schema generation handler.

        Returns:
            A Pydantic core schema that validates file-reference strings.
        """
        return no_info_plain_validator_function(
            cls,
            json_schema_input_schema=str_schema(
                min_length=1, pattern=_URL_ONLY_SCHEMA_PATTERN
            )
            if (
                bool(cls.ALLOWED_ORIGINS & _URL_ONLY_ORIGINS)
                and _FileOrigin.BASE64 not in cls.ALLOWED_ORIGINS
            )
            else str_schema(min_length=1),
            serialization=plain_serializer_function_ser_schema(repr, info_arg=False),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Return the JSON schema for this type, driven by ``ALLOWED_ORIGINS``.

        Args:
            _schema: The core schema.
            handler: Pydantic JSON schema handler.

        Returns:
            A JSON schema dict describing accepted string patterns.
        """
        schema: JsonSchemaValue = {"type": "string", "minLength": 1}
        if (
            bool(
                cls.ALLOWED_ORIGINS
                & {_FileOrigin.HTTP_URL, _FileOrigin.S3_URI, _FileOrigin.DATA_URI}
            )
            and _FileOrigin.BASE64 not in cls.ALLOWED_ORIGINS
        ):
            schema["pattern"] = _URL_ONLY_SCHEMA_PATTERN
        return schema

    @property
    def region(self) -> RegionName | None:
        """Return the S3 region, or ``None`` for non-S3 files."""
        return self._source.region()

    @property
    def is_s3(self) -> bool:
        """Returns if the file comes from S3.

        Returns:
            True if a S3 file.
        """
        return self._source.is_s3()

    async def get_filename(self) -> str | None:
        """Return a filename derived from the file source.

        Returns:
            A filename string.
        """
        return await self._source.get_filename()

    async def get_content_type(self) -> str:
        """Resolve and return the content type.

        Returns:
            The content type string.
        """
        return await self._source.get_content_type()

    async def get_content_type_tuple(self) -> tuple[str, str]:
        """Resolve and return the content type split as media type and subtype.

        Returns:
            The content type as a tuple (media type, subtype).
        """
        media_type, _, subtype = (await self._source.get_content_type()).partition("/")
        return media_type, subtype

    async def get_size(self) -> int:
        """Resolve and return the file size.

        Returns:
            The file size in bytes.
        """
        return await self._source.get_size()

    async def to_bytes(self) -> bytes:
        """Return the full file content as bytes.

        This is a **terminal method** — calling it consumes the source.

        Returns:
            The complete file content.
        """
        return await self._source.to_bytes()

    async def to_base64(self) -> str:
        """Return file content as a base64-encoded string.

        This is a **terminal method** — calling it consumes the source.

        Returns:
            The base64-encoded file content.
        """
        return await self._source.to_base64()

    async def to_data_uri(self) -> str:
        """Return file content as a ``data:`` URI with content type.

        This is a **terminal method** — calling it consumes the source.

        Returns:
            A data URI string like ``data:image/png;base64,...``.
        """
        return await self._source.to_data_uri()

    async def to_s3(
        self, region: RegionName, *, bucket: str | None = None, key: str | None = None
    ) -> S3Object:
        """Ensure the file is on S3 in *region* and return an ``S3Object``.

        Args:
            region: Target AWS region.
            bucket: Explicit destination bucket.  When ``None``, the
                regional bucket from settings is used.
            key: Explicit destination object key.  When ``None``, a
                temporary key is auto-generated.

        Returns:
            An ``S3Object`` pointing to the S3 object.

        Raises:
            ApiError: When the file cannot be uploaded or copied.
        """
        return await self._source.to_s3(region, bucket=bucket, key=key)

    async def to_bedrock_content_block(
        self,
        media_type: _BedrockMediaType | None = None,
        content_type: str | None = None,
        filename: str | None = None,
        context: str | None = None,
        *,
        citations_enabled: bool = False,
    ) -> ContentBlockTypeDef:
        """Build a partial Bedrock content block for this file.

        Args:
            media_type: Media category override. Auto-detected when None.
            content_type: Content type override. Auto-detected when None.
            filename: Filename for document blocks.
            context: Optional context string added to document blocks.
            citations_enabled: When True, enables citations on document blocks.
        """
        if content_type is None:
            content_type = await self.get_content_type()
        media_type_from_mime, _, format_from_mime = content_type.partition("/")
        self._bedrock_source: ImageSourceTypeDef = {}
        match media_type or media_type_from_mime:
            case "image":
                image_block: ImageBlockTypeDef = {
                    "format": format_from_mime,  # type: ignore[typeddict-item]
                    "source": self._bedrock_source,
                }
                return {"image": image_block}
            case "video":
                video_block: VideoBlockTypeDef = {
                    "format": MIME_TYPES_TO_VIDEO_TYPE.get(
                        format_from_mime,
                        format_from_mime,  # type: ignore[arg-type]
                    ),
                    "source": self._bedrock_source,
                }
                return {"video": video_block}
            case "audio":
                audio_block: AudioBlockTypeDef = {
                    "format": MIME_TYPES_TO_AUDIO_TYPE.get(
                        format_from_mime,
                        format_from_mime,  # type: ignore[arg-type]
                    ),
                    "source": self._bedrock_source,
                }
                return {"audio": audio_block}
            case _:
                if format_from_mime not in _BEDROCK_DOCUMENT_FORMATS:
                    msg = f"Unsupported MIME type for Bedrock document: {await self.get_content_type()!r}"
                    raise ApiError(msg)
                document_source: DocumentSourceTypeDef = self._bedrock_source  # type: ignore[assignment]
                document_block: DocumentBlockTypeDef = {
                    "format": MIME_TYPES_TO_DOCUMENT_TYPE.get(format_from_mime, "txt"),
                    "name": _BEDROCK_DOC_NAME_RE.sub(
                        "", filename or await self.get_filename() or "file"
                    )[:200],
                    "source": document_source,
                }
                if context:
                    document_block["context"] = context
                if citations_enabled:
                    document_block["citations"] = {"enabled": True}
                return {"document": document_block}

    async def resolve_bedrock_content_block(
        self, region: RegionName, *, to_s3: bool | None = None
    ) -> None:
        """Populate the partial Bedrock content block for this file with final content.

        No-op if ``to_bedrock_content_block`` has not been called first.

        Args:
            region: Target AWS region.
            to_s3: S3 routing override; None auto-selects based on file origin.
        """
        if hasattr(self, "_bedrock_source"):
            if to_s3 or (to_s3 is None and self._origin == _FileOrigin.S3_URI):
                self._bedrock_source["s3Location"] = {
                    "uri": (await self.to_s3(region)).uri
                }
            else:
                self._bedrock_source["bytes"] = await self.to_bytes()
            del self._bedrock_source

    def __repr__(self) -> str:
        """Return representation of this InputFile.

        Returns:
            A short descriptive string.
        """
        return self._source.__repr__()

    __str__ = __repr__


class InputFileUrl(InputFile):
    """``InputFile`` variant that only accepts URL schemes (http(s), s3, data URI)."""

    ALLOWED_ORIGINS: frozenset[_FileOrigin] = frozenset(
        {_FileOrigin.DATA_URI, _FileOrigin.HTTP_URL, _FileOrigin.S3_URI}
    )


class InputFileBase64(InputFile):
    """``InputFile`` variant that accepts raw base64 or data URIs."""

    ALLOWED_ORIGINS: frozenset[_FileOrigin] = frozenset(
        {_FileOrigin.BASE64, _FileOrigin.DATA_URI}
    )


def get_s3_input_regions() -> dict[RegionName, int]:
    """Return S3 region → total object size (bytes) for all S3-sourced InputFiles in the current request.

    Aggregates sizes across all tracked ``InputFile`` instances that originate
    from S3.  Only files whose size is already resolved (i.e.
    ``_resolve_metadata`` has been called) contribute their size; unresolved
    files contribute 0 bytes but still register their region so the region is
    considered as a candidate.

    Returns:
        Mapping of AWS region string to cumulative byte count.  Empty dict when
        no S3 inputs are present in the current request context.
    """
    regions: dict[RegionName, int] = {}
    for file in _CURRENT_INPUT_FILES.get(()):
        if (region := file.region) is not None:
            source = file._source  # noqa: SLF001
            size = source._size if hasattr(source, "_size") else 0  # noqa: SLF001
            regions[region] = regions.get(region, 0) + size
    return regions


async def prefetch_all_content_types() -> None:
    """Pre-fetch content types for all InputFile instances in the current request context."""
    if input_files := _CURRENT_INPUT_FILES.get([]):
        async with TaskGroup() as task_group:
            for input_file in input_files:
                task_group.create_task(input_file.get_content_type())


async def resolve_all_bedrock_content_blocks(
    region: RegionName, *, to_s3: bool | None = None
) -> None:
    """Resolve all pending Bedrock content blocks for the current request context.

    Args:
        region: Target AWS region.
        to_s3: S3 routing override passed to each resolve_bedrock_content_block call.
    """
    if input_files := _CURRENT_INPUT_FILES.get([]):
        async with TaskGroup() as task_group:
            for input_file in input_files:
                task_group.create_task(
                    input_file.resolve_bedrock_content_block(region, to_s3=to_s3)
                )
