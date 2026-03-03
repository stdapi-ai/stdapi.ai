"""Common AWS Bedrock utilities."""

from contextlib import contextmanager
from contextvars import ContextVar
from re import IGNORECASE
from re import compile as compile_regex
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

from aiohttp import ClientError as AIOHTTPClientError
from aiohttp import ClientSession
from botocore.exceptions import ClientError
from magic import from_buffer
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.config import DOWNLOAD_TIMEOUT, SETTINGS
from stdapi.security import validate_url_ssrf
from stdapi.server import HTTP_CLIENT_HEADERS
from stdapi.types import JsonMapping  # noqa: TC001
from stdapi.utils import b64decode, validation_error_handler

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

    from starlette.datastructures import Headers
    from types_aiobotocore_bedrock_runtime.literals import (
        AudioFormatType,
        DocumentFormatType,
        GuardrailTraceType,
        ImageFormatType,
        PerformanceConfigLatencyType,
        ServiceTierTypeType,
        VideoFormatType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockTypeDef,
        GuardrailStreamConfigurationTypeDef,
        ImageBlockTypeDef,
        InferenceConfigurationTypeDef,
        MessageUnionTypeDef,
        OutputConfigTypeDef,
        PerformanceConfigurationTypeDef,
        PromptVariableValuesTypeDef,
        ServiceTierTypeDef,
        SystemContentBlockTypeDef,
        ToolConfigurationTypeDef,
    )

    from stdapi.types import BaseModelRequestWithExtra

    class ConverseRequestBaseTypeDef(TypedDict):
        """Converse request base type definition.

        Common fields from "ConverseRequestTypeDef" and
        "ConverseStreamRequestTypeDef".
        """

        modelId: str
        messages: NotRequired[Sequence[MessageUnionTypeDef]]
        system: NotRequired[Sequence[SystemContentBlockTypeDef]]
        inferenceConfig: NotRequired[InferenceConfigurationTypeDef]
        toolConfig: NotRequired[ToolConfigurationTypeDef]
        additionalModelRequestFields: NotRequired[Mapping[str, Any]]
        promptVariables: NotRequired[Mapping[str, PromptVariableValuesTypeDef]]
        additionalModelResponseFieldPaths: NotRequired[Sequence[str]]
        requestMetadata: NotRequired[Mapping[str, str]]
        performanceConfig: NotRequired[PerformanceConfigurationTypeDef]
        guardrailConfig: NotRequired[GuardrailStreamConfigurationTypeDef]
        serviceTier: NotRequired[ServiceTierTypeDef]
        outputConfig: NotRequired[OutputConfigTypeDef]


#: Bedrock documents types with the matching MIME type
MIME_TYPES_TO_DOCUMENT_TYPE: dict[str, DocumentFormatType] = {
    "csv": "csv",
    "html": "html",
    "pdf": "pdf",
    "msword": "doc",
    "vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "vnd.ms-excel": "xls",
    "vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "plain": "txt",
    "markdown": "md",
}

#: Bedrock videos types with the matching MIME type
MIME_TYPES_TO_VIDEO_TYPE: dict[str, VideoFormatType] = {
    # Only list values that differ
    "x-matroska": "mkv",
    "quicktime": "mov",
    "x-flv": "flv",
    "x-ms-wmv": "wmv",
    "3gpp": "three_gp",
}

#: Bedrock audio types with the matching MIME type
MIME_TYPES_TO_AUDIO_TYPE: dict[str, AudioFormatType] = {
    # Only list values that differ
    "x-aac": "aac",
    "x-flac": "flac",
    "x-m4a": "mp4",
    "mpeg": "mp3",
    "x-wav": "wav",
}

#: Bedrock limit for sync body size (25MB), here with a little margin
BEDROCK_BODY_SIZE_LIMIT = 24_990_000

#: Bedrock supported image from data URL
_IMAGE_DATA_EXT = compile_regex(
    r"^data:image/(png|jpeg|jpg|gif|webp);base64,(.+)$", IGNORECASE
)

#: Bedrock supported image file extension
_IMAGE_URL_EXT = compile_regex(r"\.(png|jpeg|jpg|gif|webp)(?:\?|$)", IGNORECASE)

#: Bedrock error codes on model error
_BEDROCK_MODEL_ERROR_CODES = {
    "ModelErrorException",
    "ModelStreamErrorException",
    "ModelTimeoutException",
}

#: Guardtrail configuration for the request.
GUARDTRAIL_CONFIG_VAR: ContextVar[GuardrailStreamConfigurationTypeDef] = ContextVar(
    "guardtrail_configuration"
)
_GUARDTRAIL_IDENTIFIER_HEADER = "X-Amzn-Bedrock-GuardrailIdentifier"
_GUARDTRAIL_VERSION_HEADER = "X-Amzn-Bedrock-GuardrailVersion"
_GUARDTRAIL_TRACE_HEADER = "X-Amzn-Bedrock-Trace"
_GUARDTRAIL_TRACE_VALUES = {"disabled", "enabled", "enabled_full"}

#: Performance configuration for the request
PERFORMANCE_CONFIG_VAR: ContextVar[
    tuple[PerformanceConfigLatencyType | None, ServiceTierTypeType | None]
] = ContextVar("performance_configuration")
_PERFORMANCE_CONFIG_LATENCY_HEADER = "X-Amzn-Bedrock-PerformanceConfig-Latency"
_SERVICE_TIER_HEADER = "X-Amzn-Bedrock-Service-Tier"

#: Prompt caching type
PromptCaching = Literal["system", "messages", "tools"]

#: Available prompt caching
PROMPT_CACHING: frozenset[PromptCaching] = frozenset(("system", "messages", "tools"))

#: Minimal prompt caching
PROMPT_CACHING_BASIC: frozenset[PromptCaching] = frozenset(("system", "messages"))

# Default prompt caching configuration
PROMPT_CACHING_DEFAULT: ContentBlockTypeDef = {"cachePoint": {"type": "default"}}


class _DefaultModelParameters(BaseModel):
    """Default model parameters for AI/ML inference requests.

    This class defines common parameters that can be applied to AI models by default.
    Parameters are validated according to their expected ranges and types.

    These parameters control how models generate responses and are passed to the model
    as Bedrock inference parameters:
    - temperature: Controls randomness (0.0 = deterministic, 1.0 = very random)
    - top_p: Nucleus sampling threshold (0.0-1.0)
    - stop_sequences: Text patterns that halt generation
    - max_tokens: Maximum response length

    The class supports additional provider-specific parameters via the
    'extra="allow"' configuration, these parameters are then passed to the model
    as extra requests fields.

    Examples:
        Basic parameters:
            params = _DefaultModelParameters(temperature=0.7, max_tokens=1000)

        With provider-specific options:
            params = _DefaultModelParameters(
                temperature=0.5,
                anthropic_beta=["feature-flag"]
            )
    """

    model_config = ConfigDict(extra="allow", frozen=True)
    __pydantic_extra__: JsonMapping = {}

    # Validate AWS Bedrock defined inference parameters
    # With validation_alias to the Bedrock native name
    temperature: float | None = Field(
        default=None, ge=0, description="Default sampling temperature to use."
    )
    top_p: float | None = Field(
        validation_alias=AliasChoices("top_p", "topP"),
        default=None,
        ge=0,
        description="Default nucleus sampling.",
    )
    stop_sequences: str | list[str] | None = Field(
        validation_alias=AliasChoices("stop_sequences", "stopSequences"),
        default=None,
        description="Default sequences where the API will stop generating further tokens.",
    )
    max_tokens: int | None = Field(
        validation_alias=AliasChoices("max_tokens", "maxTokens"),
        default=None,
        ge=1,
        description="The default maximum number of tokens that can be generated by the model",
    )


def set_guardrail_configuration(headers: Headers) -> None:
    """Set the AWS Bedrock Guardrail configuration for the request.

    Configured globally via environment variables.

    Also, configurable per API call using the same headers as the
    AWS Bedrock OpenAI Chat Completions API (Available for OpenAI models):
    - X-Amzn-Bedrock-GuardrailIdentifier
    - X-Amzn-Bedrock-GuardrailVersion
    - X-Amzn-Bedrock-Trace

    Note: Request-level headers are allowed when aws_bedrock_allow_guardrail_override is True.
    This setting is automatically enabled when no global guardrail configuration exists.

    Args:
        headers: The headers of the request.
    """
    if (
        SETTINGS.aws_bedrock_allow_guardrail_override
        and _GUARDTRAIL_IDENTIFIER_HEADER in headers
        and _GUARDTRAIL_VERSION_HEADER in headers
    ):
        config: GuardrailStreamConfigurationTypeDef = {
            "guardrailIdentifier": headers[_GUARDTRAIL_IDENTIFIER_HEADER].strip(),
            "guardrailVersion": headers[_GUARDTRAIL_VERSION_HEADER].strip(),
        }
        trace: GuardrailTraceType = (
            headers.get(_GUARDTRAIL_TRACE_HEADER, "").strip().lower()  # type: ignore[assignment]
        )
        if trace in _GUARDTRAIL_TRACE_VALUES:
            config["trace"] = trace
    elif (
        SETTINGS.aws_bedrock_guardrail_identifier
        and SETTINGS.aws_bedrock_guardrail_version
    ):
        config = {
            "guardrailIdentifier": SETTINGS.aws_bedrock_guardrail_identifier,
            "guardrailVersion": SETTINGS.aws_bedrock_guardrail_version,
        }
        if SETTINGS.aws_bedrock_guardrail_trace:
            config["trace"] = SETTINGS.aws_bedrock_guardrail_trace
    else:
        return
    GUARDTRAIL_CONFIG_VAR.set(config)


def set_performance_configuration(headers: Headers) -> None:
    """Set the AWS Bedrock performance configuration for the request.

    Configured globally via environment variables.

    Also, configurable per API call using the same headers as the
    AWS Bedrock OpenAI Chat Completions API (Available for OpenAI models):
    - X-Amzn-Bedrock-PerformanceConfig-Latency
    - X-Amzn-Bedrock-Service-Tier

    Args:
        headers: The headers of the request.
    """
    latency: PerformanceConfigLatencyType | None = (
        headers[_PERFORMANCE_CONFIG_LATENCY_HEADER].strip()  # type: ignore[assignment]
        if _PERFORMANCE_CONFIG_LATENCY_HEADER in headers
        else None
    )
    service_tier: ServiceTierTypeType | None = (
        headers[_SERVICE_TIER_HEADER].strip()  # type: ignore[assignment]
        if _SERVICE_TIER_HEADER in headers
        else None
    )
    PERFORMANCE_CONFIG_VAR.set((latency, service_tier))


def set_inference_configuration(
    model_id: str,
    additional_request_fields: JsonMapping,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop_sequences: list[str] | str | None = None,
    **extra_params: JsonValue,
) -> InferenceConfigurationTypeDef:
    """Configures the inference settings.

    Args:
        model_id: Model identifier.
        additional_request_fields: Additional Bedrock request fields.
        temperature: Controls randomness of the output. Higher values result in more random
            outputs, while lower values make the output more deterministic.
        top_p: Limits the output tokens by cumulative probability, encouraging diverse
            outputs when set below 1.
        max_tokens : Defines the maximum number of tokens the model is allowed to generate.
        stop_sequences: Specifies sequences where the model should stop generating
            further tokens. Can be a single string or a list of strings.
        extra_params: Extra model parameters to pass as it.

    Returns:
        A dictionary containing the configured parameters for inference.
    """
    config: InferenceConfigurationTypeDef = {}
    with validation_error_handler():
        default = _DefaultModelParameters(
            **SETTINGS.default_model_params.get(model_id, {})  # type: ignore[arg-type]
        )

    # Pass Bedrock defined inference parameters
    temperature = temperature or default.temperature
    if temperature is not None:
        config["temperature"] = temperature

    top_p = top_p or default.top_p
    if top_p is not None:
        config["topP"] = top_p

    max_tokens = max_tokens or default.max_tokens
    if max_tokens is not None:
        config["maxTokens"] = max_tokens

    stop_sequences = stop_sequences or default.stop_sequences
    if stop_sequences is not None:
        config["stopSequences"] = (
            [stop_sequences] if isinstance(stop_sequences, str) else stop_sequences
        )

    # Pass other parameters as extra request fields to the model
    additional_request_fields.update(
        {
            key: value
            for key, value in ((default.model_extra or {}) | extra_params).items()
            if value is not None
        }
    )
    return config


def get_extra_model_parameters(
    model_id: str, request: BaseModelRequestWithExtra
) -> JsonMapping:
    """Fetches additional model parameters for a given model and request.

    This function retrieves the default parameters associated with the specified
    model ID and updates them with extra parameters provided in the request. If
    no default parameters are found for the model ID, an empty dictionary is used
    as the starting point.

    Args:
        model_id: The identifier for the model whose parameters are being retrieved.
        request: An instance of BaseModelRequestWithExtra containing additional
            parameters to customize or override the model's default parameters.

    Returns:
        A dictionary containing the aggregated model parameters.
    """
    try:
        params: JsonMapping = SETTINGS.default_model_params[model_id]
    except KeyError:
        params = {}
    params.update(request.model_extra or {})
    return params


#: AWS error codes to HTTP status + error type mapping
AWS_ERROR_MAP: dict[str, tuple[int, str]] = {
    **dict.fromkeys(
        {
            "ThrottlingException",
            "TooManyRequestsException",
            "ServiceQuotaExceededException",
        },
        (429, "rate_limit_error"),
    ),
    **dict.fromkeys({"AccessDeniedException"}, (403, "permission_error")),
    **dict.fromkeys(
        {
            "UnrecognizedClientException",
            "InvalidSignatureException",
            "ExpiredTokenException",
        },
        (401, "authentication_error"),
    ),
    **dict.fromkeys({"ResourceNotFoundException"}, (404, "not_found_error")),
    **dict.fromkeys(
        {"ValidationException", "BadRequestException"}, (400, "invalid_request_error")
    ),
    **dict.fromkeys(
        {
            "ServiceUnavailableException",
            "InternalServerException",
            "ServiceFailureException",
            "ReadTimeoutError",
        },
        (503, "server_error"),
    ),
}


@contextmanager
def handle_bedrock_client_error() -> Generator[None]:
    """Context manager to translate Bedrock client errors to appropriate HTTP 4XX/5XX when possible.

    Raises:
        ApiError: With a status mapped from common Bedrock error codes.

    Usage:
        with handle_bedrock_client_error():
            response = await bedrock.converse(**request)
    """
    try:
        yield
    except ClientError as error:
        error_message = error.response["Error"]["Message"]
        match error.response["Error"]["Code"]:
            case "ValidationException" if "Invalid S3 credentials" in error_message:
                msg = (
                    "Unable to access the S3 bucket. "
                    "Ensure the S3 bucket is in the same region as the Bedrock model that is called."
                )
                raise ApiError(msg) from error
            case code if code in _BEDROCK_MODEL_ERROR_CODES:  # pragma: no cover
                raise ApiError(error_message, status=500) from error
            case "ModelNotReadyException":  # pragma: no cover
                raise ApiError(error_message, status=503) from error
            case _:  # pragma: no cover
                raise


def image_block_from_bytes(data: bytes, mime: str = "") -> ContentBlockTypeDef:
    """Build a Bedrock image content block from raw bytes.

    Infers the image format using the provided MIME type when available, otherwise
    detects it from the bytes via python-magic. Supported formats include: png,
    jpeg/jpg, gif, and webp.

    Args:
        data: Raw image bytes.
        mime: Optional MIME type string (e.g., "image/png"). When empty, the
            type is detected from data.

    Returns:
        A Bedrock ContentBlockTypeDef with an image block containing bytes and
        the inferred format.
    """
    if not mime:
        mime = from_buffer(data, mime=True)
    image_format: ImageFormatType = mime.split("/", 1)[1]  # type: ignore[assignment]
    image_block: ImageBlockTypeDef = {"format": image_format, "source": {"bytes": data}}
    return {"image": image_block}


async def _download_http(url: str) -> bytes:
    """Download content from an HTTP(S) URL.

    Validates the URL against SSRF before downloading.

    Args:
        url: HTTP or HTTPS URL (caller must ensure the scheme is http/https).

    Returns:
        Downloaded bytes.

    Raises:
        ApiError: With status 400 when the download fails or the body is empty.
    """
    await validate_url_ssrf(url.lower())
    async with ClientSession(
        headers=HTTP_CLIENT_HEADERS, timeout=DOWNLOAD_TIMEOUT
    ) as session:
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                body = await resp.read()
        except AIOHTTPClientError as error:
            msg = f"Error downloading {url}: {error}"
            raise ApiError(msg) from error
        if not body:
            msg = f"Error downloading {url}: Empty body"
            raise ApiError(msg)
        return body


async def _download_s3(url: str) -> bytes:
    """Download content from an S3 URL via the S3 client.

    Args:
        url: S3 URL string like s3://bucket/key (caller must ensure s3:// scheme).

    Returns:
        Downloaded bytes.

    Raises:
        ApiError: If the S3 URL is malformed or the download fails.
    """
    path = url[5:]
    bucket, _, key = path.partition("/")
    if not bucket or not key:
        msg = f"Invalid S3 URL: {url}"
        raise ApiError(msg)
    async with get_client("s3") as s3:
        try:
            response = await s3.get_object(Bucket=bucket, Key=key)
            return await response["Body"].read()  # type: ignore[no-any-return]
        except ClientError as error:
            msg = f"Error downloading {url}: {error}"
            raise ApiError(msg) from error


def _image_block_from_s3_url(url: str) -> ContentBlockTypeDef:
    """Convert an s3:// URL to a Bedrock image content block using s3Location.

    Args:
        url: S3 URL string like s3://bucket/key

    Returns:
        Content block dict with s3Location.

    Raises:
        ApiError: If the URL does not contain a supported image extension.
    """
    if m := _IMAGE_URL_EXT.search(url):
        ext = m.group(1).lower()
        image: ImageBlockTypeDef = {
            "format": "jpeg" if ext == "jpg" else ext,  # type: ignore[typeddict-item]
            "source": {"s3Location": {"uri": url}},
        }
        return {"image": image}
    msg = f"Invalid image URL: {url}"
    raise ApiError(msg)


async def image_block_from_url(url: str) -> ContentBlockTypeDef:
    """Convert any supported image URL to a Bedrock image content block.

    Supports data URLs, s3:// URLs, and http(s) URLs.

    Args:
        url: Image URL string (data:, s3://, http://, or https://).

    Returns:
        A Bedrock ContentBlockTypeDef for the referenced image.

    Raises:
        ApiError: If the URL scheme is not supported.
    """
    url_lower = url.lower()
    match url_lower:
        case _ if url_lower.startswith("data:"):
            if m := _IMAGE_DATA_EXT.match(url):
                try:
                    data = await b64decode(m.group(2), validate=True)
                except ValueError as error:
                    msg = f"Invalid base64 in data URL starting with {url[:16]}: {error.args[0]}"
                    raise ApiError(msg) from None
                return image_block_from_bytes(data)
            msg = f"Invalid image data URL: {url}"
            raise ApiError(msg)
        case _ if url_lower.startswith("s3://"):
            return _image_block_from_s3_url(url)
        case _ if url_lower.startswith(("http://", "https://")):
            return image_block_from_bytes(await _download_http(url))
        case _:
            msg = f"Unsupported image URL: {url}"
            raise ApiError(msg)


async def download_content(url: str) -> bytes:
    """Download content from any supported URL scheme (http(s), s3).

    Args:
        url: URL string (http://, https://, or s3://).

    Returns:
        Downloaded bytes.

    Raises:
        ApiError: If the URL scheme is not supported or the download fails.
    """
    url_lower = url.lower()
    match url_lower:
        case _ if url_lower.startswith(("http://", "https://")):
            return await _download_http(url)
        case _ if url_lower.startswith("s3://"):
            return await _download_s3(url)
        case _:
            msg = f"Unsupported URL scheme: {url}"
            raise ApiError(msg)


def build_system_blocks(*content: str) -> list[SystemContentBlockTypeDef]:
    """Builds and returns a list of system content blocks from provided text elements.

    This function filters out any empty values from the given input and constructs a list
    of `SystemContentBlockTypeDef` dictionary objects, where each dictionary contains the
    text content.

    Args:
        *content: A variable number of string arguments representing the input content.

    Returns:
        A list of dictionary objects, each containing
            a `text` key with the corresponding non-None input string as its value.
    """
    return [{"text": part} for part in content if part]
