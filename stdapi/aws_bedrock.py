"""Common AWS Bedrock utilities."""

from binascii import Error as BinasciiError
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from re import IGNORECASE
from re import compile as compile_regex
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from aiohttp import ClientError as AIOHTTPClientError
from aiohttp import ClientSession
from botocore.exceptions import ClientError
from fastapi import HTTPException
from magic import from_buffer
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue
from starlette.datastructures import Headers

from stdapi.config import DOWNLOAD_TIMEOUT, SETTINGS
from stdapi.openai_exceptions import OpenaiError
from stdapi.security import validate_url_ssrf
from stdapi.server import HTTP_CLIENT_HEADERS
from stdapi.types import BaseModelRequestWithExtra
from stdapi.types.openai_chat_completions import ReasoningEffort
from stdapi.utils import b64decode, validation_error_handler

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from types_aiobotocore_bedrock_runtime.literals import (
        DocumentFormatType,
        GuardrailTraceType,
        ImageFormatType,
        VideoFormatType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockTypeDef,
        GuardrailStreamConfigurationTypeDef,
        ImageBlockTypeDef,
        InferenceConfigurationTypeDef,
        MessageUnionTypeDef,
        PerformanceConfigurationTypeDef,
        PromptVariableValuesTypeDef,
        SystemContentBlockTypeDef,
        ToolConfigurationTypeDef,
    )

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


#: Bedrock documents types with the matching MIME type
MIME_TYPES_TO_DOCUMENT_TYPE: "dict[str, DocumentFormatType]" = {
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
MIME_TYPES_TO_VIDEO_TYPE: "dict[str, VideoFormatType]" = {
    # Only list values that differ
    "x-matroska": "mkv",
    "quicktime": "mov",
    "x-flv": "flv",
    "x-ms-wmv": "wmv",
    "3gpp": "three_gp",
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
GUARDTRAIL_CONFIG_VAR: ContextVar["GuardrailStreamConfigurationTypeDef"] = ContextVar(
    "guardtrail_configuration"
)
_GUARDTRAIL_IDENTIFIER_HEADER = "X-Amzn-Bedrock-GuardrailIdentifier"
_GUARDTRAIL_VERSION_HEADER = "X-Amzn-Bedrock-GuardrailVersion"
_GUARDTRAIL_TRACE_HEADER = "X-Amzn-Bedrock-Trace"
_GUARDTRAIL_TRACE_VALUES = {"disabled", "enabled", "enabled_full"}


#: Reasoning models: Budget factor over the token max count
_REASONING_EFFORT_BUDGET_FACTOR: dict[ReasoningEffort, float] = {
    "minimal": 0.25,
    "low": 0.5,
    "medium": 0.75,
    "high": 1.0,
}


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
    __pydantic_extra__: dict[str, JsonValue] = {}

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
    """
    if (
        _GUARDTRAIL_IDENTIFIER_HEADER in headers
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


def set_inference_configuration(
    model_id: str,
    additional_request_fields: dict[str, JsonValue],
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop_sequences: list[str] | str | None = None,
    **extra_params: JsonValue,
) -> "InferenceConfigurationTypeDef":
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
) -> dict[str, JsonValue]:
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
        params: dict[str, JsonValue] = SETTINGS.default_model_params[model_id]
    except KeyError:
        params = {}
    params.update(request.model_extra or {})
    return params


def set_reasoning_configuration(
    model_id: str,
    reasoning_effort: ReasoningEffort | None,
    budget_tokens: int | None,
    max_tokens: int | None,
    additional_request_fields: dict[str, Any],
) -> None:
    """Configures reasoning parameters.

    If a budget_tokens value is provided, the reasoning configuration enables
    reasoning with the specified token budget. Otherwise, no reasoning
    configuration is applied.

    Args:
        model_id: Model identifier.
        budget_tokens: Optional. An integer specifying the maximum number
            of tokens allowed for reasoning. If None, reasoning is not
            enabled.
        max_tokens: Maximum number of tokens allowed for the model.
        reasoning_effort: string, optional. The type of budget to use for reasoning.
        additional_request_fields: A mapping that represents the request's
            fields. This will be modified to include reasoning configuration
            if a budget is provided.
    """
    if model_id.startswith("deepseek"):
        # reasoning_config string, DeepSeek case (At least for "deepseek.v3-v1:0")
        if reasoning_effort is None:
            msg = f"{model_id} only support 'reasoning_effort' to configure reasoning."
            raise OpenaiError(msg)
        additional_request_fields["reasoning_config"] = (
            "low" if reasoning_effort == "minimal" else reasoning_effort
        )
    else:
        # Default to the budget case (at least used by Anthropic Claude)
        additional_request_fields["reasoning_config"] = {
            "type": "enabled",
            "budget_tokens": budget_tokens
            # Convert effort to budget
            # Default to Anthropic Claude minimal reasoning model context size
            or max(
                1024,
                int(
                    ((max_tokens or 32768) - 1)
                    * _REASONING_EFFORT_BUDGET_FACTOR[reasoning_effort or "high"]
                ),
            ),
        }


@contextmanager
def handle_bedrock_client_error() -> Generator[None]:
    """Context manager to translate Bedrock client errors to appropriate HTTP 4XX/5XX when possible.

    Raises:
        HTTPException: With a status mapped from common Bedrock error codes.

    Usage:
        with handle_bedrock_client_error():
            response = await bedrock.converse(**request)
    """
    try:
        yield
    except ClientError as error:
        error_code = error.response["Error"]["Code"]
        error_message = error.response["Error"]["Message"]
        if (
            error.response["Error"]["Code"] == "ValidationException"
            and "Invalid S3 credentials" in error.response["Error"]["Message"]
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unable to access the S3 bucket. "
                    "Ensure the S3 bucket is in the same region as the Bedrock model that is called."
                ),
            ) from error
        if error_code in _BEDROCK_MODEL_ERROR_CODES:  # pragma: no cover
            raise HTTPException(status_code=500, detail=error_message) from error
        if error_code == "ModelNotReadyException":  # pragma: no cover
            raise HTTPException(status_code=503, detail=error_message) from error
        raise  # pragma: no cover


def image_block_from_bytes(data: bytes, mime: str = "") -> "ContentBlockTypeDef":
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


async def image_block_from_s3_url(url: str) -> "ContentBlockTypeDef | None":
    """Convert an s3:// URL to a Bedrock image content block using s3Location.

    Args:
        url: S3 URL string like s3://bucket/key

    Returns:
        Content block dict with s3Location, or None when not s3.

    Raises:
        HTTPException: If the URL does not contain a supported image extension.
    """
    if not url.lower().startswith("s3://"):
        return None  # Not an S3 URL
    match = _IMAGE_URL_EXT.search(url)
    if match:
        ext = match.group(1).lower()
        image: ImageBlockTypeDef = {
            "format": "jpeg" if ext == "jpg" else ext,  # type: ignore[typeddict-item]
            "source": {"s3Location": {"uri": url}},
        }
        return {"image": image}
    raise HTTPException(status_code=400, detail=f"Invalid image data URL: {url}")


async def image_block_from_http_url(url: str) -> "ContentBlockTypeDef | None":
    """Download an image over HTTP(S) and return a Bedrock content block.

    Args:
        url: HTTP or HTTPS URL.

    Returns:
        A ContentBlockTypeDef with image bytes and inferred format, or None if
        the URL is not HTTP(S).

    Raises:
        HTTPException: With status 400 when the download fails or the body is empty.
    """
    url_lower = url.lower()
    if url_lower.startswith(("http://", "https://")):
        await validate_url_ssrf(url_lower)
        async with ClientSession(
            headers=HTTP_CLIENT_HEADERS, timeout=DOWNLOAD_TIMEOUT
        ) as session:
            try:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    body = await resp.read()
            except AIOHTTPClientError as error:
                raise HTTPException(
                    status_code=400, detail=f"Error downloading image {url}: {error}"
                ) from error
            if not body:
                raise HTTPException(
                    status_code=400, detail=f"Error downloading image {url}: Empty body"
                )
            return image_block_from_bytes(body)
    return None


async def image_block_from_data_url(url: str) -> "ContentBlockTypeDef | None":
    """Convert a data: URL to a Bedrock image content block.

    Supports common image mime types and base64 payloads only. Returns None when
    the URL is not a supported data URL.

    Args:
        url: Data URL string like data:image/png;base64,<b64>

    Returns:
        Content block dict with image bytes and format, or None.
    """
    match = _IMAGE_DATA_EXT.match(url)
    if not match:
        return None  # Not an image data
    try:
        data = await b64decode(match.group(2), validate=True)
    except BinasciiError:
        raise HTTPException(
            status_code=400, detail=f"Invalid base64 in data URL: {url}"
        ) from None
    return image_block_from_bytes(data)
