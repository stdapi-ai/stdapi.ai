"""Local OpenAI-compatible image generation types."""

from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, field_validator, model_validator

from stdapi.config import SETTINGS
from stdapi.input_file import FileIdInputFile, InputFileUrl  # noqa: TC001
from stdapi.monitoring import log_error_details
from stdapi.types import (
    BaseModelRequest,
    BaseModelRequestWithExtra,
    BaseModelRequestWithFormExtra,
    BaseModelResponse,
)
from stdapi.types.openai import Auto

if TYPE_CHECKING:
    from stdapi.input_file import InputFile

__all__ = [
    "Image",
    "ImageBackground",
    "ImageBackgroundAuto",
    "ImageEditJsonBody",
    "ImageEditParams",
    "ImageGenCompletedEvent",
    "ImageGenPartialImageEvent",
    "ImageGenerateParams",
    "ImageInputReferenceParam",
    "ImageOutputFormats",
    "ImageOutputQuality",
    "ImageOutputQualityAuto",
    "ImageVariationJsonBody",
    "ImageVariationParams",
    "ImagesResponse",
    "Usage",
    "UsageInputTokensDetails",
]

#: Supported image output formats
ImageOutputFormats = Literal["png", "jpeg", "webp"]

#: Supported quality output level
ImageOutputQuality = Literal["low", "medium", "high"]
ImageOutputQualityAuto = Auto | ImageOutputQuality

#: Supported image background
ImageBackground = Literal["transparent", "opaque"]
ImageBackgroundAuto = Auto | ImageBackground

#: Supported image input fidelity
ImageInputFidelity = Literal["low", "high"]


# Ref: openai.types.image_input_reference_param.ImageInputReferenceParam
class ImageInputReferenceParam(BaseModelRequest):
    """Reference to an input image — either a Files API file ID or a URL/data-URI.

    Exactly one of ``file_id`` or ``image_url`` must be provided.
    """

    file_id: FileIdInputFile | None = Field(
        default=None, description="The ID of a file uploaded via the Files API."
    )
    image_url: InputFileUrl | None = Field(
        default=None, description="A fully qualified URL or base64-encoded data URL."
    )

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        """Ensure at least one image source is provided.

        Raises:
            ValueError: When both fields are None.
        """
        if self.file_id is None and self.image_url is None:
            msg = "Provide one of 'file_id' or 'image_url'."
            raise ValueError(msg)
        return self

    @property
    def input_file(self) -> InputFile:
        """Return the resolved ``InputFile`` (``file_id`` takes priority).

        Returns:
            The non-None InputFile instance.
        """
        return self.file_id or self.image_url  # type: ignore[return-value]


# Ref: openai.types.image.Image
class Image(BaseModelResponse):
    """Generated image descriptor compatible with OpenAI."""

    b64_json: str | None = Field(
        default=None,
        description="Base64-encoded image data; present when response_format is `b64_json`.",
    )
    revised_prompt: str | None = Field(
        default=None, description="The revised prompt used to generate the image."
    )
    url: str | None = Field(
        default=None,
        description="URL of the generated image; present when response_format is `url`.",
    )


# Ref: openai.types.images_response.UsageInputTokensDetails
class UsageInputTokensDetails(BaseModelResponse):
    """Detailed input token usage for image generation."""

    image_tokens: int = Field(
        default=0, ge=0, description="The number of image tokens in the input prompt."
    )
    text_tokens: int = Field(
        default=0, ge=0, description="The number of text tokens in the input prompt."
    )


# Ref: openai.types.images_response.Usage
class Usage(BaseModelResponse):
    """Image generation token usage information."""

    input_tokens: int = Field(
        default=0,
        ge=0,
        description="Number of tokens (images + text) in the input prompt.",
    )
    input_tokens_details: UsageInputTokensDetails = Field(
        description="Detailed breakdown of input token usage."
    )
    output_tokens: int = Field(
        default=0, ge=0, description="Number of output tokens generated."
    )
    total_tokens: int = Field(
        default=0, ge=0, description="Total tokens used (input + output)."
    )


# Ref: openai.types.images_response.ImagesResponse
class ImagesResponse(BaseModelResponse):
    """OpenAI-compatible non-streaming image generation response."""

    created: int = Field(
        ge=0, description="Unix timestamp (seconds) when the image was created."
    )
    background: ImageBackground | None = Field(
        default=None, description="Background setting: `transparent` or `opaque`."
    )
    data: list[Image] | None = Field(
        default=None, description="List of generated images."
    )
    output_format: ImageOutputFormats | None = Field(
        default=None, description="Output format: `png`, `webp`, or `jpeg`."
    )
    quality: ImageOutputQuality | None = Field(
        default=None, description="Quality of the generated image."
    )
    size: str | None = Field(default=None, description="Size of the generated image.")
    usage: Usage | None = Field(
        default=None, description="Token usage information for the image generation."
    )


# Ref: openai.types.image_gen_completed_event.ImageGenCompletedEvent
class ImageGenCompletedEvent(BaseModelResponse):
    """Streaming event emitted when image generation completes."""

    b64_json: str = Field(description="Base64-encoded image data.")
    background: ImageBackgroundAuto = Field(
        description="Background setting for the generated image."
    )
    created_at: int = Field(
        ge=0, description="Unix timestamp when the event was created."
    )
    output_format: ImageOutputFormats = Field(
        description="Output format for the generated image."
    )
    quality: ImageOutputQualityAuto = Field(
        description="Quality setting for the generated image."
    )
    size: str | None = Field(default=None, description="Size of the generated image.")
    type: Literal["image_generation.completed"] = Field(
        description="Event type. Always `image_generation.completed`."
    )
    usage: Usage = Field(description="Token usage for the image generation.")


# Ref: openai.types.image_gen_partial_image_event.ImageGenPartialImageEvent
class ImageGenPartialImageEvent(BaseModelResponse):
    """Streaming event emitted for partial images during generation."""

    b64_json: str = Field(description="Base64-encoded partial image data.")
    background: ImageBackgroundAuto = Field(
        description="Background setting for the requested image."
    )
    created_at: int = Field(
        ge=0, description="Unix timestamp when the event was created."
    )
    output_format: ImageOutputFormats = Field(
        description="Output format for the requested image."
    )
    partial_image_index: int = Field(
        ge=0, description="0-based index for the partial image (streaming)."
    )
    quality: ImageOutputQualityAuto = Field(
        description="Quality setting for the requested image."
    )
    size: str | None = Field(default=None, description="Size of the generated image.")
    type: Literal["image_generation.partial_image"] = Field(
        description="Event type. Always `image_generation.partial_image`."
    )


class _ImageBaseParams(BaseModelRequestWithExtra):
    """Request body for generating images."""

    model: str = Field(
        description="Model for image generation.", min_length=1, max_length=255
    )
    response_format: Literal["url", "b64_json"] = Field(
        default="url",
        description="Format for returned images: `url` or `b64_json`. "
        "URLs expire after 60 minutes. Streaming always returns `b64_json`.",
    )
    n: int = Field(default=1, description="Number of images to generate.", ge=1, le=10)
    size: str = Field(  # Support different values than OpenAI
        default="1024x1024",
        pattern=r"^(\d+)x(\d+)$",
        description="Size of the generated images. "
        "Supported values depend on the model; output size may differ.",
    )
    user: str | None = Field(
        default=None,
        description="User identifier for monitoring and abuse detection.",
        min_length=1,
        max_length=255,
    )

    @field_validator("response_format", mode="after")
    @classmethod
    def _validate_response_format(cls, value: str) -> str:
        """Ensure S3 bucket is configured for 'url' format.

        Args:
            value: The response format to validate.

        Returns:
            The validated response format.
        """
        if value == "url" and not SETTINGS.aws_s3_bucket:
            log_error_details(
                "No S3 bucket configured for presigned URLs. "
                "AWS_S3_BUCKET environment variable is not set."
            )
            msg = (
                "The url response format is not enabled on this server. "
                "Please contact the administrator to enabled it."
            )
            raise ValueError(msg)
        return value


# Ref: openai.types.image_generate_params.ImageGenerateParams
class ImageGenerateParams(_ImageBaseParams):
    """Request body for generating images."""

    prompt: str = Field(
        ..., description="A text description of the desired image(s).", min_length=1
    )
    background: ImageBackgroundAuto = Field(
        description="Background transparency setting. If `transparent`, `output_format` "
        "must be `png` or `webp`."
        "\ntransparent is UNSUPPORTED on this implementation.",
        default="auto",
    )
    moderation: Literal["low", "auto"] = Field(
        description="Content-moderation level: `low` (less restrictive) or `auto`."
        "\nlow is UNSUPPORTED on this implementation.",
        default="auto",
    )
    output_compression: int = Field(
        description="Compression level (0-100%) for generated images.",
        default=100,
        ge=1,
        le=100,
    )
    output_format: ImageOutputFormats | None = Field(
        default=None, description="Output image format: `png`, `jpeg`, or `webp`."
    )
    partial_images: int | None = Field(
        description="Number of partial images to generate during streaming "
        "(0-3; requires `stream=true`). 0 sends the final image as a single event. "
        "The final image may arrive before all partial images if generation finishes "
        "early, and partial images are only sent if the model supports them.",
        ge=0,
        le=3,
        default=None,
    )
    quality: str = Field(  # Support different values than OpenAI
        default="auto",
        description="Image quality. `auto` selects the best quality for the model; "
        "supported values depend on the model.",
        min_length=1,
        max_length=255,
    )
    style: str | None = Field(  # Support different values than OpenAI
        default=None,
        description="The style of the generated images; supported values depend on the model.",
        min_length=1,
        max_length=255,
    )
    stream: bool = Field(
        default=False, description="Generate the image in streaming mode."
    )

    @model_validator(mode="after")
    def _unsupported(self) -> Self:
        """Validate unsupported or incompatible options.

        Raises:
            ValueError: When an unsupported option is requested or options are incompatible.
        """
        if self.partial_images is not None and not self.stream:
            msg = "partial_images requires streaming mode."
            raise ValueError(msg)
        if self.background == "transparent":
            msg = "Background transparency is not supported on this backend."
            raise ValueError(msg)
        if self.moderation != "auto":
            msg = "The 'moderation' parameter is not supported on this backend."
            raise ValueError(msg)
        return self


# Ref: openai.types.image_edit_params.ImageEditParams
class _ImageEditCommonParams(_ImageBaseParams):
    """Shared parameters for image-editing requests (form and JSON body)."""

    # image/mask handled in route
    prompt: str = Field(
        description="A text description of the desired image(s). "
        "Required for a majority of models.",
        default="",
    )
    background: ImageBackgroundAuto = Field(
        description="Background transparency setting. If `transparent`, `output_format` "
        "must be `png` or `webp`."
        "\ntransparent is UNSUPPORTED on this implementation.",
        default="auto",
    )
    input_fidelity: ImageInputFidelity = Field(
        description="Effort level for matching the style and features (especially facial "
        "features) of input images."
        "\nUNSUPPORTED on this implementation.",
        default="low",
    )
    output_compression: int = Field(
        description="Compression level (0-100%) for generated images.",
        default=100,
        ge=1,
        le=100,
    )
    output_format: ImageOutputFormats | None = Field(
        default=None, description="Output image format: `png`, `jpeg`, or `webp`."
    )
    partial_images: int | None = Field(
        description="Number of partial images to generate during streaming "
        "(0-3; requires `stream=true`). 0 sends the final image as a single event. "
        "The final image may arrive before all partial images if generation finishes "
        "early, and partial images are only sent if the model supports them.",
        ge=0,
        le=3,
        default=None,
    )
    quality: str = Field(  # Support different values than OpenAI
        default="auto",
        description="Image quality. `auto` selects the best quality for the model; "
        "supported values depend on the model.",
        min_length=1,
        max_length=255,
    )
    stream: bool = Field(
        default=False, description="Generate the image in streaming mode."
    )

    @model_validator(mode="after")
    def _unsupported(self) -> Self:
        """Validate unsupported or incompatible options.

        Raises:
            ValueError: When an unsupported option is requested or options are incompatible.
        """
        if self.partial_images is not None and not self.stream:
            msg = "partial_images requires streaming mode."
            raise ValueError(msg)
        if self.background == "transparent":
            msg = "Background transparency is not supported on this backend."
            raise ValueError(msg)
        if self.input_fidelity != "low":
            msg = "The 'input_fidelity' parameter is not supported on this backend."
            raise ValueError(msg)
        return self


class ImageEditParams(_ImageEditCommonParams, BaseModelRequestWithFormExtra):
    """Request body for editing images (multipart/form-data)."""


class ImageEditJsonBody(_ImageEditCommonParams):
    """Request body for editing images (application/json).

    Uses a structured ``images`` array instead of binary file uploads,
    allowing references via Files API identifiers or HTTP/data URLs.
    """

    images: list[ImageInputReferenceParam] = Field(
        min_length=1,
        max_length=16,
        description="One or more input images to edit, each referenced by a "
        "Files API identifier (``file_*`` / ``file-*``) or an HTTP/data URL.",
    )
    mask: ImageInputReferenceParam | None = Field(
        default=None,
        description="An optional mask image indicating where edits should be applied, "
        "referenced by a Files API identifier or URL.",
    )


# Ref: openai.types.image_create_variation_params.ImageCreateVariationParams
class ImageVariationParams(_ImageBaseParams, BaseModelRequestWithFormExtra):
    """Request body for creating image variations (multipart/form-data)."""

    # image: handled in route


class ImageVariationJsonBody(_ImageBaseParams):
    """Request body for creating image variations (application/json).

    Uses a structured ``image`` field instead of a binary file upload,
    allowing reference via a Files API identifier or HTTP/data URL.
    """

    image: ImageInputReferenceParam = Field(
        description="The input image to create a variation of, referenced by a "
        "Files API identifier (``file_*`` / ``file-*``) or an HTTP/data URL."
    )
