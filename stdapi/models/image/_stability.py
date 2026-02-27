"""Stability AI base classes and shared functionality.

This module provides base classes and common utilities for all Stability AI models.
Do not import ImageModel from here - each model type has its own file with a specific
ImageModel and MATCHER.
"""

from typing import ClassVar, Literal, NotRequired, TypedDict

from stdapi.api_errors import ApiError
from stdapi.models.image import (
    ImageGenerationJobBase,
    ImageGenerationResponse,
    ImageModelBase,
)

# ============================================================================
# Common Constants
# ============================================================================

# Aspect ratios supported by text-to-image and image-to-image models
AspectRatio = Literal["16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"]

# Supported image formats per model type
NO_WEBP_FORMATS = {"png", "jpeg"}

ASPECT_RATIOS: dict[float, AspectRatio] = {
    16 / 9: "16:9",
    1.0: "1:1",
    21 / 9: "21:9",
    2 / 3: "2:3",
    3 / 2: "3:2",
    4 / 5: "4:5",
    5 / 4: "5:4",
    9 / 16: "9:16",
    9 / 21: "9:21",
}

# ============================================================================
# TypedDict Definitions (Request/Response Types)
# ============================================================================


class TextToImageRequest(TypedDict):
    """Request parameters for text-to-image and image-to-image generation."""

    prompt: str
    aspect_ratio: NotRequired[AspectRatio]
    mode: NotRequired[Literal["text-to-image", "image-to-image"]]
    output_format: NotRequired[str]
    seed: NotRequired[int]
    negative_prompt: NotRequired[str]
    image: NotRequired[str]
    strength: NotRequired[float]


class FastUpscaleRequest(TypedDict):
    """Request parameters for fast upscaling (no prompt)."""

    image: str
    output_format: NotRequired[Literal["jpeg", "png", "webp"]]


class CreativeUpscaleRequest(TypedDict):
    """Request parameters for creative/conservative upscaling (prompt-guided)."""

    prompt: str
    image: str
    output_format: NotRequired[Literal["jpeg", "png", "webp"]]
    negative_prompt: NotRequired[str]
    seed: NotRequired[int]
    creativity: NotRequired[float]


class InpaintRequest(TypedDict):
    """Request parameters for inpainting (fill masked regions)."""

    prompt: str
    image: str
    mask: NotRequired[str]
    output_format: NotRequired[Literal["jpeg", "png", "webp"]]
    negative_prompt: NotRequired[str]
    seed: NotRequired[int]
    grow_mask: NotRequired[int]


class OutpaintRequest(TypedDict):
    """Request parameters for outpainting (extend image borders)."""

    prompt: str
    image: str
    left: NotRequired[int]
    right: NotRequired[int]
    up: NotRequired[int]
    down: NotRequired[int]
    creativity: NotRequired[float]
    output_format: NotRequired[Literal["jpeg", "png", "webp"]]
    negative_prompt: NotRequired[str]
    seed: NotRequired[int]


class EraseRequest(TypedDict):
    """Request parameters for erasing objects with mask."""

    image: str
    mask: str
    output_format: NotRequired[Literal["jpeg", "png", "webp"]]
    grow_mask: NotRequired[int]


class SearchRecolorRequest(TypedDict):
    """Request parameters for search-and-recolor (automatic object recoloring)."""

    prompt: str
    image: str
    select_prompt: str
    output_format: NotRequired[Literal["jpeg", "png", "webp"]]
    negative_prompt: NotRequired[str]
    seed: NotRequired[int]
    grow_mask: NotRequired[int]


class SearchReplaceRequest(TypedDict):
    """Request parameters for search-and-replace (automatic object replacement)."""

    prompt: str
    image: str
    search_prompt: str
    output_format: NotRequired[Literal["jpeg", "png", "webp"]]
    negative_prompt: NotRequired[str]
    seed: NotRequired[int]
    grow_mask: NotRequired[int]


class RemoveBackgroundRequest(TypedDict):
    """Request parameters for automatic background removal."""

    image: str
    output_format: NotRequired[Literal["jpeg", "png", "webp"]]


class ControlRequest(TypedDict):
    """Request parameters for control-based generation."""

    prompt: str
    image: str
    control_strength: NotRequired[float]
    output_format: NotRequired[Literal["jpeg", "png", "webp"]]
    negative_prompt: NotRequired[str]
    seed: NotRequired[int]


class StyleGuideRequest(TypedDict):
    """Request parameters for style guide extraction and application."""

    prompt: str
    image: str
    output_format: NotRequired[Literal["jpeg", "png", "webp"]]
    negative_prompt: NotRequired[str]
    seed: NotRequired[int]
    fidelity: NotRequired[float]


class StyleTransferRequest(TypedDict):
    """Request parameters for style transfer between images."""

    prompt: str
    init_image: str
    style_image: str
    output_format: NotRequired[Literal["jpeg", "png", "webp"]]
    negative_prompt: NotRequired[str]
    seed: NotRequired[int]
    fidelity: NotRequired[float]


# Union type for all request types
Request = (
    TextToImageRequest
    | FastUpscaleRequest
    | CreativeUpscaleRequest
    | InpaintRequest
    | OutpaintRequest
    | EraseRequest
    | SearchRecolorRequest
    | SearchReplaceRequest
    | RemoveBackgroundRequest
    | ControlRequest
    | StyleGuideRequest
    | StyleTransferRequest
)


class Response(TypedDict):
    """Stability AI response parameters."""

    images: list[str]
    seeds: list[int]
    finish_reasons: list[str | None]


# ============================================================================
# Base Job Class
# ============================================================================


class StabilityImageGenerationJobBase(
    ImageGenerationJobBase["StabilityImageModelBase"]
):
    """Base class for Stability AI image generation jobs."""

    # Supported formats
    _OUTPUT_FORMATS: ClassVar[set[str]] = {"png", "jpeg", "webp"}

    async def _get_image_from_response(
        self, request: Request, index: int
    ) -> ImageGenerationResponse:
        """Invoke the model to generate an image.

        Args:
            request: Model request.
            index: Image index.

        Returns:
            Image data extracted from the response.

        Raises:
            ApiError: If request was filtered.
        """
        response = await self._model.invoke(request)
        try:
            finish_reasons = response["finish_reasons"]
        except KeyError:
            pass
        else:
            reasons = tuple(reason for reason in finish_reasons if reason)
            if reasons:
                msg = f"Request was filtered: {', '.join(set(reasons))}"
                raise ApiError(msg)
        return ImageGenerationResponse(image=response["images"][0], index=index)

    @staticmethod
    def _get_aspect_ratio(width: int, height: int) -> AspectRatio:
        """Convert width/height to supported aspect ratio.

        Args:
            width: Image width.
            height: Image height.

        Returns:
            Closest supported aspect ratio.
        """
        ratio = width / height
        return ASPECT_RATIOS[min(ASPECT_RATIOS.keys(), key=lambda x: abs(x - ratio))]

    def _finalize_request(self, request: Request) -> None:
        """Apply output format and extra params to request.

        Args:
            request: Request dictionary to finalize.
        """
        format_str = (
            self._output_format
            if self._output_format and self._output_format in self._OUTPUT_FORMATS
            else "png"
        )
        request["output_format"] = format_str
        self._response_output_format = format_str
        if self._extra_params:
            request.update(self._extra_params)  # type: ignore[typeddict-item]

    def _build_text_to_image_base_request(self) -> TextToImageRequest:
        """Build base request for text-to-image models."""
        self._validate_no_quality()
        self._validate_no_style()
        request: TextToImageRequest = {"prompt": self._prompt}
        self._finalize_request(request)
        return request


class StabilityImageModelBase(
    ImageModelBase[Request, Response, StabilityImageGenerationJobBase]
):
    """Base class for Stability AI image models.

    Subclasses should define:
    - MATCHER: string prefix or regex pattern for model ID matching
    - IMAGE_GENERATION_JOB_CLASS: specific job class for this model type
    """
