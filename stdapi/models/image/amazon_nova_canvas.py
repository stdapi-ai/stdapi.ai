"""Amazon Nova Canvas image generation models.

- amazon.nova-canvas-v1:0
"""

from collections.abc import Awaitable, Iterable
from typing import Literal, NotRequired, TypedDict

from fastapi import HTTPException

from stdapi.models.image import (
    ImageGenerationJobBase,
    ImageGenerationResponse,
    ImageModelBase,
)
from stdapi.models.image.amazon_titan_image_generator import (
    AmzQuality,
    get_amz_quality,
    random_seed,
)


class _TextToImageParams(TypedDict):
    """Text-to-image parameters."""

    text: str  # Required: 1-1024 characters
    negativeText: NotRequired[str]  # Optional: 1-1024 characters
    style: NotRequired[
        Literal[
            "3D_ANIMATED_FAMILY_FILM",
            "DESIGN_SKETCH",
            "FLAT_VECTOR_ILLUSTRATION",
            "GRAPHIC_NOVEL_ILLUSTRATION",
            "MAXIMALISM",
            "MIDCENTURY_RETRO",
            "PHOTOREALISM",
            "SOFT_DIGITAL_PAINTING",
        ]
    ]
    # Image conditioning parameters
    conditionImage: NotRequired[str]  # Base64 encoded image
    controlMode: NotRequired[Literal["CANNY_EDGE", "SEGMENTATION"]]
    controlStrength: NotRequired[float]  # 0.0-1.0


class _ColorGuidedGenerationParams(TypedDict):
    """Color guided generation parameters."""

    colors: list[str]  # Required: list of hexadecimal color values (up to 10)
    referenceImage: NotRequired[str]  # Base64 encoded image
    text: str  # Required: 1-1024 characters
    negativeText: NotRequired[str]  # Optional: 1-1024 characters


class _ImageVariationParams(TypedDict):
    """Image variation parameters."""

    images: list[str]  # Required: 1-5 Base64 encoded images
    similarityStrength: NotRequired[float]  # 0.2-1.0
    text: str  # Required: 1-1024 characters
    negativeText: NotRequired[str]  # Optional: 1-1024 characters


class _InPaintingParams(TypedDict):
    """Inpainting parameters."""

    image: str  # Required: Base64 encoded image
    maskPrompt: NotRequired[str]  # Either maskPrompt or maskImage required
    maskImage: NotRequired[str]  # Base64 encoded mask image
    text: str  # Required: 1-1024 characters
    negativeText: NotRequired[str]  # Optional: 1-1024 characters


class _OutPaintingParams(TypedDict):
    """Outpainting parameters."""

    image: str  # Required: Base64 encoded image
    maskPrompt: NotRequired[str]  # Either maskPrompt or maskImage required
    maskImage: NotRequired[str]  # Base64 encoded mask image
    outPaintingMode: NotRequired[Literal["DEFAULT", "PRECISE"]]
    text: str  # Required: 1-1024 characters
    negativeText: NotRequired[str]  # Optional: 1-1024 characters


class _BackgroundRemovalParams(TypedDict):
    """Background removal parameters."""

    image: str  # Required: Base64 encoded image


class _ImageBasedMask(TypedDict):
    """Image based mask for virtual try-on."""

    maskImage: str  # Base64 encoded mask image


class _GarmentStyling(TypedDict):
    """Garment styling options for virtual try-on."""

    longSleeveStyle: NotRequired[Literal["SLEEVE_DOWN", "SLEEVE_UP"]]
    tuckingStyle: NotRequired[Literal["UNTUCKED", "TUCKED"]]
    outerLayerStyle: NotRequired[Literal["CLOSED", "OPEN"]]


class _GarmentBasedMask(TypedDict):
    """Garment based mask for virtual try-on."""

    maskShape: NotRequired[Literal["CONTOUR", "BOUNDING_BOX", "DEFAULT"]]
    garmentClass: NotRequired[
        Literal[
            "UPPER_BODY",
            "LOWER_BODY",
            "FULL_BODY",
            "FOOTWEAR",
            "LONG_SLEEVE_SHIRT",
            "SHORT_SLEEVE_SHIRT",
            "NO_SLEEVE_SHIRT",
            "OTHER_UPPER_BODY",
            "LONG_PANTS",
            "SHORT_PANTS",
            "OTHER_LOWER_BODY",
            "LONG_DRESS",
            "SHORT_DRESS",
            "FULL_BODY_OUTFIT",
            "OTHER_FULL_BODY",
            "SHOES",
            "BOOTS",
            "OTHER_FOOTWEAR",
        ]
    ]
    garmentStyling: NotRequired[_GarmentStyling]


class _PromptBasedMask(TypedDict):
    """Prompt based mask for virtual try-on."""

    maskShape: NotRequired[Literal["BOUNDING_BOX", "CONTOUR", "DEFAULT"]]
    maskPrompt: str


class _MaskExclusions(TypedDict):
    """Mask exclusions for virtual try-on."""

    preserveBodyPose: NotRequired[Literal["ON", "OFF", "DEFAULT"]]
    preserveHands: NotRequired[Literal["ON", "OFF", "DEFAULT"]]
    preserveFace: NotRequired[Literal["OFF", "ON", "DEFAULT"]]


class _VirtualTryOnParams(TypedDict):
    """Virtual try-on parameters."""

    sourceImage: str  # Required: Base64 encoded image
    referenceImage: str  # Required: Base64 encoded image
    maskType: Literal["IMAGE", "GARMENT", "PROMPT"]
    imageBasedMask: NotRequired[_ImageBasedMask]
    garmentBasedMask: NotRequired[_GarmentBasedMask]
    promptBasedMask: NotRequired[_PromptBasedMask]
    maskExclusions: NotRequired[_MaskExclusions]
    mergeStyle: NotRequired[Literal["BALANCED", "SEAMLESS", "DETAILED"]]
    returnMask: NotRequired[bool]


class _ImageGenerationConfig(TypedDict):
    """Image generation configuration."""

    width: NotRequired[int]  # 320-4096, divisible by 16, default 1024
    height: NotRequired[int]  # 320-4096, divisible by 16, default 1024
    quality: NotRequired[AmzQuality]  # default "standard"
    cfgScale: NotRequired[float]  # 1.1-10, default 6.5
    seed: NotRequired[int]  # 0-2,147,483,646, default 12
    numberOfImages: NotRequired[int]  # 1-5, default 1


class _Request(TypedDict):
    """Amazon Nova Canvas request parameters."""

    taskType: Literal[
        "TEXT_IMAGE",
        "COLOR_GUIDED_GENERATION",
        "IMAGE_VARIATION",
        "INPAINTING",
        "OUTPAINTING",
        "BACKGROUND_REMOVAL",
        "VIRTUAL_TRY_ON",
    ]

    # Task-specific parameters (only one should be used based on taskType)
    textToImageParams: NotRequired[_TextToImageParams]
    colorGuidedGenerationParams: NotRequired[_ColorGuidedGenerationParams]
    imageVariationParams: NotRequired[_ImageVariationParams]
    inPaintingParams: NotRequired[_InPaintingParams]
    outPaintingParams: NotRequired[_OutPaintingParams]
    backgroundRemovalParams: NotRequired[_BackgroundRemovalParams]
    virtualTryOnParams: NotRequired[_VirtualTryOnParams]

    # Common configuration (not used for BACKGROUND_REMOVAL)
    imageGenerationConfig: NotRequired[_ImageGenerationConfig]


class _Response(TypedDict):
    """Amazon Nova Canvas response parameters."""

    images: NotRequired[list[str]]  # List of Base64 encoded images
    maskImage: NotRequired[str]  # Base64 encoded mask image (when requested)
    error: NotRequired[str]  # Error message if content doesn't align with RAI policy


class _ImageGenerationJob(ImageGenerationJobBase["ImageModel"]):
    """Image generation job."""

    @staticmethod
    async def _get_image_from_response(
        image_base64: str, index: int
    ) -> ImageGenerationResponse:
        """Get image response from model response.

        Args:
            image_base64: Base64 image.
            index: Image index.

        Response:
            Image response.
        """
        return ImageGenerationResponse(image=image_base64, index=index)

    async def _generate_images(self) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Generate images from text prompt.

        Yields:
            Images.
        """
        request = _Request(
            taskType="TEXT_IMAGE",
            textToImageParams=_TextToImageParams(text=self._prompt),
            imageGenerationConfig=_ImageGenerationConfig(
                width=self._width,
                height=self._height,
                numberOfImages=self._count,
                seed=random_seed(),
            ),
        )
        if self._extra_params:
            if "textToImageParams" in self._extra_params:
                request["textToImageParams"].update(
                    self._extra_params["textToImageParams"]  # type:ignore[typeddict-item]
                )
            if "imageGenerationConfig" in self._extra_params:
                request["imageGenerationConfig"].update(
                    self._extra_params["imageGenerationConfig"]  # type:ignore[typeddict-item]
                )

        self._response_height = self._height
        self._response_width = self._width
        self._response_output_format = "png"

        amz_quality = get_amz_quality(self._quality)
        if amz_quality:
            request["imageGenerationConfig"]["quality"] = amz_quality
            self._response_quality = "high" if amz_quality == "premium" else "medium"

        if self._style:
            request["textToImageParams"]["style"] = self._style.upper()  # type: ignore[typeddict-item]

        response = await self._model.invoke(request)
        if "error" in response:
            raise HTTPException(status_code=400, detail=response["error"])
        return tuple(
            self._get_image_from_response(image, index)
            for index, image in enumerate(response["images"])
        )


class ImageModel(ImageModelBase[_Request, _Response, _ImageGenerationJob]):
    """Amazon Nova Canvas image model."""

    MATCHER = "amazon.nova-canvas"
    IMAGE_GENERATION_JOB_CLASS = _ImageGenerationJob
