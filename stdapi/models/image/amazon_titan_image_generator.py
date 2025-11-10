"""Amazon Titan Image Generator models.

- amazon.titan-image-generator-v1
- amazon.titan-image-generator-v2:0
"""

from collections.abc import Awaitable, Iterable
from secrets import randbelow
from typing import Literal, NotRequired, TypedDict

from fastapi import HTTPException

from stdapi.models.image import (
    ImageGenerationJobBase,
    ImageGenerationResponse,
    ImageModelBase,
)
from stdapi.types.openai_images import ImageOutputQuality

AmzQuality = Literal["standard", "premium"]

AMZ_QUALITY_MAP: dict[ImageOutputQuality, AmzQuality] = {
    "low": "standard",
    "medium": "standard",
    "high": "premium",
}


def get_amz_quality(quality: ImageOutputQuality | str | None) -> AmzQuality | None:
    """Converts input image quality to the corresponding Amazon quality format.

    This function standardizes the given image output quality parameter to align
    with pre-defined Amazon quality mappings. It checks the input value against
    a mapping dictionary and returns the appropriate Amazon quality format. If
    the provided input does not match any value in the mapping, the original
    input is returned unaltered. The Bedrock API validates the final value.

    Args:
        quality: A string or ImageOutputQuality specifying the desired image
            quality. The value will be case-insensitively matched against the
            pre-defined Amazon quality map.

    Returns:
        The corresponding Amazon-specific quality format if found
            in the mapping or the original quality input if no match exists.
    """
    if quality is None:
        return None
    quality = quality.lower()
    return AMZ_QUALITY_MAP.get(quality, quality)  # type: ignore[no-any-return,call-overload]


def random_seed() -> int:
    """Generate a random seed value.

    Returns:
        Seed
    """
    return randbelow(2147483646)


class _TextToImageParams(TypedDict):
    """Text-to-image parameters."""

    text: str  # Required: The prompt text (1-512 characters)


class _ImageGenerationConfig(TypedDict):
    """Image generation configuration."""

    numberOfImages: NotRequired[int]  # 1-5, default 1
    quality: NotRequired[AmzQuality]  # default "standard"
    cfgScale: NotRequired[
        float
    ]  # 1.0-10.0, default 8.0 - how closely the model follows the prompt
    height: NotRequired[
        int
    ]  # 512, 768, 1024, 1152, 1216, 1344, 1536, 2048 (default 512)
    width: NotRequired[
        int
    ]  # 512, 768, 1024, 1152, 1216, 1344, 1536, 2048 (default 512)
    seed: NotRequired[int]  # 0-2147483646, default 42


class _Request(TypedDict):
    """Amazon Titan Image Generator request parameters."""

    taskType: Literal["TEXT_IMAGE"]  # Currently only TEXT_IMAGE is supported
    textToImageParams: _TextToImageParams
    imageGenerationConfig: NotRequired[_ImageGenerationConfig]


class _Response(TypedDict):
    """Amazon Titan Image Generator response parameters."""

    images: list[str]  # List of base64 encoded images


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
        if self._style is not None:
            raise HTTPException(
                status_code=400,
                detail='"style" parameter is not supported by this model.',
            )

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
        if self._extra_params and "imageGenerationConfig" in self._extra_params:
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

        return tuple(
            self._get_image_from_response(image, index)
            for index, image in enumerate((await self._model.invoke(request))["images"])
        )


class ImageModel(ImageModelBase[_Request, _Response, _ImageGenerationJob]):
    """Amazon Titan Image Generator model."""

    MATCHER = "amazon.titan-image-generator"
    IMAGE_GENERATION_JOB_CLASS = _ImageGenerationJob
