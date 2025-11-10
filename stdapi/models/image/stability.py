"""Stability AI image generation models.

- stability.stable-image-core-v1:1
- stability.sd3-5-large-v1:0
- stability.stable-image-ultra-v1:1
"""

from collections.abc import Awaitable, Iterable
from typing import Literal, NotRequired, TypedDict

from fastapi import HTTPException

from stdapi.models.image import (
    ImageGenerationJobBase,
    ImageGenerationResponse,
    ImageModelBase,
)

# Aspect ratios supported by the model
_AspectRatio = Literal[
    "16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"
]

# Formats supported by the model
_SUPPORTED_IMAGE_FORMATS = {"png", "jpeg"}

_ASPECT_RATIOS: dict[float, _AspectRatio] = {
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


class _Request(TypedDict):
    """Stability AI request parameters."""

    prompt: str  # Required: 0-10,000 characters
    aspect_ratio: NotRequired[_AspectRatio]
    style_preset: NotRequired[str]
    mode: NotRequired[Literal["text-to-image"]]  # Default: text-to-image
    output_format: NotRequired[Literal["JPEG", "PNG"]]  # Default: PNG
    seed: NotRequired[int]  # Range: 0 to 4294967295
    negative_prompt: NotRequired[str]  # Max: 10,000 characters


class _Response(TypedDict):
    """Stability AI response parameters."""

    images: list[str]  # List of base64 encoded images
    seeds: list[int]  # List of seeds used for generation
    finish_reasons: list[str | None]  # Finish reasons (null for success)


class _ImageGenerationJob(ImageGenerationJobBase["ImageModel"]):
    """Image generation job."""

    async def _generate_images(self) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Generate images from text prompt.

        Yields:
            Images.
        """
        if self._quality is not None:
            raise HTTPException(
                status_code=400,
                detail='"quality" parameter is not supported by this model.',
            )

        request = _Request(
            prompt=self._prompt,
            mode="text-to-image",
            aspect_ratio=self._get_aspect_ratio(self._width, self._height),
        )
        request.update(self._extra_params)  # type:ignore[typeddict-item]

        if self._style:
            request["style_preset"] = self._style
        if self._output_format:
            if self._output_format not in _SUPPORTED_IMAGE_FORMATS:
                self._response_output_format = "png"
            request["output_format"] = self._response_output_format  # type: ignore[typeddict-item]
        else:
            self._response_output_format = "jpeg"

        return tuple(
            self._get_image_from_response(request, index)
            for index in range(self._count)
        )

    async def _get_image_from_response(
        self, request: _Request, index: int
    ) -> ImageGenerationResponse:
        """Invoke the model to generate an image.

        Args:
            request: Model request.
            index: image index.

        Returns:
            Image data extracted from the provided response.

        Raises:
            HTTPException: Raised if any non-None finish reasons are present, indicating that
                the request was filtered.
        """
        response = await self._model.invoke(request)
        try:
            finish_reasons = response["finish_reasons"]
        except KeyError:
            pass
        else:
            reasons = tuple(reason for reason in finish_reasons if reason)
            if reasons:
                raise HTTPException(
                    status_code=400,
                    detail=f"Request was filtered: {', '.join(set(reasons))}",
                )
        return ImageGenerationResponse(image=response["images"][0], index=index)

    @staticmethod
    def _get_aspect_ratio(width: int, height: int) -> _AspectRatio:
        """Convert width/height to supported aspect ratio.

        Args:
            width: Image width.
            height: Image height.

        Returns:
            Closest supported aspect ratio.
        """
        ratio = width / height
        return _ASPECT_RATIOS[min(_ASPECT_RATIOS.keys(), key=lambda x: abs(x - ratio))]


class ImageModel(ImageModelBase[_Request, _Response, _ImageGenerationJob]):
    """Stability AI image model."""

    MATCHER = "stability"
    IMAGE_GENERATION_JOB_CLASS = _ImageGenerationJob
