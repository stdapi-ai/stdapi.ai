"""Stability AI "Stable diffusion" models.

Supported Models:
- stability.sd3-5-large-v1
"""

from typing import TYPE_CHECKING

from stdapi.models.image import DEFAULT_VARIATION_PROMPT
from stdapi.models.image._stability import (
    StabilityImageGenerationJobBase,
    StabilityImageModelBase,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable

    from stdapi.models.image import ImageGenerationResponse


class TextToImageJob(StabilityImageGenerationJobBase):
    """Job for text-to-image and image-to-image generation."""

    __slots__ = ("_prompt",)

    _DEFAULT_STRENGTH = 0.35  # Use the same default value as Stable Image Ultra

    async def _generate_images_from_text(
        self,
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Generate images from text prompt."""
        request = self._build_text_to_image_base_request()
        request["aspect_ratio"] = self._get_aspect_ratio(self._width, self._height)
        body = self._encode_request(request)
        return tuple(
            self._get_image_from_response(body, index) for index in range(self._count)
        )

    async def _edit_image(
        self, images: list[str], mask: str | None
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Edit images.

        Args:
            images: List of base64-encoded source images.
            mask: Base64-encoded mask image (optional).

        Returns:
            Iterable of awaitable image generation responses.
        """
        self._drop_unsupported_quality()
        self._validate_no_mask(mask)

        request = self._build_text_to_image_base_request()
        request["mode"] = "image-to-image"
        request["image"] = self._get_one_image_from_list(images)
        request.setdefault("strength", self._DEFAULT_STRENGTH)
        body = self._encode_request(request)
        return tuple(
            self._get_image_from_response(body, index) for index in range(self._count)
        )

    async def _create_image_variations(
        self, images: list[str]
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Create variations of existing images.

        Args:
            images: List of base64-encoded source images.

        Returns:
            Iterable of awaitable image generation responses.
        """
        self._prompt = DEFAULT_VARIATION_PROMPT
        request = self._build_text_to_image_base_request()
        request["mode"] = "image-to-image"
        request["image"] = self._get_one_image_from_list(images)
        request.setdefault("strength", self._DEFAULT_STRENGTH)
        body = self._encode_request(request)
        return tuple(
            self._get_image_from_response(body, index) for index in range(self._count)
        )


class ImageModel(StabilityImageModelBase):
    """Stability AI Stable diffusion model."""

    __slots__ = ()

    MATCHER = "stability.sd"
    IMAGE_GENERATION_JOB_CLASS = TextToImageJob
