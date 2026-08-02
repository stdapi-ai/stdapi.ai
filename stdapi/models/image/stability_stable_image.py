"""Stability AI "Stable Image" models.

Supported Models:
- stability.stable-image-core-v1
- stability.stable-image-ultra-v1
"""

from re import compile as compile_regex
from typing import TYPE_CHECKING, ClassVar

from stdapi.models.image._stability import StabilityImageModelBase
from stdapi.models.image.stability_stable_diffusion import TextToImageJob

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable

    from stdapi.models.image import ImageGenerationResponse


class StabilityCoreTextToImageJob(TextToImageJob):
    """Job for text-to-image models."""

    __slots__ = ()

    _OUTPUT_FORMATS: ClassVar[set[str]] = {"png", "jpeg"}

    async def _generate_images_from_text(
        self,
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Generate images from text prompt."""
        request = self._build_text_to_image_base_request()
        request["aspect_ratio"] = self._get_aspect_ratio(self._width, self._height)
        return tuple(
            self._get_image_from_response(request, index)
            for index in range(self._count)
        )


class ImageModel(StabilityImageModelBase):
    """Stability AI image models."""

    __slots__ = ()

    MATCHER = compile_regex(r"^stability\.stable-image-(?:core|ultra)-v\d+:\d+$")
    IMAGE_GENERATION_JOB_CLASS = StabilityCoreTextToImageJob
