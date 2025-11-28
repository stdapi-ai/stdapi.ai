"""Stability AI fast upscale model.

Supported Models:
- stability.stable-fast-upscale-v1:* (fast 4x upscaling)
"""

from re import compile as compile_regex
from typing import TYPE_CHECKING

from stdapi.models.image._stability import (
    FastUpscaleRequest,
    StabilityImageGenerationJobBase,
    StabilityImageModelBase,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable

    from stdapi.models.image import ImageGenerationResponse


class _FastUpscaleJob(StabilityImageGenerationJobBase):
    """Job for fast upscale model (no prompt)."""

    async def _edit_image(
        self, images: list[str], mask: str | None
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Upscale images with fast 4x upscaling.

        Do not check if there is no prompt.
        This is not used here, but is required for the OpenAI API.
        """
        self._validate_no_quality()
        self._validate_no_mask(mask)
        request: FastUpscaleRequest = {"image": self._get_one_image_from_list(images)}
        self._finalize_request(request)
        return tuple(
            self._get_image_from_response(request, index)
            for index in range(self._count)
        )


class ImageModel(StabilityImageModelBase):
    """Stability AI fast upscale model."""

    MATCHER = compile_regex(r"^stability\.stable-fast-upscale-v\d+:\d+$")
    IMAGE_GENERATION_JOB_CLASS = _FastUpscaleJob
