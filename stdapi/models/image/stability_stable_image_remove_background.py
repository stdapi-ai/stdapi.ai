"""Stability AI remove background model.

Supported Models:
- stability.stable-image-remove-background-v1:* (automatic background removal)
"""

from re import compile as compile_regex
from typing import TYPE_CHECKING

from stdapi.models.image._stability import (
    RemoveBackgroundRequest,
    StabilityImageGenerationJobBase,
    StabilityImageModelBase,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable

    from stdapi.models.image import ImageGenerationResponse


class _RemoveBackgroundJob(StabilityImageGenerationJobBase):
    """Job for remove background model."""

    async def _edit_image(
        self, images: list[str], mask: str | None
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Remove background from image."""
        self._drop_unsupported_quality()
        self._validate_no_mask(mask)

        request: RemoveBackgroundRequest = {
            "image": self._get_one_image_from_list(images)
        }
        self._finalize_request(request)
        return tuple(
            self._get_image_from_response(request, index)
            for index in range(self._count)
        )


class ImageModel(StabilityImageModelBase):
    """Stability AI remove background model."""

    MATCHER = compile_regex(r"^stability\.stable-image-remove-background-v\d+:\d+$")
    IMAGE_GENERATION_JOB_CLASS = _RemoveBackgroundJob
