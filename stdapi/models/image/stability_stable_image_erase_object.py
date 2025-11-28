"""Stability AI erase model.

Supported Models:
- stability.stable-image-erase-object-v1:* (remove objects with mask)
"""

from re import compile as compile_regex
from typing import TYPE_CHECKING

from stdapi.models.image._stability import (
    EraseRequest,
    StabilityImageGenerationJobBase,
    StabilityImageModelBase,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable

    from stdapi.models.image import ImageGenerationResponse


class _EraseJob(StabilityImageGenerationJobBase):
    """Job for erase model."""

    async def _edit_image(
        self, images: list[str], mask: str | None
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Erase objects from image."""
        self._validate_no_quality()

        request: EraseRequest = {
            "image": self._get_one_image_from_list(images),
            "mask": mask,  # type: ignore[typeddict-item]
        }
        self._finalize_request(request)
        return tuple(
            self._get_image_from_response(request, index)
            for index in range(self._count)
        )


class ImageModel(StabilityImageModelBase):
    """Stability AI erase model."""

    MATCHER = compile_regex(r"^stability\.stable-image-erase-object-v\d+:\d+$")
    IMAGE_GENERATION_JOB_CLASS = _EraseJob
