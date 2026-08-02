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
from stdapi.utils import alpha_mask_to_bw

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable

    from stdapi.models.image import ImageGenerationResponse


class _EraseJob(StabilityImageGenerationJobBase):
    """Job for erase model."""

    __slots__ = ()

    async def _edit_image(
        self, images: list[str], mask: str | None
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Erase objects from image."""
        self._drop_unsupported_quality()

        request: EraseRequest = {
            "image": self._get_one_image_from_list(images),
            "mask": mask,  # type: ignore[typeddict-item]
        }
        if mask:
            # Stability reads white as the edit (erase) region, the opposite of
            # Nova/Titan's black-marks-edit convention, so an OpenAI-style alpha
            # mask inverts.
            request["mask"] = await alpha_mask_to_bw(mask, invert=True)
        self._finalize_request(request)
        body = self._encode_request(request)
        return tuple(
            self._get_image_from_response(body, index) for index in range(self._count)
        )


class ImageModel(StabilityImageModelBase):
    """Stability AI erase model."""

    __slots__ = ()

    MATCHER = compile_regex(r"^stability\.stable-image-erase-object-v\d+:\d+$")
    IMAGE_GENERATION_JOB_CLASS = _EraseJob
