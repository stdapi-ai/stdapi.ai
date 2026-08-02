"""Stability AI inpaint model.

Supported Models:
- stability.stable-image-inpaint-v1:* (mask-based inpainting)
"""

from re import compile as compile_regex
from typing import TYPE_CHECKING

from stdapi.models.image._stability import (
    InpaintRequest,
    StabilityImageGenerationJobBase,
    StabilityImageModelBase,
)
from stdapi.utils import alpha_mask_to_bw

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable

    from stdapi.models.image import ImageGenerationResponse


class _InpaintJob(StabilityImageGenerationJobBase):
    """Job for inpaint model."""

    __slots__ = ()

    async def _edit_image(
        self, images: list[str], mask: str | None
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Inpaint image regions."""
        self._drop_unsupported_quality()

        request: InpaintRequest = {
            "prompt": self._prompt,
            "image": self._get_one_image_from_list(images),
        }
        if mask:
            # Stability reads white as the edit region, the opposite of Nova/Titan's
            # black-marks-edit convention, so an OpenAI-style alpha mask inverts.
            request["mask"] = await alpha_mask_to_bw(mask, invert=True)
        self._finalize_request(request)
        body = self._encode_request(request)
        return tuple(
            self._get_image_from_response(body, index) for index in range(self._count)
        )


class ImageModel(StabilityImageModelBase):
    """Stability AI inpaint model."""

    __slots__ = ()

    MATCHER = compile_regex(r"^stability\.stable-image-inpaint-v\d+:\d+$")
    IMAGE_GENERATION_JOB_CLASS = _InpaintJob
