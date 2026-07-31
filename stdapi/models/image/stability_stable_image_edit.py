"""Stability AI simple edit models with identical logic.

These models share the same implementation: validate quality/mask, build request with prompt+image.

Supported Models:
- stability.stable-image-control-*-v1:* (sketch-to-image, structure-preserving)
- stability.stable-image-style-guide-v1:* (extract and apply style)
- stability.stable-outpaint-v1:* (extend image beyond borders)
- stability.stable-creative-upscale-v1:* (prompt-guided creative upscaling)
- stability.stable-conservative-upscale-v1:* (detail-preserving upscaling)
"""

from re import compile as compile_regex
from typing import TYPE_CHECKING

from stdapi.models.image._stability import (
    ControlRequest,
    CreativeUpscaleRequest,
    OutpaintRequest,
    StabilityImageGenerationJobBase,
    StabilityImageModelBase,
    StyleGuideRequest,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable

    from stdapi.models.image import ImageGenerationResponse


class _SimpleEditJob(StabilityImageGenerationJobBase):
    """Job for simple edit models (control, style-guide, outpaint, upscale)."""

    async def _edit_image(
        self, images: list[str], mask: str | None
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Generate/edit images with prompt and image."""
        self._drop_unsupported_quality()
        self._validate_no_mask(mask)

        request: (
            ControlRequest
            | StyleGuideRequest
            | OutpaintRequest
            | CreativeUpscaleRequest
        ) = {"prompt": self._prompt, "image": self._get_one_image_from_list(images)}
        self._finalize_request(request)
        return tuple(
            self._get_image_from_response(request, index)
            for index in range(self._count)
        )


class ImageModel(StabilityImageModelBase):
    """Stability AI simple edit models (control, style-guide, outpaint, upscale)."""

    MATCHER = compile_regex(
        r"^stability\.stable-(?:image-(?:control-(?:sketch|structure)|style-guide)|(?:creative|conservative)-upscale|outpaint)-v\d+:\d+$"
    )
    IMAGE_GENERATION_JOB_CLASS = _SimpleEditJob
