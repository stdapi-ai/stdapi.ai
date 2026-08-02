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

    __slots__ = ()

    async def _edit_image(
        self, images: list[str], mask: str | None
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Upscale images with fast 4x upscaling.

        The prompt is unused by this model, but the OpenAI API requires one, so
        its absence is not checked.
        """
        self._drop_unsupported_quality()
        self._validate_no_mask(mask)
        request: FastUpscaleRequest = {"image": self._get_one_image_from_list(images)}
        self._finalize_request(request)
        body = self._encode_request(request)
        return tuple(
            self._get_image_from_response(body, index) for index in range(self._count)
        )


class ImageModel(StabilityImageModelBase):
    """Stability AI fast upscale model."""

    __slots__ = ()

    MATCHER = compile_regex(r"^stability\.stable-fast-upscale-v\d+:\d+$")
    IMAGE_GENERATION_JOB_CLASS = _FastUpscaleJob
