"""Stability AI style transfer model.

Supported Models:
- stability.stable-style-transfer-v1:* (transfer style between images)
"""

from re import compile as compile_regex
from typing import TYPE_CHECKING

from stdapi.api_errors import ApiError
from stdapi.models.image._stability import (
    StabilityImageGenerationJobBase,
    StabilityImageModelBase,
    StyleTransferRequest,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable

    from stdapi.models.image import ImageGenerationResponse


class _StyleTransferJob(StabilityImageGenerationJobBase):
    """Job for style transfer model."""

    async def _edit_image(
        self, images: list[str], mask: str | None
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Transfer style from one image to another."""
        self._validate_no_quality()
        if mask is None:
            try:
                mask = str(self._extra_params["style_image"])
            except KeyError as err:
                msg = '"mask" parameter is required by this model (As style_image parameter).'
                raise ApiError(msg) from err

        request: StyleTransferRequest = {
            "prompt": self._prompt,
            "init_image": self._get_one_image_from_list(images),
            "style_image": mask,
        }
        self._finalize_request(request)
        return tuple(
            self._get_image_from_response(request, index)
            for index in range(self._count)
        )


class ImageModel(StabilityImageModelBase):
    """Stability AI style transfer model."""

    MATCHER = compile_regex(r"^stability\.stable-style-transfer-v\d+:\d+$")
    IMAGE_GENERATION_JOB_CLASS = _StyleTransferJob
