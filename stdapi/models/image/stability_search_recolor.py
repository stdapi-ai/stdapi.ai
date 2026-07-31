"""Stability AI search recolor model.

Supported Models:
- stability.stable-image-search-recolor-v1:* (recolor objects by search)
"""

from re import compile as compile_regex
from typing import TYPE_CHECKING

from stdapi.api_errors import ApiError
from stdapi.models.image._stability import (
    SearchRecolorRequest,
    StabilityImageGenerationJobBase,
    StabilityImageModelBase,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable

    from stdapi.models.image import ImageGenerationResponse


class _SearchRecolorJob(StabilityImageGenerationJobBase):
    """Job for search-recolor model."""

    async def _edit_image(
        self, images: list[str], mask: str | None
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Recolor objects in image using search."""
        self._drop_unsupported_quality()
        self._validate_no_mask(mask)

        try:
            select_prompt = str(self._extra_params["select_prompt"])
        except KeyError as err:
            msg = '"select_prompt" parameter is required for this model.'
            raise ApiError(msg) from err

        request: SearchRecolorRequest = {
            "prompt": self._prompt,
            "image": self._get_one_image_from_list(images),
            "select_prompt": select_prompt,
        }
        self._finalize_request(request)

        return tuple(
            self._get_image_from_response(request, index)
            for index in range(self._count)
        )


class ImageModel(StabilityImageModelBase):
    """Stability AI search recolor model."""

    MATCHER = compile_regex(r"^stability\.stable-image-search-recolor-v\d+:\d+$")
    IMAGE_GENERATION_JOB_CLASS = _SearchRecolorJob
