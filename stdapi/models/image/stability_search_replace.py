"""Stability AI search replace model.

Supported Models:
- stability.stable-image-search-replace-v1:* (replace objects by search)
"""

from re import compile as compile_regex
from typing import TYPE_CHECKING

from stdapi.api_errors import ApiError
from stdapi.models.image._stability import (
    SearchReplaceRequest,
    StabilityImageGenerationJobBase,
    StabilityImageModelBase,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable

    from stdapi.models.image import ImageGenerationResponse


class _SearchReplaceJob(StabilityImageGenerationJobBase):
    """Job for search-replace model."""

    async def _edit_image(
        self, images: list[str], mask: str | None
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Replace objects in image using search."""
        self._drop_unsupported_quality()
        self._validate_no_mask(mask)

        try:
            search_prompt = str(self._extra_params["search_prompt"])
        except KeyError as err:
            msg = '"search_prompt" parameter is required for this model.'
            raise ApiError(msg) from err

        request: SearchReplaceRequest = {
            "prompt": self._prompt,
            "image": self._get_one_image_from_list(images),
            "search_prompt": search_prompt,
        }
        self._finalize_request(request)
        return tuple(
            self._get_image_from_response(request, index)
            for index in range(self._count)
        )


class ImageModel(StabilityImageModelBase):
    """Stability AI search replace model."""

    MATCHER = compile_regex(r"^stability\.stable-image-search-replace-v\d+:\d+$")
    IMAGE_GENERATION_JOB_CLASS = _SearchReplaceJob
