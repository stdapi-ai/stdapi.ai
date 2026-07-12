"""Luma AI Ray video generation models."""

from math import gcd
from typing import TYPE_CHECKING

from stdapi.api_errors import ApiError
from stdapi.models.video import ReferenceImage, VideoModelBase

if TYPE_CHECKING:
    from stdapi.types import JsonMapping

#: Aspect ratios supported by Luma Ray, keyed by reduced width:height fraction.
_ASPECT_RATIOS = {
    (1, 1): "1:1",
    (16, 9): "16:9",
    (9, 16): "9:16",
    (4, 3): "4:3",
    (3, 4): "3:4",
    (7, 3): "21:9",
    (3, 7): "9:21",
}

#: Resolution names keyed by the smaller frame dimension in pixels.
_RESOLUTIONS = {540: "540p", 720: "720p"}

#: Supported clip durations in seconds.
_DURATIONS = frozenset({5, 9})


class VideoModel(VideoModelBase):
    """Luma AI Ray video generation model."""

    MATCHER = "luma."
    DEFAULT_SECONDS = 5
    DEFAULT_SIZE = "1280x720"

    def build_generation_input(
        self,
        prompt: str,
        *,
        seconds: int,
        size: str,
        reference_image: ReferenceImage | None,
        extra_params: JsonMapping,
    ) -> JsonMapping:
        """Build the Luma Ray ``modelInput`` payload.

        The requested size is translated to the model's resolution and aspect
        ratio: the smaller dimension selects the resolution and the reduced
        width:height fraction the aspect ratio.

        Args:
            prompt: Text prompt describing the video.
            seconds: Video duration in seconds.
            size: Video size as "<width>x<height>".
            reference_image: Optional starting keyframe image.
            extra_params: Extra parameters merged into the payload.

        Returns:
            The ``modelInput`` payload for ``StartAsyncInvoke``.

        Raises:
            ApiError: On an unsupported duration or size.
        """
        if seconds not in _DURATIONS:
            msg = (
                f"'seconds' must be one of {sorted(_DURATIONS)} "
                f"for model '{self._model_id}'."
            )
            raise ApiError(msg)
        width, height = map(int, size.split("x"))
        divisor = gcd(width, height)
        aspect_ratio = _ASPECT_RATIOS.get((width // divisor, height // divisor))
        resolution = _RESOLUTIONS.get(min(width, height))
        if resolution is None or aspect_ratio is None:
            msg = (
                f"'size' {size} is not supported by model '{self._model_id}': "
                f"the smaller dimension must be one of {sorted(_RESOLUTIONS)} "
                f"and the aspect ratio one of {sorted(_ASPECT_RATIOS.values())}."
            )
            raise ApiError(msg)
        body: JsonMapping = {
            "prompt": prompt,
            "duration": f"{seconds}s",
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
        }
        if reference_image is not None:
            body["keyframes"] = {
                "frame0": {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": reference_image.media_type,
                        "data": reference_image.base64_data,
                    },
                }
            }
        for key, value in extra_params.items():
            body.setdefault(key, value)
        return body
