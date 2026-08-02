"""Amazon Nova Reel video generation models."""

from typing import TYPE_CHECKING

from stdapi.api_errors import ApiError
from stdapi.models.video import ReferenceImage, VideoModelBase

if TYPE_CHECKING:
    from stdapi.types import JsonMapping

#: Only frame rate supported by Nova Reel.
_FPS = 24

#: Single-shot clip duration; longer videos use the automated multi-shot task.
_SINGLE_SHOT_SECONDS = 6

#: Longest multi-shot video duration.
_MAX_SECONDS = 120

#: Only video size supported by Nova Reel.
_SIZE = "1280x720"

#: Reference image formats accepted by Nova Reel, keyed by MIME type.
_IMAGE_FORMATS = {"image/png": "png", "image/jpeg": "jpeg"}


class VideoModel(VideoModelBase):
    """Amazon Nova Reel video generation model."""

    __slots__ = ()

    MATCHER = "amazon.nova-reel"
    DEFAULT_SECONDS = _SINGLE_SHOT_SECONDS
    DEFAULT_SIZE = _SIZE

    def build_generation_input(
        self,
        prompt: str,
        *,
        seconds: int,
        size: str,
        reference_image: ReferenceImage | None,
        extra_params: JsonMapping,
    ) -> JsonMapping:
        """Build the Nova Reel ``modelInput`` payload.

        Args:
            prompt: Text prompt describing the video.
            seconds: Video duration in seconds.
            size: Video size as "<width>x<height>".
            reference_image: Optional starting keyframe image.
            extra_params: Extra parameters merged into ``videoGenerationConfig``.

        Returns:
            The ``modelInput`` payload for ``StartAsyncInvoke``.

        Raises:
            ApiError: On an unsupported duration or size, or a reference image
                combined with a multi-shot duration or in an unsupported format.
        """
        # v1:0 is single-shot only; v1:1 and later versions add multi-shot.
        if self._model_id.endswith("v1:0"):
            if seconds != _SINGLE_SHOT_SECONDS:
                msg = (
                    f"'seconds' must be {_SINGLE_SHOT_SECONDS} "
                    f"for model '{self._model_id}'."
                )
                raise ApiError(msg)
        elif seconds % _SINGLE_SHOT_SECONDS or not (
            _SINGLE_SHOT_SECONDS <= seconds <= _MAX_SECONDS
        ):
            msg = (
                f"'seconds' must be a multiple of {_SINGLE_SHOT_SECONDS} between "
                f"{_SINGLE_SHOT_SECONDS} and {_MAX_SECONDS} "
                f"for model '{self._model_id}'."
            )
            raise ApiError(msg)
        if size != _SIZE:
            msg = f"'size' must be '{_SIZE}' for model '{self._model_id}'."
            raise ApiError(msg)
        config: JsonMapping = {
            "durationSeconds": seconds,
            "fps": _FPS,
            "dimension": size,
        }
        for key, value in extra_params.items():
            config.setdefault(key, value)

        if seconds != _SINGLE_SHOT_SECONDS:
            if reference_image is not None:
                msg = (
                    "'input_reference' is only supported for "
                    f"{_SINGLE_SHOT_SECONDS}-second videos with model "
                    f"'{self._model_id}'."
                )
                raise ApiError(msg)
            return {
                "taskType": "MULTI_SHOT_AUTOMATED",
                "multiShotAutomatedParams": {"text": prompt},
                "videoGenerationConfig": config,
            }

        params: JsonMapping = {"text": prompt}
        if reference_image is not None:
            if (image_format := _IMAGE_FORMATS.get(reference_image.media_type)) is None:
                msg = "'input_reference' must be a PNG or JPEG image."
                raise ApiError(msg)
            params["images"] = [
                {
                    "format": image_format,
                    "source": {"bytes": reference_image.base64_data},
                }
            ]
        return {
            "taskType": "TEXT_VIDEO",
            "textToVideoParams": params,
            "videoGenerationConfig": config,
        }
