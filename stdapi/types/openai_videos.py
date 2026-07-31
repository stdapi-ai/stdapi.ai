"""OpenAI Videos API types."""

from typing import Literal

from pydantic import Field

from stdapi.input_file import InputFile
from stdapi.types import BaseModelRequestWithFormExtra, BaseModelResponse
from stdapi.types.openai import PaginatedListEnvelope

#: Regex pattern that a valid Videos API video ID must match.
VIDEO_ID_PATTERN: str = r"^video_[A-Za-z0-9_-]+$"

#: Video job lifecycle status.
VideoStatus = Literal["queued", "in_progress", "completed", "failed"]


class VideoCreateParams(BaseModelRequestWithFormExtra):
    """Video creation parameters shared by the JSON and multipart formats."""

    model: str = Field(
        description="The video generation model to use.", min_length=1, max_length=255
    )
    prompt: str = Field(
        description="Text prompt that describes the video to generate.", min_length=1
    )
    seconds: str | None = Field(
        default=None,
        description="Clip duration in seconds. Supported values depend on the "
        "model; defaults to the model's shortest duration.",
        pattern=r"^[0-9]+$",
    )
    size: str | None = Field(
        default=None,
        description="Output resolution as `<width>x<height>`. Supported values "
        "depend on the model.",
        pattern=r"^[1-9][0-9]*x[1-9][0-9]*$",
    )


class VideoCreateJsonBody(VideoCreateParams):
    """Video creation JSON body, with the reference image as a file reference."""

    input_reference: InputFile | None = Field(
        default=None,
        description="Optional image used as the video's first frame, as a "
        "base64 string, data URI, HTTPS URL, S3 URI, or Files API ID.",
    )


class VideoError(BaseModelResponse):
    """Error payload explaining why a video generation failed."""

    code: str = Field(description="Machine-readable error code.")
    message: str = Field(description="Human-readable description of the error.")


class Video(BaseModelResponse):
    """Video generation job (OpenAI Videos API)."""

    id: str = Field(description="Unique identifier of the video job.")
    object: Literal["video"] = Field(
        default="video", description="The object type, which is always `video`."
    )
    model: str = Field(description="The video generation model that produced the job.")
    status: VideoStatus = Field(description="Current lifecycle status of the job.")
    progress: int = Field(default=0, description="Approximate completion percentage.")
    created_at: int = Field(
        description="Unix timestamp (in seconds) when the job was created."
    )
    completed_at: int | None = Field(
        default=None,
        description="Unix timestamp (in seconds) when the job completed, if finished.",
    )
    expires_at: int | None = Field(
        default=None,
        description="Unix timestamp (in seconds) when the video will expire, "
        "when a retention period is configured on the server.",
    )
    seconds: str = Field(description="Duration of the generated clip in seconds.")
    size: str = Field(description="Resolution of the generated video.")
    prompt: str | None = Field(
        default=None, description="The prompt used to generate the video, when known."
    )
    remixed_from_video_id: str | None = Field(
        default=None,
        description="ID of the source video for remixes. Always `null` "
        "(remix is not available on this implementation).",
    )
    error: VideoError | None = Field(
        default=None, description="Error payload when generation failed."
    )


class VideoList(PaginatedListEnvelope):
    """Paginated list of video jobs returned by GET /v1/videos."""

    object: Literal["list"] = Field(
        default="list", description="The object type, which is always `list`."
    )
    data: list[Video] = Field(description="List of Video objects.")


class VideoDeleted(BaseModelResponse):
    """Video deletion confirmation."""

    id: str = Field(description="Identifier of the deleted video.")
    object: Literal["video.deleted"] = Field(
        default="video.deleted",
        description="The object type, which is always `video.deleted`.",
    )
    deleted: bool = Field(default=True, description="Whether the video was deleted.")
