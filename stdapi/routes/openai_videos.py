"""OpenAI-compatible Videos API implementation using AWS Bedrock.

This module implements the /v1/videos endpoints following the OpenAI API
specification, calling AWS Bedrock video generation models (e.g., Amazon Nova
Reel, Luma Ray) through asynchronous invocations.

Video generation is asynchronous: creating a video returns a job object whose
ID encodes the Bedrock async invocation ARN, so job state lives entirely on
AWS and the server stays stateless. Poll the job until it completes, then
download the MP4 content.

Two request formats are supported for creation:
- ``multipart/form-data``: binary reference image upload via ``input_reference``.
- ``application/json``: ``input_reference`` as a base64 string, data URI, URL,
  S3 URI, or Files API ID.
"""

from base64 import urlsafe_b64decode, urlsafe_b64encode
from contextlib import suppress
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, Form, Path, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic_core import from_json, to_json

from stdapi.api_errors import ApiError
from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.aws_bedrock import apply_guardrail_to_text, get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.input_file import InputFile
from stdapi.models import validate_model
from stdapi.models.capabilities import Capability, register_route_capability
from stdapi.models.video import (
    VideoJob,
    delete_video_output,
    get_video_job,
    get_video_model,
    list_video_jobs,
    open_video_content,
    video_expires_at,
)
from stdapi.monitoring import REQUEST_TIME, log_request_params, log_response_params
from stdapi.types.openai_videos import (
    VIDEO_ID_PATTERN,
    Video,
    VideoCreateJsonBody,
    VideoCreateParams,
    VideoDeleted,
    VideoError,
    VideoList,
)
from stdapi.utils import validation_error_handler

if TYPE_CHECKING:
    from starlette.datastructures import FormData

register_route_capability(
    "openai_video_generation",
    f"{SETTINGS.openai_routes_prefix}/v1/videos",
    "TEXT",
    "VIDEO",
    Capability.VIDEO_GENERATION,
)

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=["Videos", TAG_OPENAI]
)

#: Reusable path annotation for the ``video_id`` path parameter.
_VideoId = Annotated[
    str, Path(description="The identifier of the video job.", pattern=VIDEO_ID_PATTERN)
]

#: Flattened form keys of the OpenAI SDK's object-form `input_reference`.
_REFERENCE_FORM_KEYS = ("input_reference[image_url]", "input_reference[file_id]")

#: Model fields and file parameters handled separately in the create route.
_KNOWN_PARAMS = set(VideoCreateParams.model_fields) | {
    "input_reference",
    *_REFERENCE_FORM_KEYS,
}


def _decode_form_extras(form_data: FormData) -> dict[str, Any]:
    """Collect extra model parameters from a multipart form, JSON-decoding values.

    Decoding restores typed values (e.g. ``"true"`` → ``True``) that model
    payloads expect; non-JSON strings are kept as-is.

    Args:
        form_data: Parsed multipart form data.

    Returns:
        Extra parameters keyed by form field name.
    """
    extras: dict[str, Any] = {}
    for key, value in form_data.items():
        if key in _KNOWN_PARAMS or not isinstance(value, str):
            continue
        extras[key] = value
        with suppress(ValueError):
            extras[key] = from_json(value)
    return extras


def _encode_video_id(invocation_arn: str, seconds: int, size: str) -> str:
    """Encode the job's invocation ARN and creation parameters into a video ID.

    Args:
        invocation_arn: Bedrock async invocation ARN.
        seconds: Video duration in seconds.
        size: Video size as "<width>x<height>".

    Returns:
        Opaque video ID.
    """
    payload = to_json({"arn": invocation_arn, "seconds": str(seconds), "size": size})
    return f"video_{urlsafe_b64encode(payload).decode().rstrip('=')}"


def _decode_video_id(video_id: str) -> tuple[str, str, str]:
    """Decode a video ID back into its invocation ARN and creation parameters.

    Args:
        video_id: Video ID returned by the create endpoint.

    Returns:
        Tuple of (invocation ARN, seconds, size).

    Raises:
        ApiError: 404 when the ID cannot be decoded.
    """
    data = video_id.removeprefix("video_")
    try:
        payload = from_json(urlsafe_b64decode(data + "=" * (-len(data) % 4)))
        return str(payload["arn"]), str(payload["seconds"]), str(payload["size"])
    except (ValueError, KeyError, TypeError) as exc:
        msg = f"Video with id '{video_id}' not found."
        raise ApiError(msg, status=404) from exc


def _to_video(video_id: str, job: VideoJob, seconds: str, size: str) -> Video:
    """Convert a job state to an OpenAI ``Video`` response.

    Args:
        video_id: Video ID of the job.
        job: Job state reported by AWS Bedrock.
        seconds: Video duration decoded from the video ID.
        size: Video size decoded from the video ID.

    Returns:
        Serialisable ``Video``.
    """
    return Video(
        id=video_id,
        model=job.model_id,
        status=job.status,
        progress=100 if job.status == "completed" else 0,
        created_at=job.created_at,
        completed_at=job.completed_at,
        expires_at=video_expires_at(job),
        seconds=seconds,
        size=size,
        error=VideoError(
            code="video_generation_failed",
            message=job.failure_message or "Video generation failed.",
        )
        if job.status == "failed"
        else None,
    )


@router.post(
    "/videos",
    summary="Start a video generation job (OpenAI format)",
    operation_id="openai_video_generation",
    description=(
        "Starts an asynchronous video generation job from a text prompt "
        "(OpenAI Videos API).\n\n"
        "Returns a `Video` job object immediately. Poll `openai_video_get` until "
        "its `status` is `completed`, then download the MP4 with "
        "`openai_video_content`.\n\n"
        "**Providing a reference image:** Use `multipart/form-data` to upload a "
        "binary `input_reference` first-frame image, or `application/json` to pass "
        "it as a base64 string, data URI, URL, S3 URI, or Files API ID.\n\n"
        "**Find compatible models:** Call `search_models` with "
        "`mcp_tool=openai_video_generation` to discover model IDs that support "
        "video generation."
    ),
    response_description="The created video job.",
    responses={
        200: {"description": "Video generation job started."},
        400: {"description": "Invalid request or unsupported parameters."},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "examples": {
                        "text": {
                            "summary": "Text to video",
                            "value": {
                                "model": "amazon.nova-reel-v1:1",
                                "prompt": "A calico cat playing a piano on stage",
                            },
                        }
                    }
                },
                "application/json": {
                    "examples": {
                        "text": {
                            "summary": "Text to video",
                            "value": {
                                "model": "amazon.nova-reel-v1:1",
                                "prompt": "A calico cat playing a piano on stage",
                                "seconds": "6",
                                "size": "1280x720",
                            },
                        },
                        "image": {
                            "summary": "Image to video (first frame)",
                            "value": {
                                "model": "luma.ray-v2:0",
                                "prompt": "The cat starts to play",
                                "input_reference": "https://example.com/frame.png",
                            },
                        },
                    }
                },
            }
        }
    },
    response_model_exclude_none=True,
)
async def create_video(
    http_request: Request,
    *,
    input_reference: Annotated[
        UploadFile | None,
        File(
            description="Optional image used as the video's first frame. "
            "Use an ``application/json`` body to pass a base64 string, data URI, "
            "URL, S3 URI, or Files API ID instead."
        ),
    ] = None,
    prompt: Annotated[
        str, Form(description="Text prompt that describes the video to generate.")
    ] = "",
    model: Annotated[str, Form(description="The video generation model to use.")] = "",
    seconds: Annotated[
        str | None,
        Form(
            description="Clip duration in seconds. Supported values depend on "
            "the model; defaults to the model's shortest duration."
        ),
    ] = None,
    size: Annotated[
        str | None,
        Form(
            description="Output resolution as `<width>x<height>`. Supported "
            "values depend on the model."
        ),
    ] = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> Video:
    """Start a video generation job.

    Accepts ``multipart/form-data`` (binary reference image upload) or
    ``application/json`` (reference image as base64, data URI, URL, S3 URI,
    or Files API ID in the ``input_reference`` field).

    Returns:
        Video job object with status ``queued``.

    Raises:
        ApiError: With 404 if the model does not exist; 400 on unsupported
            options or invalid values.
    """
    if "application/json" in http_request.headers.get("content-type", ""):
        with validation_error_handler():
            body = VideoCreateJsonBody.model_validate(await http_request.json())
        reference = body.input_reference
        request: VideoCreateParams = body
    else:
        form_data = await http_request.form()
        with validation_error_handler():
            request = VideoCreateParams(
                model=model,
                prompt=prompt,
                seconds=seconds,
                size=size,
                **_decode_form_extras(form_data),
            )
        reference = InputFile(input_reference) if input_reference else None
        if reference is None:
            # The SDK flattens {"file_id"|"image_url": ...} into bracketed keys.
            for key in _REFERENCE_FORM_KEYS:
                if isinstance(value := form_data.get(key), str) and value:
                    reference = InputFile(value)
                    break

    log_request_params(request)
    model_id = (
        await validate_model(
            request.model,
            input_modality="TEXT",
            output_modality="VIDEO",
            error_status=400,
        )
    ).id
    start = await get_video_model(model_id).start_video_generation(
        await apply_guardrail_to_text(request.prompt, source="INPUT"),
        seconds=int(request.seconds) if request.seconds else None,
        size=request.size,
        reference_image=reference,
        extra_params=get_extra_model_parameters(model_id, request),
    )
    return log_response_params(
        Video(
            id=_encode_video_id(start.invocation_arn, start.seconds, start.size),
            model=model_id,
            status="queued",
            created_at=int(REQUEST_TIME.get().timestamp()),
            seconds=str(start.seconds),
            size=start.size,
            prompt=request.prompt,
        ),
        exclude={"prompt"},
    )


@router.get(
    "/videos",
    summary="List video generation jobs (OpenAI format)",
    operation_id="openai_video_list",
    description=(
        "Returns a paginated list of video generation jobs, newest first by "
        "default (OpenAI Videos API).\n\n"
        "A job stays listed for as long as its generation record is "
        "retained, independently of whether its video content still exists."
    ),
    response_description="A paginated list of Video objects.",
    response_model_exclude_none=True,
)
async def list_videos(
    after: Annotated[
        str | None,
        Query(
            description=(
                "Cursor for pagination: the video ID to start after "
                "(the last ID from a previous page)."
            ),
            pattern=VIDEO_ID_PATTERN,
        ),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=100, description="A limit on the number of objects returned."),
    ] = 20,
    order: Annotated[
        Literal["asc", "desc"],
        Query(description="Sort order by the `created_at` timestamp of the objects."),
    ] = "desc",
    _: Annotated[None, Depends(authenticate)] = None,
) -> VideoList:
    """List video generation jobs across all configured regions.

    Args:
        after: Video ID cursor; only jobs strictly after it are returned.
        limit: Maximum number of jobs to return.
        order: Sort order by creation time.

    Returns:
        Paginated list of Video objects.

    Raises:
        ApiError: With 404 if the ``after`` cursor cannot be decoded.
    """
    log_request_params({"after": after, "limit": limit, "order": order})
    after_arn = _decode_video_id(after)[0] if after else None
    listings, has_more = await list_video_jobs(
        order=order, after_arn=after_arn, limit=limit
    )
    videos = [
        _to_video(
            _encode_video_id(listing.job.invocation_arn, listing.seconds, listing.size),
            listing.job,
            str(listing.seconds),
            listing.size,
        )
        for listing in listings
    ]
    return log_response_params(
        VideoList(
            data=videos,
            has_more=has_more,
            first_id=videos[0].id if videos else None,
            last_id=videos[-1].id if videos else None,
        )
    )


@router.get(
    "/videos/{video_id}",
    summary="Retrieve a video generation job (OpenAI format)",
    operation_id="openai_video_get",
    description=(
        "Returns the current state of a video generation job (OpenAI Videos API). "
        "Poll this endpoint until `status` is `completed`, then download the MP4 "
        "with `openai_video_content`."
    ),
    response_description="The video job.",
    responses={
        200: {"description": "The video job state."},
        404: {"description": "Video not found."},
    },
    response_model_exclude_none=True,
)
async def retrieve_video(
    video_id: _VideoId, _: Annotated[None, Depends(authenticate)] = None
) -> Video:
    """Retrieve the current state of a video generation job.

    Args:
        video_id: Video job identifier.

    Returns:
        Video job object.

    Raises:
        ApiError: With 404 if the video does not exist.
    """
    log_request_params({"video_id": video_id})
    invocation_arn, seconds, size = _decode_video_id(video_id)
    job = await get_video_job(invocation_arn)
    return log_response_params(_to_video(video_id, job, seconds, size))


@router.get(
    "/videos/{video_id}/content",
    summary="Download the generated video content (OpenAI format)",
    operation_id="openai_video_content",
    description=(
        "Returns the generated MP4 video as a streaming download once the job "
        "has completed (OpenAI Videos API)."
    ),
    response_description="The raw MP4 video content.",
    responses={
        200: {"description": "The MP4 video content."},
        400: {"description": "Requested variant not available."},
        404: {"description": "Video not found or not ready yet."},
    },
)
async def get_video_content(
    video_id: _VideoId,
    variant: Annotated[
        Literal["video", "thumbnail", "spritesheet"],
        Query(description="Which downloadable asset to return."),
    ] = "video",
    _: Annotated[None, Depends(authenticate)] = None,
) -> StreamingResponse:
    """Stream the MP4 content of a completed video generation job.

    Args:
        video_id: Video job identifier.
        variant: Downloadable asset to return; only "video" is available.

    Returns:
        StreamingResponse with the MP4 video bytes.

    Raises:
        ApiError: With 404 if the video does not exist or is not completed
            yet; 400 when the variant is not available.
    """
    log_request_params({"video_id": video_id, "variant": variant})
    job = await get_video_job(_decode_video_id(video_id)[0])
    if variant != "video":
        msg = "Only the 'video' variant is available on this implementation."
        raise ApiError(msg)
    if job.status != "completed":
        # Upstream semantics: content of an unfinished job is a 404.
        msg = "Video is not ready yet, use GET /v1/videos/{video_id} to check status"
        raise ApiError(msg, status=404)
    return StreamingResponse(await open_video_content(job), media_type="video/mp4")


@router.delete(
    "/videos/{video_id}",
    summary="Delete a generated video (OpenAI format)",
    operation_id="openai_video_delete",
    description=(
        "Permanently deletes the stored output of a finished video generation "
        "job and returns a deletion confirmation (OpenAI Videos API)."
    ),
    response_description="Deletion status.",
    responses={
        200: {"description": "Video output deleted."},
        400: {"description": "Video is still being processed."},
        404: {"description": "Video not found."},
    },
    response_model_exclude_none=True,
)
async def delete_video(
    video_id: _VideoId, _: Annotated[None, Depends(authenticate)] = None
) -> VideoDeleted:
    """Delete the stored output of a video generation job.

    Args:
        video_id: Video job identifier.

    Returns:
        VideoDeleted confirmation.

    Raises:
        ApiError: With 404 if the video does not exist; 400 while generation
            is still in progress (Bedrock jobs cannot be cancelled).
    """
    log_request_params({"video_id": video_id})
    job = await get_video_job(_decode_video_id(video_id)[0])
    if job.status == "in_progress":
        # Upstream message; Bedrock async invocations cannot be cancelled.
        msg = "Video is still being processed"
        raise ApiError(msg)
    await delete_video_output(job)
    return log_response_params(VideoDeleted(id=video_id))
