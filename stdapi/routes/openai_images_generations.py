"""OpenAI-compatible ``/v1/images/generations`` endpoint using AWS Bedrock."""

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends
from sse_starlette import EventSourceResponse, JSONServerSentEvent

from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.aws_bedrock import apply_guardrail_to_text, get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.capabilities import Capability, register_route_capability
from stdapi.models.image import (
    ImageGenerationJobBase,
    ImageGenerationResponse,
    get_image_model,
)
from stdapi.monitoring import (
    REQUEST_TIME,
    log_request_params,
    log_request_stream_event,
    log_response_params,
)
from stdapi.routes._images_common import build_images_response
from stdapi.types.openai_images import (
    ImageEditCompletedEvent,
    ImageEditPartialImageEvent,
    ImageGenCompletedEvent,
    ImageGenerateParams,
    ImageGenPartialImageEvent,
    ImageOutputQuality,
    ImagesResponse,
    Usage,
    UsageInputTokensDetails,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

register_route_capability(
    "openai_image_generation",
    f"{SETTINGS.openai_routes_prefix}/v1/images/generations",
    "TEXT",
    "IMAGE",
    Capability.IMAGE_GENERATION,
)

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=["Images", TAG_OPENAI]
)

#: Uniformize the OpenAI quality levels in only 3 levels
_OPENAI_QUALITY_LEVELS: dict[str, ImageOutputQuality | None] = {
    "standard": "medium",
    "hd": "high",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "auto": None,  # if not specified, use model default
}


def _stream_event_usage(
    job: ImageGenerationJobBase[Any], input_image_count: int
) -> Usage:
    """Build a usage report from the job's tokens billed so far.

    Mirrors :func:`stdapi.routes._images_common.build_images_response`: a
    reported ``None`` falls back to the input/output image counts, a
    reported ``0`` is kept as-is. Called after each completed image, so the
    last completed event of the stream reports the job's final total,
    matching the non-streaming path.

    Args:
        job: The image generation/edit job, holding live token counts.
        input_image_count: Number of input images (0 for text-to-image).

    Returns:
        Usage report reflecting tokens billed by the job so far.
    """
    input_tokens = (
        job.input_tokens if job.input_tokens is not None else input_image_count
    )
    output_tokens = job.output_tokens if job.output_tokens is not None else job.count
    image_tokens = min(input_image_count, input_tokens)
    return Usage(
        input_tokens=input_tokens,
        input_tokens_details=UsageInputTokensDetails(
            image_tokens=image_tokens, text_tokens=input_tokens - image_tokens
        ),
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


async def stream_generator(
    image_stream: AsyncGenerator[ImageGenerationResponse],
    job: ImageGenerationJobBase[Any],
    created: int,
    input_image_count: int = 0,
    *,
    edit: bool = False,
) -> AsyncGenerator[JSONServerSentEvent]:
    """Stream server-sent events for an image generation or editing request.

    Args:
        image_stream: Async generator yielding image generation responses.
        job: The image generation/edit job containing metadata (dimensions, format, etc).
        created: Unix timestamp used as the creation time for each event.
        input_image_count: Number of input images (0 for text-to-image).
        edit: Whether to emit the edits endpoint's ``image_edit.*`` event names
            instead of the generations endpoint's ``image_generation.*`` ones.

    Yields:
        JSONServerSentEvent containing partial image data or the final completed image.
    """
    partial_event, completed_event = (
        (ImageEditPartialImageEvent, ImageEditCompletedEvent)
        if edit
        else (ImageGenPartialImageEvent, ImageGenCompletedEvent)
    )
    indexes: dict[int, int] = {}
    usage: Usage | None = None
    async for result in image_stream:
        if result.partial:
            # 0-based, as the OpenAI event declares it.
            index = indexes[result.index] = indexes.get(result.index, -1) + 1
            yield JSONServerSentEvent(
                data=partial_event(
                    partial_image_index=index,
                    b64_json=result.image,
                    created_at=created,
                    output_format=job.output_format,
                    size=f"{job.width}x{job.height}",
                    background="opaque",
                    quality=job.quality,
                ).model_dump(mode="json", exclude_none=True)
            )
        else:
            # Built after this image completes: the job's token counts are
            # only final once every image has been generated.
            usage = _stream_event_usage(job, input_image_count)
            yield JSONServerSentEvent(
                data=completed_event(
                    b64_json=result.image,
                    created_at=created,
                    output_format=job.output_format,
                    size=f"{job.width}x{job.height}",
                    background="opaque",
                    quality=job.quality,
                    usage=usage,
                ).model_dump(mode="json", exclude_none=True)
            )
    log_response_params(
        {
            "created_at": created,
            "output_format": job.output_format,
            "size": f"{job.width}x{job.height}",
            "background": "opaque",
            "quality": job.quality,
            "usage": usage,
        }
    )


@router.post(
    "/images/generations",
    response_model=ImagesResponse,
    summary="Generate images from a text prompt (OpenAI format)",
    operation_id="openai_image_generation",
    description=(
        "Creates one or more images from a text prompt (OpenAI Images Generations API).\n\n"
        "Returns image URLs or base64-encoded data (`b64_json`). Supports streaming via SSE "
        "for incremental partial-image previews while generation is in progress (`stream=true`). "
        "Multiple images can be requested with the `n` parameter.\n\n"
        "**Find compatible models:** Call `search_models` with `route=openai_image_generation` "
        "to discover model IDs that support image generation."
    ),
    response_description="The response from the image generation endpoint.",
    responses={
        200: {"description": "Images successfully generated."},
        400: {"description": "Invalid request or unsupported parameters."},
        503: {
            "description": "The 'url' response format is not enabled on this server."
        },
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "url": {
                            "summary": "Return image URL",
                            "value": {
                                "prompt": "A watercolor of a fox in the woods",
                                "model": "amazon.nova-canvas-v1:0",
                                "response_format": "url",
                            },
                        },
                        "b64": {
                            "summary": "Return base64 data",
                            "value": {
                                "prompt": "A watercolor of a fox in the woods",
                                "model": "amazon.nova-canvas-v1:0",
                                "response_format": "b64_json",
                            },
                        },
                        "stream": {
                            "summary": "Streaming SSE",
                            "value": {
                                "prompt": "A watercolor of a fox in the woods",
                                "model": "amazon.nova-canvas-v1:0",
                                "stream": True,
                            },
                        },
                    }
                }
            }
        }
    },
    response_model_exclude_none=True,
)
async def create_images(
    request: ImageGenerateParams, _: Annotated[None, Depends(authenticate)] = None
) -> ImagesResponse | EventSourceResponse:
    """Generate images from text prompts.

    Args:
        request: Image generation parameters following OpenAI API.

    Returns:
        ImagesResponse containing generated image URLs or base64 data, or
        EventSourceResponse for streaming requests.

    Raises:
        ApiError: With 404 if the model does not exist; 400 on unsupported
            options or invalid values.
    """
    log_request_params(request, user_id=request.user)
    model_id = (
        await validate_model(
            request.model,
            input_modality="TEXT",
            output_modality="IMAGE",
            error_status=400,
        )
    ).id

    width, height = map(int, request.size.split("x"))
    job = get_image_model(model_id).get_image_generation_job(
        prompt=await apply_guardrail_to_text(request.prompt, source="INPUT"),
        count=request.n,
        width=width,
        height=height,
        quality=_OPENAI_QUALITY_LEVELS.get(request.quality, request.quality),
        style=request.style,
        output_format=request.output_format,
        output_compression=request.output_compression,
        is_url=request.response_format == "url" and not request.stream,
        extra_params=get_extra_model_parameters(model_id, request),
    )

    # Handle streaming requests
    if request.stream:
        return EventSourceResponse(
            await log_request_stream_event(
                stream_generator(
                    image_stream=job.generate_images_stream(
                        partial_images=request.partial_images
                    ),
                    job=job,
                    created=int(REQUEST_TIME.get().timestamp()),
                )
            )
        )

    # Handle non-streaming requests
    return await build_images_response(
        job=job,
        results=await job.generate_images(),
        response_format=request.response_format,
        output_image_count=request.n,
    )
