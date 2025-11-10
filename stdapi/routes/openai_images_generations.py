"""OpenAI-compatible Images API implementation using AWS Bedrock.

This module implements the /v1/images/generations endpoint following the OpenAI API
specification, calling AWS Bedrock image generation models (e.g., Stability AI,
Amazon Nova Canvas) to generate images.
"""

from asyncio import create_task
from collections.abc import AsyncGenerator
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sse_starlette import EventSourceResponse, JSONServerSentEvent

from stdapi.auth import authenticate
from stdapi.aws_bedrock import get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.image import ImageGenerationJobBase, get_image_model
from stdapi.monitoring import (
    REQUEST_LOG,
    REQUEST_TIME,
    log_request_params,
    log_request_stream_event,
    log_response_params,
)
from stdapi.openai_exceptions import OpenaiError, OpenaiUnsupportedModelError
from stdapi.tokenizer import estimate_token_count
from stdapi.types.openai_images import (
    Image,
    ImageGenCompletedEvent,
    ImageGenerateParams,
    ImageGenPartialImageEvent,
    ImageOutputQuality,
    ImagesResponse,
    Usage,
    UsageInputTokensDetails,
)

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=["images", "openai"]
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


async def stream_generator(
    job: ImageGenerationJobBase[Any], partial_images: int | None, created: int
) -> AsyncGenerator[JSONServerSentEvent]:
    """Asynchronously generates a stream of server-sent events for image generation.

    This function consumes an image generation job, processes its output as it
    streams, and delivers defined JSON server-sent events to the caller.

    Args:
        job: The active image generation job to process the stream of image data.
        partial_images: Specifies the number of partial images to receive during image generation.
        created: A timestamp representing the creation time for the events.

    Yields:
        AsyncGenerator[JSONServerSentEvent, None]: An asynchronous generator that
            yields server-sent events (JSONServerSentEvent) containing either
            partial or final image data.
    """
    token_task = create_task(estimate_token_count(job.prompt))
    indexes: dict[int, int] = {}
    estimated_input_tokens: int | None = None
    async for result in job.generate_images_stream(partial_images=partial_images):
        if estimated_input_tokens is None:
            estimated_input_tokens = await token_task or 0
        usage = Usage(
            input_tokens=estimated_input_tokens,
            input_tokens_details=UsageInputTokensDetails(
                image_tokens=0, text_tokens=estimated_input_tokens
            ),
            output_tokens=job.count,
            total_tokens=estimated_input_tokens + job.count,
        )
        if result.partial:
            index = indexes[result.index] = indexes.get(result.index, 0) + 1
            yield JSONServerSentEvent(
                data=ImageGenPartialImageEvent(
                    type="image_generation.partial_image",
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
            yield JSONServerSentEvent(
                data=ImageGenCompletedEvent(
                    type="image_generation.completed",
                    b64_json=result.image,
                    created_at=created,
                    output_format=job.output_format,
                    size=f"{job.width}x{job.height}",
                    background="opaque",
                    quality=job.quality,
                    usage=usage,
                ).model_dump(mode="json", exclude_none=True)
            )
    if estimated_input_tokens is None:
        estimated_input_tokens = await token_task or 0
    usage = Usage(
        input_tokens=estimated_input_tokens,
        input_tokens_details=UsageInputTokensDetails(
            image_tokens=0, text_tokens=estimated_input_tokens
        ),
        output_tokens=job.count,
        total_tokens=estimated_input_tokens + job.count,
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
    summary="OpenAI - /v1/images/generations",
    description="Creates an image given a prompt.",
    response_description="Image generation response in OpenAI format",
    responses={
        200: {"description": "Images successfully generated."},
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
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
        HTTPException: With 404 if the model does not exist; 400 on unsupported
            options or invalid values.
    """
    log_request_params(request)
    if request.user:
        log = REQUEST_LOG.get()
        log["request_user_id"] = request.user

    try:
        await validate_model(
            request.model, input_modality="TEXT", output_modality="IMAGE"
        )
    except OpenaiUnsupportedModelError as Error:
        # This route does not return standard 404 error if invalid model.
        raise OpenaiError(Error.args[0]) from None

    width, height = map(int, request.size.split("x"))
    job = get_image_model(request.model).get_image_generation_job(
        prompt=request.prompt,
        count=request.n,
        width=width,
        height=height,
        quality=_OPENAI_QUALITY_LEVELS.get(request.quality, request.quality),
        style=request.style,
        output_format=request.output_format,
        output_compression=request.output_compression,
        is_url=request.response_format == "url" and not request.stream,
        extra_params=get_extra_model_parameters(request.model, request),
    )
    created = int(REQUEST_TIME.get().timestamp())

    # Handle streaming requests
    if request.stream:
        return EventSourceResponse(
            await log_request_stream_event(
                stream_generator(
                    job=job, partial_images=request.partial_images, created=created
                )
            )
        )

    # Handle non-streaming requests
    token_task = create_task(estimate_token_count(request.prompt))
    results = await job.generate_images()
    if request.response_format == "b64_json":
        images = [Image(b64_json=result.image) for result in results]
    else:
        images = [Image(url=result.image) for result in results]

    estimated_input_tokens = await token_task or 0
    return log_response_params(
        ImagesResponse(
            created=created,
            data=images,
            output_format=job.output_format,
            size=f"{job.width}x{job.height}",
            background="opaque",
            quality=job.quality,
            usage=Usage(
                input_tokens=estimated_input_tokens,
                input_tokens_details=UsageInputTokensDetails(
                    image_tokens=0, text_tokens=estimated_input_tokens
                ),
                output_tokens=request.n,
                total_tokens=estimated_input_tokens + request.n,
            ),
        ),
        exclude={"data"},
    )
