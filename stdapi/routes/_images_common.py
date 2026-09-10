"""Common utilities for image generation and editing endpoints."""

from typing import TYPE_CHECKING, Any

from stdapi.monitoring import REQUEST_TIME, log_response_params
from stdapi.types.openai_images import (
    Image,
    ImagesResponse,
    Usage,
    UsageInputTokensDetails,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from stdapi.models.image import ImageGenerationJobBase, ImageGenerationResponse


def image_usage(job: ImageGenerationJobBase[Any], input_image_count: int) -> Usage:
    """Report the tokens the job has been billed so far.

    Called both once per completed image while streaming and once for the whole
    response otherwise, so both paths report the same figures.

    Args:
        job: The image generation, edit or variation job.
        input_image_count: Number of input images (0 for text-to-image).

    Returns:
        Usage whose details split ``input_tokens`` into image tokens (capped at
        the input image count) and remaining text tokens.
    """
    # A reported 0 is a real value; only substitute counts when unreported.
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


async def build_images_response(
    job: ImageGenerationJobBase[Any],
    results: Iterable[ImageGenerationResponse],
    response_format: str,
    input_image_count: int = 0,
) -> ImagesResponse:
    """Build a standard ImagesResponse from job results.

    Args:
        job: The image generation or edit job.
        results: Iterable of image generation responses.
        response_format: Format for returned images ("url" or "b64_json").
        input_image_count: Number of input images.

    Returns:
        ImagesResponse whose usage details split ``input_tokens`` into image
        tokens (capped at the input image count) and remaining text tokens.
    """
    if response_format == "b64_json":
        images = [Image(b64_json=result.image) for result in results]
    else:
        images = [Image(url=result.image) for result in results]

    return log_response_params(
        ImagesResponse(
            created=int(REQUEST_TIME.get().timestamp()),
            data=images,
            output_format=job.output_format,
            size=f"{job.width}x{job.height}",
            background="opaque",
            quality=job.quality,
            usage=image_usage(job, input_image_count),
        ),
        exclude={"data"},
    )
