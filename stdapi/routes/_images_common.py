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


async def build_images_response(
    job: ImageGenerationJobBase[Any],
    results: Iterable[ImageGenerationResponse],
    response_format: str,
    image_count: int,
    text_tokens: int = 0,
    image_tokens: int = 0,
) -> ImagesResponse:
    """Build a standard ImagesResponse from job results.

    Args:
        job: The image generation or edit job.
        results: Iterable of image generation responses.
        response_format: Format for returned images ("url" or "b64_json").
        image_count: Number of images requested.
        text_tokens: Estimated input text tokens.
        image_tokens: Estimated input image tokens (for edits).

    Returns:
        ImagesResponse with all metadata and usage information.
    """
    if response_format == "b64_json":
        images = [Image(b64_json=result.image) for result in results]
    else:
        images = [Image(url=result.image) for result in results]

    total_input_tokens = text_tokens + image_tokens
    created = int(REQUEST_TIME.get().timestamp())

    return log_response_params(
        ImagesResponse(
            created=created,
            data=images,
            output_format=job.output_format,
            size=f"{job.width}x{job.height}",
            background="opaque",
            quality=job.quality,
            usage=Usage(
                input_tokens=total_input_tokens,
                input_tokens_details=UsageInputTokensDetails(
                    image_tokens=image_tokens, text_tokens=text_tokens
                ),
                output_tokens=image_count,
                total_tokens=total_input_tokens + image_count,
            ),
        ),
        exclude={"data"},
    )
