"""Image generation models base classes and dynamic registry.

This package exposes the base interfaces for image generation models and provides a
minimal plugin/registry system that auto-loads model implementations located in
this package directory and resolves them by matching the OpenAI/Bedrock model
identifier.

Design:
- Model modules expose a class named `ImageGenerationModel` with a class variable
  `MATCHER` containing a string prefix or compiled regex matching model
  identifiers.
- The package auto-loads and registers these classes once on import.
"""

from abc import ABC, abstractmethod
from asyncio import Lock, as_completed, gather
from collections.abc import AsyncGenerator, Awaitable, Iterable
from typing import Any, TypeVar

from pydantic import BaseModel, JsonValue

from stdapi.aws_s3 import put_object_and_get_url
from stdapi.models import ModelBase, get_model, load_model_plugins
from stdapi.monitoring import REQUEST_ID
from stdapi.types.openai_images import ImageOutputFormats, ImageOutputQuality
from stdapi.utils import b64decode, convert_base64_image, get_base64_image_size


class ImageGenerationResponse(BaseModel):
    """Image generation response.

    Attributes:
        images: base64 encoded image.
        partial: true if partial image.
    """

    image: str
    partial: bool = False
    index: int


ImageModelT = TypeVar("ImageModelT", bound="ImageModelBase[Any, Any, Any]")


class ImageGenerationJobBase[ImageModelT: "ImageModelBase[Any, Any, Any]"](ABC):
    """Image generation job base class."""

    __slots__ = (
        "_count",
        "_extra_params",
        "_height",
        "_is_url",
        "_model",
        "_output_compression",
        "_output_format",
        "_prompt",
        "_quality",
        "_response_height",
        "_response_output_format",
        "_response_quality",
        "_response_width",
        "_size_lock",
        "_style",
        "_width",
    )

    def __init__(
        self,
        model: "ImageModelT",
        prompt: str,
        count: int,
        width: int,
        height: int,
        quality: str | None,
        style: str | None,
        output_format: ImageOutputFormats | None,
        output_compression: int,
        extra_params: dict[str, JsonValue],
        *,
        is_url: bool = False,
    ) -> None:
        """Initialize job.

        Args:
            model: model to use for generating the images.
            prompt: Text prompt for image generation.
            count: Number of images to generate.
            width: Image width.
            height: Image height.
            quality: Image quality level.
            style: Image style.
            output_format: Output format.
            output_compression: Compression level for output images (0-100).
            extra_params: Extra model parameters.
            is_url: If True, return image URL instead of base64 image.
        """
        self._model = model
        self._prompt = prompt
        self._count = count
        self._width = width
        self._height = height
        self._quality = quality
        self._style = style
        self._output_format = output_format
        self._output_compression = output_compression
        self._is_url = is_url
        self._extra_params = extra_params

        # Image format from model, before conversion
        self._response_output_format: ImageOutputFormats = "png"

        # Real image size from model response
        self._response_width = 0
        self._response_height = 0
        self._size_lock = Lock()

        # Real image quality from model response
        self._response_quality: ImageOutputQuality = "medium"

    @property
    def prompt(self) -> str:
        """Input prompt."""
        return self._prompt

    @property
    def count(self) -> int:
        """Request image count."""
        return self._count

    @property
    def width(self) -> int:
        """Final image width."""
        return self._response_width

    @property
    def height(self) -> int:
        """Final image height."""
        return self._response_height

    @property
    def quality(self) -> ImageOutputQuality:
        """Final image quality."""
        return self._response_quality

    @property
    def output_format(self) -> ImageOutputFormats:
        """Final image quality."""
        return self._output_format or self._response_output_format

    async def generate_images(self) -> Iterable[ImageGenerationResponse]:
        """Generate images from text prompt.

        Yields:
            Images.
        """
        return await gather(
            *(
                self._ensure_image_output_format(result)
                for result in await self._generate_images()
            )
        )

    async def generate_images_stream(
        self, partial_images: int | None = None
    ) -> AsyncGenerator[ImageGenerationResponse]:
        """Generate images from text prompt.

        Args:
            partial_images: Number of partial images to generate during streaming.

        Yields:
            Images.
        """
        async for result in self._generate_images_stream(partial_images):
            yield await self._ensure_image_output_format(result)

    @abstractmethod
    async def _generate_images(self) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Generate images from text prompt.

        Yields:
            Images.
        """

    async def _generate_images_stream(
        self,
        partial_images: int | None = None,  # noqa:ARG002
    ) -> AsyncGenerator[ImageGenerationResponse]:
        """Stream generated images from text prompt.

        Args:
            partial_images: Number of partial images to generate during streaming.

        Yields:
            Streamed images.
        """
        for result in as_completed(
            self._ensure_image_output_format(result)
            for result in await self._generate_images()
        ):
            yield await result

    async def _ensure_image_output_format(
        self, response: Awaitable[ImageGenerationResponse] | ImageGenerationResponse
    ) -> ImageGenerationResponse:
        """Ensures that the output format matches the desired format.

        Args:
            response: input image awaitable.
        """
        # Get image
        image = (
            response
            if isinstance(response, ImageGenerationResponse)
            else await response
        )

        # Convert image if not in excepted format
        if (
            self._output_format is not None
            and self._response_output_format != self._output_format
        ):
            result = await convert_base64_image(
                image.image,
                output_format=self._output_format,
                compression=self._output_compression,
            )
            image.image = result[0]
            if self._response_width == 0 or self._response_height == 0:
                self._response_width = result[1]
                self._response_height = result[2]

        # Get size from the image if unknown
        else:
            async with self._size_lock:
                if self._response_width == 0 or self._response_height == 0:
                    (
                        self._response_width,
                        self._response_height,
                    ) = await get_base64_image_size(image.image)

        if self._is_url:
            image.image = await self._get_image_url(
                image.image, index=image.index, output_format=self.output_format
            )

        return image

    @staticmethod
    async def _get_image_url(
        image_data: str, index: int, output_format: ImageOutputFormats
    ) -> str:
        """Upload base64 image data to S3 and return presigned download URL.

        Args:
            image_data: Base64 encoded image data.
            index: Unique image index to use.
            output_format: Image format (png, jpeg, webp).

        Returns:
            Presigned download URL valid for 1 hour.

        Raises:
            HTTPException: If S3 operations fail.
        """
        request_id = REQUEST_ID.get()
        return await put_object_and_get_url(
            await b64decode(image_data),
            f"image/{output_format}",
            f"{request_id}/image-{request_id}-{index + 1:03d}.{'jpg' if output_format == 'jpeg' else output_format}",
        )


ImageGenerationJobT = TypeVar("ImageGenerationJobT", bound=ImageGenerationJobBase[Any])


class ImageModelBase[RequestT, ResponseT, ImageGenerationJobT](
    ModelBase[RequestT, ResponseT]
):
    """Base class for provider-specific image models."""

    IMAGE_GENERATION_JOB_CLASS: type[ImageGenerationJobT]

    def get_image_generation_job(
        self,
        prompt: str,
        count: int,
        width: int,
        height: int,
        quality: str | None,
        style: str | None,
        output_format: ImageOutputFormats | None,
        output_compression: int,
        extra_params: dict[str, JsonValue],
        *,
        is_url: bool = False,
    ) -> ImageGenerationJobT:
        """Initialize an image generation job.

        Args:
            prompt: Text prompt for image generation.
            count: Number of images to generate.
            width: Image width.
            height: Image height.
            quality: Image quality level.
            style: Image style.
            output_format: Output format.
            output_compression: Output compression.
            extra_params: Extra model parameters.
            is_url: If True, return image URL instead of base64 image.
        """
        return self.IMAGE_GENERATION_JOB_CLASS(  # type: ignore[call-arg]
            model=self,
            prompt=prompt,
            count=count,
            width=width,
            height=height,
            quality=quality,
            style=style,
            output_format=output_format,
            output_compression=output_compression,
            extra_params=extra_params,
            is_url=is_url,
        )


_MODEL_REGISTRY: list[tuple[str, type[ImageModelBase[Any, Any, Any]]]] = []
_MODEL_CACHE: dict[str, ImageModelBase[Any, Any, Any]] = {}


def get_image_model(model_id: str) -> ImageModelBase[Any, Any, Any]:
    """Resolve the image model class matching the provided identifier.

    Args:
        model_id: The provider model identifier (e.g., "stability.stable-image-core-v1:1").

    Returns:
        The image model associated to the ``model_id``.

    Raises:
        LookupError: If no registered image model matches ``model_id``.
    """
    return get_model(model_id, _MODEL_CACHE, _MODEL_REGISTRY)


load_model_plugins(
    class_type=ImageModelBase, package_name=__name__, registry=_MODEL_REGISTRY
)
