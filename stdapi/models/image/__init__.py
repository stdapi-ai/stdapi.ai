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

from asyncio import Lock, as_completed, ensure_future, gather
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from stdapi.api_errors import ApiError
from stdapi.aws_s3 import put_object_and_get_url
from stdapi.models import ModelBase, get_model, load_model_plugins
from stdapi.models.capabilities import Capability
from stdapi.monitoring import REQUEST_ID, log_error_details
from stdapi.usage import IMAGE_SPEC, record_bedrock_usage
from stdapi.utils import b64decode, convert_base64_image, get_base64_image_size

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Iterable, Mapping
    from re import Pattern

    from types_aiobotocore_bedrock_runtime.literals import ServiceTierTypeType

    from stdapi.pricing import Routing
    from stdapi.types import JsonMapping
    from stdapi.types.openai_images import ImageOutputFormats, ImageOutputQuality

__all__ = [
    "ImageGenerationJobBase",
    "ImageGenerationResponse",
    "ImageModelBase",
    "get_image_model",
]

#: Default prompt for variation if not provided
DEFAULT_VARIATION_PROMPT = "Generate variations of the image."


class ImageGenerationResponse(BaseModel):
    """Image generation response.

    Attributes:
        image: Base64-encoded image data.
        partial: ``True`` if this is a partial (streaming) image.
        index: Zero-based index of this image within the generation batch.
    """

    image: str
    partial: bool = False
    index: int


class ImageGenerationJobBase[ImageModelT: "ImageModelBase[Any, Any, Any]"]:
    """Image generation job base class supporting both generation and editing."""

    __slots__ = (
        "_count",
        "_extra_params",
        "_height",
        "_input_tokens",
        "_is_url",
        "_model",
        "_output_compression",
        "_output_format",
        "_output_tokens",
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
        model: ImageModelT,
        prompt: str,
        count: int,
        width: int,
        height: int,
        quality: str | None,
        style: str | None,
        output_format: ImageOutputFormats | None,
        output_compression: int,
        extra_params: JsonMapping,
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

        # Invocation usage from ModelBase.invoke()
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None

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
        """Final output image format."""
        return self._output_format or self._response_output_format

    @property
    def input_tokens(self) -> int | None:
        """Input tokens from the model invocation."""
        return self._input_tokens

    @property
    def output_tokens(self) -> int | None:
        """Output tokens from the model invocation."""
        return self._output_tokens

    @staticmethod
    def _validate_no_mask(mask: str | None, reason: str = "") -> None:
        """Validate mask parameter is not set.

        Args:
            mask: Mask parameter value (None or string).
            reason: Extra reason for validation failure.

        Raises:
            ApiError: If mask requirement is not met.
        """
        if mask is not None:
            msg = f'"mask" parameter is not supported{" " if reason else ""}{reason}.'
            raise ApiError(msg)

    @staticmethod
    def _validate_mask(mask: str | None, reason: str = "") -> str:
        """Validate mask parameter is set.

        Args:
            mask: Mask parameter value (None or string).
            reason: Extra reason for validation failure.

        Returns:
            The validated mask string.

        Raises:
            ApiError: If mask requirement is not met.
        """
        if mask is None:
            msg = f'"mask" parameter is required{" " if reason else ""}{reason}.'
            raise ApiError(msg)
        return mask

    def _drop_unsupported_quality(self) -> None:
        """Drop a quality this model has no control for, keeping the request alive.

        Quality steers the backend; it does not decide whether an image can be
        produced. Refusing the request over it would break every client that
        sends OpenAI's default, so it is dropped with a warning and the response
        reports the quality actually produced.
        """
        if self._quality is not None:
            log_error_details(
                '"quality" is not supported by this model; ignored.', level="warning"
            )
            self._quality = None

    def _drop_unsupported_style(self) -> None:
        """Drop a style this model has no control for, keeping the request alive.

        Same rationale as quality: a style the backend cannot honour changes the
        look of the result, never whether there is one.
        """
        if self._style is not None:
            log_error_details(
                '"style" is not supported by this model; ignored.', level="warning"
            )
            self._style = None

    @staticmethod
    def _get_one_image_from_list(images: list[str]) -> str:
        """Return the single image from a list, raising if there is not exactly one.

        Args:
            images: List of base64-encoded image strings.

        Returns:
            The single image string.

        Raises:
            ApiError: If the list does not contain exactly one image.
        """
        if len(images) != 1:
            msg = "Exactly one image must be provided."
            raise ApiError(msg)
        return images[0]

    async def generate_images(self) -> Iterable[ImageGenerationResponse]:
        """Generate images from text prompt.

        Returns:
            Images.
        """
        return await gather(
            *(
                self._ensure_image_output_format(result)
                for result in await self._generate_images_from_text()
            )
        )

    async def edit_images(
        self, images: list[str], mask: str | None = None
    ) -> Iterable[ImageGenerationResponse]:
        """Edit images using inpainting.

        Args:
            images: List of base64-encoded source images.
            mask: Base64-encoded mask image (optional).

        Returns:
            Images.
        """
        return await gather(
            *(
                self._ensure_image_output_format(result)
                for result in await self._edit_image(images, mask)
            )
        )

    async def create_variations(
        self, images: list[str]
    ) -> Iterable[ImageGenerationResponse]:
        """Create variations of existing images.

        Args:
            images: List of base64-encoded source images.

        Returns:
            Images.
        """
        return await gather(
            *(
                self._ensure_image_output_format(result)
                for result in await self._create_image_variations(images)
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

    async def edit_images_stream(
        self,
        images: list[str],
        mask: str | None = None,
        partial_images: int | None = None,
    ) -> AsyncGenerator[ImageGenerationResponse]:
        """Edit images using inpainting (streaming).

        Args:
            images: List of base64-encoded source images.
            mask: Base64-encoded mask image (optional).
            partial_images: Number of partial images to generate during streaming.

        Yields:
            Images.
        """
        async for result in self._edit_images_stream(images, mask, partial_images):
            yield await self._ensure_image_output_format(result)

    async def _generate_images_from_text(
        self,
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Generate images from text prompt.

        Returns:
            Iterable of awaitable image generation responses.

        Raises:
            ApiError: If the model does not support text-to-image generation.
        """
        msg = f"Text-to-image generation is not supported by {self._model.model.id}"
        raise ApiError(msg)

    async def _edit_image(
        self,
        images: list[str],  # noqa: ARG002
        mask: str | None,  # noqa: ARG002
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Edit images.

        Args:
            images: List of base64-encoded source images.
            mask: Base64-encoded mask image (optional).

        Returns:
            Iterable of awaitable image generation responses.

        Raises:
            ApiError: If the model does not support inpainting.
        """
        msg = "Image editing is not supported by {self._model.model.id}."
        raise ApiError(msg)

    async def _create_image_variations(
        self,
        images: list[str],  # noqa: ARG002
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Create variations of existing images.

        Args:
            images: List of base64-encoded source images.

        Returns:
            Iterable of awaitable image generation responses.

        Raises:
            ApiError: If the model does not support image variations.
        """
        msg = f"Image variations are not supported by {self._model.model.id}."
        raise ApiError(msg)

    @classmethod
    def get_supported_operations(cls) -> Capability:
        """Auto-detect supported image operations from method override presence.

        Returns:
            Capability flags for operations this job class implements.
        """
        ops = Capability(0)
        if (
            cls._generate_images_from_text
            is not ImageGenerationJobBase._generate_images_from_text
        ):
            ops |= Capability.IMAGE_GENERATION
        if cls._edit_image is not ImageGenerationJobBase._edit_image:
            ops |= Capability.IMAGE_EDITION
        if (
            cls._create_image_variations
            is not ImageGenerationJobBase._create_image_variations
        ):
            ops |= Capability.IMAGE_VARIATION
        return ops

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
        async for image in self._stream_completed_images(
            await self._generate_images_from_text()
        ):
            yield image

    async def _edit_images_stream(
        self,
        images: list[str],
        mask: str | None = None,
        partial_images: int | None = None,  # noqa:ARG002
    ) -> AsyncGenerator[ImageGenerationResponse]:
        """Stream edited images using inpainting.

        Args:
            images: List of base64-encoded source images.
            mask: Base64-encoded mask image (optional).
            partial_images: Number of partial images to generate during streaming.

        Yields:
            Streamed images.
        """
        async for image in self._stream_completed_images(
            await self._edit_image(images, mask)
        ):
            yield image

    async def _stream_completed_images(
        self, results: Iterable[Awaitable[ImageGenerationResponse]]
    ) -> AsyncGenerator[ImageGenerationResponse]:
        """Yield formatted images in completion order, stopping jobs on early exit.

        An abandoned stream (client disconnect) must cancel the in-flight
        jobs: an image completing after the last usage drain would record
        billed usage nothing ever reads.

        Args:
            results: Awaitable image generation responses.

        Yields:
            Formatted images, in completion order.
        """
        tasks = [
            ensure_future(self._ensure_image_output_format(result))
            for result in results
        ]
        try:
            for task in as_completed(tasks):
                yield await task
        finally:
            for pending in tasks:
                pending.cancel()
            # Await cancellation so asyncio doesn't log unretrieved exceptions at GC.
            await gather(*tasks, return_exceptions=True)

    async def _ensure_image_output_format(
        self, response: Awaitable[ImageGenerationResponse] | ImageGenerationResponse
    ) -> ImageGenerationResponse:
        """Convert the image to the requested output format if needed.

        Args:
            response: Awaitable or resolved image generation response.

        Returns:
            Image response with the correct output format applied.
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
            ApiError: If S3 operations fail.
        """
        request_id = REQUEST_ID.get()
        return await put_object_and_get_url(
            await b64decode(image_data),
            f"image/{output_format}",
            f"{request_id}/image-{request_id}-{index + 1:03d}.{'jpg' if output_format == 'jpeg' else output_format}",
        )


class ImageModelBase[
    RequestT,
    ResponseT,
    ImageGenerationJobT: ImageGenerationJobBase[Any],
](ModelBase[RequestT, ResponseT]):
    """Base class for provider-specific image models."""

    #: InvokeModel rejects native guardrail kwargs; ApplyGuardrail covers the route.
    NATIVE_GUARDRAIL_SUPPORTED: ClassVar[bool] = False

    IMAGE_GENERATION_JOB_CLASS: type[ImageGenerationJobT]

    @classmethod
    def get_supported_operations(cls) -> Capability:
        """Delegate capability detection to the job class.

        Returns:
            Capability flags derived from the job class implementation.
        """
        job_cls: type[ImageGenerationJobBase[Any]] | None = getattr(
            cls, "IMAGE_GENERATION_JOB_CLASS", None
        )
        return (
            job_cls.get_supported_operations() if job_cls is not None else Capability(0)
        )

    def _record_invoke_usage(
        self,
        input_tokens: int | None,
        output_tokens: int | None,
        response: Mapping[str, Any],
        *,
        region: str = "",
        tier: ServiceTierTypeType | None = None,
        routing: Routing | None = None,
    ) -> None:
        """Record invoke usage plus the billed output images.

        Args:
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens consumed.
            response: The parsed response body -- its ``images`` list (when
                present) is the actual number of images AWS billed for.
            region: Region that served the call.
            tier: Service tier that served the call.
            routing: Serving profile of the call.
        """
        super()._record_invoke_usage(
            input_tokens,
            output_tokens,
            response,
            region=region,
            tier=tier,
            routing=routing,
        )
        if images := response.get("images"):
            record_bedrock_usage(
                self._model_id,
                tier=tier,
                region=region,
                routing=routing,
                output_images=len(images),
            )
        # Clear IMAGE_SPEC to avoid stale values from earlier calls in the same context.
        IMAGE_SPEC.set("")

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
        extra_params: JsonMapping,
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

        Returns:
            Configured image generation job instance.
        """
        return self.IMAGE_GENERATION_JOB_CLASS(
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

    def get_image_edit_job(
        self,
        prompt: str,
        count: int,
        width: int,
        height: int,
        output_format: ImageOutputFormats | None,
        output_compression: int,
        extra_params: JsonMapping,
        *,
        is_url: bool = False,
    ) -> ImageGenerationJobT:
        """Initialize an image edit job.

        Args:
            prompt: Text prompt for image editing.
            count: Number of images to generate.
            width: Image width.
            height: Image height.
            output_format: Output format.
            output_compression: Output compression.
            extra_params: Extra model parameters.
            is_url: If True, return image URL instead of base64 image.

        Returns:
            Job instance - call edit_images(image, mask) on it.
        """
        return self.IMAGE_GENERATION_JOB_CLASS(
            model=self,
            prompt=prompt,
            count=count,
            width=width,
            height=height,
            quality=None,
            style=None,
            output_format=output_format,
            output_compression=output_compression,
            extra_params=extra_params,
            is_url=is_url,
        )

    def get_image_variation_job(
        self,
        count: int,
        width: int,
        height: int,
        output_format: ImageOutputFormats | None,
        output_compression: int,
        extra_params: JsonMapping,
        *,
        is_url: bool = False,
    ) -> ImageGenerationJobT:
        """Initialize an image variation job.

        Args:
            count: Number of images to generate.
            width: Image width.
            height: Image height.
            output_format: Output format.
            output_compression: Output compression.
            extra_params: Extra model parameters.
            is_url: If True, return image URL instead of base64 image.

        Returns:
            Job instance - call create_variations(images) on it.
        """
        return self.IMAGE_GENERATION_JOB_CLASS(
            model=self,
            prompt="",  # No prompt for variations
            count=count,
            width=width,
            height=height,
            quality=None,
            style=None,
            output_format=output_format,
            output_compression=output_compression,
            extra_params=extra_params,
            is_url=is_url,
        )


_MODEL_REGISTRY: list[
    tuple[str | Pattern[str], type[ImageModelBase[Any, Any, Any]]]
] = []
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
    return get_model(model_id, _MODEL_CACHE, _MODEL_REGISTRY, __name__)


load_model_plugins(
    class_type=ImageModelBase, package_name=__name__, registry=_MODEL_REGISTRY
)
