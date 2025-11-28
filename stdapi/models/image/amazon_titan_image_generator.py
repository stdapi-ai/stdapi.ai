"""Amazon Titan Image Generator models.

- amazon.titan-image-generator-v1
- amazon.titan-image-generator-v2:0
"""

from secrets import randbelow
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

from stdapi.models.image import (
    DEFAULT_VARIATION_PROMPT,
    ImageGenerationJobBase,
    ImageGenerationResponse,
    ImageModelBase,
)
from stdapi.openai_exceptions import OpenaiError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable

    from stdapi.types.openai_images import ImageOutputQuality

AmzQuality = Literal["standard", "premium"]
TaskType = Literal[
    "TEXT_IMAGE",
    "INPAINTING",
    "OUTPAINTING",
    "IMAGE_VARIATION",
    "COLOR_GUIDED_GENERATION",
    "BACKGROUND_REMOVAL",
]

AMZ_QUALITY_MAP: dict[ImageOutputQuality, AmzQuality] = {
    "low": "standard",
    "medium": "standard",
    "high": "premium",
}


def get_amz_quality(quality: ImageOutputQuality | str | None) -> AmzQuality | None:
    """Converts input image quality to the corresponding Amazon quality format.

    This function standardizes the given image output quality parameter to align
    with pre-defined Amazon quality mappings. It checks the input value against
    a mapping dictionary and returns the appropriate Amazon quality format. If
    the provided input does not match any value in the mapping, the original
    input is returned unaltered. The Bedrock API validates the final value.

    Args:
        quality: A string or ImageOutputQuality specifying the desired image
            quality. The value will be case-insensitively matched against the
            pre-defined Amazon quality map.

    Returns:
        The corresponding Amazon-specific quality format if found
            in the mapping or the original quality input if no match exists.
    """
    if quality is None:
        return None
    quality = quality.lower()
    return AMZ_QUALITY_MAP.get(quality, quality)  # type: ignore[no-any-return,call-overload]


def random_seed() -> int:
    """Generate a random seed value.

    Returns:
        Seed
    """
    return randbelow(2147483646)


class _TextToImageParams(TypedDict):
    """Text-to-image parameters."""

    text: str  # Required: The prompt text (1-512 characters)
    negativeText: NotRequired[str]  # Optional: What NOT to include (1-512 characters)
    # V2 only - condition image features
    conditionImage: NotRequired[str]  # Optional: Base64-encoded condition image
    controlMode: NotRequired[
        Literal["CANNY_EDGE", "SEGMENTATION"]
    ]  # Optional: Control mode
    controlStrength: NotRequired[
        float
    ]  # Optional: 0.0-1.0, similarity to condition image


class _InpaintingParams(TypedDict):
    """Inpainting parameters."""

    image: str  # Required: Base64-encoded source image (JPEG/PNG)
    text: NotRequired[str]  # Optional: Prompt for changes (1-512 characters)
    negativeText: NotRequired[str]  # Optional: What NOT to include (1-512 characters)
    maskPrompt: NotRequired[
        str
    ]  # Optional: Text description of mask area (alternative to maskImage)
    maskImage: NotRequired[
        str
    ]  # Optional: Base64-encoded mask image (alternative to maskPrompt)
    returnMask: NotRequired[bool]  # Optional: Return the mask image in response


class _OutpaintingParams(TypedDict):
    """Outpainting parameters."""

    text: str  # Required: What to change outside mask (1-512 characters)
    negativeText: NotRequired[str]  # Optional: What NOT to include (1-512 characters)
    image: str  # Required: Base64-encoded source image (JPEG/PNG)
    maskPrompt: NotRequired[
        str
    ]  # Optional: Text description of mask area (alternative to maskImage)
    maskImage: NotRequired[
        str
    ]  # Optional: Base64-encoded mask image (alternative to maskPrompt)
    outPaintingMode: NotRequired[
        Literal["DEFAULT", "PRECISE"]
    ]  # Optional: Outpainting mode


class _ImageVariationParams(TypedDict):
    """Image variation parameters."""

    images: list[str]  # Required: 1-5 base64-encoded images
    text: NotRequired[str]  # Optional: What to preserve/change (1-512 characters)
    negativeText: NotRequired[str]  # Optional: What NOT to include (1-512 characters)
    similarityStrength: NotRequired[
        float
    ]  # Optional: 0.2-1.0, similarity to input images


class _ColorGuidedGenerationParams(TypedDict):
    """Color-guided generation parameters (V2 only)."""

    text: str  # Required: Image generation prompt (1-512 characters)
    colors: list[str]  # Required: 1-10 hex color codes (e.g., ["#FF5733", "#33FF57"])
    negativeText: NotRequired[str]  # Optional: What NOT to include (1-512 characters)
    referenceImage: NotRequired[
        str
    ]  # Optional: Base64-encoded reference image for color palette


class _BackgroundRemovalParams(TypedDict):
    """Background removal parameters (V2 only)."""

    image: str  # Required: Base64-encoded source image (JPEG/PNG)


class _ImageGenerationConfig(TypedDict):
    """Image generation configuration."""

    numberOfImages: NotRequired[int]  # 1-5, default 1
    quality: NotRequired[AmzQuality]  # default "standard"
    cfgScale: NotRequired[
        float
    ]  # 1.1-10.0, default 8.0 - how closely the model follows the prompt
    height: NotRequired[
        int
    ]  # 512, 768, 1024, 1152, 1216, 1344, 1536, 2048 (default 512)
    width: NotRequired[
        int
    ]  # 512, 768, 1024, 1152, 1216, 1344, 1536, 2048 (default 512)
    seed: NotRequired[int]  # 0-2147483646, default 42


class _Request(TypedDict):
    """Amazon Titan Image Generator request parameters."""

    taskType: TaskType
    textToImageParams: NotRequired[_TextToImageParams]
    inPaintingParams: NotRequired[_InpaintingParams]
    outPaintingParams: NotRequired[_OutpaintingParams]
    imageVariationParams: NotRequired[_ImageVariationParams]
    colorGuidedGenerationParams: NotRequired[_ColorGuidedGenerationParams]
    backgroundRemovalParams: NotRequired[_BackgroundRemovalParams]
    imageGenerationConfig: NotRequired[_ImageGenerationConfig]


class _Response(TypedDict):
    """Amazon Titan Image Generator response parameters."""

    images: list[str]  # List of base64 encoded images


class _ImageGenerationJob(ImageGenerationJobBase["ImageModel"]):
    """Image generation job supporting both text-to-image and inpainting."""

    @staticmethod
    async def _create_response(image: str, index: int) -> ImageGenerationResponse:
        """Create an ImageGenerationResponse from image data.

        Args:
            image: Base64 encoded image.
            index: Image index.

        Returns:
            ImageGenerationResponse object.
        """
        return ImageGenerationResponse(image=image, index=index)

    async def _invoke_and_process_response(
        self, request: _Request
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Common logic to invoke model and process response.

        Args:
            request: The request to send to the model.

        Returns:
            Iterable of awaitable image generation responses.
        """
        self._response_height = self._height
        self._response_width = self._width
        self._response_output_format = "png"
        return tuple(
            self._create_response(image, index)
            for index, image in enumerate((await self._model.invoke(request))["images"])
        )

    async def _generate_images_from_text(
        self,
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Generate images from text prompt.

        Returns:
            Iterable of awaitable image generation responses.
        """
        self._validate_no_style()

        image_generation_config = _ImageGenerationConfig(
            width=self._width,
            height=self._height,
            numberOfImages=self._count,
            seed=random_seed(),
        )
        task_type: TaskType = self._extra_params.get("taskType", "TEXT_IMAGE")  # type: ignore[assignment]
        if task_type == "TEXT_IMAGE":
            request = self._get_request_text_image(image_generation_config)
        elif task_type == "COLOR_GUIDED_GENERATION":
            request = self._get_request_color_guided_generation(image_generation_config)
        else:
            msg = '"taskType" value must be "TEXT_IMAGE" or "COLOR_GUIDED_GENERATION".'
            raise OpenaiError(msg)
        return await self._invoke_and_process_response(request)

    async def _edit_image(
        self, images: list[str], mask: str | None
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Edit images.

        Args:
            images: List of base64-encoded source images.
            mask: Base64-encoded mask image (optional).

        Returns:
            Iterable of awaitable image generation responses.
        """
        image_generation_config = _ImageGenerationConfig(
            width=self._width,
            height=self._height,
            numberOfImages=self._count,
            seed=random_seed(),
        )
        image = self._get_one_image_from_list(images)
        task_type: TaskType = self._extra_params.get("taskType", "INPAINTING")  # type: ignore[assignment]
        if task_type == "INPAINTING":
            request = self._get_request_inpainting(image_generation_config, image, mask)
        elif task_type == "OUTPAINTING":
            request = self._get_request_outpainting(
                image_generation_config, image, mask
            )
        elif task_type == "BACKGROUND_REMOVAL":
            request = self._get_request_background_removal(
                image_generation_config, image, mask
            )
        else:
            msg = '"taskType" value must be "INPAINTING", "OUTPAINTING" or "BACKGROUND_REMOVAL".'
            raise OpenaiError(msg)
        return await self._invoke_and_process_response(request)

    async def _create_image_variations(
        self, images: list[str]
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Create variations of existing images.

        Args:
            images: List of base64-encoded source images.

        Returns:
            Iterable of awaitable image generation responses.
        """
        image_generation_config = _ImageGenerationConfig(
            width=self._width,
            height=self._height,
            numberOfImages=self._count,
            seed=random_seed(),
        )
        task_type: TaskType = self._extra_params.get("taskType", "IMAGE_VARIATION")  # type: ignore[assignment]
        if task_type == "IMAGE_VARIATION":
            request = self._get_request_image_variation(image_generation_config, images)
        elif task_type == "TEXT_IMAGE":
            # Conditioned image generation from an input image
            request = self._get_request_text_image(image_generation_config, images)
        elif task_type == "COLOR_GUIDED_GENERATION":
            request = self._get_request_color_guided_generation(
                image_generation_config, images
            )
        else:
            msg = (
                '"taskType" value must be "IMAGE_VARIATION", "TEXT_IMAGE" or "COLOR_GUIDED_GENERATION".',
            )
            raise OpenaiError(msg)
        return await self._invoke_and_process_response(request)

    def _get_request_text_image(
        self,
        image_generation_config: _ImageGenerationConfig,
        images: list[str] | None = None,
    ) -> _Request:
        """Generates and returns a request object configured for text-to-image generation.

        The method constructs a request object for generating images from text prompts,
        including optional configurations for condition images and additional parameters.

        Args:
            image_generation_config (_ImageGenerationConfig):
                The configuration settings for image generation.
            images (list[str] | None):
                An optional list of image file paths or URLs to be used as conditional
                inputs for the text-to-image generation process. Defaults to None.

        Returns:
            _Request:
                A fully configured request object for text-to-image generation.
        """
        request = _Request(
            taskType="TEXT_IMAGE",
            textToImageParams=_TextToImageParams(
                text=self._prompt or DEFAULT_VARIATION_PROMPT
            ),
            imageGenerationConfig=image_generation_config,
        )
        if images:
            request["textToImageParams"]["conditionImage"] = (
                self._get_one_image_from_list(images)
            )
        self._set_extra_config(request, "textToImageParams")
        return request

    def _get_request_color_guided_generation(
        self,
        image_generation_config: _ImageGenerationConfig,
        images: list[str] | None = None,
    ) -> _Request:
        """Constructs and returns a request object for the color-guided generation task.

        This method prepares an object encapsulating the parameters and configuration
        necessary to perform a color-guided image generation task. It ensures that
        the required color parameters are set and integrates additional optional
        configurations, such as reference images, if provided.

        Args:
            image_generation_config: Configuration object containing parameters for
                image generation.
            images: Optional list of image file paths or identifiers that may be used
                as reference images for the color-guided generation task.

        Returns:
            A request object configured for color-guided image generation task.
        """
        try:
            colors: list[str] = self._extra_params["colorGuidedGenerationParams"][  # type: ignore[call-overload,index,assignment]
                "colors"  # type: ignore[index]
            ]
        except (KeyError, TypeError, IndexError) as exc:
            msg = "Required parameter for COLOR_GUIDED_GENERATION: colorGuidedGenerationParams.colors"
            raise OpenaiError(msg) from exc
        request = _Request(
            taskType="COLOR_GUIDED_GENERATION",
            colorGuidedGenerationParams=_ColorGuidedGenerationParams(
                text=self._prompt or DEFAULT_VARIATION_PROMPT, colors=colors
            ),
            imageGenerationConfig=image_generation_config,
        )
        if images:
            request["colorGuidedGenerationParams"]["referenceImage"] = (
                self._get_one_image_from_list(images)
            )
        self._set_extra_config(request, "colorGuidedGenerationParams")
        return request

    def _get_request_image_variation(
        self, image_generation_config: _ImageGenerationConfig, images: list[str]
    ) -> _Request:
        """Generates a request object for image variation tasks.

        This method constructs a request object configured for the "IMAGE_VARIATION" task. It
        uses the provided image generation configuration and a list of input images to populate
        the required parameters. The method also applies any additional configurations to the
        request object.

        Args:
            image_generation_config: Configuration for image generation parameters.
            images: List of image file paths or identifiers for generating variations.

        Returns:
            A request object configured for the "IMAGE_VARIATION" task.
        """
        request = _Request(
            taskType="IMAGE_VARIATION",
            imageVariationParams=_ImageVariationParams(images=images),
            imageGenerationConfig=image_generation_config,
        )
        self._set_extra_config(request, "imageVariationParams")
        return request

    def _get_request_inpainting(
        self,
        image_generation_config: _ImageGenerationConfig,
        image: str,
        mask: str | None,
    ) -> _Request:
        """Creates and returns a request object for performing an inpainting task.

        The request includes image generation configuration and optional mask data
        to customize the inpainting process.

        Args:
            image_generation_config: Configuration settings for image generation.
            image: Base image for inpainting provided as a string.
            mask: Optional mask image to specify the painting area.

        Returns:
            _Request: A fully configured request object for the inpainting task.
        """
        request = _Request(
            taskType="INPAINTING",
            inPaintingParams=_InpaintingParams(image=image, text=self._prompt),
            imageGenerationConfig=image_generation_config,
        )
        if mask:
            request["inPaintingParams"]["maskImage"] = mask
        self._set_extra_config(request, "inPaintingParams")
        return request

    def _get_request_outpainting(
        self,
        image_generation_config: _ImageGenerationConfig,
        image: str,
        mask: str | None,
    ) -> _Request:
        """Constructs and returns an outpainting request object.

        This method prepares a request with necessary parameters to invoke an
        outpainting task, including the image and optional mask configuration.
        The request object includes task-specific parameters and configurations for
        generating the desired output.

        Args:
            image_generation_config: Configuration object for image generation
                tasks that defines various parameters for generating output images.
            image: String representing the input image required for the outpainting
                task.
            mask: Optional mask image represented as a string, specifying areas
                of interest for the outpainting operation. If `None`, the mask
                will be omitted from the request.

        Returns:
            _Request: The constructed outpainting request object containing all
            specified parameters and configurations.
        """
        request = _Request(
            taskType="OUTPAINTING",
            outPaintingParams=_OutpaintingParams(image=image, text=self._prompt),
            imageGenerationConfig=image_generation_config,
        )
        if mask:
            request["outPaintingParams"]["maskImage"] = mask
        self._set_extra_config(request, "outPaintingParams")
        return request

    def _get_request_background_removal(
        self,
        image_generation_config: _ImageGenerationConfig,
        image: str,
        mask: str | None,
    ) -> _Request:
        """Constructs and returns a request for background removal processing.

        This method is responsible for generating a `_Request` object configured
        to perform background removal on the given image. It integrates
        the provided image generation configuration and validates
        whether optional parameters are supported.

        Args:
            image_generation_config (_ImageGenerationConfig): The configuration
                for image generation, which specifies the parameters
                required by the background removal model.
            image (str): The path to the input image from which the background
                needs to be removed.
            mask (str | None): An optional parameter, which when provided,
                will raise an error indicating that masking is not supported
                by the underlying model.

        Raises:
            OpenaiError: If the mask parameter is provided, as this model
                does not support mask-based operations.

        Returns:
            _Request: A configured `_Request` object for initiating background
                removal processing.
        """
        self._validate_no_mask(mask, reason="with BACKGROUND_REMOVAL taskType")
        request = _Request(
            taskType="BACKGROUND_REMOVAL",
            backgroundRemovalParams=_BackgroundRemovalParams(image=image),
            imageGenerationConfig=image_generation_config,
        )
        self._set_extra_config(request, "backgroundRemovalParams")
        return request

    def _set_extra_config(self, request: _Request, task_key: str) -> None:
        """Updates the image generation configuration in the given request.

        Args:
            request (_Request): The request dictionary to be updated with the
                image generation configuration.
            task_key (str): The key for the task-specific parameters dictionary
        """
        if self._extra_params:
            if "imageGenerationConfig" in self._extra_params:
                request["imageGenerationConfig"].update(
                    self._extra_params["imageGenerationConfig"]  # type:ignore[typeddict-item]
                )
            if task_key in self._extra_params:
                request[task_key].update(self._extra_params[task_key])  # type:ignore[literal-required]

        amz_quality = get_amz_quality(self._quality)
        if amz_quality:
            request["imageGenerationConfig"]["quality"] = amz_quality
            self._response_quality = "high" if amz_quality == "premium" else "medium"


class ImageModel(ImageModelBase[_Request, _Response, _ImageGenerationJob]):
    """Amazon Titan Image Generator model."""

    MATCHER = "amazon.titan-image-generator"
    IMAGE_GENERATION_JOB_CLASS = _ImageGenerationJob
