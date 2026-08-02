"""Amazon Nova Canvas image generation models.

- amazon.nova-canvas-v1:0
"""

from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

from stdapi.api_errors import ApiError
from stdapi.models.image import (
    DEFAULT_VARIATION_PROMPT,
    ImageGenerationJobBase,
    ImageGenerationResponse,
    ImageModelBase,
)
from stdapi.models.image.amazon_titan_image_generator import (
    AmzQuality,
    get_amz_quality,
    image_spec,
    random_seed,
    resolve_amz_quality_echo,
)
from stdapi.usage import IMAGE_SPEC
from stdapi.utils import alpha_mask_to_bw, get_data_uri_data

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable


class _TextToImageParams(TypedDict):
    """Text-to-image parameters."""

    text: str  # Required: 1-1024 characters
    negativeText: NotRequired[str]  # Optional: 1-1024 characters
    style: NotRequired[
        Literal[
            "3D_ANIMATED_FAMILY_FILM",
            "DESIGN_SKETCH",
            "FLAT_VECTOR_ILLUSTRATION",
            "GRAPHIC_NOVEL_ILLUSTRATION",
            "MAXIMALISM",
            "MIDCENTURY_RETRO",
            "PHOTOREALISM",
            "SOFT_DIGITAL_PAINTING",
        ]
    ]
    # Image conditioning parameters
    conditionImage: NotRequired[str]  # Base64 encoded image
    controlMode: NotRequired[Literal["CANNY_EDGE", "SEGMENTATION"]]
    controlStrength: NotRequired[float]  # 0.0-1.0


class _ColorGuidedGenerationParams(TypedDict):
    """Color guided generation parameters."""

    colors: list[str]  # Required: list of hexadecimal color values (up to 10)
    referenceImage: NotRequired[str]  # Base64 encoded image
    text: str  # Required: 1-1024 characters
    negativeText: NotRequired[str]  # Optional: 1-1024 characters


class _ImageVariationParams(TypedDict):
    """Image variation parameters."""

    images: list[str]  # Required: 1-5 Base64 encoded images
    similarityStrength: NotRequired[float]  # 0.2-1.0
    text: NotRequired[str]  # Required: 1-1024 characters
    negativeText: NotRequired[str]  # Optional: 1-1024 characters


class _InPaintingParams(TypedDict):
    """Inpainting parameters."""

    image: str  # Required: Base64 encoded image
    maskPrompt: NotRequired[str]  # Either maskPrompt or maskImage required
    maskImage: NotRequired[str]  # Base64 encoded mask image
    text: str  # Required: 1-1024 characters
    negativeText: NotRequired[str]  # Optional: 1-1024 characters


class _OutPaintingParams(TypedDict):
    """Outpainting parameters."""

    image: str  # Required: Base64 encoded image
    maskPrompt: NotRequired[str]  # Either maskPrompt or maskImage required
    maskImage: NotRequired[str]  # Base64 encoded mask image
    outPaintingMode: NotRequired[Literal["DEFAULT", "PRECISE"]]
    text: str  # Required: 1-1024 characters
    negativeText: NotRequired[str]  # Optional: 1-1024 characters


class _BackgroundRemovalParams(TypedDict):
    """Background removal parameters."""

    image: str  # Required: Base64 encoded image


class _ImageBasedMask(TypedDict):
    """Image based mask for virtual try-on."""

    maskImage: str  # Base64 encoded mask image


class _GarmentStyling(TypedDict):
    """Garment styling options for virtual try-on."""

    longSleeveStyle: NotRequired[Literal["SLEEVE_DOWN", "SLEEVE_UP"]]
    tuckingStyle: NotRequired[Literal["UNTUCKED", "TUCKED"]]
    outerLayerStyle: NotRequired[Literal["CLOSED", "OPEN"]]


class _GarmentBasedMask(TypedDict):
    """Garment based mask for virtual try-on."""

    maskShape: NotRequired[Literal["CONTOUR", "BOUNDING_BOX", "DEFAULT"]]
    garmentClass: NotRequired[
        Literal[
            "UPPER_BODY",
            "LOWER_BODY",
            "FULL_BODY",
            "FOOTWEAR",
            "LONG_SLEEVE_SHIRT",
            "SHORT_SLEEVE_SHIRT",
            "NO_SLEEVE_SHIRT",
            "OTHER_UPPER_BODY",
            "LONG_PANTS",
            "SHORT_PANTS",
            "OTHER_LOWER_BODY",
            "LONG_DRESS",
            "SHORT_DRESS",
            "FULL_BODY_OUTFIT",
            "OTHER_FULL_BODY",
            "SHOES",
            "BOOTS",
            "OTHER_FOOTWEAR",
        ]
    ]
    garmentStyling: NotRequired[_GarmentStyling]


class _PromptBasedMask(TypedDict):
    """Prompt based mask for virtual try-on."""

    maskShape: NotRequired[Literal["BOUNDING_BOX", "CONTOUR", "DEFAULT"]]
    maskPrompt: str


class _MaskExclusions(TypedDict):
    """Mask exclusions for virtual try-on."""

    preserveBodyPose: NotRequired[Literal["ON", "OFF", "DEFAULT"]]
    preserveHands: NotRequired[Literal["ON", "OFF", "DEFAULT"]]
    preserveFace: NotRequired[Literal["OFF", "ON", "DEFAULT"]]


class _VirtualTryOnParams(TypedDict):
    """Virtual try-on parameters."""

    sourceImage: str  # Required: Base64 encoded image
    referenceImage: str  # Required: Base64 encoded image
    maskType: Literal["IMAGE", "GARMENT", "PROMPT"]
    imageBasedMask: NotRequired[_ImageBasedMask]
    garmentBasedMask: NotRequired[_GarmentBasedMask]
    promptBasedMask: NotRequired[_PromptBasedMask]
    maskExclusions: NotRequired[_MaskExclusions]
    mergeStyle: NotRequired[Literal["BALANCED", "SEAMLESS", "DETAILED"]]
    returnMask: NotRequired[bool]


class _ImageGenerationConfig(TypedDict):
    """Image generation configuration."""

    width: NotRequired[int]  # 320-4096, divisible by 16, default 1024
    height: NotRequired[int]  # 320-4096, divisible by 16, default 1024
    quality: NotRequired[AmzQuality]  # default "standard"
    cfgScale: NotRequired[float]  # 1.1-10, default 6.5
    seed: NotRequired[int]  # 0-2,147,483,646, default 12
    numberOfImages: NotRequired[int]  # 1-5, default 1


class _Request(TypedDict):
    """Amazon Nova Canvas request parameters."""

    taskType: Literal[
        "TEXT_IMAGE",
        "COLOR_GUIDED_GENERATION",
        "IMAGE_VARIATION",
        "INPAINTING",
        "OUTPAINTING",
        "BACKGROUND_REMOVAL",
        "VIRTUAL_TRY_ON",
    ]

    # Task-specific parameters (only one should be used based on taskType)
    textToImageParams: NotRequired[_TextToImageParams]
    colorGuidedGenerationParams: NotRequired[_ColorGuidedGenerationParams]
    imageVariationParams: NotRequired[_ImageVariationParams]
    inPaintingParams: NotRequired[_InPaintingParams]
    outPaintingParams: NotRequired[_OutPaintingParams]
    backgroundRemovalParams: NotRequired[_BackgroundRemovalParams]
    virtualTryOnParams: NotRequired[_VirtualTryOnParams]

    # Common configuration (not used for BACKGROUND_REMOVAL)
    imageGenerationConfig: NotRequired[_ImageGenerationConfig]


class _Response(TypedDict):
    """Amazon Nova Canvas response parameters."""

    images: NotRequired[list[str]]  # List of Base64 encoded images
    maskImage: NotRequired[str]  # Base64 encoded mask image (when requested)
    error: NotRequired[str]  # Error message if content doesn't align with RAI policy


class _ImageGenerationJob(ImageGenerationJobBase["ImageModel"]):
    """Image generation job supporting both text-to-image and inpainting."""

    __slots__ = (
        "_input_tokens",
        "_output_tokens",
        "_response_height",
        "_response_output_format",
        "_response_quality",
        "_response_width",
    )

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

        Raises:
            ApiError: If model returns an error.
        """
        self._response_height = self._height
        self._response_width = self._width
        self._response_output_format = "png"
        IMAGE_SPEC.set(
            image_spec(
                self._width,
                self._height,
                get_amz_quality(self._quality),
                low_tier_max=1024,
                # 1024/2048/4096 pricing tiers (Nova Canvas allows up to 4096px).
                tiers=3,
            )
        )

        result = await self._model.invoke(request)
        self._input_tokens = result.input_tokens
        self._output_tokens = result.output_tokens
        response = result.response
        if "error" in response:
            raise ApiError(response["error"])

        return tuple(
            self._create_response(image, index)
            for index, image in enumerate(response["images"])
        )

    def _apply_extra_params(self, request: _Request, task_params_key: str) -> None:
        """Apply extra parameters to request.

        Args:
            request: The request to modify.
            task_params_key: Key for task-specific params (e.g., "textToImageParams").
        """
        if self._extra_params:
            if task_params_key in self._extra_params:
                request[task_params_key].update(  # type: ignore[literal-required]
                    self._extra_params[task_params_key]
                )
            if "imageGenerationConfig" in self._extra_params:
                request["imageGenerationConfig"].update(
                    self._extra_params["imageGenerationConfig"]  # type: ignore[typeddict-item]
                )

    async def _generate_images_from_text(
        self,
    ) -> Iterable[Awaitable[ImageGenerationResponse]]:
        """Generate images from text prompt.

        Returns:
            Iterable of awaitable image generation responses.
        """
        image_generation_config = _ImageGenerationConfig(
            width=self._width,
            height=self._height,
            numberOfImages=self._count,
            seed=random_seed(),
        )

        task_type: str = self._extra_params.get("taskType", "TEXT_IMAGE")  # type: ignore[assignment]
        if task_type == "TEXT_IMAGE":
            request = self._get_request_text_image(image_generation_config)
        elif task_type == "COLOR_GUIDED_GENERATION":
            request = self._get_request_color_guided_generation(image_generation_config)
        else:
            msg = '"taskType" value must be "TEXT_IMAGE" or "COLOR_GUIDED_GENERATION".'
            raise ApiError(msg)

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
        task_type: str = self._extra_params.get(
            "taskType", "INPAINTING" if mask else "TEXT_IMAGE"
        )  # type: ignore[assignment]
        if task_type == "TEXT_IMAGE":
            request = self._get_request_text_image(image_generation_config, [image])
        elif task_type == "INPAINTING":
            request = await self._get_request_inpainting(
                image_generation_config, image, mask
            )
        elif task_type == "OUTPAINTING":
            request = await self._get_request_outpainting(
                image_generation_config, image, mask
            )
        elif task_type == "BACKGROUND_REMOVAL":
            request = self._get_request_background_removal(
                image_generation_config, image, mask
            )
        elif task_type == "VIRTUAL_TRY_ON":
            request = self._get_request_virtual_try_on(
                image_generation_config, image, mask
            )
        else:
            msg = '"taskType" value must be "TEXT_IMAGE", "INPAINTING", "OUTPAINTING", "BACKGROUND_REMOVAL" or "VIRTUAL_TRY_ON".'
            raise ApiError(msg)

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

        task_type: str = self._extra_params.get("taskType", "IMAGE_VARIATION")  # type: ignore[assignment]
        if task_type == "IMAGE_VARIATION":
            request = self._get_request_image_variation(image_generation_config, images)
        elif task_type == "TEXT_IMAGE":
            request = self._get_request_text_image(image_generation_config, images)
        elif task_type == "COLOR_GUIDED_GENERATION":
            request = self._get_request_color_guided_generation(
                image_generation_config, images
            )
        else:
            msg = '"taskType" value must be "IMAGE_VARIATION", "TEXT_IMAGE" or "COLOR_GUIDED_GENERATION".'
            raise ApiError(msg)

        return await self._invoke_and_process_response(request)

    async def _get_request_inpainting(
        self,
        image_generation_config: _ImageGenerationConfig,
        image: str,
        mask: str | None,
    ) -> _Request:
        """Get request for INPAINTING task.

        Args:
            image_generation_config: Image generation configuration.
            image: Base64-encoded source image.
            mask: Optional base64-encoded mask image.

        Returns:
            Request object for INPAINTING task.
        """
        request = _Request(
            taskType="INPAINTING",
            inPaintingParams=_InPaintingParams(image=image, text=self._prompt),
            imageGenerationConfig=image_generation_config,
        )

        if mask:
            request["inPaintingParams"]["maskImage"] = await alpha_mask_to_bw(mask)

        self._apply_extra_params(request, "inPaintingParams")
        self._response_quality = "medium"
        return request

    async def _get_request_outpainting(
        self,
        image_generation_config: _ImageGenerationConfig,
        image: str,
        mask: str | None,
    ) -> _Request:
        """Get request for OUTPAINTING task.

        Args:
            image_generation_config: Image generation configuration.
            image: Base64-encoded source image.
            mask: Optional base64-encoded mask image.

        Returns:
            Request object for OUTPAINTING task.
        """
        request = _Request(
            taskType="OUTPAINTING",
            outPaintingParams=_OutPaintingParams(image=image, text=self._prompt),
            imageGenerationConfig=image_generation_config,
        )

        if mask:
            request["outPaintingParams"]["maskImage"] = await alpha_mask_to_bw(
                mask, invert=True
            )

        self._apply_extra_params(request, "outPaintingParams")
        self._response_quality = "medium"
        return request

    def _get_request_background_removal(
        self,
        image_generation_config: _ImageGenerationConfig,
        image: str,
        mask: str | None,
    ) -> _Request:
        """Get request for BACKGROUND_REMOVAL task.

        Args:
            image_generation_config: Image generation configuration.
            image: Base64-encoded source image.
            mask: Optional mask (not supported).

        Returns:
            Request object for BACKGROUND_REMOVAL task.
        """
        self._validate_no_mask(mask, reason="with BACKGROUND_REMOVAL taskType")
        request = _Request(
            taskType="BACKGROUND_REMOVAL",
            backgroundRemovalParams=_BackgroundRemovalParams(image=image),
            imageGenerationConfig=image_generation_config,
        )

        self._apply_extra_params(request, "backgroundRemovalParams")
        self._response_quality = "medium"
        return request

    def _get_request_virtual_try_on(
        self,
        image_generation_config: _ImageGenerationConfig,
        image: str,
        mask: str | None,
    ) -> _Request:
        """Get request for VIRTUAL_TRY_ON task.

        Args:
            image_generation_config: Image generation configuration.
            image: Base64-encoded source image.
            mask: Optional mask image for specifying the try-on region.

        Returns:
            Request object for VIRTUAL_TRY_ON task.
        """
        mask_str = self._validate_mask(mask, "with VIRTUAL_TRY_ON taskType")
        user_params: _VirtualTryOnParams = self._extra_params.get(  # type: ignore[assignment]
            "virtualTryOnParams", {}
        )
        mask_type = user_params.get("maskType", "PROMPT")

        if mask_type == "PROMPT":
            params = _VirtualTryOnParams(
                sourceImage=image,
                referenceImage=mask_str,
                maskType=mask_type,
                promptBasedMask=_PromptBasedMask(maskPrompt=self._prompt),
            )
            params["promptBasedMask"].update(user_params.pop("promptBasedMask", {}))
        elif mask_type == "GARMENT":
            params = _VirtualTryOnParams(
                sourceImage=image,
                referenceImage=mask_str,
                maskType="GARMENT",
                # Try using prompt as the default garment class
                garmentBasedMask=_GarmentBasedMask(garmentClass=self._prompt),  # type: ignore[typeddict-item]
            )
            params["garmentBasedMask"].update(user_params.pop("garmentBasedMask", {}))
        elif mask_type == "IMAGE":
            params = _VirtualTryOnParams(
                sourceImage=image,
                referenceImage=mask_str,
                maskType="IMAGE",
                # Try using prompt as default base64 encoded mask image
                imageBasedMask=_ImageBasedMask(
                    maskImage=get_data_uri_data(self._prompt)
                ),
            )
            params["imageBasedMask"].update(user_params.pop("imageBasedMask", {}))
        else:
            msg = f'Invalid virtualTryOnParams.maskType "{mask_type}". Must be one of "PROMPT", "GARMENT" or "IMAGE".'
            raise ApiError(msg)

        request = _Request(
            taskType="VIRTUAL_TRY_ON",
            virtualTryOnParams=params,
            imageGenerationConfig=image_generation_config,
        )
        self._apply_extra_params(request, "virtualTryOnParams")
        self._response_quality = "medium"
        return request

    def _get_request_text_image(
        self,
        image_generation_config: _ImageGenerationConfig,
        images: list[str] | None = None,
    ) -> _Request:
        """Get request for TEXT_IMAGE task.

        Args:
            image_generation_config: Image generation configuration.
            images: Optional list of images for condition image generation.

        Returns:
            Request object for TEXT_IMAGE task.
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

        self._apply_extra_params(request, "textToImageParams")
        self._apply_quality_and_style(request, "textToImageParams")
        return request

    def _get_request_color_guided_generation(
        self,
        image_generation_config: _ImageGenerationConfig,
        images: list[str] | None = None,
    ) -> _Request:
        """Get request for COLOR_GUIDED_GENERATION task.

        Args:
            image_generation_config: Image generation configuration.
            images: Optional list of images for reference image.

        Returns:
            Request object for COLOR_GUIDED_GENERATION task.
        """
        try:
            color_params = self._extra_params["colorGuidedGenerationParams"]
            colors: list[str] = color_params["colors"]  # type: ignore[call-overload,assignment,index]
        except (KeyError, TypeError, IndexError) as exc:
            msg = "Required parameter for COLOR_GUIDED_GENERATION: colorGuidedGenerationParams.colors"
            raise ApiError(msg) from exc

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

        self._apply_extra_params(request, "colorGuidedGenerationParams")
        self._apply_quality_and_style(request, None)
        return request

    def _get_request_image_variation(
        self, image_generation_config: _ImageGenerationConfig, images: list[str]
    ) -> _Request:
        """Get request for IMAGE_VARIATION task.

        Args:
            image_generation_config: Image generation configuration.
            images: List of base64-encoded images.

        Returns:
            Request object for IMAGE_VARIATION task.
        """
        request = _Request(
            taskType="IMAGE_VARIATION",
            imageVariationParams=_ImageVariationParams(images=images),
            imageGenerationConfig=image_generation_config,
        )
        if self._prompt:
            request["imageVariationParams"]["text"] = self._prompt

        self._apply_extra_params(request, "imageVariationParams")
        self._apply_quality_and_style(request, None)
        return request

    def _apply_quality_and_style(
        self, request: _Request, task_params_key: str | None
    ) -> None:
        """Apply quality and style settings to request.

        Args:
            request: The request to modify.
            task_params_key: Key for task-specific params that support style (e.g., "textToImageParams").
        """
        # Apply quality settings
        if "imageGenerationConfig" in request and (
            resolved := resolve_amz_quality_echo(self._quality)
        ):
            amz_quality, self._response_quality = resolved
            request["imageGenerationConfig"]["quality"] = amz_quality

        # Apply style if specified and supported by the task
        if self._style and task_params_key:
            request[task_params_key]["style"] = self._style.upper()  # type: ignore[literal-required]


class ImageModel(ImageModelBase[_Request, _Response, _ImageGenerationJob]):
    """Amazon Nova Canvas image model."""

    __slots__ = ()

    MATCHER = "amazon.nova-canvas"
    IMAGE_GENERATION_JOB_CLASS = _ImageGenerationJob
