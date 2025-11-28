"""OpenAI-compatible Images API implementation using AWS Bedrock.

This module implements the /v1/images/edits endpoint following the OpenAI API
specification, calling AWS Bedrock image generation models (e.g., Amazon Nova Canvas)
to edit images using inpainting techniques.
"""

from asyncio import create_task, gather
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from sse_starlette import EventSourceResponse

from stdapi.auth import authenticate
from stdapi.aws_bedrock import get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.image import get_image_model
from stdapi.monitoring import REQUEST_TIME, log_request_params, log_request_stream_event
from stdapi.openai_exceptions import OpenaiError, OpenaiUnsupportedModelError
from stdapi.routes._images_common import build_images_response
from stdapi.routes.openai_images_generations import stream_generator
from stdapi.tokenizer import estimate_token_count
from stdapi.types.openai_images import (
    ImageBackgroundAuto,
    ImageEditParams,
    ImageInputFidelity,
    ImageOutputFormats,
    ImagesResponse,
)
from stdapi.utils import read_and_b64encode_file, validation_error_handler

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=["images", "openai"]
)

#: Includes model fields and file parameters handled separately in the route
_KNOWN_PARAMS = set(ImageEditParams.model_fields.keys()) | {"image", "mask"}


@router.post(
    "/images/edits",
    response_model=None,
    summary="OpenAI - /v1/images/edits",
    description="Creates an edited or extended image given an original image and a prompt.",
    response_description="Image edit response in OpenAI format",
    responses={
        200: {"description": "Images successfully edited."},
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
    },
    response_model_exclude_none=True,
)
async def edit_images(
    http_request: Request,
    *,
    image: Annotated[
        list[UploadFile], File(description="The image(s) to edit.", min_length=1)
    ],
    prompt: Annotated[
        str,
        Form(
            description="A text description of the desired image(s). "
            "Required for a majority of models."
        ),
    ] = "",
    model: Annotated[
        str,
        Form(
            description="The model to use for image generation.",
            min_length=1,
            max_length=255,
        ),
    ],
    mask: Annotated[
        UploadFile | None,
        File(
            description="An additional image indicating where the image should be edited. "
            "The mask format is model-specific and may be a black/white image or "
            "an image with transparency (e.g. where alpha is zero indicates areas to edit)."
        ),
    ] = None,
    response_format: Annotated[
        str,
        Form(
            description="The format in which the generated images are returned. "
            "Must be one of url or b64_json. "
            "URLs are only valid for 60 minutes after the image has been generated.\n"
            "This parameter isn't supported with streaming which will always "
            "return base64-encoded images."
        ),
    ] = "url",
    n: Annotated[
        int, Form(description="The number of images to generate.", ge=1, le=10)
    ] = 1,
    size: Annotated[
        str,
        Form(
            description="The size of the generated images."
            "\nSupported values depend on the model. "
            "With some models, output size may be different.",
            pattern=r"^(\d+)x(\d+)$",
        ),
    ] = "1024x1024",
    user: Annotated[
        str | None,
        Form(
            description="A unique identifier representing your end-user, which can help to monitor and detect abuse.",
            min_length=1,
            max_length=255,
        ),
    ] = None,
    background: Annotated[
        ImageBackgroundAuto,
        Form(
            description="Allows to set transparency for the background of the generated image(s).\n"
            "If `transparent`, the output format needs to support transparency, "
            "so it should be set to either `png` (default value) or `webp`."
            "\ntransparent is UNSUPPORTED on this implementation."
        ),
    ] = "auto",
    input_fidelity: Annotated[
        ImageInputFidelity,
        Form(
            description="Control how much effort the model will exert to match the style and "
            "features, especially facial features, of input images."
            "\nUNSUPPORTED on this implementation."
        ),
    ] = "low",
    output_compression: Annotated[
        int,
        Form(
            description="The compression level (0-100%) for the generated images.",
            ge=1,
            le=100,
        ),
    ] = 100,
    output_format: Annotated[
        ImageOutputFormats | None,
        Form(
            description="The format in which the generated images are returned. "
            "Must be one of `png`, `jpeg`, or `webp`."
        ),
    ] = None,
    partial_images: Annotated[
        int | None,
        Form(
            description="The number of partial images to generate.\n"
            "This parameter is used for streaming responses that return partial images. "
            "Value must be between 0 and 3. "
            "When set to 0, the response will be a single image sent in one streaming event.\n"
            "Note that the final image may be sent before the full number of partial images "
            "are generated if the full image is generated more quickly.\n"
            "Partial images are only sent if the model supports it.",
            ge=0,
            le=3,
        ),
    ] = None,
    quality: Annotated[
        str,
        Form(
            description="The quality of the image that will be generated.\n"
            "`auto` (default value) will automatically select the best quality for the given model.\n"
            "Supported values depend on the model.",
            min_length=1,
            max_length=255,
        ),
    ] = "auto",
    stream: Annotated[
        bool, Form(description="Generate the image in streaming mode.")
    ] = False,
    _: Annotated[None, Depends(authenticate)] = None,
) -> ImagesResponse | EventSourceResponse:
    """Edit images using a prompt and mask.

    Args:
        http_request: FastAPI request object to extract extra form parameters.
        image: The image(s) to edit (single or multiple).
        prompt: A text description of the desired image(s).
        model: The model to use for image generation.
        mask: An additional image indicating where the image should be edited.
        response_format: The format in which the generated images are returned.
        n: The number of images to generate.
        size: The size of the generated images.
        user: A unique identifier representing your end-user.
        background: Allows to set transparency for the background.
        input_fidelity: Control how much effort the model will exert to match the input.
        output_compression: The compression level (0-100%) for the generated images.
        output_format: The format in which the generated images are returned.
        partial_images: The number of partial images to generate.
        quality: The quality of the image that will be generated.
        stream: Generate the image in streaming mode.

    Returns:
        ImagesResponse containing edited image URLs or base64 data, or
        EventSourceResponse for streaming requests.

    Raises:
        HTTPException: With 404 if the model does not exist; 400 on unsupported
            options or invalid values.
    """
    with validation_error_handler():
        request = ImageEditParams(
            prompt=prompt,
            model=model,
            response_format=response_format,  # type: ignore[arg-type]
            n=n,
            size=size,
            user=user,
            background=background,
            input_fidelity=input_fidelity,
            output_compression=output_compression,
            output_format=output_format,
            partial_images=partial_images,
            quality=quality,
            stream=stream,
            **{  # type: ignore[arg-type]
                k: v
                for k, v in (await http_request.form()).items()
                if k not in _KNOWN_PARAMS
            },
        )
    log_request_params(request, user_id=request.user)
    try:
        await validate_model(
            request.model, input_modality="IMAGE", output_modality="IMAGE"
        )
    except OpenaiUnsupportedModelError as Error:
        # This route does not return standard 404 error if invalid model.
        raise OpenaiError(Error.args[0]) from None

    width, height = map(int, request.size.split("x"))
    job = get_image_model(request.model).get_image_edit_job(
        prompt=request.prompt,
        count=request.n,
        width=width,
        height=height,
        output_format=request.output_format,
        output_compression=request.output_compression,
        is_url=request.response_format == "url" and not request.stream,
        extra_params=get_extra_model_parameters(request.model, request),
    )

    # Read and encode image files in parallel
    tasks = [read_and_b64encode_file(img) for img in image]
    if mask:
        tasks.append(read_and_b64encode_file(mask))
        *images_b64, mask_b64 = await gather(*tasks)
    else:
        images_b64 = await gather(*tasks)
        mask_b64 = None

    # Handle streaming requests
    if request.stream:
        return EventSourceResponse(
            await log_request_stream_event(
                stream_generator(
                    image_stream=job.edit_images_stream(
                        images=images_b64,
                        mask=mask_b64,
                        partial_images=request.partial_images,
                    ),
                    job=job,
                    created=int(REQUEST_TIME.get().timestamp()),
                )
            )
        )

    # Handle non-streaming requests
    token_task = create_task(estimate_token_count(request.prompt))
    results = await job.edit_images(images=images_b64, mask=mask_b64)
    text_tokens = await token_task or 0

    return await build_images_response(
        job=job,
        results=results,
        response_format=request.response_format,
        image_count=request.n,
        text_tokens=text_tokens,
        image_tokens=len(image) + (0 if mask is None else 1),
    )
