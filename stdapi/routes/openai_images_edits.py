"""OpenAI-compatible Images API implementation using AWS Bedrock.

This module implements the /v1/images/edits endpoint following the OpenAI API
specification, calling AWS Bedrock image generation models (e.g., Amazon Nova Canvas)
to edit images using inpainting techniques.

Two request formats are supported:
- ``multipart/form-data``: binary file uploads via ``image`` / ``mask`` fields.
- ``application/json``: structured body with an ``images`` array of ``ImageRef``
  objects (Files API identifiers or HTTP/data URLs) and an optional ``mask``.
"""

from asyncio import create_task, gather
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import AliasChoices, ValidationError
from sse_starlette import EventSourceResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

if TYPE_CHECKING:
    from starlette.datastructures import FormData

from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.aws_bedrock import get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.input_file import InputFile
from stdapi.models import validate_model
from stdapi.models.image import get_image_model
from stdapi.monitoring import REQUEST_TIME, log_request_params, log_request_stream_event
from stdapi.routes._images_common import build_images_response
from stdapi.routes.openai_images_generations import stream_generator
from stdapi.tokenizer import estimate_token_count
from stdapi.types.openai_images import (
    ImageBackgroundAuto,
    ImageEditJsonBody,
    ImageEditParams,
    ImageInputFidelity,
    ImageOutputFormats,
    ImagesResponse,
    _ImageEditCommonParams,
)
from stdapi.utils import validation_error_handler

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=["Images", TAG_OPENAI]
)

#: Includes model fields and file parameters handled separately in the route
_KNOWN_PARAMS = set(ImageEditParams.model_fields.keys()) | {"image", "image[]", "mask"}


def _merge_image_parameters(
    form_data: FormData, image_param: list[UploadFile] | None
) -> list[UploadFile]:
    """Merge image files from both ``image`` and ``image[]`` form parameters.

    FastAPI does not support ``validation_alias`` for ``File`` parameters in
    multipart/form-data requests — alias resolution happens before Pydantic
    validation.  This function provides the workaround by manually extracting
    ``image[]``-keyed files from the raw form data and merging them with any
    files already captured by the ``image`` FastAPI parameter.

    Args:
        form_data: Parsed multipart form data from the request.
        image_param: Files uploaded via the ``image`` parameter, or ``None``.

    Returns:
        Combined list of ``UploadFile`` objects from both parameters.

    Raises:
        ValidationError: If no images are provided, or if a non-file value is
            submitted under the ``image[]`` key.
    """
    images: list[UploadFile] = list(image_param or [])

    for key, value in form_data.multi_items():
        if key == "image[]":
            if isinstance(value, StarletteUploadFile):
                images.append(value)  # type: ignore[arg-type]
            else:
                msg = "ValidationError"
                raise ValidationError.from_exception_data(
                    msg,
                    [
                        {
                            "type": "is_instance_of",
                            "loc": ("body", "image[]"),
                            "input": value,
                            "ctx": {"class": "UploadFile"},
                        }
                    ],
                )

    if not images:
        msg = "ValidationError"
        raise ValidationError.from_exception_data(
            msg,
            [
                {
                    "type": "too_short",
                    "loc": ("body", "image"),
                    "input": [],
                    "ctx": {"field_type": "List", "min_length": 1, "actual_length": 0},
                }
            ],
        )

    return images


@router.post(
    "/images/edits",
    response_model=None,
    summary="OpenAI - /v1/images/edits",
    operation_id="openai_image_edit",
    description="Creates an edited or extended image given one or more source images and a prompt.",
    response_description="The response from the image generation endpoint.",
    responses={
        200: {"description": "Images successfully edited."},
        400: {"description": "Invalid request or unsupported parameters."},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "examples": {
                        "inpaint": {
                            "summary": "Inpaint with mask",
                            "value": {
                                "model": "amazon.nova-canvas-v1:0",
                                "prompt": "A red apple on a wooden table",
                                "response_format": "url",
                                "n": 1,
                                "size": "1024x1024",
                            },
                        },
                        "stream": {
                            "summary": "Streaming response",
                            "value": {
                                "model": "amazon.nova-canvas-v1:0",
                                "prompt": "A sunset over mountains",
                                "stream": True,
                                "partial_images": 2,
                            },
                        },
                    }
                },
                "application/json": {
                    "examples": {
                        "file_id": {
                            "summary": "Edit image by Files API ID",
                            "value": {
                                "model": "amazon.nova-canvas-v1:0",
                                "prompt": "A red apple on a wooden table",
                                "images": [
                                    {"file_id": "file-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}
                                ],
                                "response_format": "url",
                                "n": 1,
                                "size": "1024x1024",
                            },
                        },
                        "image_url": {
                            "summary": "Edit image by URL",
                            "value": {
                                "model": "amazon.nova-canvas-v1:0",
                                "prompt": "A sunset over mountains",
                                "images": [
                                    {"image_url": "https://example.com/image.png"}
                                ],
                                "stream": True,
                                "partial_images": 2,
                            },
                        },
                    }
                },
            }
        }
    },
    response_model_exclude_none=True,
)
async def edit_images(
    http_request: Request,
    *,
    image: Annotated[
        list[UploadFile] | None,
        File(
            description="The image(s) to edit. Accepts binary file uploads. "
            "For Files API identifiers or URLs, use ``application/json`` body instead.",
            validation_alias=AliasChoices("image", "image[]"),
        ),
    ] = None,
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
    ] = "",
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
    """Edit images using a prompt and optional mask.

    Accepts either ``multipart/form-data`` (binary file uploads) or
    ``application/json`` (``images`` array with Files API IDs or URLs).

    Args:
        http_request: FastAPI request object used to detect content-type and
            read the raw body for JSON requests.
        image: Binary file upload(s) for multipart requests.
        prompt: A text description of the desired image(s).
        model: The model to use for image generation.
        mask: Binary mask upload for multipart requests.
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
        ApiError: With 404 if the model does not exist; 400 on unsupported
            options or invalid values.
    """
    content_type = http_request.headers.get("content-type", "")

    if "application/json" in content_type:
        # JSON body: structured images array with file_id or image_url references
        with validation_error_handler():
            body = ImageEditJsonBody.model_validate(await http_request.json())
        input_images: list[InputFile] = [ref.input_file for ref in body.images]
        input_mask: InputFile | None = ref.input_file if (ref := body.mask) else None
        request: _ImageEditCommonParams = body
    else:
        # Multipart form-data: binary file uploads only
        form_data = await http_request.form()
        with validation_error_handler():
            images = _merge_image_parameters(form_data, image)
            input_images = [InputFile(img) for img in images]
            input_mask = InputFile(mask) if mask else None
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
                    k: v for k, v in form_data.items() if k not in _KNOWN_PARAMS
                },
            )

    log_request_params(request, user_id=request.user)
    model_id = (
        await validate_model(
            request.model,
            input_modality="IMAGE",
            output_modality="IMAGE",
            error_status=400,
        )
    ).id

    width, height = map(int, request.size.split("x"))
    job = get_image_model(model_id).get_image_edit_job(
        prompt=request.prompt,
        count=request.n,
        width=width,
        height=height,
        output_format=request.output_format,
        output_compression=request.output_compression,
        is_url=request.response_format == "url" and not request.stream,
        extra_params=get_extra_model_parameters(model_id, request),
    )

    if input_mask:
        *images_b64, mask_b64 = await gather(
            *(img.to_base64() for img in [*input_images, input_mask])
        )
    else:
        images_b64 = list(await gather(*(img.to_base64() for img in input_images)))
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
        image_tokens=len(input_images) + (0 if input_mask is None else 1),
    )
