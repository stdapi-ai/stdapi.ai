"""OpenAI-compatible Images API implementation using AWS Bedrock.

This module implements the /v1/images/variations endpoint following the OpenAI API
specification, calling AWS Bedrock image generation models (e.g., Amazon Titan Image Generator)
to create variations of existing images.

Two request formats are supported:
- ``multipart/form-data``: binary file upload via the ``image`` field.
- ``application/json``: structured body with an ``image`` field referencing a
  Files API identifier or HTTP/data URL.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import ValidationError

from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.aws_bedrock import get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.input_file import InputFile
from stdapi.models import validate_model
from stdapi.models.capabilities import Capability, register_route_capability
from stdapi.models.image import get_image_model
from stdapi.monitoring import log_request_params
from stdapi.routes._images_common import build_images_response
from stdapi.types.openai_images import (
    ImagesResponse,
    ImageVariationJsonBody,
    ImageVariationParams,
)
from stdapi.utils import validation_error_handler

register_route_capability(
    "openai_image_variation",
    f"{SETTINGS.openai_routes_prefix}/v1/images/variations",
    "IMAGE",
    "IMAGE",
    Capability.IMAGE_VARIATION,
)

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=["Images", TAG_OPENAI]
)

#: Includes model fields and file parameters handled separately in the route
_KNOWN_PARAMS = set(ImageVariationParams.model_fields.keys()) | {"image"}


@router.post(
    "/images/variations",
    summary="OpenAI - /v1/images/variations",
    operation_id="openai_image_variation",
    description="Creates a variation of a given image.",
    response_description="The response from the image generation endpoint.",
    responses={
        200: {"description": "Image variations successfully created."},
        400: {"description": "Invalid request or unsupported parameters."},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "examples": {
                        "url": {
                            "summary": "Return image URL",
                            "value": {
                                "model": "amazon.nova-canvas-v1:0",
                                "response_format": "url",
                                "n": 1,
                                "size": "1024x1024",
                            },
                        },
                        "b64": {
                            "summary": "Return base64 data",
                            "value": {
                                "model": "amazon.nova-canvas-v1:0",
                                "response_format": "b64_json",
                                "n": 2,
                                "size": "512x512",
                            },
                        },
                    }
                },
                "application/json": {
                    "examples": {
                        "file_id": {
                            "summary": "Variation from Files API ID",
                            "value": {
                                "model": "amazon.nova-canvas-v1:0",
                                "image": {
                                    "file_id": "file-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                                },
                                "response_format": "url",
                                "n": 1,
                                "size": "1024x1024",
                            },
                        },
                        "image_url": {
                            "summary": "Variation from URL",
                            "value": {
                                "model": "amazon.nova-canvas-v1:0",
                                "image": {"image_url": "https://example.com/image.png"},
                                "response_format": "b64_json",
                                "n": 2,
                                "size": "512x512",
                            },
                        },
                    }
                },
            }
        }
    },
    response_model_exclude_none=True,
)
async def create_image_variations(
    http_request: Request,
    image: Annotated[
        UploadFile | None,
        File(
            description="The image to use as the basis for the variation(s). "
            "Accepts a binary file upload. "
            "For Files API identifiers or URLs, use ``application/json`` body instead."
        ),
    ] = None,
    *,
    model: Annotated[
        str,
        Form(
            description="The model to use for image generation.",
            min_length=1,
            max_length=255,
        ),
    ] = "",
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
    _: Annotated[None, Depends(authenticate)] = None,
) -> ImagesResponse:
    """Create variations of a given image.

    Accepts either ``multipart/form-data`` (binary file upload) or
    ``application/json`` (``image`` field with a Files API ID or URL).

    Args:
        http_request: FastAPI request object used to detect content-type and
            read the raw body for JSON requests.
        image: Binary file upload for multipart requests.
        model: The model to use for image generation.
        response_format: The format in which the generated images are returned.
        n: The number of images to generate.
        size: The size of the generated images.
        user: A unique identifier representing your end-user.

    Returns:
        ImagesResponse containing image variation URLs or base64 data.

    Raises:
        ApiError: With 404 if the model does not exist; 400 on unsupported
            options or invalid values.
    """
    content_type = http_request.headers.get("content-type", "")

    if "application/json" in content_type:
        # JSON body: image referenced by file_id or image_url
        with validation_error_handler():
            body = ImageVariationJsonBody.model_validate(await http_request.json())
        input_image: InputFile = body.image.input_file
        request: ImageVariationJsonBody | ImageVariationParams = body
    else:
        # Multipart form-data: binary file upload only
        form_data = await http_request.form()
        if image is None:
            msg = "ValidationError"
            raise ValidationError.from_exception_data(
                msg,
                [
                    {
                        "type": "missing",
                        "loc": ("body", "image"),
                        "input": None,
                        "ctx": {},
                    }
                ],
            )
        with validation_error_handler():
            input_image = InputFile(image)
            request = ImageVariationParams(
                model=model,
                response_format=response_format,  # type: ignore[arg-type]
                n=n,
                size=size,
                user=user,
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
    job = get_image_model(model_id).get_image_variation_job(
        count=request.n,
        width=width,
        height=height,
        output_format=None,
        output_compression=100,
        is_url=request.response_format == "url",
        extra_params=get_extra_model_parameters(model_id, request),
    )

    results = await job.create_variations(images=[await input_image.to_base64()])
    return await build_images_response(
        job=job,
        results=results,
        response_format=request.response_format,
        image_count=request.n,
        image_tokens=1,
    )
