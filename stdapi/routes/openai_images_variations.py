"""OpenAI-compatible Images API implementation using AWS Bedrock.

This module implements the /v1/images/variations endpoint following the OpenAI API
specification, calling AWS Bedrock image generation models (e.g., Amazon Titan Image Generator)
to create variations of existing images.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from stdapi.auth import authenticate
from stdapi.aws_bedrock import get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.image import get_image_model
from stdapi.monitoring import log_request_params
from stdapi.openai_exceptions import OpenaiError, OpenaiUnsupportedModelError
from stdapi.routes._images_common import build_images_response
from stdapi.types.openai_images import ImagesResponse, ImageVariationParams
from stdapi.utils import read_and_b64encode_file, validation_error_handler

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=["images", "openai"]
)

#: Includes model fields and file parameters handled separately in the route
_KNOWN_PARAMS = set(ImageVariationParams.model_fields.keys()) | {"image"}


@router.post(
    "/images/variations",
    summary="OpenAI - /v1/images/variations",
    description="Creates a variation of a given image.",
    response_description="Image variation response in OpenAI format",
    responses={
        200: {"description": "Image variations successfully created."},
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
    },
    response_model_exclude_none=True,
)
async def create_image_variations(
    http_request: Request,
    image: Annotated[
        UploadFile,
        File(..., description="The image to use as the basis for the variation(s)."),
    ],
    *,
    model: Annotated[
        str,
        Form(
            description="The model to use for image generation.",
            min_length=1,
            max_length=255,
        ),
    ],
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

    Args:
        http_request: FastAPI request object to extract extra form parameters.
        image: The image to use as the basis for the variation(s).
        model: The model to use for image generation.
        response_format: The format in which the generated images are returned.
        n: The number of images to generate.
        size: The size of the generated images.
        user: A unique identifier representing your end-user.

    Returns:
        ImagesResponse containing image variation URLs or base64 data.

    Raises:
        HTTPException: With 404 if the model does not exist; 400 on unsupported
            options or invalid values.
    """
    with validation_error_handler():
        request = ImageVariationParams(
            model=model,
            response_format=response_format,  # type: ignore[arg-type]
            n=n,
            size=size,
            user=user,
            **{  # type: ignore[arg-type]
                k: v
                for k, v in (await http_request.form()).items()
                if k not in _KNOWN_PARAMS
            },
        )
    log_request_params(request, user_id=request.user)
    try:
        model_id = (
            await validate_model(
                request.model, input_modality="IMAGE", output_modality="IMAGE"
            )
        ).id
    except OpenaiUnsupportedModelError as Error:
        # This route does not return standard 404 error if invalid model.
        raise OpenaiError(Error.args[0]) from None

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

    results = await job.create_variations(images=[await read_and_b64encode_file(image)])
    return await build_images_response(
        job=job,
        results=results,
        response_format=request.response_format,
        image_count=request.n,
        image_tokens=1,
    )
