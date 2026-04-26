"""OpenAI Responses API endpoint implementation.

This module implements the OpenAI-compatible /v1/responses endpoint, providing
AWS Bedrock Converse integration while maintaining full API compatibility.

The module provides:
    - POST /v1/responses — create a model response
"""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends

from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.chat import get_chat_model
from stdapi.monitoring import REQUEST_ID, REQUEST_TIME, log_request_params
from stdapi.types.openai_responses import Response, ResponseCreateParams

if TYPE_CHECKING:
    from sse_starlette import EventSourceResponse

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1/responses",
    tags=["Responses", TAG_OPENAI],
)


@router.post(
    "",
    summary="OpenAI - /v1/responses",
    operation_id="openai_response",
    description=(
        "Creates a model response. Provide text or other input and the model "
        "will generate a response. Compatible with the OpenAI Responses API.\n\n"
        "Supports streaming, tool calling, and structured outputs."
    ),
    response_description="A model response.",
    status_code=200,
    response_model=Response,
    responses={
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
    },
    response_model_exclude_none=True,
)
async def create_response(
    request: ResponseCreateParams, _: Annotated[None, Depends(authenticate)] = None
) -> Response | EventSourceResponse:
    """Create a model response using AWS Bedrock Converse APIs.

    Compatible with the OpenAI Responses API. Maps input items and parameters
    to the Bedrock Converse/ConverseStream APIs and returns an OpenAI-compatible
    response.

    Args:
        request: Responses API creation request.

    Returns:
        - Response when stream is False.
        - EventSourceResponse streaming ResponseStreamEvent events when stream is True.

    Raises:
        ApiError: If model is invalid or does not support text output.
    """
    log_request_params(request, user_id=request.safety_identifier or request.user)
    response_id = f"resp-{REQUEST_ID.get()}"
    created_at = REQUEST_TIME.get().timestamp()
    return await get_chat_model(
        (
            await validate_model(
                request.model, input_modality="TEXT", output_modality="TEXT"
            )
        ).id
    ).create_response(request, response_id, created_at)
