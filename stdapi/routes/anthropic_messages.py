"""Anthropic Messages API endpoint implementation.

This module implements an Anthropic-compatible endpoint for the Messages API,
providing AWS Bedrock integration while maintaining API compatibility. It handles
both streaming and non-streaming message creation, tool calling, and extended thinking.

Functions:
    create_message: Main FastAPI endpoint for Anthropic message creation.
"""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends

from stdapi.api_providers.anthropic import TAG_ANTHROPIC
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.chat import get_chat_model
from stdapi.models.chat._adapters._anthropic_message import count_tokens_via_bedrock
from stdapi.monitoring import REQUEST_ID, log_request_params, log_response_params
from stdapi.types.anthropic_messages import (
    Message,
    MessageCountTokensParams,
    MessageCreateParams,
    MessageTokensCount,
)

if TYPE_CHECKING:
    from sse_starlette import EventSourceResponse

router: APIRouter = APIRouter(
    prefix=f"{SETTINGS.anthropic_routes_prefix}/v1", tags=["Chat", TAG_ANTHROPIC]
)


@router.post(
    "/messages",
    summary="Anthropic - /v1/messages",
    description=(
        "Send a structured list of input messages with text and / or image content, and the model will generate the next message in the conversation.\n"
        "The Messages API can be used for either single queries or stateless multi-turn conversations."
    ),
    response_description="Represents a response returned by model, based on the provided input.",
    status_code=200,
    response_model=Message,
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "id": "msg-f6ed35b89b77488f8c481eb0a26ac1bf",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "I'm an AI assistant."}],
                        "model": "amazon.nova-micro-v1:0",
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 11, "output_tokens": 16},
                    }
                }
            },
        },
        400: {"description": "Invalid request or unsupported parameters."},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "basic": {
                            "summary": "Basic message",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "messages": [
                                    {"role": "user", "content": "Hello, how are you?"}
                                ],
                                "max_tokens": 1024,
                            },
                        },
                        "streaming": {
                            "summary": "Streaming response",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "messages": [
                                    {"role": "user", "content": "Tell me a story"}
                                ],
                                "max_tokens": 1024,
                                "stream": True,
                            },
                        },
                        "with_system": {
                            "summary": "With system prompt",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "system": "You are a helpful assistant.",
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Explain quantum computing",
                                    }
                                ],
                                "max_tokens": 2048,
                                "temperature": 0.7,
                            },
                        },
                    }
                }
            }
        }
    },
    response_model_exclude_none=True,
)
async def create_message(
    request: MessageCreateParams, _: Annotated[None, Depends(authenticate)] = None
) -> Message | EventSourceResponse:
    """Create a message using AWS Bedrock Converse APIs.

    This endpoint is compatible with Anthropic's Messages API. It maps the
    incoming Anthropic-style messages and parameters to AWS Bedrock's
    converse/converse_stream APIs and returns Anthropic-compatible responses.

    Args:
        request: Message creation request following Anthropic spec.

    Returns:
        - Message when stream is False.
        - EventSourceResponse streaming MessageStreamEvent events when stream is True.

    Raises:
        ApiError: If model is invalid or does not support text output.
    """
    log_request_params(
        request, user_id=request.metadata.user_id if request.metadata else None
    )
    return await get_chat_model(
        (
            await validate_model(
                request.model,
                input_modality="TEXT",
                output_modality="TEXT",
                error_status=400,
            )
        ).id
    ).create_message(request, f"msg_{REQUEST_ID.get()}")


@router.post(
    "/messages/count_tokens",
    summary="Anthropic - /v1/messages/count_tokens",
    description=(
        "Count the number of tokens in a Message.\n"
        "The Token Count API can be used to count the number of tokens in a Message, "
        "including tools, images, and documents, without creating it."
    ),
    response_description="Token count for the provided message parameters.",
    status_code=200,
    response_model=MessageTokensCount,
    responses={
        200: {
            "description": "Successful Response",
            "content": {"application/json": {"example": {"input_tokens": 2095}}},
        },
        400: {"description": "Invalid request or unsupported parameters."},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "basic": {
                            "summary": "Basic count tokens",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "messages": [
                                    {"role": "user", "content": "Hello, how are you?"}
                                ],
                            },
                        },
                        "with_system": {
                            "summary": "With system prompt",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "system": "You are a helpful assistant.",
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Explain quantum computing",
                                    }
                                ],
                            },
                        },
                    }
                }
            }
        }
    },
    response_model_exclude_none=True,
)
async def count_tokens(
    request: MessageCountTokensParams, _: Annotated[None, Depends(authenticate)] = None
) -> MessageTokensCount:
    """Count the number of tokens in a message.

    This endpoint counts the tokens for the provided messages,
    system prompt, and tools using the AWS Bedrock CountTokens API
    without creating a message.

    Args:
        request: Count tokens request following Anthropic spec.

    Returns:
        MessageTokensCount with the input token count.

    Raises:
        ApiError: If model is invalid or does not support text output.
    """
    log_request_params(request)
    model = await validate_model(
        request.model, input_modality="TEXT", output_modality="TEXT", error_status=400
    )
    return log_response_params(
        MessageTokensCount(
            input_tokens=await count_tokens_via_bedrock(
                request, model.get_id(inference_profile=False), model.regions[0]
            )
        )
    )
