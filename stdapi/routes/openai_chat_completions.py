"""Chat completions and responses endpoints implementation.

This module implements OpenAI-compatible endpoints for chat completions and responses,
providing AWS Bedrock integration while maintaining API compatibility. It handles both
streaming and non-streaming chat completions, tool calling, and various OpenAI parameters.

The module provides:
    - OpenAI-compatible chat completions API endpoint
    - AWS Bedrock integration for various language models
    - Streaming and non-streaming response modes
    - Tool calling and function execution support
    - Request validation and parameter conversion
    - Response formatting and usage tracking

Classes:
    CompletionCreateParams: Pydantic model for chat completion requests

Functions:
    create_chat_completion: Main FastAPI endpoint for chat completions
    Various helper functions for message conversion, validation, and response processing
"""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends

from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.chat import get_chat_model
from stdapi.monitoring import REQUEST_ID, REQUEST_TIME, log_request_params
from stdapi.types.openai_chat_completions import ChatCompletion, CompletionCreateParams

if TYPE_CHECKING:
    from sse_starlette import EventSourceResponse

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1/chat", tags=["Chat", TAG_OPENAI]
)


@router.post(
    "/completions",
    summary="OpenAI - /v1/chat/completions",
    operation_id="openai_chat_completion",
    description=(
        "Creates a model response for the given chat conversation. Learn more in the text generation, vision, and audio guides.\n"
        "Parameter support can differ depending on the model used to generate the response, particularly for newer reasoning models. "
        "Parameters that are only supported for reasoning models are noted below.\n"
        "Returns a chat completion object, or a streamed sequence of chat completion chunk objects if the request is streamed."
    ),
    response_description="Represents a chat completion response returned by model, based on the provided input.",
    status_code=200,
    response_model=ChatCompletion,
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "id": "chatcmpl-f6ed35b89b77488f8c481eb0a26ac1bf",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "index": 0,
                                "message": {
                                    "content": "I'm an AI assistant.",
                                    "role": "assistant",
                                },
                            }
                        ],
                        "created": 1740134957,
                        "model": "amazon.nova-micro-v1:0",
                        "object": "chat.completion",
                        "usage": {
                            "completion_tokens": 16,
                            "prompt_tokens": 11,
                            "total_tokens": 27,
                        },
                    }
                }
            },
        },
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "basic": {
                            "summary": "Basic chat completion",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "messages": [
                                    {"role": "user", "content": "Hello, how are you?"}
                                ],
                            },
                        },
                        "streaming": {
                            "summary": "Streaming response",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "messages": [
                                    {"role": "user", "content": "Tell me a story"}
                                ],
                                "stream": True,
                            },
                        },
                        "with_params": {
                            "summary": "With parameters",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": "You are a helpful assistant.",
                                    },
                                    {
                                        "role": "user",
                                        "content": "Explain quantum computing",
                                    },
                                ],
                                "temperature": 0.7,
                                "max_tokens": 1000,
                            },
                        },
                    }
                }
            }
        }
    },
    response_model_exclude_none=True,
)
async def create_chat_completion(
    request: CompletionCreateParams, _: Annotated[None, Depends(authenticate)] = None
) -> ChatCompletion | EventSourceResponse:
    """Create a chat completion using AWS Bedrock Converse APIs.

    This endpoint is compatible with OpenAI's Chat Completions API. It maps the
    incoming OpenAI-style chat messages and parameters to AWS Bedrock's
    converse/converse_stream APIs and returns OpenAI-compatible responses.

    Args:
        request: Chat completion creation request following OpenAI spec.

    Returns:
        - ChatCompletion when stream is False.
        - EventSourceResponse streaming ChatCompletionChunk events when stream is True.

    Raises:
        ApiError: If model is invalid or does not support text output.
    """
    log_request_params(request, user_id=request.safety_identifier or request.user)
    return await get_chat_model(
        (
            await validate_model(
                request.model, input_modality="TEXT", output_modality="TEXT"
            )
        ).id
    ).create_completion(
        request, f"chatcmpl-{REQUEST_ID.get()}", int(REQUEST_TIME.get().timestamp())
    )
