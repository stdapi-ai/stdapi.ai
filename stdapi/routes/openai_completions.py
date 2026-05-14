"""OpenAI-compatible Completions API endpoint.

This module implements the OpenAI-compatible Completions API endpoint, routing
``POST /v1/completions`` through ``ChatModelBase.create_text_completion`` to the
configured AWS Bedrock model. Recommended for MCP and simple text-only flows; for
richer inputs use ``/v1/chat/completions`` or ``/v1/responses``.
"""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends

from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.capabilities import register_route_capability
from stdapi.models.chat import get_chat_model
from stdapi.monitoring import REQUEST_ID, REQUEST_TIME, log_request_params
from stdapi.types.openai_completions import Completion, CompletionCreateParams

if TYPE_CHECKING:
    from sse_starlette import EventSourceResponse

register_route_capability(
    "openai_completion",
    f"{SETTINGS.openai_routes_prefix}/v1/completions",
    "TEXT",
    "TEXT",
)

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=["Chat", TAG_OPENAI]
)


@router.post(
    "/completions",
    summary="Generate a text completion (OpenAI format)",
    operation_id="openai_completion",
    description=(
        "Creates a text completion (OpenAI Completions API). Returns a "
        "``Completion`` object, or a stream of chunks terminated by "
        "``data: [DONE]`` when ``stream=true``.\n\n"
        "**Prompt shapes — how each ``prompt`` value is handled:**\n"
        '- ``"text"`` → one text completion (one choice).\n'
        '- ``["t1", "t2", …]`` → one choice per prompt (batch).\n'
        "- URL (``https://``, ``s3://``, ``data:``, ``file-id:<id>``) → the file is "
        "forwarded to the model using its detected modality (``image``, ``video``, "
        "``audio``, ``document``); the model returns an error if it does not support "
        "that modality.\n"
        '- ``["instruction", <file>, <file>, …]`` (exactly one text + ≥1 files) → '
        "packed in input order into a single multimodal request, returning one "
        "choice. **This is the recommended shape for analysing files with an "
        "instruction** (e.g. describing an image, summarising a PDF).\n"
        "- ``[<file>, <file>, …]`` (files only, no text) → one choice per file, each "
        "forwarded as its detected modality.\n\n"
        "**Streaming:** text deltas arrive as SSE chunks carrying "
        "``choices[0].index`` so clients can attribute each delta to its prompt or "
        "choice; the terminal chunk per choice has ``finish_reason`` set."
    ),
    response_description="A text completion response for the given prompt.",
    status_code=200,
    response_model=Completion,
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "id": "cmpl-6c6bfcd3b39e4d0f8c481eb0a26ac1bf",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "index": 0,
                                "text": "The capital of France is Paris.",
                            }
                        ],
                        "created": 1740134957,
                        "model": "amazon.nova-micro-v1:0",
                        "object": "text_completion",
                        "usage": {
                            "completion_tokens": 8,
                            "prompt_tokens": 6,
                            "total_tokens": 14,
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
                            "summary": "Basic completion",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "prompt": "Hello, how are you?",
                                "max_tokens": 20,
                            },
                        },
                        "streaming": {
                            "summary": "Streaming response",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "prompt": "Tell me a short story",
                                "stream": True,
                            },
                        },
                        "with_params": {
                            "summary": "With parameters",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "prompt": "Explain quantum computing",
                                "temperature": 0.7,
                                "top_p": 0.9,
                                "max_tokens": 200,
                                "stop": ["\n\n"],
                            },
                        },
                        "batch": {
                            "summary": "Batch prompts",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "prompt": ["One plus one is", "Two plus two is"],
                                "max_tokens": 5,
                            },
                        },
                        "file_id": {
                            "summary": "File reference",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "prompt": "file-id:file-abc123",
                            },
                        },
                    }
                }
            }
        }
    },
    response_model_exclude_none=True,
)
async def create_completion(
    request: CompletionCreateParams, _: Annotated[None, Depends(authenticate)] = None
) -> Completion | EventSourceResponse:
    """Create a text completion using AWS Bedrock Converse APIs.

    This endpoint is compatible with OpenAI's Completions API. It maps the
    incoming OpenAI-style completion request to AWS Bedrock's converse API
    and returns OpenAI-compatible responses.

    Args:
        request: Completion creation request following OpenAI spec.

    Returns:
        - Completion when stream is False.
        - EventSourceResponse streaming CompletionChunk events when stream is True.

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
    ).create_text_completion(
        request, f"cmpl-{REQUEST_ID.get()}", int(REQUEST_TIME.get().timestamp())
    )
