"""Ollama-compatible ``/api/generate`` endpoint using AWS Bedrock chat models."""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends
from sse_starlette import EventSourceResponse
from starlette.responses import StreamingResponse

from stdapi.api_providers.ollama import NDJSON_MEDIA_TYPE, TAG_OLLAMA
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.capabilities import register_route_capability
from stdapi.models.chat import get_chat_model
from stdapi.models.chat._adapters import _ollama as ollama_adapter
from stdapi.monitoring import (
    REQUEST_ID,
    REQUEST_TIME,
    guard_ndjson_stream_errors,
    log_request_params,
    log_response_params,
)
from stdapi.types.ollama import GenerateRequest, GenerateResponse

if TYPE_CHECKING:
    from stdapi.types.openai_chat_completions import ChatCompletion

register_route_capability(
    "ollama_generate",
    f"{SETTINGS.ollama_routes_prefix}/api/generate",
    "TEXT",
    "TEXT",
    mcp_tool=False,
)

router = APIRouter(
    prefix=f"{SETTINGS.ollama_routes_prefix}/api", tags=["Completions", TAG_OLLAMA]
)


@router.post(
    "/generate",
    summary="Generate a response for a single prompt (Ollama format)",
    operation_id="ollama_generate",
    description=(
        "Creates a model response for a single `prompt` (Ollama Generate API).\n\n"
        "Supports images, structured output through `format`, and thinking "
        "through `think`. Streams newline-delimited JSON objects by default; "
        "set `stream` to false for a single response object.\n\n"
        "`raw`, `suffix`, `template` and `context` need the model's own prompt "
        "template and are not available; `logprobs` and `top_logprobs` are not "
        "available either. Send `prompt` with `system`, or use `ollama_chat` for "
        "a conversation."
    ),
    response_description="The generated text, or a stream of partial responses.",
    response_model=GenerateResponse,
    responses={
        200: {"description": "The response was generated."},
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "basic": {
                            "summary": "Single prompt",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "prompt": "Why is the sky blue?",
                                "stream": False,
                            },
                        }
                    }
                }
            }
        }
    },
    response_model_exclude_none=True,
)
async def generate(
    request: GenerateRequest, _: Annotated[None, Depends(authenticate)] = None
) -> GenerateResponse | StreamingResponse:
    """Create a response for a single prompt using AWS Bedrock Converse APIs.

    Args:
        request: Generate parameters following the Ollama API.

    Returns:
        The Ollama generate response, or a newline-delimited JSON stream of
        partial responses when ``stream`` is true. A request carrying no
        prompt answers the single done object Ollama answers a load or an
        unload with.

    Raises:
        ApiError: With 404 if the model does not exist; 400 on unsupported
            options or invalid values.
    """
    log_request_params(request)
    model_id = (
        await validate_model(
            request.model,
            input_modality="TEXT",
            output_modality="TEXT",
            route="ollama_generate",
        )
    ).id
    if not request.prompt:
        # An empty prompt is Ollama's load, and with `keep_alive` at zero its
        # unload: the model is named and answered for, never generated with.
        return log_response_params(ollama_adapter.load_generate_response(request))
    result: ChatCompletion | EventSourceResponse = await get_chat_model(
        model_id
    ).create_completion(
        ollama_adapter.to_chat_completion_params(request, model_id),
        f"chatcmpl-{REQUEST_ID.get()}",
        int(REQUEST_TIME.get().timestamp()),
    )
    if isinstance(result, EventSourceResponse):
        # The event stream is already logged and usage-recorded; this only
        # rewrites it as Ollama's newline-delimited JSON transport.
        return StreamingResponse(
            guard_ndjson_stream_errors(
                ollama_adapter.generate_stream(
                    result.body_iterator,  # type: ignore[arg-type]
                    request.model,
                )
            ),
            media_type=NDJSON_MEDIA_TYPE,
        )
    return log_response_params(
        ollama_adapter.from_chat_completion_as_generate(result, request.model)
    )
