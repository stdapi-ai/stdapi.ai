"""Ollama-compatible ``/api/chat`` endpoint using AWS Bedrock chat models."""

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
from stdapi.types.ollama import ChatRequest, ChatResponse

if TYPE_CHECKING:
    from stdapi.types.openai_chat_completions import ChatCompletion

register_route_capability(
    "ollama_chat",
    f"{SETTINGS.ollama_routes_prefix}/api/chat",
    "TEXT",
    "TEXT",
    mcp_tool=False,
)

router = APIRouter(
    prefix=f"{SETTINGS.ollama_routes_prefix}/api", tags=["Chat", TAG_OLLAMA]
)


@router.post(
    "/chat",
    summary="Generate a chat response (Ollama format)",
    operation_id="ollama_chat",
    description=(
        "Creates a model response for the given conversation (Ollama Chat API).\n\n"
        "Supports tool calling, images, structured output through `format`, and "
        "thinking through `think`. Streams newline-delimited JSON objects by "
        "default; set `stream` to false for a single response object.\n\n"
        "Use the model names `ollama_tags` publishes. `logprobs` and "
        "`top_logprobs` are not available, and runner options such as `num_ctx` "
        "or `num_gpu`, along with `keep_alive`, are accepted and ignored."
    ),
    response_description="The assistant message, or a stream of partial messages.",
    response_model=ChatResponse,
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
                            "summary": "Single-turn chat",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "messages": [
                                    {"role": "user", "content": "Hello, how are you?"}
                                ],
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
async def chat(
    request: ChatRequest, _: Annotated[None, Depends(authenticate)] = None
) -> ChatResponse | StreamingResponse:
    """Create a chat response using AWS Bedrock Converse APIs.

    Args:
        request: Chat parameters following the Ollama API.

    Returns:
        The Ollama chat response, or a newline-delimited JSON stream of partial
        responses when ``stream`` is true. A request carrying no message
        answers the single done object Ollama answers a load or an unload with.

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
            route="ollama_chat",
        )
    ).id
    if not request.messages:
        # An empty conversation is Ollama's load, and with `keep_alive` at zero
        # its unload: the model is named and answered for, never talked to.
        return log_response_params(ollama_adapter.load_chat_response(request))
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
                ollama_adapter.chat_stream(
                    result.body_iterator,  # type: ignore[arg-type]
                    request.model,
                )
            ),
            media_type=NDJSON_MEDIA_TYPE,
        )
    return log_response_params(
        ollama_adapter.from_chat_completion(result, request.model)
    )
