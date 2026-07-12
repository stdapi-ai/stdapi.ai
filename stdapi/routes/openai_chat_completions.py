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

from typing import TYPE_CHECKING, Annotated, Literal

from fastapi import APIRouter, Depends, Path, Query

from stdapi.api_errors import ApiError
from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.capabilities import register_route_capability
from stdapi.models.chat import get_chat_model
from stdapi.monitoring import (
    REQUEST_ID,
    REQUEST_TIME,
    log_request_params,
    log_response_params,
)
from stdapi.responses_store import (
    COMPLETION_ID_PATTERN,
    delete_stored_response,
    discard_stored_response_session,
    load_stored_response,
    save_stored_response,
    try_create_stored_response_session,
)
from stdapi.routes._moderation import apply_request_moderation, build_chat_moderation
from stdapi.types.openai_chat_completions import (
    ChatCompletion,
    ChatCompletionDeleted,
    ChatCompletionStoreMessageList,
    CompletionCreateParams,
)

if TYPE_CHECKING:
    from sse_starlette import EventSourceResponse

register_route_capability(
    "openai_chat_completion",
    f"{SETTINGS.openai_routes_prefix}/v1/chat/completions",
    "TEXT",
    "TEXT",
)

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1/chat", tags=["Chat", TAG_OPENAI]
)

#: Reusable path annotation for the ``completion_id`` path parameter.
_CompletionId = Annotated[
    str,
    Path(
        description="The ID of the stored chat completion.",
        pattern=COMPLETION_ID_PATTERN,
    ),
]


@router.post(
    "/completions",
    summary="Generate a text response for a chat conversation (OpenAI format)",
    operation_id="openai_chat_completion",
    description=(
        "Creates a model response for the given chat conversation (OpenAI Chat Completions API).\n\n"
        "Supports streaming, tool/function calling, and reasoning models. "
        "Returns a `ChatCompletion` object, or a stream of `ChatCompletionChunk` objects when `stream=true`.\n\n"
        "**Extended multimodal inputs (beyond original OpenAI API):**\n"
        "- **Text, images, and vision:** Pass images via URL, data URI, base64, or Files API `file_id` "
        "in the `content` array.\n"
        "- **Audio input:** Include audio content parts (`type: input_audio`) with `wav`/`mp3` data.\n"
        "- **File references:** Reference uploaded files directly via `type: file` with a `file_id` "
        "obtained from `openai_file`.\n"
        '- **Audio output:** Request spoken audio alongside text by setting `modalities: ["text", "audio"]` '
        "and an `audio` config with the desired voice and format.\n\n"
        "**When to use:** Prefer this endpoint for OpenAI SDK compatibility or when using the "
        "`messages` array format. For the newer stateless Responses API, use `openai_response` instead. "
        "For Anthropic SDK compatibility, use `anthropic_message`.\n\n"
        "**Find compatible models:** Call `search_models` with `mcp_tool=openai_chat_completion` "
        "to discover model IDs that support this endpoint. "
        "When using extended multimodal inputs, also filter by the required modality — for example, "
        "add `input_modalities=IMAGE` for vision requests or `input_modalities=SPEECH` for audio input, "
        "so only models that support both the route and the modality are returned."
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
        ApiError: If the model is invalid.
    """
    log_request_params(request, user_id=request.safety_identifier or request.user)
    store = bool(request.store)
    if store and request.stream:
        log_error_details(
            "'store' is not supported with streaming on this backend: ignored.",
            level="warning",
        )
        store = False
    apply_request_moderation(request.moderation)
    model_id = (
        await validate_model(
            request.model, input_modality="TEXT", output_modality="TEXT"
        )
    ).id
    session_id = await try_create_stored_response_session() if store else None
    store = session_id is not None
    completion_id = (
        f"chatcmpl-{session_id}" if store else f"chatcmpl-{REQUEST_ID.get()}"
    )
    try:
        result = await get_chat_model(model_id).create_completion(
            request, completion_id, int(REQUEST_TIME.get().timestamp())
        )
    except BaseException:
        if store:
            await discard_stored_response_session(completion_id)
        raise
    if isinstance(result, ChatCompletion):
        result.moderation = build_chat_moderation(request.moderation)
        if store:
            await save_stored_response(
                completion_id,
                {
                    "messages": request.model_dump(
                        mode="json", by_alias=True, include={"messages"}
                    )["messages"],
                    "response": result.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    ),
                },
            )
    return result


@router.get(
    "/completions/{completion_id}",
    summary="Retrieve a stored chat completion (OpenAI format)",
    operation_id="openai_chat_completion_get",
    description=(
        "Returns a chat completion previously persisted with `store=true` "
        "(OpenAI Chat Completions API).\n\n"
        "Stored chat completions live in AWS Bedrock session storage."
    ),
    response_description="The stored chat completion.",
    responses={
        200: {"description": "The stored chat completion."},
        404: {"description": "Chat completion not found."},
    },
    response_model_exclude_none=True,
)
async def retrieve_chat_completion(
    completion_id: _CompletionId, _: Annotated[None, Depends(authenticate)] = None
) -> ChatCompletion:
    """Retrieve a stored chat completion.

    Args:
        completion_id: Stored chat completion identifier.

    Returns:
        The stored chat completion.

    Raises:
        ApiError: With 404 if the stored chat completion does not exist.
    """
    log_request_params({"completion_id": completion_id})
    stored = await load_stored_response(completion_id)
    return log_response_params(ChatCompletion.model_validate(stored["response"]))


@router.delete(
    "/completions/{completion_id}",
    summary="Delete a stored chat completion (OpenAI format)",
    operation_id="openai_chat_completion_delete",
    description=(
        "Deletes a chat completion previously persisted with `store=true`, "
        "along with its AWS Bedrock session (OpenAI Chat Completions API)."
    ),
    response_description="Deletion confirmation.",
    responses={
        200: {"description": "Chat completion deleted."},
        404: {"description": "Chat completion not found."},
    },
    response_model_exclude_none=True,
)
async def delete_chat_completion(
    completion_id: _CompletionId, _: Annotated[None, Depends(authenticate)] = None
) -> ChatCompletionDeleted:
    """Delete a stored chat completion.

    Args:
        completion_id: Stored chat completion identifier.

    Returns:
        Deletion confirmation.

    Raises:
        ApiError: With 404 if the stored chat completion does not exist.
    """
    log_request_params({"completion_id": completion_id})
    await delete_stored_response(completion_id)
    return log_response_params(ChatCompletionDeleted(id=completion_id))


@router.get(
    "/completions/{completion_id}/messages",
    summary="List the input messages of a stored chat completion (OpenAI format)",
    operation_id="openai_chat_completion_messages",
    description=(
        "Returns the input messages of a chat completion previously persisted "
        "with `store=true` (OpenAI Chat Completions API)."
    ),
    response_description="A paginated list of input messages.",
    responses={
        200: {"description": "The input messages."},
        404: {"description": "Chat completion not found."},
    },
    response_model_exclude_none=True,
)
async def list_chat_completion_messages(
    completion_id: _CompletionId,
    after: Annotated[
        str | None,
        Query(
            description=(
                "Cursor for pagination: the message ID to start after "
                "(the last ID from a previous page)."
            )
        ),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=100, description="A limit on the number of objects returned."),
    ] = 20,
    order: Annotated[
        Literal["asc", "desc"],
        Query(description="Sort order: `asc` is conversation order."),
    ] = "asc",
    _: Annotated[None, Depends(authenticate)] = None,
) -> ChatCompletionStoreMessageList:
    """List the input messages of a stored chat completion.

    Args:
        completion_id: Stored chat completion identifier.
        after: Message ID cursor; only messages strictly after it are returned.
        limit: Maximum number of messages to return.
        order: Sort order relative to the conversation order.

    Returns:
        Paginated list of input messages.

    Raises:
        ApiError: With 404 if the stored chat completion does not exist.
    """
    log_request_params({"completion_id": completion_id, "after": after})
    stored = await load_stored_response(completion_id)
    messages = [
        {"id": f"msg-{index}", **message}
        for index, message in enumerate(stored.get("messages") or [])
    ]
    if order == "desc":
        messages.reverse()
    if after is not None:
        index = next(
            (i for i, message in enumerate(messages) if message["id"] == after), None
        )
        messages = messages[index + 1 :] if index is not None else []
    page, has_more = messages[:limit], len(messages) > limit
    return log_response_params(
        ChatCompletionStoreMessageList(
            data=page,
            has_more=has_more,
            first_id=page[0]["id"] if page else "",
            last_id=page[-1]["id"] if page else "",
        )
    )
