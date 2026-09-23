"""Anthropic-compatible Messages API endpoints using AWS Bedrock."""

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends
from sse_starlette import ServerSentEvent

from stdapi.api_providers.anthropic import TAG_ANTHROPIC
from stdapi.auth import authenticate
from stdapi.aws_bedrock_mantle import API_PATHS, invoke, validate_pruning_extras
from stdapi.config import SETTINGS
from stdapi.models import (
    MANTLE_MODELS,
    is_model_endpoint,
    route_and_execute,
    set_effective_region,
    validate_model,
)
from stdapi.models.capabilities import register_route_capability
from stdapi.models.chat import get_chat_model, serves_via_mantle
from stdapi.models.chat._adapters._anthropic_message import (
    count_tokens_via_bedrock,
    warn_mcp_connector_ignored,
)
from stdapi.models.chat._adapters._count_tokens import (
    count_or_approximate,
    plain_server_tool_history,
    stub_server_tools,
)
from stdapi.models.chat._adapters._responses_context import (
    capped_output_budget,
    context_overflow,
)
from stdapi.models.chat._adapters._stream_open import (
    close_stream,
    context_refusal,
    open_peeked,
)
from stdapi.models.chat._mantle import get_mantle_chat_model
from stdapi.models.chat._mantle._convert import messages_payload
from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel
from stdapi.models.chat._mantle._default import messages_request_headers
from stdapi.monitoring import REQUEST_ID, log_request_params, log_response_params
from stdapi.region_routing import REGION_ROUTER
from stdapi.types.anthropic_messages import (
    CountTokensContextManagementResponse,
    Message,
    MessageCountTokensParams,
    MessageCreateParams,
    MessageParam,
    MessageTokensCount,
)
from stdapi.utils import to_json_str, try_parse_json

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable, Callable

    from sse_starlette import EventSourceResponse
    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.models import ModelDetails


register_route_capability(
    "anthropic_message",
    f"{SETTINGS.anthropic_routes_prefix}/v1/messages",
    "TEXT",
    "TEXT",
)
register_route_capability(
    "anthropic_message_count_tokens",
    f"{SETTINGS.anthropic_routes_prefix}/v1/messages/count_tokens",
    "TEXT",
    "TEXT",
)

router: APIRouter = APIRouter(
    prefix=f"{SETTINGS.anthropic_routes_prefix}/v1", tags=["Chat", TAG_ANTHROPIC]
)

#: Mantle path serving the Anthropic count_tokens API.
_MANTLE_COUNT_TOKENS_PATH = API_PATHS["messages"] + "/count_tokens"


def _stopped_at_window(
    result: Message | EventSourceResponse,
) -> Message | EventSourceResponse:
    """Report a capped answer that used its whole output budget as stopped by the window.

    Args:
        result: The answer, its output capped to what the context window leaves.

    Returns:
        The answer, a ``max_tokens`` stop reported as ``model_context_window_exceeded``.
    """
    if isinstance(result, Message):
        if result.stop_reason == "max_tokens":
            result.stop_reason = "model_context_window_exceeded"
        return result
    result.body_iterator = _window_stop_events(result.body_iterator)
    return result


async def _window_stop_events(events: AsyncIterable[Any]) -> AsyncGenerator[Any]:
    """Relay a capped answer's stream, its ``max_tokens`` stop reported as the window's.

    Args:
        events: The stream events.

    Yields:
        The stream events, ``message_delta`` rewritten when it stops on ``max_tokens``.
    """
    try:
        async for event in events:
            if (
                isinstance(event, ServerSentEvent)
                and event.event == "message_delta"
                and isinstance(event.data, str)
            ):
                payload = try_parse_json(event.data)
                if (
                    isinstance(payload, dict)
                    and isinstance(delta := payload.get("delta"), dict)
                    and delta.get("stop_reason") == "max_tokens"
                ):
                    delta["stop_reason"] = "model_context_window_exceeded"
                    event = ServerSentEvent(to_json_str(payload), event=event.event)
            yield event
    finally:
        await close_stream(events)


async def _count_tokens_via_mantle(
    request: MessageCountTokensParams, model_id: str
) -> MessageTokensCount:
    """Count tokens via the Mantle Anthropic count_tokens API.

    Mantle-only models are not reachable through the Bedrock Runtime
    CountTokens API, so the count is proxied to the Mantle endpoint with
    region routing and failover across the model's regions. Server tools and
    their replayed calls and results are counted as ``_count_tokens`` describes.

    Args:
        request: Count tokens request following Anthropic spec.
        model_id: Mantle model identifier.

    Returns:
        The token count.

    Raises:
        MantleError: When the Mantle upstream rejects the request.
    """
    # Reuse the Messages payload normalization (file inlining, system-role
    # folding, extension stripping); drop its generation-only default.
    history, history_tokens = plain_server_tool_history(request.messages)
    payload = await messages_payload(
        request.model_copy(update={"messages": history}),  # type: ignore[arg-type]
        model_id,
    )
    payload.pop("max_tokens", None)
    # The endpoint takes no server tool, nor its calls and results in history.
    server_tokens = history_tokens + stub_server_tools(payload.get("tools") or [])
    if isinstance(mantle_model := get_mantle_chat_model(model_id), MantleChatModel):
        mantle_model.drop_unapplied_context_management(payload)
    model = MANTLE_MODELS.get(model_id)
    regions = model.regions if model else SETTINGS.aws_bedrock_mantle_regions
    # route_and_execute only retries across regions when the region router is
    # enabled and there is more than one candidate; otherwise it calls the first
    # candidate exactly once, so the in-region retry below must cover it instead.
    single_region = len(regions) == 1 or REGION_ROUTER is None

    async def call(region: RegionName) -> MessageTokensCount:
        """Count the request's tokens via one region's Mantle endpoint."""
        set_effective_region(model_id, region)
        result = await invoke(
            region,
            _MANTLE_COUNT_TOKENS_PATH,
            payload,
            single_region=single_region,
            headers=messages_request_headers(payload),
        )
        return validate_pruning_extras(MessageTokensCount, result)

    counted = await route_and_execute(model_id, regions, call)
    counted.input_tokens += server_tokens
    if counted.context_management is not None:
        counted.context_management.original_input_tokens += server_tokens
    return counted


@router.post(
    "/messages",
    summary="Generate a message response (Anthropic format)",
    operation_id="anthropic_message",
    description=(
        "Creates a message response (Anthropic Messages API).\n\n"
        "Accepts a structured list of input messages and generates the next message in the conversation. "
        "Returns a `Message` object, or a stream of `MessageStreamEvent` objects when `stream=true`.\n\n"
        "**Extended multimodal inputs (beyond original Anthropic API):**\n"
        "- **Images:** Supply images inline (base64/URL) or by Files API `file_id` obtained from `anthropic_file`.\n"
        "- **Documents:** Supply PDFs (base64/URL), plain text, or files by `file_id`. "
        "Citation extraction is supported.\n\n"
        "**Extended capabilities:**\n"
        "- **Extended thinking:** Control reasoning depth via `thinking` or `output_config.effort` "
        "(`low`, `medium`, `high`, `xhigh`, `max`).\n"
        "- **Server tools:** Built-in tools such as `web_search` can be enabled without custom implementations.\n\n"
        "**When to use:** Use this endpoint for Anthropic SDK compatibility or when you need "
        "extended thinking, citations, or Anthropic-specific features. "
        "For OpenAI SDK compatibility, use `openai_chat_completion` or `openai_response` instead.\n\n"
        "**Find compatible models:** Call `search_models` with `route=anthropic_message` "
        "to discover model IDs that support this endpoint. "
        "When supplying images or documents, also add `input_modalities=IMAGE` to the filter "
        "so only models that support both the route and image input are returned."
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
    warn_mcp_connector_ignored(request)
    chat_model = get_chat_model(
        (
            await validate_model(
                request.model,
                input_modality="TEXT",
                output_modality="TEXT",
                route="anthropic_message",
            )
        ).id
    )
    message_id = f"msg_{REQUEST_ID.get()}"
    try:
        # A stream refused on its first event is refused as the unstreamed request is.
        return await open_peeked(
            chat_model.create_message(request, message_id), context_refusal
        )
    except Exception as exc:
        # Upstream serves an input the window holds alone with its output capped.
        if (budget := capped_output_budget(context_overflow(exc))) is None:
            raise
    return _stopped_at_window(
        await open_peeked(
            chat_model.create_message(
                request.model_copy(update={"max_tokens": budget}), message_id
            ),
            context_refusal,
        )
    )


@router.post(
    "/messages/count_tokens",
    summary="Count input tokens for a message without generating a response (Anthropic format)",
    operation_id="anthropic_message_count_tokens",
    description=(
        "Counts the number of tokens a given request would consume, without creating a message.\n\n"
        "Accounts for all inputs — messages, system prompt, tools, images, and documents. "
        "Useful for estimating costs or checking whether a prompt fits within a model's context window "
        "before making a full `anthropic_message` call.\n\n"
        "Every text model is counted: exactly where the model's own counter serves it, otherwise as "
        "an approximation that is never below the exact count on the content measured.\n\n"
        "**Find compatible models:** Call `search_models` with `route=anthropic_message_count_tokens` "
        "to discover model IDs that support this endpoint."
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
    """Count the number of tokens in a message, without creating it.

    Args:
        request: Count tokens request following Anthropic spec.

    Returns:
        MessageTokensCount with the input token count.

    Raises:
        ApiError: If the model is invalid or does not support text output.
    """
    log_request_params(request)
    warn_mcp_connector_ignored(request)
    model = await validate_model(
        request.model,
        input_modality="TEXT",
        output_modality="TEXT",
        route="anthropic_message_count_tokens",
    )
    model_id = model.get_id()

    async def exact(counted: MessageCountTokensParams = request) -> MessageTokensCount:
        """Count with the model's own counter."""
        if serves_via_mantle(model_id):
            return await _count_tokens_via_mantle(counted, model_id)
        return await count_tokens_via_bedrock(
            counted,
            model_id,
            model.regions[0],
            # Not Mantle-served, so this is always a Converse chat model.
            get_chat_model(model_id),  # type: ignore[arg-type]
        )

    async def proxy(counter: ModelDetails) -> MessageTokensCount:
        """Count with another Claude model's counter."""
        counter_id = counter.get_id()
        return await count_tokens_via_bedrock(
            request,
            counter_id,
            counter.regions[0],
            get_chat_model(counter_id),  # type: ignore[arg-type]
        )

    def wrap(tokens: int) -> MessageTokensCount:
        """Answer an estimate, the same with and without context editing."""
        return MessageTokensCount(
            input_tokens=tokens,
            context_management=CountTokensContextManagementResponse(
                original_input_tokens=tokens
            )
            if request.context_management is not None
            else None,
        )

    return log_response_params(
        await count_or_approximate(
            model,
            request,
            None if is_model_endpoint(model) else exact,
            proxy,
            wrap,
            _remapped,
            lambda: exact(
                MessageCountTokensParams(
                    model=request.model,
                    messages=[MessageParam(role="user", content=".")],
                )
            ),
        )
    )


def _remapped(
    counted: MessageTokensCount, scale: Callable[[int], int]
) -> MessageTokensCount:
    """Apply a function to every token count of an answer.

    Args:
        counted: The answer.
        scale: The function.

    Returns:
        The answer, changed in place.
    """
    counted.input_tokens = scale(counted.input_tokens)
    if counted.context_management is not None:
        counted.context_management.original_input_tokens = scale(
            counted.context_management.original_input_tokens
        )
    return counted
