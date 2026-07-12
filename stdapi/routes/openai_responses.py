"""OpenAI Responses API endpoint implementation.

This module implements the OpenAI-compatible /v1/responses endpoint, providing
AWS Bedrock Converse integration while maintaining full API compatibility.

The module provides:
    - POST /v1/responses — create a model response
    - POST /v1/responses/input_tokens — count input tokens without generating a response
"""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends

from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.capabilities import register_route_capability
from stdapi.models.chat import get_chat_model
from stdapi.models.chat._adapters._openai_responses import (
    count_input_tokens_via_bedrock,
    encode_compaction_content,
)
from stdapi.monitoring import (
    REQUEST_ID,
    REQUEST_TIME,
    log_request_params,
    log_response_params,
)
from stdapi.types.openai_responses import (
    CompactedResponse,
    CompactParams,
    EasyInputMessage,
    InputTokenCountParams,
    InputTokenCountResponse,
    InputTokensDetails,
    OutputTokensDetails,
    Response,
    ResponseCompactionItem,
    ResponseCreateParams,
    ResponseInputItem,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)

if TYPE_CHECKING:
    from sse_starlette import EventSourceResponse

register_route_capability(
    "openai_response", f"{SETTINGS.openai_routes_prefix}/v1/responses", "TEXT", "TEXT"
)

register_route_capability(
    "openai_response_input_tokens",
    f"{SETTINGS.openai_routes_prefix}/v1/responses/input_tokens",
    "TEXT",
    "TEXT",
)

register_route_capability(
    "openai_response_compact",
    f"{SETTINGS.openai_routes_prefix}/v1/responses/compact",
    "TEXT",
    "TEXT",
)

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1/responses", tags=["Chat", TAG_OPENAI]
)

#: Directive appended to the conversation to produce the compaction summary.
_COMPACTION_PROMPT = (
    "Summarize the conversation above in detail, preserving every fact, "
    "decision, constraint, open task, and tool result needed to continue it. "
    "Reply with the summary only."
)


@router.post(
    "",
    summary="Generate a model response using the Responses API (OpenAI format)",
    operation_id="openai_response",
    description=(
        "Creates a model response (OpenAI Responses API).\n\n"
        "Supports streaming, tool calling, and structured outputs. "
        "Returns a `Response` object, or a stream of `ResponseStreamEvent` objects when `stream=true`.\n\n"
        "**Supported input modalities:**\n"
        "- **Text:** Plain strings or `input_text` content blocks.\n"
        "- **Images:** `input_image` content blocks with a URL, data URI, base64 image, "
        "or Files API `file_id` obtained from `openai_file`.\n"
        "- **Files:** `input_file` content blocks with a URL, base64 data, "
        "or Files API `file_id` obtained from `openai_file`.\n"
        "- **Audio input** is not supported — use `openai_chat_completion` for audio input.\n\n"
        "**When to use:** This is the newer OpenAI API style. For the classic `messages`-array format, "
        "use `openai_chat_completion` instead. For Anthropic SDK compatibility, use `anthropic_message`.\n\n"
        "**Find compatible models:** Call `search_models` with `mcp_tool=openai_response` "
        "to discover model IDs that support this endpoint. "
        "For image inputs, also add `input_modalities=IMAGE` to the filter."
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


@router.post(
    "/input_tokens",
    summary="Count input tokens for a Responses request without generating a response (OpenAI format)",
    operation_id="openai_response_input_tokens",
    description=(
        "Counts the number of tokens a given request would consume, "
        "without creating a response.\n\n"
        "Accepts the same input as `openai_response` (messages, instructions, "
        "tools, images, files) and returns only the token count. Useful for "
        "estimating costs or checking context-window fit before making a full "
        "`openai_response` call.\n\n"
        "**Find compatible models:** Call `search_models` with "
        "`mcp_tool=openai_response_input_tokens` to discover model IDs that "
        "support this endpoint."
    ),
    response_description="Token count for the provided input.",
    status_code=200,
    response_model=InputTokenCountResponse,
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {"object": "response.input_tokens", "input_tokens": 142}
                }
            },
        },
        400: {"description": "Invalid request or unsupported parameters."},
    },
    response_model_exclude_none=True,
)
async def count_input_tokens(
    request: InputTokenCountParams, _: Annotated[None, Depends(authenticate)] = None
) -> InputTokenCountResponse:
    """Count the number of input tokens for a Responses request.

    Uses the AWS Bedrock CountTokens API to return an accurate,
    model-specific token count without generating a response.

    Args:
        request: Input-token count request following the OpenAI Responses spec.

    Returns:
        ResponseInputTokensCount with the input token count.

    Raises:
        ApiError: If the model is invalid or the request is unsupported.
    """
    log_request_params(request)
    model = await validate_model(
        request.model, input_modality="TEXT", output_modality="TEXT", error_status=400
    )
    return log_response_params(
        InputTokenCountResponse(
            input_tokens=await count_input_tokens_via_bedrock(
                request, model.get_id(), model.regions[0]
            )
        )
    )


@router.post(
    "/compact",
    summary="Compact a conversation into a reusable summary item (OpenAI format)",
    operation_id="openai_response_compact",
    description=(
        "Compacts a conversation into a single `compaction` output item "
        "(OpenAI Responses API).\n\n"
        "The model summarises the provided `input`; the summary is returned as "
        "an opaque `compaction` item. Include that item in the `input` of later "
        "`openai_response` calls to continue the conversation with a reduced "
        "context window.\n\n"
        "Stateless: the compaction content is self-contained, so no "
        "conversation state is stored on the server.\n\n"
        "**Find compatible models:** Call `search_models` with "
        "`mcp_tool=openai_response_compact` to discover model IDs that "
        "support this endpoint."
    ),
    response_description="The compacted response.",
    status_code=200,
    response_model=CompactedResponse,
    responses={
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
    },
    response_model_exclude_none=True,
)
async def compact_response(
    request: CompactParams, _: Annotated[None, Depends(authenticate)] = None
) -> CompactedResponse:
    """Compact a conversation into a single compaction item.

    Runs a summarisation pass on AWS Bedrock and wraps the resulting summary
    in an opaque ``compaction`` item that later requests can send back as
    input.

    Args:
        request: Compaction request following the OpenAI Responses spec.

    Returns:
        CompactedResponse holding the compaction item and token usage.

    Raises:
        ApiError: If the model is invalid or the request is unsupported.
    """
    log_request_params(request)
    model_id = (
        await validate_model(
            request.model, input_modality="TEXT", output_modality="TEXT"
        )
    ).id
    response_id = f"resp-{REQUEST_ID.get()}"
    created_at = REQUEST_TIME.get().timestamp()
    items: list[ResponseInputItem] = (
        [EasyInputMessage(role="user", content=request.input)]
        if isinstance(request.input, str)
        else list(request.input or ())
    )
    items.append(EasyInputMessage(role="user", content=_COMPACTION_PROMPT))
    generation = ResponseCreateParams(
        model=request.model,
        input=items,
        instructions=request.instructions,
        prompt_cache_key=request.prompt_cache_key,
        prompt_cache_retention=request.prompt_cache_retention,
        service_tier=request.service_tier,
    )
    response = await get_chat_model(model_id).create_response(
        generation, response_id, created_at
    )
    if not isinstance(response, Response):  # pragma: no cover - stream is never set
        msg = "Unexpected streaming response."
        raise TypeError(msg)
    summary = "".join(
        part.text
        for item in response.output
        if isinstance(item, ResponseOutputMessage)
        for part in item.content
        if isinstance(part, ResponseOutputText)
    )
    return log_response_params(
        CompactedResponse(
            id=response_id,
            created_at=int(created_at),
            output=[
                ResponseCompactionItem(
                    id=f"ci-{REQUEST_ID.get()}",
                    encrypted_content=encode_compaction_content(summary),
                    type="compaction",
                )
            ],
            usage=response.usage
            or ResponseUsage(
                input_tokens=0,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=0,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=0,
            ),
        )
    )
