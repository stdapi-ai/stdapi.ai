"""OpenAI Responses API endpoint implementation.

This module implements the OpenAI-compatible /v1/responses endpoint, providing
AWS Bedrock Converse integration while maintaining full API compatibility.

The module provides:
    - POST /v1/responses — create a model response
    - POST /v1/responses/input_tokens — count input tokens without generating a response
"""

from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, Depends, Path, Query

from stdapi.api_errors import ApiError
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
    log_error_details,
    log_request_params,
    log_response_params,
)
from stdapi.responses_store import (
    RESPONSE_ID_PATTERN,
    delete_stored_response,
    discard_stored_response_session,
    load_stored_response,
    save_stored_response,
    try_create_stored_response_session,
)
from stdapi.routes._moderation import (
    apply_request_moderation,
    build_response_moderation,
)
from stdapi.types.openai_responses import (
    CompactedResponse,
    CompactionUserMessage,
    CompactParams,
    EasyInputMessage,
    InputTokenCountParams,
    InputTokenCountResponse,
    InputTokensDetails,
    OutputTokensDetails,
    Response,
    ResponseCompactionItem,
    ResponseCreateParams,
    ResponseDeleted,
    ResponseInputItem,
    ResponseItemList,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from stdapi.utils import validation_error_handler

if TYPE_CHECKING:
    from collections.abc import Sequence

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

#: Reusable path annotation for the ``response_id`` path parameter.
_ResponseId = Annotated[
    str, Path(description="The ID of the stored response.", pattern=RESPONSE_ID_PATTERN)
]


async def _merge_previous_response(
    request: ResponseCreateParams, previous_response_id: str
) -> ResponseCreateParams:
    """Prepend a stored response's conversation to the request input.

    Rebuilds the request with the stored input items, the stored output
    items, and the new input, in that order. Instructions are not carried
    over, per the OpenAI API.

    Args:
        request: The incoming request.
        previous_response_id: ID of the stored response to continue.

    Returns:
        The rebuilt request without ``previous_response_id``.

    Raises:
        ApiError: 404 when the stored response does not exist.
    """
    stored = await load_stored_response(previous_response_id)
    data = request.model_dump(mode="json", exclude_unset=True, by_alias=True)
    data.pop("previous_response_id", None)
    new_input = data.get("input") or []
    if isinstance(new_input, str):
        new_input = [{"role": "user", "content": new_input}]
    stored_input = stored.get("input") or []
    if isinstance(stored_input, str):
        stored_input = [{"role": "user", "content": stored_input}]
    data["input"] = [
        *stored_input,
        *stored.get("response", {}).get("output", []),
        *new_input,
    ]
    with validation_error_handler():
        return ResponseCreateParams.model_validate(data)


def _normalized_input_items(stored_input: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """Normalize a stored request input into listable input items.

    Plain strings and string-content messages become ``message`` items with
    content parts; every item gets an ID for cursor pagination.

    Args:
        stored_input: The ``input`` value of a stored response document.

    Returns:
        Input items as JSON objects, in conversation order.
    """
    raw = stored_input or []
    if isinstance(raw, str):
        raw = [{"role": "user", "content": raw}]
    items: list[dict[str, Any]] = []
    for index, entry in enumerate(raw):
        item = dict(entry)
        if isinstance(content := item.get("content"), str):
            if item.get("role") == "assistant":
                item["content"] = [
                    {"type": "output_text", "text": content, "annotations": []}
                ]
            else:
                item["content"] = [{"type": "input_text", "text": content}]
            item.setdefault("type", "message")
            item.setdefault("status", "completed")
        item.setdefault("id", f"msg-{index}")
        items.append(item)
    return items


def _compaction_user_messages(items: Sequence[Any]) -> list[CompactionUserMessage]:
    """Echo the conversation's user messages for the compacted output.

    Args:
        items: Validated input items of the compaction generation request.

    Returns:
        The user messages as output items, in conversation order.
    """
    messages: list[CompactionUserMessage] = []
    for item in items:
        if getattr(item, "role", None) != "user":
            continue
        content = item.content
        parts: list[Any] = (
            [{"type": "input_text", "text": content}]
            if isinstance(content, str)
            else [
                part
                if isinstance(part, dict)
                else part.model_dump(mode="json", by_alias=True, exclude_none=True)
                for part in content or ()
            ]
        )
        messages.append(
            CompactionUserMessage(
                id=f"msg-{REQUEST_ID.get()}-{len(messages)}", content=parts
            )
        )
    return messages


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
        ApiError: If the model is invalid or ``previous_response_id`` does
            not exist (404).
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
    previous_response_id = request.previous_response_id
    if previous_response_id:
        request = await _merge_previous_response(request, previous_response_id)
    model_id = (
        await validate_model(
            request.model, input_modality="TEXT", output_modality="TEXT"
        )
    ).id
    session_id = await try_create_stored_response_session("response") if store else None
    store = session_id is not None
    response_id = f"resp-{session_id}" if store else f"resp-{REQUEST_ID.get()}"
    created_at = REQUEST_TIME.get().timestamp()
    try:
        result = await get_chat_model(model_id).create_response(
            request, response_id, created_at
        )
    except BaseException:
        if store:
            await discard_stored_response_session(response_id)
        raise
    if isinstance(result, Response):
        result.moderation = build_response_moderation(request.moderation)
        if previous_response_id:
            result.previous_response_id = previous_response_id
        if store:
            await save_stored_response(
                response_id,
                {
                    "input": request.model_dump(
                        mode="json", by_alias=True, include={"input"}
                    )["input"],
                    "instructions": request.instructions,
                    "response": result.model_dump(
                        mode="json", by_alias=True, exclude_none=True
                    ),
                },
            )
    return result


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
        "The compaction content is self-contained, so no conversation state "
        "is needed on the server; `previous_response_id` may reference a "
        "stored response to compact its conversation too.\n\n"
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
        ApiError: If the model is invalid or ``previous_response_id`` does
            not exist (404).
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
    if request.previous_response_id:
        generation = await _merge_previous_response(
            generation, request.previous_response_id
        )
    # The compaction prompt is always the last input item; don't echo it.
    user_messages = _compaction_user_messages(list(generation.input or ())[:-1])
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
                *user_messages,
                ResponseCompactionItem(
                    id=f"ci-{REQUEST_ID.get()}",
                    encrypted_content=encode_compaction_content(summary),
                    type="compaction",
                ),
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


@router.get(
    "/{response_id}",
    summary="Retrieve a stored model response (OpenAI format)",
    operation_id="openai_response_get",
    description=(
        "Returns a model response previously persisted with `store=true` "
        "(OpenAI Responses API).\n\n"
        "Stored responses live in AWS Bedrock session storage; pass their ID "
        "as `previous_response_id` in `openai_response` to continue the "
        "conversation."
    ),
    response_description="The stored response.",
    responses={
        200: {"description": "The stored response."},
        404: {"description": "Response not found."},
    },
    response_model_exclude_none=True,
)
async def retrieve_response(
    response_id: _ResponseId, _: Annotated[None, Depends(authenticate)] = None
) -> Response:
    """Retrieve a stored model response.

    Args:
        response_id: Stored response identifier.

    Returns:
        The stored response.

    Raises:
        ApiError: With 404 if the stored response does not exist.
    """
    log_request_params({"response_id": response_id})
    stored = await load_stored_response(response_id)
    return log_response_params(Response.model_validate(stored["response"]))


@router.post(
    "/{response_id}/cancel",
    summary="Cancel a background model response (OpenAI format)",
    operation_id="openai_response_cancel",
    description=(
        "Cancels a model response (OpenAI Responses API). Only responses "
        "created with `background=true` can be cancelled, and background "
        "responses are not supported on this backend, so this always fails "
        "with the OpenAI error for synchronous responses."
    ),
    response_description="Never returned; the request always fails.",
    responses={
        400: {"description": "The response is synchronous and cannot be cancelled."},
        404: {"description": "Response not found."},
    },
    response_model_exclude_none=True,
)
async def cancel_response(
    response_id: _ResponseId, _: Annotated[None, Depends(authenticate)] = None
) -> Response:
    """Cancel a background model response.

    Args:
        response_id: Stored response identifier.

    Returns:
        Never; cancellation always fails on this backend.

    Raises:
        ApiError: With 404 if the stored response does not exist, else with
            400 since all responses are synchronous on this backend.
    """
    log_request_params({"response_id": response_id})
    await load_stored_response(response_id)
    msg = "Cannot cancel a synchronous response."
    raise ApiError(msg)


@router.delete(
    "/{response_id}",
    summary="Delete a stored model response (OpenAI format)",
    operation_id="openai_response_delete",
    description=(
        "Deletes a model response previously persisted with `store=true`, "
        "along with its AWS Bedrock session (OpenAI Responses API)."
    ),
    response_description="Deletion confirmation.",
    responses={
        200: {"description": "Response deleted."},
        404: {"description": "Response not found."},
    },
    response_model_exclude_none=True,
)
async def delete_response(
    response_id: _ResponseId, _: Annotated[None, Depends(authenticate)] = None
) -> ResponseDeleted:
    """Delete a stored model response.

    Args:
        response_id: Stored response identifier.

    Returns:
        Deletion confirmation.

    Raises:
        ApiError: With 404 if the stored response does not exist.
    """
    log_request_params({"response_id": response_id})
    await delete_stored_response(response_id)
    return log_response_params(ResponseDeleted(id=response_id))


@router.get(
    "/{response_id}/input_items",
    summary="List the input items of a stored model response (OpenAI format)",
    operation_id="openai_response_input_items",
    description=(
        "Returns the input items that produced a stored model response "
        "(OpenAI Responses API)."
    ),
    response_description="A paginated list of input items.",
    responses={
        200: {"description": "The input items."},
        404: {"description": "Response not found."},
    },
    response_model_exclude_none=True,
)
async def list_response_input_items(
    response_id: _ResponseId,
    after: Annotated[
        str | None,
        Query(
            description=(
                "Cursor for pagination: the item ID to start after "
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
    ] = "desc",
    _: Annotated[None, Depends(authenticate)] = None,
) -> ResponseItemList:
    """List the input items of a stored model response.

    Args:
        response_id: Stored response identifier.
        after: Item ID cursor; only items strictly after it are returned.
        limit: Maximum number of items to return.
        order: Sort order relative to the conversation order.

    Returns:
        Paginated list of input items.

    Raises:
        ApiError: With 404 if the stored response does not exist.
    """
    log_request_params({"response_id": response_id, "after": after, "limit": limit})
    stored = await load_stored_response(response_id)
    items = _normalized_input_items(stored.get("input"))
    if order == "desc":
        items.reverse()
    if after is not None:
        index = next(
            (i for i, item in enumerate(items) if item.get("id") == after), None
        )
        items = items[index + 1 :] if index is not None else []
    page, has_more = items[:limit], len(items) > limit
    return log_response_params(
        ResponseItemList.model_validate(
            {
                "object": "list",
                "data": page,
                "first_id": page[0]["id"] if page else "",
                "last_id": page[-1]["id"] if page else "",
                "has_more": has_more,
            }
        )
    )
