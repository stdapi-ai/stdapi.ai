"""OpenAI-compatible Chat Completions API endpoints using AWS Bedrock Converse.

- POST /v1/chat/completions — create a chat completion
- GET /v1/chat/completions — list stored chat completions
- POST /v1/chat/completions/{completion_id} — update a stored chat completion
- GET /v1/chat/completions/{completion_id} — retrieve a stored chat completion
- DELETE /v1/chat/completions/{completion_id} — delete a stored chat completion
- GET /v1/chat/completions/{completion_id}/messages — list a stored chat completion's messages
"""

from asyncio import gather
from typing import TYPE_CHECKING, Annotated, Any, Literal, Never

from fastapi import APIRouter, Depends, Path, Query, Request
from pydantic import ValidationError
from pydantic_core import from_json

from stdapi.api_errors import ApiError
from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.cleanup import schedule_cleanup
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.capabilities import register_route_capability
from stdapi.models.chat import get_chat_model
from stdapi.monitoring import (
    REQUEST_ID,
    REQUEST_TIME,
    log_error_details,
    log_request_params,
    log_response_params,
)
from stdapi.responses_store import (
    COMPLETION_ID_PATTERN,
    delete_stored_response,
    discard_stored_response_session,
    list_stored_sessions,
    load_stored_response,
    save_stored_response,
    try_create_stored_response_session,
)
from stdapi.routes._moderation import apply_request_moderation, build_chat_moderation
from stdapi.types.openai_chat_completions import (
    ChatCompletion,
    ChatCompletionDeleted,
    ChatCompletionList,
    ChatCompletionStoreMessageList,
    ChatCompletionUpdateParams,
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

#: Number of candidate sessions loaded concurrently per batch when listing.
_LIST_LOAD_BATCH_SIZE: int = 10


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
        "**Find compatible models:** Call `search_models` with `route=openai_chat_completion` "
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
    model_id = (
        await validate_model(
            request.model, input_modality="TEXT", output_modality="TEXT"
        )
    ).id
    # After the model: an alias may carry the guardrail 'moderation' reports on.
    apply_request_moderation(request.moderation)
    placeholder_id = f"chatcmpl-{REQUEST_ID.get()}"
    created = int(REQUEST_TIME.get().timestamp())
    generation = get_chat_model(model_id).create_completion(
        request, placeholder_id, created
    )
    session_id: str | None
    if store:
        # Overlaps session creation with generation: create_completion never
        # sends completion_id upstream, so a placeholder is safe here and the
        # real, session-derived ID is stamped onto the result afterward.
        session_result, generation_result = await gather(
            try_create_stored_response_session("chat_completion"),
            generation,
            return_exceptions=True,
        )
        if isinstance(generation_result, BaseException):
            if not isinstance(session_result, BaseException) and session_result:
                schedule_cleanup(
                    discard_stored_response_session(
                        f"chatcmpl-{session_result}", "chat_completion"
                    )
                )
            raise generation_result
        if isinstance(session_result, BaseException):
            raise session_result
        session_id, result = session_result, generation_result
    else:
        session_id = None
        result = await generation
    store = session_id is not None
    completion_id = f"chatcmpl-{session_id}" if store else placeholder_id
    if isinstance(result, ChatCompletion):
        result.moderation = build_chat_moderation(request.moderation)
        result.metadata = request.metadata
        if store:
            # A locally stored result must carry the server-assigned ID so the
            # stored surface (GET/DELETE/messages) can serve it: some backends
            # (e.g. Mantle passthrough) ignore completion_id and return their
            # own upstream ID.
            result.id = completion_id
            try:
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
            except BaseException:
                schedule_cleanup(
                    discard_stored_response_session(completion_id, "chat_completion")
                )
                raise
    return result


def _malformed_stored_document(completion_id: str, error: Exception) -> Never:
    """Log a malformed stored chat completion document and raise the standard 404.

    Guards against a foreign or corrupt document (schema drift, a document
    written by an incompatible version) crashing route handling instead of
    surfacing as a normal not-found error.

    Args:
        completion_id: Stored chat completion identifier.
        error: The parsing or validation failure.

    Raises:
        ApiError: Always, with status 404.
    """
    log_error_details(
        f"Discarding malformed stored chat completion document for "
        f"'{completion_id}': {error}",
        level="warning",
    )
    msg = f"Chat completion with id '{completion_id}' not found."
    raise ApiError(msg, status=404)


def _store_message_content(message: dict[str, Any]) -> dict[str, Any]:
    """Split a stored message's array content into `content`/`content_parts`.

    The OpenAI store-message shape puts the text in `content` and the
    original text/image parts in `content_parts` when the request used a
    content parts array.

    Args:
        message: Stored request message.

    Returns:
        The message with OpenAI store-message content fields.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return message
    return {
        **message,
        "content": "".join(
            part.get("text", "") for part in content if part.get("type") == "text"
        ),
        "content_parts": [
            part for part in content if part.get("type") in ("text", "image_url")
        ],
    }


def _metadata_filters(request: Request) -> dict[str, str]:
    """Extract the metadata filters from the request query string.

    Accepts the OpenAI SDK's ``metadata[key]=value`` pairs, and a single
    ``metadata={"key": "value"}`` JSON object for clients that cannot send
    bracketed keys (MCP tool calls, which serialize a whole object into one
    query parameter).

    Args:
        request: The incoming HTTP request.

    Returns:
        Metadata key-value pairs to filter stored chat completions by.

    Raises:
        ApiError: With 400 if a bare ``metadata`` parameter is not a JSON
            object of string values.
    """
    filters = {
        key[len("metadata[") : -1]: value
        for key, value in request.query_params.items()
        if key.startswith("metadata[") and key.endswith("]")
    }
    raw = request.query_params.get("metadata")
    if raw is None:
        return filters
    msg = (
        "Invalid 'metadata' filter: expected a JSON object of string values, "
        'such as metadata={"key": "value"}, or one metadata[key]=value '
        "parameter per key."
    )
    try:
        parsed = from_json(raw)
    except ValueError as error:
        raise ApiError(msg, status=400) from error
    if not isinstance(parsed, dict) or not all(
        isinstance(value, str) for value in parsed.values()
    ):
        raise ApiError(msg, status=400)
    return parsed | filters


def _matches_filters(
    completion: ChatCompletion, model: str | None, metadata: dict[str, str]
) -> bool:
    """Check whether a stored chat completion matches the list filters.

    Args:
        completion: Candidate stored chat completion.
        model: Required model ID, or None to skip the check.
        metadata: Required metadata key-value pairs.

    Returns:
        True if the completion matches every given filter.
    """
    if model is not None and completion.model != model:
        return False
    stored_metadata = completion.metadata or {}
    return all(stored_metadata.get(key) == value for key, value in metadata.items())


async def _load_completion_candidate(completion_id: str) -> ChatCompletion | None:
    """Load a stored chat completion candidate for the list endpoint.

    Args:
        completion_id: Stored chat completion identifier.

    Returns:
        The chat completion, or None if its document cannot be read: still
        being generated, deleted between the session scan and the read, or
        corrupt.

    Raises:
        ApiError: When the load fails for any reason other than 404.
    """
    try:
        stored = await load_stored_response(completion_id, "chat_completion")
        return ChatCompletion.model_validate(stored["response"])
    except ApiError as error:
        if error.status == 404:
            # Session tagged at creation, but with no document yet (generation
            # in flight) or no longer any document (deleted).
            return None
        raise
    except (ValueError, KeyError, ValidationError) as error:
        log_error_details(
            f"Skipping corrupt stored chat completion '{completion_id}': {error}",
            level="warning",
        )
        return None


@router.get(
    "/completions",
    summary="List stored chat completions (OpenAI format)",
    operation_id="openai_chat_completion_list",
    description=(
        "Returns the chat completions persisted with `store=true`, sorted by "
        "creation time (OpenAI Chat Completions API).\n\n"
        "Filter by `model` and by metadata, passed either as a JSON object in "
        'the `metadata` query parameter (`metadata={"key": "value"}`) or as '
        "one `metadata[key]=value` query parameter per key. Listings cover the "
        "most recent 1,000 stored chat completions; older ones may not appear."
    ),
    response_description="A paginated list of stored chat completions.",
    responses={
        200: {"description": "The stored chat completions."},
        400: {"description": "Invalid metadata filter."},
    },
    response_model_exclude_none=True,
    openapi_extra={
        "parameters": [
            {
                "name": "metadata",
                "in": "query",
                "required": False,
                # A string, not a deepObject: clients that cannot send
                # bracketed keys serialize an object into this one parameter.
                "schema": {"type": "string"},
                "example": '{"project": "alpha"}',
                "description": (
                    "Filter by metadata key-value pairs, as a JSON object of "
                    'string values (`{"project": "alpha"}`). The equivalent '
                    "`metadata[key]=value` parameters are also accepted."
                ),
            }
        ]
    },
)
async def list_chat_completions(
    request: Request,
    after: Annotated[
        str | None,
        Query(
            description=(
                "Cursor for pagination: the chat completion ID to start after "
                "(the last ID from a previous page)."
            ),
            pattern=COMPLETION_ID_PATTERN,
        ),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=100, description="A limit on the number of objects returned."),
    ] = 20,
    model: Annotated[
        str | None,
        Query(description="Only return chat completions generated by this model."),
    ] = None,
    order: Annotated[
        Literal["asc", "desc"], Query(description="Sort order by creation time.")
    ] = "asc",
    _: Annotated[None, Depends(authenticate)] = None,
) -> ChatCompletionList:
    """List stored chat completions.

    Args:
        request: The incoming HTTP request, carrying metadata filters.
        after: Chat completion ID cursor; only later objects are returned.
        limit: Maximum number of chat completions to return.
        model: Only return chat completions generated by this model.
        order: Sort order by creation time.

    Returns:
        Paginated list of stored chat completions.

    Raises:
        ApiError: With 400 for an invalid ``metadata`` filter, or 404 if
            ``after`` does not match any stored chat completion.
    """
    metadata = _metadata_filters(request)
    log_request_params(
        {
            "after": after,
            "limit": limit,
            "order": order,
            "model": model,
            "metadata": metadata,
        }
    )
    sessions = await list_stored_sessions("chat_completion")
    sessions.sort(key=lambda session: session[1], reverse=order == "desc")
    ids = [f"chatcmpl-{session_id}" for session_id, _ in sessions]
    if after is not None:
        index = next((i for i, id_ in enumerate(ids) if id_ == after), None)
        if index is None:
            msg = f"No chat completion with id '{after}' found."
            raise ApiError(msg, status=404)
        ids = ids[index + 1 :]
    # Scans in order until `limit` readable completions are collected or the
    # IDs are exhausted: dropping unreadable candidates after truncating to
    # ids[:limit] could yield a short or empty page with has_more=True,
    # stalling SDK pagers that stop on an empty page. Unfiltered, only the
    # unreadable candidates cost an extra read.
    filtered = bool(model or metadata)
    completions: list[ChatCompletion] = []
    scanned = 0
    while scanned < len(ids) and len(completions) < limit:
        size = (
            _LIST_LOAD_BATCH_SIZE
            if filtered
            else min(_LIST_LOAD_BATCH_SIZE, limit - len(completions))
        )
        batch = ids[scanned : scanned + size]
        scanned += len(batch)
        completions.extend(
            completion
            for completion in await gather(*map(_load_completion_candidate, batch))
            if completion is not None
            and (not filtered or _matches_filters(completion, model, metadata))
        )
    has_more = scanned < len(ids) or len(completions) > limit
    del completions[limit:]
    return log_response_params(
        ChatCompletionList(
            data=completions,
            has_more=has_more,
            first_id=completions[0].id if completions else None,
            last_id=completions[-1].id if completions else None,
        )
    )


@router.post(
    "/completions/{completion_id}",
    summary="Update a stored chat completion (OpenAI format)",
    operation_id="openai_chat_completion_update",
    description=(
        "Replaces the metadata of a chat completion previously persisted with "
        "`store=true` (OpenAI Chat Completions API). Metadata is the only "
        "updatable field; `null` clears it."
    ),
    response_description="The updated chat completion.",
    responses={
        200: {"description": "The updated chat completion."},
        404: {"description": "Chat completion not found."},
    },
    response_model_exclude_none=True,
)
async def update_chat_completion(
    completion_id: _CompletionId,
    body: ChatCompletionUpdateParams,
    _: Annotated[None, Depends(authenticate)] = None,
) -> ChatCompletion:
    """Update the metadata of a stored chat completion.

    Args:
        completion_id: Stored chat completion identifier.
        body: Update request holding the replacement metadata.

    Returns:
        The updated chat completion.

    Raises:
        ApiError: With 404 if the stored chat completion does not exist.
    """
    log_request_params({"completion_id": completion_id, "metadata": body.metadata})
    stored = await load_stored_response(completion_id, "chat_completion")
    try:
        if body.metadata is None:
            stored["response"].pop("metadata", None)
        else:
            stored["response"]["metadata"] = body.metadata
        result = ChatCompletion.model_validate(stored["response"])
    except (KeyError, TypeError, AttributeError, ValidationError) as error:
        _malformed_stored_document(completion_id, error)
    await save_stored_response(completion_id, stored)
    return log_response_params(result)


@router.get(
    "/completions/{completion_id}",
    summary="Retrieve a stored chat completion (OpenAI format)",
    operation_id="openai_chat_completion_get",
    description=(
        "Returns a chat completion previously persisted with `store=true` "
        "(OpenAI Chat Completions API)."
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
    stored = await load_stored_response(completion_id, "chat_completion")
    try:
        result = ChatCompletion.model_validate(stored["response"])
    except (KeyError, TypeError, ValidationError) as error:
        _malformed_stored_document(completion_id, error)
    return log_response_params(result)


@router.delete(
    "/completions/{completion_id}",
    summary="Delete a stored chat completion (OpenAI format)",
    operation_id="openai_chat_completion_delete",
    description=(
        "Deletes a chat completion previously persisted with `store=true`, "
        "along with its stored messages (OpenAI Chat Completions API)."
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
    await delete_stored_response(completion_id, "chat_completion")
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
            ),
            pattern=r"^msg-[0-9]+$",
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
        ApiError: With 404 if the stored chat completion does not exist, or
            if ``after`` does not match any message.
    """
    log_request_params(
        {"completion_id": completion_id, "after": after, "limit": limit, "order": order}
    )
    stored = await load_stored_response(completion_id, "chat_completion")
    try:
        messages = [
            {"id": f"msg-{index}", **_store_message_content(message)}
            for index, message in enumerate(stored.get("messages") or [])
        ]
    except (AttributeError, TypeError) as error:
        _malformed_stored_document(completion_id, error)
    if order == "desc":
        messages.reverse()
    if after is not None:
        index = next(
            (i for i, message in enumerate(messages) if message["id"] == after), None
        )
        if index is None:
            msg = f"No message with id '{after}' found for this chat completion."
            raise ApiError(msg, status=404)
        messages = messages[index + 1 :]
    page, has_more = messages[:limit], len(messages) > limit
    return log_response_params(
        ChatCompletionStoreMessageList(
            data=page,
            has_more=has_more,
            first_id=page[0]["id"] if page else None,
            last_id=page[-1]["id"] if page else None,
        )
    )
