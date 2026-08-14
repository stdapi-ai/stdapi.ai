"""OpenAI-compatible Conversations API endpoints.

- POST /v1/conversations — create a conversation
- GET /v1/conversations/{conversation_id} — retrieve a conversation
- POST /v1/conversations/{conversation_id} — update a conversation's metadata
- DELETE /v1/conversations/{conversation_id} — delete a conversation
- POST /v1/conversations/{conversation_id}/items — add items to a conversation
- GET /v1/conversations/{conversation_id}/items — list a conversation's items
- GET /v1/conversations/{conversation_id}/items/{item_id} — retrieve one item
- DELETE /v1/conversations/{conversation_id}/items/{item_id} — delete one item
"""

from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, Depends, Path, Query
from pydantic import TypeAdapter

from stdapi.api_errors import ApiError
from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.conversations import (
    append_items,
    create_conversation,
    delete_conversation,
    delete_item,
    get_conversation,
    is_item_reference,
    item_not_found,
    load_items,
    stored_item,
    update_conversation,
    validate_conversation_id,
    validate_item_id,
)
from stdapi.monitoring import log_request_params, log_response_params
from stdapi.types.openai_conversations import (
    Conversation,
    ConversationCreateParams,
    ConversationDeleted,
    ConversationItemList,
    ConversationItemsCreateParams,
    ConversationUpdateParams,
)
from stdapi.types.openai_responses import ResponseIncludable, ResponseItem

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stdapi.types.openai_responses import ResponseInputItem

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1/conversations",
    tags=["Conversations", TAG_OPENAI],
)

#: Reusable path annotation for the ``conversation_id`` path parameter.
_ConversationId = Annotated[
    str, Path(description="The ID of the conversation.", max_length=255)
]

#: Reusable path annotation for the ``item_id`` path parameter.
_ItemId = Annotated[
    str, Path(description="The ID of the item in the conversation.", max_length=255)
]

#: Reusable query annotation for the ``include`` list, shared by the item routes.
_Include = Annotated[
    list[ResponseIncludable] | None,
    Query(
        description="Additional item fields to return. "
        "`reasoning.encrypted_content` returns the reasoning items' encrypted "
        "content; other values are accepted and ignored."
    ),
]

#: ``include`` value that keeps the encrypted content of reasoning items.
_INCLUDE_REASONING_CONTENT = "reasoning.encrypted_content"

#: Item field returned only when the matching ``include`` value is requested.
_ENCRYPTED_CONTENT_FIELD = "encrypted_content"

#: Item type whose encrypted content the ``include`` value above gates.
_REASONING_TYPE = "reasoning"

#: Adapter validating one stored item against the conversation item union.
_ITEM_ADAPTER: TypeAdapter[ResponseItem] = TypeAdapter[ResponseItem](ResponseItem)

#: Default number of items returned by the item listing.
_DEFAULT_LIMIT = 20

#: Maximum number of items the item listing returns in one page.
_MAX_LIMIT = 100


def _visible_items(
    items: list[dict[str, Any]], include: Sequence[str] | None
) -> list[dict[str, Any]]:
    """Drop item fields the request did not ask to include.

    Args:
        items: Stored items, in conversation order.
        include: The request's ``include`` values, if any.

    Returns:
        The items as they are returned to the client.
    """
    if include and _INCLUDE_REASONING_CONTENT in include:
        return items
    return [
        {key: value for key, value in item.items() if key != _ENCRYPTED_CONTENT_FIELD}
        if item.get("type") == _REASONING_TYPE and _ENCRYPTED_CONTENT_FIELD in item
        else item
        for item in items
    ]


def _item_list(
    items: list[dict[str, Any]],
    *,
    after: str | None = None,
    limit: int = _DEFAULT_LIMIT,
    order: Literal["asc", "desc"] = "desc",
    include: Sequence[str] | None = None,
) -> ConversationItemList:
    """Build one page of a conversation's items.

    Args:
        items: Stored items, in conversation order.
        after: Item ID the page starts after, exclusive.
        limit: Maximum number of items in the page.
        order: Page order relative to the conversation order.
        include: The request's ``include`` values, if any.

    Returns:
        The page envelope.

    Raises:
        ApiError: 404 when ``after`` names no item of this conversation.
    """
    ordered = list(reversed(items)) if order == "desc" else items
    if after is not None:
        index = next((i for i, item in enumerate(ordered) if item["id"] == after), None)
        if index is None:
            error = ApiError(f"No item found with id '{after}'.", status=404)
            error.param = "after"
            raise error
        ordered = ordered[index + 1 :]
    page, has_more = ordered[:limit], len(ordered) > limit
    return ConversationItemList.model_validate(
        {
            "object": "list",
            "data": _visible_items(page, include),
            "first_id": page[0]["id"] if page else None,
            "last_id": page[-1]["id"] if page else None,
            "has_more": has_more,
        }
    )


async def _new_items(
    items: Sequence[ResponseInputItem], conversation_id: str | None
) -> list[dict[str, Any]]:
    """Turn request items into stored items, resolving any item reference.

    Args:
        items: The request's ``items``.
        conversation_id: Conversation the references resolve against, or None
            when the conversation does not exist yet and holds no item.

    Returns:
        The items to store, each carrying its minted ID.

    Raises:
        ApiError: 404 when an ``item_reference`` names no item of the
            conversation.
    """
    payloads = [
        item.model_dump(mode="json", by_alias=True, exclude_none=True) for item in items
    ]
    if references := [
        str(payload.get("id") or "")
        for payload in payloads
        if is_item_reference(payload)
    ]:
        known = (
            {item["id"] for item in await load_items(conversation_id)}
            if conversation_id is not None
            else set()
        )
        for reference in references:
            if reference not in known:
                item_not_found(reference)
    return [
        stored_item(payload) for payload in payloads if not is_item_reference(payload)
    ]


@router.post(
    "",
    summary="Create a conversation (OpenAI format)",
    operation_id="openai_conversation",
    description=(
        "Creates a conversation (OpenAI Conversations API).\n\n"
        "A conversation holds the items of a multi-turn exchange. Pass its ID "
        "as `conversation` in `openai_response` and every request and response "
        "item of that turn is appended to it, so the next turn only has to send "
        "the new message."
    ),
    response_description="The created conversation.",
    response_model_exclude_none=True,
)
async def create(
    request: ConversationCreateParams | None = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> Conversation:
    """Create a conversation, optionally with initial items and metadata.

    Args:
        request: Conversation creation request. Optional: a conversation can be
            created with no body at all.

    Returns:
        The created conversation.
    """
    body = request or ConversationCreateParams()
    log_request_params({"items": len(body.items or ()), "metadata": body.metadata})
    metadata = {
        key: value
        for key, value in (body.metadata or {}).items()
        if isinstance(value, str)
    }
    # Prepared first, so a rejected item leaves no empty conversation behind.
    items = await _new_items(body.items or (), None)
    conversation_id, created_at = await create_conversation(metadata)
    if items:
        await append_items(conversation_id, items)
    return log_response_params(
        Conversation(
            id=conversation_id,
            object="conversation",
            created_at=created_at,
            metadata=metadata,
        )
    )


@router.get(
    "/{conversation_id}",
    summary="Retrieve a conversation (OpenAI format)",
    operation_id="openai_conversation_get",
    description="Returns a conversation by ID (OpenAI Conversations API).",
    response_description="The conversation.",
    responses={404: {"description": "Conversation not found."}},
    response_model_exclude_none=True,
)
async def retrieve(
    conversation_id: _ConversationId, _: Annotated[None, Depends(authenticate)] = None
) -> Conversation:
    """Retrieve a conversation.

    Args:
        conversation_id: Conversation identifier.

    Returns:
        The conversation.

    Raises:
        ApiError: 404 when the conversation does not exist.
    """
    log_request_params({"conversation_id": conversation_id})
    validate_conversation_id(conversation_id, "conversation_id")
    created_at, metadata = await get_conversation(conversation_id)
    return log_response_params(
        Conversation(
            id=conversation_id,
            object="conversation",
            created_at=created_at,
            metadata=metadata,
        )
    )


@router.post(
    "/{conversation_id}",
    summary="Update a conversation's metadata (OpenAI format)",
    operation_id="openai_conversation_update",
    description=(
        "Merges key-value pairs into a conversation's metadata (OpenAI "
        "Conversations API). Keys that are not sent keep their value; a key "
        "sent as null is removed."
    ),
    response_description="The updated conversation.",
    responses={404: {"description": "Conversation not found."}},
    response_model_exclude_none=True,
)
async def update(
    conversation_id: _ConversationId,
    request: ConversationUpdateParams,
    _: Annotated[None, Depends(authenticate)] = None,
) -> Conversation:
    """Update a conversation's metadata.

    Args:
        conversation_id: Conversation identifier.
        request: The metadata to merge.

    Returns:
        The updated conversation.

    Raises:
        ApiError: 404 when the conversation does not exist.
    """
    log_request_params(
        {"conversation_id": conversation_id, "metadata": request.metadata}
    )
    validate_conversation_id(conversation_id, "conversation_id")
    created_at, metadata = await update_conversation(
        conversation_id, request.metadata or {}
    )
    return log_response_params(
        Conversation(
            id=conversation_id,
            object="conversation",
            created_at=created_at,
            metadata=metadata,
        )
    )


@router.delete(
    "/{conversation_id}",
    summary="Delete a conversation (OpenAI format)",
    operation_id="openai_conversation_delete",
    description=(
        "Deletes a conversation and every item it holds (OpenAI Conversations API)."
    ),
    response_description="Deletion confirmation.",
    responses={404: {"description": "Conversation not found."}},
    response_model_exclude_none=True,
)
async def delete(
    conversation_id: _ConversationId, _: Annotated[None, Depends(authenticate)] = None
) -> ConversationDeleted:
    """Delete a conversation.

    Args:
        conversation_id: Conversation identifier.

    Returns:
        Deletion confirmation.

    Raises:
        ApiError: 404 when the conversation does not exist.
    """
    log_request_params({"conversation_id": conversation_id})
    validate_conversation_id(conversation_id, "conversation_id")
    await delete_conversation(conversation_id)
    return log_response_params(ConversationDeleted(id=conversation_id))


@router.post(
    "/{conversation_id}/items",
    summary="Add items to a conversation (OpenAI format)",
    operation_id="openai_conversation_items",
    description=(
        "Appends items to a conversation and returns the items that were added "
        "(OpenAI Conversations API). Item IDs are assigned by the server."
    ),
    response_description="The items that were added.",
    responses={404: {"description": "Conversation not found."}},
    response_model_exclude_none=True,
)
async def add_items(
    conversation_id: _ConversationId,
    request: ConversationItemsCreateParams,
    include: _Include = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> ConversationItemList:
    """Append items to a conversation.

    Args:
        conversation_id: Conversation identifier.
        request: The items to append.
        include: Additional item fields to return.

    Returns:
        The items that were added, in the order they were sent.

    Raises:
        ApiError: 404 when the conversation does not exist, or an
            ``item_reference`` names no item of it.
    """
    log_request_params(
        {
            "conversation_id": conversation_id,
            "items": len(request.items or ()),
            "include": include,
        }
    )
    validate_conversation_id(conversation_id, "conversation_id")
    items = await _new_items(request.items or (), conversation_id)
    await append_items(conversation_id, items)
    return log_response_params(
        _item_list(items, limit=len(items), order="asc", include=include)
    )


@router.get(
    "/{conversation_id}/items",
    summary="List a conversation's items (OpenAI format)",
    operation_id="openai_conversation_items_list",
    description=(
        "Returns a page of a conversation's items (OpenAI Conversations API). "
        "Pass the page's `last_id` as `after` to read the next page."
    ),
    response_description="A page of conversation items.",
    responses={404: {"description": "Conversation not found."}},
    response_model_exclude_none=True,
)
async def list_items(
    conversation_id: _ConversationId,
    after: Annotated[
        str | None,
        Query(
            description="Cursor for pagination: the item ID to start after "
            "(the last ID from a previous page).",
            max_length=255,
        ),
    ] = None,
    limit: Annotated[
        int,
        Query(
            ge=0, le=_MAX_LIMIT, description="A limit on the number of items returned."
        ),
    ] = _DEFAULT_LIMIT,
    order: Annotated[
        Literal["asc", "desc"],
        Query(description="Sort order: `asc` is conversation order."),
    ] = "desc",
    include: _Include = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> ConversationItemList:
    """List a conversation's items.

    Args:
        conversation_id: Conversation identifier.
        after: Item ID cursor; only items strictly after it are returned.
        limit: Maximum number of items to return.
        order: Sort order relative to the conversation order.
        include: Additional item fields to return.

    Returns:
        One page of the conversation's items.

    Raises:
        ApiError: 404 when the conversation does not exist, or ``after`` names
            no item of it.
    """
    log_request_params(
        {
            "conversation_id": conversation_id,
            "after": after,
            "limit": limit,
            "order": order,
            "include": include,
        }
    )
    validate_conversation_id(conversation_id, "conversation_id")
    return log_response_params(
        _item_list(
            await load_items(conversation_id),
            after=after,
            limit=limit,
            order=order,
            include=include,
        )
    )


@router.get(
    "/{conversation_id}/items/{item_id}",
    summary="Retrieve one item of a conversation (OpenAI format)",
    operation_id="openai_conversation_item_get",
    description="Returns a single item of a conversation (OpenAI Conversations API).",
    response_description="The conversation item.",
    responses={404: {"description": "Conversation or item not found."}},
    response_model_exclude_none=True,
)
async def retrieve_item(
    conversation_id: _ConversationId,
    item_id: _ItemId,
    include: _Include = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> ResponseItem:
    """Retrieve one item of a conversation.

    Args:
        conversation_id: Conversation identifier.
        item_id: Item identifier.
        include: Additional item fields to return.

    Returns:
        The conversation item.

    Raises:
        ApiError: 404 when the conversation or the item does not exist.
    """
    log_request_params(
        {"conversation_id": conversation_id, "item_id": item_id, "include": include}
    )
    validate_conversation_id(conversation_id, "conversation_id")
    validate_item_id(item_id)
    items = [
        item for item in await load_items(conversation_id) if item["id"] == item_id
    ]
    if not items:
        item_not_found(item_id)
    return log_response_params(
        _ITEM_ADAPTER.validate_python(_visible_items(items, include)[0])
    )


@router.delete(
    "/{conversation_id}/items/{item_id}",
    summary="Delete one item of a conversation (OpenAI format)",
    operation_id="openai_conversation_item_delete",
    description=(
        "Removes one item from a conversation and returns the conversation "
        "(OpenAI Conversations API)."
    ),
    response_description="The conversation the item was removed from.",
    responses={404: {"description": "Conversation or item not found."}},
    response_model_exclude_none=True,
)
async def delete_conversation_item(
    conversation_id: _ConversationId,
    item_id: _ItemId,
    _: Annotated[None, Depends(authenticate)] = None,
) -> Conversation:
    """Delete one item of a conversation.

    Args:
        conversation_id: Conversation identifier.
        item_id: Item identifier.

    Returns:
        The conversation the item was removed from.

    Raises:
        ApiError: 404 when the conversation or the item does not exist.
    """
    log_request_params({"conversation_id": conversation_id, "item_id": item_id})
    validate_conversation_id(conversation_id, "conversation_id")
    validate_item_id(item_id)
    await delete_item(conversation_id, item_id)
    created_at, metadata = await get_conversation(conversation_id)
    return log_response_params(
        Conversation(
            id=conversation_id,
            object="conversation",
            created_at=created_at,
            metadata=metadata,
        )
    )
