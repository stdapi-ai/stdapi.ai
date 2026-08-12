"""Local OpenAI-compatible Conversations API types."""

from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from stdapi.api_errors import ApiError
from stdapi.types import BaseModelRequest, BaseModelResponse
from stdapi.types.openai import Metadata, PaginatedListEnvelope
from stdapi.types.openai_responses import ResponseInputItem, ResponseItem

#: Maximum number of key-value pairs a conversation's metadata can hold.
METADATA_MAX_KEYS: int = 16

#: Maximum length of a conversation metadata key.
METADATA_MAX_KEY_LENGTH: int = 64

#: Maximum length of a conversation metadata value.
METADATA_MAX_VALUE_LENGTH: int = 512

#: Maximum number of items accepted in a single conversation items request.
ITEMS_MAX_PER_REQUEST: int = 20


def _invalid_request(message: str, code: str, param: str) -> ApiError:
    """Build the 400 error carrying an upstream error code and parameter name.

    Args:
        message: Human-readable error message.
        code: Upstream error code (e.g. ``string_above_max_length``).
        param: Name of the offending request parameter.

    Returns:
        The error to raise.
    """
    error = ApiError(message, status=400)
    error.code = code
    error.param = param
    return error


def validate_metadata(metadata: dict[str, JsonValue]) -> None:
    """Check a metadata mapping against the conversation metadata limits.

    Args:
        metadata: Metadata mapping from the request body. ``None`` values are
            accepted here; the update route reads them as key deletions.

    Raises:
        ApiError: 400 when the mapping has too many keys, a key or value is
            too long, or a value is neither a string nor ``None``.
    """
    if len(metadata) > METADATA_MAX_KEYS:
        msg = f"'metadata' supports at most {METADATA_MAX_KEYS} key-value pairs."
        raise _invalid_request(msg, "object_above_max_properties", "metadata")
    for key, value in metadata.items():
        if len(key) > METADATA_MAX_KEY_LENGTH:
            msg = (
                f"'metadata' keys must be at most {METADATA_MAX_KEY_LENGTH} "
                f"characters long, got {len(key)}."
            )
            raise _invalid_request(msg, "property_name_above_max_length", "metadata")
        if value is None:
            continue
        if not isinstance(value, str):
            msg = f"'metadata[{key}]' must be a string."
            raise _invalid_request(msg, "invalid_type", "metadata")
        if len(value) > METADATA_MAX_VALUE_LENGTH:
            msg = (
                f"'metadata[{key}]' must be at most "
                f"{METADATA_MAX_VALUE_LENGTH} characters long, got {len(value)}."
            )
            raise _invalid_request(msg, "string_above_max_length", "metadata")


def _validate_items(items: list[ResponseInputItem] | None, *, required: bool) -> None:
    """Check an ``items`` list against the per-request item limits.

    Args:
        items: The request's ``items`` value.
        required: Whether the list must be present and non-empty.

    Raises:
        ApiError: 400 when the list is missing, empty, or too long.
    """
    if items is None:
        if required:
            msg = "Missing required parameter: 'items'."
            raise _invalid_request(msg, "missing_required_parameter", "items")
        return
    if required and not items:
        msg = "'items' must contain at least one item."
        raise _invalid_request(msg, "empty_array", "items")
    if len(items) > ITEMS_MAX_PER_REQUEST:
        msg = f"'items' accepts at most {ITEMS_MAX_PER_REQUEST} items per request."
        raise _invalid_request(msg, "array_above_max_length", "items")


# Ref: openai.types.conversations.conversation.Conversation
class Conversation(BaseModelResponse):
    """A conversation holding items shared across model responses."""

    id: str = Field(description="The unique ID of the conversation.")
    object: Literal["conversation"] = Field(
        description="The object type. Always `conversation`."
    )
    created_at: int = Field(
        description="Unix timestamp (seconds) of when the conversation was created."
    )
    metadata: Metadata = Field(
        description="Key-value pairs attached to the conversation."
    )


# Ref: openai.types.conversations.conversation_deleted_resource.ConversationDeletedResource
class ConversationDeleted(BaseModelResponse):
    """Confirmation that a conversation was deleted."""

    id: str = Field(description="The unique ID of the deleted conversation.")
    object: Literal["conversation.deleted"] = Field(
        default="conversation.deleted",
        description="The object type. Always `conversation.deleted`.",
    )
    deleted: bool = Field(default=True, description="Always `true`.")


# Ref: openai.types.conversations.conversation_item_list.ConversationItemList
class ConversationItemList(PaginatedListEnvelope):
    """A paginated list of conversation items."""

    object: Literal["list"] = Field(
        default="list", description="The object type. Always `list`."
    )
    data: list[ResponseItem] = Field(description="The items in this page.")


# Ref: openai.types.conversations.conversation_create_params.ConversationCreateParams
class ConversationCreateParams(BaseModelRequest):
    """Request body for POST /v1/conversations."""

    items: list[ResponseInputItem] | None = Field(
        default=None,
        description=(
            f"Initial items of the conversation, at most {ITEMS_MAX_PER_REQUEST}."
        ),
    )
    metadata: dict[str, JsonValue] | None = Field(
        default=None,
        description=(
            f"Up to {METADATA_MAX_KEYS} key-value pairs attached to the "
            f"conversation, with keys of at most {METADATA_MAX_KEY_LENGTH} "
            f"characters and string values of at most "
            f"{METADATA_MAX_VALUE_LENGTH} characters."
        ),
    )

    @model_validator(mode="after")
    def _limits(self) -> Self:
        """Enforce the item-count and metadata limits.

        Raises:
            ApiError: 400 when a limit is exceeded.
        """
        _validate_items(self.items, required=False)
        if self.metadata is not None:
            validate_metadata(self.metadata)
        return self


# Ref: openai.types.conversations.conversation_update_params.ConversationUpdateParams
class ConversationUpdateParams(BaseModelRequest):
    """Request body for POST /v1/conversations/{conversation_id}."""

    metadata: dict[str, JsonValue] | None = Field(
        default=None,
        description=(
            "Key-value pairs merged into the conversation's metadata; a key "
            "set to null is removed. Required."
        ),
    )

    @model_validator(mode="after")
    def _limits(self) -> Self:
        """Require ``metadata`` and enforce its limits.

        ``metadata`` is declared optional so that omitting it and sending it as
        null produce the two distinct errors the API defines for them.

        Raises:
            ApiError: 400 when ``metadata`` is missing or null, or a limit is
                exceeded.
        """
        if "metadata" not in self.model_fields_set:
            msg = "Missing required parameter: 'metadata'."
            raise _invalid_request(msg, "missing_required_parameter", "metadata")
        if self.metadata is None:
            msg = "'metadata' must be an object."
            raise _invalid_request(msg, "invalid_type", "metadata")
        validate_metadata(self.metadata)
        return self


# Ref: openai.types.conversations.item_create_params.ItemCreateParams
class ConversationItemsCreateParams(BaseModelRequest):
    """Request body for POST /v1/conversations/{conversation_id}/items."""

    items: list[ResponseInputItem] | None = Field(
        default=None,
        description=(
            "Items to append to the conversation, at least one and at most "
            f"{ITEMS_MAX_PER_REQUEST}."
        ),
    )

    @model_validator(mode="after")
    def _limits(self) -> Self:
        """Enforce the item-count limits.

        Raises:
            ApiError: 400 when ``items`` is missing, empty, or too long.
        """
        _validate_items(self.items, required=True)
        return self
