"""Conversations backed by AWS Bedrock session management.

A conversation is one AWS Bedrock session (bedrock-agent-runtime): its
key-value metadata is the session metadata, and its items are appended as
invocations each holding one JSON document. Deleting an item appends a
document naming it, and reading replays every document in write order, so
the store stays append-only and two concurrent writers never overwrite each
other. AWS keeps all state, so any server instance can serve a conversation
without shared server state.

The conversation ID is ``conv-<session ID>``; the object kind is recorded as
a session tag so a conversation cannot be reached through the stored-response
routes. Conversations live in the primary Bedrock region.
"""

from contextlib import suppress
from re import compile as regex_compile
from typing import TYPE_CHECKING, Any, Never
from uuid import uuid4

from botocore.exceptions import ClientError

from stdapi.api_errors import ApiError, feature_unavailable_guard
from stdapi.aws import get_client
from stdapi.aws_bedrock import handle_bedrock_client_error
from stdapi.aws_bedrock_sessions import (
    end_and_delete_session,
    is_unknown_identifier,
    load_documents,
    not_found_as_404,
    put_document,
)
from stdapi.config import SETTINGS
from stdapi.monitoring import build_metadata
from stdapi.responses_store import KIND_TAG

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from contextlib import AbstractContextManager

    from types_aiobotocore_bedrock_agent_runtime.client import (
        AgentsforBedrockRuntimeClient,
    )

#: Regex pattern that a conversation ID minted by this server matches.
CONVERSATION_ID_PATTERN: str = r"^conv-[A-Za-z0-9-]+$"

#: Stored object kind recorded as the session tag of a conversation.
_KIND: str = "conversation"

#: Document key holding a conversation write, keeping foreign sessions out.
_DOCUMENT_KEY: str = "conversation"

#: Document field holding the items appended by one write.
_ITEMS_FIELD: str = "items"

#: Document field holding the item IDs removed by one write.
_DELETED_FIELD: str = "deleted"

#: Maximum UTF-8 bytes per invocation step text block (stays under the payload quota).
_CHUNK_SIZE: int = 200_000

#: Maximum concurrent invocation-step calls when reading or writing a conversation.
_STEP_CONCURRENCY: int = 8

#: Maximum invocation steps read before a conversation listing stops.
_SCAN_LIMIT: int = 1_000

#: Public item ID prefix per item type; other types fall back to ``item``.
_ITEM_ID_PREFIXES: dict[str, str] = {
    "message": "msg",
    "reasoning": "rs",
    "function_call": "fc",
    "function_call_output": "fco",
    "computer_call": "cu",
    "computer_call_output": "cuo",
    "custom_tool_call": "ctc",
    "custom_tool_call_output": "ctco",
    "code_interpreter_call": "cic",
    "file_search_call": "fs",
    "image_generation_call": "ig",
    "local_shell_call": "lsh",
    "local_shell_call_output": "lsho",
    "mcp_call": "mcp",
    "web_search_call": "ws",
    "compaction": "cmp",
}

#: Fallback item ID prefix for an item type without a dedicated one.
_ITEM_ID_FALLBACK_PREFIX: str = "item"

#: Conversation IDs accepted on input: this server's form and the upstream one.
_CONVERSATION_ID_INPUT = regex_compile(r"conv[-_][A-Za-z0-9_-]+")

#: Item IDs accepted on input: a type prefix, an underscore, then the payload.
_ITEM_ID_INPUT = regex_compile(r"[A-Za-z0-9]+_[A-Za-z0-9_-]+")

#: Maximum length of an identifier accepted on input.
_ID_MAX_LENGTH: int = 64

#: Item types that reference an existing item instead of adding a new one.
_REFERENCE_TYPE: str = "item_reference"

#: The feature name a caller reads when the deployment cannot serve conversations.
_FEATURE: str = "The Conversations API"

#: What an unreachable session endpoint means, for the operator.
_UNREACHABLE_DETAIL: str = (
    "The Amazon Bedrock session endpoint is unreachable or timed out: session "
    "storage is offered in fewer regions than model inference; configure a "
    "first 'aws_bedrock_regions' entry that provides it."
)


def _session_calls(*actions: str) -> AbstractContextManager[None]:
    """Answer a denied or unreachable session call as an unavailable feature.

    Args:
        *actions: The ``bedrock`` session actions the guarded calls need.

    Returns:
        The guard wrapping the calls.
    """
    permissions = ", ".join(f"bedrock:{action}" for action in actions)
    return feature_unavailable_guard(
        _FEATURE,
        missing=f"{permissions} on 'arn:aws:bedrock:*:*:session/*'",
        unreachable=_UNREACHABLE_DETAIL,
    )


def _invalid_id(value: str, param: str, expected: str) -> Never:
    """Raise the 400 an unusable identifier gets.

    Args:
        value: The rejected identifier.
        param: Name of the request parameter carrying it.
        expected: What a usable identifier looks like.

    Raises:
        ApiError: Always, with status 400.
    """
    error = ApiError(f"Invalid '{param}': '{value}'. Expected {expected}.", status=400)
    error.code = "invalid_value"
    error.param = param
    raise error


def _too_long(value: str, param: str) -> Never:
    """Raise the 400 an over-long identifier gets.

    Args:
        value: The rejected identifier.
        param: Name of the request parameter carrying it.

    Raises:
        ApiError: Always, with status 400.
    """
    error = ApiError(
        f"'{param}' must be at most {_ID_MAX_LENGTH} characters long, "
        f"got {len(value)}.",
        status=400,
    )
    error.code = "string_above_max_length"
    error.param = param
    raise error


def validate_conversation_id(conversation_id: str, param: str) -> None:
    """Reject a conversation ID no conversation can ever have.

    An ID that is well-formed but was minted elsewhere resolves to a 404, so a
    client moving over from another provider gets "not found" rather than a
    validation error.

    Args:
        conversation_id: The identifier to check.
        param: Name of the request parameter carrying it.

    Raises:
        ApiError: 400 when the identifier is malformed, 404 when it is
            well-formed but cannot name a conversation on this server.
    """
    if len(conversation_id) > _ID_MAX_LENGTH:
        _too_long(conversation_id, param)
    if not _CONVERSATION_ID_INPUT.fullmatch(conversation_id):
        _invalid_id(conversation_id, param, "an ID that begins with 'conv'")
    if conversation_id.startswith("conv_"):
        conversation_not_found(conversation_id)


def validate_item_id(item_id: str) -> None:
    """Reject an item ID no conversation item can ever have.

    Args:
        item_id: The identifier to check.

    Raises:
        ApiError: 400 when the identifier is malformed.
    """
    if len(item_id) > _ID_MAX_LENGTH:
        _too_long(item_id, "item_id")
    if not _ITEM_ID_INPUT.fullmatch(item_id):
        _invalid_id(
            item_id,
            "item_id",
            "an ID made of a type prefix, an underscore, and the item's payload",
        )


def stored_item(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Turn a request or response item into the item stored in a conversation.

    The client's own ``id`` is replaced: an item's identity belongs to the
    conversation that holds it. String message content is expanded into
    content parts so the item reads back exactly as it was listed.

    Args:
        payload: The item as JSON, from a request body or a model response.

    Returns:
        The item to store, carrying its minted public ID.
    """
    item = {key: value for key, value in payload.items() if value is not None}
    item.setdefault("type", "message")
    item["id"] = new_item_id(item["type"])
    if item["type"] == "message":
        item.setdefault("status", "completed")
        if isinstance(content := item.get("content"), str):
            item["content"] = (
                [{"type": "output_text", "text": content, "annotations": []}]
                if item.get("role") == "assistant"
                else [{"type": "input_text", "text": content}]
            )
    return item


def is_item_reference(payload: Mapping[str, Any]) -> bool:
    """Whether an item payload references an existing item instead of adding one.

    Args:
        payload: The item as JSON, from a request body.

    Returns:
        True for an ``item_reference`` item.
    """
    return payload.get("type") == _REFERENCE_TYPE


def _client() -> AgentsforBedrockRuntimeClient:
    """Return the bedrock-agent-runtime client of the primary Bedrock region."""
    client: AgentsforBedrockRuntimeClient = get_client(
        "bedrock-agent-runtime", SETTINGS.aws_bedrock_regions[0]
    )
    return client


def _session_id(conversation_id: str) -> str:
    """Return the AWS Bedrock session ID backing *conversation_id*."""
    return conversation_id.split("-", 1)[-1]


def conversation_not_found(conversation_id: str) -> Never:
    """Raise the conversation 404 error.

    Args:
        conversation_id: The conversation that does not exist.

    Raises:
        ApiError: Always, with status 404.
    """
    msg = f"Conversation with id '{conversation_id}' not found."
    raise ApiError(msg, status=404)


def item_not_found(item_id: str) -> Never:
    """Raise the conversation item 404 error.

    Args:
        item_id: The item that does not exist in the conversation.

    Raises:
        ApiError: Always, with status 404.
    """
    msg = f"Item with id '{item_id}' not found in conversation."
    raise ApiError(msg, status=404)


def new_item_id(item_type: str | None) -> str:
    """Mint a public ID for a new conversation item.

    Args:
        item_type: The item's ``type`` field, when it has one.

    Returns:
        An item ID prefixed according to the item type.
    """
    prefix = _ITEM_ID_PREFIXES.get(item_type or "", _ITEM_ID_FALLBACK_PREFIX)
    return f"{prefix}_{uuid4().hex}"


async def _session_metadata(
    client: AgentsforBedrockRuntimeClient, conversation_id: str
) -> tuple[int, dict[str, str], str]:
    """Read a conversation's session, rejecting a session of another kind.

    Args:
        client: bedrock-agent-runtime client.
        conversation_id: Public conversation identifier.

    Returns:
        Tuple of (creation timestamp, metadata, session ARN).

    Raises:
        ApiError: 404 when the conversation does not exist, its identifier
            cannot name a session, or the session holds another object kind.
    """
    session_id = _session_id(conversation_id)
    with (
        _session_calls("GetSession", "ListTagsForResource"),
        handle_bedrock_client_error(),
    ):
        try:
            session = await client.get_session(sessionIdentifier=session_id)
            tags = await client.list_tags_for_resource(
                resourceArn=session["sessionArn"]
            )
        except ClientError as error:
            if is_unknown_identifier(error):
                conversation_not_found(conversation_id)
            raise
    if tags.get("tags", {}).get(KIND_TAG) != _KIND:
        conversation_not_found(conversation_id)
    return (
        int(session["createdAt"].timestamp()),
        dict(session.get("sessionMetadata") or {}),
        session["sessionArn"],
    )


async def create_conversation(metadata: Mapping[str, str]) -> tuple[str, int]:
    """Create a conversation and its backing session.

    Args:
        metadata: Key-value pairs to attach to the conversation.

    Returns:
        Tuple of (conversation ID, creation timestamp in seconds).
    """
    client = _client()
    key = SETTINGS.aws_bedrock_session_encryption_key_arn
    with (
        _session_calls("CreateSession", "CreateInvocation", "PutInvocationStep"),
        handle_bedrock_client_error(),
    ):
        session = await client.create_session(
            tags=build_metadata(apn=True) | {KIND_TAG: _KIND},
            sessionMetadata=dict(metadata),
            **({"encryptionKeyArn": key} if key else {}),
        )
        conversation_id = f"conv-{session['sessionId']}"
        try:
            # Marks the session as a conversation, sparing the item routes a tag lookup.
            await _put_conversation_document(
                client, conversation_id, {_ITEMS_FIELD: []}
            )
        except BaseException:
            # A half-created conversation is unreachable, so drop it now.
            with suppress(ClientError):
                await end_and_delete_session(client, session["sessionId"])
            raise
    return conversation_id, int(session["createdAt"].timestamp())


async def get_conversation(conversation_id: str) -> tuple[int, dict[str, str]]:
    """Read a conversation's creation time and metadata.

    Args:
        conversation_id: Public conversation identifier.

    Returns:
        Tuple of (creation timestamp in seconds, metadata).

    Raises:
        ApiError: 404 when the conversation does not exist.
    """
    created_at, metadata, _ = await _session_metadata(_client(), conversation_id)
    return created_at, metadata


async def update_conversation(
    conversation_id: str, patch: Mapping[str, Any]
) -> tuple[int, dict[str, str]]:
    """Merge key-value pairs into a conversation's metadata.

    A key whose value is ``None`` is removed. The merge is read-modify-write,
    so a key absent from *patch* keeps its stored value.

    Args:
        conversation_id: Public conversation identifier.
        patch: Key-value pairs to merge, with ``None`` meaning removal.

    Returns:
        Tuple of (creation timestamp in seconds, resulting metadata).

    Raises:
        ApiError: 404 when the conversation does not exist.
    """
    client = _client()
    created_at, metadata, _ = await _session_metadata(client, conversation_id)
    for key, value in patch.items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    if patch:
        with _session_calls("UpdateSession"), handle_bedrock_client_error():
            await client.update_session(
                sessionIdentifier=_session_id(conversation_id), sessionMetadata=metadata
            )
    return created_at, metadata


async def delete_conversation(conversation_id: str) -> None:
    """Delete a conversation, its items and its metadata.

    Args:
        conversation_id: Public conversation identifier.

    Raises:
        ApiError: 404 when the conversation does not exist.
    """
    client = _client()
    await _session_metadata(client, conversation_id)
    with (
        _session_calls("EndSession", "DeleteSession"),
        handle_bedrock_client_error(),
        not_found_as_404(lambda: conversation_not_found(conversation_id)),
    ):
        await end_and_delete_session(client, _session_id(conversation_id))


async def _put_conversation_document(
    client: AgentsforBedrockRuntimeClient,
    conversation_id: str,
    document: Mapping[str, Any],
) -> None:
    """Append one conversation write to the backing session.

    Args:
        client: bedrock-agent-runtime client.
        conversation_id: Public conversation identifier.
        document: The write to persist (appended items or deleted item IDs).
    """
    await put_document(
        client,
        _session_id(conversation_id),
        {_DOCUMENT_KEY: dict(document)},
        chunk_size=_CHUNK_SIZE,
        concurrency=_STEP_CONCURRENCY,
    )


async def load_items(conversation_id: str) -> list[dict[str, Any]]:
    """Read a conversation's items, oldest first.

    Args:
        conversation_id: Public conversation identifier.

    Returns:
        The conversation's items, in the order they were added.

    Raises:
        ApiError: 404 when the conversation does not exist.
    """
    with (
        _session_calls("ListInvocations", "ListInvocationSteps", "GetInvocationStep"),
        handle_bedrock_client_error(),
        not_found_as_404(lambda: conversation_not_found(conversation_id)),
    ):
        documents = await load_documents(
            _client(),
            _session_id(conversation_id),
            concurrency=_STEP_CONCURRENCY,
            scan_limit=_SCAN_LIMIT,
        )
    items: dict[str, dict[str, Any]] = {}
    found = False
    for document in documents:
        write = document.get(_DOCUMENT_KEY)
        if not isinstance(write, dict):
            continue
        found = True
        for item in write.get(_ITEMS_FIELD) or ():
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                items[item["id"]] = item
        for item_id in write.get(_DELETED_FIELD) or ():
            if isinstance(item_id, str):
                items.pop(item_id, None)
    if not found:
        conversation_not_found(conversation_id)
    return list(items.values())


async def append_items(
    conversation_id: str, items: Iterable[Mapping[str, Any]]
) -> None:
    """Append items to a conversation.

    Args:
        conversation_id: Public conversation identifier.
        items: Items to append, each already carrying its public ID.

    Raises:
        ApiError: 404 when the conversation does not exist.
    """
    with (
        _session_calls("CreateInvocation", "PutInvocationStep"),
        handle_bedrock_client_error(),
        not_found_as_404(lambda: conversation_not_found(conversation_id)),
    ):
        await _put_conversation_document(
            _client(), conversation_id, {_ITEMS_FIELD: list(items)}
        )


async def delete_item(conversation_id: str, item_id: str) -> None:
    """Remove one item from a conversation.

    Args:
        conversation_id: Public conversation identifier.
        item_id: Public identifier of the item to remove.

    Raises:
        ApiError: 404 when the conversation or the item does not exist.
    """
    if not any(item["id"] == item_id for item in await load_items(conversation_id)):
        item_not_found(item_id)
    with (
        _session_calls("CreateInvocation", "PutInvocationStep"),
        handle_bedrock_client_error(),
        not_found_as_404(lambda: conversation_not_found(conversation_id)),
    ):
        await _put_conversation_document(
            _client(), conversation_id, {_DELETED_FIELD: [item_id]}
        )
