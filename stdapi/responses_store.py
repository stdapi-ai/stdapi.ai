"""Stored generations backed by AWS Bedrock session management.

Responses and chat completions created with ``store=true`` persist in AWS
Bedrock sessions (bedrock-agent-runtime): one session per stored object,
holding a single invocation whose steps carry the JSON document (chunked
into text blocks). AWS keeps all state, so any server instance can retrieve,
delete, or continue a stored object without shared server state.

The stored object ID is its API ID (``resp-<session ID>`` or
``chatcmpl-<session ID>``). Sessions live in the primary Bedrock region.
"""

from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Never

from botocore.exceptions import ClientError
from pydantic_core import from_json, to_json

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.aws_bedrock import handle_bedrock_client_error
from stdapi.config import SETTINGS
from stdapi.monitoring import build_metadata, log_error_details

if TYPE_CHECKING:
    from collections.abc import Mapping

    from types_aiobotocore_bedrock_agent_runtime.client import (
        AgentsforBedrockRuntimeClient,
    )

#: Regex pattern that a valid stored response ID must match.
RESPONSE_ID_PATTERN: str = r"^resp-[A-Za-z0-9-]+$"

#: Regex pattern that a valid stored chat completion ID must match.
COMPLETION_ID_PATTERN: str = r"^chatcmpl-[A-Za-z0-9-]+$"

#: Maximum characters per invocation step text block (stays under the payload quota).
_CHUNK_SIZE: int = 200_000

#: AWS error code surfaced as a stored-object 404.
_NOT_FOUND_CODE = "ResourceNotFoundException"

#: AWS error code meaning session storage is not enabled on this server.
_ACCESS_DENIED_CODE = "AccessDeniedException"


def _session_id(response_id: str) -> str:
    """Return the AWS Bedrock session ID backing *response_id*."""
    return response_id.split("-", 1)[-1]


def _client() -> AgentsforBedrockRuntimeClient:
    """Return the bedrock-agent-runtime client of the primary Bedrock region."""
    client: AgentsforBedrockRuntimeClient = get_client(
        "bedrock-agent-runtime", SETTINGS.aws_bedrock_regions[0]
    )
    return client


def _not_found(response_id: str) -> Never:
    """Raise the stored-object 404 error.

    Raises:
        ApiError: Always, with status 404.
    """
    noun = "Chat completion" if response_id.startswith("chatcmpl-") else "Response"
    msg = f"{noun} with id '{response_id}' not found."
    raise ApiError(msg, status=404)


async def create_stored_response_session() -> str:
    """Create the AWS Bedrock session backing a stored response.

    Returns:
        The session ID; ``resp-<session ID>`` is the stored response ID.
    """
    client = _client()
    extra: dict[str, Any] = (
        {"encryptionKeyArn": key}
        if (key := SETTINGS.aws_bedrock_session_encryption_key_arn)
        else {}
    )
    with handle_bedrock_client_error():
        response = await client.create_session(tags=build_metadata(apn=True), **extra)
    return response["sessionId"]


async def try_create_stored_response_session() -> str | None:
    """Create the backing session, or ``None`` when storage is unavailable.

    An AWS ``AccessDeniedException`` is treated as "session storage not
    enabled on this server": the request proceeds with ``store`` ignored and
    a warning is recorded in the request log for the administrator.

    Returns:
        The session ID, or ``None`` when session storage is unavailable.
    """
    try:
        return await create_stored_response_session()
    except ClientError as exc:
        if exc.response["Error"]["Code"] != _ACCESS_DENIED_CODE:
            raise
        log_error_details(
            "Bedrock session storage is not enabled (AccessDenied on "
            "bedrock:CreateSession): 'store' was ignored. Grant the Bedrock "
            "session storage IAM permissions to enable stored responses and "
            "chat completions.",
            level="warning",
        )
        return None


async def save_stored_response(response_id: str, document: Mapping[str, Any]) -> None:
    """Write the stored response document into its session.

    Steps are timestamped sequentially so reads can reorder the chunks.

    Args:
        response_id: Stored response ID (its session must already exist).
        document: JSON-serializable document to persist.
    """
    client = _client()
    session_id = _session_id(response_id)
    data = to_json(document).decode()
    start = datetime.now(tz=UTC)
    with handle_bedrock_client_error():
        invocation_id = (await client.create_invocation(sessionIdentifier=session_id))[
            "invocationId"
        ]
        for index, offset in enumerate(range(0, len(data), _CHUNK_SIZE)):
            await client.put_invocation_step(
                sessionIdentifier=session_id,
                invocationIdentifier=invocation_id,
                invocationStepTime=start + timedelta(seconds=index),
                payload={
                    "contentBlocks": [{"text": data[offset : offset + _CHUNK_SIZE]}]
                },
            )


async def load_stored_response(response_id: str) -> dict[str, Any]:
    """Read a stored response document from its session.

    Args:
        response_id: Stored response ID.

    Returns:
        The persisted document.

    Raises:
        ApiError: 404 when the stored response does not exist.
    """
    client = _client()
    session_id = _session_id(response_id)
    try:
        invocations = await client.list_invocations(sessionIdentifier=session_id)
    except ClientError as exc:
        if exc.response["Error"]["Code"] == _NOT_FOUND_CODE:
            _not_found(response_id)
        raise
    if not (summaries := invocations.get("invocationSummaries", [])):
        _not_found(response_id)
    invocation_id = summaries[0]["invocationId"]
    steps: list[Any] = []
    token: str | None = None
    with handle_bedrock_client_error():
        while True:
            page = await client.list_invocation_steps(
                sessionIdentifier=session_id,
                invocationIdentifier=invocation_id,
                **({"nextToken": token} if token else {}),  # type: ignore[arg-type]
            )
            steps.extend(page.get("invocationStepSummaries", []))
            token = page.get("nextToken")
            if not token:
                break
        steps.sort(key=lambda step: step["invocationStepTime"])
        parts: list[str] = []
        for step in steps:
            detail = await client.get_invocation_step(
                sessionIdentifier=session_id,
                invocationIdentifier=invocation_id,
                invocationStepId=step["invocationStepId"],
            )
            parts.extend(
                block["text"]
                for block in detail["invocationStep"]["payload"]["contentBlocks"]
                if "text" in block
            )
    if not parts:
        _not_found(response_id)
    document: dict[str, Any] = from_json("".join(parts))
    return document


async def delete_stored_response(response_id: str) -> None:
    """Delete a stored response and its backing session.

    Args:
        response_id: Stored response ID.

    Raises:
        ApiError: 404 when the stored response does not exist.
    """
    client = _client()
    session_id = _session_id(response_id)
    try:
        with suppress(ClientError):
            # Sessions must be ended before deletion; tolerate already-ended.
            await client.end_session(sessionIdentifier=session_id)
        await client.delete_session(sessionIdentifier=session_id)
    except ClientError as exc:
        if exc.response["Error"]["Code"] == _NOT_FOUND_CODE:
            _not_found(response_id)
        raise


async def discard_stored_response_session(response_id: str) -> None:
    """Best-effort cleanup of a session whose generation failed."""
    with suppress(ClientError, ApiError):
        await delete_stored_response(response_id)
