"""Stored generations backed by AWS Bedrock session management.

Responses and chat completions created with ``store=true`` persist in AWS
Bedrock sessions (bedrock-agent-runtime): one session per stored object,
whose invocation steps carry the JSON document (chunked into text blocks);
updates append a new invocation and reads use the latest one. AWS keeps all
state, so any server instance can retrieve, list, update, delete, or
continue a stored object without shared server state.

The stored object ID is its API ID (``resp-<session ID>`` or
``chatcmpl-<session ID>``); the object kind is recorded as a session tag so
listings can tell chat completions and responses apart. Sessions live in
the primary Bedrock region.
"""

from asyncio import Semaphore, gather
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Literal, Never

from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    EndpointResolutionError,
)
from pydantic_core import from_json, to_json

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.aws_bedrock import handle_bedrock_client_error
from stdapi.config import SETTINGS
from stdapi.monitoring import build_metadata, log_error_details

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from types_aiobotocore_bedrock_agent_runtime.client import (
        AgentsforBedrockRuntimeClient,
    )
    from types_aiobotocore_bedrock_agent_runtime.type_defs import (
        InvocationStepSummaryTypeDef,
        InvocationSummaryTypeDef,
        SessionSummaryTypeDef,
    )

#: Regex pattern that a valid stored response ID must match.
RESPONSE_ID_PATTERN: str = r"^resp-[A-Za-z0-9-]+$"

#: Regex pattern that a valid stored chat completion ID must match.
COMPLETION_ID_PATTERN: str = r"^chatcmpl-[A-Za-z0-9-]+$"

#: Kind of object persisted in a session, recorded as a session tag.
type StoredObjectKind = Literal["chat_completion", "response"]

#: Session tag key holding the stored object kind.
_KIND_TAG: str = "stdapi-ai.stored-object"

#: Maximum UTF-8 bytes per invocation step text block (stays under the payload quota).
_CHUNK_SIZE: int = 200_000

#: Maximum sessions scanned when listing stored objects.
_LIST_SCAN_LIMIT: int = 1_000

#: Maximum sessions per ListSessions page.
_LIST_PAGE_SIZE: int = 100

#: Maximum concurrent tag lookups when resolving a page of sessions' kinds.
_TAG_FETCH_CONCURRENCY: int = 16

#: Maximum concurrent invocation-step puts when persisting a chunked document.
_STEP_PUT_CONCURRENCY: int = 8

#: AWS error code surfaced as a stored-object 404.
_NOT_FOUND_CODE: str = "ResourceNotFoundException"

#: AWS error codes meaning the identifier cannot name a stored object.
_IDENTIFIER_ERROR_CODES: frozenset[str] = frozenset(
    {_NOT_FOUND_CODE, "ValidationException"}
)

#: EndSession error codes meaning the session need not or cannot be ended (no-op).
_END_TOLERATED_CODES: frozenset[str] = frozenset(
    {"ConflictException", "ValidationException"}
)

#: Cache of session ID to stored-object kind, avoiding repeated tag fetches.
_KIND_CACHE: dict[str, str] = {}

#: Cache size above which it is cleared, since the kind tag never changes.
_KIND_CACHE_LIMIT: int = 4_096

#: Sentinel cached value meaning the session has no kind tag (untagged/foreign).
_UNTAGGED: str = ""

#: AWS error code meaning session storage is not enabled on this server.
_ACCESS_DENIED_CODE = "AccessDeniedException"

#: Botocore failures meaning the session API endpoint is unreachable or timed out.
_UNREACHABLE_ERRORS = (
    EndpointConnectionError,
    EndpointResolutionError,
    ConnectTimeoutError,
)

#: Stored object kind declared by each document ``response.object`` field.
_KIND_BY_OBJECT: dict[str, StoredObjectKind] = {
    "response": "response",
    "chat.completion": "chat_completion",
}


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


def _document_kind(document: Mapping[str, Any]) -> StoredObjectKind | None:
    """Return the stored object kind declared by a document's response object field.

    Args:
        document: Loaded stored document.

    Returns:
        The declared stored object kind, or None if undeclared or unrecognized.
    """
    response = document.get("response")
    if not isinstance(response, dict):
        return None
    declared_object = response.get("object")
    return (
        _KIND_BY_OBJECT.get(declared_object)
        if isinstance(declared_object, str)
        else None
    )


def _kind_mismatches(document: Mapping[str, Any], kind: StoredObjectKind) -> bool:
    """Whether *document* lacks the minimal expected shape for *kind*.

    Args:
        document: Loaded stored document.
        kind: Expected stored object kind.

    Returns:
        True unless the document declares exactly the expected kind.
    """
    return _document_kind(document) != kind


@contextmanager
def _not_found_as_404(response_id: str) -> Generator[None]:
    """Map an identifier-lookup error to the stored-object 404 error.

    Args:
        response_id: Stored object ID reported in the 404 message.

    Raises:
        ApiError: 404 when the backing Bedrock session does not exist or the
            identifier cannot name one (e.g. it fails AWS's ID validation).
    """
    try:
        yield
    except ClientError as exc:
        if exc.response["Error"]["Code"] in _IDENTIFIER_ERROR_CODES:
            _not_found(response_id)
        raise


async def create_stored_response_session(kind: StoredObjectKind) -> str:
    """Create the AWS Bedrock session backing a stored object.

    Args:
        kind: Stored object kind, recorded as a session tag for listing.

    Returns:
        The session ID; the stored object ID is the session ID with the
        kind's API prefix.
    """
    client = _client()
    key = SETTINGS.aws_bedrock_session_encryption_key_arn
    tags = build_metadata(apn=True) | {_KIND_TAG: kind}
    with handle_bedrock_client_error():
        response = await client.create_session(
            tags=tags,
            **({"encryptionKeyArn": key} if key else {}),  # type: ignore[arg-type]
        )
    session_id: str = response["sessionId"]
    _KIND_CACHE[session_id] = kind
    return session_id


async def try_create_stored_response_session(kind: StoredObjectKind) -> str | None:
    """Create the backing session, or ``None`` when storage is unavailable.

    Two conditions are treated as "session storage not available here": an AWS
    ``AccessDeniedException``, and an endpoint that cannot be reached at all
    because the region does not serve the session API. Both leave the request to
    proceed with ``store`` ignored and record a warning for the administrator,
    rather than failing a generation the model already produced.

    Args:
        kind: Stored object kind, recorded as a session tag for listing.

    Returns:
        The session ID, or ``None`` when session storage is unavailable.
    """
    try:
        return await create_stored_response_session(kind)
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
    except _UNREACHABLE_ERRORS:
        log_error_details(
            "Amazon Bedrock session storage endpoint is unreachable or timed "
            "out: 'store' was ignored. Session storage is offered in fewer "
            "regions than model inference; configure a region that provides it "
            "to store responses and chat completions.",
            level="warning",
        )
        return None


async def _cached_kind_tag(
    client: AgentsforBedrockRuntimeClient, session_id: str, session_arn: str
) -> str | None:
    """Return a session's cached (or freshly fetched) kind tag value.

    Args:
        client: bedrock-agent-runtime client.
        session_id: Session ID, used as the cache key.
        session_arn: Session ARN, used to fetch the tag on a cache miss.

    Returns:
        The session's kind tag value, or None when untagged.
    """
    if session_id in _KIND_CACHE:
        return _KIND_CACHE[session_id] or None
    tags = await client.list_tags_for_resource(resourceArn=session_arn)
    session_kind = tags.get("tags", {}).get(_KIND_TAG)
    if len(_KIND_CACHE) > _KIND_CACHE_LIMIT:
        _KIND_CACHE.clear()
    _KIND_CACHE[session_id] = session_kind or _UNTAGGED
    return session_kind


async def _session_kind_tag_or_none(
    client: AgentsforBedrockRuntimeClient, response_id: str
) -> str | None:
    """Return a stored object's session kind tag, tolerating a missing session.

    Args:
        client: bedrock-agent-runtime client.
        response_id: Stored object ID.

    Returns:
        The session's kind tag, or None when untagged, the session does not
        exist, or the identifier cannot name one (the caller's own delete
        call then surfaces the 404).
    """
    session_id = _session_id(response_id)
    with handle_bedrock_client_error():
        try:
            session = await client.get_session(sessionIdentifier=session_id)
            return await _cached_kind_tag(client, session_id, session["sessionArn"])
        except ClientError as exc:
            if exc.response["Error"]["Code"] in _IDENTIFIER_ERROR_CODES:
                return None
            raise


async def list_stored_sessions(kind: StoredObjectKind) -> list[tuple[str, datetime]]:
    """List the sessions holding stored objects of *kind*.

    The scan is capped at ``_LIST_SCAN_LIMIT`` sessions; sessions created by
    other tools (no kind tag) are ignored.

    Args:
        kind: Stored object kind to list.

    Returns:
        Unordered ``(session ID, creation time)`` pairs.
    """
    client = _client()
    semaphore = Semaphore(_TAG_FETCH_CONCURRENCY)

    async def _kind_tag(summary: SessionSummaryTypeDef) -> str | None:
        """Return the session's cached (or fetched) kind tag value, bounding concurrency."""
        async with semaphore:
            return await _cached_kind_tag(
                client, summary["sessionId"], summary["sessionArn"]
            )

    sessions: list[tuple[str, datetime]] = []
    scanned = 0
    token: str | None = None
    with handle_bedrock_client_error():
        while scanned < _LIST_SCAN_LIMIT:
            page = await client.list_sessions(
                maxResults=min(_LIST_PAGE_SIZE, _LIST_SCAN_LIMIT - scanned),
                **({"nextToken": token} if token else {}),
            )
            summaries = page.get("sessionSummaries", [])
            scanned += len(summaries)
            kinds = await gather(*map(_kind_tag, summaries))
            sessions.extend(
                (summary["sessionId"], summary["createdAt"])
                for summary, session_kind in zip(summaries, kinds, strict=True)
                if session_kind == kind
            )
            token = page.get("nextToken")
            if not token:
                break
    return sessions


def _iter_utf8_chunks(data: bytes, limit: int) -> Generator[str]:
    """Split *data* into UTF-8-decodable chunks of at most *limit* bytes.

    Each boundary backtracks off a UTF-8 continuation byte (at most 3 bytes)
    so a chunk never splits a multibyte code point.

    Args:
        data: UTF-8 encoded bytes to split.
        limit: Maximum bytes per chunk.

    Yields:
        Each chunk, decoded back to text.
    """
    offset = 0
    length = len(data)
    while offset < length:
        end = min(offset + limit, length)
        while end > offset and end < length and data[end] & 0xC0 == 0x80:
            end -= 1
        yield data[offset:end].decode()
        offset = end


async def save_stored_response(response_id: str, document: Mapping[str, Any]) -> None:
    """Write the stored response document into its session.

    Each call appends a new invocation; reads use the latest one, so saving
    again replaces the visible document (e.g. on a metadata update). Chunks
    are written concurrently (bounded by ``_STEP_PUT_CONCURRENCY``); each
    step carries a sequential timestamp so reads can reorder the chunks.

    Args:
        response_id: Stored response ID (its session must already exist).
        document: JSON-serializable document to persist.

    Raises:
        ApiError: Via ``handle_bedrock_client_error``, when the underlying
            Bedrock call fails with a recognised error code.
    """
    client = _client()
    session_id = _session_id(response_id)
    data = to_json(document)
    start = datetime.now(tz=UTC)
    semaphore = Semaphore(_STEP_PUT_CONCURRENCY)

    async def _put_step(index: int, chunk: str, invocation_id: str) -> None:
        """Put one chunk step under the write concurrency bound."""
        async with semaphore:
            await client.put_invocation_step(
                sessionIdentifier=session_id,
                invocationIdentifier=invocation_id,
                invocationStepTime=start + timedelta(seconds=index),
                payload={"contentBlocks": [{"text": chunk}]},
            )

    with handle_bedrock_client_error():
        invocation_id = (await client.create_invocation(sessionIdentifier=session_id))[
            "invocationId"
        ]
        await gather(
            *(
                _put_step(index, chunk, invocation_id)
                for index, chunk in enumerate(_iter_utf8_chunks(data, _CHUNK_SIZE))
            )
        )


async def _load_invocation_document(
    client: AgentsforBedrockRuntimeClient, session_id: str, invocation_id: str
) -> dict[str, Any] | None:
    """Fetch and parse the document stored in one invocation.

    Args:
        client: bedrock-agent-runtime client.
        session_id: Backing session ID.
        invocation_id: Invocation to read.

    Returns:
        The parsed document, or None when the invocation has no steps or its
        payload does not parse as a JSON object (an interrupted or truncated
        write, or a foreign session storing something else entirely).
    """
    steps: list[InvocationStepSummaryTypeDef] = []
    token: str | None = None
    while True:
        steps_page = await client.list_invocation_steps(
            sessionIdentifier=session_id,
            invocationIdentifier=invocation_id,
            **({"nextToken": token} if token else {}),  # type: ignore[arg-type]
        )
        steps.extend(steps_page.get("invocationStepSummaries", []))
        token = steps_page.get("nextToken")
        if not token:
            break
    if not steps:
        return None
    steps.sort(key=lambda step: step["invocationStepTime"])
    details = await gather(
        *(
            client.get_invocation_step(
                sessionIdentifier=session_id,
                invocationIdentifier=invocation_id,
                invocationStepId=step["invocationStepId"],
            )
            for step in steps
        )
    )
    parts = [
        block["text"]
        for detail in details
        for block in detail["invocationStep"]["payload"]["contentBlocks"]
        if "text" in block
    ]
    if not parts:
        return None
    try:
        document = from_json("".join(parts))
    except ValueError:
        return None
    return document if isinstance(document, dict) else None


async def _stored_document_or_none(response_id: str) -> dict[str, Any] | None:
    """Read the newest parseable document stored for *response_id*, if any.

    Invocations are tried newest first, and the first one with a fully
    written, parseable document is returned. This tolerates an update
    interrupted between creating the latest invocation and writing its
    steps, or a truncated read racing a concurrent write, by falling back
    to the last-good invocation.

    Args:
        response_id: Stored response ID.

    Returns:
        The persisted document, or None if the backing session does not
        exist, holds no parseable document, or the identifier cannot name a
        session (e.g. it fails AWS's ID validation).
    """
    client = _client()
    session_id = _session_id(response_id)
    summaries: list[InvocationSummaryTypeDef] = []
    with handle_bedrock_client_error():
        try:
            token: str | None = None
            while True:
                page = await client.list_invocations(
                    sessionIdentifier=session_id,
                    **({"nextToken": token} if token else {}),  # type: ignore[arg-type]
                )
                summaries.extend(page.get("invocationSummaries", []))
                token = page.get("nextToken")
                if not token:
                    break
            if not summaries:
                return None
            for summary in sorted(summaries, key=itemgetter("createdAt"), reverse=True):
                document = await _load_invocation_document(
                    client, session_id, summary["invocationId"]
                )
                if document is not None:
                    return document
        except ClientError as exc:
            if exc.response["Error"]["Code"] in _IDENTIFIER_ERROR_CODES:
                return None
            raise
        else:
            return None


async def load_stored_response(
    response_id: str, kind: StoredObjectKind
) -> dict[str, Any]:
    """Read a stored object document from its session.

    Args:
        response_id: Stored object ID.
        kind: Expected stored object kind.

    Returns:
        The persisted document.

    Raises:
        ApiError: 404 when the stored object does not exist, the identifier
            is malformed, none of its invocations hold a parseable document,
            or the document does not declare the expected kind.
    """
    document = await _stored_document_or_none(response_id)
    if document is None or _kind_mismatches(document, kind):
        _not_found(response_id)
    return document


async def delete_stored_response(response_id: str, kind: StoredObjectKind) -> None:
    """Delete a stored object and its backing session.

    The mismatch check reads the session's kind tag (one cheap API call) rather
    than the full document, avoiding extra latency and throttling exposure. An
    untagged session, such as an orphan left by a failed generation, is deleted
    unconditionally.

    Args:
        response_id: Stored object ID.
        kind: Expected stored object kind.

    Raises:
        ApiError: 404 when the stored object does not exist, the identifier
            is malformed, or its session is tagged with a different kind.
    """
    client = _client()
    session_kind = await _session_kind_tag_or_none(client, response_id)
    if session_kind is not None and session_kind != kind:
        _not_found(response_id)
    session_id = _session_id(response_id)
    with _not_found_as_404(response_id):
        try:
            # Sessions must be ended before deletion; tolerate already-ended
            # (state errors defer to the delete call, which surfaces real issues).
            await client.end_session(sessionIdentifier=session_id)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in _END_TOLERATED_CODES:
                raise
        await client.delete_session(sessionIdentifier=session_id)
    _KIND_CACHE.pop(session_id, None)


async def discard_stored_response_session(
    response_id: str, kind: StoredObjectKind
) -> None:
    """Best-effort cleanup of a session whose generation failed.

    Args:
        response_id: Stored object ID.
        kind: Expected stored object kind (the just-created session's kind).
    """
    with suppress(ClientError, ApiError):
        await delete_stored_response(response_id, kind)
