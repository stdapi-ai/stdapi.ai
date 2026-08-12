"""Chunked JSON documents stored in AWS Bedrock session management.

AWS Bedrock sessions (bedrock-agent-runtime) are used as a document store:
one session per stored object, whose invocations each hold a JSON document
split across invocation step text blocks. This module carries the wire half —
chunking, writing, reading, deleting and tag-filtered listing — while the
caller keeps the client, the tuning knobs, the ownership tag and the
not-found error it raises, so every stored object kind keeps its own policy.
"""

from asyncio import Semaphore, gather
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Never

from botocore.exceptions import ClientError
from pydantic_core import from_json, to_json

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator, Mapping

    from types_aiobotocore_bedrock_agent_runtime.client import (
        AgentsforBedrockRuntimeClient,
    )
    from types_aiobotocore_bedrock_agent_runtime.type_defs import (
        InvocationStepSummaryTypeDef,
        InvocationSummaryTypeDef,
        SessionSummaryTypeDef,
    )

#: AWS error codes meaning an identifier cannot name an existing session.
_IDENTIFIER_ERROR_CODES: frozenset[str] = frozenset(
    {"ResourceNotFoundException", "ValidationException"}
)

#: EndSession error codes meaning the session need not or cannot be ended (no-op).
_END_TOLERATED_CODES: frozenset[str] = frozenset(
    {"ConflictException", "ValidationException"}
)


def is_unknown_identifier(exc: ClientError) -> bool:
    """Whether *exc* means the identifier cannot name an existing session.

    Args:
        exc: Error raised by a session API call.

    Returns:
        True when the session does not exist, or when the identifier cannot
        name one at all because it fails AWS's own ID validation.
    """
    return exc.response["Error"]["Code"] in _IDENTIFIER_ERROR_CODES


@contextmanager
def not_found_as_404(not_found: Callable[[], Never]) -> Generator[None]:
    """Map an identifier-lookup failure to the caller's own not-found error.

    Args:
        not_found: Called to raise the caller's 404 when the guarded calls
            report an identifier that cannot name a session.

    Yields:
        None, while the guarded session calls run.

    Raises:
        ClientError: Any session API failure that is not an identifier lookup.
    """
    try:
        yield
    except ClientError as exc:
        if is_unknown_identifier(exc):
            not_found()
        raise


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


async def put_document(
    client: AgentsforBedrockRuntimeClient,
    session_id: str,
    document: Mapping[str, Any],
    *,
    chunk_size: int,
    concurrency: int,
) -> None:
    """Append *document* to a session as one new chunked invocation.

    Chunks are written concurrently (bounded by *concurrency*); each step
    carries a sequential timestamp so reads can reorder the chunks.

    Args:
        client: bedrock-agent-runtime client.
        session_id: Session to append to (it must already exist).
        document: JSON-serializable document to persist.
        chunk_size: Maximum UTF-8 bytes per invocation step text block.
        concurrency: Maximum concurrent invocation step puts.
    """
    data = to_json(document)
    start = datetime.now(tz=UTC)
    semaphore = Semaphore(concurrency)

    async def _put_step(index: int, chunk: str, invocation_id: str) -> None:
        """Put one chunk step under the write concurrency bound."""
        async with semaphore:
            await client.put_invocation_step(
                sessionIdentifier=session_id,
                invocationIdentifier=invocation_id,
                invocationStepTime=start + timedelta(seconds=index),
                payload={"contentBlocks": [{"text": chunk}]},
            )

    invocation_id = (await client.create_invocation(sessionIdentifier=session_id))[
        "invocationId"
    ]
    await gather(
        *(
            _put_step(index, chunk, invocation_id)
            for index, chunk in enumerate(_iter_utf8_chunks(data, chunk_size))
        )
    )


async def list_session_invocations(
    client: AgentsforBedrockRuntimeClient, session_id: str
) -> list[InvocationSummaryTypeDef]:
    """List every invocation of a session, following the listing pages.

    Args:
        client: bedrock-agent-runtime client.
        session_id: Session to list.

    Returns:
        The invocation summaries, in AWS's own listing order.
    """
    summaries: list[InvocationSummaryTypeDef] = []
    token: str | None = None
    while True:
        page = await client.list_invocations(
            sessionIdentifier=session_id,
            **({"nextToken": token} if token else {}),  # type: ignore[arg-type]
        )
        summaries.extend(page.get("invocationSummaries", []))
        token = page.get("nextToken")
        if not token:
            return summaries


async def load_invocation_document(
    client: AgentsforBedrockRuntimeClient, session_id: str, invocation_id: str
) -> dict[str, Any] | None:
    """Fetch and parse the document stored in one invocation.

    Args:
        client: bedrock-agent-runtime client.
        session_id: Session holding the invocation.
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
    return _parse_document(
        "".join(
            block["text"]
            for detail in details
            for block in detail["invocationStep"]["payload"]["contentBlocks"]
            if "text" in block
        )
    )


def _parse_document(text: str) -> dict[str, Any] | None:
    """Parse a document's concatenated invocation step text blocks.

    Args:
        text: The joined text blocks of one invocation.

    Returns:
        The parsed document, or None when the text is empty or does not parse
        as a JSON object.
    """
    if not text:
        return None
    try:
        document = from_json(text)
    except ValueError:
        return None
    return document if isinstance(document, dict) else None


async def load_documents(
    client: AgentsforBedrockRuntimeClient,
    session_id: str,
    *,
    concurrency: int,
    scan_limit: int,
) -> list[dict[str, Any]]:
    """Read every document stored in a session, oldest write first.

    One session-wide step listing replaces a per-invocation one, so the read
    costs one listing plus one bounded-concurrency fetch per step. Documents
    that do not parse (an interrupted or truncated write) are skipped.

    Args:
        client: bedrock-agent-runtime client.
        session_id: Session to read.
        concurrency: Maximum concurrent invocation step fetches.
        scan_limit: Maximum invocation steps read before the scan stops.

    Returns:
        The parsed documents, in the order they were written.
    """
    steps: list[InvocationStepSummaryTypeDef] = []
    token: str | None = None
    while len(steps) < scan_limit:
        page = await client.list_invocation_steps(
            sessionIdentifier=session_id,
            **({"nextToken": token} if token else {}),  # type: ignore[arg-type]
        )
        steps.extend(page.get("invocationStepSummaries", []))
        token = page.get("nextToken")
        if not token:
            break
    grouped: dict[str, list[InvocationStepSummaryTypeDef]] = {}
    for step in steps[:scan_limit]:
        grouped.setdefault(step["invocationId"], []).append(step)
    for group in grouped.values():
        group.sort(key=itemgetter("invocationStepTime"))
    ordered = sorted(grouped.values(), key=lambda group: group[0]["invocationStepTime"])
    semaphore = Semaphore(concurrency)

    async def _text(step: InvocationStepSummaryTypeDef) -> str:
        """Fetch one invocation step's text blocks under the read bound."""
        async with semaphore:
            detail = await client.get_invocation_step(
                sessionIdentifier=session_id,
                invocationIdentifier=step["invocationId"],
                invocationStepId=step["invocationStepId"],
            )
        return "".join(
            block["text"]
            for block in detail["invocationStep"]["payload"]["contentBlocks"]
            if "text" in block
        )

    texts = iter(await gather(*(_text(step) for group in ordered for step in group)))
    documents = []
    for group in ordered:
        document = _parse_document("".join(next(texts) for _ in group))
        if document is not None:
            documents.append(document)
    return documents


async def load_latest_document(
    client: AgentsforBedrockRuntimeClient, session_id: str
) -> dict[str, Any] | None:
    """Read the newest fully written document stored in a session.

    Invocations are tried newest first, and the first one with a fully
    written, parseable document is returned. This tolerates a write
    interrupted between creating the latest invocation and writing its steps,
    or a truncated read racing a concurrent write, by falling back to the
    last-good invocation.

    Args:
        client: bedrock-agent-runtime client.
        session_id: Session to read.

    Returns:
        The persisted document, or None when the session does not exist,
        holds no parseable document, or the identifier cannot name a session.
    """
    try:
        summaries = await list_session_invocations(client, session_id)
        for summary in sorted(summaries, key=itemgetter("createdAt"), reverse=True):
            document = await load_invocation_document(
                client, session_id, summary["invocationId"]
            )
            if document is not None:
                return document
    except ClientError as exc:
        if is_unknown_identifier(exc):
            return None
        raise
    return None


async def end_and_delete_session(
    client: AgentsforBedrockRuntimeClient, session_id: str
) -> None:
    """End then delete a session, tolerating one that is already ended.

    Args:
        client: bedrock-agent-runtime client.
        session_id: Session to delete.
    """
    try:
        # Sessions must be ended before deletion; tolerate already-ended
        # (state errors defer to the delete call, which surfaces real issues).
        await client.end_session(sessionIdentifier=session_id)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in _END_TOLERATED_CODES:
            raise
    await client.delete_session(sessionIdentifier=session_id)


async def scan_sessions_by_tag(
    client: AgentsforBedrockRuntimeClient,
    tag_value: str,
    tag_of: Callable[[SessionSummaryTypeDef], Awaitable[str | None]],
    *,
    page_size: int,
    scan_limit: int,
    concurrency: int,
) -> list[tuple[str, datetime]]:
    """List the sessions whose ownership tag equals *tag_value*.

    Session listing carries no tags, so each summary's tag is resolved by
    *tag_of*, concurrently within a page and bounded by *concurrency*.

    Args:
        client: bedrock-agent-runtime client.
        tag_value: Ownership tag value a session must carry to be returned.
        tag_of: Resolves one session summary's ownership tag value, or None
            when the session is untagged.
        page_size: Maximum sessions per listing page.
        scan_limit: Maximum sessions scanned before the listing stops.
        concurrency: Maximum concurrent tag lookups within a page.

    Returns:
        Unordered ``(session ID, creation time)`` pairs.
    """
    semaphore = Semaphore(concurrency)

    async def _tag(summary: SessionSummaryTypeDef) -> str | None:
        """Resolve one session's ownership tag under the concurrency bound."""
        async with semaphore:
            return await tag_of(summary)

    sessions: list[tuple[str, datetime]] = []
    scanned = 0
    token: str | None = None
    while scanned < scan_limit:
        page = await client.list_sessions(
            maxResults=min(page_size, scan_limit - scanned),
            **({"nextToken": token} if token else {}),
        )
        summaries = page.get("sessionSummaries", [])
        scanned += len(summaries)
        tags = await gather(*map(_tag, summaries))
        sessions.extend(
            (summary["sessionId"], summary["createdAt"])
            for summary, tag in zip(summaries, tags, strict=True)
            if tag == tag_value
        )
        token = page.get("nextToken")
        if not token:
            break
    return sessions
