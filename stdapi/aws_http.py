"""HTTP transport shared by the AWS endpoints botocore does not cover.

Amazon Bedrock Mantle and Amazon SageMaker AI both serve OpenAI-compatible
APIs over plain HTTPS, and neither route exists in any botocore service model.
Both authenticate with a short-term bearer token that is the base64 of a SigV4
**presigned** ``Action=CallWithBearerToken`` request, built entirely
client-side from the shared credential chain; the two differ only by the
signing host, the SigV4 signing name and the token prefix.

So the token derivation, the pooled client session and the server-sent event
reader live here, and each transport module keeps only what is genuinely its
own: its endpoint URL, its request paths and its error mapping.
"""

from base64 import b64encode
from typing import TYPE_CHECKING, Final

from aiohttp import ClientError as AiohttpClientError
from aiohttp import ClientSession, ClientTimeout
from aiohttp.http_exceptions import HttpProcessingError
from botocore.auth import SigV4QueryAuth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

from stdapi import server
from stdapi.api_errors import ApiError
from stdapi.config import AWS_SESSION, SETTINGS

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from aiohttp import ClientResponse
    from types_aiobotocore_bedrock.literals import RegionName

#: Parsed server-sent event: (event name or None, raw data payload).
type SseEvent = tuple[str | None, str]

#: Presigned token validity in seconds (kept short; regeneration is local HMAC work).
TOKEN_EXPIRY: Final = 3600

#: Cached token refresh interval in seconds (below temporary-credential lifetimes).
TOKEN_TTL: Final = 300

#: Read buffer for a session relaying SSE: one event can carry a whole response JSON.
_READ_BUFSIZE: Final = 2**22


def _no_credentials_error(feature: str) -> ApiError:
    """Build the error raised when the credential chain is empty.

    Which backend the token was for is the operator's diagnosis and goes to the
    log; the caller reads a message naming no AWS service.

    Args:
        feature: Backend the token was being minted for.

    Returns:
        The error to raise.
    """
    # Imported here: stdapi.monitoring imports this module transitively.
    from stdapi.monitoring import log_error_details  # noqa: PLC0415

    log_error_details(
        f"No AWS credentials available to authenticate with {feature}: the "
        "server's credential chain resolved nothing.",
        level="warning",
    )
    return ApiError("No AWS credentials available on this server.", status=500)


async def presigned_bearer_token(
    region: RegionName, *, host: str, service: str, prefix: str, feature: str
) -> str:
    """Presign a short-term bearer token for an AWS HTTP endpoint.

    The token is derived locally from the shared botocore credential chain (no
    static secret is involved) and carries the same authority as the signing
    role, which is why the matching IAM policy bounds what it can reach.

    Args:
        region: AWS region the token is scoped to.
        host: SigV4 signing host the request is presigned against.
        service: SigV4 signing name.
        prefix: Token prefix defined by the endpoint's API-key format.
        feature: Backend the token is for, named to the operator only.

    Returns:
        Bearer token string.

    Raises:
        ApiError: When no AWS credentials are available.
    """
    if (credentials := await AWS_SESSION.get_credentials()) is None:
        raise _no_credentials_error(feature)
    frozen = await credentials.get_frozen_credentials()
    if not (frozen.access_key and frozen.secret_key):  # pragma: no cover
        raise _no_credentials_error(feature)
    request = AWSRequest(
        method="POST",
        url=f"https://{host}/",
        headers={"host": host},
        params={"Action": "CallWithBearerToken"},
    )
    SigV4QueryAuth(
        Credentials(frozen.access_key, frozen.secret_key, frozen.token),
        service,
        region,
        expires=TOKEN_EXPIRY,
    ).add_auth(request)
    return prefix + b64encode(
        f"{str(request.url).removeprefix('https://')}&Version=1".encode()
    ).decode("ascii")


def new_http_session() -> ClientSession:
    """Open a client session toward an AWS HTTP endpoint outside botocore.

    Returns:
        A session configured like the AWS SDK clients: the same connect and
        response budgets, the same proxy environment, and a read buffer sized
        for server-sent events carrying a whole response in one line.
    """
    return ClientSession(
        headers=server.HTTP_CLIENT_HEADERS,
        timeout=ClientTimeout(
            total=None,
            connect=SETTINGS.aws_connect_timeout,
            sock_read=SETTINGS.ai_response_timeout,
        ),
        read_bufsize=_READ_BUFSIZE,
        # Same proxy environment the AWS SDK already honours unconditionally.
        trust_env=True,
    )


async def iter_sse(
    response: ClientResponse, interrupted: str, error_class: type[ApiError]
) -> AsyncGenerator[SseEvent]:
    """Yield parsed server-sent events from an open response.

    Args:
        response: Open streaming HTTP response.
        interrupted: Message raised when the connection drops mid-stream.
        error_class: Error class the interruption is raised as.

    Yields:
        ``(event name or None, raw data)`` tuples, terminating before any
        ``[DONE]`` sentinel.

    Raises:
        ApiError: When the connection drops mid-stream.
    """
    event: str | None = None
    data: list[str] = []
    try:
        async with response:
            async for raw_line in response.content:
                match raw_line.decode().rstrip("\r\n"):
                    case "":
                        if data and (joined := "\n".join(data)) != "[DONE]":
                            yield event, joined
                        event, data = None, []
                    case line if line.startswith("data:"):
                        data.append(line[5:].lstrip(" "))
                    case line if line.startswith("event:"):
                        event = line[6:].lstrip(" ")
                    case _:
                        pass  # Comments and unknown fields are ignored.
        if data and (joined := "\n".join(data)) != "[DONE]":  # pragma: no cover
            yield event, joined
    except (
        AiohttpClientError,
        TimeoutError,
        HttpProcessingError,
        UnicodeDecodeError,
    ) as error:
        # HttpProcessingError covers LineTooLong, which aiohttp raises outside
        # its ClientError hierarchy for oversized SSE lines; UnicodeDecodeError
        # covers a non-UTF-8 line mid-stream.
        raise error_class(interrupted, status=502) from error
