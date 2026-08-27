"""What an Ollama client is left holding when a stream fails after its first line.

Once the headers are out there is no status code left to send, so a failure has
to arrive inside the stream. The client must end up with a terminal
``{"error": ...}`` line rather than a truncated body and a 200 in the access
log -- and on a backend failure that line must carry nothing from the backend:
Ollama's envelope is a single free-text field, which makes it the easiest one to
stuff a raw AWS message into.

The failure is injected rather than provoked: an error arriving *after* a
partial payload is the case that breaks real clients, and nothing else produces
it on demand. That is also why the guard is driven directly here instead of
through a request -- but the last test closes the loop, feeding the exact bytes
the guard emits to the official ``ollama`` client and asserting it raises.

Ref: https://docs.ollama.com/openapi.yaml (ErrorResponse)
     stdapi/monitoring.py:guard_ndjson_stream_errors
"""

from typing import TYPE_CHECKING, Any

import httpx
import ollama
import pytest
from botocore.exceptions import ClientError
from starlette.requests import Request

from stdapi.api_errors import ApiError
from stdapi.api_providers.ollama import NDJSON_MEDIA_TYPE, TAG_OLLAMA
from stdapi.monitoring import REQUEST, guard_ndjson_stream_errors

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

pytestmark = pytest.mark.gateway(
    "Exercises the gateway's own mid-stream error guard, which no upstream "
    "endpoint can be made to trigger on demand"
)

#: A backend failure carrying exactly the detail that must not reach a client.
_AWS_ERROR = ClientError(
    {
        "Error": {
            "Code": "InternalServerException",
            "Message": "anthropic.claude-x is unavailable in us-east-1",
        },
        "ResponseMetadata": {
            "RequestId": "d1e2f3a4-5678-90ab-cdef-1234567890ab",
            "HTTPStatusCode": 500,
            "HTTPHeaders": {},
            "HostId": "",
            "RetryAttempts": 0,
        },
    },
    "ConverseStream",
)


#: Message of the injected mid-stream API failure.
_UNAVAILABLE = "The model is not available."

#: Host the mocked transport of the client-level test answers for.
_MOCK_HOST = "http://ollama.test"


class _OllamaRoute:
    """Stand-in for the matched route, carrying only the tag the envelope keys on."""

    tags = (TAG_OLLAMA,)


@pytest.fixture
def ollama_request(request_log: dict[str, Any]) -> Generator[None]:
    """Bind a request whose matched route is an Ollama one.

    The error envelope is chosen from the route's tags, so a stream error
    outside a request context would be shaped by the default formatter instead.

    Args:
        request_log: Bound request log, which the error path writes into.

    Yields:
        None.
    """
    assert request_log is not None
    connection = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "headers": [],
            "route": _OllamaRoute(),
        }
    )
    token = REQUEST.set(connection)
    try:
        yield
    finally:
        REQUEST.reset(token)


async def _lines(stream: AsyncGenerator[dict[str, Any]]) -> list[bytes]:
    """Collect the guarded stream's output.

    Args:
        stream: The source objects.

    Returns:
        The emitted lines.
    """
    return [line async for line in guard_ndjson_stream_errors(stream)]


async def _partial_then_api_error() -> AsyncGenerator[dict[str, Any]]:
    """Emit one usable chat event, then fail the way a backend does.

    Yields:
        The single event preceding the failure.
    """
    yield {
        "model": "m",
        "created_at": "2026-01-02T03:04:05+00:00",
        "message": {"role": "assistant", "content": "par"},
        "done": False,
    }
    raise ApiError(_UNAVAILABLE, status=503)


@pytest.mark.usefixtures("ollama_request")
async def test_a_failure_after_the_first_line_becomes_a_terminal_error_line() -> None:
    """The client keeps the partial answer and then reads one error object.

    Ref: stdapi/monitoring.py:guard_ndjson_stream_errors
    """
    lines = await _lines(_partial_then_api_error())
    assert len(lines) == 2
    assert lines[0].endswith(b"\n")
    assert b'"content":"par"' in lines[0]
    assert lines[1] == b'{"error":"%s"}\n' % _UNAVAILABLE.encode()


@pytest.mark.usefixtures("ollama_request")
async def test_the_error_line_leaks_no_backend_detail() -> None:
    """No AWS code, message or request ID reaches the caller.

    Ref: stdapi/monitoring.py:guard_ndjson_stream_errors
    """

    async def failing() -> AsyncGenerator[dict[str, Any]]:
        """Fail with a backend error after a partial payload."""
        yield {"done": False}
        raise _AWS_ERROR

    line = (await _lines(failing()))[-1].decode()
    assert "InternalServerException" not in line
    assert "anthropic.claude-x" not in line
    assert "us-east-1" not in line
    assert "d1e2f3a4" not in line
    assert "Retry the request" in line


@pytest.mark.usefixtures("ollama_request")
async def test_an_unexpected_failure_still_closes_the_stream() -> None:
    """A bug in the translation is an error line, never a truncated body.

    Ref: stdapi/monitoring.py:guard_ndjson_stream_errors
    """

    async def failing() -> AsyncGenerator[dict[str, Any]]:
        """Fail with something no handler anticipated."""
        yield {"done": False}
        msg = "boom"
        raise RuntimeError(msg)

    lines = await _lines(failing())
    assert lines[-1] == b'{"error":"Internal Server Error"}\n'
    assert b"boom" not in lines[-1]


@pytest.mark.usefixtures("ollama_request")
async def test_the_official_client_raises_on_the_terminal_error_line() -> None:
    """The bytes the guard emits are what makes the real client raise.

    The line only matters if a client acts on it, so the guard's own output is
    replayed to ``ollama.Client`` through a mocked transport: it must yield the
    partial answer, then raise ``ResponseError`` carrying the message and
    nothing else. A live stream cannot be made to fail mid-body on demand, which
    is why the transport is mocked rather than the gateway called.

    Ref: https://github.com/ollama/ollama-python (Client._request)
    """
    body = b"".join(await _lines(_partial_then_api_error()))

    def handler(_: httpx.Request) -> httpx.Response:
        """Answer every request with the guarded stream's own bytes."""
        return httpx.Response(
            200, content=body, headers={"content-type": NDJSON_MEDIA_TYPE}
        )

    client = ollama.Client(host=_MOCK_HOST, transport=httpx.MockTransport(handler))
    stream = client.chat(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    assert next(stream).message.content == "par"
    with pytest.raises(ollama.ResponseError) as raised:
        next(stream)
    assert raised.value.error == _UNAVAILABLE
