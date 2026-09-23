"""Reading a model stream's first event before its response is committed.

Some models refuse a request only on their stream's first event, after the
call opened it. An adapter reads the backend's first event before emitting its
own, and the route reads the adapter's first event before returning the
response, so such a refusal can still be answered with a status, or retried,
before any byte is sent.
"""

from typing import TYPE_CHECKING, Any

from sse_starlette import EventSourceResponse

from stdapi.models.chat._adapters._responses_context import (
    ContextLengthExceededError,
    collect_stream_open_errors,
    record_stream_open_error,
)
from stdapi.monitoring import context_length_error, log_error_details

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterable,
        AsyncIterator,
        Awaitable,
        Callable,
        Sequence,
    )

    from stdapi.api_errors import ApiError


async def primed[T](stream: AsyncIterator[T]) -> AsyncGenerator[T]:
    """Read a stream's first event now, and replay the stream from it.

    A backend refusing the request on its first event (an oversized input, on
    some models) is then reported to the caller that opened the stream, which
    can still answer or retry before the response is announced; the client
    receives the usual failure events when the replay reaches it.

    Args:
        stream: The backend event stream.

    Returns:
        The stream, first event included, raising any error the first read did.
    """
    events = aiter(stream)
    try:
        first = await anext(events)
    except StopAsyncIteration:
        return replay_stream((), events)
    except Exception as exc:  # noqa: BLE001 - raised again by the replay
        record_stream_open_error(exc)
        return replay_stream((), events, exc)
    return replay_stream((first,), events)


async def replay_stream[T](
    first: Sequence[T], rest: AsyncIterator[T], error: Exception | None = None
) -> AsyncGenerator[T]:
    """Yield events read ahead, then the rest of their stream, closing it after.

    Args:
        first: Events already read.
        rest: The stream they were read from.
        error: The error reading ahead raised, raised here after *first*.

    Yields:
        Every event, in order.

    Raises:
        Exception: The error reading ahead raised.
    """
    try:
        for item in first:
            yield item
        if error is not None:
            raise error
        async for item in rest:
            yield item
    finally:
        await close_stream(rest)


async def close_stream(stream: AsyncIterable[Any]) -> None:
    """Close a stream, when it can be closed.

    Args:
        stream: The stream.
    """
    if (aclose := getattr(stream, "aclose", None)) is not None:
        await aclose()


async def open_peeked[R](
    call: Awaitable[R], refusal: Callable[[BaseException], BaseException | None]
) -> R:
    """Await a model call, reading its stream's first event before returning.

    That event waits only for the backend's first one, which the client waits
    for anyway.

    Args:
        call: The model call, answering a response or its stream.
        refusal: The error to raise instead of streaming, for an error the
            stream raised on its first event, or None to stream that error.

    Returns:
        The response, or its stream, the first event replayed.

    Raises:
        BaseException: Whatever the call raised, or the error ``refusal``
            returned, the stream closed.
    """
    with collect_stream_open_errors() as errors:
        result = await call
        if not isinstance(result, EventSourceResponse):
            return result
        body = aiter(result.body_iterator)
        first = await anext(body, None)
    if (error := next(filter(None, map(refusal, errors)), None)) is not None:
        await close_stream(body)
        raise error
    result.body_iterator = replay_stream(() if first is None else (first,), body)
    return result


def context_refusal(error: BaseException) -> ApiError | None:
    """Return the calling API's refusal of an input the context window cannot hold.

    Args:
        error: An error a stream raised on its first event.

    Returns:
        The API's error, as the unstreamed request gets it, or None when
        *error* is not a context-window refusal.
    """
    if isinstance(error, ContextLengthExceededError):
        return error
    if (refused := context_length_error(error)) is not None:
        # The backend's own wording names internals: logged, not sent.
        log_error_details(str(error), status=refused.status)
    return refused
