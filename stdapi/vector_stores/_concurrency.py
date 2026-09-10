"""Bounded fan-out, shared by the engine and every backend."""

from asyncio import Semaphore, ensure_future, gather
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Coroutine, Sequence
    from typing import Any


async def gather_bounded[ResultT](
    calls: Sequence[Coroutine[Any, Any, ResultT]], wave: int
) -> list[ResultT]:
    """Await *calls* concurrently, at most *wave* of them at once, keeping order.

    The bound is a ceiling on the calls in flight rather than a barrier between
    batches, so a slow call delays itself and not the ones queued behind it.

    The first failure answers the request, so the calls beside it are cancelled
    and the ones that never started are closed: one left running bills the
    account for a result nobody reads.

    Args:
        calls: The coroutines to run.
        wave: How many of them may be in flight at once; at least one.

    Returns:
        Their results, in the order they were given.

    Raises:
        BaseException: Whatever the first failing call raised, unwrapped, so a
            caller mapping a backend error by its type still recognises it.
    """
    semaphore = Semaphore(wave)

    async def _bounded(call: Coroutine[Any, Any, ResultT]) -> ResultT:
        """Await one call while holding a slot of the bound.

        Returns:
            Whatever the call returned.
        """
        try:
            async with semaphore:
                return await call
        finally:
            # Cancelled before its slot came up: closing it here keeps it from
            # being reported as a coroutine that was never awaited.
            call.close()

    tasks = [ensure_future(_bounded(call)) for call in calls]
    try:
        return await gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        # Await cancellation so asyncio doesn't log unretrieved exceptions at GC.
        await gather(*tasks, return_exceptions=True)
        raise
