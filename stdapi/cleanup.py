"""Request cleanup utilities, and the shutdown drain of detached background work."""

from asyncio import Task, create_task, gather, wait
from contextlib import suppress
from contextvars import ContextVar
from typing import TYPE_CHECKING, Final

from stdapi.monitoring import log_background_event

if TYPE_CHECKING:
    from collections.abc import Awaitable

#: Pending cleanup tasks for the current request.
CLEANUPS: ContextVar[list[Awaitable[None]]] = ContextVar("cleanups")

#: Strong references to detached cleanup tasks, held until completion.
_DETACHED_TASKS: Final[set[Task[None]]] = set()

#: Seconds a task cancelled at the drain deadline gets to unwind.
_SETTLE_TIMEOUT: Final = 1.0


def schedule_cleanup(*tasks: Awaitable[None]) -> None:
    """Register an async cleanup coroutine to run after the response is sent.

    Args:
        *tasks: Awaitable coroutine(s) to execute during cleanup.
    """
    CLEANUPS.get().extend(tasks)


async def run_scheduled_cleanups(request_id: str) -> None:
    """Execute all registered cleanup coroutines.

    The pending list is emptied before the first await, so a second run finds
    nothing left to await rather than re-awaiting a consumed coroutine, and a
    request that scheduled nothing costs no background event at all.

    Args:
        request_id: Request identifier to associate the cleanup log event with.
    """
    pending = CLEANUPS.get()
    if not pending:
        return
    tasks = tuple(pending)
    pending.clear()
    with log_background_event("cleanup", request_id):
        await gather(*tasks)


def run_cleanups_detached(request_id: str) -> None:
    """Run pending cleanups in a task that survives request cancellation.

    Used on request paths that never send a response body (unhandled errors,
    client disconnects), where the usual post-response background task is
    never attached and the request scope may already be cancelled.

    Args:
        request_id: Request identifier to associate the cleanup log event with.
    """
    if CLEANUPS.get():
        task = create_task(_run_cleanups_logged(request_id))
        _DETACHED_TASKS.add(task)
        task.add_done_callback(_DETACHED_TASKS.discard)


async def _run_cleanups_logged(request_id: str) -> None:
    """Await scheduled cleanups, discarding errors already logged as critical.

    Args:
        request_id: Request identifier to associate the cleanup log event with.
    """
    with suppress(Exception):
        await run_scheduled_cleanups(request_id)


async def drain_tasks(tasks: set[Task[None]], timeout: float) -> int:  # noqa: ASYNC109 -- the deadline must return a count, not cancel the caller
    """Await a registry of detached tasks, settling what the deadline leaves.

    Best effort by construction: the process may be killed before this returns,
    so nothing whose correctness matters may depend on it. What is still running
    at the deadline is settled rather than abandoned — cancelled, given a moment
    to unwind, and its outcome retrieved — so no task disappears carrying a
    result nobody read.

    Args:
        tasks: Registry of detached tasks, drained as it stands on entry.
        timeout: Seconds allowed before the unfinished tasks are cancelled;
            ``0`` cancels them immediately.

    Returns:
        Number of tasks that had not finished at the deadline.
    """
    if not (pending := tuple(tasks)):
        return 0
    unfinished = (await wait(pending, timeout=timeout))[1]
    if unfinished:
        for task in unfinished:
            task.cancel()
        await wait(unfinished, timeout=_SETTLE_TIMEOUT)
    for task in pending:
        # Read so a task dropped here is never reported as an unhandled error.
        if task.done() and not task.cancelled():
            task.exception()
    return len(unfinished)


async def drain_cleanups(timeout: float) -> int:  # noqa: ASYNC109 -- shared drain contract
    """Await the request cleanups still running detached from their request.

    Args:
        timeout: Seconds allowed before the unfinished cleanups are cancelled.

    Returns:
        Number of cleanups that had not finished at the deadline.
    """
    return await drain_tasks(_DETACHED_TASKS, timeout)
