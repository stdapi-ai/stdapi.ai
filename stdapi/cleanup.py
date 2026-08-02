"""Request cleanup utilities."""

from asyncio import Task, create_task, gather
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


def schedule_cleanup(*tasks: Awaitable[None]) -> None:
    """Register an async cleanup coroutine to run after the response is sent.

    Args:
        *tasks: Awaitable coroutine(s) to execute during cleanup.
    """
    CLEANUPS.get().extend(tasks)


async def run_scheduled_cleanups(request_id: str) -> None:
    """Execute all registered cleanup coroutines.

    Args:
        request_id: Request identifier to associate the cleanup log event with.
    """
    with log_background_event("cleanup", request_id):
        await gather(*CLEANUPS.get())


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
