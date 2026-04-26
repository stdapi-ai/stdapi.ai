"""Request cleanup utilities."""

from asyncio import gather
from contextvars import ContextVar
from typing import TYPE_CHECKING

from stdapi.monitoring import log_background_event

if TYPE_CHECKING:
    from collections.abc import Awaitable

#: Pending cleanup tasks for the current request.
CLEANUPS: ContextVar[list[Awaitable[None]]] = ContextVar("cleanups")


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
