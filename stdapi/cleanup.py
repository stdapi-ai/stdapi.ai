"""Request cleanup utilities."""

from asyncio import gather
from contextvars import ContextVar
from typing import TYPE_CHECKING

from stdapi.monitoring import REQUEST_ID, log_background_event

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


async def run_scheduled_cleanups() -> None:
    """Execute all registered cleanup coroutines."""
    with log_background_event("cleanup", REQUEST_ID.get()):
        await gather(*CLEANUPS.get())
