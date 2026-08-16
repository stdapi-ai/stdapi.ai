"""Bounded fan-out, shared by the engine and every backend."""

from asyncio import gather
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence


async def gather_bounded[ResultT](
    awaitables: Sequence[Awaitable[ResultT]], wave: int
) -> list[ResultT]:
    """Await *awaitables* concurrently, in waves of *wave*, keeping their order.

    Args:
        awaitables: The coroutines to run.
        wave: How many of them may be in flight at once.

    Returns:
        Their results, in the order they were given.
    """
    results: list[ResultT] = []
    for start in range(0, len(awaitables), wave):
        results.extend(await gather(*awaitables[start : start + wave]))
    return results
