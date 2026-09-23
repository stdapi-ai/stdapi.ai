"""Synchronization primitives that stay usable after the loop that created them closes.

``asyncio.Lock``/``Semaphore`` only bind to a loop once something actually has to
*wait* for them (``asyncio.locks.Lock.acquire``): an always-uncontended acquire never
calls ``_get_loop()`` and never binds, so reproducing issue #299 needs genuine
contention -- one coroutine holding the primitive while a second is forced to wait for
it -- not just a sequential acquire/release. Every module-level instance in this
codebase lives for the whole process, so a test suite that gives each test its own loop
(as ``TestClient`` does) can hit this the moment two tests in the same worker contend
for the same primitive.

Ref: https://github.com/stdapi-ai/stdapi.ai/issues/299
     stdapi/utils.py:LoopBoundLock
     stdapi/utils.py:LoopBoundSemaphore
"""

import asyncio
import threading

import pytest

from stdapi.utils import LoopBoundLock, LoopBoundSemaphore


async def _contend_for_lock(lock: asyncio.Lock | LoopBoundLock) -> None:
    """Force a second coroutine to wait for `lock`, which is what binds it to a loop."""
    holder_ready = asyncio.Event()

    async def _holder() -> None:
        async with lock:
            holder_ready.set()
            await asyncio.sleep(0.05)

    async def _waiter() -> None:
        await holder_ready.wait()
        async with lock:
            pass

    await asyncio.gather(_holder(), _waiter())


async def _contend_for_semaphore(
    semaphore: asyncio.Semaphore | LoopBoundSemaphore,
) -> None:
    """Force a second coroutine to wait for `semaphore`'s single permit."""
    holder_ready = asyncio.Event()

    async def _holder() -> None:
        await semaphore.acquire()
        try:
            holder_ready.set()
            await asyncio.sleep(0.05)
        finally:
            semaphore.release()

    async def _waiter() -> None:
        await holder_ready.wait()
        await semaphore.acquire()
        semaphore.release()

    await asyncio.gather(_holder(), _waiter())


def test_a_bare_lock_cannot_survive_its_loop_closing_once_contended() -> None:
    """Baseline: a plain, once-contended ``asyncio.Lock`` is the defect issue #299 reports.

    Proves the reproduction below is real -- the same sequence against a bare lock
    fails first, so the passing test after it is not a no-op.

    Ref: https://github.com/stdapi-ai/stdapi.ai/issues/299
    """
    lock = asyncio.Lock()

    asyncio.run(_contend_for_lock(lock))  # binds the lock to this run's loop
    with pytest.raises(RuntimeError, match="bound to a different event loop"):
        asyncio.run(_contend_for_lock(lock))


def test_loop_bound_lock_survives_its_loop_closing_once_contended() -> None:
    """A ``LoopBoundLock`` is rebound instead of raising when the loop changes.

    Same two-loop, contended sequence as the bare-lock baseline above, on the
    wrapper the model cache and the route listing caches use; it must not raise.

    Ref: stdapi/utils.py:LoopBoundLock
    """
    lock = LoopBoundLock()

    asyncio.run(_contend_for_lock(lock))
    asyncio.run(_contend_for_lock(lock))


def test_loop_bound_lock_still_excludes_concurrent_holders_on_one_loop() -> None:
    """Recreating the lock on a loop change must not weaken exclusion within one loop.

    The production concern named in issue #299: a torn-down loop cannot poison later
    requests, but requests sharing a live loop must still serialize.

    Ref: stdapi/utils.py:LoopBoundLock
    """
    lock = LoopBoundLock()
    order: list[str] = []

    async def _holder() -> None:
        async with lock:
            order.append("holder-enter")
            await asyncio.sleep(0.01)
            order.append("holder-exit")

    async def _waiter() -> None:
        await asyncio.sleep(0)  # let the holder acquire first
        async with lock:
            order.append("waiter-enter")

    async def _run() -> None:
        await asyncio.gather(_holder(), _waiter())

    asyncio.run(_run())

    assert order == ["holder-enter", "holder-exit", "waiter-enter"]


def test_loop_bound_lock_release_only_affects_its_own_loops_primitive() -> None:
    """A holder on one loop is unaffected when a second, live loop rebinds it.

    Two loops run at once in different threads, as a module-scoped pytest-asyncio
    loop and a per-test loop do. Loop 1 acquires and holds; loop 2 then rebinds,
    acquiring and releasing its own primitive while loop 1 still holds. Loop 1's
    own release must reach the primitive it acquired, not whichever one is
    current by then -- a release keyed off the current primitive would raise
    when loop 1 tries to release what loop 2 already did.

    Ref: stdapi/utils.py:LoopBoundLock
    """
    lock = LoopBoundLock()
    loop1_acquired = threading.Event()
    loop1_may_release = threading.Event()
    loop1_error: BaseException | None = None

    def _run_loop1() -> None:
        nonlocal loop1_error

        async def _hold() -> None:
            async with lock:
                loop1_acquired.set()
                await asyncio.get_running_loop().run_in_executor(
                    None, loop1_may_release.wait
                )

        try:
            asyncio.run(_hold())
        except BaseException as error:  # noqa: BLE001
            loop1_error = error

    thread = threading.Thread(target=_run_loop1)
    thread.start()
    assert loop1_acquired.wait(timeout=5), "loop 1 never acquired the lock"

    async def _rebind_and_release_on_loop2() -> None:
        async with lock:
            pass

    asyncio.run(_rebind_and_release_on_loop2())  # rebinds to its own primitive

    loop1_may_release.set()
    thread.join(timeout=5)

    assert loop1_error is None, f"loop 1's own release failed: {loop1_error!r}"


def test_a_bare_semaphore_cannot_survive_its_loop_closing_once_contended() -> None:
    """Baseline: a plain, once-contended ``asyncio.Semaphore`` has the same defect.

    Ref: https://github.com/stdapi-ai/stdapi.ai/issues/299
    """
    semaphore = asyncio.Semaphore(1)

    asyncio.run(_contend_for_semaphore(semaphore))
    with pytest.raises(RuntimeError, match="bound to a different event loop"):
        asyncio.run(_contend_for_semaphore(semaphore))


def test_loop_bound_semaphore_survives_its_loop_closing_once_contended() -> None:
    """A ``LoopBoundSemaphore`` is rebound instead of raising when the loop changes.

    Covers the call sites that acquire and release explicitly instead of using
    ``async with`` (``stdapi.batches``, ``stdapi.vector_stores.engine``).

    Ref: stdapi/utils.py:LoopBoundSemaphore
    """
    semaphore = LoopBoundSemaphore(1)

    asyncio.run(_contend_for_semaphore(semaphore))
    asyncio.run(_contend_for_semaphore(semaphore))


def test_loop_bound_semaphore_still_bounds_concurrency_on_one_loop() -> None:
    """Recreating the semaphore on a loop change must not widen its permit count.

    Ref: stdapi/utils.py:LoopBoundSemaphore
    """
    semaphore = LoopBoundSemaphore(1)
    in_flight = 0
    max_in_flight = 0

    async def _slot() -> None:
        nonlocal in_flight, max_in_flight
        await semaphore.acquire()
        try:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
        finally:
            in_flight -= 1
            semaphore.release()

    async def _run() -> None:
        await asyncio.gather(_slot(), _slot(), _slot())

    asyncio.run(_run())

    assert max_in_flight == 1


def test_loop_bound_semaphore_release_only_affects_its_own_loops_primitive() -> None:
    """A permit held on one loop is unaffected when a second, live loop rebinds it.

    Same two-thread sequence as the lock test above. A release keyed off the
    current primitive would land on loop 2's semaphore instead of loop 1's,
    silently inflating loop 2's permit count beyond its capacity instead of
    raising -- the corruption is quieter than the lock's, not absent.

    Ref: stdapi/utils.py:LoopBoundSemaphore
    """
    semaphore = LoopBoundSemaphore(1)
    loop1_acquired = threading.Event()
    loop1_may_release = threading.Event()
    loop1_primitive: asyncio.Semaphore | None = None

    def _run_loop1() -> None:
        nonlocal loop1_primitive

        async def _hold() -> None:
            nonlocal loop1_primitive
            await semaphore.acquire()
            loop1_primitive = semaphore._semaphores[asyncio.get_running_loop()]  # noqa: SLF001
            loop1_acquired.set()
            await asyncio.get_running_loop().run_in_executor(
                None, loop1_may_release.wait
            )
            semaphore.release()

        asyncio.run(_hold())

    thread = threading.Thread(target=_run_loop1)
    thread.start()
    assert loop1_acquired.wait(timeout=5), "loop 1 never acquired a permit"

    async def _rebind_and_release_on_loop2() -> None:
        await semaphore.acquire()
        semaphore.release()

    asyncio.run(_rebind_and_release_on_loop2())  # rebinds to its own primitive

    loop1_may_release.set()
    thread.join(timeout=5)

    assert loop1_primitive is not None
    assert loop1_primitive._value == 1, (  # noqa: SLF001
        "loop 1's own permit must be back at full capacity, not another loop's"
    )
