"""The model-cache lock and its sibling module-level primitives survive a closed loop.

Every module-level ``Lock``/``Semaphore`` in the codebase is a process-lifetime
singleton, so a test suite that gives each test its own loop (as ``TestClient``
does) can bind one to a loop that later closes. Reproducing this needs genuine
contention: an always-uncontended ``asyncio.Lock``/``Semaphore`` never binds to a
loop at all (only a waiter does), which is why each helper below forces a second
coroutine to wait for the primitive.

Ref: https://github.com/stdapi-ai/stdapi.ai/issues/299
     stdapi/utils.py:LoopBoundLock
     stdapi/utils.py:LoopBoundSemaphore
"""

import asyncio
from typing import TYPE_CHECKING

import stdapi.models
from stdapi import batches
from stdapi.routes import ollama_models, openai_models
from stdapi.vector_stores import engine

if TYPE_CHECKING:
    from stdapi.utils import LoopBoundLock, LoopBoundSemaphore


async def _contend_for_lock(lock: LoopBoundLock) -> None:
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


async def _contend_for_semaphore(semaphore: LoopBoundSemaphore, slots: int) -> None:
    """Drain all `slots` permits, then force a second coroutine to wait for one."""
    holder_ready = asyncio.Event()

    async def _holder() -> None:
        for _ in range(slots):
            await semaphore.acquire()
        try:
            holder_ready.set()
            await asyncio.sleep(0.05)
        finally:
            for _ in range(slots):
                semaphore.release()

    async def _waiter() -> None:
        await holder_ready.wait()
        await semaphore.acquire()
        semaphore.release()

    await asyncio.gather(_holder(), _waiter())


def test_model_cache_update_lock_survives_a_closed_event_loop() -> None:
    """The lock guarding ``_refresh_bedrock_models`` is reusable across loops.

    Reproduces the failure from issue #299 directly on the production object: a
    contended acquire runs to completion on one loop, that loop closes, and a later
    contended acquire on a new loop must not raise.

    Ref: stdapi/models/__init__.py:_refresh_bedrock_models
    """
    lock = stdapi.models._CACHE["update_lock"]  # noqa: SLF001

    asyncio.run(_contend_for_lock(lock))
    asyncio.run(_contend_for_lock(lock))


def test_model_cache_access_locks_survive_a_closed_event_loop() -> None:
    """The catalog and prompt/profile access locks share the update lock's fix.

    Ref: stdapi/models/__init__.py:_ModelCache
    """
    access_lock = stdapi.models._CACHE["access_lock"]  # noqa: SLF001
    user_profiles_access_lock = stdapi.models._CACHE["user_profiles_access_lock"]  # noqa: SLF001
    prompts_access_lock = stdapi.models._CACHE["prompts_access_lock"]  # noqa: SLF001

    for lock in (access_lock, user_profiles_access_lock, prompts_access_lock):
        asyncio.run(_contend_for_lock(lock))
        asyncio.run(_contend_for_lock(lock))


def test_route_model_list_locks_survive_a_closed_event_loop() -> None:
    """The ``/v1/models`` and ``/api/tags`` rebuild locks have the same module shape.

    Ref: stdapi/routes/openai_models.py:_ALL_MODELS_LOCK
         stdapi/routes/ollama_models.py:_LIST_LOCK
    """
    for lock in (
        openai_models._ALL_MODELS_LOCK,  # noqa: SLF001
        ollama_models._LIST_LOCK,  # noqa: SLF001
    ):
        asyncio.run(_contend_for_lock(lock))
        asyncio.run(_contend_for_lock(lock))


def test_batch_and_indexing_semaphores_survive_a_closed_event_loop() -> None:
    """The batch-creation and vector-store-indexing semaphores share the same fix.

    Ref: stdapi/batches.py:_CREATE_SEMAPHORE
         stdapi/vector_stores/engine.py:_INDEXING_SEMAPHORE
    """
    for semaphore, slots in (
        (batches._CREATE_SEMAPHORE, batches._CREATE_SLOTS),  # noqa: SLF001
        (engine._INDEXING_SEMAPHORE, engine._INDEXING_SLOTS),  # noqa: SLF001
    ):
        asyncio.run(_contend_for_semaphore(semaphore, slots))
        asyncio.run(_contend_for_semaphore(semaphore, slots))
