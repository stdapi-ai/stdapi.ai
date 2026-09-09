"""What the vector store indexer owes when it does not finish the way it began.

Indexing runs after the response that asked for it, so two things it takes for
granted are not true of it: the file it is writing chunks for may be deleted
under it, and the request whose cleanup list it would use may be finished, or
never have existed at all. Both leave a trace the caller can see -- content the
API reports as deleted still answering searches, or a whole wave of files
stranded -- so both are pinned here.

Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/deleteFile
     stdapi/vector_stores/engine.py:index_files
     stdapi/vector_stores/engine.py:_store_chunks
"""

from typing import TYPE_CHECKING, Any

import pytest

from stdapi.cleanup import CLEANUPS
from stdapi.vector_stores import (
    StoreRecord,
    detach_file,
    index_files,
    read_file,
    read_store,
    search,
    update_record,
)
from stdapi.vector_stores.records import file_key, read_record, store_key
from stdapi.vector_stores.s3_vectors import index_name
from tests._helpers import make_client_error
from tests.test_openai_vector_stores import (
    _PLANTED,
    _TEXT_FILE,
    _attach,
    _create_store,
    _FakeBackend,
    _run_cleanups,
    scheduled_cleanups,  # noqa: F401 -- re-exported so the fixture resolves here
    vector_backend,  # noqa: F401 -- re-exported so the fixture resolves here
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

pytestmark = pytest.mark.local

#: Identifier correlating a wave with the request that asked for it.
_REQUEST_ID = "test-request"


async def _expire(store: StoreRecord) -> None:
    """Move a store past the expiration it was created with.

    Args:
        store: A store created with an expiration.
    """
    await update_record(
        StoreRecord,
        store_key(store.id),
        lambda stored: setattr(stored, "last_active_at", 0),
    )


async def _index_deleted(store: StoreRecord) -> bool:
    """Whether the store has recorded the release of its storage.

    Read straight from the record: reading it through the engine would schedule
    the release a second time.

    Args:
        store: The store to read.

    Returns:
        Whether the release has been recorded.
    """
    current = await read_record(StoreRecord, store_key(store.id))
    assert current is not None
    return current[0].index_deleted


class TestChunksNoRecordAnswersForAreTakenBack:
    """Chunks written for a file the store no longer holds leave with it.

    The count a file will hold is written before the chunks exist, so a mid-write
    failure orphans none -- but it also means a delete landing during the write
    reclaims keys that are not there yet, and the chunks land behind it. Nothing
    else sweeps them: the file record naming them is gone, so a search would keep
    answering with a document the API already reported as deleted.

    Ref: stdapi/vector_stores/engine.py:_discard_vectors
         stdapi/vector_stores/engine.py:_reclaim_file
    """

    async def test_a_delete_landing_mid_write_leaves_nothing_searchable(
        self,
        vector_backend: _FakeBackend,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A file deleted while its chunks are going in is deleted for good.

        The delete is answered, and its reclaim finished, before a single chunk
        reaches the index -- which is what the whole write of any file under one
        batch does.

        Ref: stdapi/vector_stores/engine.py:detach_file
             stdapi/vector_stores/engine.py:search
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        write = vector_backend.vectors.put_vectors
        deleted = False

        async def delete_then_write(**params: Any) -> dict[str, Any]:  # noqa: ANN401
            """Answer the delete, and finish its reclaim, before writing."""
            nonlocal deleted
            if not deleted:
                deleted = True
                await detach_file(store.id, file_id)
                await _run_cleanups(CLEANUPS.get())
            return await write(**params)

        monkeypatch.setattr(vector_backend.vectors, "put_vectors", delete_then_write)
        await index_files(store.id, [file_id], "", _REQUEST_ID)

        assert vector_backend.vectors.indexes[index_name(store.id)] == {}
        settled = await read_store(store.id)
        assert settled.detaching == []
        assert file_key(store.id, file_id) not in vector_backend.records.objects
        assert (
            await search(
                settled,
                [_PLANTED],
                max_num_results=5,
                filters=None,
                score_threshold=None,
            )
            == []
        )

    async def test_a_delete_under_a_write_that_then_fails_leaves_nothing(
        self,
        vector_backend: _FakeBackend,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The chunks a failing write did land are swept the same way.

        The write reaches the index and then fails, so what the reclaim ran too
        early to catch is exactly what a successful write would have left.

        Ref: stdapi/vector_stores/engine.py:_store_chunks
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        write = vector_backend.vectors.put_vectors
        deleted = False

        async def delete_write_then_fail(**params: Any) -> dict[str, Any]:  # noqa: ANN401
            """Answer the delete, land the chunks behind it, then fail."""
            nonlocal deleted
            if not deleted:
                deleted = True
                await detach_file(store.id, file_id)
                await _run_cleanups(CLEANUPS.get())
            await write(**params)
            throttled = make_client_error("SlowDown", "PutVectors", status=503)
            raise throttled

        monkeypatch.setattr(
            vector_backend.vectors, "put_vectors", delete_write_then_fail
        )
        await index_files(store.id, [file_id], "", _REQUEST_ID)

        assert vector_backend.vectors.indexes[index_name(store.id)] == {}
        assert file_key(store.id, file_id) not in vector_backend.records.objects

    async def test_a_file_a_later_indexing_owns_keeps_its_chunks(
        self,
        vector_backend: _FakeBackend,  # noqa: F811
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A wave that lost the record leaves the chunks of the wave that has it.

        Re-attaching the file starts a second wave over the same keys: the
        first one no longer settles the record, and must not take the second
        one's chunks out of the index on its way past.

        Ref: stdapi/vector_stores/engine.py:_discard_vectors
             stdapi/vector_stores/engine.py:attach_files
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        write = vector_backend.vectors.put_vectors
        replaced = False

        async def write_then_replace(**params: Any) -> dict[str, Any]:  # noqa: ANN401
            """Let a second wave attach and finish the file behind this one."""
            nonlocal replaced
            stored = await write(**params)
            if not replaced:
                replaced = True
                await _attach(store, [file_id])
                monkeypatch.setattr(vector_backend.vectors, "put_vectors", write)
                await index_files(store.id, [file_id], "", _REQUEST_ID)
            return stored

        monkeypatch.setattr(vector_backend.vectors, "put_vectors", write_then_replace)
        await index_files(store.id, [file_id], "", _REQUEST_ID)

        assert (await read_file(store.id, file_id)).status == "completed"
        assert vector_backend.vectors.indexes[index_name(store.id)]


class TestAWaveOwnsTheCleanupsItSchedules:
    """Indexing runs its cleanups itself, wherever it was started from.

    A wave resumed from the indexing queue never ran inside a request, and one
    started by a request outlives the moment that request's own cleanups were
    run. Reading a store whose expiration has passed releases its storage that
    way, so either case would lose the release -- or, with no request at all,
    end the whole wave on the spot and leave every file it named in progress.

    Ref: stdapi/vector_stores/engine.py:index_files
         stdapi/vector_stores/engine.py:_release_expired
         stdapi/cleanup.py:schedule_cleanup
    """

    async def test_a_wave_with_no_request_around_it_still_runs_them(
        self,
        vector_backend: _FakeBackend,  # noqa: F811
    ) -> None:
        """A wave started outside any request indexes its files and cleans up.

        Ref: stdapi/vector_stores/jobs.py:_run_index_files
        """
        store = await _create_store(expires_after_days=1)
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        await _expire(store)

        await index_files(store.id, [file_id], "", _REQUEST_ID)

        assert (await read_file(store.id, file_id)).status == "completed"
        assert await _index_deleted(store)
        assert index_name(store.id) not in vector_backend.vectors.indexes

    async def test_a_wave_does_not_write_into_the_cleanups_of_its_request(
        self,
        vector_backend: _FakeBackend,  # noqa: F811
        scheduled_cleanups: list[Awaitable[None]],  # noqa: F811
    ) -> None:
        """What a wave schedules never lands on a list already drained.

        Ref: stdapi/cleanup.py:run_scheduled_cleanups
        """
        store = await _create_store(expires_after_days=1)
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        await _expire(store)

        await index_files(store.id, [file_id], "", _REQUEST_ID)

        assert scheduled_cleanups == []
        assert await _index_deleted(store)
