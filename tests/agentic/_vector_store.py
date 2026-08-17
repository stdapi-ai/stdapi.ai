"""The indexed vector store the retrieval clients in this lane answer from.

Every client that drives the ``file_search`` tool needs the same thing: one small
note indexed in a real gateway vector store, holding a fact no model can already
know. Building it is three calls and a poll -- upload, create, wait for the file
to leave ``in_progress`` -- and getting the wait wrong produces a store that
answers nothing rather than an error, so the sequence lives here once instead of
in each module.

Ref: https://developers.openai.com/api/reference/resources/vector_stores
     docs/api_openai_vector_stores.md
     stdapi/routes/openai_vector_stores.py:router
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai import OpenAI

#: Reference number planted in the note; a retrieved answer is the only way to it.
PLANTED_NUMBER = "6274"

#: The note itself, whose facts appear in no training corpus.
NOTE = (
    "Kestrel Observatory maintenance log.\n\n"
    "The primary mirror was recoated on 14 March 2031 by the night crew.\n"
    f"They logged the job under reference number {PLANTED_NUMBER}.\n"
).encode()

#: Seconds a vector store may spend indexing its single small file.
INDEX_TIMEOUT = 240.0

#: Seconds between two polls of a store that is still indexing.
_POLL_INTERVAL = 2.0


@contextmanager
def indexed_store(client: OpenAI, name: str) -> Iterator[str]:
    """Yield a vector store holding :data:`NOTE`, indexed and ready to search.

    The store and its file are deleted on the way out, including when the body
    fails: a leaked store keeps costing storage and shows up in the next run's
    listings.

    Args:
        client: OpenAI SDK client bound to the gateway under test.
        name: Name given to the store, to identify it in a listing.

    Yields:
        The vector store ID.
    """
    uploaded = client.files.create(
        file=("kestrel.txt", NOTE, "text/plain"), purpose="assistants"
    )
    store = client.vector_stores.create(name=name, file_ids=[uploaded.id])
    try:
        deadline = time.monotonic() + INDEX_TIMEOUT
        while True:
            counts = client.vector_stores.retrieve(store.id).file_counts
            if counts.in_progress == 0 and counts.total >= 1:
                break
            assert time.monotonic() < deadline, (
                f"vector store {store.id} still indexing after {INDEX_TIMEOUT}s: "
                f"{counts}"
            )
            time.sleep(_POLL_INTERVAL)
        assert counts.completed == 1, counts
        yield store.id
    finally:
        client.vector_stores.delete(store.id)
        client.files.delete(uploaded.id)
