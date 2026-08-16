"""Which backend serves a vector store.

The single place a second backend is wired in: everything else addresses a
store through :class:`~stdapi.vector_stores.backend.VectorIndex` and never
learns which one answered.

Two kinds of store are served. One this deployment created and owns, whose
identifier it minted; and one it was given — a store that already exists
elsewhere, addressed by an identifier naming it, which answers the engine's
bookkeeping questions itself through
:class:`~stdapi.vector_stores.backend.ExternalStore`.
"""

from typing import TYPE_CHECKING, Final

from stdapi.vector_stores.knowledge_base import (
    KnowledgeBaseIndex,
    is_knowledge_base_store,
)
from stdapi.vector_stores.s3_vectors import S3VectorsIndex

if TYPE_CHECKING:
    from stdapi.vector_stores.backend import ExternalStore, VectorIndex
    from stdapi.vector_stores.models import StoreRecord

#: The backend a store this deployment created is served from.
_S3_VECTORS: Final[VectorIndex] = S3VectorsIndex()

#: The backend a store addressed on a knowledge base is served from.
_KNOWLEDGE_BASE: Final[KnowledgeBaseIndex] = KnowledgeBaseIndex()

#: Where a file the store it was sent to cannot index would be indexed instead.
_KNOWLEDGE_BASE_ALTERNATIVE: Final[str] = (
    "A knowledge base store indexes this file type as it stands."
)


def default_backend() -> VectorIndex:
    """Return the backend a new vector store is created on.

    Returns:
        The backend that will hold its vectors.
    """
    return _S3_VECTORS


def backend_for(store: StoreRecord | str) -> VectorIndex:
    """Return the backend serving an existing vector store.

    Args:
        store: The store record, or its identifier when the caller does not
            already hold the record.

    Returns:
        The backend holding that store's vectors.
    """
    if is_knowledge_base_store(store if isinstance(store, str) else store.id):
        return _KNOWLEDGE_BASE
    return _S3_VECTORS


def alternative_for(media_type: str, backend: VectorIndex) -> str:
    """Return where a file *backend* cannot index would be indexed instead.

    Only the formats every knowledge base takes are offered: a wider set is a
    property of one generation of them, and would send the caller to a store
    that refuses the file all the same.

    Args:
        media_type: The media type the file was uploaded with.
        backend: The backend serving the store that refused the file.

    Returns:
        A sentence naming another kind of store that takes *media_type* as it
        stands, or ``""`` when none does.
    """
    if backend is not _KNOWLEDGE_BASE and (
        media_type in _KNOWLEDGE_BASE.capabilities.ingested_media_types
    ):
        return _KNOWLEDGE_BASE_ALTERNATIVE
    return ""


def external_store_for(store: StoreRecord | str) -> ExternalStore | None:
    """Return the backend owning a store this deployment only addresses.

    Args:
        store: The store record, or its identifier when the caller does not
            already hold the record.

    Returns:
        The backend answering for that store, or ``None`` when the deployment
        owns the store and its own records answer for it.
    """
    if is_knowledge_base_store(store if isinstance(store, str) else store.id):
        return _KNOWLEDGE_BASE
    return None


def external_stores() -> tuple[ExternalStore, ...]:
    """Return every backend serving stores this deployment only addresses.

    Returns:
        The backends a listing must ask, alongside this deployment's own records.
    """
    return (_KNOWLEDGE_BASE,)
