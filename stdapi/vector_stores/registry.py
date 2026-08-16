"""Which backend serves a vector store.

The single place a second backend is wired in: everything else addresses a
store through :class:`~stdapi.vector_stores.backend.VectorIndex` and never
learns which one answered.
"""

from typing import TYPE_CHECKING, Final

from stdapi.vector_stores.s3_vectors import S3VectorsIndex

if TYPE_CHECKING:
    from stdapi.vector_stores.backend import VectorIndex
    from stdapi.vector_stores.models import StoreRecord

#: The backend every store is served from today.
_S3_VECTORS: Final[VectorIndex] = S3VectorsIndex()


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
    del store
    return _S3_VECTORS
