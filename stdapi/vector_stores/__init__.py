"""Vector Stores — indexing, bookkeeping and semantic search.

A vector store is a vector index plus the bookkeeping the API answers with.
The package separates the two so a second index can be served without touching
the callers:

- ``backend`` — the :class:`~stdapi.vector_stores.backend.VectorIndex` contract,
  the capabilities a backend declares, and the
  :class:`~stdapi.vector_stores.backend.ExternalStore` contract a store held
  elsewhere answers through.
- ``s3_vectors`` — the Amazon S3 Vectors implementation of it.
- ``knowledge_base`` — the Amazon Bedrock Knowledge Bases implementation of it.
- ``registry`` — which backend serves which store.
- ``models`` — the records and the values the routes read.
- ``records`` — those records in the application bucket, under conditional writes.
- ``engine`` — everything backend-neutral: identifiers, chunking, indexing,
  counters and search.

This module is the surface the routes address; nothing outside the package
imports the modules above directly.
"""

from stdapi.vector_stores.engine import (
    FILE_BATCH_ID_PATTERN,
    VECTOR_STORE_FILE_ID_PATTERN,
    VECTOR_STORE_ID_PATTERN,
    attach_files,
    cancel_batch,
    check_attributes,
    check_chunking_strategy,
    chunk_text,
    create_store,
    delete_store,
    detach_file,
    index_files,
    list_batch_files,
    list_store_files,
    list_stores,
    new_batch_id,
    new_store_id,
    parse_batch_id,
    parse_store_id,
    read_batch,
    read_file,
    read_file_chunks,
    read_store,
    resolve_embedding_model,
    search,
    start_indexing,
    touch_store,
    update_file_attributes,
    update_store,
    vector_key,
)
from stdapi.vector_stores.models import (
    BatchRecord,
    FileCountsRecord,
    FileErrorRecord,
    FileRecord,
    PendingFile,
    SearchResult,
    StoreRecord,
)
from stdapi.vector_stores.records import records_bucket, update_record

__all__ = [
    "FILE_BATCH_ID_PATTERN",
    "VECTOR_STORE_FILE_ID_PATTERN",
    "VECTOR_STORE_ID_PATTERN",
    "BatchRecord",
    "FileCountsRecord",
    "FileErrorRecord",
    "FileRecord",
    "PendingFile",
    "SearchResult",
    "StoreRecord",
    "attach_files",
    "cancel_batch",
    "check_attributes",
    "check_chunking_strategy",
    "chunk_text",
    "create_store",
    "delete_store",
    "detach_file",
    "index_files",
    "list_batch_files",
    "list_store_files",
    "list_stores",
    "new_batch_id",
    "new_store_id",
    "parse_batch_id",
    "parse_store_id",
    "read_batch",
    "read_file",
    "read_file_chunks",
    "read_store",
    "records_bucket",
    "resolve_embedding_model",
    "search",
    "start_indexing",
    "touch_store",
    "update_file_attributes",
    "update_record",
    "update_store",
    "vector_key",
]
