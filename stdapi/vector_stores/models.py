"""The records a vector store is made of, and the values the routes read.

Backend-neutral by construction: nothing here knows how vectors are stored or
searched, only what the API answers with.
"""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from stdapi.types.openai_vector_stores import (
    CHUNK_OVERLAP_TOKENS_DEFAULT,
    CHUNK_SIZE_TOKENS_DEFAULT,
    Attributes,
)
from stdapi.utils import now_utc_timestamp

#: Seconds in a day, for the ``expires_after`` anchor arithmetic.
_SECONDS_PER_DAY: int = 86400


class FileCountsRecord(BaseModel):
    """Per-status file counts of a store or a batch."""

    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0

    @property
    def total(self) -> int:
        """Total number of files counted."""
        return self.in_progress + self.completed + self.failed + self.cancelled


class StoreRecord(BaseModel):
    """The bookkeeping of one vector store.

    ``embedding_model`` and ``dimensions`` are frozen at creation: an index's
    dimension cannot be changed, so a later change to the configured default
    must not reach an existing store.
    """

    id: str
    created_at: int
    last_active_at: int
    name: str = ""
    description: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)
    expires_after_days: int | None = None
    embedding_model: str
    dimensions: int
    max_chunk_size_tokens: int = CHUNK_SIZE_TOKENS_DEFAULT
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS_DEFAULT
    file_counts: FileCountsRecord = Field(default_factory=FileCountsRecord)
    usage_bytes: int = 0
    index_deleted: bool = False
    external_status: Literal["in_progress", "completed"] | None = None

    @property
    def expires_at(self) -> int | None:
        """Unix timestamp the store expires at, or ``None`` when it never does."""
        if self.expires_after_days is None:
            return None
        return self.last_active_at + self.expires_after_days * _SECONDS_PER_DAY

    @property
    def expired(self) -> bool:
        """Whether the store has reached its expiration."""
        expires_at = self.expires_at
        return expires_at is not None and now_utc_timestamp() >= expires_at

    @property
    def status(self) -> Literal["expired", "in_progress", "completed"]:
        """The store status the API reports.

        A store held elsewhere reports its own readiness: its files are not
        counted here, so the counters cannot answer for it.
        """
        if self.expired:
            return "expired"
        if self.external_status is not None:
            return self.external_status
        return "in_progress" if self.file_counts.in_progress else "completed"


class FileErrorRecord(BaseModel):
    """Why a file failed to be indexed."""

    code: Literal["server_error", "unsupported_file", "invalid_file"]
    message: str


class FileRecord(BaseModel):
    """The bookkeeping of one file attached to a vector store.

    ``attributes`` is ``None`` and ``max_chunk_size_tokens`` is ``0`` on a store
    that answers for its own files: it keeps the attributes searchable without
    reading them back, and cuts the passages itself. Both are reported as
    unknown rather than as a value this server would be inventing.
    """

    id: str
    created_at: int
    filename: str = ""
    status: Literal["in_progress", "completed", "cancelled", "failed"] = "in_progress"
    last_error: FileErrorRecord | None = None
    usage_bytes: int = 0
    chunk_count: int = 0
    previous_chunk_count: int = 0
    attributes: Attributes | None = None
    max_chunk_size_tokens: int = CHUNK_SIZE_TOKENS_DEFAULT
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS_DEFAULT
    batch_id: str = ""


class BatchRecord(BaseModel):
    """The bookkeeping of one file batch."""

    id: str
    created_at: int
    file_counts: FileCountsRecord = Field(default_factory=FileCountsRecord)
    cancel_requested: bool = False

    @property
    def status(self) -> Literal["in_progress", "completed", "cancelled", "failed"]:
        """The batch status the API reports."""
        if self.file_counts.in_progress:
            return "in_progress"
        if self.file_counts.cancelled:
            return "cancelled"
        return "failed" if self.file_counts.failed else "completed"


@dataclass(slots=True)
class SearchResult:
    """One search hit, as the search route and the retrieval tool both consume it.

    Attributes:
        file_id: Identifier of the file the chunk comes from.
        filename: Name of that file.
        score: Similarity in ``[0, 1]``, 1 being an exact match.
        text: The matching chunk text.
        attributes: The attributes stored with the file.
    """

    file_id: str
    filename: str
    score: float
    text: str
    attributes: Attributes


@dataclass(slots=True)
class PendingFile:
    """One file waiting to be indexed.

    Attributes:
        file_id: Identifier of the uploaded file.
        attributes: Attributes to store with every chunk.
        max_chunk_size_tokens: Chunk size for this file.
        chunk_overlap_tokens: Chunk overlap for this file.
    """

    file_id: str
    attributes: Attributes = field(default_factory=dict)
    max_chunk_size_tokens: int = CHUNK_SIZE_TOKENS_DEFAULT
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS_DEFAULT
