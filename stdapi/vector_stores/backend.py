"""The contract between the vector store engine and the index behind it.

The engine owns what the API answers with — per-store counters, per-file
status, per-batch progress, the text a file is split into. A backend owns
everything else: how a vector is stored, how a filter is expressed, what a
score means, and which documents it can take as they stand.

Because those differ per backend, a backend **declares** them
(:class:`IndexCapabilities`) and the engine refuses or degrades against the
declaration. Discovering a gap mid-request is the failure this replaces: a
filter operator the index cannot express, or a score that is not comparable to
a threshold, becomes a clean 400 before any backend work rather than the
backend's own error surfacing to the caller.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, NoReturn, Protocol

from stdapi.api_errors import ApiError
from stdapi.types.openai_vector_stores import (
    Attributes,
    ComparisonFilter,
    CompoundFilter,
    SearchFilter,
)
from stdapi.utils import validation_error_handler

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Iterable, Sequence

    from stdapi.types import JsonMapping

#: The feature name a caller reads when the deployment cannot serve vector stores.
FEATURE: Final[str] = "The Vector Stores API"


@dataclass(slots=True)
class IndexVector:
    """One indexed chunk, as the engine writes it and reads it back.

    The fields are the meaning of a chunk, not an index payload: how they are
    encoded — metadata keys, namespacing, reserved names — is the backend's
    business and never leaves it.

    Attributes:
        key: Identifier of the chunk within its index.
        file_id: Identifier of the file the chunk comes from.
        filename: Name of that file.
        chunk_index: Position of the chunk within its file.
        text: The chunk text.
        attributes: The caller attributes stored with the file.
        embedding: The chunk's vector; empty when it was not asked for.
    """

    key: str
    file_id: str
    filename: str
    chunk_index: int
    text: str
    attributes: Attributes = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)


@dataclass(slots=True)
class VectorMatch:
    """One hit a query answered with.

    Attributes:
        key: Identifier of the matching chunk, for de-duplication across queries.
        score: Similarity, in ``[0, 1]`` when the backend declares
            ``normalised_score``.
        file_id: Identifier of the file the chunk comes from.
        filename: Name of that file.
        text: The matching chunk text.
        attributes: The caller attributes stored with the file.
    """

    key: str
    score: float
    file_id: str
    filename: str
    text: str
    attributes: Attributes


@dataclass(frozen=True, slots=True)
class IndexCapabilities:
    """What a backend can express, declared before the engine relies on it.

    Attributes:
        filter_operators: Upstream comparison operators the backend can
            express, out of ``eq ne gt gte lt lte in nin``.
        filter_combinators: Upstream combinators it can express, out of
            ``and or``.
        ingested_media_types: Media types the backend indexes as they stand,
            beyond text.
        refused_media_types: Media types that are never indexable text, refused
            before their bytes are read.
        ingests_decodable_text: Whether any payload that decodes as text is
            indexed, whatever its media type.
        chunks_on_ingestion: Whether the backend chooses the passage boundaries
            itself. A backend declaring this cannot honour a per-request
            chunking strategy, and supplies its own ingestion path.
        normalised_score: Whether :attr:`VectorMatch.score` is a similarity in
            ``[0, 1]``, and therefore comparable to
            ``ranking_options.score_threshold``.
        max_chunk_bytes: Bytes of text one vector holds, or ``0`` when
            unbounded.
    """

    filter_operators: frozenset[str]
    filter_combinators: frozenset[str]
    ingested_media_types: frozenset[str]
    refused_media_types: frozenset[str]
    ingests_decodable_text: bool
    chunks_on_ingestion: bool
    normalised_score: bool
    max_chunk_bytes: int

    def may_ingest(self, media_type: str) -> bool:
        """Whether a payload of *media_type* is worth reading at all.

        Args:
            media_type: The media type the file was uploaded with.

        Returns:
            Whether the backend may index it. A backend taking decodable text
            still refuses bytes that turn out not to be text.
        """
        if media_type in self.refused_media_types:
            return False
        return self.ingests_decodable_text or media_type in self.ingested_media_types


class VectorIndex(Protocol):
    """The vector index of one deployment, whichever service holds it.

    Every method takes the vector store identifier and maps it to whatever the
    backend addresses — an index name, a knowledge base, a namespace. Nothing
    of that mapping is visible to the engine.
    """

    @property
    def capabilities(self) -> IndexCapabilities:
        """What this backend can express."""
        ...

    def check_configured(self) -> None:
        """Raise when the deployment lacks what this backend needs.

        Raises:
            FeatureUnavailableError: When the backend cannot be served (503),
                naming the missing settings for the operator.
        """
        ...

    def check_attributes(self, attributes: Attributes) -> None:
        """Raise when *attributes* exceed what one vector can carry.

        Args:
            attributes: The caller-supplied attributes.

        Raises:
            ApiError: When they do not fit the per-file budget (400).
        """
        ...

    async def create_index(self, store_id: str, *, dimensions: int) -> None:
        """Create the index backing a new vector store.

        Args:
            store_id: A validated vector store identifier.
            dimensions: Length of the vectors it will hold.
        """
        ...

    async def delete_index(self, store_id: str) -> None:
        """Delete the index backing *store_id*, ignoring an already-deleted one.

        Args:
            store_id: A validated vector store identifier.
        """
        ...

    async def put_vectors(
        self, store_id: str, vectors: AsyncIterable[IndexVector]
    ) -> None:
        """Write *vectors* into the index of *store_id*.

        Taken as an asynchronous stream so the engine can embed a file as the
        backend writes it: neither side holds a whole file's embeddings.

        Args:
            store_id: A validated vector store identifier.
            vectors: The chunks to write, in document order.
        """
        ...

    async def get_vectors(
        self, store_id: str, keys: Sequence[str], *, with_embeddings: bool
    ) -> list[IndexVector]:
        """Read the chunks stored under *keys*.

        Args:
            store_id: A validated vector store identifier.
            keys: The chunk keys to read.
            with_embeddings: Whether the vectors themselves are needed.

        Returns:
            The chunks that exist, in no particular order.
        """
        ...

    async def delete_vectors(self, store_id: str, keys: Sequence[str]) -> None:
        """Remove the chunks stored under *keys*, best effort.

        Keys that are already gone are not an error, and a batch that fails
        does not stop the rest: every caller of this schedules it as cleanup
        and has nothing to report a failure to.

        Args:
            store_id: A validated vector store identifier.
            keys: The chunk keys to remove.
        """
        ...

    async def query(
        self,
        store_id: str,
        embeddings: Sequence[Sequence[float]],
        *,
        max_results: int,
        search_filter: SearchFilter | None,
    ) -> list[VectorMatch]:
        """Return the chunks closest to *embeddings*.

        The engine merges, thresholds and truncates what comes back, so the
        matches of several queries may be returned as one list with repeated
        keys.

        Args:
            store_id: A validated vector store identifier.
            embeddings: One vector per query text.
            max_results: Maximum matches to return per query.
            search_filter: Restriction over the files' attributes, already
                checked against :attr:`IndexCapabilities.filter_operators`.

        Returns:
            The matches, in no particular order.
        """
        ...


def parse_filter(search_filter: SearchFilter | JsonMapping) -> SearchFilter:
    """Validate one node of a search filter.

    A filter nested more than one level deep arrives as a plain mapping,
    because a self-referential request schema cannot be published.

    Args:
        search_filter: The node as the API received it.

    Returns:
        The node as a validated filter.

    Raises:
        RequestValidationError: When the node is not a valid filter.
    """
    if not isinstance(search_filter, dict):
        return search_filter
    with validation_error_handler():
        if search_filter.get("type") in ("and", "or"):
            return CompoundFilter.model_validate(search_filter)
        return ComparisonFilter.model_validate(search_filter)


def check_filter(
    search_filter: SearchFilter | JsonMapping, capabilities: IndexCapabilities
) -> None:
    """Refuse a filter the backend cannot express, before it reaches the backend.

    Args:
        search_filter: The filter as the API received it.
        capabilities: What the backend serving the store can express.

    Raises:
        ApiError: When an operator or a combinator is not available (400).
        RequestValidationError: When a nested entry is not a valid filter.
    """
    node = parse_filter(search_filter)
    if isinstance(node, CompoundFilter):
        if node.type not in capabilities.filter_combinators:
            _raise_unsupported_filter(node.type, capabilities.filter_combinators)
        for inner in node.filters:
            check_filter(inner, capabilities)
    elif node.type not in capabilities.filter_operators:
        _raise_unsupported_filter(node.type, capabilities.filter_operators)


def _raise_unsupported_filter(used: str, available: frozenset[str]) -> NoReturn:
    """Raise the 400 answering a filter operator this store cannot apply.

    Args:
        used: The operator or combinator the request used.
        available: The ones the store accepts instead.

    Raises:
        ApiError: Always (400).
    """
    supported = ", ".join(f"'{entry}'" for entry in sorted(available))
    msg = (
        f"The '{used}' filter is not available on this vector store. "
        f"It accepts {supported}."
    )
    raise ApiError(msg)


async def as_stream(vectors: Iterable[IndexVector]) -> AsyncIterator[IndexVector]:
    """Stream *vectors* for a caller that already holds them all.

    Args:
        vectors: The chunks to write.

    Yields:
        Each chunk, in the order it was given.
    """
    for vector in vectors:
        yield vector
