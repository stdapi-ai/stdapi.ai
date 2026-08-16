"""The Amazon S3 Vectors implementation of the vector index contract.

Everything specific to that service lives here and nowhere else: the index
naming, the metadata layout, the filter dialect, the distance-to-score
conversion, the per-call batch sizes and the IAM actions each call needs.
"""

from contextlib import suppress
from typing import TYPE_CHECKING, Any, Final, NoReturn

from botocore.exceptions import BotoCoreError, ClientError

from stdapi.api_errors import (
    ApiError,
    FeatureUnavailableError,
    feature_unavailable_guard,
)
from stdapi.aws import get_client
from stdapi.config import SETTINGS
from stdapi.types.openai_vector_stores import (
    Attributes,
    AttributeValue,
    CompoundFilter,
    SearchFilter,
)
from stdapi.utils import to_json_bytes
from stdapi.vector_stores._concurrency import gather_bounded
from stdapi.vector_stores.backend import (
    FEATURE,
    IndexCapabilities,
    IndexVector,
    VectorMatch,
    parse_filter,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Sequence
    from contextlib import AbstractContextManager

    from pydantic import JsonValue

    from stdapi.types import JsonMapping

#: What an unreachable vector endpoint means, for the operator.
_UNREACHABLE_DETAIL: Final[str] = (
    "The S3 Vectors endpoint is unreachable or timed out: S3 Vectors is offered "
    "in fewer regions than model inference; set 'aws_s3_vectors_region' to a "
    "region that provides it."
)

#: Distance metric every index is created with.
_DISTANCE_METRIC: Final = "cosine"

#: Vector element type every index is created with.
_DATA_TYPE: Final = "float32"

#: Metadata key holding the chunk text; never usable in a filter.
_TEXT_KEY: Final = "_text"

#: Metadata key holding the source file name.
_FILENAME_KEY: Final = "_filename"

#: Metadata key holding the source file identifier.
_FILE_ID_KEY: Final = "_file_id"

#: Metadata key holding the chunk position within its file.
_CHUNK_INDEX_KEY: Final = "_chunk_index"

#: Metadata keys the index stores but never filters on; fixed at index creation.
_NON_FILTERABLE_KEYS: Final[tuple[str, ...]] = (
    _TEXT_KEY,
    _FILENAME_KEY,
    _FILE_ID_KEY,
    _CHUNK_INDEX_KEY,
)

#: Prefix isolating caller attribute keys from the keys above.
_ATTRIBUTE_PREFIX: Final = "a_"

#: Bytes of filterable metadata one vector accepts.
_MAX_FILTERABLE_BYTES: Final[int] = 2048

#: Bytes of chunk text one vector holds, leaving room for the other metadata.
_MAX_CHUNK_BYTES: Final[int] = 32768

#: Vectors written per index write.
_PUT_VECTORS_BATCH: Final[int] = 500

#: Vector keys read per index read.
_GET_VECTORS_BATCH: Final[int] = 100

#: Vector keys deleted per index delete.
_DELETE_VECTORS_BATCH: Final[int] = 500

#: Queries issued concurrently by one search.
_QUERY_WAVE: Final[int] = 16

#: OpenAI comparison operator → index filter operator.
_FILTER_OPERATORS: Final[dict[str, str]] = {
    "eq": "$eq",
    "ne": "$ne",
    "gt": "$gt",
    "gte": "$gte",
    "lt": "$lt",
    "lte": "$lte",
    "in": "$in",
    "nin": "$nin",
}

#: Content types whose bytes are never text, rejected before decoding is attempted.
_BINARY_CONTENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "application/gzip",
        "application/msword",
        "application/octet-stream",
        "application/pdf",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
    }
)

#: What this backend can express, as the engine reads it.
_CAPABILITIES: Final = IndexCapabilities(
    filter_operators=frozenset(_FILTER_OPERATORS),
    filter_combinators=frozenset({"and", "or"}),
    # Nothing is ingested as a document: the passages are text this server cut.
    ingested_media_types=frozenset(),
    refused_media_types=_BINARY_CONTENT_TYPES,
    ingests_decodable_text=True,
    chunks_on_ingestion=False,
    normalised_score=True,
    max_chunk_bytes=_MAX_CHUNK_BYTES,
)


def index_name(store_id: str) -> str:
    """Return the index name backing *store_id*.

    An index name accepts neither underscores nor uppercase, so the identifier's
    separator is substituted; the mapping stays total and reversible.

    Args:
        store_id: A vector store identifier.

    Returns:
        The index name.
    """
    return store_id.replace("_", "-", 1)


def attribute_key(key: str) -> str:
    """Return the metadata key storing the caller attribute *key*."""
    return f"{_ATTRIBUTE_PREFIX}{key}"


def score_from_distance(distance: float) -> float:
    """Convert a cosine distance into the similarity score the API reports.

    The distance is ``1 - cosine_similarity``, so an exact match scores 1 and an
    orthogonal one scores 0; opposite vectors clamp to 0 rather than reporting a
    negative score.

    Args:
        distance: The distance the index returned.

    Returns:
        A score in ``[0, 1]``.
    """
    return max(0.0, min(1.0, 1.0 - distance))


def translate_filter(search_filter: SearchFilter | JsonMapping) -> JsonMapping:
    """Translate an upstream search filter into an index filter.

    Args:
        search_filter: The filter as the API received it. A filter nested more
            than one level deep arrives as a plain mapping and is validated here.

    Returns:
        The equivalent index filter.

    Raises:
        RequestValidationError: When a nested filter is not a valid filter.
    """
    node = parse_filter(search_filter)
    if isinstance(node, CompoundFilter):
        return {f"${node.type}": [translate_filter(inner) for inner in node.filters]}
    value: JsonValue = node.value  # type: ignore[assignment]
    return {attribute_key(node.key): {_FILTER_OPERATORS[node.type]: value}}


def _vectors_guard(*actions: str) -> AbstractContextManager[None]:
    """Answer a denied index call as the Vector Stores API being unavailable.

    Args:
        *actions: The ``s3vectors`` actions the guarded call needs.

    Returns:
        The guard wrapping the call.
    """
    permissions = ", ".join(f"s3vectors:{action}" for action in actions)
    return feature_unavailable_guard(
        FEATURE,
        missing=f"{permissions} on the vector bucket set in 'aws_s3_vectors_bucket'",
        unreachable=_UNREACHABLE_DETAIL,
    )


def vectors_client() -> Any:  # noqa: ANN401 - no published stubs for this client
    """Return the client pinned to the region holding the vector bucket."""
    return get_client("s3vectors", SETTINGS.aws_s3_vectors_region)


def _bucket() -> str:
    """Return the configured vector bucket.

    Returns:
        The vector bucket name.

    Raises:
        FeatureUnavailableError: When no vector bucket is configured (503).
    """
    bucket = SETTINGS.aws_s3_vectors_bucket
    if not bucket:
        _raise_unconfigured()
    return bucket


def _raise_unconfigured() -> NoReturn:
    """Raise the 503 answering a deployment with no vector bucket.

    Raises:
        FeatureUnavailableError: Always (503).
    """
    raise FeatureUnavailableError(
        FEATURE,
        "Vector storage not configured (aws_s3_vectors_bucket): "
        "the Vector Stores API is disabled.",
    )


def _caller_attributes(metadata: JsonMapping) -> Attributes:
    """Return the caller attributes carried by a vector's metadata."""
    attributes: Attributes = {}
    for key, value in metadata.items():
        if key.startswith(_ATTRIBUTE_PREFIX) and isinstance(
            value, bool | float | int | str
        ):
            attributes[key[len(_ATTRIBUTE_PREFIX) :]] = _attribute_value(value)
    return attributes


def _attribute_value(value: JsonValue) -> AttributeValue:
    """Return *value* as one of the types an attribute may hold."""
    return value if isinstance(value, bool | str) else float(value)  # type: ignore[arg-type]


def _to_vector(entry: JsonMapping) -> IndexVector:
    """Return the chunk one stored vector carries."""
    metadata: JsonMapping = entry.get("metadata") or {}  # type: ignore[assignment]
    data: JsonMapping = entry.get("data") or {}  # type: ignore[assignment]
    embedding: list[float] = data.get("float32") or []  # type: ignore[assignment]
    return IndexVector(
        key=str(entry.get("key", "")),
        file_id=str(metadata.get(_FILE_ID_KEY, "")),
        filename=str(metadata.get(_FILENAME_KEY, "")),
        chunk_index=int(metadata.get(_CHUNK_INDEX_KEY, 0)),  # type: ignore[arg-type]
        text=str(metadata.get(_TEXT_KEY, "")),
        attributes=_caller_attributes(metadata),
        embedding=embedding,
    )


def _to_payload(vector: IndexVector) -> dict[str, Any]:
    """Return the index payload of one chunk."""
    return {
        "key": vector.key,
        "data": {"float32": vector.embedding},
        "metadata": {
            _TEXT_KEY: vector.text,
            _FILENAME_KEY: vector.filename,
            _FILE_ID_KEY: vector.file_id,
            _CHUNK_INDEX_KEY: vector.chunk_index,
            **{attribute_key(key): value for key, value in vector.attributes.items()},
        },
    }


class S3VectorsIndex:
    """Amazon S3 Vectors serving the vector index of every store."""

    __slots__ = ()

    @property
    def capabilities(self) -> IndexCapabilities:
        """What this backend can express."""
        return _CAPABILITIES

    def check_configured(self) -> None:
        """Raise when no vector bucket is configured.

        Raises:
            FeatureUnavailableError: When the deployment has no vector bucket (503).
        """
        if not SETTINGS.aws_s3_vectors_bucket:
            _raise_unconfigured()

    def check_attributes(self, attributes: Attributes) -> None:
        """Reject attributes that exceed the searchable-attribute budget.

        Args:
            attributes: The caller-supplied attributes.

        Raises:
            ApiError: When the attributes do not fit the per-file budget (400).
        """
        if not attributes:
            return
        size = len(to_json_bytes({attribute_key(k): v for k, v in attributes.items()}))
        if size > _MAX_FILTERABLE_BYTES:
            msg = (
                f"The 'attributes' of this file take {size} bytes, above the "
                f"{_MAX_FILTERABLE_BYTES}-byte limit for searchable attributes. "
                "Use fewer keys, or shorter values."
            )
            raise ApiError(msg)

    async def create_index(self, store_id: str, *, dimensions: int) -> None:
        """Create the index backing a new vector store.

        Args:
            store_id: A validated vector store identifier.
            dimensions: Length of the vectors it will hold.
        """
        with _vectors_guard("CreateIndex"):
            await vectors_client().create_index(
                vectorBucketName=_bucket(),
                indexName=index_name(store_id),
                dataType=_DATA_TYPE,
                dimension=dimensions,
                distanceMetric=_DISTANCE_METRIC,
                metadataConfiguration={
                    "nonFilterableMetadataKeys": list(_NON_FILTERABLE_KEYS)
                },
            )

    async def delete_index(self, store_id: str) -> None:
        """Delete the index backing *store_id*, ignoring an already-deleted one.

        Args:
            store_id: A validated vector store identifier.
        """
        with _vectors_guard("DeleteIndex"):
            try:
                await vectors_client().delete_index(
                    vectorBucketName=_bucket(), indexName=index_name(store_id)
                )
            except ClientError as exc:
                if exc.response["Error"]["Code"] != "NotFoundException":
                    raise

    async def put_vectors(
        self, store_id: str, vectors: AsyncIterable[IndexVector]
    ) -> None:
        """Write *vectors* into the index of *store_id*, one batch at a time.

        Args:
            store_id: A validated vector store identifier.
            vectors: The chunks to write, in document order.
        """
        client = vectors_client()
        bucket = _bucket()
        name = index_name(store_id)
        batch: list[dict[str, Any]] = []
        async for vector in vectors:
            batch.append(_to_payload(vector))
            if len(batch) >= _PUT_VECTORS_BATCH:
                with _vectors_guard("PutVectors"):
                    await client.put_vectors(
                        vectorBucketName=bucket, indexName=name, vectors=batch
                    )
                batch = []
        if batch:
            with _vectors_guard("PutVectors"):
                await client.put_vectors(
                    vectorBucketName=bucket, indexName=name, vectors=batch
                )

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
        client = vectors_client()
        bucket = _bucket()
        name = index_name(store_id)
        vectors: list[IndexVector] = []
        for start in range(0, len(keys), _GET_VECTORS_BATCH):
            with _vectors_guard("GetVectors"):
                response = await client.get_vectors(
                    vectorBucketName=bucket,
                    indexName=name,
                    keys=list(keys[start : start + _GET_VECTORS_BATCH]),
                    returnData=with_embeddings,
                    returnMetadata=True,
                )
            vectors.extend(_to_vector(entry) for entry in response.get("vectors", ()))
        return vectors

    async def delete_vectors(self, store_id: str, keys: Sequence[str]) -> None:
        """Remove the chunks stored under *keys*, best effort.

        Args:
            store_id: A validated vector store identifier.
            keys: The chunk keys to remove.
        """
        client = vectors_client()
        for start in range(0, len(keys), _DELETE_VECTORS_BATCH):
            # The guard's own error is suppressed too: its warning is the report.
            with (
                suppress(ApiError, BotoCoreError, ClientError),
                _vectors_guard("DeleteVectors"),
            ):
                await client.delete_vectors(
                    vectorBucketName=_bucket(),
                    indexName=index_name(store_id),
                    keys=list(keys[start : start + _DELETE_VECTORS_BATCH]),
                )

    async def query(
        self,
        store_id: str,
        embeddings: Sequence[Sequence[float]],
        *,
        max_results: int,
        search_filter: SearchFilter | None,
    ) -> list[VectorMatch]:
        """Return the chunks closest to *embeddings*.

        Args:
            store_id: A validated vector store identifier.
            embeddings: One vector per query text.
            max_results: Maximum matches to return per query.
            search_filter: Restriction over the files' attributes, if any.

        Returns:
            The matches of every query, in no particular order.
        """
        index_filter = (
            translate_filter(search_filter) if search_filter is not None else None
        )
        client = vectors_client()
        arguments: list[dict[str, Any]] = []
        for embedding in embeddings:
            query: dict[str, Any] = {
                "vectorBucketName": _bucket(),
                "indexName": index_name(store_id),
                "topK": max_results,
                "queryVector": {"float32": list(embedding)},
                "returnMetadata": True,
                "returnDistance": True,
            }
            if index_filter is not None:
                query["filter"] = index_filter
            arguments.append(query)
        with _vectors_guard("QueryVectors"):
            responses = await gather_bounded(
                [client.query_vectors(**query) for query in arguments], _QUERY_WAVE
            )
        matches: list[VectorMatch] = []
        for response in responses:
            for hit in response.get("vectors", ()):
                vector = _to_vector(hit)
                matches.append(
                    VectorMatch(
                        key=vector.key,
                        score=score_from_distance(hit.get("distance", 1.0)),
                        file_id=vector.file_id,
                        filename=vector.filename,
                        text=vector.text,
                        attributes=vector.attributes,
                    )
                )
        return matches
