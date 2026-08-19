"""Tests for the OpenAI-compatible ``/v1/vector_stores`` routes.

A vector store is a vector index plus the bookkeeping the API answers with, so
these tests split in three: the offline unit ones pin the pure translations the
engine performs (chunking, filter translation, distance-to-score, identifiers,
the conditional-update loop); the offline route and engine ones drive the whole
surface against an in-memory stand-in for the record bucket and the vector
index, so the sixteen routes and the asynchronous indexer run without
credentials; and the live ones exercise the full attach → index → search round
trip against the real backend.

Indexing is asynchronous, exactly as upstream: every test that needs a file to
be searchable polls until the store reports it ``completed``.

Ref: https://platform.openai.com/docs/api-reference/vector-stores
     https://stdapi.ai/api_openai_vector_stores/
     stdapi/routes/openai_vector_stores.py
     stdapi/vector_stores/engine.py
"""

import time
from asyncio import CancelledError, Event, create_task, gather, run, sleep
from dataclasses import replace
from itertools import count
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast
from zlib import crc32

import pytest
from botocore.exceptions import ClientError
from botocore.session import get_session as botocore_session
from fastapi.exceptions import RequestValidationError
from openai import NotFoundError, OpenAI
from pydantic import ValidationError
from pydantic_core import from_json

from stdapi import vector_stores
from stdapi.api_errors import ApiError
from stdapi.cleanup import CLEANUPS
from stdapi.config import AWS_SESSION, SETTINGS
from stdapi.models.embedding import EmbeddingResponse
from stdapi.models.embedding.cohere_embed import EmbeddingModel as CohereEmbeddingModel
from stdapi.types.openai_vector_stores import (
    ComparisonFilter,
    CompoundFilter,
    FileBatchCreateParams,
    FileBatchFile,
    StaticChunkingConfig,
)
from stdapi.utils import to_json_str, webuuid
from stdapi.vector_stores import (
    BatchRecord,
    FileCountsRecord,
    FileRecord,
    PendingFile,
    StoreRecord,
    attach_files,
    cancel_batch,
    chunk_text,
    create_store,
    delete_store,
    detach_file,
    drain_indexing,
    engine,
    index_files,
    jobs,
    list_store_files,
    new_batch_id,
    new_store_id,
    read_batch,
    read_file,
    read_file_chunks,
    read_store,
    records,
    s3_vectors,
    search,
    start_indexing,
    touch_store,
    update_file_attributes,
    update_record,
    vector_key,
)
from stdapi.vector_stores.backend import (
    IndexCapabilities,
    IndexVector,
    VectorMatch,
    parse_filter,
    unsupported_file_message,
)
from stdapi.vector_stores.jobs import MAX_JOB_FILES, IndexFilesJob
from stdapi.vector_stores.knowledge_base import CAPABILITIES as _KB_CAPABILITIES
from stdapi.vector_stores.records import batch_key, file_key, read_record, store_key
from stdapi.vector_stores.s3_vectors import (
    S3VectorsIndex,
    attribute_key,
    index_name,
    score_from_distance,
    translate_filter,
)
from tests._helpers import make_client_error, red_png
from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterable,
        AsyncIterator,
        Awaitable,
        Callable,
        Coroutine,
        Iterator,
        Sequence,
    )

    from botocore.exceptions import ClientError
    from openai.types import VectorStore
    from openai.types.vector_stores import VectorStoreFile
    from starlette.testclient import TestClient

    from stdapi.types.openai_vector_stores import Attributes, SearchFilter


#: The vector store namespace is account-wide: keep this module on one worker.
pytestmark = pytest.mark.xdist_group("openai_vector_stores")

#: A planted sentence no other test content answers, for the search assertions.
_PLANTED = "The maintenance hatch of the Kelvin observatory opens with code QUINCEY-7."

#: What S3 Vectors answers when the server role lacks the action it used.
_ACCESS_DENIED_CODE = "AccessDeniedException"

#: File content built around the planted sentence, long enough to chunk.
_TEXT_FILE: bytes = (
    "Observatory operations manual.\n\n"
    "Section one covers the daily calibration of the primary mirror.\n\n"
    f"{_PLANTED}\n\n"
    "Section three covers the archival of nightly exposure plates.\n"
).encode()

#: A second file, so filter and multi-file assertions have something to exclude.
_OTHER_FILE: bytes = b"Cafeteria rota. Monday soup, Tuesday stew, Wednesday pie.\n"

#: The comparison operators a search filter accepts.
type ComparisonOperator = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"]

#: Seconds a test waits for asynchronous indexing to settle.
_INDEX_TIMEOUT = 180.0

#: Seconds a test waits for an eventually consistent listing to catch up.
_LIST_TIMEOUT = 120.0

#: What the shipped backend declares, for the tests pinning its dialect.
_S3_CAPABILITIES = S3VectorsIndex().capabilities


def _wait_for_listed_files(client: OpenAI, store_id: str, expected: set[str]) -> None:
    """Poll the attached-file listing until it holds exactly *expected*.

    Args:
        client: OpenAI SDK client bound to the target under test.
        store_id: The vector store to list.
        expected: The file IDs the listing must report.
    """
    deadline = time.monotonic() + _LIST_TIMEOUT
    while True:
        listed = {
            entry.id
            for entry in client.vector_stores.files.list(store_id, limit=100).data
        }
        if listed == expected:
            return
        assert time.monotonic() < deadline, (
            f"attached files never listed: {listed} != {expected}"
        )
        time.sleep(2.0)


#: The ``object`` a file batch carries; the installed SDK's Literal has it wrong.
_BATCH_OBJECT: str = "vector_store.file_batch"


def _precondition_failed() -> ClientError:
    """Return the error S3 raises when a conditional write loses the race."""
    return make_client_error("PreconditionFailed", "PutObject")


def _wait_for_store(
    client: OpenAI, store_id: str, *, files: int = 0, timeout: float = _INDEX_TIMEOUT
) -> VectorStore:
    """Poll a vector store until every attached file has settled.

    The counters are eventually consistent on both targets: a store can report
    ``completed`` before the files it was created with are counted, so the
    expected total is part of the wait.

    Args:
        client: OpenAI SDK client bound to the target under test.
        store_id: The vector store to poll.
        files: Number of files the store must end up counting.
        timeout: Seconds to wait before failing.

    Returns:
        The settled vector store.
    """
    deadline = time.monotonic() + timeout
    while True:
        store = client.vector_stores.retrieve(store_id)
        settled = store.file_counts.in_progress == 0 and store.status != "in_progress"
        if settled and store.file_counts.total >= files:
            return store
        assert time.monotonic() < deadline, (
            f"Vector store {store_id} still indexing after {timeout}s: "
            f"{store.file_counts}"
        )
        time.sleep(2.0)


def _wait_for_file(
    client: OpenAI, store_id: str, file_id: str, timeout: float = _INDEX_TIMEOUT
) -> VectorStoreFile:
    """Poll one attached file until it leaves ``in_progress``.

    Args:
        client: OpenAI SDK client bound to the target under test.
        store_id: The vector store the file belongs to.
        file_id: The attached file to poll.
        timeout: Seconds to wait before failing.

    Returns:
        The settled vector store file.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            attached = client.vector_stores.files.retrieve(
                file_id, vector_store_id=store_id
            )
        except NotFoundError:
            # Not visible yet: the listing behind it is eventually consistent.
            attached = None
        if attached is not None and attached.status != "in_progress":
            return attached
        assert time.monotonic() < deadline, (
            f"File {file_id} still indexing after {timeout}s"
        )
        time.sleep(2.0)


def _upload(client: OpenAI, name: str, content: bytes) -> str:
    """Upload a file for indexing and return its ID.

    Args:
        client: OpenAI SDK client bound to the target under test.
        name: File name to store it under.
        content: The file bytes.

    Returns:
        The uploaded file ID.
    """
    return client.files.create(
        file=(name, content, "text/plain"), purpose="assistants"
    ).id


@pytest.fixture(scope="module")
def indexed_store(openai_client: OpenAI) -> Iterator[tuple[str, str, str]]:
    """A vector store holding two indexed text files.

    Module-scoped: creating a store, uploading and indexing files costs a real
    embedding call per chunk, so every read-only assertion shares one.

    Yields:
        ``(vector_store_id, planted_file_id, other_file_id)``.
    """
    planted = _upload(openai_client, "observatory.txt", _TEXT_FILE)
    other = _upload(openai_client, "cafeteria.txt", _OTHER_FILE)
    store = openai_client.vector_stores.create(
        name="stdapi-test-indexed",
        file_ids=[planted, other],
        chunking_strategy={
            "type": "static",
            "static": {"max_chunk_size_tokens": 100, "chunk_overlap_tokens": 20},
        },
    )
    try:
        _wait_for_store(openai_client, store.id, files=2)
        _wait_for_file(openai_client, store.id, planted)
        _wait_for_file(openai_client, store.id, other)
        yield store.id, planted, other
    finally:
        openai_client.vector_stores.delete(store.id)
        for file_id in (planted, other):
            openai_client.files.delete(file_id)


@pytest.fixture
def empty_store(openai_client: OpenAI) -> Iterator[str]:
    """A vector store with no file attached, deleted at the end of the test.

    Yields:
        The vector store ID.
    """
    store = openai_client.vector_stores.create(name="stdapi-test-empty")
    try:
        yield store.id
    finally:
        openai_client.vector_stores.delete(store.id)


@pytest.mark.local
class TestChunking:
    """The chunker turns a token budget into overlapping text slices.

    Ref: stdapi/vector_stores/engine.py:chunk_text
    """

    def test_short_text_is_one_chunk(self) -> None:
        """Text below the budget is not split.

        Ref: stdapi/vector_stores/engine.py:chunk_text
        """
        assert chunk_text("hello world", 100, 20, 0, 0) == ["hello world"]

    def test_long_text_is_split_and_overlaps(self) -> None:
        """A long text yields several chunks whose ends overlap.

        Ref: stdapi/vector_stores/engine.py:chunk_text
        """
        text = " ".join(f"word{i:04d}" for i in range(400))
        chunks = chunk_text(text, 100, 50, 0, 0)
        assert len(chunks) > 1
        # 100 tokens is approximated as 400 characters.
        assert all(len(chunk) <= 400 for chunk in chunks)
        assert chunks[1].split()[0] in chunks[0]

    def test_cut_falls_on_a_word_boundary(self) -> None:
        """A chunk does not end mid-word when a separator is within reach.

        Ref: stdapi/vector_stores/engine.py:chunk_text
        """
        text = " ".join("alpha" for _ in range(300))
        for chunk in chunk_text(text, 100, 0, 0, 0):
            assert set(chunk.split()) == {"alpha"}

    def test_chunk_is_clamped_to_the_model_character_limit(self) -> None:
        """A model that rejects long inputs caps the chunk regardless of the token budget.

        Ref: stdapi/models/embedding/cohere_embed.py:EmbeddingModel.max_input_characters
        """
        text = "x" * 20000
        assert all(len(chunk) <= 2048 for chunk in chunk_text(text, 4096, 0, 2048, 0))

    def test_whitespace_only_text_yields_no_chunk(self) -> None:
        """A file holding only whitespace produces nothing to index.

        Ref: stdapi/vector_stores/engine.py:chunk_text
        """
        assert chunk_text("   \n\n\t  ", 100, 20, 0, 0) == []

    def test_every_chunk_fits_the_per_vector_text_budget(self) -> None:
        """A chunk of multi-byte characters is split to fit the backend's text budget.

        Ref: stdapi/vector_stores/engine.py:_split_on_bytes
        """
        budget = _S3_CAPABILITIES.max_chunk_bytes
        assert budget == 32768
        chunks = chunk_text("é" * 40000, 4096, 0, 0, budget)
        assert chunks
        assert all(len(chunk.encode()) <= budget for chunk in chunks)


@pytest.mark.local
class TestFilterTranslation:
    """Every upstream filter operator has an index equivalent.

    Ref: openai.types.shared_params.comparison_filter.ComparisonFilter
         stdapi/vector_stores/s3_vectors.py:translate_filter
    """

    @pytest.mark.parametrize(
        ("operator", "expected", "value"),
        [
            ("eq", "$eq", "physics"),
            ("ne", "$ne", "physics"),
            # The index orders numbers only, so these four take a number.
            ("gt", "$gt", 2020),
            ("gte", "$gte", 2020),
            ("lt", "$lt", 2020),
            ("lte", "$lte", 2020),
            ("in", "$in", ["physics", "chemistry"]),
            ("nin", "$nin", ["physics", "chemistry"]),
        ],
    )
    def test_comparison_operators(
        self, operator: ComparisonOperator, expected: str, value: object
    ) -> None:
        """Each comparison operator maps to its index operator, under the attribute key.

        Ref: stdapi/vector_stores/s3_vectors.py:translate_filter
        """
        search_filter = ComparisonFilter(key="topic", type=operator, value=value)  # type: ignore[arg-type]
        translated = translate_filter(search_filter)
        assert translated == {attribute_key("topic"): {expected: search_filter.value}}

    @pytest.mark.parametrize("operator", ["gt", "gte", "lt", "lte"])
    def test_ordering_operators_refuse_a_non_numeric_bound(
        self, operator: ComparisonOperator
    ) -> None:
        """A string bound is refused here rather than by the index.

        The index applies these four to numbers only, so forwarding a string
        would answer with the backend's own message.

        Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-metadata-filtering.html
        """
        for value in ("2026-01-01", True):
            with pytest.raises(ValidationError):
                ComparisonFilter(key="date", type=operator, value=value)

    @pytest.mark.parametrize("operator", ["and", "or"])
    def test_compound_operators(self, operator: Literal["and", "or"]) -> None:
        """A compound filter nests its translated members.

        Ref: stdapi/vector_stores/s3_vectors.py:translate_filter
        """
        translated = translate_filter(
            CompoundFilter(
                type=operator,
                filters=[
                    ComparisonFilter(key="a", type="eq", value=1),
                    ComparisonFilter(key="b", type="ne", value=True),
                ],
            )
        )
        assert translated == {
            f"${operator}": [
                {attribute_key("a"): {"$eq": 1.0}},
                {attribute_key("b"): {"$ne": True}},
            ]
        }

    def test_nested_compound_filters_are_translated(self) -> None:
        """A compound filter nested inside another translates to the same depth.

        Nesting is carried as a plain object rather than a self-referential
        schema, so this is the path that proves the deeper levels still work.

        Ref: stdapi/vector_stores/s3_vectors.py:translate_filter
        """
        translated = translate_filter(
            CompoundFilter(
                type="and",
                filters=[
                    ComparisonFilter(key="a", type="eq", value="x"),
                    {
                        "type": "or",
                        "filters": [
                            {"key": "b", "type": "gt", "value": 1},
                            {"key": "c", "type": "in", "value": ["p", "q"]},
                        ],
                    },
                ],
            )
        )
        assert translated == {
            "$and": [
                {attribute_key("a"): {"$eq": "x"}},
                {
                    "$or": [
                        {attribute_key("b"): {"$gt": 1.0}},
                        {attribute_key("c"): {"$in": ["p", "q"]}},
                    ]
                },
            ]
        }

    def test_a_malformed_nested_filter_is_a_request_error(self) -> None:
        """A nested entry that is not a filter is reported as an invalid request.

        Ref: stdapi/vector_stores/s3_vectors.py:translate_filter
        """
        with pytest.raises(RequestValidationError):
            translate_filter(CompoundFilter(type="and", filters=[{"not": "a filter"}]))

    def test_attribute_keys_cannot_collide_with_stored_content(self) -> None:
        """An attribute named like a reserved key is namespaced away from it.

        The chunk text and the source file name are stored on the same vector,
        so a caller attribute must not be able to overwrite them.

        Ref: stdapi/vector_stores/s3_vectors.py:attribute_key
        """
        assert attribute_key("_text") != "_text"
        assert attribute_key("_filename") != "_filename"
        assert attribute_key("a") != attribute_key("b")


@pytest.mark.local
class TestScoreMapping:
    """Cosine distance becomes the similarity score the API reports.

    Ref: stdapi/vector_stores/s3_vectors.py:score_from_distance
    """

    @pytest.mark.parametrize(
        ("distance", "score"),
        [(0.0, 1.0), (1.0, 0.0), (2.0, 0.0), (0.2928932309150696, 0.7071067690849304)],
    )
    def test_distance_to_score(self, distance: float, score: float) -> None:
        """An identical vector scores 1, an orthogonal one 0, an opposite one 0.

        The distances are the ones measured against the real index for unit
        vectors at 0, 90, 180 and 45 degrees.

        Ref: stdapi/vector_stores/s3_vectors.py:score_from_distance
        """
        assert score_from_distance(distance) == pytest.approx(score)

    def test_score_never_leaves_the_unit_range(self) -> None:
        """A score is always reportable against ``ranking_options.score_threshold``.

        Ref: stdapi/vector_stores/s3_vectors.py:score_from_distance
        """
        assert score_from_distance(-0.5) == 1.0
        assert score_from_distance(3.0) == 0.0


@pytest.mark.local
class TestIdentifiers:
    """Store identifiers map to index names reversibly and sort by creation time.

    Ref: stdapi/vector_stores/s3_vectors.py:index_name
    """

    def test_index_name_is_derived_from_the_store_id(self) -> None:
        """The identifier's separator is the only character that changes.

        Ref: stdapi/vector_stores/s3_vectors.py:index_name
        """
        store_id = new_store_id()
        name = index_name(store_id)
        assert name == store_id.replace("_", "-", 1)
        assert name.replace("-", "_", 1) == store_id

    def test_index_name_is_accepted_by_the_index_naming_rules(self) -> None:
        """The derived name has no underscore, uppercase or dot, and fits 3-63 characters.

        Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-limitations.html
        """
        name = index_name(new_store_id())
        assert 3 <= len(name) <= 63
        assert name == name.lower()
        assert set(name) <= set("abcdefghijklmnopqrstuvwxyz0123456789-")

    def test_identifiers_sort_by_creation_time(self) -> None:
        """A later identifier sorts after an earlier one, which the listing relies on.

        Ref: stdapi/vector_stores/engine.py:new_store_id
        """
        first = new_store_id()
        time.sleep(0.005)
        assert new_store_id() > first
        first_batch = new_batch_id()
        time.sleep(0.005)
        assert new_batch_id() > first_batch

    def test_malformed_identifiers_are_rejected(self) -> None:
        """An identifier that is not one of ours never reaches a backend call.

        Ref: stdapi/vector_stores/engine.py:parse_store_id
        """
        for candidate in ("vs_", "vs-abc", "VS_ABCDEFGHIJKLMNOPQRSTUVWXYZ", "../etc"):
            with pytest.raises(ApiError) as raised:
                vector_stores.parse_store_id(candidate)
            assert raised.value.status == 404


@pytest.mark.local
class TestRequestValidation:
    """The upstream request bounds are enforced before any backend work.

    Ref: openai.types.static_file_chunking_strategy.StaticFileChunkingStrategy
    """

    @pytest.mark.parametrize("size", [99, 4097])
    def test_chunk_size_outside_bounds_is_rejected(self, size: int) -> None:
        """``max_chunk_size_tokens`` accepts 100 to 4096 only.

        Ref: openai.types.static_file_chunking_strategy.StaticFileChunkingStrategy
        """
        with pytest.raises(ValidationError):
            StaticChunkingConfig(max_chunk_size_tokens=size, chunk_overlap_tokens=0)

    def test_overlap_above_half_the_chunk_is_rejected(self) -> None:
        """The overlap must not exceed half of ``max_chunk_size_tokens``.

        Ref: openai.types.static_file_chunking_strategy.StaticFileChunkingStrategy
        """
        with pytest.raises(ValidationError):
            StaticChunkingConfig(max_chunk_size_tokens=200, chunk_overlap_tokens=101)

    def test_batch_requires_exactly_one_file_list(self) -> None:
        """``file_ids`` and ``files`` are mutually exclusive and one is required.

        Ref: openai.types.vector_stores.file_batch_create_params.FileBatchCreateParams
        """
        with pytest.raises(ValidationError):
            FileBatchCreateParams()
        with pytest.raises(ValidationError):
            FileBatchCreateParams(
                file_ids=["file-" + "0" * 32],
                files=[FileBatchFile(file_id="file-" + "0" * 32)],
            )

    def test_attributes_above_the_searchable_budget_are_rejected(self) -> None:
        """Attributes valid upstream can exceed what stays searchable here, and say so.

        Upstream allows 16 keys of up to 512 characters; the searchable budget
        is smaller, so the request is refused with the limit named rather than
        the file failing later.

        Ref: stdapi/vector_stores/engine.py:check_attributes
        """
        with pytest.raises(ApiError) as raised:
            vector_stores.check_attributes({f"k{i}": "v" * 512 for i in range(16)})
        assert raised.value.status == 400
        assert "2048" in str(raised.value)

    def test_attributes_within_the_budget_are_accepted(self) -> None:
        """A realistic attribute set passes untouched.

        Ref: stdapi/vector_stores/engine.py:check_attributes
        """
        vector_stores.check_attributes({"topic": "physics", "year": 2026.0})


@pytest.mark.local
class TestExpiryComputation:
    """``expires_at`` and ``status`` follow ``expires_after`` and ``last_active_at``.

    Ref: openai.types.vector_store.VectorStore
    """

    def test_expires_at_is_the_anchor_plus_the_days(self) -> None:
        """The expiration is counted from the last activity, as upstream anchors it.

        Ref: openai.types.vector_store.ExpiresAfter
        """
        record = StoreRecord(
            id=new_store_id(),
            created_at=1_000_000,
            last_active_at=1_000_000,
            embedding_model="m",
            dimensions=8,
            expires_after_days=2,
        )
        assert record.expires_at == 1_000_000 + 2 * 86400

    def test_no_policy_means_no_expiry(self) -> None:
        """A store without a policy never expires.

        Ref: openai.types.vector_store.VectorStore
        """
        record = StoreRecord(
            id=new_store_id(),
            created_at=1,
            last_active_at=1,
            embedding_model="m",
            dimensions=8,
        )
        assert record.expires_at is None
        assert record.status == "completed"

    def test_a_past_expiry_reads_back_as_expired(self) -> None:
        """A store past its expiration reports ``expired``, whatever its counters say.

        Ref: openai.types.vector_store.VectorStore
        """
        record = StoreRecord(
            id=new_store_id(),
            created_at=1,
            last_active_at=1,
            embedding_model="m",
            dimensions=8,
            expires_after_days=1,
            file_counts=FileCountsRecord(in_progress=1),
        )
        assert record.status == "expired"

    def test_a_store_still_indexing_is_in_progress(self) -> None:
        """The store is ``in_progress`` while any file is.

        Ref: openai.types.vector_store.VectorStore
        """
        record = StoreRecord(
            id=new_store_id(),
            created_at=1,
            last_active_at=1,
            embedding_model="m",
            dimensions=8,
            file_counts=FileCountsRecord(in_progress=1, completed=1),
        )
        assert record.status == "in_progress"
        assert record.file_counts.total == 2


class _FakeBody:
    """The streaming body an S3 ``GetObject`` answers with."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        """Return the whole object body."""
        return self._data


class _FakeObjectStore:
    """In-memory stand-in for the S3 bucket holding the bookkeeping records.

    Conditional writes are honoured, so the compare-and-swap loop the counters
    rely on is exercised rather than stubbed.
    """

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.etags: dict[str, str] = {}
        #: Errors to raise once, keyed by ``(operation, key)``.
        self.fail_once: dict[tuple[str, str], ClientError] = {}
        #: Conditional writes another writer wins first, keyed by object key.
        self.lose_writes: dict[str, int] = {}
        #: Keys returned per listing page, whatever ``MaxKeys`` asked for.
        self.page_size: int = 0
        self.reads = 0
        self.writes = 0
        self._counter = 0

    def _scheduled_error(self, operation: str, key: str) -> None:
        """Raise the error scheduled for *operation* on *key*, if any."""
        error = self.fail_once.pop((operation, key), None)
        if error is not None:
            raise error

    async def get_object(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        key = params["Key"]
        self.reads += 1
        self._scheduled_error("get_object", key)
        if key not in self.objects:
            missing = make_client_error("NoSuchKey", "GetObject")
            raise missing
        return {"Body": _FakeBody(self.objects[key]), "ETag": self.etags[key]}

    async def put_object(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        key = params["Key"]
        self.writes += 1
        self._scheduled_error("put_object", key)
        if self.lose_writes.get(key):
            # Another writer landed first: the etag the caller held is stale.
            self.lose_writes[key] -= 1
            self._counter += 1
            self.etags[key] = f'"etag-{self._counter}"'
            raise _precondition_failed()
        if "IfNoneMatch" in params and key in self.objects:
            raise _precondition_failed()
        if "IfMatch" in params and self.etags.get(key) != params["IfMatch"]:
            raise _precondition_failed()
        self._counter += 1
        self.objects[key] = params["Body"]
        self.etags[key] = f'"etag-{self._counter}"'
        return {}

    async def delete_object(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.objects.pop(params["Key"], None)
        self.etags.pop(params["Key"], None)
        return {}

    async def list_objects_v2(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        prefix = params.get("Prefix", "")
        limit = params.get("MaxKeys", 1000)
        if self.page_size:
            # S3 may answer with fewer keys than asked for; the caller must page.
            limit = min(limit, self.page_size)
        keys = sorted(key for key in self.objects if key.startswith(prefix))
        start = params.get("StartAfter") or params.get("ContinuationToken") or ""
        if start:
            keys = [key for key in keys if key > start]
        delimiter = params.get("Delimiter")
        if delimiter:
            prefixes = sorted(
                {
                    prefix + key[len(prefix) :].split(delimiter, 1)[0] + delimiter
                    for key in keys
                    if delimiter in key[len(prefix) :]
                }
            )
            return {
                "CommonPrefixes": [{"Prefix": entry} for entry in prefixes[:limit]],
                "IsTruncated": len(prefixes) > limit,
            }
        page = keys[:limit]
        return {
            "Contents": [{"Key": key} for key in page],
            "IsTruncated": len(keys) > limit,
            "NextContinuationToken": page[-1] if page else "",
        }


def _cosine_distance(left: list[float], right: list[float]) -> float:
    """Return the distance the index reports between two unit vectors."""
    return 1.0 - sum(a * b for a, b in zip(left, right, strict=True))


def _matches_filter(index_filter: dict[str, Any], metadata: dict[str, Any]) -> bool:
    """Evaluate a translated index filter against one vector's metadata.

    Args:
        index_filter: The filter as ``translate_filter`` produced it.
        metadata: The metadata stored with the vector.

    Returns:
        Whether the vector matches.
    """
    if "$and" in index_filter:
        return all(_matches_filter(entry, metadata) for entry in index_filter["$and"])
    if "$or" in index_filter:
        return any(_matches_filter(entry, metadata) for entry in index_filter["$or"])
    for key, condition in index_filter.items():
        value = metadata.get(key)
        for operator, expected in condition.items():
            if operator in ("$gt", "$gte", "$lt", "$lte") and not isinstance(
                value, float | int
            ):
                return False
            if not _COMPARISONS[operator](value, expected):
                return False
    return True


#: How the index applies each translated comparison operator.
_COMPARISONS: dict[str, Callable[[Any, Any], bool]] = {
    "$eq": lambda value, expected: value == expected,
    "$ne": lambda value, expected: value != expected,
    "$gt": lambda value, expected: value > expected,
    "$gte": lambda value, expected: value >= expected,
    "$lt": lambda value, expected: value < expected,
    "$lte": lambda value, expected: value <= expected,
    "$in": lambda value, expected: value in expected,
    "$nin": lambda value, expected: value not in expected,
}


class _FakeS3VectorsClient:
    """In-memory stand-in for the S3 Vectors service the shipped backend calls.

    Substituted at the AWS client, not at the backend, so ``S3VectorsIndex``
    itself — its metadata layout, filter dialect, score conversion and IAM
    guards — is the code the offline route and engine tests exercise.
    """

    def __init__(self) -> None:
        self.indexes: dict[str, dict[str, dict[str, Any]]] = {}
        self.deleted: list[str] = []

    def _index(self, name: str) -> dict[str, dict[str, Any]]:
        """Return an index, raising the AWS not-found error when absent."""
        try:
            return self.indexes[name]
        except KeyError:
            missing = make_client_error("NotFoundException", "QueryVectors")
            raise missing from None

    async def create_index(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.indexes[params["indexName"]] = {}
        return {}

    async def delete_index(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        name = params["indexName"]
        if name not in self.indexes:
            missing = make_client_error("NotFoundException", "DeleteIndex")
            raise missing
        del self.indexes[name]
        self.deleted.append(name)
        return {}

    async def put_vectors(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        index = self._index(params["indexName"])
        for vector in params["vectors"]:
            index[vector["key"]] = {
                "data": vector["data"],
                "metadata": dict(vector["metadata"]),
            }
        return {}

    async def get_vectors(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        index = self._index(params["indexName"])
        return {
            "vectors": [
                {"key": key, **index[key]} for key in params["keys"] if key in index
            ]
        }

    async def delete_vectors(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        index = self._index(params["indexName"])
        for key in params["keys"]:
            index.pop(key, None)
        return {}

    async def query_vectors(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        index = self._index(params["indexName"])
        query = params["queryVector"]["float32"]
        index_filter = params.get("filter")
        hits = [
            {
                "key": key,
                "distance": _cosine_distance(query, vector["data"]["float32"]),
                "metadata": vector["metadata"],
            }
            for key, vector in index.items()
            if index_filter is None or _matches_filter(index_filter, vector["metadata"])
        ]
        hits.sort(key=lambda hit: hit["distance"])
        return {"vectors": hits[: params["topK"]]}


#: How a filter operator applies to a caller attribute, before any translation.
_UPSTREAM_COMPARISONS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": lambda value, expected: value == expected,
    "ne": lambda value, expected: value != expected,
    "gt": lambda value, expected: isinstance(value, float | int) and value > expected,
    "gte": lambda value, expected: isinstance(value, float | int) and value >= expected,
    "lt": lambda value, expected: isinstance(value, float | int) and value < expected,
    "lte": lambda value, expected: isinstance(value, float | int) and value <= expected,
    "in": lambda value, expected: value in expected,
    "nin": lambda value, expected: value not in expected,
}


def _upstream_filter_matches(
    search_filter: SearchFilter | dict[str, Any] | None, attributes: Attributes
) -> bool:
    """Evaluate an upstream search filter against a file's attributes.

    Args:
        search_filter: The filter as the API received it, or ``None``.
        attributes: The attributes stored with the file.

    Returns:
        Whether the file matches.
    """
    if search_filter is None:
        return True
    node = parse_filter(search_filter)
    if isinstance(node, CompoundFilter):
        inner = (_upstream_filter_matches(entry, attributes) for entry in node.filters)
        return all(inner) if node.type == "and" else any(inner)
    return _UPSTREAM_COMPARISONS[node.type](attributes.get(node.key), node.value)


class _FakeVectorIndex:
    """An in-memory implementation of the vector index contract.

    Written against the protocol rather than against any service, and taking
    its capabilities from the test, so it is what proves the engine holds
    nothing backend-specific and refuses what a backend does not declare.
    """

    def __init__(self, capabilities: IndexCapabilities) -> None:
        self._capabilities = capabilities
        #: Stored chunks, keyed by store then by vector key.
        self.indexes: dict[str, dict[str, IndexVector]] = {}
        #: Stores whose index was deleted, in order.
        self.deleted: list[str] = []
        #: Queries this backend was actually asked to run.
        self.queries = 0

    @property
    def capabilities(self) -> IndexCapabilities:
        """What this backend declares it can express."""
        return self._capabilities

    def check_configured(self) -> None:
        """Nothing to configure: the index is this object."""

    def check_attributes(self, attributes: Attributes) -> None:
        """Accept any attribute set; the per-vector budget is the service's."""

    async def create_index(self, store_id: str, *, dimensions: int) -> None:
        """Open an empty index for *store_id*."""
        del dimensions
        self.indexes[store_id] = {}

    async def delete_index(self, store_id: str) -> None:
        """Drop the index of *store_id*, ignoring an already-deleted one."""
        if self.indexes.pop(store_id, None) is not None:
            self.deleted.append(store_id)

    async def put_vectors(
        self, store_id: str, vectors: AsyncIterable[IndexVector]
    ) -> None:
        """Store every chunk the engine streams."""
        index = self.indexes[store_id]
        async for vector in vectors:
            index[vector.key] = replace(vector, attributes=dict(vector.attributes))

    async def get_vectors(
        self, store_id: str, keys: Sequence[str], *, with_embeddings: bool
    ) -> list[IndexVector]:
        """Return the stored chunks of *keys* that exist."""
        index = self.indexes[store_id]
        return [
            replace(
                index[key],
                attributes=dict(index[key].attributes),
                embedding=list(index[key].embedding) if with_embeddings else [],
            )
            for key in keys
            if key in index
        ]

    async def delete_vectors(self, store_id: str, keys: Sequence[str]) -> None:
        """Remove the chunks of *keys*, ignoring the ones already gone."""
        index = self.indexes.get(store_id, {})
        for key in keys:
            index.pop(key, None)

    async def query(
        self,
        store_id: str,
        embeddings: Sequence[Sequence[float]],
        *,
        max_results: int,
        search_filter: SearchFilter | None,
    ) -> list[VectorMatch]:
        """Return the closest chunks of every query, filter applied."""
        index = self.indexes[store_id]
        matches: list[VectorMatch] = []
        self.queries += len(embeddings)
        for embedding in embeddings:
            query = list(embedding)
            hits = [
                vector
                for vector in index.values()
                if _upstream_filter_matches(search_filter, vector.attributes)
            ]
            hits.sort(key=lambda vector: _cosine_distance(query, vector.embedding))
            matches.extend(
                VectorMatch(
                    key=vector.key,
                    score=score_from_distance(
                        _cosine_distance(query, vector.embedding)
                    ),
                    file_id=vector.file_id,
                    filename=vector.filename,
                    text=vector.text,
                    attributes=dict(vector.attributes),
                )
                for vector in hits[:max_results]
            )
        return matches


#: Dimension of the vectors the stub embedding model produces.
_STUB_DIMENSIONS = 8


def _stub_vector(text: str) -> list[float]:
    """Return a deterministic unit vector for *text*.

    Built from the words it holds, so two texts sharing words come out closer
    than two that share none — enough for the ranking assertions.
    """
    weights = [0.0] * _STUB_DIMENSIONS
    for word in text.lower().split():
        weights[crc32(word.encode()) % _STUB_DIMENSIONS] += 1.0
    norm = sum(weight * weight for weight in weights) ** 0.5 or 1.0
    return [weight / norm for weight in weights]


class _StubEmbeddingModel:
    """Deterministic stand-in for the configured embedding model."""

    def __init__(self) -> None:
        self.max_input_characters = 0
        #: Size of each ``embed_text`` call, so the wave batching is observable.
        self.waves: list[int] = []

    async def embed_text(
        self,
        inputs: list[Any],
        dimensions: int | None,
        extra_params: Any,  # noqa: ANN401
    ) -> EmbeddingResponse:
        """Embed every input with :func:`_stub_vector`."""
        del dimensions, extra_params
        self.waves.append(len(inputs))
        return EmbeddingResponse(
            embeddings=[_stub_vector(str(text)) for text in inputs]
        )


class _FakeBackend:
    """Every backend the Vector Stores API talks to, in memory."""

    def __init__(self) -> None:
        self.records = _FakeObjectStore()
        self.vectors = _FakeS3VectorsClient()
        self.model = _StubEmbeddingModel()
        #: Uploaded file content and content type, keyed by bare file payload.
        self.uploads: dict[str, tuple[bytes, str]] = {}
        #: ``(store_id, batch_id, file_ids)`` of each attach that would index.
        self.started: list[tuple[str, str, tuple[str, ...]]] = []

    def start_indexing(self, store_id: str, file_ids: list[str], batch_id: str) -> None:
        """Record the indexing an attach starts, so a test drives it itself."""
        self.started.append((store_id, batch_id, tuple(file_ids)))

    def upload(self, content: bytes, content_type: str = "text/plain") -> str:
        """Register an uploaded file and return the identifier attaching it."""
        payload = f"{len(self.uploads):032d}"
        self.uploads[payload] = (content, content_type)
        return f"file-{payload}"

    async def get_file(self, payload: str) -> SimpleNamespace:
        """Return the uploaded file's record, or answer 404 as the Files API does."""
        if payload not in self.uploads:
            msg = f"No file found with id 'file-{payload}'."
            raise ApiError(msg, status=404)
        return SimpleNamespace(filename=f"{payload}.txt")

    async def get_file_content(self, payload: str) -> tuple[Any, str]:
        """Return the uploaded file's content as the Files API streams it."""
        content, content_type = self.uploads[payload]

        async def _stream() -> AsyncIterator[bytes]:
            for start in range(0, len(content), 1024):
                yield content[start : start + 1024]

        return _stream(), content_type


@pytest.fixture
def vector_backend(monkeypatch: pytest.MonkeyPatch) -> _FakeBackend:
    """Serve the Vector Stores API from an in-memory backend.

    Only the boundaries are replaced — the AWS clients, the embedding model and
    the Files API — so the engine, the S3 Vectors backend, the routes and the
    conditional-update loop are the code actually under test.
    """
    backend = _FakeBackend()
    monkeypatch.setattr(SETTINGS, "aws_s3_bucket", "stdapi-test-records")
    monkeypatch.setattr(SETTINGS, "aws_s3_vectors_bucket", "stdapi-test-vectors")
    # The stores of this API and no other: a knowledge base the deployment
    # allowlisted is served from AWS, which this backend stands in for.
    monkeypatch.setattr(SETTINGS, "aws_bedrock_knowledge_base_ids", [])
    monkeypatch.setattr(records, "records_client", lambda: backend.records)
    monkeypatch.setattr(s3_vectors, "vectors_client", lambda: backend.vectors)
    monkeypatch.setattr(engine, "get_embedding_model", lambda _model_id: backend.model)

    async def _validate_model(model_id: str, _modality: str) -> SimpleNamespace:
        return SimpleNamespace(id=model_id)

    monkeypatch.setattr(engine, "validate_model", _validate_model)
    monkeypatch.setattr(engine, "get_file", backend.get_file)
    monkeypatch.setattr(engine, "get_file_content", backend.get_file_content)
    # Indexing is driven by the tests: a racing task is not deterministic.
    monkeypatch.setattr(engine, "start_indexing", backend.start_indexing)
    return backend


@pytest.fixture
def fake_index(
    vector_backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., _FakeVectorIndex]:
    """Serve every store from a backend whose capabilities the test chooses.

    The seam is the registry, not a private attribute: it is the one place a
    second backend is wired in, so a test substituting here drives the engine
    exactly as another backend would.

    Returns:
        A callable taking capability overrides and installing the backend.
    """
    del vector_backend

    def install(**overrides: Any) -> _FakeVectorIndex:  # noqa: ANN401
        index = _FakeVectorIndex(replace(_S3_CAPABILITIES, **overrides))
        monkeypatch.setattr(engine, "default_backend", lambda: index)
        monkeypatch.setattr(engine, "backend_for", lambda _store: index)
        monkeypatch.setattr(records, "default_backend", lambda: index)
        return index

    return install


@pytest.fixture
def scheduled_cleanups() -> Iterator[list[Awaitable[None]]]:
    """Bind the cleanup context the engine schedules its background work in.

    Yields:
        The pending cleanups, for the test to await when it wants them run.
    """
    token = CLEANUPS.set([])
    try:
        yield CLEANUPS.get()
    finally:
        CLEANUPS.reset(token)


async def _run_cleanups(pending: list[Awaitable[None]]) -> None:
    """Await every cleanup scheduled so far, including the ones they schedule."""
    while pending:
        await pending.pop(0)


def _abandon_cleanups(pending: list[Awaitable[None]]) -> None:
    """Drop every scheduled cleanup, as a task killed before running them does."""
    for cleanup in pending:
        cast("Coroutine[Any, Any, None]", cleanup).close()
    pending.clear()


def _error_of(response: Any) -> dict[str, Any]:  # noqa: ANN401
    """Return the error envelope of a failed response."""
    payload: dict[str, Any] = response.json()["error"]
    return payload


def _seed_store_record(backend: _FakeBackend, **overrides: Any) -> StoreRecord:  # noqa: ANN401
    """Write one store record straight into the record bucket.

    Args:
        backend: The in-memory backend serving the API.
        overrides: Fields to set on the record.

    Returns:
        The record as it was written.
    """
    defaults: dict[str, Any] = {
        "id": new_store_id(),
        "created_at": 1,
        "last_active_at": 1,
        "embedding_model": "m",
        "dimensions": 8,
    }
    record = StoreRecord(**(defaults | overrides))
    key = store_key(record.id)
    backend.records.objects[key] = record.model_dump_json().encode()
    backend.records.etags[key] = '"etag-seed"'
    return record


def _seed_file_record(
    backend: _FakeBackend,
    store_id: str,
    file_id: str,
    created_at: int,
    batch_id: str = "",
) -> None:
    """Write one file record straight into the record bucket.

    Args:
        backend: The in-memory backend serving the API.
        store_id: The store the file is attached to.
        file_id: The uploaded file's identifier.
        created_at: The moment the file was attached, as the record reports it.
        batch_id: The batch the file belongs to, or ``""``.
    """
    record = FileRecord(
        id=file_id, created_at=created_at, status="completed", batch_id=batch_id
    )
    key = file_key(store_id, file_id)
    backend.records.objects[key] = record.model_dump_json().encode()
    backend.records.etags[key] = '"etag-seed"'


def _seed_batch_record(backend: _FakeBackend, store_id: str, batch_id: str) -> None:
    """Write one batch record straight into the record bucket."""
    record = BatchRecord(
        id=batch_id,
        created_at=_LISTING_BASE,
        file_counts=FileCountsRecord(completed=len(_LISTING_TIMES)),
    )
    key = batch_key(store_id, batch_id)
    backend.records.objects[key] = record.model_dump_json().encode()
    backend.records.etags[key] = '"etag-seed"'


def _walk_listing(
    app_client: TestClient, path: str, *, order: str, limit: int
) -> list[dict[str, Any]]:
    """Return every object of a listing, one *limit*-sized page at a time.

    Args:
        app_client: The client the listing is read through.
        path: The listing path, without its query string.
        order: ``"asc"`` or ``"desc"``.
        limit: Page size; below the number of objects, so the walk spans pages.

    Returns:
        The objects, concatenated in the order the pages handed them back.
    """
    collected: list[dict[str, Any]] = []
    after = ""
    for _ in range(_WALK_MAX_PAGES):
        query = f"?limit={limit}&order={order}{f'&after={after}' if after else ''}"
        response = app_client.get(f"{path}{query}")
        assert response.status_code == 200, response.text
        page = response.json()
        collected.extend(page["data"])
        if not page["has_more"] or not page["data"]:
            return collected
        after = page["last_id"]
    pytest.fail("the cursor walk did not terminate")


#: Epoch second the seeded listing times are counted from.
_LISTING_BASE: int = 1_760_000_000

#: Seconds after :data:`_LISTING_BASE` each seeded record reports, in identifier order.
_LISTING_TIMES: tuple[int, ...] = (1, 6, 2, 7, 3, 8)

#: Identifier letters the seeded files are named after, in identifier order.
_LISTING_LETTERS: tuple[str, ...] = ("a", "b", "c", "d", "e", "f")

#: Pages a cursor walk may take before the test calls it non-terminating.
_WALK_MAX_PAGES: int = 12

#: Knowledge base identifiers the merged listing test addresses stores on.
_EXTERNAL_KB_IDS: tuple[str, ...] = ("AAAAA11111", "ZZZZZ99999")


@pytest.mark.local
class TestStoreFileListingOrder:
    """A file listing orders on the attachment time it reports, across pages.

    Issue #165: the order came from the S3 key, which is the *uploaded file's*
    identifier — minted when the file was uploaded — while ``created_at`` is
    the moment the file was attached to this store. A file uploaded last week
    and attached today therefore sorted as an old one, and re-attaching a file
    rewrote its ``created_at`` without moving its row.

    The seeded times invert the identifier order in pairs, so every page of two
    is internally ordered whichever of the two quantities the listing ran on:
    only concatenating the pages tells them apart.

    Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/listFiles
         stdapi/vector_stores/records.py:list_store_files
    """

    @pytest.fixture
    def seeded_store(self, vector_backend: _FakeBackend) -> str:
        """Return a store holding one file per :data:`_LISTING_TIMES` entry."""
        store = _seed_store_record(vector_backend)
        for letter, offset in zip(_LISTING_LETTERS, _LISTING_TIMES, strict=True):
            _seed_file_record(
                vector_backend, store.id, f"file-{letter * 32}", _LISTING_BASE + offset
            )
        return store.id

    @pytest.mark.parametrize("order", ["asc", "desc"])
    def test_pages_concatenate_in_created_at_order(
        self, app_client: TestClient, seeded_store: str, order: str
    ) -> None:
        """Walking every page with the ``after`` cursor yields one ``created_at`` sequence.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/listFiles
             stdapi/vector_stores/records.py:list_store_files
        """
        files = _walk_listing(
            app_client, f"/v1/vector_stores/{seeded_store}/files", order=order, limit=2
        )
        assert len(files) == len(_LISTING_TIMES)
        created = [entry["created_at"] for entry in files]
        assert created == sorted(created, reverse=order == "desc"), (
            f"order={order} must hold across pages, got {created}"
        )

    def test_created_at_is_the_moment_the_file_was_attached(
        self, app_client: TestClient, seeded_store: str
    ) -> None:
        """Every reported ``created_at`` is the attachment the record holds.

        Ref: stdapi/vector_stores/engine.py:attach_files
        """
        files = _walk_listing(
            app_client, f"/v1/vector_stores/{seeded_store}/files", order="asc", limit=2
        )
        assert {entry["id"]: entry["created_at"] for entry in files} == {
            f"file-{letter * 32}": _LISTING_BASE + offset
            for letter, offset in zip(_LISTING_LETTERS, _LISTING_TIMES, strict=True)
        }

    @pytest.mark.parametrize(
        ("order", "cursor", "expected"),
        [("asc", "b", ("c", "e")), ("desc", "e", ("d", "b"))],
    )
    def test_before_returns_the_page_ending_at_the_cursor(
        self,
        app_client: TestClient,
        seeded_store: str,
        order: str,
        cursor: str,
        expected: tuple[str, ...],
    ) -> None:
        """``before`` names a position in the listing, so its page ends at the cursor.

        The page it names is the two files preceding the cursor in the listing's
        own direction — the two attached just before it when ascending, the two
        attached just after it when descending — and never the start of the
        listing.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/listFiles
             stdapi/vector_stores/_paging.py:page_records
        """
        page = app_client.get(
            f"/v1/vector_stores/{seeded_store}/files"
            f"?limit=2&order={order}&before=file-{cursor * 32}"
        ).json()
        assert [entry["id"] for entry in page["data"]] == [
            f"file-{letter * 32}" for letter in expected
        ]

    @pytest.mark.parametrize("order", ["asc", "desc"])
    def test_a_batch_listing_pages_in_the_same_order(
        self, app_client: TestClient, vector_backend: _FakeBackend, order: str
    ) -> None:
        """A batch's own file listing orders on the same quantity as the store's.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-file-batches/listBatchFiles
             stdapi/vector_stores/records.py:list_batch_files
        """
        store = _seed_store_record(vector_backend)
        batch_id = new_batch_id()
        _seed_batch_record(vector_backend, store.id, batch_id)
        for letter, offset in zip(_LISTING_LETTERS, _LISTING_TIMES, strict=True):
            _seed_file_record(
                vector_backend,
                store.id,
                f"file-{letter * 32}",
                _LISTING_BASE + offset,
                batch_id=batch_id,
            )
        files = _walk_listing(
            app_client,
            f"/v1/vector_stores/{store.id}/file_batches/{batch_id}/files",
            order=order,
            limit=2,
        )
        created = [entry["created_at"] for entry in files]
        assert len(created) == len(_LISTING_TIMES)
        assert created == sorted(created, reverse=order == "desc"), (
            f"order={order} must hold across pages, got {created}"
        )

    def test_reattaching_a_file_moves_it_to_the_newest_end(
        self,
        app_client: TestClient,
        vector_backend: _FakeBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A file attached again is listed as the newest, since that is what it reports.

        Re-attaching replaces the record and rewrites its ``created_at``, so a
        listing ordered on anything else reports a time that contradicts its own
        order.

        Ref: stdapi/vector_stores/engine.py:attach_files
        """
        clock = count(_LISTING_BASE)
        monkeypatch.setattr(engine, "now_utc_timestamp", lambda: next(clock))
        store_id = app_client.post("/v1/vector_stores", json={}).json()["id"]
        first = vector_backend.upload(_TEXT_FILE)
        second = vector_backend.upload(_OTHER_FILE)
        for file_id in (first, second, first):
            attached = app_client.post(
                f"/v1/vector_stores/{store_id}/files", json={"file_id": file_id}
            )
            assert attached.status_code == 200, attached.text

        page = app_client.get(f"/v1/vector_stores/{store_id}/files?limit=2").json()

        assert [entry["id"] for entry in page["data"]] == [first, second]


@pytest.mark.local
class TestMergedStoreListingOrder:
    """The listing of a deployment serving both kinds of store orders on one quantity.

    Issue #165: the merge sorted on ``record.id``, and the identifier of a store
    held elsewhere names the knowledge base rather than a moment, so external
    stores fell into an arbitrary order among themselves and the merge threw
    away the ordering the backend had already done.

    Ref: https://platform.openai.com/docs/api-reference/vector-stores/list
         stdapi/vector_stores/engine.py:list_stores
    """

    @pytest.fixture
    def merged_stores(
        self, vector_backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> dict[str, int]:
        """Serve two stores of our own and two held elsewhere, interleaved in time.

        Returns:
            The creation time of every store, keyed by identifier.
        """
        native = [
            _seed_store_record(
                vector_backend,
                created_at=_LISTING_BASE + offset,
                last_active_at=_LISTING_BASE + offset,
            )
            for offset in (2, 5)
        ]
        external = [
            StoreRecord(
                id=f"vs_kb_{kb_id}",
                created_at=_LISTING_BASE + offset,
                last_active_at=_LISTING_BASE + offset,
                embedding_model="m",
                dimensions=8,
                external_status="completed",
            )
            for kb_id, offset in zip(_EXTERNAL_KB_IDS, (1, 3), strict=True)
        ]

        class _ExternalBackend:
            """A backend answering for stores this deployment only addresses."""

            async def list_stores(self) -> list[StoreRecord]:
                """Return the external stores, ordered as the backend orders them."""
                return sorted(external, key=lambda record: record.created_at)

        monkeypatch.setattr(engine, "external_stores", lambda: (_ExternalBackend(),))
        return {record.id: record.created_at for record in native + external}

    @pytest.mark.parametrize("order", ["asc", "desc"])
    def test_pages_concatenate_in_created_at_order(
        self, app_client: TestClient, merged_stores: dict[str, int], order: str
    ) -> None:
        """Both kinds of store take their place in one ``created_at`` sequence.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/list
             stdapi/vector_stores/engine.py:list_stores
        """
        stores = _walk_listing(app_client, "/v1/vector_stores", order=order, limit=2)
        assert {entry["id"] for entry in stores} == set(merged_stores)
        created = [entry["created_at"] for entry in stores]
        assert created == sorted(created, reverse=order == "desc"), (
            f"order={order} must hold across pages, got {created}"
        )


@pytest.mark.local
class TestConditionalUpdate:
    """Counter updates retry when another writer won the conditional write.

    Ref: stdapi/vector_stores/records.py:update_record
    """

    async def test_retries_until_the_write_lands(
        self, vector_backend: _FakeBackend
    ) -> None:
        """A losing conditional write is retried against the re-read record.

        The mutation is applied to what the store holds at the moment it lands,
        never to the copy the losing attempt read: a counter incremented from a
        stale record is how totals drift.

        Ref: stdapi/vector_stores/records.py:update_record
        """
        record = _seed_store_record(vector_backend)
        key = store_key(record.id)
        vector_backend.records.lose_writes[key] = 2
        vector_backend.records.reads = 0
        vector_backend.records.writes = 0

        updated = await update_record(
            StoreRecord, key, lambda r: setattr(r, "usage_bytes", r.usage_bytes + 1)
        )

        assert vector_backend.records.writes == 3
        assert vector_backend.records.reads == 3
        assert updated.usage_bytes == 1

    async def test_missing_record_never_names_internal_storage(
        self, vector_backend: _FakeBackend
    ) -> None:
        """A record deleted mid-update answers 404 without naming its object key.

        Ref: AGENTS.md "Never Leak Internals"
        """
        del vector_backend
        with pytest.raises(ApiError) as raised:
            await update_record(
                StoreRecord, "vector_stores/vs_x/store.json", lambda _record: None
            )
        assert raised.value.status == 404
        message = str(raised.value)
        assert "vector_stores/" not in message
        assert ".json" not in message

    async def test_gives_up_with_a_retryable_status(
        self, vector_backend: _FakeBackend
    ) -> None:
        """Endless contention answers 409 rather than looping forever.

        Ref: stdapi/vector_stores/records.py:update_record
        """
        record = _seed_store_record(vector_backend)
        key = store_key(record.id)
        vector_backend.records.lose_writes[key] = 99
        with pytest.raises(ApiError) as raised:
            await update_record(StoreRecord, key, lambda _record: None)
        assert raised.value.status == 409


@pytest.mark.usefixtures("vector_stores_api")
class TestVectorStoreCrud:
    """Create, retrieve, update, list and delete a vector store.

    Ref: https://platform.openai.com/docs/api-reference/vector-stores
    """

    def test_create_returns_a_vector_store(self, openai_client: OpenAI) -> None:
        """A new store is returned with its identifier, counters and status.

        Ref: openai.types.vector_store.VectorStore
        """
        store = openai_client.vector_stores.create(name="stdapi-test-create")
        try:
            assert store.object == "vector_store"
            assert store.name == "stdapi-test-create"
            assert store.status == "completed"
            assert store.file_counts.total == 0
            assert store.usage_bytes == 0
            assert store.created_at > 0
            assert store.last_active_at is not None
        finally:
            openai_client.vector_stores.delete(store.id)

    def test_retrieve_returns_the_created_store(
        self, empty_store: str, openai_client: OpenAI
    ) -> None:
        """A store reads back with the identifier it was created with.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/retrieve
        """
        assert openai_client.vector_stores.retrieve(empty_store).id == empty_store

    def test_update_changes_name_and_metadata(
        self, empty_store: str, openai_client: OpenAI
    ) -> None:
        """``name`` and ``metadata`` round-trip through an update.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/modify
        """
        updated = openai_client.vector_stores.update(
            empty_store, name="stdapi-test-renamed", metadata={"team": "science"}
        )
        assert updated.name == "stdapi-test-renamed"
        assert updated.metadata == {"team": "science"}
        assert openai_client.vector_stores.retrieve(empty_store).name == (
            "stdapi-test-renamed"
        )

    def test_expires_after_round_trips(self, openai_client: OpenAI) -> None:
        """An expiration policy is echoed back with a consistent ``expires_at``.

        Ref: openai.types.vector_store.ExpiresAfter
        """
        store = openai_client.vector_stores.create(
            name="stdapi-test-expiry",
            expires_after={"anchor": "last_active_at", "days": 1},
        )
        try:
            assert store.expires_after is not None
            assert store.expires_after.anchor == "last_active_at"
            assert store.expires_after.days == 1
            assert store.expires_at is not None
            assert store.last_active_at is not None
            assert store.expires_at == store.last_active_at + 86400
        finally:
            openai_client.vector_stores.delete(store.id)

    def test_list_paginates_across_a_page_boundary(self, openai_client: OpenAI) -> None:
        """A page of one reports ``has_more`` and its cursor continues the listing.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/list
        """
        created = [
            openai_client.vector_stores.create(name=f"stdapi-test-page-{index}")
            for index in range(2)
        ]
        wanted = {store.id for store in created}
        try:
            # The listing is eventually consistent: a new store may miss the next page.
            deadline = time.monotonic() + _LIST_TIMEOUT
            while not wanted <= {
                entry.id for entry in openai_client.vector_stores.list(limit=100).data
            }:
                assert time.monotonic() < deadline, "created stores never listed"
                time.sleep(2.0)
            page = openai_client.vector_stores.list(limit=1, order="desc")
            assert len(page.data) == 1
            assert page.has_more
            following = openai_client.vector_stores.list(
                limit=1, order="desc", after=page.data[0].id
            )
            assert following.data
            assert following.data[0].id != page.data[0].id
        finally:
            for store in created:
                openai_client.vector_stores.delete(store.id)

    def test_delete_removes_the_store(self, openai_client: OpenAI) -> None:
        """A deleted store is confirmed deleted and no longer retrievable.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/delete
        """
        store = openai_client.vector_stores.create(name="stdapi-test-delete")
        deleted = openai_client.vector_stores.delete(store.id)
        assert deleted.deleted
        assert deleted.id == store.id
        with pytest.raises(NotFoundError):
            openai_client.vector_stores.retrieve(store.id)

    def test_unknown_store_is_not_found(self, openai_client: OpenAI) -> None:
        """A well-formed but unknown store identifier answers 404.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/retrieve
        """
        with pytest.raises(NotFoundError):
            openai_client.vector_stores.retrieve("vs_" + "0" * 26)


@pytest.mark.slow
@pytest.mark.usefixtures("vector_stores_api")
class TestFileAttachment:
    """Attaching a file indexes it asynchronously and converges the counters.

    Ref: https://platform.openai.com/docs/api-reference/vector-stores-files
    """

    def test_attach_converges_to_completed(
        self, empty_store: str, openai_client: OpenAI
    ) -> None:
        """A text file moves from ``in_progress`` to ``completed`` and bills usage bytes.

        Ref: openai.types.vector_stores.vector_store_file.VectorStoreFile
        """
        file_id = _upload(openai_client, "converge.txt", _TEXT_FILE)
        try:
            attached = openai_client.vector_stores.files.create(
                vector_store_id=empty_store,
                file_id=file_id,
                chunking_strategy={
                    "type": "static",
                    "static": {
                        "max_chunk_size_tokens": 100,
                        "chunk_overlap_tokens": 20,
                    },
                },
            )
            assert attached.object == "vector_store.file"
            assert attached.vector_store_id == empty_store
            assert attached.status in ("in_progress", "completed")
            settled = _wait_for_file(openai_client, empty_store, file_id)
            assert settled.status == "completed"
            assert settled.last_error is None
            assert settled.usage_bytes > 0
            store = _wait_for_store(openai_client, empty_store, files=1)
            assert store.file_counts.total == 1
            assert store.file_counts.completed == 1
            assert store.usage_bytes > 0
        finally:
            openai_client.files.delete(file_id)

    def test_file_content_returns_the_indexed_chunks(
        self, indexed_store: tuple[str, str, str], openai_client: OpenAI
    ) -> None:
        """The indexed passages read back in document order and hold the planted sentence.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/getContent
        """
        store_id, planted, _ = indexed_store
        chunks = list(
            openai_client.vector_stores.files.content(planted, vector_store_id=store_id)
        )
        assert chunks
        assert all(chunk.type == "text" for chunk in chunks)
        joined = "\n".join(chunk.text or "" for chunk in chunks)
        assert "QUINCEY-7" in joined
        assert joined.index("Observatory operations manual") < joined.index("QUINCEY-7")

    def test_list_files_reports_the_attached_files(
        self, indexed_store: tuple[str, str, str], openai_client: OpenAI
    ) -> None:
        """Both attached files are listed, and the status filter narrows the page.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/listFiles
        """
        store_id, planted, other = indexed_store
        _wait_for_listed_files(openai_client, store_id, {planted, other})
        completed = openai_client.vector_stores.files.list(
            store_id, limit=100, filter="completed"
        )
        assert {entry.id for entry in completed.data} == {planted, other}
        cancelled = openai_client.vector_stores.files.list(
            store_id, limit=100, filter="cancelled"
        )
        assert cancelled.data == []

    def test_attributes_round_trip_and_can_be_replaced(
        self, empty_store: str, openai_client: OpenAI
    ) -> None:
        """Attributes are echoed on the file and replaced by an update.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/updateAttributes
        """
        file_id = _upload(openai_client, "attrs.txt", _OTHER_FILE)
        try:
            attached = openai_client.vector_stores.files.create(
                vector_store_id=empty_store,
                file_id=file_id,
                attributes={"topic": "catering", "year": 2026},
            )
            assert attached.attributes == {"topic": "catering", "year": 2026}
            _wait_for_file(openai_client, empty_store, file_id)
            updated = openai_client.vector_stores.files.update(
                file_id, vector_store_id=empty_store, attributes={"topic": "menu"}
            )
            assert updated.attributes == {"topic": "menu"}
        finally:
            openai_client.files.delete(file_id)

    def test_reattaching_a_file_does_not_count_it_twice(
        self, empty_store: str, openai_client: OpenAI
    ) -> None:
        """Attaching the same file again replaces it instead of double-counting it.

        Ref: openai.types.vector_store.FileCounts
        """
        file_id = _upload(openai_client, "twice.txt", _OTHER_FILE)
        try:
            openai_client.vector_stores.files.create(
                vector_store_id=empty_store, file_id=file_id
            )
            _wait_for_file(openai_client, empty_store, file_id)
            openai_client.vector_stores.files.create(
                vector_store_id=empty_store, file_id=file_id
            )
            _wait_for_file(openai_client, empty_store, file_id)
            store = _wait_for_store(openai_client, empty_store, files=1)
            assert store.file_counts.total == 1
            assert store.file_counts.completed == 1
            attached = openai_client.vector_stores.files.retrieve(
                file_id, vector_store_id=empty_store
            )
            assert store.usage_bytes == attached.usage_bytes
            _wait_for_listed_files(openai_client, empty_store, {file_id})
        finally:
            openai_client.files.delete(file_id)

    def test_detach_removes_the_file(
        self, empty_store: str, openai_client: OpenAI
    ) -> None:
        """A detached file is gone from the store while the upload survives.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/deleteFile
        """
        file_id = _upload(openai_client, "detach.txt", _OTHER_FILE)
        try:
            openai_client.vector_stores.files.create(
                vector_store_id=empty_store, file_id=file_id
            )
            _wait_for_file(openai_client, empty_store, file_id)
            deleted = openai_client.vector_stores.files.delete(
                file_id, vector_store_id=empty_store
            )
            assert deleted.deleted
            deadline = time.monotonic() + _LIST_TIMEOUT
            while True:
                try:
                    openai_client.vector_stores.files.retrieve(
                        file_id, vector_store_id=empty_store
                    )
                except NotFoundError:
                    break
                assert time.monotonic() < deadline, "detached file still retrievable"
                time.sleep(2.0)
            assert openai_client.files.retrieve(file_id).id == file_id
        finally:
            openai_client.files.delete(file_id)

    def test_unknown_attached_file_is_not_found(
        self, empty_store: str, openai_client: OpenAI
    ) -> None:
        """A file that was never attached answers 404.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/getFile
        """
        with pytest.raises(NotFoundError):
            openai_client.vector_stores.files.retrieve(
                "file-" + "0" * 32, vector_store_id=empty_store
            )

    def test_non_text_file_fails_with_unsupported_file(
        self, empty_store: str, openai_client: OpenAI
    ) -> None:
        """A file whose bytes are not text settles as ``failed``/``unsupported_file``.

        Ref: openai.types.vector_stores.vector_store_file.LastError
        """
        file_id = openai_client.files.create(
            file=("picture.png", red_png(), "image/png"), purpose="assistants"
        ).id
        try:
            openai_client.vector_stores.files.create(
                vector_store_id=empty_store, file_id=file_id
            )
            settled = _wait_for_file(openai_client, empty_store, file_id)
            assert settled.status == "failed"
            assert settled.last_error is not None
            assert settled.last_error.code == "unsupported_file"
            # The store must settle too, or it stays `in_progress` for good.
            store = _wait_for_store(openai_client, empty_store, files=1)
            assert store.status == "completed"
            assert store.file_counts.failed == 1
            assert store.file_counts.in_progress == 0
            assert store.usage_bytes == 0
        finally:
            openai_client.files.delete(file_id)


@pytest.mark.slow
@pytest.mark.usefixtures("vector_stores_api")
class TestSearch:
    """Search returns the indexed passages closest in meaning to the query.

    Ref: https://platform.openai.com/docs/api-reference/vector-stores/search
    """

    def test_search_finds_the_planted_sentence(
        self, indexed_store: tuple[str, str, str], openai_client: OpenAI
    ) -> None:
        """A question only one passage answers returns that passage, scored and attributed.

        Ref: openai.types.vector_store_search_response.VectorStoreSearchResponse
        """
        store_id, planted, _ = indexed_store
        page = openai_client.vector_stores.search(
            store_id, query="What is the maintenance hatch code?", max_num_results=3
        )
        results = list(page)
        assert results
        assert any("QUINCEY-7" in part.text for r in results for part in r.content)
        best = results[0]
        assert 0.0 < best.score <= 1.0
        assert best.file_id == planted
        assert best.filename == "observatory.txt"
        assert best.content
        assert best.content[0].type == "text"

    def test_max_num_results_is_honoured(
        self, indexed_store: tuple[str, str, str], openai_client: OpenAI
    ) -> None:
        """The page never holds more results than asked for.

        Ref: openai.types.vector_store_search_params.VectorStoreSearchParams
        """
        store_id, _, _ = indexed_store
        results = list(
            openai_client.vector_stores.search(
                store_id, query="observatory", max_num_results=1
            )
        )
        assert len(results) <= 1

    def test_score_threshold_drops_weak_matches(
        self, indexed_store: tuple[str, str, str], openai_client: OpenAI
    ) -> None:
        """A threshold of 1.0 leaves nothing but an exact match, if any.

        Ref: openai.types.vector_store_search_params.RankingOptions
        """
        store_id, _, _ = indexed_store
        results = list(
            openai_client.vector_stores.search(
                store_id,
                query="cafeteria rota",
                ranking_options={"score_threshold": 1.0},
            )
        )
        assert all(result.score >= 1.0 for result in results)

    def test_filters_narrow_the_search_to_one_file(
        self, empty_store: str, openai_client: OpenAI
    ) -> None:
        """An attribute filter keeps only the files carrying that attribute.

        Ref: openai.types.shared_params.comparison_filter.ComparisonFilter
        """
        keep = _upload(openai_client, "keep.txt", _TEXT_FILE)
        drop = _upload(openai_client, "drop.txt", _OTHER_FILE)
        try:
            for file_id, topic in ((keep, "observatory"), (drop, "catering")):
                openai_client.vector_stores.files.create(
                    vector_store_id=empty_store,
                    file_id=file_id,
                    attributes={"topic": topic},
                )
            _wait_for_store(openai_client, empty_store, files=2)
            _wait_for_file(openai_client, empty_store, keep)
            _wait_for_file(openai_client, empty_store, drop)
            results = list(
                openai_client.vector_stores.search(
                    empty_store,
                    query="what happens here",
                    filters={"key": "topic", "type": "eq", "value": "observatory"},
                    max_num_results=10,
                )
            )
            assert results
            assert {result.file_id for result in results} == {keep}
            assert all(
                result.attributes == {"topic": "observatory"} for result in results
            )
        finally:
            for file_id in (keep, drop):
                openai_client.files.delete(file_id)

    def test_filter_on_an_unknown_attribute_returns_an_empty_page(
        self, indexed_store: tuple[str, str, str], openai_client: OpenAI
    ) -> None:
        """A filter nothing matches answers an empty page, not an error.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/search
        """
        store_id, _, _ = indexed_store
        results = list(
            openai_client.vector_stores.search(
                store_id,
                query="observatory",
                filters={"key": "no_such_attribute", "type": "eq", "value": "x"},
            )
        )
        assert results == []

    def test_search_on_unknown_store_is_not_found(self, openai_client: OpenAI) -> None:
        """Searching a store that does not exist answers 404.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/search
        """
        with pytest.raises(NotFoundError):
            openai_client.vector_stores.search("vs_" + "0" * 26, query="anything")


@pytest.mark.slow
@pytest.mark.usefixtures("vector_stores_api")
class TestFileBatches:
    """A file batch attaches several files and reports their progress together.

    Ref: https://platform.openai.com/docs/api-reference/vector-stores-file-batches
    """

    def test_batch_converges_and_lists_its_files(
        self, empty_store: str, openai_client: OpenAI
    ) -> None:
        """A two-file batch completes, counts both files and lists exactly them.

        Ref: openai.types.vector_stores.vector_store_file_batch.VectorStoreFileBatch
        """
        first = _upload(openai_client, "batch-a.txt", _TEXT_FILE)
        second = _upload(openai_client, "batch-b.txt", _OTHER_FILE)
        try:
            batch = openai_client.vector_stores.file_batches.create(
                vector_store_id=empty_store, file_ids=[first, second]
            )
            assert batch.object == _BATCH_OBJECT
            assert batch.vector_store_id == empty_store
            assert batch.file_counts.total == 2
            deadline = time.monotonic() + _INDEX_TIMEOUT
            while batch.status == "in_progress":
                assert time.monotonic() < deadline, "batch still in progress"
                time.sleep(2.0)
                batch = openai_client.vector_stores.file_batches.retrieve(
                    batch.id, vector_store_id=empty_store
                )
            assert batch.status == "completed"
            assert batch.file_counts.completed == 2
            deadline = time.monotonic() + _LIST_TIMEOUT
            while True:
                listed = {
                    entry.id
                    for entry in openai_client.vector_stores.file_batches.list_files(
                        batch.id, vector_store_id=empty_store, limit=100
                    ).data
                }
                if listed == {first, second}:
                    break
                assert time.monotonic() < deadline, (
                    f"batch files never listed: {listed}"
                )
                time.sleep(2.0)
        finally:
            for file_id in (first, second):
                openai_client.files.delete(file_id)

    @pytest.mark.gateway(
        "the vendor answers 500 to a file batch cancel, so only this gateway "
        "can be asserted against"
    )
    def test_cancel_keeps_finished_files_finished(
        self, empty_store: str, openai_client: OpenAI
    ) -> None:
        """Cancelling leaves a batch consistent, whichever files it caught in time.

        The race against the indexer cannot be won deterministically over the
        network, so this asserts the invariants a cancellation must keep;
        ``TestIndexingOffline`` is what proves the files are cancelled at all.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-file-batches/cancelBatch
             tests/test_openai_vector_stores.py:TestIndexingOffline
        """
        uploads = [
            _upload(openai_client, f"cancel-{index}.txt", _TEXT_FILE)
            for index in range(3)
        ]
        try:
            batch = openai_client.vector_stores.file_batches.create(
                vector_store_id=empty_store, file_ids=uploads
            )
            openai_client.vector_stores.file_batches.cancel(
                batch.id, vector_store_id=empty_store
            )
            deadline = time.monotonic() + _INDEX_TIMEOUT
            while batch.status == "in_progress":
                assert time.monotonic() < deadline, "batch still in progress"
                time.sleep(2.0)
                batch = openai_client.vector_stores.file_batches.retrieve(
                    batch.id, vector_store_id=empty_store
                )
            counts = batch.file_counts
            assert counts.total == len(uploads)
            assert counts.in_progress == 0
            assert counts.failed == 0
            assert counts.completed + counts.cancelled == len(uploads)
            # The batch reports cancelled exactly when a file was cancelled.
            assert (batch.status == "cancelled") == (counts.cancelled > 0)
            listed = openai_client.vector_stores.files.list(empty_store, limit=100).data
            assert sum(entry.status == "cancelled" for entry in listed) == (
                counts.cancelled
            )
            for entry in listed:
                assert entry.status in ("completed", "cancelled")
        finally:
            for file_id in uploads:
                openai_client.files.delete(file_id)

    @pytest.mark.gateway(
        "the vendor answers 500 rather than 404 for an unknown file batch id"
    )
    def test_unknown_batch_is_not_found(
        self, empty_store: str, openai_client: OpenAI
    ) -> None:
        """A well-formed but unknown batch identifier answers 404.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-file-batches/getBatch
        """
        with pytest.raises(NotFoundError):
            openai_client.vector_stores.file_batches.retrieve(
                "vsfb_" + "0" * 26, vector_store_id=empty_store
            )


@pytest.mark.slow
@pytest.mark.gateway("usage is recorded in this gateway's own request log")
@pytest.mark.usefixtures("local_test_client", "vector_stores_api")
class TestIndexingUsage:
    """The embeddings an asynchronous attach bills are visible in the usage log.

    Indexing happens off the request path, where a usage entry recorded after
    the originating request's log was finalized would be dropped: without this
    test, every attach would bill invisibly with the suite still green.

    Ref: stdapi/monitoring.py:log_background_event
         stdapi/vector_stores/engine.py:index_files
    """

    def test_indexing_records_embedding_usage(
        self, empty_store: str, openai_client: OpenAI, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """Attaching a file emits a usage entry carrying the embedding tokens.

        Ref: stdapi/usage.py:record_bedrock_usage
        """
        file_id = _upload(openai_client, "usage.txt", _TEXT_FILE)
        try:
            capfd.readouterr()
            openai_client.vector_stores.files.create(
                vector_store_id=empty_store, file_id=file_id
            )
            _wait_for_file(openai_client, empty_store, file_id)
            # The entry is written when the indexing event's log closes, which
            # trails the file reaching its terminal status.
            deadline = time.monotonic() + _LIST_TIMEOUT
            embedding: list[dict[str, Any]] = []
            while not embedding:
                assert time.monotonic() < deadline, (
                    "Indexing must record the embedding tokens it billed"
                )
                time.sleep(1.0)
                embedding += [
                    entry
                    for entry in logged_usage_entries(capfd.readouterr().out)
                    if entry.get("input_tokens", 0) > 0
                ]
        finally:
            openai_client.files.delete(file_id)


@pytest.mark.gateway("the record bucket behind a store is not an upstream concept")
@pytest.mark.usefixtures("local_test_client", "vector_stores_api")
class TestDetachIsDurableLive:
    """Detaching survives a lost task against the real index and record bucket.

    What the in-memory stand-in cannot answer: whether S3 really refuses the
    conditional write the recovery depends on, and whether the real index
    accepts a second deletion of vector keys it no longer holds — the reclaim
    is re-driven on every read, so a delete that is not idempotent would turn
    each read into an error.

    Ref: stdapi/vector_stores/engine.py:_reclaim_file
    """

    def test_a_lost_reclaim_is_finished_by_the_next_read(
        self, empty_store: str, openai_client: OpenAI, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A detach whose background work never ran leaves nothing searchable.

        The search that proves it runs after the reclaim, when the store no
        longer holds anything to filter the file out with: what answers is the
        index itself, so an empty page is the vectors really being gone.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/deleteFile
        """
        file_id = _upload(openai_client, "observatory.txt", _TEXT_FILE)
        try:
            openai_client.vector_stores.files.create(
                vector_store_id=empty_store, file_id=file_id
            )
            _wait_for_file(openai_client, empty_store, file_id)
            found = openai_client.vector_stores.search(empty_store, query="maintenance")
            assert [result for result in found.data if result.file_id == file_id]

            # The task is killed the moment the delete is answered.
            monkeypatch.setattr(
                engine,
                "schedule_cleanup",
                lambda *tasks: _abandon_cleanups(list(tasks)),
            )
            openai_client.vector_stores.files.delete(
                file_id, vector_store_id=empty_store
            )
            monkeypatch.undo()

            store = openai_client.vector_stores.retrieve(empty_store)
            assert store.file_counts.total == 0
            assert not openai_client.vector_stores.files.list(empty_store).data
            deadline = time.monotonic() + _LIST_TIMEOUT
            while openai_client.vector_stores.search(
                empty_store, query="maintenance"
            ).data:
                assert time.monotonic() < deadline, (
                    "the vectors of a deleted file are still searchable"
                )
                time.sleep(2.0)
        finally:
            openai_client.files.delete(file_id)


@pytest.mark.local
class TestUnconfiguredDeployment:
    """Without vector storage the routes answer 503, naming nothing internal.

    Ref: stdapi/vector_stores/records.py:records_bucket
    """

    def test_missing_vector_bucket_answers_503(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deployment with no vector bucket refuses cleanly.

        Ref: stdapi/vector_stores/records.py:records_bucket
        """
        monkeypatch.setattr(SETTINGS, "aws_s3_vectors_bucket", None, raising=False)
        with pytest.raises(ApiError) as raised:
            vector_stores.records_bucket()
        assert raised.value.status == 503
        message = str(raised.value)
        assert "administrator" in message
        assert "s3" not in message.lower()
        assert "bucket" not in message.lower()

    @pytest.mark.parametrize(
        ("method", "action"),
        [("create_index", "CreateIndex"), ("query_vectors", "QueryVectors")],
    )
    def test_a_denied_index_call_answers_like_an_unconfigured_one(
        self,
        app_client: TestClient,
        vector_backend: _FakeBackend,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
        method: str,
        action: str,
    ) -> None:
        """A missing ``s3vectors`` action is answered as the API being unavailable.

        The settings can be complete while the role is not, and that gap reached
        the caller as a raw 403 blaming them for the operator's IAM policy.

        Ref: stdapi/vector_stores/s3_vectors.py:_vectors_guard
             stdapi/api_errors.py:feature_unavailable_guard
        """
        store_id = app_client.post("/v1/vector_stores", json={}).json()["id"]

        denial = make_client_error(_ACCESS_DENIED_CODE, action)

        async def _denied(**_params: object) -> dict[str, Any]:
            raise denial

        monkeypatch.setattr(vector_backend.vectors, method, _denied)
        capfd.readouterr()
        response = (
            app_client.post("/v1/vector_stores", json={})
            if method == "create_index"
            else app_client.post(
                f"/v1/vector_stores/{store_id}/search", json={"query": "anything"}
            )
        )

        assert response.status_code == 503, response.text
        message = _error_of(response)["message"]
        assert "not available on the current server" in message
        assert "s3vectors" not in message
        logged = capfd.readouterr().out
        assert f"s3vectors:{action}" in logged
        assert "aws_s3_vectors_bucket" in logged


@pytest.mark.local
@pytest.mark.usefixtures("vector_backend")
class TestRoutesOffline:
    """The sixteen routes, driven end to end against the in-memory backend.

    This is the lane that runs on every push: without it the route bodies —
    every object shape, cursor, bound and error envelope — are only exercised
    when AWS credentials happen to be present.

    Ref: https://platform.openai.com/docs/api-reference/vector-stores
         stdapi/routes/openai_vector_stores.py
    """

    def test_create_retrieve_update_and_delete(self, app_client: TestClient) -> None:
        """A store round-trips through the four lifecycle routes.

        Ref: stdapi/routes/openai_vector_stores.py:create_vector_store
        """
        created = app_client.post(
            "/v1/vector_stores",
            json={
                "name": "handbook",
                "metadata": {"team": "science"},
                "expires_after": {"anchor": "last_active_at", "days": 3},
            },
        )
        assert created.status_code == 200, created.text
        store = created.json()
        assert store["object"] == "vector_store"
        assert store["status"] == "completed"
        assert store["file_counts"] == {
            "in_progress": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "total": 0,
        }
        assert store["expires_at"] == store["last_active_at"] + 3 * 86400

        retrieved = app_client.get(f"/v1/vector_stores/{store['id']}")
        assert retrieved.status_code == 200
        assert retrieved.json()["name"] == "handbook"

        updated = app_client.post(
            f"/v1/vector_stores/{store['id']}",
            json={"name": "renamed", "expires_after": None},
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["name"] == "renamed"
        assert updated.json().get("expires_at") is None

        deleted = app_client.delete(f"/v1/vector_stores/{store['id']}")
        assert deleted.status_code == 200
        assert deleted.json() == {
            "id": store["id"],
            "object": "vector_store.deleted",
            "deleted": True,
        }
        assert app_client.get(f"/v1/vector_stores/{store['id']}").status_code == 404

    def test_list_paginates_newest_first(self, app_client: TestClient) -> None:
        """The listing defaults to newest first and its cursor continues it.

        Ref: stdapi/routes/openai_vector_stores.py:list_vector_stores
        """
        ids = [
            app_client.post("/v1/vector_stores", json={"name": f"s{index}"}).json()[
                "id"
            ]
            for index in range(3)
        ]
        page = app_client.get("/v1/vector_stores?limit=2").json()
        assert [entry["id"] for entry in page["data"]] == ids[::-1][:2]
        assert page["has_more"]
        assert page["first_id"] == ids[-1]
        following = app_client.get(
            f"/v1/vector_stores?limit=2&after={page['last_id']}"
        ).json()
        assert [entry["id"] for entry in following["data"]] == [ids[0]]
        assert not following["has_more"]

    def test_ascending_listing_excludes_the_cursor(
        self, app_client: TestClient
    ) -> None:
        """An ``after`` cursor is exclusive in ascending order too.

        Ref: stdapi/vector_stores/records.py:list_ids
        """
        ids = [
            app_client.post("/v1/vector_stores", json={"name": f"a{index}"}).json()[
                "id"
            ]
            for index in range(3)
        ]
        page = app_client.get(f"/v1/vector_stores?order=asc&after={ids[0]}").json()
        assert [entry["id"] for entry in page["data"]] == ids[1:]

    @pytest.mark.parametrize(
        ("order", "expected_index"), [("asc", 0), ("desc", 2)], ids=["asc", "desc"]
    )
    def test_before_returns_the_page_ending_at_the_cursor(
        self, app_client: TestClient, order: str, expected_index: int
    ) -> None:
        """``before`` names a position, so its page is the one preceding the cursor.

        Which side of the cursor that is follows the listing's own direction: the
        older store when ascending, the newer one when descending.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/list
             stdapi/vector_stores/_paging.py:page_identifiers
        """
        ids = [
            app_client.post("/v1/vector_stores", json={"name": f"b{index}"}).json()[
                "id"
            ]
            for index in range(3)
        ]
        page = app_client.get(f"/v1/vector_stores?order={order}&before={ids[1]}").json()
        assert [entry["id"] for entry in page["data"]] == [ids[expected_index]]

    def test_unknown_identifiers_answer_404(self, app_client: TestClient) -> None:
        """Every route answers 404 for a well-formed identifier naming nothing.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/retrieve
        """
        store_id = f"vs_{'0' * 26}"
        file_id = f"file-{'0' * 32}"
        batch_id = f"vsfb_{'0' * 26}"
        for method, path in (
            ("get", f"/v1/vector_stores/{store_id}"),
            ("delete", f"/v1/vector_stores/{store_id}"),
            ("get", f"/v1/vector_stores/{store_id}/files"),
            ("get", f"/v1/vector_stores/{store_id}/files/{file_id}"),
            ("get", f"/v1/vector_stores/{store_id}/files/{file_id}/content"),
            ("delete", f"/v1/vector_stores/{store_id}/files/{file_id}"),
            ("get", f"/v1/vector_stores/{store_id}/file_batches/{batch_id}"),
            ("get", f"/v1/vector_stores/{store_id}/file_batches/{batch_id}/files"),
        ):
            response = getattr(app_client, method)(path)
            assert response.status_code == 404, f"{method} {path}: {response.text}"
            assert _error_of(response)["message"]
        for path, body in (
            (f"/v1/vector_stores/{store_id}", {"name": "x"}),
            (f"/v1/vector_stores/{store_id}/search", {"query": "x"}),
            (f"/v1/vector_stores/{store_id}/files", {"file_id": file_id}),
            (f"/v1/vector_stores/{store_id}/file_batches", {"file_ids": [file_id]}),
            (f"/v1/vector_stores/{store_id}/file_batches/{batch_id}/cancel", {}),
        ):
            response = app_client.post(path, json=body)
            assert response.status_code == 404, f"post {path}: {response.text}"

    def test_listing_bounds_are_enforced(self, app_client: TestClient) -> None:
        """A page size outside 1-100 and a malformed cursor are refused.

        Ref: stdapi/routes/openai_vector_stores.py:_Limit
        """
        assert app_client.get("/v1/vector_stores?limit=0").status_code == 400
        assert app_client.get("/v1/vector_stores?limit=101").status_code == 400
        assert app_client.get("/v1/vector_stores?after=not-a-store").status_code == 400

    def test_attributes_above_the_budget_are_refused(
        self, app_client: TestClient, vector_backend: _FakeBackend
    ) -> None:
        """Attributes too large to stay searchable are refused with the limit named.

        Ref: stdapi/vector_stores/engine.py:check_attributes
        """
        store_id = app_client.post("/v1/vector_stores", json={}).json()["id"]
        file_id = vector_backend.upload(_OTHER_FILE)
        response = app_client.post(
            f"/v1/vector_stores/{store_id}/files",
            json={
                "file_id": file_id,
                "attributes": {f"k{index}": "v" * 512 for index in range(6)},
            },
        )
        assert response.status_code == 400, response.text
        assert "2048" in _error_of(response)["message"]

    def test_more_attributes_than_upstream_accepts_are_refused(
        self, app_client: TestClient, vector_backend: _FakeBackend
    ) -> None:
        """The documented 16-pair ceiling is enforced before any backend work.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/createFile
        """
        store_id = app_client.post("/v1/vector_stores", json={}).json()["id"]
        response = app_client.post(
            f"/v1/vector_stores/{store_id}/files",
            json={
                "file_id": vector_backend.upload(_OTHER_FILE),
                "attributes": {f"k{index}": "v" for index in range(17)},
            },
        )
        assert response.status_code == 400, response.text
        assert "16" in _error_of(response)["message"]

    def test_attaching_an_unknown_file_leaves_no_store_behind(
        self, app_client: TestClient
    ) -> None:
        """A store created for files that do not exist is not left orphaned.

        Ref: stdapi/routes/openai_vector_stores.py:create_vector_store
        """
        before = app_client.get("/v1/vector_stores?limit=100").json()["data"]
        response = app_client.post(
            "/v1/vector_stores", json={"file_ids": [f"file-{'0' * 32}"]}
        )
        assert response.status_code == 404, response.text
        after = app_client.get("/v1/vector_stores?limit=100").json()["data"]
        assert [entry["id"] for entry in after] == [entry["id"] for entry in before]

    def test_attach_reports_the_file_as_in_progress(
        self, app_client: TestClient, vector_backend: _FakeBackend
    ) -> None:
        """An attached file is immediately visible, before it is indexed.

        Ref: openai.types.vector_stores.vector_store_file.VectorStoreFile
        """
        store_id = app_client.post("/v1/vector_stores", json={}).json()["id"]
        file_id = vector_backend.upload(_TEXT_FILE)
        attached = app_client.post(
            f"/v1/vector_stores/{store_id}/files",
            json={
                "file_id": file_id,
                "attributes": {"topic": "observatory"},
                "chunking_strategy": {
                    "type": "static",
                    "static": {
                        "max_chunk_size_tokens": 100,
                        "chunk_overlap_tokens": 20,
                    },
                },
            },
        )
        assert attached.status_code == 200, attached.text
        payload = attached.json()
        assert payload == {
            "id": file_id,
            "object": "vector_store.file",
            "created_at": payload["created_at"],
            "usage_bytes": 0,
            "vector_store_id": store_id,
            "status": "in_progress",
            "chunking_strategy": {
                "type": "static",
                "static": {"max_chunk_size_tokens": 100, "chunk_overlap_tokens": 20},
            },
            "attributes": {"topic": "observatory"},
        }
        assert vector_backend.started == [(store_id, "", (file_id,))]
        listed = app_client.get(f"/v1/vector_stores/{store_id}/files").json()
        assert [entry["id"] for entry in listed["data"]] == [file_id]
        assert (
            app_client.get(
                f"/v1/vector_stores/{store_id}/files?filter=completed"
            ).json()["data"]
            == []
        )
        assert app_client.get(f"/v1/vector_stores/{store_id}").json()["status"] == (
            "in_progress"
        )

    def test_file_batch_lifecycle(
        self, app_client: TestClient, vector_backend: _FakeBackend
    ) -> None:
        """A batch counts its files, lists them and accepts a cancellation.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-file-batches
        """
        store_id = app_client.post("/v1/vector_stores", json={}).json()["id"]
        files = [vector_backend.upload(_TEXT_FILE) for _ in range(2)]
        created = app_client.post(
            f"/v1/vector_stores/{store_id}/file_batches", json={"file_ids": files}
        )
        assert created.status_code == 200, created.text
        batch = created.json()
        assert batch["object"] == _BATCH_OBJECT
        assert batch["status"] == "in_progress"
        assert batch["file_counts"]["in_progress"] == 2
        assert batch["file_counts"]["total"] == 2

        retrieved = app_client.get(
            f"/v1/vector_stores/{store_id}/file_batches/{batch['id']}"
        )
        assert retrieved.json()["id"] == batch["id"]
        listed = app_client.get(
            f"/v1/vector_stores/{store_id}/file_batches/{batch['id']}/files"
        ).json()
        assert {entry["id"] for entry in listed["data"]} == set(files)

        cancelled = app_client.post(
            f"/v1/vector_stores/{store_id}/file_batches/{batch['id']}/cancel", json={}
        )
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["id"] == batch["id"]

    def test_batch_rejects_both_file_lists(
        self, app_client: TestClient, vector_backend: _FakeBackend
    ) -> None:
        """``file_ids`` and ``files`` are mutually exclusive over HTTP too.

        Ref: openai.types.vector_stores.file_batch_create_params.FileBatchCreateParams
        """
        store_id = app_client.post("/v1/vector_stores", json={}).json()["id"]
        file_id = vector_backend.upload(_TEXT_FILE)
        response = app_client.post(
            f"/v1/vector_stores/{store_id}/file_batches",
            json={"file_ids": [file_id], "files": [{"file_id": file_id}]},
        )
        assert response.status_code == 400, response.text

    def test_search_returns_the_indexed_passages(
        self, app_client: TestClient, vector_backend: _FakeBackend
    ) -> None:
        """A search answers with the passage, its file, its attributes and a score.

        Ref: openai.types.vector_store_search_response.VectorStoreSearchResponse
        """
        store_id, file_id = _seed_indexed_store(app_client, vector_backend)
        response = app_client.post(
            f"/v1/vector_stores/{store_id}/search",
            json={"query": _PLANTED, "max_num_results": 3},
        )
        assert response.status_code == 200, response.text
        page = response.json()
        assert page["object"] == "vector_store.search_results.page"
        assert page["search_query"] == [_PLANTED]
        assert page["data"]
        best = page["data"][0]
        assert best["file_id"] == file_id
        assert best["filename"].endswith(".txt")
        assert 0.0 < best["score"] <= 1.0
        assert best["attributes"] == {"topic": "observatory", "year": 2026}
        assert best["content"][0]["type"] == "text"
        assert "QUINCEY-7" in best["content"][0]["text"]

    def test_search_filters_and_thresholds_narrow_the_page(
        self, app_client: TestClient, vector_backend: _FakeBackend
    ) -> None:
        """A filter that matches nothing answers an empty page rather than an error.

        Ref: openai.types.shared_params.comparison_filter.ComparisonFilter
        """
        store_id, _ = _seed_indexed_store(app_client, vector_backend)
        matching = app_client.post(
            f"/v1/vector_stores/{store_id}/search",
            json={
                "query": _PLANTED,
                "filters": {"key": "topic", "type": "eq", "value": "observatory"},
            },
        ).json()
        assert matching["data"]
        missing = app_client.post(
            f"/v1/vector_stores/{store_id}/search",
            json={
                "query": _PLANTED,
                "filters": {"key": "topic", "type": "eq", "value": "catering"},
            },
        ).json()
        assert missing["data"] == []
        thresholded = app_client.post(
            f"/v1/vector_stores/{store_id}/search",
            json={"query": _PLANTED, "ranking_options": {"score_threshold": 1.0}},
        ).json()
        assert all(entry["score"] >= 1.0 for entry in thresholded["data"])

    def test_a_string_range_filter_is_our_error_not_the_backend_s(
        self, app_client: TestClient, vector_backend: _FakeBackend
    ) -> None:
        """An ordering operator given a string is refused without naming the index.

        The index orders numbers only: forwarding the filter would answer with
        the backend's own message, naming the internal metadata key and its
        operator dialect.

        Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-metadata-filtering.html
        """
        store_id, _ = _seed_indexed_store(app_client, vector_backend)
        response = app_client.post(
            f"/v1/vector_stores/{store_id}/search",
            json={
                "query": _PLANTED,
                "filters": {"key": "date", "type": "gte", "value": "2026-01-01"},
            },
        )
        assert response.status_code == 400, response.text
        message = _error_of(response)["message"]
        assert "number" in message
        assert "$gte" not in response.text
        assert attribute_key("date") not in response.text

    def test_a_numeric_range_filter_is_applied(
        self, app_client: TestClient, vector_backend: _FakeBackend
    ) -> None:
        """The same operator on a number reaches the index and narrows the page.

        Ref: stdapi/vector_stores/s3_vectors.py:translate_filter
        """
        store_id, _ = _seed_indexed_store(app_client, vector_backend)
        page = app_client.post(
            f"/v1/vector_stores/{store_id}/search",
            json={
                "query": _PLANTED,
                "filters": {"key": "year", "type": "gte", "value": 2020},
            },
        ).json()
        assert page["data"]
        empty = app_client.post(
            f"/v1/vector_stores/{store_id}/search",
            json={
                "query": _PLANTED,
                "filters": {"key": "year", "type": "gt", "value": 2026},
            },
        ).json()
        assert empty["data"] == []

    def test_more_queries_than_the_search_accepts_are_refused(
        self, app_client: TestClient, vector_backend: _FakeBackend
    ) -> None:
        """The per-request query fan-out is bounded at parse time.

        Ref: stdapi/types/openai_vector_stores.py:VectorStoreSearchParams
        """
        store_id, _ = _seed_indexed_store(app_client, vector_backend)
        response = app_client.post(
            f"/v1/vector_stores/{store_id}/search", json={"query": ["q"] * 17}
        )
        assert response.status_code == 400, response.text

    def test_file_content_and_detach(
        self, app_client: TestClient, vector_backend: _FakeBackend
    ) -> None:
        """The indexed passages read back, and detaching removes file and vectors.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/getContent
        """
        store_id, file_id = _seed_indexed_store(app_client, vector_backend)
        content = app_client.get(
            f"/v1/vector_stores/{store_id}/files/{file_id}/content"
        ).json()
        assert content["object"] == "vector_store.file_content.page"
        assert content["data"]
        assert all(entry["type"] == "text" for entry in content["data"])
        assert "QUINCEY-7" in "\n".join(entry["text"] for entry in content["data"])

        updated = app_client.post(
            f"/v1/vector_stores/{store_id}/files/{file_id}",
            json={"attributes": {"topic": "menu"}},
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["attributes"] == {"topic": "menu"}

        deleted = app_client.delete(f"/v1/vector_stores/{store_id}/files/{file_id}")
        assert deleted.json() == {
            "id": file_id,
            "object": "vector_store.file.deleted",
            "deleted": True,
        }
        assert (
            app_client.get(f"/v1/vector_stores/{store_id}/files/{file_id}").status_code
            == 404
        )
        assert vector_backend.vectors.indexes[index_name(store_id)] == {}

    def test_an_unconfigured_deployment_answers_503(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a vector bucket the routes refuse, naming nothing internal.

        Ref: stdapi/vector_stores/records.py:records_bucket
        """
        monkeypatch.setattr(SETTINGS, "aws_s3_vectors_bucket", None)
        response = app_client.post("/v1/vector_stores", json={})
        assert response.status_code == 503, response.text
        message = _error_of(response)["message"]
        assert "administrator" in message
        assert "bucket" not in message.lower()


def _seed_indexed_store(client: TestClient, backend: _FakeBackend) -> tuple[str, str]:
    """Create a store holding one fully indexed file, through the API.

    Args:
        client: The pre-authenticated ASGI client.
        backend: The in-memory backend serving it.

    Returns:
        ``(vector_store_id, file_id)``.
    """
    store_id = client.post(
        "/v1/vector_stores",
        json={
            "chunking_strategy": {
                "type": "static",
                "static": {"max_chunk_size_tokens": 100, "chunk_overlap_tokens": 20},
            }
        },
    ).json()["id"]
    file_id = backend.upload(_TEXT_FILE)
    client.post(
        f"/v1/vector_stores/{store_id}/files",
        json={"file_id": file_id, "attributes": {"topic": "observatory", "year": 2026}},
    )
    run(index_files(store_id, [file_id], "", "test-request"))
    return store_id, file_id


async def _create_store(**overrides: Any) -> StoreRecord:  # noqa: ANN401
    """Create a store through the engine, with small chunks by default."""
    return await create_store(
        name=overrides.pop("name", "offline"),
        description="",
        metadata={},
        expires_after_days=overrides.pop("expires_after_days", None),
        max_chunk_size_tokens=overrides.pop("max_chunk_size_tokens", 100),
        chunk_overlap_tokens=overrides.pop("chunk_overlap_tokens", 20),
    )


async def _attach(
    store: StoreRecord, file_ids: list[str], *, batch_id: str = "", size: int = 100
) -> list[str]:
    """Attach *file_ids* to *store* without indexing them yet.

    Returns:
        The identifiers of the created file records, in order.
    """
    records = await attach_files(
        store,
        [
            PendingFile(
                file_id=file_id, max_chunk_size_tokens=size, chunk_overlap_tokens=20
            )
            for file_id in file_ids
        ],
        batch_id=batch_id,
    )
    return [record.id for record in records]


@pytest.mark.local
class TestIndexingOffline:
    """The asynchronous indexer, driven deterministically against the fake backend.

    Ref: stdapi/vector_stores/engine.py:index_files
    """

    async def test_a_file_is_chunked_embedded_and_counted(
        self, vector_backend: _FakeBackend
    ) -> None:
        """Indexing embeds every chunk in waves and settles the store counters.

        Ref: stdapi/vector_stores/engine.py:_index_one_file
        """
        store = await _create_store()
        # Long enough to cross the embedding wave, which one chunk never does.
        content = "\n".join(f"Paragraph {index} of the manual." for index in range(200))
        file_id = vector_backend.upload(content.encode())
        await _attach(store, [file_id])
        vector_backend.model.waves.clear()
        await index_files(store.id, [file_id], "", "test-request")

        record = await read_file(store.id, file_id)
        assert record.status == "completed"
        assert record.last_error is None
        assert record.chunk_count > 16
        assert record.usage_bytes > 0
        assert len(vector_backend.model.waves) > 1
        assert max(vector_backend.model.waves) <= 16
        settled = await read_store(store.id)
        assert settled.status == "completed"
        assert settled.file_counts.completed == 1
        assert settled.file_counts.in_progress == 0
        assert settled.usage_bytes == record.usage_bytes
        chunks = await read_file_chunks(store.id, record)
        assert len(chunks) == record.chunk_count
        assert "Paragraph 0 " in chunks[0]
        assert "Paragraph 199" in chunks[-1]

    async def test_a_transient_failure_settles_one_file_and_the_loop_survives(
        self, vector_backend: _FakeBackend
    ) -> None:
        """A backend error on one file fails that file and indexes the rest.

        Nothing else settles these records: an error escaping the loop leaves
        every file after it — and the store — ``in_progress`` forever.

        Ref: stdapi/vector_stores/engine.py:index_files
        """
        store = await _create_store()
        files = [vector_backend.upload(_TEXT_FILE) for _ in range(3)]
        await _attach(store, files)
        vector_backend.records.fail_once[
            ("get_object", file_key(store.id, files[1]))
        ] = make_client_error("SlowDown", "GetObject", status=503)
        await index_files(store.id, files, "", "test-request")

        assert (await read_file(store.id, files[0])).status == "completed"
        assert (await read_file(store.id, files[2])).status == "completed"
        failed = await read_file(store.id, files[1])
        assert failed.status == "failed"
        assert failed.last_error is not None
        assert failed.last_error.code == "server_error"
        settled = await read_store(store.id)
        assert settled.file_counts.in_progress == 0
        assert settled.file_counts.completed == 2
        assert settled.file_counts.failed == 1
        assert settled.status == "completed"

    async def test_a_non_text_file_fails_and_the_store_still_settles(
        self, vector_backend: _FakeBackend
    ) -> None:
        """A file that is not text settles ``failed`` and leaves no store in progress.

        Ref: openai.types.vector_stores.vector_store_file.LastError
        """
        store = await _create_store()
        file_id = vector_backend.upload(red_png(), "image/png")
        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")

        record = await read_file(store.id, file_id)
        assert record.status == "failed"
        assert record.last_error is not None
        assert record.last_error.code == "unsupported_file"
        settled = await read_store(store.id)
        assert settled.status == "completed"
        assert settled.file_counts.failed == 1
        assert settled.file_counts.in_progress == 0
        assert settled.usage_bytes == 0

    async def test_cancelling_a_batch_stops_the_files_it_has_not_started(
        self, vector_backend: _FakeBackend
    ) -> None:
        """A cancelled batch settles the waiting files, keeping the finished ones.

        The live test cannot lose the race deterministically, so this is what
        proves the cancellation is honoured at all.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-file-batches/cancelBatch
        """
        store = await _create_store()
        batch_id = new_batch_id()
        files = [vector_backend.upload(_TEXT_FILE) for _ in range(2)]
        await _attach(store, files, batch_id=batch_id)
        await index_files(store.id, files[:1], batch_id, "test-request")
        await cancel_batch(store.id, batch_id)
        await index_files(store.id, files[1:], batch_id, "test-request")

        assert (await read_file(store.id, files[0])).status == "completed"
        assert (await read_file(store.id, files[1])).status == "cancelled"
        batch = await read_batch(store.id, batch_id)
        assert batch.status == "cancelled"
        assert batch.file_counts.completed == 1
        assert batch.file_counts.cancelled == 1
        assert batch.file_counts.in_progress == 0
        settled = await read_store(store.id)
        assert settled.file_counts.cancelled == 1
        assert settled.file_counts.in_progress == 0

    async def test_a_detached_file_is_not_settled_a_second_time(
        self, vector_backend: _FakeBackend, scheduled_cleanups: list[Awaitable[None]]
    ) -> None:
        """Detaching during indexing leaves no count the listing cannot show.

        Ref: stdapi/vector_stores/engine.py:_index_one_file
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        await detach_file(store.id, file_id)
        await _run_cleanups(scheduled_cleanups)
        await index_files(store.id, [file_id], "", "test-request")

        settled = await read_store(store.id)
        assert settled.file_counts.total == 0
        assert settled.file_counts.cancelled == 0
        assert settled.file_counts.in_progress == 0

    async def test_the_same_file_twice_in_one_request_is_attached_once(
        self, vector_backend: _FakeBackend
    ) -> None:
        """A repeated file identifier is counted once, so the totals can converge.

        Ref: stdapi/vector_stores/engine.py:attach_files
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        batch_id = new_batch_id()
        attached = await _attach(store, [file_id, file_id], batch_id=batch_id)
        assert attached == [file_id]
        await index_files(store.id, attached, batch_id, "test-request")
        settled = await read_store(store.id)
        assert settled.file_counts.total == 1
        assert (await read_batch(store.id, batch_id)).file_counts.total == 1

    async def test_reindexing_reclaims_the_chunks_of_the_previous_version(
        self, vector_backend: _FakeBackend
    ) -> None:
        """A file re-attached with larger chunks leaves no stale passage searchable.

        Ref: stdapi/vector_stores/engine.py:_index_one_file
        """
        store = await _create_store()
        content = "\n".join(f"Paragraph {index} of the manual." for index in range(60))
        file_id = vector_backend.upload(content.encode())
        await _attach(store, [file_id], size=100)
        await index_files(store.id, [file_id], "", "test-request")
        first = (await read_file(store.id, file_id)).chunk_count
        assert first > 1

        await _attach(store, [file_id], size=4096)
        await index_files(store.id, [file_id], "", "test-request")
        record = await read_file(store.id, file_id)
        assert record.chunk_count < first
        assert set(vector_backend.vectors.indexes[index_name(store.id)]) == {
            vector_key(file_id, index) for index in range(record.chunk_count)
        }
        settled = await read_store(store.id)
        assert settled.file_counts.total == 1
        assert settled.file_counts.completed == 1

    async def test_a_failed_reindex_still_reclaims_the_previous_vectors(
        self, vector_backend: _FakeBackend, scheduled_cleanups: list[Awaitable[None]]
    ) -> None:
        """Detaching a file whose re-indexing failed removes the vectors it still owns.

        Ref: stdapi/vector_stores/engine.py:detach_file
        """
        store = await _create_store()
        content = "\n".join(f"Paragraph {index} of the manual." for index in range(60))
        file_id = vector_backend.upload(content.encode())
        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")
        assert vector_backend.vectors.indexes[index_name(store.id)]

        vector_backend.uploads[file_id[5:]] = (red_png(), "image/png")
        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")
        assert (await read_file(store.id, file_id)).status == "failed"

        await detach_file(store.id, file_id)
        await _run_cleanups(scheduled_cleanups)
        assert vector_backend.vectors.indexes[index_name(store.id)] == {}

    async def test_updated_attributes_reach_the_stored_vectors(
        self, vector_backend: _FakeBackend, scheduled_cleanups: list[Awaitable[None]]
    ) -> None:
        """Replacing a file's attributes re-writes them onto its vectors.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/updateAttributes
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        records = await attach_files(
            store,
            [PendingFile(file_id=file_id, attributes={"topic": "catering"})],
            batch_id="",
        )
        await index_files(store.id, [records[0].id], "", "test-request")
        await update_file_attributes(store.id, file_id, {"topic": "menu"})
        await _run_cleanups(scheduled_cleanups)

        stored = vector_backend.vectors.indexes[index_name(store.id)].values()
        assert stored
        assert all(
            vector["metadata"][attribute_key("topic")] == "menu" for vector in stored
        )
        results = await search(
            await read_store(store.id),
            ["maintenance hatch"],
            max_num_results=5,
            filters=ComparisonFilter(key="topic", type="eq", value="menu"),
            score_threshold=None,
        )
        assert results
        assert results[0].attributes == {"topic": "menu"}

    async def test_a_multi_query_search_keeps_the_best_score_per_passage(
        self, vector_backend: _FakeBackend
    ) -> None:
        """Several queries are merged into one page, each passage scored at its best.

        Ref: stdapi/vector_stores/engine.py:search
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")
        record = await read_store(store.id)
        one = await search(
            record, [_PLANTED], max_num_results=10, filters=None, score_threshold=None
        )
        both = await search(
            record,
            ["cafeteria rota", _PLANTED],
            max_num_results=10,
            filters=None,
            score_threshold=None,
        )
        assert len(both) == len({result.text for result in both})
        assert max(result.score for result in both) == pytest.approx(
            max(result.score for result in one)
        )

    async def test_an_expired_store_answers_nothing_and_releases_its_index_once(
        self, vector_backend: _FakeBackend, scheduled_cleanups: list[Awaitable[None]]
    ) -> None:
        """Past its expiration a store returns no result and its index is reclaimed.

        Ref: openai.types.vector_store.VectorStore
        """
        store = await _create_store(expires_after_days=1)
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")
        await update_record(
            StoreRecord,
            store_key(store.id),
            lambda stored: setattr(stored, "last_active_at", 1),
        )

        expired = await read_store(store.id)
        assert expired.status == "expired"
        assert (
            await search(
                expired,
                [_PLANTED],
                max_num_results=5,
                filters=None,
                score_threshold=None,
            )
            == []
        )
        await _run_cleanups(scheduled_cleanups)
        assert vector_backend.vectors.deleted == [index_name(store.id)]

        # Released once: the second read must not schedule the delete again.
        assert (await read_store(store.id)).index_deleted
        assert not scheduled_cleanups

    async def test_searching_an_expired_store_does_not_resurrect_it(
        self, vector_backend: _FakeBackend, scheduled_cleanups: list[Awaitable[None]]
    ) -> None:
        """Refreshing the anchor of an expired store would leave it pointing at nothing.

        Ref: stdapi/vector_stores/engine.py:touch_store
        """
        store = await _create_store(expires_after_days=1)
        await update_record(
            StoreRecord,
            store_key(store.id),
            lambda stored: setattr(stored, "last_active_at", 1),
        )
        expired = await read_store(store.id)
        await _run_cleanups(scheduled_cleanups)
        await touch_store(expired)
        assert (await read_store(store.id)).status == "expired"

    async def test_deleting_a_store_removes_every_record_it_holds(
        self, vector_backend: _FakeBackend, scheduled_cleanups: list[Awaitable[None]]
    ) -> None:
        """Deletion is not stopped by a page boundary: no caller data is left behind.

        Ref: stdapi/vector_stores/records.py:all_record_keys
        """
        store = await _create_store()
        files = [vector_backend.upload(_OTHER_FILE) for _ in range(3)]
        await _attach(store, files, batch_id=new_batch_id())
        # One record per listing page, so the deletion has to paginate.
        vector_backend.records.page_size = 1
        await delete_store(store.id)
        await _run_cleanups(scheduled_cleanups)
        assert vector_backend.records.objects == {}
        assert vector_backend.vectors.indexes == {}


@pytest.mark.local
class TestDetachIsDurable:
    """A detached file's vectors go before the record that can reclaim them.

    Issue #177: the record was deleted first and the vectors left to a detached
    task, so a task killed in between left a document the caller had deleted
    searchable for good, with nothing left pointing at it. Every test here
    abandons the scheduled cleanup on purpose: that is the killed task.

    Ref: stdapi/vector_stores/engine.py:detach_file
    """

    async def test_a_lost_reclaim_leaves_the_vectors_to_a_later_read(
        self, vector_backend: _FakeBackend, scheduled_cleanups: list[Awaitable[None]]
    ) -> None:
        """A detach whose task never ran is finished by the next read of the store.

        Ref: stdapi/vector_stores/engine.py:_reclaim_file
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")
        index = vector_backend.vectors.indexes[index_name(store.id)]
        assert index

        await detach_file(store.id, file_id)
        # The task dies here, with the caller already told the file is gone.
        _abandon_cleanups(scheduled_cleanups)
        assert index, "the vectors outlived the record that names them"

        recovered = await read_store(store.id)
        assert vector_backend.vectors.indexes[index_name(store.id)] == {}
        assert recovered.detaching == []
        assert file_key(store.id, file_id) not in vector_backend.records.objects

    async def test_a_file_being_reclaimed_answers_as_deleted(
        self, vector_backend: _FakeBackend, scheduled_cleanups: list[Awaitable[None]]
    ) -> None:
        """Between the two deletes the file is searched, listed and read as gone.

        Ref: stdapi/vector_stores/engine.py:search
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")
        await detach_file(store.id, file_id)
        _abandon_cleanups(scheduled_cleanups)

        # Read raw: reading it through the engine would finish the reclaim.
        stored = await read_record(StoreRecord, store_key(store.id))
        assert stored is not None
        pending = stored[0]
        assert pending.detaching == [file_id]
        assert pending.file_counts.total == 0
        assert vector_backend.vectors.indexes[index_name(store.id)]
        assert (
            await search(
                pending,
                [_PLANTED],
                max_num_results=5,
                filters=None,
                score_threshold=None,
            )
            == []
        )
        assert await list_store_files(
            store.id, after="", before="", limit=10, order="desc", status=""
        ) == ([], False)
        with pytest.raises(ApiError) as raised:
            await read_file(store.id, file_id)
        assert raised.value.status == 404

    async def test_re_attaching_a_file_being_reclaimed_keeps_the_new_record(
        self, vector_backend: _FakeBackend, scheduled_cleanups: list[Awaitable[None]]
    ) -> None:
        """Attaching a file again finishes its reclaim first, so it is not deleted.

        Ref: stdapi/vector_stores/engine.py:attach_files
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")
        await detach_file(store.id, file_id)
        _abandon_cleanups(scheduled_cleanups)

        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")
        await _run_cleanups(scheduled_cleanups)
        record = await read_file(store.id, file_id)
        assert record.status == "completed"
        assert vector_backend.vectors.indexes[index_name(store.id)]
        settled = await read_store(store.id)
        assert settled.detaching == []
        assert settled.file_counts.completed == 1
        assert settled.file_counts.total == 1


@pytest.mark.local
class TestIndexingConcurrencyCap:
    """Indexing is bounded server-wide, whatever the request rate.

    Issue #178: one wave per attach, each buffering a whole file and running
    sixteen embedding calls, is a caller-controlled fan-out on a task with a
    fixed memory limit.

    Ref: stdapi/vector_stores/engine.py:_hold_slot
    """

    async def test_concurrent_waves_index_no_more_files_than_the_cap(
        self, vector_backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Four attaches at once still read and embed only two files at a time.

        Ref: stdapi/vector_stores/engine.py:_INDEXING_SLOTS
        """
        store = await _create_store()
        files = [vector_backend.upload(_TEXT_FILE) for _ in range(4)]
        await _attach(store, files)
        embedding = vector_backend.model.embed_text
        active = 0
        peak = 0

        async def _slow_embed(*args: Any, **kwargs: Any) -> EmbeddingResponse:  # noqa: ANN401
            """Embed, holding the slot long enough for the others to pile up."""
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await sleep(0.05)
                return await embedding(*args, **kwargs)
            finally:
                active -= 1

        monkeypatch.setattr(vector_backend.model, "embed_text", _slow_embed)
        await gather(
            *(index_files(store.id, [file_id], "", "test-request") for file_id in files)
        )

        # Four waves, and never four files at once: the cap is what bites.
        assert peak == engine._INDEXING_SLOTS < len(files)  # noqa: SLF001
        settled = await read_store(store.id)
        assert settled.file_counts.completed == 4
        assert settled.file_counts.in_progress == 0


@pytest.mark.local
class TestAbandonedIndexing:
    """A file whose indexing lost its task is settled by the next read.

    Issue #179: nothing sweeps these records, so a task killed mid-indexing —
    ECS sends ``SIGKILL`` 30 s after ``SIGTERM`` — left the file, and the store
    counting it, ``in_progress`` for good.

    Ref: stdapi/vector_stores/engine.py:_settle_abandoned_files
    """

    @staticmethod
    async def _abandon(store_id: str, file_id: str) -> None:
        """Age a file and its store past the lease, as a killed task leaves them."""
        await update_record(
            StoreRecord,
            store_key(store_id),
            lambda stored: setattr(stored, "indexing_expires_at", 1),
        )
        await update_record(
            FileRecord,
            file_key(store_id, file_id),
            lambda stored: setattr(stored, "created_at", 1),
            resource="file",
        )

    async def test_reading_the_store_settles_the_file_and_its_counters(
        self, vector_backend: _FakeBackend
    ) -> None:
        """A store polled after its indexing died reports the file as failed.

        Ref: stdapi/vector_stores/engine.py:_recover_store
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        await self._abandon(store.id, file_id)

        settled = await read_store(store.id)
        assert settled.status == "completed"
        assert settled.file_counts.in_progress == 0
        assert settled.file_counts.failed == 1
        record = await read_file(store.id, file_id)
        assert record.status == "failed"
        assert record.last_error is not None
        assert record.last_error.code == "server_error"
        assert "interrupted" in record.last_error.message

    async def test_a_live_indexing_is_left_alone(
        self, vector_backend: _FakeBackend
    ) -> None:
        """A file attached moments ago is owned by a task and must not be failed.

        Ref: stdapi/vector_stores/engine.py:_settle_abandoned_files
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        # The lease is gone, but the file is younger than one: still being indexed.
        await update_record(
            StoreRecord,
            store_key(store.id),
            lambda stored: setattr(stored, "indexing_expires_at", 1),
        )

        assert (await read_store(store.id)).file_counts.in_progress == 1
        assert (await read_file(store.id, file_id)).status == "in_progress"
        await index_files(store.id, [file_id], "", "test-request")
        assert (await read_file(store.id, file_id)).status == "completed"

    async def test_the_file_routes_report_the_settled_status(
        self, vector_backend: _FakeBackend, app_client: TestClient
    ) -> None:
        """Polling the API the documented way stops returning ``in_progress``.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/listFiles
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        await self._abandon(store.id, file_id)

        listed = app_client.get(f"/v1/vector_stores/{store.id}/files")
        assert listed.status_code == 200, listed.text
        assert [entry["status"] for entry in listed.json()["data"]] == ["failed"]
        retrieved = app_client.get(f"/v1/vector_stores/{store.id}")
        assert retrieved.json()["status"] == "completed"
        assert retrieved.json()["file_counts"]["failed"] == 1

    async def test_the_shutdown_drain_settles_what_it_cancels(
        self, vector_backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Indexing cut short by the drain deadline leaves no record in progress.

        Ref: stdapi/vector_stores/engine.py:drain_indexing
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        blocked = Event()

        async def _never_embed(*args: Any, **kwargs: Any) -> EmbeddingResponse:  # noqa: ANN401
            """Hang exactly as an embedding call the deadline interrupts does."""
            del args, kwargs
            await blocked.wait()
            raise AssertionError

        monkeypatch.setattr(vector_backend.model, "embed_text", _never_embed)
        start_indexing(store.id, [file_id], "")
        # Let the task reach the embedding call it will never come back from.
        await sleep(0)
        await sleep(0)

        assert await drain_indexing(0.0) == 1
        record = await read_file(store.id, file_id)
        assert record.status == "failed"
        assert record.last_error is not None
        assert "interrupted" in record.last_error.message
        assert (await read_store(store.id)).file_counts.in_progress == 0

    async def test_draining_nothing_costs_nothing(self) -> None:
        """A shutdown with no indexing in flight answers without a record read.

        Ref: stdapi/vector_stores/engine.py:drain_indexing
        """
        assert await drain_indexing(5.0) == 0


@pytest.mark.local
class TestFileSizeLimit:
    """``max_input_file_size`` bounds indexing, and zero still means unlimited.

    The zero-means-unlimited sense broke every indexing run once: read as a
    literal ceiling, a default deployment failed every file it was given.

    Ref: stdapi/vector_stores/engine.py:_load_chunks
    """

    async def test_zero_means_unlimited(
        self, vector_backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default of ``0`` indexes a file no explicit limit would allow.

        Ref: stdapi/config.py:Settings.max_input_file_size
        """
        monkeypatch.setattr(SETTINGS, "max_input_file_size", 0)
        store = await _create_store()
        file_id = vector_backend.upload(b"word " * 200_000)
        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")
        record = await read_file(store.id, file_id)
        assert record.status == "completed"
        assert record.chunk_count > 1

    async def test_a_file_above_the_limit_fails_as_invalid_file(
        self, vector_backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A configured limit reports the file as ``invalid_file``, not as an error.

        Ref: openai.types.vector_stores.vector_store_file.LastError
        """
        monkeypatch.setattr(SETTINGS, "max_input_file_size", 10)
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")
        record = await read_file(store.id, file_id)
        assert record.status == "failed"
        assert record.last_error is not None
        assert record.last_error.code == "invalid_file"
        settled = await read_store(store.id)
        assert settled.file_counts.failed == 1
        assert settled.file_counts.in_progress == 0


@pytest.mark.local
class TestEmbeddingModelLimits:
    """The chunker never hands the embedding model more than it accepts.

    Ref: stdapi/models/embedding/cohere_embed.py:EmbeddingModel.max_input_characters
    """

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("cohere.embed-english-v3", 2048),
            ("cohere.embed-multilingual-v3", 2048),
            ("cohere.embed-v4:0", 0),
            ("cohere.embed-v5:0", 0),
        ],
    )
    def test_the_character_ceiling_is_the_bounded_family_s(
        self, model_id: str, expected: int
    ) -> None:
        """Only the v3 family caps an input; a later version inherits no ceiling.

        Ref: https://docs.cohere.com/reference/embed
        """
        assert CohereEmbeddingModel(model_id).max_input_characters == expected

    async def test_chunks_are_clamped_to_what_the_model_accepts(
        self, vector_backend: _FakeBackend
    ) -> None:
        """A model with a character ceiling caps the chunk, whatever the token budget.

        Ref: stdapi/vector_stores/engine.py:_load_chunks
        """
        vector_backend.model.max_input_characters = 200
        store = await _create_store(max_chunk_size_tokens=4096)
        file_id = vector_backend.upload(b"word " * 2000)
        await _attach(store, [file_id], size=4096)
        await index_files(store.id, [file_id], "", "test-request")
        record = await read_file(store.id, file_id)
        chunks = await read_file_chunks(store.id, record)
        assert chunks
        assert all(len(chunk) <= 200 for chunk in chunks)


@pytest.mark.local
class TestBackendCapabilities:
    """The engine refuses what the backend serving a store does not declare.

    Another backend expresses fewer filter operators, ingests other formats,
    cuts its own passages, or scores on a scale of its own. Each of those is a
    declaration the engine reads before it calls, so the gap is a clean 400 or
    a settled ``unsupported_file`` rather than the backend's own error reaching
    the caller mid-search.

    Ref: stdapi/vector_stores/backend.py:IndexCapabilities
    """

    def test_the_shipped_backend_declares_the_upstream_dialect_whole(self) -> None:
        """S3 Vectors expresses every operator and combinator the API accepts.

        Nothing is degraded on the default deployment: this pins the
        declaration, so an operator quietly dropped from it becomes a failure
        here rather than a 400 on a request that used to work.

        Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-metadata-filtering.html
        """
        assert _S3_CAPABILITIES.filter_operators == frozenset(
            {"eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"}
        )
        assert _S3_CAPABILITIES.filter_combinators == frozenset({"and", "or"})
        assert _S3_CAPABILITIES.normalised_score
        assert not _S3_CAPABILITIES.chunks_on_ingestion
        assert _S3_CAPABILITIES.ingests_decodable_text
        assert _S3_CAPABILITIES.ingested_media_types == frozenset()

    async def test_an_undeclared_operator_never_reaches_the_backend(
        self, fake_index: Callable[..., _FakeVectorIndex], vector_backend: _FakeBackend
    ) -> None:
        """A filter the backend cannot express is refused before any work is paid for.

        Ref: stdapi/vector_stores/backend.py:check_filter
        """
        index = fake_index(filter_operators=frozenset({"eq", "ne"}))
        store = await _create_store()
        vector_backend.model.waves.clear()
        with pytest.raises(ApiError) as raised:
            await search(
                store,
                ["anything"],
                max_num_results=5,
                filters=ComparisonFilter(key="year", type="gte", value=2020),
                score_threshold=None,
            )
        assert raised.value.status == 400
        message = str(raised.value)
        assert "gte" in message
        # The index dialect and the internal metadata key stay out of it.
        assert "$gte" not in message
        assert attribute_key("year") not in message
        assert index.queries == 0
        assert vector_backend.model.waves == []

    async def test_an_undeclared_combinator_is_refused(
        self, fake_index: Callable[..., _FakeVectorIndex]
    ) -> None:
        """A backend expressing only ``and`` refuses an ``or`` filter.

        Ref: stdapi/vector_stores/backend.py:check_filter
        """
        index = fake_index(filter_combinators=frozenset({"and"}))
        store = await _create_store()
        with pytest.raises(ApiError) as raised:
            await search(
                store,
                ["anything"],
                max_num_results=5,
                filters=CompoundFilter(
                    type="or",
                    filters=[ComparisonFilter(key="topic", type="eq", value="x")],
                ),
                score_threshold=None,
            )
        assert raised.value.status == 400
        assert "or" in str(raised.value)
        assert index.queries == 0

    async def test_an_operator_nested_two_levels_deep_is_checked_too(
        self, fake_index: Callable[..., _FakeVectorIndex]
    ) -> None:
        """Nesting is carried as a plain object, and is walked all the same.

        Ref: stdapi/vector_stores/backend.py:check_filter
        """
        index = fake_index(filter_operators=frozenset({"eq"}))
        store = await _create_store()
        with pytest.raises(ApiError) as raised:
            await search(
                store,
                ["anything"],
                max_num_results=5,
                filters=CompoundFilter(
                    type="and",
                    filters=[
                        ComparisonFilter(key="topic", type="eq", value="x"),
                        {
                            "type": "or",
                            "filters": [{"key": "year", "type": "in", "value": [1, 2]}],
                        },
                    ],
                ),
                score_threshold=None,
            )
        assert raised.value.status == 400
        assert "'in'" in str(raised.value)
        assert index.queries == 0

    async def test_a_threshold_needs_a_score_that_means_the_same_every_time(
        self, fake_index: Callable[..., _FakeVectorIndex], vector_backend: _FakeBackend
    ) -> None:
        """A backend whose score is not normalised refuses ``score_threshold``.

        Dropping the threshold silently would answer with results the caller
        asked to be excluded, which is not a usable version of the request.

        Ref: stdapi/vector_stores/backend.py:IndexCapabilities
        """
        index = fake_index(normalised_score=False)
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")
        with pytest.raises(ApiError) as raised:
            await search(
                store, [_PLANTED], max_num_results=5, filters=None, score_threshold=0.5
            )
        assert raised.value.status == 400
        assert "score_threshold" in str(raised.value)
        assert index.queries == 0
        # The same search without a threshold is served.
        assert await search(
            store, [_PLANTED], max_num_results=5, filters=None, score_threshold=None
        )

    def test_a_backend_that_cuts_its_own_passages_refuses_a_chunking_strategy(
        self, app_client: TestClient, fake_index: Callable[..., _FakeVectorIndex]
    ) -> None:
        """A per-request chunk size cannot be honoured where the backend chunks.

        Ref: stdapi/vector_stores/engine.py:check_chunking_strategy
        """
        fake_index(chunks_on_ingestion=True)
        response = app_client.post(
            "/v1/vector_stores",
            json={
                "chunking_strategy": {
                    "type": "static",
                    "static": {
                        "max_chunk_size_tokens": 100,
                        "chunk_overlap_tokens": 20,
                    },
                }
            },
        )
        assert response.status_code == 400, response.text
        message = _error_of(response)["message"]
        assert "chunking_strategy" in message
        # The default strategy is still accepted, so the store can be created.
        assert app_client.post("/v1/vector_stores", json={}).status_code == 200

    async def test_a_media_type_the_backend_refuses_settles_as_unsupported_file(
        self, fake_index: Callable[..., _FakeVectorIndex], vector_backend: _FakeBackend
    ) -> None:
        """The refused-format list is the backend's, not the engine's.

        Ref: openai.types.vector_stores.vector_store_file.LastError
        """
        fake_index(refused_media_types=frozenset({"text/csv"}))
        store = await _create_store()
        file_id = vector_backend.upload(b"a,b\n1,2\n", "text/csv")
        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")
        record = await read_file(store.id, file_id)
        assert record.status == "failed"
        assert record.last_error is not None
        assert record.last_error.code == "unsupported_file"

    async def test_a_backend_taking_no_text_refuses_a_text_file(
        self, fake_index: Callable[..., _FakeVectorIndex], vector_backend: _FakeBackend
    ) -> None:
        """A backend ingesting only named formats refuses everything else.

        Ref: openai.types.vector_stores.vector_store_file.LastError
        """
        fake_index(
            ingests_decodable_text=False,
            ingested_media_types=frozenset({"application/pdf"}),
            refused_media_types=frozenset(),
        )
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")
        record = await read_file(store.id, file_id)
        assert record.status == "failed"
        assert record.last_error is not None
        assert record.last_error.code == "unsupported_file"

    async def test_the_engine_serves_a_whole_store_through_the_protocol(
        self,
        fake_index: Callable[..., _FakeVectorIndex],
        vector_backend: _FakeBackend,
        scheduled_cleanups: list[Awaitable[None]],
    ) -> None:
        """Attach, index, search, re-attribute, detach and delete, with no AWS call.

        The backend here shares no code with the shipped one, so a store that
        round-trips through it is what proves the engine holds nothing specific
        to a service.

        Ref: stdapi/vector_stores/backend.py:VectorIndex
        """
        index = fake_index()
        store = await _create_store()
        assert index.indexes[store.id] == {}
        file_id = vector_backend.upload(_TEXT_FILE)
        records_written = await attach_files(
            store,
            [PendingFile(file_id=file_id, attributes={"topic": "observatory"})],
            batch_id="",
        )
        await index_files(store.id, [records_written[0].id], "", "test-request")

        record = await read_file(store.id, file_id)
        assert record.status == "completed"
        assert record.chunk_count > 0
        assert set(index.indexes[store.id]) == {
            vector_key(file_id, position) for position in range(record.chunk_count)
        }
        chunks = await read_file_chunks(store.id, record)
        assert "QUINCEY-7" in "\n".join(chunks)

        results = await search(
            store,
            [_PLANTED],
            max_num_results=5,
            filters=ComparisonFilter(key="topic", type="eq", value="observatory"),
            score_threshold=None,
        )
        assert results
        assert results[0].file_id == file_id
        assert results[0].attributes == {"topic": "observatory"}
        assert 0.0 < results[0].score <= 1.0

        await update_file_attributes(store.id, file_id, {"topic": "menu"})
        await _run_cleanups(scheduled_cleanups)
        assert all(
            vector.attributes == {"topic": "menu"}
            for vector in index.indexes[store.id].values()
        )

        await detach_file(store.id, file_id)
        await _run_cleanups(scheduled_cleanups)
        assert index.indexes[store.id] == {}
        await delete_store(store.id)
        await _run_cleanups(scheduled_cleanups)
        assert index.deleted == [store.id]


@pytest.mark.local
class TestUnsupportedFileMessage:
    """A refused file is explained in the terms of the store that refused it.

    Two backends disagree about what they take, so one fixed sentence would
    describe the wrong store on one of them. The explanation is built from the
    serving backend's own declaration and names where the file would be indexed
    instead, while the settled shape — ``failed`` with ``unsupported_file`` —
    stays exactly what upstream defines.

    Ref: stdapi/vector_stores/backend.py:unsupported_file_message
         openai.types.vector_stores.vector_store_file.LastError
    """

    async def test_a_document_format_names_the_store_that_would_take_it(
        self, vector_backend: _FakeBackend
    ) -> None:
        """A PDF refused by a text-only store is pointed at a store that indexes it.

        Ref: stdapi/vector_stores/engine.py:_load_chunks
        """
        store = await _create_store()
        file_id = vector_backend.upload(b"%PDF-1.7\x00binary", "application/pdf")
        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")

        record = await read_file(store.id, file_id)
        assert record.status == "failed"
        assert record.last_error is not None
        assert record.last_error.code == "unsupported_file"
        message = record.last_error.message
        assert "text" in message
        assert "knowledge base store" in message
        # What another backend takes is never listed as what this store takes.
        assert "application/msword" not in message

    async def test_a_format_no_store_takes_offers_no_alternative(
        self, vector_backend: _FakeBackend
    ) -> None:
        """An archive is refused with nowhere else to send it.

        Ref: stdapi/vector_stores/backend.py:unsupported_file_message
        """
        store = await _create_store()
        file_id = vector_backend.upload(b"PK\x03\x04binary", "application/zip")
        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")

        record = await read_file(store.id, file_id)
        assert record.status == "failed"
        assert record.last_error is not None
        assert record.last_error.code == "unsupported_file"
        assert "knowledge base" not in record.last_error.message

    def test_the_same_file_is_explained_differently_per_backend(self) -> None:
        """The two shipped backends refuse the same archive with different sentences.

        Ref: stdapi/vector_stores/knowledge_base.py:CAPABILITIES
        """
        text_only = unsupported_file_message(_S3_CAPABILITIES)
        documents = unsupported_file_message(_KB_CAPABILITIES)
        assert text_only != documents
        assert "application/pdf" not in text_only
        assert "application/pdf" in documents

    async def test_the_message_follows_the_backend_serving_the_store(
        self, fake_index: Callable[..., _FakeVectorIndex], vector_backend: _FakeBackend
    ) -> None:
        """A backend declaring its own formats is described by them, not by a constant.

        Ref: stdapi/vector_stores/backend.py:IndexCapabilities
        """
        fake_index(
            ingests_decodable_text=False,
            ingested_media_types=frozenset({"application/x-parquet"}),
            refused_media_types=frozenset(),
        )
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        await index_files(store.id, [file_id], "", "test-request")

        record = await read_file(store.id, file_id)
        assert record.status == "failed"
        assert record.last_error is not None
        assert record.last_error.code == "unsupported_file"
        assert "application/x-parquet" in record.last_error.message


#: A well-formed queue URL, the one shape the setting accepts.
_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/123456789012/stdapi-test-indexing"


class _FakeSqsClient:
    """The queue operations an indexing job uses, in memory.

    Substituted at the AWS client, so the send, the receive, the visibility
    heartbeat, the delete and the redelivery accounting are the code under
    test — only the service behind them is stood in for.
    """

    def __init__(self) -> None:
        #: Messages waiting to be received, oldest first.
        self.pending: list[dict[str, Any]] = []
        #: Messages handed out and not deleted yet, by receipt handle.
        self.in_flight: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []
        self.kept_invisible: list[str] = []
        self.attributes: dict[str, str] = {}
        #: Raised by the next send, standing in for a denied or broken queue.
        self.send_error: Exception | None = None
        self._receipts = count()

    async def send_message(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        if self.send_error is not None:
            raise self.send_error
        self.pending.append({"Body": params["MessageBody"], "receives": 0})
        return {"MessageId": "message"}

    async def receive_message(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        del params
        if not self.pending:
            return {}
        entry = self.pending.pop(0)
        entry["receives"] += 1
        receipt = f"receipt-{next(self._receipts)}"
        self.in_flight[receipt] = entry
        return {"Messages": [self._message(receipt, entry)]}

    async def delete_message(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.in_flight.pop(params["ReceiptHandle"], None)
        self.deleted.append(params["ReceiptHandle"])
        return {}

    async def change_message_visibility(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.kept_invisible.append(params["ReceiptHandle"])
        return {}

    async def get_queue_attributes(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        del params
        return {"Attributes": dict(self.attributes)}

    @staticmethod
    def _message(receipt: str, entry: dict[str, Any]) -> dict[str, Any]:
        """Shape one delivery as ``ReceiveMessage`` answers it."""
        return {
            "MessageId": "message",
            "ReceiptHandle": receipt,
            "Body": entry["Body"],
            "Attributes": {"ApproximateReceiveCount": str(entry["receives"])},
        }

    def redeliver(self) -> None:
        """Return what a killed consumer never deleted, as the timeout does."""
        self.pending.extend(self.in_flight.values())
        self.in_flight.clear()

    async def next_message(self) -> dict[str, Any]:
        """Receive the next message, failing when the queue holds none."""
        received: list[dict[str, Any]] = (await self.receive_message()).get(
            "Messages", []
        )
        assert received, "the queue holds no message"
        return received[0]


@pytest.fixture
def indexing_queue(
    vector_backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> _FakeSqsClient:
    """Configure the indexing queue and serve it from memory.

    Returns:
        The queue the jobs are published to and consumed from.
    """
    del vector_backend
    client = _FakeSqsClient()
    monkeypatch.setattr(SETTINGS, "aws_sqs_vector_store_queue_url", _QUEUE_URL)
    monkeypatch.setattr(jobs, "_client", lambda: client)
    # What the startup probe reads off the queue is process-global state.
    monkeypatch.setattr(jobs, "_MAX_RECEIVES", jobs._DEFAULT_MAX_RECEIVES)  # noqa: SLF001
    monkeypatch.setattr(jobs, "_HAS_DEAD_LETTER_QUEUE", False)
    return client


def _job_body(**overrides: Any) -> str:  # noqa: ANN401
    """Return a message body, valid unless a case overrides a field."""
    body: dict[str, Any] = {
        "type": "vector_store.index_files",
        "store_id": new_store_id(),
        "file_ids": [f"file-{0:032d}"],
        "batch_id": "",
        "request_id": "test-request",
    }
    body.update(overrides)
    return to_json_str({key: value for key, value in body.items() if value is not None})


@pytest.mark.local
class TestIndexingIsHandedOver:
    """Where an indexing wave goes once its records are durable.

    The send is the last thing an attach does, and it happens inside the
    request: every file record, the store counters and the batch record are
    already written under conditional writes by then, and the file bytes have
    been durable since they were uploaded, so the job the message names is
    replayable from the moment it is sent.

    Ref: stdapi/vector_stores/engine.py:attach_files
         stdapi/vector_stores/jobs.py:enqueue_indexing
    """

    async def test_without_a_queue_the_attaching_server_indexes(
        self, vector_backend: _FakeBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default deployment behaves exactly as it did before the queue.

        No client is constructed either: an unset setting must not reach AWS.
        """

        def _refuse() -> None:
            opened = "an unconfigured deployment opened a queue client"
            raise AssertionError(opened)

        monkeypatch.setattr(jobs, "_client", _refuse)
        assert SETTINGS.aws_sqs_vector_store_queue_url is None
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)

        await _attach(store, [file_id])

        assert vector_backend.started == [(store.id, "", (file_id,))]

    async def test_an_attach_publishes_the_wave_instead_of_running_it(
        self, vector_backend: _FakeBackend, indexing_queue: _FakeSqsClient
    ) -> None:
        """With a queue the wave leaves the request, so any server can run it."""
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        batch_id = new_batch_id()

        await _attach(store, [file_id], batch_id=batch_id)

        assert not vector_backend.started
        (message,) = indexing_queue.pending
        job = IndexFilesJob.model_validate_json(message["Body"])
        assert job.type == "vector_store.index_files"
        assert job.store_id == store.id
        assert job.file_ids == [file_id]
        assert job.batch_id == batch_id

    async def test_the_message_carries_no_caller_content(
        self, vector_backend: _FakeBackend, indexing_queue: _FakeSqsClient
    ) -> None:
        """A job names what to index; it never copies it.

        Anything else would put caller data in a second store at rest, and give
        a message body worth tampering with.
        """
        store = await _create_store()
        file_id = vector_backend.upload(b"the observatory dome is under maintenance")

        await _attach(store, [file_id])

        body = indexing_queue.pending[0]["Body"]
        assert "observatory" not in body
        assert set(from_json(body)) == {
            "type",
            "store_id",
            "file_ids",
            "batch_id",
            "request_id",
        }

    async def test_a_queue_that_refuses_the_send_indexes_here_instead(
        self, vector_backend: _FakeBackend, indexing_queue: _FakeSqsClient
    ) -> None:
        """The attach still succeeds: the files are attached either way.

        Failing the request would be a lie — the records are written — and
        stranding the wave would leave files nothing ever settles. Running it
        here is what a deployment without a queue always does.
        """
        indexing_queue.send_error = make_client_error("AccessDenied", "SendMessage")
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)

        await _attach(store, [file_id])

        assert vector_backend.started == [(store.id, "", (file_id,))]
        assert not indexing_queue.pending

    async def test_a_wave_larger_than_a_message_may_name_is_indexed_here(
        self, indexing_queue: _FakeSqsClient
    ) -> None:
        """One message cannot fan out without bound, so an outsized wave stays.

        Ref: stdapi/vector_stores/jobs.py:MAX_JOB_FILES
        """
        oversized = [f"file-{index:032d}" for index in range(MAX_JOB_FILES + 1)]

        assert await jobs.enqueue_indexing(new_store_id(), oversized, "") is False
        assert not indexing_queue.pending


@pytest.mark.local
class TestAJobMessageIsUntrusted:
    """What comes off the queue is data, never instruction.

    Only this server's own role may write to the queue, and the message is
    still validated as if it could not be: the type selects an entry of a
    mapping built at import, unknown fields are refused, and every identifier
    is re-parsed before it can name an object key in the records bucket.

    Ref: stdapi/vector_stores/jobs.py:_parse_job
    """

    def test_a_type_no_handler_answers_for_is_refused(self) -> None:
        """A job type is looked up, never resolved into code."""
        handler, job = jobs._parse_job(_job_body(type="vector_store.delete_store"))  # noqa: SLF001
        assert handler is None
        assert job is None

    def test_a_body_that_is_not_a_job_object_is_refused(self) -> None:
        """Anything that is not a JSON object names no handler at all."""
        assert jobs._parse_job("not json at all") == (None, None)  # noqa: SLF001
        assert jobs._parse_job('["vector_store.index_files"]') == (None, None)  # noqa: SLF001

    def test_an_unknown_field_is_refused(self) -> None:
        """``extra="forbid"`` keeps a job to the fields the handler reads."""
        handler, job = jobs._parse_job(_job_body(bucket="somebody-elses-bucket"))  # noqa: SLF001
        assert handler is None
        assert job is None

    @pytest.mark.parametrize(
        "store_id",
        [
            "../../../etc/passwd",
            "vs_../store",
            "vs_kb_ABCDE12345",
            "vs_" + "a" * 26 + "/../other",
            "vs_" + "z" * 400,
        ],
    )
    def test_a_store_identifier_this_server_would_not_mint_is_refused(
        self, store_id: str
    ) -> None:
        """The store identifier becomes an object key prefix in the records bucket.

        The knowledge-base identifier is in the list on purpose: it is only
        addressable when the deployment allowlisted it, and the queue must not
        become a way around that.
        """
        with pytest.raises(ValidationError, match="store_id"):
            IndexFilesJob.model_validate_json(_job_body(store_id=store_id))

    @pytest.mark.parametrize(
        "file_ids", [["../../secret"], ["file-" + "0" * 32, "not-a-file-id"], []]
    )
    def test_a_file_identifier_this_server_would_not_mint_is_refused(
        self, file_ids: list[str]
    ) -> None:
        """Every entry becomes an object key, so every entry is re-parsed."""
        with pytest.raises(ValidationError, match="file_ids"):
            IndexFilesJob.model_validate_json(_job_body(file_ids=file_ids))

    def test_more_files_than_a_job_may_name_are_refused(self) -> None:
        """A message naming a hundred thousand files is a denial of service."""
        with pytest.raises(ValidationError, match="file_ids"):
            IndexFilesJob.model_validate_json(
                _job_body(
                    file_ids=[
                        f"file-{index:032d}" for index in range(MAX_JOB_FILES + 1)
                    ]
                )
            )

    def test_a_batch_identifier_this_server_would_not_mint_is_refused(self) -> None:
        """The batch identifier becomes an object key too."""
        with pytest.raises(ValidationError, match="batch_id"):
            IndexFilesJob.model_validate_json(_job_body(batch_id="../batches/other"))

    def test_a_request_identifier_is_kept_to_a_logging_charset(self) -> None:
        """It is only ever written to a log, and only as itself."""
        with pytest.raises(ValidationError, match="request_id"):
            IndexFilesJob.model_validate_json(_job_body(request_id="a\nb: injected"))

    async def test_a_message_no_handler_answers_for_is_taken_off_the_queue(
        self, indexing_queue: _FakeSqsClient
    ) -> None:
        """Keeping it would only buy another delivery of the same rejection."""
        await jobs._run_job(  # noqa: SLF001
            {"ReceiptHandle": "poison", "Body": _job_body(type="something.else")}
        )

        assert indexing_queue.deleted == ["poison"]


@pytest.mark.local
class TestAKilledJobIsFinishedElsewhere:
    """The requirement the queue exists for.

    A server killed mid-indexing never deletes the message, so the job becomes
    visible again and the next consumer — in another task, in another
    availability zone — finishes it. The caller sees one indexing that took
    longer, not a file it has to attach again.

    Ref: stdapi/vector_stores/jobs.py:_run_job
    """

    @staticmethod
    def _blocked_embedding(backend: _FakeBackend) -> tuple[Event, Event]:
        """Make the first embedding call block until the test releases it.

        Returns:
            ``(started, release)``: set when the embedding began, and the event
            the test sets to let it finish.
        """
        started = Event()
        release = Event()
        embed = backend.model.embed_text

        async def _blocking(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            started.set()
            await release.wait()
            return await embed(*args, **kwargs)

        backend.model.embed_text = _blocking  # type: ignore[method-assign]
        return started, release

    async def test_a_second_consumer_completes_what_a_killed_one_started(
        self, vector_backend: _FakeBackend, indexing_queue: _FakeSqsClient
    ) -> None:
        """The file reaches ``completed`` although the server that took it died.

        The kill is the real one this design assumes: the task disappears
        between reading the file and writing its outcome, with nothing running
        to settle anything.
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        started, release = self._blocked_embedding(vector_backend)

        killed = create_task(jobs._run_job(await indexing_queue.next_message()))  # noqa: SLF001
        await started.wait()
        killed.cancel()
        with pytest.raises(CancelledError):
            await killed

        # Nothing settled it, and nothing took it off the queue.
        assert (await read_file(store.id, file_id)).status == "in_progress"
        assert not indexing_queue.deleted

        release.set()
        indexing_queue.redeliver()
        message = await indexing_queue.next_message()
        assert message["Attributes"]["ApproximateReceiveCount"] == "2"
        await jobs._run_job(message)  # noqa: SLF001

        record = await read_file(store.id, file_id)
        assert record.status == "completed"
        assert record.chunk_count
        assert indexing_queue.deleted == [message["ReceiptHandle"]]
        assert (await read_store(store.id)).file_counts.completed == 1

    async def test_a_replayed_job_neither_re_embeds_nor_re_counts(
        self, vector_backend: _FakeBackend, indexing_queue: _FakeSqsClient
    ) -> None:
        """At-least-once delivery must cost the operator nothing the second time.

        Every terminal state is reached under a compare-and-set write, so a
        replay finds the file already settled and stops before the embedding
        call — which is the one that bills.
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        body = indexing_queue.pending[0]["Body"]

        await jobs._run_job(await indexing_queue.next_message())  # noqa: SLF001
        first = await read_store(store.id)
        waves = len(vector_backend.model.waves)

        await jobs._run_job({"ReceiptHandle": "replay", "Body": body})  # noqa: SLF001

        assert len(vector_backend.model.waves) == waves
        second = await read_store(store.id)
        assert second.file_counts.model_dump() == first.file_counts.model_dump()
        assert second.usage_bytes == first.usage_bytes

    async def test_a_job_is_kept_invisible_for_as_long_as_it_runs(
        self, vector_backend: _FakeBackend, indexing_queue: _FakeSqsClient
    ) -> None:
        """The base visibility is short, so the heartbeat is what holds the job.

        A long base timeout would hide a job a server was killed five seconds
        into for the whole of it, while the caller polls a file nobody indexes.
        """
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(jobs, "_HEARTBEAT_SECONDS", 0)
        try:
            store = await _create_store()
            file_id = vector_backend.upload(_TEXT_FILE)
            await _attach(store, [file_id])
            started, release = self._blocked_embedding(vector_backend)

            message = await indexing_queue.next_message()
            running = create_task(jobs._run_job(message))  # noqa: SLF001
            await started.wait()
            await sleep(0)
            release.set()
            await running
        finally:
            monkeypatch.undo()

        assert message["ReceiptHandle"] in indexing_queue.kept_invisible
        assert (await read_file(store.id, file_id)).status == "completed"

    async def test_a_job_that_waited_out_its_lease_is_still_run(
        self, vector_backend: _FakeBackend, indexing_queue: _FakeSqsClient
    ) -> None:
        """A queued job renews the lease before the store read that would void it.

        Files left ``in_progress`` past their store's lease are settled as
        abandoned by the next read — and the first thing a job does is read the
        store, so without the renewal a job would settle the very files it was
        sent to index.

        Ref: stdapi/vector_stores/engine.py:renew_indexing_lease
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        # The lease the attach wrote has run out and the file is old enough to
        # count as abandoned: exactly a job that waited behind a busy fleet.
        await update_record(
            StoreRecord,
            records.store_key(store.id),
            lambda stored: setattr(stored, "indexing_expires_at", 0),
        )
        await update_record(
            FileRecord,
            records.file_key(store.id, file_id),
            lambda stored: setattr(stored, "created_at", 0),
        )

        await jobs._run_job(await indexing_queue.next_message())  # noqa: SLF001

        assert (await read_file(store.id, file_id)).status == "completed"


@pytest.mark.local
class TestAJobThatRunsOutOfDeliveries:
    """A job nobody will run again still owes the caller an answer.

    Without this the queue reintroduces the stranded ``in_progress`` state:
    a poison file would be retried until the message expires, billing its
    embeddings every time, and the record would never leave ``in_progress``.

    Ref: stdapi/vector_stores/jobs.py:_give_up
    """

    async def test_the_files_are_settled_and_the_message_dropped(
        self,
        vector_backend: _FakeBackend,
        indexing_queue: _FakeSqsClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Past the last delivery the wave is failed, never retried again."""
        monkeypatch.setattr(jobs, "_MAX_RECEIVES", 2)
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        body = indexing_queue.pending[0]["Body"]
        waves = len(vector_backend.model.waves)

        await jobs._run_job(  # noqa: SLF001
            {
                "ReceiptHandle": "exhausted",
                "Body": body,
                "Attributes": {"ApproximateReceiveCount": "3"},
            }
        )

        record = await read_file(store.id, file_id)
        assert record.status == "failed"
        assert record.last_error is not None
        assert record.last_error.code == "server_error"
        assert "interrupted" in record.last_error.message
        # Nothing was embedded on the way to giving up: the give-up is not a run.
        assert len(vector_backend.model.waves) == waves
        assert indexing_queue.deleted == ["exhausted"]
        assert (await read_store(store.id)).file_counts.failed == 1

    async def test_a_delivery_that_is_not_the_last_still_indexes(
        self, vector_backend: _FakeBackend, indexing_queue: _FakeSqsClient
    ) -> None:
        """A redelivery is the recovery, so it must not be read as a give-up."""
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        indexing_queue.redeliver()
        await indexing_queue.next_message()
        indexing_queue.redeliver()

        message = await indexing_queue.next_message()
        assert message["Attributes"]["ApproximateReceiveCount"] == "2"
        await jobs._run_job(message)  # noqa: SLF001

        assert (await read_file(store.id, file_id)).status == "completed"


@pytest.mark.local
class TestTheQueueIsProbedAtStartup:
    """What the queue promises is read once, never guessed at per message.

    Ref: stdapi/vector_stores/jobs.py:initialize_job_queue
    """

    async def test_the_redrive_policy_decides_how_many_deliveries_a_job_gets(
        self, indexing_queue: _FakeSqsClient
    ) -> None:
        """The queue's own configuration is the authority, not a constant here."""
        indexing_queue.attributes["RedrivePolicy"] = to_json_str(
            {
                "deadLetterTargetArn": "arn:aws:sqs:us-east-1:123456789012:dlq",
                "maxReceiveCount": 7,
            }
        )
        start_event: Any = {"level": "info"}

        await jobs.initialize_job_queue(start_event)

        assert jobs._MAX_RECEIVES == 7  # noqa: SLF001
        assert "server_warnings" not in start_event

    async def test_a_queue_without_a_dead_letter_queue_is_reported(
        self, indexing_queue: _FakeSqsClient
    ) -> None:
        """The operator is told by name what their queue is missing.

        The caller is told nothing: a file that cannot be indexed reports the
        same error either way.
        """
        del indexing_queue
        start_event: Any = {"level": "info"}

        await jobs.initialize_job_queue(start_event)

        assert start_event["level"] == "warning"
        (warning,) = start_event["server_warnings"]
        assert "dead-letter queue" in warning
        assert jobs._MAX_RECEIVES == jobs._DEFAULT_MAX_RECEIVES  # noqa: SLF001

    async def test_a_queue_that_cannot_be_described_is_reported_and_still_used(
        self, indexing_queue: _FakeSqsClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One missing read permission must not turn into a refused startup."""

        async def _denied(**params: Any) -> dict[str, Any]:  # noqa: ANN401
            del params
            denial = make_client_error("AccessDenied", "GetQueueAttributes")
            raise denial

        monkeypatch.setattr(indexing_queue, "get_queue_attributes", _denied)
        start_event: Any = {"level": "info"}

        await jobs.initialize_job_queue(start_event)

        (warning,) = start_event["server_warnings"]
        assert "sqs:GetQueueAttributes" in warning

    async def test_nothing_is_probed_without_a_queue(self) -> None:
        """An unconfigured deployment makes no call at all."""
        start_event: Any = {"level": "info"}

        await jobs.initialize_job_queue(start_event)

        assert start_event == {"level": "info"}


@pytest.mark.local
class TestTheConsumerYieldsToRequests:
    """Asyncio has no priorities, so the only one a background loop has is to wait.

    Ref: stdapi/vector_stores/jobs.py:_consume
    """

    async def test_a_busy_server_does_not_ask_the_queue_for_work(
        self, indexing_queue: _FakeSqsClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Minutes of embedding are not started while clients are waiting."""
        monkeypatch.setattr(jobs, "requests_in_flight", lambda: jobs._BUSY_REQUESTS + 1)  # noqa: SLF001
        monkeypatch.setattr(jobs, "_BUSY_WAIT_SECONDS", 0.01)
        indexing_queue.pending.append({"Body": _job_body(), "receives": 0})

        jobs.open_job_consumer()
        try:
            await sleep(0.05)
            assert indexing_queue.pending
            assert not indexing_queue.in_flight
        finally:
            jobs.close_job_consumer()

    async def test_the_queue_is_left_alone_when_no_queue_is_configured(self) -> None:
        """Nothing consumes, so nothing constructs a client either."""
        assert SETTINGS.aws_sqs_vector_store_queue_url is None

        jobs.open_job_consumer()
        try:
            assert jobs._CONSUMER is None  # noqa: SLF001
        finally:
            jobs.close_job_consumer()

    async def test_the_region_comes_from_the_queue_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second setting could only disagree with the URL, so there is none."""
        assert jobs.queue_region() is None
        monkeypatch.setattr(
            SETTINGS,
            "aws_sqs_vector_store_queue_url",
            "https://sqs.eu-west-3.amazonaws.com/123456789012/indexing",
        )
        assert jobs.queue_region() == "eu-west-3"


#: Seconds a live delivery stays invisible, so a killed job comes back quickly.
_LIVE_VISIBILITY_SECONDS = 2

#: Seconds to wait for AWS to hand a killed job back to another consumer.
_LIVE_REDELIVERY_TIMEOUT = 90.0


@pytest.fixture(scope="session")
def indexing_job_queue() -> Iterator[str]:
    """A real Amazon SQS queue, with the dead-letter queue the feature requires.

    Session-scoped, and deliberately so: a deleted queue name cannot be created
    again for 60 seconds, so one queue serves the whole run.

    Yields:
        The queue URL.
    """
    region = SETTINGS.aws_bedrock_regions[0]
    # Synchronous on purpose: the queue's lifecycle is not this loop's business.
    client: Any = botocore_session().create_client("sqs", region_name=region)
    suffix = webuuid()[:16]
    dead_letter = client.create_queue(QueueName=f"stdapi-test-index-dlq-{suffix}")[
        "QueueUrl"
    ]
    try:
        arn = client.get_queue_attributes(
            QueueUrl=dead_letter, AttributeNames=["QueueArn"]
        )["Attributes"]["QueueArn"]
        queue = client.create_queue(
            QueueName=f"stdapi-test-index-{suffix}",
            Attributes={
                "RedrivePolicy": to_json_str(
                    {"deadLetterTargetArn": arn, "maxReceiveCount": 3}
                ),
                "MessageRetentionPeriod": "300",
            },
        )["QueueUrl"]
        # A queue is not usable for up to a second after it is created.
        time.sleep(1.0)
        try:
            yield queue
        finally:
            client.delete_queue(QueueUrl=queue)
    finally:
        client.delete_queue(QueueUrl=dead_letter)


@pytest.fixture
async def live_indexing_queue(
    vector_backend: _FakeBackend,
    indexing_job_queue: str,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Any]:
    """Point the indexing jobs at the real queue, on this test's own client.

    Yields:
        The queue client, for the test to receive with itself.
    """
    del vector_backend
    async with AWS_SESSION.create_client(
        "sqs", region_name=SETTINGS.aws_bedrock_regions[0]
    ) as client:
        monkeypatch.setattr(
            SETTINGS, "aws_sqs_vector_store_queue_url", indexing_job_queue
        )
        monkeypatch.setattr(jobs, "_client", lambda: client)
        yield client


async def _receive_live(
    client: Any,  # noqa: ANN401
    queue_url: str,
    within_seconds: float,
) -> dict[str, Any]:
    """Long-poll the real queue until it hands one message over.

    Args:
        client: The queue client.
        queue_url: The queue to receive from.
        within_seconds: How long to keep waiting before failing the test.

    Returns:
        The delivered message.
    """
    deadline = time.monotonic() + within_seconds
    while True:
        received: list[dict[str, Any]] = (
            await client.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=5,
                VisibilityTimeout=_LIVE_VISIBILITY_SECONDS,
                MessageSystemAttributeNames=["ApproximateReceiveCount"],
            )
        ).get("Messages", [])
        if received:
            return received[0]
        assert time.monotonic() < deadline, "the queue delivered no message"


@pytest.mark.local
@pytest.mark.gateway("an indexing job is this gateway's own durability mechanism")
@pytest.mark.xdist_group("vector_store_indexing_queue")
class TestAKilledJobIsRecoveredLive:
    """The durability claim, against the queue that has to make it true.

    What no stand-in can answer: whether Amazon SQS really hands a message back
    to another consumer when the one holding it dies without deleting it, what
    it reports as the delivery count when it does, and whether the redrive
    policy this feature depends on is read off a real queue the way the startup
    probe expects.

    The store behind it is the in-memory one on purpose: the recovery is what
    is under test here, and the records themselves are exercised against real
    S3 by ``TestDetachIsDurableLive``.

    Ref: stdapi/vector_stores/jobs.py:_run_job
         https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-visibility-timeout.html
    """

    async def test_a_second_consumer_finishes_what_a_killed_one_started(
        self,
        vector_backend: _FakeBackend,
        indexing_job_queue: str,
        live_indexing_queue: Any,  # noqa: ANN401
    ) -> None:
        """A file whose indexing server died is completed by another, not failed.

        This is the whole point of the feature: before it, the caller had to
        attach the file again.
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        started, release = TestAKilledJobIsFinishedElsewhere._blocked_embedding(  # noqa: SLF001
            vector_backend
        )

        first = await _receive_live(
            live_indexing_queue, indexing_job_queue, _LIVE_REDELIVERY_TIMEOUT
        )
        assert first["Attributes"]["ApproximateReceiveCount"] == "1"
        killed = create_task(jobs._run_job(first))  # noqa: SLF001
        await started.wait()
        killed.cancel()
        with pytest.raises(CancelledError):
            await killed
        assert (await read_file(store.id, file_id)).status == "in_progress"

        release.set()
        second = await _receive_live(
            live_indexing_queue, indexing_job_queue, _LIVE_REDELIVERY_TIMEOUT
        )
        assert second["MessageId"] == first["MessageId"]
        assert second["Attributes"]["ApproximateReceiveCount"] == "2"
        await jobs._run_job(second)  # noqa: SLF001

        record = await read_file(store.id, file_id)
        assert record.status == "completed"
        assert record.chunk_count
        assert (await read_store(store.id)).file_counts.completed == 1

    async def test_the_startup_probe_reads_the_real_redrive_policy(
        self,
        live_indexing_queue: Any,  # noqa: ANN401
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """How many deliveries a job gets comes from the queue, not from a constant.

        Ref: stdapi/vector_stores/jobs.py:initialize_job_queue
        """
        del live_indexing_queue
        monkeypatch.setattr(jobs, "_MAX_RECEIVES", 0)
        monkeypatch.setattr(jobs, "_HAS_DEAD_LETTER_QUEUE", False)
        start_event: Any = {"level": "info"}

        await jobs.initialize_job_queue(start_event)

        assert jobs._MAX_RECEIVES == 3  # noqa: SLF001
        assert jobs._HAS_DEAD_LETTER_QUEUE  # noqa: SLF001
        assert "server_warnings" not in start_event
