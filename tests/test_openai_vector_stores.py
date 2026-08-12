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
     stdapi/vector_stores.py
"""

import time
from asyncio import run
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal
from zlib import crc32

import pytest
from botocore.exceptions import ClientError
from fastapi.exceptions import RequestValidationError
from openai import NotFoundError, OpenAI
from pydantic import ValidationError

from stdapi import vector_stores
from stdapi.api_errors import ApiError
from stdapi.cleanup import CLEANUPS
from stdapi.config import SETTINGS
from stdapi.models.embedding import EmbeddingResponse
from stdapi.models.embedding.cohere_embed import EmbeddingModel as CohereEmbeddingModel
from stdapi.types.openai_vector_stores import (
    ComparisonFilter,
    CompoundFilter,
    FileBatchCreateParams,
    FileBatchFile,
    StaticChunkingConfig,
)
from stdapi.vector_stores import (
    FileCountsRecord,
    PendingFile,
    StoreRecord,
    _file_key,
    _index_files,
    _store_key,
    attach_files,
    attribute_key,
    cancel_batch,
    chunk_text,
    create_store,
    delete_store,
    detach_file,
    index_name,
    new_batch_id,
    new_store_id,
    read_batch,
    read_file,
    read_file_chunks,
    read_store,
    score_from_distance,
    search,
    touch_store,
    translate_filter,
    update_file_attributes,
    update_record,
    vector_key,
)
from tests._helpers import make_client_error, red_png
from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

    from botocore.exceptions import ClientError
    from openai.types import VectorStore
    from openai.types.vector_stores import VectorStoreFile
    from starlette.testclient import TestClient

#: The vector store namespace is account-wide, exactly like the Files API's:
#: without a group, ``--dist=loadgroup`` spreads this module across workers and
#: a listing assertion sees a sibling's half-deleted store.
pytestmark = pytest.mark.xdist_group("openai_vector_stores")

#: A planted sentence no other test content answers, for the search assertions.
_PLANTED = "The maintenance hatch of the Kelvin observatory opens with code QUINCEY-7."

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

#: Seconds a test waits for a listing to catch up with a just-written object.
#: Generous on purpose: a vector store listing is eventually consistent, and
#: lags noticeably behind a direct read of the same object.
_LIST_TIMEOUT = 120.0


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


#: The ``object`` a file batch carries. Typed as a plain string on purpose: the
#: installed SDK's Literal says ``vector_store.files_batch`` while the SDK's own
#: docstring says ``vector_store.file_batch`` — and the vendor lane
#: (``--use-official-api``) answers ``vector_store.file_batch``, so the SDK
#: Literal is what is wrong.
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
            # The attachment is not visible yet: the listing behind it is
            # eventually consistent on both targets.
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

    Ref: stdapi/vector_stores.py:chunk_text
    """

    def test_short_text_is_one_chunk(self) -> None:
        """Text below the budget is not split.

        Ref: stdapi/vector_stores.py:chunk_text
        """
        assert chunk_text("hello world", 100, 20, 0) == ["hello world"]

    def test_long_text_is_split_and_overlaps(self) -> None:
        """A long text yields several chunks whose ends overlap.

        Ref: stdapi/vector_stores.py:chunk_text
        """
        text = " ".join(f"word{i:04d}" for i in range(400))
        chunks = chunk_text(text, 100, 50, 0)
        assert len(chunks) > 1
        # 100 tokens is approximated as 400 characters.
        assert all(len(chunk) <= 400 for chunk in chunks)
        assert chunks[1].split()[0] in chunks[0]

    def test_cut_falls_on_a_word_boundary(self) -> None:
        """A chunk does not end mid-word when a separator is within reach.

        Ref: stdapi/vector_stores.py:chunk_text
        """
        text = " ".join("alpha" for _ in range(300))
        for chunk in chunk_text(text, 100, 0, 0):
            assert set(chunk.split()) == {"alpha"}

    def test_chunk_is_clamped_to_the_model_character_limit(self) -> None:
        """A model that rejects long inputs caps the chunk regardless of the token budget.

        Ref: stdapi/models/embedding/cohere_embed.py:EmbeddingModel.max_input_characters
        """
        text = "x" * 20000
        assert all(len(chunk) <= 2048 for chunk in chunk_text(text, 4096, 0, 2048))

    def test_whitespace_only_text_yields_no_chunk(self) -> None:
        """A file holding only whitespace produces nothing to index.

        Ref: stdapi/vector_stores.py:chunk_text
        """
        assert chunk_text("   \n\n\t  ", 100, 20, 0) == []

    def test_every_chunk_fits_the_per_vector_text_budget(self) -> None:
        """A chunk of multi-byte characters is split to fit the stored-text budget.

        Ref: stdapi/vector_stores.py:_split_on_bytes
        """
        chunks = chunk_text("é" * 40000, 4096, 0, 0)
        assert chunks
        assert all(len(chunk.encode()) <= 32768 for chunk in chunks)


@pytest.mark.local
class TestFilterTranslation:
    """Every upstream filter operator has an index equivalent.

    Ref: openai.types.shared_params.comparison_filter.ComparisonFilter
         stdapi/vector_stores.py:translate_filter
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

        Ref: stdapi/vector_stores.py:translate_filter
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

        Ref: stdapi/vector_stores.py:translate_filter
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

        Ref: stdapi/vector_stores.py:translate_filter
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

        Ref: stdapi/vector_stores.py:translate_filter
        """
        with pytest.raises(RequestValidationError):
            translate_filter(CompoundFilter(type="and", filters=[{"not": "a filter"}]))

    def test_attribute_keys_cannot_collide_with_stored_content(self) -> None:
        """An attribute named like a reserved key is namespaced away from it.

        The chunk text and the source file name are stored on the same vector,
        so a caller attribute must not be able to overwrite them.

        Ref: stdapi/vector_stores.py:attribute_key
        """
        assert attribute_key("_text") != "_text"
        assert attribute_key("_filename") != "_filename"
        assert attribute_key("a") != attribute_key("b")


@pytest.mark.local
class TestScoreMapping:
    """Cosine distance becomes the similarity score the API reports.

    Ref: stdapi/vector_stores.py:score_from_distance
    """

    @pytest.mark.parametrize(
        ("distance", "score"),
        [(0.0, 1.0), (1.0, 0.0), (2.0, 0.0), (0.2928932309150696, 0.7071067690849304)],
    )
    def test_distance_to_score(self, distance: float, score: float) -> None:
        """An identical vector scores 1, an orthogonal one 0, an opposite one 0.

        The distances are the ones measured against the real index for unit
        vectors at 0, 90, 180 and 45 degrees.

        Ref: stdapi/vector_stores.py:score_from_distance
        """
        assert score_from_distance(distance) == pytest.approx(score)

    def test_score_never_leaves_the_unit_range(self) -> None:
        """A score is always reportable against ``ranking_options.score_threshold``.

        Ref: stdapi/vector_stores.py:score_from_distance
        """
        assert score_from_distance(-0.5) == 1.0
        assert score_from_distance(3.0) == 0.0


@pytest.mark.local
class TestIdentifiers:
    """Store identifiers map to index names reversibly and sort by creation time.

    Ref: stdapi/vector_stores.py:index_name
    """

    def test_index_name_is_derived_from_the_store_id(self) -> None:
        """The identifier's separator is the only character that changes.

        Ref: stdapi/vector_stores.py:index_name
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

        Ref: stdapi/vector_stores.py:new_store_id
        """
        first = new_store_id()
        time.sleep(0.005)
        assert new_store_id() > first
        first_batch = new_batch_id()
        time.sleep(0.005)
        assert new_batch_id() > first_batch

    def test_malformed_identifiers_are_rejected(self) -> None:
        """An identifier that is not one of ours never reaches a backend call.

        Ref: stdapi/vector_stores.py:parse_store_id
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

        Ref: stdapi/vector_stores.py:check_attributes
        """
        with pytest.raises(ApiError) as raised:
            vector_stores.check_attributes({f"k{i}": "v" * 512 for i in range(16)})
        assert raised.value.status == 400
        assert "2048" in str(raised.value)

    def test_attributes_within_the_budget_are_accepted(self) -> None:
        """A realistic attribute set passes untouched.

        Ref: stdapi/vector_stores.py:check_attributes
        """
        vector_stores.check_attributes({"topic": "physics", "year": 2026.0})


@pytest.mark.local
class TestConditionalUpdate:
    """Counter updates retry when another writer won the conditional write.

    Ref: stdapi/vector_stores.py:update_record
    """

    async def test_retries_until_the_write_lands(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A losing conditional write is retried against the re-read record.

        Ref: stdapi/vector_stores.py:_update
        """
        record = StoreRecord(
            id=new_store_id(),
            created_at=1,
            last_active_at=1,
            embedding_model="m",
            dimensions=8,
        )
        reads = 0

        async def fake_read(_model: object, _key: str) -> tuple[StoreRecord, str]:
            nonlocal reads
            reads += 1
            return record, f"etag-{reads}"

        writes = 0

        async def fake_write(_key: str, _record: object, *, etag: str | None) -> None:
            nonlocal writes
            del etag
            writes += 1
            if writes < 3:
                raise _precondition_failed()

        monkeypatch.setattr(vector_stores, "_read", fake_read)
        monkeypatch.setattr(vector_stores, "_write", fake_write)
        updated = await update_record(
            StoreRecord, "key", lambda r: setattr(r, "usage_bytes", r.usage_bytes + 1)
        )
        assert writes == 3
        assert reads == 3
        assert updated.usage_bytes == 3

    async def test_missing_record_never_names_internal_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A record deleted mid-update answers 404 without naming its object key.

        Ref: AGENTS.md "Never Leak Internals"
        """

        async def fake_read(_model: object, _key: str) -> None:
            return None

        monkeypatch.setattr(vector_stores, "_read", fake_read)
        with pytest.raises(ApiError) as raised:
            await update_record(
                StoreRecord, "vector_stores/vs_x/store.json", lambda _record: None
            )
        assert raised.value.status == 404
        message = str(raised.value)
        assert "vector_stores/" not in message
        assert ".json" not in message

    async def test_gives_up_with_a_retryable_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Endless contention answers 409 rather than looping forever.

        Ref: stdapi/vector_stores.py:_update
        """
        record = StoreRecord(
            id=new_store_id(),
            created_at=1,
            last_active_at=1,
            embedding_model="m",
            dimensions=8,
        )

        async def fake_read(_model: object, _key: str) -> tuple[StoreRecord, str]:
            return record, "etag"

        async def fake_write(_key: str, _record: object, *, etag: str | None) -> None:
            del etag
            raise _precondition_failed()

        monkeypatch.setattr(vector_stores, "_read", fake_read)
        monkeypatch.setattr(vector_stores, "_write", fake_write)
        with pytest.raises(ApiError) as raised:
            await update_record(StoreRecord, "key", lambda _record: None)
        assert raised.value.status == 409


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
        self._counter = 0

    def _scheduled_error(self, operation: str, key: str) -> None:
        """Raise the error scheduled for *operation* on *key*, if any."""
        error = self.fail_once.pop((operation, key), None)
        if error is not None:
            raise error

    async def get_object(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        key = params["Key"]
        self._scheduled_error("get_object", key)
        if key not in self.objects:
            missing = make_client_error("NoSuchKey", "GetObject")
            raise missing
        return {"Body": _FakeBody(self.objects[key]), "ETag": self.etags[key]}

    async def put_object(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        key = params["Key"]
        self._scheduled_error("put_object", key)
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


class _FakeVectorIndex:
    """In-memory stand-in for the S3 Vectors bucket backing the stores."""

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
        self.vectors = _FakeVectorIndex()
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

    Only the boundaries are replaced — the record bucket, the vector index, the
    embedding model and the Files API — so the engine, the routes and the
    conditional-update loop are the code actually under test.
    """
    backend = _FakeBackend()
    monkeypatch.setattr(SETTINGS, "aws_s3_bucket", "stdapi-test-records")
    monkeypatch.setattr(SETTINGS, "aws_s3_vectors_bucket", "stdapi-test-vectors")
    monkeypatch.setattr(vector_stores, "_records_client", lambda: backend.records)
    monkeypatch.setattr(vector_stores, "_vectors_client", lambda: backend.vectors)
    monkeypatch.setattr(
        vector_stores, "get_embedding_model", lambda _model_id: backend.model
    )

    async def _validate_model(model_id: str, _modality: str) -> SimpleNamespace:
        return SimpleNamespace(id=model_id)

    monkeypatch.setattr(vector_stores, "validate_model", _validate_model)
    monkeypatch.setattr(vector_stores, "get_file", backend.get_file)
    monkeypatch.setattr(vector_stores, "get_file_content", backend.get_file_content)
    # Indexing is driven by the tests themselves: a task racing the assertions
    # is the one thing this backend cannot make deterministic.
    monkeypatch.setattr(vector_stores, "_start_indexing", backend.start_indexing)
    return backend


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


def _error_of(response: Any) -> dict[str, Any]:  # noqa: ANN401
    """Return the error envelope of a failed response."""
    payload: dict[str, Any] = response.json()["error"]
    return payload


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
            # The listing is eventually consistent upstream: a store created a
            # moment ago is not necessarily in the next page.
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
            # The store has to settle with it: a failure that never leaves the
            # counters makes the store permanently `in_progress`.
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
         stdapi/vector_stores.py:_index_files
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


@pytest.mark.local
class TestUnconfiguredDeployment:
    """Without vector storage the routes answer 503, naming nothing internal.

    Ref: stdapi/vector_stores.py:records_bucket
    """

    def test_missing_vector_bucket_answers_503(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deployment with no vector bucket refuses cleanly.

        Ref: stdapi/vector_stores.py:records_bucket
        """
        monkeypatch.setattr(SETTINGS, "aws_s3_vectors_bucket", None, raising=False)
        with pytest.raises(ApiError) as raised:
            vector_stores.records_bucket()
        assert raised.value.status == 503
        message = str(raised.value)
        assert "administrator" in message
        assert "s3" not in message.lower()
        assert "bucket" not in message.lower()


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

        Ref: stdapi/vector_stores.py:_list_ids
        """
        ids = [
            app_client.post("/v1/vector_stores", json={"name": f"a{index}"}).json()[
                "id"
            ]
            for index in range(3)
        ]
        page = app_client.get(f"/v1/vector_stores?order=asc&after={ids[0]}").json()
        assert [entry["id"] for entry in page["data"]] == ids[1:]

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

        Ref: stdapi/vector_stores.py:check_attributes
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

        Ref: stdapi/vector_stores.py:translate_filter
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

        Ref: stdapi/vector_stores.py:records_bucket
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
    run(_index_files(store_id, [file_id], "", "test-request"))
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

    Ref: stdapi/vector_stores.py:_index_files
    """

    async def test_a_file_is_chunked_embedded_and_counted(
        self, vector_backend: _FakeBackend
    ) -> None:
        """Indexing embeds every chunk in waves and settles the store counters.

        Ref: stdapi/vector_stores.py:_index_one_file
        """
        store = await _create_store()
        # Long enough to cross the embedding wave, which a single-chunk file
        # never does: a wave-sized backend limit would break in production only.
        content = "\n".join(f"Paragraph {index} of the manual." for index in range(200))
        file_id = vector_backend.upload(content.encode())
        await _attach(store, [file_id])
        vector_backend.model.waves.clear()
        await _index_files(store.id, [file_id], "", "test-request")

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

        Ref: stdapi/vector_stores.py:_index_files
        """
        store = await _create_store()
        files = [vector_backend.upload(_TEXT_FILE) for _ in range(3)]
        await _attach(store, files)
        vector_backend.records.fail_once[
            ("get_object", _file_key(store.id, files[1]))
        ] = make_client_error("SlowDown", "GetObject", status=503)
        await _index_files(store.id, files, "", "test-request")

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
        await _index_files(store.id, [file_id], "", "test-request")

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
        await _index_files(store.id, files[:1], batch_id, "test-request")
        await cancel_batch(store.id, batch_id)
        await _index_files(store.id, files[1:], batch_id, "test-request")

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

        Ref: stdapi/vector_stores.py:_index_one_file
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        await detach_file(store.id, file_id)
        await _run_cleanups(scheduled_cleanups)
        await _index_files(store.id, [file_id], "", "test-request")

        settled = await read_store(store.id)
        assert settled.file_counts.total == 0
        assert settled.file_counts.cancelled == 0
        assert settled.file_counts.in_progress == 0

    async def test_the_same_file_twice_in_one_request_is_attached_once(
        self, vector_backend: _FakeBackend
    ) -> None:
        """A repeated file identifier is counted once, so the totals can converge.

        Ref: stdapi/vector_stores.py:attach_files
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        batch_id = new_batch_id()
        attached = await _attach(store, [file_id, file_id], batch_id=batch_id)
        assert attached == [file_id]
        await _index_files(store.id, attached, batch_id, "test-request")
        settled = await read_store(store.id)
        assert settled.file_counts.total == 1
        assert (await read_batch(store.id, batch_id)).file_counts.total == 1

    async def test_reindexing_reclaims_the_chunks_of_the_previous_version(
        self, vector_backend: _FakeBackend
    ) -> None:
        """A file re-attached with larger chunks leaves no stale passage searchable.

        Ref: stdapi/vector_stores.py:_index_one_file
        """
        store = await _create_store()
        content = "\n".join(f"Paragraph {index} of the manual." for index in range(60))
        file_id = vector_backend.upload(content.encode())
        await _attach(store, [file_id], size=100)
        await _index_files(store.id, [file_id], "", "test-request")
        first = (await read_file(store.id, file_id)).chunk_count
        assert first > 1

        await _attach(store, [file_id], size=4096)
        await _index_files(store.id, [file_id], "", "test-request")
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

        Ref: stdapi/vector_stores.py:detach_file
        """
        store = await _create_store()
        content = "\n".join(f"Paragraph {index} of the manual." for index in range(60))
        file_id = vector_backend.upload(content.encode())
        await _attach(store, [file_id])
        await _index_files(store.id, [file_id], "", "test-request")
        assert vector_backend.vectors.indexes[index_name(store.id)]

        vector_backend.uploads[file_id[5:]] = (red_png(), "image/png")
        await _attach(store, [file_id])
        await _index_files(store.id, [file_id], "", "test-request")
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
        await _index_files(store.id, [records[0].id], "", "test-request")
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

        Ref: stdapi/vector_stores.py:search
        """
        store = await _create_store()
        file_id = vector_backend.upload(_TEXT_FILE)
        await _attach(store, [file_id])
        await _index_files(store.id, [file_id], "", "test-request")
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
        await _index_files(store.id, [file_id], "", "test-request")
        await update_record(
            StoreRecord,
            _store_key(store.id),
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

        Ref: stdapi/vector_stores.py:touch_store
        """
        store = await _create_store(expires_after_days=1)
        await update_record(
            StoreRecord,
            _store_key(store.id),
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

        Ref: stdapi/vector_stores.py:_all_record_keys
        """
        store = await _create_store()
        files = [vector_backend.upload(_OTHER_FILE) for _ in range(3)]
        await _attach(store, files, batch_id=new_batch_id())
        # One record per listing page, so the deletion has to paginate.
        monkey = pytest.MonkeyPatch()
        monkey.setattr(vector_stores, "_LIST_SCAN_MAX", 1)
        try:
            await delete_store(store.id)
            await _run_cleanups(scheduled_cleanups)
        finally:
            monkey.undo()
        assert vector_backend.records.objects == {}
        assert vector_backend.vectors.indexes == {}


@pytest.mark.local
class TestFileSizeLimit:
    """``max_input_file_size`` bounds indexing, and zero still means unlimited.

    The zero-means-unlimited sense broke every indexing run once: read as a
    literal ceiling, a default deployment failed every file it was given.

    Ref: stdapi/vector_stores.py:_load_chunks
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
        await _index_files(store.id, [file_id], "", "test-request")
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
        await _index_files(store.id, [file_id], "", "test-request")
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

        Ref: stdapi/vector_stores.py:_load_chunks
        """
        vector_backend.model.max_input_characters = 200
        store = await _create_store(max_chunk_size_tokens=4096)
        file_id = vector_backend.upload(b"word " * 2000)
        await _attach(store, [file_id], size=4096)
        await _index_files(store.id, [file_id], "", "test-request")
        record = await read_file(store.id, file_id)
        chunks = await read_file_chunks(store.id, record)
        assert chunks
        assert all(len(chunk) <= 200 for chunk in chunks)
