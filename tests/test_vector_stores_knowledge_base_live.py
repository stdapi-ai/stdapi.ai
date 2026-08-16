"""Live tests for a vector store served by a customer's Bedrock knowledge base.

The offline module simulates the knowledge base; this one addresses two real
ones — a customer-managed generation (``type = VECTOR``, backed by S3 Vectors)
and a Bedrock managed one (``type = MANAGED``) — through the same
``/v1/vector_stores`` routes, and exercises what only the service can answer:
the store the knowledge base describes itself as, the document round trip
(attach, index, read, search, delete), the data source resolution, and the
refusals that depend on the generation the service reports rather than on a
stand-in that was told which one it is.

The knowledge bases are the customer's and are never created or deleted here,
exactly as the server treats them. Only documents are written, and every one is
removed again, including when the test that attached it failed.

Which knowledge bases to address is configuration: ``TEST_KNOWLEDGE_BASE_ID``
and ``TEST_MANAGED_KNOWLEDGE_BASE_ID`` name them, and
``aws_bedrock_knowledge_base_ids`` allowlists them. A checkout without them
skips rather than fails.

Ref: https://platform.openai.com/docs/api-reference/vector-stores
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Retrieve.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/kb-direct-ingestion.html
     https://stdapi.ai/api_openai_vector_stores/#knowledge-base-stores
     stdapi/vector_stores/knowledge_base.py
"""

from __future__ import annotations

import time
from base64 import urlsafe_b64decode
from contextlib import suppress
from os import environ
from typing import TYPE_CHECKING, NamedTuple

import pytest
from openai import APIStatusError, BadRequestError, NotFoundError

from stdapi.config import SETTINGS
from stdapi.pricing import KNOWLEDGE_BASE_MODEL
from stdapi.vector_stores.knowledge_base import MANAGED_MEDIA_TYPES, STORE_ID_PREFIX
from tests._helpers import red_png
from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai import OpenAI
    from openai.types.vector_stores import VectorStoreFile

#: The knowledge bases are the account's, and the documents attached to them are
#: listed, searched and deleted across tests: one worker, or they race.
#: A knowledge base is an AWS resource with no upstream counterpart, so the whole
#: module is meaningless against the official API and valid against a gateway.
pytestmark = [pytest.mark.gateway, pytest.mark.xdist_group("knowledge_base_live")]

#: Environment variable naming the customer-managed knowledge base to address.
_VECTOR_VARIABLE = "TEST_KNOWLEDGE_BASE_ID"

#: Environment variable naming the Bedrock managed knowledge base to address.
_MANAGED_VARIABLE = "TEST_MANAGED_KNOWLEDGE_BASE_ID"

#: Environment variable naming a knowledge base, per generation.
_VARIABLES = {"VECTOR": _VECTOR_VARIABLE, "MANAGED": _MANAGED_VARIABLE}

#: Characters a search query may hold, per generation, as the backend declares.
_QUERY_LIMITS = {"VECTOR": 1000, "MANAGED": 10000}

#: A sentence no other corpus of these knowledge bases answers.
_PLANTED = "The stdapi live probe passphrase is marmalade-lighthouse."

#: The attached document, built around the planted sentence.
_DOCUMENT: bytes = (
    "stdapi.ai live probe document.\n\n"
    "This file is attached by the test suite and deleted again.\n\n"
    f"{_PLANTED}\n"
).encode()

#: Name the attached document is uploaded, and reported back, under.
_FILENAME = "stdapi-live-probe.txt"

#: Seconds a test waits for a knowledge base to finish ingesting a document.
_INGEST_TIMEOUT = 300.0

#: Seconds a test waits for a deleted document to stop being reported.
_DELETE_TIMEOUT = 180.0

#: Seconds between two polls of an asynchronous document operation.
_POLL_INTERVAL = 3.0

#: A syntactically valid knowledge base identifier no deployment allowlists.
_UNKNOWN_STORE_ID = f"{STORE_ID_PREFIX}ZZZZZZZZZZ"


class Store(NamedTuple):
    """One knowledge base under test.

    Attributes:
        generation: ``"VECTOR"`` or ``"MANAGED"``.
        id: The vector store identifier it is addressed as.
    """

    generation: str
    id: str


def _served(client: OpenAI, generation: str) -> Store:
    """Return the store of *generation*, or skip when the target serves none.

    Args:
        client: OpenAI SDK client bound to the target under test.
        generation: The knowledge base generation to address.

    Returns:
        The store under test.
    """
    variable = _VARIABLES[generation]
    knowledge_base_id = environ.get(variable, "")
    if not knowledge_base_id:
        pytest.skip(
            f"No {generation} knowledge base to address: set {variable} to one, "
            "and allowlist it in aws_bedrock_knowledge_base_ids"
        )
    store_id = f"{STORE_ID_PREFIX}{knowledge_base_id}"
    try:
        client.vector_stores.retrieve(store_id)
    except (APIStatusError, NotFoundError) as error:
        if error.status_code in {404, 503}:
            pytest.skip(
                f"The target does not serve {store_id}: allowlist the knowledge "
                "base in aws_bedrock_knowledge_base_ids"
            )
        raise
    return Store(generation, store_id)


@pytest.fixture(scope="session", params=["VECTOR", "MANAGED"])
def store(request: pytest.FixtureRequest, openai_client: OpenAI) -> Store:
    """Each knowledge base generation in turn, for what both must answer.

    Returns:
        The store under test.
    """
    generation: str = request.param
    return _served(openai_client, generation)


@pytest.fixture(scope="session")
def vector_store(openai_client: OpenAI) -> Store:
    """The customer-managed knowledge base, for what only it can answer.

    Returns:
        The store under test.
    """
    return _served(openai_client, "VECTOR")


@pytest.fixture(scope="session")
def managed_store(openai_client: OpenAI) -> Store:
    """The Bedrock managed knowledge base, for what only it can answer.

    Returns:
        The store under test.
    """
    return _served(openai_client, "MANAGED")


def _upload(client: OpenAI, name: str, content: bytes, media_type: str) -> str:
    """Upload a file for ingestion and return its identifier.

    Args:
        client: OpenAI SDK client bound to the target under test.
        name: File name to store it under.
        content: The file bytes.
        media_type: The media type it is uploaded with.

    Returns:
        The uploaded file identifier.
    """
    return client.files.create(
        file=(name, content, media_type), purpose="assistants"
    ).id


def _wait_for_settled(client: OpenAI, store_id: str, file_id: str) -> VectorStoreFile:
    """Poll one document until the knowledge base stops indexing it.

    Ingestion is asynchronous: the attach answers ``in_progress`` and the
    document settles later, either way.

    Args:
        client: OpenAI SDK client bound to the target under test.
        store_id: The store the document was attached to.
        file_id: The attached document.

    Returns:
        The document once it has settled.
    """
    deadline = time.monotonic() + _INGEST_TIMEOUT
    status = "unknown"
    while time.monotonic() < deadline:
        try:
            attached = client.vector_stores.files.retrieve(
                file_id, vector_store_id=store_id
            )
        except NotFoundError:
            # Not reported yet: the document listing behind it settles later.
            time.sleep(_POLL_INTERVAL)
            continue
        status = attached.status
        if status != "in_progress":
            return attached
        time.sleep(_POLL_INTERVAL)
    pytest.fail(
        f"{file_id} was still {status} in {store_id} after {_INGEST_TIMEOUT}s "
        "of ingestion"
    )


def _wait_for_indexed(client: OpenAI, store_id: str, file_id: str) -> None:
    """Poll one document until the knowledge base reports it indexed.

    Args:
        client: OpenAI SDK client bound to the target under test.
        store_id: The store the document was attached to.
        file_id: The attached document.
    """
    settled = _wait_for_settled(client, store_id, file_id)
    assert settled.status == "completed", (
        f"{file_id} was not indexed: {settled.status} "
        f"({settled.last_error and settled.last_error.message})"
    )


def _synced_data_source(knowledge_base_id: str) -> str:
    """Return a data source of the knowledge base that syncs its own corpus.

    Which data sources a knowledge base holds, and what kind each one is, is
    the service's answer and nothing this suite can configure. Read with the
    account's own client rather than the server's: the server's are bound to
    the event loop serving the requests under test.

    Args:
        knowledge_base_id: The knowledge base the store addresses.

    Returns:
        The data source identifier, or ``""`` when every one of them is custom.
    """
    import boto3  # type: ignore[import-untyped]  # noqa: PLC0415

    client = boto3.client("bedrock-agent", region_name=SETTINGS.aws_bedrock_regions[0])
    listed = client.list_data_sources(knowledgeBaseId=knowledge_base_id, maxResults=10)
    for summary in listed.get("dataSourceSummaries", ()):
        described = client.get_data_source(
            knowledgeBaseId=knowledge_base_id, dataSourceId=summary["dataSourceId"]
        )
        if described["dataSource"]["dataSourceConfiguration"]["type"] == "S3":
            return str(summary["dataSourceId"])
    return ""


def _detach(client: OpenAI, store_id: str, file_id: str) -> None:
    """Remove one document, ignoring a document the store no longer holds.

    Args:
        client: OpenAI SDK client bound to the target under test.
        store_id: The store holding it.
        file_id: The document to remove.
    """
    with suppress(NotFoundError):
        client.vector_stores.files.delete(file_id, vector_store_id=store_id)


@pytest.fixture(scope="session")
def indexed_document(openai_client: OpenAI, store: Store) -> Iterator[str]:
    """One indexed document per generation, shared by the assertions reading it.

    Session-scoped: ingesting a document costs a real embedding of every passage
    and minutes of waiting, so one attach serves the listing, the read and the
    search. It is removed at the end, whether or not the tests passed.

    Yields:
        The identifier of the attached document.
    """
    file_id = _upload(openai_client, _FILENAME, _DOCUMENT, "text/plain")
    try:
        openai_client.vector_stores.files.create(
            vector_store_id=store.id,
            file_id=file_id,
            attributes={"probe": "live", "revision": 1},
        )
        _wait_for_indexed(openai_client, store.id, file_id)
        yield file_id
    finally:
        _detach(openai_client, store.id, file_id)
        openai_client.files.delete(file_id)


class TestStore:
    """What a knowledge base answers as a vector store, on both generations.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_GetKnowledgeBase.html
         stdapi/vector_stores/knowledge_base.py:KnowledgeBaseIndex.read_store
    """

    def test_the_store_reports_the_knowledge_base_it_addresses(
        self, store: Store, openai_client: OpenAI
    ) -> None:
        """Reading the store answers with what the knowledge base describes itself as.

        The name and the timestamps are the service's, and the counters stay at
        zero: the corpus is the customer's and is never scanned to count it.

        Ref: stdapi/vector_stores/knowledge_base.py:_store_record
        """
        answered = openai_client.vector_stores.retrieve(store.id)

        assert answered.id == store.id
        assert answered.object == "vector_store"
        assert answered.status == "completed"
        assert answered.name, "The knowledge base reported no name"
        assert answered.created_at > 0
        assert answered.file_counts.total == 0
        assert answered.usage_bytes == 0
        assert answered.expires_at is None

    def test_the_listing_serves_every_allowlisted_knowledge_base(
        self, vector_store: Store, managed_store: Store, openai_client: OpenAI
    ) -> None:
        """Both generations are listed next to the stores the server owns.

        Ref: stdapi/vector_stores/knowledge_base.py:KnowledgeBaseIndex.list_stores
        """
        listed = {
            entry.id for entry in openai_client.vector_stores.list(limit=100).data
        }

        assert {vector_store.id, managed_store.id} <= listed

    def test_the_documents_of_the_store_can_be_listed(
        self, store: Store, openai_client: OpenAI
    ) -> None:
        """Both generations answer a listing of the documents they hold.

        The managed generation refuses a page of more than a hundred documents,
        which the other one accepts, so a listing that reads the whole data
        source in one call answers on one generation and 400s on the other.

        The service reports one timestamp per document — ``updatedAt``, the
        instant it last ingested it — and no creation time, so that is what
        ``created_at`` answers with and what the page is ordered on.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_ListKnowledgeBaseDocuments.html
             stdapi/vector_stores/knowledge_base.py:KnowledgeBaseIndex.list_documents
        """
        listed = openai_client.vector_stores.files.list(store.id, limit=10)

        # Answering at all is the assertion: a page the generation refuses is a
        # 400 before a single document is reported.
        assert all(entry.object == "vector_store.file" for entry in listed.data)
        assert all(entry.vector_store_id == store.id for entry in listed.data)
        created = [entry.created_at for entry in listed.data]
        assert all(value > 0 for value in created), created
        assert created == sorted(created, reverse=True), created

    def test_the_knowledge_base_cannot_be_deleted_through_the_api(
        self, store: Store, openai_client: OpenAI
    ) -> None:
        """Deleting the store is refused: the knowledge base is the customer's.

        Ref: stdapi/vector_stores/knowledge_base.py:KnowledgeBaseIndex.delete_index
        """
        with pytest.raises(BadRequestError) as raised:
            openai_client.vector_stores.delete(store.id)

        assert raised.value.status_code == 400
        assert "managed outside this server" in str(raised.value)

        # The knowledge base is still there, and still addressable.
        assert openai_client.vector_stores.retrieve(store.id).id == store.id

    def test_the_knowledge_base_cannot_be_renamed_through_the_api(
        self, store: Store, openai_client: OpenAI
    ) -> None:
        """Renaming the store is refused: the name is the knowledge base's own.

        Ref: stdapi/routes/openai_vector_stores.py:update_vector_store
        """
        with pytest.raises(BadRequestError) as raised:
            openai_client.vector_stores.update(store.id, name="stdapi-live-renamed")

        assert raised.value.status_code == 400
        assert "managed outside this server" in str(raised.value)


@pytest.mark.slow
class TestDocuments:
    """The document round trip against the knowledge base that can ingest one.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/kb-direct-ingestion.html
         stdapi/vector_stores/knowledge_base.py:KnowledgeBaseIndex.attach_documents
    """

    def test_an_attached_file_reaches_an_indexed_state(
        self, indexed_document: str, store: Store, openai_client: OpenAI
    ) -> None:
        """The document the fixture attached is indexed, under the file's own id.

        A document this server ingested keeps the identifier of the file it came
        from, so the caller addresses it with the identifier it already has.

        Ref: stdapi/vector_stores/knowledge_base.py:document_file_id
        """
        attached = openai_client.vector_stores.files.retrieve(
            indexed_document, vector_store_id=store.id
        )

        assert attached.id == indexed_document
        assert attached.vector_store_id == store.id
        assert attached.status == "completed"
        assert attached.last_error is None
        # The knowledge base cuts its own passages, so it reports no strategy.
        assert attached.chunking_strategy is None

    def test_the_attached_document_is_listed_among_the_store_files(
        self, indexed_document: str, store: Store, openai_client: OpenAI
    ) -> None:
        """The listing reads the data source's documents back.

        Ref: stdapi/vector_stores/knowledge_base.py:KnowledgeBaseIndex.list_documents
        """
        listed = openai_client.vector_stores.files.list(store.id, limit=100).data

        found = [entry for entry in listed if entry.id == indexed_document]
        assert found, f"{indexed_document} is not among {[e.id for e in listed]}"
        assert found[0].status == "completed"

    def test_a_status_filter_selects_among_the_documents(
        self, indexed_document: str, store: Store, openai_client: OpenAI
    ) -> None:
        """Listing the indexed documents holds the one just indexed.

        Ref: stdapi/vector_stores/knowledge_base.py:KnowledgeBaseIndex.list_documents
        """
        completed = openai_client.vector_stores.files.list(
            store.id, limit=100, filter="completed"
        ).data
        in_progress = openai_client.vector_stores.files.list(
            store.id, limit=100, filter="in_progress"
        ).data

        assert indexed_document in {entry.id for entry in completed}
        assert indexed_document not in {entry.id for entry in in_progress}

    def test_a_search_returns_a_passage_of_the_document_just_attached(
        self, indexed_document: str, store: Store, openai_client: OpenAI
    ) -> None:
        """Searching retrieves the planted passage, under the file it came from.

        The score is asserted to exist and not to be rescaled into a similarity:
        the service states no range for it.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Retrieve.html
             stdapi/vector_stores/knowledge_base.py:_to_match
        """
        found = openai_client.vector_stores.search(
            vector_store_id=store.id,
            query="What is the stdapi live probe passphrase?",
            max_num_results=20,
        )

        ours = [entry for entry in found.data if entry.file_id == indexed_document]
        assert ours, (
            "the attached document is not among the retrieved passages: "
            f"{[entry.file_id for entry in found.data]}"
        )
        passage = ours[0]
        assert passage.filename == _FILENAME
        assert isinstance(passage.score, float)
        assert any(
            "marmalade-lighthouse" in part.text
            for part in passage.content
            if part.type == "text"
        ), "no retrieved passage carries the planted sentence"

    def test_the_caller_attributes_survive_the_round_trip(
        self, indexed_document: str, store: Store, openai_client: OpenAI
    ) -> None:
        """The attributes attached with the file come back on a retrieved passage.

        The file name the server keeps is reported as ``filename`` rather than as
        an attribute, so it never collides with the caller's own keys.

        Ref: stdapi/vector_stores/knowledge_base.py:_caller_attributes
        """
        found = openai_client.vector_stores.search(
            vector_store_id=store.id, query=_PLANTED, max_num_results=20
        )

        ours = [entry for entry in found.data if entry.file_id == indexed_document]
        assert ours, "the attached document was not retrieved"
        attributes = ours[0].attributes or {}
        assert attributes.get("probe") == "live"
        assert attributes.get("revision") == 1
        assert "_filename" not in attributes

    def test_a_filter_over_the_attributes_reaches_the_retrieval(
        self, indexed_document: str, store: Store, openai_client: OpenAI
    ) -> None:
        """A filter on an attribute the document does not carry excludes it.

        Ref: stdapi/vector_stores/knowledge_base.py:translate_filter
        """
        excluded = openai_client.vector_stores.search(
            vector_store_id=store.id,
            query=_PLANTED,
            max_num_results=5,
            filters={"type": "eq", "key": "probe", "value": "absent"},
        )

        assert indexed_document not in {entry.file_id for entry in excluded.data}

    def test_deleting_a_file_removes_the_document(
        self, store: Store, openai_client: OpenAI
    ) -> None:
        """A deleted document stops being held, and is then unknown.

        Its own attach rather than the shared one: what is asserted here is the
        removal, which the shared document must survive.

        Ref: stdapi/vector_stores/knowledge_base.py:KnowledgeBaseIndex.delete_document
        """
        file_id = _upload(
            openai_client, "stdapi-live-delete.txt", _DOCUMENT, "text/plain"
        )
        try:
            openai_client.vector_stores.files.create(
                vector_store_id=store.id, file_id=file_id
            )
            _wait_for_indexed(openai_client, store.id, file_id)

            deleted = openai_client.vector_stores.files.delete(
                file_id, vector_store_id=store.id
            )
            assert deleted.deleted is True

            deadline = time.monotonic() + _DELETE_TIMEOUT
            while True:
                try:
                    openai_client.vector_stores.files.retrieve(
                        file_id, vector_store_id=store.id
                    )
                except NotFoundError:
                    break
                assert time.monotonic() < deadline, (
                    f"{file_id} is still held by {store.id} "
                    f"{_DELETE_TIMEOUT}s after it was deleted"
                )
                time.sleep(_POLL_INTERVAL)

            # The service keeps a tombstone of a deleted document, which is not
            # a file of the store and must not be listed as one.
            listed = openai_client.vector_stores.files.list(store.id, limit=100).data
            assert file_id not in {entry.id for entry in listed}
        finally:
            _detach(openai_client, store.id, file_id)
            openai_client.files.delete(file_id)


class TestCustomerOwnedDocuments:
    """Documents of the customer's own data source, which this server did not attach.

    Ref: stdapi/vector_stores/knowledge_base.py:document_file_id
         stdapi/vector_stores/knowledge_base.py:document_target
    """

    def test_a_passage_of_the_customers_own_corpus_carries_a_decodable_identifier(
        self, store: Store, openai_client: OpenAI
    ) -> None:
        """A retrieved document the server did not attach is named by its location.

        A knowledge base holds more than what this API put there — the corpus
        its own data source syncs is located by a URI and carries no file
        identifier of ours, so it is reported under an opaque one that decodes
        back into that location. The two generations report that location
        differently, which is why the encoded location rather than its shape is
        what this asserts.

        Ref: stdapi/vector_stores/knowledge_base.py:_encode_document_id
        """
        found = openai_client.vector_stores.search(
            vector_store_id=store.id,
            query="What is in the sandbox knowledge base corpus?",
            max_num_results=20,
        )

        foreign = [entry for entry in found.data if entry.file_id.startswith("kbdoc_")]
        if not foreign:
            pytest.skip(f"{store.id} retrieved no document of the customer's own")

        passage = foreign[0]
        encoded = passage.file_id.removeprefix("kbdoc_")
        decoded = urlsafe_b64decode(f"{encoded}{'=' * (-len(encoded) % 4)}").decode()
        assert "://" in decoded, f"{decoded} is not a document location"
        # The name reported is the document's own, taken from the location the
        # service reports it at, under whichever key that generation uses.
        assert passage.filename == decoded.rsplit("/", 1)[-1]
        # None of the service's own metadata is answered as the caller's.
        assert not [key for key in passage.attributes or {} if key.startswith("_")]

    def test_a_retrieved_passage_reads_back_as_the_file_it_came_from(
        self, store: Store, openai_client: OpenAI
    ) -> None:
        """Searching and reading round trip over the whole corpus of the store.

        The corpus a knowledge base syncs is not where files are attached, and
        the two generations report a synced document's location differently and
        inconsistently: only a live retrieval can show that the identifier a
        search answers with is one the per-file route resolves.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_GetKnowledgeBaseDocuments.html
             stdapi/vector_stores/knowledge_base.py:KnowledgeBaseIndex.read_document
        """
        found = openai_client.vector_stores.search(
            vector_store_id=store.id,
            query="What is in the sandbox knowledge base corpus?",
            max_num_results=20,
        )

        foreign = [entry for entry in found.data if entry.file_id.startswith("kbdoc_")]
        if not foreign:
            pytest.skip(f"{store.id} retrieved no document of the customer's own")

        read = openai_client.vector_stores.files.retrieve(
            foreign[0].file_id, vector_store_id=store.id
        )

        assert read.id == foreign[0].file_id
        assert read.vector_store_id == store.id
        assert read.status == "completed"

    def test_a_document_of_the_corpus_cannot_be_removed_through_the_api(
        self, store: Store, openai_client: OpenAI
    ) -> None:
        """A document the corpus put there is readable, and never deletable.

        Removing it would take it out of a corpus this server neither filled nor
        maintains, so the store refuses rather than obeying.

        Ref: stdapi/vector_stores/knowledge_base.py:KnowledgeBaseIndex.delete_document
        """
        found = openai_client.vector_stores.search(
            vector_store_id=store.id,
            query="What is in the sandbox knowledge base corpus?",
            max_num_results=20,
        )

        foreign = [entry for entry in found.data if entry.file_id.startswith("kbdoc_")]
        if not foreign:
            pytest.skip(f"{store.id} retrieved no document of the customer's own")

        with pytest.raises(BadRequestError) as raised:
            openai_client.vector_stores.files.delete(
                foreign[0].file_id, vector_store_id=store.id
            )

        assert raised.value.status_code == 400
        assert "managed outside this server" in str(raised.value)
        # Still there, and still readable.
        assert (
            openai_client.vector_stores.files.retrieve(
                foreign[0].file_id, vector_store_id=store.id
            ).status
            == "completed"
        )

    def test_the_managed_generation_answers_with_what_it_extracted_from_media(
        self, managed_store: Store, openai_client: OpenAI
    ) -> None:
        """A passage of an image or a presentation comes back as searchable text.

        Media extraction is what the managed generation adds over the other, and
        a retrieval is the only place it shows: the corpus holds an image and a
        presentation, and the passage answered for them is the text the
        knowledge base extracted, under the media file's own name.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/kb-managed-connect-ds.html
             stdapi/vector_stores/knowledge_base.py:MANAGED_MEDIA_TYPES
        """
        found = openai_client.vector_stores.search(
            vector_store_id=managed_store.id,
            query="Which passphrases do the sandbox image and presentation carry?",
            max_num_results=20,
        )

        media = [
            entry
            for entry in found.data
            if entry.filename.lower().endswith((".png", ".jpg", ".ppt", ".pptx"))
        ]
        if not media:
            pytest.skip(f"{managed_store.id} holds no media document to retrieve")

        passage = media[0]
        assert any(part.text for part in passage.content if part.type == "text"), (
            f"{passage.filename} was retrieved with no extracted text"
        )


class TestRefusals:
    """What the live service makes the store refuse, and how it is worded.

    Ref: stdapi/vector_stores/engine.py:search
         stdapi/vector_stores/knowledge_base.py:_check_queries
    """

    def test_a_score_threshold_is_refused(
        self, store: Store, openai_client: OpenAI
    ) -> None:
        """A threshold over a score with no stated range is refused, not applied.

        Ref: https://stdapi.ai/api_openai_vector_stores/#scores
             stdapi/vector_stores/engine.py:search
        """
        with pytest.raises(BadRequestError) as raised:
            openai_client.vector_stores.search(
                vector_store_id=store.id,
                query="passphrase",
                ranking_options={"score_threshold": 0.5},
            )

        assert raised.value.status_code == 400
        assert "score_threshold" in str(raised.value)

    def test_a_query_longer_than_the_generation_accepts_is_refused(
        self, store: Store, openai_client: OpenAI
    ) -> None:
        """The query limit is the one the generation the service reports has.

        The two generations take different query lengths, and which applies is
        read from the live knowledge base rather than assumed.

        Ref: stdapi/vector_stores/knowledge_base.py:_check_queries
        """
        limit = _QUERY_LIMITS[store.generation]

        with pytest.raises(BadRequestError) as raised:
            openai_client.vector_stores.search(
                vector_store_id=store.id, query="a" * (limit + 1)
            )

        assert raised.value.status_code == 400
        assert str(limit) in str(raised.value)

        # One character under it is a query the store accepts.
        openai_client.vector_stores.search(
            vector_store_id=store.id, query="a" * limit, max_num_results=1
        )

    def test_a_format_the_generation_does_not_index_is_refused_by_name(
        self, vector_store: Store, openai_client: OpenAI
    ) -> None:
        """A customer-managed knowledge base indexes no image, and says so.

        The refusal lists what this particular store takes, which is read from
        the live knowledge base: the managed generation additionally extracts
        media, so the list is not a constant.

        Ref: stdapi/vector_stores/knowledge_base.py:_inline_content
             stdapi/vector_stores/backend.py:unsupported_file_message
        """
        assert "image/png" in MANAGED_MEDIA_TYPES

        file_id = _upload(openai_client, "probe.png", red_png(), "image/png")
        try:
            with pytest.raises(BadRequestError) as raised:
                openai_client.vector_stores.files.create(
                    vector_store_id=vector_store.id, file_id=file_id
                )

            assert raised.value.status_code == 400
            message = str(raised.value)
            assert "application/pdf" in message
            assert "image/png" not in message
        finally:
            _detach(openai_client, vector_store.id, file_id)
            openai_client.files.delete(file_id)

    def test_a_knowledge_base_that_is_not_allowlisted_is_simply_unknown(
        self, openai_client: OpenAI
    ) -> None:
        """An identifier the deployment was not given answers as a missing store.

        The allowlist is checked before the service is called, so the answer
        cannot distinguish a knowledge base that exists in the account from one
        that does not: both are the same 404.

        Ref: stdapi/vector_stores/knowledge_base.py:check_allowlisted
        """
        with pytest.raises(NotFoundError) as unknown:
            openai_client.vector_stores.retrieve(_UNKNOWN_STORE_ID)

        assert unknown.value.status_code == 404
        assert _UNKNOWN_STORE_ID in str(unknown.value)

    def test_the_managed_generation_takes_the_image_the_other_refuses(
        self, managed_store: Store, openai_client: OpenAI
    ) -> None:
        """The same image the other generation refuses is accepted here.

        The formats a store takes are read from the knowledge base rather than
        assumed, and this is the pair that proves it: one attach of the very
        same file is a 400 on one generation and an accepted document on the
        other.

        Ref: stdapi/vector_stores/knowledge_base.py:_ingested_media_types
        """
        file_id = _upload(openai_client, "probe.png", red_png(), "image/png")
        try:
            attached = openai_client.vector_stores.files.create(
                vector_store_id=managed_store.id, file_id=file_id
            )

            assert attached.id == file_id
            assert attached.vector_store_id == managed_store.id
        finally:
            _detach(openai_client, managed_store.id, file_id)
            openai_client.files.delete(file_id)

    @pytest.mark.slow
    def test_a_declared_format_the_store_cannot_read_fails_as_a_server_error(
        self, vector_store: Store, openai_client: OpenAI
    ) -> None:
        """A file the store said it takes, and then could not index, is our failure.

        ``unsupported_file`` is the answer for a format the store does not
        index, and it is unreachable here: that file never reaches the store.
        What is left is a file of a declared format the store accepted and then
        failed on, which the service reports without naming a reason — so the
        code is ``server_error`` and the message is this API's own.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_KnowledgeBaseDocumentDetail.html
             stdapi/vector_stores/knowledge_base.py:_document_record
        """
        file_id = _upload(
            openai_client,
            "stdapi-live-broken.pdf",
            b"%PDF-1.7\nnot a document\n",
            "application/pdf",
        )
        try:
            openai_client.vector_stores.files.create(
                vector_store_id=vector_store.id, file_id=file_id
            )
            settled = _wait_for_settled(openai_client, vector_store.id, file_id)

            assert settled.status == "failed"
            assert settled.last_error is not None
            assert settled.last_error.code == "server_error"
            assert settled.last_error.message == "The file could not be indexed."
        finally:
            _detach(openai_client, vector_store.id, file_id)
            openai_client.files.delete(file_id)

    @pytest.mark.local
    def test_a_store_that_syncs_its_corpus_refuses_an_attach_in_our_words(
        self,
        vector_store: Store,
        openai_client: OpenAI,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A data source that takes no attachment is refused in this API's terms.

        Only the live service produces the refusal that is translated here, and
        it names its own concepts in it; the caller reads what the store cannot
        do instead, because they chose neither the store nor its corpus.

        Ref: AGENTS.md, "Never Leak Internals"
             stdapi/vector_stores/knowledge_base.py:_addressed_the_wrong_kind
        """
        knowledge_base_id = vector_store.id.removeprefix(STORE_ID_PREFIX)
        synced = _synced_data_source(knowledge_base_id)
        if not synced:
            pytest.skip(
                f"{knowledge_base_id} has no data source syncing its own corpus"
            )
        monkeypatch.setattr(
            SETTINGS,
            "aws_bedrock_knowledge_base_ids",
            [f"{knowledge_base_id}/{synced}"],
        )
        file_id = _upload(
            openai_client, "stdapi-live-refused.txt", _DOCUMENT, "text/plain"
        )
        try:
            with pytest.raises(BadRequestError) as raised:
                openai_client.vector_stores.files.create(
                    vector_store_id=vector_store.id, file_id=file_id
                )

            assert raised.value.status_code == 400
            message = str(raised.value)
            assert "managed outside this server" in message
            assert "files cannot be attached to it" in message
            # Nothing of the service's own vocabulary, nor of the deployment's.
            assert "dataSourceType" not in message
            assert "data source" not in message
            assert synced not in message
            assert knowledge_base_id not in message
        finally:
            openai_client.files.delete(file_id)

    @pytest.mark.local
    def test_a_knowledge_base_with_several_data_sources_needs_one_named(
        self,
        vector_store: Store,
        openai_client: OpenAI,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An allowlist entry naming no data source is refused where it is ambiguous.

        The knowledge base under test has more than one data source, which only
        the live ``ListDataSources`` can say: the entry is rewritten without the
        data source, and the document operations refuse rather than pick one.

        Ref: stdapi/vector_stores/knowledge_base.py:_data_source_id
        """
        knowledge_base_id = vector_store.id.removeprefix(STORE_ID_PREFIX)
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_knowledge_base_ids", [knowledge_base_id]
        )

        with pytest.raises(APIStatusError) as raised:
            openai_client.vector_stores.files.list(vector_store.id, limit=1)

        assert raised.value.status_code == 503
        assert "not available on the current server" in str(raised.value)


@pytest.mark.usefixtures("local_test_client")
class TestSearchBilling:
    """What a live search reports to cost tracking, per generation.

    AWS bills a managed knowledge base per retrieval call and publishes no
    per-retrieval rate for the other generation, whose search is billed by its
    embedding model and its own vector database instead. Only the live service
    says which generation a store is, so only a live search proves the right
    one is recorded -- and that the rate reaches the loaded price catalog,
    which no offline test can show since AWS publishes it on its pricing page
    rather than through the Price List API.

    Ref: https://aws.amazon.com/bedrock/pricing/
         stdapi/usage.py:record_knowledge_base_usage
    """

    def test_a_managed_search_is_recorded_per_retrieval(
        self,
        managed_store: Store,
        openai_client: OpenAI,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """One retrieval of the live store is one billed search unit.

        The entry carries a cost only where the target enables cost tracking,
        so the rate itself is asserted where it always applies:
        ``tests/test_pricing.py`` resolves it, offline and against the live
        catalog.

        Ref: https://aws.amazon.com/bedrock/pricing/
             tests/test_pricing.py:TestKnowledgeBaseRetrievalPrice
        """
        capfd.readouterr()

        openai_client.vector_stores.search(
            vector_store_id=managed_store.id, query=_PLANTED, max_num_results=1
        )

        (entry,) = logged_usage_entries(
            capfd.readouterr().out, service="bedrock-runtime"
        )
        assert entry["model"] == KNOWLEDGE_BASE_MODEL
        assert entry["search_units"] == 1

    def test_a_customer_managed_search_records_no_retrieval_charge(
        self,
        vector_store: Store,
        openai_client: OpenAI,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """The generation AWS publishes no per-retrieval rate for reports none.

        Ref: stdapi/vector_stores/knowledge_base.py:KnowledgeBaseIndex.query_text
        """
        capfd.readouterr()

        openai_client.vector_stores.search(
            vector_store_id=vector_store.id, query=_PLANTED, max_num_results=1
        )

        assert not logged_usage_entries(
            capfd.readouterr().out, service="bedrock-runtime"
        )
