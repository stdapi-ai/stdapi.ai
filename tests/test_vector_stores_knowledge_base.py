"""Tests for a vector store served by a user-provided Bedrock knowledge base.

The store is the customer's: the gateway addresses one they already created,
never creates or deletes one, and manages the documents of its data source. The
tests split in two. The offline unit ones pin the pure translations — the store
and document identifiers, the filter dialect, the status mapping and the
capability declaration the engine and the caller-facing messages are built
from. The offline route ones drive the whole surface against an in-memory
stand-in for the knowledge base, so the sixteen routes answer without
credentials, and pin what a store held elsewhere refuses.

The allowlist is a security boundary rather than a convenience: an identifier
the deployment was not given must be indistinguishable from one that does not
exist, which is asserted byte for byte.

Ref: https://platform.openai.com/docs/api-reference/vector-stores
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Retrieve.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/kb-direct-ingestion.html
     stdapi/vector_stores/knowledge_base.py
     stdapi/vector_stores/engine.py
"""

from base64 import urlsafe_b64decode
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import pytest
from pydantic import ValidationError

from stdapi.api_errors import ApiError
from stdapi.config import SETTINGS, _Settings
from stdapi.pricing import KNOWLEDGE_BASE_MODEL
from stdapi.types.openai_vector_stores import ComparisonFilter, CompoundFilter
from stdapi.vector_stores import knowledge_base
from stdapi.vector_stores.backend import ExternalStore, IndexCapabilities
from stdapi.vector_stores.knowledge_base import (
    CAPABILITIES,
    MANAGED_MEDIA_TYPES,
    KnowledgeBaseIndex,
    document_file_id,
    document_target,
    is_knowledge_base_store,
    knowledge_base_id_of,
    store_id_of,
    translate_filter,
)
from stdapi.vector_stores.registry import backend_for, external_store_for
from tests._helpers import make_client_error
from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi.testclient import TestClient

    from stdapi.types import JsonMapping

#: Knowledge base every test addresses, and the store identifier for it.
_KB_ID = "ABCDE12345"

#: The vector store identifier that knowledge base is addressed as.
_STORE_ID = f"vs_kb_{_KB_ID}"

#: Data source the stand-in reports, and the documents are ingested into.
_DATA_SOURCE_ID = "DS12345678"

#: A second data source of the same knowledge base, syncing its own corpus.
_OTHER_DATA_SOURCE_ID = "DS87654321"

#: Object the synced data source holds, as the store reports its passages under.
_SYNCED_URI = "s3://corpus/handbook.txt"

#: A syntactically valid identifier no test ever allowlists.
_UNKNOWN_STORE_ID = "vs_kb_ZZZZZZZZZZ"

#: A file identifier the Files API stand-in serves.
_FILE_ID = f"file-{'a' * 32}"

#: The moment the stand-in reports every timestamp at.
_MOMENT = datetime(2026, 8, 16, 12, 0, 0, tzinfo=UTC)

#: Documents one listing call may ask for, as the service caps it.
_LIST_PAGE_MAX = 100

#: Seconds within :data:`_MOMENT` each seeded document reports, in identifier order.
_LISTING_TIMES: tuple[int, ...] = (1, 6, 2, 7, 3, 8)

#: Identifier letters the seeded documents are named after, in identifier order.
_LISTING_LETTERS: tuple[str, ...] = ("a", "b", "c", "d", "e", "f")

#: Pages a cursor walk may take before the test calls it non-terminating.
_WALK_MAX_PAGES: int = 12


class _FakeAgentClient:
    """The knowledge base management calls, answered in memory."""

    def __init__(self) -> None:
        #: Generation the described knowledge base reports.
        self.kind = "VECTOR"
        #: Lifecycle status it reports.
        self.status = "ACTIVE"
        #: Data sources it holds.
        self.data_sources = [_DATA_SOURCE_ID]
        #: Kind of each data source, which a document identifier must match.
        self.data_source_types = {_DATA_SOURCE_ID: "CUSTOM"}
        #: Documents it holds, as ``documentDetails`` entries.
        self.documents: list[dict[str, Any]] = []
        #: Every call made, as ``(operation, arguments)``.
        self.calls: list[tuple[str, dict[str, Any]]] = []
        #: Knowledge bases that exist.
        self.known = {_KB_ID}

    def _record(self, operation: str, arguments: dict[str, Any]) -> None:
        """Record one call."""
        self.calls.append((operation, arguments))

    def operations(self) -> list[str]:
        """Return the operations called so far, in order."""
        return [operation for operation, _ in self.calls]

    def arguments_of(self, operation: str) -> dict[str, Any]:
        """Return the arguments of the last call to *operation*."""
        return next(
            arguments for name, arguments in reversed(self.calls) if name == operation
        )

    async def get_knowledge_base(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Describe the knowledge base, or report that it does not exist."""
        self._record("GetKnowledgeBase", kwargs)
        if kwargs["knowledgeBaseId"] not in self.known:
            missing = make_client_error("ResourceNotFoundException", "GetKnowledgeBase")
            raise missing from None
        configuration: dict[str, Any] = {"type": self.kind}
        if self.kind == "VECTOR":
            configuration["vectorKnowledgeBaseConfiguration"] = {
                "embeddingModelArn": (
                    "arn:aws:bedrock:us-east-1::foundation-model/"
                    "amazon.titan-embed-text-v2:0"
                )
            }
        return {
            "knowledgeBase": {
                "knowledgeBaseId": kwargs["knowledgeBaseId"],
                "name": "support-corpus",
                "description": "The handbooks.",
                "knowledgeBaseConfiguration": configuration,
                "status": self.status,
                "createdAt": _MOMENT,
                "updatedAt": _MOMENT,
            }
        }

    async def list_data_sources(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """List the data sources of the knowledge base."""
        self._record("ListDataSources", kwargs)
        return {
            "dataSourceSummaries": [
                {"dataSourceId": entry} for entry in self.data_sources
            ]
        }

    def _check_kind(self, data_source_id: str, kind: str) -> None:
        """Refuse a document addressed as a kind its data source is not.

        The service checks the two against each other and refuses the whole
        call, in its own words, whichever document operation was asked for.
        """
        if self.data_source_types.get(data_source_id, "CUSTOM") == kind:
            return
        mismatched = make_client_error(
            "ValidationException",
            "IngestKnowledgeBaseDocuments",
            message=(
                f"The dataSourceType that you specified in {kind} doesn't match "
                "the type of the data source specified in the request header. "
                "Check that both types match and retry your request."
            ),
        )
        raise mismatched from None

    async def ingest_knowledge_base_documents(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Ingest documents, reporting each one as accepted for indexing."""
        self._record("IngestKnowledgeBaseDocuments", kwargs)
        details = []
        for document in kwargs["documents"]:
            self._check_kind(
                kwargs["dataSourceId"], document["content"]["dataSourceType"]
            )
            identifier = {
                "dataSourceType": "CUSTOM",
                "custom": dict(
                    document["content"]["custom"]["customDocumentIdentifier"]
                ),
            }
            entry = {
                "knowledgeBaseId": kwargs["knowledgeBaseId"],
                "dataSourceId": kwargs["dataSourceId"],
                "identifier": identifier,
                "status": "IN_PROGRESS",
                "updatedAt": _MOMENT,
            }
            self.documents.append(entry)
            details.append(entry)
        return {"documentDetails": details}

    async def list_knowledge_base_documents(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """List one page of the documents the knowledge base holds.

        The service refuses a page over a hundred documents and hands the rest
        back behind a token, so the stand-in does the same: a caller asking for
        more is answered exactly as the managed generation answers it.
        """
        self._record("ListKnowledgeBaseDocuments", kwargs)
        size = int(kwargs["maxResults"])
        if size > _LIST_PAGE_MAX:
            too_many = make_client_error(
                "ValidationException", "ListKnowledgeBaseDocuments"
            )
            raise too_many from None
        start = int(kwargs.get("nextToken") or 0)
        page = self.documents[start : start + size]
        response: dict[str, Any] = {"documentDetails": page}
        if start + size < len(self.documents):
            response["nextToken"] = str(start + size)
        return response

    async def get_knowledge_base_documents(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Read the named documents, reporting the absent ones as not found."""
        self._record("GetKnowledgeBaseDocuments", kwargs)
        data_source_id = kwargs["dataSourceId"]
        if data_source_id not in self.data_sources:
            unknown = make_client_error(
                "ResourceNotFoundException", "GetKnowledgeBaseDocuments"
            )
            raise unknown from None
        details = []
        for wanted in kwargs["documentIdentifiers"]:
            self._check_kind(data_source_id, wanted["dataSourceType"])
            found = [
                entry
                for entry in self.documents
                if entry["identifier"] == wanted
                and entry["dataSourceId"] == data_source_id
            ]
            details.extend(found or [{**_absent(data_source_id, wanted)}])
        return {"documentDetails": details}

    async def delete_knowledge_base_documents(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Delete the named documents."""
        self._record("DeleteKnowledgeBaseDocuments", kwargs)
        data_source_id = kwargs["dataSourceId"]
        wanted = kwargs["documentIdentifiers"]
        self.documents = [
            entry
            for entry in self.documents
            if entry["identifier"] not in wanted
            or entry["dataSourceId"] != data_source_id
        ]
        return {"documentDetails": []}


def _absent(data_source_id: str, identifier: dict[str, Any]) -> dict[str, Any]:
    """Return the details a document that does not exist is reported with."""
    return {
        "knowledgeBaseId": _KB_ID,
        "dataSourceId": data_source_id,
        "identifier": identifier,
        "status": "NOT_FOUND",
        "updatedAt": _MOMENT,
    }


class _FakeRuntimeClient:
    """The retrieval call, answered in memory."""

    def __init__(self) -> None:
        #: Every retrieval issued, as its arguments.
        self.calls: list[dict[str, Any]] = []
        #: The results every retrieval answers with.
        self.results: list[dict[str, Any]] = [
            {
                "content": {"type": "TEXT", "text": "The refund window is 30 days."},
                "documentId": _FILE_ID,
                "location": {
                    "type": "CUSTOM",
                    "customDocumentLocation": {"id": _FILE_ID},
                },
                "metadata": {
                    "stdapi-filename": "handbook.txt",
                    "x-amz-bedrock-kb-data-source-id": _DATA_SOURCE_ID,
                    "team": "support",
                    "revision": 3,
                },
                "score": 12.5,
            }
        ]

    async def retrieve(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Answer one retrieval."""
        self.calls.append(kwargs)
        return {"retrievalResults": list(self.results)}


class _FakeFile:
    """The uploaded file record the Files API answers with."""

    def __init__(self, filename: str) -> None:
        self.filename = filename


class _FakeFiles:
    """The Files API, answered in memory."""

    def __init__(self) -> None:
        #: Uploaded content and media type, keyed by the bare file payload.
        self.uploads: dict[str, tuple[bytes, str, str]] = {
            "a" * 32: (b"The refund window is 30 days.", "text/plain", "handbook.txt")
        }

    def upload(self, content: bytes, media_type: str, filename: str) -> str:
        """Register an uploaded file and return the identifier attaching it."""
        payload = f"{len(self.uploads):032d}".replace("0", "b")
        self.uploads[payload] = (content, media_type, filename)
        return f"file-{payload}"

    async def get_file(self, payload: str) -> _FakeFile:
        """Return the uploaded file's record."""
        return _FakeFile(self.uploads[payload][2])

    async def get_file_content(self, payload: str) -> tuple[Any, str]:
        """Return the uploaded file's content as the Files API streams it."""
        content, media_type, _ = self.uploads[payload]

        async def _stream() -> AsyncIterator[bytes]:
            yield content

        return _stream(), media_type


class _Backend:
    """The stand-ins one test drives the knowledge base store through."""

    def __init__(self) -> None:
        self.agent = _FakeAgentClient()
        self.runtime = _FakeRuntimeClient()
        self.files = _FakeFiles()


@pytest.fixture
def knowledge_base_backend(monkeypatch: pytest.MonkeyPatch) -> _Backend:
    """Serve one allowlisted knowledge base from an in-memory stand-in.

    Only the boundaries are replaced — the two AWS clients and the Files API —
    so the routes, the engine's dispatch and this backend's own translations
    are the code actually under test.

    Returns:
        The stand-ins, for the test to seed and to assert against.
    """
    backend = _Backend()
    monkeypatch.setattr(SETTINGS, "aws_bedrock_knowledge_base_ids", [_KB_ID])
    # Given knowledge bases and nothing else: no records bucket, no index.
    monkeypatch.setattr(SETTINGS, "aws_s3_bucket", None)
    monkeypatch.setattr(knowledge_base, "agent_client", lambda: backend.agent)
    monkeypatch.setattr(knowledge_base, "runtime_client", lambda: backend.runtime)
    monkeypatch.setattr(knowledge_base, "get_file", backend.files.get_file)
    monkeypatch.setattr(
        knowledge_base, "get_file_content", backend.files.get_file_content
    )
    return backend


def _error_of(response: Any) -> dict[str, Any]:  # noqa: ANN401
    """Return the error envelope of a failed response."""
    payload: dict[str, Any] = response.json()["error"]
    return payload


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


@pytest.mark.local
class TestIdentifiers:
    """The identifiers a knowledge base store and its documents are addressed by."""

    def test_store_identifier_round_trips(self) -> None:
        """A knowledge base is addressed as a store, and read back from it.

        Ref: stdapi/vector_stores/knowledge_base.py:store_id_of
        """
        assert store_id_of(_KB_ID) == _STORE_ID
        assert knowledge_base_id_of(_STORE_ID) == _KB_ID
        assert is_knowledge_base_store(_STORE_ID)
        assert not is_knowledge_base_store("vs_" + "0" * 26)

    def test_attached_file_keeps_its_own_identifier(self) -> None:
        """A file this server attached is addressed by the id it was uploaded with.

        Ref: stdapi/vector_stores/knowledge_base.py:document_file_id
        """
        identifier: JsonMapping = {
            "dataSourceType": "CUSTOM",
            "custom": {"id": _FILE_ID},
        }
        assert document_file_id(identifier) == _FILE_ID
        # No data source travels with it: it belongs to the store's own.
        assert document_target(_FILE_ID) == ("", identifier)

    @pytest.mark.parametrize(
        "identifier",
        [
            {
                "dataSourceType": "S3",
                "s3": {"uri": "s3://corpus/handbooks/refunds.pdf"},
            },
            {"dataSourceType": "CUSTOM", "custom": {"id": "crm/case/4711"}},
        ],
    )
    def test_a_document_of_the_data_source_round_trips(
        self, identifier: dict[str, Any]
    ) -> None:
        """A document located by a URI is addressable, and decodes back unchanged.

        The data source it belongs to travels with it: a knowledge base holds
        several, and a document is only addressable in the one that holds it.

        Ref: stdapi/vector_stores/knowledge_base.py:document_target
        """
        file_id = document_file_id(identifier, _DATA_SOURCE_ID)
        assert file_id.startswith("kbdoc_")
        # Opaque, but URL-safe: it travels as a path parameter.
        assert "/" not in file_id
        assert document_target(file_id) == (_DATA_SOURCE_ID, identifier)

    def test_a_document_identifier_that_does_not_decode_is_unknown(self) -> None:
        """A forged opaque identifier is answered as an unknown file, not a 500.

        Ref: stdapi/vector_stores/knowledge_base.py:document_target
        """
        with pytest.raises(ApiError) as raised:
            document_target("kbdoc_A")
        assert raised.value.status == 404

    def test_a_forged_data_source_is_answered_as_an_unknown_file(self) -> None:
        """The data source an identifier carries is checked before it is used.

        The identifier is the caller's to forge, and the data source it names
        reaches the service: anything that is not a data source identifier is an
        unknown file rather than a call made on the caller's behalf.

        Ref: stdapi/vector_stores/knowledge_base.py:document_target
        """
        forged = document_file_id(
            {"dataSourceType": "S3", "s3": {"uri": "s3://corpus/a.pdf"}},
            "../../elsewhere",
        )
        with pytest.raises(ApiError) as raised:
            document_target(forged)
        assert raised.value.status == 404

    def test_a_passage_of_a_synced_document_is_addressed_by_its_object_uri(
        self,
    ) -> None:
        """A location reported as a custom one, but holding a URI, is an object.

        The managed generation reports a document synced from a bucket under the
        custom location, while its data source addresses that same document by
        its object URI: taking the location at face value builds an identifier
        the read path cannot resolve.

        Ref: stdapi/vector_stores/knowledge_base.py:_location_identifier
        """
        located = knowledge_base._location_identifier(  # noqa: SLF001
            {
                "location": {
                    "type": "CUSTOM",
                    "customDocumentLocation": {"id": "s3://corpus/handbook.txt"},
                }
            }
        )

        assert located == {
            "dataSourceType": "S3",
            "s3": {"uri": "s3://corpus/handbook.txt"},
        }


@pytest.mark.local
class TestCapabilityDeclaration:
    """What the backend declares it can express, which every refusal is built from."""

    def test_declares_every_filter_operator_and_combinator(self) -> None:
        """All eight operators and both combinators map one to one.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_RetrievalFilter.html
        """
        assert CAPABILITIES.filter_operators == frozenset(
            {"eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"}
        )
        assert CAPABILITIES.filter_combinators == frozenset({"and", "or"})

    def test_declares_the_documents_it_indexes_as_they_stand(self) -> None:
        """The document formats are declared, so a refusal can name them.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-ds.html
        """
        assert "application/pdf" in CAPABILITIES.ingested_media_types
        assert (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            in CAPABILITIES.ingested_media_types
        )
        assert CAPABILITIES.may_ingest("application/pdf")
        assert CAPABILITIES.may_ingest("text/plain")
        assert not CAPABILITIES.may_ingest("application/zip")
        # The wider set is a property of the managed generation alone.
        assert "image/png" not in CAPABILITIES.ingested_media_types
        assert "image/png" in MANAGED_MEDIA_TYPES

    def test_declares_that_it_chunks_and_that_its_score_is_not_normalised(self) -> None:
        """The store cuts its own passages and measures its own relevance.

        Ref: stdapi/vector_stores/backend.py:IndexCapabilities
        """
        assert CAPABILITIES.chunks_on_ingestion is True
        assert CAPABILITIES.normalised_score is False
        assert CAPABILITIES.ingests_decodable_text is True
        assert CAPABILITIES.max_chunk_bytes == 0
        assert isinstance(CAPABILITIES, IndexCapabilities)

    def test_the_backend_answers_both_contracts(self) -> None:
        """It is a vector index and a store that answers for itself.

        Ref: stdapi/vector_stores/backend.py:ExternalStore
        """
        index = KnowledgeBaseIndex()
        assert isinstance(index, ExternalStore)
        assert index.capabilities is CAPABILITIES


@pytest.mark.local
class TestFilterTranslation:
    """The filter dialect, which is the whole reason the operators are declared."""

    @pytest.mark.parametrize(
        ("operator", "translated"),
        [
            ("eq", "equals"),
            ("ne", "notEquals"),
            ("gt", "greaterThan"),
            ("gte", "greaterThanOrEquals"),
            ("lt", "lessThan"),
            ("lte", "lessThanOrEquals"),
            ("in", "in"),
            ("nin", "notIn"),
        ],
    )
    def test_every_operator_maps_one_to_one(
        self,
        operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"],
        translated: str,
    ) -> None:
        """Each upstream operator becomes its retrieval counterpart.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_RetrievalFilter.html
        """
        value: Any = ["a", "b"] if operator in ("in", "nin") else 3
        translated_filter = translate_filter(
            ComparisonFilter(key="revision", type=operator, value=value)
        )
        assert translated_filter == {translated: {"key": "revision", "value": value}}

    def test_a_combination_maps_to_its_counterpart(self) -> None:
        """`and` and `or` become the retrieval combinators, over the same keys.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_RetrievalFilter.html
        """
        translated = translate_filter(
            CompoundFilter(
                type="or",
                filters=[
                    ComparisonFilter(key="team", type="eq", value="support"),
                    {
                        "type": "and",
                        "filters": [
                            {"key": "revision", "type": "gte", "value": 2},
                            {"key": "draft", "type": "eq", "value": False},
                        ],
                    },
                ],
            )
        )
        assert translated == {
            "orAll": [
                {"equals": {"key": "team", "value": "support"}},
                {
                    "andAll": [
                        {"greaterThanOrEquals": {"key": "revision", "value": 2}},
                        {"equals": {"key": "draft", "value": False}},
                    ]
                },
            ]
        }

    def test_a_combination_of_one_is_flattened(self) -> None:
        """A combination needs two members, so a single one is emitted alone.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_RetrievalFilter.html
        """
        translated = translate_filter(
            CompoundFilter(
                type="and",
                filters=[ComparisonFilter(key="team", type="eq", value="support")],
            )
        )
        assert translated == {"equals": {"key": "team", "value": "support"}}


@pytest.mark.local
class TestAllowlistSetting:
    """The setting that makes a knowledge base addressable, and nothing else."""

    def test_rejects_an_entry_that_is_not_a_knowledge_base(self) -> None:
        """A malformed entry fails startup rather than 404ing at request time.

        Ref: stdapi/config.py:_Settings._validate_knowledge_base_ids
        """
        with pytest.raises(ValidationError) as raised:
            _Settings(aws_bedrock_knowledge_base_ids=["not-an-id"])
        assert "aws_bedrock_knowledge_base_ids" in str(raised.value)

    @pytest.mark.parametrize(
        "value", [_KB_ID, f"{_KB_ID}/{_DATA_SOURCE_ID}", f"{_KB_ID},FGHIJ67890"]
    )
    def test_accepts_an_identifier_with_and_without_its_data_source(
        self, value: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both entry forms parse, comma-separated as an environment variable does.

        The allowlist is read from the environment, which is where the
        comma-separated form comes from, so it is set there rather than passed
        pre-split.

        Ref: stdapi/config.py:_Settings.aws_bedrock_knowledge_base_ids
        """
        monkeypatch.setenv("AWS_BEDROCK_KNOWLEDGE_BASE_IDS", value)

        settings = _Settings()

        assert settings.aws_bedrock_knowledge_base_ids == value.split(",")

    def test_is_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No knowledge base is addressable until the deployment names one.

        The variable is unset first: a checkout that allowlists one for its live
        tests must not make this read as a default.

        Ref: stdapi/config.py:_Settings.aws_bedrock_knowledge_base_ids
        """
        monkeypatch.delenv("AWS_BEDROCK_KNOWLEDGE_BASE_IDS", raising=False)

        assert _Settings().aws_bedrock_knowledge_base_ids == []

    def test_the_registry_routes_the_identifier(
        self, knowledge_base_backend: _Backend
    ) -> None:
        """The store identifier alone decides which backend answers.

        Ref: stdapi/vector_stores/registry.py:external_store_for
        """
        del knowledge_base_backend
        assert isinstance(backend_for(_STORE_ID), KnowledgeBaseIndex)
        assert external_store_for(_STORE_ID) is not None
        assert external_store_for("vs_" + "0" * 26) is None


@pytest.mark.local
class TestAllowlistIsNotProbeable:
    """An identifier the deployment was not given cannot be told from an unknown one."""

    def test_an_unallowlisted_store_answers_exactly_as_a_missing_one(
        self,
        app_client: TestClient,
        knowledge_base_backend: _Backend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The same identifier answers identically, allowlisted or not.

        The identifier is held fixed and only the allowlist moves, so the two
        answers can be compared byte for byte: were they to differ, the setting
        could be probed for which knowledge bases the account holds.

        Ref: stdapi/vector_stores/engine.py:parse_store_id
             stdapi/vector_stores/knowledge_base.py:check_allowlisted
        """
        # Not allowlisted: refused before any call is made.
        refused = app_client.get(f"/v1/vector_stores/{_UNKNOWN_STORE_ID}")
        assert not knowledge_base_backend.agent.calls

        # Allowlisted, but the knowledge base does not exist.
        monkeypatch.setattr(
            SETTINGS,
            "aws_bedrock_knowledge_base_ids",
            [knowledge_base_id_of(_UNKNOWN_STORE_ID)],
        )
        missing = app_client.get(f"/v1/vector_stores/{_UNKNOWN_STORE_ID}")
        assert knowledge_base_backend.agent.operations() == ["GetKnowledgeBase"]

        assert refused.status_code == missing.status_code == 404
        assert refused.json() == missing.json()
        assert _error_of(refused)["message"] == (
            f"No vector store found with id '{_UNKNOWN_STORE_ID}'."
        )

    def test_an_unallowlisted_store_is_absent_from_the_listing(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """Only the allowlisted knowledge bases are listed.

        Ref: stdapi/vector_stores/engine.py:list_stores
        """
        del knowledge_base_backend
        listed = app_client.get("/v1/vector_stores").json()["data"]
        assert [store["id"] for store in listed] == [_STORE_ID]

    @pytest.mark.parametrize("path", ["/files", "/files/file-" + "a" * 32, "/search"])
    def test_every_route_refuses_an_unallowlisted_store(
        self, app_client: TestClient, knowledge_base_backend: _Backend, path: str
    ) -> None:
        """No route reaches the backend for an identifier that was not given.

        Ref: stdapi/vector_stores/engine.py:parse_store_id
        """
        url = f"/v1/vector_stores/{_UNKNOWN_STORE_ID}{path}"
        response = (
            app_client.post(url, json={"query": "refunds"})
            if path == "/search"
            else app_client.get(url)
        )
        assert response.status_code == 404
        assert not knowledge_base_backend.agent.calls
        assert not knowledge_base_backend.runtime.calls


@pytest.mark.local
class TestStore:
    """What a knowledge base store reports about itself."""

    def test_retrieve_reports_what_the_knowledge_base_says(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """The name, description and timestamps are read, never invented.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/retrieve
        """
        store = app_client.get(f"/v1/vector_stores/{_STORE_ID}").json()
        assert store["id"] == _STORE_ID
        assert store["object"] == "vector_store"
        assert store["name"] == "support-corpus"
        assert store["description"] == "The handbooks."
        assert store["created_at"] == int(_MOMENT.timestamp())
        assert store["status"] == "completed"
        # The corpus is the customer's: counting it is not invented here.
        assert store["usage_bytes"] == 0
        assert store["file_counts"]["total"] == 0
        assert "expires_at" not in store
        assert knowledge_base_backend.agent.arguments_of("GetKnowledgeBase") == {
            "knowledgeBaseId": _KB_ID
        }

    def test_a_knowledge_base_still_building_reports_in_progress(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """A store that is not ready to answer says so.

        Ref: stdapi/vector_stores/models.py:StoreRecord.status
        """
        knowledge_base_backend.agent.status = "CREATING"
        store = app_client.get(f"/v1/vector_stores/{_STORE_ID}").json()
        assert store["status"] == "in_progress"

    def test_the_listing_serves_them_without_a_store_of_our_own(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """A deployment given only knowledge bases still lists them.

        Ref: stdapi/vector_stores/engine.py:list_stores
        """
        del knowledge_base_backend
        page = app_client.get("/v1/vector_stores").json()
        assert page["object"] == "list"
        assert page["has_more"] is False
        assert page["data"][0]["name"] == "support-corpus"

    def test_it_cannot_be_deleted(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """The store is the customer's, so deleting it is refused.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/delete
        """
        response = app_client.delete(f"/v1/vector_stores/{_STORE_ID}")
        assert response.status_code == 400
        assert "managed outside this server" in _error_of(response)["message"]
        assert "DeleteKnowledgeBase" not in knowledge_base_backend.agent.operations()

    def test_its_name_and_expiration_cannot_be_changed(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """Name, metadata and expiration are read from it, never written to it.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/modify
        """
        del knowledge_base_backend
        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}",
            json={"name": "renamed", "expires_after": {"days": 3}},
        )
        assert response.status_code == 400
        assert "cannot be changed here" in _error_of(response)["message"]


@pytest.mark.local
class TestDocuments:
    """Attaching, listing, reading and removing the documents of the store."""

    def test_attaching_a_file_ingests_it_as_a_document(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """The uploaded file becomes a document keyed by its own identifier.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/createFile
        """
        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files",
            json={"file_id": _FILE_ID, "attributes": {"team": "support"}},
        )
        assert response.status_code == 200
        attached = response.json()
        assert attached["id"] == _FILE_ID
        assert attached["vector_store_id"] == _STORE_ID
        assert attached["status"] == "in_progress"
        # Neither is known, and neither is invented.
        assert "chunking_strategy" not in attached
        assert "attributes" not in attached

        arguments = knowledge_base_backend.agent.arguments_of(
            "IngestKnowledgeBaseDocuments"
        )
        assert arguments["knowledgeBaseId"] == _KB_ID
        assert arguments["dataSourceId"] == _DATA_SOURCE_ID
        document = arguments["documents"][0]
        assert document["content"]["dataSourceType"] == "CUSTOM"
        assert document["content"]["custom"]["customDocumentIdentifier"] == {
            "id": _FILE_ID
        }
        assert document["content"]["custom"]["inlineContent"] == {
            "type": "TEXT",
            "textContent": {"data": "The refund window is 30 days."},
        }
        attributes = {
            entry["key"]: entry["value"]
            for entry in document["metadata"]["inlineAttributes"]
        }
        assert attributes["team"] == {"type": "STRING", "stringValue": "support"}
        assert attributes["stdapi-filename"] == {
            "type": "STRING",
            "stringValue": "handbook.txt",
        }

    def test_a_document_format_is_ingested_as_it_stands(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """A PDF is sent as it stands rather than refused as unreadable text.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-ds.html
        """
        file_id = knowledge_base_backend.files.upload(
            b"%PDF-1.7\x00binary", "application/pdf", "refunds.pdf"
        )
        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files", json={"file_id": file_id}
        )
        assert response.status_code == 200
        content = knowledge_base_backend.agent.arguments_of(
            "IngestKnowledgeBaseDocuments"
        )["documents"][0]["content"]["custom"]["inlineContent"]
        assert content["type"] == "BYTE"
        assert content["byteContent"]["mimeType"] == "application/pdf"

    def test_a_format_the_store_cannot_index_is_refused_by_name(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """The refusal names the formats this store does accept.

        Ref: AGENTS.md, "Unsupported features: reject the explicit ask"
        """
        file_id = knowledge_base_backend.files.upload(
            b"PK\x03\x04binary", "application/zip", "corpus.zip"
        )
        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files", json={"file_id": file_id}
        )
        assert response.status_code == 400
        message = _error_of(response)["message"]
        assert "cannot be indexed" in message
        assert "application/pdf" in message
        assert "IngestKnowledgeBaseDocuments" not in (
            knowledge_base_backend.agent.operations()
        )

    def test_the_refusal_lists_the_formats_this_particular_store_takes(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """The formats named are the serving store's own, not a fixed list.

        The managed generation takes more than the other one, so the same file
        refused by both is explained with a different set of formats.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/kb-managed-create.html
        """
        file_id = knowledge_base_backend.files.upload(
            b"PK\x03\x04binary", "application/zip", "corpus.zip"
        )
        url = f"/v1/vector_stores/{_STORE_ID}/files"
        message = _error_of(app_client.post(url, json={"file_id": file_id}))["message"]
        assert "application/pdf" in message
        assert "image/png" not in message

        knowledge_base_backend.agent.kind = "MANAGED"
        managed = _error_of(app_client.post(url, json={"file_id": file_id}))["message"]
        assert "image/png" in managed
        assert managed != message

    def test_a_store_that_indexes_documents_is_not_sent_elsewhere(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """No alternative is offered by a store that already takes documents.

        Ref: stdapi/vector_stores/backend.py:unsupported_file_message
        """
        file_id = knowledge_base_backend.files.upload(
            b"PK\x03\x04binary", "application/zip", "corpus.zip"
        )
        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files", json={"file_id": file_id}
        )
        assert response.status_code == 400
        message = _error_of(response)["message"]
        assert "knowledge base" not in message
        # The refusal is actionable on its own: it says what to send instead.
        assert "Provide the content" in message

    def test_a_managed_store_indexes_what_the_other_generation_does_not(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """The managed generation takes slides; the other one does not.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/kb-managed-create.html
        """
        media = (
            "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )
        file_id = knowledge_base_backend.files.upload(
            b"PK\x03\x04\xff\xfedeck", media, "d.pptx"
        )
        url = f"/v1/vector_stores/{_STORE_ID}/files"
        assert app_client.post(url, json={"file_id": file_id}).status_code == 400

        knowledge_base_backend.agent.kind = "MANAGED"
        assert app_client.post(url, json={"file_id": file_id}).status_code == 200

    def test_listing_reads_the_documents_back(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """Attached and pre-existing documents are both listed and addressable.

        A document the customer ingested into the same data source under an
        identifier of their own is listed next to the files attached here, under
        an opaque identifier carrying where it lives.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/listFiles
        """
        app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files", json={"file_id": _FILE_ID}
        )
        knowledge_base_backend.agent.documents.append(
            {
                "knowledgeBaseId": _KB_ID,
                "dataSourceId": _DATA_SOURCE_ID,
                "identifier": {
                    "dataSourceType": "CUSTOM",
                    "custom": {"id": "crm/case/4711"},
                },
                "status": "INDEXED",
                "updatedAt": _MOMENT,
            }
        )
        page = app_client.get(f"/v1/vector_stores/{_STORE_ID}/files").json()
        identifiers = {entry["id"] for entry in page["data"]}
        assert _FILE_ID in identifiers
        opaque = next(entry for entry in identifiers if entry.startswith("kbdoc_"))
        encoded = opaque.removeprefix("kbdoc_")
        decoded = urlsafe_b64decode(f"{encoded}{'=' * (-len(encoded) % 4)}").decode()
        assert decoded == f"c:{_DATA_SOURCE_ID}:crm/case/4711"
        # Every listed identifier can be read back through the per-file route.
        read = app_client.get(f"/v1/vector_stores/{_STORE_ID}/files/{opaque}")
        assert read.status_code == 200
        assert read.json()["status"] == "completed"

    def test_a_deleted_document_is_not_listed_as_one_still_settling(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """A tombstone the data source keeps is not a file of the store.

        The service reports a document it no longer holds among the others, and
        a file reported as still indexing would never settle.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_KnowledgeBaseDocumentDetail.html
        """
        knowledge_base_backend.agent.documents.append(
            {
                "knowledgeBaseId": _KB_ID,
                "dataSourceId": _DATA_SOURCE_ID,
                "identifier": {
                    "dataSourceType": "CUSTOM",
                    "custom": {"id": "file-deleted-0001"},
                },
                "status": "NOT_FOUND",
                "updatedAt": _MOMENT,
            }
        )

        page = app_client.get(f"/v1/vector_stores/{_STORE_ID}/files").json()

        assert page["data"] == []

    def test_listing_reads_every_page_the_service_hands_back(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """The listing is read a hundred documents at a time, and merged.

        A page over that is refused outright by the managed generation, so the
        whole listing is walked with the service's token instead of asked for at
        once.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_ListKnowledgeBaseDocuments.html
        """
        agent = knowledge_base_backend.agent
        agent.documents.extend(
            {
                "knowledgeBaseId": _KB_ID,
                "dataSourceId": _DATA_SOURCE_ID,
                "identifier": {
                    "dataSourceType": "CUSTOM",
                    "custom": {"id": f"d{index}"},
                },
                "status": "INDEXED",
                "updatedAt": _MOMENT,
            }
            for index in range(_LIST_PAGE_MAX + 1)
        )

        page = app_client.get(f"/v1/vector_stores/{_STORE_ID}/files?limit=100")

        assert page.status_code == 200
        asked = [
            arguments["maxResults"]
            for operation, arguments in agent.calls
            if operation == "ListKnowledgeBaseDocuments"
        ]
        assert asked == [_LIST_PAGE_MAX, _LIST_PAGE_MAX]
        assert page.json()["has_more"] is True

    @pytest.mark.parametrize(
        ("reported", "status"),
        [
            ("INDEXED", "completed"),
            ("PARTIALLY_INDEXED", "completed"),
            ("STARTING", "in_progress"),
            ("FAILED", "failed"),
            ("IGNORED", "failed"),
        ],
    )
    def test_a_document_status_becomes_a_file_status(
        self,
        app_client: TestClient,
        knowledge_base_backend: _Backend,
        reported: str,
        status: str,
    ) -> None:
        """Every indexing state maps to one the API defines.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/getFile
        """
        app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files", json={"file_id": _FILE_ID}
        )
        knowledge_base_backend.agent.documents[0]["status"] = reported
        knowledge_base_backend.agent.documents[0]["statusReason"] = "Unreadable."
        read = app_client.get(f"/v1/vector_stores/{_STORE_ID}/files/{_FILE_ID}").json()
        assert read["status"] == status
        if status == "failed":
            assert read["last_error"]["message"] == "Unreadable."
            expected = "unsupported_file" if reported == "IGNORED" else "server_error"
            assert read["last_error"]["code"] == expected

    def test_a_document_the_store_fails_is_a_server_error_and_not_a_format_refusal(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """A file of a declared format that fails to index is a server error.

        ``unsupported_file`` is reserved for the ``IGNORED`` status the service
        models, and it is unreachable on this backend: a format the store does
        not declare is refused before any call is made, so a file that gets far
        enough to fail is one the store said it indexes. The service names no
        reason for it, and none is invented.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_KnowledgeBaseDocumentDetail.html
             stdapi/vector_stores/knowledge_base.py:_document_record
        """
        app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files", json={"file_id": _FILE_ID}
        )
        knowledge_base_backend.agent.documents[0]["status"] = "FAILED"
        knowledge_base_backend.agent.documents[0]["statusReason"] = ""

        read = app_client.get(f"/v1/vector_stores/{_STORE_ID}/files/{_FILE_ID}").json()

        assert read["status"] == "failed"
        assert read["last_error"]["code"] == "server_error"
        assert read["last_error"]["message"] == "The file could not be indexed."

    def test_a_format_outside_the_declaration_never_reaches_the_store(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """The refusal that ``unsupported_file`` would report happens first.

        This is the other half of why the ``IGNORED`` branch cannot be reached
        here: the only file the store would ignore for its format is one it
        never receives.

        Ref: stdapi/vector_stores/knowledge_base.py:_inline_content
        """
        file_id = knowledge_base_backend.files.upload(
            b"\x89PNG\r\n\x1a\n", "image/png", "logo.png"
        )

        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files", json={"file_id": file_id}
        )

        assert response.status_code == 400
        assert "cannot be indexed" in _error_of(response)["message"]
        assert "IngestKnowledgeBaseDocuments" not in (
            knowledge_base_backend.agent.operations()
        )

    def test_reading_a_document_the_store_does_not_hold_is_a_404(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """A file that was never attached is unknown, not an empty record.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/getFile
        """
        del knowledge_base_backend
        response = app_client.get(f"/v1/vector_stores/{_STORE_ID}/files/{_FILE_ID}")
        assert response.status_code == 404
        assert _error_of(response)["message"] == (
            f"No file found with id '{_FILE_ID}'."
        )

    def test_deleting_a_file_deletes_the_document(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """Detaching a file removes it from the knowledge base.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/deleteFile
        """
        app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files", json={"file_id": _FILE_ID}
        )
        response = app_client.delete(f"/v1/vector_stores/{_STORE_ID}/files/{_FILE_ID}")
        assert response.status_code == 200
        assert response.json() == {
            "id": _FILE_ID,
            "object": "vector_store.file.deleted",
            "deleted": True,
        }
        assert knowledge_base_backend.agent.documents == []

    def test_deleting_a_file_the_store_does_not_hold_is_a_404(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """Nothing is deleted for an identifier the store never held.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/deleteFile
        """
        response = app_client.delete(f"/v1/vector_stores/{_STORE_ID}/files/{_FILE_ID}")
        assert response.status_code == 404
        assert "DeleteKnowledgeBaseDocuments" not in (
            knowledge_base_backend.agent.operations()
        )


@pytest.mark.local
class TestDocumentListingOrder:
    """A document listing orders on the time it reports, across pages.

    Issue #165: the order came from the document identifier, which is derived
    from the source URI and carries no time at all, so the listing was in an
    arbitrary order while reporting a timestamp per document.

    The seeded times invert the identifier order in pairs, so every page of two
    is internally ordered whichever of the two quantities the listing ran on:
    only concatenating the pages tells them apart.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_KnowledgeBaseDocumentDetail.html
         stdapi/vector_stores/knowledge_base.py:list_documents
    """

    @pytest.fixture
    def seeded_documents(self, knowledge_base_backend: _Backend) -> dict[str, int]:
        """Hold one indexed document per :data:`_LISTING_TIMES` entry.

        Returns:
            The reported creation time of every document, keyed by identifier.
        """
        knowledge_base_backend.agent.documents.extend(
            {
                "knowledgeBaseId": _KB_ID,
                "dataSourceId": _DATA_SOURCE_ID,
                "identifier": {
                    "dataSourceType": "CUSTOM",
                    "custom": {"id": f"file-{letter * 32}"},
                },
                "status": "INDEXED",
                "updatedAt": _MOMENT.replace(second=offset),
            }
            for letter, offset in zip(_LISTING_LETTERS, _LISTING_TIMES, strict=True)
        )
        return {
            f"file-{letter * 32}": int(_MOMENT.replace(second=offset).timestamp())
            for letter, offset in zip(_LISTING_LETTERS, _LISTING_TIMES, strict=True)
        }

    @pytest.mark.parametrize("order", ["asc", "desc"])
    def test_pages_concatenate_in_created_at_order(
        self, app_client: TestClient, seeded_documents: dict[str, int], order: str
    ) -> None:
        """Walking every page with the ``after`` cursor yields one ``created_at`` sequence.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/listFiles
             stdapi/vector_stores/knowledge_base.py:list_documents
        """
        documents = _walk_listing(
            app_client, f"/v1/vector_stores/{_STORE_ID}/files", order=order, limit=2
        )
        assert {entry["id"] for entry in documents} == set(seeded_documents)
        created = [entry["created_at"] for entry in documents]
        assert created == sorted(created, reverse=order == "desc"), (
            f"order={order} must hold across pages, got {created}"
        )

    def test_created_at_is_the_only_time_the_service_reports(
        self, app_client: TestClient, seeded_documents: dict[str, int]
    ) -> None:
        """Each document reports the instant the knowledge base last updated it.

        The service reports no separate creation time for a document, so this is
        the quantity the listing is ordered on rather than a second one invented
        for it.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_KnowledgeBaseDocumentDetail.html
             stdapi/vector_stores/knowledge_base.py:_document_record
        """
        documents = _walk_listing(
            app_client, f"/v1/vector_stores/{_STORE_ID}/files", order="asc", limit=2
        )
        assert {
            entry["id"]: entry["created_at"] for entry in documents
        } == seeded_documents


@pytest.mark.local
class TestDataSourceResolution:
    """Which data source the documents of a store belong to."""

    def test_the_configured_data_source_is_used_as_it_stands(
        self,
        app_client: TestClient,
        knowledge_base_backend: _Backend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Naming it in the setting spares the lookup entirely.

        Ref: stdapi/vector_stores/knowledge_base.py:_data_source_id
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_knowledge_base_ids", [f"{_KB_ID}/DSNAMED0001"]
        )
        app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files", json={"file_id": _FILE_ID}
        )
        assert "ListDataSources" not in knowledge_base_backend.agent.operations()
        assert (
            knowledge_base_backend.agent.arguments_of("IngestKnowledgeBaseDocuments")[
                "dataSourceId"
            ]
            == "DSNAMED0001"
        )

    def test_a_single_data_source_is_resolved(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """With one data source there is nothing to configure.

        Ref: stdapi/vector_stores/knowledge_base.py:_data_source_id
        """
        app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files", json={"file_id": _FILE_ID}
        )
        assert "ListDataSources" in knowledge_base_backend.agent.operations()

    def test_an_ambiguous_data_source_is_reported_to_the_operator(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """Several data sources need one to be named, and nothing is guessed.

        Ref: stdapi/vector_stores/knowledge_base.py:_data_source_id
        """
        knowledge_base_backend.agent.data_sources = ["DS00000001", "DS00000002"]
        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files", json={"file_id": _FILE_ID}
        )
        assert response.status_code == 503
        # The caller reads the generic message; the operator reads the log.
        assert "DS00000001" not in response.text
        assert "IngestKnowledgeBaseDocuments" not in (
            knowledge_base_backend.agent.operations()
        )


@pytest.mark.local
class TestForeignDataSource:
    """A store whose corpus is kept in sync elsewhere, and takes no attachment."""

    def test_attaching_is_refused_in_this_apis_own_words(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """The store's own refusal reaches the caller, not the service's.

        A data source that syncs its corpus takes no file attached here, and the
        service says so in terms of its own concepts. The caller chose neither
        the store nor where its corpus comes from, so what they read is what the
        store cannot do and where to go instead.

        Ref: AGENTS.md, "Never Leak Internals"
             stdapi/vector_stores/knowledge_base.py:_addressed_the_wrong_kind
        """
        knowledge_base_backend.agent.data_source_types = {_DATA_SOURCE_ID: "S3"}

        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files", json={"file_id": _FILE_ID}
        )

        assert response.status_code == 400
        message = _error_of(response)["message"]
        assert "managed outside this server" in message
        assert "files cannot be attached to it" in message
        # Nothing of the service's own vocabulary, nor of the deployment's.
        assert "dataSourceType" not in message
        assert "data source" not in message
        assert _DATA_SOURCE_ID not in response.text
        assert _KB_ID not in response.text

    def test_the_operator_reads_which_data_source_cannot_take_the_document(
        self, request_log: dict[str, Any]
    ) -> None:
        """The real cause is named for the operator, who is the one who can fix it.

        Ref: AGENTS.md, "A feature the deployment cannot run has two audiences"
             stdapi/vector_stores/knowledge_base.py:_report_wrong_kind
        """
        knowledge_base._report_wrong_kind(_KB_ID, _DATA_SOURCE_ID)  # noqa: SLF001

        reported = str(request_log)
        assert _KB_ID in reported
        assert _DATA_SOURCE_ID in reported
        assert "aws_bedrock_knowledge_base_ids" in reported

    def test_reading_a_file_such_a_store_could_never_hold_is_a_404(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """A file identifier the store cannot address names no file of it.

        Ref: stdapi/vector_stores/knowledge_base.py:KnowledgeBaseIndex.read_document
        """
        knowledge_base_backend.agent.data_source_types = {_DATA_SOURCE_ID: "S3"}

        response = app_client.get(f"/v1/vector_stores/{_STORE_ID}/files/{_FILE_ID}")

        assert response.status_code == 404
        assert _error_of(response)["message"] == f"No file found with id '{_FILE_ID}'."
        assert "dataSourceType" not in response.text


@pytest.mark.local
class TestDocumentsOfAnotherDataSource:
    """Passages of the corpus behind the store, which search returns and read resolves."""

    @staticmethod
    def _synced(backend: _Backend, monkeypatch: pytest.MonkeyPatch) -> None:
        """Serve a managed store whose corpus is synced by a second data source."""
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_knowledge_base_ids", [f"{_KB_ID}/{_DATA_SOURCE_ID}"]
        )
        backend.agent.kind = "MANAGED"
        backend.agent.data_sources = [_DATA_SOURCE_ID, _OTHER_DATA_SOURCE_ID]
        backend.agent.data_source_types = {
            _DATA_SOURCE_ID: "CUSTOM",
            _OTHER_DATA_SOURCE_ID: "S3",
        }
        backend.agent.documents.append(
            {
                "knowledgeBaseId": _KB_ID,
                "dataSourceId": _OTHER_DATA_SOURCE_ID,
                "identifier": {"dataSourceType": "S3", "s3": {"uri": _SYNCED_URI}},
                "status": "INDEXED",
                "updatedAt": _MOMENT,
            }
        )
        # As the managed generation reports a synced object: a custom location
        # holding the object URI, and its data source named in the metadata.
        backend.runtime.results = [
            {
                "content": {"text": "The refund window is 30 days."},
                "documentId": _SYNCED_URI,
                "location": {
                    "type": "CUSTOM",
                    "customDocumentLocation": {"id": _SYNCED_URI},
                },
                "metadata": {
                    "_source_uri": _SYNCED_URI,
                    "_data_source_id": _OTHER_DATA_SOURCE_ID,
                    "_data_source_type": "CUSTOM",
                },
                "score": 0.27,
            }
        ]

    def test_a_retrieved_passage_reads_back_as_the_file_it_came_from(
        self,
        app_client: TestClient,
        knowledge_base_backend: _Backend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Search and read round trip across the data sources of one store.

        A passage comes back under an opaque identifier, and that identifier is
        the one the per-file route resolves — including when the document
        belongs to a data source other than the one files are attached to.

        Ref: stdapi/vector_stores/knowledge_base.py:document_file_id
             stdapi/vector_stores/knowledge_base.py:KnowledgeBaseIndex.read_document
        """
        self._synced(knowledge_base_backend, monkeypatch)

        found = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/search", json={"query": "refund window"}
        ).json()["data"]
        file_id = found[0]["file_id"]
        read = app_client.get(f"/v1/vector_stores/{_STORE_ID}/files/{file_id}")

        assert file_id.startswith("kbdoc_")
        assert read.status_code == 200
        assert read.json()["id"] == file_id
        assert read.json()["status"] == "completed"
        # Read where the document is, not where files are attached.
        assert (
            knowledge_base_backend.agent.arguments_of("GetKnowledgeBaseDocuments")[
                "dataSourceId"
            ]
            == _OTHER_DATA_SOURCE_ID
        )

    def test_such_a_document_cannot_be_removed_through_this_api(
        self,
        app_client: TestClient,
        knowledge_base_backend: _Backend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A document of the corpus is readable, and never deletable.

        Removing it would delete part of a corpus this API did not put there and
        does not maintain, so the store refuses it rather than obeying.

        Ref: stdapi/vector_stores/knowledge_base.py:KnowledgeBaseIndex.delete_document
        """
        self._synced(knowledge_base_backend, monkeypatch)
        file_id = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/search", json={"query": "refund window"}
        ).json()["data"][0]["file_id"]

        response = app_client.delete(f"/v1/vector_stores/{_STORE_ID}/files/{file_id}")

        assert response.status_code == 400
        assert "managed outside this server" in _error_of(response)["message"]
        assert "DeleteKnowledgeBaseDocuments" not in (
            knowledge_base_backend.agent.operations()
        )
        assert knowledge_base_backend.agent.documents


@pytest.mark.local
class TestSearch:
    """Searching a knowledge base store, and what its score means."""

    def test_a_search_retrieves_and_reports_the_raw_score(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """The relevance the store measured is reported exactly as measured.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Retrieve.html
        """
        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/search",
            json={"query": "refund window", "max_num_results": 5},
        )
        assert response.status_code == 200
        page = response.json()
        assert page["search_query"] == ["refund window"]
        result = page["data"][0]
        assert result["file_id"] == _FILE_ID
        assert result["filename"] == "handbook.txt"
        assert result["content"] == [
            {"type": "text", "text": "The refund window is 30 days."}
        ]
        # Never rescaled into a similarity it is not.
        assert result["score"] == 12.5
        # The reserved and service-owned keys stay out of the attributes.
        assert result["attributes"] == {"team": "support", "revision": 3.0}

        call = knowledge_base_backend.runtime.calls[0]
        assert call["knowledgeBaseId"] == _KB_ID
        assert call["retrievalQuery"] == {"text": "refund window"}
        assert call["retrievalConfiguration"] == {
            "vectorSearchConfiguration": {"numberOfResults": 5}
        }

    def test_a_managed_store_is_searched_through_its_own_configuration(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """The two generations take the retrieval configuration under their own key.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Retrieve.html
        """
        knowledge_base_backend.agent.kind = "MANAGED"
        app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/search", json={"query": "refund window"}
        )
        assert (
            "managedSearchConfiguration"
            in (knowledge_base_backend.runtime.calls[0]["retrievalConfiguration"])
        )

    def test_the_managed_generation_names_its_own_metadata_differently(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """A managed store's service metadata is not answered as caller attributes.

        That generation writes its own keys under a leading underscore rather
        than the ``x-amz-bedrock-kb-`` prefix the other one uses, and reports
        the document's location under ``_source_uri``: neither is the caller's,
        and the name is read from the latter.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_RetrievalResult.html
        """
        knowledge_base_backend.agent.kind = "MANAGED"
        knowledge_base_backend.runtime.results = [
            {
                "content": {"text": "The refund window is 30 days."},
                "location": {
                    "type": "CUSTOM",
                    "customDocumentLocation": {"id": "s3://corpus/handbook.txt"},
                },
                "score": 12.5,
                "metadata": {
                    "_source_uri": "s3://corpus/handbook.txt",
                    "_chunk_id": "chunk-1",
                    "_data_source_type": "CUSTOM",
                    "team": "support",
                },
            }
        ]

        page = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/search", json={"query": "refund window"}
        ).json()

        result = page["data"][0]
        assert result["attributes"] == {"team": "support"}
        assert result["filename"] == "handbook.txt"

    def test_a_filter_reaches_the_retrieval(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """Filters are expressed by the store, over its own metadata keys.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/search
        """
        app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/search",
            json={
                "query": "refund window",
                "filters": {"key": "team", "type": "eq", "value": "support"},
            },
        )
        configuration = knowledge_base_backend.runtime.calls[0][
            "retrievalConfiguration"
        ]["vectorSearchConfiguration"]
        assert configuration["filter"] == {
            "equals": {"key": "team", "value": "support"}
        }

    def test_several_queries_are_merged_and_deduplicated(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """One passage matched by two queries is reported once.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/search
        """
        page = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/search",
            json={"query": ["refund window", "returns policy"]},
        ).json()
        assert len(knowledge_base_backend.runtime.calls) == 2
        assert len(page["data"]) == 1

    def test_a_score_threshold_is_refused(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """A threshold over a score with no range would be meaningless.

        Ref: stdapi/vector_stores/engine.py:search
        """
        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/search",
            json={"query": "refunds", "ranking_options": {"score_threshold": 0.5}},
        )
        assert response.status_code == 400
        assert "score_threshold" in _error_of(response)["message"]
        assert not knowledge_base_backend.runtime.calls

    def test_an_over_length_query_is_refused_rather_than_truncated(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """The limit is named, and the query is never cut down to fit it.

        Ref: https://docs.aws.amazon.com/general/latest/gr/bedrock.html
        """
        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/search", json={"query": "a" * 1001}
        )
        assert response.status_code == 400
        assert "1000 characters" in _error_of(response)["message"]
        assert not knowledge_base_backend.runtime.calls

        # The managed generation accepts ten times as much.
        knowledge_base_backend.agent.kind = "MANAGED"
        accepted = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/search", json={"query": "a" * 1001}
        )
        assert accepted.status_code == 200


@pytest.mark.local
class TestSearchIsBilled:
    """What a search of a knowledge base store reports to cost tracking.

    Only the managed generation is billed per retrieval call: AWS publishes a
    flat rate for its standard retrieval, parsing, embedding and reranking
    included. The other generation's search costs whatever its embedding model
    and its own vector store charge, neither of which passes through this
    server, so recording anything for it would be an invented number.

    Ref: https://aws.amazon.com/bedrock/pricing/
         stdapi/usage.py:record_knowledge_base_usage
    """

    def test_a_managed_store_bills_one_search_unit_per_retrieval(
        self,
        app_client: TestClient,
        knowledge_base_backend: _Backend,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """Every retrieval a search issues is one billed unit, priced per call.

        Ref: https://aws.amazon.com/bedrock/pricing/
        """
        knowledge_base_backend.agent.kind = "MANAGED"
        capfd.readouterr()

        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/search",
            json={"query": ["refund window", "returns policy"]},
        )

        assert response.status_code == 200
        (entry,) = logged_usage_entries(
            capfd.readouterr().out, service="bedrock-runtime"
        )
        assert entry["model"] == KNOWLEDGE_BASE_MODEL
        assert entry["search_units"] == 2

    def test_a_self_managed_store_reports_no_retrieval_charge(
        self,
        app_client: TestClient,
        knowledge_base_backend: _Backend,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """The other generation's retrieval carries no rate this server can report.

        Ref: https://aws.amazon.com/bedrock/pricing/
             stdapi/vector_stores/knowledge_base.py:KnowledgeBaseIndex.query_text
        """
        capfd.readouterr()

        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/search", json={"query": "refund window"}
        )

        assert response.status_code == 200
        assert not logged_usage_entries(
            capfd.readouterr().out, service="bedrock-runtime"
        )


@pytest.mark.local
class TestRefusals:
    """What a store held elsewhere cannot express, refused rather than faked."""

    def test_a_chunking_strategy_is_refused(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """The store cuts its own passages, so a chunk size cannot be honoured.

        Ref: stdapi/vector_stores/engine.py:check_chunking_strategy
        """
        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files",
            json={
                "file_id": _FILE_ID,
                "chunking_strategy": {
                    "type": "static",
                    "static": {
                        "max_chunk_size_tokens": 400,
                        "chunk_overlap_tokens": 100,
                    },
                },
            },
        )
        assert response.status_code == 400
        assert "passage boundaries" in _error_of(response)["message"]
        assert "IngestKnowledgeBaseDocuments" not in (
            knowledge_base_backend.agent.operations()
        )

    def test_rewriting_the_attributes_of_a_file_is_refused(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """Attributes are set when the file is attached, and read no other way.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/updateAttributes
        """
        app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files", json={"file_id": _FILE_ID}
        )
        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files/{_FILE_ID}",
            json={"attributes": {"team": "billing"}},
        )
        assert response.status_code == 400
        assert "attach the file again" in _error_of(response)["message"].lower()
        assert knowledge_base_backend.agent.documents

    def test_listing_the_passages_of_a_file_is_refused(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """The store addresses documents, not the passages it cut them into.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-files/getContent
        """
        app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files", json={"file_id": _FILE_ID}
        )
        response = app_client.get(
            f"/v1/vector_stores/{_STORE_ID}/files/{_FILE_ID}/content"
        )
        assert response.status_code == 400
        assert "cannot be listed" in _error_of(response)["message"]

    def test_file_batches_are_refused(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """A batch is progress bookkeeping this server would have to own.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-file-batches
        """
        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/file_batches", json={"file_ids": [_FILE_ID]}
        )
        assert response.status_code == 400
        assert "one at a time" in _error_of(response)["message"]
        assert not knowledge_base_backend.agent.documents

    def test_reading_a_batch_is_refused(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """No batch can exist on the store, so none is reported as in progress.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores-file-batches/getBatch
        """
        del knowledge_base_backend
        batch_id = f"vsfb_{'0' * 26}"
        response = app_client.get(
            f"/v1/vector_stores/{_STORE_ID}/file_batches/{batch_id}"
        )
        assert response.status_code == 400
        assert "file batches" in _error_of(response)["message"]

    def test_the_reserved_attribute_key_is_refused(
        self, app_client: TestClient, knowledge_base_backend: _Backend
    ) -> None:
        """The key the file name is stored under cannot be a caller's.

        Ref: stdapi/vector_stores/knowledge_base.py:KnowledgeBaseIndex.check_attributes
        """
        response = app_client.post(
            f"/v1/vector_stores/{_STORE_ID}/files",
            json={
                "file_id": _FILE_ID,
                "attributes": {"stdapi-filename": "spoofed.txt"},
            },
        )
        assert response.status_code == 400
        assert "stdapi-filename" in _error_of(response)["message"]
        assert not knowledge_base_backend.agent.documents
