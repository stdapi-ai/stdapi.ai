"""Tests for the Cohere-compatible /v1/rerank and /v2/rerank routes (unit and live)."""

from os import getenv
from typing import Any

import cohere
import pytest
from starlette.testclient import TestClient

from stdapi.api_errors import UnsupportedModelError
from stdapi.models import RERANKING_MODALITY, ModelDetails
from stdapi.models.rerank import RerankedDocument, RerankResponse
from stdapi.routes import cohere_rerank, cohere_rerank_v1
from tests.test_models_rerank import RERANK_MODELS


class _StubRerankModel:
    """Stub backend recording the rerank call and returning a fixed response."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None,
        extra_params: dict[str, Any],
    ) -> RerankResponse:
        """Record the call and return one reranked document."""
        self.calls.append(
            {
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "extra_params": extra_params,
            }
        )
        return RerankResponse(
            results=[RerankedDocument(index=1, relevance_score=0.98)], search_units=1
        )


@pytest.fixture
def client(api_key: str) -> TestClient:
    """Test client without lifespan (no AWS startup), pre-authenticated."""
    from stdapi.main import app  # noqa: PLC0415

    return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})


@pytest.fixture
def rerank_backend(monkeypatch: pytest.MonkeyPatch) -> _StubRerankModel:
    """Stub model validation and the rerank backend."""

    async def _validate_model(
        model_id: str, *_args: object, **_kwargs: object
    ) -> ModelDetails:
        if model_id == "unknown-model":
            raise UnsupportedModelError(model_id)
        return ModelDetails(
            id=model_id,
            name=model_id,
            provider="Vendor",
            input_modalities=["TEXT"],
            output_modalities=[RERANKING_MODALITY],
            regions=["us-east-1"],
        )

    stub = _StubRerankModel()
    for module in (cohere_rerank, cohere_rerank_v1):
        monkeypatch.setattr(module, "validate_model", _validate_model)
        monkeypatch.setattr(module, "get_rerank_model", lambda _model_id: stub)
    return stub


@pytest.mark.local
class TestCohereRerankRoute:
    """POST /cohere/v2/rerank: response shape and error envelopes."""

    def test_rerank_success(
        self, client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """A valid request returns the Cohere v2 response shape."""
        response = client.post(
            "/cohere/v2/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "capital of the USA",
                "documents": ["Carson City", "Washington, D.C."],
                "top_n": 1,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"]
        assert body["results"] == [{"index": 1, "relevance_score": 0.98}]
        assert body["meta"] == {
            "api_version": {"version": "2"},
            "billed_units": {"search_units": 1},
        }
        assert response.headers["x-request-id"] == body["id"]
        (call,) = rerank_backend.calls
        assert call["query"] == "capital of the USA"
        assert call["documents"] == ["Carson City", "Washington, D.C."]
        assert call["top_n"] == 1

    def test_max_tokens_per_doc_forwarded(
        self, client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """max_tokens_per_doc is passed to the backend as an extra parameter."""
        response = client.post(
            "/cohere/v2/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["a"],
                "max_tokens_per_doc": 512,
            },
        )
        assert response.status_code == 200, response.text
        (call,) = rerank_backend.calls
        assert call["extra_params"]["max_tokens_per_doc"] == 512

    @pytest.mark.parametrize("priority", [5, 1000])
    def test_priority_is_ignored(
        self, client: TestClient, rerank_backend: _StubRerankModel, priority: int
    ) -> None:
        """The Cohere priority hint is accepted (no upper bound) but not forwarded to AWS."""
        response = client.post(
            "/cohere/v2/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["a"],
                "priority": priority,
            },
        )
        assert response.status_code == 200, response.text
        (call,) = rerank_backend.calls
        assert "priority" not in call["extra_params"]

    @pytest.mark.parametrize("return_documents", [True, False])
    def test_return_documents_is_ignored(
        self,
        client: TestClient,
        rerank_backend: _StubRerankModel,
        return_documents: bool,
    ) -> None:
        """The v1-only return_documents field is accepted but not forwarded to AWS."""
        response = client.post(
            "/cohere/v2/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["a"],
                "return_documents": return_documents,
            },
        )
        assert response.status_code == 200, response.text
        assert "document" not in response.json()["results"][0]
        (call,) = rerank_backend.calls
        assert "return_documents" not in call["extra_params"]

    def test_unknown_model_returns_cohere_error_envelope(
        self, client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """Model errors use the Cohere `{"message": ...}` envelope."""
        response = client.post(
            "/cohere/v2/rerank",
            json={"model": "unknown-model", "query": "q", "documents": ["a"]},
        )
        assert response.status_code == 404
        body = response.json()
        assert set(body) == {"message"}
        assert "unknown-model" in body["message"]

    def test_resolved_model_not_rerank_capable_returns_cohere_error_envelope(
        self,
        client: TestClient,
        rerank_backend: _StubRerankModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A model that resolves but has no rerank backend uses the Cohere envelope."""

        def _get_rerank_model(model_id: str) -> _StubRerankModel:
            raise UnsupportedModelError(model_id)

        monkeypatch.setattr(cohere_rerank, "get_rerank_model", _get_rerank_model)

        response = client.post(
            "/cohere/v2/rerank",
            json={
                "model": "anthropic.claude-3-5-haiku-20241022-v1:0",
                "query": "q",
                "documents": ["a"],
            },
        )

        assert response.status_code == 404
        body = response.json()
        assert set(body) == {"message"}
        assert "anthropic.claude-3-5-haiku-20241022-v1:0" in body["message"]
        assert not rerank_backend.calls

    def test_missing_query_is_rejected(
        self, client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """A request without a query fails validation with the Cohere envelope."""
        response = client.post(
            "/cohere/v2/rerank",
            json={"model": "cohere.rerank-v3-5:0", "documents": ["a"]},
        )
        assert response.status_code == 400
        assert "message" in response.json()
        assert not rerank_backend.calls

    def test_empty_documents_are_rejected(
        self, client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """An empty documents list fails validation."""
        response = client.post(
            "/cohere/v2/rerank",
            json={"model": "cohere.rerank-v3-5:0", "query": "q", "documents": []},
        )
        assert response.status_code == 400
        assert "message" in response.json()
        assert not rerank_backend.calls


@pytest.mark.local
class TestCohereRerankV1Route:
    """POST /cohere/v1/rerank: legacy v1 response shape and error envelopes."""

    def test_rerank_success_with_string_documents(
        self, client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """A valid request returns the Cohere v1 response shape."""
        response = client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "capital of the USA",
                "documents": ["Carson City", "Washington, D.C."],
                "top_n": 1,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"]
        assert response.headers["x-request-id"] == body["id"]
        assert body == {
            "id": body["id"],
            "results": [{"index": 1, "relevance_score": 0.98}],
            "meta": {
                "api_version": {"version": "1"},
                "billed_units": {"search_units": 1},
            },
        }
        (call,) = rerank_backend.calls
        assert call["query"] == "capital of the USA"
        assert call["documents"] == ["Carson City", "Washington, D.C."]
        assert call["top_n"] == 1

    def test_document_objects_are_normalized_to_text(
        self, client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """Document objects are reduced to their `text` field for the backend."""
        response = client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": [{"text": "Carson City"}, {"text": "Washington, D.C."}],
            },
        )
        assert response.status_code == 200, response.text
        (call,) = rerank_backend.calls
        assert call["documents"] == ["Carson City", "Washington, D.C."]

    def test_return_documents_echoes_documents(
        self, client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """return_documents=true echoes the input text back in each result."""
        response = client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["Carson City", {"text": "Washington, D.C."}],
                "return_documents": True,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["results"] == [
            {
                "document": {"text": "Washington, D.C."},
                "index": 1,
                "relevance_score": 0.98,
            }
        ]

    def test_documents_are_not_echoed_by_default(
        self, client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """Without return_documents, results carry only index and score."""
        response = client.post(
            "/cohere/v1/rerank",
            json={"model": "cohere.rerank-v3-5:0", "query": "q", "documents": ["a"]},
        )
        assert response.status_code == 200, response.text
        assert response.json()["results"] == [{"index": 1, "relevance_score": 0.98}]

    def test_default_rank_fields_value_is_accepted(
        self, client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """rank_fields=["text"] matches the default behavior and is accepted."""
        response = client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["a"],
                "rank_fields": ["text"],
            },
        )
        assert response.status_code == 200, response.text
        assert rerank_backend.calls

    @pytest.mark.parametrize("rank_fields", [["title"], ["title", "text"], []])
    def test_custom_rank_fields_are_rejected(
        self,
        client: TestClient,
        rerank_backend: _StubRerankModel,
        rank_fields: list[str],
    ) -> None:
        """rank_fields other than ["text"] are rejected with the Cohere envelope."""
        response = client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["a"],
                "rank_fields": rank_fields,
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body) == {"message"}
        assert "rank_fields" in body["message"]
        assert not rerank_backend.calls

    def test_max_chunks_per_doc_is_rejected(
        self, client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """max_chunks_per_doc has no Bedrock equivalent and is rejected."""
        response = client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["a"],
                "max_chunks_per_doc": 10,
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body) == {"message"}
        assert "max_chunks_per_doc" in body["message"]
        assert not rerank_backend.calls

    def test_unknown_model_returns_cohere_error_envelope(
        self, client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """Model errors use the Cohere `{"message": ...}` envelope."""
        response = client.post(
            "/cohere/v1/rerank",
            json={"model": "unknown-model", "query": "q", "documents": ["a"]},
        )
        assert response.status_code == 404
        body = response.json()
        assert set(body) == {"message"}
        assert "unknown-model" in body["message"]
        assert not rerank_backend.calls

    def test_empty_documents_are_rejected(
        self, client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """An empty documents list fails validation."""
        response = client.post(
            "/cohere/v1/rerank",
            json={"model": "cohere.rerank-v3-5:0", "query": "q", "documents": []},
        )
        assert response.status_code == 400
        assert "message" in response.json()
        assert not rerank_backend.calls


#: Documents with exactly one relevant answer (index 1) for the live tests.
_LIVE_DOCUMENTS = [
    "Carson City is the capital city of Nevada.",
    "Washington, D.C. is the capital of the United States.",
    "Capital punishment has existed in the United States since colonial times.",
]


@pytest.fixture(params=RERANK_MODELS)
def live_rerank_model(
    request: pytest.FixtureRequest, use_official_api: bool, cohere_rerank_model: str
) -> str:
    """Every Bedrock rerank model, or the official Cohere model once."""
    if use_official_api:
        if request.param != RERANK_MODELS[0]:
            pytest.skip("a single model covers the official Cohere API")
        return cohere_rerank_model
    return str(request.param)


class TestCohereRerankIntegration:
    """Live /v2/rerank calls through the official Cohere SDK."""

    def test_rerank_orders_documents(
        self, cohere_client: cohere.ClientV2, live_rerank_model: str
    ) -> None:
        """The most relevant document ranks first and top_n limits results."""
        response = cohere_client.rerank(
            model=live_rerank_model,
            query="What is the capital of the United States?",
            documents=_LIVE_DOCUMENTS,
            top_n=2,
        )
        assert response.id
        assert len(response.results) == 2
        assert response.results[0].index == 1
        scores = [result.relevance_score for result in response.results]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= score <= 1.0 for score in scores)
        assert response.meta is not None
        assert response.meta.api_version is not None
        assert response.meta.api_version.version == "2"
        assert response.meta.billed_units is not None
        assert response.meta.billed_units.search_units == 1

    def test_unsupported_extra_field_returns_clean_error(
        self, cohere_client: cohere.ClientV2, use_official_api: bool
    ) -> None:
        """AWS's rejection of an unsupported model field surfaces as a Cohere 400."""
        if use_official_api:
            pytest.skip("Bedrock-specific model")
        with pytest.raises(cohere.BadRequestError):
            cohere_client.rerank(
                model="amazon.rerank-v1:0",
                query="q",
                documents=["a"],
                max_tokens_per_doc=512,
            )

    @pytest.mark.expensive
    def test_over_one_hundred_documents_bill_two_units(
        self, cohere_client: cohere.ClientV2, live_rerank_model: str
    ) -> None:
        """A query with more than 100 documents is billed two search units."""
        response = cohere_client.rerank(
            model=live_rerank_model,
            query="the capital of the United States",
            documents=[f"Fact {i}: water is wet." for i in range(120)],
            top_n=1,
        )
        assert response.meta is not None
        assert response.meta.billed_units is not None
        assert response.meta.billed_units.search_units == 2


@pytest.fixture(scope="session")
def cohere_client_v1(
    request: pytest.FixtureRequest, test_client: TestClient | None, api_key: str
) -> cohere.Client:
    """Create a Cohere v1 client for either local or official API testing."""
    if test_client:
        return cohere.Client(
            api_key=api_key,
            base_url="http://testserver/cohere",
            httpx_client=test_client,
        )
    if request.config.getoption("--use-official-api"):
        if not getenv("CO_API_KEY"):
            pytest.skip("CO_API_KEY is required to test the official Cohere API")
        return cohere.Client()
    return cohere.Client(
        api_key=getenv("OPENAI_API_KEY", ""),
        base_url=f"{request.config.getoption('--server-url').rstrip('/')}/cohere",
    )


class TestCohereRerankV1Integration:
    """Live /v1/rerank calls through the official Cohere SDK."""

    def test_rerank_orders_documents(
        self, cohere_client_v1: cohere.Client, live_rerank_model: str
    ) -> None:
        """The most relevant document ranks first and top_n limits results."""
        response = cohere_client_v1.rerank(
            model=live_rerank_model,
            query="What is the capital of the United States?",
            documents=_LIVE_DOCUMENTS,
            top_n=2,
        )
        assert response.id
        assert len(response.results) == 2
        assert response.results[0].index == 1
        scores = [result.relevance_score for result in response.results]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= score <= 1.0 for score in scores)
        assert response.meta is not None
        assert response.meta.api_version is not None
        assert response.meta.api_version.version == "1"
        assert response.meta.billed_units is not None
        assert response.meta.billed_units.search_units == 1

    def test_return_documents_echoes_documents(
        self, cohere_client_v1: cohere.Client, live_rerank_model: str
    ) -> None:
        """return_documents=true echoes the document text back in each result."""
        response = cohere_client_v1.rerank(
            model=live_rerank_model,
            query="What is the capital of the United States?",
            documents=[{"text": text} for text in _LIVE_DOCUMENTS],
            top_n=1,
            return_documents=True,
        )
        assert len(response.results) == 1
        result = response.results[0]
        assert result.document is not None
        assert result.document.text == _LIVE_DOCUMENTS[result.index]
