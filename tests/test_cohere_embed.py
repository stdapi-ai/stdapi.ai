"""Tests for the Cohere-compatible /v1/embed and /v2/embed routes (unit and live)."""

from os import getenv
from typing import TYPE_CHECKING, Any

import cohere
import httpx
import pytest
from starlette.testclient import TestClient

from stdapi.api_errors import UnsupportedModelError
from stdapi.config import SETTINGS
from stdapi.input_file import InputFile
from stdapi.models import ModelDetails
from stdapi.models.embedding import EmbeddingResponse
from stdapi.routes import cohere_embed, cohere_embed_v1

if TYPE_CHECKING:
    from collections.abc import Generator

    from cohere.types import EmbedByTypeResponse

#: Model aliases resolved by the stubbed ``validate_model``.
_MODEL_ALIASES = {"embed-multilingual": "cohere.embed-multilingual-v3"}


class _StubEmbeddingModel:
    """Stub backend recording the embed call and returning fixed vectors."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def embed_text(
        self, inputs: list[Any], dimensions: int | None, extra_params: dict[str, Any]
    ) -> EmbeddingResponse:
        """Record the call and return one vector per input."""
        self.calls.append(
            {"inputs": inputs, "dimensions": dimensions, "extra_params": extra_params}
        )
        return EmbeddingResponse(
            embeddings=[[0.1, 0.2]] * len(inputs), prompt_tokens=7, total_tokens=7
        )


@pytest.fixture
def client(api_key: str) -> TestClient:
    """Test client without lifespan (no AWS startup), pre-authenticated."""
    from stdapi.main import app  # noqa: PLC0415

    return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})


@pytest.fixture
def embed_backend(monkeypatch: pytest.MonkeyPatch) -> _StubEmbeddingModel:
    """Stub model validation and the embedding backend."""

    async def _validate_model(model_id: str, modality: str) -> ModelDetails:
        assert modality == "EMBEDDING"
        if model_id == "unknown-model":
            raise UnsupportedModelError(model_id)
        resolved_id = _MODEL_ALIASES.get(model_id, model_id)
        return ModelDetails(
            id=resolved_id,
            name=resolved_id,
            provider="Vendor",
            input_modalities=["TEXT"],
            output_modalities=["EMBEDDING"],
            regions=["us-east-1"],
        )

    stub = _StubEmbeddingModel()
    for module in (cohere_embed, cohere_embed_v1):
        monkeypatch.setattr(module, "validate_model", _validate_model)
        monkeypatch.setattr(module, "get_embedding_model", lambda _model_id: stub)
    return stub


@pytest.mark.local
class TestCohereEmbedRoute:
    """POST /cohere/v2/embed: response shape and parameter mapping."""

    def test_embed_texts_success(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """A valid request returns the Cohere v2 response shape."""
        response = client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-multilingual-v3",
                "input_type": "search_document",
                "texts": ["hello", "world"],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"]
        assert body["response_type"] == "embeddings_by_type"
        assert body["embeddings"] == {"float": [[0.1, 0.2], [0.1, 0.2]]}
        assert body["texts"] == ["hello", "world"]
        assert body["meta"] == {
            "api_version": {"version": "2"},
            "billed_units": {"input_tokens": 7},
        }
        (call,) = embed_backend.calls
        assert call["inputs"] == ["hello", "world"]
        assert call["extra_params"]["input_type"] == "search_document"

    def test_cohere_params_forwarded_for_cohere_models(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """input_type/truncate/max_tokens and output_dimension are forwarded."""
        response = client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_query",
                "texts": ["q"],
                "truncate": "START",
                "max_tokens": 128,
                "output_dimension": 512,
            },
        )
        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        assert call["dimensions"] == 512
        assert call["extra_params"] == {
            "input_type": "search_query",
            "truncate": "START",
            "max_tokens": 128,
        }

    def test_cohere_params_dropped_for_other_providers(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Cohere-specific fields are not forwarded to non-Cohere models."""
        response = client.post(
            "/cohere/v2/embed",
            json={
                "model": "amazon.titan-embed-text-v2:0",
                "input_type": "search_document",
                "texts": ["hello"],
            },
        )
        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        assert call["extra_params"] == {}

    def test_image_data_uri_input(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Image data URIs are parsed into file inputs and billed as images."""
        response = client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "image",
                "images": ["data:image/png;base64,aGVsbG8="],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["meta"]["billed_units"] == {"input_tokens": 7, "images": 1}
        assert "texts" not in body
        (call,) = embed_backend.calls
        assert len(call["inputs"]) == 1
        assert isinstance(call["inputs"][0], InputFile)

    def test_mixed_texts_and_images_input(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Texts and images in one request are concatenated, texts first."""
        response = client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "texts": ["hello", "world"],
                "images": ["data:image/png;base64,aGVsbG8="],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["texts"] == ["hello", "world"]
        assert body["meta"]["billed_units"] == {"input_tokens": 7, "images": 1}
        (call,) = embed_backend.calls
        assert len(call["inputs"]) == 3
        assert call["inputs"][:2] == ["hello", "world"]
        assert isinstance(call["inputs"][2], InputFile)

    def test_no_input_is_rejected(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """A request without texts or images fails validation."""
        response = client.post(
            "/cohere/v2/embed",
            json={"model": "cohere.embed-v4:0", "input_type": "search_document"},
        )
        assert response.status_code == 400
        assert "message" in response.json()
        assert not embed_backend.calls

    def test_non_float_embedding_types_are_rejected(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """embedding_types other than float are rejected."""
        response = client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "texts": ["a"],
                "embedding_types": ["int8"],
            },
        )
        assert response.status_code == 400
        assert not embed_backend.calls

    def test_fused_inputs_are_rejected(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """The v2 fused multimodal `inputs` field is rejected with a clear error."""
        response = client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "inputs": [{"content": [{"type": "text", "text": "a"}]}],
            },
        )
        assert response.status_code == 400
        assert "inputs" in response.json()["message"]
        assert not embed_backend.calls

    def test_priority_is_ignored(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """The Cohere priority hint is accepted but not forwarded to AWS."""
        response = client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "texts": ["a"],
                "priority": 0,
            },
        )
        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        assert "priority" not in call["extra_params"]

    def test_null_inputs_is_treated_as_absent(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """An explicit `inputs: null` alongside `texts` is accepted and not forwarded."""
        response = client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "texts": ["a"],
                "inputs": None,
            },
        )
        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        assert "inputs" not in call["extra_params"]

    def test_request_params_override_operator_defaults(
        self,
        client: TestClient,
        embed_backend: _StubEmbeddingModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit request values win over `default_model_params` defaults."""
        monkeypatch.setitem(
            SETTINGS.default_model_params,
            "cohere.embed-multilingual-v3",
            {"input_type": "clustering", "truncate": "NONE", "max_tokens": 64},
        )
        response = client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-multilingual-v3",
                "input_type": "search_query",
                "texts": ["q"],
                "truncate": "START",
                "max_tokens": 128,
            },
        )
        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        assert call["extra_params"] == {
            "input_type": "search_query",
            "truncate": "START",
            "max_tokens": 128,
        }

    def test_alias_resolution_drives_cohere_params(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Cohere params follow the resolved model ID, not the requested alias."""
        response = client.post(
            "/cohere/v2/embed",
            json={
                "model": "embed-multilingual",
                "input_type": "search_document",
                "texts": ["hello"],
            },
        )
        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        assert call["extra_params"] == {"input_type": "search_document"}

    def test_unknown_model_returns_cohere_error_envelope(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Model errors use the Cohere `{"message": ...}` envelope."""
        response = client.post(
            "/cohere/v2/embed",
            json={
                "model": "unknown-model",
                "input_type": "search_document",
                "texts": ["a"],
            },
        )
        assert response.status_code == 404
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "unknown-model" in body["message"]
        assert not embed_backend.calls


@pytest.mark.local
class TestCohereEmbedV1Route:
    """POST /cohere/v1/embed: legacy v1 response shapes and error envelopes."""

    def test_embed_texts_floats_shape_by_default(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Without embedding_types the legacy embeddings_floats shape is returned."""
        response = client.post(
            "/cohere/v1/embed",
            json={"model": "cohere.embed-multilingual-v3", "texts": ["hello", "world"]},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"]
        assert response.headers["x-request-id"] == body["id"]
        assert body == {
            "response_type": "embeddings_floats",
            "id": body["id"],
            "embeddings": [[0.1, 0.2], [0.1, 0.2]],
            "texts": ["hello", "world"],
            "meta": {
                "api_version": {"version": "1"},
                "billed_units": {"input_tokens": 7},
            },
        }
        (call,) = embed_backend.calls
        assert call["inputs"] == ["hello", "world"]
        assert call["dimensions"] is None
        assert call["extra_params"] == {}

    def test_embedding_types_float_returns_by_type_shape(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """embedding_types=["float"] switches to the embeddings_by_type shape."""
        response = client.post(
            "/cohere/v1/embed",
            json={
                "model": "cohere.embed-multilingual-v3",
                "texts": ["hello"],
                "embedding_types": ["float"],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["response_type"] == "embeddings_by_type"
        assert body["embeddings"] == {"float": [[0.1, 0.2]]}
        assert body["texts"] == ["hello"]
        assert body["meta"] == {
            "api_version": {"version": "1"},
            "billed_units": {"input_tokens": 7},
        }

    def test_non_float_embedding_types_are_rejected(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """embedding_types other than float are rejected."""
        response = client.post(
            "/cohere/v1/embed",
            json={
                "model": "cohere.embed-multilingual-v3",
                "texts": ["a"],
                "embedding_types": ["int8"],
            },
        )
        assert response.status_code == 400
        assert "message" in response.json()
        assert not embed_backend.calls

    def test_input_type_and_truncate_forwarded_for_cohere_models(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """input_type and truncate are forwarded to Cohere models when provided."""
        response = client.post(
            "/cohere/v1/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_query",
                "texts": ["q"],
                "truncate": "END",
            },
        )
        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        assert call["extra_params"] == {"input_type": "search_query", "truncate": "END"}

    def test_cohere_params_dropped_for_other_providers(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Cohere-specific fields are not forwarded to non-Cohere models."""
        response = client.post(
            "/cohere/v1/embed",
            json={
                "model": "amazon.titan-embed-text-v2:0",
                "input_type": "search_document",
                "texts": ["hello"],
            },
        )
        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        assert call["extra_params"] == {}

    def test_image_data_uri_input(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Image data URIs are parsed into file inputs and billed as images."""
        response = client.post(
            "/cohere/v1/embed",
            json={
                "model": "cohere.embed-v4:0",
                "images": ["data:image/png;base64,aGVsbG8="],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["response_type"] == "embeddings_floats"
        assert body["meta"]["billed_units"] == {"input_tokens": 7, "images": 1}
        assert "texts" not in body
        (call,) = embed_backend.calls
        assert len(call["inputs"]) == 1
        assert isinstance(call["inputs"][0], InputFile)

    def test_no_input_is_rejected(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """A request without texts or images fails with the Cohere envelope."""
        response = client.post(
            "/cohere/v1/embed", json={"model": "cohere.embed-multilingual-v3"}
        )
        assert response.status_code == 400
        assert set(response.json()) == {"message", "id"}
        assert not embed_backend.calls

    def test_fused_inputs_are_rejected(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """The v2-style fused multimodal `inputs` field is rejected with a clear error."""
        response = client.post(
            "/cohere/v1/embed",
            json={
                "model": "cohere.embed-v4:0",
                "inputs": [{"content": [{"type": "text", "text": "a"}]}],
            },
        )
        assert response.status_code == 400
        assert "inputs" in response.json()["message"]
        assert not embed_backend.calls

    def test_null_inputs_is_treated_as_absent(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """An explicit `inputs: null` alongside `texts` is accepted and not forwarded."""
        response = client.post(
            "/cohere/v1/embed",
            json={"model": "cohere.embed-v4:0", "texts": ["a"], "inputs": None},
        )
        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        assert "inputs" not in call["extra_params"]

    def test_unknown_model_returns_cohere_error_envelope(
        self, client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Model errors use the Cohere `{"message": ...}` envelope."""
        response = client.post(
            "/cohere/v1/embed", json={"model": "unknown-model", "texts": ["a"]}
        )
        assert response.status_code == 404
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "unknown-model" in body["message"]
        assert not embed_backend.calls


@pytest.fixture
def live_client(
    request: pytest.FixtureRequest, test_client: TestClient | None, api_key: str
) -> Generator[httpx.Client]:
    """Authenticated client for the local live server or the --server-url target."""
    if test_client is not None:
        test_client.headers["authorization"] = f"Bearer {api_key}"
        yield test_client
        del test_client.headers["authorization"]
    elif server_url := request.config.getoption("--server-url"):
        with httpx.Client(
            base_url=server_url.rstrip("/"),
            headers={"authorization": f"Bearer {getenv('OPENAI_API_KEY', '')}"},
            timeout=60.0,
        ) as client:
            yield client
    else:
        pytest.skip("Cohere-compatible routes are not part of the official API")


class TestCohereEmbedIntegration:
    """Live /v2/embed calls through the official Cohere SDK."""

    @staticmethod
    def _embed(
        cohere_client: cohere.ClientV2,
        model_id: str,
        **params: Any,  # noqa: ANN401
    ) -> EmbedByTypeResponse:
        """Embed one short text with *model_id* and return the response."""
        return cohere_client.embed(
            model=model_id,
            input_type="search_document",
            texts=["The quick brown fox jumps over the lazy dog."],
            embedding_types=["float"],
            **params,
        )

    def test_embed_texts_cohere_model(
        self, cohere_client: cohere.ClientV2, cohere_embed_multilingual_model: str
    ) -> None:
        """A Cohere model returns one float vector per text with the v2 shape."""
        response = self._embed(cohere_client, cohere_embed_multilingual_model)
        assert response.id
        assert response.response_type == "embeddings_by_type"
        assert response.embeddings.float_ is not None
        (vector,) = response.embeddings.float_
        assert len(vector) > 0
        assert all(isinstance(value, float) for value in vector)
        assert response.texts == ["The quick brown fox jumps over the lazy dog."]
        assert response.meta is not None
        assert response.meta.api_version is not None
        assert response.meta.api_version.version == "2"
        assert response.meta.billed_units is not None
        assert (response.meta.billed_units.input_tokens or 0) > 0

    def test_embed_texts_non_cohere_model(
        self,
        cohere_client: cohere.ClientV2,
        embedding_model: str,
        use_official_api: bool,
    ) -> None:
        """The required input_type is accepted (and dropped) for non-Cohere models."""
        if use_official_api:
            pytest.skip("Bedrock-specific model")
        response = self._embed(cohere_client, embedding_model)
        assert response.embeddings.float_ is not None
        (vector,) = response.embeddings.float_
        assert len(vector) > 0

    def test_output_dimension(
        self, cohere_client: cohere.ClientV2, cohere_embed_v4_model: str
    ) -> None:
        """output_dimension controls the returned vector size."""
        response = self._embed(
            cohere_client, cohere_embed_v4_model, output_dimension=512
        )
        assert response.embeddings.float_ is not None
        (vector,) = response.embeddings.float_
        assert len(vector) == 512

    @pytest.mark.expensive
    def test_embed_image(
        self,
        cohere_client: cohere.ClientV2,
        cohere_embed_v4_model: str,
        sample_image_file_base64: str,
    ) -> None:
        """An image data URI is embedded and billed as one image."""
        response = cohere_client.embed(
            model=cohere_embed_v4_model,
            input_type="image",
            images=[sample_image_file_base64],
            embedding_types=["float"],
        )
        assert response.embeddings.float_ is not None
        (vector,) = response.embeddings.float_
        assert len(vector) > 0
        assert response.meta is not None
        assert response.meta.billed_units is not None
        assert response.meta.billed_units.images == 1

    @pytest.mark.expensive
    def test_every_advertised_model_embeds(
        self, cohere_client: cohere.ClientV2, live_client: httpx.Client
    ) -> None:
        """Every model advertising the route serves a text embed request."""
        response = live_client.get("/search_models", params={"route": "cohere_embed"})
        assert response.status_code == 200, response.text
        model_ids = [model["id"] for model in response.json()]
        assert model_ids
        for model_id in model_ids:
            embedded = self._embed(cohere_client, model_id)
            assert embedded.embeddings.float_ is not None
            (vector,) = embedded.embeddings.float_
            assert len(vector) > 0, model_id


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


class TestCohereEmbedV1Integration:
    """Live /v1/embed calls through the official Cohere SDK."""

    def test_embed_texts_floats_shape(
        self, cohere_client_v1: cohere.Client, cohere_embed_multilingual_model: str
    ) -> None:
        """Without embedding_types (or input_type) the legacy floats shape is returned."""
        response = cohere_client_v1.embed(
            model=cohere_embed_multilingual_model,
            texts=["The quick brown fox jumps over the lazy dog."],
        )
        assert response.response_type == "embeddings_floats"
        assert response.id
        (vector,) = response.embeddings
        assert len(vector) > 0
        assert all(isinstance(value, float) for value in vector)
        assert response.texts == ["The quick brown fox jumps over the lazy dog."]
        assert response.meta is not None
        assert response.meta.api_version is not None
        assert response.meta.api_version.version == "1"
        assert response.meta.billed_units is not None
        assert (response.meta.billed_units.input_tokens or 0) > 0

    def test_embedding_types_float_returns_by_type_shape(
        self, cohere_client_v1: cohere.Client, cohere_embed_multilingual_model: str
    ) -> None:
        """embedding_types=["float"] returns the by-type shape with v1 metadata."""
        response = cohere_client_v1.embed(
            model=cohere_embed_multilingual_model,
            input_type="search_document",
            texts=["The quick brown fox jumps over the lazy dog."],
            embedding_types=["float"],
        )
        assert response.response_type == "embeddings_by_type"
        assert response.embeddings.float_ is not None
        (vector,) = response.embeddings.float_
        assert len(vector) > 0
        assert response.meta is not None
        assert response.meta.api_version is not None
        assert response.meta.api_version.version == "1"

    def test_embed_texts_non_cohere_model(
        self,
        cohere_client_v1: cohere.Client,
        embedding_model: str,
        use_official_api: bool,
    ) -> None:
        """A non-Cohere model works without the Cohere-specific parameters."""
        if use_official_api:
            pytest.skip("Bedrock-specific model")
        response = cohere_client_v1.embed(
            model=embedding_model,
            texts=["The quick brown fox jumps over the lazy dog."],
        )
        assert response.response_type == "embeddings_floats"
        (vector,) = response.embeddings
        assert len(vector) > 0

    @pytest.mark.expensive
    def test_embed_image(
        self,
        cohere_client_v1: cohere.Client,
        cohere_embed_v4_model: str,
        sample_image_file_base64: str,
    ) -> None:
        """An image data URI is embedded and billed as one image."""
        response = cohere_client_v1.embed(
            model=cohere_embed_v4_model,
            input_type="image",
            images=[sample_image_file_base64],
        )
        assert response.response_type == "embeddings_floats"
        (vector,) = response.embeddings
        assert len(vector) > 0
        assert response.meta is not None
        assert response.meta.billed_units is not None
        assert response.meta.billed_units.images == 1
