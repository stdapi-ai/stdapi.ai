"""Tests for the OpenAI-compatible ``/v1/embeddings`` route backed by Bedrock.

The pinned models are ``amazon.titan-embed-text-v2:0`` locally and
``text-embedding-3-small`` against the official API; both return L2-normalized
float32 vectors, which is what the numeric assertions below rely on.

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://stdapi.ai/api_openai_embeddings/
     stdapi/routes/openai_embeddings.py:create_embeddings
"""

import base64
from array import array
from math import hypot
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError, NotFoundError, OpenAI

from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from starlette.testclient import TestClient as TestClientType

#: Smallest vector length any embedding model in the suite returns (Titan V2 256).
_MIN_DIMENSIONS = 256


def _decode_base64_vector(encoded: object) -> list[float]:
    """Decode an ``encoding_format=base64`` embedding into its float32 vector.

    The gateway encodes ``array("f", vector).tobytes()``, so the payload length
    must be a whole number of 4-byte little-endian floats.

    Ref: https://stdapi.ai/api_openai_embeddings/
         stdapi/routes/openai_embeddings.py:create_embeddings
    """
    assert isinstance(encoded, str), f"expected a base64 string, got {type(encoded)}"
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError:
        pytest.fail("Base64 embedding string is not valid base64")
    assert raw, "Base64 embedding decoded to an empty payload"
    assert len(raw) % 4 == 0, "Base64 payload is not a whole number of float32 values"
    return array("f", raw).tolist()


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return the cosine similarity of two equal-length vectors."""
    return sum(x * y for x, y in zip(left, right, strict=True)) / (
        hypot(*left) * hypot(*right)
    )


class TestEmbeddings:
    """OpenAI ``/v1/embeddings`` request/response contract.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         https://developers.openai.com/api/docs/guides/embeddings
         stdapi/routes/openai_embeddings.py:create_embeddings
    """

    def test_basic_single_input_embedding(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """A single string input returns one indexed float vector and token usage.

        ``CreateEmbeddingResponse`` requires ``object="list"``, the echoed model, one
        ``Embedding`` per input and a ``usage`` block; the gateway fills
        ``prompt_tokens`` from the backend's ``inputTextTokenCount`` and defaults
        ``total_tokens`` to it when the backend reports no output tokens.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
             stdapi/types/openai_embeddings.py:CreateEmbeddingResponse
        """
        response = openai_client.embeddings.create(
            model=embedding_model, input="The quick brown fox jumps over the lazy dog."
        )

        assert response.object == "list"
        assert response.model == embedding_model
        assert len(response.data) == 1

        item = response.data[0]
        assert item.object == "embedding"
        assert item.index == 0
        assert isinstance(item.embedding, list)
        assert all(isinstance(x, float) for x in item.embedding)
        assert len(item.embedding) >= _MIN_DIMENSIONS
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"
        assert hypot(*item.embedding) == pytest.approx(1.0, abs=0.05), (
            "vector is not L2-normalized"
        )

        assert response.usage.prompt_tokens > 0, "no input tokens billed for text input"
        assert response.usage.total_tokens >= response.usage.prompt_tokens

    def test_service_tier_headers(
        self, openai_client: OpenAI, embedding_model: str, use_official_api: bool
    ) -> None:
        """Bedrock performance/service-tier headers are accepted on the embeddings route.

        ``set_performance_configuration`` turns the two ``X-Amzn-Bedrock-*`` headers
        into ``performanceConfigLatency`` and ``serviceTier`` on the InvokeModel call.
        Neither is echoed in the OpenAI response body, so the observable contract is
        that the request still succeeds and returns a normal embedding.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
             https://docs.aws.amazon.com/bedrock/latest/userguide/latency-optimized-inference.html
             stdapi/aws_bedrock.py:set_performance_configuration
        """
        if use_official_api:
            pytest.skip("Service tiers headers are not supported on the official API")
        response = openai_client.embeddings.create(
            model=embedding_model,
            input="The quick brown fox jumps over the lazy dog.",
            extra_headers={
                "X-Amzn-Bedrock-PerformanceConfig-Latency": "standard",
                "X-Amzn-Bedrock-Service-Tier": "default",
            },
        )
        assert response.object == "list"
        assert response.model == embedding_model
        assert len(response.data) == 1
        assert len(response.data[0].embedding) >= _MIN_DIMENSIONS
        assert response.usage.prompt_tokens > 0

    def test_batch_input_processing(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """A string array returns one same-width vector per input, in request order.

        Titan embeds one text per InvokeModel call, so ordering is a gateway
        guarantee (the fan-out is gathered in input order) rather than a backend one,
        and ``usage`` aggregates every call.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
             stdapi/models/embedding/amazon_titan_embed.py:EmbeddingModel.embed_text
        """
        inputs = [
            "First sentence for embedding generation.",
            "Second sentence with different content.",
            "Third sentence to complete the batch.",
        ]

        response = openai_client.embeddings.create(model=embedding_model, input=inputs)

        assert response.object == "list"
        assert len(response.data) == len(inputs)

        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert all(isinstance(x, float) for x in item.embedding)

        vectors = [item.embedding for item in response.data]
        assert len({len(vector) for vector in vectors}) == 1, (
            "batch returned vectors of different widths"
        )
        assert len({tuple(vector) for vector in vectors}) == len(inputs), (
            "distinct inputs produced identical vectors"
        )

        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens >= response.usage.prompt_tokens

    def test_base64_encoding_format(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """``encoding_format=base64`` returns the vector as a base64 float32 blob.

        The gateway encodes ``array("f", vector).tobytes()``, so the decoded payload
        is a little-endian float32 array of the model's normal width, not a shortened
        or textual representation.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_embeddings.py:create_embeddings
        """
        response = openai_client.embeddings.create(
            model=embedding_model,
            input="Test sentence for base64 encoding format.",
            encoding_format="base64",
        )

        assert response.object == "list"
        assert len(response.data) == 1

        item = response.data[0]
        assert item.object == "embedding"
        assert item.index == 0
        assert isinstance(item.embedding, str), "base64 embedding is not a string"

        vector = _decode_base64_vector(item.embedding)
        assert len(vector) >= _MIN_DIMENSIONS
        assert any(x != 0.0 for x in vector), "decoded vector is all zeros"
        assert hypot(*vector) == pytest.approx(1.0, abs=0.05), (
            "decoded vector is not L2-normalized"
        )

    def test_float_encoding_format(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """``encoding_format=float`` returns the raw float vector, as the default does.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_embeddings.py:create_embeddings
        """
        response = openai_client.embeddings.create(
            model=embedding_model,
            input="Test sentence for float encoding format.",
            encoding_format="float",
        )

        assert response.object == "list"
        assert len(response.data) == 1

        item = response.data[0]
        assert item.object == "embedding"
        assert item.index == 0
        assert isinstance(item.embedding, list)
        assert all(isinstance(x, float) for x in item.embedding)
        assert len(item.embedding) >= _MIN_DIMENSIONS
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"
        assert hypot(*item.embedding) == pytest.approx(1.0, abs=0.05), (
            "vector is not L2-normalized"
        )

    @pytest.mark.parametrize("dimensions", [256, 512, 1024])
    def test_dimensions_parameter_functionality(
        self, openai_client: OpenAI, embedding_model: str, dimensions: int
    ) -> None:
        """``dimensions`` sets the exact returned vector width.

        256/512/1024 are the three values Titan Text Embeddings V2 accepts, and
        text-embedding-3-small shortens to any of them, so every parametrization
        must come back with exactly that many components.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
             https://developers.openai.com/api/docs/guides/embeddings
        """
        response = openai_client.embeddings.create(
            model=embedding_model,
            input="Test sentence for dimensions parameter.",
            dimensions=dimensions,
        )

        assert response.object == "list"
        assert len(response.data) == 1

        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) == dimensions, (
            f"requested {dimensions} dimensions, got {len(item.embedding)}"
        )
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    def test_user_parameter_functionality(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """``user`` is accepted and leaves the response unchanged.

        The gateway consumes ``user`` only as the request-log identifier
        (``log_request_params(..., user_id=request.user)``); it is never forwarded to
        Bedrock and never echoed, so the contract is a normal successful response.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_embeddings.py:create_embeddings
        """
        response = openai_client.embeddings.create(
            model=embedding_model,
            input="Test sentence with user parameter.",
            user="test-user-123",
        )

        assert response.object == "list"
        assert response.model == embedding_model
        assert len(response.data) == 1

        item = response.data[0]
        assert item.object == "embedding"
        assert item.index == 0
        assert isinstance(item.embedding, list)
        assert len(item.embedding) >= _MIN_DIMENSIONS
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    def test_mixed_batch_with_parameters(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """A batch combined with ``encoding_format=base64`` and ``user`` stays ordered.

        Every item is encoded independently, so all base64 blobs must decode to
        distinct float32 vectors of the same width.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_embeddings.py:create_embeddings
        """
        inputs = [
            "First batch item with parameters.",
            "Second batch item for comprehensive testing.",
        ]

        response = openai_client.embeddings.create(
            model=embedding_model,
            input=inputs,
            encoding_format="base64",
            user="batch-test-user",
        )

        assert response.object == "list"
        assert len(response.data) == len(inputs)

        vectors: list[list[float]] = []
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, str), "base64 embedding is not a string"
            vectors.append(_decode_base64_vector(item.embedding))

        assert len({len(vector) for vector in vectors}) == 1, (
            "batch returned vectors of different widths"
        )
        assert len({tuple(vector) for vector in vectors}) == len(inputs), (
            "distinct inputs produced identical vectors"
        )

    def test_invalid_model_error(self, openai_client: OpenAI) -> None:
        """An unknown model is a 404 ``invalid_request_error`` with ``model_not_found``.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             stdapi/api_errors.py:UnsupportedModelError
        """
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.embeddings.create(
                model="invalid-nonexistent-embedding-model",
                input="Test text for invalid model.",
            )

        error = exc_info.value
        assert error.status_code == 404
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error", error_body
        assert error_body["code"] == "model_not_found", error_body
        assert "invalid-nonexistent-embedding-model" in error_body["message"], (
            error_body
        )

    def test_invalid_encoding_format_error(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """An out-of-enum ``encoding_format`` is rejected as a 400 before any backend call.

        ``encoding_format`` is a ``Literal["float", "base64"]``, so Pydantic rejects the
        value and ``handle_validation_exception`` reports it as
        ``Validation error at body.encoding_format: ...``. The message check stays
        tolerant because the official API words its own 400 differently.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             stdapi/types/openai_embeddings.py:EmbeddingCreateParams
             stdapi/main.py:handle_validation_exception
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.embeddings.create(
                model=embedding_model,
                input="Test text for invalid encoding format.",
                encoding_format="invalid_format",  # type: ignore[arg-type]
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error", error_body
        message = error_body["message"].lower()
        assert "encoding_format" in message or "format" in message, error_body

    def test_invalid_dimensions_error(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """``dimensions=0`` is rejected as a 400 naming the offending field.

        ``dimensions`` is constrained to ``1 <= n <= 8192`` on the request model, so
        the rejection happens in validation and never reaches Bedrock.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             stdapi/types/openai_embeddings.py:EmbeddingCreateParams
             stdapi/main.py:handle_validation_exception
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.embeddings.create(
                model=embedding_model,
                input="Test text for invalid dimensions.",
                dimensions=0,
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error", error_body
        assert "dimensions" in error_body["message"].lower(), error_body

    def test_batch_size_limits(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """Batches of 1, 5 and 10 inputs return one vector each and bill more tokens.

        The inputs are near-identical sentences, so the aggregated ``prompt_tokens``
        must grow strictly with the batch size — the check that usage really is
        summed across the per-input Bedrock calls rather than reported once.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/models/embedding/amazon_titan_embed.py:EmbeddingModel.embed_text
        """
        batch_sizes = [1, 5, 10]
        prompt_tokens: list[int] = []

        for batch_size in batch_sizes:
            inputs = [
                f"Test sentence number {i} for batch processing."
                for i in range(batch_size)
            ]

            response = openai_client.embeddings.create(
                model=embedding_model, input=inputs
            )

            assert response.object == "list"
            assert len(response.data) == batch_size

            for i, item in enumerate(response.data):
                assert item.index == i
                assert item.object == "embedding"
                assert isinstance(item.embedding, list)
                assert len(item.embedding) >= _MIN_DIMENSIONS

            assert len({len(item.embedding) for item in response.data}) == 1, (
                "batch returned vectors of different widths"
            )
            assert response.usage.total_tokens >= response.usage.prompt_tokens
            prompt_tokens.append(response.usage.prompt_tokens)

        assert prompt_tokens[0] > 0
        assert prompt_tokens[0] < prompt_tokens[1] < prompt_tokens[2], (
            f"prompt_tokens did not scale with batch size: {prompt_tokens}"
        )

    def test_consistency_across_calls(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """The same input embedded three times yields the same vector every time.

        Embedding inference is deterministic, so repeated calls (possibly served by
        different Regions) must return vectors of identical width and a cosine
        similarity of 1 with the first call.

        Ref: https://developers.openai.com/api/docs/guides/embeddings
             stdapi/routes/openai_embeddings.py:create_embeddings
        """
        test_text = "Consistent test sentence for embedding generation."

        vectors: list[list[float]] = []
        for _ in range(3):
            response = openai_client.embeddings.create(
                model=embedding_model, input=test_text
            )
            assert response.object == "list"
            assert len(response.data) == 1

            item = response.data[0]
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            vectors.append(item.embedding)

        assert len({len(vector) for vector in vectors}) == 1, (
            "repeated calls returned vectors of different widths"
        )
        for vector in vectors[1:]:
            assert _cosine_similarity(vectors[0], vector) == pytest.approx(
                1.0, abs=1e-3
            ), "repeated embedding of the same text diverged"


class TestEmbeddingsUsage:
    """Usage accounting emitted by the embeddings route.

    Ref: https://stdapi.ai/api_openai_embeddings/
         stdapi/usage.py:record_bedrock_usage
    """

    def test_embedding_usage_logged(
        self,
        test_client: TestClientType | None,
        embedding_model: str,
        api_key: str,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """An embeddings request bills Bedrock input tokens and reports them as usage.

        The API-level ``usage.prompt_tokens`` and the ``bedrock-runtime`` usage log
        entry are produced from the same backend token count, so a non-empty text
        input must yield a positive count in both places.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
             stdapi/routes/openai_embeddings.py:create_embeddings
        """
        if test_client is None:
            pytest.skip("Requires local test server")

        capfd.readouterr()

        response = test_client.post(
            "/v1/embeddings",
            json={"model": embedding_model, "input": "Hello world"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert response.status_code == 200

        response_data = response.json()
        assert response_data["object"] == "list"
        assert response_data["model"] == embedding_model
        assert len(response_data["data"]) == 1
        assert "usage" in response_data
        api_usage = response_data["usage"]

        assert "prompt_tokens" in api_usage
        assert "total_tokens" in api_usage
        # Titan/Cohere backends report real Bedrock input-token counts for
        # non-empty text input (verified below via the raw bedrock usage log).
        assert api_usage["prompt_tokens"] > 0
        assert api_usage["total_tokens"] >= api_usage["prompt_tokens"]

        bedrock_entries = logged_usage_entries(
            capfd.readouterr().out,
            service="bedrock-runtime",
            operation="/v1/embeddings",
            model=embedding_model,
        )
        assert bedrock_entries, "Expected bedrock service in usage"
        bedrock_entry = bedrock_entries[0]
        assert bedrock_entry["input_tokens"] > 0


class TestEmbeddingsEmptyInputRejected:
    """Offline unit tests: an empty ``input`` array is rejected uniformly.

    Validation happens before any model dispatch or AWS call, so these tests
    run against an app instance without the AWS-touching lifespan.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/types/openai_embeddings.py:EmbeddingCreateParams
    """

    pytestmark = pytest.mark.local

    @pytest.fixture
    def client(self, api_key: str) -> TestClientType:
        """Test client without lifespan (no AWS startup), pre-authenticated."""
        from starlette.testclient import TestClient  # noqa: PLC0415

        from stdapi.main import app  # noqa: PLC0415

        return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})

    @pytest.mark.parametrize(
        "model_id", ["text-embedding-3-small", "cohere.embed-english-v3"]
    )
    def test_empty_input_array_is_rejected(
        self, client: TestClientType, model_id: str
    ) -> None:
        """An empty ``input`` array returns 400 before reaching any backend.

        The ``input`` field carries ``min_length=1``, and the failure is reported
        against ``body.input`` whichever union member Pydantic reports first. The
        client has no lifespan, so any backend dispatch would fail differently.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/main.py:handle_validation_exception
        """
        response = client.post("/v1/embeddings", json={"model": model_id, "input": []})
        assert response.status_code == 400, response.text
        error_body = response.json()
        assert error_body["error"]["type"] == "invalid_request_error"
        assert "input" in error_body["error"]["message"], error_body


class TestEmbeddingsTokenArrayRejected:
    """Offline unit tests: token-array ``input`` gets the dedicated friendly error.

    Validation happens before any model dispatch or AWS call, so these tests
    run against an app instance without the AWS-touching lifespan.

    Ref: https://stdapi.ai/api_openai_embeddings/
         stdapi/types/openai_embeddings.py:EmbeddingCreateParams
    """

    pytestmark = pytest.mark.local

    @pytest.fixture
    def client(self, api_key: str) -> TestClientType:
        """Test client without lifespan (no AWS startup), pre-authenticated."""
        from starlette.testclient import TestClient  # noqa: PLC0415

        from stdapi.main import app  # noqa: PLC0415

        return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})

    @pytest.mark.parametrize("input_value", [[1, 2, 3], [[1, 2], [3, 4]]])
    def test_token_array_input_returns_friendly_error(
        self, client: TestClientType, input_value: list[object]
    ) -> None:
        """A legacy OpenAI token-array ``input`` returns the dedicated 400 message.

        The OpenAI schema allows an array of ints and an array of int arrays; no
        Bedrock embedding model accepts pre-tokenized input, so the gateway rejects
        both shapes with its own message instead of a generic type error.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_embeddings.py:EmbeddingCreateParams._unsupported
        """
        response = client.post(
            "/v1/embeddings",
            json={"model": "text-embedding-3-small", "input": input_value},
        )
        assert response.status_code == 400, response.text
        error_body = response.json()
        assert error_body["error"]["type"] == "invalid_request_error"
        assert "Token array inputs are not supported" in error_body["error"]["message"]


class TestEmbeddingsCohereUnsupportedEmbeddingType:
    """Offline unit test: non-``float`` Cohere ``embedding_types`` are rejected cleanly.

    The Bedrock client is stubbed at the model layer (``EmbeddingModel.invoke``),
    so no AWS call is made; ``validate_model`` is stubbed too, avoiding the
    AWS-touching model-cache lookup.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html
         stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
    """

    pytestmark = pytest.mark.local

    @pytest.fixture
    def client(self, api_key: str, monkeypatch: pytest.MonkeyPatch) -> TestClientType:
        """Test client with stubbed model validation and Bedrock invocation."""
        from unittest.mock import AsyncMock  # noqa: PLC0415

        from starlette.testclient import TestClient  # noqa: PLC0415

        from stdapi.main import app  # noqa: PLC0415
        from stdapi.models import InvokeResult  # noqa: PLC0415
        from stdapi.models.embedding.cohere_embed import EmbeddingModel  # noqa: PLC0415

        class _StubModelDetails:
            """Minimal stand-in exposing only the ``id`` attribute the route reads."""

            def __init__(self, model_id: str) -> None:
                self.id = model_id

        async def _stub_validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> object:
            return _StubModelDetails(model_id)

        monkeypatch.setattr(
            "stdapi.routes.openai_embeddings.validate_model", _stub_validate_model
        )
        monkeypatch.setattr(
            EmbeddingModel,
            "invoke",
            AsyncMock(
                return_value=InvokeResult(
                    response={"embeddings": {"int8": [[1, 2, 3]]}}
                )
            ),
        )
        return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})

    def test_non_float_embedding_type_returns_400(self, client: TestClientType) -> None:
        """``embedding_types=["int8"]`` returns a 400 JSON envelope, not a 500.

        Cohere can return int8/uint8/binary/ubinary vectors, but the OpenAI
        ``Embedding`` shape only carries floats, so a response keyed without
        ``float`` is turned into a 400 instead of raising a ``KeyError``.

        Ref: https://docs.cohere.com/reference/embed
             stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
        """
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "cohere.embed-english-v3",
                "input": "Hello world",
                "embedding_types": ["int8"],
            },
        )
        assert response.status_code == 400, response.text
        error_body = response.json()
        assert error_body["error"]["type"] == "invalid_request_error"
        assert "float" in error_body["error"]["message"]
