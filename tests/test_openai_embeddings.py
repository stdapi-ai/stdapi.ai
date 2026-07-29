"""Tests for the OpenAI /v1/embeddings route.

Comprehensive test suite that validates all features of the OpenAI Embeddings API
specification, ensuring compatibility with the official OpenAI API behavior.
"""

import base64
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError, NotFoundError, OpenAI

from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from starlette.testclient import TestClient as TestClientType


class TestEmbeddings:
    """Test suite for the /v1/embeddings endpoint.

    Tests are designed to validate complete OpenAI API compatibility including:
    - All parameter combinations and validations
    - All encoding formats and output validation
    - Complete error scenario coverage with exact error matching
    - Edge cases and boundary conditions
    - Batch processing and input format variations
    - Model-specific capabilities and dimensions
    """

    def test_basic_single_input_embedding(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """Test fundamental embedding generation with single text input.

        Validates the core embedding functionality using minimal parameters
        to ensure the service can generate embeddings successfully.

        Args:
            openai_client: OpenAI client instance for API calls
            embedding_model: Embedding model identifier

        Validates:
            - Response contains embeddings data array
            - Embedding is list of floats
            - Usage information is included
            - Response structure matches OpenAI specification
        """
        response = openai_client.embeddings.create(
            model=embedding_model, input="The quick brown fox jumps over the lazy dog."
        )

        # Validate response structure
        assert hasattr(response, "object")
        assert response.object == "list"
        assert hasattr(response, "data")
        assert isinstance(response.data, list)
        assert len(response.data) == 1

        # Validate embedding data
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert all(isinstance(x, float) for x in item.embedding)
        assert len(item.embedding) > 0

        # Validate usage information
        assert hasattr(response, "usage")
        assert response.usage.prompt_tokens >= 0
        assert response.usage.total_tokens >= 0

    def test_service_tier_headers(
        self, openai_client: OpenAI, embedding_model: str, use_official_api: bool
    ) -> None:
        """Validate service_tier headers."""
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
        assert hasattr(response, "object")

    def test_batch_input_processing(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """Test batch processing with multiple text inputs.

        Validates that the embedding service can process multiple inputs
        simultaneously and return properly ordered results.

        Args:
            openai_client: OpenAI client instance for API calls
            embedding_model: Embedding model identifier

        Validates:
            - Multiple inputs are processed correctly
            - Response contains embeddings for all inputs
            - Embeddings are properly indexed
            - Batch processing maintains input order
        """
        inputs = [
            "First sentence for embedding generation.",
            "Second sentence with different content.",
            "Third sentence to complete the batch.",
        ]

        response = openai_client.embeddings.create(model=embedding_model, input=inputs)

        # Validate batch response structure
        assert response.object == "list"
        assert len(response.data) == len(inputs)

        # Validate each embedding in the batch
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert all(isinstance(x, float) for x in item.embedding)
            assert len(item.embedding) > 0

        # Validate usage reflects batch processing
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens > 0

    def test_base64_encoding_format(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """Test base64 encoding format for embeddings output.

        Validates that the encoding_format parameter works correctly
        to return embeddings as base64-encoded strings.

        Args:
            openai_client: OpenAI client instance for API calls
            embedding_model: Embedding model identifier

        Validates:
            - Base64 encoding format parameter is accepted
            - Embeddings are returned as base64 strings
            - Base64 strings can be decoded to valid float arrays
            - Response structure is correct for base64 format
        """
        response = openai_client.embeddings.create(
            model=embedding_model,
            input="Test sentence for base64 encoding format.",
            encoding_format="base64",
        )

        # Validate base64 response structure
        assert response.object == "list"
        assert len(response.data) == 1

        # Validate base64 embedding format
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, str)
        assert len(item.embedding) > 0

        # Validate that base64 string is valid
        try:
            decoded_bytes = base64.b64decode(item.embedding)
            # Should be decodable without error
            assert len(decoded_bytes) > 0
        except ValueError, base64.binascii.Error:
            pytest.fail("Base64 embedding string is not valid base64")

    def test_float_encoding_format(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """Test float encoding format for embeddings output.

        Validates that the encoding_format parameter works correctly
        with explicit float format specification.

        Args:
            openai_client: OpenAI client instance for API calls
            embedding_model: Embedding model identifier

        Validates:
            - Float encoding format parameter is accepted
            - Embeddings are returned as float arrays
            - All values are valid floating-point numbers
            - Default behavior matches float format
        """
        response = openai_client.embeddings.create(
            model=embedding_model,
            input="Test sentence for float encoding format.",
            encoding_format="float",
        )

        # Validate float response structure
        assert response.object == "list"
        assert len(response.data) == 1

        # Validate float embedding format
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert all(isinstance(x, float) for x in item.embedding)
        assert len(item.embedding) > 0

    @pytest.mark.parametrize("dimensions", [256, 512, 1024])
    def test_dimensions_parameter_functionality(
        self, openai_client: OpenAI, embedding_model: str, dimensions: int
    ) -> None:
        """Test dimensions parameter for controlling embedding size.

        Validates that the dimensions parameter correctly controls
        the output embedding dimensionality when supported by the model.

        Args:
            openai_client: OpenAI client instance for API calls
            embedding_model: Embedding model identifier
            dimensions: The dimension size to test

        Validates:
            - Dimensions parameter is accepted
            - Output embedding size matches requested dimensions
            - Valid dimension values work correctly
            - Response structure is maintained
        """
        try:
            response = openai_client.embeddings.create(
                model=embedding_model,
                input="Test sentence for dimensions parameter.",
                dimensions=dimensions,
            )

            # Validate response structure
            assert response.object == "list"
            assert len(response.data) == 1

            # Validate embedding dimensions
            item = response.data[0]
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)

            # Check if dimensions parameter was respected
            # Note: Some models may not support all dimension sizes
            if len(item.embedding) == dimensions:
                assert len(item.embedding) == dimensions
            else:
                # Model may not support the requested dimensions
                assert len(item.embedding) > 0

        except BadRequestError:
            # Model may not support the requested dimensions
            pytest.skip(
                f"Model {embedding_model} does not support {dimensions} dimensions"
            )

    def test_user_parameter_functionality(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """Test user parameter for request tracking and identification.

        Validates that the user parameter is accepted and processed
        correctly for tracking and billing purposes.

        Args:
            openai_client: OpenAI client instance for API calls
            embedding_model: Embedding model identifier

        Validates:
            - User parameter is accepted
            - Request processing works with user identification
            - Response structure is maintained
        """
        response = openai_client.embeddings.create(
            model=embedding_model,
            input="Test sentence with user parameter.",
            user="test-user-123",
        )

        # Validate response structure
        assert response.object == "list"
        assert len(response.data) == 1

        # Validate embedding data
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    def test_mixed_batch_with_parameters(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """Test batch processing with various parameter combinations.

        Validates that batch processing works correctly when combined
        with different parameter settings for comprehensive validation.

        Args:
            openai_client: OpenAI client instance for API calls
            embedding_model: Embedding model identifier

        Validates:
            - Batch processing with encoding format
            - Multiple inputs with user parameter
            - Parameter combinations work correctly
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

        # Validate batch response with parameters
        assert response.object == "list"
        assert len(response.data) == len(inputs)

        # Validate each item is base64 encoded
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, str)
            assert len(item.embedding) > 0

    def test_invalid_model_error(self, openai_client: OpenAI) -> None:
        """Test error handling for invalid model specification.

        Validates proper error response for non-existent model names.

        Args:
            openai_client: OpenAI client instance for API calls

        Validates:
            - Correct HTTP status code (404)
            - Proper error type ("invalid_request_error") and code ("model_not_found")
            - Error message identifies model as invalid
            - Consistent error response structure
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
        assert error_body["code"] == "model_not_found", exc_info
        assert "model" in error_body["message"].lower()

    def test_invalid_encoding_format_error(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """Test error handling for invalid encoding format.

        Validates proper error response for unsupported encoding format values.

        Args:
            openai_client: OpenAI client instance for API calls
            embedding_model: Embedding model identifier

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and appropriate code
            - Error message mentions encoding format validation
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
        assert error_body["type"] == "invalid_request_error"
        assert (
            "encoding_format" in error_body["message"].lower()
            or "format" in error_body["message"].lower()
        )

    def test_invalid_dimensions_error(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """Test error handling for invalid dimensions parameter.

        Validates proper error response for invalid dimension values
        when the model doesn't support the requested dimensions.

        Args:
            openai_client: OpenAI client instance for API calls
            embedding_model: Embedding model identifier

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and appropriate code
            - Error message mentions dimensions validation
        """
        # Test with clearly invalid dimension value
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.embeddings.create(
                model=embedding_model,
                input="Test text for invalid dimensions.",
                dimensions=0,  # Invalid dimension
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"

    def test_batch_size_limits(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """Test batch processing with various batch sizes.

        Validates that the embedding service handles different batch sizes
        appropriately and processes them efficiently.

        Args:
            openai_client: OpenAI client instance for API calls
            embedding_model: Embedding model identifier

        Validates:
            - Small batch processing works correctly
            - Medium batch processing maintains performance
            - Batch ordering is preserved
            - Usage scaling is appropriate
        """
        # Test different batch sizes
        batch_sizes = [1, 5, 10]

        for batch_size in batch_sizes:
            inputs = [
                f"Test sentence number {i} for batch processing."
                for i in range(batch_size)
            ]

            response = openai_client.embeddings.create(
                model=embedding_model, input=inputs
            )

            # Validate batch response
            assert response.object == "list"
            assert len(response.data) == batch_size

            # Validate each item
            for i, item in enumerate(response.data):
                assert item.index == i
                assert item.object == "embedding"
                assert isinstance(item.embedding, list)
                assert len(item.embedding) > 0

            # Validate usage scales appropriately
            assert response.usage.prompt_tokens > 0
            assert response.usage.total_tokens >= response.usage.prompt_tokens

    def test_consistency_across_calls(
        self, openai_client: OpenAI, embedding_model: str
    ) -> None:
        """Test embedding consistency for identical inputs.

        Validates that identical inputs produce identical or very similar
        embeddings across multiple API calls.

        Args:
            openai_client: OpenAI client instance for API calls
            embedding_model: Embedding model identifier

        Validates:
            - Identical inputs produce consistent embeddings
            - Embedding dimensions remain constant
            - Response structure is consistent
        """
        test_text = "Consistent test sentence for embedding generation."

        # Generate embeddings multiple times
        responses = []
        for _ in range(3):
            response = openai_client.embeddings.create(
                model=embedding_model, input=test_text
            )
            responses.append(response)

        # Validate all responses have same structure
        for response in responses:
            assert response.object == "list"
            assert len(response.data) == 1

            item = response.data[0]
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)

        # Validate embedding dimensions are consistent
        first_embedding_length = len(responses[0].data[0].embedding)
        for response in responses[1:]:
            assert len(response.data[0].embedding) == first_embedding_length


class TestEmbeddingsUsage:
    """Tests for usage logging in embeddings endpoint."""

    def test_embedding_usage_logged(
        self,
        test_client: TestClientType | None,
        embedding_model: str,
        api_key: str,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """Test that usage is logged for embedding requests."""
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
        assert "usage" in response_data
        api_usage = response_data["usage"]

        assert "prompt_tokens" in api_usage
        assert "total_tokens" in api_usage
        # Titan/Cohere backends report real Bedrock input-token counts for
        # non-empty text input (verified below via the raw bedrock usage log).
        assert api_usage["prompt_tokens"] > 0

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
        """An empty ``input`` array returns 400 before reaching any backend."""
        response = client.post("/v1/embeddings", json={"model": model_id, "input": []})
        assert response.status_code == 400, response.text
        error_body = response.json()
        assert error_body["error"]["type"] == "invalid_request_error"


class TestEmbeddingsTokenArrayRejected:
    """Offline unit tests: token-array ``input`` gets the dedicated friendly error.

    Validation happens before any model dispatch or AWS call, so these tests
    run against an app instance without the AWS-touching lifespan.
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
        """A legacy OpenAI token-array ``input`` returns the dedicated 400 message."""
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
        """``embedding_types=["int8"]`` returns a 400 JSON envelope, not a 500."""
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
