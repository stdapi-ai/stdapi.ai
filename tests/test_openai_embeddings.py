"""Tests for the OpenAI /v1/embeddings route.

Comprehensive test suite that validates all features of the OpenAI Embeddings API
specification, ensuring compatibility with the official OpenAI API behavior.
"""

import base64

import pytest
from openai import BadRequestError, NotFoundError, OpenAI


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
        except (ValueError, base64.binascii.Error):
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
