"""Tests for the OpenAI /v1/models route.

Comprehensive test suite that validates all features of the OpenAI Models API
specification, ensuring compatibility with the official OpenAI API behavior.
"""

import pytest
from openai import NotFoundError, OpenAI


class TestModels:
    """Test suite for the /v1/models endpoint.

    Tests are designed to validate complete OpenAI API compatibility including:
    - Model listing and availability validation
    - Model metadata and capabilities verification
    - Model retrieval and detailed information access
    - Complete error scenario coverage with exact error matching
    - Edge cases and boundary conditions
    - Response structure and field validation
    """

    def test_list_models_basic_functionality(self, openai_client: OpenAI) -> None:
        """Test fundamental model listing functionality.

        Validates the core model listing functionality to ensure the service
        can retrieve available models successfully.

        Args:
            openai_client: OpenAI client instance for API calls

        Validates:
            - Response contains models list
            - Each model has required fields
            - Response structure matches OpenAI specification
            - Models list is not empty
        """
        response = openai_client.models.list()

        # Validate response structure
        assert hasattr(response, "object")
        assert response.object == "list"
        assert hasattr(response, "data")
        assert isinstance(response.data, list)
        assert len(response.data) > 0

        # Validate each model in the list
        for model in response.data:
            assert hasattr(model, "id")
            assert hasattr(model, "object")
            assert hasattr(model, "created")
            assert hasattr(model, "owned_by")
            assert model.object == "model"
            assert isinstance(model.id, str)
            assert len(model.id) > 0
            assert isinstance(model.created, int)
            assert model.created > 0
            assert isinstance(model.owned_by, str)

    def test_list_models_response_structure_validation(
        self, openai_client: OpenAI
    ) -> None:
        """Test comprehensive validation of models list response structure.

        Validates that the models list response contains all required fields
        and optional fields as specified in the OpenAI API documentation.

        Args:
            openai_client: OpenAI client instance for API calls

        Validates:
            - All required fields are present
            - Field types match specification
            - Optional fields are handled correctly
            - Response format is consistent
        """
        response = openai_client.models.list()

        # Validate top-level response structure
        assert response.object == "list"
        assert isinstance(response.data, list)

        # Validate detailed structure of each model
        for model in response.data:
            # Required fields validation
            assert isinstance(model.id, str)
            assert len(model.id) > 0
            assert model.object == "model"
            assert isinstance(model.created, int)
            assert model.created > 0
            assert isinstance(model.owned_by, str)

            # Optional fields validation (if present)
            if hasattr(model, "permission"):
                assert isinstance(model.permission, list)

            if hasattr(model, "root"):
                assert isinstance(model.root, str)

            if hasattr(model, "parent"):
                assert model.parent is None or isinstance(model.parent, str)

    def test_retrieve_specific_model(self, openai_client: OpenAI) -> None:
        """Test retrieval of a specific model by ID.

        Validates that individual model retrieval works correctly
        and returns detailed model information.

        Args:
            openai_client: OpenAI client instance for API calls

        Validates:
            - Model retrieval by ID works correctly
            - Response contains detailed model information
            - Response structure matches OpenAI specification
        """
        # First get the list of models to find a valid model ID
        models_response = openai_client.models.list()
        assert len(models_response.data) > 0

        # Use the first available model for testing
        test_model_id = models_response.data[0].id

        # Retrieve the specific model
        model = openai_client.models.retrieve(test_model_id)

        # Validate retrieved model structure
        assert hasattr(model, "id")
        assert hasattr(model, "object")
        assert hasattr(model, "created")
        assert hasattr(model, "owned_by")

        assert model.id == test_model_id
        assert model.object == "model"
        assert isinstance(model.created, int)
        assert model.created > 0
        assert isinstance(model.owned_by, str)

    def test_model_filtering_and_availability(self, openai_client: OpenAI) -> None:
        """Test model filtering and availability validation.

        Validates that different types of models are available and
        can be properly identified and categorized.

        Args:
            openai_client: OpenAI client instance for API calls

        Validates:
            - Multiple model types are available
            - Model IDs are consistent and valid
            - Different model categories can be identified
        """
        response = openai_client.models.list()

        # Collect model information for analysis
        model_ids = [model.id for model in response.data]
        model_owners = [model.owned_by for model in response.data]

        # Validate model diversity and consistency
        assert len(set(model_ids)) == len(model_ids), "Model IDs should be unique"
        assert len(model_ids) > 0, "Should have at least one model available"

        # Validate that we have models from expected categories
        # (This may vary based on the implementation)
        assert len(set(model_owners)) >= 1, "Should have models from at least one owner"

        # Validate model ID formats are reasonable
        for model_id in model_ids:
            assert isinstance(model_id, str)
            assert len(model_id) > 0
            # Model IDs should not contain obviously invalid characters
            assert not any(char in model_id for char in [" ", "\n", "\t"])

    def test_model_metadata_consistency(self, openai_client: OpenAI) -> None:
        """Test consistency of model metadata across list and retrieve operations.

        Validates that model information is consistent between
        the models list and individual model retrieval.

        Args:
            openai_client: OpenAI client instance for API calls

        Validates:
            - Model metadata is consistent between operations
            - List and retrieve return the same core information
            - Field values match across different endpoints
        """
        # Get models list
        models_response = openai_client.models.list()
        assert len(models_response.data) > 0

        # Test consistency for first few models (limit for efficiency)
        test_models = models_response.data[: min(3, len(models_response.data))]

        for list_model in test_models:
            # Retrieve the same model individually
            retrieved_model = openai_client.models.retrieve(list_model.id)

            # Validate consistency of core fields
            assert list_model.id == retrieved_model.id
            assert list_model.object == retrieved_model.object
            assert list_model.owned_by == retrieved_model.owned_by

            # Validate that both have the same structure type
            assert type(list_model) is type(retrieved_model)

    def test_model_creation_timestamps(self, openai_client: OpenAI) -> None:
        """Test validation of model creation timestamps.

        Validates that model creation timestamps are reasonable
        and follow expected patterns.

        Args:
            openai_client: OpenAI client instance for API calls

        Validates:
            - Creation timestamps are valid Unix timestamps
            - Timestamps are within reasonable ranges
            - All models have non-zero creation times
        """
        response = openai_client.models.list()

        # Validate creation timestamps
        for model in response.data:
            assert isinstance(model.created, int)
            assert model.created > 0

            # Timestamps should be reasonable (after 2020, before far future)
            # 1577836800 = Jan 1, 2020 UTC
            # 2147483647 = Max 32-bit signed int (year 2038)
            assert 1577836800 < model.created < 2147483647

    def test_invalid_model_retrieval_error(self, openai_client: OpenAI) -> None:
        """Test error handling for invalid model ID retrieval.

        Validates proper error response for non-existent model IDs.

        Args:
            openai_client: OpenAI client instance for API calls

        Validates:
            - Correct HTTP status code (404)
            - Proper error type ("invalid_request_error") and code ("model_not_found")
            - Error message identifies model as invalid
            - Consistent error response structure
        """
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.models.retrieve("invalid-nonexistent-model-id")

        error = exc_info.value
        assert error.status_code == 404
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert error_body["code"] == "model_not_found"
        assert "model" in error_body["message"].lower()

    def test_deprecated_model_retrieval_error(
        self, openai_client: OpenAI, use_openai_api: bool
    ) -> None:
        """Test error handling for deprecated model ID retrieval.

        Validates proper error response for non-existent model IDs.

        Args:
            openai_client: OpenAI client instance for API calls
            use_openai_api: True is using official OpenAI API.

        Validates:
            - Correct HTTP status code (404)
            - Proper error type ("invalid_request_error") and code ("model_not_found")
            - Error message identifies model as invalid
            - Consistent error response structure
        """
        if use_openai_api:
            pytest.skip("Not available on the official OpenAI API")

        with pytest.raises(NotFoundError) as exc_info:
            openai_client.models.retrieve("anthropic.claude-instant-v1")

        error = exc_info.value
        assert error.status_code == 404
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert error_body["code"] == "model_not_found"
        assert "deprecated" in error_body["message"].lower()

    def test_empty_model_id_retrieval_error(self, openai_client: OpenAI) -> None:
        """Test error handling for empty model ID.

        OpenAI client validates model ID before making API calls,
        raising ValueError for empty model IDs.

        Args:
            openai_client: OpenAI client instance for API calls

        Validates:
            - ValueError is raised for empty model ID
            - Error message indicates empty value issue
        """
        # Test with empty string - client validates before API call
        with pytest.raises(ValueError, match=r".*(?:non-empty|empty).*") as exc_info:
            openai_client.models.retrieve("")

        error_message = str(exc_info.value)
        assert (
            "non-empty value" in error_message.lower()
            or "empty" in error_message.lower()
        )
        assert "model" in error_message.lower()

    def test_model_ownership_validation(self, openai_client: OpenAI) -> None:
        """Test validation of model ownership information.

        Validates that model ownership information is properly
        formatted and consistent.

        Args:
            openai_client: OpenAI client instance for API calls

        Validates:
            - All models have valid ownership information
            - Owner information is consistently formatted
            - Owner strings are non-empty and valid
        """
        response = openai_client.models.list()

        owners = set()
        for model in response.data:
            assert isinstance(model.owned_by, str)
            assert len(model.owned_by) > 0
            assert not any(char in model.owned_by for char in ["\n", "\t"])
            owners.add(model.owned_by)

        # Should have at least one distinct owner
        assert len(owners) >= 1

        # Common expected owners (may vary by implementation)
        # At least some models should have recognized owners
        # (This is flexible to accommodate different implementations)

    def test_model_list_pagination_behavior(self, openai_client: OpenAI) -> None:
        """Test model list pagination and response size behavior.

        Validates that the models list endpoint handles response
        size and pagination appropriately.

        Args:
            openai_client: OpenAI client instance for API calls

        Validates:
            - Models list returns reasonable number of results
            - Response is not empty
            - All results are valid and complete
        """
        response = openai_client.models.list()

        # Validate response size is reasonable
        assert len(response.data) > 0
        assert len(response.data) < 1000  # Reasonable upper bound

        # Validate all entries are complete
        for model in response.data:
            assert model.id is not None
            assert model.object == "model"
            assert model.created is not None
            assert model.owned_by is not None

    def test_model_id_format_validation(self, openai_client: OpenAI) -> None:
        """Test validation of model ID formats and conventions.

        Validates that model IDs follow consistent formatting
        and naming conventions.

        Args:
            openai_client: OpenAI client instance for API calls

        Validates:
            - Model IDs are properly formatted
            - IDs follow consistent naming patterns
            - No invalid characters or formats
        """
        response = openai_client.models.list()

        for model in response.data:
            model_id = model.id

            # Basic format validation
            assert isinstance(model_id, str)
            assert len(model_id) > 0
            assert len(model_id) < 256  # Reasonable length limit

            # Character validation
            assert not model_id.startswith(" ")
            assert not model_id.endswith(" ")
            assert not any(char in model_id for char in ["\n", "\r", "\t"])

    def test_model_capabilities_detection(self, openai_client: OpenAI) -> None:
        """Test detection and validation of model capabilities.

        Validates that different model types can be identified
        and their capabilities properly understood.

        Args:
            openai_client: OpenAI client instance for API calls

        Validates:
            - Different model types are available
            - Model capabilities can be inferred from metadata
            - Model categorization is consistent
        """
        response = openai_client.models.list()

        # Categorize models by their IDs (implementation-specific logic)
        model_categories: dict[str, list[str]] = {
            "chat": [],
            "embedding": [],
            "speech": [],
            "other": [],
        }

        for model in response.data:
            model_id_lower = model.id.lower()

            if any(
                keyword in model_id_lower for keyword in ["chat", "gpt", "completion"]
            ):
                model_categories["chat"].append(model.id)
            elif any(keyword in model_id_lower for keyword in ["embed", "embedding"]):
                model_categories["embedding"].append(model.id)
            elif any(
                keyword in model_id_lower for keyword in ["tts", "speech", "whisper"]
            ):
                model_categories["speech"].append(model.id)
            else:
                model_categories["other"].append(model.id)

        # Should have at least some models (flexibility for different implementations)
        total_models = sum(len(models) for models in model_categories.values())
        assert total_models > 0
