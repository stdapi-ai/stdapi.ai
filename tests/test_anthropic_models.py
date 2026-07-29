"""Tests for the Anthropic /v1/models route.

Comprehensive test suite that validates all features of the Anthropic Models API
specification, ensuring compatibility with the official Anthropic API behavior.
"""

from datetime import datetime
from os import getenv
from typing import TYPE_CHECKING

import pytest
from anthropic import Anthropic, AnthropicBedrock, NotFoundError

from stdapi.routes.anthropic_models import paginate_models
from stdapi.types.anthropic_messages import ModelInfo

if TYPE_CHECKING:
    from starlette.testclient import TestClient

# Model mappings for different test contexts
ANTHROPIC_MODEL_MAPPINGS = {
    "local": {"chat": "anthropic.claude-haiku-4-5-20251001-v1:0"},
    "anthropic": {"chat": "claude-haiku-4-5-20251001"},
}
ANTHROPIC_MODEL_MAPPINGS["bedrock"] = {
    key: f"global.{value}" for key, value in ANTHROPIC_MODEL_MAPPINGS["local"].items()
}


@pytest.fixture(scope="session")
def use_anthropic_api(request: pytest.FixtureRequest) -> bool:
    """Determine if we should use the official Anthropic API."""
    return request.config.getoption("--use-official-api")  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def anthropic_models(use_anthropic_api: bool) -> dict[str, str]:
    """Get model mappings based on test target."""
    return (
        (
            ANTHROPIC_MODEL_MAPPINGS["anthropic"]
            if getenv("ANTHROPIC_API_KEY")
            else ANTHROPIC_MODEL_MAPPINGS["bedrock"]
        )
        if use_anthropic_api
        else ANTHROPIC_MODEL_MAPPINGS["local"]
    )


@pytest.fixture(scope="session")
def anthropic_chat_model(anthropic_models: dict[str, str]) -> str:
    """Provide the appropriate chat model."""
    return anthropic_models["chat"]


@pytest.fixture(scope="session")
def anthropic_client(
    request: pytest.FixtureRequest, test_client: TestClient | None, api_key: str
) -> Anthropic:
    """Create an Anthropic client for either local or official API testing."""
    # Local test
    if test_client:
        return Anthropic(
            base_url="http://testserver/anthropic/",
            api_key=api_key,
            max_retries=5,
            http_client=test_client,
        )

    # Official API test
    if request.config.getoption("--use-official-api"):
        if getenv("ANTHROPIC_API_KEY"):
            return Anthropic(max_retries=5)
        # If no API key, try using official Anthropic client through Bedrock
        return AnthropicBedrock(max_retries=5)  # type: ignore[return-value]

    # Remote server test
    return Anthropic(
        base_url=f"{request.config.getoption('--server-url').rstrip('/')}/anthropic/",
        max_retries=5,
        api_key=getenv("OPENAI_API_KEY"),
    )


class TestAnthropicModels:
    """Test suite for the /v1/models endpoint (Anthropic API).

    Tests are designed to validate complete Anthropic Models API compatibility including:
    - Model listing and availability validation
    - Model metadata and response structure verification
    - Model retrieval by ID
    - Pagination behavior
    - Error handling for invalid models
    """

    @pytest.fixture(autouse=True)
    def _skip_bedrock(self, anthropic_client: Anthropic) -> None:
        """Skip all tests in this class when using AnthropicBedrock."""
        if isinstance(anthropic_client, AnthropicBedrock):
            pytest.skip("AnthropicBedrock client does not support the models API")

    def test_list_models_basic_functionality(self, anthropic_client: Anthropic) -> None:
        """Test fundamental model listing functionality.

        Validates:
            - Response contains models list
            - Each model has required fields
            - Response structure matches Anthropic specification
            - Models list is not empty
        """
        response = anthropic_client.models.list()

        assert hasattr(response, "data")
        assert isinstance(response.data, list)
        assert len(response.data) > 0

        # Validate each model in the list
        for model in response.data:
            assert hasattr(model, "id")
            assert hasattr(model, "type")
            assert hasattr(model, "display_name")
            assert hasattr(model, "created_at")
            assert model.type == "model"
            assert isinstance(model.id, str)
            assert len(model.id) > 0
            assert isinstance(model.display_name, str)
            assert len(model.display_name) > 0
            assert isinstance(model.created_at, (str, datetime))

    def test_list_models_response_structure_validation(
        self, anthropic_client: Anthropic
    ) -> None:
        """Test comprehensive validation of models list response structure.

        Validates:
            - All required fields are present
            - Pagination fields are present
            - Field types match specification
        """
        response = anthropic_client.models.list()

        assert hasattr(response, "data")
        assert hasattr(response, "has_more")
        assert hasattr(response, "first_id")
        assert hasattr(response, "last_id")
        assert isinstance(response.data, list)
        assert isinstance(response.has_more, bool)

        if response.data:
            assert response.first_id is not None
            assert response.last_id is not None
            assert response.first_id == response.data[0].id
            assert response.last_id == response.data[-1].id

    def test_retrieve_specific_model(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test retrieval of a specific model by ID.

        Validates:
            - Model can be retrieved by its ID
            - Response contains correct model information
            - All required fields are present
        """
        model = anthropic_client.models.retrieve(model_id=anthropic_chat_model)

        assert model.id == anthropic_chat_model
        assert model.type == "model"
        assert isinstance(model.display_name, str)
        assert len(model.display_name) > 0
        assert isinstance(model.created_at, (str, datetime))

    def test_retrieve_model_from_list(self, anthropic_client: Anthropic) -> None:
        """Test that a model from the list can be individually retrieved.

        Validates:
            - First model from list can be retrieved
            - Retrieved model matches list entry
        """
        list_response = anthropic_client.models.list()
        assert len(list_response.data) > 0

        first_model = list_response.data[0]
        retrieved = anthropic_client.models.retrieve(model_id=first_model.id)

        assert retrieved.id == first_model.id
        assert retrieved.type == first_model.type
        assert retrieved.display_name == first_model.display_name

    def test_invalid_model_retrieval_error(self, anthropic_client: Anthropic) -> None:
        """Test retrieval of a non-existent model returns an error.

        Validates:
            - Invalid model ID raises NotFoundError
        """
        with pytest.raises(NotFoundError):
            anthropic_client.models.retrieve(model_id="nonexistent-model-xyz")

    def test_list_models_pagination_with_limit(
        self, anthropic_client: Anthropic
    ) -> None:
        """Test model listing with a small limit for pagination.

        Validates:
            - Limit parameter restricts the number of returned models
            - has_more is True when more models exist
        """
        response = anthropic_client.models.list(limit=1)

        assert len(response.data) == 1
        assert response.has_more is True
        assert response.first_id is not None
        assert response.last_id is not None

    def test_list_models_pagination_after_id(self, anthropic_client: Anthropic) -> None:
        """Test cursor-based pagination using after_id.

        Validates:
            - after_id returns models after the specified cursor
            - Paginated results don't include the cursor model
        """
        # Get first page
        first_page = anthropic_client.models.list(limit=2)
        assert len(first_page.data) >= 2

        # Get second page using after_id
        assert first_page.last_id is not None
        second_page = anthropic_client.models.list(limit=2, after_id=first_page.last_id)

        # Verify no overlap
        first_page_ids = {m.id for m in first_page.data}
        for model in second_page.data:
            assert model.id not in first_page_ids

    def test_list_models_pagination_before_id(
        self, anthropic_client: Anthropic
    ) -> None:
        """Test cursor-based pagination using before_id.

        Validates:
            - before_id returns models before the specified cursor
            - Results don't include the cursor model
        """
        # Get full list to find a model in the middle
        full_list = anthropic_client.models.list(limit=1000)
        assert len(full_list.data) >= 3

        # Use the last model as before_id cursor
        last_model_id = full_list.data[-1].id
        before_page = anthropic_client.models.list(before_id=last_model_id)

        # The cursor model should not be in the results
        result_ids = {m.id for m in before_page.data}
        assert last_model_id not in result_ids

    def test_model_type_field(self, anthropic_client: Anthropic) -> None:
        """Test that all models have type 'model'.

        Validates:
            - Every model in the list has type == 'model'
        """
        response = anthropic_client.models.list()

        for model in response.data:
            assert model.type == "model"

    def test_model_created_at_format(self, anthropic_client: Anthropic) -> None:
        """Test that created_at field has a valid datetime format.

        Validates:
            - created_at is a non-empty string
            - created_at contains expected datetime characters
        """
        response = anthropic_client.models.list()
        assert len(response.data) > 0

        for model in response.data:
            if isinstance(model.created_at, str):
                # Should contain 'T' as ISO 8601 separator
                assert "T" in model.created_at
            else:
                assert isinstance(model.created_at, datetime)

    def test_list_models_ids_are_sorted(self, anthropic_client: Anthropic) -> None:
        """Test that model IDs in the list are sorted alphabetically.

        Validates:
            - Models are returned in sorted order by ID
        """
        response = anthropic_client.models.list(limit=1000)
        ids = [m.id for m in response.data]
        assert ids == sorted(ids)

    def test_list_models_consistency(self, anthropic_client: Anthropic) -> None:
        """Test that repeated calls return consistent results.

        Validates:
            - Two consecutive list calls return the same models
        """
        response1 = anthropic_client.models.list()
        response2 = anthropic_client.models.list()

        ids1 = [m.id for m in response1.data]
        ids2 = [m.id for m in response2.data]
        assert ids1 == ids2

    def test_list_models_known_model_present(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test that a known model appears in the models list.

        Validates:
            - The chat model used for testing is present in the list
        """
        response = anthropic_client.models.list(limit=1000)
        model_ids = {m.id for m in response.data}
        assert anthropic_chat_model in model_ids

    # --- Pagination edge cases ---

    def test_list_models_after_id_last_model(self, anthropic_client: Anthropic) -> None:
        """Test after_id pointing to the last model returns empty data.

        Validates:
            - Using after_id of the last model returns an empty list
            - has_more is False
        """
        full_list = anthropic_client.models.list(limit=1000)
        assert len(full_list.data) > 0

        assert full_list.last_id
        after_last = anthropic_client.models.list(after_id=full_list.last_id)
        assert len(after_last.data) == 0
        assert after_last.has_more is False

    def test_list_models_invalid_after_id(self, anthropic_client: Anthropic) -> None:
        """Test after_id with a non-existent model ID.

        Validates:
            - Non-existent after_id returns the full unfiltered list
        """
        full_list = anthropic_client.models.list(limit=1000)
        invalid_cursor = anthropic_client.models.list(
            limit=1000, after_id="nonexistent-cursor-id-xyz"
        )
        full_ids = [m.id for m in full_list.data]
        cursor_ids = [m.id for m in invalid_cursor.data]
        assert full_ids == cursor_ids

    def test_list_models_invalid_before_id(self, anthropic_client: Anthropic) -> None:
        """Test before_id with a non-existent model ID.

        Validates:
            - Non-existent before_id returns the full unfiltered list
        """
        full_list = anthropic_client.models.list(limit=1000)
        invalid_cursor = anthropic_client.models.list(
            limit=1000, before_id="nonexistent-cursor-id-xyz"
        )
        full_ids = [m.id for m in full_list.data]
        cursor_ids = [m.id for m in invalid_cursor.data]
        assert full_ids == cursor_ids

    def test_list_models_before_id_first_model(
        self, anthropic_client: Anthropic
    ) -> None:
        """Test before_id pointing to the first model returns empty data.

        Validates:
            - Using before_id of the first model returns an empty list
        """
        full_list = anthropic_client.models.list(limit=1000)
        assert len(full_list.data) > 0

        assert full_list.first_id
        before_first = anthropic_client.models.list(before_id=full_list.first_id)
        assert len(before_first.data) == 0

    def test_list_models_limit_1000(self, anthropic_client: Anthropic) -> None:
        """Test limit=1000 (maximum allowed) returns all models.

        Validates:
            - Max limit returns all available models
            - has_more is False when all models fit within limit
        """
        response = anthropic_client.models.list(limit=1000)
        assert len(response.data) > 0
        assert response.has_more is False

    def test_retrieve_model_fields_match_list(
        self, anthropic_client: Anthropic
    ) -> None:
        """Test that retrieved model fields match the corresponding list entry.

        Validates:
            - All fields from retrieve match the list entry exactly
        """
        list_response = anthropic_client.models.list(limit=1000)
        assert len(list_response.data) > 0

        # Pick a model from the middle of the list
        target = list_response.data[len(list_response.data) // 2]
        retrieved = anthropic_client.models.retrieve(model_id=target.id)

        assert retrieved.id == target.id
        assert retrieved.type == target.type
        assert retrieved.display_name == target.display_name


class TestPaginateModelsOffline:
    """Offline tests for the `paginate_models` cursor pagination helper."""

    @staticmethod
    def _models(count: int) -> list[ModelInfo]:
        """Build a list of `count` fake models named m0..m{count-1}."""
        return [
            ModelInfo(
                id=f"m{i}", created_at="1970-01-01T00:00:00Z", display_name=f"m{i}"
            )
            for i in range(count)
        ]

    def test_before_id_returns_page_immediately_preceding_cursor(self) -> None:
        """before_id must return the `limit` items adjacent to the cursor, not the oldest ones."""
        data = self._models(10)

        page, has_more = paginate_models(data, limit=3, after_id=None, before_id="m8")

        assert [m.id for m in page] == ["m5", "m6", "m7"]
        assert has_more is True

    def test_after_id_returns_page_immediately_following_cursor(self) -> None:
        """after_id must return the `limit` items immediately following the cursor."""
        data = self._models(10)

        page, has_more = paginate_models(data, limit=3, after_id="m1", before_id=None)

        assert [m.id for m in page] == ["m2", "m3", "m4"]
        assert has_more is True

    def test_before_id_no_more_items_before_page(self) -> None:
        """has_more is False when the returned page reaches the start of the list."""
        data = self._models(5)

        page, has_more = paginate_models(data, limit=3, after_id=None, before_id="m3")

        assert [m.id for m in page] == ["m0", "m1", "m2"]
        assert has_more is False

    def test_unknown_before_id_falls_back_to_start_of_list(self) -> None:
        """An unmatched before_id must not silently switch to the end of the list."""
        data = self._models(10)

        page, has_more = paginate_models(
            data, limit=3, after_id=None, before_id="unknown-cursor"
        )

        assert [m.id for m in page] == ["m0", "m1", "m2"]
        assert has_more is True
