"""Tests for the Anthropic-shaped ``/v1/models`` list and retrieve routes.

Anthropic's Models endpoint is listed under "Features not supported" for Claude on
Amazon Bedrock, so the gateway synthesizes it from Bedrock model metadata. Its
``ModelInfo`` therefore carries only ``id`` / ``type`` / ``display_name`` /
``created_at`` (not upstream's newer ``capabilities`` / ``max_tokens`` fields), and
models are ordered by ID rather than by release date.

Ref: https://platform.claude.com/docs/en/api/models/list
     https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy
     stdapi/routes/anthropic_models.py:list_models
     stdapi/types/anthropic_messages.py:ModelInfo
"""

from datetime import UTC, datetime
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
    """The Anthropic Models API surface: listing, retrieval and cursor pagination.

    Ref: https://platform.claude.com/docs/en/api/models/list
         stdapi/routes/anthropic_models.py:list_models
         stdapi/routes/anthropic_models.py:retrieve_model
    """

    @pytest.fixture(autouse=True)
    def _skip_bedrock(self, anthropic_client: Anthropic) -> None:
        """Skip all tests in this class when using AnthropicBedrock."""
        if isinstance(anthropic_client, AnthropicBedrock):
            pytest.skip("AnthropicBedrock client does not support the models API")

    def test_list_models_basic_functionality(self, anthropic_client: Anthropic) -> None:
        """Listed models are distinct entries with ``type="model"``, a display name and a release date.

        The gateway derives each entry from Bedrock model metadata, keeping only models
        that declare both TEXT input and TEXT output, so an empty list means model
        discovery itself failed rather than that no models exist.

        Ref: stdapi/routes/anthropic_models.py:format_bedrock_model_to_anthropic
        """
        response = anthropic_client.models.list()

        assert isinstance(response.data, list)
        assert len(response.data) > 0, "no text-to-text model was discovered"

        ids = [model.id for model in response.data]
        assert len(set(ids)) == len(ids), f"duplicate model IDs returned: {ids}"

        for model in response.data:
            assert model.type == "model"
            assert model.id
            assert model.display_name, f"{model.id} has an empty display_name"
            assert model.created_at.tzinfo is not None, (
                f"{model.id} created_at is not an RFC 3339 instant: {model.created_at!r}"
            )

    def test_list_models_response_structure_validation(
        self, anthropic_client: Anthropic
    ) -> None:
        """The list envelope reports ``first_id`` / ``last_id`` as the edge IDs of the page.

        Anthropic's cursor pagination requires ``first_id`` and ``last_id`` to be usable
        as ``before_id`` / ``after_id`` cursors, which only holds if they are exactly the
        first and last IDs of the returned page.

        Ref: stdapi/types/anthropic_messages.py:ModelListResponse
        """
        response = anthropic_client.models.list()

        assert isinstance(response.data, list)
        assert isinstance(response.has_more, bool)
        assert len(response.data) > 0, "no text-to-text model was discovered"
        assert response.first_id == response.data[0].id
        assert response.last_id == response.data[-1].id

    def test_retrieve_specific_model(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Retrieving a known model ID echoes that ID with its metadata.

        Ref: stdapi/routes/anthropic_models.py:retrieve_model
        """
        model = anthropic_client.models.retrieve(model_id=anthropic_chat_model)

        assert model.id == anthropic_chat_model
        assert model.type == "model"
        assert model.display_name
        assert model.created_at.tzinfo is not None

    def test_retrieve_model_from_list(self, anthropic_client: Anthropic) -> None:
        """Retrieving a model taken from the list returns the identical metadata.

        The list route serves a cached snapshot while retrieve resolves the model
        through ``validate_model``; both must agree field by field.

        Ref: stdapi/routes/anthropic_models.py:retrieve_model
        """
        list_response = anthropic_client.models.list()
        assert len(list_response.data) > 0

        first_model = list_response.data[0]
        retrieved = anthropic_client.models.retrieve(model_id=first_model.id)

        assert retrieved.id == first_model.id
        assert retrieved.type == first_model.type
        assert retrieved.display_name == first_model.display_name
        assert retrieved.created_at == first_model.created_at

    def test_invalid_model_retrieval_error(self, anthropic_client: Anthropic) -> None:
        """An unknown model ID is rejected with a 404 ``not_found_error`` naming the ID.

        The gateway maps ``UnsupportedModelError`` (404) onto Anthropic's
        ``not_found_error`` type and repeats the requested ID in the message.

        Ref: https://platform.claude.com/docs/en/api/errors
             stdapi/api_providers/anthropic.py:_format_error
        """
        with pytest.raises(NotFoundError) as excinfo:
            anthropic_client.models.retrieve(model_id="nonexistent-model-xyz")

        assert excinfo.value.status_code == 404
        assert excinfo.value.type == "not_found_error"
        body = excinfo.value.body
        assert isinstance(body, dict)
        assert body["type"] == "error"
        assert "nonexistent-model-xyz" in body["error"]["message"]

    def test_list_models_pagination_with_limit(
        self, anthropic_client: Anthropic
    ) -> None:
        """``limit=1`` returns exactly one model and reports ``has_more`` true.

        With a single-item page both cursors collapse onto that model's ID, which is what
        makes it usable as the next request's ``after_id``.

        Ref: stdapi/routes/anthropic_models.py:paginate_models
        """
        response = anthropic_client.models.list(limit=1)

        assert len(response.data) == 1
        assert response.has_more is True
        assert response.first_id == response.data[0].id
        assert response.last_id == response.data[0].id

    def test_list_models_pagination_after_id(self, anthropic_client: Anthropic) -> None:
        """``after_id`` returns the page of models immediately following the cursor.

        The page is compared against the same slice of the unpaginated list, so a cursor
        that silently restarts from the beginning or skips entries fails here.

        Ref: stdapi/routes/anthropic_models.py:paginate_models
        """
        full_ids = [m.id for m in anthropic_client.models.list(limit=1000).data]
        assert len(full_ids) >= 4

        first_page = anthropic_client.models.list(limit=2)
        assert [m.id for m in first_page.data] == full_ids[:2]

        assert first_page.last_id is not None
        second_page = anthropic_client.models.list(limit=2, after_id=first_page.last_id)

        assert [m.id for m in second_page.data] == full_ids[2:4]
        assert first_page.last_id not in {m.id for m in second_page.data}

    def test_list_models_pagination_before_id(
        self, anthropic_client: Anthropic
    ) -> None:
        """``before_id`` returns the page of models ending immediately before the cursor.

        The returned page must be the contiguous slice of the unpaginated list that ends
        at the cursor — not the oldest entries — so a naive "truncate at the cursor"
        implementation fails here.

        Ref: stdapi/routes/anthropic_models.py:paginate_models
        """
        full_ids = [m.id for m in anthropic_client.models.list(limit=1000).data]
        assert len(full_ids) >= 3

        cursor_id = full_ids[-1]
        before_page = anthropic_client.models.list(before_id=cursor_id)
        before_ids = [m.id for m in before_page.data]

        assert cursor_id not in before_ids
        cursor_index = len(full_ids) - 1
        assert before_ids == full_ids[cursor_index - len(before_ids) : cursor_index]

    def test_model_type_field(self, anthropic_client: Anthropic) -> None:
        """Every listed model carries the ``"model"`` object discriminator.

        Ref: stdapi/types/anthropic_messages.py:ModelInfo
        """
        response = anthropic_client.models.list()

        assert len(response.data) > 0
        assert {model.type for model in response.data} == {"model"}

    def test_model_created_at_format(self, anthropic_client: Anthropic) -> None:
        """``created_at`` is an RFC 3339 instant, falling back to the epoch when unknown.

        The gateway formats Bedrock's ``start_of_life_time`` as ``%Y-%m-%dT%H:%M:%SZ`` and
        substitutes ``1970-01-01T00:00:00Z`` when Bedrock reports no release date, so every
        value must parse as a timezone-aware datetime at or after the epoch.

        Ref: stdapi/routes/anthropic_models.py:format_bedrock_model_to_anthropic
        """
        response = anthropic_client.models.list()
        assert len(response.data) > 0

        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        for model in response.data:
            assert model.created_at.tzinfo is not None, (
                f"{model.id} created_at lost its UTC offset: {model.created_at!r}"
            )
            assert model.created_at >= epoch, (
                f"{model.id} created_at predates the epoch fallback: {model.created_at!r}"
            )

    def test_list_models_ids_are_sorted(self, anthropic_client: Anthropic) -> None:
        """The gateway orders models by ID, not by release date.

        Upstream documents "more recently released models are listed first"; the
        synthesized list instead sorts Bedrock's model IDs lexicographically, which is what
        makes the ID cursors stable across calls.

        Ref: stdapi/routes/anthropic_models.py:list_models
        """
        response = anthropic_client.models.list(limit=1000)
        ids = [m.id for m in response.data]
        assert ids == sorted(ids)

    def test_list_models_consistency(self, anthropic_client: Anthropic) -> None:
        """Repeated list calls serve the identical cached catalog.

        The route caches the formatted list in ``_ALL_MODELS`` and only rebuilds it when
        ``initialize_bedrock_models`` reports an update, so two consecutive calls must
        agree on IDs and display names.

        Ref: stdapi/routes/anthropic_models.py:list_models
        """
        response1 = anthropic_client.models.list()
        response2 = anthropic_client.models.list()

        assert [(m.id, m.display_name) for m in response1.data] == [
            (m.id, m.display_name) for m in response2.data
        ]
        assert response1.has_more == response2.has_more

    def test_list_models_known_model_present(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """The Claude model used by the other tests is advertised by the list route.

        A model reachable through ``/v1/messages`` but absent from ``/v1/models`` would
        make the catalog useless for discovery.

        Ref: stdapi/routes/anthropic_models.py:list_models
        """
        response = anthropic_client.models.list(limit=1000)
        entries = [m for m in response.data if m.id == anthropic_chat_model]

        assert entries, f"{anthropic_chat_model} missing from the models list"
        assert entries[0].type == "model"
        assert entries[0].display_name

    # --- Pagination edge cases ---

    def test_list_models_after_id_last_model(self, anthropic_client: Anthropic) -> None:
        """``after_id`` on the last model yields an empty page with no cursors.

        Ref: stdapi/routes/anthropic_models.py:paginate_models
        """
        full_list = anthropic_client.models.list(limit=1000)
        assert len(full_list.data) > 0

        assert full_list.last_id
        after_last = anthropic_client.models.list(after_id=full_list.last_id)
        assert len(after_last.data) == 0
        assert after_last.has_more is False
        assert after_last.first_id is None
        assert after_last.last_id is None

    def test_list_models_invalid_after_id(self, anthropic_client: Anthropic) -> None:
        """An unmatched ``after_id`` is ignored and the unfiltered list is returned.

        Gateway-specific: ``paginate_models`` only slices when the cursor is found, so an
        unknown cursor degrades to "no cursor" instead of erroring or returning nothing.

        Ref: stdapi/routes/anthropic_models.py:paginate_models
        """
        full_list = anthropic_client.models.list(limit=1000)
        invalid_cursor = anthropic_client.models.list(
            limit=1000, after_id="nonexistent-cursor-id-xyz"
        )
        full_ids = [m.id for m in full_list.data]
        cursor_ids = [m.id for m in invalid_cursor.data]
        assert full_ids == cursor_ids
        assert invalid_cursor.first_id == full_list.first_id
        assert invalid_cursor.last_id == full_list.last_id
        assert invalid_cursor.has_more == full_list.has_more

    def test_list_models_invalid_before_id(self, anthropic_client: Anthropic) -> None:
        """An unmatched ``before_id`` is ignored and the unfiltered list is returned.

        Gateway-specific: an unknown ``before_id`` must not silently switch the page to the
        end of the list, which is the failure mode the offline pagination tests pin down.

        Ref: stdapi/routes/anthropic_models.py:paginate_models
        """
        full_list = anthropic_client.models.list(limit=1000)
        invalid_cursor = anthropic_client.models.list(
            limit=1000, before_id="nonexistent-cursor-id-xyz"
        )
        full_ids = [m.id for m in full_list.data]
        cursor_ids = [m.id for m in invalid_cursor.data]
        assert full_ids == cursor_ids
        assert invalid_cursor.first_id == full_list.first_id
        assert invalid_cursor.last_id == full_list.last_id
        assert invalid_cursor.has_more == full_list.has_more

    def test_list_models_before_id_first_model(
        self, anthropic_client: Anthropic
    ) -> None:
        """``before_id`` on the first model yields an empty page with no cursors.

        Ref: stdapi/routes/anthropic_models.py:paginate_models
        """
        full_list = anthropic_client.models.list(limit=1000)
        assert len(full_list.data) > 0

        assert full_list.first_id
        before_first = anthropic_client.models.list(before_id=full_list.first_id)
        assert len(before_first.data) == 0
        assert before_first.has_more is False
        assert before_first.first_id is None
        assert before_first.last_id is None

    def test_list_models_limit_1000(self, anthropic_client: Anthropic) -> None:
        """``limit=1000``, the documented maximum, returns the whole catalog in one page.

        ``has_more`` false together with a page shorter than the limit is what proves the
        catalog was not truncated.

        Ref: https://platform.claude.com/docs/en/api/models/list
             stdapi/routes/anthropic_models.py:list_models
        """
        response = anthropic_client.models.list(limit=1000)
        assert len(response.data) > 0
        assert len(response.data) <= 1000
        assert response.has_more is False

    def test_retrieve_model_fields_match_list(
        self, anthropic_client: Anthropic
    ) -> None:
        """A mid-list model retrieved by ID reports the same fields as its list entry.

        Ref: stdapi/routes/anthropic_models.py:retrieve_model
        """
        list_response = anthropic_client.models.list(limit=1000)
        assert len(list_response.data) > 0

        # Pick a model from the middle of the list
        target = list_response.data[len(list_response.data) // 2]
        retrieved = anthropic_client.models.retrieve(model_id=target.id)

        assert retrieved.id == target.id
        assert retrieved.type == target.type
        assert retrieved.display_name == target.display_name
        assert retrieved.created_at == target.created_at


class TestPaginateModelsOffline:
    """The ``paginate_models`` cursor helper, exercised in-process on a synthetic catalog.

    Ref: https://platform.claude.com/docs/en/api/models/list
         stdapi/routes/anthropic_models.py:paginate_models
    """

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
        """``before_id`` returns the ``limit`` items adjacent to the cursor, not the oldest ones.

        Anthropic's ``before_id`` means "the page immediately before this object", so the
        helper must take the *tail* of the truncated list; taking its head would return
        m0..m2 here.
        """
        data = self._models(10)

        page, has_more = paginate_models(data, limit=3, after_id=None, before_id="m8")

        assert [m.id for m in page] == ["m5", "m6", "m7"]
        assert has_more is True

    def test_after_id_returns_page_immediately_following_cursor(self) -> None:
        """``after_id`` returns the ``limit`` items immediately following the cursor.

        The cursor item itself is excluded: paging starts at index+1.
        """
        data = self._models(10)

        page, has_more = paginate_models(data, limit=3, after_id="m1", before_id=None)

        assert [m.id for m in page] == ["m2", "m3", "m4"]
        assert has_more is True

    def test_before_id_no_more_items_before_page(self) -> None:
        """``has_more`` is False when the ``before_id`` page reaches the start of the list.

        Only 3 items precede the m3 cursor and the limit is 3, so nothing remains beyond
        the page.
        """
        data = self._models(5)

        page, has_more = paginate_models(data, limit=3, after_id=None, before_id="m3")

        assert [m.id for m in page] == ["m0", "m1", "m2"]
        assert has_more is False

    def test_unknown_before_id_falls_back_to_start_of_list(self) -> None:
        """An unmatched ``before_id`` is ignored rather than switching to the end of the list.

        The cursor lookup returns None, so no slicing happens and the first ``limit`` items
        are served — the same page an unpaginated request would return.
        """
        data = self._models(10)

        page, has_more = paginate_models(
            data, limit=3, after_id=None, before_id="unknown-cursor"
        )

        assert [m.id for m in page] == ["m0", "m1", "m2"]
        assert has_more is True
