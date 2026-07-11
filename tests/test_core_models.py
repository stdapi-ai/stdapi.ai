"""Tests for the GET /search_models endpoint.

Uses unittest.mock to inject deterministic test data so these tests are fast
and do not require live AWS credentials.
"""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from stdapi.models import ModelDetails

if TYPE_CHECKING:
    from collections.abc import Generator

    from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Fake model catalogue used across all tests
# ---------------------------------------------------------------------------

_TEXT_MODEL = ModelDetails(
    id="vendor.text-chat-v1",
    name="Text Chat",
    provider="Vendor",
    input_modalities=["TEXT"],
    output_modalities=["TEXT"],
    response_streaming=True,
    legacy=False,
    regions=["us-east-1"],
    supported_routes=["/v1/chat/completions"],
    supported_mcp_tools=["openai_chat"],
)

_IMAGE_MODEL = ModelDetails(
    id="vendor.image-gen-v1",
    name="Image Generator",
    provider="Vendor",
    input_modalities=["TEXT"],
    output_modalities=["IMAGE"],
    response_streaming=False,
    legacy=None,
    regions=["us-west-2"],
    supported_routes=["/v1/images/generations"],
    supported_mcp_tools=["openai_image_gen"],
)

_SPEECH_MODEL = ModelDetails(
    id="vendor.speech-v1",
    name="Speech",
    provider="Vendor",
    input_modalities=["SPEECH"],
    output_modalities=["TEXT"],
    response_streaming=False,
    legacy=True,
    regions=["us-east-1", "eu-west-1"],
    supported_routes=["/v1/audio/transcriptions"],
    supported_mcp_tools=["openai_audio_transcription"],
)

_FAKE_MODELS: dict[str, ModelDetails] = {
    m.id: m for m in [_IMAGE_MODEL, _SPEECH_MODEL, _TEXT_MODEL]
}
_FAKE_OUTPUT_MODS: dict[str, set[str]] = {
    "TEXT": {"vendor.text-chat-v1", "vendor.speech-v1"},
    "IMAGE": {"vendor.image-gen-v1"},
}
_FAKE_INPUT_MODS: dict[str, set[str]] = {
    "TEXT": {"vendor.text-chat-v1", "vendor.image-gen-v1"},
    "SPEECH": {"vendor.speech-v1"},
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client(test_client: TestClient | None) -> TestClient:
    """Return the session test client, skipping if not running locally."""
    if test_client is None:
        pytest.skip("Requires local test server")
    return test_client


@pytest.fixture
def fake_models(api_key: str) -> Generator[dict[str, str]]:
    """Patch the two model functions used by the /search_models route.

    Yields a dict with 'client_headers' for convenience.
    """

    async def _noop_init() -> bool:
        return False

    async def _fake_get_all() -> tuple[
        dict[str, ModelDetails], dict[str, set[str]], dict[str, set[str]]
    ]:
        return _FAKE_MODELS, _FAKE_OUTPUT_MODS, _FAKE_INPUT_MODS

    with (
        patch(
            "stdapi.routes.core_models.initialize_bedrock_models",
            new=AsyncMock(side_effect=_noop_init),
        ),
        patch(
            "stdapi.routes.core_models.get_all_models_details_and_modalities",
            new=AsyncMock(side_effect=_fake_get_all),
        ),
    ):
        yield {"Authorization": f"Bearer {api_key}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get(
    client: TestClient, params: dict[str, str], headers: dict[str, str]
) -> list[dict[str, object]]:
    """Perform GET /search_models and return the parsed JSON body as a model list."""
    return client.get("/search_models", params=params, headers=headers).json()  # type: ignore[no-any-return]


def _get_ids(
    client: TestClient, params: dict[str, str], headers: dict[str, str]
) -> list[str]:
    """Return model IDs from a filtered /search_models call."""
    return [str(m["id"]) for m in _get(client, params, headers)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSearchModelsAuthentication:
    """The /search_models route requires API-key authentication."""

    def test_missing_api_key_returns_401(self, client: TestClient) -> None:
        """GET /search_models without credentials is rejected with HTTP 401."""
        assert client.get("/search_models").status_code == 401

    def test_invalid_api_key_returns_401(self, client: TestClient) -> None:
        """GET /search_models with a wrong API key is rejected with HTTP 401."""
        response = client.get(
            "/search_models", headers={"Authorization": "Bearer wrong-key"}
        )
        assert response.status_code == 401


class TestSearchModels:
    """Tests for GET /search_models without filters."""

    def test_returns_200(self, client: TestClient, fake_models: dict[str, str]) -> None:
        """GET /search_models returns HTTP 200."""
        response = client.get("/search_models", headers=fake_models)
        assert response.status_code == 200

    def test_returns_all_models_sorted(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """Unfiltered response contains all fake models sorted by ID."""
        ids = _get_ids(client, {}, fake_models)
        assert ids == sorted(_FAKE_MODELS.keys())

    def test_response_item_required_fields(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """Each model in the response has the required ModelDetails fields."""
        for item in _get(client, {}, fake_models):
            assert "id" in item
            assert "name" in item
            assert "provider" in item
            assert "input_modalities" in item
            assert "output_modalities" in item
            assert "regions" in item

    def test_none_fields_excluded(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """Fields that are None must be absent from the response (response_model_exclude_none)."""
        items = {m["id"]: m for m in _get(client, {}, fake_models)}
        # _IMAGE_MODEL has legacy=None
        assert "legacy" not in items[_IMAGE_MODEL.id]
        # _IMAGE_MODEL has inference_profiles=None
        assert "inference_profiles" not in items[_IMAGE_MODEL.id]


class TestFilterByInputModality:
    """Tests for input_modalities query parameter."""

    def test_text_input_returns_text_and_image_models(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """input_modalities=TEXT returns only models accepting TEXT input."""
        ids = set(_get_ids(client, {"input_modalities": "TEXT"}, fake_models))
        assert ids == {"vendor.text-chat-v1", "vendor.image-gen-v1"}

    def test_speech_input_returns_speech_model(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """input_modalities=SPEECH returns only models accepting SPEECH input."""
        ids = set(_get_ids(client, {"input_modalities": "SPEECH"}, fake_models))
        assert ids == {"vendor.speech-v1"}

    def test_input_modality_case_insensitive(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """input_modalities matching is case-insensitive (lowercase 'text' works)."""
        ids_upper = set(_get_ids(client, {"input_modalities": "TEXT"}, fake_models))
        ids_lower = set(_get_ids(client, {"input_modalities": "text"}, fake_models))
        assert ids_upper == ids_lower

    def test_unknown_input_modality_returns_400(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """input_modalities=INVALID returns HTTP 400."""
        response = client.get(
            "/search_models",
            params={"input_modalities": "INVALID"},
            headers=fake_models,
        )
        assert response.status_code == 400


class TestFilterByOutputModality:
    """Tests for output_modalities query parameter."""

    def test_text_output_returns_text_models(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """output_modalities=TEXT returns only models producing TEXT output."""
        ids = set(_get_ids(client, {"output_modalities": "TEXT"}, fake_models))
        assert ids == {"vendor.text-chat-v1", "vendor.speech-v1"}

    def test_image_output_returns_image_model(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """output_modalities=IMAGE returns only models producing IMAGE output."""
        ids = set(_get_ids(client, {"output_modalities": "IMAGE"}, fake_models))
        assert ids == {"vendor.image-gen-v1"}

    def test_unknown_output_modality_returns_400(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """output_modalities=INVALID returns HTTP 400."""
        response = client.get(
            "/search_models",
            params={"output_modalities": "INVALID"},
            headers=fake_models,
        )
        assert response.status_code == 400


class TestFilterByRoute:
    """Tests for route query parameter — accepts both route paths and MCP tool names."""

    def test_filter_by_chat_route(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """route=/v1/chat/completions returns only chat-capable models."""
        ids = set(_get_ids(client, {"route": "/v1/chat/completions"}, fake_models))
        assert ids == {"vendor.text-chat-v1"}

    def test_filter_by_transcription_route(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """route=/v1/audio/transcriptions returns only transcription-capable models."""
        ids = set(_get_ids(client, {"route": "/v1/audio/transcriptions"}, fake_models))
        assert ids == {"vendor.speech-v1"}

    def test_filter_by_mcp_tool_name(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """route=openai_chat (MCP tool name) returns the same models as the route path."""
        ids = set(_get_ids(client, {"route": "openai_chat"}, fake_models))
        assert ids == {"vendor.text-chat-v1"}

    def test_filter_by_image_mcp_tool_name(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """route=openai_image_gen (MCP tool name) returns only the image generation model."""
        ids = set(_get_ids(client, {"route": "openai_image_gen"}, fake_models))
        assert ids == {"vendor.image-gen-v1"}

    def test_unknown_route_returns_400(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """route=/v99/nonexistent returns HTTP 400."""
        response = client.get(
            "/search_models", params={"route": "/v99/nonexistent"}, headers=fake_models
        )
        assert response.status_code == 400

    def test_unknown_mcp_tool_returns_400(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """route=nonexistent_tool returns HTTP 400."""
        response = client.get(
            "/search_models", params={"route": "nonexistent_tool"}, headers=fake_models
        )
        assert response.status_code == 400


class TestFilterByRegion:
    """Tests for region query parameter."""

    def test_filter_by_us_east_1(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """region=us-east-1 returns models available in that region."""
        ids = set(_get_ids(client, {"region": "us-east-1"}, fake_models))
        assert ids == {"vendor.text-chat-v1", "vendor.speech-v1"}

    def test_filter_by_us_west_2(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """region=us-west-2 returns only the model available in that region."""
        ids = set(_get_ids(client, {"region": "us-west-2"}, fake_models))
        assert ids == {"vendor.image-gen-v1"}

    def test_filter_by_eu_west_1(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """region=eu-west-1 returns models available in that region."""
        ids = set(_get_ids(client, {"region": "eu-west-1"}, fake_models))
        assert ids == {"vendor.speech-v1"}

    def test_unknown_region_returns_400(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """region=xx-invalid-99 returns HTTP 400."""
        response = client.get(
            "/search_models", params={"region": "xx-invalid-99"}, headers=fake_models
        )
        assert response.status_code == 400


class TestFilterByStreaming:
    """Tests for streaming query parameter."""

    def test_streaming_true_returns_streaming_models(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """streaming=true returns only models with response_streaming=True."""
        ids = set(_get_ids(client, {"streaming": "true"}, fake_models))
        assert ids == {"vendor.text-chat-v1"}

    def test_streaming_false_returns_non_streaming_models(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """streaming=false returns only models with response_streaming=False."""
        ids = set(_get_ids(client, {"streaming": "false"}, fake_models))
        # _IMAGE_MODEL has response_streaming=False; _SPEECH_MODEL has response_streaming=False
        # _TEXT_MODEL has response_streaming=True → excluded
        # _IMAGE_MODEL.legacy is None so (None is True) is False → included
        assert "vendor.image-gen-v1" in ids
        assert "vendor.speech-v1" in ids
        assert "vendor.text-chat-v1" not in ids

    def test_streaming_excludes_none_models(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """streaming=true excludes models where response_streaming is None."""
        # All fake models have an explicit streaming value, so this verifies
        # that only exact True matches are returned for streaming=true.
        ids = set(_get_ids(client, {"streaming": "true"}, fake_models))
        for model_id in ids:
            assert _FAKE_MODELS[model_id].response_streaming is True


class TestFilterByLegacy:
    """Tests for legacy query parameter."""

    def test_legacy_true_returns_legacy_models(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """legacy=true returns only models with legacy=True."""
        ids = set(_get_ids(client, {"legacy": "true"}, fake_models))
        assert ids == {"vendor.speech-v1"}

    def test_legacy_false_returns_non_legacy_models(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """legacy=false returns models where legacy is not True (False or None).

        The route uses ``(m.legacy is True) is legacy``, so legacy=False matches
        models whose legacy flag is either explicitly False or None.
        """
        ids = set(_get_ids(client, {"legacy": "false"}, fake_models))
        # _TEXT_MODEL (legacy=False) and _IMAGE_MODEL (legacy=None) both match
        assert ids == {"vendor.text-chat-v1", "vendor.image-gen-v1"}

    def test_legacy_filter_includes_none(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """legacy=false includes models where legacy is None.

        The route filter ``(m.legacy is True) is False`` is True for both
        legacy=False and legacy=None, so None-legacy models appear when legacy=false.
        """
        ids = set(_get_ids(client, {"legacy": "false"}, fake_models))
        assert _IMAGE_MODEL.id in ids


class TestCombinedFilters:
    """Tests that combine multiple query parameters."""

    def test_region_and_streaming(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """Combining region=us-east-1 and streaming=true returns the intersection."""
        ids = set(
            _get_ids(client, {"region": "us-east-1", "streaming": "true"}, fake_models)
        )
        assert ids == {"vendor.text-chat-v1"}

    def test_input_and_output_modalities(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """Combining input_modalities=TEXT and output_modalities=IMAGE returns image model only."""
        ids = set(
            _get_ids(
                client,
                {"input_modalities": "TEXT", "output_modalities": "IMAGE"},
                fake_models,
            )
        )
        assert ids == {"vendor.image-gen-v1"}

    def test_filters_that_yield_empty_intersection_return_empty_list(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """Filters with no matching models return an empty list (not an error)."""
        # speech input AND image output — no model has both
        response = client.get(
            "/search_models",
            params={"input_modalities": "SPEECH", "output_modalities": "IMAGE"},
            headers=fake_models,
        )
        assert response.status_code == 200
        assert response.json() == []
