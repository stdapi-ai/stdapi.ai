"""Tests for the GET /search_models endpoint.

Uses unittest.mock to inject deterministic test data so these tests are fast
and do not require live AWS credentials.
"""

from decimal import Decimal
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from stdapi import pricing
from stdapi.config import SETTINGS
from stdapi.models import ModelDetails
from stdapi.pricing import Dimension, Price, PriceKey, Service
from stdapi.routes import core_models

if TYPE_CHECKING:
    from collections.abc import Generator

    from starlette.testclient import TestClient

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

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

    def test_streaming_excludes_models_with_unset_streaming_support(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """A model with response_streaming=None matches neither streaming=true nor =false.

        The route filter is an identity check (``m.response_streaming is
        streaming``), so an unset value is excluded from both results.
        """
        unset_model = ModelDetails(
            id="vendor.unset-streaming-v1",
            name="Unset Streaming",
            provider="Vendor",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            response_streaming=None,
            regions=["us-east-1"],
        )
        models = {**_FAKE_MODELS, unset_model.id: unset_model}

        async def _fake_get_all() -> tuple[
            dict[str, ModelDetails], dict[str, set[str]], dict[str, set[str]]
        ]:
            return models, _FAKE_OUTPUT_MODS, _FAKE_INPUT_MODS

        with patch(
            "stdapi.routes.core_models.get_all_models_details_and_modalities",
            new=AsyncMock(side_effect=_fake_get_all),
        ):
            true_ids = set(_get_ids(client, {"streaming": "true"}, fake_models))
            false_ids = set(_get_ids(client, {"streaming": "false"}, fake_models))

        assert unset_model.id not in true_ids
        assert unset_model.id not in false_ids


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


class TestModelService:
    """The ``service`` field is client-visible via /search_models."""

    def test_runtime_model_defaults_to_bedrock_runtime_service(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """A model with no explicit ``service`` reports the AWS Bedrock Runtime default."""
        body = _get(client, {"route": "/v1/chat/completions"}, fake_models)
        assert body[0]["service"] == "AWS Bedrock Runtime"

    def test_mantle_served_model_reports_mantle_service(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """A model served by Bedrock Mantle round-trips its service through the listing."""
        mantle_model = ModelDetails(
            id="vendor.mantle-chat-v1",
            name="Mantle Chat",
            provider="Vendor",
            service="AWS Bedrock Mantle",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            regions=["us-east-1"],
            supported_routes=["/v1/chat/completions"],
        )
        models = {**_FAKE_MODELS, mantle_model.id: mantle_model}

        async def _fake_get_all() -> tuple[
            dict[str, ModelDetails], dict[str, set[str]], dict[str, set[str]]
        ]:
            return models, _FAKE_OUTPUT_MODS, _FAKE_INPUT_MODS

        with patch(
            "stdapi.routes.core_models.get_all_models_details_and_modalities",
            new=AsyncMock(side_effect=_fake_get_all),
        ):
            body = _get(client, {"route": "/v1/chat/completions"}, fake_models)

        services = {m["id"]: m["service"] for m in body}
        assert services[mantle_model.id] == "AWS Bedrock Mantle"
        assert services["vendor.text-chat-v1"] == "AWS Bedrock Runtime"


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


# ---------------------------------------------------------------------------
# GET /model_pricing
# ---------------------------------------------------------------------------


@pytest.fixture
def priced_catalog(api_key: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Enable cost tracking, seed a small price card, and return auth headers."""
    monkeypatch.setattr(SETTINGS, "cost_tracking", True)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1"])
    rows = {
        PriceKey(
            Service.BEDROCK,
            "pricedmodel",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
        ): Price(Decimal("0.000003"), "USD"),
        PriceKey(
            Service.BEDROCK,
            "pricedmodel",
            "us-east-1",
            Dimension.OUTPUT_TOKENS,
            "standard",
        ): Price(Decimal("0.000015"), "USD"),
        PriceKey(
            Service.BEDROCK, "pricedmodel", "us-east-1", Dimension.INPUT_TOKENS, "flex"
        ): Price(Decimal("0.0000015"), "USD"),
        PriceKey(
            Service.BEDROCK,
            "pricedmodel",
            "us-east-1",
            Dimension.CACHE_WRITE_TOKENS,
            "standard",
            "5m",
        ): Price(Decimal("0.00000375"), "USD"),
    }
    # Swap (don't mutate): model_prices caches a per-index grouping by identity.
    monkeypatch.setattr(
        pricing._state,  # noqa: SLF001
        "price_index",
        {**pricing._state.price_index, **rows},  # noqa: SLF001
    )
    return {"Authorization": f"Bearer {api_key}"}


class TestModelPricingEndpoint:
    """GET /model_pricing behavior against a seeded in-memory catalog."""

    def test_missing_api_key_returns_401(self, client: TestClient) -> None:
        """Requests without credentials are rejected."""
        response = client.get("/model_pricing", params={"model": "x"})
        assert response.status_code == 401

    def test_cost_tracking_disabled_hides_settings(
        self, client: TestClient, api_key: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With cost tracking disabled, the 503 does not expose settings."""
        monkeypatch.setattr(SETTINGS, "cost_tracking", False)
        response = client.get(
            "/model_pricing",
            params={"model": "x"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert response.status_code == 503
        message = response.json()["error"]
        assert "administrator" in message
        assert "cost_tracking" not in message.lower()

    def test_catalog_not_loaded_returns_retry_later(
        self, client: TestClient, api_key: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the catalog still loading, the endpoint asks to retry later."""
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        monkeypatch.setattr(pricing._state, "price_index", {})  # noqa: SLF001
        response = client.get(
            "/model_pricing",
            params={"model": "x"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert response.status_code == 503
        assert "later" in response.json()["error"]

    def test_default_card_reflects_server_configuration(
        self, client: TestClient, priced_catalog: dict[str, str]
    ) -> None:
        """The default card keeps only the configured tier, region, and routing."""
        response = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0"},
            headers=priced_catalog,
        )
        assert response.status_code == 200
        (card,) = response.json()
        assert card["id"] == "amazon.pricedmodel-v1:0"
        assert card["service"] == "bedrock-runtime"
        assert card["default_tier"] == "standard"
        assert card["default_routings"] == ["us-east-1"]
        assert len(card["prices"]) == 3
        assert all(row["tier"] == "standard" for row in card["prices"])
        base_input = next(
            row for row in card["prices"] if row["dimension"] == "input_tokens"
        )
        assert base_input == {
            "region": "us-east-1",
            "dimension": "input_tokens",
            "tier": "standard",
            "routing": "us-east-1",
            "unit_price": "0.000003",
            "currency": "USD",
        }
        cache_row = next(
            row for row in card["prices"] if row["dimension"] == "cache_write_tokens"
        )
        assert cache_row["cache_ttl"] == "5m"

    def test_all_prices_returns_full_table(
        self, client: TestClient, priced_catalog: dict[str, str]
    ) -> None:
        """all_prices=true returns every published row with exact prices."""
        response = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0", "all_prices": "true"},
            headers=priced_catalog,
        )
        assert response.status_code == 200
        (card,) = response.json()
        assert len(card["prices"]) == 4
        flex_input = next(row for row in card["prices"] if row["tier"] == "flex")
        assert flex_input["unit_price"] == "0.0000015"
        assert all(row["routing"] == "us-east-1" for row in card["prices"])

    def test_no_model_filter_prices_every_available_model(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Omitting `model` returns one card per available model, sorted by ID."""

        async def _models() -> dict[str, ModelDetails]:
            return {
                model_id: ModelDetails(
                    id=model_id,
                    name=model_id,
                    provider="Amazon",
                    input_modalities=["TEXT"],
                    output_modalities=["TEXT"],
                    regions=["us-east-1"],
                )
                for model_id in ("amazon.pricedmodel-v1:0", "amazon.freemodel-v1:0")
            }

        monkeypatch.setattr(core_models, "get_all_models_details", _models)
        response = client.get("/model_pricing", headers=priced_catalog)
        assert response.status_code == 200
        cards = response.json()
        assert [card["id"] for card in cards] == [
            "amazon.freemodel-v1:0",
            "amazon.pricedmodel-v1:0",
        ]
        assert cards[0]["prices"] == []
        assert len(cards[1]["prices"]) == 3

    def test_default_tier_from_settings_with_fallback(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A configured default tier is shown, falling back where unpublished."""
        monkeypatch.setattr(
            SETTINGS, "default_model_service_tiers", {"amazon.pricedmodel-v1:0": "flex"}
        )
        response = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0"},
            headers=priced_catalog,
        )
        (card,) = response.json()
        assert card["default_tier"] == "flex"
        rows_by_dimension = {row["dimension"]: row for row in card["prices"]}
        assert rows_by_dimension["input_tokens"]["tier"] == "flex"
        assert rows_by_dimension["input_tokens"]["unit_price"] == "0.0000015"
        assert rows_by_dimension["output_tokens"]["tier"] == "standard"

    def test_default_card_excludes_unconfigured_regions(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rows outside the configured Bedrock regions only show on demand."""
        key = PriceKey(
            Service.BEDROCK,
            "pricedmodel",
            "eu-west-1",
            Dimension.INPUT_TOKENS,
            "standard",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            key,
            Price(Decimal("0.0000045"), "USD"),
        )
        default = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0"},
            headers=priced_catalog,
        )
        (card,) = default.json()
        assert {row["region"] for row in card["prices"]} == {"us-east-1"}
        explicit = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0", "region": "eu-west-1"},
            headers=priced_catalog,
        )
        (card,) = explicit.json()
        assert {row["region"] for row in card["prices"]} == {"eu-west-1"}

    def test_routing_from_inference_profile(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rows show the geography prefix of the model's inference profile."""
        details = ModelDetails(
            id="amazon.pricedmodel-v1:0",
            name="Priced",
            provider="Amazon",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            regions=["us-east-1"],
            inference_profiles={"us-east-1": "us.amazon.pricedmodel-v1:0"},
        )

        async def _models() -> dict[str, ModelDetails]:
            return {details.id: details}

        monkeypatch.setattr(core_models, "get_all_models_details", _models)
        response = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0"},
            headers=priced_catalog,
        )
        (card,) = response.json()
        assert card["default_routings"] == ["us"]
        assert all(row["routing"] == "us" for row in card["prices"])

    def test_default_routings_lists_each_configured_geography(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Multi-geography deployments list one routing per distinct profile."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["eu-west-3", "us-east-1"])
        details = ModelDetails(
            id="amazon.pricedmodel-v1:0",
            name="Priced",
            provider="Amazon",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            regions=["eu-west-3", "us-east-1"],
            inference_profiles={
                "eu-west-3": "eu.amazon.pricedmodel-v1:0",
                "us-east-1": "us.amazon.pricedmodel-v1:0",
            },
        )

        async def _models() -> dict[str, ModelDetails]:
            return {details.id: details}

        monkeypatch.setattr(core_models, "get_all_models_details", _models)
        response = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0"},
            headers=priced_catalog,
        )
        (card,) = response.json()
        assert card["default_routings"] == ["eu", "us"]

    def test_global_routing_preferred_with_fallback(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A global profile prefers global rows and falls back to plain ones."""
        key = PriceKey(
            Service.BEDROCK,
            "pricedmodel",
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "standard",
            "",
            "global",
        )
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            key,
            Price(Decimal("0.0000028"), "USD"),
        )
        details = ModelDetails(
            id="amazon.pricedmodel-v1:0",
            name="Priced",
            provider="Amazon",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            regions=["us-east-1"],
            inference_profiles={"us-east-1": "global.amazon.pricedmodel-v1:0"},
        )

        async def _models() -> dict[str, ModelDetails]:
            return {details.id: details}

        monkeypatch.setattr(core_models, "get_all_models_details", _models)
        response = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0"},
            headers=priced_catalog,
        )
        (card,) = response.json()
        assert card["default_routings"] == ["global"]
        rows_by_dimension = {row["dimension"]: row for row in card["prices"]}
        global_input = rows_by_dimension["input_tokens"]
        assert global_input["routing"] == "global"
        assert global_input["unit_price"] == "0.0000028"
        assert rows_by_dimension["output_tokens"]["routing"] == "us-east-1"

    def test_variants_false_with_dimension_filter(
        self, client: TestClient, priced_catalog: dict[str, str]
    ) -> None:
        """The base card composed with a dimension filter trims to one row."""
        response = client.get(
            "/model_pricing",
            params={
                "model": "amazon.pricedmodel-v1:0",
                "variants": "false",
                "dimension": "input_tokens",
            },
            headers=priced_catalog,
        )
        (card,) = response.json()
        assert [row["unit_price"] for row in card["prices"]] == ["0.000003"]

    def test_multi_model_order_dedupe_and_unknown_model(
        self, client: TestClient, priced_catalog: dict[str, str]
    ) -> None:
        """Cards come back in request order, deduplicated; unknown models are empty."""
        response = client.get(
            "/model_pricing",
            params={
                "model": [
                    "vendor.unknown-v1:0",
                    "amazon.pricedmodel-v1:0",
                    "vendor.unknown-v1:0",
                ]
            },
            headers=priced_catalog,
        )
        cards = response.json()
        assert [card["id"] for card in cards] == [
            "vendor.unknown-v1:0",
            "amazon.pricedmodel-v1:0",
        ]
        assert cards[0]["prices"] == []

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            ("tier", "cheap"),
            ("dimension", "tokens"),
            ("routing", "fast"),
            ("context", "l"),
            ("currency", "XXX"),
        ],
    )
    def test_invalid_enumerated_filter_returns_400(
        self, client: TestClient, priced_catalog: dict[str, str], param: str, value: str
    ) -> None:
        """Unknown enumerated filter values are rejected with valid values listed."""
        response = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0", param: value},
            headers=priced_catalog,
        )
        assert response.status_code == 400
        assert "Valid values" in str(response.json())

    def test_currency_filter_is_case_insensitive(
        self, client: TestClient, priced_catalog: dict[str, str]
    ) -> None:
        """A lowercase catalog currency is accepted and matches its prices."""
        response = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0", "currency": "usd"},
            headers=priced_catalog,
        )
        assert response.status_code == 200
        (card,) = response.json()
        assert card["prices"]
        assert all(row["currency"] == "USD" for row in card["prices"])

    def test_mantle_model_prices_not_duplicated(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A price registered under both Bedrock services yields one Mantle row."""
        rows = {
            PriceKey(
                service, "mantlemodel", "us-east-1", Dimension.INPUT_TOKENS, "standard"
            ): Price(Decimal("0.000002"), "USD")
            for service in (Service.BEDROCK, Service.BEDROCK_MANTLE)
        }
        monkeypatch.setattr(
            pricing._state,  # noqa: SLF001
            "price_index",
            {**pricing._state.price_index, **rows},  # noqa: SLF001
        )

        async def _models() -> dict[str, ModelDetails]:
            return {
                "amazon.mantlemodel-v1:0": ModelDetails(
                    id="amazon.mantlemodel-v1:0",
                    name="mantlemodel",
                    provider="Amazon",
                    service="AWS Bedrock Mantle",
                    input_modalities=["TEXT"],
                    output_modalities=["TEXT"],
                    regions=["us-east-1"],
                )
            }

        monkeypatch.setattr(core_models, "get_all_models_details", _models)
        response = client.get(
            "/model_pricing",
            params={"model": "amazon.mantlemodel-v1:0"},
            headers=priced_catalog,
        )
        assert response.status_code == 200
        (card,) = response.json()
        assert card["service"] == "bedrock-mantle"
        assert len(card["prices"]) == 1
