"""The gateway's own GET /search_models and GET /model_pricing routes.

Both routes are stdapi.ai extensions with no upstream analogue, so they are
exercised against an injected model catalogue and an injected price index: the
Bedrock model registry and the AWS Price List loader are patched out, which
makes every expectation below exact and offline.

Errors on these routes use the default envelope ``{"error": "<message>"}``
because the ``Models`` route tag has no provider-specific formatter.

Ref: https://stdapi.ai/api_search_models/
     https://stdapi.ai/api_model_pricing/
     stdapi/routes/core_models.py
     stdapi/api_providers/__init__.py:_default_formatter
"""

from decimal import Decimal
from re import compile as compile_regex
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr
from starlette.testclient import TestClient

from stdapi import auth, pricing
from stdapi.aws_bedrock import build_alias_overlay
from stdapi.config import SETTINGS, ModelAliasConfig
from stdapi.main import app
from stdapi.models import (
    MANTLE_SERVICE,
    MODEL_ALIAS_OVERLAYS,
    MODEL_ALIASES,
    ModelDetails,
)
from stdapi.models.capabilities import ROUTE_CAPABILITIES
from stdapi.pricing import Dimension, Price, PriceKey, Service
from stdapi.routes import core_models
from tests._helpers import make_model_details
from tests.conftest import set_test_price

if TYPE_CHECKING:
    from collections.abc import Generator

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

# ---------------------------------------------------------------------------
# Fake model catalogue used across all tests
# ---------------------------------------------------------------------------

_TEXT_MODEL = make_model_details(
    "vendor.text-chat-v1",
    name="Text Chat",
    response_streaming=True,
    legacy=False,
    supported_routes=["/v1/chat/completions"],
    supported_mcp_tools=["openai_chat"],
)

_IMAGE_MODEL = make_model_details(
    "vendor.image-gen-v1",
    name="Image Generator",
    output_modalities=["IMAGE"],
    response_streaming=False,
    legacy=None,
    regions=["us-west-2"],
    supported_routes=["/v1/images/generations"],
    supported_mcp_tools=["openai_image_gen"],
)

_SPEECH_MODEL = make_model_details(
    "vendor.speech-v1",
    name="Speech",
    input_modalities=["SPEECH"],
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
def client() -> TestClient:
    """In-process ASGI client with no app lifespan.

    Every route under test below patches out the Bedrock model registry and
    the price index (``fake_models``, ``priced_catalog``), so the app's live
    AWS startup work is never needed here. Unlike ``app_client``, no
    ``Authorization`` header is baked in: every test sets its own, including
    the 401 tests in ``TestSearchModelsAuthentication``.
    """
    from stdapi.main import app  # noqa: PLC0415

    return TestClient(app)


@pytest.fixture
async def authenticated_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Lifespan-free client backed by a freshly initialized real auth handler.

    ``client`` runs with no app lifespan, so the module-global
    ``stdapi.auth._auth_handler`` is never initialized and ``authenticate``
    accepts every request regardless of credentials. Initializing a fresh
    handler here (offline: ``SETTINGS.api_key`` only, no SSM/Secrets Manager
    call) restores real 401 enforcement for the auth tests without paying for
    the app's live AWS startup work.

    Ref: stdapi/auth.py:AuthenticationHandler
         tests/test_openai_moderations.py:test_missing_or_wrong_bearer_returns_401_envelope
    """
    from stdapi.main import app  # noqa: PLC0415

    monkeypatch.setattr(SETTINGS, "api_key", SecretStr("core-models-test-key"))
    monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
    monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
    handler = auth.AuthenticationHandler()
    assert await handler.initialize() is True
    monkeypatch.setattr(auth, "_auth_handler", handler)
    return TestClient(app)


#: Shape of the search_models inverted indexes returned by the registry.
type _SearchIndexes = tuple[
    dict[str, set[str]], dict[str, set[str]], set[str], set[str], set[str], set[str]
]


def _derive_search_indexes(models: dict[str, ModelDetails]) -> _SearchIndexes:
    """Derive the search_models inverted indexes from a catalogue like the registry does.

    Ref: stdapi/models/__init__.py:update_unified_models_collections
    """
    by_route_or_tool: dict[str, set[str]] = {}
    by_region: dict[str, set[str]] = {}
    streaming: set[str] = set()
    non_streaming: set[str] = set()
    legacy: set[str] = set()
    non_legacy: set[str] = set()
    for model_id, model in models.items():
        for route_or_tool in (
            *(model.supported_routes or ()),
            *(model.supported_mcp_tools or ()),
        ):
            by_route_or_tool.setdefault(route_or_tool, set()).add(model_id)
        for region in model.regions:
            by_region.setdefault(region, set()).add(model_id)
        if model.response_streaming is True:
            streaming.add(model_id)
        elif model.response_streaming is False:
            non_streaming.add(model_id)
        if model.legacy is True:
            legacy.add(model_id)
        else:
            non_legacy.add(model_id)
    return by_route_or_tool, by_region, streaming, non_streaming, legacy, non_legacy


@pytest.fixture
def fake_models(api_key: str) -> Generator[dict[str, str]]:
    """Serve the fake catalogue from /search_models and yield the auth headers.

    Both the lazy Bedrock registry initialisation and the catalogue accessor are
    replaced, so the route never touches AWS.
    """

    async def _noop_init() -> bool:
        return False

    async def _fake_get_all() -> tuple[
        dict[str, ModelDetails], dict[str, set[str]], dict[str, set[str]]
    ]:
        return _FAKE_MODELS, _FAKE_OUTPUT_MODS, _FAKE_INPUT_MODS

    async def _fake_get_indexes() -> _SearchIndexes:
        return _derive_search_indexes(_FAKE_MODELS)

    with (
        patch(
            "stdapi.routes.core_models.initialize_bedrock_models",
            new=AsyncMock(side_effect=_noop_init),
        ),
        patch(
            "stdapi.routes.core_models.get_all_models_details_and_modalities",
            new=AsyncMock(side_effect=_fake_get_all),
        ),
        patch(
            "stdapi.routes.core_models.get_all_models_search_indexes",
            new=AsyncMock(side_effect=_fake_get_indexes),
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
    """/search_models is gated by the shared API-key dependency.

    The 401 message is flattened to ``Unauthorized`` so a missing key and a
    wrong key are indistinguishable to the caller.

    Ref: stdapi/auth.py:AuthenticationHandler.verify_credentials
         stdapi/utils.py:hide_security_details
    """

    def test_missing_api_key_returns_401(
        self, authenticated_client: TestClient
    ) -> None:
        """GET /search_models without credentials is rejected with HTTP 401."""
        response = authenticated_client.get("/search_models")
        assert response.status_code == 401
        assert response.json() == {"error": "Unauthorized"}

    def test_invalid_api_key_returns_401(
        self, authenticated_client: TestClient
    ) -> None:
        """GET /search_models with a wrong API key is rejected with the same opaque 401."""
        response = authenticated_client.get(
            "/search_models", headers={"Authorization": "Bearer wrong-key"}
        )
        assert response.status_code == 401
        assert response.json() == {"error": "Unauthorized"}


class TestSearchModels:
    """GET /search_models without filters returns the catalogue minus legacy models.

    Ref: stdapi/routes/core_models.py:search_models
         https://github.com/stdapi-ai/stdapi.ai/issues/94
    """

    def test_returns_200(self, client: TestClient, fake_models: dict[str, str]) -> None:
        """GET /search_models returns HTTP 200 with a JSON array body."""
        response = client.get("/search_models", headers=fake_models)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_returns_non_legacy_models_sorted(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """Unfiltered response excludes legacy models, sorted by ID.

        A caller discovering models through the unfiltered listing must be able
        to invoke every model it returns, so the legacy speech model — no longer
        guaranteed invokable — is left out unless ``legacy=true`` is requested.

        Ref: https://github.com/stdapi-ai/stdapi.ai/issues/94
        """
        ids = _get_ids(client, {}, fake_models)
        assert ids == sorted({_TEXT_MODEL.id, _IMAGE_MODEL.id})
        assert _SPEECH_MODEL.legacy is True
        assert _SPEECH_MODEL.id not in ids

    def test_response_item_required_fields(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """Every catalogue entry round-trips through ``ModelDetails`` unchanged.

        The route returns registry objects directly, so each item must equal its
        catalogue entry serialised with ``exclude_none``. Both legacy and
        non-legacy entries are fetched (two calls) to cover the whole catalogue.
        """
        items = {
            str(item["id"]): item
            for params in ({}, {"legacy": "true"})
            for item in _get(client, params, fake_models)
        }
        assert set(items) == set(_FAKE_MODELS)
        for item in items.values():
            assert "id" in item
            assert "name" in item
            assert "provider" in item
            assert "input_modalities" in item
            assert "output_modalities" in item
            assert "regions" in item
        for model_id, item in items.items():
            assert item == _FAKE_MODELS[model_id].model_dump(
                mode="json", exclude_none=True
            )

    def test_none_fields_excluded(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """Unset optional fields are omitted rather than serialised as ``null``.

        The route sets ``response_model_exclude_none``, so the image model's
        ``legacy=None`` and ``inference_profiles=None`` never reach the client,
        while the fields that do have values stay present.
        """
        items = {m["id"]: m for m in _get(client, {}, fake_models)}
        legacy_items = {
            m["id"]: m for m in _get(client, {"legacy": "true"}, fake_models)
        }
        assert "legacy" not in items[_IMAGE_MODEL.id]
        assert "inference_profiles" not in items[_IMAGE_MODEL.id]
        assert items[_TEXT_MODEL.id]["legacy"] is False
        assert legacy_items[_SPEECH_MODEL.id]["legacy"] is True


class TestFilterByInputModality:
    """``input_modalities`` narrows the listing to models accepting a modality.

    Ref: stdapi/routes/core_models.py:_filter_by_modality
    """

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
        ids = set(
            _get_ids(
                client, {"input_modalities": "SPEECH", "legacy": "true"}, fake_models
            )
        )
        assert ids == {"vendor.speech-v1"}

    def test_input_modality_case_insensitive(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """A lowercase ``text`` is upper-cased before lookup and matches the same models.

        The filter normalises with ``raw.strip().upper()``, so an unknown-casing
        value would otherwise raise the 400 instead of matching.
        """
        ids_upper = set(_get_ids(client, {"input_modalities": "TEXT"}, fake_models))
        ids_lower = set(_get_ids(client, {"input_modalities": " text "}, fake_models))
        assert ids_lower == {"vendor.text-chat-v1", "vendor.image-gen-v1"}
        assert ids_upper == ids_lower

    def test_unknown_input_modality_returns_400(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """An unknown input modality is a 400 naming the rejected modality."""
        response = client.get(
            "/search_models",
            params={"input_modalities": "INVALID"},
            headers=fake_models,
        )
        assert response.status_code == 400
        assert response.json() == {
            "error": "No model matching input modality: INVALID."
        }


class TestFilterByOutputModality:
    """``output_modalities`` narrows the listing to models producing a modality.

    Ref: stdapi/routes/core_models.py:_filter_by_modality
    """

    def test_text_output_returns_text_models(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """output_modalities=TEXT returns only models producing TEXT output.

        Queried with and without ``legacy=true`` (two calls) since the legacy
        speech model is otherwise excluded from the default listing.
        """
        ids = {
            mid
            for legacy in ("false", "true")
            for mid in _get_ids(
                client, {"output_modalities": "TEXT", "legacy": legacy}, fake_models
            )
        }
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
        """An unknown output modality is a 400 that names the output direction."""
        response = client.get(
            "/search_models",
            params={"output_modalities": "INVALID"},
            headers=fake_models,
        )
        assert response.status_code == 400
        assert response.json() == {
            "error": "No model matching output modality: INVALID."
        }


class TestFilterByRoute:
    """``route`` accepts a route path or an MCP tool name interchangeably.

    A single parameter is matched against both ``supported_routes`` and
    ``supported_mcp_tools``, so agents can filter with whichever identifier they
    hold.

    Ref: stdapi/routes/core_models.py:_filter_by_route_or_tool
    """

    def test_filter_by_chat_route(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """route=/v1/chat/completions returns only chat-capable models."""
        ids = set(_get_ids(client, {"route": "/v1/chat/completions"}, fake_models))
        assert ids == {"vendor.text-chat-v1"}

    def test_filter_by_transcription_route(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """route=/v1/audio/transcriptions returns only transcription-capable models.

        Queried with ``legacy=true`` since the only transcription model is
        legacy and otherwise excluded by default.
        """
        ids = set(
            _get_ids(
                client,
                {"route": "/v1/audio/transcriptions", "legacy": "true"},
                fake_models,
            )
        )
        assert ids == {"vendor.speech-v1"}

    def test_filter_by_mcp_tool_name(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """An MCP tool name selects the same models as the equivalent route path."""
        ids = set(_get_ids(client, {"route": "openai_chat"}, fake_models))
        by_path = set(_get_ids(client, {"route": "/v1/chat/completions"}, fake_models))
        assert ids == {"vendor.text-chat-v1"}
        assert ids == by_path

    def test_filter_by_image_mcp_tool_name(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """route=openai_image_gen (MCP tool name) returns only the image generation model."""
        ids = set(_get_ids(client, {"route": "openai_image_gen"}, fake_models))
        assert ids == {"vendor.image-gen-v1"}

    def test_unknown_route_returns_400(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """An unmatched route path is a 400 naming the rejected value."""
        response = client.get(
            "/search_models", params={"route": "/v99/nonexistent"}, headers=fake_models
        )
        assert response.status_code == 400
        assert response.json() == {
            "error": "No model supporting route or MCP tool: /v99/nonexistent."
        }

    def test_unknown_mcp_tool_returns_400(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """An unmatched MCP tool name is rejected by the same combined check."""
        response = client.get(
            "/search_models", params={"route": "nonexistent_tool"}, headers=fake_models
        )
        assert response.status_code == 400
        assert response.json() == {
            "error": "No model supporting route or MCP tool: nonexistent_tool."
        }


class TestFilterByRegion:
    """``region`` narrows the listing to models accessible in one AWS region.

    Ref: stdapi/routes/core_models.py:_filter_by_region
    """

    def test_filter_by_us_east_1(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """region=us-east-1 returns models available in that region.

        Queried with and without ``legacy=true`` (two calls) since the legacy
        speech model is otherwise excluded from the default listing.
        """
        ids = {
            mid
            for legacy in ("false", "true")
            for mid in _get_ids(
                client, {"region": "us-east-1", "legacy": legacy}, fake_models
            )
        }
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
        """A secondary region in a model's ``regions`` list is matched too.

        The speech model lists ``eu-west-1`` after ``us-east-1``, so matching is
        membership-based rather than keyed on a primary region. Queried with
        ``legacy=true`` since the model is legacy and otherwise excluded by
        default.
        """
        ids = set(
            _get_ids(client, {"region": "eu-west-1", "legacy": "true"}, fake_models)
        )
        assert ids == {"vendor.speech-v1"}

    def test_unknown_region_returns_400(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """A region no model is available in is a 400 naming that region."""
        response = client.get(
            "/search_models", params={"region": "xx-invalid-99"}, headers=fake_models
        )
        assert response.status_code == 400
        assert response.json() == {
            "error": "No model available in region: xx-invalid-99."
        }


class TestFilterByStreaming:
    """``streaming`` partitions the catalogue on ``response_streaming``.

    The route uses an identity check (``m.response_streaming is streaming``), so
    only explicit ``True``/``False`` flags participate.

    Ref: stdapi/routes/core_models.py:search_models
    """

    def test_streaming_true_returns_streaming_models(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """streaming=true returns only models with response_streaming=True."""
        ids = set(_get_ids(client, {"streaming": "true"}, fake_models))
        assert ids == {"vendor.text-chat-v1"}

    def test_streaming_false_returns_non_streaming_models(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """streaming=false returns exactly the models flagged response_streaming=False.

        Queried with and without ``legacy=true`` (two calls) since the legacy
        speech model is otherwise excluded from the default listing.
        """
        ids = {
            mid
            for legacy in ("false", "true")
            for mid in _get_ids(
                client, {"streaming": "false", "legacy": legacy}, fake_models
            )
        }
        assert ids == {"vendor.image-gen-v1", "vendor.speech-v1"}

    def test_streaming_excludes_models_with_unset_streaming_support(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """A model with response_streaming=None matches neither streaming=true nor =false.

        The route filter is an identity check (``m.response_streaming is
        streaming``), so an unset value is excluded from both results while the
        models with an explicit flag still come back.
        """
        unset_model = make_model_details(
            "vendor.unset-streaming-v1", name="Unset Streaming", response_streaming=None
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
        assert "vendor.text-chat-v1" in true_ids
        assert "vendor.image-gen-v1" in false_ids


class TestFilterByLegacy:
    """``legacy`` partitions the catalogue on the deprecation flag.

    Unlike ``streaming``, the route coerces first (``(m.legacy is True) is
    legacy``), so an unset flag counts as non-legacy instead of being dropped.

    Ref: stdapi/routes/core_models.py:search_models
         https://github.com/stdapi-ai/stdapi.ai/issues/94
    """

    def test_legacy_true_returns_legacy_models(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """legacy=true returns only models with legacy=True."""
        ids = set(_get_ids(client, {"legacy": "true"}, fake_models))
        assert ids == {"vendor.speech-v1"}

    def test_legacy_false_returns_non_legacy_models(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """legacy=false returns models whose legacy flag is False or unset.

        The coercion means the pair of results is a partition: ``legacy=true``
        and ``legacy=false`` together cover the whole catalogue.
        """
        ids = set(_get_ids(client, {"legacy": "false"}, fake_models))
        legacy_ids = set(_get_ids(client, {"legacy": "true"}, fake_models))
        assert ids == {"vendor.text-chat-v1", "vendor.image-gen-v1"}
        assert ids | legacy_ids == set(_FAKE_MODELS)
        assert not ids & legacy_ids

    def test_legacy_filter_includes_none(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """A model with ``legacy=None`` is reported as non-legacy, never as legacy."""
        assert _IMAGE_MODEL.legacy is None
        ids = set(_get_ids(client, {"legacy": "false"}, fake_models))
        legacy_ids = set(_get_ids(client, {"legacy": "true"}, fake_models))
        assert _IMAGE_MODEL.id in ids
        assert _IMAGE_MODEL.id not in legacy_ids


class TestModelService:
    """The ``service`` field distinguishes Bedrock Runtime from Bedrock Mantle models.

    Ref: stdapi/models/__init__.py:ModelDetails
         stdapi/models/__init__.py:MANTLE_SERVICE
    """

    def test_runtime_model_defaults_to_bedrock_runtime_service(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """A model with no explicit ``service`` reports the AWS Bedrock Runtime default."""
        body = _get(client, {"route": "/v1/chat/completions"}, fake_models)
        assert [m["id"] for m in body] == ["vendor.text-chat-v1"]
        assert _TEXT_MODEL.service == "AWS Bedrock Runtime"
        assert body[0]["service"] == "AWS Bedrock Runtime"

    def test_mantle_served_model_reports_mantle_service(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """A Mantle-served model reports its own service alongside runtime models.

        The registry value is passed straight through, so a mixed listing shows
        both services in one response.
        """
        mantle_model = make_model_details(
            "vendor.mantle-chat-v1",
            name="Mantle Chat",
            service="AWS Bedrock Mantle",
            supported_routes=["/v1/chat/completions"],
        )
        models = {**_FAKE_MODELS, mantle_model.id: mantle_model}

        async def _fake_get_all() -> tuple[
            dict[str, ModelDetails], dict[str, set[str]], dict[str, set[str]]
        ]:
            return models, _FAKE_OUTPUT_MODS, _FAKE_INPUT_MODS

        async def _fake_get_indexes() -> _SearchIndexes:
            return _derive_search_indexes(models)

        with (
            patch(
                "stdapi.routes.core_models.get_all_models_details_and_modalities",
                new=AsyncMock(side_effect=_fake_get_all),
            ),
            patch(
                "stdapi.routes.core_models.get_all_models_search_indexes",
                new=AsyncMock(side_effect=_fake_get_indexes),
            ),
        ):
            body = _get(client, {"route": "/v1/chat/completions"}, fake_models)

        services = {m["id"]: m["service"] for m in body}
        assert services[mantle_model.id] == "AWS Bedrock Mantle"
        assert services["vendor.text-chat-v1"] == "AWS Bedrock Runtime"


class TestCombinedFilters:
    """Multiple filters combine with AND logic by intersecting the candidate set.

    Ref: stdapi/routes/core_models.py:search_models
    """

    def test_region_and_streaming(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """Region and streaming intersect rather than union.

        Both the streaming chat model and the non-streaming speech model live in
        ``us-east-1``, so a union would return two models instead of one.
        """
        ids = set(
            _get_ids(client, {"region": "us-east-1", "streaming": "true"}, fake_models)
        )
        assert ids == {"vendor.text-chat-v1"}

    def test_input_and_output_modalities(
        self, client: TestClient, fake_models: dict[str, str]
    ) -> None:
        """Input and output modality filters intersect across the two modality indexes.

        Two models take TEXT input and two produce TEXT output, so only the model
        satisfying both TEXT input and IMAGE output survives.
        """
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
        """A known-but-disjoint filter pair is an empty 200, not a 400.

        Each modality exists in the catalogue, so neither triggers the unknown-
        modality error; the empty intersection is a legitimate result.
        """
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
    set_test_price(
        "pricedmodel", "us-east-1", Dimension.INPUT_TOKENS, "0.000003", "USD"
    )
    set_test_price(
        "pricedmodel", "us-east-1", Dimension.OUTPUT_TOKENS, "0.000015", "USD"
    )
    set_test_price(
        "pricedmodel",
        "us-east-1",
        Dimension.INPUT_TOKENS,
        "0.0000015",
        "USD",
        tier="flex",
    )
    set_test_price(
        "pricedmodel",
        "us-east-1",
        Dimension.CACHE_WRITE_TOKENS,
        "0.00000375",
        "USD",
        cache_ttl="5m",
    )
    return {"Authorization": f"Bearer {api_key}"}


class TestModelPricingEndpoint:
    """GET /model_pricing against a seeded in-memory price index.

    Ref: https://stdapi.ai/api_model_pricing/
         stdapi/routes/core_models.py:model_pricing
    """

    def test_missing_api_key_returns_401(
        self, authenticated_client: TestClient
    ) -> None:
        """Authentication is checked before cost tracking, so an anonymous call is a 401.

        Ref: stdapi/auth.py:AuthenticationHandler.verify_credentials
        """
        response = authenticated_client.get("/model_pricing", params={"model": "x"})
        assert response.status_code == 401
        assert response.json() == {"error": "Unauthorized"}

    def test_cost_tracking_disabled_hides_settings(
        self, client: TestClient, api_key: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A disabled cost tracker yields a 503 that names no internal setting.

        The operator-facing setting name is logged, not returned: the client only
        learns to contact the administrator.

        Ref: stdapi/routes/core_models.py:model_pricing
        """
        monkeypatch.setattr(SETTINGS, "cost_tracking", False)
        response = client.get(
            "/model_pricing",
            params={"model": "x"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert response.status_code == 503
        message = response.json()["error"]
        assert message == (
            "Model pricing is not available on the current server. "
            "Please contact the administrator to enable it."
        )
        assert "administrator" in message
        assert "cost_tracking" not in message.lower()

    def test_catalog_not_loaded_returns_retry_later(
        self, client: TestClient, api_key: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty price index is a distinct, retryable 503 from the disabled-tracking one.

        ``price_catalog_ready()`` is False during the startup window before the
        background Price List load completes.

        Ref: stdapi/pricing.py:price_catalog_ready
        """
        monkeypatch.setattr(SETTINGS, "cost_tracking", True)
        monkeypatch.setattr(pricing._state, "price_index", {})  # noqa: SLF001
        response = client.get(
            "/model_pricing",
            params={"model": "x"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert response.status_code == 503
        assert response.json() == {
            "error": "The price catalog is not loaded yet. Please try again later."
        }

    def test_default_card_reflects_server_configuration(
        self, client: TestClient, priced_catalog: dict[str, str]
    ) -> None:
        """The default card keeps only the configured tier, region, and routing.

        The seeded catalogue also holds a ``flex`` input-token row, which the
        default (``all_prices=false``) card must drop while keeping the distinctly
        priced ``5m`` cache-write variant.

        Ref: stdapi/pricing.py:select_effective_rows
        """
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
        """all_prices=true adds back the rows the configuration-scoped card filters out.

        The default card exposes three of the four seeded rows; the extra row is
        the ``flex`` input-token variant, priced at half the standard rate.

        Ref: stdapi/pricing.py:model_prices
        """
        response = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0", "all_prices": "true"},
            headers=priced_catalog,
        )
        assert response.status_code == 200
        (card,) = response.json()
        assert len(card["prices"]) == 4
        flex_input = next(row for row in card["prices"] if row["tier"] == "flex")
        assert flex_input["dimension"] == "input_tokens"
        assert flex_input["unit_price"] == "0.0000015"
        assert {row["tier"] for row in card["prices"]} == {"standard", "flex"}
        assert all(row["routing"] == "us-east-1" for row in card["prices"])

    def test_no_model_filter_prices_every_available_model(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Omitting `model` prices every available model, sorted by ID, priced or not.

        The unpriced model still gets a card: an empty ``prices`` list means AWS
        publishes no rows, not that the model is unknown.
        """

        async def _models() -> dict[str, ModelDetails]:
            return {
                model_id: make_model_details(model_id, provider="Amazon")
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
        assert cards[0]["service"] == "bedrock-runtime"
        assert len(cards[1]["prices"]) == 3

    def test_unpriced_mantle_model_reports_mantle_service(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A Mantle-only model with zero priced rows still reports bedrock-mantle.

        With no rows to read the service from, the card falls back to the
        preferred service derived from the model's registry entry.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
             stdapi/routes/core_models.py:model_pricing
        """

        async def _models() -> dict[str, ModelDetails]:
            return {
                "openai.gpt-oss-mantle": make_model_details(
                    "openai.gpt-oss-mantle",
                    name="GPT OSS",
                    provider="OpenAI",
                    service=MANTLE_SERVICE,
                )
            }

        monkeypatch.setattr(core_models, "get_all_models_details", _models)
        response = client.get(
            "/model_pricing",
            params={"model": "openai.gpt-oss-mantle"},
            headers=priced_catalog,
        )
        assert response.status_code == 200
        (card,) = response.json()
        assert card["prices"] == []
        assert card["service"] == "bedrock-mantle"

    def test_default_tier_from_settings_with_fallback(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A per-model default tier selects its rows, falling back per dimension.

        Only ``input_tokens`` has a published ``flex`` rate, so the card mixes the
        flex input row with the standard output row — the same fallback billing
        applies.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
             stdapi/pricing.py:select_effective_rows
        """
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

    def test_alias_card_carries_the_canonical_models_rows(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A card requested by alias must price the model the alias resolves to.

        Aliases are the names OpenAI clients send, and an empty ``prices``
        list means "AWS publishes no rows" — so a card keyed on the raw alias
        would contradict the request log for exactly those names.

        Ref: stdapi/models/__init__.py:resolve_model_alias
        """
        monkeypatch.setitem(MODEL_ALIASES, "priced-alias", "amazon.pricedmodel-v1:0")
        response = client.get(
            "/model_pricing", params={"model": "priced-alias"}, headers=priced_catalog
        )
        assert response.status_code == 200
        (card,) = response.json()
        assert card["id"] == "priced-alias"
        assert len(card["prices"]) == 3

    def test_alias_card_uses_the_tier_the_alias_configures(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A card requested by alias prices the tier that alias applies.

        An alias carrying a service tier is what those requests are billed at,
        so quoting the model's own default tier would understate the price of
        exactly the name the client sends.

        Ref: https://stdapi.ai/operations_configuration/#model-aliases-configuration
             stdapi/routes/core_models.py:_pricing_defaults
        """
        monkeypatch.setitem(MODEL_ALIASES, "priced-alias", "amazon.pricedmodel-v1:0")
        monkeypatch.setitem(
            MODEL_ALIAS_OVERLAYS,
            "priced-alias",
            build_alias_overlay(
                "priced-alias",
                ModelAliasConfig(model="amazon.pricedmodel-v1:0", service_tier="flex"),
            ),
        )
        response = client.get(
            "/model_pricing", params={"model": "priced-alias"}, headers=priced_catalog
        )
        assert response.status_code == 200
        (card,) = response.json()
        assert card["default_tier"] == "flex"
        rows_by_dimension = {row["dimension"]: row for row in card["prices"]}
        assert rows_by_dimension["input_tokens"]["tier"] == "flex"

    def test_alias_card_reports_the_aliased_models_service(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An alias of a non-Bedrock model must report that model's own service.

        The card reads its service from the resolved rows, so an alias that
        prices nothing also mislabels Polly/Transcribe/Comprehend models as
        ``bedrock-runtime``.
        """
        monkeypatch.setitem(MODEL_ALIASES, "tts-1", "amazon.polly-standard")
        set_test_price(
            "pollystandard",
            "us-east-1",
            Dimension.INPUT_CHARACTERS,
            "0.000004",
            "USD",
            service=Service.POLLY,
        )
        response = client.get(
            "/model_pricing", params={"model": "tts-1"}, headers=priced_catalog
        )
        assert response.status_code == 200
        (card,) = response.json()
        assert card["service"] == "polly"
        assert [row["dimension"] for row in card["prices"]] == ["input_characters"]

    def test_default_tier_card_advertises_the_billed_fallback_rate(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A standard row kept for the flex default is advertised at the flex rate.

        Cost tracking bills an unpublished flex token rate at half the standard
        one, so advertising the unscaled 0.000015 would overstate what the
        request log records by 2x.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
             stdapi/pricing.py:_TIER_PRICE_RATIO
        """
        monkeypatch.setattr(
            SETTINGS, "default_model_service_tiers", {"amazon.pricedmodel-v1:0": "flex"}
        )
        response = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0"},
            headers=priced_catalog,
        )
        (card,) = response.json()
        rows_by_dimension = {row["dimension"]: row for row in card["prices"]}
        assert rows_by_dimension["output_tokens"]["unit_price"] == "0.0000075"
        assert rows_by_dimension["input_tokens"]["unit_price"] == "0.0000015"

    def test_default_card_excludes_unconfigured_regions(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rows outside the configured Bedrock regions only show when asked for by region.

        ``eu-west-1`` is published but not in ``aws_bedrock_regions``, so it is
        invisible by default and reachable through an explicit ``region`` filter
        (which also lifts the configured-region restriction).
        """
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
        """Rows show the geography prefix of the model's inference profile, not the region.

        The model is reached through the ``us.`` cross-Region inference profile in
        ``us-east-1``, so both the card default and each row report ``us``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
             stdapi/routes/core_models.py:_row_routing
        """
        details = make_model_details(
            "amazon.pricedmodel-v1:0",
            name="Priced",
            provider="Amazon",
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
        """Multi-geography deployments list one routing per distinct profile, in configured order.

        ``aws_bedrock_regions`` is ordered ``eu-west-3`` then ``us-east-1``, and
        the routings follow that order rather than being sorted or deduplicated
        by region.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
             stdapi/routes/core_models.py:_pricing_defaults
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["eu-west-3", "us-east-1"])
        details = make_model_details(
            "amazon.pricedmodel-v1:0",
            name="Priced",
            provider="Amazon",
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
        """A global profile prefers the global-priced row and falls back per dimension.

        Only ``input_tokens`` has a distinct ``global`` rate, so that row is billed
        at the global price while ``output_tokens`` falls back to the plain
        regional row.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
             stdapi/pricing.py:select_effective_rows
        """
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
        details = make_model_details(
            "amazon.pricedmodel-v1:0",
            name="Priced",
            provider="Amazon",
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
        """variants=false drops the cache-TTL row, and the dimension filter leaves one row.

        ``variants=false`` keeps only base rows (standard tier, no cache TTL /
        routing / long-context variant), so the ``5m`` cache-write row disappears
        even though its dimension is not filtered out.
        """
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
        assert card["prices"][0]["dimension"] == "input_tokens"
        assert card["prices"][0]["tier"] == "standard"
        assert "cache_ttl" not in card["prices"][0]

    def test_multi_model_order_dedupe_and_unknown_model(
        self, client: TestClient, priced_catalog: dict[str, str]
    ) -> None:
        """Cards come back in request order, deduplicated; unknown models are empty.

        Request order is preserved via ``dict.fromkeys``, so the unknown model
        stays first even though sorting would place it last, and its repeat is
        collapsed. An unknown model is a card with no rows, not a 404.
        """
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
        assert response.status_code == 200
        assert [card["id"] for card in cards] == [
            "vendor.unknown-v1:0",
            "amazon.pricedmodel-v1:0",
        ]
        assert cards[0]["prices"] == []
        assert cards[0]["service"] == "bedrock-runtime"
        assert len(cards[1]["prices"]) == 3

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
        """Each enumerated filter rejects an unknown value with a 400 listing the valid ones.

        ``currency`` is validated against the currencies present in the loaded
        catalog rather than a fixed list, so ``XXX`` is rejected there too.

        Ref: stdapi/routes/core_models.py:_validate_filter
             stdapi/routes/core_models.py:_validated_dimensions
        """
        response = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0", param: value},
            headers=priced_catalog,
        )
        assert response.status_code == 400
        message = response.json()["error"]
        assert message.startswith(f"Invalid {param}: "), message
        assert value in message
        assert "Valid values: " in message

    def test_currency_filter_is_case_insensitive(
        self, client: TestClient, priced_catalog: dict[str, str]
    ) -> None:
        """A lowercase currency is upper-cased before validation and matches its rows.

        Without the normalisation the value would fail the catalog-currency check
        and return a 400 instead of the priced card.

        Ref: stdapi/pricing.py:available_currencies
        """
        response = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0", "currency": "usd"},
            headers=priced_catalog,
        )
        assert response.status_code == 200
        (card,) = response.json()
        assert card["prices"]
        assert all(row["currency"] == "USD" for row in card["prices"])

    def test_tier_filter_narrows_the_card_to_that_tier(
        self, client: TestClient, priced_catalog: dict[str, str]
    ) -> None:
        """tier=flex returns only what is published at that tier, not the standard rows.

        The seeded catalogue publishes a flex rate for ``input_tokens`` alone, so an
        explicit tier must collapse the three-row default card to that single row —
        and must not rewrite the ``default_tier`` the card reports for the server.

        Ref: stdapi/pricing.py:model_prices
             stdapi/routes/core_models.py:model_pricing
        """
        response = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0", "tier": "flex"},
            headers=priced_catalog,
        )
        assert response.status_code == 200, response.text
        (card,) = response.json()
        assert [
            (row["dimension"], row["tier"], row["unit_price"]) for row in card["prices"]
        ] == [("input_tokens", "flex", "0.0000015")]
        assert card["default_tier"] == "standard"

    def test_routing_filter_narrows_the_card_to_that_serving_profile(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """routing=global returns only the distinctly priced global row.

        Without the filter the card mixes the plain regional rows with the global
        variant; asking for one serving profile must drop every row AWS does not
        publish under it, rather than falling back to the regional price.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
             stdapi/pricing.py:model_prices
        """
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            PriceKey(
                Service.BEDROCK,
                "pricedmodel",
                "us-east-1",
                Dimension.INPUT_TOKENS,
                "standard",
                "",
                "global",
            ),
            Price(Decimal("0.0000028"), "USD"),
        )
        response = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0", "routing": "global"},
            headers=priced_catalog,
        )
        assert response.status_code == 200, response.text
        (card,) = response.json()
        assert [
            (row["dimension"], row["routing"], row["unit_price"])
            for row in card["prices"]
        ] == [("input_tokens", "global", "0.0000028")]

    def test_context_filter_narrows_the_card_to_the_long_context_rows(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """context=long returns only the beyond-200K-token rate, not the standard one.

        The long-context premium is a distinct published rate on the same
        dimension, so the filter has to select on the context axis alone and leave
        the standard-context row out.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html
             stdapi/pricing.py:model_prices
        """
        monkeypatch.setitem(
            pricing._state.price_index,  # noqa: SLF001
            PriceKey(
                Service.BEDROCK,
                "pricedmodel",
                "us-east-1",
                Dimension.INPUT_TOKENS,
                "standard",
                "",
                "",
                "",
                "long",
            ),
            Price(Decimal("0.000006"), "USD"),
        )
        response = client.get(
            "/model_pricing",
            params={"model": "amazon.pricedmodel-v1:0", "context": "long"},
            headers=priced_catalog,
        )
        assert response.status_code == 200, response.text
        (card,) = response.json()
        assert [
            (row["dimension"], row["context"], row["unit_price"])
            for row in card["prices"]
        ] == [("input_tokens", "long", "0.000006")]

    def test_mantle_model_prices_not_duplicated(
        self,
        client: TestClient,
        priced_catalog: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A price published under both Bedrock services yields one row, on the Mantle service.

        Bedrock Runtime and Bedrock Mantle usage are priced independently, so the
        same dimension can appear twice in the index. The card must resolve to the
        model's preferred service instead of emitting both rows.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
             stdapi/pricing.py:model_prices
        """
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
                "amazon.mantlemodel-v1:0": make_model_details(
                    "amazon.mantlemodel-v1:0",
                    name="mantlemodel",
                    provider="Amazon",
                    service="AWS Bedrock Mantle",
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
        assert card["prices"][0]["dimension"] == "input_tokens"
        assert card["prices"][0]["unit_price"] == "0.000002"


class TestModelDiscoveryHints:
    """Every ``param=value`` hint pointing at ``search_models`` names real inputs.

    The hints in the OpenAPI descriptions are what an MCP client reads before
    picking a model, and a wrong parameter name is invisible until an agent
    calls the tool: the filter is dropped and the agent gets the unfiltered
    catalogue instead of an error. Only the quoted pairs are checked, so the
    count is asserted too -- a pattern that stops matching would otherwise
    check nothing at all.

    Ref: https://modelcontextprotocol.io/specification/server/tools
         stdapi/routes/core_models.py:search_models
    """

    #: Routes carrying a quoted hint pair, as of the current route set.
    _EXPECTED_HINTS = 18

    #: Hint pattern: a ``param=value`` pair quoted near a ``search_models`` mention.
    _HINT = compile_regex(r"`search_models`[^.]{0,120}?`([a-z_]+)=([a-zA-Z0-9_./-]+)`")

    @classmethod
    def _hints(cls, spec: dict[str, Any]) -> set[tuple[str, str]]:
        """Return every ``(parameter, value)`` pair the route descriptions suggest."""
        return {
            (match.group(1), match.group(2))
            for operation in cls._operations(spec)
            for match in cls._HINT.finditer(
                f"{operation.get('summary') or ''}{operation.get('description') or ''}"
            )
        }

    @staticmethod
    def _operations(spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Return every operation object in the OpenAPI *spec*."""
        return [
            operation
            for methods in spec["paths"].values()
            for operation in methods.values()
            if isinstance(operation, dict)
        ]

    def test_the_hinted_parameter_exists_on_the_route(self) -> None:
        """Each hinted parameter is a real ``search_models`` query parameter."""
        spec = app.openapi()
        accepted = {
            parameter["name"]
            for operation in self._operations(spec)
            if operation.get("operationId") == "search_models"
            for parameter in operation.get("parameters", ())
        }
        hints = self._hints(spec)

        assert len(hints) >= self._EXPECTED_HINTS, (
            f"only {len(hints)} hints matched; the pattern no longer finds them all"
        )
        unknown = sorted({name for name, _value in hints} - accepted)
        assert not unknown, (
            f"hints name parameters search_models does not accept: {unknown}"
        )

    def test_the_hinted_value_names_an_exposed_operation(self) -> None:
        """Each hinted value is an operation ID the catalogue can filter on.

        ``route`` accepts a path or the MCP tool name, which is the operation
        ID; a renamed route would leave the hint pointing at nothing. The
        capability registry is checked alongside the OpenAPI operations because
        a WebSocket route has no OpenAPI operation at all, yet is exactly what
        ``search_models`` filters on for it.
        """
        spec = app.openapi()
        filterable = {
            operation["operationId"]
            for operation in self._operations(spec)
            if "operationId" in operation
        } | set(ROUTE_CAPABILITIES)

        unknown = sorted(
            value for _name, value in self._hints(spec) if value not in filterable
        )
        assert not unknown, f"hints name routes the server does not expose: {unknown}"
