"""Tests for the moderation models registry and their OpenAI aliases."""

from typing import TYPE_CHECKING

import pytest

from stdapi.aws_bedrock import COMPREHEND_MODERATION_MODEL, GUARDRAIL_MODERATION_MODEL
from stdapi.config import SETTINGS
from stdapi.models import (
    EXTRA_MODELS,
    EXTRA_MODELS_INPUT_MODALITY,
    EXTRA_MODELS_OUTPUT_MODALITY,
    MODEL_ALIASES,
    update_unified_models_collections,
)
from stdapi.models.moderation import initialize_moderation_models
from stdapi.routes import openai_moderations  # noqa: F401  (registers the route)

if TYPE_CHECKING:
    from collections.abc import Generator

#: All tests in this module exercise the local registry in-process.
pytestmark = pytest.mark.local

#: The OpenAI omni moderation model aliases (default guardrail).
_OMNI_ALIASES = ("omni-moderation-latest", "omni-moderation-2024-09-26")

#: The OpenAI text moderation model aliases (Comprehend).
_TEXT_ALIASES = ("text-moderation-latest", "text-moderation-stable")


@pytest.fixture(autouse=True)
def _isolated_registries() -> Generator[None]:
    """Snapshot and restore the extra-model and alias registries."""
    saved_models = dict(EXTRA_MODELS)
    saved_input = {
        key: set(value) for key, value in EXTRA_MODELS_INPUT_MODALITY.items()
    }
    saved_output = {
        key: set(value) for key, value in EXTRA_MODELS_OUTPUT_MODALITY.items()
    }
    saved_aliases = dict(MODEL_ALIASES)
    yield
    EXTRA_MODELS.clear()
    EXTRA_MODELS.update(saved_models)
    EXTRA_MODELS_INPUT_MODALITY.clear()
    EXTRA_MODELS_INPUT_MODALITY.update(saved_input)
    EXTRA_MODELS_OUTPUT_MODALITY.clear()
    EXTRA_MODELS_OUTPUT_MODALITY.update(saved_output)
    update_unified_models_collections()
    MODEL_ALIASES.clear()
    MODEL_ALIASES.update(saved_aliases)


class TestInitializeModerationModels:
    """Moderation model registration and alias resolution."""

    async def test_without_guardrail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only Comprehend is registered and every alias falls back to it."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", None)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", None)
        await initialize_moderation_models()
        assert COMPREHEND_MODERATION_MODEL in EXTRA_MODELS
        assert GUARDRAIL_MODERATION_MODEL not in EXTRA_MODELS
        update_unified_models_collections()
        for alias in _OMNI_ALIASES + _TEXT_ALIASES:
            assert MODEL_ALIASES[alias] == COMPREHEND_MODERATION_MODEL
        model = EXTRA_MODELS[COMPREHEND_MODERATION_MODEL]
        assert model.supported_routes == ["/v1/moderations"]
        assert model.supported_mcp_tools == ["openai_moderation"]
        assert model.input_modalities == ["TEXT"]
        assert model.output_modalities == ["MODERATION"]
        assert sorted(model.aliases or []) == sorted(_OMNI_ALIASES + _TEXT_ALIASES)

    async def test_with_guardrail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guardrail model is registered and takes the omni aliases."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", "gr123")
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", "1")
        await initialize_moderation_models()
        update_unified_models_collections()
        for alias in _OMNI_ALIASES:
            assert MODEL_ALIASES[alias] == GUARDRAIL_MODERATION_MODEL
        for alias in _TEXT_ALIASES:
            assert MODEL_ALIASES[alias] == COMPREHEND_MODERATION_MODEL
        model = EXTRA_MODELS[GUARDRAIL_MODERATION_MODEL]
        assert model.supported_routes == ["/v1/moderations"]
        assert model.supported_mcp_tools == ["openai_moderation"]
        assert model.input_modalities == ["TEXT", "IMAGE"]
        assert model.output_modalities == ["MODERATION"]
        assert model.regions == [SETTINGS.aws_bedrock_regions[0]]
        assert sorted(model.aliases or []) == sorted(_OMNI_ALIASES)

    async def test_guardrail_arn_sets_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A guardrail ARN registers the model in the ARN's region."""
        arn = "arn:aws:bedrock:eu-west-1:000000000000:guardrail/abc123"
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", arn)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", "1")
        await initialize_moderation_models()
        assert EXTRA_MODELS[GUARDRAIL_MODERATION_MODEL].regions == ["eu-west-1"]
