"""Tests for the moderation models registry and their OpenAI aliases.

Moderation models are not Bedrock foundation models: they are registered as
extra models so that they show up in the model listings with their OpenAI
aliases and are served through their class's ``moderate`` operation.

Ref: https://developers.openai.com/api/reference/resources/models/methods/list
     stdapi/models/moderation/__init__.py:initialize_moderation_models
     stdapi/aws_bedrock.py:COMPREHEND_MODERATION_MODEL
"""

from typing import TYPE_CHECKING

import pytest

from stdapi.aws import service_regions
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
    """Snapshot and restore the extra-model and alias registries.

    Ref: stdapi/models/__init__.py:update_unified_models_collections
    """
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
    """Moderation model registration and alias resolution.

    Ref: https://stdapi.ai/api_openai_moderations/
         stdapi/models/moderation/__init__.py:initialize_moderation_models
    """

    async def test_without_guardrail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only Comprehend is registered and every alias falls back to it.

        Comprehend toxicity detection is always available, so with no guardrail
        configured it also absorbs the ``omni-moderation-*`` aliases, and the
        model advertises TEXT input only (no image moderation on this backend).

        Ref: https://docs.aws.amazon.com/comprehend/latest/APIReference/API_DetectToxicContent.html
             stdapi/models/moderation/amazon_bedrock_guardrail.py:ModerationModel.get_aliases
        """
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
        assert model.input_modalities == ["TEXT"]  # No image moderation here.
        assert model.output_modalities == ["MODERATION"]
        assert model.regions == service_regions(SETTINGS.aws_comprehend_region)
        assert model.regions, "Comprehend must resolve to at least one region"
        assert sorted(model.aliases or []) == sorted(_OMNI_ALIASES + _TEXT_ALIASES)
        assert model.provider == "Amazon"
        assert model.service == "AWS Comprehend"

    async def test_comprehend_region_setting_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit aws_comprehend_region becomes the sole Comprehend model region.

        Comprehend exists in only ~13 Regions, so the deployment pins one
        explicitly and the advertised model region must follow that setting
        rather than the Bedrock Regions.

        Ref: https://docs.aws.amazon.com/comprehend/latest/dg/guidelines-and-limits.html
             stdapi/aws.py:service_regions
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", None)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", None)
        monkeypatch.setattr(SETTINGS, "aws_comprehend_region", "eu-west-1")
        await initialize_moderation_models()
        assert EXTRA_MODELS[COMPREHEND_MODERATION_MODEL].regions == ["eu-west-1"]
        assert GUARDRAIL_MODERATION_MODEL not in EXTRA_MODELS

    async def test_with_guardrail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The guardrail model is registered and takes the omni aliases.

        With a guardrail configured both backends coexist: the omni aliases move
        to the guardrail (which alone accepts images) while the legacy text
        aliases stay on Comprehend. A bare identifier has no region, so the model
        is advertised in the primary Bedrock Region.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
             stdapi/aws_bedrock.py:guardrail_region
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", "gr123")
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", "1")
        await initialize_moderation_models()
        update_unified_models_collections()
        assert COMPREHEND_MODERATION_MODEL in EXTRA_MODELS
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
        assert model.provider == "Amazon"
        assert model.service == "AWS Bedrock Runtime"
        # The Comprehend model keeps only the text aliases.
        assert sorted(
            EXTRA_MODELS[COMPREHEND_MODERATION_MODEL].aliases or []
        ) == sorted(_TEXT_ALIASES)

    async def test_guardrail_arn_sets_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A guardrail ARN registers the model in the ARN's region.

        Guardrails are regional, so the advertised Region comes from the ARN
        rather than from the primary Bedrock Region; the model listing must not
        promise the guardrail where it does not exist.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
             stdapi/aws_bedrock.py:guardrail_region
        """
        arn = "arn:aws:bedrock:eu-west-1:000000000000:guardrail/abc123"
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", arn)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", "1")
        await initialize_moderation_models()
        model = EXTRA_MODELS[GUARDRAIL_MODERATION_MODEL]
        assert model.regions == ["eu-west-1"]
        assert model.input_modalities == ["TEXT", "IMAGE"]
