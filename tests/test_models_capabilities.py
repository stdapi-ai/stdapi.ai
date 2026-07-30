"""Route-capability gating in the model registry.

A route is advertised for a model only when the model's modalities and its
model class' capability flags both satisfy the route's ``RouteCapability``
descriptor. ``COUNT_TOKENS`` is the discriminating flag here: Bedrock
``CountTokens`` has no Mantle equivalent.

Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
     stdapi/models/__init__.py:_compute_model_capabilities
     stdapi/models/capabilities.py:RouteCapability
"""

from unittest.mock import patch

import pytest

import stdapi.routes.openai_responses  # noqa: F401  (registers the input-tokens route capability)
from stdapi import models
from stdapi.config import SETTINGS
from stdapi.models import MANTLE_SERVICE, ModelDetails, _compute_model_capabilities
from stdapi.models.capabilities import ROUTE_CAPABILITIES, Capability
from stdapi.models.chat._default import ChatModel as ConverseChatModel
from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: MCP tool that must be gated on the COUNT_TOKENS capability.
_INPUT_TOKENS_TOOL = "openai_response_input_tokens"

#: Ungated sibling tool on the same modalities, used as the gating control.
_RESPONSE_TOOL = "openai_response"


def _text_model(service: str) -> ModelDetails:
    """Build a minimal TEXT/TEXT model detail for the given hosting service.

    Args:
        service: AWS service hosting the model.

    Returns:
        A ModelDetails with TEXT input and output modalities.
    """
    return ModelDetails(
        id="test.model-v1:0",
        name="Test Model",
        provider="Test",
        service=service,
        input_modalities=["TEXT"],
        output_modalities=["TEXT"],
        regions=["us-east-1"],
    )


class TestCountTokensCapabilityGating:
    """The input-tokens tool is advertised only by models that can count tokens.

    Ref: stdapi/models/__init__.py:_compute_model_capabilities
         stdapi/routes/openai_responses.py:count_input_tokens
    """

    def test_converse_declares_count_tokens(self) -> None:
        """Converse chat models declare the COUNT_TOKENS capability.

        Ref: stdapi/models/chat/_default.py:ChatModel
        """
        assert ConverseChatModel.get_supported_operations() & Capability.COUNT_TOKENS

    def test_mantle_does_not_declare_count_tokens(self) -> None:
        """Mantle chat models do not declare the COUNT_TOKENS capability.

        Bedrock ``CountTokens`` lives on bedrock-runtime only, so the Mantle
        chat model cannot serve it.

        Ref: stdapi/models/chat/_mantle/_default.py:ChatModel
        """
        assert not (
            MantleChatModel.get_supported_operations() & Capability.COUNT_TOKENS
        )

    def test_converse_model_advertises_input_tokens_tool(self) -> None:
        """A Converse-served TEXT model lists the input-tokens tool and route."""
        with patch.object(models, "_find_model_class", return_value=ConverseChatModel):
            routes, tools = _compute_model_capabilities(
                "test.model-v1:0", _text_model("AWS Bedrock Runtime")
            )
        assert _INPUT_TOKENS_TOOL in tools
        assert ROUTE_CAPABILITIES[_INPUT_TOKENS_TOOL].path in routes

    def test_mantle_model_hides_input_tokens_tool(self) -> None:
        """A Mantle-served TEXT model omits the input-tokens tool it always rejects.

        Only the ``COUNT_TOKENS``-gated entry is dropped: the ungated Responses
        route on the same modalities stays advertised.
        """
        with patch.object(models, "_find_model_class", return_value=MantleChatModel):
            routes, tools = _compute_model_capabilities(
                "test.model-v1:0", _text_model(MANTLE_SERVICE)
            )
        assert _INPUT_TOKENS_TOOL not in tools
        assert ROUTE_CAPABILITIES[_INPUT_TOKENS_TOOL].path not in routes
        assert _RESPONSE_TOOL in tools, (
            "gating must drop only the COUNT_TOKENS route, not every TEXT route"
        )


class TestOperatorModelAliases:
    """MODEL_ALIASES rebuild: operator aliases are applied last and therefore win.

    ``MODEL_ALIASES`` is rebuilt from the model classes on every catalog refresh, so
    an operator alias only survives because ``SETTINGS.model_aliases`` is merged
    after the class-provided ones.

    Ref: stdapi/models/__init__.py:_populate_model_aliases
         stdapi/models/__init__.py:resolve_model_alias
    """

    @staticmethod
    def _rebuild(
        monkeypatch: pytest.MonkeyPatch, operator_aliases: dict[str, str]
    ) -> dict[str, ModelDetails]:
        """Rebuild the alias table from one fake class plus *operator_aliases*.

        Returns:
            The catalog the aliases were rebuilt against.
        """

        class _AliasingModel:
            @staticmethod
            def get_aliases(_all_models: dict[str, ModelDetails]) -> dict[str, str]:
                return {"fast": "other.model-v1:0"}

        all_models = {
            "test.model-v1:0": _text_model("AWS Bedrock Runtime"),
            "other.model-v1:0": _text_model("AWS Bedrock Runtime"),
        }
        all_models["other.model-v1:0"].id = "other.model-v1:0"
        monkeypatch.setattr(models, "MODEL_ALIASES", {})
        monkeypatch.setattr(models, "_GLOBAL_MODEL_REGISTRY", [_AliasingModel])
        monkeypatch.setattr(SETTINGS, "model_aliases", operator_aliases)
        models._populate_model_aliases(all_models)  # noqa: SLF001
        return all_models

    def test_class_alias_applies_without_an_operator_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no operator alias configured, the model class' own alias is used."""
        self._rebuild(monkeypatch, {})
        assert models.resolve_model_alias("fast") == "other.model-v1:0"

    def test_operator_alias_overrides_the_class_alias(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator alias reusing a built-in name redirects it to the operator target."""
        self._rebuild(monkeypatch, {"fast": "test.model-v1:0"})
        assert models.resolve_model_alias("fast") == "test.model-v1:0"

    def test_operator_alias_is_advertised_on_the_target_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator alias pointing at a catalog model is listed in its ``aliases``.

        The reverse index is built from the merged table, so ``GET /v1/models`` shows
        the operator alias and no longer shows it on the built-in target.
        """
        all_models = self._rebuild(monkeypatch, {"fast": "test.model-v1:0"})
        assert all_models["test.model-v1:0"].aliases == ["fast"]
        assert not all_models["other.model-v1:0"].aliases

    def test_unknown_alias_is_returned_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A name that is not an alias is passed through as a model ID."""
        self._rebuild(monkeypatch, {})
        assert models.resolve_model_alias("test.model-v1:0") == "test.model-v1:0"
