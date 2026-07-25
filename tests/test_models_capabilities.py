"""Tests for route-capability gating in the model registry (unit)."""

from unittest.mock import patch

import pytest

import stdapi.routes.openai_responses  # noqa: F401  (registers the input-tokens route capability)
from stdapi import models
from stdapi.models import MANTLE_SERVICE, ModelDetails, _compute_model_capabilities
from stdapi.models.capabilities import Capability
from stdapi.models.chat._default import ChatModel as ConverseChatModel
from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: MCP tool that must be gated on the COUNT_TOKENS capability.
_INPUT_TOKENS_TOOL = "openai_response_input_tokens"


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
    """The input-tokens tool is advertised only by models that can count tokens."""

    def test_converse_declares_count_tokens(self) -> None:
        """Converse chat models declare the COUNT_TOKENS capability."""
        assert ConverseChatModel.get_supported_operations() & Capability.COUNT_TOKENS

    def test_mantle_does_not_declare_count_tokens(self) -> None:
        """Mantle chat models do not declare the COUNT_TOKENS capability."""
        assert not (
            MantleChatModel.get_supported_operations() & Capability.COUNT_TOKENS
        )

    def test_converse_model_advertises_input_tokens_tool(self) -> None:
        """A Converse-served TEXT model lists the input-tokens tool."""
        with patch.object(models, "_find_model_class", return_value=ConverseChatModel):
            _, tools = _compute_model_capabilities(
                "test.model-v1:0", _text_model("AWS Bedrock Runtime")
            )
        assert _INPUT_TOKENS_TOOL in tools

    def test_mantle_model_hides_input_tokens_tool(self) -> None:
        """A Mantle-served TEXT model omits the input-tokens tool it always rejects."""
        with patch.object(models, "_find_model_class", return_value=MantleChatModel):
            _, tools = _compute_model_capabilities(
                "test.model-v1:0", _text_model(MANTLE_SERVICE)
            )
        assert _INPUT_TOKENS_TOOL not in tools
