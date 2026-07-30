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
