"""Unit tests for the Responses tool config: ``text.format`` and server tools.

Structured Outputs are served by Bedrock's native ``outputConfig.textFormat``
JSON schema definition rather than by prompt-level instructions, so the schema
name and description supplied by the client must reach Bedrock verbatim.

Ref: https://developers.openai.com/api/docs/guides/structured-outputs
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     stdapi/models/chat/_adapters/_openai_responses.py:_build_output_config
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from stdapi.api_errors import ApiError
from stdapi.models.chat._adapters._anthropic_message import _handle_system_tool
from stdapi.models.chat._adapters._openai_responses import (
    _build_output_config,
    _resolve_integrated_tool_name,
)
from stdapi.types.anthropic_messages import WebSearchToolParam
from stdapi.types.openai_responses import ResponseTextConfig, WebSearchTool

if TYPE_CHECKING:
    from collections.abc import Mapping

    from stdapi.types.anthropic_messages import ServerTools

pytestmark = pytest.mark.local

#: Server tool map of a model serving none, as every Converse model but Nova has.
_NO_SERVER_TOOLS: Mapping[ServerTools, str] = {}

#: Server tool map of a model serving a different tool than the one requested.
_OTHER_SERVER_TOOL: Mapping[ServerTools, str] = {"code_execution": "code_interpreter"}


def test_build_output_config_returns_none_for_json_object() -> None:
    """``text.format={"type": "json_object"}`` builds no Bedrock outputConfig.

    Bedrock's strict structured output has no schema for "any JSON object": an
    empty schema is rejected and the only closed alternative,
    ``{"type": "object", "additionalProperties": false}``, admits only ``{}``,
    so the adapter must skip outputConfig entirely rather than constrain the
    model to an empty response (issue #96).
    """
    text = ResponseTextConfig.model_validate({"format": {"type": "json_object"}})
    assert _build_output_config(text) is None


def test_build_output_config_forwards_json_schema_name_and_description() -> None:
    """The client-supplied schema, name and description reach Bedrock's jsonSchema."""
    text = ResponseTextConfig.model_validate(
        {
            "format": {
                "type": "json_schema",
                "name": "weather_report",
                "description": "A weather report.",
                "schema": {"type": "object"},
            }
        }
    )
    output_config = _build_output_config(text)
    assert output_config is not None
    assert output_config["name"] == "weather_report"
    assert output_config["description"] == "A weather report."
    assert output_config["schema"] == '{"type":"object"}', (
        "the schema is forwarded as a serialised JSON string"
    )


def test_build_output_config_omits_description_when_unset() -> None:
    """An unset schema description is not forwarded to Bedrock's jsonSchema."""
    text = ResponseTextConfig.model_validate(
        {
            "format": {
                "type": "json_schema",
                "name": "weather_report",
                "schema": {"type": "object"},
            }
        }
    )
    output_config = _build_output_config(text)
    assert output_config is not None
    assert "description" not in output_config
    assert output_config["name"] == "weather_report"


class TestServerToolParity:
    """Both API surfaces answer a server tool the model cannot serve alike.

    Upstream OpenAI hosts its integrated tools, so it rejects one the model
    cannot run with a 400. Bedrock hosts them only for the models that declare
    a mapping, so the gateway forwards the tool as a stub for every other
    model: Claude requires that stub in ``toolConfig`` for Bedrock to accept
    the matching ``toolResult`` blocks in a multi-turn conversation. What must
    not diverge is the two surfaces answering the same request differently.

    Ref: https://developers.openai.com/api/docs/guides/tools-web-search
         stdapi/models/chat/_adapters/_openai_responses.py:_resolve_integrated_tool_name
         stdapi/models/chat/_adapters/_anthropic_message.py:_handle_system_tool
    """

    def test_a_model_declaring_no_server_tool_gets_the_stub(self) -> None:
        """Neither surface rejects: the Responses call site passes ``None``."""
        tool_list: list[Any] = []
        _handle_system_tool(
            WebSearchToolParam(name="web_search", type="web_search_20250305"),
            tool_list,
            tool_name_map=_NO_SERVER_TOOLS,
        )

        assert tool_list[0]["toolSpec"]["name"] == "web_search"
        assert (
            _resolve_integrated_tool_name(
                WebSearchTool(type="web_search"), _NO_SERVER_TOOLS or None
            )
            == "web_search"
        )

    def test_a_model_declaring_other_server_tools_rejects(self) -> None:
        """Both surfaces raise when the model maps tools but not this one."""
        with pytest.raises(ApiError, match="not supported by this model"):
            _handle_system_tool(
                WebSearchToolParam(name="web_search", type="web_search_20250305"),
                [],
                tool_name_map=_OTHER_SERVER_TOOL,
            )

        with pytest.raises(ApiError, match="not supported by this model"):
            _resolve_integrated_tool_name(
                WebSearchTool(type="web_search"), _OTHER_SERVER_TOOL or None
            )
