"""``web_search`` tool handling on the Bedrock system-tool path (no AWS calls).

Anthropic's web search server tool does not exist on Bedrock, so the gateway maps
it onto the model's own grounding system tool; everything asserted here is gateway
behavior, upstream only supplies the tool and parameter names.

Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_SystemTool.html
     stdapi/models/chat/_adapters/_anthropic_message.py:_handle_system_tool
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from stdapi.api_errors import ApiError
from stdapi.models.chat._adapters._anthropic_message import _build_tool_config
from stdapi.types.anthropic_messages import WebSearchToolParam

if TYPE_CHECKING:
    from collections.abc import Mapping

    from stdapi.types.anthropic_messages import ServerTools

pytestmark = pytest.mark.local

#: Bedrock system tool name map used to simulate a non-Claude (Nova) model.
_NOVA_TOOL_NAME_MAP: Mapping[ServerTools, str] = {"web_search": "nova_grounding"}


def test_web_search_filters_rejected_on_system_tool_path() -> None:
    """``allowed_domains`` is rejected rather than silently ignored on Nova.

    Amazon's ``systemTool`` grounding has no field to carry search filters, so
    silently dropping them would run an unrestricted search instead of the one
    the caller asked for.

    Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
    """
    tool = WebSearchToolParam(
        type="web_search_20250305", name="web_search", allowed_domains=["example.com"]
    )
    with pytest.raises(ApiError) as excinfo:
        _build_tool_config([tool], None, tool_name_map=_NOVA_TOOL_NAME_MAP)
    assert excinfo.value.status == 400
    message = str(excinfo.value)
    assert "allowed_domains" in message
    assert "not supported by this model" in message


def test_web_search_without_filters_is_promoted_on_system_tool_path() -> None:
    """A plain ``web_search`` tool (no filters) is renamed to the model's grounding tool.

    The stub carries an empty object schema; the model-specific promotion step turns
    it into a Bedrock ``systemTool`` entry afterwards.
    """
    tool = WebSearchToolParam(type="web_search_20250305", name="web_search")
    tool_config = _build_tool_config([tool], None, tool_name_map=_NOVA_TOOL_NAME_MAP)
    assert tool_config is not None
    (spec,) = tool_config["tools"]
    assert spec["toolSpec"]["name"] == "nova_grounding"
    assert spec["toolSpec"]["inputSchema"] == {"json": {"type": "object"}}
    assert "toolChoice" not in tool_config


def test_web_search_filters_allowed_on_claude_native_path() -> None:
    """On the Claude native path (empty tool_name_map), filters are not rejected.

    Claude models declare no rename map, so the tool keeps its Anthropic name and its
    filters are forwarded natively elsewhere in the Claude request pipeline.

    Ref: stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
    """
    tool = WebSearchToolParam(
        type="web_search_20250305", name="web_search", allowed_domains=["example.com"]
    )
    tool_config = _build_tool_config([tool], None, tool_name_map={})
    assert tool_config is not None
    (spec,) = tool_config["tools"]
    assert spec["toolSpec"]["name"] == "web_search"
