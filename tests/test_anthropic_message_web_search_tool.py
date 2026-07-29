"""Unit tests for web_search tool config handling on the system-tool path (no AWS calls)."""

from __future__ import annotations

import pytest

from stdapi.api_errors import ApiError
from stdapi.models.chat._adapters._anthropic_message import _build_tool_config
from stdapi.types.anthropic_messages import WebSearchToolParam

pytestmark = pytest.mark.local

#: Bedrock system tool name map used to simulate a non-Claude (Nova) model.
_NOVA_TOOL_NAME_MAP = {"web_search": "nova_grounding"}


def test_web_search_filters_rejected_on_system_tool_path() -> None:
    """``allowed_domains`` is rejected rather than silently ignored on Nova.

    Amazon's ``systemTool`` grounding has no field to carry search filters, so
    silently dropping them would run an unrestricted search instead of the one
    the caller asked for.
    """
    tool = WebSearchToolParam(
        type="web_search_20250305", name="web_search", allowed_domains=["example.com"]
    )
    with pytest.raises(ApiError):
        _build_tool_config([tool], None, tool_name_map=_NOVA_TOOL_NAME_MAP)


def test_web_search_without_filters_is_promoted_on_system_tool_path() -> None:
    """A plain ``web_search`` tool (no filters) is still promoted normally."""
    tool = WebSearchToolParam(type="web_search_20250305", name="web_search")
    tool_config = _build_tool_config([tool], None, tool_name_map=_NOVA_TOOL_NAME_MAP)
    assert tool_config is not None
    (spec,) = tool_config["tools"]
    assert spec["toolSpec"]["name"] == "nova_grounding"  # type: ignore[typeddict-item]


def test_web_search_filters_allowed_on_claude_native_path() -> None:
    """On the Claude native path (empty tool_name_map), filters are not rejected.

    They are forwarded natively elsewhere in the Claude request pipeline.
    """
    tool = WebSearchToolParam(
        type="web_search_20250305", name="web_search", allowed_domains=["example.com"]
    )
    tool_config = _build_tool_config([tool], None, tool_name_map={})
    assert tool_config is not None
    (spec,) = tool_config["tools"]
    assert spec["toolSpec"]["name"] == "web_search"  # type: ignore[typeddict-item]
