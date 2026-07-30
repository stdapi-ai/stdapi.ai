"""Anthropic server-tool handling on the Bedrock system-tool path (no AWS calls).

Anthropic's server tools (web search, text editor, tool search, ...) do not exist
on Bedrock, so the gateway maps them onto the model's own system tools; everything
asserted here is gateway behavior, upstream only supplies the tool and parameter
names.

Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
     https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_SystemTool.html
     stdapi/models/chat/_adapters/_anthropic_message.py:_handle_system_tool
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from stdapi.api_errors import ApiError
from stdapi.models.chat._adapters._anthropic_message import (
    _build_tool_config,
    _map_tool_choice,
)
from stdapi.types.anthropic_messages import (
    ToolChoiceAnyParam,
    ToolChoiceAutoParam,
    ToolChoiceToolParam,
    ToolSearchToolBm25Param,
    ToolSearchToolRegexParam,
    ToolTextEditorParam,
    WebSearchToolParam,
)

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


class TestToolSearchTools:
    """The BM25 and regex tool-search tools are routed down the system-tool path.

    Both types are members of ``ToolUnionParam`` and produce no ``toolSpec`` of
    their own, so ``_map_tool_spec`` returns ``None`` and ``_handle_system_tool``
    decides whether the model can serve them.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_tool_spec
    """

    @pytest.mark.parametrize(
        ("tool", "expected_name"),
        [
            (
                ToolSearchToolBm25Param(
                    type="tool_search_tool_bm25_20251119", name="tool_search_tool_bm25"
                ),
                "tool_search_tool_bm25",
            ),
            (
                ToolSearchToolRegexParam(
                    type="tool_search_tool_regex_20251119",
                    name="tool_search_tool_regex",
                ),
                "tool_search_tool_regex",
            ),
        ],
        ids=["bm25", "regex"],
    )
    def test_tool_search_tool_becomes_a_named_stub_without_a_name_map(
        self,
        tool: ToolSearchToolBm25Param | ToolSearchToolRegexParam,
        expected_name: str,
    ) -> None:
        """Without a rename map the tool keeps its Anthropic name as a schema-less stub."""
        tool_config = _build_tool_config([tool], None)
        assert tool_config is not None
        (spec,) = tool_config["tools"]
        assert spec["toolSpec"]["name"] == expected_name
        assert spec["toolSpec"]["inputSchema"] == {"json": {"type": "object"}}

    def test_tool_search_tool_rejected_by_a_model_whose_map_lacks_it(self) -> None:
        """A model with a rename map that lacks the tool gets a 400 naming the tool type.

        The map is keyed by the canonical ``ServerTools`` names, and the tool-search
        tools carry their own ``tool_search_tool_bm25``/``_regex`` names, so a model
        exposing only ``web_search`` cannot serve them.
        """
        tool = ToolSearchToolBm25Param(
            type="tool_search_tool_bm25_20251119", name="tool_search_tool_bm25"
        )
        with pytest.raises(ApiError) as excinfo:
            _build_tool_config([tool], None, tool_name_map=_NOVA_TOOL_NAME_MAP)
        assert excinfo.value.status == 400
        assert "tool_search_tool_bm25_20251119" in str(excinfo.value)
        assert "not supported by this model" in str(excinfo.value)


class TestTextEditorToolNames:
    """Both text-editor tool names are accepted and mapped to the model's Bedrock name.

    ``text_editor_20250124`` and earlier are called ``str_replace_editor``;
    ``text_editor_20250429`` and later are called ``str_replace_based_edit_tool``.
    Both spellings are ``ServerTools`` members, so a client pinned to an older
    Claude generation must not be rejected.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
         stdapi/types/anthropic_messages.py:ToolTextEditorParam
    """

    @pytest.mark.parametrize(
        ("tool_type", "tool_name"),
        [
            ("text_editor_20250124", "str_replace_editor"),
            ("text_editor_20250728", "str_replace_based_edit_tool"),
        ],
    )
    def test_editor_name_is_translated_through_the_model_name_map(
        self, tool_type: str, tool_name: str
    ) -> None:
        """Each editor name resolves through the map to the model's Bedrock tool name."""
        tool = ToolTextEditorParam(type=tool_type, name=tool_name)  # type: ignore[arg-type]
        tool_config = _build_tool_config(
            [tool],
            None,
            tool_name_map={
                "str_replace_editor": "nova_str_replace_editor",
                "str_replace_based_edit_tool": "nova_edit_tool",
            },
        )
        assert tool_config is not None
        (spec,) = tool_config["tools"]
        assert spec["toolSpec"]["name"] in {"nova_str_replace_editor", "nova_edit_tool"}

    def test_legacy_editor_name_is_rejected_by_a_model_that_only_knows_the_modern_one(
        self,
    ) -> None:
        """A map holding only the modern name rejects ``str_replace_editor`` with a 400."""
        tool = ToolTextEditorParam(
            type="text_editor_20250124", name="str_replace_editor"
        )
        with pytest.raises(ApiError) as excinfo:
            _build_tool_config(
                [tool],
                None,
                tool_name_map={"str_replace_based_edit_tool": "nova_edit_tool"},
            )
        assert excinfo.value.status == 400
        assert "text_editor_20250124" in str(excinfo.value)


class TestToolChoiceDisableParallelToolUse:
    """``disable_parallel_tool_use`` is accepted by the schema and dropped in the mapping.

    Bedrock Converse has no equivalent switch, so the flag cannot be honored; it is
    accepted rather than rejected so SDK callers that always send it keep working.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_tool_choice
    """

    def test_auto_tool_choice_drops_the_flag(self) -> None:
        """``{"type": "auto", "disable_parallel_tool_use": true}`` maps to a bare ``auto``."""
        choice = ToolChoiceAutoParam(type="auto", disable_parallel_tool_use=True)
        assert _map_tool_choice(choice) == {"auto": {}}

    def test_any_tool_choice_drops_the_flag(self) -> None:
        """``{"type": "any", "disable_parallel_tool_use": true}`` maps to a bare ``any``."""
        choice = ToolChoiceAnyParam(type="any", disable_parallel_tool_use=True)
        assert _map_tool_choice(choice) == {"any": {}}

    def test_named_tool_choice_keeps_only_the_name(self) -> None:
        """A named tool choice keeps the name and drops the flag."""
        choice = ToolChoiceToolParam(
            type="tool", name="get_weather", disable_parallel_tool_use=True
        )
        assert _map_tool_choice(choice) == {"tool": {"name": "get_weather"}}
