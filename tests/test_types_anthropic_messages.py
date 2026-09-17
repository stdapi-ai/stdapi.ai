"""Anthropic Messages type mirror against the Anthropic SDK types (no AWS calls).

The gateway never runs the ``web_fetch`` server tool itself, so a
``web_fetch_tool_result`` block can only arrive as conversation history replayed by a
client that talked to the Messages API. Every error code the SDK declares therefore
has to validate, otherwise the whole request is refused mid-conversation.

The same holds for the browser and computer toolsets and the ``browser_state`` block
they answer with: they discriminate a union arm rather than add a field, so a shape
the mirror does not model rejects the entire request instead of the entry.

Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool
     https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/web_fetch_tool_result_error_code.py
     stdapi/types/anthropic_messages.py:WebFetchToolResultErrorCode
"""

from typing import get_args

import pytest
from anthropic.types import (
    WebFetchToolResultErrorCode as SdkWebFetchToolResultErrorCode,
from anthropic.types import BrowserStateChangeParam as SdkBrowserStateChangeParam
from anthropic.types import BrowserToolset20260801Param as SdkBrowserToolsetParam
from anthropic.types import BrowserToolsetConfigsParam as SdkBrowserToolsetConfigsParam
from anthropic.types import ComputerToolset20260801Param as SdkComputerToolsetParam
from anthropic.types import (
    ComputerToolsetConfigsParam as SdkComputerToolsetConfigsParam,
)
)

from stdapi.types.anthropic_messages import (
    MessageCreateParams,
    WebFetchToolResultErrorBlock,
    WebFetchToolResultErrorCode,
)
    BrowserStateBlockParam,
    BrowserToolsetParam,
    ComputerToolsetParam,
    MessageCountTokensParams,

    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Every ``web_fetch`` tool-result error code the installed Anthropic SDK declares.
SDK_WEB_FETCH_ERROR_CODES = get_args(SdkWebFetchToolResultErrorCode)


class TestWebFetchToolResultErrorCode:
    """The ``web_fetch_tool_result_error`` code enumeration.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool
def _sdk_type_literal(typed_dict: Any) -> str:  # noqa: ANN401 (a TypedDict class)
    """Return the single ``type`` literal an SDK ``TypedDict`` declares.

    Args:
        typed_dict: SDK parameter ``TypedDict`` carrying a ``type`` discriminator.

    Returns:
        The discriminator value, read off the installed SDK so a version bump
        shows up as a failure here rather than as a rejected request.
    """
    hint = get_type_hints(typed_dict, include_extras=True)["type"]
    # ``type`` is spelled ``Required[Literal[...]]`` on every param TypedDict.
    (literal,) = get_args(hint)
    return str(get_args(literal)[0])


#: ``tools`` entry type the installed SDK gives the browser toolset.
SDK_BROWSER_TOOLSET_TYPE = _sdk_type_literal(SdkBrowserToolsetParam)

#: ``tools`` entry type the installed SDK gives the computer toolset.
SDK_COMPUTER_TOOLSET_TYPE = _sdk_type_literal(SdkComputerToolsetParam)

#: Each toolset entry type, with the mirrored model a request must resolve it to.
SDK_TOOLSETS = (
    (SDK_BROWSER_TOOLSET_TYPE, BrowserToolsetParam),
    (SDK_COMPUTER_TOOLSET_TYPE, ComputerToolsetParam),
)

#: Every (toolset type, member name) pair the installed SDK declares a config for.
SDK_TOOLSET_MEMBERS = tuple(
    (toolset_type, member)
    for toolset_type, configs in (
        (SDK_BROWSER_TOOLSET_TYPE, SdkBrowserToolsetConfigsParam),
        (SDK_COMPUTER_TOOLSET_TYPE, SdkComputerToolsetConfigsParam),
    )
    for member in get_type_hints(configs)
)

#: Every ``browser_state`` state-change variant the installed SDK declares.
SDK_STATE_CHANGE_TYPES = tuple(
    _sdk_type_literal(variant) for variant in get_args(SdkBrowserStateChangeParam)
)

#: One minimal valid body per ``browser_state`` state-change variant.
STATE_CHANGES: dict[str, dict[str, object]] = {
    "tab_opened": {"type": "tab_opened", "tab_id": "tab-2"},
    "download_started": {
        "type": "download_started",
        "download_id": "dl-1",
        "url": "https://example.com/report.pdf",
    },
    "download_completed": {
        "type": "download_completed",
        "download_id": "dl-1",
        "url": "https://example.com/report.pdf",
        "path": "/downloads/report.pdf",
        "size_bytes": 2048,
    },
    "download_failed": {
        "type": "download_failed",
        "download_id": "dl-2",
        "url": "https://example.com/missing.pdf",
        "error": "not found",
    },
}


def _create_params(**fields: object) -> MessageCreateParams:
    """Build a minimal create-message request carrying *fields*."""
    return MessageCreateParams.model_validate(
        {
            "model": "claude-x",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            **fields,
        }
    )


         stdapi/types/anthropic_messages.py:WebFetchToolResultErrorCode
    """

    def test_codes_match_the_sdk_exactly(self) -> None:
        """The mirrored literal declares the same codes as the SDK.

        Ref: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/web_fetch_tool_result_error_code.py
             stdapi/types/anthropic_messages.py:WebFetchToolResultErrorCode
        """
        assert set(SDK_WEB_FETCH_ERROR_CODES)
        assert set(get_args(WebFetchToolResultErrorCode)) == set(
            SDK_WEB_FETCH_ERROR_CODES
        )

    @pytest.mark.parametrize("error_code", SDK_WEB_FETCH_ERROR_CODES)
    def test_a_replayed_error_block_is_accepted(self, error_code: str) -> None:
        """A request replaying a ``web_fetch`` error block validates for every code.

        The block is one arm of the ``MessageParam.content`` union, so an unknown
        code fails every arm and the entire request is rejected rather than the
        block alone.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool
             stdapi/types/anthropic_messages.py:WebFetchToolResultErrorBlockParam
        """
        params = MessageCreateParams.model_validate(
            {
                "model": "claude-x",
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "server_tool_use",
                                "id": "srvtoolu_1",
                                "name": "web_fetch",
                                "input": {"url": "https://example.com/alpha"},
                            },
                            {
                                "type": "web_fetch_tool_result",
                                "tool_use_id": "srvtoolu_1",
                                "content": {
                                    "type": "web_fetch_tool_result_error",
                                    "error_code": error_code,
                                },
                            },
                        ],
                    }
                ],
            }
        )

        content = params.messages[0].content
        assert not isinstance(content, str)
        assert content[1].content.error_code == error_code  # type: ignore[union-attr]

    @pytest.mark.parametrize("error_code", SDK_WEB_FETCH_ERROR_CODES)
    def test_a_returned_error_block_is_accepted(self, error_code: str) -> None:
        """The response-side block validates for every code the API can emit.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool
             stdapi/types/anthropic_messages.py:WebFetchToolResultErrorBlock
        """
        block = WebFetchToolResultErrorBlock.model_validate(
            {"type": "web_fetch_tool_result_error", "error_code": error_code}
        )

        assert block.error_code == error_code


class TestBrowserAndComputerToolsets:
    """The browser and computer toolsets as ``tools`` entries.

    A toolset declares a whole tool family in one entry: it carries no ``name``
    and no ``input_schema``, only ``type``, ``configs`` and ``cache_control``.
    It therefore has to be its own arm of the ``tools`` union -- an unmodelled
    entry fails every other arm and the whole request is refused, which is what
    a computer-use or browser-use client built on the current SDK sends.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/types/anthropic_messages.py:ToolUnionParam
    """

    @pytest.mark.parametrize(("toolset_type", "model"), SDK_TOOLSETS)
    def test_a_toolset_entry_resolves_to_its_own_type(
        self, toolset_type: str, model: type[BrowserToolsetParam | ComputerToolsetParam]
    ) -> None:
        """Each toolset the SDK can send validates as the matching mirrored model.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/types/anthropic_messages.py:ToolUnionParam
        """
        request = _create_params(tools=[{"type": toolset_type}])

        assert request.tools is not None
        (tool,) = request.tools
        assert isinstance(tool, model)
        assert tool.type == toolset_type

    @pytest.mark.parametrize(("toolset_type", "_model"), SDK_TOOLSETS)
    def test_a_toolset_is_kept_rather_than_dropped(
        self,
        toolset_type: str,
        _model: type[BrowserToolsetParam | ComputerToolsetParam],
    ) -> None:
        """A toolset survives validation and is serialized as it was sent.

        Unlike an ``mcp_toolset``, which names tools no model here can ever
        call, a toolset is a real tool family a passthrough backend can run --
        so it must reach the backend instead of being removed.

        Ref: stdapi/types/anthropic_messages.py:_without_mcp_toolsets
        """
        request = _create_params(
            tools=[{"type": toolset_type, "configs": {"screenshot": {"enabled": True}}}]
        )

        assert request.model_dump(exclude_unset=True)["tools"] == [
            {"type": toolset_type, "configs": {"screenshot": {"enabled": True}}}
        ]

    @pytest.mark.parametrize(("toolset_type", "member"), SDK_TOOLSET_MEMBERS)
    def test_every_member_config_the_sdk_declares_is_accepted(
        self, toolset_type: str, member: str
    ) -> None:
        """Every member name of both toolsets is an accepted ``configs`` key.

        ``configs`` is keyed by member name and both families together declare
        dozens of them, so a request disabling any one member must validate.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/types/anthropic_messages.py:ToolsetMemberConfigParam
        """
        request = _create_params(
            tools=[
                {
                    "type": toolset_type,
                    "configs": {member: {"enabled": False, "defer_loading": True}},
                }
            ]
        )

        assert request.tools is not None
        tool = request.tools[0]
        assert isinstance(tool, BrowserToolsetParam | ComputerToolsetParam)
        assert tool.configs is not None
        assert tool.configs[member].enabled is False
        assert tool.configs[member].defer_loading is True

    @pytest.mark.parametrize(("toolset_type", "_model"), SDK_TOOLSETS)
    def test_a_later_toolset_version_is_accepted(
        self,
        toolset_type: str,
        _model: type[BrowserToolsetParam | ComputerToolsetParam],
    ) -> None:
        """A dated toolset type newer than the installed SDK's still validates.

        Toolset types are versioned by date, and a client tracking the API ahead
        of this mirror must not be refused for the version alone.

        Ref: stdapi/types/anthropic_messages.py:BrowserToolsetParam
        """
        future_type = f"{toolset_type.rsplit('_', 1)[0]}_29991231"
        request = _create_params(tools=[{"type": future_type}])

        assert request.tools is not None
        assert request.tools[0].type == future_type

    @pytest.mark.parametrize(("toolset_type", "_model"), SDK_TOOLSETS)
    def test_a_cache_control_breakpoint_is_accepted_on_a_toolset(
        self,
        toolset_type: str,
        _model: type[BrowserToolsetParam | ComputerToolsetParam],
    ) -> None:
        """A toolset can carry the cache breakpoint the SDK allows on it.

        Ref: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
             stdapi/types/anthropic_messages.py:BrowserToolsetParam
        """
        request = _create_params(
            tools=[{"type": toolset_type, "cache_control": {"type": "ephemeral"}}]
        )

        assert request.tools is not None
        tool = request.tools[0]
        assert isinstance(tool, BrowserToolsetParam | ComputerToolsetParam)
        assert tool.cache_control is not None
        assert tool.cache_control.type == "ephemeral"

    @pytest.mark.parametrize(("toolset_type", "_model"), SDK_TOOLSETS)
    def test_a_toolset_is_accepted_on_the_count_tokens_body(
        self,
        toolset_type: str,
        _model: type[BrowserToolsetParam | ComputerToolsetParam],
    ) -> None:
        """The count-tokens body takes the same ``tools`` entries as the create body.

        Both bodies share one tool union upstream, so a client counting the
        tokens of a request it is about to send must not be refused for a
        toolset the create body accepts.

        Ref: https://platform.claude.com/docs/en/api/messages/count_tokens
             stdapi/types/anthropic_messages.py:MessageCountTokensParams
        """
        request = MessageCountTokensParams.model_validate(
            {
                "model": "claude-x",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": toolset_type}],
            }
        )

        assert request.tools is not None
        assert request.tools[0].type == toolset_type

    @pytest.mark.parametrize(("toolset_type", "_model"), SDK_TOOLSETS)
    def test_a_tool_choice_beside_a_toolset_is_kept(
        self,
        toolset_type: str,
        _model: type[BrowserToolsetParam | ComputerToolsetParam],
    ) -> None:
        """A forced choice survives a request whose only tool is a toolset.

        The choice is dropped only when every tool was an ``mcp_toolset``, which
        is removed; a toolset stays, so the choice still has something to select
        from and dropping it would silence the request's instruction.

        Ref: stdapi/types/anthropic_messages.py:_without_orphaned_tool_choice
        """
        request = _create_params(
            tools=[{"type": toolset_type}], tool_choice={"type": "any"}
        )

        assert request.tool_choice is not None
        assert request.tool_choice.type == "any"


class TestBrowserStateBlock:
    """The ``browser_state`` part a browser toolset result carries.

    It is an arm of the ``tool_result`` content union, so a replayed transcript
    holding one is refused whole unless the arm exists.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/types/anthropic_messages.py:BrowserStateBlockParam
    """

    @staticmethod
    def _tool_result(**fields: object) -> ToolResultBlockParam:
        """Build a ``tool_result`` answering a browser toolset member call."""
        request = MessageCreateParams.model_validate(
            {
                "model": "claude-x",
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": [
                                    {
                                        "type": "browser_state",
                                        "tabs": [
                                            {
                                                "tab_id": "tab-1",
                                                "title": "Example",
                                                "url": "https://example.com/",
                                                "active": True,
                                            }
                                        ],
                                        **fields,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
        content = request.messages[0].content
        assert not isinstance(content, str)
        block = content[0]
        assert isinstance(block, ToolResultBlockParam)
        return block

    def test_the_mirrored_state_changes_match_the_sdk_exactly(self) -> None:
        """Every state-change variant the SDK declares has a body under test.

        Ref: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/browser_state_change_param.py
             stdapi/types/anthropic_messages.py:BrowserStateChangeParam
        """
        assert set(SDK_STATE_CHANGE_TYPES)
        assert set(STATE_CHANGES) == set(SDK_STATE_CHANGE_TYPES)

    def test_a_browser_state_result_keeps_its_tab_inventory(self) -> None:
        """The tab inventory validates and keeps every field it was sent with.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/types/anthropic_messages.py:BrowserStateTabEntryParam
        """
        block = self._tool_result()

        assert not isinstance(block.content, str)
        assert block.content is not None
        state = block.content[0]
        assert isinstance(state, BrowserStateBlockParam)
        (tab,) = state.tabs
        assert tab.tab_id == "tab-1"
        assert tab.title == "Example"
        assert tab.url == "https://example.com/"
        assert tab.active is True

    def test_an_empty_tab_inventory_is_accepted(self) -> None:
        """``tabs`` may legitimately be empty, so an empty list must validate.

        Ref: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/browser_state_block_param.py
             stdapi/types/anthropic_messages.py:BrowserStateBlockParam
        """
        request = MessageCreateParams.model_validate(
            {
                "model": "claude-x",
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": [{"type": "browser_state", "tabs": []}],
                            }
                        ],
                    }
                ],
            }
        )

        content = request.messages[0].content
        assert not isinstance(content, str)
        block = content[0]
        assert isinstance(block, ToolResultBlockParam)
        assert not isinstance(block.content, str)
        assert block.content is not None
        state = block.content[0]
        assert isinstance(state, BrowserStateBlockParam)
        assert state.tabs == []

    @pytest.mark.parametrize("change_type", SDK_STATE_CHANGE_TYPES)
    def test_every_state_change_variant_is_accepted(self, change_type: str) -> None:
        """Each of the four state changes validates and keeps its own fields.

        Ref: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/browser_state_change_param.py
             stdapi/types/anthropic_messages.py:BrowserStateChangeParam
        """
        change = STATE_CHANGES[change_type]
        block = self._tool_result(state_changes=[change])

        assert not isinstance(block.content, str)
        assert block.content is not None
        state = block.content[0]
        assert isinstance(state, BrowserStateBlockParam)
        assert state.state_changes is not None
        assert state.state_changes[0].model_dump(exclude_none=True) == change


class TestToolsetName:
    """The ``toolset_name`` pairing a toolset member's call with its result.

    Both blocks carry it, and the mirror ignores unknown keys -- so an unmodelled
    field validates and is then silently lost, which unpairs a replayed toolset
    turn instead of refusing it.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/types/anthropic_messages.py:ToolUseBlockParam
    """

    def test_a_tool_use_keeps_its_toolset_name(self) -> None:
        """A replayed toolset member call keeps the family it belongs to.

        Ref: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/tool_use_block_param.py
             stdapi/types/anthropic_messages.py:ToolUseBlockParam
        """
        block = ToolUseBlockParam.model_validate(
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "navigate",
                "input": {"url": "https://example.com/"},
                "toolset_name": SDK_BROWSER_TOOLSET_TYPE,
            }
        )

        assert block.toolset_name == SDK_BROWSER_TOOLSET_TYPE
        assert (
            block.model_dump(exclude_none=True)["toolset_name"]
            == SDK_BROWSER_TOOLSET_TYPE
        )

    def test_a_tool_result_keeps_its_toolset_name(self) -> None:
        """A replayed toolset member result keeps the family of the call it answers.

        Ref: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/tool_result_block_param.py
             stdapi/types/anthropic_messages.py:ToolResultBlockParam
        """
        block = ToolResultBlockParam.model_validate(
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "done",
                "toolset_name": SDK_BROWSER_TOOLSET_TYPE,
            }
        )

        assert block.toolset_name == SDK_BROWSER_TOOLSET_TYPE
        assert (
            block.model_dump(exclude_none=True)["toolset_name"]
            == SDK_BROWSER_TOOLSET_TYPE
        )

    def test_a_returned_tool_use_keeps_its_toolset_name(self) -> None:
        """A returned toolset member call names its family to the client.

        A backend serving the toolsets answers with it, and a client cannot pair
        the result it sends back without it.

        Ref: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/tool_use_block.py
             stdapi/types/anthropic_messages.py:ToolUseBlock
        """
        block = ToolUseBlock.model_validate(
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "navigate",
                "input": {"url": "https://example.com/"},
                "toolset_name": SDK_BROWSER_TOOLSET_TYPE,
            }
        )

        assert block.toolset_name == SDK_BROWSER_TOOLSET_TYPE
