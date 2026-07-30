"""Bedrock Converse response → non-streaming Anthropic ``Message`` (no AWS calls).

Ref: https://platform.claude.com/docs/en/api/messages
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     stdapi/models/chat/_adapters/_anthropic_message.py:format_response
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from stdapi.models.chat._adapters._anthropic_message import (
    _map_stop_reason,
    format_response,
)
from stdapi.types.anthropic_messages import (
    Message,
    ServerToolUseBlock,
    ToolUseBlock,
    Usage,
    WebSearchToolResultBlock,
)

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.type_defs import ContentBlockOutputTypeDef

pytestmark = pytest.mark.local


def test_map_stop_reason_preserves_context_window_exceeded() -> None:
    """Bedrock's ``model_context_window_exceeded`` stop reason is preserved.

    Anthropic's ``stop_reason`` enum carries the same value, so it must not be
    collapsed into ``max_tokens``: clients have to distinguish context exhaustion
    from the output cap, which Bedrock reports separately.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_stop_reason
    """
    assert _map_stop_reason("model_context_window_exceeded") == (
        "model_context_window_exceeded"
    )
    assert _map_stop_reason("max_tokens") == "max_tokens"


def test_message_accepts_context_window_exceeded_stop_reason() -> None:
    """The ``Message`` response model validates the ``model_context_window_exceeded`` stop reason.

    Anthropic's documented ``stop_reason`` set grew past the classic four values,
    so the mirrored response model must not raise a ``literal_error`` for it.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/types/anthropic_messages.py:Message
    """
    message = Message(
        id="msg_1",
        type="message",
        role="assistant",
        content=[],
        model="model-x",
        stop_reason="model_context_window_exceeded",
        usage=Usage(input_tokens=1, output_tokens=0),
    )
    assert message.stop_reason == "model_context_window_exceeded"


async def test_search_result_block_wrapped_in_web_search_tool_result() -> None:
    """A bare Bedrock ``searchResult`` block is nested in a ``web_search_tool_result``.

    Anthropic's ``ContentBlock`` union has no top-level ``web_search_result``
    member: a search result is only legal inside a ``web_search_tool_result``
    block's ``content`` list, keyed by ``url`` and ``title``.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_search_result_from_bedrock
    """
    contents = cast(
        "list[ContentBlockOutputTypeDef]",
        [{"searchResult": {"source": "https://example.com", "title": "Example"}}],
    )
    message = await format_response(
        contents=contents,
        stop_reason="end_turn",
        usage={},
        message_id="msg_1",
        model_id="model-x",
        forced_tool=None,
        resp_map_tool_result=lambda *_args: None,
    )
    assert len(message.content) == 1
    block = message.content[0]
    assert isinstance(block, WebSearchToolResultBlock)
    assert isinstance(block.content, list), "a tool error would be a single block"
    (result,) = block.content
    assert result.type == "web_search_result"
    assert result.url == "https://example.com"
    assert result.title == "Example"
    assert message.stop_reason == "end_turn"
    assert message.model == "model-x"


def _search_result_contents() -> list[ContentBlockOutputTypeDef]:
    """Return a Bedrock ``toolUse`` block followed by a bare ``searchResult`` one."""
    return cast(
        "list[ContentBlockOutputTypeDef]",
        [
            {
                "toolUse": {
                    "toolUseId": "tooluse_t1",
                    "name": "nova_grounding",
                    "input": cast("dict[str, Any]", {}),
                }
            },
            {"searchResult": {"source": "https://example.com", "title": "Example"}},
        ],
    )


async def test_search_result_correlates_to_preceding_tool_use_id() -> None:
    """A ``searchResult`` block is attributed to the nearest preceding tool use.

    Bedrock's ``searchResult`` block carries no ID of its own, so the wrapper's
    ``tool_use_id`` must be the Anthropic-side ID of the emitted tool-use block
    (``toolu_`` prefixed), not the raw Bedrock ``toolUseId``.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_search_result_from_bedrock
    """
    message = await format_response(
        contents=_search_result_contents(),
        stop_reason="end_turn",
        usage={},
        message_id="msg_1",
        model_id="model-x",
        forced_tool=None,
        resp_map_tool_result=lambda *_args: None,
    )
    assert [block.type for block in message.content] == [
        "tool_use",
        "web_search_tool_result",
    ]
    tool_use = next(b for b in message.content if isinstance(b, ToolUseBlock))
    web_result = next(
        b for b in message.content if isinstance(b, WebSearchToolResultBlock)
    )
    assert tool_use.id == "toolu_tooluse_t1"
    assert web_result.tool_use_id == tool_use.id


async def test_search_result_correlates_to_mapped_server_tool_use_id() -> None:
    """The correlation follows the model-specific ``server_tool_use`` mapping.

    A model whose grounding tool is surfaced as an Anthropic server tool re-ids the
    block to ``srvtoolu_...``; the search-result wrapper must track that ID rather
    than the default ``toolu_`` one.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
         stdapi/models/chat/_adapters/_anthropic_message.py:format_response
    """
    message = await format_response(
        contents=_search_result_contents(),
        stop_reason="end_turn",
        usage={},
        message_id="msg_1",
        model_id="model-x",
        forced_tool=None,
        resp_map_tool_result=lambda *_args: None,
        resp_map_tool_use=lambda tool_use_id, _name, tool_input: ServerToolUseBlock(
            type="server_tool_use",
            id=f"srvtoolu_{tool_use_id.removeprefix('tooluse_')}",
            name="web_search",
            input=tool_input,
        ),
    )
    assert [block.type for block in message.content] == [
        "server_tool_use",
        "web_search_tool_result",
    ]
    server_tool_use = next(
        b for b in message.content if isinstance(b, ServerToolUseBlock)
    )
    assert server_tool_use.id == "srvtoolu_t1"
    assert server_tool_use.name == "web_search"
    web_result = next(
        b for b in message.content if isinstance(b, WebSearchToolResultBlock)
    )
    assert web_result.tool_use_id == "srvtoolu_t1"
