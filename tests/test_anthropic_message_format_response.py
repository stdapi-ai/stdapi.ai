"""Unit tests for the non-streaming Anthropic messages response adapter (no AWS calls)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from stdapi.models.chat._adapters._anthropic_message import (
    _map_stop_reason,
    format_response,
)
from stdapi.types.anthropic_messages import Message, ServerToolUseBlock, Usage

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.type_defs import ContentBlockOutputTypeDef

pytestmark = pytest.mark.local


def test_map_stop_reason_preserves_context_window_exceeded() -> None:
    """Bedrock's ``model_context_window_exceeded`` stop reason is preserved.

    It must not be collapsed into ``max_tokens``, so clients can distinguish
    context exhaustion from the output cap.
    """
    assert _map_stop_reason("model_context_window_exceeded") == (
        "model_context_window_exceeded"
    )


def test_message_accepts_context_window_exceeded_stop_reason() -> None:
    """The ``Message`` response model validates the new stop reason.

    It must not raise a ``literal_error`` for ``model_context_window_exceeded``.
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
    member, so returning it unwrapped previously raised a ``ValidationError``.
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
    assert block.type == "web_search_tool_result"
    assert block.content[0].url == "https://example.com"  # type: ignore[union-attr]
    assert block.content[0].title == "Example"  # type: ignore[union-attr]


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

    The ``tool_use_id`` must be the Anthropic-side ID of the emitted block, not
    the raw Bedrock ``toolUseId``, otherwise clients cannot pair the two blocks.
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
    tool_use = next(b for b in message.content if b.type == "tool_use")
    web_result = next(b for b in message.content if b.type == "web_search_tool_result")
    assert web_result.tool_use_id == tool_use.id  # type: ignore[union-attr]


async def test_search_result_correlates_to_mapped_server_tool_use_id() -> None:
    """The correlation follows the model-specific ``server_tool_use`` mapping."""
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
    web_result = next(b for b in message.content if b.type == "web_search_tool_result")
    assert web_result.tool_use_id == "srvtoolu_t1"  # type: ignore[union-attr]
