"""Anthropic messages → Bedrock Converse messages cache-point placement (no AWS calls).

Ref: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CachePointBlock.html
     stdapi/models/chat/_adapters/_anthropic_message.py:_map_messages
"""

from __future__ import annotations

import pytest

from stdapi.models.chat._adapters._anthropic_message import _map_messages
from stdapi.types.anthropic_messages import (
    CacheControlEphemeralParam,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

pytestmark = pytest.mark.local

#: Cache control applied to every content block under test.
_CACHE_CONTROL = CacheControlEphemeralParam()


def _tool_use_message() -> MessageParam:
    """Return an assistant message with a cache-controlled tool_use block."""
    return MessageParam(
        role="assistant",
        content=[
            ToolUseBlockParam(
                type="tool_use",
                id="toolu_1",
                name="lookup",
                input={},
                cache_control=_CACHE_CONTROL,
            )
        ],
    )


def _tool_result_message() -> MessageParam:
    """Return a user message with a cache-controlled tool_result block."""
    return MessageParam(
        role="user",
        content=[
            ToolResultBlockParam(
                type="tool_result",
                tool_use_id="toolu_1",
                content="result",
                cache_control=_CACHE_CONTROL,
            )
        ],
    )


def _text_message() -> MessageParam:
    """Return a user message with a cache-controlled text block."""
    return MessageParam(
        role="user",
        content=[TextBlockParam(type="text", text="hi", cache_control=_CACHE_CONTROL)],
    )


class TestMapMessagesAllowToolCaching:
    """``allow_tool_caching`` gates cache points on tool_use/tool_result blocks only.

    Some Bedrock models reject a ``cachePoint`` in a turn that also carries
    ``toolUse``/``toolResult`` blocks, so the gateway drops the breakpoint there
    instead of failing the request; ``cache_control`` is silently ignored, which
    upstream permits since caching never errors.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_messages
    """

    async def test_tool_caching_disallowed_skips_cache_point_on_tool_use(self) -> None:
        """No cachePoint follows a cache-controlled tool_use block when disallowed.

        The ``toolu_`` prefix is stripped because Bedrock ``toolUseId`` values carry
        no Anthropic prefix.
        """
        result = await _map_messages(
            [_tool_use_message()], allow_explicit_caching=True, allow_tool_caching=False
        )
        assert result[0]["content"] == [
            {"toolUse": {"toolUseId": "1", "name": "lookup", "input": {}}}
        ]

    async def test_tool_caching_disallowed_skips_cache_point_on_tool_result(
        self,
    ) -> None:
        """No cachePoint follows a cache-controlled tool_result block when disallowed."""
        result = await _map_messages(
            [_tool_result_message()],
            allow_explicit_caching=True,
            allow_tool_caching=False,
        )
        assert result[0]["content"] == [
            {"toolResult": {"toolUseId": "1", "content": [{"text": "result"}]}}
        ]

    async def test_tool_caching_disallowed_keeps_cache_point_on_text(self) -> None:
        """A cachePoint still follows a cache-controlled text block when tool caching is off.

        Only tool blocks are affected, so an ordinary text breakpoint survives and
        becomes a ``cachePoint`` element after the block it terminates.
        """
        result = await _map_messages(
            [_text_message()], allow_explicit_caching=True, allow_tool_caching=False
        )
        assert result[0]["content"] == [
            {"text": "hi"},
            {"cachePoint": {"type": "default"}},
        ]

    async def test_tool_caching_allowed_keeps_cache_point_on_tool_use(self) -> None:
        """A cachePoint follows a cache-controlled tool_use block when allowed (default)."""
        result = await _map_messages(
            [_tool_use_message()], allow_explicit_caching=True, allow_tool_caching=True
        )
        assert result[0]["content"] == [
            {"toolUse": {"toolUseId": "1", "name": "lookup", "input": {}}},
            {"cachePoint": {"type": "default"}},
        ]

    async def test_tool_caching_allowed_keeps_cache_point_on_tool_result(self) -> None:
        """A cachePoint follows a cache-controlled tool_result block when allowed (default)."""
        result = await _map_messages(
            [_tool_result_message()],
            allow_explicit_caching=True,
            allow_tool_caching=True,
        )
        assert result[0]["content"] == [
            {"toolResult": {"toolUseId": "1", "content": [{"text": "result"}]}},
            {"cachePoint": {"type": "default"}},
        ]
