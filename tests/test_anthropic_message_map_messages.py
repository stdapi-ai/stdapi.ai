"""Unit tests for ``_map_messages``' ``allow_tool_caching`` behavior (no AWS calls)."""

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
    """``allow_tool_caching`` gates cache points on tool_use/tool_result blocks only."""

    async def test_tool_caching_disallowed_skips_cache_point_on_tool_use(self) -> None:
        """No cachePoint follows a cache-controlled tool_use block when disallowed."""
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
        """A cachePoint still follows a cache-controlled text block when tool caching is off."""
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
