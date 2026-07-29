"""Unit tests for system/tool cache_control breakpoint positioning (no AWS calls)."""

from __future__ import annotations

import pytest

from stdapi.models.chat._adapters._anthropic_message import (
    _build_tool_config,
    _map_system_blocks,
)
from stdapi.types.anthropic_messages import (
    CacheControlEphemeralParam,
    TextBlockParam,
    ToolInputSchema,
    ToolParam,
)

pytestmark = pytest.mark.local

#: Cache control applied to a non-last block/tool under test.
_CACHE_CONTROL = CacheControlEphemeralParam()


def test_system_cache_point_placed_after_marked_block_not_last() -> None:
    """A cache_control on a non-last system block caches only that block's prefix."""
    blocks = _map_system_blocks(
        [
            TextBlockParam(type="text", text="a", cache_control=_CACHE_CONTROL),
            TextBlockParam(type="text", text="b"),
            TextBlockParam(type="text", text="c"),
        ],
        allow_explicit_caching=True,
    )
    assert blocks == [
        {"text": "a"},
        {"cachePoint": {"type": "default"}},
        {"text": "b"},
        {"text": "c"},
    ]


def test_system_multiple_cache_breakpoints_are_preserved() -> None:
    """Multiple cache_control breakpoints on different system blocks all survive."""
    blocks = _map_system_blocks(
        [
            TextBlockParam(type="text", text="a", cache_control=_CACHE_CONTROL),
            TextBlockParam(type="text", text="b", cache_control=_CACHE_CONTROL),
        ],
        allow_explicit_caching=True,
    )
    cache_points = [b for b in blocks if "cachePoint" in b]
    assert len(cache_points) == 2


def _tool(
    name: str, cache_control: CacheControlEphemeralParam | None = None
) -> ToolParam:
    """Return a minimal ToolParam for cache_control positioning tests."""
    return ToolParam(
        name=name, input_schema=ToolInputSchema(), cache_control=cache_control
    )


def test_tool_cache_point_placed_after_marked_tool_not_last() -> None:
    """A cache_control on a non-last tool caches only that tool's prefix."""
    tool_config = _build_tool_config(
        [_tool("a", _CACHE_CONTROL), _tool("b"), _tool("c")],
        None,
        allow_explicit_caching=True,
    )
    assert tool_config is not None
    tools = tool_config["tools"]
    assert tools[0]["toolSpec"]["name"] == "a"  # type: ignore[typeddict-item]
    assert "cachePoint" in tools[1]
    assert tools[2]["toolSpec"]["name"] == "b"  # type: ignore[typeddict-item]
    assert tools[3]["toolSpec"]["name"] == "c"  # type: ignore[typeddict-item]
