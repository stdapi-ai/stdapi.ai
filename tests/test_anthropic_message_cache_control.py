"""``cache_control`` breakpoint positioning in ``system`` and ``tools`` (no AWS calls).

Anthropic marks a cache breakpoint on the block it terminates, while Bedrock
represents it as a separate ``cachePoint`` list element, so the gateway has to
insert that element directly after the marked block instead of at the end.

Ref: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CachePointBlock.html
     stdapi/models/chat/_adapters/_anthropic_message.py:_build_cache_point
"""

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
    """A cache_control on a non-last system block caches only that block's prefix.

    Ref: stdapi/models/chat/_adapters/_anthropic_message.py:_map_system_blocks
    """
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
    """Multiple cache_control breakpoints on different system blocks all survive.

    Anthropic allows up to four breakpoints per request, so consecutive marked
    blocks must each get their own ``cachePoint`` rather than being collapsed
    into a single trailing one.

    Ref: stdapi/models/chat/_adapters/_anthropic_message.py:_map_system_blocks
    """
    blocks = _map_system_blocks(
        [
            TextBlockParam(type="text", text="a", cache_control=_CACHE_CONTROL),
            TextBlockParam(type="text", text="b", cache_control=_CACHE_CONTROL),
        ],
        allow_explicit_caching=True,
    )
    assert blocks == [
        {"text": "a"},
        {"cachePoint": {"type": "default"}},
        {"text": "b"},
        {"cachePoint": {"type": "default"}},
    ]


def _tool(
    name: str, cache_control: CacheControlEphemeralParam | None = None
) -> ToolParam:
    """Return a minimal ToolParam for cache_control positioning tests."""
    return ToolParam(
        name=name, input_schema=ToolInputSchema(), cache_control=cache_control
    )


def test_tool_cache_point_placed_after_marked_tool_not_last() -> None:
    """A cache_control on a non-last tool caches only that tool's prefix.

    Inside ``toolConfig.tools`` a ``cachePoint`` is a list element of the
    ``toolSpec | systemTool | cachePoint`` union, not a field on a tool.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_build_tool_config
    """
    tool_config = _build_tool_config(
        [_tool("a", _CACHE_CONTROL), _tool("b"), _tool("c")],
        None,
        allow_explicit_caching=True,
    )
    assert tool_config is not None
    tools = tool_config["tools"]
    assert len(tools) == 4, "exactly one cachePoint element is inserted"
    assert tools[0]["toolSpec"]["name"] == "a"
    assert tools[1] == {"cachePoint": {"type": "default"}}
    assert tools[2]["toolSpec"]["name"] == "b"
    assert tools[3]["toolSpec"]["name"] == "c"
