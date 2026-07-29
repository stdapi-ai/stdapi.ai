"""Unit tests for ``tool_result`` content block mapping (no AWS calls)."""

from __future__ import annotations

import pytest

from stdapi.api_errors import ApiError
from stdapi.models.chat._adapters._anthropic_message import _map_tool_result_to_bedrock
from stdapi.types.anthropic_messages import (
    DocumentBlockParam,
    PlainTextSourceParam,
    SearchResultBlockParam,
    TextBlockParam,
    ToolReferenceBlockParam,
    ToolResultBlockParam,
)

pytestmark = pytest.mark.local


def test_tool_result_content_accepts_document_and_search_result_blocks() -> None:
    """``ToolResultBlockParam.content`` validates upstream block types.

    ``document``, ``search_result``, and ``tool_reference`` blocks must not be
    rejected with a 422.
    """
    block = ToolResultBlockParam(
        type="tool_result",
        tool_use_id="toolu_1",
        content=[
            DocumentBlockParam(
                type="document",
                source=PlainTextSourceParam(
                    type="text", media_type="text/plain", data="doc body"
                ),
            ),
            SearchResultBlockParam(
                type="search_result",
                source="https://example.com",
                title="Example",
                content=[TextBlockParam(type="text", text="snippet")],
            ),
            ToolReferenceBlockParam(type="tool_reference", tool_name="lookup"),
        ],
    )
    assert len(block.content) == 3  # type: ignore[arg-type]


async def test_map_tool_result_to_bedrock_maps_document_block() -> None:
    """A ``document`` block inside a tool result maps to a Bedrock document block."""
    block = ToolResultBlockParam(
        type="tool_result",
        tool_use_id="toolu_1",
        content=[
            DocumentBlockParam(
                type="document",
                title="notes",
                source=PlainTextSourceParam(
                    type="text", media_type="text/plain", data="doc body"
                ),
            )
        ],
    )
    result = await _map_tool_result_to_bedrock(block)
    (content_item,) = result["toolResult"]["content"]
    assert "document" in content_item


async def test_map_tool_result_to_bedrock_maps_search_result_block() -> None:
    """A ``search_result`` block inside a tool result maps to a Bedrock block.

    The mapping matches the one used for top-level content blocks.
    """
    block = ToolResultBlockParam(
        type="tool_result",
        tool_use_id="toolu_1",
        content=[
            SearchResultBlockParam(
                type="search_result",
                source="https://example.com",
                title="Example",
                content=[TextBlockParam(type="text", text="snippet")],
            )
        ],
    )
    result = await _map_tool_result_to_bedrock(block)
    (content_item,) = result["toolResult"]["content"]
    assert content_item["searchResult"]["source"] == "https://example.com"
    assert content_item["searchResult"]["title"] == "Example"
    assert content_item["searchResult"]["content"] == [{"text": "snippet"}]


async def test_map_tool_result_to_bedrock_rejects_tool_reference_block() -> None:
    """A ``tool_reference`` block has no Bedrock equivalent.

    It must raise cleanly instead of crashing with an ``AttributeError``.
    """
    block = ToolResultBlockParam(
        type="tool_result",
        tool_use_id="toolu_1",
        content=[ToolReferenceBlockParam(type="tool_reference", tool_name="lookup")],
    )
    with pytest.raises(ApiError):
        await _map_tool_result_to_bedrock(block)
