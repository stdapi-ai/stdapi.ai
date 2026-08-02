"""Anthropic ``tool_result`` content parts → Bedrock ``toolResult`` blocks (no AWS calls).

Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
     stdapi/models/chat/_adapters/_anthropic_message.py:_map_tool_result_to_bedrock
"""

from __future__ import annotations

from base64 import b64encode

import pytest

from stdapi.api_errors import ApiError
from stdapi.models.chat._adapters._anthropic_message import _map_tool_result_to_bedrock
from stdapi.types.anthropic_messages import (
    DocumentBlockParam,
    ImageBlockParam,
    PlainTextSourceParam,
    SearchResultBlockParam,
    TextBlockParam,
    ToolReferenceBlockParam,
    ToolResultBlockParam,
)

pytestmark = pytest.mark.local


def test_tool_result_content_accepts_document_and_search_result_blocks() -> None:
    """``ToolResultBlockParam.content`` validates every upstream part type.

    The union mirrors the Anthropic SDK, so ``document``, ``search_result`` and
    ``tool_reference`` parts must each keep their own discriminated model rather
    than being rejected with a 422 or coerced into a text part.

    Ref: https://github.com/anthropics/anthropic-sdk-python/tree/main/src/anthropic/types
         stdapi/types/anthropic_messages.py:ToolResultBlockParam
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
    parts = block.content
    assert isinstance(parts, list), "a block list must not be coerced to a string"
    assert [part.type for part in parts] == [
        "document",
        "search_result",
        "tool_reference",
    ]
    document, search_result, tool_reference = parts
    assert isinstance(document, DocumentBlockParam)
    assert isinstance(search_result, SearchResultBlockParam)
    assert isinstance(tool_reference, ToolReferenceBlockParam)
    assert tool_reference.tool_name == "lookup"


async def test_map_tool_result_to_bedrock_maps_document_block() -> None:
    """A ``document`` block inside a tool result maps to a Bedrock document block.

    A plain-text source is materialized as UTF-8 bytes with format ``txt``, and the
    Anthropic ``title`` becomes the Bedrock document ``name``, which only accepts
    ``[a-zA-Z0-9_-]``.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_document_to_bedrock
    """
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
    assert result["toolResult"]["toolUseId"] == "1", "the toolu_ prefix is stripped"
    (content_item,) = result["toolResult"]["content"]
    assert content_item["document"] == {
        "name": "notes",
        "format": "txt",
        "source": {"bytes": b"doc body"},
    }
    assert "status" not in result["toolResult"], "a successful result carries no status"


async def test_map_tool_result_to_bedrock_keeps_mixed_text_and_image_parts() -> None:
    """Text and image parts of one tool result keep their order and their kinds.

    ``ToolResultBlockParam.content`` is a heterogeneous list, so each part goes
    through its own Bedrock mapping while the list order — which carries the
    caller's meaning — is preserved.  The declared ``media_type`` drives the
    Bedrock image ``format`` rather than content sniffing.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultContentBlock.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_tool_result_part_to_bedrock
    """
    block = ToolResultBlockParam(
        type="tool_result",
        tool_use_id="toolu_1",
        content=[
            TextBlockParam(type="text", text="chart below"),
            ImageBlockParam.model_validate(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64encode(b"PNGDATA").decode(),
                    },
                }
            ),
            TextBlockParam(type="text", text="chart above"),
        ],
    )
    result = await _map_tool_result_to_bedrock(block)
    first, second, third = result["toolResult"]["content"]
    assert first == {"text": "chart below"}
    assert second["image"]["format"] == "png"
    assert third == {"text": "chart above"}


async def test_map_tool_result_to_bedrock_maps_search_result_block() -> None:
    """A ``search_result`` block inside a tool result maps to a Bedrock block.

    The mapping matches the one used for top-level content blocks: ``source`` and
    ``title`` are copied verbatim and nested text parts are flattened.

    Ref: stdapi/models/chat/_adapters/_anthropic_message.py:_map_search_result_to_bedrock
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

    Bedrock's ``ToolResultContentBlock`` union has no reference member, so the part
    must raise a 400-class ``ApiError`` naming the offending part by its wire
    ``type`` value — never by the internal Python class — and pointing at the
    accepted alternatives.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_tool_result_part_to_bedrock
    """
    block = ToolResultBlockParam(
        type="tool_result",
        tool_use_id="toolu_1",
        content=[ToolReferenceBlockParam(type="tool_reference", tool_name="lookup")],
    )
    with pytest.raises(ApiError) as excinfo:
        await _map_tool_result_to_bedrock(block)
    assert excinfo.value.status == 400
    message = str(excinfo.value)
    assert "'tool_reference' is not supported" in message
    assert "ToolReferenceBlockParam" not in message, (
        "the internal class name must not leak into the client-facing error"
    )
