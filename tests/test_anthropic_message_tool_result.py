"""Anthropic ``tool_result`` content parts → Bedrock ``toolResult`` blocks.

Every test but the last one maps blocks in process and makes no API call; the last
one drives the tool loop end to end against the selected target.

Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
     stdapi/models/chat/_adapters/_anthropic_message.py:_map_tool_result_to_bedrock
"""

from __future__ import annotations

from base64 import b64encode
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic.types import ToolParam

#: Side-effect-only tool: its result carries nothing back to the model.
_RECORD_EVENT_TOOL: ToolParam = {
    "name": "record_event",
    "description": "Record an event in the audit log. Returns nothing.",
    "input_schema": {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Event name."}},
        "required": ["name"],
    },
}


@pytest.mark.local
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


@pytest.mark.local
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


@pytest.mark.local
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


@pytest.mark.local
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


@pytest.mark.local
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


@pytest.mark.local
async def test_map_tool_result_keeps_an_mcp_tool_use_id_intact() -> None:
    """An ``mcptoolu_`` identifier reaches Bedrock unchanged.

    Only the ``toolu_`` prefix is stripped, and an MCP identifier does not carry
    it, so the value must survive verbatim — the matching ``toolUse`` block goes
    through the same rule, and Bedrock rejects a ``toolResult`` whose identifier
    matches no call.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/mcp-connector
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultBlock.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_tool_result_to_bedrock
    """
    mcp_id = "mcptoolu_01ABCdefGHIjklMNOpqrST"
    block = ToolResultBlockParam(
        type="tool_result", tool_use_id=mcp_id, content="the example server says hi"
    )
    result = await _map_tool_result_to_bedrock(block)
    assert result["toolResult"]["toolUseId"] == mcp_id


@pytest.mark.local
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"content": ""}, id="empty string"),
        pytest.param({"content": []}, id="empty list"),
    ],
)
async def test_map_tool_result_to_bedrock_sends_an_empty_result_as_empty_text(
    payload: dict[str, object],
) -> None:
    """Every spelling of an empty tool result reaches the model as empty text.

    A tool called for its side effect returns nothing, which a client expresses by
    omitting ``content``, by sending an empty string or by sending an empty list.
    All three must arrive as one empty text part: a result carrying no part at all
    is refused by some models, and dropping the block entirely would leave the
    ``tool_use`` it answers unpaired, which every model refuses.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultBlock.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_tool_result_to_bedrock
    """
    block = ToolResultBlockParam.model_validate(
        {"type": "tool_result", "tool_use_id": "toolu_1"} | payload
    )
    result = await _map_tool_result_to_bedrock(block)
    assert result == {"toolResult": {"toolUseId": "1", "content": [{"text": ""}]}}


@pytest.mark.local
async def test_map_tool_result_to_bedrock_keeps_the_error_status_of_an_empty_result() -> (
    None
):
    """An empty result reporting a failure still reaches the model as an error.

    ``is_error`` is what tells the model the call failed rather than returned
    nothing, so filling in the empty content must not drop it.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_tool_result_to_bedrock
    """
    block = ToolResultBlockParam.model_validate(
        {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": True}
    )
    result = await _map_tool_result_to_bedrock(block)
    assert result["toolResult"]["content"] == [{"text": ""}]
    assert result["toolResult"]["status"] == "error"


def test_a_tool_loop_answered_with_an_empty_result_continues(
    anthropic_client: Anthropic, anthropic_chat_vision_model: str
) -> None:
    """A tool answered with a ``tool_result`` carrying no content is served.

    This is the agent loop of a side-effect-only tool: the model asks for the
    call, the client runs it, and the result it sends back carries nothing.  The
    request must be answered rather than refused; which content the model then
    produces is its own decision and is not asserted.  The model is the one the
    other tool-loop tests of this route use, since forcing a call with
    ``tool_choice`` ``any`` is proven on it.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
         stdapi/types/anthropic_messages.py:ToolResultBlockParam
    """
    question = "Record the event named 'signup', then confirm it in one sentence."
    call = anthropic_client.messages.create(
        model=anthropic_chat_vision_model,
        max_tokens=300,
        messages=[{"role": "user", "content": question}],
        tools=[_RECORD_EVENT_TOOL],
        tool_choice={"type": "any"},
    )
    tool_uses = [block for block in call.content if block.type == "tool_use"]
    assert tool_uses, "the model must ask for the tool before its result can be sent"

    answer = anthropic_client.messages.create(
        model=anthropic_chat_vision_model,
        max_tokens=300,
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": call.content},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": tool_use.id}
                    for tool_use in tool_uses
                ],
            },
        ],
        tools=[_RECORD_EVENT_TOOL],
    )
    assert answer.type == "message"
    assert answer.content, "the empty result must still be answered"
    assert answer.usage.output_tokens > 0
