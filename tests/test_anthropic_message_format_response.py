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
    CitationCharLocation,
    CitationContentBlockLocation,
    CitationPageLocation,
    CitationsSearchResultLocation,
    CitationsWebSearchResultLocation,
    Message,
    ServerToolUseBlock,
    TextBlock,
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


@pytest.mark.parametrize(
    "bedrock_stop_reason",
    [
        "content_filtered",
        "guardrail_intervened",
        "malformed_model_output",
        "malformed_tool_use",
    ],
)
def test_map_stop_reason_maps_blocked_generations_to_refusal(
    bedrock_stop_reason: str,
) -> None:
    """Filtered, guardrailed and malformed Bedrock generations become ``refusal``.

    Anthropic has no per-cause stop reason for a blocked generation, so all four
    Bedrock outcomes collapse onto the single ``refusal`` value.  Mapping any of
    them to ``end_turn`` instead would tell the caller it received a complete,
    unfiltered answer.

    Ref: https://platform.claude.com/docs/en/api/messages
         https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_STOP_REASONS
    """
    assert _map_stop_reason(bedrock_stop_reason) == "refusal"


def test_map_stop_reason_incomplete_and_unknown_fallbacks() -> None:
    """The non-standard ``incomplete`` reason becomes ``max_tokens``; anything unknown ``end_turn``.

    ``incomplete`` is not in the Converse stop-reason enum but is observed from
    some Bedrock backends; an unmapped value must degrade to ``end_turn`` rather
    than leaking a non-Anthropic literal into the response model.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_stop_reason
    """
    assert _map_stop_reason("incomplete") == "max_tokens"
    assert _map_stop_reason("something_new") == "end_turn"
    assert _map_stop_reason(None) == "end_turn"


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


async def test_consecutive_search_results_aggregate_into_one_wrapper() -> None:
    """Consecutive ``searchResult`` blocks fold into a single ``web_search_tool_result``.

    Bedrock emits one ``searchResult`` block per result, while Anthropic returns
    exactly one ``web_search_tool_result`` block per search whose ``content``
    lists every result; emitting one wrapper per result would hand SDK clients
    several blocks sharing the same ``tool_use_id``.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
         stdapi/models/chat/_adapters/_anthropic_message.py:_merge_into_previous_web_search_result
    """
    contents = _search_result_contents()
    contents.append(
        cast(
            "ContentBlockOutputTypeDef",
            {"searchResult": {"source": "https://example.org", "title": "Other"}},
        )
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
    assert [block.type for block in message.content] == [
        "tool_use",
        "web_search_tool_result",
    ], "both results must share one wrapper block"
    wrapper = next(
        b for b in message.content if isinstance(b, WebSearchToolResultBlock)
    )
    assert isinstance(wrapper.content, list)
    assert [result.url for result in wrapper.content] == [
        "https://example.com",
        "https://example.org",
    ]


async def test_usage_cache_tokens_read_from_bedrock_keys() -> None:
    """``usage`` cache counters come from Bedrock's ``cacheRead/WriteInputTokens``.

    Bedrock's ``TokenUsage`` has no ``cacheCreationInputTokens`` key: cache
    writes are reported as ``cacheWriteInputTokens``, which must surface as
    Anthropic's ``cache_creation_input_tokens`` instead of staying ``None``.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
         https://platform.claude.com/docs/en/api/messages
         stdapi/models/chat/_adapters/_anthropic_message.py:format_response
    """
    message = await format_response(
        contents=cast("list[ContentBlockOutputTypeDef]", [{"text": "hi"}]),
        stop_reason="end_turn",
        usage={
            "inputTokens": 10,
            "outputTokens": 5,
            "cacheReadInputTokens": 3,
            "cacheWriteInputTokens": 7,
        },
        message_id="msg_1",
        model_id="model-x",
        forced_tool=None,
        resp_map_tool_result=lambda *_args: None,
    )
    assert message.usage.input_tokens == 10
    assert message.usage.output_tokens == 5
    assert message.usage.cache_read_input_tokens == 3
    assert message.usage.cache_creation_input_tokens == 7


async def _citation_block(location: dict[str, Any]) -> TextBlock:
    """Run ``format_response`` over one Bedrock ``citationsContent`` block."""
    contents = cast(
        "list[ContentBlockOutputTypeDef]",
        [
            {
                "citationsContent": {
                    "content": [{"text": "Guido created Python"}],
                    "citations": [
                        {
                            "title": "Python history",
                            "source": "https://example.com/doc",
                            "sourceContent": [{"text": "released in 1991"}],
                            "location": location,
                        }
                    ],
                }
            }
        ],
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
    (block,) = message.content
    assert isinstance(block, TextBlock)
    return block


async def test_citations_content_becomes_a_text_block_with_citations() -> None:
    """A Bedrock ``citationsContent`` block becomes one text block carrying its citations.

    Bedrock keeps the answer text and the citation metadata in one block, whereas
    Anthropic attaches ``citations`` to the ``text`` block itself, so the content
    items are concatenated into the block text and the cited span is taken from
    ``sourceContent`` rather than from the answer.

    Ref: https://platform.claude.com/docs/en/api/messages
         https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_citations_content_from_bedrock
    """
    block = await _citation_block({"documentChar": {"start": 3, "end": 9}})
    assert block.type == "text"
    assert block.text == "Guido created Python"
    assert block.citations is not None
    (citation,) = block.citations
    assert isinstance(citation, CitationCharLocation)
    assert citation.cited_text == "released in 1991"
    assert citation.document_title == "Python history"


async def test_citation_char_location_carries_start_and_end_indices() -> None:
    """``documentChar`` maps to ``char_location`` with the Bedrock start/end offsets.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_citations_from_bedrock
    """
    block = await _citation_block(
        {"documentChar": {"documentIndex": 2, "start": 3, "end": 9}}
    )
    assert block.citations is not None
    (citation,) = block.citations
    assert isinstance(citation, CitationCharLocation)
    assert citation.type == "char_location"
    assert citation.document_index == 2
    assert citation.start_char_index == 3
    assert citation.end_char_index == 9


async def test_citation_page_location_carries_page_numbers() -> None:
    """``documentPage`` maps to ``page_location`` with start/end page numbers.

    The Bedrock payload uses the same generic ``start``/``end`` keys for every
    location kind, so a mix-up between the page and char arms would silently emit
    character offsets as page numbers.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_citations_from_bedrock
    """
    block = await _citation_block(
        {"documentPage": {"documentIndex": 1, "start": 4, "end": 5}}
    )
    assert block.citations is not None
    (citation,) = block.citations
    assert isinstance(citation, CitationPageLocation)
    assert citation.type == "page_location"
    assert citation.document_index == 1
    assert citation.start_page_number == 4
    assert citation.end_page_number == 5


async def test_citation_document_chunk_maps_to_content_block_location() -> None:
    """``documentChunk`` maps to ``content_block_location`` with block indices.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_citations_from_bedrock
    """
    block = await _citation_block(
        {"documentChunk": {"documentIndex": 0, "start": 6, "end": 7}}
    )
    assert block.citations is not None
    (citation,) = block.citations
    assert isinstance(citation, CitationContentBlockLocation)
    assert citation.type == "content_block_location"
    assert citation.start_block_index == 6
    assert citation.end_block_index == 7


async def test_citation_web_location_falls_back_to_the_citation_source_url() -> None:
    """``web`` maps to ``web_search_result_location``, defaulting the URL to the citation source.

    Bedrock omits ``location.web.url`` when the citation already carries a
    ``source``; ``encrypted_index`` has no Bedrock equivalent and is emitted empty
    because the Anthropic model requires the field.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_citations_from_bedrock
    """
    block = await _citation_block({"web": {}})
    assert block.citations is not None
    (citation,) = block.citations
    assert isinstance(citation, CitationsWebSearchResultLocation)
    assert citation.type == "web_search_result_location"
    assert citation.url == "https://example.com/doc"
    assert citation.title == "Python history"
    assert citation.encrypted_index == ""


async def test_citation_search_result_location_keeps_source_and_index() -> None:
    """``searchResultLocation`` maps to ``search_result_location`` with source and block span.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_citations_from_bedrock
    """
    block = await _citation_block(
        {"searchResultLocation": {"searchResultIndex": 3, "start": 1, "end": 2}}
    )
    assert block.citations is not None
    (citation,) = block.citations
    assert isinstance(citation, CitationsSearchResultLocation)
    assert citation.type == "search_result_location"
    assert citation.search_result_index == 3
    assert citation.source == "https://example.com/doc"
    assert citation.start_block_index == 1
    assert citation.end_block_index == 2


async def test_citations_content_without_citations_has_none() -> None:
    """A citations block with no citation entries yields ``citations = None``, not an empty list.

    Anthropic's ``TextBlock.citations`` is optional, and an empty list would make
    a client believe the answer was grounded.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_citations_content_from_bedrock
    """
    contents = cast(
        "list[ContentBlockOutputTypeDef]",
        [{"citationsContent": {"content": [{"text": "plain"}]}}],
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
    (block,) = message.content
    assert isinstance(block, TextBlock)
    assert block.text == "plain"
    assert block.citations is None
