"""Anthropic Messages API adapter for Bedrock Converse.

Translates between Anthropic Messages API request/response types and
Bedrock Converse API-native types. Anthropic's content block format is
close to Bedrock's native format, so many mappings are near 1:1.
"""

import re
from typing import TYPE_CHECKING, Any

from pydantic_core import to_json
from sse_starlette import JSONServerSentEvent

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.aws_bedrock import (
    PROMPT_CACHING,
    PROMPT_CACHING_BASIC,
    PROMPT_CACHING_DEFAULT,
    PromptCaching,
    build_system_blocks,
    handle_bedrock_client_error,
    set_inference_configuration,
)
from stdapi.monitoring import log_response_params
from stdapi.types.anthropic_messages import (
    SERVER_TOOL_NAMES,
    Base64ImageSource,
    Base64PDFSource,
    CacheControlEphemeralParam,
    CitationCharLocation,
    CitationContentBlockLocation,
    CitationPageLocation,
    CitationsSearchResultLocation,
    CitationsWebSearchResultLocation,
    CodeExecutionToolParam,
    ContentBlock,
    ContentBlockParam,
    ContentBlockSourceParam,
    DocumentBlockParam,
    ImageBlockParam,
    InputJSONDelta,
    MemoryToolParam,
    Message,
    MessageCountTokensParams,
    MessageCreateParams,
    MessageDelta,
    MessageDeltaUsage,
    MessageParam,
    OutputConfigParam,
    PlainTextSourceParam,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    RedactedThinkingBlock,
    RedactedThinkingBlockParam,
    SearchResultBlockParam,
    ServerToolUseBlock,
    ServerToolUseBlockParam,
    ServiceTiers,
    SignatureDelta,
    StopReason,
    TextBlock,
    TextBlockParam,
    TextCitation,
    TextDelta,
    ThinkingBlock,
    ThinkingBlockParam,
    ThinkingDelta,
    ToolBashParam,
    ToolChoiceAnyParam,
    ToolChoiceAutoParam,
    ToolChoiceNoneParam,
    ToolChoiceParam,
    ToolChoiceToolParam,
    ToolParam,
    ToolResultBlockParam,
    ToolSearchToolBm25Param,
    ToolSearchToolRegexParam,
    ToolTextEditorParam,
    ToolUnionParam,
    ToolUseBlock,
    ToolUseBlockParam,
    URLImageSource,
    URLPDFSource,
    Usage,
    WebFetchToolParam,
    WebSearchResultBlock,
    WebSearchToolParam,
)
from stdapi.utils import b64decode, b64encode

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime.literals import (
        CacheTTLType,
        ServiceTierTypeType,
        StopReasonType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockDeltaEventTypeDef,
        ContentBlockDeltaTypeDef,
        ContentBlockOutputTypeDef,
        ContentBlockStartEventTypeDef,
        ContentBlockStartTypeDef,
        ContentBlockTypeDef,
        ConverseStreamOutputTypeDef,
        ConverseTokensRequestTypeDef,
        CountTokensResponseTypeDef,
        InferenceConfigurationTypeDef,
        JsonSchemaDefinitionTypeDef,
        MessageTypeDef,
        ReasoningTextBlockTypeDef,
        SystemContentBlockTypeDef,
        ToolChoiceTypeDef,
        ToolConfigurationTypeDef,
        ToolResultBlockTypeDef,
        ToolTypeDef,
    )

    from stdapi.types import JsonMapping


def _build_cache_point(
    cache_control: CacheControlEphemeralParam,
) -> ContentBlockTypeDef:
    """Build a Bedrock cache point block from an Anthropic cache control param.

    Args:
        cache_control: Anthropic cache control ephemeral param.

    Returns:
        Bedrock cache point block dict.
    """
    if ttl := cache_control.ttl:
        return {"cachePoint": {"type": "default", "ttl": ttl}}
    return PROMPT_CACHING_DEFAULT


#: Bedrock stop reasons to Anthropic stop reasons mapping
_STOP_REASONS: dict[StopReasonType | None, StopReason] = {
    "end_turn": "end_turn",
    "max_tokens": "max_tokens",
    "model_context_window_exceeded": "max_tokens",
    "stop_sequence": "stop_sequence",
    "tool_use": "tool_use",
    "content_filtered": "refusal",
    "guardrail_intervened": "refusal",
    "malformed_model_output": "refusal",
    "malformed_tool_use": "refusal",
}

#: Anthropic services tiers to Bedrock mapping
_SERVICES_TIERS: dict[ServiceTiers | None, ServiceTierTypeType] = {
    "standard_only": "default",
    # Extra bedrock specific values
    "priority": "priority",
    "flex": "flex",
    "reserved": "reserved",
}

#: Regex to sanitize document names for Bedrock (only [a-zA-Z0-9_-] allowed)
_RE_DOC_NAME = re.compile(r"[^a-zA-Z0-9_-]")


def _map_stop_reason(stop_reason: StopReasonType | None) -> StopReason:
    """Map a Bedrock stop reason to an Anthropic stop reason.

    Args:
        stop_reason: Bedrock stop reason literal, or ``None``.

    Returns:
        Corresponding Anthropic stop reason string.
    """
    return _STOP_REASONS.get(stop_reason, "end_turn")


def _map_system_blocks(
    content: str | list[TextBlockParam] | None, *, allow_explicit_caching: bool = False
) -> list[SystemContentBlockTypeDef]:
    """Convert an Anthropic system prompt to Bedrock system content blocks.

    Args:
        content: A plain string, a list of text block params, or ``None``.
        allow_explicit_caching: Whether to allow prompt caching for system blocks.

    Returns:
        List of Bedrock system content block dicts.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return build_system_blocks(content)
    blocks: list[SystemContentBlockTypeDef] = []
    cache_control: CacheControlEphemeralParam | None = None
    for part in content:
        if part.text:
            blocks.append({"text": part.text})
            cache_control = part.cache_control or cache_control
    if allow_explicit_caching and cache_control:
        blocks.append(_build_cache_point(cache_control))
    return blocks


async def _map_image_to_bedrock(
    source: Base64ImageSource | URLImageSource,
) -> ContentBlockTypeDef:
    """Convert an Anthropic image source to a Bedrock content block.

    Args:
        source: Base64-encoded or URL image source.

    Returns:
        A partial content block dict (source will be resolved later).
    """
    return await (
        source.url.to_bedrock_content_block()
        if isinstance(source, URLImageSource)
        else source.data.to_bedrock_content_block(content_type=source.media_type)
    )


async def _map_tool_result_to_bedrock(
    block: ToolResultBlockParam,
) -> ContentBlockTypeDef:
    """Convert an Anthropic tool result block to a Bedrock tool result content block.

    Args:
        block: Anthropic tool result block param.

    Returns:
        Bedrock content block dict with ``toolResult`` key.
    """
    content_parts: list[ContentBlockTypeDef]
    match block.content:
        case str(text):
            content_parts = [{"text": text}]
        case _:
            content_parts = [
                {"text": part.text}
                if isinstance(part, TextBlockParam)
                else await _map_image_to_bedrock(part.source)
                for part in block.content
            ]

    result: ToolResultBlockTypeDef = {
        "toolUseId": block.tool_use_id.removeprefix("toolu_"),
        "content": content_parts,  # type: ignore[typeddict-item]
    }
    if block.is_error:
        result["status"] = "error"
    return {"toolResult": result}


async def _map_document_to_bedrock(block: DocumentBlockParam) -> ContentBlockTypeDef:
    """Convert an Anthropic document block to a Bedrock content block.

    For PDF sources (base64 or URL), calls to_bedrock_content_block.
    For plain-text sources, returns a fully built document block immediately.

    Args:
        block: Anthropic document block param.

    Returns:
        Bedrock content block dict.

    Raises:
        ApiError: If the document source type is unsupported.
    """
    doc_name = _RE_DOC_NAME.sub("_", block.title or "document")[:200]
    match block.source:
        case Base64PDFSource(data=data):
            return await data.to_bedrock_content_block(
                filename=doc_name,
                content_type="application/pdf",
                context=block.context or None,
                citations_enabled=bool(block.citations and block.citations.enabled),
            )
        case URLPDFSource(url=url):
            return await url.to_bedrock_content_block(
                filename=doc_name,
                context=block.context or None,
                citations_enabled=bool(block.citations and block.citations.enabled),
            )
        case PlainTextSourceParam(data=data):
            doc_bytes = data.encode("utf-8")
        case ContentBlockSourceParam(content=str(text)):
            doc_bytes = text.encode("utf-8")
        case ContentBlockSourceParam(content=content):
            doc_bytes = "\n".join(
                part.text for part in content if isinstance(part, TextBlockParam)
            ).encode("utf-8")
        case _:  # pragma: no cover
            msg = f"Unsupported document source type: {type(block.source)}"
            raise ApiError(msg)
    doc: dict[str, object] = {
        "name": doc_name,
        "format": "txt",
        "source": {"bytes": doc_bytes},
    }
    if block.context:
        doc["context"] = block.context
    if block.citations and block.citations.enabled:
        doc["citations"] = {"enabled": True}
    return {"document": doc}  # type: ignore[typeddict-item]


async def _map_content_block_to_bedrock(  # noqa: PLR0911
    block: ContentBlockParam,
) -> ContentBlockTypeDef | None:
    """Convert a single Anthropic content block to a Bedrock content block.

    Returns ``None`` for unrecognized blocks (``RedactedThinkingBlockParam``
    is handled separately by the caller).

    Args:
        block: Any Anthropic content block param variant.

    Returns:
        Bedrock content block dict, or ``None`` when the caller must handle it.

    Raises:
        ApiError: If the content block type is unsupported.
    """
    match block:
        case TextBlockParam(text=text):
            return {"text": text}
        case ImageBlockParam(source=source):
            return await _map_image_to_bedrock(source)
        case DocumentBlockParam():
            return await _map_document_to_bedrock(block)
        case SearchResultBlockParam(source=source, title=title, content=sr_blocks):
            return {
                "searchResult": {
                    "source": source,
                    "title": title,
                    "content": [{"text": b.text} for b in sr_blocks if b.text],
                }
            }
        case (
            ToolUseBlockParam(id=id_, name=name, input=input_)
            | ServerToolUseBlockParam(id=id_, name=name, input=input_)
        ):
            return {
                "toolUse": {
                    "toolUseId": id_.removeprefix("toolu_"),
                    "name": name,
                    "input": input_,
                }
            }
        case ToolResultBlockParam():
            return await _map_tool_result_to_bedrock(block)
        case ThinkingBlockParam(thinking=thinking, signature=signature):
            reasoning_text: ReasoningTextBlockTypeDef = {"text": thinking}
            if signature:
                reasoning_text["signature"] = signature
            return {"reasoningContent": {"reasoningText": reasoning_text}}
        case RedactedThinkingBlockParam():
            return None
        case _:
            msg = (
                f"Unsupported content block type: {getattr(block, 'type', type(block))}"
            )
            raise ApiError(msg)


async def _map_messages(
    messages: list[MessageParam], *, allow_explicit_caching: bool = False
) -> list[MessageTypeDef]:
    """Convert a list of Anthropic messages to Bedrock messages.

    Args:
        messages: Anthropic message params.
        allow_explicit_caching: Whether to allow explicit prompt caching for messages.

    Returns:
        List of Bedrock message dicts.
    """
    result: list[MessageTypeDef] = []
    for msg in messages:
        match msg.content:
            case str(text):
                content: list[ContentBlockTypeDef] = [{"text": text}]
            case _:
                content = []
                for block in msg.content:
                    if (
                        converted := await _map_content_block_to_bedrock(block)
                    ) is not None:
                        content.append(converted)
                    elif isinstance(block, RedactedThinkingBlockParam):
                        content.append(
                            {
                                "reasoningContent": {
                                    "redactedContent": await b64decode(block.data)
                                }
                            }
                        )
                    if (
                        allow_explicit_caching
                        and hasattr(block, "cache_control")
                        and (cache_control := block.cache_control)
                    ):
                        content.append(_build_cache_point(cache_control))
        result.append({"role": msg.role, "content": content})
    return result


def _map_tool_spec(tool: ToolUnionParam) -> ToolTypeDef | None:
    """Convert an Anthropic tool definition to a Bedrock tool type.

    Args:
        tool: Anthropic tool param (custom, bash, editor, web search, etc.).

    Returns:
        Bedrock tool type dict, or ``None`` if the tool is a system tool
        (handled separately).

    Raises:
        ApiError: If the tool type is unsupported.
    """
    match tool:
        case ToolParam(name=name, description=description, input_schema=input_schema):
            return {
                "toolSpec": {
                    "name": name,
                    "description": description or "custom",
                    "inputSchema": {"json": input_schema.model_dump(exclude_none=True)},
                }
            }
        case (
            WebSearchToolParam()
            | ToolBashParam()
            | ToolTextEditorParam()
            | CodeExecutionToolParam()
            | MemoryToolParam()
            | WebFetchToolParam()
            | ToolSearchToolBm25Param()
            | ToolSearchToolRegexParam()
        ):
            return None
        case _:  # pragma: no cover
            msg = f"Unsupported tool type: {getattr(tool, 'type', type(tool))}"
            raise ApiError(msg)


def _map_tool_choice(tool_choice: ToolChoiceParam | None) -> ToolChoiceTypeDef | None:
    """Convert an Anthropic tool choice param to a Bedrock tool choice.

    Args:
        tool_choice: Anthropic tool choice param, or ``None``.

    Returns:
        Bedrock tool choice dict, or ``None`` if not specified.

    Raises:
        ApiError: If ``ToolChoiceNoneParam`` is used (not supported by Bedrock).
    """
    match tool_choice:
        case ToolChoiceAutoParam():
            return {"auto": {}}
        case ToolChoiceAnyParam():
            return {"any": {}}
        case ToolChoiceToolParam(name=name):
            return {"tool": {"name": name}}
        case ToolChoiceNoneParam():
            msg = "tool_choice 'none' is not supported by this implementation. Remove tools from the request instead."
            raise ApiError(msg)
        case None:
            return None


def _build_tool_config(
    tools: list[ToolUnionParam] | None,
    tool_choice: ToolChoiceParam | None,
    *,
    allow_explicit_caching: bool = False,
) -> tuple[ToolConfigurationTypeDef | None, list[ToolUnionParam]]:
    """Build a Bedrock tool configuration from Anthropic tools and tool choice.

    Args:
        tools: List of Anthropic tool params, or ``None``.
        tool_choice: Anthropic tool choice param, or ``None``.
        allow_explicit_caching: Whether to allow prompt caching for tools.

    Returns:
        A 2-tuple of (Bedrock tool configuration dict or ``None``,
        list of system tools extracted from the input).
    """
    if not tools:
        return None, []

    tool_list: list[ToolTypeDef] = []
    system_tools: list[ToolUnionParam] = []
    cache_control: CacheControlEphemeralParam | None = None
    for tool in tools:
        tool_bedrock = _map_tool_spec(tool)
        if tool_bedrock is None:
            system_tools.append(tool)
            continue
        if hasattr(tool, "cache_control") and tool.cache_control:
            cache_control = cache_control or tool.cache_control
        tool_list.append(tool_bedrock)
    if cache_control and allow_explicit_caching:
        tool_list.append(_build_cache_point(cache_control))  # type: ignore[arg-type]

    tool_config: ToolConfigurationTypeDef | None = None
    if tool_list:
        tool_config = {"tools": tool_list}
        if bedrock_tool_choice := _map_tool_choice(tool_choice):
            tool_config["toolChoice"] = bedrock_tool_choice
    return tool_config, system_tools


def _build_output_config(
    output_config: OutputConfigParam | None,
) -> JsonSchemaDefinitionTypeDef | None:
    """Build a Bedrock ``outputConfig`` from an Anthropic ``OutputConfigParam``.

    Args:
        output_config: Anthropic output configuration.

    Returns:
        Bedrock outputConfig dict, or ``None`` if no format is specified.
    """
    match output_config:
        case None | OutputConfigParam(format=None):
            return None
        case OutputConfigParam(format=format_spec) if format_spec is not None:
            schema = format_spec.schema_
            return {
                "schema": schema
                if isinstance(schema, str)
                else to_json(schema).decode()
            }
        case _:  # pragma: no cover
            return None


async def translate_request(
    request: MessageCreateParams,
    model_id: str,
    *,
    prompt_caching_supported: bool,
    prompt_caching_tool_supported: bool,
) -> tuple[
    list[MessageTypeDef],
    list[SystemContentBlockTypeDef],
    InferenceConfigurationTypeDef,
    JsonMapping,
    ToolConfigurationTypeDef | None,
    ServiceTierTypeType | None,
    frozenset[PromptCaching] | None,
    CacheTTLType | None,
    JsonSchemaDefinitionTypeDef | None,
    list[ToolUnionParam],
]:
    """Translate an Anthropic ``MessageCreateParams`` into Bedrock Converse inputs.

    Args:
        request: Anthropic message creation parameters.
        model_id: Bedrock model identifier.
        prompt_caching_supported: True if prompt caching is supported by the model.
        prompt_caching_tool_supported: True if tool caching is supported by the model.

    Returns:
        A 10-tuple of (messages, system blocks, inference config,
        additional request fields, tool config, service tier, automatic cache control,
        automatic cache control ttl, output config, system tools).
    """
    if request.cache_control is None:
        allow_explicit_caching = prompt_caching_supported
        automatic_prompt_caching: frozenset[PromptCaching] | None = None
        automatic_prompt_caching_ttl: CacheTTLType | None = None
    else:
        allow_explicit_caching = False
        automatic_prompt_caching = (
            PROMPT_CACHING if prompt_caching_tool_supported else PROMPT_CACHING_BASIC
        )
        automatic_prompt_caching_ttl = request.cache_control.ttl

    additional_request_fields: JsonMapping = {}
    tool_config, system_tools = _build_tool_config(
        request.tools,
        request.tool_choice,
        allow_explicit_caching=allow_explicit_caching and prompt_caching_tool_supported,
    )
    return (
        await _map_messages(
            request.messages, allow_explicit_caching=allow_explicit_caching
        ),
        _map_system_blocks(
            request.system, allow_explicit_caching=allow_explicit_caching
        ),
        set_inference_configuration(
            model_id,
            additional_request_fields,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop_sequences=request.stop_sequences,
            top_k=request.top_k,
            **(request.model_extra or {}),
        ),
        additional_request_fields,
        tool_config,
        _SERVICES_TIERS.get(request.service_tier),
        automatic_prompt_caching,
        automatic_prompt_caching_ttl,
        _build_output_config(request.output_config),
        system_tools,
    )


def _map_citations_content_from_bedrock(
    citations_content: dict[str, Any],
) -> TextBlock | None:
    """Convert a Bedrock citations content block to an Anthropic text block with citations.

    Args:
        citations_content: Bedrock citations content block dict.

    Returns:
        Anthropic TextBlock with citations, or ``None`` if no content.
    """
    text = "".join(
        item["text"] for item in citations_content.get("content", []) if "text" in item
    )
    citations_list = _map_citations_from_bedrock(citations_content.get("citations", []))
    return TextBlock(type="text", text=text, citations=citations_list or None)


def _map_search_result_from_bedrock(
    search_result: dict[str, Any],
) -> WebSearchResultBlock:
    """Convert a Bedrock search result block to an Anthropic web search result block.

    Args:
        search_result: Bedrock search result block dict.

    Returns:
        Anthropic WebSearchResultBlock.
    """
    return WebSearchResultBlock(
        type="web_search_result",
        url=search_result.get("source", ""),
        title=search_result.get("title", ""),
    )


def _map_citations_from_bedrock(citations: list[dict[str, Any]]) -> list[TextCitation]:
    """Convert Bedrock citation outputs to Anthropic text citations.

    Args:
        citations: List of Bedrock citation output dicts.

    Returns:
        List of Anthropic text citation objects.
    """
    result: list[TextCitation] = []
    for citation in citations:
        cited_text = next(
            (
                item["text"]
                for item in citation.get("sourceContent", [])
                if "text" in item
            ),
            "",
        )
        title = citation.get("title")
        source = citation.get("source", "")
        location = citation.get("location", {})

        match location:
            case {"documentChar": loc}:
                result.append(
                    CitationCharLocation(
                        type="char_location",
                        cited_text=cited_text,
                        document_index=loc.get("documentIndex", 0),
                        document_title=title,
                        start_char_index=loc.get("start", 0),
                        end_char_index=loc.get("end", 0),
                    )
                )
            case {"documentPage": loc}:
                result.append(
                    CitationPageLocation(
                        type="page_location",
                        cited_text=cited_text,
                        document_index=loc.get("documentIndex", 0),
                        document_title=title,
                        start_page_number=loc.get("start", 0),
                        end_page_number=loc.get("end", 0),
                    )
                )
            case {"documentChunk": loc}:
                result.append(
                    CitationContentBlockLocation(
                        type="content_block_location",
                        cited_text=cited_text,
                        document_index=loc.get("documentIndex", 0),
                        document_title=title,
                        start_block_index=loc.get("start", 0),
                        end_block_index=loc.get("end", 0),
                    )
                )
            case {"web": loc}:
                result.append(
                    CitationsWebSearchResultLocation(
                        type="web_search_result_location",
                        cited_text=cited_text,
                        url=loc.get("url", source),
                        title=title,
                        encrypted_index="",
                    )
                )
            case {"searchResultLocation": loc}:
                result.append(
                    CitationsSearchResultLocation(
                        type="search_result_location",
                        cited_text=cited_text,
                        search_result_index=loc.get("searchResultIndex", 0),
                        source=source,
                        title=title,
                        start_block_index=loc.get("start", 0),
                        end_block_index=loc.get("end", 0),
                    )
                )
    return result


async def _map_content_block_from_bedrock(  # noqa: PLR0911
    block: ContentBlockOutputTypeDef,
) -> ContentBlock | None:
    """Convert a Bedrock output content block to an Anthropic content block.

    Args:
        block: Bedrock output content block dict.

    Returns:
        Anthropic content block, or ``None`` if the block type is unrecognized.
    """
    match block:
        case {"text": text}:
            return TextBlock(type="text", text=text)
        case {"toolUse": {"toolUseId": id_, "name": name, "input": input_}} if (
            name in SERVER_TOOL_NAMES
        ):
            return ServerToolUseBlock(
                type="server_tool_use", id=f"toolu_{id_}", name=name, input=input_
            )
        case {"toolUse": {"toolUseId": id_, "name": name, "input": input_}}:
            return ToolUseBlock(
                type="tool_use", id=f"toolu_{id_}", name=name, input=input_
            )
        case {"reasoningContent": {"reasoningText": reasoning_text}}:
            return ThinkingBlock(
                type="thinking",
                thinking=reasoning_text["text"],
                signature=reasoning_text.get("signature", ""),
            )
        case {"reasoningContent": {"redactedContent": bytes() as data}}:
            return RedactedThinkingBlock(
                type="redacted_thinking", data=await b64encode(data)
            )
        case {"citationsContent": citations_content}:
            return _map_citations_content_from_bedrock(citations_content)  # type: ignore[arg-type]
        case {"searchResult": search_result}:
            return _map_search_result_from_bedrock(search_result)  # type: ignore[arg-type,return-value]
        case _:
            return None


async def format_response(
    contents: list[ContentBlockOutputTypeDef],
    stop_reason: StopReasonType | None,
    usage: dict[str, int],
    message_id: str,
    model_id: str,
    forced_tool: str | None,
) -> Message:
    """Format a Bedrock Converse response as an Anthropic ``Message``.

    Args:
        contents: List of Bedrock output content blocks.
        stop_reason: Bedrock stop reason literal.
        usage: Token usage dict from Bedrock.
        message_id: Unique message identifier.
        model_id: Model identifier to echo back in the response.
        forced_tool: When set, only tool_use blocks with this name are kept.

    Returns:
        Anthropic Message object.
    """
    content_blocks = [
        mapped
        for block in contents
        if (mapped := await _map_content_block_from_bedrock(block)) is not None
    ]

    if forced_tool is not None:
        content_blocks = [
            b for b in content_blocks if b.type != "tool_use" or b.name == forced_tool
        ]

    anthropic_usage = Usage(
        input_tokens=usage.get("inputTokens", 0),
        output_tokens=usage.get("outputTokens", 0),
        cache_read_input_tokens=usage.get("cacheReadInputTokens"),
        cache_creation_input_tokens=usage.get("cacheCreationInputTokens"),
    )

    return Message(
        id=message_id,
        type="message",
        role="assistant",
        content=content_blocks,
        model=model_id,
        stop_reason=_map_stop_reason(stop_reason),
        usage=anthropic_usage,
    )


def _make_block_start_event(
    index: int, content_block: ContentBlock
) -> JSONServerSentEvent:
    """Create a ``content_block_start`` SSE event.

    Args:
        index: Zero-based content block index.
        content_block: Anthropic content block to include.

    Returns:
        JSON server-sent event.
    """
    return JSONServerSentEvent(
        data=RawContentBlockStartEvent(
            type="content_block_start", index=index, content_block=content_block
        ).model_dump(mode="json", exclude_none=True),
        event="content_block_start",
    )


def _make_block_delta_event(
    index: int, delta: TextDelta | ThinkingDelta | SignatureDelta | InputJSONDelta
) -> JSONServerSentEvent:
    """Create a ``content_block_delta`` SSE event.

    Args:
        index: Zero-based content block index.
        delta: Delta payload (text, thinking, or input JSON).

    Returns:
        JSON server-sent event.
    """
    return JSONServerSentEvent(
        data=RawContentBlockDeltaEvent(
            type="content_block_delta", index=index, delta=delta
        ).model_dump(mode="json", exclude_none=True),
        event="content_block_delta",
    )


def _resolve_start_block(start: ContentBlockStartTypeDef) -> ContentBlock:
    """Determine the content block type from a ``contentBlockStart`` payload.

    Args:
        start: Bedrock content block start dict.

    Returns:
        Anthropic content block (text, tool_use, or thinking).
    """
    match start:
        case {"toolUse": {"toolUseId": id_, "name": name}} if name in SERVER_TOOL_NAMES:
            return ServerToolUseBlock(
                type="server_tool_use", id=f"toolu_{id_}", name=name, input={}
            )
        case {"toolUse": {"toolUseId": id_, "name": name}}:
            return ToolUseBlock(type="tool_use", id=f"toolu_{id_}", name=name, input={})
        case {"reasoningContent": _}:
            return ThinkingBlock(type="thinking", thinking="", signature="")
        case _:
            return TextBlock(type="text", text="")


def _synthesize_block_from_delta(delta: ContentBlockDeltaTypeDef) -> ContentBlock:
    """Infer a synthetic start block from a delta payload.

    Args:
        delta: Bedrock content block delta dict.

    Returns:
        Anthropic content block matching the delta type.
    """
    match delta:
        case {"reasoningContent": _}:
            return ThinkingBlock(type="thinking", thinking="", signature="")
        case {"toolUse": _}:
            return ToolUseBlock(type="tool_use", id="", name="", input={})
        case _:
            return TextBlock(type="text", text="")


def _map_delta(
    index: int, delta: ContentBlockDeltaTypeDef
) -> JSONServerSentEvent | None:
    """Convert a Bedrock content block delta to an Anthropic SSE delta event.

    Args:
        index: Zero-based content block index.
        delta: Bedrock content block delta dict.

    Returns:
        JSON server-sent event, or ``None`` if the delta should be skipped.
    """
    match delta:
        case {"text": text}:
            return _make_block_delta_event(
                index, TextDelta(type="text_delta", text=text)
            )
        case {"reasoningContent": {"text": text}}:
            return _make_block_delta_event(
                index, ThinkingDelta(type="thinking_delta", thinking=text)
            )
        case {"reasoningContent": {"signature": signature}}:
            return _make_block_delta_event(
                index, SignatureDelta(type="signature_delta", signature=signature)
            )
        case {"toolUse": {"input": partial_json}}:
            return _make_block_delta_event(
                index,
                InputJSONDelta(type="input_json_delta", partial_json=partial_json),
            )
        case _:
            return None


def _make_message_start_event(message_id: str, model_id: str) -> JSONServerSentEvent:
    """Create the initial ``message_start`` SSE event.

    Args:
        message_id: Unique message identifier.
        model_id: Model identifier to echo back.

    Returns:
        JSON server-sent event.
    """
    return JSONServerSentEvent(
        data=log_response_params(
            RawMessageStartEvent(
                type="message_start",
                message=Message(
                    id=message_id,
                    type="message",
                    role="assistant",
                    content=[],
                    model=model_id,
                    stop_reason=None,
                    usage=Usage(input_tokens=0, output_tokens=0),
                ),
            ).model_dump(mode="json", exclude_none=True)
        ),
        event="message_start",
    )


def _make_block_stop_event(index: int) -> JSONServerSentEvent:
    """Create a ``content_block_stop`` SSE event.

    Args:
        index: Zero-based content block index.

    Returns:
        JSON server-sent event.
    """
    return JSONServerSentEvent(
        data=RawContentBlockStopEvent(
            type="content_block_stop", index=index
        ).model_dump(mode="json", exclude_none=True),
        event="content_block_stop",
    )


def _make_message_delta_event(
    stop_reason: StopReason | None, usage_data: dict[str, int]
) -> JSONServerSentEvent:
    """Create the ``message_delta`` SSE event.

    Args:
        stop_reason: Final stop reason for the message.
        usage_data: Token usage data from Bedrock metadata.

    Returns:
        JSON server-sent event.
    """
    return JSONServerSentEvent(
        data=RawMessageDeltaEvent(
            type="message_delta",
            delta=MessageDelta(stop_reason=stop_reason),
            usage=MessageDeltaUsage(
                output_tokens=usage_data.get("outputTokens", 0),
                input_tokens=usage_data.get("inputTokens", 0),
                cache_read_input_tokens=usage_data.get("cacheReadInputTokens"),
                cache_creation_input_tokens=usage_data.get("cacheCreationInputTokens"),
            ),
        ).model_dump(mode="json", exclude_none=True),
        event="message_delta",
    )


def _is_suppressed_tool(content_block: ContentBlock, forced_tool: str | None) -> bool:
    """Return True if this block should be suppressed due to forced_tool filtering.

    Args:
        content_block: Resolved Anthropic content block.
        forced_tool: The only tool name that should be kept, or ``None`` to keep all.

    Returns:
        True when the block is a tool_use whose name doesn't match ``forced_tool``.
    """
    return (
        forced_tool is not None
        and content_block.type == "tool_use"
        and content_block.name != forced_tool
    )


async def _handle_block_start(
    start_block: ContentBlockStartEventTypeDef,
    forced_tool: str | None,
    seen: set[int],
    suppressed: set[int],
) -> JSONServerSentEvent | None:
    """Handle a ``contentBlockStart`` Bedrock event.

    Args:
        start_block: Raw Bedrock contentBlockStart payload.
        forced_tool: Tool name filter; blocks for other tools are suppressed.
        seen: Set of content block indices already started.
        suppressed: Set of content block indices being suppressed.

    Returns:
        A ``content_block_start`` SSE event, or ``None`` if the block is suppressed.
    """
    index = start_block["contentBlockIndex"]
    content_block = _resolve_start_block(start_block["start"])
    if _is_suppressed_tool(content_block, forced_tool):
        suppressed.add(index)
        return None
    seen.add(index)
    return _make_block_start_event(index, content_block)


async def _handle_block_delta(
    delta_block: ContentBlockDeltaEventTypeDef, seen: set[int], suppressed: set[int]
) -> AsyncGenerator[JSONServerSentEvent]:
    """Handle a ``contentBlockDelta`` Bedrock event.

    Args:
        delta_block: Raw Bedrock contentBlockDelta payload.
        seen: Set of content block indices already started.
        suppressed: Set of content block indices being suppressed.

    Yields:
        Zero, one, or two SSE events (synthetic start + delta).
    """
    index = delta_block["contentBlockIndex"]
    if index in suppressed:
        return
    delta = delta_block["delta"]
    if index not in seen:
        seen.add(index)
        yield _make_block_start_event(index, _synthesize_block_from_delta(delta))
    if delta_event := _map_delta(index, delta):
        yield delta_event


async def _process_stream_events(
    stream: AsyncIterator[ConverseStreamOutputTypeDef], forced_tool: str | None
) -> AsyncGenerator[JSONServerSentEvent]:
    """Process Bedrock stream events and yield Anthropic SSE events.

    Handles ``contentBlockStart``, ``contentBlockDelta``, ``contentBlockStop``,
    ``messageStop``, and ``metadata`` events, yielding corresponding Anthropic
    SSE events.

    Args:
        stream: Async iterator of Bedrock stream events.
        forced_tool: When set, tool_use blocks for other tools are suppressed.

    Yields:
        Anthropic SSE events, finishing with a ``message_delta`` event.
    """
    stop_reason: StopReason | None = None
    usage_data: dict[str, int] = {}
    seen: set[int] = set()
    suppressed: set[int] = set()

    async for event in stream:
        match event:
            case {"contentBlockStart": start_block}:
                if sse := await _handle_block_start(
                    start_block, forced_tool, seen, suppressed
                ):
                    yield sse
            case {"contentBlockDelta": delta_block}:
                async for sse in _handle_block_delta(delta_block, seen, suppressed):
                    yield sse
            case {"contentBlockStop": stop_block}:
                index = stop_block["contentBlockIndex"]
                if index in suppressed:
                    suppressed.discard(index)
                else:
                    yield _make_block_stop_event(index)
            case {"messageStop": message_stop}:
                stop_reason = _map_stop_reason(message_stop["stopReason"])
            case {"metadata": metadata}:
                usage_data = metadata["usage"]  # type: ignore[assignment]

    yield _make_message_delta_event(stop_reason, usage_data)


async def format_stream(
    message_id: str,
    model_id: str,
    stream: AsyncIterator[ConverseStreamOutputTypeDef],
    forced_tool: str | None,
) -> AsyncGenerator[JSONServerSentEvent]:
    """Convert a Bedrock Converse stream into Anthropic SSE events.

    Args:
        message_id: Unique message identifier.
        model_id: Model identifier to echo back in the response.
        stream: Async iterator of Bedrock stream events.
        forced_tool: When set, tool_use blocks for other tools are suppressed.

    Yields:
        JSON server-sent events in Anthropic streaming format.
    """
    yield _make_message_start_event(message_id, model_id)
    async for event in _process_stream_events(stream, forced_tool):
        yield event
    yield JSONServerSentEvent(
        data=RawMessageStopEvent(type="message_stop").model_dump(
            mode="json", exclude_none=True
        ),
        event="message_stop",
    )


async def count_tokens_via_bedrock(
    request: MessageCountTokensParams, model_id: str, region: RegionName
) -> int:
    """Count tokens using the AWS Bedrock Runtime CountTokens API.

    Builds a Converse-compatible input from the Anthropic request and calls
    the Bedrock ``count_tokens`` API for an accurate, model-specific count.

    Args:
        request: The count tokens request containing messages, system prompt, and tools.
        model_id: The Bedrock model identifier.
        region: The AWS region of the model.

    Returns:
        The total number of input tokens.
    """
    req: ConverseTokensRequestTypeDef = {
        "messages": await _map_messages(request.messages)
    }
    if system_blocks := _map_system_blocks(request.system):
        req["system"] = system_blocks
    tool_config, _system_tools = _build_tool_config(request.tools, request.tool_choice)
    if tool_config:
        req["toolConfig"] = tool_config

    with handle_bedrock_client_error():
        resp: CountTokensResponseTypeDef = await get_client(
            "bedrock-runtime", region
        ).count_tokens(modelId=model_id, input={"converse": req})
    return resp["inputTokens"]
