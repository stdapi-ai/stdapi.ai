"""Anthropic Messages API adapter for Bedrock Converse.

Translates between Anthropic Messages API request/response types and
Bedrock Converse API-native types. Anthropic's content block format is
close to Bedrock's native format, so many mappings are near 1:1.
"""

import re
from dataclasses import dataclass, field
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
    FileSource,
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
    ToolComputerParam,
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
    from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping

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
        ContentBlockStopEventTypeDef,
        ContentBlockTypeDef,
        ConverseStreamOutputTypeDef,
        ConverseTokensRequestTypeDef,
        CountTokensResponseTypeDef,
        DocumentBlockTypeDef,
        InferenceConfigurationTypeDef,
        JsonSchemaDefinitionTypeDef,
        MessageTypeDef,
        ReasoningTextBlockTypeDef,
        SystemContentBlockTypeDef,
        ToolChoiceTypeDef,
        ToolConfigurationTypeDef,
        ToolResultBlockTypeDef,
        ToolResultContentBlockOutputTypeDef,
        ToolTypeDef,
    )

    from stdapi.models.chat import ReasoningParams
    from stdapi.types import JsonMapping
    from stdapi.types.anthropic_messages import ServerTools


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


#: Bedrock stop reasons to Anthropic stop reasons mapping.
_STOP_REASONS: dict[StopReasonType | str | None, StopReason] = {
    "end_turn": "end_turn",
    "max_tokens": "max_tokens",
    "model_context_window_exceeded": "max_tokens",
    "stop_sequence": "stop_sequence",
    "tool_use": "tool_use",
    "content_filtered": "refusal",
    "guardrail_intervened": "refusal",
    "malformed_model_output": "refusal",
    "malformed_tool_use": "refusal",
    # Non-standard but observed
    "incomplete": "max_tokens",
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


def _map_stop_reason(stop_reason: StopReasonType | str | None) -> StopReason:
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
    source: Base64ImageSource | URLImageSource | FileSource,
) -> ContentBlockTypeDef:
    """Convert an Anthropic image source to a Bedrock content block.

    Args:
        source: Base64-encoded, URL, or Files API image source.

    Returns:
        A partial content block dict (source will be resolved later).
    """
    match source:
        case FileSource(file_id=file_input):
            return await file_input.to_bedrock_content_block()
        case URLImageSource(url=url):
            return await url.to_bedrock_content_block()
        case Base64ImageSource(data=data, media_type=media_type):
            return await data.to_bedrock_content_block(content_type=media_type)
        case _:  # pragma: no cover
            msg = f"Unsupported image source type: {type(source)}"
            raise ApiError(msg)


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
        case FileSource(file_id=file_input):
            return await file_input.to_bedrock_content_block(
                filename=doc_name,
                context=block.context or None,
                citations_enabled=bool(block.citations and block.citations.enabled),
            )
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
    doc: DocumentBlockTypeDef = {
        "name": doc_name,
        "format": "txt",
        "source": {"bytes": doc_bytes},
    }
    if block.context:
        doc["context"] = block.context
    if block.citations and block.citations.enabled:
        doc["citations"] = {"enabled": True}
    return {"document": doc}


def _map_thinking_to_bedrock(
    thinking: str, signature: str | None
) -> ContentBlockTypeDef:
    """Map a ThinkingBlockParam to a Bedrock reasoningContent block.

    Args:
        thinking: The thinking text content.
        signature: Optional signature for the reasoning block.

    Returns:
        Bedrock content block dict with ``reasoningContent``.
    """
    reasoning_text: ReasoningTextBlockTypeDef = {"text": thinking}
    if signature:
        reasoning_text["signature"] = signature
    return {"reasoningContent": {"reasoningText": reasoning_text}}


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
            return {"text": text} if text else None
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
        case ServerToolUseBlockParam(id=id_, name=name, input=input_):
            return {
                "toolUse": {
                    "toolUseId": f"tooluse_{id_.removeprefix('srvtoolu_')}",
                    "name": name,
                    "input": input_,
                }
            }
        case ToolUseBlockParam(id=id_, name=name, input=input_):
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
            return _map_thinking_to_bedrock(thinking, signature)
        case RedactedThinkingBlockParam():
            return None
        case _:
            msg = (
                f"Unsupported content block type: {getattr(block, 'type', type(block))}"
            )
            raise ApiError(msg)


def _extract_system_messages(
    messages: list[MessageParam],
) -> tuple[list[MessageParam], list[TextBlockParam]]:
    """Partition messages into non-system messages and system text blocks.

    Args:
        messages: Anthropic message params, possibly including system-role entries.

    Returns:
        A 2-tuple of (non_system_messages, system_blocks) where system_blocks
        contains the content of any system-role messages as TextBlockParams.
    """
    non_system: list[MessageParam] = []
    system_blocks: list[TextBlockParam] = []
    for msg in messages:
        match msg.role:
            case "system":
                match msg.content:
                    case str(text):
                        system_blocks.append(TextBlockParam(type="text", text=text))
                    case list(blocks):
                        system_blocks.extend(
                            b for b in blocks if isinstance(b, TextBlockParam)
                        )
            case _:
                non_system.append(msg)
    return non_system, system_blocks


def _merge_system_content(
    existing: str | list[TextBlockParam] | None, extracted: list[TextBlockParam]
) -> str | list[TextBlockParam] | None:
    """Merge top-level system content with blocks extracted from system-role messages.

    Args:
        existing: The top-level system field value.
        extracted: System blocks extracted from system-role messages.

    Returns:
        Merged system content, or the original value if nothing was extracted.
    """
    if not extracted:
        return existing
    match existing:
        case str(s):
            return [TextBlockParam(type="text", text=s), *extracted]
        case list(blocks):
            return [*blocks, *extracted]
        case _:
            return extracted


def _prepare_messages_and_system(
    messages: list[MessageParam],
    system: str | list[TextBlockParam] | None,
    *,
    system_message_as_messages: bool,
) -> tuple[list[MessageParam], str | list[TextBlockParam] | None]:
    """Resolve the messages list and system content based on model capability.

    Args:
        messages: Input message list, possibly containing system-role entries.
            System-role messages are mid-conversation system instructions valid
            only after the first user turn (Claude Opus 4.8+).
        system: Top-level system field value.
        system_message_as_messages: When True (Claude Opus 4.8+), pass system-role
            messages through as-is — the model handles them natively.  When False,
            extract them and merge their content into the system field as a fallback.

    Returns:
        A 2-tuple of (resolved_messages, resolved_system).
    """
    if system_message_as_messages:
        return messages, system
    non_system, system_blocks = _extract_system_messages(messages)
    return non_system, _merge_system_content(system, system_blocks)


async def _map_messages(
    messages: list[MessageParam],
    *,
    allow_explicit_caching: bool = False,
    allow_tool_caching: bool = True,
    req_map_content_block: Callable[[ContentBlockParam], ContentBlockTypeDef | None]
    | None = None,
) -> list[MessageTypeDef]:
    """Convert a list of Anthropic messages to Bedrock messages.

    Args:
        messages: Anthropic message params (system-role messages must already be
            extracted via ``_extract_system_messages`` before calling this).
        allow_explicit_caching: Whether to allow explicit prompt caching for messages.
        allow_tool_caching: Whether cache points may follow tool-use or
            tool-result blocks (some models reject them there).
        req_map_content_block: Optional model-specific callback for content block
            translation.  Called before the default mapper; return a
            ``ContentBlockTypeDef`` to use that result, or ``None`` to fall
            back to ``_map_content_block_to_bedrock``.

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
                        req_map_content_block is not None
                        and (override := req_map_content_block(block)) is not None
                    ):
                        content.append(override)
                    elif (
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
                        and (
                            allow_tool_caching
                            or not isinstance(
                                block, (ToolResultBlockParam, ToolUseBlockParam)
                            )
                        )
                        and hasattr(block, "cache_control")
                        and (cache_control := block.cache_control)
                    ):
                        content.append(_build_cache_point(cache_control))
        result.append({"role": msg.role, "content": content})  # type: ignore[typeddict-item]
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
            | ToolComputerParam()
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


def _handle_system_tool(
    tool: ToolUnionParam,
    tool_list: list[ToolTypeDef],
    *,
    tool_name_map: Mapping[ServerTools, str] | None,
) -> None:
    """Append a ``toolSpec`` stub for an Anthropic server tool.

    The Bedrock name is looked up via *tool_name_map* when provided, otherwise
    ``tool.name`` is used verbatim.  The stub is later promoted to a Bedrock
    ``systemTool`` entry by ``_req_promote_system_tools`` (non-Claude models),
    or kept as a ``toolSpec`` entry in ``toolConfig`` by Claude models
    (multi-turn stub mode: required so Bedrock accepts ``toolResult`` blocks in
    multi-turn conversations).

    Args:
        tool: Anthropic server tool param.
        tool_list: Mutable Bedrock tool list to append to.
        tool_name_map: Anthropic → Bedrock name map.

    Raises:
        ApiError: If *tool_name_map* is provided and the tool name is absent.
    """
    if tool_name_map:
        if tool.name not in tool_name_map:
            tool_type = getattr(tool, "type", type(tool).__name__)
            msg = f"Server tool '{tool_type}' is not supported by this model."
            raise ApiError(msg)
        bedrock_name: str = tool_name_map[tool.name]  # type: ignore[index]
    else:
        bedrock_name = tool.name
    tool_list.append(
        {
            "toolSpec": {
                "name": bedrock_name,
                "description": bedrock_name,
                "inputSchema": {"json": {"type": "object"}},
            }
        }
    )


def _build_tool_config(
    tools: list[ToolUnionParam] | None,
    tool_choice: ToolChoiceParam | None,
    *,
    allow_explicit_caching: bool = False,
    tool_name_map: Mapping[ServerTools, str] | None = None,
) -> ToolConfigurationTypeDef | None:
    """Build a Bedrock tool configuration from Anthropic tools and tool choice.

    ``ToolParam`` entries become ``toolSpec`` entries; server tools become bare
    ``toolSpec`` stubs for downstream handling by ``_req_promote_system_tools``
    or ``_req_configure_tools``.

    Args:
        tools: Anthropic tool params, or ``None``.
        tool_choice: Anthropic tool choice, or ``None``.
        allow_explicit_caching: Append a cache-point after the last tool when a
            tool carries ``cache_control``.
        tool_name_map: Anthropic server tool name → Bedrock name translation map.

    Returns:
        Bedrock ``ToolConfigurationTypeDef``, or ``None`` when *tools* is empty.

    Raises:
        ApiError: If *tool_name_map* is provided and a server tool name is absent.
    """
    if not tools:
        return None

    tool_list: list[ToolTypeDef] = []
    cache_control: CacheControlEphemeralParam | None = None
    for tool in tools:
        tool_bedrock = _map_tool_spec(tool)
        if tool_bedrock is None:
            _handle_system_tool(tool, tool_list, tool_name_map=tool_name_map)
            continue
        if hasattr(tool, "cache_control") and tool.cache_control:
            cache_control = tool.cache_control or cache_control
        tool_list.append(tool_bedrock)
    if cache_control and allow_explicit_caching:
        tool_list.append(_build_cache_point(cache_control))  # type: ignore[arg-type]

    tool_config: ToolConfigurationTypeDef | None = None
    if tool_list:
        tool_config = {"tools": tool_list}
        if bedrock_tool_choice := _map_tool_choice(tool_choice):
            tool_config["toolChoice"] = bedrock_tool_choice
    return tool_config


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
    tool_name_map: Mapping[ServerTools, str] | None = None,
    req_map_content_block: Callable[[ContentBlockParam], ContentBlockTypeDef | None]
    | None = None,
    system_message_as_messages: bool = False,
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
]:
    """Translate an Anthropic ``MessageCreateParams`` into Bedrock Converse inputs.

    Args:
        request: Anthropic message creation parameters.
        model_id: Bedrock model identifier.
        prompt_caching_supported: True if prompt caching is supported by the model.
        prompt_caching_tool_supported: True if tool caching is supported by the model.
        tool_name_map: Optional mapping from Anthropic canonical tool name to Bedrock
            system tool name.  When provided, system tool names are translated to
            Bedrock names before being added as ``toolSpec`` entries.
        req_map_content_block: Optional model-specific callback for content block
            translation.  Passed through to ``_map_messages``.
        system_message_as_messages: When True, system-role messages are forwarded
            directly to Bedrock as messages instead of being extracted into the
            system-blocks field.  The top-level ``system`` field is always sent
            unchanged regardless of this flag.

    Returns:
        A 9-tuple of (messages, system blocks, inference config,
        additional request fields, tool config, service tier, automatic cache control,
        automatic cache control ttl, output config).
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
    tool_config = _build_tool_config(
        request.tools,
        request.tool_choice,
        allow_explicit_caching=allow_explicit_caching and prompt_caching_tool_supported,
        tool_name_map=tool_name_map,
    )
    messages, combined_system = _prepare_messages_and_system(
        request.messages,
        request.system,
        system_message_as_messages=system_message_as_messages,
    )
    return (
        await _map_messages(
            messages,
            allow_explicit_caching=allow_explicit_caching,
            allow_tool_caching=prompt_caching_tool_supported,
            req_map_content_block=req_map_content_block,
        ),
        _map_system_blocks(
            combined_system, allow_explicit_caching=allow_explicit_caching
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
    )


def extract_reasoning(request: MessageCreateParams) -> ReasoningParams | None:
    """Extract reasoning parameters from an Anthropic Messages request.

    Args:
        request: Anthropic message creation parameters.

    Returns:
        Reasoning parameters to configure, or None if the request has no
        reasoning-related field set.
    """
    if request.thinking is None and (
        request.output_config is None or request.output_config.effort is None
    ):
        return None
    return {
        "enabled": (
            request.thinking is not None
            and request.thinking.type in ("enabled", "adaptive")
        )
        or (
            request.output_config is not None
            and request.output_config.effort is not None
        ),
        "reasoning_effort": (
            request.output_config.effort if request.output_config is not None else None
        ),
        "budget_tokens": (
            request.thinking.budget_tokens
            if request.thinking is not None and request.thinking.type == "enabled"
            else None
        ),
        "max_tokens": request.max_tokens,
    }


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
        case {"text": text} if text:
            return TextBlock(type="text", text=text)
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
    resp_map_tool_result: Callable[
        [str, str, list[ToolResultContentBlockOutputTypeDef]], list[ContentBlock] | None
    ],
    resp_map_tool_use: Callable[[str, str, JsonMapping], ContentBlock | None]
    | None = None,
) -> Message:
    """Format a Bedrock Converse response as an Anthropic ``Message``.

    Args:
        contents: List of Bedrock output content blocks.
        stop_reason: Bedrock stop reason literal.
        usage: Token usage dict from Bedrock.
        message_id: Unique message identifier.
        model_id: Model identifier to echo back in the response.
        forced_tool: When set, only tool_use blocks with this name are kept.
        resp_map_tool_result: Callable that maps a Bedrock ``toolResult`` block to
            zero or more Anthropic content blocks.  Receives the raw ``toolUseId``,
            the Bedrock tool name, and the full content items list from the
            ``toolResult`` block.
        resp_map_tool_use: Optional callable that maps a Bedrock ``toolUse`` block to
            an Anthropic content block.  Receives the raw ``toolUseId``, the Bedrock
            tool name, and the tool input dict.  Return ``None`` to fall back to the
            default ``_map_content_block_from_bedrock`` mapping.

    Returns:
        Anthropic Message object.
    """
    # toolUseId → Bedrock tool name; populated as toolUse blocks are encountered
    # so that subsequent toolResult blocks can be resolved to a tool name.
    tool_use_id_to_name: dict[str, str] = {}
    content_blocks: list[ContentBlock] = []

    for block in contents:
        match block:
            case {"toolUse": {"toolUseId": id_, "name": name}}:
                tool_use_id_to_name[id_] = name
                if resp_map_tool_use is not None:
                    tool_input: JsonMapping = block["toolUse"].get("input", {})
                    override = resp_map_tool_use(id_, name, tool_input)
                else:
                    override = None
                if override is not None:
                    content_blocks.append(override)
                elif (
                    mapped := await _map_content_block_from_bedrock(block)
                ) is not None:
                    content_blocks.append(mapped)
            case {"toolResult": tool_result}:
                id_ = tool_result.get("toolUseId", "")
                content_items: list[ToolResultContentBlockOutputTypeDef] = (
                    tool_result.get("content") or []
                )
                content_blocks.extend(
                    resp_map_tool_result(
                        id_, tool_use_id_to_name.get(id_, ""), content_items
                    )
                    or ()
                )
            case _:
                if (mapped := await _map_content_block_from_bedrock(block)) is not None:
                    content_blocks.append(mapped)

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


def _handle_block_start(
    start: ContentBlockStartTypeDef,
    forced_tool: str | None,
    resp_stream_map_tool_use: Callable[[str, str], ContentBlock | None] | None = None,
) -> ContentBlock | None:
    """Resolve a ``contentBlockStart`` ``start`` payload to a ContentBlock.

    Args:
        start: The ``start`` field of a Bedrock ``contentBlockStart`` event.
        forced_tool: Tool name filter; blocks for other tools return ``None``.
        resp_stream_map_tool_use: Optional model-specific callback that maps a
            Bedrock ``toolUse`` start to an Anthropic content block.  Receives the
            raw ``toolUseId`` and Bedrock tool name; return ``None`` to use the
            default ``_resolve_start_block`` mapping.

    Returns:
        The resolved ContentBlock, or ``None`` if the block should be suppressed.
    """
    content_block: ContentBlock | None = None
    if resp_stream_map_tool_use is not None and "toolUse" in start:
        tool_use = start["toolUse"]
        content_block = resp_stream_map_tool_use(
            tool_use["toolUseId"], tool_use["name"]
        )
    if content_block is None:
        content_block = _resolve_start_block(start)
    if _is_suppressed_tool(content_block, forced_tool):
        return None
    return content_block


@dataclass(slots=True)
class _StreamState:
    """Mutable state shared across the :func:`_process_stream_events` event loop.

    Attributes:
        next_index: Next Anthropic block index to assign.
        current_index: Index assigned to the block currently in
            progress, or ``None`` when no block is open.
        current_suppressed: ``True`` when the current block is being dropped.
        pending_results: Buffered ``toolResult`` data keyed by Bedrock block
            index; filled from ``contentBlockDelta`` events and consumed on
            ``contentBlockStop``.
    """

    next_index: int = 0
    current_index: int | None = None
    current_suppressed: bool = False
    pending_results: dict[int, dict[str, Any]] = field(default_factory=dict)


def _process_content_block_start(
    start_block: ContentBlockStartEventTypeDef,
    state: _StreamState,
    forced_tool: str | None,
    resp_stream_map_tool_use: Callable[[str, str], ContentBlock | None] | None,
) -> list[JSONServerSentEvent]:
    """Handle a ``contentBlockStart`` event, updating *state* in place.

    ``toolResult`` blocks (model-internal results such as ``nova_code_interpreter``)
    are registered in :attr:`_StreamState.pending_results` for later assembly;
    all other blocks are resolved to an Anthropic :class:`ContentBlock` immediately.

    Args:
        start_block: The ``contentBlockStart`` value from the Bedrock stream event.
        state: Shared stream state; mutated in place.
        forced_tool: Tool name filter for block suppression.
        resp_stream_map_tool_use: Optional model-specific ``toolUse`` mapper.

    Returns:
        SSE events to emit (zero or one ``content_block_start`` event).
    """
    bedrock_index: int = start_block["contentBlockIndex"]
    start = start_block["start"]
    if "toolResult" in start:
        tool_result = start["toolResult"]
        state.pending_results[bedrock_index] = {
            "toolUseId": tool_result["toolUseId"],
            "result_type": tool_result.get("type", ""),
            "content_items": [],
        }
        return []
    if content_block := _handle_block_start(
        start, forced_tool, resp_stream_map_tool_use
    ):
        state.current_suppressed = False
        state.current_index = state.next_index
        state.next_index += 1
        return [_make_block_start_event(state.current_index, content_block)]
    state.current_suppressed = True
    state.current_index = None
    return []


def _emit_synthesized_block(
    index: int, delta: ContentBlockDeltaTypeDef
) -> list[JSONServerSentEvent]:
    """Synthesize a ``content_block_start`` + optional delta event for *index*."""
    events: list[JSONServerSentEvent] = [
        _make_block_start_event(index, _synthesize_block_from_delta(delta))
    ]
    if delta_event := _map_delta(index, delta):
        events.append(delta_event)
    return events


def _process_content_block_delta(
    delta_block: ContentBlockDeltaEventTypeDef, state: _StreamState
) -> list[JSONServerSentEvent]:
    """Handle a ``contentBlockDelta`` event, updating *state* in place.

    Accumulates payload into the pending buffer when the delta belongs to a
    buffered ``toolResult`` block.  For regular blocks, emits a delta SSE event,
    or synthesises a ``content_block_start`` when there was no prior start event.

    Args:
        delta_block: The ``contentBlockDelta`` value from the Bedrock stream event.
        state: Shared stream state; mutated in place.

    Returns:
        SSE events to emit (zero, one, or two events).
    """
    bedrock_index: int = delta_block["contentBlockIndex"]
    delta = delta_block["delta"]
    if bedrock_index in state.pending_results:
        if "toolResult" in delta:
            state.pending_results[bedrock_index]["content_items"].extend(
                delta["toolResult"]
            )
        return []
    if state.current_index is not None:
        if delta_event := _map_delta(state.current_index, delta):
            return [delta_event]
        return []
    # Empty delta: stay suppressed (or start suppression). The block is only
    # truly discarded if contentBlockStop arrives while still deferred (Nova-style
    # preamble). Non-empty arrivals in the same block (DeepSeek V3, Gemma) are
    # handled below by falling through to synthesize the start event.
    if delta == {"text": ""}:
        state.current_suppressed = True
        return []
    # Non-empty delta with no active block: synthesize a content_block_start now.
    # If we were suppressed (deferred on an empty first delta), un-suppress first.
    if state.current_suppressed:
        state.current_suppressed = False
    state.current_index = state.next_index
    state.next_index += 1
    return _emit_synthesized_block(state.current_index, delta)


def _process_content_block_stop(
    stop_block: ContentBlockStopEventTypeDef,
    state: _StreamState,
    resp_stream_map_tool_result: Callable[[str, str, list[Any]], ContentBlock | None]
    | None,
) -> list[JSONServerSentEvent]:
    """Handle a ``contentBlockStop`` event, updating *state* in place.

    For pending ``toolResult`` blocks, translates and emits the complete block via
    *resp_stream_map_tool_result* (if set) and discards it silently otherwise.
    For regular blocks, closes the open Anthropic block or clears the suppression flag.

    Args:
        stop_block: The ``contentBlockStop`` value from the Bedrock stream event.
        state: Shared stream state; mutated in place.
        resp_stream_map_tool_result: Optional model-specific ``toolResult`` mapper.

    Returns:
        SSE events to emit (zero, one, or two events).
    """
    bedrock_index: int = stop_block["contentBlockIndex"]
    if bedrock_index in state.pending_results:
        info = state.pending_results.pop(bedrock_index)
        if (
            resp_stream_map_tool_result is not None
            and (
                content_block := resp_stream_map_tool_result(
                    info["toolUseId"], info["result_type"], info["content_items"]
                )
            )
            is not None
        ):
            anthropic_index = state.next_index
            state.next_index += 1
            return [
                _make_block_start_event(anthropic_index, content_block),
                _make_block_stop_event(anthropic_index),
            ]
        return []
    if state.current_suppressed:
        state.current_suppressed = False
        return []
    if state.current_index is not None:
        index = state.current_index
        state.current_index = None
        return [_make_block_stop_event(index)]
    return []


async def _process_stream_events(
    stream: AsyncIterator[ConverseStreamOutputTypeDef],
    forced_tool: str | None,
    resp_stream_map_tool_use: Callable[[str, str], ContentBlock | None] | None = None,
    resp_stream_map_tool_result: Callable[[str, str, list[Any]], ContentBlock | None]
    | None = None,
) -> AsyncGenerator[JSONServerSentEvent]:
    """Process Bedrock stream events and yield Anthropic SSE events.

    **Why blocks are reindexed**

    The Anthropic SDK accumulates streamed blocks into a list and uses
    ``content[event.index]`` as a direct list position — index 0 maps to
    ``content[0]``, index 1 to ``content[1]``, etc.  Bedrock assigns its own
    block indices that may not be sequential after suppression, so this
    function emits all blocks with sequential Anthropic indices (0, 1, 2, …),
    consuming a new index only when a block is actually emitted.

    **Why some blocks are skipped**

    Two situations cause a Bedrock block to be dropped entirely:

    1. *Empty text preamble* — some models (e.g. Nova) send an empty ``text``
       delta for block 0 with no preceding ``contentBlockStart``, and the block
       contains no further content.  A block whose first delta is ``{"text": ""}``
       and that receives no subsequent non-empty delta before ``contentBlockStop``
       is silently discarded.  Models like DeepSeek V3 and Gemma also send an
       empty initial delta but follow it with real content in the same block; those
       blocks are correctly surfaced once the first non-empty delta arrives.
    2. *forced_tool filter* — when a single tool is forced, ``toolUse`` blocks
       for other tools are suppressed.

    **Why toolResult blocks are buffered**

    Bedrock sends model-internal tool results (e.g. ``nova_code_interpreter``)
    as a ``toolResult`` block in the same turn as the ``toolUse`` block.  The
    gateway must translate these to a ``CodeExecutionToolResultBlock`` (or
    similar) and emit them inline.  The block is buffered until
    ``contentBlockStop`` so the full JSON payload is available before
    translation, and the Anthropic index is assigned only at emit time —
    avoiding gaps if the callback returns ``None``.

    **Concrete example: Nova code_interpreter**

    Bedrock raw stream for a ``nova_code_interpreter`` invocation::

        [0] contentBlockDelta  delta={"text": ""}           ← empty preamble, skip
        [1] contentBlockStart  start={"toolUse": {...}}
        [1] contentBlockDelta  delta={"toolUse": {"input": "..."}}
        [1] contentBlockStop
        [2] contentBlockStart  start={"toolResult": {...}}  ← buffer until stop
        [2] contentBlockDelta  delta={"toolResult": [{...}]}
        [2] contentBlockStop
        [3] contentBlockDelta  delta={"text": "Result: 27"} ← no contentBlockStart
        [3] contentBlockStop

    Anthropic SSE output (sequential indices 0, 1, 2)::

        content_block_start  index=0  ServerToolUseBlock(name="code_execution")
        content_block_delta  index=0  InputJSONDelta(...)
        content_block_stop   index=0
        content_block_start  index=1  CodeExecutionToolResultBlock(stdout="27")
        content_block_stop   index=1
        content_block_start  index=2  TextBlock(text="")         ← synthesized start
        content_block_delta  index=2  TextDelta("Result: 27")
        content_block_stop   index=2

    Args:
        stream: Async iterator of Bedrock stream events.
        forced_tool: When set, tool_use blocks for other tools are suppressed.
        resp_stream_map_tool_use: Optional model-specific callback for ``toolUse``
            start blocks.  Receives the raw ``toolUseId`` and Bedrock tool name;
            return ``None`` to use the default mapping.
        resp_stream_map_tool_result: Optional model-specific callback for
            ``toolResult`` blocks.  Receives the raw ``toolUseId``, the Bedrock
            result type string, and the accumulated content items list; return
            ``None`` to discard the result block silently.

    Yields:
        Anthropic SSE events, finishing with a ``message_delta`` event.
    """
    stop_reason: StopReason | None = None
    usage_data: dict[str, int] = {}
    state = _StreamState()

    async for event in stream:
        match event:
            case {"contentBlockStart": start_block}:
                for sse in _process_content_block_start(
                    start_block, state, forced_tool, resp_stream_map_tool_use
                ):
                    yield sse
            case {"contentBlockDelta": delta_block}:
                for sse in _process_content_block_delta(delta_block, state):
                    yield sse
            case {"contentBlockStop": stop_block}:
                for sse in _process_content_block_stop(
                    stop_block, state, resp_stream_map_tool_result
                ):
                    yield sse
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
    resp_stream_map_tool_use: Callable[[str, str], ContentBlock | None] | None = None,
    resp_stream_map_tool_result: Callable[[str, str, list[Any]], ContentBlock | None]
    | None = None,
) -> AsyncGenerator[JSONServerSentEvent]:
    """Convert a Bedrock Converse stream into Anthropic SSE events.

    Args:
        message_id: Unique message identifier.
        model_id: Model identifier to echo back in the response.
        stream: Async iterator of Bedrock stream events.
        forced_tool: When set, tool_use blocks for other tools are suppressed.
        resp_stream_map_tool_use: Optional model-specific callback for ``toolUse``
            start blocks.  Passed through to ``_process_stream_events``.
        resp_stream_map_tool_result: Optional model-specific callback for
            ``toolResult`` blocks.  Passed through to ``_process_stream_events``.

    Yields:
        JSON server-sent events in Anthropic streaming format.
    """
    yield _make_message_start_event(message_id, model_id)
    async for event in _process_stream_events(
        stream,
        forced_tool,
        resp_stream_map_tool_use=resp_stream_map_tool_use,
        resp_stream_map_tool_result=resp_stream_map_tool_result,
    ):
        yield event
    yield JSONServerSentEvent(
        data=RawMessageStopEvent(type="message_stop").model_dump(
            mode="json", exclude_none=True
        ),
        event="message_stop",
    )


async def count_tokens_via_bedrock(
    request: MessageCountTokensParams,
    model_id: str,
    region: RegionName,
    *,
    system_message_as_messages: bool = False,
) -> int:
    """Count tokens using the AWS Bedrock Runtime CountTokens API.

    Builds a Converse-compatible input from the Anthropic request and calls
    the Bedrock ``count_tokens`` API for an accurate, model-specific count.

    Args:
        request: The count tokens request containing messages, system prompt, and tools.
        model_id: The Bedrock model identifier.
        region: The AWS region of the model.
        system_message_as_messages: When True, system-role messages are forwarded
            directly to Bedrock as messages instead of being extracted into the
            system-blocks field.

    Returns:
        The total number of input tokens.
    """
    messages, combined_system = _prepare_messages_and_system(
        request.messages,
        request.system,
        system_message_as_messages=system_message_as_messages,
    )
    req: ConverseTokensRequestTypeDef = {"messages": await _map_messages(messages)}
    if system_blocks := _map_system_blocks(combined_system):
        req["system"] = system_blocks
    tool_config = _build_tool_config(request.tools, request.tool_choice)
    if tool_config:
        req["toolConfig"] = tool_config

    with handle_bedrock_client_error():
        resp: CountTokensResponseTypeDef = await get_client(
            "bedrock-runtime", region
        ).count_tokens(modelId=model_id, input={"converse": req})
    return resp["inputTokens"]
