"""OpenAI Chat Completions API adapter for Bedrock Converse.

Translates between OpenAI Chat Completions API request/response types and
Bedrock Converse API-native types. Handles tool mapping, stop reason mapping,
message mapping, and response formatting (both streaming and non-streaming).
"""

from asyncio import Task, create_task
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from pydantic_core import to_json
from sse_starlette import JSONServerSentEvent, ServerSentEvent

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import build_system_blocks, set_inference_configuration
from stdapi.models.audio import synthesize_speech
from stdapi.models.chat._adapters import _common, _openai_common
from stdapi.monitoring import log_response_params
from stdapi.types.openai import (
    FunctionDefinition,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
    ResponseFormatText,
)
from stdapi.types.openai_chat_completions import (
    Annotation,
    AnnotationURLCitation,
    ChatCompletion,
    ChatCompletionAudio,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolUnionParam,
    Choice,
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    ChunkChoice,
    CompletionUsage,
    File,
    FunctionCall,
    PromptTokensDetails,
)
from stdapi.utils import b64encode, try_parse_json

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterable

    from pydantic import JsonValue
    from types_aiobotocore_bedrock_runtime.literals import (
        CacheTTLType,
        ConversationRoleType,
        ServiceTierTypeType,
        StopReasonType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockDeltaEventTypeDef,
        ContentBlockOutputTypeDef,
        ContentBlockTypeDef,
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
        InferenceConfigurationTypeDef,
        JsonSchemaDefinitionTypeDef,
        MessageTypeDef,
        SystemContentBlockTypeDef,
        ToolChoiceTypeDef,
        ToolConfigurationTypeDef,
        ToolResultContentBlockUnionTypeDef,
        ToolTypeDef,
        ToolUseBlockTypeDef,
    )

    from stdapi.models.chat import ReasoningParams
    from stdapi.types import JsonMapping
    from stdapi.types.openai_chat_completions import (
        ChatCompletionAssistantMessageParam,
        ChatCompletionAudioParam,
        ChatCompletionFunctionMessageParam,
        ChatCompletionMessageParam,
        ChatCompletionNamedToolChoiceCustomParam,
        ChatCompletionToolChoiceOptionParam,
        ChatCompletionToolMessageParam,
        CompletionCreateParams,
        FinishReason,
        FunctionCallParam,
        OutputModalities,
        ServiceTiers,
    )

#: Bedrock stop reasons to OpenAI finish reasons mapping
_FINISH_REASONS: dict[StopReasonType | str | None, FinishReason] = {
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
    "malformed_model_output": "content_filter",
    "malformed_tool_use": "content_filter",
    "tool_use": "tool_calls",
    # Non-standard but observed
    "incomplete": "length",
}

#: Empty tool schema for Bedrock tool configuration
_EMPTY_TOOL: dict[str, str] = {"type": "object"}


#: System message role names recognized by Bedrock
_SYSTEM_ROLES: frozenset[str] = frozenset({"system", "developer"})

#: Context variable tracking legacy function call format usage
_LEGACY_FUNCTION: ContextVar[bool] = ContextVar("legacy_function")

#: Default output modalities when none specified
DEFAULT_OUTPUT_MODALITIES: list[str] = ["text"]


def map_bedrock_stop_reason(
    stop_reason: StopReasonType | str | None, *, legacy_function: bool
) -> FinishReason:
    """Translate Bedrock stop reasons to OpenAI finish reasons.

    Args:
        stop_reason: Bedrock stop reason value (or None).
        legacy_function: Whether to use legacy function_call format.

    Returns:
        OpenAI stop reason value.
    """
    reason = _FINISH_REASONS.get(stop_reason, "stop")
    if legacy_function and reason == "tool_calls":
        return "function_call"
    return reason


def _map_tools(request: CompletionCreateParams) -> list[ChatCompletionToolUnionParam]:
    """Maps the tools and functions from the given request into a unified list.

    Args:
        request: The request object containing the tools and/or functions.

    Returns:
        A list of tools derived from the request.
    """
    tools: list[ChatCompletionToolUnionParam] = (
        list(request.tools) if request.tools is not None else []
    )
    if not tools and request.functions is not None:
        tools.extend(
            ChatCompletionFunctionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=function_spec.name,
                    description=function_spec.description,
                    parameters=function_spec.parameters,
                ),
            )
            for function_spec in request.functions
        )
    return tools


def _map_tool_spec(
    tool: ChatCompletionToolUnionParam, tools: list[ToolTypeDef]
) -> None:
    """Map a tool spec to a Bedrock ``toolSpec`` entry.

    Args:
        tool: The tool to be processed and mapped.
        tools: Accumulator for Bedrock tool specifications.
    """
    if tool.type == "function":
        tools.append(
            {
                "toolSpec": {
                    "name": tool.function.name,
                    "description": tool.function.description or "function",
                    "inputSchema": {"json": tool.function.parameters or _EMPTY_TOOL},
                }
            }
        )
    else:
        msg = f"Unsupported tool type '{tool.type}': {to_json(tool).decode()}"
        raise ApiError(msg)


def _map_tool_choice_literal(value: str) -> ToolChoiceTypeDef:
    """Map OpenAI tool_choice literal to Bedrock ToolChoiceTypeDef.

    Args:
        value: One of 'auto', 'required', 'none'.

    Returns:
        Bedrock toolChoice equivalent.
    """
    match value:
        case "auto":
            return {"auto": {}}
        case "required":
            return {"any": {}}
        case _:  # pragma: no cover
            msg = f"Unsupported tool choice literal: {value}"
            raise ApiError(msg)


def _map_tool_choice(
    tool_choice: ChatCompletionToolChoiceOptionParam | None,
) -> ToolChoiceTypeDef | None:
    """Convert OpenAI tool_choice union to a Bedrock ToolChoiceTypeDef.

    Args:
        tool_choice: None, a literal ('auto'|'required'|'none'), or a named tool choice.

    Returns:
        The Bedrock-specific toolChoice representation, or None.
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return _map_tool_choice_literal(tool_choice)
    tool_type = tool_choice.type
    if tool_type == "function":
        function_choice: ChatCompletionNamedToolChoiceParam = tool_choice  # type: ignore[assignment]
        return {"tool": {"name": function_choice.function.name}}
    if tool_type == "custom":
        custom_choice: ChatCompletionNamedToolChoiceCustomParam = tool_choice  # type: ignore[assignment]
        return {"tool": {"name": custom_choice.custom.name}}
    msg = f"Unsupported tool choice type '{tool_type}': {to_json(tool_choice).decode()}"
    raise ApiError(msg)


def _map_function_call(
    function_call: FunctionCallParam | None,
) -> ToolChoiceTypeDef | None:
    """Map legacy function_call to Bedrock ToolChoiceTypeDef.

    Args:
        function_call: Legacy function_call value (literal or dict with name).

    Returns:
        The corresponding Bedrock toolChoice representation, or None.
    """
    if function_call is None:
        return None
    if isinstance(function_call, str):
        return _map_tool_choice_literal(function_call)
    return {"tool": {"name": function_call.name}}


def build_tool_config(
    request: CompletionCreateParams,
) -> ToolConfigurationTypeDef | None:
    """Build a Bedrock tool configuration from an OpenAI request.

    All function tools are mapped to ``toolSpec`` entries as-is.  System tool
    routing (``SUPPORTED_SYSTEM_TOOLS`` auto-promotion) is handled at the model
    layer by ``_req_promote_system_tools``.

    When ``tool_choice`` (or legacy ``function_call``) is ``'none'``, no tool
    config is returned so the model behaves as if no tools were passed.  If the
    message history still requires a ``toolConfig`` (it contains ``toolUse``/
    ``toolResult`` blocks), the model layer synthesizes a permissive one.

    Args:
        request: The request object containing the data to map and configure tools.

    Returns:
        Bedrock tool configuration, or ``None`` if no tools are present or tool
        calling is disabled via ``'none'``.
    """
    if request.tool_choice == "none" or (
        "function_call" in request.model_fields_set and request.function_call == "none"
    ):
        return None

    tools: list[ToolTypeDef] = []
    for tool in _map_tools(request):
        _map_tool_spec(tool, tools)
    if not tools:
        return None

    tool_config: ToolConfigurationTypeDef = {"tools": tools}
    tool_choice_bedrock = _map_tool_choice(request.tool_choice) or _map_function_call(
        request.function_call
    )
    if tool_choice_bedrock:
        tool_config["toolChoice"] = tool_choice_bedrock
    return tool_config


def build_output_config(
    response_format: ResponseFormatText
    | ResponseFormatJSONObject
    | ResponseFormatJSONSchema
    | None,
) -> JsonSchemaDefinitionTypeDef | None:
    """Convert an OpenAI ``response_format`` to a Bedrock outputConfig schema.

    Args:
        response_format: OpenAI response format specification.

    Returns:
        Bedrock JSON schema definition dict, or ``None`` for plain text format.
    """
    match response_format:
        case None | ResponseFormatText():
            return None
        case ResponseFormatJSONObject():
            return {"schema": "{}"}
        case ResponseFormatJSONSchema(json_schema=js):
            schema = js.schema_
            json_schema: JsonSchemaDefinitionTypeDef = {
                "schema": schema
                if isinstance(schema, str)
                else to_json(schema).decode(),
                "name": js.name,
            }
            if js.description is not None:
                json_schema["description"] = js.description
            return json_schema


def translate_request(
    request: CompletionCreateParams, model_id: str
) -> tuple[
    InferenceConfigurationTypeDef,
    JsonMapping,
    ToolConfigurationTypeDef | None,
    ServiceTierTypeType | None,
    ServiceTiers | None,
    int,
    JsonSchemaDefinitionTypeDef | None,
    dict[str, str] | None,
]:
    """Translate OpenAI-specific request parameters into Bedrock Converse inputs.

    Handles inference configuration, tool configuration, service tier mapping,
    prompt cache settings, and response format (structured output).  Message
    mapping is handled separately by the ChatModel since it requires async
    image/audio/file processing.

    Args:
        request: OpenAI chat completion creation request.
        model_id: The Bedrock model identifier.

    Returns:
        Tuple of (inference_cfg, additional_request_fields, tool_config,
        bedrock_service_tier, openai_service_tier, choices_count, output_config,
        request_metadata).
    """
    max_tokens = request.max_completion_tokens or request.max_tokens
    additional_request_fields: JsonMapping = {}
    inference_cfg = set_inference_configuration(
        model_id,
        additional_request_fields,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=max_tokens,
        stop_sequences=request.stop,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        logit_bias=request.logit_bias,  # type: ignore[arg-type]
        seed=request.seed,
        top_logprobs=request.top_logprobs,
        top_k=request.top_k,
        **request.model_extra,
    )

    bedrock_service_tier, openai_service_tier = _openai_common.map_service_tier(
        request.service_tier
    )
    tool_config = build_tool_config(request)

    # Legacy format is declared by `functions`, or detected from the message history
    # by `map_messages` (which runs first) when no `tools` are declared.
    _LEGACY_FUNCTION.set(
        request.functions is not None
        or (request.tools is None and _LEGACY_FUNCTION.get(False))
    )
    return (
        inference_cfg,
        additional_request_fields,
        tool_config,
        bedrock_service_tier,
        openai_service_tier,
        request.n or 1,
        build_output_config(request.response_format),
        request.metadata or None,
    )


def extract_reasoning(request: CompletionCreateParams) -> ReasoningParams | None:
    """Extract reasoning parameters from an OpenAI Chat Completions request.

    Args:
        request: OpenAI chat completion creation request.

    Returns:
        Reasoning parameters to configure, or None if the request has no
        reasoning-related field set.
    """
    if (
        request.reasoning_effort is None
        and request.enable_thinking is None
        and request.thinking is None
    ):
        return None
    return {
        "enabled": (
            (
                request.reasoning_effort is not None
                and request.reasoning_effort != "none"
            )
            or request.enable_thinking is True
            or (request.thinking is not None and request.thinking.type == "enabled")
        ),
        "reasoning_effort": request.reasoning_effort,
        "budget_tokens": request.thinking_budget,
        "max_tokens": request.max_completion_tokens or request.max_tokens,
    }


def _extract_system_content_blocks(
    content: str | Iterable[ChatCompletionContentPartTextParam],
    cache_point: ContentBlockTypeDef | None = None,
) -> list[SystemContentBlockTypeDef]:
    """Extract Bedrock system content blocks from an OpenAI content field.

    Args:
        content: Message content which may be a plain string or a list of
            ChatCompletionContentPartParam entries.
        cache_point: Cache point block to insert after each part marked with
            ``prompt_cache_breakpoint``, or ``None`` to ignore breakpoints.

    Returns:
        A list of Bedrock SystemContentBlockTypeDef items (text blocks) in order.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return build_system_blocks(content)
    blocks: list[SystemContentBlockTypeDef] = []
    for part in content:
        blocks += build_system_blocks(part.text)
        # Empty parts yield no block: never lead with nor repeat a cache point
        if (
            cache_point is not None
            and part.prompt_cache_breakpoint
            and blocks
            and "cachePoint" not in blocks[-1]
        ):
            blocks.append(cache_point)
    return blocks


_SYNC_OPENAI_PARTS = (ChatCompletionContentPartTextParam,)


async def _convert_content_part(
    part: ChatCompletionContentPartParam | ChatCompletionContentPartRefusalParam,
) -> ContentBlockTypeDef:
    """Convert a content part into a Bedrock content block.

    Args:
        part: The content part to be converted.

    Returns:
        A Bedrock content block dict.

    Raises:
        ApiError: If the provided content part has an unsupported type.
    """
    match part:
        case ChatCompletionContentPartImageParam():
            return await part.image_url.url.to_bedrock_content_block()
        case ChatCompletionContentPartInputAudioParam():
            return await part.input_audio.data.to_bedrock_content_block(
                "audio", content_type=f"audio/{part.input_audio.format}"
            )
        case File():
            file_input = part.file.file_id or part.file.file_data
            if file_input is None:  # pragma: no cover
                msg = f"Missing file in {part}"
                raise ApiError(msg)
            return await file_input.to_bedrock_content_block(
                filename=part.file.filename
            )
        case _:  # pragma: no cover
            msg = f"Unsupported content part type: {getattr(part, 'type', type(part))}"
            raise ApiError(msg)


async def _extract_content_blocks(
    content: (
        str
        | Iterable[
            ChatCompletionContentPartParam | ChatCompletionContentPartRefusalParam
        ]
        | None
    ),
    cache_point: ContentBlockTypeDef | None = None,
) -> list[ContentBlockTypeDef]:
    """Extract Bedrock content blocks from OpenAI message content.

    Args:
        content: OpenAI message content (string, list of parts, or None).
        cache_point: Cache point block to insert after each part marked with
            ``prompt_cache_breakpoint``, or ``None`` to ignore breakpoints.

    Returns:
        List of Bedrock content blocks.
    """
    if isinstance(content, str):
        return [{"text": content}]
    blocks: list[ContentBlockTypeDef] = []
    for part in content or ():
        blocks.append(
            {"text": part.text}
            if isinstance(part, _SYNC_OPENAI_PARTS)
            else await _convert_content_part(part)
        )
        if cache_point is not None and part.prompt_cache_breakpoint:
            blocks.append(cache_point)
    return blocks


def _build_tool_use_block(
    name: str, arguments: str | JsonValue, call_id: str
) -> ContentBlockTypeDef:
    """Build a Bedrock toolUse content block from OpenAI function call data.

    Args:
        name: Function/tool name.
        arguments: Either a JSON string (function tools) or a JSON value (custom tools).
        call_id: Optional stable tool call id.

    Returns:
        A ContentBlockTypeDef representing a toolUse block.
    """
    tool_input = (
        try_parse_json(arguments) if isinstance(arguments, str) else arguments
    ) or {}
    if isinstance(tool_input, dict):
        tool_use: ToolUseBlockTypeDef = {
            "toolUseId": call_id,
            "name": name,
            "input": tool_input,
        }
        return {"toolUse": tool_use}
    msg = f'Invalid arguments for tool call "{name}" with ID "{call_id}": {arguments}'
    raise ApiError(msg)


def _map_assistant_content(
    content_blocks: list[ContentBlockTypeDef],
    message_param: ChatCompletionAssistantMessageParam,
    cache_point: ContentBlockTypeDef | None = None,
) -> None:
    """Append text/refusal content blocks from an assistant message.

    Args:
        content_blocks: Mutable list to append blocks to.
        message_param: The assistant message.
        cache_point: Cache point block to insert after each part marked with
            ``prompt_cache_breakpoint``, or ``None`` to ignore breakpoints.

    Raises:
        ApiError: If a content part has an unsupported type.
    """
    if (content := message_param.content) is None:
        return
    if isinstance(content, str):
        if content:
            content_blocks.append({"text": content})
        return
    for part in content:
        match part:
            case (
                ChatCompletionContentPartTextParam(text=text)
                | ChatCompletionContentPartRefusalParam(refusal=text)
            ):
                if text:
                    content_blocks.append({"text": text})
                    if cache_point is not None and part.prompt_cache_breakpoint:
                        content_blocks.append(cache_point)
            case _:  # pragma: no cover
                msg = f"Unsupported message type: {part}"
                raise ApiError(msg)


def _map_assistant_reasoning_content(
    content_blocks: list[ContentBlockTypeDef],
    message_param: ChatCompletionAssistantMessageParam,
) -> None:
    """Append a reasoning content block from an assistant message.

    Args:
        content_blocks: Mutable list to append blocks to.
        message_param: The assistant message.

    Raises:
        ApiError: If a reasoning content part has an unsupported type.
    """
    if (reasoning_content := message_param.reasoning_content) is None:
        return
    if isinstance(reasoning_content, str):
        text = reasoning_content
    else:
        parts: list[str] = []
        for part in reasoning_content:
            if isinstance(part, ChatCompletionContentPartTextParam):
                parts.append(part.text)
            else:  # pragma: no cover
                msg = f"Unsupported message type: {part}"
                raise ApiError(msg)
        text = "".join(parts)
    content_blocks.append({"reasoningContent": {"reasoningText": {"text": text}}})


def _extract_assistant_blocks(
    message_param: ChatCompletionAssistantMessageParam,
    cache_point: ContentBlockTypeDef | None = None,
) -> list[ContentBlockTypeDef]:
    """Append assistant tool use and content blocks.

    Appends Bedrock toolUse blocks derived from OpenAI assistant message
    `tool_calls` or legacy `function_call`, followed by any textual content
    (including refusal text when present).  An `audio` reference to a previous
    audio response has no replayable content and yields no block, so such a turn
    is dropped instead of producing an empty Bedrock message.

    Args:
        message_param: The assistant message to convert (may include tool calls).
        cache_point: Cache point block to insert after each content part marked
            with ``prompt_cache_breakpoint``, or ``None`` to ignore breakpoints.

    Returns:
        Content blocks.

    Raises:
        ApiError: If an unsupported tool call or content part type is encountered.
    """
    content_blocks: list[ContentBlockTypeDef] = []

    _map_assistant_content(content_blocks, message_param, cache_point)
    _map_assistant_reasoning_content(content_blocks, message_param)

    # Tools and function calls must be at the end
    for tool_call in message_param.tool_calls or []:
        if tool_call.type == "function":
            name, arguments = tool_call.function.name, tool_call.function.arguments
        elif tool_call.type == "custom":
            name, arguments = tool_call.custom.name, tool_call.custom.input
        else:  # pragma: no cover
            msg = f"Unsupported tool call type: {tool_call}"
            raise ApiError(msg)
        content_blocks.append(
            _build_tool_use_block(name=name, arguments=arguments, call_id=tool_call.id)
        )

    function_call = message_param.function_call
    if function_call is not None:
        content_blocks.append(
            _build_tool_use_block(
                name=function_call.name,
                arguments=function_call.arguments,
                call_id=function_call.name,
            )
        )
        _LEGACY_FUNCTION.set(True)

    return content_blocks


async def _extract_tool_blocks(
    message_param: ChatCompletionToolMessageParam,
) -> list[ContentBlockTypeDef]:
    """Convert a tool message to a Bedrock toolResult block.

    Args:
        message_param: Tool message containing content and tool call ID.

    Returns:
        Single-element list with a ``toolResult`` content block.
    """
    parts: list[
        ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam
    ] = (
        [ChatCompletionContentPartTextParam(text=message_param.content, type="text")]
        if isinstance(message_param.content, str)
        else list(message_param.content)
    )
    content: list[ToolResultContentBlockUnionTypeDef] = [
        await part.image_url.url.to_bedrock_content_block()  # type: ignore[misc]
        if isinstance(part, ChatCompletionContentPartImageParam)
        else _openai_common.parse_tool_content(part.text)
        for part in parts
    ]
    return [
        {"toolResult": {"toolUseId": message_param.tool_call_id, "content": content}}
    ]


def _extract_function_blocks(
    message_param: ChatCompletionFunctionMessageParam,
) -> list[ContentBlockTypeDef]:
    """Convert a legacy function message to a Bedrock toolResult block.

    Args:
        message_param: Legacy function message with name and content.

    Returns:
        Single-element list with a ``toolResult`` content block.
    """
    _LEGACY_FUNCTION.set(True)
    content: list[ToolResultContentBlockUnionTypeDef] = (
        [_openai_common.parse_tool_content(message_param.content)]
        if message_param.content is not None
        else []
    )
    return [{"toolResult": {"toolUseId": message_param.name, "content": content}}]


async def map_messages(
    messages: list[ChatCompletionMessageParam],
    *,
    allow_explicit_caching: bool = False,
    cache_ttl: CacheTTLType | None = None,
) -> tuple[list[MessageTypeDef], list[SystemContentBlockTypeDef]]:
    """Convert OpenAI message params into Bedrock messages and system blocks.

    Consecutive messages mapping to the same Bedrock role are merged into a
    single message.

    Args:
        messages: OpenAI message params to convert.
        allow_explicit_caching: Whether ``prompt_cache_breakpoint`` marks on
            content parts emit a Bedrock cache point.  Breakpoints on tool and
            function result messages are always ignored.
        cache_ttl: Cache TTL of the emitted cache points, ``None`` for the default.

    Returns:
        Tuple of (bedrock_messages, system_blocks).
    """
    bedrock_messages: list[MessageTypeDef] = []
    system_blocks: list[SystemContentBlockTypeDef] = []
    cache_point = (
        _openai_common.build_cache_point(cache_ttl) if allow_explicit_caching else None
    )

    for message_param in messages:
        role_name = message_param.role
        role: ConversationRoleType = "assistant" if role_name == "assistant" else "user"

        if role_name in _SYSTEM_ROLES:
            system_input: str | list[ChatCompletionContentPartTextParam]
            content_value = message_param.content
            if isinstance(content_value, str):
                system_input = content_value
            else:
                # Only text parts are allowed/used for system messages
                system_input = [
                    p
                    for p in (content_value or [])
                    if isinstance(p, ChatCompletionContentPartTextParam)
                ]
            system_blocks += _extract_system_content_blocks(system_input, cache_point)
            continue

        if role_name == "tool":
            tool_msg: ChatCompletionToolMessageParam = message_param  # type: ignore[assignment]
            content_blocks = await _extract_tool_blocks(tool_msg)
        elif role_name == "function":
            function_msg: ChatCompletionFunctionMessageParam = message_param  # type: ignore[assignment]
            content_blocks = _extract_function_blocks(function_msg)
        elif role_name == "assistant":
            assistant_msg: ChatCompletionAssistantMessageParam = message_param  # type: ignore[assignment]
            content_blocks = _extract_assistant_blocks(assistant_msg, cache_point)
        else:
            content_blocks = await _extract_content_blocks(
                message_param.content, cache_point
            )

        _common.append_or_merge(bedrock_messages, role, content_blocks)

    return bedrock_messages, system_blocks


def extract_output_text(
    contents: list[ContentBlockOutputTypeDef],
) -> tuple[str | None, str | None]:
    """Extracts output text and reasoning text from content blocks.

    Blocks are concatenated without separator so the result matches the
    concatenation of the streamed deltas.

    Args:
        contents: A list of Bedrock content blocks.

    Returns:
        Tuple of (content_text, reasoning_text), each None if not found.
    """
    content_text: list[str] = []
    reasoning_text: list[str] = []
    for block in contents:
        if "text" in block:
            content_text.append(block["text"])
        if (rc := block.get("reasoningContent")) and (rt := rc.get("reasoningText")):
            reasoning_text.append(rt["text"])
    return (
        "".join(content_text) if content_text else None,
        "".join(reasoning_text) if reasoning_text else None,
    )


def extract_citations(
    contents: list[ContentBlockOutputTypeDef],
) -> list[Annotation] | None:
    """Extracts citation annotations from content blocks.

    Args:
        contents: A list of Bedrock content blocks.

    Returns:
        A list of annotations if citations are found, or None.
    """
    annotations: list[Annotation] = []
    for block in contents:
        if "citationsContent" not in block:
            continue
        citations = block["citationsContent"].get("citations", ())

        for citation in citations:
            location = citation.get("location", {})
            if not (web_location := location.get("web")):
                continue
            if not (url := web_location.get("url")):
                continue
            annotations.append(
                Annotation(
                    type="url_citation",
                    url_citation=AnnotationURLCitation(
                        url=url,
                        title=citation.get("title") or web_location.get("domain", ""),
                        start_index=0,
                        end_index=0,
                    ),
                )
            )

    return annotations or None


def extract_tool_calls(
    contents: list[ContentBlockOutputTypeDef],
    *,
    legacy_function: bool,
    suppress_tool_names: frozenset[str] | None = None,
) -> tuple[list[ChatCompletionMessageToolCallUnion] | None, FunctionCall | None]:
    """Extracts tool calls and function calls from conversation response content.

    Args:
        contents: A list of content blocks containing response data.
        legacy_function: Whether to return a single legacy-style function call.
        suppress_tool_names: Optional set of Bedrock tool names to exclude from
            the returned tool_calls (e.g. system tools handled server-side).

    Returns:
        Tuple of (tool_calls, function_call).
    """
    tool_calls: list[ChatCompletionMessageToolCallUnion] = []
    for content in contents:
        if "toolUse" not in content:
            continue
        tool_use = content["toolUse"]
        if suppress_tool_names and tool_use["name"] in suppress_tool_names:
            continue
        function = FunctionCall(
            name=tool_use["name"], arguments=to_json(tool_use["input"]).decode()
        )
        if legacy_function:
            return None, function
        tool_calls.append(
            ChatCompletionMessageFunctionToolCall(
                type="function", id=tool_use["toolUseId"], function=function
            )
        )

    return tool_calls or None, None


async def _get_or_generate_audio(
    audio_params: ChatCompletionAudioParam,
    contents: list[ContentBlockOutputTypeDef],
    completion_id: str,
    content: str,
    created: int,
    index: int,
) -> ChatCompletionAudio:
    """Return the audio output for a completion choice, generating it via TTS if absent.

    If *contents* already contains an ``audio`` block (model-native audio), that
    data is used directly.  Otherwise, the text is synthesised via TTS.

    Args:
        audio_params: Audio format and voice configuration.
        contents: Bedrock output content blocks for this choice.
        completion_id: Unique completion identifier.
        content: Text to synthesise if no model audio is present.
        created: Unix timestamp of the request.
        index: Zero-based choice index.

    Returns:
        ChatCompletionAudio with the audio data.
    """
    for block in contents:
        if (audio := block.get("audio")) and (source := audio.get("source")):
            audio_content = source["bytes"]
            break
    else:
        audio_content = b"".join(
            [
                chunk
                async for chunk in await synthesize_speech(
                    text=content,
                    voice=audio_params.voice,
                    resp_format="pcm"
                    if audio_params.format == "pcm16"
                    else audio_params.format,
                )
            ]
        )
    return ChatCompletionAudio(
        id=f"audio-{completion_id}-{index}",
        data=await b64encode(audio_content),
        expires_at=created,
        transcript=content,
    )


async def format_response(
    completion_id: str,
    created: int,
    model_id: str,
    responses: list[ConverseResponseTypeDef],
    service_tier: ServiceTiers | None,
    audio_params: ChatCompletionAudioParam | None,
    modalities: list[OutputModalities],
    suppress_tool_names: frozenset[str] | None = None,
) -> ChatCompletion:
    """Format Bedrock Converse responses as an OpenAI ChatCompletion.

    Args:
        completion_id: Unique identifier for the completion request.
        created: Timestamp indicating when the request was created.
        model_id: The model identifier.
        responses: Pre-executed Converse API responses, one per choice.
        service_tier: Optional tier of service for the request.
        audio_params: Optional parameters for audio generation.
        modalities: List of output modalities such as text or audio.
        suppress_tool_names: Optional set of Bedrock tool names to exclude
            from the returned tool_calls (e.g. system tools handled server-side).

    Returns:
        A structured ChatCompletion response.
    """
    choices: list[Choice] = []
    usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    tts_tasks: dict[int, Task[Any]] = {}
    cached_tokens = 0
    cache_write_tokens = 0
    legacy_function = _LEGACY_FUNCTION.get()
    for index, response in enumerate(responses):
        # OpenAI semantics: prompt_tokens covers the full prompt, cache buckets included.
        cache_read = response["usage"].get("cacheReadInputTokens", 0)
        cache_write = response["usage"].get("cacheWriteInputTokens", 0)
        usage.prompt_tokens += (
            response["usage"]["inputTokens"] + cache_read + cache_write
        )
        usage.completion_tokens += response["usage"]["outputTokens"]
        cached_tokens += cache_read
        cache_write_tokens += cache_write
        message = response["output"]["message"]["content"]
        tool_calls, function_call = extract_tool_calls(
            message,
            legacy_function=legacy_function,
            suppress_tool_names=suppress_tool_names,
        )
        content, reasoning_content = extract_output_text(message)
        annotations = extract_citations(message)
        if audio_params and content:
            tts_tasks[index] = create_task(
                _get_or_generate_audio(
                    audio_params, message, completion_id, content, created, index
                )
            )
        choices.append(
            Choice(
                finish_reason=map_bedrock_stop_reason(
                    response["stopReason"], legacy_function=legacy_function
                ),
                index=index,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content if "text" in modalities else None,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                    function_call=function_call,
                    annotations=annotations,
                ),
            )
        )
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
    if cached_tokens or cache_write_tokens:
        usage.prompt_tokens_details = PromptTokensDetails(
            cached_tokens=cached_tokens, cache_write_tokens=cache_write_tokens or None
        )

    for index, tts_task in tts_tasks.items():
        choices[index].message.audio = await tts_task

    return log_response_params(
        ChatCompletion(
            id=completion_id,
            choices=choices,
            created=created,
            model=model_id,
            object="chat.completion",
            usage=usage,
            service_tier=service_tier,
        )
    )


def _stream_get_content_block_delta(
    choice_delta: ChoiceDelta,
    delta_block: ContentBlockDeltaEventTypeDef,
    tool_call_indices: dict[int, int],
    *,
    legacy_function: bool,
) -> None:
    """Process a content block delta and update the choice delta.

    Args:
        choice_delta: The choice delta object to update.
        delta_block: The delta event containing changes.
        tool_call_indices: Mutable mapping of content block index to tool call
            position.
        legacy_function: Whether to use legacy function handling.
    """
    delta = delta_block["delta"]
    if "text" in delta:
        choice_delta.content = delta["text"]
    if (rc := delta.get("reasoningContent")) and "text" in rc:
        choice_delta.reasoning_content = rc["text"]
    if "toolUse" not in delta:
        return
    delta_tool_use = delta["toolUse"]
    function = ChoiceDeltaFunctionCall(arguments=delta_tool_use["input"])
    if legacy_function:
        choice_delta.function_call = function
    else:
        choice_delta.tool_calls = [
            ChoiceDeltaToolCall(
                index=tool_call_indices.setdefault(
                    delta_block["contentBlockIndex"], len(tool_call_indices)
                ),
                type="function",
                function=function,
            )
        ]


def _stream_delta_chunk(
    completion_id: str,
    created: int,
    model_id: str,
    event: ConverseStreamOutputTypeDef,
    service_tier: ServiceTiers | None,
    tool_call_indices: dict[int, int],
    *,
    legacy_function: bool,
    chunk: ChatCompletionChunk | None = None,
) -> tuple[ChatCompletionChunk | None, bool]:
    """Process a streaming event into a ChatCompletionChunk.

    Args:
        completion_id: The unique identifier for the chat completion.
        created: The timestamp when the chunk was created.
        model_id: The model identifier.
        event: The event data containing updates.
        service_tier: The service tier information.
        tool_call_indices: Mutable mapping of content block index to tool call
            position.
        legacy_function: Whether to use legacy function handling.
        chunk: The current chunk to update. Defaults to None.

    Returns:
        Tuple of (updated chunk, whether stream has ended).
    """
    if chunk:
        choice = chunk.choices[0]
        choice_delta = choice.delta
    else:
        choice_delta = ChoiceDelta()
        choice = ChunkChoice(index=0, delta=choice_delta)
        chunk = ChatCompletionChunk(
            id=completion_id,
            choices=[choice],
            created=created,
            model=model_id,
            object="chat.completion.chunk",
            service_tier=service_tier,
        )
    end = False

    match event:
        case {"contentBlockStart": start_block}:
            start = start_block["start"]
            if "toolUse" not in start:
                return None, end
            start_tool_use = start["toolUse"]
            tool_id = start_tool_use["toolUseId"]
            function = ChoiceDeltaFunctionCall(name=start_tool_use["name"])
            if legacy_function:
                choice_delta.function_call = function
            else:
                choice_delta.tool_calls = [
                    ChoiceDeltaToolCall(
                        index=tool_call_indices.setdefault(
                            start_block["contentBlockIndex"], len(tool_call_indices)
                        ),
                        id=tool_id,
                        type="function",
                        function=function,
                    )
                ]

        case {"contentBlockDelta": delta_block}:
            _stream_get_content_block_delta(
                choice_delta,
                delta_block,
                tool_call_indices,
                legacy_function=legacy_function,
            )

        case {"messageStop": stop_block}:
            choice.finish_reason = map_bedrock_stop_reason(
                stop_block["stopReason"], legacy_function=legacy_function
            )
            end = True

    return chunk, end


def _suppress_system_tool_event(
    event: ConverseStreamOutputTypeDef,
    suppress_tool_names: frozenset[str],
    suppressed_indices: set[int],
) -> bool:
    """Return True and update *suppressed_indices* when *event* belongs to a suppressed tool.

    Args:
        event: A Bedrock stream event dict.
        suppress_tool_names: Bedrock tool names whose stream events should be dropped.
        suppressed_indices: Mutable set of content block indices currently suppressed;
            updated in place.

    Returns:
        ``True`` when the event should be silently skipped by the caller.
    """
    if block_start := event.get("contentBlockStart"):
        start = block_start["start"]
        if (tool_use := start.get("toolUse")) and tool_use.get(
            "name"
        ) in suppress_tool_names:
            suppressed_indices.add(block_start["contentBlockIndex"])
            return True
    elif block_delta := event.get("contentBlockDelta"):
        if block_delta["contentBlockIndex"] in suppressed_indices:
            return True
    elif block_stop := event.get("contentBlockStop"):
        if (idx := block_stop["contentBlockIndex"]) in suppressed_indices:
            suppressed_indices.discard(idx)
            return True
    return False


async def format_stream(
    completion_id: str,
    created: int,
    model_id: str,
    stream: AsyncIterator[ConverseStreamOutputTypeDef],
    service_tier: ServiceTiers | None,
    *,
    include_usage: bool = False,
    suppress_tool_names: frozenset[str] | None = None,
) -> AsyncGenerator[ServerSentEvent]:
    """Stream Bedrock Converse events as OpenAI ChatCompletionChunk SSE events.

    When ``include_usage`` is set, usage is reported in its own trailing chunk
    with empty ``choices`` (per OpenAI spec), separate from the finish-reason
    chunk. The stream always ends with a ``[DONE]`` sentinel.

    Args:
        completion_id: Unique identifier for the completion.
        created: Timestamp when the request was initiated.
        model_id: The model identifier.
        stream: Pre-opened Bedrock ConverseStream event iterator.
        service_tier: Service tier being used.
        include_usage: Whether to include usage information.
        suppress_tool_names: Optional set of Bedrock tool names whose
            contentBlockStart/contentBlockDelta/contentBlockStop events are
            silently dropped (e.g. system tools handled server-side).

    Yields:
        JSONServerSentEvent chunks, terminated by the ``[DONE]`` sentinel.
    """
    yield JSONServerSentEvent(
        data=log_response_params(
            ChatCompletionChunk(
                id=completion_id,
                choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant"))],
                created=created,
                model=model_id,
                object="chat.completion.chunk",
                service_tier=service_tier,
            ).model_dump(mode="json", exclude_none=True)
        )
    )

    legacy_function = _LEGACY_FUNCTION.get()
    end_state = False
    chunk: ChatCompletionChunk | None = None
    suppressed_indices: set[int] = set()
    # Bedrock content block index -> contiguous position in the OpenAI tool_calls array
    tool_call_indices: dict[int, int] = {}
    async for event in stream:
        if suppress_tool_names and _suppress_system_tool_event(
            event, suppress_tool_names, suppressed_indices
        ):
            continue
        if end_state:
            # Past the finish chunk: only a trailing usage-only chunk remains to emit.
            if include_usage and (usage := _openai_common.extract_stream_usage(event)):
                yield JSONServerSentEvent(
                    data=ChatCompletionChunk(
                        id=completion_id,
                        choices=[],
                        created=created,
                        model=model_id,
                        object="chat.completion.chunk",
                        service_tier=service_tier,
                        usage=usage,
                    ).model_dump(mode="json", exclude_none=True)
                )
            continue
        chunk, end = _stream_delta_chunk(
            completion_id,
            created,
            model_id,
            event,
            service_tier,
            tool_call_indices,
            legacy_function=legacy_function,
            chunk=chunk,
        )
        end_state |= end
        if chunk:
            yield JSONServerSentEvent(
                data=chunk.model_dump(mode="json", exclude_none=True)
            )
            chunk = None
    yield ServerSentEvent(data="[DONE]", event=None)
