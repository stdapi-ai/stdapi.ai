"""OpenAI Chat Completions API adapter for Bedrock Converse.

Translates between OpenAI Chat Completions API request/response types and
Bedrock Converse API-native types. Handles tool mapping, stop reason mapping,
message mapping, and response formatting (both streaming and non-streaming).
"""

from asyncio import Task, TaskGroup, create_task, gather
from contextvars import ContextVar
from os.path import splitext
from typing import TYPE_CHECKING, Any

from pydantic_core import from_json, to_json
from sse_starlette import JSONServerSentEvent

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import (
    MIME_TYPES_TO_AUDIO_TYPE,
    MIME_TYPES_TO_DOCUMENT_TYPE,
    MIME_TYPES_TO_VIDEO_TYPE,
    PROMPT_CACHING,
    build_system_blocks,
    handle_bedrock_client_error,
    image_block_from_bytes,
    image_block_from_url,
    set_inference_configuration,
)
from stdapi.models.audio import synthesize_speech
from stdapi.monitoring import log_response_params
from stdapi.tokenizer import estimate_token_count
from stdapi.types.anthropic_messages import (
    MemoryToolParam,
    ServerTools,
    ToolBashParam,
    ToolTextEditorParam,
)
from stdapi.types.openai import FunctionDefinition
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
    CompletionTokensDetails,
    CompletionUsage,
    File,
    FunctionCall,
    PromptCacheRetention,
    PromptTokensDetails,
)
from stdapi.utils import b64decode_data_or_uri_with_mime, b64encode, try_parse_json

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterable

    from pydantic import JsonValue
    from types_aiobotocore_bedrock_runtime import BedrockRuntimeClient
    from types_aiobotocore_bedrock_runtime.literals import (
        AudioFormatType,
        CacheTTLType,
        ConversationRoleType,
        ServiceTierTypeType,
        StopReasonType,
        VideoFormatType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        AudioBlockTypeDef,
        ContentBlockDeltaEventTypeDef,
        ContentBlockOutputTypeDef,
        ContentBlockTypeDef,
        ConverseStreamOutputTypeDef,
        DocumentBlockTypeDef,
        InferenceConfigurationTypeDef,
        MessageTypeDef,
        ReasoningContentBlockUnionTypeDef,
        SystemContentBlockTypeDef,
        SystemToolTypeDef,
        ToolChoiceTypeDef,
        ToolConfigurationTypeDef,
        ToolResultContentBlockUnionTypeDef,
        ToolSpecificationTypeDef,
        ToolTypeDef,
        ToolUseBlockTypeDef,
        VideoBlockTypeDef,
    )

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef, PromptCaching
    from stdapi.types import JsonMapping
    from stdapi.types.anthropic_messages import ToolUnionParam
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

    #: Union of Anthropic server tool parameter classes used in tool mappings
    type _AnthropicToolClass = type[
        ToolBashParam | ToolTextEditorParam | MemoryToolParam
    ]

#: Bedrock stop reasons to OpenAI finish reasons mapping
_FINISH_REASONS: dict[StopReasonType | None, FinishReason] = {
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
    "malformed_model_output": "content_filter",
    "malformed_tool_use": "content_filter",
    "tool_use": "tool_calls",
}

#: Empty tool schema for Bedrock tool configuration
_EMPTY_TOOL: dict[str, str] = {"type": "object"}

#: Prefix for system-provided tool identifiers
_SYSTEM_TOOL_PREFIX: str = "systemTool_"

#: OpenAI services tiers to Bedrock mapping
_SERVICES_TIERS: dict[ServiceTiers, ServiceTierTypeType] = {
    "priority": "priority",
    "flex": "flex",
    # Extra bedrock specific values
    "reserved": "reserved",
}


#: OpenAI to Bedrock prompt cache retention mapping
CACHE_TTL: dict[PromptCacheRetention | None, CacheTTLType | None] = {
    "in-memory": None,
    "24h": "1h",  # max current value
    "1h": "1h",
    "5m": "5m",
}

#: System message role names recognized by Bedrock
_SYSTEM_ROLES: frozenset[str] = frozenset({"system", "developer"})

#: Context variable tracking legacy function call format usage
_LEGACY_FUNCTION: ContextVar[bool] = ContextVar("legacy_function")

#: Default output modalities when none specified
DEFAULT_OUTPUT_MODALITIES: list[str] = ["text"]


#: Versioned Anthropic tool name lookup: type prefix → (ParamClass, canonical_name)
_ANTHROPIC_TOOL_PREFIXES: dict[str, tuple[_AnthropicToolClass, ServerTools]] = {
    "bash": (ToolBashParam, "bash"),
    "text_editor": (ToolTextEditorParam, "str_replace_editor"),
    "memory": (MemoryToolParam, "memory"),
}


def _map_anthropic_tool_name(stripped: str) -> ToolUnionParam | None:
    """Map a stripped ``systemTool_`` name to an Anthropic ``ToolUnionParam``.

    Accepts versioned names with an 8-digit date suffix (``bash_20250124``).
    Returns ``None`` for unknown names so the caller can fall back to the
    regular Bedrock ``systemTool`` path.
    """
    for prefix, (cls, default_name) in _ANTHROPIC_TOOL_PREFIXES.items():
        suffix = stripped.removeprefix(f"{prefix}_")
        if suffix != stripped and len(suffix) == 8 and suffix.isdigit():
            return cls(type=stripped, name=default_name)  # type: ignore[arg-type]

    return None


def map_bedrock_stop_reason(
    stop_reason: StopReasonType | None, *, legacy_function: bool
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


def map_service_tier(
    value: ServiceTiers | None,
) -> tuple[ServiceTierTypeType | None, ServiceTiers | None]:
    """Map OpenAI service tier to Bedrock service tier.

    Args:
        value: OpenAI service tier.

    Returns:
        Bedrock service tier, Effective OpenAI service tier
    """
    if value is None:
        return None, None
    if value in _SERVICES_TIERS:
        return _SERVICES_TIERS[value], value
    return None, "default"


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
    tool: ChatCompletionToolUnionParam,
    tools: list[ToolTypeDef],
    anthropic_system_tools: list[ToolUnionParam],
) -> None:
    """Map a tool spec to Bedrock tools or Anthropic server tools.

    Args:
        tool: The tool to be processed and mapped.
        tools: Accumulator for regular Bedrock tool specifications.
        anthropic_system_tools: Accumulator for Anthropic-native server tools.
    """
    tool_type = tool.type
    if tool_type == "function":
        function_tool: ChatCompletionFunctionToolParam = tool  # type: ignore[assignment]
        function_spec = function_tool.function
        name = function_spec.name
        if name.startswith(_SYSTEM_TOOL_PREFIX) and not function_spec.parameters:
            stripped: str = name.removeprefix(_SYSTEM_TOOL_PREFIX)
            if anthropic_tool := _map_anthropic_tool_name(stripped):
                anthropic_system_tools.append(anthropic_tool)
            else:
                system_tool: SystemToolTypeDef = {"name": stripped}
                tools.append({"systemTool": system_tool})
        else:
            tool_spec: ToolSpecificationTypeDef = {
                "name": function_spec.name,
                "description": function_spec.description or tool_type,
                "inputSchema": {"json": function_spec.parameters or _EMPTY_TOOL},
            }
            tools.append({"toolSpec": tool_spec})
    else:  # pragma: no cover
        msg = f"Unsupported tool type '{tool_type}': {to_json(tool).decode()}"
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
) -> tuple[ToolConfigurationTypeDef | None, list[ToolUnionParam]]:
    """Builds a configuration for tools based on the provided request.

    Separates Anthropic-native server tools (detected via the ``systemTool_``
    prefix and matching a known Anthropic tool name) from regular Bedrock
    tools.  Anthropic server tools are returned separately so the model
    class can route them through ``_req_configure_system_tools``.

    Args:
        request: The request object containing the data to map and configure tools.

    Returns:
        Tuple of (tool_config, anthropic_system_tools).
    """
    tools: list[ToolTypeDef] = []
    anthropic_system_tools: list[ToolUnionParam] = []
    for tool in _map_tools(request):
        _map_tool_spec(tool, tools, anthropic_system_tools)
    if not tools:
        return None, anthropic_system_tools

    tool_config: ToolConfigurationTypeDef = {"tools": tools}
    tool_choice_bedrock = _map_tool_choice(request.tool_choice) or _map_function_call(
        request.function_call
    )
    if tool_choice_bedrock:
        tool_config["toolChoice"] = tool_choice_bedrock
    return tool_config, anthropic_system_tools


def translate_request(
    request: CompletionCreateParams, model_id: str
) -> tuple[
    InferenceConfigurationTypeDef,
    JsonMapping,
    ToolConfigurationTypeDef | None,
    list[ToolUnionParam],
    ServiceTierTypeType | None,
    ServiceTiers | None,
    int,
]:
    """Translate OpenAI-specific request parameters into Bedrock Converse inputs.

    Handles inference configuration, tool configuration, service tier mapping,
    and prompt cache settings. Message mapping is handled separately by the
    ChatModel since it requires async image/audio/file processing.

    Args:
        request: OpenAI chat completion creation request.
        model_id: The Bedrock model identifier.

    Returns:
        Tuple of (inference_cfg, additional_request_fields, tool_config,
        system_tools, bedrock_service_tier, openai_service_tier, choices_count).
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

    bedrock_service_tier, openai_service_tier = map_service_tier(request.service_tier)
    tool_config, system_tools = build_tool_config(request)

    _LEGACY_FUNCTION.set(request.functions is not None)
    return (
        inference_cfg,
        additional_request_fields,
        tool_config,
        system_tools,
        bedrock_service_tier,
        openai_service_tier,
        request.n or 1,
    )


def _extract_system_content_blocks(
    content: str | Iterable[ChatCompletionContentPartTextParam],
) -> list[SystemContentBlockTypeDef]:
    """Extract Bedrock system content blocks from an OpenAI content field.

    Args:
        content: Message content which may be a plain string or a list of
            ChatCompletionContentPartParam entries.

    Returns:
        A list of Bedrock SystemContentBlockTypeDef items (text blocks) in order.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return build_system_blocks(content)
    return build_system_blocks(*(part.text for part in content))


async def _extract_image_content_block(
    image_part: ChatCompletionContentPartImageParam,
) -> ContentBlockTypeDef:
    """Convert an OpenAI image_url section to a Bedrock content block.

    Supports data URLs, s3:// URIs, and http(s) URLs (downloaded via aiohttp).

    Args:
        image_part: Image content part as provided by OpenAI Chat API.

    Returns:
        A Bedrock ContentBlockTypeDef for the referenced image.

    Raises:
        ApiError: If the URL is invalid or unsupported by this implementation.
    """
    return await image_block_from_url(image_part.image_url.url)


async def _extract_audio_content_block(
    audio_part: ChatCompletionContentPartInputAudioParam,
) -> ContentBlockTypeDef:
    """Convert an OpenAI input_audio section to a Bedrock content block.

    Args:
        audio_part: Audio content part as provided by OpenAI Chat API.

    Returns:
        A Bedrock ContentBlockTypeDef for the referenced audio.

    Raises:
        ApiError: If the URL is invalid or unsupported by this implementation.
    """
    try:
        data = (await b64decode_data_or_uri_with_mime(audio_part.input_audio.data))[0]
    except ValueError as error:
        raise ApiError(error.args[0]) from None
    audio_block_bytes: AudioBlockTypeDef = {
        "source": {"bytes": data},
        "format": audio_part.input_audio.format,
    }
    return {"audio": audio_block_bytes}


async def _extract_file_content_block(file_part: File) -> ContentBlockTypeDef:
    """Convert an OpenAI file section to a Bedrock content block.

    The OpenAI File part contains base64-encoded bytes (file_data). This helper
    detects the file's MIME type using python-magic and maps it to the proper
    Bedrock content block:
    - image/* → image block with inferred format and bytes
    - video/* → video block with inferred/normalized format and bytes
    - audio/* → audio block with inferred/normalized format and bytes
    - text/* or application/* → document block with inferred/normalized format and bytes

    Args:
        file_part: OpenAI chat content part with type "file".

    Returns:
        A Bedrock ContentBlockTypeDef containing an image, video, or document block
        depending on the detected MIME type.

    Raises:
        ApiError: When file_data is missing/invalid/empty or the detected
            MIME type is not supported by this implementation.
    """
    file_section = file_part.file
    b64_data = file_section.file_data
    try:
        data, mime = await b64decode_data_or_uri_with_mime(b64_data, validate=True)
    except ValueError as error:
        msg = f"Invalid {file_part}: {error.args[0]}"
        raise ApiError(msg) from None

    if mime.startswith("image/"):
        return image_block_from_bytes(data, mime)

    file_format = mime.split("/", 1)[1]
    if mime.startswith("video/"):
        video_format: VideoFormatType = MIME_TYPES_TO_VIDEO_TYPE.get(
            file_format,
            file_format,  # type: ignore[arg-type]
        )
        video_block_bytes: VideoBlockTypeDef = {
            "source": {"bytes": data},
            "format": video_format,
        }
        return {"video": video_block_bytes}

    if mime.startswith("audio/"):
        audio_format: AudioFormatType = MIME_TYPES_TO_AUDIO_TYPE.get(
            file_format,
            file_format,  # type: ignore[arg-type]
        )
        audio_block_bytes: AudioBlockTypeDef = {
            "source": {"bytes": data},
            "format": audio_format,
        }
        return {"audio": audio_block_bytes}

    if mime.startswith(("text/", "application/")):
        # Default to 'txt' when the MIME subtype is unknown
        document_format = MIME_TYPES_TO_DOCUMENT_TYPE.get(file_format, "txt")
        name_value = (
            # Remove file extension, "." is not supported
            splitext(file_section.filename)[0]  # noqa: PTH122
            if file_section.filename is not None
            else f"file-{document_format}"
        )
        document_block_bytes: DocumentBlockTypeDef = {
            "name": name_value,
            "source": {"bytes": data},
            "format": document_format,
        }
        return {"document": document_block_bytes}

    msg = f"Unsupported file MIME type for 'file' ({mime}): {file_part}"
    raise ApiError(msg)


_SYNC_OPENAI_PARTS = (ChatCompletionContentPartTextParam,)


async def _convert_content_part(
    part: ChatCompletionContentPartParam | ChatCompletionContentPartRefusalParam,
) -> ContentBlockTypeDef:
    """Converts a content part into a corresponding content block representation based on its specific type.

    The method processes different types of
    content parts and handles them accordingly. If the content part type
    is not supported, an error is raised.

    Args:
        part : The content part to be converted. The type determines how the content
            is processed into a content block.

    Returns:
        The converted content block representation of the
        provided content part.

    Raises:
        ApiError: If the provided content part has an unsupported type.
    """
    match part:
        case ChatCompletionContentPartImageParam():
            return await _extract_image_content_block(part)
        case ChatCompletionContentPartInputAudioParam():
            return await _extract_audio_content_block(part)
        case File():
            return await _extract_file_content_block(part)
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
) -> list[ContentBlockTypeDef]:
    """Extract Bedrock content blocks from OpenAI message content.

    Supports:
    - text parts
    - image_url parts with data URLs (base64), s3:// URIs, and http(s) downloads via aiohttp
    - file parts (image/video/document) with base64 body and MIME sniffing
    """
    if isinstance(content, str):
        return [{"text": content}]

    parts = tuple(content or ())
    if not parts:
        return []

    async with TaskGroup() as tg:
        tasks: tuple[Task[ContentBlockTypeDef] | None, ...] = tuple(
            None
            if isinstance(p, _SYNC_OPENAI_PARTS)
            else tg.create_task(_convert_content_part(p))
            for p in parts
        )

    return [
        {"text": p.text} if t is None else t.result()  # type: ignore[union-attr]
        for p, t in zip(parts, tasks, strict=False)
    ]


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


def _append_text_content_block(
    content: str, content_blocks: list[ContentBlockTypeDef]
) -> None:
    """Adds a new content block to the list if content is provided.

    Args:
        content: The content string to be evaluated and added as a
            content block if not empty.
        content_blocks: The list of content blocks to which the new content block will be appended.
    """
    if content:
        content_blocks.append({"text": content})


def _map_assistant_content(
    content_blocks: list[ContentBlockTypeDef],
    message_param: ChatCompletionAssistantMessageParam,
) -> None:
    """Maps the assistant message content into content block structures.

    Args:
        content_blocks: The list of content blocks to append the message content to.
        message_param: The assistant message.

    Raises:
        ApiError: If the content part in `message_param` contains an unsupported
            type.
    """
    content = message_param.content
    if content is not None:
        if isinstance(content, str):
            _append_text_content_block(content, content_blocks)
        else:
            for part in content:
                match part:
                    case ChatCompletionContentPartTextParam(text=text):
                        _append_text_content_block(text, content_blocks)
                    case ChatCompletionContentPartRefusalParam(refusal=refusal):
                        _append_text_content_block(refusal, content_blocks)
                    case _:  # pragma: no cover
                        msg = f"Unsupported message type: {part}"
                        raise ApiError(msg)


def _map_assistant_reasoning_content(
    content_blocks: list[ContentBlockTypeDef],
    message_param: ChatCompletionAssistantMessageParam,
) -> None:
    """Maps the reasoning message content into content block structures.

    Args:
        content_blocks: The list of content blocks to append the message content to.
        message_param: The assistant message.

    Raises:
        ApiError: If the content part in `message_param` contains an unsupported
            type.
    """
    reasoning_content = message_param.reasoning_content
    if reasoning_content is not None:
        reasoning_block: ReasoningContentBlockUnionTypeDef = {}
        if isinstance(reasoning_content, str):
            reasoning_block["reasoningText"] = {"text": reasoning_content}
        else:
            text: list[str] = []
            for part in reasoning_content:
                if isinstance(part, ChatCompletionContentPartTextParam):
                    text.append(part.text)
                else:  # pragma: no cover
                    msg = f"Unsupported message type: {part}"
                    raise ApiError(msg)
            reasoning_block["reasoningText"] = {"text": "".join(text)}
        content_blocks.append({"reasoningContent": reasoning_block})


def _extract_assistant_blocks(
    message_param: ChatCompletionAssistantMessageParam,
) -> list[ContentBlockTypeDef]:
    """Append assistant tool use and content blocks.

    Appends Bedrock toolUse blocks derived from OpenAI assistant message
    `tool_calls` or legacy `function_call`, followed by any textual content
    (including refusal text when present).

    Args:
        message_param: The assistant message to convert (may include tool calls).

    Returns:
        Content blocks.

    Raises:
        ApiError: If an unsupported tool call or content part type is encountered.
    """
    content_blocks: list[ContentBlockTypeDef] = []

    _map_assistant_content(content_blocks, message_param)
    _map_assistant_reasoning_content(content_blocks, message_param)

    # Tools and function calls must be at the end
    tool_calls: list[ChatCompletionMessageToolCallUnion] = (
        message_param.tool_calls if message_param.tool_calls is not None else []
    )
    for tool_call in tool_calls:
        call_id = tool_call.id
        if tool_call.type == "function":
            function_tool = tool_call.function
            name = function_tool.name
            arguments = function_tool.arguments
        elif tool_call.type == "custom":
            custom_tool = tool_call.custom
            name = custom_tool.name
            arguments = custom_tool.input
        else:  # pragma: no cover
            msg = f"Unsupported tool call type: {tool_call}"
            raise ApiError(msg)
        content_blocks.append(
            _build_tool_use_block(name=name, arguments=arguments, call_id=call_id)
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


def _parse_tool_content(text_content: str) -> ToolResultContentBlockUnionTypeDef:
    """Parses the content of a tool's textual output to determine its structure.

    The function attempts to parse the provided textual content as JSON. If it
    succeeds, the JSON structure is returned. If the parsing fails, the content
    is assumed to be plain text and is returned encapsulated in a dictionary.

    Args:
        text_content: The textual content to be parsed.

    Returns:
        A dictionary containing either the parsed JSON mapping or the original
        text content.
    """
    try:
        json_content = from_json(text_content)
    except ValueError:
        return {"text": text_content}
    else:
        return (
            {"json": json_content}
            if isinstance(json_content, dict)
            else {"text": text_content}
        )


def _extract_tool_blocks(
    message_param: ChatCompletionToolMessageParam,
) -> list[ContentBlockTypeDef]:
    """Extracts tool blocks from the given message parameter.

    Args:
        message_param: The message parameter containing tool invocation data, including
            content and tool call ID.

    Returns:
        A list of structured content blocks extracted and formatted from the given
        message parameter.
    """
    content_parts: list[ChatCompletionContentPartTextParam] = (
        [ChatCompletionContentPartTextParam(text=message_param.content, type="text")]
        if isinstance(message_param.content, str)
        else message_param.content
    )

    content: list[ToolResultContentBlockUnionTypeDef] = []
    for part in content_parts:
        if part.type == "text":
            text_content = part.text
            content.append(_parse_tool_content(text_content))

    return [
        {"toolResult": {"toolUseId": message_param.tool_call_id, "content": content}}
    ]


def _extract_function_blocks(
    message_param: ChatCompletionFunctionMessageParam,
) -> list[ContentBlockTypeDef]:
    """Extracts function blocks from the given message parameter.

    Args:
        message_param: A structured input parameter containing details about
            the chat completion function message.

    Returns:
        A list of structured content blocks with parsed tool results.
    """
    _LEGACY_FUNCTION.set(True)
    content: list[ToolResultContentBlockUnionTypeDef] = []
    text_content = message_param.content
    if text_content is not None:
        content.append(_parse_tool_content(text_content))
    return [{"toolResult": {"toolUseId": message_param.name, "content": content}}]


async def map_messages(
    messages: list[ChatCompletionMessageParam],
) -> tuple[list[MessageTypeDef], list[SystemContentBlockTypeDef]]:
    """Processes and maps a list of OpenAI message parameters into Bedrock structures.

    One for chat messages and one for system content blocks. This is done by
    categorizing and processing the input based on the role of each message and its content.

    Args:
        messages: A list of message parameters to be processed.

    Returns:
        A tuple where the first element is a list of structured chat messages and
        the second element is a list of extracted system content blocks.
    """
    bedrock_messages: list[MessageTypeDef] = []
    system_blocks: list[SystemContentBlockTypeDef] = []

    pending_tasks: dict[int, Task[list[ContentBlockTypeDef]]] = {}
    sync_roles = {"tool", "function", "assistant"}
    for i, message_param in enumerate(messages):
        role_name = message_param.role
        if role_name not in _SYSTEM_ROLES and role_name not in sync_roles:
            pending_tasks[i] = create_task(
                _extract_content_blocks(message_param.content)
            )

    previous_role_name = ""
    for i, message_param in enumerate(messages):
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
            system_blocks += _extract_system_content_blocks(system_input)
            continue

        if role_name == "tool":
            tool_msg: ChatCompletionToolMessageParam = message_param  # type: ignore[assignment]
            content_blocks = _extract_tool_blocks(tool_msg)
            if previous_role_name == "tool":
                # All consecutive tool blocks must be merged
                bedrock_messages[-1]["content"] += content_blocks  # type: ignore[operator]
                continue
        elif role_name == "function":
            function_msg: ChatCompletionFunctionMessageParam = message_param  # type: ignore[assignment]
            content_blocks = _extract_function_blocks(function_msg)
        elif role_name == "assistant":
            assistant_msg: ChatCompletionAssistantMessageParam = message_param  # type: ignore[assignment]
            content_blocks = _extract_assistant_blocks(assistant_msg)
        else:
            content_blocks = await pending_tasks[i]

        bedrock_messages.append({"role": role, "content": content_blocks})
        previous_role_name = role_name

    return bedrock_messages, system_blocks


def parse_prompt_cache_key(prompt_cache_key: str | None) -> set[PromptCaching]:
    """Parses and validates the given prompt cache key.

    Args:
        prompt_cache_key: The cache key string to be parsed.

    Returns:
        A set containing valid keys derived from the input `prompt_cache_key`.
    """
    if prompt_cache_key:
        return (
            set(prompt_cache_key.split(".")) & PROMPT_CACHING  # type: ignore[return-value]
        ) or PROMPT_CACHING
    return set()


def extract_output_text(
    contents: list[ContentBlockOutputTypeDef],
) -> tuple[str | None, str | None]:
    """Extracts output text and reasoning text from content blocks.

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
        "\n".join(content_text) if content_text else None,
        "\n".join(reasoning_text) if reasoning_text else None,
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
    contents: list[ContentBlockOutputTypeDef], *, legacy_function: bool
) -> tuple[list[ChatCompletionMessageToolCallUnion] | None, FunctionCall | None]:
    """Extracts tool calls and function calls from conversation response content.

    Args:
        contents: A list of content blocks containing response data.
        legacy_function: Whether to return a single legacy-style function call.

    Returns:
        Tuple of (tool_calls, function_call).
    """
    tool_calls: list[ChatCompletionMessageToolCallUnion] = []
    for content in contents:
        if "toolUse" not in content:
            continue
        tool_use = content["toolUse"]
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
    """Handles the generation or retrieval of audio content.

    Args:
        audio_params: Configuration parameters for audio synthesis.
        contents: A list of content blocks.
        completion_id: A unique identifier for the completion task.
        content: The text content used for synthesizing audio.
        created: A timestamp indicating the creation time.
        index: The position of this content block.

    Returns:
        ChatCompletionAudio with the generated or retrieved audio.
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
    bedrock_runtime: BedrockRuntimeClient,
    request: ConverseRequestBaseTypeDef,
    service_tier: ServiceTiers | None,
    choices_count: int,
    audio_params: ChatCompletionAudioParam | None,
    modalities: list[OutputModalities],
) -> ChatCompletion:
    """Format Bedrock Converse responses as an OpenAI ChatCompletion.

    Args:
        completion_id: Unique identifier for the completion request.
        created: Timestamp indicating when the request was created.
        model_id: The model identifier.
        bedrock_runtime: Client used to execute the Converse API.
        request: Payload for the Converse API request.
        service_tier: Optional tier of service for the request.
        choices_count: Number of response choices to generate.
        audio_params: Optional parameters for audio generation.
        modalities: List of output modalities such as text or audio.

    Returns:
        A structured ChatCompletion response.
    """
    with handle_bedrock_client_error():
        responses = await gather(
            *(bedrock_runtime.converse(**request) for _ in range(choices_count))
        )

    choices: list[Choice] = []
    usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    reasoning_contents: list[str] = []
    tts_tasks: dict[int, Task[Any]] = {}
    cached_tokens = 0
    legacy_function = _LEGACY_FUNCTION.get()
    for index, response in enumerate(responses):
        usage.prompt_tokens += response["usage"]["inputTokens"]
        usage.completion_tokens += response["usage"]["outputTokens"]
        usage.total_tokens += response["usage"]["totalTokens"]
        cached_tokens += response["usage"].get("cacheReadInputTokens", 0)
        message = response["output"]["message"]["content"]
        tool_calls, function_call = extract_tool_calls(
            message, legacy_function=legacy_function
        )
        content, reasoning_content = extract_output_text(message)
        annotations = extract_citations(message)
        if reasoning_content:
            reasoning_contents.append(reasoning_content)
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
    if cached_tokens:
        usage.prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens)
    if reasoning_contents:
        reasoning_tokens = await estimate_token_count(*reasoning_contents)
        if reasoning_tokens:
            usage.completion_tokens_details = CompletionTokensDetails(
                reasoning_tokens=reasoning_tokens
            )
            usage.total_tokens += reasoning_tokens
            usage.completion_tokens += reasoning_tokens

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
    *,
    legacy_function: bool,
) -> None:
    """Process a content block delta and update the choice delta.

    Args:
        choice_delta: The choice delta object to update.
        delta_block: The delta event containing changes.
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
                index=delta_block["contentBlockIndex"],
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
                        index=start_block["contentBlockIndex"],
                        id=tool_id,
                        type="function",
                        function=function,
                    )
                ]

        case {"contentBlockDelta": delta_block}:
            _stream_get_content_block_delta(
                choice_delta, delta_block, legacy_function=legacy_function
            )

        case {"messageStop": stop_block}:
            choice.finish_reason = map_bedrock_stop_reason(
                stop_block["stopReason"], legacy_function=legacy_function
            )
            end = True

    return chunk, end


def _stream_extract_usage_from_metadata(
    stream_event: ConverseStreamOutputTypeDef,
) -> CompletionUsage | None:
    """Extract usage data from a stream metadata event.

    Args:
        stream_event: The stream event containing metadata.

    Returns:
        CompletionUsage if metadata found, otherwise None.
    """
    if "metadata" not in stream_event:
        return None
    usage = stream_event["metadata"]["usage"]
    completion_usage = CompletionUsage(
        completion_tokens=usage["outputTokens"],
        prompt_tokens=usage["inputTokens"],
        total_tokens=usage["totalTokens"],
    )
    if (cached := usage.get("cacheReadInputTokens")) is not None:
        completion_usage.prompt_tokens_details = PromptTokensDetails(
            cached_tokens=cached
        )
    return completion_usage


async def format_stream(
    completion_id: str,
    created: int,
    model_id: str,
    bedrock_runtime: BedrockRuntimeClient,
    request: ConverseRequestBaseTypeDef,
    service_tier: ServiceTiers | None,
    *,
    include_usage: bool = False,
) -> AsyncGenerator[JSONServerSentEvent]:
    """Stream Bedrock Converse events as OpenAI ChatCompletionChunk SSE events.

    Args:
        completion_id: Unique identifier for the completion.
        created: Timestamp when the request was initiated.
        model_id: The model identifier.
        bedrock_runtime: Client for the Bedrock runtime.
        request: Request payload for the converse stream.
        service_tier: Service tier being used.
        include_usage: Whether to include usage information.

    Yields:
        JSONServerSentEvent containing the formatted response payload.
    """
    with handle_bedrock_client_error():
        stream: AsyncIterator[ConverseStreamOutputTypeDef] = (
            await bedrock_runtime.converse_stream(**request)
        )["stream"]

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
    async for event in stream:
        chunk, end = _stream_delta_chunk(
            completion_id,
            created,
            model_id,
            event,
            service_tier,
            legacy_function=legacy_function,
            chunk=chunk,
        )
        end_state |= end
        if end_state:
            if include_usage and chunk:
                usage = _stream_extract_usage_from_metadata(event)
                if usage:
                    chunk.usage = usage
        elif chunk:
            yield JSONServerSentEvent(
                data=chunk.model_dump(mode="json", exclude_none=True)
            )
            chunk = None
    if chunk:
        yield JSONServerSentEvent(data=chunk.model_dump(mode="json", exclude_none=True))
