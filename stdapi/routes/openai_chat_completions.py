"""Chat completions and responses endpoints implementation.

This module implements OpenAI-compatible endpoints for chat completions and responses,
providing AWS Bedrock integration while maintaining API compatibility. It handles both
streaming and non-streaming chat completions, tool calling, and various OpenAI parameters.

The module provides:
    - OpenAI-compatible chat completions API endpoint
    - AWS Bedrock integration for various language models
    - Streaming and non-streaming response modes
    - Tool calling and function execution support
    - Request validation and parameter conversion
    - Response formatting and usage tracking

Classes:
    CompletionCreateParams: Pydantic model for chat completion requests

Functions:
    create_chat_completion: Main FastAPI endpoint for chat completions
    Various helper functions for message conversion, validation, and response processing
"""

import contextlib
from asyncio import create_task, gather
from binascii import Error as BinasciiError
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Iterable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException
from magic import from_buffer
from pydantic import JsonValue
from pydantic_core import to_json
from sse_starlette import EventSourceResponse, JSONServerSentEvent

from stdapi.auth import authenticate
from stdapi.aws_bedrock import (
    MIME_TYPES_TO_DOCUMENT_TYPE,
    MIME_TYPES_TO_VIDEO_TYPE,
    handle_bedrock_client_error,
    image_block_from_bytes,
    image_block_from_data_url,
    image_block_from_http_url,
    image_block_from_s3_url,
    set_inference_configuration,
    set_reasoning_configuration,
)
from stdapi.config import SETTINGS
from stdapi.models import prepare_converse_request, validate_model
from stdapi.monitoring import (
    REQUEST_ID,
    REQUEST_LOG,
    REQUEST_TIME,
    log_request_params,
    log_request_stream_event,
    log_response_params,
)
from stdapi.routes.openai_audio_speech import generate_audio
from stdapi.tokenizer import estimate_token_count
from stdapi.types.openai import FunctionDefinition
from stdapi.types.openai_chat_completions import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionAudio,
    ChatCompletionAudioParam,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolUnionParam,
    Choice,
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    ChunkChoice,
    CompletionCreateParams,
    CompletionTokensDetails,
    CompletionUsage,
    File,
    FinishReason,
    FunctionCall,
    FunctionCallParam,
    OutputModalities,
    ServiceTiers,
)
from stdapi.utils import b64decode, b64encode, parse_json_mapping

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.client import BedrockRuntimeClient
    from types_aiobotocore_bedrock_runtime.literals import (
        ConversationRoleType,
        PerformanceConfigLatencyType,
        StopReasonType,
        VideoFormatType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockDeltaEventTypeDef,
        ContentBlockOutputTypeDef,
        ContentBlockStartEventTypeDef,
        ContentBlockTypeDef,
        ConverseStreamMetadataEventTypeDef,
        ConverseStreamOutputTypeDef,
        DocumentBlockTypeDef,
        MessageStopEventTypeDef,
        MessageTypeDef,
        PerformanceConfigurationTypeDef,
        ReasoningContentBlockUnionTypeDef,
        SystemContentBlockTypeDef,
        ToolChoiceTypeDef,
        ToolConfigurationTypeDef,
        ToolResultContentBlockUnionTypeDef,
        ToolSpecificationTypeDef,
        VideoBlockTypeDef,
    )

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef
    from stdapi.types.openai_chat_completions import (
        ChatCompletionNamedToolChoiceCustomParam,
    )

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1/chat", tags=["chat", "openai"]
)

#: OpenAI finish reasons to Badrock mapping
_FINISH_REASONS: "dict[StopReasonType | None, FinishReason]" = {
    "max_tokens": "length",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
    "tool_use": "tool_calls",
}

#: OpenAI "System" roles
_SYSTEM_ROLES = {"system", "developer"}

#: True if a legacy function is used
_LEGACY_FUNCTION: ContextVar[bool] = ContextVar("legacy_function")

#: Default output modalities
_DEFAULT_OUTPUT_MODALITIES: list[OutputModalities] = ["text"]


def _req_extract_system_content_blocks(
    content: str | Iterable[ChatCompletionContentPartTextParam],
) -> "list[SystemContentBlockTypeDef]":
    """Extract Bedrock system content blocks from an OpenAI content field.

    Args:
        content: Message content which may be a plain string or a list of
            ChatCompletionContentPartParam entries.

    Returns:
        A list of Bedrock SystemContentBlockTypeDef items (text blocks) in order.
    """
    results: list[SystemContentBlockTypeDef] = []
    if isinstance(content, str):
        results.append({"text": content})
    else:
        results.extend({"text": part.text} for part in content)
    return results


_IMAGES_FUNCTIONS = (
    image_block_from_data_url,
    image_block_from_s3_url,
    image_block_from_http_url,
)


async def _req_extract_image_content_block(
    image_part: ChatCompletionContentPartImageParam,
) -> "ContentBlockTypeDef":
    """Convert an OpenAI image_url section to a Bedrock content block.

    Supports data URLs, s3:// URIs, and http(s) URLs (downloaded via aiohttp).

    Args:
        image_part: Image content part as provided by OpenAI Chat API.

    Returns:
        A Bedrock ContentBlockTypeDef for the referenced image.

    Raises:
        HTTPException: If the URL is invalid or unsupported by this implementation.
    """
    url = image_part.image_url.url
    for func in _IMAGES_FUNCTIONS:
        content_block = await func(url)
        if content_block:
            return content_block
    raise HTTPException(status_code=400, detail=f"Invalid image URL: {url}")


async def _req_extract_file_content_block(file_part: File) -> "ContentBlockTypeDef":
    """Convert an OpenAI file section to a Bedrock content block.

    The OpenAI File part contains base64-encoded bytes (file_data). This helper
    detects the file's MIME type using python-magic and maps it to the proper
    Bedrock content block:
    - image/* ➜ image block with inferred format and bytes
    - video/* ➜ video block with inferred/normalized format and bytes
    - text/* or application/* ➜ document block with inferred/normalized format and bytes

    Args:
        file_part: OpenAI chat content part with type "file".

    Returns:
        A Bedrock ContentBlockTypeDef containing an image, video, or document block
        depending on the detected MIME type.

    Raises:
        HTTPException: When file_data is missing/invalid/empty or the detected
            MIME type is not supported by this implementation.
    """
    file_section = file_part.file
    b64_data = file_section.file_data
    try:
        data = await b64decode(b64_data.encode(), validate=True)
    except BinasciiError:
        raise HTTPException(
            status_code=400, detail=f"Invalid base64 data in file: {file_part}"
        ) from None
    if not data:
        raise HTTPException(
            status_code=400, detail=f"Empty file data in file: {file_part}"
        ) from None

    mime: str = from_buffer(data, mime=True)
    file_format = mime.split("/", 1)[1]

    if mime.startswith("image/"):
        return image_block_from_bytes(data, mime)

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

    if mime.startswith(("text/", "application/")):
        # Default to 'txt' when the MIME subtype is unknown
        document_format = MIME_TYPES_TO_DOCUMENT_TYPE.get(file_format, "txt")
        name_value = (
            file_section.filename
            if file_section.filename is not None
            else f"file-{document_format}"
        )
        document_block_bytes: DocumentBlockTypeDef = {
            "name": name_value,
            "source": {"bytes": data},
            "format": document_format,
        }
        return {"document": document_block_bytes}

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file MIME type for 'file' ({mime}): {file_part}",
    )


async def _req_extract_content_blocks(
    content: (
        str
        | Iterable[
            ChatCompletionContentPartParam | ChatCompletionContentPartRefusalParam
        ]
        | None
    ),
) -> list["ContentBlockTypeDef"]:
    """Extract Bedrock content blocks from OpenAI message content.

    Supports:
    - text parts
    - image_url parts with data URLs (base64), s3:// URIs, and http(s) downloads via aiohttp
    - file parts (image/video/document) with base64 body and MIME sniffing
    """
    blocks: list[ContentBlockTypeDef] = []
    if isinstance(content, str):
        blocks.append({"text": content})
        return blocks

    for part in content or ():
        if isinstance(part, ChatCompletionContentPartTextParam):
            blocks.append({"text": part.text})
        elif isinstance(part, ChatCompletionContentPartImageParam):
            blocks.append(await _req_extract_image_content_block(part))
        elif isinstance(part, File):
            blocks.append(await _req_extract_file_content_block(part))
        else:  # pragma: no cover
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content part type: {getattr(part, 'type', type(part))}",
            )

    return blocks


def _req_build_tool_use_block(
    name: str, arguments: str | JsonValue, call_id: str
) -> "ContentBlockTypeDef":
    """Build a Bedrock toolUse content block from OpenAI function call data.

    Args:
        name: Function/tool name.
        arguments: Either a JSON string (function tools) or a JSON value (custom tools).
        call_id: Optional stable tool call id..

    Returns:
        A ContentBlockTypeDef representing a toolUse block.
    """
    return {
        "toolUse": {
            "toolUseId": call_id,
            "name": name,
            "input": parse_json_mapping(arguments)
            if isinstance(arguments, str)
            else (arguments if isinstance(arguments, dict) else {}),
        }
    }


def _req_extract_assistant_blocks(
    message_param: ChatCompletionAssistantMessageParam,
) -> list["ContentBlockTypeDef"]:
    """Append assistant tool use and content blocks.

    Appends Bedrock toolUse blocks derived from OpenAI assistant message
    `tool_calls` or legacy `function_call`, followed by any textual content
    (including refusal text when present).

    Args:
        message_param: The assistant message to convert (may include tool calls).

    Returns:
        Content blocks.

    Raises:
        HTTPException: If an unsupported tool call or content part type is encountered.
    """
    content_blocks: list[ContentBlockTypeDef] = []

    _req_map_assistant_content(content_blocks, message_param)
    _req_map_assistant_reasoning_content(content_blocks, message_param)

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
            raise HTTPException(
                status_code=400, detail=f"Unsupported tool call type: {tool_call}"
            )
        content_blocks.append(
            _req_build_tool_use_block(name=name, arguments=arguments, call_id=call_id)
        )

    function_call = message_param.function_call
    if function_call is not None:
        content_blocks.append(
            _req_build_tool_use_block(
                name=function_call.name,
                arguments=function_call.arguments,
                call_id=function_call.name,
            )
        )
        _LEGACY_FUNCTION.set(True)

    return content_blocks


def _req_map_assistant_content(
    content_blocks: "list[ContentBlockTypeDef]",
    message_param: ChatCompletionAssistantMessageParam,
) -> None:
    """Maps the assistant message content into content block structures.

    Args:
        content_blocks: The list of content blocks to append the message content to.
        message_param: The assistant message.

    Raises:
        HTTPException: If the content part in `message_param` contains an unsupported
            type.
    """
    content = message_param.content
    if content is not None:
        if isinstance(content, str):
            content_blocks.append({"text": content})
        else:
            for part in content:
                if isinstance(part, ChatCompletionContentPartTextParam):
                    content_blocks.append({"text": part.text})
                elif isinstance(part, ChatCompletionContentPartRefusalParam):
                    content_blocks.append({"text": part.refusal})
                else:  # pragma: no cover
                    raise HTTPException(
                        status_code=400, detail=f"Unsupported message type: {part}"
                    )


def _req_map_assistant_reasoning_content(
    content_blocks: "list[ContentBlockTypeDef]",
    message_param: ChatCompletionAssistantMessageParam,
) -> None:
    """Maps the reasoning message content into content block structures.

    Args:
        content_blocks: The list of content blocks to append the message content to.
        message_param: The assistant message.

    Raises:
        HTTPException: If the content part in `message_param` contains an unsupported
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
                    raise HTTPException(
                        status_code=400, detail=f"Unsupported message type: {part}"
                    )
            reasoning_block["reasoningText"] = {"text": "".join(text)}
        content_blocks.append({"reasoningContent": reasoning_block})


def _req_parse_tool_content(text_content: str) -> "ToolResultContentBlockUnionTypeDef":
    """Parses the content of a tool's textual output to determine its structure.

    The function attempts to parse the provided textual content as JSON. If it
    succeeds, the JSON structure is returned. If the parsing fails, the content
    is assumed to be plain text and is returned encapsulated in a dictionary.

    Args:
        text_content (str): The textual content to be parsed.

    Returns:
        ToolResultContentBlockUnionTypeDef: A dictionary containing either the
        parsed JSON mapping or the original text content. The result will be
        in the form {"json": JSON} if the content is valid JSON, or {"text":
        text_content} otherwise.
    """
    try:
        json_content = parse_json_mapping(text_content)
    except ValueError:
        return {"text": text_content}
    else:
        return {"json": json_content}


def _req_extract_tool_blocks(
    message_param: ChatCompletionToolMessageParam,
) -> list["ContentBlockTypeDef"]:
    """Extracts tool blocks from the given message parameter.

    This function processes the content of a `ChatCompletionToolMessageParam` to break it
    into structured content blocks. It supports both string-based and object-based message
    content. Text content is parsed and converted into a standard block format for further
    processing.

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
            content.append(_req_parse_tool_content(text_content))

    return [
        {"toolResult": {"toolUseId": message_param.tool_call_id, "content": content}}
    ]


def _req_extract_function_blocks(
    message_param: ChatCompletionFunctionMessageParam,
) -> list["ContentBlockTypeDef"]:
    """Extracts function blocks from the given message parameter.

    This function processes a message parameter with possible tool content,
    parsing it into structured content blocks. It also sets an indicator
    for legacy function usage and organizes the results into the required
    format.

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
        content.append(_req_parse_tool_content(text_content))
    return [{"toolResult": {"toolUseId": message_param.name, "content": content}}]


async def _req_map_messages(
    messages: list[ChatCompletionMessageParam],
) -> tuple[list["MessageTypeDef"], list["SystemContentBlockTypeDef"]]:
    """Convert OpenAI chat messages to Bedrock Converse format.

    Splits incoming OpenAI messages into Bedrock messages (user/assistant) and
    top-level system content blocks. Bedrock expects system instructions via the
    top-level "system" parameter rather than within the messages list.

    Behaviors:
    - Maps text, image_url, and file parts into Bedrock content blocks (including
      image/video/document when applicable).
    - Converts assistant tool_calls and legacy function_call into Bedrock toolUse
      content blocks.
    - Converts tool role messages into Bedrock toolResult blocks attached to the
      user role.
    - Ensures empty user/tool messages contain an empty text block to satisfy
      Bedrock requirements.
    """
    bedrock_messages: list[MessageTypeDef] = []
    system_blocks: list[SystemContentBlockTypeDef] = []

    previous_role_name = ""
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
            system_blocks += _req_extract_system_content_blocks(system_input)
            continue

        if role_name == "tool":
            tool_msg: ChatCompletionToolMessageParam = message_param  # type: ignore[assignment]
            content_blocks = _req_extract_tool_blocks(tool_msg)
            if previous_role_name == "tool":
                # All consecutive tool blocks must be merged
                bedrock_messages[-1]["content"] += content_blocks  # type: ignore[operator]
                continue
        elif role_name == "function":
            function_msg: ChatCompletionFunctionMessageParam = message_param  # type: ignore[assignment]
            content_blocks = _req_extract_function_blocks(function_msg)
        elif role_name == "assistant":
            assistant_msg: ChatCompletionAssistantMessageParam = message_param  # type: ignore[assignment]
            content_blocks = _req_extract_assistant_blocks(assistant_msg)
        else:
            content_blocks = await _req_extract_content_blocks(message_param.content)

        bedrock_messages.append({"role": role, "content": content_blocks})
        previous_role_name = role_name

    return bedrock_messages, system_blocks


def _req_map_tools(
    request: "CompletionCreateParams",
) -> list[ChatCompletionToolUnionParam]:
    """Collect OpenAI tools (including legacy `functions` as function tools).

    Returns a list of OpenAI tool-like dictionaries. When legacy `functions`
    are provided without `tools`, they are converted into function tools.
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


def _req_map_tool_choice_literal(value: str) -> "ToolChoiceTypeDef":
    """Map OpenAI tool_choice literal to Bedrock ToolChoiceTypeDef.

    Args:
        value: One of 'auto', 'required', 'none'.

    Returns:
        Bedrock toolChoice equivalent or None when no explicit choice.
    """
    if value == "auto":
        return {"auto": {}}
    if value == "required":
        return {"any": {}}
    raise HTTPException(  # pragma: no cover
        status_code=400, detail=f"Unsupported tool choice literal: {value}"
    )


def _req_map_tool_choice(
    tool_choice: ChatCompletionToolChoiceOptionParam | None,
) -> "ToolChoiceTypeDef | None":
    """Convert OpenAI tool_choice union to a Bedrock ToolChoiceTypeDef.

    Args:
        tool_choice: None, a literal ('auto'|'required'|'none'), or a named tool choice.

    Returns:
        The Bedrock-specific toolChoice representation, or None.

    Raises:
        HTTPException: If the tool choice type is unsupported.
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return _req_map_tool_choice_literal(tool_choice)
    tool_type = tool_choice.type
    if tool_type == "function":
        function_choice: ChatCompletionNamedToolChoiceParam = tool_choice  # type: ignore[assignment]
        return {"tool": {"name": function_choice.function.name}}
    if tool_type == "custom":
        custom_choice: ChatCompletionNamedToolChoiceCustomParam = tool_choice  # type: ignore[assignment]
        return {"tool": {"name": custom_choice.custom.name}}
    raise HTTPException(  # pragma: no cover
        status_code=400,
        detail=f"Unsupported tool choice type '{tool_type}': {to_json(tool_choice).decode()}",
    )


def _req_map_function_call(
    function_call: FunctionCallParam | None,
) -> "ToolChoiceTypeDef | None":
    """Map legacy function_call to Bedrock ToolChoiceTypeDef.

    Args:
        function_call: Legacy function_call value (literal or dict with name).

    Returns:
        The corresponding Bedrock toolChoice representation, or None.
    """
    if function_call is None:
        return None
    if isinstance(function_call, str):
        return _req_map_tool_choice_literal(function_call)
    return {"tool": {"name": function_call.name}}


def _req_map_tool_or_function(
    request: "CompletionCreateParams",
) -> "ToolChoiceTypeDef | None":
    """Map OpenAI tool_choice/function_call to Bedrock ToolChoiceTypeDef."""
    return _req_map_tool_choice(request.tool_choice) or _req_map_function_call(
        request.function_call
    )


def _req_map_tool_spec(
    tool: ChatCompletionToolUnionParam,
) -> "ToolSpecificationTypeDef":
    """Convert an OpenAI tool dict to a Bedrock ToolSpecification, if possible."""
    tool_type = tool.type
    if tool_type == "function":
        function_tool: ChatCompletionFunctionToolParam = tool  # type: ignore[assignment]
        function_spec = function_tool.function
        return {
            "name": function_spec.name,
            "description": function_spec.description or tool_type,
            "inputSchema": {"json": function_spec.parameters or {}},
        }
    raise HTTPException(  # pragma: no cover
        status_code=400,
        detail=f"Unsupported tool type '{tool_type}': {to_json(tool).decode()}",
    )


def _req_build_tool_config(
    request: "CompletionCreateParams",
) -> "ToolConfigurationTypeDef | None":
    """Build Bedrock tool configuration from OpenAI tools/function fields.

    Returns None when no usable function tools are provided.
    """
    tools_specs: list[ToolSpecificationTypeDef] = []
    tools_specs.extend(_req_map_tool_spec(tool) for tool in _req_map_tools(request))
    if not tools_specs:
        return None

    tool_config: ToolConfigurationTypeDef = {
        "tools": [{"toolSpec": spec} for spec in tools_specs]
    }
    tool_choice_bedrock = _req_map_tool_or_function(request)
    if tool_choice_bedrock:
        tool_config["toolChoice"] = tool_choice_bedrock
    return tool_config


def _resp_stream_initial_chunk(
    completion_id: str, created: int, model: str, service_tier: ServiceTiers | None
) -> ChatCompletionChunk:
    """Build the initial streaming chunk setting the assistant role.

    Args:
        completion_id: Stable identifier for the streamed completion.
        created: Unix timestamp (seconds) of the request.
        model: Model identifier echoed in the response.
        service_tier: Service tier used.

    Returns:
        ChatCompletionChunk containing the initial role delta.
    """
    return ChatCompletionChunk(
        id=completion_id,
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant"))],
        created=created,
        model=model,
        object="chat.completion.chunk",
        service_tier=service_tier,
    )


def _resp_stream_get_content_block_delta(
    choice_delta: ChoiceDelta, delta_block: "ContentBlockDeltaEventTypeDef"
) -> None:
    """Apply Bedrock contentBlockDelta to an OpenAI ChoiceDelta.

    This handles text, reasoning content, and tool use arguments.
    Reasoning details aren't provided with OpenAI chat/compressions API,
    but available in Deepseek variant.
    """
    delta = delta_block["delta"]
    with contextlib.suppress(KeyError):
        choice_delta.content = delta["text"]
    with contextlib.suppress(KeyError):
        choice_delta.reasoning_content = delta["reasoningContent"]["text"]
    try:
        delta_tool_use = delta["toolUse"]
    except KeyError:
        return
    function = ChoiceDeltaFunctionCall(
        arguments=to_json(delta_tool_use["input"]).decode()
    )
    if _LEGACY_FUNCTION.get():
        choice_delta.function_call = function
    else:
        choice_delta.tool_calls = [
            ChoiceDeltaToolCall(
                index=delta_block["contentBlockIndex"],
                type="function",
                function=function,
            )
        ]


def _resp_stream_delta_chunk(
    completion_id: str,
    created: int,
    model: str,
    event: "ConverseStreamOutputTypeDef",
    service_tier: ServiceTiers | None,
    chunk: ChatCompletionChunk | None = None,
) -> "tuple[ChatCompletionChunk | None, bool]":
    """Build or update a streaming chunk based on a Bedrock stream event.

    Handles contentBlockStart (tool call start), contentBlockDelta (text/reasoning
    deltas and toolUse argument deltas), contentBlockEnd, and messageStop (finish
    reason). When a new piece of content arrives and no chunk exists yet, a new
    chunk is created; otherwise the provided chunk is updated.

    Args:
        completion_id: Stable identifier for the streamed completion.
        created: Unix timestamp (seconds) of the request.
        model: Model identifier echoed in the response.
        service_tier: Service tier used.
        event: Bedrock stream output event.
        chunk: Existing partial chunk to update, if any.

    Returns:
        A tuple (chunk_or_none, end):
        - chunk_or_none: The updated ChatCompletionChunk, or None when the event
          does not produce an emit-ready chunk (e.g., contentBlockStart without
          content).
        - end: True when the content block or message has ended and the current
          chunk should be flushed by the caller; False otherwise.
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
            model=model,
            object="chat.completion.chunk",
            service_tier=service_tier,
        )
    end = False

    if "contentBlockStart" in event:
        start_block: ContentBlockStartEventTypeDef = event["contentBlockStart"]
        start = start_block["start"]
        try:
            start_tool_use = start["toolUse"]
        except KeyError:
            return None, end
        tool_id = start_tool_use["toolUseId"]
        function = ChoiceDeltaFunctionCall(name=start_tool_use["name"])
        if _LEGACY_FUNCTION.get():
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

    elif "contentBlockDelta" in event:
        _resp_stream_get_content_block_delta(choice_delta, event["contentBlockDelta"])

    elif "messageStop" in event:
        stop_block: MessageStopEventTypeDef = event["messageStop"]
        choice.finish_reason = _resp_map_bedrock_stop_reason(stop_block["stopReason"])
        end = True

    elif "contentBlockStop" in event:
        end = True

    return chunk, end


def _resp_map_bedrock_stop_reason(stop_reason: "StopReasonType | None") -> FinishReason:
    """Translate Bedrock stop reasons to OpenAI finish reasons.

    Args:
        stop_reason: Bedrock stop reason value (or None).

    Returns:
        OpenAI stop reason value.
    """
    reason = _FINISH_REASONS.get(stop_reason, "stop")
    if _LEGACY_FUNCTION.get() and reason == "tool_calls":
        return "function_call"
    return reason


def _resp_extract_output_text_from_converse(
    contents: "list[ContentBlockOutputTypeDef]",
) -> tuple[str | None, str | None]:
    """Extract concatenated text from Bedrock Converse response output.

    Includes both standard text blocks and reasoning content blocks.

    Args:
        contents: content blocks.

    Returns:
        Concatenated text content & reasoning content.
    """
    content_text: list[str] = []
    reasoning_text: list[str] = []
    for block in contents:
        with contextlib.suppress(KeyError):
            content_text.append(block["text"])
        with contextlib.suppress(KeyError):
            reasoning_text.append(block["reasoningContent"]["reasoningText"]["text"])
    return "".join(content_text) if content_text else None, "".join(
        reasoning_text
    ) if reasoning_text else None


def _resp_extract_tool_calls_from_converse(
    contents: "list[ContentBlockOutputTypeDef]",
) -> tuple[list[ChatCompletionMessageToolCallUnion] | None, FunctionCall | None]:
    """Extract tool call from a Bedrock Converse response.

    Parses the first assistant message's content blocks and converts any
    Bedrock toolUse blocks into OpenAI ChatCompletionMessage tool_calls.

    Args:
        contents: content blocks.

    Returns:
        Tools and legacy function calls.
    """
    tool_calls: list[ChatCompletionMessageToolCallUnion] = []
    for content in contents:
        try:
            tool_use = content["toolUse"]
        except KeyError:
            continue
        function = FunctionCall(
            name=tool_use["name"], arguments=to_json(tool_use["input"]).decode()
        )
        if _LEGACY_FUNCTION.get():
            return None, function
        tool_calls.append(
            ChatCompletionMessageFunctionToolCall(
                type="function", id=tool_use["toolUseId"], function=function
            )
        )

    return tool_calls or None, None


def _resp_stream_extract_usage_from_metadata(
    stream_event: "ConverseStreamOutputTypeDef",
) -> "CompletionUsage | None":
    """Extract token usage from a Bedrock stream metadata event.

    Args:
        stream_event: Bedrock Converse stream event potentially carrying metadata.

    Returns:
        CompletionUsage object when usage is present; otherwise None.
    """
    try:
        metadata_event: ConverseStreamMetadataEventTypeDef = stream_event["metadata"]
    except KeyError:
        return None
    usage = metadata_event["usage"]
    return CompletionUsage(
        completion_tokens=usage["outputTokens"],
        prompt_tokens=usage["inputTokens"],
        total_tokens=usage["totalTokens"],
    )


async def _streaming_completion(
    completion_id: str,
    created: int,
    model_id: str,
    bedrock_runtime: "BedrockRuntimeClient",
    request: "ConverseRequestBaseTypeDef",
    service_tier: ServiceTiers | None,
    *,
    include_usage: bool = False,
) -> AsyncGenerator[JSONServerSentEvent]:
    """Stream Bedrock ConverseStream events as OpenAI ChatCompletionChunk SSEs.

    Args:
        completion_id: Stable identifier for the streamed completion.
        created: Unix timestamp (seconds) of the request.
        model_id: OpenAI model id mapped to Bedrock model/profile.
        bedrock_runtime: Bedrock client.
        request: Bedrock Converse request payload.
        service_tier: Service tier used.
        include_usage: Whether to include token usage in the final chunk.

    Yields:
        JSONServerSentEvent instances representing OpenAI ChatCompletionChunk frames.
    """
    with handle_bedrock_client_error():
        stream: AsyncIterator[ConverseStreamOutputTypeDef] = (
            await bedrock_runtime.converse_stream(**request)
        )["stream"]

    yield JSONServerSentEvent(
        data=log_response_params(
            _resp_stream_initial_chunk(
                completion_id, created, model_id, service_tier
            ).model_dump(mode="json", exclude_none=True)
        )
    )

    end_state = False
    chunk: ChatCompletionChunk | None = None
    async for event in stream:
        chunk, end = _resp_stream_delta_chunk(
            completion_id, created, model_id, event, service_tier, chunk
        )
        end_state |= end
        if end_state:
            if include_usage and chunk:
                usage = _resp_stream_extract_usage_from_metadata(event)
                if usage:
                    chunk.usage = usage
        elif chunk:
            yield JSONServerSentEvent(
                data=chunk.model_dump(mode="json", exclude_none=True)
            )
            chunk = None
    if chunk:
        yield JSONServerSentEvent(data=chunk.model_dump(mode="json", exclude_none=True))


async def _non_streaming_completion(
    completion_id: str,
    created: int,
    model_id: str,
    bedrock_runtime: "BedrockRuntimeClient",
    request: "ConverseRequestBaseTypeDef",
    service_tier: ServiceTiers | None,
    choices_count: int,
    audio_params: ChatCompletionAudioParam | None,
    modalities: list[OutputModalities],
) -> ChatCompletion:
    """Execute a non-streaming completion on Bedrock and format OpenAI response.

    Uses precise Bedrock typed responses and avoids defaulting for required fields.

    Args:
        completion_id: Stable identifier for the streamed completion.
        created: Unix timestamp (seconds) of the request.
        model_id: OpenAI model id mapped to Bedrock model/profile.
        bedrock_runtime: Bedrock client.
        request: Bedrock Converse request payload.
        service_tier: Service tier used.
        choices_count: Number of choices to return.
        audio_params: Optional audio output parameters; when provided and the
            generated content is textual, an audio response is synthesized and
            attached to the assistant message.
        modalities: Output modalities.

    Returns:
        A fully formed OpenAI ChatCompletion object.
    """
    with handle_bedrock_client_error():
        responses = await gather(
            *(bedrock_runtime.converse(**request) for _ in range(choices_count))
        )

    choices: list[Choice] = []
    usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    reasoning_contents: list[str] = []
    tts_tasks: dict[int, Awaitable[ChatCompletionAudio]] = {}
    for index, response in enumerate(responses):
        usage.prompt_tokens += response["usage"]["inputTokens"]
        usage.completion_tokens += response["usage"]["outputTokens"]
        usage.total_tokens += response["usage"]["totalTokens"]
        message = response["output"]["message"]["content"]
        tool_calls, function_call = _resp_extract_tool_calls_from_converse(message)
        content, reasoning_content = _resp_extract_output_text_from_converse(message)
        if reasoning_content:
            reasoning_contents.append(reasoning_content)
        if audio_params and content:
            tts_tasks[index] = create_task(
                _resp_generate_audio(
                    audio_params, completion_id, content, created, index
                )
            )
        choices.append(
            Choice(
                finish_reason=_resp_map_bedrock_stop_reason(response["stopReason"]),
                index=index,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content if "text" in modalities else None,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                    function_call=function_call,
                ),
            )
        )

    if reasoning_contents:
        # Estimate reasoning tokens, not included in Bedrock result
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


async def _resp_generate_audio(
    audio_params: ChatCompletionAudioParam,
    completion_id: str,
    content: str,
    created: int,
    index: int,
) -> ChatCompletionAudio:
    """Generates audio response using text-to-speech (TTS) based on provided parameters.

    Args:
        audio_params: Configuration for the generated audio.
        completion_id: Unique identifier for the audio completion process.
        content: Text content to be converted into audio. If None, no audio will
            be generated.
        created: Timestamp of when the completion process was initiated.
        index: Index value to uniquely identify this audio response.

    Returns:
        An instance of ChatCompletionAudio containing the generated audio data and
        associated metadata.
    """
    return ChatCompletionAudio(
        id=f"audio-{completion_id}-{index}",
        data=await b64encode(
            b"".join(
                [
                    chunk
                    async for chunk in await generate_audio(
                        content,
                        voice=audio_params.voice,
                        resp_format="pcm"
                        if audio_params.format == "pcm16"
                        else audio_params.format,
                    )
                ]
            )
        ),
        expires_at=created,  # Not stored on the server, so expire immediately
        transcript=content,
    )


@router.post(
    "/completions",
    response_model=None,
    summary="OpenAI - /v1/chat/completions",
    description=(
        "Creates a model response for the given chat conversation.\n"
        "This endpoint follows OpenAI's Chat Completions API shape and supports both standard and streaming responses."
    ),
    response_description="Chat completion or streaming chunks in OpenAI-compatible format",
    responses={
        200: {"description": "Chat completion created successfully."},
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "basic": {
                            "summary": "Simple text chat",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Write a haiku about the sea.",
                                    }
                                ],
                            },
                        },
                        "streaming": {
                            "summary": "Streaming response",
                            "value": {
                                "model": "amazon.nova-micro-v1:0",
                                "stream": True,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Explain gravity in one sentence.",
                                    }
                                ],
                            },
                        },
                    }
                }
            }
        }
    },
    response_model_exclude_none=True,
)
async def create_chat_completion(
    request: CompletionCreateParams, _: Annotated[None, Depends(authenticate)] = None
) -> ChatCompletion | EventSourceResponse:
    """Create a chat completion using AWS Bedrock Converse APIs.

    This endpoint is compatible with OpenAI's Chat Completions API. It maps the
    incoming OpenAI-style chat messages and parameters to AWS Bedrock's
    converse/converse_stream APIs and returns OpenAI-compatible responses.

    Args:
        request: Chat completion creation request following OpenAI spec.

    Returns:
        - ChatCompletion when stream is False.
        - EventSourceResponse streaming ChatCompletionChunk events when stream is True.

    Raises:
        HTTPException: If model is invalid or does not support text output.
    """
    log_request_params(request)
    request_user_id = request.safety_identifier or request.user
    if request_user_id:
        log = REQUEST_LOG.get()
        log["request_user_id"] = request_user_id

    model = await validate_model(request.model)
    created = int(REQUEST_TIME.get().timestamp())
    completion_id = f"chatcmpl-{REQUEST_ID.get()}"
    bedrock_messages, system_blocks = await _req_map_messages(request.messages)
    max_tokens = request.max_completion_tokens or request.max_tokens
    choices_count = request.n or 1
    additional_request_fields: dict[str, JsonValue] = {}
    inference_cfg = set_inference_configuration(
        request.model,
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
    performance_config: PerformanceConfigurationTypeDef = {}
    _LEGACY_FUNCTION.set(request.functions is not None)

    if request.reasoning_effort is not None or request.enable_thinking:
        set_reasoning_configuration(
            request.model,
            request.reasoning_effort,
            request.thinking_budget,
            max_tokens,
            additional_request_fields,
        )

    if request.service_tier is not None:
        if request.service_tier == "priority":
            latency: PerformanceConfigLatencyType = "optimized"
            service_tier: ServiceTiers | None = "priority"
        else:
            latency = "standard"
            service_tier = "default"
        performance_config["latency"] = latency
    else:
        service_tier = None

    bedrock_runtime, bedrock_request = await prepare_converse_request(
        model=model,
        bedrock_messages=bedrock_messages,
        inference_cfg=inference_cfg,
        system_blocks=system_blocks,
        tool_config=_req_build_tool_config(request),
        additional_request_fields=additional_request_fields,
        performance_config=performance_config,
    )

    if request.stream:
        return EventSourceResponse(
            await log_request_stream_event(
                _streaming_completion(
                    completion_id,
                    created,
                    request.model,
                    bedrock_runtime,
                    bedrock_request,
                    service_tier,
                    include_usage=(
                        request.stream_options is not None
                        and request.stream_options.include_usage is True
                    ),
                )
            )
        )
    return await _non_streaming_completion(
        completion_id,
        created,
        request.model,
        bedrock_runtime,
        bedrock_request,
        service_tier,
        choices_count,
        request.audio,
        request.modalities or _DEFAULT_OUTPUT_MODALITIES,
    )
