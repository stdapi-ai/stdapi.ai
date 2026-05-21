"""OpenAI Responses API adapter for Bedrock Converse.

Translates between OpenAI Responses API request/response types and
Bedrock Converse API-native types. Handles tool mapping, input mapping,
response formatting (both streaming and non-streaming), and streaming events.
"""

from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from pydantic_core import to_json
from sse_starlette import JSONServerSentEvent

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.aws_bedrock import (
    build_system_blocks,
    handle_bedrock_client_error,
    set_inference_configuration,
)
from stdapi.input_file import FileIdInputFile, InputFile
from stdapi.models import validate_model
from stdapi.models.chat._adapters import _openai_common
from stdapi.models.image import get_image_model
from stdapi.monitoring import log_error_details, log_response_params
from stdapi.tokenizer import estimate_token_count
from stdapi.types.openai import ResponseFormatJSONObject, ResponseFormatText
from stdapi.types.openai_responses import (
    CodeInterpreter,
    EasyInputMessage,
    FunctionCallInput,
    FunctionCallOutput,
    FunctionTool,
    ImageGeneration,
    ImageGenerationCall,
    IncompleteDetails,
    InputMessage,
    InputTokenCountParams,
    InputTokensDetails,
    OutputTokensDetails,
    ReasoningItemContent,
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateParams,
    ResponseFormatTextJSONSchemaConfig,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseInProgressEvent,
    ResponseInputFile,
    ResponseInputImage,
    ResponseInputText,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseOutputTextContent,
    ResponseReasoningItem,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    Tool,
    ToolChoiceFunction,
    WebSearchActionSearch,
    WebSearchActionSource,
    WebSearchPreviewTool,
    WebSearchTool,
)
from stdapi.utils import json_sse, try_parse_json

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Callable,
        Generator,
        Mapping,
    )

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime.literals import (
        CacheTTLType,
        ServiceTierTypeType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        CitationsDeltaTypeDef,
        CitationTypeDef,
        ContentBlockDeltaEventTypeDef,
        ContentBlockOutputTypeDef,
        ContentBlockStartEventTypeDef,
        ContentBlockTypeDef,
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
        ConverseTokensRequestTypeDef,
        CountTokensResponseTypeDef,
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

    from stdapi.types import JsonMapping
    from stdapi.types.anthropic_messages import ServerTools
    from stdapi.types.openai_responses import (
        ResponseInputContent,
        ResponseInputItem,
        ResponseOutputItem,
        ResponseTextConfig,
        ToolChoice,
    )

#: Empty tool schema for Bedrock tool configuration.
_EMPTY_TOOL: dict[str, str] = {"type": "object"}

#: Role names that map to Bedrock system blocks.
_SYSTEM_ROLES: frozenset[str] = frozenset({"system", "developer"})

#: PEP 695 alias for the prompt-caching scope set passed through the pipeline.
type PromptCachingScopes = frozenset[Literal["system", "messages", "tools"]]


#: Mapping from OpenAI integrated tool classes to canonical server tool names (translated to Bedrock names via tool_name_map in ``_build_tool_config``).
_TOOL_TYPE_TO_BEDROCK_NAME: MappingProxyType[type[Tool], str] = MappingProxyType(
    {
        CodeInterpreter: "code_execution",
        WebSearchTool: "web_search",
        WebSearchPreviewTool: "web_search",
    }
)

#: Input schema for the synthetic ``image_generation`` function tool presented to the LLM (gateway executes the actual generation).
_IMAGE_GENERATION_SCHEMA: JsonMapping = {
    "type": "object",
    "required": ["prompt"],
    "properties": {
        "prompt": {
            "type": "string",
            "description": "Detailed text description of the image to generate.",
        },
        "model": {
            "type": "string",
            "description": "Bedrock image model ID to use (optional, overrides gateway default).",
        },
        "size": {
            "type": "string",
            "description": "Image dimensions as WIDTHxHEIGHT, e.g. 1024x1024. Minimum 320x320.",
        },
        "quality": {"type": "string", "enum": ["low", "medium", "high"]},
        "output_format": {"type": "string", "enum": ["png", "jpeg", "webp"]},
    },
}


def _map_tool_choice(tool_choice: ToolChoice | None) -> ToolChoiceTypeDef | None:
    """Convert a Responses API tool_choice to a Bedrock ToolChoiceTypeDef.

    Args:
        tool_choice: OpenAI Responses tool choice (literal, function, or types).

    Returns:
        Bedrock toolChoice, or ``None`` when no constraint applies (``"none"`` is
        handled upstream by omitting the tool configuration altogether; built-in
        ``ToolChoiceTypes`` variants like ``file_search`` are not natively
        supported by Bedrock and are therefore ignored).

    Raises:
        ApiError: If ``tool_choice`` is an unknown string literal.
    """
    match tool_choice:
        case None:
            return None
        case "auto" | "none":
            return {"auto": {}}
        case "required":
            return {"any": {}}
        case ToolChoiceFunction(name=name):
            return {"tool": {"name": name}}
        case str():  # pragma: no cover — Pydantic filters unknown literals
            msg = f"Unsupported tool_choice literal: {tool_choice!r}"
            raise ApiError(msg)
        case _:  # pragma: no cover — ToolChoiceTypes variants unsupported by Bedrock
            return None


def _resolve_integrated_tool_name(
    tool: Tool, tool_name_map: Mapping[ServerTools, str] | None
) -> str | None:
    """Return the Bedrock name for an integrated tool, or ``None`` if not recognized.

    Args:
        tool: OpenAI Responses API tool instance.
        tool_name_map: Optional canonical → Bedrock name map.  When provided,
            the canonical name must exist in the map or ``ApiError`` is raised.

    Returns:
        Bedrock tool name, or ``None`` when *tool* is not in
        ``_TOOL_TYPE_TO_BEDROCK_NAME``.

    Raises:
        ApiError: If *tool_name_map* is provided and the canonical name is absent.
    """
    if (canonical_name := _TOOL_TYPE_TO_BEDROCK_NAME.get(type(tool))) is None:
        return None
    if tool_name_map is None:
        return canonical_name
    if canonical_name not in tool_name_map:
        tool_type = getattr(tool, "type", type(tool).__name__)
        msg = f"Server tool '{tool_type}' is not supported by this model."
        raise ApiError(msg)
    return tool_name_map[canonical_name]  # type: ignore[index]


def _build_tool_config(
    request: ResponseCreateParams,
    tool_name_map: Mapping[ServerTools, str] | None = None,
) -> ToolConfigurationTypeDef | None:
    """Build a Bedrock tool configuration from a Responses API request.

    Maps ``FunctionTool`` entries to Bedrock toolSpec and OpenAI integrated
    tool types (code_interpreter, web_search, image_generation) to their
    Bedrock equivalents.  Unsupported tool types are rejected at the Pydantic
    validation layer before this function is reached.

    When ``tool_choice="none"``, no tool config is returned so that the model
    cannot call any tools.

    Args:
        request: Responses API creation request.
        tool_name_map: Optional mapping from canonical server tool name to
            Bedrock system tool name.  When provided, integrated tool types are
            translated to Bedrock names; if a canonical name is absent, an
            ``ApiError`` is raised.

    Returns:
        Bedrock tool configuration, or ``None`` if no tools are present
        or tool calling is disabled.

    Raises:
        ApiError: If ``tool_name_map`` is provided and an integrated tool's
            canonical name is absent from the map.
    """
    if request.tool_choice == "none" or not request.tools:
        return None

    tools: list[ToolTypeDef] = []
    has_image_gen = False

    for tool in request.tools:
        if isinstance(tool, FunctionTool):
            tools.append(
                {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description or "function",
                        "inputSchema": {"json": tool.parameters or _EMPTY_TOOL},
                    }
                }
            )
        elif isinstance(tool, ImageGeneration):
            # Gateway handles ImageGeneration; expose a synthetic function tool so the LLM can request it.
            if not has_image_gen:
                tools.append(
                    {
                        "toolSpec": {
                            "name": "image_generation",
                            "description": "Generate an image from a text description.",
                            "inputSchema": {"json": _IMAGE_GENERATION_SCHEMA},
                        }
                    }
                )
                has_image_gen = True
        elif bedrock_name := _resolve_integrated_tool_name(tool, tool_name_map):
            tools.append(
                {
                    "toolSpec": {
                        "name": bedrock_name,
                        "inputSchema": {"json": dict(_EMPTY_TOOL)},
                    }
                }
            )

    if not tools:
        return None

    tool_config: ToolConfigurationTypeDef = {"tools": tools}
    if bedrock_choice := _map_tool_choice(request.tool_choice):
        tool_config["toolChoice"] = bedrock_choice
    return tool_config


def get_image_generation_tool(request: ResponseCreateParams) -> ImageGeneration | None:
    """Return the ImageGeneration tool from request.tools, if present.

    Part of this module's cross-module public surface: consumed by
    :mod:`stdapi.models.chat._default` to decide whether to intercept
    ``image_generation`` tool calls before/after the Bedrock Converse round-trip.

    Args:
        request: Responses API creation request.

    Returns:
        The first ``ImageGeneration`` tool found, or ``None`` if absent.
    """
    return next(
        (t for t in request.tools or () if isinstance(t, ImageGeneration)), None
    )


#: Status values for ``ImageGenerationCall.status``.
_ImageStatus = Literal["in_progress", "completed", "generating", "failed"]


def _str_args(text: str) -> dict[str, str]:
    """Parse JSON text and return only its string-valued fields.

    Args:
        text: Raw JSON string (typically tool-call arguments).

    Returns:
        Mapping of string key → string value; non-string values and parse
        failures are silently dropped.
    """
    if not isinstance((raw := try_parse_json(text)), dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(v, str)}


async def _generate_image_b64(
    args: dict[str, str], tool: ImageGeneration, fallback_model: str | None
) -> str | None:
    """Run a single image-generation job and return its base64 payload.

    Model resolution priority: ``args["model"]`` → ``tool.model`` → *fallback_model*.

    Args:
        args: String-valued arguments parsed from the tool call JSON.
        tool: The ``ImageGeneration`` tool definition from the request.
        fallback_model: Operator-configured default image model ID, or ``None``.

    Returns:
        Base64-encoded image, or ``None`` if no image was produced.

    Raises:
        ApiError: If no image model is configured.
    """
    if not (model_id := args.get("model") or tool.model or fallback_model):
        msg = (
            "No image model configured for the image_generation tool. "
            "Set IMAGE_GENERATION_MODEL or specify a model in the tool definition."
        )
        raise ApiError(msg, status=400)
    validated = await validate_model(
        model_id, input_modality="TEXT", output_modality="IMAGE", error_status=400
    )
    size = args.get("size") or tool.size or "1024x1024"
    width, height = map(int, ("1024x1024" if size == "auto" else size).split("x"))
    job = get_image_model(validated.id).get_image_generation_job(
        prompt=args.get("prompt") or "",
        count=1,
        width=width,
        height=height,
        quality=args.get("quality") or tool.quality,
        style=None,
        output_format=args.get("output_format") or tool.output_format or "png",  # type: ignore[arg-type]
        output_compression=tool.output_compression or 100,
        extra_params={},
    )
    images = list(await job.generate_images())
    return images[0].image if images else None


async def execute_image_generation_calls(
    output_items: list[ResponseOutputItem],
    image_gen_tool: ImageGeneration,
    response_id: str,
    fallback_model: str | None,
) -> list[ResponseOutputItem]:
    """Replace ``image_generation`` function-call items with ``ImageGenerationCall``.

    All other items pass through unchanged. Executes synchronously; errors propagate.

    Args:
        output_items: Output items from the Bedrock response.
        image_gen_tool: The ``ImageGeneration`` tool definition from the request.
        response_id: Response ID used to generate stable ``ImageGenerationCall`` IDs.
        fallback_model: Operator-configured default image model ID, or ``None``.

    Returns:
        New output list with image-generation calls materialised.

    Raises:
        ApiError: If no image model is configured and none was specified.
    """
    result: list[ResponseOutputItem] = []
    counter = 0
    for item in output_items:
        if not (
            isinstance(item, ResponseFunctionToolCall)
            and item.name == "image_generation"
        ):
            result.append(item)
            continue
        counter += 1
        b64 = await _generate_image_b64(
            _str_args(item.arguments), image_gen_tool, fallback_model
        )
        result.append(
            ImageGenerationCall(
                id=f"{response_id}-img-{counter}",
                status="completed" if b64 else "failed",
                type="image_generation_call",
                result=b64,
            )
        )
    return result


async def image_generation_stream_handler(
    state: _StreamState,
    image_gen_tool: ImageGeneration,
    response_id: str,
    fallback_model: str | None,
) -> AsyncGenerator[JSONServerSentEvent]:
    """Emit SSE events for suppressed ``image_generation`` tool calls post-stream.

    Invoked after the main Bedrock stream completes. For each suppressed call,
    runs generation, appends the resulting ``ImageGenerationCall`` to
    ``state.output_items``, and yields ``response.output_item.added`` +
    ``response.output_item.done`` events. Exceptions are suppressed so a single
    failed image does not abort the stream.

    Args:
        state: Mutable stream state from ``format_stream``.
        image_gen_tool: The ``ImageGeneration`` tool definition from the request.
        response_id: Response ID used to generate stable item IDs.
        fallback_model: Operator-configured default image model ID, or ``None``.

    Yields:
        SSE events for each generated image.
    """
    counter = 0
    for _tool_id, tool_name, args_json in state.suppressed_tool_calls:
        if tool_name != "image_generation":
            continue

        b64: str | None = None
        status: _ImageStatus = "failed"
        with suppress(Exception):
            b64 = await _generate_image_b64(
                _str_args(args_json), image_gen_tool, fallback_model
            )
            status = "completed" if b64 else "failed"

        counter += 1
        image_call = ImageGenerationCall(
            id=f"{response_id}-img-{counter}",
            status=status,
            type="image_generation_call",
            result=b64,
        )
        state.output_items.append(image_call)
        for event_type, event_cls in (
            ("response.output_item.added", ResponseOutputItemAddedEvent),
            ("response.output_item.done", ResponseOutputItemDoneEvent),
        ):
            yield JSONServerSentEvent(
                event=event_type,
                data=event_cls(
                    item=image_call,
                    output_index=state.output_index,
                    sequence_number=state.next_seq(),
                    type=event_type,  # type: ignore[arg-type]
                ).model_dump(mode="json", exclude_none=True),
            )
        state.output_index += 1


def _build_output_config(
    text: ResponseTextConfig | None,
) -> JsonSchemaDefinitionTypeDef | None:
    """Convert a Responses API ``text.format`` to a Bedrock ``outputConfig`` schema.

    Uses Bedrock's native structured output (``outputConfig.textFormat``)
    instead of injecting JSON instructions into the system prompt: this is
    enforced at the decoding layer, does not pollute the model's context,
    and keeps behaviour consistent with the Chat Completions adapter. Models
    that do not support ``outputConfig`` will surface a Bedrock-side error,
    which is the correct signal rather than a silent fallback to best-effort
    prompting.

    Args:
        text: Text configuration specifying the desired output format.

    Returns:
        Bedrock JSON schema definition dict, or ``None`` for plain-text format.
    """
    if text is None or text.format is None:
        return None
    match text.format:
        case ResponseFormatText():
            return None
        case ResponseFormatJSONObject():
            # Bedrock rejects an empty ``{}`` schema ("Empty schema that
            # accepts any JSON value is not supported") and, under strict
            # structured output, requires ``additionalProperties: false``
            # on every object. The closest universal representation of
            # "any JSON object" is therefore an object with no declared
            # properties and additional properties disallowed.
            return {"schema": '{"type": "object", "additionalProperties": false}'}
        case ResponseFormatTextJSONSchemaConfig(schema_=schema):
            return {
                "schema": schema
                if isinstance(schema, str)
                else to_json(schema).decode()
            }
        case _:  # pragma: no cover
            return None


def translate_request(
    request: ResponseCreateParams,
    model_id: str,
    *,
    tool_name_map: Mapping[ServerTools, str] | None = None,
) -> tuple[
    InferenceConfigurationTypeDef,
    JsonMapping,
    ToolConfigurationTypeDef | None,
    JsonSchemaDefinitionTypeDef | None,
    ServiceTierTypeType | None,
    PromptCachingScopes | None,
    CacheTTLType | None,
    dict[str, str] | None,
]:
    """Translate OpenAI Responses API request parameters into Bedrock Converse inputs.

    Args:
        request: Responses API creation request.
        model_id: The Bedrock model identifier.
        tool_name_map: Optional mapping from canonical server tool name to Bedrock
            system tool name.  When provided, integrated tool types are translated
            to Bedrock names before being added as ``toolSpec`` entries.

    Returns:
        Tuple of ``(inference_cfg, additional_request_fields, tool_config,
        output_config, service_tier, prompt_caching, prompt_caching_ttl,
        request_metadata)``. ``output_config`` is a Bedrock
        ``JsonSchemaDefinitionTypeDef`` passed through to Converse's native
        ``outputConfig.textFormat`` when JSON output is requested.

    Raises:
        ApiError: If ``tool_name_map`` is provided and an integrated tool's
            canonical name is absent from the map.
    """
    additional_request_fields: JsonMapping = {}
    return (
        set_inference_configuration(
            model_id,
            additional_request_fields,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_output_tokens,
        ),
        additional_request_fields,
        _build_tool_config(request, tool_name_map),
        _build_output_config(request.text),
        _openai_common.map_service_tier(request.service_tier)[0],
        (
            frozenset(_openai_common.parse_prompt_cache_key(request.prompt_cache_key))
            if request.prompt_cache_key
            else None
        ),
        (
            _openai_common.CACHE_TTL.get(request.prompt_cache_retention)
            if request.prompt_cache_retention
            else None
        ),
        (dict(request.metadata) if request.metadata else None),
    )


async def _convert_input_content(part: ResponseInputContent) -> ContentBlockTypeDef:
    """Convert a single ResponseInputContent part to a Bedrock content block.

    Args:
        part: An input content item (text, image, or file).

    Returns:
        Bedrock content block.

    Raises:
        ApiError: If the content type is not supported.
    """
    match part:
        case ResponseInputText(text=text) | ResponseOutputTextContent(text=text):
            return {"text": text}
        case ResponseInputImage(file_id=fid) if fid is not None:
            return await FileIdInputFile(fid).to_bedrock_content_block()
        case ResponseInputImage(image_url=url) if url is not None:
            return await InputFile(url).to_bedrock_content_block()
        case ResponseInputFile(file_id=fid) if fid is not None:
            return await FileIdInputFile(fid).to_bedrock_content_block()
        case ResponseInputFile(file_url=url) if url is not None:
            return await InputFile(url).to_bedrock_content_block()
        case ResponseInputFile(file_data=data) if data is not None:
            return await InputFile(data).to_bedrock_content_block()
        case _:
            msg = f"Unsupported input content type: {getattr(part, 'type', type(part))}"
            raise ApiError(msg)


def _append_or_merge(
    bedrock_messages: list[MessageTypeDef],
    role: Literal["assistant", "user"],
    blocks: list[ContentBlockTypeDef],
) -> None:
    """Append ``blocks`` to ``bedrock_messages`` under ``role``, merging with the last.

    Merges into the trailing message when its ``role`` matches; otherwise appends
    a new message.  Empty ``blocks`` lists are a no-op.

    Args:
        bedrock_messages: Mutable Bedrock messages list to append to.
        role: Bedrock role of the blocks to append.
        blocks: Content blocks to append.
    """
    if not blocks:
        return
    if bedrock_messages and bedrock_messages[-1]["role"] == role:
        bedrock_messages[-1]["content"] += blocks  # type: ignore[operator]
    else:
        bedrock_messages.append({"role": role, "content": blocks})


async def _map_message_item(
    item: EasyInputMessage | InputMessage,
    bedrock_messages: list[MessageTypeDef],
    system_blocks: list[SystemContentBlockTypeDef],
) -> None:
    """Map a single message item into Bedrock messages and system blocks.

    Handles user, assistant, system, and developer roles.  Consecutive
    messages with the same Bedrock role are merged into a single message.

    Args:
        item: The message input item to map.
        bedrock_messages: Mutable Bedrock messages list to append to.
        system_blocks: Mutable system blocks list to append to.
    """
    content = item.content
    if (role := item.role) in _SYSTEM_ROLES:
        text = (
            content
            if isinstance(content, str)
            else " ".join(p.text for p in content if isinstance(p, ResponseInputText))
        )
        if text:
            system_blocks.extend(build_system_blocks(text))
        return

    if isinstance(content, str):
        blocks: list[ContentBlockTypeDef] = [{"text": content}] if content else []
    else:
        blocks = [await _convert_input_content(p) for p in content]

    bedrock_role: Literal["assistant", "user"] = (
        "assistant" if role == "assistant" else "user"
    )
    _append_or_merge(bedrock_messages, bedrock_role, blocks)


async def _map_output_message(
    item: ResponseOutputMessage, bedrock_messages: list[MessageTypeDef]
) -> None:
    """Map a ResponseOutputMessage (echoed assistant output) as a Bedrock assistant message.

    When a client echoes back the full previous API response in the input array
    (as done by Codex CLI), assistant messages arrive as ``ResponseOutputMessage``
    items with ``role="assistant"`` and content blocks of type ``output_text``.
    These are mapped to plain Bedrock text blocks.

    Args:
        item: The output message to map.
        bedrock_messages: Mutable Bedrock messages list to append to.
    """
    _append_or_merge(
        bedrock_messages,
        "assistant",
        [
            {"text": part.text}
            for part in item.content
            if isinstance(part, ResponseOutputText)
        ],
    )


async def _map_function_call(
    item: FunctionCallInput, bedrock_messages: list[MessageTypeDef]
) -> None:
    """Map a function_call input item as a Bedrock toolUse assistant message.

    A ``function_call`` item echoes back the model's previous tool invocation
    so Bedrock can reconstruct the conversation context.  Consecutive
    ``function_call`` items are merged into the same assistant message.

    Args:
        item: The function call input item.
        bedrock_messages: Mutable Bedrock messages list to append to.
    """
    tool_input: ToolUseBlockTypeDef = {
        "toolUseId": item.call_id,
        "name": item.name,
        "input": try_parse_json(item.arguments) if item.arguments else {},  # type: ignore[typeddict-item]
    }
    _append_or_merge(bedrock_messages, "assistant", [{"toolUse": tool_input}])


async def _map_function_call_output(
    item: FunctionCallOutput, bedrock_messages: list[MessageTypeDef]
) -> None:
    """Map a function_call_output item as a Bedrock toolResult user message.

    Merges with the previous user message when possible.

    Args:
        item: The function call output item.
        bedrock_messages: Mutable Bedrock messages list to append to.
    """
    output = item.output
    tool_content: list[ToolResultContentBlockUnionTypeDef] = (
        [_openai_common.parse_tool_content(output)]
        if isinstance(output, str)
        else [
            _openai_common.parse_tool_content(part.text)
            for part in output
            if isinstance(part, ResponseInputText)
        ]
    )
    _append_or_merge(
        bedrock_messages,
        "user",
        [{"toolResult": {"toolUseId": item.call_id, "content": tool_content}}],
    )


async def map_input(
    input_param: str | list[ResponseInputItem] | None, instructions: str | None
) -> tuple[list[MessageTypeDef], list[SystemContentBlockTypeDef]]:
    """Convert a Responses API input to Bedrock messages and system blocks.

    Args:
        input_param: The ``input`` field from ResponseCreateParams.
        instructions: Optional system-level instruction string.

    Returns:
        Tuple of ``(bedrock_messages, system_blocks)``.
    """
    bedrock_messages: list[MessageTypeDef] = []
    system_blocks: list[SystemContentBlockTypeDef] = []

    if instructions:
        system_blocks.extend(build_system_blocks(instructions))

    if input_param is None:
        return bedrock_messages, system_blocks

    if isinstance(input_param, str):
        bedrock_messages.append({"role": "user", "content": [{"text": input_param}]})
        return bedrock_messages, system_blocks

    for item in input_param:
        match item:
            case EasyInputMessage() | InputMessage():
                await _map_message_item(item, bedrock_messages, system_blocks)
            case ResponseOutputMessage():
                await _map_output_message(item, bedrock_messages)
            case FunctionCallInput():
                await _map_function_call(item, bedrock_messages)
            case FunctionCallOutput():
                await _map_function_call_output(item, bedrock_messages)
            case ResponseReasoningItem():
                _map_reasoning_item(item, bedrock_messages)

    return bedrock_messages, system_blocks


def _map_reasoning_item(
    item: ResponseReasoningItem, bedrock_messages: list[MessageTypeDef]
) -> None:
    """Map an echoed ``ResponseReasoningItem`` to a Bedrock ``reasoningContent`` block.

    Clients (e.g. Codex CLI) echo reasoning items back as part of the input
    history.  Each non-empty ``content`` entry is converted to a Bedrock
    ``reasoningContent`` block and merged into the current assistant message.
    Empty items are dropped without logging.

    Args:
        item: The reasoning item to map.
        bedrock_messages: Mutable Bedrock messages list to append to.
    """
    _append_or_merge(
        bedrock_messages,
        "assistant",
        [
            {"reasoningContent": {"reasoningText": {"text": part.text}}}
            for part in item.content or ()
            if isinstance(part, ReasoningItemContent) and part.text
        ],
    )


def _extract_web_search_sources(
    contents: list[ContentBlockOutputTypeDef],
) -> list[WebSearchActionSource] | None:
    """Extract web search source URLs from Bedrock citationsContent blocks.

    When ``nova_grounding`` is used, Bedrock embeds citation URLs in
    ``citationsContent`` within text blocks.  This helper collects those URLs
    as ``WebSearchActionSource`` objects suitable for populating the
    ``action.sources`` field of a ``ResponseFunctionWebSearch`` item.

    Args:
        contents: Bedrock response content blocks.

    Returns:
        List of source objects, or ``None`` if no citations were found.
    """
    return [
        WebSearchActionSource(type="url", url=web["url"])
        for block in contents
        if (citations_block := block.get("citationsContent"))
        for citation in citations_block.get("citations", ())
        if (web := citation.get("location", {}).get("web")) and web.get("url")
    ] or None


def _extract_output_items(
    contents: list[ContentBlockOutputTypeDef],
    response_id: str,
    suppress_tool_names: frozenset[str] | None,
    web_search_tool_names: frozenset[str] | None = None,
) -> list[ResponseOutputItem]:
    """Extract ResponseOutputItem objects from Bedrock response content.

    Processes content blocks in their original Bedrock order so that
    ``web_search_call`` items appear before the assistant message, matching
    the official OpenAI API output ordering.

    Args:
        contents: Bedrock response content blocks.
        response_id: The response identifier for generating item IDs.
        suppress_tool_names: Tool names whose function_call items should be
            filtered from output (e.g. ``{"nova_code_interpreter"}``).
        web_search_tool_names: Tool names that should be emitted as
            ``web_search_call`` output items instead of being suppressed
            (e.g. ``{"nova_grounding"}``).  Sources are populated from
            ``citationsContent`` blocks in the response.

    Returns:
        List of output items. ``web_search_call`` and ``function_call`` items
        keep their original Bedrock block position; the assembled ``message``
        item (if any text blocks were present) is appended last.
    """
    sources: list[WebSearchActionSource] | None = (
        _extract_web_search_sources(contents) if web_search_tool_names else None
    )

    text_parts: list[str] = []
    output_items: list[ResponseOutputItem] = []

    for block in contents:
        if tool_use := block.get("toolUse"):
            name: str = tool_use["name"]
            if web_search_tool_names and name in web_search_tool_names:
                input_data = tool_use["input"]
                query = (
                    input_data.get("query", "") if isinstance(input_data, dict) else ""
                )
                output_items.append(
                    ResponseFunctionWebSearch(
                        id=f"{response_id}-ws-{tool_use['toolUseId']}",
                        type="web_search_call",
                        status="completed",
                        action=WebSearchActionSearch(
                            type="search",
                            query=query,
                            queries=[query] if query else None,
                            sources=sources,
                        ),
                    )
                )
            elif suppress_tool_names is None or name not in suppress_tool_names:
                output_items.append(
                    ResponseFunctionToolCall(
                        arguments=to_json(tool_use["input"]).decode(),
                        call_id=tool_use["toolUseId"],
                        name=name,
                        type="function_call",
                        id=f"{response_id}-fc-{tool_use['toolUseId']}",
                        status="completed",
                    )
                )
        elif text := block.get("text"):
            text_parts.append(text)

    if text_parts:
        output_items.append(
            ResponseOutputMessage(
                id=f"{response_id}-msg-0",
                content=[
                    ResponseOutputText(
                        annotations=[], text="\n".join(text_parts), type="output_text"
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        )

    return output_items


#: Maps Bedrock stop reasons; unknown reasons default to ``"max_output_tokens"``.
_INCOMPLETE_REASONS: dict[str, Literal["max_output_tokens", "content_filter"]] = {
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
}


def _map_stop_reason(
    stop_reason: str | None,
) -> tuple[Literal["completed", "incomplete", "failed"], IncompleteDetails | None]:
    """Map a Bedrock stop reason to a Responses API status and incomplete details.

    Args:
        stop_reason: Bedrock stop reason value.

    Returns:
        Tuple of ``(status, incomplete_details)`` where ``status`` is one of
        ``"completed"``, ``"incomplete"``, or ``"failed"`` and ``incomplete_details``
        is populated when the response was cut short.
    """
    match stop_reason:
        case "malformed_model_output" | "malformed_tool_use":
            return "failed", None
        case "end_turn" | "stop_sequence" | "tool_use" | None:
            return "completed", None
        case _:
            if stop_reason not in _INCOMPLETE_REASONS:
                log_error_details(
                    f"Unknown Bedrock stopReason: {stop_reason!r}", level="warning"
                )
            return "incomplete", IncompleteDetails(
                reason=_INCOMPLETE_REASONS.get(stop_reason, "max_output_tokens")
            )


def _build_response_object(
    response_id: str,
    created_at: float,
    model_id: str,
    output_items: list[ResponseOutputItem],
    status: Literal["completed", "incomplete", "failed", "in_progress"],
    incomplete_details: IncompleteDetails | None,
    usage: ResponseUsage | None,
    request: ResponseCreateParams,
) -> Response:
    """Construct a Response object from accumulated state.

    Args:
        response_id: Unique identifier for this response.
        created_at: Unix timestamp of response creation.
        model_id: The Bedrock model identifier.
        output_items: Accumulated output items (messages and function calls).
        status: Response completion status (``"in_progress"`` for lifecycle events).
        incomplete_details: Details explaining why the response is incomplete, or ``None``.
        usage: Token usage statistics, or ``None`` for in-progress responses.
        request: The original Responses API creation request.

    Returns:
        Constructed Response object.
    """
    return Response(
        id=response_id,
        created_at=created_at,
        model=model_id,
        object="response",
        output=output_items,
        parallel_tool_calls=request.parallel_tool_calls is not False,
        temperature=request.temperature,
        tool_choice=request.tool_choice or "auto",
        tools=list(request.tools) if request.tools else [],
        top_p=request.top_p,
        status=status,
        incomplete_details=incomplete_details,
        metadata=request.metadata,
        max_output_tokens=request.max_output_tokens,
        previous_response_id=request.previous_response_id,
        text=request.text,
        top_logprobs=request.top_logprobs,
        usage=usage,
        user=request.user,
    )


async def format_response(
    response_id: str,
    created_at: float,
    model_id: str,
    bedrock_response: ConverseResponseTypeDef,
    request: ResponseCreateParams,
    suppress_tool_names: frozenset[str] | None = None,
    web_search_tool_names: frozenset[str] | None = None,
) -> Response:
    """Build a Response from a Bedrock Converse response.

    Args:
        response_id: Unique identifier for this response.
        created_at: Unix timestamp of response creation.
        model_id: The Bedrock model identifier.
        bedrock_response: Raw Bedrock converse response.
        request: The original Responses API creation request.
        suppress_tool_names: Tool names to filter from output items.
        web_search_tool_names: Tool names to emit as ``web_search_call`` output
            items (populated with query and ``citationsContent`` sources).

    Returns:
        Completed Response object.
    """
    contents: list[ContentBlockOutputTypeDef] = bedrock_response["output"]["message"][
        "content"
    ]
    usage_raw = bedrock_response["usage"]

    input_tokens = usage_raw["inputTokens"]
    output_tokens = usage_raw["outputTokens"]

    reasoning_parts = [
        rc["reasoningText"]["text"]
        for block in contents
        if (rc := block.get("reasoningContent")) and "reasoningText" in rc
    ]
    reasoning_tokens = (
        (await estimate_token_count(*reasoning_parts) or 0) if reasoning_parts else 0
    )

    return log_response_params(
        _build_response_object(
            response_id,
            created_at,
            model_id,
            _extract_output_items(
                contents, response_id, suppress_tool_names, web_search_tool_names
            ),
            *_map_stop_reason(bedrock_response.get("stopReason")),
            ResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=InputTokensDetails(
                    cached_tokens=usage_raw.get("cacheReadInputTokens", 0)
                ),
                output_tokens=output_tokens + reasoning_tokens,
                output_tokens_details=OutputTokensDetails(
                    reasoning_tokens=reasoning_tokens
                ),
                total_tokens=input_tokens + output_tokens + reasoning_tokens,
            ),
            request,
        )
    )


class _BlockKind(Enum):
    """Kind of content block currently being streamed.

    Replaces the mutually-exclusive boolean flags (``is_text_block``,
    ``is_tool_block``, ``suppress_block``, ``is_web_search_block``) with a
    single tagged value so that invalid combinations cannot be represented.
    """

    NONE = "none"
    TEXT = "text"
    TOOL = "tool"
    SUPPRESSED = "suppressed"
    WEB_SEARCH = "web_search"


@dataclass(slots=True)
class _StreamState:
    """Mutable state accumulated during a Bedrock ConverseStream response.

    Centralises all per-stream variables so the block-handler generators can
    receive a single argument instead of a sprawling set of nonlocal bindings.

    ``pending_web_search_sources`` is populated from post-stop
    ``citationsContent`` blocks and folded back into the corresponding
    ``ResponseFunctionWebSearch`` in ``output_items`` before the final
    ``response.completed`` event is emitted.
    """

    response_id: str
    seq: int = 0
    output_index: int = 0
    output_items: list[ResponseOutputItem] = field(default_factory=list)
    current_item_id: str | None = None
    current_tool_name: str | None = None
    current_tool_id: str | None = None
    current_text: str = ""
    current_args: str = ""
    block_kind: _BlockKind = _BlockKind.NONE
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    #: Accumulated reasoning text across ``reasoningContent`` deltas.
    reasoning_text: str = ""
    #: Completed suppressed tool calls as ``(tool_id, tool_name, args_json)``.
    suppressed_tool_calls: list[tuple[str, str, str]] = field(default_factory=list)
    #: Web-search sources keyed by ``item_id`` (see class docstring).
    pending_web_search_sources: dict[str, list[WebSearchActionSource]] = field(
        default_factory=dict
    )

    def next_seq(self) -> int:
        """Return the current sequence number and advance the counter.

        Returns:
            The sequence number assigned to the caller.
        """
        self.seq += 1
        return self.seq - 1

    def reset_block(self) -> None:
        """Reset per-content-block tracking fields."""
        self.block_kind = _BlockKind.NONE
        self.current_item_id = None
        self.current_tool_name = None
        self.current_tool_id = None
        self.current_text = ""
        self.current_args = ""


def _handle_block_start(
    state: _StreamState,
    start_block: ContentBlockStartEventTypeDef,
    suppress_tool_names: frozenset[str] | None,
    web_search_tool_names: frozenset[str] | None = None,
) -> Generator[JSONServerSentEvent]:
    """Emit SSE events for a Bedrock ``contentBlockStart`` event.

    Initialises per-block state and emits ``response.output_item.added``
    plus (for text blocks) ``response.content_part.added``.

    For web-search system tools (``nova_grounding``), emits a
    ``web_search_call`` output item with ``in_progress`` status followed by a
    ``response.web_search_call.in_progress`` event instead of a
    ``function_call`` item.

    Args:
        state: Mutable stream state.
        start_block: The ``contentBlockStart`` event from Bedrock.
        suppress_tool_names: Tool names whose output should be fully suppressed.
        web_search_tool_names: Tool names to emit as ``web_search_call`` items.

    Yields:
        ``JSONServerSentEvent`` objects for the block start.
    """
    start = start_block["start"]
    state.current_text = ""
    state.current_args = ""

    if tool_use := start.get("toolUse"):
        state.current_tool_name = tool_use["name"]
        state.current_tool_id = tool_use["toolUseId"]
        state.block_kind = _BlockKind.TOOL

        if web_search_tool_names and state.current_tool_name in web_search_tool_names:
            state.block_kind = _BlockKind.WEB_SEARCH
            state.current_item_id = f"{state.response_id}-ws-{state.current_tool_id}"
            ws_item = ResponseFunctionWebSearch(
                id=state.current_item_id,
                type="web_search_call",
                status="in_progress",
                action=WebSearchActionSearch(type="search", query="", sources=None),
            )
            yield json_sse(
                "response.output_item.added",
                ResponseOutputItemAddedEvent(
                    item=ws_item,
                    output_index=state.output_index,
                    sequence_number=state.next_seq(),
                    type="response.output_item.added",
                ),
            )
            yield json_sse(
                "response.web_search_call.in_progress",
                ResponseWebSearchCallInProgressEvent(
                    item_id=state.current_item_id,
                    output_index=state.output_index,
                    sequence_number=state.next_seq(),
                    type="response.web_search_call.in_progress",
                ),
            )
            return

        state.current_item_id = f"{state.response_id}-fc-{state.current_tool_id}"
        if suppress_tool_names and state.current_tool_name in suppress_tool_names:
            state.block_kind = _BlockKind.SUPPRESSED
            return

        tool_item = ResponseFunctionToolCall(
            arguments="",
            call_id=state.current_tool_id,
            name=state.current_tool_name,
            type="function_call",
            id=state.current_item_id,
            status="in_progress",
        )
        yield json_sse(
            "response.output_item.added",
            ResponseOutputItemAddedEvent(
                item=tool_item,
                output_index=state.output_index,
                sequence_number=state.next_seq(),
                type="response.output_item.added",
            ),
        )
    else:
        state.current_item_id = f"{state.response_id}-msg-{state.output_index}"
        state.block_kind = _BlockKind.TEXT
        yield from _emit_text_block_start(state)


def _emit_text_block_start(state: _StreamState) -> Generator[JSONServerSentEvent]:
    """Yield the two SSE events that open a new text output block.

    Emits ``response.output_item.added`` (for the assistant message) followed
    by ``response.content_part.added`` (for the text content part).

    Args:
        state: Mutable stream state.

    Yields:
        Two ``JSONServerSentEvent`` objects.
    """
    if state.current_item_id is None:
        return  # pragma: no cover — always set before this function is called
    item_id = state.current_item_id
    yield json_sse(
        "response.output_item.added",
        ResponseOutputItemAddedEvent(
            item=ResponseOutputMessage(
                id=item_id,
                content=[],
                role="assistant",
                status="in_progress",
                type="message",
            ),
            output_index=state.output_index,
            sequence_number=state.next_seq(),
            type="response.output_item.added",
        ),
    )
    yield json_sse(
        "response.content_part.added",
        ResponseContentPartAddedEvent(
            item_id=item_id,
            output_index=state.output_index,
            content_index=0,
            part=ResponseOutputText(annotations=[], text="", type="output_text"),
            sequence_number=state.next_seq(),
            type="response.content_part.added",
        ),
    )


def _emit_text_delta(
    state: _StreamState, text_delta: str
) -> Generator[JSONServerSentEvent]:
    """Emit an ``output_text.delta`` event, synthesising a text block start if needed.

    Args:
        state: Mutable stream state.
        text_delta: New text chunk from Bedrock.

    Yields:
        SSE events (optional block-start followed by the text delta).
    """
    if state.block_kind is _BlockKind.NONE:
        # Synthesise a text block start if contentBlockStart was never received.
        state.current_text = ""
        state.current_item_id = f"{state.response_id}-msg-{state.output_index}"
        state.block_kind = _BlockKind.TEXT
        yield from _emit_text_block_start(state)
    if state.block_kind is not _BlockKind.TEXT or not state.current_item_id:
        return  # pragma: no cover
    state.current_text += text_delta
    yield json_sse(
        "response.output_text.delta",
        ResponseTextDeltaEvent(
            item_id=state.current_item_id,
            output_index=state.output_index,
            content_index=0,
            delta=text_delta,
            logprobs=[],
            sequence_number=state.next_seq(),
            type="response.output_text.delta",
        ),
    )


def _handle_block_delta(
    state: _StreamState, delta_block: ContentBlockDeltaEventTypeDef
) -> Generator[JSONServerSentEvent]:
    """Emit SSE delta events for a Bedrock ``contentBlockDelta`` event.

    Handles text, tool-use argument, and reasoning deltas.  Synthesises a
    text block start when a text delta arrives without a prior block start.
    Reasoning deltas are accumulated for later token estimation but are not
    currently surfaced as a Responses event.

    Args:
        state: Mutable stream state.
        delta_block: The ``contentBlockDelta`` event from Bedrock.

    Yields:
        ``JSONServerSentEvent`` objects for the delta.
    """
    delta = delta_block["delta"]

    if state.block_kind in (_BlockKind.SUPPRESSED, _BlockKind.WEB_SEARCH):
        if tool_delta := delta.get("toolUse"):
            state.current_args += tool_delta["input"]
        return

    if text_delta := delta.get("text"):
        yield from _emit_text_delta(state, text_delta)
    elif (
        (tool_delta := delta.get("toolUse"))
        and state.block_kind is _BlockKind.TOOL
        and state.current_item_id
    ):
        args_delta = tool_delta["input"]
        state.current_args += args_delta
        yield json_sse(
            "response.function_call_arguments.delta",
            ResponseFunctionCallArgumentsDeltaEvent(
                item_id=state.current_item_id,
                output_index=state.output_index,
                delta=args_delta,
                sequence_number=state.next_seq(),
                type="response.function_call_arguments.delta",
            ),
        )
    elif (reasoning_delta := delta.get("reasoningContent")) and (
        reasoning_text := reasoning_delta.get("text")
    ):
        state.reasoning_text += reasoning_text
    elif citation_delta := delta.get("citation"):
        _record_citation(state, citation_delta)


def _record_citation(
    state: _StreamState, citation: CitationsDeltaTypeDef | CitationTypeDef
) -> None:
    """Accumulate a single web citation against the latest web-search item.

    Bedrock streams ``citationsContent`` after the ``web_search`` tool block has
    already closed, so the citation cannot be forwarded as a delta of the now
    completed ``response.output_item.done`` event.  Instead we collect them in
    ``state.pending_web_search_sources`` keyed by the target ``item_id`` and
    patch the stored ``ResponseFunctionWebSearch`` in
    ``state.output_items`` before ``response.completed`` is emitted.

    Args:
        state: Mutable stream state.
        citation: A Bedrock citation delta or completed citation.
    """
    url = ((citation.get("location") or {}).get("web") or {}).get("url")
    target_id = next(
        (
            item.id
            for item in reversed(state.output_items)
            if isinstance(item, ResponseFunctionWebSearch)
        ),
        None,
    )
    if not url or target_id is None:
        return
    state.pending_web_search_sources.setdefault(target_id, []).append(
        WebSearchActionSource(type="url", url=url)
    )


def _emit_tool_done(state: _StreamState) -> Generator[JSONServerSentEvent]:
    """Emit ``function_call_arguments.done`` and ``output_item.done`` for a tool block.

    Args:
        state: Mutable stream state (must point at a tool block with
            ``current_item_id``/``current_tool_name``/``current_tool_id`` set).

    Yields:
        SSE events closing the tool block.
    """
    if (
        state.current_item_id is None
        or state.current_tool_name is None
        or state.current_tool_id is None
    ):  # pragma: no cover — caller guarantees these are set for TOOL blocks
        return
    yield json_sse(
        "response.function_call_arguments.done",
        ResponseFunctionCallArgumentsDoneEvent(
            item_id=state.current_item_id,
            output_index=state.output_index,
            arguments=state.current_args,
            name=state.current_tool_name,
            sequence_number=state.next_seq(),
            type="response.function_call_arguments.done",
        ),
    )
    done_tool_item = ResponseFunctionToolCall(
        arguments=state.current_args,
        call_id=state.current_tool_id,
        name=state.current_tool_name,
        type="function_call",
        id=state.current_item_id,
        status="completed",
    )
    yield json_sse(
        "response.output_item.done",
        ResponseOutputItemDoneEvent(
            item=done_tool_item,
            output_index=state.output_index,
            sequence_number=state.next_seq(),
            type="response.output_item.done",
        ),
    )
    state.output_items.append(done_tool_item)
    state.output_index += 1


def _handle_block_stop(state: _StreamState) -> Generator[JSONServerSentEvent]:
    """Emit finalisation events for a Bedrock ``contentBlockStop``.

    Text blocks emit ``output_text.done`` + ``content_part.done`` +
    ``output_item.done``; non-suppressed tool blocks emit
    ``function_call_arguments.done`` + ``output_item.done``; web-search blocks
    emit ``web_search_call.completed`` + ``output_item.done``; suppressed tool
    blocks are only recorded in ``state.suppressed_tool_calls``.

    Args:
        state: Mutable stream state.

    Yields:
        ``JSONServerSentEvent`` objects for the block stop.
    """
    if state.block_kind is _BlockKind.TEXT and state.current_item_id:
        text_part = ResponseOutputText(
            annotations=[], text=state.current_text, type="output_text"
        )
        yield json_sse(
            "response.output_text.done",
            ResponseTextDoneEvent(
                item_id=state.current_item_id,
                output_index=state.output_index,
                content_index=0,
                logprobs=[],
                text=state.current_text,
                sequence_number=state.next_seq(),
                type="response.output_text.done",
            ),
        )
        done_text_item = ResponseOutputMessage(
            id=state.current_item_id,
            content=[text_part],
            role="assistant",
            status="completed",
            type="message",
        )
        yield json_sse(
            "response.content_part.done",
            ResponseContentPartDoneEvent(
                item_id=state.current_item_id,
                output_index=state.output_index,
                content_index=0,
                part=text_part,
                sequence_number=state.next_seq(),
                type="response.content_part.done",
            ),
        )
        yield json_sse(
            "response.output_item.done",
            ResponseOutputItemDoneEvent(
                item=done_text_item,
                output_index=state.output_index,
                sequence_number=state.next_seq(),
                type="response.output_item.done",
            ),
        )
        state.output_items.append(done_text_item)
        state.output_index += 1
        state.reset_block()

    elif (
        state.block_kind
        in (_BlockKind.TOOL, _BlockKind.WEB_SEARCH, _BlockKind.SUPPRESSED)
        and state.current_item_id
        and state.current_tool_name
        and state.current_tool_id
    ):
        if state.block_kind is _BlockKind.WEB_SEARCH:
            parsed = try_parse_json(state.current_args) if state.current_args else None
            query = (
                q
                if isinstance(parsed, dict)
                and isinstance(q := parsed.get("query"), str)
                else ""
            )
            ws_item = ResponseFunctionWebSearch(
                id=state.current_item_id,
                type="web_search_call",
                status="completed",
                action=WebSearchActionSearch(
                    type="search",
                    query=query,
                    queries=[query] if query else None,
                    # citationsContent arrives after the block stops; patched in before response.completed.
                    sources=None,
                ),
            )
            yield json_sse(
                "response.web_search_call.completed",
                ResponseWebSearchCallCompletedEvent(
                    item_id=state.current_item_id,
                    output_index=state.output_index,
                    sequence_number=state.next_seq(),
                    type="response.web_search_call.completed",
                ),
            )
            yield json_sse(
                "response.output_item.done",
                ResponseOutputItemDoneEvent(
                    item=ws_item,
                    output_index=state.output_index,
                    sequence_number=state.next_seq(),
                    type="response.output_item.done",
                ),
            )
            state.output_items.append(ws_item)
            state.output_index += 1
        elif state.block_kind is _BlockKind.SUPPRESSED:
            state.suppressed_tool_calls.append(
                (state.current_tool_id, state.current_tool_name, state.current_args)
            )
        else:
            yield from _emit_tool_done(state)
    state.reset_block()


def _process_stream_event(
    state: _StreamState,
    event: ConverseStreamOutputTypeDef,
    suppress_tool_names: frozenset[str] | None,
    web_search_tool_names: frozenset[str] | None = None,
) -> Generator[JSONServerSentEvent]:
    """Dispatch a single Bedrock stream event to the appropriate handler.

    Delegates to ``_handle_block_start``, ``_handle_block_delta``, or
    ``_handle_block_stop`` for content events; updates ``state`` metadata fields
    for ``messageStop`` and ``metadata`` events.

    Args:
        state: Mutable stream state.
        event: A single Bedrock ConverseStream event.
        suppress_tool_names: Tool names whose output items should be filtered.
        web_search_tool_names: Tool names to emit as ``web_search_call`` items.

    Yields:
        ``JSONServerSentEvent`` objects produced by the event handler, if any.
    """
    match event:
        case {"contentBlockStart": start}:
            yield from _handle_block_start(
                state, start, suppress_tool_names, web_search_tool_names
            )
        case {"contentBlockDelta": delta}:
            yield from _handle_block_delta(state, delta)
        case {"contentBlockStop": _}:
            yield from _handle_block_stop(state)
        case {"messageStop": {"stopReason": stop_reason}}:
            state.stop_reason = stop_reason
        case {"metadata": {"usage": usage}}:
            state.input_tokens = usage["inputTokens"]
            state.output_tokens = usage["outputTokens"]
            state.cached_tokens = usage.get("cacheReadInputTokens", 0)


async def format_stream(
    response_id: str,
    created_at: float,
    model_id: str,
    stream: AsyncIterator[ConverseStreamOutputTypeDef],
    request: ResponseCreateParams,
    suppress_tool_names: frozenset[str] | None = None,
    post_suppress_handler: Callable[[_StreamState], AsyncGenerator[JSONServerSentEvent]]
    | None = None,
    web_search_tool_names: frozenset[str] | None = None,
) -> AsyncGenerator[JSONServerSentEvent]:
    """Stream Bedrock Converse events as OpenAI Responses API SSE events.

    Emits the full lifecycle event sequence: ``response.created`` →
    ``response.in_progress`` → per-block events → optional post-suppress items →
    ``response.completed``.

    Args:
        response_id: Unique identifier for this response.
        created_at: Unix timestamp when the response was created.
        model_id: The Bedrock model identifier.
        stream: Bedrock ConverseStream event iterator.
        request: The original Responses API creation request.
        suppress_tool_names: Tool names whose output items should be filtered.
        post_suppress_handler: Optional async generator called after the main stream
            but before ``response.completed``.  Receives the stream state and may
            append items to ``state.output_items`` and yield additional SSE events
            (e.g. for server-side image generation results).
        web_search_tool_names: Tool names to emit as ``web_search_call`` output
            items with query and completion events.

    Yields:
        ``JSONServerSentEvent`` for each Responses API stream event.
    """
    state = _StreamState(response_id)
    initial_response = _build_response_object(
        response_id, created_at, model_id, [], "in_progress", None, None, request
    )

    yield json_sse(
        "response.created",
        ResponseCreatedEvent(
            response=initial_response,
            sequence_number=state.next_seq(),
            type="response.created",
        ),
    )
    yield json_sse(
        "response.in_progress",
        ResponseInProgressEvent(
            response=initial_response,
            sequence_number=state.next_seq(),
            type="response.in_progress",
        ),
    )

    async for event in stream:
        for sse in _process_stream_event(
            state, event, suppress_tool_names, web_search_tool_names
        ):
            yield sse  # `yield from` is not permitted inside async generators.

    # Reasoning deltas are not billed by Bedrock; estimate locally to match the non-streaming path.
    reasoning_tokens = (
        (await estimate_token_count(state.reasoning_text) or 0)
        if state.reasoning_text
        else 0
    )

    if post_suppress_handler:
        async for sse in post_suppress_handler(state):
            yield sse

    # Patch accumulated citation sources into the stored output items so ``response.completed`` includes them.
    if state.pending_web_search_sources:
        for idx, item in enumerate(state.output_items):
            if isinstance(item, ResponseFunctionWebSearch) and (
                sources := state.pending_web_search_sources.get(item.id)
            ):
                state.output_items[idx] = item.model_copy(
                    update={
                        "action": item.action.model_copy(update={"sources": sources})
                    }
                )

    yield json_sse(
        "response.completed",
        ResponseCompletedEvent(
            response=log_response_params(
                _build_response_object(
                    response_id,
                    created_at,
                    model_id,
                    state.output_items,
                    *_map_stop_reason(state.stop_reason),
                    ResponseUsage(
                        input_tokens=state.input_tokens,
                        input_tokens_details=InputTokensDetails(
                            cached_tokens=state.cached_tokens
                        ),
                        output_tokens=state.output_tokens + reasoning_tokens,
                        output_tokens_details=OutputTokensDetails(
                            reasoning_tokens=reasoning_tokens
                        ),
                        total_tokens=state.input_tokens
                        + state.output_tokens
                        + reasoning_tokens,
                    ),
                    request,
                )
            ),
            sequence_number=state.next_seq(),
            type="response.completed",
        ),
    )


async def count_input_tokens_via_bedrock(
    request: InputTokenCountParams, model_id: str, region: RegionName
) -> int:
    """Count input tokens using the AWS Bedrock Runtime CountTokens API.

    Builds a Converse-compatible request from the OpenAI Responses input and
    calls Bedrock's ``count_tokens`` API for an accurate, model-specific count.

    Args:
        request: The input-token count request (model + input + tools/etc.).
        model_id: The Bedrock model identifier.
        region: The AWS region of the model.

    Returns:
        The total number of input tokens.

    Raises:
        ApiError: If the input exceeds the model's context window.
    """
    bedrock_messages, system_blocks = await map_input(
        request.input, request.instructions
    )

    req: ConverseTokensRequestTypeDef = {"messages": bedrock_messages}
    if system_blocks:
        req["system"] = system_blocks

    if request.tools and request.tool_choice != "none":
        tool_specs: list[ToolTypeDef] = [
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description or "function",
                    "inputSchema": {"json": tool.parameters or _EMPTY_TOOL},
                }
            }
            for tool in request.tools
            if isinstance(tool, FunctionTool)
        ]
        if tool_specs:
            tool_config: ToolConfigurationTypeDef = {"tools": tool_specs}
            if bedrock_choice := _map_tool_choice(request.tool_choice):
                tool_config["toolChoice"] = bedrock_choice
            req["toolConfig"] = tool_config

    with handle_bedrock_client_error():
        resp: CountTokensResponseTypeDef = await get_client(
            "bedrock-runtime", region
        ).count_tokens(modelId=model_id, input={"converse": req})
    return resp["inputTokens"]
