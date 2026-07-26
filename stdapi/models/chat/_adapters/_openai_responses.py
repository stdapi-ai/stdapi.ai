"""OpenAI Responses API adapter for Bedrock Converse.

Translates between OpenAI Responses API request/response types and
Bedrock Converse API-native types. Handles tool mapping, input mapping,
response formatting (both streaming and non-streaming), and streaming events.
"""

from base64 import b64decode, b64encode, urlsafe_b64encode
from dataclasses import dataclass, field
from enum import Enum
from time import time
from traceback import format_exception
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from botocore.exceptions import ClientError, HTTPClientError
from botocore.exceptions import ConnectionError as BotocoreConnectionError
from pydantic_core import from_json, to_json

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.aws_bedrock import (
    AWS_ERROR_MAP,
    PromptCaching,
    build_system_blocks,
    handle_bedrock_client_error,
    set_inference_configuration,
)
from stdapi.input_file import FileIdInputFile, InputFile
from stdapi.models import validate_model
from stdapi.models.chat._adapters import _openai_common
from stdapi.models.image import get_image_model
from stdapi.monitoring import (
    SseHandledStreamError,
    log_error_details,
    log_response_params,
)
from stdapi.types.openai import ResponseFormatJSONObject, ResponseFormatText
from stdapi.types.openai_responses import (
    AnnotationURLCitation,
    CodeInterpreter,
    CompactionItemParam,
    ContentPartReasoningText,
    CustomToolCallInput,
    CustomToolCallOutput,
    EasyInputMessage,
    FunctionCallInput,
    FunctionCallOutput,
    FunctionTool,
    ImageGeneration,
    ImageGenerationCall,
    ImageGenerationCallInput,
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
    ResponseError,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFormatTextJSONSchemaConfig,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseImageGenCallCompletedEvent,
    ResponseImageGenCallGeneratingEvent,
    ResponseImageGenCallInProgressEvent,
    ResponseIncompleteEvent,
    ResponseInProgressEvent,
    ResponseInputFile,
    ResponseInputImage,
    ResponseInputText,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseOutputTextContent,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    Tool,
    ToolChoiceAllowed,
    ToolChoiceFunction,
    WebSearchActionSearch,
    WebSearchActionSource,
    WebSearchPreviewTool,
    WebSearchTool,
)
from stdapi.utils import hide_security_details, json_sse, try_parse_json

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Callable,
        Generator,
        Iterable,
        Mapping,
    )

    from sse_starlette import JSONServerSentEvent
    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime.literals import (
        CacheTTLType,
        ImageFormatType,
        ServiceTierTypeType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        CitationOutputTypeDef,
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
        ReasoningContentBlockDeltaTypeDef,
        ReasoningContentBlockOutputTypeDef,
        ReasoningTextBlockTypeDef,
        SystemContentBlockTypeDef,
        ToolChoiceTypeDef,
        ToolConfigurationTypeDef,
        ToolResultContentBlockUnionTypeDef,
        ToolTypeDef,
        ToolUseBlockOutputTypeDef,
        ToolUseBlockTypeDef,
    )

    from stdapi.config import LogLevel
    from stdapi.models.chat import ReasoningParams
    from stdapi.types import JsonMapping
    from stdapi.types.anthropic_messages import ServerTools
    from stdapi.types.openai import ResponseModeration
    from stdapi.types.openai_responses import (
        Annotation,
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

#: Prompt-caching scopes enabled for this request.
type PromptCachingScopes = frozenset[PromptCaching]

#: Status values produced for a generated ``ImageGenerationCall`` item.
type _ImageStatus = Literal["completed", "failed"]


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

#: Magic-byte prefixes used to sniff the format of echoed generated images.
_IMAGE_MAGIC_FORMATS: tuple[tuple[bytes, ImageFormatType], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF8", "gif"),
    (b"RIFF", "webp"),
)


def _map_tool_choice(tool_choice: ToolChoice | None) -> ToolChoiceTypeDef | None:
    """Convert a Responses API tool_choice to a Bedrock ToolChoiceTypeDef.

    Args:
        tool_choice: OpenAI Responses tool choice (literal, function, or types).

    Returns:
        Bedrock toolChoice, or ``None`` when no constraint applies (``"none"`` is
        handled upstream by omitting the tool configuration altogether; built-in
        ``ToolChoiceTypes`` variants like ``file_search`` are not natively
        supported by Bedrock and are therefore ignored).  ``allowed_tools`` is
        approximated: ``required`` with exactly one allowed function tool forces
        that tool, ``required`` with several forces any tool, ``auto`` maps to
        Bedrock auto.

    Raises:
        ApiError: If ``tool_choice`` is an unknown string literal.
    """
    match tool_choice:
        case None:
            return None
        case ToolChoiceAllowed(mode="required", tools=tools):
            names = [
                name
                for tool in tools
                if tool.get("type") == "function"
                and isinstance(name := tool.get("name"), str)
            ]
            return {"tool": {"name": names[0]}} if len(names) == 1 else {"any": {}}
        case "auto" | "none" | ToolChoiceAllowed():
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


def _function_tool_spec(tool: FunctionTool) -> ToolTypeDef:
    """Build a Bedrock ``toolSpec`` dict from an OpenAI ``FunctionTool``.

    Args:
        tool: The function tool definition.

    Returns:
        Bedrock ``ToolTypeDef`` with name, description, and input schema.
    """
    return {
        "toolSpec": {
            "name": tool.name,
            "description": tool.description or "function",
            "inputSchema": {"json": tool.parameters or _EMPTY_TOOL},
        }
    }


def _build_tool_config(
    request: ResponseCreateParams | InputTokenCountParams,
    tool_name_map: Mapping[ServerTools, str] | None = None,
) -> ToolConfigurationTypeDef | None:
    """Build a Bedrock tool configuration from a Responses API request.

    Maps ``FunctionTool`` entries to Bedrock toolSpec and OpenAI integrated
    tool types (code_interpreter, web_search, image_generation) to their
    Bedrock equivalents.  Tool types without a backend equivalent are
    accepted for compatibility and dropped.

    When ``tool_choice="none"``, no tool config is returned so that the model
    cannot call any tools.

    Args:
        request: Responses API creation or input-token count request.
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
            tools.append(_function_tool_spec(tool))
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
                        "inputSchema": {"json": _EMPTY_TOOL},
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

    Used by :mod:`stdapi.models.chat._default` to intercept ``image_generation``
    tool calls.

    Args:
        request: Responses API creation request.

    Returns:
        The first ``ImageGeneration`` tool found, or ``None`` if absent.
    """
    return next(
        (t for t in request.tools or () if isinstance(t, ImageGeneration)), None
    )


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
    try:
        width, height = map(int, ("1024x1024" if size == "auto" else size).split("x"))
    except ValueError:
        width, height = 1024, 1024
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


async def _try_generate_image_b64(
    args: dict[str, str], tool: ImageGeneration, fallback_model: str | None
) -> str | None:
    """Run image generation, logging any failure instead of raising.

    Args:
        args: String-valued tool-call arguments.
        tool: The ``ImageGeneration`` tool definition from the request.
        fallback_model: Operator-configured default image model ID, or ``None``.

    Returns:
        Base64-encoded image on success, ``None`` when generation failed.
    """
    try:
        return await _generate_image_b64(args, tool, fallback_model)
    except Exception as exc:  # noqa: BLE001
        log_error_details(f"image_generation tool call failed: {exc}", level="warning")
        return None


async def execute_image_generation_calls(
    output_items: list[ResponseOutputItem],
    image_gen_tool: ImageGeneration,
    response_id: str,
    fallback_model: str | None,
) -> list[ResponseOutputItem]:
    """Replace ``image_generation`` function-call items with ``ImageGenerationCall``.

    All other items pass through unchanged. Executes synchronously; generation
    errors are caught and produce a ``status="failed"`` item instead of aborting
    the request.

    Args:
        output_items: Output items from the Bedrock response.
        image_gen_tool: The ``ImageGeneration`` tool definition from the request.
        response_id: Response ID used to generate stable ``ImageGenerationCall`` IDs.
        fallback_model: Operator-configured default image model ID, or ``None``.

    Returns:
        New output list with image-generation calls materialised.
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
        b64 = await _try_generate_image_b64(
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
    yields ``response.output_item.added`` +
    ``response.image_generation_call.in_progress`` +
    ``response.image_generation_call.generating``, runs generation, appends the
    resulting ``ImageGenerationCall`` to ``state.output_items``, and yields
    ``response.image_generation_call.completed`` (successful generations only)
    + ``response.output_item.done``. Generation failures are logged and produce
    a ``status="failed"`` item so a single failed image does not abort the
    stream.

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

        counter += 1
        item_id = f"{response_id}-img-{counter}"
        yield json_sse(
            "response.output_item.added",
            ResponseOutputItemAddedEvent(
                item=ImageGenerationCall(
                    id=item_id,
                    status="in_progress",
                    type="image_generation_call",
                    result=None,
                ),
                output_index=state.output_index,
                sequence_number=state.next_seq(),
                type="response.output_item.added",
            ),
        )
        yield json_sse(
            "response.image_generation_call.in_progress",
            ResponseImageGenCallInProgressEvent(
                item_id=item_id,
                output_index=state.output_index,
                sequence_number=state.next_seq(),
                type="response.image_generation_call.in_progress",
            ),
        )
        yield json_sse(
            "response.image_generation_call.generating",
            ResponseImageGenCallGeneratingEvent(
                item_id=item_id,
                output_index=state.output_index,
                sequence_number=state.next_seq(),
                type="response.image_generation_call.generating",
            ),
        )
        b64 = await _try_generate_image_b64(
            _str_args(args_json), image_gen_tool, fallback_model
        )
        status: _ImageStatus = "completed" if b64 else "failed"
        image_call = ImageGenerationCall(
            id=item_id, status=status, type="image_generation_call", result=b64
        )
        state.output_items.append(image_call)
        if status == "completed":
            yield json_sse(
                "response.image_generation_call.completed",
                ResponseImageGenCallCompletedEvent(
                    item_id=item_id,
                    output_index=state.output_index,
                    sequence_number=state.next_seq(),
                    type="response.image_generation_call.completed",
                ),
            )
        yield json_sse(
            "response.output_item.done",
            ResponseOutputItemDoneEvent(
                item=image_call,
                output_index=state.output_index,
                sequence_number=state.next_seq(),
                type="response.output_item.done",
            ),
        )
        state.output_index += 1


def _build_output_config(
    text: ResponseTextConfig | None,
) -> JsonSchemaDefinitionTypeDef | None:
    """Convert a Responses API ``text.format`` to a Bedrock ``outputConfig`` schema.

    Uses Bedrock's native structured output (``outputConfig.textFormat``); models
    that do not support it surface a Bedrock-side error.

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
    PromptCachingScopes,
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
        frozenset(_openai_common.parse_prompt_cache_key(request.prompt_cache_key)),
        (
            _openai_common.CACHE_TTL.get(request.prompt_cache_retention)
            if request.prompt_cache_retention
            else None
        ),
        (dict(request.metadata) if request.metadata else None),
    )


def extract_reasoning(request: ResponseCreateParams) -> ReasoningParams | None:
    """Extract reasoning parameters from an OpenAI Responses request.

    Args:
        request: Responses API creation request.

    Returns:
        Reasoning parameters to configure, or None if the request has no
        ``reasoning`` field set.  A ``reasoning`` object without ``effort``
        enables reasoning at the upstream default ``medium`` effort; only
        ``effort="none"`` disables it.
    """
    if request.reasoning is None:
        return None
    effort = request.reasoning.effort or "medium"
    return {
        "enabled": effort != "none",
        "reasoning_effort": effort,
        "budget_tokens": None,
        "max_tokens": request.max_output_tokens,
    }


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
            else " ".join(
                p.text
                for p in content
                if isinstance(p, (ResponseInputText, ResponseOutputTextContent))
            )
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


def _map_output_message(
    item: ResponseOutputMessage, bedrock_messages: list[MessageTypeDef]
) -> None:
    """Map a ResponseOutputMessage (echoed assistant output) as a Bedrock assistant message.

    When a client echoes back the full previous API response in the input array
    (as done by Codex CLI), assistant messages arrive as ``ResponseOutputMessage``
    items with ``role="assistant"`` and content blocks of type ``output_text`` or
    ``refusal``.  Both are mapped to plain Bedrock text blocks so that a refusal
    turn is not lost from the conversation history.

    Args:
        item: The output message to map.
        bedrock_messages: Mutable Bedrock messages list to append to.
    """
    _append_or_merge(
        bedrock_messages,
        "assistant",
        [
            {
                "text": part.text
                if isinstance(part, ResponseOutputText)
                else part.refusal
            }
            for part in item.content
        ],
    )


def _tool_use_input(arguments: str) -> JsonMapping:
    """Parse raw tool-call arguments into the JSON object Bedrock expects.

    Args:
        arguments: Raw arguments string (JSON or freeform text).

    Returns:
        The parsed JSON object, or ``{"input": arguments}`` when the string is
        not a JSON object.
    """
    parsed = try_parse_json(arguments)
    return parsed if isinstance(parsed, dict) else {"input": arguments}


def _map_function_call(
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
        "input": _tool_use_input(item.arguments) if item.arguments else {},
    }
    _append_or_merge(bedrock_messages, "assistant", [{"toolUse": tool_input}])


async def _map_function_call_output(
    item: FunctionCallOutput, bedrock_messages: list[MessageTypeDef]
) -> None:
    """Map a function_call_output item as a Bedrock toolResult user message.

    String outputs become text/json blocks; content-part lists map text parts to
    text/json blocks and image/file parts to Bedrock toolResult image, document,
    or video blocks (resolved through the input-file machinery).  Merges with
    the previous user message when possible.

    Args:
        item: The function call output item.
        bedrock_messages: Mutable Bedrock messages list to append to.

    Raises:
        ApiError: If a content part cannot be converted to a Bedrock block.
    """
    output = item.output
    if isinstance(output, str):
        tool_content: list[ToolResultContentBlockUnionTypeDef] = [
            _openai_common.parse_tool_content(output)
        ]
    else:
        tool_content = []
        for part in output:
            if isinstance(part, (ResponseInputText, ResponseOutputTextContent)):
                tool_content.append(_openai_common.parse_tool_content(part.text))
            else:
                # Image/document/video blocks share their shape with toolResult
                # content blocks.
                tool_content.append(
                    cast(
                        "ToolResultContentBlockUnionTypeDef",
                        await _convert_input_content(part),
                    )
                )
    _append_or_merge(
        bedrock_messages,
        "user",
        [{"toolResult": {"toolUseId": item.call_id, "content": tool_content}}],
    )


def _map_custom_tool_call(
    item: CustomToolCallInput, bedrock_messages: list[MessageTypeDef]
) -> None:
    """Map a custom_tool_call input item as a Bedrock toolUse assistant message.

    The freeform ``input`` string is parsed as a JSON object when possible and
    wrapped as ``{"input": <raw>}`` otherwise, since Bedrock requires a JSON
    object for ``toolUse.input``.

    Args:
        item: The custom tool call input item.
        bedrock_messages: Mutable Bedrock messages list to append to.
    """
    tool_use: ToolUseBlockTypeDef = {
        "toolUseId": item.call_id,
        "name": item.name,
        "input": _tool_use_input(item.input) if item.input else {},
    }
    _append_or_merge(bedrock_messages, "assistant", [{"toolUse": tool_use}])


def _map_custom_tool_call_output(
    item: CustomToolCallOutput, bedrock_messages: list[MessageTypeDef]
) -> None:
    """Map a custom_tool_call_output item as a Bedrock toolResult user message.

    Custom tool outputs are freeform text, so string outputs become a single
    text block and content-part lists keep their text parts as text blocks.

    Args:
        item: The custom tool call output item.
        bedrock_messages: Mutable Bedrock messages list to append to.
    """
    output = item.output
    tool_content: list[ToolResultContentBlockUnionTypeDef] = (
        [{"text": output}]
        if isinstance(output, str)
        else [
            {"text": part.text}
            for part in output
            if isinstance(part, (ResponseInputText, ResponseOutputTextContent))
        ]
    )
    _append_or_merge(
        bedrock_messages,
        "user",
        [{"toolResult": {"toolUseId": item.call_id, "content": tool_content}}],
    )


def _sniff_image_format(data: bytes) -> ImageFormatType:
    """Detect a Bedrock image format from magic bytes.

    Args:
        data: Raw image bytes.

    Returns:
        Detected Bedrock image format, defaulting to ``png``.
    """
    for magic, image_format in _IMAGE_MAGIC_FORMATS:
        if data.startswith(magic):
            return image_format
    return "png"


def _map_image_generation_call(
    item: ImageGenerationCallInput, bedrock_messages: list[MessageTypeDef]
) -> None:
    """Map an echoed image_generation_call item back into the conversation.

    The gateway emits ``image_generation_call`` output items for the synthetic
    ``image_generation`` tool; when echoed back the call is replayed as an
    assistant ``toolUse`` followed by a user ``toolResult`` carrying the
    generated image.  Items without a result are dropped.

    Args:
        item: The image generation call input item.
        bedrock_messages: Mutable Bedrock messages list to append to.

    Raises:
        ApiError: If the result is not valid base64 content.
    """
    if not item.result:
        return
    try:
        image = b64decode(item.result, validate=True)
    except ValueError as exc:
        msg = "Invalid image_generation_call result content."
        raise ApiError(msg) from exc
    tool_use: ToolUseBlockTypeDef = {
        "toolUseId": item.id,
        "name": "image_generation",
        "input": {},
    }
    _append_or_merge(bedrock_messages, "assistant", [{"toolUse": tool_use}])
    tool_content: list[ToolResultContentBlockUnionTypeDef] = [
        {"image": {"format": _sniff_image_format(image), "source": {"bytes": image}}}
    ]
    _append_or_merge(
        bedrock_messages,
        "user",
        [{"toolResult": {"toolUseId": item.id, "content": tool_content}}],
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
        await _map_input_item(item, bedrock_messages, system_blocks)

    return bedrock_messages, system_blocks


async def _map_input_item(
    item: ResponseInputItem,
    bedrock_messages: list[MessageTypeDef],
    system_blocks: list[SystemContentBlockTypeDef],
) -> None:
    """Map a single Responses input item to Bedrock messages or system blocks.

    Unsupported item types (e.g. hosted-tool calls such as ``web_search_call``
    or ``item_reference`` entries) are accepted and silently dropped.

    Args:
        item: The input item to map.
        bedrock_messages: Mutable Bedrock messages list to append to.
        system_blocks: Mutable system blocks list to append to.
    """
    match item:
        case EasyInputMessage() | InputMessage():
            await _map_message_item(item, bedrock_messages, system_blocks)
        case ResponseOutputMessage():
            _map_output_message(item, bedrock_messages)
        case FunctionCallInput():
            _map_function_call(item, bedrock_messages)
        case FunctionCallOutput():
            await _map_function_call_output(item, bedrock_messages)
        case CustomToolCallInput():
            _map_custom_tool_call(item, bedrock_messages)
        case CustomToolCallOutput():
            _map_custom_tool_call_output(item, bedrock_messages)
        case ImageGenerationCallInput():
            _map_image_generation_call(item, bedrock_messages)
        case ResponseReasoningItem():
            _map_reasoning_item(item, bedrock_messages)
        case CompactionItemParam():
            _map_compaction_item(item, bedrock_messages)


#: Marker identifying locally-encoded compaction content; ":" is outside the base64url alphabet, so upstream ciphertext can never collide with it.
COMPACTION_CONTENT_PREFIX = "v1:"


def encode_compaction_content(summary: str) -> str:
    """Encode a conversation summary as opaque compaction item content.

    The content is self-contained so that compaction round-trips work without
    any server-side state; it is marker-prefixed and encoded, not encrypted.

    Args:
        summary: Conversation summary text.

    Returns:
        Opaque content for a ``compaction`` item.
    """
    return f"{COMPACTION_CONTENT_PREFIX}{urlsafe_b64encode(summary.encode()).decode()}"


def _map_compaction_item(
    item: CompactionItemParam, bedrock_messages: list[MessageTypeDef]
) -> None:
    """Map a ``compaction`` input item back to a user message with its summary.

    Args:
        item: The compaction item produced by POST /v1/responses/compact.
        bedrock_messages: Mutable Bedrock messages list to append to.

    Raises:
        ApiError: When the content lacks the local marker (e.g. an item
            produced by the upstream OpenAI API) or cannot be decoded.
    """
    msg = (
        "Invalid compaction item content: only compaction items produced "
        "by this server can be expanded."
    )
    encoded = item.encrypted_content.removeprefix(COMPACTION_CONTENT_PREFIX)
    if encoded == item.encrypted_content:
        raise ApiError(msg)
    try:
        summary = b64decode(encoded, altchars=b"-_", validate=True).decode()
    except (ValueError, UnicodeDecodeError) as exc:
        raise ApiError(msg) from exc
    _append_or_merge(
        bedrock_messages,
        "user",
        [{"text": f"Summary of the earlier conversation:\n{summary}"}],
    )


def encode_reasoning_content(signatures: list[str], redacted: list[bytes]) -> str:
    """Encode reasoning signatures and redacted payloads as opaque content.

    The content is self-contained so that reasoning round-trips work without
    any server-side state; it is encoded, not encrypted (mirrors
    ``encode_compaction_content``).

    Args:
        signatures: Bedrock ``reasoningText`` signatures, in block order.
        redacted: Bedrock ``redactedContent`` payloads, in block order.

    Returns:
        Opaque ``encrypted_content`` value for a reasoning item.
    """
    payload = {
        "signatures": signatures,
        "redacted": [b64encode(data).decode() for data in redacted],
    }
    return urlsafe_b64encode(to_json(payload)).decode()


def decode_reasoning_content(
    encrypted_content: str,
) -> tuple[list[str], list[bytes]] | None:
    """Decode an ``encrypted_content`` envelope produced by this gateway.

    Args:
        encrypted_content: Opaque reasoning content from an echoed item.

    Returns:
        Tuple of ``(signatures, redacted)``, or ``None`` when the content is
        not a valid local envelope (e.g. OpenAI-encrypted content).
    """
    try:
        payload = from_json(b64decode(encrypted_content, altchars=b"-_", validate=True))
        signatures = payload["signatures"]
        redacted = [b64decode(data, validate=True) for data in payload["redacted"]]
    except ValueError, TypeError, KeyError:
        return None
    if not (
        isinstance(signatures, list)
        and all(isinstance(signature, str) for signature in signatures)
    ):
        return None
    return signatures, redacted


def _map_reasoning_item(
    item: ResponseReasoningItem, bedrock_messages: list[MessageTypeDef]
) -> None:
    """Map an echoed ``ResponseReasoningItem`` to Bedrock ``reasoningContent`` blocks.

    Clients (e.g. Codex CLI) echo reasoning items back as part of the input
    history.  Non-empty ``content`` entries (or, failing that, ``summary``
    entries) are converted to Bedrock ``reasoningText`` blocks and merged into
    the current assistant message.  A local ``encrypted_content`` envelope
    re-attaches signatures and appends ``redactedContent`` blocks; foreign
    envelopes are ignored.  Signatures are computed over ``content`` blocks, so
    they are never attached to summary fallback texts.  Empty items are dropped
    without logging.

    Args:
        item: The reasoning item to map.
        bedrock_messages: Mutable Bedrock messages list to append to.
    """
    texts = [
        part.text
        for part in item.content or ()
        if isinstance(part, ReasoningItemContent) and part.text
    ]
    from_summary = not texts
    if from_summary:
        texts = [part.text for part in item.summary if part.text]

    signatures: list[str] = []
    redacted: list[bytes] = []
    if item.encrypted_content and (
        decoded := decode_reasoning_content(item.encrypted_content)
    ):
        signatures, redacted = decoded
    if from_summary:
        signatures = []

    blocks: list[ContentBlockTypeDef] = []
    for index, text in enumerate(texts):
        reasoning_text: ReasoningTextBlockTypeDef = {"text": text}
        if index < len(signatures):
            reasoning_text["signature"] = signatures[index]
        blocks.append({"reasoningContent": {"reasoningText": reasoning_text}})
    blocks.extend({"reasoningContent": {"redactedContent": data}} for data in redacted)
    _append_or_merge(bedrock_messages, "assistant", blocks)


def _citation_sources(
    citations: Iterable[CitationOutputTypeDef],
) -> Generator[WebSearchActionSource]:
    """Build web-search sources from Bedrock citations.

    When ``nova_grounding`` is used, Bedrock embeds citation URLs in
    ``citationsContent`` blocks.  The URLs are collected as
    ``WebSearchActionSource`` objects suitable for populating the
    ``action.sources`` field of a ``ResponseFunctionWebSearch`` item.

    Args:
        citations: Bedrock citation objects.

    Yields:
        One source object per citation carrying a web URL.
    """
    for citation in citations:
        if (web := citation.get("location", {}).get("web")) and (url := web.get("url")):
            yield WebSearchActionSource(type="url", url=url)


def _build_reasoning_item(
    item_id: str,
    text: str,
    signatures: list[str],
    redacted: list[bytes],
    *,
    include_encrypted_reasoning: bool,
) -> ResponseReasoningItem:
    """Build a completed ``reasoning`` output item from accumulated block data.

    Args:
        item_id: Identifier for the reasoning item.
        text: Concatenated reasoning text (may be empty for redacted-only).
        signatures: Bedrock ``reasoningText`` signatures, in block order.
        redacted: Bedrock ``redactedContent`` payloads, in block order.
        include_encrypted_reasoning: Whether to attach the round-trip envelope.

    Returns:
        Completed reasoning item with an empty ``summary``.
    """
    return ResponseReasoningItem(
        id=item_id,
        summary=[],
        type="reasoning",
        content=(
            [ReasoningItemContent(text=text, type="reasoning_text")] if text else []
        ),
        encrypted_content=(
            encode_reasoning_content(signatures, redacted)
            if include_encrypted_reasoning and (signatures or redacted)
            else None
        ),
        status="completed",
    )


def _accumulate_reasoning_block(
    reasoning_content: ReasoningContentBlockOutputTypeDef,
    texts: list[str],
    signatures: list[str],
    redacted: list[bytes],
) -> None:
    """Accumulate one Bedrock ``reasoningContent`` block into run buffers.

    Args:
        reasoning_content: The ``reasoningContent`` payload of a block.
        texts: Mutable list receiving ``reasoningText`` texts.
        signatures: Mutable list receiving ``reasoningText`` signatures.
        redacted: Mutable list receiving ``redactedContent`` payloads.
    """
    if reasoning_text := reasoning_content.get("reasoningText"):
        texts.append(reasoning_text["text"])
        if signature := reasoning_text.get("signature"):
            signatures.append(signature)
    if (data := reasoning_content.get("redactedContent")) is not None:
        redacted.append(data)


def _citation_annotations(
    citations: Iterable[CitationOutputTypeDef], offset: int
) -> Generator[AnnotationURLCitation]:
    """Build ``url_citation`` annotations from Bedrock citations.

    Args:
        citations: Bedrock citation objects.
        offset: Character index anchoring the citations (start and end index).

    Yields:
        One ``AnnotationURLCitation`` per citation carrying a web URL.
    """
    for citation in citations:
        if (web := citation.get("location", {}).get("web")) and (url := web.get("url")):
            yield AnnotationURLCitation(
                start_index=offset,
                end_index=offset,
                title=citation.get("title") or url,
                type="url_citation",
                url=url,
            )


def _tool_use_output_item(
    tool_use: ToolUseBlockOutputTypeDef,
    response_id: str,
    suppress_tool_names: frozenset[str] | None,
    web_search_tool_names: frozenset[str] | None,
) -> ResponseOutputItem | None:
    """Map a Bedrock ``toolUse`` block to an output item.

    Args:
        tool_use: The Bedrock ``toolUse`` block.
        response_id: The response identifier for generating item IDs.
        suppress_tool_names: Tool names whose items should be filtered out.
        web_search_tool_names: Tool names emitted as ``web_search_call`` items.

    Returns:
        ``web_search_call`` or ``function_call`` item, or ``None`` when the
        tool is suppressed.  Web-search sources are attributed afterwards by
        ``_attach_web_search_sources``.
    """
    name: str = tool_use["name"]
    if web_search_tool_names and name in web_search_tool_names:
        input_data = tool_use["input"]
        query = input_data.get("query", "") if isinstance(input_data, dict) else ""
        return ResponseFunctionWebSearch(
            id=f"{response_id}-ws-{tool_use['toolUseId']}",
            type="web_search_call",
            status="completed",
            action=WebSearchActionSearch(
                type="search",
                query=query,
                queries=[query] if query else None,
                sources=None,
            ),
        )
    if suppress_tool_names is None or name not in suppress_tool_names:
        return ResponseFunctionToolCall(
            arguments=to_json(tool_use["input"]).decode(),
            call_id=tool_use["toolUseId"],
            name=name,
            type="function_call",
            id=f"{response_id}-fc-{tool_use['toolUseId']}",
            status="completed",
        )
    return None


def _reasoning_block_item(
    reasoning_content: ReasoningContentBlockOutputTypeDef,
    item_id: str,
    *,
    include_encrypted_reasoning: bool,
) -> ResponseReasoningItem | None:
    """Build a ``reasoning`` item from a single Bedrock ``reasoningContent`` block.

    Args:
        reasoning_content: The ``reasoningContent`` payload of a block.
        item_id: Identifier for the reasoning item.
        include_encrypted_reasoning: Whether to attach the round-trip envelope.

    Returns:
        Completed reasoning item, or ``None`` when the block is empty.
    """
    texts: list[str] = []
    signatures: list[str] = []
    redacted: list[bytes] = []
    _accumulate_reasoning_block(reasoning_content, texts, signatures, redacted)
    if not (texts or signatures or redacted):
        return None
    return _build_reasoning_item(
        item_id,
        "\n".join(texts),
        signatures,
        redacted,
        include_encrypted_reasoning=include_encrypted_reasoning,
    )


def _attach_web_search_sources(
    output_items: list[ResponseOutputItem],
    ws_sources: dict[str | None, list[WebSearchActionSource]],
) -> None:
    """Attach collected web-search sources to their ``web_search_call`` items.

    Args:
        output_items: Mutable output items list to patch in place.
        ws_sources: Sources keyed by ``web_search_call`` item id; the ``None``
            key holds sources seen before any call, attributed to the first one.
    """
    first_ws_id = next(
        (
            item.id
            for item in output_items
            if isinstance(item, ResponseFunctionWebSearch)
        ),
        None,
    )
    if (deferred := ws_sources.pop(None, None)) and first_ws_id is not None:
        ws_sources[first_ws_id] = [*deferred, *ws_sources.get(first_ws_id, [])]
    for idx, item in enumerate(output_items):
        if isinstance(item, ResponseFunctionWebSearch) and (
            sources := ws_sources.get(item.id)
        ):
            output_items[idx] = item.model_copy(
                update={"action": item.action.model_copy(update={"sources": sources})}
            )


def _flush_message_item(
    output_items: list[ResponseOutputItem],
    text_parts: list[str],
    annotations: list[Annotation],
    response_id: str,
) -> None:
    """Append a message item for a pending contiguous text run, if any.

    Consumes *text_parts* and *annotations* (both are cleared); the message id
    is derived from the item's output index, matching the streaming path.

    Args:
        output_items: Mutable output items list to append to.
        text_parts: Text blocks of the run, joined with newlines.
        annotations: ``url_citation`` annotations collected for the run.
        response_id: The response identifier for generating item IDs.
    """
    if not text_parts:
        return
    output_items.append(
        ResponseOutputMessage(
            id=f"{response_id}-msg-{len(output_items)}",
            content=[
                ResponseOutputText(
                    annotations=list(annotations),
                    text="\n".join(text_parts),
                    type="output_text",
                )
            ],
            role="assistant",
            status="completed",
            type="message",
        )
    )
    text_parts.clear()
    annotations.clear()


def _record_block_citations(
    citations: Iterable[CitationOutputTypeDef],
    text_parts: list[str],
    annotations: list[Annotation],
    pending_annotations: list[Annotation],
    ws_sources: dict[str | None, list[WebSearchActionSource]],
    last_ws_id: str | None,
    *,
    collect_sources: bool,
) -> None:
    """Record one ``citationsContent`` block's annotations and sources.

    Annotations attach to the open text run (or stay pending when none is
    open); sources are attributed to the nearest preceding ``web_search_call``
    (``None`` key when none precedes).

    Args:
        citations: Bedrock citation objects of the block.
        text_parts: Text blocks of the currently open run.
        annotations: Mutable annotation list of the open run.
        pending_annotations: Mutable list for annotations outside a run.
        ws_sources: Mutable web-search source accumulator.
        last_ws_id: Id of the nearest preceding ``web_search_call`` item.
        collect_sources: Whether web-search sources should be collected.
    """
    citations = list(citations)
    # Approximation: Bedrock gives no character indices, so anchor the
    # citation at the end of the text accumulated so far.
    new_annotations = _citation_annotations(citations, len("\n".join(text_parts)))
    (annotations if text_parts else pending_annotations).extend(new_annotations)
    if collect_sources and (sources := list(_citation_sources(citations))):
        ws_sources.setdefault(last_ws_id, []).extend(sources)


def _patch_message_annotations(
    output_items: list[ResponseOutputItem], pending_annotations: list[Annotation]
) -> None:
    """Fold annotations recorded outside a text run into the last message item.

    Args:
        output_items: Mutable output items list to patch in place.
        pending_annotations: Annotations to append to the last message.
    """
    if not pending_annotations:
        return
    for idx in range(len(output_items) - 1, -1, -1):
        item = output_items[idx]
        if (
            isinstance(item, ResponseOutputMessage)
            and item.content
            and isinstance(part := item.content[0], ResponseOutputText)
        ):
            new_part = part.model_copy(
                update={"annotations": [*part.annotations, *pending_annotations]}
            )
            output_items[idx] = item.model_copy(
                update={"content": [new_part, *item.content[1:]]}
            )
            return


def _extract_output_items(
    contents: list[ContentBlockOutputTypeDef],
    response_id: str,
    suppress_tool_names: frozenset[str] | None,
    web_search_tool_names: frozenset[str] | None = None,
    *,
    include_encrypted_reasoning: bool = False,
) -> list[ResponseOutputItem]:
    """Extract ResponseOutputItem objects from Bedrock response content.

    Processes content blocks in their original Bedrock order, mirroring the
    streaming path: each ``reasoningContent`` block becomes its own
    ``reasoning`` item, each contiguous run of text blocks becomes a
    ``message`` item at its block position, and ``toolUse`` blocks become
    ``web_search_call`` or ``function_call`` items.

    Args:
        contents: Bedrock response content blocks.
        response_id: The response identifier for generating item IDs.
        suppress_tool_names: Tool names whose function_call items should be
            filtered from output (e.g. ``{"nova_code_interpreter"}``).
        web_search_tool_names: Tool names that should be emitted as
            ``web_search_call`` output items instead of being suppressed
            (e.g. ``{"nova_grounding"}``).  Sources are populated from
            ``citationsContent`` blocks in the response.
        include_encrypted_reasoning: Whether the request's ``include`` asked
            for ``reasoning.encrypted_content`` (adds the round-trip envelope
            to reasoning items).

    Returns:
        List of output items in Bedrock block order.  ``message`` items carry
        ``url_citation`` annotations from the ``citationsContent`` blocks of
        their text run (indices approximated to the accumulated text length at
        citation position); web-search sources are attributed to the nearest
        preceding ``web_search_call`` item (the first one when none precedes),
        matching the streaming path.
    """
    output_items: list[ResponseOutputItem] = []
    text_parts: list[str] = []
    annotations: list[Annotation] = []
    pending_annotations: list[Annotation] = []
    reasoning_count = 0
    last_ws_id: str | None = None
    ws_sources: dict[str | None, list[WebSearchActionSource]] = {}

    for block in contents:
        if reasoning_content := block.get("reasoningContent"):
            _flush_message_item(output_items, text_parts, annotations, response_id)
            if reasoning_item := _reasoning_block_item(
                reasoning_content,
                f"{response_id}-rs-{reasoning_count}",
                include_encrypted_reasoning=include_encrypted_reasoning,
            ):
                output_items.append(reasoning_item)
                reasoning_count += 1
        elif tool_use := block.get("toolUse"):
            _flush_message_item(output_items, text_parts, annotations, response_id)
            if tool_item := _tool_use_output_item(
                tool_use, response_id, suppress_tool_names, web_search_tool_names
            ):
                output_items.append(tool_item)
                if isinstance(tool_item, ResponseFunctionWebSearch):
                    last_ws_id = tool_item.id
        elif text := block.get("text"):
            text_parts.append(text)
        elif citations_block := block.get("citationsContent"):
            _record_block_citations(
                citations_block.get("citations", ()),
                text_parts,
                annotations,
                pending_annotations,
                ws_sources,
                last_ws_id,
                collect_sources=bool(web_search_tool_names),
            )

    _flush_message_item(output_items, text_parts, annotations, response_id)
    _patch_message_annotations(output_items, pending_annotations)
    if ws_sources:
        _attach_web_search_sources(output_items, ws_sources)
    return output_items


#: Maps Bedrock stop reasons; unknown reasons default to ``"max_output_tokens"``.
_INCOMPLETE_REASONS: dict[str, Literal["max_output_tokens", "content_filter"]] = {
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
    "max_tokens": "max_output_tokens",
}


def _map_stop_reason(
    stop_reason: str | None,
) -> tuple[
    Literal["completed", "incomplete", "failed"],
    IncompleteDetails | None,
    ResponseError | None,
]:
    """Map a Bedrock stop reason to a Responses API status and error details.

    Args:
        stop_reason: Bedrock stop reason value.

    Returns:
        Tuple of ``(status, incomplete_details, error)`` where ``status`` is one
        of ``"completed"``, ``"incomplete"``, or ``"failed"``,
        ``incomplete_details`` is populated when the response was cut short, and
        ``error`` is populated when the response failed.
    """
    match stop_reason:
        case "malformed_model_output" | "malformed_tool_use":
            return (
                "failed",
                None,
                ResponseError(
                    code="server_error",
                    message="The model failed to generate a valid response "
                    f"(Bedrock stop reason: {stop_reason}).",
                ),
            )
        case "end_turn" | "stop_sequence" | "tool_use" | None:
            return "completed", None, None
        case _:
            if stop_reason not in _INCOMPLETE_REASONS:
                log_error_details(
                    f"Unknown Bedrock stopReason: {stop_reason!r}", level="warning"
                )
            return (
                "incomplete",
                IncompleteDetails(
                    reason=_INCOMPLETE_REASONS.get(stop_reason, "max_output_tokens")
                ),
                None,
            )


def _build_response_object(
    response_id: str,
    created_at: float,
    model_id: str,
    output_items: list[ResponseOutputItem],
    status: Literal["completed", "incomplete", "failed", "in_progress"],
    incomplete_details: IncompleteDetails | None,
    error: ResponseError | None,
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
        error: Error details when the response failed, or ``None``.
        usage: Token usage statistics, or ``None`` for in-progress responses.
        request: The original Responses API creation request.

    Returns:
        Constructed Response object.
    """
    return Response(
        id=response_id,
        created_at=int(created_at),
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
        error=error,
        completed_at=int(time()) if status == "completed" else None,
        instructions=request.instructions,
        metadata=request.metadata,
        max_output_tokens=request.max_output_tokens,
        previous_response_id=request.previous_response_id,
        prompt_cache_key=request.prompt_cache_key,
        prompt_cache_retention=request.prompt_cache_retention,
        reasoning=request.reasoning,
        service_tier=request.service_tier,
        text=request.text,
        top_logprobs=request.top_logprobs,
        usage=usage,
        user=request.user,
    )


def _includes_encrypted_reasoning(request: ResponseCreateParams) -> bool:
    """Whether the request's ``include`` asks for ``reasoning.encrypted_content``.

    Args:
        request: The original Responses API creation request.

    Returns:
        True when reasoning items must carry the round-trip envelope.
    """
    return bool(request.include and "reasoning.encrypted_content" in request.include)


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

    # OpenAI semantics: input_tokens covers the full prompt, cache buckets included.
    cache_read = usage_raw.get("cacheReadInputTokens", 0)
    cache_write = usage_raw.get("cacheWriteInputTokens", 0)
    input_tokens = usage_raw["inputTokens"] + cache_read + cache_write
    output_tokens = usage_raw["outputTokens"]

    return log_response_params(
        _build_response_object(
            response_id,
            created_at,
            model_id,
            _extract_output_items(
                contents,
                response_id,
                suppress_tool_names,
                web_search_tool_names,
                include_encrypted_reasoning=_includes_encrypted_reasoning(request),
            ),
            *_map_stop_reason(bedrock_response.get("stopReason")),
            ResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=InputTokensDetails(
                    cached_tokens=cache_read, cache_write_tokens=cache_write
                ),
                output_tokens=output_tokens,
                output_tokens_details=OutputTokensDetails(),
                total_tokens=input_tokens + output_tokens,
            ),
            request,
        )
    )


class _BlockKind(Enum):
    """Kind of the content block currently being streamed."""

    NONE = "none"
    TEXT = "text"
    TOOL = "tool"
    SUPPRESSED = "suppressed"
    WEB_SEARCH = "web_search"
    REASONING = "reasoning"


@dataclass(slots=True)
class _StreamState:
    """Mutable state accumulated during a Bedrock ConverseStream response.

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
    #: Text (or reasoning-text) delta chunks of the current block, joined at close.
    current_text_parts: list[str] = field(default_factory=list)
    #: Tool-argument delta chunks of the current block, joined at close.
    current_args_parts: list[str] = field(default_factory=list)
    #: Running length of the current text block, for annotation offsets.
    current_text_len: int = 0
    block_kind: _BlockKind = _BlockKind.NONE
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    #: Tokens written to the prompt cache (separate Bedrock usage bucket).
    cache_write_tokens: int = 0
    #: Whether ``include`` requested ``reasoning.encrypted_content``.
    include_encrypted_reasoning: bool = False
    #: Signatures from reasoning deltas of the current block.
    reasoning_signatures: list[str] = field(default_factory=list)
    #: Redacted reasoning payloads from the current block.
    reasoning_redacted: list[bytes] = field(default_factory=list)
    #: Completed suppressed tool calls as ``(tool_id, tool_name, args_json)``.
    suppressed_tool_calls: list[tuple[str, str, str]] = field(default_factory=list)
    #: Web-search sources keyed by ``item_id`` (see class docstring).
    pending_web_search_sources: dict[str, list[WebSearchActionSource]] = field(
        default_factory=dict
    )
    #: ``url_citation`` annotations accumulated for the open text block.
    current_annotations: list[Annotation] = field(default_factory=list)
    #: Annotations recorded outside a text block, patched into the last message.
    pending_annotations: list[Annotation] = field(default_factory=list)

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
        self.current_text_parts = []
        self.current_args_parts = []
        self.current_text_len = 0
        self.reasoning_signatures = []
        self.reasoning_redacted = []
        self.current_annotations = []


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
    yield from _close_reasoning_block(state)
    start = start_block["start"]
    state.current_text_parts = []
    state.current_args_parts = []
    state.current_text_len = 0

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
    yield from _close_reasoning_block(state)
    if state.block_kind is _BlockKind.NONE:
        # Synthesise a text block start if contentBlockStart was never received.
        state.current_text_parts = []
        state.current_text_len = 0
        state.current_item_id = f"{state.response_id}-msg-{state.output_index}"
        state.block_kind = _BlockKind.TEXT
        yield from _emit_text_block_start(state)
    if state.block_kind is not _BlockKind.TEXT or not state.current_item_id:
        return  # pragma: no cover
    state.current_text_parts.append(text_delta)
    state.current_text_len += len(text_delta)
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


def _handle_reasoning_delta(
    state: _StreamState, reasoning_delta: ReasoningContentBlockDeltaTypeDef
) -> Generator[JSONServerSentEvent]:
    """Emit reasoning SSE events for a Bedrock ``reasoningContent`` delta.

    The first delta of a reasoning block opens a ``reasoning`` output item
    (``response.output_item.added`` with empty content) followed by
    ``response.content_part.added`` for the reasoning-text part.  Text deltas
    emit ``response.reasoning_text.delta``; signature and redacted deltas are
    accumulated silently for the ``encrypted_content`` envelope.

    Args:
        state: Mutable stream state.
        reasoning_delta: The ``reasoningContent`` delta payload from Bedrock.

    Yields:
        ``JSONServerSentEvent`` objects for the reasoning delta.
    """
    if state.block_kind is not _BlockKind.REASONING:
        state.current_text_parts = []
        state.current_item_id = f"{state.response_id}-rs-{state.output_index}"
        state.block_kind = _BlockKind.REASONING
        yield json_sse(
            "response.output_item.added",
            ResponseOutputItemAddedEvent(
                item=ResponseReasoningItem(
                    id=state.current_item_id,
                    summary=[],
                    type="reasoning",
                    content=[],
                    status="in_progress",
                ),
                output_index=state.output_index,
                sequence_number=state.next_seq(),
                type="response.output_item.added",
            ),
        )
        yield json_sse(
            "response.content_part.added",
            ResponseContentPartAddedEvent(
                item_id=state.current_item_id,
                output_index=state.output_index,
                content_index=0,
                part=ContentPartReasoningText(text="", type="reasoning_text"),
                sequence_number=state.next_seq(),
                type="response.content_part.added",
            ),
        )
    if signature := reasoning_delta.get("signature"):
        state.reasoning_signatures.append(signature)
    if (data := reasoning_delta.get("redactedContent")) is not None:
        state.reasoning_redacted.append(data)
    if (text_delta := reasoning_delta.get("text")) and state.current_item_id:
        state.current_text_parts.append(text_delta)
        yield json_sse(
            "response.reasoning_text.delta",
            ResponseReasoningTextDeltaEvent(
                item_id=state.current_item_id,
                output_index=state.output_index,
                content_index=0,
                delta=text_delta,
                sequence_number=state.next_seq(),
                type="response.reasoning_text.delta",
            ),
        )


def _close_reasoning_block(state: _StreamState) -> Generator[JSONServerSentEvent]:
    """Close an open reasoning block, if any.

    Emits ``response.reasoning_text.done`` (when text was streamed) followed by
    ``response.content_part.done`` for the reasoning-text part and
    ``response.output_item.done`` with the completed reasoning item, records the
    item, and resets the per-block state.  No-op outside a reasoning block.

    Args:
        state: Mutable stream state.

    Yields:
        ``JSONServerSentEvent`` objects closing the reasoning block.
    """
    if state.block_kind is not _BlockKind.REASONING or state.current_item_id is None:
        return
    if reasoning_text := "".join(state.current_text_parts):
        yield json_sse(
            "response.reasoning_text.done",
            ResponseReasoningTextDoneEvent(
                item_id=state.current_item_id,
                output_index=state.output_index,
                content_index=0,
                text=reasoning_text,
                sequence_number=state.next_seq(),
                type="response.reasoning_text.done",
            ),
        )
    yield json_sse(
        "response.content_part.done",
        ResponseContentPartDoneEvent(
            item_id=state.current_item_id,
            output_index=state.output_index,
            content_index=0,
            part=ContentPartReasoningText(text=reasoning_text, type="reasoning_text"),
            sequence_number=state.next_seq(),
            type="response.content_part.done",
        ),
    )
    done_item = _build_reasoning_item(
        state.current_item_id,
        reasoning_text,
        state.reasoning_signatures,
        state.reasoning_redacted,
        include_encrypted_reasoning=state.include_encrypted_reasoning,
    )
    yield json_sse(
        "response.output_item.done",
        ResponseOutputItemDoneEvent(
            item=done_item,
            output_index=state.output_index,
            sequence_number=state.next_seq(),
            type="response.output_item.done",
        ),
    )
    state.output_items.append(done_item)
    state.output_index += 1
    state.reset_block()


def _handle_block_delta(
    state: _StreamState, delta_block: ContentBlockDeltaEventTypeDef
) -> Generator[JSONServerSentEvent]:
    """Emit SSE delta events for a Bedrock ``contentBlockDelta`` event.

    Handles text, tool-use argument, reasoning, and citation deltas.
    Synthesises a text block start when a text delta arrives without a prior
    block start; reasoning deltas open and feed a ``reasoning`` output item.

    Args:
        state: Mutable stream state.
        delta_block: The ``contentBlockDelta`` event from Bedrock.

    Yields:
        ``JSONServerSentEvent`` objects for the delta.
    """
    delta = delta_block["delta"]

    if state.block_kind in (_BlockKind.SUPPRESSED, _BlockKind.WEB_SEARCH):
        if tool_delta := delta.get("toolUse"):
            state.current_args_parts.append(tool_delta["input"])
        return

    if reasoning_delta := delta.get("reasoningContent"):
        yield from _handle_reasoning_delta(state, reasoning_delta)
    elif text_delta := delta.get("text"):
        yield from _emit_text_delta(state, text_delta)
    elif (
        (tool_delta := delta.get("toolUse"))
        and state.block_kind is _BlockKind.TOOL
        and state.current_item_id
    ):
        args_delta = tool_delta["input"]
        state.current_args_parts.append(args_delta)
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
    elif citation_delta := delta.get("citation"):
        yield from _record_citation(state, citation_delta)


def _record_citation(
    state: _StreamState, citation: CitationsDeltaTypeDef | CitationTypeDef
) -> Generator[JSONServerSentEvent]:
    """Record a single web citation as sources and ``url_citation`` annotation.

    Bedrock streams ``citationsContent`` after the ``web_search`` tool block has
    already closed, so the citation cannot be forwarded as a delta of the now
    completed ``response.output_item.done`` event.  Instead we collect them in
    ``state.pending_web_search_sources`` keyed by the target ``item_id`` and
    patch the stored ``ResponseFunctionWebSearch`` in
    ``state.output_items`` before the terminal event is emitted.

    The citation is also recorded as a ``url_citation`` annotation: when a text
    block is open, it is attached to it and a
    ``response.output_text.annotation.added`` event is emitted (indices
    approximated to the streamed text length); otherwise it is kept pending and
    patched into the last message before the terminal event.

    Args:
        state: Mutable stream state.
        citation: A Bedrock citation delta or completed citation.

    Yields:
        ``response.output_text.annotation.added`` events, when applicable.
    """
    url = ((citation.get("location") or {}).get("web") or {}).get("url")
    if not url:
        return
    target_id = next(
        (
            item.id
            for item in reversed(state.output_items)
            if isinstance(item, ResponseFunctionWebSearch)
        ),
        None,
    )
    if target_id is not None:
        state.pending_web_search_sources.setdefault(target_id, []).append(
            WebSearchActionSource(type="url", url=url)
        )
    annotation = AnnotationURLCitation(
        start_index=state.current_text_len,
        end_index=state.current_text_len,
        title=citation.get("title") or url,
        type="url_citation",
        url=url,
    )
    if state.block_kind is not _BlockKind.TEXT or state.current_item_id is None:
        state.pending_annotations.append(annotation)
        return
    state.current_annotations.append(annotation)
    yield json_sse(
        "response.output_text.annotation.added",
        ResponseOutputTextAnnotationAddedEvent(
            item_id=state.current_item_id,
            output_index=state.output_index,
            content_index=0,
            annotation=annotation,
            annotation_index=len(state.current_annotations) - 1,
            sequence_number=state.next_seq(),
            type="response.output_text.annotation.added",
        ),
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
    # Match non-streaming output: no argument deltas means an empty JSON object.
    arguments = "".join(state.current_args_parts) or "{}"
    yield json_sse(
        "response.function_call_arguments.done",
        ResponseFunctionCallArgumentsDoneEvent(
            item_id=state.current_item_id,
            output_index=state.output_index,
            arguments=arguments,
            name=state.current_tool_name,
            sequence_number=state.next_seq(),
            type="response.function_call_arguments.done",
        ),
    )
    done_tool_item = ResponseFunctionToolCall(
        arguments=arguments,
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
    ``output_item.done``; reasoning blocks emit ``reasoning_text.done`` +
    ``output_item.done``; non-suppressed tool blocks emit
    ``function_call_arguments.done`` + ``output_item.done``; web-search blocks
    emit ``web_search_call.completed`` + ``output_item.done``; suppressed tool
    blocks are only recorded in ``state.suppressed_tool_calls``.

    Args:
        state: Mutable stream state.

    Yields:
        ``JSONServerSentEvent`` objects for the block stop.
    """
    yield from _close_reasoning_block(state)
    if state.block_kind is _BlockKind.TEXT and state.current_item_id:
        text = "".join(state.current_text_parts)
        text_part = ResponseOutputText(
            annotations=list(state.current_annotations), text=text, type="output_text"
        )
        yield json_sse(
            "response.output_text.done",
            ResponseTextDoneEvent(
                item_id=state.current_item_id,
                output_index=state.output_index,
                content_index=0,
                logprobs=[],
                text=text,
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
    elif (
        state.block_kind
        in (_BlockKind.TOOL, _BlockKind.WEB_SEARCH, _BlockKind.SUPPRESSED)
        and state.current_item_id
        and state.current_tool_name
        and state.current_tool_id
    ):
        if state.block_kind is _BlockKind.WEB_SEARCH:
            args = "".join(state.current_args_parts)
            parsed = try_parse_json(args) if args else None
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
                (
                    state.current_tool_id,
                    state.current_tool_name,
                    "".join(state.current_args_parts),
                )
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
            state.cache_write_tokens = usage.get("cacheWriteInputTokens", 0)


def _classify_stream_error(
    exc: Exception,
) -> tuple[int, str, str | None, str | None, str, LogLevel | None]:
    """Classify a mid-stream exception for spec error events and logging.

    Mirrors the branches of ``log_request_sse_stream_event`` so Responses
    streams report the same status, sanitized message, and log detail as the
    legacy REST-envelope error path.

    Args:
        exc: The exception raised while streaming.

    Returns:
        Tuple of ``(status, client_message, param, code, log_message, log_level)``.
    """
    if isinstance(exc, ApiError):
        return (
            exc.status,
            hide_security_details(exc.status, exc.args[0]),
            exc.param,
            exc.code,
            exc.args[0],
            None,
        )
    if isinstance(exc, ClientError):
        error = exc.response["Error"]
        status = AWS_ERROR_MAP.get(error["Code"], (502, "server_error"))[0]
        message = error["Message"]
    elif isinstance(exc, HTTPClientError | BotocoreConnectionError):
        status = AWS_ERROR_MAP.get(exc.__class__.__name__, (503, "server_error"))[0]
        message = str(exc)
    else:
        return (
            500,
            "Internal Server Error",
            None,
            "server_error",
            "\n".join(format_exception(exc)),
            "critical",
        )
    return (
        status,
        hide_security_details(status, message),
        None,
        "server_error",
        message,
        None,
    )


def _finalize_output_items(state: _StreamState) -> None:
    """Patch pending citation sources and annotations into stored output items.

    Applied just before the terminal event so it reflects data (web-search
    sources, ``url_citation`` annotations) that arrived after the related
    output items were already closed.

    Args:
        state: Mutable stream state.
    """
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
    _patch_message_annotations(state.output_items, state.pending_annotations)


def _terminal_event(
    status: Literal["completed", "incomplete", "failed"],
    final_response: Response,
    state: _StreamState,
) -> JSONServerSentEvent:
    """Build the terminal stream event matching the final response status.

    Args:
        status: Final response status.
        final_response: Fully-built terminal response snapshot.
        state: Mutable stream state (provides the sequence number).

    Returns:
        ``response.completed``, ``response.incomplete``, or ``response.failed``
        SSE event.
    """
    match status:
        case "incomplete":
            return json_sse(
                "response.incomplete",
                ResponseIncompleteEvent(
                    response=final_response,
                    sequence_number=state.next_seq(),
                    type="response.incomplete",
                ),
            )
        case "failed":
            return json_sse(
                "response.failed",
                ResponseFailedEvent(
                    response=final_response,
                    sequence_number=state.next_seq(),
                    type="response.failed",
                ),
            )
        case "completed":
            return json_sse(
                "response.completed",
                ResponseCompletedEvent(
                    response=final_response,
                    sequence_number=state.next_seq(),
                    type="response.completed",
                ),
            )


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
    moderation_builder: Callable[[], ResponseModeration | None] | None = None,
) -> AsyncGenerator[JSONServerSentEvent]:
    """Stream Bedrock Converse events as OpenAI Responses API SSE events.

    Emits the full lifecycle event sequence: ``response.created`` →
    ``response.in_progress`` → per-block events → optional post-suppress items →
    a terminal ``response.completed``, ``response.incomplete``, or
    ``response.failed`` event matching the Bedrock stop reason.  Mid-stream
    exceptions emit a spec ``error`` event followed by a ``response.failed``
    snapshot, then re-raise as :class:`SseHandledStreamError` so the monitoring
    wrapper logs the error without emitting the legacy REST-envelope event.

    Args:
        response_id: Unique identifier for this response.
        created_at: Unix timestamp when the response was created.
        model_id: The Bedrock model identifier.
        stream: Bedrock ConverseStream event iterator.
        request: The original Responses API creation request.
        suppress_tool_names: Tool names whose output items should be filtered.
        post_suppress_handler: Optional async generator called after the main stream
            but before the terminal event.  Receives the stream state and may
            append items to ``state.output_items`` and yield additional SSE events
            (e.g. for server-side image generation results).
        web_search_tool_names: Tool names to emit as ``web_search_call`` output
            items with query and completion events.
        moderation_builder: Optional callable building the response ``moderation``
            field from the guardrail trace, invoked at stream end so the
            terminal event carries the complete trace.

    Yields:
        ``JSONServerSentEvent`` for each Responses API stream event.

    Raises:
        SseHandledStreamError: When a mid-stream exception was already reported
            to the client via spec error events.
    """
    state = _StreamState(
        response_id, include_encrypted_reasoning=_includes_encrypted_reasoning(request)
    )
    try:
        initial_response = _build_response_object(
            response_id,
            created_at,
            model_id,
            [],
            "in_progress",
            None,
            None,
            None,
            request,
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

        # Defensive: close a reasoning block left open by a stream without contentBlockStop.
        for sse in _close_reasoning_block(state):
            yield sse

        if post_suppress_handler:
            async for sse in post_suppress_handler(state):
                yield sse

        _finalize_output_items(state)

        status, incomplete_details, error = _map_stop_reason(state.stop_reason)
        input_tokens = (
            state.input_tokens + state.cached_tokens + state.cache_write_tokens
        )
        final_response = _build_response_object(
            response_id,
            created_at,
            model_id,
            state.output_items,
            status,
            incomplete_details,
            error,
            ResponseUsage(
                input_tokens=input_tokens,
                input_tokens_details=InputTokensDetails(
                    cached_tokens=state.cached_tokens,
                    cache_write_tokens=state.cache_write_tokens,
                ),
                output_tokens=state.output_tokens,
                output_tokens_details=OutputTokensDetails(),
                total_tokens=input_tokens + state.output_tokens,
            ),
            request,
        )
        if moderation_builder is not None:
            final_response.moderation = moderation_builder()
        log_response_params(final_response)
        yield _terminal_event(status, final_response, state)
    except Exception as exc:
        status_code, message, param, code, log_message, log_level = (
            _classify_stream_error(exc)
        )
        yield json_sse(
            "error",
            ResponseErrorEvent(
                message=message,
                code=code,
                param=param,
                sequence_number=state.next_seq(),
                type="error",
            ),
        )
        yield json_sse(
            "response.failed",
            ResponseFailedEvent(
                response=_build_response_object(
                    response_id,
                    created_at,
                    model_id,
                    state.output_items,
                    "failed",
                    None,
                    ResponseError(
                        code=(
                            "rate_limit_exceeded"
                            if status_code == 429
                            else "server_error"
                        ),
                        message=message,
                    ),
                    None,
                    request,
                ),
                sequence_number=state.next_seq(),
                type="response.failed",
            ),
        )
        raise SseHandledStreamError(
            log_message, status=status_code, level=log_level
        ) from exc


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

    # Reuse the converse tool mapping so synthetic and integrated tools count too.
    if tool_config := _build_tool_config(request):
        req["toolConfig"] = tool_config

    with handle_bedrock_client_error():
        resp: CountTokensResponseTypeDef = await get_client(
            "bedrock-runtime", region
        ).count_tokens(modelId=model_id, input={"converse": req})
    return resp["inputTokens"]
