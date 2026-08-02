"""Local OpenAI-compatible chat completions types."""

from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import (
    AliasChoices,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from stdapi.api_errors import UnsupportedParameterError
from stdapi.config import SETTINGS
from stdapi.input_file import FileIdInputFile, InputFile
from stdapi.types import (
    BaseModelRequest,
    BaseModelRequestWithExtra,
    BaseModelResponse,
    JsonMapping,
)
from stdapi.types.bedrock import AmazonBedrockGuardrailConfigParams
from stdapi.types.openai import (
    AssistantRoleLiteral,
    ChatModeration,
    CustomLiteral,
    FunctionDefinition,
    FunctionLiteral,
    LegacyFunction,
    Metadata,
    PaginatedListEnvelope,
    RequestModeration,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
    ResponseFormatText,
    TextLiteral,
)

#: Reasoning effort selector for reasoning models.
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

#: Finish reasons compatible with OpenAI.
FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]

#: Service tiers
ServiceTiers = Literal[
    "auto",
    "default",
    "flex",
    "scale",
    "priority",
    # Extra bedrock specific values
    "reserved",
]

#: Prompt cache retention
PromptCacheRetention = Literal[
    "in_memory",
    "24h",
    # Extra bedrock specific values
    "1h",
    "5m",
]

#: Prompt cache retention of the `prompt_cache_options.ttl` field
PromptCacheOptionsTTL = Literal["30m"]

#: Tool choice literal values used in multiple request fields (OpenAI-compatible).
ToolChoiceLiteral = Literal["none", "auto", "required"]

#: Common three-level setting used by multiple parameters (e.g., verbosity, search_context_size).
VerbosityLevel = Literal["low", "medium", "high"]

#: Supported output modalities
OutputModalities = Literal["text", "audio"]


# Ref: openai.types.chat.chat_completion_content_part_text_param.PromptCacheBreakpoint
class PromptCacheBreakpoint(BaseModelRequest):
    """Explicit prompt-cache breakpoint set on a content part."""

    mode: Literal["explicit"] = Field(
        default="explicit",
        description="Breakpoint mode. Always `explicit`: the prompt prefix ending "
        "with this content part is cached.",
    )


#: Description shared by every per-content-part `prompt_cache_breakpoint` field.
_CACHE_BREAKPOINT_DESCRIPTION = (
    "Cache the prompt prefix ending with this content part. Honored on models "
    "supporting prompt caching, accepted and ignored on the others."
)


# Ref: openai.types.chat.chat_completion_content_part_text_param.ChatCompletionContentPartTextParam
class ChatCompletionContentPartTextParam(BaseModelRequest):
    """Text message content part."""

    type: TextLiteral = Field(description="Content part type. Always `text`.")
    text: str = Field(description="Text content of the message part.")
    prompt_cache_breakpoint: PromptCacheBreakpoint | None = Field(
        default=None, description=_CACHE_BREAKPOINT_DESCRIPTION
    )


# Ref: openai.types.chat.chat_completion_content_part_refusal_param.ChatCompletionContentPartRefusalParam
class ChatCompletionContentPartRefusalParam(BaseModelRequest):
    """Refusal message content part."""

    type: Literal["refusal"] = Field(description="Content part type. Always `refusal`.")
    refusal: str = Field(description="Refusal content text.")
    prompt_cache_breakpoint: PromptCacheBreakpoint | None = Field(
        default=None, description=_CACHE_BREAKPOINT_DESCRIPTION
    )


# Ref: openai.types.chat.chat_completion_content_part_image.ImageURL
class ImageURL(BaseModelRequest):
    """Image URL detail for image content part."""

    url: InputFile = Field(
        description="Image URL string, data URI, S3 URI, or base64-encoded string."
    )
    detail: Literal["low", "high", "auto"] | None = Field(
        default=None,
        description="Image resolution: `low`, `high`, or `auto`. Default: `auto`.",
    )


# Ref: openai.types.chat.chat_completion_content_part_image_param.ChatCompletionContentPartImageParam
class ChatCompletionContentPartImageParam(BaseModelRequest):
    """Image message content part (via URL)."""

    type: Literal["image_url"] = Field(
        description="Content part type. Always `image_url`."
    )
    image_url: ImageURL = Field(
        description="URL descriptor containing the image `url` field."
    )
    prompt_cache_breakpoint: PromptCacheBreakpoint | None = Field(
        default=None, description=_CACHE_BREAKPOINT_DESCRIPTION
    )


# Ref: openai.types.chat.chat_completion_content_part_param.FileFile
class FileFile(BaseModelRequest):
    """File content descriptor."""

    file_id: FileIdInputFile | None = Field(
        default=None, description="ID of an uploaded file to use as input."
    )
    file_data: InputFile | None = Field(
        default=None, description="Base64-encoded file data, data URI, S3 URI, or URL."
    )
    filename: str | None = Field(
        default=None, description="Name of the file when passing as a string."
    )

    @model_validator(mode="after")
    def _validate_file_source(self) -> Self:
        """Validate that either file_id or file_data is present.

        Returns:
            Self: The validated instance.

        Raises:
            ValueError: If neither file_id nor file_data is provided.
        """
        if (self.file_id is None and self.file_data is None) or (
            self.file_id is not None and self.file_data is not None
        ):
            msg = "Either 'file_id' or 'file_data' must be provided."
            raise ValueError(msg)
        return self


# Ref: openai.types.chat.chat_completion_content_part_param.File
class File(BaseModelRequest):
    """File message content part."""

    type: Literal["file"] = Field(description="Content part type. Always `file`.")
    file: FileFile = Field(description="Content descriptor containing base64 bytes.")
    prompt_cache_breakpoint: PromptCacheBreakpoint | None = Field(
        default=None, description=_CACHE_BREAKPOINT_DESCRIPTION
    )


# Ref: openai.types.chat.chat_completion_content_part_input_audio_param.InputAudio
class InputAudio(BaseModelRequest):
    """Input audio descriptor."""

    data: InputFile = Field(
        description="Base64-encoded audio data, data URI, S3 URI, or URL."
    )
    format: Literal["wav", "mp3"] = Field(description="Audio format: `wav` or `mp3`.")


# Ref: openai.types.chat.chat_completion_content_part_input_audio_param.ChatCompletionContentPartInputAudioParam
class ChatCompletionContentPartInputAudioParam(BaseModelRequest):
    """Input audio message content part."""

    input_audio: InputAudio = Field(description="Audio data descriptor.")
    type: Literal["input_audio"] = Field(
        description="Content part type. Always `input_audio`."
    )
    prompt_cache_breakpoint: PromptCacheBreakpoint | None = Field(
        default=None, description=_CACHE_BREAKPOINT_DESCRIPTION
    )


# Ref: openai.types.chat.chat_completion_content_part_param.ChatCompletionContentPartParam
ChatCompletionContentPartParam = Annotated[
    ChatCompletionContentPartTextParam
    | ChatCompletionContentPartImageParam
    | ChatCompletionContentPartInputAudioParam
    | File,
    Field(discriminator="type"),
]


# Ref: openai.types.chat.chat_completion_prediction_content_param.ChatCompletionPredictionContentParam
class ChatCompletionPredictionContentParam(BaseModelRequest):
    """Predicted content hint to speed up responses."""

    type: Literal["content"] = Field(
        description="The type of the predicted content. Always `content`."
    )
    content: str | list[ChatCompletionContentPartTextParam] = Field(
        description=(
            "Content to match for predicted output. If generated tokens match this, response returns quickly."
        )
    )


# Ref: openai.types.chat.chat_completion_message.FunctionCall
# Ref: openai.types.chat.chat_completion_message_function_tool_call.Function
# Ref: openai.types.chat.chat_completion_message_function_tool_call_param.Function
class FunctionCall(BaseModelResponse):
    """Function tool call payload used within assistant tool calls."""

    name: str = Field(description="The name of the function to call.")
    arguments: str = Field(
        description="JSON arguments for the function call. May be invalid or hallucinated; validate before use."
    )


# Ref: openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCallFunction
# Ref: openai.types.chat.chat_completion_chunk.ChoiceDeltaFunctionCall
class ChoiceDeltaFunctionCall(BaseModelResponse):
    """Function tool call."""

    name: str | None = Field(
        default=None, description="The name of the function to call."
    )
    arguments: str | None = Field(
        default=None,
        description="JSON arguments for the function call. May be invalid or hallucinated; validate before use.",
    )


# Ref: openai.types.chat.chat_completion_message_custom_tool_call.Custom
# Ref: openai.types.chat.chat_completion_message_custom_tool_call_param.Custom
class CustomTool(BaseModelResponse):
    """Custom tool call payload used within assistant tool calls.

    UNSUPPORTED on this implementation.
    """

    name: str = Field(
        description="The name of the custom tool to call.\nUNSUPPORTED on this implementation."
    )
    input: str = Field(
        description="The input for the custom tool call generated by the model.\nUNSUPPORTED on this implementation."
    )


# Ref: openai.types.chat.chat_completion_message_function_tool_call.ChatCompletionMessageFunctionToolCall
# Ref: openai.types.chat.chat_completion_message_function_tool_call_param.ChatCompletionMessageFunctionToolCallParam
class ChatCompletionMessageFunctionToolCall(BaseModelResponse):
    """Assistant tool call for a function tool."""

    type: FunctionLiteral = Field(description="Tool type. Always `function`.")
    function: FunctionCall = Field(description="The function that the model called.")
    id: str = Field(description="The ID of the tool call.")


# Ref: openai.types.chat.chat_completion_message_custom_tool_call.ChatCompletionMessageCustomToolCall
# Ref: openai.types.chat.chat_completion_message_custom_tool_call_param.ChatCompletionMessageCustomToolCallParam
class ChatCompletionMessageCustomToolCall(BaseModelResponse):
    """Assistant tool call for a custom tool.

    UNSUPPORTED on this implementation.
    """

    type: CustomLiteral = Field(
        description="Tool type. Always `custom`. UNSUPPORTED on this implementation."
    )
    custom: CustomTool = Field(
        description="The custom tool that the model called. UNSUPPORTED on this implementation."
    )
    id: str = Field(description="The ID of the tool call.")


# Ref: openai.types.chat.chat_completion_message_tool_call.ChatCompletionMessageToolCallUnion
# Ref: openai.types.chat.chat_completion_message_tool_call_union_param.ChatCompletionMessageToolCallUnionParam
ChatCompletionMessageToolCallUnion = Annotated[
    ChatCompletionMessageFunctionToolCall | ChatCompletionMessageCustomToolCall,
    Field(discriminator="type"),
]


# Ref: openai.types.chat.chat_completion_function_tool_param.ChatCompletionFunctionToolParam
class ChatCompletionFunctionToolParam(BaseModelRequest):
    """Function tool specification."""

    type: FunctionLiteral = Field(description="Tool type. Always `function`.")
    function: FunctionDefinition = Field(description="Function definition.")


# Ref: openai.types.chat.chat_completion_custom_tool_param.CustomFormatText
class CustomFormatText(BaseModelRequest):
    """Unconstrained text format. Always `text`.

    UNSUPPORTED on this implementation.
    """

    type: TextLiteral = Field(
        description="Format type. Always `text`. UNSUPPORTED on this implementation."
    )


# Ref: openai.types.chat.chat_completion_custom_tool_param.CustomFormatGrammarGrammar
class CustomFormatGrammarGrammar(BaseModelRequest):
    """The grammar definition and syntax for a grammar-based custom tool input.

    UNSUPPORTED on this implementation.
    """

    definition: str = Field(
        description="Grammar definition. UNSUPPORTED on this implementation."
    )
    syntax: Literal["lark", "regex"] = Field(
        description="Grammar syntax: `lark` or `regex`. UNSUPPORTED on this implementation."
    )


# Ref: openai.types.chat.chat_completion_custom_tool_param.CustomFormatGrammar
class CustomFormatGrammar(BaseModelRequest):
    """Grammar format. Always `grammar`.

    UNSUPPORTED on this implementation.
    """

    type: Literal["grammar"] = Field(
        description="Format type. Always `grammar`. UNSUPPORTED on this implementation."
    )
    grammar: CustomFormatGrammarGrammar = Field(
        description="Grammar definition. UNSUPPORTED on this implementation."
    )


# Ref: openai.types.chat.chat_completion_custom_tool_param.CustomFormat
CustomFormat = Annotated[
    CustomFormatText | CustomFormatGrammar, Field(discriminator="type")
]


# Ref: openai.types.chat.chat_completion_custom_tool_param.Custom
class Custom(BaseModelRequest):
    """Properties of the custom tool used for custom tool calling.

    UNSUPPORTED on this implementation.
    """

    name: str = Field(
        description="Name of the custom tool. UNSUPPORTED on this implementation."
    )
    description: str | None = Field(
        default=None,
        description="Description of the custom tool. UNSUPPORTED on this implementation.",
    )
    format: CustomFormat | None = Field(
        default=CustomFormatText(type="text"),
        description="Input format for the custom tool. Default: unconstrained text. UNSUPPORTED on this implementation.",
    )


# Ref: openai.types.chat.chat_completion_custom_tool_param.ChatCompletionCustomToolParam
class ChatCompletionCustomToolParam(BaseModelRequest):
    """Custom tool specification.

    UNSUPPORTED on this implementation.
    """

    type: CustomLiteral = Field(
        description="Tool type. Always `custom`. UNSUPPORTED on this implementation."
    )
    custom: Custom = Field(
        description="Custom tool properties. UNSUPPORTED on this implementation."
    )


# Ref: openai.types.chat.chat_completion_tool_union_param.ChatCompletionToolUnionParam
ChatCompletionToolUnionParam = Annotated[
    ChatCompletionFunctionToolParam | ChatCompletionCustomToolParam,
    Field(discriminator="type"),
]


# Ref: openai.types.chat.chat_completion_named_tool_choice_param.Function
# Ref: openai.types.chat.chat_completion_function_call_option_param.ChatCompletionFunctionCallOptionParam
class FunctionToolChoiceParam(BaseModelRequest):
    """The function to call by name."""

    name: str = Field(description="The name of the function to call.")


# Ref: openai.types.chat.chat_completion_named_tool_choice_param.ChatCompletionNamedToolChoiceParam
class ChatCompletionNamedToolChoiceParam(BaseModelRequest):
    """Named tool choice for function tools."""

    type: FunctionLiteral = Field(
        description="For function calling, the type is always `function`."
    )
    function: FunctionToolChoiceParam = Field(
        description="The function to call by name."
    )


# Ref: openai.types.chat.chat_completion_named_tool_choice_custom_param.Custom
class CustomToolChoice(BaseModelRequest):
    """The custom tool to call by name.

    UNSUPPORTED on this implementation.
    """

    name: str = Field(
        description="The name of the custom tool to call.\nUNSUPPORTED on this implementation."
    )


# Ref: openai.types.chat.chat_completion_named_tool_choice_custom_param.ChatCompletionNamedToolChoiceCustomParam
class ChatCompletionNamedToolChoiceCustomParam(BaseModelRequest):
    """Named tool choice for custom tools.

    UNSUPPORTED on this implementation.
    """

    type: CustomLiteral = Field(
        description="Tool type. Always `custom`. UNSUPPORTED on this implementation."
    )
    custom: CustomToolChoice = Field(
        description="Custom tool to call by name. UNSUPPORTED on this implementation."
    )


ChatCompletionAllowedToolsToolsParam = Annotated[
    ChatCompletionNamedToolChoiceParam | ChatCompletionNamedToolChoiceCustomParam,
    Field(discriminator="type"),
]


# Ref: openai.types.chat.chat_completion_allowed_tools_param.ChatCompletionAllowedToolsParam
class ChatCompletionAllowedToolsParam(BaseModelRequest):
    """Allowed tools for function tools."""

    mode: Literal["auto", "required"] = Field(
        description="Tool selection mode: `auto` lets model pick, `required` forces a tool call."
    )
    tools: list[ChatCompletionAllowedToolsToolsParam] = Field(
        description="List of tool definitions the model can call."
    )


# Ref: openai.types.chat.chat_completion_allowed_tool_choice_param.ChatCompletionAllowedToolChoiceParam
class ChatCompletionAllowedToolChoiceParam(BaseModelRequest):
    """Allowed tools list choice. Used by OpenAI; mapped to auto in this project."""

    type: Literal["allowed_tools"] = Field(
        description="Allowed tools choice type. Always `allowed_tools`."
    )
    allowed_tools: ChatCompletionAllowedToolsParam = Field(
        description="Constrains the tools available to the model to a pre-defined set."
    )


# Ref: openai.types.chat.chat_completion_tool_choice_option_param.ChatCompletionToolChoiceOptionParam
ChatCompletionToolChoiceOptionParam = (
    ToolChoiceLiteral
    | Annotated[
        ChatCompletionNamedToolChoiceParam
        | ChatCompletionNamedToolChoiceCustomParam
        | ChatCompletionAllowedToolChoiceParam,
        Field(discriminator="type"),
    ]
)

# Ref: openai.types.chat.completion_create_params.FunctionCall
FunctionCallParam = Literal["none", "auto"] | FunctionToolChoiceParam


# Ref: openai.types.chat.chat_completion_assistant_message_param.Audio
class Audio(BaseModelRequest):
    """Data about a previous audio response from the model."""

    id: str = Field(description="ID of a previous audio response from the model.")


class _MessageParam(BaseModelRequest):
    """Common role message fields."""

    name: str | None = Field(
        default=None,
        description="Optional participant name to differentiate same-role participants.",
    )


# Ref: openai.types.chat.chat_completion_assistant_message_param.ChatCompletionAssistantMessageParam
class ChatCompletionAssistantMessageParam(_MessageParam):
    """Assistant role message."""

    role: AssistantRoleLiteral = Field(
        description="Message author role. Always `assistant`."
    )
    audio: Audio | None = Field(
        default=None, description="Data about a previous audio response from the model."
    )
    content: (
        str
        | list[
            Annotated[
                ChatCompletionContentPartTextParam
                | ChatCompletionContentPartRefusalParam,
                Field(discriminator="type"),
            ]
        ]
        | None
    ) = Field(
        default=None,
        description="Assistant message content. Required unless `tool_calls` or `function_call` is specified.",
    )
    function_call: FunctionCall | None = Field(
        default=None,
        description="Deprecated. Use `tool_calls` instead. Function name and arguments.",
    )
    refusal: str | None = Field(
        default=None, description="Refusal message from the assistant."
    )
    tool_calls: list[ChatCompletionMessageToolCallUnion] | None = Field(
        default=None, description="Tool calls generated by the model."
    )
    # Deepseek Chat Completion API fields.
    reasoning_content: str | list[ChatCompletionContentPartTextParam] | None = Field(
        default=None,
        description="Reasoning content, also accepted under the `reasoning` name. Extra field from Deepseek Chat Completion API.",
    )
    prefix: bool | None = Field(
        default=None,
        description="Force model to start with the provided prefix. UNSUPPORTED on this implementation.",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_reasoning_alias(cls, data: Any) -> Any:  # noqa: ANN401
        """Read a replayed `reasoning` field as `reasoning_content`.

        Args:
            data: Raw assistant message.

        Returns:
            The message, with `reasoning` folded into `reasoning_content` when
            the latter carries nothing.
        """
        if not isinstance(data, dict) or "reasoning" not in data:
            return data
        data = dict(data)
        alias = data.pop("reasoning")
        if data.get("reasoning_content") is None:
            data["reasoning_content"] = alias
        return data


# Ref: openai.types.chat.chat_completion_user_message_param.ChatCompletionUserMessageParam
class ChatCompletionUserMessageParam(_MessageParam):
    """User role message."""

    role: Literal["user"] = Field(description="Message author role. Always `user`.")
    content: str | list[ChatCompletionContentPartParam] = Field(
        description="User message content."
    )


# Ref: openai.types.chat.chat_completion_system_message_param.ChatCompletionSystemMessageParam
class ChatCompletionSystemMessageParam(_MessageParam):
    """System role message."""

    role: Literal["system"] = Field(description="Message author role. Always `system`.")
    content: str | list[ChatCompletionContentPartTextParam] = Field(
        description="System message content."
    )


# Ref: openai.types.chat.chat_completion_developer_message_param.ChatCompletionDeveloperMessageParam
class ChatCompletionDeveloperMessageParam(_MessageParam):
    """Developer role message."""

    role: Literal["developer"] = Field(
        description="Message author role. Always `developer`."
    )
    content: str | list[ChatCompletionContentPartTextParam] = Field(
        description="Developer message content."
    )


# Ref: openai.types.chat.chat_completion_tool_message_param.ChatCompletionToolMessageParam
class ChatCompletionToolMessageParam(BaseModelRequest):
    """Tool role message."""

    role: Literal["tool"] = Field(description="Message author role. Always `tool`.")
    content: str | list[ChatCompletionContentPartTextParam] = Field(
        description="Text content or list of text parts."
    )
    tool_call_id: str = Field(description="Tool call this message responds to.")


# Ref: openai.types.chat.chat_completion_function_message_param.ChatCompletionFunctionMessageParam
class ChatCompletionFunctionMessageParam(BaseModelRequest):
    """Function role message."""

    role: FunctionLiteral = Field(description="Message author role. Always `function`.")
    name: str = Field(description="The name of the function to call.")
    content: str | None = Field(description="Function message content.")


# Ref: openai.types.chat.chat_completion_message_param.ChatCompletionMessageParam
ChatCompletionMessageParam = Annotated[
    ChatCompletionDeveloperMessageParam
    | ChatCompletionSystemMessageParam
    | ChatCompletionUserMessageParam
    | ChatCompletionAssistantMessageParam
    | ChatCompletionToolMessageParam
    | ChatCompletionFunctionMessageParam,
    Field(discriminator="role"),
]


# Ref: openai.types.chat.chat_completion_audio_param.ChatCompletionAudioParam
class ChatCompletionAudioParam(BaseModelRequest):
    """Parameters for audio output."""

    format: Literal["wav", "aac", "mp3", "flac", "opus", "pcm16"] = Field(
        description="Output audio format: `wav`, `aac`, `mp3`, `flac`, `opus`, or `pcm16`."
    )
    voice: str = Field(description="Voice for audio response.")


# Ref: openai.types.chat.chat_completion_stream_options_param.ChatCompletionStreamOptionsParam
class ChatCompletionStreamOptionsParam(BaseModelRequest):
    """Options for streaming responses."""

    include_usage: bool = Field(
        default=False,
        description="If true, streams a usage chunk before `data: [DONE]` with token statistics. "
        "The `choices` field will be empty. Other chunks include a null usage field.",
    )
    include_obfuscation: bool = Field(
        default=False,
        description="Enable stream obfuscation to normalize payload sizes for security. "
        "Adds overhead to the data stream; set to false to optimize bandwidth.",
    )


# Ref: openai.types.chat.completion_create_params.PromptCacheOptions
class PromptCacheOptions(BaseModelRequest):
    """Prompt caching options."""

    mode: Literal["implicit", "explicit"] | None = Field(
        default=None,
        description="Caching mode. `explicit`: only content parts marked with "
        "`prompt_cache_breakpoint` are cached. `implicit` (default): sections "
        "selected with `prompt_cache_key` are cached too.",
    )
    ttl: PromptCacheOptionsTTL | None = Field(
        default=None,
        description="Cache retention: `30m` is applied as 1h. Ignored when `prompt_cache_retention` is set.",
    )


# Ref: openai.types.chat.completion_create_params.ResponseFormat
ResponseFormat = Annotated[
    ResponseFormatText | ResponseFormatJSONSchema | ResponseFormatJSONObject,
    Field(discriminator="type"),
]


# Qwen Chat Completion API fields.
class QwenTranslationTerm(BaseModelRequest):
    """Term intervention for Qwen translation.

    Allows specifying custom translations for specific terms.
    """

    source: str = Field(description="Source term to translate")
    target: str = Field(description="Target translation for the term")


class QwenTranslationMemory(BaseModelRequest):
    """Translation memory entry for Qwen translation.

    Provides example translations to guide the model.
    """

    source: str = Field(description="Source statement")
    target: str = Field(description="Target translation statement")


class MoonshotThinkingOptions(BaseModelRequest):
    """Thinking configuration for Moonshot models.

    Controls whether thinking is enabled for the model.
    """

    type: Literal["enabled", "disabled"] = Field(
        description="Enable or disable thinking capability."
    )


class ReasoningOptions(BaseModelRequest):
    """Reasoning configuration accepted as a single object.

    Groups the reasoning knobs the flat fields expose individually: `effort`
    matches `reasoning_effort`, `max_tokens` matches `thinking_budget`, and
    `enabled` matches `enable_thinking`.
    """

    effort: ReasoningEffort | None = Field(
        default=None,
        description="Reasoning effort, as `reasoning_effort`. Mutually exclusive with `max_tokens`.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Max reasoning length in tokens, as `thinking_budget`. Mutually exclusive with `effort`.",
    )
    exclude: bool | None = Field(
        default=None,
        description="Omit the reasoning text from the response; it is still generated and billed.",
    )
    enabled: bool | None = Field(
        default=None, description="Enable reasoning, as `enable_thinking`."
    )


class QwenTranslationOptions(BaseModelRequest):
    """Translation options for Qwen models with translation capabilities.

    Configures source/target languages, custom term translations, translation
    memory, and domain hints for specialized translation tasks.
    """

    source_lang: str = Field(
        description='Source language name in English, or "auto" for auto-detection.'
    )
    target_lang: str = Field(description="Target language name in English.")
    terms: list[QwenTranslationTerm] | None = Field(
        default=None, description="Custom term translations with source and target."
    )
    tm_list: list[QwenTranslationMemory] | None = Field(
        default=None,
        description="Translation memory with source and target statements.",
    )
    domains: str | None = Field(
        default=None,
        description='Domain hint in English (e.g., "medical", "legal", "technical").',
    )


# Ref: openai.types.chat.completion_create_params.WebSearchOptionsUserLocationApproximate
class WebSearchOptionsUserLocationApproximate(BaseModelResponse):
    """Approximate user location for web search.

    UNSUPPORTED on this implementation.
    """

    city: str | None = Field(
        default=None,
        description="User city, e.g., `San Francisco`. UNSUPPORTED on this implementation.",
    )
    country: str | None = Field(
        default=None,
        description="Two-letter ISO country code, e.g., `US`. UNSUPPORTED on this implementation.",
    )
    region: str | None = Field(
        default=None,
        description="User region, e.g., `California`. UNSUPPORTED on this implementation.",
    )
    timezone: str | None = Field(
        default=None,
        description="IANA timezone, e.g., `America/Los_Angeles`. UNSUPPORTED on this implementation.",
    )


# Ref: openai.types.chat.completion_create_params.WebSearchOptionsUserLocation
class WebSearchOptionsUserLocation(BaseModelResponse):
    """User location parameters for web search (approximate).

    UNSUPPORTED on this implementation.
    """

    type: Literal["approximate"] = Field(
        description="Location type. Always `approximate`. UNSUPPORTED on this implementation."
    )
    approximate: WebSearchOptionsUserLocationApproximate = Field(
        description="Approximate location parameters. UNSUPPORTED on this implementation."
    )


# Ref: openai.types.chat.completion_create_params.WebSearchOptions
class WebSearchOptions(BaseModelResponse):
    """Web search tool options.

    UNSUPPORTED on this implementation.
    """

    search_context_size: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Search context size: `low`, `medium`, or `high`. Default: `medium`. UNSUPPORTED on this implementation.",
    )
    user_location: WebSearchOptionsUserLocation | None = Field(
        default=None,
        description="Approximate location parameters. UNSUPPORTED on this implementation.",
    )


# Ref: openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCall
class ChoiceDeltaToolCall(BaseModelResponse):
    """Tool call delta information for streaming chunks."""

    index: int = Field(ge=0, description="Position index of the tool call.")
    type: FunctionLiteral = Field(
        description="Tool type. Currently only `function` is supported."
    )
    function: ChoiceDeltaFunctionCall | None = Field(
        default=None, description="Partial function call details."
    )
    id: str | None = Field(default=None, description="The ID of the tool call.")


# Ref: openai.types.chat.chat_completion_chunk.ChoiceDelta
class ChoiceDelta(BaseModelResponse):
    """Delta updates for a streaming choice."""

    content: str | None = Field(default=None, description="Chunk message content.")
    function_call: ChoiceDeltaFunctionCall | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated. Use `tool_calls` instead. Function name and arguments.",
    )
    refusal: str | None = Field(
        default=None, description="Refusal message from the model."
    )
    role: Literal["developer", "system", "user", "assistant", "tool"] | None = Field(
        default=None, description="Message author role."
    )
    tool_calls: list[ChoiceDeltaToolCall] | None = Field(
        default=None, description="Partial tool call entries."
    )
    # Deepseek Chat Completion API fields.
    reasoning_content: str | None = Field(
        default=None,
        description="Reasoning content. Extra field from Deepseek Chat Completion API.",
    )

    @model_serializer(mode="wrap")
    def _emit_reasoning_under_the_configured_name(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        """Serialize, then rename or drop the reasoning text per the setting.

        Returns:
            The serialized mapping.
        """
        return _rename_emitted_reasoning(handler(self))


# Ref: openai.types.chat.chat_completion_message.AnnotationURLCitation
class AnnotationURLCitation(BaseModelResponse):
    """A URL citation when using web search."""

    end_index: int = Field(
        ge=0, description="Last character index of the URL citation in the message."
    )
    start_index: int = Field(
        ge=0, description="First character index of the URL citation in the message."
    )
    title: str = Field(description="Title of the web resource.")
    url: str = Field(description="URL of the web resource.")


# Ref: openai.types.chat.chat_completion_message.Annotation
class Annotation(BaseModelResponse):
    """Annotation for the message when using web search."""

    type: Literal["url_citation"] = Field(
        description="Citation type. Always `url_citation`."
    )
    url_citation: AnnotationURLCitation = Field(
        description="URL citation from web search."
    )


# Ref: openai.types.chat.chat_completion_audio.ChatCompletionAudio
class ChatCompletionAudio(BaseModelResponse):
    """If audio output modality is requested, contains data about the audio response."""

    id: str = Field(description="Unique ID for this audio response.")
    data: str = Field(description="Base64-encoded audio bytes in the requested format.")
    expires_at: int = Field(
        ge=0,
        description="Unix timestamp (seconds) when audio expires and is no longer accessible.",
    )
    transcript: str = Field(description="Transcript of the generated audio.")


# Ref: openai.types.chat.chat_completion_message.ChatCompletionMessage
class ChatCompletionMessage(BaseModelResponse):
    """Assistant message object in the non-streaming ChatCompletion."""

    role: AssistantRoleLiteral = Field(
        description="Message author role. Always `assistant`."
    )
    content: str | None = Field(default=None, description="Message content.")
    refusal: str | None = Field(
        default=None, description="Refusal message from the model."
    )
    annotations: list[Annotation] | None = Field(
        default=None, description="Message annotations from web search."
    )
    audio: ChatCompletionAudio | None = Field(
        default=None, description="Audio response data when audio output is requested."
    )
    function_call: FunctionCall | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated. Use `tool_calls` instead. Function name and arguments.",
    )
    tool_calls: list[ChatCompletionMessageToolCallUnion] | None = Field(
        default=None, description="Tool calls generated by the model."
    )
    # Deepseek Chat Completion API fields.
    reasoning_content: str | None = Field(
        default=None,
        description="Reasoning content. Extra field from Deepseek Chat Completion API.",
    )

    @model_serializer(mode="wrap")
    def _emit_reasoning_under_the_configured_name(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        """Serialize, then rename or drop the reasoning text per the setting.

        Returns:
            The serialized mapping.
        """
        return _rename_emitted_reasoning(handler(self))


#: Emitted reasoning field, or nothing at all, as the operator configured it.
def _rename_emitted_reasoning(data: dict[str, Any]) -> dict[str, Any]:
    """Move the serialized reasoning text under the configured field name.

    The OpenAI API returns no thinking text on this route, so vendors picked
    different names for it and clients read whichever their first backend used.
    The name is an operator setting rather than a request field: inventing a
    per-request selector would be a gateway-specific API field, and no vendor
    defines one.

    Args:
        data: Serialized message or delta, modified in place.

    Returns:
        The same mapping.
    """
    if (text := data.pop("reasoning_content", None)) is None:
        return data
    field = SETTINGS.chat_completions_reasoning_field
    if field != "none":
        data[field] = text
    return data


# Ref: openai.types.chat.chat_completion_token_logprob.TopLogprob
class TopLogprob(BaseModelResponse):
    """Top log probability token information."""

    token: str = Field(description="The token.")
    bytes: list[int] | None = Field(
        default=None,
        description="UTF-8 byte representation of the token. Can be null if unavailable.",
    )
    logprob: float = Field(
        description="Log probability if in top 20 tokens, otherwise -9999.0."
    )


# Ref: openai.types.chat.chat_completion_token_logprob.ChatCompletionTokenLogprob
class ChatCompletionTokenLogprob(BaseModelResponse):
    """Chat completion token log probability information."""

    token: str = Field(description="The token.")
    bytes: list[int] | None = Field(
        default=None,
        description="UTF-8 byte representation of the token. Can be null if unavailable.",
    )
    logprob: float = Field(
        description="Log probability if in top 20 tokens, otherwise -9999.0."
    )
    top_logprobs: list[TopLogprob] = Field(
        description=(
            "List of the most likely tokens and their log probability, at this token position. "
            "In rare cases, there may be fewer than the number of requested `top_logprobs` returned."
        )
    )


# Ref: openai.types.chat.chat_completion.ChoiceLogprobs
# Ref: openai.types.chat.chat_completion_chunk.ChoiceLogprobs
class ChoiceLogprobs(BaseModelResponse):
    """Log probability information for the choice."""

    content: list[ChatCompletionTokenLogprob] | None = Field(
        default=None,
        description="A list of message content tokens with log probability information.",
    )
    refusal: list[ChatCompletionTokenLogprob] | None = Field(
        default=None,
        description="A list of message refusal tokens with log probability information.",
    )


class _Choice(BaseModelResponse):
    """Common choice element."""

    index: int = Field(ge=0, description="Index of the choice in the list of choices.")
    finish_reason: FinishReason | None = Field(
        default=None,
        description="Reason the model stopped: `stop`, `length`, `content_filter`, `tool_calls`, or `function_call` (deprecated).",
    )
    logprobs: ChoiceLogprobs | None = Field(
        default=None, description="Log probability information for the choice."
    )


# Ref: openai.types.chat.chat_completion.Choice
class Choice(_Choice):
    """Non-streaming choice element for ChatCompletion."""

    message: ChatCompletionMessage = Field(description="Assistant message.")


# Ref: openai.types.completion_usage.CompletionTokensDetails
class CompletionTokensDetails(BaseModelResponse):
    """Breakdown of tokens used in a completion."""

    accepted_prediction_tokens: int | None = Field(
        default=None,
        description="Predicted tokens that appeared in the completion (Predicted Outputs).",
    )
    audio_tokens: int | None = Field(
        default=None, description="Audio input tokens generated by the model."
    )
    reasoning_tokens: int | None = Field(
        default=None, description="Tokens generated for reasoning."
    )
    rejected_prediction_tokens: int | None = Field(
        default=None,
        description="Predicted tokens that did not appear in the completion. Counted for billing and context limits.",
    )


# Ref: openai.types.completion_usage.PromptTokensDetails
class PromptTokensDetails(BaseModelResponse):
    """Breakdown of tokens used in the prompt."""

    audio_tokens: int | None = Field(
        default=None, description="Audio input tokens present in the prompt."
    )
    cached_tokens: int | None = Field(
        default=None, description="Cached tokens present in the prompt."
    )
    cache_write_tokens: int | None = Field(
        default=None,
        description="Extra feature: Tokens written to the prompt cache "
        "(reported by some models).",
    )


# Ref: openai.types.completion_usage.CompletionUsage
class CompletionUsage(BaseModelResponse):
    """Token usage statistics, compatible with OpenAI."""

    prompt_tokens: int = Field(description="Number of tokens in the prompt.")
    completion_tokens: int = Field(
        description="Number of tokens in the generated completion."
    )
    total_tokens: int = Field(
        description="Total number of tokens used in the request (prompt + completion)."
    )
    completion_tokens_details: CompletionTokensDetails | None = Field(
        default=None, description="Breakdown of tokens used in a completion."
    )
    prompt_tokens_details: PromptTokensDetails | None = Field(
        default=None, description="Breakdown of tokens used in the prompt."
    )


class _Completion(BaseModelResponse):
    """OpenAI-compatible chat completion object (non-streaming)."""

    id: str = Field(description="Unique ID for the chat completion.")
    created: int = Field(
        description="Unix timestamp (seconds) when the chat completion was created."
    )
    model: str = Field(description="Model used for the chat completion.")
    usage: CompletionUsage | None = Field(
        default=None, description="Usage statistics for the completion request."
    )
    service_tier: ServiceTiers | None = Field(
        default=None,
        description="Processing type: 'auto', 'priority', 'flex', 'default', 'scale', or 'reserved'.",
    )
    system_fingerprint: str | None = Field(
        default=None,
        description="Backend configuration fingerprint. Use with `seed` to check for determinism changes.",
    )
    moderation: ChatModeration | None = Field(
        default=None,
        description="Guardrail moderation results, when the request set "
        "`moderation` (non-streaming responses only).",
    )


# Ref: openai.types.chat.chat_completion.ChatCompletion
class ChatCompletion(_Completion):
    """OpenAI-compatible chat completion object (non-streaming)."""

    id: str = Field(description="Unique ID for the chat completion.")
    choices: list[Choice] = Field(
        description="List of chat completion choices. Can be multiple if `n > 1`."
    )
    object: Literal["chat.completion"] = Field(
        description="Object type. Always `chat.completion`."
    )
    metadata: Metadata | None = Field(
        default=None,
        description="Key-value pairs attached to the chat completion, echoed "
        "from the request and updatable on stored chat completions.",
    )


# Ref: openai.types.chat.chat_completion_chunk.Choice
class ChunkChoice(_Choice):
    """Streaming choice element for ChatCompletionChunk."""

    delta: ChoiceDelta = Field(description="Delta from streamed model responses.")


# Ref: openai.types.chat.chat_completion_chunk.ChatCompletionChunk
class ChatCompletionChunk(_Completion):
    """OpenAI-compatible streaming chat completion chunk."""

    id: str = Field(
        description="Unique ID for the chat completion. Same for all chunks."
    )
    choices: list[ChunkChoice] = Field(
        description="List of chat completion choices. Can be multiple if `n > 1`, or empty with `stream_options.include_usage`."
    )
    created: int = Field(
        ge=0,
        description="Unix timestamp (seconds) when the chat completion was created. Same for all chunks.",
    )
    object: Literal["chat.completion.chunk"] = Field(
        description="Object type. Always `chat.completion.chunk`."
    )


# Ref: openai.types.chat.completion_create_params.CompletionCreateParams
class CompletionCreateParams(BaseModelRequestWithExtra):
    """Create chat completion request following OpenAI API specification."""

    messages: list[ChatCompletionMessageParam] = Field(
        ...,
        min_length=1,
        description="List of messages comprising the conversation. Supports text, document, video, image, and audio depending on the model.",
    )
    model: str = Field(
        ..., min_length=1, description="Model ID to generate the response"
    )
    audio: ChatCompletionAudioParam | None = Field(
        default=None,
        description="Audio output parameters. Required when `modalities=['audio']`.",
    )
    frequency_penalty: float | None = Field(
        default=None,
        description="Penalize token repetition based on frequency. Only supported on some models.",
    )
    function_call: FunctionCallParam | None = Field(
        default=None,
        description="Deprecated. Use `tool_choice` instead. Controls which function is called.",
        deprecated=True,
    )
    functions: list[LegacyFunction] | None = Field(
        default=None,
        description="Deprecated. Use `tools` instead. List of functions the model can call.",
        deprecated=True,
    )
    logit_bias: dict[str, int] | None = Field(
        default=None,
        description="Token likelihood modification via token ID to bias mapping. Only supported on some models.",
    )
    logprobs: bool | None = Field(
        default=False,
        description="Return log probabilities of output tokens. UNSUPPORTED on this implementation.",
    )
    max_completion_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Upper bound for tokens in completion, including reasoning tokens.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        # maxTokens: Bedrock Inference parameter name.
        validation_alias=AliasChoices("max_tokens", "maxTokens"),
        description="Deprecated. Use `max_completion_tokens` instead.",
        deprecated=True,
    )
    metadata: Metadata | None = Field(
        default=None, description="Key-value pairs for filtering invocation logs."
    )
    modalities: list[OutputModalities] | None = Field(
        default=None,
        description="Output types to generate. Default: `['text']`. Audio is synthesized from text for text-only models.",
    )
    n: int | None = Field(
        default=1,
        ge=1,
        le=128,
        description="Number of completion choices to generate. n>1 with streaming is UNSUPPORTED.",
    )
    parallel_tool_calls: bool | None = Field(
        default=True,
        description="Enable parallel function calling. Accepted for every model; "
        "`false` is honored only by models able to constrain tool use, and the "
        "response reports the tool calls actually made.",
    )
    prediction: ChatCompletionPredictionContentParam | None = Field(
        default=None,
        description="Static predicted output content. UNSUPPORTED on this implementation.",
    )
    presence_penalty: float | None = Field(
        default=None,
        description="Penalize new tokens based on prior appearance. Only supported on some models.",
    )
    prompt_cache_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Cache key for similar requests. Use dot-separated 'system', 'messages', 'tools' for section-specific caching. "
        "Custom hash keys are UNSUPPORTED.",
    )
    prompt_cache_options: PromptCacheOptions | None = Field(
        default=None,
        description="Prompt caching options. `ttl` `30m` is applied as 1h "
        "unless `prompt_cache_retention` is set.",
    )
    prompt_cache_retention: PromptCacheRetention | None = Field(
        default=None,
        description="Cache retention: `in_memory` is applied as 5m, `24h` as 1h.",
    )
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        description="Reasoning effort: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. "
        "On budget-based models, calculated as a fraction of `max_completion_tokens`: "
        "`low`=0.25x, `medium`=0.5x, `high`=0.75x, `xhigh`/`max`=1x (`minimal` uses the minimal budget).",
    )
    response_format: ResponseFormat | None = Field(
        default=None,
        description="Output format. Use `json_schema` for structured outputs, `json_object` for JSON mode.",
    )
    safety_identifier: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Stable user identifier for usage policy detection. Recommend hashing username/email.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Seed for deterministic sampling. Not guaranteed. Only supported on some models.",
    )
    service_tier: ServiceTiers | None = Field(
        default=None,
        description="Processing tier: `auto` (default), `priority` (mission-critical), "
        "`flex` (cost-efficient), `default`/`scale` (standard), `reserved`.",
    )
    stop: str | list[str] | None = Field(
        default=None,
        # Aliases: stopSequences (Bedrock Inference), stop_sequences (various models).
        validation_alias=AliasChoices("stop", "stop_sequences", "stopSequences"),
        description="Stop sequences. Generated text will not contain the stop sequence.",
    )
    store: bool | None = Field(
        default=None,
        description="Persist the chat completion for later retrieval. "
        "Defaults to false on this implementation. Ignored (with a "
        "request-log warning) when streaming or when storage is not "
        "enabled on the server.",
    )
    stream_options: ChatCompletionStreamOptionsParam | None = Field(
        default=None,
        description="Streaming options. Only set when `stream: true`. Only `include_usage` is supported.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0,
        description="Sampling temperature. Higher values increase randomness, lower values increase determinism. "
        "Use `top_p` or `temperature`, not both. Only supported on some models.",
    )
    tool_choice: ChatCompletionToolChoiceOptionParam | None = Field(
        default=None,
        description="Tool selection: `none` (no tool), `auto` (model decides), `required` (must call tool), "
        "or specify a tool by name.",
    )
    tools: list[ChatCompletionToolUnionParam] | None = Field(
        default=None,
        description="List of tools the model may call (custom or function tools).",
    )
    top_logprobs: int | None = Field(
        default=None,
        ge=0,
        description="Number of most likely tokens to return at each position with log probabilities. Only supported on some models.",
    )
    top_p: float | None = Field(
        default=None,
        # topP: Bedrock Inference parameter name.
        validation_alias=AliasChoices("top_p", "topP"),
        ge=0,
        description="Nucleus sampling: considers tokens comprising top_p probability mass. Use `temperature` or `top_p`, not both. "
        "Only supported on some models.",
    )
    user: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Deprecated. Use `safety_identifier` or `prompt_cache_key` instead. End-user identifier.",
        deprecated=True,
    )
    verbosity: VerbosityLevel | None = Field(
        default=None,
        description="Response verbosity: `low`, `medium`, or `high`. UNSUPPORTED on this implementation.",
    )
    web_search_options: WebSearchOptions | None = Field(
        default=None,
        description="Web search tool options. UNSUPPORTED on this implementation.",
    )
    stream: bool = Field(
        default=False,
        description="Stream response data to client using server-sent events.",
    )
    # AWS Bedrock OpenAI Chat Completions API fields.
    amazon_bedrock_guardrail_config: AmazonBedrockGuardrailConfigParams | None = Field(
        default=None,
        alias="amazon-bedrock-guardrailConfig",
        description="Amazon Bedrock Guardrail configuration.",
    )

    # Qwen Chat Completion API fields.
    top_k: int | None = Field(
        default=None,
        ge=0,
        description="Candidate set size for sampling. Larger increases randomness, smaller increases determinism. "
        "Extra field from Qwen Chat Completion API.",
    )
    enable_thinking: bool | None = Field(
        default=None,
        description="Enable thinking/reasoning mode. Extra field from Qwen Chat Completion API.",
    )
    thinking_budget: int | None = Field(
        default=None,
        ge=0,
        description="Max thinking length in tokens. Requires `enable_thinking: true`. Default: model's max chain-of-thought length. "
        "Extra field from Qwen Chat Completion API.",
    )
    translation_options: QwenTranslationOptions | None = Field(
        default=None,
        description="Translation options (source/target languages, terms, memory, domains). Extra field from Qwen Chat Completion API. "
        "UNSUPPORTED on this implementation.",
    )

    # Moonshot Chat Completion API fields.
    thinking: MoonshotThinkingOptions | None = Field(
        default=None,
        description="Enable/disable thinking. Extra field from Moonshot Chat Completion API.",
    )

    # OpenRouter Chat Completion API fields.
    reasoning: ReasoningOptions | None = Field(
        default=None,
        description="Reasoning options, equivalent to `reasoning_effort`, `thinking_budget` and `enable_thinking`. "
        "Conflicting values are rejected. Extra field from OpenRouter Chat Completion API.",
    )
    include_reasoning: bool | None = Field(
        default=None,
        description="Include the reasoning text in the response; `false` is equivalent to `reasoning.exclude: true`. "
        "Extra field from OpenRouter Chat Completion API.",
    )

    moderation: RequestModeration | None = Field(
        default=None,
        description="Apply an AWS Bedrock guardrail to this request; results "
        "are reported in the response `moderation` field (non-streaming only).",
    )

    # Extra validations
    _UNSUPPORTED: ClassVar[frozenset[str]] = frozenset(
        {
            "logprobs",
            "prediction",
            "verbosity",
            "web_search_options",
            "translation_options",
        }
    )

    #: `reasoning` sub-field paired with the flat field carrying the same setting.
    _REASONING_EQUIVALENTS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("effort", "reasoning_effort"),
        ("max_tokens", "thinking_budget"),
        ("enabled", "enable_thinking"),
    )

    @property
    def suppress_reasoning(self) -> bool:
        """Whether the reasoning text must be kept out of the response."""
        return self.include_reasoning is False or (
            self.reasoning is not None and self.reasoning.exclude is True
        )

    @model_validator(mode="before")
    @classmethod
    def _normalize_reasoning(cls, data: Any) -> Any:  # noqa: ANN401
        """Fold the `reasoning` object onto the flat fields it duplicates.

        Args:
            data: Raw request payload.

        Returns:
            The payload, with `reasoning.effort`, `reasoning.max_tokens` and
            `reasoning.enabled` applied to `reasoning_effort`,
            `thinking_budget` and `enable_thinking`.

        Raises:
            ValueError: If a `reasoning` entry contradicts another field.
        """
        if not isinstance(data, dict):
            return data
        reasoning = data.get("reasoning")
        if isinstance(reasoning, ReasoningOptions):
            reasoning = reasoning.model_dump(exclude_none=True)
        if not isinstance(reasoning, dict):
            return data
        cls._validate_reasoning_conflicts(reasoning, data)
        data = dict(data)
        for source, target in cls._REASONING_EQUIVALENTS:
            if (value := reasoning.get(source)) is not None:
                data[target] = value
        # An explicit budget implies reasoning is on, as `thinking_budget` requires.
        if (
            reasoning.get("max_tokens") is not None
            and data.get("enable_thinking") is None
        ):
            data["enable_thinking"] = True
        return data

    @classmethod
    def _validate_reasoning_conflicts(
        cls, reasoning: JsonMapping, data: JsonMapping
    ) -> None:
        """Reject a `reasoning` object contradicting itself or another field.

        Args:
            reasoning: Raw `reasoning` object.
            data: Raw request payload carrying it.

        Raises:
            ValueError: If two entries request opposite behaviors.
        """
        effort = reasoning.get("effort")
        if effort is not None and reasoning.get("max_tokens") is not None:
            msg = "Only one of `reasoning.effort` or `reasoning.max_tokens` can be specified."
            raise ValueError(msg)
        # The flat spelling counts as well: the two are merged a moment later, so
        # checking only the object would accept one wording of a contradiction and
        # reject the other.
        effective_effort = (
            effort if effort is not None else data.get("reasoning_effort")
        )
        if reasoning.get("enabled") is False and effective_effort not in (None, "none"):
            msg = "`reasoning.effort` requires `reasoning.enabled` to be `true`."
            raise ValueError(msg)
        exclude, include = reasoning.get("exclude"), data.get("include_reasoning")
        # Compared as booleans, not for truthiness: `exclude: false` beside
        # `include_reasoning: false` is the same contradiction as the other way
        # round, and silently suppressing there would drop the reasoning entirely.
        if (
            exclude is not None
            and include is not None
            and bool(exclude) == bool(include)
        ):
            msg = "`reasoning.exclude` and `include_reasoning` request the opposite behavior."
            raise ValueError(msg)
        for source, target in cls._REASONING_EQUIVALENTS:
            value = reasoning.get(source)
            if value is not None and data.get(target) not in (None, value):
                msg = f"`reasoning.{source}` and `{target}` must have the same value."
                raise ValueError(msg)

    @model_validator(mode="after")
    def _unsupported(self) -> Self:
        """Validate unsupported or incompatible chat completion options.

        Returns:
            Self: The validated parameters instance.

        Raises:
            ValueError: If incompatible options are provided (e.g., n>1 with stream, conflicting tools).
            UnsupportedParameterError: If a request parameter marked as unsupported is used.
        """
        if self.n is not None and self.n != 1 and self.stream is True:
            msg = "Multiple choices (n>1) are not supported with streaming enabled on this backend."
            raise ValueError(msg)
        self._validate_audio_modalities()
        if self.functions is not None and self.tools is not None:
            msg = "Only one of `functions` or `tools` can be specified. `functions` is deprecated."
            raise ValueError(msg)
        self._validate_tool_choice()
        self._validate_thinking_options()
        self._validate_no_custom_tools()
        self._validate_stop_sequences()
        for key in self._UNSUPPORTED & self.model_fields_set:
            # `null`/`false` request the supported default behavior, like omission
            value = getattr(self, key)
            if value is not None and value is not False:
                raise UnsupportedParameterError(key)
        return self

    def _validate_audio_modalities(self) -> None:
        """Validate audio modality options."""
        if self.modalities is not None and "audio" in self.modalities:
            if "text" not in self.modalities:
                msg = "Invalid value for 'modalities'. Only ['text'] and ['text', 'audio'] are supported."
                raise ValueError(msg)
            if self.stream:
                msg = "Audio output with streaming is not supported on this backend."
                raise ValueError(msg)
            if self.audio is None:
                msg = "`audio` parameters are required when requesting audio output modality."
                raise ValueError(msg)

    def _validate_tool_choice(self) -> None:
        """Validate tool_choice parameter restrictions.

        ``tool_choice='none'`` is accepted: the adapter omits the tool config so
        the model behaves as if no tools were passed (OpenAI ``none`` semantics).
        """
        if isinstance(self.tool_choice, ChatCompletionAllowedToolChoiceParam):
            msg = "`allowed_tools` tool_choice is not supported on this backend."
            raise ValueError(msg)  # noqa: TRY004

    def _validate_thinking_options(self) -> None:
        """Validate thinking budget and reasoning effort options."""
        if self.thinking_budget is not None and self.reasoning_effort is not None:
            msg = (
                "Only one of `thinking_budget` or `reasoning_effort` can be specified."
            )
            raise ValueError(msg)
        if self.thinking_budget is not None and not self.enable_thinking:
            msg = "`thinking_budget` requires `enable_thinking` to be set to `true` ."
            raise ValueError(msg)

    def _validate_no_custom_tools(self) -> None:
        """Validate that custom tools are not used."""
        if (
            any(
                isinstance(tool, ChatCompletionCustomToolParam)
                for tool in (self.tools or [])
            )
            or isinstance(self.tool_choice, ChatCompletionNamedToolChoiceCustomParam)
            or (
                isinstance(self.tool_choice, ChatCompletionAllowedToolChoiceParam)
                and any(
                    isinstance(allowed, ChatCompletionNamedToolChoiceCustomParam)
                    for allowed in self.tool_choice.allowed_tools.tools
                )
            )
        ):
            msg = "`custom` tools are not supported on this backend."
            raise ValueError(msg)

    def _validate_stop_sequences(self) -> None:
        """Validate stop sequences are not whitespace-only."""
        sequences = [self.stop] if isinstance(self.stop, str) else self.stop or []
        if any(not sequence.strip() for sequence in sequences):
            msg = "Stop sequences must contain at least one non-whitespace character."
            raise ValueError(msg)


# Ref: openai.types.chat.chat_completion_deleted.ChatCompletionDeleted
class ChatCompletionDeleted(BaseModelResponse):
    """Stored chat completion deletion confirmation."""

    id: str = Field(description="Identifier of the deleted chat completion.")
    object: Literal["chat.completion.deleted"] = Field(
        default="chat.completion.deleted",
        description="The object type, which is always `chat.completion.deleted`.",
    )
    deleted: bool = Field(
        default=True, description="Whether the chat completion was deleted."
    )


# Ref: openai.types.chat.completion_update_params.CompletionUpdateParams
class ChatCompletionUpdateParams(BaseModelRequest):
    """Request body for updating a stored chat completion."""

    metadata: Metadata | None = Field(
        description="Key-value pairs replacing the stored chat completion's "
        "metadata; `null` clears it."
    )


class ChatCompletionList(PaginatedListEnvelope):
    """Paginated list of stored chat completions."""

    object: Literal["list"] = Field(
        default="list", description="The object type, which is always `list`."
    )
    data: list[ChatCompletion] = Field(description="Stored chat completions.")


class ChatCompletionStoreMessageList(PaginatedListEnvelope):
    """Paginated list of the input messages of a stored chat completion."""

    object: Literal["list"] = Field(
        default="list", description="The object type, which is always `list`."
    )
    data: list[JsonMapping] = Field(
        description="Input messages of the stored chat completion."
    )
