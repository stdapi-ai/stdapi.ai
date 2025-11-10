"""Local OpenAI-compatible chat completions types."""

from typing import Annotated, ClassVar, Literal, Self

from pydantic import AliasChoices, Field, model_validator

from stdapi.openai_exceptions import OpenaiUnsupportedParameterError
from stdapi.types import BaseModelRequest, BaseModelRequestWithExtra, BaseModelResponse
from stdapi.types.bedrock import AmazonBedrockGuardrailConfigParams
from stdapi.types.openai import (
    AssistantRoleLiteral,
    Base64Str,
    CustomLiteral,
    FunctionDefinition,
    FunctionLiteral,
    LegacyFunction,
    Metadata,
    NameStr,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
    ResponseFormatText,
    TextLiteral,
    UrlStr,
)

#: Reasoning effort selector for reasoning models.
ReasoningEffort = Literal["minimal", "low", "medium", "high"]

#: Finish reasons compatible with OpenAI.
FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]

#: Service tiers
ServiceTiers = Literal["auto", "default", "flex", "scale", "priority"]

#: Tool choice literal values used in multiple request fields (OpenAI-compatible).
ToolChoiceLiteral = Literal["none", "auto", "required"]

#: Common three-level setting used by multiple parameters (e.g., verbosity, search_context_size).
VerbosityLevel = Literal["low", "medium", "high"]

#: Supported output modalities
OutputModalities = Literal["text", "audio"]


# Ref: openai.types.chat.chat_completion_content_part_text_param.ChatCompletionContentPartTextParam
class ChatCompletionContentPartTextParam(BaseModelRequest):
    """Text message content part."""

    type: TextLiteral = Field(description="Content part type. Always `text`.")
    text: str = Field(description="Text content of the message part.")


# Ref: openai.types.chat.chat_completion_content_part_refusal_param.ChatCompletionContentPartRefusalParam
class ChatCompletionContentPartRefusalParam(BaseModelRequest):
    """Refusal message content part."""

    type: Literal["refusal"] = Field(description="Content part type. Always `refusal`.")
    refusal: str = Field(description="Refusal content text.")


# Ref: openai.types.chat.chat_completion_content_part_image.ImageURL
class ImageURL(BaseModelRequest):
    """Image URL detail for image content part."""

    url: UrlStr = Field(description="Image URL string (http(s), s3, or data URL).")
    detail: Literal["low", "high", "auto"] | None = Field(
        default=None,
        description=(
            "The resolution of the image to use. One of `low`, `high`, or `auto`."
            " The default is `auto`."
        ),
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


# Ref: openai.types.chat.chat_completion_content_part_param.FileFile
class FileFile(BaseModelRequest):
    """File content descriptor."""

    file_id: str = Field(description="The ID of an uploaded file to use as input.")
    file_data: Base64Str = Field(
        description="The base64 encoded file data, used when passing the file to the model as a string."
    )
    filename: str | None = Field(
        default=None,
        description="The name of the file, used when passing the file to the model as a string.",
    )


# Ref: openai.types.chat.chat_completion_content_part_param.File
class File(BaseModelRequest):
    """File message content part."""

    type: Literal["file"] = Field(description="Content part type. Always `file`.")
    file: FileFile = Field(description="Content descriptor containing base64 bytes.")


# Ref: openai.types.chat.chat_completion_content_part_input_audio_param.InputAudio
class InputAudio(BaseModelRequest):
    """Input audio descriptor."""

    data: str = Field(description="The base64 encoded audio data.")
    format: Literal["wav", "mp3"] = Field(
        description="The format of the encoded audio data. Currently supports `wav` and `mp3`."
    )


# Ref: openai.types.chat.chat_completion_content_part_input_audio_param.ChatCompletionContentPartInputAudioParam
class ChatCompletionContentPartInputAudioParam(BaseModelRequest):
    """Input audio message content part."""

    input_audio: InputAudio = Field(description="Descriptor containing the audio data.")
    type: Literal["input_audio"] = Field(
        description="The type of the content part. Always `input_audio`."
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
            "The content that should be matched when generating a model response. If generated tokens would match this content, the entire model response can be returned much more quickly."
        )
    )


# Ref: openai.types.chat.chat_completion_message.FunctionCall
# Ref: openai.types.chat.chat_completion_message_function_tool_call.Function
# Ref: openai.types.chat.chat_completion_message_function_tool_call_param.Function
class FunctionCall(BaseModelResponse):
    """Function tool call payload used within assistant tool calls."""

    name: NameStr = Field(description="The name of the function to call.")
    arguments: str = Field(
        description="The arguments to call the function with, as generated by the model in JSON format. "
        "Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema."
        "Validate the arguments in your code before calling your function."
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
        description="The arguments to call the function with, as generated by the model in JSON format. "
        "Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema."
        "Validate the arguments in your code before calling your function.",
    )


# Ref: openai.types.chat.chat_completion_message_custom_tool_call.Custom
# Ref: openai.types.chat.chat_completion_message_custom_tool_call_param.Custom
class CustomTool(BaseModelResponse):
    """Custom tool call payload used within assistant tool calls.

    UNSUPPORTED on this implementation.
    """

    name: NameStr = Field(
        description="The name of the custom tool to call.\nUNSUPPORTED on this implementation."
    )
    input: str = Field(
        description="The input for the custom tool call generated by the model.\nUNSUPPORTED on this implementation."
    )


# Ref: openai.types.chat.chat_completion_message_function_tool_call.ChatCompletionMessageFunctionToolCall
# Ref: openai.types.chat.chat_completion_message_function_tool_call_param.ChatCompletionMessageFunctionToolCallParam
class ChatCompletionMessageFunctionToolCall(BaseModelResponse):
    """Assistant tool call for a function tool."""

    type: FunctionLiteral = Field(
        description="The type of the tool. Always `function` for function tools."
    )
    function: FunctionCall = Field(description="The function that the model called.")
    id: str = Field(description="The ID of the tool call.")


# Ref: openai.types.chat.chat_completion_message_custom_tool_call.ChatCompletionMessageCustomToolCall
# Ref: openai.types.chat.chat_completion_message_custom_tool_call_param.ChatCompletionMessageCustomToolCallParam
class ChatCompletionMessageCustomToolCall(BaseModelResponse):
    """Assistant tool call for a custom tool.

    UNSUPPORTED on this implementation.
    """

    type: CustomLiteral = Field(
        description="The type of the tool. Always `custom` for custom tools.\nUNSUPPORTED on this implementation."
    )
    custom: CustomTool = Field(
        description="The custom tool that the model called.\nUNSUPPORTED on this implementation."
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

    type: FunctionLiteral = Field(
        description="The type of the tool. Always `function`."
    )
    function: FunctionDefinition = Field(description="Function definition.")


# Ref: openai.types.chat.chat_completion_custom_tool_param.CustomFormatText
class CustomFormatText(BaseModelRequest):
    """Unconstrained text format. Always `text`.

    UNSUPPORTED on this implementation.
    """

    type: TextLiteral = Field(
        description="Unconstrained text format. Always `text`.\nUNSUPPORTED on this implementation."
    )


# Ref: openai.types.chat.chat_completion_custom_tool_param.CustomFormatGrammarGrammar
class CustomFormatGrammarGrammar(BaseModelRequest):
    """The grammar definition and syntax for a grammar-based custom tool input.

    UNSUPPORTED on this implementation.
    """

    definition: str = Field(
        description="The grammar definition.\nUNSUPPORTED on this implementation."
    )
    syntax: Literal["lark", "regex"] = Field(
        description="The syntax of the grammar definition. One of `lark` or `regex`.\nUNSUPPORTED on this implementation."
    )


# Ref: openai.types.chat.chat_completion_custom_tool_param.CustomFormatGrammar
class CustomFormatGrammar(BaseModelRequest):
    """Grammar format. Always `grammar`.

    UNSUPPORTED on this implementation.
    """

    type: Literal["grammar"] = Field(
        description="Grammar format. Always `grammar`.\nUNSUPPORTED on this implementation."
    )
    grammar: CustomFormatGrammarGrammar = Field(
        description="Your chosen grammar.\nUNSUPPORTED on this implementation."
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

    name: NameStr = Field(
        description="The name of the custom tool, used to identify it in tool calls.\nUNSUPPORTED on this implementation."
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the custom tool, used to provide more context.\nUNSUPPORTED on this implementation.",
    )
    format: CustomFormat | None = Field(
        default=CustomFormatText(type="text"),
        description="The input format for the custom tool. Default is unconstrained text.\nUNSUPPORTED on this implementation.",
    )


# Ref: openai.types.chat.chat_completion_custom_tool_param.ChatCompletionCustomToolParam
class ChatCompletionCustomToolParam(BaseModelRequest):
    """Custom tool specification.

    UNSUPPORTED on this implementation.
    """

    type: CustomLiteral = Field(
        description="The type of the custom tool. Always `custom`.\nUNSUPPORTED on this implementation."
    )
    custom: Custom = Field(
        description="Properties of the custom tool.\nUNSUPPORTED on this implementation."
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

    name: NameStr = Field(description="The name of the function to call.")


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

    name: NameStr = Field(
        description="The name of the custom tool to call.\nUNSUPPORTED on this implementation."
    )


# Ref: openai.types.chat.chat_completion_named_tool_choice_custom_param.ChatCompletionNamedToolChoiceCustomParam
class ChatCompletionNamedToolChoiceCustomParam(BaseModelRequest):
    """Named tool choice for custom tools.

    UNSUPPORTED on this implementation.
    """

    type: CustomLiteral = Field(
        description="For custom tool calling, the type is always `custom`.\nUNSUPPORTED on this implementation."
    )
    custom: CustomToolChoice = Field(
        description="The custom tool to call by name.\nUNSUPPORTED on this implementation."
    )


ChatCompletionAllowedToolsToolsParam = Annotated[
    ChatCompletionNamedToolChoiceParam | ChatCompletionNamedToolChoiceCustomParam,
    Field(discriminator="type"),
]


# Ref: openai.types.chat.chat_completion_allowed_tools_param.ChatCompletionAllowedToolsParam
class ChatCompletionAllowedToolsParam(BaseModelRequest):
    """Allowed tools for function tools."""

    mode: Literal["auto", "required"] = Field(
        description="Constrains the tools available to the model to a pre-defined set.\n"
        "`auto` allows the model to pick from among the allowed tools and generate a message.\n"
        "`required` requires the model to call one or more of the allowed tools."
    )
    tools: list[ChatCompletionAllowedToolsToolsParam] = Field(
        description="A list of tool definitions that the model should be allowed to call."
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

    id: str = Field(
        description="Unique identifier for a previous audio response from the model."
    )


class _MessageParam(BaseModelRequest):
    """Common role message fields."""

    name: str | None = Field(
        default=None,
        description="An optional name for the participant.\n"
        "Provides the model information to differentiate between participants of the same role.",
    )


# Ref: openai.types.chat.chat_completion_assistant_message_param.ChatCompletionAssistantMessageParam
class ChatCompletionAssistantMessageParam(_MessageParam):
    """Assistant role message."""

    role: AssistantRoleLiteral = Field(
        description="The role of the messages author. Always `assistant`."
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
        description=(
            "The contents of the assistant message. Required unless `tool_calls` or `function_call` is specified."
        ),
    )
    function_call: FunctionCall | None = Field(
        default=None,
        description=(
            "Deprecated and replaced by `tool_calls`. The name and arguments of a function that should be called."
        ),
    )
    refusal: str | None = Field(
        default=None, description="The refusal message by the assistant."
    )
    tool_calls: list[ChatCompletionMessageToolCallUnion] | None = Field(
        default=None,
        description="The tool calls generated by the model, such as function calls.",
    )
    # Deepseek Chat Completion API fields.
    reasoning_content: str | list[ChatCompletionContentPartTextParam] | None = Field(
        default=None,
        description=(
            "The reasoning contents of the assistant message.\nExtra field from Deepseek Chat Completion API."
        ),
    )
    prefix: bool | None = Field(
        default=None,
        description=(
            "Set this to true to force the model to start its answer by the content of the supplied prefix in this assistant message."
            "\nUNSUPPORTED on this implementation."
        ),
    )


# Ref: openai.types.chat.chat_completion_user_message_param.ChatCompletionUserMessageParam
class ChatCompletionUserMessageParam(_MessageParam):
    """User role message."""

    role: Literal["user"] = Field(
        description="The role of the message author. Always `user`."
    )
    content: str | list[ChatCompletionContentPartParam] = Field(
        description="The contents of the user message."
    )


# Ref: openai.types.chat.chat_completion_system_message_param.ChatCompletionSystemMessageParam
class ChatCompletionSystemMessageParam(_MessageParam):
    """System role message."""

    role: Literal["system"] = Field(
        description="The role of the message author. Always `system`."
    )
    content: str | list[ChatCompletionContentPartTextParam] = Field(
        description="The contents of the system message."
    )


# Ref: openai.types.chat.chat_completion_developer_message_param.ChatCompletionDeveloperMessageParam
class ChatCompletionDeveloperMessageParam(_MessageParam):
    """Developer role message."""

    role: Literal["developer"] = Field(
        description="The role of the message author. Always `developer`."
    )
    content: str | list[ChatCompletionContentPartTextParam] = Field(
        description="The contents of the developer message."
    )


# Ref: openai.types.chat.chat_completion_tool_message_param.ChatCompletionToolMessageParam
class ChatCompletionToolMessageParam(BaseModelRequest):
    """Tool role message."""

    role: Literal["tool"] = Field(
        description="The role of the message author. Always `tool`."
    )
    content: str | list[ChatCompletionContentPartTextParam] = Field(
        description="String or list of text content parts."
    )
    tool_call_id: str = Field(
        description="Tool call that this message is responding to."
    )


# Ref: openai.types.chat.chat_completion_function_message_param.ChatCompletionFunctionMessageParam
class ChatCompletionFunctionMessageParam(BaseModelRequest):
    """Function role message."""

    role: FunctionLiteral = Field(
        description="The role of the message author. Always `function`."
    )
    name: NameStr = Field(description="The name of the function to call.")
    content: str | None = Field(description="The contents of the function message.")


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
        description="Specifies the output audio format. Must be one of `wav`, `aac`, `mp3`, `flac`, `opus`, or `pcm16`."
    )
    voice: str = Field(description="The voice the model uses to respond.")


# Ref: openai.types.chat.chat_completion_stream_options_param.ChatCompletionStreamOptionsParam
class ChatCompletionStreamOptionsParam(BaseModelRequest):
    """Options for streaming responses."""

    include_usage: bool = Field(
        default=False,
        description="If set, an additional chunk will be streamed before the `data: [DONE]` message.\n"
        "The `usage` field on this chunk shows the token usage statistics for the entire request, "
        "and the `choices` field will always be an empty array.\n"
        "All other chunks will also include a `usage` field, but with a null value.\n"
        "**NOTE:** If the stream is interrupted, you may not receive the final usage "
        "chunk which contains the total token usage for the request.",
    )
    include_obfuscation: bool = Field(
        default=False,
        description="When true, stream obfuscation will be enabled.\n"
        "Stream obfuscation adds random characters to an `obfuscation` field on streaming "
        "delta events to normalize payload sizes as a mitigation to certain side-channel attacks. "
        "These obfuscation fields are included by default, "
        "but add a small amount of overhead to the data stream. "
        "You can set `include_obfuscation` to false to optimize for bandwidth "
        "if you trust the network links between your application and the API.",
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


class QwenTranslationOptions(BaseModelRequest):
    """Translation options for Qwen models with translation capabilities.

    Configures source/target languages, custom term translations, translation
    memory, and domain hints for specialized translation tasks.
    """

    source_lang: str = Field(
        description='Full English name of the source language, or "auto" for automatic detection'
    )
    target_lang: str = Field(description="Full English name of the target language")
    terms: list[QwenTranslationTerm] | None = Field(
        default=None,
        description="Term intervention list with source and target properties for custom translations",
    )
    tm_list: list[QwenTranslationMemory] | None = Field(
        default=None,
        description="Translation memory list with source and target statements",
    )
    domains: str | None = Field(
        default=None,
        description='Domain/realm hint statement in English (e.g., "medical", "legal", "technical")',
    )


# Ref: openai.types.chat.completion_create_params.WebSearchOptionsUserLocationApproximate
class WebSearchOptionsUserLocationApproximate(BaseModelResponse):
    """Approximate user location for web search.

    UNSUPPORTED on this implementation.
    """

    city: str | None = Field(
        default=None,
        description="Free text input for the city of the user, e.g. `San Francisco`."
        "\nUNSUPPORTED on this implementation.",
    )
    country: str | None = Field(
        default=None,
        description="The two-letter ISO country code of the user, e.g. `US`."
        "\nUNSUPPORTED on this implementation.",
    )
    region: str | None = Field(
        default=None,
        description="Free text input for the region of the user, e.g. `California`."
        "\nUNSUPPORTED on this implementation.",
    )
    timezone: str | None = Field(
        default=None,
        description="The IANA timezone of the user, e.g. `America/Los_Angeles`."
        "\nUNSUPPORTED on this implementation.",
    )


# Ref: openai.types.chat.completion_create_params.WebSearchOptionsUserLocation
class WebSearchOptionsUserLocation(BaseModelResponse):
    """User location parameters for web search (approximate).

    UNSUPPORTED on this implementation.
    """

    type: Literal["approximate"] = Field(
        description="The type of location approximation. Always `approximate`.\nUNSUPPORTED on this implementation."
    )
    approximate: WebSearchOptionsUserLocationApproximate = Field(
        description="Approximate location parameters for the search.\nUNSUPPORTED on this implementation."
    )


# Ref: openai.types.chat.completion_create_params.WebSearchOptions
class WebSearchOptions(BaseModelResponse):
    """Web search tool options.

    UNSUPPORTED on this implementation.
    """

    search_context_size: Literal["low", "medium", "high"] = Field(
        default="medium",
        description=(
            "High level guidance for the amount of context window space to use for the search. "
            "One of `low`, `medium`, or `high`. `medium` is the default."
            "\nUNSUPPORTED on this implementation."
        ),
    )
    user_location: WebSearchOptionsUserLocation | None = Field(
        default=None,
        description="Approximate location parameters for the search."
        "\nUNSUPPORTED on this implementation.",
    )


# Ref: openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCall
class ChoiceDeltaToolCall(BaseModelResponse):
    """Tool call delta information for streaming chunks."""

    index: int = Field(ge=0, description="Position index of the tool call.")
    type: FunctionLiteral = Field(
        description="The type of the tool. Currently, only `function` is supported."
    )
    function: ChoiceDeltaFunctionCall | None = Field(
        default=None, description="Partial function call details."
    )
    id: str | None = Field(default=None, description="The ID of the tool call.")


# Ref: openai.types.chat.chat_completion_chunk.ChoiceDelta
class ChoiceDelta(BaseModelResponse):
    """Delta updates for a streaming choice."""

    content: str | None = Field(
        default=None, description="The contents of the chunk message."
    )
    function_call: ChoiceDeltaFunctionCall | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated and replaced by `tool_calls`.\n"
        "The name and arguments of a function that should be called, as generated by the model.",
    )
    refusal: str | None = Field(
        default=None, description="The refusal message generated by the model."
    )
    role: Literal["developer", "system", "user", "assistant", "tool"] | None = Field(
        default=None, description="The role of the author of this message."
    )
    tool_calls: list[ChoiceDeltaToolCall] | None = Field(
        default=None, description="Partial tool call entries."
    )
    # Deepseek Chat Completion API fields.
    reasoning_content: str | None = Field(
        default=None,
        description="The reasoning contents of the chunk message.\nExtra field from Deepseek Chat Completion API.",
    )


# Ref: openai.types.chat.chat_completion_message.AnnotationURLCitation
class AnnotationURLCitation(BaseModelResponse):
    """A URL citation when using web search."""

    end_index: int = Field(
        ge=0,
        description="The index of the last character of the URL citation in the message.",
    )
    start_index: int = Field(
        ge=0,
        description="The index of the first character of the URL citation in the message.",
    )
    title: str = Field(description="The title of the web resource.")
    url: str = Field(description="The URL of the web resource.")


# Ref: openai.types.chat.chat_completion_message.Annotation
class Annotation(BaseModelResponse):
    """Annotation for the message when using web search."""

    type: Literal["url_citation"] = Field(
        description="The type of the URL citation. Always `url_citation`."
    )
    url_citation: AnnotationURLCitation = Field(
        description="A URL citation when using web search."
    )


# Ref: openai.types.chat.chat_completion_audio.ChatCompletionAudio
class ChatCompletionAudio(BaseModelResponse):
    """If audio output modality is requested, contains data about the audio response."""

    id: str = Field(description="Unique identifier for this audio response.")
    data: str = Field(
        description=(
            "Base64 encoded audio bytes generated by the model, in the format specified in the request."
        )
    )
    expires_at: int = Field(
        ge=0,
        description=(
            "The Unix timestamp (in seconds) for when this audio response will no longer be accessible on the server."
        ),
    )
    transcript: str = Field(
        description="Transcript of the audio generated by the model."
    )


# Ref: openai.types.chat.chat_completion_message.ChatCompletionMessage
class ChatCompletionMessage(BaseModelResponse):
    """Assistant message object in the non-streaming ChatCompletion."""

    role: AssistantRoleLiteral = Field(
        description="The role of the author of this message. Always `assistant`."
    )
    content: str | None = Field(
        default=None, description="The contents of the message."
    )
    refusal: str | None = Field(
        default=None, description="The refusal message generated by the model."
    )
    annotations: list[Annotation] | None = Field(
        default=None,
        description=(
            "Annotations for the message, when applicable, as when using the web search tool."
        ),
    )
    audio: ChatCompletionAudio | None = Field(
        default=None,
        description=(
            "If the audio output modality is requested, contains data about the audio response from the model."
        ),
    )
    function_call: FunctionCall | None = Field(
        default=None,
        deprecated=True,
        description=(
            "Deprecated and replaced by `tool_calls`. The name and arguments of a function that should be called."
        ),
    )
    tool_calls: list[ChatCompletionMessageToolCallUnion] | None = Field(
        default=None,
        description="The tool calls generated by the model, such as function calls.",
    )
    # Deepseek Chat Completion API fields.
    reasoning_content: str | None = Field(
        default=None,
        description="The reasoning contents of the message.\nExtra field from Deepseek Chat Completion API.",
    )


# Ref: openai.types.chat.chat_completion_token_logprob.TopLogprob
class TopLogprob(BaseModelResponse):
    """Top log probability token information."""

    token: str = Field(description="The token.")
    bytes: list[int] | None = Field(
        default=None,
        description=(
            "A list of integers representing the UTF-8 bytes representation of the token. "
            "Useful in instances where characters are represented by multiple tokens and "
            "their byte representations must be combined to generate the correct text "
            "representation. Can be `null` if there is no bytes representation for the token."
        ),
    )
    logprob: float = Field(
        description=(
            "The log probability of this token, if it is within the top 20 most likely tokens. "
            "Otherwise, the value `-9999.0` is used to signify that the token is very unlikely."
        )
    )


# Ref: openai.types.chat.chat_completion_token_logprob.ChatCompletionTokenLogprob
class ChatCompletionTokenLogprob(BaseModelResponse):
    """Chat completion token log probability information."""

    token: str = Field(description="The token.")
    bytes: list[int] | None = Field(
        default=None,
        description=(
            "A list of integers representing the UTF-8 bytes representation of the token. "
            "Useful in instances where characters are represented by multiple tokens and "
            "their byte representations must be combined to generate the correct text "
            "representation. Can be `null` if there is no bytes representation for the token."
        ),
    )
    logprob: float = Field(
        description=(
            "The log probability of this token, if it is within the top 20 most likely tokens. "
            "Otherwise, the value `-9999.0` is used to signify that the token is very unlikely."
        )
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

    index: int = Field(
        ge=0, description="The index of the choice in the list of choices."
    )
    finish_reason: FinishReason | None = Field(
        default=None,
        description="The reason the model stopped generating tokens.\n"
        "This will be `stop` if the model hit a natural stop point or a provided stop "
        "sequence, `length` if the maximum number of tokens specified in the request was "
        "reached, `content_filter` if content was omitted due to a flag from our content "
        "filters, `tool_calls` if the model called a tool, or `function_call` "
        "(deprecated) if the model called a function.",
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
        description=(
            "When using Predicted Outputs, the number of tokens in the prediction that "
            "appeared in the completion."
        ),
    )
    audio_tokens: int | None = Field(
        default=None, description="Audio input tokens generated by the model."
    )
    reasoning_tokens: int | None = Field(
        default=None, description="Tokens generated by the model for reasoning."
    )
    rejected_prediction_tokens: int | None = Field(
        default=None,
        description=(
            "When using Predicted Outputs, the number of tokens in the prediction that did "
            "not appear in the completion. However, like reasoning tokens, these tokens are "
            "still counted in the total completion tokens for purposes of billing, output, "
            "and context window limits."
        ),
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

    id: str = Field(description="A unique identifier for the chat completion.")
    created: int = Field(
        description="The Unix timestamp (in seconds) of when the chat completion was created."
    )
    model: str = Field(description="The model used for the chat completion.")
    usage: CompletionUsage | None = Field(
        default=None, description="Usage statistics for the completion request."
    )
    service_tier: ServiceTiers | None = Field(
        default=None,
        description="Specifies the processing type used for serving the request.",
    )
    system_fingerprint: str | None = Field(
        default=None,
        description="This fingerprint represents the backend configuration that the model runs with.\n"
        "Can be used in conjunction with the `seed` request parameter to understand when "
        "backend changes have been made that might impact determinism.",
    )


# Ref: openai.types.chat.chat_completion.ChatCompletion
class ChatCompletion(_Completion):
    """OpenAI-compatible chat completion object (non-streaming)."""

    id: str = Field(description="A unique identifier for the chat completion.")
    choices: list[Choice] = Field(
        description="A list of chat completion choices. "
        "Can be more than one if `n` is greater than 1."
    )
    object: Literal["chat.completion"] = Field(
        description="The object type, which is always `chat.completion`."
    )


# Ref: openai.types.chat.chat_completion_chunk.Choice
class ChunkChoice(_Choice):
    """Streaming choice element for ChatCompletionChunk."""

    delta: ChoiceDelta = Field(
        description="A chat completion delta generated by streamed model responses."
    )


# Ref: openai.types.chat.chat_completion_chunk.ChatCompletionChunk
class ChatCompletionChunk(_Completion):
    """OpenAI-compatible streaming chat completion chunk."""

    id: str = Field(
        description="A unique identifier for the chat completion. Each chunk has the same ID."
    )
    choices: list[ChunkChoice] = Field(
        description="A list of chat completion choices.\n"
        "Can contain more than one elements if `n` is greater than 1. "
        'Can also be empty for the last chunk if you set `stream_options: {"include_usage": true}`.'
    )
    created: int = Field(
        ge=0,
        description="The Unix timestamp (in seconds) of when the chat completion was created.\n"
        "Each chunk has the same timestamp.",
    )
    object: Literal["chat.completion.chunk"] = Field(
        description="The object type, which is always `chat.completion.chunk`."
    )


# Ref: openai.types.chat.completion_create_params.CompletionCreateParams
class CompletionCreateParams(BaseModelRequestWithExtra):
    """Create chat completion request following OpenAI API specification."""

    messages: list[ChatCompletionMessageParam] = Field(
        ...,
        min_length=1,
        description="A list of messages comprising the conversation so far.\n"
        "Depending on the model you use, different message types (modalities) are supported, "
        "like text, document, video, image, and audio.\n"
        "You can include up to 20 images. "
        "Each image's size, height, and width must be no more than 3.75 MB, 8000 px, and 8000 px, respectively.\n"
        "You can include up to five documents. "
        "Each document's size must be no more than 4.5 MB.\n"
        "Audio is UNSUPPORTED on this implementation.",
    )
    model: str = Field(
        ..., min_length=1, description="Model ID used to generate the response"
    )
    audio: ChatCompletionAudioParam | None = Field(
        default=None,
        description="Parameters for audio output.\n"
        "Required when audio output is requested with modalities=['audio'].",
    )
    frequency_penalty: float | None = Field(
        default=None,
        description="Positive values penalize new tokens based on their existing frequency in the text so far, "
        "decreasing the model's likelihood to repeat the same line verbatim.\n"
        "Only supported on some models, possibles values depends on the model.",
    )
    function_call: FunctionCallParam | None = Field(
        default=None,
        description="Deprecated in favor of `tool_choice`.\n"
        "Controls which (if any) function is called by the model.\n"
        "`none` means the model will not call a function and instead generates a message.\n"
        "`auto` means the model can pick between generating a message or calling a function.\n"
        'Specifying a particular function via `{"name": "my_function"}` forces the model to call that function.\n'
        "`none` is the default when no functions are present. `auto` is the default if functions are present.",
        deprecated=True,
    )
    functions: list[LegacyFunction] | None = Field(
        default=None,
        description="Deprecated in favor of `tools`.\n"
        "A list of functions the model may generate JSON inputs for.",
        deprecated=True,
    )
    logit_bias: dict[str, int] | None = Field(
        default=None,
        description="Modify the likelihood of specified tokens appearing in the completion.\n"
        "Accepts a JSON object that maps tokens (specified by their token ID in the "
        "tokenizer) to an associated bias value. Mathematically, the "
        "bias is added to the logits generated by the model prior to sampling. The exact "
        "effect will vary per model.\n"
        "Only supported on some models, possibles values depends on the model.",
    )
    logprobs: bool | None = Field(
        default=False,
        description="Whether to return log probabilities of the output tokens or not.\n"
        "If true, returns the log probabilities of each output token returned in the "
        "`content` of `message`.\n"
        "UNSUPPORTED on this implementation.",
    )
    max_completion_tokens: int | None = Field(
        default=None,
        ge=1,
        description="An upper bound for the number of tokens that can be generated for a completion, "
        "including visible output tokens and reasoning tokens",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        # Aliases:
        # - maxTokens: Bedrock Inference parameter name
        validation_alias=AliasChoices("max_tokens", "maxTokens"),
        description="The maximum number of tokens that can be generated in the chat "
        "completion.\n"
        "This value is now deprecated in favor of `max_completion_tokens`",
        deprecated=True,
    )
    metadata: Metadata | None = Field(
        default=None,
        description="Set of key-value pairs that can be attached to an object.\n"
        "This can be useful for storing additional information about the object in a "
        "structured format.\n"
        "UNSUPPORTED on this implementation.",
    )
    modalities: list[OutputModalities] | None = Field(
        default=None,
        description="Output types that you would like the model to generate. Most models are capable "
        'of generating text, which is the default: `["text"]`\n'
        "If audio output is requested for a text-only model, "
        "the audio is synthesized via text-to-speech from the model's text output.",
    )
    n: int | None = Field(
        default=1,
        ge=1,
        le=128,
        description="How many chat completion choices to generate for each input message.\n"
        "n>1 is UNSUPPORTED on this implementation if streaming is enabled.",
    )
    parallel_tool_calls: bool | None = Field(
        default=True,
        description="Whether to enable parallel function calling during tool use.\n"
        "UNSUPPORTED on this implementation.",
    )
    prediction: ChatCompletionPredictionContentParam | None = Field(
        default=None,
        description="Static predicted output content, such as the content of a text file that is being regenerated.\n"
        "UNSUPPORTED on this implementation.",
    )
    presence_penalty: float | None = Field(
        default=None,
        description="Positive values penalize new tokens based on whether they appear in the text so "
        "far, increasing the model's likelihood to talk about new topics.\n"
        "Only supported on some models, possibles values depends on the model.",
    )
    prompt_cache_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Used to cache responses for similar requests.\n"
        "Replaces the `user` field.\n"
        "UNSUPPORTED on this implementation.",
    )
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        description="Constrains effort on reasoning for reasoning models.\n"
        "Currently supported values are `minimal`, `low`, `medium`, and `high`.\n"
        "Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response.\n"
        "Enable reasoning on opt-in reasoning models. "
        "Effort is calculated based on the `max_completion_tokens` value as follow: "
        "minimal = 0.25 x max tokens, low = 0.50 x max tokens, medium = 0.75 x max tokens, high = max tokens",
    )
    response_format: ResponseFormat | None = Field(
        default=None,
        description="An object specifying the format that the model must output.\n"
        'Setting to `{ "type": "json_schema", "json_schema": {...} }` enables Structured '
        "Outputs which ensures the model will match your supplied JSON schema.\n"
        'Setting to `{ "type": "json_object" }` enables the older JSON mode, which '
        "ensures the message the model generates is valid JSON. Using `json_schema` is "
        "preferred for models that support it.\n"
        "UNSUPPORTED on this implementation.",
    )
    safety_identifier: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="A stable identifier used to help detect users of your application that may be "
        "violating usage policies. The IDs should be a string that uniquely "
        "identifies each user. We recommend hashing their username or email address, in "
        "order to avoid sending any identifying information.",
    )
    seed: int | None = Field(
        default=None,
        ge=0,
        description="If specified, the system will make a best effort to "
        "sample deterministically, such that repeated requests with the same `seed` and "
        "parameters should return the same result. Determinism is not guaranteed.\n"
        "Only supported on some models.",
    )
    service_tier: ServiceTiers | None = Field(
        default=None,
        description="Specifies the processing type used for serving the request.\n"
        "- If set to 'auto', then the request will be processed with the default service tier.\n"
        "- If set to 'priority', then the request will be processed with the "
        "optimized latency performance configuration.\n"
        "- If set to any other value, then the request will be processed with standard "
        "processing for the selected model.\n"
        "- When not set, the default behavior is 'auto'.\n"
        "When the `service_tier` parameter is set, the response body will include the "
        "`service_tier` value based on the processing mode actually used to serve the "
        "request. This response value may be different from the value set in the "
        "parameter.",
    )
    stop: str | list[str] | None = Field(
        default=None,
        # Aliases:
        # - stopSequences: Bedrock Inference parameter name
        # - stop_sequences: Parameter name for various models
        validation_alias=AliasChoices("stop", "stop_sequences", "stopSequences"),
        description="Sequences where the API will stop generating further tokens.\n"
        "The returned text will not contain the stop sequence.",
    )
    store: bool | None = Field(
        default=None,
        description="Whether or not to store the output of this chat completion request.\n"
        "Supports text and image inputs.\n"
        "UNSUPPORTED on this implementation.",
    )
    stream_options: ChatCompletionStreamOptionsParam | None = Field(
        default=None,
        description="Options for streaming response.\n"
        "Only set this when you set `stream: true`.\n"
        "Only include_usage is supported; other sub-fields are UNSUPPORTED on this implementation.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0,
        description="What sampling temperature to use.\n"
        "Higher values like 0.8 will make the output more random, while lower values like "
        "0.2 will make it more focused and deterministic.\nWe generally recommend altering "
        "this or `top_p` but not both.\n"
        "Only supported on some models, possibles values depends on the model.",
    )
    tool_choice: ChatCompletionToolChoiceOptionParam | None = Field(
        default=None,
        description="Controls which (if any) tool is called by the model. `none` means the model will "
        "not call any tool and instead generates a message. `auto` means the model can "
        "pick between generating a message or calling one or more tools. `required` means "
        "the model must call one or more tools. Specifying a particular tool via "
        '`{"type": "function", "function": {"name": "my_function"}}` forces the model to '
        "call that tool. "
        "`none` is the default when no tools are present. `auto` is the default if tools "
        "are present.",
    )
    tools: list[ChatCompletionToolUnionParam] | None = Field(
        default=None,
        description="A list of tools the model may call.\n"
        "You can provide either custom tools or function tools.",
    )
    top_logprobs: int | None = Field(
        default=None,
        ge=0,
        description="An integer between specifying the number of most likely tokens to "
        "return at each token position, each with an associated log probability.\n"
        "Only supported on some models, possibles values depends on the model.",
    )
    top_p: float | None = Field(
        default=None,
        # Aliases:
        # - topP: Bedrock Inference parameter name
        validation_alias=AliasChoices("top_p", "topP"),
        ge=0,
        description="An alternative to sampling with temperature, called nucleus sampling, where the "
        "model considers the results of the tokens with top_p probability mass. So 0.1 "
        "means only the tokens comprising the top 10% probability mass are considered.\n"
        "We generally recommend altering this or `temperature` but not both.\n"
        "Only supported on some models, possibles values depends on the model.",
    )
    user: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="This field is being replaced by `safety_identifier` and `prompt_cache_key`. "
        "Use `prompt_cache_key` instead to maintain caching optimizations. A stable "
        "identifier for your end-users.",
        deprecated=True,
    )
    verbosity: VerbosityLevel | None = Field(
        default=None,
        description="Constrains the verbosity of the model's response.\n"
        "Lower values will result in more concise responses, while higher values will "
        "result in more verbose responses. Currently supported values are `low`, "
        "`medium`, and `high`.\n"
        "UNSUPPORTED on this implementation.",
    )
    web_search_options: WebSearchOptions | None = Field(
        default=None,
        description="This tool searches the web for relevant results to use in a response.\n"
        "UNSUPPORTED on this implementation.",
    )
    stream: bool = Field(
        default=False,
        description="If set to true, the model response data will be streamed to the client as it is "
        "generated using server-sent events.",
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
        description="The size of the candidate set for sampling during generation. "
        "For example, if the value is 50, only the 50 tokens with the highest scores "
        "in a single generation are included in the candidate set for random sampling. "
        "A larger value increases randomness, while a smaller value increases determinism. "
        "Only supported on some models, possibles values depends on the model.\n"
        "Extra field from Qwen Chat Completion API.",
    )
    enable_thinking: bool | None = Field(
        default=None,
        description="Specifies whether to enable thinking/reasoning mode. "
        "Only supported on some models.\n"
        "Extra field from Qwen Chat Completion API.",
    )
    thinking_budget: int | None = Field(
        default=None,
        ge=0,
        description="The maximum length of the thinking process in tokens. "
        "This parameter is effective only when enable_thinking is true. "
        "Only supported on some models."
        "The default value is the model's maximum chain-of-thought length.\n"
        "Extra field from Qwen Chat Completion API.",
    )
    translation_options: QwenTranslationOptions | None = Field(
        default=None,
        description="Translation options for models with translation capabilities. "
        "Configures source/target languages, custom term translations, translation memory, "
        "and domain hints for specialized translation tasks.\n"
        "Extra field from Qwen Chat Completion API.\n"
        "UNSUPPORTED on this implementation.",
    )

    # Extra validations
    _UNSUPPORTED: ClassVar[set[str]] = {
        "logprobs",
        "metadata",
        "prediction",
        "prompt_cache_key",
        "response_format",
        "store",
        "verbosity",
        "web_search_options",
        "translation_options",
    }

    @model_validator(mode="after")
    def _unsupported(self) -> Self:  # noqa: C901,PLR0912
        """Validate unsupported or incompatible chat completion options.

        Returns:
            Self: The validated parameters instance.

        Raises:
            ValueError: If incompatible options are provided (e.g., n>1 with stream, conflicting tools).
            OpenaiUnsupportedParameterError: If a request parameter marked as unsupported is used.
        """
        if self.n is not None and self.n != 1 and self.stream is True:
            msg = "Multiple choices (n>1) are not supported with streaming enabled on this backend."
            raise ValueError(msg)
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
        if self.functions is not None and self.tools is not None:
            msg = "Only one of `functions` or `tools` can be specified. `functions` is deprecated."
            raise ValueError(msg)
        if self.parallel_tool_calls is False:
            msg = "parallel_tool_calls=False is not supported on this backend."
            raise ValueError(msg)
        if isinstance(self.tool_choice, ChatCompletionAllowedToolChoiceParam):
            msg = "`allowed_tools` tool_choice is not supported on this backend."
            raise ValueError(msg)  # noqa: TRY004
        if self.thinking_budget is not None and self.reasoning_effort is not None:
            msg = (
                "Only one of `thinking_budget` or `reasoning_effort` can be specified."
            )
            raise ValueError(msg)
        if self.thinking_budget is not None and not self.enable_thinking:
            msg = "`thinking_budget` requires `enable_thinking` to be set to `true` ."
            raise ValueError(msg)
        if self.tool_choice == "none":
            msg = "`none` tool_choice is not supported on this backend."
            raise ValueError(msg)
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
        for key in self._UNSUPPORTED & self.model_fields_set:
            raise OpenaiUnsupportedParameterError(key)
        return self
