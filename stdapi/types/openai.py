"""Local OpenAI-compatible common types."""

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, JsonValue, StringConstraints

from stdapi.types import BaseModelRequest, BaseModelResponse

# Constrained string aliases
NameStr = Annotated[
    str, StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
]
UrlStr = Annotated[
    str, StringConstraints(min_length=1, pattern=r"^(?:https?://|s3://|data:).+$")
]
Base64Str = Annotated[
    str, StringConstraints(min_length=1, pattern=r"^[A-Za-z0-9+/=\s]+$")
]

# Repeated single-value literal aliases
TextLiteral = Literal["text"]
FunctionLiteral = Literal["function"]
CustomLiteral = Literal["custom"]
AssistantRoleLiteral = Literal["assistant"]
Auto = Literal["auto"]

#: Arbitrary metadata key/value mapping attached to requests.
Metadata = dict[str, str]


# Ref: openai.types.shared_params.function_parameters
FunctionParameters = dict[str, JsonValue]


class _Strict(BaseModelRequest):
    """Function tool definition following OpenAI shared schema."""

    strict: bool | None = Field(
        default=None,
        description=(
            "Whether to enable strict schema adherence when generating the function call.\n"
            "If set to true, the model will follow the exact schema defined in the `parameters` field. "
            "Only a subset of JSON Schema is supported when `strict` is `true`."
            "Learn more about Structured Outputs in the "
            "[function calling guide](https://platform.openai.com/docs/guides/function-calling)."
        ),
    )


# Ref: openai.types.chat.completion_create_params.Function
class LegacyFunction(BaseModelRequest):
    """Legacy function definition (deprecated in favor of tools)."""

    name: NameStr = Field(description="The name of the function to be called.")
    description: str | None = Field(
        default=None,
        description="A description of what the function does, used by the model to choose when and how to call the function.",
    )
    parameters: dict[str, JsonValue] | None = Field(
        default=None,
        description="The parameters the functions accepts, described as a JSON Schema object.\n"
        "See the [guide](https://platform.openai.com/docs/guides/function-calling) for examples, and the "
        "[JSON Schema reference](https://json-schema.org/understanding-json-schema/) for documentation about the format.\n"
        "Omitting `parameters` defines a function with an empty parameter list.",
    )


# Ref: openai.types.shared_params.function_definition.FunctionDefinition
class FunctionDefinition(LegacyFunction, _Strict):
    """Function tool definition following OpenAI shared schema."""


# Ref: openai.types.shared_params.response_format_text.ResponseFormatText
class ResponseFormatText(BaseModelResponse):
    """The type of response format being defined."""

    type: TextLiteral = Field(
        default="text",
        description="The type of response format being defined. Always `text`.",
    )


# Ref: openai.types.shared_params.response_format_json_object.ResponseFormatJSONObject
class ResponseFormatJSONObject(BaseModelResponse):
    """The type of response format being defined."""

    type: Literal["json_object"] = Field(
        default="json_object",
        description="The type of response format being defined. Always `json_object`.",
    )


# Ref: openai.types.shared_params.response_format_json_schema.JSONSchema
class JSONSchema(_Strict):
    """Structured Outputs JSON Schema options."""

    model_config = ConfigDict(populate_by_name=True)

    name: NameStr = Field(
        description=(
            "The name of the response format. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64."
        )
    )
    description: str | None = Field(
        default=None,
        description=(
            "A description of what the response format is for, used by the model to determine how to respond in the format."
        ),
    )
    schema_: dict[str, JsonValue] = Field(
        alias="schema",
        serialization_alias="schema",
        description=(
            "The schema for the response format, described as a JSON Schema object. Learn how to build JSON schemas [here](https://json-schema.org/)."
        ),
    )


# Ref: openai.types.shared_params.response_format_json_schema.ResponseFormatJSONSchema
class ResponseFormatJSONSchema(BaseModelResponse, _Strict):
    """Structured Outputs configuration options, including a JSON Schema.

    Attributes:
        type: Must be "json_schema".
        json_schema: Structured Outputs JSON Schema configuration.
    """

    type: Literal["json_schema"] = Field(
        default="json_schema",
        description="The type of response format being defined. Always `json_schema`.",
    )
    json_schema: JSONSchema = Field(
        description="Structured Outputs configuration options, including a JSON Schema."
    )
