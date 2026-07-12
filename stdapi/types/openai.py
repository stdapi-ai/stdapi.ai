"""Local OpenAI-compatible common types."""

from typing import Literal

from pydantic import ConfigDict, Field

from stdapi.types import BaseModelRequest, BaseModelResponse, JsonMapping

# Repeated single-value literal aliases
TextLiteral = Literal["text"]
FunctionLiteral = Literal["function"]
CustomLiteral = Literal["custom"]
AssistantRoleLiteral = Literal["assistant"]
Auto = Literal["auto"]

#: Arbitrary metadata key/value mapping attached to requests.
Metadata = dict[str, str]


class _Strict(BaseModelRequest):
    """Function tool definition following OpenAI shared schema."""

    strict: bool | None = Field(
        default=None,
        description=(
            "Whether to enable strict schema adherence when generating the function call. "
            "If true, the model follows the exact schema in `parameters`; only a subset of "
            "JSON Schema is supported in that case."
        ),
    )


# Ref: openai.types.chat.completion_create_params.Function
class LegacyFunction(BaseModelRequest):
    """Legacy function definition (deprecated in favor of tools)."""

    name: str = Field(description="The name of the function to be called.")
    description: str | None = Field(
        default=None, description="A description of what the function does."
    )
    parameters: JsonMapping | None = Field(
        default=None,
        description="The parameters the function accepts, described as a JSON Schema "
        "object. Omitting `parameters` defines a function with an empty parameter list.",
    )


# Ref: openai.types.shared_params.function_definition.FunctionDefinition
class FunctionDefinition(LegacyFunction, _Strict):
    """Function tool definition following OpenAI shared schema."""


# Ref: openai.types.shared_params.response_format_text.ResponseFormatText
class ResponseFormatText(BaseModelResponse):
    """The type of response format being defined."""

    type: TextLiteral = Field(
        default="text", description="The type of response format. Always `text`."
    )


# Ref: openai.types.shared_params.response_format_json_object.ResponseFormatJSONObject
class ResponseFormatJSONObject(BaseModelResponse):
    """The type of response format being defined."""

    type: Literal["json_object"] = Field(
        default="json_object",
        description="The type of response format. Always `json_object`.",
    )


# Ref: openai.types.shared_params.response_format_json_schema.JSONSchema
class JSONSchema(_Strict):
    """Structured Outputs JSON Schema options."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(description="The name of the response format (max 64 chars).")
    description: str | None = Field(
        default=None,
        description=(
            "A description of what the response format is for, used by the model to determine how to respond in the format."
        ),
    )
    schema_: JsonMapping = Field(
        alias="schema",
        serialization_alias="schema",
        description=(
            "The schema for the response format, described as a JSON Schema object."
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
        description="The type of response format. Always `json_schema`.",
    )
    json_schema: JSONSchema = Field(
        description="Structured Outputs JSON Schema configuration."
    )


# Ref: openai.types.responses.response_create_params.Moderation
class RequestModeration(BaseModelRequest):
    """Moderation configuration applied to a generation request."""

    model: str = Field(
        min_length=1,
        max_length=2048,
        description="AWS Bedrock guardrail as `<id>`, `<id>:<version>`, or ARN; "
        "OpenAI moderation model names resolve to the server's configured "
        "guardrail.",
    )


class ModerationResult(BaseModelResponse):
    """Moderation outcome for one direction of a generation."""

    type: Literal["moderation_result"] = Field(
        default="moderation_result",
        description="The object type, which is always `moderation_result`.",
    )
    flagged: bool = Field(description="Whether the guardrail flagged the content.")
    categories: dict[str, bool] = Field(
        description="Per-category violation flags (OpenAI moderation categories)."
    )
    category_scores: dict[str, float] = Field(
        description="Per-category scores derived from guardrail confidence levels."
    )
    model: str = Field(description="The guardrail that classified the content.")


# Ref: openai.types.responses.response.Moderation
class ResponseModeration(BaseModelResponse):
    """Moderation results for the request input and generated output."""

    input: ModerationResult | None = Field(
        default=None, description="Moderation result for the request input."
    )
    output: ModerationResult | None = Field(
        default=None, description="Moderation result for the generated output."
    )
