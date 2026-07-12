"""OpenAI Moderations API types."""

from typing import Literal

from pydantic import Field

from stdapi.input_file import IngestInputFile
from stdapi.types import BaseModelRequest, BaseModelResponse


class ModerationImageURL(BaseModelRequest):
    """Image container for an image moderation input."""

    url: IngestInputFile = Field(
        description="Image as an HTTPS URL, data URI, base64 string, or S3 URI."
    )


class ModerationImageURLInput(BaseModelRequest):
    """An image input to a moderation request."""

    type: Literal["image_url"] = Field(description="Input type. Always `image_url`.")
    image_url: ModerationImageURL = Field(description="The image to classify.")


class ModerationTextInput(BaseModelRequest):
    """A text input to a moderation request."""

    type: Literal["text"] = Field(description="Input type. Always `text`.")
    text: str = Field(description="The text to classify.")


# Ref: openai.types.moderation_create_params.ModerationCreateParams
class ModerationCreateParams(BaseModelRequest):
    """Request body for POST /v1/moderations."""

    input: str | list[str] | list[ModerationImageURLInput | ModerationTextInput] = (
        Field(
            description="Text or image inputs to classify. Each element yields "
            "one result."
        )
    )
    model: str | None = Field(
        default=None,
        min_length=1,
        max_length=2048,
        description="The moderation model: an AWS Bedrock guardrail "
        "(`amazon.bedrock-runtime-guardrail` for the server's default "
        "guardrail, or an explicit `<id>`, `<id>:<version>`, or ARN), or "
        "`amazon.comprehend-toxicity` for Amazon Comprehend toxicity "
        "detection. Omitted resolves to the server's default moderation "
        "model; OpenAI moderation model names are aliases "
        "(`omni-moderation-*` for the default guardrail, falling back to "
        "Comprehend when none is configured; `text-moderation-*` for "
        "Comprehend).",
    )


# Ref: openai.types.moderation.Categories
class ModerationCategories(BaseModelResponse):
    """Per-category violation flags.

    Categories without a model counterpart (guardrail content filter or
    Comprehend toxicity label) are always ``false``; violations they would
    cover surface through the closest mapped category or the overall
    ``flagged`` field.
    """

    harassment: bool = Field(
        default=False,
        description="Harassing content (guardrail INSULTS filter; Comprehend "
        "HARASSMENT_OR_ABUSE and INSULT labels).",
    )
    harassment_threatening: bool = Field(
        default=False,
        alias="harassment/threatening",
        description="Threatening harassment. Always `false` (no backend counterpart).",
    )
    hate: bool = Field(
        default=False,
        description="Hateful content (guardrail HATE filter; Comprehend "
        "HATE_SPEECH label).",
    )
    hate_threatening: bool = Field(
        default=False,
        alias="hate/threatening",
        description="Threatening hate. Always `false` (no backend counterpart).",
    )
    illicit: bool = Field(
        default=False,
        description="Advice on wrongdoing (guardrail MISCONDUCT filter; no "
        "Comprehend label).",
    )
    illicit_violent: bool = Field(
        default=False,
        alias="illicit/violent",
        description="Violent wrongdoing. Always `false` (no backend counterpart).",
    )
    self_harm: bool = Field(
        default=False,
        alias="self-harm",
        description="Self-harm content. Always `false` (no backend counterpart).",
    )
    self_harm_instructions: bool = Field(
        default=False,
        alias="self-harm/instructions",
        description="Self-harm instructions. Always `false` (no backend counterpart).",
    )
    self_harm_intent: bool = Field(
        default=False,
        alias="self-harm/intent",
        description="Self-harm intent. Always `false` (no backend counterpart).",
    )
    sexual: bool = Field(
        default=False,
        description="Sexual content (guardrail SEXUAL filter; Comprehend "
        "SEXUAL label).",
    )
    sexual_minors: bool = Field(
        default=False,
        alias="sexual/minors",
        description="Sexual content involving minors. Always `false` "
        "(no backend counterpart).",
    )
    violence: bool = Field(
        default=False,
        description="Violent content (guardrail VIOLENCE filter; Comprehend "
        "VIOLENCE_OR_THREAT label).",
    )
    violence_graphic: bool = Field(
        default=False,
        alias="violence/graphic",
        description="Graphic violence (Comprehend GRAPHIC label; no guardrail filter).",
    )


# Ref: openai.types.moderation.CategoryScores
class ModerationCategoryScores(BaseModelResponse):
    """Per-category scores from guardrail confidence levels or Comprehend labels."""

    harassment: float = Field(default=0.0, description="Harassment score.")
    harassment_threatening: float = Field(
        default=0.0,
        alias="harassment/threatening",
        description="Threatening harassment score.",
    )
    hate: float = Field(default=0.0, description="Hate score.")
    hate_threatening: float = Field(
        default=0.0, alias="hate/threatening", description="Threatening hate score."
    )
    illicit: float = Field(default=0.0, description="Illicit score.")
    illicit_violent: float = Field(
        default=0.0, alias="illicit/violent", description="Violent illicit score."
    )
    self_harm: float = Field(
        default=0.0, alias="self-harm", description="Self-harm score."
    )
    self_harm_instructions: float = Field(
        default=0.0,
        alias="self-harm/instructions",
        description="Self-harm instructions score.",
    )
    self_harm_intent: float = Field(
        default=0.0, alias="self-harm/intent", description="Self-harm intent score."
    )
    sexual: float = Field(default=0.0, description="Sexual content score.")
    sexual_minors: float = Field(
        default=0.0, alias="sexual/minors", description="Sexual minors score."
    )
    violence: float = Field(default=0.0, description="Violence score.")
    violence_graphic: float = Field(
        default=0.0, alias="violence/graphic", description="Graphic violence score."
    )


# Ref: openai.types.moderation.Moderation
class Moderation(BaseModelResponse):
    """Moderation result for a single input element."""

    flagged: bool = Field(
        description="Whether any policy or toxicity label flagged this input."
    )
    categories: ModerationCategories = Field(
        description="Per-category violation flags."
    )
    category_scores: ModerationCategoryScores = Field(
        description="Per-category scores from guardrail confidence levels or "
        "Comprehend label scores."
    )


# Ref: openai.types.moderation_create_response.ModerationCreateResponse
class ModerationCreateResponse(BaseModelResponse):
    """Response body for POST /v1/moderations."""

    id: str = Field(description="Unique identifier of the moderation request.")
    model: str = Field(
        description="The guardrail or Comprehend model used to classify the inputs."
    )
    results: list[Moderation] = Field(description="One result per input element.")
