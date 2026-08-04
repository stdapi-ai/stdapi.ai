"""AWS Bedrock guardrail moderation model."""

from typing import TYPE_CHECKING

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.aws_bedrock import (
    COMPREHEND_MODERATION_MODEL,
    GUARDRAIL_MODERATION_MODEL,
    guardrail_region,
    handle_bedrock_client_error,
    map_guardrail_filters,
)
from stdapi.models.moderation import (
    GUARDRAIL_CHECKS_MODERATION_MODEL,
    ModerationModelBase,
    applied_input_types,
    unflagged_moderation,
)
from stdapi.types.openai_moderations import (
    Moderation,
    ModerationCategories,
    ModerationCategoryScores,
    ModerationImageURLInput,
    ModerationTextInput,
)
from stdapi.usage import record_guardrail_policy_usage

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.client import BedrockRuntimeClient
    from types_aiobotocore_bedrock_runtime.type_defs import GuardrailContentBlockTypeDef

    from stdapi.models import ModelDetails
    from stdapi.models.moderation import ModerationInput

#: Guardrail-supported image formats by MIME type.
_IMAGE_FORMATS: dict[str, str] = {"image/png": "png", "image/jpeg": "jpeg"}


async def _to_content_block(item: ModerationInput) -> GuardrailContentBlockTypeDef:
    """Convert a moderation input element to a guardrail content block.

    Args:
        item: Input element (plain string, text part, or image part).

    Returns:
        Guardrail content block.

    Raises:
        ApiError: When an image is not PNG or JPEG.
    """
    match item:
        case str():
            return {"text": {"text": item}}
        case ModerationTextInput():
            return {"text": {"text": item.text}}
        case _:
            file = item.image_url.url
            image_format = _IMAGE_FORMATS.get(await file.get_content_type())
            if image_format is None:
                msg = "'image_url' must be a PNG or JPEG image."
                raise ApiError(msg)
            return {
                "image": {
                    "format": image_format,  # type: ignore[typeddict-item]
                    "source": {"bytes": await file.to_bytes()},
                }
            }


class ModerationModel(ModerationModelBase):
    """AWS Bedrock guardrail moderation model."""

    __slots__ = ("_client", "_identifier", "_region", "_version")

    MATCHER = GUARDRAIL_MODERATION_MODEL

    def __init__(self, model_id: str, identifier: str, version: str) -> None:
        """Initialize the model for one resolved guardrail.

        Args:
            model_id: Moderation model ID reported in responses and usage records.
            identifier: Guardrail identifier or ARN.
            version: Guardrail version.
        """
        super().__init__(model_id)
        self._identifier = identifier
        self._version = version
        self._region = guardrail_region(identifier)
        self._client: BedrockRuntimeClient = get_client("bedrock-runtime", self._region)

    @classmethod
    def get_aliases(cls, all_models: dict[str, ModelDetails]) -> dict[str, str]:
        """Return the OpenAI omni moderation model aliases.

        The aliases target the default guardrail model when it is registered
        (a guardrail is configured), then the guardrail checks model when a
        configured region offers it, and fall back to Amazon Comprehend
        toxicity detection as a last resort.

        Args:
            all_models: All available models keyed by model ID.

        Returns:
            A dict mapping alias to model ID.
        """
        if GUARDRAIL_MODERATION_MODEL in all_models:
            target = GUARDRAIL_MODERATION_MODEL
        elif GUARDRAIL_CHECKS_MODERATION_MODEL in all_models:
            target = GUARDRAIL_CHECKS_MODERATION_MODEL
        else:
            target = COMPREHEND_MODERATION_MODEL
        return {"omni-moderation-latest": target, "omni-moderation-2024-09-26": target}

    async def moderate(self, item: ModerationInput) -> Moderation:
        """Classify one input element with the guardrail.

        An exactly-empty text input returns an unflagged result without an AWS
        call (OpenAI parity; ApplyGuardrail rejects empty content).

        Args:
            item: Input element to classify.

        Returns:
            The moderation result.
        """
        text: str | None
        match item:
            case str():
                text = item
            case ModerationTextInput():
                text = item.text
            case _:
                text = None
        if text == "":
            return unflagged_moderation()
        content = await _to_content_block(item)
        with handle_bedrock_client_error():
            response = await self._client.apply_guardrail(
                guardrailIdentifier=self._identifier,
                guardrailVersion=self._version,
                source="INPUT",
                content=[content],
                # FULL also returns non-flagged filter entries with their real
                # confidence, instead of omitting them (score would default to 0.0).
                outputScope="FULL",
            )
        record_guardrail_policy_usage(response.get("usage", {}), region=self._region)
        categories, scores, intervened = map_guardrail_filters(
            response.get("assessments", ())
        )
        return Moderation(
            flagged=response.get("action") == "GUARDRAIL_INTERVENED"
            or intervened
            or any(categories.values()),
            categories=ModerationCategories(**categories),
            category_scores=ModerationCategoryScores(**scores),
            category_applied_input_types=applied_input_types(
                image=isinstance(item, ModerationImageURLInput)
            ),
        )
