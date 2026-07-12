"""Default AWS Bedrock guardrail moderation model."""

from typing import TYPE_CHECKING

from stdapi.aws_bedrock import COMPREHEND_MODERATION_MODEL, GUARDRAIL_MODERATION_MODEL
from stdapi.models.moderation import ModerationModelBase

if TYPE_CHECKING:
    from stdapi.models import ModelDetails


class ModerationModel(ModerationModelBase):
    """Default AWS Bedrock guardrail moderation model."""

    MATCHER = GUARDRAIL_MODERATION_MODEL

    @classmethod
    def get_aliases(cls, all_models: dict[str, ModelDetails]) -> dict[str, str]:
        """Return the OpenAI omni moderation model aliases.

        The aliases target the default guardrail model when it is registered
        (a guardrail is configured) and fall back to Amazon Comprehend
        toxicity detection otherwise.

        Args:
            all_models: All available models keyed by model ID.

        Returns:
            A dict mapping alias to model ID.
        """
        target = (
            GUARDRAIL_MODERATION_MODEL
            if GUARDRAIL_MODERATION_MODEL in all_models
            else COMPREHEND_MODERATION_MODEL
        )
        return {"omni-moderation-latest": target, "omni-moderation-2024-09-26": target}
