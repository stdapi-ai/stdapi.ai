"""Amazon Comprehend toxicity detection moderation model."""

from typing import TYPE_CHECKING

from stdapi.aws_bedrock import COMPREHEND_MODERATION_MODEL
from stdapi.models.moderation import ModerationModelBase

if TYPE_CHECKING:
    from stdapi.models import ModelDetails


class ModerationModel(ModerationModelBase):
    """Amazon Comprehend toxicity detection moderation model."""

    MATCHER = COMPREHEND_MODERATION_MODEL

    @classmethod
    def get_aliases(
        cls,
        all_models: dict[str, ModelDetails],  # noqa: ARG003
    ) -> dict[str, str]:
        """Return the OpenAI text moderation model aliases.

        Args:
            all_models: All available models keyed by model ID.

        Returns:
            A dict mapping alias to model ID.
        """
        return {
            "text-moderation-latest": COMPREHEND_MODERATION_MODEL,
            "text-moderation-stable": COMPREHEND_MODERATION_MODEL,
        }
