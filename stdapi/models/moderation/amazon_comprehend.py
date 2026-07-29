"""Amazon Comprehend toxicity detection moderation model."""

from itertools import batched
from typing import TYPE_CHECKING

from stdapi.api_errors import ApiError
from stdapi.aws import call_with_region_failover, service_regions
from stdapi.aws_bedrock import COMPREHEND_MODERATION_MODEL
from stdapi.config import SETTINGS
from stdapi.models.moderation import ModerationModelBase, applied_input_types
from stdapi.monitoring import log_error_details
from stdapi.types.openai_moderations import (
    Moderation,
    ModerationCategories,
    ModerationCategoryScores,
    ModerationTextInput,
)
from stdapi.usage import record_comprehend_usage

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_comprehend.client import ComprehendClient
    from types_aiobotocore_comprehend.type_defs import (
        DetectDominantLanguageResponseTypeDef,
        DetectToxicContentResponseTypeDef,
    )

    from stdapi.models import ModelDetails
    from stdapi.models.moderation import ModerationInput

#: Comprehend toxicity labels mapped to OpenAI moderation categories.
_TOXICITY_CATEGORIES: dict[str, str] = {
    "HATE_SPEECH": "hate",
    "HARASSMENT_OR_ABUSE": "harassment",
    "INSULT": "harassment",
    "SEXUAL": "sexual",
    "VIOLENCE_OR_THREAT": "violence",
    "GRAPHIC": "violence/graphic",
}

#: Score at or above which Comprehend toxicity results are flagged.
_TOXICITY_THRESHOLD: float = 0.5

#: Maximum UTF-8 bytes per Comprehend text segment (1 KB API limit).
_TOXICITY_SEGMENT_BYTES: int = 1_000

#: Maximum text segments per Comprehend DetectToxicContent call.
_TOXICITY_SEGMENTS_PER_CALL: int = 10

#: Languages supported by Comprehend DetectToxicContent.
_TOXICITY_LANGUAGES: frozenset[str] = frozenset(
    {"en", "es", "fr", "de", "it", "pt", "ar", "hi", "ja", "ko", "zh", "zh-TW"}
)

#: Sample size (characters) used for language detection.
_LANG_DETECT_SAMPLE_SIZE: int = 500


def _split_toxicity_segments(text: str) -> list[str]:
    """Split *text* into segments within the Comprehend per-segment size limit.

    Args:
        text: Non-empty text to split.

    Returns:
        Segments of at most ``_TOXICITY_SEGMENT_BYTES`` UTF-8 bytes each.
    """
    segments: list[str] = []
    current: list[str] = []
    size = 0
    for char in text:
        char_size = len(char.encode())
        if size + char_size > _TOXICITY_SEGMENT_BYTES:
            segments.append("".join(current))
            current, size = [], 0
        current.append(char)
        size += char_size
    segments.append("".join(current))
    return segments


async def _detect_toxicity_language(text: str) -> str:
    """Detect the dominant language of *text*, restricted to languages DetectToxicContent supports.

    Args:
        text: Full input text to detect the language from.

    Returns:
        A DetectToxicContent-supported language code, or ``"en"`` as a fallback.
    """
    sample_text = text[:_LANG_DETECT_SAMPLE_SIZE]

    def _detect(
        client: ComprehendClient, _region: RegionName
    ) -> Awaitable[DetectDominantLanguageResponseTypeDef]:
        """Start the language detection call on one region's client."""
        return client.detect_dominant_language(Text=sample_text)

    response, region = await call_with_region_failover(
        "comprehend", service_regions(SETTINGS.aws_comprehend_region), _detect
    )
    record_comprehend_usage(len(sample_text), "language-detection", region=region)
    languages = response.get("Languages") or ()
    if languages:
        detected = max(languages, key=lambda item: item.get("Score", 0.0)).get(
            "LanguageCode", ""
        )
        if detected in _TOXICITY_LANGUAGES:
            return detected
    return "en"


async def _detect_toxicity(
    segments: tuple[str, ...], language: str
) -> DetectToxicContentResponseTypeDef:
    """Run one Comprehend toxicity detection call with region failover.

    Args:
        segments: Text segments within the per-call API limits.
        language: DetectToxicContent-supported language code of the segments.

    Returns:
        The toxicity detection response.
    """

    def _detect(
        client: ComprehendClient, _region: RegionName
    ) -> Awaitable[DetectToxicContentResponseTypeDef]:
        """Start the toxicity detection call on one region's client."""
        return client.detect_toxic_content(
            TextSegments=[{"Text": segment} for segment in segments],
            LanguageCode=language,  # type: ignore[arg-type]
        )

    response, region = await call_with_region_failover(
        "comprehend", service_regions(SETTINGS.aws_comprehend_region), _detect
    )
    record_comprehend_usage(sum(map(len, segments)), "toxicity", region=region)
    return response


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

    async def moderate(self, item: ModerationInput) -> Moderation:
        """Classify one input element with Amazon Comprehend toxicity detection.

        Long inputs are split into API-sized segments and the highest score per
        category is kept.

        Args:
            item: Input element to classify.

        Returns:
            The moderation result.

        Raises:
            ApiError: When the input is an image (not supported by this model).
        """
        match item:
            case str():
                text = item
            case ModerationTextInput():
                text = item.text
            case _:
                log_error_details(
                    "Image moderation requires a guardrail "
                    "(aws_bedrock_guardrail_identifier): request rejected.",
                    level="warning",
                )
                msg = (
                    "Image moderation is not supported by the selected moderation "
                    "model. Pass an AWS Bedrock guardrail as the moderation "
                    "model, or contact the administrator to configure a default "
                    "guardrail."
                )
                raise ApiError(msg)
        scores = dict.fromkeys(_TOXICITY_CATEGORIES.values(), 0.0)
        top_score = 0.0
        if text:
            language = await _detect_toxicity_language(text)
            for batch in batched(
                _split_toxicity_segments(text),
                _TOXICITY_SEGMENTS_PER_CALL,
                strict=False,
            ):
                detection = await _detect_toxicity(batch, language)
                for result in detection.get("ResultList", []):
                    top_score = max(top_score, result.get("Toxicity", 0.0))
                    for label in result.get("Labels", []):
                        score = label.get("Score", 0.0)
                        top_score = max(top_score, score)
                        if category := _TOXICITY_CATEGORIES.get(label.get("Name", "")):
                            scores[category] = max(scores[category], score)
        return Moderation(
            flagged=top_score >= _TOXICITY_THRESHOLD,
            categories=ModerationCategories(
                **{name: score >= _TOXICITY_THRESHOLD for name, score in scores.items()}
            ),
            category_scores=ModerationCategoryScores(**scores),
            category_applied_input_types=applied_input_types(image=False),
        )
