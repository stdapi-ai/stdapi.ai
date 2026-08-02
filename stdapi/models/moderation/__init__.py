"""Moderation models base classes and dynamic registry.

Moderation models are not invoked through AWS Bedrock foundation models: they
are registered as extra models so that they appear in the model listings with
their OpenAI moderation model aliases, and are served by the Moderations API
through the ``moderate`` operation of their model class.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING

from stdapi.aws import service_regions
from stdapi.aws_bedrock import (
    COMPREHEND_MODERATION_MODEL,
    GUARDRAIL_MODERATION_MODEL,
    guardrail_region,
)
from stdapi.config import SETTINGS
from stdapi.models import (
    EXTRA_MODELS,
    EXTRA_MODELS_INPUT_MODALITY,
    EXTRA_MODELS_OUTPUT_MODALITY,
    ModelBase,
    ModelDetails,
    load_model_plugins,
)
from stdapi.types.openai_moderations import (
    Moderation,
    ModerationCategories,
    ModerationCategoryAppliedInputTypes,
    ModerationCategoryScores,
    ModerationImageURLInput,
    ModerationTextInput,
)

if TYPE_CHECKING:
    from re import Pattern

    from types_aiobotocore_bedrock.literals import RegionName

#: Output modality advertised by moderation models (classification results).
MODERATION_MODALITY: str = "MODERATION"

#: Moderation model ID selecting the Bedrock InvokeGuardrailChecks backend.
GUARDRAIL_CHECKS_MODERATION_MODEL: str = "amazon.bedrock-runtime-guardrail-checks"

#: Regions offering the Bedrock InvokeGuardrailChecks operation.
GUARDRAIL_CHECKS_REGIONS: frozenset[str] = frozenset(
    {
        "us-east-1",
        "us-east-2",
        "us-west-2",
        "eu-west-2",
        "eu-north-1",
        "ap-northeast-1",
        "ap-southeast-2",
    }
)

#: A single Moderations API input element.
type ModerationInput = str | ModerationTextInput | ModerationImageURLInput

#: JSON keys of all OpenAI moderation categories.
ALL_CATEGORIES: tuple[str, ...] = tuple(
    field.alias or name
    for name, field in ModerationCategoryAppliedInputTypes.model_fields.items()
)

#: Categories whose applied input types may include images (per the OpenAI schema).
IMAGE_CATEGORIES: frozenset[str] = frozenset(
    {
        "self-harm",
        "self-harm/instructions",
        "self-harm/intent",
        "sexual",
        "violence",
        "violence/graphic",
    }
)


def guardrail_checks_regions() -> list[RegionName]:
    """Return the configured Bedrock regions offering InvokeGuardrailChecks.

    Returns:
        The candidate regions in configured priority order; empty when the
        operation is available in none of the configured Bedrock regions.
    """
    return [
        region
        for region in SETTINGS.aws_bedrock_regions
        if region in GUARDRAIL_CHECKS_REGIONS
    ]


def applied_input_types(*, image: bool) -> ModerationCategoryAppliedInputTypes:
    """Build the per-category applied input types for one input element.

    Args:
        image: Whether the classified input is an image.

    Returns:
        Every category with ``["text"]`` for text inputs; for image inputs,
        ``["image"]`` for image-capable categories and ``[]`` otherwise.
    """
    return ModerationCategoryAppliedInputTypes.model_validate(
        {
            category: (["image"] if category in IMAGE_CATEGORIES else [])
            if image
            else ["text"]
            for category in ALL_CATEGORIES
        }
    )


def unflagged_moderation(*, image: bool = False) -> Moderation:
    """Build an all-clean moderation result (no category flagged, all scores 0).

    Args:
        image: Whether the classified input is an image.

    Returns:
        The unflagged moderation result.
    """
    return Moderation(
        flagged=False,
        categories=ModerationCategories(),
        category_scores=ModerationCategoryScores(),
        category_applied_input_types=applied_input_types(image=image),
    )


class ModerationModelBase(ModelBase[None, None]):
    """Base class for provider-specific moderation models."""

    @abstractmethod
    async def moderate(self, item: ModerationInput) -> Moderation:
        """Classify one input element.

        Args:
            item: Input element (plain string, text part, or image part).

        Returns:
            The moderation result.
        """


#: Registered moderation model classes, populated by ``load_model_plugins``.
_MODEL_REGISTRY: list[tuple[str | Pattern[str], type[ModerationModelBase]]] = []


def _deregister_extra_model(model_id: str) -> None:
    """Remove a conditionally-registered moderation model from the listings.

    Args:
        model_id: Extra model ID to remove.
    """
    for modality_map in (EXTRA_MODELS_INPUT_MODALITY, EXTRA_MODELS_OUTPUT_MODALITY):
        for models in modality_map.values():
            models.discard(model_id)
    EXTRA_MODELS.pop(model_id, None)


async def initialize_moderation_models() -> None:
    """Register the moderation models as extra models.

    Amazon Comprehend toxicity detection is always available; the guardrail
    checks model is registered only when a configured Bedrock region offers
    InvokeGuardrailChecks, and the default guardrail model only when a
    guardrail is configured. Models whose condition no longer holds are
    deregistered so re-initialization reflects the current settings.
    """
    EXTRA_MODELS_INPUT_MODALITY.setdefault("TEXT", set()).add(
        COMPREHEND_MODERATION_MODEL
    )
    EXTRA_MODELS_OUTPUT_MODALITY.setdefault(MODERATION_MODALITY, set()).add(
        COMPREHEND_MODERATION_MODEL
    )
    EXTRA_MODELS[COMPREHEND_MODERATION_MODEL] = ModelDetails(
        id=COMPREHEND_MODERATION_MODEL,
        name="Comprehend Toxicity Detection",
        provider="Amazon",
        service="AWS Comprehend",
        input_modalities=["TEXT"],
        output_modalities=[MODERATION_MODALITY],
        regions=service_regions(SETTINGS.aws_comprehend_region),
    )
    if checks_regions := guardrail_checks_regions():
        EXTRA_MODELS_INPUT_MODALITY.setdefault("TEXT", set()).add(
            GUARDRAIL_CHECKS_MODERATION_MODEL
        )
        EXTRA_MODELS_OUTPUT_MODALITY.setdefault(MODERATION_MODALITY, set()).add(
            GUARDRAIL_CHECKS_MODERATION_MODEL
        )
        EXTRA_MODELS[GUARDRAIL_CHECKS_MODERATION_MODEL] = ModelDetails(
            id=GUARDRAIL_CHECKS_MODERATION_MODEL,
            name="Bedrock Guardrail Checks",
            provider="Amazon",
            service="AWS Bedrock Runtime",
            input_modalities=["TEXT"],
            output_modalities=[MODERATION_MODALITY],
            regions=checks_regions,
        )
    else:
        _deregister_extra_model(GUARDRAIL_CHECKS_MODERATION_MODEL)
    if not (identifier := SETTINGS.aws_bedrock_guardrail_identifier):
        _deregister_extra_model(GUARDRAIL_MODERATION_MODEL)
        return
    for modality in ("TEXT", "IMAGE"):
        EXTRA_MODELS_INPUT_MODALITY.setdefault(modality, set()).add(
            GUARDRAIL_MODERATION_MODEL
        )
    EXTRA_MODELS_OUTPUT_MODALITY.setdefault(MODERATION_MODALITY, set()).add(
        GUARDRAIL_MODERATION_MODEL
    )
    EXTRA_MODELS[GUARDRAIL_MODERATION_MODEL] = ModelDetails(
        id=GUARDRAIL_MODERATION_MODEL,
        name="Bedrock Guardrail",
        provider="Amazon",
        service="AWS Bedrock Runtime",
        input_modalities=["TEXT", "IMAGE"],
        output_modalities=[MODERATION_MODALITY],
        regions=[guardrail_region(identifier)],
    )


load_model_plugins(
    class_type=ModerationModelBase,  # type: ignore[type-abstract]
    package_name=__name__,
    registry=_MODEL_REGISTRY,
)
