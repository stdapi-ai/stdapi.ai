"""Moderation models registry.

Moderation models are not invoked through AWS Bedrock foundation models: they
are registered as extra models so that they appear in the model listings with
their OpenAI moderation model aliases, and are served by the Moderations API.
"""

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

if TYPE_CHECKING:
    from re import Pattern

#: Output modality advertised by moderation models (classification results).
MODERATION_MODALITY: str = "MODERATION"


class ModerationModelBase(ModelBase[None, None]):
    """Base class for moderation models."""


#: Registered moderation model classes, populated by ``load_model_plugins``.
_MODEL_REGISTRY: list[tuple[str | Pattern[str], type[ModerationModelBase]]] = []


async def initialize_moderation_models() -> None:
    """Register the moderation models as extra models.

    Amazon Comprehend toxicity detection is always available; the default
    guardrail model is registered only when a guardrail is configured.
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
    if not (identifier := SETTINGS.aws_bedrock_guardrail_identifier):
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
        service="AWS Bedrock",
        input_modalities=["TEXT", "IMAGE"],
        output_modalities=[MODERATION_MODALITY],
        regions=[guardrail_region(identifier)],
    )


load_model_plugins(
    class_type=ModerationModelBase, package_name=__name__, registry=_MODEL_REGISTRY
)
