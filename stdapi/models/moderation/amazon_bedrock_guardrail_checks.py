"""AWS Bedrock guardrail checks (InvokeGuardrailChecks) moderation model."""

from math import ceil
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError

from stdapi.api_errors import ApiError
from stdapi.aws import call_with_region_failover
from stdapi.aws_bedrock import handle_bedrock_client_error
from stdapi.models.moderation import (
    GUARDRAIL_CHECKS_MODERATION_MODEL,
    ModerationModelBase,
    applied_input_types,
    guardrail_checks_regions,
    unflagged_moderation,
)
from stdapi.models.moderation.amazon_comprehend import (
    ModerationModel as ComprehendModerationModel,
)
from stdapi.monitoring import log_error_details
from stdapi.types.openai_moderations import (
    Moderation,
    ModerationCategories,
    ModerationCategoryScores,
    ModerationTextInput,
)
from stdapi.usage import record_guardrail_usage

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime.client import BedrockRuntimeClient
    from types_aiobotocore_bedrock_runtime.type_defs import (
        InvokeGuardrailChecksResponseTypeDef,
    )

    from stdapi.models import ModelDetails
    from stdapi.models.moderation import ModerationInput

#: Guardrail checks content filter categories mapped to OpenAI moderation categories.
_CONTENT_FILTER_CATEGORIES: dict[str, str] = {
    "HATE": "hate",
    "INSULTS": "harassment",
    "SEXUAL": "sexual",
    "VIOLENCE": "violence",
    "MISCONDUCT": "illicit",
}

#: Severity score at or above which guardrail checks results are flagged.
_SEVERITY_THRESHOLD: float = 0.5


class ModerationModel(ModerationModelBase):
    """AWS Bedrock guardrail checks (InvokeGuardrailChecks) moderation model."""

    MATCHER = GUARDRAIL_CHECKS_MODERATION_MODEL

    def __init__(self, model_id: str, *, comprehend_fallback: bool = False) -> None:
        """Initialize the model for the configured guardrail checks regions.

        Args:
            model_id: Moderation model ID reported in responses and usage records.
            comprehend_fallback: Whether to degrade to Amazon Comprehend
                toxicity detection when the ``bedrock:InvokeGuardrailChecks``
                permission is missing (default-model resolution only).

        Raises:
            ApiError: When no configured Bedrock region offers the
                InvokeGuardrailChecks operation.
        """
        super().__init__(model_id)
        self._regions: list[RegionName] = guardrail_checks_regions()
        if not self._regions:
            msg = (
                "Guardrail checks moderation is not available in the server's "
                "AWS Bedrock regions. Pass another moderation model, or "
                "contact the administrator to configure a supported region."
            )
            raise ApiError(msg)
        self._fallback = (
            ComprehendModerationModel(model_id) if comprehend_fallback else None
        )
        self._degraded = False

    @classmethod
    def get_aliases(
        cls,
        all_models: dict[str, ModelDetails],  # noqa: ARG003
    ) -> dict[str, str]:
        """Return no aliases: the omni aliases are owned by the guardrail model class.

        Args:
            all_models: All available models keyed by model ID.

        Returns:
            An empty dict.
        """
        return {}

    async def _invoke_checks(self, text: str) -> Moderation | None:
        """Run one InvokeGuardrailChecks call and map its content filter results.

        Args:
            text: Non-empty text to classify.

        Returns:
            The moderation result, or ``None`` to degrade to the Comprehend
            fallback (missing ``bedrock:InvokeGuardrailChecks`` permission).

        Raises:
            ApiError: For recognised Bedrock client errors.
        """

        def _invoke(
            client: BedrockRuntimeClient, _region: RegionName
        ) -> Awaitable[InvokeGuardrailChecksResponseTypeDef]:
            """Start the guardrail checks call on one region's client."""
            return client.invoke_guardrail_checks(
                messages=[{"role": "user", "content": [{"text": text}]}],
                checks={
                    "contentFilter": {
                        "categories": [
                            {"category": category}  # type: ignore[typeddict-item]
                            for category in _CONTENT_FILTER_CATEGORIES
                        ]
                    }
                },
            )

        try:
            response, region = await call_with_region_failover(
                "bedrock-runtime", self._regions, _invoke
            )
        except ClientError as error:
            if (
                self._fallback is not None
                and error.response["Error"]["Code"] == "AccessDeniedException"
            ):
                log_error_details(
                    "AccessDenied on InvokeGuardrailChecks (missing "
                    "bedrock:InvokeGuardrailChecks permission): falling back "
                    "to Amazon Comprehend toxicity detection.",
                    level="warning",
                )
                self._degraded = True
                return None
            with handle_bedrock_client_error():
                raise
        usage = response["usage"].get("contentFilter")
        record_guardrail_usage(
            self._model_id,
            text_units=usage["textUnits"] if usage else ceil(len(text) / 1000),
            region=region,
        )
        scores = dict.fromkeys(_CONTENT_FILTER_CATEGORIES.values(), 0.0)
        content_filter = response["results"].get("contentFilter")
        for entry in content_filter["results"] if content_filter else ():
            if category := _CONTENT_FILTER_CATEGORIES.get(entry["category"]):
                scores[category] = max(scores[category], entry["severityScore"])
        return Moderation(
            flagged=any(score >= _SEVERITY_THRESHOLD for score in scores.values()),
            categories=ModerationCategories(
                **{name: score >= _SEVERITY_THRESHOLD for name, score in scores.items()}
            ),
            category_scores=ModerationCategoryScores(**scores),
            category_applied_input_types=applied_input_types(image=False),
        )

    async def moderate(self, item: ModerationInput) -> Moderation:
        """Classify one input element with inline guardrail content filter checks.

        An exactly-empty text input returns an unflagged result without an AWS
        call (OpenAI parity, same shortcut as the guardrail backend).

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
        if text == "":
            return unflagged_moderation()
        if self._fallback is not None and self._degraded:
            return await self._fallback.moderate(item)
        result = await self._invoke_checks(text)
        if result is None:
            # _invoke_checks just set self._degraded on AccessDenied.
            return await self._fallback.moderate(item)  # type: ignore[union-attr]
        return result
