"""AWS Bedrock guardrail checks (InvokeGuardrailChecks) moderation model."""

from math import ceil
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError

from stdapi.api_errors import ApiError
from stdapi.aws import call_with_region_failover
from stdapi.aws_bedrock import COMPREHEND_MODERATION_MODEL, handle_bedrock_client_error
from stdapi.config import SETTINGS
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
from stdapi.pricing import guardrail_policy_model
from stdapi.types.openai_moderations import (
    Moderation,
    ModerationCategories,
    ModerationCategoryScores,
    ModerationTextInput,
)
from stdapi.usage import record_guardrail_usage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping

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

#: Static InvokeGuardrailChecks content filter check payload (same on every call).
_CONTENT_FILTER_CHECK: dict[str, Any] = {
    "categories": [{"category": category} for category in _CONTENT_FILTER_CATEGORIES]
}

#: Static InvokeGuardrailChecks prompt attack check payload (every published category).
_PROMPT_ATTACK_CHECK: dict[str, Any] = {
    "categories": [
        {"category": category}
        for category in ("JAILBREAK", "PROMPT_INJECTION", "PROMPT_LEAKAGE")
    ]
}

#: Requested check name to the synthetic model its billed text units are recorded against.
_CHECK_USAGE_MODELS: dict[str, str] = {
    "promptAttack": guardrail_policy_model("checks-prompt-attack"),
    "sensitiveInformation": guardrail_policy_model("checks-sensitive-information"),
}

#: Per-category score template (zeroed), copied per call instead of rebuilt via dict.fromkeys().
_SCORE_TEMPLATE: dict[str, float] = dict.fromkeys(
    _CONTENT_FILTER_CATEGORIES.values(), 0.0
)


class ModerationModel(ModerationModelBase):
    """AWS Bedrock guardrail checks (InvokeGuardrailChecks) moderation model."""

    __slots__ = ("_checks", "_degraded", "_fallback", "_regions", "_usage_models")

    MATCHER = GUARDRAIL_CHECKS_MODERATION_MODEL

    def __init__(self, model_id: str, *, comprehend_fallback: bool = False) -> None:
        """Initialize the model for the configured guardrail checks regions.

        The content filter is always evaluated; the prompt attack and sensitive
        information checks are added when the deployment configures them, each
        billed as a check of its own.

        Args:
            model_id: Moderation model ID this backend's usage is billed against.
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
        self._checks: dict[str, Any] = {"contentFilter": _CONTENT_FILTER_CHECK}
        if SETTINGS.aws_bedrock_guardrail_checks_prompt_attack:
            self._checks["promptAttack"] = _PROMPT_ATTACK_CHECK
        if entities := SETTINGS.aws_bedrock_guardrail_checks_pii_entities:
            self._checks["sensitiveInformation"] = {
                "entities": [{"type": entity} for entity in entities]
            }
        self._usage_models = {"contentFilter": model_id} | {
            check: model
            for check in self._checks
            if (model := _CHECK_USAGE_MODELS.get(check)) is not None
        }
        self._fallback = (
            ComprehendModerationModel(COMPREHEND_MODERATION_MODEL)
            if comprehend_fallback
            else None
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
        """Run one InvokeGuardrailChecks call and map every requested check.

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
                checks=self._checks,  # type: ignore[arg-type]
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
        # The usage TypedDict is read by check name, which it cannot be indexed by.
        self._record_usage(response["usage"], len(text), region)  # type: ignore[arg-type]
        results = response["results"]
        scores = _SCORE_TEMPLATE.copy()
        content_filter = results.get("contentFilter")
        for entry in content_filter["results"] if content_filter else ():
            if category := _CONTENT_FILTER_CATEGORIES.get(entry["category"]):
                scores[category] = max(scores[category], entry["severityScore"])
        prompt_attack = results.get("promptAttack")
        sensitive = results.get("sensitiveInformation")
        return Moderation(
            # Neither extra check maps to an OpenAI category, so a detection
            # raises "flagged" alone -- as a guardrail's unmapped policies do.
            flagged=any(score >= _SEVERITY_THRESHOLD for score in scores.values())
            or any(
                entry["severityScore"] >= _SEVERITY_THRESHOLD
                for entry in (prompt_attack["results"] if prompt_attack else ())
            )
            # Only detected entities are reported, so any entry is a hit.
            or bool(sensitive and sensitive["results"]),
            categories=ModerationCategories(
                **{name: score >= _SEVERITY_THRESHOLD for name, score in scores.items()}
            ),
            category_scores=ModerationCategoryScores(**scores),
            category_applied_input_types=applied_input_types(image=False),
        )

    def _record_usage(
        self,
        usage: Mapping[str, Mapping[str, int]],
        characters: int,
        region: RegionName,
    ) -> None:
        """Record the text units every requested check billed.

        AWS prices the checks at three different rates and reports their units
        separately, so each is recorded against its own model; a check whose
        units the response omits falls back to the metering rule of one unit
        per 1,000 characters.

        Args:
            usage: The InvokeGuardrailChecks response's ``usage`` map.
            characters: Length of the classified text.
            region: Region that served the call.
        """
        fallback = ceil(characters / 1000)
        for check, model in self._usage_models.items():
            reported = usage.get(check)
            record_guardrail_usage(
                model,
                text_units=reported["textUnits"] if reported else fallback,
                region=region,
            )

    async def moderate(self, item: ModerationInput) -> Moderation:
        """Classify one input element with the configured inline guardrail checks.

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
