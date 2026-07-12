"""Shared handling of the request-level ``moderation`` parameter.

The ``moderation`` parameter points a generation request at an AWS Bedrock
guardrail; the guardrail trace of the resulting Converse call is mapped back
to OpenAI-style moderation results.
"""

from typing import TYPE_CHECKING

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import (
    GUARDRAIL_TRACE_VAR,
    GUARDTRAIL_CONFIG_VAR,
    is_comprehend_moderation_model,
    map_guardrail_filters,
    resolve_guardrail_model,
)
from stdapi.types.openai import ModerationResult, RequestModeration, ResponseModeration

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any


def apply_request_moderation(moderation: RequestModeration | None) -> None:
    """Point the request's guardrail configuration at the moderation guardrail.

    Enables the guardrail trace so that moderation results can be reported in
    the response.

    Args:
        moderation: The request ``moderation`` parameter, if any.

    Raises:
        ApiError: When no guardrail is available or the override is not allowed.
    """
    if moderation is None:
        return
    if is_comprehend_moderation_model(moderation.model):
        msg = (
            "Amazon Comprehend moderation is only available on the Moderations "
            "API; the 'moderation' parameter requires an AWS Bedrock guardrail."
        )
        raise ApiError(msg)
    identifier, version = resolve_guardrail_model(moderation.model)
    GUARDTRAIL_CONFIG_VAR.set(
        {
            "guardrailIdentifier": identifier,
            "guardrailVersion": version,
            "trace": "enabled",
        }
    )
    GUARDRAIL_TRACE_VAR.set({})


def _to_moderation_result(
    assessments: Sequence[Mapping[str, Any]], model: str
) -> ModerationResult | None:
    """Map guardrail trace assessments to a moderation result.

    Args:
        assessments: Guardrail assessments for one direction.
        model: The request moderation model value, echoed in the result.

    Returns:
        The moderation result, or ``None`` without assessments.
    """
    if not assessments:
        return None
    categories, scores, intervened = map_guardrail_filters(assessments)
    return ModerationResult(
        flagged=intervened or any(categories.values()),
        categories=categories,
        category_scores=scores,
        model=model,
    )


def build_response_moderation(
    moderation: RequestModeration | None,
) -> ResponseModeration | None:
    """Build the response ``moderation`` field from the guardrail trace.

    Args:
        moderation: The request ``moderation`` parameter, if any.

    Returns:
        Moderation results, or ``None`` when not requested or no trace is
        available (e.g. streaming responses).
    """
    if moderation is None or not (trace := GUARDRAIL_TRACE_VAR.get(None)):
        return None
    input_result = _to_moderation_result(
        list(trace.get("inputAssessment", {}).values()), moderation.model
    )
    output_result = _to_moderation_result(
        [
            assessment
            for assessments in trace.get("outputAssessments", {}).values()
            for assessment in assessments
        ],
        moderation.model,
    )
    if input_result is None and output_result is None:
        return None
    return ResponseModeration(input=input_result, output=output_result)
