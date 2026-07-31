"""Shared handling of the request-level ``moderation`` parameter.

The ``moderation`` parameter points a generation request at an AWS Bedrock
guardrail; the guardrail trace of the resulting Converse call is mapped back
to OpenAI-style moderation results.
"""

from typing import TYPE_CHECKING, Literal

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import (
    GUARDRAIL_CONFIG_VAR,
    GUARDRAIL_TRACE_VAR,
    is_comprehend_moderation_model,
    map_guardrail_filters,
    resolve_guardrail_model,
)
from stdapi.types.openai import (
    ChatModeration,
    ChatModerationResults,
    ModerationResult,
    RequestModeration,
    ResponseModeration,
)

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
            "The selected moderation model can only be used on the Moderations "
            "API; the 'moderation' parameter requires an Amazon Bedrock guardrail."
        )
        raise ApiError(msg)
    identifier, version = resolve_guardrail_model(moderation.model)
    GUARDRAIL_CONFIG_VAR.set(
        {
            "guardrailIdentifier": identifier,
            "guardrailVersion": version,
            "trace": "enabled_full",
        }
    )
    GUARDRAIL_TRACE_VAR.set({})


def _to_moderation_result(
    assessments: Sequence[Mapping[str, Any]], model: str
) -> ModerationResult:
    """Map guardrail trace assessments to a moderation result.

    Args:
        assessments: Guardrail assessments for one direction.
        model: The request moderation model value, echoed in the result.

    Returns:
        The moderation result; unflagged with empty categories when
        `assessments` is empty.
    """
    categories, scores, intervened = map_guardrail_filters(assessments)
    applied_types: dict[str, list[Literal["text", "image"]]] = {
        category: ["text"] for category in categories
    }
    return ModerationResult(
        flagged=intervened or any(categories.values()),
        categories=categories,
        category_scores=scores,
        category_applied_input_types=applied_types,
        model=model,
    )


def _trace_results(
    moderation: RequestModeration,
) -> tuple[ModerationResult, ModerationResult] | None:
    """Map the guardrail trace to input and output moderation results.

    Args:
        moderation: The request ``moderation`` parameter.

    Returns:
        Tuple of (input result, output result), or ``None`` when no trace is
        available (e.g. streaming responses).
    """
    if not (trace := GUARDRAIL_TRACE_VAR.get(None)):
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
    return input_result, output_result


def build_response_moderation(
    moderation: RequestModeration | None,
) -> ResponseModeration | None:
    """Build the Responses API ``moderation`` field from the guardrail trace.

    Args:
        moderation: The request ``moderation`` parameter, if any.

    Returns:
        Moderation results, or ``None`` when not requested or no trace is
        available (e.g. streaming responses).
    """
    if moderation is None or (results := _trace_results(moderation)) is None:
        return None
    input_result, output_result = results
    return ResponseModeration(input=input_result, output=output_result)


def build_chat_moderation(
    moderation: RequestModeration | None,
) -> ChatModeration | None:
    """Build the Chat Completions ``moderation`` field from the guardrail trace.

    Args:
        moderation: The request ``moderation`` parameter, if any.

    Returns:
        Moderation results, or ``None`` when not requested or no trace is
        available (e.g. streaming responses).
    """
    if moderation is None or (results := _trace_results(moderation)) is None:
        return None
    input_result, output_result = results
    return ChatModeration(
        input=ChatModerationResults(model=moderation.model, results=[input_result]),
        output=ChatModerationResults(model=moderation.model, results=[output_result]),
    )
