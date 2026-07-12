"""OpenAI-compatible Moderations API backed by AWS Bedrock Guardrails.

This module implements the /v1/moderations endpoint following the OpenAI API
specification, classifying content with the AWS Bedrock ApplyGuardrail API.

The ``model`` parameter selects the guardrail: OpenAI moderation model names
(or an omitted model) resolve to the server's configured guardrail, while an
explicit guardrail ``<id>``, ``<id>:<version>``, or ARN is honored when
guardrail override is allowed.
"""

from asyncio import gather
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends

from stdapi.api_errors import ApiError
from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.aws import get_client
from stdapi.aws_bedrock import (
    handle_bedrock_client_error,
    map_guardrail_filters,
    resolve_guardrail_model,
)
from stdapi.config import SETTINGS
from stdapi.monitoring import REQUEST_ID, log_request_params, log_response_params
from stdapi.types.openai_moderations import (
    Moderation,
    ModerationCategories,
    ModerationCategoryScores,
    ModerationCreateParams,
    ModerationCreateResponse,
    ModerationImageURLInput,
    ModerationTextInput,
)

if TYPE_CHECKING:
    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime.client import BedrockRuntimeClient
    from types_aiobotocore_bedrock_runtime.type_defs import GuardrailContentBlockTypeDef

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=["Moderations", TAG_OPENAI]
)

#: Guardrail-supported image formats by MIME type.
_IMAGE_FORMATS: dict[str, str] = {"image/png": "png", "image/jpeg": "jpeg"}


def _guardrail_region(identifier: str) -> RegionName:
    """Return the AWS region hosting the guardrail.

    Args:
        identifier: Guardrail identifier or ARN.

    Returns:
        The region embedded in the ARN, or the primary Bedrock region.
    """
    if identifier.startswith("arn:"):
        return identifier.split(":")[3]  # type: ignore[return-value]
    return SETTINGS.aws_bedrock_regions[0]


async def _to_content_block(
    item: str | ModerationTextInput | ModerationImageURLInput,
) -> GuardrailContentBlockTypeDef:
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


async def _moderate(
    client: BedrockRuntimeClient,
    identifier: str,
    version: str,
    item: str | ModerationTextInput | ModerationImageURLInput,
) -> Moderation:
    """Classify one input element with the guardrail.

    Args:
        client: Bedrock runtime client in the guardrail's region.
        identifier: Guardrail identifier or ARN.
        version: Guardrail version.
        item: Input element to classify.

    Returns:
        The moderation result.
    """
    content = await _to_content_block(item)
    with handle_bedrock_client_error():
        response = await client.apply_guardrail(
            guardrailIdentifier=identifier,
            guardrailVersion=version,
            source="INPUT",
            content=[content],
        )
    categories, scores, intervened = map_guardrail_filters(
        response.get("assessments", ())
    )
    return Moderation(
        flagged=response.get("action") == "GUARDRAIL_INTERVENED"
        or intervened
        or any(categories.values()),
        categories=ModerationCategories(**categories),
        category_scores=ModerationCategoryScores(**scores),
    )


@router.post(
    "/moderations",
    summary="Classify content with AWS Bedrock Guardrails (OpenAI format)",
    operation_id="openai_moderation",
    description=(
        "Classifies text and image inputs with an AWS Bedrock guardrail "
        "(OpenAI Moderations API).\n\n"
        "Each input element is evaluated independently and yields one result. "
        "Guardrail content filters map to the OpenAI moderation categories "
        "(`HATE`→`hate`, `INSULTS`→`harassment`, `SEXUAL`→`sexual`, "
        "`VIOLENCE`→`violence`, `MISCONDUCT`→`illicit`); other guardrail "
        "policies (denied topics, word filters, sensitive information) "
        "surface through the overall `flagged` field.\n\n"
        "**Selecting the guardrail:** omit `model` (or pass an OpenAI "
        "moderation model name) to use the server's configured guardrail, or "
        "pass `<guardrail-id>`, `<guardrail-id>:<version>`, or a guardrail "
        "ARN when the server allows guardrail overrides.\n\n"
        "**MCP / AI agent usage:** image inputs accept a base64 string, data "
        "URI, HTTPS URL, or S3 URI in `image_url.url`."
    ),
    response_description="Moderation results, one per input element.",
    responses={400: {"description": "Invalid request or no guardrail available."}},
    response_model_exclude_none=True,
)
async def create_moderation(
    body: ModerationCreateParams, _: Annotated[None, Depends(authenticate)] = None
) -> ModerationCreateResponse:
    """Classify content against an AWS Bedrock guardrail.

    Args:
        body: Moderation request following the OpenAI Moderations spec.

    Returns:
        Moderation results, one per input element.

    Raises:
        ApiError: When no guardrail is available or an input is invalid.
    """
    log_request_params(body)
    identifier, version = resolve_guardrail_model(body.model)
    client: BedrockRuntimeClient = get_client(
        "bedrock-runtime", _guardrail_region(identifier)
    )
    items = [body.input] if isinstance(body.input, str) else list(body.input)
    results = await gather(
        *(_moderate(client, identifier, version, item) for item in items)
    )
    return log_response_params(
        ModerationCreateResponse(
            id=f"modr-{REQUEST_ID.get()}",
            model=body.model or f"{identifier}:{version}",
            results=list(results),
        )
    )
