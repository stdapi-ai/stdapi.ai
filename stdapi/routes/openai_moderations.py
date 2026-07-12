"""OpenAI-compatible Moderations API backed by AWS Bedrock Guardrails or Amazon Comprehend.

This module implements the /v1/moderations endpoint following the OpenAI API
specification, classifying content with the AWS Bedrock ApplyGuardrail API or
with Amazon Comprehend toxicity detection.

The ``model`` parameter selects the moderation model: an AWS Bedrock
guardrail (``amazon.bedrock-runtime-guardrail`` for the server's default
guardrail, or an explicit ``<id>``, ``<id>:<version>``, or ARN) or Amazon
Comprehend toxicity detection (``amazon.comprehend-toxicity``). OpenAI
moderation model names are aliases: ``omni-moderation-*`` for the default
guardrail (falling back to Comprehend when none is configured) and
``text-moderation-*`` for Comprehend.
"""

from asyncio import gather
from itertools import batched
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends

from stdapi.api_errors import ApiError
from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.aws import call_with_region_failover, get_client, service_regions
from stdapi.aws_bedrock import (
    COMPREHEND_MODERATION_MODEL,
    guardrail_region,
    handle_bedrock_client_error,
    map_guardrail_filters,
    resolve_moderation_model,
)
from stdapi.config import SETTINGS
from stdapi.models.capabilities import register_route_capability
from stdapi.models.moderation import MODERATION_MODALITY
from stdapi.monitoring import (
    REQUEST_ID,
    log_error_details,
    log_request_params,
    log_response_params,
)
from stdapi.types.openai_moderations import (
    Moderation,
    ModerationCategories,
    ModerationCategoryAppliedInputTypes,
    ModerationCategoryScores,
    ModerationCreateParams,
    ModerationCreateResponse,
    ModerationImageURLInput,
    ModerationTextInput,
)
from stdapi.usage import record_comprehend_usage

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime.client import BedrockRuntimeClient
    from types_aiobotocore_bedrock_runtime.type_defs import GuardrailContentBlockTypeDef
    from types_aiobotocore_comprehend.client import ComprehendClient
    from types_aiobotocore_comprehend.type_defs import DetectToxicContentResponseTypeDef

register_route_capability(
    "openai_moderation",
    f"{SETTINGS.openai_routes_prefix}/v1/moderations",
    "TEXT",
    MODERATION_MODALITY,
)

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=["Moderations", TAG_OPENAI]
)

#: Guardrail-supported image formats by MIME type.
_IMAGE_FORMATS: dict[str, str] = {"image/png": "png", "image/jpeg": "jpeg"}

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

#: Number of inputs classified concurrently per batch.
_INPUT_BATCH_SIZE: int = 10

#: JSON keys of all OpenAI moderation categories.
_ALL_CATEGORIES: tuple[str, ...] = tuple(
    field.alias or name
    for name, field in ModerationCategoryAppliedInputTypes.model_fields.items()
)

#: Categories whose applied input types may include images (per the OpenAI schema).
_IMAGE_CATEGORIES: frozenset[str] = frozenset(
    {
        "self-harm",
        "self-harm/instructions",
        "self-harm/intent",
        "sexual",
        "violence",
        "violence/graphic",
    }
)


def _applied_input_types(*, image: bool) -> ModerationCategoryAppliedInputTypes:
    """Build the per-category applied input types for one input element.

    Args:
        image: Whether the classified input is an image.

    Returns:
        Every category with ``["text"]`` for text inputs; for image inputs,
        ``["image"]`` for image-capable categories and ``[]`` otherwise.
    """
    return ModerationCategoryAppliedInputTypes.model_validate(
        {
            category: (["image"] if category in _IMAGE_CATEGORIES else [])
            if image
            else ["text"]
            for category in _ALL_CATEGORIES
        }
    )


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
        category_applied_input_types=_applied_input_types(
            image=isinstance(item, ModerationImageURLInput)
        ),
    )


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


async def _detect_toxicity(
    segments: tuple[str, ...],
) -> DetectToxicContentResponseTypeDef:
    """Run one Comprehend toxicity detection call with region failover.

    Args:
        segments: Text segments within the per-call API limits.

    Returns:
        The toxicity detection response.
    """

    def _detect(
        client: ComprehendClient, _region: RegionName
    ) -> Awaitable[DetectToxicContentResponseTypeDef]:
        """Start the toxicity detection call on one region's client."""
        return client.detect_toxic_content(
            TextSegments=[{"Text": segment} for segment in segments], LanguageCode="en"
        )

    response, region = await call_with_region_failover(
        "comprehend", service_regions(SETTINGS.aws_comprehend_region), _detect
    )
    record_comprehend_usage(sum(map(len, segments)), "toxicity", region=region)
    return response


async def _moderate_toxicity(
    item: str | ModerationTextInput | ModerationImageURLInput,
) -> Moderation:
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
        for batch in batched(
            _split_toxicity_segments(text), _TOXICITY_SEGMENTS_PER_CALL, strict=False
        ):
            for result in (await _detect_toxicity(batch)).get("ResultList", []):
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
        category_applied_input_types=_applied_input_types(image=False),
    )


@router.post(
    "/moderations",
    summary="Classify text and images for content moderation (OpenAI format)",
    operation_id="openai_moderation",
    description=(
        "Classifies inputs with a content moderation model "
        "(OpenAI Moderations API).\n\n"
        "Each input element is evaluated independently and yields one "
        "result, with the model's findings mapped to the OpenAI moderation "
        "categories.\n\n"
        "**Available models:**\n"
        "- **AWS Bedrock guardrail** (`amazon.bedrock-runtime-guardrail` for "
        "the server's default guardrail, or an explicit `<guardrail-id>`, "
        "`<guardrail-id>:<version>`, or guardrail ARN) — text and image "
        "inputs. Content filters map to the OpenAI categories "
        "(`HATE`→`hate`, `INSULTS`→`harassment`, `SEXUAL`→`sexual`, "
        "`VIOLENCE`→`violence`, `MISCONDUCT`→`illicit`); other guardrail "
        "policies (denied topics, word filters, sensitive information) "
        "surface through the overall `flagged` field.\n"
        "- **`amazon.comprehend-toxicity`** (Amazon Comprehend toxicity "
        "detection) — English text only, no images.\n\n"
        "Omit `model` to use the server's default moderation model. OpenAI "
        "moderation model names are accepted as aliases: `omni-moderation-*` "
        "for the default guardrail (falling back to Comprehend when none is "
        "configured) and `text-moderation-*` for Comprehend.\n\n"
        "**MCP / AI agent usage:** image inputs accept a base64 string, data "
        "URI, HTTPS URL, or S3 URI in `image_url.url`."
    ),
    response_description="Moderation results, one per input element.",
    responses={
        400: {
            "description": "Invalid input, disallowed model selection, or "
            "image input not supported by the selected model."
        }
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "text": {
                            "summary": "Classify a text",
                            "value": {"input": "Some text to classify"},
                        },
                        "image": {
                            "summary": "Classify a text and an image",
                            "value": {
                                "input": [
                                    {"type": "text", "text": "Describe this image"},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "https://example.com/photo.png"
                                        },
                                    },
                                ]
                            },
                        },
                    }
                }
            }
        }
    },
    response_model_exclude_none=True,
)
async def create_moderation(
    body: ModerationCreateParams, _: Annotated[None, Depends(authenticate)] = None
) -> ModerationCreateResponse:
    """Classify content with the selected moderation model.

    Args:
        body: Moderation request following the OpenAI Moderations spec.

    Returns:
        Moderation results, one per input element.

    Raises:
        ApiError: When the model selection is not allowed or an input is
            invalid.
    """
    log_request_params(body)
    items = [body.input] if isinstance(body.input, str) else list(body.input)
    results: list[Moderation] = []
    if (resolved := resolve_moderation_model(body.model)) is None:
        for batch in batched(items, _INPUT_BATCH_SIZE, strict=False):
            results.extend(await gather(*map(_moderate_toxicity, batch)))
        model = body.model or COMPREHEND_MODERATION_MODEL
    else:
        identifier, version = resolved
        client: BedrockRuntimeClient = get_client(
            "bedrock-runtime", guardrail_region(identifier)
        )
        for batch in batched(items, _INPUT_BATCH_SIZE, strict=False):
            results.extend(
                await gather(
                    *(_moderate(client, identifier, version, item) for item in batch)
                )
            )
        model = body.model or f"{identifier}:{version}"
    return log_response_params(
        ModerationCreateResponse(
            id=f"modr-{REQUEST_ID.get()}", model=model, results=results
        )
    )
