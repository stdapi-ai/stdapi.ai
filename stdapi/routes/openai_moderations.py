"""OpenAI-compatible Moderations API backed by AWS Bedrock Guardrails or Amazon Comprehend.

The ``model`` parameter selects the moderation model: an AWS Bedrock
guardrail (``amazon.bedrock-runtime-guardrail`` for the server's default
guardrail, or an explicit ``<id>``, ``<id>:<version>``, or ARN), inline
guardrail content filter checks (``amazon.bedrock-runtime-guardrail-checks``,
no guardrail resource needed) or Amazon Comprehend toxicity detection
(``amazon.comprehend-toxicity``). OpenAI moderation model names are aliases:
``omni-moderation-*`` for the default guardrail (falling back to guardrail
checks, then Comprehend, when none is configured) and ``text-moderation-*``
for Comprehend.
"""

from asyncio import gather
from itertools import batched
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends

from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.aws_bedrock import (
    COMPREHEND_MODERATION_MODEL,
    is_comprehend_moderation_model,
    resolve_moderation_model,
)
from stdapi.config import SETTINGS
from stdapi.models.capabilities import register_route_capability
from stdapi.models.moderation import (
    GUARDRAIL_CHECKS_MODERATION_MODEL,
    MODERATION_MODALITY,
    guardrail_checks_regions,
)
from stdapi.models.moderation.amazon_bedrock_guardrail import (
    ModerationModel as GuardrailModerationModel,
)
from stdapi.models.moderation.amazon_bedrock_guardrail_checks import (
    ModerationModel as GuardrailChecksModerationModel,
)
from stdapi.models.moderation.amazon_comprehend import (
    ModerationModel as ComprehendModerationModel,
)
from stdapi.monitoring import REQUEST_ID, log_request_params, log_response_params
from stdapi.types.openai_moderations import (
    ModerationCreateParams,
    ModerationCreateResponse,
)

if TYPE_CHECKING:
    from stdapi.models.moderation import ModerationModelBase
    from stdapi.types.openai_moderations import Moderation

register_route_capability(
    "openai_moderation",
    f"{SETTINGS.openai_routes_prefix}/v1/moderations",
    "TEXT",
    MODERATION_MODALITY,
)

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1", tags=["Moderations", TAG_OPENAI]
)

#: Number of inputs classified concurrently per batch.
_INPUT_BATCH_SIZE: int = 10


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
        "- **`amazon.bedrock-runtime-guardrail-checks`** (AWS Bedrock inline "
        "guardrail checks) — text inputs only, no guardrail resource needed. "
        "Content filter checks map to the same OpenAI categories, with "
        "severity scores (`0.0` to `1.0`) reported directly. Available only "
        "when a configured Bedrock region offers the InvokeGuardrailChecks "
        "operation.\n"
        "- **`amazon.comprehend-toxicity`** (Amazon Comprehend toxicity "
        "detection) — English text only, no images. `flagged` can be true "
        "even with every category `false` and every score `0.0`: it also "
        "reflects the overall toxicity score and unmapped labels such as "
        "profanity.\n\n"
        "Omit `model` to use the server's default moderation model. OpenAI "
        "moderation model names are accepted as aliases: `omni-moderation-*` "
        "for the default guardrail (falling back to guardrail checks, then "
        "Comprehend, when none is configured) and `text-moderation-*` for "
        "Comprehend.\n\n"
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
    if body.model == GUARDRAIL_CHECKS_MODERATION_MODEL:
        model = body.model
        moderation_model: ModerationModelBase = GuardrailChecksModerationModel(model)
    elif (resolved := resolve_moderation_model(body.model)) is not None:
        identifier, version = resolved
        model = body.model or f"{identifier}:{version}"
        moderation_model = GuardrailModerationModel(model, identifier, version)
    elif (
        body.model is None or not is_comprehend_moderation_model(body.model)
    ) and guardrail_checks_regions():
        # No guardrail configured: an omitted model or an omni-moderation-*
        # alias uses guardrail checks before Comprehend as a last resort.
        model = body.model or GUARDRAIL_CHECKS_MODERATION_MODEL
        moderation_model = GuardrailChecksModerationModel(
            model, comprehend_fallback=True
        )
    else:
        model = body.model or COMPREHEND_MODERATION_MODEL
        moderation_model = ComprehendModerationModel(model)
    results: list[Moderation] = []
    for batch in batched(items, _INPUT_BATCH_SIZE, strict=False):
        results.extend(await gather(*map(moderation_model.moderate, batch)))
    return log_response_params(
        ModerationCreateResponse(
            id=f"modr-{REQUEST_ID.get()}", model=model, results=results
        )
    )
