"""Amazon Bedrock Marketplace model endpoints.

An operator deploys a Marketplace listing onto a managed endpoint in their own
account; this module discovers those endpoints and publishes them as ordinary
chat models. The server never creates, updates or deletes one -- an endpoint is
paid, capacity-bearing infrastructure whose lifecycle belongs to the operator.

Invocation needs no transport of its own: ``bedrock-runtime`` accepts the
endpoint ARN as ``modelId`` on ``Converse``, ``ConverseStream``, ``InvokeModel``
and ``InvokeModelWithResponseStream``. The wire model for those operations
enumerates the form explicitly (``ConversationalModelId`` and
``InvokeModelIdentifier`` in ``botocore/data/bedrock-runtime/*/service-2.json``
both carry ``^arn:aws:sagemaker:<region>:<account>:endpoint/<name>$``), and the
User Guide documents the same call:
https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-marketplace-call-the-endpoint.html
"""

from asyncio import gather
from typing import TYPE_CHECKING, Final

from botocore.exceptions import BotoCoreError, ClientError

from stdapi.aws import get_client, pooled_clients
from stdapi.config import SETTINGS
from stdapi.models import MARKETPLACE_ENDPOINT_MODELS, MARKETPLACE_SERVICE, ModelDetails
from stdapi.utils import match_marketplace_endpoint_arn, match_sagemaker_hub_content_arn

if TYPE_CHECKING:
    from types_aiobotocore_bedrock.client import BedrockClient
    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock.type_defs import MarketplaceModelEndpointTypeDef

#: Provider published when the listing name carries no recognisable vendor prefix.
_UNKNOWN_PROVIDER: Final = "Amazon Bedrock Marketplace"

#: Bedrock registration status of an endpoint Bedrock can map onto its own API.
_REGISTERED_STATUS: Final = "REGISTERED"

#: SageMaker endpoint statuses that serve a request, compared case-insensitively.
_SERVING_STATUSES: Final[frozenset[str]] = frozenset(
    {"inservice", "updating", "systemupdating"}
)

#: Statuses an endpoint passes through on its own, not worth reporting to the operator.
_TRANSIENT_STATUSES: Final[frozenset[str]] = frozenset({"creating", "deleting"})

#: Vendor prefixes a public-hub listing name starts with, longest first so a compound wins.
_LISTING_PROVIDERS: Final[tuple[tuple[str, str], ...]] = tuple(
    sorted(
        (
            ("ai21", "AI21 Labs"),
            ("alexaai", "Alexa"),
            ("bge", "BAAI"),
            ("cohere", "Cohere"),
            ("deepseek", "DeepSeek"),
            ("google", "Google"),
            ("huggingface", "Hugging Face"),
            ("ibm", "IBM"),
            ("jumpstart", "Amazon"),
            ("meta", "Meta"),
            ("mistral", "Mistral AI"),
            ("model", _UNKNOWN_PROVIDER),
            ("nvidia", "NVIDIA"),
            ("qwen", "Qwen"),
            ("stabilityai", "Stability AI"),
            ("tiiuae", "TII"),
            ("upstage", "Upstage"),
            ("writer", "Writer"),
        ),
        key=lambda item: -len(item[0]),
    )
)


def marketplace_endpoint_regions() -> list[RegionName]:
    """Return the regions searched for Marketplace model endpoints.

    Returns:
        The configured subset, or every Bedrock region when none is set. Empty
        when the feature is disabled.
    """
    if not SETTINGS.aws_bedrock_marketplace_endpoints_enabled:
        return []
    return SETTINGS.aws_bedrock_marketplace_endpoint_regions or [
        *SETTINGS.aws_bedrock_regions
    ]


def _listing_provider(listing: str) -> str:
    """Return the provider a public-hub listing name announces.

    Args:
        listing: The hub content name, e.g. ``huggingface-reasoning-qwen3-4b``.

    Returns:
        The provider display name, or a generic one when the prefix is unknown.
    """
    for prefix, provider in _LISTING_PROVIDERS:
        if listing.startswith(f"{prefix}-"):
            return provider
    return _UNKNOWN_PROVIDER


def _display_name(listing: str) -> str:
    """Build a human-readable model name from a public-hub listing name.

    Args:
        listing: The hub content name, e.g. ``huggingface-reasoning-qwen3-4b``.

    Returns:
        The listing name with its vendor and task prefix segments title-cased.
    """
    return " ".join(word.capitalize() for word in listing.split("-"))


def _model_from_endpoint(
    endpoint: MarketplaceModelEndpointTypeDef, region: RegionName
) -> tuple[ModelDetails | None, str]:
    """Build the catalogue entry for one discovered endpoint.

    Readiness needs both statuses: an endpoint Amazon Bedrock could not register
    refuses the invocation, and one SageMaker has not yet brought into service
    cannot answer it. ``Updating`` and ``SystemUpdating`` are served: SageMaker
    shifts traffic onto the new instances with no availability loss, so a scale
    operation would otherwise unpublish a model that answers throughout it. A
    declined endpoint says why, because a deployment that publishes nothing
    otherwise reads as a deployment with nothing to publish.

    Args:
        endpoint: A ``MarketplaceModelEndpoint`` from ``GetMarketplaceModelEndpoint``.
        region: The region the endpoint was discovered in.

    Returns:
        Tuple of (model details, reason it was declined). Exactly one is set,
        and the reason is empty for a decline the operator cannot act on -- an
        endpoint still coming up gets there by itself.
    """
    endpoint_arn = endpoint["endpointArn"]
    if (status := endpoint.get("status")) != _REGISTERED_STATUS:
        return None, (
            f"'{endpoint_arn}' is not registered with Amazon Bedrock (status "
            f"'{status or 'absent'}'), so bedrock-runtime cannot invoke it: "
            "register it again, or delete it."
        )
    # endpointStatus is an open string, not an enum: SageMaker's own statuses
    # flow through unchanged and Amazon Bedrock does not republish them.
    endpoint_status = endpoint["endpointStatus"]
    if (lowered := endpoint_status.lower()) not in _SERVING_STATUSES:
        if lowered in _TRANSIENT_STATUSES:
            return None, ""
        return None, (
            f"'{endpoint_arn}' is in status '{endpoint_status}' rather than "
            "InService, so it cannot answer an invocation."
        )
    arn_match = match_marketplace_endpoint_arn(endpoint_arn)
    if not arn_match or arn_match.group("region") != region:
        return None, (
            f"'{endpoint_arn}' is not a SageMaker endpoint ARN of {region}: "
            "bedrock-runtime invokes a model endpoint in its own region only."
        )
    source = endpoint["modelSourceIdentifier"]
    listing_match = match_sagemaker_hub_content_arn(source)
    if not listing_match:
        return None, (
            f"'{endpoint_arn}' was registered from '{source}', which is not "
            "SageMaker public-hub content, so it has no listing name to be "
            "published under. Name it by ARN instead, with "
            "AWS_BEDROCK_ALLOW_MARKETPLACE_ENDPOINT_ARN=true."
        )
    listing: str = listing_match.group("name")
    return (
        ModelDetails(
            id=listing,
            name=_display_name(listing),
            provider=_listing_provider(listing),
            service=MARKETPLACE_SERVICE,
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            # ConverseStream takes the same modelId shape as Converse, and the
            # documented policy for this path grants
            # InvokeEndpointWithResponseStream.
            response_streaming=True,
            # Batch inference takes a foundation model or an inference profile.
            batch=False,
            regions=[region],
            marketplace_endpoints={region: endpoint_arn},
        ),
        "",
    )


def _no_client_reason(region: RegionName) -> str:
    """Explain a ``bedrock`` control-plane client the pool could not supply.

    Two different faults reach here and the operator fixes them differently, so
    they are told apart rather than reported as one: an absent pool is a
    deployment that never finished starting, while an absent region in a
    populated pool is a region this deployment does not serve.

    Args:
        region: The region whose client was asked for.

    Returns:
        The reason to record for the operator.
    """
    if pooled := pooled_clients("bedrock"):
        return (
            f"No Amazon Bedrock control-plane client is pooled for {region}, so "
            "its Marketplace model endpoints were not listed. The pooled regions "
            f"are {', '.join(sorted(pooled))}: add {region} to AWS_BEDROCK_REGIONS, "
            "or drop it from AWS_BEDROCK_MARKETPLACE_ENDPOINT_REGIONS."
        )
    return (
        "The Amazon Bedrock control-plane clients are not available, so no "
        "Marketplace model endpoint was listed in any region. The client pool is "
        "built at start-up, so this is a deployment that did not finish starting "
        "rather than a setting to change: check the earlier start-up errors, or "
        "set AWS_BEDROCK_MARKETPLACE_ENDPOINTS_ENABLED=false to stop discovering "
        "endpoints at all."
    )


async def _endpoints_in_region(
    region: RegionName,
) -> tuple[list[ModelDetails], list[str]]:
    """List the servable Marketplace model endpoints of one region.

    ``ListMarketplaceModelEndpoints`` returns a summary that carries neither the
    SageMaker status nor the endpoint configuration, so readiness needs a ``Get``
    per endpoint. Both are free control-plane calls.

    Args:
        region: The region to search.

    Returns:
        Tuple of (one entry per endpoint that can serve a request today, the
        reasons the others were declined). A region with no usable client
        declines rather than raising, so an optional feature cannot take the
        whole catalogue refresh down with it.
    """
    try:
        client: BedrockClient = get_client("bedrock", region)
    except KeyError:
        # Scoped to the client lookup alone: get_client signals both "no such
        # service pool" and "no such region in it" with a bare KeyError, and
        # neither is an AWS error the caller's handler would catch. A KeyError
        # from the discovery calls below is a real defect and still propagates.
        return [], [_no_client_reason(region)]
    arns: list[str] = []
    next_token = ""
    while True:
        page = (
            await client.list_marketplace_model_endpoints(nextToken=next_token)
            if next_token
            else await client.list_marketplace_model_endpoints()
        )
        arns.extend(
            summary["endpointArn"]
            for summary in page.get("marketplaceModelEndpoints") or ()
        )
        if not (next_token := page.get("nextToken", "")):
            break
    details = await gather(
        *(client.get_marketplace_model_endpoint(endpointArn=arn) for arn in arns)
    )
    models: list[ModelDetails] = []
    declined: list[str] = []
    for detail in details:
        if (endpoint := detail.get("marketplaceModelEndpoint")) is None:
            continue
        model, reason = _model_from_endpoint(endpoint, region)
        if model is not None:
            models.append(model)
        elif reason:
            declined.append(reason)
    return models, declined


async def collect_marketplace_endpoint_models(
    failed_regions: dict[str, str],
) -> dict[str, ModelDetails]:
    """Discover the Marketplace model endpoints of every searched region.

    A region whose discovery fails is skipped and retried on the next cache
    refresh, mirroring the bedrock-runtime behavior: one degraded region must
    not empty the catalogue.

    Args:
        failed_regions: Accumulator mapping unreachable regions to the error.

    Returns:
        Servable endpoints keyed by model ID. Two endpoints of one listing in
        one region collide; the first discovered wins and the other is reported
        through *failed_regions* so the operator sees it, as is every endpoint
        discovery declined for a reason the operator can act on.
    """
    regions = marketplace_endpoint_regions()
    if not regions:
        return {}
    results = await gather(
        *(_endpoints_in_region(region) for region in regions), return_exceptions=True
    )
    models: dict[str, ModelDetails] = {}
    for region, result in zip(regions, results, strict=True):
        if isinstance(result, BaseException):
            if not isinstance(result, (BotoCoreError, ClientError)):
                raise result
            failed_regions[f"{region} (Marketplace endpoints)"] = (
                f"{type(result).__name__}: {result}. The server role needs "
                "bedrock:ListMarketplaceModelEndpoints and "
                "bedrock:GetMarketplaceModelEndpoint, or set "
                "AWS_BEDROCK_MARKETPLACE_ENDPOINTS_ENABLED=false"
            )
            continue
        region_models, declined = result
        if declined:
            failed_regions[f"{region} (Marketplace endpoints)"] = " ".join(declined)
        for model in region_models:
            if (existing := models.get(model.id)) is None:
                models[model.id] = model
                continue
            endpoints = model.marketplace_endpoints or {}
            if existing.marketplace_endpoints and not (
                endpoints.keys() & existing.marketplace_endpoints.keys()
            ):
                existing.regions.extend(model.regions)
                existing.marketplace_endpoints.update(endpoints)
            else:
                # Keyed by the colliding listing: one key per region would
                # report only the last of several duplicated listings.
                failed_regions[f"{model.id} (Marketplace endpoints in {region})"] = (
                    f"Several endpoints serve '{model.id}' in {region}: only one "
                    "is published, since they share the listing name. Set "
                    "AWS_BEDROCK_ALLOW_MARKETPLACE_ENDPOINT_ARN=true to let a "
                    "client name the others by ARN, or delete the duplicates."
                )
    return models


def merge_marketplace_endpoint_models(
    all_models: dict[str, ModelDetails], marketplace_models: dict[str, ModelDetails]
) -> None:
    """Merge discovered Marketplace endpoints into the resolved model catalogue.

    A serverless model of the same name keeps priority: it costs nothing at rest
    and is available in every region the deployment serves, so shadowing it with
    one endpoint would be a silent downgrade.

    Args:
        all_models: Resolved bedrock-runtime models, updated in place.
        marketplace_models: Endpoints from
            :func:`collect_marketplace_endpoint_models`.
    """
    MARKETPLACE_ENDPOINT_MODELS.clear()
    for model_id, model in marketplace_models.items():
        if model_id not in all_models:
            all_models[model_id] = model
            MARKETPLACE_ENDPOINT_MODELS[model_id] = model
