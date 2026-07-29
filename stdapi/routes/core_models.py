"""Custom Models API."""

from typing import TYPE_CHECKING, Annotated, Final, cast

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from stdapi.api_errors import ApiError
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.models import (
    MANTLE_SERVICE,
    ModelDetails,
    get_all_models_details,
    get_all_models_details_and_modalities,
    initialize_bedrock_models,
    resolve_model_alias,
)
from stdapi.monitoring import log_error_details, log_request_params, log_response_params
from stdapi.pricing import (
    Dimension,
    PriceKey,
    Service,
    available_currencies,
    model_prices,
    price_catalog_ready,
    select_effective_rows,
)
from stdapi.usage import format_cost

if TYPE_CHECKING:
    from types_aiobotocore_bedrock.literals import RegionName

router = APIRouter(prefix="", tags=["Models"])

#: Service tiers accepted by the model_pricing tier filter.
_PRICING_TIERS: Final[frozenset[str]] = frozenset(
    {"standard", "flex", "priority", "batch"}
)

#: Serving profiles accepted by the model_pricing routing filter.
_PRICING_ROUTINGS: Final[frozenset[str]] = frozenset({"global", "latency"})

#: Context-length buckets accepted by the model_pricing context filter.
_PRICING_CONTEXTS: Final[frozenset[str]] = frozenset({"long"})


class PriceRow(BaseModel):
    """One published unit price for a model.

    Attributes:
        region: AWS region the price applies to.
        dimension: Billed dimension (same vocabulary as request-log usage).
        tier: Service tier (standard, flex, priority, batch).
        cache_ttl: Prompt-cache write TTL bucket, when distinctly priced.
        routing: Serving profile: "global" (global cross-region inference), a
            geography prefix like "eu"/"us" (regional cross-region inference),
            an AWS region (single-region inference), or "latency" for the
            latency-optimized price variant.
        spec: Media bucket (e.g. image "resolution:quality"), when applicable.
        context: "long" for the beyond-200K-tokens prompt rate.
        unit_price: Exact plain-decimal price per ONE billed unit.
        currency: ISO currency code of the price.
    """

    region: str
    dimension: Dimension
    tier: str
    cache_ttl: str | None = None
    routing: str
    spec: str | None = None
    context: str | None = None
    unit_price: str
    currency: str


class ModelPricing(BaseModel):
    """Published price card for one model.

    Attributes:
        id: The model ID as requested.
        service: AWS service/API the prices apply to (e.g. bedrock-runtime).
        default_tier: Service tier this server applies to the model by default.
        default_routings: Serving profiles this server can use for the model
            across its configured regions ("global", geography prefixes, or
            AWS regions), in configured-region order.
        prices: Matching price rows; empty when AWS publishes none.
    """

    id: str
    service: str
    default_tier: str
    default_routings: list[str]
    prices: list[PriceRow]


@router.get(
    "/search_models",
    summary="Search available models with optional filters",
    operation_id="search_models",
    description=(
        "Search the catalogue of currently available models and return extended metadata "
        "(modalities, supported API routes, MCP tool names, AWS regions, streaming support, "
        "legacy status). Supplements the standard `/v1/models` list.\n\n"
        "All filters are optional and combined with **AND** logic — only models matching every "
        "supplied filter are returned, sorted by ID.\n\n"
        "**Agent workflow:**\n"
        "1. Call this tool first to find the right model ID, then pass it to the target endpoint.\n"
        "2. Use `route` with either a route path **or** an MCP tool name — both are accepted "
        "transparently (e.g. `route=/v1/images/generations` and `route=openai_image_generation` "
        "return the same models).\n"
        "3. **Combine filters for multimodal tasks:** when a tool supports extended input modalities "
        "(e.g. images in `openai_chat_completion`), add `input_modalities=IMAGE` alongside "
        "`route` — this ensures the model supports *both* the route and the required modality. "
        "A model that only handles text would otherwise appear in a route-only search and then fail "
        "at request time.\n"
        "4. **Exclude legacy models:** Add `legacy=false` to skip deprecated models unless you "
        "specifically need one.\n\n"
        "**Examples:**\n"
        "- Text generation: `route=openai_chat_completion&legacy=false`\n"
        "- Vision (image input): `route=openai_chat_completion&input_modalities=IMAGE&legacy=false`\n"
        "- Audio understanding: `route=openai_chat_completion&input_modalities=SPEECH&legacy=false`\n"
        "- Embeddings: `route=openai_embedding&legacy=false`\n"
        "- Image generation: `route=openai_image_generation&legacy=false`\n\n"
        '**Note:** Audio *output* from `openai_chat_completion` (via `modalities=["text","audio"]`) '
        "is a model-specific capability not separately tracked — use a `route` search "
        "and verify audio output support in the model documentation."
    ),
    response_description="A list of extended model details sorted by model ID",
    response_model_exclude_none=True,
    responses={
        200: {"description": "OK"},
        400: {"description": "Invalid modality, route, or MCP tool filter."},
    },
)
async def search_models(
    input_modalities: Annotated[
        set[str] | None,
        Query(
            description="Filter by expected input modalities (e.g., TEXT, IMAGE, SPEECH)."
        ),
    ] = None,
    output_modalities: Annotated[
        set[str] | None,
        Query(
            description="Filter by expected output modalities (e.g., TEXT, IMAGE, AUDIO)."
        ),
    ] = None,
    route: Annotated[
        str | None,
        Query(
            description=(
                "Filter to models that support a specific route path "
                "(e.g. /v1/chat/completions) or MCP tool name (e.g. openai_chat_completion). "
                "Both formats are accepted transparently."
            )
        ),
    ] = None,
    region: Annotated[
        str | None,
        Query(
            description="Filter to models available in a specific AWS region (e.g. us-east-1)."
        ),
    ] = None,
    streaming: Annotated[
        bool | None,
        Query(
            description="Filter by streaming support (true = streaming only, false = non-streaming only)."
        ),
    ] = None,
    legacy: Annotated[
        bool | None,
        Query(
            description="Filter by legacy status (true = deprecated models only, false = non-deprecated models only)."
        ),
    ] = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> list[ModelDetails]:
    """Search the model catalogue with optional filters and return extended metadata.

    Args:
        input_modalities: Filter to models that accept these input modalities (e.g. TEXT, IMAGE).
        output_modalities: Filter to models that produce these output modalities (e.g. TEXT, IMAGE).
        route: Filter to models that support this route path (e.g. /v1/chat/completions) or MCP
            tool name (e.g. openai_chat_completion). Both formats are accepted transparently.
        region: Filter to models available in a specific AWS region.
        streaming: Filter by streaming support.
        legacy: Filter by legacy/deprecated status.

    Returns:
        Filtered and sorted list of model details.

    Raises:
        ApiError: When an unknown modality, route, or MCP tool filter is specified (400) or
            models cannot be retrieved from backend services (500).
    """
    log_request_params(
        {
            "input_modalities": input_modalities,
            "output_modalities": output_modalities,
            "route": route,
            "region": region,
            "streaming": streaming,
            "legacy": legacy,
        }
    )
    await initialize_bedrock_models()
    (
        models,
        models_output_modalities,
        models_input_modalities,
    ) = await get_all_models_details_and_modalities()
    models_ids = set(models.keys())
    _filter_by_modality(input_modalities, models_ids, models_input_modalities, "input")
    _filter_by_modality(
        output_modalities, models_ids, models_output_modalities, "output"
    )
    _filter_by_route_or_tool(route, models_ids, models)
    if region is not None:
        _filter_by_region(region, models_ids, models)
    if streaming is not None:
        models_ids &= {
            mid for mid, m in models.items() if m.response_streaming is streaming
        }
    if legacy is not None:
        models_ids &= {mid for mid, m in models.items() if (m.legacy is True) is legacy}
    return log_response_params([models[model_id] for model_id in sorted(models_ids)])


def _filter_by_modality(
    modalities: set[str] | None,
    models_ids: set[str],
    models_by_modalities: dict[str, set[str]],
    modality_type: str,
) -> None:
    """Filters the provided models based on specific modalities.

    Args:
        modalities:
            A set of modality names to filter the models by. If None, no filtering is applied.
        models_ids:
            A set of model identifiers to be filtered. This set is modified in place.
        models_by_modalities:
            A dictionary mapping modality names (as keys) to sets of corresponding model
            identifiers (as values).
        modality_type:
            A string representing the descriptive name or type of modality, used for error messages.
    """
    if not modalities:
        return
    matched: set[str] = set()
    for raw in modalities:
        modality = raw.strip().upper()
        if (ids := models_by_modalities.get(modality)) is None:
            msg = f"No model matching {modality_type} modality: {modality}."
            raise ApiError(msg)
        matched |= ids
    models_ids &= matched


def _filter_by_route_or_tool(
    value: str | None, models_ids: set[str], models: dict[str, ModelDetails]
) -> None:
    """Filter models to those supporting a specific route path or MCP tool name.

    Checks both ``supported_routes`` and ``supported_mcp_tools`` and accepts either
    format transparently (e.g. ``/v1/images/generations`` and ``openai_image_generation``
    both match the same set of models).

    Args:
        value: Route path or MCP tool name to filter by. No-op when ``None``.
        models_ids: Set of model identifiers to filter in place.
        models: All model details keyed by model ID.
    """
    if not value:
        return
    if not (
        matched := {
            mid
            for mid, m in models.items()
            if value in m.supported_routes or value in m.supported_mcp_tools
        }
    ):
        msg = f"No model supporting route or MCP tool: {value}."
        raise ApiError(msg)
    models_ids &= matched


def _filter_by_region(
    region: str, models_ids: set[str], models: dict[str, ModelDetails]
) -> None:
    """Filter models to those available in the specified AWS region.

    Args:
        region: AWS region name to filter by (e.g. ``us-east-1``).
        models_ids: Set of model identifiers to filter in place.
        models: All model details keyed by model ID.
    """
    if not (matched := {mid for mid, m in models.items() if region in m.regions}):
        msg = f"No model available in region: {region}."
        raise ApiError(msg)
    models_ids &= matched


def _validate_filter(value: str | None, allowed: frozenset[str], name: str) -> None:
    """Reject an unknown enumerated filter value with a 400 error.

    Args:
        value: The supplied filter value; None means "no filter" and passes.
        allowed: Accepted values for this filter.
        name: Query-parameter name, used in the error message.

    Raises:
        ApiError: When *value* is not None and not in *allowed* (400).
    """
    if value is not None and value not in allowed:
        msg = f"Invalid {name}: {value}. Valid values: {', '.join(sorted(allowed))}."
        raise ApiError(msg)


def _validated_dimensions(dimension: set[str] | None) -> set[Dimension] | None:
    """Convert dimension filter strings to Dimension members.

    Args:
        dimension: Raw dimension names from the query string, or None.

    Returns:
        The corresponding Dimension members, or None when unfiltered.

    Raises:
        ApiError: When a name is not a known billed dimension (400).
    """
    if dimension is None:
        return None
    try:
        return {Dimension(name) for name in dimension}
    except ValueError as error:
        valid = ", ".join(sorted(Dimension))
        msg = f"Invalid dimension: {error.args[0]}. Valid values: {valid}."
        raise ApiError(msg) from None


def _pricing_defaults(
    model_id: str, details: ModelDetails | None
) -> tuple[str, list[str]]:
    """Return the (tier, routings) this server applies to a model by default.

    Args:
        model_id: The canonical model ID (aliases already resolved).
        details: The model's registry entry, when known.

    Returns:
        Tuple of (service tier, serving profiles): the tier from
        ``default_model_service_tiers`` (``standard`` when unset) and the
        distinct serving profiles across the configured Bedrock regions —
        ``["global"]``, or per region the geography prefix of the model's
        inference profile or the region itself, restricted to the model's
        regions when known.
    """
    configured_tier = SETTINGS.default_model_service_tiers.get(model_id, "default")
    tier = "standard" if configured_tier == "default" else configured_tier
    profiles = (details.inference_profiles if details else None) or {}
    if any(profile.startswith("global.") for profile in profiles.values()):
        return tier, ["global"]
    model_regions = set(details.regions) if details else None
    routings: list[str] = []
    for region in SETTINGS.aws_bedrock_regions:
        if model_regions is not None and region not in model_regions:
            continue
        routing = (
            profile.split(".", 1)[0] if (profile := profiles.get(region)) else region
        )
        if routing not in routings:
            routings.append(routing)
    return tier, routings


def _row_routing(key: PriceKey, details: ModelDetails | None) -> str:
    """Return the serving profile displayed for one price row.

    Args:
        key: The row's price key.
        details: The model's registry entry, when known.

    Returns:
        The key's own "global"/"latency" variant, the geography prefix of the
        model's inference profile in the row's region, or the row's region.
    """
    if key.routing:
        return key.routing
    profiles = (details.inference_profiles if details else None) or {}
    region = cast("RegionName", key.region)
    if (profile := profiles.get(region)) and not profile.startswith("global."):
        return profile.split(".", 1)[0]
    return key.region


@router.get(
    "/model_pricing",
    summary="Get exact AWS unit prices for one or more models",
    operation_id="model_pricing",
    description=(
        "Return the exact AWS unit prices for the requested models (or every "
        "available model when `model` is omitted), straight from the same AWS "
        "Price List catalog used for request cost tracking.\n\n"
        "Each row is one published price: a billed dimension (same vocabulary as "
        "the request-log usage entries) plus its published variants — service "
        "tier, prompt-cache write TTL, serving profile (routing), media spec "
        "(e.g. image resolution:quality), and long-context bucket. `unit_price` "
        "is an exact decimal string per **one** billed unit (token, image, "
        "second, character, request, search unit) in `currency`.\n\n"
        "By default the card reflects **this server's configuration**: only the "
        "configured Bedrock regions, the model's default service tier, and its "
        "effective serving profile (routing), with the same fallbacks billing "
        "applies. Pass `all_prices=true` for the model's full published price "
        "table, and explicit `region`/`tier`/`routing` filters to override any "
        "axis in either mode.\n\n"
        "Prices are indexed eagerly: models not currently accessible to this "
        "deployment can still be priced. An empty `prices` list means AWS "
        "publishes no rows for that model (or none match the filters). A missing "
        "variant row means AWS publishes no distinct rate for it — billing then "
        "falls back the same way cost tracking does.\n\n"
        "All filters are optional and combined with **AND** logic.\n\n"
        "**Agent workflow:**\n"
        "1. Call `search_models` first to shortlist model IDs for the task.\n"
        "2. Call this tool with the shortlist (repeat `model`) and compare.\n"
        "3. Keep responses small: use `variants=false` for the base price card "
        "and `dimension` filters for token-only comparisons.\n\n"
        "**Examples:**\n"
        "- Card of a shortlist as billed here: `model=A&model=B`\n"
        "- Every available model as billed here: no parameters\n"
        "- Full published table: `model=A&all_prices=true`\n"
        "- Token rates in one region: `model=A&region=us-east-1&"
        "dimension=input_tokens&dimension=output_tokens`\n"
        "- Latency-optimized premium: `model=A&routing=latency`\n\n"
        "Requires cost tracking to be enabled (`COST_TRACKING` setting)."
    ),
    response_description=(
        "One price card per requested model, in request order (all available "
        "models, sorted by ID, when `model` is omitted)"
    ),
    response_model_exclude_none=True,
    responses={
        200: {"description": "OK"},
        400: {"description": "Invalid tier, dimension, routing, or context filter."},
        503: {
            "description": "Model pricing is not available on this server, or "
            "the price catalog is not loaded yet (retry later)."
        },
    },
)
async def model_pricing(
    *,
    model: Annotated[
        list[str] | None,
        Query(
            min_length=1,
            description=(
                "Model IDs or aliases to price (repeat for several). "
                "Omit for the full list of available models."
            ),
        ),
    ] = None,
    region: Annotated[
        str | None,
        Query(description="Only prices for this AWS region (e.g. us-east-1)."),
    ] = None,
    tier: Annotated[
        str | None,
        Query(description="Only this service tier (standard, flex, priority, batch)."),
    ] = None,
    dimension: Annotated[
        set[str] | None,
        Query(
            description=(
                "Only these billed dimensions (e.g. input_tokens, output_tokens; "
                "repeat for several)."
            )
        ),
    ] = None,
    variants: Annotated[
        bool,
        Query(
            description=(
                "false = base price card only: standard tier without cache-TTL, "
                "routing, or long-context variant rows (media spec rows are kept)."
            )
        ),
    ] = True,
    currency: Annotated[
        str | None,
        Query(description="Only prices in this ISO currency code (e.g. USD, EUR)."),
    ] = None,
    routing: Annotated[
        str | None,
        Query(
            description=(
                "Only this published serving-profile price variant (global or "
                "latency). Row `routing` values enriched for display -- "
                "geography prefixes like eu/us, or AWS regions -- cannot be "
                "filtered on; use `region` for those."
            )
        ),
    ] = None,
    context: Annotated[
        str | None, Query(description="Only this context-length bucket (long).")
    ] = None,
    all_prices: Annotated[
        bool,
        Query(
            description=(
                "true = the model's full published price table. false "
                "(default) = only the prices matching this server's "
                "configuration: configured Bedrock regions, the model's "
                "default service tier, and its effective serving profile "
                "(routing), with the same fallbacks billing applies."
            )
        ),
    ] = False,
    _: Annotated[None, Depends(authenticate)] = None,
) -> list[ModelPricing]:
    """Return the published price card for each requested model.

    Args:
        model: Model IDs or aliases to price; every available model when None.
        region: Only prices for this AWS region.
        tier: Only this service tier.
        dimension: Only these billed dimensions.
        variants: When False, only base rows (see the query description).
        currency: Only prices in this ISO currency code.
        routing: Only this serving profile.
        context: Only this context-length bucket.
        all_prices: When False, only the current-configuration prices (see
            the query description).

    Returns:
        One price card per requested model, in request order (duplicates
        removed) or sorted by model ID when unfiltered, with rows sorted by
        region then remaining axes.

    Raises:
        ApiError: When an enumerated filter value is unknown (400), or model
            pricing is unavailable or not loaded yet (503).
    """
    log_request_params(
        {
            "model": model,
            "region": region,
            "tier": tier,
            "dimension": dimension,
            "variants": variants,
            "currency": currency,
            "routing": routing,
            "context": context,
            "all_prices": all_prices,
        }
    )
    if not SETTINGS.cost_tracking:
        log_error_details(
            "Cost tracking is disabled (cost_tracking): the model pricing API "
            "is unavailable.",
            level="warning",
        )
        msg = (
            "Model pricing is not available on the current server. "
            "Please contact the administrator to enable it."
        )
        raise ApiError(msg, status=503)
    if not price_catalog_ready():
        msg = "The price catalog is not loaded yet. Please try again later."
        raise ApiError(msg, status=503)
    _validate_filter(tier, _PRICING_TIERS, "tier")
    _validate_filter(routing, _PRICING_ROUTINGS, "routing")
    _validate_filter(context, _PRICING_CONTEXTS, "context")
    if currency:
        currency = currency.upper()
        _validate_filter(currency, available_currencies(), "currency")
    dimensions = _validated_dimensions(dimension)

    all_models = await get_all_models_details()
    results = []
    for model_id in dict.fromkeys(model) if model else sorted(all_models):
        details = all_models.get(resolve_model_alias(model_id))
        default_tier, default_routings = _pricing_defaults(
            resolve_model_alias(model_id), details
        )
        preferred_service = (
            Service.BEDROCK_MANTLE
            if details is not None and details.service == MANTLE_SERVICE
            else Service.BEDROCK
        )
        rows = model_prices(
            model_id,
            region=region,
            tier=tier,
            dimensions=dimensions,
            currency=currency or None,
            routing=routing,  # type: ignore[arg-type]
            context=context,  # type: ignore[arg-type]
            variants=variants,
            preferred_service=preferred_service,
        )
        if not all_prices:
            rows = select_effective_rows(
                rows,
                regions=None if region else set(SETTINGS.aws_bedrock_regions),
                tier=None if tier else default_tier,
                routing=None
                if routing
                else ("global" if default_routings == ["global"] else ""),
            )
        results.append(
            ModelPricing(
                id=model_id,
                service=rows[0][0].service.value if rows else preferred_service.value,
                default_tier=default_tier,
                default_routings=default_routings,
                prices=[
                    PriceRow(
                        region=key.region,
                        dimension=key.dimension,
                        tier=key.tier,
                        cache_ttl=key.cache_ttl or None,
                        routing=_row_routing(key, details),
                        spec=key.spec or None,
                        context=key.context or None,
                        unit_price=format_cost(price.amount),
                        currency=price.currency,
                    )
                    for key, price in rows
                ],
            )
        )
    return log_response_params(results)
