"""AWS pricing resolution for per-request cost tracking.

In-memory price index keyed by :class:`PriceKey`. Non-blocking: async I/O
via aiobotocore, synchronous in-memory lookup. Ingestion is eager: every
published row that can be keyed unambiguously is indexed, independent of
model availability/permissions (those vary per region, account and period).
Data-quality issues (collisions, invalid overrides) are collected as
diagnostics and surfaced as one summary warning on the owning operation's
log event. AWS Pricing API errors propagate to the caller.

MAINTENANCE -- this module is model-agnostic; new models normally need no
code change here:
- New model unpriced? Add one line to ``stdapi/models/pricing_overrides.py``
  (workflow in that module's docstring). Model-specific data lives in
  ``stdapi.models``, never here.
- Model priced only on the AWS pricing page (absent from the Price List
  API)? Add its page rate to ``DEFAULT_MODEL_PRICES`` in that same module.
- Collision warnings at startup ("Price catalog collision on ...") mean AWS
  introduced a pricing axis or usagetype schema this module doesn't key yet:
  extend :class:`PriceKey` and/or the ingestion resolvers (`_resolve_tier`,
  `_native_routing`, `_price_context`, `_native_price_spec`,
  `_USAGETYPE_FALLBACK_DIMENSIONS`, `_MARKETPLACE_USAGETYPE_DIMENSIONS`).
- New billed unit? Add a :class:`Dimension` member here plus one
  ``_DIMENSION_INFO`` row in ``stdapi.usage`` -- the rest hooks in.
- Validate any change offline against live data with
  ``tests/test_pricing.py::test_bedrock_model_pricing_coverage --expensive``.
"""

import asyncio
import json
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Literal

from stdapi.aws import get_client
from stdapi.config import AWS_REGION, SETTINGS

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from types_aiobotocore_pricing.client import PricingClient

    from stdapi.monitoring import EventLog


class Dimension(StrEnum):
    """Billed dimension types tracked for cost computation."""

    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    CACHE_READ_TOKENS = "cache_read_tokens"
    CACHE_WRITE_TOKENS = "cache_write_tokens"
    OUTPUT_IMAGES = "output_images"
    INPUT_SECONDS = "input_seconds"
    INPUT_CHARACTERS = "input_characters"
    COMPREHEND_UNITS = "comprehend_units"
    GROUNDING_REQUESTS = "grounding_requests"
    SEARCH_UNITS = "search_units"
    INPUT_IMAGES = "input_images"
    OUTPUT_SECONDS = "output_seconds"


class Service(StrEnum):
    """AWS services/APIs that have pricing support.

    Bedrock is split per invocation API: the app serves and records via
    bedrock-runtime only; bedrock-mantle rates are indexed for the future.
    """

    BEDROCK = "bedrock-runtime"
    BEDROCK_MANTLE = "bedrock-mantle"
    POLLY = "polly"
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"
    COMPREHEND = "comprehend"


#: Default currency per AWS partition.
PARTITION_CURRENCY: Final[dict[str, str]] = {
    "aws": "USD",
    "aws-eusc": "EUR",
    "aws-us-gov": "USD",
    "aws-cn": "CNY",
}


def partition_of_region(region: str) -> str:
    """Determine the AWS partition for a given region.

    Args:
        region: AWS region code (e.g., us-east-1, eusc-de-east-1).

    Returns:
        Partition identifier (aws, aws-eusc, aws-us-gov, aws-cn).
    """
    if region.startswith("eusc-"):
        return "aws-eusc"
    if region.startswith("us-gov-"):
        return "aws-us-gov"
    if region.startswith("cn-"):
        return "aws-cn"
    return "aws"


#: Parse scale from unit strings (e.g., "1K tokens" -> 1000).
_UNIT_SCALE_PATTERN = re.compile(r"^(\d+)([KM])\s", re.IGNORECASE)


def parse_unit_scale(unit: str) -> int:
    """Parse the scale multiplier from a price dimension unit.

    Args:
        unit: The unit string (e.g., "1K tokens", "1M Characters", "Tokens").

    Returns:
        Units the price is quoted per -- e.g. "10K tokens" -> 10000.
    """
    if match := _UNIT_SCALE_PATTERN.match(unit):
        return int(match.group(1)) * (
            1000 if match.group(2).upper() == "K" else 1_000_000
        )
    return 1


#: Normalized inferenceType prefix to dimension (most specific first).
_INFERENCE_TYPE_PREFIXES: Final[tuple[tuple[str, Dimension], ...]] = (
    ("prompt cache read", Dimension.CACHE_READ_TOKENS),
    ("prompt cache write", Dimension.CACHE_WRITE_TOKENS),
    ("input tokens", Dimension.INPUT_TOKENS),
    ("output tokens", Dimension.OUTPUT_TOKENS),
    ("text input token", Dimension.INPUT_TOKENS),
    ("text output token", Dimension.OUTPUT_TOKENS),
    ("speech understanding input token", Dimension.INPUT_TOKENS),
    ("speech understanding output token", Dimension.OUTPUT_TOKENS),
)


def inference_type_to_dimension(
    inference_type: str, usagetype: str, service: Service
) -> Dimension | None:
    """Map AWS inferenceType to Dimension enum, falling back to usagetype patterns.

    Args:
        inference_type: The inferenceType attribute from AWS pricing.
        usagetype: The usagetype attribute (used as fallback for images/units).
        service: The Service this price-list entry belongs to. The
            usagetype images/units fallback is Bedrock-only: other
            services' usagetypes can innocently contain "unit"/"image".

    Returns:
        The corresponding Dimension, or None if not mappable.
    """
    # Try prefix matching on inferenceType (e.g., "Output tokens flex" -> OUTPUT_TOKENS)
    normalized = inference_type.strip().lower()
    for prefix, dimension in _INFERENCE_TYPE_PREFIXES:
        if normalized.startswith(prefix):
            return dimension

    # Image/video-generation models: inferenceType is "T2I/I2I <resolution>
    # <quality>" or "T2V/I2V <fps> <resolution>" -- see _image_generation_spec().
    if _IMAGE_GENERATION_SPEC_PATTERN.match(normalized):
        return Dimension.OUTPUT_IMAGES
    if _VIDEO_GENERATION_PATTERN.match(normalized):
        return Dimension.OUTPUT_SECONDS

    if service != Service.BEDROCK:
        return None

    # Rows with no inferenceType/featuretype at all (grounding, media inputs,
    # token rows like xai.grok's "<model>-mantle-<dim>-<tier>"): resolve from
    # usagetype text. Rows with an unrecognized non-empty inferenceType stay
    # unmapped to avoid colliding with the plain dimensions.
    if not normalized:
        normalized_usagetype = _normalize_usagetype(usagetype)
        for pattern, dimension in _USAGETYPE_FALLBACK_DIMENSIONS:
            if pattern in normalized_usagetype:
                return dimension

    return _generic_usagetype_dimension(usagetype)


def _generic_usagetype_dimension(usagetype: str) -> Dimension | None:
    """Last-resort usagetype fallback for Bedrock rows.

    "token" rows are excluded from the image/unit branch (multimodal token
    counts are already billed within plain tokens), and "searchunit"
    (rerank) is matched before the generic "unit".

    Args:
        usagetype: The usagetype attribute.

    Returns:
        The resolved dimension, or None.
    """
    normalized = _normalize_usagetype(usagetype)
    if "searchunit" in normalized:
        return Dimension.SEARCH_UNITS
    if "token" not in normalized and (
        "unit" in normalized or ("image" in normalized and "input" not in normalized)
    ):
        return Dimension.OUTPUT_IMAGES
    return None


#: usagetype substring to dimension for inferenceType-less native rows (ordered).
_USAGETYPE_FALLBACK_DIMENSIONS: Final[tuple[tuple[str, Dimension], ...]] = (
    ("novagrounding", Dimension.GROUNDING_REQUESTS),
    ("inputstandardimage", Dimension.INPUT_IMAGES),
    ("inputdocumentimage", Dimension.INPUT_IMAGES),
    ("inputaudiosecond", Dimension.INPUT_SECONDS),
    ("inputvideosecond", Dimension.INPUT_SECONDS),
    ("cachewrite", Dimension.CACHE_WRITE_TOKENS),
    ("cacheread", Dimension.CACHE_READ_TOKENS),
    ("inputtoken", Dimension.INPUT_TOKENS),
    ("outputtoken", Dimension.OUTPUT_TOKENS),
)


#: Provider prefixes to strip from model IDs.
_PROVIDER_PREFIXES: Final[tuple[str, ...]] = (
    "anthropic.",
    "amazon.",
    "meta.",
    "google.",
    "mistral.",
    "deepseek.",
    "cohere.",
    "stability.",
    "nvidia.",
    "ai21.",
    "mistralai.",
    "moonshot.",
    "moonshotai.",
    "minimax.",
    "qwen.",
    "writer.",
    "luma.",
    "lumaai.",
    "twelvelabs.",
    "kimi.",
    "zai.",
    "openai.",
)

#: Cross-region inference prefixes to strip.
_CROSS_REGION_PREFIXES: Final[tuple[str, ...]] = ("us.", "eu.", "apac.", "global.")

#: All prefixes normalize_model_key() strips (cross-region may precede provider).
_MODEL_ID_STRIP_PREFIXES: Final[tuple[str, ...]] = (
    *_PROVIDER_PREFIXES,
    *_CROSS_REGION_PREFIXES,
)

#: Parenthetical suffixes to strip (e.g. "(dense)", "(24.02)").
_PARENTHETICAL_PATTERN = re.compile(r"\([^)]*\)")

#: Trailing context-window suffix (e.g. ":24k", ":300k") -- same billed model/price as base.
_CONTEXT_WINDOW_SUFFIX_PATTERN = re.compile(r":\d+k$", re.IGNORECASE)

#: Trailing version suffix (v1, v2:0, v1:0, etc.).
_VERSION_PATTERN = re.compile(r"[-:]?v\d+(:\d+)?$", re.IGNORECASE)

#: Trailing 8-digit snapshot date (e.g. "-20241022").
_DATE_SUFFIX_PATTERN = re.compile(r"-\d{8}$")

#: Trailing "Latency Optimized" price-list qualifier (a Routing axis, not a model).
_LATENCY_OPTIMIZED_SUFFIX_PATTERN = re.compile(
    r"\s+latency\s+optimized\s*$", re.IGNORECASE
)

#: Match separators in model names.
_SEPARATOR_PATTERN = re.compile(r"[-._: ]+")


def normalize_model_key(model_id: str) -> str:
    """Normalize a model ID for price index lookup.

    Strips provider/cross-region prefixes, parenthetical qualifiers, context-window
    suffixes, version suffixes, and snapshot dates.

    Models that don't match automatically need an override in
    ``stdapi/models/pricing_overrides.py`` (see :func:`resolve_model_key`).

    Args:
        model_id: The full model ID (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0").

    Returns:
        Normalized model key for indexing (e.g., "claude35sonnet").
    """
    model_id = model_id.strip()
    lower = model_id.lower()
    stripped = True
    while stripped:
        stripped = False
        for prefix in _MODEL_ID_STRIP_PREFIXES:
            if lower.startswith(prefix):
                # Slice the lowered copy too, avoiding a re-lower per strip.
                model_id = model_id[len(prefix) :]
                lower = lower[len(prefix) :]
                stripped = True

    model_id = _PARENTHETICAL_PATTERN.sub("", model_id)
    model_id = _LATENCY_OPTIMIZED_SUFFIX_PATTERN.sub("", model_id)
    model_id = _CONTEXT_WINDOW_SUFFIX_PATTERN.sub("", model_id)
    model_id = _VERSION_PATTERN.sub("", model_id)
    model_id = _DATE_SUFFIX_PATTERN.sub("", model_id)

    return _SEPARATOR_PATTERN.sub("", model_id.lower())


def normalize_usagetype_model(usagetype: str) -> str:
    """Extract and normalize model token from usagetype.

    The usagetype format is: <REGIONPREFIX>-<ModelCamel>-<dimension>[-suffix].
    A Marketplace-style "MP:" prefix (with its own duplicated region code) is
    additionally unwrapped when present.

    Args:
        usagetype: The usagetype string (e.g., "USE1-NovaLite-input-tokens").

    Returns:
        Normalized model key (e.g., "novalite").
    """
    parts = usagetype.split("-")

    # Find where the dimension part starts
    dimension_parts = {
        "input",
        "output",
        "cache",
        "prompt",
        "character",
        "token",
        "units",
        "second",
        "image",
    }
    model_end = len(parts)
    for i, part in enumerate(parts):
        if part.lower() in dimension_parts:
            model_end = i
            break

    # Take model parts (skip region prefix at index 0)
    if not (model_parts := parts[1:model_end]):
        return ""

    # Handle MP: prefix (Marketplace models)
    if (model_token := "-".join(model_parts)).startswith("MP:"):
        model_token = model_token[3:]
        # Strip a duplicated region-code segment after MP:, e.g.
        # "USE1_created_image_stable_image_core" -> "created_image_stable_image_core".
        # Must split on "_" (MP:'s own separator), not re-scan the "-"-split
        # `parts` from above -- those still have "MP:" attached and are
        # never a bare 4-char region code, so that pattern can never match.
        segments = model_token.split("_")
        if len(segments) > 1 and len(segments[0]) == 4 and segments[0].isupper():
            model_token = "_".join(segments[1:])

    # Strip trailing "-Read"/"-Write"; other dimension words never reach model_token.
    for suffix in ("-read", "-write"):
        if model_token.lower().endswith(suffix):
            model_token = model_token[: -len(suffix)]
            break

    return _VERSION_PATTERN.sub("", _SEPARATOR_PATTERN.sub("", model_token.lower()))


#: Cache TTL bucket: "" (undifferentiated) or an AWS CacheTTLType value.
type CacheTtlBucket = Literal["", "5m", "1h"]

#: Serving profile: "" (plain/regional), "global" routing, or "latency"-optimized.
type Routing = Literal["", "global", "latency"]

#: Context-length pricing bucket: "" (standard) or "long" (>200K-token prompt rate).
type ContextLength = Literal["", "long"]


@dataclass(frozen=True, slots=True)
class PriceKey:
    """Composite key identifying one billed dimension's unit price."""

    service: Service
    model: str  # Normalized via normalize_model_key
    region: str
    dimension: Dimension
    tier: str
    cache_ttl: CacheTtlBucket = ""
    routing: Routing = ""
    # "<resolution>:<quality>" (generated images) or a modality qualifier
    # ("speech"/"document"/"audio"/"video"/"hd"); "" is the default bucket.
    spec: str = ""
    context: ContextLength = ""


@dataclass(frozen=True, slots=True)
class Price:
    """A resolved unit price, per one billed unit."""

    amount: Decimal
    currency: str


@dataclass(slots=True)
class _PriceCatalogState:
    """Mutable module state: the in-memory price index and its refresh lock."""

    price_index: dict[PriceKey, Price] = field(default_factory=dict)
    # Serializes on-demand refreshes to prevent duplicate catalog reloads.
    refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


#: Module state. Swapping price_index wholesale keeps concurrent readers safe.
_state = _PriceCatalogState()


#: Model-key override registry, populated by register_model_key_overrides().
_MODEL_KEY_OVERRIDES: dict[str, str] = {}


def register_model_key_overrides(overrides: Mapping[str, str]) -> None:
    """Merge model-ID-to-price-key overrides into the lookup registry.

    Called by ``stdapi.models`` at import time with its
    ``pricing_overrides.MODEL_KEY_OVERRIDES`` table -- this module stays
    model-agnostic and holds no model list of its own.

    Args:
        overrides: Model IDs mapped to normalized price-catalog keys.
    """
    _MODEL_KEY_OVERRIDES.update(overrides)


def resolve_model_key(model_id: str) -> str:
    """Resolve a model ID to its price-catalog key (overrides first).

    Args:
        model_id: The full model ID.

    Returns:
        The registered override, or :func:`normalize_model_key` output.
    """
    return _MODEL_KEY_OVERRIDES.get(model_id) or normalize_model_key(model_id)


#: Built-in default prices registry, populated by register_default_prices().
_DEFAULT_PRICES: dict[PriceKey, Price] = {}


def register_default_prices(
    prices: Mapping[str, Mapping[Dimension, str]], regions: Iterable[str]
) -> None:
    """Register built-in default prices for models absent from the Price List API.

    Last-resort rates hand-copied from the AWS pricing page, the only place
    AWS publishes them. Called by ``stdapi.models`` at import time with its
    ``pricing_overrides.DEFAULT_MODEL_PRICES`` table. Applied at catalog load
    only to models with no published row at all (see
    :func:`_apply_default_prices`); ``cost_price_overrides`` still wins.

    Args:
        prices: Model ID to per-dimension USD price (exact decimal text).
        regions: Regions the prices apply to, per the pricing page.
    """
    for model_id, dimension_prices in prices.items():
        model = resolve_model_key(model_id)
        for dimension, amount in dimension_prices.items():
            for region in regions:
                _DEFAULT_PRICES[
                    PriceKey(Service.BEDROCK, model, region, dimension, "standard")
                ] = Price(Decimal(amount), "USD")


#: AWS tier-price ratios: flex/batch = 0.5 (50% discount), priority = 1.75 (75% premium).
_TIER_PRICE_RATIO: Final[dict[str, Decimal]] = {
    "flex": Decimal("0.5"),
    "batch": Decimal("0.5"),
    "priority": Decimal("1.75"),
}

#: Neutral tier ratio (exact-tier price, no scaling).
_ONE: Final = Decimal(1)

#: Dimensions the AWS tier ratios apply to; per-request fees are tier-flat.
_TIER_SCALED_DIMENSIONS: Final[frozenset[Dimension]] = frozenset(
    {
        Dimension.INPUT_TOKENS,
        Dimension.OUTPUT_TOKENS,
        Dimension.CACHE_READ_TOKENS,
        Dimension.CACHE_WRITE_TOKENS,
    }
)


def resolve_price(
    service: Service,
    model: str,
    region: str,
    dimension: Dimension,
    tier: str = "standard",
    cache_ttl: CacheTtlBucket = "",
    routing: Routing = "",
    spec: str = "",
    context: ContextLength = "",
) -> Price | None:
    """Resolve the unit price for a service/model/region/dimension/tier.

    Falls back to the standard-tier price when *tier* isn't indexed (scaled
    by ``_TIER_PRICE_RATIO`` for token dimensions only), and to the
    undifferentiated (``""``) price when *cache_ttl*/*routing*/*spec*/
    *context* have no distinct entry.

    Args:
        service: The AWS service/API.
        model: The model ID, resolved via :func:`resolve_model_key`.
        region: The AWS region.
        dimension: The billed dimension.
        tier: The service tier (standard, flex, priority, batch).
        cache_ttl: Cache TTL bucket ("5m"/"1h"), only meaningful for
            ``Dimension.CACHE_WRITE_TOKENS``.
        routing: Serving profile ("global", "latency" or "" -- regional, a
            safe default when the caller doesn't track the profile).
        spec: Media spec bucket -- see :attr:`PriceKey.spec`.
        context: Context-length bucket ("long" when the call's prompt
            exceeded 200K tokens).

    Returns:
        The resolved :class:`Price`, or None if not found.
    """
    normalized_model = resolve_model_key(model)
    tier_normalized = tier.lower() if tier else "standard"

    # Candidate order, most exact first. Context relaxes outermost: the
    # long-context premium is a real published rate, so a scaled/relaxed
    # long-context price beats an exact-tier standard-context one. Tier is
    # next (a scaled standard price only when the exact tier isn't indexed;
    # the ratio only applies to token rates -- per-request fees are flat),
    # then routing. cache_ttl and spec never coexist, so they relax
    # together as one axis pair.
    fallback_ratio = (
        _TIER_PRICE_RATIO.get(tier_normalized, _ONE)
        if dimension in _TIER_SCALED_DIMENSIONS
        else _ONE
    )
    tier_candidates: tuple[tuple[str, Decimal], ...] = (
        ((tier_normalized, _ONE),)
        if tier_normalized == "standard"
        else ((tier_normalized, _ONE), ("standard", fallback_ratio))
    )
    context_candidates: tuple[ContextLength, ...] = (context, "") if context else ("",)
    routing_candidates: tuple[Routing, ...] = (routing, "") if routing else ("",)
    axis_candidates: tuple[tuple[CacheTtlBucket, str], ...] = (
        ((cache_ttl, spec), ("", "")) if (cache_ttl or spec) else ((cache_ttl, spec),)
    )
    for cx in context_candidates:
        for t, ratio in tier_candidates:
            for r in routing_candidates:
                for ct, sp in axis_candidates:
                    key = PriceKey(
                        service, normalized_model, region, dimension, t, ct, r, sp, cx
                    )
                    if (price := _state.price_index.get(key)) is not None:
                        return (
                            price
                            if ratio == _ONE
                            else Price(price.amount * ratio, price.currency)
                        )
    return None


#: AWS Price List service codes mapped to their internal service.
_SERVICE_CODE_TO_SERVICE: Final[dict[str, Service]] = {
    "AmazonBedrock": Service.BEDROCK,
    "AmazonBedrockService": Service.BEDROCK,
    "AmazonBedrockFoundationModels": Service.BEDROCK,
    "AmazonPolly": Service.POLLY,
    "AmazonTranscribe": Service.TRANSCRIBE,
    "transcribe": Service.TRANSCRIBE,
    "AmazonTranslate": Service.TRANSLATE,
    "translate": Service.TRANSLATE,
    "AmazonComprehend": Service.COMPREHEND,
    "comprehend": Service.COMPREHEND,
}

#: Max time the initial catalog load may delay app startup.
_STARTUP_LOAD_TIMEOUT: Final[int] = 60

#: Regions hosting the AWS Price List API (commercial endpoints serve identical data).
type _PricingEndpoint = Literal[
    "us-east-1", "eu-central-1", "ap-south-1", "eusc-de-east-1", "cn-north-1"
]

#: Price List API endpoint per geography prefix (us-east-1 for the rest).
_PRICING_ENDPOINT_BY_GEOGRAPHY: Final[dict[str, _PricingEndpoint]] = {
    "eu": "eu-central-1",
    "me": "eu-central-1",
    "af": "eu-central-1",
    "il": "eu-central-1",
    "ap": "ap-south-1",
    "eusc": "eusc-de-east-1",
    "cn": "cn-north-1",
}

#: Partitions with no Price List API endpoint at all (verified: no DNS).
_UNPRICED_PARTITIONS: Final[frozenset[str]] = frozenset({"aws-us-gov"})


def pricing_endpoint_region() -> _PricingEndpoint | None:
    """Nearest AWS Price List API endpoint region for this deployment.

    The API exists in three commercial regions serving identical data
    (verified live), plus one per sovereign partition serving only that
    partition's rows (e.g. eusc-de-east-1, the only source for EUSC prices
    -- commercial endpoints publish none, and cross-partition credentials
    don't work anyway). The endpoint is picked by geography instead of
    being operator-configured: from the first configured Bedrock region
    (the operator's stated regional preference), else the home region.

    Returns:
        The Price List API endpoint region, or None when the deployment
        partition has no such endpoint (GovCloud): prices then stay
        unresolved instead of failing startup on an unreachable call.
    """
    preferred = next(iter(SETTINGS.aws_bedrock_regions), None) or AWS_REGION or ""
    if partition_of_region(preferred) in _UNPRICED_PARTITIONS:
        return None
    return _PRICING_ENDPOINT_BY_GEOGRAPHY.get(_region_family(preferred), "us-east-1")


async def _fetch_service_pricing(
    pricing_client: PricingClient,
    service_code: str,
    region: str,
    diagnostics: list[str],
) -> tuple[dict[PriceKey, Price], dict[PriceKey, str]]:
    """Fetch pricing for a single AWS service code and region.

    Accumulates into fetch-local dicts: sharing across concurrent fetches
    would let colliding keys win by network completion order. Cross-fetch
    merging and collision detection happen in :func:`_load_price_catalog`.
    AWS Pricing API errors propagate to the caller.

    Args:
        pricing_client: An aiobotocore pricing client.
        service_code: A ``_SERVICE_CODE_TO_SERVICE`` key (e.g., "AmazonBedrock").
        region: The AWS region to filter by.
        diagnostics: Intra-fetch collision descriptions, appended to in place.

    Returns:
        (results, claims) -- the fetched PriceKey-to-Price mapping and each
        key's claiming usagetype (see :func:`_store_price`).
    """
    our_service = _SERVICE_CODE_TO_SERVICE[service_code]
    results: dict[PriceKey, Price] = {}
    claims: dict[PriceKey, str] = {}
    default_currency = PARTITION_CURRENCY.get(partition_of_region(region), "USD")

    async for page in pricing_client.get_paginator("get_products").paginate(
        ServiceCode=service_code,
        Filters=[{"Type": "TERM_MATCH", "Field": "regionCode", "Value": region}],
        PaginationConfig={"PageSize": 100},
    ):
        for price_list_str in page.get("PriceList", []):
            _ingest_price_list_item(
                price_list_str,
                our_service,
                region,
                default_currency,
                results,
                claims,
                diagnostics,
            )

    return results, claims


#: Fallback dimension by service (for services without inferenceType).
_SERVICE_FALLBACK_DIMENSION: Final[dict[Service, Dimension]] = {
    Service.POLLY: Dimension.INPUT_CHARACTERS,
    Service.TRANSLATE: Dimension.INPUT_CHARACTERS,
    Service.TRANSCRIBE: Dimension.INPUT_SECONDS,
    Service.COMPREHEND: Dimension.COMPREHEND_UNITS,
}


#: featuretype to Dimension; authoritative for prompt caching (inferenceType lacks it).
_FEATURE_TYPE_TO_DIMENSION: Final[dict[str, Dimension]] = {
    "prompt cache read": Dimension.CACHE_READ_TOKENS,
    "prompt cache write": Dimension.CACHE_WRITE_TOKENS,
}


def _resolve_dimension(
    our_service: Service, inference_type: str, feature_type: str, usagetype: str
) -> Dimension | None:
    """Resolve the billed dimension for one price-list item.

    Args:
        our_service: The Service this price-list entry belongs to.
        inference_type: The inferenceType attribute from AWS pricing.
        feature_type: The featuretype attribute from AWS pricing (checked
            before inferenceType -- see _FEATURE_TYPE_TO_DIMENSION).
        usagetype: The usagetype attribute.

    Returns:
        The resolved dimension, or None if it can't be determined.
    """
    if dimension := _FEATURE_TYPE_TO_DIMENSION.get(feature_type.strip().lower()):
        return dimension
    if dimension := inference_type_to_dimension(inference_type, usagetype, our_service):
        return dimension
    if usagetype:
        return _SERVICE_FALLBACK_DIMENSION.get(our_service)
    return None


def _parse_price(raw: str, scale: int, currency: str) -> Price | None:
    """Parse a raw AWS ``pricePerUnit`` string into a per-one-unit Price.

    Args:
        raw: The raw price string (e.g. "0.00008").
        scale: Units the raw price is quoted per.
        currency: The currency code to attach.

    Returns:
        The normalized per-one-unit Price, or None when *raw* is unparsable,
        non-finite, zero, or negative, or *scale* isn't positive.
    """
    try:
        price_value = Decimal(raw)
    except InvalidOperation, TypeError:
        return None
    if price_value.is_finite() and price_value > 0 and scale > 0:
        return Price(price_value / scale, currency)
    return None


def _extract_price(
    price_dim: Mapping[str, Any], scale: int | None = None, default_currency: str = ""
) -> Price | None:
    """Extract the normalized per-one-unit Price from one priceDimensions entry.

    Args:
        price_dim: One priceDimensions value from the Price List API.
        scale: Units the raw price is quoted per, or None to parse it from
            the entry's ``unit`` string.
        default_currency: Currency to assume when the price list omits one;
            with the "" default, currency-less entries yield None.

    Returns:
        The Price, or None when the currency is missing or the price is
        zero, non-finite, or unparsable.
    """
    price_per_unit = price_dim.get("pricePerUnit", {})
    if not (currency := next(iter(price_per_unit), None) or default_currency):
        return None
    if scale is None:
        scale = parse_unit_scale(price_dim.get("unit", "1"))
    return _parse_price(price_per_unit.get(currency, "0"), scale, currency)


def _synthesize_service_model_key(
    our_service: Service, attrs: Mapping[str, Any]
) -> str:
    """Build synthetic model identifier matching what usage.record_*_usage() uses.

    Polly/Translate/Transcribe/Comprehend have no `model` attribute; matching
    them requires reconstructing the exact synthetic model string from
    `engine` (Polly) or `operation` (Translate/Transcribe/Comprehend).

    Args:
        our_service: The Service this price-list entry belongs to.
        attrs: The price-list item's product.attributes.

    Returns:
        The synthetic model string (e.g., "amazon.polly-neural"), or ""
        when this row doesn't correspond to an operation this app bills for.
    """
    match our_service:
        case Service.POLLY:
            engine = attrs.get("engine", "")
            return f"amazon.polly-{engine.lower()}" if engine else ""
        case Service.TRANSLATE:
            return (
                "amazon.translate" if attrs.get("operation") == "TranslateText" else ""
            )
        case Service.TRANSCRIBE:
            return (
                "amazon.transcribe"
                if attrs.get("operation") == "TranscribeAudio"
                else ""
            )
        case Service.COMPREHEND:
            return (
                "amazon.comprehend-language-detection"
                if attrs.get("operation") == "DetectDominantLanguage"
                else ""
            )
        case _:
            return ""


#: Marketplace servicename suffix (AWS Marketplace listings).
_MARKETPLACE_SERVICENAME_SUFFIX: Final[str] = " (Amazon Bedrock Edition)"

#: Legacy "(100K)" listings: pricier SKUs that would fold onto the base model's key.
_MARKETPLACE_CONTEXT_WINDOW_LISTING_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\(\d+\s*k\)\s*$", re.IGNORECASE
)

#: usagetype substring to (Dimension, cache_ttl, spec), most specific first.
_MARKETPLACE_USAGETYPE_DIMENSIONS: Final[
    tuple[tuple[str, Dimension, CacheTtlBucket, str], ...]
] = (
    ("cachewrite1h", Dimension.CACHE_WRITE_TOKENS, "1h", ""),
    ("cachewritetokens1h", Dimension.CACHE_WRITE_TOKENS, "1h", ""),
    ("cachewrite", Dimension.CACHE_WRITE_TOKENS, "", ""),
    ("cacheread", Dimension.CACHE_READ_TOKENS, "", ""),
    ("inputtoken", Dimension.INPUT_TOKENS, "", ""),
    ("outputtoken", Dimension.OUTPUT_TOKENS, "", ""),
    ("createdimage", Dimension.OUTPUT_IMAGES, "", ""),
    ("searchunit", Dimension.SEARCH_UNITS, "", ""),
    ("inputimagecount", Dimension.INPUT_IMAGES, "", ""),
    ("inputaudiosecond", Dimension.INPUT_SECONDS, "", "audio"),
    ("inputvideosecond", Dimension.INPUT_SECONDS, "", "video"),
)

#: Dimensions Marketplace listings quote per 1M units (their unit="Units" is opaque).
_MARKETPLACE_PER_MILLION_DIMENSIONS: Final[frozenset[Dimension]] = frozenset(
    {
        Dimension.INPUT_TOKENS,
        Dimension.OUTPUT_TOKENS,
        Dimension.CACHE_READ_TOKENS,
        Dimension.CACHE_WRITE_TOKENS,
    }
)


def _normalize_usagetype(usagetype: str) -> str:
    """Lowercase and strip "-"/"_"/" " separators, for substring matching."""
    return usagetype.lower().replace("-", "").replace("_", "").replace(" ", "")


def _marketplace_dimension_tier_ttl(
    usagetype: str,
) -> tuple[Dimension | None, str, CacheTtlBucket, str]:
    """Resolve (dimension, tier, cache_ttl, spec) from a Marketplace-listing usagetype.

    Returns:
        dimension is None when the usagetype doesn't match any known pattern.
    """
    normalized = _normalize_usagetype(usagetype)
    tier = "batch" if "batch" in normalized else "standard"
    for pattern, dimension, cache_ttl, spec in _MARKETPLACE_USAGETYPE_DIMENSIONS:
        if pattern in normalized:
            return dimension, tier, cache_ttl, spec
    return None, tier, "", ""


def _marketplace_routing(usagetype: str) -> Routing:
    """Resolve the serving profile ("latency", "global" or "") from a Marketplace usagetype.

    Global routing is paradoxically cheaper than plain (~$3.00/M vs $3.30/M
    for Claude Sonnet 4.5); "_LatencyOptimized" is pricier than both.
    """
    normalized = _normalize_usagetype(usagetype)
    if "latencyoptimized" in normalized:
        return "latency"
    return "global" if "global" in normalized else ""


def _store_price(
    results: dict[PriceKey, Price],
    claims: dict[PriceKey, str],
    key: PriceKey,
    price: Price,
    usagetype: str,
    diagnostics: list[str],
) -> None:
    """Merge resolved price into results, recording cross-row PriceKey collisions.

    Different usagetypes claiming the same PriceKey with different prices
    means an unmodeled pricing axis is folding distinct rates onto one key.
    Repeated bands from the same usagetype (tiered volume pricing) are
    expected and ignored.

    Args:
        results: Mapping to merge the resolved price into, in place.
        claims: Tracks which usagetype last claimed each PriceKey, scoped to
            one ingestion batch.
        key: The resolved PriceKey for this price point.
        price: The resolved Price.
        usagetype: The usagetype attribute of the row producing this price.
        diagnostics: Collision descriptions for this batch, appended to in
            place; surfaced as one summary warning rather than per collision.
    """
    existing_price = results.get(key)
    existing_usagetype = claims.get(key)
    if (
        existing_price is not None
        and existing_usagetype is not None
        and existing_usagetype != usagetype
        and existing_price != price
    ):
        diagnostics.append(
            f"Price catalog collision on {key}: usagetype {existing_usagetype!r} "
            f"(${existing_price.amount}) overwritten by {usagetype!r} (${price.amount})"
        )
    results[key] = price
    claims[key] = usagetype


def _ingest_marketplace_item(
    attrs: Mapping[str, Any],
    item: Mapping[str, Any],
    region: str,
    results: dict[PriceKey, Price],
    claims: dict[PriceKey, str],
    diagnostics: list[str],
) -> None:
    """Parse one AWS Marketplace-listed Bedrock price item into results.

    Args:
        attrs: The price-list item's ``product.attributes``.
        item: The full parsed price-list item (for ``terms``).
        region: The AWS region being fetched.
        results: Mapping to merge resolved price points into, in place.
        claims: Tracks which usagetype last claimed each PriceKey -- see
            :func:`_store_price`.
        diagnostics: Collision descriptions, appended to in place -- see
            :func:`_store_price`.
    """
    listing_name = attrs["servicename"][: -len(_MARKETPLACE_SERVICENAME_SUFFIX)]
    if _MARKETPLACE_CONTEXT_WINDOW_LISTING_PATTERN.search(listing_name):
        return  # Legacy "(100K)" SKU -- would fold onto the base model's key.
    if not (normalized_model := normalize_model_key(listing_name)):
        return

    usagetype = attrs.get("usagetype", "")
    dimension, tier, cache_ttl, spec = _marketplace_dimension_tier_ttl(usagetype)
    if dimension is None:
        return
    routing = _marketplace_routing(usagetype)

    if not (terms := item.get("terms", {}).get("OnDemand", {})):
        return

    scale = 1_000_000 if dimension in _MARKETPLACE_PER_MILLION_DIMENSIONS else 1
    key = PriceKey(
        _bedrock_api_service(Service.BEDROCK, usagetype),
        normalized_model,
        region,
        dimension,
        tier,
        cache_ttl,
        routing,
        spec,
        _price_context(usagetype),
    )
    # Distinct Marketplace products share identical usagetype strings, so
    # claims must include the listing name to diagnose cross-product overwrites.
    claim_tag = f"{listing_name}:{usagetype}"
    for term in terms.values():
        for price_dim in term.get("priceDimensions", {}).values():
            if price := _extract_price(price_dim, scale):
                _store_price(results, claims, key, price, claim_tag, diagnostics)


def _ingest_price_list_item(
    price_list_str: str,
    our_service: Service,
    region: str,
    default_currency: str,
    results: dict[PriceKey, Price],
    claims: dict[PriceKey, str] | None = None,
    diagnostics: list[str] | None = None,
) -> None:
    """Parse one AWS Price List item and merge its price points into *results*.

    Dispatches to the AWS Marketplace listing path (see
    :func:`_ingest_marketplace_item`) or the native-Bedrock path
    (:func:`_ingest_native_item`) based on the `servicename` attribute.

    Args:
        price_list_str: One raw JSON price-list entry from the Price List API.
        our_service: The Service this price-list entry belongs to.
        region: The AWS region being fetched.
        default_currency: Currency to assume when the price list omits one.
        results: Mapping to merge resolved price points into, in place.
        claims: Tracks which usagetype last claimed each PriceKey, for
            cross-item collision detection (see :func:`_store_price`) --
            share one across every item in an ingestion batch to catch
            collisions between them. Defaults to a fresh (single-item-scoped)
            dict when omitted, e.g. for one-off calls that don't need it.
        diagnostics: Collision descriptions, appended to in place -- see
            :func:`_store_price`. Defaults to a fresh (discarded) list when
            omitted, e.g. for one-off calls that don't need it.
    """
    try:
        item = json.loads(price_list_str)
    except json.JSONDecodeError:
        return

    attrs = item.get("product", {}).get("attributes", {})
    if claims is None:
        claims = {}
    if diagnostics is None:
        diagnostics = []

    if our_service == Service.BEDROCK and attrs.get("servicename", "").endswith(
        _MARKETPLACE_SERVICENAME_SUFFIX
    ):
        _ingest_marketplace_item(attrs, item, region, results, claims, diagnostics)
    else:
        _ingest_native_item(
            attrs,
            item,
            our_service,
            region,
            default_currency,
            results,
            claims,
            diagnostics,
        )


#: inferenceType suffixes signaling a non-standard tier when service_tier is absent.
_INFERENCE_TYPE_TIER_SUFFIXES: Final[tuple[str, ...]] = ("flex", "priority", "batch")


def _resolve_tier(attrs: Mapping[str, Any]) -> str:
    """Resolve the service tier (standard/flex/priority/batch) for one price-list item.

    AWS signals tier four incompatible ways across its price-list schemas:
    a `service_tier` attribute, a trailing `inferenceType` word ("Output
    tokens flex"), `feature` == "Batch Inference", or a usagetype segment
    ("...batch...", "...-<tier>" suffix). Missing any of them folds that
    tier's rows onto the standard-tier PriceKey.

    Args:
        attrs: The price-list item's ``product.attributes``.

    Returns:
        One of "standard", "flex", "priority", "batch".
    """
    if service_tier := attrs.get("service_tier"):
        return str(service_tier).lower()
    inference_type = attrs.get("inferenceType", "").strip().lower()
    for suffix in _INFERENCE_TYPE_TIER_SUFFIXES:
        if inference_type.endswith(f" {suffix}"):
            return suffix
    # AWS spells this both "Batch Inference" and "BatchInference".
    if attrs.get("feature", "").strip().lower().replace(" ", "") == "batchinference":
        return "batch"
    normalized_usagetype = _normalize_usagetype(attrs.get("usagetype", ""))
    if "batch" in normalized_usagetype:
        return "batch"
    # flex/priority are suffix-anchored to avoid matching mid-string text.
    for suffix in ("flex", "priority"):
        if normalized_usagetype.endswith(suffix):
            return suffix
    return "standard"


def _bedrock_api_service(our_service: Service, usagetype: str) -> Service:
    """Key Bedrock rows under their invocation API (bedrock-runtime vs bedrock-mantle).

    The two APIs have distinct published rates for the same model (confirmed
    live: qwen3-next-80b-a3b in ap-south-1); "mantle" rows are signaled by a
    usagetype segment.

    Args:
        our_service: The Service this price-list entry belongs to.
        usagetype: The usagetype attribute.

    Returns:
        ``Service.BEDROCK_MANTLE`` for mantle rows, else *our_service*.
    """
    if our_service == Service.BEDROCK and "mantle" in _normalize_usagetype(usagetype):
        return Service.BEDROCK_MANTLE
    return our_service


def _price_context(usagetype: str) -> ContextLength:
    """Resolve the context-length bucket from a usagetype.

    The >200K-prompt rate is signaled by a "long-context" usagetype segment
    on every affected dimension, including the prompt-cache ones.

    Args:
        usagetype: The usagetype attribute (native or Marketplace).

    Returns:
        "long" or "" (standard).
    """
    return "long" if "longcontext" in _normalize_usagetype(usagetype) else ""


def _native_routing(
    our_service: Service, attrs: Mapping[str, Any], usagetype: str
) -> Routing:
    """Resolve the serving profile ("latency", "global" or "") for a native row.

    Global routing is a "-cross-region-global" usagetype suffix;
    latency-optimized rates are a " Latency Optimized" `model` attribute
    suffix (e.g. "Nova Pro Latency Optimized"). Only meaningful for Bedrock.

    Args:
        our_service: The Service this price-list entry belongs to.
        attrs: The price-list item's ``product.attributes``.
        usagetype: The usagetype attribute.

    Returns:
        "latency", "global" or "" (plain/regional).
    """
    if our_service != Service.BEDROCK:
        return ""
    blob = _normalize_usagetype(usagetype + (attrs.get("model") or ""))
    if "latencyoptimized" in blob:
        return "latency"
    return "global" if "crossregionglobal" in blob else ""


#: Image-generation inferenceType "T2I/I2I <res> <quality>" ("Custom " never matches).
_IMAGE_GENERATION_SPEC_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?:t2i|i2i)\s+(\d+)\s+(standard|premium)$", re.IGNORECASE
)


#: Video-generation inferenceType: "T2V/I2V ..." (Nova Reel) or bare "Video" (Luma Ray).
_VIDEO_GENERATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:t2v|i2v)\s|video$)", re.IGNORECASE
)


def _native_price_spec(inference_type: str, usagetype: str) -> str:
    """Resolve the media/modality spec bucket for a native non-image-spec row.

    Disambiguates same-dimension rows priced per modality or resolution:
    Nova Sonic speech vs text tokens, Nova MME document vs standard input
    images and audio vs video seconds, Nova Reel HD vs standard output video.

    Args:
        inference_type: The inferenceType attribute.
        usagetype: The usagetype attribute.

    Returns:
        "speech", "document", "audio", "video", "hd", or "" (default bucket).
    """
    if inference_type.strip().lower().startswith("speech understanding"):
        return "speech"
    normalized = _normalize_usagetype(usagetype)
    for pattern, spec in (
        ("inputdocumentimage", "document"),
        ("inputaudiosecond", "audio"),
        ("inputvideosecond", "video"),
        ("hdres", "hd"),
    ):
        if pattern in normalized:
            return spec
    return ""


def _image_generation_spec(inference_type: str) -> str:
    """Resolve a spec key ("<resolution>:<quality>") from an image-generation inferenceType.

    Args:
        inference_type: The inferenceType attribute.

    Returns:
        "<resolution>:<quality>" (e.g. "1024:standard"), or "" if unmatched
        (including deliberately, for "Custom " variants -- see
        :data:`_IMAGE_GENERATION_SPEC_PATTERN`).
    """
    if not (match := _IMAGE_GENERATION_SPEC_PATTERN.match(inference_type.strip())):
        return ""
    resolution, quality = match.groups()
    return f"{resolution}:{quality.lower()}"


def _resolve_native_model(
    our_service: Service, attrs: Mapping[str, Any], usagetype: str
) -> tuple[str, str]:
    """Resolve the normalized model key (and spec bucket, if applicable) for a native item.

    Titan Image Generator rows carry no `model` attribute -- only `titanModel`
    (e.g. "Titan Image Generator G1") -- and their inferenceType encodes a
    resolution/quality that can't be separated from the model name, so they
    get a dedicated path. Nova Canvas uses the same inferenceType shape but
    does carry a `model` attribute.

    Args:
        our_service: The Service this price-list entry belongs to.
        attrs: The price-list item's product.attributes.
        usagetype: The usagetype attribute.

    Returns:
        (normalized_model, spec) -- normalized_model is "" when this
        row shouldn't be ingested at all (e.g. a Titan Image Generator
        "Custom" row).
    """
    inference_type = attrs.get("inferenceType", "")
    if model_attr := attrs.get("model", ""):
        spec = _image_generation_spec(inference_type) or _native_price_spec(
            inference_type, usagetype
        )
        return normalize_model_key(model_attr), spec

    if "image generator" in (titan_model := attrs.get("titanModel", "")).lower():
        spec = _image_generation_spec(inference_type)
        if not spec:
            return "", ""  # Customized/fine-tuned variant -- not billed by this app.
        return normalize_model_key(titan_model), spec

    if synthetic_model := _synthesize_service_model_key(our_service, attrs):
        return normalize_model_key(synthetic_model), ""
    if our_service == Service.BEDROCK:
        return normalize_usagetype_model(usagetype), _native_price_spec(
            inference_type, usagetype
        )
    return normalize_usagetype_model(usagetype), ""


def _ingest_native_item(
    attrs: Mapping[str, Any],
    item: Mapping[str, Any],
    our_service: Service,
    region: str,
    default_currency: str,
    results: dict[PriceKey, Price],
    claims: dict[PriceKey, str],
    diagnostics: list[str],
) -> None:
    """Parse one natively-priced (non-Marketplace) AWS Price List item into *results*.

    Args:
        attrs: The price-list item's ``product.attributes``.
        item: The full parsed price-list item (for ``terms``).
        our_service: The Service this price-list entry belongs to.
        region: The AWS region being fetched.
        default_currency: Currency to assume when the price list omits one.
        results: Mapping to merge resolved price points into, in place.
        claims: Tracks which usagetype last claimed each PriceKey -- see
            :func:`_store_price`.
        diagnostics: Collision descriptions, appended to in place -- see
            :func:`_store_price`.
    """
    # Non-on-demand rate families (fine-tuned copies, monthly Provisioned
    # Throughput, reserved token-per-minute capacity): this app never bills
    # them, and their rows would collide with the base on-demand PriceKeys.
    feature = attrs.get("feature", "").strip().lower()
    if feature == "model customization" or feature.startswith(
        ("provisioned throughput", "reserved")
    ):
        return

    usagetype = attrs.get("usagetype", "")
    if (
        dimension := _resolve_dimension(
            our_service,
            attrs.get("inferenceType", ""),
            attrs.get("featuretype", ""),
            usagetype,
        )
    ) is None:
        return

    normalized_model, spec = _resolve_native_model(our_service, attrs, usagetype)
    if not normalized_model:
        return

    if not (terms := item.get("terms", {}).get("OnDemand", {})):
        return

    cache_ttl: CacheTtlBucket = (
        "1h"
        if dimension == Dimension.CACHE_WRITE_TOKENS
        and "1hour" in _normalize_usagetype(usagetype)
        else ""
    )
    key = PriceKey(
        _bedrock_api_service(our_service, usagetype),
        normalized_model,
        region,
        dimension,
        _resolve_tier(attrs),
        cache_ttl,
        _native_routing(our_service, attrs, usagetype),
        spec,
        _price_context(usagetype),
    )
    for term in terms.values():
        for price_dim in term.get("priceDimensions", {}).values():
            # Same-item price bands (tiered beginRange/endRange rows) share
            # one usagetype: last band wins, silently by design.
            if price := _extract_price(price_dim, default_currency=default_currency):
                _store_price(results, claims, key, price, usagetype, diagnostics)


def _catalog_regions() -> set[str]:
    """Regions to fetch pricing for: configured service regions plus safe fallbacks.

    Returns:
        Non-empty set of AWS region codes.
    """
    # Always include the regional-fallback anchor regions, where most models
    # are available -- keeps this set and _FALLBACK_ANCHOR_REGIONS in sync.
    return (
        set(SETTINGS.aws_bedrock_regions)
        | set(_FALLBACK_ANCHOR_REGIONS)
        | {
            r
            for attr in (
                "aws_polly_region",
                "aws_transcribe_region",
                "aws_translate_region",
                "aws_comprehend_region",
            )
            if (r := getattr(SETTINGS, attr, None))
        }
        | ({AWS_REGION} if AWS_REGION else set())
    ) or {"us-east-1"}


def _apply_default_prices(index: dict[PriceKey, Price]) -> None:
    """Backfill registered built-in default prices into *index*, in place.

    Model-level guard: a model with any published row keeps its published
    prices only, so pricing-page defaults never mix with real rows.

    Args:
        index: Price index to update.
    """
    published_models = {key.model for key in index if key.service == Service.BEDROCK}
    index.update(
        {
            key: price
            for key, price in _DEFAULT_PRICES.items()
            if key.model not in published_models
        }
    )


def _apply_price_overrides(
    index: dict[PriceKey, Price],
    regions: set[str],
    diagnostics: list[str] | None = None,
) -> None:
    """Merge operator-supplied ``COST_PRICE_OVERRIDES`` into *index*, in place.

    Applies each override to every configured region, using that region's own
    partition to resolve the override's currency (regions can span multiple
    AWS partitions -- aws, aws-eusc, aws-us-gov, aws-cn -- each with its own
    currency). Always applied at the standard tier -- there's no override
    mechanism for flex/priority/batch pricing.

    Args:
        index: Price index to update.
        regions: Regions to apply each override to.
        diagnostics: Invalid-override descriptions, appended to in place --
            surfaced as one summary warning by the caller. Defaults to a
            fresh (discarded) list when omitted, e.g. for one-off calls that
            don't need it.
    """
    if diagnostics is None:
        diagnostics = []
    for model_key, dimension_prices in SETTINGS.cost_price_overrides.items():
        # resolve (not just normalize): overrides for models with a registered
        # key override must land where resolve_price will look them up.
        normalized_model = resolve_model_key(model_key)
        for dim_str, price in dimension_prices.items():
            try:
                dimension = Dimension(dim_str)
            except ValueError:
                diagnostics.append(
                    f"cost_price_overrides: unknown dimension {dim_str!r} for model "
                    f"{model_key!r}, ignoring"
                )
                continue
            if not math.isfinite(price) or price <= 0:
                diagnostics.append(
                    f"cost_price_overrides: invalid price {price!r} for "
                    f"{model_key!r}/{dim_str!r}, ignoring"
                )
                continue
            for region in regions:
                currency = PARTITION_CURRENCY.get(partition_of_region(region), "USD")
                index[
                    PriceKey(
                        Service.BEDROCK, normalized_model, region, dimension, "standard"
                    )
                ] = Price(Decimal(str(price)), currency)


#: Always-fetched last-resort fallback source regions, tried in this order.
_FALLBACK_ANCHOR_REGIONS: Final[tuple[str, ...]] = (
    "us-east-1",
    "eu-west-1",
    "us-west-2",
)


#: Regional-fallback group key: every PriceKey axis except region.
type _FallbackGroupKey = tuple[
    Service, str, Dimension, str, CacheTtlBucket, Routing, str, ContextLength
]


def _region_family(region: str) -> str:
    """Geo-prefix of an AWS region code.

    Args:
        region: AWS region code (e.g. "eu-west-3", "ap-southeast-2").

    Returns:
        The region's geography prefix, up to its first hyphen (e.g. "eu", "ap").
    """
    return region.split("-", 1)[0]


def _apply_regional_fallback(index: dict[PriceKey, Price], regions: set[str]) -> None:
    """Backfill regions with no price for a group (every axis but region), in place.

    A best-effort approximation for models AWS hasn't published a price for
    in every region (common for older/deprecated models). Prefers a region
    in the same geography (:func:`_region_family`), then falls back to
    :data:`_FALLBACK_ANCHOR_REGIONS`. Never crosses a partition boundary: a
    copied :class:`Price` keeps its source currency, so e.g. a eusc-*/EUR
    region backfilled from an "aws"/USD source would silently report the
    wrong currency -- such regions are left unpriced instead.

    Args:
        index: Price index to backfill, updated in place.
        regions: All regions the catalog was fetched for.
    """
    # Group by EVERY axis except region: dropping any (e.g. routing) would
    # collide same-model keys in region_keys and backfill the wrong price.
    groups: dict[_FallbackGroupKey, dict[str, PriceKey]] = {}
    for key in index:
        group: _FallbackGroupKey = (
            key.service,
            key.model,
            key.dimension,
            key.tier,
            key.cache_ttl,
            key.routing,
            key.spec,
            key.context,
        )
        groups.setdefault(group, {})[key.region] = key

    for group, region_keys in groups.items():
        for region in regions - region_keys.keys():
            target_partition = partition_of_region(region)
            family = _region_family(region)
            same_family = sorted(
                r
                for r in region_keys
                if _region_family(r) == family
                and partition_of_region(r) == target_partition
            )
            source_region = (
                same_family[0]
                if same_family
                else next(
                    (
                        r
                        for r in _FALLBACK_ANCHOR_REGIONS
                        if r in region_keys
                        and partition_of_region(r) == target_partition
                    ),
                    None,
                )
            )
            if source_region is not None:
                index[PriceKey(group[0], group[1], region, *group[2:])] = index[
                    region_keys[source_region]
                ]


async def _load_price_catalog(diagnostics: list[str]) -> None:
    """Fetch pricing for all configured regions/services and atomically swap the index.

    AWS Pricing API errors propagate to the caller.

    Args:
        diagnostics: Collision and invalid-override descriptions for this
            load, appended to in place; the caller surfaces them as one
            warning on its own operation-level log event.
    """
    if (endpoint := pricing_endpoint_region()) is None:
        diagnostics.append(
            "The AWS Price List API has no endpoint in this partition;"
            " only cost_price_overrides prices apply"
        )
        # Overrides are local (no API): keep them as the sole price source.
        override_index: dict[PriceKey, Price] = {}
        _apply_price_overrides(override_index, _catalog_regions(), diagnostics)
        _state.price_index = override_index
        return

    regions = _catalog_regions()
    fetch_specs = [
        (region, service_code)
        for region in sorted(regions)
        for service_code in _SERVICE_CODE_TO_SERVICE
    ]

    # type-ignore: the RegionName stub Literal lags EUSC/China (works live).
    client = get_client("pricing", endpoint)  # type: ignore[arg-type]
    fetch_results = await asyncio.gather(
        *(
            _fetch_service_pricing(client, service_code, region, diagnostics)
            for region, service_code in fetch_specs
        )
    )

    # Merge in deterministic fetch_specs order (not completion order) via
    # _store_price with one shared claims dict, so cross-fetch PriceKey
    # collisions (e.g. the three Bedrock service codes) warn and resolve
    # deterministically. Claims are tagged with the service code so equal
    # usagetype text across fetches isn't mistaken for a repeated price band.
    new_index: dict[PriceKey, Price] = {}
    claims: dict[PriceKey, str] = {}
    for (_region, service_code), (fetch_index, fetch_claims) in zip(
        fetch_specs, fetch_results, strict=True
    ):
        for key, price in fetch_index.items():
            _store_price(
                new_index,
                claims,
                key,
                price,
                f"{service_code}:{fetch_claims[key]}",
                diagnostics,
            )

    _apply_default_prices(new_index)
    _apply_regional_fallback(new_index, regions)
    _apply_price_overrides(new_index, regions, diagnostics)

    _state.price_index = new_index


def _all_models_priced(
    model_ids: Iterable[str], service: Service = Service.BEDROCK
) -> bool:
    """Check whether every model in *model_ids* has a price-catalog entry.

    Builds the priced-key set once per batch instead of scanning the whole
    index per model.

    Args:
        model_ids: Model IDs, resolved via :func:`resolve_model_key`.
        service: The service to check against. Defaults to Bedrock.

    Returns:
        True if every model has at least one :class:`PriceKey` for *service*.
    """
    priced_keys = {key.model for key in _state.price_index if key.service == service}
    return all(resolve_model_key(model_id) in priced_keys for model_id in model_ids)


def is_model_priced(model_id: str, service: Service = Service.BEDROCK) -> bool:
    """Check whether the price index has at least one entry for *model_id*.

    Args:
        model_id: The model ID, resolved via :func:`resolve_model_key`.
        service: The service to check against. Defaults to Bedrock.

    Returns:
        True if at least one :class:`PriceKey` in the current price index
        matches this model's normalized key for *service*.
    """
    return _all_models_priced((model_id,), service)


async def refresh_price_catalog_for_new_models(model_ids: Iterable[str]) -> None:
    """Immediately refresh the price catalog if any of *model_ids* isn't priced yet.

    Called by ``initialize_bedrock_models()`` when its own lazy on-demand
    refresh (never the initial startup call, which already gets a full
    catalog from :func:`start_price_catalog`) discovers Bedrock models that
    weren't previously registered. Like the Bedrock model cache's own
    on-demand refresh, this is a blocking, awaited reload that can add
    latency to whichever request triggers it, keeping the price catalog
    self-healing for newly released models without a proactive polling loop.

    The reload is silent: its diagnostics are discarded, and AWS Pricing
    API errors propagate to the caller.

    Args:
        model_ids: Bedrock model IDs just discovered by
            ``initialize_bedrock_models()``. No-op when cost tracking is
            disabled, or when every given model already has at least one
            price-catalog entry.
    """
    if not SETTINGS.cost_tracking:
        return
    model_ids = list(model_ids)
    if not model_ids or _all_models_priced(model_ids):
        return

    async with _state.refresh_lock:
        # Re-check under the lock: a concurrent caller may have already
        # refreshed the catalog while this one was waiting for it.
        if _all_models_priced(model_ids):
            return
        await _load_price_catalog([])


async def start_price_catalog(start_event: EventLog | None = None) -> None:
    """Initialize the price catalog at application startup.

    Should be called during application startup. Diagnostics collected while
    loading the catalog (collisions, invalid overrides, or a slow initial
    load) are recorded as a single warning on *start_event*, if given, via
    :func:`stdapi.monitoring.add_server_warning` (imported inside the
    function: ``stdapi.monitoring`` transitively imports this module).
    AWS Pricing API failures other than the startup timeout propagate to
    the caller.

    There is no periodic background refresh afterward: the catalog is kept
    current on demand by :func:`refresh_price_catalog_for_new_models`,
    triggered by ``initialize_bedrock_models()`` whenever its own lazy
    refresh discovers a model with no price-catalog entry.

    Args:
        start_event: Optional startup event log to record a diagnostics
            summary on.
    """
    if not SETTINGS.cost_tracking:
        return

    diagnostics: list[str] = []
    try:
        await asyncio.wait_for(
            _load_price_catalog(diagnostics), timeout=_STARTUP_LOAD_TIMEOUT
        )
    except TimeoutError:
        # The cancelled load never swapped the index: the catalog stays empty.
        diagnostics.append(
            f"Price catalog initialization exceeded {_STARTUP_LOAD_TIMEOUT}s; continuing "
            "startup with an empty catalog"
        )
    if diagnostics and start_event is not None:
        from stdapi.monitoring import add_server_warning  # noqa: PLC0415

        # add_server_warning serializes concurrent startup writers under its lock.
        add_server_warning(
            start_event,
            {"price_catalog": diagnostics},  # type: ignore[dict-item]
        )
