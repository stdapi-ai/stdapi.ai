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
import math
import re
from contextvars import Context
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any, Final, Literal

from botocore.exceptions import BotoCoreError, ClientError
from pydantic_core import from_json

from stdapi.aws import get_client
from stdapi.config import AWS_REGION, SETTINGS

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from collections.abc import Set as AbstractSet

    from types_aiobotocore_pricing.client import PricingClient

    from stdapi.config import LogLevel


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
    TEXT_UNITS = "text_units"


class Service(StrEnum):
    """AWS services/APIs that have pricing support.

    Bedrock is split per invocation API: bedrock-runtime (Converse) and
    bedrock-mantle usage are recorded and priced independently.
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


#: Parse scale from unit strings: "1K tokens" -> 1000, "1M" -> 1e6, "1000 tokens" -> 1000.
_UNIT_SCALE_PATTERN = re.compile(r"^(\d+)([KM])?(?:\s|$)", re.IGNORECASE)


def parse_unit_scale(unit: str) -> int:
    """Parse the scale multiplier from a price dimension unit.

    Args:
        unit: The unit string (e.g., "1K tokens", "1M", "1000 tokens", "1 image").

    Returns:
        Units the price is quoted per -- e.g. "10K tokens" -> 10000, "1 image" -> 1.
    """
    if not (match := _UNIT_SCALE_PATTERN.match(unit)):
        return 1
    count, multiplier = int(match.group(1)), match.group(2)
    if not multiplier:
        return count
    return count * (1000 if multiplier.upper() == "K" else 1_000_000)


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
    (rerank) and the Guardrails rows are matched before the generic "unit".

    Args:
        usagetype: The usagetype attribute.

    Returns:
        The resolved dimension, or None.
    """
    normalized = _normalize_usagetype(usagetype)
    if "searchunit" in normalized:
        return Dimension.SEARCH_UNITS
    if "guardrail" in normalized:
        # Every policy's usagetype is "...UnitsConsumed", which the generic
        # branch below would read as generated images. Guardrails bill per
        # 1,000-character text unit, or per image for the image content
        # policy -- the two dimensions the guardrail usage records use.
        return (
            Dimension.INPUT_IMAGES
            if "imageunit" in normalized
            else Dimension.TEXT_UNITS
        )
    if "token" not in normalized and (
        "unit" in normalized or ("image" in normalized and "input" not in normalized)
    ):
        return Dimension.OUTPUT_IMAGES
    return None


#: usagetype substring to dimension for inferenceType-less native rows (ordered).
_USAGETYPE_FALLBACK_DIMENSIONS: Final[tuple[tuple[str, Dimension], ...]] = (
    ("novagrounding", Dimension.GROUNDING_REQUESTS),
    ("websearchqueries", Dimension.GROUNDING_REQUESTS),
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
        # Split on "_", MP:'s own separator: the "-"-split `parts` above still
        # carry "MP:" and can never match a bare 4-char region code.
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
    # Background startup load task, owned here and cancelled at shutdown.
    load_task: asyncio.Task[None] | None = None
    # (region, service_code) pairs still failing after the last load attempt,
    # or None when the last attempt had nothing left to retry.
    pending_fetch_specs: list[tuple[str, str]] | None = None
    #: True once a _load_price_catalog call published a catalog with no failed fetch.
    catalog_complete: bool = False
    # Raw fetch results/claims accumulated across retry attempts for
    # pending_fetch_specs (pre default-price/fallback/override backfill).
    pending_index: dict[PriceKey, Price] = field(default_factory=dict)
    pending_claims: dict[PriceKey, str] = field(default_factory=dict)
    # Model ID to perf_counter_ns() expiry: models a completed reload still
    # couldn't price, exempted from retriggering one until the cooldown lapses.
    unpriced_cooldown: dict[str, int] = field(default_factory=dict)
    # (indexed dict, its rows grouped by model): rebuilt when price_index is
    # swapped; holding the indexed dict makes the identity check GC-safe.
    rows_by_model: (
        tuple[dict[PriceKey, Price], dict[str, list[tuple[PriceKey, Price]]]] | None
    ) = None
    # (indexed dict, model keys with a batch-tier row): same identity-check
    # caching as rows_by_model, for the catalogue's batch advertisement.
    batch_priced: tuple[dict[PriceKey, Price], frozenset[str]] | None = None


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
    ``pricing_overrides.DEFAULT_MODEL_PRICES`` table, and below for the
    model-less rates this module mints a synthetic model for. Applied at
    catalog load only to models with no published row at all (see
    :func:`_apply_default_prices`); ``cost_price_overrides`` still wins.

    Args:
        prices: Model ID to per-dimension USD price (exact decimal text).
        regions: Regions the prices apply to, per the pricing page.
    """
    for model_id, dimension_prices in prices.items():
        model = resolve_model_key(model_id)
        for dimension, amount in dimension_prices.items():
            for region in regions:
                # The pricing page doesn't distinguish invocation APIs; key
                # under both so Mantle-only models resolve at runtime too.
                for service in (Service.BEDROCK, Service.BEDROCK_MANTLE):
                    _DEFAULT_PRICES[
                        PriceKey(service, model, region, dimension, "standard")
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
    # long-context price beats an exact-tier standard-context one. Then tier
    # (a scaled standard price only when the exact tier isn't indexed, and the
    # ratio only applies to token rates), then routing. cache_ttl and spec
    # never coexist, so they relax together as one axis pair.
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

#: Bounds concurrent Pricing API fetches -- the API enforces a very low request-rate quota.
_MAX_CONCURRENT_FETCHES: Final[int] = 4

#: Initial delay before retrying a failed background catalog load.
_LOAD_RETRY_INITIAL_SECONDS: Final[int] = 60

#: Max delay between background catalog load retries.
_LOAD_RETRY_MAX_SECONDS: Final[int] = 900

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

    Picked by geography from the first configured Bedrock region, else the
    home region. Sovereign partitions (e.g. EUSC) have their own endpoint
    serving only that partition's prices; commercial endpoints publish none.

    Returns:
        The Price List API endpoint region, or None when the deployment
        partition has no such endpoint (GovCloud).
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


#: Prefix shared by every synthetic Bedrock Guardrails model.
GUARDRAIL_MODEL_PREFIX: Final = "amazon.bedrock-runtime-guardrail"

#: Synthetic model the built-in web search tool's per-query rate is billed against.
WEB_SEARCH_MODEL: Final = "amazon.bedrock-web-search"

#: usagetype fragment identifying the model-less Bedrock web search rows.
_WEB_SEARCH_USAGETYPE_FRAGMENT: Final = "websearchqueries"

#: Synthetic model a managed knowledge base's per-retrieval rate is billed against.
KNOWLEDGE_BASE_MODEL: Final = "amazon.bedrock-knowledge-base"

#: Managed knowledge base retrieval, $1.00 per 1,000 calls (aws.amazon.com/bedrock/pricing/).
_KNOWLEDGE_BASE_RETRIEVAL_PRICE: Final = "0.001"

#: Regions the pricing page quotes the managed knowledge base rates for.
_KNOWLEDGE_BASE_PRICE_REGIONS: Final = ("us-east-1", "us-east-2", "us-west-2")

# The Price List API publishes no row for it, so the pricing page's rate is the
# only source; the regional fallback carries it to the other configured regions.
register_default_prices(
    {KNOWLEDGE_BASE_MODEL: {Dimension.SEARCH_UNITS: _KNOWLEDGE_BASE_RETRIEVAL_PRICE}},
    _KNOWLEDGE_BASE_PRICE_REGIONS,
)

#: Guardrails usagetype fragment to the policy slug it prices (ordered).
_GUARDRAIL_POLICY_SLUGS: Final[tuple[tuple[str, str], ...]] = (
    ("contentpolicy", "content"),
    ("topicpolicy", "topic"),
    ("wordpolicy", "word"),
    ("sensitiveinformationpolicyfree", "sensitive-information-free"),
    ("sensitiveinformationpolicypaid", "sensitive-information"),
    ("contextualgroundingpolicy", "contextual-grounding"),
    ("automatedreasoningpolicy", "automated-reasoning"),
)


def guardrail_policy_model(policy: str) -> str:
    """Build the synthetic model a guardrail policy's usage is billed against.

    Args:
        policy: The policy slug (see :data:`_GUARDRAIL_POLICY_SLUGS`).

    Returns:
        The synthetic model string.
    """
    return f"{GUARDRAIL_MODEL_PREFIX}-{policy}"


def _is_guardrail_usagetype(usagetype: str) -> bool:
    """Whether *usagetype* names a Bedrock Guardrails policy or guardrail check."""
    return "guardrail" in _normalize_usagetype(usagetype)


def _guardrail_model(usagetype: str) -> str:
    """Build the synthetic guardrail model string for a Bedrock Guardrails row.

    Guardrails rows carry no `model` attribute, and their usagetype names the
    evaluated policy rather than a model, so the billed model is read from the
    usagetype instead. AWS prices each policy separately and ApplyGuardrail
    reports each policy's units separately, so every policy gets its own model
    rather than sharing one: a guardrail applies every policy the operator
    configured, and their rates sum.

    Args:
        usagetype: The usagetype attribute.

    Returns:
        The synthetic model string the guardrail usage records bill against,
        or "" when the usagetype names no guardrail, a check the gateway never
        requests, or a policy this app does not model.
    """
    if not _is_guardrail_usagetype(usagetype):
        return ""
    normalized = _normalize_usagetype(usagetype)
    if "check" in normalized:
        # InvokeGuardrailChecks is only ever called with the content filter,
        # so the other checks' rates would fold a rate this app never pays
        # onto the same key.
        return (
            guardrail_policy_model("checks")
            if "contentfiltercheck" in normalized
            else ""
        )
    for fragment, policy in _GUARDRAIL_POLICY_SLUGS:
        if fragment in normalized:
            return guardrail_policy_model(policy)
    return ""


def _bedrock_synthetic_model(usagetype: str) -> str:
    """Build the synthetic model a model-less Bedrock row bills against.

    Args:
        usagetype: The usagetype attribute.

    Returns:
        The synthetic model string, or "" when the row names no model-less
        operation this app bills for.
    """
    if _WEB_SEARCH_USAGETYPE_FRAGMENT in _normalize_usagetype(usagetype):
        return WEB_SEARCH_MODEL
    return _guardrail_model(usagetype)


def _synthesize_service_model_key(
    our_service: Service, attrs: Mapping[str, Any]
) -> str:
    """Build synthetic model identifier matching what usage.record_*_usage() uses.

    Polly/Translate/Transcribe/Comprehend have no `model` attribute; matching
    them requires reconstructing the exact synthetic model string from
    `engine` (Polly) or `operation` (Translate/Transcribe/Comprehend). Bedrock
    Guardrails rows have none either -- see :func:`_guardrail_model` -- and
    neither does the built-in web search, whose one flat per-query rate applies
    to every model that can call it.

    Args:
        our_service: The Service this price-list entry belongs to.
        attrs: The price-list item's product.attributes.

    Returns:
        The synthetic model string (e.g., "amazon.polly-neural"), or ""
        when this row doesn't correspond to an operation this app bills for.
    """
    match our_service:
        case Service.BEDROCK:
            return _bedrock_synthetic_model(attrs.get("usagetype", ""))
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
            return {
                "DetectDominantLanguage": "amazon.comprehend-language-detection",
                "DetectToxicContent": "amazon.comprehend-toxicity",
            }.get(attrs.get("operation", ""), "")
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
        item = from_json(price_list_str)
    except ValueError:
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


#: 1-hour cache-write TTL marker in a usagetype: "-1h-", "-1-hour" or "-1hour-".
_CACHE_WRITE_1H_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[-_])1[-_]?h(?:our)?(?:[-_]|$)", re.IGNORECASE
)


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
        our_service == Service.BEDROCK
        and _is_guardrail_usagetype(usagetype)
        and not _guardrail_model(usagetype)
    ):
        # Guardrails rows are keyed by policy rather than model, so an
        # unmapped one would mint a model key of its own that nothing ever
        # prices against. A check the gateway never invokes is expected; a
        # policy AWS added since is not, and bills at nothing until mapped.
        if "check" not in _normalize_usagetype(usagetype):
            diagnostics.append(
                f"Unmodeled Bedrock Guardrails policy usagetype {usagetype!r}: "
                "its usage is billed at no cost until it is mapped."
            )
        return

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
        and _CACHE_WRITE_1H_PATTERN.search(usagetype)
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
    service_regions: tuple[str | None, ...] = (
        SETTINGS.aws_polly_region,
        SETTINGS.aws_transcribe_region,
        SETTINGS.aws_translate_region,
        SETTINGS.aws_comprehend_region,
    )
    return (
        set(SETTINGS.aws_bedrock_regions)
        # Mantle can be served from regions the Converse endpoint isn't
        # configured for; its usage is recorded under the serving region.
        | set(SETTINGS.aws_bedrock_mantle_regions)
        | set(_FALLBACK_ANCHOR_REGIONS)
        | {r for r in service_regions if r}
        | ({AWS_REGION} if AWS_REGION else set())
    ) or {"us-east-1"}


def _apply_default_prices(index: dict[PriceKey, Price]) -> None:
    """Backfill registered built-in default prices into *index*, in place.

    Model-level guard: a model with any published row keeps its published
    prices only, so pricing-page defaults never mix with real rows.

    Args:
        index: Price index to update.
    """
    published_models = {
        key.model
        for key in index
        if key.service in (Service.BEDROCK, Service.BEDROCK_MANTLE)
    }
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
    partition to resolve the currency (configured regions can span partitions,
    each with its own). Always applied at the standard tier -- there's no
    override mechanism for flex/priority/batch pricing -- and under both
    ``Service.BEDROCK`` and ``Service.BEDROCK_MANTLE``, so overrides also win
    for Mantle-routed requests.

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
                for service in (Service.BEDROCK, Service.BEDROCK_MANTLE):
                    index[
                        PriceKey(
                            service, normalized_model, region, dimension, "standard"
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


async def _fetch_or_capture(
    semaphore: asyncio.Semaphore,
    client: PricingClient,
    service_code: str,
    region: str,
    diagnostics: list[str],
) -> tuple[dict[PriceKey, Price], dict[PriceKey, str]] | BotoCoreError | ClientError:
    """Fetch one (service_code, region) pair, bounded, capturing AWS errors instead of raising.

    Args:
        semaphore: Concurrency limiter shared across the whole load attempt.
        client: An aiobotocore pricing client.
        service_code: A ``_SERVICE_CODE_TO_SERVICE`` key (e.g., "AmazonBedrock").
        region: The AWS region to filter by.
        diagnostics: Intra-fetch collision descriptions, appended to in place.

    Returns:
        The fetch's (results, claims) pair, or the caught exception -- letting
        the caller keep every sibling fetch's outcome instead of cancelling
        them all on one failure.
    """
    async with semaphore:
        try:
            return await _fetch_service_pricing(
                client, service_code, region, diagnostics
            )
        except (BotoCoreError, ClientError) as exception:
            return exception


async def _load_price_catalog(diagnostics: list[str]) -> None:
    """Fetch pricing for all configured regions/services and atomically swap the index.

    Concurrency is bounded by :data:`_MAX_CONCURRENT_FETCHES`. A fetch
    failing (e.g. throttling) doesn't cancel or discard its siblings: every
    successfully fetched (region, service_code) pair is merged and published
    immediately, so a partial catalog is usable right away. Failed pairs are
    recorded on ``_state.pending_fetch_specs`` and retried -- carrying the
    accumulated successes forward -- the next time this function is called,
    typically by :func:`_load_price_catalog_with_retry`'s backoff loop.

    Args:
        diagnostics: Collision and invalid-override descriptions for this
            load, appended to in place; the caller surfaces them as one
            warning on its own operation-level log event.
    """
    regions = _catalog_regions()

    if (endpoint := pricing_endpoint_region()) is None:
        diagnostics.append(
            "The AWS Price List API has no endpoint in this partition;"
            " only cost_price_overrides prices apply"
        )
        # Overrides are local (no API): keep them as the sole price source.
        override_index: dict[PriceKey, Price] = {}
        _apply_price_overrides(override_index, regions, diagnostics)
        _state.price_index = override_index
        _state.pending_fetch_specs = None
        _state.catalog_complete = True
        return

    if _state.pending_fetch_specs is not None:
        # Resume a previous partial failure: only the fetches that failed,
        # carrying forward what already succeeded.
        fetch_specs = _state.pending_fetch_specs
        new_index = dict(_state.pending_index)
        claims = dict(_state.pending_claims)
    else:
        fetch_specs = [
            (region, service_code)
            for region in sorted(regions)
            for service_code in _SERVICE_CODE_TO_SERVICE
        ]
        new_index = {}
        claims = {}

    # type-ignore: the RegionName stub Literal lags EUSC/China (works live).
    client = get_client("pricing", endpoint)  # type: ignore[arg-type]
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)
    outcomes = await asyncio.gather(
        *(
            _fetch_or_capture(semaphore, client, service_code, region, diagnostics)
            for region, service_code in fetch_specs
        )
    )

    # Merge in fetch_specs order (not completion order) via _store_price with
    # one shared claims dict, so cross-fetch PriceKey collisions (e.g. the
    # three Bedrock service codes) warn and resolve deterministically. Claims
    # are tagged with the service code so equal usagetype text across fetches
    # isn't mistaken for a repeated price band.
    failed_specs: list[tuple[str, str]] = []
    for (region, service_code), outcome in zip(fetch_specs, outcomes, strict=True):
        if isinstance(outcome, BotoCoreError | ClientError):
            failed_specs.append((region, service_code))
            continue
        fetch_index, fetch_claims = outcome
        for key, price in fetch_index.items():
            _store_price(
                new_index,
                claims,
                key,
                price,
                f"{service_code}:{fetch_claims[key]}",
                diagnostics,
            )

    # Backfill/override on a copy: pending_index/claims must stay the raw
    # fetch results only, so a later retry's _store_price collision check
    # isn't confused by keys these steps added without a claim.
    published_index = dict(new_index)
    _apply_default_prices(published_index)
    _apply_regional_fallback(published_index, regions)
    _apply_price_overrides(published_index, regions, diagnostics)
    _state.price_index = published_index

    _state.catalog_complete = not failed_specs
    if failed_specs:
        diagnostics.append(
            f"Price catalog load: {len(failed_specs)}/{len(fetch_specs)} fetch(es) "
            "failed; a partial catalog was published and only those will be retried"
        )
        _state.pending_fetch_specs = failed_specs
        _state.pending_index = new_index
        _state.pending_claims = claims
    else:
        _state.pending_fetch_specs = None
        _state.pending_index = {}
        _state.pending_claims = {}


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


def price_catalog_ready() -> bool:
    """Whether the price catalog has completed its initial load.

    Returns:
        True when the in-memory price index holds at least one entry.
    """
    return bool(_state.price_index)


def available_currencies() -> frozenset[str]:
    """Return the ISO currency codes present in the loaded price catalog.

    Returns:
        Currencies of the published prices (e.g. USD, or EUR in partitions
        priced in euros).
    """
    return frozenset(price.currency for price in _state.price_index.values())


def batch_priced_models() -> frozenset[str] | None:
    """Return the price-catalog keys AWS publishes a batch-tier rate for.

    AWS publishes a batch price dimension only for models that can run batch
    inference, so this is the catalogue's best-effort signal of batch support.
    Cached until the price index is swapped; the result's identity is stable
    while the catalog is, which lets callers skip a re-derivation.

    Returns:
        Model keys (as resolved by :func:`resolve_model_key`) holding at least
        one ``batch``-tier price row, or None while the catalog is unloaded --
        which is "unknown", not "no model supports batch".
    """
    index = _state.price_index
    if not index:
        return None
    if _state.batch_priced is None or _state.batch_priced[0] is not index:
        _state.batch_priced = (
            index,
            frozenset(key.model for key in index if key.tier == "batch"),
        )
    return _state.batch_priced[1]


def _rows_by_model() -> dict[str, list[tuple[PriceKey, Price]]]:
    """Group the current price index by model, cached until the index is swapped.

    Returns:
        Mapping of model key to its (key, price) pairs.
    """
    index = _state.price_index
    if _state.rows_by_model is None or _state.rows_by_model[0] is not index:
        grouped: dict[str, list[tuple[PriceKey, Price]]] = {}
        for key, price in index.items():
            grouped.setdefault(key.model, []).append((key, price))
        _state.rows_by_model = (index, grouped)
    return _state.rows_by_model[1]


def _dedupe_service_rows(
    rows: list[tuple[PriceKey, Price]], preferred_service: Service
) -> list[tuple[PriceKey, Price]]:
    """Collapse rows identical apart from service, keeping *preferred_service*.

    Default and override prices are registered under both Bedrock services;
    without this, dual-service models would list every such row twice.

    Args:
        rows: (key, price) pairs to reduce.
        preferred_service: The service whose row to keep when duplicated.

    Returns:
        The reduced (key, price) pairs.
    """
    groups: dict[PriceKey, tuple[PriceKey, Price]] = {}
    for key, price in rows:
        canonical = replace(key, service=Service.BEDROCK)
        current = groups.get(canonical)
        if current is None or (
            key.service is preferred_service
            and current[0].service is not preferred_service
        ):
            groups[canonical] = (key, price)
    return list(groups.values())


def model_prices(
    model_id: str,
    *,
    region: str | None = None,
    tier: str | None = None,
    dimensions: AbstractSet[Dimension] | None = None,
    currency: str | None = None,
    routing: Routing | None = None,
    context: ContextLength | None = None,
    variants: bool = True,
    preferred_service: Service = Service.BEDROCK,
) -> list[tuple[PriceKey, Price]]:
    """List the indexed price rows for one model, filtered and sorted.

    A pure in-memory read of the current price catalog (no AWS call), for
    the pricing API. Filters combine with AND; None means "no filter".
    Rows identical apart from their Bedrock service (default/override prices
    are registered under both) are collapsed to the *preferred_service* row.

    Args:
        model_id: The canonical model ID, resolved via
            :func:`resolve_model_key` (aliases are not resolved here).
        region: Only rows for this AWS region.
        tier: Only rows for this service tier.
        dimensions: Only rows for these billed dimensions.
        currency: Only rows in this currency.
        routing: Only rows for this serving profile.
        context: Only rows for this context-length bucket.
        variants: When False, keep only base rows (standard tier, no cache
            TTL, routing, or long-context variant); spec rows are kept
            because media buckets are distinct products, not variants.
        preferred_service: The Bedrock service whose row to keep when a
            price is registered under both (see :func:`_dedupe_service_rows`).

    Returns:
        Matching (key, price) pairs, sorted by region then remaining axes.
    """
    model_key = resolve_model_key(model_id)
    rows = [
        (key, price)
        for key, price in _rows_by_model().get(model_key, ())
        if (region is None or key.region == region)
        and (tier is None or key.tier == tier)
        and (dimensions is None or key.dimension in dimensions)
        and (currency is None or price.currency == currency)
        and (routing is None or key.routing == routing)
        and (context is None or key.context == context)
        and (
            variants
            or (
                key.tier == "standard"
                and not key.cache_ttl
                and not key.routing
                and not key.context
            )
        )
    ]
    rows = _dedupe_service_rows(rows, preferred_service)
    rows.sort(key=_row_sort_key)
    return rows


def _row_sort_key(row: tuple[PriceKey, Price]) -> tuple[str, ...]:
    """Sort key for price rows: region first, then the remaining axes.

    Args:
        row: One (key, price) pair.

    Returns:
        The sortable axis tuple.
    """
    key = row[0]
    return (
        key.region,
        key.dimension,
        key.tier,
        key.context,
        key.routing,
        key.cache_ttl,
        key.spec,
    )


def _select_axis(
    rows: list[tuple[PriceKey, Price]], axis: str, preferred: str, fallback: str
) -> list[tuple[PriceKey, Price]]:
    """Keep one row per otherwise-identical axes: *preferred*, else *fallback*.

    Args:
        rows: (key, price) pairs to reduce.
        axis: The :class:`PriceKey` field to select on.
        preferred: The axis value to keep when published.
        fallback: The axis value to fall back to otherwise.

    Returns:
        The reduced (key, price) pairs.
    """
    if preferred == fallback:
        return [row for row in rows if getattr(row[0], axis) == fallback]
    normalized_axis: dict[str, Any] = {axis: fallback}
    groups: dict[PriceKey, dict[str, tuple[PriceKey, Price]]] = {}
    for key, price in rows:
        group = groups.setdefault(replace(key, **normalized_axis), {})
        group[getattr(key, axis)] = (key, price)
    return [
        group[preferred if preferred in group else fallback]
        for group in groups.values()
        if preferred in group or fallback in group
    ]


def _scale_tier_fallback(
    rows: list[tuple[PriceKey, Price]], tier: str
) -> list[tuple[PriceKey, Price]]:
    """Scale standard-tier rows kept as *tier*'s fallback by the AWS tier ratio.

    Args:
        rows: (key, price) pairs already reduced on the tier axis.
        tier: The effective service tier the rows were selected for.

    Returns:
        The (key, price) pairs, with every fallback row repriced at what
        :func:`resolve_price` bills for *tier*.
    """
    ratio = _TIER_PRICE_RATIO.get(tier.lower(), _ONE)
    if ratio == _ONE:
        return rows
    return [
        (key, Price(price.amount * ratio, price.currency))
        if key.tier == "standard" and key.dimension in _TIER_SCALED_DIMENSIONS
        else (key, price)
        for key, price in rows
    ]


def select_effective_rows(
    rows: list[tuple[PriceKey, Price]],
    *,
    regions: AbstractSet[str] | None = None,
    tier: str | None = None,
    routing: Routing | None = None,
) -> list[tuple[PriceKey, Price]]:
    """Reduce full price rows to a deployment's effective configuration.

    Mirrors the :func:`resolve_price` fallbacks: when no distinct rate is
    published for *tier* or *routing*, the standard-tier or plain row is kept
    instead, exactly like cost tracking bills -- a standard-tier token row
    kept for a discounted or premium tier is repriced at that tier's rate.

    Args:
        rows: (key, price) pairs from :func:`model_prices`.
        regions: Only rows for these AWS regions, or None for all.
        tier: The effective service tier, or None to keep every tier.
        routing: The effective serving profile, or None to keep every profile.

    Returns:
        The reduced (key, price) pairs, sorted like :func:`model_prices`.
    """
    if regions is not None:
        rows = [row for row in rows if row[0].region in regions]
    if tier is not None:
        rows = _scale_tier_fallback(_select_axis(rows, "tier", tier, "standard"), tier)
    if routing is not None:
        rows = _select_axis(rows, "routing", routing, "")
    return sorted(rows, key=_row_sort_key)


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


#: How long a model a completed reload still couldn't price is exempted from retriggering one.
_UNPRICED_MODEL_COOLDOWN_NS: Final[int] = 15 * 60 * 1_000_000_000


async def refresh_price_catalog_for_new_models(model_ids: Iterable[str]) -> None:
    """Reload the price catalog immediately if any of *model_ids* isn't priced yet.

    Blocking and awaited: called by ``initialize_bedrock_models()`` when its
    lazy on-demand refresh discovers unregistered Bedrock models, so the
    catalog self-heals for newly released models without a polling loop.
    Diagnostics from the reload are discarded. A model a completed reload
    still couldn't price is exempted from retriggering one for
    :data:`_UNPRICED_MODEL_COOLDOWN_NS`, so it doesn't force a reload on
    every request.

    Args:
        model_ids: Bedrock model IDs to check. No-op when cost tracking is
            disabled, or every given model is on cooldown or already priced.
    """
    if not SETTINGS.cost_tracking:
        return
    now = perf_counter_ns()
    due_ids = [
        model_id
        for model_id in model_ids
        if _state.unpriced_cooldown.get(model_id, 0) <= now
    ]
    if not due_ids or _all_models_priced(due_ids):
        return

    async with _state.refresh_lock:
        # Re-check under the lock: a concurrent caller may have already
        # refreshed the catalog while this one was waiting for it.
        if _all_models_priced(due_ids):
            return
        await _load_price_catalog([])
        for model_id in due_ids:
            if _all_models_priced((model_id,)):
                _state.unpriced_cooldown.pop(model_id, None)
            else:
                _state.unpriced_cooldown[model_id] = now + _UNPRICED_MODEL_COOLDOWN_NS


def start_price_catalog() -> None:
    """Start loading the price catalog in a background task at startup.

    Returns immediately: server readiness never waits on the AWS Price List
    API. Requests served before the load completes record usage without a
    cost. Each load attempt's outcome (diagnostics, or the AWS error being
    retried with backoff) is written as a ``background`` log event named
    ``price_catalog_load``. Stop the task via :func:`stop_price_catalog`.

    There is no periodic refresh afterward: the catalog is kept current on
    demand by :func:`refresh_price_catalog_for_new_models`.

    Idempotent: a call while a load task already exists is a no-op, so a
    duplicate startup never orphans a running task.
    """
    if not SETTINGS.cost_tracking or _state.load_task is not None:
        return
    # The task outlives the caller's span/request context: run it in a fresh one.
    _state.load_task = asyncio.create_task(
        _load_price_catalog_with_retry(), context=Context()
    )


async def stop_price_catalog() -> None:
    """Cancel and await the background catalog load, if still running.

    Never raises: any exception other than cancellation left in the task is
    logged and swallowed, so shutdown always completes cleanly.
    """
    if (task := _state.load_task) is None:
        return
    _state.load_task = None
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as exception:  # noqa: BLE001 -- logged and swallowed by design
        _log_price_catalog_event(
            "error",
            [
                (
                    "Price catalog load task raised during shutdown "
                    f"({type(exception).__name__}: {exception})"
                )
            ],
            None,
        )


async def _load_price_catalog_with_retry() -> None:
    """Load the catalog, retrying any failure with capped exponential backoff.

    An unexpected (non-AWS) exception is logged at error level and retried on
    the same schedule, so a transient anomaly self-heals instead of disabling
    cost tracking for the process lifetime. A partially failed load keeps
    retrying too, each ``_load_price_catalog`` call resuming from
    ``_state.pending_fetch_specs``. The loop exits without another load when a
    concurrent on-demand refresh already completed the catalog.
    """
    delay = _LOAD_RETRY_INITIAL_SECONDS
    while True:
        diagnostics: list[str] = []
        start = perf_counter_ns()
        try:
            # Serialized with refresh_price_catalog_for_new_models against the
            # low-quota Pricing API; _load_price_catalog never takes this lock
            # itself, so no deadlock risk from nesting.
            async with _state.refresh_lock:
                if _state.catalog_complete:
                    _log_price_catalog_event(
                        "info",
                        [
                            (
                                "Price catalog load already completed by a "
                                "concurrent on-demand refresh"
                            )
                        ],
                        start,
                    )
                    return
                await _load_price_catalog(diagnostics)
        except (BotoCoreError, ClientError) as exception:
            diagnostics.append(
                f"Price catalog load failed ({type(exception).__name__}: "
                f"{exception}); retrying in {delay}s"
            )
            _log_price_catalog_event("warning", diagnostics, start)
        except Exception as exception:  # noqa: BLE001 -- logged and retried by design
            diagnostics.append(
                f"Price catalog load failed on unexpected error "
                f"({type(exception).__name__}: {exception}); retrying in {delay}s"
            )
            _log_price_catalog_event("error", diagnostics, start)
        else:
            if not _state.pending_fetch_specs:
                _log_price_catalog_event(
                    "warning" if diagnostics else "info", diagnostics, start
                )
                return
            diagnostics.append(f"Price catalog load: retrying in {delay}s")
            _log_price_catalog_event("warning", diagnostics, start)
        await asyncio.sleep(delay)
        delay = min(delay * 2, _LOAD_RETRY_MAX_SECONDS)


def _log_price_catalog_event(
    level: LogLevel, diagnostics: list[str], start_ns: int | None
) -> None:
    """Write a ``background`` log event for one price-catalog load attempt.

    Args:
        level: Event severity.
        diagnostics: Load diagnostics, recorded as the event's error detail.
        start_ns: ``perf_counter_ns()`` timestamp when the attempt started,
            or None to omit ``execution_time_ms`` (no load attempt timed).
    """
    # Imported here: stdapi.monitoring transitively imports this module.
    from stdapi import server  # noqa: PLC0415
    from stdapi.metering import SERVER_FULL_VERSION  # noqa: PLC0415
    from stdapi.monitoring import EventLog, write_log_event  # noqa: PLC0415

    event = EventLog(
        type="background",
        level=level,
        date=SETTINGS.now(),
        server_id=server.SERVER_NAME,
        server_version=SERVER_FULL_VERSION,
        event="price_catalog_load",
    )
    if start_ns is not None:
        event["execution_time_ms"] = (perf_counter_ns() - start_ns) // 1000000
    if diagnostics:
        # list() copy: mypy needs a fresh list[JsonValue] (list is invariant).
        event["error_detail"] = list(diagnostics)
    write_log_event(event)
