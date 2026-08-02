"""Real AWS-billed usage tracking per request/model.

Records quantities AWS bills (tokens, characters, media seconds, images,
search/Comprehend units), merged into request logs and optionally emitted
as CloudWatch Embedded Metric Format (EMF).

MAINTENANCE -- this module is model-agnostic (model-specific data belongs
in ``stdapi.models``):
- New billed unit: add one ``Dimension`` member in ``pricing.py`` and one
  ``_DIMENSION_INFO`` row here -- log/EMF/cost hooks in automatically; then
  pass the quantity from the recording call site (e.g. a model class).
- Quantities priced per bucket (TTL, media spec) follow the ``*_by_ttl`` /
  ``*_by_spec`` breakdown pattern on :class:`UsageRecord`.
"""

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from decimal import Decimal
from math import ceil
from time import time_ns
from typing import TYPE_CHECKING, Final, Literal, TypedDict

from stdapi.config import SETTINGS
from stdapi.pricing import (
    CacheTtlBucket,
    ContextLength,
    Dimension,
    Price,
    Routing,
    Service,
    price_catalog_ready,
    resolve_price,
)
from stdapi.utils import stdout_write

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from pydantic import JsonValue
    from types_aiobotocore_bedrock_runtime.literals import ServiceTierTypeType


@dataclass(frozen=True, slots=True)
class _DimensionInfo:
    """Per-dimension metadata: public log key + EMF metric name/unit."""

    log_key: str
    emf_unit: str
    #: log_key precomputed in PascalCase (e.g. "input_tokens" -> "InputTokens").
    emf_name: str = field(init=False)

    def __post_init__(self) -> None:
        """Precompute the PascalCase EMF metric name from log_key."""
        object.__setattr__(
            self,
            "emf_name",
            "".join(word.capitalize() for word in self.log_key.split("_")),
        )


#: Billable dimension registry: public log key and EMF metric unit.
_DIMENSION_INFO: Final[dict[Dimension, _DimensionInfo]] = {
    Dimension.INPUT_TOKENS: _DimensionInfo("input_tokens", "Count"),
    Dimension.OUTPUT_TOKENS: _DimensionInfo("output_tokens", "Count"),
    Dimension.CACHE_READ_TOKENS: _DimensionInfo("cached_tokens", "Count"),
    Dimension.CACHE_WRITE_TOKENS: _DimensionInfo("cache_write_tokens", "Count"),
    Dimension.OUTPUT_IMAGES: _DimensionInfo("output_images", "Count"),
    Dimension.INPUT_SECONDS: _DimensionInfo("input_seconds", "Seconds"),
    Dimension.INPUT_CHARACTERS: _DimensionInfo("input_characters", "Count"),
    Dimension.COMPREHEND_UNITS: _DimensionInfo("comprehend_units", "Count"),
    Dimension.GROUNDING_REQUESTS: _DimensionInfo("grounding_requests", "Count"),
    Dimension.SEARCH_UNITS: _DimensionInfo("search_units", "Count"),
    Dimension.INPUT_IMAGES: _DimensionInfo("input_images", "Count"),
    Dimension.OUTPUT_SECONDS: _DimensionInfo("output_seconds", "Seconds"),
    Dimension.TEXT_UNITS: _DimensionInfo("text_units", "Count"),
}

#: Per-call prompt-token threshold (input + cache read/write) for long-context billing.
_LONG_CONTEXT_THRESHOLD: Final[int] = 200_000


@dataclass(frozen=True, slots=True)
class UsageKey:
    """Composite key identifying one aggregated usage record within a request."""

    service: Service
    model: str
    operation: str
    region: str
    tier: str
    routing: Routing = ""
    context: ContextLength = ""


@dataclass(slots=True)
class UsageRecord:
    """One billed-usage record, aggregated across calls within a request."""

    service: Service
    model: str
    operation: str
    region: str = ""
    tier: str = "standard"  # Service tier (standard, flex, priority, batch)
    routing: Routing = ""  # See resolve_price's `routing`.
    context: ContextLength = ""  # See resolve_price's `context`.
    quantities: dict[Dimension, int] = field(default_factory=dict)
    # Informational only (Converse API); not billed, no Dimension
    total_tokens: int = 0
    # By cache TTL bucket ("5m"/"1h"): AWS charges each TTL differently.
    cache_write_tokens_by_ttl: dict[CacheTtlBucket, int] = field(default_factory=dict)
    # By "<resolution>:<quality>" image spec (falls back to flat per-image).
    output_images_by_spec: dict[str, int] = field(default_factory=dict)
    # By resolution spec bucket ("hd"); falls back to flat per-second.
    output_seconds_by_spec: dict[str, int] = field(default_factory=dict)
    # Input-media breakdowns by spec bucket ("document", "audio", "video", ...).
    input_images_by_spec: dict[str, int] = field(default_factory=dict)
    input_seconds_by_spec: dict[str, int] = field(default_factory=dict)
    cost: Decimal = Decimal(0)  # Computed in compute_costs()
    currency: str = ""  # Computed in compute_costs()
    # Populated instead of cost/currency only when dimensions span multiple currencies
    costs: dict[str, Decimal] = field(default_factory=dict)


class UsageLogEntry(TypedDict, total=False):
    """One entry in the request log ``usage`` list (zero fields omitted)."""

    service: str
    model: str
    operation: str
    region: str
    tier: str  # Service tier: standard, flex, priority, batch
    routing: Literal["global", "latency"]  # Only present for a non-plain profile
    context: Literal["long"]  # Only present when the prompt exceeded 200K tokens
    cost: str  # Exact plain-decimal text -- see format_cost
    currency: str
    costs: dict[str, str]  # Populated when dimensions spanned more than one currency
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    cache_write_tokens_by_ttl: dict[CacheTtlBucket, int]
    output_images: int
    output_images_by_spec: dict[str, int]
    output_seconds: int
    output_seconds_by_spec: dict[str, int]
    input_images: int
    input_images_by_spec: dict[str, int]
    input_seconds: int
    input_seconds_by_spec: dict[str, int]
    input_characters: int
    comprehend_units: int
    grounding_requests: int
    search_units: int
    text_units: int


#: Per-request usage records, keyed by every UsageKey axis.
USAGE: ContextVar[dict[UsageKey, UsageRecord]] = ContextVar("usage")

#: Current request operation (route path), set once per request by ``monitoring``.
OPERATION: ContextVar[str] = ContextVar("operation", default="")


@dataclass(slots=True)
class ModelInvocationState:
    """Shared, best-effort default region/tier/routing for a model within the current request.

    Mutated in place by models/__init__.py before each call, so it holds
    the *last-written* value across every call to the same model within the
    request -- not an isolated per-call value. Every current caller that
    can invoke the same model concurrently with differing region/tier/
    routing (Converse, InvokeModel) always passes those values explicitly
    to :func:`record_bedrock_usage`, bypassing this shared state and
    staying race-free; only callers without differentiated per-call values
    (rerank, embeddings, video) rely on the fallback below. A future call
    site that invokes a model concurrently with differing values without
    passing them explicitly would risk misattributing usage to the wrong
    region/tier/routing.
    """

    region: str | None = None
    service_tier: ServiceTierTypeType = "default"
    routing: Routing | None = None


#: Per-request default invocation state keyed by model ID, set by models/__init__.py.
MODEL_STATE: ContextVar[dict[str, ModelInvocationState]] = ContextVar("model_state")

#: Current image spec ("<resolution>:<quality>"), set by image jobs before invoking.
IMAGE_SPEC: ContextVar[str] = ContextVar("image_spec", default="")


def init_model_state() -> Token[dict[str, ModelInvocationState]]:
    """Install a fresh per-request model-state dict; return a token to restore the previous one."""
    return MODEL_STATE.set({})


def get_model_state(model: str) -> ModelInvocationState:
    """Return this request's invocation state for *model*, creating it on first access.

    Args:
        model: Bedrock model ID.

    Returns:
        The model's mutable invocation state. A throwaway instance when no
        request scope is installed (mirrors ``USAGE`` no-op-without-scope).
    """
    states = MODEL_STATE.get(None)
    if states is None:
        return ModelInvocationState()
    return states.setdefault(model, ModelInvocationState())


def _positive_values[K](mapping: Mapping[K, int] | None) -> dict[K, int]:
    """Filter a mapping to keep only strictly-positive values.

    Args:
        mapping: The mapping to filter, or None.

    Returns:
        A new dict with only the entries whose value is > 0.
    """
    return {key: value for key, value in (mapping or {}).items() if value > 0}


def init_usage() -> Token[dict[UsageKey, UsageRecord]]:
    """Install a fresh per-request usage map; return a token to restore the previous one.

    Request-handling code MUST capture and reset the token, otherwise nested
    in-process calls permanently replace the outer request's USAGE dict.
    """
    return USAGE.set({})


def _record_usage(
    service: Service,
    model: str,
    region: str = "",
    tier: str = "standard",
    *,
    routing: Routing = "",
    context: ContextLength = "",
    quantities: Mapping[Dimension, int] | None = None,
    total_tokens: int = 0,
    cache_write_tokens_by_ttl: Mapping[CacheTtlBucket, int] | None = None,
    output_images_by_spec: Mapping[str, int] | None = None,
    output_seconds_by_spec: Mapping[str, int] | None = None,
    input_images_by_spec: Mapping[str, int] | None = None,
    input_seconds_by_spec: Mapping[str, int] | None = None,
) -> None:
    """Record real AWS-billed usage for ``service``/``model``/``tier``.

    The operation is the current ``OPERATION`` context value. Only positive
    values accumulate; zeros are ignored. Repeated calls with the same key
    sum into one record. No-op when no usage map is installed.

    Args:
        service: The AWS service.
        model: The model ID.
        region: The region, or "" to fall back to this model's ``MODEL_STATE``
            entry for Bedrock.
        tier: The service tier (standard, flex, priority, batch).
        routing: Serving profile ("global", "latency" or "").
        context: Context-length bucket ("long" when this call's prompt
            exceeded the 200K-token threshold).
        quantities: Per-dimension billed quantities.
        total_tokens: Total tokens reported by the Converse API (informational
            only, not billed).
        cache_write_tokens_by_ttl: Cache-write token counts broken down by
            TTL bucket.
        output_images_by_spec: Output image counts broken down by
            resolution/quality spec.
        output_seconds_by_spec: Output media seconds broken down by
            resolution spec bucket.
        input_images_by_spec: Input image counts broken down by spec bucket.
        input_seconds_by_spec: Input media seconds broken down by spec bucket.
    """
    if (records := USAGE.get(None)) is None:
        return
    quantities_to_add = _positive_values(quantities)
    cache_ttl_to_add = _positive_values(cache_write_tokens_by_ttl)
    spec_breakdowns_to_add = tuple(
        _positive_values(breakdown)
        for breakdown in (
            output_images_by_spec,
            output_seconds_by_spec,
            input_images_by_spec,
            input_seconds_by_spec,
        )
    )
    if (
        not quantities_to_add
        and not cache_ttl_to_add
        and not any(spec_breakdowns_to_add)
        and total_tokens <= 0
    ):
        return
    # MODEL_STATE region is Bedrock-only; using it for other services would
    # leak Bedrock's region into nested Polly/Transcribe/Translate calls.
    # Race-free only because concurrent same-model callers pass `region`
    # explicitly (see ModelInvocationState).
    effective_region = region or (
        (get_model_state(model).region or "") if service == Service.BEDROCK else ""
    )
    operation = OPERATION.get("")
    # Include tier/routing/context in the key so calls that differ on any
    # of them don't merge (they're priced differently).
    key = UsageKey(service, model, operation, effective_region, tier, routing, context)
    record = records.get(key)
    if record is None:
        record = records[key] = UsageRecord(
            service, model, operation, effective_region, tier, routing, context
        )
    for dimension, value in quantities_to_add.items():
        record.quantities[dimension] = record.quantities.get(dimension, 0) + value
    if total_tokens > 0:
        record.total_tokens += total_tokens
    _add_cache_ttl_breakdown(record, cache_ttl_to_add, quantities_to_add)
    record_breakdowns = (
        record.output_images_by_spec,
        record.output_seconds_by_spec,
        record.input_images_by_spec,
        record.input_seconds_by_spec,
    )
    for target, breakdown in zip(
        record_breakdowns, spec_breakdowns_to_add, strict=True
    ):
        for spec, count in breakdown.items():
            target[spec] = target.get(spec, 0) + count


def _add_cache_ttl_breakdown(
    record: UsageRecord,
    cache_ttl_to_add: Mapping[CacheTtlBucket, int],
    quantities_to_add: Mapping[Dimension, int],
) -> None:
    """Accumulate a per-TTL cache-write breakdown into *record*, in place.

    Tops the flat quantity up by the breakdown-over-flat deficit so the
    excess still enters ``quantities`` and gets priced.

    Args:
        record: The usage record to update.
        cache_ttl_to_add: This call's positive per-TTL token counts.
        quantities_to_add: This call's positive per-dimension quantities.
    """
    for ttl, tokens in cache_ttl_to_add.items():
        record.cache_write_tokens_by_ttl[ttl] = (
            record.cache_write_tokens_by_ttl.get(ttl, 0) + tokens
        )
    if cache_ttl_to_add:
        deficit = sum(cache_ttl_to_add.values()) - quantities_to_add.get(
            Dimension.CACHE_WRITE_TOKENS, 0
        )
        if deficit > 0:
            record.quantities[Dimension.CACHE_WRITE_TOKENS] = (
                record.quantities.get(Dimension.CACHE_WRITE_TOKENS, 0) + deficit
            )


def _default_region(service: Service) -> str:
    """Get the configured region for a non-Bedrock service.

    Args:
        service: One of POLLY, TRANSCRIBE, TRANSLATE, or COMPREHEND.

    Returns:
        The region string, or empty string if not configured.
    """
    match service:
        case Service.POLLY:
            return SETTINGS.aws_polly_region or ""
        case Service.TRANSCRIBE:
            return SETTINGS.aws_transcribe_region or ""
        case Service.TRANSLATE:
            return SETTINGS.aws_translate_region or ""
        case _:
            return SETTINGS.aws_comprehend_region or ""


def compute_costs() -> list[str]:
    """Compute costs for all usage records using pricing data.

    Looks up unit prices for each non-zero dimension. Cache-write tokens are
    priced per TTL bucket using the breakdown when available (AWS charges
    differently per TTL). Falls back to flat total when no breakdown recorded.

    Omits costs when pricing is unavailable or a record has no resolved
    region. Multi-currency records populate ``costs`` (per-currency
    breakdown); single-currency records populate ``cost``/``currency``.
    While the background price catalog hasn't finished its initial load,
    costs are skipped entirely and no pricing-miss warning is emitted --
    every record served during that startup window would otherwise miss.

    Returns:
        Warning messages for records with an unpriced dimension or that
        resolved to multiple currencies.
    """
    if (records := USAGE.get(None)) is None or not price_catalog_ready():
        return []

    return [
        warning
        for record in records.values()
        if record.region
        and (warning := _apply_record_cost(record, *_compute_record_totals(record)))
    ]


def _reconcile_buckets(
    buckets: list[tuple[int, CacheTtlBucket, str]], total_quantity: int
) -> list[tuple[int, CacheTtlBucket, str]]:
    """Append an undifferentiated bucket for any quantity a breakdown doesn't cover.

    A breakdown can end up covering only part of a dimension's total when
    mixed breakdown-bearing and flat-only calls occur within one request.
    Without this, the remainder would be silently priced as $0.

    Args:
        buckets: Buckets already derived from the breakdown dict.
        total_quantity: The dimension's full accumulated quantity.

    Returns:
        *buckets*, plus one extra undifferentiated (``"", ""``) bucket for
        any shortfall.
    """
    if (covered := sum(qty for qty, _, _ in buckets)) < total_quantity:
        return [*buckets, (total_quantity - covered, "", "")]
    return buckets


def _dimension_price_buckets(
    dimension: Dimension, quantity: int, record: UsageRecord
) -> list[tuple[int, CacheTtlBucket, str]]:
    """Split a dimension's quantity into buckets for pricing.

    Returns multiple buckets when a per-TTL or per-spec breakdown was
    recorded (AWS charges each bucket differently). Otherwise a single flat
    bucket covers the total. Partial breakdowns get an extra undifferentiated
    bucket for the remainder (see :func:`_reconcile_buckets`).

    Args:
        dimension: The dimension being priced.
        quantity: The dimension's total recorded quantity.
        record: The usage record (for its breakdown dicts).

    Returns:
        One or more (bucket_quantity, cache_ttl, spec) tuples.
    """
    if dimension == Dimension.CACHE_WRITE_TOKENS and record.cache_write_tokens_by_ttl:
        buckets = [
            (qty, ttl, "") for ttl, qty in record.cache_write_tokens_by_ttl.items()
        ]
        return _reconcile_buckets(buckets, quantity)
    spec_breakdown = {
        Dimension.OUTPUT_IMAGES: record.output_images_by_spec,
        Dimension.OUTPUT_SECONDS: record.output_seconds_by_spec,
        Dimension.INPUT_IMAGES: record.input_images_by_spec,
        Dimension.INPUT_SECONDS: record.input_seconds_by_spec,
    }.get(dimension)
    if spec_breakdown:
        buckets = [(qty, "", spec) for spec, qty in spec_breakdown.items()]
        return _reconcile_buckets(buckets, quantity)
    return [(quantity, "", "")]


def _compute_record_totals(
    record: UsageRecord,
) -> tuple[dict[str, Decimal], set[Dimension]]:
    """Resolve and sum one record's per-dimension costs, bucketed by currency.

    Args:
        record: The usage record to price.

    Returns:
        Per-currency running totals (see :func:`_add_dimension_cost`), and the
        dimensions for which at least one bucket had no resolvable price.
    """
    totals: dict[str, Decimal] = {}
    unpriced: set[Dimension] = set()
    for dimension, quantity in record.quantities.items():
        if quantity <= 0:
            continue
        for bucket_quantity, ttl, spec in _dimension_price_buckets(
            dimension, quantity, record
        ):
            price = resolve_price(
                record.service,
                record.model,
                record.region,
                dimension,
                record.tier,
                ttl,
                record.routing,
                spec,
                record.context,
            )
            if price is None:
                unpriced.add(dimension)
            _add_dimension_cost(bucket_quantity, price, totals)
    return totals, unpriced


def _apply_record_cost(
    record: UsageRecord, totals: dict[str, Decimal], unpriced: set[Dimension]
) -> str | None:
    """Set a record's cost/currency, or costs when it spans multiple currencies.

    Args:
        record: The usage record to update in place.
        totals: Per-currency totals from :func:`_compute_record_totals`.
        unpriced: Dimensions from :func:`_compute_record_totals` with no
            resolvable price.

    Returns:
        A warning message if some dimensions had no resolvable price or the
        record resolved to multiple currencies, else None.
    """
    warnings = (
        [
            (
                f"No price found for {record.service}/{record.model} in "
                f"{record.region}: {sorted(dimension.value for dimension in unpriced)}"
            )
        ]
        if unpriced
        else []
    )

    if not totals:
        return "; ".join(warnings) or None

    if len(totals) > 1:
        record.costs = {
            currency: amount.quantize(Decimal("0.000001"))
            for currency, amount in totals.items()
            if amount > 0
        }
        warnings.append(
            f"Multiple currencies resolved for {record.service}/{record.model} "
            f"in {record.region}: {sorted(totals)}"
        )
    else:
        currency, amount = next(iter(totals.items()))
        if amount > 0:
            record.cost = amount.quantize(Decimal("0.000001"))
            record.currency = currency

    return "; ".join(warnings) or None


def _add_dimension_cost(
    quantity: int, price: Price | None, totals: dict[str, Decimal]
) -> None:
    """Accumulate one dimension's cost into its currency's running subtotal, in place.

    Args:
        quantity: Billed quantity for this dimension (or cache-TTL bucket).
        price: The resolved price, or None if unavailable.
        totals: Per-currency running totals, updated in place.
    """
    if price is None or quantity <= 0:
        return
    totals[price.currency] = (
        totals.get(price.currency, Decimal(0)) + Decimal(quantity) * price.amount
    )


def record_comprehend_usage(
    text_length: int,
    feature: Literal["language-detection", "toxicity"],
    *,
    region: str = "",
) -> int:
    """Record AWS Comprehend usage.

    Args:
        text_length: Length of the text analyzed.
        feature: Comprehend feature.
        region: Region that served the call; configured default when empty.

    Returns:
        Billed units (100-character units, minimum 3).
    """
    billed_units = max(ceil(text_length / 100), 3)
    _record_usage(
        Service.COMPREHEND,
        f"amazon.comprehend-{feature}",
        region or _default_region(Service.COMPREHEND),
        quantities={Dimension.COMPREHEND_UNITS: billed_units},
    )
    return billed_units


def record_guardrail_usage(
    model: str, *, text_units: int = 0, images: int = 0, region: str = ""
) -> None:
    """Record AWS Bedrock Guardrails (ApplyGuardrail or InvokeGuardrailChecks) usage.

    AWS bills guardrail policies and guardrail checks per text unit (1,000
    characters) for text content and per image for image content.

    Args:
        model: Guardrail model ID reported in the moderation response.
        text_units: Billed text units for text content.
        images: Number of classified input images.
        region: Region hosting the guardrail.
    """
    _record_usage(
        Service.BEDROCK,
        model,
        region,
        quantities={Dimension.TEXT_UNITS: text_units, Dimension.INPUT_IMAGES: images},
    )


def record_polly_usage(
    characters: int,
    engine: Literal["standard", "neural", "long-form", "generative"],
    *,
    region: str = "",
) -> int:
    """Record AWS Polly usage.

    Args:
        characters: Number of characters in the input text.
        engine: Polly engine used for synthesis.
        region: Region that served the call; configured default when empty.

    Returns:
        Billed characters.
    """
    _record_usage(
        Service.POLLY,
        f"amazon.polly-{engine}",
        region or _default_region(Service.POLLY),
        quantities={Dimension.INPUT_CHARACTERS: characters},
    )
    return characters


def record_translate_usage(characters: int, *, region: str = "") -> int:
    """Record AWS Translate usage.

    Args:
        characters: Number of characters in the input text.
        region: Region that served the call; configured default when empty.

    Returns:
        Billed characters.
    """
    _record_usage(
        Service.TRANSLATE,
        "amazon.translate",
        region or _default_region(Service.TRANSLATE),
        quantities={Dimension.INPUT_CHARACTERS: characters},
    )
    return characters


def record_transcribe_usage(audio_duration: float, *, region: str = "") -> int:
    """Record AWS Transcribe usage.

    Args:
        audio_duration: Actual audio duration in seconds.
        region: Region that served the job; configured default when empty.

    Returns:
        Billed seconds (minimum 15).
    """
    billed_seconds = max(ceil(audio_duration), 15)
    _record_usage(
        Service.TRANSCRIBE,
        "amazon.transcribe",
        region or _default_region(Service.TRANSCRIBE),
        quantities={Dimension.INPUT_SECONDS: billed_seconds},
    )
    return billed_seconds


def record_bedrock_usage(
    model: str,
    *,
    service: Service = Service.BEDROCK,
    tier: str | None = None,
    region: str = "",
    routing: Routing | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    cached_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    output_images: int | None = None,
    output_seconds: int | None = None,
    output_seconds_spec: str = "",
    grounding_requests: int | None = None,
    search_units: int | None = None,
    input_images: int | None = None,
    input_seconds: int | None = None,
    media_spec: str = "",
    cache_write_tokens_by_ttl: Mapping[CacheTtlBucket, int] | None = None,
) -> None:
    """Record AWS Bedrock usage.

    Args:
        model: Bedrock model ID.
        service: Serving endpoint (bedrock-runtime or bedrock-mantle).
        tier: Service tier (standard, flex, priority, batch). Defaults to
            this model's shared invocation state (see
            :func:`get_model_state`) -- pass it explicitly for any call that
            may run concurrently with a sibling call to the same model
            using a different tier, or the shared state may be overwritten
            first by the sibling.
        region: Region that served the call. Defaults like *tier*: pass it
            explicitly for concurrent same-model calls, whose shared state
            may be overwritten by a sibling call.
        routing: Serving profile of this call. Defaults like *tier*.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        total_tokens: Total tokens (when returned by Converse API).
        cached_tokens: Number of cached (prompt cache) tokens.
        cache_write_tokens: Number of tokens written to prompt cache.
        output_images: Number of generated images.
        output_seconds: Generated media duration in seconds (e.g. video).
        output_seconds_spec: Resolution spec bucket for output media
            ("hd", ...), when the model's pricing distinguishes it.
        grounding_requests: Number of built-in grounding-tool invocations
            (e.g. Amazon Nova Grounding, billed per request).
        search_units: Number of rerank search units.
        input_images: Number of input images (e.g. multimodal embeddings).
        input_seconds: Input media duration in seconds.
        media_spec: Spec bucket for input media ("document", "audio",
            "video", ...) -- one media item per call, so a single value.
        cache_write_tokens_by_ttl: Token counts grouped by cache TTL.
    """
    state = get_model_state(model)
    image_spec = IMAGE_SPEC.get("") if output_images else ""
    # "default" means no differentiated tier; normalize it (and unset) to "standard".
    # Race-free only because concurrent same-model callers pass tier/routing
    # explicitly (see ModelInvocationState).
    effective_tier = tier or state.service_tier
    # AWS bills the whole call at the long-context rate when the prompt
    # (fresh + cache read/write tokens) exceeds the threshold.
    prompt_tokens = (
        (input_tokens or 0) + (cached_tokens or 0) + (cache_write_tokens or 0)
    )
    _record_usage(
        service,
        model,
        region,
        tier="standard" if effective_tier == "default" else effective_tier,
        routing=routing if routing is not None else (state.routing or ""),
        context="long" if prompt_tokens > _LONG_CONTEXT_THRESHOLD else "",
        quantities={
            Dimension.INPUT_TOKENS: input_tokens or 0,
            Dimension.OUTPUT_TOKENS: output_tokens or 0,
            Dimension.CACHE_READ_TOKENS: cached_tokens or 0,
            Dimension.CACHE_WRITE_TOKENS: cache_write_tokens or 0,
            Dimension.OUTPUT_IMAGES: output_images or 0,
            Dimension.OUTPUT_SECONDS: output_seconds or 0,
            Dimension.GROUNDING_REQUESTS: grounding_requests or 0,
            Dimension.SEARCH_UNITS: search_units or 0,
            Dimension.INPUT_IMAGES: input_images or 0,
            Dimension.INPUT_SECONDS: input_seconds or 0,
        },
        total_tokens=total_tokens or 0,
        cache_write_tokens_by_ttl=cache_write_tokens_by_ttl,
        output_images_by_spec={image_spec: output_images}
        if image_spec and output_images
        else None,
        output_seconds_by_spec={output_seconds_spec: output_seconds}
        if output_seconds_spec and output_seconds
        else None,
        input_images_by_spec={media_spec: input_images}
        if media_spec and input_images
        else None,
        input_seconds_by_spec={media_spec: input_seconds}
        if media_spec and input_seconds
        else None,
    )


def format_cost(amount: Decimal) -> str:
    """Format a billing amount as exact plain-decimal text.

    Args:
        amount: The cost amount.

    Returns:
        Decimal notation with no exponent and no trailing zeros (e.g.
        "0.000015", "12.5") -- float conversion would reintroduce both.
    """
    return format(amount.normalize(), "f")


def total_costs_by_currency(entries: Iterable[UsageLogEntry]) -> dict[str, Decimal]:
    """Sum a list of usage log entries' cost(s), grouped by currency.

    Args:
        entries: Usage log entries, each carrying either cost/currency or costs.

    Returns:
        Total cost per currency, summed in Decimal to keep the rollup exact.
    """
    totals: dict[str, Decimal] = {}
    for entry in entries:
        if (cost := entry.get("cost")) and (currency := entry.get("currency")):
            pairs: Iterable[tuple[str, str]] = ((currency, cost),)
        elif costs := entry.get("costs"):
            pairs = costs.items()
        else:
            continue
        for entry_currency, entry_cost in pairs:
            totals[entry_currency] = totals.get(entry_currency, Decimal(0)) + Decimal(
                entry_cost
            )
    return totals


def _add_cost_fields(entry: UsageLogEntry, record: UsageRecord) -> None:
    """Add cost/currency (or costs, if multi-currency) to a log entry, in place.

    Args:
        entry: The log entry to update in place.
        record: The usage record to read cost/currency from.
    """
    if record.currency and record.cost > 0:
        entry["cost"] = format_cost(record.cost)
        entry["currency"] = record.currency
    elif record.costs:
        entry["costs"] = {c: format_cost(v) for c, v in record.costs.items()}


def _base_log_entry(record: UsageRecord) -> UsageLogEntry:
    """Build a log entry with a record's identity fields (empty ones omitted).

    Args:
        record: The usage record to read identity fields from.

    Returns:
        A new log entry with service/model/operation and any non-empty
        region/tier/routing/context.
    """
    entry: UsageLogEntry = {
        "service": record.service.value,
        "model": record.model,
        "operation": record.operation,
    }
    if record.region:
        entry["region"] = record.region
    if record.tier:
        entry["tier"] = record.tier
    if record.routing:
        entry["routing"] = record.routing
    if record.context:
        entry["context"] = record.context
    return entry


def usage_log_entries() -> list[UsageLogEntry]:
    """Return the recorded usage as log entries, omitting zero/empty fields."""
    entries: list[UsageLogEntry] = []
    for record in (USAGE.get(None) or {}).values():
        entry = _base_log_entry(record)
        _add_cost_fields(entry, record)
        for dimension, info in _DIMENSION_INFO.items():
            if (value := record.quantities.get(dimension, 0)) > 0:
                entry[info.log_key] = value  # type: ignore[literal-required]
        if record.total_tokens > 0:
            entry["total_tokens"] = record.total_tokens
        if record.cache_write_tokens_by_ttl:
            entry["cache_write_tokens_by_ttl"] = dict(record.cache_write_tokens_by_ttl)
        if record.output_images_by_spec:
            entry["output_images_by_spec"] = dict(record.output_images_by_spec)
        if record.output_seconds_by_spec:
            entry["output_seconds_by_spec"] = dict(record.output_seconds_by_spec)
        if record.input_images_by_spec:
            entry["input_images_by_spec"] = dict(record.input_images_by_spec)
        if record.input_seconds_by_spec:
            entry["input_seconds_by_spec"] = dict(record.input_seconds_by_spec)
        entries.append(entry)
    return entries


def emit_usage_metrics() -> None:
    """Write one EMF line per usage record when ``cloudwatch_metrics`` is on.

    Each line carries only non-zero metric fields. ``Model`` is the dimension;
    ``Currency`` is added when a cost is resolved (EMF requires declared
    dimensions to have matching fields). Emitted directly to stdout.

    Multi-currency records emit one line per currency, quantity metrics only
    on the first line to avoid double-counting.
    """
    if not SETTINGS.cloudwatch_metrics:
        return
    if not (records := USAGE.get(None)):
        return
    timestamp = time_ns() // 1_000_000
    for record in records.values():
        quantity_metrics: list[JsonValue] = []
        quantities_payload: dict[str, int] = {}
        for dimension, info in _DIMENSION_INFO.items():
            if (value := record.quantities.get(dimension, 0)) > 0:
                quantity_metrics.append({"Name": info.emf_name, "Unit": info.emf_unit})
                quantities_payload[info.emf_name] = value

        cost_by_currency = record.costs or (
            {record.currency: record.cost}
            if record.currency and record.cost > 0
            else {}
        )
        if not quantity_metrics and not cost_by_currency:
            continue

        base: dict[str, JsonValue] = {
            "Model": record.model,
            "operation": record.operation,
        }

        if not cost_by_currency:
            stdout_write(
                base
                | quantities_payload
                | {
                    "_aws": {
                        "Timestamp": timestamp,
                        "CloudWatchMetrics": [
                            {
                                "Namespace": SETTINGS.cloudwatch_metrics_namespace,
                                "Dimensions": [["Model"]],
                                "Metrics": quantity_metrics,
                            }
                        ],
                    }
                }
            )
            continue

        for index, (currency, amount) in enumerate(cost_by_currency.items()):
            payload: dict[str, JsonValue] = (
                dict(quantities_payload) if index == 0 else {}
            ) | base
            # EMF metric values must be JSON numbers (CloudWatch doubles).
            payload["Cost"] = float(amount)
            payload["Currency"] = currency
            # Separate directives per dimension set: EMF publishes every metric
            # in a directive under every one of its dimension sets, so a single
            # directive spanning both ["Model"] and ["Model", "Currency"] would
            # also publish Cost bare-by-Model -- silently summing currencies.
            directives: list[JsonValue] = [
                {
                    "Namespace": SETTINGS.cloudwatch_metrics_namespace,
                    "Dimensions": [["Model", "Currency"]],
                    "Metrics": [{"Name": "Cost", "Unit": "None"}],
                }
            ]
            if index == 0 and quantity_metrics:
                directives.insert(
                    0,
                    {
                        "Namespace": SETTINGS.cloudwatch_metrics_namespace,
                        "Dimensions": [["Model"]],
                        "Metrics": quantity_metrics,
                    },
                )
            payload["_aws"] = {"Timestamp": timestamp, "CloudWatchMetrics": directives}
            stdout_write(payload)
