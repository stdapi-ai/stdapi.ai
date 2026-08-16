"""Automatic region routing for AWS Bedrock invocations.

Distributes requests across multiple configured regions to handle quota
exhaustion and regional unavailability. Supports ordered, lowest-latency,
and round-robin routing strategies.
"""

from asyncio import gather
from dataclasses import dataclass
from math import ceil, isinf
from secrets import randbelow
from statistics import mean, pstdev
from time import monotonic
from typing import TYPE_CHECKING

from aiohttp import ClientError as AIOHTTPClientError
from aiohttp import ClientSession, ClientTimeout

from stdapi.config import AWS_SESSION, SETTINGS
from stdapi.monitoring import REQUEST, RegionLatenciesStatsKeys, log_error_details
from stdapi.server import HTTP_CLIENT_HEADERS

if TYPE_CHECKING:
    from fastapi import Request
    from types_aiobotocore_bedrock.literals import RegionName

#: Resolves each region's partition-correct hostname (aws / aws-cn / aws-eusc / ...).
_ENDPOINT_RESOLVER = AWS_SESSION.get_component("endpoint_resolver")


#: AWS error codes that trigger quota-based exponential backoff.
_QUOTA_ERROR_CODES: frozenset[str] = frozenset(
    {"ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException"}
)

#: AWS error codes that trigger fixed unavailability backoff.
_UNAVAILABLE_ERROR_CODES: frozenset[str] = frozenset(
    {"ServiceUnavailableException", "InternalServerException", "ModelNotReadyException"}
)

#: Union of quota and unavailability error codes; used by callers to decide whether to retry.
ROUTING_RETRYABLE_CODES: frozenset[str] = _QUOTA_ERROR_CODES | _UNAVAILABLE_ERROR_CODES

#: Hard ceiling on quota backoff in seconds (1 hour).
_MAX_QUOTA_BACKOFF: int = SETTINGS.aws_bedrock_region_routing_max_quota_backoff_seconds

#: Stale-counter reset threshold: if the last quota error was older than this, reset the counter.
_QUOTA_STALE_THRESHOLD: int = (
    _MAX_QUOTA_BACKOFF * SETTINGS.aws_bedrock_region_routing_quota_stale_factor
)

#: Per-region measured latency in milliseconds, shared across all routing strategy instances.
_REGION_LATENCIES: dict[RegionName, float] = {}

#: Bedrock regions in priority order
ORDERED_BEDROCK_REGIONS: list[RegionName] = (
    SETTINGS.aws_bedrock_regions.copy()  # typing: ignore[assignment]
)

#: Request-scope state key carrying the quota backoff to surface as ``retry-after``.
_QUOTA_BACKOFF_STATE_KEY: str = "stdapi_quota_backoff_seconds"


def _publish_quota_backoff(seconds: float) -> None:
    """Record a quota backoff on the current request scope.

    The ASGI scope is shared across the whole middleware stack, so the value
    survives the task boundaries that isolate ``ContextVar`` writes made while
    serving the request. The smallest backoff seen wins: the request can be
    retried as soon as the first blocked region leaves its cooldown.

    Args:
        seconds: Backoff just applied to the region that returned a quota error.
    """
    if (request := REQUEST.get(None)) is None:
        return
    state = request.scope.setdefault("state", {})
    if seconds < state.get(_QUOTA_BACKOFF_STATE_KEY, float("inf")):
        state[_QUOTA_BACKOFF_STATE_KEY] = seconds


def quota_retry_after(request: Request) -> int | None:
    """Return the ``retry-after`` delay recorded while serving *request*.

    Args:
        request: Request whose scope may carry a recorded quota backoff.

    Returns:
        The router's quota backoff rounded up to whole seconds, or None when no
        region was blocked for a quota error during the request (no delay known).
    """
    seconds = (request.scope.get("state") or {}).get(_QUOTA_BACKOFF_STATE_KEY)
    return ceil(seconds) if seconds is not None else None


@dataclass(slots=True)
class RegionState:
    """Health state of a single region, tracked per model for routing decisions.

    Attributes:
        region: AWS region identifier this state belongs to.
        quota_blocked_until: Monotonic timestamp until which quota errors block this region.
        unavailable_until: Monotonic timestamp until which unavailability blocks this region.
        consecutive_quota_errors: Count of consecutive quota errors since the region was last usable.
        last_quota_error_time: Monotonic timestamp of the most recent quota error (0.0 if none).
    """

    region: str
    quota_blocked_until: float = 0.0
    unavailable_until: float = 0.0
    consecutive_quota_errors: int = 0
    last_quota_error_time: float = 0.0

    @property
    def is_usable(self) -> bool:
        """Whether the region is currently usable.

        Returns:
            True when both quota and unavailability cooldowns have expired.
        """
        now = monotonic()
        return self.quota_blocked_until <= now and self.unavailable_until <= now


class _ModelRegionIndex:
    """Nested index of RegionState objects keyed by (model_id, region).

    A fresh RegionState is created transparently on first access.
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, RegionState]] = {}

    def get(self, model_id: str, region: str) -> RegionState:
        """Returns the existing RegionState for (model_id, region), creating one if absent.

        Args:
            model_id: Bedrock model identifier.
            region: AWS region identifier.

        Returns:
            The RegionState for the given model and region.
        """
        inner = self._data.setdefault(model_id, {})
        if region not in inner:
            inner[region] = RegionState(region=region)
        return inner[region]


class RegionRouter:
    """Selects the optimal region for Bedrock invocations.

    Maintains per-model, per-region health state and delegates region ordering
    to a pluggable RoutingOrder callable.
    """

    __slots__ = ("_index", "_order", "_round_robin_counters")

    def __init__(self) -> None:
        """Initialises the router with a region ordering callable."""
        self._index = _ModelRegionIndex()
        if SETTINGS.aws_bedrock_region_routing == "round_robin":
            self._round_robin_counters: dict[str, int] = {}
            self._order = self._round_robin_order
        else:
            self._order = self._identity_order

    def ordered_regions(
        self, model_id: str, available_regions: list[RegionName]
    ) -> list[RegionName]:
        """Returns regions in priority order for failover iteration.

        Usable regions come first (ordered by strategy); blocked regions are
        appended as a last-resort fallback so callers never receive an empty list.

        Args:
            model_id: Bedrock model identifier.
            available_regions: Regions where this model is available.

        Returns:
            Ordered list of regions to try.
        """
        if len(available_regions) == 1:
            return available_regions

        usable: list[RegionName] = []
        blocked: list[RegionName] = []
        for region in available_regions:
            (usable if self._index.get(model_id, region).is_usable else blocked).append(
                region
            )

        return (
            (self._order(model_id, usable) + blocked)
            if usable
            else self._order(model_id, blocked)
        )

    def mark_error(self, model_id: str, region: str, error_code: str) -> None:
        """Applies a backoff penalty to a region based on the AWS error code received.

        Quota errors escalate exponentially up to _MAX_QUOTA_BACKOFF, and are
        published on the request scope so a terminal 429 can carry them as a
        ``retry-after`` header. Unavailability errors apply a fixed backoff.

        Args:
            model_id: Bedrock model identifier.
            region: AWS region that produced the error.
            error_code: AWS ClientError code string.
        """
        state = self._index.get(model_id, region)
        now = monotonic()

        if error_code in _QUOTA_ERROR_CODES:
            if (now - state.last_quota_error_time) > _QUOTA_STALE_THRESHOLD:
                state.consecutive_quota_errors = 1  # Stale: restart the escalation.
            else:
                state.consecutive_quota_errors += 1
            state.last_quota_error_time = now
            backoff = min(
                SETTINGS.aws_bedrock_region_routing_quota_backoff_seconds
                * (2 ** (state.consecutive_quota_errors - 1)),
                _MAX_QUOTA_BACKOFF,
            )
            state.quota_blocked_until = now + backoff
            _publish_quota_backoff(backoff)
            message = (
                f"warning: Region {region} blocked for model {model_id} "
                f"due to {error_code} (backoff: {backoff}s)"
            )
        else:
            backoff = SETTINGS.aws_bedrock_region_routing_unavailable_backoff_seconds
            state.unavailable_until = now + backoff
            message = (
                f"warning: Region {region} unavailable for model {model_id} "
                f"due to {error_code} (backoff: {backoff}s)"
            )

        log_error_details(message, level="warning")

    def mark_success(self, model_id: str, region: str) -> None:
        """Resets all backoff state for a region after a successful request.

        Args:
            model_id: Bedrock model identifier.
            region: AWS region that succeeded.
        """
        state = self._index.get(model_id, region)
        state.consecutive_quota_errors = 0
        state.last_quota_error_time = 0.0
        state.quota_blocked_until = 0.0
        state.unavailable_until = 0.0

    @staticmethod
    def _identity_order(
        model_id: str,  # noqa: ARG004
        usable: list[RegionName],
    ) -> list[RegionName]:
        """Returns usable unchanged; used by both ordered and lowest-latency strategies.

        Args:
            model_id: Unused.
            usable: Regions that are currently not blocked.

        Returns:
            usable as-is.
        """
        return usable

    def _round_robin_order(
        self, model_id: str, usable: list[RegionName]
    ) -> list[RegionName]:
        """Returns usable rotated so the next region leads, advancing the counter.

        Args:
            model_id: Identifies the per-model rotation counter.
            usable: Regions that are currently not blocked.

        Returns:
            usable rotated by the current counter index for this model.
        """
        if (idx := self._round_robin_counters.get(model_id)) is None:
            idx = randbelow(len(usable))
        idx %= len(usable)
        self._round_robin_counters[model_id] = idx + 1
        return usable[idx:] + usable[:idx]


#: Module-level router singleton. None when routing is disabled or fewer than two regions are configured.
REGION_ROUTER: RegionRouter | None = (
    None
    if (
        SETTINGS.aws_bedrock_region_routing == "disabled"
        or len(SETTINGS.aws_bedrock_regions) < 2
    )
    else RegionRouter()
)


async def _single_probe(session: ClientSession, url: str | None) -> float:
    """Runs a single HEAD probe and returns latency in ms.

    Args:
        session: Shared aiohttp ClientSession to reuse across probes.
        url: Endpoint URL to probe, or None if the region's endpoint could not
            be resolved (returns inf without probing).

    Returns:
        Round-trip latency in milliseconds, or inf if the probe fails.
    """
    if url is None:
        return float("inf")
    t0 = monotonic()
    try:
        async with session.head(url):
            pass
    except TimeoutError, AIOHTTPClientError, OSError:
        return float("inf")
    return (monotonic() - t0) * 1000


def _bedrock_probe_url(region: RegionName) -> str | None:
    """Return the partition-correct bedrock-runtime probe URL for *region*.

    Resolved via botocore so aws-cn/aws-eusc/... regions probe their own
    hostname instead of an ``.amazonaws.com`` one that never resolves for them.

    Args:
        region: AWS region to resolve.

    Returns:
        HTTPS probe URL, or None if the region has no known endpoint.
    """
    endpoint = _ENDPOINT_RESOLVER.construct_endpoint("bedrock-runtime", region)
    return f"https://{endpoint['hostname']}" if endpoint else None


async def measure_region_latencies() -> (
    dict[RegionName, dict[RegionLatenciesStatsKeys, float]] | None
):
    """Measures network latency to each configured Bedrock region and updates routing state.

    Returns:
        Per-region dict with latency_ms and stddev_ms (rounded to 1 decimal place),
        sorted by lowest latency first. None if routing is disabled or strategy is
        not lowest_latency.
    """
    if not (REGION_ROUTER and SETTINGS.aws_bedrock_region_routing == "lowest_latency"):
        return None

    probes_count = 3
    probe_urls = [
        _bedrock_probe_url(region)
        for region in ORDERED_BEDROCK_REGIONS
        for _ in range(probes_count)
    ]
    async with ClientSession(
        headers=HTTP_CLIENT_HEADERS,
        timeout=ClientTimeout(total=5),
        # Probe the path the invocations take: same proxy environment as the SDK.
        trust_env=True,
    ) as session:
        raw = list(await gather(*(_single_probe(session, url) for url in probe_urls)))

    results: dict[RegionName, dict[RegionLatenciesStatsKeys, float]] = {}
    for i, region in enumerate(ORDERED_BEDROCK_REGIONS):
        if samples := [
            ms for ms in raw[i * probes_count : (i + 1) * probes_count] if not isinf(ms)
        ]:
            latency = mean(samples)
            _REGION_LATENCIES[region] = latency
            results[region] = {
                "latency_ms": round(latency, 1),
                "stddev_ms": round(pstdev(samples), 1),
            }

    ORDERED_BEDROCK_REGIONS.clear()
    ORDERED_BEDROCK_REGIONS.extend(
        sorted(
            SETTINGS.aws_bedrock_regions,
            key=lambda r: _REGION_LATENCIES.get(r, float("inf")),
        )
    )
    return dict(sorted(results.items(), key=lambda item: item[1]["latency_ms"]))
