"""Per-tenant request and token limits, counted per minute in the shared table.

A tenant key is limited when its record declares ``requests_per_minute`` or
``tokens_per_minute``, or when the deployment defaults either
(``tenant_rate_limit_requests_per_minute``,
``tenant_rate_limit_tokens_per_minute``); the record wins. Only tenant keys
are limited: the deployment key and Cognito principals carry no per-key
identity to count against. With nothing declared, a request costs one
``None`` check and no table call.

Windows are fixed, sixty seconds long and aligned to the minute, one counter
item per key and window (``pk=LIMIT#<key id>``, ``sk=<window start>``), which
the table's time-to-live drops two minutes after the window ends. Every
instance shares the item through one atomic ``ADD`` that answers with the
totals, so a grant decided from those totals can never exceed the limit
across instances -- an over-count in the item is never an admission.

The write is kept off the steady-state request path. Each instance keeps a
ledger per key and window: request slots are reserved in batches that double
within the window, a refill starts asynchronously when the local remainder
drops below half the last batch, and only a request that finds the ledger
empty waits for a write -- the first request of each minute per key per
instance. Tokens are only known once the model has answered, so a request
is admitted on an estimate -- this instance's mean tokens per billed request
for the key, a fixed share of the limit until one was billed, never above the
limit itself -- and reconciled from the billed usage afterwards, on the
window that is current when the usage arrives; billed tokens are flushed at
most once a second per key per instance, and every write refreshes the
ledger's view of the global counters. The overshoot a token limit allows is
therefore bounded by the requests in flight times their estimation error,
plus one second of every instance's traffic -- and since the mean is taught
by the key's own traffic, the requests a key may hold unbilled in flight are
themselves capped per instance while only a token limit is declared. Usage a
request settles rather than produces -- a finished batch job's -- is never
counted.

A counter that cannot be written fails closed once the slots already granted
are used up: the refusal is the generic feature-unavailable ``503``, since a
``429`` would tell the client it is over a limit it never reached, and the
operator reads the IAM action, the table and the region in the log.
"""

from __future__ import annotations

from asyncio import Task, create_task, shield, sleep
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from time import monotonic, time
from typing import TYPE_CHECKING, Any, Final, NoReturn

from stdapi.api_errors import FeatureUnavailableError, RateLimitExceededError
from stdapi.api_providers.anthropic import TAG_ANTHROPIC
from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.aws_dynamodb import TableUnavailableError, add_to_item, item_key
from stdapi.config import SETTINGS
from stdapi.monitoring import (
    REQUEST,
    add_server_warning,
    log_background_event,
    log_error_details,
)
from stdapi.pricing import Dimension
from stdapi.server import INTERNAL_REQUEST_ID_HEADER, MCP_USER_AGENT
from stdapi.utils import webuuid

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, MutableMapping

    from fastapi import Request
    from starlette.requests import HTTPConnection

    from stdapi.aws_dynamodb import Item
    from stdapi.monitoring import EventLog, Tenant
    from stdapi.usage import UsageRecord

#: Seconds in one fixed window; every limit is per minute.
WINDOW_SECONDS: Final = 60

#: Counter attribute holding the requests admitted in the window.
REQUESTS_ATTRIBUTE: Final = "requests"

#: Counter attribute holding the tokens billed in the window.
TOKENS_ATTRIBUTE: Final = "tokens"

#: Request-scope state key carrying the reservation the response is answered from.
STATE_KEY: Final = "stdapi_tenant_rate_limit"

#: Partition namespace of the counter items, the only ones written in place.
_KIND: Final = "LIMIT"

#: Key ID the startup probe counts under; no minted key ID is this short.
_PROBE_KEY_ID: Final = "startup"

#: Seconds a counter item outlives its window before the table's time-to-live drops it.
_EXPIRY_GRACE: Final = 120

#: Feature name the fail-closed refusal answers with.
_FEATURE: Final = "Tenant rate limiting"

#: Seconds between two writes flushing billed tokens, per key per instance.
_FLUSH_SECONDS: Final = 1.0

#: Seconds a failed counter write is not retried, so a broken table is not hammered.
_RETRY_SECONDS: Final = 1.0

#: Divisor of the request limit bounding one slot batch, so one instance strands little of it.
_BATCH_DIVISOR: Final = 8

#: Divisor of the token limit estimating a request of a key never billed here.
_PRIOR_DIVISOR: Final = 8

#: Requests a key may hold unbilled in flight per instance on a token limit alone.
_INFLIGHT_MAX: Final = 64

#: Ledger count past which the ledgers of past windows are swept.
_LEDGERS_MAX: Final = 4096

#: Service tier of usage a request settles rather than produces, never counted.
_BATCH_TIER: Final = "batch"

#: Dimensions a token limit counts: cache reads are free, cache writes are not.
_COUNTED_DIMENSIONS: Final = (
    Dimension.INPUT_TOKENS,
    Dimension.CACHE_WRITE_TOKENS,
    Dimension.OUTPUT_TOKENS,
)

#: User agent of the internal MCP-to-API leg, as the ASGI scope carries it.
_MCP_USER_AGENT: Final = MCP_USER_AGENT.encode()

#: Request-ID header of the internal MCP-to-API leg, as the ASGI scope carries it.
_INTERNAL_HEADER: Final = INTERNAL_REQUEST_ID_HEADER.lower().encode()


@dataclass(slots=True)
class _Ledger:
    """One instance's view of one key's counter for one window.

    Attributes:
        key_id: The tenant key.
        window: Epoch second the window starts at.
        requests_limit: Requests the key may make in the window, if limited.
        tokens_limit: Tokens the key may bill in the window, if limited.
        estimate: Mean tokens per request carried over from the previous window.
        synced: Whether a write has answered with the global totals yet.
        granted: Request slots reserved here and not used yet.
        last_batch: Slots the last reservation asked for.
        waiting: Requests waiting for a slot or for the totals.
        exhausted: Whether the window's request limit is reached globally.
        known_requests: Requests counted globally, as of the last write.
        known_tokens: Tokens counted globally, as of the last write.
        unflushed_tokens: Tokens billed here and not written yet.
        inflight_estimate: Estimated tokens of the requests admitted and not billed yet.
        inflight_requests: Requests admitted here and not billed yet.
        observed_tokens: Tokens billed here this window, for the estimate.
        observed_requests: Requests billed here this window, for the estimate.
        failure: What the operator must fix, while the last write failed.
        failed_at: Monotonic clock reading of that failure.
        last_flush: Monotonic clock reading of the last successful write.
        sync_task: The write in flight, if any.
        flush_task: The delayed token flush, if any.
    """

    key_id: str
    window: int
    requests_limit: int | None
    tokens_limit: int | None
    estimate: int = 0
    synced: bool = False
    granted: int = 0
    last_batch: int = 0
    waiting: int = 0
    exhausted: bool = False
    known_requests: int = 0
    known_tokens: int = 0
    unflushed_tokens: int = 0
    inflight_estimate: int = 0
    inflight_requests: int = 0
    observed_tokens: int = 0
    observed_requests: int = 0
    failure: str | None = None
    failed_at: float = 0.0
    last_flush: float = 0.0
    sync_task: Task[None] | None = None
    flush_task: Task[None] | None = None

    @property
    def window_end(self) -> int:
        """Epoch second the window ends at."""
        return self.window + WINDOW_SECONDS

    def seconds_to_reset(self) -> int:
        """Whole seconds until the window ends, at least one."""
        return min(WINDOW_SECONDS, max(1, ceil(self.window_end - time())))

    def current_estimate(self) -> int:
        """Mean tokens per request billed here, else the carried-over one."""
        if self.observed_requests:
            return self.observed_tokens // self.observed_requests
        return self.estimate

    def admission_estimate(self) -> int:
        """Tokens a request is counted for in flight, 0 without a token limit.

        A key never billed on this instance is estimated at a fixed share of
        its limit, so a burst is bounded before the first response teaches
        the mean; an estimate above the limit would refuse every window's
        first request for ever, so it is capped at the limit.
        """
        if self.tokens_limit is None:
            return 0
        estimate = self.current_estimate() or self.tokens_limit // _PRIOR_DIVISOR
        return min(estimate, self.tokens_limit)

    def projected_tokens(self) -> int:
        """Tokens the window holds once everything in flight is billed."""
        return self.known_tokens + self.unflushed_tokens + self.inflight_estimate

    def remaining_requests(self, limit: int) -> int:
        """Requests this instance can still admit, never below zero."""
        return max(0, limit - self.known_requests + self.granted)

    def remaining_tokens(self, limit: int) -> int:
        """Tokens this instance can still admit, never below zero."""
        return max(0, limit - self.projected_tokens())

    def next_batch(self) -> int:
        """Slots the next reservation asks for: doubling, sized to the waiters, capped."""
        limit = self.requests_limit or 1
        cap = max(1, limit // _BATCH_DIVISOR)
        return min(cap, max(1, self.last_batch * 2, self.waiting))


@dataclass(slots=True)
class _Reservation:
    """What one admitted -- or refused -- request holds on its ledger.

    Attributes:
        ledger: The ledger the request was decided from.
        estimate: Tokens counted in flight for it until billed.
        refused: Whether the request was refused rather than admitted.
        released: Whether the estimate was given back to the ledger.
        counted_window: Window the request was last counted in the mean of.
    """

    ledger: _Ledger
    estimate: int
    refused: bool = False
    released: bool = False
    counted_window: int = 0

    def release(self) -> None:
        """Give the in-flight estimate back, once, to the ledger now holding it.

        A rollover carries what is still in flight onto the new window's
        ledger, so a request admitted in an earlier window is given back
        there rather than to the ledger it was taken from.
        """
        if not self.released and not self.refused:
            self.released = True
            ledger = _current_ledger(self.ledger)
            ledger.inflight_estimate -= self.estimate
            ledger.inflight_requests -= 1

    def debit(self, tokens: int) -> None:
        """Count the tokens the request was billed, on the key's current window.

        A debit of nothing settles nothing: a streamed response drains its
        request scope before any token arrives, and its estimate stays in
        flight until the trailing usage does. Only a billed request feeds the
        mean, once per window it bills in: requests billing nothing cannot
        talk the estimate down, and a session draining every turn does not
        teach the mean one turn's tokens.

        Args:
            tokens: Tokens billed since the last debit.
        """
        if self.refused or not tokens:
            return
        self.release()
        ledger = _current_ledger(self.ledger)
        ledger.observed_tokens += tokens
        if ledger.window != self.counted_window:
            self.counted_window = ledger.window
            ledger.observed_requests += 1
        ledger.unflushed_tokens += tokens
        if ledger.tokens_limit is not None:
            _schedule_flush(ledger)


#: Ledgers by key ID, replaced at the window boundary.
_LEDGERS: dict[str, _Ledger] = {}

#: Window the ledgers of past windows were last swept in.
_SWEPT_WINDOW = 0

#: Reservation of the current request, inherited by the internal MCP-to-API leg.
_RESERVATION: ContextVar[_Reservation | None] = ContextVar("reservation", default=None)


def resolve_limits(tenant: Tenant) -> tuple[int | None, int | None]:
    """Return the request and token limits that apply to *tenant*.

    Args:
        tenant: The verified tenant.

    Returns:
        The tenant record's limits, each falling back to the deployment
        default; None where neither declares one.
    """
    return (
        tenant.requests_per_minute or SETTINGS.tenant_rate_limit_requests_per_minute,
        tenant.tokens_per_minute or SETTINGS.tenant_rate_limit_tokens_per_minute,
    )


def _window(now: float) -> int:
    """Epoch second the window holding *now* starts at."""
    return int(now) // WINDOW_SECONDS * WINDOW_SECONDS


def _ledger_for(
    key_id: str, requests_limit: int | None, tokens_limit: int | None
) -> _Ledger:
    """Return the ledger of *key_id* for the current window, rolling it over.

    Args:
        key_id: The tenant key.
        requests_limit: The request limit in force.
        tokens_limit: The token limit in force.

    Returns:
        The ledger, fresh when the window or the limits changed.
    """
    global _SWEPT_WINDOW  # noqa: PLW0603
    window = _window(time())
    ledger = _LEDGERS.get(key_id)
    if (
        ledger is not None
        and ledger.window == window
        and ledger.requests_limit == requests_limit
        and ledger.tokens_limit == tokens_limit
    ):
        return ledger
    estimate = ledger.current_estimate() if ledger is not None else 0
    inflight_requests = ledger.inflight_requests if ledger is not None else 0
    inflight_estimate = ledger.inflight_estimate if ledger is not None else 0
    # Once a window: a registry full of this window's keys has nothing to drop.
    if ledger is None and len(_LEDGERS) >= _LEDGERS_MAX and window != _SWEPT_WINDOW:
        _SWEPT_WINDOW = window
        for stale in [k for k, v in _LEDGERS.items() if v.window < window]:
            del _LEDGERS[stale]
    ledger = _Ledger(
        key_id,
        window,
        requests_limit,
        tokens_limit,
        estimate=estimate,
        # What is still in flight is still in flight: a request admitted last
        # window and not yet billed keeps counting against the ceiling, or a
        # long session would buy a fresh allowance every minute it stays open.
        inflight_requests=inflight_requests,
        inflight_estimate=inflight_estimate,
    )
    _LEDGERS[key_id] = ledger
    return ledger


def _current_ledger(ledger: _Ledger) -> _Ledger:
    """Return the ledger of *ledger*'s key for the current window.

    Args:
        ledger: The ledger a request was admitted from, possibly of a window
            that has ended since.

    Returns:
        The key's ledger for the current window: *ledger* itself while its
        window lasts, the one built since when the limits changed, else a
        fresh one carrying its estimate.
    """
    current = _LEDGERS.get(ledger.key_id)
    if current is not None and current.window == _window(time()):
        return current
    return _ledger_for(ledger.key_id, ledger.requests_limit, ledger.tokens_limit)


def _count(totals: Item, attribute: str) -> int:
    """Read one counter total off a write's answer.

    Args:
        totals: The attributes the write answered with.
        attribute: The counter's name.

    Returns:
        The total, 0 when absent.
    """
    value = totals.get(attribute)
    return value if isinstance(value, int) else 0


async def _sync(ledger: _Ledger, slots: int) -> None:
    """Write *slots* reservations and the pending tokens, and read the totals back.

    Args:
        ledger: The ledger to write for.
        slots: Request slots to reserve, 0 for a token flush.
    """
    tokens = ledger.unflushed_tokens
    ledger.unflushed_tokens = 0
    try:
        totals = await add_to_item(
            item_key(_KIND, ledger.key_id),
            str(ledger.window),
            {REQUESTS_ATTRIBUTE: slots, TOKENS_ATTRIBUTE: tokens},
            expires_at=ledger.window_end + _EXPIRY_GRACE,
        )
    except TableUnavailableError as error:
        ledger.unflushed_tokens += tokens
        ledger.failure = error.detail
        ledger.failed_at = monotonic()
        # A waiter reports the failure with its own refusal; a background
        # flush has nobody to report it, so it does so itself.
        if not ledger.waiting:
            with log_background_event("tenant_rate_limit_sync", webuuid()):
                log_error_details(error.detail, level="warning")
        return
    finally:
        ledger.sync_task = None
    ledger.failure = None
    ledger.synced = True
    ledger.last_flush = monotonic()
    ledger.known_requests = _count(totals, REQUESTS_ATTRIBUTE)
    ledger.known_tokens = _count(totals, TOKENS_ATTRIBUTE)
    if slots and ledger.requests_limit is not None:
        ledger.last_batch = slots
        # Only what fits under the limit is granted: the item may over-count.
        grant = max(
            0, min(slots, ledger.requests_limit - (ledger.known_requests - slots))
        )
        ledger.granted += grant
        if grant < slots:
            ledger.exhausted = True
    if ledger.unflushed_tokens and ledger.tokens_limit is not None:
        _schedule_flush(ledger)


async def _sync_now(ledger: _Ledger, slots: int) -> None:
    """Await a write for *ledger*, joining the one in flight if any.

    Args:
        ledger: The ledger to write for.
        slots: Request slots to reserve when starting a write.
    """
    if ledger.sync_task is None:
        ledger.sync_task = create_task(_sync(ledger, slots))
    # Shielded: a waiter cancelled by its client must not cancel the write
    # every other waiter is waiting on.
    await shield(ledger.sync_task)


def _kick(ledger: _Ledger, slots: int) -> None:
    """Start a write for *ledger* without waiting for it, unless one is in flight.

    Args:
        ledger: The ledger to write for.
        slots: Request slots to reserve.
    """
    if ledger.sync_task is None:
        ledger.sync_task = create_task(_sync(ledger, slots))


async def _flush_later(ledger: _Ledger, delay: float) -> None:
    """Flush the pending tokens once the coalescing delay has passed.

    Args:
        ledger: The ledger to flush.
        delay: Seconds to wait first.
    """
    await sleep(delay)
    ledger.flush_task = None
    # A window that ended is forgiven, and is never retried forever.
    if ledger.unflushed_tokens and _window(time()) == ledger.window:
        await _sync_now(ledger, 0)


def _schedule_flush(ledger: _Ledger) -> None:
    """Flush the pending tokens within the coalescing delay, once.

    Args:
        ledger: The ledger to flush.
    """
    if ledger.flush_task is None:
        delay = max(0.0, _FLUSH_SECONDS - (monotonic() - ledger.last_flush))
        ledger.flush_task = create_task(_flush_later(ledger, delay))


def _refuse(
    ledger: _Ledger, reservation: _Reservation, limit: int, unit: str
) -> NoReturn:
    """Refuse the request with the vendors' 429, leaving its state readable.

    Args:
        ledger: The ledger the limit was reached on.
        reservation: The request's reservation, marked refused.
        limit: The limit reached.
        unit: What it counts, ``"request"`` or ``"token"``.

    Raises:
        RateLimitExceededError: Always.
    """
    reservation.refused = True
    raise RateLimitExceededError(ledger.key_id, limit, unit, ledger.seconds_to_reset())


async def _ensure(ledger: _Ledger, *, need_slot: bool) -> None:
    """Write until the ledger can answer: a slot is held, or the totals are known.

    Args:
        ledger: The ledger to bring up to date.
        need_slot: Whether a request slot must be held on return.

    Raises:
        FeatureUnavailableError: The counter cannot be written and nothing is
            held to serve from.
    """
    ledger.waiting += 1
    try:
        while True:
            # Slots already held keep serving through a failing counter.
            if need_slot and (ledger.granted > 0 or ledger.exhausted):
                return
            if ledger.failure is not None:
                if monotonic() - ledger.failed_at < _RETRY_SECONDS:
                    raise FeatureUnavailableError(_FEATURE, ledger.failure)
            elif not need_slot and ledger.synced:
                return
            await _sync_now(ledger, ledger.next_batch() if need_slot else 0)
    finally:
        ledger.waiting -= 1


def _is_internal_leg(scope: Mapping[str, Any] | None) -> bool:
    """Whether *scope* is the internal MCP-to-API leg of a tool call.

    Tests the scope it is handed, which may be another connection's or none
    at all, where ``mcp.is_mcp`` tests the ambient request.

    Args:
        scope: The connection's ASGI scope.

    Returns:
        True when it carries both markers the MCP transport stamps, which no
        external client can present together.
    """
    if scope is None:
        return False
    found = 0
    for name, value in scope.get("headers") or ():
        if name == b"user-agent" and value == _MCP_USER_AGENT:
            found |= 1
        elif name == _INTERNAL_HEADER:
            found |= 2
    return found == 3


async def admit_tenant_request(
    scope: MutableMapping[str, Any] | None, tenant: Tenant
) -> None:
    """Count one request of *tenant* against its limits, or refuse it.

    No-op when the tenant has no limit. The internal MCP-to-API leg of a
    tool call is not counted again: it inherits the MCP request's
    reservation, so its billed tokens land on the same one -- as long as
    that reservation is the same tenant's and its window still lasts, since
    a stateful MCP session outlives the request that opened it.

    Args:
        scope: The connection's ASGI scope, whose state carries the
            reservation the response headers and the usage debit read.
        tenant: The verified tenant.

    Raises:
        RateLimitExceededError: A limit is reached for this window, or the
            key holds as many unbilled requests in flight as a token limit
            alone allows.
        FeatureUnavailableError: The counter cannot be written and no slot
            is held to serve from.
    """
    requests_limit, tokens_limit = resolve_limits(tenant)
    if requests_limit is None and tokens_limit is None:
        return
    state: MutableMapping[str, Any] = (
        scope.setdefault("state", {}) if scope is not None else {}
    )
    if (
        _is_internal_leg(scope)
        and (outer := _RESERVATION.get()) is not None
        and outer.ledger.key_id == tenant.key_id
        and outer.ledger.window == _window(time())
    ):
        state[STATE_KEY] = outer
        return
    ledger = _ledger_for(tenant.key_id, requests_limit, tokens_limit)
    reservation = _Reservation(ledger, ledger.admission_estimate())
    state[STATE_KEY] = reservation
    _RESERVATION.set(reservation)
    try:
        await _ensure(ledger, need_slot=requests_limit is not None)
    except BaseException:
        # Nothing was taken -- the 503, a cancellation, any failure -- so the
        # settling of the exit has nothing to give back.
        reservation.refused = True
        raise
    # No await from here to the take: the checks and the take are one step.
    if tokens_limit is not None and (
        ledger.projected_tokens() + max(reservation.estimate, 1) > tokens_limit
        or (requests_limit is None and ledger.inflight_requests >= _INFLIGHT_MAX)
    ):
        _refuse(ledger, reservation, tokens_limit, "token")
    if requests_limit is not None:
        if ledger.granted == 0:
            _refuse(ledger, reservation, requests_limit, "request")
        ledger.granted -= 1
        if (
            ledger.granted < ledger.last_batch // 2
            and not ledger.exhausted
            and ledger.failure is None
        ):
            _kick(ledger, ledger.next_batch())
    ledger.inflight_estimate += reservation.estimate
    ledger.inflight_requests += 1
    if scope is None:
        # Nothing carries the reservation, so nothing could ever release it.
        reservation.release()


def _reservation_of(scope: Mapping[str, Any]) -> _Reservation | None:
    """Read the reservation the connection's scope carries, if any.

    Args:
        scope: The connection's ASGI scope.

    Returns:
        The reservation, or None when the request was not limited.
    """
    reservation = (scope.get("state") or {}).get(STATE_KEY)
    return reservation if isinstance(reservation, _Reservation) else None


def debit_tenant_tokens(records: Iterable[UsageRecord]) -> None:
    """Count the billed tokens of the current request against its tenant.

    Called wherever the usage of a response is drained -- the request scope,
    the streamed-body scope and a realtime turn -- so a token limit sees what
    AWS billed, whichever shape the response took. A finished batch job's
    usage is settled by whichever request reads it first, which may be another
    tenant's: it is never the reader's to pay for.

    Args:
        records: The usage records being drained.
    """
    if (request := REQUEST.get(None)) is None:
        return
    if (reservation := _reservation_of(request.scope)) is None:
        return
    reservation.debit(
        sum(
            record.quantities.get(dimension, 0)
            for record in records
            if record.tier != _BATCH_TIER
            for dimension in _COUNTED_DIMENSIONS
        )
    )


async def settle_tenant_reservation(connection: HTTPConnection) -> None:
    """Give the connection's in-flight estimate back once it is answered or closed.

    A response billed no tokens at all never debits, so this is what releases
    its estimate; a response that did has released it already. A WebSocket
    bills nothing before its first turn, so its close is what releases a
    handshake refused after admission or a session closed unbilled.

    Args:
        connection: The request whose response was sent, or the WebSocket
            that closed.
    """
    if (reservation := _reservation_of(connection.scope)) is not None:
        reservation.release()


def tenant_retry_after(request: Request) -> int | None:
    """Return the ``retry-after`` delay of a request refused for its tenant's limit.

    Args:
        request: The request.

    Returns:
        Whole seconds until the window resets, or None when the request was
        not refused by the limiter.
    """
    reservation = _reservation_of(request.scope)
    if reservation is None or not reservation.refused:
        return None
    return reservation.ledger.seconds_to_reset()


def _go_duration(seconds: int) -> str:
    """Format a delay as Go's ``Duration`` prints it, e.g. ``12s`` or ``1m0s``."""
    return f"{seconds // 60}m{seconds % 60}s" if seconds >= 60 else f"{seconds}s"


def _rfc3339(epoch: int) -> str:
    """Format an epoch second as an RFC 3339 UTC timestamp."""
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _openai_headers(ledger: _Ledger) -> dict[str, str]:
    """Build the OpenAI rate-limit headers from *ledger*."""
    reset = _go_duration(ledger.seconds_to_reset())
    headers: dict[str, str] = {}
    if (limit := ledger.requests_limit) is not None:
        headers["x-ratelimit-limit-requests"] = str(limit)
        headers["x-ratelimit-remaining-requests"] = str(
            ledger.remaining_requests(limit)
        )
        headers["x-ratelimit-reset-requests"] = reset
    if (limit := ledger.tokens_limit) is not None:
        headers["x-ratelimit-limit-tokens"] = str(limit)
        headers["x-ratelimit-remaining-tokens"] = str(ledger.remaining_tokens(limit))
        headers["x-ratelimit-reset-tokens"] = reset
    return headers


def _anthropic_headers(ledger: _Ledger) -> dict[str, str]:
    """Build the Anthropic rate-limit headers from *ledger*."""
    reset = _rfc3339(ledger.window_end)
    headers: dict[str, str] = {}
    if (limit := ledger.requests_limit) is not None:
        headers["anthropic-ratelimit-requests-limit"] = str(limit)
        headers["anthropic-ratelimit-requests-remaining"] = str(
            ledger.remaining_requests(limit)
        )
        headers["anthropic-ratelimit-requests-reset"] = reset
    if (limit := ledger.tokens_limit) is not None:
        headers["anthropic-ratelimit-tokens-limit"] = str(limit)
        headers["anthropic-ratelimit-tokens-remaining"] = str(
            ledger.remaining_tokens(limit)
        )
        headers["anthropic-ratelimit-tokens-reset"] = reset
    return headers


#: Route tag to the builder of that dialect's rate-limit headers.
_HEADERS_BY_TAG: Final[dict[str, Callable[[_Ledger], dict[str, str]]]] = {
    TAG_OPENAI: _openai_headers,
    TAG_ANTHROPIC: _anthropic_headers,
}


def rate_limit_headers(request: Request) -> dict[str, str]:
    """Return the rate-limit headers of the request's dialect, from its reservation.

    The remaining counts are this instance's view: what it holds reserved on
    top of the global totals it last read.

    Args:
        request: The request.

    Returns:
        The headers, empty when the request was not limited or its dialect
        publishes none.
    """
    if (reservation := _reservation_of(request.scope)) is None:
        return {}
    route = request.scope.get("route")
    for tag in getattr(route, "tags", None) or ():
        if (build := _HEADERS_BY_TAG.get(tag)) is not None:
            return build(reservation.ledger)
    return {}


async def verify_counter_access(start_event: EventLog) -> None:
    """Write one probe counter at startup, so a denied write is known there.

    Reported and never fatal, like every other startup check of an
    operator-provided resource.

    Args:
        start_event: Startup log event the finding is reported on.
    """
    try:
        await add_to_item(
            item_key(_KIND, _PROBE_KEY_ID),
            "probe",
            {REQUESTS_ATTRIBUTE: 1},
            expires_at=int(time()) + WINDOW_SECONDS,
        )
    except TableUnavailableError as error:
        add_server_warning(
            start_event, f"Tenant rate limits cannot be enforced yet: {error.detail}"
        )
