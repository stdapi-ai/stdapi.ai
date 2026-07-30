"""AWS Bedrock region routing, failover backoff and ``InputFile`` source resolution.

Covers the ``RegionRouter`` strategies (ordered / round-robin / lowest-latency),
``route_and_execute`` failover bookkeeping, ``compute_candidate_regions`` S3
locality rules, and the ``InputFile`` source backends those rules depend on.

Uses the session-scoped ``openai_client`` from conftest directly.
Per-strategy fixtures patch SETTINGS in-place, swap the two name-bound
REGION_ROUTER copies, and inject fresh no-retry clients for AioStubber.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html
     stdapi/region_routing.py:RegionRouter
     stdapi/models/__init__.py:route_and_execute
     stdapi/input_file.py:InputFile
"""

from contextlib import AsyncExitStack, asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any, Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from botocore.exceptions import ConnectionError as BotocoreConnectionError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, Iterator

    from aiobotocore.config import AioConfig
    from openai import OpenAI
    from openai.types.chat import ChatCompletionUserMessageParam
    from types_aiobotocore_bedrock.literals import RegionName

    import stdapi.region_routing as _rr_mod
    from stdapi.region_routing import RegionState


pytestmark = [
    pytest.mark.xdist_group("region_routing"),
    pytest.mark.asyncio(loop_scope="module"),
]

MODEL = "amazon.nova-micro-v1:0"
_MESSAGES: list[ChatCompletionUserMessageParam] = [
    {"role": "user", "content": "Say the number 1 only."}
]
#: Output-token cap for the routing calls -- only the presence of content is asserted.
_MAX_TOKENS = 8
ROUTING_PRIMARY: RegionName = "us-east-1"
ROUTING_SECONDARY: RegionName = "us-west-2"
_ROUTING_REGIONS: list[RegionName] = [ROUTING_PRIMARY, ROUTING_SECONDARY]
#: Quota backoff base in seconds -- mirrors the fixture override below.
_QUOTA_BACKOFF_BASE = 60
#: Unavailability backoff in seconds -- mirrors the fixture override below.
_UNAVAILABLE_BACKOFF = 30


def _no_retry_config() -> AioConfig:
    """Build a single-attempt AioConfig for injected stub clients."""
    from aiobotocore.config import AioConfig  # noqa: PLC0415

    return AioConfig(
        retries={"max_attempts": 1, "mode": "adaptive"},
        max_pool_connections=10,
        parameter_validation=False,
        connect_timeout=5,
    )


# ---------------------------------------------------------------------------
# RoutingFixture
# ---------------------------------------------------------------------------


@dataclass
class RoutingFixture:
    """Test handle providing the OpenAI client, router, and stub helpers."""

    openai: OpenAI
    router: _rr_mod.RegionRouter | None
    no_retry_clients: dict[str, Any]

    def get_state(self, model: str, region: str) -> RegionState:
        """Return the live RegionState for the given model/region pair."""
        assert self.router is not None
        return self.router._index.get(model, region)  # noqa: SLF001

    def mark_success(self, model: str, region: str) -> None:
        """Reset a blocked region back to usable."""
        assert self.router is not None
        self.router.mark_success(model, region)

    def reset(self, model: str) -> None:
        """Clear all error state and round-robin counters for *model*."""
        if self.router is None:
            return
        for region in _ROUTING_REGIONS:
            self.router.mark_success(model, region)
        if hasattr(self.router, "_round_robin_counters"):
            self.router._round_robin_counters.pop(model, None)  # noqa: SLF001

    def set_rr_counter(self, model: str, idx: int) -> None:
        """Force the round-robin counter to a specific index."""
        if self.router is not None and hasattr(self.router, "_round_robin_counters"):
            self.router._round_robin_counters[model] = idx  # noqa: SLF001

    @contextmanager
    def stub_errors(self, region: str, errors: list[tuple[str, str]]) -> Iterator[None]:
        """Enqueue stubbed AWS errors for *region* for the duration of the block."""
        from aiobotocore.stub import AioStubber  # noqa: PLC0415

        stubber = AioStubber(self.no_retry_clients[region])
        for op, code in errors:
            stubber.add_client_error(op, service_error_code=code)
        stubber.activate()
        try:
            yield
        finally:
            stubber.deactivate()


# ---------------------------------------------------------------------------
# Core fixture context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _routing_fixture_context(
    strategy: str, openai_client: OpenAI, extra_settings: dict[str, Any] | None = None
) -> AsyncGenerator[RoutingFixture]:
    """Set up a routing test environment for *strategy* and yield a RoutingFixture.

    All stdapi imports are deferred to inside this function so that importing
    this test module during pytest collection does not instantiate SETTINGS
    before the test_client fixture has set environ["api_key"].
    """
    import stdapi.aws as _aws_mod  # noqa: PLC0415
    import stdapi.models as _models_mod  # noqa: PLC0415
    import stdapi.region_routing as _rr_mod  # noqa: PLC0415
    from stdapi.config import AWS_SESSION, SETTINGS  # noqa: PLC0415

    effective_regions: list[str] = (extra_settings or {}).get(
        "aws_bedrock_regions", _ROUTING_REGIONS
    )

    # Temporarily set the strategy so RegionRouter.__init__ picks it up,
    # then restore immediately — the patch.object below handles test-time value.
    orig_strategy = SETTINGS.aws_bedrock_region_routing
    SETTINGS.aws_bedrock_region_routing = strategy  # type: ignore[assignment]
    router: _rr_mod.RegionRouter | None = (
        _rr_mod.RegionRouter() if len(effective_regions) >= 2 else None
    )
    SETTINGS.aws_bedrock_region_routing = orig_strategy

    # create_client() returns a ClientCreatorContext (async context manager),
    # not a plain coroutine — use AsyncExitStack to own all clients.
    exit_stack = AsyncExitStack()
    no_retry_clients: dict[str, Any] = {}
    await exit_stack.__aenter__()
    try:
        for region in effective_regions:
            no_retry_clients[region] = await exit_stack.enter_async_context(
                AWS_SESSION.create_client(
                    "bedrock-runtime", region_name=region, config=_no_retry_config()
                )
            )
    except BaseException:
        with suppress(RuntimeError):
            await exit_stack.__aexit__(None, None, None)
        raise

    orig_no_retry = _aws_mod._CLIENTS.get("bedrock-runtime.no-retry")  # noqa: SLF001
    _aws_mod._CLIENTS["bedrock-runtime.no-retry"] = no_retry_clients  # type: ignore[assignment]  # noqa: SLF001

    try:
        # compute_candidate_regions returns all model.available_regions from the
        # startup-time cache, which may include regions absent in effective_regions.
        # Wrap it to restrict candidates so bedrock_client() never KeyErrors.
        _orig_ccr = _models_mod.compute_candidate_regions
        _effective_regions_set = set(effective_regions)

        async def _filtered_ccr(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            candidates = await _orig_ccr(*args, **kwargs)
            filtered = [r for r in candidates if r in _effective_regions_set]
            return filtered or candidates  # fall back if nothing matches

        with (
            patch.object(SETTINGS, "aws_bedrock_regions", effective_regions),
            patch.object(SETTINGS, "aws_bedrock_region_routing", strategy),
            patch.object(SETTINGS, "aws_bedrock_max_retries", 2),
            patch.object(
                SETTINGS,
                "aws_bedrock_region_routing_quota_backoff_seconds",
                _QUOTA_BACKOFF_BASE,
            ),
            patch.object(
                SETTINGS,
                "aws_bedrock_region_routing_unavailable_backoff_seconds",
                _UNAVAILABLE_BACKOFF,
            ),
            patch.object(_rr_mod, "REGION_ROUTER", router),
            patch.object(_rr_mod, "ORDERED_BEDROCK_REGIONS", list(effective_regions)),
            patch.object(_models_mod, "REGION_ROUTER", router),
            patch.object(_models_mod, "compute_candidate_regions", _filtered_ccr),
        ):
            yield RoutingFixture(
                openai=openai_client, router=router, no_retry_clients=no_retry_clients
            )
    finally:
        if orig_no_retry is None:
            _aws_mod._CLIENTS.pop("bedrock-runtime.no-retry", None)  # noqa: SLF001
        else:
            _aws_mod._CLIENTS["bedrock-runtime.no-retry"] = orig_no_retry  # noqa: SLF001
        # suppress RuntimeError: the module event loop may differ at teardown,
        # leaving the aiohttp connector's _wait_for_close unawaited (harmless).
        with suppress(RuntimeError):
            await exit_stack.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Per-strategy fixtures
# ---------------------------------------------------------------------------


def _make_strategy_fixture(strategy: str) -> Any:  # noqa: ANN401
    """Return a module-scoped pytest-asyncio fixture for *strategy*."""

    @pytest_asyncio.fixture(scope="module", loop_scope="module")
    async def _fixture(openai_client: OpenAI) -> AsyncGenerator[RoutingFixture]:
        async with _routing_fixture_context(strategy, openai_client) as ctx:
            yield ctx

    _fixture.__name__ = f"routing_{strategy}"
    return _fixture


routing_ordered: Any = _make_strategy_fixture("ordered")
routing_round_robin: Any = _make_strategy_fixture("round_robin")
routing_lowest_latency: Any = _make_strategy_fixture("lowest_latency")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def routing_single_region(
    openai_client: OpenAI,
) -> AsyncGenerator[RoutingFixture]:
    """Fixture with a single configured region — no router is created."""
    async with _routing_fixture_context(
        "ordered",
        openai_client,
        extra_settings={"aws_bedrock_regions": [ROUTING_PRIMARY]},
    ) as ctx:
        yield ctx


# ---------------------------------------------------------------------------
# Per-test cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_router_state(request: pytest.FixtureRequest) -> Generator[None]:
    """Reset router error/counter state for MODEL after each test."""
    yield
    for name in ("routing_ordered", "routing_round_robin", "routing_lowest_latency"):
        try:
            fx: RoutingFixture = request.getfixturevalue(name)
        except pytest.FixtureLookupError:
            continue
        fx.reset(MODEL)
        break


@pytest.fixture(autouse=True, scope="module")
def _require_local_mode(request: pytest.FixtureRequest) -> None:
    """Skip this module when running against a remote server."""
    if request.config.getoption("--use-official-api") or request.config.getoption(
        "--server-url"
    ):
        pytest.skip("Region routing tests require local mode")


# ---------------------------------------------------------------------------
# Group 1 — Ordered strategy
# ---------------------------------------------------------------------------


class TestOrderedRouting:
    """Ordered strategy always prefers the first usable region in the list.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html
         stdapi/region_routing.py:RegionRouter.ordered_regions
         stdapi/models/__init__.py:route_and_execute
    """

    def test_success_uses_primary_region(self, routing_ordered: RoutingFixture) -> None:
        """Successful request leaves the primary region leading with no quota errors recorded."""
        response = routing_ordered.openai.chat.completions.create(
            model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
        )
        assert response.choices[0].message.content
        state = routing_ordered.get_state(MODEL, ROUTING_PRIMARY)
        assert state.is_usable
        assert state.consecutive_quota_errors == 0
        assert state.quota_blocked_until == 0.0
        assert routing_ordered.router is not None
        # "ordered" keeps the configured order, so the primary stays the first try.
        assert routing_ordered.router.ordered_regions(MODEL, _ROUTING_REGIONS) == [
            ROUTING_PRIMARY,
            ROUTING_SECONDARY,
        ]

    def test_throttling_fails_over_to_secondary(
        self, routing_ordered: RoutingFixture
    ) -> None:
        """ThrottlingException on the primary causes failover and one base-length quota backoff."""
        before = monotonic()
        with routing_ordered.stub_errors(
            ROUTING_PRIMARY, [("converse", "ThrottlingException")]
        ):
            response = routing_ordered.openai.chat.completions.create(
                model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
            )
        assert response.choices[0].message.content
        primary = routing_ordered.get_state(MODEL, ROUTING_PRIMARY)
        assert not primary.is_usable
        assert primary.consecutive_quota_errors == 1
        # First quota error => backoff is exactly the configured base, not escalated.
        assert before + _QUOTA_BACKOFF_BASE <= primary.quota_blocked_until
        assert primary.quota_blocked_until <= monotonic() + _QUOTA_BACKOFF_BASE
        assert primary.unavailable_until == 0.0
        # The secondary served the request, so mark_success cleared its state.
        secondary = routing_ordered.get_state(MODEL, ROUTING_SECONDARY)
        assert secondary.is_usable
        assert secondary.consecutive_quota_errors == 0

    def test_unavailable_fails_over_to_secondary(
        self, routing_ordered: RoutingFixture
    ) -> None:
        """ServiceUnavailableException applies the fixed unavailability backoff, not a quota backoff."""
        before = monotonic()
        with routing_ordered.stub_errors(
            ROUTING_PRIMARY, [("converse", "ServiceUnavailableException")]
        ):
            response = routing_ordered.openai.chat.completions.create(
                model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
            )
        assert response.choices[0].message.content
        state = routing_ordered.get_state(MODEL, ROUTING_PRIMARY)
        assert not state.is_usable
        assert before + _UNAVAILABLE_BACKOFF <= state.unavailable_until
        assert state.unavailable_until <= monotonic() + _UNAVAILABLE_BACKOFF
        assert state.quota_blocked_until == 0.0
        assert state.consecutive_quota_errors == 0
        assert routing_ordered.get_state(MODEL, ROUTING_SECONDARY).is_usable

    def test_non_retryable_error_raises_immediately(
        self, routing_ordered: RoutingFixture
    ) -> None:
        """ValidationException is not retried across regions and surfaces as a 400.

        Bedrock returns ``ValidationException`` for a malformed request, which is
        outside ``ROUTING_RETRYABLE_CODES``; the gateway re-raises it so the client
        sees the AWS code in ``error.code`` instead of a region failover.
        """
        from openai import BadRequestError  # noqa: PLC0415

        with (
            routing_ordered.stub_errors(
                ROUTING_PRIMARY, [("converse", "ValidationException")]
            ),
            pytest.raises(BadRequestError) as excinfo,
        ):
            routing_ordered.openai.chat.completions.create(
                model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
            )
        assert excinfo.value.status_code == 400
        error = excinfo.value.response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["code"] == "ValidationException"
        # Primary remains usable — validation errors do not trigger backoff.
        assert routing_ordered.get_state(MODEL, ROUTING_PRIMARY).is_usable
        # Only the primary was stubbed: had the router failed over, the live
        # secondary would have answered and no exception would have been raised.
        assert routing_ordered.get_state(MODEL, ROUTING_SECONDARY).is_usable

    def test_both_regions_throttled_raises(
        self, routing_ordered: RoutingFixture
    ) -> None:
        """When every region is throttled the request ends as a 429 carrying ``retry-after``.

        Enough errors are queued to cover all ``aws_bedrock_max_retries + 1``
        attempts, so the terminal error is the router's own ThrottlingException
        rather than an exhausted stub queue. ``retry-after`` reports the smallest
        backoff applied, i.e. the un-escalated base value.
        """
        from openai import RateLimitError  # noqa: PLC0415

        throttles = [("converse", "ThrottlingException")] * 3
        with (
            routing_ordered.stub_errors(ROUTING_PRIMARY, throttles),
            routing_ordered.stub_errors(ROUTING_SECONDARY, throttles),
            pytest.raises(RateLimitError) as excinfo,
        ):
            routing_ordered.openai.chat.completions.create(
                model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
            )
        assert excinfo.value.status_code == 429
        error = excinfo.value.response.json()["error"]
        assert error["type"] == "rate_limit_error"
        assert error["code"] == "ThrottlingException"
        assert excinfo.value.response.headers["retry-after"] == str(_QUOTA_BACKOFF_BASE)
        assert not routing_ordered.get_state(MODEL, ROUTING_PRIMARY).is_usable
        assert not routing_ordered.get_state(MODEL, ROUTING_SECONDARY).is_usable


# ---------------------------------------------------------------------------
# Group 2 — Quota backoff escalation
# ---------------------------------------------------------------------------


class TestQuotaBackoffEscalation:
    """Backoff duration grows exponentially with consecutive quota errors.

    Ref: stdapi/region_routing.py:RegionRouter.mark_error
         stdapi/region_routing.py:RegionRouter.mark_success
    """

    def test_quota_backoff_escalates_on_repeated_errors(
        self, routing_ordered: RoutingFixture
    ) -> None:
        """A second quota error while still blocked doubles the backoff to ``2 * base``."""
        before = monotonic()
        with routing_ordered.stub_errors(
            ROUTING_PRIMARY, [("converse", "ThrottlingException")]
        ):
            routing_ordered.openai.chat.completions.create(
                model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
            )

        state = routing_ordered.get_state(MODEL, ROUTING_PRIMARY)
        blocked_until_1 = state.quota_blocked_until
        assert state.consecutive_quota_errors == 1
        assert (
            before + _QUOTA_BACKOFF_BASE
            <= blocked_until_1
            <= (monotonic() + _QUOTA_BACKOFF_BASE)
        )

        # log_error_details is patched out: mark_error is called here outside the
        # request handler that normally provides the request-log context.
        assert routing_ordered.router is not None
        before_2 = monotonic()
        with patch("stdapi.region_routing.log_error_details"):
            routing_ordered.router.mark_error(
                MODEL, ROUTING_PRIMARY, "ThrottlingException"
            )

        state2 = routing_ordered.get_state(MODEL, ROUTING_PRIMARY)
        assert state2.consecutive_quota_errors == 2
        assert state2.quota_blocked_until > blocked_until_1
        assert before_2 + 2 * _QUOTA_BACKOFF_BASE <= state2.quota_blocked_until
        assert state2.quota_blocked_until <= monotonic() + 2 * _QUOTA_BACKOFF_BASE

    def test_success_resets_quota_state(self, routing_ordered: RoutingFixture) -> None:
        """A successful request after a quota error resets the error counters."""
        with routing_ordered.stub_errors(
            ROUTING_PRIMARY, [("converse", "ThrottlingException")]
        ):
            routing_ordered.openai.chat.completions.create(
                model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
            )
        assert not routing_ordered.get_state(MODEL, ROUTING_PRIMARY).is_usable

        routing_ordered.mark_success(MODEL, ROUTING_PRIMARY)
        assert routing_ordered.get_state(MODEL, ROUTING_PRIMARY).is_usable

        response = routing_ordered.openai.chat.completions.create(
            model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
        )
        assert response.choices[0].message.content
        state = routing_ordered.get_state(MODEL, ROUTING_PRIMARY)
        assert state.consecutive_quota_errors == 0
        assert state.quota_blocked_until == 0.0
        assert state.last_quota_error_time == 0.0


# ---------------------------------------------------------------------------
# Group 3 — Round-robin strategy
# ---------------------------------------------------------------------------


class TestRoundRobinRouting:
    """Round-robin strategy cycles the lead region across successive calls.

    Ref: stdapi/region_routing.py:RegionRouter._round_robin_order
    """

    def test_rotates_lead_region_across_calls(
        self, routing_round_robin: RoutingFixture
    ) -> None:
        """Both regions take the lead position across two consecutive calls.

        The first call seeds the per-model counter with a random index, so only the
        rotation across two calls is deterministic, not which region leads first.
        """
        # SETTINGS.aws_bedrock_region_routing is patched to "round_robin" by the fixture.
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415

        rr = _rr_mod.RegionRouter()
        rr_model = f"{MODEL}.__rr_rotation_test__"
        orderings = [rr.ordered_regions(rr_model, _ROUTING_REGIONS) for _ in range(2)]
        # Each ordering is a rotation of the full list, and the lead alternates.
        assert all(sorted(o) == sorted(_ROUTING_REGIONS) for o in orderings)
        assert orderings[0][0] != orderings[1][0]
        assert {o[0] for o in orderings} == {ROUTING_PRIMARY, ROUTING_SECONDARY}
        # The rotation is counter-driven, so the model now has a counter entry.
        assert rr_model in rr._round_robin_counters  # noqa: SLF001

    def test_blocked_region_is_always_last(
        self, routing_round_robin: RoutingFixture
    ) -> None:
        """A throttled region is deprioritised regardless of round-robin position."""
        # Force the counter to 0 so the primary would normally lead.
        routing_round_robin.set_rr_counter(MODEL, 0)
        with routing_round_robin.stub_errors(
            ROUTING_PRIMARY, [("converse", "ThrottlingException")]
        ):
            response = routing_round_robin.openai.chat.completions.create(
                model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
            )
        assert response.choices[0].message.content
        assert not routing_round_robin.get_state(MODEL, ROUTING_PRIMARY).is_usable
        assert routing_round_robin.router is not None
        # Rotation only applies to usable regions; blocked ones are appended last.
        ordered = routing_round_robin.router.ordered_regions(MODEL, _ROUTING_REGIONS)
        assert ordered == [ROUTING_SECONDARY, ROUTING_PRIMARY]


# ---------------------------------------------------------------------------
# Group 4 — Lowest-latency strategy
# ---------------------------------------------------------------------------


class TestLowestLatencyRouting:
    """Lowest-latency strategy picks the region with the best observed latency.

    Ordering is applied once, at startup, by rewriting ``ORDERED_BEDROCK_REGIONS``;
    the router itself keeps the candidate order untouched (``_identity_order``).

    Ref: stdapi/region_routing.py:measure_region_latencies
         stdapi/region_routing.py:RegionRouter._identity_order
    """

    def test_succeeds_and_uses_a_region(
        self, routing_lowest_latency: RoutingFixture
    ) -> None:
        """Request succeeds and the strategy preserves the candidate order it is given."""
        response = routing_lowest_latency.openai.chat.completions.create(
            model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
        )
        assert response.choices[0].message.content
        assert (
            routing_lowest_latency.get_state(MODEL, ROUTING_PRIMARY).is_usable
            or routing_lowest_latency.get_state(MODEL, ROUTING_SECONDARY).is_usable
        )
        # Unlike round_robin, lowest_latency never rotates: a model with no
        # recorded errors keeps the candidate list exactly as supplied.
        assert routing_lowest_latency.router is not None
        probe_model = f"{MODEL}.__latency_order_probe__"
        assert (
            routing_lowest_latency.router.ordered_regions(probe_model, _ROUTING_REGIONS)
            == _ROUTING_REGIONS
        )

    def test_throttling_fails_over(
        self, routing_lowest_latency: RoutingFixture
    ) -> None:
        """ThrottlingException on the selected region causes failover to the other."""
        with routing_lowest_latency.stub_errors(
            ROUTING_PRIMARY, [("converse", "ThrottlingException")]
        ):
            response = routing_lowest_latency.openai.chat.completions.create(
                model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
            )
        assert response.choices[0].message.content
        primary = routing_lowest_latency.get_state(MODEL, ROUTING_PRIMARY)
        assert not primary.is_usable
        assert primary.consecutive_quota_errors == 1
        # The secondary answered, so mark_success left it unblocked.
        assert routing_lowest_latency.get_state(MODEL, ROUTING_SECONDARY).is_usable


# ---------------------------------------------------------------------------
# Group 5 — Streaming failover
# ---------------------------------------------------------------------------


class TestStreamingFailover:
    """Failover works for streaming responses as well as non-streaming.

    ``ConverseStream`` fails before the first event when the region is throttled, so
    the router can still switch region without a partially emitted response.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
         stdapi/models/__init__.py:route_and_execute
    """

    def test_throttling_failover_with_streaming(
        self, routing_lowest_latency: RoutingFixture
    ) -> None:
        """ThrottlingException during a streaming converse triggers region failover."""
        with (
            routing_lowest_latency.stub_errors(
                ROUTING_PRIMARY, [("converse_stream", "ThrottlingException")]
            ),
            routing_lowest_latency.openai.chat.completions.create(
                model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS, stream=True
            ) as stream,
        ):
            content = "".join(
                chunk.choices[0].delta.content or ""
                for chunk in stream
                if chunk.choices
            )
        assert content
        primary = routing_lowest_latency.get_state(MODEL, ROUTING_PRIMARY)
        assert not primary.is_usable
        assert primary.quota_blocked_until > monotonic()
        assert primary.unavailable_until == 0.0
        # The secondary served the stream, so mark_success left it unblocked.
        secondary = routing_lowest_latency.get_state(MODEL, ROUTING_SECONDARY)
        assert secondary.is_usable
        assert secondary.consecutive_quota_errors == 0


# ---------------------------------------------------------------------------
# Group 6 — Single-region mode
# ---------------------------------------------------------------------------


class TestSingleRegionMode:
    """With a single configured region the router is None and requests still work.

    ``route_and_execute`` short-circuits to ``candidates[0]`` in that case and lets
    botocore's own adaptive retries handle transient errors inside the region.

    Ref: stdapi/region_routing.py:REGION_ROUTER
         stdapi/models/__init__.py:route_and_execute
    """

    def test_chat_succeeds_without_router(
        self, routing_single_region: RoutingFixture
    ) -> None:
        """Non-streaming chat completion works when REGION_ROUTER is None."""
        assert routing_single_region.router is None
        response = routing_single_region.openai.chat.completions.create(
            model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
        )
        assert response.choices[0].message.content

    def test_streaming_succeeds_without_router(
        self, routing_single_region: RoutingFixture
    ) -> None:
        """Streaming chat completion works when REGION_ROUTER is None."""
        assert routing_single_region.router is None
        with routing_single_region.openai.chat.completions.create(
            model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS, stream=True
        ) as stream:
            content = "".join(
                chunk.choices[0].delta.content or ""
                for chunk in stream
                if chunk.choices
            )
        assert content


# ---------------------------------------------------------------------------
# Group 7 — compute_candidate_regions (pure logic, no real AWS)
# ---------------------------------------------------------------------------


def _make_model_details(available_regions: list[str]) -> Any:  # noqa: ANN401
    """Build a minimal ModelDetails instance for use in candidate-region tests."""
    from stdapi.models import ModelDetails  # noqa: PLC0415

    return ModelDetails(
        id=MODEL,
        name=MODEL,
        provider="Amazon",
        input_modalities=["TEXT"],
        output_modalities=["TEXT"],
        regions=available_regions,  # type: ignore[arg-type]
    )


class TestCandidateRegions:
    """``compute_candidate_regions`` S3-locality rules, with model details mocked.

    S3 content blocks are resolved as a terminal operation, so a request carrying
    S3 inputs is pinned to a single region: retrying elsewhere would hand Bedrock a
    cross-region ``s3Location`` it cannot read.

    Ref: stdapi/models/__init__.py:compute_candidate_regions
         stdapi/input_file.py:get_s3_input_regions
    """

    async def test_s3_input_overlap_returns_single_best_region(self) -> None:
        """The single model region holding the most S3 input bytes is returned."""
        from stdapi.models import compute_candidate_regions  # noqa: PLC0415

        model = _make_model_details([ROUTING_PRIMARY, ROUTING_SECONDARY])
        with (
            patch("stdapi.models.get_model_details", new=AsyncMock(return_value=model)),
            patch(
                "stdapi.models.get_s3_input_regions",
                # Secondary holds the file (larger byte count wins)
                return_value={ROUTING_PRIMARY: 100, ROUTING_SECONDARY: 500},
            ),
        ):
            assert await compute_candidate_regions(MODEL) == [ROUTING_SECONDARY]

    async def test_s3_input_no_overlap_falls_back_to_bucketed_region(self) -> None:
        """When the S3 file region doesn't match any model region, fall back to a region with a configured bucket."""
        from stdapi.models import compute_candidate_regions  # noqa: PLC0415

        model = _make_model_details([ROUTING_PRIMARY, ROUTING_SECONDARY])
        with (
            patch("stdapi.models.get_model_details", new=AsyncMock(return_value=model)),
            patch(
                "stdapi.models.get_s3_input_regions", return_value={"eu-west-1": 200}
            ),
            patch(
                "stdapi.models.get_s3_bucket_for_region",
                side_effect=lambda r: "b" if r == ROUTING_PRIMARY else None,
            ),
        ):
            assert await compute_candidate_regions(MODEL) == [ROUTING_PRIMARY]

    async def test_s3_input_no_overlap_no_bucket_raises(self) -> None:
        """A 400 ApiError naming both region sets is raised when neither overlap nor bucket exists."""
        from stdapi.api_errors import ApiError  # noqa: PLC0415
        from stdapi.models import compute_candidate_regions  # noqa: PLC0415

        model = _make_model_details([ROUTING_PRIMARY, ROUTING_SECONDARY])
        with (
            patch("stdapi.models.get_model_details", new=AsyncMock(return_value=model)),
            patch(
                "stdapi.models.get_s3_input_regions", return_value={"eu-west-1": 200}
            ),
            patch("stdapi.models.get_s3_bucket_for_region", return_value=None),
            pytest.raises(ApiError) as excinfo,
        ):
            await compute_candidate_regions(MODEL)

        message = str(excinfo.value)
        assert excinfo.value.status == 400
        assert "S3 input data is located in ['eu-west-1']" in message
        assert MODEL in message
        assert ROUTING_PRIMARY in message
        assert ROUTING_SECONDARY in message

    async def test_s3_required_input_overlap_without_bucket_falls_back_to_bucketed_region(
        self,
    ) -> None:
        """s3_required=True never pins to an overlapping S3 input region lacking a bucket."""
        from stdapi.models import compute_candidate_regions  # noqa: PLC0415

        model = _make_model_details([ROUTING_PRIMARY, ROUTING_SECONDARY])
        with (
            patch("stdapi.models.get_model_details", new=AsyncMock(return_value=model)),
            patch(
                "stdapi.models.get_s3_input_regions",
                # Primary holds the S3 input but has no configured bucket.
                return_value={ROUTING_PRIMARY: 500},
            ),
            patch(
                "stdapi.models.get_s3_bucket_for_region",
                side_effect=lambda r: "b" if r == ROUTING_SECONDARY else None,
            ),
        ):
            assert await compute_candidate_regions(MODEL, s3_required=True) == [
                ROUTING_SECONDARY
            ]

    async def test_s3_required_with_bucket_returns_capable_regions(self) -> None:
        """With s3_required=True, only regions with a configured bucket are returned."""
        from stdapi.models import compute_candidate_regions  # noqa: PLC0415

        model = _make_model_details([ROUTING_PRIMARY, ROUTING_SECONDARY])
        with (
            patch("stdapi.models.get_model_details", new=AsyncMock(return_value=model)),
            patch("stdapi.models.get_s3_input_regions", return_value={}),
            patch(
                "stdapi.models.get_s3_bucket_for_region",
                side_effect=lambda r: "b" if r == ROUTING_SECONDARY else None,
            ),
        ):
            assert await compute_candidate_regions(MODEL, s3_required=True) == [
                ROUTING_SECONDARY
            ]

    async def test_s3_required_no_bucket_raises(self) -> None:
        """A 400 ApiError reporting the missing bucket is raised when s3_required has no candidate."""
        from stdapi.api_errors import ApiError  # noqa: PLC0415
        from stdapi.models import compute_candidate_regions  # noqa: PLC0415

        model = _make_model_details([ROUTING_PRIMARY, ROUTING_SECONDARY])
        with (
            patch("stdapi.models.get_model_details", new=AsyncMock(return_value=model)),
            patch("stdapi.models.get_s3_input_regions", return_value={}),
            patch("stdapi.models.get_s3_bucket_for_region", return_value=None),
            pytest.raises(ApiError) as excinfo,
        ):
            await compute_candidate_regions(MODEL, s3_required=True)

        message = str(excinfo.value)
        assert excinfo.value.status == 400
        assert (
            f"Model '{MODEL}' requires an S3 bucket but none is configured" in message
        )
        assert ROUTING_PRIMARY in message
        assert ROUTING_SECONDARY in message


# ---------------------------------------------------------------------------
# Group 8 — RegionRouter unit tests (pure logic, no real AWS)
# ---------------------------------------------------------------------------


class TestRegionRouterUnit:
    """RegionRouter health bookkeeping and region ordering, without any AWS call.

    Quota errors escalate exponentially from
    ``aws_bedrock_region_routing_quota_backoff_seconds`` and are capped at
    ``_MAX_QUOTA_BACKOFF``; unavailability errors apply a single fixed backoff.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html
         stdapi/region_routing.py:RegionRouter.mark_error
         stdapi/region_routing.py:RegionRouter.ordered_regions
    """

    def _make_router(self, strategy: str = "ordered") -> Any:  # noqa: ANN401
        """Return a fresh RegionRouter with the given strategy patched in."""
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        with patch.object(SETTINGS, "aws_bedrock_region_routing", strategy):
            return _rr_mod.RegionRouter()

    # -- ordered_regions: single-region short-circuit --

    def test_ordered_regions_single_region_returns_as_is(self) -> None:
        """ordered_regions returns the very same list object when only one region is given.

        The short-circuit runs before any health lookup, so identity — not just
        equality — is what distinguishes it from the usable/blocked partition path.
        """
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = self._make_router()
        regions = [ROUTING_PRIMARY]
        assert router.ordered_regions(MODEL, regions) is regions

        # Still returned as-is once the region is blocked: there is no alternative.
        with (
            patch.object(
                SETTINGS,
                "aws_bedrock_region_routing_quota_backoff_seconds",
                _QUOTA_BACKOFF_BASE,
            ),
            patch("stdapi.region_routing.log_error_details"),
        ):
            router.mark_error(MODEL, ROUTING_PRIMARY, "ThrottlingException")
        assert router.ordered_regions(MODEL, regions) is regions

    # -- mark_error: quota escalation while still blocked --

    def test_mark_error_quota_escalates_while_blocked(self) -> None:
        """Two quota errors in a row give backoffs of exactly ``base`` then ``2 * base``."""
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = self._make_router()
        with (
            patch.object(
                SETTINGS,
                "aws_bedrock_region_routing_quota_backoff_seconds",
                _QUOTA_BACKOFF_BASE,
            ),
            patch("stdapi.region_routing.log_error_details"),
        ):
            # First error — region starts unblocked so counter goes to 1.
            before_1 = monotonic()
            router.mark_error(MODEL, ROUTING_PRIMARY, "ThrottlingException")
            state = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
            assert state.consecutive_quota_errors == 1
            blocked_until_1 = state.quota_blocked_until
            assert before_1 + _QUOTA_BACKOFF_BASE <= blocked_until_1
            assert blocked_until_1 <= monotonic() + _QUOTA_BACKOFF_BASE
            assert state.last_quota_error_time >= before_1

            # Second error — region is still blocked (quota_blocked_until > now).
            before_2 = monotonic()
            router.mark_error(MODEL, ROUTING_PRIMARY, "ThrottlingException")
            state2 = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
            assert state2.consecutive_quota_errors == 2
            assert state2.quota_blocked_until > blocked_until_1
            assert before_2 + 2 * _QUOTA_BACKOFF_BASE <= state2.quota_blocked_until
            assert state2.quota_blocked_until <= monotonic() + 2 * _QUOTA_BACKOFF_BASE
            # Quota errors never touch the unavailability window.
            assert state2.unavailable_until == 0.0

    # -- mark_error: else branch (not blocked, not stale) --

    def test_mark_error_quota_increments_when_not_blocked_and_not_stale(self) -> None:
        """A quota error after a recent one keeps escalating even once the backoff expired."""
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = self._make_router()
        state = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001

        # Simulate a recent prior error: set last_quota_error_time to just now
        # and quota_blocked_until to 0 (expired), so the region is usable again
        # but the last error is within the stale threshold.
        state.consecutive_quota_errors = 1
        state.last_quota_error_time = monotonic()  # recent
        state.quota_blocked_until = 0.0  # not currently blocked

        with (
            patch.object(
                SETTINGS,
                "aws_bedrock_region_routing_quota_backoff_seconds",
                _QUOTA_BACKOFF_BASE,
            ),
            patch("stdapi.region_routing.log_error_details"),
        ):
            before = monotonic()
            router.mark_error(MODEL, ROUTING_PRIMARY, "ThrottlingException")

        state_after = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
        assert state_after.consecutive_quota_errors == 2
        # Counter 2 => backoff base * 2**1, applied from the moment of the error.
        assert before + 2 * _QUOTA_BACKOFF_BASE <= state_after.quota_blocked_until
        assert state_after.quota_blocked_until <= monotonic() + 2 * _QUOTA_BACKOFF_BASE

    # -- mark_error: stale-counter reset --

    def test_mark_error_quota_resets_counter_when_stale(self) -> None:
        """A quota error older than the stale threshold restarts the counter and the base backoff."""
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = self._make_router()
        state = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001

        # Simulate a prior error that happened more than _QUOTA_STALE_THRESHOLD seconds ago.
        state.consecutive_quota_errors = 5
        state.last_quota_error_time = monotonic() - (_rr_mod._QUOTA_STALE_THRESHOLD + 1)  # noqa: SLF001
        state.quota_blocked_until = (
            0.0  # not currently blocked (backoff has long expired)
        )

        with (
            patch.object(
                SETTINGS,
                "aws_bedrock_region_routing_quota_backoff_seconds",
                _QUOTA_BACKOFF_BASE,
            ),
            patch("stdapi.region_routing.log_error_details"),
        ):
            before = monotonic()
            router.mark_error(MODEL, ROUTING_PRIMARY, "ThrottlingException")

        state_after = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
        # Counter must be reset to 1, not incremented from 5.
        assert state_after.consecutive_quota_errors == 1
        # ...and the backoff must therefore be the un-escalated base, not base * 2**5.
        assert before + _QUOTA_BACKOFF_BASE <= state_after.quota_blocked_until
        assert state_after.quota_blocked_until <= monotonic() + _QUOTA_BACKOFF_BASE

    # -- mark_error: unavailability error (else branch) --

    def test_mark_error_unavailability_sets_unavailable_until_not_quota(self) -> None:
        """ServiceUnavailableException applies exactly the fixed backoff and no quota penalty.

        AWS documents 503 as service-side capacity, unrelated to account quotas, so
        it must not escalate the exponential quota counter.
        """
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = self._make_router()
        backoff = _UNAVAILABLE_BACKOFF
        before = monotonic()
        with (
            patch.object(
                SETTINGS,
                "aws_bedrock_region_routing_unavailable_backoff_seconds",
                backoff,
            ),
            patch("stdapi.region_routing.log_error_details"),
        ):
            router.mark_error(MODEL, ROUTING_PRIMARY, "ServiceUnavailableException")

        state = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
        assert state.unavailable_until >= before + backoff
        assert state.unavailable_until <= monotonic() + backoff
        assert not state.is_usable
        assert state.quota_blocked_until == 0.0
        assert state.consecutive_quota_errors == 0
        assert state.last_quota_error_time == 0.0

    # -- mark_error: BotocoreConnectionError takes the unavailability path --

    def test_mark_error_connection_error_class_name_takes_unavailability_path(
        self,
    ) -> None:
        """A connection-error class name (not in quota codes) applies the fixed unavailability backoff.

        ``route_and_execute`` labels connection failures with the exception class
        name rather than an AWS code, which must still land on the non-quota branch.
        """
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = self._make_router()
        backoff = _UNAVAILABLE_BACKOFF
        before = monotonic()
        with (
            patch.object(
                SETTINGS,
                "aws_bedrock_region_routing_unavailable_backoff_seconds",
                backoff,
            ),
            patch("stdapi.region_routing.log_error_details"),
        ):
            # Mimic what route_and_execute passes for a BotocoreConnectionError.
            router.mark_error(MODEL, ROUTING_PRIMARY, "ConnectTimeoutError")

        state = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
        assert state.unavailable_until >= before + backoff
        assert state.unavailable_until <= monotonic() + backoff
        assert not state.is_usable
        assert state.quota_blocked_until == 0.0
        assert state.consecutive_quota_errors == 0

    # -- ordered_regions: all regions blocked → fallback list returned --

    def test_ordered_regions_all_blocked_returns_full_list(self) -> None:
        """When every region is blocked, ordered_regions still returns them all, in order.

        Callers iterate the result, so an empty list would turn a transient
        all-regions-throttled state into an IndexError instead of a 429.
        """
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = self._make_router()
        # Block both regions via mark_error.
        with (
            patch.object(
                SETTINGS,
                "aws_bedrock_region_routing_quota_backoff_seconds",
                _QUOTA_BACKOFF_BASE,
            ),
            patch("stdapi.region_routing.log_error_details"),
        ):
            router.mark_error(MODEL, ROUTING_PRIMARY, "ThrottlingException")
            router.mark_error(MODEL, ROUTING_SECONDARY, "ThrottlingException")

        assert not router._index.get(MODEL, ROUTING_PRIMARY).is_usable  # noqa: SLF001
        assert not router._index.get(MODEL, ROUTING_SECONDARY).is_usable  # noqa: SLF001
        result = router.ordered_regions(MODEL, _ROUTING_REGIONS)
        # All regions must appear in the fallback, keeping the "ordered" order.
        assert result == [ROUTING_PRIMARY, ROUTING_SECONDARY]

    # -- mark_error: quota backoff is capped at _MAX_QUOTA_BACKOFF --

    def test_mark_error_quota_backoff_capped_at_max(self) -> None:
        """Quota backoff saturates at ``_MAX_QUOTA_BACKOFF`` instead of doubling unbounded.

        ``base * 2**100`` would otherwise park the region for longer than the process
        lifetime; the counter itself keeps climbing.
        """
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = self._make_router()
        state = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
        # Force a very high consecutive error count so the raw backoff would overflow the cap.
        state.consecutive_quota_errors = 100
        state.quota_blocked_until = monotonic() + 1  # still blocked
        # The escalate-vs-reset choice keys off the age of the last error, so the
        # timestamp must be recent for the counter to be carried forward.
        state.last_quota_error_time = monotonic()

        before = monotonic()
        with (
            patch.object(
                SETTINGS,
                "aws_bedrock_region_routing_quota_backoff_seconds",
                _QUOTA_BACKOFF_BASE,
            ),
            patch("stdapi.region_routing.log_error_details"),
        ):
            router.mark_error(MODEL, ROUTING_PRIMARY, "ThrottlingException")

        cap = _rr_mod._MAX_QUOTA_BACKOFF  # noqa: SLF001
        state_after = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
        assert state_after.consecutive_quota_errors == 101
        # Backoff window is exactly the cap, neither shorter nor longer.
        assert before + cap <= state_after.quota_blocked_until <= monotonic() + cap

    # -- mark_success resets unavailable_until as well --

    def test_mark_success_resets_unavailable_until(self) -> None:
        """mark_success clears unavailable_until in addition to quota state."""
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = self._make_router()
        with (
            patch.object(
                SETTINGS,
                "aws_bedrock_region_routing_unavailable_backoff_seconds",
                _UNAVAILABLE_BACKOFF,
            ),
            patch("stdapi.region_routing.log_error_details"),
        ):
            router.mark_error(MODEL, ROUTING_PRIMARY, "ServiceUnavailableException")

        state = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
        assert state.unavailable_until > monotonic()

        router.mark_success(MODEL, ROUTING_PRIMARY)
        state_after = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
        assert state_after.unavailable_until == 0.0
        assert state_after.quota_blocked_until == 0.0
        assert state_after.consecutive_quota_errors == 0
        assert state_after.last_quota_error_time == 0.0
        assert state_after.is_usable


# ---------------------------------------------------------------------------
# Group 8b — _bedrock_probe_url (pure logic, local botocore endpoint data only)
# ---------------------------------------------------------------------------


class TestBedrockProbeUrl:
    """_bedrock_probe_url: partition-correct hostname resolution, not a hardcoded suffix.

    Resolution goes through botocore's on-disk endpoint data, so no network access
    and no credentials are involved.

    Ref: botocore/data/endpoints.json
         stdapi/region_routing.py:_bedrock_probe_url
    """

    @pytest.mark.parametrize(
        ("region", "url"),
        [
            ("us-east-1", "https://bedrock-runtime.us-east-1.amazonaws.com"),
            # Regression: the aws-eusc and aws-cn partitions must not be probed
            # on the never-resolving .amazonaws.com suffix.
            ("eusc-de-east-1", "https://bedrock-runtime.eusc-de-east-1.amazonaws.eu"),
            ("cn-north-1", "https://bedrock-runtime.cn-north-1.amazonaws.com.cn"),
            # A region with no known endpoint resolves to None instead of raising.
            ("not-a-real-region", None),
        ],
    )
    def test_region_resolves_to_its_own_partition_hostname(
        self, region: RegionName, url: str | None
    ) -> None:
        """Each region is probed on the hostname of its own AWS partition."""
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415

        assert _rr_mod._bedrock_probe_url(region) == url  # noqa: SLF001


# ---------------------------------------------------------------------------
# Group 9 — measure_region_latencies (pure logic, no real network)
# ---------------------------------------------------------------------------


class TestMeasureRegionLatencies:
    """``measure_region_latencies`` startup probing, with all network I/O mocked.

    Three HEAD probes per configured region feed the mean latency; regions whose
    probes all fail are dropped rather than being ranked last.

    Ref: stdapi/region_routing.py:measure_region_latencies
         stdapi/region_routing.py:_single_probe
    """

    async def test_returns_none_when_routing_disabled(self) -> None:
        """Returns None immediately when region routing strategy is not lowest_latency."""
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        with (
            patch.object(_rr_mod, "REGION_ROUTER", _rr_mod.RegionRouter()),
            patch.object(SETTINGS, "aws_bedrock_region_routing", "ordered"),
        ):
            result = await _rr_mod.measure_region_latencies()
        assert result is None

    async def test_returns_none_when_router_is_none(self) -> None:
        """Returns None when REGION_ROUTER is None (routing disabled / single region)."""
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        with (
            patch.object(_rr_mod, "REGION_ROUTER", None),
            patch.object(SETTINGS, "aws_bedrock_region_routing", "lowest_latency"),
        ):
            result = await _rr_mod.measure_region_latencies()
        assert result is None

    async def test_probes_regions_and_sorts_by_latency(self) -> None:
        """Each region is HEAD-probed three times and results come back sorted by latency."""
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        # _single_probe uses `async with session.head(url):`, so head() must return
        # an async context manager directly (not a coroutine). AsyncMock() supports
        # `async with` natively; MagicMock(return_value=AsyncMock()) satisfies this
        # without wrapping the call in a coroutine.
        # __aenter__ must explicitly return mock_session so that `session` inside
        # `async with ClientSession() as session:` is the same object we configured.
        mock_response = AsyncMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.head = MagicMock(return_value=mock_response)

        with (
            patch.object(_rr_mod, "REGION_ROUTER", _rr_mod.RegionRouter()),
            patch.object(SETTINGS, "aws_bedrock_region_routing", "lowest_latency"),
            patch.object(SETTINGS, "aws_bedrock_regions", _ROUTING_REGIONS),
            patch.object(_rr_mod, "ORDERED_BEDROCK_REGIONS", list(_ROUTING_REGIONS)),
            patch.object(_rr_mod, "_REGION_LATENCIES", {}),
            patch("stdapi.region_routing.ClientSession", return_value=mock_session),
        ):
            result = await _rr_mod.measure_region_latencies()
            measured = dict(_rr_mod._REGION_LATENCIES)  # noqa: SLF001

        assert result is not None
        assert set(result.keys()) == set(_ROUTING_REGIONS)
        # 3 probes per region, each on that region's own partition hostname.
        assert mock_session.head.call_count == 3 * len(_ROUTING_REGIONS)
        assert {call.args[0] for call in mock_session.head.call_args_list} == {
            f"https://bedrock-runtime.{region}.amazonaws.com"
            for region in _ROUTING_REGIONS
        }
        for stats in result.values():
            assert isinstance(stats["latency_ms"], float)
            assert isinstance(stats["stddev_ms"], float)
            assert stats["latency_ms"] >= 0.0
            assert stats["stddev_ms"] >= 0.0
        latencies = [stats["latency_ms"] for stats in result.values()]
        assert latencies == sorted(latencies), "results must be lowest-latency first"
        # The shared per-region latency map drives ORDERED_BEDROCK_REGIONS afterwards.
        assert set(measured) == set(_ROUTING_REGIONS)

    async def test_failed_probes_are_excluded_from_results(self) -> None:
        """A region whose three probes all time out is omitted; the reachable one keeps its mean."""
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        # 3 probes per region: primary all fail, secondary=50ms
        probe_values = [float("inf"), float("inf"), float("inf"), 50.0, 50.0, 50.0]

        with (
            patch.object(_rr_mod, "REGION_ROUTER", _rr_mod.RegionRouter()),
            patch.object(SETTINGS, "aws_bedrock_region_routing", "lowest_latency"),
            patch.object(SETTINGS, "aws_bedrock_regions", _ROUTING_REGIONS),
            patch.object(_rr_mod, "ORDERED_BEDROCK_REGIONS", list(_ROUTING_REGIONS)),
            patch.object(_rr_mod, "_REGION_LATENCIES", {}),
            patch.object(
                _rr_mod, "_single_probe", new=AsyncMock(side_effect=probe_values)
            ),
        ):
            result = await _rr_mod.measure_region_latencies()

        assert result is not None
        assert ROUTING_PRIMARY not in result
        # mean/pstdev of three identical 50 ms samples.
        assert result[ROUTING_SECONDARY] == {"latency_ms": 50.0, "stddev_ms": 0.0}


# ---------------------------------------------------------------------------
# Group 10 — route_and_execute unit tests (pure logic, no real AWS)
# ---------------------------------------------------------------------------


class TestRouteAndExecute:
    """route_and_execute failover classification and retry budget, with ``fn`` mocked.

    The loop runs ``aws_bedrock_max_retries + 1`` attempts, marks the region for each
    retryable failure and re-raises the last error once the budget is spent.

    Ref: stdapi/models/__init__.py:route_and_execute
         stdapi/models/__init__.py:_region_failover_label
    """

    async def test_botocore_connection_error_is_reraised_after_retries_exhausted(
        self,
    ) -> None:
        """A connection error is retried once more, then re-raised unchanged."""
        import stdapi.models as _models  # noqa: PLC0415
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        # route_and_execute reads REGION_ROUTER from its own module namespace, so both
        # bindings must point at the test router for the routed path to be exercised.
        router = _rr_mod.RegionRouter()
        calls: list[str] = []

        async def always_fails(region: str) -> None:
            calls.append(region)
            raise BotocoreConnectionError(error=Exception("Connection timed out"))

        with (
            patch.object(_rr_mod, "REGION_ROUTER", router),
            patch.object(_models, "REGION_ROUTER", router),
            patch.object(SETTINGS, "aws_bedrock_region_routing", "ordered"),
            patch.object(SETTINGS, "aws_bedrock_regions", _ROUTING_REGIONS),
            patch.object(SETTINGS, "aws_bedrock_max_retries", 1),
            patch("stdapi.region_routing.log_error_details"),
            pytest.raises(BotocoreConnectionError) as excinfo,
        ):
            await _models.route_and_execute(MODEL, list(_ROUTING_REGIONS), always_fails)

        # max_retries=1 => 2 attempts; the blocked primary hands attempt 2 to the secondary.
        assert calls == [ROUTING_PRIMARY, ROUTING_SECONDARY]
        assert "Connection timed out" in str(excinfo.value)
        # Both regions were penalised on the unavailability branch.
        for region in _ROUTING_REGIONS:
            assert router._index.get(MODEL, region).unavailable_until > monotonic()  # noqa: SLF001

    async def test_botocore_connection_error_calls_mark_error(self) -> None:
        """BotocoreConnectionError is recorded under its exception class name, once per attempt.

        Connection failures carry no AWS error code, so ``_region_failover_label``
        falls back to the class name and the router applies the unavailability backoff.
        """
        import stdapi.models as _models  # noqa: PLC0415
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        # RegionRouter uses __slots__, so we cannot patch instance attributes.
        # Patch the class method instead and track all calls during the test.
        router = _rr_mod.RegionRouter()
        calls: list[str] = []

        async def always_fails(region: str) -> None:
            calls.append(region)
            raise BotocoreConnectionError(error=Exception("Connection timed out"))

        with (
            patch.object(_rr_mod, "REGION_ROUTER", router),
            patch.object(_models, "REGION_ROUTER", router),
            patch.object(SETTINGS, "aws_bedrock_region_routing", "ordered"),
            patch.object(SETTINGS, "aws_bedrock_regions", _ROUTING_REGIONS),
            patch.object(SETTINGS, "aws_bedrock_max_retries", 0),
            patch.object(_rr_mod.RegionRouter, "mark_error") as mock_mark_error,
            pytest.raises(BotocoreConnectionError) as excinfo,
        ):
            await _models.route_and_execute(MODEL, list(_ROUTING_REGIONS), always_fails)

        # max_retries=0 => a single attempt, on the leading region only.
        assert calls == [ROUTING_PRIMARY]
        assert "Connection timed out" in str(excinfo.value)
        mock_mark_error.assert_called_once_with(
            MODEL, _ROUTING_REGIONS[0], "ConnectionError"
        )

    async def test_mantle_error_failover_throttling_retries_next_region(self) -> None:
        """A failover MantleError with status 429 is recorded as ThrottlingException and retried.

        Bedrock Mantle reports HTTP statuses rather than Converse error codes, so the
        router reuses the Converse taxonomy to pick the quota backoff.
        """
        import stdapi.models as _models  # noqa: PLC0415
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.aws_bedrock_mantle import MantleError  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = _rr_mod.RegionRouter()
        original_mark_error = _rr_mod.RegionRouter.mark_error
        calls: list[str] = []

        async def fn(region: str) -> str:
            calls.append(region)
            if len(calls) == 1:
                msg = "throttled"
                raise MantleError(msg, status=429, failover=True)
            return "ok"

        with (
            patch.object(_rr_mod, "REGION_ROUTER", router),
            patch.object(_models, "REGION_ROUTER", router),
            patch.object(SETTINGS, "aws_bedrock_region_routing", "ordered"),
            patch.object(SETTINGS, "aws_bedrock_regions", _ROUTING_REGIONS),
            patch.object(SETTINGS, "aws_bedrock_max_retries", 1),
            patch.object(
                _rr_mod.RegionRouter,
                "mark_error",
                side_effect=lambda model_id, region, code: original_mark_error(
                    router, model_id, region, code
                ),
            ) as mock_mark_error,
        ):
            result = await _models.route_and_execute(MODEL, list(_ROUTING_REGIONS), fn)

        assert result == "ok"
        assert calls == [ROUTING_PRIMARY, ROUTING_SECONDARY]
        mock_mark_error.assert_called_once_with(
            MODEL, _ROUTING_REGIONS[0], "ThrottlingException"
        )
        # 429 must land on the quota branch, and the retry must clear the survivor.
        primary = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
        assert primary.consecutive_quota_errors == 1
        assert primary.quota_blocked_until > monotonic()
        assert primary.unavailable_until == 0.0
        assert router._index.get(MODEL, ROUTING_SECONDARY).is_usable  # noqa: SLF001

    async def test_mantle_error_failover_unavailable_retries_next_region(self) -> None:
        """A failover MantleError with a non-429 status takes the unavailability branch and is retried."""
        import stdapi.models as _models  # noqa: PLC0415
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.aws_bedrock_mantle import MantleError  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = _rr_mod.RegionRouter()
        original_mark_error = _rr_mod.RegionRouter.mark_error
        calls: list[str] = []

        async def fn(region: str) -> str:
            calls.append(region)
            if len(calls) == 1:
                msg = "unavailable"
                raise MantleError(msg, status=503, failover=True)
            return "ok"

        with (
            patch.object(_rr_mod, "REGION_ROUTER", router),
            patch.object(_models, "REGION_ROUTER", router),
            patch.object(SETTINGS, "aws_bedrock_region_routing", "ordered"),
            patch.object(SETTINGS, "aws_bedrock_regions", _ROUTING_REGIONS),
            patch.object(SETTINGS, "aws_bedrock_max_retries", 1),
            patch.object(
                _rr_mod.RegionRouter,
                "mark_error",
                side_effect=lambda model_id, region, code: original_mark_error(
                    router, model_id, region, code
                ),
            ) as mock_mark_error,
        ):
            result = await _models.route_and_execute(MODEL, list(_ROUTING_REGIONS), fn)

        assert result == "ok"
        assert calls == [ROUTING_PRIMARY, ROUTING_SECONDARY]
        mock_mark_error.assert_called_once_with(
            MODEL, _ROUTING_REGIONS[0], "ServiceUnavailableException"
        )
        # 503 must not escalate the quota counter.
        primary = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
        assert primary.unavailable_until > monotonic()
        assert primary.quota_blocked_until == 0.0
        assert primary.consecutive_quota_errors == 0

    async def test_mantle_error_without_failover_is_reraised_immediately(self) -> None:
        """A non-failover MantleError is re-raised as-is, without a second region attempt.

        ``failover=False`` marks a client-side error (here 400) that another region
        would reject identically, so retrying would only waste the retry budget.
        """
        import stdapi.models as _models  # noqa: PLC0415
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.aws_bedrock_mantle import MantleError  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = _rr_mod.RegionRouter()
        calls: list[str] = []

        async def fn(region: str) -> str:
            calls.append(region)
            msg = "bad request"
            raise MantleError(msg, status=400, failover=False)

        with (
            patch.object(_rr_mod, "REGION_ROUTER", router),
            patch.object(_models, "REGION_ROUTER", router),
            patch.object(SETTINGS, "aws_bedrock_region_routing", "ordered"),
            patch.object(SETTINGS, "aws_bedrock_regions", _ROUTING_REGIONS),
            patch.object(SETTINGS, "aws_bedrock_max_retries", 1),
            pytest.raises(MantleError) as exc_info,
        ):
            await _models.route_and_execute(MODEL, list(_ROUTING_REGIONS), fn)

        assert exc_info.value.status == 400
        assert exc_info.value.failover is False
        assert str(exc_info.value) == "bad request"
        assert calls == [ROUTING_PRIMARY]
        # No backoff bookkeeping for a non-retryable error.
        assert router._index.get(MODEL, ROUTING_PRIMARY).is_usable  # noqa: SLF001


# ---------------------------------------------------------------------------
# Group 10b — no-retry client warm-up (pure logic, no real AWS)
# ---------------------------------------------------------------------------


class TestNoRetryClientWarmUp:
    """__aenter__ warms a single-attempt client pool per region-rotated Bedrock service.

    Region failover must not sit through botocore's own adaptive retries first, so
    routed calls use a parallel pool created with ``max_attempts=1``.

    Ref: stdapi/aws.py:AWSConnectionManager
         stdapi/aws_bedrock.py:bedrock_client
    """

    #: Client creation is fully stubbed: exercises the local implementation only.
    pytestmark = pytest.mark.local

    @staticmethod
    def _stub_create_client(
        monkeypatch: pytest.MonkeyPatch,
    ) -> dict[tuple[str, str], list[Any]]:
        """Stub client creation and record the configs passed per (service, region)."""
        import stdapi.aws as _aws_mod  # noqa: PLC0415
        from stdapi.config import AWS_SESSION, SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(_aws_mod, "_CLIENTS", {})
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_enabled", False)
        recorded: dict[tuple[str, str], list[Any]] = {}

        class _FakeClientCM:
            async def __aenter__(self) -> Self:
                return self

            async def __aexit__(self, *_exc: object) -> None:
                return None

        def _fake_create_client(
            service: str,
            *,
            region_name: str,
            config: Any,  # noqa: ANN401
        ) -> _FakeClientCM:
            recorded.setdefault((service, region_name), []).append(config)
            return _FakeClientCM()

        monkeypatch.setattr(AWS_SESSION, "create_client", _fake_create_client)
        return recorded

    @pytest.mark.parametrize("service", ["bedrock-runtime", "bedrock-agent-runtime"])
    async def test_no_retry_pool_built_per_region(
        self, monkeypatch: pytest.MonkeyPatch, service: str
    ) -> None:
        """With routing active, each region gets an extra ``max_attempts=1`` client."""
        import stdapi.aws as _aws_mod  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "aws_bedrock_region_routing", "ordered")
        recorded = self._stub_create_client(monkeypatch)

        manager = _aws_mod.AWSConnectionManager(
            (service, ROUTING_PRIMARY), (service, ROUTING_SECONDARY)
        )
        await manager.__aenter__()
        try:
            pool = _aws_mod._CLIENTS[f"{service}.no-retry"]  # noqa: SLF001
            assert set(pool) == set(_ROUTING_REGIONS)
            for region in _ROUTING_REGIONS:
                base_config, no_retry_config = recorded[(service, region)]
                # Identity check: botocore may rewrite the shared retries dict
                # in place (max_attempts -> total_max_attempts) once a real
                # client is created with it elsewhere in the session.
                assert base_config is _aws_mod.CONFIG
                assert no_retry_config.retries["max_attempts"] == 1
                base_client = _aws_mod._CLIENTS[service][region]  # noqa: SLF001
                assert pool[region] is not base_client
        finally:
            await manager.__aexit__(None, None, None)

    @pytest.mark.parametrize("service", ["bedrock-runtime", "bedrock-agent-runtime"])
    async def test_disabled_routing_builds_no_pool(
        self, monkeypatch: pytest.MonkeyPatch, service: str
    ) -> None:
        """With routing disabled, only the full-retry clients are created."""
        import stdapi.aws as _aws_mod  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        monkeypatch.setattr(SETTINGS, "aws_bedrock_region_routing", "disabled")
        recorded = self._stub_create_client(monkeypatch)

        manager = _aws_mod.AWSConnectionManager(
            (service, ROUTING_PRIMARY), (service, ROUTING_SECONDARY)
        )
        await manager.__aenter__()
        try:
            assert f"{service}.no-retry" not in _aws_mod._CLIENTS  # noqa: SLF001
            for region in _ROUTING_REGIONS:
                assert len(recorded[(service, region)]) == 1
        finally:
            await manager.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Group 11 — S3 integration tests (require aws_s3_bucket to be configured)
# ---------------------------------------------------------------------------


def _require_s3_bucket() -> None:
    """Skip the test if aws_s3_bucket is not configured in SETTINGS."""
    from stdapi.config import SETTINGS  # noqa: PLC0415

    if not SETTINGS.aws_s3_bucket:
        pytest.skip("aws_s3_bucket not configured — skipping S3 integration test")


@pytest.fixture(autouse=True)
def _request_log_context() -> Iterator[None]:
    """Provide the request-log context required by logging outside request scope."""
    from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

    token = REQUEST_LOG.set({"level": "info"})  # type: ignore[typeddict-item]
    yield
    REQUEST_LOG.reset(token)


@dataclass
class S3FileFixture:
    """Test handle for a temporary S3 object created for testing."""

    bucket: str
    key: str
    region: RegionName
    content: bytes
    content_type: str

    @property
    def uri(self) -> str:
        """Return the s3:// URI for the uploaded object."""
        return f"s3://{self.bucket}/{self.key}"


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def s3_file(sample_image_file: bytes) -> AsyncGenerator[S3FileFixture]:
    """Upload *sample_image_file* to S3 and yield an :class:`S3FileFixture`.

    Skipped when ``aws_s3_bucket`` is not configured.  The uploaded object is
    deleted from S3 after the module finishes.

    Creates a fresh S3 client bound to the module event loop (bypassing the
    module-level ``_CLIENTS`` cache that is bound to the app startup loop) and
    injects it into ``_CLIENTS["s3"]`` for the duration of the module so that
    all S3 calls inside the tests resolve correctly.
    """
    _require_s3_bucket()

    import stdapi.aws as _aws_mod  # noqa: PLC0415
    from stdapi.aws_s3 import BUCKET_TO_REGION  # noqa: PLC0415
    from stdapi.config import AWS_SESSION, SETTINGS  # noqa: PLC0415

    bucket: str = SETTINGS.aws_s3_bucket  # type: ignore[assignment]
    content_type = "image/png"
    key = f"tmp/test_region_routing_{__import__('uuid').uuid4().hex}.png"
    region: RegionName = BUCKET_TO_REGION.get(bucket) or SETTINGS.aws_bedrock_regions[0]

    # Create a fresh S3 client on the current (module) event loop.
    exit_stack = AsyncExitStack()
    await exit_stack.__aenter__()
    try:
        s3_client = await exit_stack.enter_async_context(
            AWS_SESSION.create_client("s3", region_name=region)
        )
    except BaseException:
        with suppress(RuntimeError):
            await exit_stack.__aexit__(None, None, None)
        raise

    # Inject it so get_client("s3", region) resolves to this loop-local client.
    # Save only the original per-region entry (not the whole dict reference) to
    # avoid the in-place mutation bug: setdefault() returns the existing dict, so
    # saving it before mutating would still capture the post-mutation state.
    orig_s3_region_client = _aws_mod._CLIENTS.get("s3", {}).get(region)  # noqa: SLF001
    _aws_mod._CLIENTS.setdefault("s3", {})[region] = s3_client  # noqa: SLF001

    try:
        await s3_client.put_object(
            Bucket=bucket, Key=key, Body=sample_image_file, ContentType=content_type
        )

        yield S3FileFixture(
            bucket=bucket,
            key=key,
            region=region,
            content=sample_image_file,
            content_type=content_type,
        )

        # Clean up the S3 object after the module finishes.
        with suppress(Exception):
            await s3_client.delete_object(Bucket=bucket, Key=key)

    finally:
        # Restore the original _CLIENTS["s3"] entry and close the client.
        if orig_s3_region_client is None:
            _aws_mod._CLIENTS.get("s3", {}).pop(region, None)  # noqa: SLF001
        else:
            _aws_mod._CLIENTS.setdefault("s3", {})[region] = orig_s3_region_client  # noqa: SLF001
        with suppress(RuntimeError):
            await exit_stack.__aexit__(None, None, None)


class TestS3InputRegions:
    """``get_s3_input_regions`` reports the region and byte weight of each S3 input.

    The byte totals are what ``compute_candidate_regions`` ranks regions by, so an
    unresolved file must still register its region, with weight 0.

    Ref: stdapi/input_file.py:get_s3_input_regions
         stdapi/models/__init__.py:compute_candidate_regions
    """

    async def test_get_s3_input_regions_returns_region_and_size(
        self, s3_file: S3FileFixture
    ) -> None:
        """A resolved S3 InputFile contributes its region mapped to its exact byte count."""
        _require_s3_bucket()
        from stdapi.input_file import (  # noqa: PLC0415
            _CURRENT_INPUT_FILES,
            InputFile,
            get_s3_input_regions,
            prefetch_all_content_types,
        )

        token = _CURRENT_INPUT_FILES.set([])
        try:
            InputFile(s3_file.uri)
            # Resolve metadata so size is available.
            await prefetch_all_content_types()
            regions = get_s3_input_regions()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert regions == {s3_file.region: len(s3_file.content)}

    async def test_get_s3_input_regions_without_metadata_contributes_zero_size(
        self, s3_file: S3FileFixture
    ) -> None:
        """S3 InputFile before metadata is resolved contributes 0 bytes but registers its region."""
        _require_s3_bucket()
        from stdapi.input_file import (  # noqa: PLC0415
            _CURRENT_INPUT_FILES,
            InputFile,
            get_s3_input_regions,
        )

        token = _CURRENT_INPUT_FILES.set([])
        try:
            InputFile(s3_file.uri)
            regions = get_s3_input_regions()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert regions == {s3_file.region: 0}

    async def test_get_s3_input_regions_empty_when_no_s3_files(self) -> None:
        """get_s3_input_regions returns an empty dict when no S3 InputFiles are in context."""
        from stdapi.input_file import (  # noqa: PLC0415
            _CURRENT_INPUT_FILES,
            get_s3_input_regions,
        )

        token = _CURRENT_INPUT_FILES.set([])
        try:
            regions = get_s3_input_regions()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert regions == {}


class TestS3SourceToS3:
    """``_S3Source.to_s3`` returns the object in place, or copies it across regions.

    Bedrock can only read an ``s3Location`` from its own region, so a cross-region
    input has to be copied before invocation; a same-region input must not be.

    Ref: stdapi/input_file.py:_S3Source.to_s3
    """

    async def test_to_s3_same_region_returns_same_object(
        self, s3_file: S3FileFixture
    ) -> None:
        """to_s3 for an S3 file already in the target region returns the same bucket and key."""
        _require_s3_bucket()
        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(s3_file.uri)
            result = await f.to_s3(s3_file.region)
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert result.bucket == s3_file.bucket
        assert result.key == s3_file.key

    async def test_to_s3_cross_region_copies_object(
        self, s3_file: S3FileFixture
    ) -> None:
        """to_s3 copies the object, byte-for-byte, into the target region's bucket under a new key."""
        _require_s3_bucket()
        import stdapi.aws as _aws_mod  # noqa: PLC0415
        from stdapi.aws_s3 import get_s3_bucket_for_region  # noqa: PLC0415
        from stdapi.cleanup import CLEANUPS  # noqa: PLC0415
        from stdapi.config import AWS_SESSION, SETTINGS  # noqa: PLC0415
        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        # Find a secondary region that has an S3 bucket configured.
        dest_region = next(
            (
                r
                for r in SETTINGS.aws_bedrock_regions
                if r != s3_file.region and get_s3_bucket_for_region(r) is not None
            ),
            None,
        )
        if dest_region is None:
            pytest.skip("No secondary region with an S3 bucket configured")

        dest_bucket = get_s3_bucket_for_region(dest_region)
        assert dest_bucket is not None

        # Create a loop-local S3 client for the dest region so that copy_s3_object
        # and the verification head_object call don't hit a client bound to a
        # different event loop (the app-startup loop).
        exit_stack = AsyncExitStack()
        await exit_stack.__aenter__()
        try:
            dest_s3 = await exit_stack.enter_async_context(
                AWS_SESSION.create_client("s3", region_name=dest_region)
            )
        except BaseException:
            with suppress(RuntimeError):
                await exit_stack.__aexit__(None, None, None)
            raise

        orig_dest_client = _aws_mod._CLIENTS.get("s3", {}).get(dest_region)  # noqa: SLF001
        _aws_mod._CLIENTS.setdefault("s3", {})[dest_region] = dest_s3  # noqa: SLF001

        token = _CURRENT_INPUT_FILES.set([])
        copied_key: str | None = None
        cleanups_token = CLEANUPS.set([])
        try:
            f = InputFile(s3_file.uri)
            result = await f.to_s3(dest_region)
            copied_key = result.key
            # Verify the copy is accessible.
            head = await dest_s3.head_object(Bucket=result.bucket, Key=result.key)
            assert head["ContentLength"] == len(s3_file.content)
        finally:
            CLEANUPS.reset(cleanups_token)
            _CURRENT_INPUT_FILES.reset(token)
            if copied_key:
                with suppress(Exception):
                    await dest_s3.delete_object(Bucket=dest_bucket, Key=copied_key)
            # Restore original _CLIENTS entry and close the dest client.
            if orig_dest_client is None:
                _aws_mod._CLIENTS.get("s3", {}).pop(dest_region, None)  # noqa: SLF001
            else:
                _aws_mod._CLIENTS.setdefault("s3", {})[dest_region] = orig_dest_client  # noqa: SLF001
            with suppress(RuntimeError):
                await exit_stack.__aexit__(None, None, None)

        assert result.bucket == dest_bucket
        assert result.key != s3_file.key


class TestS3ComputeCandidateRegions:
    """``compute_candidate_regions`` region pinning driven by a real S3 object.

    Ref: stdapi/models/__init__.py:compute_candidate_regions
         stdapi/aws_s3.py:get_s3_bucket_for_region
    """

    async def test_s3_input_routes_to_file_region(self, s3_file: S3FileFixture) -> None:
        """compute_candidate_regions pins to the single region where the S3 file lives."""
        _require_s3_bucket()
        from stdapi.input_file import (  # noqa: PLC0415
            _CURRENT_INPUT_FILES,
            InputFile,
            prefetch_all_content_types,
        )
        from stdapi.models import compute_candidate_regions  # noqa: PLC0415

        model = _make_model_details([s3_file.region, ROUTING_SECONDARY])

        token = _CURRENT_INPUT_FILES.set([])
        try:
            InputFile(s3_file.uri)
            await prefetch_all_content_types()
            with patch(
                "stdapi.models.get_model_details", new=AsyncMock(return_value=model)
            ):
                candidates = await compute_candidate_regions(MODEL)
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert candidates == [s3_file.region]

    async def test_s3_input_no_model_overlap_falls_back_to_bucketed_region(
        self, s3_file: S3FileFixture
    ) -> None:
        """When the S3 file region isn't among the model's regions, fall back to a bucket region."""
        _require_s3_bucket()
        from stdapi.aws_s3 import get_s3_bucket_for_region  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415
        from stdapi.input_file import (  # noqa: PLC0415
            _CURRENT_INPUT_FILES,
            InputFile,
            prefetch_all_content_types,
        )
        from stdapi.models import compute_candidate_regions  # noqa: PLC0415

        # Pick a model region that is NOT the s3_file region but has a bucket.
        bucketed_region = next(
            (
                r
                for r in SETTINGS.aws_bedrock_regions
                if r != s3_file.region and get_s3_bucket_for_region(r) is not None
            ),
            None,
        )
        if bucketed_region is None:
            pytest.skip(
                "No secondary region with an S3 bucket — cannot test cross-region fallback"
            )

        # Model is only available in bucketed_region (not where the file lives).
        model = _make_model_details([bucketed_region])

        token = _CURRENT_INPUT_FILES.set([])
        try:
            InputFile(s3_file.uri)
            await prefetch_all_content_types()
            with patch(
                "stdapi.models.get_model_details", new=AsyncMock(return_value=model)
            ):
                candidates = await compute_candidate_regions(MODEL)
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert candidates == [bucketed_region]


class TestResolveBedrockContentBlocksS3:
    """``resolve_all_bedrock_content_blocks`` rewrites pending blocks in place.

    ``to_s3=True`` yields an ``s3Location`` reference (used for async invocation) and
    ``to_s3=False`` inlines the bytes; the two source variants are mutually exclusive
    in the Bedrock ``ImageSource`` union.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
         stdapi/input_file.py:resolve_all_bedrock_content_blocks
    """

    async def test_resolve_to_s3_location_writes_s3_uri(
        self, s3_file: S3FileFixture
    ) -> None:
        """resolve_all_bedrock_content_blocks with to_s3=True replaces the source with an s3Location."""
        _require_s3_bucket()
        from stdapi.input_file import (  # noqa: PLC0415
            _CURRENT_INPUT_FILES,
            InputFile,
            resolve_all_bedrock_content_blocks,
        )

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(s3_file.uri)
            block = await f.to_bedrock_content_block(content_type=s3_file.content_type)
            await resolve_all_bedrock_content_blocks(s3_file.region, to_s3=True)
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        # After resolution, _bedrock_source should be removed and replaced with s3Location.
        assert "image" in block
        source = block["image"]["source"]
        assert source["s3Location"]["uri"].startswith(f"s3://{s3_file.bucket}/")
        # The union is exclusive, and the pending-source handle is consumed.
        assert "bytes" not in source
        assert not hasattr(f, "_bedrock_source")

    async def test_resolve_to_bytes_downloads_content(
        self, s3_file: S3FileFixture
    ) -> None:
        """resolve_all_bedrock_content_blocks with to_s3=False inlines the object's bytes."""
        _require_s3_bucket()
        from stdapi.input_file import (  # noqa: PLC0415
            _CURRENT_INPUT_FILES,
            InputFile,
            resolve_all_bedrock_content_blocks,
        )

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(s3_file.uri)
            block = await f.to_bedrock_content_block(content_type=s3_file.content_type)
            await resolve_all_bedrock_content_blocks(s3_file.region, to_s3=False)
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert "image" in block
        source = block["image"]["source"]
        assert source["bytes"] == s3_file.content
        assert "s3Location" not in source
        assert not hasattr(f, "_bedrock_source")


# ---------------------------------------------------------------------------
# Group 12 — _FileSource base-class methods (get_size, get_filename, is_s3,
#             to_data_uri, to_s3 via put_s3_object) — S3 integration
# ---------------------------------------------------------------------------


class TestFileSourceBaseMethodsS3:
    """_FileSource.get_size / get_filename / is_s3 / to_data_uri / to_s3 over a real S3 object.

    Metadata is lazy: the first accessor triggers a live ``HeadObject`` against the
    uploaded fixture object.

    Ref: stdapi/input_file.py:_S3Source
         stdapi/input_file.py:_FileSource
    """

    async def test_s3source_get_size_via_head_object(
        self, s3_file: S3FileFixture
    ) -> None:
        """_FileSource.get_size resolves metadata and returns the object's exact byte count."""
        _require_s3_bucket()
        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(s3_file.uri)
            size = await f.get_size()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert size == len(s3_file.content)

    async def test_s3source_get_filename_from_key(self, s3_file: S3FileFixture) -> None:
        """_FileSource.get_filename returns the last path component of the S3 key."""
        _require_s3_bucket()
        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(s3_file.uri)
            filename = await f.get_filename()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert filename == s3_file.key.rsplit("/", 1)[-1]

    async def test_s3source_is_s3_true(self, s3_file: S3FileFixture) -> None:
        """InputFile.is_s3 is True for an s3:// URI."""
        _require_s3_bucket()
        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(s3_file.uri)
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert f.is_s3 is True

    async def test_s3source_to_data_uri(self, s3_file: S3FileFixture) -> None:
        """_FileSource.to_data_uri returns a valid data: URI with the correct content type."""
        _require_s3_bucket()
        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(s3_file.uri)
            data_uri = await f.to_data_uri()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert data_uri.startswith(f"data:{s3_file.content_type};base64,")
        # Verify the payload decodes back to the original content.
        import base64  # noqa: PLC0415

        payload = data_uri.split(",", 1)[1]
        assert base64.b64decode(payload) == s3_file.content

    async def test_filesource_to_s3_via_base64_source(
        self, s3_file: S3FileFixture, sample_image_file: bytes
    ) -> None:
        """_FileSource.to_s3 uploads the decoded bytes and returns the resulting S3 object.

        A base64 source has no S3 identity of its own, so the base-class path must
        materialise the payload through ``put_s3_object`` in the target region.
        """
        _require_s3_bucket()
        import base64 as _b64  # noqa: PLC0415

        from stdapi.cleanup import CLEANUPS  # noqa: PLC0415
        from stdapi.config import AWS_SESSION  # noqa: PLC0415
        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        b64_value = _b64.b64encode(sample_image_file).decode()
        token = _CURRENT_INPUT_FILES.set([])
        cleanups_token = CLEANUPS.set([])
        result = None
        async with AWS_SESSION.create_client("s3", region_name=s3_file.region) as s3:
            try:
                f = InputFile(b64_value)
                # Pre-set content_type and inject our loop-local client so
                # put_s3_object resolves it without touching stdapi's _CLIENTS cache.
                f._source._content_type = "image/png"  # noqa: SLF001
                import stdapi.aws as _aws_mod  # noqa: PLC0415

                orig = _aws_mod._CLIENTS.get("s3", {}).get(s3_file.region)  # noqa: SLF001
                _aws_mod._CLIENTS.setdefault("s3", {})[s3_file.region] = s3  # noqa: SLF001
                try:
                    result = await f.to_s3(s3_file.region, bucket=s3_file.bucket)
                    head = await s3.head_object(Bucket=result.bucket, Key=result.key)
                    assert head["ContentLength"] == len(sample_image_file)
                finally:
                    if orig is None:
                        _aws_mod._CLIENTS.get("s3", {}).pop(s3_file.region, None)  # noqa: SLF001
                    else:
                        _aws_mod._CLIENTS.setdefault("s3", {})[s3_file.region] = orig  # noqa: SLF001
            finally:
                CLEANUPS.reset(cleanups_token)
                _CURRENT_INPUT_FILES.reset(token)
                if result is not None:
                    with suppress(Exception):
                        await s3.delete_object(Bucket=result.bucket, Key=result.key)

        assert result is not None
        assert result.bucket == s3_file.bucket
        # A fresh key, not the fixture object's.
        assert result.key
        assert result.key != s3_file.key


# ---------------------------------------------------------------------------
# Group 13 — _HttpSource: _resolve_metadata, _content_type_from_partial,
#             _read, to_s3 — via presigned URL (real S3) + mocked standard URL
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def s3_presigned_url(s3_file: S3FileFixture) -> str:
    """Generate a SigV4 presigned GET URL for the s3_file object."""
    _require_s3_bucket()
    from aiobotocore.config import AioConfig  # noqa: PLC0415

    from stdapi.config import AWS_SESSION  # noqa: PLC0415

    # Resolve the bucket's physical region: the app's BUCKET_TO_REGION model
    # assumes the default bucket lives in the primary Bedrock region, which a
    # SigV4 presign (unlike redirect-following SDK calls) cannot tolerate.
    async with AWS_SESSION.create_client("s3") as probe:
        location = await probe.get_bucket_location(Bucket=s3_file.bucket)
    bucket_region = location.get("LocationConstraint") or "us-east-1"
    # Pin SigV4 explicitly: post-2014 regions reject SigV2 presigns with a 400.
    async with AWS_SESSION.create_client(
        "s3", region_name=bucket_region, config=AioConfig(signature_version="s3v4")
    ) as s3_client:
        url: str = await s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": s3_file.bucket, "Key": s3_file.key},
            ExpiresIn=3600,
        )
    return url


class TestHttpSourceWithPresignedUrl:
    """_HttpSource behavior driven by a presigned S3 URL — real network, no SSRF target.

    Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html
         stdapi/input_file.py:_HttpSource
    """

    async def test_resolve_metadata_sets_content_type_and_size(
        self, s3_presigned_url: str, s3_file: S3FileFixture
    ) -> None:
        """_HttpSource._resolve_metadata reads Content-Type and Content-Length from HEAD."""
        _require_s3_bucket()
        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(s3_presigned_url)
            size = await f.get_size()
            content_type = await f.get_content_type()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert size == len(s3_file.content)
        assert content_type == s3_file.content_type

    async def test_resolve_metadata_sets_filename_from_url_path(
        self, s3_presigned_url: str, s3_file: S3FileFixture
    ) -> None:
        """_HttpSource._resolve_metadata derives filename from the URL path."""
        _require_s3_bucket()
        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(s3_presigned_url)
            filename = await f.get_filename()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        # The filename comes from the URL path's last segment (key's last component).
        assert filename == s3_file.key.rsplit("/", 1)[-1]

    async def test_http_source_read_downloads_content(
        self, s3_presigned_url: str, s3_file: S3FileFixture
    ) -> None:
        """_HttpSource._read downloads and returns the full file content."""
        _require_s3_bucket()
        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(s3_presigned_url)
            content = await f.to_bytes()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert content == s3_file.content

    async def test_http_source_to_s3_uploads_from_stream(
        self, s3_presigned_url: str, s3_file: S3FileFixture
    ) -> None:
        """A presigned URL for an accepted bucket is normalised to s3://, so to_s3 re-uploads nothing.

        The query string is excluded from the parsed key, so the normalised source
        addresses the very object the URL was signed for and the same-region ``to_s3``
        returns it untouched.
        """
        _require_s3_bucket()
        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(s3_presigned_url)
            assert f.is_s3 is True
            result = await f.to_s3(s3_file.region)
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert result.bucket == s3_file.bucket
        assert result.key == s3_file.key

    async def test_http_source_to_s3_via_presigned_url(
        self, s3_presigned_url: str, s3_file: S3FileFixture, sample_image_file: bytes
    ) -> None:
        """_HttpSource.to_s3 streams a real HTTP download directly to S3.

        Patches _ACCEPTED_BUCKETS to empty so the presigned URL is treated as
        a plain HTTP source (not converted to s3://), exercising the real HTTP→S3
        streaming upload path.  Cleanup uses a raw aiobotocore client.
        """
        _require_s3_bucket()
        from unittest.mock import patch  # noqa: PLC0415

        import stdapi.aws as _aws_mod  # noqa: PLC0415
        import stdapi.security as _security_mod  # noqa: PLC0415
        from stdapi.cleanup import CLEANUPS  # noqa: PLC0415
        from stdapi.config import AWS_SESSION  # noqa: PLC0415
        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        # Clear any cached DNS resolver that may be bound to a different event loop;
        # the SSRF connector's resolver lazily recreates one on the current loop.
        _security_mod._RESOLVER_CACHE.clear()  # noqa: SLF001

        result = None
        async with AWS_SESSION.create_client("s3", region_name=s3_file.region) as s3:
            orig = _aws_mod._CLIENTS.get("s3", {}).get(s3_file.region)  # noqa: SLF001
            _aws_mod._CLIENTS.setdefault("s3", {})[s3_file.region] = s3  # noqa: SLF001
            token = _CURRENT_INPUT_FILES.set([])
            cleanups_token = CLEANUPS.set([])
            try:
                with patch("stdapi.input_file._ACCEPTED_BUCKETS", frozenset()):
                    f = InputFile(s3_presigned_url)
                    assert f.is_s3 is False  # confirm HTTP source, not S3
                    result = await f.to_s3(s3_file.region, bucket=s3_file.bucket)
            finally:
                CLEANUPS.reset(cleanups_token)
                _CURRENT_INPUT_FILES.reset(token)
                if orig is None:
                    _aws_mod._CLIENTS.get("s3", {}).pop(s3_file.region, None)  # noqa: SLF001
                else:
                    _aws_mod._CLIENTS.setdefault("s3", {})[s3_file.region] = orig  # noqa: SLF001
                if result is not None:
                    with suppress(Exception):
                        await s3.delete_object(Bucket=result.bucket, Key=result.key)

        assert result is not None
        assert result.bucket == s3_file.bucket
        assert result.key != s3_file.key  # new key, not the original


class TestHttpSourceMocked:
    """_HttpSource fallbacks with aiohttp mocked out — no real network required.

    Ref: stdapi/input_file.py:_HttpSource._content_type_from_partial
    """

    async def test_content_type_from_partial_when_head_has_no_content_type(
        self, sample_image_file: bytes
    ) -> None:
        """A HEAD response without Content-Type falls back to magic detection on a ranged read."""
        from unittest.mock import AsyncMock, MagicMock, patch  # noqa: PLC0415

        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        # HEAD response: no Content-Type, with Content-Length
        mock_head_resp = AsyncMock()
        mock_head_resp.__aenter__ = AsyncMock(return_value=mock_head_resp)
        mock_head_resp.raise_for_status = MagicMock()
        mock_head_resp.headers = {
            "Content-Length": str(len(sample_image_file))
            # No Content-Type — triggers _content_type_from_partial
        }

        # GET response for partial read
        mock_get_resp = AsyncMock()
        mock_get_resp.__aenter__ = AsyncMock(return_value=mock_get_resp)
        mock_get_resp.status = 206
        mock_get_resp.content = AsyncMock()
        mock_get_resp.content.read = AsyncMock(return_value=sample_image_file[:8192])

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.head = MagicMock(return_value=mock_head_resp)
        mock_session.get = MagicMock(return_value=mock_get_resp)

        token = _CURRENT_INPUT_FILES.set([])
        try:
            with (
                patch(
                    "stdapi.input_file.ssrf_safe_connector", return_value=MagicMock()
                ),
                patch("stdapi.input_file.ClientSession", return_value=mock_session),
            ):
                f = InputFile("https://example.com/file.png")
                content_type = await f.get_content_type()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert content_type == "image/png"
        # HEAD alone was not enough, so exactly one follow-up GET was issued.
        assert mock_session.head.call_count == 1
        assert mock_session.get.call_count == 1
        assert mock_session.get.call_args.args == ("https://example.com/file.png",)
        # Only the magic prefix is read, never the whole body.
        mock_get_resp.content.read.assert_awaited_once()

    async def test_http_source_is_s3_false(self) -> None:
        """InputFile from an http:// URL is not an S3 file."""
        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile("https://example.com/file.png")
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert f.is_s3 is False


# ---------------------------------------------------------------------------
# Group 14 — _DataUriSource: to_base64, to_data_uri
# ---------------------------------------------------------------------------


class TestDataUriSource:
    """_DataUriSource keeps the payload encoded: no decode/re-encode round trip.

    Ref: stdapi/input_file.py:_DataUriSource
    """

    async def test_to_base64_returns_payload_without_decoding(
        self, sample_image_file: bytes
    ) -> None:
        """_DataUriSource.to_base64 returns the raw base64 payload from the data URI."""
        import base64  # noqa: PLC0415

        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        b64 = base64.b64encode(sample_image_file).decode()
        data_uri = f"data:image/png;base64,{b64}"

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(data_uri)
            result = await f.to_base64()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert result == b64

    async def test_to_data_uri_returns_original_string(
        self, sample_image_file: bytes
    ) -> None:
        """_DataUriSource.to_data_uri returns the original data URI unchanged."""
        import base64  # noqa: PLC0415

        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        b64 = base64.b64encode(sample_image_file).decode()
        data_uri = f"data:image/png;base64,{b64}"

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(data_uri)
            result = await f.to_data_uri()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert result == data_uri

    async def test_data_uri_get_size(self, sample_pdf_file: bytes) -> None:
        """_DataUriSource.get_size returns the decoded byte count."""
        import base64  # noqa: PLC0415

        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        b64 = base64.b64encode(sample_pdf_file).decode()
        data_uri = f"data:application/pdf;base64,{b64}"

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(data_uri)
            size = await f.get_size()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert size == len(sample_pdf_file)

    async def test_data_uri_get_filename_is_none(
        self, sample_image_file: bytes
    ) -> None:
        """_DataUriSource.get_filename returns None (no filename in data URIs)."""
        import base64  # noqa: PLC0415

        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        b64 = base64.b64encode(sample_image_file).decode()
        data_uri = f"data:image/png;base64,{b64}"

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(data_uri)
            filename = await f.get_filename()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert filename is None


# ---------------------------------------------------------------------------
# Group 15 — _Base64Source: _read ValueError, to_base64, to_data_uri
# ---------------------------------------------------------------------------


class TestBase64Source:
    """_Base64Source terminal methods and its strict-validation failure mode.

    Ref: stdapi/input_file.py:_Base64Source
         stdapi/utils.py:b64decode
    """

    async def test_to_base64_returns_original_string(
        self, sample_image_file: bytes
    ) -> None:
        """_Base64Source.to_base64 returns the raw base64 string passed at construction."""
        import base64  # noqa: PLC0415

        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        b64 = base64.b64encode(sample_image_file).decode()

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(b64)
            result = await f.to_base64()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert result == b64

    async def test_to_data_uri_contains_content_type_and_payload(
        self, sample_image_file: bytes
    ) -> None:
        """_Base64Source.to_data_uri builds a data: URI with the magic-detected content type."""
        import base64  # noqa: PLC0415

        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        b64 = base64.b64encode(sample_image_file).decode()

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(b64)
            result = await f.to_data_uri()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert result.startswith("data:image/png;base64,")
        assert result.endswith(b64)

    async def test_read_value_error_on_invalid_base64(self) -> None:
        """Invalid base64 surfaces as a 400 ApiError quoting the binascii reason.

        ``b64decode(validate=True)`` rejects the out-of-alphabet characters instead of
        silently discarding them, and the resulting ``ValueError`` is re-raised as a
        client error rather than a 500.
        """
        from stdapi.api_errors import ApiError  # noqa: PLC0415
        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        # Use a string that looks like base64 but fails strict validation.
        bad_b64 = "not!valid==base64!!"

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(bad_b64)
            with pytest.raises(ApiError) as excinfo:
                await f.to_bytes()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert excinfo.value.status == 400
        message = str(excinfo.value)
        # Gateway-owned prefix; the tail is binascii's own wording.
        assert message.startswith("Invalid base64 data: ")
        assert "Non-base64 digit found" in message


# ---------------------------------------------------------------------------
# Group 16 — InputFile.__new__ with content_type, _normalize_and_detect_origin
#             S3 HTTP URL match, __get_pydantic_json_schema__, is_s3, get_filename,
#             get_size, to_data_uri
# ---------------------------------------------------------------------------


class TestInputFilePublicApi:
    """InputFile construction, S3 URL normalisation and JSON-schema exposure.

    An HTTP URL that addresses a configured bucket is rewritten to an ``s3://``
    source so Bedrock reads it directly instead of the gateway proxying the bytes.

    Ref: stdapi/input_file.py:InputFile._normalize_and_detect_origin
         stdapi/input_file.py:InputFile.__get_pydantic_json_schema__
    """

    def test_new_with_content_type_sets_source_content_type(self) -> None:
        """InputFile.__new__ with content_type pre-populates _source._content_type."""
        import base64  # noqa: PLC0415

        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        b64 = base64.b64encode(b"fake image bytes").decode()
        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(b64, content_type="image/jpeg")
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert f._source._content_type == "image/jpeg"  # noqa: SLF001

    async def test_normalize_s3_virtual_host_http_url_accepted_bucket(
        self, s3_file: S3FileFixture
    ) -> None:
        """An S3 virtual-hosted HTTP URL for an accepted bucket is normalised to s3:// origin."""
        _require_s3_bucket()
        from stdapi.input_file import (  # noqa: PLC0415
            _CURRENT_INPUT_FILES,
            InputFile,
            _FileOrigin,
        )

        # Build a virtual-hosted style URL for the accepted bucket/key.
        http_url = f"https://{s3_file.bucket}.s3.amazonaws.com/{s3_file.key}"

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(http_url)
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert f._origin == _FileOrigin.S3_URI  # noqa: SLF001
        assert f.is_s3 is True
        # The rewritten source addresses the same bucket and key, query string dropped.
        assert repr(f) == f"s3://{s3_file.bucket}/{s3_file.key}"

    async def test_normalize_s3_path_style_http_url_accepted_bucket(
        self, s3_file: S3FileFixture
    ) -> None:
        """An S3 path-style HTTP URL for an accepted bucket is normalised to s3:// origin."""
        _require_s3_bucket()
        from stdapi.input_file import (  # noqa: PLC0415
            _CURRENT_INPUT_FILES,
            InputFile,
            _FileOrigin,
        )

        http_url = f"https://s3.amazonaws.com/{s3_file.bucket}/{s3_file.key}"

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(http_url)
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert f._origin == _FileOrigin.S3_URI  # noqa: SLF001
        assert f.is_s3 is True
        # The rewritten source addresses the same bucket and key, query string dropped.
        assert repr(f) == f"s3://{s3_file.bucket}/{s3_file.key}"

    def test_normalize_s3_http_url_unknown_bucket_treated_as_http(self) -> None:
        """An S3 HTTP URL for a non-accepted bucket is treated as a plain HTTP URL."""
        from stdapi.input_file import (  # noqa: PLC0415
            _CURRENT_INPUT_FILES,
            InputFile,
            _FileOrigin,
        )

        http_url = "https://unknown-bucket-xyz.s3.amazonaws.com/some/key.png"

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(http_url)
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert f._origin == _FileOrigin.HTTP_URL  # noqa: SLF001
        assert f.is_s3 is False

    def test_get_pydantic_json_schema_with_base64_allowed(self) -> None:
        """__get_pydantic_json_schema__ omits the URL pattern when BASE64 is in ALLOWED_ORIGINS."""
        from pydantic_core import CoreSchema  # noqa: PLC0415

        from stdapi.input_file import InputFile, _FileOrigin  # noqa: PLC0415

        handler = None  # handler is not used in the implementation
        schema = InputFile.__get_pydantic_json_schema__(
            CoreSchema,  # type: ignore[arg-type]
            handler,  # type: ignore[arg-type]
        )
        # InputFile has BASE64 in ALLOWED_ORIGINS → no pattern restriction.
        assert _FileOrigin.BASE64 in InputFile.ALLOWED_ORIGINS
        assert schema == {"type": "string", "minLength": 1}

    def test_get_pydantic_json_schema_url_only_has_pattern(self) -> None:
        """A base64-rejecting subclass advertises a URL-only ``pattern`` in its JSON schema."""
        import re  # noqa: PLC0415

        from pydantic_core import CoreSchema  # noqa: PLC0415

        from stdapi.input_file import InputFileUrl, _FileOrigin  # noqa: PLC0415

        assert _FileOrigin.BASE64 not in InputFileUrl.ALLOWED_ORIGINS
        schema = InputFileUrl.__get_pydantic_json_schema__(
            CoreSchema,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
        )
        assert schema["type"] == "string"
        pattern = schema["pattern"]
        assert pattern.startswith("^(?:https?://")
        # The advertised pattern must accept the URL forms and reject bare base64.
        assert re.match(pattern, "https://example.com/file.png")
        assert re.match(pattern, "s3://bucket/key.png")
        assert not re.match(pattern, "aGVsbG8gd29ybGQ=")

    async def test_inputfile_get_size_via_data_uri(
        self, sample_pdf_file: bytes
    ) -> None:
        """InputFile.get_size resolves size through the data-URI source."""
        import base64  # noqa: PLC0415

        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        b64 = base64.b64encode(sample_pdf_file).decode()
        data_uri = f"data:application/pdf;base64,{b64}"

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(data_uri)
            size = await f.get_size()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert size == len(sample_pdf_file)

    async def test_inputfile_get_filename_via_data_uri(
        self, sample_image_file: bytes
    ) -> None:
        """InputFile.get_filename returns None for a data-URI source."""
        import base64  # noqa: PLC0415

        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        b64 = base64.b64encode(sample_image_file).decode()
        data_uri = f"data:image/png;base64,{b64}"

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(data_uri)
            filename = await f.get_filename()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert filename is None

    async def test_inputfile_to_data_uri_via_base64_source(
        self, sample_image_file: bytes
    ) -> None:
        """InputFile.to_data_uri builds a valid data: URI from a raw base64 source."""
        import base64  # noqa: PLC0415

        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        b64 = base64.b64encode(sample_image_file).decode()

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(b64)
            result = await f.to_data_uri()
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert result.startswith("data:image/png;base64,")
        assert base64.b64decode(result.split(",", 1)[1]) == sample_image_file
