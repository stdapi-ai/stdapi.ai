"""Integration tests for AWS Bedrock region routing strategies.

Uses the session-scoped ``openai_client`` from conftest directly.
Per-strategy fixtures patch SETTINGS in-place, swap the two name-bound
REGION_ROUTER copies, and inject fresh no-retry clients for AioStubber.
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
_QUOTA_BACKOFF_BASE = 60  # seconds — mirrors the fixture override below


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
                SETTINGS, "aws_bedrock_region_routing_unavailable_backoff_seconds", 30
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
    """Ordered strategy always prefers the first usable region in the list."""

    def test_success_uses_primary_region(self, routing_ordered: RoutingFixture) -> None:
        """Successful request goes to the primary region with no quota errors recorded."""
        response = routing_ordered.openai.chat.completions.create(
            model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
        )
        assert response.choices[0].message.content
        assert routing_ordered.get_state(MODEL, ROUTING_PRIMARY).is_usable
        assert (
            routing_ordered.get_state(MODEL, ROUTING_PRIMARY).consecutive_quota_errors
            == 0
        )

    def test_throttling_fails_over_to_secondary(
        self, routing_ordered: RoutingFixture
    ) -> None:
        """ThrottlingException on the primary causes failover; primary is quota-blocked."""
        with routing_ordered.stub_errors(
            ROUTING_PRIMARY, [("converse", "ThrottlingException")]
        ):
            response = routing_ordered.openai.chat.completions.create(
                model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
            )
        assert response.choices[0].message.content
        assert not routing_ordered.get_state(MODEL, ROUTING_PRIMARY).is_usable
        assert (
            +routing_ordered.get_state(MODEL, ROUTING_PRIMARY).quota_blocked_until
            > monotonic()
        )
        assert (
            routing_ordered.get_state(MODEL, ROUTING_SECONDARY).consecutive_quota_errors
            == 0
        )

    def test_unavailable_fails_over_to_secondary(
        self, routing_ordered: RoutingFixture
    ) -> None:
        """ServiceUnavailableException marks primary unavailable, not quota-blocked."""
        with routing_ordered.stub_errors(
            ROUTING_PRIMARY, [("converse", "ServiceUnavailableException")]
        ):
            response = routing_ordered.openai.chat.completions.create(
                model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
            )
        assert response.choices[0].message.content
        state = routing_ordered.get_state(MODEL, ROUTING_PRIMARY)
        assert state.unavailable_until > monotonic()
        assert state.quota_blocked_until == 0.0
        assert state.consecutive_quota_errors == 0

    def test_non_retryable_error_raises_immediately(
        self, routing_ordered: RoutingFixture
    ) -> None:
        """ValidationException is not retried across regions and is re-raised as-is."""
        from openai import APIError  # noqa: PLC0415

        with (
            routing_ordered.stub_errors(
                ROUTING_PRIMARY, [("converse", "ValidationException")]
            ),
            pytest.raises(APIError),
        ):
            routing_ordered.openai.chat.completions.create(
                model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
            )
        # Primary remains usable — validation errors do not trigger backoff
        assert routing_ordered.get_state(MODEL, ROUTING_PRIMARY).is_usable
        assert (
            routing_ordered.get_state(MODEL, ROUTING_SECONDARY).consecutive_quota_errors
            == 0
        )

    def test_both_regions_throttled_raises(
        self, routing_ordered: RoutingFixture
    ) -> None:
        """When all regions are quota-blocked the router raises instead of looping."""
        from openai import APIError  # noqa: PLC0415

        with (
            routing_ordered.stub_errors(
                ROUTING_PRIMARY, [("converse", "ThrottlingException")]
            ),
            routing_ordered.stub_errors(
                ROUTING_SECONDARY, [("converse", "ThrottlingException")]
            ),
            pytest.raises(APIError),
        ):
            routing_ordered.openai.chat.completions.create(
                model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
            )
        assert not routing_ordered.get_state(MODEL, ROUTING_PRIMARY).is_usable
        assert not routing_ordered.get_state(MODEL, ROUTING_SECONDARY).is_usable


# ---------------------------------------------------------------------------
# Group 2 — Quota backoff escalation
# ---------------------------------------------------------------------------


class TestQuotaBackoffEscalation:
    """Backoff duration grows exponentially with consecutive quota errors."""

    def test_quota_backoff_escalates_on_repeated_errors(
        self, routing_ordered: RoutingFixture
    ) -> None:
        """Second quota error produces a longer backoff than the first."""
        with routing_ordered.stub_errors(
            ROUTING_PRIMARY, [("converse", "ThrottlingException")]
        ):
            routing_ordered.openai.chat.completions.create(
                model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
            )

        state = routing_ordered.get_state(MODEL, ROUTING_PRIMARY)
        blocked_until_1 = state.quota_blocked_until
        assert state.consecutive_quota_errors == 1

        # Mutate state directly — router.mark_error() requires a live REQUEST_LOG
        # context that only exists inside a real request handler.
        state.consecutive_quota_errors += 1  # → 2
        state.quota_blocked_until = monotonic() + min(
            _QUOTA_BACKOFF_BASE * (2 ** (state.consecutive_quota_errors - 1)), 3600
        )

        state2 = routing_ordered.get_state(MODEL, ROUTING_PRIMARY)
        assert state2.consecutive_quota_errors == 2
        assert state2.quota_blocked_until > blocked_until_1

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

        response = routing_ordered.openai.chat.completions.create(
            model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
        )
        assert response.choices[0].message.content
        state = routing_ordered.get_state(MODEL, ROUTING_PRIMARY)
        assert state.consecutive_quota_errors == 0
        assert state.quota_blocked_until == 0.0


# ---------------------------------------------------------------------------
# Group 3 — Round-robin strategy
# ---------------------------------------------------------------------------


class TestRoundRobinRouting:
    """Round-robin strategy cycles the lead region across successive calls."""

    def test_rotates_lead_region_across_calls(
        self, routing_round_robin: RoutingFixture
    ) -> None:
        """Both regions take the lead position across two consecutive calls."""
        # SETTINGS.aws_bedrock_region_routing is patched to "round_robin" by the fixture.
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415

        rr = _rr_mod.RegionRouter()
        rr_model = f"{MODEL}.__rr_rotation_test__"
        # Collect which region leads on each of two calls; both must appear.
        regions_used = {
            rr.ordered_regions(rr_model, _ROUTING_REGIONS)[0] for _ in range(2)
        }
        assert ROUTING_PRIMARY in regions_used
        assert ROUTING_SECONDARY in regions_used

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
        assert routing_round_robin.router is not None
        ordered = routing_round_robin.router.ordered_regions(MODEL, _ROUTING_REGIONS)
        assert ordered[0] == ROUTING_SECONDARY
        assert ordered[-1] == ROUTING_PRIMARY


# ---------------------------------------------------------------------------
# Group 4 — Lowest-latency strategy
# ---------------------------------------------------------------------------


class TestLowestLatencyRouting:
    """Lowest-latency strategy picks the region with the best observed latency."""

    def test_succeeds_and_uses_a_region(
        self, routing_lowest_latency: RoutingFixture
    ) -> None:
        """Request succeeds and at least one region remains usable."""
        response = routing_lowest_latency.openai.chat.completions.create(
            model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
        )
        assert response.choices[0].message.content
        assert (
            routing_lowest_latency.get_state(MODEL, ROUTING_PRIMARY).is_usable
            or routing_lowest_latency.get_state(MODEL, ROUTING_SECONDARY).is_usable
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
        assert not routing_lowest_latency.get_state(MODEL, ROUTING_PRIMARY).is_usable


# ---------------------------------------------------------------------------
# Group 5 — Streaming failover
# ---------------------------------------------------------------------------


class TestStreamingFailover:
    """Failover works for streaming responses as well as non-streaming."""

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
        assert not routing_lowest_latency.get_state(MODEL, ROUTING_PRIMARY).is_usable
        assert (
            routing_lowest_latency.get_state(
                MODEL, ROUTING_SECONDARY
            ).consecutive_quota_errors
            == 0
        )


# ---------------------------------------------------------------------------
# Group 6 — Single-region mode
# ---------------------------------------------------------------------------


class TestSingleRegionMode:
    """With a single configured region the router is None and requests still work."""

    def test_chat_succeeds_without_router(
        self, routing_single_region: RoutingFixture
    ) -> None:
        """Non-streaming chat completion works when REGION_ROUTER is None."""
        response = routing_single_region.openai.chat.completions.create(
            model=MODEL, messages=_MESSAGES, max_tokens=_MAX_TOKENS
        )
        assert response.choices[0].message.content

    def test_streaming_succeeds_without_router(
        self, routing_single_region: RoutingFixture
    ) -> None:
        """Streaming chat completion works when REGION_ROUTER is None."""
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
    """Unit tests for ``compute_candidate_regions`` — fully mocked, no real AWS."""

    async def test_s3_input_overlap_returns_single_best_region(self) -> None:
        """When an S3 input file is in one region, only that region is returned."""
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
        """ApiError is raised when the S3 file has no overlap and no bucket is configured."""
        from stdapi.api_errors import ApiError  # noqa: PLC0415
        from stdapi.models import compute_candidate_regions  # noqa: PLC0415

        model = _make_model_details([ROUTING_PRIMARY, ROUTING_SECONDARY])
        with (
            patch("stdapi.models.get_model_details", new=AsyncMock(return_value=model)),
            patch(
                "stdapi.models.get_s3_input_regions", return_value={"eu-west-1": 200}
            ),
            patch("stdapi.models.get_s3_bucket_for_region", return_value=None),
            pytest.raises(ApiError),
        ):
            await compute_candidate_regions(MODEL)

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
        """ApiError is raised when s3_required=True but no region has a bucket."""
        from stdapi.api_errors import ApiError  # noqa: PLC0415
        from stdapi.models import compute_candidate_regions  # noqa: PLC0415

        model = _make_model_details([ROUTING_PRIMARY, ROUTING_SECONDARY])
        with (
            patch("stdapi.models.get_model_details", new=AsyncMock(return_value=model)),
            patch("stdapi.models.get_s3_input_regions", return_value={}),
            patch("stdapi.models.get_s3_bucket_for_region", return_value=None),
            pytest.raises(ApiError),
        ):
            await compute_candidate_regions(MODEL, s3_required=True)


# ---------------------------------------------------------------------------
# Group 8 — RegionRouter unit tests (pure logic, no real AWS)
# ---------------------------------------------------------------------------


class TestRegionRouterUnit:
    """Unit tests for RegionRouter methods — fully mocked, no real AWS calls."""

    def _make_router(self, strategy: str = "ordered") -> Any:  # noqa: ANN401
        """Return a fresh RegionRouter with the given strategy patched in."""
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        with patch.object(SETTINGS, "aws_bedrock_region_routing", strategy):
            return _rr_mod.RegionRouter()

    # -- ordered_regions: single-region short-circuit --

    def test_ordered_regions_single_region_returns_as_is(self) -> None:
        """ordered_regions returns the list unchanged when only one region is given."""
        router = self._make_router()
        result = router.ordered_regions(MODEL, [ROUTING_PRIMARY])
        assert result == [ROUTING_PRIMARY]

    # -- mark_error: quota escalation while still blocked --

    def test_mark_error_quota_escalates_while_blocked(self) -> None:
        """Second quota error while the region is still blocked increments the counter and extends backoff."""
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
            router.mark_error(MODEL, ROUTING_PRIMARY, "ThrottlingException")
            state = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
            assert state.consecutive_quota_errors == 1
            blocked_until_1 = state.quota_blocked_until

            # Second error — region is still blocked (quota_blocked_until > now).
            router.mark_error(MODEL, ROUTING_PRIMARY, "ThrottlingException")
            state2 = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
            assert state2.consecutive_quota_errors == 2
            assert state2.quota_blocked_until > blocked_until_1

    # -- mark_error: else branch (not blocked, not stale) --

    def test_mark_error_quota_increments_when_not_blocked_and_not_stale(self) -> None:
        """Quota error while not currently blocked but with a recent prior error increments the counter."""
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
            router.mark_error(MODEL, ROUTING_PRIMARY, "ThrottlingException")

        state_after = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
        assert state_after.consecutive_quota_errors == 2

    # -- mark_error: stale-counter reset --

    def test_mark_error_quota_resets_counter_when_stale(self) -> None:
        """Quota error whose last occurrence was beyond the stale threshold resets counter to 1."""
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
            router.mark_error(MODEL, ROUTING_PRIMARY, "ThrottlingException")

        state_after = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
        # Counter must be reset to 1, not incremented from 5.
        assert state_after.consecutive_quota_errors == 1

    # -- mark_error: unavailability error (else branch) --

    def test_mark_error_unavailability_sets_unavailable_until_not_quota(self) -> None:
        """Unavailability error sets unavailable_until with the fixed backoff and leaves quota state untouched."""
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = self._make_router()
        backoff = 30
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
        assert state.quota_blocked_until == 0.0
        assert state.consecutive_quota_errors == 0

    # -- mark_error: BotocoreConnectionError takes the unavailability path --

    def test_mark_error_connection_error_class_name_takes_unavailability_path(
        self,
    ) -> None:
        """A connection-error class name (not in quota codes) applies the fixed unavailability backoff."""
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = self._make_router()
        backoff = 30
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
        assert state.quota_blocked_until == 0.0

    # -- ordered_regions: all regions blocked → fallback list returned --

    def test_ordered_regions_all_blocked_returns_full_list(self) -> None:
        """When every region is blocked, ordered_regions returns all regions as a last-resort fallback."""
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

        result = router.ordered_regions(MODEL, _ROUTING_REGIONS)
        # All regions must appear in the fallback — none must be dropped.
        assert set(result) == {ROUTING_PRIMARY, ROUTING_SECONDARY}
        assert len(result) == 2

    # -- mark_error: quota backoff is capped at _MAX_QUOTA_BACKOFF --

    def test_mark_error_quota_backoff_capped_at_max(self) -> None:
        """Quota backoff is capped at _MAX_QUOTA_BACKOFF (3600 s) regardless of error count."""
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = self._make_router()
        state = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
        # Force a very high consecutive error count so the raw backoff would overflow the cap.
        state.consecutive_quota_errors = 100
        state.quota_blocked_until = monotonic() + 1  # still blocked → escalation branch

        with (
            patch.object(
                SETTINGS,
                "aws_bedrock_region_routing_quota_backoff_seconds",
                _QUOTA_BACKOFF_BASE,
            ),
            patch("stdapi.region_routing.log_error_details"),
        ):
            router.mark_error(MODEL, ROUTING_PRIMARY, "ThrottlingException")

        state_after = router._index.get(MODEL, ROUTING_PRIMARY)  # noqa: SLF001
        now = monotonic()
        # Backoff window must not exceed the cap.
        assert state_after.quota_blocked_until <= now + _rr_mod._MAX_QUOTA_BACKOFF + 1  # noqa: SLF001

    # -- mark_success resets unavailable_until as well --

    def test_mark_success_resets_unavailable_until(self) -> None:
        """mark_success clears unavailable_until in addition to quota state."""
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = self._make_router()
        with (
            patch.object(
                SETTINGS, "aws_bedrock_region_routing_unavailable_backoff_seconds", 30
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


# ---------------------------------------------------------------------------
# Group 8b — _bedrock_probe_url (pure logic, local botocore endpoint data only)
# ---------------------------------------------------------------------------


class TestBedrockProbeUrl:
    """_bedrock_probe_url: partition-correct hostname resolution, not a hardcoded suffix."""

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
    """Unit tests for ``measure_region_latencies`` — all network I/O is mocked."""

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
        """Successful probes return a dict keyed by region with latency_ms and stddev_ms floats."""
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

        assert result is not None
        assert set(result.keys()) == set(_ROUTING_REGIONS)
        for stats in result.values():
            assert isinstance(stats["latency_ms"], float)
            assert isinstance(stats["stddev_ms"], float)

    async def test_failed_probes_are_excluded_from_results(self) -> None:
        """Regions whose probes all fail (latency=None) are excluded from the returned dict."""
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
        assert ROUTING_SECONDARY in result


# ---------------------------------------------------------------------------
# Group 10 — route_and_execute unit tests (pure logic, no real AWS)
# ---------------------------------------------------------------------------


class TestRouteAndExecute:
    """Unit tests for route_and_execute — fully mocked, no real AWS calls."""

    async def test_botocore_connection_error_is_reraised_after_retries_exhausted(
        self,
    ) -> None:
        """BotocoreConnectionError exhausts all retries and is re-raised."""
        import stdapi.models as _models  # noqa: PLC0415
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        async def always_fails(_region: str) -> None:
            raise BotocoreConnectionError(error=Exception("Connection timed out"))

        with (
            patch.object(_rr_mod, "REGION_ROUTER", _rr_mod.RegionRouter()),
            patch.object(SETTINGS, "aws_bedrock_region_routing", "ordered"),
            patch.object(SETTINGS, "aws_bedrock_regions", _ROUTING_REGIONS),
            patch.object(SETTINGS, "aws_bedrock_max_retries", 1),
            pytest.raises(BotocoreConnectionError),
        ):
            await _models.route_and_execute(MODEL, list(_ROUTING_REGIONS), always_fails)

    async def test_botocore_connection_error_calls_mark_error(self) -> None:
        """BotocoreConnectionError triggers mark_error with the exception class name."""
        import stdapi.models as _models  # noqa: PLC0415
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        # RegionRouter uses __slots__, so we cannot patch instance attributes.
        # Patch the class method instead and track all calls during the test.
        router = _rr_mod.RegionRouter()

        async def always_fails(_region: str) -> None:
            raise BotocoreConnectionError(error=Exception("Connection timed out"))

        with (
            patch.object(_rr_mod, "REGION_ROUTER", router),
            patch.object(_models, "REGION_ROUTER", router),
            patch.object(SETTINGS, "aws_bedrock_region_routing", "ordered"),
            patch.object(SETTINGS, "aws_bedrock_regions", _ROUTING_REGIONS),
            patch.object(SETTINGS, "aws_bedrock_max_retries", 0),
            patch.object(_rr_mod.RegionRouter, "mark_error") as mock_mark_error,
            pytest.raises(BotocoreConnectionError),
        ):
            await _models.route_and_execute(MODEL, list(_ROUTING_REGIONS), always_fails)

        mock_mark_error.assert_called_once_with(
            MODEL, _ROUTING_REGIONS[0], "ConnectionError"
        )

    async def test_mantle_error_failover_throttling_retries_next_region(self) -> None:
        """A failover MantleError with status 429 is mapped to ThrottlingException and retried."""
        import stdapi.models as _models  # noqa: PLC0415
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.aws_bedrock_mantle import MantleError  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = _rr_mod.RegionRouter()
        original_mark_error = _rr_mod.RegionRouter.mark_error
        calls = 0

        async def fn(_region: str) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
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
        assert calls == 2
        mock_mark_error.assert_called_once_with(
            MODEL, _ROUTING_REGIONS[0], "ThrottlingException"
        )

    async def test_mantle_error_failover_unavailable_retries_next_region(self) -> None:
        """A failover MantleError with status 503 is mapped to ServiceUnavailableException and retried."""
        import stdapi.models as _models  # noqa: PLC0415
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.aws_bedrock_mantle import MantleError  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        router = _rr_mod.RegionRouter()
        original_mark_error = _rr_mod.RegionRouter.mark_error
        calls = 0

        async def fn(_region: str) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
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
        assert calls == 2
        mock_mark_error.assert_called_once_with(
            MODEL, _ROUTING_REGIONS[0], "ServiceUnavailableException"
        )

    async def test_mantle_error_without_failover_is_reraised_immediately(self) -> None:
        """A non-failover MantleError is re-raised without retrying another region."""
        import stdapi.models as _models  # noqa: PLC0415
        import stdapi.region_routing as _rr_mod  # noqa: PLC0415
        from stdapi.aws_bedrock_mantle import MantleError  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        calls = 0

        async def fn(_region: str) -> str:
            nonlocal calls
            calls += 1
            msg = "bad request"
            raise MantleError(msg, status=400, failover=False)

        with (
            patch.object(_rr_mod, "REGION_ROUTER", _rr_mod.RegionRouter()),
            patch.object(SETTINGS, "aws_bedrock_region_routing", "ordered"),
            patch.object(SETTINGS, "aws_bedrock_regions", _ROUTING_REGIONS),
            patch.object(SETTINGS, "aws_bedrock_max_retries", 1),
            pytest.raises(MantleError) as exc_info,
        ):
            await _models.route_and_execute(MODEL, list(_ROUTING_REGIONS), fn)

        assert exc_info.value.status == 400
        assert calls == 1


# ---------------------------------------------------------------------------
# Group 10b — no-retry client warm-up (pure logic, no real AWS)
# ---------------------------------------------------------------------------


class TestNoRetryClientWarmUp:
    """__aenter__ warms a single-attempt client pool per region-rotated Bedrock service."""

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
        """With routing active, each region gets an extra single-attempt client."""
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
    """Tests for ``get_s3_input_regions`` and ``InputFile`` S3 tracking."""

    async def test_get_s3_input_regions_returns_region_and_size(
        self, s3_file: S3FileFixture
    ) -> None:
        """InputFile created from an S3 URI contributes its region and size to get_s3_input_regions."""
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

        assert s3_file.region in regions
        assert regions[s3_file.region] == len(s3_file.content)

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

        assert s3_file.region in regions
        assert regions[s3_file.region] == 0

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
    """Tests for ``_S3Source.to_s3`` — same-region return and cross-region copy."""

    async def test_to_s3_same_region_returns_same_object(
        self, s3_file: S3FileFixture
    ) -> None:
        """to_s3 for an S3 file already in the target region returns a reference to the same object."""
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
        """to_s3 for an S3 file copies it when the target region differs from the source."""
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
    """Integration tests for ``compute_candidate_regions`` with real S3 inputs."""

    async def test_s3_input_routes_to_file_region(self, s3_file: S3FileFixture) -> None:
        """compute_candidate_regions pins to the region where the S3 file lives."""
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
    """Tests for ``resolve_all_bedrock_content_blocks`` using a real S3 file."""

    async def test_resolve_to_s3_location_writes_s3_uri(
        self, s3_file: S3FileFixture
    ) -> None:
        """resolve_bedrock_content_block with to_s3=True writes s3Location into the block."""
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
        assert "s3Location" in source
        assert source["s3Location"]["uri"].startswith("s3://")

    async def test_resolve_to_bytes_downloads_content(
        self, s3_file: S3FileFixture
    ) -> None:
        """resolve_bedrock_content_block with to_s3=False fetches bytes from S3."""
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
        assert "bytes" in source
        assert source["bytes"] == s3_file.content


# ---------------------------------------------------------------------------
# Group 12 — _FileSource base-class methods (get_size, get_filename, is_s3,
#             to_data_uri, to_s3 via put_s3_object) — S3 integration
# ---------------------------------------------------------------------------


class TestFileSourceBaseMethodsS3:
    """Exercise _FileSource.get_size / get_filename / is_s3 / to_data_uri / to_s3.

    Uses a real _S3Source so the metadata path hits the live S3 HeadObject.
    """

    async def test_s3source_get_size_via_head_object(
        self, s3_file: S3FileFixture
    ) -> None:
        """_FileSource.get_size triggers _resolve_metadata and returns correct byte count."""
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
        """_FileSource.to_s3 (base class path) uploads bytes and returns an S3Object."""
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
        assert result.key


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
    """_HttpSource tests using a presigned S3 URL — real network, no SSRF."""

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
        """_HttpSource.to_s3 returns a valid S3Object.

        The presigned URL belongs to an accepted bucket so InputFile normalises it
        to an s3:// source.  to_s3 for a same-region S3 source returns the original
        object without any upload; either way the result must point to a valid bucket+key.
        """
        _require_s3_bucket()
        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(s3_presigned_url)
            result = await f.to_s3(s3_file.region)
        finally:
            _CURRENT_INPUT_FILES.reset(token)

        assert result.bucket
        assert result.key

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
    """_HttpSource tests using aiohttp mocking — no real network required."""

    async def test_content_type_from_partial_when_head_has_no_content_type(
        self, sample_image_file: bytes
    ) -> None:
        """_HttpSource._content_type_from_partial is called when HEAD response has no Content-Type."""
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
    """Tests for _DataUriSource terminal methods."""

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
    """Tests for _Base64Source terminal methods and error handling."""

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
        """_Base64Source._read raises ApiError when the base64 string is invalid."""
        from stdapi.api_errors import ApiError  # noqa: PLC0415
        from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile  # noqa: PLC0415

        # Use a string that looks like base64 but fails strict validation.
        bad_b64 = "not!valid==base64!!"

        token = _CURRENT_INPUT_FILES.set([])
        try:
            f = InputFile(bad_b64)
            with pytest.raises(ApiError):
                await f.to_bytes()
        finally:
            _CURRENT_INPUT_FILES.reset(token)


# ---------------------------------------------------------------------------
# Group 16 — InputFile.__new__ with content_type, _normalize_and_detect_origin
#             S3 HTTP URL match, __get_pydantic_json_schema__, is_s3, get_filename,
#             get_size, to_data_uri
# ---------------------------------------------------------------------------


class TestInputFilePublicApi:
    """Tests for InputFile public API methods and constructor variants."""

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

        from stdapi.input_file import InputFile  # noqa: PLC0415

        handler = None  # handler is not used in the implementation
        schema = InputFile.__get_pydantic_json_schema__(
            CoreSchema,  # type: ignore[arg-type]
            handler,  # type: ignore[arg-type]
        )
        # InputFile has BASE64 in ALLOWED_ORIGINS → no pattern restriction.
        assert schema["type"] == "string"
        assert "pattern" not in schema

    def test_get_pydantic_json_schema_url_only_has_pattern(self) -> None:
        """__get_pydantic_json_schema__ adds URL pattern when BASE64 is not allowed."""
        from pydantic_core import CoreSchema  # noqa: PLC0415

        from stdapi.input_file import InputFileUrl  # noqa: PLC0415

        schema = InputFileUrl.__get_pydantic_json_schema__(
            CoreSchema,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
        )
        assert schema.get("pattern") is not None
        assert schema["pattern"].startswith("^(?:https?://")

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
