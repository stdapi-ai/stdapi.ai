"""Unit tests for the Bedrock model refresh helpers.

Covers `_check_model_availability`'s warning matrix (the AWS
availability API can report ``regionAvailability != AVAILABLE`` for a model
``list_foundation_models`` just advertised — seen live, e.g.
``amazon.titan-embed-g1-text-02`` — which is skipped silently), the
round-based `_check_candidates` resolution, and `_collect_region_candidates`'
per-region fault isolation.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

import stdapi.models
from stdapi import region_routing
from stdapi.config import SETTINGS
from stdapi.models import (
    ModelDetails,
    _check_candidates,
    _check_model_availability,
    _collect_region_candidates,
    _merge_candidate,
    _trigger_price_catalog_refresh,
)
from stdapi.monitoring import REQUEST_LOG, EventLog

if TYPE_CHECKING:
    from collections.abc import Generator

    from pydantic import JsonValue
    from types_aiobotocore_bedrock.literals import RegionName


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


def _make_model(
    model_id: str = "vendor.some-model-v1", region: RegionName = "us-east-1"
) -> ModelDetails:
    """Build a minimal ModelDetails instance with a single region."""
    return ModelDetails(
        id=model_id,
        name=model_id,
        provider="Vendor",
        input_modalities=["TEXT"],
        output_modalities=["TEXT"],
        regions=[region],
    )


class _StubBedrockClient:
    """Stub Bedrock control-plane client returning a fixed availability payload."""

    def __init__(self, availability: dict[str, Any] | Exception) -> None:
        self._availability = availability

    async def get_foundation_model_availability(
        self,
        modelId: str,  # noqa: N803
    ) -> dict[str, Any]:
        """Return the fixed availability payload regardless of the requested model."""
        if isinstance(self._availability, Exception):
            raise self._availability
        return self._availability


def _availability(
    *,
    authorization: str = "AUTHORIZED",
    entitlement: str = "AVAILABLE",
    region: str = "AVAILABLE",
    agreement: str = "AVAILABLE",
) -> dict[str, Any]:
    """Build an availability payload in the shape returned by get_foundation_model_availability."""
    return {
        "authorizationStatus": authorization,
        "entitlementAvailability": entitlement,
        "regionAvailability": region,
        "agreementAvailability": {"status": agreement},
    }


def _stub_client(
    monkeypatch: pytest.MonkeyPatch, availability: dict[str, Any] | Exception
) -> None:
    """Route stdapi.models.get_client to a stub returning *availability*."""
    client = _StubBedrockClient(availability)
    monkeypatch.setattr(
        stdapi.models, "get_client", lambda _service, _region=None: client
    )


def _client_error() -> ClientError:
    """Build a throttling ClientError for the availability API."""
    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
        "GetFoundationModelAvailability",
    )


class TestCheckModelAvailability:
    """_check_model_availability: issue labels per availability payload."""

    async def test_fully_available_model_has_no_issues(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model with no issues at all returns an empty issue list."""
        _stub_client(monkeypatch, _availability())
        assert await _check_model_availability(_make_model()) == []

    async def test_unauthorized_alone_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A denied authorization is reported as ["unauthorized"]."""
        _stub_client(monkeypatch, _availability(authorization="DENIED"))
        assert await _check_model_availability(_make_model()) == ["unauthorized"]

    async def test_unauthorized_and_unavailable_are_combined(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multiple failing statuses are all reported, in a stable order."""
        _stub_client(
            monkeypatch, _availability(authorization="DENIED", region="UNAVAILABLE")
        )
        assert await _check_model_availability(_make_model()) == [
            "unauthorized",
            "unavailable",
        ]

    async def test_aws_error_propagates_to_the_caller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An AWS error during the check is raised; _check_candidates handles it."""
        _stub_client(monkeypatch, _client_error())
        with pytest.raises(ClientError):
            await _check_model_availability(_make_model())


class TestMergeCandidate:
    """_merge_candidate: later regions extend an already-confirmed model."""

    def test_new_region_and_profile_are_appended(self) -> None:
        """A candidate from a new region adds its region and inference profile."""
        existing = _make_model()
        candidate = _make_model(region="eu-west-1")
        candidate.set_inference_profile("eu-west-1", "arn:aws:bedrock:eu-west-1::p/x")

        _merge_candidate(existing, candidate)

        assert existing.regions == ["us-east-1", "eu-west-1"]
        assert (existing.inference_profiles or {})["eu-west-1"] == (
            "arn:aws:bedrock:eu-west-1::p/x"
        )

    def test_duplicate_region_is_ignored(self) -> None:
        """A candidate from an already-known region changes nothing."""
        existing = _make_model()
        _merge_candidate(existing, _make_model())
        assert existing.regions == ["us-east-1"]


class TestCheckCandidates:
    """_check_candidates: round-based resolution across candidate regions."""

    @staticmethod
    def _patch_availability(
        monkeypatch: pytest.MonkeyPatch,
        issues_by_region: dict[str, list[str] | Exception],
    ) -> list[str]:
        """Fake _check_model_availability with per-region issues; returns call log."""
        calls: list[str] = []

        async def _fake(model: ModelDetails) -> list[str]:
            calls.append(f"{model.id}@{model.regions[0]}")
            result = issues_by_region.get(model.regions[0], [])
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(stdapi.models, "_check_model_availability", _fake)
        return calls

    async def test_first_region_success_merges_later_regions_unchecked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model passing in its first region absorbs later regions with one check."""
        calls = self._patch_availability(monkeypatch, {})
        candidates = {"m1": [_make_model("m1"), _make_model("m1", region="eu-west-1")]}
        unavailable: dict[str, dict[str, list[str]]] = {}

        result = await _check_candidates(candidates, unavailable)

        assert result["m1"].regions == ["us-east-1", "eu-west-1"]
        assert calls == ["m1@us-east-1"]
        assert unavailable == {}

    async def test_failure_falls_through_to_the_next_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model unauthorized in region 1 is retried and found in region 2."""
        calls = self._patch_availability(monkeypatch, {"us-east-1": ["unauthorized"]})
        candidates = {"m1": [_make_model("m1"), _make_model("m1", region="eu-west-1")]}
        unavailable: dict[str, dict[str, list[str]]] = {}

        result = await _check_candidates(candidates, unavailable)

        assert result["m1"].regions == ["eu-west-1"]
        assert calls == ["m1@us-east-1", "m1@eu-west-1"]
        assert unavailable == {"m1": {"us-east-1": ["unauthorized"]}}

    async def test_model_unavailable_everywhere_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model failing in every candidate region ends up in unavailable only."""
        self._patch_availability(
            monkeypatch, {"us-east-1": ["unauthorized"], "eu-west-1": ["no_agreement"]}
        )
        candidates = {"m1": [_make_model("m1"), _make_model("m1", region="eu-west-1")]}
        unavailable: dict[str, dict[str, list[str]]] = {}

        result = await _check_candidates(candidates, unavailable)

        assert result == {}
        assert unavailable == {
            "m1": {"us-east-1": ["unauthorized"], "eu-west-1": ["no_agreement"]}
        }

    async def test_region_unavailable_alone_is_skipped_silently(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Issues == ["unavailable"] alone: no unavailable_models entry (AWS quirk)."""
        self._patch_availability(monkeypatch, {"us-east-1": ["unavailable"]})
        unavailable: dict[str, dict[str, list[str]]] = {}

        result = await _check_candidates({"m1": [_make_model("m1")]}, unavailable)

        assert result == {}
        assert unavailable == {}

    async def test_all_models_are_checked_in_one_round(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Models first listed in different regions are all checked in round one."""
        calls = self._patch_availability(monkeypatch, {})
        candidates = {
            "m1": [_make_model("m1")],
            "m2": [_make_model("m2", region="eu-west-1")],
        }

        await _check_candidates(candidates, {})

        assert sorted(calls) == ["m1@us-east-1", "m2@eu-west-1"]

    async def test_all_checks_erroring_raises_the_first_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every availability check errors, the refresh fails fast."""
        self._patch_availability(
            monkeypatch, {"us-east-1": _client_error(), "eu-west-1": _client_error()}
        )
        candidates = {
            "m1": [_make_model("m1")],
            "m2": [_make_model("m2", region="eu-west-1")],
        }

        with pytest.raises(ClientError):
            await _check_candidates(candidates, {})

    async def test_partial_check_errors_degrade_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One erroring check is a per-region issue; the model retries elsewhere."""
        calls = self._patch_availability(monkeypatch, {"us-east-1": _client_error()})
        candidates = {
            "m1": [_make_model("m1"), _make_model("m1", region="eu-west-1")],
            "m2": [_make_model("m2", region="eu-west-1")],
        }
        unavailable: dict[str, dict[str, list[str]]] = {}

        result = await _check_candidates(candidates, unavailable)

        assert result["m1"].regions == ["eu-west-1"]
        assert result["m2"].regions == ["eu-west-1"]
        assert calls == ["m1@us-east-1", "m2@eu-west-1", "m1@eu-west-1"]
        assert unavailable == {
            "m1": {"us-east-1": ["availability check failed: ClientError"]}
        }

    async def test_non_aws_check_errors_propagate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Programming errors in a check are never swallowed as an issue."""
        self._patch_availability(monkeypatch, {"us-east-1": ValueError("bug")})

        with pytest.raises(ValueError, match="bug"):
            await _check_candidates({"m1": [_make_model("m1")]}, {})


def _patch_regions_and_fetch(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, list[ModelDetails] | Exception]
) -> None:
    """Fake the region list and per-region fetch results."""
    monkeypatch.setattr(region_routing, "ORDERED_BEDROCK_REGIONS", list(results))

    async def _fake(region: RegionName) -> list[ModelDetails]:
        result = results[region]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(stdapi.models, "_get_bedrock_models_from_region", _fake)


class TestCollectRegionCandidates:
    """_collect_region_candidates: per-region fault isolation."""

    @staticmethod
    def _aws_error() -> EndpointConnectionError:
        return EndpointConnectionError(endpoint_url="https://bedrock.invalid")

    async def test_one_failed_region_is_tolerated_and_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing region is skipped with a diagnostic; others still merge."""
        _patch_regions_and_fetch(
            monkeypatch,
            {"us-east-1": self._aws_error(), "eu-west-1": [_make_model("m1")]},
        )
        failed: dict[str, str] = {}

        candidates = await _collect_region_candidates(failed)

        assert list(candidates) == ["m1"]
        assert list(failed) == ["us-east-1"]
        assert failed["us-east-1"].startswith("EndpointConnectionError")

    async def test_all_regions_failing_raises_the_first_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every region fails, the refresh fails (startup keeps failing fast)."""
        _patch_regions_and_fetch(
            monkeypatch,
            {"us-east-1": self._aws_error(), "eu-west-1": self._aws_error()},
        )

        with pytest.raises(EndpointConnectionError):
            await _collect_region_candidates({})

    async def test_non_aws_errors_propagate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Programming errors are never swallowed as a region failure."""
        _patch_regions_and_fetch(
            monkeypatch,
            {"us-east-1": ValueError("bug"), "eu-west-1": [_make_model("m1")]},
        )

        with pytest.raises(ValueError, match="bug"):
            await _collect_region_candidates({})

    async def test_candidates_keep_region_priority_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-model candidate lists follow the configured region order."""
        _patch_regions_and_fetch(
            monkeypatch,
            {
                "us-east-1": [_make_model("m1")],
                "eu-west-1": [_make_model("m1", region="eu-west-1")],
            },
        )

        candidates = await _collect_region_candidates({})

        assert [m.regions[0] for m in candidates["m1"]] == ["us-east-1", "eu-west-1"]


def _unreachable_regions(entries: list[JsonValue]) -> dict[str, JsonValue]:
    """Extract the unreachable_bedrock_regions payload from log entries."""
    (payload,) = [
        entry["unreachable_bedrock_regions"]
        for entry in entries
        if isinstance(entry, dict) and "unreachable_bedrock_regions" in entry
    ]
    assert isinstance(payload, dict)
    return payload


@pytest.fixture
def _isolated_model_cache(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Force a stale cache and snapshot/restore the module-level model state."""
    saved = {
        name: dict(getattr(stdapi.models, name))
        for name in (
            "_MODELS",
            "_ALL_MODELS",
            "_MODELS_INPUT_MODALITY",
            "_MODELS_OUTPUT_MODALITY",
            "_ALL_MODELS_INPUT_MODALITY",
            "_ALL_MODELS_OUTPUT_MODALITY",
            "MODEL_ALIASES",
        )
    }
    monkeypatch.setitem(stdapi.models._CACHE, "update_next", None)  # noqa: SLF001
    yield
    for name, content in saved.items():
        target = getattr(stdapi.models, name)
        target.clear()
        target.update(content)


@pytest.mark.usefixtures("_isolated_model_cache")
class TestInitializeBedrockModelsFaultIsolation:
    """initialize_bedrock_models: end-to-end behavior with a failing region."""

    @staticmethod
    def _patch_fetches(monkeypatch: pytest.MonkeyPatch) -> None:
        """One healthy region, one failing region, all checks passing."""
        results: dict[str, list[ModelDetails] | Exception] = {
            "us-east-1": EndpointConnectionError(endpoint_url="https://bad.invalid"),
            "eu-west-1": [_make_model("m1", region="eu-west-1")],
        }
        _patch_regions_and_fetch(monkeypatch, results)

        async def _available(_model: ModelDetails) -> list[str]:
            return []

        monkeypatch.setattr(stdapi.models, "_check_model_availability", _available)

    @staticmethod
    def _start_event() -> EventLog:
        """Build a minimal startup event log."""
        return EventLog(
            type="start",
            level="info",
            date=datetime.now(UTC),
            server_id="test",
            server_version="0.0.0",
        )

    async def test_startup_fails_when_every_region_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup fails fast when no region listing succeeds."""
        _patch_regions_and_fetch(
            monkeypatch,
            {
                "us-east-1": EndpointConnectionError(endpoint_url="https://a.invalid"),
                "eu-west-1": EndpointConnectionError(endpoint_url="https://b.invalid"),
            },
        )

        with pytest.raises(EndpointConnectionError):
            await stdapi.models.initialize_bedrock_models(self._start_event())

    async def test_startup_fails_when_every_availability_check_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup fails fast when every availability check raises an AWS error."""
        _patch_regions_and_fetch(
            monkeypatch, {"us-east-1": [_make_model("m1")], "eu-west-1": []}
        )

        async def _denied(_model: ModelDetails) -> list[str]:
            raise _client_error()

        monkeypatch.setattr(stdapi.models, "_check_model_availability", _denied)

        with pytest.raises(ClientError):
            await stdapi.models.initialize_bedrock_models(self._start_event())

    async def test_startup_survives_one_region_and_warns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup keeps the healthy region's models and warns about the other."""
        self._patch_fetches(monkeypatch)
        start_event = EventLog(
            type="start",
            level="info",
            date=datetime.now(UTC),
            server_id="test",
            server_version="0.0.0",
        )

        assert await stdapi.models.initialize_bedrock_models(start_event) is True

        assert "m1" in stdapi.models._MODELS  # noqa: SLF001
        assert "us-east-1" in _unreachable_regions(start_event["server_warnings"])
        # The TTL was armed: the failed region is retried on the next refresh.
        assert stdapi.models._CACHE["update_next"] is not None  # noqa: SLF001

    async def test_lazy_refresh_warns_in_the_request_log(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lazy refresh surfaces the unreachable region on the request log."""
        self._patch_fetches(monkeypatch)
        # Pin the price-catalog refresh guard so this unit test never hits
        # the live AWS Pricing API when cost tracking is enabled.
        monkeypatch.setattr(SETTINGS, "cost_tracking", False)
        request_log = EventLog(
            type="request",
            level="info",
            date=datetime.now(UTC),
            server_id="test",
            server_version="0.0.0",
        )
        token = REQUEST_LOG.set(request_log)
        try:
            await stdapi.models.initialize_bedrock_models()
        finally:
            REQUEST_LOG.reset(token)

        assert request_log["level"] == "warning"
        assert "us-east-1" in _unreachable_regions(request_log["error_detail"])


class _StubModelListClient:
    """Stub bedrock client returning pre-defined foundation model summaries."""

    def __init__(self, summaries: list[dict[str, Any]]) -> None:
        self._summaries = summaries

    async def list_foundation_models(self) -> dict[str, Any]:
        """Return the pre-defined model summaries."""
        return {"modelSummaries": self._summaries}


def _summary(
    model_id: str,
    *,
    status: str = "ACTIVE",
    legacy_time: datetime | None = None,
    eol_time: datetime | None = None,
) -> dict[str, Any]:
    """Build a minimal foundation model summary with the given lifecycle."""
    lifecycle: dict[str, Any] = {"status": status}
    if legacy_time is not None:
        lifecycle["legacyTime"] = legacy_time
    if eol_time is not None:
        lifecycle["endOfLifeTime"] = eol_time
    return {
        "modelId": model_id,
        "modelName": model_id,
        "providerName": "Vendor",
        "inputModalities": ["TEXT"],
        "outputModalities": ["TEXT"],
        "modelLifecycle": lifecycle,
        "inferenceTypesSupported": ["ON_DEMAND"],
    }


class TestBedrockModelLifecycleFilter:
    """_get_bedrock_models_from_region: lifecycle filtering and legacy flag."""

    #: A legacy time already in the past (like Amazon Nova Reel since 2026-03).
    _PAST = datetime(2020, 1, 1, tzinfo=UTC)

    #: A lifecycle transition far in the future.
    _FUTURE = datetime(2100, 1, 1, tzinfo=UTC)

    @staticmethod
    async def _fetch(
        monkeypatch: pytest.MonkeyPatch,
        summaries: list[dict[str, Any]],
        *,
        legacy: bool,
    ) -> list[ModelDetails]:
        """Run the region fetch against stubbed AWS clients."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_legacy", legacy)
        client = _StubModelListClient(summaries)
        monkeypatch.setattr(
            stdapi.models, "get_client", lambda _service, _region: client
        )

        async def _empty_provisioned(_client: object) -> set[str]:
            return set()

        async def _empty_profiles(_client: object) -> dict[str, str]:
            return {}

        monkeypatch.setattr(
            stdapi.models, "_get_provisioned_models", _empty_provisioned
        )
        monkeypatch.setattr(stdapi.models, "_get_inference_profiles", _empty_profiles)
        return await stdapi.models._get_bedrock_models_from_region("us-east-1")  # noqa: SLF001

    async def test_legacy_models_hidden_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the legacy flag, LEGACY and past-legacy-time models are hidden."""
        models = await self._fetch(
            monkeypatch,
            [
                _summary("vendor.active"),
                _summary("vendor.legacy", status="LEGACY", legacy_time=self._PAST),
                _summary("vendor.future-legacy", legacy_time=self._FUTURE),
            ],
            legacy=False,
        )
        assert [model.id for model in models] == [
            "vendor.active",
            "vendor.future-legacy",
        ]
        assert all(model.legacy is None for model in models)

    async def test_legacy_flag_exposes_past_legacy_time_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the legacy flag, models past their legacy time stay usable."""
        models = await self._fetch(
            monkeypatch,
            [
                _summary("vendor.active"),
                _summary("vendor.legacy", status="LEGACY", legacy_time=self._PAST),
            ],
            legacy=True,
        )
        assert [model.id for model in models] == ["vendor.active", "vendor.legacy"]
        assert models[1].legacy is True

    async def test_end_of_life_models_always_hidden(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Models past their end-of-life time are hidden even with the flag."""
        models = await self._fetch(
            monkeypatch,
            [_summary("vendor.eol", status="LEGACY", eol_time=self._PAST)],
            legacy=True,
        )
        assert models == []


class TestTriggerPriceCatalogRefresh:
    """_trigger_price_catalog_refresh: a Pricing API failure is warned, not raised."""

    async def test_client_error_is_warned_and_does_not_propagate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ClientError from the catalog reload is recorded as a warning on the request log."""

        async def _fail(_model_ids: set[str]) -> None:
            error = ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
                "GetProducts",
            )
            raise error

        monkeypatch.setattr(
            stdapi.models, "refresh_price_catalog_for_new_models", _fail
        )
        request_log = EventLog(
            type="request",
            level="info",
            date=datetime.now(UTC),
            server_id="test",
            server_version="0.0.0",
        )
        token = REQUEST_LOG.set(request_log)
        try:
            # Does not raise: model listing must still succeed.
            await _trigger_price_catalog_refresh(None, {"vendor.new-model"})
        finally:
            REQUEST_LOG.reset(token)

        assert request_log["level"] == "warning"
        assert any(
            "Price-catalog refresh" in str(detail)
            for detail in request_log["error_detail"]
        )
