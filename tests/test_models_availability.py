"""Bedrock model-catalog refresh: discovery, availability resolution and fault isolation.

Model discovery fans out over every configured region and then confirms each model
with ``GetFoundationModelAvailability``, because ``ListFoundationModels`` advertises
models the account cannot actually invoke (unauthorized, unentitled, no Marketplace
agreement) and even models the region reports as unavailable — seen live for
``amazon.titan-embed-g1-text-02``. The refresh therefore has to tolerate a single
degraded region while still failing fast when nothing at all is reachable.

Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_GetFoundationModelAvailability.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html
     stdapi/models/__init__.py:initialize_bedrock_models
"""

from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

import stdapi.models
from stdapi import region_routing
from stdapi.api_errors import DENIED_CALL_KEY
from stdapi.config import SETTINGS
from stdapi.models import (
    ModelBase,
    ModelDetails,
    _check_candidates,
    _check_model_availability,
    _collect_region_candidates,
    _filter_inference_profiles,
    _merge_candidate,
    _request_uses_system_tool,
    _trigger_price_catalog_refresh,
)
from stdapi.monitoring import REQUEST_ID
from tests._helpers import make_client_error, make_event_log, make_model_details

if TYPE_CHECKING:
    from collections.abc import Generator

    from pydantic import JsonValue
    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef
    from stdapi.monitoring import EventLog


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


def _make_model(
    model_id: str = "vendor.some-model-v1", region: RegionName = "us-east-1"
) -> ModelDetails:
    """Build a minimal ModelDetails instance with a single region."""
    return make_model_details(model_id, regions=[region])


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


class TestCheckModelAvailability:
    """_check_model_availability: map an availability payload to operator-facing issue labels.

    The four payload fields are compared against their expected value, so any value
    other than ``AUTHORIZED`` / ``AVAILABLE`` yields the corresponding label.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_GetFoundationModelAvailability.html
         stdapi/models/__init__.py:_check_model_availability
    """

    async def test_fully_available_model_has_no_issues(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An authorized, entitled, agreed and region-available model reports no issue."""
        _stub_client(monkeypatch, _availability())
        assert await _check_model_availability(_make_model()) == []

    async def test_unauthorized_alone_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-``AUTHORIZED`` authorization status yields exactly ``["unauthorized"]``.

        The other three fields stay available, so the label set must not be widened to
        them: each field maps to its own label independently.
        """
        _stub_client(monkeypatch, _availability(authorization="DENIED"))
        assert await _check_model_availability(_make_model()) == ["unauthorized"]

    async def test_unauthorized_and_unavailable_are_combined(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two failing fields both report, in the fixed field order of the mapping table.

        The order is part of the contract: these labels are logged and compared as a
        list, so a set-like ordering would make the warnings unstable.
        """
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
        """An AWS error is propagated unchanged rather than becoming an issue label.

        ``_check_candidates`` is the layer that decides whether a failed check degrades
        one region or fails the whole refresh, so this function must not swallow it.
        """
        _stub_client(
            monkeypatch,
            make_client_error(
                "ThrottlingException",
                "GetFoundationModelAvailability",
                message="slow down",
            ),
        )
        with pytest.raises(ClientError) as excinfo:
            await _check_model_availability(_make_model())
        assert excinfo.value.response["Error"]["Code"] == "ThrottlingException"
        assert excinfo.value.operation_name == "GetFoundationModelAvailability"


class TestMergeCandidate:
    """_merge_candidate: fold a later region's listing into an already-confirmed model.

    One ``ModelDetails`` per model carries every region it can be served from, so the
    per-region listings discovered in parallel are merged rather than kept separate.

    Ref: stdapi/models/__init__.py:_merge_candidate
    """

    def test_new_region_and_profile_are_appended(self) -> None:
        """A candidate from a new region contributes both its region and its profile.

        The profile has to travel with the region: it is what ``get_id`` returns for
        that region, and a region without one would resolve to the bare model ID.
        """
        existing = _make_model()
        candidate = _make_model(region="eu-west-1")
        candidate.set_inference_profile("eu-west-1", "arn:aws:bedrock:eu-west-1::p/x")

        _merge_candidate(existing, candidate)

        assert existing.regions == ["us-east-1", "eu-west-1"]
        assert existing.inference_profiles == {
            "eu-west-1": "arn:aws:bedrock:eu-west-1::p/x"
        }
        assert existing.get_id("eu-west-1", inference_profile=True) == (
            "arn:aws:bedrock:eu-west-1::p/x"
        )

    def test_duplicate_region_is_ignored(self) -> None:
        """Re-merging an already-known region leaves the region list unchanged.

        Idempotence matters because the same region can be merged twice when a model
        resolves in a later round after an earlier region degraded.
        """
        existing = _make_model()
        _merge_candidate(existing, _make_model())
        assert existing.regions == ["us-east-1"]
        assert existing.inference_profiles is None

    def test_regional_profile_is_appended_alongside_the_preferred_one(self) -> None:
        """A candidate's geo-scoped profile is merged into ``inference_profiles_regional``.

        This is what lets a system-tool request avoid the ``global.`` profile for a
        region reached through a later merge, not just the first one (issue #92).
        """
        existing = _make_model()
        candidate = _make_model(region="eu-west-1")
        candidate.set_inference_profile("eu-west-1", "global.vendor.some-model-v1")
        candidate.set_inference_profile_regional("eu-west-1", "eu.vendor.some-model-v1")

        _merge_candidate(existing, candidate)

        assert existing.inference_profiles_regional == {
            "eu-west-1": "eu.vendor.some-model-v1"
        }
        assert (
            existing.get_id("eu-west-1", inference_profile=True, prefer_regional=True)
            == "eu.vendor.some-model-v1"
        )


#: Application inference profile ARN an operator maps a model onto.
_APP_PROFILE_ARN = (
    "arn:aws:bedrock:eu-west-3:123456789012:application-inference-profile/abc123"
)

#: Prompt router ARN an operator maps a model onto.
_PROMPT_ROUTER_ARN = "arn:aws:bedrock:us-east-1:123456789012:prompt-router/my-router"


class TestApplyUserProfiles:
    """_apply_user_profiles: AWS_BEDROCK_MODEL_ARN_MAPPING becomes the invoked model ID.

    The setting is how an operator routes a catalogue model through their own
    application inference profile (for cost allocation tags) or prompt router. The
    mapping is only worth anything if the ARN replaces the model ID Bedrock is
    actually called with, and if the ARN's own region joins the model's region list
    -- otherwise the profile is stored somewhere nothing reads, or is stored for a
    region the router will never route to.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-use.html
         stdapi/config.py:_Settings.aws_bedrock_model_arn_mapping
         stdapi/models/__init__.py:_apply_user_profiles
    """

    @pytest.mark.parametrize(
        ("arn", "region"),
        [
            pytest.param(_APP_PROFILE_ARN, "eu-west-3", id="application-profile"),
            pytest.param(_PROMPT_ROUTER_ARN, "us-east-1", id="prompt-router"),
        ],
    )
    def test_mapped_arn_becomes_the_id_invoked_in_its_region(
        self, monkeypatch: pytest.MonkeyPatch, arn: str, region: RegionName
    ) -> None:
        """Both accepted ARN shapes are installed as the model's profile for their region.

        ``get_id(..., inference_profile=True)`` is what the Converse/Invoke callers
        pass as ``modelId``, so asserting on it pins the mapping end to end rather
        than just its storage.
        """
        model = _make_model(region="us-west-2")
        monkeypatch.setattr(SETTINGS, "aws_bedrock_model_arn_mapping", {model.id: arn})

        assert stdapi.models._apply_user_profiles({model.id: model}) == {}  # noqa: SLF001

        assert model.regions == ["us-west-2", region], (
            "the ARN's region must be added, and the discovered regions kept"
        )
        assert model.get_id(region, inference_profile=True) == arn
        assert model.get_id(region) == model.id, (
            "the bare model ID is still what a non-profile call uses"
        )

    def test_arn_region_already_known_is_not_duplicated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mapping an ARN in a region the model already serves leaves the list unchanged.

        A duplicated region would be tried twice by the region-routing loop.
        """
        model = _make_model(region="eu-west-3")
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_model_arn_mapping", {model.id: _APP_PROFILE_ARN}
        )

        stdapi.models._apply_user_profiles({model.id: model})  # noqa: SLF001

        assert model.regions == ["eu-west-3"]
        assert model.get_id("eu-west-3", inference_profile=True) == _APP_PROFILE_ARN

    def test_mapping_for_an_absent_model_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mapping naming a model the account cannot see degrades to a startup warning.

        A typo or a model lost to a region change must not abort the catalogue
        refresh: the entry is returned for the startup event log instead.

        Ref: stdapi/models/__init__.py:initialize_bedrock_models
        """
        model = _make_model()
        monkeypatch.setattr(
            SETTINGS,
            "aws_bedrock_model_arn_mapping",
            {"vendor.not-in-this-account-v1": _APP_PROFILE_ARN},
        )

        invalid = stdapi.models._apply_user_profiles({model.id: model})  # noqa: SLF001

        assert invalid == {
            "vendor.not-in-this-account-v1": "Model not found in available Bedrock models"
        }
        assert model.inference_profiles is None


class TestFilterInferenceProfiles:
    """_filter_inference_profiles: pick the preferred profile per model, keep a regional fallback.

    Amazon Nova 2 Lite advertises both a ``global.`` and a ``us.`` system-defined
    profile; with global cross-region inference on, ``global.`` wins the preferred
    slot, but Bedrock rejects the ``nova_grounding`` system tool on it, so the
    geo-scoped ``us.`` candidate must still be recorded for callers to fall back on.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
         stdapi/models/__init__.py:_filter_inference_profiles
    """

    def test_global_preferred_still_records_the_regional_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both the preferred (global) and the regional profile end up recorded."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_cross_region_inference_global", True)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_model_region_restrict", {})
        profiles: dict[str, str] = {}
        regional_profiles: dict[str, str] = {}

        _filter_inference_profiles(
            profiles,
            regional_profiles,
            {
                "amazon.nova-2-lite-v1:0": [
                    "us.amazon.nova-2-lite-v1:0",
                    "global.amazon.nova-2-lite-v1:0",
                ]
            },
        )

        assert profiles == {"amazon.nova-2-lite-v1:0": "global.amazon.nova-2-lite-v1:0"}
        assert regional_profiles == {
            "amazon.nova-2-lite-v1:0": "us.amazon.nova-2-lite-v1:0"
        }

    def test_global_only_model_has_no_regional_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model with only a ``global.`` profile leaves the regional map untouched."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_cross_region_inference_global", True)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_model_region_restrict", {})
        profiles: dict[str, str] = {}
        regional_profiles: dict[str, str] = {}

        _filter_inference_profiles(
            profiles,
            regional_profiles,
            {"amazon.global-only-v1:0": ["global.amazon.global-only-v1:0"]},
        )

        assert profiles == {"amazon.global-only-v1:0": "global.amazon.global-only-v1:0"}
        assert regional_profiles == {}


class TestGetIdPreferRegional:
    """ModelDetails.get_id(prefer_regional=True): route a request around the global profile.

    Ref: stdapi/models/__init__.py:ModelDetails.get_id
    """

    def test_prefers_the_cached_regional_profile_for_the_region(self) -> None:
        """A regional profile cached for the target region wins over the preferred (global) one."""
        details = make_model_details(
            "amazon.nova-2-lite-v1:0",
            inference_profiles={"us-east-1": "global.amazon.nova-2-lite-v1:0"},
            inference_profiles_regional={"us-east-1": "us.amazon.nova-2-lite-v1:0"},
        )
        assert (
            details.get_id("us-east-1", inference_profile=True, prefer_regional=True)
            == "us.amazon.nova-2-lite-v1:0"
        )

    def test_falls_back_to_the_default_profile_without_a_regional_candidate(
        self,
    ) -> None:
        """No cached regional profile: behaves exactly like ``prefer_regional=False``."""
        details = make_model_details(
            "amazon.global-only-v1:0",
            inference_profiles={"us-east-1": "global.amazon.global-only-v1:0"},
        )
        assert (
            details.get_id("us-east-1", inference_profile=True, prefer_regional=True)
            == "global.amazon.global-only-v1:0"
        )

    def test_ignored_when_inference_profile_is_not_requested(self) -> None:
        """``prefer_regional`` has no effect on the bare on-demand model ID path."""
        details = make_model_details(
            "amazon.nova-2-lite-v1:0",
            inference_profiles={"us-east-1": "global.amazon.nova-2-lite-v1:0"},
            inference_profiles_regional={"us-east-1": "us.amazon.nova-2-lite-v1:0"},
        )
        assert (
            details.get_id("us-east-1", prefer_regional=True)
            == "amazon.nova-2-lite-v1:0"
        )


class TestRequestUsesSystemTool:
    """_request_uses_system_tool: detect a promoted Bedrock system tool in a Converse request.

    Ref: stdapi/models/__init__.py:_request_uses_system_tool
         stdapi/models/chat/_default.py:ChatModelBase._req_promote_system_tools
    """

    def test_true_when_a_system_tool_entry_is_present(self) -> None:
        """A ``systemTool`` entry in ``toolConfig.tools`` is detected."""
        request: ConverseRequestBaseTypeDef = {
            "modelId": "",
            "toolConfig": {"tools": [{"systemTool": {"name": "nova_grounding"}}]},
        }
        assert _request_uses_system_tool(request) is True

    def test_false_for_a_regular_tool(self) -> None:
        """A plain ``toolSpec`` entry, with no system tool, is not detected."""
        request: ConverseRequestBaseTypeDef = {
            "modelId": "",
            "toolConfig": {
                "tools": [{"toolSpec": {"name": "get_weather", "inputSchema": {}}}]
            },
        }
        assert _request_uses_system_tool(request) is False

    def test_false_without_a_tool_config(self) -> None:
        """A request with no tools at all reports no system tool."""
        request: ConverseRequestBaseTypeDef = {"modelId": ""}
        assert _request_uses_system_tool(request) is False


@pytest.fixture
def _request_id_context(request_log: EventLog) -> Generator[EventLog]:
    """Bind ``REQUEST_ID`` alongside the request log for code called outside a real request.

    ``_prepare_converse_request_for_region`` builds ``requestMetadata`` via
    ``build_metadata``, which reads both context variables unconditionally.

    Yields:
        The bound request log.
    """
    token = REQUEST_ID.set("req-system-tool-routing")
    yield request_log
    REQUEST_ID.reset(token)


async def _capturing_resolve(
    model_id: str, _region: RegionName, captured: dict[str, object], **kwargs: object
) -> str:
    """Stand in for resolve_routed_model_id, recording its keyword arguments."""
    captured.update(kwargs)
    return model_id


@pytest.mark.usefixtures("_request_id_context")
class TestPrepareConverseRequestRoutesSystemTools:
    """ModelBase._prepare_converse_request_for_region: route system-tool requests off ``global.``.

    Ref: stdapi/models/__init__.py:ModelBase._prepare_converse_request_for_region (issue #92)
    """

    async def test_system_tool_request_prefers_the_regional_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request carrying a ``systemTool`` entry is resolved with ``prefer_regional=True``."""
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            stdapi.models,
            "resolve_routed_model_id",
            partial(_capturing_resolve, captured=captured),
        )
        request: ConverseRequestBaseTypeDef = {
            "modelId": "",
            "toolConfig": {"tools": [{"systemTool": {"name": "nova_grounding"}}]},
        }

        model: ModelBase[Any, Any] = ModelBase("amazon.nova-2-lite-v1:0")
        await model._prepare_converse_request_for_region(  # noqa: SLF001
            request, "us-east-1"
        )

        assert captured["prefer_regional"] is True

    async def test_plain_request_does_not_prefer_the_regional_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request without a system tool keeps the default (possibly global) routing."""
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            stdapi.models,
            "resolve_routed_model_id",
            partial(_capturing_resolve, captured=captured),
        )
        request: ConverseRequestBaseTypeDef = {
            "modelId": "",
            "toolConfig": {
                "tools": [{"toolSpec": {"name": "get_weather", "inputSchema": {}}}]
            },
        }

        model: ModelBase[Any, Any] = ModelBase("amazon.nova-2-lite-v1:0")
        await model._prepare_converse_request_for_region(  # noqa: SLF001
            request, "us-east-1"
        )

        assert captured["prefer_regional"] is False


class TestCheckCandidates:
    """_check_candidates: resolve every model in parallel rounds across its candidate regions.

    Each round checks all still-unresolved models against their next candidate region
    at once; the round loop only runs again for models that failed. Once a model passes
    anywhere, its remaining regions are merged unchecked to keep the fan-out bounded.

    Ref: stdapi/models/__init__.py:_check_candidates
    """

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
        """A model passing in its first region absorbs its later regions with one check.

        The saving is the point: one check per model instead of one per (model, region)
        keeps a full refresh within the control-plane API's rate quota.
        """
        calls = self._patch_availability(monkeypatch, {})
        candidates = {"m1": [_make_model("m1"), _make_model("m1", region="eu-west-1")]}
        unavailable: dict[str, dict[str, list[str]]] = {}

        result = await _check_candidates(candidates, unavailable)

        assert list(result) == ["m1"]
        assert result["m1"].regions == ["us-east-1", "eu-west-1"]
        assert calls == ["m1@us-east-1"]
        assert unavailable == {}

    async def test_failure_falls_through_to_the_next_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model failing in region 1 is retried in region 2 and kept with that region only.

        The failing region is dropped from the model's region list, not merged: routing
        a request there would fail on every attempt.
        """
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
        """A model failing in every candidate region is dropped, with its per-region reasons.

        Both regions' issue lists are preserved so the startup warning can tell an
        operator which grant is missing where.
        """
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
        """``["unavailable"]`` alone drops the model without recording an operator warning.

        ``ListFoundationModels`` and ``GetFoundationModelAvailability`` disagree on this
        state for some models (``amazon.titan-embed-g1-text-02``): nothing is
        misconfigured and nothing an operator can grant, so it is skipped silently. Any
        other label — even alongside ``unavailable`` — is still reported.
        """
        self._patch_availability(monkeypatch, {"us-east-1": ["unavailable"]})
        unavailable: dict[str, dict[str, list[str]]] = {}

        result = await _check_candidates({"m1": [_make_model("m1")]}, unavailable)

        assert result == {}
        assert unavailable == {}

    async def test_all_models_are_checked_in_one_round(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Models first listed in different regions are all checked in the same round.

        Rounds are per-model progress, not per-region passes: a model whose first
        candidate is a later region is not deferred to a second round.
        """
        calls = self._patch_availability(monkeypatch, {})
        candidates = {
            "m1": [_make_model("m1")],
            "m2": [_make_model("m2", region="eu-west-1")],
        }

        result = await _check_candidates(candidates, {})

        assert sorted(result) == ["m1", "m2"]
        assert sorted(calls) == ["m1@us-east-1", "m2@eu-west-1"]
        assert len(calls) == 2, "one round, one check per model"

    async def test_all_checks_erroring_raises_the_first_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When EVERY availability check errors, the first error is re-raised.

        All checks failing means the control plane, not the catalogue, is broken:
        publishing an empty model list would look like every model disappeared.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html
        """
        self._patch_availability(
            monkeypatch,
            {
                "us-east-1": make_client_error(
                    "ThrottlingException",
                    "GetFoundationModelAvailability",
                    message="slow down",
                ),
                "eu-west-1": make_client_error(
                    "ThrottlingException",
                    "GetFoundationModelAvailability",
                    message="slow down",
                ),
            },
        )
        candidates = {
            "m1": [_make_model("m1")],
            "m2": [_make_model("m2", region="eu-west-1")],
        }
        unavailable: dict[str, dict[str, list[str]]] = {}

        with pytest.raises(ClientError) as excinfo:
            await _check_candidates(candidates, unavailable)

        assert excinfo.value.response["Error"]["Code"] == "ThrottlingException"
        assert excinfo.value.operation_name == "GetFoundationModelAvailability"
        assert unavailable == {
            "m1": {"us-east-1": ["availability check failed: ClientError"]},
            "m2": {"eu-west-1": ["availability check failed: ClientError"]},
        }

    async def test_partial_check_errors_degrade_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single erroring check becomes a per-region issue and the model retries elsewhere.

        The call log also pins the round structure: ``m1``'s retry in ``eu-west-1``
        happens in a second round, after both first-round checks.
        """
        calls = self._patch_availability(
            monkeypatch,
            {
                "us-east-1": make_client_error(
                    "ThrottlingException",
                    "GetFoundationModelAvailability",
                    message="slow down",
                )
            },
        )
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
        """A non-AWS exception propagates immediately instead of degrading a region.

        Only ``BotoCoreError`` / ``ClientError`` are tolerated as region issues; a
        ``ValueError`` is a gateway bug and must not be reported as "model unavailable
        in us-east-1", nor be absorbed by the all-checks-failed accounting.
        """
        self._patch_availability(monkeypatch, {"us-east-1": ValueError("bug")})
        unavailable: dict[str, dict[str, list[str]]] = {}

        with pytest.raises(ValueError, match=r"^bug$"):
            await _check_candidates({"m1": [_make_model("m1")]}, unavailable)

        assert unavailable == {}, "a gateway bug is not a model-availability issue"


def _patch_regions_and_fetch(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, list[ModelDetails] | Exception]
) -> None:
    """Fake the region list and per-region fetch results."""
    monkeypatch.setattr(region_routing, "ORDERED_BEDROCK_REGIONS", list(results))

    async def _fake(
        region: RegionName, _denied: dict[str, str] | None = None
    ) -> list[ModelDetails]:
        result = results[region]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(stdapi.models, "_get_bedrock_models_from_region", _fake)


class TestCollectRegionCandidates:
    """_collect_region_candidates: fan out the listing per region and isolate a bad region.

    A region-level failure must not take the catalogue down; the region is recorded and
    retried on the next refresh, and models exclusive to it simply disappear until then.

    Ref: stdapi/models/__init__.py:_collect_region_candidates
    """

    @staticmethod
    def _aws_error() -> EndpointConnectionError:
        return EndpointConnectionError(endpoint_url="https://bedrock.invalid")

    async def test_one_failed_region_is_tolerated_and_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing region is skipped and recorded, while the healthy region still merges.

        The recorded value carries the error class so the startup warning names what
        went wrong, not just which region.
        """
        _patch_regions_and_fetch(
            monkeypatch,
            {
                "us-east-1": self._aws_error(),
                "eu-west-1": [_make_model("m1", "eu-west-1")],
            },
        )
        failed: dict[str, str] = {}

        candidates = await _collect_region_candidates(failed)

        assert list(candidates) == ["m1"]
        assert [model.regions[0] for model in candidates["m1"]] == ["eu-west-1"]
        assert list(failed) == ["us-east-1"]
        assert failed["us-east-1"].startswith("EndpointConnectionError")

    async def test_all_regions_failing_raises_the_first_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every region fails, the first error is raised instead of an empty catalogue.

        An empty result would be published as "no model exists", so total failure has to
        stay an error — while still recording every region for the diagnostic.
        """
        _patch_regions_and_fetch(
            monkeypatch,
            {"us-east-1": self._aws_error(), "eu-west-1": self._aws_error()},
        )
        failed: dict[str, str] = {}

        with pytest.raises(EndpointConnectionError) as excinfo:
            await _collect_region_candidates(failed)

        assert "https://bedrock.invalid" in str(excinfo.value)
        assert sorted(failed) == ["eu-west-1", "us-east-1"]

    async def test_non_aws_errors_propagate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-AWS exception propagates even though another region succeeded.

        Region tolerance is scoped to ``BotoCoreError`` / ``ClientError``; a gateway bug
        must surface loudly instead of silently shrinking the catalogue.
        """
        _patch_regions_and_fetch(
            monkeypatch,
            {"us-east-1": ValueError("bug"), "eu-west-1": [_make_model("m1")]},
        )
        failed: dict[str, str] = {}

        with pytest.raises(ValueError, match=r"^bug$"):
            await _collect_region_candidates(failed)

        assert failed == {}, "a gateway bug is not a region failure"

    async def test_candidates_keep_region_priority_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-model candidates follow the configured region order, not completion order.

        The regions are queried concurrently, so the order has to come from zipping the
        results back onto the ordered region list rather than from whichever finished
        first — it is what decides the model's preferred region.
        """
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
    # Both, not just the deadline: "never refreshed" is what makes the cache
    # cold, and a cold cache is the one state that still refreshes inline.
    monkeypatch.setitem(stdapi.models._CACHE, "update_next", None)  # noqa: SLF001
    monkeypatch.setitem(stdapi.models._CACHE, "updated_at", None)  # noqa: SLF001
    yield
    for name, content in saved.items():
        target = getattr(stdapi.models, name)
        target.clear()
        target.update(content)


@pytest.mark.usefixtures("_isolated_model_cache")
class TestInitializeBedrockModelsFaultIsolation:
    """initialize_bedrock_models: what a degraded region does to a whole refresh.

    Startup must fail loudly when discovery is entirely broken, but a single bad region
    only costs its own models plus a warning — and arms the TTL so the next refresh
    retries it.

    Ref: stdapi/models/__init__.py:initialize_bedrock_models
         stdapi/models/__init__.py:_warn_bedrock_refresh_issues
    """

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
        return make_event_log(type="start")

    async def test_startup_fails_when_every_region_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup fails, and leaves the model cache untouched, when no region lists models.

        The refresh TTL must stay unarmed too: serving with an empty catalogue would
        turn a connectivity outage into "every model is gone" for the whole TTL.
        """
        _patch_regions_and_fetch(
            monkeypatch,
            {
                "us-east-1": EndpointConnectionError(endpoint_url="https://a.invalid"),
                "eu-west-1": EndpointConnectionError(endpoint_url="https://b.invalid"),
            },
        )

        with pytest.raises(EndpointConnectionError) as excinfo:
            await stdapi.models.initialize_bedrock_models(self._start_event())

        assert ".invalid" in str(excinfo.value)
        assert stdapi.models._CACHE["update_next"] is None  # noqa: SLF001

    async def test_startup_fails_when_every_availability_check_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup fails when the listing worked but every availability check errored.

        This is the second way discovery can be wholly broken — the listing succeeds and
        only ``GetFoundationModelAvailability`` is denied or throttled — and it must fail
        the same way, without arming the refresh TTL.
        """
        _patch_regions_and_fetch(
            monkeypatch, {"us-east-1": [_make_model("m1")], "eu-west-1": []}
        )

        async def _denied(_model: ModelDetails) -> list[str]:
            error = make_client_error(
                "ThrottlingException",
                "GetFoundationModelAvailability",
                message="slow down",
            )
            raise error

        monkeypatch.setattr(stdapi.models, "_check_model_availability", _denied)

        with pytest.raises(ClientError) as excinfo:
            await stdapi.models.initialize_bedrock_models(self._start_event())

        assert excinfo.value.response["Error"]["Code"] == "ThrottlingException"
        assert stdapi.models._CACHE["update_next"] is None  # noqa: SLF001

    async def test_startup_survives_one_region_and_warns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup succeeds on one healthy region, warns about the other, and arms the TTL.

        The return value is the "cache was refreshed" flag the caller uses to publish the
        new model list; the warning lands on the startup event rather than a request log.
        """
        self._patch_fetches(monkeypatch)
        start_event = make_event_log(type="start")

        assert await stdapi.models.initialize_bedrock_models(start_event) is True

        assert "m1" in stdapi.models._MODELS  # noqa: SLF001
        assert stdapi.models._MODELS["m1"].regions == ["eu-west-1"]  # noqa: SLF001
        unreachable = _unreachable_regions(start_event["server_warnings"])
        assert "us-east-1" in unreachable
        assert "eu-west-1" not in unreachable
        assert str(unreachable["us-east-1"]).startswith("EndpointConnectionError")
        # The TTL was armed: the failed region is retried on the next refresh.
        assert stdapi.models._CACHE["update_next"] is not None  # noqa: SLF001

    async def test_lazy_refresh_warns_in_the_request_log(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A lazy refresh surfaces the unreachable region on the current request log.

        Outside startup there is no start event to carry the warning, so the degraded
        region is only visible if the in-flight request's log is raised to ``warning``.
        """
        self._patch_fetches(monkeypatch)
        # Pin the price-catalog refresh guard so this unit test never hits
        # the live AWS Pricing API when cost tracking is enabled.
        monkeypatch.setattr(SETTINGS, "cost_tracking", False)

        await stdapi.models.initialize_bedrock_models()

        assert request_log["level"] == "warning"
        unreachable = _unreachable_regions(request_log["error_detail"])
        assert "us-east-1" in unreachable
        assert "eu-west-1" not in unreachable
        assert "m1" in stdapi.models._MODELS  # noqa: SLF001


#: How AWS words the denial that hides behind a region reported as unreachable.
_DENIAL_MESSAGE = (
    "User: arn:aws:sts::123456789012:assumed-role/stdapi/task is not authorized to "
    "perform: bedrock:ListProvisionedModelThroughputs"
)


def _denied_regions(entries: list[JsonValue]) -> dict[str, JsonValue]:
    """Extract the bedrock_regions_missing_iam_permission payload from log entries."""
    (payload,) = [
        entry["bedrock_regions_missing_iam_permission"]
        for entry in entries
        if isinstance(entry, dict) and "bedrock_regions_missing_iam_permission" in entry
    ]
    assert isinstance(payload, dict)
    return payload


@pytest.mark.usefixtures("_isolated_model_cache")
class TestRegionDeniedByIam:
    """A region the role has no permission in is not a region that is down.

    "Unreachable" is what an operator reads to decide whether a model really
    went away, so an IAM gap wearing that label sends them looking for a
    regional outage instead of at their own policy. The two states are
    reported under separate keys, and the denied one names the permission.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListProvisionedModelThroughputs.html
         stdapi/models/__init__.py:_collect_region_candidates
         stdapi/models/__init__.py:skipped_regions_detail
    """

    @staticmethod
    def _patch_denied_region(monkeypatch: pytest.MonkeyPatch) -> None:
        """One healthy region, one refused for a missing IAM permission.

        Mantle is switched off so the only skipped region is the denied one:
        its endpoint is a real HTTPS host no unit test may depend on.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_enabled", False)
        _patch_regions_and_fetch(
            monkeypatch,
            {
                "us-east-1": make_client_error(
                    "AccessDeniedException",
                    "ListProvisionedModelThroughputs",
                    message=_DENIAL_MESSAGE,
                ),
                "eu-west-1": [_make_model("m1", region="eu-west-1")],
            },
        )

        async def _available(_model: ModelDetails) -> list[str]:
            return []

        monkeypatch.setattr(stdapi.models, "_check_model_availability", _available)

    async def test_a_denied_region_is_recorded_apart_from_a_failed_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sweep sorts a denial into its own accumulator, naming the permission."""
        self._patch_denied_region(monkeypatch)
        failed: dict[str, str] = {}
        denied: dict[str, str] = {}

        await _collect_region_candidates(failed, denied)

        assert failed == {}, "a missing permission is not an unreachable endpoint"
        assert "bedrock:ListProvisionedModelThroughputs" in denied["us-east-1"]

    async def test_the_startup_warning_does_not_call_it_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Startup warns under the denied key only, and names what to grant."""
        self._patch_denied_region(monkeypatch)
        start_event = make_event_log(type="start")

        assert await stdapi.models.initialize_bedrock_models(start_event) is True

        warnings = start_event["server_warnings"]
        assert not any(
            isinstance(entry, dict) and "unreachable_bedrock_regions" in entry
            for entry in warnings
        ), "an IAM denial must not be reported as a region outage"
        denied = _denied_regions(warnings)
        assert "bedrock:ListProvisionedModelThroughputs" in str(denied["us-east-1"])
        assert "eu-west-1" not in denied

    async def test_a_denied_provisioned_listing_keeps_the_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The region keeps its on-demand models when only the PT listing is refused.

        Provisioned throughput is offered in a subset of regions and the control
        plane refuses the listing elsewhere, with prose AWS has already reworded
        once. Failing the region over it deleted every model that region serves;
        the denial is recorded and the listing proceeds.

        Ref: stdapi/models/__init__.py:_get_provisioned_models
        """
        client = _StubModelListClient([_summary("vendor.on-demand-v1")])
        monkeypatch.setattr(
            stdapi.models, "get_client", lambda _service, _region: client
        )

        async def _empty_profiles(
            _client: object,
        ) -> tuple[dict[str, str], dict[str, str]]:
            return {}, {}

        monkeypatch.setattr(stdapi.models, "_get_inference_profiles", _empty_profiles)
        denied: dict[str, str] = {}

        models = await stdapi.models._get_bedrock_models_from_region(  # noqa: SLF001
            "af-south-1", denied
        )

        assert [model.id for model in models] == ["vendor.on-demand-v1"]
        assert "bedrock:ListProvisionedModelThroughputs" in denied["af-south-1"]
        assert "provisioned model discovery skipped" in denied["af-south-1"]

    async def test_a_transport_failure_is_still_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The split does not reclassify the failures that really are outages."""
        _patch_regions_and_fetch(
            monkeypatch,
            {
                "us-east-1": EndpointConnectionError(endpoint_url="https://x.invalid"),
                "eu-west-1": [_make_model("m1", region="eu-west-1")],
            },
        )
        failed: dict[str, str] = {}
        denied: dict[str, str] = {}

        await _collect_region_candidates(failed, denied)

        assert denied == {}
        assert "us-east-1" in failed


class _StubModelListClient:
    """Stub bedrock client returning pre-defined foundation model summaries."""

    def __init__(self, summaries: list[dict[str, Any]]) -> None:
        self._summaries = summaries

    async def list_foundation_models(self) -> dict[str, Any]:
        """Return the pre-defined model summaries."""
        return {"modelSummaries": self._summaries}

    async def list_provisioned_model_throughputs(
        self, **_params: str
    ) -> dict[str, Any]:
        """Refuse exactly as Bedrock does where provisioned throughput is absent.

        The response carries what ``stdapi.aws._record_after_call`` records on a
        real denial, since it is the hook, not the service, that names the
        action for a message worded like this one.

        Raises:
            ClientError: Always, with the verbatim message AWS answers with.
        """
        response: Any = {
            "Error": {
                "Code": "AccessDeniedException",
                "Message": "Your account is not authorized to invoke this API operation.",
            },
            DENIED_CALL_KEY: {
                "action": "bedrock:ListProvisionedModelThroughputs",
                "region": "af-south-1",
            },
        }
        raise ClientError(response, "ListProvisionedModelThroughputs")


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
    """_get_bedrock_models_from_region: lifecycle filtering and the ``legacy`` annotation.

    ``modelLifecycle.status`` only ever holds ``ACTIVE`` or ``LEGACY`` on the wire — EOL
    is a documentation state, at which the model simply stops being listed. The gateway
    therefore also treats the ``legacyTime`` / ``endOfLifeTime`` timestamps as lifecycle
    input, comparing them against the NEXT cache refresh so a model that transitions
    between two refreshes is dropped early rather than served until it breaks.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html
         stdapi/models/__init__.py:_get_bedrock_models_from_region
    """

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

        async def _empty_provisioned(
            _client: object, _region: str, _denied: dict[str, str]
        ) -> set[str]:
            return set()

        async def _empty_profiles(
            _client: object,
        ) -> tuple[dict[str, str], dict[str, str]]:
            return {}, {}

        monkeypatch.setattr(
            stdapi.models, "_get_provisioned_models", _empty_provisioned
        )
        monkeypatch.setattr(stdapi.models, "_get_inference_profiles", _empty_profiles)
        return await stdapi.models._get_bedrock_models_from_region("us-east-1")  # noqa: SLF001

    async def test_legacy_models_hidden_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without ``aws_bedrock_legacy``, both LEGACY-status and past-``legacyTime`` models drop.

        A ``legacyTime`` still in the future is not yet legacy, so that model survives and
        carries no ``legacy`` annotation — proving the timestamp is compared, not merely
        present.
        """
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
        """With ``aws_bedrock_legacy``, a LEGACY model is kept but flagged ``legacy=True``.

        Opting in exposes the model without hiding what it is: the flag is what the
        ``/v1/models`` payload surfaces so a caller can tell it is on borrowed time.
        """
        models = await self._fetch(
            monkeypatch,
            [
                _summary("vendor.active"),
                _summary("vendor.legacy", status="LEGACY", legacy_time=self._PAST),
            ],
            legacy=True,
        )
        assert [model.id for model in models] == ["vendor.active", "vendor.legacy"]
        assert models[0].legacy is None
        assert models[1].legacy is True
        assert models[1].legacy_time == self._PAST

    async def test_end_of_life_models_always_hidden(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A past ``endOfLifeTime`` hides the model even with ``aws_bedrock_legacy`` set.

        The legacy opt-in is not an EOL opt-in: past EOL the model no longer serves
        traffic at all, so advertising it could only produce failures.
        """
        models = await self._fetch(
            monkeypatch,
            [_summary("vendor.eol", status="LEGACY", eol_time=self._PAST)],
            legacy=True,
        )
        assert models == []


class TestTriggerPriceCatalogRefresh:
    """_trigger_price_catalog_refresh: a Pricing API failure degrades cost tracking only.

    A lazy refresh that discovers a newly released model pulls its prices immediately so
    cost tracking does not wait for the next background poll. The Pricing API has a very
    low rate quota, so that opportunistic call must never be able to fail model listing.

    Ref: stdapi/models/__init__.py:_trigger_price_catalog_refresh
    """

    async def test_client_error_is_warned_and_does_not_propagate(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A ``ClientError`` from the catalog reload becomes a request-log warning, not a raise.

        Ref: stdapi/pricing.py:refresh_price_catalog_for_new_models
        """
        requested: list[set[str]] = []

        async def _fail(model_ids: set[str]) -> None:
            requested.append(model_ids)
            error = make_client_error(
                "ThrottlingException", "GetProducts", message="slow down"
            )
            raise error

        monkeypatch.setattr(
            stdapi.models, "refresh_price_catalog_for_new_models", _fail
        )
        # Does not raise: model listing must still succeed.
        await _trigger_price_catalog_refresh(None, {"vendor.new-model"})

        assert requested == [{"vendor.new-model"}], (
            "only the newly discovered model IDs are re-priced"
        )
        assert request_log["level"] == "warning"
        assert any(
            "Price-catalog refresh" in str(detail) and "slow down" in str(detail)
            for detail in request_log["error_detail"]
        ), request_log.get("error_detail")
