"""Unit tests for region-safe model identifier resolution and failover.

Covers :meth:`ModelDetails.get_id` (never emit a geo-mismatched inference
profile), :func:`route_and_execute` (skip a region that cannot serve the
model instead of surfacing the error), and :func:`_get_prompt_router_models`
/ :func:`_get_application_inference_profile_models` (reject an ARN region
that isn't configured, instead of crashing later with an unhandled
``KeyError``), all fast and AWS-free.
"""

from typing import TYPE_CHECKING, cast

import pytest
from botocore.exceptions import ClientError

import stdapi.models as models_module
from stdapi import region_routing
from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import handle_bedrock_client_error
from stdapi.config import SETTINGS
from stdapi.models import (
    ModelDetails,
    ModelRegionUnavailableError,
    _collect_region_candidates,
    _get_application_inference_profile_models,
    _get_prompt_router_models,
    _is_invalid_model_identifier,
    _region_restriction_for,
    route_and_execute,
)
from stdapi.monitoring import REQUEST_LOG
from stdapi.region_routing import RegionRouter

if TYPE_CHECKING:
    from types_aiobotocore_bedrock.literals import RegionName


def _model(
    inference_profiles: dict[str, str] | None, regions: list[str]
) -> ModelDetails:
    """Build a minimal ModelDetails for identifier-resolution tests."""
    return ModelDetails(
        id="vendor.model-v1",
        name="Model",
        provider="Vendor",
        input_modalities=["TEXT"],
        output_modalities=["TEXT"],
        regions=regions,  # type: ignore[arg-type]
        inference_profiles=inference_profiles,  # type: ignore[arg-type]
    )


def _invalid_identifier_error() -> ClientError:
    """Build the Bedrock ValidationException for an invalid model identifier."""
    return ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "The provided model identifier is invalid.",
            }
        },
        "InvokeModel",
    )


class TestGetId:
    """ModelDetails.get_id region-safe identifier resolution."""

    def test_returns_bare_id_when_inference_profile_not_requested(self) -> None:
        """Without inference_profile, the bare foundation-model ID is returned."""
        model = _model({"us-east-1": "us.vendor.model-v1"}, ["us-east-1"])
        assert model.get_id("us-east-1") == "vendor.model-v1"

    def test_returns_bare_id_for_on_demand_model(self) -> None:
        """A model with no profiles resolves to its bare on-demand ID."""
        model = _model(None, ["us-east-1", "eu-west-1"])
        assert model.get_id("eu-west-1", inference_profile=True) == "vendor.model-v1"

    def test_returns_per_region_profile_when_present(self) -> None:
        """The profile matching the requested region is returned verbatim."""
        model = _model(
            {"us-east-1": "us.vendor.model-v1", "eu-west-1": "eu.vendor.model-v1"},
            ["us-east-1", "eu-west-1"],
        )
        assert model.get_id("eu-west-1", inference_profile=True) == "eu.vendor.model-v1"

    def test_falls_back_to_global_profile_for_missing_region(self) -> None:
        """A global. profile is valid everywhere, so it covers a region with no entry."""
        model = _model(
            {"us-east-1": "global.vendor.model-v1"}, ["us-east-1", "eu-west-1"]
        )
        assert (
            model.get_id("eu-west-1", inference_profile=True)
            == "global.vendor.model-v1"
        )

    def test_raises_for_missing_region_with_only_geo_profiles(self) -> None:
        """A geo-scoped profile is never returned for a different region."""
        model = _model({"us-east-1": "us.vendor.model-v1"}, ["us-east-1", "eu-west-1"])
        with pytest.raises(ModelRegionUnavailableError) as excinfo:
            model.get_id("eu-west-1", inference_profile=True)
        assert excinfo.value.region == "eu-west-1"

    def test_returns_any_profile_when_region_unspecified(self) -> None:
        """With region=None the caller accepts any profile (a single geo profile here)."""
        model = _model({"us-east-1": "us.vendor.model-v1"}, ["us-east-1"])
        assert model.get_id(inference_profile=True) == "us.vendor.model-v1"


def test_is_invalid_model_identifier_matches_only_that_validation_error() -> None:
    """Only a ValidationException about an invalid model identifier is recognised."""
    assert _is_invalid_model_identifier(_invalid_identifier_error())
    other = ClientError(
        {
            "Error": {
                "Code": "ValidationException",
                "Message": "Malformed input request.",
            }
        },
        "InvokeModel",
    )
    assert not _is_invalid_model_identifier(other)


@pytest.fixture
def routed(monkeypatch: pytest.MonkeyPatch) -> RegionRouter:
    """Install a fresh two-region router as the module singleton with retries enabled."""
    router = RegionRouter()
    monkeypatch.setattr(models_module, "REGION_ROUTER", router)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_max_retries", 3)
    REQUEST_LOG.set({"level": "info"})  # type: ignore[typeddict-item]
    return router


_CANDIDATES = cast("list[RegionName]", ["us-east-1", "eu-west-1"])


class TestRouteAndExecuteFailover:
    """route_and_execute skips regions that cannot serve the model."""

    async def test_skips_region_on_invalid_model_identifier(
        self, routed: RegionRouter
    ) -> None:
        """An invalid-identifier ValidationException fails over to the next region."""
        seen: list[str] = []

        async def fn(region: RegionName) -> str:
            seen.append(region)
            if region == "us-east-1":
                raise _invalid_identifier_error()
            return f"ok:{region}"

        result = await route_and_execute("vendor.model-v1", _CANDIDATES, fn)

        assert result == "ok:eu-west-1"
        assert seen == ["us-east-1", "eu-west-1"]
        assert not routed._index.get(  # noqa: SLF001
            "vendor.model-v1", "us-east-1"
        ).is_usable

    async def test_skips_region_on_model_region_unavailable(
        self, routed: RegionRouter
    ) -> None:
        """A ModelRegionUnavailableError fails over to the next region."""

        async def fn(region: RegionName) -> str:
            if region == "us-east-1":
                msg = "no profile"
                raise ModelRegionUnavailableError(msg, region=region)
            return f"ok:{region}"

        result = await route_and_execute("vendor.model-v1", _CANDIDATES, fn)
        assert result == "ok:eu-west-1"

    async def test_reraises_unrelated_validation_error(
        self, routed: RegionRouter
    ) -> None:
        """A non-identifier ValidationException is not retried across regions."""

        async def fn(region: RegionName) -> str:  # noqa: ARG001
            raise ClientError(
                {
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "Malformed input request.",
                    }
                },
                "InvokeModel",
            )

        with pytest.raises(ClientError):
            await route_and_execute("vendor.model-v1", _CANDIDATES, fn)

    async def test_skips_region_on_model_not_ready(self, routed: RegionRouter) -> None:
        """Regression: ModelNotReadyException (wrapped as ApiError) fails over."""

        async def fn(region: RegionName) -> str:
            if region == "us-east-1":
                with handle_bedrock_client_error():
                    raise ClientError(
                        {
                            "Error": {
                                "Code": "ModelNotReadyException",
                                "Message": "The model is not ready.",
                            }
                        },
                        "Converse",
                    )
            return f"ok:{region}"

        result = await route_and_execute("vendor.model-v1", _CANDIDATES, fn)

        assert result == "ok:eu-west-1"
        assert not routed._index.get(  # noqa: SLF001
            "vendor.model-v1", "us-east-1"
        ).is_usable

    async def test_reraises_unrelated_api_error(self, routed: RegionRouter) -> None:
        """A non-retryable ApiError (e.g. invalid S3 credentials) is not retried."""

        async def fn(region: RegionName) -> str:  # noqa: ARG001
            with handle_bedrock_client_error():
                raise ClientError(
                    {
                        "Error": {
                            "Code": "ValidationException",
                            "Message": "Invalid S3 credentials received.",
                        }
                    },
                    "Converse",
                )

        with pytest.raises(ApiError):
            await route_and_execute("vendor.model-v1", _CANDIDATES, fn)

    async def test_single_region_unavailable_becomes_api_error(self) -> None:
        """With one candidate, an unavailable region surfaces as a clean ApiError."""

        async def fn(region: RegionName) -> str:
            msg = "no profile"
            raise ModelRegionUnavailableError(msg, region=region)

        with pytest.raises(ApiError):
            await route_and_execute(
                "vendor.model-v1", cast("list[RegionName]", ["us-east-1"]), fn
            )


class TestRegionRestrictOrder:
    """aws_bedrock_model_region_restrict list order drives region priority."""

    def test_restriction_lookup_exact_then_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exact model ID keys win; prefix keys match as fallback."""
        monkeypatch.setattr(
            SETTINGS,
            "aws_bedrock_model_region_restrict",
            {"vendor.model-v1": ("eu-west-1",), "vendor.other": ("us-east-1",)},
        )
        assert _region_restriction_for("vendor.model-v1") == ("eu-west-1",)
        assert _region_restriction_for("vendor.other-v2:0") == ("us-east-1",)
        assert _region_restriction_for("vendor.unknown-v1") is None

    async def test_candidates_follow_restriction_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Restricted models get candidates reordered to the configured order."""
        monkeypatch.setattr(
            SETTINGS,
            "aws_bedrock_model_region_restrict",
            {"vendor.model-v1": ("us-west-2", "us-east-1")},
        )
        monkeypatch.setattr(
            region_routing, "ORDERED_BEDROCK_REGIONS", ["us-east-1", "us-west-2"]
        )

        async def fake_fetch(region: RegionName) -> list[ModelDetails]:
            return [_model(None, [region])]

        monkeypatch.setattr(
            models_module, "_get_bedrock_models_from_region", fake_fetch
        )
        candidates = await _collect_region_candidates({})
        assert [
            candidate.regions[0] for candidate in candidates["vendor.model-v1"]
        ] == ["us-west-2", "us-east-1"]


class TestArnRegionValidation:
    """ARN-derived regions must be rejected before reaching get_client()."""

    async def test_prompt_router_arn_unconfigured_region_raises_api_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: an unconfigured ARN region must not crash get_client() with KeyError."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_prompt_router_arn", True)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1"])
        arn = "arn:aws:bedrock:ap-south-1:123456789012:prompt-router/r1"

        with pytest.raises(ApiError):
            await _get_prompt_router_models(arn)

    async def test_inference_profile_arn_unconfigured_region_raises_api_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: an unconfigured ARN region must not crash get_client() with KeyError."""
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_application_inference_profile_arn", True
        )
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1"])
        arn = "arn:aws:bedrock:ap-south-1:123456789012:application-inference-profile/p1"

        with pytest.raises(ApiError):
            await _get_application_inference_profile_models(arn)
