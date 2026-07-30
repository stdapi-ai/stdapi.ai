"""Region-safe Bedrock model identifier resolution and cross-region retry.

A cross-region inference profile ID is only valid inside its own geography, except
for the ``global.`` prefix which is valid everywhere. Sending a geo-scoped profile
to the wrong region makes Bedrock answer ``ValidationException`` ("The provided
model identifier is invalid."), so the gateway resolves the identifier per region
and treats a region that cannot serve the model as merely skippable.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
     stdapi/models/__init__.py:ModelDetails.get_id
     stdapi/models/__init__.py:route_and_execute
"""

from typing import TYPE_CHECKING, cast

import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

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

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


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
    """ModelDetails.get_id: pick the identifier that the target region actually accepts.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-use.html
         stdapi/models/__init__.py:ModelDetails.get_id
    """

    def test_returns_bare_id_when_inference_profile_not_requested(self) -> None:
        """Without ``inference_profile`` the bare foundation-model ID wins over a known profile.

        The model does have a profile for the requested region, so returning the bare
        ID is a decision and not a fallback: callers that cannot use a profile (e.g.
        control-plane lookups) must never receive one.
        """
        model = _model({"us-east-1": "us.vendor.model-v1"}, ["us-east-1"])
        assert model.get_id("us-east-1") == "vendor.model-v1"

    def test_returns_bare_id_for_on_demand_model(self) -> None:
        """A model with no discovered profile resolves to its bare on-demand ID.

        On-demand-only models have no inference profile at all, so requesting one is
        not an error — the foundation-model ID is the correct ``modelId``.
        """
        model = _model(None, ["us-east-1", "eu-west-1"])
        assert model.get_id("eu-west-1", inference_profile=True) == "vendor.model-v1"

    def test_returns_per_region_profile_when_present(self) -> None:
        """The profile registered for the requested region is returned verbatim.

        Both geographies have their own profile here, so the lookup must select by
        region rather than take the first entry.
        """
        model = _model(
            {"us-east-1": "us.vendor.model-v1", "eu-west-1": "eu.vendor.model-v1"},
            ["us-east-1", "eu-west-1"],
        )
        assert model.get_id("eu-west-1", inference_profile=True) == "eu.vendor.model-v1"
        assert model.get_id("us-east-1", inference_profile=True) == "us.vendor.model-v1"

    def test_falls_back_to_global_profile_for_missing_region(self) -> None:
        """A ``global.`` profile covers a region that has no profile entry of its own.

        Global cross-Region inference profiles accept requests from any commercial
        Region, so the region key they were discovered under does not constrain them.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
        """
        model = _model(
            {"us-east-1": "global.vendor.model-v1"}, ["us-east-1", "eu-west-1"]
        )
        assert (
            model.get_id("eu-west-1", inference_profile=True)
            == "global.vendor.model-v1"
        )

    def test_raises_for_missing_region_with_only_geo_profiles(self) -> None:
        """A geo-scoped profile is never returned for a region outside its geography.

        Bedrock would reject the mismatched profile with ``ValidationException``; the
        error carries the offending region so ``route_and_execute`` can skip it.
        """
        model = _model({"us-east-1": "us.vendor.model-v1"}, ["us-east-1", "eu-west-1"])
        with pytest.raises(ModelRegionUnavailableError) as excinfo:
            model.get_id("eu-west-1", inference_profile=True)
        assert excinfo.value.region == "eu-west-1"
        assert "eu-west-1" in str(excinfo.value)
        assert "vendor.model-v1" in str(excinfo.value)

    def test_returns_any_profile_when_region_unspecified(self) -> None:
        """With ``region=None`` the single geo profile is accepted rather than refused.

        A region-less caller (model listing, capability probing) has no geography to
        violate, so the profile is returned instead of raising.
        """
        model = _model({"us-east-1": "us.vendor.model-v1"}, ["us-east-1"])
        assert model.get_id(inference_profile=True) == "us.vendor.model-v1"


def test_is_invalid_model_identifier_matches_only_that_validation_error() -> None:
    """Only a ``ValidationException`` whose message names an invalid model identifier matches.

    The classification decides retryability, so it keys on the message and not on the
    code: every malformed Converse request shares the ``ValidationException`` code,
    and retrying those across regions would be pure latency.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html
         stdapi/models/__init__.py:_is_invalid_model_identifier
    """
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
    throttling = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
        "InvokeModel",
    )
    assert not _is_invalid_model_identifier(throttling)


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
    """route_and_execute: skip a region that cannot serve the model, keep fatal errors fatal.

    Discovery snapshots go stale: a region can advertise a model minutes before its
    inference profile propagates there. Skipping such a region (and marking it
    unusable for that model) lets the request self-heal instead of 400-ing.

    Ref: stdapi/models/__init__.py:route_and_execute
         stdapi/region_routing.py:RegionRouter.mark_error
    """

    async def test_skips_region_on_invalid_model_identifier(
        self, routed: RegionRouter
    ) -> None:
        """An invalid-identifier ``ValidationException`` fails over and blocks that region.

        ``ValidationException`` is not in the router's retryable code set, so this
        failover comes purely from the invalid-identifier message match.
        """
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
        assert routed._index.get("vendor.model-v1", "eu-west-1").is_usable  # noqa: SLF001

    async def test_skips_region_on_model_region_unavailable(
        self, routed: RegionRouter
    ) -> None:
        """A ``ModelRegionUnavailableError`` fails over and blocks the offending region.

        This is the pre-flight variant: the gateway refuses to build a mismatched
        identifier locally instead of letting Bedrock reject it.
        """
        seen: list[str] = []

        async def fn(region: RegionName) -> str:
            seen.append(region)
            if region == "us-east-1":
                msg = "no profile"
                raise ModelRegionUnavailableError(msg, region=region)
            return f"ok:{region}"

        result = await route_and_execute("vendor.model-v1", _CANDIDATES, fn)

        assert result == "ok:eu-west-1"
        assert seen == ["us-east-1", "eu-west-1"]
        assert not routed._index.get(  # noqa: SLF001
            "vendor.model-v1", "us-east-1"
        ).is_usable

    async def test_reraises_unrelated_validation_error(
        self, routed: RegionRouter
    ) -> None:
        """A non-identifier ``ValidationException`` is raised as-is, without a second region.

        The caller's request is malformed, so every region would reject it identically;
        the region is also left usable because nothing about it is unhealthy.
        """
        seen: list[str] = []

        async def fn(region: RegionName) -> str:
            seen.append(region)
            raise ClientError(
                {
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "Malformed input request.",
                    }
                },
                "InvokeModel",
            )

        with pytest.raises(ClientError) as excinfo:
            await route_and_execute("vendor.model-v1", _CANDIDATES, fn)

        assert excinfo.value.response["Error"]["Code"] == "ValidationException"
        assert "Malformed input request." in str(excinfo.value)
        assert len(seen) == 1, "a fatal caller error must not be retried elsewhere"
        assert routed._index.get("vendor.model-v1", seen[0]).is_usable  # noqa: SLF001

    async def test_skips_region_on_model_not_ready(self, routed: RegionRouter) -> None:
        """``ModelNotReadyException`` survives its ``ApiError`` wrapping and still fails over.

        ``handle_bedrock_client_error`` converts it to a 503 ``ApiError`` before the
        router sees it, so retryability has to be recovered from ``__cause__``; missing
        that made a warming model a hard failure.

        Ref: stdapi/aws_bedrock.py:handle_bedrock_client_error
             stdapi/models/__init__.py:_client_error_code
        """

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
        """An ``ApiError`` carrying a non-retryable AWS code is raised, not failed over.

        The S3-credential ``ValidationException`` also reaches the router as an
        ``ApiError``, so recovering the wrapped code must not turn every wrapped error
        into a retryable one: the bucket/region mismatch is identical in every region.

        Ref: stdapi/aws_bedrock.py:handle_bedrock_client_error
        """
        seen: list[str] = []

        async def fn(region: RegionName) -> str:
            seen.append(region)
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

        with pytest.raises(ApiError) as excinfo:
            await route_and_execute("vendor.model-v1", _CANDIDATES, fn)

        assert excinfo.value.status == 400
        assert "same region as the Bedrock model" in str(excinfo.value)
        assert len(seen) == 1, "a bucket/region mismatch is fatal in every region"

    async def test_read_timeout_is_not_retried_in_another_region(
        self, routed: RegionRouter
    ) -> None:
        """A ``ReadTimeoutError`` propagates from the first region instead of failing over.

        The request already reached Bedrock and is billed whatever the client does, so
        a second region would double-bill the invocation rather than recover it. The
        region is also left usable: a client-side timeout says nothing about its health.

        Ref: stdapi/models/__init__.py:_region_failover_label
        """
        seen: list[str] = []

        async def fn(region: RegionName) -> str:
            seen.append(region)
            raise ReadTimeoutError(endpoint_url="https://bedrock.example")

        with pytest.raises(ReadTimeoutError):
            await route_and_execute("vendor.model-v1", _CANDIDATES, fn)
        assert seen == ["us-east-1"]
        assert routed._index.get("vendor.model-v1", "us-east-1").is_usable  # noqa: SLF001

    async def test_single_region_unavailable_becomes_api_error(self) -> None:
        """With a single candidate, an unavailable region becomes a 400 ``ApiError``.

        There is no region left to skip to, so the internal signal must be translated
        into a client-facing error instead of escaping as an unknown exception type.
        """

        async def fn(region: RegionName) -> str:
            msg = "no profile"
            raise ModelRegionUnavailableError(msg, region=region)

        with pytest.raises(ApiError) as excinfo:
            await route_and_execute(
                "vendor.model-v1", cast("list[RegionName]", ["us-east-1"]), fn
            )

        assert excinfo.value.status == 400
        assert str(excinfo.value) == "no profile"
        assert isinstance(excinfo.value.__cause__, ModelRegionUnavailableError)


class TestRegionRestrictOrder:
    """aws_bedrock_model_region_restrict: the configured list order drives region priority.

    The setting is a data-residency control, so its order is not advisory: candidates
    are re-sorted into it rather than left in discovery order.

    Ref: stdapi/models/__init__.py:_region_restriction_for
         stdapi/models/__init__.py:_collect_region_candidates
    """

    def test_restriction_lookup_exact_then_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exact model-ID key wins, a prefix key matches as fallback, otherwise ``None``.

        Prefix matching is what lets one entry cover a whole family including the
        ``-v2:0`` revision suffixes Bedrock appends.
        """
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
        """A restricted model's candidates are re-sorted into the restriction order.

        The discovery order here is the reverse of the restriction order, so the result
        can only come from the re-sort and not from the region list.
        """
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
    """A region parsed out of a client-supplied ARN is validated before any client lookup.

    ``get_client`` indexes a pool built from ``aws_bedrock_regions``, so an
    unvalidated ARN region reaches it as an unhandled ``KeyError`` — a 500 for what is
    really a bad request. Both ARN kinds are gated behind their own setting, so the
    tests enable it first to reach the region check rather than the permission check.

    Ref: stdapi/models/__init__.py:_validate_bedrock_region
         stdapi/aws.py:get_client
    """

    async def test_prompt_router_arn_unconfigured_region_raises_api_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A prompt-router ARN in an unconfigured region is a 400 naming that region.

        Ref: stdapi/models/__init__.py:_get_prompt_router_models
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_prompt_router_arn", True)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1"])
        arn = "arn:aws:bedrock:ap-south-1:123456789012:prompt-router/r1"

        with pytest.raises(ApiError) as excinfo:
            await _get_prompt_router_models(arn)

        assert excinfo.value.status == 400
        assert "ap-south-1" in str(excinfo.value)
        assert "not a configured Bedrock region" in str(excinfo.value)

    async def test_inference_profile_arn_unconfigured_region_raises_api_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An application-inference-profile ARN in an unconfigured region is a 400 too.

        Ref: stdapi/models/__init__.py:_get_application_inference_profile_models
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_application_inference_profile_arn", True
        )
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1"])
        arn = "arn:aws:bedrock:ap-south-1:123456789012:application-inference-profile/p1"

        with pytest.raises(ApiError) as excinfo:
            await _get_application_inference_profile_models(arn)

        assert excinfo.value.status == 400
        assert "ap-south-1" in str(excinfo.value)
        assert "not a configured Bedrock region" in str(excinfo.value)
