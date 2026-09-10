"""Model resolution when a request names a Bedrock ARN as its model.

An application inference profile, a cross-region inference profile and a prompt
router each name one caller's own resource, and the request that named one is
invoked on it. The catalogue entry the details are built from is shared by every
other caller and by the public model listing, so it has to come out of the
resolution exactly as it went in: a resolution that wrote the ARN there would
re-point everybody's requests at that caller's profile — billing them to it, and
failing them once its principal loses access — and publish the ARN, account ID
included, in the listing.

Both halves are one contract, so both are pinned here: the caller reaches its own
ARN, and nobody else does.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
     stdapi/models/__init__.py:_validate_model_from_arn
     stdapi/models/__init__.py:ModelDetails.get_id
"""

from asyncio import create_task, gather, sleep
from typing import TYPE_CHECKING, Any

import pytest

from stdapi import models
from stdapi.api_errors import ApiError
from stdapi.config import SETTINGS
from stdapi.models import (
    ModelDetails,
    get_model_details,
    resolve_routed_model_id,
    validate_model,
)

from ._helpers import make_model_details

if TYPE_CHECKING:
    from collections.abc import Coroutine, Generator

    from types_aiobotocore_bedrock.literals import RegionName

#: Every test here drives model resolution outside a request, which writes to the log.
pytestmark = pytest.mark.usefixtures("request_log")

#: Catalogue model the seeded application inference profile points at.
_MODEL_ID = "vendor.model-v1"
#: Region the seeded catalogue entry is served in.
_HOME_REGION: RegionName = "us-east-1"
#: Region the caller's own profile lives in, absent from the catalogue entry.
_OTHER_REGION: RegionName = "eu-west-1"
#: System-managed profile the catalogue entry routes ``_HOME_REGION`` to.
_SYSTEM_PROFILE = "global.vendor.model-v1"
#: Geo-scoped profile the catalogue entry offers for ``_HOME_REGION``.
_REGIONAL_PROFILE = "us.vendor.model-v1"
#: Foundation-model ARN the stubbed profile lookup reports as its only member.
_MEMBER_ARN = f"arn:aws:bedrock:{_HOME_REGION}::foundation-model/{_MODEL_ID}"
#: A second catalogue model, which no ARN in these tests names.
_OTHER_MODEL_ID = "vendor.other-v1"
#: System-managed profile that second model routes ``_HOME_REGION`` to.
_OTHER_SYSTEM_PROFILE = "global.vendor.other-v1"


async def _one_request[T](work: Coroutine[Any, Any, T]) -> T:
    """Await *work* in a context of its own, the way one request runs.

    Each request is served by its own task, which starts from a copy of the
    context: whatever the gateway binds while serving one request is invisible
    to the next. Awaiting two resolutions straight from a test body would share
    a single context and let the first leak into the second, which is the very
    thing these tests are about.

    Args:
        work: The resolution one request performs.

    Returns:
        Whatever *work* returned.
    """
    return await create_task(work)


def _application_profile_arn(region: str, profile: str = "abc123xyz") -> str:
    """Return an application-inference-profile ARN in *region*.

    Args:
        region: Region owning the caller's profile.
        profile: Identifier of the caller's own profile.

    Returns:
        The ARN a caller would pass as its ``model``.
    """
    return (
        f"arn:aws:bedrock:{region}:123456789012:application-inference-profile/{profile}"
    )


def _stub_profile_lookup(monkeypatch: pytest.MonkeyPatch, region: str) -> None:
    """Answer the inference-profile lookup with one member model, without AWS.

    What the resolution then does with the catalogue entry is decided in
    process, and no profile the account could hold would change it; creating a
    real one would add an account resource and prove nothing more. That the
    lookup itself works against Bedrock is covered live by
    ``test_openai_chat_completions.py::TestChatCompletions::test_prompt_router_as_model``.

    Args:
        monkeypatch: Patcher the stub is installed with.
        region: Region the profile is reported in.
    """

    async def lookup(_arn: str) -> tuple[list[dict[str, str]], str]:
        """Report the seeded catalogue model as the profile's only member."""
        return [{"modelArn": _MEMBER_ARN}], region

    monkeypatch.setattr(models, "_get_application_inference_profile_models", lookup)


@pytest.fixture
def seeded_catalog(monkeypatch: pytest.MonkeyPatch) -> Generator[ModelDetails]:
    """Install a one-model catalogue carrying inference profiles of its own.

    The real catalogue and the resolved-ARN cache are put back afterwards, and
    the background refresh is disabled so a lookup answers from what was seeded
    rather than sweeping AWS.

    Yields:
        The catalogue entry every caller naming the plain model ID resolves to.
    """
    saved_models = dict(models._MODELS)  # noqa: SLF001
    saved_profiles = dict(models._USER_PROFILES)  # noqa: SLF001
    entry = make_model_details(
        _MODEL_ID,
        regions=[_HOME_REGION],
        inference_profiles={_HOME_REGION: _SYSTEM_PROFILE},
        inference_profiles_regional={_HOME_REGION: _REGIONAL_PROFILE},
    )
    models._MODELS.clear()  # noqa: SLF001
    models._MODELS[_MODEL_ID] = entry  # noqa: SLF001
    models._USER_PROFILES.clear()  # noqa: SLF001

    async def no_refresh(*_args: object, **_kwargs: object) -> None:
        """Answer from the seeded catalogue instead of sweeping AWS."""

    monkeypatch.setattr(models, "refresh_stale_catalog", no_refresh)
    monkeypatch.setattr(models, "_refresh_due", lambda: False)
    monkeypatch.setattr(
        SETTINGS, "aws_bedrock_allow_application_inference_profile_arn", True
    )

    yield entry

    models._MODELS.clear()  # noqa: SLF001
    models._MODELS.update(saved_models)  # noqa: SLF001
    models._USER_PROFILES.clear()  # noqa: SLF001
    models._USER_PROFILES.update(saved_profiles)  # noqa: SLF001


class _CountingProfileLookup:
    """A profile lookup recording how many of its calls Bedrock serves at once.

    Attributes:
        calls: How many lookups reached it.
        peak: The most it ever had in flight together.
    """

    def __init__(self, region: RegionName, error: Exception | None = None) -> None:
        self.calls = 0
        self.peak = 0
        self._region = region
        self._error = error
        self._in_flight = 0

    async def __call__(self, _arn: str) -> tuple[list[dict[str, str]], RegionName]:
        """Report the seeded catalogue model, after one turn of the event loop.

        Returns:
            The profile's only member model, and the region it lives in.

        Raises:
            Exception: Whatever the test asked every lookup to fail with.
        """
        self.calls += 1
        self._in_flight += 1
        self.peak = max(self.peak, self._in_flight)
        try:
            await sleep(0)
            if self._error is not None:
                raise self._error
            return [{"modelArn": _MEMBER_ARN}], self._region
        finally:
            self._in_flight -= 1


@pytest.mark.local
@pytest.mark.usefixtures("seeded_catalog")
class TestArnResolutionsDoNotWaitOnEachOther:
    """One caller resolving an ARN never holds up another resolving a different one.

    The resolved profile is cached for every caller, so the cache read and write
    are serialised; the control-plane calls between them are not, because two
    ARNs share nothing but the cache. Two callers naming the *same* cold ARN
    still make a single call: they would otherwise hit the account's
    control-plane quota with a burst of identical lookups.

    Ref: stdapi/models/__init__.py:_validate_model_from_arn
         stdapi/models/__init__.py:_single_flight
    """

    async def test_two_arns_are_resolved_at_the_same_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two callers naming different ARNs overlap their control-plane calls.

        Ref: stdapi/models/__init__.py:_validate_model_from_arn
        """
        lookup = _CountingProfileLookup(_HOME_REGION)
        monkeypatch.setattr(models, "_get_application_inference_profile_models", lookup)

        await gather(
            _one_request(
                validate_model(_application_profile_arn(_HOME_REGION, "profile-one"))
            ),
            _one_request(
                validate_model(_application_profile_arn(_HOME_REGION, "profile-two"))
            ),
        )

        assert lookup.calls == 2
        assert lookup.peak == 2, "one ARN's lookup must not serialise another's"

    async def test_one_cold_arn_is_looked_up_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Concurrent callers naming one uncached ARN share a single lookup.

        Ref: stdapi/models/__init__.py:_single_flight
        """
        lookup = _CountingProfileLookup(_HOME_REGION)
        monkeypatch.setattr(models, "_get_application_inference_profile_models", lookup)
        arn = _application_profile_arn(_HOME_REGION)

        first, second = await gather(
            _one_request(validate_model(arn)), _one_request(validate_model(arn))
        )

        assert lookup.calls == 1
        assert first is second

    async def test_a_failed_lookup_reaches_every_caller_and_is_not_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refusal answers the whole wave, and the next caller tries again.

        Ref: stdapi/models/__init__.py:_single_flight
        """
        refusal = ApiError("the profile could not be read")
        failing = _CountingProfileLookup(_HOME_REGION, refusal)
        monkeypatch.setattr(
            models, "_get_application_inference_profile_models", failing
        )
        arn = _application_profile_arn(_HOME_REGION)

        outcomes = await gather(
            _one_request(validate_model(arn)),
            _one_request(validate_model(arn)),
            return_exceptions=True,
        )

        assert [type(outcome) for outcome in outcomes] == [ApiError, ApiError]
        assert failing.calls == 1
        assert models._PENDING_USER_PROFILES == {}  # noqa: SLF001
        assert models._USER_PROFILES == {}, "a failure must never be cached"  # noqa: SLF001

        working = _CountingProfileLookup(_HOME_REGION)
        monkeypatch.setattr(
            models, "_get_application_inference_profile_models", working
        )
        assert await _one_request(validate_model(arn))
        assert working.calls == 1


@pytest.mark.local
@pytest.mark.usefixtures("seeded_catalog")
class TestTheCallerReachesTheArnItNamed:
    """A request naming an ARN is served on that ARN.

    Ref: stdapi/models/__init__.py:_validate_model_from_arn
    """

    async def test_details_answered_for_an_arn_route_to_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The resolved details name the base model and route to the ARN.

        Ref: stdapi/models/__init__.py:ModelDetails.get_id
        """
        _stub_profile_lookup(monkeypatch, _HOME_REGION)
        arn = _application_profile_arn(_HOME_REGION)

        model = await _one_request(validate_model(arn))

        assert model.id == _MODEL_ID
        assert model.get_id(_HOME_REGION, inference_profile=True) == arn

    async def test_the_identifier_sent_upstream_is_the_arn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The invocation path routes that request to the ARN, not to the catalogue.

        The route hands the invocation a model ID rather than the details it
        resolved, so the ARN has to survive that second lookup: this asserts
        what the request is actually invoked on.

        Ref: stdapi/models/__init__.py:resolve_routed_model_id
        """
        _stub_profile_lookup(monkeypatch, _HOME_REGION)
        arn = _application_profile_arn(_HOME_REGION)

        async def serve_one_request() -> str:
            """Resolve the ARN and then route it, as one request does."""
            model = await validate_model(arn)
            return await resolve_routed_model_id(
                model.id, _HOME_REGION, inference_profile=True
            )

        assert await _one_request(serve_one_request()) == arn

    async def test_another_model_in_the_same_request_is_not_routed_to_the_arn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ARN applies to the model it names, not to everything that request resolves.

        A single request can resolve more than one model — a completion and the
        voice that speaks it — and the ARN names exactly one of them.

        Ref: stdapi/models/__init__.py:resolve_routed_model_id
        """
        _stub_profile_lookup(monkeypatch, _HOME_REGION)
        other = make_model_details(
            _OTHER_MODEL_ID,
            regions=[_HOME_REGION],
            inference_profiles={_HOME_REGION: _OTHER_SYSTEM_PROFILE},
        )
        models._MODELS[_OTHER_MODEL_ID] = other  # noqa: SLF001

        async def serve_one_request() -> str:
            """Resolve the ARN, then the second model the same request needs."""
            await validate_model(_application_profile_arn(_HOME_REGION))
            return await resolve_routed_model_id(
                _OTHER_MODEL_ID, _HOME_REGION, inference_profile=True
            )

        assert await _one_request(serve_one_request()) == _OTHER_SYSTEM_PROFILE

    async def test_a_region_the_catalogue_does_not_list_is_added_for_that_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A profile elsewhere makes its own region serviceable for that request.

        Ref: stdapi/models/__init__.py:_collect_region_candidates
        """
        _stub_profile_lookup(monkeypatch, _OTHER_REGION)
        arn = _application_profile_arn(_OTHER_REGION)

        model = await _one_request(validate_model(arn))

        assert _OTHER_REGION in model.regions
        assert model.get_id(_OTHER_REGION, inference_profile=True) == arn


@pytest.mark.local
class TestArnResolutionLeavesTheCatalogAlone:
    """One caller's ARN must never become another caller's routing.

    Ref: stdapi/models/__init__.py:_validate_model_from_arn
    """

    async def test_catalog_profiles_are_unchanged_by_an_arn_lookup(
        self, seeded_catalog: ModelDetails, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The catalogue entry keeps the profiles it had before the ARN was named.

        Ref: stdapi/models/__init__.py:ModelDetails.set_inference_profile
        """
        _stub_profile_lookup(monkeypatch, _HOME_REGION)

        await _one_request(validate_model(_application_profile_arn(_HOME_REGION)))

        assert seeded_catalog.inference_profiles == {_HOME_REGION: _SYSTEM_PROFILE}
        assert seeded_catalog.inference_profiles_regional == {
            _HOME_REGION: _REGIONAL_PROFILE
        }

    async def test_catalog_regions_are_unchanged_by_an_arn_lookup(
        self, seeded_catalog: ModelDetails, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller's profile does not widen where the catalogue serves the model.

        Ref: stdapi/models/__init__.py:_collect_region_candidates
        """
        _stub_profile_lookup(monkeypatch, _OTHER_REGION)

        await _one_request(validate_model(_application_profile_arn(_OTHER_REGION)))

        assert seeded_catalog.regions == [_HOME_REGION]

    async def test_the_next_request_routes_to_the_catalogue_profile(
        self, seeded_catalog: ModelDetails, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A later request naming the model itself is not sent to the first one's ARN.

        This is the failure the isolation exists for: the identifier the
        invocation path resolves is read back by model ID, so an ARN written
        into the catalogue would bill every later request in that region to a
        profile its caller never named.

        Ref: stdapi/models/__init__.py:resolve_routed_model_id
        """
        _stub_profile_lookup(monkeypatch, _HOME_REGION)
        await _one_request(validate_model(_application_profile_arn(_HOME_REGION)))

        async def serve_next_request() -> str:
            """Route the plain model ID, as the next caller's request does."""
            return await resolve_routed_model_id(
                _MODEL_ID, _HOME_REGION, inference_profile=True
            )

        assert await _one_request(serve_next_request()) == _SYSTEM_PROFILE
        assert (await get_model_details(_MODEL_ID)) is seeded_catalog

    async def test_resolved_details_share_no_container_with_the_catalog(
        self, seeded_catalog: ModelDetails, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every mutable field of the answer is its own object.

        Asserted structurally rather than field by field, so a mutable field
        added to the model later cannot reintroduce the sharing unnoticed.

        Ref: stdapi/models/__init__.py:ModelDetails
        """
        _stub_profile_lookup(monkeypatch, _HOME_REGION)

        model = await _one_request(
            validate_model(_application_profile_arn(_HOME_REGION))
        )

        shared_fields = [
            name
            for name in ModelDetails.model_fields
            if isinstance(getattr(seeded_catalog, name), list | dict | set)
            and getattr(model, name) is getattr(seeded_catalog, name)
        ]
        assert not shared_fields
